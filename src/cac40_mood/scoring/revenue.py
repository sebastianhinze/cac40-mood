"""Firm size: annual revenue per fiscal year, read from the firms' own results releases.

Why: the scoring prompt tells the model roughly how large the issuer is, so that it can judge
whether a piece of news is material (DECISIONS #36). The figures must come from a documented
source and must not leak later information (DECISIONS #41).

Source: the annual-results announcements already in our filings table. A small extraction
prompt asks the model, for each candidate document, whether it announces full-year results and,
if so, the group's revenue for that year and for the year before, with a verbatim quote. Every
figure in the resulting table therefore points to one filing and one sentence.

No look-ahead: revenue of the fiscal year that ended in calendar year F is used for documents
dated from 1 July of year F+1 to 30 June of year F+2. By 1 July every firm has published its
annual results for the previous fiscal year. Only train and validation documents are read; for
documents dated after June 2025 the latest available year (ending 2023) is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, ValidationError

from cac40_mood.config import PROCESSED_DIR
from cac40_mood.scoring import llm
from cac40_mood.scoring.batch import BatchItem, run_batch
from cac40_mood.scoring.prompt import describe_size, truncate_text
from cac40_mood.scoring.spend import SpendLog

REVENUE_PROMPT_VERSION: str = "revenue-v2"

FACTS_PATH: Path = PROCESSED_DIR / "firm_revenue_facts.parquet"
TABLE_PATH: Path = PROCESSED_DIR / "firm_revenue.parquet"

# The headline revenue is usually on the first page; some groups show it in a table a few
# pages in, so a little more than the first page is read.
MAX_CHARS: int = 15_000
MAX_OUTPUT_TOKENS: int = 600

# Two extractions of the same fiscal year (from its own release and as the "previous year" in
# the next release) may differ after restatements; beyond this ratio they are flagged.
RESTATEMENT_TOLERANCE: float = 0.15

SYSTEM_PROMPT: str = """\
You extract one fact from an announcement filed by a listed company: that company's total \
consolidated revenue for a full fiscal year. The listed company is named before the text.

First decide what the announcement is:
- full_year: it announces the listed company's own results or revenue for a complete fiscal \
year (twelve months).
- half_year: first-half results or revenue.
- quarter: a single quarter or nine months, without the full-year revenue.
- other: anything else. This includes announcements about a subsidiary, an affiliate or \
another company (for example a subsidiary's own annual results), even if they were filed \
under the listed company's name.

A fourth-quarter announcement that also states the full-year revenue counts as full_year.

Only for full_year, fill in the revenue fields. Use the headline top-line figure of the \
listed company's consolidated group as it reports it: revenue or sales (chiffre d'affaires) \
for industrial and service companies, net banking income (produit net bancaire, revenus) for \
banks, total revenues for insurers, gross rental income for property companies. If the text \
gives figures both for the listed company and for a wider group it belongs to (for example a \
mutual banking group), use the listed company's. Do not use the figure of a segment, a \
division or a subsidiary. Give amounts in millions of the reporting currency (12.3 billion \
is 12300). Use the reported figure, not an organic or like-for-like variant. If a figure is \
not stated in the text, leave it null; never compute or guess one.

Fields:
- period: full_year, half_year, quarter or other.
- fiscal_year_end: year and month in which the fiscal year ended, as YYYY-MM (for example \
2023-12, or 2023-06 for a year ending in June). Empty string if not full_year.
- revenue_millions: revenue of that fiscal year, in millions, or null.
- prior_year_revenue_millions: revenue of the previous fiscal year as stated in this \
announcement, in millions, or null.
- currency: EUR, USD or other.
- quote: the exact words from the text that state the revenue (at most 25 words), or an empty \
string.
"""

REVENUE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "period": {"type": "string", "enum": ["full_year", "half_year", "quarter", "other"]},
        "fiscal_year_end": {"type": "string"},
        "revenue_millions": {"type": ["number", "null"]},
        "prior_year_revenue_millions": {"type": ["number", "null"]},
        "currency": {"type": "string", "enum": ["EUR", "USD", "other"]},
        "quote": {"type": "string"},
    },
    "required": [
        "period",
        "fiscal_year_end",
        "revenue_millions",
        "prior_year_revenue_millions",
        "currency",
        "quote",
    ],
    "additionalProperties": False,
}


class RevenueFact(BaseModel):
    """The validated answer for one document."""

    period: Literal["full_year", "half_year", "quarter", "other"]
    fiscal_year_end: str
    revenue_millions: float | None
    prior_year_revenue_millions: float | None
    currency: Literal["EUR", "USD", "other"]
    quote: str


def candidate_documents(documents: pd.DataFrame) -> pd.DataFrame:
    """Documents that may announce annual results: everything published January-April, plus
    every filing the feed codes as a results release (001007, 001008) in any month.

    The second part catches firms whose fiscal year does not end in December.
    """
    month = documents["ts_paris"].dt.month
    results_coded = documents["type_code"].isin(["001007", "001008"])
    return documents[month.isin([1, 2, 3, 4]) | results_coded].reset_index(drop=True)


def build_request(model: str, issuer: str, title: str, text: str) -> dict[str, Any]:
    body, _ = truncate_text(text, MAX_CHARS)
    message = f"Listed company: {issuer}\nTitle: {title.strip()}\n\n<document>\n{body}\n</document>"
    return {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": message}],
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": REVENUE_JSON_SCHEMA},
        },
    }


def extract_facts(
    client: anthropic.Anthropic,
    model: str,
    candidates: pd.DataFrame,
    cache_dir: Path,
    spend_log: SpendLog,
    state_path: Path,
) -> pd.DataFrame:
    """Ask the model about every candidate (through a batch) and return the valid answers."""
    items = []
    for row in candidates.itertuples():
        request = build_request(model, row.issuer, row.title, row.text)
        items.append(BatchItem(row.doc_id, request, llm.cache_key(request, REVENUE_PROMPT_VERSION)))
    answers = run_batch(client, items, REVENUE_PROMPT_VERSION, cache_dir, spend_log, state_path)
    rows = []
    for row in candidates.itertuples():
        answer = answers[row.doc_id]
        if answer is None or answer["stop_reason"] != "end_turn":
            continue
        try:
            fact = RevenueFact.model_validate_json(answer["text"])
        except ValidationError:
            continue
        rows.append(
            {
                "doc_id": row.doc_id,
                "firm_id": row.firm_id,
                "published": row.ts_paris,
                **fact.model_dump(),
            }
        )
    return pd.DataFrame(rows)


def fiscal_year_table(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per firm and fiscal year: revenue, currency, source document and quote.

    For each fiscal year the figure from that year's own results release is preferred
    ("reported"). If no such release was found, the "previous year" figure from the following
    year's release is used ("comparative"). When both exist and differ by more than the
    tolerance, the row is flagged for a manual look.
    """
    annual = facts[
        (facts["period"] == "full_year") & facts["fiscal_year_end"].str.match(r"^\d{4}-\d{2}$")
    ]
    annual = annual.assign(fy=annual["fiscal_year_end"].str.slice(0, 4).astype(int))
    reported = annual[annual["revenue_millions"].notna()].assign(
        revenue=lambda frame: frame["revenue_millions"], basis="reported"
    )
    comparative = annual[annual["prior_year_revenue_millions"].notna()].assign(
        revenue=lambda frame: frame["prior_year_revenue_millions"],
        fy=lambda frame: frame["fy"] - 1,
        basis="comparative",
    )
    columns = ["firm_id", "fy", "revenue", "currency", "basis", "doc_id", "published", "quote"]
    both = pd.concat([reported[columns], comparative[columns]], ignore_index=True)
    # Earliest publication first, "reported" before "comparative": the first row per firm-year
    # is the figure as first announced.
    both["basis_rank"] = (both["basis"] == "comparative").astype(int)
    both = both.sort_values(["firm_id", "fy", "basis_rank", "published"])
    table = both.groupby(["firm_id", "fy"], as_index=False).first()
    spread = both.groupby(["firm_id", "fy"])["revenue"].agg(
        lambda values: values.max() / values.min()
    )
    table["max_over_min"] = spread.to_numpy()
    table["needs_check"] = table["max_over_min"] > 1 + RESTATEMENT_TOLERANCE
    return table.drop(columns="basis_rank")


def fiscal_year_for(document_dates: pd.Series) -> pd.Series:
    """The fiscal year whose revenue may be used for a document of this date (1 July rule)."""
    dates = pd.to_datetime(document_dates)
    return dates.dt.year - 1 - (dates.dt.month < 7).astype(int)


def attach_size(documents: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add `size` (text for the prompt) and `size_fy` to each document.

    A document gets the revenue of the fiscal year allowed by the 1 July rule. If the table has
    no figure for that year, the latest earlier year is used; if there is none either (start of
    the sample), the earliest later year is used and `size_backfilled` is set. Such a figure was
    public at the time, we simply have no filing from before 2019 to read it from.
    """
    documents = documents.copy()
    documents["size_fy_wanted"] = fiscal_year_for(documents["ts_paris"].dt.tz_localize(None))
    by_firm = {firm: group.sort_values("fy") for firm, group in table.groupby("firm_id")}
    sizes, years, backfilled = [], [], []
    for firm_id, wanted in zip(documents["firm_id"], documents["size_fy_wanted"], strict=True):
        rows = by_firm[firm_id]
        earlier = rows[rows["fy"] <= wanted]
        row = earlier.iloc[-1] if len(earlier) else rows.iloc[0]
        sizes.append(describe_size(row["revenue"] / 1000.0, row["currency"]))
        years.append(int(row["fy"]))
        backfilled.append(len(earlier) == 0)
    documents["size"] = sizes
    documents["size_fy"] = years
    documents["size_backfilled"] = backfilled
    return documents.drop(columns="size_fy_wanted")


def with_size(documents: pd.DataFrame) -> pd.DataFrame:
    """`attach_size` using the stored fiscal-year table."""
    return attach_size(documents, pd.read_parquet(TABLE_PATH))
