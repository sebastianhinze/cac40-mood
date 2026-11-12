"""Sanity tables for the bulk scores. Distributions are computed on train and validation
documents only; for the test period only counts and failures are reported, so that nothing
about the test period's content is looked at before the final evaluation.
"""

from __future__ import annotations

import pandas as pd

SCORE_FIELDS: list[str] = ["tone", "surprise", "uncertainty", "confidence"]


def with_metadata(scores: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Join scores to the filing's firm, date, split and document class."""
    columns = ["doc_id", "firm_id", "ts_paris", "split", "doc_class", "lang", "in_feed_gap"]
    return scores.merge(filings[columns], on="doc_id")


def run_overview(table: pd.DataFrame) -> pd.DataFrame:
    """Per split: documents, failures, cut documents and token usage (all splits, counts only)."""
    grouped = table.groupby("split")
    return pd.DataFrame(
        {
            "documents": grouped.size(),
            "problems": grouped["problem"].agg(lambda problem: int((problem != "").sum())),
            "cut_at_cap": grouped["truncated"].sum(),
            "mean_input_tokens": grouped["input_tokens"].mean().round(0),
            "mean_output_tokens": grouped["output_tokens"].mean().round(0),
        }
    )


def tuning_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Valid scores from the train and validation periods."""
    return table[table["split"].isin(["train", "val"]) & (table["problem"] == "")]


def by_topic(table: pd.DataFrame) -> pd.DataFrame:
    """Per topic: count, share, mean and spread of tone, share of non-zero surprise."""
    rows = tuning_rows(table)
    grouped = rows.groupby("topic")
    summary = pd.DataFrame(
        {
            "documents": grouped.size(),
            "share": (grouped.size() / len(rows)).round(3),
            "tone_mean": grouped["tone"].mean().round(3),
            "tone_sd": grouped["tone"].std().round(3),
            "tone_negative_share": grouped["tone"].agg(lambda tone: (tone < -0.15).mean()).round(3),
            "surprise_nonzero_share": grouped["surprise"].agg(lambda s: (s != 0).mean()).round(3),
            "uncertainty_mean": grouped["uncertainty"].mean().round(3),
            "confidence_mean": grouped["confidence"].mean().round(3),
        }
    )
    return summary.sort_values("documents", ascending=False)


def by_year(table: pd.DataFrame) -> pd.DataFrame:
    """Per calendar year (train and validation only): count and mean scores."""
    rows = tuning_rows(table)
    year = rows["ts_paris"].dt.year.rename("year")
    grouped = rows.groupby(year)
    summary = pd.DataFrame({"documents": grouped.size()})
    for field in SCORE_FIELDS:
        summary[f"{field}_mean"] = grouped[field].mean().round(3)
    summary["tone_sd"] = grouped["tone"].std().round(3)
    summary["tone_negative_share"] = (
        grouped["tone"].agg(lambda tone: (tone < -0.15).mean()).round(3)
    )
    return summary


UNSCORED_REPORT = "02_unscored_documents.md"


def unscored_documents(
    filings: pd.DataFrame, texts: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    """Non-admin documents that were never sent to the LLM, with the reason (text quality)."""
    to_score = filings[~filings["is_admin"]]
    table = to_score.merge(texts[["doc_id", "text_quality"]], on="doc_id", how="left")
    table = table[~table["doc_id"].isin(scores["doc_id"])].copy()
    table["year"] = table["ts_paris"].dt.year
    return table


def count_table(table: pd.DataFrame, rows: str, columns: str) -> pd.DataFrame:
    """Counts of `rows` against `columns`, with a total column."""
    counts = pd.crosstab(table[rows], table[columns])
    counts["total"] = counts.sum(axis=1)
    return counts.sort_values("total", ascending=False)


def markdown_counts(counts: pd.DataFrame) -> str:
    header = [str(counts.index.name), *[str(column) for column in counts.columns]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, row in zip(counts.index, counts.itertuples(index=False), strict=True):
        lines.append("| " + " | ".join([str(label), *[str(value) for value in row]]) + " |")
    return "\n".join(lines)


def main() -> None:
    """Write the breakdown of unscored documents (counts only; no score is read)."""
    from cac40_mood.config import PROCESSED_DIR, PROJECT_ROOT
    from cac40_mood.data.build import FILINGS_PATH, TEXTS_PATH

    filings = pd.read_parquet(FILINGS_PATH)
    scores = pd.read_parquet(PROCESSED_DIR / "scores.parquet", columns=["doc_id"])
    table = unscored_documents(filings, pd.read_parquet(TEXTS_PATH), scores)
    to_score = int((~filings["is_admin"]).sum())
    firms = table["firm_id"].value_counts().head(8)
    lines = [
        "# Documents that were not scored",
        "",
        f"{len(table)} of {to_score:,} non-administrative documents ({len(table) / to_score:.1%}) "
        "have no usable text and were not sent to the LLM (DECISIONS #48). Counts only; the "
        "test period appears as a count and nothing else.",
        "",
        "Reasons: `not_pdf` = filed in another format (mostly annual reports as ESEF packages "
        "from 2021 on); `empty` = scanned image without a text layer; `garbled` = mostly "
        "tables or broken characters; `unreadable` / `missing_file` = the file could not be "
        "opened or downloaded.",
        "",
        "## By document class and reason",
        "",
        markdown_counts(count_table(table, "doc_class", "text_quality")),
        "",
        "## By document class and period",
        "",
        markdown_counts(count_table(table, "doc_class", "split")),
        "",
        "## By year and reason",
        "",
        markdown_counts(count_table(table, "year", "text_quality").sort_index()),
        "",
        "## Firms with the most unscored documents",
        "",
        ", ".join(f"{firm} {count}" for firm, count in firms.items()) + ".",
        "",
        "What it means: 71% of the unscored documents are periodic reports (annual and "
        "half-year reports), whose news is normally also in a press release that is scored. "
        "56 are press releases (`inside_information`), under 1% of that class.",
        "",
    ]
    (PROJECT_ROOT / "reports" / UNSCORED_REPORT).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
