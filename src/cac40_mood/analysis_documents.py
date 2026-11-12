"""The one set of scored documents that every analysis (volatility, mood, event study) uses.

Having a single selection, written once, is what keeps the analyses comparable and the rules
enforced in one place. A document is used when all of this holds:

- it is in the **train or validation** period (the test period is dropped first, before
  anything else happens);
- its firm is one of the 36 firms of the main analysis (DECISIONS #29);
- it is not an administrative notice and it has an event session;
- it lies outside the spring-2020 hole in the feed (DECISIONS #28);
- the LLM returned a valid score for it;
- the firm-size figure given to the LLM was read from a filing published **before** the
  document itself (DECISIONS #53). Otherwise the score had an input from the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cac40_mood.config import PROCESSED_DIR
from cac40_mood.data.build import FILINGS_PATH
from cac40_mood.data.universe import EXCLUDED_FROM_MAIN_ANALYSIS
from cac40_mood.scoring.revenue import TABLE_PATH

FILING_COLUMNS: list[str] = [
    "doc_id",
    "firm_id",
    "ts_paris",
    "event_session",
    "intraday",
    "split",
    "doc_class",
    "is_admin",
    "in_feed_gap",
]
SCORE_COLUMNS: list[str] = [
    "doc_id",
    "problem",
    "tone",
    "surprise",
    "uncertainty",
    "confidence",
    "topic",
    "size_fy",
]


LOUGHRAN_COLUMNS: list[str] = ["lmd_tone", "lmd_uncertainty"]
LOUGHRAN_PATH = PROCESSED_DIR / "baseline_loughran.parquet"


def flag_size_source_later(documents: pd.DataFrame, revenue: pd.DataFrame) -> pd.DataFrame:
    """Add `revenue` (the figure the LLM was given) and `size_source_later`.

    `size_source_later` is True when the filing that the revenue figure was read from was
    published after the document. That covers two cases: a fiscal year whose results were not
    out yet, and an older year's figure read from a later release (where it may have been
    restated). Both are inputs the model could not have had on the day.
    """
    source = revenue[["firm_id", "fy", "revenue", "published"]].rename(columns={"fy": "size_fy"})
    merged = documents.merge(source, on=["firm_id", "size_fy"], how="left")
    merged["size_source_later"] = merged["published"] > merged["ts_paris"]
    return merged.drop(columns="published")


def select_documents(
    filings: pd.DataFrame,
    scores: pd.DataFrame,
    lexicon: pd.DataFrame,
    local_model: pd.DataFrame,
    revenue: pd.DataFrame,
    loughran: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (documents used in the analysis, a small table of how many each rule removed)."""
    in_sample = filings.loc[filings["split"].isin(["train", "val"]), FILING_COLUMNS]
    table = in_sample.merge(scores[SCORE_COLUMNS], on="doc_id", how="left")
    table = table.merge(lexicon[["doc_id", "lex_tone", "lex_uncertainty"]], on="doc_id", how="left")
    table = table.merge(local_model[["doc_id", "lm_tone"]], on="doc_id", how="left")
    # Loughran-McDonald scores exist only for documents with an official English version.
    if loughran is None:
        loughran = pd.DataFrame(columns=["doc_id", *LOUGHRAN_COLUMNS])
    table = table.merge(loughran[["doc_id", *LOUGHRAN_COLUMNS]], on="doc_id", how="left")
    table[LOUGHRAN_COLUMNS] = table[LOUGHRAN_COLUMNS].astype(float)
    table = flag_size_source_later(table, revenue)

    rules = [
        ("administrative notice", table["is_admin"]),
        ("firm outside the main analysis", table["firm_id"].isin(EXCLUDED_FROM_MAIN_ANALYSIS)),
        ("no event session", table["event_session"].isna()),
        ("inside the spring-2020 feed hole", table["in_feed_gap"]),
        ("no valid LLM score (no usable text)", table["problem"].ne("") | table["tone"].isna()),
        ("firm-size figure from a later filing", table["size_source_later"]),
    ]
    keep = np.ones(len(table), dtype=bool)
    removed = []
    for name, rule in rules:
        drops = keep & rule.to_numpy()
        removed.append({"rule": name, "documents_removed": int(drops.sum())})
        keep &= ~rule.to_numpy()
    removed.append({"rule": "documents used", "documents_removed": int(keep.sum())})
    return table[keep].reset_index(drop=True), pd.DataFrame(removed)


def load_documents() -> tuple[pd.DataFrame, pd.DataFrame]:
    """`select_documents` on the stored tables."""
    return select_documents(
        pd.read_parquet(FILINGS_PATH),
        pd.read_parquet(PROCESSED_DIR / "scores.parquet"),
        pd.read_parquet(PROCESSED_DIR / "baseline_lexicon.parquet"),
        pd.read_parquet(PROCESSED_DIR / "baseline_local_model.parquet"),
        pd.read_parquet(TABLE_PATH),
        pd.read_parquet(LOUGHRAN_PATH) if LOUGHRAN_PATH.exists() else None,
    )
