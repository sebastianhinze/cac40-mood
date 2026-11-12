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
