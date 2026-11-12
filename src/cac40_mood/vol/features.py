"""Announcement and sentiment columns for the volatility models, one row per firm and day.

The forecast made at the close of day t may use documents whose event session is t or earlier
(the event-session rule already moved after-close documents to the next session). Every column
here describes the documents of day t itself, so nothing later than t can enter.

Three groups of columns, matching the models of the spec:
- B1, "an announcement of this type happened": one 0/1 column per announcement type, taken
  from the regulator's type code (not from the LLM);
- M, "what the LLM read in it": |tone| and uncertainty of the day's documents;
- the same from the two baseline scorers, to see whether the LLM is worth paying for.

Days without a document get 0 in every column. Days on which we cannot know whether there was
a document (before the sample starts, and the spring-2020 hole in the feed) get missing values,
so they drop out of estimation and evaluation instead of counting as "no news".
"""

from __future__ import annotations

import pandas as pd

from cac40_mood.data.filings import feed_is_observed
from cac40_mood.splits import TRAIN_START

# Regulator's document class -> announcement-type column. Small classes share one column.
CLASS_TO_EVENT: dict[str, str] = {
    "inside_information": "ev_news",
    "periodic_report": "ev_report",
    "quarterly_information": "ev_quarterly",
}
EVENT_COLUMNS: list[str] = ["ev_news", "ev_report", "ev_quarterly", "ev_other"]

# Primary score features (DECISIONS #40): tone and uncertainty. For volatility the size of
# the news matters, not its sign, hence the absolute value of tone.
SCORE_COLUMNS: list[str] = ["abs_tone", "uncertainty"]
# Exploratory extras: bad news separately (volatility reacts more to bad news), and surprise.
EXTRA_COLUMNS: list[str] = ["neg_tone", "abs_surprise"]
LEXICON_COLUMNS: list[str] = ["lex_abs_tone", "lex_uncertainty"]
LOCAL_MODEL_COLUMNS: list[str] = ["lm_abs_tone"]
ALL_COLUMNS: list[str] = [
    *EVENT_COLUMNS,
    *SCORE_COLUMNS,
    *EXTRA_COLUMNS,
    *LEXICON_COLUMNS,
    *LOCAL_MODEL_COLUMNS,
]

MIN_WEIGHT: float = 0.05  # a document with confidence 0 still counts a little


def daily_document_features(documents: pd.DataFrame) -> pd.DataFrame:
    """One row per firm and event session that has at least one document.

    LLM scores of several documents on the same day are averaged with the model's confidence
    as weight; baseline scores, which have no confidence, with equal weights.
    """
    docs = documents.copy()
    docs["weight"] = docs["confidence"].clip(lower=MIN_WEIGHT)
    for field in ("tone", "surprise", "uncertainty"):
        docs[f"weighted_{field}"] = docs["weight"] * docs[field]
    docs["event_type"] = docs["doc_class"].map(CLASS_TO_EVENT).fillna("ev_other")
    for column in EVENT_COLUMNS:
        docs[column] = (docs["event_type"] == column).astype(float)

    by_day = docs.groupby(["firm_id", "event_session"])
    sums = by_day[["weight", "weighted_tone", "weighted_surprise", "weighted_uncertainty"]].sum()
    day = by_day[EVENT_COLUMNS].max()  # 1 if any document of that type arrived
    tone = sums["weighted_tone"] / sums["weight"]
    day["abs_tone"] = tone.abs()
    day["neg_tone"] = (-tone).clip(lower=0.0)
    day["abs_surprise"] = (sums["weighted_surprise"] / sums["weight"]).abs()
    day["uncertainty"] = sums["weighted_uncertainty"] / sums["weight"]
    day["lex_abs_tone"] = by_day["lex_tone"].mean().abs()
    day["lex_uncertainty"] = by_day["lex_uncertainty"].mean()
    day["lm_abs_tone"] = by_day["lm_tone"].mean().abs()
    return day.reset_index().rename(columns={"event_session": "date"})


def attach_features(frame: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Add the announcement and sentiment columns to a firm-day table (columns firm_id, date)."""
    day = daily_document_features(documents)
    merged = frame.merge(day, on=["firm_id", "date"], how="left")
    merged[ALL_COLUMNS] = merged[ALL_COLUMNS].fillna(0.0)
    known = feed_is_observed(merged["date"]) & (merged["date"] >= pd.Timestamp(TRAIN_START))
    merged.loc[~known, ALL_COLUMNS] = float("nan")
    return merged
