"""Compare two sets of scores for the same documents (two models, two settings, two runs).

Used three ways in the pilot, always with the same numbers so they can be read side by side:
- run 1 vs run 2 of the same setting  -> how noisy the scores are (the yardstick);
- short cap vs long cap                -> does reading more text change the scores;
- cheap model vs reference model       -> does the cheap model agree with the stronger one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FIELDS: tuple[str, ...] = ("tone", "surprise", "uncertainty", "confidence")

# Tone or surprise within +/- 0.15 of zero counts as "neutral" when comparing directions.
NEUTRAL_BAND: float = 0.15


def paired(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Join two score tables on doc_id, keeping only documents both sides scored validly."""
    columns = ["doc_id", *NUMERIC_FIELDS, "topic"]
    left_valid = left.loc[left["problem"] == "", columns]
    right_valid = right.loc[right["problem"] == "", columns]
    return left_valid.merge(right_valid, on="doc_id", suffixes=("_a", "_b"))


def direction(values: pd.Series) -> pd.Series:
    """-1, 0 or 1: negative, neutral (inside the band) or positive."""
    return pd.Series(np.sign(values.where(values.abs() > NEUTRAL_BAND, 0.0)), index=values.index)


def cohens_kappa(first: pd.Series, second: pd.Series) -> float:
    """Agreement on categories corrected for chance: 0 = chance level, 1 = perfect.

    kappa = (observed agreement - agreement expected by chance) / (1 - expected by chance),
    where "expected by chance" uses how often each side picks each category.
    """
    observed = float((first == second).mean())
    categories = sorted(set(first) | set(second))
    share_first = first.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    share_second = second.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    expected = float((share_first * share_second).sum())
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def agreement_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per numeric field: correlation, rank correlation, mean absolute difference, mean shift."""
    rows = []
    for field in NUMERIC_FIELDS:
        a, b = pairs[f"{field}_a"], pairs[f"{field}_b"]
        constant = a.nunique() < 2 or b.nunique() < 2
        rows.append(
            {
                "field": field,
                "n": len(pairs),
                "pearson": np.nan if constant else round(float(a.corr(b)), 3),
                "spearman": np.nan if constant else round(float(a.corr(b, method="spearman")), 3),
                "mean_abs_diff": round(float((a - b).abs().mean()), 3),
                "mean_b_minus_a": round(float((b - a).mean()), 3),
            }
        )
    return pd.DataFrame(rows).set_index("field")


def categorical_agreement(pairs: pd.DataFrame) -> dict[str, float]:
    """Share of documents with the same tone direction, surprise direction and topic."""
    return {
        "n": len(pairs),
        "same_tone_direction": round(
            float((direction(pairs["tone_a"]) == direction(pairs["tone_b"])).mean()), 3
        ),
        "opposite_tone_sign": round(
            float((direction(pairs["tone_a"]) * direction(pairs["tone_b"]) < 0).mean()), 3
        ),
        "same_surprise_direction": round(
            float((direction(pairs["surprise_a"]) == direction(pairs["surprise_b"])).mean()), 3
        ),
        "same_topic": round(float((pairs["topic_a"] == pairs["topic_b"]).mean()), 3),
        "topic_kappa": round(cohens_kappa(pairs["topic_a"], pairs["topic_b"]), 3),
    }
