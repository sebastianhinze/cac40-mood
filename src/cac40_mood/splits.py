"""The train / validation / test split (DECISIONS #14).

Why this is its own tiny module: the split is the single most important guard against fooling
ourselves. Every later phase labels its rows with `label_split` and filters on that label, so
there is exactly one place where the boundaries are written down.

The test period (2024-07 -> 2025-12) is evaluated once, in phase 6. Before that, analysis code
must call `drop_test_period` (or filter on the label) so test rows never reach a model or a plot.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

TRAIN_START: date = date(2019, 1, 1)
TRAIN_END: date = date(2022, 12, 31)
VAL_END: date = date(2024, 6, 30)
TEST_END: date = date(2025, 12, 31)

# Robustness run (DECISIONS #28): repeat the analysis on a sample that starts after the spring-2020
# hole in the filings feed, to show the results do not depend on how the hole is handled.
ROBUSTNESS_SAMPLE_START: date = date(2020, 7, 1)


def label_split(days: pd.Series) -> pd.Series:
    """Label each date as 'pre', 'train', 'val', 'test' or 'post'.

    'pre' is the 2018 price warm-up year; 'post' is anything after the sample.
    Missing dates stay missing so a bad join is visible instead of silently labelled.
    """
    day = pd.to_datetime(days).dt.normalize()
    labels = pd.Series("post", index=days.index, dtype="object")
    labels[day <= pd.Timestamp(TEST_END)] = "test"
    labels[day <= pd.Timestamp(VAL_END)] = "val"
    labels[day <= pd.Timestamp(TRAIN_END)] = "train"
    labels[day < pd.Timestamp(TRAIN_START)] = "pre"
    labels[day.isna()] = None
    return labels


def drop_test_period(frame: pd.DataFrame, day_column: str) -> pd.DataFrame:
    """Return only the rows dated on or before the end of validation.

    Rows without a date are dropped too: an undated row cannot be shown to be outside the test set.
    """
    day = pd.to_datetime(frame[day_column]).dt.normalize()
    return frame[day <= pd.Timestamp(VAL_END)]
