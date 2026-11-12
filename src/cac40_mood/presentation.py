"""Read the frozen results back out of the committed reports, for the README and the dashboard.

Why parse Markdown: the reports in `reports/` are the record of what was run, and they are the
only results that are committed (everything under `data/` stays local). Reading the numbers
from them means the README figures and the dashboard's "validation vs test" panel show exactly
what the reports say, and work on a fresh clone without any data or API key.

Nothing here estimates anything. It only finds tables and picks rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cac40_mood.config import PROJECT_ROOT

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
EVENT_REPORT: Path = REPORTS_DIR / "03_event_study.md"
VOL_REPORT: Path = REPORTS_DIR / "04_vol_sentiment_validation.md"
NONLINEAR_REPORT: Path = REPORTS_DIR / "05_nonlinear_mood_validation.md"
TEST_REPORT: Path = REPORTS_DIR / "06_test_results.md"

TRAIN_VAL: str = "train + validation"
TEST: str = "test (frozen, run once)"
VALIDATION: str = "validation"


def _cells(line: str) -> list[str]:
    """Split one Markdown table row into stripped cell texts."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    """True for the |---|---| line under a table header."""
    return all(set(cell) <= set("-: ") and cell for cell in _cells(line))


def _to_number(value: str) -> float | str:
    """Turn '0.0117' or '1e-05' into a float; leave labels as they are."""
    try:
        return float(value)
    except ValueError:
        return value


def read_markdown_tables(path: Path) -> list[pd.DataFrame]:
    """Every pipe table in a Markdown file, in order, with numeric cells as floats.

    A table is a header row, a separator row and the rows that follow until a line that does
    not start with '|'.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    tables: list[pd.DataFrame] = []
    index = 0
    while index < len(lines) - 1:
        header, separator = lines[index], lines[index + 1]
        if header.startswith("|") and separator.startswith("|") and _is_separator(separator):
            columns = _cells(header)
            rows = []
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([_to_number(cell) for cell in _cells(lines[index])])
                index += 1
            tables.append(pd.DataFrame(rows, columns=columns))
        else:
            index += 1
    return tables


def find_table(path: Path, columns: list[str], occurrence: int = 0) -> pd.DataFrame:
    """The table whose header starts with exactly these columns (the n-th such table).

    Matching on the header prefix, not on position in the file, keeps this working if a section
    is added to a report.
    """
    matches = [
        table
        for table in read_markdown_tables(path)
        if list(table.columns)[: len(columns)] == columns
    ]
    if len(matches) <= occurrence:
        raise ValueError(f"No table {occurrence} with columns {columns} in {path.name}")
    return matches[occurrence]


def day0_terciles() -> pd.DataFrame:
    """Mean day-0 abnormal return by tone third, train+validation and test."""
    columns = ["outcome", "tone_tercile", "events", "mean", "std_error", "t"]
    frames = []
    for path, period in ((EVENT_REPORT, TRAIN_VAL), (TEST_REPORT, TEST)):
        table = find_table(path, columns)
        table = table[table["outcome"] == "CAR[0] (RQ1)"].copy()
        table["period"] = period
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def scorer_slopes() -> pd.DataFrame:
    """Day-0 abnormal return per standard deviation of tone, for each scorer and period."""
    columns = ["scorer", "coefficient", "ci_low", "ci_high", "t", "p_value", "r_squared", "n"]
    frames = []
    for path, period in ((EVENT_REPORT, TRAIN_VAL), (TEST_REPORT, TEST)):
        table = find_table(path, columns)
        table["period"] = period
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class Comparison:
    """One volatility comparison at h = 5: what was added, and the two periods' results."""

    label: str
    model: str
    against: str
    status: str  # "pre-registered" or "exploratory"


VOL_COMPARISONS: tuple[Comparison, ...] = (
    Comparison("RQ3: scores on top of announcement dummies", "M", "B1", "pre-registered"),
    Comparison("RQ4: nonlinear mood model on top of scores", "M3", "M", "pre-registered"),
    Comparison("Per-firm mood on top of scores", "M+firm mood", "M", "exploratory"),
)


def _validation_row(comparison: Comparison) -> dict[str, float]:
    """Validation-period numbers at h = 5 from the phase 4 and phase 5 reports."""
    if comparison.model == "M":
        table = find_table(VOL_REPORT, ["model", "horizon", "qlike", "mse_log"])
        row = table[(table["model"] == "M") & (table["horizon"] == 5)].iloc[0]
        difference = float(row["qlike_minus_baseline"])
        baseline = float(row["qlike"]) - difference
        return {
            "qlike_baseline": baseline,
            "qlike_minus_baseline": difference,
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "dm_p_value": float(row["dm_p_value"]),
        }
    # M3 and the per-firm mood (called M+SVmood in the phase 5 report) are both in its last
    # comparison table.
    name = "M3" if comparison.model == "M3" else "M+SVmood"
    table = find_table(NONLINEAR_REPORT, ["horizon", "model", "against"], occurrence=1)
    row = table[(table["model"] == name) & (table["against"] == "M")].iloc[0]
    return {
        key: float(row[key])
        for key in ("qlike_baseline", "qlike_minus_baseline", "ci_low", "ci_high", "dm_p_value")
    }


def _test_row(comparison: Comparison) -> dict[str, float]:
    """Test-period numbers at h = 5 from the phase 6 report."""
    if comparison.status == "pre-registered":
        table = find_table(TEST_REPORT, ["model", "against", "firm_days"])
    else:
        table = find_table(TEST_REPORT, ["horizon", "model", "against", "firm_days"])
        table = table[table["horizon"] == 5]
    row = table[(table["model"] == comparison.model) & (table["against"] == comparison.against)]
    row = row.iloc[0]
    return {
        key: float(row[key])
        for key in ("qlike_baseline", "qlike_minus_baseline", "ci_low", "ci_high", "dm_p_value")
    }


def volatility_validation_vs_test() -> pd.DataFrame:
    """QLIKE change at h = 5 in percent of the baseline's QLIKE, validation and test.

    Negative = the richer model forecasts better. Percent of the baseline makes the two periods
    comparable: test-period QLIKE is higher overall because volatility behaved differently.
    """
    rows = []
    for comparison in VOL_COMPARISONS:
        for period, numbers in (
            (VALIDATION, _validation_row(comparison)),
            (TEST, _test_row(comparison)),
        ):
            scale = 100.0 / numbers["qlike_baseline"]
            rows.append(
                {
                    "comparison": comparison.label,
                    "status": comparison.status,
                    "period": period,
                    "change_pct": numbers["qlike_minus_baseline"] * scale,
                    "ci_low_pct": numbers["ci_low"] * scale,
                    "ci_high_pct": numbers["ci_high"] * scale,
                    "dm_p_value": numbers["dm_p_value"],
                }
            )
    return pd.DataFrame(rows)
