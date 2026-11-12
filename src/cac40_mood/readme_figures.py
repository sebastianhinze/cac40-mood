"""The three README charts, drawn from the committed reports (no data, no API calls).

    uv run python -m cac40_mood.readme_figures

Writes to reports/figures/:
- 07_day0_by_tone.png             day-0 abnormal return by tone third, train+validation vs test
- 07_scorers_day0.png             LLM against the three baselines on the day-0 return
- 07_vol_validation_vs_test.png   the volatility gains seen on validation, and what was left on test
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cac40_mood import plotting
from cac40_mood.presentation import (
    REPORTS_DIR,
    TEST,
    TRAIN_VAL,
    VALIDATION,
    day0_terciles,
    scorer_slopes,
    volatility_validation_vs_test,
)

FIGURES_DIR: Path = REPORTS_DIR / "figures"

# 95% interval from a standard error (normal approximation, as in the reports).
Z_95: float = 1.96

PERIOD_COLOURS: dict[str, str] = {
    TRAIN_VAL: plotting.SERIES[0],
    VALIDATION: plotting.SERIES[0],
    TEST: plotting.SERIES[1],
}


def day0_figure() -> Path:
    """Grouped bars: mean CAR[0] (in %) for the low, middle and high tone third, two periods."""
    data = day0_terciles()
    order = ["low", "middle", "high"]
    figure, axes = plt.subplots(figsize=(7.2, 3.9))
    width = 0.36
    positions = np.arange(len(order))
    for offset, period in ((-width / 2, TRAIN_VAL), (width / 2, TEST)):
        rows = data[data["period"] == period].set_index("tone_tercile").loc[order]
        means = rows["mean"].to_numpy() * 100
        errors = rows["std_error"].to_numpy() * 100 * Z_95
        events = int(rows["events"].sum())
        axes.bar(
            positions + offset,
            means,
            width=width - 0.03,
            color=PERIOD_COLOURS[period],
            label=f"{period}, {events:,} events",
        )
        axes.errorbar(
            positions + offset,
            means,
            yerr=errors,
            fmt="none",
            ecolor=plotting.INK_SECONDARY,
            elinewidth=1,
            capsize=3,
        )
    axes.axhline(0, color=plotting.AXIS, linewidth=1)
    axes.set_xticks(positions, [f"{name} tone third" for name in order])
    axes.set_ylabel("abnormal return on day 0 (%)")
    axes.legend(loc="upper left")
    plotting.titled(
        axes,
        "Prices move with the LLM's tone on the announcement day",
        "Mean day-0 abnormal return by tone third; bars show 95% intervals",
    )
    path = plotting.save(figure, FIGURES_DIR / "07_day0_by_tone.png")
    plt.close(figure)
    return path


def scorers_figure() -> Path:
    """Dot-and-interval chart: day-0 return per standard deviation of tone, four scorers."""
    data = scorer_slopes()
    scorers = ["LLM", "Local model", "Loughran-McDonald", "Lexicon"]
    labels = {
        "LLM": "LLM (Claude Haiku)",
        "Local model": "Local model (FinBERT, FR/EN)",
        "Loughran-McDonald": "Loughran-McDonald dictionary",
        "Lexicon": "My own word list",
    }
    figure, axes = plt.subplots(figsize=(7.2, 3.6))
    rows_y = np.arange(len(scorers))[::-1]
    for offset, period in ((0.14, TRAIN_VAL), (-0.14, TEST)):
        rows = data[data["period"] == period].set_index("scorer").loc[scorers]
        n = int(rows["n"].iloc[0])
        coefficients = rows["coefficient"].to_numpy() * 100
        low = coefficients - rows["ci_low"].to_numpy() * 100
        high = rows["ci_high"].to_numpy() * 100 - coefficients
        axes.errorbar(
            coefficients,
            rows_y + offset,
            xerr=[low, high],
            fmt="o",
            markersize=7,
            color=PERIOD_COLOURS[period],
            ecolor=PERIOD_COLOURS[period],
            elinewidth=2,
            capsize=0,
            label=f"{period}, {n:,} events",
        )
    axes.axvline(0, color=plotting.AXIS, linewidth=1)
    axes.set_yticks(rows_y, [labels[name] for name in scorers])
    axes.grid(axis="x")
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("day-0 abnormal return per 1 SD of tone (%)")
    axes.legend(loc="center right")
    plotting.titled(
        axes,
        "The LLM's tone has the largest day-0 slope of the four scorers",
        "Same events for every scorer; 95% intervals clustered by date. On test they overlap",
    )
    path = plotting.save(figure, FIGURES_DIR / "07_scorers_day0.png")
    plt.close(figure)
    return path


def volatility_figure() -> Path:
    """Dot-and-interval chart: QLIKE change at h = 5, validation vs the frozen test."""
    data = volatility_validation_vs_test()
    comparisons = list(dict.fromkeys(data["comparison"]))
    figure, axes = plt.subplots(figsize=(7.2, 3.4))
    rows_y = np.arange(len(comparisons))[::-1]
    for offset, period in ((0.14, VALIDATION), (-0.14, TEST)):
        rows = data[data["period"] == period].set_index("comparison").loc[comparisons]
        change = rows["change_pct"].to_numpy()
        axes.errorbar(
            change,
            rows_y + offset,
            xerr=[change - rows["ci_low_pct"].to_numpy(), rows["ci_high_pct"].to_numpy() - change],
            fmt="o",
            markersize=7,
            color=PERIOD_COLOURS[period],
            ecolor=PERIOD_COLOURS[period],
            elinewidth=2,
            capsize=0,
            label=period,
        )
        # The p-value sits just right of each interval, so the two periods never collide.
        ends = rows["ci_high_pct"].to_numpy()
        for x, y, p in zip(ends, rows_y + offset, rows["dm_p_value"], strict=True):
            axes.annotate(
                f"p = {p:.3f}" if p >= 0.001 else "p < 0.001",
                xy=(x, y),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=plotting.INK_SECONDARY,
            )
    axes.axvline(0, color=plotting.AXIS, linewidth=1)
    tick_labels = [
        f"{name}\n({status})"
        for name, status in data.drop_duplicates("comparison")[["comparison", "status"]].itertuples(
            index=False
        )
    ]
    axes.set_yticks(rows_y, tick_labels)
    axes.set_ylim(rows_y.min() - 0.6, rows_y.max() + 0.6)
    axes.grid(axis="x")
    axes.grid(axis="y", visible=False)
    axes.set_xlabel("change in 5-day QLIKE loss, % of baseline (left = better forecast)")
    axes.set_xlim(right=1.15)
    axes.legend(loc="upper left")
    plotting.titled(
        axes,
        "Volatility gains seen on validation did not survive the frozen test",
        "Diebold-Mariano p-values (unadjusted); lines show 95% intervals",
    )
    path = plotting.save(figure, FIGURES_DIR / "07_vol_validation_vs_test.png")
    plt.close(figure)
    return path


def main() -> None:
    """Draw all three charts and print where they went."""
    plotting.apply_style()
    for path in (day0_figure(), scorers_figure(), volatility_figure()):
        print(path.relative_to(REPORTS_DIR.parent))
    pd.set_option("display.width", 140)
    print(volatility_validation_vs_test().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
