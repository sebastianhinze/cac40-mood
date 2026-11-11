"""Volatility baselines on the validation period.

    uv run python -m cac40_mood.vol.run

For each horizon (1, 5 and 22 trading days) this produces out-of-sample forecasts for every
validation day from two baselines, pooled HAR with firm fixed effects and GARCH(1,1), and a
comparison table. Models that add announcement or sentiment columns to HAR are evaluated with
the same two functions (`har.expanding_forecasts`, `evaluate.compare_models`).

Only the 36 firms of the main analysis and only train and validation days are used. The last
forecast day for horizon h is h trading days before the end of validation, so that no target
window reaches into the test period.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cac40_mood.config import PROCESSED_DIR, PROJECT_ROOT
from cac40_mood.data.build import PRICES_PATH
from cac40_mood.data.universe import load_universe, main_analysis_firms
from cac40_mood.splits import TRAIN_END, VAL_END
from cac40_mood.vol import evaluate, garch, har
from cac40_mood.vol.realized import variance_panel

FORECASTS_PATH: Path = PROCESSED_DIR / "vol" / "baseline_forecasts_validation.parquet"
REPORT_PATH: Path = PROJECT_ROOT / "reports" / "04_vol_baseline_validation.md"


def load_panel() -> pd.DataFrame:
    """Daily variance and returns for the 36 firms, up to the end of validation."""
    prices = pd.read_parquet(PRICES_PATH)
    return variance_panel(prices, main_analysis_firms(load_universe()))


def baseline_forecasts(panel: pd.DataFrame) -> dict[int, dict[str, pd.DataFrame]]:
    """{horizon: {"HAR": forecasts, "GARCH": forecasts}} for the validation period."""
    features = har.har_features(panel)
    first_origin = pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)
    results: dict[int, dict[str, pd.DataFrame]] = {}
    for horizon in har.HORIZONS:
        har_table = har.expanding_forecasts(
            features, har.HAR_COLUMNS, horizon, first_origin, pd.Timestamp(VAL_END)
        )
        results[horizon] = {
            "HAR": har_table,
            "GARCH": garch.garch_forecasts(panel, har_table, horizon),
        }
    return results


def comparison_table(results: dict[int, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    tables = [
        evaluate.compare_models(models, "HAR", horizon) for horizon, models in results.items()
    ]
    return pd.concat(tables, ignore_index=True)


def write_report(table: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Save the comparison as a small Markdown report."""
    shown = table.round(4).astype(object).where(table.notna(), "")
    lines = [
        "# Volatility baselines on the validation period",
        "",
        f"36 firms, validation days 2023-01-02 to {VAL_END}; models re-estimated every 22 trading",
        "days on an expanding window that starts in 2018. Target: Garman-Klass variance, averaged",
        "over the next h trading days. Lower QLIKE is better. `qlike_minus_baseline` and the",
        "Diebold-Mariano columns compare each model with HAR (negative = better than HAR).",
        "",
        "| " + " | ".join(shown.columns) + " |",
        "|" + "---|" * len(shown.columns),
    ]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in shown.to_numpy()]
    lines += ["", f"Firm-days in the panel (2018 to end of validation): {len(panel):,}.", ""]
    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    panel = load_panel()
    results = baseline_forecasts(panel)
    table = comparison_table(results)
    stacked = pd.concat(
        [frame.assign(model=name) for models in results.values() for name, frame in models.items()],
        ignore_index=True,
    )
    FORECASTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stacked.to_parquet(FORECASTS_PATH, index=False)
    write_report(table, panel)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
