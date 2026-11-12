"""Phase 6: the one-off evaluation on the test period.

    uv run python -m cac40_mood.phase6 rehearsal    # same pipeline on the VALIDATION period
    uv run python -m cac40_mood.phase6 test         # the real run: once, on the tagged commit

What it does is fixed in docs/PHASE6_PLAN.md and `phase6_config.py`. This module only passes a
later end date to the functions that produced the validation results; it contains no model
logic of its own.

Two safeguards:
- `rehearsal` runs every step on the validation period (2023-01 to 2024-06) and loads nothing
  after it. It must reproduce the validation numbers, which proves the plumbing before the
  test period is touched.
- `test` refuses to run unless the working tree is clean, the current commit carries the tag
  `phase6-freeze`, and no test result exists yet.

Everything that was estimated stays as it was: the tone centre and the mood and volatility-
equation parameters come from `phase6_config.py` (train period). Only the HAR-type regressions
are re-estimated every 22 trading days on an expanding window, as on validation.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cac40_mood import phase6_config as config
from cac40_mood.analysis_documents import load_documents
from cac40_mood.config import PROCESSED_DIR, PROJECT_ROOT
from cac40_mood.data.build import PRICES_PATH, SESSIONS_PATH
from cac40_mood.events import run as events_run
from cac40_mood.mood import real, sv, sv_run
from cac40_mood.mood.model import MoodParameters
from cac40_mood.multiple_tests import benjamini_hochberg, holm
from cac40_mood.splits import TRAIN_END, VAL_END
from cac40_mood.vol import evaluate, features, har
from cac40_mood.vol.run import load_panel
from cac40_mood.vol.sentiment_run import MODELS, build_frame, markdown, with_interval

FREEZE_TAG: str = "phase6-freeze"
REPORT_PATH: Path = PROJECT_ROOT / "reports" / "06_test_results.md"
OUTPUT_DIR: Path = PROCESSED_DIR / "phase6"
HORIZONS: tuple[int, ...] = (1, 5, 22)


@dataclass(frozen=True)
class Period:
    """The evaluation period and what may be loaded for it."""

    name: str
    first_origin: date
    last_day: date
    splits: tuple[str, ...]


TEST_PERIOD = Period("test", config.TEST_START, config.TEST_END, ("train", "val", "test"))
REHEARSAL_PERIOD = Period("rehearsal (validation)", date(2023, 1, 1), VAL_END, ("train", "val"))


# ------------------------------------------------------------------ models


def models_for(horizon: int) -> dict[str, list[str]]:
    """Column lists of every model at one horizon (the mood models use that horizon's forecast)."""
    m = MODELS["M"]
    return {
        "B0": MODELS["B0"],
        "B1": MODELS["B1"],
        "M": m,
        "M3": [*m, f"sv_log_forecast_{horizon}", "sv_mood"],
        "M2": MODELS["M2"],
        "M+firm mood": [*m, "sv_mood"],
        "B1+lexicon": MODELS["B1+lexicon"],
        "B1+local": MODELS["B1+local"],
        "B1+LM": MODELS["B1+LM"],
    }


# (model, baseline): the comparisons reported at every horizon.
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("M", "B1"),
    ("M3", "M"),
    ("B0", "B1"),
    ("M2", "M"),
    ("M+firm mood", "M"),
    ("B1+lexicon", "B1"),
    ("B1+local", "B1"),
    ("B1+LM", "B1"),
)
PRIMARY: tuple[tuple[str, str], ...] = (
    (config.RQ3_MODEL, config.RQ3_BASELINE),
    (config.RQ4_MODEL, config.RQ4_BASELINE),
)


# ---------------------------------------------------------------- features


def nonlinear_columns(documents: pd.DataFrame, period: Period) -> pd.DataFrame:
    """Filtered mood and variance forecasts of the nonlinear model, with the frozen parameters.

    Nothing is estimated here. The filter starts in 2019 and runs to the end of the period.
    """
    calendar = real.sample_sessions(pd.read_parquet(SESSIONS_PATH), period.last_day)
    dates = calendar["date"]
    missing_days = np.flatnonzero(~calendar["feed_observed"].to_numpy())
    n_train_days = int((dates <= pd.Timestamp(TRAIN_END)).sum())
    firms = sorted(documents["firm_id"].unique())
    mood = MoodParameters(**config.MOOD_PARAMETERS)
    parameters = sv.SVParameters(**config.SV_PARAMETERS)
    readings = real.make_readings(documents, dates, config.TONE_CENTRE_TRAIN)
    returns = sv_run.returns_matrix(load_panel(period.last_day), firms, dates)
    data = sv_run.build_data(readings, returns, firms, mood, missing_days, n_train_days)
    result = sv.sv_filter(mood, parameters, data, config.FILTER_PARTICLES, config.FILTER_SEED)
    pieces = []
    for number, firm in enumerate(firms):
        columns = {"firm_id": firm, "date": dates.to_numpy(), "sv_mood": result.mood[number]}
        for horizon in HORIZONS:
            columns[f"sv_log_forecast_{horizon}"] = result.log_forecast[horizon][number]
        pieces.append(pd.DataFrame(columns))
    return pd.concat(pieces, ignore_index=True)


def build_model_frame(period: Period) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(firm-day table with every model's columns, the documents behind it)."""
    documents, _ = load_documents(period.splits)
    linear_mood, _ = real.run(documents, pd.read_parquet(SESSIONS_PATH), period.last_day)
    frame = build_frame(load_panel(period.last_day), documents, linear_mood)
    nonlinear = nonlinear_columns(documents, period)
    frame = frame.merge(nonlinear, on=["firm_id", "date"]).dropna(subset=["sv_mood"])
    return frame.reset_index(drop=True), documents


# -------------------------------------------------------------- evaluation


def forecasts_for(frame: pd.DataFrame, period: Period, horizon: int) -> dict[str, pd.DataFrame]:
    """Out-of-sample forecasts of every model for the period's origins."""
    return {
        name: har.expanding_forecasts(
            frame,
            columns,
            horizon,
            pd.Timestamp(period.first_origin),
            pd.Timestamp(period.last_day),
            config.REFIT_EVERY_TRADING_DAYS,
        )
        for name, columns in models_for(horizon).items()
    }


def comparison_row(
    forecasts: dict[str, pd.DataFrame], model: str, baseline: str, horizon: int
) -> dict[str, object]:
    """`model` against `baseline`: QLIKE of both, the difference and its test."""
    pair = {baseline: forecasts[baseline], model: forecasts[model]}
    table = evaluate.compare_models(pair, baseline, horizon).set_index("model")
    return {
        "horizon": horizon,
        "model": model,
        "against": baseline,
        "firm_days": int(table.loc[model, "firm_days"]),
        "qlike_baseline": table.loc[baseline, "qlike"],
        "qlike_model": table.loc[model, "qlike"],
        "qlike_minus_baseline": table.loc[model, "qlike_minus_baseline"],
        "dm_statistic": table.loc[model, "dm_statistic"],
        "dm_p_value": table.loc[model, "dm_p_value"],
    }


def evaluate_models(
    frame: pd.DataFrame, period: Period
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(primary tests, exploratory comparisons, model fit table).

    Primary: the two pre-registered comparisons at the primary horizon, Holm-adjusted; a
    result is `confirmed` if the adjusted p-value is below the level and the model is better.
    Exploratory: every other comparison, with Benjamini-Hochberg p-values.
    """
    rows, fits = [], []
    for horizon in HORIZONS:
        forecasts = forecasts_for(frame, period, horizon)
        rows += [comparison_row(forecasts, model, base, horizon) for model, base in COMPARISONS]
        fits.append(evaluate.compare_models(forecasts, "B1", horizon))
    table = with_interval(pd.DataFrame(rows))
    pairs = list(zip(table["model"], table["against"], strict=True))
    pre_registered = np.array([pair in PRIMARY for pair in pairs])
    is_primary = (table["horizon"] == config.PRIMARY_HORIZON).to_numpy() & pre_registered

    primary = table[is_primary].copy()
    primary["holm_p_value"] = holm(primary["dm_p_value"].tolist())
    primary["confirmed"] = (primary["holm_p_value"] < config.SIGNIFICANCE_LEVEL) & (
        primary["qlike_minus_baseline"] < 0
    )
    exploratory = table[~is_primary].copy()
    exploratory["bh_p_value"] = benjamini_hochberg(exploratory["dm_p_value"].tolist())
    fit_columns = ["model", "horizon", "firm_days", "qlike", "mse_log", "mz_slope", "mz_r2"]
    return primary, exploratory, pd.concat(fits, ignore_index=True)[fit_columns]


def event_study(documents: pd.DataFrame, period: Period) -> dict[str, pd.DataFrame]:
    """Event-study tables for events inside the evaluation period (exploratory)."""
    prices = pd.read_parquet(PRICES_PATH)
    events = events_run.build_events(documents, prices, period.last_day, period.splits)
    events = events[events["event_session"] >= pd.Timestamp(period.first_origin)]
    return {
        "events": events,
        "scores": events_run.score_regressions(events),
        "terciles": events_run.tercile_means(events),
        "scorers": events_run.scorer_comparison(events),
    }


# ------------------------------------------------------------------ report


def report_text(
    period: Period,
    commit: str,
    primary: pd.DataFrame,
    exploratory: pd.DataFrame,
    fits: pd.DataFrame,
    events: dict[str, pd.DataFrame],
    frame: pd.DataFrame,
) -> str:
    in_period = frame[frame["date"] >= pd.Timestamp(period.first_origin)]
    announcement_days = int((in_period[features.EVENT_COLUMNS].sum(axis=1) > 0).sum())
    with_car = int(events["events"]["car_0"].notna().sum())
    lines = [
        f"# Phase 6 results: {period.name} period",
        "",
        f"Period {period.first_origin} to {period.last_day}.",
        f"Commit `{commit}` (tag `{FREEZE_TAG}`).",
        "Run once; nothing was changed afterwards. Plan: `docs/PHASE6_PLAN.md`.",
        f"{in_period['firm_id'].nunique()} firms, {len(in_period):,} firm-days in the period, "
        f"{announcement_days:,} with a scored announcement.",
        "",
        "## Confirmatory results (pre-registered, Holm-adjusted)",
        "",
        "h = 5 trading days, QLIKE on Garman-Klass variance, Diebold-Mariano on the daily",
        "cross-firm mean loss difference. Negative `qlike_minus_baseline` = the model is better.",
        "`confirmed` = Holm p-value below 0.05 and the model better.",
        "",
        markdown(primary.drop(columns=["horizon"]), 5),
        "",
        "RQ3 is the row M against B1 (do scores add to announcement dummies?). RQ4 is the row M3",
        "against M (does the mood model add to the scores?).",
        "",
        "## Exploratory results (not pre-registered; Benjamini-Hochberg p-values)",
        "",
        "### Volatility: other horizons and other models",
        "",
        markdown(exploratory, 5),
        "",
        "B0 = HAR only. M2 = M + linear two-stage mood. `M+firm mood` = M + the per-firm mood",
        "alone. `B1+lexicon`, `B1+local`, `B1+LM` = the baseline scorers in place of the LLM.",
        "",
        "### Fit of every model",
        "",
        markdown(fits, 5),
        "",
        f"### Event study on the period's events ({len(events['events']):,} events, "
        f"{with_car:,} with a complete window)",
        "",
        markdown(events["scores"]),
        "",
        markdown(events["terciles"], 5),
        "",
        "Scorers, slope per standard deviation of tone on CAR[0]:",
        "",
        markdown(events["scorers"], 5),
        "",
    ]
    return "\n".join(lines)


def run(period: Period, commit: str) -> str:
    """Build everything for the period and return the report text."""
    frame, documents = build_model_frame(period)
    primary, exploratory, fits = evaluate_models(frame, period)
    events = event_study(documents, period)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label = "test" if period is TEST_PERIOD else "rehearsal"
    primary.to_csv(OUTPUT_DIR / f"{label}_primary.csv", index=False)
    exploratory.to_csv(OUTPUT_DIR / f"{label}_exploratory.csv", index=False)
    return report_text(period, commit, primary, exploratory, fits, events, frame)


# ------------------------------------------------------------------ guards


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def check_frozen(report_path: Path = REPORT_PATH) -> str:
    """Return the commit hash if the test run is allowed, otherwise raise.

    Allowed means: no result exists yet, nothing is uncommitted, and the commit we are on is
    the one tagged `phase6-freeze`.
    """
    if report_path.exists():
        raise RuntimeError(f"{report_path.name} exists: the test period has already been run.")
    if git_output("status", "--porcelain"):
        raise RuntimeError("Uncommitted changes: commit and tag before the test run.")
    head = git_output("rev-parse", "HEAD")
    tagged = git_output("rev-list", "-n", "1", FREEZE_TAG)
    if head != tagged:
        raise RuntimeError(f"HEAD is not the commit tagged {FREEZE_TAG}.")
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: one-off test-period evaluation.")
    parser.add_argument("step", choices=["rehearsal", "test"])
    args = parser.parse_args()
    if args.step == "rehearsal":
        text = run(REHEARSAL_PERIOD, git_output("rev-parse", "HEAD"))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "rehearsal.md").write_text(text, encoding="utf-8")
    else:
        commit = check_frozen()
        text = run(TEST_PERIOD, commit)
        REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
