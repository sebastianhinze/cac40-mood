"""Do announcement scores and the filtered mood improve volatility forecasts? (validation)

    uv run python -m cac40_mood.mood.real          # first: fit and filter the mood
    uv run python -m cac40_mood.vol.sentiment_run

Models, all pooled log-HAR with firm fixed effects, re-estimated every 22 trading days:

    B0  HAR only
    B1  B0 + "an announcement of this type happened today" (4 columns)
    M   B1 + |tone| and uncertainty of today's documents            <- RQ3: M against B1
    M2  M  + the firm's filtered mood and the market mood           <- RQ4: M2 against M

Exploratory: M plus bad-news tone and |surprise|; B1 plus the lexicon scores; B1 plus the
local-model score (RQ5: is the LLM worth it?).

Every model sees exactly the same firm-days: 2019 onwards, without the spring-2020 feed hole
(where announcement columns are missing, not zero). This is the validation period; the
confirmatory test is the one-off run on the test period in phase 6.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cac40_mood.analysis_documents import load_documents
from cac40_mood.config import PROCESSED_DIR, PROJECT_ROOT
from cac40_mood.mood.real import MOOD_PATH
from cac40_mood.multiple_tests import holm
from cac40_mood.splits import ROBUSTNESS_SAMPLE_START, TRAIN_END, VAL_END
from cac40_mood.vol import evaluate, features, har
from cac40_mood.vol.run import load_panel

MOOD_COLUMNS: list[str] = ["mood_total", "mood_market"]
B1_COLUMNS: list[str] = [*har.HAR_COLUMNS, *features.EVENT_COLUMNS]
M_COLUMNS: list[str] = [*B1_COLUMNS, *features.SCORE_COLUMNS]
MODELS: dict[str, list[str]] = {
    "B0": har.HAR_COLUMNS,
    "B1": B1_COLUMNS,
    "M": M_COLUMNS,
    "M2": [*M_COLUMNS, *MOOD_COLUMNS],
    "M+extra": [*M_COLUMNS, *features.EXTRA_COLUMNS],
    "B1+lexicon": [*B1_COLUMNS, *features.LEXICON_COLUMNS],
    "B1+local": [*B1_COLUMNS, *features.LOCAL_MODEL_COLUMNS],
    "B1+LM": [*B1_COLUMNS, *features.LOUGHRAN_COLUMNS],
}
PRIMARY_HORIZON: int = 5

FORECASTS_PATH: Path = PROCESSED_DIR / "vol" / "sentiment_forecasts_validation.parquet"
REPORT_PATH: Path = PROJECT_ROOT / "reports" / "04_vol_sentiment_validation.md"


def build_frame(panel: pd.DataFrame, documents: pd.DataFrame, mood: pd.DataFrame) -> pd.DataFrame:
    """HAR columns and targets, announcement and score columns, mood columns; one row per
    firm-day, keeping only days on which every model's columns are known."""
    frame = har.har_features(panel)
    frame = features.attach_features(frame, documents)
    frame = frame.merge(
        mood[["firm_id", "date", *MOOD_COLUMNS]], on=["firm_id", "date"], how="left"
    )
    return frame.dropna(subset=[*features.ALL_COLUMNS, *MOOD_COLUMNS]).reset_index(drop=True)


def forecasts_for(frame: pd.DataFrame, horizon: int) -> dict[str, pd.DataFrame]:
    """Validation-period forecasts of every model for one horizon."""
    first_origin = pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)
    return {
        name: har.expanding_forecasts(frame, columns, horizon, first_origin, pd.Timestamp(VAL_END))
        for name, columns in MODELS.items()
    }


def with_interval(table: pd.DataFrame) -> pd.DataFrame:
    """Add a 95% interval for the QLIKE difference: difference +/- 1.96 standard errors.

    The standard error is not stored by the test, but statistic = difference / standard error.
    """
    standard_error = table["qlike_minus_baseline"] / table["dm_statistic"]
    table = table.copy()
    table["ci_low"] = table["qlike_minus_baseline"] - 1.96 * standard_error
    table["ci_high"] = table["qlike_minus_baseline"] + 1.96 * standard_error
    return table


def comparison_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(every model against B1, M2 against M, all forecasts stacked) for the three horizons."""
    against_b1, mood_against_m, stacked = [], [], []
    for horizon in har.HORIZONS:
        forecasts = forecasts_for(frame, horizon)
        against_b1.append(evaluate.compare_models(forecasts, "B1", horizon))
        pair = {"M": forecasts["M"], "M2": forecasts["M2"]}
        mood_against_m.append(evaluate.compare_models(pair, "M", horizon))
        stacked += [table.assign(model=name) for name, table in forecasts.items()]
    return (
        with_interval(pd.concat(against_b1, ignore_index=True)),
        with_interval(pd.concat(mood_against_m, ignore_index=True)),
        pd.concat(stacked, ignore_index=True),
    )


def primary_tests(against_b1: pd.DataFrame, mood_against_m: pd.DataFrame) -> pd.DataFrame:
    """The two pre-registered tests at h = 5 with Holm-adjusted p-values."""
    rq3 = against_b1[(against_b1["model"] == "M") & (against_b1["horizon"] == PRIMARY_HORIZON)]
    rq4 = mood_against_m[
        (mood_against_m["model"] == "M2") & (mood_against_m["horizon"] == PRIMARY_HORIZON)
    ]
    table = pd.concat([rq3.assign(test="RQ3: M vs B1"), rq4.assign(test="RQ4: M2 vs M")])
    table["holm_p_value"] = holm(table["dm_p_value"].tolist())
    columns = ["test", "qlike_minus_baseline", "ci_low", "ci_high", "dm_statistic", "dm_p_value"]
    return table[[*columns, "holm_p_value"]].reset_index(drop=True)


def train_coefficients(frame: pd.DataFrame) -> pd.DataFrame:
    """Slopes of model M2 at h = 5, fitted on the train period (description, not a test)."""
    target = f"target_{PRIMARY_HORIZON}"
    train = frame[frame[f"target_end_{PRIMARY_HORIZON}"] <= pd.Timestamp(TRAIN_END)]
    model = har.fit_pooled(train, MODELS["M2"], target)
    return model.slopes.rename("slope").to_frame()


def markdown(table: pd.DataFrame, digits: int = 4) -> str:
    """A table as Markdown, numbers rounded, missing values blank."""
    shown = table.round(digits).astype(object).where(table.notna(), "")
    lines = ["| " + " | ".join(str(c) for c in shown.columns) + " |", "|" + "---|" * shown.shape[1]]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in shown.to_numpy()]
    return "\n".join(lines)


def write_report(
    against_b1: pd.DataFrame,
    mood_against_m: pd.DataFrame,
    primary: pd.DataFrame,
    robustness: pd.DataFrame,
    slopes: pd.DataFrame,
    removed: pd.DataFrame,
    frame: pd.DataFrame,
) -> None:
    feature_days = int((frame[features.EVENT_COLUMNS].sum(axis=1) > 0).sum())
    lines = [
        "# Volatility: do scores and mood add to HAR? (validation period)",
        "",
        f"36 firms, {len(frame):,} firm-days from 2019 to {VAL_END} without the spring-2020 feed",
        f"hole; {feature_days:,} of them have at least one scored announcement. Forecasts are",
        "out-of-sample for 2023-01 to 2024-06, models re-estimated every 22 trading days.",
        "Target: Garman-Klass variance averaged over the next h days. Lower QLIKE is better;",
        "a negative `qlike_minus_baseline` means better than the baseline. Tests are",
        "Diebold-Mariano on the daily cross-firm average loss difference (Newey-West).",
        "**This is the validation period. The confirmatory test is the one-off test-period run.**",
        "",
        "## Primary tests at h = 5 (pre-registered; Holm-adjusted)",
        "",
        markdown(primary, 5),
        "",
        "## Every model against B1 (HAR + announcement dummies)",
        "",
        markdown(against_b1.drop(columns=["firm_days"]), 5),
        "",
        "B0 = HAR only. M = B1 + |tone| + uncertainty. M2 = M + filtered mood. `M+extra`,",
        "`B1+lexicon`, `B1+local` and `B1+LM` (Loughran-McDonald on official English versions)",
        "are exploratory.",
        "",
        "## Mood against raw scores: M2 against M",
        "",
        markdown(mood_against_m.drop(columns=["firm_days"]), 5),
        "",
        f"## Robustness at h = 5: sample starts {ROBUSTNESS_SAMPLE_START}",
        "",
        markdown(robustness.drop(columns=["firm_days"]), 5),
        "",
        "## Slopes of model M2 at h = 5, train period (description only)",
        "",
        markdown(slopes.reset_index(names="column"), 4),
        "",
        "## Documents behind the features",
        "",
        markdown(removed),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    documents, removed = load_documents()
    frame = build_frame(load_panel(), documents, pd.read_parquet(MOOD_PATH))
    against_b1, mood_against_m, stacked = comparison_tables(frame)
    primary = primary_tests(against_b1, mood_against_m)

    late = frame[frame["date"] >= pd.Timestamp(ROBUSTNESS_SAMPLE_START)]
    late_forecasts = forecasts_for(late, PRIMARY_HORIZON)
    keep = {name: late_forecasts[name] for name in ("B1", "M", "M2")}
    robustness = with_interval(evaluate.compare_models(keep, "B1", PRIMARY_HORIZON))

    FORECASTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stacked.to_parquet(FORECASTS_PATH, index=False)
    write_report(
        against_b1, mood_against_m, primary, robustness, train_coefficients(frame), removed, frame
    )
    print(primary.round(5).to_string(index=False))
    print(against_b1.round(5).to_string(index=False))
    print(mood_against_m.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
