"""Does the nonlinear mood model improve volatility forecasts? (validation; DECISIONS #58)

    uv run python -m cac40_mood.mood.sv_run        # first: fit and filter
    uv run python -m cac40_mood.vol.nonlinear_run

Pre-registered primary test: at h = 5, QLIKE, Diebold-Mariano,

    M3 = M + the nonlinear model's log forecast for the horizon + its filtered mood
    against M (HAR + announcement dummies + |tone| + uncertainty).

Exploratory: M3 against `M+extra`; against `M+SV0` (the same model with beta = 0, to separate
the mood from "any stochastic-volatility forecast"); against M2 (linear mood); h = 1 and 22.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cac40_mood.analysis_documents import load_documents
from cac40_mood.config import PROJECT_ROOT
from cac40_mood.mood.real import MOOD_PATH
from cac40_mood.mood.sv import HORIZONS
from cac40_mood.mood.sv_run import SV_PARAMETERS_PATH, SV_PATH
from cac40_mood.splits import TRAIN_END, VAL_END
from cac40_mood.vol import evaluate, har
from cac40_mood.vol.run import load_panel
from cac40_mood.vol.sentiment_run import MODELS, build_frame, markdown, with_interval

REPORT_PATH: Path = PROJECT_ROOT / "reports" / "05_nonlinear_mood_validation.md"
PRIMARY_HORIZON: int = 5
BASELINES: tuple[str, ...] = ("M", "M+extra", "M+SV0", "M2")


def models_for(horizon: int) -> dict[str, list[str]]:
    """Column lists of the models compared at one horizon."""
    return {
        "M": MODELS["M"],
        "M+extra": MODELS["M+extra"],
        "M2": MODELS["M2"],
        "M+SV0": [*MODELS["M"], f"sv0_log_forecast_{horizon}"],
        "M3": [*MODELS["M"], f"sv_log_forecast_{horizon}", "sv_mood"],
        # The two halves of M3, added after the result was seen, to find out which half helps.
        "M+SVforecast": [*MODELS["M"], f"sv_log_forecast_{horizon}"],
        "M+SVmood": [*MODELS["M"], "sv_mood"],
    }


def add_nonlinear_columns(frame: pd.DataFrame, nonlinear: pd.DataFrame) -> pd.DataFrame:
    """Attach the nonlinear model's columns; firm-days without them are dropped."""
    columns = [c for c in nonlinear.columns if c.startswith(("sv_log", "sv0_log"))] + ["sv_mood"]
    merged = frame.merge(nonlinear[["firm_id", "date", *columns]], on=["firm_id", "date"])
    return merged.dropna(subset=columns).reset_index(drop=True)


def all_forecasts(frame: pd.DataFrame) -> dict[int, dict[str, pd.DataFrame]]:
    """Validation forecasts of every model, per horizon."""
    first_origin = pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)
    return {
        horizon: {
            name: har.expanding_forecasts(
                frame, columns, horizon, first_origin, pd.Timestamp(VAL_END)
            )
            for name, columns in models_for(horizon).items()
        }
        for horizon in HORIZONS
    }


def pair_row(
    forecasts: dict[str, pd.DataFrame], model: str, baseline: str, horizon: int
) -> dict[str, object]:
    """One comparison: `model` against `baseline` (negative difference = model is better)."""
    pair = {baseline: forecasts[baseline], model: forecasts[model]}
    table = evaluate.compare_models(pair, baseline, horizon).set_index("model")
    return {
        "horizon": horizon,
        "model": model,
        "against": baseline,
        "qlike_baseline": table.loc[baseline, "qlike"],
        "qlike_model": table.loc[model, "qlike"],
        "qlike_minus_baseline": table.loc[model, "qlike_minus_baseline"],
        "dm_statistic": table.loc[model, "dm_statistic"],
        "dm_p_value": table.loc[model, "dm_p_value"],
    }


def comparisons(forecasts: dict[int, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """M3 against each baseline at each horizon."""
    rows = [
        pair_row(models, "M3", baseline, horizon)
        for horizon, models in forecasts.items()
        for baseline in BASELINES
    ]
    return with_interval(pd.DataFrame(rows))


def decomposition(forecasts: dict[int, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Which part of M3 helps? Each part alone against M, at h = 5 (post hoc, exploratory)."""
    models = forecasts[PRIMARY_HORIZON]
    rows = [
        pair_row(models, name, "M", PRIMARY_HORIZON)
        for name in ("M+SV0", "M+SVforecast", "M+SVmood", "M3", "M2")
    ]
    return with_interval(pd.DataFrame(rows))


def train_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    """Slopes of M3 at h = 5 fitted on the train period (description only)."""
    train = frame[frame[f"target_end_{PRIMARY_HORIZON}"] <= pd.Timestamp(TRAIN_END)]
    model = har.fit_pooled(train, models_for(PRIMARY_HORIZON)["M3"], f"target_{PRIMARY_HORIZON}")
    return model.slopes.rename("slope").to_frame().reset_index(names="column")


def write_report(
    table: pd.DataFrame, parts: pd.DataFrame, slopes: pd.DataFrame, fitted: dict
) -> None:
    primary = table[(table["horizon"] == PRIMARY_HORIZON) & (table["against"] == "M")]
    with_mood, without_mood, health = fitted["with_mood"], fitted["without_mood"], fitted["filter"]
    gains = ", ".join(f"{gain:.1f}" for gain in fitted["same_gain_under_three_other_seeds"])
    lines = [
        "# Nonlinear mood model (mood feeds into log-volatility): validation result",
        "",
        "Pre-registered in DECISIONS #58 before any of this was run. Model per firm: hidden mood",
        "(read through the tone scores) and hidden log-variance h, with",
        "h_t = mu + phi_h (h_{t-1} - mu) + beta * mood_{t-1} + shock, and",
        "return ~ Normal(0, exp(h)).",
        "Fitted on the train period with the particle filter; validation is 2023-01 to 2024-06.",
        "",
        "## Primary test: M3 against M at h = 5 (QLIKE, Diebold-Mariano)",
        "",
        markdown(primary.drop(columns=["horizon"]), 5),
        "",
        "## All comparisons (the others are exploratory)",
        "",
        "`M+SV0` = M plus the forecast of the same model with beta = 0 (no mood). `M2` = M plus",
        "the linear filtered mood. Negative `qlike_minus_baseline` = M3 is better.",
        "",
        markdown(table, 5),
        "",
        "## Which part of M3 helps? (h = 5, each against M; added after seeing the result)",
        "",
        "`M+SVforecast` = M plus only the nonlinear model's variance forecast. `M+SVmood` = M plus",
        "only its filtered mood. With beta at zero (below), that mood is in effect a linear",
        "per-firm mood filter: returns tell it nothing.",
        "",
        markdown(parts, 5),
        "",
        "## Fitted parameters (train period)",
        "",
        "| | phi_h | sigma_h | beta |",
        "|---|---|---|---|",
        f"| with mood | {with_mood['phi_h']:.4f} | {with_mood['sigma_h']:.4f} | "
        f"{with_mood['beta']:.4f} |",
        f"| without mood | {without_mood['phi_h']:.4f} | {without_mood['sigma_h']:.4f} | 0 |",
        "",
        f"Mood part: phi {fitted['mood']['phi']:.4f}, sigma_state "
        f"{fitted['mood']['sigma_state']:.4f}, sigma_obs {fitted['mood']['sigma_obs']:.4f}.",
        "",
        f"Train log-likelihood gain from letting beta differ from 0: "
        f"{fitted['train_log_likelihood_gain_from_beta']:.1f} (same seed); under three other "
        f"seeds: {gains}. A gain of about 2 would be the usual threshold for one extra parameter "
        "if the likelihood were exact; here it is estimated, so read it with that noise in mind.",
        "",
        "## Health of the particle filter (train and validation, final run)",
        "",
        f"- {health['particles']:,} particles per firm.",
        f"- Effective sample size: median {health['ess_median']:,.0f}, 5th percentile "
        f"{health['ess_5th_percentile']:,.0f}, minimum {health['ess_minimum']:,.0f}.",
        f"- Firm-days with an effective sample size below 100: {health['firm_days_ess_below_100']}"
        f" of {health['firm_days']:,} (days with an extreme return; the cloud is resampled"
        " right after).",
        f"- Resampled on {health['share_of_firm_days_resampled']:.1%} of firm-days.",
        "",
        "## Slopes of M3 at h = 5, train period (description only)",
        "",
        markdown(slopes, 4),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    documents, _ = load_documents()
    frame = build_frame(load_panel(), documents, pd.read_parquet(MOOD_PATH))
    frame = add_nonlinear_columns(frame, pd.read_parquet(SV_PATH))
    forecasts = all_forecasts(frame)
    table, parts = comparisons(forecasts), decomposition(forecasts)
    fitted = json.loads(SV_PARAMETERS_PATH.read_text())
    ess = pd.read_parquet(SV_PATH, columns=["sv_ess"])["sv_ess"]
    fitted["filter"]["firm_days_ess_below_100"] = int((ess < 100).sum())
    fitted["filter"]["firm_days"] = len(ess)
    write_report(table, parts, train_slopes(frame), fitted)
    print(table.round(5).to_string(index=False))
    print(parts.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
