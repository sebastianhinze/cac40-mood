"""The nonlinear mood model on real data: fit on train, filter train and validation.

    uv run python -m cac40_mood.mood.sv_run

Steps (pre-registered in DECISIONS #58):
1. Readings as in the linear model: tone minus the train-period mean, confidence, event session.
2. Mood parameters by exact Kalman likelihood on the train readings (one mood per firm, all
   firms share the parameters). These are then held fixed.
3. Volatility parameters (phi_h, sigma_h, beta) by particle-filter likelihood on the train
   returns and readings; the same again with beta fixed at 0 (the "no mood" twin).
4. Filter train and validation with both parameter sets and store, per firm and day, the
   filtered mood and the log forecasts of the next 1, 5 and 22 days' average variance.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from cac40_mood.analysis_documents import load_documents
from cac40_mood.data.build import SESSIONS_PATH
from cac40_mood.mood import sv
from cac40_mood.mood.estimate import fit_gaussian
from cac40_mood.mood.model import MoodParameters, make_observations
from cac40_mood.mood.real import MOOD_DIR, make_readings, sample_sessions
from cac40_mood.splits import TRAIN_END
from cac40_mood.vol.run import load_panel

SV_PATH: Path = MOOD_DIR / "nonlinear_mood.parquet"
SV_PARAMETERS_PATH: Path = MOOD_DIR / "nonlinear_parameters.json"

FIT_PARTICLES: int = 1_000
FILTER_PARTICLES: int = 5_000
SEED: int = 58


def fit_firm_mood(readings: pd.DataFrame, n_days: int, missing_days: np.ndarray) -> MoodParameters:
    """One mood per firm, same parameters for all firms, exact likelihood (normal errors)."""
    series = [
        make_observations(
            n_days,
            rows["day"].to_numpy(),
            rows["value"].to_numpy(),
            rows["confidence"].to_numpy(),
            missing_days,
        )
        for _, rows in readings.groupby("firm")
    ]
    return fit_gaussian(series)


def returns_matrix(panel: pd.DataFrame, firms: list[str], dates: pd.Series) -> np.ndarray:
    """Daily log returns as an array (firms, days) on the trading calendar; NaN where missing."""
    wide = panel.pivot_table(index="firm_id", columns="date", values="ret", aggfunc="first")
    return wide.reindex(index=firms, columns=pd.DatetimeIndex(dates)).to_numpy()


def build_data(
    readings: pd.DataFrame,
    returns: np.ndarray,
    firms: list[str],
    mood: MoodParameters,
    missing_days: np.ndarray,
    n_train_days: int,
) -> sv.SVData:
    """Filter input for all days. The variance level of each firm is a train-period number."""
    mean, precision = sv.combine_readings(
        readings, firms, returns.shape[1], mood.sigma_obs, missing_days
    )
    level = np.log(np.nanmean(returns[:, :n_train_days] ** 2, axis=1))
    return sv.SVData(firms, returns, mean, precision, level)


def train_part(data: sv.SVData, n_train_days: int) -> sv.SVData:
    """The same data cut at the last train day."""
    return sv.SVData(
        data.firms,
        data.returns[:, :n_train_days],
        data.reading_mean[:, :n_train_days],
        data.reading_precision[:, :n_train_days],
        data.log_variance_level,
    )


def result_table(
    with_mood: sv.SVResult, without_mood: sv.SVResult, firms: list[str], dates: pd.Series
) -> pd.DataFrame:
    """Long table: one row per firm and day with the columns the volatility models use."""
    pieces = []
    for number, firm in enumerate(firms):
        columns = {
            "firm_id": firm,
            "date": dates.to_numpy(),
            "sv_mood": with_mood.mood[number],
            "sv_ess": with_mood.effective_sample_size[number],
            "sv_resampled": with_mood.resampled[number],
        }
        for horizon in sv.HORIZONS:
            columns[f"sv_log_forecast_{horizon}"] = with_mood.log_forecast[horizon][number]
            columns[f"sv0_log_forecast_{horizon}"] = without_mood.log_forecast[horizon][number]
        pieces.append(pd.DataFrame(columns))
    return pd.concat(pieces, ignore_index=True)


def ess_summary(result: sv.SVResult, n_particles: int) -> dict[str, float]:
    """How healthy the particle cloud was: typical and worst effective sample size."""
    ess = result.effective_sample_size
    return {
        "particles": n_particles,
        "ess_median": float(np.median(ess)),
        "ess_5th_percentile": float(np.percentile(ess, 5)),
        "ess_minimum": float(ess.min()),
        "share_of_firm_days_resampled": float(result.resampled.mean()),
    }


def main() -> None:
    documents, _ = load_documents()
    calendar = sample_sessions(pd.read_parquet(SESSIONS_PATH))
    dates = calendar["date"]
    missing_days = np.flatnonzero(~calendar["feed_observed"].to_numpy())
    n_train_days = int((dates <= pd.Timestamp(TRAIN_END)).sum())
    firms = sorted(documents["firm_id"].unique())

    train_documents = documents[documents["split"] == "train"]
    centre = float(train_documents["tone"].mean())
    mood = fit_firm_mood(make_readings(train_documents, dates, centre), n_train_days, missing_days)
    print("mood parameters:", mood, flush=True)

    returns = returns_matrix(load_panel(), firms, dates)
    data = build_data(
        make_readings(documents, dates, centre), returns, firms, mood, missing_days, n_train_days
    )
    train = train_part(data, n_train_days)
    with_mood, likelihood = sv.fit_sv(mood, train, FIT_PARTICLES, SEED, with_mood=True)
    print("with mood:", with_mood, likelihood, flush=True)
    without_mood, likelihood_0 = sv.fit_sv(mood, train, FIT_PARTICLES, SEED, with_mood=False)
    print("without mood:", without_mood, likelihood_0, flush=True)

    # How noisy is the likelihood? The same two parameter sets under three other seeds.
    other_seeds = [SEED + 1, SEED + 2, SEED + 3]
    gains = [
        sv.sv_filter(mood, with_mood, train, FIT_PARTICLES, seed, False).log_likelihood
        - sv.sv_filter(mood, without_mood, train, FIT_PARTICLES, seed, False).log_likelihood
        for seed in other_seeds
    ]

    filtered = sv.sv_filter(mood, with_mood, data, FILTER_PARTICLES, SEED)
    filtered_0 = sv.sv_filter(mood, without_mood, data, FILTER_PARTICLES, SEED)
    fitted = {
        "tone_centre_train": centre,
        "mood": asdict(mood),
        "with_mood": asdict(with_mood),
        "without_mood": asdict(without_mood),
        "train_log_likelihood_gain_from_beta": likelihood - likelihood_0,
        "same_gain_under_three_other_seeds": gains,
        "fit_particles": FIT_PARTICLES,
        "seed": SEED,
        "filter": ess_summary(filtered, FILTER_PARTICLES),
    }
    MOOD_DIR.mkdir(parents=True, exist_ok=True)
    result_table(filtered, filtered_0, firms, dates).to_parquet(SV_PATH, index=False)
    SV_PARAMETERS_PATH.write_text(json.dumps(fitted, indent=2))
    print(json.dumps(fitted, indent=2))


if __name__ == "__main__":
    main()
