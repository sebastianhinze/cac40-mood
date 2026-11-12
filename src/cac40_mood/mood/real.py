"""The mood model on real scores: readings, parameter estimation on train, filtering.

    uv run python -m cac40_mood.mood.real

Steps (the plan is section 10 of docs/mood_model.md):
1. Readings: one per scored document, value = tone minus the average train tone, confidence
   from the LLM, day = position of the event session in the trading calendar.
2. Parameters, on the **train period only**, by exact (Kalman) maximum likelihood with normal
   errors. Market and firm parameters depend on each other, so they are fitted in turn, twice.
3. Filter train and validation with the two-stage particle filter and those parameters.
   The filter is causal: the mood of day t uses readings up to day t only.

Why the tone is centred: the model's mood drifts back to zero when there is no news, but the
average document is mildly positive (companies publish more good news than bad). Without
centring, "no news" would look like "bad news". The average is a train-period number.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from cac40_mood.analysis_documents import load_documents
from cac40_mood.config import PROCESSED_DIR
from cac40_mood.data.build import SESSIONS_PATH
from cac40_mood.mood.estimate import fit_gaussian
from cac40_mood.mood.hierarchy import HierarchyResult, two_stage_filter
from cac40_mood.mood.kalman import kalman_filter
from cac40_mood.mood.model import MoodParameters, make_observations
from cac40_mood.splits import TRAIN_END, TRAIN_START, VAL_END

MOOD_DIR: Path = PROCESSED_DIR / "mood"
MOOD_PATH: Path = MOOD_DIR / "filtered_mood.parquet"
PARAMETERS_PATH: Path = MOOD_DIR / "parameters.json"

N_PARTICLES: int = 2_000
SEED: int = 5
FIT_ROUNDS: int = 2


def sample_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Trading sessions from the start of train to the end of validation, in order."""
    inside = sessions["date"].between(pd.Timestamp(TRAIN_START), pd.Timestamp(VAL_END))
    return sessions[inside].sort_values("date").reset_index(drop=True)


def make_readings(documents: pd.DataFrame, dates: pd.Series, centre: float) -> pd.DataFrame:
    """One reading per document: day number, firm, centred tone, confidence."""
    day = pd.DatetimeIndex(dates).get_indexer(documents["event_session"])
    if (day < 0).any():
        raise ValueError("a document's event session is not in the trading calendar")
    return pd.DataFrame(
        {
            "day": day,
            "firm": documents["firm_id"].to_numpy(),
            "value": documents["tone"].to_numpy() - centre,
            "confidence": documents["confidence"].to_numpy(),
        }
    )


def fit_two_stage(
    readings: pd.DataFrame, n_days: int, missing_days: np.ndarray, rounds: int = FIT_ROUNDS
) -> tuple[MoodParameters, MoodParameters]:
    """Estimate (market parameters, firm parameters) from `readings`.

    The two sets depend on each other: seen from the market, a firm's own mood is extra noise
    (its size depends on the firm parameters), and the firm readings are what is left after
    subtracting the filtered market mood. So we alternate:
    - fit the market on all readings, given the current guess of the firm noise;
    - filter the market, subtract it, fit the firm parameters on what is left (all firms share
      one set);
    and do that `rounds` times, starting from "no firm noise".
    """
    firm_variance = 0.0
    market_parameters = firm_parameters = None
    for _ in range(rounds):
        market_observations = make_observations(
            n_days,
            readings["day"].to_numpy(),
            readings["value"].to_numpy(),
            readings["confidence"].to_numpy(),
            missing_days,
            extra_variance=np.full(len(readings), firm_variance),
        )
        market_parameters = fit_gaussian([market_observations])
        market = kalman_filter(market_parameters, market_observations)

        firm_series = []
        for _, rows in readings.groupby("firm"):
            days = rows["day"].to_numpy()
            firm_series.append(
                make_observations(
                    n_days,
                    days,
                    rows["value"].to_numpy() - market.mean[days],
                    rows["confidence"].to_numpy(),
                    missing_days,
                    extra_variance=market.variance[days],
                )
            )
        firm_parameters = fit_gaussian(firm_series)
        firm_variance = firm_parameters.stationary_variance
    return market_parameters, firm_parameters


def mood_table(result: HierarchyResult, dates: pd.Series) -> pd.DataFrame:
    """Long table: firm_id, date, market mood, total mood of the firm and its standard deviation."""
    pieces = []
    for firm in result.total_mean.columns:
        pieces.append(
            pd.DataFrame(
                {
                    "firm_id": firm,
                    "date": dates.to_numpy(),
                    "mood_market": result.market.mean,
                    "mood_market_low": result.market.lower,
                    "mood_market_high": result.market.upper,
                    "mood_total": result.total_mean[firm].to_numpy(),
                    "mood_sd": np.sqrt(result.total_variance[firm].to_numpy()),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def run(documents: pd.DataFrame, sessions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit on train, filter train and validation. Returns (mood table, fitted numbers)."""
    calendar = sample_sessions(sessions)
    dates = calendar["date"]
    missing_days = np.flatnonzero(~calendar["feed_observed"].to_numpy())
    train_documents = documents[documents["split"] == "train"]
    centre = float(train_documents["tone"].mean())

    n_train_days = int((dates <= pd.Timestamp(TRAIN_END)).sum())
    train_readings = make_readings(train_documents, dates, centre)
    market_parameters, firm_parameters = fit_two_stage(train_readings, n_train_days, missing_days)

    readings = make_readings(documents, dates, centre)
    result = two_stage_filter(
        readings, len(dates), market_parameters, firm_parameters, N_PARTICLES, SEED, missing_days
    )
    fitted = {
        "tone_centre_train": centre,
        "market": asdict(market_parameters),
        "firm": asdict(firm_parameters),
        "train_readings": len(train_readings),
        "all_readings": len(readings),
        "particles": N_PARTICLES,
        "seed": SEED,
    }
    return mood_table(result, dates), fitted


def main() -> None:
    documents, _ = load_documents()
    table, fitted = run(documents, pd.read_parquet(SESSIONS_PATH))
    MOOD_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(MOOD_PATH, index=False)
    PARAMETERS_PATH.write_text(json.dumps(fitted, indent=2))
    print(json.dumps(fitted, indent=2))


if __name__ == "__main__":
    main()
