"""Market mood plus firm-specific mood, filtered in two stages (DECISIONS #44).

Model: a reading for firm i on day t is  y = g_t + u_{i,t} + error, where g is a market-wide
mood shared by all firms and u_i is the firm's own deviation; both are AR(1).

Stage 1, market: every reading of every firm is treated as a reading of g. Seen from the
market, the firm's own mood is just more noise, so each reading's error variance is increased
by the long-run variance of u.

Stage 2, firm: for each firm, subtract the filtered market mood from its readings and filter
what is left as u_i. The market mood is itself uncertain, so its filtered variance on that day
is added to the reading's error variance.

The firm's total mood is then (market mean + firm mean), with the two variances added.

Why two stages instead of one filter over all 37 states: each stage is the simple scalar
filter that is validated against the Kalman filter, 36 firms cost 36 small runs, and each
piece can be explained on its own. The price: stage 1 ignores that a firm's deviation is
persistent, so the market mood is a little less sharp than a joint filter would make it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cac40_mood.mood.model import MoodParameters, make_observations
from cac40_mood.mood.particle import ParticleResult, particle_filter


@dataclass(frozen=True)
class HierarchyResult:
    """Filtered market mood and, per firm, the firm part and the total mood."""

    market: ParticleResult
    firm_part: dict[str, ParticleResult]
    total_mean: pd.DataFrame  # index: day, columns: firms
    total_variance: pd.DataFrame


def two_stage_filter(
    readings: pd.DataFrame,
    n_days: int,
    market_parameters: MoodParameters,
    firm_parameters: MoodParameters,
    n_particles: int,
    seed: int,
    missing_days: np.ndarray | None = None,
) -> HierarchyResult:
    """Filter market and firm moods from `readings` (columns: day, firm, value, confidence)."""
    market_observations = make_observations(
        n_days,
        readings["day"].to_numpy(),
        readings["value"].to_numpy(),
        readings["confidence"].to_numpy(),
        missing_days,
        extra_variance=np.full(len(readings), firm_parameters.stationary_variance),
    )
    market = particle_filter(market_parameters, market_observations, n_particles, seed)

    firm_part: dict[str, ParticleResult] = {}
    for number, (firm, rows) in enumerate(readings.groupby("firm"), start=1):
        days = rows["day"].to_numpy()
        firm_observations = make_observations(
            n_days,
            days,
            rows["value"].to_numpy() - market.mean[days],
            rows["confidence"].to_numpy(),
            missing_days,
            extra_variance=market.variance[days],
        )
        firm_part[str(firm)] = particle_filter(
            firm_parameters, firm_observations, n_particles, seed + number
        )

    total_mean = pd.DataFrame({firm: market.mean + part.mean for firm, part in firm_part.items()})
    total_variance = pd.DataFrame(
        {firm: market.variance + part.variance for firm, part in firm_part.items()}
    )
    return HierarchyResult(market, firm_part, total_mean, total_variance)


def simulate_hierarchy(
    market_parameters: MoodParameters,
    firm_parameters: MoodParameters,
    n_firms: int,
    n_days: int,
    seed: int,
    readings_per_day: float = 0.15,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Simulate (true market mood, true total mood per firm, readings) for validation."""
    from cac40_mood.mood.model import reading_scale
    from cac40_mood.mood.simulate import simulate_mood

    rng = np.random.default_rng(seed)
    market = simulate_mood(market_parameters, n_days, rng)
    totals, pieces = {}, []
    for number in range(n_firms):
        firm = f"F{number:02d}"
        total = market + simulate_mood(firm_parameters, n_days, rng)
        totals[firm] = total
        day = np.repeat(np.arange(n_days), rng.poisson(readings_per_day, n_days))
        confidence = rng.uniform(0.3, 1.0, len(day))
        errors = rng.normal(0.0, 1.0, len(day)) * reading_scale(
            firm_parameters.sigma_obs, confidence
        )
        pieces.append(
            pd.DataFrame(
                {"day": day, "firm": firm, "value": total[day] + errors, "confidence": confidence}
            )
        )
    return market, pd.DataFrame(totals), pd.concat(pieces, ignore_index=True)
