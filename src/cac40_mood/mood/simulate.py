"""Simulated data with a known hidden mood, used to validate the filters before real data.

The simulation mirrors what the real data look like: most days have no document, some days
have several, confidence varies, a few readings can be wild (outliers), and there can be a
window in which the feed is broken and every reading is lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cac40_mood.mood.model import MoodParameters, Observations, make_observations, reading_scale


@dataclass(frozen=True)
class Simulation:
    """The true mood path and the readings generated from it."""

    mood: np.ndarray
    observations: Observations
    day: np.ndarray
    value: np.ndarray
    confidence: np.ndarray


def simulate_mood(parameters: MoodParameters, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """One AR(1) path that starts from the long-run distribution."""
    mood = np.empty(n_days)
    mood[0] = rng.normal(0.0, np.sqrt(parameters.stationary_variance))
    shocks = rng.normal(0.0, parameters.sigma_state, n_days)
    for day in range(1, n_days):
        mood[day] = parameters.phi * mood[day - 1] + shocks[day]
    return mood


def simulate(
    parameters: MoodParameters,
    n_days: int,
    seed: int,
    readings_per_day: float = 0.15,
    outlier_share: float = 0.0,
    missing_days: np.ndarray | None = None,
) -> Simulation:
    """Simulate a mood path and irregular readings of it.

    The number of readings per day is Poisson with mean `readings_per_day` (so most days have
    none and some have two or more). A share `outlier_share` of readings is replaced by pure
    noise five times the base scale. Readings on `missing_days` are generated and then dropped
    by `make_observations`, like filings that never reached the feed.
    """
    rng = np.random.default_rng(seed)
    mood = simulate_mood(parameters, n_days, rng)
    counts = rng.poisson(readings_per_day, n_days)
    day = np.repeat(np.arange(n_days), counts)
    confidence = rng.uniform(0.3, 1.0, len(day))
    scale = reading_scale(parameters.sigma_obs, confidence)
    if parameters.dof is None:
        errors = rng.normal(0.0, 1.0, len(day)) * scale
    else:
        errors = rng.standard_t(parameters.dof, len(day)) * scale
    value = mood[day] + errors
    outliers = rng.random(len(day)) < outlier_share
    value[outliers] = rng.normal(0.0, 5.0 * parameters.sigma_obs, int(outliers.sum()))
    observations = make_observations(n_days, day, value, confidence, missing_days)
    return Simulation(mood, observations, day, value, confidence)
