"""Exact Kalman filter for the linear-Gaussian mood model: the reference the particle filter
must reproduce.

Each day has two steps.
Predict (no information yet):   mean -> phi * mean,   variance -> phi^2 * variance + sigma_state^2.
Update (once per reading y with error variance R):
    gain     = variance / (variance + R)          how much to trust the reading (0..1)
    mean     = mean + gain * (y - mean)           move towards the reading
    variance = (1 - gain) * variance              we are now more certain
A day without readings stops after the predict step, so the variance grows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cac40_mood.mood.model import MoodParameters, Observations


@dataclass(frozen=True)
class FilterResult:
    """Filtered mean and variance per day (after that day's readings) and the log-likelihood."""

    mean: np.ndarray
    variance: np.ndarray
    log_likelihood: float


def kalman_filter(parameters: MoodParameters, observations: Observations) -> FilterResult:
    """Run the filter over the daily grid. Day 0 starts from the long-run distribution."""
    if parameters.dof is not None:
        raise ValueError("The Kalman filter is exact only for normal reading errors (dof=None)")
    mean, variance = 0.0, parameters.stationary_variance
    means = np.empty(observations.n_days)
    variances = np.empty(observations.n_days)
    log_likelihood = 0.0
    for day in range(observations.n_days):
        if day > 0:
            mean = parameters.phi * mean
            variance = parameters.phi**2 * variance + parameters.sigma_state**2
        values, scales = observations.of_day(day, parameters.sigma_obs)
        for value, scale in zip(values, scales, strict=True):
            total = variance + scale**2  # variance of the reading before we see it
            log_likelihood += -0.5 * (np.log(2.0 * np.pi * total) + (value - mean) ** 2 / total)
            gain = variance / total
            mean = mean + gain * (value - mean)
            variance = (1.0 - gain) * variance
        means[day], variances[day] = mean, variance
    return FilterResult(means, variances, log_likelihood)
