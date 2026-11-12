"""The mood model in one place: parameters, observations, and the densities both filters use.

Hidden state (one number per trading day): the mood m_t follows an AR(1),

    m_t = phi * m_{t-1} + sigma_state * noise_t,        noise_t ~ N(0, 1).

With 0 < phi < 1 the mood drifts back towards 0 and its long-run variance is
sigma_state^2 / (1 - phi^2).

Observations arrive irregularly: a scored document k on day t is a noisy reading of the mood,

    y_k = m_t + error_k,     scale of error_k = sigma_obs / sqrt(confidence_k).

A confident score (confidence 1) has the base noise; a score with confidence 0.25 has twice
the noise. Days without a document simply have no observation. With `dof` = None the error is
normal (the linear-Gaussian model, where the Kalman filter is exact). With a finite `dof` the
error is Student-t: heavy tails, so a single wild score moves the mood much less. That robust
version has no exact filter, which is why a particle filter is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

MIN_CONFIDENCE: float = 0.05  # keeps the error scale finite for a confidence of 0


@dataclass(frozen=True)
class MoodParameters:
    """phi: persistence (0..1). sigma_state: daily shock. sigma_obs: base reading noise.
    dof: Student-t degrees of freedom of the reading error, or None for a normal error."""

    phi: float
    sigma_state: float
    sigma_obs: float
    dof: float | None = None

    @property
    def stationary_variance(self) -> float:
        return self.sigma_state**2 / (1.0 - self.phi**2)


@dataclass(frozen=True)
class Observations:
    """All readings of one mood series, sorted by day.

    `n_days` is the length of the daily grid (0 .. n_days-1). `day`, `value`, `confidence` and
    `extra_variance` have one entry per reading; several readings may share a day and most days
    have none. `first_of_day[t] : first_of_day[t+1]` are the positions of day t's readings.

    `extra_variance` is noise in a reading that does not come from the scorer (default 0). The
    market/firm filter uses it: seen from the market mood, a firm's own mood is extra noise.
    """

    n_days: int
    day: np.ndarray
    value: np.ndarray
    confidence: np.ndarray
    extra_variance: np.ndarray
    first_of_day: np.ndarray

    def of_day(self, day: int, sigma_obs: float) -> tuple[np.ndarray, np.ndarray]:
        """(values, error scales) of the readings on `day` (empty arrays if there are none)."""
        start, stop = self.first_of_day[day], self.first_of_day[day + 1]
        scales = reading_scale(
            sigma_obs, self.confidence[start:stop], self.extra_variance[start:stop]
        )
        return self.value[start:stop], scales


def reading_scale(
    sigma_obs: float, confidence: np.ndarray, extra_variance: np.ndarray | float = 0.0
) -> np.ndarray:
    """Error scale of each reading: sqrt(sigma_obs^2 / confidence + extra variance)."""
    scorer_variance = sigma_obs**2 / np.clip(confidence, MIN_CONFIDENCE, 1.0)
    return np.sqrt(scorer_variance + extra_variance)


def make_observations(
    n_days: int,
    day: np.ndarray,
    value: np.ndarray,
    confidence: np.ndarray,
    missing_days: np.ndarray | None = None,
    extra_variance: np.ndarray | None = None,
) -> Observations:
    """Sort the readings by day and drop those on days flagged as missing.

    `missing_days` marks days on which the data feed is known to be broken (the spring-2020
    hole): any reading there is discarded, so the filter only predicts through the window and
    its uncertainty grows. A day without a reading is treated the same way by the filter;
    the difference matters for features built from counts, not for the mood itself.
    """
    day = np.asarray(day, dtype=int)
    extra = (
        np.zeros(len(day)) if extra_variance is None else np.asarray(extra_variance, dtype=float)
    )
    keep = np.ones(len(day), dtype=bool)
    if missing_days is not None:
        keep = ~np.isin(day, missing_days)
    order = np.argsort(day[keep], kind="stable")
    sorted_day = day[keep][order]
    return Observations(
        n_days,
        sorted_day,
        np.asarray(value, dtype=float)[keep][order],
        np.asarray(confidence, dtype=float)[keep][order],
        extra[keep][order],
        np.searchsorted(sorted_day, np.arange(n_days + 1)),
    )


def log_reading_density(
    value: float, scale: float, mood: np.ndarray, dof: float | None
) -> np.ndarray:
    """log p(reading | mood) for every candidate mood: normal, or Student-t if `dof` is set."""
    standardised = (value - mood) / scale
    if dof is None:
        return -0.5 * np.log(2.0 * np.pi * scale**2) - 0.5 * standardised**2
    constant = (
        gammaln((dof + 1.0) / 2.0) - gammaln(dof / 2.0) - 0.5 * np.log(dof * np.pi * scale**2)
    )
    return constant - 0.5 * (dof + 1.0) * np.log1p(standardised**2 / dof)
