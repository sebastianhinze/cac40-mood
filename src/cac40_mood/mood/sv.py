"""Nonlinear mood model: the mood feeds into the (hidden) log-variance of returns.

Per firm, two hidden numbers per trading day:

    mood          m_t = phi_m * m_{t-1} + sigma_m * shock
    log-variance  h_t = mu + phi_h * (h_{t-1} - mu) + beta * m_{t-1} + sigma_h * shock

and two kinds of data:

    a scored document:  score = m_t + error          (as in the linear mood model)
    the day's return:   r_t ~ Normal(0, exp(h_t))    (stochastic volatility)

A negative beta means "bad mood, more volatility ahead". The return depends on h through
exp(), so the model is not linear and the Kalman filter cannot be used: this is the case the
particle filter exists for. Each particle is one guess of the pair (m, h).

The filter runs all firms at once (arrays of shape firms x particles) only for speed; firms do
not interact. Everything is causal: the numbers stored for day t use data up to day t.

Pre-registration of how this model is tested: DECISIONS #58.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cac40_mood.mood.model import MoodParameters, reading_scale
from cac40_mood.mood.particle import RESAMPLE_BELOW

HORIZONS: tuple[int, ...] = (1, 5, 22)


@dataclass(frozen=True)
class SVParameters:
    """phi_h: persistence of the log-variance. sigma_h: its daily shock. beta: effect of
    yesterday's mood on today's log-variance."""

    phi_h: float
    sigma_h: float
    beta: float


@dataclass(frozen=True)
class SVData:
    """What the filter reads, as arrays of shape (firms, days).

    `reading_mean` and `reading_precision` summarise the day's documents (see
    `combine_readings`); precision 0 means no document. `returns` may contain NaN.
    `log_variance_level` is, per firm, the log of the average squared train return.
    """

    firms: list[str]
    returns: np.ndarray
    reading_mean: np.ndarray
    reading_precision: np.ndarray
    log_variance_level: np.ndarray


@dataclass(frozen=True)
class SVResult:
    """Per firm and day: filtered mood, filtered log-variance, effective sample size, whether
    the cloud was resampled, and the log of the forecast average variance over the next
    1, 5 and 22 days. `log_likelihood` is the total over firms (up to a constant)."""

    mood: np.ndarray
    log_variance: np.ndarray
    effective_sample_size: np.ndarray
    resampled: np.ndarray
    log_forecast: dict[int, np.ndarray]
    log_likelihood: float


def combine_readings(
    readings: pd.DataFrame,
    firms: list[str],
    n_days: int,
    sigma_obs: float,
    missing_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarise each firm-day's documents as one reading: (mean, precision).

    Several normal readings of the same mood are worth exactly one reading whose precision
    (1 / variance) is the sum of the precisions and whose value is the precision-weighted
    average. Days flagged missing get precision 0, as do days without a document.
    """
    position = {firm: number for number, firm in enumerate(firms)}
    precision = np.zeros((len(firms), n_days))
    weighted_sum = np.zeros((len(firms), n_days))
    scales = reading_scale(sigma_obs, readings["confidence"].to_numpy())
    rows = readings["firm"].map(position).to_numpy()
    days = readings["day"].to_numpy()
    np.add.at(precision, (rows, days), 1.0 / scales**2)
    np.add.at(weighted_sum, (rows, days), readings["value"].to_numpy() / scales**2)
    precision[:, missing_days] = 0.0
    mean = np.divide(weighted_sum, precision, out=np.zeros_like(precision), where=precision > 0)
    return mean, precision


def transition(mood: MoodParameters, sv: SVParameters) -> tuple[np.ndarray, np.ndarray]:
    """The model as x_t = A x_{t-1} + noise with x = (m, h - mu): returns (A, noise covariance)."""
    matrix = np.array([[mood.phi, 0.0], [sv.beta, sv.phi_h]])
    noise = np.diag([mood.sigma_state**2, sv.sigma_h**2])
    return matrix, noise


def long_run_covariance(matrix: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Covariance of (m, h - mu) after the model has run for a long time.

    Found by repeating "one more day": P <- A P A' + Q, until it stops changing.
    """
    covariance = noise.copy()
    for _ in range(20_000):
        updated = matrix @ covariance @ matrix.T + noise
        if np.max(np.abs(updated - covariance)) < 1e-12:
            break
        covariance = updated
    return covariance


def ahead_coefficients(
    matrix: np.ndarray, noise: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """For j = 1..steps days ahead: h_{t+j} - mu given today's (m, h - mu) is normal with mean
    weights[j-1] @ (m, h - mu) and variance variances[j-1].

    One day ahead the mean is the second row of A applied to today's state and the variance is
    sigma_h^2; each further day applies A once more and adds one more day's noise.
    """
    weights, variances = np.empty((steps, 2)), np.empty(steps)
    power, accumulated = np.eye(2), np.zeros((2, 2))
    for step in range(steps):
        accumulated = matrix @ accumulated @ matrix.T + noise
        power = matrix @ power
        weights[step], variances[step] = power[1], accumulated[1, 1]
    return weights, variances


def resample_rows(
    weights: np.ndarray, uniforms: np.ndarray, rows: np.ndarray
) -> dict[int, np.ndarray]:
    """Systematic resampling for the firms in `rows`: {firm row: particle indices to keep}."""
    count = weights.shape[1]
    chosen = {}
    for row in rows:
        positions = (uniforms[row] + np.arange(count)) / count
        cumulative = np.cumsum(weights[row])
        chosen[int(row)] = np.minimum(np.searchsorted(cumulative, positions), count - 1)
    return chosen


def sv_filter(
    mood: MoodParameters,
    sv: SVParameters,
    data: SVData,
    n_particles: int,
    seed: int,
    with_forecasts: bool = True,
) -> SVResult:
    """Bootstrap particle filter for all firms. The same seed uses the same random numbers
    whatever the parameters (needed to compare likelihoods across parameter values)."""
    rng = np.random.default_rng(seed)
    n_firms, n_days = data.returns.shape
    matrix, noise = transition(mood, sv)
    covariance = long_run_covariance(matrix, noise)
    # Level chosen so that the model's average variance, exp(mu + var(h)/2), equals the
    # firm's average squared return in the train period.
    mu = (data.log_variance_level - 0.5 * covariance[1, 1])[:, None]
    ahead = {h: ahead_coefficients(matrix, noise, h) for h in HORIZONS} if with_forecasts else {}

    start = rng.normal(size=(n_firms, n_particles, 2)) @ np.linalg.cholesky(covariance).T
    m, h = start[:, :, 0], start[:, :, 1]  # h is kept as the deviation from mu
    log_weights = np.full((n_firms, n_particles), -np.log(n_particles))

    shape = (n_firms, n_days)
    mood_mean, h_mean, effective = np.empty(shape), np.empty(shape), np.empty(shape)
    resampled = np.zeros(shape, dtype=bool)
    log_forecast = {horizon: np.empty(shape) for horizon in ahead}
    log_likelihood = 0.0

    for day in range(n_days):
        shocks = rng.normal(size=(n_firms, n_particles, 2))
        uniforms = rng.random(n_firms)
        if day > 0:  # 1. move: h uses yesterday's mood, so update h first
            h = sv.phi_h * h + sv.beta * m + sv.sigma_h * shocks[:, :, 1]
            m = mood.phi * m + mood.sigma_state * shocks[:, :, 0]

        # 2. weigh by the day's documents and by the day's return
        precision = data.reading_precision[:, day, None]
        log_weights = log_weights - 0.5 * precision * (data.reading_mean[:, day, None] - m) ** 2
        returns = data.returns[:, day, None]
        seen = ~np.isnan(returns)
        level = mu + h
        return_term = -0.5 * (level + np.where(seen, returns, 0.0) ** 2 * np.exp(-level))
        log_weights = log_weights + np.where(seen, return_term, 0.0)

        largest = log_weights.max(axis=1, keepdims=True)
        step = largest + np.log(np.exp(log_weights - largest).sum(axis=1, keepdims=True))
        log_likelihood += float(step.sum())
        log_weights = log_weights - step
        weights = np.exp(log_weights)

        mood_mean[:, day] = (weights * m).sum(axis=1)
        h_mean[:, day] = (weights * level).sum(axis=1)
        effective[:, day] = 1.0 / (weights**2).sum(axis=1)
        for horizon, (ahead_weights, ahead_variances) in ahead.items():
            # Expected variance j days ahead for one particle: exp(mean + variance / 2).
            expected = np.zeros_like(m)
            for j in range(horizon):
                mean_j = mu + ahead_weights[j, 0] * m + ahead_weights[j, 1] * h
                expected += np.exp(mean_j + 0.5 * ahead_variances[j])
            log_forecast[horizon][:, day] = np.log((weights * expected).sum(axis=1) / horizon)

        # 3. resample the firms whose weights have become too uneven
        rows = np.flatnonzero(effective[:, day] < RESAMPLE_BELOW * n_particles)
        for row, keep in resample_rows(weights, uniforms, rows).items():
            m[row], h[row] = m[row, keep], h[row, keep]
            log_weights[row] = -np.log(n_particles)
        resampled[rows, day] = True

    return SVResult(mood_mean, h_mean, effective, resampled, log_forecast, log_likelihood)


START_PHI_H: float = 0.95
START_SIGMA_H: float = 0.2
START_BETAS: tuple[float, ...] = (-0.2, 0.0, 0.2)


def fit_sv(
    mood: MoodParameters,
    data: SVData,
    n_particles: int,
    seed: int,
    with_mood: bool,
    start_betas: tuple[float, ...] = START_BETAS,
) -> tuple[SVParameters, float]:
    """Maximum likelihood for (phi_h, sigma_h, beta), or with beta fixed at 0 if not `with_mood`.

    The particle filter's likelihood is noisy; using the same seed for every parameter value
    makes the noise largely cancel when two values are compared. The search is started from
    several values of beta (negative, zero, positive) and the best end point is kept, because
    a search started at zero can stay there: a larger sigma_h imitates part of a mood effect.
    Returns the parameters and the log-likelihood at the optimum.
    """

    def unpack(vector: np.ndarray) -> SVParameters:
        phi_h = 1.0 / (1.0 + np.exp(-vector[0]))
        beta = float(vector[2]) if with_mood else 0.0
        return SVParameters(float(phi_h), float(np.exp(vector[1])), beta)

    def negative_log_likelihood(vector: np.ndarray) -> float:
        return -sv_filter(mood, unpack(vector), data, n_particles, seed, False).log_likelihood

    best_parameters, best_value = None, np.inf
    for beta in start_betas if with_mood else (0.0,):
        first = [np.log(START_PHI_H / (1.0 - START_PHI_H)), np.log(START_SIGMA_H)]
        if with_mood:
            first.append(beta)
        result = minimize(
            negative_log_likelihood,
            np.array(first),
            method="Nelder-Mead",
            options={"xatol": 1e-3, "fatol": 0.05, "maxiter": 150},
        )
        if result.fun < best_value:
            best_parameters, best_value = unpack(result.x), float(result.fun)
    return best_parameters, -best_value


def simulate_sv(
    mood: MoodParameters,
    sv: SVParameters,
    n_firms: int,
    n_days: int,
    seed: int,
    readings_per_day: float = 0.2,
    level: float = -8.5,
) -> tuple[SVData, np.ndarray, np.ndarray]:
    """Simulate (data, true mood, true log-variance) for validation of the filter."""
    rng = np.random.default_rng(seed)
    matrix, noise = transition(mood, sv)
    covariance = long_run_covariance(matrix, noise)
    state = rng.normal(size=(n_firms, 2)) @ np.linalg.cholesky(covariance).T
    true_mood, true_h = np.empty((n_firms, n_days)), np.empty((n_firms, n_days))
    for day in range(n_days):
        if day > 0:
            shocks = rng.normal(size=(n_firms, 2)) * np.sqrt(np.diag(noise))
            state = state @ matrix.T + shocks
        true_mood[:, day], true_h[:, day] = state[:, 0], level + state[:, 1]
    returns = rng.normal(size=(n_firms, n_days)) * np.exp(true_h / 2.0)
    firms = [f"F{number:02d}" for number in range(n_firms)]
    pieces = []
    for number, firm in enumerate(firms):
        day = np.repeat(np.arange(n_days), rng.poisson(readings_per_day, n_days))
        confidence = rng.uniform(0.3, 1.0, len(day))
        errors = rng.normal(size=len(day)) * reading_scale(mood.sigma_obs, confidence)
        value = true_mood[number, day] + errors
        pieces.append(
            pd.DataFrame({"day": day, "firm": firm, "value": value, "confidence": confidence})
        )
    readings = pd.concat(pieces, ignore_index=True)
    mean, precision = combine_readings(readings, firms, n_days, mood.sigma_obs, np.array([], int))
    log_level = np.log((returns**2).mean(axis=1))
    return SVData(firms, returns, mean, precision, log_level), true_mood, true_h
