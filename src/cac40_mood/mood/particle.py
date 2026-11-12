"""Bootstrap particle filter for the mood model, written from scratch in NumPy.

Idea: carry a cloud of N guesses ("particles") of today's mood.
1. Move: each particle takes one AR(1) step with its own random shock (the "bootstrap" choice:
   propose from the model's own dynamics).
2. Weigh: when a reading arrives, each particle's weight is multiplied by how likely that
   reading would be if the particle were the true mood. Several readings on one day multiply.
3. Resample when needed: if a few particles hold most of the weight (effective sample size
   below half of N), draw a new cloud in proportion to the weights and reset the weights.
The weighted average of the cloud is the filtered mood; its spread is the uncertainty.

On a day without readings only step 1 happens, so the cloud widens: uncertainty grows, exactly
like the predict-only step of the Kalman filter. The same code handles normal and Student-t
reading errors, because only the weighing function changes.

Weights are kept as logarithms so that multiplying many small densities cannot underflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cac40_mood.mood.model import MoodParameters, Observations, log_reading_density

RESAMPLE_BELOW: float = 0.5  # resample when the effective sample size drops below N/2


@dataclass(frozen=True)
class ParticleResult:
    """Per day: filtered mean and variance, a 90% band, the effective sample size; plus the
    estimated log-likelihood and the number of resampling steps."""

    mean: np.ndarray
    variance: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    effective_sample_size: np.ndarray
    log_likelihood: float
    resamplings: int


def log_sum_exp(values: np.ndarray) -> float:
    """log(sum(exp(values))) without overflow: factor out the largest term."""
    largest = values.max()
    return float(largest + np.log(np.exp(values - largest).sum()))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of the particles to keep, by systematic resampling.

    Lay the weights end to end on [0, 1), then read off N equally spaced points that share one
    random offset. A particle with weight w is picked floor(N*w) or ceil(N*w) times, which adds
    less randomness than drawing N independent indices.
    """
    count = len(weights)
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against rounding
    return np.searchsorted(cumulative, positions)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, probability * cumulative[-1])])


def particle_filter(
    parameters: MoodParameters,
    observations: Observations,
    n_particles: int,
    seed: int,
    with_bands: bool = True,
) -> ParticleResult:
    """Run the bootstrap filter. The same seed gives the same result (needed for estimation)."""
    rng = np.random.default_rng(seed)
    particles = rng.normal(0.0, np.sqrt(parameters.stationary_variance), n_particles)
    log_weights = np.full(n_particles, -np.log(n_particles))
    n_days = observations.n_days
    mean, variance = np.empty(n_days), np.empty(n_days)
    lower, upper = np.full(n_days, np.nan), np.full(n_days, np.nan)
    effective = np.empty(n_days)
    log_likelihood, resamplings = 0.0, 0

    for day in range(n_days):
        if day > 0:  # 1. move
            particles = parameters.phi * particles + parameters.sigma_state * rng.normal(
                size=n_particles
            )
        values, scales = observations.of_day(day, parameters.sigma_obs)
        for value, scale in zip(values, scales, strict=True):  # 2. weigh
            log_weights = log_weights + log_reading_density(value, scale, particles, parameters.dof)
            # The average weight is this reading's likelihood given everything before it.
            step = log_sum_exp(log_weights)
            log_likelihood += step
            log_weights = log_weights - step

        weights = np.exp(log_weights)
        mean[day] = weights @ particles
        variance[day] = weights @ (particles - mean[day]) ** 2
        if with_bands:
            lower[day] = weighted_quantile(particles, weights, 0.05)
            upper[day] = weighted_quantile(particles, weights, 0.95)
        effective[day] = 1.0 / (weights @ weights)

        if effective[day] < RESAMPLE_BELOW * n_particles:  # 3. resample
            particles = particles[systematic_resample(weights, rng)]
            log_weights = np.full(n_particles, -np.log(n_particles))
            resamplings += 1

    return ParticleResult(mean, variance, lower, upper, effective, log_likelihood, resamplings)
