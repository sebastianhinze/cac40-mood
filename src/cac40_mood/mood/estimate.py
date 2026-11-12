"""Estimate the mood model's parameters by maximum likelihood.

Two likelihoods are available:
- exact, from the Kalman filter, when reading errors are normal;
- estimated, from the particle filter, for Student-t errors. The particle estimate is noisy, so
  the same random seed is used for every parameter value tried ("common random numbers"):
  the noise then largely cancels when two parameter values are compared.

The optimiser works on transformed parameters so that every value it tries is valid:
phi = 1 / (1 + exp(-a)) lies in (0, 1); the two sigmas are exp(b) and exp(c), always positive.
Parameters are fitted on the train period only and then frozen.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from cac40_mood.mood.kalman import kalman_filter
from cac40_mood.mood.model import MoodParameters, Observations
from cac40_mood.mood.particle import particle_filter

DEFAULT_START = MoodParameters(phi=0.9, sigma_state=0.1, sigma_obs=0.3)


def to_vector(parameters: MoodParameters) -> np.ndarray:
    return np.array(
        [
            np.log(parameters.phi / (1.0 - parameters.phi)),
            np.log(parameters.sigma_state),
            np.log(parameters.sigma_obs),
        ]
    )


def from_vector(vector: np.ndarray, dof: float | None) -> MoodParameters:
    phi = 1.0 / (1.0 + np.exp(-vector[0]))
    return MoodParameters(float(phi), float(np.exp(vector[1])), float(np.exp(vector[2])), dof)


def fit_gaussian(
    series: list[Observations], start: MoodParameters = DEFAULT_START
) -> MoodParameters:
    """Exact maximum likelihood for normal errors. `series` may hold several independent
    mood series (for example one per firm) that share the same parameters."""

    def negative_log_likelihood(vector: np.ndarray) -> float:
        parameters = from_vector(vector, None)
        return -sum(
            kalman_filter(parameters, observations).log_likelihood for observations in series
        )

    result = minimize(
        negative_log_likelihood,
        to_vector(start),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000},
    )
    return from_vector(result.x, None)


def fit_particle(
    series: list[Observations],
    dof: float | None,
    n_particles: int,
    seed: int,
    start: MoodParameters = DEFAULT_START,
) -> MoodParameters:
    """Maximum likelihood with the particle filter's likelihood and common random numbers."""

    def negative_log_likelihood(vector: np.ndarray) -> float:
        parameters = from_vector(vector, dof)
        total = 0.0
        for number, observations in enumerate(series):
            result = particle_filter(parameters, observations, n_particles, seed + number, False)
            total += result.log_likelihood
        return -total

    result = minimize(
        negative_log_likelihood,
        to_vector(start),
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 400},
    )
    return from_vector(result.x, dof)
