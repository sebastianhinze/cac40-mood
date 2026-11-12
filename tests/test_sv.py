"""Nonlinear mood model: the particle filter with a hidden mood and a hidden log-variance."""

import numpy as np
import pandas as pd
import pytest

from cac40_mood.mood import sv
from cac40_mood.mood.kalman import kalman_filter
from cac40_mood.mood.model import MoodParameters, make_observations

MOOD = MoodParameters(0.97, 0.05, 0.2)
TRUE = sv.SVParameters(0.95, 0.2, -0.15)


def test_documents_of_one_day_combine_into_one_reading_like_the_kalman_filter_would() -> None:
    readings = pd.DataFrame(
        {"day": [1, 1, 3], "firm": "A", "value": [0.4, -0.2, 0.1], "confidence": [1.0, 0.25, 0.5]}
    )
    mean, precision = sv.combine_readings(readings, ["A"], 5, 0.3, np.array([3]))
    # Day 1: variances 0.09 and 0.36, so precisions 11.11 and 2.78.
    assert precision[0, 1] == pytest.approx(1 / 0.09 + 1 / 0.36)
    assert mean[0, 1] == pytest.approx((0.4 / 0.09 - 0.2 / 0.36) / (1 / 0.09 + 1 / 0.36))
    assert precision[0, 0] == 0.0  # no document
    assert precision[0, 3] == 0.0  # a document, but on a day flagged missing


def test_without_returns_the_mood_matches_the_exact_kalman_filter() -> None:
    """Take the returns away: what is left is the linear mood model, which has an exact answer."""
    data, _, _ = sv.simulate_sv(MOOD, TRUE, n_firms=2, n_days=300, seed=2)
    blind = sv.SVData(
        data.firms,
        np.full_like(data.returns, np.nan),
        data.reading_mean,
        data.reading_precision,
        data.log_variance_level,
    )
    result = sv.sv_filter(MOOD, sv.SVParameters(0.95, 0.2, 0.0), blind, 20_000, seed=1)
    for row in range(2):
        days = np.flatnonzero(data.reading_precision[row] > 0)
        # One combined reading per day with its own variance: pass it as "extra variance".
        observations = make_observations(
            300,
            days,
            data.reading_mean[row, days],
            np.ones(len(days)),
            extra_variance=1.0 / data.reading_precision[row, days] - MOOD.sigma_obs**2,
        )
        exact = kalman_filter(MOOD, observations)
        assert np.max(np.abs(result.mood[row] - exact.mean)) < 0.02


def test_forecast_coefficients_match_a_brute_force_simulation() -> None:
    matrix, noise = sv.transition(MOOD, TRUE)
    weights, variances = sv.ahead_coefficients(matrix, noise, 5)
    rng = np.random.default_rng(0)
    state = np.tile(np.array([0.3, -0.5]), (200_000, 1))
    for step in range(5):
        state = state @ matrix.T + rng.normal(size=state.shape) * np.sqrt(np.diag(noise))
        assert state[:, 1].mean() == pytest.approx(weights[step] @ np.array([0.3, -0.5]), abs=0.003)
        assert state[:, 1].var() == pytest.approx(variances[step], rel=0.02)


def test_filter_tracks_the_hidden_log_variance_and_keeps_a_healthy_cloud() -> None:
    data, true_mood, true_h = sv.simulate_sv(MOOD, TRUE, n_firms=6, n_days=600, seed=1)
    result = sv.sv_filter(MOOD, TRUE, data, 2_000, seed=3)
    centred_estimate = result.log_variance - result.log_variance.mean(axis=1, keepdims=True)
    centred_truth = true_h - true_h.mean(axis=1, keepdims=True)
    assert np.corrcoef(centred_estimate.ravel(), centred_truth.ravel())[0, 1] > 0.7
    assert np.corrcoef(result.mood.ravel(), true_mood.ravel())[0, 1] > 0.6
    assert np.percentile(result.effective_sample_size, 5) > 400
    # The 5-day forecast is a variance forecast: it must go with the squared returns that follow.
    future = pd.DataFrame(data.returns.T**2).rolling(5).mean().shift(-5).to_numpy().T
    known = ~np.isnan(future)
    forecast = np.exp(result.log_forecast[5])
    assert np.corrcoef(forecast[known], future[known])[0, 1] > 0.3


def test_likelihood_prefers_the_true_mood_effect_and_estimation_finds_its_sign() -> None:
    data, _, _ = sv.simulate_sv(MOOD, TRUE, n_firms=12, n_days=800, seed=1)
    at_truth = sv.sv_filter(MOOD, TRUE, data, 1_000, 3, False).log_likelihood
    at_zero = sv.sv_filter(MOOD, sv.SVParameters(0.95, 0.2, 0.0), data, 1_000, 3, False)
    assert at_truth - at_zero.log_likelihood > 20
    fitted, _ = sv.fit_sv(MOOD, data, 300, 3, with_mood=True, start_betas=(-0.2, 0.2))
    assert -0.3 < fitted.beta < -0.05
    assert fitted.phi_h == pytest.approx(0.95, abs=0.04)


def test_later_data_changes_nothing_earlier() -> None:
    data, _, _ = sv.simulate_sv(MOOD, TRUE, n_firms=2, n_days=120, seed=5)
    changed_returns = data.returns.copy()
    changed_returns[:, 80:] *= 3.0
    changed = sv.SVData(
        data.firms,
        changed_returns,
        data.reading_mean,
        data.reading_precision,
        data.log_variance_level,
    )
    first = sv.sv_filter(MOOD, TRUE, data, 500, seed=1)
    second = sv.sv_filter(MOOD, TRUE, changed, 500, seed=1)
    np.testing.assert_allclose(first.log_forecast[5][:, :80], second.log_forecast[5][:, :80])
    np.testing.assert_allclose(first.mood[:, :80], second.mood[:, :80])
    assert not np.allclose(first.log_forecast[5][:, 80:], second.log_forecast[5][:, 80:])
