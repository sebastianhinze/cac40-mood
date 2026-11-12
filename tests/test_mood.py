"""Mood model: Kalman reference, particle filter, irregular readings, gaps, robustness.

Everything here runs on simulated data where the true hidden mood is known.
"""

import numpy as np
import pytest
from scipy import stats

from cac40_mood.mood import estimate, hierarchy
from cac40_mood.mood.kalman import kalman_filter
from cac40_mood.mood.model import (
    MoodParameters,
    log_reading_density,
    make_observations,
    reading_scale,
)
from cac40_mood.mood.particle import (
    log_sum_exp,
    particle_filter,
    systematic_resample,
    weighted_quantile,
)
from cac40_mood.mood.simulate import simulate

TRUE = MoodParameters(phi=0.97, sigma_state=0.08, sigma_obs=0.3)


def readings(n_days, days, values, confidences, missing=None):
    return make_observations(
        n_days, np.array(days), np.array(values, dtype=float), np.array(confidences), missing
    )


# -------------------------------------------------------------------- model


def test_reading_density_matches_scipy() -> None:
    mood = np.array([-0.5, 0.0, 0.7])
    normal = log_reading_density(0.2, 0.3, mood, None)
    assert normal == pytest.approx(stats.norm.logpdf(0.2, loc=mood, scale=0.3))
    student = log_reading_density(0.2, 0.3, mood, 4.0)
    assert student == pytest.approx(stats.t.logpdf(0.2, df=4.0, loc=mood, scale=0.3))


def test_low_confidence_means_a_noisier_reading() -> None:
    scales = reading_scale(0.3, np.array([1.0, 0.25, 0.0]))
    assert scales[0] == pytest.approx(0.3)
    assert scales[1] == pytest.approx(0.6)  # a quarter of the confidence: twice the noise
    assert np.isfinite(scales[2])  # confidence 0 is clipped, not a division by zero
    assert reading_scale(0.3, np.array([1.0]), 0.16)[0] == pytest.approx(0.5)  # sqrt(0.09+0.16)


def test_observations_are_sorted_and_missing_days_are_dropped() -> None:
    observations = readings(
        6, [4, 1, 4, 2], [0.4, 0.1, 0.5, 0.2], [1, 1, 1, 1], missing=np.array([2])
    )
    assert observations.day.tolist() == [1, 4, 4]
    values, _ = observations.of_day(4, 0.3)
    assert values.tolist() == [0.4, 0.5]
    assert len(observations.of_day(2, 0.3)[0]) == 0 and len(observations.of_day(0, 0.3)[0]) == 0


# ------------------------------------------------------------------- Kalman


def test_kalman_filter_matches_the_worked_example_in_the_docs() -> None:
    """Three days by hand (docs/mood_model.md): a reading, a silent day, a weak reading."""
    parameters = MoodParameters(phi=0.9, sigma_state=0.2, sigma_obs=0.3)
    result = kalman_filter(parameters, readings(3, [0, 2], [0.5, -0.1], [1.0, 0.25]))

    variance = 0.2**2 / (1 - 0.9**2)  # long-run variance: 0.2105
    gain = variance / (variance + 0.3**2)
    mean_0, variance_0 = gain * 0.5, (1 - gain) * variance
    mean_1, variance_1 = 0.9 * mean_0, 0.81 * variance_0 + 0.04  # no reading: predict only
    predicted_mean, predicted_variance = 0.9 * mean_1, 0.81 * variance_1 + 0.04
    gain_2 = predicted_variance / (predicted_variance + 0.3**2 / 0.25)
    mean_2 = predicted_mean + gain_2 * (-0.1 - predicted_mean)
    variance_2 = (1 - gain_2) * predicted_variance

    assert result.mean == pytest.approx([mean_0, mean_1, mean_2])
    assert result.variance == pytest.approx([variance_0, variance_1, variance_2])
    assert result.mean == pytest.approx([0.3503, 0.3152, 0.1916], abs=5e-4)
    assert result.variance == pytest.approx([0.0630, 0.0911, 0.0864], abs=5e-4)
    assert variance_1 > variance_0  # a day without news makes us less sure


def test_two_readings_on_one_day_do_not_depend_on_their_order() -> None:
    first = kalman_filter(TRUE, readings(2, [1, 1], [0.4, -0.2], [1.0, 0.5]))
    second = kalman_filter(TRUE, readings(2, [1, 1], [-0.2, 0.4], [0.5, 1.0]))
    assert first.mean == pytest.approx(second.mean)
    assert first.log_likelihood == pytest.approx(second.log_likelihood)


def test_kalman_refuses_student_t_errors() -> None:
    with pytest.raises(ValueError, match="normal"):
        kalman_filter(MoodParameters(0.9, 0.1, 0.3, dof=4.0), readings(2, [0], [0.1], [1.0]))


# ---------------------------------------------------------- particle filter


def test_particle_filter_matches_the_kalman_filter_in_the_linear_gaussian_case() -> None:
    simulation = simulate(TRUE, 400, seed=1)
    exact = kalman_filter(TRUE, simulation.observations)
    approx = particle_filter(TRUE, simulation.observations, 20_000, seed=2)
    assert np.abs(approx.mean - exact.mean).max() < 0.02
    assert approx.variance == pytest.approx(exact.variance, rel=0.08)
    assert approx.log_likelihood == pytest.approx(exact.log_likelihood, abs=0.3)


def test_log_likelihood_is_stable_across_seeds() -> None:
    simulation = simulate(TRUE, 400, seed=1)
    exact = kalman_filter(TRUE, simulation.observations).log_likelihood
    estimates = [
        particle_filter(TRUE, simulation.observations, 2_000, seed, with_bands=False).log_likelihood
        for seed in range(12)
    ]
    assert np.std(estimates) < 0.25
    assert np.mean(estimates) == pytest.approx(exact, abs=0.3)


def test_same_seed_same_result_different_seed_different_result() -> None:
    simulation = simulate(TRUE, 200, seed=1)
    first = particle_filter(TRUE, simulation.observations, 500, seed=7)
    again = particle_filter(TRUE, simulation.observations, 500, seed=7)
    other = particle_filter(TRUE, simulation.observations, 500, seed=8)
    assert np.array_equal(first.mean, again.mean) and first.log_likelihood == again.log_likelihood
    assert not np.array_equal(first.mean, other.mean)


def test_days_without_readings_only_predict_and_uncertainty_grows() -> None:
    """One reading on day 0, then silence: the mean decays by phi and the variance rises."""
    observations = readings(30, [0], [0.6], [1.0])
    exact = kalman_filter(TRUE, observations)
    approx = particle_filter(TRUE, observations, 50_000, seed=3)
    assert exact.mean[1:] == pytest.approx(exact.mean[0] * TRUE.phi ** np.arange(1, 30))
    assert np.all(np.diff(exact.variance) > 0)
    assert approx.mean == pytest.approx(exact.mean, abs=0.01)
    assert approx.variance == pytest.approx(exact.variance, rel=0.05)
    assert approx.resamplings <= 1  # only the single reading on day 0 can trigger one


def test_missing_data_gap_widens_the_band_until_readings_return() -> None:
    """A broken feed from day 100 to 169: readings there are dropped, not treated as news."""
    gap = np.arange(100, 170)
    with_gap = simulate(TRUE, 300, seed=5, readings_per_day=0.5, missing_days=gap)
    without_gap = simulate(TRUE, 300, seed=5, readings_per_day=0.5)
    assert not np.isin(with_gap.observations.day, gap).any()
    assert np.isin(without_gap.observations.day, gap).any()

    exact = kalman_filter(TRUE, with_gap.observations)
    approx = particle_filter(TRUE, with_gap.observations, 20_000, seed=6)
    inside = exact.variance[100:170]
    assert np.all(np.diff(inside) > 0)  # uncertainty grows every day of the gap
    assert inside[-1] > 3 * exact.variance[99]
    assert inside[-1] < TRUE.stationary_variance  # ... towards the long-run level, never past it
    assert approx.variance[100:170] == pytest.approx(inside, rel=0.08)
    band_before = approx.upper[99] - approx.lower[99]
    band_at_end = approx.upper[169] - approx.lower[169]
    assert band_at_end > 1.5 * band_before
    first_reading_after = with_gap.observations.day[with_gap.observations.day >= 170][0]
    assert exact.variance[first_reading_after] < exact.variance[first_reading_after - 1]


def test_band_covers_the_truth_about_ninety_percent_of_the_time() -> None:
    covered = []
    for seed in range(8):
        simulation = simulate(TRUE, 500, seed=seed)
        result = particle_filter(TRUE, simulation.observations, 2_000, seed=100 + seed)
        covered.append(
            np.mean((simulation.mood >= result.lower) & (simulation.mood <= result.upper))
        )
    assert 0.85 < np.mean(covered) < 0.95


def test_systematic_resampling_keeps_each_particle_about_n_times_its_weight() -> None:
    rng = np.random.default_rng(0)
    weights = rng.random(1000)
    weights /= weights.sum()
    indices = systematic_resample(weights, rng)
    counts = np.bincount(indices, minlength=1000)
    expected = 1000 * weights
    assert len(indices) == 1000 and np.all(np.diff(indices) >= 0)
    assert np.all(counts >= np.floor(expected)) and np.all(counts <= np.ceil(expected))
    assert systematic_resample(np.full(50, 1 / 50), rng).tolist() == list(range(50))


def test_helpers() -> None:
    assert log_sum_exp(np.array([-1000.0, -1000.0])) == pytest.approx(-1000.0 + np.log(2.0))
    values, weights = np.array([3.0, 1.0, 2.0]), np.array([0.2, 0.5, 0.3])
    assert weighted_quantile(values, weights, 0.4) == 1.0
    assert weighted_quantile(values, weights, 0.9) == 3.0


# --------------------------------------------------------------- robustness


def test_student_t_filter_shrugs_off_a_wild_reading() -> None:
    days = [0, 1, 2, 3, 4]
    calm = readings(6, days, [0.1, 0.12, 0.08, 0.1, 0.11], [1.0] * 5)
    wild = readings(6, [*days, 5], [0.1, 0.12, 0.08, 0.1, 0.11, 3.0], [1.0] * 6)
    robust = MoodParameters(TRUE.phi, TRUE.sigma_state, TRUE.sigma_obs, dof=3.0)
    jump_normal = (
        particle_filter(TRUE, wild, 20_000, 1).mean[5]
        - particle_filter(TRUE, calm, 20_000, 1).mean[5]
    )
    jump_robust = (
        particle_filter(robust, wild, 20_000, 1).mean[5]
        - particle_filter(robust, calm, 20_000, 1).mean[5]
    )
    assert jump_normal > 0.5  # the normal model believes the outlier
    assert jump_robust < 0.3 * jump_normal


def test_student_t_filter_tracks_better_when_readings_contain_outliers() -> None:
    robust = MoodParameters(TRUE.phi, TRUE.sigma_state, TRUE.sigma_obs, dof=3.0)
    errors_normal, errors_robust = [], []
    for seed in range(4):
        simulation = simulate(TRUE, 600, seed=seed, readings_per_day=0.4, outlier_share=0.08)
        normal = kalman_filter(TRUE, simulation.observations).mean
        heavy = particle_filter(robust, simulation.observations, 3_000, seed, with_bands=False).mean
        errors_normal.append(np.sqrt(np.mean((normal - simulation.mood) ** 2)))
        errors_robust.append(np.sqrt(np.mean((heavy - simulation.mood) ** 2)))
    assert np.mean(errors_robust) < 0.9 * np.mean(errors_normal)


# --------------------------------------------------------------- estimation


def test_transform_round_trip_keeps_parameters_valid() -> None:
    back = estimate.from_vector(estimate.to_vector(TRUE), None)
    assert (back.phi, back.sigma_state, back.sigma_obs) == pytest.approx((0.97, 0.08, 0.3))
    extreme = estimate.from_vector(np.array([50.0, -30.0, 5.0]), None)
    assert 0 < extreme.phi <= 1 and extreme.sigma_state > 0


def test_exact_likelihood_recovers_the_parameters_from_irregular_readings() -> None:
    series = [
        simulate(TRUE, 2_000, seed=seed, readings_per_day=0.3).observations for seed in range(6)
    ]
    fitted = estimate.fit_gaussian(series)
    assert fitted.phi == pytest.approx(TRUE.phi, abs=0.015)
    assert fitted.sigma_state == pytest.approx(TRUE.sigma_state, rel=0.25)
    assert fitted.sigma_obs == pytest.approx(TRUE.sigma_obs, rel=0.1)


def test_particle_likelihood_recovers_the_parameters_too() -> None:
    series = [
        simulate(TRUE, 1_500, seed=seed, readings_per_day=0.4).observations for seed in range(3)
    ]
    fitted = estimate.fit_particle(series, None, n_particles=600, seed=11)
    assert fitted.phi == pytest.approx(TRUE.phi, abs=0.04)
    assert fitted.sigma_state == pytest.approx(TRUE.sigma_state, rel=0.5)
    assert fitted.sigma_obs == pytest.approx(TRUE.sigma_obs, rel=0.15)


# ---------------------------------------------------------- market and firm


def test_two_stage_filter_recovers_market_and_firm_moods() -> None:
    market_parameters = MoodParameters(0.98, 0.05, 0.3)
    firm_parameters = MoodParameters(0.95, 0.08, 0.3)
    market, totals, frame = hierarchy.simulate_hierarchy(
        market_parameters, firm_parameters, n_firms=12, n_days=1_000, seed=3
    )
    result = hierarchy.two_stage_filter(
        frame, 1_000, market_parameters, firm_parameters, n_particles=1_500, seed=4
    )
    assert np.corrcoef(result.market.mean, market)[0, 1] > 0.6
    errors = result.total_mean.to_numpy() - totals.to_numpy()
    assert np.sqrt((errors**2).mean()) < 0.85 * totals.to_numpy().std()
    standardised = errors / np.sqrt(result.total_variance.to_numpy())
    assert 0.85 < standardised.std() < 1.2  # the stated uncertainty is about right
    assert list(result.total_mean.columns) == sorted(frame["firm"].unique())
