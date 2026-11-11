"""Volatility: range estimators, HAR features without look-ahead, GARCH recursion, evaluation."""

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from cac40_mood.splits import VAL_END
from cac40_mood.vol import evaluate, garch, har
from cac40_mood.vol.realized import garman_klass_variance, parkinson_variance, variance_panel


def simulated_panel(days: int = 600, firms: int = 3, seed: int = 7) -> pd.DataFrame:
    """Log variance follows a persistent AR(1) per firm; returns are drawn with that variance."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=days)
    rows = []
    for firm in range(firms):
        log_rv = np.empty(days)
        log_rv[0] = -9.0
        for day in range(1, days):
            log_rv[day] = -9.0 + 0.9 * (log_rv[day - 1] + 9.0) + rng.normal(0, 0.3)
        rv = np.exp(log_rv + firm * 0.2)
        ret = rng.normal(0, np.sqrt(rv))
        rows.append(pd.DataFrame({"firm_id": f"F{firm}", "date": dates, "rv": rv, "ret": ret}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------- estimators


def test_range_estimators_on_hand_computed_values() -> None:
    open_, high, low, close = (pd.Series([value]) for value in (100.0, 102.0, 99.0, 101.0))
    log_range, log_move = np.log(102 / 99), np.log(101 / 100)
    expected_gk = 0.5 * log_range**2 - (2 * np.log(2) - 1) * log_move**2
    assert garman_klass_variance(open_, high, low, close).iloc[0] == pytest.approx(expected_gk)
    assert parkinson_variance(high, low).iloc[0] == pytest.approx(log_range**2 / (4 * np.log(2)))


def test_estimators_recover_the_true_variance_of_a_simulated_day() -> None:
    """A driftless random walk with 2% daily volatility: both averages are near 0.0004."""
    rng = np.random.default_rng(1)
    steps = rng.normal(0, 0.02 / np.sqrt(390), size=(4000, 390)).cumsum(axis=1)
    paths = np.hstack([np.zeros((4000, 1)), steps])
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100 * np.exp(paths.max(axis=1)),
            "low": 100 * np.exp(paths.min(axis=1)),
            "close": 100 * np.exp(paths[:, -1]),
        }
    )
    gk = garman_klass_variance(frame["open"], frame["high"], frame["low"], frame["close"]).mean()
    parkinson = parkinson_variance(frame["high"], frame["low"]).mean()
    # A path sampled 390 times misses the true extremes a little, so both read slightly low.
    assert gk == pytest.approx(0.0004, rel=0.12)
    assert parkinson == pytest.approx(0.0004, rel=0.12)


def test_variance_panel_stops_at_the_end_of_validation_and_drops_zero_ranges() -> None:
    dates = pd.to_datetime(["2024-06-27", "2024-06-28", "2024-07-01", "2024-07-02"])
    prices = pd.DataFrame(
        {
            "firm_id": "MC",
            "date": dates,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [102.0, 100.0, 103.0, 101.0],
            "low": [99.0, 100.0, 98.0, 99.0],
            "close": [101.0, 100.0, 100.0, 100.0],
            "adj_close": [101.0, 100.0, 100.0, 100.0],
        }
    )
    panel = variance_panel(prices, ["MC"])
    assert panel["date"].max() == pd.Timestamp(VAL_END).normalize() - pd.Timedelta(days=2)
    assert len(panel) == 2
    assert np.isnan(panel["rv"].iloc[1])  # high = low: no variance estimate, not a zero
    assert panel["ret"].iloc[1] == pytest.approx(np.log(100 / 101))


# -------------------------------------------------------------------- HAR


def test_features_use_only_the_past_and_targets_only_the_future() -> None:
    panel = simulated_panel(days=120, firms=1)
    features = har.har_features(panel)
    row = 60
    rv = panel["rv"].to_numpy()
    assert features.loc[row, "log_rv_d"] == pytest.approx(np.log(rv[row]))
    assert features.loc[row, "log_rv_w"] == pytest.approx(np.log(rv[row - 4 : row + 1].mean()))
    assert features.loc[row, "log_rv_m"] == pytest.approx(np.log(rv[row - 21 : row + 1].mean()))
    assert features.loc[row, "target_5"] == pytest.approx(np.log(rv[row + 1 : row + 6].mean()))
    assert features.loc[row, "target_end_5"] == panel.loc[row + 5, "date"]
    assert features["target_22"].iloc[-22:].isna().all()  # window would run past the data


def test_changing_the_future_changes_no_past_feature() -> None:
    """The plan's test: alter everything after day t and compare the features up to t."""
    panel = simulated_panel(days=200, firms=2)
    cut = panel["date"].sort_values().unique()[120]
    altered = panel.copy()
    altered.loc[altered["date"] > cut, "rv"] *= 50.0
    before = har.har_features(panel)
    after = har.har_features(altered)
    past = before["date"] <= cut
    pd.testing.assert_frame_equal(
        before.loc[past, har.HAR_COLUMNS], after.loc[past, har.HAR_COLUMNS]
    )
    assert not np.allclose(
        before.loc[past, "target_22"].dropna(), after.loc[past, "target_22"].dropna()
    )


def test_pooled_fit_recovers_known_slopes_and_firm_effects() -> None:
    rng = np.random.default_rng(3)
    count = 4000
    frame = pd.DataFrame(
        {
            "firm_id": rng.choice(["A", "B", "C"], size=count),
            "x1": rng.normal(size=count),
            "x2": rng.normal(size=count),
        }
    )
    effects = {"A": -1.0, "B": 0.5, "C": 2.0}
    frame["y"] = (
        frame["firm_id"].map(effects)
        + 0.7 * frame["x1"]
        - 0.2 * frame["x2"]
        + rng.normal(0, 0.1, count)
    )
    model = har.fit_pooled(frame, ["x1", "x2"], "y")
    assert model.slopes["x1"] == pytest.approx(0.7, abs=0.01)
    assert model.slopes["x2"] == pytest.approx(-0.2, abs=0.01)
    assert model.firm_effects.to_dict() == pytest.approx(effects, abs=0.02)
    assert model.residual_variance == pytest.approx(0.01, rel=0.1)


def test_refits_never_use_a_target_that_ends_after_the_refit_day(monkeypatch) -> None:
    panel = simulated_panel(days=400, firms=2)
    features = har.har_features(panel)
    dates = np.sort(features["date"].unique())
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    original = har.fit_pooled

    def spy(frame, columns, target):
        seen.append((frame["target_end_22"].max(), frame["date"].max()))
        return original(frame, columns, target)

    monkeypatch.setattr(har, "fit_pooled", spy)
    forecasts = har.expanding_forecasts(features, har.HAR_COLUMNS, 22, dates[300], dates[370])
    refit_days = np.sort(forecasts["date"].unique())[:: har.REFIT_EVERY]
    assert len(seen) == len(refit_days)
    for (latest_target_end, _), refit_day in zip(seen, refit_days, strict=True):
        assert latest_target_end <= refit_day


def test_har_forecasts_beat_a_constant_forecast_on_persistent_data() -> None:
    panel = simulated_panel(days=700, firms=3)
    features = har.har_features(panel)
    dates = np.sort(features["date"].unique())
    forecasts = har.expanding_forecasts(features, har.HAR_COLUMNS, 5, dates[450], dates[690])
    constant = forecasts.assign(forecast=forecasts["actual"].mean())
    table = evaluate.compare_models({"HAR": forecasts, "constant": constant}, "HAR", 5)
    by_model = table.set_index("model")
    assert by_model.loc["HAR", "qlike"] < by_model.loc["constant", "qlike"]
    assert by_model.loc["constant", "dm_statistic"] > 2  # the constant is significantly worse


# ------------------------------------------------------------------ GARCH


def test_variance_recursion_matches_the_arch_package() -> None:
    rng = np.random.default_rng(5)
    returns = rng.normal(0, 1.2, size=1500)
    result = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(
        disp="off"
    )
    parameters = garch.fit_garch(returns)
    ours = garch.next_day_variances(returns, parameters)
    # arch's conditional variance of day t+1 is our "next day" value of day t (different start
    # values fade out after a few hundred days).
    assert ours[500:-1] == pytest.approx(result.conditional_volatility[501:] ** 2, rel=1e-3)
    ahead = result.forecast(horizon=5, reindex=False).variance.to_numpy()[-1]
    assert garch.average_variance_ahead(ours[-1:], parameters, 5)[0] == pytest.approx(
        ahead.mean(), rel=1e-3
    )


def test_multi_day_forecast_moves_towards_the_long_run_level() -> None:
    parameters = garch.GarchParameters(mu=0.0, omega=0.1, alpha=0.1, beta=0.8)  # long run = 1.0
    high_now = garch.average_variance_ahead(np.array([4.0]), parameters, 1)[0]
    month = garch.average_variance_ahead(np.array([4.0]), parameters, 22)[0]
    far = garch.average_variance_ahead(np.array([4.0]), parameters, 2000)[0]
    assert high_now == pytest.approx(4.0)
    assert 1.0 < month < high_now
    assert far == pytest.approx(1.0, abs=0.02)


def test_garch_forecasts_line_up_with_har_targets() -> None:
    panel = simulated_panel(days=500, firms=2)
    features = har.har_features(panel)
    dates = np.sort(features["date"].unique())
    har_table = har.expanding_forecasts(features, har.HAR_COLUMNS, 5, dates[400], dates[470])
    garch_table = garch.garch_forecasts(panel, har_table, 5)
    assert len(garch_table) == len(har_table)
    merged = har_table.merge(garch_table, on=["firm_id", "date"], suffixes=("_har", "_garch"))
    assert np.allclose(merged["actual_har"], merged["actual_garch"])
    assert (garch_table["forecast"] > 0).all()


# -------------------------------------------------------------- evaluation


def test_qlike_is_zero_for_a_perfect_forecast_and_punishes_low_forecasts_more() -> None:
    actual = pd.Series([1.0, 1.0, 1.0])
    forecast = pd.Series([1.0, 0.5, 2.0])
    loss = evaluate.qlike(actual, forecast)
    assert loss.iloc[0] == pytest.approx(0.0)
    assert loss.iloc[1] > loss.iloc[2] > 0  # forecasting half is worse than forecasting double


def test_newey_west_equals_the_plain_variance_without_lags() -> None:
    values = np.random.default_rng(2).normal(size=500)
    plain = values.var() / len(values)
    assert evaluate.newey_west_variance_of_mean(values, 0) == pytest.approx(plain)


def test_diebold_mariano_detects_a_real_difference_and_not_noise() -> None:
    rng = np.random.default_rng(9)
    noise = pd.Series(rng.normal(0, 1, 400))
    assert evaluate.diebold_mariano(noise, 5)["p_value"] > 0.05
    shifted = evaluate.diebold_mariano(noise - 0.5, 5)
    assert shifted["statistic"] < -3 and shifted["p_value"] < 0.01
    assert shifted["lags"] >= 4  # at least horizon - 1


def test_mincer_zarnowitz_of_a_perfect_forecast() -> None:
    values = pd.Series(np.linspace(-10, -8, 50))
    fit = evaluate.mincer_zarnowitz(values, values)
    assert fit["slope"] == pytest.approx(1.0) and fit["r_squared"] == pytest.approx(1.0)
