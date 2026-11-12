"""Score volatility forecasts and test whether one model beats another.

Losses
- QLIKE = actual/forecast - log(actual/forecast) - 1. Zero for a perfect forecast, and it
  punishes under-forecasting more than over-forecasting. It is the standard loss for variance
  forecasts because it stays valid when the "actual" variance is itself a noisy estimate
  (Patton 2011). This is the primary loss (DECISIONS #16).
- Squared error of the log variance, as a second view.

Test (Diebold-Mariano): for each day take the average loss difference between two models
across firms, then ask whether the mean of that daily series is zero. The standard error is
Newey-West with at least h-1 lags, because h-day-ahead forecast errors overlap. Averaging
across firms first means that all firms moving together on a bad day counts as one
observation, not 36.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def qlike(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    ratio = actual / forecast
    return ratio - np.log(ratio) - 1.0


def squared_log_error(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    return (np.log(actual) - np.log(forecast)) ** 2


def newey_west_variance_of_mean(series: np.ndarray, lags: int) -> float:
    """Variance of the sample mean of an autocorrelated series (Bartlett weights)."""
    count = len(series)
    centred = series - series.mean()
    long_run = centred @ centred / count
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run += 2.0 * weight * (centred[lag:] @ centred[:-lag]) / count
    return long_run / count


def diebold_mariano(daily_loss_difference: pd.Series, horizon: int) -> dict[str, float]:
    """Test that the mean daily loss difference (model A minus model B) is zero.

    A negative statistic means model A has the lower loss. Returns the mean difference, the
    statistic, the two-sided p-value (normal approximation) and the number of days.
    """
    values = daily_loss_difference.dropna().to_numpy()
    lags = max(horizon - 1, int(np.floor(4 * (len(values) / 100.0) ** (2.0 / 9.0))))
    variance = newey_west_variance_of_mean(values, lags)
    if variance <= 0:
        return {
            "mean_difference": float(values.mean()),
            "statistic": np.nan,
            "p_value": np.nan,
            "days": float(len(values)),
            "lags": float(lags),
        }
    statistic = values.mean() / np.sqrt(variance)
    return {
        "mean_difference": float(values.mean()),
        "statistic": float(statistic),
        "p_value": float(2.0 * (1.0 - stats.norm.cdf(abs(statistic)))),
        "days": float(len(values)),
        "lags": float(lags),
    }


def mincer_zarnowitz(log_actual: pd.Series, log_forecast: pd.Series) -> dict[str, float]:
    """Regress the log outcome on the log forecast: a good forecast has slope 1 and a high R²."""
    slope, intercept, correlation, _, _ = stats.linregress(log_forecast, log_actual)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": float(correlation**2),
    }


def compare_models(forecasts: dict[str, pd.DataFrame], baseline: str, horizon: int) -> pd.DataFrame:
    """One row per model: average losses, Mincer-Zarnowitz fit, and the test against `baseline`.

    Each entry of `forecasts` is a table with firm_id, date, forecast, actual. Models are
    compared on the firm-days that every model has a forecast for.
    """
    keys = ["firm_id", "date"]
    common = forecasts[baseline][keys]
    for table in forecasts.values():
        common = common.merge(table[keys], on=keys)
    rows = []
    base = common.merge(forecasts[baseline], on=keys)
    base_loss = qlike(base["actual"], base["forecast"])
    for name, table in forecasts.items():
        data = common.merge(table, on=keys)
        loss = qlike(data["actual"], data["forecast"])
        fit = mincer_zarnowitz(np.log(data["actual"]), np.log(data["forecast"]))
        row = {
            "model": name,
            "horizon": horizon,
            "firm_days": len(data),
            "qlike": float(loss.mean()),
            "mse_log": float(squared_log_error(data["actual"], data["forecast"]).mean()),
            "mz_slope": fit["slope"],
            "mz_r2": fit["r_squared"],
        }
        if name != baseline:
            daily = (loss - base_loss.to_numpy()).groupby(data["date"].to_numpy()).mean()
            test = diebold_mariano(daily, horizon)
            row.update(
                qlike_minus_baseline=test["mean_difference"],
                dm_statistic=test["statistic"],
                dm_p_value=test["p_value"],
            )
        rows.append(row)
    return pd.DataFrame(rows)
