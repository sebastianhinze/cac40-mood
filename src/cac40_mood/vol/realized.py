"""Daily variance estimates from open/high/low/close prices.

Free data gives one bar per day, so the classic "realized variance" (sum of squared 5-minute
returns) is out of reach. Range-based estimators are the next best thing: the distance between
the day's high and low says much more about that day's volatility than one close-to-close
return does (DECISIONS #11).

Both estimators measure the trading session only (open to close). The overnight jump from
yesterday's close to today's open is not included; that is the same for every model we compare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cac40_mood.splits import VAL_END

TWO_LOG_TWO_MINUS_ONE: float = 2.0 * np.log(2.0) - 1.0


def garman_klass_variance(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Garman-Klass (1980) daily variance: 0.5*ln(H/L)^2 - (2 ln 2 - 1)*ln(C/O)^2.

    The first term uses the day's range; the second subtracts the part of the range that is
    explained by the open-to-close move (a trending day has a wide range without being "noisy").
    """
    log_range = np.log(high / low)
    log_open_to_close = np.log(close / open_)
    return 0.5 * log_range**2 - TWO_LOG_TWO_MINUS_ONE * log_open_to_close**2


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980) daily variance: ln(H/L)^2 / (4 ln 2). Uses the range only."""
    return np.log(high / low) ** 2 / (4.0 * np.log(2.0))


def variance_panel(prices: pd.DataFrame, firms: list[str]) -> pd.DataFrame:
    """One row per firm and trading day: Garman-Klass variance, Parkinson variance, log return.

    Only days up to the end of the validation period are returned, so nothing built on this
    table can see the test period. A day whose variance estimate is not positive (high = low,
    which happens on a trading halt) is set to missing: its logarithm does not exist.
    """
    kept = prices[prices["firm_id"].isin(firms) & (prices["date"] <= pd.Timestamp(VAL_END))]
    kept = kept.sort_values(["firm_id", "date"]).reset_index(drop=True)
    panel = kept[["firm_id", "date"]].copy()
    panel["rv"] = garman_klass_variance(kept["open"], kept["high"], kept["low"], kept["close"])
    panel["rv_parkinson"] = parkinson_variance(kept["high"], kept["low"])
    for column in ("rv", "rv_parkinson"):
        panel[column] = panel[column].where(panel[column] > 0)
    log_price = np.log(kept["adj_close"])
    panel["ret"] = log_price.groupby(kept["firm_id"]).diff()
    return panel
