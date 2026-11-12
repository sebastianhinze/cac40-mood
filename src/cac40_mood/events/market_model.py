"""Abnormal returns from the market model.

For one event (a firm and its event session d0):
1. Estimation window: trading days -250 to -30 before d0. Regress the firm's daily return on
   the CAC 40's return there: return = alpha + beta * market_return + error.
2. Abnormal return on any later day: actual return - (alpha + beta * market return that day).
   It is the part of the move that the market's own move does not explain.
3. Cumulative abnormal return (CAR): the sum of abnormal returns over a window.
   CAR[0] is the reaction on the event day; CAR[+1,+5] is the drift afterwards;
   CAR[-5,-1] is a placebo (nothing should happen before the news, unless it leaked).

The estimation window stops 30 days before the event, so the event itself (and any run-up)
cannot distort alpha and beta. All windows count trading days, not calendar days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ESTIMATION_START: int = -250
ESTIMATION_END: int = -30
EVENT_WINDOW: tuple[int, int] = (-5, 5)
MIN_ESTIMATION_DAYS: int = 120  # at least about half of the 221-day window must have returns

CAR_WINDOWS: dict[str, tuple[int, int]] = {
    "car_0": (0, 0),
    "car_1_5": (1, 5),
    "car_placebo": (-5, -1),
}


@dataclass(frozen=True)
class ReturnPanel:
    """Daily log returns: one column per firm, plus the market, on the trading calendar."""

    dates: pd.DatetimeIndex
    firm_returns: pd.DataFrame
    market_returns: pd.Series


def build_return_panel(prices: pd.DataFrame, market_id: str, last_day: pd.Timestamp) -> ReturnPanel:
    """Log returns from dividend-adjusted closes, up to `last_day` (end of validation).

    Cutting the panel at `last_day` is what keeps test-period returns out of every result:
    an event whose window would need a later day simply gets no CAR.
    """
    kept = prices[prices["date"] <= last_day]
    # One row per date, one column per firm. There is exactly one price per firm and date, so
    # "first" just picks it.
    closes = kept.pivot_table(index="date", columns="firm_id", values="adj_close", aggfunc="first")
    closes = closes.sort_index()
    returns = np.log(closes).diff()
    return ReturnPanel(
        pd.DatetimeIndex(returns.index),
        returns.drop(columns=market_id),
        returns[market_id],
    )


def fit_market_model(firm: np.ndarray, market: np.ndarray) -> tuple[float, float, float] | None:
    """(alpha, beta, residual standard deviation), or None if too few usable days."""
    usable = ~(np.isnan(firm) | np.isnan(market))
    if usable.sum() < MIN_ESTIMATION_DAYS:
        return None
    x, y = market[usable], firm[usable]
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha = y.mean() - beta * x.mean()
    residuals = y - alpha - beta * x
    return float(alpha), float(beta), float(residuals.std(ddof=2))


def abnormal_returns(panel: ReturnPanel, events: pd.DataFrame) -> pd.DataFrame:
    """For each event (columns firm_id, event_session) return alpha, beta, the abnormal return
    on each day of the event window (ar_-5 .. ar_5) and the three CARs.

    An event without enough estimation data, or whose event window is not fully inside the
    panel, gets missing values rather than a partial sum.
    """
    position_of = {date: position for position, date in enumerate(panel.dates)}
    market = panel.market_returns.to_numpy()
    first, last = EVENT_WINDOW
    relative_days = list(range(first, last + 1))
    rows = []
    for event in events.itertuples(index=False):
        row: dict[str, float] = {"alpha": np.nan, "beta": np.nan, "residual_sd": np.nan}
        row.update({f"ar_{day}": np.nan for day in relative_days})
        position = position_of.get(event.event_session)
        if position is not None and event.firm_id in panel.firm_returns.columns:
            firm = panel.firm_returns[event.firm_id].to_numpy()
            start, stop = position + ESTIMATION_START, position + ESTIMATION_END + 1
            window_inside = position + first >= 0 and position + last < len(firm)
            model = fit_market_model(firm[start:stop], market[start:stop]) if start >= 0 else None
            if model is not None and window_inside:
                alpha, beta, residual_sd = model
                days = slice(position + first, position + last + 1)
                abnormal = firm[days] - (alpha + beta * market[days])
                row.update(alpha=alpha, beta=beta, residual_sd=residual_sd)
                row.update(dict(zip([f"ar_{day}" for day in relative_days], abnormal, strict=True)))
        rows.append(row)
    result = pd.DataFrame(rows, index=events.index)
    for name, (start_day, end_day) in CAR_WINDOWS.items():
        columns = [f"ar_{day}" for day in range(start_day, end_day + 1)]
        # skipna=False: one missing day makes the whole CAR missing.
        result[name] = result[columns].sum(axis=1, skipna=False)
    return pd.concat([events, result], axis=1)
