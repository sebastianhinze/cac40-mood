"""Trading calendar and the event-time rule: which trading session can first react to a filing?

This is the look-ahead guard of the whole project (CLAUDE.md, "Statistical hygiene"):

    Paris time of the filing                       -> session d0            intraday
    trading day, before 09:00                      -> the same day          False
    trading day, 09:00 <= time < 17:30             -> the same day          True
    trading day, 17:30 or later                    -> next trading day      False
    weekend or exchange holiday (any time)         -> next trading day      False

"Intraday" filings are kept but flagged, because a daily close-to-close return on d0 mixes
minutes before and after the news. `event_session_conservative` moves them to the next session,
which is the robustness variant.

The calendar is the list of dates on which the CAC 40 index (^FCHI) has a price. Weekdays are not
enough: Paris is closed on Good Friday, Easter Monday, 1 May, 25-26 December and 1 January, but
open on most French public holidays (14 July, 15 August, 11 November).

Known simplification: on the two half-days per year (24 and 31 December, close at 14:05) a filing
between 14:05 and 17:30 is still mapped to that day. It is flagged intraday, so the conservative
variant already moves it to the next session.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from cac40_mood.config import PARIS_TZ

MARKET_OPEN: time = time(9, 0)
MARKET_CLOSE: time = time(17, 30)


def to_paris(ts_utc: pd.Series) -> pd.Series:
    """Convert timezone-aware timestamps to Europe/Paris (handles summer/winter time)."""
    return ts_utc.dt.tz_convert(PARIS_TZ)


def trading_sessions(index_prices: pd.DataFrame) -> pd.DatetimeIndex:
    """The trading calendar: every date with a non-missing ^FCHI close, sorted, without time."""
    days = pd.to_datetime(index_prices.loc[index_prices["close"].notna(), "date"])
    return pd.DatetimeIndex(days.dt.normalize().unique()).sort_values()


def _session_after(days: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """First trading session strictly after each calendar day (missing if the calendar ends)."""
    positions = sessions.searchsorted(days.to_numpy(), side="right")
    has_next = positions < len(sessions)
    result = np.full(len(days), np.datetime64("NaT", "ns"))
    result[has_next] = sessions.to_numpy()[positions[has_next]]
    return pd.Series(result, index=days.index)


def align_to_sessions(ts_paris: pd.Series, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Map each Paris-time timestamp to its event session `d0`.

    Returns a frame (same index as the input) with:
    - `event_session`: d0 by the rule in the module docstring;
    - `intraday`: True when the filing arrived during trading hours of d0;
    - `event_session_conservative`: d0, except intraday filings move to the next session.

    Timestamps that are missing, earlier than the first calendar date, or after the last session
    get a missing `event_session`: we refuse to guess rather than mis-date an event.
    """
    sessions = pd.DatetimeIndex(sessions).sort_values()
    result = pd.DataFrame(
        {
            "event_session": pd.NaT,
            "intraday": False,
            "event_session_conservative": pd.NaT,
        },
        index=ts_paris.index,
    ).astype({"event_session": "datetime64[ns]", "event_session_conservative": "datetime64[ns]"})
    usable = ts_paris.notna()
    if len(sessions) == 0 or not usable.any():
        return result

    # Wall-clock Paris time without the timezone tag, so it can be compared with session dates.
    local = ts_paris[usable].dt.tz_localize(None).astype("datetime64[ns]")
    day = local.dt.normalize()
    time_of_day = local - day
    open_offset = pd.Timedelta(hours=MARKET_OPEN.hour, minutes=MARKET_OPEN.minute)
    close_offset = pd.Timedelta(hours=MARKET_CLOSE.hour, minutes=MARKET_CLOSE.minute)

    is_session_day = day.isin(sessions)
    same_day = is_session_day & (time_of_day < close_offset)
    intraday = same_day & (time_of_day >= open_offset)
    next_session = _session_after(day, sessions)

    event_session = day.where(same_day, next_session)
    in_calendar = day >= sessions[0]
    event_session = event_session.where(in_calendar)
    conservative = event_session.where(~intraday, next_session)

    result.loc[usable, "event_session"] = event_session
    result.loc[usable, "intraday"] = intraday & in_calendar
    result.loc[usable, "event_session_conservative"] = conservative
    return result
