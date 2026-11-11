"""Event-time alignment: the rule that keeps future information out of every later result."""

import numpy as np
import pandas as pd
import pytest

from cac40_mood.data.sessions import align_to_sessions, to_paris, trading_sessions

# Paris exchange holidays in 2024 that fall on a weekday.
HOLIDAYS_2024 = pd.to_datetime(
    ["2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-12-25", "2024-12-26"]
)


@pytest.fixture
def sessions() -> pd.DatetimeIndex:
    """A realistic 2024 calendar: weekdays minus the exchange holidays."""
    weekdays = pd.bdate_range("2024-01-01", "2024-12-31")
    return weekdays.difference(HOLIDAYS_2024)


def align_one(paris_time: str, sessions: pd.DatetimeIndex) -> pd.Series:
    stamps = pd.Series([pd.Timestamp(paris_time, tz="Europe/Paris")])
    return align_to_sessions(stamps, sessions).iloc[0]


@pytest.mark.parametrize(
    ("paris_time", "expected_session", "expected_intraday"),
    [
        ("2024-01-23 07:30:00", "2024-01-23", False),  # before the open
        ("2024-01-23 08:59:59", "2024-01-23", False),  # last second before the open
        ("2024-01-23 09:00:00", "2024-01-23", True),  # the open itself counts as intraday
        ("2024-01-23 12:00:00", "2024-01-23", True),  # lunchtime
        ("2024-01-23 17:29:59", "2024-01-23", True),  # last second of trading
        ("2024-01-23 17:30:00", "2024-01-24", False),  # the close itself is too late
        ("2024-01-23 23:59:59", "2024-01-24", False),  # evening
        ("2024-01-26 18:00:00", "2024-01-29", False),  # Friday evening -> Monday
        ("2024-01-27 10:00:00", "2024-01-29", False),  # Saturday -> Monday, never intraday
        ("2024-01-28 23:00:00", "2024-01-29", False),  # Sunday night -> Monday
        ("2024-03-28 18:00:00", "2024-04-02", False),  # Thursday before Easter -> Tuesday
        ("2024-03-29 08:00:00", "2024-04-02", False),  # Good Friday morning
        ("2024-04-01 12:00:00", "2024-04-02", False),  # Easter Monday at noon is not intraday
        ("2024-05-01 10:00:00", "2024-05-02", False),  # 1 May
        ("2024-07-12 18:00:00", "2024-07-15", False),  # Bastille Day weekend: 14 July is a Sunday
        ("2024-12-24 18:00:00", "2024-12-27", False),  # Christmas: 25 and 26 are closed
    ],
)
def test_rule_table(sessions, paris_time, expected_session, expected_intraday) -> None:
    row = align_one(paris_time, sessions)
    assert row["event_session"] == pd.Timestamp(expected_session)
    assert bool(row["intraday"]) is expected_intraday


def test_conservative_session_moves_only_intraday_filings(sessions) -> None:
    intraday = align_one("2024-01-26 12:00:00", sessions)  # Friday noon
    assert intraday["event_session"] == pd.Timestamp("2024-01-26")
    assert intraday["event_session_conservative"] == pd.Timestamp("2024-01-29")
    pre_open = align_one("2024-01-26 07:00:00", sessions)
    assert pre_open["event_session_conservative"] == pre_open["event_session"]


def test_daylight_saving_is_handled_from_utc(sessions) -> None:
    """The same UTC clock time is before or after the Paris close depending on the season.

    Winter (UTC+1): 16:45 UTC = 17:45 Paris -> after the close.
    Summer (UTC+2): 15:45 UTC = 17:45 Paris -> after the close; 15:15 UTC = 17:15 -> intraday.
    The clocks changed on 31 March and 27 October 2024.
    """
    utc = pd.Series(
        pd.to_datetime(
            [
                "2024-03-28 16:45:00",  # winter time, 17:45 Paris, Thursday before Easter
                "2024-04-02 15:45:00",  # summer time, 17:45 Paris
                "2024-04-02 15:15:00",  # summer time, 17:15 Paris
                "2024-10-25 15:40:00",  # last summer-time Friday, 17:40 Paris
                "2024-10-28 15:40:00",  # first winter-time Monday, 16:40 Paris
            ],
            utc=True,
        )
    )
    aligned = align_to_sessions(to_paris(utc), sessions)
    assert list(aligned["event_session"]) == list(
        pd.to_datetime(["2024-04-02", "2024-04-03", "2024-04-02", "2024-10-28", "2024-10-28"])
    )
    assert list(aligned["intraday"]) == [False, False, True, False, True]


def test_missing_and_out_of_calendar_timestamps_get_no_session(sessions) -> None:
    stamps = pd.Series(
        [
            pd.NaT,
            pd.Timestamp("2023-12-29 10:00", tz="Europe/Paris"),  # before the calendar starts
            pd.Timestamp("2024-12-31 18:00", tz="Europe/Paris"),  # after the last session closes
            pd.Timestamp("2024-12-31 08:00", tz="Europe/Paris"),  # last session, before the open
        ]
    )
    aligned = align_to_sessions(stamps, sessions)
    assert aligned["event_session"].isna().tolist() == [True, True, True, False]
    assert aligned["intraday"].tolist() == [False, False, False, False]


def test_index_is_preserved() -> None:
    sessions = pd.bdate_range("2024-01-01", "2024-01-31")
    stamps = pd.Series(
        [pd.Timestamp("2024-01-10 10:00", tz="Europe/Paris")] * 2, index=["first", "second"]
    )
    assert list(align_to_sessions(stamps, sessions).index) == ["first", "second"]


def test_no_look_ahead_against_brute_force(sessions) -> None:
    """For random timestamps, d0 must be the first session whose 17:30 close is after the filing.

    This restates the rule in the most literal way and checks it one timestamp at a time:
    a filing can only be traded on after it exists, so the earliest session whose close comes
    strictly after the timestamp is the only acceptable d0.
    """
    rng = np.random.default_rng(20240101)
    seconds = rng.integers(0, 330 * 24 * 3600, size=3000)
    stamps = pd.Series(pd.Timestamp("2024-01-02", tz="UTC") + pd.to_timedelta(seconds, unit="s"))
    paris = to_paris(stamps)
    aligned = align_to_sessions(paris, sessions)

    closes = sessions.tz_localize("Europe/Paris") + pd.Timedelta(hours=17, minutes=30)
    for stamp, session in zip(paris, aligned["event_session"], strict=True):
        first_close_after = closes[closes > stamp][0]
        assert session == first_close_after.tz_localize(None).normalize()


def test_trading_sessions_come_from_index_prices() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"]),
            "close": [7500.0, 7400.0, np.nan],
        }
    )
    assert list(trading_sessions(prices)) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
