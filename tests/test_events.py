"""Event study on synthetic data: known alpha, beta and injected abnormal returns."""

import numpy as np
import pandas as pd
import pytest

from cac40_mood.events import market_model as mm
from cac40_mood.events import study

N_DAYS = 900
EVENT_DAY_EFFECT = 0.03
DRIFT_PER_DAY = 0.002


def synthetic_panel(seed: int = 0, inject: bool = True):
    """Three firms with known alpha and beta; firm A gets abnormal returns around its events.

    Returns (panel, events). Events sit every 40 trading days from day 300 on.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=N_DAYS)
    market = rng.normal(0.0003, 0.01, N_DAYS)
    betas = {"A": 1.3, "B": 0.7, "C": 1.0}
    alphas = {"A": 0.0002, "B": -0.0001, "C": 0.0}
    firms = {}
    for firm, beta in betas.items():
        firms[firm] = alphas[firm] + beta * market + rng.normal(0, 0.008, N_DAYS)
    event_positions = np.arange(300, N_DAYS - 10, 40)
    if inject:
        for position in event_positions:
            firms["A"][position] += EVENT_DAY_EFFECT
            firms["A"][position + 1 : position + 6] += DRIFT_PER_DAY
    panel = mm.ReturnPanel(dates, pd.DataFrame(firms, index=dates), pd.Series(market, index=dates))
    events = pd.DataFrame({"firm_id": "A", "event_session": dates[event_positions]})
    return panel, events, betas


def test_market_model_recovers_beta_and_the_injected_abnormal_returns() -> None:
    panel, events, betas = synthetic_panel()
    table = mm.abnormal_returns(panel, events)
    assert table["beta"].mean() == pytest.approx(betas["A"], abs=0.08)
    # Standard error of a mean CAR[0] over 15 events with 0.8% daily noise is about 0.2%.
    assert table["car_0"].mean() == pytest.approx(EVENT_DAY_EFFECT, abs=0.006)
    assert table["car_1_5"].mean() == pytest.approx(5 * DRIFT_PER_DAY, abs=0.012)
    assert abs(table["car_placebo"].mean()) < 0.012


def test_event_returns_do_not_leak_into_alpha_and_beta() -> None:
    """The estimation window ends 30 days before the event, so the event cannot move the fit."""
    with_events, events, _ = synthetic_panel(inject=True)
    without_events, _, _ = synthetic_panel(inject=False)
    first = events.iloc[[0]]
    fitted = mm.abnormal_returns(with_events, first).iloc[0]
    clean = mm.abnormal_returns(without_events, first).iloc[0]
    assert fitted["alpha"] == pytest.approx(clean["alpha"])
    assert fitted["beta"] == pytest.approx(clean["beta"])
    assert fitted["car_0"] - clean["car_0"] == pytest.approx(EVENT_DAY_EFFECT)


def test_abnormal_return_is_actual_minus_alpha_minus_beta_times_market() -> None:
    panel, events, _ = synthetic_panel()
    row = mm.abnormal_returns(panel, events.iloc[[3]]).iloc[0]
    position = panel.dates.get_loc(row["event_session"])
    actual = panel.firm_returns["A"].iloc[position + 2]
    market = panel.market_returns.iloc[position + 2]
    assert row["ar_2"] == pytest.approx(actual - row["alpha"] - row["beta"] * market)
    assert row["car_1_5"] == pytest.approx(sum(row[f"ar_{day}"] for day in range(1, 6)))
    assert row["car_placebo"] == pytest.approx(sum(row[f"ar_{day}"] for day in range(-5, 0)))


def test_windows_count_trading_days_not_calendar_days() -> None:
    panel, events, _ = synthetic_panel()
    row = mm.abnormal_returns(panel, events.iloc[[0]]).iloc[0]
    position = panel.dates.get_loc(row["event_session"])
    five_sessions_later = panel.dates[position + 5]
    assert (five_sessions_later - row["event_session"]).days == 7  # five sessions span a weekend
    expected = panel.firm_returns["A"].iloc[position + 5] - (
        row["alpha"] + row["beta"] * panel.market_returns.iloc[position + 5]
    )
    assert row["ar_5"] == pytest.approx(expected)


def test_events_without_enough_history_or_too_close_to_the_end_get_no_car() -> None:
    panel, _, _ = synthetic_panel()
    events = pd.DataFrame(
        {
            "firm_id": ["A", "A", "A", "Z", "A"],
            "event_session": [
                panel.dates[100],  # fewer than 250 days of history
                panel.dates[N_DAYS - 3],  # days +3..+5 are beyond the panel
                panel.dates[500],  # fine
                panel.dates[500],  # unknown firm
                pd.Timestamp("2030-01-01"),  # not a trading day in the panel
            ],
        }
    )
    table = mm.abnormal_returns(panel, events)
    assert table["car_0"].isna().tolist() == [True, True, False, True, True]
    assert table["car_1_5"].isna().tolist() == [True, True, False, True, True]


def test_estimation_needs_enough_usable_days() -> None:
    rng = np.random.default_rng(1)
    market = rng.normal(0, 0.01, 221)
    firm = 1.1 * market + rng.normal(0, 0.005, 221)
    assert mm.fit_market_model(firm, market) is not None
    firm[: 221 - mm.MIN_ESTIMATION_DAYS + 1] = np.nan
    assert mm.fit_market_model(firm, market) is None


def test_return_panel_stops_at_the_last_allowed_day() -> None:
    dates = pd.bdate_range("2024-06-24", periods=8)
    prices = pd.DataFrame(
        {
            "firm_id": ["MC"] * 8 + ["FCHI"] * 8,
            "date": list(dates) * 2,
            "adj_close": list(np.linspace(100, 107, 8)) + list(np.linspace(7000, 7070, 8)),
        }
    )
    panel = mm.build_return_panel(prices, "FCHI", pd.Timestamp("2024-06-28"))
    assert panel.dates.max() == pd.Timestamp("2024-06-28")
    assert list(panel.firm_returns.columns) == ["MC"]
    assert panel.firm_returns["MC"].iloc[1] == pytest.approx(np.log(101 / 100))


# ------------------------------------------------------------------- study


def documents_frame() -> pd.DataFrame:
    rows = [
        # doc, firm, session, intraday, admin, split, gap, tone, confidence
        ("a1", "MC", "2021-03-01", False, False, "train", False, 0.8, 0.9),
        ("a2", "MC", "2021-03-01", True, False, "train", False, 0.2, 0.3),
        ("b1", "OR", "2021-03-01", False, False, "train", False, -0.4, 0.5),
        ("adm", "OR", "2021-03-02", False, True, "train", False, 0.0, 0.5),
        ("gap", "OR", "2020-04-01", False, False, "train", True, 0.5, 0.5),
        ("tst", "OR", "2025-01-10", False, False, "test", False, 0.5, 0.5),
        ("urw", "URW", "2021-03-01", False, False, "train", False, 0.5, 0.5),
        ("nos", "MC", None, False, False, "val", False, 0.5, 0.5),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "doc_id",
            "firm_id",
            "event_session",
            "intraday",
            "is_admin",
            "split",
            "in_feed_gap",
            "tone",
            "confidence",
        ],
    )
    frame["event_session"] = pd.to_datetime(frame["event_session"])
    return frame


def test_events_are_one_per_firm_day_and_exclusions_apply() -> None:
    events = study.prepare_events(documents_frame(), ["tone"]).set_index("firm_id")
    assert sorted(events.index) == ["MC", "OR"]  # admin, gap, test, URW and undated rows are out
    assert events.loc["MC", "n_documents"] == 2 and bool(events.loc["MC", "intraday"])
    # confidence-weighted: (0.8*0.9 + 0.2*0.3) / 1.2 = 0.65
    assert events.loc["MC", "tone"] == pytest.approx(0.65)
    assert events.loc["OR", "tone"] == pytest.approx(-0.4)


def test_regression_recovers_a_known_slope() -> None:
    rng = np.random.default_rng(4)
    count = 3000
    table = pd.DataFrame(
        {
            "tone": rng.uniform(-1, 1, count),
            "event_session": pd.to_datetime("2021-01-01")
            + pd.to_timedelta(rng.integers(0, 300, count), "D"),
        }
    )
    table["car_0"] = 0.02 * table["tone"] + rng.normal(0, 0.02, count)
    result = study.clustered_regression(table, "car_0", ["tone"])
    assert result.loc["tone", "coefficient"] == pytest.approx(0.02, abs=0.002)
    assert result.loc["tone", "ci_low"] < 0.02 < result.loc["tone", "ci_high"]
    assert result.loc["tone", "n"] == count and result.loc["tone", "clusters"] <= 300


def test_clustering_by_date_widens_the_error_when_days_share_a_shock() -> None:
    """Fifty dates, forty events each; every date has its own common shock and its own tone.

    Treating the 2,000 events as independent overstates the evidence; clustering corrects it.
    """
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2021-01-01", periods=50)
    table = pd.DataFrame({"event_session": np.repeat(dates, 40)})
    day_tone = rng.uniform(-1, 1, 50)
    day_shock = rng.normal(0, 0.02, 50)
    table["tone"] = np.repeat(day_tone, 40) + rng.normal(0, 0.05, 2000)
    table["car_0"] = np.repeat(day_shock, 40) + rng.normal(0, 0.01, 2000)  # no true effect
    clustered = study.clustered_regression(table, "car_0", ["tone"]).loc["tone"]
    naive = study.clustered_regression(table.assign(row=np.arange(2000)), "car_0", ["tone"], "row")
    assert clustered["std_error"] > 3 * naive.loc["tone", "std_error"]
    assert clustered["clusters"] == 50


def test_group_means_and_cumulative_paths() -> None:
    panel, events, _ = synthetic_panel()
    table = mm.abnormal_returns(panel, events)
    table["group"] = "all"
    means = study.mean_car_by_group(table, "group", "car_0")
    assert means.loc[0, "events"] == len(events)
    assert means.loc[0, "mean"] == pytest.approx(table["car_0"].mean())
    assert means.loc[0, "t"] > 5  # a 3% jump is unmistakable
    paths = study.cumulative_paths(table, "group")
    assert list(paths.index) == list(range(-5, 6))
    assert paths.loc[0, "all"] - paths.loc[-1, "all"] == pytest.approx(table["ar_0"].mean())
    assert paths.loc[5, "all"] == pytest.approx(
        table[[f"ar_{d}" for d in range(-5, 6)]].sum(axis=1).mean()
    )


def test_terciles_split_into_three_equal_groups() -> None:
    labels = study.terciles(pd.Series(np.arange(30, dtype=float)))
    assert labels.value_counts().to_dict() == {"low": 10, "middle": 10, "high": 10}
