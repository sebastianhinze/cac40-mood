"""Main analysis: document selection, sentiment features without look-ahead, mood on real-style
readings, multiple-test corrections, and the volatility and event-study runs on made-up data."""

import numpy as np
import pandas as pd
import pytest

from cac40_mood import analysis_documents, multiple_tests
from cac40_mood.events import run as events_run
from cac40_mood.mood import real
from cac40_mood.mood.hierarchy import simulate_hierarchy, two_stage_filter
from cac40_mood.mood.model import MoodParameters
from cac40_mood.vol import features, sentiment_run


def make_documents(rows: list[tuple]) -> pd.DataFrame:
    """Documents as the analysis sees them. Each row: firm, day, class, tone, confidence."""
    frame = pd.DataFrame(
        rows, columns=["firm_id", "event_session", "doc_class", "tone", "confidence"]
    )
    frame["event_session"] = pd.to_datetime(frame["event_session"])
    frame["doc_id"] = [f"d{number}" for number in range(len(frame))]
    frame["surprise"] = 0.0
    frame["uncertainty"] = 0.5
    frame["lex_tone"] = 0.6
    frame["lex_uncertainty"] = 1.0
    frame["lm_tone"] = frame["tone"] / 2
    frame["lmd_tone"] = -0.2
    frame["lmd_uncertainty"] = 1.5
    frame["split"] = "train"
    return frame


# ------------------------------------------------------------ selection


def selection_inputs() -> tuple[pd.DataFrame, ...]:
    rows = [
        # doc, firm, timestamp, split, admin, in gap
        ("ok", "MC", "2021-03-01 08:00", "train", False, False),
        ("test", "MC", "2025-03-01 08:00", "test", False, False),
        ("admin", "MC", "2021-03-02 08:00", "train", True, False),
        ("urw", "URW", "2021-03-01 08:00", "train", False, False),
        ("gap", "MC", "2020-04-01 08:00", "train", False, True),
        ("notext", "MC", "2021-03-03 08:00", "train", False, False),
        ("early", "MC", "2019-01-10 08:00", "train", False, False),
    ]
    filings = pd.DataFrame(
        rows, columns=["doc_id", "firm_id", "ts", "split", "is_admin", "in_feed_gap"]
    )
    filings["ts_paris"] = pd.to_datetime(filings["ts"]).dt.tz_localize("Europe/Paris")
    filings["event_session"] = pd.to_datetime(filings["ts"]).dt.normalize()
    filings["intraday"] = False
    filings["doc_class"] = "inside_information"
    scored = filings[filings["doc_id"] != "notext"]
    scores = pd.DataFrame(
        {
            "doc_id": scored["doc_id"],
            "problem": "",
            "tone": 0.3,
            "surprise": 0.0,
            "uncertainty": 0.2,
            "confidence": 0.8,
            "topic": "results",
            "size_fy": 2018,
        }
    )
    lexicon = pd.DataFrame({"doc_id": filings["doc_id"], "lex_tone": 0.5, "lex_uncertainty": 1.0})
    local_model = pd.DataFrame({"doc_id": filings["doc_id"], "lm_tone": 0.1})
    revenue = pd.DataFrame(
        {
            "firm_id": ["MC", "URW"],
            "fy": [2018, 2018],
            "revenue": [46826.0, 2000.0],
            # The 2018 figure was read from the release of 29 January 2019.
            "published": pd.to_datetime(["2019-01-29 18:00", "2019-02-13 18:00"]).tz_localize(
                "Europe/Paris"
            ),
        }
    )
    return filings, scores, lexicon, local_model, revenue


def test_selection_applies_every_rule_and_counts_what_each_removed() -> None:
    documents, removed = analysis_documents.select_documents(*selection_inputs())
    assert list(documents["doc_id"]) == ["ok"]
    assert "test" not in set(documents["split"])
    counts = removed.set_index("rule")["documents_removed"]
    assert counts["administrative notice"] == 1
    assert counts["firm outside the main analysis"] == 1
    assert counts["inside the spring-2020 feed hole"] == 1
    assert counts["no valid LLM score (no usable text)"] == 1
    # "early" was scored with a revenue figure published 19 days after it.
    assert counts["firm-size figure from a later filing"] == 1
    assert counts["documents used"] == 1
    assert documents["revenue"].iloc[0] == 46826.0


# ------------------------------------------------------------- features


def firm_days(start: str, end: str, firms: tuple[str, ...] = ("MC",)) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    return pd.DataFrame(
        [(firm, date) for firm in firms for date in dates], columns=["firm_id", "date"]
    )


def test_day_features_weight_tone_by_confidence_and_mark_the_announcement_type() -> None:
    documents = make_documents(
        [
            ("MC", "2021-03-02", "inside_information", 0.8, 0.9),
            ("MC", "2021-03-02", "periodic_report", -0.4, 0.3),
            ("MC", "2021-03-04", "takeover", -0.5, 1.0),
        ]
    )
    table = features.attach_features(firm_days("2021-03-01", "2021-03-05"), documents)
    table = table.set_index("date")
    busy = table.loc["2021-03-02"]
    expected_tone = (0.9 * 0.8 + 0.3 * -0.4) / (0.9 + 0.3)
    assert busy["abs_tone"] == pytest.approx(expected_tone)
    assert busy["neg_tone"] == 0.0
    assert (busy["ev_news"], busy["ev_report"], busy["ev_other"]) == (1.0, 1.0, 0.0)
    bad = table.loc["2021-03-04"]
    assert (bad["abs_tone"], bad["neg_tone"], bad["ev_other"]) == (0.5, 0.5, 1.0)
    # A day without a document: every column is 0, not missing.
    assert (table.loc["2021-03-03", features.ALL_COLUMNS] == 0.0).all()


def test_unknown_days_are_missing_not_zero() -> None:
    documents = make_documents([("MC", "2020-02-20", "inside_information", 0.5, 0.9)])
    frame = pd.concat(
        [firm_days("2018-12-27", "2019-01-03"), firm_days("2020-03-02", "2020-03-06")]
    )
    table = features.attach_features(frame, documents).set_index("date")
    assert table.loc["2018-12-28", features.ALL_COLUMNS].isna().all()  # before the sample
    assert table.loc["2019-01-02", features.ALL_COLUMNS].notna().all()
    assert table.loc["2020-03-04", features.ALL_COLUMNS].notna().all()
    assert table.loc["2020-03-05", features.ALL_COLUMNS].isna().all()  # first day of the feed hole


def test_a_later_document_changes_no_earlier_feature() -> None:
    """The no-future-information test: add news on 10 March, compare everything before it."""
    documents = make_documents(
        [
            ("MC", "2021-03-02", "inside_information", 0.8, 0.9),
            ("OR", "2021-03-08", "quarterly_information", -0.2, 0.6),
        ]
    )
    later = make_documents([("MC", "2021-03-10", "inside_information", -0.9, 1.0)])
    frame = firm_days("2021-03-01", "2021-03-12", ("MC", "OR"))
    before = features.attach_features(frame, documents)
    after = features.attach_features(frame, pd.concat([documents, later], ignore_index=True))
    past = before["date"] < pd.Timestamp("2021-03-10")
    pd.testing.assert_frame_equal(before[past], after[past])
    changed = after[(after["firm_id"] == "MC") & (after["date"] == pd.Timestamp("2021-03-10"))]
    assert changed["abs_tone"].iloc[0] == 0.9


# ----------------------------------------------------------------- mood


def test_readings_are_centred_and_placed_on_the_trading_calendar() -> None:
    dates = pd.Series(pd.to_datetime(["2021-03-01", "2021-03-02", "2021-03-04"]))  # 3 March closed
    documents = make_documents([("MC", "2021-03-04", "inside_information", 0.5, 0.7)])
    readings = real.make_readings(documents, dates, centre=0.1)
    assert readings.iloc[0].to_dict() == {"day": 2, "firm": "MC", "value": 0.4, "confidence": 0.7}
    off_calendar = make_documents([("MC", "2021-03-03", "inside_information", 0.5, 0.7)])
    with pytest.raises(ValueError, match="not in the trading calendar"):
        real.make_readings(off_calendar, dates, centre=0.1)


def test_filtered_mood_does_not_change_when_later_readings_are_added() -> None:
    market, firm = MoodParameters(0.95, 0.05, 0.2), MoodParameters(0.98, 0.03, 0.2)
    _, _, readings = simulate_hierarchy(market, firm, n_firms=3, n_days=120, seed=3)
    early = readings[readings["day"] < 80]
    full = two_stage_filter(readings, 120, market, firm, 500, seed=1)
    cut = two_stage_filter(early, 120, market, firm, 500, seed=1)
    np.testing.assert_allclose(full.total_mean.iloc[:80], cut.total_mean.iloc[:80])
    assert not np.allclose(full.total_mean.iloc[80:], cut.total_mean.iloc[80:])


def test_two_stage_fit_recovers_the_rough_size_of_the_parameters() -> None:
    market, firm = MoodParameters(0.95, 0.06, 0.25), MoodParameters(0.97, 0.05, 0.25)
    _, _, readings = simulate_hierarchy(market, firm, 8, 500, seed=11, readings_per_day=0.3)
    fitted_market, fitted_firm = real.fit_two_stage(readings, 500, np.array([], dtype=int))
    assert 0.85 < fitted_market.phi < 0.995
    assert 0.9 < fitted_firm.phi < 0.999
    assert fitted_firm.sigma_obs == pytest.approx(0.25, rel=0.25)
    assert fitted_market.sigma_state == pytest.approx(0.06, rel=0.6)


# ------------------------------------------------------- multiple tests


def test_holm_and_benjamini_hochberg_on_a_hand_example() -> None:
    p_values = [0.01, 0.04, 0.03]
    # Holm: 0.01*3, then 0.03*2, then 0.04*1 but never below the one before it.
    assert multiple_tests.holm(p_values) == pytest.approx([0.03, 0.06, 0.06])
    # BH: 0.01*3/1, 0.03*3/2 = 0.045 but capped by the next one, 0.04*3/3.
    assert multiple_tests.benjamini_hochberg(p_values) == pytest.approx([0.03, 0.04, 0.04])
    assert multiple_tests.holm([0.6, 0.9]) == pytest.approx([1.0, 1.0])


# ------------------------------------------------------- volatility run


def simulated_inputs(seed: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three firms, 2021 to mid-2024. An announcement raises the next days' variance, and
    more so the stronger its tone; the mood columns are pure noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", "2024-06-28")
    panels, documents, moods = [], [], []
    for firm in ("F0", "F1", "F2"):
        event = rng.random(len(dates)) < 0.08
        tone = np.where(event, rng.uniform(-1, 1, len(dates)), 0.0)
        log_rv = np.full(len(dates), -9.0)
        for day in range(1, len(dates)):
            push = 2.5 * abs(tone[day - 1])
            log_rv[day] = -9.0 + 0.6 * (log_rv[day - 1] + 9.0) + push + rng.normal(0, 0.3)
        panels.append(pd.DataFrame({"firm_id": firm, "date": dates, "rv": np.exp(log_rv)}))
        documents += [
            (firm, date, "inside_information", value, 0.8)
            for date, value, happened in zip(dates, tone, event, strict=True)
            if happened
        ]
        noise = rng.normal(0, 0.1, len(dates))
        moods.append(
            pd.DataFrame({"firm_id": firm, "date": dates, "mood_total": noise, "mood_market": 0.0})
        )
    return pd.concat(panels, ignore_index=True), make_documents(documents), pd.concat(moods)


def test_scores_help_when_they_truly_matter_and_noise_mood_does_not() -> None:
    panel, documents, mood = simulated_inputs()
    frame = sentiment_run.build_frame(panel, documents, mood)
    against_b1, mood_against_m, _ = sentiment_run.comparison_tables(frame)
    one_day = against_b1[against_b1["horizon"] == 1].set_index("model")
    assert one_day.loc["M", "dm_statistic"] < -3  # |tone| carries real information here
    assert one_day.loc["M", "ci_high"] < 0
    noise = mood_against_m[mood_against_m["horizon"] == 1].set_index("model")
    assert abs(noise.loc["M2", "dm_statistic"]) < 2.5  # noise columns: no reliable gain
    primary = sentiment_run.primary_tests(against_b1, mood_against_m)
    assert list(primary["test"]) == ["RQ3: M vs B1", "RQ4: M2 vs M"]
    assert (primary["holm_p_value"] >= primary["dm_p_value"]).all()


# ---------------------------------------------------------- event study


def simulated_events(seed: int = 9, count: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tone = rng.uniform(-1, 1, count)
    return pd.DataFrame(
        {
            "firm_id": "MC",
            "event_session": pd.bdate_range("2021-01-01", periods=count),
            "intraday": rng.random(count) < 0.2,
            "intraday_flag": (rng.random(count) < 0.2).astype(float),
            "tone": tone,
            "surprise": rng.uniform(-1, 1, count),
            "uncertainty": rng.uniform(0, 1, count),
            "lm_tone": tone + rng.normal(0, 1.0, count),
            "lex_tone": rng.uniform(-1, 1, count),
            "lmd_tone": rng.uniform(-1, 1, count),
            "residual_sd": rng.uniform(0.01, 0.03, count),
            "log_revenue": rng.uniform(8.0, 12.0, count),
            "topic": rng.choice(["results", "financing", "other"], count, p=[0.6, 0.3, 0.1]),
            "car_0": 0.01 * tone + rng.normal(0, 0.01, count),
            "car_1_5": rng.normal(0, 0.02, count),
            "car_placebo": rng.normal(0, 0.02, count),
        }
    )


def test_topic_helpers() -> None:
    documents = make_documents(
        [
            ("MC", "2021-03-02", "inside_information", 0.8, 0.4),
            ("MC", "2021-03-02", "inside_information", 0.1, 0.9),
        ]
    )
    documents["topic"] = ["financing", "results"]
    assert events_run.main_topic(documents)["topic"].tolist() == ["results"]  # the confident one
    table, columns = events_run.with_topic_dummies(simulated_events())
    assert columns == ["topic_financing", "topic_other"]  # "results" is the comparison group
    assert table["topic_financing"].isin([0.0, 1.0]).all()


def test_regressions_find_the_tone_effect_and_rank_the_scorers() -> None:
    events = simulated_events()
    scores = events_run.score_regressions(events)
    day_zero = scores[(scores["outcome"] == "CAR[0] (RQ1)") & (scores["regressor"] == "tone")]
    assert day_zero["coefficient"].tolist() == pytest.approx([0.01, 0.01], abs=0.002)
    drift = scores[scores["outcome"] == "CAR[+1,+5] (RQ2)"]
    assert (drift["bh_p_value"] > 0.05).all()  # no drift was put in
    assert (scores["bh_p_value"] >= scores["p_value"] - 1e-12).all()
    scorers = events_run.scorer_comparison(events).set_index("scorer")
    assert scorers.loc["LLM", "r_squared"] > scorers.loc["Local model", "r_squared"]
    assert scorers.loc["Local model", "r_squared"] > scorers.loc["Lexicon", "r_squared"]


def test_gold_check_matches_each_labelled_document_to_its_firm_day() -> None:
    events = simulated_events(count=40).assign(n_documents=1)
    filings = events[["firm_id", "event_session"]].assign(doc_id=[f"g{n}" for n in range(40)])
    scores = pd.DataFrame({"doc_id": filings["doc_id"], "tone": events["tone"]})
    labels = pd.DataFrame({"doc_id": filings["doc_id"], "tone": np.sign(events["tone"]) * 2})
    result = events_run.gold_check(events, labels, filings, scores).set_index("scorer")
    assert result.loc["LLM tone", "n"] == 40
    assert result.loc["LLM tone", "coefficient"] > 0
    assert result.loc["My label", "spearman_with_car_0"] > 0
