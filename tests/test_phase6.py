"""Phase 6 runner, tested on made-up data only. No test here reads a real file."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from cac40_mood import analysis_documents, phase6
from cac40_mood import phase6_config as config
from cac40_mood.events import study
from cac40_mood.splits import VAL_END
from cac40_mood.vol import sentiment_run
from cac40_mood.vol.realized import variance_panel

MADE_UP = phase6.Period("made-up", date(2024, 7, 1), date(2025, 12, 31), ("train", "val", "test"))


def made_up_frame(seed: int = 6) -> pd.DataFrame:
    """Three firms, 2021 to 2025. |tone| of an announcement raises the next days' variance;
    the mood columns are noise. Built with the same frame builder as the real run."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", "2025-12-31")
    panels, documents, moods = [], [], []
    for firm in ("F0", "F1", "F2"):
        event = rng.random(len(dates)) < 0.08
        tone = np.where(event, rng.uniform(-1, 1, len(dates)), 0.0)
        log_rv = np.full(len(dates), -9.0)
        for day in range(1, len(dates)):
            push = 2.5 * abs(tone[day - 1])
            log_rv[day] = -9.0 + 0.6 * (log_rv[day - 1] + 9.0) + push + rng.normal(0, 0.3)
        panels.append(pd.DataFrame({"firm_id": firm, "date": dates, "rv": np.exp(log_rv)}))
        for day in np.flatnonzero(event):
            documents.append(
                {
                    "firm_id": firm,
                    "event_session": dates[day],
                    "doc_class": "inside_information",
                    "tone": tone[day],
                    "confidence": 0.8,
                    "surprise": 0.0,
                    "uncertainty": 0.5,
                    "lex_tone": 0.6,
                    "lex_uncertainty": 1.0,
                    "lm_tone": tone[day] / 2,
                    "lmd_tone": 0.1,
                    "lmd_uncertainty": 1.0,
                }
            )
        noise = rng.normal(0, 0.1, (len(dates), 5))
        moods.append(
            pd.DataFrame(
                {
                    "firm_id": firm,
                    "date": dates,
                    "mood_total": noise[:, 0],
                    "mood_market": 0.0,
                    "sv_mood": noise[:, 1],
                    "sv_log_forecast_1": noise[:, 2],
                    "sv_log_forecast_5": noise[:, 3],
                    "sv_log_forecast_22": noise[:, 4],
                }
            )
        )
    mood = pd.concat(moods, ignore_index=True)
    frame = sentiment_run.build_frame(pd.concat(panels), pd.DataFrame(documents), mood)
    sv_columns = ["firm_id", "date", "sv_mood", *[f"sv_log_forecast_{h}" for h in (1, 5, 22)]]
    return frame.merge(mood[sv_columns], on=["firm_id", "date"])


@pytest.fixture(scope="module")
def results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return phase6.evaluate_models(made_up_frame(), MADE_UP)


def test_primary_family_is_exactly_the_two_pre_registered_tests(results) -> None:
    primary, exploratory, _ = results
    assert list(zip(primary["model"], primary["against"], strict=True)) == [
        ("M", "B1"),
        ("M3", "M"),
    ]
    assert (primary["horizon"] == 5).all()
    assert (primary["holm_p_value"] >= primary["dm_p_value"] - 1e-12).all()
    # Every other comparison is exploratory: 8 comparisons x 3 horizons minus the 2 primary ones.
    assert len(exploratory) == 22
    assert (exploratory["bh_p_value"] >= exploratory["dm_p_value"] - 1e-12).all()


def test_confirmed_needs_a_small_adjusted_p_value_and_a_better_model(results) -> None:
    primary, _, _ = results
    rq3, rq4 = primary.iloc[0], primary.iloc[1]
    # |tone| truly drives the made-up variance: RQ3 must be confirmed.
    assert rq3["qlike_minus_baseline"] < 0 and rq3["holm_p_value"] < 0.05 and rq3["confirmed"]
    # The made-up mood columns are noise: RQ4 must not be confirmed.
    assert not rq4["confirmed"]
    expected = (primary["holm_p_value"] < config.SIGNIFICANCE_LEVEL) & (
        primary["qlike_minus_baseline"] < 0
    )
    assert (primary["confirmed"] == expected).all()


def test_forecasts_start_at_the_period_and_never_need_a_later_day() -> None:
    frame = made_up_frame()
    forecasts = phase6.forecasts_for(frame, MADE_UP, 5)["M"]
    assert forecasts["date"].min() >= pd.Timestamp(MADE_UP.first_origin)
    # The last origin lies 5 trading days before the last day: its target window ends inside.
    last_days = np.sort(frame["date"].unique())
    assert forecasts["date"].max() == last_days[-6]


def test_model_columns_follow_the_frozen_config() -> None:
    models = phase6.models_for(config.PRIMARY_HORIZON)
    assert models["M3"] == config.COLUMNS["M3"]
    assert models["M"] == config.COLUMNS["M"]
    assert models["B1"] == config.COLUMNS["B1"]
    assert phase6.PRIMARY == (("M", "B1"), ("M3", "M"))


def test_rehearsal_cannot_reach_the_test_period() -> None:
    assert phase6.REHEARSAL_PERIOD.last_day == VAL_END
    assert "test" not in phase6.REHEARSAL_PERIOD.splits
    assert phase6.TEST_PERIOD.first_origin == date(2024, 7, 1)


def test_the_test_run_is_refused_unless_frozen_clean_and_first(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "06_test_results.md"
    answers = {"status": "", "rev-parse": "abc", "rev-list": "abc"}
    monkeypatch.setattr(phase6, "git_output", lambda *arguments: answers[arguments[0]])
    assert phase6.check_frozen(missing) == "abc"

    answers["status"] = " M src/cac40_mood/phase6.py"
    with pytest.raises(RuntimeError, match="Uncommitted"):
        phase6.check_frozen(missing)
    answers["status"] = ""

    answers["rev-list"] = "def"
    with pytest.raises(RuntimeError, match="not the commit tagged"):
        phase6.check_frozen(missing)
    answers["rev-list"] = "abc"

    missing.write_text("already there")
    with pytest.raises(RuntimeError, match="already been run"):
        phase6.check_frozen(missing)


def test_end_date_parameters_default_to_the_end_of_validation() -> None:
    dates = pd.to_datetime(["2024-06-27", "2024-06-28", "2024-07-01", "2024-07-02"])
    prices = pd.DataFrame(
        {"firm_id": "MC", "date": dates, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0}
    ).assign(adj_close=[100.0, 101.0, 102.0, 103.0])
    assert variance_panel(prices, ["MC"])["date"].max() == pd.Timestamp("2024-06-28")
    assert variance_panel(prices, ["MC"], date(2025, 12, 31))["date"].max() == dates[-1]

    documents = pd.DataFrame(
        {
            "firm_id": "MC",
            "event_session": pd.to_datetime(["2024-06-03", "2024-09-03"]),
            "split": ["val", "test"],
            "is_admin": False,
            "in_feed_gap": False,
            "intraday": False,
            "tone": [0.2, 0.4],
        }
    )
    assert len(study.prepare_events(documents, ["tone"])) == 1
    assert len(study.prepare_events(documents, ["tone"], ("train", "val", "test"))) == 2
    assert analysis_documents.TUNING_SPLITS == ("train", "val")
