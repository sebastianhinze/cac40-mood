"""Data-quality tables: small hand-made inputs with known answers."""

import pandas as pd

from cac40_mood.data import quality


def filings_frame() -> pd.DataFrame:
    rows = [
        # doc_id, firm, Paris time, event_session, intraday, admin, class, language
        ("pre", "MC", "2024-01-10 07:30", "2024-01-10", False, False, "inside_information", "fr"),
        ("intra", "MC", "2024-01-10 12:00", "2024-01-10", True, False, "inside_information", "fr"),
        ("late", "OR", "2024-01-10 18:00", "2024-01-11", False, False, "inside_information", "en"),
        ("weekend", "OR", "2024-01-13 10:00", "2024-01-15", False, True, "voting_rights", "fr"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "doc_id",
            "firm_id",
            "ts_paris",
            "event_session",
            "intraday",
            "is_admin",
            "doc_class",
            "lang",
        ],
    )
    frame["ts_paris"] = pd.to_datetime(frame["ts_paris"]).dt.tz_localize("Europe/Paris")
    frame["event_session"] = pd.to_datetime(frame["event_session"])
    return frame


SESSIONS = pd.bdate_range("2024-01-01", "2024-01-31")


def test_timing_categories() -> None:
    timing = quality.timing_category(filings_frame(), SESSIONS)
    assert timing.tolist() == ["pre_open", "intraday", "after_close", "non_trading_day"]


def test_firm_year_counts_include_a_total() -> None:
    table = quality.firm_year_counts(filings_frame())
    assert table.loc["MC", 2024] == 2 and table.loc["OR", "total"] == 2


def test_monthly_counts_split_admin_from_scored() -> None:
    table = quality.monthly_counts(filings_frame())
    assert table.loc["2024-01", "to_score"] == 3 and table.loc["2024-01", "admin"] == 1


def test_class_summary_counts_languages_and_timing_shares() -> None:
    summary = quality.class_summary(filings_frame(), SESSIONS)
    inside = summary.loc["inside_information"]
    assert (inside["documents"], inside["french"], inside["english"]) == (3, 2, 1)
    assert inside["intraday"] == round(1 / 3, 3)
    assert bool(summary.loc["voting_rights", "admin"])


def test_source_overlap_separates_the_three_origins() -> None:
    frame = filings_frame()
    frame["in_api"] = [True, True, False, True]
    frame["in_tarball"] = [True, False, True, True]
    table = quality.source_overlap(frame, "2024-01-01", "2024-02-01")
    assert table.loc["2024-01"].to_dict() == {"both": 2, "api_only": 1, "archive_only": 1}


def test_text_quality_summary_reports_the_usable_share() -> None:
    texts = pd.DataFrame({"doc_id": ["pre", "intra"], "text_quality": ["ok", "empty"]})
    table = quality.text_quality_summary(filings_frame(), texts)
    # Three documents to score: one ok, one empty, one never extracted.
    assert table.loc["ALL", "ok"] == 1
    assert table.loc["ALL", "not_extracted"] == 1
    assert table.loc["ALL", "usable_share"] == round(1 / 3, 4)
