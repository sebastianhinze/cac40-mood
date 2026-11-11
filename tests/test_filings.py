"""Raw filing records -> clean table: timestamps, classes, translations, source merge."""

from datetime import date

import pandas as pd
import pytest

from cac40_mood.data import filings as fl
from cac40_mood.data.schema import FILING_FIELDS


def make_record(**overrides: str) -> dict[str, str]:
    """A complete filing record with sensible defaults; tests override what they care about."""
    record = dict.fromkeys(FILING_FIELDS, "")
    record.update(
        doc_id="FCMKW000001_20240125",
        source="api",
        pdf_path="MKW/2024/01/FCMKW000001_20240125.pdf",
        isin="FR0000121014",
        lang="fr",
        type_code="001007",
        ts_amf="2024-01-25T16:45:52+00:00",
        ts_market="2024-01-25T16:45:52+00:00",
        ts_emit="2024-01-25T16:45:52+00:00",
    )
    record.update(overrides)
    return record


# ---------------------------------------------------------------- timestamps


def test_timestamps_with_any_offset_become_utc() -> None:
    parsed = fl.parse_feed_timestamp(
        pd.Series(["2019-07-25T14:33:03+02:00", "2024-01-25T16:45:52+00:00"])
    )
    assert parsed[0] == pd.Timestamp("2019-07-25 12:33:03", tz="UTC")
    assert parsed[1] == pd.Timestamp("2024-01-25 16:45:52", tz="UTC")


@pytest.mark.parametrize(
    "placeholder",
    ["8887-12-31T23:00:00+00:00", "8888-01-01T00:00:00+02:00", "", None, "not a date"],
)
def test_placeholder_and_broken_timestamps_become_missing(placeholder) -> None:
    assert fl.parse_feed_timestamp(pd.Series([placeholder])).isna().all()


def test_conservative_timestamp_is_the_latest_one() -> None:
    """Deposited at 14:33 Paris with a market release at 18:00: the event time is 18:00."""
    frame = pd.DataFrame(
        [
            make_record(
                ts_amf="2019-07-25T14:33:03+02:00",
                ts_market="2019-07-25T18:00:00+02:00",
                ts_emit="2019-07-25T14:33:03+02:00",
            )
        ]
    )
    result = fl.conservative_timestamp(frame).iloc[0]
    assert result["ts_utc"] == pd.Timestamp("2019-07-25 16:00:00", tz="UTC")
    assert result["ts_spread_minutes"] == pytest.approx(206.95)


def test_placeholder_never_wins_the_latest_rule() -> None:
    frame = pd.DataFrame([make_record(ts_market="8887-12-31T23:00:00+00:00")])
    result = fl.conservative_timestamp(frame).iloc[0]
    assert result["ts_utc"] == pd.Timestamp("2024-01-25 16:45:52", tz="UTC")
    assert result["ts_spread_minutes"] == 0.0


def test_row_without_any_valid_timestamp_stays_missing() -> None:
    frame = pd.DataFrame([make_record(ts_amf="", ts_market="", ts_emit="")])
    assert fl.conservative_timestamp(frame)["ts_utc"].isna().all()


# ------------------------------------------------------------ classification


@pytest.mark.parametrize(
    ("type_code", "expected_class", "expected_admin"),
    [
        ("001007", "inside_information", False),
        ("015000", "inside_information", False),
        ("005001", "quarterly_information", False),
        ("004000", "periodic_report", False),
        ("300005", "periodic_report", False),
        ("008000", "governance", False),
        ("009007", "takeover", False),
        ("006000", "voting_rights", True),
        ("013006", "own_shares", True),
        ("023000", "threshold_crossing", True),
        ("002009", "availability_notice", True),
        ("020000", "debt_terms", True),
        ("014000", "other", False),
        ("", "other", False),
    ],
)
def test_type_codes_map_to_classes(type_code, expected_class, expected_admin) -> None:
    doc_class = fl.classify_type_code(type_code)
    assert doc_class == expected_class
    assert (doc_class in fl.ADMIN_CLASSES) is expected_admin


# -------------------------------------------------------------- translations


def translation_frame(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """Rows of (doc_id, firm_id, type_code, lang, UTC time)."""
    frame = pd.DataFrame(rows, columns=["doc_id", "firm_id", "type_code", "lang", "ts_utc"])
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True)
    return frame


def test_english_twin_of_a_french_filing_is_flagged() -> None:
    frame = translation_frame(
        [
            ("fr1", "MC", "001007", "fr", "2024-01-25 16:45:52"),
            ("en1", "MC", "001007", "en", "2024-01-25 16:46:20"),
        ]
    )
    flags = fl.flag_translations(frame)
    assert flags["is_translation"].tolist() == [False, True]
    assert flags.loc[1, "twin_doc_id"] == "fr1"


def test_two_releases_sent_together_stay_two_documents() -> None:
    """Two French and two English filings within a minute: both English ones are translations."""
    frame = translation_frame(
        [
            ("fr_accounts", "SU", "001007", "fr", "2019-07-25 05:00:00"),
            ("en_accounts", "SU", "001007", "en", "2019-07-25 05:00:00"),
            ("fr_results", "SU", "001007", "fr", "2019-07-25 05:00:01"),
            ("en_results", "SU", "001007", "en", "2019-07-25 05:00:20"),
        ]
    )
    flags = fl.flag_translations(frame)
    assert flags["is_translation"].tolist() == [False, True, False, True]
    assert set(flags["twin_doc_id"].dropna()) == {"fr_accounts", "fr_results"}


def test_one_french_filing_absorbs_only_one_english_filing() -> None:
    frame = translation_frame(
        [
            ("fr1", "AC", "001006", "fr", "2024-03-01 07:00:00"),
            ("en_far", "AC", "001006", "en", "2024-03-01 07:10:00"),
            ("en_near", "AC", "001006", "en", "2024-03-01 07:00:30"),
        ]
    )
    flags = fl.flag_translations(frame)
    assert flags.set_index(frame["doc_id"])["is_translation"].to_dict() == {
        "fr1": False,
        "en_far": False,
        "en_near": True,
    }


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"firm_id": "OR"}, "another firm"),
        ({"type_code": "001006"}, "another type code"),
        ({"ts_utc": "2024-01-25 17:30:00"}, "more than 30 minutes apart"),
        ({"lang": "fr"}, "also French"),
    ],
)
def test_filings_that_are_not_translations_are_kept(change, reason) -> None:
    second = {
        "doc_id": "second",
        "firm_id": "MC",
        "type_code": "001007",
        "lang": "en",
        "ts_utc": "2024-01-25 16:46:00",
    }
    second.update(change)
    frame = translation_frame(
        [("first", "MC", "001007", "fr", "2024-01-25 16:45:52"), tuple(second.values())]
    )
    assert not fl.flag_translations(frame)["is_translation"].any(), reason


def test_english_only_firms_keep_their_filings() -> None:
    frame = translation_frame(
        [
            ("en1", "ENX", "001006", "en", "2024-02-01 07:00:00"),
            ("en2", "ENX", "001006", "en", "2024-02-01 07:00:05"),
        ]
    )
    assert not fl.flag_translations(frame)["is_translation"].any()


# ------------------------------------------------------------- source merge


def test_archive_copy_wins_and_origin_is_remembered() -> None:
    api = [make_record(doc_id="both", title="from api"), make_record(doc_id="api_only")]
    tarball = [
        make_record(doc_id="both", source="tarball", title="from archive"),
        make_record(doc_id="tarball_only", source="tarball"),
    ]
    combined = fl.combine_sources(api, tarball).set_index("doc_id")
    assert len(combined) == 3
    assert combined.loc["both", "title"] == "from archive"
    assert combined.loc["both", ["in_api", "in_tarball"]].tolist() == [True, True]
    assert combined.loc["api_only", ["in_api", "in_tarball"]].tolist() == [True, False]
    assert combined.loc["tarball_only", ["in_api", "in_tarball"]].tolist() == [False, True]


def test_combining_works_when_one_source_is_empty() -> None:
    assert len(fl.combine_sources([make_record()], [])) == 1
    assert len(fl.combine_sources([], [make_record(source="tarball")])) == 1


# ------------------------------------------------------------- whole pipeline


@pytest.fixture
def built() -> pd.DataFrame:
    """A small end-to-end run on hand-made records (calendar: weekdays of 2024)."""
    records = [
        # LVMH annual results, 17:45 Paris -> next session; with an English twin.
        make_record(doc_id="fr_results"),
        make_record(doc_id="en_results", lang="en", ts_amf="2024-01-25T16:46:10+00:00"),
        # Michelin filing under its pre-split ISIN, intraday.
        make_record(doc_id="old_isin", isin="FR0000121261", ts_amf="2024-02-06T10:00:00+00:00"),
        # Voting-rights form: admin.
        make_record(doc_id="admin", type_code="006000", ts_amf="2024-02-07T07:00:00+00:00"),
        # Another company's filing: must be dropped.
        make_record(doc_id="stranger", isin="FR0000031775"),
        # Outside the sample window.
        make_record(doc_id="too_early", ts_amf="2023-12-29T10:00:00+00:00"),
        # 23:30 UTC on 31 Dec is 00:30 on 1 Jan in Paris: outside a window ending 31 Dec.
        make_record(doc_id="paris_next_year", ts_amf="2024-12-31T23:30:00+00:00"),
    ]
    for record in records:
        record["ts_market"] = record["ts_emit"] = record["ts_amf"]
    sessions = pd.bdate_range("2024-01-01", "2024-12-31")
    isin_to_firm = {"FR0000121014": "MC", "FR001400AJ45": "ML", "FR0000121261": "ML"}
    combined = fl.combine_sources(records, [])
    return fl.build_filings(
        combined, isin_to_firm, sessions, date(2024, 1, 1), date(2024, 12, 31)
    ).set_index("doc_id")


def test_pipeline_keeps_only_our_firms_inside_the_window(built) -> None:
    assert set(built.index) == {"fr_results", "en_results", "old_isin", "admin"}


def test_pipeline_maps_previous_isin_to_the_firm(built) -> None:
    assert built.loc["old_isin", "firm_id"] == "ML"


def test_pipeline_aligns_to_sessions_in_paris_time(built) -> None:
    assert built.loc["fr_results", "ts_paris"].hour == 17  # 16:45 UTC is 17:45 in January
    assert built.loc["fr_results", "event_session"] == pd.Timestamp("2024-01-26")
    assert not built.loc["fr_results", "intraday"]
    assert built.loc["old_isin", "event_session"] == pd.Timestamp("2024-02-06")
    assert built.loc["old_isin", "intraday"]
    assert built.loc["old_isin", "event_session_conservative"] == pd.Timestamp("2024-02-07")


def test_pipeline_flags_admin_translations_and_split(built) -> None:
    assert built.loc["admin", "is_admin"]
    assert not built.loc["fr_results", "is_admin"]
    assert built.loc["en_results", "is_translation"]
    assert built.loc["en_results", "twin_doc_id"] == "fr_results"
    assert (built["split"] == "val").all()  # Jan-Feb 2024 lies in the validation period
    kept = fl.keep_documents(built.reset_index())
    assert set(kept["doc_id"]) == {"fr_results", "old_isin", "admin"}


def test_feed_gap_flag_marks_the_spring_2020_window() -> None:
    records = [
        make_record(doc_id="before", ts_amf="2020-03-04T07:00:00+00:00"),
        make_record(doc_id="inside", ts_amf="2020-04-23T07:00:00+00:00"),
        make_record(doc_id="after", ts_amf="2020-06-16T07:00:00+00:00"),
    ]
    for record in records:
        record["ts_market"] = record["ts_emit"] = record["ts_amf"]
    built = fl.build_filings(
        fl.combine_sources(records, []),
        {"FR0000121014": "MC"},
        pd.bdate_range("2020-01-01", "2020-12-31"),
        date(2020, 1, 1),
        date(2020, 12, 31),
    ).set_index("doc_id")
    assert built["in_feed_gap"].to_dict() == {"before": False, "inside": True, "after": False}


def test_feed_is_observed_marks_the_hole_as_missing_data() -> None:
    days = pd.Series(pd.to_datetime(["2020-03-04", "2020-03-05", "2020-06-15", "2020-06-16", None]))
    assert fl.feed_is_observed(days).tolist() == [True, False, False, True, False]
