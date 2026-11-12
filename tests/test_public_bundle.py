"""The committed public bundle: what it may and may not contain (DECISIONS #67, #69).

These run on the files in `public_data/`, so they also protect against a rebuild that adds a
column it should not (filing text, prices) or drops the attribution.
"""

import json

import pandas as pd

from cac40_mood import dashboard_data

BUNDLE = dashboard_data.BUNDLE_DIR

ANNOUNCEMENT_COLUMNS = {
    "doc_id",
    "firm_id",
    "company",
    "published_paris",
    "event_session",
    "intraday",
    "period",
    "lang",
    "title",
    "topic",
    "tone",
    "surprise",
    "uncertainty",
    "confidence",
    "rationale",
    "pdf_url",
}


def test_announcements_hold_metadata_and_scores_only():
    table = pd.read_parquet(BUNDLE / "announcements.parquet")
    assert set(table.columns) == ANNOUNCEMENT_COLUMNS  # no text, no excerpt, no prices
    assert table["doc_id"].is_unique
    assert table["firm_id"].nunique() == 36
    assert table["tone"].between(-1, 1).all()
    assert table["pdf_url"].str.startswith("https://fr.ftp.opendatasoft.com/datadila/").all()


def test_mood_covers_the_sample_with_a_band():
    mood = pd.read_parquet(BUNDLE / "mood.parquet")
    assert mood["firm_id"].nunique() == 36
    assert mood["date"].min() == pd.Timestamp("2019-01-02")
    assert mood["date"].max() == pd.Timestamp("2025-12-31")
    assert (mood["mood_low"] <= mood["mood_total"]).all()
    assert (mood["mood_total"] <= mood["mood_high"]).all()
    test_part = mood["date"] > pd.Timestamp("2024-06-30")
    assert (mood.loc[test_part, "period"] == "test (display only)").all()


def test_no_price_columns_anywhere():
    for path in BUNDLE.glob("*.csv"):
        columns = {name.lower() for name in pd.read_csv(path).columns}
        # "low" / "high" are tone thirds here, so only unambiguous price names are checked.
        assert not columns & {"open", "close", "adj close", "volume", "price", "return"}


def test_manifest_carries_the_attribution():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    assert "Licence Ouverte" in manifest["attribution"]
    assert "DILA" in manifest["attribution"]
    assert manifest["rows"]["announcements.parquet"] == 6860


def test_dashboard_reads_the_bundle():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    assert dashboard_data.tone_centre() == manifest["tone_centre_train"]
    paths = dashboard_data.car_paths_by_tone()
    assert list(paths.columns) == ["low", "middle", "high"]
    assert list(paths.index) == list(range(-5, 6))
