"""Prices, PDF text, the universe file, the split labels and the polite HTTP helper."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from cac40_mood import splits
from cac40_mood.data import pdf_text, polite_http, universe
from cac40_mood.data import prices as px

# ------------------------------------------------------------------- prices


def yahoo_like_table() -> pd.DataFrame:
    """Three days in Yahoo's layout; the middle day has no close (the stock did not trade)."""
    return pd.DataFrame(
        {
            "Adj Close": [95.0, np.nan, 97.0],
            "Close": [100.0, np.nan, 102.0],
            "High": [101.0, np.nan, 103.0],
            "Low": [99.0, np.nan, 100.5],
            "Open": [99.5, np.nan, 101.0],
            "Volume": [1000, 0, 1200],
        },
        index=pd.Index(["2024-01-03", "2024-01-04", "2024-01-05"], name="Date"),
    )


def test_tidy_prices_renames_sorts_and_drops_untraded_days() -> None:
    tidy = px.tidy_prices(yahoo_like_table().iloc[::-1], "MC")
    assert list(tidy.columns) == ["firm_id", "date", *px.PRICE_COLUMNS]
    assert tidy["date"].tolist() == list(pd.to_datetime(["2024-01-03", "2024-01-05"]))
    assert tidy["adj_close"].tolist() == [95.0, 97.0]
    assert (tidy["firm_id"] == "MC").all()


def test_price_problems_catch_impossible_bars() -> None:
    tidy = px.tidy_prices(yahoo_like_table(), "MC")
    assert px.price_problems(tidy).empty
    broken = tidy.copy()
    broken.loc[0, "high"] = 98.0  # below the close of the same day
    broken.loc[1, "low"] = 0.0
    problems = px.price_problems(broken)
    assert problems["high_below_other_price"].tolist() == [True, False]
    assert problems["non_positive_price"].tolist() == [False, True]


def test_missing_sessions_are_counted_from_the_first_price_on() -> None:
    sessions = pd.bdate_range("2024-01-01", "2024-01-10")
    tidy = px.tidy_prices(yahoo_like_table(), "MC")
    report = px.missing_sessions(tidy, sessions).iloc[0]
    # From 3 Jan to 10 Jan there are 6 sessions; prices exist for 2 of them.
    assert report["n_missing"] == 4
    assert report["first_day"] == pd.Timestamp("2024-01-03")


# ----------------------------------------------------------------- PDF text


def minimal_pdf(text: str) -> bytes:
    """Hand-written one-page PDF that shows `text` in Helvetica."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for number, content in enumerate(objects, start=1):
        offsets.append(len(body))
        body += str(number).encode() + b" 0 obj\n" + content + b"\nendobj\n"
    xref_at = len(body)
    body += b"xref\n0 6\n0000000000 65535 f \n"
    body += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    body += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_at).encode() + b"\n%%EOF"
    return body


def test_text_is_extracted_from_a_real_pdf_file(tmp_path) -> None:
    path = tmp_path / "release.pdf"
    path.write_bytes(minimal_pdf("Chiffre en croissance de 12 %"))
    extracted = pdf_text.extract_text(path)
    assert extracted.text == "Chiffre en croissance de 12 %"
    assert (extracted.n_pages, extracted.pages_read) == (1, 1)


def test_clean_text_normalises_whitespace_without_touching_words() -> None:
    raw = (
        "Résultats\N{NO-BREAK SPACE}annuels \r\n\r\n\r\n\r\n"
        "  croissance   de 12 %\ufffe\x0cPage 2\x00"
    )
    assert pdf_text.clean_text(raw) == "Résultats annuels\n\ncroissance de 12 %\nPage 2"


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("", "empty"),
        ("Trop court.", "empty"),
        ("Le chiffre d'affaires progresse nettement sur la période. " * 10, "ok"),
        ("12 345 678,90 | 98 765 432,10 | 45,6 % | (1 234) " * 10, "garbled"),
    ],
)
def test_text_quality_labels(text, label) -> None:
    assert pdf_text.text_quality(text) == label


def test_extract_document_never_raises(tmp_path) -> None:
    assert pdf_text.extract_document(tmp_path / "absent.pdf")["text_quality"] == "missing_file"

    attachment = tmp_path / "file.docx"
    attachment.write_bytes(b"not a pdf")
    assert pdf_text.extract_document(attachment)["text_quality"] == "not_pdf"

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not really a PDF")
    record = pdf_text.extract_document(broken)
    assert record["text_quality"] == "unreadable" and record["text_chars"] == 0

    good = tmp_path / "good.pdf"
    good.write_bytes(minimal_pdf("Bonjour"))
    record = pdf_text.extract_document(good)
    assert record["text"] == "Bonjour" and record["text_quality"] == "empty"  # under 200 chars


# ----------------------------------------------------------------- universe


def test_universe_has_37_distinct_firms_without_the_foreign_filers() -> None:
    firms = universe.load_universe()
    assert len(firms) == 37
    assert firms["firm_id"].is_unique and firms["isin"].is_unique
    assert firms["yahoo_ticker"].str.endswith(".PA").all()
    for excluded in universe.EXCLUDED_FOREIGN_FILERS:
        assert not firms["name"].str.contains(excluded).any()


def test_previous_isins_map_to_the_same_firm() -> None:
    isin_to_firm = universe.all_isins(universe.load_universe())
    assert len(isin_to_firm) == 40  # 37 current + 3 pre-split ISINs
    assert isin_to_firm["FR001400AJ45"] == isin_to_firm["FR0000121261"] == "ML"
    assert isin_to_firm["FR0014003TT8"] == isin_to_firm["FR0000130650"] == "DSY"
    assert isin_to_firm["FR0014000MR3"] == isin_to_firm["FR0000038259"] == "ERF"


def test_main_analysis_leaves_out_urw_only() -> None:
    firms = universe.load_universe()
    main = universe.main_analysis_firms(firms)
    assert len(main) == 36 and "URW" not in main and "MC" in main
    assert set(firms["firm_id"]) - set(main) == set(universe.EXCLUDED_FROM_MAIN_ANALYSIS)


def test_isin_and_lei_have_the_right_shape() -> None:
    firms = universe.load_universe()
    assert firms["isin"].str.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]").all()
    assert firms["lei"].str.fullmatch(r"[A-Z0-9]{18}[0-9]{2}").all()


# ------------------------------------------------------------------- splits


def test_split_labels_follow_decision_14() -> None:
    days = pd.Series(
        pd.to_datetime(
            [
                "2018-12-31",
                "2019-01-02",
                "2022-12-30",
                "2023-01-02",
                "2024-06-28",
                "2024-07-01",
                "2025-12-31",
                "2026-01-02",
                None,
            ]
        )
    )
    assert splits.label_split(days).tolist() == [
        "pre",
        "train",
        "train",
        "val",
        "val",
        "test",
        "test",
        "post",
        None,
    ]


def test_robustness_sample_starts_after_the_feed_gap() -> None:
    from cac40_mood.data.filings import FEED_GAP_END

    assert splits.ROBUSTNESS_SAMPLE_START > FEED_GAP_END
    assert splits.TRAIN_START < splits.ROBUSTNESS_SAMPLE_START < splits.TRAIN_END


def test_drop_test_period_removes_test_and_undated_rows() -> None:
    frame = pd.DataFrame({"day": pd.to_datetime(["2022-05-02", "2024-06-28", "2024-07-01", None])})
    assert splits.drop_test_period(frame, "day")["day"].tolist() == list(
        pd.to_datetime(["2022-05-02", "2024-06-28"])
    )


# --------------------------------------------------------------------- HTTP


class StubResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content


class StubSession:
    def __init__(self, response: StubResponse) -> None:
        self.response = response
        self.calls = 0

    def get(self, url: str) -> StubResponse:
        self.calls += 1
        return self.response


def test_download_writes_once_and_then_uses_the_file_as_cache(tmp_path) -> None:
    session = StubSession(StubResponse(200, b"%PDF"))
    target = tmp_path / "a" / "b.pdf"
    assert (
        polite_http.download_to_file(session, "https://example.org/b.pdf", target) == "downloaded"
    )
    assert target.read_bytes() == b"%PDF"
    assert polite_http.download_to_file(session, "https://example.org/b.pdf", target) == "cached"
    assert session.calls == 1
    assert not list(tmp_path.rglob("*.part"))


def test_download_reports_a_missing_file_without_creating_one(tmp_path) -> None:
    session = StubSession(StubResponse(404))
    target = tmp_path / "gone.pdf"
    assert (
        polite_http.download_to_file(session, "https://example.org/gone.pdf", target) == "missing"
    )
    assert not target.exists()


def test_polite_session_spaces_requests(monkeypatch) -> None:
    """Two requests in a row must be at least the configured gap apart."""
    clock = {"now": 100.0, "slept": 0.0}
    monkeypatch.setattr(polite_http.time, "monotonic", lambda: clock["now"])

    def fake_sleep(seconds: float) -> None:
        clock["slept"] += seconds
        clock["now"] += seconds

    monkeypatch.setattr(polite_http.time, "sleep", fake_sleep)
    session = polite_http.PoliteSession("test-agent", min_seconds_between_requests=0.5)
    monkeypatch.setattr(session._session, "get", lambda *args, **kwargs: StubResponse(200))
    session.get("https://example.org/1")
    session.get("https://example.org/2")
    assert clock["slept"] == pytest.approx(0.5)


def test_sample_window_defaults_match_the_spec() -> None:
    from cac40_mood.config import load_settings

    settings = load_settings()
    assert (settings.sample_start, settings.sample_end) == (date(2019, 1, 1), date(2025, 12, 31))
