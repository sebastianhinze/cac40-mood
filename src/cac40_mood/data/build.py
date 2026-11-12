"""Command line for the data phase: one sub-command per step, in this order.

    uv run python -m cac40_mood.data.build fetch-tarballs   # DILA archive, Jan 2019 - Jun 2020
    uv run python -m cac40_mood.data.build fetch-api        # API metadata, whole sample
    uv run python -m cac40_mood.data.build fetch-prices     # yfinance OHLC
    uv run python -m cac40_mood.data.build prices           # raw -> prices.parquet       (offline)
    uv run python -m cac40_mood.data.build filings          # raw -> filings.parquet      (offline)
    uv run python -m cac40_mood.data.build fetch-pdfs       # PDFs of the documents to be scored
    uv run python -m cac40_mood.data.build texts            # PDF -> texts.parquet        (offline)

The `fetch-*` steps talk to the network and only add files under `data/raw/`. The other steps
read `data/raw/` and write `data/processed/`; they are deterministic and run offline.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date

import pandas as pd

from cac40_mood.config import PRICE_START, PROCESSED_DIR, RAW_DIR, load_settings
from cac40_mood.data import amf_api, dila_archive, pdf_text
from cac40_mood.data import filings as filings_module
from cac40_mood.data import prices as prices_module
from cac40_mood.data.polite_http import PoliteSession, download_to_file
from cac40_mood.data.schema import PDF_BASE_URL
from cac40_mood.data.sessions import trading_sessions
from cac40_mood.data.universe import all_isins, load_universe
from cac40_mood.splits import label_split

# DECISIONS #4: the archive bundles are the source of record for the early sample.
TARBALL_WINDOW_START: date = date(2019, 1, 1)
TARBALL_WINDOW_END: date = date(2020, 6, 30)

API_RETRIES: int = 8
API_RETRY_PAUSE_SECONDS: float = 300.0

PDF_DIR = RAW_DIR / "pdf"
FILINGS_ALL_PATH = PROCESSED_DIR / "filings_all.parquet"
FILINGS_PATH = PROCESSED_DIR / "filings.parquet"
PRICES_PATH = PROCESSED_DIR / "prices.parquet"
TEXTS_PATH = PROCESSED_DIR / "texts.parquet"
TEXTS_EN_PATH = PROCESSED_DIR / "texts_en.parquet"
SESSIONS_PATH = PROCESSED_DIR / "sessions.parquet"


def _session() -> PoliteSession:
    settings = load_settings()
    return PoliteSession(settings.http_user_agent, settings.http_min_seconds_between_requests)


def fetch_tarballs(start: date, end: date) -> None:
    """Download the archive bundles of the window and keep the files of our ISINs."""
    isins = set(all_isins(load_universe()))
    totals = dila_archive.fetch_window(_session(), isins, start, end, RAW_DIR / "dila", PDF_DIR)
    print(f"tarballs: {totals}")


def fetch_api() -> None:
    """Take one consistent API snapshot; wait and retry while the API index is being rebuilt."""
    settings = load_settings()
    isins = sorted(all_isins(load_universe()))
    for attempt in range(1, API_RETRIES + 1):
        try:
            counts = amf_api.fetch_snapshot(
                _session(), isins, settings.sample_start, settings.sample_end, RAW_DIR / "amf_api"
            )
        except amf_api.IndexRebuildingError as error:
            print(f"attempt {attempt}: {error}; waiting {API_RETRY_PAUSE_SECONDS:.0f} s")
            time.sleep(API_RETRY_PAUSE_SECONDS)
        else:
            print(f"api: {sum(counts.values())} rows for {len(counts)} ISINs")
            return
    raise SystemExit("The API index never settled; run this step again later.")


def fetch_prices() -> None:
    """Download daily prices for the 37 firms and the CAC 40 index."""
    settings = load_settings()
    universe = load_universe()
    tickers = {prices_module.INDEX_ID: prices_module.INDEX_TICKER}
    tickers.update(dict(zip(universe["firm_id"], universe["yahoo_ticker"], strict=True)))
    counts = prices_module.fetch_prices(
        tickers, PRICE_START, settings.sample_end, RAW_DIR / "prices", pause_seconds=1.0
    )
    print(f"prices: {len(counts)} tickers, {sum(counts.values())} rows")


def build_prices() -> pd.DataFrame:
    """raw CSVs -> prices.parquet, restricted to the index's trading days."""
    raw = prices_module.load_raw_prices(RAW_DIR / "prices")
    sessions = trading_sessions(raw[raw["firm_id"] == prices_module.INDEX_ID])
    # Yahoo occasionally carries a stale row for a stock on an exchange holiday (for example
    # 25 Dec 2019). The index calendar decides what a trading day is, so those rows go.
    on_session = raw["date"].isin(sessions)
    prices = raw[on_session].reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICES_PATH, index=False)
    # One row per trading day: its split and whether the filings feed is observed that day.
    # Later phases join on this table, so "missing feed" is never confused with "no news".
    calendar = pd.DataFrame({"date": sessions})
    calendar["split"] = label_split(calendar["date"])
    calendar["feed_observed"] = filings_module.feed_is_observed(calendar["date"])
    calendar.to_parquet(SESSIONS_PATH, index=False)
    snapshot = json.loads((RAW_DIR / "prices" / "snapshot.json").read_text())
    print(
        f"prices: {len(prices)} rows, {len(sessions)} sessions, "
        f"{(~on_session).sum()} off-calendar rows dropped, snapshot {snapshot['snapshot_utc']}"
    )
    return prices


def build_filings() -> pd.DataFrame:
    """raw API rows + archive XML -> filings_all.parquet (every row) and filings.parquet."""
    settings = load_settings()
    prices = pd.read_parquet(PRICES_PATH)
    sessions = trading_sessions(prices[prices["firm_id"] == prices_module.INDEX_ID])
    combined = filings_module.combine_sources(
        amf_api.read_raw_records(RAW_DIR / "amf_api"),
        dila_archive.read_raw_records(RAW_DIR / "dila"),
    )
    everything = filings_module.build_filings(
        combined,
        all_isins(load_universe()),
        sessions,
        settings.sample_start,
        settings.sample_end,
    )
    kept = filings_module.keep_documents(everything)
    everything.to_parquet(FILINGS_ALL_PATH, index=False)
    kept.to_parquet(FILINGS_PATH, index=False)
    print(
        f"filings: {len(everything)} rows -> {len(kept)} documents after removing translations "
        f"({(~kept['is_admin']).sum()} to score, {kept['is_admin'].sum()} admin)"
    )
    return kept


def _documents_to_score() -> pd.DataFrame:
    filings = pd.read_parquet(FILINGS_PATH)
    return filings[~filings["is_admin"]].reset_index(drop=True)


def fetch_pdfs() -> None:
    """Download the PDF of every document that will be scored and is not on disk yet."""
    session = _session()
    outcomes = {"cached": 0, "downloaded": 0, "missing": 0}
    for pdf_path in _documents_to_score()["pdf_path"]:
        outcomes[download_to_file(session, PDF_BASE_URL + pdf_path, PDF_DIR / pdf_path)] += 1
    print(f"pdfs: {outcomes}")


def english_versions() -> pd.DataFrame:
    """Official English versions of the French documents to score (train and validation).

    The feed often carries the same announcement twice, in French and in English. The French
    one is the document we score; its English twin was set aside in phase 1. Returns one row
    per French document that has a twin: doc_id (French), en_doc_id, pdf_path (English file).
    """
    everything = pd.read_parquet(FILINGS_ALL_PATH)
    twins = everything.loc[
        everything["is_translation"] & (everything["lang"] == "en"),
        ["doc_id", "twin_doc_id", "pdf_path"],
    ].rename(columns={"doc_id": "en_doc_id", "twin_doc_id": "doc_id"})
    documents = _documents_to_score()
    wanted = documents.loc[documents["split"].isin(["train", "val"]), ["doc_id"]]
    return wanted.merge(twins, on="doc_id").drop_duplicates("doc_id").reset_index(drop=True)


def fetch_english_pdfs() -> None:
    """Download the English twins that are not on disk yet."""
    session = _session()
    outcomes = {"cached": 0, "downloaded": 0, "missing": 0}
    for pdf_path in english_versions()["pdf_path"]:
        outcomes[download_to_file(session, PDF_BASE_URL + pdf_path, PDF_DIR / pdf_path)] += 1
    print(f"english pdfs: {outcomes}")


def build_english_texts() -> pd.DataFrame:
    """Extract the English twins -> texts_en.parquet (keyed by the French document's doc_id)."""
    versions = english_versions()
    records = [pdf_text.extract_document(PDF_DIR / path) for path in versions["pdf_path"]]
    texts = pd.concat([versions[["doc_id", "en_doc_id"]], pd.DataFrame(records)], axis=1)
    texts.to_parquet(TEXTS_EN_PATH, index=False)
    print(f"english texts: {texts['text_quality'].value_counts().to_dict()}")
    return texts


def build_texts() -> pd.DataFrame:
    """Extract the first pages of every document to be scored -> texts.parquet."""
    documents = _documents_to_score()
    records = [pdf_text.extract_document(PDF_DIR / path) for path in documents["pdf_path"]]
    texts = pd.concat([documents[["doc_id"]], pd.DataFrame(records)], axis=1)
    texts.to_parquet(TEXTS_PATH, index=False)
    print(f"texts: {texts['text_quality'].value_counts().to_dict()}")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Data phase: fetch raw data and build tables.")
    steps = parser.add_subparsers(dest="step", required=True)
    tarballs = steps.add_parser("fetch-tarballs")
    tarballs.add_argument("--start", type=date.fromisoformat, default=TARBALL_WINDOW_START)
    tarballs.add_argument("--end", type=date.fromisoformat, default=TARBALL_WINDOW_END)
    simple_steps = {
        "fetch-api": fetch_api,
        "fetch-prices": fetch_prices,
        "prices": build_prices,
        "filings": build_filings,
        "fetch-pdfs": fetch_pdfs,
        "texts": build_texts,
        "fetch-pdfs-en": fetch_english_pdfs,
        "texts-en": build_english_texts,
    }
    for name in simple_steps:
        steps.add_parser(name)
    args = parser.parse_args()
    if args.step == "fetch-tarballs":
        fetch_tarballs(args.start, args.end)
    else:
        simple_steps[args.step]()


if __name__ == "__main__":
    main()
