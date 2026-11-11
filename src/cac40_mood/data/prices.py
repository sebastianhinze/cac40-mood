"""Daily prices from Yahoo Finance (via the yfinance package), DECISIONS #12.

What we store per firm and per day: open, high, low, close (as traded, adjusted for splits),
`adj_close` (also adjusted for dividends) and volume.
- Returns for the event study use `adj_close`, so a dividend payment is not mistaken for a loss.
- Range-based volatility (Garman-Klass, Parkinson) uses open/high/low/close of the same day.
  Those are ratios within one day, so the dividend adjustment does not matter for them.

Yahoo's terms allow personal use only: prices stay under `data/` (git-ignored) and the snapshot
date is recorded, because Yahoo revises history (for example after a new dividend).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

INDEX_TICKER: str = "^FCHI"
INDEX_ID: str = "FCHI"
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")


def download_ticker(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download one ticker's daily bars for [start, end], exactly as Yahoo returns them."""
    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # Yahoo's end date is exclusive
        auto_adjust=False,  # keep raw OHLC and the separate "Adj Close" column
        progress=False,
        multi_level_index=False,
    )
    raw.index.name = "Date"
    return raw


def fetch_prices(
    tickers: dict[str, str], start: date, end: date, out_dir: Path, pause_seconds: float
) -> dict[str, int]:
    """Download every ticker that is not on disk yet; return {ticker: number of rows}.

    `tickers` maps our id (e.g. 'MC') to the Yahoo ticker (e.g. 'MC.PA'). One CSV per ticker is
    written, plus `snapshot.json` with the download time and the yfinance version.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for firm_id, ticker in tickers.items():
        target = out_dir / f"{firm_id}.csv"
        if not target.exists():
            download_ticker(ticker, start, end).to_csv(target)
            time.sleep(pause_seconds)
        counts[ticker] = len(pd.read_csv(target))
    snapshot = {
        "snapshot_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": f"yfinance {yf.__version__}",
        "window": [start.isoformat(), end.isoformat()],
        "tickers": tickers,
    }
    snapshot_path = out_dir / "snapshot.json"
    if not snapshot_path.exists():
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
    return counts


def tidy_prices(raw: pd.DataFrame, firm_id: str) -> pd.DataFrame:
    """One ticker's raw Yahoo table -> tidy rows (firm_id, date, open, ..., volume).

    Days on which Yahoo has no close are dropped: they are days the stock did not trade.
    """
    tidy = raw.rename(columns=lambda name: str(name).strip().lower().replace(" ", "_"))
    tidy = tidy.rename(columns={"date": "date"}).reset_index(drop=False)
    tidy = tidy.rename(columns={tidy.columns[0]: "date"})
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.tz_localize(None).dt.normalize()
    tidy = tidy[tidy["close"].notna()]
    tidy.insert(0, "firm_id", firm_id)
    return tidy[["firm_id", "date", *PRICE_COLUMNS]].sort_values("date").reset_index(drop=True)


def load_raw_prices(raw_dir: Path) -> pd.DataFrame:
    """Read all stored CSVs into one tidy table (firms and the index stacked)."""
    frames = []
    for path in sorted(raw_dir.glob("*.csv")):
        raw = pd.read_csv(path, index_col=0)
        frames.append(tidy_prices(raw, path.stem))
    return pd.concat(frames, ignore_index=True)


def price_problems(prices: pd.DataFrame) -> pd.DataFrame:
    """Rows that break basic sanity rules: the day's high must be the highest price, etc.

    A small tolerance (0.01%) absorbs Yahoo's rounding. The data-quality report lists what is
    left; nothing is fixed silently.
    """
    tolerance = 1e-4
    highest = prices[["open", "close", "low"]].max(axis=1)
    lowest = prices[["open", "close", "high"]].min(axis=1)
    problems = pd.DataFrame(
        {
            "high_below_other_price": prices["high"] < highest * (1 - tolerance),
            "low_above_other_price": prices["low"] > lowest * (1 + tolerance),
            "non_positive_price": (prices[["open", "high", "low", "close"]] <= 0).any(axis=1),
            "missing_ohlc": prices[["open", "high", "low", "close"]].isna().any(axis=1),
        }
    )
    return prices[problems.any(axis=1)].join(problems[problems.any(axis=1)])


def missing_sessions(prices: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """For each firm, how many index trading days have no price (after its first price)."""
    rows = []
    for firm_id, firm_prices in prices.groupby("firm_id"):
        first_day = firm_prices["date"].min()
        expected = sessions[sessions >= first_day]
        missing = expected.difference(pd.DatetimeIndex(firm_prices["date"]))
        rows.append(
            {
                "firm_id": firm_id,
                "first_day": first_day,
                "n_days": len(firm_prices),
                "n_missing": len(missing),
            }
        )
    return pd.DataFrame(rows)
