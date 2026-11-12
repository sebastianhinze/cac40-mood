"""Build the public data bundle in `public_data/` (DECISIONS #67, #69).

    uv run python -m cac40_mood.public_bundle

The bundle is what the public dashboard runs on when the local pipeline outputs are missing
(a fresh clone, Streamlit Community Cloud). It holds **derived results only**:

- `announcements.parquet`: one row per scored, non-administrative filing of the 36 firms:
  filing metadata (issuer, title, timestamps, link to the official PDF on the DILA server) and
  the LLM's scores and one-line rationale. **No text from the filings themselves.**
- `mood.parquet`: the filtered mood per firm and trading day with its 95% band, 2019-2025.
- `event_car_paths.csv`: average cumulative abnormal return by tone third (train + validation),
  plus the event-study and volatility tables as they appear in the reports.
- `manifest.json`: sources, attribution, row counts, the git commit it was built from.

Not in the bundle: prices or anything per firm and day derived from them (Yahoo's terms),
filing texts or excerpts (the issuers' documents), the Loughran-McDonald word list.

About the test-period mood (DECISIONS #69): phase 6 filtered the test period in memory and
saved only the forecast comparisons. For display, this module runs the **same linear
two-stage filter with the parameters, tone centre, particle count and seed saved in phase 4**
(`data/processed/mood/parameters.json`, all from the train period) from 2019 to the end of
2025. Nothing is estimated. Because the filter is causal and seeded, the 2019 to mid-2024 part
must equal the saved series; `display_mood` checks that and refuses to continue otherwise. The
result is never used by any analysis, report or test of a hypothesis.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cac40_mood import dashboard_data, presentation
from cac40_mood.analysis_documents import load_documents
from cac40_mood.config import PROCESSED_DIR
from cac40_mood.data.build import FILINGS_PATH, SESSIONS_PATH
from cac40_mood.data.schema import PDF_BASE_URL
from cac40_mood.data.universe import EXCLUDED_FROM_MAIN_ANALYSIS, load_universe
from cac40_mood.events import study
from cac40_mood.mood import real
from cac40_mood.mood.hierarchy import two_stage_filter
from cac40_mood.mood.model import MoodParameters
from cac40_mood.splits import TEST_END, VAL_END

BUNDLE_DIR: Path = dashboard_data.BUNDLE_DIR
SCORES_PATH: Path = PROCESSED_DIR / "scores.parquet"
SAVED_MOOD_PATH: Path = PROCESSED_DIR / "mood" / "filtered_mood.parquet"
MOOD_PARAMETERS_PATH: Path = PROCESSED_DIR / "mood" / "parameters.json"
EVENTS_PATH: Path = PROCESSED_DIR / "events" / "events_train_val.parquet"
PHASE6_DIR: Path = PROCESSED_DIR / "phase6"

# Rounding keeps the files small; 4 decimals is far below the scores' own noise (about 0.02).
DECIMALS: int = 4

# The display run must reproduce the saved 2019 to mid-2024 mood to this precision.
REPRODUCTION_TOLERANCE: float = 1e-9

ATTRIBUTION: str = (
    "Filing metadata (issuer, title, timestamps, document links): DILA, info-financiere "
    "(https://www.info-financiere.gouv.fr), data received from the AMF; reused under the "
    "Licence Ouverte / Open Licence 2.0 (Etalab). Feed snapshot of 2026-11-11."
)


def rounded(table: pd.DataFrame) -> pd.DataFrame:
    """Round the numeric columns to `DECIMALS` (dates and text are left alone)."""
    numeric = table.select_dtypes("number").columns
    return table.assign(**{name: table[name].round(DECIMALS) for name in numeric})


# ----------------------------------------------------------------------------- mood


def display_mood() -> pd.DataFrame:
    """Linear two-stage mood, 2019 to the end of the test period, saved parameters only."""
    saved = json.loads(MOOD_PARAMETERS_PATH.read_text())
    market = MoodParameters(**saved["market"])
    firm = MoodParameters(**saved["firm"])
    centre = float(saved["tone_centre_train"])

    documents, _ = load_documents(("train", "val", "test"))
    calendar = real.sample_sessions(pd.read_parquet(SESSIONS_PATH), TEST_END)
    dates = calendar["date"]
    missing_days = np.flatnonzero(~calendar["feed_observed"].to_numpy())
    readings = real.make_readings(documents, dates, centre)
    result = two_stage_filter(
        readings,
        len(dates),
        market,
        firm,
        int(saved["particles"]),
        int(saved["seed"]),
        missing_days,
    )
    mood = real.mood_table(result, dates)
    check_reproduces_saved(mood)
    return mood


def check_reproduces_saved(mood: pd.DataFrame) -> None:
    """Raise unless the 2019 to mid-2024 part equals the series saved in phase 4."""
    saved = pd.read_parquet(SAVED_MOOD_PATH)
    columns = ["mood_market", "mood_total", "mood_sd"]
    merged = saved.merge(mood, on=["firm_id", "date"], suffixes=("_saved", "_display"))
    if len(merged) != len(saved):
        raise RuntimeError("The display run does not cover every saved firm-day.")
    worst = max(
        float((merged[f"{name}_saved"] - merged[f"{name}_display"]).abs().max()) for name in columns
    )
    if worst > REPRODUCTION_TOLERANCE:
        raise RuntimeError(f"The display run differs from the saved mood by up to {worst:.2e}.")
    print(f"display mood reproduces the saved 2019 to mid-2024 series (max difference {worst:.1e})")


def mood_for_bundle(mood: pd.DataFrame) -> pd.DataFrame:
    """Firm's total mood with its 95% band, the market mood with its band, and the period."""
    table = pd.DataFrame(
        {
            "firm_id": mood["firm_id"],
            "date": pd.to_datetime(mood["date"]),
            "mood_total": mood["mood_total"],
            "mood_low": mood["mood_total"] - dashboard_data.Z_95 * mood["mood_sd"],
            "mood_high": mood["mood_total"] + dashboard_data.Z_95 * mood["mood_sd"],
            "mood_market": mood["mood_market"],
            "mood_market_low": mood["mood_market_low"],
            "mood_market_high": mood["mood_market_high"],
        }
    )
    table["period"] = np.where(table["date"] > pd.Timestamp(VAL_END), "test (display only)", "")
    table.loc[table["period"] == "", "period"] = "train + validation"
    return rounded(table)


# -------------------------------------------------------------------- announcements


def announcements_for_bundle() -> pd.DataFrame:
    """Metadata + scores + rationale for every scored, non-administrative filing. No text."""
    filings = pd.read_parquet(FILINGS_PATH)
    scores = pd.read_parquet(SCORES_PATH)
    scores = scores[scores["problem"] == ""]
    universe = load_universe()
    names = {
        firm: name
        for firm, name in zip(universe["firm_id"], universe["name"], strict=True)
        if firm not in EXCLUDED_FROM_MAIN_ANALYSIS
    }
    table = filings[~filings["is_admin"] & filings["firm_id"].isin(names)].merge(
        scores[["doc_id", "tone", "surprise", "uncertainty", "topic", "confidence", "rationale"]],
        on="doc_id",
    )
    table = table.assign(
        company=table["firm_id"].map(names),
        published_paris=table["ts_paris"].dt.strftime("%Y-%m-%d %H:%M"),
        pdf_url=PDF_BASE_URL + table["pdf_path"],
        period=table["split"].map({"train": "train", "val": "validation", "test": "test"}),
    )
    columns = [
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
    ]
    table = table[columns].sort_values(["published_paris", "doc_id"], ascending=[False, True])
    return rounded(table).reset_index(drop=True)


# ----------------------------------------------------------------------- event study


def car_paths() -> pd.DataFrame:
    """Average cumulative abnormal return, days -5..+5, by tone third (train + validation)."""
    events = pd.read_parquet(EVENTS_PATH)
    table = events.assign(tone_tercile=study.terciles(events["tone"]))
    paths = study.cumulative_paths(table, "tone_tercile")
    return paths.round(6).reset_index()


# --------------------------------------------------------------------------- build


def git_commit() -> str:
    """The commit the bundle was built from (marked '+changes' if the tree is not clean)."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return commit + ("+changes" if dirty else "")


def build(directory: Path = BUNDLE_DIR) -> dict[str, object]:
    """Write every bundle file and return the manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    mood = display_mood()
    tables: dict[str, pd.DataFrame] = {
        "announcements.parquet": announcements_for_bundle(),
        "mood.parquet": mood_for_bundle(mood),
        "event_car_paths.csv": car_paths(),
        "event_day0_terciles.csv": presentation.day0_terciles(),
        "event_scorer_slopes.csv": presentation.scorer_slopes(),
        "vol_validation_vs_test.csv": presentation.volatility_validation_vs_test(),
    }
    for name, table in tables.items():
        path = directory / name
        if name.endswith(".parquet"):
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
    for name in ("test_primary.csv", "test_exploratory.csv"):
        shutil.copyfile(PHASE6_DIR / name, directory / f"vol_{name}")

    manifest = {
        "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "tone_centre_train": json.loads(MOOD_PARAMETERS_PATH.read_text())["tone_centre_train"],
        "rows": {name: len(table) for name, table in tables.items()},
        "attribution": ATTRIBUTION,
        "derived_data_licence": "CC BY 4.0",
        "not_included": [
            "prices or per-firm daily series derived from them",
            "filing texts or excerpts",
            "the Loughran-McDonald word list",
        ],
        "test_period_mood": (
            "display only: same filter and saved train-period parameters as phase 4, run to "
            "2025-12-31; not used in any analysis"
        ),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    manifest = build()
    print(json.dumps(manifest["rows"], indent=2))
    for path in sorted(BUNDLE_DIR.iterdir()):
        print(f"{path.name:32s} {path.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
