"""Tables for the Streamlit dashboard (`app/main.py`), read from saved results only.

The dashboard never calls an API and never estimates anything. It reads the public data bundle
in `public_data/` (built from the local pipeline outputs by `cac40_mood.public_bundle`, and
committed), so it runs the same on a fresh clone, on Streamlit Community Cloud and locally.

One thing only exists locally: the filing texts. When `data/processed/texts.parquet` is on this
machine, the announcement viewer also shows the first few hundred characters of a document.
The public version shows the link to the official PDF instead (the texts are the issuers').

This module is kept light on purpose (pandas only, no pipeline imports), so the hosted app
needs few dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cac40_mood.config import PROCESSED_DIR, PROJECT_ROOT

BUNDLE_DIR: Path = PROJECT_ROOT / "public_data"
TEXTS_PATH: Path = PROCESSED_DIR / "texts.parquet"

# Excerpts (local only) stay short: enough to recognise the document, not to republish it.
EXCERPT_CHARACTERS: int = 400

# 95% band around the filtered mood (normal approximation).
Z_95: float = 1.96


def excerpts_available() -> bool:
    """True when the local filing texts are on this machine (never in the public version)."""
    return TEXTS_PATH.exists()


def excerpt(text: str, length: int = EXCERPT_CHARACTERS) -> str:
    """First `length` characters with whitespace collapsed, cut at a word boundary."""
    flat = " ".join(text.split())
    if len(flat) <= length:
        return flat
    return flat[:length].rsplit(" ", 1)[0] + " …"


def load_announcements() -> pd.DataFrame:
    """Scored announcements, newest first; with an `excerpt` column when texts exist locally."""
    table = pd.read_parquet(BUNDLE_DIR / "announcements.parquet")
    if excerpts_available():
        texts = pd.read_parquet(TEXTS_PATH, columns=["doc_id", "text"])
        texts = texts[texts["doc_id"].isin(table["doc_id"])]
        texts["excerpt"] = texts["text"].map(excerpt)
        table = table.merge(texts[["doc_id", "excerpt"]], on="doc_id", how="left")
    return table


def load_mood() -> pd.DataFrame:
    """Filtered mood per firm and trading day with its 95% band, 2019-2025."""
    return pd.read_parquet(BUNDLE_DIR / "mood.parquet")


def tone_centre() -> float:
    """The train-period average tone that the mood filter subtracts from every reading.

    The mood is in "tone above the usual level" units, so announcement tones are shifted by the
    same amount before they are drawn next to it.
    """
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text())
    return float(manifest["tone_centre_train"])


def car_paths_by_tone() -> pd.DataFrame:
    """Average cumulative abnormal return, days -5 to +5, by tone third (train + validation)."""
    return pd.read_csv(BUNDLE_DIR / "event_car_paths.csv").set_index("day")
