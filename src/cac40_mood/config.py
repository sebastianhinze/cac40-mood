"""Project settings and fixed paths.

Why this module exists: every phase needs the same sample dates, the same folders and the same
"polite client" settings. Reading them in one place means a notebook, a test and a bulk job can
never disagree about, say, where the sample ends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# src/cac40_mood/config.py -> the repository root is two levels above the package folder.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CACHE_DIR: Path = DATA_DIR / "cache"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

# Prices start a year before the event sample so that market-model estimation windows
# ([-250, -30] trading days) and HAR monthly lags have history on the first event day.
PRICE_START: date = date(2018, 1, 1)

PARIS_TZ: str = "Europe/Paris"


@dataclass(frozen=True)
class Settings:
    """Everything read from `.env` except the API key (the Anthropic client reads that itself)."""

    sample_start: date
    sample_end: date
    http_user_agent: str
    http_min_seconds_between_requests: float
    llm_model: str
    llm_reference_model: str
    llm_spend_cap_usd: float
    llm_pilot_size: int
    llm_max_doc_tokens: int


def load_settings(env_file: Path | None = None) -> Settings:
    """Read settings from `.env` (falling back to the documented defaults in `.env.example`).

    Defaults are repeated here on purpose: the data phase needs no secrets, so it should run on
    a fresh clone even before `.env` has been created.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    return Settings(
        sample_start=date.fromisoformat(os.getenv("SAMPLE_START", "2019-01-01")),
        sample_end=date.fromisoformat(os.getenv("SAMPLE_END", "2025-12-31")),
        http_user_agent=os.getenv("HTTP_USER_AGENT", "cac40-mood-research/0.1"),
        http_min_seconds_between_requests=float(
            os.getenv("HTTP_MIN_SECONDS_BETWEEN_REQUESTS", "0.5")
        ),
        llm_model=os.getenv("LLM_MODEL", "claude-haiku-5-5"),
        llm_reference_model=os.getenv("LLM_REFERENCE_MODEL", "claude-sonnet-5-5"),
        llm_spend_cap_usd=float(os.getenv("LLM_SPEND_CAP_USD", "30")),
        llm_pilot_size=int(os.getenv("LLM_PILOT_SIZE", "20")),
        llm_max_doc_tokens=int(os.getenv("LLM_MAX_DOC_TOKENS", "4000")),
    )
