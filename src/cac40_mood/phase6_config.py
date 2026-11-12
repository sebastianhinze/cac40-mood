"""Frozen configuration for the one-off test-period evaluation (phase 6).

Approved in DECISIONS #64. Nothing in this file may change after the git tag `phase6-freeze`
is set. The runner is `phase6.py`; what it does, step by step: docs/PHASE6_PLAN.md.

The numbers below were estimated on the train period (2019-2022) and are copied from
`data/processed/mood/nonlinear_parameters.json`.
"""

from __future__ import annotations

from datetime import date

from cac40_mood.vol.sentiment_run import MODELS

# Periods. Forecast origins are the trading days of the test period; the last origin for
# horizon h lies h trading days before the last test session, so every target is observed.
TEST_START: date = date(2024, 7, 1)
TEST_END: date = date(2025, 12, 31)

# Primary family: two tests, both at this horizon, both on QLIKE with a Diebold-Mariano test
# on the daily cross-firm mean loss difference (Newey-West, at least h - 1 lags), two-sided,
# Holm-adjusted across the two.
PRIMARY_HORIZON: int = 5
SIGNIFICANCE_LEVEL: float = 0.05
RQ3_MODEL: str = "M"  # HAR + announcement dummies + |tone| + uncertainty
RQ3_BASELINE: str = "B1"  # HAR + announcement dummies
RQ4_MODEL: str = "M3"  # M + nonlinear model's 5-day log forecast + its filtered mood (#58)
RQ4_BASELINE: str = "M"

# Columns of the models (taken from the code that produced the validation results).
COLUMNS: dict[str, list[str]] = {
    "B1": MODELS["B1"],
    "M": MODELS["M"],
    "M3": [*MODELS["M"], f"sv_log_forecast_{PRIMARY_HORIZON}", "sv_mood"],
}

# Estimation scheme of the HAR-type models: as on validation.
REFIT_EVERY_TRADING_DAYS: int = 22
EXPANDING_WINDOW_FROM: date = date(2019, 1, 1)

# Fixed inputs estimated on the train period only (never re-estimated):
TONE_CENTRE_TRAIN: float = 0.1256961018297534
MOOD_PARAMETERS: dict[str, float] = {
    "phi": 0.9535477072229647,
    "sigma_state": 0.06281965064168012,
    "sigma_obs": 0.17914645867888845,
}
SV_PARAMETERS: dict[str, float] = {
    "phi_h": 0.9552993555785635,
    "sigma_h": 0.26317387121026287,
    "beta": 0.0003774319740721278,
}
FILTER_PARTICLES: int = 5_000
FILTER_SEED: int = 58

# Scores: the bulk run of 2026-11-12 (prompt v2, claude-haiku-5-5, thinking low, first 12,000
# tokens). Test-period documents were scored then and have not been looked at.
PROMPT_VERSION: str = "v2"
