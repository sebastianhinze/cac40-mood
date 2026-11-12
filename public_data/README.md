# Public data bundle

Derived results of the CAC 40 Mood project: what the dashboard runs on. About 2 MB. Built by
`uv run python -m cac40_mood.public_bundle` from the local pipeline outputs; `manifest.json` records
the build time, the git commit and the row counts.

| File | Content |
|---|---|
| `announcements.parquet` | 6,860 scored, non-administrative filings of 36 CAC 40 firms, 2019-2025: issuer, title, publication time (Paris), event session, period, language, and the LLM's topic, tone, surprise, uncertainty, confidence and one-line rationale; link to the official PDF |
| `mood.parquet` | Filtered mood per firm and trading day with its 95% band, and the market mood. From July 2024 (test period) **for display only**: the same filter with the frozen train-period parameters, used by no analysis |
| `event_car_paths.csv` | Average cumulative abnormal return, days -5 to +5, by tone third (train + validation) |
| `event_day0_terciles.csv`, `event_scorer_slopes.csv` | Day-0 tables of `reports/03_event_study.md` and `reports/06_test_results.md` |
| `vol_validation_vs_test.csv`, `vol_test_*.csv` | Volatility comparisons of `reports/04_*`, `05_*` and `06_test_results.md` |

**Not included:** prices or per-firm daily series derived from them (Yahoo Finance terms), the texts of the
filings or excerpts of them (the issuers' documents: follow `pdf_url`), the Loughran-McDonald word list.

**Attribution.** Filing metadata (issuer, title, timestamps, document links): DILA, info-financière
(https://www.info-financiere.gouv.fr), data received from the AMF; reused under the Licence Ouverte /
Open Licence 2.0 (Etalab). Feed snapshot of 2026-11-11.

**Licence of the derived values** (scores, rationales, mood, tables): CC BY 4.0. Scores and rationales
were produced by `claude-haiku-5-5` (prompt v2) and are model judgements, not facts about the issuers.
