# Build Plan

Effort = focused working days (rough). Each phase ends with: stop, ≤10-bullet summary, DECISIONS.md updated, wait for OK.

| Phase | Effort | Status |
|-------|--------|--------|
| 0 Setup & planning | 0.5 | ✅ 2026-11-11 |
| 1 Data | 3–4 | ✅ 2026-11-11 |
| 2 Scoring | 3 | 🟡 bulk run done 2026-11-12 (6,996 documents, $4.96 spent in total), baselines scored; open: my gold labels and the agreement report |
| 3 Event study | 2 | 🟡 code and synthetic-data tests done 2026-11-11; results wait for scores |
| 4 Volatility | 3 | 🟡 baselines B0 (HAR) and GARCH done on validation 2026-11-11; B1/M wait for scores |
| 5 Particle filter | 4 | 🟡 implemented and validated on simulated data 2026-11-11 (docs/mood_model.md); real scores next |
| 6 Final test evaluation | 1 | ⬜ |
| 7 Dashboard | 2 | ⬜ |
| 8 Write-up | 2 | ⬜ |

## Phase 1: Data
- `universe.csv`: 37 firms → name, ticker, ISIN, LEI, index-entry date. Cross-check with Euronext.
- Ingest the AMF API (Jul 2020–Dec 2025) and the DILA tarballs (Jan 2019–Jun 2020) → one `filings.parquet` with a unified schema.
  Store raw responses and tarball manifests so the ingest can be re-run offline.
- Dedupe FR/EN pairs, tag admin types, pick the conservative timestamp, convert to Europe/Paris.
- Choose and implement the PDF text extractor (pypdf vs. pdfplumber vs. PyMuPDF; check the AGPL licence). Store `text_chars` and
  an extraction-quality flag.
- Prices: yfinance → `prices.parquet` (snapshot date recorded). Trading calendar from `^FCHI`.
- Event alignment: `ts_paris` → `event_session` (d0), `intraday` flag.
- **Done when:** tests cover pre-open / intraday / after-close / weekend / holiday / DST-switch mapping; per-firm-year count
  table looks sane with no hole in 2019–20; ≥95% of non-admin docs have usable text; a data-quality notebook is rendered.

## Phase 2: Scoring
- Pydantic schema + prompt v1. Disk cache, spend logger and hard cap (from `.env`). Dry-run cost estimator.
- Pilot on 20 docs. I review them, iterate the prompt, and re-pilot (each version is cached separately).
- Gold set: I hand-label about 50 docs blind. Measure agreement for the LLM and the baselines.
- Baselines: pick a French finance lexicon and a local multilingual model (adds torch/transformers deps).
- Bulk run via the Batches API, then QA: score distributions, missing or invalid outputs, cost report.
- **Done when:** every non-admin doc is scored once, spend is logged and under the cap, gold-set agreement is reported,
  and the baseline scores are stored alongside.

## Phase 3: Event study
- Market-model ARs, CAR[0], CAR[+1,+5], placebo CAR[−5,−1]. Tercile CAR-path plots by tone.
- Cross-sectional regressions with date-clustered SEs. Only train+val periods are used for exploration.
- Feed gap (DECISIONS #28): events inside 5 Mar – 15 Jun 2020 are left out. Robustness: sample from 2020-07-01. URW is out (#29).
- **Done when:** RQ1/RQ2 results tables + plots are generated from one script, with tests for the AR computation on synthetic data.

## Phase 4: Volatility
- Garman–Klass / Parkinson RV. HAR B0, B1 (+event dummies), M (+scores). GARCH-X robustness.
- Rolling out-of-sample on validation. QLIKE/MSE, DM tests, MZ regressions.
- Feed gap (#28): announcement features are MISSING in the window (not zero) and those days are left out of estimation and
  evaluation. Robustness: sample from 2020-07-01. 36 firms (#29).
- **Done when:** a validation-period comparison table B0/B1/M at h = 1, 5, 22 exists, the feature pipeline passes a
  "no future information" test (shifting the data by one day changes no past feature), and the config is frozen for phase 6.

## Phase 5: Particle filter
- Exact Kalman filter (reference) → bootstrap PF (NumPy, vectorised, systematic resampling, ESS-triggered).
- Tests: PF ≈ KF on linear-Gaussian simulated data; parameter recovery on simulated irregular observations;
  log-likelihood is stable across seeds.
- Market + firm structure (decide: joint PF vs. two-stage vs. Rao-Blackwellised). Robust observation model.
- Fit on train, filter through val. Mood features → HAR model M2.
- Feed gap (#28): the filter runs predict-only through the window (no observations, wider bands). Test this explicitly.
- **Done when:** the tests pass, the validation comparison M vs. M2 is reported, and runtime for the full panel is under a few minutes on the M3.

## Phase 6: Final test evaluation (one shot)
- Freeze config (git tag), run the primary hypotheses on the test set, apply Holm, and write down the results as they came out.
- **Done when:** a test-results table is committed with the tag hash, and nothing was changed after seeing it.

## Phase 7: Dashboard (Streamlit)
- Firm picker → filtered mood ± bands (smoothed shown separately, labelled), announcement list with scores and rationale, an
  event-study tab, and a model-comparison tab.
- **Done when:** `uv run streamlit run app/main.py` starts in < 10 s on processed data and shows no raw full texts.

## Phase 8: Write-up
- README (5-minute story), a 2–4 page report (method, results, biases, limitations), and an interview Q&A cheat sheet from DECISIONS.md.
- **Done when:** a fresh clone with the documented steps reproduces the figures.
