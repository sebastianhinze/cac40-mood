# CAC 40 Mood: Project Spec

> Does the tone of official CAC 40 announcements predict abnormal returns and, above all, future volatility
> beyond what a standard volatility model already predicts?

Status: spec v0.1 (2026-11-11). Every change gets a line in DECISIONS.md.

## 1. Research questions

| # | Question | Role |
|---|----------|------|
| RQ1 | Do announcement scores (tone, surprise) explain the **day-0 abnormal return**? | Checks that the scores carry information. Not prediction, since news is public at day 0. |
| RQ2 | Do scores predict **abnormal returns on days +1…+5** (drift)? | Predictability / market efficiency. A null result is expected and fine. |
| RQ3 | Do scores (uncertainty, \|tone\|, surprise) improve **range-based volatility forecasts** (h = 1, 5, 22 days) beyond HAR **plus announcement-type dummies**? | **Primary question** |
| RQ4 | Does the **filtered latent mood** (particle filter) add forecasting power beyond the raw per-document scores? | Signature feature |
| RQ5 | Does the LLM scorer beat the simple baselines (French lexicon, local multilingual model)? | Justifies paying for an LLM |

The key control in RQ3: announcements alone raise volatility (earnings days are noisy). Sentiment has to add value
**beyond the fact that an announcement of a given type happened**. Otherwise the result just says "earnings days are volatile".

## 2. Sample & data

- **Period:** 2019-01-01 → 2025-12-31 for events. Prices from 2018-01 so event-study estimation windows and HAR lags have history.
- **Universe:** current CAC 40 snapshot (2026-11-11) **minus Airbus, ArcelorMittal, Stellantis** (they don't file with the
  French OAM). That leaves **37 firms**. Survivorship and selection bias are reported, not corrected.
- **Announcements:** DILA info-financiere (AMF regulated information). API for Jul 2020 onward, raw DILA archive
  tarballs for Jan 2019–Jun 2020. Details: `docs/DATA_SOURCES.md`.
  - All regulated-information types are kept and tagged. Pure admin types (voting-rights totals, buyback logs, liquidity
    contract reports, threshold crossings) are **counted but not LLM-scored**. Their counts can still be features.
  - Language: French version when it exists, otherwise English. FR/EN duplicates are dropped and `lang` kept as a column.
  - Timestamp: conservative = **latest** of the provided UTC timestamps, converted to Europe/Paris.
- **Prices:** yfinance daily OHLC + Adj Close for 37 `.PA` tickers + `^FCHI`. The trading calendar comes from `^FCHI` dates.
- **Phase 1 amendments (2026-11-11):** URW has no usable price history before 2023-04, so the main analysis uses
  **36 firms** (DECISIONS #29). The filings feed is nearly empty from 5 Mar to 15 Jun 2020; that window is excluded
  and treated as missing data, with a robustness run starting 2020-07-01 (DECISIONS #28).

## 3. Methods

### 3.1 LLM scoring
- Model `claude-haiku-5-5`. Input: title + first ~4k tokens of the extracted text (truncation logged). Output is strict JSON with this schema:
  - `tone` ∈ [-1, 1], `surprise` ∈ [-1, 1] (vs. implied expectations, 0 = in line/unknown), `uncertainty` ∈ [0, 1]
    (hedging / risk language), `topic` (fixed enum), `confidence` ∈ [0, 1], `rationale` (≤ 30 words, English).
- The schema is enforced with structured outputs. `prompt_version` is stored with every result.
- Bulk scoring runs through the Message Batches API (−50%). Every response is cached on disk and keyed by (model, prompt_version, doc hash).
- **Baselines:** (a) French finance lexicon, tone = (pos − neg) / (pos + neg); (b) a free local multilingual
  sentiment model on the M3. The exact lexicon and model are chosen in phase 2.
- **Validation set:** about 50 documents hand-labelled by me (blind to model scores). Report agreement for the LLM and the baselines.

### 3.2 Event study (RQ1, RQ2)
- Market model vs. `^FCHI`, estimation window [−250, −30] trading days. Windows CAR[0], CAR[+1,+5], CAR[−5,−1] (placebo).
- Cross-sectional regression: CAR on scores + controls (topic, size proxy, prior volatility, intraday-release flag).
  Standard errors clustered by event date (results seasons cluster).
- Sort plot: CAR paths by tone tercile.

### 3.3 Volatility (RQ3)
- Target: **Garman–Klass** daily variance (Parkinson as robustness), log-transformed for regression and as a level for QLIKE.
- Baseline B0: HAR (daily/weekly/monthly lags) per firm or pooled with firm fixed effects.
  B1 = B0 + event-type dummies. Model M = B1 + score features (on event days, 0 plus an indicator otherwise).
- Robustness: GARCH(1,1) / GJR-GARCH with an exogenous variance regressor (GARCH-X) on a subset.
- Out-of-sample evaluation: rolling or expanding refits. Losses: QLIKE (primary) and MSE on log-RV. Diebold–Mariano (HAC),
  Mincer–Zarnowitz regressions.

### 3.4 Latent mood state-space model (RQ4)
- State: `m_{i,t} = g_t + u_{i,t}`, with `g_t = φ_g g_{t−1} + ε_t` (market mood) and `u_{i,t} = φ_u u_{i,t−1} + ν_{i,t}`
  (firm part). Time step = trading day.
- Observation: when a scored doc k arrives for firm i on day t, `y_k = m_{i,t} + e_k`, with noise scale shrinking
  as LLM confidence grows. Robust variant: Student-t or bounded (tanh-link) observation, which is the reason for
  using a particle filter at all. No observation on a day → predict step only, so uncertainty grows.
- Estimation: bootstrap particle filter written from scratch in NumPy. Parameters fitted on the **train** period only
  (PF log-likelihood with common random numbers, or PMMH; decided in phase 5).
- **Correctness check:** in the linear-Gaussian case the PF must match an exact Kalman filter (test).
- Forecast features use the **filtered** mood (no smoothing). Smoothed paths appear only in the dashboard, labelled.

## 4. Evaluation rules (binding)

- **No look-ahead:** event-to-session mapping rule in CLAUDE.md. Every feature for a forecast made at day t uses only
  information public before the 17:30 close on day t.
- **Time split:** train 2019-01 → 2022-12 · validation 2023-01 → 2024-06 · test 2024-07 → 2025-12.
  All model, prompt and feature choices are made on train/val. **The test set is evaluated once**, in phase 6, with a
  frozen config that has its git hash recorded.
- **Primary hypotheses fixed before testing:** RQ3 at h = 5 (QLIKE, DM test vs. B1) and RQ4 at h = 5.
  Holm correction across the primary family. Everything else is labelled exploratory (BH-FDR).
- HAC (Newey–West, lag ≥ h−1) or Driscoll–Kraay standard errors for panel regressions. Clustered by date for event CARs.
- Any trading strategy shown includes ≥10 bps per side in costs, plus a cost-sensitivity line.
- Effect sizes come with confidence intervals. A null result is reported plainly.
- Biases listed in the write-up: survivorship (current members), exclusion of 3 foreign filers, index-joiners,
  LLM training-data leakage (the model may "know" what happened after a release).

## 5. Success criteria

- Reproducible: one command per phase rebuilds its outputs from raw caches. `pytest` and `ruff` are clean.
- Total LLM spend ≤ $30 (logged). The pilot step was followed for every prompt version.
- Clear answers to RQ1–RQ5 with CIs, whatever the sign.
- The dashboard runs locally from the processed data and shows mood ± bands, recent announcements with scores, and event-study charts.
- A 2–4 page write-up and a README I can talk through in 5 minutes in an interview.

## 6. Out of scope

- Intraday/tick data, implied volatility (VCAC), options.
- News media, social media, analyst reports. Official filings only.
- Historical index membership, and the 3 foreign-filed members.
- Translation, LLM fine-tuning, the large-model scorer at scale (a Sonnet subsample for robustness is allowed).
- Live trading, real-time ingestion, cloud deployment.
