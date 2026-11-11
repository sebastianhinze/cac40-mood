# CLAUDE.md: working rules for this repo

Read first: `PROJECT_SPEC.md` (what/why), `PLAN.md` (phases), `DECISIONS.md` (what's settled), `docs/DATA_SOURCES.md`.

## How we work
- **One phase at a time** (PLAN.md). At the end of each phase: stop, summarise in ≤10 bullets, update DECISIONS.md
  and PLAN.md status, then **wait for my OK** before starting the next phase.
- **I make the key decisions.** Ask before locking in anything that changes the methods, the data, the evaluation or spend.
  Offer 2–3 options with trade-offs and a recommendation. Every decision goes into DECISIONS.md with a one-line reason.
- Keep summaries short and scannable: bullets, no long paragraphs.
- I must be able to explain every line in an interview: no magic, no clever one-liners, no unexplained libraries.

## Statistical hygiene (non-negotiable)
- **No look-ahead bias.** An announcement counts only from the first trading moment after its timestamp:
  - Timestamps are UTC → convert to `Europe/Paris`. Use the **latest** of the provided timestamps (conservative).
  - Trading day, ts < 09:00 → d0 = that session. 09:00 ≤ ts < 17:30 → d0 = that session, `intraday=True`
    (robustness: drop, or shift to d+1). ts ≥ 17:30, weekend or holiday → d0 = **next** trading session.
  - A forecast made at the close of day t may only use documents with d0 ≤ t and prices up to t.
  - The trading calendar comes from `^FCHI` dates, never from weekdays.
- **Strict time split:** train 2019-01→2022-12 · val 2023-01→2024-06 · test 2024-07→2025-12 (DECISIONS #14).
  All tuning happens on train/val. **The test set is touched once**, in phase 6, with a frozen, git-tagged config.
  Parameters (incl. particle-filter parameters, scalers and thresholds) are fitted on train only.
- **Report survivorship bias** from using current CAC 40 members and the 3 excluded foreign filers.
- **Robust inference:** HAC (Newey–West, lag ≥ h−1) or Driscoll–Kraay for panels. Cluster by date for event CARs.
- **Multiple tests:** Holm on the pre-registered primary family. BH-FDR on exploratory results, labelled as exploratory.
- **Trading costs** (≥10 bps per side + sensitivity) on any strategy that is shown.
- **A null result reported honestly is a fine result.** Never p-hack, and never quietly drop a specification.

## LLM cost control
- **Cache every response to disk** (`data/cache/llm/`), keyed by sha256(model + prompt_version + document text).
  Never re-score the same document with the same model and prompt version.
- **Hard spending cap** = `LLM_SPEND_CAP_USD` from `.env`. Log every call's token usage and cost to
  `data/cache/llm/spend.jsonl`. Refuse to send a request when spent + worst-case cost of that request > cap.
- **Pilot first:** any new or changed prompt runs on `LLM_PILOT_SIZE` (=20) docs, I review them, then the bulk run.
- Print a dry-run cost estimate before every bulk run. Use the Message Batches API for bulk.
- Model ID comes from `.env` (`LLM_MODEL`). Don't hard-code it.

## Data rules
- Never commit `.env`, API keys, raw data, prices or LLM outputs. Everything under `data/` stays git-ignored.
- Filter filings by **ISIN**, never by the feed's stale `cac40` flag. Facts about the sources: `docs/DATA_SOURCES.md`.
- Be polite to public servers: generic User-Agent from `.env`, rate limit, cache raw downloads, resumable ingest.
- Raw → processed is deterministic and re-runnable offline from `data/raw/`. Record snapshot dates (prices, universe).
- Don't redistribute full filing texts. The dashboard shows short excerpts plus a link.

## Code
- Readable over clever. Type hints everywhere. Small functions. Docstrings explain **why**, not just what.
- Logic lives in `src/cac40_mood/`. Notebooks only call it and plot, never define core logic.
- Tests are required for: event-time alignment (DST, holidays, after-close, weekends), no-future-information in
  features, AR/CAR computation, the Kalman filter, and the particle filter (PF ≈ KF in the linear-Gaussian case,
  parameter recovery on simulated data). Seed every random process.
- Commands:
  - `uv sync --all-groups`: install
  - `uv run pytest`: tests (`-m "not network"` offline)
  - `uv run ruff check . && uv run ruff format .`: lint/format
  - `uv run streamlit run app/main.py`: dashboard (phase 7)
- Add dependencies with `uv add <pkg>` and give the reason in DECISIONS.md if it matters to the method.

## Layout
```
src/cac40_mood/   package (data/, scoring/, events/, vol/, mood/ subpackages added per phase)
tests/            pytest
notebooks/        exploration & figures only
app/              Streamlit dashboard
data/raw|processed|cache   git-ignored
docs/             data-source notes, write-up drafts
reports/figures/  generated figures
```
