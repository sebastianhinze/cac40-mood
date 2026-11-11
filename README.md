# CAC 40 Mood

**Question:** does the tone of official CAC 40 announcements predict abnormal returns and, more importantly, future
volatility beyond what a HAR/GARCH model already predicts?

**Approach**
- Official AMF regulated filings (DILA info-financiere) with exact timestamps, plus daily prices (yfinance).
- An LLM (Claude Haiku) scores each filing in French → tone, surprise, uncertainty, topic, confidence. Compared with a
  lexicon baseline and a local-model baseline.
- Event study (abnormal returns) and volatility-forecast study (HAR + announcement dummies vs. + sentiment).
- Signature piece: a latent **mood** state-space model (market + firm AR(1)) estimated with a from-scratch NumPy
  particle filter on irregularly timed LLM observations.
- Streamlit dashboard.

**Status:** phase 1 (data) built, waiting for review. See [PLAN.md](PLAN.md).

## Quick start
```bash
uv sync --all-groups
cp .env.example .env   # add ANTHROPIC_API_KEY
uv run pytest
```

## Rebuild the data (phase 1)
Network steps only add files under `data/raw/`; the others are deterministic and run offline.
```bash
uv run python -m cac40_mood.data.build fetch-tarballs   # DILA archive Jan 2019 - Jun 2020 (14.6 GB read, ~100 min)
uv run python -m cac40_mood.data.build fetch-api        # API metadata 2019-2025 (under a minute)
uv run python -m cac40_mood.data.build fetch-prices     # yfinance OHLC, 37 firms + ^FCHI
uv run python -m cac40_mood.data.build prices           # -> data/processed/prices.parquet
uv run python -m cac40_mood.data.build filings          # -> data/processed/filings.parquet
uv run python -m cac40_mood.data.build fetch-pdfs       # PDFs of the documents to score (~100 min)
uv run python -m cac40_mood.data.build texts            # -> data/processed/texts.parquet
uv run jupyter nbconvert --to html --execute notebooks/01_data_quality.ipynb --output-dir reports
```

## Docs
- [PROJECT_SPEC.md](PROJECT_SPEC.md): research questions, methods, evaluation rules
- [PLAN.md](PLAN.md): build phases and done-when criteria
- [DECISIONS.md](DECISIONS.md): design decisions with reasons
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md): what was tested, licences, fallbacks

Data and LLM outputs are not distributed. Filings are © their issuers via DILA (Licence Ouverte). Prices come via Yahoo (personal use).
