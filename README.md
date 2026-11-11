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

**Status:** phase 0 (setup) done. See [PLAN.md](PLAN.md).

## Quick start
```bash
uv sync --all-groups
cp .env.example .env   # add ANTHROPIC_API_KEY
uv run pytest
```

## Docs
- [PROJECT_SPEC.md](PROJECT_SPEC.md): research questions, methods, evaluation rules
- [PLAN.md](PLAN.md): build phases and done-when criteria
- [DECISIONS.md](DECISIONS.md): design decisions with reasons
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md): what was tested, licences, fallbacks

Data and LLM outputs are not distributed. Filings are © their issuers via DILA (Licence Ouverte). Prices come via Yahoo (personal use).
