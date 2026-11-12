# Run guide

Everything below runs from the repository root. Network steps only add files under `data/raw/`;
every other step is deterministic and runs offline from what is already there. `data/` is
git-ignored: nothing it holds is distributed (see `docs/DATA_SOURCES.md` §11).

## 0. Install

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
cp .env.example .env        # then add ANTHROPIC_API_KEY (only needed for the scoring steps)
uv run pytest -m "not network"
```

Without uv: `python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`
(`requirements.txt` is exported from `uv.lock` with `uv export --no-hashes --all-groups --no-emit-project`).

## 1. Without any data: figures and dashboard

```bash
uv run python -m cac40_mood.readme_figures      # README charts, read from reports/*.md
uv run streamlit run app/main.py                # dashboard, reads public_data/ and reports/
```

## 2. Rebuild the data (phase 1, a few hours, polite rate limits)

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

Prices from Yahoo change slightly over time (revisions); the snapshot used here was taken on
2026-11-11, so a rebuild can differ in the last digits.

## 3. Score the filings (phase 2, about $5, needs an API key)

Every call is cached in `data/cache/llm/` by a hash of the full request and logged to
`data/cache/llm/spend.jsonl`; the code refuses any request that could push the total past
`LLM_SPEND_CAP_USD`.

```bash
uv run python -m cac40_mood.scoring.pilot count-tokens    # real token counts (free endpoint)
uv run python -m cac40_mood.scoring.pilot pilot           # 20 documents, scored twice
uv run python -m cac40_mood.scoring.pilot reference       # 50 documents, Haiku vs Sonnet
uv run python -m cac40_mood.scoring.run revenue           # firm size per fiscal year (for the prompt)
uv run python -m cac40_mood.scoring.run bulk --dry-run    # cost estimate only
uv run python -m cac40_mood.scoring.run bulk              # score every document (Message Batches API)
uv run python -m cac40_mood.scoring.baselines lexicon
uv run python -m cac40_mood.scoring.baselines local-model # needs the "baselines" dependency group
uv run python -m cac40_mood.data.build fetch-pdfs-en  # official English versions (about 40 min)
uv run python -m cac40_mood.data.build texts-en
uv run python -m cac40_mood.scoring.loughran          # needs the Loughran-McDonald CSV, see below
uv run python -m cac40_mood.scoring.sanity
```

The Loughran-McDonald dictionary is free for academic research but may not be redistributed.
Download the master dictionary CSV from https://sraf.nd.edu/loughranmcdonald-master-dictionary/
and save it as `data/raw/lexicon/download/lm_master_dictionary.csv`.

Gold set (my 40 blind labels) and the memory check:

```bash
uv run streamlit run app/label_gold.py                # labelling page; labels -> data/processed/gold/labels.csv
uv run python -m cac40_mood.scoring.gold_report       # refuses to run before all 40 are labelled
uv run python -m cac40_mood.scoring.masking full      # re-score 100 documents with names hidden
uv run python -m cac40_mood.scoring.memory_report
```

## 4. Analysis on train and validation (phases 3-5)

```bash
uv run python -m cac40_mood.events.run           # event study
uv run python -m cac40_mood.vol.run              # HAR / GARCH baselines
uv run python -m cac40_mood.mood.real            # fit the mood model on train, filter through validation
uv run python -m cac40_mood.vol.sentiment_run    # HAR with announcement, score and mood columns
uv run python -m cac40_mood.mood.sv_run          # nonlinear mood model (about 15 min)
uv run python -m cac40_mood.vol.nonlinear_run    # its pre-registered validation test
uv run python -m cac40_mood.vol.garchx           # GARCH-X robustness on ten firms
```

## 5. The test period (phase 6): already done, once

```bash
uv run python -m cac40_mood.phase6 rehearsal     # same pipeline on the validation period
uv run python -m cac40_mood.phase6 test          # refuses: the report exists, and HEAD must be the tag
```

The test was run once on tag `phase6-freeze`; `reports/06_test_results.md` is that run's output.
The runner refuses a second run once that report exists.

## 6. Public data bundle and dashboard

```bash
uv run python -m cac40_mood.public_bundle       # rebuild public_data/ from data/processed/
uv run streamlit run app/main.py
```

The dashboard reads `public_data/` and `reports/`; no API calls. When `data/processed/texts.parquet`
exists locally it also shows a short excerpt of each filing (never in the public version).

## 7. Deploy the dashboard on Streamlit Community Cloud

The repository is ready for it: entry point `app/main.py`, dependencies in `app/requirements.txt`
(read before anything at the repository root), theme in `.streamlit/config.toml`, data in
`public_data/` and `reports/`. No secrets are needed.

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository and allow
   Streamlit to read it.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository `<owner>/cac40-mood`, branch `main`, main file path `app/main.py`.
4. Optional: choose an app URL (for example `cac40-mood`).
5. **Advanced settings** → Python version **3.12**. No secrets.
6. **Deploy**. The first build takes a few minutes; later pushes to `main` redeploy automatically.
