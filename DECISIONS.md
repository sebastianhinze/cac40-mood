# Decision log

✅ = agreed with me · 🟡 = proposed by Claude, needs my OK · ❓ = open, decide in the named phase.
Append-only. If a decision is reversed, add a new row that references the old number.

| # | Date | Decision | Reason | Status |
|---|------|----------|--------|--------|
| 1 | 2026-11-11 | Event sample **2019-01-01 → 2025-12-31** (prices from 2018-01 for warm-up) | The official feed has no filings for Jul–Dec 2018, so starting in 2019 avoids the gap. 7 years incl. COVID and the 2022 regimes | ✅ |
| 2 | 2026-11-11 | Universe = **current CAC 40 members**, survivorship bias reported, not fixed | Reproducible, and prices are free. Rebuilding historical membership + delisted prices is out of budget | ✅ |
| 3 | 2026-11-11 | **Exclude Airbus, ArcelorMittal, Stellantis** → 37 firms | They file with the NL/LU regulators, so they're almost absent from the French OAM. Keeps one consistent official source | ✅ |
| 4 | 2026-11-11 | Primary text source = **DILA info-financiere** (API + DILA tarballs for Jan 2019–Jun 2020) | Official regulated information, exact UTC timestamps, Licence Ouverte. Tested in DATA_SOURCES.md | ✅ |
| 5 | 2026-11-11 | **Score all regulated-info types, tagged by topic**. Admin types counted but not LLM-scored | Most events. Topic tags allow per-type analysis. Admin filings have no tone | ✅ |
| 6 | 2026-11-11 | LLM reads the **French original** (English if that's the only version), with a `lang` column | Haiku reads French well. No translation cost or drift | ✅ |
| 7 | 2026-11-11 | Long docs: **first ~4k tokens** (headline + summary), truncation logged | Tone lives in the opening. Keeps cost bounded. Avoids a chunk-aggregation rule | ✅ |
| 8 | 2026-11-11 | LLM = **claude-haiku-5-5**, hard cap **$30**, Batches API for bulk | About $5 expected ($0.10/$0.50 per M tok). Leaves room to iterate prompts + a Sonnet subsample | ✅ |
| 9 | 2026-11-11 | Baselines: **French finance lexicon + local multilingual sentiment model** | Two levels of a "cheap" alternative. Makes the LLM's added value measurable | ✅ (exact lexicon/model ❓ phase 2) |
| 10 | 2026-11-11 | Latent mood = **market factor + firm-specific AR(1)** | Separates market-wide from firm news. Richer story than 37 independent filters | ✅ (filter architecture ❓ phase 5) |
| 11 | 2026-11-11 | Vol target **Garman–Klass range variance** (Parkinson robustness). Baseline **HAR**, GARCH(1,1)/GJR-X as a check | Free data is daily OHLC only. Range estimators are far less noisy than squared returns. HAR takes extra regressors easily | ✅ |
| 12 | 2026-11-11 | Prices = **yfinance**, local only, snapshot date recorded | Tested, complete for 2018–2025. Yahoo terms are personal-use only, so never redistributed | ✅ |
| 13 | 2026-11-11 | Timestamp = **latest** of the feed's UTC timestamps → Europe/Paris. After-close mapping rule in CLAUDE.md | Being too late is harmless. Being too early is look-ahead | ✅ |
| 14 | 2026-11-11 | Time split: train 2019-01→2022-12 · val 2023-01→2024-06 · **test 2024-07→2025-12 (one touch)** | About 4y / 1.5y / 1.5y. Test is long enough for ~2 results seasons per firm | ✅ |
| 15 | 2026-11-11 | RQ3 baseline must include **announcement-type dummies** (B1), not just HAR | Otherwise "sentiment helps" may only mean "earnings days are volatile" | ✅ |
| 16 | 2026-11-11 | Primary hypotheses: RQ3 and RQ4 at **h = 5**, QLIKE + DM, Holm. Everything else exploratory | Pre-registration guards against forking paths | ✅ |
| 17 | 2026-11-11 | **Gold set**: I hand-label ~50 docs blind in phase 2 | The only direct check that LLM scores measure tone | ✅ |
| 18 | 2026-11-11 | Tooling: **uv**, Python 3.12, ruff, pytest, src-layout package `cac40_mood` | Fast, reproducible lockfile, standard layout | ✅ |
| 19 | — | PDF text extractor (pypdf vs. pdfplumber vs. PyMuPDF/AGPL) | pypdf adds spaces inside words in the test | ❓ phase 1 |
| 20 | 2026-11-11 | PDF extractor = **pypdfium2** (answers #19). Only the **first 15 pages** of each PDF are read | Best word recovery and least junk on 111 real filings, 10–20× faster than pypdf/pdfplumber, permissive licence (PyMuPDF is AGPL). The LLM reads ~4k tokens anyway (#7) | ✅ |
| 21 | 2026-11-11 | Admin list (#5) extended by **"document available" notices (002xxx)** and **bond final terms (020000)**: counted, not scored | Both are pointers or term sheets without tone (about 820 documents). Class is a column, so this is reversible without re-ingesting | ✅ (checked on 10 random excluded documents: all notices or bond term sheets) |
| 22 | 2026-11-11 | Filings are matched on **every ISIN a firm has had** (pre-split ISINs of Dassault Systèmes, Eurofins, Michelin) | The current ISIN alone drops all filings before the split | 🟡 |
| 23 | 2026-11-11 | **Feed gap 5 Mar – 15 Jun 2020** (source itself is ~95% empty for our firms). Rows are flagged `in_feed_gap`; treatment still open: (a) keep the window out of event/vol estimation, (b) backfill from wire/issuer sites with weaker timestamps, (c) start the sample in Jul 2020 | Found in phase 1. The plan's "no hole in 2019–20" cannot be met from the official source | → decided in #28 |
| 24 | 2026-11-11 | **URW prices exist only from 2023-04-14** on Yahoo. Kept in the filings table; treatment still open: (a) use URW from 2023-04 only, (b) manual Euronext CSV export for the earlier years, (c) drop URW (36 firms) | The Amsterdam line is gone from Yahoo; the Frankfurt line barely trades | → decided in #29 |
| 25 | 2026-11-11 | FR/EN de-duplication: an English filing is dropped if the same firm filed a French one with the **same type code within 30 minutes**; each French filing absorbs one English filing | 99% of real pairs are under 3 minutes apart; exact-timestamp matching (as assumed in DATA_SOURCES) would miss most pairs | ✅ |
| 26 | 2026-11-11 | Sources: archive bundles for Jan 2019–Jun 2020 (as #4) **plus the API for the whole sample**; if a document is in both, the archive copy wins. An API snapshot is accepted only when the index is complete before and after the download | The API index is rebuilt during the day and returns partial answers meanwhile; the bundles are immutable and double as a completeness check | ✅ |
| 27 | 2026-11-11 | New dependencies: pypdfium2 (replaces pypdf), matplotlib (figures), nbconvert/nbformat (dev, to render notebooks) | Tooling only, no effect on the method | 🟡 |
| 28 | 2026-11-11 | **Feed gap 5 Mar – 15 Jun 2020: excluded from estimation and evaluation and marked as MISSING, never as "no news"** (answers #23). Announcement features are missing there, not zero. The particle filter runs predict-only through the window (no observations, uncertainty grows). **Robustness run with the sample starting 2020-07-01** | A day without a filing inside the hole says nothing about the firm. Treating it as "no news" would bias event dummies and the mood towards calm exactly during the COVID crash | ✅ |
| 29 | 2026-11-11 | **URW is excluded from the main analysis → 36 firms** (answers #24). Its filings stay in the data tables; listed under limitations | Yahoo has URW prices only from 2023-04-14, so no returns or volatility for most of the sample | ✅ |
