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
