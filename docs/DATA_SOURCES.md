# Data sources: probe results (2026-11-11)

Small samples were actually fetched for each source. Scripts were throwaway and are not in the repo.

## 1. Announcements: DILA "info-financiere" (French OAM for AMF regulated information) ✅ primary

**A. Opendatasoft API** (works, no key)
- Endpoint: `https://www.info-financiere.gouv.fr/api/explore/v2.1/catalog/datasets/flux-amf-new-prod/records`
- ~540k filings across all issuers, 2008 to today. One row per document: issuer name, ISIN, LEI, ticker,
  AMF type/subtype (FR + EN labels), language, title, `url_de_recuperation` (PDF link).
- Timestamps: `uin_dat_amf`, `uin_dat_mar`, `informationdeposee_inf_dat_emt`. They are **true UTC**, which
  I checked by DST: LVMH results land at 16:45 UTC in January and 15:45 UTC in July, i.e. 17:45 Paris both times.
  Ignore the `timezone_uin_dat_amf` field.
- Filter by **ISIN**. Don't use the `cac40`/`name_cac40` flag: it is a stale ~2011 list (EADS, Lafarge, Technip…).
- Query tips: ODSQL. `group_by=year(uin_dat_amf) as yr` works; `in (...)` does not, so use OR chains.
  `search(field,'X')` does fuzzy name matching.
- PDF download tested (LVMH Q1 2024, 3 pages, 10.8k chars). pypdf extraction works but adds spaces inside words
  ("cr oissance", "20 24"). Picking the extractor is a phase-1 decision.

**B. Coverage hole in the API** *(partly superseded by section 6: the "2019 almost empty" reading was taken while
the API index was being rebuilt; the real hole is 5 Mar – 15 Jun 2020 and it is in the source itself)*
- Jul 2018 – Jun 2020 is almost empty (2019: about 1k rows vs. about 25k in a normal year).
- **Fix (tested):** the raw DILA archive `https://echanges.dila.gouv.fr/OPENDATA/AMF/` has
  `amf-gz_DDMMYYYY_HHMM.tar.gz` bundles from **Jan 2019** on, each about 7 MB, about 180 per month, all issuers.
  Each bundle holds `xml/…/MD*.xml` metadata (same fields, timestamps **with an explicit offset**, e.g.
  `+02:00`) plus `pdf/…/FC*.pdf`. PDFs are also served directly at the same `fr.ftp.opendatasoft.com/...` path.
- **Jul–Dec 2018 is missing from every source I tried**, which is why the sample starts in 2019.
- Cost of the fix: about 2.4k tarballs (about 17 GB) for Jan 2019–Jun 2020. Stream each one, keep the XML plus the PDFs
  of our 37 ISINs, discard the rest.

**C. Coverage by issuer**
- French-domiciled members: good. For example, over 2021–25 the 37-firm sample has about 16k filings (FR+EN).
- **Missing or sparse:** Airbus (NL, 0 rows), ArcelorMittal (LU, ~0 since 2021), Stellantis (NL, ~4/yr).
  These firms file with their home regulator. **Excluded** (see DECISIONS.md).
- Mostly English: STMicroelectronics, Eurofins, Euronext. Policy: use the French version when it exists, otherwise English, and keep a language flag.
- FR and EN versions arrive as separate rows with the same timestamp and subtype, so they need deduplicating.

**D. Volume (37 firms, 2021–25, both languages)**
- Inside information ("Informations privilégiées" = real press releases): about 3.7k per language.
- Admin filings: voting-rights totals ~3.2k, buyback logs ~2.7k, threshold crossings ~1.1k.
- Reports and URDs: a few hundred (long PDFs).

**Licence:** data.gouv.fr lists the DILA "INFO-FINANCIERE" dataset under **Licence Ouverte (Etalab)**, which allows reuse
with attribution. The PDFs are issuer-authored, so don't redistribute full texts. Show short excerpts plus a link.
Raw files stay out of git.

## 2. Prices: yfinance ✅ primary
- Tested 2018-01-02 → 2025-12-30: MC.PA, TTE.PA, SAN.PA, AIR.PA, STLAP.PA, MT.AS, ^FCHI.
  2,048 rows each, OHLC + Adj Close + Volume, no NaN closes. The index is tz-naive (exchange-local dates).
- Use Adj Close for returns (dividends) and raw OHLC for range estimators (the intraday ratios are unaffected by
  adjustment).
- **Terms:** yfinance is an unofficial Yahoo scraper. Yahoo's terms allow personal, non-commercial use only. So: never commit
  or redistribute prices, cache locally, and pin a snapshot date.
- Fallbacks: **Stooq** CSV now sits behind a JavaScript challenge, so it can't be automated. Next option is a manual CSV export from Euronext
  live (allowed under its terms for personal use), then paid EODHD (~€20/month) if needed.

## 3. Company IR sites / Euronext news pages ⚠️ not used
- Airbus newsroom 200. ArcelorMittal and Stellantis 403 (bot protection). Euronext company-news page 200.
- Weaker timestamps and less clear terms than the OAM. Kept as a documented fallback only.

## 4. Universe: current CAC 40 (snapshot 2026-11-11, from en.wikipedia "CAC 40", CC BY-SA)
Cross-check against Euronext's official composition page in phase 1.
Accor, Air Liquide, Airbus*, ArcelorMittal*, AXA, BNP Paribas, Bouygues, Bureau Veritas, Capgemini, Carrefour,
Crédit Agricole, Danone, Dassault Systèmes, Eiffage, Engie, EssilorLuxottica, Eurofins, Euronext, Hermès, Kering,
L'Oréal, Legrand, LVMH, Michelin, Orange, Pernod Ricard, Publicis, Renault, Safran, Saint-Gobain, Sanofi,
Schneider Electric, Société Générale, Stellantis*, STMicroelectronics, Thales, TotalEnergies, URW, Veolia, Vinci.
(* = excluded, missing from the French OAM.)
Some members (e.g. Bureau Veritas, Eiffage, Euronext) joined the index during the sample. This adds to the survivorship/selection bias.

## 5. LLM pricing (checked 2026-11-11)
- `claude-haiku-5-5`: $0.10 / $0.50 per M input/output tokens (prompts ≤100k). Message Batches cost 50% less.
- Rough estimate: about 6–7k substantive docs × (~4k doc + ~1k prompt tokens in, ~400 out) ≈ **$5 standard, about $2.5 batch**.
  That leaves room under the $30 cap for prompt iteration and a Sonnet robustness subsample.

## 6. Phase 1 findings (2026-11-11), measured on the full download

**API behaviour**
- The search index behind the API is **rebuilt several times a day**. During a rebuild whole years of rows disappear
  and come back in chunks of about 10k rows per minute (seen live: 2019 went 25.8k → 10.6k → 25.8k rows within minutes).
  Rule in the code: a snapshot is accepted only if the live row count equals the catalog's `records_count` **before and
  after** the download (`amf_api.fetch_snapshot`). The whole 40-ISIN snapshot takes under a minute.
- With a complete index the API does cover 2019 (about 25.8k rows for all issuers). The earlier "about 1k rows in 2019"
  was a mid-rebuild reading.
- `exports/jsonl?where=…` returns all rows of one ISIN in one request (no 10k limit).

**The real hole: 5 March – 15 June 2020**
- The feed is at roughly 5% of normal volume for our firms in that window, in the API, in the archive bundles and in
  the per-distributor folder tree alike (e.g. `MKW/2020/`: 700 PDFs in Feb, 150 in Mar, 86 in Apr, 67 in May, 962 in Jul).
  The distribution chain into the OAM was disrupted during the first lockdown; the documents were never loaded.
- The AMF's own database (BDIF) holds prospectuses, thresholds and takeover documents, not issuer press releases, so
  it cannot fill the hole. Wire-service and issuer sites could, with weaker timestamps (not done; DECISIONS #23).
- In the tables the window is flagged `in_feed_gap`.

**ISINs**
- Three firms changed ISIN after a share split: Dassault Systèmes (FR0000130650 → FR0014003TT8, Jul 2021), Eurofins
  (FR0000038259 → FR0014000MR3, Nov 2020), Michelin (FR0000121261 → FR001400AJ45, Jun 2022). Filtering on the current
  ISIN alone loses everything before the split. `universe.csv` lists both.
- The feed's LEI field is unreliable (empty before 2019-04/12, outdated for AXA, EssilorLuxottica, Publicis; empty for
  Euronext and STMicroelectronics). LEIs in `universe.csv` come from GLEIF. Never filter on the feed's LEI.

**Timestamps**
- `uin_dat_mar` ("market date") holds placeholders in the years **8887/8888** when unset; they must be ignored.
- The three timestamps differ for about a quarter of documents. Typical case: a report is *deposited* at 14:33 with a
  *market release* at 18:00. The "latest" rule dates it after the close, which is right.
- In about 1% of documents the AMF timestamp is more than a day later than the market timestamp (late filing with the
  OAM). The "latest" rule then dates the event late: no look-ahead, but the true day-0 reaction is missed. Column
  `ts_spread_minutes` lets the event study drop them as a robustness check.
- French and English versions are separate rows filed seconds to minutes apart (99% within 3 minutes), not at the
  exact same timestamp.

**Type codes** (`INF_STP_PRI`, six digits; the API's text labels are coarser than the codes)
- 001xxx and 015000 inside information (001007 = results, 001002 = operations, 001006 = other press releases),
  005xxx quarterly information, 003000/004000/300005 annual and half-year reports and URDs, 008000 governance,
  009xxx takeover releases, 002xxx "document is available" notices, 006000 voting rights, 013xxx own-share
  transactions and liquidity contracts, 023000 threshold crossings, 020000 bond final terms.
- A few attachments are not PDFs (`.docx`, `.htm`, `.zip`).

**Prices (yfinance)**
- 36 of 37 tickers: complete 2018-01-02 → 2025-12-31, no missing sessions, no OHLC inconsistencies.
- **URW.PA starts on 2023-04-14** on Yahoo. Before that the main listing was Amsterdam and Yahoo no longer serves that
  line; the Frankfurt line (1BR1.DE) barely trades, so its daily range is meaningless.
- Yahoo carries a stale row for 36 stocks on 2019-12-25 (exchange closed). Rows on non-index days are dropped.

**Universe cross-check**
- Euronext's composition page (dated 2026-11-11) lists the same 40 names; all 37 current ISINs match. The JSON endpoint
  behind that page is obfuscated for scripts, so the check was done by reading the page.
- Index entries inside or near the sample (fr.wikipedia "CAC 40", composition history): Hermès 2018-06, Dassault
  Systèmes 2018-09, Thales 2019-06, Eurofins 2021-09, Accor 2024-03 (it had left in 2020-09), Bureau Veritas 2024-12,
  Euronext 2025-09, Eiffage 2025-12.

**PDF text extraction** (111 real filings, first 5 pages, long-word agreement between extractors)
| Extractor | Licence | Words recovered | Junk words | Time for 111 files |
|---|---|---|---|---|
| pypdfium2 | Apache-2.0 / BSD | 98–100% | 0.1–0.3% | 1.5 s |
| PyMuPDF | AGPL | 98–100% | 0.0–0.1% | 3.5 s |
| pypdf | BSD | 97–99% | 0.6–1.8% | 15.8 s |
| pdfplumber | MIT | 97–98% | 0.5–3.6% | 31.1 s |

## 7. LLM scoring: measured in the phase 2 pilot (2026-11-11)
- **Tokens:** French filing text has a median of **2.37 characters per token** on `claude-haiku-5-5` (20 documents, range
  2.0–2.75; the one English document 2.75). The earlier guess of about 3 was too optimistic.
- **Fixed overhead per request:** about 1,120 input tokens (instructions about 600, JSON schema about 520).
- **Output:** about 125 tokens per document with thinking off or at low effort, on Haiku and on Sonnet alike.
- **Prices (Anthropic pricing page):** Haiku 5.5 $0.10 / $0.50 per million input / output tokens for prompts up to 100k
  tokens; Sonnet 5.5 $2 / $10. The Batches API costs half. The token-counting endpoint is free.
- **Full run, 7,223 documents, measured token counts:** cap 4,000 tokens $3.06 standard / $1.53 batch; cap 8,000
  $3.81 / $1.91; cap 12,000 $4.36 / $2.18.
- **Pilot spend:** $0.57 in total (Haiku $0.13 for 209 calls, Sonnet $0.44 for 50 calls).
- Haiku 5.5 rejects a temperature setting, so identical requests can return slightly different scores; the pilot
  measures that noise by scoring twice.
- Do not run two scoring jobs at once: both can miss the cache for the same document and pay for it twice (this
  happened once in the pilot, cost $0.0004).
- **Prompt v2 at the 12,000-token cap (same day):** fixed overhead about 2,170 input tokens per request (longer
  instructions), about 122 output tokens. Full run of 7,223 documents: **$5.08 standard / $2.54 batch**.
  Pilot spend to date: $1.39 (v1 and v2 pilots, two Sonnet comparisons).

## 8. Firm size: revenue per fiscal year (2026-11-12)

**What it is for:** the scoring prompt tells the model the issuer's approximate annual revenue so it can judge
materiality. The figures are *not* typed in by hand.

**Source:** each firm's own annual-results announcements in our filings table (train and validation periods only).
- Candidates: every document published January–April, plus every filing coded as a results release (001007, 001008)
  in any month: 2,386 documents.
- Extraction: `scoring/revenue.py` asks `claude-haiku-5-5` (prompt `revenue-v2`, first 15,000 characters) whether the
  document announces the listed company's full-year results and, if so, for the revenue of that year and of the year
  before, the currency, and a verbatim quote. 362 documents were classified as full-year announcements.
- Table: `data/processed/firm_revenue.parquet`, one row per firm and fiscal year with revenue, currency, basis
  ("reported" from that year's own release, or "comparative" = the previous-year figure in the next release), the
  source `doc_id`, its publication time and the quote. Facts per document: `firm_revenue_facts.parquet`.

**No look-ahead:** revenue of the fiscal year ending in year F is used for documents dated 1 July F+1 to 30 June F+2.
If the table has no figure for that year, the latest earlier year is used.

**Coverage and known weaknesses**
- 246 of 259 firm-years for fiscal years 2017–2023 (211 reported, 35 comparative).
- First pass (`revenue-v1`) picked up a subsidiary's revenue for Eiffage (APRR files under Eiffage's ISIN) and mixed
  Crédit Agricole S.A. with the wider Crédit Agricole group. `revenue-v2` names the listed company and excludes
  subsidiary announcements; both series are consistent now.
- TotalEnergies: only fiscal years 2017 and 2018 were extracted (USD 149bn and 184bn, "revenues from sales"); later
  results releases show revenue beyond the text that is read. All later TotalEnergies documents therefore use the
  2018 figure. Right order of magnitude, but stale.
- URW: the extracted figures are inconsistent (0.9–2.5bn). URW is outside the main analysis.
- 119 documents (1.7%) from the first months of the sample have no earlier figure for their firm (Sanofi, Pernod
  Ricard, Air Liquide, Dassault Systèmes, Bouygues, L'Oréal) and use the earliest later year instead; they are
  flagged `size_backfilled`. For 28 of them that year had not been published yet at the document's date. The figure
  only enters the prompt as a rounded size ("about EUR 9.2 billion").
- Two firm-years are flagged `needs_check` because two releases disagree by more than 15% (Engie 2020 after a
  restatement, URW 2018).
- Cost: $0.48 (first pass) + $0.62 (second pass), Batches API.

## 9. Loughran-McDonald dictionary (baseline, added 2026-11-12)

- **What:** Loughran-McDonald Master Dictionary with sentiment word lists (negative 2,355 words, positive 354,
  uncertainty 297). English only.
- **Source:** https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (University of Notre Dame, Software
  Repository for Accounting and Finance). CSV version covering 1993–2025, page updated March 2026, downloaded
  2026-11-12 from the Google Drive link on that page.
- **Licence, as stated on the page:** "The dictionary/sentiment lists are free for use in academic research."
  Commercial licences: contact the authors. This project is non-commercial research; the file stays in
  `data/raw/lexicon/` (git-ignored) and is not redistributed. **If the project were ever used commercially,
  this baseline would need a licence or would have to be removed.**
- **Reference:** Loughran and McDonald (2011), "When is a liability not a liability?", Journal of Finance.

## 10. Official English versions of French filings (added 2026-11-12)

- The feed often carries an announcement twice, French and English. Phase 1 kept the French one and set the
  English twin aside (`is_translation`, `twin_doc_id` in `filings_all.parquet`).
- For the Loughran-McDonald baseline the English twins of train/validation documents are downloaded from the
  same DILA server with the same polite settings (`build fetch-pdfs-en`, `build texts-en` →
  `data/processed/texts_en.parquet`). 4,565 French documents to score have a twin.
- The twins are the firms' own English versions, not machine translations.


## 11. What may be shared: terms checked before going public (2026-11-12)

**Filings (DILA "info-financière", the French OAM for AMF regulated information)**
- The dataset page on data.gouv.fr (https://www.data.gouv.fr/datasets/info-financiere/informations) lists the
  licence as **Licence Ouverte / Open Licence (Etalab)**: reuse allowed, including commercial, with attribution
  (source and date of last update). Two extra conditions on that page: reusers must take into account the update
  requests DILA posts on the dataset's discussion forum, and the licence does not exempt anyone from French data
  protection law (Informatique et Libertés) where documents contain personal data. The API's own catalogue
  metadata has an empty licence field.
- The Licence Ouverte does not cover material on which third parties hold rights. The PDFs are written by the
  issuers, so the conservative reading stays as before: **metadata (issuer, title, timestamps, type codes) may be
  shown with attribution; full texts are not redistributed.** The dashboard shows at most about 400 characters
  of a document plus a link to the official PDF on the DILA server.
- Personal data: some filings name executives (share dealings, governance). The repository contains none beyond
  document titles; the dashboard runs locally on local data.

**Other inputs**
- Prices (Yahoo via yfinance): personal, non-commercial use. No prices and no per-firm price-derived series are
  committed; the reports and figures contain aggregates only (mean abnormal returns, loss differences, test
  statistics).
- Loughran-McDonald dictionary: free for academic research, not to be redistributed. It lives only in
  `data/raw/lexicon/` (git-ignored); the code reads it from there and contains no part of it.
- LLM outputs (scores, rationales): Anthropic's terms leave outputs to the user. They are published in the bundle
  below (DECISIONS #67).

**Public data bundle (`public_data/`, DECISIONS #67)**
- Built by `uv run python -m cac40_mood.public_bundle` from the local pipeline outputs; committed so the dashboard
  runs on a fresh clone and on Streamlit Community Cloud. About 2 MB.
- Contents: `announcements.parquet` (6,860 scored, non-administrative filings of the 36 firms: issuer, title,
  publication time, event session, period, language, topic, tone, surprise, uncertainty, confidence, the model's
  one-line rationale, link to the official PDF on the DILA server); `mood.parquet` (filtered mood per firm and
  trading day, 2019-2025, with its 95% band; the test-period part is for display only, DECISIONS #69);
  `event_car_paths.csv` (average cumulative abnormal return by tone third, train + validation) and the event-study
  and volatility tables as published in `reports/`; `manifest.json` (sources, row counts, git commit).
- **Not included:** prices or any per-firm daily series derived from them; filing texts or excerpts (the public
  dashboard shows only the link to the official PDF); the Loughran-McDonald word list.
- **Attribution (Licence Ouverte 2.0):** "Filing metadata (issuer, title, timestamps, document links): DILA,
  info-financière (https://www.info-financiere.gouv.fr), data received from the AMF; reused under the Licence
  Ouverte / Open Licence 2.0 (Etalab). Feed snapshot of 2026-11-11." This line is in the bundle's `manifest.json`
  and `README.md`, in the dashboard's sidebar and in the repository README.
- Derived values (scores, rationales, mood, aggregate tables) are offered under CC BY 4.0.
- Update requests that DILA posts on the dataset's data.gouv.fr discussion forum apply to this bundle too: a
  withdrawn or corrected filing is removed or corrected here by rebuilding the bundle.

**What the repository contains:** code, tests, docs, the reports in `reports/` (aggregate tables; a few tables
list document titles and issuer names, `02_gold_agreement.md` lists two), aggregate figures, dashboard
screenshots, the public data bundle above, and the executed data-quality notebook. The notebook's excerpt
output (three filing excerpts, one with a press-office phone number and e-mail) was removed on 2026-11-12 and the
HTML report regenerated from it (DECISIONS #70); earlier commits in the git history still contain that output.

**Audit of the git history (all commits, 2026-11-12):** no `.env` was ever committed; no API-key pattern
(`sk-ant-…` or long key assignments) appears in any diff; nothing under `data/` was committed except `.gitkeep`
files; no Loughran-McDonald word list. Done before the first push to GitHub.
