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
