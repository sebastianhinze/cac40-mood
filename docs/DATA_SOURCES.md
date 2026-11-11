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

**B. Coverage hole in the API**
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
