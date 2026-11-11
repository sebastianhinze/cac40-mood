# Limitations (running list for the write-up)

Each item says what the limitation is, which way it could bias results, and what we do about it.

| # | Limitation | Likely effect | Handling |
|---|-----------|---------------|----------|
| 1 | **Survivorship / selection:** the universe is today's CAC 40 (snapshot 2026-11-11). Six firms joined during or after the sample start (Thales 2019, Eurofins 2021, Accor 2024, Bureau Veritas 2024, Euronext 2025, Eiffage 2025) | Firms that did well are over-represented; average returns are flattering. Less clear for volatility forecasts | Reported, not corrected (DECISIONS #2) |
| 2 | **Three foreign filers excluded:** Airbus, ArcelorMittal, Stellantis file with the Dutch / Luxembourg regulator | Three large industrials missing | Reported (DECISIONS #3) |
| 3 | **URW excluded from the main analysis (36 firms):** Yahoo has its prices only from 2023-04-14 | One real-estate firm missing; the only listed property company in the index | Filings kept in the tables; no returns or volatility used (DECISIONS #29) |
| 4 | **Feed gap 5 Mar – 15 Jun 2020:** the official feed has about 5% of its usual volume | The COVID crash, the period with the most extreme volatility, cannot inform the announcement results | Window excluded and treated as missing, never as "no news"; particle filter predict-only; robustness run from 2020-07-01 (DECISIONS #28) |
| 5 | **Late filings:** in about 1% of documents the regulator's timestamp is more than a day after the market release; the "latest timestamp" rule then dates the event late | No look-ahead, but the day-0 reaction is missed for those events (weakens RQ1) | `ts_spread_minutes` allows dropping them as a robustness check |
| 6 | **Text coverage 96.9%:** 152 documents are not PDFs (mostly annual reports filed as ESEF packages from 2021 on), 47 are scanned images, 19 are mostly tables | The unread documents are mainly long periodic reports, not press releases | Flagged in `texts.parquet`; not scored |
| 7 | **Only the first 15 pages / about 4k tokens are read** | Tone deep inside long reports is ignored | By design (DECISIONS #7, #20) |
| 8 | **A few firms file far more than the rest:** TotalEnergies is about 18% of documents to score; its press-release filings drop sharply in 2025 | Pooled results lean on heavy filers; filing behaviour is not constant over time | Firm fixed effects; per-firm robustness in later phases |
| 9 | **Prices from Yahoo (unofficial)**, daily only | Possible revisions; no intraday reaction for the 22% of filings made during trading hours | Snapshot date recorded; intraday filings flagged, robustness shifts them to the next session |
| 10 | **LLM training-data leakage:** the model may know what happened after a release | Scores could contain hindsight | Discussed in the write-up (spec section 4) |
