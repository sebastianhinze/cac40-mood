# Memory check: masked re-scoring and release timestamps

## 1. Does the model score differently when names and years are hidden?

100 train/validation documents (35 with negative surprise, 35 positive, 30
zero), re-scored with prompt v2 after hiding names (companies, brands, people, places,
tickers) and replacing years by their distance to the publication year. Numbers were
left in. For comparison: two identical runs on unmasked text differ by about 0.02 in
tone (pilot).

| documents | n | field | correlation | mean_abs_difference | mean_masked_minus_original | same_direction |
|---|---|---|---|---|---|---|
| all | 100 | tone | 0.974 | 0.067 | -0.009 | 0.89 |
| all | 100 | surprise | 0.912 | 0.065 | 0.001 | 0.86 |
| all | 100 | uncertainty | 0.941 | 0.05 | 0.013 | 0.95 |
| company still recognised | 83 | tone | 0.98 | 0.063 | -0.01 | 0.892 |
| company still recognised | 83 | surprise | 0.931 | 0.061 | 0.001 | 0.867 |
| company still recognised | 83 | uncertainty | 0.93 | 0.054 | 0.011 | 0.964 |
| company not recognised | 17 | tone | 0.922 | 0.088 | -0.006 | 0.882 |
| company not recognised | 17 | surprise | 0.702 | 0.082 | 0.0 | 0.824 |
| company not recognised | 17 | uncertainty | 0.979 | 0.032 | 0.021 | 0.882 |

- Surprise changed sign on 1 document(s); 7 went from non-zero to 0.
- **The masking does not hide the company.** Asked to guess from the masked text, the
  model named the right company for 83% of the
  documents and the right year for 68%. The figures
  and the description of the business are enough.

## 2. Returns on original and masked surprise (the sample only)

Coefficient = change in the abnormal return for a surprise higher by 1; standard errors
clustered by date. About 100 documents: wide intervals.

| outcome | regressor | coefficient | ci_low | ci_high | t | p_value | n |
|---|---|---|---|---|---|---|---|
| CAR[-5,-1] | original surprise | 0.0055 | -0.0139 | 0.0249 | 0.5566 | 0.5778 | 100 |
| CAR[-5,-1] | masked surprise | 0.0118 | -0.0057 | 0.0293 | 1.3234 | 0.1857 | 100 |
| CAR[-5,-1] | original minus masked | -0.0343 | -0.1207 | 0.0522 | -0.7767 | 0.4373 | 100 |
| CAR[0] | original surprise | 0.0279 | 0.0086 | 0.0473 | 2.8312 | 0.0046 | 100 |
| CAR[0] | masked surprise | 0.025 | 0.0051 | 0.0449 | 2.4574 | 0.014 | 100 |
| CAR[0] | original minus masked | 0.0207 | -0.0082 | 0.0496 | 1.405 | 0.16 | 100 |

## 3. Is the feed's timestamp later than the firm's own release time?

649 results releases (train and validation);
72 print their own release date and time at the top. For those:

- feed time within 15 minutes of the printed time: 88% (63 of 72);
- feed more than 15 minutes **later**: 0; feed more than 15 minutes **earlier**: 9 (a printed time that is not the release time, e.g. of a conference call);
- median gap 0.0 minutes.

Twenty of them, drawn at random:

| firm_id | printed | feed | minutes_feed_later | event_session |
|---|---|---|---|---|
| FGR | 2019-01-29 17:45 | 2019-01-29 17:45 | 0.0 | 2019-01-30 |
| FGR | 2019-02-27 17:40 | 2019-02-27 17:42 | 2.0 | 2019-02-28 |
| STMPA | 2019-10-24 09:30 | 2019-10-03 15:00 | -29910.0 | 2019-10-03 |
| EN | 2019-11-14 07:30 | 2019-11-06 07:35 | -11515.0 | 2019-11-06 |
| OR | 2020-02-06 18:00 | 2020-02-06 18:00 | 0.0 | 2020-02-07 |
| OR | 2020-07-30 18:00 | 2020-07-30 18:00 | 0.0 | 2020-07-31 |
| OR | 2021-02-11 18:00 | 2021-02-11 18:00 | 0.0 | 2021-02-12 |
| EL | 2021-03-12 07:00 | 2021-03-12 07:00 | 0.0 | 2021-03-12 |
| DG | 2021-10-19 17:45 | 2021-10-19 17:45 | 0.0 | 2021-10-20 |
| GLE | 2021-11-04 21:00 | 2021-11-04 06:43 | -856.1 | 2021-11-04 |
| FGR | 2022-01-26 17:45 | 2022-01-26 17:47 | 2.0 | 2022-01-27 |
| FGR | 2022-02-23 17:40 | 2022-02-23 17:40 | 0.3 | 2022-02-24 |
| FGR | 2022-07-19 17:45 | 2022-07-19 17:48 | 3.6 | 2022-07-20 |
| SGO | 2022-07-27 18:00 | 2022-07-27 18:02 | 2.1 | 2022-07-28 |
| FGR | 2022-08-31 17:40 | 2022-08-31 17:40 | 0.0 | 2022-09-01 |
| OR | 2023-02-09 18:00 | 2023-02-09 18:00 | 0.0 | 2023-02-10 |
| FGR | 2024-01-29 17:45 | 2024-01-29 17:45 | 0.1 | 2024-01-30 |
| CS | 2024-02-22 07:00 | 2024-02-22 07:09 | 9.0 | 2024-02-22 |
| FGR | 2024-02-28 17:40 | 2024-02-28 17:40 | 0.0 | 2024-02-29 |
| EL | 2024-04-18 18:00 | 2024-04-18 18:00 | 0.0 | 2024-04-19 |
