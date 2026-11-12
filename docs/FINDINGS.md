# CAC 40 Mood: findings on one page

**Question.** Does the tone of official CAC 40 announcements predict abnormal returns and, above all, future
volatility beyond what a standard volatility model already predicts?

**Answer.** No for volatility: on the untouched test period neither the scores nor the mood model improved the
forecasts. Yes for the day-0 price reaction: prices move with the tone on the announcement day, in every period,
and do not drift afterwards.

## Data and method

- 36 CAC 40 firms, 2019–2025; 7,223 non-administrative regulated filings, 6,996 scored by an LLM (tone,
  surprise, uncertainty, topic) for $2.47; total API spend $5.26 of a $30 cap.
- Strict split: train 2019–2022, validation 2023 to mid-2024, test mid-2024 to 2025. The test period was
  evaluated once, on a git-tagged frozen configuration (`phase6-freeze`), after everything else was fixed.
- Volatility: pooled HAR on Garman-Klass variance; models compared out of sample with QLIKE and
  Diebold-Mariano tests. Mood: a hidden state filtered from the scores (Kalman and particle filters).

## Confirmatory results (test period, pre-registered, Holm-adjusted)

| Question | Comparison at 5 days | QLIKE change | 95% interval | p | Verdict |
|---|---|---|---|---|---|
| RQ3: do scores add to "an announcement happened"? | M vs B1 | −0.00007 | −0.00030 to +0.00016 | 0.55 | not confirmed |
| RQ4: does the mood model add to the scores? | M3 vs M | −0.00061 | −0.00289 to +0.00166 | 0.60 | not confirmed |

On validation both had looked significant (p = 0.0005 and p = 0.03), with gains of about 0.1% and 0.8% in QLIKE.
Neither held up. The validation gains were small enough to be noise among the variants that were tried.

## Exploratory results (not pre-registered)

- **Day-0 reaction (RQ1).** One unit more tone goes with a +1.2% abnormal return on train/validation (t = 6.6)
  and +1.8% on test (t = 3.4). Top tone third: +0.37% on the day; bottom third: −0.12% (train/validation).
- **No drift (RQ2).** Days +1 to +5 show nothing in any period, as an efficient market predicts.
- **LLM against baselines (RQ5).** For the day-0 return the LLM's tone explains more than a local sentiment
  model, my own word list or the Loughran-McDonald dictionary in every period (test: t = 3.6 against 1.9, 2.3
  and 2.9). For volatility no scorer helps.
- **Mood model.** The link from mood to volatility in the nonlinear model was estimated at zero; the particle
  filter worked (median effective sample size 3,503 of 5,000) but was not needed. A simple per-firm mood looked
  useful on validation (p = 0.002) and did not hold on test (p = 0.10).
- **GARCH-X** on ten firms: announcement and score columns made GARCH slightly worse.

## What to keep in mind

- **Gold set:** on 40 documents I labelled blind, the LLM's tone agrees only moderately with mine (rank
  correlation 0.43); it is far more conservative, by design of the prompt. Topic agreement is low (35%).
- **Hindsight in the LLM cannot be excluded.** Hiding names and years barely changed the scores, but the model
  still recognised the firm in 83% of the masked documents, so that check is inconclusive.
- **Survivorship:** today's index members only; three foreign filers and one firm without price history are out.
- **Feed gap:** March–June 2020 is missing from the official feed and is excluded.
- Full list: `docs/LIMITATIONS.md` (25 items).

## What I would do next

- Test the day-0 result on documents published after the model's training cut-off (the clean hindsight test).
- Use intraday prices: 22% of filings arrive during trading hours and daily data blur the reaction.
- A second labeller and a larger gold set, with the prompt's definitions in hand.

Reports: `reports/06_test_results.md` (test), `reports/04_vol_sentiment_validation.md`,
`reports/05_nonlinear_mood_validation.md`, `reports/03_event_study.md`, `reports/02_*.md`.
