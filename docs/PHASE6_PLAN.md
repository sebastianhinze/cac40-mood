# Phase 6: what will run on the test period (proposed freeze, waiting for my OK)

Nothing below has been run. The test period is 2024-07-01 to 2025-12-31. It is evaluated once.

## Before the run

1. I approve this plan and the one open choice (section "Open choice").
2. The runner is written. It only passes a later end date to the existing functions (price panel, document
   selection, trading calendar, return panel); no model logic changes. It is tested on made-up data.
3. The commit is tagged `phase6-freeze`; the tag's hash goes into the results file.
4. The run. Results are written down as they come out. Nothing is changed afterwards.

## Inputs (all already on disk)

- **Prices:** the Yahoo snapshot from phase 1 (through 2025-12-31). Target: Garman-Klass variance.
- **Scores:** the bulk run of 2026-11-12 (prompt v2, Haiku, first 12,000 tokens). Test documents were scored
  then; none has been looked at. No new LLM call is made.
- **Documents:** the same selection rule as on train/validation (36 firms, non-admin, valid score, firm-size
  figure from an earlier filing). Roughly 1,500 test documents before the rule.
- **Train-only numbers, fixed:** tone centre 0.126; mood parameters; volatility-equation parameters; 5,000
  particles, seed 58 (`src/cac40_mood/phase6_config.py`).

## The two primary tests (the only confirmatory results)

| | Model | Against | Horizon | Loss | Test |
|---|---|---|---|---|---|
| RQ3 | **M** = HAR + announcement dummies + \|tone\| + uncertainty | **B1** = HAR + announcement dummies | 5 days | QLIKE | Diebold-Mariano on the daily cross-firm mean loss difference, Newey-West ≥ 4 lags, two-sided |
| RQ4 | **M3** = M + nonlinear model's 5-day log forecast + its filtered mood | **M** | 5 days | QLIKE | same |

- Holm adjustment across the two p-values; a result counts at adjusted p < 0.05 **and** a negative QLIKE
  difference (the model must be better, not just different).
- Forecast origins: every test trading day up to 5 days before the last session. Models are re-estimated every
  22 trading days on all firm-days whose target was already observed (2019 onwards, feed hole excluded), exactly
  as on validation. The mood filter simply continues from validation into the test period.
- Validation results for reference: RQ3 −0.00023 (p = 0.0005); RQ4 with M3 −0.00131 (p = 0.03).

## Reported alongside, labelled exploratory (Benjamini-Hochberg)

- The same two comparisons at 1 and 22 days; B0 (HAR only); M2 (linear two-stage mood); M + per-firm mood alone;
  the three baseline scorers in place of the LLM (RQ5).
- Event study on test-period events: CAR[0], CAR[+1,+5], CAR[−5,−1] on tone alone and on scores with controls;
  CAR paths by tone third.
- Mincer-Zarnowitz fits and the MSE of log variance for every model.

## Not run

- GARCH and GARCH-X (robustness only; they did not help on validation).
- Any new prompt, model, feature or parameter search.
- Any trading strategy.

## Open choice: which mood model carries RQ4?

On validation the nonlinear link itself turned out to be zero (beta = 0.0004), and the volatility forecast of
that model adds nothing. All of M3's gain comes from its filtered mood, which with beta = 0 is a plain per-firm
linear mood filter.

1. **M3 as pre-registered (recommended).** It is the model whose test was fixed in advance (DECISIONS #58) and
   it passed on validation. Its extra forecast column is dead weight, but dropping it now would be a choice made
   after seeing the result.
2. **M2**, the two-stage market/firm mood that RQ4 was first written for. It did not beat M on validation at
   5 days (p = 0.28).
3. **M + per-firm mood only.** The simplest and the strongest on validation (p = 0.002), but found after the
   fact: it is the third mood variant looked at.

Also to confirm: HAR slopes are re-estimated on an expanding window through the test period (as on validation).
The alternative is to freeze them at the end of validation; CLAUDE.md's "fitted on train only" is applied to
the mood model, the tone centre and the scorer, not to these rolling refits.
