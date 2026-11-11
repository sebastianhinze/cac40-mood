# The mood model in plain language

Status: implemented and validated on **simulated data only** (phase 5). It has not been run on real scores yet.
Code: `src/cac40_mood/mood/`. Tests: `tests/test_mood.py`.

## 1. What problem it solves

Each scored announcement gives one noisy reading of how things are going for a firm. Readings arrive irregularly:
nothing for two weeks, then three documents on a results day. A raw score also jumps around from one document to
the next. We want one smooth number per firm and trading day, the **mood**, together with an honest statement of
how sure we are about it. The mood on day *t* may only use documents published up to day *t*.

## 2. The model

**The hidden mood moves slowly.** Tomorrow's mood is today's mood, pulled a little towards zero, plus a small
random shock:

    mood_t = phi × mood_{t-1} + shock_t          shock_t ~ Normal(0, sigma_state²)

- `phi` (between 0 and 1) is the persistence. With `phi = 0.97` half of a mood shift is gone after about 23
  trading days.
- Left alone, the mood wanders around zero with variance `sigma_state² / (1 − phi²)`: the long-run variance.

**A document is a noisy reading of the mood.**

    score_k = mood_t + error_k          scale of error_k = sigma_obs / sqrt(confidence_k)

- A score the model was confident about (confidence 1) carries the base noise `sigma_obs`. A score with
  confidence 0.25 carries twice that noise, so it moves the mood less.
- Several documents on one day are several readings of the same mood.
- A day without a document has no reading. That is not "neutral news"; it is no information.

**Two error shapes.** With normal errors the model is linear and Gaussian and has an exact solution, the Kalman
filter. With Student-t errors (heavy tails) one wild score is treated as probably an outlier and moves the mood
much less. That robust version has no exact solution, which is the reason for the particle filter.

## 3. Worked example: three days, by hand (Kalman filter)

Parameters: `phi = 0.9`, `sigma_state = 0.2`, `sigma_obs = 0.3`.
Long-run variance = 0.2² / (1 − 0.9²) = 0.04 / 0.19 = **0.2105**. Before any document we believe: mean 0,
variance 0.2105.

Each day has up to two steps.

- **Predict** (time passes): mean → phi × mean, variance → phi² × variance + sigma_state².
- **Update** (a reading y with error variance R arrives):
  gain = variance / (variance + R); mean → mean + gain × (y − mean); variance → (1 − gain) × variance.

| Day | What happens | Calculation | Mean | Variance | ± 1 sd |
|---|---|---|---|---|---|
| 0 | Reading 0.5, confidence 1 (R = 0.09) | gain = 0.2105 / 0.3005 = 0.70; mean = 0 + 0.70 × 0.5 | **0.350** | 0.063 | 0.25 |
| 1 | No document | mean = 0.9 × 0.350; variance = 0.81 × 0.063 + 0.04 | **0.315** | 0.091 | 0.30 |
| 2 | Reading −0.1, confidence 0.25 (R = 0.09 / 0.25 = 0.36) | predict: mean 0.284, variance 0.114; gain = 0.114 / 0.474 = 0.24; mean = 0.284 + 0.24 × (−0.1 − 0.284) | **0.192** | 0.086 | 0.29 |

What to notice:

- Day 0: the first reading is trusted 70%, because we knew little before it.
- Day 1: without news the mood drifts towards zero and **the uncertainty grows** (0.063 → 0.091).
- Day 2: the negative reading has low confidence, so it gets a weight of only 24% and the mood stays positive.

`test_kalman_filter_matches_the_worked_example_in_the_docs` checks exactly these numbers.

## 4. The particle filter: the same thing with a cloud of guesses

Instead of tracking a mean and a variance, keep N guesses ("particles") of today's mood.

1. **Move.** Each particle takes one step of the mood equation with its own random shock.
2. **Weigh.** When a reading arrives, each particle's weight is multiplied by how likely that reading would be if
   the particle were the true mood. Particles close to the reading gain weight.
3. **Resample when needed.** If a few particles hold most of the weight, draw a new cloud in proportion to the
   weights and reset the weights. "Needed" is measured by the effective sample size, 1 / Σ weight²; we resample
   when it falls below N / 2.

The weighted average of the cloud is the filtered mood; the 5% and 95% weighted quantiles are the band.

**Day 0 of the example with five particles** (seed 42, numbers from the code):

| | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| Start: draws from the long-run distribution | 0.140 | −0.477 | 0.344 | 0.432 | −0.895 |
| Weight after the reading 0.5 | 0.208 | 0.002 | 0.374 | 0.416 | 0.000 |

- Weighted mean = **0.336** (the exact answer is 0.350). Particles far from the reading have almost no weight.
- Effective sample size = 2.81 of 5. That is just above the threshold of 2.5, so the code would not resample here.
  Had it done so, the new cloud would be P1, P3, P4, P4, P4: the two near-weightless particles disappear.

Five particles are only for illustration. With the same seed the three daily means are 0.336, 0.224, 0.286 with
5 particles, and 0.348, 0.314, 0.191 with 10,000, against the exact 0.350, 0.315, 0.192.

On a day **without** a document only step 1 happens. Every particle receives a fresh shock, the cloud spreads
out, and the band widens: the same "uncertainty grows" as in the Kalman table.

## 5. The spring-2020 hole in the filings feed

From 5 March to 15 June 2020 the official feed lost about 95% of its documents. For the mood this window is
**missing data, not quiet news**: any reading inside it is discarded, the filter only predicts, and the band
widens day after day towards the long-run spread. When documents return, the first reading pulls the mood back
and the band narrows again. `test_missing_data_gap_widens_the_band_until_readings_return` checks this.

## 6. Market mood and firm mood

A firm's mood is modelled as a market-wide part plus its own part: `mood_i = market + firm_i`, both moving slowly.
They are filtered in two stages (DECISIONS #44):

1. **Market.** Every reading of every firm is a reading of the market mood. Seen from the market, a firm's own
   mood is extra noise, so each reading's error variance is increased by the long-run variance of the firm part.
2. **Firm.** For each firm, subtract the filtered market mood from its readings and filter the remainder. Because
   the market mood is itself uncertain, its variance on that day is added to the reading's error variance.

Total mood = market mean + firm mean; the two variances add. The alternative, one filter over all 37 states, was
not chosen: the two-stage version reuses the scalar filter that is validated against the Kalman filter and costs
36 small runs. Its weakness is that stage 1 ignores that a firm's deviation persists, so the market mood is
somewhat less sharp than a joint filter would make it.

## 7. Estimating the parameters

`phi`, `sigma_state` and `sigma_obs` are chosen to make the observed readings as likely as possible (maximum
likelihood), on the **train period only**.

- With normal errors the Kalman filter gives the likelihood exactly.
- With Student-t errors the particle filter estimates it: the average weight at each reading is that reading's
  likelihood. The estimate is noisy, so every parameter value is tried with the **same random seed**; the noise
  then largely cancels when two values are compared.

## 8. What the validation on simulated data showed

All numbers come from simulations where the true mood is known (`tests/test_mood.py`).

| Check | Result |
|---|---|
| Particle filter vs. Kalman filter, normal errors, 1,500 days, 5,000 particles | log-likelihood −144.12 vs. −144.11; largest gap between the two daily means 0.017 |
| Log-likelihood across 10 seeds, 2,000 particles | standard deviation 0.24 |
| Tracking error (RMSE) against the true mood | 0.258 filtered vs. 0.356 for "always zero" |
| Parameter recovery, exact likelihood, 6 series × 2,000 days | phi 0.973 (true 0.97), sigma_state 0.078 (0.08), sigma_obs 0.295 (0.30) |
| Readings with 8% outliers | Student-t filter has a clearly lower tracking error than the normal one |
| Market + 12 firms, two-stage filter | market mood correlates 0.76 with the truth; errors divided by the stated standard deviation have spread 1.01 |

## 9. Limits

- The model assumes documents arrive at times unrelated to the mood. If firms publish more when things go badly,
  the timing itself carries information that this model ignores.
- One persistence and one noise level are shared by all firms.
- The two-stage market/firm filter is an approximation (section 6).
- Forecast features will use the **filtered** mood only (information up to day t). Smoothed paths, which also use
  later documents, are for charts and will be labelled as such.
