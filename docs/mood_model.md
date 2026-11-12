# The mood model in plain language

Status: validated on simulated data (sections 1–9) and run on the real scores of the train and validation periods (section 10). The test period has not been touched.
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

## 10. Real data: the plan and what came out (train and validation only)

Code: `src/cac40_mood/mood/real.py`. Run: `uv run python -m cac40_mood.mood.real`.

**Plan, in the order it is done**

1. **Readings.** One per scored document of the 36 firms (the document set of DECISIONS #53). Value = tone minus
   the average train-period tone (0.126); confidence from the LLM; day = the document's event session.
   Why centre: the mood drifts to zero without news, but the average document is mildly positive. Without
   centring, silence would read as "worse than usual".
2. **Missing days.** The 5 Mar – 15 Jun 2020 sessions are flagged missing: the filter only predicts there.
3. **Parameters, train period only.** Exact Kalman likelihood with normal errors. Market and firm parameters
   depend on each other, so they are fitted in turn, twice: fit the market (a firm's own mood counted as extra
   noise), filter it, subtract it, fit the firm part on what is left (one parameter set for all firms).
4. **Filter train and validation** with the two-stage particle filter (2,000 particles, fixed seed) and the
   train parameters. The filter is causal: day t uses readings up to day t.
5. **Features.** `mood_total` (market + firm part) and `mood_market`, added to the volatility model M to give M2.
6. **Not done yet:** the Student-t variant on real data; the smoothed path for the dashboard; anything on the
   test period (phase 6).

**What came out**

| | phi | half-life (days) | sigma_state | long-run sd | sigma_obs |
|---|---|---|---|---|---|
| Market mood | 0.926 | 9 | 0.044 | 0.12 | 0.19 |
| Firm part | 0.993 | 96 | 0.014 | 0.12 | 0.18 |

- 3,771 train readings for estimation; 5,239 readings filtered (train and validation); about one minute.
- The market mood is short-lived (half of a move is gone in 9 trading days); a firm's own mood is very
  persistent (about 96 days). Both have the same long-run spread, 0.12 on the tone scale.
- A single reading is noisy next to that: its error (0.18–0.19 at full confidence) is larger than the whole
  long-run spread of the mood, so one document moves the mood only a little. The typical stated uncertainty of a
  firm's mood is 0.11.
- The lowest market mood is in the third quarter of 2020 (average −0.17), the first full quarter after the feed
  hole; from 2021 on it stays within about ±0.04 per quarter.
- **Forecasting value (validation):** adding the two mood columns to model M improves QLIKE at 1 day
  (p = 0.009) but not at the pre-registered 5 days (p = 0.28) and makes 22 days worse (not significant).
  Details: `reports/04_vol_sentiment_validation.md`.

## 11. The nonlinear model: mood feeds into volatility

Code: `src/cac40_mood/mood/sv.py`, `sv_run.py`. Tests: `tests/test_sv.py`. Pre-registration: DECISIONS #58.

**Why another model.** In sections 1–10 the mood is read only through the scores, and every equation is linear
with normal errors, so the Kalman filter already gives the exact answer; the particle filter adds nothing there
except a check. Here the mood also drives something we observe in a nonlinear way: returns.

**The model, per firm.** Two hidden numbers per day, the mood `m` and the log-variance `h`:

    m_t = phi_m × m_{t-1} + shock                                  (as before)
    h_t = mu + phi_h × (h_{t-1} − mu) + beta × m_{t-1} + shock     (new)

and two kinds of data:

    a scored document:  score = m_t + error                        (as before)
    the day's return:   r_t ~ Normal(0, exp(h_t))                  (new)

- `h` is the logarithm of the day's return variance. It is persistent (`phi_h` near 1) and wanders around the
  firm's own level `mu`.
- `beta` is the link. Negative `beta`: a bad mood yesterday pushes today's log-variance up.
- A return of −3% is likely if `exp(h)` is large and very unlikely if it is small. That `exp()` is what breaks
  the Kalman filter: the data are no longer a straight-line function of the hidden numbers plus normal noise.

**The particle filter here.** Each particle is a pair (m, h). Every day:

1. **Move** each pair one step with its own random shocks (h uses the particle's mood of the day before).
2. **Weigh** it by the day's documents (is its mood close to the scores?) and by the day's return (is a return
   of this size plausible with its variance?). A day with a 4% move kills the particles that believed in calm.
3. **Resample** when the weights have become too uneven (effective sample size below half the particles).

Because returns arrive every day, the cloud is weighed every day, not only on document days. The information
flows both ways: documents inform the mood, and if `beta` is not zero, unusual volatility also says something
about the mood.

**Forecast.** For each particle the expected variance j days ahead has a closed form (the two equations are
linear in m and h; only the return is not), so the forecast for the next 5 days is the weighted average over
particles of that formula. No extra simulation is needed.

**Worked example of one weighing step.** Three particles, all with weight 1/3, and a return of −3% (r² = 0.0009):

| Particle | h (log-variance) | Daily volatility exp(h/2) | log weight change = −½ (h + r² / exp(h)) | New weight |
|---|---|---|---|---|
| A: calm | −9.2 | 1.0% | −½ (−9.2 + 0.0009 / 0.000101) = 0.15 | 0.03 |
| B: normal | −8.1 | 1.7% | −½ (−8.1 + 0.0009 / 0.000304) = 2.57 | 0.38 |
| C: nervous | −7.0 | 3.0% | −½ (−7.0 + 0.0009 / 0.000912) = 3.01 | 0.59 |

After a −3% day the "calm" particle keeps 3% of the weight. Effective sample size = 1 / (0.03² + 0.38² + 0.59²)
= 2.0 of 3.

**How it is estimated** (train period only): the mood parameters by the exact Kalman likelihood on the scores,
then `phi_h`, `sigma_h` and `beta` by the particle filter's likelihood, started from three values of `beta` and
always with the same random numbers so that two parameter values can be compared. A twin model with `beta = 0`
is fitted the same way: it is the fair comparison ("is it the mood, or would any volatility model of this kind
help?").

**What came out (train fit, validation test)**

| | phi | sigma (daily shock) | beta |
|---|---|---|---|
| Mood (one per firm) | 0.954 (half-life 15 days) | 0.063 | |
| Log-variance, with mood | 0.955 | 0.263 | **0.0004** |
| Log-variance, beta fixed at 0 | 0.945 | 0.288 | 0 |

- **The link is zero.** The fitted `beta` is 0.0004. Letting it differ from zero raises the train
  log-likelihood by 4.7 under the estimation seed and *lowers* it under three other seeds (−0.7, −1.8, −14.6):
  no evidence that mood enters the volatility equation.
- **Filter health** (5,000 particles per firm, 50,688 firm-days): median effective sample size 3,503; 5th
  percentile 2,037; below 100 on 100 firm-days (days with an extreme return); resampled on 14% of firm-days.
- **Pre-registered test (h = 5, QLIKE):** M3 beats M by 0.0013 (0.16941 → 0.16810, p = 0.03).
- **But not because of the nonlinear part.** The model's volatility forecast alone adds nothing to M
  (p = 0.93). Its filtered mood alone gives the whole gain (p = 0.002), and with `beta` at zero that mood is a
  plain linear filter of the scores: a per-firm mood with a 15-day half-life. Higher mood, lower volatility
  over the next five days.
- So on this data the particle filter was not needed for forecasting; what helps is a simpler mood than the
  two-stage market/firm one of section 6. That simpler mood was not a pre-planned variant, which is why it is
  reported as exploratory. Details: `reports/05_nonlinear_mood_validation.md`.

