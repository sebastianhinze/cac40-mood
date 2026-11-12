# Nonlinear mood model (mood feeds into log-volatility): validation result

Pre-registered in DECISIONS #58 before any of this was run. Model per firm: hidden mood
(read through the tone scores) and hidden log-variance h, with
h_t = mu + phi_h (h_{t-1} - mu) + beta * mood_{t-1} + shock, and
return ~ Normal(0, exp(h)).
Fitted on the train period with the particle filter; validation is 2023-01 to 2024-06.

## Primary test: M3 against M at h = 5 (QLIKE, Diebold-Mariano)

| model | against | qlike_baseline | qlike_model | qlike_minus_baseline | dm_statistic | dm_p_value | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|
| M3 | M | 0.16941 | 0.1681 | -0.00131 | -2.1634 | 0.03051 | -0.00249 | -0.00012 |

## All comparisons (the others are exploratory)

`M+SV0` = M plus the forecast of the same model with beta = 0 (no mood). `M2` = M plus
the linear filtered mood. Negative `qlike_minus_baseline` = M3 is better.

| horizon | model | against | qlike_baseline | qlike_model | qlike_minus_baseline | dm_statistic | dm_p_value | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|---|
| 1 | M3 | M | 0.33908 | 0.33809 | -0.001 | -1.07121 | 0.28407 | -0.00283 | 0.00083 |
| 1 | M3 | M+extra | 0.33907 | 0.33809 | -0.00099 | -1.06228 | 0.28811 | -0.00281 | 0.00084 |
| 1 | M3 | M+SV0 | 0.33915 | 0.33809 | -0.00106 | -3.71544 | 0.0002 | -0.00162 | -0.0005 |
| 1 | M3 | M2 | 0.33871 | 0.33809 | -0.00062 | -0.65278 | 0.5139 | -0.0025 | 0.00125 |
| 5 | M3 | M | 0.16941 | 0.1681 | -0.00131 | -2.1634 | 0.03051 | -0.00249 | -0.00012 |
| 5 | M3 | M+extra | 0.16936 | 0.1681 | -0.00126 | -2.08033 | 0.0375 | -0.00245 | -7e-05 |
| 5 | M3 | M+SV0 | 0.16935 | 0.1681 | -0.00125 | -3.14641 | 0.00165 | -0.00202 | -0.00047 |
| 5 | M3 | M2 | 0.16907 | 0.1681 | -0.00097 | -1.21693 | 0.22363 | -0.00252 | 0.00059 |
| 22 | M3 | M | 0.09795 | 0.09764 | -0.00031 | -0.85184 | 0.3943 | -0.00101 | 0.0004 |
| 22 | M3 | M+extra | 0.09795 | 0.09764 | -0.00031 | -0.84782 | 0.39654 | -0.00102 | 0.0004 |
| 22 | M3 | M+SV0 | 0.09797 | 0.09764 | -0.00032 | -1.81509 | 0.06951 | -0.00067 | 3e-05 |
| 22 | M3 | M2 | 0.09941 | 0.09764 | -0.00177 | -0.91917 | 0.35801 | -0.00554 | 0.002 |

## Which part of M3 helps? (h = 5, each against M; added after seeing the result)

`M+SVforecast` = M plus only the nonlinear model's variance forecast. `M+SVmood` = M plus
only its filtered mood. With beta at zero (below), that mood is in effect a linear
per-firm mood filter: returns tell it nothing.

| horizon | model | against | qlike_baseline | qlike_model | qlike_minus_baseline | dm_statistic | dm_p_value | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|---|
| 5 | M+SV0 | M | 0.16941 | 0.16935 | -6e-05 | -0.08749 | 0.93028 | -0.00148 | 0.00135 |
| 5 | M+SVforecast | M | 0.16941 | 0.16935 | -6e-05 | -0.09043 | 0.92794 | -0.00145 | 0.00132 |
| 5 | M+SVmood | M | 0.16941 | 0.16816 | -0.00126 | -3.13293 | 0.00173 | -0.00204 | -0.00047 |
| 5 | M3 | M | 0.16941 | 0.1681 | -0.00131 | -2.1634 | 0.03051 | -0.00249 | -0.00012 |
| 5 | M2 | M | 0.16941 | 0.16907 | -0.00034 | -1.08824 | 0.27649 | -0.00096 | 0.00027 |

## Fitted parameters (train period)

| | phi_h | sigma_h | beta |
|---|---|---|---|
| with mood | 0.9553 | 0.2632 | 0.0004 |
| without mood | 0.9449 | 0.2885 | 0 |

Mood part: phi 0.9535, sigma_state 0.0628, sigma_obs 0.1791.

Train log-likelihood gain from letting beta differ from 0: 4.7 (same seed); under three other seeds: -0.7, -1.8, -14.6. A gain of about 2 would be the usual threshold for one extra parameter if the likelihood were exact; here it is estimated, so read it with that noise in mind.

## Health of the particle filter (train and validation, final run)

- 5,000 particles per firm.
- Effective sample size: median 3,503, 5th percentile 2,037, minimum 2.
- Firm-days with an effective sample size below 100: 100 of 50,688 (days with an extreme return; the cloud is resampled right after).
- Resampled on 13.8% of firm-days.

## Slopes of M3 at h = 5, train period (description only)

| column | slope |
|---|---|
| log_rv_d | 0.1261 |
| log_rv_w | 0.183 |
| log_rv_m | 0.2967 |
| ev_news | -0.0357 |
| ev_report | -0.0748 |
| ev_quarterly | -0.073 |
| ev_other | -0.1957 |
| abs_tone | -0.1317 |
| uncertainty | 0.0746 |
| sv_log_forecast_5 | 0.1631 |
| sv_mood | -0.1464 |
