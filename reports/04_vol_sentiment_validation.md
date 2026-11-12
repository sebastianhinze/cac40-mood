# Volatility: do scores and mood add to HAR? (validation period)

36 firms, 48,168 firm-days from 2019 to 2024-06-30 without the spring-2020 feed
hole; 4,382 of them have at least one scored announcement. Forecasts are
out-of-sample for 2023-01 to 2024-06, models re-estimated every 22 trading days.
Target: Garman-Klass variance averaged over the next h days. Lower QLIKE is better;
a negative `qlike_minus_baseline` means better than the baseline. Tests are
Diebold-Mariano on the daily cross-firm average loss difference (Newey-West).
**This is the validation period. The confirmatory test is the one-off test-period run.**

## Primary tests at h = 5 (pre-registered; Holm-adjusted)

| test | qlike_minus_baseline | ci_low | ci_high | dm_statistic | dm_p_value | holm_p_value |
|---|---|---|---|---|---|---|
| RQ3: M vs B1 | -0.00023 | -0.00035 | -0.0001 | -3.48604 | 0.00049 | 0.00098 |
| RQ4: M2 vs M | -0.00034 | -0.00096 | 0.00027 | -1.08824 | 0.27649 | 0.27649 |

## Every model against B1 (HAR + announcement dummies)

| model | horizon | qlike | mse_log | mz_slope | mz_r2 | qlike_minus_baseline | dm_statistic | dm_p_value | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 1 | 0.33895 | 0.70321 | 0.84868 | 0.21932 | -0.00014 | -2.23832 | 0.0252 | -0.00026 | -2e-05 |
| B1 | 1 | 0.33909 | 0.70332 | 0.84867 | 0.21937 |  |  |  |  |  |
| M | 1 | 0.33908 | 0.70309 | 0.84852 | 0.2193 | -0.0 | -0.03032 | 0.97581 | -0.00012 | 0.00012 |
| M2 | 1 | 0.33871 | 0.70151 | 0.85142 | 0.21972 | -0.00038 | -2.29612 | 0.02167 | -0.0007 | -6e-05 |
| M+extra | 1 | 0.33907 | 0.70313 | 0.84868 | 0.21923 | -1e-05 | -0.19283 | 0.84709 | -0.00014 | 0.00012 |
| B1+lexicon | 1 | 0.33904 | 0.70344 | 0.84855 | 0.21933 | -4e-05 | -0.96372 | 0.33519 | -0.00013 | 5e-05 |
| B1+local | 1 | 0.3391 | 0.7033 | 0.84845 | 0.21946 | 1e-05 | 0.48586 | 0.62707 | -5e-05 | 8e-05 |
| B1+LM | 1 | 0.33905 | 0.70322 | 0.84849 | 0.21936 | -3e-05 | -0.51567 | 0.60609 | -0.00016 | 9e-05 |
| B0 | 5 | 0.16991 | 0.33771 | 0.81111 | 0.30805 | 0.00027 | 1.84 | 0.06577 | -2e-05 | 0.00055 |
| B1 | 5 | 0.16964 | 0.33739 | 0.81194 | 0.30856 |  |  |  |  |  |
| M | 5 | 0.16941 | 0.3369 | 0.81294 | 0.3091 | -0.00023 | -3.48604 | 0.00049 | -0.00035 | -0.0001 |
| M2 | 5 | 0.16907 | 0.33679 | 0.81376 | 0.31026 | -0.00057 | -1.78793 | 0.07379 | -0.00119 | 5e-05 |
| M+extra | 5 | 0.16936 | 0.33687 | 0.81306 | 0.30918 | -0.00027 | -3.54764 | 0.00039 | -0.00043 | -0.00012 |
| B1+lexicon | 5 | 0.16964 | 0.3374 | 0.81186 | 0.30854 | 1e-05 | 0.68589 | 0.49278 | -1e-05 | 2e-05 |
| B1+local | 5 | 0.16963 | 0.33737 | 0.81208 | 0.30856 | -1e-05 | -0.98229 | 0.32596 | -4e-05 | 1e-05 |
| B1+LM | 5 | 0.16964 | 0.33739 | 0.81187 | 0.30853 | 0.0 | 0.43619 | 0.6627 | -1e-05 | 2e-05 |
| B0 | 22 | 0.0984 | 0.22947 | 0.81221 | 0.42962 | 0.00035 | 1.44506 | 0.14844 | -0.00012 | 0.00082 |
| B1 | 22 | 0.09806 | 0.2284 | 0.81446 | 0.43227 |  |  |  |  |  |
| M | 22 | 0.09795 | 0.22809 | 0.81525 | 0.43298 | -0.00011 | -2.39224 | 0.01675 | -0.00019 | -2e-05 |
| M2 | 22 | 0.09941 | 0.23379 | 0.80398 | 0.43034 | 0.00135 | 0.74861 | 0.45409 | -0.00219 | 0.0049 |
| M+extra | 22 | 0.09795 | 0.22811 | 0.81509 | 0.4331 | -0.00011 | -1.89815 | 0.05768 | -0.00022 | 0.0 |
| B1+lexicon | 22 | 0.09806 | 0.2284 | 0.81426 | 0.43228 | 0.0 | 0.2218 | 0.82447 | -2e-05 | 3e-05 |
| B1+local | 22 | 0.09806 | 0.22842 | 0.81448 | 0.43229 | 0.0 | 0.41105 | 0.68103 | -2e-05 | 2e-05 |
| B1+LM | 22 | 0.09806 | 0.22841 | 0.81441 | 0.43243 | -0.0 | -0.04761 | 0.96203 | -3e-05 | 3e-05 |

B0 = HAR only. M = B1 + |tone| + uncertainty. M2 = M + filtered mood. `M+extra`,
`B1+lexicon`, `B1+local` and `B1+LM` (Loughran-McDonald on official English versions)
are exploratory.

## Mood against raw scores: M2 against M

| model | horizon | qlike | mse_log | mz_slope | mz_r2 | qlike_minus_baseline | dm_statistic | dm_p_value | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| M | 1 | 0.33908 | 0.70309 | 0.84852 | 0.2193 |  |  |  |  |  |
| M2 | 1 | 0.33871 | 0.70151 | 0.85142 | 0.21972 | -0.00037 | -2.60698 | 0.00913 | -0.00066 | -9e-05 |
| M | 5 | 0.16941 | 0.3369 | 0.81294 | 0.3091 |  |  |  |  |  |
| M2 | 5 | 0.16907 | 0.33679 | 0.81376 | 0.31026 | -0.00034 | -1.08824 | 0.27649 | -0.00096 | 0.00027 |
| M | 22 | 0.09795 | 0.22809 | 0.81525 | 0.43298 |  |  |  |  |  |
| M2 | 22 | 0.09941 | 0.23379 | 0.80398 | 0.43034 | 0.00146 | 0.79918 | 0.42419 | -0.00212 | 0.00505 |

## Robustness at h = 5: sample starts 2020-07-01

| model | horizon | qlike | mse_log | mz_slope | mz_r2 | qlike_minus_baseline | dm_statistic | dm_p_value | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 5 | 0.17221 | 0.35106 | 0.8177 | 0.30855 |  |  |  |  |  |
| M | 5 | 0.17192 | 0.35042 | 0.81877 | 0.30924 | -0.00029 | -3.71988 | 0.0002 | -0.00045 | -0.00014 |
| M2 | 5 | 0.17146 | 0.34995 | 0.81903 | 0.31123 | -0.00075 | -4.11006 | 4e-05 | -0.00111 | -0.00039 |

## Slopes of model M2 at h = 5, train period (description only)

| column | slope |
|---|---|
| log_rv_d | 0.1346 |
| log_rv_w | 0.2383 |
| log_rv_m | 0.3437 |
| ev_news | -0.0394 |
| ev_report | -0.0773 |
| ev_quarterly | -0.0958 |
| ev_other | -0.202 |
| abs_tone | -0.1817 |
| uncertainty | 0.1326 |
| mood_total | -0.2228 |
| mood_market | 0.4099 |

## Documents behind the features

| rule | documents_removed |
|---|---|
| administrative notice | 5712 |
| firm outside the main analysis | 120 |
| no event session | 0 |
| inside the spring-2020 feed hole | 28 |
| no valid LLM score (no usable text) | 166 |
| firm-size figure from a later filing | 154 |
| documents used | 5239 |
