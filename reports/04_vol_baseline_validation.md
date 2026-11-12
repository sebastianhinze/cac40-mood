# Volatility baselines on the validation period

36 firms, validation days 2023-01-02 to 2024-06-30; models re-estimated every 22 trading
days on an expanding window that starts in 2018. Target: Garman-Klass variance, averaged
over the next h trading days. Lower QLIKE is better. `qlike_minus_baseline` and the
Diebold-Mariano columns compare each model with HAR (negative = better than HAR).

| model | horizon | firm_days | qlike | mse_log | mz_slope | mz_r2 | qlike_minus_baseline | dm_statistic | dm_p_value |
|---|---|---|---|---|---|---|---|---|---|
| HAR | 1 | 13680 | 0.3418 | 0.7062 | 0.81 | 0.2167 |  |  |  |
| GARCH | 1 | 13680 | 0.3718 | 0.873 | 0.6882 | 0.154 | 0.03 | 4.7117 | 0.0 |
| HAR | 5 | 13536 | 0.1735 | 0.345 | 0.7639 | 0.3 |  |  |  |
| GARCH | 5 | 13536 | 0.198 | 0.4489 | 0.6487 | 0.2226 | 0.0245 | 4.7919 | 0.0 |
| HAR | 22 | 12924 | 0.0964 | 0.2218 | 0.7733 | 0.4148 |  |  |  |
| GARCH | 22 | 12924 | 0.1273 | 0.3147 | 0.7256 | 0.3244 | 0.0309 | 6.4011 | 0.0 |

Firm-days in the panel (2018 to end of validation): 59,868.
