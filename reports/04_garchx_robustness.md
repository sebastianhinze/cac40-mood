# GARCH-X robustness (validation period)

Ten firms with the most scored documents: TTE, SAN, DG, AI, ACA, SGO, STMPA, RNO, FGR, ENX. GARCH(1,1) on daily
returns with announcement columns added to the variance equation; forecasts rescaled
to the Garman-Klass target as for the plain GARCH baseline. Lower QLIKE is better;
negative `qlike_minus_baseline` = better than plain GARCH. Exploratory.

| model | horizon | qlike | mse_log | mz_slope | mz_r2 | qlike_minus_baseline | dm_statistic | dm_p_value | ci_low | ci_high |
|---|---|---|---|---|---|---|---|---|---|---|
| GARCH | 1 | 0.37323 | 0.92689 | 0.73845 | 0.21741 |  |  |  |  |  |
| GARCH-X event | 1 | 0.37547 | 0.93657 | 0.73524 | 0.21835 | 0.00224 | 2.23499 | 0.02542 | 0.00028 | 0.00421 |
| GARCH-X scores | 1 | 0.37751 | 0.94684 | 0.71642 | 0.21439 | 0.00428 | 1.36658 | 0.17176 | -0.00186 | 0.01043 |
| GARCH | 5 | 0.21213 | 0.5051 | 0.72381 | 0.31529 |  |  |  |  |  |
| GARCH-X event | 5 | 0.2149 | 0.51508 | 0.7207 | 0.31497 | 0.00277 | 3.1824 | 0.00146 | 0.00106 | 0.00448 |
| GARCH-X scores | 5 | 0.21376 | 0.5179 | 0.71835 | 0.31939 | 0.00163 | 0.58751 | 0.55686 | -0.00381 | 0.00707 |

`GARCH-X event`: x = an announcement arrived today. `GARCH-X scores`: x = that, plus
the day's |tone| and uncertainty.

## gamma per firm, scores variant, train period (in squared-percent units)

| firm_id | event | abs_tone | uncertainty |
|---|---|---|---|
| ACA | -0.296 | 0.088 | 0.526 |
| AI | -0.271 | 1.425 | 0.681 |
| DG | 0.228 | -0.482 | -0.377 |
| ENX | 0.006 | -3.543 | 6.079 |
| FGR | -0.144 | 1.075 | -0.125 |
| RNO | 0.232 | -1.672 | 5.292 |
| SAN | -0.031 | -0.364 | 0.268 |
| SGO | -0.483 | -1.664 | 2.713 |
| STMPA | 0.631 | -0.741 | -2.975 |
| TTE | -0.469 | -0.691 | 2.572 |
