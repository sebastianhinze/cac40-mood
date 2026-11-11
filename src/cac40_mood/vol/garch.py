"""GARCH(1,1) as a second baseline, on daily close-to-close returns.

    return_t   = mu + e_t,   e_t ~ N(0, s2_t)
    s2_{t+1}   = omega + alpha * e_t^2 + beta * s2_t

Tomorrow's variance is a weighted mix of a constant, today's squared surprise and today's
variance. The parameters are estimated with the `arch` package; the forecasts are then computed
here with the explicit recursion, so every number can be traced by hand.

Two details that matter for a fair comparison with HAR:
- the h-day forecast is the average of the next h daily variances. For GARCH(1,1) the expected
  variance k days ahead moves geometrically from tomorrow's variance towards the long-run level:
  E[s2_{t+k}] = long_run + (alpha+beta)^(k-1) * (s2_{t+1} - long_run);
- GARCH models close-to-close returns, while our target is the range-based variance of the
  trading session, which is smaller on average. Each forecast is therefore rescaled by the
  firm's ratio (mean target variance / mean GARCH variance) on the estimation sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

from cac40_mood.vol.har import REFIT_EVERY

PERCENT: float = 100.0  # the optimiser is better behaved on percentage returns


@dataclass(frozen=True)
class GarchParameters:
    mu: float
    omega: float
    alpha: float
    beta: float


def fit_garch(returns_percent: np.ndarray) -> GarchParameters:
    """Maximum-likelihood GARCH(1,1) with normal errors on percentage returns."""
    model = arch_model(returns_percent, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)
    result = model.fit(disp="off", show_warning=False)
    parameters = result.params
    return GarchParameters(
        float(parameters["mu"]),
        float(parameters["omega"]),
        float(parameters["alpha[1]"]),
        float(parameters["beta[1]"]),
    )


def next_day_variances(returns_percent: np.ndarray, parameters: GarchParameters) -> np.ndarray:
    """s2_{t+1} for every day t, by the recursion (started at the sample variance)."""
    surprises_squared = (returns_percent - parameters.mu) ** 2
    variances = np.empty(len(returns_percent))
    current = surprises_squared.mean()
    for day, surprise_squared in enumerate(surprises_squared):
        current = parameters.omega + parameters.alpha * surprise_squared + parameters.beta * current
        variances[day] = current
    return variances


def average_variance_ahead(
    next_day_variance: np.ndarray, parameters: GarchParameters, horizon: int
) -> np.ndarray:
    """Average expected variance over the next `horizon` days, given tomorrow's variance."""
    persistence = parameters.alpha + parameters.beta
    if persistence >= 1.0:  # no finite long-run level: the forecast stays at tomorrow's variance
        return next_day_variance.copy()
    long_run = parameters.omega / (1.0 - persistence)
    weights = persistence ** np.arange(horizon)  # (alpha+beta)^(k-1) for k = 1..h
    return long_run + weights.mean() * (next_day_variance - long_run)


def firm_garch_forecasts(
    firm: pd.DataFrame, horizon: int, origins: np.ndarray, refit_every: int = REFIT_EVERY
) -> pd.DataFrame:
    """Out-of-sample GARCH forecasts for one firm (rows: date, ret, rv), rescaled to the target.

    Like the HAR baseline, parameters are re-estimated every `refit_every` trading days on all
    data up to the re-estimation day and then held fixed until the next one.
    """
    firm = firm.dropna(subset=["ret"]).sort_values("date").reset_index(drop=True)
    returns = firm["ret"].to_numpy() * PERCENT
    dates = firm["date"].to_numpy()
    pieces = []
    for start in range(0, len(origins), refit_every):
        block = origins[start : start + refit_every]
        in_sample = dates <= block[0]
        parameters = fit_garch(returns[in_sample])
        ahead = average_variance_ahead(next_day_variances(returns, parameters), parameters, horizon)
        ahead = ahead / PERCENT**2  # back to squared decimal returns
        # Rescale from close-to-close variance to the level of the range-based target.
        scale = np.nanmean(firm["rv"].to_numpy()[in_sample]) / ahead[in_sample].mean()
        in_block = np.isin(dates, block)
        pieces.append(pd.DataFrame({"date": dates[in_block], "forecast": ahead[in_block] * scale}))
    return pd.concat(pieces, ignore_index=True)


def garch_forecasts(panel: pd.DataFrame, har_forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """GARCH forecasts on exactly the firm-days of `har_forecasts` (same origins, same targets)."""
    origins = np.sort(har_forecasts["date"].unique())
    pieces = []
    for firm_id, firm in panel.groupby("firm_id"):
        forecasts = firm_garch_forecasts(firm, horizon, origins).assign(firm_id=firm_id)
        pieces.append(forecasts)
    forecasts = pd.concat(pieces, ignore_index=True)
    targets = har_forecasts[["firm_id", "date", "horizon", "actual", "log_actual"]]
    merged = targets.merge(forecasts, on=["firm_id", "date"])
    return merged.assign(log_forecast=np.log(merged["forecast"]))
