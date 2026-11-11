"""HAR baseline: tomorrow's volatility from the last day, week and month (Corsi 2009).

The model, in logs, for firm i and forecast horizon h (in trading days):

    log(average variance over days t+1..t+h)
        = firm effect_i + b_d*log(variance of day t)
                        + b_w*log(average variance of the last 5 days)
                        + b_m*log(average variance of the last 22 days) + error

One set of slopes for all firms ("pooled"), one intercept per firm ("fixed effects").
Extra columns (announcement dummies, sentiment scores) can be added to the right-hand side;
that is how the later phases test whether sentiment adds anything beyond this baseline.

No look-ahead, in two places:
- every feature of day t uses data up to and including day t only;
- when the model is re-estimated on day T, it only uses rows whose target window has fully
  ended by T (a 22-day-ahead target that ends after T is not known on T).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS: tuple[int, ...] = (1, 5, 22)
HAR_COLUMNS: list[str] = ["log_rv_d", "log_rv_w", "log_rv_m"]
WEEK: int = 5
MONTH: int = 22
REFIT_EVERY: int = 22  # trading days between re-estimations (about a month)


def har_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the three HAR regressors and, per horizon, the target and the day its window ends.

    Needs columns firm_id, date, rv. Rolling averages need a complete window, so the first 21
    days of a firm have no monthly regressor and are dropped later.
    """
    frame = panel.sort_values(["firm_id", "date"]).reset_index(drop=True)
    by_firm = frame.groupby("firm_id")["rv"]
    frame["log_rv_d"] = np.log(frame["rv"])
    frame["log_rv_w"] = np.log(by_firm.transform(lambda rv: rv.rolling(WEEK).mean()))
    frame["log_rv_m"] = np.log(by_firm.transform(lambda rv: rv.rolling(MONTH).mean()))
    for horizon in HORIZONS:
        # Average over days t+1..t+h: the rolling mean that ends at t+h, moved back to row t.
        ahead = by_firm.transform(lambda rv, h=horizon: rv.rolling(h).mean().shift(-h))
        frame[f"target_{horizon}"] = np.log(ahead)
        frame[f"target_end_{horizon}"] = frame.groupby("firm_id")["date"].shift(-horizon)
    return frame


@dataclass(frozen=True)
class PooledModel:
    """Fitted slopes, one intercept per firm, and the residual variance of the log target."""

    slopes: pd.Series
    firm_effects: pd.Series
    residual_variance: float


def fit_pooled(frame: pd.DataFrame, columns: list[str], target: str) -> PooledModel:
    """Least squares with firm fixed effects, done the transparent way.

    Subtracting each firm's own mean from the target and from every regressor removes the firm
    intercepts; ordinary least squares on the demeaned data gives the common slopes; each
    firm's intercept is then its mean target minus the slopes times its mean regressors.
    """
    data = frame.dropna(subset=[*columns, target])
    firm_means = data.groupby("firm_id")[[*columns, target]].mean()
    demeaned = data[[*columns, target]] - firm_means.loc[data["firm_id"]].to_numpy()
    slopes, *_ = np.linalg.lstsq(
        demeaned[columns].to_numpy(), demeaned[target].to_numpy(), rcond=None
    )
    slopes = pd.Series(slopes, index=columns)
    firm_effects = firm_means[target] - firm_means[columns].to_numpy() @ slopes.to_numpy()
    residuals = demeaned[target].to_numpy() - demeaned[columns].to_numpy() @ slopes.to_numpy()
    degrees_of_freedom = len(data) - len(firm_means) - len(columns)
    return PooledModel(slopes, firm_effects, float(residuals @ residuals / degrees_of_freedom))


def predict_log(model: PooledModel, frame: pd.DataFrame) -> pd.Series:
    """Predicted log variance; missing for firms the model has not seen or incomplete rows."""
    columns = list(model.slopes.index)
    firm_part = frame["firm_id"].map(model.firm_effects)
    return firm_part + frame[columns].to_numpy() @ model.slopes.to_numpy()


def expanding_forecasts(
    frame: pd.DataFrame,
    columns: list[str],
    horizon: int,
    first_origin: pd.Timestamp,
    last_origin: pd.Timestamp,
    refit_every: int = REFIT_EVERY,
) -> pd.DataFrame:
    """Out-of-sample forecasts for every origin day between `first_origin` and `last_origin`.

    The model is re-estimated every `refit_every` trading days on all rows whose target window
    had ended by the re-estimation day (expanding window), then used unchanged until the next
    re-estimation. Returns firm_id, date, horizon, forecast and actual (both as variances) and
    their logs. The variance forecast adds half the residual variance to the predicted log
    before exponentiating, because the average of a log-normal variable lies above exp(mean).
    """
    target, target_end = f"target_{horizon}", f"target_end_{horizon}"
    usable = frame.dropna(subset=[*columns, target, target_end])
    origins = np.sort(
        usable.loc[usable["date"].between(first_origin, last_origin), "date"].unique()
    )
    pieces = []
    for start in range(0, len(origins), refit_every):
        block = origins[start : start + refit_every]
        known = usable[usable[target_end] <= block[0]]
        model = fit_pooled(known, columns, target)
        rows = usable[usable["date"].isin(block)]
        log_forecast = predict_log(model, rows)
        pieces.append(
            pd.DataFrame(
                {
                    "firm_id": rows["firm_id"],
                    "date": rows["date"],
                    "horizon": horizon,
                    "log_forecast": log_forecast,
                    "forecast": np.exp(log_forecast + 0.5 * model.residual_variance),
                    "log_actual": rows[target],
                    "actual": np.exp(rows[target]),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True).dropna(subset=["forecast"])
