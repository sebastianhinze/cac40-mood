"""GARCH-X robustness: GARCH(1,1) with announcement columns in the variance equation.

    uv run python -m cac40_mood.vol.garchx

    return_t  = mu + e_t,   e_t ~ N(0, s2_t)
    s2_{t+1}  = omega + alpha * e_t^2 + beta * s2_t + gamma . x_t

x_t holds what is known at the close of day t: "a scored announcement arrived today" and, in
the second variant, the day's |tone| and uncertainty. This asks the same question as the HAR
models B1 and M, inside a different volatility model: if scores only helped in HAR by
accident, they should not help here.

Details:
- Estimated by maximum likelihood with the explicit recursion below (the `arch` package has
  no exogenous variance term), so every number can be traced. With no x columns the result
  must equal ordinary GARCH; a test checks that against `arch`.
- gamma may be negative (volatility often falls back after a news day), so the variance is
  floored at 1% of the sample variance to keep it positive.
- For forecasts beyond one day, future x is unknown and is replaced by its in-sample average.
- Subset: the ten firms with the most scored documents, chosen by that count before looking
  at any result. Sample from 2019 (no announcements are known for 2018); days in the
  spring-2020 feed hole get the in-sample average of x (neutral), since the recursion needs
  a value for every day.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cac40_mood.analysis_documents import load_documents
from cac40_mood.config import PROJECT_ROOT
from cac40_mood.splits import TRAIN_END, TRAIN_START, VAL_END
from cac40_mood.vol import evaluate, features, har
from cac40_mood.vol.garch import PERCENT
from cac40_mood.vol.run import load_panel
from cac40_mood.vol.sentiment_run import markdown, with_interval

REPORT_PATH: Path = PROJECT_ROOT / "reports" / "04_garchx_robustness.md"
N_FIRMS: int = 10
HORIZONS: tuple[int, ...] = (1, 5)
VARIANCE_FLOOR_SHARE: float = 0.01
VARIANTS: dict[str, list[str]] = {
    "GARCH": [],
    "GARCH-X event": ["event"],
    "GARCH-X scores": ["event", "abs_tone", "uncertainty"],
}


@dataclass(frozen=True)
class GarchXParameters:
    mu: float
    omega: float
    alpha: float
    beta: float
    gamma: np.ndarray  # one entry per x column (empty for plain GARCH)


def next_day_variances(
    returns_percent: np.ndarray, x: np.ndarray, parameters: GarchXParameters
) -> np.ndarray:
    """s2_{t+1} for every day t, by the recursion (started at the sample variance)."""
    surprises_squared = (returns_percent - parameters.mu) ** 2
    floor = VARIANCE_FLOOR_SHARE * surprises_squared.mean()
    extra = x @ parameters.gamma if x.shape[1] else np.zeros(len(returns_percent))
    variances = np.empty(len(returns_percent))
    current = surprises_squared.mean()
    for day in range(len(returns_percent)):
        current = (
            parameters.omega
            + parameters.alpha * surprises_squared[day]
            + parameters.beta * current
            + extra[day]
        )
        current = max(current, floor)
        variances[day] = current
    return variances


def negative_log_likelihood(
    returns_percent: np.ndarray, x: np.ndarray, parameters: GarchXParameters
) -> float:
    """Normal log-likelihood of days 2..T, each given the variance forecast made the day before."""
    forecast = next_day_variances(returns_percent, x, parameters)[:-1]
    surprises_squared = (returns_percent[1:] - parameters.mu) ** 2
    return float(0.5 * np.sum(np.log(2.0 * np.pi * forecast) + surprises_squared / forecast))


def fit_garchx(returns_percent: np.ndarray, x: np.ndarray) -> GarchXParameters:
    """Maximum likelihood for (omega, alpha, beta, gamma); mu is the sample mean return."""
    mu = float(returns_percent.mean())
    variance = float(returns_percent.var())

    def unpack(vector: np.ndarray) -> GarchXParameters:
        return GarchXParameters(mu, vector[0], vector[1], vector[2], vector[3:])

    start = np.concatenate([[0.04 * variance, 0.08, 0.88], np.zeros(x.shape[1])])
    bounds = [(1e-8, None), (0.0, 0.5), (0.0, 0.999), *[(None, None)] * x.shape[1]]
    result = minimize(
        lambda vector: negative_log_likelihood(returns_percent, x, unpack(vector)),
        start,
        method="L-BFGS-B",
        bounds=bounds,
    )
    return unpack(result.x)


def average_variance_ahead(
    next_day_variance: np.ndarray, parameters: GarchXParameters, mean_x: np.ndarray, horizon: int
) -> np.ndarray:
    """Average expected variance over the next `horizon` days.

    Beyond tomorrow, x is replaced by its average, which simply adds gamma . mean(x) to omega;
    from there the usual GARCH formula applies (geometric move towards the long-run level).
    """
    persistence = parameters.alpha + parameters.beta
    if persistence >= 1.0:
        return next_day_variance.copy()
    long_run = (parameters.omega + float(mean_x @ parameters.gamma)) / (1.0 - persistence)
    weights = persistence ** np.arange(horizon)
    return long_run + weights.mean() * (next_day_variance - long_run)


def firm_forecasts(
    firm: pd.DataFrame, columns: list[str], horizon: int, origins: np.ndarray
) -> pd.DataFrame:
    """Out-of-sample forecasts for one firm, re-estimated every 22 trading days, rescaled to
    the level of the range-based target (as for the plain GARCH baseline)."""
    firm = firm.dropna(subset=["ret"]).sort_values("date").reset_index(drop=True)
    returns = firm["ret"].to_numpy() * PERCENT
    dates = firm["date"].to_numpy()
    pieces = []
    for start in range(0, len(origins), har.REFIT_EVERY):
        block = origins[start : start + har.REFIT_EVERY]
        in_sample = dates <= block[0]
        x = firm[columns].to_numpy(dtype=float)
        mean_x = np.nanmean(x[in_sample], axis=0) if columns else np.zeros(0)
        x = np.where(np.isnan(x), mean_x, x)  # feed-hole days: the neutral value
        parameters = fit_garchx(returns[in_sample], x[in_sample])
        tomorrow = next_day_variances(returns, x, parameters)
        ahead = average_variance_ahead(tomorrow, parameters, mean_x, horizon) / PERCENT**2
        scale = np.nanmean(firm["rv"].to_numpy()[in_sample]) / ahead[in_sample].mean()
        in_block = np.isin(dates, block)
        pieces.append(pd.DataFrame({"date": dates[in_block], "forecast": ahead[in_block] * scale}))
    return pd.concat(pieces, ignore_index=True)


def subset_firms(documents: pd.DataFrame, count: int = N_FIRMS) -> list[str]:
    """The firms with the most scored documents."""
    return documents["firm_id"].value_counts().head(count).index.tolist()


def build_frame(panel: pd.DataFrame, documents: pd.DataFrame, firms: list[str]) -> pd.DataFrame:
    """Firm-days from 2019 with HAR targets, returns and the x columns."""
    frame = har.har_features(panel[panel["firm_id"].isin(firms)])
    frame = features.attach_features(frame, documents)
    frame["event"] = frame[features.EVENT_COLUMNS].max(axis=1)
    return frame[frame["date"] >= pd.Timestamp(TRAIN_START)].reset_index(drop=True)


def comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """GARCH against the two GARCH-X variants on validation, per horizon."""
    first_origin = pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)
    tables = []
    for horizon in HORIZONS:
        target = f"target_{horizon}"
        rows = frame.dropna(subset=[target])
        rows = rows[rows["date"].between(first_origin, pd.Timestamp(VAL_END))]
        origins = np.sort(rows["date"].unique())
        targets = rows[["firm_id", "date"]].assign(actual=np.exp(rows[target]))
        forecasts = {}
        for name, columns in VARIANTS.items():
            pieces = [
                firm_forecasts(firm, columns, horizon, origins).assign(firm_id=firm_id)
                for firm_id, firm in frame.groupby("firm_id")
            ]
            forecasts[name] = targets.merge(pd.concat(pieces), on=["firm_id", "date"])
        tables.append(evaluate.compare_models(forecasts, "GARCH", horizon))
    return with_interval(pd.concat(tables, ignore_index=True))


def full_sample_gammas(frame: pd.DataFrame) -> pd.DataFrame:
    """gamma per firm for the scores variant, estimated on the train period (description)."""
    columns = VARIANTS["GARCH-X scores"]
    rows = []
    for firm_id, firm in frame[frame["date"] <= pd.Timestamp(TRAIN_END)].groupby("firm_id"):
        firm = firm.dropna(subset=["ret"])
        x = firm[columns].to_numpy(dtype=float)
        x = np.where(np.isnan(x), np.nanmean(x, axis=0), x)
        parameters = fit_garchx(firm["ret"].to_numpy() * PERCENT, x)
        rows.append({"firm_id": firm_id, **dict(zip(columns, parameters.gamma, strict=True))})
    return pd.DataFrame(rows)


def main() -> None:
    documents, _ = load_documents()
    firms = subset_firms(documents)
    frame = build_frame(load_panel(), documents, firms)
    table = comparison(frame)
    gammas = full_sample_gammas(frame)
    lines = [
        "# GARCH-X robustness (validation period)",
        "",
        f"Ten firms with the most scored documents: {', '.join(firms)}. GARCH(1,1) on daily",
        "returns with announcement columns added to the variance equation; forecasts rescaled",
        "to the Garman-Klass target as for the plain GARCH baseline. Lower QLIKE is better;",
        "negative `qlike_minus_baseline` = better than plain GARCH. Exploratory.",
        "",
        markdown(table.drop(columns=["firm_days"]), 5),
        "",
        "`GARCH-X event`: x = an announcement arrived today. `GARCH-X scores`: x = that, plus",
        "the day's |tone| and uncertainty.",
        "",
        "## gamma per firm, scores variant, train period (in squared-percent units)",
        "",
        markdown(gammas, 3),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(table.round(5).to_string(index=False))
    print(gammas.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
