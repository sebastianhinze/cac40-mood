"""From filings to events, and from abnormal returns to tests.

An "event" is one firm on one event session. Several documents of the same firm that map to the
same session are one event: the market reacts to the day's news as a whole, and counting them
separately would count the same return several times.

Inference: events cluster in time (results seasons; many firms react to the same macro day), so
standard errors are clustered by event date, as fixed in the project rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cac40_mood.data.universe import EXCLUDED_FROM_MAIN_ANALYSIS


def prepare_events(documents: pd.DataFrame, score_columns: list[str]) -> pd.DataFrame:
    """One row per firm and event session, from scored documents.

    Kept: non-admin documents of the 36 main-analysis firms, in the train or validation period,
    outside the spring-2020 feed hole, with an event session. Scores of several documents on
    the same firm-day are averaged with the model's confidence as weight (plain mean if there
    is no confidence column). `intraday` is True if any of the day's documents arrived during
    trading hours.
    """
    keep = (
        ~documents["is_admin"]
        & ~documents["firm_id"].isin(EXCLUDED_FROM_MAIN_ANALYSIS)
        & documents["split"].isin(["train", "val"])
        & ~documents["in_feed_gap"]
        & documents["event_session"].notna()
    )
    kept = documents[keep]
    weights = (
        kept["confidence"] if "confidence" in kept.columns else pd.Series(1.0, index=kept.index)
    )
    weights = weights.fillna(0.0).clip(lower=1e-6)
    rows = []
    for (firm_id, session), group in kept.groupby(["firm_id", "event_session"]):
        row = {
            "firm_id": firm_id,
            "event_session": session,
            "n_documents": len(group),
            "intraday": bool(group["intraday"].any()),
        }
        for column in score_columns:
            valid = group[column].notna()
            row[column] = (
                np.average(group.loc[valid, column], weights=weights[group.index][valid])
                if valid.any()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def clustered_regression(
    table: pd.DataFrame, outcome: str, regressors: list[str], cluster: str = "event_session"
) -> pd.DataFrame:
    """OLS of `outcome` on `regressors` (plus a constant) with standard errors clustered by date.

    Returns one row per coefficient: estimate, standard error, t statistic, p-value, and the
    95% confidence interval; `n` and `clusters` are repeated on every row for convenience.
    """
    data = table.dropna(subset=[outcome, *regressors])
    design = sm.add_constant(data[regressors].astype(float))
    groups = pd.factorize(data[cluster])[0]
    fit = sm.OLS(data[outcome].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    interval = fit.conf_int()
    return pd.DataFrame(
        {
            "coefficient": fit.params,
            "std_error": fit.bse,
            "t": fit.tvalues,
            "p_value": fit.pvalues,
            "ci_low": interval[0],
            "ci_high": interval[1],
            "n": int(fit.nobs),
            "clusters": len(np.unique(groups)),
        }
    )


def mean_car_by_group(table: pd.DataFrame, group: str, outcome: str) -> pd.DataFrame:
    """Average CAR per group with a date-clustered standard error (regression on a constant)."""
    rows = []
    for label, members in table.dropna(subset=[outcome]).groupby(group, observed=True):
        constant_only = members.assign(_one=1.0)
        groups = pd.factorize(constant_only["event_session"])[0]
        fit = sm.OLS(constant_only[outcome].astype(float), constant_only[["_one"]]).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        rows.append(
            {
                group: label,
                "events": len(members),
                "mean": float(fit.params["_one"]),
                "std_error": float(fit.bse["_one"]),
                "t": float(fit.tvalues["_one"]),
            }
        )
    return pd.DataFrame(rows)


def cumulative_paths(table: pd.DataFrame, group: str) -> pd.DataFrame:
    """Average cumulative abnormal return from day -5 to day +5, one column per group.

    This is the data behind the "CAR path by tone tercile" chart.
    """
    columns = sorted(
        (name for name in table.columns if name.startswith("ar_")),
        key=lambda name: int(name.split("_")[1]),
    )
    complete = table.dropna(subset=columns)
    paths = complete.groupby(group, observed=True)[columns].mean().cumsum(axis=1).T
    paths.index = [int(name.split("_")[1]) for name in columns]
    paths.index.name = "day"
    return paths


def terciles(values: pd.Series) -> pd.Series:
    """Label each value low / middle / high by thirds of the distribution."""
    return pd.qcut(values, 3, labels=["low", "middle", "high"])
