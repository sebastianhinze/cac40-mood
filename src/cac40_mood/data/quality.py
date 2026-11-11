"""Data-quality tables for phase 1. The notebook only calls these functions and plots.

Everything here counts documents or checks coverage. Nothing looks at returns or volatility,
so including the test period (2024-07 onwards) in these tables does not "touch" it in the
sense of the evaluation rules: no outcome is observed.
"""

from __future__ import annotations

import pandas as pd


def firm_year_counts(filings: pd.DataFrame) -> pd.DataFrame:
    """Documents per firm (rows) and calendar year (columns), by Paris date of the filing."""
    years = filings["ts_paris"].dt.year.rename("year")
    table = pd.crosstab(filings["firm_id"], years)
    table["total"] = table.sum(axis=1)
    return table


def monthly_counts(filings: pd.DataFrame) -> pd.DataFrame:
    """Documents per month, split into scored and admin, to make coverage holes visible."""
    month = filings["ts_paris"].dt.tz_localize(None).dt.to_period("M").rename("month")
    kind = filings["is_admin"].map({True: "admin", False: "to_score"}).rename("kind")
    return pd.crosstab(month, kind)


def source_overlap(filings_all: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Per month inside [start, end]: documents seen in both sources, API only, archive only.

    This is the check on the API: the archive bundles are immutable files, so a document that is
    in a bundle but not in the API means the API copy is incomplete.
    """
    day = filings_all["ts_paris"].dt.tz_localize(None)
    window = filings_all[(day >= pd.Timestamp(start)) & (day < pd.Timestamp(end))]
    month = window["ts_paris"].dt.tz_localize(None).dt.to_period("M").rename("month")
    origin = pd.Series("both", index=window.index, name="origin")
    origin[window["in_api"] & ~window["in_tarball"]] = "api_only"
    origin[~window["in_api"] & window["in_tarball"]] = "archive_only"
    table = pd.crosstab(month, origin)
    return table.reindex(columns=["both", "api_only", "archive_only"], fill_value=0)


def timing_category(filings: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.Series:
    """Label each filing: pre_open, intraday, after_close or non_trading_day."""
    day = filings["ts_paris"].dt.tz_localize(None).dt.normalize()
    category = pd.Series("after_close", index=filings.index, name="timing")
    category[~day.isin(sessions)] = "non_trading_day"
    category[day.isin(sessions) & (filings["event_session"] == day)] = "pre_open"
    category[filings["intraday"]] = "intraday"
    return category


def class_summary(filings: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Per document class: how many documents, in which language, and when they arrive."""
    timing = timing_category(filings, sessions)
    summary = filings.groupby("doc_class").agg(
        documents=("doc_id", "size"),
        admin=("is_admin", "first"),
        french=("lang", lambda lang: int((lang == "fr").sum())),
        english=("lang", lambda lang: int((lang == "en").sum())),
    )
    shares = pd.crosstab(filings["doc_class"], timing, normalize="index").round(3)
    return summary.join(shares).sort_values("documents", ascending=False)


def text_quality_summary(filings: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """Per document class: extraction outcome counts and the share of usable ('ok') texts."""
    scored = filings[~filings["is_admin"]].merge(texts, on="doc_id", how="left")
    scored["text_quality"] = scored["text_quality"].fillna("not_extracted")
    table = pd.crosstab(scored["doc_class"], scored["text_quality"])
    table.loc["ALL"] = table.sum()
    table["usable_share"] = (table.get("ok", 0) / table.sum(axis=1)).round(4)
    return table


def alignment_examples(filings: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """One real filing per interesting case, with its timestamps and the session it maps to.

    The cases are picked by rule (first match in time order within train/validation), not by
    hand, so the table is reproducible.
    """
    scored = filings[~filings["is_admin"] & filings["split"].isin(["train", "val"])].copy()
    scored["timing"] = timing_category(scored, sessions)
    local = scored["ts_paris"].dt.tz_localize(None)
    weekday = local.dt.dayofweek
    cases = {
        "before the open": scored["timing"] == "pre_open",
        "during trading hours": scored["timing"] == "intraday",
        "after the close": (scored["timing"] == "after_close") & (weekday < 4),
        "Friday after the close": (scored["timing"] == "after_close") & (weekday == 4),
        "weekend": scored["timing"].eq("non_trading_day") & (weekday >= 5),
        "exchange holiday on a weekday": scored["timing"].eq("non_trading_day") & (weekday < 5),
        "deposited intraday, released after the close": (scored["ts_spread_minutes"] > 60)
        & (scored["timing"] == "after_close")
        & (pd.to_datetime(scored["ts_amf"], utc=True).dt.tz_convert("Europe/Paris").dt.hour < 17),
        "summer time (UTC+2)": (local.dt.month == 7) & (scored["timing"] == "after_close"),
    }
    rows = []
    for label, mask in cases.items():
        matches = scored[mask]
        if matches.empty:
            continue
        row = matches.iloc[len(matches) // 2]  # a document from the middle of the sample
        rows.append(
            {
                "case": label,
                "firm": row["firm_id"],
                "title": row["title"][:48],
                "ts_amf (feed)": row["ts_amf"],
                "ts_market (feed)": row["ts_market"],
                "ts_paris (used)": row["ts_paris"].strftime("%a %Y-%m-%d %H:%M"),
                "event_session": row["event_session"].strftime("%a %Y-%m-%d"),
                "intraday": bool(row["intraday"]),
            }
        )
    return pd.DataFrame(rows)


def timestamp_rule_effect(filings: pd.DataFrame) -> dict[str, int]:
    """How often "latest timestamp" changes the event session compared with the AMF time alone."""
    from cac40_mood.data.filings import parse_feed_timestamp
    from cac40_mood.data.sessions import to_paris

    amf_paris = to_paris(parse_feed_timestamp(filings["ts_amf"])).dt.tz_localize(None)
    used = filings["ts_paris"].dt.tz_localize(None)
    close = pd.Timedelta(hours=17, minutes=30)
    amf_before_close = (amf_paris - amf_paris.dt.normalize()) < close
    used_before_close = (used - used.dt.normalize()) < close
    different_day = amf_paris.dt.normalize() != used.dt.normalize()
    return {
        "documents": len(filings),
        "timestamps_disagree": int((filings["ts_spread_minutes"] > 0).sum()),
        "disagree_by_more_than_a_day": int((filings["ts_spread_minutes"] > 1440).sum()),
        "moved_to_a_later_session_side": int(
            (different_day | (amf_before_close & ~used_before_close)).sum()
        ),
    }
