"""Report for the memory check: masked against original scores, and the timestamp check.

    uv run python -m cac40_mood.scoring.masking full     # first: the masked re-scoring
    uv run python -m cac40_mood.events.run               # first: abnormal returns per event
    uv run python -m cac40_mood.scoring.memory_report

Two questions behind the finding that surprise is related to returns *before* the filing date:
1. Hindsight: does the model score differently when it cannot tell which company and year?
2. Dating: do filings reach the feed later than the firm's own release time?
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from cac40_mood.config import PROJECT_ROOT
from cac40_mood.data.build import FILINGS_PATH, TEXTS_PATH
from cac40_mood.events import study
from cac40_mood.events.run import EVENTS_PATH
from cac40_mood.scoring.agreement import direction
from cac40_mood.scoring.masking import RESULT_PATH

REPORT_PATH: Path = PROJECT_ROOT / "reports" / "02_memory_check.md"
FIELDS: tuple[str, ...] = ("tone", "surprise", "uncertainty")
OUTCOMES: dict[str, str] = {"car_placebo": "CAR[-5,-1]", "car_0": "CAR[0]"}

MONTHS: dict[str, int] = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "décembre": 12, "decembre": 12, "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}  # fmt: skip
MONTH_NAMES: str = "|".join(MONTHS)
# "27 février 2019" or "le 1er mars 2021", followed (within a few characters) by "17h40",
# "17:40" or "7h".
PRINTED_DATE = re.compile(rf"(\d{{1,2}})(?:er)?\s+({MONTH_NAMES})\s+(20\d{{2}})", re.IGNORECASE)
PRINTED_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])\s?[hH:]\s?([0-5]\d)?(?!\d)")
HEAD_CHARS: int = 1_500
TIME_WITHIN_CHARS: int = 25


# ------------------------------------------------- masked against original


def agreement_rows(table: pd.DataFrame, label: str) -> list[dict[str, object]]:
    """Per score field: how close the masked score is to the original one."""
    rows = []
    for field in FIELDS:
        original, masked = table[field], table[f"masked_{field}"]
        rows.append(
            {
                "documents": label,
                "n": len(table),
                "field": field,
                "correlation": round(float(original.corr(masked)), 3),
                "mean_abs_difference": round(float((original - masked).abs().mean()), 3),
                "mean_masked_minus_original": round(float((masked - original).mean()), 3),
                "same_direction": round(
                    float((direction(original) == direction(masked)).mean()), 3
                ),
            }
        )
    return rows


def agreement_table(table: pd.DataFrame) -> pd.DataFrame:
    """All documents, then split by whether the model still named the right company."""
    rows = agreement_rows(table, "all")
    rows += agreement_rows(table[table["guessed_company"]], "company still recognised")
    rows += agreement_rows(table[~table["guessed_company"]], "company not recognised")
    return pd.DataFrame(rows)


def return_regressions(table: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Pre-event and day-0 abnormal returns on surprise: original, masked, and the difference.

    If hindsight drove the original scores, the part of surprise that disappears under masking
    (original minus masked) should itself go with the returns.
    """
    returns = events[["firm_id", "event_session", *OUTCOMES]]
    merged = table.merge(returns, on=["firm_id", "event_session"]).dropna(subset=list(OUTCOMES))
    merged = merged.assign(surprise_difference=merged["surprise"] - merged["masked_surprise"])
    regressors = {
        "original surprise": "surprise",
        "masked surprise": "masked_surprise",
        "original minus masked": "surprise_difference",
    }
    rows = []
    for outcome, label in OUTCOMES.items():
        for name, column in regressors.items():
            fit = study.clustered_regression(merged, outcome, [column]).loc[column]
            rows.append(
                {
                    "outcome": label,
                    "regressor": name,
                    "coefficient": fit["coefficient"],
                    "ci_low": fit["ci_low"],
                    "ci_high": fit["ci_high"],
                    "t": fit["t"],
                    "p_value": fit["p_value"],
                    "n": int(fit["n"]),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------- timestamp check


def printed_release_time(text: str) -> pd.Timestamp | None:
    """The release date and time a firm printed at the top of its release, if it printed both.

    Only a date that is directly followed by a time is used: a date alone is often the
    period end ("31 décembre 2022") or a meeting date, not the release date.
    """
    head = text[:HEAD_CHARS]
    for found in PRINTED_DATE.finditer(head):
        time = PRINTED_TIME.search(head[found.end() : found.end() + TIME_WITHIN_CHARS])
        if time is None:
            continue
        day, month, year = int(found.group(1)), MONTHS[found.group(2).lower()], int(found.group(3))
        try:
            return pd.Timestamp(year, month, day, int(time.group(1)), int(time.group(2) or 0))
        except ValueError:
            continue
    return None


def timestamp_check(filings: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """Results releases (train and validation) that print their own release time, with the
    feed's timestamp and the gap between the two in minutes (positive = feed is later)."""
    results = filings[
        filings["type_code"].isin(["001007", "001008"])
        & filings["split"].isin(["train", "val"])
        & ~filings["is_admin"]
    ].merge(texts[["doc_id", "text"]], on="doc_id")
    results = results.assign(printed=results["text"].map(printed_release_time))
    found = results.dropna(subset=["printed"]).copy()
    feed = found["ts_paris"].dt.tz_localize(None)
    found["minutes_feed_later"] = (feed - found["printed"]).dt.total_seconds() / 60.0
    found.attrs["results_releases"] = len(results)
    columns = ["doc_id", "firm_id", "printed", "ts_paris", "minutes_feed_later", "event_session"]
    return found[columns]


def markdown(table: pd.DataFrame, digits: int = 4) -> str:
    shown = table.round(digits).astype(object).where(table.notna(), "")
    lines = ["| " + " | ".join(str(c) for c in shown.columns) + " |", "|" + "---|" * shown.shape[1]]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in shown.to_numpy()]
    return "\n".join(lines)


def main() -> None:
    masked = pd.read_parquet(RESULT_PATH.with_name("masked_scores_full.parquet"))
    events = pd.read_parquet(EVENTS_PATH)
    agreement = agreement_table(masked)
    regressions = return_regressions(masked, events)
    times = timestamp_check(pd.read_parquet(FILINGS_PATH), pd.read_parquet(TEXTS_PATH))

    gap = times["minutes_feed_later"]
    close = gap.abs() <= 15
    shown = times.sample(min(20, len(times)), random_state=20).sort_values("printed")
    shown = shown.assign(printed=shown["printed"].dt.strftime("%Y-%m-%d %H:%M"))
    shown["feed"] = shown["ts_paris"].dt.strftime("%Y-%m-%d %H:%M")
    shown["event_session"] = shown["event_session"].dt.strftime("%Y-%m-%d")
    shown = shown[["firm_id", "printed", "feed", "minutes_feed_later", "event_session"]]
    surprise_lost = int(((masked["surprise"] != 0) & (masked["masked_surprise"] == 0)).sum())
    flips = int((np.sign(masked["surprise"]) * np.sign(masked["masked_surprise"]) < 0).sum())

    lines = [
        "# Memory check: masked re-scoring and release timestamps",
        "",
        "## 1. Does the model score differently when names and years are hidden?",
        "",
        f"{len(masked)} train/validation documents (35 with negative surprise, 35 positive, 30",
        "zero), re-scored with prompt v2 after hiding names (companies, brands, people, places,",
        "tickers) and replacing years by their distance to the publication year. Numbers were",
        "left in. For comparison: two identical runs on unmasked text differ by about 0.02 in",
        "tone (pilot).",
        "",
        markdown(agreement),
        "",
        f"- Surprise changed sign on {flips} document(s); {surprise_lost} went from non-zero to 0.",
        "- **The masking does not hide the company.** Asked to guess from the masked text, the",
        f"  model named the right company for {masked['guessed_company'].mean():.0%} of the",
        f"  documents and the right year for {masked['guessed_year'].mean():.0%}. The figures",
        "  and the description of the business are enough.",
        "",
        "## 2. Returns on original and masked surprise (the sample only)",
        "",
        "Coefficient = change in the abnormal return for a surprise higher by 1; standard errors",
        "clustered by date. About 100 documents: wide intervals.",
        "",
        markdown(regressions),
        "",
        "## 3. Is the feed's timestamp later than the firm's own release time?",
        "",
        f"{times.attrs['results_releases']} results releases (train and validation);",
        f"{len(times)} print their own release date and time at the top. For those:",
        "",
        f"- feed time within 15 minutes of the printed time: {close.mean():.0%} "
        f"({int(close.sum())} of {len(times)});",
        f"- feed more than 15 minutes **later**: {int((gap > 15).sum())}; "
        f"feed more than 15 minutes **earlier**: {int((gap < -15).sum())} "
        "(a printed time that is not the release time, e.g. of a conference call);",
        f"- median gap {gap.median():.1f} minutes.",
        "",
        "Twenty of them, drawn at random:",
        "",
        markdown(shown, 1),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
