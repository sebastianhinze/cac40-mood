"""Turn raw filing records (API + archive) into one clean table.

Steps, each a small pure function so it can be tested on hand-made rows:
1. combine the two sources, one row per document;
2. attach the firm (by ISIN, including pre-split ISINs);
3. pick the conservative timestamp and convert it to Paris time;
4. classify the AMF type code and tag admin filings;
5. flag English translations of French filings;
6. map each filing to its trading session (see sessions.py).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from cac40_mood.data.schema import FILING_FIELDS
from cac40_mood.data.sessions import align_to_sessions, to_paris
from cac40_mood.splits import label_split

TIMESTAMP_COLUMNS: tuple[str, ...] = ("ts_amf", "ts_market", "ts_emit")

# The feed itself is nearly empty between these dates (about 5% of the usual volume for our
# firms, in the API and in the raw archive alike): the distribution chain was disrupted during
# the first COVID lockdown. DECISIONS #28: the window is excluded from estimation and
# evaluation, and it is treated as MISSING data, never as "no news". A day inside the window
# with no filing tells us nothing about whether the firm published something.
FEED_GAP_START: date = date(2020, 3, 5)
FEED_GAP_END: date = date(2020, 6, 15)


def feed_is_observed(days: pd.Series) -> pd.Series:
    """True where the filings feed can be trusted, False inside the spring-2020 hole.

    Later phases must use this instead of testing "was there a filing?": inside the hole,
    announcement features (dummies, counts, scores) are set to missing, the rows are left out
    of regressions and forecast evaluation, and the particle filter runs its predict step only
    (no observation, so the uncertainty about the mood grows). Missing dates count as not
    observed.
    """
    day = pd.to_datetime(days).dt.normalize()
    in_gap = day.between(pd.Timestamp(FEED_GAP_START), pd.Timestamp(FEED_GAP_END))
    return ~in_gap & day.notna()


# First three digits of the AMF type code -> our own coarse document class.
TYPE_FAMILY_TO_CLASS: dict[str, str] = {
    "001": "inside_information",  # press releases: results, deals, guidance, contracts, ...
    "015": "inside_information",  # labelled "Informations privilégiées" in the feed as well
    "005": "quarterly_information",  # Q1 / Q3 sales and trading updates
    "003": "periodic_report",  # annual financial report
    "004": "periodic_report",  # half-year financial report
    "300": "periodic_report",  # universal registration document (URD)
    "008": "governance",
    "009": "takeover",
    "002": "availability_notice",  # "the annual report / AGM documents are now available"
    "006": "voting_rights",  # monthly total of shares and voting rights
    "013": "own_shares",  # buyback transaction logs and liquidity-contract statements
    "023": "threshold_crossing",
    "020": "debt_terms",  # final terms of bond issues
}

# Admin classes are counted (their counts can be features) but never sent to the LLM:
# they are forms and tables without any tone. The first three are DECISIONS #5; the notices and
# bond terms were added in phase 1 (DECISIONS #21).
ADMIN_CLASSES: frozenset[str] = frozenset(
    {"voting_rights", "own_shares", "threshold_crossing", "availability_notice", "debt_terms"}
)

# A French filing and its English translation are sent by the same wire within seconds or a
# few minutes of each other. Half an hour is generous without reaching the next press release.
TRANSLATION_WINDOW: pd.Timedelta = pd.Timedelta(minutes=30)


def combine_sources(
    api_records: list[dict[str, str]], tarball_records: list[dict[str, str]]
) -> pd.DataFrame:
    """One row per document. When both sources have it, keep the archive's copy.

    The archive files never change, while the API is a rebuilt copy, so the archive wins.
    `in_api` / `in_tarball` remember where a document was seen; the data-quality report uses
    them to measure how complete each source is.
    """
    columns = list(FILING_FIELDS)
    tarball = pd.DataFrame(tarball_records, columns=columns)
    api = pd.DataFrame(api_records, columns=columns)
    combined = pd.concat([tarball, api], ignore_index=True)
    combined["in_api"] = combined["doc_id"].isin(set(api["doc_id"]))
    combined["in_tarball"] = combined["doc_id"].isin(set(tarball["doc_id"]))
    combined = combined.drop_duplicates(subset="doc_id", keep="first")
    return combined.reset_index(drop=True)


def parse_feed_timestamp(values: pd.Series) -> pd.Series:
    """Feed timestamp strings (any UTC offset) -> UTC timestamps.

    The feed uses placeholder dates in the years 8887/8888 for "no market time". Those, and
    anything else outside 2000-2100, become missing so they can never win the "latest" rule.
    """
    text = values.astype("string")
    year = text.str.slice(0, 4)
    plausible = year.between("2000", "2100")
    parsed = pd.to_datetime(text.where(plausible), utc=True, errors="coerce", format="ISO8601")
    return parsed.dt.as_unit("ns")


def conservative_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Latest of the feed's timestamps (DECISIONS #13), plus the spread between them.

    Why the latest: some filings are deposited during the day with a release time after the
    close. Dating them too late costs a little power; dating them too early is look-ahead.
    """
    parsed = pd.DataFrame({name: parse_feed_timestamp(frame[name]) for name in TIMESTAMP_COLUMNS})
    latest = parsed.max(axis=1)
    earliest = parsed.min(axis=1)
    return pd.DataFrame(
        {
            "ts_utc": latest,
            "ts_spread_minutes": (latest - earliest).dt.total_seconds() / 60.0,
        }
    )


def classify_type_code(type_code: str) -> str:
    """AMF type code (e.g. '001007') -> document class (e.g. 'inside_information')."""
    return TYPE_FAMILY_TO_CLASS.get(str(type_code)[:3], "other")


def flag_translations(frame: pd.DataFrame) -> pd.DataFrame:
    """Find English filings that are translations of a French filing.

    Rule: within one firm and one AMF type code, an English filing is a translation if a French
    filing exists within 30 minutes of it. Each French filing can absorb only one English
    filing (the closest one in time), so two different press releases sent together stay two.

    Returns `is_translation` (True -> drop it) and `twin_doc_id` (the French original).
    Needs columns: doc_id, firm_id, type_code, lang, ts_utc.
    """
    result = pd.DataFrame(
        {
            "is_translation": False,
            "twin_doc_id": pd.Series(pd.NA, index=frame.index, dtype="string"),
        },
        index=frame.index,
    )
    dated = frame[frame["ts_utc"].notna()]
    for _, group in dated.groupby(["firm_id", "type_code"], sort=False):
        french = group[group["lang"] == "fr"].sort_values("ts_utc")
        english = group[group["lang"] == "en"].sort_values("ts_utc")
        if french.empty or english.empty:
            continue
        unused_french = dict(zip(french.index, french["ts_utc"], strict=True))
        for english_row, english_time in zip(english.index, english["ts_utc"], strict=True):
            if not unused_french:
                break
            closest = min(unused_french, key=lambda row: abs(unused_french[row] - english_time))
            if abs(unused_french[closest] - english_time) <= TRANSLATION_WINDOW:
                result.loc[english_row, "is_translation"] = True
                result.loc[english_row, "twin_doc_id"] = frame.loc[closest, "doc_id"]
                del unused_french[closest]
    return result


def build_filings(
    combined: pd.DataFrame,
    isin_to_firm: dict[str, str],
    sessions: pd.DatetimeIndex,
    sample_start: date,
    sample_end: date,
) -> pd.DataFrame:
    """Run steps 2-6 of the module docstring and return every filing with its flags.

    The result still contains translations (`is_translation`) and admin filings (`is_admin`);
    `keep_documents` gives the de-duplicated table the later phases work with.
    """
    filings = combined.copy()
    filings["firm_id"] = filings["isin"].map(isin_to_firm)
    filings = filings[filings["firm_id"].notna()].reset_index(drop=True)

    filings = filings.join(conservative_timestamp(filings))
    filings["ts_paris"] = to_paris(filings["ts_utc"])
    paris_day = filings["ts_paris"].dt.tz_localize(None).dt.normalize()
    in_sample = paris_day.between(pd.Timestamp(sample_start), pd.Timestamp(sample_end))
    filings = filings[in_sample].reset_index(drop=True)
    paris_day = paris_day[in_sample].reset_index(drop=True)

    filings["doc_class"] = filings["type_code"].map(classify_type_code)
    filings["is_admin"] = filings["doc_class"].isin(ADMIN_CLASSES)
    filings = filings.join(flag_translations(filings))
    filings = filings.join(align_to_sessions(filings["ts_paris"], sessions))
    filings["split"] = label_split(filings["event_session"])
    filings["in_feed_gap"] = ~feed_is_observed(paris_day)
    return filings.sort_values(["ts_utc", "doc_id"]).reset_index(drop=True)


def keep_documents(filings: pd.DataFrame) -> pd.DataFrame:
    """The table later phases use: one row per document, English translations removed."""
    return filings[~filings["is_translation"]].reset_index(drop=True)
