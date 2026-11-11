"""The firm universe: 37 current CAC 40 members that file with the French OAM.

`universe.csv` is a hand-checked snapshot (2026-11-11):
- names and current ISINs match Euronext's official index composition page of that date;
- LEIs come from the GLEIF registry (looked up by ISIN);
- index-entry dates come from the composition history on fr.wikipedia "CAC 40" (CC BY-SA) and are
  only filled for firms that joined from 2017 on; an empty date means a long-standing member.
- `revenue_2018_eur_bn` is the firm's rounded 2018 revenue in billion euros (net banking income
  for the banks, gross revenues for AXA, the pro-forma figure for EssilorLuxottica), entered by
  hand as an order of magnitude. The scoring prompt uses it so the model can judge how much a
  piece of news matters for a firm of that size. It is fixed at the year before the sample
  starts, so it never carries information from later years (DECISIONS #37).

Why `isin_previous` matters: three firms changed ISIN after a share split inside the sample
(Dassault Systèmes 2021, Eurofins 2020, Michelin 2022). Filtering the feed on the current ISIN
alone would silently drop every filing made before the split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

UNIVERSE_CSV: Path = Path(__file__).with_name("universe.csv")

# Index members we deliberately leave out (DECISIONS #3): they file with the Dutch or
# Luxembourg regulator, so the French feed has almost nothing for them.
EXCLUDED_FOREIGN_FILERS: tuple[str, ...] = ("Airbus", "ArcelorMittal", "Stellantis")

# Firms kept in the data tables but left out of the main analysis (DECISIONS #29), with the reason.
# URW: Yahoo has its Paris prices only from 2023-04-14 (the earlier Amsterdam line is no longer
# served), so it has no returns or volatility for most of the sample.
EXCLUDED_FROM_MAIN_ANALYSIS: dict[str, str] = {
    "URW": "Yahoo prices start on 2023-04-14",
}


def load_universe(path: Path = UNIVERSE_CSV) -> pd.DataFrame:
    """Load the universe table, one row per firm, with dates parsed and empty cells as ''."""
    universe = pd.read_csv(path, dtype="string", keep_default_na=False)
    universe["index_entry_date"] = pd.to_datetime(universe["index_entry_date"], errors="coerce")
    universe["revenue_2018_eur_bn"] = universe["revenue_2018_eur_bn"].astype(float)
    if universe["firm_id"].duplicated().any():
        raise ValueError("universe.csv has a duplicated firm_id")
    return universe


def all_isins(universe: pd.DataFrame) -> dict[str, str]:
    """Map every ISIN a firm has used (current and previous) to its `firm_id`."""
    isin_to_firm: dict[str, str] = {}
    for firm in universe.itertuples(index=False):
        codes = [firm.isin, *[code for code in firm.isin_previous.split(";") if code]]
        for code in codes:
            if code in isin_to_firm:
                raise ValueError(f"ISIN {code} is listed for two firms")
            isin_to_firm[code] = firm.firm_id
    return isin_to_firm


def main_analysis_firms(universe: pd.DataFrame) -> list[str]:
    """The firm ids used in the event study and the volatility models (36 firms)."""
    return [firm for firm in universe["firm_id"] if firm not in EXCLUDED_FROM_MAIN_ANALYSIS]
