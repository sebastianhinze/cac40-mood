"""Download filing metadata from the DILA "info-financiere" API (the French OAM).

One request per ISIN returns every filing of that issuer in the date window. The raw answer is
stored untouched under `data/raw/amf_api/`, one JSON-lines file per ISIN, plus a manifest that
records when and how it was fetched. Processing later reads only those files, so it can be
re-run offline and always sees the same input.

Facts about the API that shaped this code (see docs/DATA_SOURCES.md):
- we filter on the ISIN field; the feed's own `cac40` flag is a stale 2011 list;
- ODSQL has no usable `in (...)`, so each ISIN gets its own request;
- the `exports` endpoint returns all matching rows at once (the `records` endpoint stops at 10k);
- **the search index is rebuilt several times a day, and while that runs whole years of rows
  are temporarily missing.** A snapshot is therefore only accepted when the live row count
  equals the catalog's row count both before and after the download.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from cac40_mood.data.polite_http import PoliteSession
from cac40_mood.data.schema import doc_id_from_pdf_path, normalise_language

API_DATASET_URL: str = (
    "https://www.info-financiere.gouv.fr/api/explore/v2.1/catalog/datasets/flux-amf-new-prod"
)
ISIN_FIELD: str = "identificationsociete_iso_cd_isi"
DATE_FIELD: str = "uin_dat_amf"


class IndexRebuildingError(RuntimeError):
    """The API index was incomplete or changed during the download; try again later."""


def build_where(isin: str, start: date, end: date) -> str:
    """ODSQL filter: one ISIN, AMF timestamp inside [start - 1 day, end + 2 days).

    The API compares in UTC while our sample is defined in Paris time, so we ask for one extra
    day on each side and cut exactly later, once timestamps are converted to Europe/Paris.
    """
    first = start - timedelta(days=1)
    last = end + timedelta(days=2)
    return (
        f"{ISIN_FIELD}='{isin}' and {DATE_FIELD}>=date'{first.isoformat()}' "
        f"and {DATE_FIELD}<date'{last.isoformat()}'"
    )


def index_row_counts(session: PoliteSession) -> tuple[int, int]:
    """Return (rows the search index answers with right now, rows the catalog says exist)."""
    live = session.get(f"{API_DATASET_URL}/records", params={"limit": "0"})
    live.raise_for_status()
    catalog = session.get(API_DATASET_URL)
    catalog.raise_for_status()
    return (
        int(live.json()["total_count"]),
        int(catalog.json()["metas"]["default"]["records_count"]),
    )


def fetch_isin(session: PoliteSession, isin: str, start: date, end: date) -> list[dict]:
    """Return all API rows for one ISIN as a list of dicts (one dict per filing)."""
    response = session.get(
        f"{API_DATASET_URL}/exports/jsonl", params={"where": build_where(isin, start, end)}
    )
    response.raise_for_status()
    lines = response.text.splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def fetch_snapshot(
    session: PoliteSession, isins: list[str], start: date, end: date, out_dir: Path
) -> dict[str, int]:
    """Download all ISINs as one consistent snapshot and return {isin: number of rows}.

    All-or-nothing: rows are written to a temporary folder that only replaces `out_dir` when
    the index was complete before and after the run. The whole download takes under a minute,
    so starting over is cheaper than reasoning about a half-trustworthy snapshot.
    """
    before = index_row_counts(session)
    if before[0] != before[1]:
        raise IndexRebuildingError(f"index has {before[0]} of {before[1]} rows")
    partial_dir = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(partial_dir, ignore_errors=True)
    partial_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    for isin in isins:
        rows = fetch_isin(session, isin, start, end)
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        (partial_dir / f"{isin}.jsonl").write_text("\n".join(lines) + "\n")
        counts[isin] = len(rows)
    after = index_row_counts(session)
    if after != before:
        shutil.rmtree(partial_dir)
        raise IndexRebuildingError(f"index changed during the download: {before} -> {after}")
    manifest = {
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": [start.isoformat(), end.isoformat()],
        "index_rows": after[0],
        "rows_per_isin": counts,
    }
    (partial_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(out_dir, ignore_errors=True)
    partial_dir.rename(out_dir)
    return counts


def _type_codes(value: object) -> list[str]:
    """The API gives one code as a string and several codes as a list (or a JSON-encoded list)."""
    if isinstance(value, list):
        return [str(code) for code in value]
    text = str(value or "").strip()
    if text.startswith("["):
        return [str(code) for code in json.loads(text)]
    return [text] if text else []


def normalise_api_row(row: dict) -> dict[str, str]:
    """Translate one raw API row into the common filing record (see schema.py)."""
    pdf_path = row.get("fichierdecontenu_inf_fic_nom") or ""
    codes = _type_codes(row.get("informationdeposee_inf_stp_inf"))
    primary = str(row.get("informationdeposee_inf_stp_pri") or (codes[0] if codes else ""))
    return {
        "doc_id": doc_id_from_pdf_path(pdf_path),
        "source": "api",
        "pdf_path": pdf_path,
        "isin": (row.get(ISIN_FIELD) or "").strip(),
        "issuer_name": (row.get("identificationsociete_iso_nom_soc") or "").strip(),
        "feed_lei": (row.get("identificationsociete_iso_cd_lei") or "").strip(),
        "feed_ticker": (row.get("identificationsociete_iso_code_tkr_iso_cd_tkr") or "").strip(),
        "distributor": row.get("identificationdiffuseur_idi_cod_dif") or "",
        "lang": normalise_language(row.get("informationdeposee_inf_lng_inf")),
        "title": (row.get("informationdeposee_inf_tit_inf") or "").strip(),
        "type_code": primary,
        "type_codes": ";".join(codes),
        "status": row.get("informationdeposee_inf_upg_inf_inf_upg_sts") or "",
        "diffusion": row.get("informationdeposee_inf_cod_dif") or "",
        "ts_amf": row.get("uin_dat_amf") or "",
        "ts_market": row.get("uin_dat_mar") or "",
        "ts_emit": row.get("informationdeposee_inf_dat_emt") or "",
    }


def read_raw_records(raw_dir: Path) -> list[dict[str, str]]:
    """Read back every stored API row (all ISIN files) as filing records, in a stable order."""
    records: list[dict[str, str]] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open() as handle:
            records.extend(normalise_api_row(json.loads(line)) for line in handle if line.strip())
    return records
