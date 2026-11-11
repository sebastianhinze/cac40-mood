"""Read the raw DILA archive: hourly `.tar.gz` bundles with one XML + one PDF per filing.

Why we need it: the API is a convenient copy of this archive, but it is rebuilt regularly and can
return incomplete answers while a rebuild is running. The bundles are plain files that never
change, so for the early part of the sample (Jan 2019 - Jun 2020, DECISIONS #4) they are the
source of record, and they let us measure how complete the API is.

A bundle contains every issuer (about 7 MB on average). We only keep the XML and PDF files of our
own ISINs and write one manifest line per bundle, so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from cac40_mood.data.polite_http import PoliteSession
from cac40_mood.data.schema import doc_id_from_pdf_path, normalise_language

ARCHIVE_URL: str = "https://echanges.dila.gouv.fr/OPENDATA/AMF/"

# amf-gz_DDMMYYYY_HHMM.tar.gz -> the moment the bundle was produced.
TARBALL_NAME = re.compile(r"amf-gz_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})\.tar\.gz")

# Only member names of this exact shape are ever written to disk. Anything else (for example a
# name containing "..") is skipped, so a malformed archive cannot write outside our folders.
# The content folder is called "pdf/" but a few filings are .docx, .htm or .zip files.
SAFE_MEMBER = re.compile(r"(xml|pdf)/[A-Za-z0-9]+/\d{4}/\d{2}/[A-Za-z0-9_\-]+\.[A-Za-z0-9]{2,5}")

# A filing sent late in the evening lands in a bundle of the next day, and corrections can
# trickle in later, so we read bundles for a few extra days after the window we care about.
TRAILING_DAYS: int = 7


@dataclass(frozen=True)
class Tarball:
    """One bundle in the archive listing."""

    name: str
    bundled_at: datetime


def parse_listing(html: str) -> list[Tarball]:
    """Extract all bundle names from the archive's directory page, oldest first."""
    found: dict[str, Tarball] = {}
    for match in TARBALL_NAME.finditer(html):
        day, month, year, hour, minute = (int(part) for part in match.groups())
        found[match.group(0)] = Tarball(match.group(0), datetime(year, month, day, hour, minute))
    return sorted(found.values(), key=lambda tarball: tarball.bundled_at)


def select_tarballs(tarballs: list[Tarball], start: date, end: date) -> list[Tarball]:
    """Bundles produced between `start` and `end` (+ a few trailing days for late arrivals)."""
    first = datetime(start.year, start.month, start.day)
    last = datetime(end.year, end.month, end.day) + timedelta(days=TRAILING_DAYS + 1)
    return [tarball for tarball in tarballs if first <= tarball.bundled_at < last]


def parse_metadata_xml(xml_bytes: bytes) -> dict[str, str]:
    """Translate one `MD*.xml` metadata file into the common filing record (see schema.py)."""
    root = ET.fromstring(xml_bytes)
    company = root.find("identificationSociete")
    deposit = root.find("InformationDeposee")
    content = root.find("FichierDeContenu")
    distributor = root.find("identificationDiffuseur")
    if company is None or deposit is None or content is None:
        raise ValueError("metadata XML lacks a company, deposit or content element")
    ticker = company.find("ISO_CODE_TKR")
    update = deposit.find("INF_UPG_INF")
    pdf_path = content.get("INF_FIC_NOM", "")
    type_codes = [node.text.strip() for node in deposit.findall("INF_STP_INF") if node.text]
    return {
        "doc_id": doc_id_from_pdf_path(pdf_path),
        "source": "tarball",
        "pdf_path": pdf_path,
        "isin": company.get("ISO_CD_ISI", "").strip(),
        "issuer_name": company.get("ISO_NOM_SOC", "").strip(),
        "feed_lei": company.get("ISO_CD_LEI", "").strip(),
        "feed_ticker": ticker.get("ISO_CD_TKR", "").strip() if ticker is not None else "",
        "distributor": distributor.get("IDI_COD_DIF", "") if distributor is not None else "",
        "lang": normalise_language(deposit.get("INF_LNG_INF")),
        "title": deposit.get("INF_TIT_INF", "").strip(),
        "type_code": deposit.get("INF_STP_PRI", "").strip(),
        "type_codes": ";".join(type_codes),
        "status": update.get("INF_UPG_STS", "") if update is not None else "",
        "diffusion": deposit.get("INF_COD_DIF", ""),
        "ts_amf": root.get("UIN_DAT_AMF", ""),
        "ts_market": root.get("UIN_DAT_MAR", ""),
        "ts_emit": deposit.get("INF_DAT_EMT", ""),
    }


def _write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def extract_our_filings(
    tar_bytes: bytes, isins: set[str], xml_dir: Path, pdf_dir: Path
) -> dict[str, int]:
    """Keep the XML and PDF files of our ISINs from one bundle; return simple counts.

    Two passes over the archive: first read every XML to learn which filings are ours, then save
    the matching PDFs. (We cannot rely on a PDF always coming right after its XML.)
    """
    counts = {"n_xml": 0, "n_ours": 0, "n_pdf_saved": 0, "n_bad_xml": 0, "n_unsafe_names": 0}
    wanted_pdfs: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        for member in members:
            if not SAFE_MEMBER.fullmatch(member.name):
                counts["n_unsafe_names"] += 1
                continue
            if not member.name.endswith(".xml"):
                continue
            counts["n_xml"] += 1
            data = archive.extractfile(member).read()
            try:
                record = parse_metadata_xml(data)
            except (ET.ParseError, ValueError):
                counts["n_bad_xml"] += 1
                continue
            if record["isin"] in isins:
                counts["n_ours"] += 1
                _write_bytes(xml_dir / member.name.removeprefix("xml/"), data)
                wanted_pdfs.add(record["pdf_path"])
        for member in members:
            pdf_path = member.name.removeprefix("pdf/")
            if member.name.startswith("pdf/") and pdf_path in wanted_pdfs:
                if not SAFE_MEMBER.fullmatch(member.name):
                    continue
                _write_bytes(pdf_dir / pdf_path, archive.extractfile(member).read())
                counts["n_pdf_saved"] += 1
    return counts


def read_manifest(manifest_path: Path) -> dict[str, dict]:
    """Return {bundle name: manifest entry} for every bundle processed so far."""
    if not manifest_path.exists():
        return {}
    entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    return {entry["name"]: entry for entry in entries}


def fetch_window(
    session: PoliteSession,
    isins: set[str],
    start: date,
    end: date,
    raw_dir: Path,
    pdf_dir: Path,
) -> dict[str, int]:
    """Download and filter every bundle of the window that is not in the manifest yet.

    `raw_dir` receives `listing.html`, `manifest.jsonl` and the kept XML files; PDFs go to the
    shared `pdf_dir` (same relative paths as in the feed). A bundle that cannot be read is
    recorded with an "error" entry and skipped, so one bad file does not stop a long run.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    listing = session.get(ARCHIVE_URL)
    listing.raise_for_status()
    (raw_dir / "listing.html").write_text(listing.text)
    manifest_path = raw_dir / "manifest.jsonl"
    # Bundles that failed last time carry an "error" entry and are tried again.
    done = {name for name, entry in read_manifest(manifest_path).items() if "error" not in entry}
    in_window = select_tarballs(parse_listing(listing.text), start, end)
    todo = [tarball for tarball in in_window if tarball.name not in done]
    totals = {"bundles": 0, "errors": 0, "n_ours": 0}
    for tarball in todo:
        entry: dict[str, object] = {
            "name": tarball.name,
            "processed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        try:
            response = session.get(ARCHIVE_URL + tarball.name, timeout=600.0)
            response.raise_for_status()
            entry["bytes"] = len(response.content)
            entry.update(extract_our_filings(response.content, isins, raw_dir / "xml", pdf_dir))
            totals["n_ours"] += int(entry["n_ours"])
        except Exception as error:  # any failure is logged and the run continues
            entry["error"] = f"{type(error).__name__}: {error}"
            totals["errors"] += 1
        with manifest_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        totals["bundles"] += 1
    return totals


def read_raw_records(raw_dir: Path) -> list[dict[str, str]]:
    """Parse every kept XML file back into filing records, in a stable order."""
    records: list[dict[str, str]] = []
    for path in sorted((raw_dir / "xml").rglob("*.xml")):
        records.append(parse_metadata_xml(path.read_bytes()))
    return records
