"""The one record layout that both filing sources are translated into.

The API returns JSON rows with long field names; the DILA archive holds one XML file per filing.
Each source module has a small "translator" that produces a dict with exactly these keys, so
everything downstream (timestamps, dedupe, alignment) is written once and does not care where a
filing came from.
"""

from __future__ import annotations

FILING_FIELDS: tuple[str, ...] = (
    "doc_id",  # PDF file name without extension, e.g. FCMKW110450_20240125 (unique per filing)
    "source",  # "api" or "tarball"
    "pdf_path",  # path of the PDF inside the feed, e.g. MKW/2024/01/FCMKW110450_20240125.pdf
    "isin",
    "issuer_name",
    "feed_lei",  # LEI as typed into the feed (often empty or outdated; never used for filtering)
    "feed_ticker",
    "distributor",  # wire service that sent the filing (MKW, BWR, ECO, ...)
    "lang",  # "fr", "en" or the raw value in lower case
    "title",
    "type_code",  # primary AMF type code, e.g. 001007
    "type_codes",  # all type codes of the filing joined by ";"
    "status",  # NEW, REC (correction) or ADD
    "diffusion",  # OBL (mandatory) or OPT (optional)
    "ts_amf",  # the three timestamps exactly as the feed gives them (ISO strings)
    "ts_market",
    "ts_emit",
)

PDF_BASE_URL: str = "https://fr.ftp.opendatasoft.com/datadila/INFOFI/"


def doc_id_from_pdf_path(pdf_path: str) -> str:
    """'MKW/2024/01/FCMKW110450_20240125.pdf' -> 'FCMKW110450_20240125'."""
    file_name = pdf_path.rsplit("/", 1)[-1]
    return file_name.rsplit(".", 1)[0]


def normalise_language(raw: str | None) -> str:
    """Map the feed's language labels ('Français', 'FR', 'Anglais', 'EN', ...) to 'fr' / 'en'."""
    value = (raw or "").strip().lower()
    if value in {"fr", "français", "francais", "french"}:
        return "fr"
    if value in {"en", "anglais", "english"}:
        return "en"
    return value
