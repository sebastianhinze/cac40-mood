"""PDF -> plain text for the documents that will be scored.

Extractor: pypdfium2 (Google's PDFium engine; Apache-2.0 / BSD licence). Chosen in phase 1
(DECISIONS #20) after a side-by-side test on 111 real filings against pypdf and pdfplumber: it
recovered the most words, produced the least junk (pypdf splits words such as "cr oissance"),
and was 10-25 times faster. PyMuPDF was as good but is AGPL-licensed.

Only the first pages are read: the LLM sees roughly the first 4,000 tokens of a document
(DECISIONS #7), and the baselines must see the same text, so reading all 400 pages of an annual
report would be wasted work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

# 4,000 tokens are about 16,000 characters of French; a press-release page holds 2,000-3,500
# characters. Fifteen pages leave a wide margin even when the first pages are cover sheets.
MAX_PAGES: int = 15

# Quality thresholds. A real press release has thousands of characters; fewer than 200 means a
# scanned image or an empty shell. Normal prose is about 75-80% letters; tables of numbers
# lower that, so 40% only catches text that is mostly symbols or encoding debris.
MIN_CHARS: int = 200
MIN_LETTER_SHARE: float = 0.40


@dataclass(frozen=True)
class ExtractedText:
    """Text of the first pages of one PDF, with the bookkeeping needed for quality checks."""

    text: str
    n_pages: int
    pages_read: int


def clean_text(raw: str) -> str:
    """Normalise whitespace so the same document always gives the same text (and cache key).

    - Windows line ends -> "\\n"; form feeds and non-breaking spaces -> normal spaces;
    - characters PDFium emits for unreadable glyphs (U+FFFE, U+FFFD, NUL) are removed;
    - runs of spaces collapse to one space, runs of blank lines to one blank line.
    Words are never joined or altered.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n").replace("\N{NO-BREAK SPACE}", " ")
    text = re.sub("[\ufffe\ufffd\x00]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(pdf_path: Path, max_pages: int = MAX_PAGES) -> ExtractedText:
    """Read up to `max_pages` pages of a PDF and return their cleaned text."""
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        n_pages = len(document)
        pages_read = min(n_pages, max_pages)
        parts: list[str] = []
        for page_number in range(pages_read):
            page = document[page_number]
            text_page = page.get_textpage()
            parts.append(text_page.get_text_range())
            text_page.close()
            page.close()
    finally:
        document.close()
    return ExtractedText(clean_text("\n\n".join(parts)), n_pages, pages_read)


def text_quality(text: str) -> str:
    """Label extracted text as 'ok', 'empty' (too short to score) or 'garbled' (few letters)."""
    if len(text) < MIN_CHARS:
        return "empty"
    letters = sum(character.isalpha() for character in text)
    if letters / len(text) < MIN_LETTER_SHARE:
        return "garbled"
    return "ok"


def extract_document(pdf_path: Path) -> dict[str, object]:
    """Extract one stored file and describe the outcome as a flat record.

    Never raises: a missing file, a non-PDF attachment or a broken PDF becomes a quality label,
    because one bad file must not stop a run over thousands of documents.
    """
    record: dict[str, object] = {"text": "", "n_pages": 0, "pages_read": 0}
    if not pdf_path.exists():
        record["text_quality"] = "missing_file"
    elif pdf_path.suffix.lower() != ".pdf":
        record["text_quality"] = "not_pdf"
    else:
        try:
            extracted = extract_text(pdf_path)
        except Exception:  # PDFium raises its own error types for broken or encrypted files
            record["text_quality"] = "unreadable"
        else:
            record.update(
                text=extracted.text, n_pages=extracted.n_pages, pages_read=extracted.pages_read
            )
            record["text_quality"] = text_quality(extracted.text)
    record["text_chars"] = len(str(record["text"]))
    return record
