# Documents that were not scored

227 of 7,223 non-administrative documents (3.1%) have no usable text and were not sent to the LLM (DECISIONS #48). Counts only; the test period appears as a count and nothing else.

Reasons: `not_pdf` = filed in another format (mostly annual reports as ESEF packages from 2021 on); `empty` = scanned image without a text layer; `garbled` = mostly tables or broken characters; `unreadable` / `missing_file` = the file could not be opened or downloaded.

## By document class and reason

| doc_class | empty | garbled | missing_file | not_pdf | unreadable | total |
|---|---|---|---|---|---|---|
| periodic_report | 9 | 1 | 0 | 151 | 0 | 161 |
| inside_information | 32 | 18 | 0 | 1 | 5 | 56 |
| other | 5 | 0 | 1 | 0 | 0 | 6 |
| quarterly_information | 0 | 0 | 2 | 0 | 1 | 3 |
| takeover | 1 | 0 | 0 | 0 | 0 | 1 |

## By document class and period

| doc_class | test | train | val | total |
|---|---|---|---|---|
| periodic_report | 32 | 54 | 75 | 161 |
| inside_information | 18 | 33 | 5 | 56 |
| other | 1 | 4 | 1 | 6 |
| quarterly_information | 1 | 2 | 0 | 3 |
| takeover | 0 | 1 | 0 | 1 |

## By year and reason

| year | empty | garbled | missing_file | not_pdf | unreadable | total |
|---|---|---|---|---|---|---|
| 2019 | 11 | 3 | 0 | 1 | 1 | 16 |
| 2020 | 4 | 10 | 0 | 0 | 0 | 14 |
| 2021 | 4 | 1 | 2 | 12 | 4 | 23 |
| 2022 | 7 | 1 | 0 | 33 | 0 | 41 |
| 2023 | 5 | 1 | 0 | 35 | 0 | 41 |
| 2024 | 3 | 1 | 0 | 40 | 1 | 45 |
| 2025 | 13 | 2 | 1 | 31 | 0 | 47 |

## Firms with the most unscored documents

DG 25, BNP 17, SAN 17, ENGI 11, VIE 10, URW 9, ACA 8, EN 7.

What it means: 71% of the unscored documents are periodic reports (annual and half-year reports), whose news is normally also in a press release that is scored. 56 are press releases (`inside_information`), under 1% of that class.
