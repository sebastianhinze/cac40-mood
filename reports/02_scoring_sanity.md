# Bulk scoring run (prompt v2): sanity report

Run of 2026-11-12, `claude-haiku-5-5`, thinking at low effort, first 12,000 tokens per document, Batches API.
6,996 documents with usable text were scored (the 227 without usable text were not). Cost of this run: $2.47.

Distributions below use **train and validation documents only**. For the test period only counts and failures are
shown; its scores exist but have not been looked at.

## Run overview (counts only, all periods)

| split | documents | problems | cut_at_cap | mean_input_tokens | mean_output_tokens |
|---|---|---|---|---|---|
| test | 1464 | 0 | 245 | 6516.0 | 123.0 |
| train | 4047 | 0 | 674 | 6441.0 | 123.0 |
| val | 1485 | 0 | 223 | 6333.0 | 123.0 |

No request failed and no answer was invalid or out of range.

## Scores per topic (train + validation, 5,532 documents)

| topic | documents | share | tone_mean | tone_sd | tone_negative_share | surprise_nonzero_share | uncertainty_mean | confidence_mean |
|---|---|---|---|---|---|---|---|---|
| m_and_a | 960 | 0.174 | 0.081 | 0.105 | 0.03 | 0.045 | 0.261 | 0.693 |
| results | 863 | 0.156 | 0.34 | 0.514 | 0.195 | 0.771 | 0.319 | 0.718 |
| contract_win | 553 | 0.1 | 0.126 | 0.06 | 0.002 | 0.004 | 0.183 | 0.724 |
| other | 502 | 0.091 | 0.008 | 0.093 | 0.024 | 0.016 | 0.11 | 0.807 |
| sales_update | 446 | 0.081 | 0.207 | 0.329 | 0.126 | 0.482 | 0.269 | 0.684 |
| strategy_operations | 406 | 0.073 | 0.121 | 0.144 | 0.034 | 0.106 | 0.275 | 0.686 |
| governance | 397 | 0.072 | 0.015 | 0.107 | 0.055 | 0.053 | 0.146 | 0.81 |
| shareholder_returns | 386 | 0.07 | 0.071 | 0.123 | 0.008 | 0.036 | 0.082 | 0.813 |
| financing | 384 | 0.069 | 0.046 | 0.107 | 0.029 | 0.047 | 0.155 | 0.785 |
| product_pipeline | 316 | 0.057 | 0.163 | 0.138 | 0.035 | 0.079 | 0.295 | 0.695 |
| esg | 144 | 0.026 | 0.093 | 0.055 | 0.0 | 0.0 | 0.167 | 0.805 |
| legal_regulatory | 101 | 0.018 | -0.128 | 0.321 | 0.535 | 0.446 | 0.457 | 0.66 |
| guidance | 74 | 0.013 | 0.386 | 0.52 | 0.203 | 0.946 | 0.364 | 0.749 |

`tone_negative_share` = share of documents with tone below -0.15. `surprise_nonzero_share` = share with a
surprise other than 0.

## Scores per year (train + validation)

| year | documents | tone_mean | surprise_mean | uncertainty_mean | confidence_mean | tone_sd | tone_negative_share |
|---|---|---|---|---|---|---|---|
| 2019.0 | 1068.0 | 0.125 | 0.018 | 0.206 | 0.731 | 0.238 | 0.059 |
| 2020.0 | 783.0 | 0.013 | -0.008 | 0.308 | 0.727 | 0.336 | 0.188 |
| 2021.0 | 1145.0 | 0.126 | 0.051 | 0.232 | 0.735 | 0.288 | 0.071 |
| 2022.0 | 1052.0 | 0.184 | 0.059 | 0.223 | 0.738 | 0.267 | 0.04 |
| 2023.0 | 961.0 | 0.179 | 0.053 | 0.206 | 0.736 | 0.254 | 0.041 |
| 2024.0 | 523.0 | 0.126 | 0.023 | 0.182 | 0.753 | 0.231 | 0.046 |

2024 covers January to June only.

## What to read from this

- Tone is mildly positive on average (0.13); half of all documents lie between
  0.0 and 0.2. Corporate news is mostly good or neutral news.
- 2020 stands out: lowest average tone, highest uncertainty, and about one document in five is negative.
  The spring-2020 feed gap means few 2020 documents come from the worst months.
- Results and guidance documents carry almost all of the non-zero surprise and the widest spread of tone.
  Contract wins, ESG and financing news cluster near zero, as the materiality instruction intends.
- Legal and regulatory news is the only topic with a negative average tone.
- Surprise is 0 for 79% of documents, which is why it is a secondary feature (DECISIONS #40).
