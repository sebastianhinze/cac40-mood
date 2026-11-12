# Baseline scorers vs. the LLM (train + validation, 5,532 documents)

All three scorers read the same text (the first 12,000 tokens of each document). Test-period documents are
scored too, but nothing is compared on them. Agreement with my own labels follows once the gold set is labelled.

| Scorer | What it is | Mean tone |
|---|---|---|
| LLM | `claude-haiku-5-5`, prompt v2 | 0.13 |
| Local model | French: `bardsai/finance-sentiment-fr-base`; English: `ProsusAI/finbert`; mean of P(positive) − P(negative) over up to 100 prose sentences (62 on average) | 0.15 |
| Lexicon | hand-written French/English finance word lists; (positive − negative) / (positive + negative) | 0.60 |

Rank correlation of tone (Spearman):

| | LLM | Local model | Lexicon |
|---|---|---|---|
| LLM | 1.00 | 0.54 | 0.07 |
| Local model | 0.54 | 1.00 | 0.41 |
| Lexicon | 0.07 | 0.41 | 1.00 |

- The local model agrees moderately with the LLM, about equally for French (0.55) and English (0.53) documents.
- The lexicon hardly agrees with the LLM overall. On results, guidance and sales documents (1,383) it does
  much better (0.50; local model 0.57): word counts work where the text is about numbers going up or down,
  and fail on contracts, deals and governance news, where positive words are everywhere.
- The lexicon's average tone of 0.60 shows the positive wording of corporate releases; the LLM was told to judge
  substance and materiality and averages 0.13.
- Lexicon uncertainty and LLM uncertainty: rank correlation 0.33.

## Loughran-McDonald on the official English versions (added 2026-11-12)

The standard finance dictionary (Loughran and McDonald 2011) is English only. It is applied to documents with an
official English version: English originals (582) and the firms' own English twins of French filings (4,522);
5,104 train/validation documents in total, 4,823 of the 5,239 documents used in the analysis (92%). French-only
documents get no score. Licence: free for academic research, not redistributed (`docs/DATA_SOURCES.md` §9).

Rank correlation of tone (Spearman) on the 4,823 common documents:

| | LLM | Local model | My lexicon | Loughran-McDonald |
|---|---|---|---|---|
| LLM | 1.00 | 0.55 | 0.04 | 0.09 |
| Local model | 0.55 | 1.00 | 0.37 | 0.44 |
| My lexicon | 0.04 | 0.37 | 1.00 | 0.67 |
| Loughran-McDonald | 0.09 | 0.44 | 0.67 | 1.00 |

- Mean tone: Loughran-McDonald 0.30, my lexicon 0.61, local model 0.16, LLM 0.14.
- The two word lists agree with each other (0.67) and hardly with the LLM. My hand-written list behaves like the
  standard dictionary, so the weak lexicon result is not an artefact of my word choice.
- On results, guidance and sales documents (1,170) Loughran-McDonald vs. the LLM: 0.40.
- Uncertainty: Loughran-McDonald vs. the LLM 0.20.
- Against my 40 hand labels (36 with an English version): rank correlation 0.11 (`reports/02_gold_agreement.md`).
- Day-0 abnormal returns per standard deviation of tone, 4,111 common events: LLM 0.32% (t = 6.8), local model
  0.18% (t = 4.8), Loughran-McDonald 0.04% (t = 1.5), my lexicon 0.03% (t = 1.1) (`reports/03_event_study.md`).
- Volatility: adding Loughran-McDonald scores to HAR + announcement dummies changes nothing at any horizon
  (`reports/04_vol_sentiment_validation.md`).
