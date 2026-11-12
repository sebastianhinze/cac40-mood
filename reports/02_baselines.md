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
