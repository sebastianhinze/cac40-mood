# Gold set: do the scorers agree with the hand labels?

40 documents from train and validation, labelled blind (tone -2..+2 and topic). None of them was used to tune the prompt. With this few documents every number below is rough: a rank correlation of 0.6 on 40 documents has a 95% interval of about 0.35 to 0.77.

## Tone

`spearman`: rank correlation with my label. `same_direction`: share with the same negative / neutral / positive class. `opposite_sign`: share where one says good and the other bad.

| scorer | n | spearman | same_direction | opposite_sign |
|---|---|---|---|---|
| LLM | 40 | 0.434 | 0.475 | 0.05 |
| Local model | 40 | 0.495 | 0.625 | 0.05 |
| Lexicon | 40 | 0.294 | 0.7 | 0.075 |
| Loughran-McDonald | 36 | 0.113 | 0.444 | 0.139 |

## My direction (rows) against the LLM's (columns)

| mine \ LLM | negative | neutral | positive |
|---|---|---|---|
| negative | 2 | 2 | 2 |
| neutral | 1 | 8 | 1 |
| positive | 0 | 15 | 9 |

## Topic (LLM only)

Same topic on 35% of 40 documents (kappa 0.264).

## Documents where the LLM and I point in opposite directions

| issuer | title | human_tone | llm_tone | human_topic | llm_topic |
|---|---|---|---|---|---|
| Carrefour | Communiqués au titre de l'obligation d'information permanente / Information sur chiffre d’affaires annuel | -1 | 0.3 | sales_update | sales_update |
| Vinci | Précision concernant les perspectives 2023 | -1 | 0.3 | results | guidance |
