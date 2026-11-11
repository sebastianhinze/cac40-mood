"""Score documents: build the request, get the answer (cache, direct call or batch), validate.

For every document:
1. build the exact request that would be sent;
2. reuse the stored answer if the cache has one, so no document is paid for twice;
3. otherwise get an answer, either with one direct call or as part of a batch;
4. validate the answer against the schema and return one flat result row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from pydantic import ValidationError

from cac40_mood.scoring import llm
from cac40_mood.scoring.batch import BatchItem, run_batch
from cac40_mood.scoring.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from cac40_mood.scoring.schema import SCORE_JSON_SCHEMA, DocumentScore
from cac40_mood.scoring.spend import SpendLog

# Bulk-run settings. Thinking at low effort was chosen in the pilot (DECISIONS #31). Answers
# are about 125 tokens; 1,000 leaves room for some reasoning while keeping the worst-case cost
# of a request (which counts against the spend cap) small.
BULK_THINKING: str = "low"
BULK_MAX_OUTPUT_TOKENS: int = 1_000


@dataclass(frozen=True)
class ScoringConfig:
    """Everything that defines "how" a document is scored.

    - `thinking`: "off" (no reasoning before the answer) or an effort level such as "low" or
      "medium" (the model may reason first; that reasoning is billed as output tokens).
    - `max_doc_chars`: where the document text is cut.
    - `repeat`: 1 for normal runs. The pilot scores a document a second time with repeat=2 to
      measure how much the scores move between two identical requests (the model cannot be
      made deterministic: this model family does not accept a temperature setting).
    """

    model: str
    thinking: str
    max_doc_chars: int
    max_output_tokens: int
    repeat: int = 1

    @property
    def version_tag(self) -> str:
        """Human-readable label for logs and result files (the cache uses the full hash)."""
        return (
            f"{PROMPT_VERSION}|thinking={self.thinking}|chars={self.max_doc_chars}"
            f"|rep={self.repeat}"
        )


def request_arguments(config: ScoringConfig, user_message: str) -> dict[str, Any]:
    """The keyword arguments of the API call for one document."""
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": SCORE_JSON_SCHEMA}}
    if config.thinking == "off":
        thinking: dict[str, str] = {"type": "disabled"}
    else:
        thinking = {"type": "adaptive"}
        output_config["effort"] = config.thinking
    return {
        "model": config.model,
        "max_tokens": config.max_output_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
        "thinking": thinking,
        "output_config": output_config,
    }


def build_request(
    config: ScoringConfig, document: dict[str, Any]
) -> tuple[dict[str, Any], str, bool]:
    """Return (request, cache key, whether the text was cut) for one document.

    `document` needs the keys issuer, size, title and text.
    """
    user_message, was_cut = build_user_message(
        document["issuer"],
        document["size"],
        document["title"],
        document["text"],
        config.max_doc_chars,
    )
    request = request_arguments(config, user_message)
    return request, llm.cache_key(request, PROMPT_VERSION, config.repeat), was_cut


def parse_answer(answer: dict[str, Any] | None) -> tuple[DocumentScore | None, str]:
    """Return (validated score or None, problem description or '')."""
    if answer is None:
        return None, "no answer (request failed or expired)"
    if answer["stop_reason"] != "end_turn":
        return None, f"stop_reason={answer['stop_reason']}"
    try:
        return DocumentScore.model_validate_json(answer["text"]), ""
    except ValidationError as error:
        return None, f"invalid answer: {error.errors()[0]['msg']}"


def result_record(
    doc_id: str, config: ScoringConfig, was_cut: bool, answer: dict[str, Any] | None
) -> dict[str, Any]:
    """One flat row: bookkeeping columns plus the scores (absent if there was a problem)."""
    score, problem = parse_answer(answer)
    usage = answer or {"input_tokens": 0, "output_tokens": 0}
    return {
        "doc_id": doc_id,
        "model": config.model,
        "version_tag": config.version_tag,
        "truncated": was_cut,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "problem": problem,
        **(score.model_dump() if score else {}),
    }


def score_document(
    client: anthropic.Anthropic,
    config: ScoringConfig,
    document: dict[str, Any],
    cache_dir: Path,
    spend_log: SpendLog,
) -> dict[str, Any]:
    """Score one document with a direct call (used by the pilot)."""
    request, key, was_cut = build_request(config, document)
    answer = llm.call_once(
        client, request, key, config.version_tag, document["doc_id"], cache_dir, spend_log
    )
    record = result_record(document["doc_id"], config, was_cut, answer)
    return {**record, "from_cache": answer["from_cache"]}


def score_documents_in_batch(
    client: anthropic.Anthropic,
    config: ScoringConfig,
    documents: list[dict[str, Any]],
    cache_dir: Path,
    spend_log: SpendLog,
    state_path: Path,
) -> pd.DataFrame:
    """Score many documents through the Batches API and return one row per document."""
    built = {document["doc_id"]: build_request(config, document) for document in documents}
    items = [BatchItem(doc_id, request, key) for doc_id, (request, key, _) in built.items()]
    answers = run_batch(client, items, config.version_tag, cache_dir, spend_log, state_path)
    records = [
        result_record(doc_id, config, was_cut, answers[doc_id])
        for doc_id, (_, _, was_cut) in built.items()
    ]
    return pd.DataFrame(records)
