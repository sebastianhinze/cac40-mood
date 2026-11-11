"""Score one document: cache lookup, spend-cap check, one API call, validation, logging.

Order of operations for every document:
1. build the exact message that would be sent;
2. look it up in the disk cache (key = sha256 of model + version tag + message) and return the
   stored answer if it exists, so the same document is never paid for twice;
3. otherwise check the spend cap against the request's worst-case cost;
4. call the API with a JSON schema, validate the answer, log tokens and cost, store it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from pydantic import ValidationError

from cac40_mood.scoring.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from cac40_mood.scoring.schema import SCORE_JSON_SCHEMA, DocumentScore
from cac40_mood.scoring.spend import SpendLog, worst_case_cost_usd


@dataclass(frozen=True)
class ScoringConfig:
    """Everything that defines "how" a document is scored. All of it goes into the cache key.

    - `thinking`: "off" (no reasoning before the answer) or an effort level such as "low" or
      "medium" (the model may reason first; that reasoning is billed as output tokens).
    - `max_doc_chars`: where the document text is cut.
    - `outlook_chars`: if > 0, also send this much of the outlook section found after the cut.
    - `repeat`: 1 for normal runs. The pilot scores a document a second time with repeat=2 to
      measure how much the scores move between two identical requests (the model cannot be
      made deterministic: this model family does not accept a temperature setting).
    """

    model: str
    thinking: str
    max_doc_chars: int
    max_output_tokens: int
    repeat: int = 1
    outlook_chars: int = 0

    @property
    def version_tag(self) -> str:
        tag = f"{PROMPT_VERSION}|thinking={self.thinking}|chars={self.max_doc_chars}"
        if self.outlook_chars:
            tag += f"|outlook={self.outlook_chars}"
        return f"{tag}|rep={self.repeat}"


def cache_key(model: str, version_tag: str, user_message: str) -> str:
    """sha256 over model, version tag and the exact text sent (CLAUDE.md cache rule)."""
    joined = "\n".join([model, version_tag, user_message])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    """Two-character sub-folders keep any one folder from holding thousands of files."""
    return cache_dir / key[:2] / f"{key}.json"


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


def parse_response(response: anthropic.types.Message) -> tuple[DocumentScore | None, str]:
    """Return (validated score or None, problem description or '')."""
    if response.stop_reason != "end_turn":
        return None, f"stop_reason={response.stop_reason}"
    text = next((block.text for block in response.content if block.type == "text"), "")
    try:
        return DocumentScore.model_validate_json(text), ""
    except ValidationError as error:
        return None, f"invalid answer: {error.errors()[0]['msg']}"


def score_document(
    client: anthropic.Anthropic,
    config: ScoringConfig,
    doc_id: str,
    title: str,
    text: str,
    cache_dir: Path,
    spend_log: SpendLog,
) -> dict[str, Any]:
    """Score one document (from cache if possible) and return a flat result record."""
    user_message, was_cut = build_user_message(
        title, text, config.max_doc_chars, config.outlook_chars
    )
    key = cache_key(config.model, config.version_tag, user_message)
    path = cache_path(cache_dir, key)
    if path.exists():
        return {**json.loads(path.read_text()), "from_cache": True}

    worst_case = worst_case_cost_usd(
        config.model, len(SYSTEM_PROMPT) + len(user_message), config.max_output_tokens
    )
    spend_log.check_can_spend(worst_case)
    response = client.messages.create(**request_arguments(config, user_message))
    score, problem = parse_response(response)
    cost = spend_log.record(
        config.model,
        config.version_tag,
        doc_id,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
    record: dict[str, Any] = {
        "doc_id": doc_id,
        "model": config.model,
        "version_tag": config.version_tag,
        "truncated": was_cut,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": cost,
        "problem": problem,
        **(score.model_dump() if score else {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return {**record, "from_cache": False}
