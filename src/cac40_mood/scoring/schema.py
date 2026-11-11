"""What the LLM must return for one document, and the JSON schema sent to the API.

Two descriptions of the same thing live here on purpose:
- `SCORE_JSON_SCHEMA` is sent to the API ("structured outputs"), which then guarantees valid JSON
  with exactly these fields and one of the allowed topics;
- `DocumentScore` (Pydantic) re-checks the answer on our side, including the number ranges,
  which the API schema cannot express.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Fixed topic list (DECISIONS #30, extended in #36). Kept short so every topic has enough
# documents later. The wording the model sees for each topic is in prompt.py.
TOPICS: tuple[str, ...] = (
    "results",
    "sales_update",
    "guidance",
    "m_and_a",
    "financing",
    "shareholder_returns",
    "contract_win",
    "product_pipeline",
    "strategy_operations",
    "governance",
    "legal_regulatory",
    "esg",
    "other",
)

Topic = Literal[
    "results",
    "sales_update",
    "guidance",
    "m_and_a",
    "financing",
    "shareholder_returns",
    "contract_win",
    "product_pipeline",
    "strategy_operations",
    "governance",
    "legal_regulatory",
    "esg",
    "other",
]

# The model occasionally leaks a closing brace, bracket, quote or a tag-like fragment such as
# "</something>" into the end of the last text field. Those characters are never part of a
# sentence, so they are removed from the end of the rationale.
TRAILING_JUNK = re.compile(r"(\s|[}\]\"'`]|</?[A-Za-z_][A-Za-z0-9_]*>)+$")


def clean_rationale(text: str) -> str:
    """Strip stray closing characters and tags from the end of the rationale."""
    return TRAILING_JUNK.sub("", text).strip()


class DocumentScore(BaseModel):
    """The validated scores of one document."""

    tone: float = Field(ge=-1.0, le=1.0)
    surprise: float = Field(ge=-1.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    topic: Topic
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _clean_rationale(cls, value: str) -> str:
        return clean_rationale(value)


SCORE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "tone": {"type": "number", "description": "-1 (clearly bad) to 1 (clearly good)"},
        "surprise": {"type": "number", "description": "-1 (worse than expected) to 1 (better)"},
        "uncertainty": {"type": "number", "description": "0 (certain) to 1 (very uncertain)"},
        "topic": {"type": "string", "enum": list(TOPICS)},
        "confidence": {"type": "number", "description": "0 to 1"},
        "rationale": {"type": "string", "description": "One plain sentence, at most 30 words"},
    },
    "required": ["tone", "surprise", "uncertainty", "topic", "confidence", "rationale"],
    "additionalProperties": False,
}
