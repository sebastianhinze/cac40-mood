"""What the LLM must return for one document, and the JSON schema sent to the API.

Two descriptions of the same thing live here on purpose:
- `SCORE_JSON_SCHEMA` is sent to the API ("structured outputs"), which then guarantees valid JSON
  with exactly these fields and one of the allowed topics;
- `DocumentScore` (Pydantic) re-checks the answer on our side, including the number ranges,
  which the API schema cannot express.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Fixed topic list (DECISIONS #30). Kept short so every topic has enough documents later.
TOPICS: tuple[str, ...] = (
    "results",  # annual, half-year or quarterly earnings
    "sales_update",  # revenue / trading update without full earnings
    "guidance",  # outlook or targets set, confirmed, raised, cut or withdrawn
    "m_and_a",  # acquisitions, disposals, mergers, takeover bids
    "financing",  # bonds, loans, capital increases, ratings
    "shareholder_returns",  # dividends, buyback programmes
    "contract_win",  # orders, contracts, partnerships, product approvals
    "strategy_operations",  # strategy plans, restructuring, plants, capital markets days
    "governance",  # board and management changes, AGM matters, remuneration
    "legal_regulatory",  # litigation, fines, investigations, regulatory decisions
    "esg",  # sustainability, climate, social commitments
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
    "strategy_operations",
    "governance",
    "legal_regulatory",
    "esg",
    "other",
]


class DocumentScore(BaseModel):
    """The validated scores of one document."""

    tone: float = Field(ge=-1.0, le=1.0)
    surprise: float = Field(ge=-1.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    topic: Topic
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


SCORE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "tone": {"type": "number", "description": "-1 (clearly bad) to 1 (clearly good)"},
        "surprise": {"type": "number", "description": "-1 (worse than expected) to 1 (better)"},
        "uncertainty": {"type": "number", "description": "0 (certain) to 1 (very uncertain)"},
        "topic": {"type": "string", "enum": list(TOPICS)},
        "confidence": {"type": "number", "description": "0 to 1"},
        "rationale": {"type": "string", "description": "At most 30 words, in English"},
    },
    "required": ["tone", "surprise", "uncertainty", "topic", "confidence", "rationale"],
    "additionalProperties": False,
}
