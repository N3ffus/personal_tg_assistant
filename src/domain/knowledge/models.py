from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

KnowledgeText = Annotated[str, Field(min_length=1, max_length=4000)]
MAX_KNOWLEDGE_RESULTS = 20
DEFAULT_KNOWLEDGE_RESULTS = 5

# A fixed seed keeps the derived namespace stable across restarts while keeping
# raw Telegram identifiers out of the knowledge graph.
_NAMESPACE_SEED = uuid5(NAMESPACE_URL, "https://personal-ai-assistant/knowledge")


class KnowledgeSourceType(StrEnum):
    """Where the raw material of an episode came from."""

    TELEGRAM_MESSAGE = "telegram_message"
    NOTE = "note"
    BUSINESS_NOTE = "business_note"


SOURCE_DESCRIPTIONS: dict[KnowledgeSourceType, str] = {
    KnowledgeSourceType.TELEGRAM_MESSAGE: "telegram chat message",
    KnowledgeSourceType.NOTE: "assistant note saved by the owner",
    KnowledgeSourceType.BUSINESS_NOTE: "note extracted from a business dialog",
}


def namespace_for(user_id: int) -> str:
    """Return the Graphiti ``group_id`` isolating one user's knowledge.

    The caller must pass a trusted internal user id; the value never comes from
    the LLM.
    """
    if user_id <= 0:
        raise ValueError("user_id must be a positive identifier")
    # Graphiti only accepts alphanumerics, dashes and underscores in a group id.
    return f"user_{uuid5(_NAMESPACE_SEED, str(user_id))}"


class KnowledgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class KnowledgeEpisode(KnowledgeModel):
    """Raw user-authored material handed to Graphiti for extraction."""

    content: KnowledgeText
    source_id: str = Field(min_length=1, max_length=200)
    source_type: KnowledgeSourceType
    reference_time: AwareDatetime

    @property
    def name(self) -> str:
        return f"{self.source_type.value}:{self.source_id}"

    @property
    def source_description(self) -> str:
        return SOURCE_DESCRIPTIONS[self.source_type]


class KnowledgeFact(KnowledgeModel):
    """A temporal fact derived by Graphiti, trimmed down for the agent."""

    fact: str
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    source: str | None = None

    def as_payload(self) -> dict[str, str | None]:
        return {
            "fact": self.fact,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "source": self.source,
        }
