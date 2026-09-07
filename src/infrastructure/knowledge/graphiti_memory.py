import logging
from datetime import UTC
from itertools import zip_longest

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_RRF,
)

from src.application.ports.knowledge import KnowledgeMemoryError
from src.domain.knowledge.models import (
    KnowledgeEpisode,
    KnowledgeFact,
    KnowledgeSourceType,
)

logger = logging.getLogger(__name__)

# Graphiti reads message episodes as "speaker: text". Naming the speaker gives
# first-person facts ("я перешёл в компанию B") a subject to attach to.
MESSAGE_SPEAKER = "Пользователь"

# Extraction does not always build an edge: a detail stated in passing often
# survives only in the summary of an entity node. Searching edges alone made
# such a fact unreachable, so the search spans both layers. Episodes and
# communities stay out of it: they add queries without adding facts.
SEARCH_CONFIG = SearchConfig(
    edge_config=EDGE_HYBRID_SEARCH_RRF.edge_config,
    node_config=NODE_HYBRID_SEARCH_RRF.node_config,
)

EPISODE_TYPES: dict[KnowledgeSourceType, EpisodeType] = {
    # Conversational material keeps speaker attribution; notes are plain text.
    KnowledgeSourceType.TELEGRAM_MESSAGE: EpisodeType.message,
    KnowledgeSourceType.BUSINESS_NOTE: EpisodeType.message,
    KnowledgeSourceType.NOTE: EpisodeType.text,
}


class GraphitiKnowledgeMemory:
    """Graphiti-backed knowledge memory sharing one client per process."""

    def __init__(self, *, graphiti: Graphiti) -> None:
        self._graphiti = graphiti

    async def initialize(self) -> None:
        """Create Graphiti indices; safe to repeat on every start."""
        try:
            await self._graphiti.build_indices_and_constraints()
        except Exception as error:
            raise KnowledgeMemoryError(_reason(error)) from error

    async def remember(self, *, namespace: str, episode: KnowledgeEpisode) -> None:
        source = EPISODE_TYPES[episode.source_type]
        body = (
            f"{MESSAGE_SPEAKER}: {episode.content}"
            if source is EpisodeType.message
            else episode.content
        )
        try:
            await self._graphiti.add_episode(
                name=episode.name,
                episode_body=body,
                source=source,
                source_description=episode.source_description,
                # The Neo4j driver segfaults inside CPython's _zoneinfo when a
                # datetime carries a ZoneInfo tzinfo, so hand it plain UTC.
                # The instant is unchanged, only its representation.
                reference_time=episode.reference_time.astimezone(UTC),
                group_id=namespace,
            )
        except Exception as error:
            raise KnowledgeMemoryError(_reason(error)) from error

    async def search(
        self, *, namespace: str, query: str, limit: int
    ) -> list[KnowledgeFact]:
        try:
            results = await self._graphiti.search_(
                query=query,
                group_ids=[namespace],
                config=SEARCH_CONFIG.model_copy(update={"limit": limit}),
            )
            # Defence in depth: never surface a result from another namespace.
            edges = [edge for edge in results.edges if edge.group_id == namespace]
            nodes = [node for node in results.nodes if node.group_id == namespace]
            sources = await self._episode_names(
                namespace=namespace,
                uuids=[uuid for edge in edges for uuid in edge.episodes],
            )
        except Exception as error:
            raise KnowledgeMemoryError(_reason(error)) from error
        edge_facts = [
            KnowledgeFact(
                fact=edge.fact,
                valid_from=edge.valid_at,
                valid_until=edge.invalid_at,
                source=next(
                    (sources[uuid] for uuid in edge.episodes if uuid in sources), None
                ),
            )
            for edge in edges
        ]
        # Both endpoints of an edge repeat its fact in their summary, so the same
        # sentence arrives up to three times. Only new wording earns a slot.
        seen = {fact.fact for fact in edge_facts}
        node_facts: list[KnowledgeFact] = []
        for node in nodes:
            # An entity without a summary still names something the user mentioned.
            text = node.summary or node.name
            if not text or text in seen:
                continue
            seen.add(text)
            node_facts.append(KnowledgeFact(fact=text))
        return _interleaved(edge_facts, node_facts)[:limit]

    async def healthcheck(self) -> bool:
        try:
            # A constant, parameterless probe: it never carries user input.
            await self._graphiti.driver.execute_query("RETURN 1 AS ok", routing_="r")
        except Exception as error:
            logger.warning("knowledge.healthcheck.failed reason=%s", _reason(error))
            return False
        return True

    async def close(self) -> None:
        await self._graphiti.close()  # type: ignore[no-untyped-call]

    async def _episode_names(
        self, *, namespace: str, uuids: list[str]
    ) -> dict[str, str]:
        """Map episode uuids to their provenance names within one namespace."""
        unique = list(dict.fromkeys(uuids))
        if not unique:
            return {}
        episodes: list[EpisodicNode] = await EpisodicNode.get_by_uuids(
            self._graphiti.driver, unique
        )
        return {
            episode.uuid: episode.name
            for episode in episodes
            if episode.group_id == namespace
        }


def _interleaved(
    edge_facts: list[KnowledgeFact], node_facts: list[KnowledgeFact]
) -> list[KnowledgeFact]:
    """Alternate both layers so the limit can never starve one of them."""
    ordered: list[KnowledgeFact] = []
    for pair in zip_longest(edge_facts, node_facts):
        ordered.extend(fact for fact in pair if fact is not None)
    return ordered


def _reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
