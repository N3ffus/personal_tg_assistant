import logging
from datetime import UTC, datetime
from itertools import zip_longest
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_RRF,
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

# Extraction is lossy in two steps. A detail stated in passing often survives
# only in the summary of an entity node, and sometimes not even there: in
# production a birth date the user asked to remember left no edge and no
# summary, while the episode still held the sentence verbatim. The search
# therefore spans all three layers, ending at the user's own words.
# Communities stay out: they add a query without adding facts.
SEARCH_CONFIG = SearchConfig(
    edge_config=EDGE_HYBRID_SEARCH_RRF.edge_config,
    node_config=NODE_HYBRID_SEARCH_RRF.node_config,
    episode_config=COMBINED_HYBRID_SEARCH_RRF.episode_config,
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
            episodes = [
                episode for episode in results.episodes if episode.group_id == namespace
            ]
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
        # Both endpoints of an edge repeat its fact in their summary, and the
        # episode may repeat it once more. Only new wording earns a slot.
        seen = {fact.fact for fact in edge_facts}
        node_facts: list[KnowledgeFact] = []
        for node in nodes:
            # An entity without a summary still names something the user mentioned.
            text = node.summary or node.name
            if not text or text in seen:
                continue
            seen.add(text)
            node_facts.append(KnowledgeFact(fact=text))
        episode_facts: list[KnowledgeFact] = []
        for episode in episodes:
            text = _without_speaker(episode.content)
            if not text or text in seen:
                continue
            seen.add(text)
            episode_facts.append(
                KnowledgeFact(
                    fact=text, valid_from=episode.valid_at, source=episode.name
                )
            )
        facts = _interleaved(edge_facts, node_facts, episode_facts)[:limit]
        if len(facts) < limit:
            facts += await self._recent_episodes(
                namespace=namespace, limit=limit - len(facts), seen=seen
            )
        return facts

    async def _recent_episodes(
        self, *, namespace: str, limit: int, seen: set[str]
    ) -> list[KnowledgeFact]:
        """Fill the leftover slots with what the user most recently said.

        Episode search is BM25 only, so "дата рождения" never matches "родился"
        and a fact extraction dropped stays hidden behind its own wording. A
        recall that did not fill its limit has room for the raw material; a
        recall that did never reaches this query.
        """
        try:
            rows, _, _ = await self._graphiti.driver.execute_query(
                "MATCH (episode:Episodic {group_id: $namespace}) "
                "RETURN episode.name AS name, episode.content AS content, "
                "episode.valid_at AS valid_at "
                "ORDER BY episode.valid_at DESC LIMIT $limit",
                namespace=namespace,
                limit=limit + len(seen),
                routing_="r",
            )
        except Exception as error:
            raise KnowledgeMemoryError(_reason(error)) from error
        facts: list[KnowledgeFact] = []
        for row in rows:
            text = _without_speaker(row["content"])
            if not text or text in seen:
                continue
            seen.add(text)
            facts.append(
                KnowledgeFact(
                    fact=text,
                    valid_from=_as_datetime(row["valid_at"]),
                    source=row["name"],
                )
            )
            if len(facts) == limit:
                break
        return facts

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


def _as_datetime(value: Any) -> datetime | None:
    """Convert the driver's own temporal type; Graphiti's models do it for us.

    A hand-written Cypher query returns ``neo4j.time.DateTime``, which the
    strict domain field rejects — that turned every top-up into a degraded
    search until this conversion existed.
    """
    if isinstance(value, datetime):
        return value
    to_native = getattr(value, "to_native", None)
    converted = to_native() if callable(to_native) else None
    return converted if isinstance(converted, datetime) else None


def _without_speaker(content: str) -> str:
    """Drop the ``"Пользователь: "`` prefix ``remember`` adds for extraction."""
    prefix = f"{MESSAGE_SPEAKER}: "
    return content.removeprefix(prefix).strip() if content else ""


def _interleaved(*layers: list[KnowledgeFact]) -> list[KnowledgeFact]:
    """Alternate the layers so the limit can never starve one of them."""
    ordered: list[KnowledgeFact] = []
    for group in zip_longest(*layers):
        ordered.extend(fact for fact in group if fact is not None)
    return ordered


def _reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
