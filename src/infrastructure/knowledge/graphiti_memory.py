import asyncio
import logging
from collections import OrderedDict
from datetime import UTC, datetime
from itertools import zip_longest
from time import monotonic, perf_counter
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search import search as graphiti_search
from graphiti_core.search.search_config import (
    EdgeSearchMethod,
    NodeSearchMethod,
    SearchConfig,
)
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters

from src.application.ports.knowledge import KnowledgeMemoryError
from src.domain.knowledge.models import (
    KnowledgeEpisode,
    KnowledgeFact,
    KnowledgeSourceType,
)
from src.infrastructure.knowledge.cypher import (
    DROP_ORPHANS,
    EPISODE_SIMILARITY,
    FORGET_EDGES,
    FORGET_EPISODES,
    PROFILE_EPISODES,
    READ_EDGE_FACTS,
    READ_EPISODE_CONTENTS,
    READ_SUMMARIES,
    RECENT_WATCHED_FILMS,
    SET_EPISODE_EMBEDDING,
    SET_SUMMARY,
    WATCHED_FILM_CONTENTS,
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

# Unrelated Russian sentences sit well below this; paraphrases sit above it.
MIN_EPISODE_SIMILARITY = 0.5
QUERY_EMBEDDING_CACHE_SIZE = 128
QUERY_EMBEDDING_TTL_SECONDS = 300

# Handles the model gets for facts it may ask to forget. Every erasing query
# still matches the namespace, so a ref from another user erases nothing.
EDGE_REF = "edge:"
NODE_REF = "node:"
EPISODE_REF = "episode:"

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
        # Cache vectors only. Facts are always read afresh, including after writes
        # or imports, and every graph query still carries the user's namespace.
        self._query_embeddings: OrderedDict[str, tuple[float, list[float]]] = (
            OrderedDict()
        )

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
            results = await self._graphiti.add_episode(
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
        await self._embed_episode(uuid=results.episode.uuid, content=body)

    async def _embed_episode(self, *, uuid: str, content: str) -> None:
        """Attach the vector that makes this episode findable by meaning.

        The episode is already stored, so a failing embedder degrades recall to
        keyword search instead of losing the fact.
        """
        try:
            embedding = await self._graphiti.embedder.create(input_data=content)
            await self._graphiti.driver.execute_query(
                SET_EPISODE_EMBEDDING,
                uuid=uuid,
                embedding=embedding,
            )
        except Exception as error:
            logger.warning(
                "knowledge.embed.failed uuid=%s reason=%s", uuid, _reason(error)
            )

    async def search(
        self, *, namespace: str, query: str, limit: int
    ) -> list[KnowledgeFact]:
        facts = await self.forget_candidates(
            namespace=namespace, query=query, limit=limit
        )
        # A ref is a handle for erasing; an ordinary answer has no use for it.
        return [fact.model_copy(update={"ref": None}) for fact in facts]

    async def forget_candidates(
        self, *, namespace: str, query: str, limit: int
    ) -> list[KnowledgeFact]:
        if not query.strip():
            return []
        started = perf_counter()
        embedding = await self._query_embedding(query)
        embedded = perf_counter()
        try:
            layers, similar = await asyncio.gather(
                self._search_layers(
                    namespace=namespace, query=query, limit=limit, embedding=embedding
                ),
                self._similar_episodes(
                    namespace=namespace, embedding=embedding, limit=limit
                ),
                return_exceptions=True,
            )
            # Both reads have finished before an error can escape. Cancellation
            # of the caller also cancels gather's children.
            if isinstance(layers, BaseException):
                raise layers
            if isinstance(similar, BaseException):
                raise similar
        except Exception as error:
            raise KnowledgeMemoryError(_reason(error)) from error
        seen = {fact.fact for layer in layers for fact in layer}
        unique_similar = []
        for fact in similar:
            if fact.fact not in seen:
                seen.add(fact.fact)
                unique_similar.append(fact)
        logger.info(
            "knowledge.search.timing embedding_ms=%.0f retrieval_ms=%.0f",
            (embedded - started) * 1000,
            (perf_counter() - embedded) * 1000,
        )
        return _interleaved(*layers, unique_similar)[:limit]

    async def _query_embedding(self, query: str) -> list[float] | None:
        query = query.replace("\n", " ").strip()
        cached = self._query_embeddings.get(query)
        if cached is not None and monotonic() - cached[0] < QUERY_EMBEDDING_TTL_SECONDS:
            self._query_embeddings.move_to_end(query)
            return cached[1]
        try:
            embedding = await self._graphiti.embedder.create(input_data=query)
        except Exception as error:
            logger.warning("knowledge.embed.failed query reason=%s", _reason(error))
            return None
        self._query_embeddings[query] = (monotonic(), embedding)
        self._query_embeddings.move_to_end(query)
        while len(self._query_embeddings) > QUERY_EMBEDDING_CACHE_SIZE:
            self._query_embeddings.popitem(last=False)
        return embedding

    async def _search_layers(
        self,
        *,
        namespace: str,
        query: str,
        limit: int,
        embedding: list[float] | None,
    ) -> tuple[list[KnowledgeFact], list[KnowledgeFact], list[KnowledgeFact]]:
        config = SEARCH_CONFIG.model_copy(deep=True, update={"limit": limit})
        if embedding is None:
            # A failed embedder must not retry inside Graphiti and lose keyword
            # results. Keep all three layers, using BM25 on this request only.
            if config.edge_config is not None:
                config.edge_config.search_methods = [EdgeSearchMethod.bm25]
            if config.node_config is not None:
                config.node_config.search_methods = [NodeSearchMethod.bm25]
        try:
            # Graphiti.search_ doesn't expose query_vector; its underlying search
            # does. Share this request's vector with the episode similarity query.
            results = await graphiti_search(
                clients=self._graphiti.clients,
                query=query,
                group_ids=[namespace],
                config=config,
                search_filter=SearchFilters(),
                query_vector=embedding,
                driver=self._graphiti.driver,
            )
            # Defence in depth: never surface a result from another namespace.
            edges = [edge for edge in results.edges if edge.group_id == namespace]
            nodes = [node for node in results.nodes if node.group_id == namespace]
            episodes = [
                episode for episode in results.episodes if episode.group_id == namespace
            ]
            sources = await self._episode_sources(
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
                stated_at=_latest(sources, edge.episodes),
                source=next(
                    (sources[uuid][0] for uuid in edge.episodes if uuid in sources),
                    None,
                ),
                ref=f"{EDGE_REF}{edge.uuid}",
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
            node_facts.append(KnowledgeFact(fact=text, ref=f"{NODE_REF}{node.uuid}"))
        episode_facts: list[KnowledgeFact] = []
        for episode in episodes:
            text = _without_speaker(episode.content)
            if not text or text in seen:
                continue
            seen.add(text)
            episode_facts.append(
                KnowledgeFact(
                    fact=text,
                    valid_from=episode.valid_at,
                    stated_at=episode.valid_at,
                    source=episode.name,
                    ref=f"{EPISODE_REF}{episode.uuid}",
                )
            )
        return edge_facts, node_facts, episode_facts

    async def _similar_episodes(
        self, *, namespace: str, embedding: list[float] | None, limit: int
    ) -> list[KnowledgeFact]:
        """Recall episodes whose meaning matches, not whose words do."""
        if embedding is None:
            return []
        try:
            rows, _, _ = await self._graphiti.driver.execute_query(
                EPISODE_SIMILARITY,
                namespace=namespace,
                embedding=embedding,
                min_score=MIN_EPISODE_SIMILARITY,
                limit=limit,
                routing_="r",
            )
        except Exception as error:
            # The other layers can still answer if episode similarity is down.
            logger.warning("knowledge.embed.failed query reason=%s", _reason(error))
            return []
        facts: list[KnowledgeFact] = []
        seen: set[str] = set()
        for row in rows:
            text = _without_speaker(row["content"])
            if not text or text in seen:
                continue
            seen.add(text)
            facts.append(
                KnowledgeFact(
                    fact=text,
                    valid_from=_as_datetime(row["valid_at"]),
                    stated_at=_as_datetime(row["valid_at"]),
                    source=row["name"],
                    ref=f"{EPISODE_REF}{row['uuid']}",
                )
            )
        return facts

    async def forget(self, *, namespace: str, refs: list[str]) -> list[str]:
        """Erase the referenced facts and every copy of their wording.

        One fact lives in up to three places: an edge, the episode that stated
        it and the summaries of the entities it joins. Erasing only the
        selected element would let recall find the same sentence in another
        layer, so the wording of whatever was selected goes everywhere in this
        namespace. The statements run one by one: a failure midway leaves less
        of the fact behind, never more.
        """
        edges = _refs(refs, EDGE_REF)
        nodes = set(_refs(refs, NODE_REF))
        episodes = _refs(refs, EPISODE_REF)
        try:
            texts = [
                row["text"]
                for row in await self._read(READ_EDGE_FACTS, namespace, uuids=edges)
            ] + [
                _without_speaker(row["content"])
                for row in await self._read(
                    READ_EPISODE_CONTENTS, namespace, uuids=episodes
                )
            ]
            wording = sorted({text for text in texts if text})
            deleted_edges = await self._write(
                FORGET_EDGES, namespace, uuids=edges, texts=wording
            )
            deleted_episodes = await self._write(
                FORGET_EPISODES,
                namespace,
                uuids=episodes,
                contents=[
                    *wording,
                    *(f"{MESSAGE_SPEAKER}: {text}" for text in wording),
                ],
            )
            touched = {row["source"] for row in deleted_edges} | {
                row["target"] for row in deleted_edges
            }
            forgotten = [row["text"] for row in deleted_edges] + [
                _without_speaker(row["content"]) for row in deleted_episodes
            ]
            for row in await self._read(READ_SUMMARIES, namespace):
                summary, erased = _erased(
                    row["summary"], wording=wording, whole=row["uuid"] in nodes
                )
                if not erased:
                    continue
                await self._write(
                    SET_SUMMARY, namespace, uuid=row["uuid"], summary=summary
                )
                touched.add(row["uuid"])
                forgotten.extend(erased)
            await self._write(DROP_ORPHANS, namespace, uuids=sorted(touched))
        except Exception as error:
            raise KnowledgeMemoryError(_reason(error)) from error
        return list(dict.fromkeys(text for text in forgotten if text))

    async def _read(self, query: str, namespace: str, **parameters: Any) -> list[Any]:
        rows, _, _ = await self._graphiti.driver.execute_query(
            query, namespace=namespace, routing_="r", **parameters
        )
        return list(rows)

    async def _write(self, query: str, namespace: str, **parameters: Any) -> list[Any]:
        rows, _, _ = await self._graphiti.driver.execute_query(
            query, namespace=namespace, **parameters
        )
        return list(rows)

    async def healthcheck(self) -> bool:
        try:
            # A constant, parameterless probe: it never carries user input.
            await self._graphiti.driver.execute_query("RETURN 1 AS ok", routing_="r")
        except Exception as error:
            logger.warning("knowledge.healthcheck.failed reason=%s", _reason(error))
            return False
        return True

    async def recent_watched_films(
        self, *, namespace: str, limit: int
    ) -> list[KnowledgeFact]:
        # A catalogue query must rank the entire dated collection, not a small
        # semantic shortlist. Missing watch dates never mean "watched today".
        try:
            rows, _, _ = await self._graphiti.driver.execute_query(
                RECENT_WATCHED_FILMS,
                namespace=namespace,
                limit=limit,
                routing_="r",
            )
            return [
                KnowledgeFact(
                    fact=row["content"],
                    source=row["name"],
                    valid_from=_as_datetime(row["watched_at"]),
                )
                for row in rows
            ]
        except Exception as error:
            raise KnowledgeMemoryError(_reason(error)) from error

    async def close(self) -> None:
        await self._graphiti.close()  # type: ignore[no-untyped-call]

    async def profile_facts(self, *, namespace: str, limit: int) -> list[KnowledgeFact]:
        try:
            rows, _, _ = await self._graphiti.driver.execute_query(
                PROFILE_EPISODES, namespace=namespace, limit=limit, routing_="r"
            )
        except Exception as error:
            raise KnowledgeMemoryError(_reason(error)) from error
        facts: list[KnowledgeFact] = []
        seen: set[str] = set()
        for row in reversed(rows):
            text = _without_speaker(row["content"] or "")
            if not text or text in seen:
                continue
            seen.add(text)
            facts.append(
                KnowledgeFact(
                    fact=text,
                    valid_from=_as_datetime(row["valid_at"]),
                    stated_at=_as_datetime(row["valid_at"]),
                    source=row["name"],
                )
            )
        return facts

    async def watched_film_titles(self, *, namespace: str) -> list[str]:
        try:
            rows, _, _ = await self._graphiti.driver.execute_query(
                WATCHED_FILM_CONTENTS,
                namespace=namespace,
                routing_="r",
            )
            prefix = "Просмотренный фильм: "
            return list(
                dict.fromkeys(
                    row["content"].removeprefix(prefix).split(";", 1)[0].strip()
                    for row in rows
                    if row["content"].startswith(prefix)
                )
            )
        except Exception as error:
            raise KnowledgeMemoryError(_reason(error)) from error

    async def _episode_sources(
        self, *, namespace: str, uuids: list[str]
    ) -> dict[str, tuple[str, datetime | None]]:
        """Map episode uuids to their name and time within one namespace."""
        unique = list(dict.fromkeys(uuids))
        if not unique:
            return {}
        episodes: list[EpisodicNode] = await EpisodicNode.get_by_uuids(
            self._graphiti.driver, unique
        )
        return {
            episode.uuid: (episode.name, episode.valid_at)
            for episode in episodes
            if episode.group_id == namespace
        }


def _latest(
    sources: dict[str, tuple[str, datetime | None]], uuids: list[str]
) -> datetime | None:
    """When the fact was last stated, over the episodes an edge came from."""
    stated = [
        moment
        for uuid in uuids
        if (moment := sources.get(uuid, ("", None))[1]) is not None
    ]
    return max(stated) if stated else None


def _as_datetime(value: Any) -> datetime | None:
    """Convert the driver's own temporal type; Graphiti's models do it for us."""
    if isinstance(value, datetime):
        return value
    to_native = getattr(value, "to_native", None)
    converted = to_native() if callable(to_native) else None
    return converted if isinstance(converted, datetime) else None


def _refs(refs: list[str], prefix: str) -> list[str]:
    return sorted({ref.removeprefix(prefix) for ref in refs if ref.startswith(prefix)})


def _erased(summary: str, *, wording: list[str], whole: bool) -> tuple[str, list[str]]:
    """Drop the summary lines that repeat forgotten wording.

    A summary the model selected itself goes entirely when no line matches:
    Graphiti may have paraphrased the fact into prose.
    """
    forgotten = {text.casefold() for text in wording}
    kept: list[str] = []
    erased: list[str] = []
    for line in summary.splitlines():
        (erased if line.strip().casefold() in forgotten else kept).append(line)
    if whole and not erased:
        return "", [summary]
    return "\n".join(kept), [line.strip() for line in erased]


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
