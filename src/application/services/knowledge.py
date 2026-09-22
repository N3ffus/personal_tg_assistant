import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from time import perf_counter

from src.application.ports.knowledge import KnowledgeMemory, KnowledgeMemoryError
from src.domain.assistant.models import SearchKnowledgeAction
from src.domain.knowledge.films import title_keys
from src.domain.knowledge.models import (
    DEFAULT_KNOWLEDGE_RESULTS,
    MAX_KNOWLEDGE_RESULTS,
    MAX_PROFILE_FACTS,
    KnowledgeEpisode,
    KnowledgeFact,
    KnowledgeSourceType,
    namespace_for,
)

logger = logging.getLogger(__name__)


async def _observed[T](
    operation: Callable[[], Awaitable[T]], *, event: str, context: str
) -> tuple[T, float]:
    started = perf_counter()
    try:
        result = await operation()
    except KnowledgeMemoryError as error:
        logger.warning("%s.failed %s reason=%s", event, context, error)
        raise
    except Exception as error:
        logger.warning("%s.failed %s reason=%s", event, context, type(error).__name__)
        raise KnowledgeMemoryError(str(error)) from error
    return result, (perf_counter() - started) * 1000


class KnowledgeService:
    """Application entry point to the long-term knowledge graph.

    The namespace is always derived here from a trusted internal user id, so no
    caller — least of all the LLM — can address another user's knowledge.
    """

    def __init__(self, *, memory: KnowledgeMemory) -> None:
        self._memory = memory

    async def remember(
        self,
        *,
        user_id: int,
        content: str,
        source_id: str,
        source_type: KnowledgeSourceType,
        reference_time: datetime,
    ) -> None:
        namespace = namespace_for(user_id)
        episode = KnowledgeEpisode(
            content=content,
            source_id=source_id,
            source_type=source_type,
            reference_time=reference_time,
        )
        # Note content stays out of the logs: only its provenance is observable.
        context = (
            f"user={user_id} group={namespace} source={episode.name} "
            f"chars={len(episode.content)}"
        )
        logger.info("knowledge.remember.started %s", context)
        _, latency_ms = await _observed(
            lambda: self._memory.remember(namespace=namespace, episode=episode),
            event="knowledge.remember",
            context=context,
        )
        logger.info(
            "knowledge.remember.completed %s latency_ms=%.0f", context, latency_ms
        )

    async def search(
        self,
        *,
        user_id: int,
        query: str,
        limit: int = DEFAULT_KNOWLEDGE_RESULTS,
    ) -> list[KnowledgeFact]:
        namespace = namespace_for(user_id)
        bounded = max(1, min(limit, MAX_KNOWLEDGE_RESULTS))
        context = f"user={user_id} group={namespace} limit={bounded}"
        logger.info("knowledge.search.started %s", context)
        facts, latency_ms = await _observed(
            lambda: self._memory.search(
                namespace=namespace, query=query, limit=bounded
            ),
            event="knowledge.search",
            context=context,
        )
        logger.info(
            "knowledge.search.completed %s results=%s latency_ms=%.0f",
            context,
            len(facts),
            latency_ms,
        )
        return facts

    async def forget_candidates(
        self, *, user_id: int, query: str, limit: int = MAX_KNOWLEDGE_RESULTS
    ) -> list[KnowledgeFact]:
        """Recall what a forget request may cover, each fact with its ``ref``."""
        namespace = namespace_for(user_id)
        facts, _ = await _observed(
            lambda: self._memory.forget_candidates(
                namespace=namespace,
                query=query,
                limit=max(1, min(limit, MAX_KNOWLEDGE_RESULTS)),
            ),
            event="knowledge.forget_candidates",
            context=f"user={user_id} group={namespace}",
        )
        return facts

    async def forget(self, *, user_id: int, refs: list[str]) -> list[str]:
        """Erase the referenced facts; returns the wording that was erased."""
        namespace = namespace_for(user_id)
        if not refs:
            return []
        context = f"user={user_id} group={namespace} refs={len(refs)}"
        forgotten, latency_ms = await _observed(
            lambda: self._memory.forget(namespace=namespace, refs=refs),
            event="knowledge.forget",
            context=context,
        )
        # The erased wording stays out of the logs, like remembered content.
        logger.info(
            "knowledge.forget.completed %s forgotten=%s latency_ms=%.0f",
            context,
            len(forgotten),
            latency_ms,
        )
        return forgotten

    async def healthcheck(self) -> bool:
        try:
            return await self._memory.healthcheck()
        except Exception as error:
            logger.warning("knowledge.healthcheck.failed reason=%s", error)
            return False

    async def recent_watched_films(
        self, *, user_id: int, limit: int = DEFAULT_KNOWLEDGE_RESULTS
    ) -> list[KnowledgeFact]:
        return await self._memory.recent_watched_films(
            namespace=namespace_for(user_id),
            limit=max(1, min(limit, MAX_KNOWLEDGE_RESULTS)),
        )

    async def close(self) -> None:
        await self._memory.close()

    async def profile_facts(self, *, user_id: int) -> list[KnowledgeFact]:
        """Every statement the user made, oldest first, films aside.

        Semantic top-k cannot answer «все мои поездки», «что изменилось» or
        «назови 20 фактов обо мне»: whatever falls below the cut looks unknown.
        """
        return await self._memory.profile_facts(
            namespace=namespace_for(user_id), limit=MAX_PROFILE_FACTS
        )

    async def lookup(
        self, search: SearchKnowledgeAction, *, user_id: int
    ) -> list[KnowledgeFact]:
        """Run one search_knowledge action in whichever mode the model chose."""
        if search.mode == "watched_film_catalogue":
            return await self.watched_film_catalogue(user_id=user_id)
        if search.mode == "profile":
            return await self.profile_facts(user_id=user_id)
        if search.mode == "recent_watched_films":
            return await self.recent_watched_films(user_id=user_id, limit=search.limit)
        return await self.search(
            user_id=user_id, query=search.query, limit=search.limit
        )

    async def watched_film_titles(self, *, user_id: int) -> list[str]:
        return await self._memory.watched_film_titles(namespace=namespace_for(user_id))

    async def watched_title_keys(self, *, user_id: int) -> set[str]:
        """Every spelling key of every watched film, for catalogue exclusion."""
        return {
            key
            for title in await self.watched_film_titles(user_id=user_id)
            for key in title_keys(title)
        }

    async def watched_film_catalogue(self, *, user_id: int) -> list[KnowledgeFact]:
        titles = await self.watched_film_titles(user_id=user_id)
        return (
            [
                KnowledgeFact(
                    fact="Полный каталог просмотренных фильмов для исключения из рекомендаций:\n"
                    + "\n".join(titles)
                )
            ]
            if titles
            else []
        )
