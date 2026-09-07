import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from time import perf_counter

from src.application.ports.knowledge import KnowledgeMemory, KnowledgeMemoryError
from src.domain.knowledge.models import (
    DEFAULT_KNOWLEDGE_RESULTS,
    MAX_KNOWLEDGE_RESULTS,
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

    async def healthcheck(self) -> bool:
        try:
            return await self._memory.healthcheck()
        except Exception as error:
            logger.warning("knowledge.healthcheck.failed reason=%s", error)
            return False

    async def close(self) -> None:
        await self._memory.close()
