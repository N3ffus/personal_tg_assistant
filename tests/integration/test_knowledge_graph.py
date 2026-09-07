"""Live checks against a real Neo4j and a real LLM provider.

Run them with ``make test-integration`` after ``make neo4j-up``. They stay out of
the default suite because they cost LLM calls and take minutes.
"""

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodicNode

from src.application.ports.knowledge import KnowledgeMemoryError
from src.application.services.knowledge import KnowledgeService
from src.config import Settings
from src.domain.knowledge.models import KnowledgeSourceType, namespace_for
from src.infrastructure.knowledge.factory import build_graphiti
from src.infrastructure.knowledge.graphiti_memory import GraphitiKnowledgeMemory

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
]

# Disposable identities so a developer's real memory stays untouched.
USER_A = 900_000_101
USER_B = 900_000_102

WORKED_AT_A = datetime(2025, 3, 1, 9, 0, tzinfo=UTC)
MOVED_TO_B = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
WANTED_TO_WATCH = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)

SEED = [
    (USER_A, "int-a-1", "Я работаю в компании A.", WORKED_AT_A),
    (USER_A, "int-a-2", "Я перешёл из компании A в компанию B.", MOVED_TO_B),
    (USER_A, "int-a-3", "Я хочу посмотреть фильм Blade Runner.", WANTED_TO_WATCH),
    (USER_B, "int-b-1", "Я работаю в компании Contoso.", WORKED_AT_A),
]


@pytest.fixture(scope="module")
def settings() -> Settings:
    configured = Settings()  # type: ignore[call-arg]
    if not configured.knowledge_memory_enabled:
        pytest.skip("KNOWLEDGE_MEMORY_ENABLED is not set")
    return configured


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def knowledge(settings: Settings) -> AsyncIterator[KnowledgeService]:
    memory = GraphitiKnowledgeMemory(graphiti=build_graphiti(settings))
    await memory.initialize()
    service = KnowledgeService(memory=memory)
    for user_id, source_id, content, reference_time in SEED:
        await service.remember(
            user_id=user_id,
            content=content,
            source_id=source_id,
            source_type=KnowledgeSourceType.TELEGRAM_MESSAGE,
            reference_time=reference_time,
        )
    try:
        yield service
    finally:
        await service.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def graph(settings: Settings) -> AsyncIterator[Graphiti]:
    """A raw client used only to inspect and clean up what the memory wrote."""
    client = build_graphiti(settings)
    try:
        yield client
    finally:
        await client.close()  # type: ignore[no-untyped-call]


async def test_graphiti_connects_to_neo4j(knowledge: KnowledgeService) -> None:
    assert await knowledge.healthcheck() is True


async def test_episodes_keep_their_provenance(
    knowledge: KnowledgeService, graph: Graphiti
) -> None:
    episodes = await EpisodicNode.get_by_group_ids(
        graph.driver, [namespace_for(USER_A)]
    )

    names = {episode.name for episode in episodes}
    assert {"telegram_message:int-a-1", "telegram_message:int-a-3"} <= names
    assert {episode.source_description for episode in episodes} == {
        "telegram chat message"
    }
    # reference_time is preserved instead of being replaced by the ingestion time.
    assert WORKED_AT_A in {episode.valid_at for episode in episodes}


async def test_remembering_extracts_graph_data(knowledge: KnowledgeService) -> None:
    facts = await knowledge.search(user_id=USER_A, query="компания", limit=10)

    assert facts, "Graphiti extracted no entity edges"
    assert all(fact.source is not None for fact in facts)


async def test_search_recalls_a_remembered_fact(knowledge: KnowledgeService) -> None:
    facts = await knowledge.search(
        user_id=USER_A, query="Какой фильм я хотел посмотреть?", limit=10
    )

    assert any("Blade Runner" in fact.fact for fact in facts)


async def test_temporal_facts_reflect_the_latest_change(
    knowledge: KnowledgeService,
) -> None:
    facts = await knowledge.search(user_id=USER_A, query="Где я работаю?", limit=10)

    current = [fact for fact in facts if fact.valid_until is None]
    assert any("B" in fact.fact for fact in current)
    # History is kept: the previous employer is still stored, not deleted.
    assert any("A" in fact.fact for fact in facts)


async def test_users_never_see_each_others_memory(
    knowledge: KnowledgeService,
) -> None:
    for_a = await knowledge.search(user_id=USER_A, query="компания", limit=20)
    for_b = await knowledge.search(user_id=USER_B, query="компания", limit=20)

    assert not any("Contoso" in fact.fact for fact in for_a)
    assert any("Contoso" in fact.fact for fact in for_b)
    assert not any("Blade Runner" in fact.fact for fact in for_b)


async def test_memory_survives_a_client_restart(settings: Settings) -> None:
    restarted = GraphitiKnowledgeMemory(graphiti=build_graphiti(settings))
    service = KnowledgeService(memory=restarted)
    try:
        facts = await service.search(
            user_id=USER_A, query="Какой фильм я хотел посмотреть?", limit=10
        )
    finally:
        await service.close()

    assert any("Blade Runner" in fact.fact for fact in facts)


async def test_unreachable_neo4j_fails_in_a_controlled_way(
    settings: Settings,
) -> None:
    offline = GraphitiKnowledgeMemory(
        graphiti=build_graphiti(
            settings.model_copy(update={"neo4j_uri": "bolt://127.0.0.1:1"})
        )
    )
    service = KnowledgeService(memory=offline)
    try:
        assert await service.healthcheck() is False
        with pytest.raises(KnowledgeMemoryError):
            await service.search(user_id=USER_A, query="что угодно", limit=1)
    finally:
        await service.close()


async def test_cleanup_removes_the_integration_namespaces(graph: Graphiti) -> None:
    if os.environ.get("KNOWLEDGE_KEEP_TEST_DATA"):
        pytest.skip("KNOWLEDGE_KEEP_TEST_DATA keeps the graph for manual inspection")
    for user_id in (USER_A, USER_B):
        await graph.driver.execute_query(
            "MATCH (n) WHERE n.group_id = $group_id DETACH DELETE n",
            group_id=namespace_for(user_id),
        )

    remaining = await EpisodicNode.get_by_group_ids(
        graph.driver, [namespace_for(USER_A)]
    )
    assert remaining == []
