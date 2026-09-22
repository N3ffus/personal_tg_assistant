import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest
from cryptography.fernet import Fernet
from graphiti_core.nodes import EpisodeType
from pydantic import ValidationError

from src.application.ports.films import FilmDirectoryError
from src.application.ports.knowledge import KnowledgeMemoryError
from src.application.services.action_executor import (
    MEMORY_UNAVAILABLE,
    ActionExecutor,
)
from src.application.services.film_recommendations import MAX_TOPUP_ROUNDS
from src.application.services.knowledge import KnowledgeService
from src.application.use_cases.process_message import ProcessMessageUseCase
from src.config import Settings
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    AssistantDecision,
    ChatAction,
    FilmCandidate,
    ForgetKnowledgeAction,
    RecommendFilmsAction,
    RememberKnowledgeAction,
    SaveNoteAction,
    SearchKnowledgeAction,
)
from src.domain.knowledge.films import (
    FilmRecord,
    requested_count,
    select_unwatched,
    title_key,
    title_keys,
)
from src.domain.knowledge.models import (
    MAX_PROFILE_FACTS,
    KnowledgeEpisode,
    KnowledgeFact,
    KnowledgeSourceType,
    namespace_for,
)
from src.infrastructure.knowledge.factory import (
    build_graphiti,
    create_knowledge_service,
)
from src.infrastructure.knowledge.graphiti_memory import GraphitiKnowledgeMemory

NOW = datetime(2026, 9, 3, 9, 0, tzinfo=UTC)
OTHER_NAMESPACE = "user_00000000-0000-0000-0000-000000000000"


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "telegram_bot_token": "telegram-token",
        "telegram_allowed_user_id": 42,
        "llm_api_key": "llm-key",
        "linear_api_key": "linear-key",
        "linear_team_id": "linear-team",
        "google_oauth_client_id": "google-client",
        "google_oauth_client_secret": "google-secret",
        "google_oauth_redirect_uri": "https://example.test/oauth/google/callback",
        "google_token_encryption_key": Fernet.generate_key().decode(),
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def edge(
    *,
    fact: str = "Андрей работает с Python",
    group_id: str,
    episodes: list[str] | None = None,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=f"edge-{abs(hash(fact))}",
        fact=fact,
        group_id=group_id,
        episodes=episodes if episodes is not None else ["episode-1"],
        valid_at=valid_at,
        invalid_at=invalid_at,
    )


def episodic(
    uuid: str, *, name: str, group_id: str, valid_at: datetime | None = None
) -> SimpleNamespace:
    return SimpleNamespace(uuid=uuid, name=name, group_id=group_id, valid_at=valid_at)


def found_episode(
    *,
    content: str = "Пользователь: родился 17.11.2001 в г. Лесноград",
    group_id: str,
    name: str = "telegram_message:627",
    valid_at: datetime | None = NOW,
) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=f"found-{name}",
        content=content,
        group_id=group_id,
        name=name,
        valid_at=valid_at,
    )


def entity(
    *,
    name: str = "Лесноград",
    group_id: str,
    summary: str | None = "Лесноград — город рождения Льва (род. 17.11.2001).",
) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=f"node-{name}", name=name, group_id=group_id, summary=summary
    )


class FakeGraphiti:
    def __init__(
        self,
        *,
        edges: list[SimpleNamespace] | None = None,
        nodes: list[SimpleNamespace] | None = None,
        found_episodes: list[SimpleNamespace] | None = None,
        similar: list[dict[str, Any]] | None = None,
        embedding_error: Exception | None = None,
        error: Exception | None = None,
    ) -> None:
        self.edges = edges or []
        self.nodes = nodes or []
        self.found_episodes = found_episodes or []
        self.error = error
        self.episodes: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self.closed = False
        self.indices_built = False
        self.similar = similar or []
        self.embedder = SimpleNamespace(
            create=AsyncMock(return_value=embedding_error or [0.1, 0.2, 0.3])
        )
        if isinstance(embedding_error, Exception):
            self.embedder.create = AsyncMock(side_effect=embedding_error)
        self.queries: list[dict[str, Any]] = []
        self.driver = SimpleNamespace(execute_query=self._execute_query)
        self.clients = self

    async def _execute_query(
        self, query: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], None, None]:
        self.queries.append({"query": query, **kwargs})
        return (self.similar if "vector.similarity" in query else [], None, None)

    async def build_indices_and_constraints(self) -> None:
        if self.error:
            raise self.error
        self.indices_built = True

    async def add_episode(self, **kwargs: Any) -> SimpleNamespace:
        if self.error:
            raise self.error
        self.episodes.append(kwargs)
        return SimpleNamespace(episode=SimpleNamespace(uuid="episode-uuid"))

    async def search_(self, **kwargs: Any) -> SimpleNamespace:
        if self.error:
            raise self.error
        self.searches.append(kwargs)
        return SimpleNamespace(
            edges=self.edges, nodes=self.nodes, episodes=self.found_episodes
        )

    async def close(self) -> None:
        self.closed = True


def awaited_kwargs(mock: AsyncMock) -> Mapping[str, Any]:
    assert mock.await_args is not None
    return mock.await_args.kwargs


def memory(graphiti: FakeGraphiti) -> GraphitiKnowledgeMemory:
    return GraphitiKnowledgeMemory(graphiti=graphiti)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def fake_graphiti_search(monkeypatch: pytest.MonkeyPatch) -> None:
    async def search(*, clients: FakeGraphiti, **kwargs: Any) -> SimpleNamespace:
        return await clients.search_(**kwargs)

    monkeypatch.setattr(
        "src.infrastructure.knowledge.graphiti_memory.graphiti_search", search
    )


# --- namespace isolation -----------------------------------------------------


def test_namespace_is_stable_and_opaque() -> None:
    namespace = namespace_for(736523898)

    assert namespace == namespace_for(736523898)
    assert namespace.startswith("user_")
    # A raw Telegram identifier must never end up in the graph.
    assert "736523898" not in namespace


def test_namespaces_differ_between_users() -> None:
    assert namespace_for(1) != namespace_for(2)


@pytest.mark.parametrize("user_id", [0, -1])
def test_namespace_rejects_invalid_user(user_id: int) -> None:
    with pytest.raises(ValueError):
        namespace_for(user_id)


# --- domain models -----------------------------------------------------------


def test_episode_exposes_provenance() -> None:
    episode = KnowledgeEpisode(
        content="Люблю Python",
        source_id="512",
        source_type=KnowledgeSourceType.TELEGRAM_MESSAGE,
        reference_time=NOW,
    )

    assert episode.name == "telegram_message:512"
    assert episode.source_description == "telegram chat message"


def test_episode_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        KnowledgeEpisode(
            content="",
            source_id="1",
            source_type=KnowledgeSourceType.NOTE,
            reference_time=NOW,
        )


def test_fact_payload_is_compact() -> None:
    fact = KnowledgeFact(fact="Работает в A", valid_from=NOW, source="note:1")

    assert fact.as_payload() == {
        "fact": "Работает в A",
        "valid_from": NOW.isoformat(),
        "valid_until": None,
        "stated_at": None,
        "source": "note:1",
    }


# --- application service -----------------------------------------------------


@pytest.mark.asyncio
async def test_service_remembers_within_the_user_namespace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = SimpleNamespace(remember=AsyncMock())
    service = KnowledgeService(memory=backend)

    with caplog.at_level(logging.INFO):
        await service.remember(
            user_id=42,
            content="Секретная заметка",
            source_id="512",
            source_type=KnowledgeSourceType.NOTE,
            reference_time=NOW,
        )

    kwargs = awaited_kwargs(backend.remember)
    assert kwargs["namespace"] == namespace_for(42)
    assert kwargs["episode"].reference_time == NOW
    events = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("knowledge.remember.started") for message in events)
    assert any(message.startswith("knowledge.remember.completed") for message in events)
    # Private note content never reaches the logs.
    assert not any("Секретная заметка" in message for message in events)


@pytest.mark.asyncio
async def test_service_clamps_the_search_limit() -> None:
    backend = SimpleNamespace(search=AsyncMock(return_value=[]))
    service = KnowledgeService(memory=backend)

    await service.search(user_id=42, query="фильм", limit=500)

    kwargs = awaited_kwargs(backend.search)
    assert kwargs["limit"] == 20
    assert kwargs["namespace"] == namespace_for(42)


@pytest.mark.asyncio
async def test_service_reports_failures_as_controlled_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = SimpleNamespace(search=AsyncMock(side_effect=RuntimeError("boom")))
    service = KnowledgeService(memory=backend)

    with caplog.at_level(logging.WARNING), pytest.raises(KnowledgeMemoryError):
        await service.search(user_id=42, query="фильм")

    assert any(
        record.getMessage().startswith("knowledge.search.failed")
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_service_healthcheck_survives_a_broken_backend() -> None:
    backend = SimpleNamespace(
        healthcheck=AsyncMock(side_effect=RuntimeError("no route"))
    )
    service = KnowledgeService(memory=backend)

    assert await service.healthcheck() is False


@pytest.mark.asyncio
async def test_service_closes_the_backend() -> None:
    backend = SimpleNamespace(close=AsyncMock())
    service = KnowledgeService(memory=backend)

    await service.close()

    backend.close.assert_awaited_once()


# --- Graphiti adapter --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_type", "expected"),
    [
        (KnowledgeSourceType.TELEGRAM_MESSAGE, EpisodeType.message),
        (KnowledgeSourceType.NOTE, EpisodeType.text),
        (KnowledgeSourceType.BUSINESS_NOTE, EpisodeType.message),
    ],
)
async def test_adapter_adds_episodes_with_provenance(
    source_type: KnowledgeSourceType, expected: EpisodeType
) -> None:
    graphiti = FakeGraphiti()

    await memory(graphiti).remember(
        namespace="user_abc",
        episode=KnowledgeEpisode(
            content="Андрей изучает Kafka",
            source_id="7",
            source_type=source_type,
            reference_time=NOW,
        ),
    )

    stored = graphiti.episodes[0]
    assert stored["group_id"] == "user_abc"
    assert stored["name"] == f"{source_type.value}:7"
    assert stored["source"] is expected
    assert stored["reference_time"] == NOW
    # Message episodes name their speaker so first-person facts get a subject.
    assert stored["episode_body"] == (
        "Пользователь: Андрей изучает Kafka"
        if expected is EpisodeType.message
        else "Андрей изучает Kafka"
    )


@pytest.mark.asyncio
async def test_adapter_hands_the_driver_plain_utc() -> None:
    """A ZoneInfo tzinfo segfaults the Neo4j driver inside CPython's _zoneinfo."""
    graphiti = FakeGraphiti()
    moscow = datetime(2026, 9, 3, 12, 0, tzinfo=ZoneInfo("Europe/Moscow"))

    await memory(graphiti).remember(
        namespace="user_abc",
        episode=KnowledgeEpisode(
            content="Я перешёл в компанию B",
            source_id="626",
            source_type=KnowledgeSourceType.TELEGRAM_MESSAGE,
            reference_time=moscow,
        ),
    )

    stored = graphiti.episodes[0]["reference_time"]
    assert stored.tzinfo is UTC
    # The instant itself must survive the conversion.
    assert stored == moscow


@pytest.mark.asyncio
async def test_adapter_builds_indices() -> None:
    graphiti = FakeGraphiti()

    await memory(graphiti).initialize()

    assert graphiti.indices_built is True


@pytest.mark.asyncio
async def test_adapter_maps_temporal_facts_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphiti = FakeGraphiti(
        edges=[
            edge(
                fact="Андрей работает в A",
                group_id="user_abc",
                valid_at=NOW,
                invalid_at=None,
            )
        ]
    )
    monkeypatch.setattr(
        "src.infrastructure.knowledge.graphiti_memory.EpisodicNode.get_by_uuids",
        AsyncMock(
            return_value=[episodic("episode-1", name="note:7", group_id="user_abc")]
        ),
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="работа", limit=3)

    assert len(graphiti.searches) == 1
    call = graphiti.searches[0]
    assert call["query"] == "работа"
    assert call["group_ids"] == ["user_abc"]
    # The limit bounds every layer of the combined search, not the edges alone.
    assert call["config"].limit == 3
    assert facts == [
        KnowledgeFact(
            fact="Андрей работает в A",
            valid_from=NOW,
            valid_until=None,
            source="note:7",
        )
    ]


@pytest.mark.asyncio
async def test_adapter_surfaces_facts_kept_only_on_entity_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction often keeps a detail in an entity summary and builds no edge.

    The birth date of a user who stated it in one sentence lived only in the
    summary of the city node, so an edge-only search could never answer
    "сколько мне лет".
    """
    graphiti = FakeGraphiti(
        edges=[edge(fact="Пользователь называется Андрей", group_id="user_abc")],
        nodes=[entity(group_id="user_abc")],
    )
    monkeypatch.setattr(
        "src.infrastructure.knowledge.graphiti_memory.EpisodicNode.get_by_uuids",
        AsyncMock(
            return_value=[episodic("episode-1", name="note:7", group_id="user_abc")]
        ),
    )

    facts = await memory(graphiti).search(
        namespace="user_abc", query="дата рождения", limit=5
    )

    assert [fact.fact for fact in facts] == [
        "Пользователь называется Андрей",
        "Лесноград — город рождения Льва (род. 17.11.2001).",
    ]


@pytest.mark.asyncio
async def test_adapter_drops_node_summaries_that_repeat_a_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both endpoints of an edge summarise it, so the same text arrives 3 times."""
    graphiti = FakeGraphiti(
        edges=[edge(fact="Пользователь называется Андрей", group_id="user_abc")],
        nodes=[
            entity(
                name="Пользователь",
                group_id="user_abc",
                summary="Пользователь называется Андрей",
            ),
            entity(
                name="Андрей",
                group_id="user_abc",
                summary="Пользователь называется Андрей",
            ),
            entity(group_id="user_abc"),
        ],
    )
    monkeypatch.setattr(
        "src.infrastructure.knowledge.graphiti_memory.EpisodicNode.get_by_uuids",
        AsyncMock(
            return_value=[episodic("episode-1", name="note:7", group_id="user_abc")]
        ),
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=5)

    assert [fact.fact for fact in facts] == [
        "Пользователь называется Андрей",
        "Лесноград — город рождения Льва (род. 17.11.2001).",
    ]
    # The edge keeps its provenance: the duplicate dropped is the summary.
    assert facts[0].source == "note:7"


@pytest.mark.asyncio
async def test_adapter_surfaces_the_episode_when_extraction_kept_nothing() -> None:
    """Production lost a birth date entirely: no edge, no summary, only the episode.

    The user's own sentence is the ground truth, so it stays reachable even when
    extraction produced nothing usable from it.
    """
    graphiti = FakeGraphiti(found_episodes=[found_episode(group_id="user_abc")])

    facts = await memory(graphiti).search(
        namespace="user_abc", query="дата рождения", limit=5
    )

    assert facts == [
        KnowledgeFact(
            # The speaker prefix is Graphiti's parsing aid, not part of the fact.
            fact="родился 17.11.2001 в г. Лесноград",
            valid_from=NOW,
            stated_at=NOW,
            source="telegram_message:627",
        )
    ]


@pytest.mark.asyncio
async def test_adapter_never_returns_an_episode_from_another_namespace() -> None:
    graphiti = FakeGraphiti(
        found_episodes=[
            found_episode(content="Чужое", group_id=OTHER_NAMESPACE),
            found_episode(content="Своё", group_id="user_abc"),
        ]
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=5)

    assert [fact.fact for fact in facts] == ["Своё"]


@pytest.mark.asyncio
async def test_adapter_drops_an_episode_that_only_repeats_a_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphiti = FakeGraphiti(
        edges=[edge(fact="Работает с Python", group_id="user_abc", episodes=[])],
        found_episodes=[
            found_episode(content="Работает с Python", group_id="user_abc")
        ],
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=5)

    assert [fact.fact for fact in facts] == ["Работает с Python"]


@pytest.mark.asyncio
async def test_adapter_embeds_the_episode_it_stores() -> None:
    """Dates never become edges, so the episode must be searchable by meaning."""
    graphiti = FakeGraphiti()

    await memory(graphiti).remember(
        namespace="user_abc",
        episode=KnowledgeEpisode(
            content="Пользователь родился 17.11.2001",
            source_id="627",
            source_type=KnowledgeSourceType.TELEGRAM_MESSAGE,
            reference_time=NOW,
        ),
    )

    graphiti.embedder.create.assert_awaited_once()
    written = [
        query for query in graphiti.queries if "content_embedding" in query["query"]
    ]
    assert len(written) == 1
    assert written[0]["uuid"] == "episode-uuid"
    assert written[0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_adapter_keeps_the_episode_when_embedding_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fact is already stored; losing its vector must not lose the write."""
    graphiti = FakeGraphiti(embedding_error=RuntimeError("embedder down"))

    with caplog.at_level(logging.WARNING):
        await memory(graphiti).remember(
            namespace="user_abc",
            episode=KnowledgeEpisode(
                content="Факт",
                source_id="1",
                source_type=KnowledgeSourceType.NOTE,
                reference_time=NOW,
            ),
        )

    assert len(graphiti.episodes) == 1
    assert any(
        record.getMessage().startswith("knowledge.embed.failed")
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_adapter_recalls_episodes_by_meaning() -> None:
    """«дата рождения» never matches «родился» by words, only by vector."""
    graphiti = FakeGraphiti(
        similar=[
            {
                "uuid": "similar-627",
                "name": "telegram_message:627",
                "content": "Пользователь: Пользователь родился 17.11.2001",
                "valid_at": NOW,
            }
        ]
    )

    facts = await memory(graphiti).search(
        namespace="user_abc", query="дата рождения", limit=5
    )

    assert facts == [
        KnowledgeFact(
            fact="Пользователь родился 17.11.2001",
            valid_from=NOW,
            stated_at=NOW,
            source="telegram_message:627",
        )
    ]
    vector = [
        query for query in graphiti.queries if "vector.similarity" in query["query"]
    ]
    assert vector[0]["namespace"] == "user_abc"


@pytest.mark.asyncio
async def test_adapter_never_repeats_an_episode_found_both_ways() -> None:
    graphiti = FakeGraphiti(
        found_episodes=[
            found_episode(content="Пользователь: Один факт", group_id="user_abc")
        ],
        similar=[
            {
                "uuid": "similar-627",
                "name": "telegram_message:627",
                "content": "Пользователь: Один факт",
                "valid_at": NOW,
            }
        ],
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=5)

    assert [fact.fact for fact in facts] == ["Один факт"]


@pytest.mark.asyncio
async def test_adapter_still_recalls_when_the_query_cannot_be_embedded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graphiti = FakeGraphiti(
        edges=[edge(fact="Работает с Python", group_id="user_abc", episodes=[])],
        embedding_error=RuntimeError("embedder down"),
    )

    with caplog.at_level(logging.WARNING):
        facts = await memory(graphiti).search(
            namespace="user_abc", query="работа", limit=5
        )

    assert [fact.fact for fact in facts] == ["Работает с Python"]
    assert any(
        record.getMessage().startswith("knowledge.embed.failed")
        for record in caplog.records
    )
    from graphiti_core.search.search_config import EdgeSearchMethod, NodeSearchMethod

    config = graphiti.searches[0]["config"]
    assert config.edge_config.search_methods == [EdgeSearchMethod.bm25]
    assert config.node_config.search_methods == [NodeSearchMethod.bm25]
    graphiti.embedder.create.assert_awaited_once()
    assert not graphiti.queries
    # Fallback is request-local: the next working embedding restores vectors.
    graphiti.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    await memory(graphiti).search(namespace="user_abc", query="работа", limit=5)
    assert (
        EdgeSearchMethod.cosine_similarity
        in graphiti.searches[1]["config"].edge_config.search_methods
    )


@pytest.mark.asyncio
async def test_cached_query_vector_is_shared_but_facts_and_namespaces_stay_fresh() -> (
    None
):
    graph = FakeGraphiti()
    adapter = memory(graph)
    assert (
        await adapter.search(namespace="user_abc", query="дата рождения", limit=5) == []
    )
    graph.found_episodes = [found_episode(group_id="user_abc")]
    facts = await adapter.search(namespace="user_abc", query="дата рождения", limit=5)
    assert "17.11.2001" in facts[0].fact
    assert (
        await adapter.search(namespace="user_other", query="дата рождения", limit=5)
        == []
    )
    graph.embedder.create.assert_awaited_once()
    assert len(graph.searches) == 3
    for graph_search, episode_search in zip(graph.searches, graph.queries, strict=True):
        assert graph_search["query_vector"] == episode_search["embedding"]
        assert graph_search["group_ids"] == [episode_search["namespace"]]


@pytest.mark.asyncio
async def test_query_vector_cache_expires_and_evicts_least_recently_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        "src.infrastructure.knowledge.graphiti_memory.monotonic", lambda: clock[0]
    )
    monkeypatch.setattr(
        "src.infrastructure.knowledge.graphiti_memory.QUERY_EMBEDDING_CACHE_SIZE", 2
    )
    graph = FakeGraphiti()
    adapter = memory(graph)
    for query in ["one", "two", "one", "three", "two"]:
        await adapter.search(namespace="user_abc", query=query, limit=5)
    assert graph.embedder.create.await_count == 4
    clock[0] = 301
    await adapter.search(namespace="user_abc", query="two", limit=5)
    assert graph.embedder.create.await_count == 5


@pytest.mark.asyncio
async def test_empty_search_skips_embedding_and_graph() -> None:
    graph = FakeGraphiti()
    assert await memory(graph).search(namespace="user_abc", query=" \n", limit=5) == []
    graph.embedder.create.assert_not_awaited()
    assert not graph.searches and not graph.queries


@pytest.mark.asyncio
async def test_graph_and_episode_searches_run_concurrently() -> None:
    graph = FakeGraphiti()
    graph_started, episode_started = asyncio.Event(), asyncio.Event()
    original_search = graph.search_
    original_query = graph.driver.execute_query

    async def search(**kwargs: Any) -> SimpleNamespace:
        graph_started.set()
        await episode_started.wait()
        return await original_search(**kwargs)

    async def query(cypher: str, **kwargs: Any) -> Any:
        episode_started.set()
        await graph_started.wait()
        return await original_query(cypher, **kwargs)

    graph.search_ = search  # type: ignore[method-assign]
    graph.driver.execute_query = query
    await asyncio.wait_for(
        memory(graph).search(namespace="user_abc", query="факт", limit=5), timeout=2
    )
    graph.embedder.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelling_recall_cancels_both_graph_reads() -> None:
    graph = FakeGraphiti()
    started = [asyncio.Event(), asyncio.Event()]
    cancelled: set[int] = set()

    async def block(index: int) -> Any:
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(index)

    async def search(**kwargs: Any) -> Any:
        return await block(0)

    async def query(*args: Any, **kwargs: Any) -> Any:
        return await block(1)

    graph.search_ = search  # type: ignore[method-assign]
    graph.driver.execute_query = query
    task = asyncio.create_task(
        memory(graph).search(namespace="user_abc", query="факт", limit=5)
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started)), timeout=2
        )
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert cancelled == {0, 1}


@pytest.mark.asyncio
async def test_episode_query_failure_keeps_other_search_layers() -> None:
    graph = FakeGraphiti(found_episodes=[found_episode(group_id="user_abc")])
    graph.driver.execute_query = AsyncMock(side_effect=RuntimeError("similarity down"))
    facts = await memory(graph).search(namespace="user_abc", query="родился", limit=5)
    assert "17.11.2001" in facts[0].fact


@pytest.mark.asyncio
async def test_adapter_never_returns_a_node_from_another_namespace() -> None:
    graphiti = FakeGraphiti(
        nodes=[
            entity(name="Чужой", group_id=OTHER_NAMESPACE, summary="Чужая сводка"),
            entity(name="Свой", group_id="user_abc", summary="Своя сводка"),
        ]
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=5)

    assert [fact.fact for fact in facts] == ["Своя сводка"]


@pytest.mark.asyncio
async def test_adapter_falls_back_to_the_node_name_without_a_summary() -> None:
    graphiti = FakeGraphiti(
        nodes=[entity(name="Лесноград", group_id="user_abc", summary=None)]
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=5)

    assert [fact.fact for fact in facts] == ["Лесноград"]


@pytest.mark.asyncio
async def test_adapter_keeps_node_facts_within_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edges must not starve node facts: the answer can live on either side."""
    graphiti = FakeGraphiti(
        edges=[
            edge(fact=f"Ребро {index}", group_id="user_abc", episodes=[])
            for index in range(5)
        ],
        nodes=[entity(name="Узел", group_id="user_abc", summary="Сводка узла")],
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=3)

    assert len(facts) == 3
    assert "Сводка узла" in [fact.fact for fact in facts]


@pytest.mark.asyncio
async def test_adapter_never_returns_another_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphiti = FakeGraphiti(
        edges=[
            edge(fact="Чужой факт", group_id=OTHER_NAMESPACE),
            edge(fact="Свой факт", group_id="user_abc", episodes=[]),
        ]
    )
    monkeypatch.setattr(
        "src.infrastructure.knowledge.graphiti_memory.EpisodicNode.get_by_uuids",
        AsyncMock(
            return_value=[
                episodic("episode-1", name="note:9", group_id=OTHER_NAMESPACE)
            ]
        ),
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="факт", limit=5)

    assert [fact.fact for fact in facts] == ["Свой факт"]
    # A foreign episode never becomes provenance either.
    assert facts[0].source is None


@pytest.mark.asyncio
async def test_adapter_skips_the_episode_lookup_without_edges() -> None:
    graphiti = FakeGraphiti()

    assert await memory(graphiti).search(namespace="user_abc", query="x", limit=5) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["initialize", "remember", "search"])
async def test_adapter_translates_backend_failures(operation: str) -> None:
    graphiti = FakeGraphiti(error=RuntimeError("Neo4j unavailable"))
    adapter = memory(graphiti)

    with pytest.raises(KnowledgeMemoryError, match="Neo4j unavailable"):
        if operation == "initialize":
            await adapter.initialize()
        elif operation == "remember":
            await adapter.remember(
                namespace="user_abc",
                episode=KnowledgeEpisode(
                    content="x",
                    source_id="1",
                    source_type=KnowledgeSourceType.NOTE,
                    reference_time=NOW,
                ),
            )
        else:
            await adapter.search(namespace="user_abc", query="x", limit=1)


@pytest.mark.asyncio
async def test_adapter_healthcheck_reports_connectivity() -> None:
    graphiti = FakeGraphiti()

    assert await memory(graphiti).healthcheck() is True

    graphiti.driver.execute_query = AsyncMock(side_effect=OSError("connection refused"))

    assert await memory(graphiti).healthcheck() is False


@pytest.mark.asyncio
async def test_adapter_closes_the_client() -> None:
    graphiti = FakeGraphiti()

    await memory(graphiti).close()

    assert graphiti.closed is True


# --- factory -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_factory_returns_nothing_when_disabled() -> None:
    assert await create_knowledge_service(settings()) is None


@pytest.mark.asyncio
async def test_factory_degrades_when_neo4j_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    graphiti = FakeGraphiti(error=RuntimeError("Neo4j unavailable"))
    monkeypatch.setattr(
        "src.infrastructure.knowledge.factory.build_graphiti", lambda _: graphiti
    )

    with caplog.at_level(logging.ERROR):
        service = await create_knowledge_service(
            settings(knowledge_memory_enabled=True, neo4j_password="secret")
        )

    assert service is None
    assert graphiti.closed is True
    assert any(
        "knowledge.startup.failed" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_factory_builds_a_single_shared_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphiti = FakeGraphiti()
    builds = 0

    def build(_: Settings) -> FakeGraphiti:
        nonlocal builds
        builds += 1
        return graphiti

    monkeypatch.setattr("src.infrastructure.knowledge.factory.build_graphiti", build)

    service = await create_knowledge_service(
        settings(knowledge_memory_enabled=True, neo4j_password="secret")
    )

    assert service is not None
    assert builds == 1
    assert graphiti.indices_built is True


def test_factory_reuses_the_main_llm_provider_by_default() -> None:
    configured = settings(
        knowledge_memory_enabled=True,
        neo4j_password="secret",
        llm_base_url="https://api.example.test/v1",
        llm_model="strong-model",
    )

    assert configured.resolved_graphiti_llm_model == "strong-model"
    assert configured.resolved_graphiti_llm_base_url == "https://api.example.test/v1"
    assert configured.resolved_graphiti_embedding_api_key == "llm-key"

    build_graphiti(configured)


def test_factory_prefers_a_dedicated_cheap_model() -> None:
    configured = settings(
        knowledge_memory_enabled=True,
        neo4j_password="secret",
        graphiti_llm_model="cheap-model",
        graphiti_llm_api_key="cheap-key",
        graphiti_llm_base_url="https://cheap.test/v1",
        graphiti_embedding_base_url="https://embed.test/v1",
    )

    assert configured.resolved_graphiti_llm_model == "cheap-model"
    assert configured.resolved_graphiti_llm_api_key == "cheap-key"
    assert configured.resolved_graphiti_embedding_api_key == "cheap-key"
    assert configured.resolved_graphiti_embedding_base_url == "https://embed.test/v1"


def test_settings_require_a_neo4j_password_when_enabled() -> None:
    with pytest.raises(ValidationError):
        settings(knowledge_memory_enabled=True)


def test_settings_reject_a_non_bolt_neo4j_uri() -> None:
    with pytest.raises(ValidationError):
        settings(neo4j_uri="https://neo4j:7687")


# --- agent tools -------------------------------------------------------------


def executor(
    *,
    knowledge: KnowledgeService | None = None,
) -> ActionExecutor:
    return ActionExecutor(
        calendar=SimpleNamespace(),
        pending_operations=SimpleNamespace(),
        task_tracker=SimpleNamespace(),
        knowledge=knowledge,
    )


def service(**mocks: object) -> KnowledgeService:
    return KnowledgeService(memory=SimpleNamespace(**mocks))


@pytest.mark.asyncio
async def test_recommendations_exclude_entire_catalogue_aliases_and_duplicates() -> (
    None
):
    titles = [f"Старый фильм {i}" for i in range(440)] + [
        "Достать ножи: Стеклянная луковица",
        "Начало",
        "Славные парни (1990)",
        "Ёлки",
    ]
    catalogue = AsyncMock(return_value=titles)
    knowledge = service(watched_film_titles=catalogue)
    action = RecommendFilmsAction(
        type=ActionType.RECOMMEND_FILMS,
        limit=3,
        candidates=[
            FilmCandidate(title="INCEPTION", aliases=["«Начало»"], reason="a"),
            FilmCandidate(title="Стеклянная луковица", reason="уже просмотрен"),
            FilmCandidate(title="Славные-парни", reason="b"),
            FilmCandidate(title="Елки", reason="c"),
            FilmCandidate(title="Старый фильм 439", reason="d"),
            FilmCandidate(title="Новый фильм", aliases=["New film"], reason="e"),
            FilmCandidate(title="New film", reason="f"),
            FilmCandidate(title="Ещё новый фильм", reason="g"),
        ],
    )
    result = await executor(knowledge=knowledge).execute_many(
        [ChatAction(type=ActionType.CHAT, text="Советую Начало"), action],
        user_id=42,
        now=NOW,
    )
    assert isinstance(result, str)
    assert "Новый фильм" in result and "Ещё новый фильм" in result
    assert all(
        title not in result
        for title in [
            "Начало",
            "INCEPTION",
            "Славные",
            "Елки",
            "Старый",
            "New film",
            "Стеклянная луковица",
        ]
    )
    catalogue.assert_awaited_once_with(namespace=namespace_for(42))


def test_select_unwatched_drops_watched_aliases_blanks_and_repeats() -> None:
    watched = {key for title in ["Контакт", "Ёлки"] for key in title_keys(title)}
    candidates = [
        FilmCandidate(title="Контакт", reason="просмотрен"),
        FilmCandidate(title="Прибытие", aliases=["Елки"], reason="алиас просмотрен"),
        FilmCandidate(title="  ", reason="пустое название"),
        FilmCandidate(title="Дюна", aliases=["Dune"], reason="новый"),
        FilmCandidate(title="Дюна (2021)", reason="тот же фильм"),
        FilmCandidate(title="Магнолия", reason="новый"),
        FilmCandidate(title="Пианист", reason="сверх лимита"),
    ]

    selected = select_unwatched(candidates, watched=watched, limit=2)

    assert [candidate.title for candidate in selected] == ["Дюна", "Магнолия"]


@pytest.mark.asyncio
async def test_recommendations_show_the_release_year_when_the_model_gives_one() -> None:
    action = RecommendFilmsAction(
        type=ActionType.RECOMMEND_FILMS,
        candidates=[FilmCandidate(title="Дюна", year=2021, reason="Эпос")],
    )

    result = await executor(
        knowledge=service(watched_film_titles=AsyncMock(return_value=["Контакт"]))
    ).execute_many([action], user_id=42, now=NOW)

    assert isinstance(result, str)
    assert "«Дюна» (2021) — Эпос" in result


@pytest.mark.asyncio
async def test_a_candidate_without_a_reason_still_reaches_the_user() -> None:
    """A dropped reason once invalidated the whole decision and lost the answer."""
    action = RecommendFilmsAction.model_validate(
        {
            "type": ActionType.RECOMMEND_FILMS,
            "limit": 2,
            "candidates": [
                {"title": "Безмолвный фильм", "reason": "  "},
                {"title": "Второй фильм", "reason": "Причина"},
            ],
        }
    )
    result = await executor(
        knowledge=service(watched_film_titles=AsyncMock(return_value=["Начало"]))
    ).execute_many([action], user_id=42, now=NOW)

    assert isinstance(result, str)
    assert "«Безмолвный фильм»" in result and "—  " not in result
    assert "«Второй фильм» — Причина" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_unavailable_catalogue_never_leaks_unchecked_recommendations(
    failure: bool,
) -> None:
    catalogue = AsyncMock(return_value=[])
    if failure:
        catalogue.side_effect = KnowledgeMemoryError("offline")
    action = RecommendFilmsAction(
        type=ActionType.RECOMMEND_FILMS,
        candidates=[
            FilmCandidate(title="Непроверенный фильм", reason="Непроверенная причина")
        ],
    )
    result = await executor(
        knowledge=service(watched_film_titles=catalogue)
    ).execute_many([action], user_id=42, now=NOW)
    assert isinstance(result, str)
    assert "Непроверенный" not in result


@pytest.mark.asyncio
async def test_catalogue_keeps_old_undated_titles_without_search_limit() -> None:
    graph = FakeGraphiti()
    query = AsyncMock(
        return_value=(
            [
                {"content": f"Просмотренный фильм: Фильм {i}; жанр: драма"}
                for i in range(450)
            ],
            None,
            None,
        )
    )
    graph.driver.execute_query = query
    titles = await memory(graph).watched_film_titles(namespace=namespace_for(42))
    assert len(titles) == 450
    assert titles[-1] == "Фильм 449"
    args = query.await_args
    assert args is not None
    assert args.kwargs["namespace"] == namespace_for(42)
    assert "LIMIT" not in args.args[0] and "watched_at" not in args.args[0]


@pytest.mark.asyncio
async def test_remember_tool_uses_the_trusted_user_id() -> None:
    backend = AsyncMock()
    knowledge = KnowledgeService(memory=SimpleNamespace(remember=backend))

    result = await executor(knowledge=knowledge).execute(
        RememberKnowledgeAction(
            type=ActionType.REMEMBER_KNOWLEDGE, content="Люблю Python"
        ),
        user_id=42,
        now=NOW,
        message_id=512,
    )

    episode = awaited_kwargs(backend)["episode"]
    assert awaited_kwargs(backend)["namespace"] == namespace_for(42)
    assert episode.name == "telegram_message:512"
    assert episode.reference_time == NOW
    assert result == "🧠 Запомнил: Люблю Python"


@pytest.mark.asyncio
async def test_remember_tool_falls_back_to_a_timestamp_source() -> None:
    backend = AsyncMock()
    knowledge = KnowledgeService(memory=SimpleNamespace(remember=backend))

    await executor(knowledge=knowledge).execute(
        RememberKnowledgeAction(type=ActionType.REMEMBER_KNOWLEDGE, content="Факт"),
        user_id=42,
        now=NOW,
    )

    assert awaited_kwargs(backend)["episode"].source_id == "20260903T090000+0000"


@pytest.mark.asyncio
async def test_notes_are_persisted_as_knowledge() -> None:
    backend = AsyncMock()
    knowledge = KnowledgeService(memory=SimpleNamespace(remember=backend))

    result = await executor(knowledge=knowledge).execute(
        SaveNoteAction(type=ActionType.SAVE_NOTE, text="Пароль лежит на роутере"),
        user_id=42,
        now=NOW,
        message_id=7,
    )

    assert awaited_kwargs(backend)["episode"].name == "note:7"
    assert result == "📝 Заметка сохранена:\nПароль лежит на роутере"


@pytest.mark.asyncio
async def test_notes_still_work_without_memory() -> None:
    result = await executor().execute(
        SaveNoteAction(type=ActionType.SAVE_NOTE, text="Люблю Python"),
        user_id=42,
        now=NOW,
    )

    assert result == "📝 Понял, нужно сохранить заметку:\nЛюблю Python"


@pytest.mark.asyncio
async def test_search_tool_returns_compact_temporal_facts() -> None:
    knowledge = service(
        search=AsyncMock(
            return_value=[
                KnowledgeFact(fact="Работает в A", valid_from=NOW, valid_until=NOW),
                KnowledgeFact(fact="Работает в B", valid_from=NOW),
                KnowledgeFact(fact="Хотел посмотреть Blade Runner"),
            ]
        )
    )

    result = await executor(knowledge=knowledge).execute(
        SearchKnowledgeAction(type=ActionType.SEARCH_KNOWLEDGE, query="работа"),
        user_id=42,
        now=NOW,
    )

    assert result == (
        "🧠 Нашёл в памяти:\n"
        "• Работает в A (с 03.09.2026, до 03.09.2026)\n"
        "• Работает в B (с 03.09.2026)\n"
        "• Хотел посмотреть Blade Runner"
    )


@pytest.mark.asyncio
async def test_search_tool_reports_an_empty_memory() -> None:
    knowledge = service(search=AsyncMock(return_value=[]))

    result = await executor(knowledge=knowledge).execute(
        SearchKnowledgeAction(type=ActionType.SEARCH_KNOWLEDGE, query="работа"),
        user_id=42,
        now=NOW,
    )

    assert result == "🧠 В долговременной памяти ничего не нашлось по этому запросу."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (
            RememberKnowledgeAction(type=ActionType.REMEMBER_KNOWLEDGE, content="Факт"),
            "⚠️ Долговременная память отключена, сохранить факт не могу.",
        ),
        (
            SearchKnowledgeAction(type=ActionType.SEARCH_KNOWLEDGE, query="факт"),
            "⚠️ Долговременная память отключена, поиск по ней недоступен.",
        ),
    ],
)
async def test_knowledge_tools_explain_a_disabled_memory(
    action: RememberKnowledgeAction | SearchKnowledgeAction, expected: str
) -> None:
    assert await executor().execute(action, user_id=42, now=NOW) == expected


@pytest.mark.asyncio
async def test_agent_keeps_working_when_memory_is_down() -> None:
    knowledge = service(
        search=AsyncMock(side_effect=KnowledgeMemoryError("Neo4j unavailable"))
    )

    result = await executor(knowledge=knowledge).execute_many(
        [SearchKnowledgeAction(type=ActionType.SEARCH_KNOWLEDGE, query="работа")],
        user_id=42,
        now=NOW,
    )

    assert result == MEMORY_UNAVAILABLE


@pytest.mark.asyncio
async def test_business_notes_never_fail_on_a_memory_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from src.domain.assistant.business import BusinessNote

    knowledge = service(
        remember=AsyncMock(side_effect=KnowledgeMemoryError("Neo4j unavailable"))
    )

    with caplog.at_level(logging.WARNING):
        result = await executor(knowledge=knowledge).execute_business(
            BusinessNote(type=ActionType.SAVE_NOTE, text="Клиент любит созвоны"),
            user_id=42,
            now=NOW,
            source_id="action-1",
        )

    assert result == "📝 Сохранена заметка: Клиент любит созвоны"
    assert any(
        "knowledge.remember.skipped" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_business_notes_reach_the_graph_with_provenance() -> None:
    backend = AsyncMock()
    knowledge = KnowledgeService(memory=SimpleNamespace(remember=backend))

    from src.domain.assistant.business import BusinessNote

    await executor(knowledge=knowledge).execute_business(
        BusinessNote(type=ActionType.SAVE_NOTE, text="Клиент любит созвоны"),
        user_id=42,
        now=NOW,
        source_id="action-1",
    )

    assert awaited_kwargs(backend)["episode"].name == "business_note:action-1"


# --- tool-based retrieval ----------------------------------------------------


def use_case(
    *,
    decisions: list[AssistantDecision],
    knowledge: KnowledgeService | None,
    films: object | None = None,
) -> tuple[ProcessMessageUseCase, SimpleNamespace, SimpleNamespace]:
    llm = SimpleNamespace(parse_message=AsyncMock(side_effect=decisions))
    action_executor = SimpleNamespace(execute_many=AsyncMock(return_value="Ответ"))
    return (
        ProcessMessageUseCase(
            llm=llm,
            action_executor=action_executor,  # type: ignore[arg-type]
            knowledge=knowledge,
            films=films,  # type: ignore[arg-type]
        ),
        llm,
        action_executor,
    )


def known(*films: FilmRecord) -> SimpleNamespace:
    """A directory that knows exactly these films, by either title."""
    by_key = {
        title_key(title): record
        for record in films
        for title in (record.title, record.original_title)
    }

    async def find(*, title: str, year: int | None) -> FilmRecord | None:
        return by_key.get(title_key(title))

    return SimpleNamespace(find=find, close=AsyncMock())


def chat(text: str) -> AssistantDecision:
    return AssistantDecision(actions=[ChatAction(type=ActionType.CHAT, text=text)])


def recommending(*titles: str, limit: int = 3) -> AssistantDecision:
    return AssistantDecision(
        actions=[
            RecommendFilmsAction(
                type=ActionType.RECOMMEND_FILMS,
                limit=limit,
                candidates=[
                    FilmCandidate(title=title, reason="Причина") for title in titles
                ],
            )
        ]
    )


def searching(query: str = "фильм") -> AssistantDecision:
    return AssistantDecision(
        actions=[SearchKnowledgeAction(type=ActionType.SEARCH_KNOWLEDGE, query=query)]
    )


@pytest.mark.asyncio
async def test_recommendation_generation_receives_full_exclusion_catalogue() -> None:
    titles = [f"Фильм {i}" for i in range(441)]
    knowledge = service(watched_film_titles=AsyncMock(return_value=titles))
    recommendation = AssistantDecision(
        actions=[
            RecommendFilmsAction(
                type=ActionType.RECOMMEND_FILMS,
                limit=1,
                candidates=[FilmCandidate(title="Новый фильм", reason="Причина")],
            )
        ]
    )
    process, llm, _ = use_case(
        decisions=[recommendation, recommendation], knowledge=knowledge
    )
    await process.execute(
        text="Посоветуй фильмы которые я не смотрел",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )
    recalled = awaited_kwargs(llm.parse_message)["knowledge"]
    assert "Фильм 440" in recalled and "Фильм 0" in recalled


@pytest.mark.asyncio
async def test_a_title_no_directory_knows_never_reaches_the_user() -> None:
    """The bot offered «Список контактов», retelling the plot of «Контакт»."""
    knowledge = service(watched_film_titles=AsyncMock(return_value=["Контакт"]))
    batch = recommending("Список контактов", "Дюна", "Магнолия", limit=2)
    process, _, executed = use_case(
        decisions=[batch, batch],
        knowledge=knowledge,
        films=known(
            FilmRecord(title="Дюна", original_title="Dune", year=2021),
            FilmRecord(title="Магнолия", original_title="Magnolia", year=1999),
        ),
    )

    await process.execute(
        text="Посоветуй фильмы которые я не смотрел",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    actions = executed.execute_many.await_args.args[0]
    assert [candidate.title for candidate in actions[0].candidates] == [
        "Дюна",
        "Магнолия",
    ]
    assert actions[0].candidates[0].year == 2021
    assert "Dune" in actions[0].candidates[0].aliases


@pytest.mark.asyncio
async def test_a_renamed_watched_film_is_excluded_under_its_real_title() -> None:
    knowledge = service(watched_film_titles=AsyncMock(return_value=["Контакт"]))
    batch = recommending("Контакт с внеземным разумом", "Дюна", limit=1)
    # The directory resolves the loose title back to the watched film.
    resolved = {
        "Контакт с внеземным разумом": FilmRecord("Контакт", "Contact", 1997),
        "Дюна": FilmRecord("Дюна", "Dune", 2021),
    }

    async def find(*, title: str, year: int | None) -> FilmRecord | None:
        return resolved.get(title)

    process, _, executed = use_case(
        decisions=[batch, batch],
        knowledge=knowledge,
        films=SimpleNamespace(find=find, close=AsyncMock()),
    )

    await process.execute(
        text="Посоветуй фильмы которые я не смотрел",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    actions = executed.execute_many.await_args.args[0]
    assert [candidate.title for candidate in actions[0].candidates] == ["Дюна"]


@pytest.mark.asyncio
async def test_an_unreachable_directory_never_swallows_the_whole_batch() -> None:
    knowledge = service(watched_film_titles=AsyncMock(return_value=["Контакт"]))
    batch = recommending("Дюна", "Магнолия", limit=2)
    films = SimpleNamespace(
        find=AsyncMock(side_effect=FilmDirectoryError("offline")), close=AsyncMock()
    )
    process, _, executed = use_case(
        decisions=[batch, batch], knowledge=knowledge, films=films
    )

    await process.execute(
        text="Посоветуй фильмы которые я не смотрел",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    actions = executed.execute_many.await_args.args[0]
    assert [candidate.title for candidate in actions[0].candidates] == [
        "Дюна",
        "Магнолия",
    ]


@pytest.mark.asyncio
async def test_a_watched_batch_is_regenerated_once_with_the_rejects_named() -> None:
    """A model that proposes only watched films must not end in a dead end."""
    knowledge = service(watched_film_titles=AsyncMock(return_value=["Контакт", "Дюна"]))
    watched_batch = recommending("Контакт", "Дюна", limit=2)
    fresh_batch = recommending("Магнолия", "Пианист", limit=2)
    process, llm, executed = use_case(
        decisions=[watched_batch, watched_batch, fresh_batch], knowledge=knowledge
    )

    await process.execute(
        text="Посоветуй фильмы которые я не смотрел",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert llm.parse_message.await_count == 3
    retry_knowledge = llm.parse_message.await_args.kwargs["knowledge"]
    assert "Контакт" in retry_knowledge and "Дюна" in retry_knowledge
    actions = executed.execute_many.await_args.args[0]
    assert [candidate.title for candidate in actions[0].candidates] == [
        "Магнолия",
        "Пианист",
    ]


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("Посоветуй топ 10 фильмов для меня", 10),
        ("Посоветуй топ-5 фильмов", 5),
        ("дай 7 лучших фильмов, которые я не смотрел", 7),
        ("Посоветуй пять фильмов", 5),
        ("Порекомендуй десять хороших фильмов на вечер", 10),
        ("Посоветуй пару фильмов", 2),
        ("Посоветуй один фильм", 1),
        ("Подкинь десяток фильмов", 10),
        ("Посоветуй 20 фильмов", 10),
        ("Посоветуй фильмы, которые я не смотрел", None),
        ("Посоветуй фильм 2019 года", None),
        ("Посоветуй фильмы про 90-е", None),
    ],
)
def test_requested_count_reads_the_number_the_user_asked_for(
    text: str, count: int | None
) -> None:
    assert requested_count(text) == count


@pytest.mark.asyncio
async def test_an_explicit_count_overrides_the_limit_the_model_chose() -> None:
    """«Посоветуй топ 10 фильмов» reached production with limit=3 (2026-09-15)."""
    knowledge = service(watched_film_titles=AsyncMock(return_value=["Контакт"]))
    batch = recommending(*(f"Новый фильм {i}" for i in range(12)), limit=3)
    process, llm, executed = use_case(decisions=[batch, batch], knowledge=knowledge)

    await process.execute(
        text="Посоветуй топ 10 фильмов для меня",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert llm.parse_message.await_count == 2
    action = executed.execute_many.await_args.args[0][0]
    assert action.limit == 10
    assert len(action.candidates) == 10


@pytest.mark.asyncio
async def test_regeneration_continues_until_the_requested_count_is_filled() -> None:
    """Production showed 2 of 10: the single retry also came back watched."""
    watched = [f"Классика {i}" for i in range(20)]
    knowledge = service(watched_film_titles=AsyncMock(return_value=watched))
    first = recommending(*watched[:10], "Новый 1", limit=4)
    second = recommending(*watched[10:], "Новый 1", "Новый 2", limit=4)
    third = recommending("Новый 3", "Новый 4", "Новый 5", limit=4)
    process, llm, executed = use_case(
        decisions=[first, first, second, third], knowledge=knowledge
    )

    await process.execute(
        text="Посоветуй фильмы которые я не смотрел",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert llm.parse_message.await_count == 4
    # Every round names everything rejected so far, not just the last batch.
    last_retry = llm.parse_message.await_args.kwargs["knowledge"]
    assert "Классика 0" in last_retry and "Классика 19" in last_retry
    action = executed.execute_many.await_args.args[0][0]
    assert [candidate.title for candidate in action.candidates] == [
        "Новый 1",
        "Новый 2",
        "Новый 3",
        "Новый 4",
    ]


@pytest.mark.asyncio
async def test_regeneration_gives_up_after_a_bounded_number_of_rounds() -> None:
    watched = ["Контакт", "Дюна"]
    knowledge = service(watched_film_titles=AsyncMock(return_value=watched))
    batch = recommending(*watched, "Магнолия", limit=5)
    process, llm, executed = use_case(decisions=[batch] * 10, knowledge=knowledge)

    await process.execute(
        text="Посоветуй фильмы которые я не смотрел",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert llm.parse_message.await_count == 2 + MAX_TOPUP_ROUNDS
    action = executed.execute_many.await_args.args[0][0]
    assert [candidate.title for candidate in action.candidates] == ["Магнолия"]


@pytest.mark.asyncio
async def test_a_short_list_says_it_is_shorter_than_asked() -> None:
    action = RecommendFilmsAction(
        type=ActionType.RECOMMEND_FILMS,
        limit=10,
        candidates=[
            FilmCandidate(title="Контакт", reason="просмотрен"),
            FilmCandidate(title="Дюна", reason="Эпос"),
            FilmCandidate(title="Магнолия", reason="Драма"),
        ],
    )

    result = await executor(
        knowledge=service(watched_film_titles=AsyncMock(return_value=["Контакт"]))
    ).execute_many([action], user_id=42, now=NOW)

    assert isinstance(result, str)
    assert "«Дюна» — Эпос" in result and "«Магнолия» — Драма" in result
    assert "только 2 из 10" in result


@pytest.mark.asyncio
async def test_a_full_list_carries_no_shortfall_note() -> None:
    action = RecommendFilmsAction(
        type=ActionType.RECOMMEND_FILMS,
        limit=1,
        candidates=[FilmCandidate(title="Дюна", reason="Эпос")],
    )

    result = await executor(
        knowledge=service(watched_film_titles=AsyncMock(return_value=["Контакт"]))
    ).execute_many([action], user_id=42, now=NOW)

    assert isinstance(result, str)
    assert "только" not in result


@pytest.mark.asyncio
async def test_a_full_batch_of_new_films_is_never_regenerated() -> None:
    knowledge = service(watched_film_titles=AsyncMock(return_value=["Контакт"]))
    fresh_batch = recommending("Магнолия", "Пианист", limit=2)
    process, llm, executed = use_case(
        decisions=[fresh_batch, fresh_batch], knowledge=knowledge
    )

    await process.execute(
        text="Посоветуй фильмы которые я не смотрел",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert llm.parse_message.await_count == 2
    actions = executed.execute_many.await_args.args[0]
    assert [candidate.title for candidate in actions[0].candidates] == [
        "Магнолия",
        "Пианист",
    ]


@pytest.mark.asyncio
async def test_catalogue_answer_cannot_bypass_filter_in_chat() -> None:
    knowledge = service(watched_film_titles=AsyncMock(return_value=["Начало"]))
    decision = AssistantDecision(
        actions=[
            SearchKnowledgeAction(
                type=ActionType.SEARCH_KNOWLEDGE,
                query="фильмы",
                mode="watched_film_catalogue",
            )
        ]
    )
    process, _, executed = use_case(
        decisions=[decision, chat("Советую Начало")], knowledge=knowledge
    )
    await process.execute(
        text="Посоветуй непросмотренное", now=NOW, timezone="Europe/Moscow", user_id=42
    )
    actions = executed.execute_many.await_args.args[0]
    assert actions[0].type == ActionType.CHAT and "Начало" not in actions[0].text


@pytest.mark.asyncio
async def test_recent_films_use_dated_catalogue_and_reach_answer_turn() -> None:
    recent = AsyncMock(
        return_value=[KnowledgeFact(fact="Последний фильм", valid_from=NOW)]
    )
    semantic = AsyncMock()
    knowledge = KnowledgeService(
        memory=SimpleNamespace(recent_watched_films=recent, search=semantic)
    )
    decision = AssistantDecision(
        actions=[
            SearchKnowledgeAction(
                type=ActionType.SEARCH_KNOWLEDGE,
                query="недавние фильмы",
                mode="recent_watched_films",
                limit=3,
            )
        ]
    )
    process, llm, _ = use_case(
        decisions=[decision, chat("Последний фильм")], knowledge=knowledge
    )
    await process.execute(
        text="Какие фильмы недавно я смотрел",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )
    recent.assert_awaited_once_with(namespace=namespace_for(42), limit=3)
    semantic.assert_not_awaited()
    assert "Последний фильм" in awaited_kwargs(llm.parse_message)["knowledge"]
    assert NOW.isoformat() in awaited_kwargs(llm.parse_message)["knowledge"]


@pytest.mark.asyncio
async def test_recent_films_convert_neo4j_dates_and_handle_unavailability() -> None:
    from neo4j.time import DateTime

    graph = FakeGraphiti()
    query = AsyncMock(
        return_value=(
            [
                {
                    "content": "Просмотренный фильм: Пример",
                    "name": "catalogue:1",
                    "watched_at": DateTime.from_native(NOW),
                }
            ],
            None,
            None,
        )
    )
    graph.driver.execute_query = query
    facts = await memory(graph).recent_watched_films(
        namespace=namespace_for(42), limit=3
    )
    assert facts[0].valid_from == NOW
    query.side_effect = RuntimeError("offline")
    with pytest.raises(KnowledgeMemoryError):
        await memory(graph).recent_watched_films(namespace=namespace_for(42), limit=3)


@pytest.mark.asyncio
async def test_an_updated_fact_keeps_the_time_it_was_stated() -> None:
    """Graphiti kept the superseded edge's valid_at when the city changed."""
    moved = NOW
    graph = FakeGraphiti(
        edges=[
            edge(
                fact="Пользователь живёт в Леснограде",
                group_id=namespace_for(42),
                episodes=["said-later"],
                valid_at=NOW - timedelta(days=40),
            )
        ]
    )
    episodes = AsyncMock(
        return_value=[
            SimpleNamespace(
                uuid="said-later",
                name="telegram_message:9",
                group_id=namespace_for(42),
                valid_at=moved,
            )
        ]
    )
    monkeypatched = "src.infrastructure.knowledge.graphiti_memory.EpisodicNode"
    with patch(monkeypatched, SimpleNamespace(get_by_uuids=episodes)):
        facts = await memory(graph).search(
            namespace=namespace_for(42), query="город", limit=5
        )

    assert facts[0].valid_from == NOW - timedelta(days=40)
    assert facts[0].stated_at == moved
    assert facts[0].as_payload()["stated_at"] == moved.isoformat()


@pytest.mark.asyncio
async def test_profile_reads_every_statement_oldest_first_without_films() -> None:
    from neo4j.time import DateTime

    older = NOW - timedelta(days=200)
    graph = FakeGraphiti()
    # The query ranks newest first so the limit keeps the recent statements.
    query = AsyncMock(
        return_value=(
            [
                {
                    "uuid": "e2",
                    "name": "telegram_message:2",
                    "content": "Пользователь: Мой вес 78 кг",
                    "valid_at": DateTime.from_native(NOW),
                },
                {
                    "uuid": "e1",
                    "name": "telegram_message:1",
                    "content": "Пользователь: Мой вес 84 кг",
                    "valid_at": older,
                },
                {
                    "uuid": "e0",
                    "name": "telegram_message:0",
                    "content": "Мой вес 84 кг",
                    "valid_at": older,
                },
            ],
            None,
            None,
        )
    )
    graph.driver.execute_query = query

    facts = await memory(graph).profile_facts(namespace=namespace_for(42), limit=300)

    assert [(fact.fact, fact.valid_from) for fact in facts] == [
        ("Мой вес 84 кг", older),
        ("Мой вес 78 кг", NOW),
    ]
    args = query.await_args
    assert args is not None
    assert args.kwargs["namespace"] == namespace_for(42)
    assert args.kwargs["limit"] == 300
    assert "structured film export" in args.args[0]
    query.side_effect = RuntimeError("offline")
    with pytest.raises(KnowledgeMemoryError):
        await memory(graph).profile_facts(namespace=namespace_for(42), limit=300)


@pytest.mark.asyncio
async def test_profile_mode_hands_the_whole_profile_to_the_answer_turn() -> None:
    profile = AsyncMock(
        return_value=[
            KnowledgeFact(fact="В 2019 году ездил на Байкал", valid_from=NOW),
            KnowledgeFact(fact="В феврале 2026 ездили с Машей в Карелию"),
        ]
    )
    semantic = AsyncMock()
    knowledge = KnowledgeService(
        memory=SimpleNamespace(profile_facts=profile, search=semantic)
    )
    decision = AssistantDecision(
        actions=[
            SearchKnowledgeAction(
                type=ActionType.SEARCH_KNOWLEDGE, query="поездки", mode="profile"
            )
        ]
    )
    process, llm, _ = use_case(
        decisions=[decision, chat("Байкал, Карелия")], knowledge=knowledge
    )

    await process.execute(
        text="Составь все мои поездки по годам",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    profile.assert_awaited_once_with(
        namespace=namespace_for(42), limit=MAX_PROFILE_FACTS
    )
    semantic.assert_not_awaited()
    recalled = awaited_kwargs(llm.parse_message)["knowledge"]
    assert "Байкал" in recalled and "Карелию" in recalled


@pytest.mark.asyncio
async def test_plain_questions_never_touch_memory() -> None:
    backend = AsyncMock()
    knowledge = KnowledgeService(memory=SimpleNamespace(search=backend))
    process, llm, _ = use_case(decisions=[chat("4")], knowledge=knowledge)

    await process.execute(
        text="Сколько будет 2 + 2?", now=NOW, timezone="Europe/Moscow", user_id=42
    )

    backend.assert_not_awaited()
    assert llm.parse_message.await_count == 1


@pytest.mark.asyncio
async def test_recalled_facts_are_handed_back_to_the_agent() -> None:
    knowledge = service(
        search=AsyncMock(
            return_value=[
                KnowledgeFact(fact="Хотел посмотреть Blade Runner", valid_from=NOW)
            ]
        )
    )
    process, llm, action_executor = use_case(
        decisions=[searching(), chat("Blade Runner")], knowledge=knowledge
    )

    await process.execute(
        text="Какой фильм я хотел посмотреть?",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    recalled = awaited_kwargs(llm.parse_message)["knowledge"]
    assert "Blade Runner" in recalled
    assert NOW.isoformat() in recalled
    assert action_executor.execute_many.await_args is not None
    executed = action_executor.execute_many.await_args.args[0]
    assert executed == [ChatAction(type=ActionType.CHAT, text="Blade Runner")]


@pytest.mark.asyncio
async def test_multiple_memory_requests_are_bounded_deduplicated_and_keep_order() -> (
    None
):
    active = 0
    peak = 0
    ready, release = asyncio.Event(), asyncio.Event()

    async def find(*, namespace: str, query: str, limit: int) -> list[KnowledgeFact]:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 4:
            ready.set()
        try:
            await release.wait()
            return [KnowledgeFact(fact=query), KnowledgeFact(fact="Общий факт")]
        finally:
            active -= 1

    backend = AsyncMock(side_effect=find)
    process, _, _ = use_case(decisions=[], knowledge=service(search=backend))
    searches = [
        SearchKnowledgeAction(type=ActionType.SEARCH_KNOWLEDGE, query=str(i))
        for i in range(6)
    ]
    task = asyncio.create_task(process._recall([*searches, searches[0]], user_id=42))
    try:
        await asyncio.wait_for(ready.wait(), timeout=2)
        assert peak == 4
    finally:
        release.set()
        _, facts = await task
    assert backend.await_count == 6
    assert [fact.fact for fact in facts] == ["0", "Общий факт", "1", "2", "3", "4", "5"]


@pytest.mark.asyncio
async def test_unexpected_catalogue_failure_is_not_silenced() -> None:
    backend = SimpleNamespace(lookup=AsyncMock(side_effect=ValueError("bad catalogue")))
    process, _, _ = use_case(decisions=[], knowledge=backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bad catalogue"):
        await process._recall(
            [
                SearchKnowledgeAction(
                    type=ActionType.SEARCH_KNOWLEDGE,
                    query="фильмы",
                    mode="watched_film_catalogue",
                )
            ],
            user_id=42,
        )


@pytest.mark.asyncio
async def test_a_second_search_never_loops() -> None:
    knowledge = service(
        search=AsyncMock(return_value=[KnowledgeFact(fact="Работает в B")])
    )
    process, llm, action_executor = use_case(
        decisions=[searching(), searching(), searching()], knowledge=knowledge
    )

    await process.execute(
        text="Где я работаю?", now=NOW, timezone="Europe/Moscow", user_id=42
    )

    assert action_executor.execute_many.await_args is not None
    executed = action_executor.execute_many.await_args.args[0]
    assert executed == [
        ChatAction(type=ActionType.CHAT, text="🧠 Нашёл в памяти:\n• Работает в B")
    ]
    # One decision, one answer turn, one retry: never more.
    assert llm.parse_message.await_count == 3
    # Only the retry is told the search is done; the user's words stay first.
    retry = awaited_kwargs(llm.parse_message)["text"]
    assert retry.startswith("Где я работаю?") and "уже прочитана" in retry


@pytest.mark.asyncio
async def test_an_answer_turn_that_searches_again_is_asked_once_more() -> None:
    """«Посоветуй пиццу» got the raw fact list instead of advice (2026-09-14)."""
    knowledge = service(
        search=AsyncMock(return_value=[KnowledgeFact(fact="Не любит грибы")])
    )
    advice = chat("Возьми Маргариту — в ней нет грибов.")
    process, llm, action_executor = use_case(
        decisions=[searching("пицца еда"), searching("пицца"), advice],
        knowledge=knowledge,
    )

    await process.execute(
        text="Посоветуй пиццу", now=NOW, timezone="Europe/Moscow", user_id=42
    )

    assert llm.parse_message.await_count == 3
    assert "Не любит грибы" in awaited_kwargs(llm.parse_message)["knowledge"]
    assert action_executor.execute_many.await_args.args[0] == advice.actions


@pytest.mark.asyncio
async def test_the_agent_answers_when_memory_is_unavailable() -> None:
    knowledge = service(
        search=AsyncMock(side_effect=KnowledgeMemoryError("Neo4j unavailable"))
    )
    process, llm, _ = use_case(
        decisions=[searching(), chat("Не помню")], knowledge=knowledge
    )

    result = await process.execute(
        text="Где я работаю?", now=NOW, timezone="Europe/Moscow", user_id=42
    )

    assert "unavailable" in awaited_kwargs(llm.parse_message)["knowledge"]
    assert result == "Ответ"


@pytest.mark.asyncio
async def test_search_is_skipped_without_a_configured_memory() -> None:
    process, llm, action_executor = use_case(decisions=[searching()], knowledge=None)

    await process.execute(
        text="Где я работаю?", now=NOW, timezone="Europe/Moscow", user_id=42
    )

    assert llm.parse_message.await_count == 1
    assert action_executor.execute_many.await_args is not None
    assert action_executor.execute_many.await_args.args[0] == searching().actions


# --- forgetting ----------------------------------------------------------------


def forgetting(query: str = "работает в Ozon", *refs: str) -> AssistantDecision:
    return AssistantDecision(
        actions=[
            ForgetKnowledgeAction(
                type=ActionType.FORGET_KNOWLEDGE, query=query, refs=list(refs)
            )
        ]
    )


class ForgettingDriver:
    """A namespace holding one current and one past employer."""

    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []

    async def execute_query(
        self, query: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], None, None]:
        self.queries.append({"query": query, **kwargs})
        rows: list[dict[str, Any]] = []
        if "RETURN fact.fact AS text" in query and kwargs["uuids"]:
            rows = [{"text": "Пользователь работает в Ozon"}]
        elif "DELETE fact" in query and kwargs["texts"]:
            rows = [
                {
                    "text": "Пользователь работает в Ozon",
                    "source": "me",
                    "target": "ozon",
                }
            ]
        elif "DETACH DELETE episode" in query and kwargs["contents"]:
            rows = [{"content": "Пользователь: Пользователь работает в Ozon"}]
        elif "RETURN node.uuid AS uuid, node.summary AS summary" in query:
            rows = [
                {
                    "uuid": "me",
                    "summary": "Пользователь работает в Яндексе\n"
                    "Пользователь работает в Ozon",
                },
                {"uuid": "yandex", "summary": "Пользователь работает в Яндексе"},
            ]
        return rows, None, None

    def matching(self, fragment: str) -> list[dict[str, Any]]:
        return [query for query in self.queries if fragment in query["query"]]


def test_fact_payload_shows_a_ref_only_when_the_fact_can_be_forgotten() -> None:
    fact = KnowledgeFact(fact="Работает в A", ref="edge:1")

    assert fact.as_payload()["ref"] == "edge:1"
    assert "ref" not in KnowledgeFact(fact="Работает в A").as_payload()


@pytest.mark.asyncio
async def test_adapter_hands_refs_to_forgetting_but_not_to_answers() -> None:
    graphiti = FakeGraphiti(
        edges=[
            edge(fact="Пользователь работает в Ozon", group_id="user_abc", episodes=[])
        ]
    )

    candidates = await memory(graphiti).forget_candidates(
        namespace="user_abc", query="Ozon", limit=5
    )
    answers = await memory(graphiti).search(namespace="user_abc", query="Ozon", limit=5)

    assert [fact.ref for fact in candidates] == [f"edge:{graphiti.edges[0].uuid}"]
    assert [fact.ref for fact in answers] == [None]


@pytest.mark.asyncio
async def test_forgetting_erases_every_copy_of_the_fact_and_nothing_else() -> None:
    driver = ForgettingDriver()
    graphiti = FakeGraphiti()
    graphiti.driver = driver  # type: ignore[assignment]

    forgotten = await memory(graphiti).forget(
        namespace="user_abc", refs=["edge:ozon-edge", "bogus"]
    )

    assert forgotten == ["Пользователь работает в Ozon"]
    assert all(query["namespace"] == "user_abc" for query in driver.queries)
    [edges] = driver.matching("DELETE fact")
    assert edges["uuids"] == ["ozon-edge"]
    assert edges["texts"] == ["Пользователь работает в Ozon"]
    [episodes] = driver.matching("DETACH DELETE episode")
    # The episode still carries the speaker prefix remember added.
    assert "Пользователь: Пользователь работает в Ozon" in episodes["contents"]
    # The past employer stays in the summary; only the forgotten line goes.
    [summary] = driver.matching("SET node.summary")
    assert summary["uuid"] == "me"
    assert summary["summary"] == "Пользователь работает в Яндексе"
    [orphans] = driver.matching("DETACH DELETE node")
    assert orphans["uuids"] == ["me", "ozon"]


@pytest.mark.asyncio
async def test_a_selected_prose_summary_is_forgotten_whole() -> None:
    driver = ForgettingDriver()
    graphiti = FakeGraphiti()
    graphiti.driver = driver  # type: ignore[assignment]

    forgotten = await memory(graphiti).forget(
        namespace="user_abc", refs=["node:yandex"]
    )

    assert "Пользователь работает в Яндексе" in forgotten
    assert [query["uuid"] for query in driver.matching("SET node.summary")] == [
        "yandex"
    ]


@pytest.mark.asyncio
async def test_forgetting_reports_backend_failures_as_controlled_errors() -> None:
    graphiti = FakeGraphiti()
    graphiti.driver = SimpleNamespace(
        execute_query=AsyncMock(side_effect=RuntimeError("neo4j down"))
    )

    with pytest.raises(KnowledgeMemoryError):
        await memory(graphiti).forget(namespace="user_abc", refs=["edge:1"])


@pytest.mark.asyncio
async def test_service_forgets_within_the_user_namespace() -> None:
    backend = SimpleNamespace(forget=AsyncMock(return_value=["Работает в Ozon"]))
    knowledge = KnowledgeService(memory=backend)

    assert await knowledge.forget(user_id=42, refs=["edge:1"]) == ["Работает в Ozon"]
    assert awaited_kwargs(backend.forget)["namespace"] == namespace_for(42)

    backend.forget.reset_mock()
    assert await knowledge.forget(user_id=42, refs=[]) == []
    backend.forget.assert_not_awaited()


@pytest.mark.asyncio
async def test_forgetting_lets_the_model_pick_only_recalled_refs() -> None:
    candidates = [
        KnowledgeFact(fact="Пользователь работает в Ozon", ref="edge:ozon"),
        KnowledgeFact(fact="Пользователь работает в Яндексе", ref="edge:yandex"),
    ]
    knowledge = service(forget_candidates=AsyncMock(return_value=candidates))
    process, llm, executed = use_case(
        decisions=[
            forgetting(),
            forgetting("работает в Ozon", "edge:ozon", "edge:someone-else"),
        ],
        knowledge=knowledge,
    )

    await process.execute(
        text="Забудь, что я работаю в Ozon",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    shown = awaited_kwargs(llm.parse_message)["knowledge"]
    assert "edge:yandex" in shown and "edge:ozon" in shown
    # A ref the recall never produced cannot reach the executor.
    assert (
        executed.execute_many.await_args.args[0]
        == forgetting("работает в Ozon", "edge:ozon").actions
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_picks_something_else", [False, True])
async def test_forgetting_nothing_matching_erases_nothing(
    model_picks_something_else: bool,
) -> None:
    found = (
        [KnowledgeFact(fact="Пользователь работает в Яндексе", ref="edge:y")]
        if model_picks_something_else
        else []
    )
    knowledge = service(forget_candidates=AsyncMock(return_value=found))
    process, llm, executed = use_case(
        decisions=[forgetting(), forgetting()], knowledge=knowledge
    )

    await process.execute(
        text="Забудь, что я работаю в Ozon",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert llm.parse_message.await_count == (2 if model_picks_something_else else 1)
    assert executed.execute_many.await_args.args[0] == [
        ChatAction(type=ActionType.CHAT, text="В памяти этого нет — забывать нечего.")
    ]


@pytest.mark.asyncio
async def test_forgetting_degrades_when_memory_is_down() -> None:
    knowledge = service(
        forget_candidates=AsyncMock(side_effect=KnowledgeMemoryError("down"))
    )
    process, _, executed = use_case(decisions=[forgetting()], knowledge=knowledge)

    await process.execute(
        text="Забудь Ozon", now=NOW, timezone="Europe/Moscow", user_id=42
    )

    assert executed.execute_many.await_args.args[0] == [
        ChatAction(type=ActionType.CHAT, text=MEMORY_UNAVAILABLE)
    ]


@pytest.mark.asyncio
async def test_forget_tool_reports_what_it_erased() -> None:
    erase = AsyncMock(side_effect=[["Пользователь работает в Ozon"], []])
    tool = executor(knowledge=service(forget=erase))
    action = forgetting("работает в Ozon", "edge:ozon").actions[0]

    assert await tool.execute(action, user_id=42, now=NOW) == (
        "🧹 Забыл:\n• Пользователь работает в Ozon"
    )
    assert awaited_kwargs(erase)["namespace"] == namespace_for(42)
    assert await tool.execute(action, user_id=42, now=NOW) == (
        "В памяти этого нет — забывать нечего."
    )
    disabled = await executor(knowledge=None).execute(action, user_id=42, now=NOW)
    assert "отключена" in str(disabled)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_picks", [("edge:yandex",), ()])
async def test_forgetting_never_drops_the_fact_stated_alongside(
    model_picks: tuple[str, ...],
) -> None:
    """«Уволился из Яндекса, теперь в Ozon» came as forget + remember.

    Resolving the forget returned only the forget, the Ozon fact was never
    stored, and the next «Где я работаю?» found nothing (eval, 2026-09-14).
    """
    remember = RememberKnowledgeAction(
        type=ActionType.REMEMBER_KNOWLEDGE, content="Пользователь работает в Ozon"
    )
    first = AssistantDecision(
        actions=[*forgetting("работает в Яндексе").actions, remember]
    )
    knowledge = service(
        forget_candidates=AsyncMock(
            return_value=[
                KnowledgeFact(fact="Пользователь работает в Яндексе", ref="edge:yandex")
            ]
        )
    )
    process, _, executed = use_case(
        decisions=[first, forgetting("работает в Яндексе", *model_picks)],
        knowledge=knowledge,
    )

    await process.execute(
        text="Я уволился из Яндекса и теперь работаю в Ozon",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    expected = (
        [*forgetting("работает в Яндексе", *model_picks).actions, remember]
        if model_picks
        else [remember]
    )
    assert executed.execute_many.await_args.args[0] == expected


@pytest.mark.asyncio
async def test_an_admitted_ignorance_triggers_the_search_the_model_skipped() -> None:
    """«Я пока не знаю, куда мы ездили» came with the trip stored (eval, 2026-09-14)."""
    trip = [KnowledgeFact(fact="Пользователь ездил в Выборг с Машей в субботу")]
    search = AsyncMock(return_value=trip)
    knowledge = service(search=search)
    process, llm, executed = use_case(
        decisions=[
            chat(
                "Я пока не знаю, куда мы ездили на выходных. Расскажите, и я запомню."
            ),
            chat("В Выборг."),
        ],
        knowledge=knowledge,
    )

    await process.execute(
        text="Куда мы ездили на выходных?",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert awaited_kwargs(search)["query"] == "Куда мы ездили на выходных?"
    assert "Выборг" in awaited_kwargs(llm.parse_message)["knowledge"]
    assert executed.execute_many.await_args.args[0] == chat("В Выборг.").actions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "configured"),
    [("51", True), ("Не знаю, ты не говорил.", False)],
)
async def test_a_confident_chat_or_a_memoryless_bot_is_left_alone(
    reply: str, configured: bool
) -> None:
    knowledge = service(search=AsyncMock(return_value=[])) if configured else None
    process, llm, executed = use_case(decisions=[chat(reply)], knowledge=knowledge)

    await process.execute(
        text="Сколько будет 17 умножить на 3?",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert llm.parse_message.await_count == 1
    assert executed.execute_many.await_args.args[0] == chat(reply).actions


@pytest.mark.asyncio
async def test_an_empty_candidate_list_answers_instead_of_failing_validation() -> None:
    """A model slip must not cost the whole reply (memory eval, 2026-09-17)."""
    decision = AssistantDecision.model_validate(
        {"actions": [{"type": "recommend_films", "candidates": [], "limit": 10}]}
    )
    knowledge = service(watched_film_titles=AsyncMock(return_value=["Начало"]))

    reply = await executor(knowledge=knowledge).execute(
        decision.actions[0], user_id=42, now=NOW
    )

    assert isinstance(reply, str) and "уже всё смотрел" in reply


def _numbered(*items: str) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


@pytest.mark.asyncio
async def test_a_long_list_is_reviewed_against_the_recalled_facts() -> None:
    """The model wrote «Рамен с грибами (без грибов)» whatever the prompt said."""
    knowledge = service(
        search=AsyncMock(return_value=[KnowledgeFact(fact="Не любит грибы")])
    )
    draft = _numbered("Рамен с грибами", "Том ям", "Удон", "Гёдза", "Рис")
    fixed = _numbered("Рамен с тофу", "Том ям", "Удон", "Гёдза", "Рис")
    process, llm, executed = use_case(
        decisions=[searching(), chat(draft), chat(fixed)], knowledge=knowledge
    )

    await process.execute(
        text="Предложи 5 блюд без того, что я не люблю",
        now=NOW,
        timezone="Europe/Moscow",
        user_id=42,
    )

    assert executed.execute_many.await_args.args[0] == chat(fixed).actions
    # Decision, answer, review — the draft is never sent to the user.
    assert llm.parse_message.await_count == 3
    reviewed = awaited_kwargs(llm.parse_message)["text"]
    assert draft in reviewed and "черновик" in reviewed


@pytest.mark.asyncio
async def test_a_review_that_shortens_the_list_is_discarded() -> None:
    knowledge = service(search=AsyncMock(return_value=[KnowledgeFact(fact="Факт")]))
    draft = _numbered("Раз", "Два", "Три", "Четыре", "Пять")
    process, llm, executed = use_case(
        decisions=[searching(), chat(draft), chat(_numbered("Раз", "Два"))],
        knowledge=knowledge,
    )

    await process.execute(
        text="Назови 5 идей", now=NOW, timezone="Europe/Moscow", user_id=42
    )

    assert executed.execute_many.await_args.args[0] == chat(draft).actions


@pytest.mark.asyncio
async def test_a_short_answer_is_not_reviewed() -> None:
    knowledge = service(search=AsyncMock(return_value=[KnowledgeFact(fact="Джек")]))
    process, llm, executed = use_case(
        decisions=[searching(), chat("Твою собаку зовут Джек.")], knowledge=knowledge
    )

    await process.execute(
        text="Как зовут мою собаку?", now=NOW, timezone="Europe/Moscow", user_id=42
    )

    assert llm.parse_message.await_count == 2
    assert (
        executed.execute_many.await_args.args[0]
        == chat("Твою собаку зовут Джек.").actions
    )
