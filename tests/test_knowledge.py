import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from cryptography.fernet import Fernet
from graphiti_core.nodes import EpisodeType
from pydantic import ValidationError

from src.application.ports.knowledge import KnowledgeMemoryError
from src.application.services.action_executor import (
    MEMORY_UNAVAILABLE,
    ActionExecutor,
)
from src.application.services.knowledge import KnowledgeService
from src.application.use_cases.process_message import ProcessMessageUseCase
from src.config import Settings
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    AssistantDecision,
    ChatAction,
    RememberKnowledgeAction,
    SaveNoteAction,
    SearchKnowledgeAction,
)
from src.domain.knowledge.models import (
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
        fact=fact,
        group_id=group_id,
        episodes=episodes if episodes is not None else ["episode-1"],
        valid_at=valid_at,
        invalid_at=invalid_at,
    )


def episodic(uuid: str, *, name: str, group_id: str) -> SimpleNamespace:
    return SimpleNamespace(uuid=uuid, name=name, group_id=group_id)


def found_episode(
    *,
    content: str = "Пользователь: родился 02.05.2003 в г. Красноярск",
    group_id: str,
    name: str = "telegram_message:627",
    valid_at: datetime | None = NOW,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, group_id=group_id, name=name, valid_at=valid_at
    )


def entity(
    *,
    name: str = "Красноярск",
    group_id: str,
    summary: str | None = "Красноярск — город рождения Андрея (род. 02.05.2003).",
) -> SimpleNamespace:
    return SimpleNamespace(name=name, group_id=group_id, summary=summary)


class FakeGraphiti:
    def __init__(
        self,
        *,
        edges: list[SimpleNamespace] | None = None,
        nodes: list[SimpleNamespace] | None = None,
        found_episodes: list[SimpleNamespace] | None = None,
        latest: list[dict[str, Any]] | None = None,
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
        self.latest = latest or []
        self.driver = SimpleNamespace(
            execute_query=AsyncMock(return_value=(self.latest, None, None))
        )

    async def build_indices_and_constraints(self) -> None:
        if self.error:
            raise self.error
        self.indices_built = True

    async def add_episode(self, **kwargs: Any) -> None:
        if self.error:
            raise self.error
        self.episodes.append(kwargs)

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
        "Красноярск — город рождения Андрея (род. 02.05.2003).",
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
        "Красноярск — город рождения Андрея (род. 02.05.2003).",
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
            fact="родился 02.05.2003 в г. Красноярск",
            valid_from=NOW,
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
        found_episodes=[found_episode(content="Работает с Python", group_id="user_abc")],
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=5)

    assert [fact.fact for fact in facts] == ["Работает с Python"]


@pytest.mark.asyncio
async def test_adapter_tops_up_a_thin_recall_with_the_newest_episodes() -> None:
    """Episode search is BM25: «дата рождения» never matches «родился».

    The stored sentence is the only copy of a fact extraction dropped, so a
    recall with room left over falls back to what the user recently said.
    """
    graphiti = FakeGraphiti(
        edges=[edge(fact="Работает с Python", group_id="user_abc", episodes=[])],
        latest=[
            {
                "name": "telegram_message:627",
                "content": "Пользователь: родился 02.05.2003",
                "valid_at": NOW,
            }
        ],
    )

    facts = await memory(graphiti).search(
        namespace="user_abc", query="дата рождения", limit=5
    )

    assert [fact.fact for fact in facts] == [
        "Работает с Python",
        "родился 02.05.2003",
    ]
    assert graphiti.driver.execute_query.await_args is not None
    assert graphiti.driver.execute_query.await_args.kwargs["namespace"] == "user_abc"


@pytest.mark.asyncio
async def test_adapter_skips_the_top_up_when_the_search_filled_the_limit() -> None:
    graphiti = FakeGraphiti(
        edges=[
            edge(fact=f"Факт {index}", group_id="user_abc", episodes=[])
            for index in range(3)
        ],
        latest=[{"name": "n", "content": "Лишнее", "valid_at": NOW}],
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=3)

    assert [fact.fact for fact in facts] == ["Факт 0", "Факт 1", "Факт 2"]
    graphiti.driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_adapter_never_repeats_a_fact_in_the_top_up() -> None:
    graphiti = FakeGraphiti(
        found_episodes=[found_episode(content="Своё", group_id="user_abc")],
        latest=[
            {"name": "telegram_message:627", "content": "Своё", "valid_at": NOW},
            {"name": "telegram_message:628", "content": "Другое", "valid_at": NOW},
        ],
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=5)

    assert [fact.fact for fact in facts] == ["Своё", "Другое"]


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
        nodes=[entity(name="Красноярск", group_id="user_abc", summary=None)]
    )

    facts = await memory(graphiti).search(namespace="user_abc", query="x", limit=5)

    assert [fact.fact for fact in facts] == ["Красноярск"]


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
) -> tuple[ProcessMessageUseCase, SimpleNamespace, SimpleNamespace]:
    llm = SimpleNamespace(parse_message=AsyncMock(side_effect=decisions))
    action_executor = SimpleNamespace(execute_many=AsyncMock(return_value="Ответ"))
    return (
        ProcessMessageUseCase(
            llm=llm,
            action_executor=action_executor,  # type: ignore[arg-type]
            knowledge=knowledge,
        ),
        llm,
        action_executor,
    )


def chat(text: str) -> AssistantDecision:
    return AssistantDecision(actions=[ChatAction(type=ActionType.CHAT, text=text)])


def searching(query: str = "фильм") -> AssistantDecision:
    return AssistantDecision(
        actions=[SearchKnowledgeAction(type=ActionType.SEARCH_KNOWLEDGE, query=query)]
    )


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
async def test_a_second_search_never_loops() -> None:
    knowledge = service(
        search=AsyncMock(return_value=[KnowledgeFact(fact="Работает в B")])
    )
    process, _, action_executor = use_case(
        decisions=[searching(), searching()], knowledge=knowledge
    )

    await process.execute(
        text="Где я работаю?", now=NOW, timezone="Europe/Moscow", user_id=42
    )

    assert action_executor.execute_many.await_args is not None
    executed = action_executor.execute_many.await_args.args[0]
    assert executed == [
        ChatAction(type=ActionType.CHAT, text="🧠 Нашёл в памяти:\n• Работает в B")
    ]


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
