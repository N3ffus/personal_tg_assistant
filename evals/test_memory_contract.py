"""Offline checks of the memory eval harness: no LLM, Neo4j or judge.

A scripted bot remembers every statement and answers every question with what
memory returned. That is enough to exercise history windows, restarts, new
chats and user isolation, and to prove the hard checks catch a broken run.
"""

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import pytest

from evals.harness import RecordedCall
from evals.memory_harness import (
    PERSONA_USER_ID,
    Backend,
    MemoryRun,
    TurnResult,
    assert_memory_contract,
    ensure_persona,
    list_items,
    run_memory_scenario,
    transcript,
    user_id_for,
)
from evals.memory_scenarios import (
    MEMORY_NOW,
    MEMORY_SCENARIOS,
    PERSONA,
    MemoryScenario,
    Turn,
)
from src.domain.assistant.context import MAX_MESSAGES
from src.domain.assistant.models import AssistantDecision
from src.domain.knowledge.models import KnowledgeEpisode, KnowledgeFact, namespace_for

BY_ID = {scenario.id: scenario for scenario in MEMORY_SCENARIOS}


class ScriptedLLM:
    async def parse_message(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
        context: str = "",
        knowledge: str = "",
    ) -> AssistantDecision:
        if knowledge:
            facts = [fact["fact"] for fact in json.loads(knowledge)]
            action = {"type": "chat", "text": "Из памяти: " + "; ".join(facts)}
        elif text.endswith("?"):
            action = {"type": "search_knowledge", "query": text}
        else:
            action = {"type": "remember_knowledge", "content": text}
        return AssistantDecision.model_validate({"actions": [action]})


@dataclass
class Graph:
    """What survives a restart: the database, not the clients."""

    episodes: dict[str, list[str]] = field(default_factory=dict)
    opened: int = 0


@dataclass
class FakeMemory:
    graph: Graph

    async def remember(self, *, namespace: str, episode: KnowledgeEpisode) -> None:
        self.graph.episodes.setdefault(namespace, []).append(episode.content)

    async def search(
        self, *, namespace: str, query: str, limit: int
    ) -> list[KnowledgeFact]:
        found = self.graph.episodes.get(namespace, [])
        return [KnowledgeFact(fact=content) for content in found][:limit]

    async def healthcheck(self) -> bool:
        return True

    async def recent_watched_films(
        self, *, namespace: str, limit: int
    ) -> list[KnowledgeFact]:
        return []

    async def close(self) -> None:
        return None

    async def watched_film_titles(self, *, namespace: str) -> list[str]:
        return []

    async def profile_facts(self, *, namespace: str, limit: int) -> list[KnowledgeFact]:
        found = self.graph.episodes.get(namespace, [])
        return [KnowledgeFact(fact=content) for content in found][-limit:]

    async def forget_candidates(
        self, *, namespace: str, query: str, limit: int
    ) -> list[KnowledgeFact]:
        found = self.graph.episodes.get(namespace, [])
        return [
            KnowledgeFact(fact=content, ref=str(index))
            for index, content in enumerate(found)
        ][:limit]

    async def forget(self, *, namespace: str, refs: list[str]) -> list[str]:
        found = self.graph.episodes.get(namespace, [])
        erased = [found[int(ref)] for ref in refs]
        self.graph.episodes[namespace] = [c for c in found if c not in erased]
        return erased


@dataclass
class FakeProcess:
    graph: Graph
    llm: ScriptedLLM = field(default_factory=ScriptedLLM)

    @property
    def memory(self) -> FakeMemory:
        return FakeMemory(self.graph)

    async def entity_names(self, *, namespace: str) -> list[str]:
        return list(self.graph.episodes.get(namespace, []))

    async def episode_contents(self, *, namespace: str) -> list[str]:
        return list(self.graph.episodes.get(namespace, []))

    async def wipe(self, *, namespace: str) -> None:
        self.graph.episodes.pop(namespace, None)

    async def close(self) -> None:
        return None


def backend_for(graph: Graph) -> Backend:
    async def open_process() -> FakeProcess:
        graph.opened += 1
        return FakeProcess(graph)

    return open_process


def test_scenarios_are_well_formed() -> None:
    ids = [scenario.id for scenario in MEMORY_SCENARIOS]
    assert len(ids) == len(set(ids))
    own = [scenario for scenario in MEMORY_SCENARIOS if not scenario.persona]
    owners = [user_id_for(scenario, "owner") for scenario in own]
    strangers = [user_id_for(scenario, "stranger") for scenario in MEMORY_SCENARIOS]
    assert len(set(owners) | set(strangers)) == len(own) + len(MEMORY_SCENARIOS)
    assert PERSONA_USER_ID not in owners + strangers
    stated = [memory.stated for memory in PERSONA]
    assert stated == sorted(stated), "the persona must be seeded in order"
    assert all(memory.stated < f"{MEMORY_NOW:%Y-%m-%d}" for memory in PERSONA)
    for scenario in MEMORY_SCENARIOS:
        assert scenario.setup or scenario.persona, f"{scenario.id}: nothing to remember"
        if scenario.persona:
            assert len(scenario.turns) == 1, f"{scenario.id}: persona asks one question"
        assert scenario.expected.strip()
        days = [turn.day for turn in scenario.turns]
        assert days == sorted(days), f"{scenario.id}: turns go back in time"
        assert all(turn.user == "owner" for turn in scenario.setup)
        if scenario.probe_contains or scenario.probe_excludes:
            assert scenario.probe_query, f"{scenario.id}: a probe needs a query"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_id",
    [
        "a_fact_outlives_the_conversation_history",
        "a_fact_survives_a_backend_restart",
        "a_fact_is_available_in_another_conversation",
        "memory_never_leaks_between_users",
    ],
)
async def test_an_echo_bot_satisfies_the_contract(
    scenario_id: str, tmp_path: Path
) -> None:
    scenario = BY_ID[scenario_id]
    graph = Graph()

    run = await run_memory_scenario(
        scenario, backend_for(graph), database=tmp_path / "context.db"
    )

    assert_memory_contract(scenario, run)
    assert graph.opened == 1 + sum(turn.restart for turn in scenario.turns)
    assert [result.turn for result in run.turns] == list(scenario.turns)
    assert scenario.question.text in transcript(run)


@pytest.mark.asyncio
async def test_a_fact_still_in_history_proves_nothing(tmp_path: Path) -> None:
    scenario = replace(
        BY_ID["a_fact_outlives_the_conversation_history"], history_window=MAX_MESSAGES
    )

    run = await run_memory_scenario(
        scenario, backend_for(Graph()), database=tmp_path / "context.db"
    )

    assert "Я инженер-гидролог" in run.answer.history
    with pytest.raises(AssertionError, match="still in the chat history"):
        assert_memory_contract(scenario, run)


@pytest.mark.asyncio
async def test_another_users_fact_in_the_reply_is_caught(tmp_path: Path) -> None:
    scenario = BY_ID["memory_never_leaks_between_users"]
    run = await run_memory_scenario(
        scenario, backend_for(Graph()), database=tmp_path / "context.db"
    )
    run.answer.reply = "Твоего кота зовут Шуршик."

    with pytest.raises(AssertionError, match="шуршик"):
        assert_memory_contract(scenario, run)


@pytest.mark.asyncio
async def test_reruns_start_from_an_empty_namespace(tmp_path: Path) -> None:
    scenario: MemoryScenario = BY_ID["a_fact_is_available_in_another_conversation"]
    graph = Graph()

    for attempt in range(2):
        await run_memory_scenario(
            scenario, backend_for(graph), database=tmp_path / f"{attempt}.db"
        )

    assert sum(len(found) for found in graph.episodes.values()) == 1


@pytest.mark.asyncio
async def test_the_persona_is_seeded_once_and_reseeded_after_a_write(
    tmp_path: Path,
) -> None:
    graph = Graph()
    namespace = namespace_for(PERSONA_USER_ID)

    assert await ensure_persona(backend_for(graph)) is True
    assert await ensure_persona(backend_for(graph)) is False
    assert graph.episodes[namespace] == [memory.text for memory in PERSONA]

    # The echo bot remembers the question: the shared persona is now dirty.
    scenario = replace(
        BY_ID["the_newest_weight_is_current"], turns=(Turn("Я люблю грибы"),)
    )
    await run_memory_scenario(scenario, backend_for(graph), database=tmp_path / "c.db")

    assert namespace not in graph.episodes
    assert await ensure_persona(backend_for(graph)) is True


def test_list_items_count_top_level_entries_only() -> None:
    reply = "Вот идеи:\n1. **Рамен** — остро\n   - без грибов\n2) Том ям\n\nИтого два."

    assert list_items(reply) == ["**Рамен** — остро", "Том ям"]
    assert list_items("- Норвиния\n- Эстрелия") == ["Норвиния", "Эстрелия"]
    assert list_items("Вот: 1. Рамен, 2. Том ям; 3) Удон.") == [
        "Рамен",
        "Том ям",
        "Удон.",
    ]
    # A year ending a sentence does not start a numbered list.
    assert list_items("Был там в 2039. Потом ещё раз.") == []


def _answered(scenario: MemoryScenario, reply: str, tmp_path: Path) -> MemoryRun:
    return MemoryRun(
        turns=[
            TurnResult(
                turn=scenario.question,
                user_id=user_id_for(scenario, "owner"),
                now=MEMORY_NOW,
                history="",
                reply=reply,
                writes=[],
                lookups=[
                    RecordedCall(
                        "knowledge.search",
                        {"namespace": namespace_for(user_id_for(scenario, "owner"))},
                    )
                ],
            )
        ],
        probe=[],
        entities=[],
        persona=True,
    )


def test_a_wrong_item_count_is_caught(tmp_path: Path) -> None:
    scenario = BY_ID["a_request_outranks_a_stored_dislike"]
    four = "\n".join(f"{index}. Грибной суп" for index in range(1, 5))

    with pytest.raises(AssertionError, match="4 list items instead of 5"):
        assert_memory_contract(scenario, _answered(scenario, four, tmp_path))
    assert_memory_contract(
        scenario, _answered(scenario, four + "\n5. Жульен с грибами", tmp_path)
    )


def test_an_excluded_item_is_caught_but_a_negation_is_not(tmp_path: Path) -> None:
    scenario = replace(BY_ID["ten_dishes_skip_every_dislike"], item_count=2)

    assert_memory_contract(
        scenario, _answered(scenario, "1. Рамен без грибов\n2. Том ям", tmp_path)
    )
    with pytest.raises(AssertionError, match="гриб"):
        assert_memory_contract(
            scenario, _answered(scenario, "1. Рамен\n2. Грибной суп", tmp_path)
        )
    dates = replace(BY_ID["ten_date_ideas_cross_several_memories"], item_count=2)
    assert_memory_contract(
        dates, _answered(dates, "1. Керамика — без театра и бара\n2. Каток", tmp_path)
    )
    # A preposition between the negation and the noun still negates it:
    # «не в баре» promises to avoid bars, it does not suggest one.
    prepositions = replace(dates, item_count=3)
    assert_memory_contract(
        prepositions,
        _answered(
            prepositions,
            "1. Квиз — проходят в кафе, не в баре\n"
            "2. Прогулка без похода в театр\n"
            "3. Каток вместо посиделок в баре",
            tmp_path,
        ),
    )


def test_order_and_repeated_mentions_are_enforced(tmp_path: Path) -> None:
    chronology = BY_ID["a_trip_chronology_keeps_approximate_dates"]
    in_order = (
        "Туманный Мыс (примерно 2025), Серебряное, Кедровая Падь, Каменная Гавань, "
        "Зарянск, Ветрогорье, Ледяные Озёра"
    )
    swapped = (
        "Серебряное, Туманный Мыс (примерно 2025), Кедровая Падь, Каменная Гавань, "
        "Зарянск, Ветрогорье, Ледяные Озёра"
    )

    assert_memory_contract(chronology, _answered(chronology, in_order, tmp_path))
    with pytest.raises(AssertionError, match="out of order"):
        assert_memory_contract(chronology, _answered(chronology, swapped, tmp_path))

    countries = BY_ID["wished_countries_are_listed_once"]
    with pytest.raises(AssertionError, match="appears 2 times"):
        assert_memory_contract(
            countries,
            _answered(countries, "Япония, Исландия, Грузия и снова Япония", tmp_path),
        )
