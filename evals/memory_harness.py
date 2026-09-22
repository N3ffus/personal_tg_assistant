"""Drive a multi-turn memory scenario through the bot the way Telegram does.

A scenario runs against a *backend*: something that opens one bot process —
an LLM client plus a knowledge memory — and can open it again after a restart.
The live eval opens the configured LLM and Graphiti on Neo4j; the offline contract test
opens fakes. Chat history lives in a real SQLite ``ContextStorage`` that
outlives restarts, exactly as the production volume does.
"""

import re
import zlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from cryptography.fernet import Fernet

from evals.harness import RecordedCall, RecordingIntegrations
from evals.memory_scenarios import MEMORY_NOW, PERSONA, MemoryScenario, Turn
from evals.scenarios import Scenario
from src.application.ports.knowledge import KnowledgeMemory
from src.application.ports.llm import LLMClient
from src.application.services.action_executor import ActionExecutor
from src.application.services.context import ContextService
from src.application.services.knowledge import KnowledgeService
from src.application.use_cases.process_message import ProcessMessageUseCase
from src.domain.assistant.context import ChatContext, ContextChat, ContextMessage
from src.domain.assistant.replies import AssistantReply
from src.domain.knowledge.models import (
    KnowledgeEpisode,
    KnowledgeFact,
    KnowledgeSourceType,
    namespace_for,
)
from src.infrastructure.context.storage import ContextStorage

TIMEZONE = "Europe/Moscow"
# Far above real Telegram ids used in development, and distinct from the ids of
# tests/integration, so wiping a scenario namespace never touches real memory.
FIRST_EVAL_USER_ID = 900_300_000
# Below every per-scenario id, so no scenario wipe can reach the shared persona.
PERSONA_USER_ID = FIRST_EVAL_USER_ID - 1
FILLER = (
    ("Сколько будет 12 умножить на 7?", "84"),
    ("Переведи на английский «доброе утро»", "Good morning"),
    ("Сколько минут в трёх часах?", "180"),
    ("Как пишется «в течение»?", "Слитно «в», раздельно «течение»: в течение."),
)


class BotProcess(Protocol):
    """One running bot: the clients a restart throws away."""

    @property
    def llm(self) -> LLMClient: ...

    @property
    def memory(self) -> KnowledgeMemory: ...

    async def entity_names(self, *, namespace: str) -> list[str]: ...

    async def episode_contents(self, *, namespace: str) -> list[str]: ...

    async def wipe(self, *, namespace: str) -> None: ...

    async def close(self) -> None: ...


Backend = Callable[[], Awaitable[BotProcess]]


def user_id_for(scenario: MemoryScenario, user: Literal["owner", "stranger"]) -> int:
    """A stable per-scenario identity, so reruns wipe and reuse one namespace."""
    if scenario.persona and user == "owner":
        return PERSONA_USER_ID
    offset = zlib.crc32(scenario.id.encode()) % 100_000 * 2
    return FIRST_EVAL_USER_ID + offset + (1 if user == "stranger" else 0)


@dataclass
class RecordingMemory:
    """Pass memory through while noting which namespace read or wrote what."""

    inner: KnowledgeMemory
    writes: list[RecordedCall] = field(default_factory=list)
    lookups: list[RecordedCall] = field(default_factory=list)
    erasures: list[RecordedCall] = field(default_factory=list)

    async def remember(self, *, namespace: str, episode: KnowledgeEpisode) -> None:
        self.writes.append(
            RecordedCall(
                "knowledge.remember",
                {"namespace": namespace, "content": episode.content},
            )
        )
        await self.inner.remember(namespace=namespace, episode=episode)

    async def search(
        self, *, namespace: str, query: str, limit: int
    ) -> list[KnowledgeFact]:
        call = RecordedCall(
            "knowledge.search",
            {"namespace": namespace, "query": query, "limit": limit},
        )
        self.lookups.append(call)
        facts = await self.inner.search(namespace=namespace, query=query, limit=limit)
        call.output = [_rendered(fact) for fact in facts]
        return facts

    async def healthcheck(self) -> bool:
        return await self.inner.healthcheck()

    async def recent_watched_films(
        self, *, namespace: str, limit: int
    ) -> list[KnowledgeFact]:
        call = RecordedCall(
            "knowledge.recent_watched_films", {"namespace": namespace, "limit": limit}
        )
        self.lookups.append(call)
        facts = await self.inner.recent_watched_films(namespace=namespace, limit=limit)
        call.output = [_rendered(fact) for fact in facts]
        return facts

    async def close(self) -> None:
        await self.inner.close()

    async def watched_film_titles(self, *, namespace: str) -> list[str]:
        call = RecordedCall("knowledge.watched_film_titles", {"namespace": namespace})
        self.lookups.append(call)
        titles = await self.inner.watched_film_titles(namespace=namespace)
        call.output = titles
        return titles

    async def profile_facts(self, *, namespace: str, limit: int) -> list[KnowledgeFact]:
        call = RecordedCall(
            "knowledge.profile_facts", {"namespace": namespace, "limit": limit}
        )
        self.lookups.append(call)
        facts = await self.inner.profile_facts(namespace=namespace, limit=limit)
        call.output = [_rendered(fact) for fact in facts]
        return facts

    async def forget_candidates(
        self, *, namespace: str, query: str, limit: int
    ) -> list[KnowledgeFact]:
        call = RecordedCall(
            "knowledge.forget_candidates",
            {"namespace": namespace, "query": query, "limit": limit},
        )
        self.lookups.append(call)
        facts = await self.inner.forget_candidates(
            namespace=namespace, query=query, limit=limit
        )
        call.output = [_rendered(fact) for fact in facts]
        return facts

    async def forget(self, *, namespace: str, refs: list[str]) -> list[str]:
        call = RecordedCall("knowledge.forget", {"namespace": namespace, "refs": refs})
        self.erasures.append(call)
        forgotten = await self.inner.forget(namespace=namespace, refs=refs)
        call.output = forgotten
        return forgotten


def _rendered(fact: KnowledgeFact) -> str:
    if fact.valid_from is None and fact.valid_until is None:
        return fact.fact
    period = f"{fact.valid_from:%d.%m.%Y}" if fact.valid_from else "?"
    if fact.valid_until:
        period += f"–{fact.valid_until:%d.%m.%Y}"
    return f"{fact.fact} (действует с {period})"


class WindowedContextStorage:
    """Production storage that keeps only the newest ``window`` messages.

    ``ChatContext`` holds 60 messages; replaying that many turns per scenario
    would cost minutes. A smaller window pushes a fact out of the history after
    a few filler exchanges, which is the property under test.
    """

    def __init__(self, *, inner: ContextStorage, window: int) -> None:
        self._inner = inner
        self._window = window

    async def ensure_chat(
        self,
        *,
        owner_id: int,
        kind: Literal["bot", "business"],
        chat_id: int,
        title: str,
    ) -> ContextChat:
        return await self._inner.ensure_chat(
            owner_id=owner_id, kind=kind, chat_id=chat_id, title=title
        )

    async def list_chats(self, *, owner_id: int) -> list[ContextChat]:
        return await self._inner.list_chats(owner_id=owner_id)

    async def load(self, *, owner_id: int, context_id: int) -> ChatContext | None:
        return await self._inner.load(owner_id=owner_id, context_id=context_id)

    async def save(
        self, *, owner_id: int, context_id: int, context: ChatContext
    ) -> None:
        overflow = context.messages[: -self._window]
        if overflow:
            del context.messages[: len(overflow)]
            context.discarded_message_id = max(
                context.discarded_message_id,
                *(message.message_id or 0 for message in overflow),
            )
        await self._inner.save(
            owner_id=owner_id, context_id=context_id, context=context
        )


class UnusedSummarizer:
    async def summarize(self, *, text: str) -> str:
        raise AssertionError("Memory scenarios never compact a chat")


@dataclass
class TurnResult:
    turn: Turn
    user_id: int
    now: datetime
    # The history the bot was given for this turn, as the LLM saw it.
    history: str
    reply: str
    writes: list[RecordedCall]
    lookups: list[RecordedCall]


@dataclass
class MemoryRun:
    turns: list[TurnResult]
    probe: list[str]
    entities: list[str]
    persona: bool = False

    @property
    def answer(self) -> TurnResult:
        return self.turns[-1]


class _Session:
    def __init__(
        self, scenario: MemoryScenario, backend: Backend, database: Path
    ) -> None:
        self._scenario = scenario
        self._backend = backend
        self._storage = ContextStorage(
            database_path=str(database),
            encryption_key=Fernet.generate_key().decode(),
        )
        self._contexts = ContextService(
            storage=WindowedContextStorage(
                inner=self._storage, window=scenario.history_window
            ),
            summarizer=UnusedSummarizer(),
        )
        self._message_id = 0
        self.process: BotProcess | None = None
        self.memory: RecordingMemory | None = None

    async def open(self) -> None:
        # Idempotent: the SQLite history survives restarts like the volume does.
        await self._storage.initialize()
        self.process = await self._backend()
        self.memory = RecordingMemory(self.process.memory)

    async def close(self) -> None:
        if self.process is not None:
            await self.process.close()
            self.process = None

    def _next_message_id(self) -> int:
        self._message_id += 1
        return self._message_id

    async def send(self, turn: Turn) -> TurnResult:
        if turn.restart:
            await self.close()
            await self.open()
        assert self.process is not None and self.memory is not None
        user_id = user_id_for(self._scenario, turn.user)
        chat_id = user_id * 10 + turn.chat
        now = MEMORY_NOW + timedelta(days=turn.day)
        # The use case creates the same chat row; its id is needed first for
        # filler and /clear.
        context_id = (
            await self._contexts.ensure_chat(
                owner_id=user_id, kind="bot", chat_id=chat_id, title="Чат с ботом"
            )
        ).id
        if turn.clear_history:
            await self._contexts.clear(owner_id=user_id, context_id=context_id)
        for index in range(turn.filler):
            question, answer = FILLER[index % len(FILLER)]
            sent_at = now - timedelta(minutes=2 * (turn.filler - index))
            for role, sender, text in (
                ("user", "Вы", question),
                ("assistant", "Бот", answer),
            ):
                await self._contexts.record(
                    owner_id=user_id,
                    context_id=context_id,
                    message=ContextMessage(
                        role=role,  # type: ignore[arg-type]
                        sender=sender,
                        text=text,
                        sent_at=sent_at,
                        message_id=self._next_message_id(),
                    ),
                )
        history = (
            await self._contexts.read(owner_id=user_id, context_id=context_id)
        ).render()

        knowledge = KnowledgeService(memory=self.memory)
        use_case = ProcessMessageUseCase(
            llm=self.process.llm,
            action_executor=ActionExecutor(
                calendar=_OFFLINE_INTEGRATIONS,
                pending_operations=_OFFLINE_INTEGRATIONS,
                task_tracker=_OFFLINE_INTEGRATIONS,
                knowledge=knowledge,
            ),
            contexts=self._contexts,
            knowledge=knowledge,
        )
        writes, lookups = len(self.memory.writes), len(self.memory.lookups)
        reply = await use_case.execute(
            text=turn.text,
            now=now,
            timezone=TIMEZONE,
            user_id=user_id,
            chat_id=chat_id,
            message_id=self._next_message_id(),
        )
        # The bot's reply shares the user's message id space in Telegram.
        self._next_message_id()
        return TurnResult(
            turn=turn,
            user_id=user_id,
            now=now,
            history=history,
            reply=_text(reply),
            writes=self.memory.writes[writes:],
            lookups=self.memory.lookups[lookups:],
        )


# Memory scenarios never reach Linear or Calendar; the recording fakes turn an
# accidental task or event into a visible call instead of a crash.
_OFFLINE_INTEGRATIONS = RecordingIntegrations(
    Scenario("memory", "", (), (), "", user_id=FIRST_EVAL_USER_ID)
)


def _text(reply: str | AssistantReply) -> str:
    if isinstance(reply, str):
        return reply
    return "\n\n".join(
        [
            reply.text,
            *(confirmation.text for confirmation in reply.confirmations),
            *(page.text for page in reply.pages),
        ]
    ).strip()


def persona_episodes() -> list[KnowledgeEpisode]:
    return [
        KnowledgeEpisode(
            content=memory.text,
            source_id=f"persona-{index}",
            source_type=KnowledgeSourceType.TELEGRAM_MESSAGE,
            reference_time=datetime.fromisoformat(f"{memory.stated}T12:00:00+03:00"),
        )
        for index, memory in enumerate(PERSONA)
    ]


async def ensure_persona(backend: Backend) -> bool:
    """Write PERSONA into its namespace unless it already holds exactly that.

    Extracting fifty facts takes many minutes, so an intact persona is reused
    across runs. Anything else — a partial seed, a changed persona, a fact a
    scenario wrote — wipes it and seeds again. Returns whether it seeded.
    """
    namespace = namespace_for(PERSONA_USER_ID)
    episodes = persona_episodes()
    process = await backend()
    try:
        stored = await process.episode_contents(namespace=namespace)
        # The adapter prefixes the speaker; compare the user's own words.
        if sorted(text.removeprefix("Пользователь: ") for text in stored) == sorted(
            episode.content for episode in episodes
        ):
            return False
        await process.wipe(namespace=namespace)
        # In order: a later weight or employer must reach Graphiti as an update.
        for episode in episodes:
            await process.memory.remember(namespace=namespace, episode=episode)
        return True
    finally:
        await process.close()


async def run_memory_scenario(
    scenario: MemoryScenario, backend: Backend, *, database: Path
) -> MemoryRun:
    """Play every turn, then inspect memory directly for the hard checks.

    The scenario's namespaces are wiped before the run, so a rerun never sees
    the previous run's facts. They are left in place afterwards for inspection
    and wiped again by the next run.
    """
    session = _Session(scenario, backend, database)
    await session.open()
    try:
        assert session.process is not None
        for user in ("owner", "stranger"):
            if scenario.persona and user == "owner":
                continue
            await session.process.wipe(
                namespace=namespace_for(user_id_for(scenario, user))
            )
        turns = [await session.send(turn) for turn in scenario.turns]
        assert session.process is not None
        owner = namespace_for(user_id_for(scenario, "owner"))
        probe = (
            [
                _rendered(fact)
                for fact in await session.process.memory.search(
                    namespace=owner, query=scenario.probe_query, limit=20
                )
            ]
            if scenario.probe_query
            else []
        )
        entities = await session.process.entity_names(namespace=owner)
        if scenario.persona and any(turn.writes for turn in turns):
            # The persona is shared: the next run must not inherit this write.
            await session.process.wipe(namespace=owner)
    finally:
        await session.close()
    return MemoryRun(
        turns=turns, probe=probe, entities=entities, persona=scenario.persona
    )


def _has(text: str, alternatives: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(fragment.casefold() in folded for fragment in alternatives)


NUMBERED_ITEM = re.compile(r"^\s*\d{1,2}[.)]\s+(\S.*)$", re.MULTILINE)
BULLET_ITEM = re.compile(r"^\s*[-•*–]\s+(\S.*)$", re.MULTILINE)
# «без грибов», «не театр», «кроме Эрарты» name what an item avoids. A
# preposition may stand between the negation and the noun («не в баре»), and the
# noun may carry one of its own («без похода в театр»); both belong to what is
# avoided, so they are stripped with it — otherwise the bare noun looks like a
# suggestion and a correct item is failed.
PREPOSITION = r"(?:в|во|на|с|со|из|к|ко|у|по|за|при|про|об|о)"
WORD = r"[\w«»\"-]+"
NEGATED = re.compile(
    rf"(?<!\w)(?:без|не|нет|кроме|вместо)"
    rf"(?:\s+{PREPOSITION})?\s+{WORD}"
    rf"(?:\s+{PREPOSITION}\s+{WORD})?"
    rf"(?:\s+(?:и|или)\s+{WORD})*",
    re.IGNORECASE,
)
# «1. Рамен, 2. Том ям» written on one line still counts as a list.
INLINE_NUMBER = re.compile(r"(?:^|(?<=[\s;:,.]))(\d{1,2})[.)]\s+")


def list_items(reply: str) -> list[str]:
    """Top-level list items: numbered ones if any, bullets otherwise.

    An explanation nested as bullets under a numbered item stays part of it.
    Numbering run into one line counts only as an unbroken 1, 2, 3… sequence,
    so a stray «2021.» inside a sentence is not an item.
    """
    lines = NUMBERED_ITEM.findall(reply)
    if len(lines) > 1:
        return lines
    starts: list[re.Match[str]] = []
    for match in INLINE_NUMBER.finditer(reply):
        if int(match.group(1)) == len(starts) + 1:
            starts.append(match)
    if len(starts) > 1:
        ends = [match.start() for match in starts[1:]] + [len(reply)]
        return [
            reply[match.end() : end].strip(" ;,\n")
            for match, end in zip(starts, ends, strict=True)
        ]
    return lines or BULLET_ITEM.findall(reply)


def assert_memory_contract(scenario: MemoryScenario, run: MemoryRun) -> None:
    """Invariants a generous judge score must never waive."""
    answer = run.answer
    assert answer.reply.strip(), f"{scenario.id}: empty reply"
    items = list_items(answer.reply)
    if scenario.item_count is not None:
        assert len(items) == scenario.item_count, (
            f"{scenario.id}: {len(items)} list items instead of "
            f"{scenario.item_count}: {answer.reply!r}"
        )
    for item in items:
        affirmed = NEGATED.sub(" ", item)
        for pattern in scenario.items_exclude:
            assert not re.search(rf"(?<!\w){pattern}", affirmed, re.IGNORECASE), (
                f"{scenario.id}: list item matches {pattern!r}: {item!r}"
            )
    folded = answer.reply.casefold()
    positions = []
    for alternatives in scenario.reply_order:
        offsets = [folded.find(fragment.casefold()) for fragment in alternatives]
        assert max(offsets) >= 0, f"{scenario.id}: reply lacks any of {alternatives}"
        positions.append(min(offset for offset in offsets if offset >= 0))
    assert positions == sorted(positions), (
        f"{scenario.id}: {scenario.reply_order} out of order: {answer.reply!r}"
    )
    for fragment, most in scenario.max_mentions:
        count = folded.count(fragment.casefold())
        assert count <= most, (
            f"{scenario.id}: {fragment!r} appears {count} times: {answer.reply!r}"
        )
    for alternatives in scenario.reply_contains:
        assert _has(answer.reply, alternatives), (
            f"{scenario.id}: reply lacks any of {alternatives}: {answer.reply!r}"
        )
    for fragment in scenario.reply_excludes:
        assert not _has(answer.reply, (fragment,)), (
            f"{scenario.id}: forbidden {fragment!r} in reply: {answer.reply!r}"
        )
    for fragment in scenario.history_excludes:
        assert not _has(answer.history, (fragment,)), (
            f"{scenario.id}: {fragment!r} is still in the chat history, so the "
            "answer does not prove long-term memory"
        )

    namespace = namespace_for(answer.user_id)
    for call in answer.lookups:
        # KnowledgeService derives the namespace; a foreign one is a leak.
        assert call.arguments["namespace"] == namespace, (
            f"{scenario.id}: user {answer.user_id} read {call.arguments['namespace']}"
        )
    if scenario.recall_required:
        assert answer.lookups, f"{scenario.id}: the answer never consulted memory"
    found = "\n".join(str(call.output) for call in answer.lookups)
    for fragment in scenario.lookup_excludes:
        assert not _has(found, (fragment,)), (
            f"{scenario.id}: memory lookup returned {fragment!r}: {found!r}"
        )

    written = "\n".join(
        str(call.arguments["content"]) for turn in run.turns for call in turn.writes
    )
    for alternatives in scenario.remembered:
        assert _has(written, alternatives), (
            f"{scenario.id}: nothing written to memory mentions any of "
            f"{alternatives}; writes: {written!r}"
        )
    for fragment in scenario.never_remembered:
        assert not _has(written, (fragment,)), (
            f"{scenario.id}: {fragment!r} was written to memory: {written!r}"
        )

    probe = "\n".join(run.probe)
    for alternatives in scenario.probe_contains:
        assert _has(probe, alternatives), (
            f"{scenario.id}: memory no longer holds any of {alternatives}: {probe!r}"
        )
    for fragment in scenario.probe_excludes:
        assert not _has(probe, (fragment,)), (
            f"{scenario.id}: memory holds {fragment!r}: {probe!r}"
        )
    if scenario.max_entities is not None:
        fragment, limit = scenario.max_entities
        matching = [name for name in run.entities if _has(name, (fragment,))]
        assert len(matching) <= limit, (
            f"{scenario.id}: {len(matching)} entities for one person: {matching}"
        )


def transcript(run: MemoryRun) -> str:
    """The dialog as the judge reads it, with the facts that make it a test."""
    lines: list[str] = []
    if run.persona:
        lines.append("[память до диалога уже содержит stored_profile пользователя]")
    for result in run.turns:
        turn = result.turn
        notes = [f"день +{turn.day:g}", f"чат {turn.chat}"]
        if turn.user == "stranger":
            notes.append("ДРУГОЙ пользователь")
        if turn.restart:
            notes.append("бот перезапущен перед сообщением")
        if turn.clear_history:
            notes.append("история чата очищена")
        if turn.filler:
            notes.append(f"до этого {turn.filler} посторонних обменов репликами")
        lines.append(f"[{', '.join(notes)}] Пользователь: {turn.text}")
        if result is not run.answer:
            lines.append(f"Бот: {result.reply}")
    return "\n".join(lines)
