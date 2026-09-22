import asyncio
import json
import logging
import re
from datetime import datetime
from time import perf_counter

from src.application.ports.films import FilmDirectory
from src.application.ports.knowledge import KnowledgeMemoryError
from src.application.ports.llm import LLMClient
from src.application.services.action_executor import (
    MEMORY_UNAVAILABLE,
    ActionExecutor,
    format_facts,
)
from src.application.services.context import ContextService
from src.application.services.explicit_commands import (
    parse_bulk_deletion,
    parse_view_request,
)
from src.application.services.film_recommendations import FilmRecommender
from src.application.services.knowledge import KnowledgeService
from src.application.services.turn import Turn
from src.domain.assistant.context import ContextMessage
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    MAX_FORGOTTEN_FACTS,
    AssistantAction,
    AssistantDecision,
    ChatAction,
    ForgetKnowledgeAction,
    RecommendFilmsAction,
    SearchKnowledgeAction,
)
from src.domain.assistant.replies import AssistantReply, ResultPage
from src.domain.knowledge.models import KnowledgeFact

logger = logging.getLogger(__name__)
MAX_PARALLEL_MEMORY_SEARCHES = 4
NOTHING_TO_FORGET = "В памяти этого нет — забывать нечего."


# A long list is where the model breaks its own rules: it writes «Рамен с
# грибами шиитаке (без грибов)» or offers a place the user has already been to
# and excuses it in brackets. Prompt wording did not stop it, so a list this
# long earns one review turn against the recalled facts.
MIN_REVIEWED_ITEMS = 5
LIST_ITEM = re.compile(r"^\s*\d{1,2}[.)]\s+\S", re.MULTILINE)
REVIEW_INSTRUCTION = (
    "(Это твой черновик ответа. Проверь каждый пункт против ограничений запроса и "
    "recalled_knowledge: пункт с нарушением («с грибами», место, где пользователь "
    "уже был, запрещённый тип) замени другим, а не оговаривай в скобках. Число "
    "пунктов сохрани. Верни chat с готовым ответом.)"
)


SEARCH_ALREADY_DONE = (
    "(Память уже прочитана целиком, всё найденное — в recalled_knowledge. "
    "Ответь на сообщение выше действием chat, без search_knowledge.)"
)


def _chat(text: str) -> AssistantDecision:
    return AssistantDecision(actions=[ChatAction(type=ActionType.CHAT, text=text)])


def _without_searches(decision: AssistantDecision) -> list[AssistantAction]:
    # A second search would loop: answer from what memory already returned.
    return [
        action
        for action in decision.actions
        if not isinstance(action, SearchKnowledgeAction)
    ]


# A chat reply confessing it does not know something. The prompt forbids saying
# so about the user without searching, yet the model still answered «Я пока не
# знаю, куда мы ездили на выходных» with the trip stored (eval, 2026-09-14).
ADMITS_IGNORANCE = re.compile(
    r"не\s+(знаю|помню|нашл|говорил|рассказывал|упоминал|сохран)"
    r"|нет\s+(информации|данных|сведений)"
    r"|(расскажи|напомни)(те)?\b",
    re.IGNORECASE,
)


class ProcessMessageUseCase:
    def __init__(
        self,
        *,
        llm: LLMClient,
        action_executor: ActionExecutor,
        contexts: ContextService | None = None,
        knowledge: KnowledgeService | None = None,
        films: FilmDirectory | None = None,
    ) -> None:
        self._llm = llm
        self._action_executor = action_executor
        self._contexts = contexts
        self._knowledge = knowledge
        self._films = (
            FilmRecommender(llm=llm, knowledge=knowledge, films=films)
            if knowledge is not None
            else None
        )

    async def execute(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
        user_id: int,
        chat_id: int | None = None,
        message_id: int | None = None,
    ) -> str | AssistantReply:
        if self._contexts is None:
            return await self._respond(
                text=text,
                now=now,
                timezone=timezone,
                user_id=user_id,
                message_id=message_id,
            )
        chat = await self._contexts.ensure_chat(
            owner_id=user_id,
            kind="bot",
            chat_id=chat_id if chat_id is not None else user_id,
            title="Чат с ботом",
        )
        async with self._contexts.session(
            owner_id=user_id, context_id=chat.id
        ) as context:
            response = await self._respond(
                text=text,
                now=now,
                timezone=timezone,
                user_id=user_id,
                message_id=message_id,
                context=context.render(),
            )
            context.append(
                ContextMessage(
                    role="user",
                    sender="Вы",
                    text=text,
                    sent_at=now,
                    message_id=message_id,
                )
            )
            reply_text = (
                response
                if isinstance(response, str)
                else "\n\n".join(
                    [
                        response.text,
                        *(confirmation.text for confirmation in response.confirmations),
                        *(page.text for page in response.pages),
                    ]
                ).strip()
            )
            context.append(
                ContextMessage(
                    role="assistant", sender="Бот", text=reply_text, sent_at=now
                )
            )
            return response

    async def _respond(
        self,
        *,
        text: str,
        now: datetime,
        timezone: str,
        user_id: int,
        message_id: int | None = None,
        context: str = "",
    ) -> str | AssistantReply:
        turn = Turn(
            text=text, now=now, timezone=timezone, user_id=user_id, context=context
        )
        started = perf_counter()
        outcome = "failed"
        try:
            decision = (
                parse_view_request(text)
                or parse_bulk_deletion(text)
                or await turn.ask(self._llm)
            )
            decision = self._searched_before_admitting_ignorance(decision, text=text)
            decision = await self._with_recalled_knowledge(decision, turn=turn)
            reply = await self._action_executor.execute_many(
                decision.actions, user_id=user_id, now=now, message_id=message_id
            )
            outcome = "ok"
            return reply
        finally:
            logger.info(
                "turn.completed outcome=%s latency_ms=%.0f llm_calls=%d llm_ms=%.0f",
                outcome,
                (perf_counter() - started) * 1000,
                len(turn.llm_seconds),
                sum(turn.llm_seconds) * 1000,
            )

    def _searched_before_admitting_ignorance(
        self, decision: AssistantDecision, *, text: str
    ) -> AssistantDecision:
        """Never let «не знаю» stand when memory was not even consulted.

        Only a reply made of chat alone qualifies: any other action means the
        model did act on the message. The search uses the user's own words,
        and the answer turn that follows may still say the fact is unknown.
        """
        if self._knowledge is None or not all(
            isinstance(action, ChatAction) for action in decision.actions
        ):
            return decision
        if not any(
            ADMITS_IGNORANCE.search(action.text)
            for action in decision.actions
            if isinstance(action, ChatAction)
        ):
            return decision
        logger.info("knowledge.recall.forced reason=chat_admitted_ignorance")
        return AssistantDecision(
            actions=[
                SearchKnowledgeAction(
                    type=ActionType.SEARCH_KNOWLEDGE, query=text.strip()[:400] or "?"
                )
            ]
        )

    async def _with_recalled_knowledge(
        self, decision: AssistantDecision, *, turn: Turn
    ) -> AssistantDecision:
        """Answer memory questions with the facts the graph actually holds.

        Retrieval is tool-based; recommendations also fetch the full exclusion
        catalogue before the final candidate generation.
        """
        if self._knowledge is None:
            return decision
        forgets = [
            action
            for action in decision.actions
            if isinstance(action, ForgetKnowledgeAction)
        ]
        if forgets:
            resolved = await self._forgetting(forgets, turn=turn)
            # «Уволился из Яндекса, теперь в Ozon» arrived as forget + remember;
            # resolving the forget must never cost the user the new fact.
            others: list[AssistantAction] = [
                action
                for action in decision.actions
                if not isinstance(
                    action, (ForgetKnowledgeAction, SearchKnowledgeAction)
                )
            ]
            if isinstance(resolved, ChatAction) and others:
                return AssistantDecision(actions=others)
            return AssistantDecision(actions=[resolved, *others])
        searches = [
            action
            for action in decision.actions
            if isinstance(action, SearchKnowledgeAction)
        ]
        recommends = any(
            isinstance(action, RecommendFilmsAction) for action in decision.actions
        )
        if recommends and not any(
            search.mode == "watched_film_catalogue" for search in searches
        ):
            searches.append(
                SearchKnowledgeAction(
                    type=ActionType.SEARCH_KNOWLEDGE,
                    query="Просмотренные фильмы для исключения",
                    mode="watched_film_catalogue",
                )
            )
        if not searches:
            return decision
        recalled, facts = await self._recall(searches, user_id=turn.user_id)
        informed = await turn.ask(self._llm, knowledge=recalled)
        if any(search.mode == "watched_film_catalogue" for search in searches):
            return await self._recommendation(informed, turn=turn, facts=facts)
        actions = _without_searches(informed)
        if not actions:
            # The answer turn searched again instead of answering: «Посоветуй
            # пиццу» got the raw fact list below on production (2026-09-14),
            # one answer turn in five. A second ask almost always answers.
            # Asking identically was not enough once the whole profile came
            # back: «15 идей» searched again twice and dumped 50 raw facts
            # (memory eval, 2026-09-16). The retry says the search is done.
            logger.warning("llm.answer.repeated_search retry=1")
            actions = _without_searches(
                await turn.ask(
                    self._llm,
                    knowledge=recalled,
                    text=f"{turn.text}\n\n{SEARCH_ALREADY_DONE}",
                )
            )
        actions = await self._reviewed(actions, turn=turn, recalled=recalled)
        if not actions:
            return _chat(format_facts(facts))
        return AssistantDecision(actions=actions)

    async def _recommendation(
        self, informed: AssistantDecision, *, turn: Turn, facts: list[KnowledgeFact]
    ) -> AssistantDecision:
        recommendation = next(
            (
                action
                for action in informed.actions
                if isinstance(action, RecommendFilmsAction)
            ),
            None,
        )
        if recommendation is None or self._films is None:
            return _chat(
                "Не удалось подобрать и проверить новые фильмы; попробуй уточнить жанр."
            )
        return AssistantDecision(
            actions=[
                await self._films.recommend(recommendation, turn=turn, facts=facts)
            ]
        )

    async def _reviewed(
        self, actions: list[AssistantAction], *, turn: Turn, recalled: str
    ) -> list[AssistantAction]:
        """Give a long personalised list one pass against the recalled facts."""
        if len(actions) != 1 or not isinstance(actions[0], ChatAction):
            return actions
        draft = actions[0].text
        items = len(LIST_ITEM.findall(draft))
        if items < MIN_REVIEWED_ITEMS:
            return actions
        logger.info("llm.answer.reviewed items=%s", items)
        reviewed = _without_searches(
            await turn.ask(
                self._llm,
                knowledge=recalled,
                text=f"{turn.text}\n\n{draft}\n\n{REVIEW_INSTRUCTION}",
                with_context=False,
            )
        )
        checked = [action for action in reviewed if isinstance(action, ChatAction)]
        # A review that answers with anything but chat, or drops the list, is
        # no improvement: the draft already answered the question.
        if len(checked) != 1 or len(LIST_ITEM.findall(checked[0].text)) != items:
            logger.warning("llm.answer.review_discarded")
            return actions
        return list(checked)

    async def _forgetting(
        self, forgets: list[ForgetKnowledgeAction], *, turn: Turn
    ) -> ForgetKnowledgeAction | ChatAction:
        """Let the model pick, by ref, exactly the facts the request covers.

        Erasing by resemblance alone would take the previous employer along
        with the one the user asked to forget: the two sentences differ in a
        single word. Only refs this recall produced may reach the executor.
        """
        assert self._knowledge is not None
        try:
            found = [
                fact
                for forget in forgets
                for fact in await self._knowledge.forget_candidates(
                    user_id=turn.user_id, query=forget.query
                )
            ]
        except KnowledgeMemoryError as error:
            logger.warning("knowledge.forget.degraded reason=%s", error)
            return ChatAction(type=ActionType.CHAT, text=MEMORY_UNAVAILABLE)
        candidates = list({fact.ref: fact for fact in found if fact.ref}.values())
        if not candidates:
            return ChatAction(type=ActionType.CHAT, text=NOTHING_TO_FORGET)
        informed = await turn.ask(
            self._llm,
            knowledge=json.dumps(
                [fact.as_payload() for fact in candidates], ensure_ascii=False
            ),
        )
        allowed = {fact.ref for fact in candidates}
        refs = list(
            dict.fromkeys(
                ref
                for action in informed.actions
                if isinstance(action, ForgetKnowledgeAction)
                for ref in action.refs
                if ref in allowed
            )
        )
        if not refs:
            return ChatAction(type=ActionType.CHAT, text=NOTHING_TO_FORGET)
        return ForgetKnowledgeAction(
            type=ActionType.FORGET_KNOWLEDGE,
            query=forgets[0].query,
            refs=refs[:MAX_FORGOTTEN_FACTS],
        )

    async def _recall(
        self, searches: list[SearchKnowledgeAction], *, user_id: int
    ) -> tuple[str, list[KnowledgeFact]]:
        knowledge = self._knowledge
        if knowledge is None:  # pragma: no cover - guarded by the caller
            return "[]", []
        unique: dict[tuple[str, str, int], SearchKnowledgeAction] = {}
        for search in searches:
            key = (
                search.mode,
                search.query if search.mode == "semantic" else "",
                search.limit
                if search.mode in {"semantic", "recent_watched_films"}
                else 0,
            )
            unique.setdefault(key, search)
        semaphore = asyncio.Semaphore(MAX_PARALLEL_MEMORY_SEARCHES)

        async def lookup(search: SearchKnowledgeAction) -> list[KnowledgeFact]:
            async with semaphore:
                return await knowledge.lookup(search, user_id=user_id)

        results = await asyncio.gather(
            *(lookup(search) for search in unique.values()), return_exceptions=True
        )
        facts: list[KnowledgeFact] = []
        for found in results:
            if isinstance(found, KnowledgeMemoryError):
                logger.warning("knowledge.search.degraded reason=%s", found)
                return (
                    json.dumps(
                        {"error": "knowledge memory is temporarily unavailable"},
                        ensure_ascii=False,
                    ),
                    [],
                )
            if isinstance(found, BaseException):
                raise found
            facts.extend(fact for fact in found if fact not in facts)
        payload = [fact.as_payload() for fact in facts]
        return json.dumps(payload, ensure_ascii=False), facts

    async def resolve_deletion(
        self, *, user_id: int, operation_id: str, confirm: bool
    ) -> str:
        return await self._action_executor.resolve_deletion(
            user_id=user_id, operation_id=operation_id, confirm=confirm
        )

    async def browse(self, *, user_id: int, data: str) -> ResultPage | str:
        return await self._action_executor.browse(user_id=user_id, data=data)
