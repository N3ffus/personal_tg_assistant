import asyncio
import json
import logging
from datetime import datetime

from src.application.ports.films import FilmDirectory
from src.application.ports.knowledge import KnowledgeMemoryError
from src.application.ports.llm import LLMClient
from src.application.services.action_executor import (
    ActionExecutor,
)
from src.application.services.context import ContextService
from src.application.services.explicit_commands import (
    parse_bulk_deletion,
    parse_view_request,
)
from src.application.services.knowledge import KnowledgeService
from src.domain.assistant.context import ContextMessage
from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    MAX_FILM_ALIASES,
    MAX_FILM_CANDIDATES,
    AssistantAction,
    AssistantDecision,
    ChatAction,
    FilmCandidate,
    RecommendFilmsAction,
    SearchKnowledgeAction,
)
from src.domain.assistant.replies import AssistantReply, ResultPage
from src.domain.knowledge.films import FilmRecord, select_unwatched, title_keys
from src.domain.knowledge.models import KnowledgeFact

logger = logging.getLogger(__name__)


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
        self._films = films

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
        decision = (
            parse_view_request(text)
            or parse_bulk_deletion(text)
            or await self._llm.parse_message(
                text=text,
                now=now,
                timezone=timezone,
                **({"context": context} if context else {}),
            )
        )
        decision = await self._with_recalled_knowledge(
            decision,
            text=text,
            now=now,
            timezone=timezone,
            user_id=user_id,
            context=context,
        )

        return await self._action_executor.execute_many(
            decision.actions,
            user_id=user_id,
            now=now,
            message_id=message_id,
        )

    async def _with_recalled_knowledge(
        self,
        decision: AssistantDecision,
        *,
        text: str,
        now: datetime,
        timezone: str,
        user_id: int,
        context: str,
    ) -> AssistantDecision:
        """Answer memory questions with the facts the graph actually holds.

        Retrieval is tool-based; recommendations also fetch the full exclusion
        catalogue before the final candidate generation.
        """
        searches = [
            action
            for action in decision.actions
            if isinstance(action, SearchKnowledgeAction)
        ]
        if any(
            isinstance(action, RecommendFilmsAction) for action in decision.actions
        ) and not any(search.mode == "watched_film_catalogue" for search in searches):
            searches.append(
                SearchKnowledgeAction(
                    type=ActionType.SEARCH_KNOWLEDGE,
                    query="Просмотренные фильмы для исключения",
                    mode="watched_film_catalogue",
                )
            )
        if not searches or self._knowledge is None:
            return decision
        recalled, facts = await self._recall(searches, user_id=user_id)
        informed = await self._llm.parse_message(
            text=text,
            now=now,
            timezone=timezone,
            knowledge=recalled,
            **({"context": context} if context else {}),
        )
        if any(search.mode == "watched_film_catalogue" for search in searches):
            recommendations = [
                action
                for action in informed.actions
                if isinstance(action, RecommendFilmsAction)
            ]
            if not recommendations:
                return AssistantDecision(
                    actions=[
                        ChatAction(
                            type=ActionType.CHAT,
                            text="Не удалось подобрать и проверить новые фильмы; попробуй уточнить жанр.",
                        )
                    ]
                )
            return AssistantDecision(
                actions=[
                    await self._topped_up(
                        recommendations[0],
                        text=text,
                        now=now,
                        timezone=timezone,
                        user_id=user_id,
                        context=context,
                        facts=facts,
                    )
                ]
            )
        actions: list[AssistantAction] = [
            action
            for action in informed.actions
            # A second search would loop: answer from what memory already returned.
            if not isinstance(action, SearchKnowledgeAction)
        ]
        if not actions:
            return AssistantDecision(
                actions=[
                    ChatAction(
                        type=ActionType.CHAT,
                        text=ActionExecutor.format_facts(facts),
                    )
                ]
            )
        return AssistantDecision(actions=actions)

    async def _topped_up(
        self,
        action: RecommendFilmsAction,
        *,
        text: str,
        now: datetime,
        timezone: str,
        user_id: int,
        context: str,
        facts: list[KnowledgeFact],
    ) -> RecommendFilmsAction:
        """Ask once more when the catalogue swallowed most of the batch.

        The model tends to propose exactly as many candidates as the user asked
        for, and against a catalogue of hundreds of films most of them are
        already watched — leaving the user with a dead end. The rejected titles
        go back with the request so the second batch is genuinely different.
        """
        if self._knowledge is None:  # pragma: no cover - guarded by the caller
            return action
        try:
            watched = {
                key
                for title in await self._knowledge.watched_film_titles(user_id=user_id)
                for key in title_keys(title)
            }
        except KnowledgeMemoryError as error:
            logger.warning("films.topup.skipped reason=%s", error)
            return action
        # Verification comes first: it also rewrites a loose title to the one the
        # directory knows, which is what the watched catalogue is spelled in.
        real = await self._verified(action.candidates)
        survivors = select_unwatched(real, watched=watched, limit=action.limit)
        logger.info(
            "films.recommend.proposed candidates=%s real=%s unwatched=%s limit=%s",
            len(action.candidates),
            len(real),
            len(survivors),
            action.limit,
        )
        if not watched or len(survivors) >= action.limit:
            return action.model_copy(update={"candidates": survivors or real})
        kept = {id(candidate) for candidate in survivors}
        rejected = [
            candidate.title for candidate in real if id(candidate) not in kept
        ] or [candidate.title for candidate in action.candidates]
        retry = KnowledgeFact(
            fact="Эти кандидаты уже просмотрены и отклонены, предложи другие фильмы: "
            + ", ".join(rejected)
        )
        second = await self._llm.parse_message(
            text=text,
            now=now,
            timezone=timezone,
            knowledge=json.dumps(
                [fact.as_payload() for fact in [*facts, retry]], ensure_ascii=False
            ),
            **({"context": context} if context else {}),
        )
        extra = await self._verified(
            [
                candidate
                for other in second.actions
                if isinstance(other, RecommendFilmsAction)
                for candidate in other.candidates
            ]
        )
        topped_up = [*survivors, *extra][:MAX_FILM_CANDIDATES]
        # An action must always carry a candidate; the executor reports an empty
        # result in the one voice the user sees for it.
        return action.model_copy(update={"candidates": topped_up or action.candidates})

    async def _verified(self, candidates: list[FilmCandidate]) -> list[FilmCandidate]:
        """Drop candidates no film directory knows, and canonicalise the rest.

        The model renames a watched film to slip it past the catalogue and
        invents titles outright; nothing inside the application can tell those
        from real films. A directory failure degrades to the unverified batch:
        an unchecked recommendation beats no answer at all.
        """
        if self._films is None or not candidates:
            return candidates
        found = await asyncio.gather(
            *(
                self._films.find(title=candidate.title, year=candidate.year)
                for candidate in candidates
            ),
            return_exceptions=True,
        )
        for record in found:
            if isinstance(record, BaseException):
                # A partly checked batch would drop real films on a network blip.
                logger.warning("films.directory.degraded reason=%s", record)
                return candidates
        verified: list[FilmCandidate] = []
        for candidate, record in zip(candidates, found, strict=True):
            if not isinstance(record, FilmRecord):
                logger.info("films.candidate.unknown title=%r", candidate.title)
                continue
            verified.append(
                candidate.model_copy(
                    update={
                        "title": record.title,
                        "year": record.year or candidate.year,
                        "aliases": [*candidate.aliases, record.original_title][
                            :MAX_FILM_ALIASES
                        ],
                    }
                )
            )
        return verified

    async def _recall(
        self, searches: list[SearchKnowledgeAction], *, user_id: int
    ) -> tuple[str, list[KnowledgeFact]]:
        if self._knowledge is None:  # pragma: no cover - guarded by the caller
            return "[]", []
        facts: list[KnowledgeFact] = []
        for search in searches:
            try:
                if search.mode == "watched_film_catalogue":
                    found = await self._knowledge.watched_film_catalogue(
                        user_id=user_id
                    )
                elif search.mode == "recent_watched_films":
                    found = await self._knowledge.recent_watched_films(
                        user_id=user_id, limit=search.limit
                    )
                else:
                    found = await self._knowledge.search(
                        user_id=user_id, query=search.query, limit=search.limit
                    )
            except KnowledgeMemoryError as error:
                logger.warning("knowledge.search.degraded reason=%s", error)
                return (
                    json.dumps(
                        {"error": "knowledge memory is temporarily unavailable"},
                        ensure_ascii=False,
                    ),
                    [],
                )
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
