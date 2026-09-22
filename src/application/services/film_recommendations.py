import asyncio
import json
import logging

from src.application.ports.films import FilmDirectory
from src.application.ports.knowledge import KnowledgeMemoryError
from src.application.ports.llm import LLMClient
from src.application.services.knowledge import KnowledgeService
from src.application.services.turn import Turn
from src.domain.assistant.models import (
    MAX_FILM_ALIASES,
    MAX_FILM_CANDIDATES,
    FilmCandidate,
    RecommendFilmsAction,
)
from src.domain.knowledge.films import FilmRecord, requested_count, select_unwatched
from src.domain.knowledge.models import KnowledgeFact

logger = logging.getLogger(__name__)

# Each round is a full LLM call (~30 s on production); two bound the wait.
MAX_TOPUP_ROUNDS = 2


class FilmRecommender:
    """Turn the model's candidates into films that exist and were not watched."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        knowledge: KnowledgeService,
        films: FilmDirectory | None = None,
    ) -> None:
        self._llm = llm
        self._knowledge = knowledge
        self._films = films

    async def recommend(
        self,
        action: RecommendFilmsAction,
        *,
        turn: Turn,
        facts: list[KnowledgeFact],
    ) -> RecommendFilmsAction:
        count = requested_count(turn.text)
        if count is not None and count != action.limit:
            # «топ 10» once came back as limit=3 (production, 2026-09-15).
            logger.info(
                "films.recommend.limit_override model=%s requested=%s",
                action.limit,
                count,
            )
            action = action.model_copy(update={"limit": count})
        return await self._topped_up(action, turn=turn, facts=facts)

    async def _topped_up(
        self,
        action: RecommendFilmsAction,
        *,
        turn: Turn,
        facts: list[KnowledgeFact],
    ) -> RecommendFilmsAction:
        """Ask again while the catalogue swallows most of each batch.

        The model tends to propose exactly as many candidates as the user asked
        for, and against a catalogue of hundreds of films most of them are
        already watched — leaving the user with a dead end. Every title rejected
        so far goes back with the request so the next batch is genuinely
        different. One retry was not enough: «топ 10» ended with 11 of 12 and
        then 9 of 10 watched, two films shown (production, 2026-09-15).
        """
        try:
            watched = await self._knowledge.watched_title_keys(user_id=turn.user_id)
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
        batch, rejected = action.candidates, list[str]()
        for round_number in range(1, MAX_TOPUP_ROUNDS + 1):
            kept = {id(candidate) for candidate in survivors}
            rejected.extend(
                [candidate.title for candidate in real if id(candidate) not in kept]
                or [candidate.title for candidate in batch]
            )
            retry = _retry_fact(
                rejected=rejected,
                chosen=[candidate.title for candidate in survivors],
                missing=action.limit - len(survivors),
            )
            answer = await turn.ask(
                self._llm,
                knowledge=json.dumps(
                    [fact.as_payload() for fact in [*facts, retry]], ensure_ascii=False
                ),
            )
            batch = [
                candidate
                for other in answer.actions
                if isinstance(other, RecommendFilmsAction)
                for candidate in other.candidates
            ]
            real = await self._verified(batch)
            survivors = select_unwatched(
                [*survivors, *real], watched=watched, limit=action.limit
            )
            logger.info(
                "films.recommend.topup round=%s candidates=%s real=%s shown=%s limit=%s",
                round_number,
                len(batch),
                len(real),
                len(survivors),
                action.limit,
            )
            if len(survivors) >= action.limit:
                break
        # An action must always carry a candidate; the executor reports an empty
        # result in the one voice the user sees for it.
        return action.model_copy(update={"candidates": survivors or action.candidates})

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


def _retry_fact(
    *, rejected: list[str], chosen: list[str], missing: int
) -> KnowledgeFact:
    already = f"Уже выбраны, не повторяй: {', '.join(chosen)}. " if chosen else ""
    return KnowledgeFact(
        fact="Эти кандидаты уже просмотрены или отклонены, не предлагай их "
        f"снова: {', '.join(dict.fromkeys(rejected))}. "
        f"{already}"
        f"Нужно ещё {missing}. Популярное почти всё просмотрено: предложи "
        f"{MAX_FILM_CANDIDATES} других, менее очевидных фильмов."
    )
