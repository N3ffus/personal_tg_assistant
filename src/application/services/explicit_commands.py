import re

from src.domain.assistant.enums import ActionType
from src.domain.assistant.models import (
    AssistantDecision,
    DeleteAllEventsAction,
    DeleteAllTasksAction,
    ListEventsAction,
    ListTasksAction,
)


def parse_view_request(text: str) -> AssistantDecision | None:
    normalized = " ".join(text.casefold().split()).strip(" .!\n")
    if (
        normalized in {"/tasks", "покажи задачи в linear", "покажи задачи с linear"}
        or re.fullmatch(
            r"(?:покажи|выведи)?\s*(?:список\s+)?(?:всех?\s+)?задач(?:и|ек)?(?:\s+(?:в|с|из)\s+(?:linear|линеар))?",
            normalized,
        )
        or normalized in {"список задач", "задачи", "все задачи"}
    ):
        return AssistantDecision(actions=[ListTasksAction(type=ActionType.LIST_TASKS)])
    if normalized in {"/calendar", "покажи события календаря"} or re.fullmatch(
        r"(?:покажи|выведи)?\s*(?:список\s+)?(?:всех?\s+)?событи[йя](?:\s+(?:в|с|из)\s+(?:календар[яе]|google\s+calendar))?",
        normalized,
    ):
        return AssistantDecision(
            actions=[ListEventsAction(type=ActionType.LIST_EVENTS)]
        )
    if normalized in {
        "/agenda",
        "покажи задачи из linear и события календаря",
        "покажи задачи с linear и с календаря",
    } or re.fullmatch(
        r"(?:покажи|выведи)?\s*(?:список\s+)?задач(?:и|ек)?\s+(?:из|с|в)\s+(?:linear|линеар)\s+и\s+(?:событи[йя]|встреч[и]?)\s+(?:из|с|в\s+)?календар[яе]",
        normalized,
    ):
        return AssistantDecision(
            actions=[
                ListTasksAction(type=ActionType.LIST_TASKS),
                ListEventsAction(type=ActionType.LIST_EVENTS),
            ]
        )
    return None


def parse_bulk_deletion(text: str) -> AssistantDecision | None:
    """Recognize only complete, unqualified bulk commands before LLM routing.

    Negations, questions, exclusions and mixed requests fall through to the LLM.
    Textual confirmation never executes a deletion; it still produces buttons.
    """
    normalized = " ".join(text.casefold().split()).strip(" .!\n")
    prefix = r"(?:пожалуйста,? )?(?:удали|удалить) все "
    suffix = r"(?:[, ]+(?:все|подтверждаю|пожалуйста))*"
    if re.fullmatch(
        prefix + r"задачи (?:из|с|в) (?:linear|линеар)" + suffix, normalized
    ):
        return AssistantDecision(
            actions=[DeleteAllTasksAction(type=ActionType.DELETE_ALL_TASKS)]
        )
    if re.fullmatch(
        prefix + r"события (?:из|с|в) (?:(?:google|гугл) )?календар[яе]" + suffix,
        normalized,
    ):
        return AssistantDecision(
            actions=[DeleteAllEventsAction(type=ActionType.DELETE_ALL_EVENTS)]
        )
    return None
