from enum import StrEnum


class ActionType(StrEnum):
    CHAT = "chat"
    CREATE_TASK = "create_task"
    CREATE_EVENT = "create_event"
    LIST_EVENTS = "list_events"
    UPDATE_EVENT = "update_event"
    DELETE_EVENT = "delete_event"
    SAVE_NOTE = "save_note"
