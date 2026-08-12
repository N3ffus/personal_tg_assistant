from enum import StrEnum


class ActionType(StrEnum):
    CHAT = "chat"
    CREATE_TASK = "create_task"
    CREATE_EVENT = "create_event"
    SAVE_NOTE = "save_note"
