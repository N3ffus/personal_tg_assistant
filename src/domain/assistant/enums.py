from enum import StrEnum


class ActionType(StrEnum):
    CHAT = "chat"
    CREATE_TASK = "create_task"
    LIST_TASKS = "list_tasks"
    DELETE_TASK = "delete_task"
    DELETE_ALL_TASKS = "delete_all_tasks"
    CREATE_EVENT = "create_event"
    LIST_EVENTS = "list_events"
    UPDATE_EVENT = "update_event"
    DELETE_EVENT = "delete_event"
    DELETE_ALL_EVENTS = "delete_all_events"
    SAVE_NOTE = "save_note"
    REMEMBER_KNOWLEDGE = "remember_knowledge"
    SEARCH_KNOWLEDGE = "search_knowledge"
