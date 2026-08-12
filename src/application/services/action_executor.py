from src.domain.assistant.models import (
    AssistantAction,
    ChatAction,
    CreateEventAction,
    CreateTaskAction,
    SaveNoteAction,
)


class ActionExecutor:
    async def execute(
        self,
        action: AssistantAction,
    ) -> str:
        if isinstance(action, ChatAction):
            return action.text

        if isinstance(action, CreateTaskAction):
            return "✅ Понял, нужно создать задачу:\n" f"{action.title}"

        if isinstance(action, CreateEventAction):
            starts_at = action.starts_at.strftime("%d.%m.%Y %H:%M")

            return (
                "📅 Понял, нужно создать событие:\n"
                f"{action.title}\n"
                f"Время: {starts_at}"
            )

        if isinstance(action, SaveNoteAction):
            return "📝 Понял, нужно сохранить заметку:\n" f"{action.text}"

        raise ValueError(f"Unsupported action: {type(action)!r}")
