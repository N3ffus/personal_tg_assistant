from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

MAX_MESSAGES = 60
MAX_CONTEXT_CHARS = 24_000
MAX_MESSAGE_CHARS = 6_000
MAX_SUMMARY_CHARS = 4_000


def clipped(text: str, limit: int) -> str:
    suffix = "\n[Текст сокращён]"
    return text if len(text) <= limit else text[: limit - len(suffix)] + suffix


class ContextMessage(BaseModel):
    role: Literal["user", "assistant"]
    sender: str
    text: str
    sent_at: datetime
    message_id: int | None = None
    reply_to_message_id: int | None = None
    sender_id: int | None = None
    is_business_bot: bool = False
    is_forwarded: bool = False

    def render(self) -> str:
        return f"[{self.sent_at.isoformat()}] {self.sender}: {self.text}"


class ChatContext(BaseModel):
    summary: str = ""
    messages: list[ContextMessage] = Field(default_factory=list)
    # Retain a watermark after clear/compact so old edits cannot restore history.
    last_message_id: int = 0
    discarded_message_id: int = 0

    def render(self) -> str:
        parts = (
            [f"Резюме предыдущей переписки:\n{self.summary}"] if self.summary else []
        )
        parts.extend(message.render() for message in self.messages)
        return "\n\n".join(parts)

    def append(self, message: ContextMessage) -> None:
        message = message.model_copy(
            update={
                "text": clipped(message.text, MAX_MESSAGE_CHARS),
                "sender": message.sender[:100],
            }
        )
        if message.message_id is not None:
            for index, existing in enumerate(self.messages):
                if existing.message_id == message.message_id:
                    self.messages[index] = message
                    break
            else:
                if message.message_id <= self.discarded_message_id:
                    return
                self.messages.append(message)
            self.last_message_id = max(self.last_message_id, message.message_id)
            if all(item.message_id is not None for item in self.messages):
                self.messages.sort(key=lambda item: item.message_id or 0)
        else:
            self.messages.append(message)
        while (
            len(self.messages) > MAX_MESSAGES or len(self.render()) > MAX_CONTEXT_CHARS
        ):
            removed = self.messages.pop(0)
            self.discarded_message_id = max(
                self.discarded_message_id, removed.message_id or 0
            )


class ContextChat(BaseModel):
    id: int
    owner_id: int
    kind: Literal["bot", "business"]
    chat_id: int
    title: str
