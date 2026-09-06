from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Confirmation:
    text: str
    operation_id: str


@dataclass(frozen=True, slots=True)
class ReplyButton:
    text: str
    callback_data: str


@dataclass(frozen=True, slots=True)
class ResultPage:
    text: str
    buttons: tuple[tuple[ReplyButton, ...], ...]


@dataclass(frozen=True, slots=True)
class AssistantReply:
    text: str
    confirmations: tuple[Confirmation, ...]
    pages: tuple[ResultPage, ...] = ()
