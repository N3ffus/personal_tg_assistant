from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Confirmation:
    text: str
    operation_id: str


@dataclass(frozen=True, slots=True)
class AssistantReply:
    text: str
    confirmations: tuple[Confirmation, ...]
