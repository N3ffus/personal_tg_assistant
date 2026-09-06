from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DeletionResource = Literal["linear", "calendar"]


class DeletionTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str
    label: str = Field(min_length=1)


class DeletionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource: DeletionResource
    targets: list[DeletionTarget] = Field(min_length=1)
