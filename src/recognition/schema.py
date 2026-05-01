from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, Field


class Predicate(BaseModel):
    field: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in"]
    value: Any


class Rule(BaseModel):
    id: str
    activity: str
    weight: float = 1.0
    when_all: List[Predicate] = Field(default_factory=list)
    when_any: List[Predicate] = Field(default_factory=list)
    evidence_template: str = ""


class HardConstraint(BaseModel):
    type: Literal["hard_room_type"]
    activity: str
    allowed_room_types: List[str]
    penalty: Literal["-inf"] = "-inf"


class RuleBook(BaseModel):
    version: str = "1.0"
    activities: List[str]
    constraints: List[HardConstraint] = Field(default_factory=list)
    rules: List[Rule] = Field(default_factory=list)
