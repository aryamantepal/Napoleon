"""The interface every player implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from napoleon.schema import Action, Snapshot
from napoleon.usage import Usage


@dataclass(frozen=True)
class Decision:
    action: Action | dict | str
    reasoning: str | None = None
    usage: Usage | None = None
    events: tuple[dict, ...] = ()  # agent-side happenings worth showing in a replay


@runtime_checkable
class Agent(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, snapshot: Snapshot) -> Decision: ...
