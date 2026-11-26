"""The interface every game backend implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from napoleon.schema import Action, Snapshot, StepResult


@runtime_checkable
class GameBackend(Protocol):
    """A game the adapter can drive.

    Backends only need to apply actions they list as legal. ``CivEnv`` checks legality
    before calling ``apply``, so a backend may assume it never sees an illegal action.
    """

    def reset(self, scenario: str, seed: int) -> None: ...

    def snapshot(self) -> Snapshot: ...

    def apply(self, action: Action) -> StepResult: ...
