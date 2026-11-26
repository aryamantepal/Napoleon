"""Porygon-style adapter: ``reset`` / ``observe`` / ``step`` over any ``GameBackend``."""

from __future__ import annotations

from pydantic import ValidationError

from napoleon.backend import GameBackend
from napoleon.schema import Action, EndTurn, Snapshot, StepResult, parse_action


class EnvNotReset(RuntimeError):
    """Raised when ``observe`` or ``step`` is called before ``reset``."""


class CivEnv:
    """Wraps a backend with action parsing, legality checks and turn-blocker handling.

    An illegal or malformed action never raises: ``step`` returns ``ok=False`` with an
    error message and leaves the game unchanged, so a weak agent cannot crash a run.
    """

    def __init__(self, backend: GameBackend) -> None:
        self.backend = backend
        self._ready = False
        self.steps = 0

    def reset(self, scenario: str = "default", seed: int = 0) -> Snapshot:
        self.backend.reset(scenario, seed)
        self._ready = True
        self.steps = 0
        return self.backend.snapshot()

    def observe(self) -> Snapshot:
        self._check_ready()
        return self.backend.snapshot()

    def step(self, action: Action | dict | str) -> tuple[Snapshot, StepResult]:
        self._check_ready()
        self.steps += 1
        snap = self.backend.snapshot()
        if snap.done:
            return snap, StepResult(ok=False, error="game is over", done=True)

        try:
            parsed = action if not isinstance(action, dict | str) else parse_action(action)
        except ValidationError as exc:
            return snap, StepResult(ok=False, error=f"malformed action: {exc.errors()[0]['msg']}")

        blocking = [b for b in snap.blockers if b.kind != "other"]
        if isinstance(parsed, EndTurn) and blocking:
            reasons = "; ".join(b.message or b.kind for b in blocking)
            return snap, StepResult(ok=False, error=f"turn is blocked: {reasons}")

        if parsed not in snap.legal_actions:
            return snap, StepResult(ok=False, error=f"illegal action: {parsed.model_dump_json()}")

        result = self.backend.apply(parsed)
        return self.backend.snapshot(), result

    def _check_ready(self) -> None:
        if not self._ready:
            raise EnvNotReset("call reset() first")
