"""Step-by-step run recording as JSONL, and a reader for replay and metrics.

A run file holds one ``header`` line, then one ``step`` line per ``CivEnv.step`` call,
then an optional ``footer``. Each step stores the action, the agent's reasoning, the
result and timing. A full snapshot is stored on the first step of every turn (and on the
first step of the run); other steps carry only a hash of the snapshot they saw, which
keeps files small while still letting a viewer rebuild every turn.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from napoleon import __version__
from napoleon.schema import SCHEMA_VERSION, Action, Snapshot, StepResult
from napoleon.usage import Usage


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunHeader(_Model):
    kind: Literal["header"] = "header"
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    backend: str
    scenario: str
    seed: int
    agent: str
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: str = SCHEMA_VERSION
    napoleon_version: str = __version__
    config: dict[str, Any] = Field(default_factory=dict)


class StepRecord(_Model):
    kind: Literal["step"] = "step"
    step: int
    turn: int
    snapshot_hash: str
    snapshot: Snapshot | None = None
    action: Action | None = None
    raw_action: Any = None
    reasoning: str | None = None
    result: StepResult
    elapsed_ms: float = 0.0
    usage: Usage | None = None


class RunFooter(_Model):
    kind: Literal["footer"] = "footer"
    ended_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    final_snapshot: Snapshot
    steps: int
    reason: str = "done"


def snapshot_hash(snap: Snapshot) -> str:
    return hashlib.sha256(snap.model_dump_json().encode()).hexdigest()[:16]


class Recorder:
    """Appends a run to a JSONL file. Use as a context manager or call ``close``."""

    def __init__(self, path: str | Path, header: RunHeader) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = header
        self._fh = self.path.open("w", encoding="utf-8")
        self._write(header)
        self._step = 0
        self._last_turn: int | None = None

    def record(
        self,
        seen: Snapshot,
        action: Action | dict | str | None,
        result: StepResult,
        reasoning: str | None = None,
        elapsed_ms: float = 0.0,
        usage: Usage | None = None,
    ) -> StepRecord:
        """Record one step. ``seen`` is the snapshot the agent acted on."""
        full = seen if seen.turn != self._last_turn else None
        self._last_turn = seen.turn
        typed = action if isinstance(action, BaseModel) else None
        rec = StepRecord(
            step=self._step,
            turn=seen.turn,
            snapshot_hash=snapshot_hash(seen),
            snapshot=full,
            action=typed,
            raw_action=None if typed is not None else action,
            reasoning=reasoning,
            result=result,
            elapsed_ms=elapsed_ms,
            usage=usage,
        )
        self._write(rec)
        self._step += 1
        return rec

    def close(self, final: Snapshot, reason: str = "done") -> None:
        if self._fh.closed:
            return
        self._write(RunFooter(final_snapshot=final, steps=self._step, reason=reason))
        self._fh.close()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        if not self._fh.closed:
            self._fh.close()

    def _write(self, model: BaseModel) -> None:
        self._fh.write(model.model_dump_json() + "\n")
        self._fh.flush()


class Run(_Model):
    header: RunHeader
    steps: list[StepRecord]
    footer: RunFooter | None = None

    def turn_snapshots(self) -> list[Snapshot]:
        """The snapshot at the start of each turn, in order, plus the final one."""
        snaps = [s.snapshot for s in self.steps if s.snapshot is not None]
        if self.footer is not None:
            snaps.append(self.footer.final_snapshot)
        return snaps


def iter_records(path: str | Path) -> Iterator[RunHeader | StepRecord | RunFooter]:
    kinds = {"header": RunHeader, "step": StepRecord, "footer": RunFooter}
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                yield kinds[data["kind"]].model_validate(data)
            except KeyError as exc:
                raise ValueError(f"{path}:{n}: unknown record kind") from exc


def load_run(path: str | Path) -> Run:
    """Read a run file. A run cut off before its footer still loads, with ``footer=None``."""
    header: RunHeader | None = None
    steps: list[StepRecord] = []
    footer: RunFooter | None = None
    for rec in iter_records(path):
        if isinstance(rec, RunHeader):
            header = rec
        elif isinstance(rec, StepRecord):
            steps.append(rec)
        else:
            footer = rec
    if header is None:
        raise ValueError(f"{path}: no header line")
    return Run(header=header, steps=steps, footer=footer)
