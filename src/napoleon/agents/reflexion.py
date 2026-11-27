"""Reflexion: play a game, write a verbal reflection on it, and carry that into later games.

The inner agent (direct or ReAct) plays each game unchanged except that its system prompt
carries the most recent reflections. After each game the model reads the run's metrics and
a digest of its trajectory and writes a short reflection, which is appended to a JSONL
memory file. The file is what persists, so a later process picks up where this one stopped.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from napoleon.agents.base import Agent, Decision
from napoleon.agents.direct import DirectAgent
from napoleon.agents.prompting import action_json
from napoleon.agents.react import ReActAgent
from napoleon.llm import LLMClient, clean_reply, make_client
from napoleon.metrics import compute_metrics
from napoleon.recording import Run
from napoleon.schema import Snapshot
from napoleon.usage import Usage

BASES = {"direct": DirectAgent, "react": ReActAgent}
DEFAULT_MEMORY = Path("runs/memory/reflexion.jsonl")

COACH_SYSTEM = (
    "You coach an agent that plays a turn-based strategy game. After each game you write a "
    "short reflection that the agent reads before its next game. Be specific and practical."
)


class Reflection(BaseModel):
    episode: int
    run_id: str
    scenario: str
    seed: int
    final_score: int
    text: str
    usage: Usage = Usage()
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ReflectionMemory:
    """An append-only JSONL file of reflections."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[Reflection]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [Reflection.model_validate_json(ln) for ln in lines if ln.strip()]

    def append(self, reflection: Reflection) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(reflection.model_dump_json() + "\n")

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def lessons_text(reflections: list[Reflection]) -> str:
    if not reflections:
        return ""
    lines = ["", "", "Lessons from your previous games, oldest first. Apply them:"]
    for r in reflections:
        lines.append(
            f"- Game {r.episode} ({r.scenario}, seed {r.seed}, final score {r.final_score}):"
        )
        lines.append("  " + r.text.replace("\n", "\n  "))
    return "\n".join(lines)


def digest_run(run: Run, max_rows: int = 12, max_failures: int = 8, max_samples: int = 8) -> str:
    """Metrics and a trimmed trajectory, small enough to fit in one reflection prompt."""
    m = compute_metrics(run)
    h = run.header
    ended = run.footer.reason if run.footer else "cut off"
    lines = [
        f"Game: scenario {h.scenario}, seed {h.seed}, agent {h.agent}, "
        f"{m.turns_played} turns played, ended: {ended}.",
        f"Result: final score {m.final_score}, cities {int(m.cities_by_turn[-1][1])}, "
        f"population {int(m.population_by_turn[-1][1])}, techs {int(m.techs_by_turn[-1][1])}, "
        f"victory progress {m.final_victory_progress or 'none'}.",
        f"Steps: {m.steps}, rejected by the game: {m.failed_steps} "
        f"({m.illegal_action_rate:.0%} illegal-action rate).",
    ]
    turns = [t for t, _ in m.score_by_turn]
    stride = max(1, len(turns) // max_rows)
    lines.append("Trajectory (turn: score / cities / population / techs):")
    for i in range(0, len(turns), stride):
        lines.append(
            f"  {turns[i]}: {m.score_by_turn[i][1]:g} / {m.cities_by_turn[i][1]:g} / "
            f"{m.population_by_turn[i][1]:g} / {m.techs_by_turn[i][1]:g}"
        )
    failed = [s for s in run.steps if not s.result.ok]
    if failed:
        lines.append("Rejected steps:")
        for s in failed[:max_failures]:
            act = action_json(s.action) if s.action is not None else str(s.raw_action)[:120]
            lines.append(f"  turn {s.turn}: {act} -> {s.result.error}")
        if len(failed) > max_failures:
            lines.append(f"  ... and {len(failed) - max_failures} more")
    with_why = [s for s in run.steps if s.reasoning]
    if with_why:
        stride = max(1, len(with_why) // max_samples)
        lines.append("Sample of decisions:")
        for s in with_why[::stride][:max_samples]:
            act = action_json(s.action) if s.action is not None else str(s.raw_action)[:80]
            why = " ".join((s.reasoning or "").split())[:160]
            lines.append(f"  turn {s.turn}: {act} because {why}")
    return "\n".join(lines)


class ReflexionAgent:
    name = "reflexion"

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        base: str = "react",
        memory_path: str | Path = DEFAULT_MEMORY,
        max_reflections: int = 3,
        reflection_tokens: int = 700,
        **base_kwargs,
    ) -> None:
        if base not in BASES:
            raise ValueError(f"base must be one of {sorted(BASES)}, got {base!r}")
        self._client = client
        self.base = base
        self.memory = ReflectionMemory(memory_path)
        self.max_reflections = max_reflections
        self.reflection_tokens = reflection_tokens
        self.inner: Agent = BASES[base](client, **base_kwargs)
        self.loaded = 0

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = make_client()
        return self._client

    def config(self) -> dict:
        return {
            **self.inner.config(),
            "base": self.base,
            "memory_path": str(self.memory.path),
            "max_reflections": self.max_reflections,
            "reflections_loaded": self.loaded,
        }

    def reset(self, seed: int) -> None:
        recent = self.memory.load()[-self.max_reflections :] if self.max_reflections else []
        self.loaded = len(recent)
        self.inner.lessons = lessons_text(recent)
        self.inner.reset(seed)

    def act(self, snapshot: Snapshot) -> Decision:
        return self.inner.act(snapshot)

    def end_game(self, run: Run) -> Usage:
        """Write and store a reflection on the finished game. Returns what it cost."""
        past = self.memory.load()
        prompt = digest_run(run)
        if past:
            prompt += "\n\nReflections you wrote on earlier games:\n" + "\n".join(
                f"- {r.text}" for r in past[-self.max_reflections :]
            )
        prompt += (
            "\n\nWrite a reflection of at most 150 words in plain text, no JSON: what hurt the "
            "result, what helped, and three to five concrete rules for the next game. Refer to "
            "specific actions and turns."
        )
        reply = self.client.complete(
            [{"role": "system", "content": COACH_SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=self.reflection_tokens,
        )
        text = clean_reply(reply.text)
        if text and run.footer is not None:
            final = run.footer.final_snapshot
            self.memory.append(
                Reflection(
                    episode=len(past) + 1,
                    run_id=run.header.run_id,
                    scenario=run.header.scenario,
                    seed=run.header.seed,
                    final_score=final.player.score,
                    text=text,
                    usage=reply.usage,
                )
            )
        return reply.usage
