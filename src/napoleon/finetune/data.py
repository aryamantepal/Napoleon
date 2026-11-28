"""Training pairs, teachers, and the process reward that decides which play gets relabeled."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from napoleon.agents.base import Decision
from napoleon.agents.prompting import direct_messages
from napoleon.schema import Action, Snapshot, StepResult

# Seeds for building training data. Evaluation uses small seeds (0 up) and the few-shot demo
# pool uses 1000 up, so these never overlap with either.
TRAIN_SEEDS = tuple(range(2000, 2008))


class Example(BaseModel):
    """One supervised pair: the snapshot the agent saw, and the action it should take."""

    model_config = ConfigDict(frozen=True)

    snapshot: Snapshot
    action: Action
    reasoning: str | None = None
    source: str = "teacher"


def target_text(action: Action, reasoning: str | None, cot: bool = True) -> str:
    body: dict = {"action": action.model_dump(mode="json")}
    if cot and reasoning:
        body = {"reasoning": reasoning, **body}
    return json.dumps(body, separators=(",", ":"))


def to_chat(ex: Example, cot: bool = True) -> dict:
    """An mlx-lm chat-format record: the direct agent's exact messages plus the target reply."""
    messages = direct_messages(ex.snapshot, cot)
    reply = target_text(ex.action, ex.reasoning, cot)
    return {"messages": [*messages, {"role": "assistant", "content": reply}]}


# Teachers --------------------------------------------------------------------------------


class Teacher(Protocol):
    name: str

    def label(self, snapshot: Snapshot) -> tuple[Action, str | None]: ...


class GreedyTeacher:
    """Behavior cloning target: the rule-based greedy agent's action. Free and always legal."""

    name = "greedy"

    def __init__(self) -> None:
        from napoleon.agents.greedy import GreedyAgent

        self._agent = GreedyAgent()

    def label(self, snapshot: Snapshot) -> tuple[Action, str | None]:
        d = self._agent.act(snapshot)
        return d.action, d.reasoning


class LLMTeacher:
    """A stronger model labels the state (the paper's DAgger relabel step).

    The answer must be one of the snapshot's legal actions. If the model's reply is unusable
    or illegal the greedy teacher answers instead, so a label is always valid.
    """

    def __init__(self, client, *, cot: bool = True, max_tokens: int = 2048) -> None:
        from napoleon.agents.direct import DirectAgent

        self._agent = DirectAgent(client, cot=cot, max_tokens=max_tokens)
        self._fallback = GreedyTeacher()
        self.name = f"llm:{client.model}"
        self.fallbacks = 0

    def label(self, snapshot: Snapshot) -> tuple[Action, str | None]:
        d = self._agent.act(snapshot)
        if isinstance(d.action, BaseModel) and d.action in snapshot.legal_actions:
            return d.action, d.reasoning
        self.fallbacks += 1
        return self._fallback.label(snapshot)


# Process reward --------------------------------------------------------------------------


@dataclass(frozen=True)
class Move:
    """One step of play: what the agent saw, what it did, and what the game said."""

    snapshot: Snapshot
    decision: Decision
    result: StepResult
    after: Snapshot

    @property
    def format_ok(self) -> bool:
        return isinstance(self.decision.action, BaseModel)


class WindowScore(BaseModel):
    reward: float
    legal_rate: float
    format_rate: float
    score_gain: int
    steps: int


def window_reward(moves: Sequence[Move], gain_ref: float = 1.0) -> WindowScore:
    """A simple process reward for a run of steps, in [0, 1].

    0.4 for the share of actions the game accepted, 0.3 for the share of replies that parsed
    as an action at all, and 0.3 for score gained over the window (full marks at ``gain_ref``
    points, none for no gain). Legality and format are separate because a model can fail
    either way: unparseable text, or a well-formed action that is not legal.
    """
    n = len(moves)
    legal = sum(m.result.ok for m in moves) / n
    fmt = sum(m.format_ok for m in moves) / n
    gain = moves[-1].after.player.score - moves[0].snapshot.player.score
    reward = 0.4 * legal + 0.3 * fmt + 0.3 * min(1.0, max(0.0, gain / gain_ref))
    return WindowScore(
        reward=round(reward, 4), legal_rate=legal, format_rate=fmt, score_gain=gain, steps=n
    )


def split_windows(moves: Sequence[Move], size: int) -> list[Sequence[Move]]:
    return [moves[i : i + size] for i in range(0, len(moves), size) if moves[i : i + size]]


def relabel(
    moves: Sequence[Move],
    teacher: Teacher,
    *,
    window: int = 8,
    bad_below: float = 0.75,
    keep_good: bool = True,
) -> tuple[list[Example], list[WindowScore]]:
    """Examples from a rollout: bad windows get the teacher's action on every state, good
    windows keep the student's own (accepted) action when ``keep_good``."""
    examples: list[Example] = []
    scores: list[WindowScore] = []
    for win in split_windows(moves, window):
        score = window_reward(win)
        scores.append(score)
        if score.reward < bad_below:
            for m in win:
                action, why = teacher.label(m.snapshot)
                examples.append(
                    Example(
                        snapshot=m.snapshot,
                        action=action,
                        reasoning=why,
                        source=f"relabel:{teacher.name}",
                    )
                )
        elif keep_good:
            examples += [
                Example(
                    snapshot=m.snapshot,
                    action=m.decision.action,
                    reasoning=m.decision.reasoning,
                    source="student",
                )  # fmt: skip
                for m in win
                if m.result.ok and m.format_ok
            ]
    return examples, scores


# Building a dataset ----------------------------------------------------------------------


def examples_from_runs(
    paths: Iterable[str | Path], teacher: Teacher | None = None
) -> list[Example]:
    """Label every state of finished fake-game runs (replayed from their files).

    With no teacher the recorded (accepted) actions are the labels, so a greedy run gives
    plain behavior cloning. With a teacher the states are kept and the teacher relabels them,
    which is how a student's own run becomes DAgger data.
    """
    from napoleon.fewshot import replay_steps

    out: list[Example] = []
    for path in paths:
        for snap, step in replay_steps(path):
            if teacher is not None:
                action, why = teacher.label(snap)
                out.append(
                    Example(snapshot=snap, action=action, reasoning=why, source=teacher.name)
                )
            elif step.result.ok and step.action is not None:
                out.append(
                    Example(
                        snapshot=snap,
                        action=step.action,
                        reasoning=step.reasoning,
                        source=f"run:{Path(path).name}",
                    )
                )
    return out


def greedy_examples(
    seeds: Sequence[int] = TRAIN_SEEDS,
    scenarios: Sequence[str] = ("default", "bad_opening"),
    turns: int = 40,
) -> list[Example]:
    """Behavior-cloning data from fresh greedy games on the training seeds."""
    from napoleon.env import CivEnv
    from napoleon.fake import FakeCivGame

    teacher, out = GreedyTeacher(), []
    for scenario in scenarios:
        for seed in seeds:
            env = CivEnv(FakeCivGame(max_turns=turns))
            snap = env.reset(scenario, seed)
            while not snap.done:
                action, why = teacher.label(snap)
                out.append(Example(snapshot=snap, action=action, reasoning=why, source="greedy"))
                snap, _ = env.step(action)
    return out


def write_chat_data(
    examples: Sequence[Example],
    out_dir: str | Path,
    *,
    cot: bool = True,
    valid_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, int]:
    """Write ``train.jsonl`` and ``valid.jsonl`` in the layout ``mlx_lm lora --data`` reads."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    n_valid = max(1, round(len(order) * valid_fraction)) if len(order) > 1 else 0
    splits = {"valid": order[:n_valid], "train": order[n_valid:]}
    for name, idx in splits.items():
        lines = [json.dumps(to_chat(examples[i], cot)) for i in idx]
        (out / f"{name}.jsonl").write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
    return {name: len(idx) for name, idx in splits.items()}
