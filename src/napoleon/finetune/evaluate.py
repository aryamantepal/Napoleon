"""Compare students (base model, tuned adapter) on held-out fake-game seeds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import mean

from napoleon.agents.direct import DirectAgent
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.llm import LLMClient
from napoleon.recording import load_run
from napoleon.runner import run_game


def evaluate_student(
    make_client: Callable[[], LLMClient],
    out_dir: str | Path,
    *,
    scenarios: Sequence[str] = ("default",),
    seeds: Sequence[int] = (0, 1),
    turns: int = 12,
    max_steps: int = 120,
    cot: bool = True,
    max_tokens: int = 256,
    label: str = "student",
) -> dict:
    """Play the direct agent on each (scenario, seed) with no retries and summarize.

    ``illegal_rate`` is the share of steps the game rejected (illegal or malformed);
    ``format_invalid_rate`` the share whose reply did not parse as an action at all. A game
    ends at ``turns`` or after ``max_steps`` steps, whichever comes first, so a model that
    dithers is scored on what it got done.
    """
    client = make_client()
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            path = Path(out_dir) / f"{label}-{scenario}-seed{seed}.jsonl"
            agent = DirectAgent(client, max_retries=0, max_tokens=max_tokens, cot=cot)
            env = CivEnv(FakeCivGame(max_turns=turns))
            m = run_game(env, agent, path, backend="fake", scenario=scenario, seed=seed,
                         max_steps=max_steps)  # fmt: skip
            run = load_run(path)
            bad_format = sum(s.action is None for s in run.steps)
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "score": m.final_score,
                    "turns": m.turns_played,
                    "steps": m.steps,
                    "illegal_rate": m.illegal_action_rate,
                    "format_invalid_rate": bad_format / max(1, m.steps),
                }
            )
    return {
        "label": label,
        "model": client.model,
        "games": len(rows),
        "score_mean": round(mean(r["score"] for r in rows), 2),
        "turns_mean": round(mean(r["turns"] for r in rows), 2),
        "illegal_rate": round(mean(r["illegal_rate"] for r in rows), 4),
        "format_invalid_rate": round(mean(r["format_invalid_rate"] for r in rows), 4),
        "rows": rows,
    }


def greedy_reference(scenarios: Sequence[str], seeds: Sequence[int], turns: int) -> float:
    """Greedy's mean score on the same games, for scale. It never plays an illegal action."""
    from napoleon.agents.greedy import GreedyAgent

    scores = []
    for scenario in scenarios:
        for seed in seeds:
            env, agent = CivEnv(FakeCivGame(max_turns=turns)), GreedyAgent()
            snap = env.reset(scenario, seed)
            while not snap.done:
                snap, _ = env.step(agent.act(snap).action)
            scores.append(snap.player.score)
    return round(mean(scores), 2)
