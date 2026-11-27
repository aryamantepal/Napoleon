"""Play one game with one agent, record it, and return its metrics."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from napoleon.agents.base import Agent
from napoleon.env import CivEnv
from napoleon.metrics import RunMetrics, compute_metrics
from napoleon.recording import Recorder, Run, RunFooter, RunHeader, StepRecord, load_run


def run_game(
    env: CivEnv,
    agent: Agent,
    out: str | Path,
    *,
    backend: str,
    scenario: str = "default",
    seed: int = 0,
    max_steps: int = 100_000,
) -> RunMetrics:
    """Run until the game ends or ``max_steps`` is hit. The run file is written either way."""
    agent.reset(seed)
    attach = getattr(agent, "attach_run", None)
    if attach is not None:
        attach(Path(out))  # lets an agent keep files next to the run file
    snap = env.reset(scenario, seed)
    config = agent.config() if hasattr(agent, "config") else {}
    header = RunHeader(
        backend=backend, scenario=scenario, seed=seed, agent=agent.name, config=config
    )
    observe = getattr(agent, "observe_result", None)
    reason = "done"
    records: list[StepRecord] = []
    overhead = None
    with Recorder(out, header) as rec:
        for _ in range(max_steps):
            if snap.done:
                break
            start = time.perf_counter()
            decision = agent.act(snap)
            nxt, result = env.step(decision.action)
            elapsed = (time.perf_counter() - start) * 1000
            records.append(
                rec.record(
                    snap,
                    decision.action,
                    result,
                    decision.reasoning,
                    elapsed,
                    decision.usage,
                    decision.events,
                )
            )
            if observe is not None:
                observe(decision, result, nxt)
            snap = nxt
        else:
            reason = "step_limit"
        end_game = getattr(agent, "end_game", None)
        if end_game is not None:
            # Agents that learn between games see the finished run before its footer is written,
            # and report what the learning cost so the run's total does too.
            footer = RunFooter(final_snapshot=snap, steps=len(records), reason=reason)
            overhead = end_game(Run(header=header, steps=records, footer=footer))
        rec.close(snap, reason=reason, overhead_usage=overhead)
    return compute_metrics(load_run(out))


def run_episodes(
    env: CivEnv,
    agent: Agent,
    out_dir: str | Path,
    *,
    backend: str,
    stem: str,
    scenario: str = "default",
    seeds: Sequence[int] = (0,),
    max_steps: int = 100_000,
) -> list[RunMetrics]:
    """Play one game per seed in order with the same agent object, writing ``<stem>-ep<N>.jsonl``.

    The agent is not rebuilt between games, so anything it keeps (a reflection memory, a skill
    library) carries from one episode to the next. Repeat a seed to give it the same game again.
    """
    results = []
    for n, seed in enumerate(seeds, 1):
        path = Path(out_dir) / f"{stem}-ep{n}.jsonl"
        results.append(
            run_game(
                env, agent, path, backend=backend, scenario=scenario, seed=seed, max_steps=max_steps
            )
        )
    return results
