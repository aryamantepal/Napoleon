"""Play one game with one agent, record it, and return its metrics."""

from __future__ import annotations

import time
from pathlib import Path

from napoleon.agents.base import Agent
from napoleon.env import CivEnv
from napoleon.metrics import RunMetrics, compute_metrics
from napoleon.recording import Recorder, RunHeader, load_run


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
    snap = env.reset(scenario, seed)
    header = RunHeader(backend=backend, scenario=scenario, seed=seed, agent=agent.name)
    reason = "done"
    with Recorder(out, header) as rec:
        for _ in range(max_steps):
            if snap.done:
                break
            start = time.perf_counter()
            decision = agent.act(snap)
            nxt, result = env.step(decision.action)
            elapsed = (time.perf_counter() - start) * 1000
            rec.record(snap, decision.action, result, decision.reasoning, elapsed)
            snap = nxt
        else:
            reason = "step_limit"
        rec.close(snap, reason=reason)
    return compute_metrics(load_run(out))
