"""Metrics over a recorded run."""

from __future__ import annotations

from statistics import median

from pydantic import BaseModel

from napoleon.recording import Run
from napoleon.usage import Usage, total_usage

Series = list[tuple[int, float]]


class RunMetrics(BaseModel):
    turns_played: int
    steps: int
    failed_steps: int
    illegal_action_rate: float
    final_score: int
    score_by_turn: Series
    cities_by_turn: Series
    population_by_turn: Series
    techs_by_turn: Series
    final_victory_progress: dict[str, float]
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def compute_metrics(run: Run) -> RunMetrics:
    snaps = run.turn_snapshots()
    if not snaps:
        raise ValueError("run has no snapshots")
    by_turn = {s.turn: s for s in snaps}  # last snapshot per turn wins
    turns = sorted(by_turn)
    failed = sum(1 for s in run.steps if not s.result.ok)
    final = snaps[-1]
    usage = total_usage([s.usage for s in run.steps])
    if run.footer is not None:
        usage = usage + (run.footer.overhead_usage or Usage())
    return RunMetrics(
        turns_played=final.turn - snaps[0].turn,
        steps=len(run.steps),
        failed_steps=failed,
        illegal_action_rate=failed / len(run.steps) if run.steps else 0.0,
        final_score=final.player.score,
        score_by_turn=[(t, by_turn[t].player.score) for t in turns],
        cities_by_turn=[(t, len(by_turn[t].cities)) for t in turns],
        population_by_turn=[(t, sum(c.population for c in by_turn[t].cities)) for t in turns],
        techs_by_turn=[(t, len(by_turn[t].tech.completed)) for t in turns],
        final_victory_progress=dict(final.victory_progress),
        llm_calls=usage.calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
    )


def median_trajectory(series: list[Series]) -> Series:
    """Per-turn median across runs, over the turns every run reached."""
    if not series:
        return []
    maps = [dict(s) for s in series]
    common = set(maps[0]).intersection(*maps[1:])
    return [(t, median(m[t] for m in maps)) for t in sorted(common)]


def recovery_turn(series: Series, reference: Series) -> int | None:
    """First turn from which ``series`` stays at or above ``reference`` for good.

    Use it on a bad-opening run against the median of normal-opening runs: the result is
    the turn the agent caught up, or ``None`` if it never did within the shared turns.
    """
    ref = dict(reference)
    shared = [(t, v) for t, v in series if t in ref]
    caught_up: int | None = None
    for t, v in shared:
        if v >= ref[t]:
            if caught_up is None:
                caught_up = t
        else:
            caught_up = None
    return caught_up
