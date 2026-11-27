"""Compact replays for viewers.

A run file is built for exact reloading: it holds whole snapshots and every step. A
replay keeps only what a viewer draws, in short arrays, so a 100-turn game fits in a
few hundred kilobytes. ``docs/replay-format.md`` describes the format.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from napoleon.metrics import RunMetrics, compute_metrics
from napoleon.recording import Run, load_run
from napoleon.schema import Snapshot
from napoleon.usage import Usage

FORMAT = "napoleon-replay"
VERSION = 1


def _frame(snap: Snapshot) -> dict[str, Any]:
    return {
        "turn": snap.turn,
        "player": snap.player.id,
        "score": snap.player.score,
        "gold": round(snap.player.gold, 1),
        "tiles": [[t.coord.q, t.coord.r, t.terrain, t.owner, int(t.visible)] for t in snap.tiles],
        "units": [[u.id, u.owner, u.type, u.coord.q, u.coord.r, u.hp] for u in snap.units],
        "cities": [
            [c.id, c.owner, c.name, c.coord.q, c.coord.r, c.population, c.production]
            for c in snap.cities
        ],
        "tech": snap.tech.current,
        "civic": snap.civics.current,
        "victory": snap.victory_progress,
        "contacts": [[c.player_id, c.civ, int(c.at_war), c.score] for c in snap.contacts],
    }


def _usage(u: Usage | None) -> dict[str, Any] | None:
    if u is None:
        return None
    return {"calls": u.calls, "in": u.input_tokens, "out": u.output_tokens, "cost": u.cost_usd}


def _series(run: Run, m: RunMetrics) -> dict[str, Any]:
    """Per-turn series for the chart. Cost is cumulative and charged to the turn it was spent."""
    rival: dict[int, int] = {}
    for snap in run.turn_snapshots():
        scores = [c.score for c in snap.contacts if c.score is not None]
        if scores:
            rival[snap.turn] = max(scores)
    spent: dict[int, float] = {}
    calls: dict[int, int] = {}
    for step in run.steps:
        if step.usage is not None:
            spent[step.turn] = spent.get(step.turn, 0.0) + step.usage.cost_usd
            calls[step.turn] = calls.get(step.turn, 0) + step.usage.calls
    turns = [t for t, _ in m.score_by_turn]
    total, cost, n_calls = 0.0, [], 0
    cum_calls = []
    for t in turns:
        total += spent.get(t, 0.0)
        n_calls += calls.get(t, 0)
        cost.append([t, round(total, 6)])
        cum_calls.append([t, n_calls])
    return {
        "score": m.score_by_turn,
        "cities": m.cities_by_turn,
        "population": m.population_by_turn,
        "techs": m.techs_by_turn,
        "rival": sorted(rival.items()),
        "cost": cost,
        "calls": cum_calls,
    }


def build_replay(run: Run) -> dict[str, Any]:
    frames: dict[int, dict[str, Any]] = {}
    for snap in run.turn_snapshots():
        frames[snap.turn] = _frame(snap)
    m = compute_metrics(run)
    steps = [
        {
            "step": s.step,
            "turn": s.turn,
            "action": s.action.model_dump(mode="json") if s.action else s.raw_action,
            "reasoning": s.reasoning,
            "ok": s.result.ok,
            "error": s.result.error,
            "events": list(s.result.events),
            "usage": _usage(s.usage),
            "ms": round(s.elapsed_ms),
        }
        for s in run.steps
    ]
    h = run.header
    return {
        "format": FORMAT,
        "version": VERSION,
        "run": {
            "run_id": h.run_id,
            "backend": h.backend,
            "scenario": h.scenario,
            "seed": h.seed,
            "agent": h.agent,
            "started_at": h.started_at,
            "schema_version": h.schema_version,
            "ended": run.footer.reason if run.footer else None,
            "config": h.config,
        },
        "totals": {
            "steps": m.steps,
            "failed_steps": m.failed_steps,
            "illegal_action_rate": m.illegal_action_rate,
            "llm_calls": m.llm_calls,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "cost_usd": m.cost_usd,
            "final_score": m.final_score,
        },
        "frame_fields": {
            "tiles": ["q", "r", "terrain", "owner", "visible"],
            "units": ["id", "owner", "type", "q", "r", "hp"],
            "cities": ["id", "owner", "name", "q", "r", "population", "production"],
        },
        "frames": [frames[t] for t in sorted(frames)],
        "series": _series(run, m),
        "steps": steps,
    }


def render_html(replay: dict[str, Any]) -> str:
    """A self-contained page: hex map per turn, score chart, and each step's reasoning."""
    template = resources.files("napoleon").joinpath("replay_page.html").read_text("utf-8")
    data = json.dumps(replay, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("/*REPLAY_DATA*/null", data)


def write_replay(run_path: str | Path, out: str | Path, html: str | Path | None = None) -> dict:
    replay = build_replay(load_run(run_path))
    Path(out).write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    if html is not None:
        Path(html).write_text(render_html(replay), encoding="utf-8")
    return replay


def build_compare(runs: Sequence[tuple[str, Run]]) -> dict[str, Any]:
    """Series and totals for several runs, labeled, for the comparison page."""
    out = []
    for label, run in runs:
        replay = build_replay(run)
        out.append(
            {
                "label": label,
                "series": replay["series"],
                "totals": replay["totals"],
                "ended": replay["run"]["ended"],
                "cities": replay["series"]["cities"][-1][1],
                "techs": replay["series"]["techs"][-1][1],
            }
        )
    return {"format": "napoleon-compare", "version": 1, "runs": out}


def _label(run: Run) -> str:
    h = run.header
    model = h.config.get("model")
    return f"{h.agent}{f' ({model})' if model else ''} · {h.scenario} seed {h.seed}"


def write_compare(paths: Sequence[str | Path], html: str | Path) -> dict[str, Any]:
    """Write one page overlaying the given run files."""
    runs = [load_run(p) for p in paths]
    data = build_compare([(_label(r), r) for r in runs])
    template = resources.files("napoleon").joinpath("compare_page.html").read_text("utf-8")
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    Path(html).write_text(template.replace("/*COMPARE_DATA*/null", payload), encoding="utf-8")
    return data
