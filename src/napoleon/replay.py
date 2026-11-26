"""Compact replays for viewers.

A run file is built for exact reloading: it holds whole snapshots and every step. A
replay keeps only what a viewer draws, in short arrays, so a 100-turn game fits in a
few hundred kilobytes. ``docs/replay-format.md`` describes the format.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from napoleon.metrics import compute_metrics
from napoleon.recording import Run, load_run
from napoleon.schema import Snapshot

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
        },
        "frame_fields": {
            "tiles": ["q", "r", "terrain", "owner", "visible"],
            "units": ["id", "owner", "type", "q", "r", "hp"],
            "cities": ["id", "owner", "name", "q", "r", "population", "production"],
        },
        "frames": [frames[t] for t in sorted(frames)],
        "series": {
            "score": m.score_by_turn,
            "cities": m.cities_by_turn,
            "population": m.population_by_turn,
            "techs": m.techs_by_turn,
        },
        "steps": steps,
    }


def render_html(replay: dict[str, Any]) -> str:
    """A self-contained page: hex map per turn, score chart, and each step's reasoning."""
    template = resources.files("napoleon").joinpath("replay_page.html").read_text("utf-8")
    data = json.dumps(replay, separators=(",", ":")).replace("</", "<\\/")
    return template.replace("/*REPLAY_DATA*/null", data)


def write_replay(run_path: str | Path, out: str | Path, html: str | Path | None = None) -> dict:
    replay = build_replay(load_run(run_path))
    Path(out).write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
    if html is not None:
        Path(html).write_text(render_html(replay), encoding="utf-8")
    return replay
