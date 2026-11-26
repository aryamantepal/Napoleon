"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from napoleon import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="napoleon", description=__doc__)
    parser.add_argument("--version", action="version", version=f"napoleon {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="play one game and record it")
    run.add_argument("--agent", choices=["random", "greedy"], default="greedy")
    run.add_argument(
        "--backend",
        choices=["fake", "civ6", "civ6-scripted"],
        default="fake",
        help="fake game, real Civ VI through civ6-mcp, or civ6-mcp's scripted stand-in",
    )
    run.add_argument(
        "--save",
        action="append",
        default=[],
        metavar="SCENARIO=SAVE",
        help="civ6 only: the save to load for a scenario (repeatable)",
    )
    run.add_argument("--port", type=int, default=4318, help="civ6 only: FireTuner port")
    run.add_argument("--scenario", default="default")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--turns", type=int, default=100)
    run.add_argument(
        "--out", type=Path, default=None, help="run file (default runs/<agent>-<seed>.jsonl)"
    )

    metrics = sub.add_parser("metrics", help="print metrics for a recorded run")
    metrics.add_argument("path", type=Path)

    replay = sub.add_parser("replay", help="turn a run file into a replay JSON and HTML page")
    replay.add_argument("path", type=Path)
    replay.add_argument("--out", type=Path, default=None, help="default: <run>.replay.json")
    replay.add_argument("--html", type=Path, default=None, help="default: <run>.replay.html")
    replay.add_argument("--no-html", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from napoleon.agents import AGENTS
        from napoleon.env import CivEnv
        from napoleon.fake import FakeCivGame
        from napoleon.runner import run_game

        name = f"{args.backend}-{args.agent}-{args.scenario}-{args.seed}.jsonl"
        out = args.out or Path("runs") / name
        if args.backend == "fake":
            backend = FakeCivGame(max_turns=args.turns)
        else:
            from napoleon.civ6 import Civ6Backend

            saves = dict(s.split("=", 1) for s in args.save)
            if args.backend == "civ6":
                from napoleon.civ6.live import connect

                backend = connect(saves, port=args.port, max_turns=args.turns)
            else:
                from napoleon.civ6.scripted import ScriptedSource

                saves = saves or {args.scenario: "napoleon_default"}
                backend = Civ6Backend(ScriptedSource(), saves, max_turns=args.turns)
        env = CivEnv(backend)
        m = run_game(
            env,
            AGENTS[args.agent](),
            out,
            backend=args.backend,
            scenario=args.scenario,
            seed=args.seed,
        )
        print(json.dumps(_summary(m) | {"run_file": str(out)}, indent=2))
        return 0
    if args.command == "metrics":
        from napoleon.metrics import compute_metrics
        from napoleon.recording import load_run

        print(json.dumps(_summary(compute_metrics(load_run(args.path))), indent=2))
        return 0
    if args.command == "replay":
        from napoleon.replay import write_replay

        stem = args.path.with_suffix("")
        out = args.out or stem.with_suffix(".replay.json")
        html = None if args.no_html else (args.html or stem.with_suffix(".replay.html"))
        replay = write_replay(args.path, out, html)
        files = {"replay": str(out)} | ({"html": str(html)} if html else {})
        print(json.dumps({"turns": len(replay["frames"]), "steps": len(replay["steps"])} | files))
        return 0
    build_parser().print_help()
    return 0


def _summary(m) -> dict:
    return {
        "turns_played": m.turns_played,
        "steps": m.steps,
        "failed_steps": m.failed_steps,
        "final_score": m.final_score,
        "cities": int(m.cities_by_turn[-1][1]),
        "population": int(m.population_by_turn[-1][1]),
        "techs": int(m.techs_by_turn[-1][1]),
        "victory_progress": m.final_victory_progress,
    }


if __name__ == "__main__":
    raise SystemExit(main())
