"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from napoleon import __version__
from napoleon.agents import AGENTS, LLM_AGENTS
from napoleon.agents.factory import make_agent
from napoleon.llm import API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_MODEL, LLMError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="napoleon", description=__doc__)
    parser.add_argument("--version", action="version", version=f"napoleon {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="play one game and record it")
    run.add_argument("--agent", choices=sorted(AGENTS), default="greedy")
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
        "--max-steps",
        type=int,
        default=100_000,
        help="stop after this many steps even if the game is not over (default: no limit)",
    )
    run.add_argument(
        "--out", type=Path, default=None, help="run file (default runs/<agent>-<seed>.jsonl)"
    )
    llm = run.add_argument_group("LLM agents", "for any agent that talks to a model")
    llm.add_argument("--model", default=None, help=f"default {DEFAULT_MODEL}")
    llm.add_argument(
        "--base-url",
        default=None,
        help=f"OpenAI-compatible endpoint (default {DEFAULT_BASE_URL}, Ollama)",
    )
    llm.add_argument(
        "--api-key-env",
        default=API_KEY_ENV,
        metavar="NAME",
        help=f"environment variable holding the API key, if the endpoint needs one "
        f"(default {API_KEY_ENV})",
    )
    llm.add_argument("--temperature", type=float, default=0.2)
    llm.add_argument("--max-retries", type=int, default=2, help="retries on unusable output")
    llm.add_argument("--max-tokens", type=int, default=2048, help="per model call")
    llm.add_argument("--max-inner-steps", type=int, default=None, help="react: lookups per step")
    llm.add_argument("--no-think", action="store_true", help="append /no_think (Qwen3)")
    llm.add_argument("--json-mode", action="store_true", help="send response_format=json_object")
    llm.add_argument("--price-in", type=float, default=0.0, help="USD per million input tokens")
    llm.add_argument("--price-out", type=float, default=0.0, help="USD per million output tokens")

    refl = run.add_argument_group("episodes and memory")
    refl.add_argument("--episodes", type=int, default=1, help="games to play in a row")
    refl.add_argument(
        "--vary-seed",
        action="store_true",
        help="episode N uses seed+N-1 (default: every episode replays the same seed)",
    )
    refl.add_argument("--base", choices=["direct", "react"], default=None, help="reflexion: player")
    refl.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="reflexion: reflection file (default runs/memory/reflexion.jsonl)",
    )
    refl.add_argument(
        "--clear-memory", action="store_true", help="reflexion: start with no lessons"
    )
    refl.add_argument("--max-reflections", type=int, default=None, help="reflexion: lessons shown")

    sk = run.add_argument_group("skills")
    sk.add_argument(
        "--skills",
        type=Path,
        default=None,
        help="voyager: skill library file (default runs/skills/library.json)",
    )
    sk.add_argument("--clear-skills", action="store_true", help="voyager: start with no skills")
    sk.add_argument("--no-verify", action="store_true", help="voyager: store skills untested")

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
        from napoleon.env import CivEnv
        from napoleon.fake import FakeCivGame
        from napoleon.runner import run_episodes, run_game

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
        try:
            agent = build_agent(args)
            if args.episodes > 1:
                seeds = [args.seed + (i if args.vary_seed else 0) for i in range(args.episodes)]
                results = run_episodes(
                    env,
                    agent,
                    out.parent,
                    backend=args.backend,
                    stem=out.stem,
                    scenario=args.scenario,
                    seeds=seeds,
                    max_steps=args.max_steps,
                )
                rows = [
                    _summary(m)
                    | {
                        "episode": n,
                        "seed": s,
                        "run_file": str(out.parent / f"{out.stem}-ep{n}.jsonl"),
                    }
                    for n, (m, s) in enumerate(zip(results, seeds, strict=True), 1)
                ]
                print(json.dumps({"episodes": rows}, indent=2))
                return 0
            m = run_game(
                env,
                agent,
                out,
                backend=args.backend,
                scenario=args.scenario,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        except LLMError as exc:
            print(f"napoleon: {exc}", file=sys.stderr)
            return 2
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


def build_agent(args: argparse.Namespace):
    """Make the agent a ``run`` command asked for, with an LLM client where it needs one."""
    if args.agent not in LLM_AGENTS:
        return make_agent(args.agent)
    from napoleon.llm import make_client
    from napoleon.usage import Pricing

    client = make_client(
        args.model,
        args.base_url,
        args.api_key_env,
        temperature=args.temperature,
        pricing=Pricing(input_per_mtok=args.price_in, output_per_mtok=args.price_out),
        no_think=args.no_think,
        json_mode=args.json_mode,
    )
    agent = make_agent(
        args.agent,
        client,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        max_inner_steps=args.max_inner_steps,
        base=args.base,
        memory_path=args.memory,
        max_reflections=args.max_reflections,
        skills_path=args.skills,
        verify=False if args.no_verify else None,
    )
    if args.clear_memory and hasattr(agent, "memory"):
        agent.memory.clear()
    if args.clear_skills and hasattr(agent, "library"):
        agent.library.clear()
    return agent


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
        "llm_calls": m.llm_calls,
        "input_tokens": m.input_tokens,
        "output_tokens": m.output_tokens,
        "cost_usd": round(m.cost_usd, 6),
    }


if __name__ == "__main__":
    raise SystemExit(main())
