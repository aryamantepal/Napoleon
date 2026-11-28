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


def _add_llm_options(parser: argparse.ArgumentParser) -> None:
    """Options for building an LLM agent, shared by ``run`` and ``sweep``."""
    llm = parser.add_argument_group("LLM agents", "for any agent that talks to a model")
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
    llm.add_argument(
        "--no-think",
        action="store_true",
        help="turn off Qwen3 thinking (reasoning_effort none, plus /no_think)",
    )
    llm.add_argument(
        "--cot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ask for a short reasoning before the action (default); --no-cot asks for the "
        "action alone. Unrelated to --no-think, which controls the model's own thinking.",
    )
    llm.add_argument(
        "--demos",
        type=Path,
        default=None,
        metavar="FILE",
        help="few-shot demonstrations (JSONL from `napoleon demos`); default: greedy play "
        "on fake-game seeds 1000+",
    )
    llm.add_argument("--json-mode", action="store_true", help="send response_format=json_object")
    llm.add_argument("--price-in", type=float, default=0.0, help="USD per million input tokens")
    llm.add_argument("--price-out", type=float, default=0.0, help="USD per million output tokens")
    llm.add_argument("--base", choices=["direct", "react"], default=None, help="reflexion: player")
    llm.add_argument("--max-reflections", type=int, default=None, help="reflexion: lessons shown")
    llm.add_argument("--no-verify", action="store_true", help="voyager: store skills untested")
    cont = parser.add_argument_group("continual harness", "for --agent continual")
    cont.add_argument(
        "--bootstrap-from",
        type=Path,
        default=None,
        metavar="DIR",
        help="start from a saved harness (a .harness directory); frozen unless --bootstrap-update",
    )
    cont.add_argument("--bootstrap-update", action="store_true", help="keep refining it")
    cont.add_argument("--refine-warmup", type=int, default=None, help="first refinement (25)")
    cont.add_argument("--refine-every", type=int, default=None, help="steps between them (25)")
    cont.add_argument("--refine-until", type=int, default=None, help="go slow after step (200)")
    cont.add_argument("--refine-late-every", type=int, default=None, help="late interval (100)")
    cont.add_argument("--max-prompt-chars", type=int, default=None, help="strategy cap (4000)")
    cont.add_argument("--max-tool-calls", type=int, default=None, help="tool calls per step (4)")
    cont.add_argument("--no-rollback", action="store_true", help="never roll back a refinement")
    cont.add_argument(
        "--rollback-illegal-delta",
        type=float,
        default=None,
        help="roll back if the illegal-action rate rises by this much (0.15)",
    )
    cont.add_argument(
        "--rollback-score-margin",
        type=float,
        default=None,
        help="roll back if score gain per turn falls by this fraction (0.5)",
    )


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
    _add_llm_options(run)
    run.add_argument(
        "--shots", type=int, default=0, help="few-shot demonstrations per prompt (0 = zero-shot)"
    )

    refl = run.add_argument_group("episodes and memory")
    refl.add_argument("--episodes", type=int, default=1, help="games to play in a row")
    refl.add_argument(
        "--vary-seed",
        action="store_true",
        help="episode N uses seed+N-1 (default: every episode replays the same seed)",
    )
    refl.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="reflexion: reflection file (default runs/memory/reflexion.jsonl)",
    )
    refl.add_argument(
        "--clear-memory", action="store_true", help="reflexion: start with no lessons"
    )

    sk = run.add_argument_group("skills")
    sk.add_argument(
        "--skills",
        type=Path,
        default=None,
        help="voyager: skill library file (default runs/skills/library.json)",
    )
    sk.add_argument("--clear-skills", action="store_true", help="voyager: start with no skills")

    sweep = sub.add_parser(
        "sweep", help="run an agents x scenarios x seeds matrix and write comparison tables"
    )
    sweep.add_argument("--agents", nargs="+", choices=sorted(AGENTS), required=True)
    sweep.add_argument("--scenarios", nargs="+", default=["default", "bad_opening"])
    sweep.add_argument(
        "--seeds", nargs="+", default=["0", "1", "2"], help="seeds, or ranges like 0-4"
    )
    sweep.add_argument("--turns", type=int, default=50)
    sweep.add_argument("--out", type=Path, default=Path("runs/sweep"))
    sweep.add_argument("--workers", type=int, default=1, help="cells to run in parallel threads")
    sweep.add_argument(
        "--episodes", type=int, default=1, help="games per cell (for learning agents)"
    )
    sweep.add_argument("--max-steps", type=int, default=100_000, help="per game")
    sweep.add_argument(
        "--resume", action="store_true", help="skip cells whose run files are already complete"
    )
    _add_llm_options(sweep)
    sweep.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[0],
        help="few-shot counts to compare, for example 0 3 (LLM agents only)",
    )

    demos = sub.add_parser(
        "demos", help="write a few-shot demonstration file from greedy play or recorded runs"
    )
    demos.add_argument("--out", type=Path, required=True)
    demos.add_argument(
        "--from",
        dest="sources",
        type=Path,
        nargs="*",
        default=[],
        help="finished fake-game run files to replay (default: roll greedy out on --seeds)",
    )
    demos.add_argument("--seeds", nargs="+", default=["1000-1005"], help="for the greedy rollout")
    demos.add_argument("--turns", type=int, default=40)

    report = sub.add_parser("report", help="aggregate the run files in a directory into tables")
    report.add_argument("path", type=Path)
    report.add_argument("--out", type=Path, default=None, help="default: the same directory")

    metrics = sub.add_parser("metrics", help="print metrics for a recorded run")
    metrics.add_argument("path", type=Path)

    compare = sub.add_parser("compare", help="overlay several run files on one HTML page")
    compare.add_argument("paths", type=Path, nargs="+")
    compare.add_argument("--html", type=Path, default=Path("compare.html"))

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
    if args.command == "sweep":
        return _sweep(args)
    if args.command == "demos":
        from napoleon.fewshot import DEMO_SCENARIOS, demos_from_run, greedy_demos, save_demos
        from napoleon.sweep import parse_seeds

        if args.sources:
            found = [d for p in args.sources for d in demos_from_run(p)]
        else:
            found = greedy_demos(parse_seeds(args.seeds), DEMO_SCENARIOS, args.turns)
        save_demos(found, args.out)
        print(json.dumps({"demos": len(found), "out": str(args.out)}))
        return 0
    if args.command == "report":
        from napoleon.sweep import write_report

        summary, out = write_report(args.path, args.out)
        print((out / "summary.md").read_text(encoding="utf-8"))
        return 0 if summary else 1
    if args.command == "metrics":
        from napoleon.metrics import compute_metrics
        from napoleon.recording import load_run

        print(json.dumps(_summary(compute_metrics(load_run(args.path))), indent=2))
        return 0
    if args.command == "compare":
        from napoleon.replay import write_compare

        try:
            data = write_compare(args.paths, args.html)
        except ValueError as exc:  # a run with no snapshots, or a file that is not a run
            print(f"napoleon: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"runs": len(data["runs"]), "html": str(args.html)}))
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


def build_client(args: argparse.Namespace):
    from napoleon.llm import make_client
    from napoleon.usage import Pricing

    return make_client(
        args.model,
        args.base_url,
        args.api_key_env,
        temperature=args.temperature,
        pricing=Pricing(input_per_mtok=args.price_in, output_per_mtok=args.price_out),
        no_think=args.no_think,
        json_mode=args.json_mode,
    )


def _continual_options(args: argparse.Namespace) -> dict:
    from napoleon.harness import Limits
    from napoleon.refiner import RollbackPolicy, Schedule

    def given(**kw):
        return {k: v for k, v in kw.items() if v is not None}

    sched = given(
        warmup=args.refine_warmup,
        interval=args.refine_every,
        until=args.refine_until,
        late_interval=args.refine_late_every,
    )
    roll = given(illegal_delta=args.rollback_illegal_delta, score_margin=args.rollback_score_margin)
    return {
        "schedule": Schedule(**sched),
        "rollback": RollbackPolicy(enabled=not args.no_rollback, **roll),
        "limits": Limits(**given(max_prompt_chars=args.max_prompt_chars)),
        "bootstrap_from": args.bootstrap_from,
        "bootstrap_update": args.bootstrap_update,
        "max_tool_calls": args.max_tool_calls,
    }


def _agent_options(args: argparse.Namespace) -> dict:
    return {
        **_continual_options(args),
        "max_retries": args.max_retries,
        "max_tokens": args.max_tokens,
        "cot": args.cot,
        "demos": args.demos,
        "shots": args.shots if isinstance(args.shots, int) else None,
        "max_inner_steps": args.max_inner_steps,
        "base": args.base,
        "max_reflections": args.max_reflections,
        "verify": False if args.no_verify else None,
    }


def build_agent(args: argparse.Namespace):
    """Make the agent a ``run`` command asked for, with an LLM client where it needs one."""
    if args.agent not in LLM_AGENTS:
        return make_agent(args.agent)
    agent = make_agent(
        args.agent,
        build_client(args),
        memory_path=args.memory,
        skills_path=args.skills,
        **_agent_options(args),
    )
    if args.clear_memory and hasattr(agent, "memory"):
        agent.memory.clear()
    if args.clear_skills and hasattr(agent, "library"):
        agent.library.clear()
    return agent


def _sweep(args: argparse.Namespace) -> int:
    from napoleon.sweep import Cell, parse_seeds, run_sweep, write_report

    out: Path = args.out
    client = build_client(args) if LLM_AGENTS & set(args.agents) else None

    def factory(cell: Cell):
        # Learning agents get their own memory and skill files per cell, so cells never share
        # lessons and can run in parallel.
        tag = f"{cell.agent}-{cell.stem}"  # the stem carries scenario, seed and shots
        agent = make_agent(
            cell.agent,
            client,
            memory_path=out / "memory" / f"{tag}.jsonl",
            skills_path=out / "skills" / f"{tag}.json",
            **{**_agent_options(args), "shots": cell.shots},
        )
        if hasattr(agent, "memory"):
            agent.memory.clear()
        if hasattr(agent, "library"):
            agent.library.clear()
        return agent

    def progress(result) -> None:
        c = result.cell
        state = (
            "skipped" if result.skipped else f"error: {result.error}" if result.error else "done"
        )
        print(f"[{c.agent} {c.scenario} seed {c.seed} shots {c.shots}] {state}", file=sys.stderr)

    results = run_sweep(
        args.agents,
        args.scenarios,
        parse_seeds(args.seeds),
        out,
        factory,
        turns=args.turns,
        episodes=args.episodes,
        workers=args.workers,
        max_steps=args.max_steps,
        resume=args.resume,
        on_done=progress,
        shots=args.shots,
    )
    summary, _ = write_report(out)
    print((out / "summary.md").read_text(encoding="utf-8"))
    failed = [r for r in results if r.error]
    for r in failed:
        print(f"napoleon: {r.cell}: {r.error}", file=sys.stderr)
    return 1 if failed or not summary else 0


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
