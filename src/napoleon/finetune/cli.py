"""The ``napoleon finetune`` subcommands, kept out of ``napoleon.cli`` to keep that readable."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from napoleon.finetune import data as D
from napoleon.finetune.mlx import (
    DEFAULT_MODEL,
    LoraConfig,
    MLXUnavailable,
    lora_command,
    require_mlx,
)


def _lora(args: argparse.Namespace, **defaults) -> LoraConfig:
    cfg = LoraConfig(model=args.model or DEFAULT_MODEL, **defaults)
    given = {
        "iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_layers": args.num_layers,
        "max_seq_length": args.max_seq_length,
    }
    return replace(cfg, **{k: v for k, v in given.items() if v is not None})


def _teacher(args: argparse.Namespace) -> D.Teacher:
    if args.teacher == "greedy":
        return D.GreedyTeacher()
    from napoleon.llm import make_client

    client = make_client(args.teacher_model, args.teacher_base_url, no_think=True)
    return D.LLMTeacher(client, cot=True)


def _seeds(tokens):
    from napoleon.sweep import parse_seeds

    return parse_seeds(tokens)


def _examples(args: argparse.Namespace) -> list[D.Example]:
    teacher = _teacher(args) if args.teacher != "greedy" else None
    if args.sources:
        return D.examples_from_runs(args.sources, teacher)
    return D.greedy_examples(_seeds(args.seeds), turns=args.turns)


def run(args: argparse.Namespace) -> int:
    try:
        return _run(args)
    except MLXUnavailable as exc:
        print(f"napoleon: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    cmd = args.finetune_command
    if cmd == "data":
        counts = D.write_chat_data(_examples(args), args.out, cot=args.cot)
        print(json.dumps({"out": str(args.out), **counts}))
        return 0
    if cmd == "sft":
        examples = _examples(args)
        data_dir = args.out / "data"
        counts = D.write_chat_data(examples, data_dir, cot=args.cot)
        cfg = _lora(args)
        command = lora_command(cfg, data_dir, args.out)
        if args.dry_run:
            print(json.dumps({"data": str(data_dir), **counts, "command": command}, indent=2))
            return 0
        require_mlx()
        from napoleon.finetune.mlx import run_lora

        run_lora(cfg, data_dir, args.out, log_path=args.out / "train.log")
        print(json.dumps({"adapter": str(args.out), **counts, "log": str(args.out / "train.log")}))
        return 0
    if cmd == "dagger":
        require_mlx()
        from napoleon.finetune.dagger import DaggerConfig, dagger

        cfg = DaggerConfig(
            iterations=args.iterations,
            steps_per_iter=args.steps_per_iter,
            window=args.window,
            bad_below=args.bad_below,
            turns=args.turns,
            cot=args.cot,
            lora=_lora(args, learning_rate=2e-5, iters=60),
        )
        rows = dagger(
            args.out,
            _teacher(args),
            cfg,
            model=args.model or DEFAULT_MODEL,
            adapter=args.adapter,
            log=lambda line: print(line, file=sys.stderr, flush=True),
        )
        print(json.dumps({"out": str(args.out), "iterations": rows}, indent=2))
        return 0
    # eval
    require_mlx()
    from napoleon.finetune.evaluate import evaluate_student, greedy_reference
    from napoleon.finetune.mlx import MLXClient

    model = args.model or DEFAULT_MODEL
    scenarios, seeds = args.scenarios, _seeds(args.seeds)
    result = evaluate_student(
        lambda: MLXClient(model, args.adapter),
        args.out,
        scenarios=scenarios,
        seeds=seeds,
        turns=args.turns,
        max_steps=args.max_steps,
        cot=args.cot,
        label=args.label or ("tuned" if args.adapter else "base"),
    )
    result["greedy_score_mean"] = greedy_reference(scenarios, seeds, args.turns)
    print(json.dumps(result, indent=2))
    return 0
