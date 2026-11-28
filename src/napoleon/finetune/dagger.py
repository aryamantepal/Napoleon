"""Roll out the student, relabel its bad windows, soft-SFT, repeat (reset-free).

Each iteration plays ``steps_per_iter`` steps of the fake game with the current student,
continuing the same game where the last iteration stopped (a new game starts only when one
ends), scores the play in windows with the process reward, has the teacher relabel the
windows that scored badly, adds the examples to everything gathered so far (DAgger's
aggregation), and trains the adapter a little more at a low learning rate (soft SFT). The
game, the examples and the adapter all live in ``out_dir``, so a stopped run resumes.
"""

from __future__ import annotations

import json
import pickle
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from napoleon.agents.direct import DirectAgent
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.finetune import data as D
from napoleon.finetune.mlx import ADAPTER_FILE, LoraConfig, MLXClient, run_lora
from napoleon.llm import LLMClient
from napoleon.metrics import compute_metrics
from napoleon.recording import Recorder, RunHeader, load_run
from napoleon.schema import Snapshot

Trainer = Callable[[Path, Path, bool], None]  # (data dir, adapter dir, resume)
ClientFactory = Callable[[Path | None], LLMClient]  # adapter dir, or None for the base model


@dataclass
class GameState:
    """A fake game in progress that can be saved and picked up by another process."""

    env: CivEnv
    scenario: str
    seed: int
    snapshot: Snapshot
    games_started: int = 1

    @classmethod
    def start(cls, scenarios: Sequence[str], seeds: Sequence[int], n: int, turns: int) -> GameState:
        scenario, seed = scenarios[n % len(scenarios)], seeds[n % len(seeds)] + n // len(seeds)
        env = CivEnv(FakeCivGame(max_turns=turns))
        return cls(env, scenario, seed, env.reset(scenario, seed), n + 1)

    def save(self, path: Path) -> None:
        blob = {"backend": self.env.backend, "scenario": self.scenario, "seed": self.seed,
                "games_started": self.games_started}  # fmt: skip
        path.write_bytes(pickle.dumps(blob))

    @classmethod
    def load(cls, path: Path) -> GameState:
        blob = pickle.loads(path.read_bytes())  # our own file, written by ``save``
        env = CivEnv(blob["backend"])
        env._ready = True
        return cls(env, blob["scenario"], blob["seed"], env.observe(), blob["games_started"])


@dataclass
class DaggerConfig:
    iterations: int = 3
    steps_per_iter: int = 48
    window: int = 8
    bad_below: float = 0.75
    keep_good: bool = True
    max_examples: int = 1500
    turns: int = 30
    scenarios: Sequence[str] = ("default", "bad_opening")
    train_seeds: Sequence[int] = D.TRAIN_SEEDS
    cot: bool = True
    max_tokens: int = 256
    lora: LoraConfig = field(default_factory=lambda: LoraConfig(learning_rate=2e-5, iters=60))


def rollout(
    agent: DirectAgent, state: GameState, steps: int, out_path: Path
) -> tuple[list[D.Move], bool]:
    """Play up to ``steps`` steps from where the game is. Returns the moves and whether the
    game ended. The run is recorded to ``out_path`` like any other run."""
    agent.reset(state.seed)
    header = RunHeader(backend="fake", scenario=state.scenario, seed=state.seed,
                       agent=agent.name, config=agent.config())  # fmt: skip
    moves: list[D.Move] = []
    with Recorder(out_path, header) as rec:
        start = time.perf_counter()
        while len(moves) < steps and not state.snapshot.done:
            seen = state.snapshot
            decision = agent.act(seen)
            after, result = state.env.step(decision.action)
            rec.record(seen, decision.action, result, decision.reasoning,
                       (time.perf_counter() - start) * 1000, decision.usage)  # fmt: skip
            moves.append(D.Move(seen, decision, result, after))
            state.snapshot = after
            start = time.perf_counter()
        rec.close(state.snapshot, reason="done" if state.snapshot.done else "step_limit")
    return moves, state.snapshot.done


def _read_examples(path: Path) -> list[D.Example]:
    if not path.exists():
        return []
    return [D.Example.model_validate_json(ln) for ln in path.read_text().splitlines() if ln]


def _append_examples(path: Path, examples: Sequence[D.Example]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.writelines(ex.model_dump_json() + "\n" for ex in examples)


def default_trainer(cfg: LoraConfig, log: Path) -> Trainer:
    return lambda data, adapter, resume: run_lora(cfg, data, adapter, resume=resume, log_path=log)


def dagger(
    out_dir: str | Path,
    teacher: D.Teacher,
    cfg: DaggerConfig,
    *,
    model: str,
    adapter: str | Path | None = None,
    client_factory: ClientFactory | None = None,
    trainer: Trainer | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run ``cfg.iterations`` rounds in ``out_dir``. Returns one summary dict per round."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapters = out / "adapters"
    if adapter is not None and not (adapters / ADAPTER_FILE).exists():
        shutil.copytree(adapter, adapters, dirs_exist_ok=True)
    cfg.lora = replace(cfg.lora, model=model)
    make_client = client_factory or (lambda path: MLXClient(model, path))
    train = trainer or default_trainer(cfg.lora, out / "train.log")
    state_path, examples_path = out / "state.pkl", out / "examples.jsonl"
    state = (
        GameState.load(state_path)
        if state_path.exists()
        else GameState.start(cfg.scenarios, cfg.train_seeds, 0, cfg.turns)
    )
    log_path = out / "dagger_log.jsonl"
    done_before = len(log_path.read_text().splitlines()) if log_path.exists() else 0
    summaries: list[dict] = []
    for it in range(done_before + 1, done_before + cfg.iterations + 1):
        trained = (adapters / ADAPTER_FILE).exists()
        client = make_client(adapters if trained else None)
        agent = DirectAgent(client, max_retries=0, max_tokens=cfg.max_tokens, cot=cfg.cot)
        if state.snapshot.done:
            state = GameState.start(cfg.scenarios, cfg.train_seeds, state.games_started, cfg.turns)
        started = (state.scenario, state.seed, state.snapshot.turn)
        moves, ended = rollout(agent, state, cfg.steps_per_iter, out / f"iter-{it:02d}.jsonl")
        examples, scores = D.relabel(moves, teacher, window=cfg.window, bad_below=cfg.bad_below,
                                     keep_good=cfg.keep_good)  # fmt: skip
        _append_examples(examples_path, examples)
        state.save(state_path)
        pool = _read_examples(examples_path)[-cfg.max_examples :]
        counts = D.write_chat_data(pool, out / "data", cot=cfg.cot)
        if counts["train"]:
            train(out / "data", adapters, trained)
        metrics = compute_metrics(load_run(out / f"iter-{it:02d}.jsonl"))
        mean_reward = round(sum(s.reward for s in scores) / len(scores), 4) if scores else None
        row = {
            "iteration": it,
            "game": {"scenario": started[0], "seed": started[1], "from_turn": started[2]},
            "steps": len(moves),
            "game_ended": ended,
            "score": state.snapshot.player.score,
            "illegal_rate": round(metrics.illegal_action_rate, 4),
            "format_invalid_rate": round(1 - sum(m.format_ok for m in moves) / len(moves), 4)
            if moves else 0.0,
            "window_reward": mean_reward,
            "bad_windows": sum(s.reward < cfg.bad_below for s in scores),
            "windows": len(scores),
            "new_examples": len(examples),
            "pool": len(pool),
            "trained": bool(counts["train"]),
            "teacher": teacher.name,
        }  # fmt: skip
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        summaries.append(row)
        if log:
            log(json.dumps(row))
    return summaries
