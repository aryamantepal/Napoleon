"""Run an agents x scenarios x seeds matrix, and aggregate run files into comparison tables.

``run_sweep`` plays every cell on the fake game and writes one run file per cell. Cells can
run in parallel threads, which suits LLM agents that spend their time waiting on a model.
``summarize`` turns any directory of run files into one row per agent and scenario, with
score, size, illegal-action rate, cost, and recovery from the bad opening: how the agent's
bad-opening games compare with its own normal-opening games.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev

from napoleon.agents import LLM_AGENTS
from napoleon.agents.base import Agent
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.llm import LLMError
from napoleon.metrics import RunMetrics, compute_metrics, median_trajectory, recovery_turn
from napoleon.recording import load_run
from napoleon.runner import run_episodes

REFERENCE = "default"
BAD_OPENING = "bad_opening"
_EPISODE = re.compile(r"-ep(\d+)$")


@dataclass(frozen=True)
class Cell:
    agent: str
    scenario: str
    seed: int
    shots: int = 0  # few-shot demonstrations; 0 is the zero-shot baseline

    @property
    def stem(self) -> str:
        # Zero-shot cells keep the old name so existing sweeps still resume.
        return f"{self.scenario}-seed{self.seed}" + (f"-shots{self.shots}" if self.shots else "")


@dataclass(frozen=True)
class CellResult:
    cell: Cell
    files: tuple[Path, ...] = ()
    skipped: bool = False
    error: str | None = None


AgentFactory = Callable[[Cell], Agent]


def parse_seeds(tokens: Sequence[str]) -> list[int]:
    """``["0", "2-4"]`` becomes ``[0, 2, 3, 4]``."""
    seeds: list[int] = []
    for tok in tokens:
        lo, sep, hi = tok.partition("-")
        if sep and lo and hi:
            seeds += range(int(lo), int(hi) + 1)
        else:
            seeds.append(int(tok))
    return seeds


def cell_files(out_dir: Path, cell: Cell, episodes: int) -> list[Path]:
    base = out_dir / cell.agent
    if episodes == 1:
        return [base / f"{cell.stem}.jsonl"]
    return [base / f"{cell.stem}-ep{n}.jsonl" for n in range(1, episodes + 1)]


def _finished(path: Path) -> bool:
    try:
        return path.exists() and load_run(path).footer is not None
    except ValueError:
        return False


def run_cell(
    cell: Cell,
    factory: AgentFactory,
    out_dir: Path,
    *,
    turns: int,
    episodes: int = 1,
    max_steps: int = 100_000,
    resume: bool = False,
) -> CellResult:
    files = cell_files(out_dir, cell, episodes)
    if resume and all(_finished(f) for f in files):
        return CellResult(cell, tuple(files), skipped=True)
    agent_dir = out_dir / cell.agent
    agent_dir.mkdir(parents=True, exist_ok=True)
    env = CivEnv(FakeCivGame(max_turns=turns))
    try:
        if episodes == 1:
            from napoleon.runner import run_game

            run_game(
                env,
                factory(cell),
                files[0],
                backend="fake",
                scenario=cell.scenario,
                seed=cell.seed,
                max_steps=max_steps,
            )
        else:
            run_episodes(
                env,
                factory(cell),
                agent_dir,
                backend="fake",
                stem=cell.stem,
                scenario=cell.scenario,
                seeds=[cell.seed] * episodes,
                max_steps=max_steps,
            )
    except LLMError as exc:
        return CellResult(cell, tuple(f for f in files if f.exists()), error=str(exc))
    return CellResult(cell, tuple(files))


def run_sweep(
    agents: Sequence[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    out_dir: str | Path,
    factory: AgentFactory,
    *,
    turns: int = 50,
    episodes: int = 1,
    workers: int = 1,
    max_steps: int = 100_000,
    resume: bool = False,
    on_done: Callable[[CellResult], None] | None = None,
    shots: Sequence[int] = (0,),
) -> list[CellResult]:
    """Play every (agent, scenario, seed, shots) cell. ``factory`` makes a fresh agent per cell.

    Only LLM agents take demonstrations, so the other agents run once, zero-shot."""
    out = Path(out_dir)
    cells = [
        Cell(a, sc, s, k)
        for a in agents
        for k in (shots if a in LLM_AGENTS else (0,))
        for sc in scenarios
        for s in seeds
    ]

    def one(cell: Cell) -> CellResult:
        result = run_cell(
            cell,
            factory,
            out,
            turns=turns,
            episodes=episodes,
            max_steps=max_steps,
            resume=resume,
        )
        if on_done:
            on_done(result)
        return result

    if workers <= 1:
        return [one(c) for c in cells]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, cells))


# Aggregation ------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRow:
    agent: str
    scenario: str
    seed: int
    episode: int
    path: Path
    metrics: RunMetrics
    reason: str | None
    shots: int = 0  # prompting condition, from the run header
    cot: bool = True


def collect_runs(root: str | Path) -> list[RunRow]:
    """Every run file under ``root``. Where a cell has several episodes, only its last counts."""
    best: dict[tuple, RunRow] = {}
    for path in sorted(Path(root).rglob("*.jsonl")):
        try:
            run = load_run(path)
            metrics = compute_metrics(run)
        except (ValueError, KeyError):
            continue  # not a run file, or an empty one
        m = _EPISODE.search(path.stem)
        row = RunRow(
            run.header.agent,
            run.header.scenario,
            run.header.seed,
            int(m.group(1)) if m else 1,
            path,
            metrics,
            run.footer.reason if run.footer else None,
            int(run.header.config.get("shots") or 0),
            bool(run.header.config.get("cot", True)),
        )
        key = (row.agent, row.scenario, row.seed, row.shots, row.cot)
        if key not in best or row.episode > best[key].episode:
            best[key] = row
    return sorted(best.values(), key=lambda r: (r.agent, r.shots, r.cot, r.scenario, r.seed))


def _sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize(rows: Sequence[RunRow]) -> list[dict]:
    """One dict per (agent, prompting condition, scenario), in a stable order."""
    groups: dict[tuple, list[RunRow]] = defaultdict(list)
    for r in rows:
        groups[(r.agent, r.shots, r.cot, r.scenario)].append(r)
    out = []
    for (agent, shots, cot, scenario), rs in sorted(groups.items()):
        ms = [r.metrics for r in rs]
        scores = [m.final_score for m in ms]
        row = {
            "agent": agent,
            "shots": shots,
            "cot": cot,
            "scenario": scenario,
            "runs": len(rs),
            "score_mean": round(mean(scores), 2),
            "score_median": median(scores),
            "score_sd": round(_sd(scores), 2),
            "cities": round(mean(m.cities_by_turn[-1][1] for m in ms), 2),
            "population": round(mean(m.population_by_turn[-1][1] for m in ms), 2),
            "techs": round(mean(m.techs_by_turn[-1][1] for m in ms), 2),
            "illegal_rate": round(mean(m.illegal_action_rate for m in ms), 4),
            "steps": round(mean(m.steps for m in ms), 1),
            "unfinished": sum(1 for r in rs if r.reason != "done"),
            "llm_calls": round(mean(m.llm_calls for m in ms), 1),
            "actions_per_turn": round(mean(m.actions_per_turn for m in ms), 2),
            "calls_per_turn": round(mean(m.calls_per_turn for m in ms), 2),
            "input_tokens": round(mean(m.input_tokens for m in ms)),
            "output_tokens": round(mean(m.output_tokens for m in ms)),
            "cost_usd": round(sum(m.cost_usd for m in ms), 6),
            "cost_per_run_usd": round(mean(m.cost_usd for m in ms), 6),
            "recovery_ratio": None,
            "recovery_turn": None,
            "recovered": None,
        }
        out.append(row)
    _add_recovery(out, rows)
    return out


def _add_recovery(summary: list[dict], rows: Sequence[RunRow]) -> None:
    """Compare each agent's bad-opening games with its own normal-opening games.

    ``recovery_ratio`` is the bad-opening mean final score over the normal-opening one.
    ``recovery_turn`` is the first turn from which the median bad-opening score stays at or
    above the median normal-opening score, over the turns both reached; ``recovered`` says
    whether that happened at all, so an empty turn never reads as an early recovery.
    """
    by = defaultdict(list)
    for r in rows:
        by[(r.agent, r.shots, r.cot, r.scenario)].append(r.metrics)
    for row in summary:
        if row["scenario"] != BAD_OPENING:
            continue
        cond = (row["agent"], row["shots"], row["cot"])
        ref = by.get((*cond, REFERENCE))
        if not ref:
            continue
        ref_mean = mean(m.final_score for m in ref)
        bad = by[(*cond, BAD_OPENING)]
        row["recovery_ratio"] = (
            round(mean(m.final_score for m in bad) / ref_mean, 3) if ref_mean else None
        )
        turn = recovery_turn(
            median_trajectory([m.score_by_turn for m in bad]),
            median_trajectory([m.score_by_turn for m in ref]),
        )
        row["recovery_turn"] = turn
        row["recovered"] = turn is not None


COLUMNS = [
    "agent",
    "shots",
    "cot",
    "scenario",
    "runs",
    "score_mean",
    "score_median",
    "score_sd",
    "cities",
    "population",
    "techs",
    "illegal_rate",
    "steps",
    "unfinished",
    "llm_calls",
    "actions_per_turn",
    "calls_per_turn",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "cost_per_run_usd",
    "recovery_ratio",
    "recovery_turn",
    "recovered",
]


def write_csv(summary: Sequence[dict], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: "" if row[k] is None else row[k] for k in COLUMNS})


def write_runs_csv(rows: Sequence[RunRow], path: str | Path) -> None:
    cols = [
        "agent", "shots", "cot", "scenario", "seed", "episode", "ended", "turns", "steps",
        "final_score",
        "cities", "population", "techs", "illegal_rate", "llm_calls", "input_tokens",
        "output_tokens", "cost_usd", "run_file",
    ]  # fmt: skip
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for r in rows:
            m = r.metrics
            writer.writerow(
                [
                    r.agent, r.shots, r.cot, r.scenario, r.seed, r.episode,
                    r.reason or "cut off", m.turns_played,
                    m.steps, m.final_score, m.cities_by_turn[-1][1], m.population_by_turn[-1][1],
                    m.techs_by_turn[-1][1], round(m.illegal_action_rate, 4), m.llm_calls,
                    m.input_tokens, m.output_tokens, round(m.cost_usd, 6), str(r.path),
                ]
            )  # fmt: skip


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _table(labels: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in labels) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return lines


def markdown(summary: Sequence[dict]) -> str:
    main_cols = [
        ("agent", "agent"), ("shots", "shots"), ("cot", "cot"), ("scenario", "scenario"),
        ("runs", "runs"),
        ("score_mean", "score"), ("score_sd", "sd"), ("cities", "cities"),
        ("techs", "techs"), ("illegal_rate", "illegal rate"), ("steps", "steps"),
        ("llm_calls", "model calls"), ("actions_per_turn", "actions/turn"),
        ("calls_per_turn", "calls/turn"), ("cost_usd", "cost USD"),
    ]  # fmt: skip
    lines = ["## Results", ""]
    lines += _table(
        [label for _, label in main_cols],
        [[row[k] for k, _ in main_cols] for row in summary],
    )
    rec = [r for r in summary if r["scenario"] == BAD_OPENING and r["recovery_ratio"] is not None]
    if rec:
        lines += ["", "## Recovery from a bad opening", ""]
        lines += _table(
            ["agent", "shots", "bad-opening score / normal score", "caught up", "from turn"],
            [
                [r["agent"], r["shots"], r["recovery_ratio"], r["recovered"], r["recovery_turn"]]
                for r in rec
            ],
        )
    unfinished = [r for r in summary if r["unfinished"]]
    if unfinished:
        lines += ["", "Unfinished runs (hit the step limit or were cut off):"]
        lines += [
            f"- {r['agent']} ({r['shots']}-shot) on {r['scenario']}: "
            f"{r['unfinished']} of {r['runs']}"
            for r in unfinished
        ]
    return "\n".join(lines) + "\n"


def write_report(root: str | Path, out_dir: str | Path | None = None) -> tuple[list[dict], Path]:
    """Aggregate every run file under ``root`` into summary.csv, runs.csv and summary.md."""
    rows = collect_runs(root)
    summary = summarize(rows)
    out = Path(out_dir) if out_dir else Path(root)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(summary, out / "summary.csv")
    write_runs_csv(rows, out / "runs.csv")
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary, out
