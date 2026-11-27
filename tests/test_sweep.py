import csv
import json

import pytest
from stubs import PickLegal, chat_server

from napoleon.agents import DirectAgent, GreedyAgent, RandomAgent
from napoleon.cli import main
from napoleon.llm import LLMError, ScriptedClient
from napoleon.sweep import (
    Cell,
    collect_runs,
    parse_seeds,
    run_sweep,
    summarize,
    write_report,
)
from napoleon.usage import Pricing


def basic_factory(cell: Cell):
    return {"random": RandomAgent, "greedy": GreedyAgent}[cell.agent]()


def sweep(tmp_path, **kw):
    kw.setdefault("turns", 25)
    return run_sweep(
        kw.pop("agents", ["random", "greedy"]),
        kw.pop("scenarios", ["default", "bad_opening"]),
        kw.pop("seeds", [0, 1]),
        tmp_path,
        kw.pop("factory", basic_factory),
        **kw,
    )


def test_parse_seeds():
    assert parse_seeds(["0", "2-4", "9"]) == [0, 2, 3, 4, 9]
    assert parse_seeds(["5-5"]) == [5]


def test_every_cell_gets_a_run_file(tmp_path):
    results = sweep(tmp_path)
    assert len(results) == 8 and not any(r.error or r.skipped for r in results)
    assert (tmp_path / "greedy" / "bad_opening-seed1.jsonl").exists()
    rows = collect_runs(tmp_path)
    assert [(r.agent, r.scenario, r.seed) for r in rows][:2] == [
        ("greedy", "bad_opening", 0),
        ("greedy", "bad_opening", 1),
    ]
    assert all(r.metrics.turns_played == 25 and r.reason == "done" for r in rows)


def test_resume_skips_finished_cells_and_redoes_cut_off_ones(tmp_path):
    sweep(tmp_path, agents=["greedy"], scenarios=["default"], seeds=[0, 1])
    first = tmp_path / "greedy" / "default-seed0.jsonl"
    before = first.read_text()
    cut = tmp_path / "greedy" / "default-seed1.jsonl"
    cut.write_text("\n".join(cut.read_text().splitlines()[:3]) + "\n")  # no footer
    results = sweep(tmp_path, agents=["greedy"], scenarios=["default"], seeds=[0, 1], resume=True)
    assert [r.skipped for r in results] == [True, False]
    assert first.read_text() == before
    assert collect_runs(tmp_path)[1].reason == "done"


def test_parallel_matches_serial(tmp_path):
    sweep(tmp_path / "a", workers=1)
    sweep(tmp_path / "b", workers=4)

    def key(d):
        rows = summarize(collect_runs(d))
        return [(r["agent"], r["scenario"], r["score_mean"], r["steps"]) for r in rows]

    assert key(tmp_path / "a") == key(tmp_path / "b")


def test_a_failing_cell_does_not_stop_the_matrix(tmp_path):
    class Flaky(GreedyAgent):
        fail = False

        def act(self, snap):
            if self.fail:
                raise LLMError("server went away")
            return super().act(snap)

    def flaky(cell):
        agent = Flaky()
        agent.fail = cell.seed == 1
        return agent

    seen = []
    results = sweep(
        tmp_path, agents=["greedy"], scenarios=["default"], seeds=[0, 1, 2], factory=flaky,
        on_done=seen.append,
    )  # fmt: skip
    assert [bool(r.error) for r in results] == [False, True, False]
    assert "went away" in results[1].error and len(seen) == 3
    assert {r.seed for r in collect_runs(tmp_path)} == {0, 2}  # the cut-off run has no steps


def test_only_the_last_episode_counts_and_other_jsonl_is_ignored(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "m.jsonl").write_text('{"episode": 1, "text": "x"}\n')
    (tmp_path / "empty.jsonl").write_text("")
    sweep(tmp_path, agents=["greedy"], scenarios=["default"], seeds=[0], episodes=3)
    rows = collect_runs(tmp_path)
    assert len(rows) == 1 and rows[0].episode == 3
    assert rows[0].path.name == "default-seed0-ep3.jsonl"
    assert len(list((tmp_path / "greedy").glob("*.jsonl"))) == 3


def test_summary_statistics_and_recovery(tmp_path):
    sweep(tmp_path, seeds=[0, 1, 2], turns=30)
    rows = collect_runs(tmp_path)
    summary = {(r["agent"], r["scenario"]): r for r in summarize(rows)}
    greedy = [r.metrics.final_score for r in rows if (r.agent, r.scenario) == ("greedy", "default")]
    g = summary[("greedy", "default")]
    assert g["runs"] == 3 and g["score_mean"] == pytest.approx(sum(greedy) / 3, abs=0.01)
    assert g["recovery_ratio"] is None and g["unfinished"] == 0
    bad = [
        r.metrics.final_score for r in rows if (r.agent, r.scenario) == ("greedy", "bad_opening")
    ]
    b = summary[("greedy", "bad_opening")]
    assert b["recovery_ratio"] == pytest.approx((sum(bad) / 3) / (sum(greedy) / 3), abs=0.001)
    assert b["recovered"] is (b["recovery_turn"] is not None)


def test_recovery_needs_a_reference_scenario(tmp_path):
    sweep(tmp_path, agents=["greedy"], scenarios=["bad_opening"], seeds=[0])
    (row,) = summarize(collect_runs(tmp_path))
    assert row["recovery_ratio"] is None and row["recovered"] is None


def test_cost_and_calls_are_aggregated(tmp_path):
    def factory(cell):
        client = ScriptedClient(
            PickLegal(), pricing=Pricing(input_per_mtok=1e6, output_per_mtok=1e6)
        )
        return DirectAgent(client)

    run_sweep(["direct"], ["default"], [0, 1], tmp_path, factory, turns=4)
    (row,) = summarize(collect_runs(tmp_path))
    assert row["runs"] == 2 and row["llm_calls"] > 0 and row["input_tokens"] > 0
    assert row["cost_usd"] > 0 and row["cost_per_run_usd"] == pytest.approx(row["cost_usd"] / 2)
    assert row["illegal_rate"] == 0


def test_unfinished_runs_are_flagged(tmp_path):
    sweep(tmp_path, agents=["random"], scenarios=["default"], seeds=[0], max_steps=5)
    (row,) = summarize(collect_runs(tmp_path))
    assert row["unfinished"] == 1


def test_report_files(tmp_path):
    sweep(tmp_path, seeds=[0, 1])
    summary, out = write_report(tmp_path)
    with (out / "summary.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(summary) == 4 and "recovery_ratio" in rows[0] and "cost_usd" in rows[0]
    with (out / "runs.csv").open() as fh:
        assert len(list(csv.DictReader(fh))) == 8
    md = (out / "summary.md").read_text()
    assert "## Results" in md and "## Recovery from a bad opening" in md
    assert "| greedy | default | 2 |" in md


def test_empty_directory_reports_nothing(tmp_path):
    summary, out = write_report(tmp_path)
    assert summary == [] and (out / "summary.md").exists()


# CLI --------------------------------------------------------------------------------


def test_cli_sweep_and_report(tmp_path, capsys):
    out = tmp_path / "s"
    code = main(
        ["sweep", "--agents", "random", "greedy", "--seeds", "0-1", "--turns", "20"]
        + ["--out", str(out), "--workers", "2"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "## Results" in printed and (out / "summary.csv").exists()
    (out / "summary.md").unlink()
    assert main(["report", str(out)]) == 0
    assert "greedy" in capsys.readouterr().out
    assert main(["report", str(tmp_path / "nothing")]) == 1


def test_cli_sweep_llm_agents_get_their_own_memory_and_skills(tmp_path, capsys):
    out = tmp_path / "s"
    reply = PickLegal()

    def model(messages):
        if "coach" in messages[0]["content"]:
            return "Be bolder."
        return reply(messages)

    with chat_server(model) as (url, seen):
        code = main(
            ["sweep", "--agents", "reflexion", "--base", "direct", "--scenarios", "default"]
            + ["--seeds", "0-1", "--turns", "3", "--episodes", "2", "--out", str(out)]
            + ["--base-url", url, "--price-in", "1", "--price-out", "1"]
        )
    assert code == 0
    summary = json.loads(json.dumps(summarize(collect_runs(out))))[0]
    assert summary["runs"] == 2 and summary["llm_calls"] > 0 and summary["cost_usd"] > 0
    assert sorted(p.name for p in (out / "memory").glob("*.jsonl")) == [
        "reflexion-default-seed0.jsonl",
        "reflexion-default-seed1.jsonl",
    ]
    assert len(seen) > 0
    capsys.readouterr()


def test_cli_sweep_reports_unreachable_models(tmp_path, capsys):
    code = main(
        ["sweep", "--agents", "direct", "--scenarios", "default", "--seeds", "0", "--turns", "3"]
        + ["--out", str(tmp_path / "s"), "--base-url", "http://127.0.0.1:1/v1"]
    )
    assert code == 1
    assert "could not get a reply" in capsys.readouterr().err
