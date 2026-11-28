import json

import pytest
from stubs import PickLegal, chat_server

from napoleon.agents import DirectAgent, ReActAgent, ReflexionAgent, VoyagerAgent
from napoleon.agents.continual import ContinualAgent
from napoleon.agents.factory import make_agent
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.fewshot import (
    DEMO_SEEDS,
    FewShot,
    default_pool,
    demos_from_run,
    greedy_demos,
    load_demos,
    render_summary,
    save_demos,
    situation_features,
)
from napoleon.llm import ScriptedClient
from napoleon.recording import load_run
from napoleon.runner import run_game
from napoleon.sweep import Cell, collect_runs, run_sweep, summarize


@pytest.fixture(scope="module")
def pool():
    return greedy_demos(seeds=(1000, 1001), turns=25)


@pytest.fixture
def snap():
    return CivEnv(FakeCivGame(max_turns=20)).reset(seed=0)


def test_pool_is_greedy_play_on_seeds_away_from_evaluation(pool):
    assert {d.seed for d in pool} == {1000, 1001} and min(DEMO_SEEDS) >= 1000
    assert all(d.action in d.snapshot.legal_actions for d in pool)
    assert len(default_pool()) > 100


def test_picks_are_similar_diverse_and_deterministic(pool, snap):
    fs = FewShot(3, demos=pool)
    fs.reset(0)
    picked = fs.pick(snap)
    assert len(picked) == 3 and picked == fs.pick(snap)
    assert len({d.action.type for d in picked}) == 3  # diversified
    # the opening has a settler that can found a city, so the nearest demo is one that did
    assert "can_found" in situation_features(picked[0].snapshot)
    assert fs.pick(snap)[0].action.type == "unit_command"


def test_the_evaluated_seed_is_never_a_source(pool, snap):
    fs = FewShot(5, demos=pool)
    fs.reset(1000)
    assert all(d.seed != 1000 for d in fs.pick(snap))
    assert fs.config()["demo_pool"] < len(fs.pool)


def test_messages_alternate_and_respect_cot(pool, snap):
    fs = FewShot(2, demos=pool)
    msgs = fs.messages(snap, "thought")
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 2
    first = json.loads(msgs[1]["content"])
    assert "thought" in first and first["action"]["type"]
    assert (
        msgs[0]["content"].startswith("Example 1 of 2.") and "Legal actions:" in msgs[0]["content"]
    )
    bare = FewShot(2, demos=pool, cot=False).messages(snap)
    assert all(list(json.loads(m["content"])) == ["action"] for m in bare[1::2])
    assert "Legal actions" in render_summary(snap)


def test_demo_file_round_trip_and_replay_from_a_recorded_run(tmp_path):
    env = CivEnv(FakeCivGame(max_turns=10))
    from napoleon.agents import GreedyAgent

    run_game(env, GreedyAgent(), tmp_path / "g.jsonl", backend="fake", seed=7)
    demos = demos_from_run(tmp_path / "g.jsonl")
    n_steps = len(load_run(tmp_path / "g.jsonl").steps)
    assert len(demos) == n_steps and {d.seed for d in demos} == {7}
    save_demos(demos, tmp_path / "d.jsonl")
    assert load_demos(tmp_path / "d.jsonl") == demos
    # a run that does not replay is refused
    lines = (tmp_path / "g.jsonl").read_text().splitlines()
    bad = json.loads(lines[3])
    bad["snapshot_hash"] = "0" * 16
    lines[3] = json.dumps(bad)
    (tmp_path / "bad.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="does not replay"):
        demos_from_run(tmp_path / "bad.jsonl")


def test_direct_agent_puts_shots_between_system_and_user(pool, snap):
    client = ScriptedClient(PickLegal())
    agent = DirectAgent(client, shots=2)
    agent.fewshot = FewShot(2, demos=pool)
    agent.reset(0)
    agent.act(snap)
    roles = [m["role"] for m in client.calls[0]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert "reasoning" in client.calls[0][0]["content"]
    zero = ScriptedClient(PickLegal())
    DirectAgent(zero).act(snap)
    assert [m["role"] for m in zero.calls[0]] == ["system", "user"]


def test_no_cot_changes_the_instruction_for_every_llm_agent(snap):
    for cls in (DirectAgent, ReActAgent, VoyagerAgent, ContinualAgent):
        kw = {"skills_path": None} if cls is VoyagerAgent else {}
        for cot in (True, False):
            client = ScriptedClient('{"action": {"type": "end_turn"}}')
            cls(client, cot=cot, **kw).act(snap)
            system = client.calls[0][0]["content"]
            assert ("Do not explain your choice" in system) != cot or cls is DirectAgent
            if cls is DirectAgent:
                assert ('{"action": <one entry' in system) != cot


def test_condition_is_recorded_in_the_run_header(tmp_path):
    agent = make_agent("direct", ScriptedClient(PickLegal()), shots=3, cot=False)
    env = CivEnv(FakeCivGame(max_turns=3))
    run_game(env, agent, tmp_path / "r.jsonl", backend="fake", seed=0, max_steps=5)
    cfg = load_run(tmp_path / "r.jsonl").header.config
    assert cfg["shots"] == 3 and cfg["cot"] is False and cfg["demos"] == "greedy"
    base = make_agent("direct", ScriptedClient(PickLegal()))
    assert base.config()["shots"] == 0 and base.config()["cot"] is True
    refl = make_agent("reflexion", ScriptedClient(PickLegal()), base="direct", shots=2)
    assert isinstance(refl, ReflexionAgent) and refl.config()["shots"] == 2


def factory_for(shots_seen):
    def factory(cell: Cell):
        shots_seen.append((cell.agent, cell.shots, cell.stem))
        return make_agent(cell.agent, ScriptedClient(PickLegal()), shots=cell.shots)

    return factory


def test_sweep_splits_by_shots_and_report_groups_them(tmp_path):
    seen = []
    run_sweep(
        ["direct", "greedy"], ["default"], [0], tmp_path, factory_for(seen), turns=4, shots=[0, 2],
    )  # fmt: skip
    # the non-LLM agent ignores the shots dimension
    assert sorted(seen) == [
        ("direct", 0, "default-seed0"),
        ("direct", 2, "default-seed0-shots2"),
        ("greedy", 0, "default-seed0"),
    ]
    rows = collect_runs(tmp_path)
    assert [(r.agent, r.shots) for r in rows] == [("direct", 0), ("direct", 2), ("greedy", 0)]
    summary = summarize(rows)
    assert [(r["agent"], r["shots"], r["cot"]) for r in summary] == [
        ("direct", 0, True),
        ("direct", 2, True),
        ("greedy", 0, True),
    ]
    assert (tmp_path / "direct" / "default-seed0-shots2.jsonl").exists()


def test_cli_run_and_sweep_take_shots(tmp_path, capsys):
    with chat_server(PickLegal()) as (url, seen):
        out = tmp_path / "x.jsonl"
        code = main(
            ["run", "--agent", "direct", "--turns", "3", "--shots", "2", "--no-cot",
             "--base-url", url, "--out", str(out)]
        )  # fmt: skip
        assert code == 0
        assert [m["role"] for m in seen[0]["messages"]].count("assistant") == 2
        assert load_run(out).header.config["shots"] == 2
        sweep = tmp_path / "sw"
        code = main(
            ["sweep", "--agents", "direct", "--scenarios", "default", "--seeds", "0",
             "--turns", "3", "--shots", "0", "2", "--base-url", url, "--out", str(sweep)]
        )  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    assert "| direct | 0 | yes |" in printed and "| direct | 2 | yes |" in printed


def test_demos_command_writes_a_file(tmp_path, capsys):
    out = tmp_path / "d.jsonl"
    assert main(["demos", "--out", str(out), "--seeds", "1000", "--turns", "10"]) == 0
    assert json.loads(capsys.readouterr().out)["demos"] == len(load_demos(out)) > 5
