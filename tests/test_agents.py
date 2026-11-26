import json

import pytest

from napoleon.agents import AGENTS, Agent, GreedyAgent, RandomAgent
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import SCENARIOS, FakeCivGame
from napoleon.recording import load_run
from napoleon.runner import run_game


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_agents_satisfy_protocol(name):
    assert isinstance(AGENTS[name](), Agent)


@pytest.mark.parametrize("agent_cls", [RandomAgent, GreedyAgent])
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sanity_agents_finish_cleanly(tmp_path, agent_cls, scenario, seed):
    env = CivEnv(FakeCivGame(max_turns=40))
    m = run_game(
        env, agent_cls(), tmp_path / "r.jsonl", backend="fake", scenario=scenario, seed=seed
    )
    assert m.failed_steps == 0
    assert load_run(tmp_path / "r.jsonl").footer.reason == "done"
    assert m.turns_played >= 1


def test_greedy_beats_random_on_score(tmp_path):
    def score(agent, seed):
        env = CivEnv(FakeCivGame(max_turns=60))
        return run_game(
            env, agent, tmp_path / f"{agent.name}{seed}.jsonl", backend="fake", seed=seed
        ).final_score

    greedy = sum(score(GreedyAgent(), s) for s in range(3))
    rand = sum(score(RandomAgent(), s) for s in range(3))
    assert greedy > rand


def test_greedy_expands(tmp_path):
    env = CivEnv(FakeCivGame(max_turns=80))
    m = run_game(env, GreedyAgent(), tmp_path / "g.jsonl", backend="fake", seed=0)
    assert m.cities_by_turn[-1][1] >= 2


def test_step_limit_is_recorded(tmp_path):
    env = CivEnv(FakeCivGame(max_turns=40))
    run_game(env, RandomAgent(), tmp_path / "s.jsonl", backend="fake", max_steps=5)
    run = load_run(tmp_path / "s.jsonl")
    assert run.footer.reason == "step_limit"
    assert len(run.steps) == 5


def test_cli_run_and_metrics(tmp_path, capsys):
    out = tmp_path / "cli.jsonl"
    assert main(["run", "--agent", "greedy", "--turns", "10", "--out", str(out)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["turns_played"] == 10
    assert main(["metrics", str(out)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["final_score"] == first["final_score"]
