import json

import pytest
from stubs import PickLegal, ReactPlayer, chat_server

from napoleon.agents import AGENTS, LLM_AGENTS, GreedyAgent, ReflexionAgent
from napoleon.agents.factory import make_agent
from napoleon.agents.reflexion import Reflection, ReflectionMemory, digest_run, lessons_text
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.llm import ScriptedClient
from napoleon.recording import load_run
from napoleon.runner import run_episodes, run_game


def coach_or_play(reflection="Expand sooner and stop flipping production."):
    """Answers the coach's prompt with a reflection and every other prompt as a player."""
    player = PickLegal()

    def reply(messages):
        if "coach" in messages[0]["content"]:
            return reflection
        return player(messages)

    return reply


def is_coach(call):
    return "coach" in call[0]["content"]


def reflection(n=1, text="lesson"):
    return Reflection(episode=n, run_id="r", scenario="default", seed=0, final_score=9, text=text)


def test_memory_roundtrip_and_clear(tmp_path):
    mem = ReflectionMemory(tmp_path / "deep" / "m.jsonl")
    assert mem.load() == []
    mem.append(reflection(1, "a"))
    mem.append(reflection(2, "b"))
    assert [r.text for r in ReflectionMemory(tmp_path / "deep" / "m.jsonl").load()] == ["a", "b"]
    mem.clear()
    assert mem.load() == []
    mem.clear()  # clearing nothing is fine


def test_lessons_text():
    assert lessons_text([]) == ""
    text = lessons_text([reflection(1, "first\nsecond")])
    assert "Game 1 (default, seed 0, final score 9)" in text and "  second" in text


def test_digest_has_metrics_failures_and_sampled_decisions(tmp_path):
    client = ScriptedClient(["junk", "junk", "junk"])  # a model that never produces an action
    from napoleon.agents import DirectAgent

    run_game(
        CivEnv(FakeCivGame(max_turns=5)),
        DirectAgent(client, max_retries=0),
        tmp_path / "bad.jsonl",
        backend="fake",
        max_steps=6,
    )
    text = digest_run(load_run(tmp_path / "bad.jsonl"))
    assert "100% illegal-action rate" in text and "Rejected steps:" in text
    assert "malformed action" in text and "Sample of decisions:" in text
    run_game(CivEnv(FakeCivGame(max_turns=30)), GreedyAgent(), tmp_path / "g.jsonl", backend="fake")
    good = digest_run(load_run(tmp_path / "g.jsonl"))
    assert "Trajectory" in good and "Rejected steps" not in good
    assert len(good.splitlines()) < 40


def test_registered_and_validated(tmp_path):
    assert AGENTS["reflexion"] is ReflexionAgent and "reflexion" in LLM_AGENTS
    with pytest.raises(ValueError, match="base must be"):
        ReflexionAgent(base="voyager")
    assert make_agent("reflexion", None, base="direct", max_inner_steps=3).base == "direct"
    assert make_agent("greedy", None, max_retries=1).name == "greedy"
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("nope")


def test_reset_loads_only_the_latest_reflections(tmp_path):
    path = tmp_path / "m.jsonl"
    mem = ReflectionMemory(path)
    for i in range(1, 5):
        mem.append(reflection(i, f"lesson {i}"))
    agent = ReflexionAgent(ScriptedClient("x"), base="direct", memory_path=path, max_reflections=2)
    agent.reset(0)
    assert agent.loaded == 2 and "lesson 4" in agent.inner.lessons
    assert "lesson 3" in agent.inner.lessons and "lesson 2" not in agent.inner.lessons
    assert agent.config()["reflections_loaded"] == 2 and agent.config()["base"] == "direct"


def test_reflection_is_written_after_a_game_and_billed_to_it(tmp_path):
    from napoleon.usage import Pricing

    client = ScriptedClient(coach_or_play(), pricing=Pricing(input_per_mtok=1e6))
    agent = ReflexionAgent(client, base="direct", memory_path=tmp_path / "m.jsonl")
    m = run_game(CivEnv(FakeCivGame(max_turns=4)), agent, tmp_path / "r.jsonl", backend="fake")
    run = load_run(tmp_path / "r.jsonl")
    saved = ReflectionMemory(tmp_path / "m.jsonl").load()
    assert len(saved) == 1 and saved[0].text.startswith("Expand sooner")
    assert saved[0].run_id == run.header.run_id and saved[0].final_score == m.final_score
    assert run.footer.overhead_usage.calls == 1
    steps_calls = sum(s.usage.calls for s in run.steps)
    assert m.llm_calls == steps_calls + 1  # the reflection counts toward the run
    assert m.cost_usd == pytest.approx(
        sum(s.usage.cost_usd for s in run.steps) + run.footer.overhead_usage.cost_usd
    )
    assert sum(is_coach(c) for c in client.calls) == 1


def test_empty_reflection_is_not_stored(tmp_path):
    client = ScriptedClient(coach_or_play(reflection="<think>hm</think>"))
    agent = ReflexionAgent(client, base="direct", memory_path=tmp_path / "m.jsonl")
    run_game(CivEnv(FakeCivGame(max_turns=3)), agent, tmp_path / "r.jsonl", backend="fake")
    assert ReflectionMemory(tmp_path / "m.jsonl").load() == []


def test_later_games_see_earlier_reflections(tmp_path):
    client = ScriptedClient(coach_or_play("Never flip production twice."))
    agent = ReflexionAgent(client, base="direct", memory_path=tmp_path / "m.jsonl")
    results = run_episodes(
        CivEnv(FakeCivGame(max_turns=3)),
        agent,
        tmp_path,
        backend="fake",
        stem="reflex",
        seeds=[0, 0, 1],
    )
    assert len(results) == 3
    assert [p.name for p in sorted(tmp_path.glob("reflex-ep*.jsonl"))] == [
        "reflex-ep1.jsonl",
        "reflex-ep2.jsonl",
        "reflex-ep3.jsonl",
    ]
    loaded = [load_run(tmp_path / f"reflex-ep{n}.jsonl").header for n in (1, 2, 3)]
    assert [h.config["reflections_loaded"] for h in loaded] == [0, 1, 2]
    assert [h.seed for h in loaded] == [0, 0, 1]
    play_calls = [c for c in client.calls if not is_coach(c)]
    assert "Lessons from your previous games" not in play_calls[0][0]["content"]
    assert "Never flip production twice." in play_calls[-1][0]["content"]
    assert len(ReflectionMemory(tmp_path / "m.jsonl").load()) == 3


def test_memory_persists_across_agent_instances(tmp_path):
    path = tmp_path / "m.jsonl"
    env = CivEnv(FakeCivGame(max_turns=3))
    first = ReflexionAgent(ScriptedClient(coach_or_play("one")), base="direct", memory_path=path)
    run_game(env, first, tmp_path / "a.jsonl", backend="fake")
    client = ScriptedClient(coach_or_play("two"))
    second = ReflexionAgent(client, base="direct", memory_path=path)
    run_game(env, second, tmp_path / "b.jsonl", backend="fake")
    assert "one" in client.calls[0][0]["content"]
    assert [r.text for r in ReflectionMemory(path).load()] == ["one", "two"]


def test_react_base_plays_and_reflects(tmp_path):
    def reply(messages):
        if "coach" in messages[0]["content"]:
            return "Look before moving."
        return react(messages)

    react = ReactPlayer()
    agent = ReflexionAgent(ScriptedClient(reply), base="react", memory_path=tmp_path / "m.jsonl")
    m = run_game(CivEnv(FakeCivGame(max_turns=3)), agent, tmp_path / "r.jsonl", backend="fake")
    assert m.failed_steps == 0
    assert load_run(tmp_path / "r.jsonl").header.config["max_inner_steps"] == 5
    assert len(ReflectionMemory(tmp_path / "m.jsonl").load()) == 1


def test_run_episodes_works_for_any_agent(tmp_path):
    results = run_episodes(
        CivEnv(FakeCivGame(max_turns=5)),
        GreedyAgent(),
        tmp_path,
        backend="fake",
        stem="greedy",
        seeds=[0, 0],
    )
    assert results[0].final_score == results[1].final_score  # same seed, same game


def test_cli_episodes_with_reflexion(tmp_path, capsys):
    memory = tmp_path / "mem.jsonl"
    out = tmp_path / "x.jsonl"
    with chat_server(coach_or_play("Settle early.")) as (url, seen):
        code = main(
            ["run", "--agent", "reflexion", "--base", "direct", "--episodes", "2", "--turns", "3",
             "--out", str(out), "--base-url", url, "--memory", str(memory), "--clear-memory"]
        )  # fmt: skip
    assert code == 0
    rows = json.loads(capsys.readouterr().out)["episodes"]
    assert [r["episode"] for r in rows] == [1, 2] and [r["seed"] for r in rows] == [0, 0]
    assert rows[0]["run_file"].endswith("x-ep1.jsonl")
    assert len(ReflectionMemory(memory).load()) == 2
    assert sum(1 for r in seen if "coach" in r["messages"][0]["content"]) == 2
    # a fresh run with --clear-memory forgets
    with chat_server(coach_or_play("Settle early.")) as (url, _):
        main(["run", "--agent", "reflexion", "--base", "direct", "--turns", "3", "--out", str(out),
              "--base-url", url, "--memory", str(memory), "--clear-memory"])  # fmt: skip
    assert len(ReflectionMemory(memory).load()) == 1


def test_cli_vary_seed(tmp_path, capsys):
    out = tmp_path / "g.jsonl"
    main(
        ["run", "--agent", "greedy", "--episodes", "3", "--vary-seed", "--seed", "4"]
        + ["--turns", "3", "--out", str(out)]
    )
    rows = json.loads(capsys.readouterr().out)["episodes"]
    assert [r["seed"] for r in rows] == [4, 5, 6]
