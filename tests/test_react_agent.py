import json

import pytest
from stubs import ReactPlayer, chat_server

from napoleon.agents import AGENTS, LLM_AGENTS, ReActAgent
from napoleon.agents.prompting import render_snapshot
from napoleon.agents.react import ToolCall, parse_step
from napoleon.agents.tools import SnapshotTools, ToolError
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.llm import ScriptedClient
from napoleon.recording import load_run
from napoleon.runner import run_game
from napoleon.schema import EndTurn


@pytest.fixture
def snap():
    return CivEnv(FakeCivGame(max_turns=20)).reset(seed=0)


def lookup(tool, **args):
    return json.dumps({"thought": f"check {tool}", "tool": tool, "args": args})


def commit(action, thought="done looking"):
    return json.dumps({"thought": thought, "action": action})


# Tools ------------------------------------------------------------------------


def test_tools_are_read_only_and_answer_from_the_snapshot(snap):
    tools = SnapshotTools(snap)
    before = snap.model_dump_json()
    settler = next(u for u in snap.units if u.type == "settler")
    assert "settler" in tools.run("inspect_unit", {"unit_id": settler.id})
    assert "found_city" in tools.run("inspect_unit", {"unit_id": settler.id})
    tile = tools.run("inspect_tile", {"q": settler.coord.q, "r": settler.coord.r})
    assert "settler" in tile and "in sight" in tile
    assert "unknown" in tools.run("inspect_tile", {"q": 99, "r": 99})
    assert "end_turn" not in tools.run("list_options", {"type": "unit_command"})
    assert tools.run("list_options", {"type": "nothing"}) == "no legal actions match"
    assert "candidate city sites" in tools.run("find_city_sites", {"unit_id": settler.id})
    assert "no foreign units" in tools.run("threats", {})
    assert snap.model_dump_json() == before


def test_city_lookup_lists_what_it_can_build():
    env = CivEnv(FakeCivGame(max_turns=20))
    s = env.reset(seed=0)
    settler = next(u for u in s.units if u.type == "settler")
    from napoleon.schema import UnitCommand

    s, _ = env.step(UnitCommand(unit_id=settler.id, command="found_city"))
    city = s.cities[0]
    text = SnapshotTools(s).run("inspect_city", {"city_id": city.id})
    assert city.name in text and "can build: " in text and "warrior" in text


@pytest.mark.parametrize(
    "name, args, message",
    [
        ("teleport", {}, "unknown tool"),
        ("inspect_unit", {"unit_id": "x"}, "must be an integer"),
        ("inspect_unit", {"unit_id": 9999}, "no unit"),
        ("inspect_city", {"city_id": 9999}, "no city"),
        ("inspect_tile", {"q": 1}, "'r'"),
        ("list_options", [1], "JSON object"),
    ],
)
def test_bad_calls_raise_readable_errors(snap, name, args, message):
    with pytest.raises(ToolError, match=message):
        SnapshotTools(snap).run(name, args)


def test_compact_prompt_leaves_out_tiles_and_actions(snap):
    compact = render_snapshot(snap, tiles=False, legal=False)
    assert "Known tiles" not in compact and '{"type"' not in compact
    assert f"Legal actions: {len(snap.legal_actions)}" in compact and "list_options" in compact


def test_parse_step():
    call = parse_step('<think>x</think>{"thought": "t", "tool": "threats", "args": {}}')
    assert call == ToolCall("t", "threats", {})
    assert parse_step(commit({"type": "end_turn"})).action == EndTurn()
    assert parse_step("hmm").action is None


# Agent -------------------------------------------------------------------------


def test_registered():
    assert AGENTS["react"] is ReActAgent and "react" in LLM_AGENTS


def test_lookups_then_commit_with_a_readable_trace(snap):
    settler = next(u for u in snap.units if u.type == "settler")
    found = {"type": "unit_command", "unit_id": settler.id, "command": "found_city"}
    client = ScriptedClient(
        [lookup("inspect_unit", unit_id=settler.id), lookup("threats"), commit(found)]
    )
    d = ReActAgent(client).act(snap)
    assert d.action.command == "found_city"
    assert d.usage.calls == 3 and len(client.calls) == 3
    assert "Lookup: inspect_unit" in d.reasoning and "Observation: settler" in d.reasoning
    assert "no foreign units" in d.reasoning and "Action:" in d.reasoning
    # the observation went back to the model with the remaining budget
    assert "(lookups left: 4)" in client.calls[1][-1]["content"]
    assert "(lookups left: 3)" in client.calls[2][-1]["content"]


def test_lookup_errors_become_observations(snap):
    client = ScriptedClient([lookup("inspect_unit", unit_id=9999), commit({"type": "end_turn"})])
    d = ReActAgent(client).act(snap)
    assert "error: no unit 9999" in client.calls[1][-1]["content"]
    assert d.action == EndTurn()


def test_inner_steps_are_bounded(snap):
    client = ScriptedClient([lookup("threats")])  # never commits
    agent = ReActAgent(client, max_inner_steps=2, max_retries=1)
    d = agent.act(snap)
    # 2 lookups, 1 refused lookup, then 1 retry call: 2 + 1 + 1 = max_inner_steps + 1 + max_retries
    assert len(client.calls) == 4
    assert client.calls[3][-1]["content"].startswith("No lookups left")
    assert not isinstance(d.action, EndTurn) and d.reasoning.count("Lookup:") == 2


def test_out_of_lookups_the_model_can_still_commit(snap):
    client = ScriptedClient([lookup("threats"), lookup("threats"), commit({"type": "end_turn"})])
    d = ReActAgent(client, max_inner_steps=2).act(snap)
    assert d.action == EndTurn() and len(client.calls) == 3


def test_unusable_replies_are_retried_then_surface_as_raw_output(snap):
    client = ScriptedClient(["blah", "still blah", "again blah"])
    d = ReActAgent(client, max_retries=2, max_inner_steps=1).act(snap)
    assert d.action == "again blah" and "Rejected reply" in d.reasoning
    ok = ScriptedClient(["blah", commit({"type": "end_turn"})])
    assert ReActAgent(ok).act(snap).action == EndTurn()


def test_recent_actions_are_remembered_within_a_game_and_cleared_on_reset(snap):
    client = ScriptedClient([commit({"type": "end_turn"})])
    agent = ReActAgent(client, memory=2)
    for _ in range(3):
        agent.act(snap)
    prompt = client.calls[-1][1]["content"]
    assert prompt.count("turn 1: {") == 2  # capped at memory=2
    assert "recent actions" not in client.calls[0][1]["content"]
    agent.reset(0)
    agent.act(snap)
    assert "recent actions" not in client.calls[-1][1]["content"]


def test_full_game_records_trace_and_cost(tmp_path):
    client = ScriptedClient(ReactPlayer(), model="stub-react")
    m = run_game(
        CivEnv(FakeCivGame(max_turns=5)),
        ReActAgent(client),
        tmp_path / "r.jsonl",
        backend="fake",
    )
    run = load_run(tmp_path / "r.jsonl")
    assert run.footer.reason == "done" and m.failed_steps == 0
    assert run.header.agent == "react" and run.header.config["max_inner_steps"] == 5
    assert all("Lookup: list_options" in s.reasoning for s in run.steps)
    assert all(s.usage.calls == 2 for s in run.steps)
    assert m.llm_calls == 2 * len(run.steps) and m.cost_usd == 0.0 and m.input_tokens > 0


def test_cli_runs_react_against_a_server(tmp_path, capsys):
    out = tmp_path / "c.jsonl"
    with chat_server(ReactPlayer()) as (url, seen):
        code = main(
            ["run", "--agent", "react", "--turns", "2", "--out", str(out), "--base-url", url]
        )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["turns_played"] == 2 and summary["llm_calls"] == len(seen)


def test_null_arguments_count_as_absent(snap):
    unit = next(u for u in snap.units if u.type == "settler")
    text = SnapshotTools(snap).run("list_options", {"unit_id": unit.id, "city_id": None})
    assert "legal action(s)" in text


def test_rejected_replies_come_back_with_a_format_example(snap):
    client = ScriptedClient(["blah", commit({"type": "end_turn"})])
    ReActAgent(client).act(snap)
    note = client.calls[1][-1]["content"]
    assert "no JSON object found" in note and "Actions look like this: {" in note
