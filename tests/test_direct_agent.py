import json

from stubs import PickLegal, chat_server, legal_lines, pick_legal

from napoleon.agents import AGENTS, LLM_AGENTS, DirectAgent
from napoleon.agents.prompting import parse_reply, render_snapshot
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.llm import ScriptedClient
from napoleon.recording import load_run
from napoleon.runner import run_game
from napoleon.schema import EndTurn, MoveUnit, UnitCommand


def _fresh():
    env = CivEnv(FakeCivGame(max_turns=20))
    return env.reset(seed=0)


def test_prompt_lists_state_and_every_legal_action():
    s = _fresh()
    text = render_snapshot(s)
    assert f"Turn {s.turn} of {s.max_turns}" in text
    assert "settler" in text and "Blockers" in text
    assert len(legal_lines([{"role": "user", "content": text}])) == len(s.legal_actions)


def test_parse_reply_shapes():
    nested = parse_reply('{"reasoning": "go", "action": {"type": "end_turn"}}')
    assert nested.action == EndTurn() and nested.reasoning == "go"
    bare = parse_reply('```json\n{"type": "unit_command", "unit_id": 1, "command": "fortify"}\n```')
    assert bare.action == UnitCommand(unit_id=1, command="fortify")
    flat = parse_reply(
        '<think>..</think>{"thought": "t", "type": "move_unit", "unit_id": 2, '
        '"to": {"q": 1, "r": 0}}'
    )
    assert flat.action == MoveUnit(unit_id=2, to={"q": 1, "r": 0}) and flat.reasoning == "t"


def test_parse_reply_failures_say_why():
    assert "no JSON" in parse_reply("I think we should expand").error
    bad = parse_reply('{"action": {"type": "teleport"}}')
    assert bad.action is None and "invalid action" in bad.error and bad.raw == {"type": "teleport"}


def test_agent_registered_and_constructible_without_client():
    assert AGENTS["direct"] is DirectAgent and "direct" in LLM_AGENTS
    assert DirectAgent().name == "direct"


def test_act_returns_action_reasoning_and_usage():
    client = ScriptedClient(pick_legal, model="stub-1")
    d = DirectAgent(client).act(_fresh())
    assert d.reasoning == "stub pick"
    assert d.usage.calls == 1 and d.usage.input_tokens > 0
    assert len(client.calls) == 1


def test_malformed_output_is_retried_then_accepted():
    client = ScriptedClient(["not json", "{bad", pick_legal])
    d = DirectAgent(client, max_retries=2).act(_fresh())
    assert len(client.calls) == 3
    assert d.usage.calls == 3
    assert d.reasoning.startswith("[after 2 retries]")
    # the retry prompt tells the model what went wrong
    assert "rejected" in client.calls[1][-1]["content"]
    assert client.calls[1][-2] == {"role": "assistant", "content": "not json"}


def test_retries_exhausted_passes_raw_output_to_the_env():
    client = ScriptedClient(["I would rather talk"])
    agent = DirectAgent(client, max_retries=1)
    d = agent.act(_fresh())
    assert len(client.calls) == 2
    assert d.action == "I would rather talk" and "unusable output" in d.reasoning
    env = CivEnv(FakeCivGame(max_turns=20))
    env.reset(seed=0)
    _, result = env.step(d.action)
    assert not result.ok and "malformed" in result.error


def test_zero_retries_means_one_call():
    client = ScriptedClient(["junk"])
    DirectAgent(client, max_retries=0).act(_fresh())
    assert len(client.calls) == 1


def test_wellformed_but_illegal_action_is_not_retried():
    client = ScriptedClient(['{"action": {"type": "set_research", "tech": "nonsense"}}'])
    d = DirectAgent(client).act(_fresh())
    assert len(client.calls) == 1 and d.action.tech == "nonsense"


def test_agent_is_stateless_between_decisions():
    client = ScriptedClient(pick_legal)
    agent = DirectAgent(client)
    agent.act(_fresh())
    agent.act(_fresh())
    assert [len(c) for c in client.calls] == [2, 2]


def test_full_game_records_reasoning_usage_and_cost(tmp_path):
    from napoleon.usage import Pricing

    client = ScriptedClient(
        PickLegal(), model="stub-1", pricing=Pricing(input_per_mtok=3, output_per_mtok=15)
    )
    m = run_game(
        CivEnv(FakeCivGame(max_turns=8)),
        DirectAgent(client),
        tmp_path / "d.jsonl",
        backend="fake",
    )
    run = load_run(tmp_path / "d.jsonl")
    assert run.header.agent == "direct" and run.header.config["model"] == "stub-1"
    assert run.footer.reason == "done"
    assert all(s.reasoning == "stub pick" and s.usage.calls == 1 for s in run.steps)
    assert m.llm_calls == len(run.steps) == len(client.calls)
    assert m.input_tokens > 0 and m.output_tokens > 0 and m.cost_usd > 0
    assert m.failed_steps == 0


def test_garbage_model_shows_up_as_illegal_actions(tmp_path):
    m = run_game(
        CivEnv(FakeCivGame(max_turns=8)),
        DirectAgent(ScriptedClient(["???"]), max_retries=0),
        tmp_path / "g.jsonl",
        backend="fake",
        max_steps=10,
    )
    assert m.illegal_action_rate == 1.0 and m.steps == 10
    run = load_run(tmp_path / "g.jsonl")
    assert run.steps[0].raw_action == "???" and run.steps[0].action is None


def test_cli_drives_an_openai_compatible_server(tmp_path, capsys):
    out = tmp_path / "cli.jsonl"
    with chat_server(PickLegal()) as (url, seen):
        code = main(
            [
                "run",
                "--agent",
                "direct",
                "--turns",
                "5",
                "--out",
                str(out),
                "--base-url",
                url,
                "--model",
                "qwen3:8b",
                "--price-in",
                "1",
                "--price-out",
                "2",
            ]
        )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["turns_played"] == 5 and summary["failed_steps"] == 0
    assert summary["llm_calls"] == len(seen) and summary["cost_usd"] > 0
    assert seen[0]["model"] == "qwen3:8b"
    assert load_run(out).header.config["model"] == "qwen3:8b"


def test_cli_reports_an_unreachable_server(tmp_path, capsys):
    code = main(
        [
            "run",
            "--agent",
            "direct",
            "--turns",
            "3",
            "--out",
            str(tmp_path / "x.jsonl"),
            "--base-url",
            "http://127.0.0.1:1/v1",
        ]
    )
    assert code == 2
    assert "could not get a reply" in capsys.readouterr().err
