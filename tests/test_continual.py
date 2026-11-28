import json

import pytest
from stubs import PickLegal, legal_lines
from test_skills import SETTLE_AT, TIDY

from napoleon import harness as hx
from napoleon.agents import AGENTS, LLM_AGENTS, ContinualAgent
from napoleon.agents.factory import make_agent
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.llm import ScriptedClient
from napoleon.metrics import compute_metrics
from napoleon.recording import load_run
from napoleon.refiner import (
    Refiner,
    RollbackPolicy,
    Schedule,
    StepLog,
    parse_edits,
    parse_markdown,
    stall_flags,
    window_stats,
)
from napoleon.replay import build_replay
from napoleon.runner import run_game
from napoleon.skills import Verification


def ok_verifier(skill, args):
    return Verification(True, "ok", 1)


class Stub:
    """Plays like PickLegal when acting and answers each refiner pass from a table."""

    def __init__(self, refine=None, sub=None, act=None):
        self.refine = refine or {}
        self.sub = sub
        self.act = act or PickLegal()
        self.refine_calls = 0

    def __call__(self, messages):
        system, user = messages[0]["content"], messages[1]["content"]
        if system.startswith("You improve the harness"):
            self.refine_calls += 1
            key = user.split("PASS: ")[1].split(".")[0]
            reply = self.refine.get(key, "{}")
            return reply(user) if callable(reply) else reply
        if "You are a sub-agent" in system:
            return self.sub(messages) if self.sub else '{"note": "nothing to add"}'
        return self.act(messages)


def agent_with(stub, tmp_path, **kw):
    kw.setdefault("schedule", Schedule(warmup=5, interval=5, until=1000))
    kw.setdefault("verifier", ok_verifier)
    return ContinualAgent(ScriptedClient(stub), harness_dir=tmp_path / "h", **kw)


@pytest.fixture
def snap():
    return CivEnv(FakeCivGame(max_turns=20)).reset(seed=0)


# Pure pieces ---------------------------------------------------------------------------


def test_schedule_warmup_then_interval_then_late():
    s = Schedule(warmup=25, interval=25, until=200, late_interval=100)
    due = [n for n in range(0, 520) if s.due(n)]
    assert due == [25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500]
    assert not Schedule(warmup=3).due(0)


def log(step, turn, score=0, action="a", ok=True):
    return StepLog(step, turn, score, "state", action=action, ok=ok)


def test_stall_flags_and_window_stats():
    stuck = [log(i, 3, action=f"x{i % 2}", ok=i % 5 != 0) for i in range(12)]
    assert any("did not advance" in f for f in stall_flags(stuck))
    repeat = [log(i, i, action="same") for i in range(6)]
    assert any("repeated 6 times" in f for f in stall_flags(repeat))
    streak = [log(i, i, action=str(i), ok=i > 3) for i in range(8)]
    assert any("4 illegal" in f for f in stall_flags(streak))
    assert stall_flags([log(i, i, action=str(i)) for i in range(10)]) == []
    stats = window_stats([log(0, 1, 2), log(1, 3, 10, ok=False)])
    assert stats.illegal_rate == 0.5 and stats.score_trend == 4.0


def test_rollback_policy_needs_a_clear_regression():
    p = RollbackPolicy(min_steps=5)
    base = window_stats([log(i, i, score=i * 4) for i in range(10)])
    worse_illegal = window_stats([log(i, i, score=i * 4, ok=i % 2 == 0) for i in range(10)])
    worse_score = window_stats([log(i, i, score=i) for i in range(10)])
    same = window_stats([log(i, i, score=i * 4 - 1) for i in range(10)])
    assert "illegal-action rate" in p.worse(base, worse_illegal)
    assert "score trend" in p.worse(base, worse_score)
    assert p.worse(base, same) is None
    assert RollbackPolicy(enabled=False).worse(base, worse_score) is None
    assert p.worse(window_stats(log_n(3)), worse_score) is None  # too little evidence


def log_n(n):
    return [log(i, i, score=i * 4) for i in range(n)]


def test_parse_edits_is_lenient():
    text = (
        'Sure!\n```json\n{"analysis": "x", "create": {"title": "t"}, "remove": [{"id": 4}, 7]}\n```'
    )
    analysis, adds, updates, retires = parse_edits(text)
    assert (analysis, adds, updates, retires) == ("x", [{"title": "t"}], [], [4, 7])
    with pytest.raises(ValueError):
        parse_edits("no json here")
    assert parse_markdown("<think>hm</think>```markdown\n# S\n- a\n```") == "# S\n- a"


# Harness edits -------------------------------------------------------------------------


def test_prompt_cap_and_subagent_clamping():
    h, lim = hx.Harness(), hx.Limits(max_prompt_chars=50)
    assert hx.set_prompt(h, "x" * 51, lim)[0].startswith("rejected")
    assert hx.set_prompt(h, "# short", lim)[0].startswith("ok")
    msg = hx.add_subagent(
        h,
        {
            "name": "scout",
            "description": "looks around",
            "directive": "find threats",
            "allowed_tools": ["threats", "rm_rf", "run_skill", "read_memory"],
            "max_turns": 99,
        },
        lim,
    )
    sub = h.subagents[0]
    assert msg.startswith("ok") and sub.max_turns == 10
    assert sub.allowed_tools == ["threats", "read_memory"]
    assert hx.add_subagent(h, {"name": "scout"}, lim).startswith("rejected")


def test_skill_must_parse_and_pass_verification():
    h, lim = hx.Harness(), hx.Limits()
    bad_dsl = {**TIDY, "name": "bad_one", "rules": [{"do": {"type": "launch_nukes"}}]}
    assert hx.add_skill(h, bad_dsl, lim, ok_verifier).startswith("rejected")
    failing = lambda s, a: Verification(False, "wrecks the game")  # noqa: E731
    assert "wrecks the game" in hx.add_skill(h, TIDY, lim, failing)
    assert hx.add_skill(h, TIDY, lim).startswith("ok")  # the real fake-game verification
    assert h.skills[0].body.verification["fired"] > 0
    assert "already exists" in hx.add_skill(h, TIDY, lim, ok_verifier)
    # a parametrized skill is verified only when example_args are given
    assert "schema checked" in hx.add_skill(h, SETTLE_AT, lim)
    assert len(h.skills) == 2


def test_memory_evicts_the_weakest_and_clamps_importance():
    h, lim = hx.Harness(), hx.Limits(max_memory=2, max_memory_chars=10)
    hx.add_memory(h, {"title": "a", "content": "1" * 50, "importance": 9}, lim)
    hx.add_memory(h, {"title": "b", "content": "two", "importance": 1}, lim)
    assert h.memory[0].importance == 5 and len(h.memory[0].content) == 10
    hx.add_memory(h, {"title": "c", "content": "three", "importance": 2}, lim)
    assert [m.title for m in h.memory] == ["a", "c"]
    assert hx.add_memory(h, {"title": "d", "content": "x", "importance": 1}, lim).startswith(
        "rejected"
    )


# The refiner ---------------------------------------------------------------------------


def test_each_pass_is_isolated_and_edits_are_applied():
    boom = "not json at all"
    replies = {
        "strategy prompt": "```markdown\n# New\n- expand\n```",
        "sub-agents": boom,
        "skills": json.dumps({"analysis": "a", "create": [TIDY]}),
        "memory": json.dumps(
            {"add": [{"title": "lesson", "content": "end turns", "importance": 4}]}
        ),
    }
    stub = Stub(replies)
    refiner = Refiner(ScriptedClient(stub), "FIXED", verifier=ok_verifier)
    out = refiner.refine(hx.Harness(), [log(i, 1) for i in range(5)])
    assert out.passes["subagents"].status == "error" and "no JSON" in out.passes["subagents"].error
    assert out.passes["prompt"].status == "ok" and out.harness.prompt == "# New\n- expand"
    assert [s.name for s in out.harness.skills] == ["tidy_up"]
    assert out.harness.memory[0].title == "lesson"
    assert out.changed and out.harness.generation == 1 and out.usage.calls == 4


def test_refiner_sees_the_fixed_prompt_and_the_evidence():
    seen = []
    client = ScriptedClient(lambda m: seen.append(m) or "{}")
    window = [StepLog(7, 2, 5, "S", "because", "ACT", False, "illegal action: boom", ["t: bad"])]
    Refiner(client, "FIXED-PROMPT").refine(hx.Harness(), window)
    system, user = seen[0][0]["content"], seen[0][1]["content"]
    assert "FIXED-PROMPT" in system and "cannot edit the fixed system prompt" in system
    assert "REJECTED illegal action: boom" in user and "t: bad" in user


# The agent -----------------------------------------------------------------------------


def test_registered_and_built_by_the_factory(tmp_path):
    assert AGENTS["continual"] is ContinualAgent and "continual" in LLM_AGENTS
    made = make_agent("continual", None, bootstrap_from=tmp_path, bootstrap_update=True)
    assert made.refine and made.bootstrap_from == tmp_path
    assert not make_agent("continual", None, bootstrap_from=tmp_path).refine
    assert make_agent("continual", None).refine


def run(agent, tmp_path, turns=20, name="run.jsonl", max_steps=2000):
    env = CivEnv(FakeCivGame(max_turns=turns))
    m = run_game(env, agent, tmp_path / name, backend="fake", seed=0, max_steps=max_steps)
    return m, load_run(tmp_path / name)


def test_a_game_refines_persists_generations_and_records_events(tmp_path):
    stub = Stub(
        {
            "strategy prompt": lambda u: "# Gen\n- prefer cities",
            "memory": lambda u: json.dumps(
                {"add": [{"title": f"note {u.count('[')}", "content": "c", "importance": 3}]}
            ),
        }
    )
    agent = agent_with(stub, tmp_path, rollback=RollbackPolicy(enabled=False))
    m, recorded = run(agent, tmp_path)
    assert m.steps > 20 and stub.refine_calls >= 8
    events = [e for s in recorded.steps for e in s.agent_events]
    refined = [e for e in events if e["type"] == "harness_refined"]
    assert len(refined) >= 2 and refined[0]["step"] == 5 and refined[0]["changed"]
    assert refined[0]["passes"]["prompt"]["status"] == "ok"
    d = tmp_path / "h"
    assert (d / "harness.json").exists() and (d / "prompt_evolution" / "gen-001.md").exists()
    assert (d / "prompt_evolution" / "gen-000.md").read_text() == hx.DEFAULT_PROMPT
    lines = [json.loads(ln) for ln in (d / "evolution_log.jsonl").read_text().splitlines()]
    assert [ln["kind"] for ln in lines] == ["refine"] * len(refined)
    assert hx.load_harness(d).memory
    # refinement calls are charged to the step that triggered them, and the replay shows them
    first = next(s for s in recorded.steps if s.agent_events)
    assert first.usage.calls >= 5
    replay = build_replay(recorded)
    assert any(s["agent_events"] for s in replay["steps"])
    assert recorded.header.config["schedule"]["warmup"] == 5
    assert compute_metrics(recorded).calls_per_turn > 1


def test_run_file_dir_is_the_default_harness_dir(tmp_path):
    agent = ContinualAgent(ScriptedClient(Stub()), schedule=Schedule(warmup=5, interval=5))
    run(agent, tmp_path, turns=8, name="g.jsonl")
    assert (tmp_path / "g.harness" / "harness.json").exists()


def test_worse_window_rolls_the_harness_back(tmp_path):
    # Good play until the first refinement installs "# Bad", after which the model sends nonsense.
    class Act:
        def __init__(self):
            self.good = PickLegal()

        def __call__(self, messages):
            if "# Bad" in messages[0]["content"]:
                return json.dumps({"action": {"type": "set_research", "tech": "nope"}})
            return self.good(messages)

    stub = Stub({"strategy prompt": "# Bad"}, act=Act())
    policy = RollbackPolicy(min_steps=3, illegal_delta=0.2)
    agent = agent_with(stub, tmp_path, rollback=policy, max_retries=0)
    _, recorded = run(agent, tmp_path, turns=12, max_steps=40)
    events = [e for s in recorded.steps for e in s.agent_events]
    rollbacks = [e for e in events if e["type"] == "harness_rollback"]
    assert rollbacks and "illegal-action rate" in rollbacks[0]["reason"]
    log_kinds = [
        json.loads(ln)["kind"]
        for ln in (tmp_path / "h" / "evolution_log.jsonl").read_text().splitlines()
    ]
    assert "rollback" in log_kinds
    # after the rollback the strategy text is back to the default before the next rewrite
    assert agent.rollbacks >= 1


def test_bootstrap_is_frozen_unless_told_to_update(tmp_path):
    src = hx.Harness(prompt="# Saved", generation=3)
    hx.add_memory(src, {"title": "known", "content": "x", "importance": 5}, hx.Limits())
    hx.save_harness(src, tmp_path / "src")
    stub = Stub({"strategy prompt": "# Rewritten"})
    frozen = agent_with(stub, tmp_path, bootstrap_from=tmp_path / "src")
    _, recorded = run(frozen, tmp_path, turns=8)
    assert stub.refine_calls == 0 and frozen.h.prompt == "# Saved"
    assert not any(s.agent_events for s in recorded.steps)
    assert recorded.header.config["bootstrap_from"] and not recorded.header.config["refining"]
    stub2 = Stub({"strategy prompt": "# Rewritten"})
    live = agent_with(stub2, tmp_path, bootstrap_from=tmp_path / "src", bootstrap_update=True)
    run(live, tmp_path, turns=8, name="live.jsonl")
    assert live.h.prompt == "# Rewritten" and live.h.generation > 3 and live.h.memory


def test_overview_tools_skills_and_subagents(tmp_path, snap):
    h = hx.Harness()
    hx.add_memory(
        h, {"title": "settle early", "content": "found by turn 3", "importance": 5}, hx.Limits()
    )
    hx.add_skill(h, TIDY, hx.Limits(), ok_verifier)
    hx.add_subagent(
        h,
        {
            "name": "scout",
            "description": "checks threats",
            "directive": "Report threats.",
            "allowed_tools": ["threats"],
            "max_turns": 2,
        },
        hx.Limits(),
    )
    hx.save_harness(h, tmp_path / "boot")
    replies = [
        json.dumps({"tool": "read_memory", "args": {"id": 1}}),
        json.dumps({"tool": "write_memory", "args": {"title": "t", "content": "c"}}),
        json.dumps({"tool": "delegate", "args": {"name": "scout"}}),
        json.dumps({"tool": "bogus", "args": {}}),
        json.dumps({"tool": "run_skill", "args": {"name": "tidy_up"}}),
    ]
    sub_replies = ['{"note": "all quiet"}']
    seq = iter(replies)

    def stub(messages):
        if "You are a sub-agent" in messages[0]["content"]:
            return sub_replies[0]
        return next(seq)

    agent = ContinualAgent(
        ScriptedClient(stub), bootstrap_from=tmp_path / "boot", max_retries=1, max_tool_calls=4
    )
    agent.reset(0)
    d = agent.act(snap)
    prompts = agent.client.calls
    assert "- 1 [5] settle early" in prompts[0][1]["content"]
    assert (
        "tidy_up()" in prompts[0][1]["content"]
        and "scout: checks threats" in prompts[0][1]["content"]
    )
    assert "found by turn 3" in prompts[1][-1]["content"]
    assert any(m["title"] == "t" for m in map(lambda e: e.model_dump(), agent.h.memory))
    assert "Sub-agent scout note: all quiet" in prompts[4][-1]["content"]
    assert "not available" in prompts[5][-1]["content"]
    assert d.action.command == "found_city" and agent.active is not None  # skill started
    assert agent.trajectory[-1].tool_failures and "bogus" in agent.trajectory[-1].tool_failures[0]
    assert agent.delegations == 1


def test_subagent_can_return_an_action_and_cannot_use_unlisted_tools(tmp_path, snap):
    h = hx.Harness()
    hx.add_subagent(
        h,
        {"name": "builder", "description": "d", "directive": "do it", "allowed_tools": ["threats"]},
        hx.Limits(),
    )
    h.subagents[0].allowed_tools = ["threats", "run_skill", "delegate"]  # a hand-edited file
    hx.save_harness(h, tmp_path / "boot")
    action = json.dumps({"action": json.loads(legal_lines_of(snap)[0])})

    def stub(messages):
        if "You are a sub-agent" in messages[0]["content"]:
            assert "Allowed tools: ['threats']" in messages[0]["content"]
            return action
        return json.dumps({"tool": "delegate", "args": {"name": "builder"}})

    agent = ContinualAgent(ScriptedClient(stub), bootstrap_from=tmp_path / "boot")
    agent.reset(0)
    d = agent.act(snap)
    assert d.reasoning.count("[sub-agent builder]") == 1 and d.action in snap.legal_actions
    assert d.usage.calls == 2


def legal_lines_of(snap):
    return legal_lines(
        [
            {
                "content": "Legal actions\n"
                + "\n".join(
                    json.dumps(a.model_dump(mode="json"), separators=(",", ":"))
                    for a in snap.legal_actions
                )
            }
        ]
    )


def test_skill_steps_cost_no_model_call_and_update_effectiveness(tmp_path):
    h = hx.Harness()
    hx.add_skill(h, TIDY, hx.Limits(), ok_verifier)
    hx.save_harness(h, tmp_path / "boot")
    calls = {"n": 0}
    pick = PickLegal()

    def stub(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "run_skill", "args": {"name": "tidy_up"}})
        return pick(messages)

    agent = ContinualAgent(ScriptedClient(stub), bootstrap_from=tmp_path / "boot")
    m, recorded = run(agent, tmp_path, turns=6)
    free = [s for s in recorded.steps if s.usage is None]
    assert free and agent.free_steps == len(free)
    entry = agent.h.skill("tidy_up")
    assert entry.uses == len(free) + 1 and entry.failures == 0
    assert m.actions_per_turn > m.calls_per_turn


def test_cli_runs_continual_with_a_stub_server(tmp_path, capsys):
    from stubs import chat_server

    pick = Stub()
    with chat_server(pick) as (url, _):
        out = tmp_path / "c.jsonl"
        code = main(
            [
                "run", "--agent", "continual", "--turns", "6", "--base-url", url,
                "--refine-warmup", "4", "--refine-every", "4", "--no-rollback", "--out", str(out),
            ]
        )  # fmt: skip
    assert code == 0 and json.loads(capsys.readouterr().out)["final_score"] >= 0
    run_ = load_run(out)
    assert (
        run_.header.config["schedule"]["warmup"] == 4
        and not run_.header.config["rollback"]["enabled"]
    )
    assert (tmp_path / "c.harness" / "evolution_log.jsonl").exists()


def test_sweep_registers_continual_and_reports_efficiency(tmp_path, capsys):
    from stubs import chat_server

    with chat_server(Stub()) as (url, _):
        out = tmp_path / "sw"
        code = main(
            [
                "sweep", "--agents", "continual", "--scenarios", "default", "--seeds", "0",
                "--turns", "6", "--base-url", url, "--refine-warmup", "4", "--refine-every", "4",
                "--out", str(out),
            ]
        )  # fmt: skip
    assert code == 0
    assert (out / "continual" / "default-seed0.harness" / "harness.json").exists()
    header = (out / "summary.csv").read_text().splitlines()[0].split(",")
    assert "calls_per_turn" in header and "actions_per_turn" in header
    assert "calls/turn" in capsys.readouterr().out
