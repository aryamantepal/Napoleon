import json

import pytest
from stubs import PickLegal, chat_server
from test_skills import SETTLE_AT, TIDY

from napoleon.agents import AGENTS, LLM_AGENTS, VoyagerAgent
from napoleon.agents.factory import make_agent
from napoleon.agents.voyager import retrieval_query
from napoleon.cli import main
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.llm import ScriptedClient
from napoleon.recording import load_run
from napoleon.runner import run_episodes, run_game
from napoleon.skills import SkillLibrary, Verification, parse_skill


def new_skill(skill=TIDY, args=None, thought="a routine I will need again"):
    return json.dumps({"thought": thought, "new_skill": skill, "args": args or {}})


def use_skill(name, args=None):
    return json.dumps({"thought": "reuse", "use_skill": name, "args": args or {}})


def always_ok(skill, args):
    return Verification(True, "ok", 3, {"default": 9}, {"default": 9})


@pytest.fixture
def snap():
    return CivEnv(FakeCivGame(max_turns=20)).reset(seed=0)


def agent_for(client, tmp_path, **kw):
    kw.setdefault("verifier", always_ok)
    return VoyagerAgent(client, skills_path=tmp_path / "lib.json", **kw)


def test_registered_and_built_by_the_factory(tmp_path):
    assert AGENTS["voyager"] is VoyagerAgent and "voyager" in LLM_AGENTS
    made = make_agent("voyager", None, skills_path=tmp_path / "s.json", verify=False)
    assert made.verify is False and made.library.path == tmp_path / "s.json"


def test_new_skill_is_verified_stored_and_then_runs_without_the_model(tmp_path):
    env = CivEnv(FakeCivGame(max_turns=20))
    s = env.reset(seed=0)
    client = ScriptedClient([new_skill(), '{"action": {"type": "end_turn"}}'])
    agent = agent_for(client, tmp_path)
    agent.reset(0)
    first = agent.act(s)
    assert first.action.command == "found_city" and first.usage.calls == 1
    assert first.reasoning.startswith("[new skill tidy_up, tested ok]")
    assert "tidy_up" in SkillLibrary(tmp_path / "lib.json")
    # the skill keeps playing with no further model calls
    decisions = []
    s, _ = env.step(first.action)
    while (d := agent.act(s)).usage is None:
        decisions.append(d)
        s, result = env.step(d.action)
        assert result.ok
    assert len(decisions) >= 3 and all("no model call" in d.reasoning for d in decisions)
    assert len(client.calls) == 2  # the first decision, then the one after the skill finished
    assert agent.created == 1 and agent.free_steps == len(decisions)
    assert agent.library.get("tidy_up").uses == 1
    assert agent.library.get("tidy_up").verification["fired"] == 3


def test_failed_testing_is_fed_back_and_nothing_is_stored(tmp_path, snap):
    bad = lambda skill, args: Verification(False, "the skill never applied")  # noqa: E731
    client = ScriptedClient([new_skill(), '{"action": {"type": "end_turn"}}'])
    agent = agent_for(client, tmp_path, verifier=bad)
    d = agent.act(snap)
    assert "failed testing: the skill never applied" in client.calls[1][-1]["content"]
    assert d.action.type == "end_turn"
    assert len(agent.library) == 0 and d.reasoning.startswith("[after 1 retries]")


def test_verify_false_skips_the_verifier(tmp_path, snap):
    boom = lambda skill, args: (_ for _ in ()).throw(AssertionError("should not run"))  # noqa: E731
    agent = agent_for(ScriptedClient([new_skill()]), tmp_path, verify=False, verifier=boom)
    assert agent.act(snap).action.command == "found_city"
    assert agent.library.get("tidy_up").verification is None


def test_real_verification_end_to_end(tmp_path, snap):
    agent = VoyagerAgent(ScriptedClient([new_skill()]), skills_path=tmp_path / "l.json")
    d = agent.act(snap)
    assert d.action.command == "found_city"
    v = agent.library.get("tidy_up").verification
    assert v["reason"] == "ok" and v["fired"] > 0


@pytest.mark.parametrize(
    "reply, message",
    [
        (new_skill({**TIDY, "name": "Bad Name"}), "snake_case"),
        (new_skill(SETTLE_AT), "missing argument"),
        (use_skill("nope"), "no skill named"),
        (json.dumps({"new_skill": "x"}), "JSON object"),
        ("no json", "no JSON"),
    ],
)
def test_bad_skill_replies_are_retried_with_the_reason(tmp_path, snap, reply, message):
    client = ScriptedClient([reply, '{"action": {"type": "end_turn"}}'])
    agent_for(client, tmp_path).act(snap)
    assert message in client.calls[1][-1]["content"]


def test_duplicate_names_are_refused(tmp_path, snap):
    lib = SkillLibrary(tmp_path / "lib.json")
    lib.add(parse_skill(TIDY))
    client = ScriptedClient([new_skill(), use_skill("tidy_up")])
    agent_for(client, tmp_path).act(snap)
    assert "already exists" in client.calls[1][-1]["content"]


def test_unusable_replies_exhaust_the_budget_into_a_raw_action(tmp_path, snap):
    d = agent_for(ScriptedClient(["???"]), tmp_path, max_retries=1).act(snap)
    assert d.action == "???" and "unusable output after 2 attempts" in d.reasoning


def test_primitive_actions_still_work(tmp_path, snap):
    client = ScriptedClient(['{"thought": "go", "action": {"type": "end_turn"}}'])
    d = agent_for(client, tmp_path).act(snap)
    assert d.action.type == "end_turn" and d.reasoning == "go"


def test_existing_skills_are_retrieved_into_the_prompt_and_reused(tmp_path, snap):
    lib = SkillLibrary(tmp_path / "lib.json")
    lib.add(parse_skill(TIDY))
    lib.add(parse_skill(SETTLE_AT))
    client = ScriptedClient([use_skill("settle_at", {"q": 2, "r": 0})])
    agent = agent_for(client, tmp_path)
    agent.reset(0)
    assert agent.loaded == 2 and agent.config()["skills_loaded"] == 2
    d = agent.act(snap)
    prompt = client.calls[0][1]["content"]
    assert "Relevant skills in your library" in prompt and '"name":"settle_at"' in prompt
    assert d.reasoning.startswith("[skill settle_at]")
    assert agent.reused == 1 and agent.library.get("settle_at").uses == 1
    assert d.action.type in ("unit_command", "move_unit")


def test_inapplicable_skill_is_bounced_back(tmp_path, snap):
    lib = SkillLibrary(tmp_path / "lib.json")
    lib.add(
        parse_skill(
            {**TIDY, "rules": [{"do": {"type": "end_turn"}, "when": {"turn_at_least": 99}}]}
        )
    )
    client = ScriptedClient([use_skill("tidy_up"), '{"action": {"type": "end_turn"}}'])
    agent_for(client, tmp_path).act(snap)
    assert "nothing to do right now" in client.calls[1][-1]["content"]


def test_empty_library_is_stated_and_retrieval_query_reflects_the_situation(tmp_path, snap):
    client = ScriptedClient(['{"action": {"type": "end_turn"}}'])
    agent_for(client, tmp_path).act(snap)
    assert "Your skill library is empty" in client.calls[0][1]["content"]
    query = retrieval_query(snap)
    assert "settler" in query and "unit_needs_orders" in query and "found city" in query


def voyager_model():
    """Defines tidy_up once, calls it whenever blockers need clearing, ends turns otherwise."""
    pick = PickLegal()

    def reply(messages):
        text = messages[1]["content"]
        if "Your skill library is empty" in text:
            return new_skill()
        blocked = "unit_needs_orders" in text or "city_needs_production" in text
        if blocked and "tidy_up" in text:
            return use_skill("tidy_up")
        return pick(messages)

    return reply


def test_full_game_skills_cut_model_calls_and_show_up_in_the_record(tmp_path):
    client = ScriptedClient(voyager_model(), model="stub-voyager")
    agent = agent_for(client, tmp_path)
    m = run_game(CivEnv(FakeCivGame(max_turns=25)), agent, tmp_path / "v.jsonl", backend="fake")
    run = load_run(tmp_path / "v.jsonl")
    assert run.footer.reason == "done" and m.failed_steps == 0
    free = [s for s in run.steps if s.usage is None]
    assert free and all("no model call" in s.reasoning for s in free)
    assert m.llm_calls == len(client.calls) < m.steps
    assert any("[new skill tidy_up" in (s.reasoning or "") for s in run.steps)
    assert run.header.config["skills_path"].endswith("lib.json")
    assert agent.created == 1 and agent.reused >= 1


def test_library_carries_across_episodes(tmp_path):
    client = ScriptedClient(voyager_model())
    agent = agent_for(client, tmp_path)
    run_episodes(
        CivEnv(FakeCivGame(max_turns=4)),
        agent,
        tmp_path,
        backend="fake",
        stem="v",
        seeds=[0, 0],
    )
    h1, h2 = (load_run(tmp_path / f"v-ep{n}.jsonl").header for n in (1, 2))
    assert (h1.config["skills_loaded"], h2.config["skills_loaded"]) == (0, 1)
    second = load_run(tmp_path / "v-ep2.jsonl")
    assert not any("[new skill" in (s.reasoning or "") for s in second.steps)


def test_cli_voyager_against_a_server(tmp_path, capsys):
    lib = tmp_path / "lib.json"
    out = tmp_path / "v.jsonl"
    with chat_server(voyager_model()) as (url, seen):
        code = main(
            ["run", "--agent", "voyager", "--turns", "4", "--out", str(out), "--base-url", url]
            + ["--skills", str(lib), "--clear-skills", "--no-verify"]
        )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["failed_steps"] == 0 and summary["llm_calls"] == len(seen) < summary["steps"]
    assert "tidy_up" in SkillLibrary(lib)
