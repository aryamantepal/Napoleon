import json

import pytest

from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.hexgrid import distance
from napoleon.schema import MoveUnit
from napoleon.skills import (
    Skill,
    SkillError,
    SkillLibrary,
    SkillRun,
    bind_args,
    holds,
    parse_skill,
    tokens,
    verify_skill,
)

TIDY = {
    "name": "tidy_up",
    "description": "Clear every blocker: found cities, set production, research, civics, orders",
    "rules": [
        {"do": {"type": "unit_command", "command": "found_city"}, "unit_type": "settler"},
        {"do": {"type": "set_production", "item": "warrior"}, "blocked_only": True},
        {"do": {"type": "set_research"}, "when": {"blocker": "research_needed"}},
        {"do": {"type": "set_civic"}, "when": {"blocker": "civic_needed"}},
        {"do": {"type": "unit_command", "command": "fortify"}, "blocked_only": True},
        {"do": {"type": "unit_command", "command": "skip"}, "blocked_only": True},
    ],
}
SETTLE_AT = {
    "name": "settle_at",
    "description": "Found a city with a settler, walking to the tile first",
    "params": {"q": "int", "r": "int"},
    "rules": [
        {"do": {"type": "unit_command", "command": "found_city"}, "unit_type": "settler"},
        {
            "do": {"type": "move_unit"},
            "unit_type": "settler",
            "choose": {"nearest_to": {"q": "$q", "r": "$r"}},
        },
    ],
}


@pytest.fixture
def fresh():
    return CivEnv(FakeCivGame(max_turns=20)).reset(seed=0)


def with_(base, **changes):
    return {**base, **changes}


# Parsing and validation ---------------------------------------------------------------


def test_good_skills_parse():
    assert parse_skill(TIDY).name == "tidy_up"
    assert parse_skill(SETTLE_AT).params == {"q": "int", "r": "int"}


@pytest.mark.parametrize(
    "bad, message",
    [
        ("text", "JSON object"),
        (with_(TIDY, name="Tidy Up"), "snake_case"),
        (with_(TIDY, description="short"), "description"),
        (with_(TIDY, rules=[]), "at least one rule"),
        (with_(TIDY, rules=[{"do": {"type": "teleport"}}]), "rule 0"),
        (with_(TIDY, rules=[{"do": {"type": "end_turn", "bogus": 1}}]), "no field 'bogus'"),
        (
            with_(TIDY, rules=[{"do": {"type": "end_turn"}, "when": {"flying": 1}}]),
            "unknown condition",
        ),
        (with_(TIDY, rules=[{"do": {"type": "end_turn"}, "when": {"all": []}}]), "non-empty"),
        (with_(TIDY, rules=[{"do": {"type": "end_turn"}, "choose": {"random": 1}}]), "choose is"),
        (with_(TIDY, rules=[{"do": {"type": "set_research", "tech": "$t"}}]), "not declared"),
        (with_(TIDY, rules=[{"do": {"type": "end_turn"}}] * 13), "at most 12"),
        (with_(TIDY, max_steps=500), "max_steps"),
        (with_(TIDY, until={"a": 1, "b": 2}), "one key"),
        (with_(TIDY, extra="x"), "invalid skill"),
    ],
)
def test_bad_skills_say_why(bad, message):
    with pytest.raises(SkillError, match=message):
        parse_skill(bad)


def test_deeply_nested_conditions_are_rejected():
    cond = {"no_blockers": True}
    for _ in range(6):
        cond = {"not": cond}
    with pytest.raises(SkillError, match="nest"):
        parse_skill(with_(TIDY, until=cond))


def test_bind_args_checks_and_coerces():
    skill = parse_skill(SETTLE_AT)
    assert bind_args(skill, {"q": "3", "r": 4}) == {"q": 3, "r": 4}
    for args, message in [
        ({"q": 1}, "missing argument 'r'"),
        ({"q": 1, "r": 2, "z": 3}, "unknown arguments"),
        ({"q": "x", "r": 1}, "must be int"),
        ({"q": 1.5, "r": 1}, "must be int"),
        ({"q": True, "r": 1}, "must be int"),
    ]:
        with pytest.raises(SkillError, match=message):
            bind_args(skill, args)


# Running ------------------------------------------------------------------------------


def play_skill(skill, args, env):
    """Run a skill to completion against a live game. Returns the actions it issued."""
    run = SkillRun(skill, args)
    snap, issued = env.observe(), []
    while (action := run.next_action(snap)) is not None:
        snap, result = env.step(action)
        assert result.ok, result.error
        issued.append(action)
    return issued, snap


def test_tidy_up_clears_every_blocker_with_legal_actions():
    env = CivEnv(FakeCivGame(max_turns=20))
    env.reset(seed=0)
    issued, snap = play_skill(parse_skill(TIDY), {}, env)
    assert issued and issued[0].command == "found_city"
    assert not [b for b in snap.blockers if b.kind != "other"]


def test_skill_never_repeats_an_identical_action():
    env = CivEnv(FakeCivGame(max_turns=20))
    env.reset(seed=0)
    skip = parse_skill(
        with_(
            TIDY,
            rules=[{"do": {"type": "unit_command", "command": "skip"}}],
            max_steps=40,
        )
    )
    issued, _ = play_skill(skip, {}, env)
    keys = [a.model_dump_json() for a in issued]
    assert keys and len(keys) == len(set(keys))


def test_nearest_to_walks_toward_the_target(fresh):
    settler = next(u for u in fresh.units if u.type == "settler")
    skill = parse_skill(
        with_(
            SETTLE_AT,
            rules=[
                {
                    "do": {"type": "move_unit"},
                    "unit_type": "settler",
                    "choose": {"nearest_to": {"q": "$q", "r": "$r"}},
                }
            ],
        )
    )
    action = SkillRun(skill, {"q": 5, "r": -2}).next_action(fresh)
    assert isinstance(action, MoveUnit) and action.unit_id == settler.id
    from napoleon.schema import Coord

    goal = Coord(q=5, r=-2)
    best = min(
        distance(a.to, goal)
        for a in fresh.legal_actions
        if isinstance(a, MoveUnit) and a.unit_id == settler.id
    )
    assert distance(action.to, goal) == best


def test_other_choosers_pick_moves(fresh):
    for chooser in ({"farthest_from_cities": True}, {"nearest_enemy": True}):
        skill = parse_skill(with_(TIDY, rules=[{"do": {"type": "move_unit"}, "choose": chooser}]))
        assert isinstance(SkillRun(skill, {}).next_action(fresh), MoveUnit)


def test_limits_and_until(fresh):
    skill = parse_skill(with_(TIDY, max_steps=1))
    run = SkillRun(skill, {})
    assert run.next_action(fresh) is not None and run.next_action(fresh) is None and run.done
    until = parse_skill(with_(TIDY, until={"turn_at_least": 1}))
    assert SkillRun(until, {}).next_action(fresh) is None
    later = Skill.model_validate(with_(TIDY, max_turns=1))
    run = SkillRun(later, {})
    run.next_action(fresh)
    assert run.next_action(fresh.model_copy(update={"turn": 2})) is None


def test_conditions(fresh):
    cases = {
        '{"has_unit": "settler"}': True,
        '{"has_unit": "warrior"}': True,
        '{"cities_below": 1}': True,
        '{"cities_at_least": 1}': False,
        '{"turn_at_least": 2}': False,
        '{"turn_below": 2}': True,
        '{"enemy_within": 5}': False,
        '{"blocker": "unit_needs_orders"}': True,
        '{"no_blockers": true}': False,
        '{"legal": {"type": "unit_command", "command": "found_city"}}': True,
        '{"not": {"has_unit": "settler"}}': False,
        '{"all": [{"has_unit": "settler"}, {"turn_below": 2}]}': True,
        '{"any": [{"cities_at_least": 9}, {"turn_below": 2}]}': True,
    }
    for text, expected in cases.items():
        assert holds(json.loads(text), fresh) is expected, text


def test_when_gates_a_rule(fresh):
    skill = parse_skill(
        with_(TIDY, rules=[{"do": {"type": "end_turn"}, "when": {"turn_at_least": 5}}])
    )
    assert SkillRun(skill, {}).next_action(fresh) is None


# Library ------------------------------------------------------------------------------


def test_library_persists_and_counts_uses(tmp_path):
    path = tmp_path / "skills" / "lib.json"
    lib = SkillLibrary(path)
    lib.add(parse_skill(TIDY))
    lib.record_use("tidy_up")
    again = SkillLibrary(path)
    assert list(again.skills) == ["tidy_up"] and again.get("tidy_up").uses == 1
    assert "tidy_up" in again and len(again) == 1
    again.clear()
    assert len(SkillLibrary(path)) == 0 and not path.exists()


def test_library_rejects_duplicates_and_overflow(tmp_path):
    lib = SkillLibrary(tmp_path / "l.json", max_skills=1)
    lib.add(parse_skill(TIDY))
    with pytest.raises(SkillError, match="already exists"):
        lib.add(parse_skill(TIDY))
    with pytest.raises(SkillError, match="full"):
        lib.add(parse_skill(SETTLE_AT))


def test_retrieval_ranks_by_description():
    lib = SkillLibrary()
    lib.add(parse_skill(TIDY))
    lib.add(parse_skill(SETTLE_AT))
    defend = with_(
        TIDY,
        name="hunt_barbarians",
        description="Attack barbarian enemy units that come close to our cities",
    )
    lib.add(parse_skill(defend))
    assert lib.retrieve("settler found city")[0].name == "settle_at"
    assert lib.retrieve("enemy barbarian near city")[0].name == "hunt_barbarians"
    assert lib.retrieve("blockers need orders")[0].name == "tidy_up"
    assert lib.retrieve("zzzz qqqq") == []
    assert len(lib.retrieve("city", k=2)) <= 2
    assert SkillLibrary().retrieve("anything") == []
    assert tokens("found_city, the Settler!") == ["found", "city", "settler"]


# Verification -------------------------------------------------------------------------


def test_good_skills_verify_in_fake_game_rollouts():
    ok = verify_skill(parse_skill(TIDY))
    assert ok.ok and ok.fired > 0 and set(ok.scores) == {"default", "bad_opening"}
    assert verify_skill(parse_skill(SETTLE_AT), {"q": 2, "r": 0}).ok


def test_verification_failures():
    never = parse_skill(
        with_(TIDY, rules=[{"do": {"type": "end_turn"}, "when": {"turn_at_least": 999}}])
    )
    assert "never applied" in verify_skill(never).reason
    stall = parse_skill(
        with_(
            TIDY,
            rules=[{"do": {"type": "unit_command", "command": "skip"}, "unit_type": "settler"}],
        )
    )
    assert "stalls" in verify_skill(stall).reason
    assert "below" in verify_skill(parse_skill(TIDY), min_ratio=50.0).reason
    assert "missing argument" in verify_skill(parse_skill(SETTLE_AT), {}).reason
