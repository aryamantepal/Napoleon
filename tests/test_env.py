import pytest

from napoleon.backend import GameBackend
from napoleon.env import CivEnv, EnvNotReset
from napoleon.fake import FakeCivGame
from napoleon.schema import EndTurn, MoveUnit, SetResearch, UnitCommand


def make_env(**kw) -> CivEnv:
    return CivEnv(FakeCivGame(**kw))


def settler(snap):
    return next(u for u in snap.units if u.type == "settler")


def test_fake_game_satisfies_protocol():
    assert isinstance(FakeCivGame(), GameBackend)


def test_observe_before_reset_raises():
    with pytest.raises(EnvNotReset):
        make_env().observe()


def test_reset_is_deterministic_per_seed():
    a = make_env().reset(seed=3)
    b = make_env().reset(seed=3)
    c = make_env().reset(seed=4)
    assert a == b
    assert a.tiles != c.tiles


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError):
        make_env().reset(scenario="nope")


def test_start_has_settler_warrior_and_blockers():
    snap = make_env().reset()
    assert {u.type for u in snap.units} == {"settler", "warrior"}
    assert snap.blockers
    assert EndTurn() not in snap.legal_actions


def test_end_turn_while_blocked_is_refused_without_change():
    env = make_env()
    before = env.reset()
    after, result = env.step(EndTurn())
    assert not result.ok
    assert "blocked" in result.error
    assert after == before


def test_malformed_action_returns_error():
    env = make_env()
    env.reset()
    _, result = env.step({"type": "move_unit", "unit_id": "x"})
    assert not result.ok
    assert result.error.startswith("malformed action")


def test_illegal_move_returns_error():
    env = make_env()
    snap = env.reset()
    s = settler(snap)
    far = MoveUnit(unit_id=s.id, to=s.coord.model_copy(update={"q": s.coord.q + 5}))
    _, result = env.step(far)
    assert not result.ok
    assert result.error.startswith("illegal action")


def test_dict_and_json_actions_are_accepted():
    env = make_env()
    snap = env.reset()
    s = settler(snap)
    _, result = env.step({"type": "unit_command", "unit_id": s.id, "command": "found_city"})
    assert result.ok
    snap = env.observe()
    _, result = env.step(
        f'{{"type": "unit_command", "unit_id": {snap.units[0].id}, "command": "skip"}}'
    )
    assert result.ok


def play_opening(env: CivEnv):
    snap = env.reset()
    env.step(UnitCommand(unit_id=settler(snap).id, command="found_city"))
    snap = env.observe()
    city = snap.cities[0]
    env.step({"type": "set_production", "city_id": city.id, "item": "warrior"})
    env.step(SetResearch(tech="pottery"))
    env.step({"type": "set_civic", "civic": "code_of_laws"})
    for u in env.observe().units:
        env.step(UnitCommand(unit_id=u.id, command="fortify"))
    return env.observe()


def test_founding_a_city_then_clearing_blockers_allows_end_turn():
    env = make_env()
    snap = play_opening(env)
    assert len(snap.cities) == 1
    assert snap.blockers == ()
    snap, result = env.step(EndTurn())
    assert result.ok and result.turn_ended
    assert snap.turn == 2


def test_fog_hides_unseen_tiles():
    snap = make_env(radius=7).reset()
    total = 3 * 7 * 8 + 1
    assert 0 < len(snap.tiles) < total
    assert all(t.visible for t in snap.tiles)


def test_moving_reveals_tiles_and_keeps_memory():
    env = make_env()
    snap = env.reset()
    w = next(u for u in snap.units if u.type == "warrior")
    seen = {t.coord for t in snap.tiles}
    for _ in range(3):
        moves = [
            a for a in env.observe().legal_actions if isinstance(a, MoveUnit) and a.unit_id == w.id
        ]
        if not moves:
            break
        env.step(moves[0])
    after = env.observe()
    assert seen <= {t.coord for t in after.tiles}


def test_game_ends_at_turn_limit_and_refuses_more_actions():
    env = make_env(max_turns=5)
    play_opening(env)
    snap = env.observe()
    while not snap.done:
        for b in snap.blockers:
            if b.kind == "unit_needs_orders":
                env.step(UnitCommand(unit_id=b.ref_id, command="skip"))
            elif b.kind == "city_needs_production":
                env.step({"type": "set_production", "city_id": b.ref_id, "item": "monument"})
            elif b.kind == "research_needed":
                env.step(SetResearch(tech=env.observe().tech.available[0]))
            elif b.kind == "civic_needed":
                env.step({"type": "set_civic", "civic": env.observe().civics.available[0]})
        snap, _ = env.step(EndTurn())
    assert snap.turn == 6
    assert snap.legal_actions == ()
    _, result = env.step(EndTurn())
    assert not result.ok and result.done


def test_bad_opening_starts_with_only_a_settler_and_no_gold():
    snap = make_env().reset(scenario="bad_opening")
    assert [u.type for u in snap.units] == ["settler"]
    assert snap.player.gold == 0
