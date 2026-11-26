"""The civ6-mcp backend against a scripted stand-in for the game."""

import pytest

from napoleon.agents import GreedyAgent, RandomAgent
from napoleon.civ6 import Civ6Backend
from napoleon.civ6.coords import axial_to_offset, offset_to_axial
from napoleon.civ6.names import short
from napoleon.civ6.scripted import ScriptedSource
from napoleon.env import CivEnv
from napoleon.hexgrid import distance
from napoleon.recording import load_run
from napoleon.runner import run_game
from napoleon.schema import (
    MoveUnit,
    SetCivic,
    SetProduction,
    SetResearch,
    UnitCommand,
)

SAVES = {"default": "napoleon_default"}


def make() -> tuple[CivEnv, ScriptedSource]:
    source = ScriptedSource()
    return CivEnv(Civ6Backend(source, SAVES, max_turns=12)), source


def own(snap, kind):
    return next(u for u in snap.units if u.owner == snap.player.id and u.type == kind)


@pytest.mark.parametrize("x,y", [(0, 0), (5, 3), (10, 10), (7, 8), (23, 19)])
def test_coordinates_round_trip(x, y):
    assert axial_to_offset(offset_to_axial(x, y)) == (x, y)


def test_offset_neighbors_are_one_step_apart():
    # odd-r: on an odd row, (x+1, y±1) are neighbors; on an even row, (x-1, y±1) are.
    assert distance(offset_to_axial(4, 3), offset_to_axial(5, 4)) == 1
    assert distance(offset_to_axial(4, 4), offset_to_axial(3, 5)) == 1
    assert distance(offset_to_axial(4, 4), offset_to_axial(5, 5)) == 2


def test_short_names():
    assert short("UNIT_WARRIOR") == "warrior"
    assert short("TECHNOLOGY_BRONZE_WORKING") == "bronze_working"
    assert short("TERRAIN_GRASS_HILLS") == "grass_hills"
    assert short("NONE") is None


def test_unknown_scenario_and_bad_save():
    backend = Civ6Backend(ScriptedSource(), {"default": "missing"})
    with pytest.raises(ValueError):
        backend.reset("bad_opening", 0)
    with pytest.raises(RuntimeError):
        backend.reset("default", 0)


def test_first_snapshot():
    env, _ = make()
    snap = env.reset()
    assert snap.turn == 1
    assert snap.player.civ == "France"
    assert {u.type for u in snap.units if u.owner == 0} == {"settler", "warrior"}
    barb = [u for u in snap.units if u.owner == 63]
    assert len(barb) == 1 and barb[0].id < 0
    assert {b.kind for b in snap.blockers} == {"unit_needs_orders"}
    assert snap.tech.available == ("pottery", "mining")
    assert all(
        t.terrain in ("grass", "plains", "plains_hills", "grass_mountain", "ocean")
        for t in snap.tiles
    )
    assert UnitCommand(unit_id=own(snap, "settler").id, command="found_city") in snap.legal_actions


def test_mountains_are_not_legal_moves():
    env, _ = make()
    snap = env.reset()
    mountains = {t.coord for t in snap.tiles if "mountain" in t.terrain}
    moves = {a.to for a in snap.legal_actions if isinstance(a, MoveUnit)}
    assert not mountains & moves


def test_actions_map_onto_civ6_mcp_calls():
    env, source = make()
    snap = env.reset()
    settler, warrior = own(snap, "settler"), own(snap, "warrior")

    _, r = env.step(UnitCommand(unit_id=settler.id, command="found_city"))
    assert r.ok and source.calls[-1] == ("found_city", 1)
    snap = env.observe()
    city = snap.cities[0]
    assert city.is_capital and city.name == "Paris"
    kinds = {b.kind for b in snap.blockers}
    assert {"city_needs_production", "research_needed", "civic_needed"} <= kinds

    # Districts need a tile, so they are not offered.
    items = {a.item for a in snap.legal_actions if isinstance(a, SetProduction)}
    assert items == {"warrior", "settler", "monument"}
    env.step(SetProduction(city_id=city.id, item="monument"))
    assert source.calls[-1] == ("set_city_production", city.id, "BUILDING", "BUILDING_MONUMENT")
    env.step(SetResearch(tech="pottery"))
    assert source.calls[-1] == ("set_research", "TECHNOLOGY_POTTERY")
    env.step(SetCivic(civic="code_of_laws"))
    assert source.calls[-1] == ("set_civic", "CIVIC_CODE_OF_LAWS")
    snap = env.observe()
    assert snap.tech.current == "pottery"

    # A move onto the barbarian's tile is an attack. The warrior starts at plot (11, 10)
    # and the barbarian waits at (12, 9), two steps away through (11, 9).
    barb = next(u for u in snap.units if u.owner == 63)
    assert axial_to_offset(barb.coord) == (12, 9)
    _, r = env.step(MoveUnit(unit_id=warrior.id, to=offset_to_axial(11, 9)))
    assert r.ok and source.calls[-1] == ("move_unit", 2, 11, 9)
    attack = MoveUnit(unit_id=warrior.id, to=barb.coord)
    assert attack in env.observe().legal_actions
    _, r = env.step(attack)
    assert r.ok and source.calls[-1] == ("attack_unit", 2, 12, 9)
    assert not [u for u in env.observe().units if u.owner == 63]


def test_game_refusal_becomes_a_failed_step():
    env, source = make()
    snap = env.reset()
    warrior = own(snap, "warrior")
    env.step(UnitCommand(unit_id=warrior.id, command="found_city"))  # not legal for a warrior
    assert all(c[0] != "found_city" for c in source.calls)
    source.units[warrior.id].moves_remaining = 0.0  # the game changed under us
    move = next(
        a for a in snap.legal_actions if isinstance(a, MoveUnit) and a.unit_id == warrior.id
    )
    _, r = env.step(move)
    assert not r.ok and "NO_MOVES" in r.error


def test_other_blockers_do_not_stop_end_turn():
    env, _ = make()
    snap = env.reset()
    agent = GreedyAgent()
    saw_other = False
    while not snap.done and snap.turn < 5:
        saw_other |= any(b.kind == "other" for b in snap.blockers)
        snap, _ = env.step(agent.act(snap).action)
    assert saw_other
    assert snap.turn == 5


def test_runs_record_through_the_civ6_backend(tmp_path):
    env, _ = make()
    m = run_game(env, RandomAgent(), tmp_path / "c.jsonl", backend="civ6-scripted")
    run = load_run(tmp_path / "c.jsonl")
    assert run.footer is not None and run.footer.reason == "done"
    assert m.turns_played >= 10
