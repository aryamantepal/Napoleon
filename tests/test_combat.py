"""Barbarians, rivals and fighting in the fake game."""

from napoleon.agents import GreedyAgent
from napoleon.env import CivEnv
from napoleon.fake import FakeCivGame
from napoleon.fake.game import IMPASSABLE, MAX_BARBARIANS
from napoleon.hexgrid import neighbors
from napoleon.schema import MoveUnit, UnitCommand


def setup(seed: int = 0) -> tuple[CivEnv, FakeCivGame]:
    game = FakeCivGame()
    env = CivEnv(game)
    env.reset(seed=seed)
    return env, game


def own(game: FakeCivGame, kind: str):
    return next(u for u in game.units.values() if u.owner == game.PLAYER and u.type == kind)


def free_neighbor(game: FakeCivGame, coord):
    taken = {u.coord for u in game.units.values()} | {c.coord for c in game.cities.values()}
    return next(
        n
        for n in neighbors(coord)
        if n in game.terrain and game.terrain[n] not in IMPASSABLE and n not in taken
    )


def test_attack_on_adjacent_enemy_is_legal_and_hurts():
    env, game = setup()
    warrior = own(game, "warrior")
    spot = free_neighbor(game, warrior.coord)
    barb = game._new_unit("warrior", spot, owner=game.BARBARIANS)
    game._update_sight()
    attack = MoveUnit(unit_id=warrior.id, to=spot)
    assert attack in env.observe().legal_actions
    _, result = env.step(attack)
    assert result.ok
    assert any("attacked" in e or "destroyed" in e for e in result.events)
    assert barb.id not in game.units or game.units[barb.id].hp < 100


def test_barbarian_captures_a_lone_settler():
    _, game = setup()
    settler = own(game, "settler")
    spot = free_neighbor(game, settler.coord)
    game._new_unit("warrior", spot, owner=game.BARBARIANS)
    events = game._barbarian_phase()
    assert any("captured" in e for e in events)
    assert settler.id not in game.units


def test_a_city_shelters_units_inside_it():
    env, game = setup()
    settler = own(game, "settler")
    env.step(UnitCommand(unit_id=settler.id, command="found_city"))
    city = next(iter(c for c in game.cities.values() if c.owner == game.PLAYER))
    guest = game._new_unit("settler", city.coord)
    spot = free_neighbor(game, city.coord)
    game._new_unit("warrior", spot, owner=game.BARBARIANS)
    for other in [u for u in game.units.values() if u.owner == game.PLAYER and u is not guest]:
        other.coord = city.coord  # keep everyone inside the walls
    events = game._barbarian_phase()
    assert guest.id in game.units
    assert any(city.name in e for e in events)


def test_barbarians_show_up_and_are_capped():
    env, game = setup()
    agent = GreedyAgent()
    seen = []
    snap = env.observe()
    while not snap.done and game.turn <= 60:
        snap, result = env.step(agent.act(snap).action)
        seen += [e for e in result.events if "barbarians appeared" in e]
        barbs = [u for u in game.units.values() if u.owner == game.BARBARIANS]
        assert len(barbs) <= MAX_BARBARIANS
    assert seen


def test_rival_expands_on_schedule():
    _, game = setup()
    before = sum(c.owner == game.RIVAL for c in game.cities.values())
    game.turn = 20
    game._rival_phase()
    after = sum(c.owner == game.RIVAL for c in game.cities.values())
    assert after == before + 1


def test_city_asks_for_orders_after_every_unit():
    env, game = setup()
    settler = own(game, "settler")
    env.step(UnitCommand(unit_id=settler.id, command="found_city"))
    city = next(c for c in game.cities.values() if c.owner == game.PLAYER)
    city.production = "warrior"
    game._complete(city)
    assert city.production is None
    kinds = {b.kind for b in env.observe().blockers}
    assert "city_needs_production" in kinds
