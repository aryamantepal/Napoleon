import pytest
from pydantic import ValidationError

from napoleon.schema import (
    SCHEMA_VERSION,
    City,
    Coord,
    EndTurn,
    MoveUnit,
    Player,
    SetProduction,
    Snapshot,
    Tile,
    Unit,
    UnitCommand,
    Yields,
    parse_action,
    snapshot_json_schema,
)


def make_snapshot() -> Snapshot:
    return Snapshot(
        turn=3,
        max_turns=100,
        player=Player(id=0, civ="Rome", gold=12.5, score=4),
        cities=(
            City(
                id=1,
                owner=0,
                name="Roma",
                coord=Coord(q=0, r=0),
                population=2,
                yields=Yields(food=3, production=2),
                production="warrior",
                is_capital=True,
            ),
        ),
        units=(
            Unit(id=7, owner=0, type="warrior", coord=Coord(q=1, r=0), moves_left=2, max_moves=2),
        ),
        tiles=(Tile(coord=Coord(q=0, r=0), terrain="grassland", owner=0),),
        legal_actions=(MoveUnit(unit_id=7, to=Coord(q=2, r=0)), EndTurn()),
        victory_progress={"science": 0.01},
    )


def test_snapshot_round_trips_through_json():
    snap = make_snapshot()
    again = Snapshot.model_validate_json(snap.model_dump_json())
    assert again == snap
    assert again.schema_version == SCHEMA_VERSION


def test_legal_actions_keep_their_types():
    again = Snapshot.model_validate_json(make_snapshot().model_dump_json())
    assert isinstance(again.legal_actions[0], MoveUnit)
    assert isinstance(again.legal_actions[1], EndTurn)


@pytest.mark.parametrize(
    "data, cls",
    [
        ({"type": "move_unit", "unit_id": 1, "to": {"q": 0, "r": 1}}, MoveUnit),
        ({"type": "unit_command", "unit_id": 1, "command": "found_city"}, UnitCommand),
        ({"type": "set_production", "city_id": 2, "item": "settler"}, SetProduction),
        ('{"type": "end_turn"}', EndTurn),
    ],
)
def test_parse_action(data, cls):
    assert isinstance(parse_action(data), cls)


def test_unknown_action_type_is_rejected():
    with pytest.raises(ValidationError):
        parse_action({"type": "launch_nukes"})


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        parse_action({"type": "end_turn", "please": True})


def test_models_are_immutable():
    snap = make_snapshot()
    with pytest.raises(ValidationError):
        snap.turn = 4


def test_json_schema_exports():
    schema = snapshot_json_schema()
    assert schema["title"] == "Snapshot"
    assert "legal_actions" in schema["properties"]
