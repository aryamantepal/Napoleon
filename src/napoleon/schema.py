"""Turn snapshot and action schema.

This is the contract between a game backend and every agent. The fake game emits it
today; the civ6-mcp structured layer will emit the same shape for real Civ VI.

Coordinates are axial hex coordinates ``(q, r)``. ``units`` holds the player's own
units and any other unit in sight, so check ``Unit.owner``. A ``MoveUnit`` onto an enemy
unit's tile is an attack. Only tiles the player has revealed appear in a snapshot, and
``Tile.visible`` says whether the player sees the tile this turn (``False`` means it is
remembered from an earlier turn, under fog).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_VERSION = "0.2"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Coord(_Model):
    q: int
    r: int


class Yields(_Model):
    food: float = 0.0
    production: float = 0.0
    gold: float = 0.0
    science: float = 0.0
    culture: float = 0.0
    faith: float = 0.0


class Tile(_Model):
    coord: Coord
    terrain: str
    feature: str | None = None
    resource: str | None = None
    river: bool = False
    owner: int | None = None
    visible: bool = True


class Unit(_Model):
    id: int
    owner: int
    type: str
    coord: Coord
    hp: int = 100
    moves_left: int
    max_moves: int
    fortified: bool = False


class City(_Model):
    id: int
    owner: int
    name: str
    coord: Coord
    population: int
    yields: Yields
    food_stored: float = 0.0
    production: str | None = None
    production_progress: float = 0.0
    production_cost: float | None = None
    is_capital: bool = False
    hp: int = 100


class ResearchState(_Model):
    """Progress on the tech tree or the civic tree."""

    completed: tuple[str, ...] = ()
    current: str | None = None
    progress: float = 0.0
    cost: float | None = None
    available: tuple[str, ...] = ()


class Contact(_Model):
    player_id: int
    civ: str
    at_war: bool = False
    score: int | None = None


class Player(_Model):
    id: int
    civ: str
    leader: str | None = None
    gold: float = 0.0
    yields: Yields = Yields()
    score: int = 0
    era: str | None = None


BlockerKind = Literal[
    "unit_needs_orders", "city_needs_production", "research_needed", "civic_needed"
]


class Blocker(_Model):
    """Something the player must resolve before the turn can end."""

    kind: BlockerKind
    ref_id: int | None = None
    message: str = ""


# Actions -----------------------------------------------------------------------------


class MoveUnit(_Model):
    type: Literal["move_unit"] = "move_unit"
    unit_id: int
    to: Coord


UnitCommandName = Literal["fortify", "skip", "found_city"]


class UnitCommand(_Model):
    type: Literal["unit_command"] = "unit_command"
    unit_id: int
    command: UnitCommandName


class SetProduction(_Model):
    type: Literal["set_production"] = "set_production"
    city_id: int
    item: str


class SetResearch(_Model):
    type: Literal["set_research"] = "set_research"
    tech: str


class SetCivic(_Model):
    type: Literal["set_civic"] = "set_civic"
    civic: str


class EndTurn(_Model):
    type: Literal["end_turn"] = "end_turn"


Action = Annotated[
    MoveUnit | UnitCommand | SetProduction | SetResearch | SetCivic | EndTurn,
    Field(discriminator="type"),
]

ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def parse_action(data: dict | str) -> Action:
    """Parse an action from a dict or a JSON string. Raises ``pydantic.ValidationError``."""
    if isinstance(data, str):
        return ACTION_ADAPTER.validate_json(data)
    return ACTION_ADAPTER.validate_python(data)


# Snapshot and results ----------------------------------------------------------------


class Snapshot(_Model):
    """Everything one player can see at one decision point."""

    schema_version: str = SCHEMA_VERSION
    turn: int
    max_turns: int
    player: Player
    cities: tuple[City, ...] = ()
    units: tuple[Unit, ...] = ()
    tech: ResearchState = ResearchState()
    civics: ResearchState = ResearchState()
    contacts: tuple[Contact, ...] = ()
    tiles: tuple[Tile, ...] = ()
    victory_progress: dict[str, float] = Field(default_factory=dict)
    blockers: tuple[Blocker, ...] = ()
    legal_actions: tuple[Action, ...] = ()
    done: bool = False


class StepResult(_Model):
    """What happened when an action was applied."""

    ok: bool
    error: str | None = None
    events: tuple[str, ...] = ()
    turn_ended: bool = False
    done: bool = False


def snapshot_json_schema() -> dict:
    """JSON Schema for ``Snapshot``, for backends written in other languages."""
    return Snapshot.model_json_schema()
