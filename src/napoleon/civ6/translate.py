"""civ6-mcp state to a Napoleon ``Snapshot``.

``RawState`` holds what one round of reads returned. ``translate`` turns it into a
snapshot and a ``Lookup`` that maps the snapshot's short names and ids back onto what
civ6-mcp's action methods take.

Known gaps, to close against the live game:

- civ6-mcp reports no fortified flag, so ``Unit.fortified`` is always ``False``.
- Foreign units come from tile reads, which give a label such as ``Barbarian WARRIOR``
  but no id or owner. They get a negative id from their plot, and owner 63 (Civ VI's
  barbarian player) for barbarians or -1 otherwise.
- Districts need a placement tile, so they are left out of ``SetProduction``.
- Completed techs and civics are counted by civ6-mcp but not listed, so
  ``ResearchState.completed`` is empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from napoleon.civ6.coords import axial_to_offset, offset_to_axial
from napoleon.civ6.names import short
from napoleon.hexgrid import neighbors
from napoleon.schema import (
    Action,
    Blocker,
    City,
    Contact,
    Coord,
    EndTurn,
    MoveUnit,
    Player,
    ResearchState,
    SetCivic,
    SetProduction,
    SetResearch,
    Snapshot,
    Tile,
    Unit,
    UnitCommand,
    Yields,
)

BARBARIAN_PLAYER = 63
UNKNOWN_OWNER = -1
DEFAULT_MAX_TURNS = 500
IMPASSABLE_FOR_LAND = ("ocean", "coast")
PRODUCIBLE = ("UNIT", "BUILDING")


@dataclass
class RawState:
    overview: Any
    units: list[Any]
    cities: list[Any]
    tech: Any
    diplomacy: list[Any] = field(default_factory=list)
    tiles: list[Any] = field(default_factory=list)
    production: dict[int, list[Any]] = field(default_factory=dict)
    blockers: list[tuple[str, str]] = field(default_factory=list)
    ready_units: list[int] = field(default_factory=list)
    game_over: Any | None = None


@dataclass
class Lookup:
    """Maps snapshot names back to civ6-mcp arguments."""

    production: dict[tuple[int, str], tuple[str, str]] = field(default_factory=dict)
    techs: dict[str, str] = field(default_factory=dict)
    civics: dict[str, str] = field(default_factory=dict)
    foreign_units: set[Coord] = field(default_factory=set)


def coord(obj: Any) -> Coord:
    return offset_to_axial(obj.x, obj.y)


def foreign_unit_id(c: Coord) -> int:
    x, y = axial_to_offset(c)
    return -(1 + y * 10_000 + x)


def _owner(owner_id: int) -> int | None:
    return None if owner_id is None or owner_id < 0 else owner_id


def _passable(tile: Tile) -> bool:
    return tile.terrain not in IMPASSABLE_FOR_LAND and "mountain" not in tile.terrain


def translate(raw: RawState, *, max_turns: int | None = None) -> tuple[Snapshot, Lookup]:
    ov = raw.overview
    look = Lookup()
    me = ov.player_id

    tiles: dict[Coord, Tile] = {}
    foreign: list[Unit] = []
    for t in raw.tiles:
        if getattr(t, "visibility", "visible") == "unexplored":
            continue
        c = coord(t)
        if c in tiles:
            continue
        visible = getattr(t, "visibility", "visible") == "visible"
        terrain = short(t.terrain) or "unknown"
        if getattr(t, "is_hills", False) and "hills" not in terrain:
            terrain += "_hills"
        tiles[c] = Tile(
            coord=c,
            terrain=terrain,
            feature=short(t.feature),
            resource=short(t.resource),
            river=bool(getattr(t, "is_river", False)),
            owner=_owner(t.owner_id),
            visible=visible,
        )
        for label in (getattr(t, "units", None) or []) if visible else []:
            kind = label.split()[-1]
            barbarian = label.lower().startswith("barbarian")
            foreign.append(
                Unit(
                    id=foreign_unit_id(c),
                    owner=BARBARIAN_PLAYER if barbarian else UNKNOWN_OWNER,
                    type=short(kind if kind.startswith("UNIT_") else "UNIT_" + kind) or kind,
                    coord=c,
                    moves_left=0,
                    max_moves=0,
                )
            )
            look.foreign_units.add(c)

    units = tuple(
        Unit(
            id=u.unit_id,
            owner=me,
            type=short(u.unit_type) or u.unit_type,
            coord=coord(u),
            hp=round(100 * u.health / u.max_health) if u.max_health else 100,
            moves_left=int(u.moves_remaining),
            max_moves=int(u.max_moves),
        )
        for u in sorted(raw.units, key=lambda u: u.unit_id)
    ) + tuple(sorted(foreign, key=lambda u: u.id))

    cities = tuple(
        City(
            id=c.city_id,
            owner=me,
            name=c.name,
            coord=coord(c),
            population=c.population,
            yields=Yields(
                food=c.food,
                production=c.production,
                gold=c.gold,
                science=c.science,
                culture=c.culture,
                faith=c.faith,
            ),
            food_stored=getattr(c, "food_stored", 0.0),
            production=short(getattr(c, "currently_building", None)),
            is_capital="BUILDING_PALACE" in (getattr(c, "buildings", None) or []),
        )
        for c in sorted(raw.cities, key=lambda c: c.city_id)
    )

    tech = raw.tech
    for t in tech.available_techs:
        look.techs[short(t.tech_type)] = t.tech_type
    for c in tech.available_civics:
        look.civics[short(c.civic_type)] = c.civic_type
    tech_state = _research(tech.current_research, tech.available_techs, "tech_type")
    civic_state = _research(tech.current_civic, tech.available_civics, "civic_type")

    contacts = tuple(
        Contact(player_id=d.player_id, civ=d.civ_name, at_war=d.is_at_war)
        for d in sorted(raw.diplomacy, key=lambda d: d.player_id)
        if d.has_met and d.player_id != me
    )

    player = Player(
        id=me,
        civ=ov.civ_name,
        leader=ov.leader_name,
        gold=ov.gold,
        yields=Yields(
            gold=ov.gold_per_turn,
            science=ov.science_yield,
            culture=ov.culture_yield,
        ),
        score=ov.score,
        era=getattr(ov, "era_name", None) or None,
    )

    for city_id, options in raw.production.items():
        for o in options:
            name = short(o.item_name)
            if o.category in PRODUCIBLE and name and not getattr(o, "is_repair", False):
                look.production[(city_id, name)] = (o.category, o.item_name)

    blockers = _blockers(raw, cities)
    legal = _legal(units, me, tiles, look)
    limit = max_turns or getattr(ov, "max_turns", 0) or DEFAULT_MAX_TURNS
    done = raw.game_over is not None or ov.turn > limit

    snap = Snapshot(
        turn=ov.turn,
        max_turns=limit,
        player=player,
        cities=cities,
        units=units,
        tech=tech_state,
        civics=civic_state,
        contacts=contacts,
        tiles=tuple(tiles[c] for c in sorted(tiles, key=lambda c: (c.q, c.r))),
        blockers=blockers,
        legal_actions=() if done else legal,
        done=done,
    )
    return snap, look


def _research(current: str, options: list[Any], type_field: str) -> ResearchState:
    available = tuple(short(getattr(o, type_field)) for o in options)
    now = next(
        (o for o in options if current in (o.name, getattr(o, type_field))),
        None,
    )
    if now is None:
        return ResearchState(current=short(current) if current else None, available=available)
    return ResearchState(
        current=short(getattr(now, type_field)),
        progress=now.cost * now.progress_pct / 100,
        cost=now.cost,
        available=available,
    )


def _blockers(raw: RawState, cities: tuple[City, ...]) -> tuple[Blocker, ...]:
    out: list[Blocker] = []
    for kind, message in raw.blockers:
        if kind == "ENDTURN_BLOCKING_UNITS":
            out += [
                Blocker(kind="unit_needs_orders", ref_id=uid, message=message)
                for uid in sorted(raw.ready_units)
            ] or [Blocker(kind="unit_needs_orders", message=message)]
        elif kind == "ENDTURN_BLOCKING_PRODUCTION":
            idle = [c for c in cities if c.production is None]
            out += [
                Blocker(kind="city_needs_production", ref_id=c.id, message=c.name) for c in idle
            ] or [Blocker(kind="city_needs_production", message=message)]
        elif kind == "ENDTURN_BLOCKING_RESEARCH":
            out.append(Blocker(kind="research_needed", message=message))
        elif kind == "ENDTURN_BLOCKING_CIVIC":
            out.append(Blocker(kind="civic_needed", message=message))
        else:
            label = kind.removeprefix("ENDTURN_BLOCKING_").lower()
            out.append(Blocker(kind="other", message=f"{label}: {message}" if message else label))
    return tuple(out)


def _legal(
    units: tuple[Unit, ...],
    me: int,
    tiles: dict[Coord, Tile],
    look: Lookup,
) -> tuple[Action, ...]:
    acts: list[Action] = []
    for u in units:
        if u.owner != me or u.moves_left <= 0:
            continue
        for n in neighbors(u.coord):
            tile = tiles.get(n)
            if tile is None or not _passable(tile):
                continue
            if n in look.foreign_units and u.type in ("settler", "builder", "trader"):
                continue
            acts.append(MoveUnit(unit_id=u.id, to=n))
        acts.append(UnitCommand(unit_id=u.id, command="skip"))
        if u.type not in ("settler", "builder", "trader", "scout"):
            acts.append(UnitCommand(unit_id=u.id, command="fortify"))
        if u.type == "settler":
            acts.append(UnitCommand(unit_id=u.id, command="found_city"))
    for city_id, name in sorted(look.production):
        acts.append(SetProduction(city_id=city_id, item=name))
    acts += [SetResearch(tech=t) for t in sorted(look.techs)]
    acts += [SetCivic(civic=c) for c in sorted(look.civics)]
    acts.append(EndTurn())
    return tuple(acts)
