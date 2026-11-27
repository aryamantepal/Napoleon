"""Read-only lookups over a snapshot, for agents that gather facts before they act.

Every tool returns plain text and never changes the game. Bad arguments raise
``ToolError`` with a message the model can read and correct.
"""

from __future__ import annotations

from collections.abc import Callable

from napoleon.agents.prompting import action_json
from napoleon.hexgrid import distance
from napoleon.schema import Coord, Snapshot

BAD_SITES = {"ocean", "mountains"}
MIN_CITY_DISTANCE = 3

TOOL_DOCS = """\
Lookup tools (they never change the game; each call uses one lookup):
- inspect_tile {"q": int, "r": int}: terrain, owner and anything standing on a known tile
- inspect_unit {"unit_id": int}: one unit in detail, with its legal actions
- inspect_city {"city_id": int}: one city in detail, with what it can build
- list_options {"type": str?, "unit_id": int?, "city_id": int?}: legal actions, optionally
  filtered by action type (move_unit, unit_command, set_production, set_research, set_civic,
  end_turn) or by unit or city
- find_city_sites {"unit_id": int?}: known tiles where a city could go, nearest first
- threats {}: foreign units near your cities and units
"""


class ToolError(ValueError):
    """The call could not be answered, with a reason the model can act on."""


def _int(args: dict, key: str) -> int:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"argument {key!r} must be an integer")
    return value


class SnapshotTools:
    def __init__(self, snap: Snapshot, max_rows: int = 40) -> None:
        self.snap = snap
        self.me = snap.player.id
        self.max_rows = max_rows
        self._tools: dict[str, Callable[[dict], str]] = {
            "inspect_tile": self.inspect_tile,
            "inspect_unit": self.inspect_unit,
            "inspect_city": self.inspect_city,
            "list_options": self.list_options,
            "find_city_sites": self.find_city_sites,
            "threats": self.threats,
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def run(self, name: str, args: dict | None) -> str:
        if name not in self._tools:
            raise ToolError(f"unknown tool {name!r}; tools are {', '.join(self._tools)}")
        if args is not None and not isinstance(args, dict):
            raise ToolError("args must be a JSON object")
        return self._tools[name](args or {})

    # Lookups --------------------------------------------------------------------

    def inspect_tile(self, args: dict) -> str:
        coord = Coord(q=_int(args, "q"), r=_int(args, "r"))
        tile = next((t for t in self.snap.tiles if t.coord == coord), None)
        if tile is None:
            return f"({coord.q},{coord.r}) is unknown: not revealed yet."
        bits = [tile.terrain]
        bits += [x for x in (tile.feature, tile.resource) if x]
        if tile.river:
            bits.append("river")
        bits.append("owner " + (str(tile.owner) if tile.owner is not None else "none"))
        bits.append("in sight" if tile.visible else "fogged, remembered")
        out = [f"({coord.q},{coord.r}): " + ", ".join(bits)]
        out += [
            f"city {c.id} {c.name} (player {c.owner}, pop {c.population})"
            for c in self.snap.cities
            if c.coord == coord
        ]
        out += [
            f"{u.type} {u.id} (player {u.owner}, hp {u.hp})"
            for u in self.snap.units
            if u.coord == coord
        ]
        return "\n".join(out)

    def inspect_unit(self, args: dict) -> str:
        uid = _int(args, "unit_id")
        unit = next((u for u in self.snap.units if u.id == uid), None)
        if unit is None:
            raise ToolError(f"no unit {uid} in sight")
        out = [
            f"{unit.type} {unit.id}, player {unit.owner}, at ({unit.coord.q},{unit.coord.r}), "
            f"hp {unit.hp}, moves {unit.moves_left}/{unit.max_moves}"
            + (", fortified" if unit.fortified else "")
        ]
        if unit.owner == self.me:
            acts = [a for a in self.snap.legal_actions if getattr(a, "unit_id", None) == uid]
            out.append(f"legal actions ({len(acts)}):")
            out += [action_json(a) for a in acts[: self.max_rows]]
        return "\n".join(out)

    def inspect_city(self, args: dict) -> str:
        cid = _int(args, "city_id")
        city = next((c for c in self.snap.cities if c.id == cid), None)
        if city is None:
            raise ToolError(f"no city {cid} in sight")
        y = ", ".join(f"{k} {v:g}" for k, v in city.yields.model_dump().items() if v)
        cost = f"/{city.production_cost:g}" if city.production_cost else ""
        out = [
            f"city {city.id} {city.name}, player {city.owner}, at ({city.coord.q},{city.coord.r}), "
            f"pop {city.population}, hp {city.hp}, food stored {city.food_stored:g}",
            f"yields: {y or 'none'}",
            f"building: {city.production or 'nothing'} {city.production_progress:g}{cost}",
        ]
        if city.owner == self.me:
            items = [
                a.item
                for a in self.snap.legal_actions
                if getattr(a, "city_id", None) == cid and a.type == "set_production"
            ]
            out.append(f"can build: {', '.join(items) or 'nothing'}")
        return "\n".join(out)

    def list_options(self, args: dict) -> str:
        acts = list(self.snap.legal_actions)
        kind = args.get("type")
        if kind is not None:
            acts = [a for a in acts if a.type == kind]
        for key in ("unit_id", "city_id"):
            if key in args:
                want = _int(args, key)
                acts = [a for a in acts if getattr(a, key, None) == want]
        if not acts:
            return "no legal actions match"
        out = [f"{len(acts)} legal action(s):"]
        out += [action_json(a) for a in acts[: self.max_rows]]
        if len(acts) > self.max_rows:
            out.append(
                f"... {len(acts) - self.max_rows} more; narrow with type, unit_id or city_id"
            )
        return "\n".join(out)

    def find_city_sites(self, args: dict) -> str:
        if "unit_id" in args:
            uid = _int(args, "unit_id")
            unit = next((u for u in self.snap.units if u.id == uid), None)
            if unit is None:
                raise ToolError(f"no unit {uid} in sight")
            origin = unit.coord
        else:
            mine = [c.coord for c in self.snap.cities if c.owner == self.me]
            origin = mine[0] if mine else Coord(q=0, r=0)
        cities = [c.coord for c in self.snap.cities]
        sites = [
            t
            for t in self.snap.tiles
            if t.terrain not in BAD_SITES
            and t.owner is None
            and all(distance(t.coord, c) >= MIN_CITY_DISTANCE for c in cities)
        ]
        sites.sort(key=lambda t: (distance(origin, t.coord), t.coord.q, t.coord.r))
        if not sites:
            return "no known free site at least 3 tiles from every city"
        rows = [
            f"({t.coord.q},{t.coord.r}) {t.terrain}"
            + (f" {t.resource}" if t.resource else "")
            + (" river" if t.river else "")
            + f", distance {distance(origin, t.coord)}"
            for t in sites[:8]
        ]
        return f"candidate city sites by distance from ({origin.q},{origin.r}):\n" + "\n".join(rows)

    def threats(self, args: dict) -> str:
        mine = [c.coord for c in self.snap.cities if c.owner == self.me]
        mine += [u.coord for u in self.snap.units if u.owner == self.me]
        found = []
        for u in self.snap.units:
            if u.owner == self.me:
                continue
            near = min((distance(u.coord, m) for m in mine), default=None)
            if near is not None and near <= 4:
                found.append((near, u))
        if not found:
            return "no foreign units within 4 tiles of your cities and units"
        found.sort(key=lambda p: p[0])
        return "\n".join(
            f"{u.type} of player {u.owner} at ({u.coord.q},{u.coord.r}), hp {u.hp}, "
            f"{d} tiles from your nearest city or unit"
            for d, u in found
        )
