"""``GameBackend`` over civ6-mcp."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping
from typing import Any, TypeVar

from napoleon.civ6.coords import axial_to_offset
from napoleon.civ6.names import civic_type, tech_type
from napoleon.civ6.source import Civ6Source, is_error, unit_index
from napoleon.civ6.translate import Lookup, RawState, translate
from napoleon.schema import (
    Action,
    EndTurn,
    MoveUnit,
    SetCivic,
    SetProduction,
    SetResearch,
    Snapshot,
    StepResult,
    UnitCommand,
)

T = TypeVar("T")

CITY_SIGHT = 3
UNIT_SIGHT = 2
MAX_EVENT_CHARS = 2000


class Civ6Backend:
    """Real Civ VI through civ6-mcp.

    ``saves`` maps scenario names to save names that ``load_game_save`` accepts, so
    ``reset("bad_opening", seed)`` loads the save named for ``bad_opening``. A save fixes
    the map and the AI, so ``seed`` is recorded by the runner but does not change the
    game.

    civ6-mcp is async; this class owns an event loop and runs each call to completion,
    so it fits the synchronous ``GameBackend`` protocol. Pass ``loop`` when the source
    was built on an existing loop (``napoleon.civ6.live`` does this).
    """

    def __init__(
        self,
        source: Civ6Source,
        saves: Mapping[str, str],
        *,
        max_turns: int | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.source = source
        self.saves = dict(saves)
        self.max_turns = max_turns
        self.loop = loop or asyncio.new_event_loop()
        self._snap: Snapshot | None = None
        self._lookup = Lookup()
        self._ready = False

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        return self.loop.run_until_complete(coro)

    # GameBackend -----------------------------------------------------------------

    def reset(self, scenario: str, seed: int) -> None:
        if scenario not in self.saves:
            raise ValueError(f"no save for scenario {scenario!r}; known: {sorted(self.saves)}")
        result = self.run(self.source.load_game_save(self.saves[scenario]))
        if is_error(result):
            raise RuntimeError(f"could not load {self.saves[scenario]!r}: {result}")
        self._ready = True
        self._snap = None

    def snapshot(self) -> Snapshot:
        if not self._ready:
            raise RuntimeError("call reset() first")
        if self._snap is None:
            raw = self.run(self._read())
            self._snap, self._lookup = translate(raw, max_turns=self.max_turns)
        return self._snap

    def apply(self, action: Action) -> StepResult:
        before = self.snapshot()
        try:
            result = self.run(self._dispatch(action))
        finally:
            self._snap = None
        if is_error(result):
            return StepResult(ok=False, error=_clip(result))
        after = self.snapshot()
        return StepResult(
            ok=True,
            events=(_clip(result),) if result else (),
            turn_ended=after.turn > before.turn,
            done=after.done,
        )

    # Reads -----------------------------------------------------------------------

    async def _read(self) -> RawState:
        s = self.source
        overview = await s.get_game_overview()
        units = await s.get_units()
        cities, _ = await s.get_cities()
        tech = await s.get_tech_civics()
        diplomacy = await s.get_diplomacy()
        blockers = await s.end_turn_blockers()
        ready = await s.ready_unit_ids()
        game_over = await s.check_game_over()

        centers: dict[tuple[int, int], int] = {}
        for c in cities:
            centers[(c.x, c.y)] = max(centers.get((c.x, c.y), 0), CITY_SIGHT)
        for u in units:
            centers[(u.x, u.y)] = max(centers.get((u.x, u.y), 0), UNIT_SIGHT)
        tiles: list[Any] = []
        for (x, y), radius in sorted(centers.items()):
            tiles += await s.get_map_area(x, y, radius)

        production = {c.city_id: await s.list_city_production(c.city_id) for c in cities}
        return RawState(
            overview=overview,
            units=units,
            cities=cities,
            tech=tech,
            diplomacy=diplomacy,
            tiles=tiles,
            production=production,
            blockers=blockers,
            ready_units=ready,
            game_over=game_over,
        )

    # Actions ---------------------------------------------------------------------

    async def _dispatch(self, action: Action) -> str:
        s = self.source
        match action:
            case MoveUnit(unit_id=uid, to=to):
                x, y = axial_to_offset(to)
                if to in self._lookup.foreign_units:
                    return await s.attack_unit(unit_index(uid), x, y)
                return await s.move_unit(unit_index(uid), x, y)
            case UnitCommand(unit_id=uid, command="found_city"):
                return await s.found_city(unit_index(uid))
            case UnitCommand(unit_id=uid, command="fortify"):
                return await s.fortify_unit(unit_index(uid))
            case UnitCommand(unit_id=uid, command="skip"):
                return await s.skip_unit(unit_index(uid))
            case SetProduction(city_id=cid, item=item):
                category, name = self._lookup.production[(cid, item)]
                return await s.set_city_production(cid, category, name)
            case SetResearch(tech=t):
                return await s.set_research(self._lookup.techs.get(t, tech_type(t)))
            case SetCivic(civic=c):
                return await s.set_civic(self._lookup.civics.get(c, civic_type(c)))
            case EndTurn():
                return await s.end_turn()
        raise TypeError(f"unsupported action {action!r}")


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_EVENT_CHARS else text[: MAX_EVENT_CHARS - 1] + "…"
