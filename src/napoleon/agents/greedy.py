"""A fixed-rule player: settle, expand to a few cities, build up, research in order."""

from __future__ import annotations

from napoleon.agents.base import Decision
from napoleon.hexgrid import distance
from napoleon.schema import (
    EndTurn,
    MoveUnit,
    SetCivic,
    SetProduction,
    SetResearch,
    Snapshot,
    UnitCommand,
)

BUILD_ORDER = ("monument", "granary", "library")


class GreedyAgent:
    name = "greedy"

    def __init__(self, max_cities: int = 4) -> None:
        self.max_cities = max_cities

    def reset(self, seed: int) -> None:
        pass

    def act(self, snap: Snapshot) -> Decision:
        legal = set(snap.legal_actions)

        for u in snap.units:
            found = UnitCommand(unit_id=u.id, command="found_city")
            if found in legal:
                return Decision(found, f"settler {u.id} can found a city here")

        for u in snap.units:
            if u.type == "settler" and u.moves_left > 0:
                moves = [a for a in legal if isinstance(a, MoveUnit) and a.unit_id == u.id]
                if moves:
                    best = max(moves, key=lambda a: self._room(snap, a))
                    return Decision(best, f"settler {u.id} moves away from existing cities")

        for b in snap.blockers:
            if b.kind == "unit_needs_orders":
                fortify = UnitCommand(unit_id=b.ref_id, command="fortify")
                if fortify in legal:
                    return Decision(fortify, f"unit {b.ref_id} guards")
                return Decision(UnitCommand(unit_id=b.ref_id, command="skip"), "nothing to do")
            if b.kind == "city_needs_production":
                return Decision(self._production(snap, b.ref_id, legal), "keep the city busy")
            if b.kind == "research_needed":
                return Decision(SetResearch(tech=snap.tech.available[0]), "next tech in order")
            if b.kind == "civic_needed":
                return Decision(SetCivic(civic=snap.civics.available[0]), "next civic in order")

        return Decision(EndTurn(), "nothing left this turn")

    def _room(self, snap: Snapshot, move: MoveUnit) -> int:
        if not snap.cities:
            return 0
        return min(distance(move.to, c.coord) for c in snap.cities)

    def _production(self, snap: Snapshot, city_id: int, legal: set) -> SetProduction:
        city = next(c for c in snap.cities if c.id == city_id)
        settlers = sum(1 for u in snap.units if u.type == "settler")
        wanted = []
        if len(snap.cities) + settlers < self.max_cities and city.population >= 2:
            wanted.append("settler")
        if not any(u.type == "warrior" for u in snap.units):
            wanted.append("warrior")
        wanted += [*BUILD_ORDER, "warrior"]
        for item in wanted:
            action = SetProduction(city_id=city_id, item=item)
            if action in legal:
                return action
        return next(a for a in legal if isinstance(a, SetProduction) and a.city_id == city_id)
