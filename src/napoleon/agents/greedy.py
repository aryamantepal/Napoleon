"""A fixed-rule player: settle, expand to a few cities, build up, research in order."""

from __future__ import annotations

from napoleon.agents.base import Decision
from napoleon.hexgrid import distance
from napoleon.schema import (
    Coord,
    EndTurn,
    MoveUnit,
    SetCivic,
    SetProduction,
    SetResearch,
    Snapshot,
    UnitCommand,
)

BUILD_ORDER = ("monument", "granary", "library")
BAD_SITES = {"ocean", "mountains"}


class GreedyAgent:
    name = "greedy"

    def __init__(self, max_cities: int = 4) -> None:
        self.max_cities = max_cities

    def reset(self, seed: int) -> None:
        pass

    def act(self, snap: Snapshot) -> Decision:
        legal = set(snap.legal_actions)
        mine = [u for u in snap.units if u.owner == snap.player.id]
        enemies = {u.coord for u in snap.units if u.owner != snap.player.id}

        threats = [e for e in enemies if any(distance(e, c.coord) <= 4 for c in snap.cities)]
        for u in mine:
            if u.type != "warrior" or u.hp < 50:
                continue
            moves = [a for a in legal if isinstance(a, MoveUnit) and a.unit_id == u.id]
            for a in moves:
                if a.to in enemies:
                    return Decision(a, f"warrior {u.id} attacks an adjacent enemy")
            if threats and moves:
                target = min(threats, key=lambda e: distance(u.coord, e))
                best = min(moves, key=lambda a: distance(a.to, target))
                if distance(best.to, target) < distance(u.coord, target):
                    return Decision(best, f"warrior {u.id} closes on a threat near our cities")

        for u in mine:
            found = UnitCommand(unit_id=u.id, command="found_city")
            if found in legal:
                return Decision(found, f"settler {u.id} can found a city here")

        for u in mine:
            if u.type == "settler" and u.moves_left > 0:
                moves = [
                    a
                    for a in legal
                    if isinstance(a, MoveUnit) and a.unit_id == u.id and a.to not in enemies
                ]
                if not moves:
                    continue
                site = self._site(snap, u.coord)
                if site is not None:
                    best = min(moves, key=lambda a: (distance(a.to, site), a.to.q, a.to.r))
                    if distance(best.to, site) < distance(u.coord, site):
                        return Decision(best, f"settler {u.id} heads for a city site at {site}")
                else:
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

    def _site(self, snap: Snapshot, start: Coord) -> Coord | None:
        """The nearest known tile where a city could go, clear of everyone's land."""
        foreign = [t.coord for t in snap.tiles if t.owner not in (None, snap.player.id)]
        sites = [
            t.coord
            for t in snap.tiles
            if t.terrain not in BAD_SITES
            and t.owner is None
            and all(distance(t.coord, c.coord) >= 3 for c in snap.cities)
            and all(distance(t.coord, f) >= 4 for f in foreign)
        ]
        if not sites:
            return None
        return min(sites, key=lambda c: (distance(start, c), c.q, c.r))

    def _room(self, snap: Snapshot, move: MoveUnit) -> int:
        if not snap.cities:
            return 0
        return min(distance(move.to, c.coord) for c in snap.cities)

    def _production(self, snap: Snapshot, city_id: int, legal: set) -> SetProduction:
        city = next(c for c in snap.cities if c.id == city_id)
        mine = [u for u in snap.units if u.owner == snap.player.id]
        settlers = sum(1 for u in mine if u.type == "settler")
        warriors = sum(1 for u in mine if u.type == "warrior")
        wanted = []
        if len(snap.cities) + settlers < self.max_cities and city.population >= 2:
            wanted.append("settler")
        if warriors < len(snap.cities):
            wanted.insert(0, "warrior")
        wanted += [*BUILD_ORDER, "warrior"]
        for item in wanted:
            action = SetProduction(city_id=city_id, item=item)
            if action in legal:
                return action
        return next(a for a in legal if isinstance(a, SetProduction) and a.city_id == city_id)
