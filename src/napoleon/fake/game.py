"""A small deterministic Civ-like game for building and testing without Civ VI.

It is not a Civ VI simulation. It keeps the parts of the shape that matter for the
adapter and the agents: a fogged hex map, units that move and found cities, cities that
grow and build, tech and civic trees, turn blockers, a rival, score and victory
progress. Same scenario and seed always give the same game.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from napoleon.hexgrid import distance, neighbors, within
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
    StepResult,
    Tile,
    Unit,
    UnitCommand,
    Yields,
)

SCENARIOS = ("default", "bad_opening")

TERRAIN_YIELDS: dict[str, Yields] = {
    "grassland": Yields(food=2),
    "plains": Yields(food=1, production=1),
    "hills": Yields(production=2),
    "desert": Yields(),
    "tundra": Yields(food=1),
    "ocean": Yields(food=1, gold=1),
    "mountains": Yields(),
}
IMPASSABLE = {"ocean", "mountains"}
MOVE_COST = {"hills": 2}

UNIT_MOVES = {"settler": 2, "warrior": 2}

# item -> (cost, required tech or None)
PRODUCTION: dict[str, tuple[int, str | None]] = {
    "warrior": (30, None),
    "settler": (60, None),
    "monument": (40, None),
    "granary": (50, "pottery"),
    "library": (60, "writing"),
}
BUILDINGS = {"monument", "granary", "library"}

# name -> (cost, prerequisites)
TECHS: dict[str, tuple[int, tuple[str, ...]]] = {
    "pottery": (20, ()),
    "mining": (20, ()),
    "writing": (35, ("pottery",)),
    "bronze_working": (35, ("mining",)),
    "currency": (50, ("writing",)),
    "mathematics": (60, ("currency", "bronze_working")),
}
CIVICS: dict[str, tuple[int, tuple[str, ...]]] = {
    "code_of_laws": (20, ()),
    "craftsmanship": (30, ("code_of_laws",)),
    "early_empire": (35, ("code_of_laws",)),
    "political_philosophy": (55, ("craftsmanship", "early_empire")),
}

# Combat. A unit attacks by moving onto an enemy unit's tile.
UNIT_STRENGTH = {"settler": 0, "warrior": 20}
FORTIFY_BONUS = 6
CITY_STRENGTH = 15
MAX_HP = 100
HEAL_PER_TURN = 10
BARBARIAN_FIRST_TURN = 8
BARBARIAN_EVERY = 10
MAX_BARBARIANS = 2
RIVAL_EXPANDS_EVERY = 20

MIN_CITY_DISTANCE = 3
SIGHT = 2
CITY_NAMES = ("Lutetia", "Massilia", "Lugdunum", "Burdigala", "Tolosa", "Arelate", "Nemausus")


@dataclass
class _Unit:
    id: int
    owner: int
    type: str
    coord: Coord
    moves_left: int
    fortified: bool = False
    skipped: bool = False
    hp: int = MAX_HP


@dataclass
class _City:
    id: int
    owner: int
    name: str
    coord: Coord
    population: int = 1
    food_stored: float = 0.0
    production: str | None = None
    production_progress: float = 0.0
    buildings: set[str] = field(default_factory=set)
    is_capital: bool = False
    hp: int = MAX_HP


@dataclass
class _Tree:
    spec: dict[str, tuple[int, tuple[str, ...]]]
    completed: list[str] = field(default_factory=list)
    current: str | None = None
    progress: float = 0.0

    def available(self) -> list[str]:
        done = set(self.completed)
        return [n for n, (_, pre) in self.spec.items() if n not in done and set(pre) <= done]

    def state(self) -> ResearchState:
        cost = self.spec[self.current][0] if self.current else None
        return ResearchState(
            completed=tuple(self.completed),
            current=self.current,
            progress=self.progress,
            cost=cost,
            available=tuple(self.available()),
        )

    def add(self, points: float) -> str | None:
        """Add points; return the name of anything completed."""
        if self.current is None:
            return None
        self.progress += points
        cost = self.spec[self.current][0]
        if self.progress < cost:
            return None
        done = self.current
        self.completed.append(done)
        self.progress -= cost
        self.current = None
        return done


class FakeCivGame:
    """Implements ``GameBackend`` for one human-controlled player (id 0) and one rival."""

    PLAYER = 0
    RIVAL = 1
    BARBARIANS = 2

    def __init__(self, radius: int = 7, max_turns: int = 100) -> None:
        self.radius = radius
        self.max_turns = max_turns
        self._initialized = False

    # GameBackend -----------------------------------------------------------------

    def reset(self, scenario: str, seed: int) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
        self.scenario = scenario
        self.rng = random.Random(f"{scenario}:{seed}")
        self.turn = 1
        self.gold = 10.0 if scenario == "default" else 0.0
        self.rival_score = 0
        self.met_rival = False
        self.tech = _Tree(TECHS)
        self.civics = _Tree(CIVICS)
        self.units: dict[int, _Unit] = {}
        self.cities: dict[int, _City] = {}
        self._next_id = 1
        self.terrain = self._generate_map()
        self.revealed: set[Coord] = set()
        self._place_start()
        self._update_sight()
        self._initialized = True

    def snapshot(self) -> Snapshot:
        self._require()
        visible = self._visible()
        tiles = tuple(
            Tile(
                coord=c,
                terrain=self.terrain[c],
                owner=self._owner(c),
                visible=c in visible,
            )
            for c in sorted(self.revealed, key=lambda c: (c.q, c.r))
        )
        cities = tuple(self._city_view(c) for c in self.cities.values() if c.owner == self.PLAYER)
        # Other players' units show only while in sight.
        units = tuple(
            Unit(
                id=u.id,
                owner=u.owner,
                type=u.type,
                coord=u.coord,
                hp=u.hp,
                moves_left=u.moves_left if u.owner == self.PLAYER else 0,
                max_moves=UNIT_MOVES[u.type],
                fortified=u.fortified,
            )
            for u in sorted(self.units.values(), key=lambda u: u.id)
            if u.owner == self.PLAYER or u.coord in visible
        )
        contacts = (
            (Contact(player_id=self.RIVAL, civ="Carthage", score=self.rival_score),)
            if self.met_rival
            else ()
        )
        blockers = self._blockers()
        done = self._done()
        return Snapshot(
            turn=self.turn,
            max_turns=self.max_turns,
            player=Player(
                id=self.PLAYER,
                civ="France",
                leader="Napoleon",
                gold=self.gold,
                yields=self._empire_yields(),
                score=self.score(),
            ),
            cities=cities,
            units=units,
            tech=self.tech.state(),
            civics=self.civics.state(),
            contacts=contacts,
            tiles=tiles,
            victory_progress={
                "science": len(self.tech.completed) / len(TECHS),
                "culture": len(self.civics.completed) / len(CIVICS),
            },
            blockers=blockers,
            legal_actions=() if done else self._legal_actions(blockers),
            done=done,
        )

    def apply(self, action: Action) -> StepResult:
        self._require()
        if isinstance(action, MoveUnit):
            return self._move(action)
        if isinstance(action, UnitCommand):
            return self._command(action)
        if isinstance(action, SetProduction):
            city = self.cities[action.city_id]
            city.production = action.item
            return StepResult(ok=True, events=(f"{city.name} now building {action.item}",))
        if isinstance(action, SetResearch):
            self.tech.current = action.tech
            return StepResult(ok=True, events=(f"researching {action.tech}",))
        if isinstance(action, SetCivic):
            self.civics.current = action.civic
            return StepResult(ok=True, events=(f"developing {action.civic}",))
        if isinstance(action, EndTurn):
            return self._end_turn()
        raise TypeError(f"unhandled action {action!r}")  # pragma: no cover

    # Public helpers ----------------------------------------------------------------

    def score(self) -> int:
        pop = sum(c.population for c in self.cities.values() if c.owner == self.PLAYER)
        ncities = sum(1 for c in self.cities.values() if c.owner == self.PLAYER)
        buildings = sum(len(c.buildings) for c in self.cities.values() if c.owner == self.PLAYER)
        return (
            5 * ncities
            + 2 * pop
            + 3 * len(self.tech.completed)
            + 2 * len(self.civics.completed)
            + buildings
        )

    # Map and setup -------------------------------------------------------------------

    def _generate_map(self) -> dict[Coord, str]:
        origin = Coord(q=0, r=0)
        land = ["grassland"] * 5 + ["plains"] * 4 + ["hills"] * 2 + ["desert", "tundra"]
        terrain: dict[Coord, str] = {}
        for c in within(origin, self.radius):
            if distance(c, origin) == self.radius:
                terrain[c] = "ocean"
            elif self.rng.random() < 0.06:
                terrain[c] = "mountains"
            else:
                terrain[c] = self.rng.choice(land)
        return terrain

    def _place_start(self) -> None:
        origin = Coord(q=0, r=0)
        rival_at = Coord(q=self.radius - 2, r=-(self.radius - 2))
        self.terrain[rival_at] = "plains"
        rival = self._new_city(self.RIVAL, rival_at, name="Carthage")
        rival.population = 3

        if self.scenario == "default":
            start = origin
            for c in [start, *neighbors(start)]:
                if self.terrain[c] in IMPASSABLE:
                    self.terrain[c] = "plains"
            self.terrain[start] = "grassland"
            self._new_unit("settler", start)
            self._new_unit("warrior", neighbors(start)[0])
        else:
            # Bad opening: a lone settler on poor land near the edge, no escort, no gold.
            start = Coord(q=-(self.radius - 2), r=self.radius - 2)
            for c in within(start, 1):
                if c in self.terrain and self.terrain[c] != "ocean":
                    self.terrain[c] = self.rng.choice(["desert", "tundra", "desert"])
            self.terrain[start] = "desert"
            self._new_unit("settler", start)

    def _new_unit(self, kind: str, coord: Coord, owner: int = PLAYER) -> _Unit:
        unit = _Unit(id=self._take_id(), owner=owner, type=kind, coord=coord, moves_left=0)
        unit.moves_left = UNIT_MOVES[kind]
        self.units[unit.id] = unit
        return unit

    def _new_city(self, owner: int, coord: Coord, name: str | None = None) -> _City:
        own = [c for c in self.cities.values() if c.owner == owner]
        if name is None:
            name = CITY_NAMES[len(own) % len(CITY_NAMES)]
        city = _City(id=self._take_id(), owner=owner, name=name, coord=coord, is_capital=not own)
        self.cities[city.id] = city
        return city

    def _take_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # Views -----------------------------------------------------------------------------

    def _owner(self, c: Coord) -> int | None:
        for city in self.cities.values():
            if distance(c, city.coord) <= 1:
                return city.owner
        return None

    def _visible(self) -> set[Coord]:
        eyes = [u.coord for u in self.units.values() if u.owner == self.PLAYER]
        eyes += [c.coord for c in self.cities.values() if c.owner == self.PLAYER]
        seen: set[Coord] = set()
        for e in eyes:
            seen.update(c for c in within(e, SIGHT) if c in self.terrain)
        return seen

    def _update_sight(self) -> None:
        seen = self._visible()
        self.revealed |= seen
        rival = next(c for c in self.cities.values() if c.owner == self.RIVAL)
        if rival.coord in seen:
            self.met_rival = True

    def _city_yields(self, city: _City) -> Yields:
        center = TERRAIN_YIELDS[self.terrain[city.coord]]
        food, prod, gold = 2 + center.food, 1 + center.production, 2 + center.gold
        worked = sorted(
            (c for c in within(city.coord, 1) if c != city.coord and c in self.terrain),
            key=lambda c: (
                -(TERRAIN_YIELDS[self.terrain[c]].food + TERRAIN_YIELDS[self.terrain[c]].production)
            ),
        )[: city.population]
        for c in worked:
            y = TERRAIN_YIELDS[self.terrain[c]]
            food, prod, gold = food + y.food, prod + y.production, gold + y.gold
        if "granary" in city.buildings:
            food += 1
        science = 1 + 0.5 * city.population + (2 if "library" in city.buildings else 0)
        culture = 1 + (2 if "monument" in city.buildings else 0)
        return Yields(food=food, production=prod, gold=gold, science=science, culture=culture)

    def _city_view(self, city: _City) -> City:
        cost = PRODUCTION[city.production][0] if city.production else None
        return City(
            id=city.id,
            owner=city.owner,
            name=city.name,
            coord=city.coord,
            population=city.population,
            yields=self._city_yields(city),
            food_stored=city.food_stored,
            production=city.production,
            production_progress=city.production_progress,
            production_cost=cost,
            is_capital=city.is_capital,
            hp=city.hp,
        )

    def _empire_yields(self) -> Yields:
        total = Yields()
        for city in self.cities.values():
            if city.owner != self.PLAYER:
                continue
            y = self._city_yields(city)
            total = Yields(
                food=total.food + y.food,
                production=total.production + y.production,
                gold=total.gold + y.gold,
                science=total.science + y.science,
                culture=total.culture + y.culture,
            )
        return total

    # Rules -------------------------------------------------------------------------------

    def _can_found(self, coord: Coord) -> bool:
        if self.terrain[coord] in IMPASSABLE:
            return False
        return all(distance(coord, c.coord) >= MIN_CITY_DISTANCE for c in self.cities.values())

    def _buildable(self, city: _City) -> list[str]:
        out = []
        for item, (_, tech) in PRODUCTION.items():
            if tech and tech not in self.tech.completed:
                continue
            if item in BUILDINGS and item in city.buildings:
                continue
            if item == "settler" and city.population < 2:
                continue
            out.append(item)
        return out

    def _blockers(self) -> tuple[Blocker, ...]:
        out: list[Blocker] = []
        for u in self.units.values():
            if u.owner == self.PLAYER and u.moves_left > 0 and not (u.fortified or u.skipped):
                out.append(
                    Blocker(kind="unit_needs_orders", ref_id=u.id, message=f"{u.type} {u.id}")
                )
        for c in self.cities.values():
            if c.owner == self.PLAYER and c.production is None:
                out.append(Blocker(kind="city_needs_production", ref_id=c.id, message=f"{c.name}"))
        if self.cities and self.tech.current is None and self.tech.available():
            out.append(Blocker(kind="research_needed", message="choose a tech"))
        if self.cities and self.civics.current is None and self.civics.available():
            out.append(Blocker(kind="civic_needed", message="choose a civic"))
        return tuple(out)

    def _legal_actions(self, blockers: tuple[Blocker, ...]) -> tuple[Action, ...]:
        acts: list[Action] = []
        occupied = {u.coord for u in self.units.values()}
        enemies = {u.coord for u in self.units.values() if u.owner != self.PLAYER}
        foreign = {c.coord for c in self.cities.values() if c.owner != self.PLAYER}
        for u in sorted(self.units.values(), key=lambda u: u.id):
            if u.owner != self.PLAYER or u.moves_left <= 0:
                continue
            for n in neighbors(u.coord):
                if n not in self.terrain or self.terrain[n] in IMPASSABLE or n in foreign:
                    continue
                attack = n in enemies and UNIT_STRENGTH[u.type] > 0
                if attack or n not in occupied:
                    acts.append(MoveUnit(unit_id=u.id, to=n))
            acts.append(UnitCommand(unit_id=u.id, command="skip"))
            if u.type == "warrior" and not u.fortified:
                acts.append(UnitCommand(unit_id=u.id, command="fortify"))
            if u.type == "settler" and self._can_found(u.coord):
                acts.append(UnitCommand(unit_id=u.id, command="found_city"))
        for c in sorted(self.cities.values(), key=lambda c: c.id):
            if c.owner != self.PLAYER:
                continue
            for item in self._buildable(c):
                if item != c.production:
                    acts.append(SetProduction(city_id=c.id, item=item))
        if self.cities:
            acts += [SetResearch(tech=t) for t in self.tech.available() if t != self.tech.current]
            acts += [SetCivic(civic=t) for t in self.civics.available() if t != self.civics.current]
        if not blockers:
            acts.append(EndTurn())
        return tuple(acts)

    def _move(self, action: MoveUnit) -> StepResult:
        u = self.units[action.unit_id]
        target = self._unit_at(action.to)
        if target is not None and target.owner != u.owner:
            return StepResult(ok=True, events=tuple(self._attack(u, target)))
        u.coord = action.to
        u.moves_left = max(0, u.moves_left - MOVE_COST.get(self.terrain[action.to], 1))
        u.fortified = False
        before = self.met_rival
        self._update_sight()
        events = [f"{u.type} {u.id} moved to ({action.to.q}, {action.to.r})"]
        if self.met_rival and not before:
            events.append("met Carthage")
        return StepResult(ok=True, events=tuple(events))

    def _unit_at(self, c: Coord) -> _Unit | None:
        return next((u for u in self.units.values() if u.coord == c), None)

    def _city_at(self, c: Coord) -> _City | None:
        return next((x for x in self.cities.values() if x.coord == c), None)

    def _damage(self, attack: float, defend: float) -> int:
        """Civ-style damage: about 30 at equal strength, scaled by the difference."""
        return max(1, round(30 * math.exp((attack - defend) / 25) * self.rng.uniform(0.8, 1.2)))

    def _attack(self, att: _Unit, dfn: _Unit) -> list[str]:
        """Unit-on-unit fight. The winner moves in if the defender dies."""
        a = UNIT_STRENGTH[att.type] * (0.5 + att.hp / (2 * MAX_HP))
        d = UNIT_STRENGTH[dfn.type] + (FORTIFY_BONUS if dfn.fortified else 0)
        d *= 0.5 + dfn.hp / (2 * MAX_HP)
        events: list[str] = []
        att.moves_left = 0
        att.fortified = False
        if UNIT_STRENGTH[dfn.type] == 0:
            del self.units[dfn.id]
            att.coord = dfn.coord
            return [f"{self._label(att)} captured {self._label(dfn)}"]
        dfn.hp -= self._damage(a, d)
        att.hp -= self._damage(d, a)
        if dfn.hp <= 0:
            del self.units[dfn.id]
            events.append(f"{self._label(att)} destroyed {self._label(dfn)}")
            if att.hp > 0:
                att.coord = dfn.coord
        else:
            events.append(f"{self._label(att)} attacked {self._label(dfn)} ({dfn.hp} hp left)")
        if att.hp <= 0:
            del self.units[att.id]
            events.append(f"{self._label(att)} died attacking")
        self._update_sight()
        return events

    def _city_strength(self, city: _City) -> float:
        garrison = self._unit_at(city.coord)
        bonus = UNIT_STRENGTH[garrison.type] / 2 if garrison and garrison.owner == city.owner else 0
        return CITY_STRENGTH + 2 * city.population + bonus

    def _city_strikes(self) -> list[str]:
        """Each player city hits one adjacent barbarian, as cities do in Civ."""
        events: list[str] = []
        for city in [c for c in self.cities.values() if c.owner == self.PLAYER]:
            near = [
                u
                for u in self.units.values()
                if u.owner == self.BARBARIANS and distance(u.coord, city.coord) == 1
            ]
            if not near:
                continue
            b = min(near, key=lambda u: (u.hp, u.id))
            b.hp -= self._damage(self._city_strength(city), UNIT_STRENGTH[b.type])
            if b.hp <= 0:
                del self.units[b.id]
                events.append(f"{city.name} destroyed {self._label(b)}")
            else:
                events.append(f"{city.name} struck {self._label(b)} ({b.hp} hp left)")
        return events

    def _attack_city(self, att: _Unit, city: _City) -> list[str]:
        strength = self._city_strength(city)
        city.hp -= self._damage(UNIT_STRENGTH[att.type], strength)
        att.hp -= self._damage(strength, UNIT_STRENGTH[att.type])
        events = [f"{self._label(att)} attacked {city.name}"]
        if city.hp <= 0:
            city.population = max(1, city.population - 1)
            city.production_progress /= 2
            city.hp = MAX_HP // 2
            events.append(f"{city.name} was pillaged and shrank to {city.population}")
        if att.hp <= 0:
            del self.units[att.id]
            events.append(f"{self._label(att)} died at {city.name}")
        return events

    def _label(self, u: _Unit) -> str:
        who = {self.PLAYER: "", self.RIVAL: "Carthaginian ", self.BARBARIANS: "barbarian "}
        return f"{who[u.owner]}{u.type} {u.id}"

    def _barbarian_phase(self) -> list[str]:
        events = self._city_strikes()
        mine = [c for c in self.cities.values() if c.owner == self.PLAYER]
        if (
            mine
            and sum(u.owner == self.BARBARIANS for u in self.units.values()) < MAX_BARBARIANS
            and self.turn >= BARBARIAN_FIRST_TURN
            and (self.turn - BARBARIAN_FIRST_TURN) % BARBARIAN_EVERY == 0
        ):
            occupied = {u.coord for u in self.units.values()}
            spots = [
                c
                for c, t in self.terrain.items()
                if t not in IMPASSABLE
                and c not in occupied
                and self._city_at(c) is None
                and 4 <= min(distance(c, x.coord) for x in mine) <= 5
            ]
            if spots:
                spot = self.rng.choice(sorted(spots, key=lambda c: (c.q, c.r)))
                b = self._new_unit("warrior", spot, owner=self.BARBARIANS)
                events.append(f"barbarians appeared ({self._label(b)})")
        targets = [u.coord for u in self.units.values() if u.owner == self.PLAYER]
        targets += [c.coord for c in mine]
        for b in [u for u in self.units.values() if u.owner == self.BARBARIANS]:
            if not targets:
                break
            goal = min(targets, key=lambda t: (distance(b.coord, t), t.q, t.r))
            if distance(b.coord, goal) == 1:
                # A city shelters whatever stands in it, so a city tile is fought as a city.
                city = self._city_at(goal)
                unit = self._unit_at(goal)
                if city is not None:
                    events += self._attack_city(b, city)
                elif unit is not None:
                    events += self._attack(b, unit)
                continue
            occupied = {u.coord for u in self.units.values()}
            steps = [
                n
                for n in neighbors(b.coord)
                if n in self.terrain
                and self.terrain[n] not in IMPASSABLE
                and n not in occupied
                and self._city_at(n) is None
            ]
            if steps:
                b.coord = min(steps, key=lambda n: (distance(n, goal), n.q, n.r))
        return events

    def _rival_phase(self) -> list[str]:
        if self.turn % RIVAL_EXPANDS_EVERY != 0:
            return []
        home = next(c for c in self.cities.values() if c.owner == self.RIVAL and c.is_capital)
        spots = [
            c
            for c in within(home.coord, 4)
            if c in self.terrain and self._can_found(c) and self._unit_at(c) is None
        ]
        if not spots:
            return []
        spot = self.rng.choice(sorted(spots, key=lambda c: (c.q, c.r)))
        self._new_city(self.RIVAL, spot, name=f"Carthage {self.turn // RIVAL_EXPANDS_EVERY + 1}")
        self.rival_score += 5
        return ["Carthage founded a new city"] if self.met_rival else []

    def _command(self, action: UnitCommand) -> StepResult:
        u = self.units[action.unit_id]
        if action.command == "skip":
            u.skipped = True
            return StepResult(ok=True, events=(f"{u.type} {u.id} skips",))
        if action.command == "fortify":
            u.fortified = True
            return StepResult(ok=True, events=(f"{u.type} {u.id} fortified",))
        # found_city
        city = self._new_city(self.PLAYER, u.coord)
        del self.units[u.id]
        self._update_sight()
        return StepResult(ok=True, events=(f"founded {city.name}",))

    def _end_turn(self) -> StepResult:
        events: list[str] = []
        science = culture = gold = 0.0
        for city in [c for c in self.cities.values() if c.owner == self.PLAYER]:
            y = self._city_yields(city)
            science, culture, gold = science + y.science, culture + y.culture, gold + y.gold
            # Growth
            city.food_stored += y.food - 2 * city.population
            threshold = 15 + 8 * (city.population - 1)
            if city.food_stored >= threshold:
                city.population += 1
                city.food_stored -= threshold
                events.append(f"{city.name} grew to {city.population}")
            elif city.food_stored < 0:
                city.food_stored = 0.0
            # Production
            if city.production:
                city.production_progress += y.production
                cost = PRODUCTION[city.production][0]
                if city.production_progress >= cost:
                    events.append(self._complete(city))
                    city.production_progress -= cost
        self.gold += gold
        if done := self.tech.add(science):
            events.append(f"discovered {done}")
        if done := self.civics.add(culture):
            events.append(f"adopted {done}")
        # The rival grows on a fixed schedule so contact scores move.
        if self.turn % 2 == 0:
            self.rival_score += 1
        events += self._barbarian_phase()
        events += self._rival_phase()
        for u in self.units.values():
            if u.fortified or u.moves_left == UNIT_MOVES[u.type]:
                u.hp = min(MAX_HP, u.hp + HEAL_PER_TURN)
            u.moves_left = UNIT_MOVES[u.type]
            u.skipped = False
        for city in self.cities.values():
            city.hp = min(MAX_HP, city.hp + HEAL_PER_TURN)
        self.turn += 1
        self._update_sight()
        done = self._done()
        if done:
            events.append("game over")
        return StepResult(ok=True, events=tuple(events), turn_ended=True, done=done)

    def _complete(self, city: _City) -> str:
        item = city.production
        assert item is not None
        if item in BUILDINGS:
            city.buildings.add(item)
            city.production = None
            return f"{city.name} built {item}"
        if item == "settler":
            city.population = max(1, city.population - 1)
        spot = city.coord
        occupied = {u.coord for u in self.units.values()}
        if spot in occupied:
            free = [
                n
                for n in neighbors(spot)
                if n in self.terrain and self.terrain[n] not in IMPASSABLE and n not in occupied
            ]
            if not free:
                # Keep the progress so the unit appears once a tile frees up.
                city.production_progress += PRODUCTION[item][0]
                return f"{city.name} finished {item} but had no room; it waits"
            spot = free[0]
        self._new_unit(item, spot)
        # Like Civ VI, the city asks for new orders after every item.
        city.production = None
        return f"{city.name} trained {item}"

    def _done(self) -> bool:
        return self.turn > self.max_turns or len(self.tech.completed) == len(TECHS)

    def _require(self) -> None:
        if not self._initialized:
            raise RuntimeError("call reset() first")
