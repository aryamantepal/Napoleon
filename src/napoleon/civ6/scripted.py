"""A small stand-in for civ6-mcp, for tests and dry runs without the game.

``ScriptedSource`` returns objects with the same fields as civ6-mcp's dataclasses
(``GameOverview``, ``UnitInfo``, ``CityInfo``, ``TileInfo``, ``TechCivicStatus``,
``ProductionOption``) and answers actions with civ6-mcp-style result text. It plays a
tiny, rule-light opening: units move, settlers found cities, research and production
get set, and turns end. It is not a game; it exercises the translation and the
action mapping end to end.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

PLAYER = 0
START = (10, 10)
MAP_W, MAP_H = 24, 20

TECHS = [("Pottery", "TECHNOLOGY_POTTERY", 25), ("Mining", "TECHNOLOGY_MINING", 25)]
CIVICS = [("Code of Laws", "CIVIC_CODE_OF_LAWS", 20)]
PRODUCTION = [
    ("UNIT", "UNIT_WARRIOR", 40),
    ("UNIT", "UNIT_SETTLER", 80),
    ("BUILDING", "BUILDING_MONUMENT", 60),
    ("DISTRICT", "DISTRICT_CAMPUS", 54),
]


def _terrain(x: int, y: int) -> str:
    if x in (0, MAP_W - 1) or y in (0, MAP_H - 1):
        return "TERRAIN_OCEAN"
    if (x * 7 + y * 3) % 11 == 0:
        return "TERRAIN_GRASS_MOUNTAIN"
    return ("TERRAIN_GRASS", "TERRAIN_PLAINS", "TERRAIN_PLAINS_HILLS")[(x + 2 * y) % 3]


class ScriptedSource:
    def __init__(self, barbarian_at: tuple[int, int] | None = (12, 9)) -> None:
        self.calls: list[tuple] = []
        self._barbarian_at = barbarian_at
        self.saves = {"napoleon_default": "default"}
        self.loaded: str | None = None

    # Lifecycle -------------------------------------------------------------------

    async def load_game_save(self, save_name: str) -> str:
        self.calls.append(("load_game_save", save_name))
        if save_name not in self.saves:
            return f"Error: save {save_name!r} not found"
        self.loaded = save_name
        self.turn = 1
        self.units = {
            65537: NS(
                unit_id=65537,
                unit_index=1,
                name="Settler",
                unit_type="UNIT_SETTLER",
                x=START[0],
                y=START[1],
                moves_remaining=2.0,
                max_moves=2.0,
                health=100,
                max_health=100,
                combat_strength=0,
            ),
            65538: NS(
                unit_id=65538,
                unit_index=2,
                name="Warrior",
                unit_type="UNIT_WARRIOR",
                x=START[0] + 1,
                y=START[1],
                moves_remaining=2.0,
                max_moves=2.0,
                health=100,
                max_health=100,
                combat_strength=20,
            ),
        }
        self.cities: dict[int, NS] = {}
        self.research: str | None = None
        self.civic: str | None = None
        self.orders: set[int] = set()
        return f"Loaded {save_name}"

    # Reads -----------------------------------------------------------------------

    async def get_game_overview(self) -> NS:
        return NS(
            turn=self.turn,
            player_id=PLAYER,
            civ_name="France",
            leader_name="Catherine",
            gold=10.0,
            gold_per_turn=2.0,
            science_yield=1.5,
            culture_yield=1.0,
            faith=0.0,
            current_research=self._name(self.research, TECHS),
            current_civic=self._name(self.civic, CIVICS),
            num_cities=len(self.cities),
            num_units=len(self.units),
            score=len(self.cities) * 5,
            era_name="Ancient Era",
            max_turns=0,
        )

    async def get_units(self) -> list[NS]:
        return list(self.units.values())

    async def get_cities(self) -> tuple[list[NS], list[str]]:
        return list(self.cities.values()), []

    async def get_tech_civics(self) -> NS:
        def opts(rows, field):
            return [
                NS(
                    name=n,
                    **{field: t},
                    cost=c,
                    progress_pct=0,
                    turns=10,
                    boosted=False,
                    boost_desc="",
                )
                for n, t, c in rows
            ]

        return NS(
            current_research=self._name(self.research, TECHS) or "None",
            current_research_turns=0,
            current_civic=self._name(self.civic, CIVICS) or "None",
            current_civic_turns=0,
            available_techs=opts(TECHS, "tech_type"),
            available_civics=opts(CIVICS, "civic_type"),
        )

    async def get_diplomacy(self) -> list[NS]:
        return [
            NS(
                player_id=1,
                civ_name="Rome",
                leader_name="Trajan",
                has_met=self.turn > 1,
                is_at_war=False,
            )
        ]

    async def get_map_area(self, center_x: int, center_y: int, radius: int = 2) -> list[NS]:
        out = []
        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                if not (0 <= x < MAP_W and 0 <= y < MAP_H):
                    continue
                units = ["Barbarian WARRIOR"] if (x, y) == self._barbarian_at else None
                out.append(
                    NS(
                        x=x,
                        y=y,
                        terrain=_terrain(x, y),
                        feature=None,
                        resource=None,
                        is_hills=_terrain(x, y).endswith("HILLS"),
                        is_river=x == 9,
                        is_coastal=False,
                        improvement=None,
                        owner_id=self._owner(x, y),
                        visibility="visible",
                        units=units,
                    )
                )
        return out

    async def list_city_production(self, city_id: int) -> list[NS]:
        return [NS(category=c, item_name=n, cost=k, turns=5) for c, n, k in PRODUCTION]

    async def check_game_over(self) -> None:
        return None

    async def end_turn_blockers(self) -> list[tuple[str, str]]:
        out = []
        if await self.ready_unit_ids():
            out.append(("ENDTURN_BLOCKING_UNITS", "A unit needs orders"))
        if any(c.currently_building == "NONE" for c in self.cities.values()):
            out.append(("ENDTURN_BLOCKING_PRODUCTION", "Choose production"))
        if self.cities and self.research is None:
            out.append(("ENDTURN_BLOCKING_RESEARCH", "Choose research"))
        if self.cities and self.civic is None:
            out.append(("ENDTURN_BLOCKING_CIVIC", "Choose a civic"))
        if self.turn == 3:
            out.append(("ENDTURN_BLOCKING_PANTHEON", "Choose a pantheon"))
        return out

    async def ready_unit_ids(self) -> list[int]:
        return [
            uid for uid, u in self.units.items() if u.moves_remaining > 0 and uid not in self.orders
        ]

    # Actions ---------------------------------------------------------------------

    async def move_unit(self, unit_index: int, target_x: int, target_y: int) -> str:
        self.calls.append(("move_unit", unit_index, target_x, target_y))
        u = self._by_index(unit_index)
        if u is None:
            return "Error: no such unit"
        if u.moves_remaining <= 0:
            return "Error: NO_MOVES|Unit has no movement points remaining this turn."
        if "MOUNTAIN" in _terrain(target_x, target_y):
            return "Error: CANNOT_MOVE|Impassable"
        u.x, u.y = target_x, target_y
        u.moves_remaining -= 1
        return f"MOVING_TO|{target_x},{target_y}"

    async def attack_unit(self, unit_index: int, target_x: int, target_y: int) -> str:
        self.calls.append(("attack_unit", unit_index, target_x, target_y))
        u = self._by_index(unit_index)
        if u is None or (target_x, target_y) != self._barbarian_at:
            return f"Error: NO_ENEMY|No hostile unit at ({target_x},{target_y})"
        u.moves_remaining = 0
        u.health -= 20
        self._barbarian_at = None
        return "Attack resolved: barbarian warrior destroyed"

    async def found_city(self, unit_index: int) -> str:
        self.calls.append(("found_city", unit_index))
        u = self._by_index(unit_index)
        if u is None or u.unit_type != "UNIT_SETTLER":
            return "Error: CANNOT_FOUND|Only settlers can found cities"
        del self.units[u.unit_id]
        cid = 131072 + len(self.cities)
        self.cities[cid] = NS(
            city_id=cid,
            name="Paris",
            x=u.x,
            y=u.y,
            population=1,
            food=2.0,
            production=2.0,
            gold=5.0,
            science=1.5,
            culture=1.0,
            faith=0.0,
            housing=5.0,
            amenities=0,
            turns_to_grow=10,
            food_stored=0.0,
            currently_building="NONE",
            buildings=["BUILDING_PALACE"],
        )
        return f"FOUNDED|{u.x},{u.y}"

    async def fortify_unit(self, unit_index: int) -> str:
        self.calls.append(("fortify_unit", unit_index))
        u = self._by_index(unit_index)
        self.orders.add(u.unit_id)
        return "FORTIFIED"

    async def skip_unit(self, unit_index: int) -> str:
        self.calls.append(("skip_unit", unit_index))
        u = self._by_index(unit_index)
        self.orders.add(u.unit_id)
        return "SKIPPED"

    async def set_city_production(self, city_id: int, item_type: str, item_name: str) -> str:
        self.calls.append(("set_city_production", city_id, item_type, item_name))
        self.cities[city_id].currently_building = item_name
        return f"PRODUCING|{item_name}"

    async def set_research(self, tech_name: str) -> str:
        self.calls.append(("set_research", tech_name))
        if tech_name not in {t for _, t, _ in TECHS}:
            return f"Error: unknown tech {tech_name}"
        self.research = tech_name
        return f"RESEARCHING|{tech_name}"

    async def set_civic(self, civic_name: str) -> str:
        self.calls.append(("set_civic", civic_name))
        self.civic = civic_name
        return f"PROGRESSING|{civic_name}"

    async def end_turn(self) -> str:
        self.calls.append(("end_turn",))
        hard = [b for b in await self.end_turn_blockers() if b[0] != "ENDTURN_BLOCKING_PANTHEON"]
        if hard:
            return "Error: cannot end turn: " + "; ".join(m for _, m in hard)
        self.turn += 1
        self.orders.clear()
        for u in self.units.values():
            u.moves_remaining = u.max_moves
        return f"Turn {self.turn - 1} ended. Now turn {self.turn}."

    # Helpers ---------------------------------------------------------------------

    def _by_index(self, index: int) -> NS | None:
        return next((u for u in self.units.values() if u.unit_index == index), None)

    def _owner(self, x: int, y: int) -> int:
        for c in self.cities.values():
            if abs(c.x - x) <= 1 and abs(c.y - y) <= 1:
                return PLAYER
        return -1

    @staticmethod
    def _name(type_name: str | None, rows) -> str | None:
        return next((n for n, t, _ in rows if t == type_name), None)
