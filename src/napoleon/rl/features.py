"""Hand features of (snapshot, action) pairs, so one policy scores a variable legal-action list.

Each legal action becomes a row ``phi``: features of the action itself, a global context
vector ``g``, and the action type crossed with ``g`` (so even a linear scorer can say "found
a city when there are no cities yet"). There is no fixed action vocabulary and masking is
free: only legal actions get rows, and the policy is a softmax over those rows.
"""

from __future__ import annotations

import numpy as np

from napoleon.fake.game import CIVICS, PRODUCTION, TECHS
from napoleon.hexgrid import distance
from napoleon.schema import MoveUnit, SetCivic, SetProduction, SetResearch, Snapshot, UnitCommand

VERSION = 1
TYPES = ("move_unit", "unit_command", "set_production", "set_research", "set_civic", "end_turn")
COMMANDS = ("fortify", "skip", "found_city")
NAMES = (*PRODUCTION, *TECHS, *CIVICS)
GOOD_TERRAIN = {"grassland", "plains", "hills"}
BAD_SITES = {"ocean", "mountains"}
MIN_CITY_DISTANCE = 3

N_ACTION = len(TYPES) + len(COMMANDS) + 2 + 8 + len(NAMES)
N_GLOBAL = 8
DIM = N_ACTION + N_GLOBAL + len(TYPES) * N_GLOBAL
STATE_DIM = N_GLOBAL + len(TYPES) + 2


def _delta(before: int, after: int) -> float:
    return max(-1.0, min(1.0, (before - after) / 3))


def _nearest(coord, others) -> int | None:
    return min((distance(coord, o) for o in others), default=None)


def _sites(snap: Snapshot, mine_cities) -> set:
    foreign = [t.coord for t in snap.tiles if t.owner not in (None, snap.player.id)]
    return {
        t.coord
        for t in snap.tiles
        if t.terrain not in BAD_SITES
        and t.owner is None
        and all(distance(t.coord, c) >= MIN_CITY_DISTANCE for c in mine_cities)
        and all(distance(t.coord, f) >= 4 for f in foreign)
    }


def global_features(snap: Snapshot) -> np.ndarray:
    me = snap.player.id
    mine = [u for u in snap.units if u.owner == me]
    city_at = [c.coord for c in snap.cities if c.owner == me]
    enemies = [u.coord for u in snap.units if u.owner != me]
    near = any((_nearest(e, city_at + [u.coord for u in mine]) or 99) <= 4 for e in enemies)
    return np.array(
        [
            snap.turn / max(1, snap.max_turns),
            min(len(city_at), 6) / 4,
            min(len(mine), 8) / 6,
            min(snap.player.gold, 200) / 100,
            float(near),
            min(sum(b.kind != "other" for b in snap.blockers), 8) / 5,
            min(snap.player.score, 200) / 100,
            min(sum(u.type == "settler" for u in mine), 3) / 2,
        ]
    )


def step_features(snap: Snapshot) -> tuple[np.ndarray, np.ndarray]:
    """``(phi, s)``: a row per legal action, and a state vector for the value baseline."""
    me = snap.player.id
    g = global_features(snap)
    units = {u.id: u for u in snap.units if u.owner == me}
    cities = [c.coord for c in snap.cities if c.owner == me]
    enemies = [u.coord for u in snap.units if u.owner != me]
    enemy_set = set(enemies)
    tiles = {t.coord: t.terrain for t in snap.tiles}
    sites = _sites(snap, cities) if any(u.type == "settler" for u in units.values()) else set()
    name_ix = {n: i for i, n in enumerate(NAMES)}

    rows = np.zeros((len(snap.legal_actions), N_ACTION))
    type_ix = np.zeros(len(snap.legal_actions), dtype=int)
    base_cmd = len(TYPES)
    base_flags = base_cmd + len(COMMANDS)
    base_move = base_flags + 2
    base_name = base_move + 8
    for i, a in enumerate(snap.legal_actions):
        row = rows[i]
        type_ix[i] = TYPES.index(a.type)
        row[type_ix[i]] = 1.0
        if isinstance(a, UnitCommand | MoveUnit):
            u = units.get(a.unit_id)
            if u is not None:
                row[base_flags] = u.type == "settler"
                row[base_flags + 1] = u.type == "warrior"
            if isinstance(a, UnitCommand):
                row[base_cmd + COMMANDS.index(a.command)] = 1.0
            elif u is not None:
                to = a.to
                row[base_move] = to in enemy_set
                if enemies:
                    row[base_move + 1] = _delta(_nearest(u.coord, enemies), _nearest(to, enemies))
                if cities:
                    row[base_move + 2] = _delta(_nearest(u.coord, cities), _nearest(to, cities))
                if sites and u.type == "settler":
                    row[base_move + 3] = _delta(_nearest(u.coord, sites), _nearest(to, sites))
                row[base_move + 4] = to not in tiles
                row[base_move + 5] = tiles.get(to) in GOOD_TERRAIN
                row[base_move + 6] = to in sites
                row[base_move + 7] = u.hp / 100
        elif isinstance(a, SetProduction):
            row[base_name + name_ix[a.item]] = 1.0
        elif isinstance(a, SetResearch):
            row[base_name + name_ix[a.tech]] = 1.0
        elif isinstance(a, SetCivic):
            row[base_name + name_ix[a.civic]] = 1.0
    cross = (np.eye(len(TYPES))[type_ix][:, :, None] * g[None, None, :]).reshape(len(rows), -1)
    phi = np.concatenate([rows, np.broadcast_to(g, (len(rows), N_GLOBAL)), cross], axis=1)
    counts = np.bincount(type_ix, minlength=len(TYPES)) / max(1, len(rows))
    state = np.concatenate([g, counts, [min(len(rows), 60) / 60, 1.0]])
    return phi, state
