"""Civ VI plot coordinates to Napoleon's axial hex coordinates.

Civ VI addresses plots as offset ``(x, y)`` with odd rows shifted half a hex right
("odd-r"). The conversion below assumes that layout; ``docs/civ6-live-setup.md`` has a
check against ``Map.GetPlotDistance`` to run once with the game open.
"""

from __future__ import annotations

from napoleon.schema import Coord


def offset_to_axial(x: int, y: int) -> Coord:
    return Coord(q=x - (y - (y & 1)) // 2, r=y)


def axial_to_offset(c: Coord) -> tuple[int, int]:
    return c.q + (c.r - (c.r & 1)) // 2, c.r
