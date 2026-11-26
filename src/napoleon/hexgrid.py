"""Axial hex-grid helpers."""

from __future__ import annotations

from collections.abc import Iterator

from napoleon.schema import Coord

DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def neighbors(c: Coord) -> list[Coord]:
    return [Coord(q=c.q + dq, r=c.r + dr) for dq, dr in DIRECTIONS]


def distance(a: Coord, b: Coord) -> int:
    dq, dr = a.q - b.q, a.r - b.r
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def within(center: Coord, radius: int) -> Iterator[Coord]:
    """Every coordinate at most ``radius`` steps from ``center``, center included."""
    for dq in range(-radius, radius + 1):
        for dr in range(max(-radius, -dq - radius), min(radius, -dq + radius) + 1):
            yield Coord(q=center.q + dq, r=center.r + dr)
