from napoleon.hexgrid import distance, neighbors, within
from napoleon.schema import Coord

ORIGIN = Coord(q=0, r=0)


def test_neighbors_are_distance_one():
    assert all(distance(ORIGIN, n) == 1 for n in neighbors(ORIGIN))
    assert len(set(neighbors(ORIGIN))) == 6


def test_within_counts_hexes():
    # A hex of radius r holds 3r(r+1)+1 cells.
    for r in range(4):
        cells = list(within(ORIGIN, r))
        assert len(cells) == 3 * r * (r + 1) + 1
        assert all(distance(ORIGIN, c) <= r for c in cells)
