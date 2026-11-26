"""Real Civ VI through civ6-mcp.

civ6-mcp drives the game over FireTuner and reads state into dataclasses before it
narrates them. This package reads those dataclasses directly and maps them onto
Napoleon's schema, so nothing parses narrated text.

``Civ6Backend`` works with any object that has civ6-mcp ``GameState``'s async methods
(see ``Civ6Source``). Tests use a recorded fake; ``napoleon.civ6.live`` builds the real
one and needs civ6-mcp installed and Civ VI running.
"""

from napoleon.civ6.backend import Civ6Backend
from napoleon.civ6.source import Civ6Source

__all__ = ["Civ6Backend", "Civ6Source"]
