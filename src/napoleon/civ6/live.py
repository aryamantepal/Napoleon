"""Connect ``Civ6Backend`` to a running Civ VI through civ6-mcp.

Needs civ6-mcp importable (``pip install -e path/to/civ6-mcp``) and Civ VI running with
FireTuner enabled and a game in progress. See ``docs/civ6-live-setup.md``.

Nothing here runs in CI. The Lua in ``READY_UNITS_LUA`` is unverified until it has run
against the game once; the setup notes list it as a first-session check.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from napoleon.civ6.backend import Civ6Backend

READY_UNITS_LUA = """
local me = Game.GetLocalPlayer()
for _, unit in Players[me]:GetUnits():Members() do
    if unit:GetX() ~= -9999 and unit:GetMovesRemaining() > 0 then
        local ok, activity = pcall(function() return UnitManager.GetActivityType(unit) end)
        if not ok or activity == ActivityTypes.ACTIVITY_AWAKE then
            print("READY|" .. unit:GetID())
        end
    end
end
print("{sentinel}")
"""


class LiveSource:
    """civ6-mcp's ``GameState`` plus the two reads ``Civ6Source`` adds."""

    def __init__(
        self,
        game_state: Any,
        lua: Any,
        *,
        load_timeout: float = 120.0,
        poll_every: float = 2.0,
    ) -> None:
        self._gs = game_state
        self._lq = lua
        self.load_timeout = load_timeout
        self.poll_every = poll_every

    async def load_game_save(self, save_name: str) -> str:
        """Load a save and return once the reloaded game answers reads.

        civ6-mcp's ``load_game_save`` returns as soon as the load starts. The reload
        tears down the Lua states, so this reconnects and polls until the game reads
        back. It also forgets earlier loads: civ6-mcp counts loads across different
        turns as save scumming and aborts the run, but every Napoleon episode starts
        with one.
        """
        result = await self._gs.load_game_save(save_name)
        if result.lstrip().startswith(("Error", "ERR", "FAILED")):
            return result
        self._gs._save_load_history.clear()
        self._gs._run_aborted = False
        self._gs._last_snapshot = None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.load_timeout
        last_error: Exception | None = None
        while loop.time() < deadline:
            await asyncio.sleep(self.poll_every)
            try:
                await self._gs.conn.reconnect()
                await self._gs.get_game_overview()
                return result
            except Exception as exc:  # the game is still loading
                last_error = exc
        return f"Error: {save_name!r} did not finish loading: {last_error}"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gs, name)

    async def end_turn_blockers(self) -> list[tuple[str, str]]:
        lines = await self._gs.conn.execute_write(self._lq.build_end_turn_blocking_query())
        return self._lq.parse_end_turn_blocking(lines)

    async def ready_unit_ids(self) -> list[int]:
        lines = await self._gs.conn.execute_write(
            READY_UNITS_LUA.replace("{sentinel}", self._lq.SENTINEL)
        )
        return [int(line.split("|")[1]) for line in lines if line.startswith("READY|")]


def connect(
    saves: Mapping[str, str],
    *,
    host: str = "127.0.0.1",
    port: int = 4318,
    max_turns: int | None = None,
) -> Civ6Backend:
    """Open a FireTuner connection and return a ready-to-reset backend."""
    try:
        from civ_mcp import lua as lq
        from civ_mcp.connection import GameConnection
        from civ_mcp.game_state import GameState
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError("civ6-mcp is not installed; see docs/civ6-live-setup.md") from exc

    loop = asyncio.new_event_loop()
    conn = GameConnection(host=host, port=port)
    loop.run_until_complete(conn.connect())
    source = LiveSource(GameState(conn), lq)
    return Civ6Backend(source, saves, max_turns=max_turns, loop=loop)
