"""LiveSource's wrapping of civ6-mcp's GameState, against a stand-in GameState."""

import asyncio
from types import SimpleNamespace

from napoleon.civ6.live import LiveSource
from napoleon.civ6.source import is_error


class FakeConn:
    def __init__(self) -> None:
        self.reconnects = 0

    async def reconnect(self) -> None:
        self.reconnects += 1


class FakeGameState:
    """Answers reads only after ``loading_polls`` reconnects, like a reloading game."""

    def __init__(self, load_result: str = "Loading save: s.", loading_polls: int = 2) -> None:
        self.conn = FakeConn()
        self.load_result = load_result
        self.loading_polls = loading_polls
        self._save_load_history = [(0.0, 40, "s"), (1.0, 80, "s")]
        self._run_aborted = True
        self._last_snapshot = object()

    async def load_game_save(self, save_name: str) -> str:
        return self.load_result

    async def get_game_overview(self):
        if self.conn.reconnects <= self.loading_polls:
            raise ConnectionError("GameCore_Tuner/InGame states not found")
        return SimpleNamespace(turn=1)


def load(gs: FakeGameState, timeout: float = 1.0) -> str:
    src = LiveSource(gs, lua=None, load_timeout=timeout, poll_every=0.001)
    return asyncio.run(src.load_game_save("s"))


def test_load_waits_for_the_game_and_forgets_earlier_loads():
    gs = FakeGameState()
    result = load(gs)
    assert not is_error(result)
    assert gs.conn.reconnects == 3
    assert gs._save_load_history == [] and gs._run_aborted is False
    assert gs._last_snapshot is None


def test_load_that_never_finishes_is_an_error():
    result = load(FakeGameState(loading_polls=10**9), timeout=0.02)
    assert is_error(result) and "did not finish loading" in result


def test_refused_load_is_passed_through_without_waiting():
    gs = FakeGameState(load_result="Error: Save 's' not found")
    assert is_error(load(gs)) and gs.conn.reconnects == 0


def test_end_turn_refusals_count_as_errors():
    assert is_error("End turn blocked (turn 12): Choose production")
    assert is_error("End turn requested (turn is still 12). Check get_pending_diplomacy")
    assert is_error("RUN ABORTED — save scumming threshold exceeded.")
    assert not is_error("Turn 12 ended. Now turn 13.")
