# Napoleon

An open testbed for one question: what does an agent's architecture buy you on a long-horizon 4X strategy game?

Napoleon wraps Civilization VI (through [civ6-mcp](https://github.com/lmwilki/civ6-mcp)) in a thin `reset` / `observe` / `step` interface and pits the same lineup of players against it: a stateless direct policy, then ReAct, Reflexion, and a Voyager-style agent with a growing skill library. Every game is recorded step by step with the agent's reasoning, and scored on victory progress, score over turns, and recovery from bad openings.

Status: early. The game-independent parts (schema, adapter, fake game, recording, metrics, sanity agents) come first.

## Development

```
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```
