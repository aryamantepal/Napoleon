# Napoleon

An open testbed for one question: what does an agent's architecture buy you on a long-horizon 4X strategy game?

Napoleon wraps Civilization VI (through [civ6-mcp](https://github.com/lmwilki/civ6-mcp)) in a thin `reset` / `observe` / `step` interface and pits the same lineup of players against it: a stateless direct policy, then ReAct, Reflexion, and a Voyager-style agent with a growing skill library. Every game is recorded step by step with the agent's reasoning, and scored on victory progress, score over turns, and recovery from bad openings.

Status: early. Everything below runs without Civ VI, on a small deterministic fake game. The civ6-mcp backend comes next.

## Layout

- `napoleon.schema`: the turn snapshot and the six typed actions, versioned. This is the contract every backend emits.
- `napoleon.env.CivEnv`: `reset(scenario, seed)`, `observe()`, `step(action)` over any `GameBackend`. Illegal or malformed actions come back as errors and never change the game.
- `napoleon.fake.FakeCivGame`: a fogged hex map, settlers, cities, production, tech and civic trees, turn blockers and a rival. Scenarios: `default` and `bad_opening`.
- `napoleon.recording`: JSONL run files with every step, the agent's reasoning, and a full snapshot per turn.
- `napoleon.metrics`: score, cities, population and techs over turns, illegal-action rate, and recovery from a bad opening.
- `napoleon.agents`: `random` and `greedy` sanity agents.

## Try it

```
napoleon run --agent greedy --turns 100
napoleon run --agent random --scenario bad_opening --seed 3
napoleon metrics runs/greedy-default-0.jsonl
```

## Development

```
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```
