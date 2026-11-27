# Napoleon

An open testbed for one question: what does an agent's architecture buy you on a long-horizon 4X strategy game?

Napoleon wraps Civilization VI (through [civ6-mcp](https://github.com/lmwilki/civ6-mcp)) in a thin `reset` / `observe` / `step` interface and pits the same lineup of players against it: a stateless direct policy, then ReAct, Reflexion, and a Voyager-style agent with a growing skill library. Every game is recorded step by step with the agent's reasoning, and scored on victory progress, score over turns, and recovery from bad openings.

Status: early. Everything runs on a small deterministic fake game. The civ6-mcp backend is written and tested against a scripted stand-in, and waits on its first session with the game open ([setup and checks](docs/civ6-live-setup.md)).

## Layout

- `napoleon.schema`: the turn snapshot and the six typed actions, versioned. This is the contract every backend emits.
- `napoleon.env.CivEnv`: `reset(scenario, seed)`, `observe()`, `step(action)` over any `GameBackend`. Illegal or malformed actions come back as errors and never change the game.
- `napoleon.fake.FakeCivGame`: a fogged hex map, settlers, warriors, cities, production, tech and civic trees, turn blockers, barbarians, simple combat and an expanding rival. Scenarios: `default` and `bad_opening`.
- `napoleon.recording`: JSONL run files with every step, the agent's reasoning, and a full snapshot per turn.
- `napoleon.metrics`: score, cities, population and techs over turns, illegal-action rate, and recovery from a bad opening.
- `napoleon.agents`: `random` and `greedy` sanity agents, and the LLM agents: `direct`, a stateless policy that sees one snapshot and answers with one JSON action.
- `napoleon.llm`: a one-method `LLMClient` protocol, an OpenAI-compatible HTTP client (Ollama, LM Studio, vLLM, llama.cpp, hosted Qwen), and a scripted client for tests. Agents ask for JSON and parse it leniently rather than rely on native tool calling.
- `napoleon.usage`: token and cost accounting. Each step in a run file carries its usage, and metrics sum it.
- `napoleon.replay`: compact replay JSON and a self-contained HTML page per run ([format](docs/replay-format.md)).
- `napoleon.civ6`: `Civ6Backend` over civ6-mcp's game state, a live connection, and a scripted stand-in for tests.

## Try it

```
napoleon run --agent greedy --turns 100
napoleon run --agent random --scenario bad_opening --seed 3
napoleon metrics runs/fake-greedy-default-0.jsonl
napoleon replay runs/fake-greedy-default-0.jsonl   # writes .replay.json and .replay.html
napoleon run --agent direct --turns 20             # Ollama on localhost, model qwen3:8b
napoleon run --agent direct --model qwen2.5:7b-instruct --base-url http://localhost:1234/v1
napoleon run --backend civ6-scripted --turns 15
napoleon run --backend civ6 --save default=napoleon_default   # needs Civ VI open
```

LLM agents talk to any OpenAI-compatible endpoint. Defaults are Ollama at
`http://localhost:11434/v1` and `qwen3:8b`; override with `--model` and `--base-url`, or the
`NAPOLEON_LLM_MODEL` and `NAPOLEON_LLM_BASE_URL` environment variables. If the endpoint needs a
key, put it in `NAPOLEON_LLM_API_KEY` (or name another variable with `--api-key-env`). Hosted
models cost money: pass `--price-in` and `--price-out` (USD per million tokens) to have cost
tracked; local models report tokens at zero cost. `--no-think` appends `/no_think` for Qwen3.
Unusable output is retried (`--max-retries`, default 2), then sent to the game as a malformed
action, which counts in the illegal-action rate.

## Development

```
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```
