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
- `napoleon.agents`: `random` and `greedy` sanity agents, and the LLM agents. `direct` is a stateless policy: one snapshot in, one JSON action out. `react` first looks things up with read-only tools (`inspect_tile`, `inspect_unit`, `inspect_city`, `list_options`, `find_city_sites`, `threats`), up to a bounded number of lookups, then commits one action, and keeps a short log of its own recent actions. Its whole thought and lookup trace is stored as the step's reasoning. `reflexion` wraps either one (`--base direct|react`): after each game the model reads the run's metrics and a digest of its trajectory, writes a short reflection, and appends it to a JSONL memory file (`runs/memory/reflexion.jsonl`, or `--memory`). The latest few reflections go into the system prompt of every later game, in the same process or a later one. The reflection call's cost is recorded in the run's footer and counted in its metrics. `voyager` keeps a growing skill library: the model can play a primitive action, call a stored skill, or define a new one. A skill is data in a small DSL over the six actions, never code (`napoleon.skills`): ordered rules that each match a legal action, with parameters, conditions and a few ways to choose among moves. New skills are played in fake-game rollouts and stored only if they pass, retrieved by description, and once running they take their following steps without a model call. The library is a JSON file (`runs/skills/library.json`, or `--skills`), so it carries across games.
- `continual` (`napoleon.harness`, `napoleon.refiner`, `napoleon.agents.continual`): the Continual Harness. The agent's strategy prompt, skills, memory and sub-agents are edited by the same model on a schedule while the game runs, with size caps, skill verification, a tool whitelist and rollback ([details](docs/continual-harness.md)).
- `napoleon.llm`: a one-method `LLMClient` protocol, an OpenAI-compatible HTTP client (Ollama, LM Studio, vLLM, llama.cpp, hosted Qwen), and a scripted client for tests. Agents ask for JSON and parse it leniently rather than rely on native tool calling.
- `napoleon.skills`: the skill DSL, its safe interpreter, the JSON skill library with TF-IDF retrieval, and fake-game verification.
- `napoleon.runner`: `run_game` for one game and `run_episodes` for several in a row with the same agent object, so whatever the agent keeps carries over.
- `napoleon.sweep`: runs an agents x scenarios x seeds matrix on the fake game (optionally in parallel threads, resumable) and aggregates run files into `summary.csv`, `runs.csv` and `summary.md`: score, size, illegal-action rate, steps, model calls, tokens and cost, and recovery from the bad opening against the same agent's normal games.
- `napoleon.usage`: token and cost accounting. Each step in a run file carries its usage, and metrics sum it.
- `napoleon.replay`: compact replay JSON and a self-contained HTML page per run: hex map with units and cities, click a step to see the map after it, per-step reasoning with model usage and cost, and a chart of score (with the rival), cities, population, techs, cost and calls ([format](docs/replay-format.md)). `napoleon compare` overlays several runs on one page.
- `napoleon.civ6`: `Civ6Backend` over civ6-mcp's game state, a live connection, and a scripted stand-in for tests.

## Try it

```
napoleon run --agent greedy --turns 100
napoleon run --agent random --scenario bad_opening --seed 3
napoleon metrics runs/fake-greedy-default-0.jsonl
napoleon replay runs/fake-greedy-default-0.jsonl   # writes .replay.json and .replay.html
napoleon compare runs/a.jsonl runs/b.jsonl --html compare.html
napoleon run --agent direct --turns 20             # Ollama on localhost, model qwen3:8b
napoleon run --agent direct --model qwen2.5:7b-instruct --base-url http://localhost:1234/v1
napoleon run --agent react --turns 20              # same flags as direct
napoleon run --agent reflexion --base react --episodes 3 --turns 30   # 3 games, learning between them
napoleon run --agent voyager --episodes 3 --turns 30 --skills runs/skills/a.json --clear-skills
napoleon run --agent reflexion --episodes 3 --vary-seed --clear-memory --memory runs/memory/a.jsonl
napoleon run --agent continual --turns 30          # harness in runs/<run>.harness/
napoleon run --agent continual --bootstrap-from runs/x.harness   # frozen; add --bootstrap-update to keep refining
napoleon sweep --agents random greedy direct react --seeds 0-4 --turns 50 --workers 4 --out runs/sweep
napoleon report runs/sweep                          # re-aggregate existing run files
napoleon run --backend civ6-scripted --turns 15
napoleon run --backend civ6 --save default=napoleon_default   # needs Civ VI open
```

LLM agents talk to any OpenAI-compatible endpoint. Defaults are Ollama at
`http://localhost:11434/v1` and `qwen3:8b`; override with `--model` and `--base-url`, or the
`NAPOLEON_LLM_MODEL` and `NAPOLEON_LLM_BASE_URL` environment variables. If the endpoint needs a
key, put it in `NAPOLEON_LLM_API_KEY` (or name another variable with `--api-key-env`). Hosted
models cost money: pass `--price-in` and `--price-out` (USD per million tokens) to have cost
tracked; local models report tokens at zero cost. `--no-think` turns off Qwen3 thinking: it sends `reasoning_effort: "none"` (what Ollama honors) and appends `/no_think`. If a reply has no content but a reasoning field, the agent parses the action out of that.
Ollama's default context window is 4096 tokens, which truncates the prompt from the front once a game has many units and legal actions; start the server with a larger one (for example `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`). Unusable output is retried (`--max-retries`, default 2), then sent to the game as a malformed
action, which counts in the illegal-action rate.

## Development

```
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```
