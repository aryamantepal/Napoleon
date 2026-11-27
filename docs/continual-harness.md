# Continual harness

`--agent continual` ports the Continual Harness idea (Karten et al. 2026, arXiv 2605.09998): an
agent whose harness improves while it plays, in one episode, with no resets. The model
that acts also edits its own scaffolding on a schedule, from text alone.

## The harness H = (p, G, K, M)

| Part | What it is | Code |
|---|---|---|
| p | Strategy prompt, markdown, editable. The fixed system prompt (rules, action schema, tool and output formats) is separate and the refiner can read but not edit it. | `harness.Harness.prompt`, `agents/continual.FIXED_SYSTEM` |
| K | Skills. Entries have `id, name, description, effectiveness, body`. The body is a skill in the data-only DSL of `napoleon.skills` (Voyager's), never code. The agent runs one with `run_skill(name, args)`. | `harness.SkillEntry` |
| M | Memory entries `id, title, content, importance 1-5`. Every acting prompt lists ids and titles; `read_memory` and `write_memory` open and write entries. | `harness.MemoryEntry` |
| G | Sub-agents `id, name, description, directive, allowed_tools, max_turns`. The orchestrator calls `delegate(name, task)`. A sub-agent runs one or several model calls (`max_turns`, at most 10) and returns an action, which ends the step, or a note, which goes back to the orchestrator. | `harness.SubAgent` |

## The refiner

`refiner.Refiner` is the same model, text only. It runs at step 25, then every 25 steps up to
step 200, then every 100 (`--refine-warmup`, `--refine-every`, `--refine-until`,
`--refine-late-every`). Its input is the last 50 steps (state summary, reasoning, action,
result, each clipped to 300 characters), a tool-failure section, and stall flags: the turn not
advancing, the same action repeated, a streak of illegal actions. It then runs four independent
passes, each in its own try/except: prompt (a full markdown rewrite), sub-agents, skills and
memory (JSON `{analysis, add/create, update, retire}`, parsed leniently: aliases, fences,
single objects instead of lists). Refinement calls are charged to the step that triggered
them, so cost and calls-per-turn include them.

## Guardrails the paper lacks

- The strategy prompt is capped (`--max-prompt-chars`, 4000). An over-long rewrite is rejected, not truncated.
- A new skill must parse as DSL and pass fake-game verification (`skills.verify_skill`: it must fire, never be rejected, never stall, and stay within 75% of greedy's score). A skill with parameters is verified with `example_args`; without them only the schema is checked.
- Sub-agent tools are checked against a whitelist (the read-only lookups plus `read_memory` and `write_memory`) when saved and again when run. `max_turns` is clamped to 10. Sub-agents cannot delegate or start skills.
- Memory, skills and sub-agents have size limits, and memory evicts its least important entry.
- Before each refinement H is snapshotted. At the next refinement the steps since are compared with the steps before. If the illegal-action rate rose by more than `--rollback-illegal-delta` (0.15), or the score gain per turn fell by more than `--rollback-score-margin` (50%, only when it was positive), H is restored and a `harness_rollback` event is logged. `--no-rollback` turns this off. Memory the agent wrote itself between the snapshot and a rollback is lost with the rollback.

## Output

Next to the run file `X.jsonl` the agent writes `X.harness/`:

- `harness.json`: the latest H and its generation.
- `prompt_evolution/gen-NNN.md`: the strategy at each generation.
- `evolution_log.jsonl`: one line per refinement or rollback, with each pass's outcome.

The same information is in the run file as `agent_events` on the step that triggered it (`harness_refined`, `harness_rollback`), and `napoleon replay` shows them in the step list. The metrics gain `actions_per_turn` and `calls_per_turn`, and `sweep` reports both.

## Bootstrap modes

- From scratch: `napoleon run --agent continual`.
- Frozen: `--bootstrap-from runs/x.harness` plays with that harness and never refines.
- Bootstrap plus updating: add `--bootstrap-update`.

## Ported, dropped, and how it differs

Ported: the four-part harness, the scheduled in-episode refiner with its input and four passes, the three bootstrap modes, the orchestrator with sub-agents, skills and memory.

Dropped: vision (the game is text), and PRM co-learning, where the paper also trains the model on the harness's trajectories. Fine-tuning the model is a separate item here.

Versus the other agents:

- Reflexion learns between episodes: a reflection is written after a game and read in the next. The continual harness changes mid-game and edits more than a prompt.
- Voyager learns skills only, and only when the model defines one itself. Here skills are one of four parts and a separate pass curates them from evidence, with retirement.
- Both are reset-per-episode. Continual's harness lives for the whole game, and persists to disk so a later game can start from it.
