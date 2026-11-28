# Prompting conditions: zero-shot vs few-shot, with and without reasoning

Every LLM agent (`direct`, `react`, `reflexion`, `voyager`, `continual`) takes three options:

- `--shots K` (default 0): K demonstrations before the real prompt.
- `--cot` / `--no-cot` (default `--cot`): `--cot` asks for a short reasoning field before the action, as the agents always did. `--no-cot` asks for the action alone. It is unrelated to `--no-think`, which only appends Qwen3's `/no_think` switch.
- `--demos FILE`: a curated demonstration file instead of the built-in one.

The condition (`shots`, `cot`, `demos`, `demo_pool`) is stored in the run header's `config`. `napoleon sweep --shots 0 3` runs both, writing `<scenario>-seed<N>-shots3.jsonl` for the few-shot cells (zero-shot file names are unchanged, so old sweeps still resume). Non-LLM agents run once. `sweep` and `report` group by agent, shots, cot and scenario, and `runs.csv` and `summary.csv` have `shots` and `cot` columns.

## Where demonstrations come from

By default the greedy agent is rolled out on the fake game on seeds 1000 to 1005 (both scenarios, 40 turns), and every step it took is a candidate, with end-of-turn steps thinned to one per game. Greedy's one-line reason is the demonstration's reasoning. `napoleon demos --out demos.jsonl` writes the same pool to a file, `napoleon demos --from runs/greedy.jsonl --out demos.jsonl` builds it from a finished fake-game run by replaying it (the replay must reproduce every recorded snapshot hash), and the file can be edited by hand.

## Why similarity and not a fixed set

The right action depends on the situation: an idle city, a settler on a founding site, a barbarian beside a city. Three fixed examples would mostly show unrelated situations and mostly cost tokens. So the K demonstrations are the pool entries most similar to the current snapshot, by Jaccard overlap of discrete features (blocker kinds, unit types, legal action types, whether a city can be founded, city count, enemy nearby). It is deterministic and needs no embedding model. Picks are diversified so the K examples have different action types where possible. The most similar example comes last, closest to the real prompt.

## Keeping demonstrations out of the evaluated game

`agent.reset(seed)` drops every demonstration recorded on that seed, so an agent never sees greedy's play on the game it is about to play. The default pool uses seeds 1000 and up, away from the 0 to a few evaluation seeds used in this repo.

## What a demonstration looks like

A user message with a compact summary (turn, score, cities, units, blockers, research, the legal actions by count, not the full list) followed by an assistant message with the JSON the agent should produce. The summary is shorter than the real prompt, which also lists tiles and every legal action, to keep K examples cheap. That is a deliberate mismatch: the examples teach what to choose, and the real prompt still carries the legal list.
