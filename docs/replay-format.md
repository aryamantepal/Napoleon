# Replay format

`napoleon replay RUN.jsonl` writes `RUN.replay.json` and a self-contained
`RUN.replay.html`. The JSON is what a viewer reads; the HTML page is the reference for
the Civ panel in Porygon's viewer.

## Top level

| Key | What it holds |
|---|---|
| `format`, `version` | `"napoleon-replay"`, `1`. Bump `version` on any breaking change. |
| `run` | `run_id`, `backend`, `scenario`, `seed`, `agent`, `started_at`, `schema_version`, `ended` (`done`, `step_limit` or `null` for a run cut short), `config` (what the agent reported about itself, such as `model`). |
| `totals` | `steps`, `failed_steps`, `illegal_action_rate`, `llm_calls`, `input_tokens`, `output_tokens`, `cost_usd`, `final_score`. The LLM fields are zero for the sanity agents. |
| `frame_fields` | Column names for the arrays in each frame, so readers don't hard-code positions. |
| `frames` | One per turn, in turn order. |
| `series` | `score`, `cities`, `population`, `techs`, `rival` (the best contact score seen that turn; empty until contact), `cost` (cumulative USD) and `calls` (cumulative model calls): lists of `[turn, value]`. |
| `steps` | Every step, in order. |

## Frames

A frame is the player's snapshot at the start of a turn (the last frame is the end of
the game).

```json
{
  "turn": 12, "player": 0, "score": 21, "gold": 34.0,
  "tiles":  [[q, r, "plains_hills", owner_or_null, 1]],
  "units":  [[id, owner, "warrior", q, r, hp]],
  "cities": [[id, owner, "Lutetia", q, r, population, "monument"]],
  "tech": "pottery", "civic": "code_of_laws",
  "victory": {"science": 0.33},
  "contacts": [[1, "Carthage", 0, 14]]
}
```

- `contacts` rows are `[player_id, civ, at_war, score]`.
- Coordinates are axial `(q, r)`. For pointy-top hexes of size `s`:
  `x = s * sqrt(3) * (q + r / 2)`, `y = s * 1.5 * r`.
- Tiles are only those the player has revealed. The last column is `1` when the tile is
  in sight this turn and `0` when it is remembered under fog.
- `units` holds the player's own units and any other unit in sight. Real Civ VI gives
  foreign units no id, so theirs are negative; barbarians have owner `63` on Civ VI and
  `2` on the fake game.

## Steps

```json
{"step": 40, "turn": 12, "action": {"type": "move_unit", "unit_id": 3, "to": {"q": 1, "r": -2}},
 "reasoning": "warrior 3 closes on a threat near our cities",
 "ok": true, "error": null, "events": ["warrior 3 moved to (1, -2)"],
 "usage": {"calls": 2, "in": 1469, "out": 35, "cost": 0.0033}, "ms": 4210}
```

`action` follows `napoleon.schema` (six types). When the agent's output didn't parse,
`action` holds the raw output instead and `ok` is `false`. `usage` is `null` when the step
used no model call: always for the sanity agents, and for the steps a Voyager skill plays
on its own. `reasoning` can be several lines; ReAct stores its whole thought and lookup
trace there. Everything added in this section is additive, so `version` stays `1`.

## Porting the panel into the viewer

The HTML page shows what the panel needs; port behaviour, not the code.

1. **Loading.** Accept a `.replay.json` (drop or URL) and check `format` and
   `version`. Don't read raw run files in the browser: they're larger and carry whole
   snapshots.
2. **Turn control.** Slider over `frames`, previous and next, and play at about two
   turns a second. Left and right arrows step turns, space plays.
3. **Map.** Hexes from `tiles`, colored by terrain with fogged tiles faded and owned
   tiles outlined in the owner's color. Cities as squares labeled with population,
   units as dots with the type's first letter. Hover shows the tile or unit.
4. **Steps.** For the selected turn: each step's action in words, its reasoning
   (multi-line, long ones collapsed), any error in red, its events, and a usage badge
   (calls, tokens, cost) or "no model call". Clicking a step shows the map after it:
   a frame is only the start of a turn, so the page replays the turn's accepted moves,
   city foundings and production changes over the frame, and rings the unit or city the
   step touched, with a line to the destination of a move.
5. **Chart.** Tabs for score (with the rival's score dashed), cities, population, techs,
   and, when there were model calls, cumulative cost and calls. A marker follows the
   selected turn.
6. **Header.** Agent, backend, scenario, seed, how the run ended, and chips for model,
   final score, illegal-action rate, model calls, tokens and cost.
7. **Links.** The page keeps `#turn=12&step=2` in its URL, so a moment can be shared.

## Comparing runs

`napoleon compare RUN.jsonl RUN2.jsonl ... --html compare.html` writes one page with the
runs' series overlaid (a tab per series) and a table of final score, cities, techs,
illegal-action rate, steps, model calls, tokens, cost and how each run ended. Each run is
labeled with its agent, model, scenario and seed, so runs of the same scenario and seed
with different agents read as the comparison. For a whole matrix use `napoleon sweep`.
