# Replay format

`napoleon replay RUN.jsonl` writes `RUN.replay.json` and a self-contained
`RUN.replay.html`. The JSON is what a viewer reads; the HTML page is the reference for
the Civ panel in Porygon's viewer.

## Top level

| Key | What it holds |
|---|---|
| `format`, `version` | `"napoleon-replay"`, `1`. Bump `version` on any breaking change. |
| `run` | `run_id`, `backend`, `scenario`, `seed`, `agent`, `started_at`, `schema_version`, `ended` (`done`, `step_limit` or `null` for a run cut short). |
| `frame_fields` | Column names for the arrays in each frame, so readers don't hard-code positions. |
| `frames` | One per turn, in turn order. |
| `series` | `score`, `cities`, `population`, `techs`: lists of `[turn, value]`. |
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
  "victory": {"science": 0.33}
}
```

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
 "ok": true, "error": null, "events": ["warrior 3 moved to (1, -2)"]}
```

`action` follows `napoleon.schema` (six types). When the agent's output didn't parse,
`action` holds the raw output instead and `ok` is `false`.

## Porting the panel into the viewer

The HTML page shows what the panel needs; port behaviour, not the code.

1. **Loading.** Accept a `.replay.json` (drop or URL) and check `format` and
   `version`. Don't read raw run files in the browser: they're larger and carry whole
   snapshots.
2. **Turn control.** Slider over `frames`, previous and next, and play at about two
   turns a second.
3. **Map.** Hexes from `tiles`, colored by terrain with fogged tiles faded and owned
   tiles outlined in the owner's color. Cities as squares labeled with population,
   units as dots with the type's first letter. Hover shows the tile or unit.
4. **Score chart.** `series.score` with a marker at the selected turn. Cities and
   population on the same axis are optional.
5. **Steps.** For the selected turn: each step's action in words, its reasoning, any
   error in red, and its events.
6. **Header.** Agent, backend, scenario, seed and how the run ended.

Comparing two runs side by side (same scenario and seed, different agents) is the
next thing worth adding.
