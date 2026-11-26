# Running Napoleon against real Civ VI

`napoleon.civ6` drives Civ VI through [civ6-mcp](https://github.com/lmwilki/civ6-mcp). It
imports civ6-mcp as a library and reads its dataclasses, so no MCP client and no text
parsing are involved. Everything up to the FireTuner connection is tested in CI against
`napoleon.civ6.scripted.ScriptedSource`; the steps below are for the first sessions with
the game open.

## Setup

1. Civ VI (Steam, Gathering Storm) with the FireTuner port enabled. Follow civ6-mcp's
   README for your OS: on Windows the Tuner is an in-game option and needs the Civ VI
   SDK; on macOS and Linux set `EnableTuner 1` in `AppOptions.txt`. Only one tuner
   connection is allowed, so close FireTuner.exe and any civ6-mcp MCP server first.
2. Python 3.12 or newer (civ6-mcp needs it), in one environment:

   ```
   pip install civ6-mcp
   pip install -e path/to/napoleon
   ```

3. Start a game, then check the connection with civ6-mcp's
   `python scripts/test_connection.py` from its repo.

## Starting saves

`reset(scenario, seed)` loads a save by name, so each scenario needs one. Make them in
the game (Quick speed, Small map, Prince) and save on turn 1 with the settler unmoved:

| Scenario | Save name | What it is |
|---|---|---|
| `default` | `napoleon_default` | Normal start |
| `bad_opening` | `napoleon_bad_desert` | Settler on desert or tundra, no fresh water nearby |
| `bad_neighbor` | `napoleon_bad_neighbor` | An aggressive AI one or two cities away |

The seed is recorded with each run but does not change a save. For seed variety, keep
several saves per scenario (`napoleon_default_2`, and so on).

## Running

```
napoleon run --backend civ6 --save default=napoleon_default --agent greedy --turns 30
napoleon metrics runs/civ6-greedy-default-0.jsonl
```

To dry-run the same code path without the game:

```
napoleon run --backend civ6-scripted --agent greedy --turns 15
```

## First-session checks

These are assumptions in the code that only the live game can confirm. Each one is a
quick `run_lua` call or a short run.

1. **Coordinates.** `napoleon.civ6.coords` assumes Civ VI's odd-r offset layout. With
   a unit at plot (x, y), compare our `distance` with
   `Map.GetPlotDistance(x1, y1, x2, y2)` for a few plot pairs on odd and even rows.
   If they disagree, switch the formula to even-r.
2. **Units wanting orders.** `READY_UNITS_LUA` in `napoleon.civ6.live` uses
   `UnitManager.GetActivityType`. Fortify a unit and skip another, then confirm
   neither shows up in `ready_unit_ids()`.
3. **Blocker names.** Note every `ENDTURN_BLOCKING_*` type seen in the first 30 turns.
   Any besides units, production, research and civic arrive as `other` blockers.
   Decide which civ6-mcp `end_turn` clears by itself and which need a new action.
4. **Attacks.** Attack a barbarian with a warrior through `MoveUnit` and check that the
   event text comes back and the barbarian's hp or tile changes in the next snapshot.
5. **Snapshot cost.** Time one `snapshot()` on turn 1 and turn 50. It makes one map
   read per unit and city plus one production read per city; if it's slow, switch to
   civ6-mcp's strategic map read.
6. **Fog.** Confirm foreign units in snapshots are only ones the player can see.

Write what you find back into this file.
