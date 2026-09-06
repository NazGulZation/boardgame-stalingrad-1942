# Stalingrad 1942 — HTTP API Contract

Server: `C:\Anaconda\python.exe app.py` on `http://127.0.0.1:5000`
(hot-seat; a single global `Game` instance lives in `app.py`).

No authentication. The vanilla-JS frontend (`static/js/game.js`) consumes
every endpoint below and re-renders from the returned state. **Field names
are the frontend contract** — renaming any breaks the UI silently (and fails
tests). Add fields, don't rename.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | — | HTML page |
| GET | `/api/state` | — | full state JSON (schema below) |
| POST | `/api/legal_moves` | `{"unit_id": "u1"}` | `{"moves": [[x,y],...], "targets": ["u8",...]}` |
| POST | `/api/move` | `{"unit_id": "u1", "x": 1, "y": 3}` | full state |
| POST | `/api/attack` | `{"attacker_id": "u1", "target_id": "u8"}` | full state |
| POST | `/api/end_turn` | `{}` | full state |
| POST | `/api/set_ai` | `{"axis": true, "soviet": false}` | full state (new game) |
| POST | `/api/ai_turn` | `{}` | full state |
| POST | `/api/reset` | `{}` | full state (new game) |
| GET | `/api/training/status` | — | training telemetry, SPS, losses, reward, reward_history, checkpoints |
| POST | `/api/training/start` | `{"total_timesteps": 20000, "num_envs": 4, "train_side": "axis"|"soviet", "opponent": "heuristic"|"checkpoint", "opponent_checkpoint": str|null}` | start response + training status |
| POST | `/api/training/stop` | `{}` | stop response + training status |
| POST | `/api/training/select_model` | `{"model": "stalingrad_1v1_ppo_final.pt"}` | updated training status |
| POST | `/api/set_ai_type` | `{"axis": "heuristic"|"rl", "soviet": "heuristic"|"rl", "axis_model": str, "soviet_model": str}` | full state |

### Errors

Any validation failure returns HTTP **400** with `{"error": "<message>"}`.
Common messages: `"That tile is out of movement range."`,
`"That tile is occupied."`, `"The Volga cannot be crossed outside the ferry."`,
`"Target is out of weapon range."`, `"That unit has already attacked this
turn."`, `"It is not the <Team>'s turn."`, `"The battle is over."`,
`"Unknown or destroyed unit: <id>"`.

## Full state schema (`GET /api/state`, and return of action endpoints)

```json
{
  "turn": "axis",
  "turn_name": "Axis 6th Army",
  "round": 1,
  "max_rounds": 12,
  "winner": null,
  "board": {
    "width": 12,
    "height": 10,
    "terrain": [["open", "...11 tiles per row...", "river"]]
  },
  "units": [
    {
      "id": "u1", "type": "rifle", "name": "Rifle Squad", "team": "axis",
      "x": 0, "y": 3, "hp": 3, "max_hp": 3,
      "move": 2, "attack": 3, "range": 1,
      "moved": false, "attacked": false
    }
  ],
  "objectives": [
    {"name": "Mamayev Kurgan", "x": 5, "y": 2, "controlled_by": null}
  ],
  "log": ["Round 1 - the Axis 6th Army storms Stalingrad!"]
}
```

### Field reference

| Field | Type / values | Notes |
|---|---|---|
| `turn` | `"axis" \| "soviet"` | whose turn it is |
| `turn_name` | str | display name for the banner |
| `round` | int (1..12) | increments after a full cycle |
| `max_rounds` | int | always 12 |
| `winner` | `null \| "axis" \| "soviet"` | set once the battle ends |
| `board.width` / `board.height` | int | 12 / 10 |
| `board.terrain` | `string[][]` `[y][x]` | `"open" \| "ruins" \| "river" \| "ferry"` |
| `units[]` | object[] | **alive units only**; `id`, `type` (`"rifle"\|"sniper"\|"tank"`), `name`, `team`, `x`, `y`, `hp`, `max_hp`, `move`, `attack`, `range`, `moved`, `attacked` |
| `objectives[]` | object[] | `name`, `x`, `y`, `controlled_by` (`null` or team) |
| `ai` | `{"axis": bool, "soviet": bool}` | which sides are computer-controlled (optional extra field) |
| `ai_types` | `{"axis": "heuristic"\|"rl", "soviet": "heuristic"\|"rl"}` | AI algorithm per side |
| `ai_models` | `{"axis": str\|null, "soviet": str\|null}` | active checkpoint per side |
| `training` | object | RL training telemetry (`status`, `step`, `sps`, `win_rate`, `pool_eval`, `opponents`), checkpoint list, and per-checkpoint completed steps |

### Behavior notes

- `units[]` contains only alive units; destroyed units disappear.
- `legal_moves`: `moves` is `[[x, y], ...]`, `targets` is enemy unit IDs in
  weapon range; both are `[]` for out-of-turn / moved / already-resolved
  units, and the endpoint returns **400** only for unknown unit IDs.
- `move` rejects out-of-range, river, off-board, occupied tiles, moving
  twice, and out-of-turn actions.
- `attack` rejects friendly fire, out-of-range, already-attacked, and
  targeting destroyed units.
- `end_turn` rejects calls after the battle is over; `reset` always works.
- `set_ai` picks which side(s) are computer-controlled; the frontend reflects
  this in the "Computer opponents" panel. `ai_turn` plays one full turn for
  the side whose turn it currently is — but **only** if that side is enabled
  as AI (otherwise it returns **400** with an error). The AI never rolls its
  own dice, so `Game.rng` stays the single source of randomness.
- All action endpoints (except `legal_moves`) return the **full state**,
  exactly like `GET /api/state`.