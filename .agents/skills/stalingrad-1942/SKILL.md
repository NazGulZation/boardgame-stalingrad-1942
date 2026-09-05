---
name: stalingrad-1942
description: Develop, extend, refactor, and test this Stalingrad 1942 hot-seat board wargame (Python + Flask web app). Covers the pure-Python turn-based engine (board/terrain/BFS movement, unit stats, d6 combat, turns/rounds, victory), the greedy heuristic AI (`ai.py`), the Flask JSON API contract consumed by the vanilla-JS frontend, and the stdlib unittest + regression/golden-scenario suites. Use when editing board.py, units.py, game.py, ai.py, app.py, templates/, static/, or any test_*.py in this workspace.
---

# Stalingrad 1942 — Battleground Skill

Use this skill when working on the "Stalingrad 1942" turn-based board
wargame web app in this workspace. It captures the exact architecture,
conventions, and validation rules so you can extend or refactor safely.

## Hard rules (non-negotiable)

1. **700-line per-file budget.** No file may exceed 700 lines. If a file is
   near the limit, split it (see "Refactoring" below). After any refactor,
   re-run the full test suite and `scripts/check_file_sizes.py`.
2. **Public API contract is regression-locked.** `Game.to_dict()` and
   `Unit.to_dict()` field names are consumed verbatim by the frontend
   (`static/js/game.js`) and asserted by tests. Do not rename or remove
   fields. Extension = add optional fields only.
3. **Engine modules stay pure Python.** `board.py`, `units.py`, and `game.py`
   must not import Flask, HTTP, or JSON. Only `app.py` talks to the web.
4. **All randomness is injectable.** `Game(rng=...)` uses `rng.randint(1, 6)`
   for dice. Tests use a `FakeRng` roll queue so every roll is pinned and the
   suites are fully deterministic.
5. **Tests are stdlib `unittest` only** (no pytest). One suite per module:
   `test_board.py`, `test_units.py`, `test_game.py`, `test_app.py`. The
   golden-scenario test (`test_game.TestGoldenScenario`) and the API-cycle
   regression test (`test_app.TestTurnAndReset`) pin exact state — update
   them only when a rule deliberately changes, and say so in the log.
6. **Always finish by validating:** full test suite green AND size budget OK.

## Environment & commands (Windows)

- Python: `C:\Anaconda\python.exe` (Python 3.11.5). Flask 2.2.2 + Jinja2 are
  already installed — never run `pip install`.
- Run the game: from the workspace root,
  `C:\Anaconda\python.exe app.py` -> http://127.0.0.1:5000
  (`run.bat` starts the server and opens the browser). Hot-seat: two players
  share one screen; a second browser tab shows the same state. There is a
  single global `GAME` object — restart the process to start fresh.
- Run tests: `C:\Anaconda\python.exe -m unittest discover -s . -p "test_*.py"`
  (`run_tests.bat`). All 96 tests must pass.
- Size budget: `C:\Anaconda\python.exe
  .agents\skills\stalingrad-1942\scripts\check_file_sizes.py`
- PowerShell gotchas: `&&` is unsupported (use `;`); python stderr is
  surfaced as PowerShell errors — redirect with `2>&1` to a file if noisy.

## Project map

| File | Responsibility |
|---|---|
| `board.py` | Board constants (12x10), terrain grid, 5 objectives, BFS movement range, defense cover |
| `units.py` | `UNIT_STATS`, `TEAM_NAMES`, `Unit` class, `resolve_attack` damage math |
| `game.py` | `Game`: deployments, turn/round flow, move/attack/end_turn validation, sticky objective control, victory, serialization |
| `ai.py` | Greedy heuristic AI: `play_turn(game, team)` plays one full side through the engine (pure Python, no Flask) |
| `app.py` | Thin Flask controller: routes, global `GAME`, `ValueError` -> HTTP 400, AI endpoints |
| `templates/index.html` | Board page shell (side panels, overlay, buttons, AI controls) |
| `static/css/style.css` | Layout + winter Stalingrad theme |
| `static/js/game.js` | Rendering, click handling, `/api/*` calls, 2.5s state polling, AI auto-play |
| `test_board.py` | Board/terrain/movement tests (14) |
| `test_units.py` | Unit stats + combat tests (12) |
| `test_game.py` | Rules/turn/victory/reinforcement + golden scenario (31) |
| `test_app.py` | Flask API integration + regression cycle + AI endpoints (26) |
| `test_ai.py` | AI combat/positioning/turn-flow + full seeded battles (11) |

## Common tasks

### Add a unit type
1. Add a `UNIT_STATS` entry in `units.py` (keys: `name`, `move`, `attack`,
   `hp`, `range`, `ignores_ruins`).
2. Optionally add a deployment entry in `DEPLOYMENTS` (`game.py`).
3. Add tests in `test_units.py`; adjust deployment counts in `test_game.py`
   if you changed the starting roster.
4. Add an icon char in `game.js` `UNIT_ICONS` and a shape class in
   `style.css` if desired.

### Move / add an objective
1. Edit `OBJECTIVES` in `board.py`; add the tile to `_RUIN_TILES` if it
   should sit in ruins (+2 cover).
2. Update tests that enumerate `board.OBJECTIVES` (`test_board.py`,
   `test_game.py` victory tests).

### Tweak combat / damage
1. Change `resolve_attack` (`units.py`) or `defense_bonus` (`board.py`).
2. Update the pinned-damage tests in `test_units.py` and the golden scenario
   in `test_game.py` only if the arithmetic intentionally changes.
3. **Check balance** after any lethality change: run an AI-vs-AI simulation
   (see "Run a balance simulation" below). The game should land near 40-60%
   win split with a healthy fraction of games reaching the round-12 objective
   vote. A 100% one-sided split or 0% reaching objectives means combat is too
   deterministic (min-damage >= HP) or the first-mover dominates.

### Add an API endpoint
1. Add a route in `app.py` (use `_run_action` for engine actions).
2. Add an integration test in `test_app.py`.
3. Document request/response in `references/api-contract.md`.

### Tune the AI
1. Adjust scoring knobs in `ai.py` (`OBJECTIVE_HOLD`, `THREAT_PENALTY`,
   `ENEMY_APPROACH_WEIGHT`, `FORWARD_WEIGHT`, etc.).
2. Add/adjust tests in `test_ai.py`; run a balance simulation to confirm the
   win split stays competitive.
3. The AI must stay pure Python (no Flask) and deterministic (no internal
   RNG — only the engine dice rolls consume randomness).

### Run a balance simulation
1. Write a temporary script that loops `Game(rng=random.Random(seed))` ->
   `play_turn(game, game.turn)` until `game.winner`, collecting win rates,
   win type (elimination vs objective), rounds, and objectives held.
2. Run >= 1000 games. Target: ~40-60% win split, 30-60% reaching the
   round-12 objective vote, and variable attack counts (not a fixed number —
   a fixed count means combat is fully deterministic and the dice don't matter).
3. Delete the temporary script when done — it is a diagnostic, not a project file.

### Refactor a file near the 700-line budget
1. Extract cohesive chunks into new sibling modules under the same package.
2. Keep public names stable, or update every import site (check `game.py`,
   `app.py`, and the tests).
3. Re-run the full suite and the size script. A refactor must be
   behavior-neutral: the golden-scenario + API-cycle regression tests assert
   exactly that.

## Deep references

- **Game rules & data** (board, terrain, units, turn flow, victory):
  [references/game-rules.md](references/game-rules.md)
- **HTTP API & `/api/state` schema** (frontend contract):
  [references/api-contract.md](references/api-contract.md)

## Interoperability note

This skill lives in `.agents/skills/` using the shared Agent Skills format
(`SKILL.md` + YAML frontmatter with `name` and `description`). Claude Code,
Cline, Cursor, and other standards-compatible agents can load it.
`.agents/README.md` has the index and pointer-file instructions.