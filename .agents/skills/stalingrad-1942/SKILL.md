---
name: stalingrad-1942
description: Develop, extend, refactor, and test this Stalingrad 1942 hot-seat board wargame (Python + Flask web app) and its CleanRL Reinforcement Learning system. Covers the pure-Python turn-based engine (board/terrain/BFS movement, unit stats, d6 combat, turns/rounds, victory), greedy heuristic AI (`ai.py`), Maskable PPO RL pipeline (`rl/`), web training hub, Flask JSON API contract consumed by the vanilla-JS frontend, and stdlib unittest suites. Use when editing board.py, units.py, game.py, ai.py, app.py, rl/, templates/, static/, or test_*.py.
---

# Stalingrad 1942 — Battleground Skill

Use this skill when working on the "Stalingrad 1942" turn-based board
wargame web app and its CleanRL Reinforcement Learning system in this workspace.
It captures the exact architecture, conventions, and validation rules so you can
extend or refactor safely.

## Hard rules (non-negotiable)

1. **700-line per-file budget.** No file may exceed 700 lines. If a file is
   near the limit, split it (see "Refactoring" below). After any refactor,
   re-run the full test suite and `scripts/check_file_sizes.py`.
2. **Public API contract is regression-locked.** `Game.to_dict()` and
   `Unit.to_dict()` field names are consumed verbatim by the frontend
   (`static/js/game.js`) and asserted by tests. Do not rename or remove
   fields. Extension = add optional fields only.
3. **Engine modules stay pure Python.** `board.py`, `units.py`, and `game.py`
   must not import Flask, HTTP, JSON, or RL frameworks. Only `app.py` talks
   to the web, and `rl/` wraps the engine without modifying it.
4. **All randomness is injectable.** `Game(rng=...)` uses `rng.randint(1, 6)`
   for dice. Tests use a `FakeRng` roll queue so every roll is pinned and the
   suites are fully deterministic.
5. **Tests are stdlib `unittest` only** (no pytest). Suites: `test_board.py`,
   `test_units.py`, `test_game.py`, `test_app.py`, `test_ai.py`, `test_rl.py`.
   All tests must pass under both base and RL Python environments.
6. **Always finish by validating:** full test suite green AND size budget OK.

## Environment & commands (Windows)

- Base Python: `C:\Anaconda\python.exe` (Python 3.11.5). Flask + Jinja2 installed.
- RL Python: `C:\Anaconda\envs\stalingrad-rl\python.exe` (Python 3.11 with PyTorch
  2.5.1 + CUDA 12.1 + GPU acceleration for training and evaluation).
- Run the web app:
  `C:\Anaconda\envs\stalingrad-rl\python.exe app.py` -> http://127.0.0.1:5000
  Hot-seat + AI options + live web-based RL Training Hub in the sidebar.
- Run tests:
  `C:\Anaconda\envs\stalingrad-rl\python.exe -m unittest discover -s . -p "test_*.py"`
  (All 135 tests must pass).
- Size budget:
  `C:\Anaconda\python.exe .agents\skills\stalingrad-1942\scripts\check_file_sizes.py`
- Train RL model directly:
  `C:\Anaconda\envs\stalingrad-rl\python.exe rl/train_ppo.py --total-timesteps 50000 --num-envs 4 --train-side axis --opponents heuristic`
  (Optionally `--opponents heuristic checkpoints/model1.pt` and `--resume-checkpoint checkpoints/<model>.pt` to continue training).

## Project map

| File | Responsibility |
|---|---|
| `board.py` | Board constants (12x10), terrain grid, 5 objectives, BFS movement range, defense cover |
| `units.py` | `UNIT_STATS`, `TEAM_NAMES`, `Unit` class, `resolve_attack` damage math |
| `game.py` | `Game`: deployments, turn/round flow, move/attack/end_turn validation, sticky objective control, victory, serialization |
| `ai.py` | Greedy heuristic AI: `play_turn(game, team)` plays one full side through the engine (pure Python, no Flask) |
| `app.py` | Thin Flask controller: routes, global `GAME`, `ValueError` -> HTTP 400, AI & training endpoints |
| `rl/stalingrad_env.py` | 1v1 Gym environment, (15, 10, 12) spatial tensor observation, 897 discrete action space with action masking, multi-opponent round-robin pool |
| `rl/models.py` | ResNet Actor-Critic architecture and `CategoricalMasked` distribution |
| `rl/train_ppo.py` | CleanRL single-file Maskable PPO training loop with GAE, multi-opponent pool round-robin, resume support, and status JSON reporting |
| `rl/agent.py` | `RLAgent` inference wrapper matching `ai.play_turn`, plus head-to-head evaluation utilities |
| `rl/train_manager.py` | Non-blocking background subprocess training manager, multi-agent cache, status tracker, and model loader |
| `templates/index.html` | Board page shell (balanced 3-column dashboard, overlay, buttons, per-team AI controls, RL Training Hub) |
| `static/css/style.css` | Layout + winter Stalingrad theme, 3-column responsive grid |
| `static/css/training.css` | RL Training Hub styles, opponent pool checklist, custom step controls, badges |
| `static/js/game.js` | Board rendering, click handling, `/api/*` calls, 2.0s state polling, AI auto-play, overlay |
| `static/js/training_hub.js` | RL Training Hub UI controller, opponent pool checklist sync, subprocess status polling |
| `static/js/training_graph.js` | Interactive HTML5 Canvas reward trend graph, N-steps windowing, hover crosshairs, 5s live polling |
| `test_board.py` | Board/terrain/movement tests (17) |
| `test_units.py` | Unit stats + combat tests (10) |
| `test_game.py` | Rules/turn/victory/reinforcement + golden scenario (40) |
| `test_app.py` | Flask API integration + regression cycle + AI & training endpoints (36) |
| `test_ai.py` | AI combat/positioning/turn-flow + full seeded battles (11) |
| `test_rl.py` | RL observation encoding, action masking, step execution, opponent pool, and model forward pass tests (21) |

## Common tasks

### Train an RL model
1. In the web app: open the sidebar "RL Training Hub", select timesteps, click "Start Training".
2. Headless: run `C:\Anaconda\envs\stalingrad-rl\python.exe rl/train_ppo.py --total-timesteps 50000`.
3. Checkpoints are automatically saved to `checkpoints/` and can be loaded in-game.

### Tune the Heuristic or RL AI
1. Heuristic: adjust knobs in `ai.py` (`OBJECTIVE_HOLD`, `THREAT_PENALTY`, etc.).
2. RL: adjust reward shaping in `rl/stalingrad_env.py` or hyperparameters in `rl/train_ppo.py`.
3. Evaluate head-to-head via `rl.agent.evaluate_matchup(rl_agent, ai.play_turn, num_games=50)`.

### Refactor a file near the 700-line budget
1. Extract cohesive chunks into new sibling modules under the same package.
2. Keep public names stable, or update every import site.
3. Re-run the full suite (`run_tests.bat`) and the size budget script.

## Deep references

- **Game rules & data** (board, terrain, units, turn flow, victory):
  [references/game-rules.md](references/game-rules.md)
- **HTTP API & `/api/state` schema** (frontend contract):
  [references/api-contract.md](references/api-contract.md)
- **RL architecture & training** (encodings, masking, CleanRL PPO, models):
  [references/rl-architecture.md](references/rl-architecture.md)

## Interoperability note

This skill lives in `.agents/skills/` using the shared Agent Skills format
(`SKILL.md` + YAML frontmatter with `name` and `description`). Claude Code,
Cline, Cursor, and other standards-compatible agents can load it.
`.agents/README.md` has the index and pointer-file instructions.