# Stalingrad 1942 - a simple 2-player board wargame

A very simple turn-based board wargame set in the Battle of Stalingrad,
played hot-seat (2 players share one screen) as a Python web app.

## How to run

- **Play:** double-click `run.bat` (or run `C:\Anaconda\python.exe app.py`)
  and open http://127.0.0.1:5000. Opening a second browser tab/window shows
  the same battle - handy for passing the turn back and forth.
- **Run all tests:** double-click `run_tests.bat`, or
  `C:\Anaconda\python.exe -m unittest discover -s . -p "test_*.py" -v`

No packages need to be installed - Flask is already in the Anaconda
distribution.

## Teams

- **Axis (grey)** - German 6th Army, deploys on the west bank.
- **Soviets (red)** - 62nd Army, deploys on the east bank and crosses the
  Volga via the ferry. A reinforcement Rifle Squad lands near the ferry on
  the Soviet turn of round 3.

## Play against the computer

Either side (or both) can be computer-controlled. Tick **Axis 6th Army
plays itself** and/or **Soviets 62nd Army plays itself** in the sidebar.
When a side plays itself, its turn runs automatically a moment after it
begin — so you can play Axis-vs-AI, AI-vs-Soviets, or watch a full AI-vs-AI
battle. The AI is a greedy heuristic: it advances on the objectives, closes
to attack, favors killing weakened/high-value targets, and holds objectives
it already controls.

## Units

| Unit | Move | Attack | HP | Range | Special |
|---|---|---|---|---|---|
| Rifle Squad (R) | 2 | 3 | 5 | 1 | - |
| Sniper Team (S) | 1 | 2 | 4 | 3 | Ignores ruins cover |
| Tank (T) | 3 | 4 | 6 | 1 | Ignores ruins cover |

## Rules

- Players alternate turns. Each unit may move once and attack once per
  turn; attacking uses up the unit's whole turn.
- Movement is orthogonal, 1 tile per move point. River is impassable
  except the ferry tiles. Units cannot pass through other units.
- Damage = attack + 1d6 - cover, minimum 1. Ruins give +2 cover, other
  objective tiles +1.
- **Victory:** the Soviets begin holding all 5 objectives and must defend
  them; the Axis must capture a majority (3+). After 12 rounds the side
  holding more objectives wins (an even split is a Soviet victory - the city
  holds). Destroying every enemy unit wins immediately.

## Objectives

Mamayev Kurgan, Central Station, Pavlov's House, Grain Elevator,
Tractor Factory (marked with a star on the board).

## Project layout

Every file is kept under the 700-line budget:

| File | Role |
|---|---|
| `board.py` | Terrain, map, objectives, movement BFS |
| `units.py` | Unit stats and combat resolution |
| `game.py` | Game state, turns, victory conditions |
| `ai.py` | Greedy heuristic AI (pure Python, no Flask) |
| `app.py` | Flask routes (thin controller) |
| `templates/index.html`, `static/css/style.css`, `static/js/game.js` | Frontend |
| `test_board.py`, `test_units.py`, `test_game.py`, `test_app.py`, `test_ai.py` | Unit, regression, API and AI tests |

## Testing

`run_tests.bat` runs the full suite: engine unit tests, pinned-dice combat
tests, victory/reinforcement rules, serialization checks, a scripted
golden-scenario regression test, and Flask API integration tests.
