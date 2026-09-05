# Stalingrad 1942 — Game Rules Reference

Canonical rules and data tables for the `board.py`, `units.py`, and `game.py`
engine. Source of truth is the code; this reference mirrors it.

## Overview

2-player hot-seat board wargame set in the Battle of Stalingrad.

- **Axis (Grey)** — German 6th Army, deploys on the west bank, moves first.
- **Soviet (Red)** — 62nd Army, deploys on the east bank, crosses the Volga
  via the ferry, receives one reinforcement in round 3.

## Board

- Grid: **12 columns** (x = 0..11, increasing eastward) × **10 rows**
  (y = 0..9, increasing southward).
- **Volga**: column 10 is impassable river, except the **ferry** tiles at
  `(10, 4)` and `(10, 5)`.
- Out-of-bounds coordinates return `RIVER`.
- Static terrain grid (legend: `.` open, `#` ruins, `~` Volga river,
  `F` ferry; `M/C/P/G/T` = objectives on their tile type):

```
      x  0   1   2   3   4   5   6   7   8   9  10  11
y=0:      .   .   .   .   .   .   .   .   .   .   ~   .
y=1:      .   .   #   #   .   .   .   .   T   .   ~   .
y=2:      .   .   .   .   .   M   #   .   .   #   ~   .
y=3:      .   .   .   .   .   .   .   #   .   .   ~   .
y=4:      .   .   #   .   .   .   .   .   .   .   F   .
y=5:      .   .   .   C   .   P   #   .   .   .   F   .
y=6:      .   .   .   .   .   .   .   #   .   .   ~   .
y=7:      .   .   .   #   .   .   .   .   .   .   ~   .
y=8:      .   #   .   .   G   .   #   .   #   .   ~   .
y=9:      .   .   .   .   .   .   .   .   .   .   ~   .
```

## Objectives (5)

A unit standing on an objective tile controls it. Controlling more at the
end of round 12 wins (a tie goes to the Soviets).

| Name | Tile | Terrain / cover |
|---|---|---|
| Mamayev Kurgan | (5, 2) | ruins (+2) |
| Central Station | (3, 5) | open (+1) |
| Pavlov's House | (5, 5) | ruins (+2) |
| Grain Elevator | (4, 8) | ruins (+2) |
| Tractor Factory | (8, 1) | ruins (+2) |

## Terrain cover

`board.defense_bonus(x, y, ignore_ruins=False)`:

| Tile | Cover | Notes |
|---|---|---|
| open / ferry | 0 | — |
| objective (non-ruined) | +1 | only Central Station |
| ruins | +2 | ignored (`0`) when the attacker has `ignores_ruins` |

Snipers and Tanks have `ignores_ruins = True` (they also ignore the ruins
bonus on ruined objectives); Rifle Squads do not.

## Units

| Type | Name | Icon | Move | Attack | HP | Range | Ignores ruins |
|---|---|---|---|---|---|---|---|
| rifle | Rifle Squad | R | 2 | 3 | 3 | 1 | no |
| sniper | Sniper Team | S | 1 | 2 | 2 | 3 | yes |
| tank | Tank | T | 3 | 4 | 4 | 1 | yes |

### Start deployments (IDs u1..u12)

| ID | Unit | Start | ID | Unit | Start |
|---|---|---|---|---|---|
| u1 | Rifle | (0, 3) | u7 | Rifle | (11, 3) |
| u2 | Rifle | (0, 5) | u8 | Rifle | (11, 5) |
| u3 | Rifle | (0, 7) | u9 | Rifle | (11, 7) |
| u4 | Sniper | (1, 4) | u10 | Sniper | (11, 4) |
| u5 | Sniper | (1, 6) | u11 | Sniper | (11, 6) |
| u6 | Tank | (1, 5) | u12 | Tank | (11, 5) |

### Reinforcement

On the **Soviet turn of round 3**, one Soviet Rifle Squad (`u13`) lands at
the first free spot in order: `(9, 4)`, `(9, 5)`, `(9, 3)`. If all are
occupied, no reinforcement lands that round.

## Turn flow

1. Axis moves first; rounds increase after each full Axis+Soviet cycle.
2. Each turn, every unit may **move once** then **attack once**.
3. **Movement**: orthogonal (4-direction) BFS up to `move` points. Cannot
   enter river tiles or any occupied tile; cannot pass through occupied
   tiles. `legal_moves()` returns sorted `[x, y]` pairs, excluding start.
4. **Attack**: range is Chebyshev distance. Costs the unit its whole turn
   (sets both `attacked` and `moved`).
5. `end_turn()` switches sides, resets every alive unit's flags, then checks
   elimination and (on the Soviet turn of round 3) spawns the reinforcement.

## Combat

```
damage = max(1, attacker attack + d6(1..6) - defender cover)
```

- Resolved via `units.resolve_attack(attacker, defender, cover, rng)`.
- A unit with `hp <= 0` is destroyed and removed from the alive set.

## Victory

1. **Elimination** (checked when a side's turn ends): if all Soviets die,
   Axis wins; if all Axis die, Soviets win; if both die, Soviets hold.
2. **Objective count** (after round 12 Soviet phase ends): count objectives
   with a standing unit. Axis wins if it holds strictly more; otherwise the
   Soviets win (more, or tie — "the objectives are split; Stalingrad holds").

Since rounds 1..12 and 13..? — the game sets `winner` and rejects further
actions once any victory condition has triggered.