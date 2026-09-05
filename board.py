"""Board, terrain, objectives and movement helpers for Stalingrad 1942.

The Volga runs along the east edge (column 10) with a single ferry crossing.
Axis deploys on the west bank, Soviets on the east bank.
"""

from collections import deque

WIDTH = 12
HEIGHT = 10
MAX_ROUNDS = 12

OPEN = "open"
RUINS = "ruins"
RIVER = "river"
FERRY = "ferry"

# Objective tiles: (x, y) -> name. Controlling 3+ at the end wins the battle.
OBJECTIVES = {
    (5, 2): "Mamayev Kurgan",
    (3, 5): "Central Station",
    (5, 5): "Pavlov's House",
    (4, 8): "Grain Elevator",
    (8, 1): "Tractor Factory",
}

_FERRY_TILES = [(10, 4), (10, 5)]

# Bombed-out city blocks (including the objectives that sit in ruins).
_RUIN_TILES = [
    (2, 1), (3, 1), (6, 2), (7, 3), (2, 4),
    (6, 5), (7, 6), (3, 7), (8, 8), (1, 8),
    (6, 8), (9, 2), (5, 2), (5, 5), (4, 8), (8, 1),
]


def build_terrain():
    """Return the static terrain grid as a list of rows (y) of tiles (x)."""
    grid = [[OPEN for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for y in range(HEIGHT):
        grid[y][10] = RIVER            # the Volga
    for x, y in _FERRY_TILES:
        grid[y][x] = FERRY             # the only crossing
    for x, y in _RUIN_TILES:
        grid[y][x] = RUINS             # ruined city blocks
    return grid


TERRAIN = build_terrain()


def in_bounds(x, y):
    return 0 <= x < WIDTH and 0 <= y < HEIGHT


def terrain_at(x, y):
    if not in_bounds(x, y):
        return RIVER
    return TERRAIN[y][x]


def defense_bonus(x, y, ignore_ruins=False):
    """Defensive cover of a tile. Ruins +2, other objectives +1, open 0."""
    tile = terrain_at(x, y)
    if tile == RUINS:
        return 0 if ignore_ruins else 2
    if (x, y) in OBJECTIVES:
        return 1
    return 0


def reachable(start, move_points, occupied):
    """BFS: tiles reachable within `move_points` steps from `start`.

    River tiles and occupied tiles (friendly or enemy) block movement and
    cannot be passed through. The start tile itself is never included.
    """
    result = set()
    if move_points <= 0:
        return result
    seen = {start}
    queue = deque([(start[0], start[1], 0)])
    while queue:
        x, y, dist = queue.popleft()
        if dist == move_points:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if nxt in seen or not in_bounds(*nxt):
                continue
            if terrain_at(*nxt) == RIVER or nxt in occupied:
                continue
            seen.add(nxt)
            result.add(nxt)
            queue.append((nxt[0], nxt[1], dist + 1))
    return result


def chebyshev(a, b):
    """Chebyshev (king-move) distance, used for weapon ranges."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
