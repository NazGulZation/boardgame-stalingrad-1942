"""Deterministic greedy AI that plays one side's whole turn through the engine.

Pure Python: imports only `board` and drives the validated Game methods for
every move/attack/end_turn, so battle logs, dice injection and win checks
behave exactly as in a hot-seat turn.

The AI never rolls dice; the only randomness in a game stays in `Game.rng`.
Each turn the planner rescans every ready friendly unit, scores each reachable
destination (staying put included) against every enemy it could attack from
there, applies the single best action, and repeats until nothing useful
remains - then ends the turn.

Heuristics:
  * attack ~ expected damage (attack + 3.5 mean roll - cover) vs target HP;
    likely kills are worth 2x, and weakened high-value targets are preferred;
  * position ~ objectives held/captured, ruins cover, forward progress and
    proximity to objectives, minus tiles an enemy could reach and hit;
  * style ~ Axis pushes hard west-to-east; Soviets defend and play for the
    round-12 objective tally (a split is a Soviet win).
"""

import board

# ------------------------------------------------------------------ knobs
UNIT_VALUE = {"tank": 12, "sniper": 10, "rifle": 7}

OBJECTIVE_HOLD = 16.0    # bonus for standing on an objective tile
OBJECTIVE_RETAIN = 8.0   # extra for staying on an objective you already hold
OBJECTIVE_KILL = 3.0     # extra for killing a unit standing on one
PROXIMITY_BAND = 6       # proximity bonus decays over this Chebyshev band
PROXIMITY_STEP = 0.6
RUIN_COVER = 3.0         # standing in ruins (cover vs rifle fire)
OBJECTIVE_COVER = 1.0    # standing on a non-ruined objective
THREAT_PENALTY = 1.5     # per enemy that could reach and hit the tile
THREAT_ATTACK_SCALE = 3.0
ENEMY_APPROACH_MAX = 12  # distance span used to reward closing on the foe
ENEMY_APPROACH_WEIGHT = 1.1  # pull toward the nearest enemy (drives combat)

ATTACK_MEAN_ROLL = 3.5
KILL_MULTIPLIER = 2.0

FORWARD_WEIGHT = {"axis": 0.90, "soviet": 0.55}
OBJECTIVE_WEIGHT = {"axis": 1.05, "soviet": 1.45}
COVER_WEIGHT = {"axis": 1.00, "soviet": 1.20}


def play_turn(game, team):
    """Play a full turn for `team` through the engine; return the action list.

    Actions are tuples ("move", unit_id, (x, y)), ("attack", attacker_id,
    target_id) and a closing ("end_turn", None, None). Returns [] when the
    game is over or it is not `team`'s turn.
    """
    return AI(game, team).play()


class AI:
    """Greedy single-turn planner for one side of a Game."""

    def __init__(self, game, team):
        self.game = game
        self.team = team
        self.opponent = "soviet" if team == "axis" else "axis"
        self.actions = []
        self.style = {
            "forward": FORWARD_WEIGHT[team],
            "objective": OBJECTIVE_WEIGHT[team],
            "cover": COVER_WEIGHT[team],
        }

    # ---------------------------------------------------------------- turn
    def play(self):
        if self.game.winner or self.game.turn != self.team:
            return list(self.actions)
        for _ in range(len(self.game.units) + 2):
            best = self._best_action()
            if best is None:
                break
            self._apply(*best)
        self.game.end_turn()
        self.actions.append(("end_turn", None, None))
        return list(self.actions)

    def _apply(self, unit_id, dest, target_id):
        if dest:
            self.game.move(unit_id, dest[0], dest[1])
            self.actions.append(("move", unit_id, dest))
        if target_id:
            self.game.attack(unit_id, target_id)
            self.actions.append(("attack", unit_id, target_id))
        if not dest and not target_id:
            # Holding position still consumes the unit's turn.
            self.game.units[unit_id].moved = True

    # ------------------------------------------------------------ planning
    def _best_action(self):
        """Return the best (unit_id, dest-or-None, target-or-None)."""
        best = None
        best_score = float("-inf")
        for unit in self.game.units.values():
            if not unit.alive or unit.team != self.team or unit.moved:
                continue
            # Baseline: stay put (hold the current tile). Without this a unit
            # that has no enemy in range is forced to move even from a tile it
            # should hold, such as an objective.
            stay_score = self._position_score(unit, unit.x, unit.y)
            if stay_score > best_score:
                best_score, best = stay_score, (unit.id, None, None)
            # Attack from the current tile.
            for target in self._reachable_hits(unit, unit.x, unit.y):
                score = (self._attack_score(unit, target)
                         + self._position_score(unit, unit.x, unit.y))
                if score > best_score:
                    best_score, best = score, (unit.id, None, target.id)
            # Move, and possibly attack, from every legal tile.
            for dest in self.game.legal_moves(unit.id):
                pos = self._position_score(unit, *dest)
                hits = self._reachable_hits(unit, *dest)
                if hits:
                    for target in hits:
                        score = pos + self._attack_score(unit, target)
                        if score > best_score:
                            best_score, best = score, (unit.id, dest, target.id)
                elif pos > best_score:
                    best_score, best = pos, (unit.id, dest, None)
        return best

    # -------------------------------------------------------------- scoring
    def _position_score(self, unit, x, y):
        style = self.style
        score = 0.0
        for ox, oy in board.OBJECTIVES:
            gap = PROXIMITY_BAND - board.chebyshev((x, y), (ox, oy))
            score += max(0.0, gap) * PROXIMITY_STEP
        if (x, y) in board.OBJECTIVES:
            score += OBJECTIVE_HOLD * style["objective"]
            # Reward staying put on an objective already held, so the AI does
            # not casually abandon a tile it controls for a vacant one.
            if (x, y) == (unit.x, unit.y) and self.game.unit_at(x, y) is unit:
                score += OBJECTIVE_RETAIN * style["objective"]
        cover = board.defense_bonus(x, y)
        if cover >= 2:
            score += RUIN_COVER * style["cover"]
        elif cover == 1:
            score += OBJECTIVE_COVER * style["cover"]
        if self.team == "axis":
            score += style["forward"] * x
        else:
            score += style["forward"] * (board.WIDTH - 1 - x)
        for enemy in self._enemies():
            if board.chebyshev((enemy.x, enemy.y), (x, y)) \
                    <= enemy.stats["move"] + enemy.stats["range"] + 1:
                score -= THREAT_PENALTY \
                    * (enemy.stats["attack"] / THREAT_ATTACK_SCALE)
        # Reward closing the distance to the nearest enemy so the army
        # advances to meet the foe instead of sitting on its own bank.
        nearest = min(board.chebyshev((x, y), (e.x, e.y))
                      for e in self._enemies()) if self._enemies() else 0
        score += ENEMY_APPROACH_WEIGHT * (ENEMY_APPROACH_MAX - nearest)
        return score

    def _attack_score(self, attacker, defender):
        cover = board.defense_bonus(
            defender.x, defender.y,
            ignore_ruins=attacker.stats["ignores_ruins"])
        damage = max(1, attacker.stats["attack"] + ATTACK_MEAN_ROLL - cover)
        value = UNIT_VALUE[defender.type]
        if damage >= defender.hp:
            score = value * KILL_MULTIPLIER
        else:
            score = value * (damage / defender.stats["hp"])
        if (defender.x, defender.y) in board.OBJECTIVES:
            score += OBJECTIVE_KILL
        return score

    # --------------------------------------------------------------- helpers
    def _enemies(self):
        return [u for u in self.game.units.values()
                if u.alive and u.team == self.opponent]

    def _reachable_hits(self, unit, x, y):
        reach = unit.stats["range"]
        return [other for other in self._enemies()
                if board.chebyshev((x, y), (other.x, other.y)) <= reach]