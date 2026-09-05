"""Game state and rules for Stalingrad 1942 (2 players, hot-seat).

Pure-Python engine: no Flask imports. The RNG is injectable so tests can
pin dice rolls and be fully deterministic.
"""

import random

import board
from units import TEAM_NAMES, Unit, resolve_attack

DEPLOYMENTS = {
    "axis": [
        ("rifle", 0, 3), ("rifle", 0, 5), ("rifle", 0, 7),
        ("sniper", 1, 4), ("sniper", 1, 6), ("tank", 1, 5),
    ],
    "soviet": [
        ("rifle", 11, 3), ("rifle", 11, 5), ("rifle", 11, 7),
        ("sniper", 11, 4), ("sniper", 11, 6), ("tank", 11, 8),
    ],
}

REINFORCEMENT_ROUND = 2          # Reinforcement arrives earlier (was round 3)
REINFORCEMENT_SPOTS = [(9, 4), (9, 5), (9, 3)]

LOG_LIMIT = 100


class Game:
    """Holds all state for one battle. Mutated only via validated actions."""

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self.reset()

    # ------------------------------------------------------------- setup
    def reset(self):
        self.units = {}
        self._next_id = 1
        for team, deployment in DEPLOYMENTS.items():
            for type_key, x, y in deployment:
                self._spawn(type_key, team, x, y)
        self.turn = "axis"
        self.round = 1
        self.winner = None
        # The Soviets already hold every objective; the Axis must capture a
        # majority. Control is sticky - it stays with the last side to have a
        # unit on the tile until the enemy captures it in turn.
        self.objective_control_state = {
            name: "soviet" for name in board.OBJECTIVES.values()}
        self.log = ["Round 1 - the Axis 6th Army storms Stalingrad!"]

    def _spawn(self, type_key, team, x, y):
        unit = Unit("u%d" % self._next_id, type_key, team, x, y)
        self._next_id += 1
        self.units[unit.id] = unit
        return unit

    def _log(self, message):
        self.log.append(message)
        if len(self.log) > LOG_LIMIT:
            del self.log[:-LOG_LIMIT]

    # ------------------------------------------------------------ queries
    def unit_at(self, x, y):
        for unit in self.units.values():
            if unit.alive and unit.x == x and unit.y == y:
                return unit
        return None

    def occupied(self):
        return {(u.x, u.y) for u in self.units.values() if u.alive}

    def get(self, unit_id):
        unit = self.units.get(unit_id)
        if unit is None or not unit.alive:
            raise ValueError("Unknown or destroyed unit: %s" % unit_id)
        return unit

    def legal_moves(self, unit_id):
        unit = self.get(unit_id)
        if self.winner or unit.team != self.turn or unit.moved:
            return []
        return sorted(board.reachable(
            (unit.x, unit.y), unit.stats["move"], self.occupied()))

    def attackable(self, unit_id):
        unit = self.get(unit_id)
        if self.winner or unit.team != self.turn or unit.attacked:
            return []
        weapon_range = unit.stats["range"]
        return sorted(
            other.id for other in self.units.values()
            if other.alive and other.team != unit.team
            and board.chebyshev(
                (unit.x, unit.y), (other.x, other.y)) <= weapon_range)

    def objective_control(self):
        """Map objective name -> controlling team.

        A unit standing on an objective shows its team (intuitive, and keeps
        the capture visible). An empty objective falls back to the sticky
        capture state: the Soviets begin holding them all, and control flips
        to whichever side last moved a unit onto the tile, persisting even
        after that unit leaves."""
        control = dict(self.objective_control_state)
        for (x, y), name in board.OBJECTIVES.items():
            unit = self.unit_at(x, y)
            if unit:
                control[name] = unit.team
        return control

    # ------------------------------------------------------------ actions
    def _check_active(self, unit):
        if self.winner:
            raise ValueError("The battle is over.")
        if unit.team != self.turn:
            raise ValueError("It is not the %s's turn." % TEAM_NAMES[unit.team])
        return unit

    def move(self, unit_id, x, y):
        unit = self._check_active(self.get(unit_id))
        if unit.moved:
            raise ValueError("That unit has already moved this turn.")
        if not board.in_bounds(x, y):
            raise ValueError("That tile is off the board.")
        if board.terrain_at(x, y) == board.RIVER:
            raise ValueError("The Volga cannot be crossed outside the ferry.")
        if (x, y) in self.occupied():
            raise ValueError("That tile is occupied.")
        legal = board.reachable(
            (unit.x, unit.y), unit.stats["move"], self.occupied())
        if (x, y) not in legal:
            raise ValueError("That tile is out of movement range.")
        unit.x, unit.y = x, y
        unit.moved = True
        if (x, y) in board.OBJECTIVES:
            self.objective_control_state[board.OBJECTIVES[(x, y)]] = unit.team
        self._log("%s moved to (%d, %d)." % (unit.name, x, y))

    def attack(self, attacker_id, target_id):
        attacker = self._check_active(self.get(attacker_id))
        target = self.get(target_id)
        if attacker.team == target.team:
            raise ValueError("You cannot attack a friendly unit.")
        if attacker.attacked:
            raise ValueError("That unit has already attacked this turn.")
        if board.chebyshev(
                (attacker.x, attacker.y),
                (target.x, target.y)) > attacker.stats["range"]:
            raise ValueError("Target is out of weapon range.")
        ignores_ruins = attacker.stats["ignores_ruins"]
        cover = board.defense_bonus(
            target.x, target.y, ignore_ruins=ignores_ruins)
        damage = resolve_attack(attacker, target, cover, self.rng)
        target.hp -= damage
        attacker.attacked = True
        attacker.moved = True  # an attack uses up the unit's whole turn
        if target.hp <= 0:
            self._log("%s destroyed %s (%d damage)!"
                      % (attacker.name, target.name, damage))
        else:
            self._log("%s hit %s for %d damage (%d HP left)."
                      % (attacker.name, target.name, damage, target.hp))

    def end_turn(self):
        if self.winner:
            raise ValueError("The battle is over.")
        if self.turn == "soviet":
            self.round += 1
            if self.round > board.MAX_ROUNDS:
                self._decide_by_objectives()
                return
        self.turn = "soviet" if self.turn == "axis" else "axis"
        for unit in self.units.values():
            if unit.alive:
                unit.start_turn()
        self._check_elimination()
        if (not self.winner and self.turn == "soviet"
                and self.round == REINFORCEMENT_ROUND):
            self._soviet_reinforcement()
        if not self.winner:
            self._log("Round %d - %s to move."
                      % (self.round, TEAM_NAMES[self.turn]))

    # ------------------------------------------------------ win / special
    def _win(self, team, reason):
        self.winner = team
        self._log("%s wins - %s" % (TEAM_NAMES[team], reason))

    def _check_elimination(self):
        alive = {"axis": 0, "soviet": 0}
        for unit in self.units.values():
            if unit.alive:
                alive[unit.team] += 1
        if alive["soviet"] == 0 and alive["axis"] == 0:
            self._win("soviet", "both armies are gone; the city holds.")
        elif alive["soviet"] == 0:
            self._win("axis", "the 62nd Army is destroyed.")
        elif alive["axis"] == 0:
            self._win("soviet", "the 6th Army is destroyed.")

    def _decide_by_objectives(self):
        counts = {"axis": 0, "soviet": 0}
        for team in self.objective_control().values():
            if team:
                counts[team] += 1
        if counts["axis"] > counts["soviet"]:
            self._win("axis", "it holds %d of %d objectives after %d rounds."
                      % (counts["axis"], len(board.OBJECTIVES),
                         board.MAX_ROUNDS))
        elif counts["soviet"] > counts["axis"]:
            self._win("soviet", "it holds %d of %d objectives after %d rounds."
                      % (counts["soviet"], len(board.OBJECTIVES),
                         board.MAX_ROUNDS))
        else:
            self._win("soviet", "the objectives are split; Stalingrad holds.")

    def _soviet_reinforcement(self):
        """Spawn a Soviet reinforcement unit on an empty ferry spot.

        Reinforcement arrives on round 2 (earlier than original round 3)
        to give Soviets board presence to contest objectives.
        """
        for x, y in REINFORCEMENT_SPOTS:
            if self.unit_at(x, y) is None:
                unit = self._spawn("rifle", "soviet", x, y)
                self._log("Reinforcement! A Rifle Squad lands near (%d, %d)."
                          % (x, y))
                return unit
        self._log("The ferry is jammed - no reinforcements land this round.")
        return None

    # ------------------------------------------------------------ export
    def to_dict(self):
        control = self.objective_control()
        return {
            "turn": self.turn,
            "turn_name": TEAM_NAMES[self.turn],
            "round": self.round,
            "max_rounds": board.MAX_ROUNDS,
            "winner": self.winner,
            "board": {
                "width": board.WIDTH,
                "height": board.HEIGHT,
                "terrain": [
                    [board.terrain_at(x, y) for x in range(board.WIDTH)]
                    for y in range(board.HEIGHT)
                ],
            },
            "units": [u.to_dict() for u in self.units.values() if u.alive],
            "objectives": [
                {"name": name, "x": x, "y": y, "controlled_by": control[name]}
                for (x, y), name in board.OBJECTIVES.items()
            ],
            "log": self.log[-25:],
        }
