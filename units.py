"""Unit definitions and combat resolution for Stalingrad 1942."""

UNIT_STATS = {
    "rifle": {
        "name": "Rifle Squad", "move": 2, "attack": 3,
        "hp": 5, "range": 1, "ignores_ruins": False,
    },
    "sniper": {
        "name": "Sniper Team", "move": 1, "attack": 2,
        "hp": 4, "range": 3, "ignores_ruins": True,
    },
    "tank": {
        "name": "Tank", "move": 3, "attack": 4,
        "hp": 6, "range": 1, "ignores_ruins": True,
    },
}

TEAM_NAMES = {
    "axis": "Axis 6th Army",
    "soviet": "Soviets 62nd Army",
}


class Unit:
    """One unit on the board. Stats come from UNIT_STATS by `type`."""

    __slots__ = ("id", "type", "team", "x", "y", "hp", "moved", "attacked")

    def __init__(self, uid, type_key, team, x, y):
        if type_key not in UNIT_STATS:
            raise ValueError("Unknown unit type: %s" % type_key)
        if team not in TEAM_NAMES:
            raise ValueError("Unknown team: %s" % team)
        self.id = uid
        self.type = type_key
        self.team = team
        self.x = x
        self.y = y
        self.hp = UNIT_STATS[type_key]["hp"]
        self.moved = False
        self.attacked = False

    @property
    def stats(self):
        return UNIT_STATS[self.type]

    @property
    def name(self):
        return self.stats["name"]

    @property
    def alive(self):
        return self.hp > 0

    def start_turn(self):
        """Refresh per-turn action flags."""
        self.moved = False
        self.attacked = False

    def to_dict(self):
        stats = self.stats
        return {
            "id": self.id,
            "type": self.type,
            "name": stats["name"],
            "team": self.team,
            "x": self.x,
            "y": self.y,
            "hp": self.hp,
            "max_hp": stats["hp"],
            "move": stats["move"],
            "attack": stats["attack"],
            "range": stats["range"],
            "moved": self.moved,
            "attacked": self.attacked,
        }


def resolve_attack(attacker, defender, defense_bonus, rng):
    """Resolve one attack, returning the damage dealt.

    Damage = attacker strength + d6 roll - defender cover, minimum 1.
    `rng` must provide randint(1, 6); pass a seeded RNG for determinism.
    """
    roll = rng.randint(1, 6)
    return max(1, attacker.stats["attack"] + roll - defense_bonus)
