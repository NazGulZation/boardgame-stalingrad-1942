"""Unit tests for units.py - stats and combat resolution."""

import unittest

from units import UNIT_STATS, Unit, resolve_attack


class FakeRng:
    """Deterministic RNG returning queued rolls."""

    def __init__(self, rolls):
        self.rolls = list(rolls)

    def randint(self, low, high):
        if not self.rolls:
            raise AssertionError("FakeRng ran out of rolls")
        return self.rolls.pop(0)


def make_unit(type_key="rifle", team="axis", x=0, y=0, uid="u1"):
    return Unit(uid, type_key, team, x, y)


class TestUnitStats(unittest.TestCase):
    def test_all_types_have_complete_stats(self):
        for stats in UNIT_STATS.values():
            for field in ("name", "move", "attack", "hp", "range",
                          "ignores_ruins"):
                self.assertIn(field, stats)
            self.assertGreater(stats["move"], 0)
            self.assertGreater(stats["hp"], 0)

    def test_expected_unit_types(self):
        self.assertEqual(set(UNIT_STATS), {"rifle", "sniper", "tank"})

    def test_invalid_type_rejected(self):
        with self.assertRaises(ValueError):
            Unit("uX", "bazooka", "axis", 0, 0)

    def test_invalid_team_rejected(self):
        with self.assertRaises(ValueError):
            Unit("uX", "rifle", "usa", 0, 0)


class TestUnitLifecycle(unittest.TestCase):
    def test_new_unit_is_ready(self):
        unit = make_unit()
        self.assertTrue(unit.alive)
        self.assertFalse(unit.moved)
        self.assertFalse(unit.attacked)

    def test_start_turn_resets_flags(self):
        unit = make_unit()
        unit.moved = True
        unit.attacked = True
        unit.start_turn()
        self.assertFalse(unit.moved)
        self.assertFalse(unit.attacked)

    def test_to_dict_shape(self):
        data = make_unit("sniper", "soviet", 3, 4, "u7").to_dict()
        self.assertEqual(data["id"], "u7")
        self.assertEqual(data["type"], "sniper")
        self.assertEqual(data["team"], "soviet")
        self.assertEqual((data["x"], data["y"]), (3, 4))
        self.assertEqual(data["hp"], data["max_hp"])
        self.assertEqual(data["range"], 3)


class TestCombat(unittest.TestCase):
    def test_damage_formula(self):
        attacker = make_unit("rifle")                    # attack 3
        defender = make_unit("rifle", "soviet")
        self.assertEqual(
            resolve_attack(attacker, defender, 0, FakeRng([6])), 9)
        self.assertEqual(
            resolve_attack(attacker, defender, 2, FakeRng([4])), 5)

    def test_minimum_damage_is_one(self):
        attacker = make_unit("sniper")                   # attack 2
        defender = make_unit("rifle", "soviet")
        self.assertEqual(
            resolve_attack(attacker, defender, 5, FakeRng([1])), 1)

    def test_roll_bounds_respected(self):
        attacker = make_unit("rifle")
        defender = make_unit("rifle", "soviet")
        for roll in (1, 3, 6):
            damage = resolve_attack(attacker, defender, 0, FakeRng([roll]))
            self.assertTrue(3 <= damage <= 9)


if __name__ == "__main__":
    unittest.main()
