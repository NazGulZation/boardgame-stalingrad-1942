"""Tests for game.py: rules, turns, victory, and regression scenarios.

All dice are pinned with a FakeRng so every test is deterministic; the
golden-scenario test acts as the main regression net for rules changes.
"""

import unittest

import board
from game import Game


class FakeRng:
    def __init__(self, rolls):
        self.rolls = list(rolls)

    def randint(self, low, high):
        return self.rolls.pop(0)


class GameTestBase(unittest.TestCase):
    def setUp(self):
        self.game = Game(rng=FakeRng([]))


class TestSetup(GameTestBase):
    def test_twelve_units_deployed(self):
        self.assertEqual(len(self.game.units), 12)
        teams = [u.team for u in self.game.units.values()]
        self.assertEqual(teams.count("axis"), 6)
        self.assertEqual(teams.count("soviet"), 6)

    def test_deployments_on_correct_banks(self):
        for unit in self.game.units.values():
            if unit.team == "axis":
                self.assertLessEqual(unit.x, 1)
            else:
                self.assertEqual(unit.x, 11)

    def test_initial_state(self):
        self.assertEqual(self.game.turn, "axis")
        self.assertEqual(self.game.round, 1)
        self.assertIsNone(self.game.winner)

    def test_reset_restores_initial_state(self):
        first = self.game.to_dict()
        self.game.units["u1"].hp = 0
        self.game.round = 5
        self.game.reset()
        self.assertEqual(self.game.to_dict(), first)


class TestMovement(GameTestBase):
    def test_legal_move(self):
        self.game.move("u1", 1, 3)
        unit = self.game.units["u1"]
        self.assertEqual((unit.x, unit.y), (1, 3))
        self.assertTrue(unit.moved)

    def test_cannot_move_twice(self):
        self.game.move("u1", 1, 3)
        with self.assertRaisesRegex(ValueError, "already moved"):
            self.game.move("u1", 2, 3)

    def test_out_of_range_rejected(self):
        with self.assertRaisesRegex(ValueError, "movement range"):
            self.game.move("u1", 8, 3)

    def test_river_rejected(self):
        with self.assertRaisesRegex(ValueError, "Volga"):
            self.game.move("u1", 10, 3)

    def test_occupied_tile_rejected(self):
        with self.assertRaisesRegex(ValueError, "occupied"):
            self.game.move("u1", 1, 5)        # tank starts at (1, 5)

    def test_wrong_team_rejected(self):
        with self.assertRaisesRegex(ValueError, "turn"):
            self.game.move("u7", 10, 3)

    def test_off_board_rejected(self):
        with self.assertRaisesRegex(ValueError, "off the board"):
            self.game.move("u1", -1, 3)

    def test_cannot_move_through_units(self):
        # The axis tank at (1, 5) blocks the path along row 5.
        with self.assertRaisesRegex(ValueError, "movement range"):
            self.game.move("u2", 2, 5)

    def test_sniper_cannot_move_far(self):
        with self.assertRaisesRegex(ValueError, "movement range"):
            self.game.move("u4", 3, 4)        # sniper move 1

    def test_move_range_matches_unit_stats(self):
        legal = self.game.legal_moves("u6")   # tank, move 3
        self.assertIn((4, 5), legal)
        self.assertNotIn((5, 5), legal)


class TestTurns(GameTestBase):
    def test_end_turn_switches_side(self):
        self.game.end_turn()
        self.assertEqual(self.game.turn, "soviet")

    def test_round_increments_after_full_cycle(self):
        self.game.end_turn()
        self.game.end_turn()
        self.assertEqual(self.game.round, 2)
        self.assertEqual(self.game.turn, "axis")

    def test_flags_reset_each_turn(self):
        self.game.move("u1", 1, 3)
        self.game.end_turn()
        self.assertFalse(self.game.units["u1"].moved)

    def test_enemy_unit_cannot_act_out_of_turn(self):
        self.game.end_turn()                  # now the soviets
        with self.assertRaisesRegex(ValueError, "turn"):
            self.game.move("u1", 1, 3)

    def test_log_records_turn_change(self):
        self.game.end_turn()
        self.assertTrue(any("Soviets" in line for line in self.game.log))

class TestCombat(GameTestBase):
    def setUp(self):
        super().setUp()
        # Pull a soviet rifle next to the axis rifle for direct fights.
        self.game.units["u1"].x, self.game.units["u1"].y = 2, 3
        self.game.units["u8"].x, self.game.units["u8"].y = 2, 4

    def test_attack_deals_pinned_damage(self):
        self.game.rng = FakeRng([4])
        self.game.attack("u1", "u8")      # rifle vs ruins: 3+4-2=5 -> kills
        self.assertFalse(self.game.units["u8"].alive)

    def test_destroyed_unit_removed_from_state(self):
        self.game.rng = FakeRng([6])      # 3+6-2=7 damage
        self.game.attack("u1", "u8")
        self.assertFalse(self.game.units["u8"].alive)
        self.assertNotIn(
            "u8", [u["id"] for u in self.game.to_dict()["units"]])

    def test_survivor_keeps_damage_and_attacker_is_done(self):
        self.game.rng = FakeRng([1])      # 3+1-2=2 damage -> 1 HP left
        self.game.attack("u1", "u8")
        self.assertEqual(self.game.units["u8"].hp, 1)
        self.assertTrue(self.game.units["u1"].attacked)
        self.assertTrue(self.game.units["u1"].moved)

    def test_cannot_attack_twice(self):
        self.game.rng = FakeRng([1])
        self.game.attack("u1", "u8")
        with self.assertRaisesRegex(ValueError, "already attacked"):
            self.game.attack("u1", "u8")

    def test_cannot_attack_friend(self):
        with self.assertRaisesRegex(ValueError, "friendly"):
            self.game.attack("u1", "u2")

    def test_out_of_range_rejected(self):
        self.game.units["u8"].x, self.game.units["u8"].y = 6, 6
        with self.assertRaisesRegex(ValueError, "range"):
            self.game.attack("u1", "u8")

    def test_sniper_range_three_and_ignores_ruins(self):
        self.game.units["u4"].x, self.game.units["u4"].y = 2, 1
        # u8 sits in ruins at (2, 4); Chebyshev distance 3, cover ignored.
        self.game.rng = FakeRng([1])      # 2+1-0=3 damage -> kills
        self.game.attack("u4", "u8")
        self.assertFalse(self.game.units["u8"].alive)

    def test_rifle_cannot_reach_three_tiles(self):
        self.game.units["u8"].x, self.game.units["u8"].y = 5, 3
        with self.assertRaisesRegex(ValueError, "range"):
            self.game.attack("u1", "u8")

    def test_destroyed_unit_cannot_be_attacked(self):
        self.game.rng = FakeRng([6])
        self.game.attack("u1", "u8")
        with self.assertRaisesRegex(ValueError, "destroyed"):
            self.game.attack("u4", "u8")


class TestVictory(GameTestBase):
    def park(self, team, spots):
        units = [u for u in self.game.units.values() if u.team == team]
        for unit, (x, y) in zip(units, spots):
            unit.x, unit.y = x, y

    def test_elimination_victory(self):
        for unit in self.game.units.values():
            if unit.team == "soviet":
                unit.hp = 0
        self.game.end_turn()
        self.assertEqual(self.game.winner, "axis")

    def test_axis_objective_majority_wins(self):
        self.park("axis", list(board.OBJECTIVES)[:3])
        self.game.round = board.MAX_ROUNDS
        self.game.turn = "soviet"
        self.game.end_turn()
        self.assertEqual(self.game.winner, "axis")

    def test_objective_tie_goes_to_soviets(self):
        spots = list(board.OBJECTIVES)
        self.park("axis", spots[:2])
        self.park("soviet", spots[2:4])
        self.game.round = board.MAX_ROUNDS
        self.game.turn = "soviet"
        self.game.end_turn()
        self.assertEqual(self.game.winner, "soviet")

    def test_game_over_blocks_actions(self):
        for unit in self.game.units.values():
            if unit.team == "soviet":
                unit.hp = 0
        self.game.end_turn()
        with self.assertRaisesRegex(ValueError, "over"):
            self.game.move("u1", 1, 3)
        with self.assertRaisesRegex(ValueError, "over"):
            self.game.end_turn()

class TestReinforcement(GameTestBase):
    def test_reinforcement_lands_on_round_3_soviet_turn(self):
        self.game.round = 2
        self.game.turn = "soviet"
        self.game.end_turn()              # -> axis, round 3, no spawn yet
        self.assertNotIn("u13", self.game.units)
        self.game.end_turn()              # -> soviets, round 3: spawn
        unit = self.game.units["u13"]
        self.assertEqual((unit.x, unit.y), (9, 4))
        self.assertEqual(unit.team, "soviet")
        self.assertEqual(unit.type, "rifle")

    def test_reinforcement_skips_blocked_spot(self):
        self.game.round = 2
        self.game.turn = "soviet"
        self.game.end_turn()
        self.game.units["u1"].x, self.game.units["u1"].y = 9, 4
        self.game.end_turn()
        self.assertEqual((self.game.units["u13"].x,
                          self.game.units["u13"].y), (9, 5))


class TestSerialization(GameTestBase):
    def test_state_keys(self):
        data = self.game.to_dict()
        for key in ("turn", "turn_name", "round", "max_rounds", "winner",
                    "board", "units", "objectives", "log"):
            self.assertIn(key, data)
        self.assertEqual(data["board"]["width"], 12)
        self.assertEqual(data["board"]["height"], 10)

    def test_to_dict_is_stable_and_alive_only(self):
        self.game.move("u1", 1, 3)
        snapshot = self.game.to_dict()
        self.assertEqual(self.game.to_dict(), snapshot)
        self.assertEqual({u["id"] for u in snapshot["units"]},
                         {u.id for u in self.game.units.values() if u.alive})


class TestGoldenScenario(unittest.TestCase):
    """Regression net: a scripted battle with pinned dice must always end
    in exactly this state. If it fails after a refactor, rules changed."""

    def test_scripted_skirmish_snapshot(self):
        g = Game(rng=FakeRng([6, 4]))
        # Setup: pull units into a skirmish near Pavlov's House.
        g.units["u1"].x, g.units["u1"].y = 2, 3    # axis rifle
        g.units["u8"].x, g.units["u8"].y = 2, 4    # soviet rifle (ruins)
        g.units["u6"].x, g.units["u6"].y = 3, 5    # axis tank
        g.units["u9"].x, g.units["u9"].y = 4, 5    # soviet rifle

        g.attack("u1", "u8")   # 3 + 6 - 2 ruins = 7 -> u8 destroyed
        g.move("u6", 3, 4)
        g.attack("u6", "u9")   # 4 + 4 - 0 open = 8 -> u9 destroyed
        self.assertTrue(g.units["u1"].attacked)
        self.assertTrue(g.units["u6"].attacked)
        g.end_turn()           # Soviets to move, still round 1
        g.move("u7", 11, 1)    # soviet rifle marches up the east bank

        self.assertEqual(g.turn, "soviet")
        self.assertEqual(g.round, 1)
        self.assertIsNone(g.winner)
        self.assertEqual(
            sorted(u.id for u in g.units.values() if u.alive),
            ["u1", "u10", "u11", "u12", "u2", "u3", "u4", "u5", "u6", "u7"])
        self.assertEqual((g.units["u1"].x, g.units["u1"].y), (2, 3))
        self.assertEqual((g.units["u6"].x, g.units["u6"].y), (3, 4))
        self.assertEqual((g.units["u7"].x, g.units["u7"].y), (11, 1))
        self.assertTrue(g.units["u7"].moved)
        self.assertFalse(g.units["u7"].attacked)
        self.assertEqual(g.objective_control(),
                         {name: None for name in board.OBJECTIVES.values()})
        self.assertEqual(len(g.log), 6)
        self.assertIn("destroyed Rifle Squad", g.log[1])
        self.assertIn("destroyed Rifle Squad", g.log[3])


if __name__ == "__main__":
    unittest.main()
