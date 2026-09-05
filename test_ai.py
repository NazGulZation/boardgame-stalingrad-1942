"""Tests for ai.py: deterministic greedy AI behavior and full battles.

The AI itself rolls no dice, so tests that pin combat use a FakeRng with a
generous roll buffer; full-battle tests use seeded random.Random to prove
determinism. The AI is allowed to pick whichever legal unit makes the play -
tests assert the *outcome* (enemy dies, objective held), not which unit acts.
"""

import random
import unittest

import board
from ai import play_turn
from game import Game


class FakeRng:
    def __init__(self, rolls):
        self.rolls = list(rolls)

    def randint(self, low, high):
        if not self.rolls:
            raise AssertionError("FakeRng ran out of rolls")
        return self.rolls.pop(0)


def _generous_rolls(n=12):
    """Enough high rolls that every AI attack lands and no roll ever runs out."""
    return FakeRng([6] * n)


class AITestBase(unittest.TestCase):
    def make_game(self, rolls=None):
        if rolls is None:
            rng = _generous_rolls()
        elif isinstance(rolls, list):
            rng = FakeRng(rolls)
        else:
            rng = rolls
        return Game(rng=rng)


class TestAICombat(AITestBase):
    def test_axis_attacks_adjacent_enemy(self):
        game = self.make_game()
        game.units["u1"].x, game.units["u1"].y = 2, 3     # axis rifle
        game.units["u8"].x, game.units["u8"].y = 2, 4     # soviet rifle
        play_turn(game, "axis")
        self.assertEqual(game.turn, "soviet")
        self.assertFalse(game.units["u8"].alive)

    def test_ai_favors_killing_weak_target(self):
        # Two soviet rifles are reachable; u8 is already wounded to 1 HP, so a
        # single hit finishes it. The AI should include u8 among its kills.
        game = self.make_game()
        game.units["u1"].x, game.units["u1"].y = 2, 3
        game.units["u8"].x, game.units["u8"].y = 2, 4
        game.units["u9"].x, game.units["u9"].y = 3, 3
        game.units["u8"].hp = 1
        play_turn(game, "axis")
        self.assertFalse(game.units["u8"].alive)

    def test_ai_never_acts_for_enemy_side(self):
        game = self.make_game()
        positions = {u.id: (u.x, u.y) for u in game.units.values()}
        play_turn(game, "axis")
        for unit in game.units.values():
            if unit.team == "soviet":
                self.assertFalse(unit.moved)
                self.assertFalse(unit.attacked)
                self.assertEqual((unit.x, unit.y), positions[unit.id])

    def test_ai_skips_when_game_over(self):
        game = self.make_game()
        game.units["u7"].hp = 0
        game.winner = "soviet"
        actions = play_turn(game, "axis")
        self.assertEqual(actions, [])
        self.assertEqual(game.turn, "axis")   # end_turn not called


class TestAIPositioning(AITestBase):
    def test_ai_moves_to_capture_objective(self):
        # Park u1 where Central Station (3,5) is the only reachable objective.
        game = self.make_game()
        for unit in game.units.values():
            if unit.team == "axis" and unit.id != "u1":
                unit.x, unit.y = 0, unit.y
        game.units["u1"].x, game.units["u1"].y = 2, 5
        self.assertIn((3, 5), game.legal_moves("u1"))
        play_turn(game, "axis")
        self.assertEqual(game.objective_control()["Central Station"], "axis")
        self.assertEqual((game.units["u1"].x, game.units["u1"].y), (3, 5))

    def test_ai_holds_already_held_objective(self):
        # Already standing on Central Station with no better play: stay put.
        game = self.make_game()
        game.units["u1"].x, game.units["u1"].y = 3, 5
        for unit in game.units.values():
            if unit.team == "axis" and unit.id != "u1":
                unit.x, unit.y = 0, unit.y
        play_turn(game, "axis")
        self.assertEqual((game.units["u1"].x, game.units["u1"].y), (3, 5))
        self.assertEqual(game.objective_control()["Central Station"], "axis")


class TestAITurnFlow(AITestBase):
    def test_turn_switches_and_round_advances(self):
        game = self.make_game()
        play_turn(game, "axis")
        self.assertEqual(game.turn, "soviet")
        self.assertEqual(game.round, 1)
        play_turn(game, "soviet")
        self.assertEqual(game.turn, "axis")
        self.assertEqual(game.round, 2)

    def test_unusable_turn_returns_no_actions(self):
        game = self.make_game()
        self.assertEqual(play_turn(game, "soviet"), [])  # not their turn


class TestFullBattle(AITestBase):
    def test_seeded_battle_is_deterministic(self):
        first = _run_battle(7)
        second = _run_battle(7)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_battle_always_ends_with_winner(self):
        for seed in (1, 2, 3, 4, 5):
            game = _run_battle(seed)
            self.assertIsNotNone(game.winner, "seed %d" % seed)
            self.assertLessEqual(game.round, board.MAX_ROUNDS + 1)

    def test_battle_produces_combat_and_reinforcement(self):
        game = _run_battle(42)
        self.assertIsNotNone(game.winner)
        kills = sum(1 for line in game.log if "destroyed" in line)
        self.assertGreater(kills, 2)


def _run_battle(seed):
    game = Game(rng=random.Random(seed))
    guard = 0
    while not game.winner and guard < 40:
        play_turn(game, game.turn)
        guard += 1
    return game


if __name__ == "__main__":
    unittest.main()