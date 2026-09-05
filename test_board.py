"""Unit tests for board.py - terrain, cover, bounds and movement BFS."""

import unittest

import board


class TestTerrain(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(board.WIDTH, 12)
        self.assertEqual(board.HEIGHT, 10)
        self.assertEqual(len(board.TERRAIN), board.HEIGHT)
        self.assertEqual(len(board.TERRAIN[0]), board.WIDTH)

    def test_volga_runs_down_column_10(self):
        for y in range(board.HEIGHT):
            expected = board.FERRY if y in (4, 5) else board.RIVER
            self.assertEqual(board.terrain_at(10, y), expected)

    def test_ferry_is_the_only_crossing(self):
        self.assertEqual(board.terrain_at(10, 4), board.FERRY)
        self.assertEqual(board.terrain_at(10, 5), board.FERRY)
        self.assertEqual(board.terrain_at(9, 4), board.OPEN)
        self.assertEqual(board.terrain_at(11, 4), board.OPEN)

    def test_five_objectives_exist(self):
        self.assertEqual(len(board.OBJECTIVES), 5)
        for x, y in board.OBJECTIVES:
            self.assertTrue(board.in_bounds(x, y))
        self.assertIn("Mamayev Kurgan", board.OBJECTIVES.values())

    def test_out_of_bounds_is_river(self):
        self.assertEqual(board.terrain_at(-1, 0), board.RIVER)
        self.assertEqual(board.terrain_at(0, 99), board.RIVER)
        self.assertFalse(board.in_bounds(12, 0))
        self.assertTrue(board.in_bounds(11, 9))


class TestCover(unittest.TestCase):
    def test_open_ground_has_no_cover(self):
        self.assertEqual(board.defense_bonus(0, 0), 0)

    def test_ruins_give_two(self):
        self.assertEqual(board.defense_bonus(2, 1), 2)

    def test_plain_objective_gives_one(self):
        self.assertEqual(board.defense_bonus(3, 5), 1)   # Central Station

    def test_ruined_objective_gives_two(self):
        self.assertEqual(board.defense_bonus(5, 5), 2)   # Pavlov's House

    def test_ignore_ruins_flag(self):
        self.assertEqual(board.defense_bonus(2, 1, ignore_ruins=True), 0)
        self.assertEqual(board.defense_bonus(5, 5, ignore_ruins=True), 0)


class TestReachable(unittest.TestCase):
    def test_zero_move_is_empty(self):
        self.assertEqual(board.reachable((0, 0), 0, set()), set())

    def test_blocked_by_river(self):
        result = board.reachable((11, 0), 3, set())
        self.assertNotIn((10, 0), result)        # river column
        self.assertIn((11, 1), result)

    def test_blocked_by_occupied(self):
        result = board.reachable((0, 0), 3, {(1, 0)})
        self.assertNotIn((1, 0), result)
        self.assertNotIn((2, 0), result)         # cannot pass through
        self.assertIn((0, 1), result)

    def test_range_two_extents(self):
        result = board.reachable((5, 5), 2, set())
        self.assertNotIn((5, 5), result)
        self.assertIn((3, 5), result)
        self.assertNotIn((2, 5), result)

    def test_ferry_is_passable(self):
        result = board.reachable((11, 4), 3, set())
        self.assertIn((10, 4), result)
        self.assertIn((9, 4), result)

    def test_board_edges_block(self):
        for x, y in board.reachable((0, 0), 5, set()):
            self.assertTrue(board.in_bounds(x, y))


class TestChebyshev(unittest.TestCase):
    def test_distances(self):
        self.assertEqual(board.chebyshev((0, 0), (3, 2)), 3)
        self.assertEqual(board.chebyshev((5, 5), (5, 5)), 0)
        self.assertEqual(board.chebyshev((2, 3), (2, 4)), 1)


if __name__ == "__main__":
    unittest.main()
