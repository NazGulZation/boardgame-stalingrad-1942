"""Integration tests for app.py using the Flask test client (no server)."""

import unittest

import app as app_module
from game import Game


class FakeRng:
    def __init__(self, rolls):
        self.rolls = list(rolls)

    def randint(self, low, high):
        return self.rolls.pop(0)


class AppTestBase(unittest.TestCase):
    def setUp(self):
        app_module.GAME = Game(rng=FakeRng([]))
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()


class TestPages(AppTestBase):
    def test_index_renders(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Stalingrad 1942", res.data)


class TestState(AppTestBase):
    def test_state_shape(self):
        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        for key in ("turn", "turn_name", "round", "max_rounds", "winner",
                    "board", "units", "objectives", "log"):
            self.assertIn(key, data)
        self.assertEqual(len(data["units"]), 12)
        self.assertEqual(len(data["objectives"]), 5)
        self.assertEqual(data["board"]["width"], 12)
        self.assertEqual(data["board"]["height"], 10)

    def test_fresh_state_has_no_winner(self):
        self.assertIsNone(self.client.get("/api/state").get_json()["winner"])


class TestLegalMoves(AppTestBase):
    def test_legal_moves_for_own_unit(self):
        res = self.client.post("/api/legal_moves", json={"unit_id": "u1"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["moves"])

    def test_no_moves_for_enemy_unit(self):
        res = self.client.post("/api/legal_moves", json={"unit_id": "u7"})
        self.assertEqual(res.get_json()["moves"], [])

    def test_unknown_unit_is_400(self):
        res = self.client.post("/api/legal_moves", json={"unit_id": "zz"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())


class TestMove(AppTestBase):
    def test_move_ok(self):
        res = self.client.post(
            "/api/move", json={"unit_id": "u1", "x": 1, "y": 3})
        self.assertEqual(res.status_code, 200)
        unit = next(u for u in res.get_json()["units"] if u["id"] == "u1")
        self.assertEqual((unit["x"], unit["y"]), (1, 3))

    def test_illegal_move_is_400(self):
        res = self.client.post(
            "/api/move", json={"unit_id": "u1", "x": 10, "y": 3})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_enemy_unit_move_is_400(self):
        res = self.client.post(
            "/api/move", json={"unit_id": "u7", "x": 10, "y": 3})
        self.assertEqual(res.status_code, 400)


class TestAttack(AppTestBase):
    def setUp(self):
        super().setUp()
        game = app_module.GAME
        game.units["u1"].x, game.units["u1"].y = 2, 3
        game.units["u8"].x, game.units["u8"].y = 2, 4

    def test_attack_destroys_target(self):
        app_module.GAME.rng = FakeRng([6])            # 3+6-2=7 damage
        res = self.client.post(
            "/api/attack", json={"attacker_id": "u1", "target_id": "u8"})
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("u8", [u["id"] for u in res.get_json()["units"]])

    def test_out_of_range_attack_is_400(self):
        res = self.client.post(
            "/api/attack", json={"attacker_id": "u1", "target_id": "u9"})
        self.assertEqual(res.status_code, 400)

    def test_friendly_fire_is_400(self):
        res = self.client.post(
            "/api/attack", json={"attacker_id": "u1", "target_id": "u2"})
        self.assertEqual(res.status_code, 400)


class TestTurnAndReset(AppTestBase):
    def test_end_turn_cycle(self):
        self.assertEqual(
            self.client.get("/api/state").get_json()["turn"], "axis")
        res = self.client.post("/api/end_turn", json={})
        self.assertEqual(res.get_json()["turn"], "soviet")
        res = self.client.post("/api/end_turn", json={})
        self.assertEqual(res.get_json()["round"], 2)
        self.assertEqual(res.get_json()["turn"], "axis")

    def test_reset_restores_new_game(self):
        self.client.post("/api/move", json={"unit_id": "u1", "x": 1, "y": 3})
        res = self.client.post("/api/reset", json={})
        data = res.get_json()
        self.assertEqual(data["round"], 1)
        self.assertEqual(data["turn"], "axis")
        unit = next(u for u in data["units"] if u["id"] == "u1")
        self.assertEqual((unit["x"], unit["y"]), (0, 3))
        self.assertFalse(unit["moved"])

    def test_full_api_regression_cycle(self):
        """Scripted API session: the state must transition exactly as before.
        Any accidental change to routes, payloads or rule flow fails here."""
        # 1. Axis moves and ends turn.
        self.client.post("/api/move", json={"unit_id": "u1", "x": 1, "y": 3})
        self.client.post("/api/end_turn", json={})
        # 2. Soviets move a unit toward the ferry and end turn.
        res = self.client.post("/api/move",
                               json={"unit_id": "u7", "x": 11, "y": 1})
        self.assertEqual(res.status_code, 200)
        by_id = {u["id"]: u for u in res.get_json()["units"]}
        self.assertEqual((by_id["u1"]["x"], by_id["u1"]["y"]), (1, 3))
        self.assertFalse(by_id["u1"]["attacked"])  # reset by the end turn
        res = self.client.post("/api/end_turn", json={})
        data = res.get_json()
        # 3. Pinned expectations after one full round (flags were reset).
        self.assertEqual(data["round"], 2)
        self.assertEqual(data["turn"], "axis")
        by_id = {u["id"]: u for u in data["units"]}
        self.assertEqual((by_id["u1"]["x"], by_id["u1"]["y"]), (1, 3))
        self.assertEqual((by_id["u7"]["x"], by_id["u7"]["y"]), (11, 1))
        self.assertFalse(by_id["u1"]["moved"])
        self.assertFalse(by_id["u1"]["attacked"])


if __name__ == "__main__":
    unittest.main()
