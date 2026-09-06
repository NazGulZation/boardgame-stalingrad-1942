"""Integration tests for app.py using the Flask test client (no server)."""

import os
import unittest
from unittest import mock

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
        app_module.AI_SIDES = {"axis": False, "soviet": False}
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


class TestAI(AppTestBase):
    def test_state_includes_ai_config(self):
        data = self.client.get("/api/state").get_json()
        self.assertIn("ai", data)
        self.assertEqual(data["ai"], {"axis": False, "soviet": False})

    def test_set_ai_toggles_config(self):
        res = self.client.post("/api/set_ai", json={"axis": True})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ai"]["axis"])
        self.assertFalse(data["ai"]["soviet"])
        # Toggling back off leaves a clean state.
        res = self.client.post("/api/set_ai", json={"axis": False})
        self.assertFalse(res.get_json()["ai"]["axis"])

    def test_ai_turn_rejected_when_not_enabled(self):
        res = self.client.post("/api/ai_turn", json={})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_ai_turn_runs_full_turn_when_enabled(self):
        app_module.GAME.rng = FakeRng([4] * 12)
        # Snapshot the pre-turn positions of every Axis unit.
        before = {u.id: (u.x, u.y)
                  for u in app_module.GAME.units.values()
                  if u.team == "axis"}
        res = self.client.post("/api/set_ai", json={"axis": True})
        self.assertEqual(res.status_code, 200)
        res = self.client.post("/api/ai_turn", json={})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        # A full Axis turn was played and the turn passed to the Soviets.
        self.assertEqual(data["turn"], "soviet")
        self.assertEqual(data["round"], 1)
        # At least one Axis unit changed position during the AI turn.
        by_id = {u["id"]: u for u in data["units"]}
        self.assertTrue(
            any((by_id[uid]["x"], by_id[uid]["y"]) != pos
                for uid, pos in before.items()))


class TestTrainingEndpoints(AppTestBase):
    def test_training_status_get(self):
        res = self.client.get("/api/training/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)
        self.assertIn("checkpoints", data)
        self.assertIn("checkpoint_steps", data)
        self.assertIsInstance(data["checkpoint_steps"], dict)
        self.assertIn("progress", data)
        if "stalingrad_1v1_ppo_final.pt" in data["checkpoints"]:
            self.assertGreater(data["checkpoint_steps"].get("stalingrad_1v1_ppo_final.pt", 0), 0)

    def test_checkpoint_steps_tracking(self):
        steps = app_module.TRAINING_MANAGER.get_checkpoint_steps("stalingrad_1v1_ppo_final.pt")
        self.assertIsInstance(steps, int)
        self.assertGreater(steps, 0)
        self.assertIsNone(app_module.TRAINING_MANAGER.get_checkpoint_steps("nonexistent.pt"))

    @mock.patch.object(app_module.TRAINING_MANAGER, "start_training", return_value=(True, "started"))
    def test_training_start_custom_timesteps(self, mock_start):
        res = self.client.post("/api/training/start", json={"total_timesteps": 35000})
        self.assertEqual(res.status_code, 200)
        mock_start.assert_called_once()
        self.assertEqual(mock_start.call_args[1]["total_timesteps"], 35000)

    @mock.patch.object(app_module.TRAINING_MANAGER, "start_training", return_value=(True, "started"))
    def test_training_start_with_side_and_opponent(self, mock_start):
        res = self.client.post("/api/training/start", json={
            "total_timesteps": 20000,
            "train_side": "soviet",
            "opponent": "checkpoint",
            "opponent_checkpoint": "stalingrad_1v1_ppo_final.pt",
        })
        self.assertEqual(res.status_code, 200)
        mock_start.assert_called_once()
        kwargs = mock_start.call_args[1]
        self.assertEqual(kwargs["train_side"], "soviet")
        self.assertEqual(kwargs["opponent"], "checkpoint")
        self.assertEqual(kwargs["opponent_checkpoint"], "stalingrad_1v1_ppo_final.pt")

    @mock.patch.object(app_module.TRAINING_MANAGER, "start_training", return_value=(True, "started"))
    def test_training_start_with_opponents_list(self, mock_start):
        res = self.client.post("/api/training/start", json={
            "total_timesteps": 20000,
            "train_side": "axis",
            "opponents": ["heuristic", "stalingrad_axis_v1.pt"],
        })
        self.assertEqual(res.status_code, 200)
        mock_start.assert_called_once()
        kwargs = mock_start.call_args[1]
        self.assertEqual(kwargs["opponents"], ["heuristic", "stalingrad_axis_v1.pt"])

    def test_training_start_with_empty_opponents_returns_400(self):
        res = self.client.post("/api/training/start", json={
            "total_timesteps": 20000,
            "train_side": "axis",
            "opponents": [],
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_set_ai_type(self):
        res = self.client.post("/api/set_ai_type", json={"axis": "rl", "soviet": "heuristic"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["ai_types"]["axis"], "rl")
        self.assertEqual(data["ai_types"]["soviet"], "heuristic")

    def test_state_includes_ai_models(self):
        data = self.client.get("/api/state").get_json()
        self.assertIn("ai_models", data)
        self.assertIn("axis", data["ai_models"])
        self.assertIn("soviet", data["ai_models"])

    def test_set_per_team_models(self):
        cps = app_module.TRAINING_MANAGER.list_checkpoints()
        if cps:
            cp = cps[0]
            res = self.client.post("/api/set_ai_type", json={"axis_model": cp, "soviet_model": cp})
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["ai_models"]["axis"], cp)
            self.assertEqual(data["ai_models"]["soviet"], cp)

    def test_select_nonexistent_model_returns_400(self):
        res = self.client.post("/api/training/select_model", json={"model": "non_existent_12345.pt"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_ai_turn_with_rl_mode_runs_successfully(self):
        app_module.GAME.rng = FakeRng([4] * 12)
        res = self.client.post("/api/set_ai", json={"axis": True})
        self.assertEqual(res.status_code, 200)
        res = self.client.post("/api/set_ai_type", json={"axis": "rl"})
        self.assertEqual(res.status_code, 200)
        res = self.client.post("/api/ai_turn", json={})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["turn"], "soviet")

    @mock.patch.object(app_module.TRAINING_MANAGER, "start_training", return_value=(True, "started"))
    def test_training_start_with_model_name(self, mock_start):
        res = self.client.post("/api/training/start", json={
            "total_timesteps": 10000,
            "train_side": "axis",
            "model_name": "my_axis_model",
        })
        self.assertEqual(res.status_code, 200)
        mock_start.assert_called_once()
        kwargs = mock_start.call_args[1]
        self.assertEqual(kwargs["model_name"], "my_axis_model")

    def test_generate_default_model_name(self):
        tm = app_module.TRAINING_MANAGER
        axis_name = tm._generate_default_model_name("axis")
        self.assertTrue(axis_name.startswith("stalingrad_axis_v"))
        self.assertTrue(axis_name.endswith(".pt"))
        soviet_name = tm._generate_default_model_name("soviet")
        self.assertTrue(soviet_name.startswith("stalingrad_soviet_v"))
        self.assertTrue(soviet_name.endswith(".pt"))

    def test_training_completed_auto_assigns_team_checkpoint(self):
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            from rl.train_manager import TrainingManager
            tm = TrainingManager(checkpoints_dir=tmpdir)
            # Create a dummy checkpoint file
            test_cp = "test_auto_assign.pt"
            with open(os.path.join(tmpdir, test_cp), "w") as f:
                f.write("dummy")
            # Write completed train status
            status_data = {
                "status": "completed",
                "train_side": "axis",
                "latest_checkpoint": os.path.join(tmpdir, test_cp),
            }
            with open(tm.status_file, "w") as f:
                json.dump(status_data, f)

            status = tm.get_status()
            self.assertEqual(tm.team_checkpoints["axis"], test_cp)
            self.assertEqual(status["team_checkpoints"]["axis"], test_cp)

    @mock.patch.object(app_module.TRAINING_MANAGER, "start_training", return_value=(True, "started"))
    def test_training_start_with_num_envs(self, mock_start):
        res = self.client.post("/api/training/start", json={
            "total_timesteps": 10000,
            "num_envs": 8,
            "train_side": "axis",
        })
        self.assertEqual(res.status_code, 200)
        mock_start.assert_called_once()
        kwargs = mock_start.call_args[1]
        self.assertEqual(kwargs["num_envs"], 8)

    def test_training_status_includes_reward_and_reward_history(self):
        res = self.client.get("/api/training/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("reward", data)
        self.assertIn("reward_history", data)
        self.assertIn("num_envs", data)
        self.assertIsInstance(data["reward_history"], list)

    def test_training_manager_start_training_opponents(self):
        import tempfile
        import subprocess
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmpdir:
            from rl.train_manager import TrainingManager
            tm = TrainingManager(checkpoints_dir=tmpdir)
            with patch.object(subprocess, "Popen") as mock_popen:
                mock_proc = mock.MagicMock()
                mock_proc.poll.return_value = None
                mock_popen.return_value = mock_proc
                success, msg = tm.start_training(
                    total_timesteps=5000,
                    train_side="axis",
                    opponents=["heuristic", "model_a.pt", "model_b.pt"]
                )
                self.assertTrue(success)
                mock_popen.assert_called_once()
                cmd = mock_popen.call_args[0][0]
                self.assertIn("--opponents", cmd)
                idx = cmd.index("--opponents")
                self.assertEqual(cmd[idx + 1], "heuristic,model_a.pt,model_b.pt")


if __name__ == "__main__":
    unittest.main()

