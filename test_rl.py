"""Unit tests for Reinforcement Learning environment, models, and action masking.

Follows project conventions: stdlib unittest only, fully deterministic tests.
"""

import random
import unittest
import numpy as np

import board
import ai
from game import Game
from rl.stalingrad_env import (
    Stalingrad1v1Env,
    encode_observation,
    get_action_mask,
    decode_action,
    ACTION_SPACE_SIZE,
    ACTION_END_TURN,
    ACTIONS_PER_UNIT,
    NUM_TILES,
)

try:
    import torch
    from rl.models import StalingradResNet, CategoricalMasked
    from rl.agent import RLAgent
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False




class FakeRng:
    """Deterministic roll queue mirroring test_game.FakeRng."""

    def __init__(self, rolls):
        self.rolls = list(rolls)

    def randint(self, a, b):
        if not self.rolls:
            raise AssertionError("FakeRng queue empty.")
        return self.rolls.pop(0)


class TestRLObservation(unittest.TestCase):

    def setUp(self):
        self.game = Game(rng=random.Random(42))

    def test_observation_shape(self):
        obs = encode_observation(self.game, "axis")
        self.assertEqual(obs.shape, (15, 10, 12))
        self.assertEqual(obs.dtype, np.float32)

    def test_terrain_channels(self):
        obs = encode_observation(self.game, "axis")
        # Ferry at (10, 4) and (10, 5)
        self.assertEqual(obs[2, 4, 10], 1.0)
        self.assertEqual(obs[2, 5, 10], 1.0)
        # Ruins at (5, 2) Mamayev Kurgan
        self.assertEqual(obs[0, 2, 5], 1.0)
        # Volga at (10, 0)
        self.assertEqual(obs[1, 0, 10], 1.0)

    def test_objective_channels(self):
        obs = encode_observation(self.game, "axis")
        # All 5 objectives start controlled by Soviet
        # From Axis perspective, control channel should be -1.0
        for (x, y) in board.OBJECTIVES.keys():
            self.assertEqual(obs[3, y, x], 1.0)
            self.assertEqual(obs[4, y, x], -1.0)

        # From Soviet perspective, control channel should be +1.0
        sov_obs = encode_observation(self.game, "soviet")
        for (x, y) in board.OBJECTIVES.keys():
            self.assertEqual(sov_obs[4, y, x], 1.0)

    def test_perspective_channel(self):
        obs_axis = encode_observation(self.game, "axis")
        self.assertTrue(np.all(obs_axis[14] == 1.0))

        obs_sov = encode_observation(self.game, "soviet")
        self.assertTrue(np.all(obs_sov[14] == 0.0))


class TestRLActionMasking(unittest.TestCase):

    def setUp(self):
        self.game = Game(rng=random.Random(42))

    def test_mask_dimensions_and_end_turn(self):
        mask = get_action_mask(self.game, "axis")
        self.assertEqual(mask.shape, (ACTION_SPACE_SIZE,))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[ACTION_END_TURN])

    def test_axis_legal_moves_match_engine(self):
        mask = get_action_mask(self.game, "axis")
        # u1 is first Axis rifle at (0, 3)
        u1_legal = self.game.legal_moves("u1")
        self.assertGreater(len(u1_legal), 0)

        base = 1 + 0 * ACTIONS_PER_UNIT
        # Pass is legal
        self.assertTrue(mask[base + 0])

        # Moves match legal_moves
        for x, y in u1_legal:
            tile_idx = y * board.WIDTH + x
            self.assertTrue(mask[base + 1 + tile_idx])

        # A tile far away should be illegal
        far_tile_idx = 9 * board.WIDTH + 11
        self.assertFalse(mask[base + 1 + far_tile_idx])

    def test_mask_disallows_action_after_unit_acted(self):
        u1 = self.game.get("u1")
        u1.moved = True
        u1.attacked = True

        mask = get_action_mask(self.game, "axis")
        base = 1 + 0 * ACTIONS_PER_UNIT
        # All actions for u1 should now be False
        self.assertFalse(np.any(mask[base:base + ACTIONS_PER_UNIT]))

    def test_decode_action_roundtrip(self):
        act_type, uid, target = decode_action(ACTION_END_TURN, self.game, "axis")
        self.assertEqual(act_type, "end_turn")

        # Decode move for u1
        base = 1 + 0 * ACTIONS_PER_UNIT
        tile_idx = 3 * board.WIDTH + 1  # (1, 3)
        act_type, uid, target = decode_action(base + 1 + tile_idx, self.game, "axis")
        self.assertEqual(act_type, "move")
        self.assertEqual(uid, "u1")
        self.assertEqual(target, (1, 3))

    def test_attack_masking(self):
        # Place u1 next to u7 (Soviet rifle at 11, 3)
        u1 = self.game.get("u1")
        u1.x, u1.y = 10, 3  # distance 1 from u7 at (11, 3)
        u7 = self.game.get("u7")

        mask = get_action_mask(self.game, "axis")
        base = 1 + 0 * ACTIONS_PER_UNIT
        # Soviet units sorted: u7 is index 0
        attack_act_idx = base + 1 + NUM_TILES + 0
        self.assertTrue(mask[attack_act_idx])

        act_type, uid, target_id = decode_action(attack_act_idx, self.game, "axis")
        self.assertEqual(act_type, "attack")
        self.assertEqual(uid, "u1")
        self.assertEqual(target_id, "u7")

    def test_strict_turn_completion_masking(self):
        # With ready units having legal moves, ACTION_END_TURN should be False in strict mode
        mask_permissive = get_action_mask(self.game, "axis", strict_turn_completion=False)
        self.assertTrue(mask_permissive[ACTION_END_TURN])

        mask_strict = get_action_mask(self.game, "axis", strict_turn_completion=True)
        self.assertFalse(mask_strict[ACTION_END_TURN])

        # Mark all friendly units moved and attacked (as if passed or acted)
        for u in self.game.units.values():
            if u.team == "axis":
                u.moved = True
                u.attacked = True

        mask_done = get_action_mask(self.game, "axis", strict_turn_completion=True)
        self.assertTrue(mask_done[ACTION_END_TURN])


class TestRLEnvironment(unittest.TestCase):

    def test_env_reset(self):
        env = Stalingrad1v1Env(rng=random.Random(123))
        obs, info = env.reset()
        self.assertEqual(obs.shape, (15, 10, 12))
        self.assertEqual(info["turn"], "axis")
        self.assertEqual(info["round"], 1)
        self.assertTrue(info["action_mask"][0])

    def test_env_move_and_turn_cycle(self):
        env = Stalingrad1v1Env(rng=random.Random(123))
        env.reset()

        # Execute move for u1 to (1, 3)
        base = 1 + 0 * ACTIONS_PER_UNIT
        tile_idx = 3 * board.WIDTH + 1
        obs, reward, terminated, truncated, info = env.step(base + 1 + tile_idx)
        self.assertFalse(terminated)
        self.assertEqual(env.game.get("u1").x, 1)
        self.assertEqual(env.game.get("u1").y, 3)

        # End turn
        obs, reward, terminated, truncated, info = env.step(ACTION_END_TURN)
        self.assertEqual(env.game.turn, "soviet")
        self.assertEqual(info["turn"], "soviet")

    def test_env_single_sided_axis_vs_ai(self):
        env = Stalingrad1v1Env(rng=random.Random(42), train_side="axis",
                               opponent_policy=ai.play_turn, strict_turn_completion=False)
        obs, info = env.reset()
        self.assertEqual(info["turn"], "axis")
        self.assertEqual(info["round"], 1)

        # Agent ends turn -> AI automatically plays Soviet turn
        obs, reward, terminated, truncated, info = env.step(ACTION_END_TURN)
        # Turn cycles back to axis in round 2 (unless battle ended)
        if not terminated:
            self.assertEqual(env.game.turn, "axis")
            self.assertEqual(env.game.round, 2)

    def test_env_single_sided_soviet_vs_ai(self):
        env = Stalingrad1v1Env(rng=random.Random(42), train_side="soviet",
                               opponent_policy=ai.play_turn, strict_turn_completion=False)
        obs, info = env.reset()
        # Axis plays first turn automatically during reset
        self.assertEqual(env.game.turn, "soviet")
        self.assertEqual(info["turn"], "soviet")
        self.assertEqual(env.game.round, 1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for model tests")
class TestRLModel(unittest.TestCase):

    def test_forward_pass_and_masked_sampling(self):
        model = StalingradResNet()
        obs = torch.randn(2, 15, 10, 12)
        masks = torch.zeros(2, ACTION_SPACE_SIZE, dtype=torch.bool)
        # Only allow actions 0 and 5
        masks[:, 0] = True
        masks[:, 5] = True

        action, logprob, entropy, value = model.get_action_and_value(obs, masks=masks)
        self.assertEqual(action.shape, (2,))
        self.assertEqual(value.shape, (2, 1))

        # Sampled actions must be either 0 or 5
        for a in action.tolist():
            self.assertIn(a, [0, 5])

    def test_rl_agent_play_turn(self):
        model = StalingradResNet()
        agent = RLAgent(model, device="cpu")
        game = Game(rng=random.Random(100))

        actions = agent.play_turn(game, "axis", deterministic=True)
        self.assertGreater(len(actions), 0)
        self.assertEqual(actions[-1][0], "end_turn")
        self.assertEqual(game.turn, "soviet")

    def test_train_ppo_parse_args_model_name(self):
        import sys
        from unittest.mock import patch
        from rl.train_ppo import parse_args
        test_args = ["train_ppo.py", "--model-name", "custom_model_v1.pt", "--total-timesteps", "1000"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertEqual(args.model_name, "custom_model_v1.pt")
            self.assertEqual(args.total_timesteps, 1000)

    def test_train_ppo_parse_args_num_envs(self):
        import sys
        from unittest.mock import patch
        from rl.train_ppo import parse_args
        test_args = ["train_ppo.py", "--num-envs", "8", "--total-timesteps", "2048"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            self.assertEqual(args.num_envs, 8)
            self.assertEqual(args.batch_size, 8 * args.num_steps)

    def test_status_file_reward_history(self):
        import tempfile
        import json
        import os
        from rl.train_ppo import write_status
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = os.path.join(tmpdir, "train_status.json")
            status_data = {
                "status": "training",
                "step": 512,
                "reward": 0.125,
                "reward_history": [[256, 0.08], [512, 0.125]],
                "num_envs": 4,
            }
            write_status(status_path, status_data)
            self.assertTrue(os.path.isfile(status_path))
            with open(status_path, "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["reward"], 0.125)
            self.assertEqual(len(loaded["reward_history"]), 2)
            self.assertEqual(loaded["num_envs"], 4)


if __name__ == "__main__":
    unittest.main()
