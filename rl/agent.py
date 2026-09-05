"""Inference adapter and evaluation utilities for Stalingrad RL agents.

Provides an RLAgent interface compatible with ai.play_turn and tools to run
head-to-head simulations against the greedy heuristic AI in ai.py.
"""

import random
import torch
import numpy as np

import board
from game import Game
from rl.stalingrad_env import (
    encode_observation,
    get_action_mask,
    decode_action,
    ACTION_END_TURN,
)
from rl.models import StalingradResNet


class RLAgent:
    """Agent that drives a Game instance using a trained PyTorch model."""

    def __init__(self, model_or_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if isinstance(model_or_path, str):
            self.model = StalingradResNet().to(self.device)
            state_dict = torch.load(model_or_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            self.model = model_or_path.to(self.device)

        self.model.eval()

    def play_turn(self, game, team, deterministic=True):
        """Play one side's entire turn through the game engine.

        Returns list of executed action tuples, matching ai.play_turn:
            [("move", unit_id, (x, y)), ("attack", attacker_id, target_id), ...]
        """
        actions = []
        if game.winner or game.turn != team:
            return actions

        max_steps = len(game.units) * 2 + 2

        for _ in range(max_steps):
            if game.winner or game.turn != team:
                break

            obs = encode_observation(game, team)
            mask = get_action_mask(game, team)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                feat = self.model.conv_in(obs_t)
                feat = self.model.res1(feat)
                feat = self.model.res2(feat)
                shared = self.model.fc_shared(feat)
                logits = self.model.actor(shared)

                huge_neg = torch.tensor(-1e8, device=self.device)
                masked_logits = torch.where(mask_t, logits, huge_neg)

                if deterministic:
                    action_idx = int(torch.argmax(masked_logits, dim=-1).item())
                else:
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    action_idx = int(dist.sample().item())

            if action_idx == ACTION_END_TURN:
                game.end_turn()
                actions.append(("end_turn", None, None))
                break

            act_type, unit_id, target = decode_action(action_idx, game, team)

            if act_type == "end_turn":
                game.end_turn()
                actions.append(("end_turn", None, None))
                break
            elif act_type == "pass":
                unit = game.get(unit_id)
                unit.moved = True
                unit.attacked = True
            elif act_type == "move":
                game.move(unit_id, target[0], target[1])
                actions.append(("move", unit_id, target))
            elif act_type == "attack":
                game.attack(unit_id, target)
                actions.append(("attack", unit_id, target))

        if not game.winner and game.turn == team:
            game.end_turn()
            actions.append(("end_turn", None, None))

        return actions


def evaluate_matchup(axis_player, soviet_player, num_games=50, start_seed=1000):
    """Run simulated battles between two players and collect statistics.

    Players can be an RLAgent or a callable like ai.play_turn.
    """
    results = {"axis": 0, "soviet": 0, "rounds": []}

    for i in range(num_games):
        seed = start_seed + i
        game = Game(rng=random.Random(seed))

        while not game.winner:
            turn = game.turn
            if turn == "axis":
                if hasattr(axis_player, "play_turn"):
                    axis_player.play_turn(game, "axis")
                else:
                    axis_player(game, "axis")
            else:
                if hasattr(soviet_player, "play_turn"):
                    soviet_player.play_turn(game, "soviet")
                else:
                    soviet_player(game, "soviet")

        results[game.winner] += 1
        results["rounds"].append(game.round)

    return {
        "games": num_games,
        "axis_wins": results["axis"],
        "soviet_wins": results["soviet"],
        "axis_win_rate": results["axis"] / num_games,
        "avg_rounds": float(np.mean(results["rounds"])),
    }
