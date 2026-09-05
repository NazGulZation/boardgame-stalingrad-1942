"""Gym-compatible 1v1 Environment for Stalingrad 1942.

Wraps the pure-Python Game engine with a 15-channel (15, 10, 12) spatial
observation tensor and an 897-discrete action space with invalid action masking.
Supports injectable RNG for 100% deterministic test execution.
"""

import random
import numpy as np

import board
from game import Game
from units import UNIT_STATS

NUM_UNIT_SLOTS = 7
NUM_TILES = board.WIDTH * board.HEIGHT  # 120
ACTIONS_PER_UNIT = 1 + NUM_TILES + NUM_UNIT_SLOTS  # 1 (pass) + 120 (move) + 7 (attack) = 128
ACTION_SPACE_SIZE = 1 + NUM_UNIT_SLOTS * ACTIONS_PER_UNIT  # 1 (end_turn) + 7 * 128 = 897

ACTION_END_TURN = 0


def get_team_units(game, team):
    """Return sorted list of units for a team (up to NUM_UNIT_SLOTS)."""
    units = [u for u in game.units.values() if u.team == team]
    units.sort(key=lambda u: int(u.id[1:]))
    return units


def encode_observation(game, team=None):
    """Encode game state as a (15, 10, 12) float32 spatial tensor.

    Channels:
      0: Ruins (#)
      1: River (~)
      2: Ferry (F)
      3: Objective locations
      4: Sticky objective control (+1 friendly, -1 enemy, 0 neutral)
      5: Friendly Rifles (normalized HP)
      6: Friendly Snipers (normalized HP)
      7: Friendly Tanks (normalized HP)
      8: Enemy Rifles (normalized HP)
      9: Enemy Snipers (normalized HP)
      10: Enemy Tanks (normalized HP)
      11: Friendly unit can move (1.0 at unit tile)
      12: Friendly unit can attack (1.0 at unit tile)
      13: Normalized game round (round / 12.0)
      14: Perspective (+1.0 Axis, 0.0 Soviet)
    """
    if team is None:
        team = game.turn

    obs = np.zeros((15, board.HEIGHT, board.WIDTH), dtype=np.float32)

    # Static terrain
    for y in range(board.HEIGHT):
        for x in range(board.WIDTH):
            t = board.terrain_at(x, y)
            if t == board.RUINS:
                obs[0, y, x] = 1.0
            elif t == board.RIVER:
                obs[1, y, x] = 1.0
            elif t == board.FERRY:
                obs[2, y, x] = 1.0

    # Objectives & Control
    control = game.objective_control()
    for (ox, oy), name in board.OBJECTIVES.items():
        obs[3, oy, ox] = 1.0
        c = control.get(name)
        if c == team:
            obs[4, oy, ox] = 1.0
        elif c is not None:
            obs[4, oy, ox] = -1.0

    # Units
    type_channel_offset = {"rifle": 0, "sniper": 1, "tank": 2}
    for unit in game.units.values():
        if not unit.alive:
            continue
        max_hp = float(UNIT_STATS[unit.type]["hp"])
        norm_hp = unit.hp / max_hp
        ch_base = 5 if unit.team == team else 8
        ch = ch_base + type_channel_offset[unit.type]
        obs[ch, unit.y, unit.x] = norm_hp

        if unit.team == team:
            if not unit.moved:
                obs[11, unit.y, unit.x] = 1.0
            if not unit.attacked:
                obs[12, unit.y, unit.x] = 1.0

    # Global features mapped across the grid
    obs[13, :, :] = float(game.round) / float(board.MAX_ROUNDS)
    obs[14, :, :] = 1.0 if team == "axis" else 0.0

    return obs


def get_action_mask(game, team=None):
    """Compute boolean action mask of shape (897,)."""
    if team is None:
        team = game.turn

    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)

    if game.winner or game.turn != team:
        # If battle is over or not team's turn, only pass/no-op is possible
        mask[ACTION_END_TURN] = True
        return mask

    # End turn is always legal
    mask[ACTION_END_TURN] = True

    friendly_units = get_team_units(game, team)
    enemy_team = "soviet" if team == "axis" else "axis"
    enemy_units = get_team_units(game, enemy_team)

    for u_idx, unit in enumerate(friendly_units):
        if u_idx >= NUM_UNIT_SLOTS:
            break
        if not unit.alive or (unit.moved and unit.attacked):
            continue

        base = 1 + u_idx * ACTIONS_PER_UNIT

        # Pass action (marks unit as moved and attacked)
        mask[base + 0] = True

        # Move actions
        if not unit.moved:
            legal_moves = game.legal_moves(unit.id)
            for mx, my in legal_moves:
                tile_idx = my * board.WIDTH + mx
                mask[base + 1 + tile_idx] = True

        # Attack actions
        if not unit.attacked:
            attackable_ids = set(game.attackable(unit.id))
            for e_idx, enemy in enumerate(enemy_units):
                if e_idx >= NUM_UNIT_SLOTS:
                    break
                if enemy.alive and enemy.id in attackable_ids:
                    mask[base + 1 + NUM_TILES + e_idx] = True

    return mask


def decode_action(action_idx, game, team=None):
    """Decode a discrete action integer into an executable engine command.

    Returns:
        tuple: (action_type, unit_id, target)
        where action_type in ("end_turn", "pass", "move", "attack")
    """
    if action_idx == ACTION_END_TURN:
        return ("end_turn", None, None)

    if team is None:
        team = game.turn

    friendly_units = get_team_units(game, team)
    enemy_team = "soviet" if team == "axis" else "axis"
    enemy_units = get_team_units(game, enemy_team)

    u_idx = (action_idx - 1) // ACTIONS_PER_UNIT
    sub_action = (action_idx - 1) % ACTIONS_PER_UNIT

    if u_idx >= len(friendly_units):
        return ("end_turn", None, None)

    unit = friendly_units[u_idx]

    if sub_action == 0:
        return ("pass", unit.id, None)

    if 1 <= sub_action <= NUM_TILES:
        tile_idx = sub_action - 1
        x = tile_idx % board.WIDTH
        y = tile_idx // board.WIDTH
        return ("move", unit.id, (x, y))

    # Attack target
    e_idx = sub_action - 1 - NUM_TILES
    if e_idx < len(enemy_units):
        target = enemy_units[e_idx]
        return ("attack", unit.id, target.id)

    return ("end_turn", None, None)


class Stalingrad1v1Env:
    """Turn-based 1v1 Environment for Stalingrad 1942."""

    def __init__(self, rng=None, reward_shaping=True):
        self._injectable_rng = rng
        self.reward_shaping = reward_shaping
        self.game = None
        self.reset()

    def reset(self, seed=None):
        """Reset the game state. Returns (obs, info)."""
        if seed is not None:
            rng = random.Random(seed)
        elif self._injectable_rng is not None:
            rng = self._injectable_rng
        else:
            rng = random.Random()

        self.game = Game(rng=rng)
        self._prev_ctrl = dict(self.game.objective_control())
        self._prev_hp = {
            u.id: u.hp for u in self.game.units.values() if u.alive
        }

        obs = encode_observation(self.game, self.game.turn)
        mask = get_action_mask(self.game, self.game.turn)
        info = {
            "turn": self.game.turn,
            "round": self.game.round,
            "action_mask": mask,
        }
        return obs, info

    def action_mask(self, team=None):
        return get_action_mask(self.game, team)

    def observe(self, team=None):
        return encode_observation(self.game, team)

    def step(self, action_idx):
        """Execute action for the current turn's team.

        Returns: (obs, reward, terminated, truncated, info)
        """
        current_team = self.game.turn
        enemy_team = "soviet" if current_team == "axis" else "axis"

        if self.game.winner:
            obs = encode_observation(self.game, current_team)
            mask = get_action_mask(self.game, current_team)
            return obs, 0.0, True, False, {
                "turn": current_team,
                "winner": self.game.winner,
                "action_mask": mask,
            }

        act_type, unit_id, target = decode_action(action_idx, self.game, current_team)

        reward = 0.0
        try:
            if act_type == "end_turn":
                self.game.end_turn()
            elif act_type == "pass":
                unit = self.game.get(unit_id)
                unit.moved = True
                unit.attacked = True
            elif act_type == "move":
                self.game.move(unit_id, target[0], target[1])
            elif act_type == "attack":
                target_unit = self.game.get(target)
                hp_before = target_unit.hp
                self.game.attack(unit_id, target)
                if self.reward_shaping:
                    damage_dealt = max(0, hp_before - target_unit.hp)
                    reward += 0.02 * damage_dealt
                    if not target_unit.alive:
                        reward += 0.10
        except ValueError:
            # Action was illegal - penalize and force pass
            reward -= 0.10

        # Objective capture shaping
        if self.reward_shaping:
            current_ctrl = self.game.objective_control()
            for obj_name, new_owner in current_ctrl.items():
                old_owner = self._prev_ctrl.get(obj_name)
                if new_owner != old_owner:
                    if new_owner == current_team:
                        reward += 0.20
                    elif new_owner == enemy_team:
                        reward -= 0.20
            self._prev_ctrl = dict(current_ctrl)

        terminated = bool(self.game.winner)
        if terminated:
            if self.game.winner == current_team:
                reward += 1.0
            else:
                reward -= 1.0

        truncated = False
        next_team = self.game.turn
        obs = encode_observation(self.game, next_team)
        mask = get_action_mask(self.game, next_team)

        info = {
            "turn": next_team,
            "round": self.game.round,
            "winner": self.game.winner,
            "action_mask": mask,
        }

        return obs, reward, terminated, truncated, info
