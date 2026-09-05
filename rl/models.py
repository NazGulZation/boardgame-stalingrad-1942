"""PyTorch Actor-Critic Neural Network with Invalid Action Masking.

Residual Convolutional architecture designed for 12x10 Stalingrad board state.
Includes CategoricalMasked distribution following CleanRL standards.
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from rl.stalingrad_env import ACTION_SPACE_SIZE


class ResidualBlock(nn.Module):
    """2-layer 3x3 Conv residual block with ReLU activations."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.relu(out + residual)
        return out


class CategoricalMasked(Categorical):
    """Categorical distribution that zeroes out probabilities for invalid actions."""

    def __init__(self, logits=None, probs=None, masks=None):
        if masks is not None:
            # Mask out invalid actions by assigning a huge negative logit (-1e8)
            huge_neg = torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
            masked_logits = torch.where(masks.bool(), logits, huge_neg)
            super().__init__(logits=masked_logits)
        else:
            super().__init__(logits=logits, probs=probs)


class StalingradResNet(nn.Module):
    """Actor-Critic network for Stalingrad 1942."""

    def __init__(self, in_channels=15, action_dim=ACTION_SPACE_SIZE):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)

        # 64 channels * 10 height * 12 width = 7680
        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 12, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def get_value(self, x):
        """Compute state value V(s)."""
        feat = self.conv_in(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        shared = self.fc_shared(feat)
        return self.critic(shared)

    def get_action_and_value(self, x, action=None, masks=None):
        """Compute action, log probability, entropy, and state value."""
        feat = self.conv_in(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        shared = self.fc_shared(feat)

        logits = self.actor(shared)
        dist = CategoricalMasked(logits=logits, masks=masks)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), self.critic(shared)
