from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .features import amp_features_from_robot_indices, joint_indices, update_amp_history_


class AMPStateCache:
    """Shared AMP feature/history cache for reward and runner code.

    The reward manager updates this once per environment step so AMP remains a
    normal logged reward term. The runner consumes the same history for
    discriminator training instead of recomputing features/history independently.
    """

    def __init__(
        self,
        *,
        robot: Any,
        joint_names: Iterable[str],
        num_envs: int,
        history_length: int,
        device: torch.device | str,
    ) -> None:
        self.robot = robot
        self.joint_names = tuple(joint_names)
        self.joint_ids = joint_indices(robot, self.joint_names, device)
        self.num_envs = num_envs
        self.history_length = history_length
        self.feature_dim = len(self.joint_names) * 2 + 6
        self.features = torch.empty((num_envs, self.feature_dim), device=device)
        self.history = torch.zeros((num_envs, history_length, self.feature_dim), device=device)
        self.history_shift = torch.empty((num_envs, max(history_length - 1, 0), self.feature_dim), device=device)

    def compute_features(self) -> torch.Tensor:
        return amp_features_from_robot_indices(self.robot, self.joint_ids, self.features)

    def update(self) -> torch.Tensor:
        update_amp_history_(self.history, self.compute_features(), self.history_shift)
        return self.history

    def refresh_reset_envs(self, env_ids: torch.Tensor) -> None:
        """Fill reset env histories with their current post-reset features."""
        if env_ids.numel() == 0:
            return
        features = self.compute_features()
        self.history[env_ids] = features[env_ids].unsqueeze(1).expand(-1, self.history_length, -1)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.history[env_ids] = 0.0
