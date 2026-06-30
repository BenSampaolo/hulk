from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw from batched quaternions in ``[w, x, y, z]`` order."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_world_xy_to_yaw_frame(vector_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame XY vectors into a yaw-only body frame."""
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    return torch.stack(
        (
            cos_yaw * vector_w[:, 0] + sin_yaw * vector_w[:, 1],
            -sin_yaw * vector_w[:, 0] + cos_yaw * vector_w[:, 1],
        ),
        dim=-1,
    )


class KickCommand(CommandTerm):
    """PiPlus-style per-episode kick command state for K1 kick training."""

    cfg: KickCommandCfg

    def __init__(self, cfg: KickCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.ball: Entity = env.scene[cfg.ball_name]

        self.kick_direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.kick_direction_b = torch.zeros_like(self.kick_direction_w)
        # Backward-compatible explicit names for reward/debug code.
        self.kick_direction_command_w = self.kick_direction_w
        self.kick_direction_command_b = self.kick_direction_b

        self.phase = torch.zeros(self.num_envs, 2, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, device=self.device)
        self.max_ball_speed = torch.zeros(self.num_envs, device=self.device)
        self.reached_ball = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_since_reached_ball = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ball_speed = torch.zeros(self.num_envs, device=self.device)

        self.metrics["ball_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_ball_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reached_ball"] = torch.zeros(self.num_envs, device=self.device)
        self._last_compute_dt = 0.0

    @property
    def phase_observation(self) -> torch.Tensor:
        return torch.cat((torch.cos(self.phase), torch.sin(self.phase)), dim=-1)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            (
                self.kick_direction_b[:, :2],
                (self.max_ball_speed / self.cfg.max_ball_speed_normalizer).unsqueeze(-1),
            ),
            dim=-1,
        )

    def compute(self, dt: float) -> None:
        self._last_compute_dt = dt
        super().compute(dt)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = self._all_env_ids() if env_ids is None or isinstance(env_ids, slice) else torch.as_tensor(env_ids)
        self.notify_ball_reset(env_ids)
        return super().reset(env_ids)

    def notify_ball_reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = self._all_env_ids()
        self.reached_ball[env_ids] = False
        self.steps_since_reached_ball[env_ids] = 0
        self.ball_speed[env_ids] = 0.0

    def _all_env_ids(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    def _update_metrics(self) -> None:
        ball_vel = self.ball.data.root_link_vel_w[:, :3]
        self.ball_speed = torch.linalg.norm(ball_vel, dim=-1)
        self.metrics["ball_speed"] += self.ball_speed / self._max_command_steps()
        self.metrics["max_ball_speed"] += self.max_ball_speed / self._max_command_steps()
        self.metrics["reached_ball"] += self.reached_ball.float() / self._max_command_steps()

    def _max_command_steps(self) -> float:
        return max(self.cfg.resampling_time_range[1] / max(self._env.step_dt, 1e-6), 1.0)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        num_envs = len(env_ids)
        first_sample = self.command_counter[env_ids] == 0
        ball_stationary = self._ball_speed(env_ids) < self.cfg.kick_resample_ball_speed_threshold
        should_resample_direction = first_sample | ball_stationary

        new_angles = torch.empty(num_envs, device=self.device).uniform_(0.0, 2.0 * math.pi)
        new_direction = torch.stack(
            (torch.cos(new_angles), torch.sin(new_angles), torch.zeros_like(new_angles)),
            dim=-1,
        )
        self.kick_direction_w[env_ids] = torch.where(
            should_resample_direction.unsqueeze(-1),
            new_direction,
            self.kick_direction_w[env_ids],
        )
        if bool(first_sample.any().item()):
            first_env_ids = env_ids[first_sample]
            self.max_ball_speed[first_env_ids] = torch.empty(len(first_env_ids), device=self.device).uniform_(
                *self.cfg.max_ball_speed_range
            )
            self.gait_frequency[first_env_ids] = torch.empty(len(first_env_ids), device=self.device).uniform_(
                *self.cfg.gait_frequency_range
            )
            self.phase[first_env_ids, 0] = 0.0
            self.phase[first_env_ids, 1] = math.pi
            self.notify_ball_reset(first_env_ids)

        self._update_kick_direction_body(env_ids)

    def _update_command(self) -> None:
        env_ids = self._all_env_ids()
        if self._last_compute_dt > 0.0:
            phase_dt = 2.0 * math.pi * self._last_compute_dt * self.gait_frequency
            self.phase = torch.remainder(self.phase + phase_dt.unsqueeze(-1) + math.pi, 2.0 * math.pi) - math.pi

        self._update_kick_direction_body(env_ids)
        self._update_reached_ball(env_ids)

    def _update_kick_direction_body(self, env_ids: torch.Tensor) -> None:
        self.kick_direction_b[env_ids] = quat_apply_inverse(
            self.robot.data.root_link_quat_w[env_ids],
            self.kick_direction_w[env_ids],
        )

    def _update_reached_ball(self, env_ids: torch.Tensor) -> None:
        ball_pos_b = self._ball_position_yaw_frame(env_ids)
        dist = torch.linalg.norm(ball_pos_b[:, :2], dim=-1)
        angle = torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])
        reached_now = (dist < self.cfg.reached_ball_distance) & (torch.abs(angle) < self.cfg.reached_ball_angle)
        self.reached_ball[env_ids] |= reached_now
        self.steps_since_reached_ball[env_ids] = torch.where(
            self.reached_ball[env_ids],
            self.steps_since_reached_ball[env_ids] + 1,
            self.steps_since_reached_ball[env_ids],
        )

    def _ball_position_yaw_frame(self, env_ids: torch.Tensor) -> torch.Tensor:
        rel_pos_w = self.ball.data.root_link_pos_w[env_ids] - self.robot.data.root_link_pos_w[env_ids]
        return quat_apply_inverse(self.robot.data.root_link_quat_w[env_ids], rel_pos_w)

    def _ball_speed(self, env_ids: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(self.ball.data.root_link_vel_w[env_ids, :3], dim=-1)


@dataclass(kw_only=True)
class KickCommandCfg(CommandTermCfg):
    entity_name: str
    ball_name: str
    max_ball_speed_range: tuple[float, float] = (0.3, 3.0)
    max_ball_speed_normalizer: float = 5.0
    gait_frequency_range: tuple[float, float] = (1.25, 1.5)
    kick_resample_ball_speed_threshold: float = 0.05
    reached_ball_distance: float = 0.35
    reached_ball_angle: float = math.pi / 4.0

    @dataclass
    class Ranges:
        kick_angle: tuple[float, float] = (0.0, 2.0 * math.pi)

    ranges: Ranges = field(default_factory=Ranges)

    def build(self, env: ManagerBasedRlEnv) -> KickCommand:
        return KickCommand(self, env)
