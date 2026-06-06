from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from konerl.tasks.k1_velocity_tracking.kick_command import KickCommand, KickCommandCfg


def _make_command(num_envs: int, cfg: KickCommandCfg) -> KickCommand:
    command = object.__new__(KickCommand)
    command._env = SimpleNamespace(num_envs=num_envs, device="cpu")
    command.cfg = cfg
    command.vel_command_b = torch.zeros(num_envs, 3)
    command.vel_command_w = torch.zeros(num_envs, 3)
    command.kick_direction_command_b = torch.zeros(num_envs, 3)
    command.kick_direction_command_w = torch.zeros(num_envs, 3)
    command.behavior_flags = torch.zeros(num_envs, 4)
    command.is_standing_env = torch.zeros(num_envs, dtype=torch.bool)
    command.is_walking_env = torch.zeros(num_envs, dtype=torch.bool)
    command.is_approach_env = torch.zeros(num_envs, dtype=torch.bool)
    command.is_kicking_env = torch.zeros(num_envs, dtype=torch.bool)
    command.is_dribble_env = torch.zeros(num_envs, dtype=torch.bool)
    return command


class TestKickCommand(unittest.TestCase):
    def test_uses_walking_velocity_ranges(self) -> None:
        num_envs = 256
        cfg = KickCommandCfg(
            entity_name="robot",
            ball_name="ball",
            resampling_time_range=(3.0, 15.0),
            rel_standing_envs=0.0,
            rel_walk_envs=1.0,
            rel_kick_envs=0.0,
            rel_dribble_envs=0.0,
            ranges=KickCommandCfg.Ranges(
                walking_lin_vel_x=(2.0, 3.0),
                walking_lin_vel_y=(1.2, 1.5),
                walking_ang_vel_z=(1.6, 2.0),
                dribble_lin_vel_x=(-0.5, 0.5),
                dribble_lin_vel_y=(-0.5, 0.5),
                dribble_ang_vel_z=(-1.0, 1.0),
            ),
        )
        command = _make_command(num_envs, cfg)

        command._resample_command(torch.arange(num_envs))

        self.assertTrue(torch.all(command.is_walking_env).item())
        self.assertTrue(torch.all(command.vel_command_b[:, 0] >= 2.0).item())
        self.assertTrue(torch.all(command.vel_command_b[:, 0] <= 3.0).item())
        self.assertTrue(torch.all(command.vel_command_b[:, 1] >= 1.2).item())
        self.assertTrue(torch.all(command.vel_command_b[:, 1] <= 1.5).item())
        self.assertTrue(torch.all(command.vel_command_b[:, 2] >= 1.6).item())
        self.assertTrue(torch.all(command.vel_command_b[:, 2] <= 2.0).item())


if __name__ == "__main__":
    unittest.main()
