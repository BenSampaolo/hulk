from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from konerl.tasks.k1_velocity_tracking.observations import BallObservationCache
from konerl.tasks.k1_velocity_tracking.rewards import KickDetector


class _Scene(dict):
    def __init__(self, *args, sensors=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensors = sensors or {}


class _CommandManager:
    def __init__(self, command: torch.Tensor, is_kicking: torch.Tensor):
        self.command = command
        self.term = SimpleNamespace(is_kicking_env=is_kicking)

    def get_command(self, name: str) -> torch.Tensor:
        assert name == "twist"
        return self.command

    def get_term(self, name: str):
        assert name == "twist"
        return self.term


class TestKickTrainingTerms(unittest.TestCase):
    def test_ball_observation_cache_exposes_3d_current_previous_and_velocity(self) -> None:
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.tensor([[0.0, 0.0, 0.5]]),
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        )
        ball = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.tensor([[1.0, 0.2, 0.1]]),
                root_link_vel_w=torch.tensor([[0.3, -0.1, 0.0, 0.0, 0.0, 0.0]]),
            )
        )
        env = SimpleNamespace(
            scene={"robot": robot, "ball": ball},
            num_envs=1,
            device="cpu",
            common_step_counter=0,
            episode_length_buf=torch.tensor([1]),
        )
        cache = BallObservationCache(max_delay_steps=0, max_position_noise=0.0)

        current = cache.position(env, previous=False, noisy=False, delayed=True)
        velocity = cache.velocity(env, delayed=True)

        ball.data.root_link_pos_w = torch.tensor([[1.5, 0.4, 0.1]])
        env.common_step_counter = 1
        current_after_move = cache.position(env, previous=False, noisy=False, delayed=True)
        previous_after_move = cache.position(env, previous=True, noisy=False, delayed=True)

        self.assertEqual(tuple(current.shape), (1, 3))
        self.assertEqual(tuple(previous_after_move.shape), (1, 3))
        self.assertEqual(tuple(velocity.shape), (1, 3))
        torch.testing.assert_close(current, torch.tensor([[1.0, 0.2, -0.4]]))
        torch.testing.assert_close(current_after_move, torch.tensor([[1.5, 0.4, -0.4]]))
        torch.testing.assert_close(previous_after_move, torch.tensor([[1.0, 0.2, -0.4]]))
        torch.testing.assert_close(velocity, torch.tensor([[0.3, -0.1, 0.0]]))

    def test_kick_detector_requires_contact_ball_change_and_foot_motion(self) -> None:
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                site_pos_w=torch.tensor([[[0.9, 0.0, 0.1]]]),
                site_lin_vel_w=torch.tensor([[[1.0, 0.0, 0.0]]]),
            )
        )
        ball = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.tensor([[1.0, 0.0, 0.1]]),
                root_link_vel_w=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            )
        )
        sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0]])))
        command = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
        env = SimpleNamespace(
            scene=_Scene({"robot": robot, "ball": ball}, sensors={"robot_ball_collision": sensor}),
            command_manager=_CommandManager(command, torch.tensor([True])),
            num_envs=1,
            device="cpu",
            common_step_counter=0,
        )
        detector = KickDetector(env)
        detector.steps_since_ball_reset[:] = 10

        detector.update(
            env=env,
            command_name="twist",
            robot_cfg=SceneEntityCfg("robot", site_ids=[0]),
            ball_cfg=SceneEntityCfg("ball"),
            sensor_name="robot_ball_collision",
            foot_ball_distance_threshold=0.28,
            min_ball_speed_delta=0.15,
            min_ball_speed=0.25,
            min_foot_speed_towards_ball=0.1,
            min_steps_since_ball_reset=10,
            reward_window_steps=30,
        )

        self.assertTrue(detector.just_detected.item())
        self.assertEqual(detector.remaining_reward_steps.item(), 30.0)
        torch.testing.assert_close(detector.projected_ball_speed, torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
