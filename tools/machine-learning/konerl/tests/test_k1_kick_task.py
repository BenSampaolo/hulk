from __future__ import annotations

import contextlib
import io
import math
import unittest
from types import SimpleNamespace

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import konerl.tasks.k1_kick  # noqa: F401 - registers Mjlab-Kick-K1
from konerl.tasks.k1_kick.env_cfg import k1_kick_env_cfg
from konerl.tasks.k1_kick.observations import BallObservationCache
from konerl.tasks.k1_kick.randomization import randomize_ball_properties, reset_ball_relative_to_robot
from konerl.tasks.k1_kick.rewards import feet_air_time_reward, feet_slip_penalty


class TestK1KickTask(unittest.TestCase):
    def test_task_registers_and_config_matches_contract(self) -> None:
        self.assertIn("Mjlab-Kick-K1", list_tasks())
        cfg = load_env_cfg("Mjlab-Kick-K1")
        play_cfg = load_env_cfg("Mjlab-Kick-K1", play=True)
        rl_cfg = load_rl_cfg("Mjlab-Kick-K1")

        self.assertEqual(rl_cfg.experiment_name, "k1_kick")
        self.assertEqual(cfg.sim.mujoco.timestep, 0.002)
        self.assertEqual(cfg.sim.mujoco.iterations, 50)
        self.assertEqual(cfg.sim.mujoco.ls_iterations, 50)
        self.assertEqual(cfg.decimation, 10)
        self.assertEqual(cfg.episode_length_s, 10.0)
        self.assertIn("robot", cfg.scene.entities)
        self.assertIn("ball", cfg.scene.entities)
        sensor_names = {sensor.name for sensor in cfg.scene.sensors}
        self.assertIn("feet_ground_contact", sensor_names)
        self.assertNotIn("robot_ball_collision", sensor_names)

        action_cfg = cfg.actions["joint_pos"]
        self.assertEqual(action_cfg.scale, 0.5)
        self.assertEqual(action_cfg.actuator_names, (".*",))
        self.assertTrue(action_cfg.use_default_offset)

        self.assertIn("kick", cfg.commands)
        self.assertEqual(cfg.commands["kick"].max_ball_speed_range, (0.3, 3.0))
        self.assertEqual(cfg.commands["kick"].gait_frequency_range, (1.25, 1.5))

        self.assertEqual(cfg.rewards["orient_to_kick_dir"].weight, 0.3)
        self.assertEqual(cfg.rewards["wrong_approach"].weight, -0.1)
        self.assertEqual(cfg.rewards["too_fast"].weight, -2.0)
        self.assertEqual(cfg.rewards["too_fast"].params["threshold"], 0.7)
        self.assertEqual(cfg.rewards["feet_air_time"].weight, 4.0)
        for key in ("lin_vel_x", "kick_motion", "base_height", "alive"):
            self.assertIn(key, cfg.rewards)
            self.assertEqual(cfg.rewards[key].weight, 0.0)

        for key in (
            "move_to_ball",
            "orient_to_ball",
            "ball_height",
            "ball_speed",
            "orient_to_kick_dir",
            "wrong_approach",
            "too_fast",
            "feet_air_time",
        ):
            self.assertIn(key, cfg.rewards)
        for key in (
            "velocity_tracking",
            "tracking_lin_vel",
            "upright",
            "self_collision",
            "non_feet_ground_contact",
        ):
            self.assertNotIn(key, cfg.rewards)

        event_keys = set(cfg.events)
        self.assertLessEqual(
            {
                "terrain_friction",
                "joint_friction_loss",
                "joint_armature",
                "joint_damping",
                "actuator_gains",
                "robot_body_mass",
                "trunk_mass",
                "trunk_com",
                "body_com_jitter",
                "default_joint_pos",
                "ball_properties",
                "encoder_bias",
                "imu_bias",
                "push_robot",
                "push_ball",
            },
            event_keys,
        )
        for key in ("ball_friction", "ball_mass", "ball_radius"):
            self.assertNotIn(key, event_keys)
        for key in ("randomize_terrain", "reset_ball_on_contact", "teleport_ball", "impulse"):
            self.assertNotIn(key, event_keys)
        self.assertNotIn("randomize_terrain", play_cfg.events)

        for key in ("time_out", "bad_orientation"):
            self.assertIn(key, cfg.terminations)
        for key in ("bad_base_height", "out_of_terrain_bounds", "ball_out_of_bounds"):
            self.assertNotIn(key, cfg.terminations)

    def test_control_arms_true_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "control_arms=True"):
            k1_kick_env_cfg(control_arms=True)

    def test_ball_observation_delay_history_resets_to_zero(self) -> None:
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.zeros((1, 3)),
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        )
        ball = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.tensor([[0.5, -0.25, 0.15]]),
            )
        )
        env = SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            scene={"robot": robot, "ball": ball},
            step_counter=0,
            episode_length_buf=torch.ones(1, dtype=torch.long),
        )
        cache = BallObservationCache(max_delay_steps=10, position_noise_std=0.02)

        torch.testing.assert_close(cache.position(env, noisy=False, delayed=True), torch.zeros((1, 2)))
        self.assertEqual(cache.max_delay_steps, 10)
        self.assertEqual(cache.position_noise_std, 0.02)

        ball.data.root_link_pos_w = torch.tensor([[1.0, 0.25, 0.15]])
        env.step_counter = 1
        torch.testing.assert_close(cache.position(env, noisy=False, delayed=False), torch.tensor([[1.0, 0.25]]))

        ball.data.root_link_pos_w = torch.tensor([[2.0, 0.75, 0.15]])
        cache.reset(env, torch.tensor([0]))
        torch.testing.assert_close(cache.position(env, noisy=False, delayed=True), torch.zeros((1, 2)))

    def test_feet_air_time_allows_short_step_negative_before_weighting(self) -> None:
        class _ContactSensor:
            data = SimpleNamespace(last_air_time=torch.tensor([[0.1, 0.4]]))

            def compute_first_contact(self, step_dt: float) -> torch.Tensor:
                del step_dt
                return torch.tensor([[1.0, 1.0]])

        env = SimpleNamespace(
            num_envs=1,
            device=torch.device("cpu"),
            step_dt=0.02,
            scene={"feet_ground_contact": _ContactSensor()},
        )

        torch.testing.assert_close(
            feet_air_time_reward(env, threshold_min=0.2, threshold_max=0.5),
            torch.tensor([0.1]),
        )

    def test_feet_slip_uses_root_global_xy_speed_per_contact(self) -> None:
        env = SimpleNamespace(
            num_envs=2,
            device=torch.device("cpu"),
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        root_link_lin_vel_w=torch.tensor([[3.0, 4.0, 1.0], [0.0, 2.0, 0.0]]),
                        site_lin_vel_w=torch.zeros((2, 2, 3)),
                    )
                ),
                "feet_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(found=torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
                ),
            },
        )

        torch.testing.assert_close(feet_slip_penalty(env), torch.tensor([5.0, 4.0]))

    def test_live_env_ball_properties_randomize_inertia_from_mass_and_radius(self) -> None:
        cfg = k1_kick_env_cfg(control_arms=False)
        with contextlib.redirect_stdout(io.StringIO()):
            env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            self.assertIn("body_inertia", env.event_manager.domain_randomization_fields)
            ball_body_cfg = SceneEntityCfg("ball", body_names=("ball",))
            ball_geom_cfg = SceneEntityCfg("ball", geom_names=("ball",))
            ball_body_cfg.resolve(env.scene)
            ball_geom_cfg.resolve(env.scene)
            body_id = env.scene["ball"].indexing.body_ids[ball_body_cfg.body_ids][0]
            geom_id = env.scene["ball"].indexing.geom_ids[ball_geom_cfg.geom_ids][0]
            default_mass = env.sim.get_default_field("body_mass")[body_id]
            default_radius = env.sim.get_default_field("geom_size")[geom_id, 0]
            default_friction = env.sim.get_default_field("geom_friction")[geom_id]

            randomize_ball_properties(
                env,
                torch.tensor([0], device=env.device),
                mass_range=(2.0, 2.0),
                radius_range=(3.0, 3.0),
                friction_range=(4.0, 4.0),
                body_cfg=ball_body_cfg,
                geom_cfg=ball_geom_cfg,
            )

            mass = default_mass * 2.0
            radius = default_radius * 3.0
            torch.testing.assert_close(env.sim.model.body_mass[0, body_id], mass)
            torch.testing.assert_close(env.sim.model.geom_size[0, geom_id, 0], radius)
            torch.testing.assert_close(env.sim.model.geom_friction[0, geom_id], default_friction * 4.0)
            torch.testing.assert_close(env.sim.model.body_inertia[0, body_id], torch.full((3,), 0.4 * mass * radius**2))
        finally:
            env.close()

    def test_live_env_builds_expected_command_and_observation_terms(self) -> None:
        cfg = k1_kick_env_cfg(control_arms=False)
        with contextlib.redirect_stdout(io.StringIO()):
            env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                obs, _ = env.reset()

            self.assertEqual(env.step_dt, 0.02)
            self.assertEqual(env.max_episode_length, 500)
            self.assertEqual(env.observation_manager.active_terms["actor"][-2:], ["phase", "command"])
            self.assertEqual(env.observation_manager.active_terms["critic"][5:7], ["phase", "command"])
            self.assertIn("actor", obs)
            self.assertIn("critic", obs)

            command_term = env.command_manager.get_term("kick")
            command = env.command_manager.get_command("kick")
            self.assertEqual(tuple(command.shape), (1, 3))
            self.assertEqual(tuple(command_term.phase_observation.shape), (1, 4))
            self.assertGreaterEqual(command_term.max_ball_speed.item(), 0.3)
            self.assertLessEqual(command_term.max_ball_speed.item(), 3.0)
            torch.testing.assert_close(
                torch.linalg.norm(command_term.kick_direction_w[:, :2], dim=-1),
                torch.ones(1),
                atol=1e-6,
                rtol=1e-6,
            )
        finally:
            env.close()

    def test_registered_default_leg_only_env_resets_and_steps_zero_action(self) -> None:
        cfg = load_env_cfg("Mjlab-Kick-K1")
        with contextlib.redirect_stdout(io.StringIO()):
            env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset()
                action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
                obs, rewards, terminated, time_outs, extras = env.step(action)

            self.assertIn("actor", obs)
            self.assertIn("critic", obs)
            self.assertEqual(tuple(rewards.shape), (env.num_envs,))
            self.assertEqual(tuple(terminated.shape), (env.num_envs,))
            self.assertEqual(tuple(time_outs.shape), (env.num_envs,))
            self.assertIsInstance(extras, dict)
        finally:
            env.close()

    def test_reset_ball_relative_to_robot_uses_source_distance_and_angle_ranges(self) -> None:
        cfg = k1_kick_env_cfg(control_arms=False)
        with contextlib.redirect_stdout(io.StringIO()):
            env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset()
            ball_geom_cfg = SceneEntityCfg("ball", geom_names=("ball",))
            ball_geom_cfg.resolve(env.scene)
            reset_ball_relative_to_robot(
                env,
                torch.tensor([0], device=env.device),
                distance_range=(0.25, 0.4),
                angle_range=(-math.pi / 4.0, math.pi / 4.0),
                velocity_range={},
                ball_cfg=ball_geom_cfg,
                refresh_robot_pose=True,
            )
            env.scene.write_data_to_sim()
            env.sim.forward()

            robot = env.scene["robot"]
            ball = env.scene["ball"]
            rel = ball.data.root_link_pos_w[0, :2] - robot.data.root_link_pos_w[0, :2]
            dist = torch.linalg.norm(rel).item()
            self.assertGreaterEqual(dist, 0.25 - 1e-5)
            self.assertLessEqual(dist, 0.4 + 1e-5)
            global_geom_id = ball.indexing.geom_ids[ball_geom_cfg.geom_ids][0]
            expected_z = env.scene.env_origins[0, 2] + env.sim.model.geom_size[0, global_geom_id, 0]
            self.assertAlmostEqual(ball.data.root_link_pos_w[0, 2].item(), expected_z.item(), places=5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
