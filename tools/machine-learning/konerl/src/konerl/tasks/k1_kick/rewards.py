from __future__ import annotations

import math
from typing import Any

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse


def _command_term(env: ManagerBasedRlEnv, command_name: str = "kick") -> Any:
    return env.command_manager.get_term(command_name)


def _kick_direction_w(env: ManagerBasedRlEnv, command_name: str = "kick") -> torch.Tensor:
    term = _command_term(env, command_name)
    direction = term.kick_direction_w[:, :2]
    return direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)


def _max_ball_speed(env: ManagerBasedRlEnv, command_name: str = "kick") -> torch.Tensor:
    term = _command_term(env, command_name)
    return term.max_ball_speed.clamp(min=0.1)


def _ball_pos_b(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    rel_pos_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return quat_apply_inverse(robot.data.root_link_quat_w, rel_pos_w)


def _robot_forward_w(robot: Entity) -> torch.Tensor:
    forward_b = torch.zeros(robot.data.root_link_quat_w.shape[0], 3, device=robot.data.root_link_quat_w.device)
    forward_b[:, 0] = 1.0
    forward_w = quat_apply(robot.data.root_link_quat_w, forward_b)[:, :2]
    return forward_w / torch.linalg.norm(forward_w, dim=-1, keepdim=True).clamp(min=1e-6)


def contact_any(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    if sensor.data.found is None:
        return torch.zeros(env.num_envs, device=env.device)
    found = sensor.data.found.reshape(env.num_envs, -1)
    return torch.any(found > 0.5, dim=1).float()


def move_to_ball_reward(
    env: ManagerBasedRlEnv,
    too_fast_threshold: float = 0.7,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    ball_pos_b = _ball_pos_b(env, robot_cfg, ball_cfg)[:, :2]
    ball_dir_b = ball_pos_b / torch.linalg.norm(ball_pos_b, dim=-1, keepdim=True).clamp(min=1e-6)
    speed_toward_ball = torch.sum(robot.data.root_link_lin_vel_b[:, :2] * ball_dir_b, dim=-1)
    return (speed_toward_ball / too_fast_threshold).clamp(min=0.0, max=1.0)


def orient_to_ball_reward(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    ball_pos_b = _ball_pos_b(env, robot_cfg, ball_cfg)
    ball_angle = torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])
    return torch.exp(-torch.abs(ball_angle))


def ball_height_reward(env: ManagerBasedRlEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    ball: Entity = env.scene[ball_cfg.name]
    return ball.data.root_link_pos_w[:, 2]


def ball_speed_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "kick",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    angle_tolerance: float = math.radians(15.0),
    moving_speed_threshold: float = 0.05,
) -> torch.Tensor:
    ball: Entity = env.scene[ball_cfg.name]
    ball_vel_w = ball.data.root_link_vel_w[:, :3]
    speed = torch.linalg.norm(ball_vel_w, dim=-1)
    xy_speed = torch.linalg.norm(ball_vel_w[:, :2], dim=-1)
    kick_dir_w = _kick_direction_w(env, command_name)
    cos_angle = torch.sum(ball_vel_w[:, :2] * kick_dir_w, dim=-1) / xy_speed.clamp(min=1e-6)
    correct_dir = cos_angle > math.cos(angle_tolerance)
    speed_reward = torch.minimum(speed / _max_ball_speed(env, command_name), torch.ones_like(speed)) ** 2
    reward = torch.where(correct_dir, speed_reward, torch.full_like(speed_reward, -0.05))
    return reward * (speed > moving_speed_threshold).float()


def kick_dir_bonus_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "kick",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    angle_tolerance: float = math.radians(5.0),
    moving_speed_threshold: float = 0.2,
) -> torch.Tensor:
    ball: Entity = env.scene[ball_cfg.name]
    ball_vel_w = ball.data.root_link_vel_w[:, :3]
    speed = torch.linalg.norm(ball_vel_w, dim=-1)
    xy_speed = torch.linalg.norm(ball_vel_w[:, :2], dim=-1)
    cos_angle = torch.sum(ball_vel_w[:, :2] * _kick_direction_w(env, command_name), dim=-1) / xy_speed.clamp(min=1e-6)
    return ((speed > moving_speed_threshold) & (cos_angle > math.cos(angle_tolerance))).float()


def zero_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def orient_to_kick_dir_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "kick",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    dist = torch.linalg.norm(_ball_pos_b(env, robot_cfg, ball_cfg)[:, :2], dim=-1)
    proximity_weight = (1.0 - dist / 1.0).clamp(min=0.0, max=1.0)
    alignment = torch.sum(_robot_forward_w(robot) * _kick_direction_w(env, command_name), dim=-1).clamp(min=0.0)
    return alignment * proximity_weight


def sidestep_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "kick",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    ball_pos_b = _ball_pos_b(env, robot_cfg, ball_cfg)
    dist = torch.linalg.norm(ball_pos_b[:, :2], dim=-1)
    in_circle = (dist > 0.35) & (dist < 0.58)
    fwd_xy = _robot_forward_w(robot)
    kick_dir = _kick_direction_w(env, command_name)
    not_aligned = torch.sum(fwd_xy * kick_dir, dim=-1) < math.cos(math.radians(10.0))
    cross = fwd_xy[:, 0] * kick_dir[:, 1] - fwd_xy[:, 1] * kick_dir[:, 0]
    correct_lateral = -torch.sign(cross) * robot.data.root_link_lin_vel_b[:, 1]
    lateral_speed = (correct_lateral.clamp(min=0.0) / 0.2).clamp(max=1.0)
    facing_ball = torch.abs(torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])) < math.radians(45.0)
    return lateral_speed * in_circle.float() * not_aligned.float() * facing_ball.float()


def aligned_approach_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "kick",
    too_fast_threshold: float = 0.7,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    ball_pos_b = _ball_pos_b(env, robot_cfg, ball_cfg)
    fwd_xy = _robot_forward_w(robot)
    kick_dir = _kick_direction_w(env, command_name)
    facing_kick = torch.sum(fwd_xy * kick_dir, dim=-1) > math.cos(math.radians(45.0))
    facing_ball = torch.abs(torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])) < math.radians(45.0)
    forward_speed = (robot.data.root_link_lin_vel_b[:, 0] / too_fast_threshold).clamp(min=0.0, max=1.0)
    return forward_speed * (facing_kick & facing_ball).float()


def wrong_approach_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "kick",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    ball_pos_b = _ball_pos_b(env, robot_cfg, ball_cfg)
    dist = torch.linalg.norm(ball_pos_b[:, :2], dim=-1)
    fwd_xy = _robot_forward_w(robot)
    kick_dir = _kick_direction_w(env, command_name)
    not_facing_kick = torch.sum(fwd_xy * kick_dir, dim=-1) < math.cos(math.radians(45.0))
    not_facing_ball = torch.abs(torch.atan2(ball_pos_b[:, 1], ball_pos_b[:, 0])) > math.radians(45.0)
    in_sweet_spot = (dist > 0.35) & (dist < 0.58)
    return (~in_sweet_spot & (not_facing_kick | not_facing_ball)).float()


def too_fast_penalty(
    env: ManagerBasedRlEnv,
    threshold: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    speed = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=-1)
    return (speed > threshold).float()


def orientation_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity_b = quat_apply_inverse(asset.data.root_link_quat_w, asset.data.gravity_vec_w)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=-1)


def ang_vel_xy_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_w[:, :2]), dim=-1)


def joint_torques_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=-1)


def joint_energy_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    actuator_force = asset.data.actuator_force[:, asset_cfg.actuator_ids]
    dim = min(joint_vel.shape[1], actuator_force.shape[1])
    return torch.sum(torch.abs(joint_vel[:, :dim]) * torch.abs(actuator_force[:, :dim]), dim=-1)


def _joint_names_for_ids(asset: Entity, joint_ids: Any) -> tuple[str, ...]:
    joint_names = asset.joint_names
    if isinstance(joint_ids, slice):
        return joint_names[joint_ids]
    if isinstance(joint_ids, torch.Tensor):
        joint_ids = joint_ids.detach().cpu().tolist()
    if isinstance(joint_ids, int):
        joint_ids = [joint_ids]
    return tuple(joint_names[i] for i in joint_ids)


def _pose_weights_for_joint_names(joint_names: tuple[str, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = [0.01 if "hip_pitch" in name.lower() or "knee_pitch" in name.lower() else 1.0 for name in joint_names]
    return torch.tensor(weights, device=device, dtype=dtype)


def joint_default_pose_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    weights = _pose_weights_for_joint_names(_joint_names_for_ids(asset, joint_ids), error.device, error.dtype)
    return torch.sum(torch.square(error) * weights, dim=-1)


def _get_env_step_index(env: Any) -> int | None:
    for attr in ("common_step_counter", "global_step_counter", "step_counter"):
        if not hasattr(env, attr):
            continue
        value = getattr(env, attr)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.item()
        if isinstance(value, int):
            return value
    return None


def _mirror_permutation_and_sign(joint_names: tuple[str, ...]) -> tuple[list[int], list[float]]:
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    permutation: list[int] = []
    signs: list[float] = []
    for i, name in enumerate(joint_names):
        partner = name
        if "Left" in name:
            partner = name.replace("Left", "Right")
        elif "Right" in name:
            partner = name.replace("Right", "Left")
        permutation.append(name_to_idx.get(partner, i))
        signs.append(-1.0)
    return permutation, signs


class ActionSymmetryPenalty:
    def __init__(self, joint_names: tuple[str, ...], max_delay_steps: int = 25):
        self.max_delay_steps = max_delay_steps
        permutation, signs = _mirror_permutation_and_sign(joint_names)
        self._permutation = torch.tensor(permutation, dtype=torch.long)
        self._signs = torch.tensor(signs)
        self._state_by_env: dict[int, dict[str, Any]] = {}

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        for state in self._state_by_env.values():
            state["history"][env_ids] = 0.0

    def _ensure_state(self, env: ManagerBasedRlEnv, action: torch.Tensor) -> dict[str, Any]:
        env_key = id(env)
        expected_shape = (env.num_envs, self.max_delay_steps, action.shape[1])
        state = self._state_by_env.get(env_key)
        if state is None or state["history"].shape != expected_shape or state["history"].device != action.device:
            state = {
                "history": torch.zeros(expected_shape, device=action.device, dtype=action.dtype),
                "last_step": None,
            }
            self._state_by_env[env_key] = state
        return state

    def _update_history(self, env: ManagerBasedRlEnv, action: torch.Tensor, state: dict[str, Any]) -> None:
        step_index = _get_env_step_index(env)
        if step_index is not None and state["last_step"] == step_index:
            return

        episode_length_buf = getattr(env, "episode_length_buf", None)
        if isinstance(episode_length_buf, torch.Tensor) and episode_length_buf.shape[0] == env.num_envs:
            reset_ids = torch.nonzero(episode_length_buf == 0, as_tuple=False).flatten()
            if len(reset_ids) > 0:
                state["history"][reset_ids] = 0.0

        state["history"][:, 1:] = state["history"][:, :-1].clone()
        state["history"][:, 0] = action
        state["last_step"] = step_index

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "kick",
        ball_distance_threshold: float = 0.5,
    ) -> torch.Tensor:
        action = env.action_manager.action
        if action.shape[1] != len(self._permutation):
            return torch.zeros(env.num_envs, device=env.device)

        state = self._ensure_state(env, action)
        self._update_history(env, action, state)

        term = _command_term(env, command_name)
        gait_frequency = getattr(term, "gait_frequency", None)
        if gait_frequency is None:
            return torch.zeros(env.num_envs, device=env.device)

        step_dt = getattr(env, "step_dt", 0.0)
        if isinstance(step_dt, torch.Tensor):
            step_dt = step_dt.item() if step_dt.numel() == 1 else 0.0
        step_dt = max(float(step_dt), 1e-6)
        half_period_steps = torch.round(0.5 / (step_dt * gait_frequency.clamp(min=1e-6))).long()
        half_period_steps = half_period_steps.clamp(min=0, max=self.max_delay_steps - 1)

        env_ids = torch.arange(env.num_envs, device=action.device)
        delayed_action = state["history"][env_ids, half_period_steps]
        permutation = self._permutation.to(device=action.device)
        signs = self._signs.to(device=action.device, dtype=action.dtype)
        mirrored_action = action[:, permutation] * signs
        error = torch.sum(torch.square(mirrored_action - delayed_action), dim=-1)

        episode_length_buf = getattr(env, "episode_length_buf", None)
        if isinstance(episode_length_buf, torch.Tensor) and episode_length_buf.shape[0] == env.num_envs:
            has_data = episode_length_buf >= half_period_steps
        else:
            step_index = _get_env_step_index(env)
            has_data = torch.full((env.num_envs,), step_index is None, device=env.device, dtype=torch.bool)
            if step_index is not None:
                has_data = torch.full((env.num_envs,), step_index >= int(half_period_steps.max()), device=env.device)

        far_from_ball = torch.linalg.norm(_ball_pos_b(env)[:, :2], dim=-1) > ball_distance_threshold
        return error * has_data.float() * far_from_ball.float()


def _gait_rz(phase: torch.Tensor, swing_height: float) -> torch.Tensor:
    x = (phase + math.pi) / (2.0 * math.pi)

    def cubic_bezier_interpolation(y_start: float, y_end: float, t: torch.Tensor) -> torch.Tensor:
        bezier = t**3 + 3.0 * (t**2 * (1.0 - t))
        return y_start + (y_end - y_start) * bezier

    stance = cubic_bezier_interpolation(0.0, swing_height, 2.0 * x)
    swing = cubic_bezier_interpolation(swing_height, 0.0, 2.0 * x - 1.0)
    return torch.where(x <= 0.5, stance, swing)


def feet_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    threshold_min: float = 0.2,
    threshold_max: float = 0.5,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    last_air_time = sensor.data.last_air_time
    if last_air_time is None:
        return torch.zeros(env.num_envs, device=env.device)
    first_contact = sensor.compute_first_contact(env.step_dt).float()
    air_time = (last_air_time - threshold_min) * first_contact
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    return torch.sum(air_time, dim=-1)


def feet_slip_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    if contact_sensor.data.found is None:
        return torch.zeros(env.num_envs, device=env.device)
    contact = (contact_sensor.data.found > 0).float().reshape(env.num_envs, -1)
    base_speed_xy = torch.linalg.norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1)
    return torch.sum(base_speed_xy[:, None] * contact, dim=-1)


def feet_phase_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "kick",
    max_foot_height: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
) -> torch.Tensor:
    term = _command_term(env, command_name)
    phase = getattr(term, "phase", None)
    if phase is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    target_z = _gait_rz(phase, swing_height=max_foot_height)
    error = torch.sum(torch.square(torch.clamp(target_z - foot_z, min=0.0)), dim=-1)
    return torch.exp(-error / 0.01)


def feet_level_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids]
    foot_quat_w = asset.data.site_quat_w[:, asset_cfg.site_ids]
    normal_b = torch.zeros_like(foot_pos_w)
    normal_b[..., 2] = 1.0
    foot_normal_w = quat_apply(
        foot_quat_w.reshape(-1, 4),
        normal_b.reshape(-1, 3),
    ).reshape_as(normal_b)
    pitch_error = torch.square(foot_normal_w[..., 0])

    rel_pos_w = foot_pos_w - asset.data.root_link_pos_w[:, None, :]
    root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(-1, rel_pos_w.shape[1], -1)
    foot_pos_b = quat_apply_inverse(
        root_quat_w.reshape(-1, 4),
        rel_pos_w.reshape(-1, 3),
    ).reshape_as(rel_pos_w)
    foot_forward = foot_pos_b[..., 0]
    weight = torch.exp(-torch.square(foot_forward) / (0.033**2))
    return torch.sum(pitch_error * weight, dim=-1)


def make_reward_cfg(controlled_joint_names: tuple[str, ...] | None = None) -> dict[str, RewardTermCfg]:
    foot_cfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    all_joints_cfg = SceneEntityCfg(
        "robot",
        joint_names=controlled_joint_names if controlled_joint_names is not None else (".*",),
        preserve_order=controlled_joint_names is not None,
    )
    all_actuators_cfg = SceneEntityCfg("robot", actuator_names=".*")
    symmetry_penalty = ActionSymmetryPenalty(controlled_joint_names or ())

    return {
        # Zero-scale PiPlus metrics.
        "lin_vel_x": RewardTermCfg(func=zero_reward, weight=0.0),
        "stop_for_kick": RewardTermCfg(func=zero_reward, weight=0.0),
        "ball_proximity": RewardTermCfg(func=zero_reward, weight=0.0),
        "ball_travel": RewardTermCfg(func=zero_reward, weight=0.0),
        "kick_foot_velocity": RewardTermCfg(func=zero_reward, weight=0.0),
        "kick_motion": RewardTermCfg(func=zero_reward, weight=0.0),
        "kick_dir_bonus": RewardTermCfg(
            func=kick_dir_bonus_reward,
            weight=0.0,
            params={"command_name": "kick"},
        ),
        "ball_too_fast": RewardTermCfg(func=zero_reward, weight=0.0),
        "lin_vel_z": RewardTermCfg(func=zero_reward, weight=0.0),
        "base_height": RewardTermCfg(func=zero_reward, weight=0.0),
        "feet_clearance": RewardTermCfg(func=zero_reward, weight=0.0),
        "feet_height": RewardTermCfg(func=zero_reward, weight=0.0),
        "alive": RewardTermCfg(func=zero_reward, weight=0.0),
        "stand_still": RewardTermCfg(func=zero_reward, weight=0.0),
        "joint_deviation_hip": RewardTermCfg(func=zero_reward, weight=0.0),
        "joint_deviation_knee": RewardTermCfg(func=zero_reward, weight=0.0),
        # Nonzero PiPlus terms.
        "move_to_ball": RewardTermCfg(func=move_to_ball_reward, weight=1.0),
        "orient_to_ball": RewardTermCfg(func=orient_to_ball_reward, weight=0.5),
        "ball_height": RewardTermCfg(func=ball_height_reward, weight=0.3),
        "ball_speed": RewardTermCfg(
            func=ball_speed_reward,
            weight=10.0,
            params={"command_name": "kick"},
        ),
        "orient_to_kick_dir": RewardTermCfg(
            func=orient_to_kick_dir_reward,
            weight=0.3,
            params={"command_name": "kick"},
        ),
        "sidestep": RewardTermCfg(func=sidestep_reward, weight=0.3, params={"command_name": "kick"}),
        "aligned_approach": RewardTermCfg(
            func=aligned_approach_reward,
            weight=0.5,
            params={"command_name": "kick"},
        ),
        "wrong_approach": RewardTermCfg(
            func=wrong_approach_penalty,
            weight=-0.1,
            params={"command_name": "kick"},
        ),
        "too_fast": RewardTermCfg(func=too_fast_penalty, weight=-2.0, params={"threshold": 0.7}),
        "symmetry": RewardTermCfg(
            func=symmetry_penalty,
            weight=-0.05,
            params={"command_name": "kick", "ball_distance_threshold": 0.5},
        ),
        "ang_vel_xy": RewardTermCfg(func=ang_vel_xy_l2, weight=-0.15),
        "orientation": RewardTermCfg(func=orientation_l2, weight=-1.0),
        "torques": RewardTermCfg(
            func=joint_torques_abs,
            weight=-1.0e-3,
            params={"asset_cfg": all_actuators_cfg},
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
        "energy": RewardTermCfg(func=joint_energy_abs, weight=-2.0e-3, params={"asset_cfg": all_actuators_cfg}),
        "feet_air_time": RewardTermCfg(
            func=feet_air_time_reward,
            weight=4.0,
            params={"sensor_name": "feet_ground_contact", "threshold_min": 0.2, "threshold_max": 0.5},
        ),
        "feet_slip": RewardTermCfg(
            func=feet_slip_penalty,
            weight=-0.25,
            params={"sensor_name": "feet_ground_contact", "asset_cfg": foot_cfg},
        ),
        "feet_phase": RewardTermCfg(
            func=feet_phase_reward,
            weight=1.0,
            params={"command_name": "kick", "max_foot_height": 0.08, "asset_cfg": foot_cfg},
        ),
        "feet_level": RewardTermCfg(func=feet_level_penalty, weight=-10.0, params={"asset_cfg": foot_cfg}),
        "termination": RewardTermCfg(func=mdp.is_terminated, weight=-10.0),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "pose": RewardTermCfg(func=joint_default_pose_l2, weight=-1.0, params={"asset_cfg": all_joints_cfg}),
    }
