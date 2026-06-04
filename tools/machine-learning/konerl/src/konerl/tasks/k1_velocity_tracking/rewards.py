import inspect
import math
from typing import Any, Callable
from click import command
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from konerl.scripts.AMP.optimizer import AMPOptimizer
from konerl.scripts.AMP.cache import AMPStateCache
from konerl.scripts.AMP.features import K1_AMP_ARM_JOINT_NAMES, K1_AMP_JOINT_NAMES


def get_yaw_from_quaternion(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw from a quaternion [w, x, y, z]."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Create a yaw-only quaternion [w, x, y, z] from a yaw angle."""
    half_yaw = yaw / 2.0
    w = torch.cos(half_yaw)
    z = torch.sin(half_yaw)
    x = torch.zeros_like(yaw)
    y = torch.zeros_like(yaw)
    return torch.stack([w, x, y, z], dim=-1)


class BoundedPenaltyWrapper:
    """Wraps both stateless functions and stateful classes to bound their penalties."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.std = cfg.params.get("std", cfg.params.get("sigma", 1.0))
        inner_func = cfg.params["func"]
        if inspect.isclass(inner_func):
            self.inner_callable = inner_func(cfg, env)
        else:
            self.inner_callable = inner_func

    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        inner_kwargs = {k: v for k, v in kwargs.items() if k not in ["func", "sigma", "std"]}

        raw_penalty = self.inner_callable(env, **inner_kwargs)
        return torch.exp(-torch.abs(raw_penalty) / self.std)


def bad_base_height(
    env: ManagerBasedRlEnv,
    limit_height: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Penalizes when the base height falls below the limit height."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < limit_height


def target_base_height(
    env: ManagerBasedRlEnv,
    target_height: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
):
    """Rewards when the base height is close to the target height."""
    asset: Entity = env.scene[asset_cfg.name]
    height_error = torch.abs(asset.data.root_link_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / std**2)


def base_height_below_l2(
    env: ManagerBasedRlEnv,
    threshold: float = 0.48,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous penalty for base height below a minimum threshold."""
    asset: Entity = env.scene[asset_cfg.name]
    height_error = torch.clamp(threshold - asset.data.root_link_pos_w[:, 2], min=0.0)
    return torch.square(height_error)


class TargetBaseHeightMean:
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self._env = env
        device = getattr(env, "device", "cpu")
        self._base_height_ema = torch.zeros(size=[env.num_envs], device=device)
        self._asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg",
            SceneEntityCfg("robot"),
        )

    def __call__(
        self,
        env,
        ema_alpha=0.8,
        std=0.25,
        target_height=0.5,
    ):
        asset: Entity = env.scene[self._asset_cfg.name]
        assert asset is not None

        self._base_height_ema *= 1 - ema_alpha
        self._base_height_ema += ema_alpha * asset.data.root_link_pos_w[:, 2]
        height_error = torch.abs(self._base_height_ema - target_height)
        return torch.exp(-height_error / std**2)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._base_height_ema.fill_(0.0)
        else:
            self._base_height_ema[env_ids] = 0.0


class ActionJerkL2:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        device = getattr(env, "device", "cpu")
        self._prev_prev_prev_action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=device)
        self._prev_prev_action = torch.zeros_like(self._prev_prev_prev_action)
        self._prev_action = torch.zeros_like(self._prev_prev_prev_action)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._prev_prev_prev_action.fill_(0.0)
            self._prev_prev_action.fill_(0.0)
            self._prev_action.fill_(0.0)
        else:
            self._prev_prev_prev_action[env_ids] = 0.0
            self._prev_prev_action[env_ids] = 0.0
            self._prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        current_action = env.action_manager.action
        jerk = current_action - 3.0 * self._prev_action + 3.0 * self._prev_prev_action - self._prev_prev_prev_action
        self._prev_prev_prev_action[:] = self._prev_prev_action
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = current_action

        return torch.sum(torch.square(jerk), dim=1)


def is_walking_env(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert command is not None
    assert hasattr(command, "is_walking_env")
    walking_env_flag = getattr(command, "is_walking_env", None)
    assert walking_env_flag is not None
    return walking_env_flag


def is_kicking_env(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert command is not None
    assert hasattr(command, "is_kicking_env")
    kicking_env_flag = getattr(command, "is_kicking_env", None)
    assert kicking_env_flag is not None
    return kicking_env_flag


def is_dribble_env(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert command is not None
    assert hasattr(command, "is_dribble_env")
    dribble_env_flag = getattr(command, "is_dribble_env", None)
    assert dribble_env_flag is not None
    return dribble_env_flag


def is_standing_env(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert command is not None
    assert hasattr(command, "is_standing_env")
    standing_env_flag = getattr(command, "is_standing_env", None)
    assert standing_env_flag is not None
    return standing_env_flag


def is_approach_env(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    assert command is not None
    approach_env_flag = getattr(command, "is_approach_env", None)
    if approach_env_flag is not None:
        return approach_env_flag
    command_tensor = env.command_manager.get_command(command_name)
    assert command_tensor is not None
    ball_free = command_tensor[:, 6] > 0.5
    kick = command_tensor[:, 7] > 0.5
    dribble = command_tensor[:, 8] > 0.5
    return ball_free & ~kick & ~dribble


def distance_to_ball_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_distance: float = 0.4,
    std: float = 0.25,
) -> torch.Tensor:
    """Rewards maintaining a useful base distance to the ball for approach/kick setup."""
    robot: Entity = env.scene[robot_cfg.name]
    assert robot is not None
    ball: Entity = env.scene[ball_cfg.name]
    assert ball is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    can_touch = command[:, 6] > 0.5

    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]

    dist = torch.norm(ball_pos - robot_pos, dim=-1)
    error = torch.abs(dist - target_distance)
    proximity = torch.exp(-error / std**2)
    reward = torch.zeros_like(proximity)

    active = is_approach_env(env, command_name) & can_touch
    return torch.where(active, proximity, reward)


class KickContactReward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.has_contacted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.has_contacted.fill_(False)
        else:
            self.has_contacted[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        sensor_name: str = "robot_ball_collision",
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_name]
        assert sensor is not None
        command = env.command_manager.get_command(command_name)
        assert command is not None

        active = is_kicking_env(env, command_name) & (command[:, 6] > 0.5)
        contact = (sensor.data.found[:, 0] > 0.5) & active
        first_contact = contact & ~self.has_contacted
        self.has_contacted.logical_or_(contact)
        return first_contact.float()


class KickVelocityReward:
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.env = env
        self.cfg = cfg
        self.remaining_steps = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self.latched_command_b = torch.zeros(env.num_envs, 3, dtype=torch.float, device=env.device)
        self.has_contacted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.remaining_steps.fill_(0.0)
            self.latched_command_b.fill_(0.0)
            self.has_contacted.fill_(False)
        else:
            self.remaining_steps[env_ids] = 0.0
            self.latched_command_b[env_ids] = 0.0
            self.has_contacted[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        sensor_name: str = "robot_ball_collision",
        std: float = 0.75,
        reward_window_steps: int = 20,
        min_command_speed: float = 0.1,
    ) -> torch.Tensor:
        "Rewards ball speed along the kick command for several frames after kick contact."
        sensor = env.scene.sensors[sensor_name]
        assert sensor is not None
        command = env.command_manager.get_command(command_name)
        assert command is not None
        robot: Entity = env.scene[robot_cfg.name]
        assert robot is not None
        ball: Entity = env.scene[ball_cfg.name]
        assert ball is not None

        can_touch = command[:, 6] > 0.5
        active = is_kicking_env(env, command_name) & can_touch
        command_b = command[:, 3:6]
        command_speed = torch.linalg.norm(command_b, dim=-1)
        contact = (sensor.data.found[:, 0] > 0.5) & active & (command_speed > min_command_speed)
        first_contact = contact & ~self.has_contacted
        self.has_contacted.logical_or_(contact)

        self.latched_command_b = torch.where(first_contact.unsqueeze(-1), command_b, self.latched_command_b)
        self.remaining_steps = torch.where(
            first_contact,
            torch.full_like(self.remaining_steps, float(reward_window_steps)),
            (self.remaining_steps - 1.0).clamp(min=0.0),
        )

        yaw = get_yaw_from_quaternion(robot.data.root_link_quat_w)
        yaw_only_quat_w = quat_from_yaw(yaw)
        ball_vel_heading = quat_apply(quat_inv(yaw_only_quat_w), ball.data.root_link_vel_w[:, :3])

        latched_speed = torch.linalg.norm(self.latched_command_b, dim=-1).clamp(min=min_command_speed)
        latched_dir = self.latched_command_b / latched_speed.unsqueeze(-1)
        projected_speed = torch.sum(ball_vel_heading * latched_dir, dim=-1).clamp(min=0.0)
        speed_error = projected_speed - latched_speed
        speed_gate = (projected_speed / latched_speed).clamp(min=0.0, max=1.0)
        reward = torch.exp(-torch.square(speed_error) / std**2) * speed_gate

        active_window = self.remaining_steps > 0.0
        if "log" in env.extras:
            active_count = active_window.float().sum().clamp(min=1.0)
            env.extras["log"]["Metrics/kick_projected_ball_speed"] = (
                projected_speed * active_window.float()
            ).sum() / active_count

        return reward * active_window.float()


class TrackLinearVelocityMean:
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self._env = env
        device = getattr(env, "device", "cpu")
        self._xy_command_ema = torch.zeros(size=[env.num_envs, 2], device=device)
        self._xy_velocity_ema = torch.zeros(size=[env.num_envs, 2], device=device)
        self._asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg",
            SceneEntityCfg("robot", site_names="Trunk"),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._xy_command_ema.fill_(0.0)
            self._xy_velocity_ema.fill_(0.0)
        else:
            self._xy_command_ema[env_ids] = 0.0
            self._xy_velocity_ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        ema_alpha: float = 0.8,
        std: float = 0.25,
    ) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        assert command is not None

        self._xy_command_ema *= 1 - ema_alpha
        self._xy_command_ema += ema_alpha * command[:, :2]

        asset: Entity = env.scene[self._asset_cfg.name]
        xy_velocity = asset.data.root_link_lin_vel_b[:, :2]

        self._xy_velocity_ema *= 1 - ema_alpha
        self._xy_velocity_ema += ema_alpha * xy_velocity

        xy_error = torch.sum(torch.square(self._xy_command_ema - self._xy_velocity_ema), dim=1)
        reward = torch.exp(-xy_error / std**2)

        return torch.where(is_kicking_env(env, command_name), torch.zeros_like(reward), reward)


class TrackAngularVelocityMean:
    def __init__(
        self,
        cfg: RewardTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self._env = env
        device = getattr(env, "device", "cpu")
        self._z_command_ema = torch.zeros(size=[env.num_envs], device=device)
        self._z_velocity_ema = torch.zeros(size=[env.num_envs], device=device)
        self._asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg",
            SceneEntityCfg("robot", site_names="Trunk"),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._z_command_ema.fill_(0.0)
            self._z_velocity_ema.fill_(0.0)
        else:
            self._z_command_ema[env_ids] = 0.0
            self._z_velocity_ema[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        ema_alpha: float = 0.8,
        std: float = 0.25,
    ) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        assert command is not None
        self._z_command_ema *= 1 - ema_alpha
        self._z_command_ema += ema_alpha * command[:, 2]

        asset: Entity = env.scene[self._asset_cfg.name]
        z_velocity = asset.data.root_link_ang_vel_b[:, 2]

        self._z_velocity_ema *= 1 - ema_alpha
        self._z_velocity_ema += ema_alpha * z_velocity

        z_error = torch.square(self._z_command_ema - self._z_velocity_ema)

        return torch.where(
            is_standing_env(env, command_name) | is_walking_env(env, command_name) | is_dribble_env(env, command_name),
            torch.exp(-z_error / std**2),
            torch.zeros_like(z_error),
        )


def ball_dribble_position(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_distance: float = 0.4,
    std: float = 0.3,
    in_range_dist: float = 1.0,
) -> torch.Tensor:
    """Rewards keeping the ball at a specific distance directly in front of the robot for dribbling, if free and in range."""
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    can_touch = command[:, 6] > 0.5

    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]

    dist_to_ball = torch.norm(ball_pos - robot_pos, dim=-1)
    in_range = dist_to_ball <= in_range_dist

    heading = robot.data.heading_w
    forward_vec = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)

    target_pos = robot_pos + forward_vec * target_distance
    error = torch.norm(ball_pos - target_pos, dim=-1)

    reward = torch.exp(-error / std**2)

    active = is_dribble_env(env, command_name) & can_touch & in_range
    return torch.where(active, reward, torch.zeros_like(reward))


def ball_approach_alignment(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Rewards moving towards the ball and facing it; penalizes turning away, applied when ball is not in range."""
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    robot_pos_w = robot.data.root_link_pos_w
    ball_pos_w = ball.data.root_link_pos_w

    relative_pos_w = ball_pos_w - robot_pos_w

    robot_quat_w = robot.data.root_link_quat_w
    relative_pos_b = quat_apply(quat_inv(robot_quat_w), relative_pos_w)

    angle_to_ball = torch.atan2(relative_pos_b[:, 1], relative_pos_b[:, 0])

    active = is_approach_env(env, command_name)

    alignment_score = 1.0 - (abs(angle_to_ball) / torch.pi)

    return alignment_score * active


def kick_base_velocity_to_setup(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    sensor_name: str = "robot_ball_collision",
    target_distance: float = 0.4,
    target_speed: float = 0.45,
    overspeed_std: float = 0.3,
    distance_margin: float = 0.08,
) -> torch.Tensor:
    """Reward controlled base motion to a command-dependent kick setup pose, not into the ball."""
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    sensor = env.scene.sensors[sensor_name]
    assert sensor is not None
    command = env.command_manager.get_command(command_name)
    assert command is not None

    command_b = command[:, 3:6].clone()
    command_b[:, 2] = 0.0
    command_dir_b = command_b / torch.linalg.norm(command_b, dim=-1).clamp(min=1e-6).unsqueeze(-1)

    yaw = get_yaw_from_quaternion(robot.data.root_link_quat_w)
    yaw_only_quat_w = quat_from_yaw(yaw)
    command_dir_w = quat_apply(yaw_only_quat_w, command_dir_b)[:, :2]

    robot_pos = robot.data.root_link_pos_w[:, :2]
    ball_pos = ball.data.root_link_pos_w[:, :2]
    setup_pos = ball_pos - command_dir_w * target_distance
    to_setup = setup_pos - robot_pos
    setup_error = torch.linalg.norm(to_setup, dim=-1)
    setup_dir = to_setup / setup_error.clamp(min=1e-6).unsqueeze(-1)

    robot_vel = robot.data.root_link_vel_w[:, :2]
    projected_speed = torch.sum(robot_vel * setup_dir, dim=-1).clamp(min=0.0)
    speed_up_reward = (projected_speed / target_speed).clamp(max=1.0)
    overspeed_penalty = torch.exp(-torch.square((projected_speed - target_speed).clamp(min=0.0)) / overspeed_std**2)
    reward = speed_up_reward * overspeed_penalty

    active = (
        is_kicking_env(env, command_name)
        & (command[:, 6] > 0.5)
        & (sensor.data.found[:, 0] <= 0.5)
        & (setup_error > distance_margin)
    )
    active_f = active.float()

    if "log" in env.extras:
        active_count = active_f.sum().clamp(min=1.0)
        env.extras["log"]["Metrics/kick_base_projected_speed"] = (projected_speed * active_f).sum() / active_count
        env.extras["log"]["Metrics/kick_base_setup_error"] = (setup_error * active_f).sum() / active_count

    return reward * active_f


class KickFootProgressReward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.best_distance = torch.full((env.num_envs,), float("inf"), dtype=torch.float, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.best_distance.fill_(float("inf"))
        else:
            self.best_distance[env_ids] = float("inf")

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        sensor_name: str = "robot_ball_collision",
        progress_scale: float = 0.05,
        proximity_std: float = 0.35,
    ) -> torch.Tensor:
        """Reward new best foot-ball distance only; standing near the ball pays zero."""
        robot: Entity = env.scene[robot_cfg.name]
        ball: Entity = env.scene[ball_cfg.name]
        sensor = env.scene.sensors[sensor_name]
        assert sensor is not None

        foot_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids, :]
        ball_pos_w = ball.data.root_link_pos_w[:, None, :]
        distances = torch.linalg.norm(foot_pos_w - ball_pos_w, dim=-1)
        min_distance = torch.min(distances, dim=1).values

        active = is_kicking_env(env, command_name) & (sensor.data.found[:, 0] <= 0.5)
        previous_best = self.best_distance.clone()
        initialized = torch.isfinite(previous_best)
        progress = torch.where(initialized, previous_best - min_distance, torch.zeros_like(min_distance)).clamp(min=0.0)
        proximity = torch.exp(-torch.square(min_distance) / proximity_std**2)
        reward = (progress / progress_scale).clamp(max=1.0) * proximity

        self.best_distance = torch.where(
            active,
            torch.minimum(self.best_distance, min_distance),
            min_distance,
        )

        active_f = active.float()
        if "log" in env.extras:
            active_count = active_f.sum().clamp(min=1.0)
            env.extras["log"]["Metrics/kick_min_foot_ball_distance"] = (min_distance * active_f).sum() / active_count

        return reward * active_f


def kick_foot_velocity_towards_ball(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    sensor_name: str = "robot_ball_collision",
    proximity_std: float = 0.45,
    min_command_speed: float = 0.1,
) -> torch.Tensor:
    """Dense kick shaping: swing a nearby foot along the commanded kick direction."""
    command = env.command_manager.get_command(command_name)
    assert command is not None
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    sensor = env.scene.sensors[sensor_name]
    assert sensor is not None

    command_b = command[:, 3:6]
    command_speed = torch.linalg.norm(command_b, dim=-1).clamp(min=min_command_speed)
    command_dir_b = command_b / command_speed.unsqueeze(-1)

    foot_vel_w = robot.data.site_lin_vel_w[:, robot_cfg.site_ids, :]
    root_quat_inv = quat_inv(robot.data.root_link_quat_w).unsqueeze(1).expand(-1, foot_vel_w.shape[1], -1)
    foot_vel_b = quat_apply(root_quat_inv.reshape(-1, 4), foot_vel_w.reshape(-1, 3)).reshape_as(foot_vel_w)
    projected_speed = torch.sum(foot_vel_b * command_dir_b[:, None, :], dim=-1).clamp(min=0.0)

    foot_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids, :]
    ball_pos_w = ball.data.root_link_pos_w[:, None, :]
    distances = torch.linalg.norm(foot_pos_w - ball_pos_w, dim=-1)
    proximity = torch.exp(-torch.square(distances) / proximity_std**2)
    per_foot_reward = (projected_speed / command_speed[:, None]).clamp(max=1.0) * proximity
    reward = torch.max(per_foot_reward, dim=1).values
    active = is_kicking_env(env, command_name) & (sensor.data.found[:, 0] <= 0.5)
    active_f = active.float()

    if "log" in env.extras:
        active_count = active_f.sum().clamp(min=1.0)
        env.extras["log"]["Metrics/kick_foot_projected_speed"] = (
            torch.max(projected_speed, dim=1).values * active_f
        ).sum() / active_count

    return reward * active_f


def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str = "twist",
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward feet air time."""
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    command = env.command_manager.get_command(command_name)
    assert command is not None
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(num_in_air, min=1)
    env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

    command_active = (command[:, :2] ** 2).sum(dim=-1).sqrt() > command_threshold

    active = is_standing_env(env, command_name) < 0.5 * command_active.float()

    return reward * active


class feet_swing_height:
    """Penalize deviation from target swing height, evaluated at landing."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        height_sensor = env.scene[cfg.params["height_sensor_name"]]
        assert isinstance(height_sensor, TerrainHeightSensor), (
            f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
        )
        num_feet = height_sensor.num_frames
        self.peak_heights = torch.zeros((env.num_envs, num_feet), device=env.device, dtype=torch.float32)
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        height_sensor_name: str,
        target_height: float,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        command = env.command_manager.get_command(command_name)
        assert command is not None
        height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
        foot_heights = height_sensor.data.heights
        in_air = contact_sensor.data.found == 0
        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )
        first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
        active = is_standing_env(env, command_name) < 0.5
        error = self.peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
        num_landings = torch.sum(first_contact.float())
        peak_heights_at_landing = self.peak_heights * first_contact.float()
        mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(num_landings, min=1)
        env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return cost


def contact_any(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalty signal for any contact reported by a contact sensor."""
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.found is None:
        return torch.zeros(env.num_envs, device=env.device)
    found = sensor.data.found.reshape(env.num_envs, -1)
    return torch.any(found > 0.5, dim=1).float()


def joint_default_pose_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mean squared deviation from configured default joint pose."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.mean(torch.square(error), dim=1)


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # [B, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            active = is_standing_env(env, command_name) < 0.5
            cost = cost * active
    return cost


class amp_reward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.env = env
        self.cfg = cfg

        self.amp_optimizer: AMPOptimizer | None = getattr(env, "amp_optimizer", None)

        joint_names = tuple(cfg.params["asset_cfg"].joint_names or ())
        if not joint_names:
            raise ValueError("AMP reward requires explicit asset_cfg.joint_names")
        self.cache = AMPStateCache(
            robot=env.scene[cfg.params["asset_cfg"].name],
            joint_names=joint_names,
            num_envs=env.num_envs,
            history_length=cfg.params["history_length"],
            device=env.device,
        )
        if not hasattr(env, "amp_cache"):
            setattr(env, "amp_cache", self.cache)

    def _cache(self, env: ManagerBasedRlEnv) -> AMPStateCache:
        cache = getattr(env, "amp_cache", self.cache)
        if cache is not self.cache:
            self.cache = cache
        return self.cache

    def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, history_length: int) -> torch.Tensor:
        if not hasattr(env, "amp_optimizer"):
            raise RuntimeError("AMP reward is configured but env.amp_optimizer is missing")
        if self.amp_optimizer is None:
            self.amp_optimizer = getattr(env, "amp_optimizer")

        assert self.amp_optimizer is not None, "AMP optimizer is not set in the environment"

        del asset_cfg, history_length
        cache = self._cache(env)
        cache.update()

        return self.amp_optimizer.calculate_amp_rewards(cache.history).squeeze(-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        cache = self._cache(self.env)
        if env_ids is None:
            cache.reset()
        else:
            cache.reset(env_ids)


def make_reward_cfg(
    *,
    amp: bool = False,
    amp_joint_names: tuple[str, ...] = K1_AMP_JOINT_NAMES,
    arm_default_pose: bool = False,
) -> dict[str, RewardTermCfg]:
    rewards = {
        "survival": RewardTermCfg(
            func=mdp.is_alive,
            weight=2,
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=0.2,
            params={
                "std": 0.5,
                "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            },
        ),
        "termination": RewardTermCfg(
            func=mdp.is_terminated,
            weight=-200.0,
        ),
        "target_base_height": RewardTermCfg(
            func=TargetBaseHeightMean,
            weight=0.2,
            params={
                "target_height": 0.55,
                "ema_alpha": 0.1,
                "std": math.sqrt(0.3),
            },
        ),
        # "low_base_height": RewardTermCfg(
        #     func=base_height_below_l2,
        #     weight=-50.0,
        #     params={"threshold": 0.48},
        # ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "self_collision": RewardTermCfg(
            func=contact_any,
            weight=-1.0,
            params={"sensor_name": "self_collision"},
        ),
        "non_feet_ground_contact": RewardTermCfg(
            func=contact_any,
            weight=-1.0,
            params={"sensor_name": "non_feet_ground_contact"},
        ),
        # "electrical_power_cost": RewardTermCfg(
        #     func=mdp.electrical_power_cost,
        #     weight=-0.003,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=(
        #             "Left_Hip_Pitch", "Right_Hip_Pitch",
        #             "Left_Hip_Roll", "Right_Hip_Roll",
        #             "Left_Hip_Yaw", "Right_Hip_Yaw",
        #             "Left_Knee_Pitch", "Right_Knee_Pitch",
        #             "Left_Ankle_Pitch", "Right_Ankle_Pitch",
        #             "Left_Ankle_Roll", "Right_Ankle_Roll"
        #         )),
        #     },
        # ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.005),
        "torque_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5),
        "velocity_tracking": RewardTermCfg(
            func=TrackLinearVelocityMean,
            weight=2.0,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.4),
                "ema_alpha": 0.2,
            },
        ),
        "velocity_tracking_ang": RewardTermCfg(
            func=TrackAngularVelocityMean,
            weight=2.0,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.4),
                "ema_alpha": 0.2,
            },
        ),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.05,
                "threshold_max": 0.45,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.05,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-5,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "ball_approach_alignment": RewardTermCfg(
            func=ball_approach_alignment,
            weight=0.5,
            params={"command_name": "twist"},
        ),
        "kick_ready_distance": RewardTermCfg(
            func=distance_to_ball_reward,
            weight=0.75,
            params={
                "command_name": "twist",
                "target_distance": 0.4,
                "std": 0.25,
            },
        ),
        "kick_base_velocity_to_setup": RewardTermCfg(
            func=kick_base_velocity_to_setup,
            weight=0.5,
            params={
                "command_name": "twist",
                "sensor_name": "robot_ball_collision",
                "target_distance": 0.4,
                "target_speed": 0.45,
                "overspeed_std": 0.3,
                "distance_margin": 0.08,
            },
        ),
        "kick_foot_progress": RewardTermCfg(
            func=KickFootProgressReward,
            weight=1.0,
            params={
                "command_name": "twist",
                "robot_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
                "sensor_name": "robot_ball_collision",
                "progress_scale": 0.05,
                "proximity_std": 0.35,
            },
        ),
        "kick_foot_velocity": RewardTermCfg(
            func=kick_foot_velocity_towards_ball,
            weight=2.0,
            params={
                "command_name": "twist",
                "robot_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
                "sensor_name": "robot_ball_collision",
                "proximity_std": 0.45,
            },
        ),
        "kick_contact": RewardTermCfg(
            func=KickContactReward,
            weight=4.0,
            params={
                "command_name": "twist",
                "sensor_name": "robot_ball_collision",
            },
        ),
        "kick_velocity": RewardTermCfg(
            func=KickVelocityReward,
            weight=8.0,
            params={
                "command_name": "twist",
                "sensor_name": "robot_ball_collision",
                "std": 0.75,
                "reward_window_steps": 30,
            },
        ),
    }

    if arm_default_pose:
        rewards["arm_default_pose"] = RewardTermCfg(
            func=joint_default_pose_l2,
            weight=-0.01,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=K1_AMP_ARM_JOINT_NAMES,
                    preserve_order=True,
                ),
            },
        )

    if amp:
        rewards["amp"] = RewardTermCfg(
            func=amp_reward,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=amp_joint_names),
                "history_length": 5,
            },
        )

    return rewards
