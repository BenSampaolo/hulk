import math
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

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


class KickDetector:
    def __init__(self, env: ManagerBasedRlEnv):
        self.last_step: int | None = None
        self.last_ball_vel_w = torch.zeros(env.num_envs, 3, dtype=torch.float, device=env.device)
        self.steps_since_ball_reset = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.remaining_reward_steps = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self.latched_command_b = torch.zeros(env.num_envs, 3, dtype=torch.float, device=env.device)
        self.has_detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.just_detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.ball_speed = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self.ball_direction_error = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self.projected_ball_speed = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.last_step = None
        self.steps_since_ball_reset[env_ids] = 0
        self.remaining_reward_steps[env_ids] = 0.0
        self.latched_command_b[env_ids] = 0.0
        self.has_detected[env_ids] = False
        self.just_detected[env_ids] = False
        self.ball_speed[env_ids] = 0.0
        self.ball_direction_error[env_ids] = 0.0
        self.projected_ball_speed[env_ids] = 0.0

    def notify_ball_reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.reset(env_ids)

    def update(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        robot_cfg: SceneEntityCfg,
        ball_cfg: SceneEntityCfg,
        sensor_name: str,
        foot_ball_distance_threshold: float,
        min_ball_speed_delta: float,
        min_ball_speed: float,
        min_foot_speed_towards_ball: float,
        min_steps_since_ball_reset: int,
        reward_window_steps: int,
    ) -> "KickDetector":
        step = getattr(env, "common_step_counter", None)
        if isinstance(step, torch.Tensor):
            step = int(step.item())
        if isinstance(step, int) and self.last_step == step:
            return self

        robot: Entity = env.scene[robot_cfg.name]
        ball: Entity = env.scene[ball_cfg.name]
        sensor = env.scene.sensors[sensor_name]
        assert sensor is not None
        command = env.command_manager.get_command(command_name)
        assert command is not None

        ball_vel_w = ball.data.root_link_vel_w[:, :3]
        ball_speed_delta = torch.linalg.norm(ball_vel_w - self.last_ball_vel_w, dim=-1)
        current_ball_speed = torch.linalg.norm(ball_vel_w, dim=-1)

        foot_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids, :]
        foot_vel_w = robot.data.site_lin_vel_w[:, robot_cfg.site_ids, :]
        ball_pos_w = ball.data.root_link_pos_w[:, None, :]
        foot_to_ball = ball_pos_w - foot_pos_w
        foot_ball_dist = torch.linalg.norm(foot_to_ball, dim=-1)
        min_foot_ball_dist = torch.min(foot_ball_dist, dim=1).values
        foot_to_ball_dir = foot_to_ball / foot_ball_dist.clamp(min=1e-6).unsqueeze(-1)
        foot_speed_towards_ball = torch.sum(foot_vel_w * foot_to_ball_dir, dim=-1).max(dim=1).values

        command_b = command[:, 3:6]
        command_speed = torch.linalg.norm(command_b, dim=-1)
        active = is_kicking_env(env, command_name) & (command[:, 6] > 0.5) & (command_speed > 0.1)
        contact = sensor.data.found[:, 0] > 0.5
        valid_foot = min_foot_ball_dist <= foot_ball_distance_threshold
        valid_ball_change = (ball_speed_delta >= min_ball_speed_delta) | (current_ball_speed >= min_ball_speed)
        valid_foot_motion = foot_speed_towards_ball >= min_foot_speed_towards_ball
        old_enough = self.steps_since_ball_reset >= min_steps_since_ball_reset
        detected = active & contact & valid_foot & valid_ball_change & valid_foot_motion & old_enough & ~self.has_detected

        self.just_detected = detected
        self.has_detected.logical_or_(detected)
        self.latched_command_b = torch.where(detected.unsqueeze(-1), command_b, self.latched_command_b)
        self.remaining_reward_steps = torch.where(
            detected,
            torch.full_like(self.remaining_reward_steps, float(reward_window_steps)),
            (self.remaining_reward_steps - 1.0).clamp(min=0.0),
        )

        yaw = get_yaw_from_quaternion(robot.data.root_link_quat_w)
        yaw_only_quat_w = quat_from_yaw(yaw)
        ball_vel_heading = quat_apply(quat_inv(yaw_only_quat_w), ball_vel_w)
        latched_speed = torch.linalg.norm(self.latched_command_b, dim=-1).clamp(min=0.1)
        latched_dir = self.latched_command_b / latched_speed.unsqueeze(-1)
        ball_speed = torch.linalg.norm(ball_vel_heading, dim=-1).clamp(min=1e-6)
        ball_dir = ball_vel_heading / ball_speed.unsqueeze(-1)
        cos_error = torch.sum(ball_dir * latched_dir, dim=-1).clamp(min=-1.0, max=1.0)

        self.ball_speed = ball_speed
        self.ball_direction_error = torch.acos(cos_error)
        self.projected_ball_speed = torch.sum(ball_vel_heading * latched_dir, dim=-1).clamp(min=0.0)
        self.last_ball_vel_w = ball_vel_w.clone()
        self.steps_since_ball_reset += 1
        if isinstance(step, int):
            self.last_step = step
        return self


def _kick_detector(env: ManagerBasedRlEnv) -> KickDetector:
    detector = getattr(env, "kick_detector", None)
    if detector is None:
        detector = KickDetector(env)
        setattr(env, "kick_detector", detector)
    return detector


def _updated_kick_detector(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    sensor_name: str = "robot_ball_collision",
    foot_ball_distance_threshold: float = 0.28,
    min_ball_speed_delta: float = 0.15,
    min_ball_speed: float = 0.25,
    min_foot_speed_towards_ball: float = 0.1,
    min_steps_since_ball_reset: int = 10,
    reward_window_steps: int = 30,
) -> KickDetector:
    return _kick_detector(env).update(
        env=env,
        command_name=command_name,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        sensor_name=sensor_name,
        foot_ball_distance_threshold=foot_ball_distance_threshold,
        min_ball_speed_delta=min_ball_speed_delta,
        min_ball_speed=min_ball_speed,
        min_foot_speed_towards_ball=min_foot_speed_towards_ball,
        min_steps_since_ball_reset=min_steps_since_ball_reset,
        reward_window_steps=reward_window_steps,
    )


def _kick_reference_pose(
    env: ManagerBasedRlEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    distance: float,
    lateral_offset: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    command_xy_b = command[:, 3:5]
    command_xy_b = command_xy_b / torch.linalg.norm(command_xy_b, dim=-1, keepdim=True).clamp(min=1e-6)
    yaw = get_yaw_from_quaternion(robot.data.root_link_quat_w)
    yaw_only_quat_w = quat_from_yaw(yaw)
    command_dir_w = quat_apply(
        yaw_only_quat_w,
        torch.cat((command_xy_b, torch.zeros(command_xy_b.shape[0], 1, device=command_xy_b.device)), dim=-1),
    )[:, :2]
    command_dir_w = command_dir_w / torch.linalg.norm(command_dir_w, dim=-1, keepdim=True).clamp(min=1e-6)
    perp_w = torch.stack((-command_dir_w[:, 1], command_dir_w[:, 0]), dim=-1)

    robot_pos = robot.data.root_link_pos_w[:, :2]
    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_from_ball = robot_pos - ball_pos
    side = torch.sign(torch.sum(robot_from_ball * perp_w, dim=-1, keepdim=True))
    side = torch.where(side == 0.0, torch.ones_like(side), side)
    reference_pos = ball_pos - command_dir_w * distance + perp_w * side * lateral_offset
    return reference_pos, command_dir_w, perp_w


def _kick_setup_state(
    env: ManagerBasedRlEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    distance: float,
    lateral_offset: float,
    ready_distance: float,
    ready_yaw_error: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    setup_pos, command_dir_w, perp_w = _kick_reference_pose(
        env, command_name, robot_cfg, ball_cfg, distance, lateral_offset
    )
    robot_pos = robot.data.root_link_pos_w[:, :2]
    ball_pos = ball.data.root_link_pos_w[:, :2]
    setup_error = torch.linalg.norm(setup_pos - robot_pos, dim=-1)

    ball_rel = ball_pos - robot_pos
    target_yaw = torch.atan2(ball_rel[:, 1], ball_rel[:, 0])
    heading = robot.data.heading_w
    yaw_error = torch.atan2(torch.sin(target_yaw - heading), torch.cos(target_yaw - heading))
    ready = (setup_error < ready_distance) & (torch.abs(yaw_error) < ready_yaw_error)
    return setup_pos, command_dir_w, perp_w, setup_error, yaw_error, ready


class KickDirectionReward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        _kick_detector(env)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # The detector is reset by ball-reset events and environment resets through other terms.
        del env_ids

    def __call__(self, env: ManagerBasedRlEnv, std: float = 0.35, **detector_kwargs) -> torch.Tensor:
        detector = _updated_kick_detector(env, **detector_kwargs)
        reward = (torch.exp(-torch.square(detector.ball_direction_error) / std**2) - 0.5) * 2.0
        return reward * (detector.remaining_reward_steps > 0.0).float()


class KickVelocityReward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        _kick_detector(env)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids

    def __call__(self, env: ManagerBasedRlEnv, std: float = 0.75, direction_std: float = 0.35, **detector_kwargs) -> torch.Tensor:
        detector = _updated_kick_detector(env, **detector_kwargs)
        target_speed = torch.linalg.norm(detector.latched_command_b, dim=-1).clamp(min=0.1)
        speed_reward = torch.exp(-torch.square(detector.projected_ball_speed - target_speed) / std**2)
        direction_gate = torch.exp(-torch.square(detector.ball_direction_error) / direction_std**2)
        return speed_reward * direction_gate * (detector.remaining_reward_steps > 0.0).float()


def kick_setup_and_strike_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    distance: float = 0.4,
    lateral_offset: float = 0.15,
    target_speed: float = 0.6,
    ready_distance: float = 0.18,
    ready_yaw_error: float = 0.7,
    kick_through_distance: float = 0.05,
    setup_std: float = 0.25,
    yaw_std: float = 0.7,
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    setup_pos, command_dir_w, _, setup_error, yaw_error, ready = _kick_setup_state(
        env, command_name, robot_cfg, ball_cfg, distance, lateral_offset, ready_distance, ready_yaw_error
    )

    ball_pos = ball.data.root_link_pos_w[:, :2]
    kick_target = ball_pos + command_dir_w * kick_through_distance
    target_pos = torch.where(ready.unsqueeze(-1), kick_target, setup_pos)
    to_target = target_pos - robot.data.root_link_pos_w[:, :2]
    target_error = torch.linalg.norm(to_target, dim=-1)
    target_dir = to_target / target_error.clamp(min=1e-6).unsqueeze(-1)
    projected_speed = torch.sum(robot.data.root_link_vel_w[:, :2] * target_dir, dim=-1).clamp(min=0.0)
    move_reward = (projected_speed / target_speed).clamp(max=1.0)

    yaw_reward = torch.exp(-torch.square(yaw_error) / yaw_std**2)
    setup_reward = torch.exp(-torch.square(setup_error) / setup_std**2) * yaw_reward
    approach_reward = 0.5 * move_reward * yaw_reward + 0.5 * setup_reward
    through_ball_reward = move_reward * yaw_reward

    active = is_kicking_env(env, command_name)
    return torch.where(ready, through_ball_reward, approach_reward) * active.float()


def kick_ball_avoidance_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    reference_robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance: float = 0.4,
    lateral_offset: float = 0.15,
    std: float = 0.25,
    ready_distance: float = 0.18,
    ready_yaw_error: float = 0.7,
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    _, _, _, _, _, ready = _kick_setup_state(
        env, command_name, reference_robot_cfg, ball_cfg, distance, lateral_offset, ready_distance, ready_yaw_error
    )
    foot_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids, :]
    ball_pos_w = ball.data.root_link_pos_w[:, None, :]
    min_distance = torch.linalg.norm(foot_pos_w - ball_pos_w, dim=-1).min(dim=1).values
    active = is_kicking_env(env, command_name) & ~ready
    return torch.exp(-torch.square(min_distance) / std**2) * active.float()


def kick_pose_overshoot_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    distance: float = 0.4,
    lateral_offset: float = 0.15,
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    _, command_dir_w, _ = _kick_reference_pose(env, command_name, robot_cfg, ball_cfg, distance, lateral_offset)
    robot_to_ball = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    behind_ball = torch.sum(robot_to_ball * command_dir_w, dim=-1)
    active = is_kicking_env(env, command_name)
    return (behind_ball < 0.0).float() * active.float()


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
        "kick_setup_and_strike": RewardTermCfg(
            func=kick_setup_and_strike_reward,
            weight=1.0,
            params={
                "command_name": "twist",
                "distance": 0.4,
                "lateral_offset": 0.15,
                "target_speed": 0.6,
                "ready_distance": 0.18,
                "ready_yaw_error": 0.7,
                "kick_through_distance": 0.05,
                "setup_std": 0.25,
                "yaw_std": 0.7,
            },
        ),
        "kick_ball_avoidance": RewardTermCfg(
            func=kick_ball_avoidance_reward,
            weight=-3.0,
            params={
                "command_name": "twist",
                "distance": 0.4,
                "lateral_offset": 0.15,
                "std": 0.25,
                "ready_distance": 0.18,
                "ready_yaw_error": 0.7,
            },
        ),
        "kick_pose_overshoot_penalty": RewardTermCfg(
            func=kick_pose_overshoot_penalty,
            weight=-20.0,
            params={
                "command_name": "twist",
                "distance": 0.5,
                "lateral_offset": 0.15,
            },
        ),
        "kick_direction": RewardTermCfg(
            func=KickDirectionReward,
            weight=4.0,
            params={
                "command_name": "twist",
                "sensor_name": "robot_ball_collision",
                "robot_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
                "std": 0.35,
                "foot_ball_distance_threshold": 0.28,
                "min_ball_speed_delta": 0.15,
                "min_ball_speed": 0.25,
                "min_foot_speed_towards_ball": 0.1,
                "min_steps_since_ball_reset": 10,
                "reward_window_steps": 30,
            },
        ),
        "kick_velocity": RewardTermCfg(
            func=KickVelocityReward,
            weight=10.0,
            params={
                "command_name": "twist",
                "sensor_name": "robot_ball_collision",
                "robot_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
                "std": 0.75,
                "direction_std": 0.35,
                "foot_ball_distance_threshold": 0.28,
                "min_ball_speed_delta": 0.15,
                "min_ball_speed": 0.25,
                "min_foot_speed_towards_ball": 0.1,
                "min_steps_since_ball_reset": 10,
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
            weight=2.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=amp_joint_names),
                "history_length": 5,
            },
        )

    return rewards
