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
            "asset_cfg", SceneEntityCfg("robot"),
        )
    
    def __call__(
            self, 
            env, 
            ema_alpha = 0.8, 
            std = 0.25,
            target_height = 0.5,
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
        self._prev_prev_prev_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=device
        )
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
        jerk = (
            current_action
            - 3.0 * self._prev_action
            + 3.0 * self._prev_prev_action
            - self._prev_prev_prev_action
        )
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

def distance_to_ball_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_distance: float = 0.4,
    std: float = 0.25,
) -> torch.Tensor:
    """Rewards maintaining an exact distance to the ball when the ball is not free."""
    robot: Entity = env.scene[robot_cfg.name]
    assert robot is not None
    ball: Entity = env.scene[ball_cfg.name]
    assert ball is not None
    
    command = env.command_manager.get_command(command_name)
    assert command is not None
    
    can_touch = command[:, 6] > 0.5
    not_free = ~can_touch
    
    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    
    dist = torch.norm(ball_pos - robot_pos, dim=-1)
    error = torch.abs(dist - target_distance)
    proximity = torch.exp(-error / std**2)
    reward = torch.zeros_like(proximity)
    
    active = (is_kicking_env(env, command_name) | is_dribble_env(env, command_name)) & not_free
    return torch.where(active, proximity, reward)

def kick_contact_reward(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    sensor_name: str = "robot_ball_collision",
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    assert sensor is not None
    command = env.command_manager.get_command(command_name)
    assert command is not None

    can_touch = command[:, 6] > 0.5
    active = is_kicking_env(env, command_name) | is_dribble_env(env, command_name)
    contact = ((sensor.data.found[:, 0] > 0.5) & active).float()

    return torch.where(can_touch, contact, -contact)

class KickVelocityReward:
    def __init__(
        self, 
        cfg: RewardTermCfg, 
        env: ManagerBasedRlEnv,
    ):
        self.env = env
        self.cfg = cfg
        self.active: torch.Tensor = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.active.fill_(0.0)
        else:
            self.active[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str = "twist",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        sensor_name: str = "robot_ball_collision",
        std: float = 5.0
    ) -> torch.Tensor:
        "Rewards the ball's velocity in the commanded direction when contact is made during a kick"
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
        contact = (sensor.data.found[:, 0] > 0.5) & active

        self.active = torch.where(contact, torch.ones_like(self.active) * 5, (self.active - 1.0).clamp(min=-1.0))

        trunk_quat_w = robot.data.root_link_quat_w
        ball_vel_w = ball.data.root_link_vel_w[:, :3]
        
        yaw = get_yaw_from_quaternion(trunk_quat_w)
        yaw_only_quat_w = quat_from_yaw(yaw)
        
        ball_vel_heading = quat_apply(quat_inv(yaw_only_quat_w), ball_vel_w)

        error = torch.norm(ball_vel_heading - command[:, 3:6], dim=-1)

        return torch.exp(-error / std**2) * (self.active > 0.0).float()

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
            "asset_cfg", SceneEntityCfg("robot", site_names="Trunk"),
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
        ball: Entity = env.scene["ball"]
        assert ball is not None
        robot: Entity = env.scene["robot"]
        assert robot is not None
        command = env.command_manager.get_command(command_name)
        assert command is not None

        trunk_pos_w = robot.data.root_link_pos_w
        trunk_quat_w = robot.data.root_link_quat_w
        ball_pos_w = ball.data.root_link_pos_w
        
        rel_pos_w = ball_pos_w - trunk_pos_w
        
        yaw = get_yaw_from_quaternion(trunk_quat_w)
        yaw_only_quat_w = quat_from_yaw(yaw)
        
        ball_pos_heading = quat_apply(quat_inv(yaw_only_quat_w), rel_pos_w)
        self._xy_command_ema *= 1 - ema_alpha


        # Chooses the command based on the env type
        self._xy_command_ema += torch.where(
            is_kicking_env(env, command_name).unsqueeze(-1),
            ema_alpha * torch.linalg.norm(ball_pos_heading[:, :2], dim=-1, keepdim=True),
            ema_alpha * command[:, :2],
        )

        asset: Entity = env.scene[self._asset_cfg.name]
        xy_velocity = asset.data.root_link_lin_vel_b[:, :2]

        self._xy_velocity_ema *= 1 - ema_alpha
        self._xy_velocity_ema += ema_alpha * xy_velocity

        xy_error = torch.sum(torch.square(self._xy_command_ema - self._xy_velocity_ema), dim=1)

        return torch.exp(-xy_error / std ** 2)


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
            "asset_cfg", SceneEntityCfg("robot", site_names="Trunk"),
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
            is_walking_env(env, command_name) | is_dribble_env(env, command_name),
            torch.exp(-z_error / std ** 2),
            torch.zeros_like(z_error)
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

    active = is_dribble_env(env, command_name) | is_kicking_env(env, command_name)

    alignment_score = 1.0 - (abs(angle_to_ball) / torch.pi)

    return alignment_score * active

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
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
        num_in_air, min=1
    )
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
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
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
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost

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

def feet_swing(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    swing_period: float = 0.5,
    sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
    """Rewards lifting the foot when the gait phase indicates it should be swinging."""
    command = env.command_manager.get_command(command_name)
    command_cfg = env.command_manager.get_term(command_name)
    assert command is not None
    assert command_cfg is not None

    gait_process = getattr(command_cfg, "gait_process")
    gait_frequency = getattr(command_cfg, "gait_frequency")

    contact_sensor = env.scene.sensors[sensor_name]
    assert contact_sensor is not None
    feet_contact = contact_sensor.data.found < 0.5
    
    vel_norms = torch.norm(command[:, :2], dim=-1)
    is_active = (vel_norms >= 0.05) & (gait_frequency > 1.0e-8)

    # Copied from Booster_Gym's foot swing reward
    left_swing = (torch.abs(gait_process - 0.25) < 0.5 * swing_period) & is_active
    right_swing = (torch.abs(gait_process - 0.75) < 0.5 * swing_period) & is_active
    reward = (left_swing & ~feet_contact[:, 0]).float() + (right_swing & ~feet_contact[:, 1]).float()
    return reward

def action_gait_freq_penalty(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """Penalizes the agent for using the gait frequency offset action."""
    if "gait_frequency" not in env.action_manager.active_terms:
        return torch.zeros(env.num_envs, device=env.device)
        
    gait_action = env.action_manager.get_term("gait_frequency")
    freq_offset = getattr(gait_action, "freq_offset").squeeze(-1)
    
    return torch.square(freq_offset)

def make_reward_cfg() -> dict[str, RewardTermCfg]:
    return {
        "survival": RewardTermCfg(
            func=mdp.is_alive,
            weight=0.25,
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=1.0,
            params={
                "std": 0.5,
                "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            },
        ),
        "termination": RewardTermCfg(
            func=mdp.is_terminated,
            weight=-150.0,
        ),
        "target_base_height": RewardTermCfg(
            func=TargetBaseHeightMean,
            weight=0.6,
            params={
                "target_height": 0.5,
                "ema_alpha": 0.1,
                "std": math.sqrt(0.3),
            },
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.15),
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.15),
        "action_jerk_l2": RewardTermCfg(func=ActionJerkL2, weight=-0.05),
        "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.002),
        "torque_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.0003),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.001,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Trunk")
            },  
        ),
        "soft_landing": RewardTermCfg(
            func=soft_landing,
            weight=-0.001,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
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
        "kick_contact": RewardTermCfg(
            func=kick_contact_reward,
            weight=50.0,
            params={
                "command_name": "twist",
                "sensor_name": "robot_ball_collision",
            }
        ),
        "kick_velocity": RewardTermCfg(
            func=KickVelocityReward,
            weight=50.0,
            params={
                "command_name": "twist",
                "ball_cfg": SceneEntityCfg("ball"),
                "std": 5.0,
            }
        ),
        "ball_distance": RewardTermCfg(
            func=distance_to_ball_reward,
            weight=1.0,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
                "target_distance": 0.4,
                "std": 0.25,
            }
        ),
        "ball_dribble": RewardTermCfg(
            func=ball_dribble_position,
            weight=3.0,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
                "target_distance": 0.4,
                "std": 1.0,
                "in_range_dist": 1.0,
            }
        ),
        "ball_approach": RewardTermCfg(
            func=ball_approach_alignment,
            weight=2.0,
            params={
                "command_name": "twist",
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            }
        ),
        "feet_swing": RewardTermCfg(
            func=feet_swing,
            weight=1.0,
            params={
                "command_name": "twist",
                "swing_period": 0.5,
                "sensor_name": "feet_ground_contact",
            }
        ),
        "gait_freq_penalty": RewardTermCfg(
            func=action_gait_freq_penalty,
            weight=-0.5,
        ),
    }