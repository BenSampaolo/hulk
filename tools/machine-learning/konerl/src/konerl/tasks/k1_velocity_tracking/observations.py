from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
import math
import torch
from mjlab.entity import Entity
from typing import Any
from typing_extensions import override
from dataclasses import dataclass
from mjlab.utils.noise.noise_cfg import NoiseCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_inv
from .rewards import is_walking_env, is_kicking_env, is_standing_env, is_dribble_env

_JOINT_VEL_EMA_ALPHA = 0.78


def get_yaw_from_quaternion(quat: torch.Tensor) -> torch.Tensor:
    # Extracts yaw (z-axis rotation) from a quaternion [w, x, y, z]
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw

def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    # Creates a quaternion [w, x, y, z] from yaw (z-axis rotation)
    half_yaw = yaw * 0.5
    w = torch.cos(half_yaw)
    z = torch.sin(half_yaw)
    x = torch.zeros_like(yaw)
    y = torch.zeros_like(yaw)
    return torch.stack([w, x, y, z], dim=-1)

def obs_gait_sin(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
) -> torch.Tensor:
    """The sine of the current gait phase. Scaled by 0 if not walking/kicking."""
    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_cfg = env.command_manager.get_term(command_name)
    assert command_cfg is not None
    
    gait_process = getattr(command_cfg, "gait_process")
    vel_norms = torch.norm(command[:, :2], dim=-1)
    active = (vel_norms >= 0.05).float()
    return (torch.sin(2 * torch.pi * gait_process) * active).unsqueeze(-1)


def obs_gait_cos(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
) -> torch.Tensor:
    """The cosine of the current gait phase. Scaled by 0 if not walking/kicking."""
    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_cfg = env.command_manager.get_term(command_name)
    assert command_cfg is not None
    
    gait_process = getattr(command_cfg, "gait_process")
    vel_norms = torch.norm(command[:, :2], dim=-1)
    active = (vel_norms >= 0.05).float()
    return (torch.cos(2 * torch.pi * gait_process) * active).unsqueeze(-1)

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



class JointVelRelSmoothed:
    def __init__(self, ema_alpha: float = _JOINT_VEL_EMA_ALPHA):
        self.ema_alpha = ema_alpha
        self._state_by_env: dict[int, dict[str, Any]] = {}

    def _ensure_state(self, env: Any, asset: Entity) -> dict[str, Any]:
        env_key = id(env)
        joint_pos = asset.data.joint_pos
        safe_joint_pos = torch.where(torch.isfinite(joint_pos), joint_pos, torch.zeros_like(joint_pos))
        num_envs, num_joints = joint_pos.shape

        state = self._state_by_env.get(env_key)
        if state is None:
            state = {
                "last_joint_pos": safe_joint_pos.clone(),
                "last_smoothed_vel": torch.zeros((num_envs, num_joints), device=joint_pos.device),
                "last_step": None,
            }
            self._state_by_env[env_key] = state
        else:
            shape_mismatch = state["last_joint_pos"].shape != joint_pos.shape
            device_mismatch = state["last_joint_pos"].device != joint_pos.device
            if shape_mismatch or device_mismatch:
                state["last_joint_pos"] = safe_joint_pos.clone()
                state["last_smoothed_vel"] = torch.zeros((num_envs, num_joints), device=joint_pos.device)
                state["last_step"] = None

        episode_length_buf = getattr(env, "episode_length_buf", None)
        if isinstance(episode_length_buf, torch.Tensor) and episode_length_buf.shape[0] == num_envs:
            reset_mask = episode_length_buf == 0
            if bool(reset_mask.any().item()):
                state["last_joint_pos"][reset_mask] = safe_joint_pos[reset_mask].clone()
                state["last_smoothed_vel"][reset_mask] = 0.0

        return state

    def __call__(
        self,
        env: Any,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        state = self._ensure_state(env, asset)
        jnt_ids = asset_cfg.joint_ids

        step_index = _get_env_step_index(env)
        if step_index is not None and state["last_step"] == step_index:
            return state["last_smoothed_vel"][:, jnt_ids]

        current_pos = asset.data.joint_pos[:, jnt_ids]
        prev_pos = state["last_joint_pos"][:, jnt_ids]
        safe_prev_pos = torch.where(torch.isfinite(prev_pos), prev_pos, torch.zeros_like(prev_pos))
        safe_current_pos = torch.where(torch.isfinite(current_pos), current_pos, safe_prev_pos)

        step_dt = getattr(env, "step_dt", 0.0)
        if isinstance(step_dt, torch.Tensor):
            step_dt = step_dt.item() if step_dt.numel() == 1 else 0.0
        step_dt = float(step_dt)
        safe_step_dt = abs(step_dt) if math.isfinite(step_dt) else 0.0
        safe_step_dt = max(safe_step_dt, 1e-6)

        current_vel = (safe_current_pos - safe_prev_pos) / safe_step_dt
        smoothed_vel = self.ema_alpha * state["last_smoothed_vel"][:, jnt_ids] + (1 - self.ema_alpha) * current_vel

        state["last_smoothed_vel"][:, jnt_ids] = smoothed_vel
        state["last_joint_pos"][:, jnt_ids] = safe_current_pos
        state["last_step"] = step_index

        return smoothed_vel


JOINT_VEL_REL_SMOOTHED = JointVelRelSmoothed()

@dataclass
class ClippedGaussianNoiseCfg(NoiseCfg):
  mean: torch.Tensor | float = 0.0
  std: torch.Tensor | float = 1.0
  min: float = -2.0
  max: float = 2.0

  def __post_init__(self):
    if isinstance(self.std, (int, float)) and self.std <= 0:
      raise ValueError(f"std ({self.std}) must be positive")

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    self.mean = torch.as_tensor(self.mean, device=data.device)
    self.std = torch.as_tensor(self.std, device=data.device)

    # Generate standard normal noise and scale.
    noise = torch.clamp(self.mean + self.std * torch.randn_like(data), min=self.min, max=self.max)

    if self.operation == "add":
      return data + noise
    elif self.operation == "scale":
      return data * noise
    elif self.operation == "abs":
      return noise
    else:
      raise ValueError(f"Unsupported noise operation: {self.operation}")

def obs_ball_pos_heading_frame(
    env: ManagerBasedRlEnv,
    outside_info: bool = False,
) -> torch.Tensor:
    """Ball position relative to trunk, ignoring trunk roll and pitch (yaw-only frame)."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    
    trunk_pos_w = robot.data.root_link_pos_w
    trunk_quat_w = robot.data.root_link_quat_w
    ball_pos_w = ball.data.root_link_pos_w
    
    rel_pos_w = ball_pos_w - trunk_pos_w
    
    yaw = get_yaw_from_quaternion(trunk_quat_w)
    yaw_only_quat_w = quat_from_yaw(yaw)
    
    ball_pos_heading = quat_apply(quat_inv(yaw_only_quat_w), rel_pos_w)
    if not outside_info:
        ball_pos_heading = torch.where(ball_pos_heading[:, 1:2] < 0, torch.zeros_like(ball_pos_heading), ball_pos_heading)
    return ball_pos_heading

def obs_ball_vel_heading_frame(
    env: ManagerBasedRlEnv,
    outside_info: bool = False,
) -> torch.Tensor:
    """Ball linear velocity, ignoring trunk roll and pitch (yaw-only frame)."""
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    
    trunk_quat_w = robot.data.root_link_quat_w
    ball_vel_w = ball.data.root_link_vel_w[:, :3] # just the linear part
    
    yaw = get_yaw_from_quaternion(trunk_quat_w)
    yaw_only_quat_w = quat_from_yaw(yaw)
    
    ball_vel_heading = quat_apply(quat_inv(yaw_only_quat_w), ball_vel_w)
    if not outside_info:
        ball_vel_heading = torch.where(ball_vel_heading[:, 1:2] < 0, torch.zeros_like(ball_vel_heading), ball_vel_heading)
    return ball_vel_heading

##
# Randomization observations
##

def obs_trunk_mass(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.sim.model.body_mass[:, asset_cfg.body_ids]

def obs_foot_friction(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.sim.model.geom_friction[:, asset_cfg.geom_ids, 0]

def obs_base_com(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.sim.model.body_ipos[:, asset_cfg.body_ids].view(env.num_envs, -1)


def obs_foot_height_world(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]

def obs_pd_gains(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    
    if isinstance(asset_cfg.actuator_ids, list):
        actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
    elif isinstance(asset_cfg.actuator_ids, slice):
        actuators = asset.actuators[asset_cfg.actuator_ids]
    else:
        actuators = [asset.actuators[asset_cfg.actuator_ids]]

    kp_list, kd_list = [], []
    for a in actuators:
        base_a = getattr(a, "base_actuator", a)
        
        if hasattr(base_a, "stiffness"):  # IdealPdActuator
            kp_list.append(base_a.stiffness)
            kd_list.append(base_a.damping)
        else:  # BuiltinPositionActuator or XmlPositionActuator
            ctrl_ids = base_a.ctrl_ids
            kp_list.append(env.sim.model.actuator_gainprm[:, ctrl_ids, 0])
            # Kd is stored as the negative derivative gain in biasprm
            kd_list.append(-env.sim.model.actuator_biasprm[:, ctrl_ids, 2])

    kp = torch.cat(kp_list, dim=-1)
    kd = torch.cat(kd_list, dim=-1)
    return torch.cat((kp, kd), dim=-1)

def obs_actuator_lag(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    
    if isinstance(asset_cfg.actuator_ids, list):
        actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
    elif isinstance(asset_cfg.actuator_ids, slice):
        actuators = asset.actuators[asset_cfg.actuator_ids]
    else:
        actuators = [asset.actuators[asset_cfg.actuator_ids]]

    lag_values: list[float] = []
    for actuator in actuators:
        min_lag_raw = getattr(actuator, "delay_min_lag", None)
        max_lag_raw = getattr(actuator, "delay_max_lag", None)
        if min_lag_raw is None and max_lag_raw is None:
            continue
        if min_lag_raw is None:
            min_lag_raw = max_lag_raw
        if max_lag_raw is None:
            max_lag_raw = min_lag_raw

        # At this point both values are guaranteed to be present.
        if min_lag_raw is None or max_lag_raw is None:
            continue

        lag_values.append(0.5 * (float(min_lag_raw) + float(max_lag_raw)))

    if not lag_values:
        return torch.zeros((env.num_envs, 0), device=env.device)

    mean_lag = float(sum(lag_values) / len(lag_values))
    return torch.full((env.num_envs, 1), mean_lag, device=env.device)

def obs_encoder_bias(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # Slices the encoder_bias tensor using the joint_ids defined in your SceneEntityCfg
    return asset.data.encoder_bias[:, asset_cfg.joint_ids]

def obs_push_force(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    wrench = getattr(asset.data, "body_external_wrench", None)

    if wrench is None:
        # Backward-compatible fallback for stacks that expose MuJoCo's raw xfrc.
        sim_data = getattr(getattr(env, "sim", None), "data", None)
        wrench = getattr(sim_data, "xfrc_applied", None)

    if wrench is None:
        num_envs = int(getattr(env, "num_envs", 1))
        num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 1
        device = getattr(env, "device", "cpu")
        return torch.zeros((num_envs, num_bodies * 3), device=device)

    # Keep only force components (Fx, Fy, Fz), not torques.
    forces = wrench[..., :3]
    body_ids = asset_cfg.body_ids

    if forces.ndim == 3:
        # (num_envs, num_bodies, 3)
        if body_ids is not None:
            forces = forces[:, body_ids, :]
        return forces.reshape(forces.shape[0], -1)

    if forces.ndim == 2:
        # (num_bodies, 3) -> single-env fallback
        if body_ids is not None:
            forces = forces[body_ids, :]
        return forces.reshape(1, -1)

    # Unexpected shape; fall back to zeros for stability.
    num_envs = int(getattr(env, "num_envs", 1))
    device = getattr(forces, "device", getattr(env, "device", "cpu"))
    return torch.zeros((num_envs, 3), device=device)

def last_last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.prev_action

def safe_foot_contact_forces(env, sensor_name: str) -> torch.Tensor:
    obs = mdp.foot_contact_forces(env, sensor_name)
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

##
# Gait
##

def obs_gait(env, _asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None and hasattr(action_manager, "get_term"):
        try:
            freq_term = action_manager.get_term("gait_frequency")
            current_phase = getattr(freq_term, "current_phase", None)
            if isinstance(current_phase, torch.Tensor):
                phase = current_phase.reshape(env.num_envs, -1)[:, 0]
                gait_sin = torch.sin(phase)
                gait_cos = torch.cos(phase)
                return torch.stack((gait_sin, gait_cos), dim=-1).to(device=getattr(env, "device", "cpu"), dtype=torch.float32)
        except Exception:
            pass
            
    # Fallback if term doesn't exist (e.g. testing without it)
    return torch.zeros((getattr(env, "num_envs", 1), 2), device=getattr(env, "device", "cpu"), dtype=torch.float32)

def gait_frequency_multiplier(env, _asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None and hasattr(action_manager, "get_term"):
        try:
            freq_term = action_manager.get_term("gait_frequency")
            multiplier = getattr(freq_term, "freq_multiplier", None)
            if isinstance(multiplier, torch.Tensor):
                return multiplier.reshape(env.num_envs, -1)[:, 0:1]
        except Exception:
            pass
    return torch.ones((getattr(env, "num_envs", 1), 1), device=getattr(env, "device", "cpu"), dtype=torch.float32)


def make_observation_cfg() -> dict[str, ObservationGroupCfg]:
    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=ClippedGaussianNoiseCfg(mean=0, std=0.005, min=-0.03, max=0.03),
            delay_min_lag=1,
            delay_max_lag=2,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=ClippedGaussianNoiseCfg(mean=0, std=0.04, min=-0.1, max=0.1),
            delay_min_lag=1,
            delay_max_lag=2,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=ClippedGaussianNoiseCfg(mean=0, std=0.003, min=-0.03, max=0.03),
            delay_min_lag=1,
            delay_max_lag=2,
        ),
        "joint_vel": ObservationTermCfg(
            func=JOINT_VEL_REL_SMOOTHED,
            noise=ClippedGaussianNoiseCfg(mean=0, std=0.015, min=-0.05, max=0.05),
            delay_min_lag=1,
            delay_max_lag=2,
        ),
        "ball_pos": ObservationTermCfg(
            func=obs_ball_pos_heading_frame,
            noise=ClippedGaussianNoiseCfg(mean=0, std=0.03, min=-0.1, max=0.1),
            delay_min_lag=1,
            delay_max_lag=2,
        ),
         "ball_vel": ObservationTermCfg(
            func=obs_ball_vel_heading_frame,
            noise=ClippedGaussianNoiseCfg(mean=0, std=0.03, min=-0.1, max=0.1),
            delay_min_lag=1,
            delay_max_lag=2,
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "gait_sin": ObservationTermCfg(
            func=obs_gait_sin,
            params={"command_name": "twist"},
        ),
        "gait_cos": ObservationTermCfg(
            func=obs_gait_cos,
            params={"command_name": "twist"},
        ),
    }

    critic_terms = {
        ##
        # Policy terms without noise and delay
        ##
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
        ),
        "joint_vel": ObservationTermCfg(
            func=JOINT_VEL_REL_SMOOTHED,
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
        ),
        "prev_prev_actions": ObservationTermCfg(
            func=last_last_action,
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "gait_sin": ObservationTermCfg(
            func=obs_gait_sin,
            params={"command_name": "twist"},
        ),
        "gait_cos": ObservationTermCfg(
            func=obs_gait_cos,
            params={"command_name": "twist"},
        ),
        # "gait_frequency": ObservationTermCfg(
        #     func=gait_frequency_multiplier,
        # ),

        ##
        # Exclusive critic terms
        ##

        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "foot_height": ObservationTermCfg(
            func=obs_foot_height_world,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=safe_foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
        ##
        # Randomization observations
        ##
        "trunk_mass": ObservationTermCfg(
            func=obs_trunk_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",))}
        ),
        "foot_friction": ObservationTermCfg(
            func=obs_foot_friction,
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=("left_foot", "right_foot"))}
        ),
        "base_com": ObservationTermCfg(
            func=obs_base_com,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",))}
        ),
        "default_KpKd_gains": ObservationTermCfg(
            func=obs_pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(
                    ".*Shoulder_Pitch.*", ".*Shoulder_Roll.*", ".*Elbow_Pitch.*", 
                    ".*Elbow_Yaw.*", 
                    ".*Hip_Pitch.*", ".*Hip_Yaw.*", ".*Knee_Pitch.*"
                ))
            }
        ),
        "special_KpKd_gains": ObservationTermCfg(
            func=obs_pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(
                    ".*Hip_Roll.*", ".*Ankle_Pitch.*", ".*Ankle_Roll.*"
                ))
            }
        ),
        "actuator_lag": ObservationTermCfg(
            func=obs_actuator_lag,
            params={"asset_cfg": SceneEntityCfg("robot")}
        ),
        "encoder_bias": ObservationTermCfg(
            func=obs_encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot")}
        ),
        "push_force": ObservationTermCfg(
            func=obs_push_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",))},
        ),
    }

    return {
        "actor": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=1,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }

if __name__ == "__main__":
    print("Running math tests for heading frame transformations...")
    
    def test_yaw_quaternion_roundtrip():
        yaws = torch.tensor([0.0, math.pi/2, math.pi, -math.pi/2])
        quats = quat_from_yaw(yaws)
        extracted_yaws = get_yaw_from_quaternion(quats)
        # Normalize extracted yaw to [-pi, pi] to handle pi/-pi wrapping
        extracted_yaws = torch.atan2(torch.sin(extracted_yaws), torch.cos(extracted_yaws))
        yaws_norm = torch.atan2(torch.sin(yaws), torch.cos(yaws))
        torch.testing.assert_close(yaws_norm, extracted_yaws)
        print("✓ test_yaw_quaternion_roundtrip passed")

    def test_yaw_ignores_roll_pitch():
        roll, pitch, yaw = math.pi/4, math.pi/6, math.pi/2
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        quat = torch.tensor([[w, x, y, z]])
        extracted_yaw = get_yaw_from_quaternion(quat)
        torch.testing.assert_close(extracted_yaw, torch.tensor([math.pi/2]))
        print("✓ test_yaw_ignores_roll_pitch passed")

    def test_heading_frame_transformation():
        yaw = torch.tensor([math.pi/2])
        robot_yaw_quat = quat_from_yaw(yaw)
        rel_pos_w = torch.tensor([[0.0, 5.0, 0.0]])
        
        ball_pos_heading = quat_apply(quat_inv(robot_yaw_quat), rel_pos_w)
        expected_pos = torch.tensor([[5.0, 0.0, 0.0]])
        
        torch.testing.assert_close(ball_pos_heading, expected_pos, atol=1e-6, rtol=1e-6)
        print("✓ test_heading_frame_transformation passed")

    test_yaw_quaternion_roundtrip()
    test_yaw_ignores_roll_pitch()
    test_heading_frame_transformation()
    print("All math tests passed!")
