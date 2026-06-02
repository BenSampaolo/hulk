from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import events as event_fns, dr
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.entity import Entity
import torch

from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

class ResetRootStateUniformOnContact:
    """Event function to reset ball after contact with the robot has been detected."""
    def __init__(
        self, 
        cfg: EventTermCfg,
        env: ManagerBasedRlEnv,
    ):
        self.cfg = cfg
        self.env = env
        self.timer = torch.full((env.num_envs,), -1.0, dtype=torch.float, device=env.device)
    
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.timer.fill_(-1.0)
        else:
            self.timer[env_ids] = -1.0

    def __call__(
        self, 
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]] | None = None,
        sensor_name: str = "robot_ball_collision",
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        delay_frames: int = 5,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
            
        sensor = self.env.scene.sensors[sensor_name]
        contact = sensor.data.found[env_ids, 0] > 0.5
        timer_vals = self.timer[env_ids]
        just_touched = contact & (timer_vals < 0)

        timer_vals = torch.where(just_touched, float(delay_frames), timer_vals)
        active = timer_vals > 0
        timer_vals = torch.where(active, timer_vals - 1.0, timer_vals)

        reset_mask = timer_vals == 0.0
        local_reset_indices = torch.where(reset_mask)[0]
        timer_vals = torch.where(reset_mask, -1.0, timer_vals)

        self.timer[env_ids] = timer_vals

        if len(local_reset_indices) > 0:
            global_reset_ids = env_ids[local_reset_indices]
            offset_root_state_uniform(
                env=env,
                env_ids=global_reset_ids,
                pose_range=pose_range,
                velocity_range=velocity_range,
                asset_cfg=ball_cfg
            )

def offset_root_state_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Event function to apply random offsets to the root state."""
    asset = env.scene[asset_cfg.name]
    assert asset is not None

    pos_offset = torch.stack([
        sample_uniform(*pose_range["x"], env_ids.shape, device=env.device),
        sample_uniform(*pose_range["y"], env_ids.shape, device=env.device),
        sample_uniform(*pose_range["z"], env_ids.shape, device=env.device),
    ], dim=1)

    vel_offset = torch.zeros_like(pos_offset)
    current_vel = asset.data.root_link_vel_w[env_ids]
    if velocity_range is not None:
        # 2. Generate 3D linear velocity offset
        lin_vel_offset = torch.stack([
            sample_uniform(*velocity_range.get("x", (0.0, 0.0)), env_ids.shape, device=env.device),
            sample_uniform(*velocity_range.get("y", (0.0, 0.0)), env_ids.shape, device=env.device),
            sample_uniform(*velocity_range.get("z", (0.0, 0.0)), env_ids.shape, device=env.device),
        ], dim=1)

        # 3. Generate 3D angular velocity offset (default to 0 if not provided)
        ang_vel_offset = torch.stack([
            sample_uniform(*velocity_range.get("roll", (0.0, 0.0)), env_ids.shape, device=env.device),
            sample_uniform(*velocity_range.get("pitch", (0.0, 0.0)), env_ids.shape, device=env.device),
            sample_uniform(*velocity_range.get("yaw", (0.0, 0.0)), env_ids.shape, device=env.device),
        ], dim=1)

        # 4. Concatenate into a 6D tensor and add to current velocity
        vel_offset_6d = torch.cat([lin_vel_offset, ang_vel_offset], dim=-1)
        new_vel = current_vel + vel_offset_6d
    else:
        new_vel = current_vel
    
    current_pos = asset.data.root_link_pos_w[env_ids]
    current_quat = asset.data.root_link_quat_w[env_ids] 

    new_pose = torch.cat([current_pos + pos_offset, current_quat], dim=-1)

    asset.write_root_link_pose_to_sim(new_pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(new_vel, env_ids=env_ids)
    

def make_events_cfg(control_arms: bool) -> dict[str, EventTermCfg]:
    events = {
        "reset_scene": EventTermCfg(
            func=event_fns.reset_scene_to_default,
            mode="reset",
        ),
        "reset_base": EventTermCfg(
            func=event_fns.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=event_fns.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.3, 0.3),
                "velocity_range": (-0.2, 0.2),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "reset_ball": EventTermCfg(
            func=event_fns.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.05, 0.1)},
                "velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-0.15, 0.5)},
                "asset_cfg": SceneEntityCfg("ball")
            },
        ),
        "reset_ball_on_contact": EventTermCfg(
            func=ResetRootStateUniformOnContact,
            mode="step",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.05, 0.1)},
                "velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-0.15, 0.5)},
                "ball_cfg": SceneEntityCfg("ball"),
                "sensor_name": "robot_ball_collision",
                "delay_frames": 5
            },
        ),
        "teleport_ball": EventTermCfg(
            func=offset_root_state_uniform,
            mode="interval",
            interval_range_s=(10.0, 30.0),
            min_step_count_between_reset=500,
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.05, 0.1)},
                "velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-0.15, 0.5)},
                "asset_cfg": SceneEntityCfg("ball")
            },
        ),
        "push_ball": EventTermCfg(
            func=event_fns.apply_body_impulse,
            mode="step",
            min_step_count_between_reset=500,
            params={
                "force_range": (10.0, 30.0),
                "torque_range": (-0.0, 0.0),
                "duration_s": (0.01, 0.01),
                "cooldown_s": (10.0, 20.0),
                "asset_cfg": SceneEntityCfg("ball"),
            },
        ),
        "push_robot": EventTermCfg(
            func=event_fns.apply_body_impulse,
            mode="step",
            min_step_count_between_reset=500,
            params={
                "force_range": (-80.0, 80.0),
                "torque_range": (-8.0, 8.0),
                "duration_s": (0.01, 0.2),
                "cooldown_s": (8.0, 12.0),
                "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
            },
        ),
        "impulse": EventTermCfg(
            func=event_fns.apply_body_impulse,
            mode="step",
            min_step_count_between_reset=500,
            params={
                "force_range": (-10.0, 10.0),
                "torque_range": (-3.0, 3.0),
                "duration_s": (0.3, 2.0),
                "cooldown_s": (8.0, 12.0),
                "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
            },
        ),
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(r"(left|right)_foot", )),
                "ranges": (1.0, 2.4),
                "operation": "abs",
                "shared_random": True,
            },
        ),
        "terrain_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("terrain"), 
                "ranges": (1.0, 2.4),
                "operation": "abs",
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="reset",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="reset",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),  # Set per-robot.
                "operation": "add",
                "ranges": {
                    0: (-0.03, 0.03),
                    1: (-0.03, 0.03),
                    2: (-0.03, 0.03),
                },
            },
        ),
        "trunk_mass": EventTermCfg(
            mode="reset",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),  # Set per-robot.
                "operation": "scale",
                "ranges": (0.85, 1.15),
            },
        ),
        "default_kp_kd": EventTermCfg(
            mode="reset",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(
                    ".*Shoulder_Pitch.*",
                    ".*Shoulder_Roll.*",
                    ".*Elbow_Pitch.*",
                    ".*Elbow_Yaw.*",

                    ".*Hip_Pitch.*",
                    ".*Hip_Yaw.*",
                    ".*Knee_Pitch.*",
                ) if control_arms else (
                    ".*Hip_Pitch.*",
                    ".*Hip_Yaw.*",
                    ".*Knee_Pitch.*",
                    )),
                "kp_range": (0.95, 1.05),
                "kd_range": (0.95, 1.05),
            },
        ),
        "special_kp_kd": EventTermCfg(
            mode="reset",
            func=dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(
                    ".*Hip_Roll.*",
                    ".*Ankle_Pitch.*",
                    ".*Ankle_Roll.*"
                    )),
                "kp_range": (0.8, 1.2),
                "kd_range": (0.8, 1.2),
            },
        ),
        "armature": EventTermCfg(
            mode="reset",
            func=dr.joint_armature,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "ranges": (0.8, 1.2),
                "operation": "scale",
            },   
        ),
    }

    delay_randomizer = getattr(dr, "sync_actuator_delays", None) or getattr(dr, "actuator_delays", None)
    if delay_randomizer is not None:
        events["actuator_lag"] = EventTermCfg(
            mode="interval",
            interval_range_s=(0.01, 0.01),
            func=delay_randomizer,
            params={
                "lag_range": (1, 3),
            },
        )

    return events
