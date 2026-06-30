from __future__ import annotations

import math

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.mdp import dr, events as event_fns
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform


def _tensor_env_ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _notify_ball_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    ball_obs_cache = getattr(env, "ball_observation_cache", None)
    if ball_obs_cache is not None:
        ball_obs_cache.reset(env, env_ids)

    command_manager = getattr(env, "command_manager", None)
    if command_manager is not None:
        try:
            command_term = command_manager.get_term("kick")
        except Exception:
            command_term = None
        if command_term is not None and hasattr(command_term, "notify_ball_reset"):
            command_term.notify_ball_reset(env_ids)


def reset_robot_joints_by_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    scale_range: tuple[float, float] = (0.5, 1.5),
    velocity: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> None:
    """PiPlus reset: multiply default joint qpos by U(0.5, 1.5), set joint velocity to zero."""
    env_ids = _tensor_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return

    asset = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids_tensor = torch.arange(asset.num_joints, device=env.device, dtype=torch.long)
        joint_ids_arg = joint_ids
    else:
        joint_ids_tensor = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
        joint_ids_arg = joint_ids_tensor

    joint_pos = default_joint_pos[env_ids][:, joint_ids_tensor].clone()
    joint_pos *= sample_uniform(*scale_range, joint_pos.shape, device=env.device)
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, joint_ids_tensor]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel = torch.full_like(joint_pos, float(velocity))

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids_arg,
    )


def _current_ball_radius_from_geom_cfg(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ball,
    ball_cfg: SceneEntityCfg,
) -> torch.Tensor | None:
    if ball_cfg.geom_names is not None and isinstance(ball_cfg.geom_ids, slice):
        ball_cfg.resolve(env.scene)

    if isinstance(ball_cfg.geom_ids, slice) and ball_cfg.geom_names is None:
        return None

    geom_size = getattr(env.sim.model, "geom_size", None)
    if geom_size is None or not hasattr(geom_size, "shape"):
        return None

    indexing = getattr(ball, "indexing", getattr(ball.data, "indexing", None))
    if indexing is None:
        return None

    global_geom_ids = torch.as_tensor(
        indexing.geom_ids[ball_cfg.geom_ids], device=env.device, dtype=torch.long
    ).flatten()
    if global_geom_ids.numel() == 0:
        return None

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    radius = geom_size[env_ids[:, None], global_geom_ids[None, :], 0].amax(dim=1)
    return radius.to(device=env.device)


@requires_model_fields(
    "body_mass",
    "body_inertia",
    "geom_size",
    "geom_friction",
    "geom_rbound",
    "geom_aabb",
    recompute=RecomputeLevel.set_const,
)
def randomize_ball_properties(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    mass_range: tuple[float, float] = (0.75, 1.25),
    radius_range: tuple[float, float] = (0.75, 1.25),
    friction_range: tuple[float, float] = (0.75, 1.25),
    body_cfg: SceneEntityCfg = SceneEntityCfg("ball", body_names=("ball",)),
    geom_cfg: SceneEntityCfg = SceneEntityCfg("ball", geom_names=("ball",)),
) -> None:
    """PiPlus ball property randomization with consistent sphere inertia."""
    env_ids = _tensor_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return

    if body_cfg.body_names is not None and isinstance(body_cfg.body_ids, slice):
        body_cfg.resolve(env.scene)
    if geom_cfg.geom_names is not None and isinstance(geom_cfg.geom_ids, slice):
        geom_cfg.resolve(env.scene)

    body_asset = env.scene[body_cfg.name]
    geom_asset = env.scene[geom_cfg.name]
    body_ids = torch.as_tensor(
        body_asset.indexing.body_ids[body_cfg.body_ids], device=env.device, dtype=torch.long
    ).flatten()
    geom_ids = torch.as_tensor(
        geom_asset.indexing.geom_ids[geom_cfg.geom_ids], device=env.device, dtype=torch.long
    ).flatten()
    if body_ids.numel() != 1 or geom_ids.numel() != 1:
        raise ValueError("randomize_ball_properties expects exactly one ball body and one ball geom")

    body_mass_field = env.sim.model.body_mass
    body_inertia_field = env.sim.model.body_inertia
    geom_size_field = env.sim.model.geom_size
    geom_friction_field = env.sim.model.geom_friction

    default_mass = env.sim.get_default_field("body_mass").to(device=env.device, dtype=body_mass_field.dtype)[body_ids]
    default_radius = env.sim.get_default_field("geom_size").to(device=env.device, dtype=geom_size_field.dtype)[
        geom_ids, 0
    ]
    default_friction = env.sim.get_default_field("geom_friction").to(
        device=env.device, dtype=geom_friction_field.dtype
    )[geom_ids]

    mass = default_mass.unsqueeze(0) * sample_uniform(
        *mass_range, (len(env_ids), body_ids.numel()), device=env.device
    ).to(dtype=body_mass_field.dtype)
    radius = default_radius.unsqueeze(0) * sample_uniform(
        *radius_range, (len(env_ids), geom_ids.numel()), device=env.device
    ).to(dtype=geom_size_field.dtype)
    friction = default_friction.unsqueeze(0) * sample_uniform(
        *friction_range, (len(env_ids), geom_ids.numel(), 3), device=env.device
    ).to(dtype=geom_friction_field.dtype)
    inertia = (0.4 * mass * radius.square()).unsqueeze(-1).expand(-1, -1, 3)

    env_grid_body, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
    body_mass_field[env_grid_body, body_grid] = mass
    body_inertia_field[env_grid_body, body_grid] = inertia.to(dtype=body_inertia_field.dtype)

    env_grid_geom, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    geom_size_field[env_grid_geom, geom_grid, 0] = radius
    geom_friction_field[env_grid_geom, geom_grid] = friction
    env.sim.model.geom_rbound[env_grid_geom, geom_grid] = radius.to(dtype=env.sim.model.geom_rbound.dtype)
    env.sim.model.geom_aabb[env_grid_geom, geom_grid, 1] = radius.unsqueeze(-1).expand(-1, -1, 3).to(
        dtype=env.sim.model.geom_aabb.dtype
    )


def reset_ball_relative_to_robot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    distance_range: tuple[float, float] = (0.25, 0.4),
    angle_range: tuple[float, float] = (-math.pi / 4.0, math.pi / 4.0),
    z_offset_range: tuple[float, float] = (0.0, 0.0),
    velocity_range: dict[str, tuple[float, float]] | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    refresh_robot_pose: bool = False,
) -> None:
    """Reset the ball in front of the robot with PiPlus distance/angle sampling and zero default velocity."""
    env_ids = _tensor_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return

    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]

    if refresh_robot_pose:
        env.scene.write_data_to_sim()
        env.sim.forward()

    distance = sample_uniform(*distance_range, env_ids.shape, device=env.device)
    rel_angle = sample_uniform(*angle_range, env_ids.shape, device=env.device)
    z_offset = sample_uniform(*z_offset_range, env_ids.shape, device=env.device)

    robot_pos_w = robot.data.root_link_pos_w[env_ids]
    robot_yaw = _yaw_from_quat_wxyz(robot.data.root_link_quat_w[env_ids])
    angle_w = robot_yaw + rel_angle
    ball_default_state = ball.data.default_root_state[env_ids]

    ball_radius = _current_ball_radius_from_geom_cfg(env, env_ids, ball, ball_cfg)
    ball_z = env.scene.env_origins[env_ids, 2] + z_offset
    if ball_radius is None:
        ball_z = ball_z + ball_default_state[:, 2]
    else:
        ball_z = ball_z + ball_radius

    ball_pos_w = torch.stack(
        (
            robot_pos_w[:, 0] + torch.cos(angle_w) * distance,
            robot_pos_w[:, 1] + torch.sin(angle_w) * distance,
            ball_z,
        ),
        dim=-1,
    )
    ball_quat_w = ball_default_state[:, 3:7]
    ball.write_root_link_pose_to_sim(torch.cat((ball_pos_w, ball_quat_w), dim=-1), env_ids=env_ids)

    velocity_range = velocity_range or {}
    lin_vel_b = torch.stack(
        (
            sample_uniform(*velocity_range.get("x", (0.0, 0.0)), env_ids.shape, device=env.device),
            sample_uniform(*velocity_range.get("y", (0.0, 0.0)), env_ids.shape, device=env.device),
            sample_uniform(*velocity_range.get("z", (0.0, 0.0)), env_ids.shape, device=env.device),
        ),
        dim=-1,
    )
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    lin_vel_w = torch.stack(
        (
            cos_yaw * lin_vel_b[:, 0] - sin_yaw * lin_vel_b[:, 1],
            sin_yaw * lin_vel_b[:, 0] + cos_yaw * lin_vel_b[:, 1],
            lin_vel_b[:, 2],
        ),
        dim=-1,
    )
    ang_vel_w = torch.stack(
        (
            sample_uniform(*velocity_range.get("roll", (0.0, 0.0)), env_ids.shape, device=env.device),
            sample_uniform(*velocity_range.get("pitch", (0.0, 0.0)), env_ids.shape, device=env.device),
            sample_uniform(*velocity_range.get("yaw", (0.0, 0.0)), env_ids.shape, device=env.device),
        ),
        dim=-1,
    )
    ball.write_root_link_velocity_to_sim(torch.cat((lin_vel_w, ang_vel_w), dim=-1), env_ids=env_ids)

    initial_ball_pos = getattr(env, "_k1_kick_initial_ball_pos", None)
    if not isinstance(initial_ball_pos, torch.Tensor) or initial_ball_pos.shape != (env.num_envs, 3):
        initial_ball_pos = torch.zeros((env.num_envs, 3), device=env.device)
        setattr(env, "_k1_kick_initial_ball_pos", initial_ball_pos)
    initial_ball_pos[env_ids] = ball_pos_w

    env.sim.forward()
    _notify_ball_reset(env, env_ids)


def push_root_xy_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    speed_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Add a random-direction XY velocity to an asset root, matching PiPlus push events."""
    env_ids = _tensor_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return

    asset = env.scene[asset_cfg.name]
    theta = sample_uniform(0.0, 2.0 * math.pi, env_ids.shape, device=env.device)
    speed = sample_uniform(*speed_range, env_ids.shape, device=env.device)
    delta_xy = torch.stack((torch.cos(theta) * speed, torch.sin(theta) * speed), dim=-1)

    root_vel_w = asset.data.root_link_vel_w[env_ids].clone()
    root_vel_w[:, :2] += delta_xy
    asset.write_root_link_velocity_to_sim(root_vel_w, env_ids=env_ids)


def reset_encoder_bias_normal(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    std: float = math.radians(2.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample per-episode joint encoder/calibration bias from N(0, 2deg)."""
    env_ids = _tensor_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return

    asset = env.scene[asset_cfg.name]
    encoder_bias = getattr(asset.data, "encoder_bias", None)
    if encoder_bias is None:
        return

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids_tensor = torch.arange(asset.num_joints, device=env.device, dtype=torch.long)
    else:
        joint_ids_tensor = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)

    samples = torch.randn((len(env_ids), len(joint_ids_tensor)), device=env.device) * std
    encoder_bias[env_ids[:, None], joint_ids_tensor] = samples


def reset_imu_bias(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    std: float = 0.01,
) -> None:
    """Sample per-episode IMU roll/pitch mounting bias from N(0, 0.01 rad)."""
    env_ids = _tensor_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return

    bias = getattr(env, "_k1_kick_imu_bias_rp", None)
    if not isinstance(bias, torch.Tensor) or bias.shape != (env.num_envs, 2):
        bias = torch.zeros((env.num_envs, 2), device=env.device)
        setattr(env, "_k1_kick_imu_bias_rp", bias)
    bias[env_ids] = torch.randn((len(env_ids), 2), device=env.device) * std


def make_events_cfg() -> dict[str, EventTermCfg]:
    leg_joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    robot_body_cfg = SceneEntityCfg("robot", body_names=(".*",))
    trunk_body_cfg = SceneEntityCfg("robot", body_names=("Trunk",))
    robot_actuator_cfg = SceneEntityCfg("robot", actuator_names=".*")

    return {
        "reset_scene": EventTermCfg(
            func=event_fns.reset_scene_to_default,
            mode="reset",
        ),
        "terrain_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("terrain", geom_names=("terrain",)),
                "ranges": (0.4, 1.0),
                "operation": "abs",
                "shared_random": True,
            },
        ),
        "joint_friction_loss": EventTermCfg(
            func=dr.joint_friction,
            mode="reset",
            params={"asset_cfg": leg_joint_cfg, "ranges": (0.9, 1.1), "operation": "scale"},
        ),
        "joint_armature": EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={"asset_cfg": leg_joint_cfg, "ranges": (1.0, 1.05), "operation": "scale"},
        ),
        "joint_damping": EventTermCfg(
            func=dr.joint_damping,
            mode="reset",
            params={"asset_cfg": leg_joint_cfg, "ranges": (0.95, 1.05), "operation": "scale"},
        ),
        "actuator_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="reset",
            params={
                "asset_cfg": robot_actuator_cfg,
                "kp_range": (0.9, 1.1),
                "kd_range": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "robot_body_mass": EventTermCfg(
            func=dr.body_mass,
            mode="reset",
            params={"asset_cfg": robot_body_cfg, "ranges": (0.9, 1.1), "operation": "scale"},
        ),
        "trunk_mass": EventTermCfg(
            func=dr.body_mass,
            mode="reset",
            params={"asset_cfg": trunk_body_cfg, "ranges": (-0.5, 0.5), "operation": "add"},
        ),
        "trunk_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="reset",
            params={
                "asset_cfg": trunk_body_cfg,
                "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
                "operation": "add",
            },
        ),
        "body_com_jitter": EventTermCfg(
            func=dr.body_com_offset,
            mode="reset",
            params={"asset_cfg": robot_body_cfg, "ranges": (-0.01, 0.01), "operation": "add"},
        ),
        "default_joint_pos": EventTermCfg(
            func=dr.joint_default_pos,
            mode="reset",
            params={"asset_cfg": leg_joint_cfg, "ranges": (-0.05, 0.05), "operation": "add"},
        ),
        "ball_properties": EventTermCfg(
            func=randomize_ball_properties,
            mode="reset",
            params={
                "mass_range": (0.75, 1.25),
                "radius_range": (0.75, 1.25),
                "friction_range": (0.75, 1.25),
                "body_cfg": SceneEntityCfg("ball", body_names=("ball",)),
                "geom_cfg": SceneEntityCfg("ball", geom_names=("ball",)),
            },
        ),
        "encoder_bias": EventTermCfg(
            func=reset_encoder_bias_normal,
            mode="reset",
            params={"asset_cfg": leg_joint_cfg, "std": math.radians(2.0)},
        ),
        "imu_bias": EventTermCfg(
            func=reset_imu_bias,
            mode="reset",
            params={"std": 0.01},
        ),
        "reset_base": EventTermCfg(
            func=event_fns.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=reset_robot_joints_by_scale,
            mode="reset",
            params={
                "scale_range": (0.5, 1.5),
                "velocity": 0.0,
                "asset_cfg": leg_joint_cfg,
            },
        ),
        "reset_ball": EventTermCfg(
            func=reset_ball_relative_to_robot,
            mode="reset",
            params={
                "distance_range": (0.25, 0.4),
                "angle_range": (-math.pi / 4.0, math.pi / 4.0),
                "z_offset_range": (0.0, 0.0),
                "velocity_range": {},
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball", geom_names=("ball",)),
                "refresh_robot_pose": True,
            },
        ),
        "push_robot": EventTermCfg(
            func=push_root_xy_velocity,
            mode="interval",
            interval_range_s=(5.0, 10.0),
            min_step_count_between_reset=250,
            params={"speed_range": (0.05, 1.0), "asset_cfg": SceneEntityCfg("robot")},
        ),
        "push_ball": EventTermCfg(
            func=push_root_xy_velocity,
            mode="interval",
            interval_range_s=(3.0, 8.0),
            min_step_count_between_reset=150,
            params={"speed_range": (0.1, 2.5), "asset_cfg": SceneEntityCfg("ball")},
        ),
    }
