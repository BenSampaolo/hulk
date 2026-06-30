from __future__ import annotations

import math
from typing import Any

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg

_MAX_BALL_DELAY_STEPS = 10
_BALL_POS_NOISE_STD = 0.02
_KICK_DIR_ANGLE_NOISE_STD = math.radians(10.0)


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


def _imu_bias_rp(env: ManagerBasedRlEnv) -> torch.Tensor:
    bias = getattr(env, "_k1_kick_imu_bias_rp", None)
    if isinstance(bias, torch.Tensor) and bias.shape == (env.num_envs, 2):
        return bias.to(device=env.device)
    return torch.zeros((env.num_envs, 2), device=env.device)


def _apply_imu_rp_bias(vector_b: torch.Tensor, bias_rp: torch.Tensor) -> torch.Tensor:
    roll = bias_rp[:, 0]
    pitch = bias_rp[:, 1]
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)

    rot = torch.zeros((vector_b.shape[0], 3, 3), device=vector_b.device, dtype=vector_b.dtype)
    rot[:, 0, 0] = cp
    rot[:, 0, 1] = sp * sr
    rot[:, 0, 2] = sp * cr
    rot[:, 1, 1] = cr
    rot[:, 1, 2] = -sr
    rot[:, 2, 0] = -sp
    rot[:, 2, 1] = cp * sr
    rot[:, 2, 2] = cp * cr
    return torch.bmm(rot, vector_b.unsqueeze(-1)).squeeze(-1)


def obs_biased_gyro(env: ManagerBasedRlEnv, sensor_name: str = "robot/imu_ang_vel") -> torch.Tensor:
    return _apply_imu_rp_bias(mdp.builtin_sensor(env, sensor_name), _imu_bias_rp(env))


def obs_biased_projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _apply_imu_rp_bias(mdp.projected_gravity(env, asset_cfg), _imu_bias_rp(env))


def obs_ball_pos_body_xy(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    rel_pos_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return quat_apply_inverse(robot.data.root_link_quat_w, rel_pos_w)[:, :2]


class BallObservationCache:
    def __init__(self, max_delay_steps: int = _MAX_BALL_DELAY_STEPS, position_noise_std: float = _BALL_POS_NOISE_STD):
        self.max_delay_steps = max_delay_steps
        self.position_noise_std = position_noise_std
        self._state_by_env: dict[int, dict[str, Any]] = {}

    def _sample_lag(self, shape: tuple[int, ...] | torch.Size, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.max_delay_steps + 1, shape, device=device, dtype=torch.long)

    def _raw_pos(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return obs_ball_pos_body_xy(env)

    def _ensure_state(self, env: ManagerBasedRlEnv) -> dict[str, Any]:
        env_key = id(env)
        pos = self._raw_pos(env)
        num_envs = pos.shape[0]
        history_len = self.max_delay_steps + 1
        state = self._state_by_env.get(env_key)
        if state is None or state["pos_history"].shape[:2] != (num_envs, history_len):
            state = {
                "pos_history": torch.zeros(
                    (num_envs, history_len, pos.shape[-1]), device=pos.device, dtype=pos.dtype
                ),
                "lag": self._sample_lag((num_envs,), pos.device),
                "last_step": _get_env_step_index(env),
                "cache_key": None,
                "cache_value": None,
            }
            self._state_by_env[env_key] = state
            setattr(env, "ball_observation_cache", self)
        return state

    def reset(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
        state = self._ensure_state(env)
        pos = self._raw_pos(env)
        if env_ids is None:
            env_ids = torch.arange(pos.shape[0], device=pos.device, dtype=torch.long)
        history_len = self.max_delay_steps + 1
        state["pos_history"][env_ids] = torch.zeros(
            (len(env_ids), history_len, pos.shape[-1]), device=pos.device, dtype=pos.dtype
        )
        state["lag"][env_ids] = self._sample_lag(env_ids.shape, pos.device)
        state["last_step"] = _get_env_step_index(env)
        state["cache_key"] = None
        state["cache_value"] = None

    def _update(self, env: ManagerBasedRlEnv) -> dict[str, Any]:
        state = self._ensure_state(env)
        step_index = _get_env_step_index(env)
        if step_index is not None and state["last_step"] == step_index:
            return state

        pos = self._raw_pos(env)
        state["pos_history"][:, 1:] = state["pos_history"][:, :-1].clone()
        state["pos_history"][:, 0] = pos

        episode_length_buf = getattr(env, "episode_length_buf", None)
        if isinstance(episode_length_buf, torch.Tensor) and episode_length_buf.shape[0] == pos.shape[0]:
            reset_ids = torch.nonzero(episode_length_buf == 0, as_tuple=False).flatten()
            if len(reset_ids) > 0:
                self.reset(env, reset_ids)

        state["last_step"] = step_index
        state["cache_key"] = None
        state["cache_value"] = None
        return state

    def position(self, env: ManagerBasedRlEnv, *, noisy: bool, delayed: bool) -> torch.Tensor:
        state = self._update(env)
        step_index = _get_env_step_index(env)
        cache_key = (step_index, noisy, delayed)
        if state["cache_key"] == cache_key and state["cache_value"] is not None:
            return state["cache_value"]

        lag = state["lag"] if delayed else torch.zeros_like(state["lag"])
        env_ids = torch.arange(state["pos_history"].shape[0], device=state["pos_history"].device)
        pos = state["pos_history"][env_ids, lag]
        if noisy:
            pos = pos + torch.randn_like(pos) * self.position_noise_std

        state["cache_key"] = cache_key
        state["cache_value"] = pos
        return pos


BALL_OBSERVATION_CACHE = BallObservationCache()


def obs_noisy_delayed_ball_pos_body_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    return BALL_OBSERVATION_CACHE.position(env, noisy=True, delayed=True)


def obs_gait_phase(env: ManagerBasedRlEnv, command_name: str = "kick") -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    if command_term is None or not hasattr(command_term, "phase_observation"):
        return torch.zeros((env.num_envs, 4), device=env.device)
    return getattr(command_term, "phase_observation")


def obs_kick_direction_body_xy(env: ManagerBasedRlEnv, command_name: str = "kick") -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    if command_term is None or not hasattr(command_term, "kick_direction_b"):
        return torch.zeros((env.num_envs, 2), device=env.device)
    return getattr(command_term, "kick_direction_b")[:, :2]


def obs_noisy_kick_command(
    env: ManagerBasedRlEnv,
    command_name: str = "kick",
    angle_std: float = _KICK_DIR_ANGLE_NOISE_STD,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    if command_term is None or not hasattr(command_term, "kick_direction_b"):
        return torch.zeros((env.num_envs, 3), device=env.device)

    normalizer = float(getattr(getattr(command_term, "cfg", None), "max_ball_speed_normalizer", 5.0))
    command = torch.cat(
        (
            getattr(command_term, "kick_direction_b")[:, :2],
            (getattr(command_term, "max_ball_speed") / normalizer).unsqueeze(-1),
        ),
        dim=-1,
    )

    step_index = _get_env_step_index(env)
    cache = getattr(env, "_k1_kick_noisy_command_cache", None)
    if step_index is not None and cache is not None:
        cached_source = cache.get("source")
        if (
            cache.get("command_name") == command_name
            and cache.get("angle_std") == angle_std
            and cache.get("step_index") == step_index
            and cached_source is not None
            and cached_source.shape == command.shape
            and torch.equal(cached_source, command)
        ):
            return cache["value"]

    direction = command[:, :2]
    angle_noise = torch.randn(direction.shape[0], device=direction.device, dtype=direction.dtype) * angle_std
    cos_noise = torch.cos(angle_noise)
    sin_noise = torch.sin(angle_noise)
    noisy_direction = torch.stack(
        (
            cos_noise * direction[:, 0] - sin_noise * direction[:, 1],
            sin_noise * direction[:, 0] + cos_noise * direction[:, 1],
        ),
        dim=-1,
    )
    noisy_command = torch.cat((noisy_direction, command[:, 2:3]), dim=-1)

    if step_index is not None:
        setattr(
            env,
            "_k1_kick_noisy_command_cache",
            {
                "command_name": command_name,
                "angle_std": angle_std,
                "step_index": step_index,
                "source": command.detach().clone(),
                "value": noisy_command,
            },
        )
    return noisy_command


def obs_ball_speed(env: ManagerBasedRlEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    ball: Entity = env.scene[ball_cfg.name]
    return torch.linalg.norm(ball.data.root_link_vel_w[:, :3], dim=-1, keepdim=True)


def obs_ball_height(env: ManagerBasedRlEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    ball: Entity = env.scene[ball_cfg.name]
    return ball.data.root_link_pos_w[:, 2:3]


def obs_base_lin_vel_body(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b


def obs_base_ang_vel_world(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_w


def obs_root_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]


def obs_actuator_forces(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force[:, asset_cfg.actuator_ids]


def obs_foot_lin_vel_world(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :].flatten(start_dim=1)


def obs_foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    if current_air_time is None:
        return torch.zeros((env.num_envs, 0), device=env.device)
    return torch.nan_to_num(current_air_time, nan=0.0, posinf=0.0, neginf=0.0)


def obs_reached_ball(env: ManagerBasedRlEnv, command_name: str = "kick") -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    if command_term is None or not hasattr(command_term, "reached_ball"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return getattr(command_term, "reached_ball").float().unsqueeze(-1)


def make_observation_cfg(controlled_joint_names: tuple[str, ...] | None = None) -> dict[str, ObservationGroupCfg]:
    joint_asset_cfg = SceneEntityCfg(
        "robot",
        joint_names=controlled_joint_names,
        preserve_order=True,
    )
    foot_site_cfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    actuator_asset_cfg = SceneEntityCfg("robot", actuator_names=".*")

    policy_terms = {
        "ball_pos": ObservationTermCfg(
            func=obs_noisy_delayed_ball_pos_body_xy,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=obs_biased_gyro,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=GaussianNoiseCfg(mean=0.0, std=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=obs_biased_projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": joint_asset_cfg, "biased": True},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
        ),
        "phase": ObservationTermCfg(
            func=obs_gait_phase,
            params={"command_name": "kick"},
        ),
        "command": ObservationTermCfg(
            func=obs_noisy_kick_command,
            params={"command_name": "kick"},
        ),
    }

    critic_terms = {
        **policy_terms,
        "ball_speed": ObservationTermCfg(
            func=obs_ball_speed,
        ),
        "true_ball_pos": ObservationTermCfg(
            func=obs_ball_pos_body_xy,
        ),
        "true_kick_direction": ObservationTermCfg(
            func=obs_kick_direction_body_xy,
            params={"command_name": "kick"},
        ),
        "ball_height": ObservationTermCfg(
            func=obs_ball_height,
        ),
        "true_gyro": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
        ),
        "accelerometer": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_acc"},
        ),
        "true_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
        ),
        "base_lin_vel": ObservationTermCfg(
            func=obs_base_lin_vel_body,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "base_ang_vel_world": ObservationTermCfg(
            func=obs_base_ang_vel_world,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "true_joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": joint_asset_cfg},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": joint_asset_cfg},
        ),
        "root_height": ObservationTermCfg(
            func=obs_root_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "actuator_forces": ObservationTermCfg(
            func=obs_actuator_forces,
            params={"asset_cfg": actuator_asset_cfg},
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_vel": ObservationTermCfg(
            func=obs_foot_lin_vel_world,
            params={"asset_cfg": foot_site_cfg},
        ),
        "foot_air_time": ObservationTermCfg(
            func=obs_foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "reached_ball": ObservationTermCfg(
            func=obs_reached_ball,
            params={"command_name": "kick"},
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
            enable_corruption=True,
            history_length=1,
        ),
    }
