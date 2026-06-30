import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp


def fallen_past_90_degrees(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 2] > 0.0


def nan_state(env) -> torch.Tensor:
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for entity_name in ("robot", "ball"):
        asset = env.scene[entity_name]
        for attr in ("root_link_pos_w", "root_link_quat_w", "root_link_vel_w", "joint_pos", "joint_vel"):
            value = getattr(asset.data, attr, None)
            if isinstance(value, torch.Tensor):
                done |= ~torch.isfinite(value.reshape(env.num_envs, -1)).all(dim=1)
    return done


def make_termination_cfg() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=fallen_past_90_degrees,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "nan_state": TerminationTermCfg(func=nan_state),
    }
