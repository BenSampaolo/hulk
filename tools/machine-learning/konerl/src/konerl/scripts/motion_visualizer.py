import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from konerl.k1_config import K1_DEFAULT_JOINT_POS
from konerl.scripts.AMP.features import MOCAP_TO_K1


class MocapPlaybackPolicy:
    """Directly injects ground-truth mocap frames into the simulation state."""
    
    def __init__(self, env: RslRlVecEnvWrapper, raw_mocap_data: dict, action_scale: float = 0.5):
        self.env = env
        self.unwrapped_env = env.unwrapped
        self.action_shape = env.action_space.shape
        self.device = env.device
        self.dof_pos = raw_mocap_data["dof_pos"] 
        self.dof_vel = raw_mocap_data["dof_vel"]
        self.num_frames = self.dof_pos.shape[0]
        self.frame_idx = 0
        self.action_scale = action_scale

        self.active_joints = []
        if hasattr(self.unwrapped_env.cfg.scene.entities.get("robot", None), "articulation"):
            articulation = self.unwrapped_env.cfg.scene.entities["robot"].articulation
            if articulation is not None:
                for act in articulation.actuators:
                    if hasattr(act, 'target_names_expr'):
                        self.active_joints.extend(act.target_names_expr)
        if not self.active_joints:
            self.active_joints = list(MOCAP_TO_K1.values())
        self.default_offset = np.array(
            [K1_DEFAULT_JOINT_POS.get(joint_name, K1_DEFAULT_JOINT_POS[".*"]) for joint_name in self.active_joints],
            dtype=np.float32,
        )

        self.model = getattr(self.unwrapped_env, "model", None)
        self.data = getattr(self.unwrapped_env, "data", None)
        
        k1_to_mocap = {v: k for k, v in MOCAP_TO_K1.items()}
        node_names = raw_mocap_data['link_body_list']
        
        self.mocap_to_mj_map = []
        for j_idx, j_name in enumerate(self.active_joints):
            mocap_name = k1_to_mocap.get(j_name)
            if mocap_name and mocap_name in node_names:
                mocap_joint_idx = node_names.index(mocap_name) - 1
                self.mocap_to_mj_map.append((mocap_joint_idx, j_name))

    def __call__(self, obs) -> torch.Tensor:
        del obs
        
        if self.model is not None and self.data is not None:
            target_pos = self.dof_pos[self.frame_idx] * self.action_scale + self.default_offset
            target_vel = self.dof_vel[self.frame_idx]

            for mocap_idx, j_name in self.mocap_to_mj_map:
                try:
                    import mujoco
                    j_id = mujoco.mj_name2id(self.model, int(mujoco.mjtObj.mjOBJ_JOINT), j_name)
                    if j_id != -1:
                        qpos_adr = self.model.jnt_qposadr[j_id]
                        qvel_adr = self.model.jnt_dofadr[j_id]
                        
                        self.data.qpos[qpos_adr] = target_pos[mocap_idx]
                        self.data.qvel[qvel_adr] = target_vel[mocap_idx]
                except Exception:
                    pass

            self.data.qpos[0:3] = [0.0, 0.0, 0.9]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[0:6] = 0.0

        self.frame_idx = (self.frame_idx + 1) % self.num_frames
        
        return torch.zeros(self.action_shape, device=self.device)


@dataclass(frozen=True)
class PlayConfig:
  mocap_pkl_file: str | None = None
  """Path to local .pkl motion file containing the mocap dictionary data."""
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.terminations = {}

  if cfg.mocap_pkl_file is None:
      raise ValueError("You must provide a path via `--mocap-pkl-file` to visualize ground truths.")
      
  pkl_path = Path(cfg.mocap_pkl_file)
  if not pkl_path.exists():
    raise FileNotFoundError(f"Mocap file not found: {pkl_path}")
  
  print(f"[INFO]: Injecting local ground truth mocap data from: {pkl_path}")
  raw_mocap_data: dict = np.load(pkl_path, allow_pickle=True)
  assert "dof_pos" in raw_mocap_data and "dof_vel" in raw_mocap_data

  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  policy = MocapPlaybackPolicy(env, raw_mocap_data, action_scale=0.5)

  if cfg.viewer == "auto":
    resolved_viewer = "native" if bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy, checkpoint_manager=None).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  maybe_print_top_level_help("play")
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()