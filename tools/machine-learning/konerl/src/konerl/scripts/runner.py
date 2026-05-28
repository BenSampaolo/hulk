from __future__ import annotations

import os
from pathlib import Path
import time

import torch
import numpy as np
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .AMP import (
    AMPDiscriminator,
    AMPOptimizer,
    AMPConfig,
    MocapBuffer,
    MOCAP_TO_K1,
    amp_features_from_robot,
    controlled_joint_names_from_env,
)


class SimpleAMPBuilder:
    """
    A minimal builder to interface with MocapBuffer.load().
    Maps raw mocap dictionaries (e.g. from .pkl or .npy) to the AMP feature format
    dynamically based on the active actuators.
    """
    def __init__(self, active_joints: list[str]):
        self.active_joints = active_joints
        self.k1_to_mocap = {v: k for k, v in MOCAP_TO_K1.items()}
        self.feature_dim = len(self.active_joints) * 2 + 6  # pos and vel + root vel and ang vel

    def build_from_mocap_dict(self, raw_data: dict) -> np.ndarray | None:
        assert 'link_body_list' in raw_data, "Mocap data missing 'link_body_list'"
            
        node_names = raw_data['link_body_list']
        
        indices = []
        for j in self.active_joints:
            mocap_name = self.k1_to_mocap.get(j)
            if mocap_name and mocap_name in node_names:
                indices.append(node_names.index(mocap_name) - 1) # mocap data has a root node at index 0
            else:
                print(f"[AMP Builder] Warning: Could not map active joint '{j}' to mocap nodes.")
                return None

        dof_pos = raw_data["dof_pos"]
        dof_vel = raw_data["dof_vel"]
        root_vel = raw_data["root_vel"]
        root_ang_vel = raw_data["root_ang_vel"]

        return np.concatenate(
            [dof_pos[:, indices], dof_vel[:, indices], root_vel, root_ang_vel],
            axis=-1,
        ).astype(np.float32)


class MjlabAMPOnPolicyRunner(VelocityOnPolicyRunner):
    """Mjlabs default on-policy runner with AMP"""

    alg: PPO
    env: RslRlVecEnvWrapper
    discriminator: AMPDiscriminator

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self.active_joints = list(controlled_joint_names_from_env(self.env))
        if not self.active_joints:
            raise RuntimeError("AMP requires at least one controlled robot joint")

        builder = SimpleAMPBuilder(self.active_joints)
        self.amp_feature_dim = builder.feature_dim
        self.num_envs = self.env.num_envs

        self.dt = self.env.cfg.sim.mujoco.timestep if self.env.cfg.scale_rewards_by_dt else 1.0

        self.amp_config = AMPConfig()

        self.discriminator = AMPDiscriminator(
            input_dim=self.amp_feature_dim,
            history_length=self.amp_config.history_length,
            hidden_dims=[256, 256],
        ).to(self.device)

        amp_data_dir = Path(self.cfg.get("amp_data_dir", "motions/CMU_Certified_Speed"))
        
        if not amp_data_dir.exists() or not any(amp_data_dir.glob("**/*.*")):
            raise FileNotFoundError(f"AMP mocap data directory '{amp_data_dir}' not found or empty")
        self.expert_buffer = MocapBuffer.load(amp_data_dir, builder, self.amp_config.history_length, self.device)

        self.amp_optimizer = AMPOptimizer(self.amp_config, self.device, self.expert_buffer, self.discriminator)

        setattr(self.env.unwrapped, "amp_optimizer", self.amp_optimizer)
        
        self.amp_history = torch.zeros((self.num_envs, self.amp_config.history_length, self.amp_feature_dim), device=self.device)

    def _broadcast_amp_state(self) -> None:
        if not self.is_distributed:
            return
        amp_state = [
            self.discriminator.state_dict(),
            self.amp_optimizer.optimizer.state_dict(),
        ]
        torch.distributed.broadcast_object_list(amp_state, src=0)
        self.discriminator.load_state_dict(amp_state[0])
        self.amp_optimizer.optimizer.load_state_dict(amp_state[1])

    def _compute_amp_features(self, env: RslRlVecEnvWrapper, obs: torch.Tensor) -> torch.Tensor:
        """
        Helper to compute AMP features directly in the learning loop.
        Map the current state (from 'obs' or 'env.unwrapped') to the AMP feature space.
        """
        robot = env.unwrapped.scene["robot"]
        return amp_features_from_robot(robot, self.active_joints, self.device)

    def save(self, path: str, infos=None) -> None:
        amp_state = {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.amp_optimizer.optimizer.state_dict(),
            "normalizer": self.discriminator.normalizer.state_dict(),
            "config": self.amp_config,
        }
        infos = {**(infos or {}), "amp_state": amp_state}
        super().save(path, infos=infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        amp_state = infos.get("amp_state") if infos else None
        if amp_state is not None:
            self.discriminator.load_state_dict(amp_state["discriminator"], strict=strict)
            self.amp_optimizer.optimizer.load_state_dict(amp_state["optimizer"])
            if "normalizer" in amp_state:
                self.discriminator.normalizer.load_state_dict(amp_state["normalizer"], strict=strict)
        return infos

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            self._broadcast_amp_state()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            amp_rollout_data = []

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    
                    amp_features = self._compute_amp_features(self.env, obs)
                    self.amp_history = torch.roll(self.amp_history, shifts=-1, dims=1)
                    self.amp_history[:, -1, :] = amp_features
                    
                    reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    if len(reset_env_ids) > 0:
                        self.amp_history[reset_env_ids] = amp_features[reset_env_ids].unsqueeze(1).expand(-1, self.amp_config.history_length, -1)
                    
                    amp_rollout_data.append(self.amp_history.clone())

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            amp_obs_batch = torch.cat(amp_rollout_data, dim=0) # [num_steps * num_envs, H, D]
            if not self.is_distributed or self.gpu_global_rank == 0:
                amp_metrics = self.amp_optimizer.train_step(amp_obs_batch)
                amp_metrics["amp/steps_skipped"] = self.amp_optimizer.skip_counter
                loss_dict.update(amp_metrics)
            self._broadcast_amp_state()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=getattr(self.alg.rnd, 'weight', None) if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
