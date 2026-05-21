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

from .AMP import AMPDiscriminator, AMPOptimizer, AMPConfig, MocapBuffer


class SimpleAMPBuilder:
    """
    A minimal builder to interface with MocapBuffer.load().
    Maps raw mocap dictionaries (e.g. from .pkl or .npy) to the AMP feature format
    dynamically based on the active actuators.
    """
    MOCAP_TO_K1 = {
        'Head1': 'AAHead_Yaw',
        'Head2': 'Head_Pitch',
        'Left_Arm_1': 'ALeft_Shoulder_Pitch',
        'Right_Arm_1': 'ARight_Shoulder_Pitch',
        'Left_Arm_2': 'Left_Shoulder_Roll',
        'Right_Arm_2': 'Right_Shoulder_Roll',
        'Left_Arm_3': 'Left_Elbow_Pitch',
        'Right_Arm_3': 'Right_Elbow_Pitch',
        'left_hand_link': 'Left_Elbow_Yaw', 
        'right_hand_link': 'Right_Elbow_Yaw',
        
        'Left_Hip_Pitch': 'Left_Hip_Pitch',
        'Right_Hip_Pitch': 'Right_Hip_Pitch',
        'Left_Hip_Roll': 'Left_Hip_Roll',
        'Right_Hip_Roll': 'Right_Hip_Roll',
        'Left_Hip_Yaw': 'Left_Hip_Yaw',
        'Right_Hip_Yaw': 'Right_Hip_Yaw',
        
        'Left_Shank': 'Left_Knee_Pitch',
        'Right_Shank': 'Right_Knee_Pitch',
        
        'Left_Ankle_Cross': 'Left_Ankle_Pitch',
        'Right_Ankle_Cross': 'Right_Ankle_Pitch',
        
        'left_foot_link': 'Left_Ankle_Roll',
        'right_foot_link': 'Right_Ankle_Roll'
    }

    def __init__(self, active_joints: list[str]):
        self.active_joints = active_joints
        self.k1_to_mocap = {v: k for k, v in self.MOCAP_TO_K1.items()}
        self.feature_dim = len(self.active_joints) * 2 # pos and vel

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

        features = np.concatenate([dof_pos[:, indices], dof_vel[:, indices]], axis=-1).astype(np.float32)
        return features


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
        self.active_joints = []
        if hasattr(self.env.cfg.scene.entities.get("robot", None), "articulation"):
            articulation = self.env.cfg.scene.entities["robot"].articulation
            assert articulation is not None, "Articulation not found in robot entity."
            actuators = articulation.actuators
            for act in actuators:
                if hasattr(act, 'target_names_expr'):
                    self.active_joints.extend(act.target_names_expr)
                    
        if not self.active_joints:
            print("[AMP] Warning: No active joints found in env config. Falling back to default list.")
            self.active_joints = list(SimpleAMPBuilder.MOCAP_TO_K1.values())

        builder = SimpleAMPBuilder(self.active_joints)
        self.amp_feature_dim = builder.feature_dim
        self.num_envs = self.env.num_envs

        self.dt = self.env.cfg.sim.mujoco.timestep if self.env.cfg.scale_rewards_by_dt else 1.0

        self.amp_config = AMPConfig(
            history_length=self.cfg.get("history_length", 20),
            batch_size=self.cfg.get("amp_batch_size", 512),
            weight_decay=self.cfg.get("amp_weight_decay", 1e-3),
            gradient_penalty_weight=self.cfg.get("amp_gradient_penalty_weight", 5.0),
            learning_rate=self.cfg.get("amp_learning_rate", 1e-5),
        )

        self.discriminator = AMPDiscriminator(
            input_dim=self.amp_feature_dim,
            history_length=self.amp_config.history_length,
            hidden_dims=[128, 128],
        ).to(self.device)

        amp_data_dir = Path(self.cfg.get("amp_data_dir", "motions/CMU/"))
        
        if amp_data_dir.exists() and any(amp_data_dir.glob("**/*.*")):
            self.expert_buffer = MocapBuffer.load(amp_data_dir, builder, self.amp_config.history_length, self.device)
        else:
            print(f"[AMP] Warning: Mocap data directory '{amp_data_dir}' not found or empty. Creating a dummy expert buffer for functional testing.")
            dummy_obs = torch.zeros((1000, self.amp_feature_dim), device=self.device)
            dummy_idx = torch.arange(1000 - self.amp_config.history_length + 1, device=self.device)
            self.expert_buffer = MocapBuffer(dummy_obs, dummy_idx, self.amp_config.history_length, self.device)

        self.amp_optimizer = AMPOptimizer(self.amp_config, self.device, self.expert_buffer, self.discriminator)
        
        self.amp_history = torch.zeros((self.num_envs, self.amp_config.history_length, self.amp_feature_dim), device=self.device)

    def _compute_amp_features(self, env: RslRlVecEnvWrapper, obs: torch.Tensor) -> torch.Tensor:
        """
        Helper to compute AMP features directly in the learning loop.
        Map the current state (from 'obs' or 'env.unwrapped') to the AMP feature space.
        """
        robot = env.unwrapped.scene["robot"]

        joint_names = robot.joint_names
        joint_indices = []
        for name in self.active_joints:
            try:
                idx = joint_names.index(name)
                joint_indices.append(idx)
            except ValueError:
                joint_indices.append(0) 

        jnt_ids = torch.tensor(joint_indices, device=self.device, dtype=torch.long)
        
        j_pos = robot.data.joint_pos[:, jnt_ids]  # [num_envs, num_active_joints]
        j_vel = robot.data.joint_vel[:, jnt_ids]  # [num_envs, num_active_joints]
        
        return torch.cat([j_pos, j_vel], dim=-1)

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

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            amp_rollout_data = []
            
            # Rollout
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

                    amp_reward = self.amp_optimizer.calculate_amp_rewards(self.amp_history).squeeze(-1)
                    amp_reward_weight = self.cfg.get("amp_reward_weight", 4.0)
                    rewards = rewards + (amp_reward_weight * amp_reward * self.dt)
                    
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
            amp_metrics = self.amp_optimizer.train_step(amp_obs_batch)
            loss_dict.update(amp_metrics)

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
