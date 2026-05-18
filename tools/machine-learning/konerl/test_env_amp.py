import torch
import numpy as np
from konerl.scripts.runner import SimpleAMPBuilder, MjlabAMPOnPolicyRunner
from rsl_rl.runners import OnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from konerl.tasks.k1_velocity_tracking import * # Trigger registration

def main():
    cfg = load_env_cfg("Mjlab-Velocity-Rough-K1")
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    wrapped_env = RslRlVecEnvWrapper(env)
    
    class DummyRunner:
        def __init__(self):
            self.device = "cpu"
            self.active_joints = list(SimpleAMPBuilder.MOCAP_TO_K1.values())
            
        def _compute_amp_features(self, env: RslRlVecEnvWrapper, obs: torch.Tensor) -> torch.Tensor:
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
            j_pos = robot.data.joint_pos[:, jnt_ids]
            j_vel = robot.data.joint_vel[:, jnt_ids]
            return torch.cat([j_pos, j_vel], dim=-1)

    runner = DummyRunner()
    
    obs, _ = wrapped_env.reset()
    features = runner._compute_amp_features(wrapped_env, obs)
    
    print(f"Features shape from simulation: {features.shape}")
    print(f"Num ENVs: {wrapped_env.num_envs}")
    print(f"Feature Dim: {len(runner.active_joints) * 2}")

    # Step the env and check again
    actions = torch.zeros((wrapped_env.num_envs, wrapped_env.num_actions))
    obs, rewards, dones, infos = wrapped_env.step(actions)
    
    features2 = runner._compute_amp_features(wrapped_env, obs)
    print(f"Features shape from simulation (step 1): {features2.shape}")
    
    # Check if they changed (meaning we got actual dynamic data)
    diff = (features - features2).abs().mean().item()
    print(f"Mean diff between steps: {diff:.6f}")

if __name__ == "__main__":
    main()
