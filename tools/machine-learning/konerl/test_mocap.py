import numpy as np
import torch
from konerl.scripts.runner import SimpleAMPBuilder

def main():
    data = np.load('motions/CMU/01/01_01_stageii.npy', allow_pickle=True).item()
    active_joints = list(SimpleAMPBuilder.MOCAP_TO_K1.values())
    builder = SimpleAMPBuilder(active_joints)
    features = builder.build_from_mocap_dict(data)
    
    print(f"Features shape: {features.shape}")
    print(f"Contains NaNs: {np.isnan(features).any()}")
    
    # Check stats for first few pos and vel
    print("\nPosition stats (min, max, mean):")
    for i in range(min(5, len(active_joints))):
        print(f"{active_joints[i]}: {features[:, i].min():.3f}, {features[:, i].max():.3f}, {features[:, i].mean():.3f}")
        
    num_joints = len(active_joints)
    print("\nVelocity stats (min, max, mean):")
    for i in range(min(5, len(active_joints))):
        v_idx = i + num_joints
        print(f"{active_joints[i]}: {features[:, v_idx].min():.3f}, {features[:, v_idx].max():.3f}, {features[:, v_idx].mean():.3f}")

if __name__ == "__main__":
    main()
