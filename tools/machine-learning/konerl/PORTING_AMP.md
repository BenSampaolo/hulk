# Porting AMP to Konerl

## Plan

1. [x] Copy the necessary AMP components (`discriminator.py`, `optimizer.py`, `mocap_buffer.py`) from the original `k1-rl-walking` repository to the new destination (`/Users/ben/hulks/hulk-konerl/tools/machine-learning/konerl/src/konerl/scripts/AMP/`).
2. [x] Adapt the components for pure PyTorch usage (removing `warp` dependencies).
3. [x] Integrate the AMP components into the `MjlabAMPOnPolicyRunner` in `runner.py`.
4. [x] Define the feature builder logic for AMP directly within the runner (`_compute_amp_features`).
5. [x] Provide a summary of the porting progress.

## Summary

The AMP logic has been fully ported into `runner.py` and the `AMP` directory as raw, pure PyTorch modules.

- **`discriminator.py` & `normalizer.py`**: Pure PyTorch modules handling the ML architecture.
- **`optimizer.py`**: The GAN loop. Exposes `.train_step(amp_obs_batch)` and `.calculate_amp_rewards(history_batch)`. `warp` and `torchmetrics` were completely stripped.
- **`runner.py`**:
  - Subclasses `MjlabOnPolicyRunner`.
  - Defines a generic `SimpleAMPBuilder` placeholder for `mocap_buffer.py` to use when iterating over `.pkl` files.
  - Maintains a sliding `amp_history` buffer `[num_envs, History, FeatureDim]`.
  - At each step, uses `_compute_amp_features` to grab `amp_features`. It rolls the history window and inserts the new observation.
  - Generates the AMP reward (`amp_optimizer.calculate_amp_rewards`) and adds it directly to the environment's `rewards`.
  - At the end of the rollout, before PPO updates, it concatenates the rolled history data and passes it to `amp_optimizer.train_step(amp_obs_batch)`. The discriminator update metrics are automatically forwarded to the logger via `loss_dict.update(amp_metrics)`.

### Next Steps for Implementation

1. **Feature Parsing**: In `runner.py`, implement the logic inside `_compute_amp_features` to parse the `obs` tensor (or query the environment) into the actual joint/state vectors you wish the Discriminator to see.
2. **Mocap Build Step**: Modify `SimpleAMPBuilder.build_from_mocap_dict` in `runner.py` so that it parses your specific `.pkl` files into exactly the same shape and order of features outputted by `_compute_amp_features`.
3. Configuration handling: Right now the `amp_reward_weight` defaults to `0.5`, `history_length` to `5`, and `amp_data_dir` to `motions/`. Make sure these exist in your experiment configs.

## Update: Dynamic Mocap Alignment

The `SimpleAMPBuilder` in `runner.py` now dynamically pulls the `active_joints` from the `env.cfg.scene.entities["robot"].articulation.actuators` config. 

It maps the internal K1 joint names (e.g. `ALeft_Shoulder_Pitch`) to the CMU `.npy` Mocap names (e.g. `Left_Arm_1`) using a predefined dictionary `MOCAP_TO_K1`. When loading motions, it automatically plucks out only the rotation arrays and angular velocity arrays for those specific active joints and creates a flat feature vector per timestep. 

The `amp_feature_dim` is now dynamically computed as `len(active_joints) * 7` (4 dims for quaternion rotation + 3 dims for angular velocity).
