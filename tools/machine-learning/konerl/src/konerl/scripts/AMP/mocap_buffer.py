import pickle
from pathlib import Path
import torch
import numpy as np

class MocapBuffer:
    def __init__(self, observations: torch.Tensor, valid_indices: torch.Tensor, history_length: int, device):
        self.observations = observations
        self.valid_indices = valid_indices
        self.history_length = history_length
        self.device = device

    @classmethod
    def load(cls, data_dir: Path, builder, history_length: int, device: torch.device | str):
        # Allow loading either pkl or npy files
        files = [f for f in data_dir.glob("**/*") if f.suffix in (".pkl", ".npy") and not f.name.endswith(".qpos.pkl")]
        if not files:
            raise FileNotFoundError(f"No motion files (.pkl/.npy) in {data_dir}")

        obs_list = []
        indices_list = []
        offset = 0

        print(f"[AMP] Loading {len(files)} motion files...")

        for f in files:
            try:
                if f.suffix == ".pkl":
                    raw_data = pickle.loads(f.read_bytes())
                else: # .npy
                    raw_data = np.load(f, allow_pickle=True).item()

                # Build Feature Vector using the BUILDER
                # The builder should handle upness checks and extracting the right tensors
                features = builder.build_from_mocap_dict(raw_data)
                
                if features is None:
                    continue

                n_steps = features.shape[0]
                if n_steps < history_length:
                    continue

                obs_list.append(torch.from_numpy(features))

                # Calculate valid start indices for history windows
                valid_starts = torch.arange(n_steps - history_length + 1) + offset
                indices_list.append(valid_starts)

                offset += n_steps

            except Exception as e:
                print(f"Skipping {f.name}: {e}")

        if not obs_list:
            raise RuntimeError("No valid motion data found!")

        full_obs = torch.cat(obs_list, dim=0).to(device=device, dtype=torch.float32)
        full_idx = torch.cat(indices_list, dim=0).to(device=device, dtype=torch.long)
        
        print(f"[AMP] Successfully loaded {len(obs_list)} valid motion files into {full_obs.shape[0]} frames.")

        return cls(full_obs, full_idx, history_length, device)

    def sample(self, batch_size: int) -> torch.Tensor:
        # Randomly select valid start points
        idx_indices = torch.randint(0, self.valid_indices.size(0), (batch_size,), device=self.device)
        start_frames = self.valid_indices[idx_indices]

        # Create windows [B, H, D]
        # [0, 1, 2...]
        offsets = torch.arange(self.history_length, device=self.device)
        # [B, H]
        gather_indices = start_frames.unsqueeze(1) + offsets.unsqueeze(0)

        flat_data = self.observations[gather_indices.view(-1)]
        return flat_data.view(batch_size, self.history_length, -1)
