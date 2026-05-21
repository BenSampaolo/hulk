import torch.nn as nn
from torch import Tensor

from .normalizer import ObservationNormalizer

class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        history_length: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.history_length = history_length

        mlp_input_dim = input_dim * history_length

        self.normalizer = ObservationNormalizer((mlp_input_dim,))

        layers = []
        curr_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.utils.spectral_norm(nn.Linear(curr_dim, hidden_dim)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, 1))
        self.trunk = nn.Sequential(*layers)

    def forward(self, amp_observations: Tensor) -> Tensor:
        """
        Args:
            amp_observations: (Batch, History, Feature_Dim)
        """
        batch_size = amp_observations.shape[0]

        # Flatten time and features: (B, T, D) -> (B, T*D)
        flattened_obs = amp_observations.view(batch_size, -1)

        normalized_obs = self.normalizer(flattened_obs)
        return self.trunk(normalized_obs)
