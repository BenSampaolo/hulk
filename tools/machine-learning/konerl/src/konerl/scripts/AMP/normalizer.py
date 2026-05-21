import torch
import torch.nn as nn

class ObservationNormalizer(nn.Module):
    """
    Computes a running mean and variance for observation normalization.

    This is a stateful module. Its update must be called
    (e.g., with the rollout buffer) before the PPO train_step.
    """

    def __init__(self, observation_shape: tuple):
        super().__init__()
        self.shape = observation_shape

        # We use float64 for the running stats to avoid precision loss
        # over millions of steps, but normalize to float32
        self.running_mean = nn.Parameter(torch.zeros(self.shape, dtype=torch.float64), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(self.shape, dtype=torch.float64), requires_grad=False)
        self.count = nn.Parameter(torch.tensor(1e-4, dtype=torch.float64), requires_grad=False)

        self.epsilon = 1e-8

    @torch.no_grad()
    def update(self, observations: torch.Tensor):
        """
        Updates the running mean and variance with a new batch of observations.
        'obs_batch' should be a 2D tensor [batch_size, obs_dim].
        """
        # Flatten anything > 2D
        if observations.ndim > 2:
            observations = observations.view(-1, observations.size(-1))

        mean = observations.mean(dim=0).to(torch.float64)
        variance = observations.var(dim=0, unbiased=False).to(torch.float64)
        count = observations.size(0)

        delta = mean - self.running_mean
        tot_count = self.count + count

        # --- Welford's online algorithm for parallel update ---
        new_mean = self.running_mean + delta * (count / tot_count)

        m_a = self.running_var * self.count
        m_b = variance * count
        m_2 = m_a + m_b + torch.square(delta) * (self.count * count / tot_count)

        new_var = m_2 / tot_count

        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(tot_count)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a batch of observations.
        """
        original_shape = obs.shape
        if obs.ndim > 2:
            obs = obs.view(-1, obs.size(-1))
            
        _, nfeatures = obs.shape
        # Note: We use .to(obs.dtype) to convert our float64 stats
        # to float32 for the network pass.
        mean = self.running_mean.to(obs.dtype)
        std = torch.sqrt(self.running_var + self.epsilon).to(obs.dtype)

        normalized_obs = (obs - mean[:nfeatures]) / std[:nfeatures]

        # Clip to prevent extreme values from rare states
        clamped = torch.clamp(normalized_obs, -50.0, 50.0)
        
        return clamped.view(original_shape)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.running_var + self.epsilon).to(torch.float32)

    @property
    def mean(self) -> torch.Tensor:
        return self.running_mean.to(torch.float32)
