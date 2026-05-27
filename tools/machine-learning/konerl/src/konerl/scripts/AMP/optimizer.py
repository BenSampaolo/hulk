import torch
from torch.optim import Adam
import dataclasses

from .discriminator import AMPDiscriminator
from .mocap_buffer import MocapBuffer

import numpy as np

class DiscriminatorReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device | str):
        self.capacity = capacity
        self.device = device
        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, observations: torch.Tensor) -> None:
        n = observations.shape[0]
        if n > self.capacity:
            indices = torch.randperm(n, device=self.device)[: self.capacity]
            self.observations[:] = observations[indices]
            self.ptr, self.size = 0, self.capacity
        else:
            space = self.capacity - self.ptr
            if n <= space:
                self.observations[self.ptr : self.ptr + n] = observations
                self.ptr = (self.ptr + n) % self.capacity
            else:
                self.observations[self.ptr :] = observations[:space]
                self.observations[: n - space] = observations[space:]
                self.ptr = n - space
            self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.observations[indices]


@dataclasses.dataclass
class AMPConfig:
    learning_rate: float = 1.0e-4
    weight_decay: float = 1e-3
    replay_buffer_size: int = 1000000
    history_length: int = 1
    batch_size: int = 48
    gradient_penalty_weight: float = 1.0
    update_interval: int = 1
    instance_noise_std: float = 0.0
    target_accuracy: float = 1.0


class AMPOptimizer:
    def __init__(
        self,
        config: AMPConfig,
        device: torch.device | str,
        expert_buffer: MocapBuffer,
        discriminator: AMPDiscriminator,
    ):
        self.config = config
        self.device = device
        self.expert_buffer = expert_buffer
        self.discriminator = discriminator

        self.optimizer = Adam(
            self.discriminator.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        self.input_dim = discriminator.input_dim

        self.replay_buffer = DiscriminatorReplayBuffer(
            capacity=config.replay_buffer_size, obs_shape=(config.history_length, self.input_dim), device=self.device
        )
        self.step_count = 0

        self.metrics = {
            "amp/loss": [],
            "amp/logits_real": [],
            "amp/logits_fake": [],
            "amp/grad_penalty": [],
            "amp/accuracy_real": [],
            "amp/accuracy_fake": [],
        }

        self._initialize_normalizer()

        self.skip_counter = 0

    def _initialize_normalizer(self) -> None:
        """
        Computes mean and variance from the expert buffer and freezes the discriminator's normalizer.
        This ensures stationary input distributions for the discriminator.
        """
        print("[AMP] Computing fixed normalizer statistics from expert data...")

        n_samples = min(len(self.expert_buffer.valid_indices), 100000)
        expert_batch = self.expert_buffer.sample(n_samples)  # Shape: (N, History, Dim)

        flat_expert_batch = expert_batch.view(n_samples, -1)

        mean = torch.mean(flat_expert_batch, dim=0).double()
        var = torch.var(flat_expert_batch, dim=0).double()

        self.discriminator.normalizer.running_mean.data.copy_(mean)
        self.discriminator.normalizer.running_var.data.copy_(var)
        self.discriminator.normalizer.count.data.fill_(1e9)  # Prevent incidental updates

        print(f"[AMP] Normalizer initialized with {n_samples} expert samples.")

    @torch.inference_mode()
    def calculate_amp_rewards(self, amp_observations: torch.Tensor) -> torch.Tensor:
        """
        Computes AMP reward for given AMP observations.
        The discriminator logits are +1 when the observation is classified as expert-like.
        """
        with torch.no_grad():
            d_logits = self.discriminator(amp_observations)
        # LSGAN Reward: Maximize D(s) -> 1
        return torch.maximum(torch.zeros_like(d_logits), 1.0 - 0.25 * torch.square(d_logits - 1))

    def train_step(self, amp_obs: torch.Tensor) -> dict[str, float]:
        """
        amp_obs: expected to be collected history from rollout of shape [N, H, D]
        """
        self.step_count += 1
        if self.step_count % self.config.update_interval != 0:
            return {}

        for k in self.metrics.keys():
            self.metrics[k] = []

        if amp_obs.shape[0] > 0:
            self.replay_buffer.add(amp_obs.detach())

        iterations = int(amp_obs.shape[0] / self.config.batch_size)
        iterations = max(1, min(iterations, 10))

        for _ in range(iterations):              
            fake_batch = self.replay_buffer.sample(self.config.batch_size)
            real_batch = self.expert_buffer.sample(self.config.batch_size)

            self.optimizer.zero_grad()

            real_logits = self.discriminator(real_batch)
            fake_logits = self.discriminator(fake_batch)
            
            acc_real = (real_logits > 0.0).float().mean().item()
            acc_fake = (fake_logits < 0.0).float().mean().item()
            
            # if acc_fake >= self.config.target_accuracy:
            #     print(f"[AMP] Skipping discriminator update due to high fake accuracy ({acc_fake:.2f} >= {self.config.target_accuracy:.2f}).")
            #     self.skip_counter += 1
                
            #     self.metrics["amp/loss"].append(0.0)
            #     self.metrics["amp/logits_real"].append(real_logits.mean().item())
            #     self.metrics["amp/logits_fake"].append(fake_logits.mean().item())
            #     self.metrics["amp/grad_penalty"].append(0.0)
            #     self.metrics["amp/accuracy_real"].append(acc_real)
            #     self.metrics["amp/accuracy_fake"].append(acc_fake)
            #     break

            loss_real = torch.mean(torch.square(real_logits - 1))
            loss_fake = torch.mean(torch.square(fake_logits + 1))
            loss = (2 * loss_real + loss_fake) / 3.0

            real_batch.requires_grad_(True)
            d_real = self.discriminator(real_batch)
            grads = torch.autograd.grad(
                outputs=d_real,
                inputs=real_batch,
                grad_outputs=torch.ones_like(d_real),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # Flatten to (B, -1) to calculate norm
            grads = grads.view(real_batch.size(0), -1)
            gp = (grads.norm(2, dim=1) ** 2).mean()

            total_loss = loss + self.config.gradient_penalty_weight * gp
            total_loss.backward()
            self.optimizer.step()

            self.metrics["amp/loss"].append(total_loss.item())
            self.metrics["amp/logits_real"].append(real_logits.mean().item())
            self.metrics["amp/logits_fake"].append(fake_logits.mean().item())
            self.metrics["amp/grad_penalty"].append(gp.item())
            self.metrics["amp/accuracy_real"].append(acc_real)
            self.metrics["amp/accuracy_fake"].append(acc_fake)

        results: dict[str, float] = {k: float(np.mean(v))if len(v) > 0 else 0.0 for k, v in self.metrics.items()}
        acc_real = results["amp/accuracy_real"]
        acc_fake = results["amp/accuracy_fake"]
        results["amp/accuracy"] = (acc_real + acc_fake) / 2.0

        return results

