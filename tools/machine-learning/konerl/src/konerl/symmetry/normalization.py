from __future__ import annotations

import torch
from rsl_rl.modules import EmpiricalNormalization

from .reflection import ReflectionSpec


class EquivariantEmpiricalNormalization(EmpiricalNormalization):
    """RSL-RL empirical normalizer that preserves a reflection symmetry.

    After each normal RSL statistics update, the mean and variance are projected
    back onto the symmetric subspace:

    - mean obeys ``mean = reflect(mean)`` including signs;
    - variance obeys ``var = var[perm]`` because signs vanish when squared.
    """

    def __init__(self, spec: ReflectionSpec, eps: float = 1e-2, until: int | None = None) -> None:
        super().__init__(spec.dim, eps=eps, until=until)
        self.spec = spec
        self.register_buffer("perm", spec._perm_tensor.to(dtype=torch.long))
        self.register_buffer("sign", spec._sign_tensor.to(dtype=torch.float32))

    @torch.no_grad()
    def _symmetrize_stats(self) -> None:
        sign = self.sign.to(device=self._mean.device, dtype=self._mean.dtype).view(1, -1)
        perm = self.perm.to(device=self._mean.device)

        reflected_mean = torch.index_select(self._mean * sign, dim=1, index=perm)
        self._mean.copy_(0.5 * (self._mean + reflected_mean))

        reflected_var = torch.index_select(self._var, dim=1, index=perm)
        self._var.copy_(0.5 * (self._var + reflected_var))
        self._std.copy_(torch.sqrt(self._var))

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        super().update(x)
        if self.training:
            self._symmetrize_stats()
