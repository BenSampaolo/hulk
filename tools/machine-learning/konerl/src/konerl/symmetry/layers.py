from __future__ import annotations

from copy import deepcopy
from typing import Literal

import torch
from torch import nn

from .reflection import ReflectionSpec


class ReflectionEquivariantLinear(nn.Module):
    """Linear layer constrained to commute with two reflection specs.

    For input reflection ``M_in`` and output reflection ``M_out``, the effective
    weight is projected so ``W M_in = M_out W``. That makes the layer exactly
    equivariant by construction rather than by a penalty term.
    """

    def __init__(self, reflection_in: ReflectionSpec, reflection_out: ReflectionSpec, bias: bool = True) -> None:
        super().__init__()
        self.reflection_in = reflection_in
        self.reflection_out = reflection_out
        self.weight_raw = nn.Parameter(torch.randn(reflection_out.dim, reflection_in.dim) * 0.02)
        self.bias_raw = nn.Parameter(torch.zeros(reflection_out.dim)) if bias else None

    @torch.no_grad()
    def project_parameters_(self) -> None:
        device = self.weight_raw.device
        self.reflection_in.to(device)
        self.reflection_out.to(device)

        weight = self.weight_raw
        reflected_weight = self.reflection_out.apply(self.reflection_in.apply(weight, dim=1), dim=0)
        self.weight_raw.copy_(0.5 * (weight + reflected_weight))

        if self.bias_raw is not None:
            bias = self.bias_raw
            reflected_bias = self.reflection_out.apply(bias, dim=0)
            self.bias_raw.copy_(0.5 * (bias + reflected_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.project_parameters_()
        y = torch.matmul(x, self.weight_raw.T)
        if self.bias_raw is not None:
            y = y + self.bias_raw
        return y


class EquiSwish(nn.Module):
    """Odd Swish-like activation: ``f(-x) == -f(x)``.

    Ordinary ELU/ReLU would break odd reflection channels. This activation keeps
    even/odd hidden representations equivariant.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x.abs())


class ReflectionEquivariantMlp(nn.Sequential):
    def __init__(
        self,
        reflection_specs: list[ReflectionSpec],
        activation: nn.Module | None = None,
        last_activation: bool = False,
    ) -> None:
        if len(reflection_specs) < 2:
            raise ValueError("Need at least input and output reflection specs.")
        activation = activation if activation is not None else EquiSwish()

        layers: list[nn.Module] = []
        for i in range(len(reflection_specs) - 1):
            layers.append(ReflectionEquivariantLinear(reflection_specs[i], reflection_specs[i + 1], bias=True))
            if i < len(reflection_specs) - 2 or last_activation:
                layers.append(deepcopy(activation))
        super().__init__(*layers)


class ReflectionInvariantify(nn.Module):
    """Convert an equivariant mixed representation to an invariant one.

    Even channels are copied. Odd channels are made even with ``square`` or
    ``abs`` so a following ordinary MLP can produce invariant scalar values.
    """

    def __init__(self, spec: ReflectionSpec, mode: Literal["square", "abs"] = "square") -> None:
        super().__init__()
        self.spec = spec
        self.operation = {"square": torch.square, "abs": torch.abs}[mode]
        self.register_buffer("even_idx", spec.even_indices())
        self.register_buffer("odd_idx", spec.odd_indices())
        self.out_spec = ReflectionSpec.hidden(even_dim=spec.dim, odd_dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        even = x[..., self.even_idx]
        odd = self.operation(x[..., self.odd_idx])
        return torch.cat([even, odd], dim=-1)
