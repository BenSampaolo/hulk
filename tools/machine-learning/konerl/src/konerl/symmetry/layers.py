from __future__ import annotations

from copy import deepcopy
from typing import Callable, Literal, cast

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

    def _project_weight(self, weight: torch.Tensor) -> torch.Tensor:
        reflected_weight = self.reflection_out.apply(self.reflection_in.apply(weight, dim=1), dim=0)
        return 0.5 * (weight + reflected_weight)

    def _project_bias(self, bias: torch.Tensor) -> torch.Tensor:
        reflected_bias = self.reflection_out.apply(bias, dim=0)
        return 0.5 * (bias + reflected_bias)

    @torch.no_grad()
    def project_parameters_(self) -> None:
        self.weight_raw.copy_(self._project_weight(self.weight_raw))
        if self.bias_raw is not None:
            self.bias_raw.copy_(self._project_bias(self.bias_raw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._project_weight(self.weight_raw)
        y = torch.matmul(x, weight.T)
        if self.bias_raw is not None:
            y = y + self._project_bias(self.bias_raw)
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
    """Convert an equivariant mixed representation to invariant features.

    Fixed even channels pass through. Fixed odd channels are made even with
    ``square`` or ``abs``. Swapped channel pairs are converted to an invariant
    even combination plus an invariantized odd combination.
    """

    def __init__(self, spec: ReflectionSpec, mode: Literal["square", "abs"] = "square") -> None:
        super().__init__()
        self.spec = spec
        self.operation: Callable[[torch.Tensor], torch.Tensor] = {"square": torch.square, "abs": torch.abs}[mode]

        fixed_even: list[int] = []
        fixed_odd: list[int] = []
        pair_left: list[int] = []
        pair_right: list[int] = []
        pair_sign: list[int] = []
        seen: set[int] = set()
        for i, j in enumerate(spec.perm):
            if i in seen:
                continue
            seen.add(i)
            if i == j:
                if spec.sign[i] > 0:
                    fixed_even.append(i)
                else:
                    fixed_odd.append(i)
                continue
            seen.add(j)
            pair_left.append(i)
            pair_right.append(j)
            pair_sign.append(spec.sign[i])

        self.register_buffer("fixed_even_idx", torch.tensor(fixed_even, dtype=torch.long))
        self.register_buffer("fixed_odd_idx", torch.tensor(fixed_odd, dtype=torch.long))
        self.register_buffer("pair_left_idx", torch.tensor(pair_left, dtype=torch.long))
        self.register_buffer("pair_right_idx", torch.tensor(pair_right, dtype=torch.long))
        self.register_buffer("pair_sign", torch.tensor(pair_sign, dtype=torch.float32))
        self.out_spec = ReflectionSpec.hidden(even_dim=spec.dim, odd_dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fixed_even_idx = cast(torch.Tensor, self.fixed_even_idx)
        fixed_odd_idx = cast(torch.Tensor, self.fixed_odd_idx)
        pair_left_idx = cast(torch.Tensor, self.pair_left_idx)
        pair_right_idx = cast(torch.Tensor, self.pair_right_idx)
        pair_sign = cast(torch.Tensor, self.pair_sign)

        pieces: list[torch.Tensor] = []
        if fixed_even_idx.numel() > 0:
            pieces.append(x[..., fixed_even_idx])
        if pair_left_idx.numel() > 0:
            left = x[..., pair_left_idx]
            right = x[..., pair_right_idx]
            sign = pair_sign.to(device=x.device, dtype=x.dtype)
            pieces.append(0.5 * (left + sign * right))
        if fixed_odd_idx.numel() > 0:
            pieces.append(self.operation(x[..., fixed_odd_idx]))
        if pair_left_idx.numel() > 0:
            left = x[..., pair_left_idx]
            right = x[..., pair_right_idx]
            sign = pair_sign.to(device=x.device, dtype=x.dtype)
            pieces.append(self.operation(0.5 * (left - sign * right)))
        if not pieces:
            return x.new_empty(*x.shape[:-1], 0)
        return torch.cat(pieces, dim=-1)
