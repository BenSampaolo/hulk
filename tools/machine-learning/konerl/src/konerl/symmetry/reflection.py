from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReflectionSpec:
    """Signed permutation representation of a mirror/reflection symmetry.

    The reflection maps a vector ``x`` to ``M x`` by first applying ``sign``
    channel-wise and then permuting with ``perm``. The map must be an
    involution: applying it twice returns the original vector.
    """

    perm: list[int]
    sign: list[int]

    def __post_init__(self) -> None:
        dim = len(self.perm)
        if len(self.sign) != dim:
            raise ValueError("Sign vector must match permutation length.")
        if set(self.perm) != set(range(dim)):
            raise ValueError("Permutation must contain every channel exactly once.")
        if any(s not in (-1, 1) for s in self.sign):
            raise ValueError("Signs must be ±1.")

        self.dim = dim
        self._sign_tensor = torch.tensor(self.sign)
        self._perm_tensor = torch.tensor(self.perm, dtype=torch.long)

        if not all(self.perm[self.perm[i]] == i for i in range(dim)):
            raise ValueError("ReflectionSpec must be order 2: permutation is not an involution.")

        for i in range(dim):
            j = self.perm[i]
            if self.sign[i] * self.sign[j] != 1:
                raise ValueError(
                    f"ReflectionSpec must be order 2: inconsistent signs for swap ({i}, {j}). "
                    "Swapped channels must have the same sign."
                )

    def to(self, device: str | torch.device) -> ReflectionSpec:
        self._sign_tensor = self._sign_tensor.to(device)
        self._perm_tensor = self._perm_tensor.to(device)
        return self

    def even_indices(self) -> torch.Tensor:
        return torch.where(self._sign_tensor > 0)[0]

    def odd_indices(self) -> torch.Tensor:
        return torch.where(self._sign_tensor < 0)[0]

    def matrix(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        matrix = torch.zeros(self.dim, self.dim, device=device, dtype=dtype)
        for row, src_col in enumerate(self.perm):
            matrix[row, src_col] = self.sign[src_col]
        return matrix

    def apply(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Apply this reflection along ``dim`` of ``x``."""
        if x.shape[dim] != self.dim:
            raise ValueError(f"Expected dimension {self.dim} at axis {dim}, got {x.shape[dim]}.")

        self.to(x.device)
        shape = [1] * x.ndim
        shape[dim] = self.dim
        signed = x * self._sign_tensor.to(dtype=x.dtype).view(shape)
        return torch.index_select(signed, dim, self._perm_tensor)

    def combine(self, other: ReflectionSpec) -> ReflectionSpec:
        """Return the reflection for concatenated vectors ``[self, other]``."""
        return ReflectionSpec(
            perm=self.perm + [self.dim + p for p in other.perm],
            sign=[*self.sign, *other.sign],
        )

    @staticmethod
    def combine_many(specs: list[ReflectionSpec]) -> ReflectionSpec:
        if not specs:
            raise ValueError("Need at least one ReflectionSpec to combine.")
        result = specs[0]
        for spec in specs[1:]:
            result = result.combine(spec)
        return result

    @classmethod
    def identity(cls, dim: int) -> ReflectionSpec:
        return cls(perm=list(range(dim)), sign=[1] * dim)

    @classmethod
    def hidden(cls, even_dim: int, odd_dim: int) -> ReflectionSpec:
        """Hidden representation: even channels stay fixed, odd channels flip sign."""
        return cls(perm=list(range(even_dim + odd_dim)), sign=[1] * even_dim + [-1] * odd_dim)

    @classmethod
    def from_joint_names(cls, names: list[str]) -> ReflectionSpec:
        """Build a K1-style joint/action reflection from Left/Right joint names.

        Left/Right partners are swapped. Roll and yaw coordinates additionally
        flip sign under a sagittal-plane mirror. Unpaired joints are treated as
        self-paired and only use the sign rule.
        """
        perm = list(range(len(names)))
        sign = [1] * len(names)
        name_to_idx = {name: i for i, name in enumerate(names)}

        for i, name in enumerate(names):
            if "Left" in name:
                partner = name.replace("Left", "Right")
                if partner in name_to_idx:
                    perm[i] = name_to_idx[partner]
            elif "Right" in name:
                partner = name.replace("Right", "Left")
                if partner in name_to_idx:
                    perm[i] = name_to_idx[partner]

            lower = name.lower()
            if "roll" in lower or "yaw" in lower:
                sign[i] = -1

        return cls(perm=perm, sign=sign)
