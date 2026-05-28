from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories

from .layers import ReflectionEquivariantMlp, ReflectionInvariantify
from .normalization import EquivariantEmpiricalNormalization
from .reflection import ReflectionSpec


def create_hidden_reflection_spec(dim: int) -> ReflectionSpec:
    """Default hidden representation: first half even, second half odd."""
    even_dim = dim // 2
    return ReflectionSpec.hidden(even_dim=even_dim, odd_dim=dim - even_dim)


def _coerce_spec(
    *,
    name: str,
    dim: int,
    spec: ReflectionSpec | None = None,
    perm: Sequence[int] | None = None,
    sign: Sequence[int] | None = None,
) -> ReflectionSpec:
    if spec is not None:
        if spec.dim != dim:
            raise ValueError(f"{name} ReflectionSpec has dim {spec.dim}, expected {dim}.")
        return spec
    if perm is None and sign is None:
        return ReflectionSpec.identity(dim)
    if perm is None or sign is None:
        raise ValueError(f"{name} reflection requires both perm and sign.")
    result = ReflectionSpec(list(perm), list(sign))
    if result.dim != dim:
        raise ValueError(f"{name} reflection has dim {result.dim}, expected {dim}.")
    return result


class EquivariantMLPModel(nn.Module):
    """RSL-RL compatible MLP model using exact reflection-equivariant layers.

    This intentionally mirrors the public interface of ``rsl_rl.models.MLPModel``
    so it can be selected via an RSL config ``class_name``. It is not wired into
    the existing K1 task by default; existing training behavior is unchanged.

    Config arguments of interest:

    - ``input_reflection_perm`` / ``input_reflection_sign`` for the selected obs;
    - ``output_reflection_perm`` / ``output_reflection_sign`` for actor actions;
    - ``invariant_output=True`` for critic/value models.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "equiswish",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        input_reflection_perm: Sequence[int] | None = None,
        input_reflection_sign: Sequence[int] | None = None,
        output_reflection_perm: Sequence[int] | None = None,
        output_reflection_sign: Sequence[int] | None = None,
        input_reflection_spec: ReflectionSpec | None = None,
        output_reflection_spec: ReflectionSpec | None = None,
        invariant_output: bool = False,
        invariant_mode: str = "square",
        cnn_cfg: dict | None = None,
        rnn_type: str | None = None,
        rnn_hidden_dim: int | None = None,
        rnn_num_layers: int | None = None,
    ) -> None:
        del rnn_hidden_dim, rnn_num_layers
        super().__init__()
        if cnn_cfg is not None or rnn_type is not None:
            raise ValueError("EquivariantMLPModel currently supports MLP-only, non-recurrent models.")
        if activation.lower() not in {"equiswish", "equivariant_swish"}:
            raise ValueError("EquivariantMLPModel requires activation='equiswish' to preserve odd channels.")

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.input_reflection = _coerce_spec(
            name="input",
            dim=self.obs_dim,
            spec=input_reflection_spec,
            perm=input_reflection_perm,
            sign=input_reflection_sign,
        )

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EquivariantEmpiricalNormalization(self.input_reflection)
        else:
            self.obs_normalizer = nn.Identity()

        if not isinstance(output_dim, int):
            raise TypeError("EquivariantMLPModel currently supports integer output_dim only.")

        if distribution_cfg is not None:
            dist_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))  # type: ignore[assignment]
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        if not isinstance(mlp_output_dim, int):
            raise TypeError("EquivariantMLPModel only supports distributions with integer input_dim.")

        self.invariant_output = invariant_output
        hidden_specs = [create_hidden_reflection_spec(dim) for dim in hidden_dims]

        if invariant_output:
            # Critic/value path: equivariant feature extractor, invariantify, scalar head.
            specs = [self.input_reflection] + hidden_specs
            if len(specs) == 1:
                self.equivariant = nn.Identity()
                final_spec = self.input_reflection
            else:
                self.equivariant = ReflectionEquivariantMlp(specs, last_activation=True)
                final_spec = specs[-1]
            self.invariantify = ReflectionInvariantify(final_spec, mode=invariant_mode)  # type: ignore[arg-type]
            self.head = nn.Linear(final_spec.dim, mlp_output_dim)
            self.output_reflection = ReflectionSpec.hidden(even_dim=mlp_output_dim, odd_dim=0)
        else:
            self.output_reflection = _coerce_spec(
                name="output",
                dim=mlp_output_dim,
                spec=output_reflection_spec,
                perm=output_reflection_perm,
                sign=output_reflection_sign,
            )
            self.equivariant = ReflectionEquivariantMlp(
                [self.input_reflection] + hidden_specs + [self.output_reflection],
                last_activation=False,
            )
            self.invariantify = nn.Identity()
            self.head = nn.Identity()

        if self.distribution is not None:
            self.distribution.init_mlp_weights(nn.Sequential(self.equivariant, self.invariantify, self.head))

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self._forward_network(latent)
        if self.distribution is not None:
            self._project_distribution_std_()
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def _forward_network(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.equivariant(latent)
        x = self.invariantify(x)
        return self.head(x)

    @torch.no_grad()
    def _project_distribution_std_(self) -> None:
        """Tie state-independent Gaussian std across mirrored action channels.

        Equivariance of the action mean is not enough for stochastic PPO: left
        and right paired actions also need identical standard deviations.
        """
        if self.distribution is None:
            return
        perm = self.output_reflection._perm_tensor.to(next(self.distribution.parameters(), torch.empty(0)).device)
        for attr in ("std_param", "log_std_param"):
            param = getattr(self.distribution, attr, None)
            if isinstance(param, torch.nn.Parameter):
                param.copy_(0.5 * (param + torch.index_select(param, dim=0, index=perm.to(param.device))))

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del masks, hidden_state
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        return self.obs_normalizer(latent)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Model has no output distribution.")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Model has no output distribution.")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Model has no output distribution.")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        if self.distribution is None:
            raise RuntimeError("Model has no output distribution.")
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Model has no output distribution.")
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Model has no output distribution.")
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        return _ExportEquivariantMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        del verbose
        return _ExportEquivariantMLPModel(self)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(mlp_obs)  # type: ignore[attr-defined]

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"EquivariantMLPModel only supports 1D observations, got {obs[obs_group].shape} for {obs_group!r}."
                )
            obs_dim += obs[obs_group].shape[-1]
        return list(active_obs_groups), obs_dim


class K1VelocityEquivariantMLPModel(EquivariantMLPModel):
    """Opt-in K1 velocity model with built-in reflection specs.

    This avoids extending mjlab's dataclass config schema with custom perm/sign
    fields: selecting this ``class_name`` is enough for the current K1 velocity
    actor/critic observation layout.
    """

    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int, **kwargs) -> None:
        from .k1_specs import (
            K1_ACTION_SPEC,
            K1_FULL_BODY_ACTION_SPEC,
            K1_FULL_BODY_VELOCITY_ACTOR_SPEC,
            K1_FULL_BODY_VELOCITY_CRITIC_SPEC,
            K1_VELOCITY_ACTOR_SPEC,
            K1_VELOCITY_CRITIC_SPEC,
        )

        obs_dim = sum(obs[group].shape[-1] for group in obs_groups[obs_set])
        if obs_set == "actor":
            if output_dim == K1_FULL_BODY_ACTION_SPEC.dim:
                input_spec = K1_FULL_BODY_VELOCITY_ACTOR_SPEC
                output_spec = K1_FULL_BODY_ACTION_SPEC
            elif output_dim == K1_ACTION_SPEC.dim:
                input_spec = K1_VELOCITY_ACTOR_SPEC
                output_spec = K1_ACTION_SPEC
            else:
                raise ValueError(f"Unsupported K1 actor output dimension: {output_dim}.")
            kwargs.setdefault("input_reflection_perm", input_spec.perm)
            kwargs.setdefault("input_reflection_sign", input_spec.sign)
            kwargs.setdefault("output_reflection_perm", output_spec.perm)
            kwargs.setdefault("output_reflection_sign", output_spec.sign)
        elif obs_set == "critic":
            if obs_dim == K1_FULL_BODY_VELOCITY_CRITIC_SPEC.dim:
                input_spec = K1_FULL_BODY_VELOCITY_CRITIC_SPEC
            elif obs_dim == K1_VELOCITY_CRITIC_SPEC.dim:
                input_spec = K1_VELOCITY_CRITIC_SPEC
            else:
                raise ValueError(f"Unsupported K1 critic observation dimension: {obs_dim}.")
            kwargs.setdefault("input_reflection_perm", input_spec.perm)
            kwargs.setdefault("input_reflection_sign", input_spec.sign)
            kwargs.setdefault("invariant_output", True)
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)


class _ExportEquivariantMLPModel(nn.Module):
    is_recurrent: bool = False

    def __init__(self, model: EquivariantMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.equivariant = copy.deepcopy(model.equivariant)
        self.invariantify = copy.deepcopy(model.invariantify)
        self.head = copy.deepcopy(model.head)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        x = self.equivariant(x)
        x = self.invariantify(x)
        x = self.head(x)
        return self.deterministic_output(x)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

    @torch.jit.export
    def reset(self) -> None:
        pass
