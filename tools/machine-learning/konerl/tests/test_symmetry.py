from __future__ import annotations

import unittest
from typing import Any, cast

import torch
from tensordict import TensorDict

from konerl.symmetry.k1_specs import (
    K1_ACTION_SPEC,
    K1_FULL_BODY_ACTION_SPEC,
    K1_FULL_BODY_VELOCITY_ACTOR_SPEC,
    K1_FULL_BODY_VELOCITY_CRITIC_SPEC,
    K1_VELOCITY_ACTOR_SPEC,
    K1_VELOCITY_CRITIC_SPEC,
)
from konerl.symmetry.layers import ReflectionEquivariantLinear, ReflectionInvariantify
from konerl.symmetry.reflection import ReflectionSpec
from konerl.symmetry.rsl_model import K1VelocityEquivariantMLPModel


class TestReflectionSpec(unittest.TestCase):
    def test_apply_matches_dense_matrix(self) -> None:
        spec = ReflectionSpec(perm=[1, 0, 2], sign=[1, 1, -1])
        x = torch.randn(5, spec.dim)

        torch.testing.assert_close(spec.apply(x), x @ spec.matrix(dtype=x.dtype).T)

    def test_equivariant_linear_commutes_with_reflection(self) -> None:
        input_spec = ReflectionSpec(perm=[1, 0, 2], sign=[1, 1, -1])
        output_spec = ReflectionSpec.hidden(even_dim=2, odd_dim=2)
        layer = ReflectionEquivariantLinear(input_spec, output_spec)
        x = torch.randn(16, input_spec.dim)

        torch.testing.assert_close(
            layer(input_spec.apply(x)),
            output_spec.apply(layer(x)),
            atol=1e-6,
            rtol=1e-6,
        )


class TestK1SymmetrySpecs(unittest.TestCase):
    def test_k1_spec_dimensions(self) -> None:
        self.assertEqual(K1_VELOCITY_ACTOR_SPEC.dim, 52)
        self.assertEqual(K1_VELOCITY_CRITIC_SPEC.dim, 148)
        self.assertEqual(K1_ACTION_SPEC.dim, 12)

        self.assertEqual(K1_FULL_BODY_VELOCITY_ACTOR_SPEC.dim, 76)
        self.assertEqual(K1_FULL_BODY_VELOCITY_CRITIC_SPEC.dim, 220)
        self.assertEqual(K1_FULL_BODY_ACTION_SPEC.dim, 20)

    def test_k1_specs_are_involutions(self) -> None:
        for spec in (
            K1_VELOCITY_ACTOR_SPEC,
            K1_VELOCITY_CRITIC_SPEC,
            K1_ACTION_SPEC,
            K1_FULL_BODY_VELOCITY_ACTOR_SPEC,
            K1_FULL_BODY_VELOCITY_CRITIC_SPEC,
            K1_FULL_BODY_ACTION_SPEC,
        ):
            x = torch.randn(3, spec.dim)
            torch.testing.assert_close(spec.apply(spec.apply(x)), x)

    def test_gain_blocks_swap_left_right(self) -> None:
        # Critic layout before gains is:
        # base_ang_vel, projected_gravity, q, qd, action, prev_action,
        # command, base_lin_vel, foot_height, foot_air_time, foot_contact,
        # foot_contact_forces, trunk_mass, foot_friction, base_com.
        default_gain_offset = 85
        default_gain_dim = 24

        perm = K1_VELOCITY_CRITIC_SPEC.perm[default_gain_offset : default_gain_offset + default_gain_dim]

        # obs_pd_gains returns kp for all actuators, then kd for all actuators.
        # The current leg actuator order is interleaved left/right, so the first
        # Kp pair should mirror as channel 0 <-> 1 inside the gain block.
        self.assertEqual(perm[0], default_gain_offset + 1)
        self.assertEqual(perm[1], default_gain_offset + 0)
        self.assertEqual(perm[12], default_gain_offset + 13)
        self.assertEqual(perm[13], default_gain_offset + 12)

        full_body_default_gain_offset = 117
        full_body_default_gain_dim = 40
        full_body_perm = K1_FULL_BODY_VELOCITY_CRITIC_SPEC.perm[
            full_body_default_gain_offset : full_body_default_gain_offset + full_body_default_gain_dim
        ]
        self.assertEqual(full_body_perm[0], full_body_default_gain_offset + 1)
        self.assertEqual(full_body_perm[1], full_body_default_gain_offset + 0)
        self.assertEqual(full_body_perm[20], full_body_default_gain_offset + 21)
        self.assertEqual(full_body_perm[21], full_body_default_gain_offset + 20)


class TestK1EquivariantModel(unittest.TestCase):
    def test_actor_is_equivariant_and_critic_is_invariant(self) -> None:
        cases = (
            (K1_VELOCITY_ACTOR_SPEC, K1_VELOCITY_CRITIC_SPEC, K1_ACTION_SPEC),
            (K1_FULL_BODY_VELOCITY_ACTOR_SPEC, K1_FULL_BODY_VELOCITY_CRITIC_SPEC, K1_FULL_BODY_ACTION_SPEC),
        )
        batch = 8
        groups = {"actor": ["actor"], "critic": ["critic"]}

        for actor_spec, critic_spec, action_spec in cases:
            with self.subTest(actor_dim=actor_spec.dim, critic_dim=critic_spec.dim, action_dim=action_spec.dim):
                obs = TensorDict(
                    {
                        "actor": torch.randn(batch, actor_spec.dim),
                        "critic": torch.randn(batch, critic_spec.dim),
                    },
                    batch_size=[batch],
                )
                mirrored = TensorDict(
                    {
                        "actor": actor_spec.apply(obs["actor"]),
                        "critic": critic_spec.apply(obs["critic"]),
                    },
                    batch_size=[batch],
                )

                actor = K1VelocityEquivariantMLPModel(
                    obs,
                    groups,
                    "actor",
                    action_spec.dim,
                    hidden_dims=(32, 32),
                    activation="equiswish",
                )
                critic = K1VelocityEquivariantMLPModel(
                    obs,
                    groups,
                    "critic",
                    1,
                    hidden_dims=(32, 32),
                    activation="equiswish",
                )

                torch.testing.assert_close(
                    actor(mirrored),
                    action_spec.apply(actor(obs)),
                    atol=1e-6,
                    rtol=1e-6,
                )
                torch.testing.assert_close(
                    critic(mirrored),
                    critic(obs),
                    atol=1e-6,
                    rtol=1e-6,
                )

    def test_stochastic_std_is_tied_across_mirrored_actions(self) -> None:
        batch = 4
        obs = TensorDict({"actor": torch.randn(batch, K1_VELOCITY_ACTOR_SPEC.dim)}, batch_size=[batch])
        actor = K1VelocityEquivariantMLPModel(
            obs,
            {"actor": ["actor"]},
            "actor",
            K1_ACTION_SPEC.dim,
            hidden_dims=(32,),
            activation="equiswish",
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "log"},
        )

        assert actor.distribution is not None
        with torch.no_grad():
            log_std_param = cast(torch.Tensor, actor.distribution.log_std_param)
            log_std_param.copy_(torch.arange(K1_ACTION_SPEC.dim, dtype=torch.float32))

        cast(Any, actor)(obs, stochastic_output=True)
        std = actor.output_std[0]

        torch.testing.assert_close(std, std[K1_ACTION_SPEC._perm_tensor])


class TestSymmetryRegressionFixes(unittest.TestCase):
    def test_invariantify_handles_swapped_representations(self) -> None:
        spec = ReflectionSpec(perm=[1, 0], sign=[1, 1])
        invariantify = ReflectionInvariantify(spec)
        x = torch.tensor([[1.0, 2.0]])

        torch.testing.assert_close(invariantify(spec.apply(x)), invariantify(x))

    def test_linear_projection_projects_gradients(self) -> None:
        input_spec = ReflectionSpec(perm=[1, 0], sign=[1, 1])
        output_spec = ReflectionSpec.hidden(even_dim=1, odd_dim=1)
        layer = ReflectionEquivariantLinear(input_spec, output_spec)
        x = torch.randn(8, input_spec.dim)

        loss = layer(x).square().sum()
        loss.backward()
        grad = layer.weight_raw.grad
        assert grad is not None

        reflected_grad = output_spec.apply(input_spec.apply(grad, dim=1), dim=0)
        torch.testing.assert_close(grad, reflected_grad, atol=1e-6, rtol=1e-6)

    def test_reflection_spec_recovers_from_inference_tensor_cache(self) -> None:
        input_spec = ReflectionSpec(perm=[1, 0], sign=[1, 1])
        output_spec = ReflectionSpec.hidden(even_dim=1, odd_dim=1)
        layer = ReflectionEquivariantLinear(input_spec, output_spec)

        with torch.inference_mode():
            for spec in (input_spec, output_spec):
                spec._sign_tensor = spec._sign_tensor.clone()
                spec._perm_tensor = spec._perm_tensor.clone()
                self.assertTrue(spec._sign_tensor.is_inference())
                self.assertTrue(spec._perm_tensor.is_inference())

        x = torch.randn(8, input_spec.dim, requires_grad=True)
        loss = layer(x).square().sum()
        loss.backward()

        self.assertIsNotNone(layer.weight_raw.grad)
        self.assertFalse(input_spec._sign_tensor.is_inference())
        self.assertFalse(input_spec._perm_tensor.is_inference())
        self.assertFalse(output_spec._sign_tensor.is_inference())
        self.assertFalse(output_spec._perm_tensor.is_inference())


if __name__ == "__main__":
    unittest.main()
