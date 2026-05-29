from __future__ import annotations

import contextlib
import io
import math
import unittest
from types import SimpleNamespace

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from konerl.symmetry.k1_specs import (
    AXIAL_VECTOR3,
    COMMAND_SPEC,
    K1_ACTION_SPEC,
    K1_FULL_BODY_ACTION_SPEC,
    K1_FULL_BODY_GAIN_NAMES,
    K1_FULL_BODY_GAIN_SPEC,
    K1_FULL_BODY_JOINT_NAMES,
    K1_FULL_BODY_JOINT_SPEC,
    K1_FULL_BODY_VELOCITY_ACTOR_SPEC,
    K1_FULL_BODY_VELOCITY_CRITIC_SPEC,
    K1_LEG_GAIN_NAMES,
    K1_LEG_GAIN_SPEC,
    K1_LEG_JOINT_NAMES,
    K1_LEG_JOINT_SPEC,
    K1_VELOCITY_ACTOR_SPEC,
    K1_VELOCITY_CRITIC_SPEC,
    LEFT_RIGHT_SCALARS,
    LEFT_RIGHT_VECTOR3_BLOCKS,
    SCALAR,
    TRUE_VECTOR3,
)
from konerl.symmetry.reflection import ReflectionSpec
from konerl.tasks.k1_velocity_tracking.env_cfg import k1_rough_env_cfg
from konerl.tasks.k1_velocity_tracking.observations import obs_push_force, quat_from_yaw

ZERO_DIM = ReflectionSpec.identity(0)


def _actor_term_specs(joint_spec: ReflectionSpec, action_spec: ReflectionSpec) -> list[tuple[str, ReflectionSpec]]:
    return [
        ("base_ang_vel", AXIAL_VECTOR3),
        ("projected_gravity", TRUE_VECTOR3),
        ("joint_pos", joint_spec),
        ("joint_vel", joint_spec),
        ("actions", action_spec),
        ("command", COMMAND_SPEC),
    ]


def _critic_term_specs(
    joint_spec: ReflectionSpec,
    action_spec: ReflectionSpec,
    gain_spec: ReflectionSpec,
) -> list[tuple[str, ReflectionSpec]]:
    return [
        ("base_ang_vel", AXIAL_VECTOR3),
        ("projected_gravity", TRUE_VECTOR3),
        ("joint_pos", joint_spec),
        ("joint_vel", joint_spec),
        ("actions", action_spec),
        ("prev_prev_actions", action_spec),
        ("command", COMMAND_SPEC),
        ("base_lin_vel", TRUE_VECTOR3),
        ("foot_height", LEFT_RIGHT_SCALARS),
        ("foot_air_time", LEFT_RIGHT_SCALARS),
        ("foot_contact", LEFT_RIGHT_SCALARS),
        ("foot_contact_forces", LEFT_RIGHT_VECTOR3_BLOCKS),
        ("trunk_mass", SCALAR),
        ("foot_friction", LEFT_RIGHT_SCALARS),
        ("base_com", TRUE_VECTOR3),
        ("default_KpKd_gains", gain_spec),
        ("special_KpKd_gains", gain_spec),
        ("actuator_lag", ZERO_DIM),
        ("encoder_bias", joint_spec),
        ("push_force", TRUE_VECTOR3),
    ]


def _combined(specs: list[tuple[str, ReflectionSpec]]) -> ReflectionSpec:
    return ReflectionSpec.combine_many([spec for _, spec in specs])


class TestK1SymmetrySpecConfig(unittest.TestCase):
    def test_named_observation_specs_reconstruct_flat_specs(self) -> None:
        cases = (
            (
                K1_LEG_JOINT_SPEC,
                K1_ACTION_SPEC,
                K1_LEG_GAIN_SPEC,
                K1_VELOCITY_ACTOR_SPEC,
                K1_VELOCITY_CRITIC_SPEC,
            ),
            (
                K1_FULL_BODY_JOINT_SPEC,
                K1_FULL_BODY_ACTION_SPEC,
                K1_FULL_BODY_GAIN_SPEC,
                K1_FULL_BODY_VELOCITY_ACTOR_SPEC,
                K1_FULL_BODY_VELOCITY_CRITIC_SPEC,
            ),
        )
        for joint_spec, action_spec, gain_spec, actor_spec, critic_spec in cases:
            with self.subTest(actor_dim=actor_spec.dim, critic_dim=critic_spec.dim):
                actor_terms = _actor_term_specs(joint_spec, action_spec)
                critic_terms = _critic_term_specs(joint_spec, action_spec, gain_spec)
                self.assertEqual(_combined(actor_terms).perm, actor_spec.perm)
                self.assertEqual(_combined(actor_terms).sign, actor_spec.sign)
                self.assertEqual(_combined(critic_terms).perm, critic_spec.perm)
                self.assertEqual(_combined(critic_terms).sign, critic_spec.sign)

    def test_live_env_observation_terms_match_named_specs(self) -> None:
        cases = (
            (
                False,
                K1_LEG_JOINT_NAMES,
                K1_LEG_GAIN_NAMES,
                K1_ACTION_SPEC,
                K1_LEG_JOINT_SPEC,
                K1_LEG_GAIN_SPEC,
                K1_VELOCITY_ACTOR_SPEC,
                K1_VELOCITY_CRITIC_SPEC,
            ),
            (
                True,
                K1_FULL_BODY_JOINT_NAMES,
                K1_FULL_BODY_GAIN_NAMES,
                K1_FULL_BODY_ACTION_SPEC,
                K1_FULL_BODY_JOINT_SPEC,
                K1_FULL_BODY_GAIN_SPEC,
                K1_FULL_BODY_VELOCITY_ACTOR_SPEC,
                K1_FULL_BODY_VELOCITY_CRITIC_SPEC,
            ),
        )
        for control_arms, joint_names, gain_names, action_spec, joint_spec, gain_spec, actor_spec, critic_spec in cases:
            with self.subTest(control_arms=control_arms):
                cfg = k1_rough_env_cfg(amp=False, control_arms=control_arms)
                with contextlib.redirect_stdout(io.StringIO()):
                    env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
                try:
                    action_term = env.action_manager.get_term("joint_pos")
                    self.assertEqual(tuple(action_term.target_names), joint_names)
                    self.assertEqual(action_spec.perm, ReflectionSpec.from_joint_names(list(joint_names)).perm)
                    self.assertEqual(action_spec.sign, ReflectionSpec.from_joint_names(list(joint_names)).sign)

                    actual_gain_names = tuple(
                        joint_name
                        for actuator in env.scene["robot"].actuators
                        for joint_name in actuator.target_names
                    )
                    self.assertEqual(actual_gain_names, gain_names)

                    expected = {
                        "actor": _actor_term_specs(joint_spec, action_spec),
                        "critic": _critic_term_specs(joint_spec, action_spec, gain_spec),
                    }
                    expected_flat = {"actor": actor_spec, "critic": critic_spec}
                    for group_name, term_specs in expected.items():
                        self.assertEqual(env.observation_manager.active_terms[group_name], [n for n, _ in term_specs])
                        self.assertEqual(sum(spec.dim for _, spec in term_specs), expected_flat[group_name].dim)
                        for dims, (term_name, term_spec) in zip(
                            env.observation_manager.group_obs_term_dim[group_name], term_specs, strict=True
                        ):
                            dim = math.prod(dims)
                            self.assertEqual(dim, term_spec.dim, f"{group_name}.{term_name}")
                    self.assertEqual(env.observation_manager.group_obs_dim["actor"], (actor_spec.dim,))
                    self.assertEqual(env.observation_manager.group_obs_dim["critic"], (critic_spec.dim,))
                finally:
                    env.close()

    def test_push_force_observation_is_root_body_frame(self) -> None:
        yaw = torch.tensor([math.pi / 2.0])
        root_quat_w = quat_from_yaw(yaw)
        world_force = torch.tensor([[[1.0, 0.0, 0.0]]])
        wrench = torch.cat([world_force, torch.zeros_like(world_force)], dim=-1)
        asset = SimpleNamespace(
            data=SimpleNamespace(
                body_external_wrench=wrench,
                root_link_quat_w=root_quat_w,
            )
        )
        env = SimpleNamespace(scene={"robot": asset}, num_envs=1, device="cpu")

        force_b = obs_push_force(env, SceneEntityCfg("robot", body_ids=[0]))

        torch.testing.assert_close(force_b, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
