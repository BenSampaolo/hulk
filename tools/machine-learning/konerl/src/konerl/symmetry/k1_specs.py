from __future__ import annotations

from .reflection import ReflectionSpec


# Mirrors the K1 robot/action order exposed by mjlab after compiling model/K1.xml.
K1_ARM_JOINT_NAMES: tuple[str, ...] = (
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
)

K1_LEG_JOINT_NAMES: tuple[str, ...] = (
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)

K1_FULL_BODY_JOINT_NAMES: tuple[str, ...] = K1_ARM_JOINT_NAMES + K1_LEG_JOINT_NAMES

# obs_pd_gains currently exposes kp for all actuators, then kd for all actuators,
# in compiled actuator order rather than policy action/joint order.
K1_ARM_GAIN_NAMES: tuple[str, ...] = (
    "ALeft_Shoulder_Pitch",
    "ARight_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Right_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Right_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Elbow_Yaw",
)

K1_LEG_GAIN_NAMES: tuple[str, ...] = (
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Ankle_Roll",
)

K1_FULL_BODY_GAIN_NAMES: tuple[str, ...] = K1_ARM_GAIN_NAMES + K1_LEG_GAIN_NAMES


TRUE_VECTOR3 = ReflectionSpec(perm=[0, 1, 2], sign=[1, -1, 1])
"""Polar vector under sagittal mirror, e.g. position, linear velocity, gravity, force."""

AXIAL_VECTOR3 = ReflectionSpec(perm=[0, 1, 2], sign=[-1, 1, -1])
"""Axial vector under sagittal mirror, e.g. angular velocity or torque."""

SCALAR = ReflectionSpec.identity(1)
LEFT_RIGHT_SCALARS = ReflectionSpec(perm=[1, 0], sign=[1, 1])
LEFT_RIGHT_VECTOR3_BLOCKS = ReflectionSpec(perm=[3, 4, 5, 0, 1, 2], sign=[1, -1, 1, 1, -1, 1])


K1_ARM_JOINT_SPEC = ReflectionSpec.from_joint_names(list(K1_ARM_JOINT_NAMES))
K1_LEG_JOINT_SPEC = ReflectionSpec.from_joint_names(list(K1_LEG_JOINT_NAMES))
K1_FULL_BODY_JOINT_SPEC = ReflectionSpec.from_joint_names(list(K1_FULL_BODY_JOINT_NAMES))


def _left_right_scalar_spec(names: tuple[str, ...]) -> ReflectionSpec:
    perm = list(range(len(names)))
    name_to_idx = {name: i for i, name in enumerate(names)}
    for i, name in enumerate(names):
        if "Left" in name:
            partner = name.replace("Left", "Right")
        elif "Right" in name:
            partner = name.replace("Right", "Left")
        else:
            continue
        if partner in name_to_idx:
            perm[i] = name_to_idx[partner]
    return ReflectionSpec(perm=perm, sign=[1] * len(names))


def _kp_kd_gain_spec(names: tuple[str, ...]) -> ReflectionSpec:
    scalar_spec = _left_right_scalar_spec(names)
    return scalar_spec.combine(scalar_spec)


K1_LEG_GAIN_SPEC = _kp_kd_gain_spec(K1_LEG_GAIN_NAMES)
K1_FULL_BODY_GAIN_SPEC = _kp_kd_gain_spec(K1_FULL_BODY_GAIN_NAMES)

# Backward-compatible leg-only aliases.
K1_JOINT_SPEC = K1_LEG_JOINT_SPEC
K1_ACTION_SPEC = K1_LEG_JOINT_SPEC
K1_FULL_BODY_ACTION_SPEC = K1_FULL_BODY_JOINT_SPEC


COMMAND_SPEC = ReflectionSpec.combine_many(
    [
        ReflectionSpec(perm=[0, 1, 2], sign=[1, -1, -1]),  # command: lin_x, lin_y, yaw_rate
        TRUE_VECTOR3,  # kick direction vector
        ReflectionSpec.identity(4),  # behavior flags
    ]
)


def _actor_spec(joint_spec: ReflectionSpec, action_spec: ReflectionSpec) -> ReflectionSpec:
    return ReflectionSpec.combine_many(
        [
            AXIAL_VECTOR3,  # base_ang_vel
            TRUE_VECTOR3,  # projected_gravity
            joint_spec,  # joint_pos
            joint_spec,  # joint_vel
            action_spec,  # last_action
            COMMAND_SPEC,  # command
        ]
    )


def _critic_spec(joint_spec: ReflectionSpec, action_spec: ReflectionSpec, gain_spec: ReflectionSpec) -> ReflectionSpec:
    return ReflectionSpec.combine_many(
        [
            AXIAL_VECTOR3,  # base_ang_vel
            TRUE_VECTOR3,  # projected_gravity
            joint_spec,  # joint_pos
            joint_spec,  # joint_vel
            action_spec,  # last_action
            action_spec,  # prev_prev_actions
            COMMAND_SPEC,  # command
            TRUE_VECTOR3,  # base_lin_vel
            LEFT_RIGHT_SCALARS,  # foot_height
            LEFT_RIGHT_SCALARS,  # foot_air_time
            LEFT_RIGHT_SCALARS,  # foot_contact
            LEFT_RIGHT_VECTOR3_BLOCKS,  # foot contact forces, left xyz then right xyz
            SCALAR,  # trunk_mass
            LEFT_RIGHT_SCALARS,  # foot_friction
            TRUE_VECTOR3,  # base_com
            gain_spec,  # default_KpKd_gains: kp block then kd block in actuator order
            gain_spec,  # special_KpKd_gains: kp block then kd block in actuator order
            joint_spec,  # encoder_bias
            TRUE_VECTOR3,  # push_force
        ]
    )


K1_VELOCITY_ACTOR_SPEC = _actor_spec(K1_LEG_JOINT_SPEC, K1_LEG_JOINT_SPEC)
K1_VELOCITY_CRITIC_SPEC = _critic_spec(K1_LEG_JOINT_SPEC, K1_LEG_JOINT_SPEC, K1_LEG_GAIN_SPEC)

K1_FULL_BODY_VELOCITY_ACTOR_SPEC = _actor_spec(K1_FULL_BODY_JOINT_SPEC, K1_FULL_BODY_ACTION_SPEC)
K1_FULL_BODY_VELOCITY_CRITIC_SPEC = _critic_spec(
    K1_FULL_BODY_JOINT_SPEC,
    K1_FULL_BODY_ACTION_SPEC,
    K1_FULL_BODY_GAIN_SPEC,
)


def k1_velocity_actor_model_kwargs() -> dict[str, object]:
    """Config fragment for an equivariant actor model.

    Usage is intentionally opt-in. Existing task registrations still use the
    normal RSL-RL MLPModel unless their ``class_name`` is changed explicitly.
    """
    return {
        "class_name": "konerl.symmetry.rsl_model.K1VelocityEquivariantMLPModel",
        "activation": "equiswish",
    }


def k1_velocity_critic_model_kwargs() -> dict[str, object]:
    """Config fragment for an invariant critic model."""
    return {
        "class_name": "konerl.symmetry.rsl_model.K1VelocityEquivariantMLPModel",
        "activation": "equiswish",
    }
