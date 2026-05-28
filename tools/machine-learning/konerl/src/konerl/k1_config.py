from pathlib import Path
import math
from typing import Any, cast

import mujoco
from mjlab.actuator.builtin_actuator import BuiltinPositionActuatorCfg as BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

K1_XML = Path("model/K1.xml").resolve()
assert K1_XML.exists()

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

K1_DEFAULT_JOINT_POS: dict[str, float] = {
    "ALeft_Shoulder_Pitch": 0.25,
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Pitch": 0.15,
    "Left_Elbow_Yaw": -2.25,
    "ARight_Shoulder_Pitch": 0.25,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Pitch": 0.15,
    "Right_Elbow_Yaw": 2.25,
    "Left_Hip_Pitch": -0.3,
    "Left_Hip_Roll": 0.1,
    "Left_Knee_Pitch": 0.6,
    "Left_Ankle_Pitch": -0.3,
    "Left_Ankle_Roll": -0.1,
    "Right_Hip_Pitch": -0.3,
    "Right_Hip_Roll": -0.1,
    "Right_Knee_Pitch": 0.6,
    "Right_Ankle_Pitch": -0.3,
    "Right_Ankle_Roll": 0.1,
    ".*": 0.0,
}


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    mesh_root = (K1_XML.parent / meshdir).resolve()
    if not mesh_root.exists():
        return assets
    for path in mesh_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(mesh_root).as_posix()
        assets[f"{meshdir}/{rel}"] = path.read_bytes()
    return assets


def _axis_angle_quat(axis, angle: float) -> tuple[float, float, float, float]:
    norm = math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(0.5 * angle) / norm
    return (math.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s)


def _mul_quat(a, b) -> tuple[float, float, float, float]:
    return (
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


def _bake_joint_pose_into_body(spec: mujoco.MjSpec, joint_name: str, joint_pos: float) -> None:
    joint = cast(Any, spec).joint(joint_name)
    body = joint.parent
    joint_quat = _axis_angle_quat(joint.axis, joint_pos)
    body_quat = tuple(float(x) for x in body.quat)
    body.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    body.quat[:] = _mul_quat(joint_quat, body_quat)


def _bake_default_arm_pose(spec: mujoco.MjSpec) -> None:
    for joint_name in K1_ARM_JOINT_NAMES:
        _bake_joint_pose_into_body(spec, joint_name, K1_DEFAULT_JOINT_POS[joint_name])


def get_spec(*, control_arms: bool = False) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(K1_XML))
    for geom in tuple(spec.geoms):
        if geom.name == "ground":
            spec.delete(geom)  # type: ignore[attr-defined]
            break
    if not control_arms:
        # XML zero is a neutral T-pose. Bake the configured K1 default arm pose
        # before welding out arm joints for leg-only policies.
        _bake_default_arm_pose(spec)
        spec_any = cast(Any, spec)
        for joint_name in K1_ARM_JOINT_NAMES:
            spec_any.delete(spec_any.joint(joint_name))
    spec.assets = get_assets(spec.meshdir)
    return spec


# Initial States

ZERO_POSE = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.6),
    joint_pos=dict(K1_DEFAULT_JOINT_POS),
    joint_vel={".*": 0.0},
)

# Collison Configs

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision.*", r"(left|right)_foot"),
    condim=3,
    priority=1,
    friction=(2.0,),
    solimp=(0.01, 0.995, 0.0),
    solref=(0.01, 1.0),
)

FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"(left|right)_foot",),
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(2.0,),
    solimp=(0.01, 0.995, 0.025),
    solref=(0.01, 1.0),
)

# Actuators and Articulation

HEAD = BuiltinPositionActuatorCfg(
    target_names_expr=("AAHead_yaw", "Head_pitch"),
    stiffness=10.0,
    damping=1.0,
    effort_limit=6.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

SHOULDER_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=("ALeft_Shoulder_Pitch", "ARight_Shoulder_Pitch"),
    stiffness=10.0,
    damping=1.0,
    effort_limit=14.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

SHOULDER_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Shoulder_Roll", "Right_Shoulder_Roll"),
    stiffness=10.0,
    damping=1.0,
    effort_limit=14.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

ELBOW_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Elbow_Pitch", "Right_Elbow_Pitch"),
    stiffness=10.0,
    damping=1.0,
    effort_limit=14.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

ELBOW_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Elbow_Yaw", "Right_Elbow_Yaw"),
    stiffness=10.0,
    damping=1.0,
    effort_limit=14.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

HIP_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Hip_Pitch", "Right_Hip_Pitch"),
    stiffness=80.0,
    damping=4.0,
    effort_limit=30.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Hip_Roll", "Right_Hip_Roll"),
    stiffness=80.0,
    damping=4.0,
    effort_limit=35.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

HIP_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Hip_Yaw", "Right_Hip_Yaw"),
    stiffness=80.0,
    damping=4.0,
    effort_limit=20.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

KNEE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Knee_Pitch", "Right_Knee_Pitch"),
    stiffness=80.0,
    damping=4.0,
    effort_limit=40.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

ANKLE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Ankle_Pitch", "Right_Ankle_Pitch"),
    stiffness=25.0,
    damping=2.0,
    effort_limit=20.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

ANKLE_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Ankle_Roll", "Right_Ankle_Roll"),
    stiffness=60.0,
    damping=2.0,
    effort_limit=20.0,
    delay_min_lag=1,
    delay_max_lag=2,
)

K1_LEG_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        HIP_PITCH,
        HIP_ROLL,
        HIP_YAW,
        KNEE_PITCH,
        ANKLE_PITCH,
        ANKLE_ROLL,
    ),
    soft_joint_pos_limit_factor=0.9,
)

K1_FULL_BODY_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        SHOULDER_PITCH,
        SHOULDER_ROLL,
        ELBOW_PITCH,
        ELBOW_YAW,
        HIP_PITCH,
        HIP_ROLL,
        HIP_YAW,
        KNEE_PITCH,
        ANKLE_PITCH,
        ANKLE_ROLL,
    ),
    soft_joint_pos_limit_factor=0.9,
)

# Backward-compatible alias for existing leg-only code.
K1_ARTICULATION = K1_LEG_ARTICULATION


def get_k1_robot_cfg(*, control_arms: bool = False) -> EntityCfg:
    return EntityCfg(
        init_state=ZERO_POSE,
        collisions=(FULL_COLLISION,),
        spec_fn=lambda: get_spec(control_arms=control_arms),
        articulation=K1_FULL_BODY_ARTICULATION if control_arms else K1_LEG_ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    scene = Scene(SceneCfg(terrain=TerrainEntityCfg(), entities={"robot": get_k1_robot_cfg()}), "cpu")
    model = scene.compile()
    viewer.launch(model)
