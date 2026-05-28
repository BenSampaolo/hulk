from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

# Keep the AMP feature joint order identical for expert mocap, policy rollouts,
# and reward computation. This is the robot's joint order for the controlled legs.
K1_AMP_JOINT_NAMES: tuple[str, ...] = (
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

MOCAP_TO_K1: dict[str, str] = {
    "Head1": "AAHead_yaw",
    "Head2": "Head_pitch",
    "Left_Arm_1": "ALeft_Shoulder_Pitch",
    "Right_Arm_1": "ARight_Shoulder_Pitch",
    "Left_Arm_2": "Left_Shoulder_Roll",
    "Right_Arm_2": "Right_Shoulder_Roll",
    "Left_Arm_3": "Left_Elbow_Pitch",
    "Right_Arm_3": "Right_Elbow_Pitch",
    "left_hand_link": "Left_Elbow_Yaw",
    "right_hand_link": "Right_Elbow_Yaw",
    "Left_Hip_Pitch": "Left_Hip_Pitch",
    "Right_Hip_Pitch": "Right_Hip_Pitch",
    "Left_Hip_Roll": "Left_Hip_Roll",
    "Right_Hip_Roll": "Right_Hip_Roll",
    "Left_Hip_Yaw": "Left_Hip_Yaw",
    "Right_Hip_Yaw": "Right_Hip_Yaw",
    "Left_Shank": "Left_Knee_Pitch",
    "Right_Shank": "Right_Knee_Pitch",
    "Left_Ankle_Cross": "Left_Ankle_Pitch",
    "Right_Ankle_Cross": "Right_Ankle_Pitch",
    "left_foot_link": "Left_Ankle_Roll",
    "right_foot_link": "Right_Ankle_Roll",
}


def controlled_joint_names_from_env(env: Any) -> tuple[str, ...]:
    """Return controlled joints in robot data order, not actuator declaration order."""
    robot = env.unwrapped.scene["robot"] if hasattr(env, "unwrapped") else env.scene["robot"]
    robot_joint_names = tuple(robot.joint_names)

    controlled: set[str] = set()
    robot_cfg = env.cfg.scene.entities.get("robot", None)
    articulation = getattr(robot_cfg, "articulation", None)
    if articulation is not None:
        for actuator in articulation.actuators:
            controlled.update(getattr(actuator, "target_names_expr", ()))

    if not controlled:
        return tuple(name for name in K1_AMP_JOINT_NAMES if name in robot_joint_names)

    return tuple(name for name in robot_joint_names if name in controlled)


def joint_indices(robot: Any, joint_names: Iterable[str], device: torch.device | str) -> torch.Tensor:
    indices: list[int] = []
    available = tuple(robot.joint_names)
    for name in joint_names:
        try:
            indices.append(available.index(name))
        except ValueError as exc:
            raise KeyError(f"AMP joint '{name}' is not present in robot joints: {available}") from exc
    return torch.tensor(indices, device=device, dtype=torch.long)


def amp_features_from_robot(robot: Any, joint_names: Iterable[str], device: torch.device | str) -> torch.Tensor:
    ids = joint_indices(robot, joint_names, device)
    return torch.cat(
        [
            robot.data.joint_pos[:, ids],
            robot.data.joint_vel[:, ids],
            robot.data.root_link_lin_vel_b,
            robot.data.root_link_ang_vel_b,
        ],
        dim=-1,
    )
