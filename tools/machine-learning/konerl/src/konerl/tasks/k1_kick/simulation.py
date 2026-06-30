from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from konerl.ball_config import get_ball_cfg
from konerl.k1_config import get_k1_robot_cfg
from konerl.scripts.AMP.features import K1_AMP_JOINT_NAMES

from .actions import make_actions_cfg, make_commands_cfg
from .curriculum import make_curriculum_cfg
from .metrics import make_metric_cfg
from .observations import make_observation_cfg
from .randomization import make_events_cfg
from .rewards import make_reward_cfg
from .termination import make_termination_cfg


def _ensure_leg_only(control_arms: bool) -> None:
    if control_arms:
        raise ValueError("k1_kick is a PiPlus leg-only kick task; control_arms=True is not supported")


def make_scene_cfg(terrain_type: Literal["flat"] = "flat", *, control_arms: bool = False) -> SceneCfg:
    _ensure_leg_only(control_arms)
    if terrain_type != "flat":
        raise ValueError("k1_kick is flat-only; rough/bumpy terrain is not part of the PiPlus kick task")

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=("Left_Ankle_Cross", "Right_Ankle_Cross"),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    return SceneCfg(
        terrain=TerrainEntityCfg(),
        sensors=(feet_ground_cfg,),
        entities={
            "robot": get_k1_robot_cfg(control_arms=False),
            "ball": get_ball_cfg(),
        },
        num_envs=1,
        extent=2.0,
    )


def make_kick_env_cfg(play: bool, *, control_arms: bool = False) -> ManagerBasedRlEnvCfg:
    _ensure_leg_only(control_arms)
    controlled_joint_names = K1_AMP_JOINT_NAMES
    return ManagerBasedRlEnvCfg(
        scene=make_scene_cfg("flat", control_arms=False),
        observations=make_observation_cfg(controlled_joint_names=controlled_joint_names),
        actions=make_actions_cfg(),
        commands=make_commands_cfg(),
        events=make_events_cfg(),
        rewards=make_reward_cfg(controlled_joint_names=controlled_joint_names),
        terminations=make_termination_cfg(),
        curriculum=make_curriculum_cfg("flat"),
        metrics=make_metric_cfg(),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="Trunk",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=128,
            njmax=256,
            mujoco=MujocoCfg(
                timestep=0.002,
                iterations=50,
                ls_iterations=50,
            ),
        ),
        decimation=10,
        episode_length_s=int(1e9) if play else 10.0,
    )
