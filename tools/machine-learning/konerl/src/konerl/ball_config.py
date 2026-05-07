from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

BALL_XML = Path("model/ball.xml").resolve()
assert BALL_XML.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(BALL_XML))


# Initial States

BALL_ZERO_POSE = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.15),
)

# Collision Configs

BALL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*",),
    contype=1,
    conaffinity=1,
    condim=3,
    priority=0,
    friction=(0.8,),
)


def get_ball_cfg() -> EntityCfg:
    """Get the ball entity configuration."""
    return EntityCfg(
        init_state=BALL_ZERO_POSE,
        collisions=(BALL_COLLISION,),
        spec_fn=get_spec,
    )
