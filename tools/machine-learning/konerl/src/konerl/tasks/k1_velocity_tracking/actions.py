from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg

from .kick_command import KickCommandCfg


def make_actions_cfg() -> dict[str, ActionTermCfg]:
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        ),
    }


def make_commands_cfg() -> dict[str, CommandTermCfg]:
    return {
        "twist": KickCommandCfg(
            entity_name="robot",
            ball_name="ball",
            resampling_time_range=(3.0, 15.0),
            rel_standing_envs=0.0,
            rel_walk_envs=0.0,
            rel_approach_envs=0.45,
            rel_kick_envs=0.55,
            rel_dribble_envs=0.0,
            debug_vis=True,
            ranges=KickCommandCfg.Ranges(
                walking_lin_vel_x=(-1.0, 2.0),
                walking_lin_vel_y=(-1.0, 1.0),
                walking_ang_vel_z=(-2.0, 2.0),
                dribble_lin_vel_x=(-0.5, 0.5),
                dribble_lin_vel_y=(-0.5, 0.5),
                dribble_ang_vel_z=(-1.0, 1.0),
                approach_lin_vel_x=(-0.2, 1.0),
                approach_lin_vel_y=(-0.5, 0.5),
                approach_ang_vel_z=(-1.5, 1.5),
                kick_vel=(0.8, 2.5),
                kick_lateral_offset=(-0.2, 0.2),
            ),
        )
    }
