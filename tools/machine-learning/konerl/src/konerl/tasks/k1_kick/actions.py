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
        "kick": KickCommandCfg(
            entity_name="robot",
            ball_name="ball",
            resampling_time_range=(3.0, 8.0),
            debug_vis=True,
        ),
    }
