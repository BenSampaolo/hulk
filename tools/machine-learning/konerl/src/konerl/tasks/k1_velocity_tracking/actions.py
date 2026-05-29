from dataclasses import dataclass

import torch
import math

from mjlab.envs.mdp.actions.actions import JointPositionActionCfg, BaseAction
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .kick_command import KickCommandCfg
from .smoothed_command import SmoothedVelocityCommandCfg

DEFAULT_MOTION_RANGES = SmoothedVelocityCommandCfg.SmoothRanges(
    lin_vel_x=(-1.0, 2.0),
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-1.0, 1.0),
)

REL_STANDING_ENVS = 0.3

class GaitFrequencyAction(ActionTerm):
    """Controls an additive offset to the commanded gait frequency.

    The policy outputs an action in [-1, 1], which is scaled by `offset_scale`
    and added directly to the environment's nominal gait frequency.
    """

    def __init__(self, cfg: "GaitFrequencyActionCfg", env):
        super().__init__(cfg=cfg, env=env)
        self.cfg = cfg
        self._action_dim = 1
        
        self._raw_action = torch.zeros((self.num_envs, 1), device=self.device)
        self._freq_offset = torch.zeros((self.num_envs, 1), device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    @property
    def freq_offset(self) -> torch.Tensor:
        """The calculated frequency offset [Hz]."""
        return self._freq_offset

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_action[:] = actions
        
        # Clamped action [-1, 1] scaled by offset_scale
        clamped = torch.clamp(actions, min=-1.0, max=1.0)
        self._freq_offset[:] = clamped * self.cfg.offset_scale

    def apply_actions(self) -> None:
        pass

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_action[env_ids] = 0.0
        self._freq_offset[env_ids] = 0.0

@dataclass(kw_only=True)
class GaitFrequencyActionCfg(ActionTermCfg):
    offset_scale: float = 1.5 # The maximum Hz offset when action is 1.0 or -1.0

    def build(self, env) -> ActionTerm:
        return GaitFrequencyAction(cfg=self, env=env)


def make_actions_cfg() -> dict[str, ActionTermCfg]:
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        ),
        # "gait_frequency": GaitFrequencyActionCfg(
        #     entity_name="robot",
        #     offset_scale=0.8,
        # ),
    }


def make_commands_cfg() -> dict[str, CommandTermCfg]:
    return {
        # "twist": UniformVelocityCommandCfg(
        #     entity_name="robot",
        #     resampling_time_range=(3.0, 8.0),
        #     rel_standing_envs=0.1,
        #     rel_heading_envs=0.3,
        #     rel_forward_envs=0.2,
        #     heading_command=True,
        #     heading_control_stiffness=0.5,
        #     debug_vis=True,
        #     ranges=UniformVelocityCommandCfg.Ranges(
        #         lin_vel_x=(-1.0, 1.0),
        #         lin_vel_y=(-1.0, 1.0),
        #         ang_vel_z=(-1.5, 1.5),
        #         heading=(-math.pi, math.pi),
        #     ),
        # ),
        "twist": KickCommandCfg(
            entity_name="robot",
            ball_name="ball",
            resampling_time_range=(3.0, 15.0),
            rel_standing_envs=0.2,
            rel_walk_envs=0.8,
            rel_kick_envs=0.0,
            rel_dribble_envs=0.0,
            debug_vis=True,
            ranges=KickCommandCfg.Ranges(
                walking_lin_vel_x=(-1.5, 3.0),
                walking_lin_vel_y=(-1.5, 1.5),
                walking_ang_vel_z=(-2.0, 2.0),
                dribble_lin_vel_x=(-0.5, 0.5),
                dribble_lin_vel_y=(-0.5, 0.5),
                dribble_ang_vel_z=(-1.0, 1.0),
            ),
        )
    }

