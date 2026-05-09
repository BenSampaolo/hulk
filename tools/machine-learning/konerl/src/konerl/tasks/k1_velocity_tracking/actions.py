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
    """Controls the operating frequency of the gait clock.

    Instead of shifting a global clock (which requires infinite integration to track),
    this action outputs a frequency multiplier for the current timestep.
    Base frequency is 1 / gait_period.
    Multiplier range: [1.0, 3.0] (can step up to 3x faster to catch a fall).
    """

    def __init__(self, cfg: "GaitFrequencyActionCfg", env):
        super().__init__(cfg=cfg, env=env)
        self.cfg = cfg
        self._action_dim = 1
        
        # Raw action from policy [-1, 1] mapped to multiplier
        self._raw_action = torch.zeros((self.num_envs, 1), device=self.device)
        
        # The integrated phase accumulator in [0, 2pi)
        self._current_phase = torch.zeros((self.num_envs, 1), device=self.device)
        
        # The calculated frequency multiplier for this step
        self._freq_multiplier = torch.ones((self.num_envs, 1), device=self.device)
        
        self.base_freq = 1.0 / cfg.gait_period_s
        self.step_dt = getattr(env, "step_dt", 0.02) # Fallback to 50Hz

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    @property
    def current_phase(self) -> torch.Tensor:
        """The absolute phase of the gait cycle [0, 2pi)."""
        return self._current_phase
        
    @property
    def freq_multiplier(self) -> torch.Tensor:
        """The commanded frequency multiplier [1.0, 3.0]."""
        return self._freq_multiplier

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_action[:] = actions
        
        # Map policy output [-1, 1] -> [1.0, 3.0]
        # We clamp it just in case the policy outputs out of bounds
        clamped = torch.clamp(actions, min=-1.0, max=1.0)
        self._freq_multiplier[:] = 1.0 + (clamped + 1.0) # -1 -> 1.0, 1 -> 3.0
        
        # Integrate phase
        phase_delta = (2.0 * math.pi * self.base_freq * self._freq_multiplier) * self.step_dt
        self._current_phase[:] = (self._current_phase + phase_delta) % (2.0 * math.pi)

    def apply_actions(self) -> None:
        # This action only modifies internal state used by observations/rewards
        pass

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_action[env_ids] = 0.0
        self._current_phase[env_ids] = 0.0
        self._freq_multiplier[env_ids] = 1.0


@dataclass(kw_only=True)
class GaitFrequencyActionCfg(ActionTermCfg):
    gait_period_s: float = 0.8 # Base period in seconds

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
        #     gait_period_s=0.8,
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
            resampling_time_range=(10.0, 15.0),
            rel_standing_envs=0.3,
            rel_walk_envs=0.7,
            rel_kick_envs=0.0,
            rel_dribble_envs=0.0,
            debug_vis=True,
            ranges=KickCommandCfg.Ranges(
                walking_lin_vel_x=(-1.0, 1.0),
                walking_lin_vel_y=(-1.0, 1.0),
                walking_ang_vel_z=(-1.5, 1.5),
                dribble_lin_vel_x=(-0.5, 0.5),
                dribble_lin_vel_y=(-0.5, 0.5),
                dribble_ang_vel_z=(-1.0, 1.0),
                kick_vel=(0.2, 5.0),
                gait_frequency=(0.8, 2.5),
            ),
        )
    }

