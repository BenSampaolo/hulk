from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class KickCommand(CommandTerm):
    cfg: KickCommandCfg

    def __init__(self, cfg: KickCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        total_frac = self.cfg.rel_dribble_envs + self.cfg.rel_kick_envs + self.cfg.rel_standing_envs + self.cfg.rel_walk_envs
        if not math.isclose(total_frac, 1.0, rel_tol=1e-5):
            raise ValueError(f"The sum of environment type fractions must be 1.0, got {total_frac}")

        self.robot: Entity = env.scene[cfg.entity_name]
        self.ball: Entity = env.scene[cfg.ball_name]
        
        self.vel_command_b = torch.zeros([self.num_envs, 3], device=self.device)
        self.vel_command_w = torch.zeros([self.num_envs, 3], device=self.device)

        self.kick_direction_command_b = torch.zeros([self.num_envs, 3], device=self.device)
        self.kick_direction_command_w = torch.zeros([self.num_envs, 3], device=self.device)
        # Behavior flags: [
        #   Ball free, tells if the ball can be touched
        #   Kick, tells if the robot should kick the ball
        #   Dribble, tells if the robot should walk with the ball
        #   Standup allowed, tells if the robot can stand up
        # ]
        self.behavior_flags = torch.zeros(self.num_envs, 4, device=self.device)

        self.is_standing_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_walking_env = torch.zeros_like(self.is_standing_env)
        self.is_kicking_env = torch.zeros_like(self.is_standing_env)
        self.is_dribble_env = torch.zeros_like(self.is_standing_env)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_kick_acc"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_dribble_dst"] = torch.zeros(self.num_envs, device=self.device)
        
        self._joystick_enabled: viser.GuiCheckboxHandle | None = None
        self._joystick_sliders: list[viser.GuiSliderHandle] = []
        self._joystick_get_env_idx: Callable[[], int] | None = None

    """
    # Angular velocity 3x
    # Projected gravity 3x
    # Joint Position 12x
    # Joint velocity 12x
    # Ball Position 3x
    # Ball Velocity 3x
    # Last action 12x

    Walk command 3x
    Kick direction 3x
    Not touch 1x
    Kick 1x
    dribble 1x
    standup 1x
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([
            self.vel_command_b, 
            self.kick_direction_command_b, 
            self.behavior_flags
        ], dim=-1)

    def _update_metrics(self) -> None:
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
            ) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
            / max_command_step
        )
    
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)

        cumulative_fractions = torch.tensor([
            self.cfg.rel_dribble_envs,
            self.cfg.rel_dribble_envs + self.cfg.rel_kick_envs,
            self.cfg.rel_dribble_envs + self.cfg.rel_kick_envs + self.cfg.rel_standing_envs,
        ], device=self.device)

        rand_vals = r.uniform_(0.0, 1.0)

        self.is_dribble_env[env_ids] = rand_vals < cumulative_fractions[0]
        self.is_kicking_env[env_ids] = (rand_vals >= cumulative_fractions[0]) & (rand_vals < cumulative_fractions[1])
        self.is_standing_env[env_ids] = (rand_vals >= cumulative_fractions[1]) & (rand_vals < cumulative_fractions[2])
        self.is_walking_env[env_ids] = rand_vals >= cumulative_fractions[2] 


        self.vel_command_b[env_ids, 0] = torch.where(
            self.is_walking_env[env_ids],
            r.uniform_(*self.cfg.ranges.walking_lin_vel_x),
            torch.where(
                self.is_dribble_env[env_ids],
                r.uniform_(*self.cfg.ranges.dribble_lin_vel_x),
                torch.zeros_like(self.vel_command_b[env_ids, 0], device=self.device)
            )
        )
        self.vel_command_b[env_ids, 1] = torch.where(
            self.is_walking_env[env_ids],
            r.uniform_(*self.cfg.ranges.walking_lin_vel_y),
            torch.where(
                self.is_dribble_env[env_ids],
                r.uniform_(*self.cfg.ranges.dribble_lin_vel_y), 
                torch.zeros_like(self.vel_command_b[env_ids, 1], device=self.device)
            )
        )
        self.vel_command_b[env_ids, 2] = torch.where(
            self.is_walking_env[env_ids],
            r.uniform_(*self.cfg.ranges.walking_ang_vel_z),
            torch.where(
                self.is_dribble_env[env_ids],
                r.uniform_(*self.cfg.ranges.dribble_ang_vel_z),
                torch.zeros_like(self.vel_command_b[env_ids, 2], device=self.device)
            )
        )

        # This samples a random point of the upper half of the unit 
        # sphere and scales it by the kick velocity range
        kick_magnitudes = r.uniform_(*self.cfg.ranges.kick_vel).unsqueeze(-1)
        kick_directions = torch.nn.functional.normalize(
            torch.cat((
                torch.randn(len(env_ids), 2, device=self.device), 
                torch.randn(len(env_ids), 1, device=self.device).abs()
            ), dim=1), dim=1
        )
        self.kick_direction_command_b[env_ids] = torch.where(
            self.is_kicking_env[env_ids].unsqueeze(-1),
            kick_magnitudes * kick_directions,
            torch.zeros_like(self.kick_direction_command_b[env_ids], device=self.device)
        )

        self.behavior_flags[env_ids] = torch.stack((
            # Ball free 
            torch.where( 
                self.is_dribble_env[env_ids] | self.is_kicking_env[env_ids],
                torch.ones_like(self.behavior_flags[env_ids][:, 0], device=self.device),
                torch.zeros_like(self.behavior_flags[env_ids][:, 0], device=self.device)
            ),
            # Kick
            torch.where(
                self.is_kicking_env[env_ids],
                torch.ones_like(self.behavior_flags[env_ids][:, 1], device=self.device),
                torch.zeros_like(self.behavior_flags[env_ids][:, 1], device=self.device)
            ),
            # Dribble
            torch.where(
                self.is_dribble_env[env_ids],
                torch.ones_like(self.behavior_flags[env_ids][:, 2], device=self.device),
                torch.zeros_like(self.behavior_flags[env_ids][:, 2], device=self.device)
            ),
            torch.zeros_like(self.behavior_flags[env_ids][:, 3], device=self.device)
        ), dim=1)
        
    def _update_command(self) -> None:
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
        self.vel_command_w[standing_env_ids, :] = 0.0
        self.kick_direction_command_b[standing_env_ids, :] = 0.0
        self.kick_direction_command_w[standing_env_ids, :] = 0.0

    def create_gui(
        self,
        name, 
        server, 
        get_env_idx, 
        on_change = None, 
        request_action = None
    ) -> None:
        """Create velocity joystick sliders in the Viser viewer."""
        from viser import Icon

        ranges = self.cfg.ranges

        axes = [
            ("lin_vel_x", ranges.walking_lin_vel_x[1]),
            ("lin_vel_y", ranges.walking_lin_vel_y[1]),
            ("ang_vel_z", ranges.walking_ang_vel_z[1]),
            ("kick_vel", ranges.kick_vel[1]),
        ]
        sliders: list = []

        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable ", initial_value=False)

            for label, max_val in axes:
                max_input = server.gui.add_slider(
                    f"Max {label}",
                    initial_value=max_val,
                    step=0.1,
                    min=0.1,
                    max=10.0,
                )
                slider = server.gui.add_slider(
                    label,
                    min=-max_val,
                    max=max_val,
                    step=0.05,
                    initial_value=0.0,
                )

                @max_input.on_update
                def _(_ev, _s=slider, _m=max_input) -> None:
                    _s.min = -_m.value
                    _s.max = _m.value
                
                sliders.append(slider)

            zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

            @zero_btn.on_click
            def _(_ev) -> None:
                for slider in sliders:
                    slider.value = 0.0
        
        self._joystick_enabled = enabled
        self._joystick_sliders = sliders
        self._joystick_get_env_idx = get_env_idx
    
    def compute(self, dt: float) -> None:
        super().compute(dt)
        if self._joystick_enabled is not None and self._joystick_enabled.value:
            assert self._joystick_get_env_idx is not None
            idx = self._joystick_get_env_idx()
            for i, s in enumerate(self._joystick_sliders):
                if i < self.vel_command_b.shape[1]:
                    self.vel_command_b[idx, i] = s.value
    
    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        # Visualize the velocity command as an arrow in the robot's local frame
        
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        
        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
        lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
        ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset

        for batch in env_indices:
            base_pos_w = base_pos_ws[batch]
            base_mat_w = base_mat_ws[batch]
            cmd = cmds[batch]
            lin_vel_b = lin_vel_bs[batch]
            ang_vel_b = ang_vel_bs[batch]

            # Skip if robot appears uninitialized (at origin).
            if np.linalg.norm(base_pos_w) < 1e-6:
                continue
            
            # Helper to transform local to world coordinates.
            def local_to_world(
                vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
            ) -> np.ndarray:
                return pos + mat @ vec
            
            # Command linear velocity arrow (blue).
            cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
            cmd_lin_to = local_to_world(
                (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
            )
            visualizer.add_arrow(
                cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
            )

            # Command angular velocity arrow (green).
            cmd_ang_from = cmd_lin_from
            cmd_ang_to = local_to_world(
                (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
            )
            visualizer.add_arrow(
                cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
            )

            # Actual linear velocity arrow (cyan).
            act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
            act_lin_to = local_to_world(
                (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
            )
            visualizer.add_arrow(
                act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
            )

            # Actual angular velocity arrow (light green).
            act_ang_from = act_lin_from
            act_ang_to = local_to_world(
                (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
            )
            visualizer.add_arrow(
                act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
            )

            # Kick direction arrow (red)
            kick_cmd_from = self.ball.data.root_link_pos_w[batch].cpu().numpy()
            kick_cmd_to = kick_cmd_from + cmd[3:6] * scale
            visualizer.add_arrow(
                kick_cmd_from, kick_cmd_to, color=(0.8, 0.2, 0.2, 0.7), width=0.015
            )


@dataclass(kw_only=True)
class KickCommandCfg(CommandTermCfg):
    entity_name: str
    ball_name: str

    rel_dribble_envs: float = 0.0
    rel_kick_envs: float = 0.0
    rel_standing_envs: float = 0.0
    rel_walk_envs: float = 0.0

    @dataclass
    class Ranges:
        walking_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        walking_lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        walking_ang_vel_z: tuple[float, float] = (-1.5, 1.5)

        dribble_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        dribble_lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        dribble_ang_vel_z: tuple[float, float] = (-1.5, 1.5)

        kick_vel: tuple[float, float] = (0.1, 5.0)
    
    ranges: Ranges

    @dataclass
    class VizCfg:
        z_offset: float = 0.0
        scale: float = 0.5
    
    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> KickCommand:
        return KickCommand(self, env)