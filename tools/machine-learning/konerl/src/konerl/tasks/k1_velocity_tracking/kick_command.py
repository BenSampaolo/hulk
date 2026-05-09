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


def _calculate_yaw_only_rotation(trunk_quat_w: torch.Tensor) -> np.ndarray:
    """Helper to compute yaw-only rotation matrix."""
    yaw = torch.atan2(
        2.0 * (trunk_quat_w[0] * trunk_quat_w[3] + trunk_quat_w[1] * trunk_quat_w[2]),
        1.0 - 2.0 * (trunk_quat_w[2]**2 + trunk_quat_w[3]**2)
    )
    yaw_only_quat_w = torch.stack([
        torch.cos(yaw / 2.0),
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        torch.sin(yaw / 2.0)
    ], dim=-1)
    return matrix_from_quat(yaw_only_quat_w).cpu().numpy()

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

        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_kick_acc"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_dribble_dst"] = torch.zeros(self.num_envs, device=self.device)
        
        self._joystick_enabled: viser.GuiCheckboxHandle | None = None
        self._joystick_sliders: list[viser.GuiSliderHandle] = []
        self._joystick_gait_freq_slider: viser.GuiSliderHandle | None = None
        self._joystick_get_env_idx: Callable[[], int] | None = None

        self._show_ball_pos: viser.GuiCheckboxHandle | None = None
        self._show_ball_vel: viser.GuiCheckboxHandle | None = None
        self._show_robot_vel: viser.GuiCheckboxHandle | None = None
        self._show_behavior_flags: viser.GuiCheckboxHandle | None = None
        self._show_projected_gravity: viser.GuiCheckboxHandle | None = None

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

        # Assign gait frequency based on env type. Walking gets frequency, standing gets 0.
        # Dribbling and kicking might also want a gait frequency if they involve movement.
        is_moving = self.is_walking_env[env_ids] | self.is_dribble_env[env_ids] | self.is_kicking_env[env_ids]
        
        self.gait_frequency[env_ids] = torch.where(
            is_moving,
            r.uniform_(*self.cfg.ranges.gait_frequency),
            torch.zeros_like(self.gait_frequency[env_ids], device=self.device)
        )

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

            # Add Gait Frequency Slider
            gait_freq_slider = server.gui.add_slider(
                "gait_freq",
                min=0.5,
                max=5.0,
                step=0.1,
                initial_value=1.0,
            )
        
        self._joystick_enabled = enabled
        self._joystick_sliders = sliders
        self._joystick_gait_freq_slider = gait_freq_slider
        self._joystick_get_env_idx = get_env_idx
        
        with server.gui.add_folder("Debug Visualization"):
            self._show_behavior_flags = server.gui.add_checkbox("Show Behavior Flags", initial_value=True)
            self._show_projected_gravity = server.gui.add_checkbox("Show Projected Gravity", initial_value=False)
            self._show_ball_pos = server.gui.add_checkbox("Show Ball Position Obs", initial_value=False)
            self._show_ball_vel = server.gui.add_checkbox("Show Ball Velocity Obs", initial_value=False)
            self._show_robot_vel = server.gui.add_checkbox("Show Robot Vel Commands", initial_value=True)
    
    def compute(self, dt: float) -> None:
        super().compute(dt)
        if self._joystick_enabled is not None and self._joystick_enabled.value:
            assert self._joystick_get_env_idx is not None
            idx = self._joystick_get_env_idx()
            for i, s in enumerate(self._joystick_sliders):
                if i < self.vel_command_b.shape[1]:
                    self.vel_command_b[idx, i] = s.value
            if self._joystick_gait_freq_slider is not None:
                self.gait_frequency[idx] = self._joystick_gait_freq_slider.value

        # Dynamically suppress gait frequency if velocity command is below threshold (0.05)
        vel_norms = torch.norm(self.vel_command_b, dim=-1)
        effective_gait_freq = torch.where(
            vel_norms < 0.05,
            torch.zeros_like(self.gait_frequency),
            self.gait_frequency
        )
        
        # Advance gait process
        self.gait_process[:] = torch.fmod(self.gait_process + dt * effective_gait_freq, 1.0)
    
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
            
            show_robot_vel = self._show_robot_vel is None or self._show_robot_vel.value

            if show_robot_vel:
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
                    (np.array([0, 0, z_offset]) + np.array([ang_vel_b[0], ang_vel_b[1], ang_vel_b[2]])) * scale * 0.5
                )
                visualizer.add_arrow(
                    act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
                )

            # Projected Gravity visualization (purple downward vector)
            show_proj_grav = self._show_projected_gravity is None or self._show_projected_gravity.value
            if show_proj_grav:
                grav_from = local_to_world(np.array([0, 0, z_offset]) * scale)
                grav_to = local_to_world(np.array([0, 0, z_offset]) * scale) - np.array([0, 0, 1.0]) * scale
                visualizer.add_arrow(
                    grav_from, grav_to, color=(0.8, 0.0, 0.8, 0.7), width=0.015
                )

            # Behavior Flags dots (Above robot's head)
            show_flags = self._show_behavior_flags is None or self._show_behavior_flags.value
            if show_flags:
                is_dribble = bool(self.is_dribble_env[batch].item())
                is_kicking = bool(self.is_kicking_env[batch].item())
                is_standing = bool(self.is_standing_env[batch].item())
                is_walking = bool(self.is_walking_env[batch].item())

                flag_height = 0.5 + z_offset # above head
                spacing = 0.1
                
                # 1. Kicking (Red)
                kick_color = (1.0, 0.0, 0.0, 1.0) if is_kicking else (0.1, 0.0, 0.0, 0.2)
                visualizer.add_sphere(
                    local_to_world(np.array([-0.5 * spacing, -0.5 * spacing, flag_height])), radius=0.03, color=kick_color
                )
                
                # 2. Walking (Green)
                walk_color = (0.0, 1.0, 0.0, 1.0) if is_walking else (0.0, 0.1, 0.0, 0.2)
                visualizer.add_sphere(
                    local_to_world(np.array([-0.5 * spacing, 0.5 * spacing, flag_height])), radius=0.03, color=walk_color
                )

                # 3. Dribbling (Blue)
                dribble_color = (0.0, 0.0, 1.0, 1.0) if is_dribble else (0.0, 0.0, 0.1, 0.2)
                visualizer.add_sphere(
                    local_to_world(np.array([0.5 * spacing, -0.5 * spacing, flag_height])), radius=0.03, color=dribble_color
                )

                # 4. Standing (White)
                stand_color = (1.0, 1.0, 1.0, 1.0) if is_standing else (0.1, 0.1, 0.1, 0.2)
                visualizer.add_sphere(
                    local_to_world(np.array([0.5 * spacing, 0.5 * spacing, flag_height])), radius=0.03, color=stand_color
                )

            # Kick direction arrow (red)
            kick_cmd_from = self.ball.data.root_link_pos_w[batch].cpu().numpy()
            kick_cmd_to = kick_cmd_from + cmd[3:6] * scale
            visualizer.add_arrow(
                kick_cmd_from, kick_cmd_to, color=(0.8, 0.2, 0.2, 0.7), width=0.015
            )

            if self._show_ball_pos is not None and self._show_ball_pos.value:
                from konerl.tasks.k1_velocity_tracking.observations import obs_ball_pos_heading_frame
                
                # Fetch observation logic from observations.py
                ball_pos_obs = obs_ball_pos_heading_frame(self._env, outside_info=True)
                
                trunk_pos_w = self.robot.data.root_link_pos_w[batch].cpu().numpy()
                trunk_quat_w = self.robot.data.root_link_quat_w[batch]
                
                rot_mat = _calculate_yaw_only_rotation(trunk_quat_w)
                
                obs_local = ball_pos_obs[batch].cpu().numpy()
                
                # Re-project to world frame
                obs_world_rel = rot_mat @ obs_local
                
                visualizer.add_arrow(
                    trunk_pos_w, trunk_pos_w + obs_world_rel, color=(1.0, 1.0, 0.0, 0.7), width=0.012
                )
            
            if self._show_ball_vel is not None and self._show_ball_vel.value:
                from konerl.tasks.k1_velocity_tracking.observations import obs_ball_vel_heading_frame
                
                # Fetch observation logic from observations.py
                ball_vel_obs = obs_ball_vel_heading_frame(self._env, outside_info=True)
                
                ball_pos_w = self.ball.data.root_link_pos_w[batch].cpu().numpy()
                trunk_quat_w = self.robot.data.root_link_quat_w[batch]

                rot_mat = _calculate_yaw_only_rotation(trunk_quat_w)
                
                obs_local = ball_vel_obs[batch].cpu().numpy()

                # Re-project to world frame
                obs_world_vel = rot_mat @ obs_local

                visualizer.add_arrow(
                    ball_pos_w, ball_pos_w + obs_world_vel * scale, color=(1.0, 0.5, 0.0, 0.7), width=0.012
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
        gait_frequency: tuple[float, float] = (1.0, 2.0)  # steps per second
    
    ranges: Ranges

    @dataclass
    class VizCfg:
        z_offset: float = 0.0
        scale: float = 0.5
    
    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> KickCommand:
        return KickCommand(self, env)