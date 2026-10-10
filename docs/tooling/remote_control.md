# Remote Control

`./pepsi gammaray <robot>` disables the manufacturer controller services for HULK remote control.
`./pepsi boosterize <robot>` restores the `RemoteController` section in `/opt/booster/Daemon/bin/child.ini` and enables and starts `joystick_ros2`.

## Controls

Put the robot in `Initial`, then press Start on the controller to enable remote control.
Remote control cannot override `Damping`, `Prepare`, or `Stop`.

| Control | Action |
| --- | --- |
| Press Start | Toggle remote control on/off (starts off) |
| X / Square | Damping; overrides stand-up, including simultaneous button presses |
| Press A / Cross | Guarded, one-shot floor recovery (see below) |
| Primary state changes | Disable remote control |
| Controller disconnected or input stale | Stand while remote control remains enabled |
| Left stick up/down | Walk forward/backward |
| Left stick left/right | Walk sideways |
| Right stick left/right | Turn the robot |
| D-pad left/right | Move the head left/right |
| D-pad up/down | Move the head up/down |
| Hold left shoulder, L1/LB | Rumpelstilzchen kick |
| Hold right shoulder, R1/RB | Schlong kick |
| Analog triggers, L2/LT and R2/RT | Unbound |
| Right stick up/down | Unbound |

## Temporary Manual Floor Recovery

A / Cross works without enabling remote control. It requires a direct `Initial` → `Damping`
transition, then a new button press after all of these guards pass:

- Face up or down within 20° of horizontal, with quiet IMU angular velocity.
- Every joint within its configured prepare-position tolerance: head pitch 0.45 rad,
  shoulder pitch 0.675 rad, elbows 1.275 rad, knees 0.375 rad, and all other joints 0.30 rad.
  Absolute joint velocity must be at most 0.05 rad/s on every joint. These pose conditions
  must hold continuously for 300 ms. The upright-gravity prepare check is not used.
- Matching sensor/pose-approval source timestamps, at most 100 ms old; connected controller
  input at most 250 ms old; `fall_detection/status` reporting
  `Fallen { ready_for_standup: true }`, with both its publication and embedded sensor
  timestamps at most 100 ms old and not in the future. Detector readiness is an additional
  guard; it does not replace the manual pose checks above.

A rejected or held press is not queued: release and press again. Both release and press must
postdate the last stop; queued pre-stop input cannot activate recovery. An accepted press consumes
that `Initial` → `Damping` authorization and enters `ManualStandUp`, using the `GetUp` policy.
X / Square and emergency stops take priority. Stale/invalid inputs, a get-up inference failure,
detector `Falling` or `Fallen { ready_for_standup: false }`, or an eight-second timeout abort
to Damping without automatically restarting. `StandingUp` allows the attempt to continue.
A fresh `Upright` report with publication and sensor timestamps after the attempt started,
and upright IMU (within 0.2 rad), return the robot to
`Initial`, with remote control disabled. Another attempt requires a new `Initial` → `Damping`
transition. Ordinary remote-control commands and injected states cannot override the attempt.

Booster fall flags remain available as telemetry but are not used for control decisions.
The detector's upright dwell is 100 ms; this does not reproduce Booster's former recovery dwell.
These are activation guards, not a guarantee of physical safety or a fix for policy oscillation.
Use controlled, supported testing. In-flight inference is bounded by its configured timeout;
a newer Damping command is checked again before publishing its result.

## Restore Manufacturer Controls

Boosterize restores the `RemoteController` section in `/opt/booster/Daemon/bin/child.ini` and enables and starts `joystick_ros2`.
