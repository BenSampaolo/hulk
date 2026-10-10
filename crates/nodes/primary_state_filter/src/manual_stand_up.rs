use std::time::Duration;

use booster::LowState;
use ros_z::time::Time;
use types::{
    controller_input::{Button, ControllerInput},
    fall_detection::{FallDetection, MAXIMUM_FALL_DETECTION_AGE, Posture},
};

const CONTROLLER_MAX_AGE: Duration = Duration::from_millis(250);
const SENSOR_MAX_AGE: Duration = Duration::from_millis(100);
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(8);

pub(super) struct Sample<T> {
    latest: Option<(Time, T)>,
    last_source_time: Option<Time>,
}

impl<T> Default for Sample<T> {
    fn default() -> Self {
        Self {
            latest: None,
            last_source_time: None,
        }
    }
}

impl<T> Sample<T> {
    fn update(&mut self, value: T, source_time: Time, now: Time, max_age: Duration) -> bool {
        let ordered = self.last_source_time.is_none_or(|last| source_time > last);
        if source_time <= now && ordered {
            self.last_source_time = Some(source_time);
        }
        if !ordered || source_time > now || now.duration_since(source_time) > max_age {
            self.invalidate();
            return false;
        }
        self.latest = Some((source_time, value));
        true
    }

    pub(super) fn invalidate(&mut self) {
        self.latest = None;
    }

    fn fresh(&self, now: Time, max_age: Duration) -> Option<(Time, &T)> {
        self.latest.as_ref().and_then(|(source_time, value)| {
            (*source_time <= now && now.duration_since(*source_time) <= max_age)
                .then_some((*source_time, value))
        })
    }
}

#[derive(Default)]
pub(super) struct ManualStandUp {
    pub(super) eligible: bool,
    pub(super) started: Option<Time>,
    pub(super) low_state: Sample<bool>,
    pub(super) safe_pose: Sample<bool>,
    pub(super) fall: Sample<FallDetection>,
    controller: Sample<(bool, bool)>,
    south_pressed: Option<bool>,
    damping_requested_at: Option<Time>,
}

impl ManualStandUp {
    pub(super) fn update_controller(
        &mut self,
        input: &ControllerInput,
        source_time: Time,
        now: Time,
    ) -> bool {
        if !self.controller_is_fresh(now) {
            self.south_pressed = None;
        }
        let valid = input.axes.iter().all(|axis| axis.value.is_finite())
            && input.buttons.iter().all(|button| button.value.is_finite());
        if !self.controller.update(
            (input.connected, input.is_pressed(Button::West)),
            source_time,
            now,
            CONTROLLER_MAX_AGE,
        ) || !valid
            || !input.connected
        {
            self.invalidate_controller();
            return false;
        }
        if self
            .damping_requested_at
            .is_some_and(|stop| source_time <= stop)
        {
            self.south_pressed = None;
            return false;
        }
        let pressed = input.is_pressed(Button::South);
        let rising_edge = self.south_pressed == Some(false) && pressed;
        self.south_pressed = Some(pressed);
        rising_edge
    }

    pub(super) fn invalidate_controller(&mut self) {
        self.controller.invalidate();
        self.south_pressed = None;
    }

    pub(super) fn update_low_state(&mut self, low_state: &LowState, source_time: Time, now: Time) {
        let imu = &low_state.imu_state;
        let valid = low_state.motor_state_serial.len() == 22
            && low_state.motor_state_serial.iter().all(|motor| {
                [
                    motor.position,
                    motor.velocity,
                    motor.acceleration,
                    motor.torque,
                ]
                .iter()
                .all(|value| value.is_finite())
                    && motor.lost == 0
            })
            && imu
                .roll_pitch_yaw
                .inner
                .iter()
                .chain(imu.angular_velocity.inner.iter())
                .chain(imu.linear_acceleration.inner.iter())
                .all(|value| value.is_finite());
        let upright = imu.roll_pitch_yaw.x().cos() * imu.roll_pitch_yaw.y().cos() >= 0.2_f32.cos();
        self.low_state
            .update(upright, source_time, now, SENSOR_MAX_AGE);
        if !valid {
            self.low_state.invalidate();
        }
    }

    pub(super) fn update_safe_pose(&mut self, safe: bool, source_time: Time, now: Time) {
        self.safe_pose
            .update(safe, source_time, now, SENSOR_MAX_AGE);
    }

    pub(super) fn update_fall(&mut self, fall: FallDetection, source_time: Time, now: Time) {
        self.fall
            .update(fall, source_time, now, MAXIMUM_FALL_DETECTION_AGE);
        if !fall.is_fresh(now) {
            self.fall.invalidate();
        }
    }

    fn fresh_fall(&self, now: Time) -> Option<(Time, &FallDetection)> {
        self.fall
            .fresh(now, MAXIMUM_FALL_DETECTION_AGE)
            .filter(|(_, fall)| fall.is_fresh(now))
    }

    fn controller_is_fresh(&self, now: Time) -> bool {
        self.controller
            .fresh(now, CONTROLLER_MAX_AGE)
            .is_some_and(|(_, (connected, _))| *connected)
    }

    pub(super) fn damping_pressed(&self, now: Time) -> bool {
        self.controller
            .fresh(now, CONTROLLER_MAX_AGE)
            .is_some_and(|(_, (_, west))| *west)
    }

    pub(super) fn start_rejection(&self, now: Time) -> Option<&'static str> {
        if !self.eligible {
            Some("requires a direct Initial -> Damping transition")
        } else if !self.controller_is_fresh(now) {
            Some("controller is disconnected, invalid or older than 250ms")
        } else if self.damping_pressed(now) {
            Some("damping button is pressed")
        } else if self.low_state.fresh(now, SENSOR_MAX_AGE).is_none() {
            Some("aligned LowState is invalid or older than 100ms")
        } else if !self
            .safe_pose
            .fresh(now, SENSOR_MAX_AGE)
            .is_some_and(|(_, safe)| *safe)
        {
            Some("manual floor pose is unsafe or older than 100ms")
        } else if self
            .low_state
            .fresh(now, SENSOR_MAX_AGE)
            .map(|(time, _)| time)
            != self
                .safe_pose
                .fresh(now, SENSOR_MAX_AGE)
                .map(|(time, _)| time)
        {
            Some("latest aligned LowState has no matching manual-safe-pose approval")
        } else if !self.fresh_fall(now).is_some_and(|(_, fall)| {
            fall.posture
                == Posture::Fallen {
                    ready_for_standup: true,
                }
        }) {
            Some("requires fresh Fallen with ready_for_standup, no older than 100ms")
        } else {
            None
        }
    }

    pub(super) fn must_abort(&self, now: Time) -> bool {
        let Some(started) = self.started else {
            return true;
        };
        now < started
            || now.duration_since(started) >= ATTEMPT_TIMEOUT
            || !self.controller_is_fresh(now)
            || self.low_state.fresh(now, SENSOR_MAX_AGE).is_none()
            || self.fresh_fall(now).is_none_or(|(_, fall)| {
                !matches!(
                    fall.posture,
                    Posture::Upright
                        | Posture::StandingUp
                        | Posture::Fallen {
                            ready_for_standup: true
                        }
                )
            })
    }

    pub(super) fn is_complete(&self, now: Time) -> bool {
        self.started.is_some_and(|started| {
            self.fresh_fall(now).is_some_and(|(time, fall)| {
                time > started && fall.time > started && fall.posture == Posture::Upright
            }) && self
                .low_state
                .fresh(now, SENSOR_MAX_AGE)
                .is_some_and(|(_, upright)| *upright)
        })
    }

    pub(super) fn record_damping_request(&mut self, now: Time) {
        self.damping_requested_at =
            Some(self.damping_requested_at.map_or(now, |stop| stop.max(now)));
        self.reset_button_edge();
    }

    pub(super) fn reset_button_edge(&mut self) {
        self.south_pressed = None;
    }

    pub(super) fn expire_controller_edge(&mut self, now: Time) {
        if !self.controller_is_fresh(now) {
            self.south_pressed = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time(milliseconds: i64) -> Time {
        Time::from_nanos(milliseconds * 1_000_000)
    }

    fn armed(now: Time) -> ManualStandUp {
        let mut manual = ManualStandUp {
            eligible: true,
            ..Default::default()
        };
        manual
            .controller
            .update((true, false), now, now, CONTROLLER_MAX_AGE);
        manual.low_state.update(false, now, now, SENSOR_MAX_AGE);
        manual.update_safe_pose(true, now, now);
        manual
    }

    #[test]
    fn only_ready_fallen_posture_allows_entry() {
        let now = time(1000);
        assert!(armed(now).start_rejection(now).is_some());
        for posture in [
            Posture::Upright,
            Posture::Falling,
            Posture::StandingUp,
            Posture::Fallen {
                ready_for_standup: false,
            },
            Posture::Fallen {
                ready_for_standup: true,
            },
        ] {
            let mut manual = armed(now);
            manual.update_fall(FallDetection { time: now, posture }, now, now);
            assert_eq!(
                manual.start_rejection(now).is_none(),
                posture
                    == Posture::Fallen {
                        ready_for_standup: true
                    }
            );
            manual.eligible = false;
            assert!(manual.start_rejection(now).is_some());
            manual.eligible = true;
            manual.update_safe_pose(false, time(1001), time(1001));
            assert!(manual.start_rejection(time(1001)).is_some());
        }
    }

    #[test]
    fn both_detector_timestamps_must_remain_fresh() {
        let now = time(1000);
        for (sensor, publication, valid) in [
            (1000, 1000, true),
            (900, 900, true),
            (899, 1000, false),
            (1000, 899, false),
            (1001, 1000, false),
            (1000, 1001, false),
        ] {
            let mut manual = armed(now);
            manual.update_fall(
                FallDetection {
                    time: time(sensor),
                    posture: Posture::Fallen {
                        ready_for_standup: true,
                    },
                },
                time(publication),
                now,
            );
            assert_eq!(manual.start_rejection(now).is_none(), valid);
        }
        let mut manual = armed(now);
        manual.update_fall(
            FallDetection {
                time: time(900),
                posture: Posture::Fallen {
                    ready_for_standup: true,
                },
            },
            now,
            now,
        );
        assert!(manual.start_rejection(now).is_none());
        assert!(manual.start_rejection(time(1001)).is_some());
        manual.started = Some(now);
        assert!(manual.must_abort(time(1001)));
    }

    #[test]
    fn active_attempt_accepts_only_safe_detector_postures() {
        let now = time(1000);
        for (posture, abort) in [
            (Posture::StandingUp, false),
            (Posture::Upright, false),
            (
                Posture::Fallen {
                    ready_for_standup: true,
                },
                false,
            ),
            (
                Posture::Fallen {
                    ready_for_standup: false,
                },
                true,
            ),
            (Posture::Falling, true),
        ] {
            let mut manual = armed(now);
            manual.started = Some(time(950));
            manual.update_fall(FallDetection { time: now, posture }, now, now);
            assert_eq!(manual.must_abort(now), abort);
            manual.fall.invalidate();
            assert!(manual.must_abort(now));
        }
    }

    #[test]
    fn completion_requires_both_timestamps_after_start_and_upright_imu() {
        let now = time(1010);
        for (sensor, publication, posture, complete) in [
            (1001, 1002, Posture::Upright, true),
            (1000, 1002, Posture::Upright, false),
            (1001, 1000, Posture::Upright, false),
            (1001, 1002, Posture::StandingUp, false),
            (909, 1010, Posture::Upright, false),
            (1011, 1010, Posture::Upright, false),
        ] {
            let mut manual = armed(now);
            manual.started = Some(time(1000));
            manual.low_state.latest = Some((now, true));
            manual.update_fall(
                FallDetection {
                    time: time(sensor),
                    posture,
                },
                time(publication),
                now,
            );
            assert_eq!(manual.is_complete(now), complete);
            manual.low_state.latest = Some((now, false));
            assert!(!manual.is_complete(now));
            manual.low_state.latest = Some((time(900), true));
            assert!(!manual.is_complete(now));
        }
    }

    #[test]
    fn rejected_publications_do_not_poison_detector_recovery() {
        let now = time(1000);
        let fall = FallDetection {
            time: now,
            posture: Posture::Fallen {
                ready_for_standup: true,
            },
        };
        let mut manual = armed(now);
        manual.update_fall(fall, now, now);
        manual.update_fall(fall, now, now);
        assert!(manual.start_rejection(now).is_some());
        manual.update_fall(fall, time(2000), now);
        assert!(manual.start_rejection(now).is_some());
        manual.update_fall(fall, time(1001), time(1001));
        assert!(manual.start_rejection(time(1001)).is_none());
    }
}
