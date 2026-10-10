use std::sync::Arc;
use std::time::Duration;
use std::{boxed::Box, future::Future, pin::Pin};

use approx::AbsDiffEq;
use color_eyre::Result;
use serde::{Deserialize, Serialize};

use booster::{ImuState, JointsMotorState, LowState, MotorState};
use coordinate_systems::Robot;
use kinematics::joints::{Joints, JointsName, arm::ArmJoint, head::HeadJoint, leg::LegJoint};
use linear_algebra::Vector3;
use ros_z::prelude::*;
use ros_z::time::Time;

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub prep_mode_serial_motor_states: Joints<MotorState>,
    pub prep_mode_imu_state: ImuState,
    pub joint_position_threshold: f32,
    pub joint_velocity_threshold: f32,
    pub angular_velocity_threshold: f32,
    pub linear_acceleration_threshold: f32,
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("safe_pose_checker").build().await?;

    let parameters = node.bind_parameter_as::<Parameters>("safe_pose_checker")?;
    let imu_state_sub = node
        .subscriber::<ImuState>("inputs/imu_state")
        .build()
        .await?;
    let serial_motor_states_cache = node
        .subscriber::<Joints<MotorState>>("inputs/serial_motor_states")
        .cache(10)
        .build()
        .await?;
    let joint_position_difference_to_safe_pub = node
        .publisher::<Joints>("joint_position_difference_to_safe")
        .build()
        .await?;
    let joint_velocities_difference_to_safe_pub = node
        .publisher::<Joints>("joint_velocities_difference_to_safe")
        .build()
        .await?;
    let angular_velocities_difference_to_safe_pub = node
        .publisher::<Vector3<Robot>>("angular_velocities_difference_to_safe")
        .build()
        .await?;
    let linear_accelerations_difference_to_safe_pub = node
        .publisher::<Vector3<Robot>>("linear_accelerations_difference_to_safe")
        .build()
        .await?;
    let is_safe_pose_pub = node.publisher::<bool>("is_safe_pose").build().await?;
    let low_state_sub = node
        .subscriber::<LowState>("inputs/low_state")
        .build()
        .await?;
    let manual_is_safe_pose_pub = node
        .publisher::<bool>("manual_stand_up/is_safe_pose")
        .build()
        .await?;
    let mut manual_pose_dwell = ManualPoseDwell::default();
    loop {
        let parameters_snapshot = parameters.snapshot();
        let parameters = parameters_snapshot.typed();

        let received_imu_state = tokio::select! {
            received = imu_state_sub.recv_with_metadata() => received?,
            received = low_state_sub.recv_with_metadata() => {
                let received = received?;
                let sample_is_safe = manual_low_state_is_safe(
                    &received,
                    &parameters.prep_mode_serial_motor_states,
                    parameters.angular_velocity_threshold,
                );
                let is_safe_pose = manual_pose_dwell.update(
                    received.source_time,
                    node.clock().now(),
                    sample_is_safe,
                );
                manual_is_safe_pose_pub
                    .publish_with_source_time(&is_safe_pose, received.source_time)
                    .await?;
                continue;
            }
        };

        let Some(serial_motor_states) =
            serial_motor_states_cache.get_nearest(received_imu_state.source_time)
        else {
            continue;
        };
        let imu_state = received_imu_state.into_message();

        let joint_position_difference_to_safe =
            serial_motor_states.positions() - parameters.prep_mode_serial_motor_states.positions();
        joint_position_difference_to_safe_pub
            .publish(&joint_position_difference_to_safe)
            .await?;

        let joint_velocities_difference_to_safe = serial_motor_states.velocities()
            - parameters.prep_mode_serial_motor_states.velocities();
        joint_velocities_difference_to_safe_pub
            .publish(&joint_velocities_difference_to_safe)
            .await?;

        let linear_accelerations_difference_to_safe =
            imu_state.linear_acceleration - parameters.prep_mode_imu_state.linear_acceleration;
        linear_accelerations_difference_to_safe_pub
            .publish(&linear_accelerations_difference_to_safe)
            .await?;

        let angular_velocities_difference_to_safe =
            imu_state.angular_velocity - parameters.prep_mode_imu_state.angular_velocity;
        angular_velocities_difference_to_safe_pub
            .publish(&angular_velocities_difference_to_safe)
            .await?;

        let motor_states_are_safe = motor_states_are_safe(
            &serial_motor_states,
            &parameters.prep_mode_serial_motor_states,
            parameters.joint_position_threshold,
            parameters.joint_velocity_threshold,
        );

        let imu_state_is_safe = imu_state_is_safe(
            &imu_state,
            &parameters.prep_mode_imu_state,
            parameters.angular_velocity_threshold,
            parameters.linear_acceleration_threshold,
        );

        let is_safe_pose = motor_states_are_safe && imu_state_is_safe;

        is_safe_pose_pub.publish(&is_safe_pose).await?;
    }
}

const MANUAL_JOINT_POSITION_THRESHOLD: f32 = 0.30;
const MANUAL_JOINT_VELOCITY_THRESHOLD: f32 = 0.05;
const MANUAL_POSE_DWELL: Duration = Duration::from_millis(300);
const MANUAL_SENSOR_MAX_AGE: Duration = Duration::from_millis(100);

fn manual_low_state_is_safe(
    low_state: &LowState,
    prep_mode_serial_motor_states: &Joints<MotorState>,
    angular_velocity_threshold: f32,
) -> bool {
    low_state.serial_motor_states().is_ok_and(|motor_states| {
        manual_motor_states_are_safe(&motor_states, prep_mode_serial_motor_states)
            && manual_imu_state_is_safe(&low_state.imu_state, angular_velocity_threshold)
    })
}

fn manual_joint_position_threshold(joint: JointsName) -> f32 {
    match joint {
        JointsName::Head(HeadJoint::Pitch) => 0.45,
        JointsName::LeftArm(ArmJoint::ShoulderPitch)
        | JointsName::RightArm(ArmJoint::ShoulderPitch) => 0.675,
        JointsName::LeftArm(ArmJoint::Elbow) | JointsName::RightArm(ArmJoint::Elbow) => 1.275,
        JointsName::LeftLeg(LegJoint::Knee) | JointsName::RightLeg(LegJoint::Knee) => 0.375,
        _ => MANUAL_JOINT_POSITION_THRESHOLD,
    }
}

fn manual_motor_states_are_safe(
    serial_motor_states: &Joints<MotorState>,
    prep_mode_serial_motor_states: &Joints<MotorState>,
) -> bool {
    serial_motor_states
        .enumerate()
        .zip(*prep_mode_serial_motor_states)
        .all(|((name, current), prepare)| {
            current.lost == 0
                && current.position.is_finite()
                && prepare.position.is_finite()
                && (current.position - prepare.position).abs()
                    <= manual_joint_position_threshold(name)
                && current.velocity.is_finite()
                && current.velocity.abs() <= MANUAL_JOINT_VELOCITY_THRESHOLD
        })
}

fn manual_imu_state_is_safe(imu_state: &ImuState, angular_velocity_threshold: f32) -> bool {
    imu_state.roll_pitch_yaw.inner.iter().all(|v| v.is_finite())
        && imu_state.roll_pitch_yaw.y().sin().abs() >= 20_f32.to_radians().cos()
        && angular_velocity_threshold.is_finite()
        && imu_state
            .angular_velocity
            .inner
            .iter()
            .all(|v| v.is_finite() && v.abs() <= angular_velocity_threshold)
}

#[derive(Default)]
struct ManualPoseDwell {
    safe_since: Option<Time>,
    previous_source_time: Option<Time>,
}

impl ManualPoseDwell {
    fn update(&mut self, source_time: Time, now: Time, sample_is_safe: bool) -> bool {
        let previous = self.previous_source_time.replace(source_time);
        if source_time > now
            || now.duration_since(source_time) > MANUAL_SENSOR_MAX_AGE
            || previous.is_some_and(|previous| source_time <= previous)
            || !sample_is_safe
        {
            self.safe_since = None;
            return false;
        }
        if previous
            .is_some_and(|previous| source_time.duration_since(previous) > MANUAL_SENSOR_MAX_AGE)
        {
            self.safe_since = None;
        }
        let safe_since = *self.safe_since.get_or_insert(source_time);
        source_time.duration_since(safe_since) >= MANUAL_POSE_DWELL
    }
}

fn motor_states_are_safe(
    serial_motor_states: &Joints<MotorState>,
    prep_mode_serial_motor_states: &Joints<MotorState>,
    joint_position_threshold: f32,
    joint_velocity_threshold: f32,
) -> bool {
    serial_motor_states
        .into_iter()
        .zip(*prep_mode_serial_motor_states)
        .all(|(current_motor_state, safe_motor_state)| {
            current_motor_state
                .position
                .abs_diff_eq(&safe_motor_state.position, joint_position_threshold)
                && current_motor_state
                    .velocity
                    .abs_diff_eq(&safe_motor_state.velocity, joint_velocity_threshold)
        })
}

fn imu_state_is_safe(
    imu_state: &ImuState,
    prep_mode_imu_state: &ImuState,
    angular_velocity_threshold: f32,
    linear_acceleration_threshold: f32,
) -> bool {
    imu_state.angular_velocity.abs_diff_eq(
        &prep_mode_imu_state.angular_velocity,
        angular_velocity_threshold,
    ) && imu_state.linear_acceleration.abs_diff_eq(
        &prep_mode_imu_state.linear_acceleration,
        linear_acceleration_threshold,
    )
}
