use linear_algebra::{Orientation2, Pose2, vector};
use types::{
    behavior_tree::Status,
    motion_command::{BodyMotion, HeadMotion, ImageRegion, MotionCommand, OrientationMode},
};

use crate::{node::Blackboard, walk::walk_to};

pub fn has_suggested_search_position(blackboard: &mut Blackboard) -> bool {
    blackboard.world_state.suggested_search_position.is_some()
}

pub fn leuchtturm(blackboard: &mut Blackboard) -> Status {
    let angular_velocity = get_leuchtturm_direction(blackboard);

    blackboard.body_motion = Some(BodyMotion::WalkWithVelocity {
        velocity: vector!(0.0, 0.0),
        angular_velocity,
    });
    Status::Success
}

fn get_leuchtturm_direction(blackboard: &Blackboard) -> f32 {
    if let MotionCommand::WalkWithVelocity {
        angular_velocity, ..
    } = blackboard.last_motion_command
        && angular_velocity.abs() > f32::EPSILON
    {
        return angular_velocity.signum();
    }

    if let (Some(last_ball), Some(ground_to_field)) = (
        &blackboard.last_ball,
        blackboard.world_state.robot.ground_to_field,
    ) {
        let ball_in_ground = ground_to_field.inverse() * last_ball.position;

        if ball_in_ground.y() < 0.0 {
            return -1.0;
        }
    }

    1.0
}

pub fn walk_to_search_position(blackboard: &mut Blackboard) -> Status {
    if let (Some(search_position), Some(ground_to_field)) = (
        blackboard.world_state.suggested_search_position,
        blackboard.world_state.robot.ground_to_field,
    ) {
        let search_position_in_ground = ground_to_field.inverse() * search_position;
        let look_at_in_ground = blackboard
            .world_state
            .suggested_search_look_at
            .map(|look_at| ground_to_field.inverse() * look_at);
        let target_pose = look_at_in_ground
            .and_then(|target| {
                let final_view_direction = target - search_position_in_ground;
                (final_view_direction.norm() > f32::EPSILON).then(|| {
                    Pose2::from_parts(
                        search_position_in_ground,
                        Orientation2::from_vector(final_view_direction),
                    )
                })
            })
            .unwrap_or_else(|| Pose2::from(search_position_in_ground));
        let orientation_mode = look_at_in_ground
            .map(|target| OrientationMode::LookAt {
                target,
                tolerance: blackboard.parameters.walk_and_stand.orientation_tolerance,
            })
            .unwrap_or(OrientationMode::AlignWithPath);
        if let Some(target) = look_at_in_ground {
            blackboard.head_motion = Some(HeadMotion::LookAt {
                target,
                image_region_target: ImageRegion::Center,
            });
        }

        return walk_to(
            blackboard,
            target_pose,
            blackboard.parameters.walk_speed.search,
            orientation_mode,
            blackboard
                .parameters
                .walk_and_stand
                .normal_distance_to_be_aligned,
            blackboard.parameters.walk_and_stand.hysteresis,
        );
    }

    Status::Failure
}
