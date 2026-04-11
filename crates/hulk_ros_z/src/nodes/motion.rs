use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};

use booster_sdk::types::RobotMode;
use color_eyre::Result;
use ros_z::{
    Builder, ExtendedMessageTypeInfo, MessageTypeInfo,
    context::ZContext,
    dynamic::{EnumPayloadSchema, EnumSchema, EnumVariantSchema, FieldSchema, FieldType},
    msg::{SerdeCdrSerdes, ZMessage},
};
use ros_z_config::prelude::*;
use serde::{Deserialize, Serialize};
use types::motion_command::{HeadMotion, ImageRegion, MotionCommand};
use types::motion_runtime::MotionRuntime;

use crate::{IntoEyreResultExt, config::MotionConfig};

pub async fn run(ctx: Arc<ZContext>) -> Result<()> {
    let node = ctx
        .create_node("motion")
        .with_type_description_service()
        .with_extended_type_description_service()
        .build()
        .into_eyre()?;
    let config = node
        .bind_config_with_metadata_as::<MotionConfig>("motion")
        .into_eyre()?;
    config
        .add_validation_hook(|cfg: &MotionConfig| {
            if !cfg.timing.publish_hz.is_finite() || cfg.timing.publish_hz <= 0.0 {
                return Err("motion.timing.publish_hz must be > 0".to_owned());
            }
            if cfg.limits.max_forward < 0.0 || !cfg.limits.max_forward.is_finite() {
                return Err("motion.limits.max_forward must be finite and >= 0".to_owned());
            }
            if cfg.limits.max_lateral < 0.0 || !cfg.limits.max_lateral.is_finite() {
                return Err("motion.limits.max_lateral must be finite and >= 0".to_owned());
            }
            if cfg.limits.max_angular < 0.0 || !cfg.limits.max_angular.is_finite() {
                return Err("motion.limits.max_angular must be finite and >= 0".to_owned());
            }
            if cfg.output.mode != "log_only" && cfg.output.mode != "publish" {
                return Err("motion.output.mode must be one of: log_only, publish".to_owned());
            }
            Ok(())
        })
        .into_eyre()?;

    let motion_command_sub = node
        .create_sub::<MotionCommand>("behavior/motion_command")
        .build()
        .into_eyre()?;

    let robot_mode_sub = node
        .create_sub::<RobotModeMsg>("robot_hw/robot_mode")
        .build()
        .into_eyre()?;

    let command_pub = node
        .create_pub::<HighLevelCommand>("robot_hw/high_level_command")
        .build()
        .into_eyre()?;

    let mut last_motion_command = MotionCommand::Stand {
        head: HeadMotion::Center {
            image_region_target: (ImageRegion::Top),
        },
    };

    let mut latest_command = MotionCommand::Stand {
        head: HeadMotion::Center {
            image_region_target: (ImageRegion::Top),
        },
    };

    let mut latest_robot_mode = RobotMode::Unknown;

    let mut timer = node.clock().timer(Duration::from_secs_f64(1.0));

    let mut last_mode_switch_time = SystemTime::now();

    let mut current_mode = LookAroundMode::Left;

    let runtime = MotionRuntime::Booster;

    loop {
        let cfg = config.snapshot().typed().clone();
        let publish_hz = cfg.timing.publish_hz.max(1.0);
        timer.set_period(Duration::from_secs_f64(1.0 / publish_hz));

        tokio::select! {
            msg = motion_command_sub.async_recv() => {
                latest_command = msg.into_eyre()?;
            }
            msg = robot_mode_sub.async_recv() => {
                latest_robot_mode = msg.into_eyre()?.mode;
            }
            _ = timer.tick() => {
                if latest_robot_mode == RobotMode::Walking && runtime == MotionRuntime::Booster {
                    match latest_command {
                        MotionCommand::WalkWithVelocity {
                            velocity,
                            angular_velocity,
                            ..
                        } => {
                            command_pub.publish(
                                &HighLevelCommand::MoveRobot {
                                    forward: velocity.x().clamp(-cfg.limits.max_forward, cfg.limits.max_forward),
                                    left: velocity.y().clamp(-cfg.limits.max_lateral, cfg.limits.max_lateral),
                                    turn: angular_velocity.clamp(-cfg.limits.max_angular, cfg.limits.max_angular),
                                }
                            ).into_eyre()?;
                        }
                        MotionCommand::Stand { .. } => {
                            command_pub.publish(
                                &HighLevelCommand::MoveRobot {
                                    forward: 0.0,
                                    left: 0.0,
                                    turn: 0.0,
                                }
                            ).into_eyre()?;
                        },
                        MotionCommand::StandUp => {
                            if !matches!(last_motion_command, MotionCommand::StandUp) {
                                command_pub.publish(&HighLevelCommand::GetUp).into_eyre()?;
                            }
                        },
                        _ => (),
                    };
                }

                let head_motion = match latest_command {
                    MotionCommand::Stand { head, .. } => head,
                    MotionCommand::Walk { head, .. } => head,
                    MotionCommand::WalkWithVelocity { head, .. } => head,
                    _ => types::motion_command::HeadMotion::Center {
                        image_region_target: ImageRegion::Center,
                    },
                };

                let head_angles = compute_head_angles(&head_motion, &mut last_mode_switch_time, &Duration::from_secs_f32(1.0), &mut current_mode);

                command_pub.publish(
                    &HighLevelCommand::RotateHead {
                        pitch: head_angles.0,
                        yaw: head_angles.1,
                    }
                ).into_eyre()?;

                last_motion_command = latest_command.clone();
            }
            // -- Messages have their own select branch where order is unbiassed --
            msg_result = async {
                tokio::select! {
                    msg = motion_command_sub.async_recv() => {
                        latest_command = msg.into_eyre()?;
                    }
                    msg = robot_mode_sub.async_recv() => {
                        latest_robot_mode = msg.into_eyre()?.mode;
                    }
                }
                Ok::<(), color_eyre::Report>(())
            } => {
                msg_result?;
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct RobotModeMsg {
    pub mode: RobotMode,
}

impl MessageTypeInfo for RobotModeMsg {
    fn type_name() -> &'static str {
        "hulk_ros_z/msg/RobotMode"
    }

    fn type_hash() -> ros_z::TypeHash {
        ros_z::TypeHash::zero()
    }
}

impl ZMessage for RobotModeMsg {
    type Serdes = SerdeCdrSerdes<Self>;
}

impl ExtendedMessageTypeInfo for RobotModeMsg {
    fn extended_message_schema() -> Arc<ros_z::dynamic::MessageSchema> {
        Arc::new(ros_z::dynamic::MessageSchema {
            type_name: "hulk_ros_z/msg/RobotMode".to_string(),
            fields: vec![FieldSchema::new(
                "mode".to_string(),
                FieldType::Enum(Arc::new(EnumSchema {
                    type_name: "RobotMode".to_string(),
                    variants: vec![
                        EnumVariantSchema::new("Unknown".to_string(), EnumPayloadSchema::Unit),
                        EnumVariantSchema::new("Idle".to_string(), EnumPayloadSchema::Unit),
                        EnumVariantSchema::new("Walking".to_string(), EnumPayloadSchema::Unit),
                        EnumVariantSchema::new("Falling".to_string(), EnumPayloadSchema::Unit),
                        EnumVariantSchema::new("GettingUp".to_string(), EnumPayloadSchema::Unit),
                        EnumVariantSchema::new(
                            "SpecialAction".to_string(),
                            EnumPayloadSchema::Unit,
                        ),
                    ],
                })),
            )],
            package: "hulk_ros_z".to_string(),
            name: "RobotMode".to_string(),
            type_hash: Default::default(),
        })
    }

    fn extended_field_type() -> ros_z::dynamic::FieldType {
        FieldType::Enum(Arc::new(EnumSchema {
            type_name: "RobotMode".to_string(),
            variants: vec![
                EnumVariantSchema::new("Unknown".to_string(), EnumPayloadSchema::Unit),
                EnumVariantSchema::new("Idle".to_string(), EnumPayloadSchema::Unit),
                EnumVariantSchema::new("Walking".to_string(), EnumPayloadSchema::Unit),
                EnumVariantSchema::new("Falling".to_string(), EnumPayloadSchema::Unit),
                EnumVariantSchema::new("GettingUp".to_string(), EnumPayloadSchema::Unit),
                EnumVariantSchema::new("SpecialAction".to_string(), EnumPayloadSchema::Unit),
            ],
        }))
    }
}

#[derive(Serialize, Deserialize, ExtendedMessageTypeInfo)]
#[ros_msg(type_name = "hulk_ros_z/msg/HighLevelCommand")]
pub enum HighLevelCommand {
    ChangeMode { mode: i32 },
    MoveRobot { forward: f32, left: f32, turn: f32 },
    RotateHead { pitch: f32, yaw: f32 },
    RotateHeadWithDirection { pitch: i32, yaw: i32 },
    LieDown,
    GetUp,
    GetUpWithMode { mode: i32 },
    EnterWbcGait,
    ExitWbcGait,
    VisualKick { start: bool },
    ResetOdometer,
}
impl ZMessage for HighLevelCommand {
    type Serdes = SerdeCdrSerdes<Self>;
}

// if context.hardware_interface.get_motion_runtime_type()? != MotionRuntime::Booster
//             || !matches!(context.robot_mode, RobotMode::Walking)
//         {
//             return Ok(MainOutputs {});
//         }

fn compute_head_angles(
    head_motion: &HeadMotion,
    last_mode_switch: &mut SystemTime,
    look_around_duration: &Duration,
    last_look_around_mode: &mut LookAroundMode,
) -> (f32, f32) {
    match head_motion {
        HeadMotion::Center {
            image_region_target: ImageRegion::Top,
        } => (0.0, 0.0),
        HeadMotion::Center { .. } => (0.4, 0.0),
        HeadMotion::LookAround | HeadMotion::SearchForLostBall => look_around(
            last_mode_switch,
            look_around_duration,
            last_look_around_mode,
        ),
        _ => Default::default(),
    }
}

enum LookAroundMode {
    Left,
    Right,
}

fn look_around(
    last_mode_switch: &mut SystemTime,
    look_around_duration: &Duration,
    last_look_around_mode: &mut LookAroundMode,
) -> (f32, f32) {
    let now = SystemTime::now();
    if now.duration_since(*last_mode_switch).unwrap() > *look_around_duration {
        *last_mode_switch = now;

        *last_look_around_mode = match last_look_around_mode {
            LookAroundMode::Left => LookAroundMode::Right,
            LookAroundMode::Right => LookAroundMode::Left,
        };
    }
    match last_look_around_mode {
        LookAroundMode::Left => (0.4, 0.5),
        LookAroundMode::Right => (0.4, -0.5),
    }
}
