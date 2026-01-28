use booster::{CommandType, LowCommand};
use color_eyre::{eyre::WrapErr, Result};
use context_attribute::context;
use framework::AdditionalOutput;
use hardware::{LowCommandInterface, TimeInterface};
use serde::{Deserialize, Serialize};
use types::{
    joints::{head::HeadJoints, Joints},
    motion_command::{HeadMotion, MotionCommand},
    parameters::MotorCommandParameters,
};

#[derive(Deserialize, Serialize)]
pub struct CommandSender {
    time_index: f32,
    motor_index: usize,
    filtered_target_joint_positions: Joints,
}

#[context]
pub struct CreationContext {
    prepare_motor_command_parameters: Parameter<MotorCommandParameters, "prepare_motor_command">,
}

#[context]
pub struct CycleContext {
    low_command: AdditionalOutput<LowCommand, "low_command">,

    look_at: Input<HeadJoints<f32>, "look_at">,
    motion_command: Input<MotionCommand, "selected_motion_command">,

    walk_motor_command_parameters: Parameter<MotorCommandParameters, "common_motor_command">,
    _prepare_motor_command_parameters: Parameter<MotorCommandParameters, "prepare_motor_command">,

    hardware_interface: HardwareInterface,
    // look_at: Input<HeadJoints<f32>, "look_at">,
    // motion_command: Input<MotionCommand, "selected_motion_command">,
    // head_joints_command: Input<HeadJoints<f32>, "head_joints_command">,
    combined_joint_positions: Input<MotorCommands<Joints<f32>>, "combined_joint_positions">,
}

#[context]
#[derive(Default)]
pub struct MainOutputs {}

impl CommandSender {
    pub fn new(context: CreationContext) -> Result<Self> {
        Ok(Self {
            time_index: 0.0,
            motor_index: 0,
            filtered_target_joint_positions: context
                .prepare_motor_command_parameters
                .default_positions,
        })
    }

    pub fn cycle(
        &mut self,
        mut context: CycleContext<impl LowCommandInterface + TimeInterface>,
    ) -> Result<MainOutputs> {
        let look_at_head_joints = match context.motion_command.head_motion() {
            Some(HeadMotion::LookAt { .. }) => Some(*context.look_at),
            _ => None,
        };
        let walk_low_command = LowCommand::new(
            &context.combined_joint_positions.positions.,
            &context.combined_joint_positions.stiffnesses,
            context.walk_motor_command_parameters,
            CommandType::Serial,
        );

        context
            .hardware_interface
            .write_low_command(walk_low_command.clone())
            .wrap_err("failed to write to actuators")?;

        context
            .low_command
            .fill_if_subscribed(|| walk_low_command.clone());

        Ok(MainOutputs {})
    }
}
