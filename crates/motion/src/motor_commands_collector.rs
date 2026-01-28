use booster::{JointsMotorState, MotorState};
use color_eyre::Result;
use context_attribute::context;
use energy_optimization::current_minimizer::CurrentMinimizer;
use framework::MainOutput;
use serde::{Deserialize, Serialize};
use types::{
    joints::{arm::ArmJoints, body::BodyJoints, head::HeadJoints, leg::LegJoints, Joints},
    motion_command::MotionCommand,
    motor_commands::MotorCommands,
};

#[derive(Deserialize, Serialize)]
pub struct MotorCommandCollector {
    current_minimizer: CurrentMinimizer,
}

#[context]
pub struct CreationContext {}

#[context]
pub struct CycleContext {
    head_joints_command: Input<MotorCommands<HeadJoints<f32>>, "head_joints_command">,
    motion_commmand: Input<MotionCommand, "selected_motion_command">,
    serial_motor_states: Input<Joints<MotorState>, "serial_motor_states">,
    walk_motor_commands: Input<MotorCommands<Joints>, "target_joint_positions">,
    default_motion_stiffness_upper_body: Parameter<f32, "default_motion_stiffness_upper_body">,
}

#[context]
#[derive(Default)]
pub struct MainOutputs {
    pub combined_joint_positions: MainOutput<MotorCommands<Joints<f32>>>,
}

impl MotorCommandCollector {
    pub fn new(_context: CreationContext) -> Result<Self> {
        Ok(Self {
            current_minimizer: CurrentMinimizer::default(),
        })
    }

    pub fn cycle(&mut self, context: CycleContext) -> Result<MainOutputs> {
        let measured_positions = context.serial_motor_states.positions();
        let head_joints_command = *context.head_joints_command;
        let motion_command = context.motion_commmand;
        let walk = *context.walk_motor_commands;

        let (positions, stiffnesses) = match motion_command {
            MotionCommand::Stand { .. } => (
                Joints::from_head_and_body(head_joints_command.positions, walk.positions.body()),
                Joints::from_head_and_body(
                    head_joints_command.stiffnesses,
                    walk.stiffnesses.body(),
                ),
            ),
            MotionCommand::Unstiff => (measured_positions, Joints::fill(0.0)),
            MotionCommand::WalkWithVelocity { .. } => (
                Joints::from_head_and_body(head_joints_command.positions, walk.positions.body()),
                Joints::from_head_and_body(
                    head_joints_command.stiffnesses,
                    walk.stiffnesses.body(),
                ),
            ),
            _ => (
                Joints::from_head_and_body(head_joints_command.positions, walk.positions.body()),
                Joints::from_head_and_body(
                    head_joints_command.stiffnesses,
                    walk.stiffnesses.body(),
                ),
            ),
        };

        let combined_joint_positions = MotorCommands {
            positions,
            stiffnesses,
        };
        Ok(MainOutputs {
            combined_joint_positions: combined_joint_positions.into(),
        })
    }
}

fn default_motion_stiffness(context: &CycleContext<'_>) -> Joints {
    Joints::from_head_and_body(
        HeadJoints::fill(*context.default_motion_stiffness_upper_body),
        BodyJoints {
            left_arm: ArmJoints::fill(*context.default_motion_stiffness_upper_body),
            right_arm: ArmJoints::fill(*context.default_motion_stiffness_upper_body),
            left_leg: LegJoints::fill(1.0),
            right_leg: LegJoints::fill(1.0),
        },
    )
}
