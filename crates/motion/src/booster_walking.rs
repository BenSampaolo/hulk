use std::time::{Duration, SystemTime};

use booster::{ImuState, MotorState};
use color_eyre::{eyre::bail, Result};
use context_attribute::context;
use framework::MainOutput;
use hardware::{PathsInterface, TimeInterface};
use serde::{Deserialize, Serialize};
use types::{
    cycle_time::CycleTime,
    joints::Joints,
    motion_command::MotionCommand,
    parameters::{MotorCommandParameters, RLWalkingParameters},
};
use walking_inference::inference::WalkingInference;

#[derive(Deserialize, Serialize)]
pub struct RLWalking {
    walking_inference: WalkingInference,
    smoothed_target_joint_positions: Joints,
    next_inference_time: SystemTime,
}

#[context]
pub struct CreationContext {
    prepare_motor_command_parameters: Parameter<MotorCommandParameters, "prepare_motor_command">,
    walking_parameters: Parameter<RLWalkingParameters, "rl_walking">,

    hardware_interface: HardwareInterface,
}

#[context]
pub struct CycleContext {
    walking_parameters: Parameter<RLWalkingParameters, "rl_walking">,
    common_motor_command_parameters: Parameter<MotorCommandParameters, "common_motor_command">,

    imu_state: Input<ImuState, "imu_state">,
    serial_motor_states: Input<Joints<MotorState>, "serial_motor_states">,
    motion_command: Input<MotionCommand, "selected_motion_command">,
    cycle_time: Input<CycleTime, "cycle_time">,
}

#[context]
#[derive(Default)]
pub struct MainOutputs {
    pub target_joint_positions: MainOutput<Joints>,
}

impl RLWalking {
    pub fn new(context: CreationContext<impl PathsInterface + TimeInterface>) -> Result<Self> {
        let paths = context.hardware_interface.get_paths();
        let neural_network_folder = paths.neural_networks;

        let walking_inference = WalkingInference::new(
            &neural_network_folder,
            context.prepare_motor_command_parameters,
            context.walking_parameters.observation_history_length,
        )?;

        Ok(Self {
            walking_inference,
            smoothed_target_joint_positions: context
                .prepare_motor_command_parameters
                .default_positions,
            next_inference_time: context.hardware_interface.get_now(),
        })
    }

    pub fn cycle(&mut self, context: CycleContext) -> Result<MainOutputs> {
        let MotionCommand::WalkWithVelocity {
            velocity,
            angular_velocity,
            ..
        } = context.motion_command
        else {
            bail!("only MotionCommand::WalkWithVelocity is supported for walking inference");
        };

        if context.cycle_time.start_time < self.next_inference_time {
            return Ok(MainOutputs {
                target_joint_positions: self.smoothed_target_joint_positions.into(),
            });
        }

        self.next_inference_time = context.cycle_time.start_time
            + Duration::from_secs_f32(
                context.walking_parameters.control.dt
                    * context.walking_parameters.control.decimation,
            );

        let scaled_inference_output_positions = self.walking_inference.do_inference(
            context.cycle_time.last_cycle_duration,
            *velocity,
            *angular_velocity,
            context.imu_state,
            *context.serial_motor_states,
            context.walking_parameters,
            context.common_motor_command_parameters,
        )?;

        let target_joint_positions = context.common_motor_command_parameters.default_positions
            + scaled_inference_output_positions;

        self.smoothed_target_joint_positions = self.smoothed_target_joint_positions
            * context.walking_parameters.joint_position_smoothing_factor
            + target_joint_positions
                * (1.0 - context.walking_parameters.joint_position_smoothing_factor);

        Ok(MainOutputs {
            target_joint_positions: self.smoothed_target_joint_positions.into(),
        })
    }
}
