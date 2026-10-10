use std::{boxed::Box, future::Future, pin::Pin};
use std::{collections::HashSet, sync::Arc, time::Duration};

mod manual_stand_up;

use booster::LowState;
use manual_stand_up::ManualStandUp;

use color_eyre::Result;
use serde::{Deserialize, Serialize};

use hsl_network_messages::{GamePhase, PlayerNumber};
use ros_z::{prelude::*, qos::QosDurability, time::Time};
use tracing::{info, warn};
use types::{
    buttons::{ButtonPressType, Buttons},
    controller_input::{Button, ControllerInput},
    fall_detection::FallDetection,
    filtered_game_controller_state::FilteredGameControllerState,
    filtered_game_state::FilteredGameState,
    primary_state::PrimaryState,
};

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub injected_primary_state: Option<PrimaryState>,
    pub recorded_primary_states: HashSet<PrimaryState>,
}

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("primary_state_filter").build().await?;

    let parameters = node.bind_parameter_as::<Parameters>("primary_state_filter")?;
    let player_number_cache = node
        .subscriber::<PlayerNumber>("player_number")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .cache(1)
        .build()
        .await?;

    let filtered_game_controller_state_sub = node
        .subscriber::<FilteredGameControllerState>("filtered_game_controller_state")
        .build()
        .await?;
    let buttons_sub = node
        .subscriber::<Buttons<Option<ButtonPressType>>>("buttons")
        .build()
        .await?;
    let emergency_stop_sub = node
        .subscriber::<()>("motion/emergency_stop")
        .build()
        .await?;
    let is_safe_pose_cache = node
        .subscriber::<bool>("is_safe_pose")
        .cache(1)
        .build()
        .await?;
    let controller_input_sub = node
        .subscriber::<ControllerInput>("inputs/controller_input")
        .build()
        .await?;

    let low_state_sub = node
        .subscriber::<LowState>("inputs/low_state")
        .build()
        .await?;
    let manual_safe_pose_sub = node
        .subscriber::<bool>("manual_stand_up/is_safe_pose")
        .build()
        .await?;
    let fall_detection_sub = node
        .subscriber::<FallDetection>("fall_detection/status")
        .build()
        .await?;

    let primary_state_pub = node
        .publisher::<PrimaryState>("primary_state")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;

    let mut primary_state_filter = PrimaryStateFilter::default();
    primary_state_pub
        .publish(&primary_state_filter.primary_state)
        .await?;

    let mut watchdog = tokio::time::interval(Duration::from_millis(25));
    watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_published = node.clock().now();
    let mut previous_injected_primary_state = None;

    loop {
        let previous_state = primary_state_filter.primary_state;
        let mut publish = true;
        let mut watchdog_tick = false;
        let parameters_snapshot = parameters.snapshot();
        let parameters = parameters_snapshot.typed();
        tokio::select! {
            biased;
            _ = emergency_stop_sub.recv() => {
                warn!("received emergency stop signal, entering damping...");
                primary_state_filter.request_damping(node.clock().now());
            }
            received_buttons = buttons_sub.recv() => {
                let buttons = received_buttons?;
                let safe_pose = is_safe_pose_cache.get_latest();
                if matches!(buttons.f1, Some(ButtonPressType::Short)) || safe_pose.is_some() {
                    let is_safe_pose = safe_pose.is_some_and(|safe| *safe);
                    let previous_primary_state = primary_state_filter.primary_state;
                    if buttons.f1.is_some() || buttons.stand.is_some() || buttons.walking.is_some() {
                        info!(
                            target: "primary_state_filter::input",
                            ?buttons,
                            is_safe_pose,
                            previous_primary_state = ?previous_primary_state,
                            "received button input"
                        );
                    }
                    primary_state_filter.update_with_buttons(&buttons, is_safe_pose, node.clock().now());
                    if primary_state_filter.primary_state != previous_primary_state {
                        info!(
                            target: "primary_state_filter::input",
                            ?buttons,
                            is_safe_pose,
                            previous_primary_state = ?previous_primary_state,
                            primary_state = ?primary_state_filter.primary_state,
                            "primary state changed from buttons"
                        );
                    }
                }
            }
            received_controller_input = controller_input_sub.recv_with_metadata() => {
                match received_controller_input {
                    Ok(input) => primary_state_filter.update_with_controller_input(
                        &input, input.source_time, node.clock().now(),
                    ),
                    Err(ros_z::Error::Wire(_)) => primary_state_filter.manual.invalidate_controller(),
                    Err(error) => return Err(error.into()),
                }
            }
            _ = watchdog.tick() => {
                publish = false;
                watchdog_tick = true;
            }
            received_filtered_game_controller_state = filtered_game_controller_state_sub.recv() => {
                let Some(player_number) = player_number_cache.get_latest() else {continue};

                let filtered_game_controller_state = received_filtered_game_controller_state?;

                let previous_primary_state = primary_state_filter.primary_state;
                primary_state_filter.update_with_filtered_game_controller_state(
                    &filtered_game_controller_state,
                    *player_number,
                );
                if primary_state_filter.primary_state != previous_primary_state {
                    info!(
                        target: "primary_state_filter::input",
                        previous_primary_state = ?previous_primary_state,
                        primary_state = ?primary_state_filter.primary_state,
                        game_state = ?filtered_game_controller_state.game_state,
                        player_number = ?*player_number,
                        penalty = ?filtered_game_controller_state.penalties[*player_number],
                        "primary state changed from game controller"
                    );
                }
            }
            received_low_state = low_state_sub.recv_with_metadata() => {
                publish = false;
                match received_low_state {
                    Ok(input) => primary_state_filter.manual.update_low_state(
                        &input, input.source_time, node.clock().now(),
                    ),
                    Err(ros_z::Error::Wire(_)) => primary_state_filter.manual.low_state.invalidate(),
                    Err(error) => return Err(error.into()),
                }
            }
            received_safe_pose = manual_safe_pose_sub.recv_with_metadata() => {
                publish = false;
                match received_safe_pose {
                    Ok(input) => primary_state_filter.manual.update_safe_pose(
                        *input, input.source_time, node.clock().now(),
                    ),
                    Err(ros_z::Error::Wire(_)) => primary_state_filter.manual.safe_pose.invalidate(),
                    Err(error) => return Err(error.into()),
                }
            }
            received_fall = fall_detection_sub.recv_with_metadata() => {
                publish = false;
                match received_fall {
                    Ok(input) => primary_state_filter.manual.update_fall(
                        *input, input.source_time, node.clock().now(),
                    ),
                    Err(ros_z::Error::Wire(_)) => primary_state_filter.manual.fall.invalidate(),
                    Err(error) => return Err(error.into()),
                }
            }
        }

        let injection_changed =
            parameters.injected_primary_state != previous_injected_primary_state;
        previous_injected_primary_state = parameters.injected_primary_state;
        if let Some(injected_primary_state) = parameters.injected_primary_state
            && (injection_changed
                || injected_primary_state != PrimaryState::Damping
                || primary_state_filter.primary_state != PrimaryState::Damping)
        {
            let previous_primary_state = primary_state_filter.primary_state;
            primary_state_filter
                .update_with_injected_primary_state(injected_primary_state, node.clock().now());
            if primary_state_filter.primary_state != previous_primary_state {
                info!(
                    target: "primary_state_filter::input",
                    previous_primary_state = ?previous_primary_state,
                    primary_state = ?primary_state_filter.primary_state,
                    "primary state changed from injected parameter"
                );
            }
        }

        let now = node.clock().now();
        primary_state_filter.watchdog(now);
        let heartbeat = watchdog_tick
            && primary_state_filter.primary_state == PrimaryState::ManualStandUp
            && (now < last_published
                || now.duration_since(last_published) >= Duration::from_millis(50));
        if publish || heartbeat || primary_state_filter.primary_state != previous_state {
            primary_state_pub
                .publish(&primary_state_filter.primary_state)
                .await?;
            last_published = now;
        }
    }
}

#[derive(Default)]
struct PrimaryStateFilter {
    pub primary_state: PrimaryState,
    manual: ManualStandUp,
    last_observed_time: Option<Time>,
}

impl PrimaryStateFilter {
    fn request_damping(&mut self, now: Time) {
        self.manual.record_damping_request(now);
        self.transition_to(PrimaryState::Damping);
    }

    fn transition_to(&mut self, state: PrimaryState) {
        if self.primary_state == state {
            return;
        }
        self.manual.eligible =
            self.primary_state == PrimaryState::Initial && state == PrimaryState::Damping;
        if state != PrimaryState::ManualStandUp {
            self.manual.started = None;
        }
        self.primary_state = state;
    }

    fn watchdog(&mut self, now: Time) {
        let time_went_backwards = self
            .last_observed_time
            .replace(now)
            .is_some_and(|last| now < last);
        if time_went_backwards {
            self.manual.reset_button_edge();
            if self.primary_state == PrimaryState::ManualStandUp {
                warn!("manual stand-up aborted: clock moved backwards");
                self.request_damping(now);
                return;
            }
        }
        self.manual.expire_controller_edge(now);
        if self.manual.damping_pressed(now) {
            if self.primary_state != PrimaryState::Damping {
                self.request_damping(now);
            }
        } else if self.primary_state == PrimaryState::ManualStandUp {
            if self.manual.must_abort(now) {
                warn!(
                    "manual stand-up aborted: stale/invalid inputs, unsafe fall state or timeout"
                );
                self.request_damping(now);
            } else if self.manual.is_complete(now) {
                info!("manual stand-up complete; returning to Initial");
                self.transition_to(PrimaryState::Initial);
            }
        }
    }

    fn update_with_filtered_game_controller_state(
        &mut self,
        filtered_game_controller_state: &FilteredGameControllerState,
        player_number: PlayerNumber,
    ) {
        let is_penalized = filtered_game_controller_state.penalties[player_number].is_some();
        let filtered_game_state = filtered_game_controller_state.game_state;
        let is_penalty_shootout = matches!(
            filtered_game_controller_state.game_phase,
            GamePhase::PenaltyShootout { .. }
        );

        let next_state = match (self.primary_state, filtered_game_state) {
            (PrimaryState::Damping | PrimaryState::ManualStandUp, _) => self.primary_state,
            (PrimaryState::Initial, FilteredGameState::Set) if is_penalty_shootout => {
                PrimaryState::Set
            }
            (PrimaryState::Initial, game_state) => {
                game_state_to_primary_state(game_state, is_penalized)
            }
            (PrimaryState::Ready, FilteredGameState::Set) if !is_penalized => PrimaryState::Set,
            (PrimaryState::Set, FilteredGameState::Playing { .. }) if !is_penalized => {
                PrimaryState::Playing
            }
            (PrimaryState::Playing, FilteredGameState::Ready) if !is_penalized => {
                PrimaryState::Ready
            }
            (state, FilteredGameState::Finished)
                if !matches!(state, PrimaryState::Damping | PrimaryState::Prepare) =>
            {
                PrimaryState::Finished
            }
            (state, _)
                if is_penalized
                    && !matches!(state, PrimaryState::Damping | PrimaryState::Prepare) =>
            {
                PrimaryState::Penalized
            }
            (state, FilteredGameState::Stop)
                if !matches!(state, PrimaryState::Damping | PrimaryState::Prepare) =>
            {
                PrimaryState::Stop
            }
            (PrimaryState::Stop, game_state) => {
                game_state_to_primary_state(game_state, is_penalized)
            }
            (PrimaryState::Penalized, game_state) if !is_penalized => {
                game_state_to_primary_state(game_state, false)
            }
            _ => self.primary_state,
        };
        self.transition_to(next_state);
    }

    fn update_with_buttons(
        &mut self,
        buttons: &Buttons<Option<ButtonPressType>>,
        is_safe_pose: bool,
        now: Time,
    ) {
        if matches!(buttons.f1, Some(ButtonPressType::Short)) {
            self.request_damping(now);
            return;
        }
        let next_state = match (self.primary_state, buttons) {
            (PrimaryState::ManualStandUp, _) => PrimaryState::ManualStandUp,
            (
                _,
                Buttons {
                    stand: Some(ButtonPressType::Short),
                    ..
                },
            ) => PrimaryState::Prepare,
            (
                PrimaryState::Prepare,
                Buttons {
                    stand: Some(ButtonPressType::Long),
                    ..
                },
            ) if is_safe_pose => PrimaryState::Initial,
            (
                PrimaryState::Initial,
                Buttons {
                    walking: Some(ButtonPressType::Long),
                    ..
                },
            ) => PrimaryState::Playing,
            _ => self.primary_state,
        };
        self.transition_to(next_state);
    }

    fn update_with_injected_primary_state(
        &mut self,
        injected_primary_state: PrimaryState,
        now: Time,
    ) {
        if injected_primary_state == PrimaryState::ManualStandUp {
            return;
        }
        if injected_primary_state == PrimaryState::Damping {
            self.request_damping(now);
            return;
        }
        if matches!(
            self.primary_state,
            PrimaryState::Damping | PrimaryState::Prepare | PrimaryState::ManualStandUp
        ) {
            return;
        }
        self.transition_to(injected_primary_state);
    }

    fn update_with_controller_input(
        &mut self,
        controller_input: &ControllerInput,
        source_time: Time,
        now: Time,
    ) {
        let rising_edge = self
            .manual
            .update_controller(controller_input, source_time, now);
        if controller_input.is_pressed(Button::West) {
            log::warn!("Primary state DAMPING triggered by controller action");
            self.request_damping(now);
        } else if rising_edge && self.primary_state == PrimaryState::Damping {
            if let Some(reason) = self.manual.start_rejection(now) {
                warn!(reason, "manual stand-up A press rejected");
            } else {
                self.manual.started = Some(now);
                self.transition_to(PrimaryState::ManualStandUp);
                info!("manual stand-up activated");
            }
        }
    }
}

fn game_state_to_primary_state(game_state: FilteredGameState, is_penalized: bool) -> PrimaryState {
    if is_penalized {
        if game_state == FilteredGameState::Finished {
            return PrimaryState::Finished;
        }
        PrimaryState::Penalized
    } else {
        match game_state {
            FilteredGameState::Initial => PrimaryState::Initial,
            FilteredGameState::Ready => PrimaryState::Ready,
            FilteredGameState::Set => PrimaryState::Set,
            FilteredGameState::Playing { .. } => PrimaryState::Playing,
            FilteredGameState::Finished => PrimaryState::Finished,
            FilteredGameState::Stop => PrimaryState::Stop,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_with_buttons_enters_playing_from_initial_with_safe_long_stand_press() {
        let mut primary_state_filter = PrimaryStateFilter {
            primary_state: PrimaryState::Initial,
            ..Default::default()
        };
        let buttons = Buttons {
            f1: None,
            stand: None,
            walking: Some(ButtonPressType::Long),
        };

        primary_state_filter.update_with_buttons(&buttons, true, Time::from_nanos(2_000_000_000));

        assert_eq!(primary_state_filter.primary_state, PrimaryState::Playing);
    }
}
