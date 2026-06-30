use color_eyre::{Result, eyre::ensure};
use coordinate_systems::{Field, Ground};
use hsl_network_messages::{HulkMessage, PlayerNumber, StateMessage};
use linear_algebra::{Isometry2, Point2, Pose2};
use ros_z::{prelude::*, qos::QosDurability};
use std::{
    boxed::Box,
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};
use types::{
    ball_position::{BallPosition, HypotheticalBallPosition},
    field_dimensions::FieldDimensions,
    filtered_game_controller_state::FilteredGameControllerState,
    filtered_game_state::FilteredGameState,
    messages::IncomingMessage,
    obstacles::Obstacle,
    parameters::SearchSuggestorParameters,
    primary_state::PrimaryState,
    time_wrapper::TimeWrapper,
};
pub mod heatmap;
use heatmap::Heatmap;

pub fn run_boxed(ctx: Arc<Context>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> {
    Box::pin(run(ctx))
}

async fn run(ctx: Arc<Context>) -> Result<()> {
    let node = ctx.create_node("search_suggestor").build().await?;

    let parameters = node.bind_parameter_as::<SearchSuggestorParameters>("search_suggestor")?;
    let field_dimensions_sub = node
        .subscriber::<FieldDimensions>("field_dimensions")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .build()
        .await?;
    let player_number_cache = node
        .subscriber::<PlayerNumber>("player_number")
        .qos(QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        })
        .cache(1)
        .build()
        .await?;
    let ball_position_sub = node
        .subscriber::<Option<BallPosition<Ground>>>("ball_filter/ball_position")
        .build()
        .await?;
    let hypothetical_ball_positions_sub = node
        .subscriber::<Vec<HypotheticalBallPosition<Ground>>>(
            "ball_filter/hypothetical_ball_positions",
        )
        .build()
        .await?;
    let ground_to_field_cache = node
        .subscriber::<Isometry2<Ground, Field>>("ground_to_field")
        .cache(10)
        .build()
        .await?;
    let primary_state_cache = node
        .subscriber::<PrimaryState>("primary_state")
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
    let network_message_sub = node
        .subscriber::<TimeWrapper<IncomingMessage>>("filtered_message")
        .build()
        .await?;
    let obstacles_sub = node
        .subscriber::<Vec<Obstacle>>("obstacles")
        .build()
        .await?;
    let additional_heatmap_pub = node
        .publisher::<types::heatmap::Heatmap>("ball_search_heatmap")
        .build()
        .await?;
    let suggested_search_position_pub = node
        .publisher::<Point2<Field>>("suggested_search_position")
        .build()
        .await?;
    let suggested_search_look_at_pub = node
        .publisher::<Option<Point2<Field>>>("suggested_search_look_at")
        .build()
        .await?;

    let field_dimensions = field_dimensions_sub.recv().await?;
    let initial_parameters_snapshot = parameters.snapshot();
    let initial_parameters = initial_parameters_snapshot.typed();
    let (heatmap_length, heatmap_width) =
        Heatmap::heatmap_dimensions(field_dimensions, initial_parameters.cells_per_meter);

    ensure!(
        heatmap_length > 0,
        "heatmap_length must at least be 1 - current value is {heatmap_length}"
    );
    ensure!(
        heatmap_width > 0,
        format!("heatmap_width must at least be 1 - current value is {heatmap_width}")
    );

    let mut heatmap = Heatmap::new_uniform_with_dimensions(
        heatmap_length,
        heatmap_width,
        initial_parameters.cells_per_meter,
    );
    let mut last_heatmap_update = Instant::now();
    let mut last_known_ball_position = None;
    let mut last_voronoi_filtered_search_position = None;
    let mut latest_filtered_game_controller_state = None;
    let mut latest_obstacles = Vec::new();
    let mut teammate_poses = HashMap::new();

    loop {
        let parameters_snapshot = parameters.snapshot();
        let parameters = parameters_snapshot.typed();

        let ground_to_field = ground_to_field_cache
            .get_latest()
            .map(|ground_to_field| *ground_to_field);
        let own_player_number = player_number_cache
            .get_latest()
            .map(|player_number| *player_number);
        let primary_state = primary_state_cache.get_latest();
        let primary_state = primary_state.as_deref();
        let mut ball_was_seen = false;

        while ball_position_sub.is_ready() {
            let ball_position = ball_position_sub.recv().await?;
            if let (Some(ball_position), Some(ground_to_field)) = (ball_position, ground_to_field) {
                ball_was_seen = true;
                last_known_ball_position = Some(heatmap.update_with_ball_position(
                    field_dimensions,
                    ball_position,
                    ground_to_field,
                ));
            }
        }
        while hypothetical_ball_positions_sub.is_ready() {
            let hypothetical_ball_positions = hypothetical_ball_positions_sub.recv().await?;
            if !ball_was_seen && let Some(ground_to_field) = ground_to_field {
                heatmap.update_with_hypothetical_ball_positions(
                    field_dimensions,
                    hypothetical_ball_positions,
                    ground_to_field,
                    parameters,
                );
            }
        }
        while obstacles_sub.is_ready() {
            latest_obstacles = obstacles_sub.recv().await?;
        }
        while network_message_sub.is_ready() {
            let network_message = network_message_sub.recv().await?;
            update_latest_teammate_pose(&network_message, own_player_number, &mut teammate_poses);
            let team_ball_position = if let Some(ground_to_field) = ground_to_field {
                heatmap.update_with_team_ball_with_obstacles(
                    field_dimensions,
                    network_message,
                    parameters,
                    &latest_obstacles,
                    ground_to_field,
                )
            } else {
                heatmap.update_with_team_ball(field_dimensions, network_message, parameters)
            };
            if let Some(team_ball_position) = team_ball_position {
                ball_was_seen = true;
                last_known_ball_position = Some(team_ball_position);
            }
        }
        while filtered_game_controller_state_sub.is_ready() {
            let filtered_game_controller_state = filtered_game_controller_state_sub.recv().await?;
            if !ball_was_seen && let Some(primary_state) = primary_state {
                heatmap.update_with_rule_ball(
                    &filtered_game_controller_state,
                    &field_dimensions,
                    primary_state,
                    parameters,
                );
            }
            latest_filtered_game_controller_state = Some(filtered_game_controller_state);
        }

        let now = Instant::now();
        let elapsed = now.duration_since(last_heatmap_update);
        last_heatmap_update = now;
        if !ball_was_seen {
            if let (Some(primary_state), Some(filtered_game_controller_state)) = (
                primary_state,
                latest_filtered_game_controller_state.as_ref(),
            ) && should_recover_rule_ball_uncertainty(
                primary_state,
                filtered_game_controller_state,
            ) {
                heatmap.recover_rule_ball_uncertainty(
                    filtered_game_controller_state,
                    &field_dimensions,
                    primary_state,
                    elapsed,
                    parameters.recovery_duration,
                );
            } else if should_recover_uncertainty(
                primary_state,
                latest_filtered_game_controller_state.as_ref(),
            ) {
                heatmap.recover_uncertainty(
                    field_dimensions,
                    last_known_ball_position,
                    elapsed,
                    parameters.recovery_duration,
                    parameters.recovery_minimum_factor,
                    parameters.recovery_gaussian_sigma,
                );
            }
        }

        if !ball_was_seen && let Some(ground_to_field) = ground_to_field {
            heatmap.decay_tiles_in_robot_fov_with_obstacles(
                field_dimensions,
                ground_to_field.as_pose().position().coords(),
                ground_to_field.orientation().angle(),
                parameters.decay_distance_factor,
                parameters.heatmap_decay_range.clone(),
                parameters.heatmap_full_decay_distance,
                parameters.heatmap_decay_falloff_distance,
                &latest_obstacles,
                ground_to_field,
            );
        }

        if let Some(ground_to_field) = ground_to_field {
            heatmap.clear_obstacle_occupied_cells(
                field_dimensions,
                &latest_obstacles,
                ground_to_field,
            );
        }

        heatmap.update_selected_target(
            parameters.minimum_validity,
            parameters.tile_switch_hysteresis,
        );

        let suggested_search_target = if let (Some(own_player_number), Some(ground_to_field)) =
            (own_player_number, ground_to_field)
        {
            let sites = collect_voronoi_sites(
                own_player_number,
                ground_to_field,
                &teammate_poses,
                parameters.goal_keeper_number,
            );
            heatmap.voronoi_filtered_search_target_with_hysteresis(
                field_dimensions,
                own_player_number,
                &sites,
                last_voronoi_filtered_search_position,
                parameters.minimum_validity,
                parameters.tile_switch_hysteresis,
            )
        } else {
            heatmap.selected_search_target(field_dimensions)
        };

        if let Some(suggested_search_target) = suggested_search_target {
            last_voronoi_filtered_search_position = Some(suggested_search_target.position);
            suggested_search_position_pub
                .publish(&suggested_search_target.position)
                .await?;
        }
        suggested_search_look_at_pub
            .publish(&suggested_search_target.and_then(|target| target.look_at))
            .await?;

        additional_heatmap_pub
            .publish_if_subscribed(|| async { heatmap.to_message() })
            .await?;

        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn should_recover_uncertainty(
    primary_state: Option<&PrimaryState>,
    filtered_game_controller_state: Option<&FilteredGameControllerState>,
) -> bool {
    matches!(primary_state, Some(PrimaryState::Playing))
        && matches!(
            filtered_game_controller_state.map(|state| state.game_state),
            Some(FilteredGameState::Playing {
                ball_is_free: true,
                ..
            })
        )
}

fn should_recover_rule_ball_uncertainty(
    primary_state: &PrimaryState,
    filtered_game_controller_state: &FilteredGameControllerState,
) -> bool {
    matches!(primary_state, PrimaryState::Playing)
        && matches!(
            filtered_game_controller_state.game_state,
            FilteredGameState::Playing {
                ball_is_free: false,
                ..
            }
        )
}

fn update_latest_teammate_pose(
    network_message: &TimeWrapper<IncomingMessage>,
    own_player_number: Option<PlayerNumber>,
    teammate_poses: &mut HashMap<PlayerNumber, Pose2<Field>>,
) {
    let IncomingMessage::Hsl(HulkMessage::State(StateMessage {
        player_number,
        pose,
        ..
    })) = &network_message.inner
    else {
        return;
    };

    if Some(*player_number) != own_player_number {
        teammate_poses.insert(*player_number, *pose);
    }
}

fn collect_voronoi_sites(
    own_player_number: PlayerNumber,
    ground_to_field: Isometry2<Ground, Field>,
    teammate_poses: &HashMap<PlayerNumber, Pose2<Field>>,
    goal_keeper_number: PlayerNumber,
) -> Vec<(Pose2<Field>, PlayerNumber)> {
    if own_player_number == goal_keeper_number {
        return vec![(ground_to_field.as_pose(), own_player_number)];
    }

    let mut sites = vec![(ground_to_field.as_pose(), own_player_number)];

    for (player_number, pose) in teammate_poses {
        if *player_number != own_player_number && *player_number != goal_keeper_number {
            sites.push((*pose, *player_number));
        }
    }

    sites
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::point;

    #[test]
    fn collect_voronoi_sites_excludes_goalkeeper_teammate() {
        let own_player_number = PlayerNumber::Four;
        let ground_to_field = Isometry2::identity();
        let mut teammate_poses = HashMap::new();
        teammate_poses.insert(PlayerNumber::Two, Pose2::new(point![-3.8, 0.0], 0.0));
        teammate_poses.insert(PlayerNumber::Five, Pose2::new(point![1.0, 0.0], 0.0));

        let sites = collect_voronoi_sites(
            own_player_number,
            ground_to_field,
            &teammate_poses,
            PlayerNumber::Two,
        );

        assert_eq!(sites.len(), 2);
        assert!(
            sites
                .iter()
                .any(|(_, player_number)| *player_number == PlayerNumber::Four)
        );
        assert!(
            sites
                .iter()
                .any(|(_, player_number)| *player_number == PlayerNumber::Five)
        );
        assert!(
            !sites
                .iter()
                .any(|(_, player_number)| *player_number == PlayerNumber::Two)
        );
    }
}
