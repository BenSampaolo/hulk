use std::{
    f32::consts,
    ops::Range,
    time::{Duration, SystemTime},
};

use color_eyre::{Result, eyre::WrapErr as _};
use coordinate_systems::{Field, Ground};
use geometry::direction::{Direction, Rotate90Degrees};
use hsl_network_messages::{HulkMessage, PlayerNumber, StateMessage, SubState, Team};
use itertools::Itertools;
use linear_algebra::{Isometry2, Point2, Pose2, Vector2, point, vector};
use nalgebra::clamp;
use ndarray::{Array2, array};
use ndarray_conv::{ConvExt, ConvMode, PaddingMode};
use ros_z::time::Time;
use serde::{Deserialize, Serialize};
use types::{
    ball_position::{BallPosition, HypotheticalBallPosition},
    field_dimensions::{FieldDimensions, Half, Side},
    filtered_game_controller_state::FilteredGameControllerState,
    filtered_game_state::FilteredGameState,
    heatmap::Heatmap as HeatmapMessage,
    messages::IncomingMessage,
    obstacles::{Obstacle, ObstacleKind},
    parameters::SearchSuggestorParameters,
    primary_state::PrimaryState,
    time_wrapper::TimeWrapper,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldObstacle {
    pub position: Point2<Field>,
    pub radius: f32,
}

impl FieldObstacle {
    pub fn new(position: Point2<Field>, radius: f32) -> Self {
        Self { position, radius }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Heatmap {
    pub(crate) map: Array2<f32>,
    pub(crate) cells_per_meter: f32,
    pub(crate) last_maximum_heatmap_position: Option<(usize, usize)>,
    pub(crate) has_decided_for_heatmap_tile: bool,
    #[serde(default)]
    pub(crate) active_rule_ball_hypotheses: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchTarget {
    pub position: Point2<Field>,
    pub look_at: Option<Point2<Field>>,
}

impl Heatmap {
    pub fn heatmap_dimensions(
        field_dimensions: FieldDimensions,
        cells_per_meter: f32,
    ) -> (usize, usize) {
        (
            (field_dimensions.length * cells_per_meter).round() as usize,
            (field_dimensions.width * cells_per_meter).round() as usize,
        )
    }

    pub fn new_uniform(field_dimensions: FieldDimensions, cells_per_meter: f32) -> Self {
        let (length, width) = Self::heatmap_dimensions(field_dimensions, cells_per_meter);
        assert!(
            length > 0,
            "heatmap_length must at least be 1 - current value is {length}"
        );
        assert!(
            width > 0,
            "heatmap_width must at least be 1 - current value is {width}"
        );
        Self::new_uniform_with_dimensions(length, width, cells_per_meter)
    }

    pub(crate) fn new_uniform_with_dimensions(
        length: usize,
        width: usize,
        cells_per_meter: f32,
    ) -> Self {
        Self {
            map: Array2::ones((length, width)),
            cells_per_meter,
            last_maximum_heatmap_position: None,
            has_decided_for_heatmap_tile: false,
            active_rule_ball_hypotheses: Vec::new(),
        }
    }

    pub fn dimensions(&self) -> (usize, usize) {
        self.map.dim()
    }

    pub fn cells_per_meter(&self) -> f32 {
        self.cells_per_meter
    }

    pub fn value_at(&self, heatmap_point: (usize, usize)) -> f32 {
        self.map[heatmap_point]
    }

    pub fn clear_cells_matching(
        &mut self,
        field_dimensions: FieldDimensions,
        mut should_clear: impl FnMut(Point2<Field>, f32) -> bool,
    ) {
        let cell_size = 1.0 / self.cells_per_meter;
        let cells_per_meter = self.cells_per_meter;
        for ((x, y), value) in self.map.indexed_iter_mut() {
            let cell_center = heatmap_index_to_field(field_dimensions, cells_per_meter, (x, y));
            if should_clear(cell_center, cell_size) {
                *value = 0.0;
            }
        }
    }

    pub fn clear_obstacle_occupied_cells(
        &mut self,
        field_dimensions: FieldDimensions,
        obstacles: &[Obstacle],
        ground_to_field: Isometry2<Ground, Field>,
    ) {
        let field_obstacles = field_obstacles_from_ground_obstacles(obstacles, ground_to_field);
        self.clear_obstacle_occupied_cells_in_field(field_dimensions, &field_obstacles);
    }

    pub fn clear_obstacle_occupied_cells_in_field(
        &mut self,
        field_dimensions: FieldDimensions,
        obstacles: &[FieldObstacle],
    ) {
        self.clear_cells_matching(field_dimensions, |cell_center, cell_size| {
            heatmap_cell_intersects_obstacles(cell_center, cell_size, obstacles)
        });
    }

    fn clamp_values(&mut self) {
        self.map
            .iter_mut()
            .for_each(|value| *value = clamp_heatmap_value(*value));
    }

    pub(crate) fn to_message(&self) -> HeatmapMessage {
        let (length, width) = self.map.dim();
        HeatmapMessage {
            length: length as u32,
            width: width as u32,
            values: self.map.iter().copied().collect(),
        }
    }

    pub(crate) fn update_with_ball_position(
        &mut self,
        field_dimensions: FieldDimensions,
        ball_position: BallPosition<Ground>,
        ground_to_field: Isometry2<Ground, Field>,
    ) -> Point2<Field> {
        let field_ball_position = ground_to_field * ball_position.position;
        self.update_with_field_ball_position(field_dimensions, field_ball_position);
        field_ball_position
    }

    pub fn update_with_field_ball_position(
        &mut self,
        field_dimensions: FieldDimensions,
        ball_position: Point2<Field>,
    ) {
        self.active_rule_ball_hypotheses.clear();
        self.map.fill(0.0);
        let heatmap_point = self.field_to_heatmap(field_dimensions, ball_position);
        self.map[heatmap_point] = 1.0;
    }

    pub(crate) fn update_with_hypothetical_ball_positions(
        &mut self,
        field_dimensions: FieldDimensions,
        hypothetical_ball_positions: Vec<HypotheticalBallPosition<Ground>>,
        ground_to_field: Isometry2<Ground, Field>,
        parameters: &SearchSuggestorParameters,
    ) {
        self.active_rule_ball_hypotheses.clear();
        for ball_hypothesis in hypothetical_ball_positions {
            let ball_hypothesis_position = ground_to_field * ball_hypothesis.position;
            let heatmap_point = self.field_to_heatmap(field_dimensions, ball_hypothesis_position);
            self.map[heatmap_point] = clamp_heatmap_value(
                (self.map[heatmap_point] + ball_hypothesis.validity * parameters.own_ball_weight)
                    / 2.0,
            );
        }
    }

    pub fn update_with_rule_ball(
        &mut self,
        filtered_game_controller_state: &FilteredGameControllerState,
        field_dimensions: &FieldDimensions,
        primary_state: &PrimaryState,
        parameters: &SearchSuggestorParameters,
    ) {
        let rule_ball_hypotheses = get_rule_hypotheses(
            *primary_state,
            filtered_game_controller_state,
            *field_dimensions,
        );

        if is_ball_not_free(filtered_game_controller_state) {
            let rule_ball_hypothesis_indices =
                self.rule_ball_hypothesis_indices(*field_dimensions, rule_ball_hypotheses);
            if !rule_ball_hypothesis_indices.is_empty() {
                self.apply_rule_ball_hypothesis_mask(rule_ball_hypothesis_indices);
            }
            return;
        }

        self.active_rule_ball_hypotheses.clear();
        for rule_ball_hypothesis in rule_ball_hypotheses {
            let heatmap_point = self.field_to_heatmap(*field_dimensions, rule_ball_hypothesis);
            self.map[heatmap_point] = clamp_heatmap_value(
                self.map[heatmap_point] + parameters.rule_ball_weight_increment,
            );
        }
    }

    pub fn recover_rule_ball_uncertainty(
        &mut self,
        filtered_game_controller_state: &FilteredGameControllerState,
        field_dimensions: &FieldDimensions,
        primary_state: &PrimaryState,
        delta: Duration,
        recovery_duration: Duration,
    ) {
        if !is_ball_not_free(filtered_game_controller_state) {
            return;
        }

        let rule_ball_hypotheses = get_rule_hypotheses(
            *primary_state,
            filtered_game_controller_state,
            *field_dimensions,
        );
        let rule_ball_hypothesis_indices =
            self.rule_ball_hypothesis_indices(*field_dimensions, rule_ball_hypotheses);
        if !rule_ball_hypothesis_indices.is_empty() {
            self.apply_rule_ball_hypothesis_mask(rule_ball_hypothesis_indices);
        }

        let recovery_duration_seconds = recovery_duration.as_secs_f32();
        if recovery_duration_seconds <= 0.0 {
            for heatmap_point in self.active_rule_ball_hypotheses.iter().copied() {
                self.map[heatmap_point] = 1.0;
            }
            return;
        }

        let base_recovery = delta.as_secs_f32() / recovery_duration_seconds;
        if base_recovery <= 0.0 {
            return;
        }

        for heatmap_point in self.active_rule_ball_hypotheses.iter().copied() {
            self.map[heatmap_point] = clamp_heatmap_value(self.map[heatmap_point] + base_recovery);
        }
    }

    fn rule_ball_hypothesis_indices(
        &self,
        field_dimensions: FieldDimensions,
        rule_ball_hypotheses: Vec<Point2<Field>>,
    ) -> Vec<(usize, usize)> {
        rule_ball_hypotheses
            .into_iter()
            .map(|rule_ball_hypothesis| {
                self.field_to_heatmap(field_dimensions, rule_ball_hypothesis)
            })
            .unique()
            .collect()
    }

    fn apply_rule_ball_hypothesis_mask(
        &mut self,
        rule_ball_hypothesis_indices: Vec<(usize, usize)>,
    ) {
        let previous_rule_ball_hypotheses = self.active_rule_ball_hypotheses.clone();
        let continuing_values = rule_ball_hypothesis_indices
            .iter()
            .copied()
            .filter(|heatmap_point| previous_rule_ball_hypotheses.contains(heatmap_point))
            .map(|heatmap_point| (heatmap_point, self.map[heatmap_point]))
            .collect::<Vec<_>>();

        self.map.fill(0.0);
        self.active_rule_ball_hypotheses = rule_ball_hypothesis_indices;

        for heatmap_point in self.active_rule_ball_hypotheses.iter().copied() {
            self.map[heatmap_point] = 1.0;
        }
        for (heatmap_point, value) in continuing_values {
            self.map[heatmap_point] = clamp_heatmap_value(value);
        }
    }

    pub(crate) fn update_with_team_ball(
        &mut self,
        field_dimensions: FieldDimensions,
        network_message: TimeWrapper<IncomingMessage>,
        parameters: &SearchSuggestorParameters,
    ) -> Option<Point2<Field>> {
        let IncomingMessage::Hsl(message) = network_message.inner else {
            return None;
        };
        self.add_teamballs(
            field_dimensions,
            network_message.time.to_wallclock(),
            message,
            parameters,
        )
    }

    pub(crate) fn get_maximum_position(&self, minimum_validity: f32) -> Option<(usize, usize)> {
        let linear_maximum_heat_heatmap_position =
            self.map.iter().position_max_by(|a, b| a.total_cmp(b))?;
        let maximum_heat_heatmap_position = (
            linear_maximum_heat_heatmap_position / self.map.dim().1,
            linear_maximum_heat_heatmap_position % self.map.dim().1,
        );
        if self.map[maximum_heat_heatmap_position] > minimum_validity {
            return Some(maximum_heat_heatmap_position);
        }
        None
    }

    pub fn decay_tiles_in_robot_fov(
        &mut self,
        field_dimensions: FieldDimensions,
        robot_position: Vector2<Field>,
        viewing_orientation: f32,
        decay_distance_factor: f32,
        heatmap_decay_range: Range<f32>,
        heatmap_full_decay_distance: f32,
        heatmap_decay_falloff_distance: f32,
    ) {
        self.decay_tiles_in_robot_fov_with_visibility_filter(
            field_dimensions,
            robot_position,
            viewing_orientation,
            decay_distance_factor,
            heatmap_decay_range,
            heatmap_full_decay_distance,
            heatmap_decay_falloff_distance,
            |_| true,
        );
    }

    pub fn decay_tiles_in_robot_fov_with_obstacles(
        &mut self,
        field_dimensions: FieldDimensions,
        robot_position: Vector2<Field>,
        viewing_orientation: f32,
        decay_distance_factor: f32,
        heatmap_decay_range: Range<f32>,
        heatmap_full_decay_distance: f32,
        heatmap_decay_falloff_distance: f32,
        obstacles: &[Obstacle],
        ground_to_field: Isometry2<Ground, Field>,
    ) {
        let field_obstacles = field_obstacles_from_ground_obstacles(obstacles, ground_to_field);
        self.decay_tiles_in_robot_fov_with_field_obstacles(
            field_dimensions,
            robot_position,
            viewing_orientation,
            decay_distance_factor,
            heatmap_decay_range,
            heatmap_full_decay_distance,
            heatmap_decay_falloff_distance,
            &field_obstacles,
        );
    }

    pub fn decay_tiles_in_robot_fov_with_field_obstacles(
        &mut self,
        field_dimensions: FieldDimensions,
        robot_position: Vector2<Field>,
        viewing_orientation: f32,
        decay_distance_factor: f32,
        heatmap_decay_range: Range<f32>,
        heatmap_full_decay_distance: f32,
        heatmap_decay_falloff_distance: f32,
        obstacles: &[FieldObstacle],
    ) {
        let robot_point = point![robot_position.x(), robot_position.y()];
        self.decay_tiles_in_robot_fov_with_visibility_filter(
            field_dimensions,
            robot_position,
            viewing_orientation,
            decay_distance_factor,
            heatmap_decay_range,
            heatmap_full_decay_distance,
            heatmap_decay_falloff_distance,
            |tile_center| !line_of_sight_blocked_by_obstacles(robot_point, tile_center, obstacles),
        );
    }

    pub fn decay_tiles_in_robot_fov_with_visibility_filter(
        &mut self,
        field_dimensions: FieldDimensions,
        robot_position: Vector2<Field>,
        viewing_orientation: f32,
        decay_distance_factor: f32,
        heatmap_decay_range: Range<f32>,
        heatmap_full_decay_distance: f32,
        heatmap_decay_falloff_distance: f32,
        is_tile_visible: impl Fn(Point2<Field>) -> bool,
    ) {
        let fov_angle_offset = 45.0 * consts::PI / 180.0;
        let left_angle = viewing_orientation - fov_angle_offset;
        let right_angle = viewing_orientation + fov_angle_offset;
        let left_edge: Vector2<Field> = vector!(left_angle.cos(), left_angle.sin());
        let right_edge: Vector2<Field> = vector!(right_angle.cos(), right_angle.sin());

        self.decay_tiles_in_fov(
            field_dimensions,
            robot_position,
            left_edge,
            right_edge,
            decay_distance_factor,
            heatmap_decay_range,
            heatmap_full_decay_distance,
            heatmap_decay_falloff_distance,
            is_tile_visible,
        );
    }

    fn decay_tiles_in_fov(
        &mut self,
        field_dimensions: FieldDimensions,
        robot_position: Vector2<Field>,
        left_edge: Vector2<Field>,
        right_edge: Vector2<Field>,
        decay_distance_factor: f32,
        heatmap_decay_range: Range<f32>,
        heatmap_full_decay_distance: f32,
        heatmap_decay_falloff_distance: f32,
        is_tile_visible: impl Fn(Point2<Field>) -> bool,
    ) {
        let cells_per_meter = self.cells_per_meter;
        self.map.indexed_iter_mut().for_each(|((x, y), value)| {
            let tile_center_in_field =
                heatmap_index_to_field(field_dimensions, cells_per_meter, (x, y));
            let robot_to_tile = tile_center_in_field.coords() - robot_position;
            let distance_to_tile = robot_to_tile.norm();
            let is_inside_sight = get_direction(left_edge, robot_to_tile)
                == Direction::Counterclockwise
                && get_direction(right_edge, robot_to_tile) == Direction::Clockwise;
            if is_inside_sight
                && heatmap_decay_range.contains(&distance_to_tile)
                && is_tile_visible(tile_center_in_field)
            {
                let decay_scale = heatmap_decay_scale(
                    distance_to_tile,
                    heatmap_full_decay_distance,
                    heatmap_decay_falloff_distance,
                );
                *value = clamp_heatmap_value(*value * (1.0 - decay_distance_factor * decay_scale));
            }
        });
    }

    pub fn recover_uncertainty(
        &mut self,
        field_dimensions: FieldDimensions,
        last_known_ball_position: Option<Point2<Field>>,
        delta: Duration,
        recovery_duration: Duration,
        minimum_recovery_factor: f32,
        gaussian_sigma: f32,
    ) {
        self.active_rule_ball_hypotheses.clear();
        let recovery_duration_seconds = recovery_duration.as_secs_f32();
        if recovery_duration_seconds <= 0.0 {
            self.map.fill(1.0);
            return;
        }

        let base_recovery = delta.as_secs_f32() / recovery_duration_seconds;
        if base_recovery <= 0.0 {
            return;
        }

        let minimum_recovery_factor = clamp_heatmap_value(minimum_recovery_factor);
        let sigma_squared = gaussian_sigma * gaussian_sigma;
        let cells_per_meter = self.cells_per_meter;

        self.map.indexed_iter_mut().for_each(|((x, y), value)| {
            let factor = last_known_ball_position
                .filter(|_| sigma_squared > 0.0)
                .map(|center| {
                    let tile_center =
                        heatmap_index_to_field(field_dimensions, cells_per_meter, (x, y));
                    let distance_squared = (tile_center - center).norm_squared();
                    let gaussian = (-distance_squared / (2.0 * sigma_squared)).exp();
                    minimum_recovery_factor + (1.0 - minimum_recovery_factor) * gaussian
                })
                .unwrap_or(minimum_recovery_factor);
            *value = clamp_heatmap_value(*value + base_recovery * factor);
        });
    }

    pub fn apply_convolution(&mut self, alpha: f32) -> Result<()> {
        let kernel = array![
            [alpha, alpha, alpha],
            [alpha, 1.0 - alpha, alpha],
            [alpha, alpha, alpha]
        ] / (1.0 + 7.0 * alpha);
        self.map = self
            .map
            .conv(&kernel, ConvMode::Same, PaddingMode::Replicate)
            .wrap_err("heatmap convolution failed")?;
        self.clamp_values();
        Ok(())
    }

    pub fn apply_convolution_and_normalize(&mut self, alpha: f32) -> Result<()> {
        self.apply_convolution(alpha)
    }

    pub fn update_selected_target(&mut self, minimum_validity: f32, tile_switch_hysteresis: f32) {
        if !self.has_decided_for_heatmap_tile {
            let suggested_search_index = self.get_maximum_position(minimum_validity);
            if suggested_search_index.is_some() {
                self.has_decided_for_heatmap_tile = true;
            }
            self.last_maximum_heatmap_position = suggested_search_index;
        } else if let Some(last_maximum_heatmap_index) = self.last_maximum_heatmap_position {
            let global_max_value = self
                .get_maximum_position(0.0)
                .map_or(0.0, |idx| self.map[idx]);
            let current_tile_value = self.map[last_maximum_heatmap_index];

            if current_tile_value < global_max_value * tile_switch_hysteresis {
                self.has_decided_for_heatmap_tile = false;
            }
        }
    }

    pub fn selected_search_position(
        &self,
        field_dimensions: FieldDimensions,
    ) -> Option<Point2<Field>> {
        self.selected_search_target(field_dimensions)
            .map(|target| target.position)
    }

    pub fn selected_search_target(
        &self,
        field_dimensions: FieldDimensions,
    ) -> Option<SearchTarget> {
        self.last_maximum_heatmap_position
            .map(|index| SearchTarget {
                position: self.heatmap_to_field(field_dimensions, index),
                look_at: None,
            })
    }

    pub fn voronoi_filtered_search_position(
        &self,
        field_dimensions: FieldDimensions,
        own_player_number: PlayerNumber,
        sites: &[(Pose2<Field>, PlayerNumber)],
        minimum_validity: f32,
    ) -> Option<Point2<Field>> {
        self.voronoi_filtered_search_target(
            field_dimensions,
            own_player_number,
            sites,
            minimum_validity,
        )
        .map(|target| target.position)
    }

    pub fn voronoi_filtered_search_target(
        &self,
        field_dimensions: FieldDimensions,
        own_player_number: PlayerNumber,
        sites: &[(Pose2<Field>, PlayerNumber)],
        minimum_validity: f32,
    ) -> Option<SearchTarget> {
        self.voronoi_filtered_search_target_with_hysteresis(
            field_dimensions,
            own_player_number,
            sites,
            None,
            minimum_validity,
            0.0,
        )
    }

    pub fn voronoi_filtered_search_position_with_hysteresis(
        &self,
        field_dimensions: FieldDimensions,
        own_player_number: PlayerNumber,
        sites: &[(Pose2<Field>, PlayerNumber)],
        previous_search_position: Option<Point2<Field>>,
        minimum_validity: f32,
        tile_switch_hysteresis: f32,
    ) -> Option<Point2<Field>> {
        self.voronoi_filtered_search_target_with_hysteresis(
            field_dimensions,
            own_player_number,
            sites,
            previous_search_position,
            minimum_validity,
            tile_switch_hysteresis,
        )
        .map(|target| target.position)
    }

    pub fn voronoi_filtered_search_target_with_hysteresis(
        &self,
        field_dimensions: FieldDimensions,
        own_player_number: PlayerNumber,
        sites: &[(Pose2<Field>, PlayerNumber)],
        previous_search_position: Option<Point2<Field>>,
        minimum_validity: f32,
        tile_switch_hysteresis: f32,
    ) -> Option<SearchTarget> {
        let Some((own_pose, _)) = sites
            .iter()
            .find(|(_, player_number)| *player_number == own_player_number)
        else {
            return self.selected_search_target(field_dimensions);
        };

        let mut voronoi = voronoi::VoronoiGrid::new(
            voronoi::VoronoiBounds {
                grid_min: point![
                    -field_dimensions.length / 2.0,
                    -field_dimensions.width / 2.0
                ],
                grid_max: point![field_dimensions.length / 2.0, field_dimensions.width / 2.0],
                centroid_min: point![
                    -field_dimensions.length / 2.0,
                    -field_dimensions.width / 2.0
                ],
                centroid_max: point![field_dimensions.length / 2.0, field_dimensions.width / 2.0],
            },
            1.0 / self.cells_per_meter,
        );
        voronoi.multi_source_dijkstra(sites, 0.0);

        let best_owned_tile = self.best_voronoi_owned_tile(
            field_dimensions,
            own_player_number,
            &voronoi,
            minimum_validity,
        );

        if let (Some(previous_search_position), Some((_, best_value))) =
            (previous_search_position, best_owned_tile.as_ref())
        {
            let previous_heatmap_point = self
                .heatmap_points_containing_field_point(field_dimensions, previous_search_position)
                .into_iter()
                .filter(|heatmap_point| {
                    let tile_center = self.heatmap_to_field(field_dimensions, *heatmap_point);
                    matches!(
                        voronoi.ownership_at(tile_center),
                        Some(voronoi::Ownership::Robot(player_number))
                            if player_number == own_player_number
                    )
                })
                .max_by(|a, b| self.map[*a].total_cmp(&self.map[*b]))
                .unwrap_or_else(|| {
                    self.field_to_heatmap(field_dimensions, previous_search_position)
                });
            let previous_tile_center =
                self.heatmap_to_field(field_dimensions, previous_heatmap_point);
            let previous_value = self.map[previous_heatmap_point];
            let previous_tile_is_still_owned = matches!(
                voronoi.ownership_at(previous_tile_center),
                Some(voronoi::Ownership::Robot(player_number)) if player_number == own_player_number
            );

            if previous_tile_is_still_owned
                && previous_value > minimum_validity
                && previous_value >= *best_value * tile_switch_hysteresis
            {
                return Some(self.adjust_search_target_for_robot_proximity(
                    field_dimensions,
                    previous_tile_center,
                    own_pose.position(),
                ));
            }
        }

        best_owned_tile.map(|(position, _)| {
            self.adjust_search_target_for_robot_proximity(
                field_dimensions,
                position,
                own_pose.position(),
            )
        })
    }

    fn best_voronoi_owned_tile(
        &self,
        field_dimensions: FieldDimensions,
        own_player_number: PlayerNumber,
        voronoi: &voronoi::VoronoiGrid,
        minimum_validity: f32,
    ) -> Option<(Point2<Field>, f32)> {
        let mut best_tile = None;

        for ((x, y), value) in self.map.indexed_iter() {
            if !value.is_finite() || *value <= minimum_validity {
                continue;
            }

            let tile_center = self.heatmap_to_field(field_dimensions, (x, y));
            if !matches!(
                voronoi.ownership_at(tile_center),
                Some(voronoi::Ownership::Robot(player_number)) if player_number == own_player_number
            ) {
                continue;
            }

            if best_tile
                .map(|(_, best_value)| value.total_cmp(&best_value).is_gt())
                .unwrap_or(true)
            {
                best_tile = Some((tile_center, *value));
            }
        }

        best_tile
    }

    fn adjust_search_target_for_robot_proximity(
        &self,
        field_dimensions: FieldDimensions,
        selected_position: Point2<Field>,
        robot_position: Point2<Field>,
    ) -> SearchTarget {
        let selected_tile = self.field_to_heatmap(field_dimensions, selected_position);
        let selected_tile_center = self.heatmap_to_field(field_dimensions, selected_tile);
        let cell_radius = std::f32::consts::SQRT_2 / (2.0 * self.cells_per_meter);
        let robot_is_near_selected_tile =
            self.tile_contains_field_point(field_dimensions, selected_tile, robot_position)
                || (selected_tile_center - robot_position).norm() <= cell_radius;

        if robot_is_near_selected_tile {
            SearchTarget {
                position: self.clamp_search_target_inside_field(
                    field_dimensions,
                    self.nearest_tile_corner(field_dimensions, selected_tile, robot_position),
                ),
                look_at: Some(selected_tile_center),
            }
        } else {
            SearchTarget {
                position: self
                    .clamp_search_target_inside_field(field_dimensions, selected_position),
                look_at: None,
            }
        }
    }

    fn clamp_search_target_inside_field(
        &self,
        field_dimensions: FieldDimensions,
        target: Point2<Field>,
    ) -> Point2<Field> {
        let edge_margin = 0.25 / self.cells_per_meter;
        let x_limit = (field_dimensions.length / 2.0 - edge_margin).max(0.0);
        let y_limit = (field_dimensions.width / 2.0 - edge_margin).max(0.0);
        point![
            target.x().clamp(-x_limit, x_limit),
            target.y().clamp(-y_limit, y_limit)
        ]
    }

    fn nearest_tile_corner(
        &self,
        field_dimensions: FieldDimensions,
        heatmap_point: (usize, usize),
        robot_position: Point2<Field>,
    ) -> Point2<Field> {
        let corners = self.heatmap_tile_corners(field_dimensions, heatmap_point);
        *corners
            .iter()
            .min_by(|a, b| {
                let distance_a = (**a - robot_position).norm_squared();
                let distance_b = (**b - robot_position).norm_squared();
                distance_a.total_cmp(&distance_b)
            })
            .expect("heatmap tile has corners")
    }

    fn tile_contains_field_point(
        &self,
        field_dimensions: FieldDimensions,
        heatmap_point: (usize, usize),
        field_point: Point2<Field>,
    ) -> bool {
        let (min_x, max_x, min_y, max_y) =
            self.heatmap_tile_bounds(field_dimensions, heatmap_point);
        field_point.x() >= min_x
            && field_point.x() <= max_x
            && field_point.y() >= min_y
            && field_point.y() <= max_y
    }

    fn heatmap_points_containing_field_point(
        &self,
        field_dimensions: FieldDimensions,
        field_point: Point2<Field>,
    ) -> Vec<(usize, usize)> {
        let base = self.field_to_heatmap(field_dimensions, field_point);
        [
            base,
            (base.0.saturating_sub(1), base.1),
            (base.0, base.1.saturating_sub(1)),
            (base.0.saturating_sub(1), base.1.saturating_sub(1)),
        ]
        .into_iter()
        .unique()
        .filter(|heatmap_point| {
            self.tile_contains_field_point(field_dimensions, *heatmap_point, field_point)
        })
        .collect()
    }

    fn heatmap_tile_corners(
        &self,
        field_dimensions: FieldDimensions,
        heatmap_point: (usize, usize),
    ) -> [Point2<Field>; 4] {
        let (min_x, max_x, min_y, max_y) =
            self.heatmap_tile_bounds(field_dimensions, heatmap_point);
        [
            point![min_x, min_y],
            point![max_x, min_y],
            point![min_x, max_y],
            point![max_x, max_y],
        ]
    }

    fn heatmap_tile_bounds(
        &self,
        field_dimensions: FieldDimensions,
        heatmap_point: (usize, usize),
    ) -> (f32, f32, f32, f32) {
        let field_min_x = -field_dimensions.length / 2.0;
        let field_max_x = field_dimensions.length / 2.0;
        let field_min_y = -field_dimensions.width / 2.0;
        let field_max_y = field_dimensions.width / 2.0;
        let min_x = (heatmap_point.0 as f32 / self.cells_per_meter - field_dimensions.length / 2.0)
            .clamp(field_min_x, field_max_x);
        let max_x = ((heatmap_point.0 + 1) as f32 / self.cells_per_meter
            - field_dimensions.length / 2.0)
            .clamp(field_min_x, field_max_x);
        let min_y = (heatmap_point.1 as f32 / self.cells_per_meter - field_dimensions.width / 2.0)
            .clamp(field_min_y, field_max_y);
        let max_y = ((heatmap_point.1 + 1) as f32 / self.cells_per_meter
            - field_dimensions.width / 2.0)
            .clamp(field_min_y, field_max_y);
        (min_x, max_x, min_y, max_y)
    }

    pub fn heatmap_to_field(
        &self,
        field_dimensions: FieldDimensions,
        heatmap_point: (usize, usize),
    ) -> Point2<Field> {
        heatmap_index_to_field(field_dimensions, self.cells_per_meter, heatmap_point)
    }

    fn field_to_heatmap(
        &self,
        field_dimensions: FieldDimensions,
        field_point: Point2<Field>,
    ) -> (usize, usize) {
        let heatmap_point = (
            ((field_point.x() + field_dimensions.length / 2.0) * self.cells_per_meter) as usize,
            ((field_point.y() + field_dimensions.width / 2.0) * self.cells_per_meter) as usize,
        );
        (
            clamp(heatmap_point.0, 0, self.map.dim().0 - 1),
            clamp(heatmap_point.1, 0, self.map.dim().1 - 1),
        )
    }

    fn add_teamballs(
        &mut self,
        field_dimensions: FieldDimensions,
        time: SystemTime,
        message: HulkMessage,
        parameters: &SearchSuggestorParameters,
    ) -> Option<Point2<Field>> {
        let HulkMessage::State(StateMessage { ball_position, .. }) = message;

        if let Some(ball) = ball_position {
            self.active_rule_ball_hypotheses.clear();
            let ball_position = BallPosition {
                position: ball.position,
                velocity: Vector2::zeros(),
                last_seen: Time::from_wallclock(time) - ball.age,
            };
            let field_ball_position = ball_position.position;
            self.map.fill(0.0);
            let heatmap_point = self.field_to_heatmap(field_dimensions, field_ball_position);
            self.map[heatmap_point] = clamp_heatmap_value(parameters.team_ball_weight);
            Some(field_ball_position)
        } else {
            None
        }
    }
}

fn heatmap_index_to_field(
    field_dimensions: FieldDimensions,
    cells_per_meter: f32,
    heatmap_point: (usize, usize),
) -> Point2<Field> {
    point![
        ((heatmap_point.0 as f32 + 1.0 / 2.0) / cells_per_meter - field_dimensions.length / 2.0),
        ((heatmap_point.1 as f32 + 1.0 / 2.0) / cells_per_meter - field_dimensions.width / 2.0)
    ]
}

fn clamp_heatmap_value(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn field_obstacles_from_ground_obstacles(
    obstacles: &[Obstacle],
    ground_to_field: Isometry2<Ground, Field>,
) -> Vec<FieldObstacle> {
    obstacles
        .iter()
        .filter(|obstacle| obstacle.kind != ObstacleKind::Ball)
        .map(|obstacle| {
            FieldObstacle::new(
                ground_to_field * obstacle.position,
                obstacle
                    .radius_at_foot_height
                    .max(obstacle.radius_at_hip_height),
            )
        })
        .collect()
}

fn heatmap_cell_intersects_obstacles(
    cell_center: Point2<Field>,
    cell_size: f32,
    obstacles: &[FieldObstacle],
) -> bool {
    let half_cell_size = cell_size / 2.0;
    obstacles.iter().any(|obstacle| {
        circle_intersects_axis_aligned_square(
            obstacle.position,
            obstacle.radius,
            cell_center,
            half_cell_size,
        )
    })
}

fn circle_intersects_axis_aligned_square(
    circle_center: Point2<Field>,
    radius: f32,
    square_center: Point2<Field>,
    half_square_size: f32,
) -> bool {
    if !radius.is_finite() || radius <= 0.0 || !half_square_size.is_finite() {
        return false;
    }

    let closest_point = point![
        circle_center.x().clamp(
            square_center.x() - half_square_size,
            square_center.x() + half_square_size
        ),
        circle_center.y().clamp(
            square_center.y() - half_square_size,
            square_center.y() + half_square_size
        ),
    ];
    (closest_point - circle_center).norm_squared() <= radius * radius
}

fn line_of_sight_blocked_by_obstacles(
    origin: Point2<Field>,
    target: Point2<Field>,
    obstacles: &[FieldObstacle],
) -> bool {
    obstacles.iter().any(|obstacle| {
        line_segment_intersects_circle_before_target(
            origin,
            target,
            obstacle.position,
            obstacle.radius,
        )
    })
}

fn line_segment_intersects_circle_before_target(
    origin: Point2<Field>,
    target: Point2<Field>,
    circle_center: Point2<Field>,
    radius: f32,
) -> bool {
    if !radius.is_finite() || radius <= 0.0 {
        return false;
    }

    let segment = target - origin;
    let segment_length_squared = segment.norm_squared();
    if segment_length_squared <= f32::EPSILON {
        return false;
    }

    let origin_to_center = origin - circle_center;
    if origin_to_center.norm_squared() <= radius * radius {
        return false;
    }

    let a = segment_length_squared;
    let b = 2.0 * origin_to_center.dot(&segment);
    let c = origin_to_center.norm_squared() - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return false;
    }

    let discriminant_root = discriminant.sqrt();
    let first_intersection = (-b - discriminant_root) / (2.0 * a);
    let second_intersection = (-b + discriminant_root) / (2.0 * a);
    [first_intersection, second_intersection]
        .into_iter()
        .any(|intersection| intersection > f32::EPSILON && intersection < 1.0 - f32::EPSILON)
}

pub fn heatmap_decay_scale(
    distance: f32,
    full_decay_distance: f32,
    decay_falloff_distance: f32,
) -> f32 {
    if !distance.is_finite() {
        return 0.0;
    }

    let full_decay_distance = if full_decay_distance.is_finite() {
        full_decay_distance
    } else {
        0.0
    };

    if distance <= full_decay_distance {
        return 1.0;
    }

    if !decay_falloff_distance.is_finite() || decay_falloff_distance <= 0.0 {
        return 0.0;
    }

    (-(distance - full_decay_distance) / decay_falloff_distance).exp()
}

fn get_rule_hypotheses(
    primary_state: PrimaryState,
    filtered_game_controller_state: &FilteredGameControllerState,
    field_dimensions: FieldDimensions,
) -> Vec<Point2<Field>> {
    let kicking_team_half = kicking_team_half(filtered_game_controller_state.kicking_team);

    match (
        primary_state,
        filtered_game_controller_state.game_state,
        filtered_game_controller_state.sub_state,
    ) {
        (
            _,
            FilteredGameState::Playing {
                ball_is_free: false,
                kick_off: true,
            },
            None,
        ) => {
            vec![field_dimensions.center()]
        }
        (PrimaryState::Ready, _, Some(SubState::PenaltyKick)) => {
            let kicking_team_half = kicking_team_half.unwrap_or(Half::Own).mirror();
            vec![field_dimensions.penalty_spot(kicking_team_half)]
        }
        // Kick-off
        (PrimaryState::Ready, _, None) => vec![field_dimensions.center()],
        (PrimaryState::Playing, _, Some(SubState::CornerKick)) => {
            if let Some(kicking_team_half) = kicking_team_half {
                let kicking_team_half = kicking_team_half.mirror();
                vec![
                    field_dimensions.corner(kicking_team_half, Side::Left),
                    field_dimensions.corner(kicking_team_half, Side::Right),
                ]
            } else {
                vec![
                    field_dimensions.corner(Half::Own, Side::Left),
                    field_dimensions.corner(Half::Opponent, Side::Left),
                    field_dimensions.corner(Half::Own, Side::Right),
                    field_dimensions.corner(Half::Opponent, Side::Right),
                ]
            }
        }
        (PrimaryState::Playing, _, Some(SubState::GoalKick)) => {
            if let Some(kicking_team_half) = kicking_team_half {
                vec![
                    field_dimensions.goal_box_corner(kicking_team_half, Side::Left),
                    field_dimensions.goal_box_corner(kicking_team_half, Side::Right),
                ]
            } else {
                vec![
                    field_dimensions.goal_box_corner(Half::Own, Side::Left),
                    field_dimensions.goal_box_corner(Half::Opponent, Side::Left),
                    field_dimensions.goal_box_corner(Half::Own, Side::Right),
                    field_dimensions.goal_box_corner(Half::Opponent, Side::Right),
                ]
            }
        }
        (_, _, _) => Vec::new(),
    }
}

fn is_ball_not_free(filtered_game_controller_state: &FilteredGameControllerState) -> bool {
    matches!(
        filtered_game_controller_state.game_state,
        FilteredGameState::Playing {
            ball_is_free: false,
            ..
        }
    )
}

fn kicking_team_half(kicking_team: Option<Team>) -> Option<Half> {
    match kicking_team {
        Some(Team::Opponent) => Some(Half::Opponent),
        Some(Team::Hulks) => Some(Half::Own),
        None => None,
    }
}

fn get_direction(base_vector: Vector2<Field>, vector_to_test: Vector2<Field>) -> Direction {
    let clockwise_normal_vector = base_vector.rotate_90_degrees(Direction::Clockwise);
    let directed_cathetus = clockwise_normal_vector.dot(&vector_to_test);

    match directed_cathetus {
        0.0 => Direction::Collinear,
        f if f > 0.0 => Direction::Clockwise,
        f if f < 0.0 => Direction::Counterclockwise,
        f => panic!("directed cathetus was not a real number: {f}"),
    }
}

#[cfg(test)]
mod tests {
    use std::{f32::consts::FRAC_PI_2, time::Duration};

    use hsl_network_messages::{HulkMessage, PlayerNumber, StateMessage, Team};
    use linear_algebra::{Pose2, point};
    use ndarray::Array2;
    use types::{
        field_dimensions::{FieldDimensions, Half, Side},
        filtered_game_state::FilteredGameState,
        messages::IncomingMessage,
        obstacles::{Obstacle, ObstacleKind},
        parameters::SearchSuggestorParameters,
        time_wrapper::TimeWrapper,
    };

    use super::*;

    #[test]
    fn new_uniform_initializes_cells_as_fully_unknown() {
        let heatmap = Heatmap::new_uniform_with_dimensions(3, 2, 1.0);

        assert!(heatmap.map.iter().all(|value| *value == 1.0));
    }

    #[test]
    fn uncertainty_recovery_at_last_known_ball_position_uses_full_base_rate() {
        let field_dimensions = FieldDimensions {
            length: 3.0,
            width: 3.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.0);

        let center_tile = (1, 1);
        let last_known_ball_position = heatmap.heatmap_to_field(field_dimensions, center_tile);
        heatmap.recover_uncertainty(
            field_dimensions,
            Some(last_known_ball_position),
            Duration::from_secs(10),
            Duration::from_secs(20),
            0.4,
            3.0,
        );

        assert!((heatmap.map[center_tile] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn uncertainty_recovery_far_from_last_known_ball_position_uses_minimum_factor() {
        let field_dimensions = FieldDimensions {
            length: 100.0,
            width: 1.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.0);

        let last_known_ball_position = heatmap.heatmap_to_field(field_dimensions, (0, 0));
        let far_tile = (99, 0);
        heatmap.recover_uncertainty(
            field_dimensions,
            Some(last_known_ball_position),
            Duration::from_secs(10),
            Duration::from_secs(20),
            0.4,
            3.0,
        );

        assert!((heatmap.map[far_tile] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn uncertainty_recovery_without_last_known_ball_position_uses_minimum_factor() {
        let field_dimensions = FieldDimensions {
            length: 2.0,
            width: 2.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.0);

        heatmap.recover_uncertainty(
            field_dimensions,
            None,
            Duration::from_secs(10),
            Duration::from_secs(20),
            0.4,
            3.0,
        );

        assert!(heatmap.map.iter().all(|value| (*value - 0.2).abs() < 1e-6));
    }

    #[test]
    fn uncertainty_recovery_remains_clamped() {
        let field_dimensions = FieldDimensions {
            length: 2.0,
            width: 2.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.95);

        heatmap.recover_uncertainty(
            field_dimensions,
            None,
            Duration::from_secs(20),
            Duration::from_secs(20),
            0.4,
            3.0,
        );

        assert!(heatmap.map.iter().all(|value| *value == 1.0));
    }

    #[test]
    fn convolution_keeps_cell_scale_without_sum_normalization() {
        let mut heatmap = Heatmap::new_uniform_with_dimensions(3, 3, 1.0);

        heatmap
            .apply_convolution(0.1)
            .expect("heatmap convolution should succeed");

        assert!((heatmap.map.sum() - 9.0).abs() < 1e-6);
        assert!(heatmap.map.iter().all(|value| (*value - 1.0).abs() < 1e-6));
    }

    #[test]
    fn rule_ball_sets_only_hypothesis_when_ball_is_not_free() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let parameters = SearchSuggestorParameters::default();
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.7);

        heatmap.update_with_rule_ball(
            &FilteredGameControllerState {
                game_state: FilteredGameState::Playing {
                    ball_is_free: false,
                    kick_off: true,
                },
                ..Default::default()
            },
            &field_dimensions,
            &PrimaryState::Playing,
            &parameters,
        );

        let center = heatmap.field_to_heatmap(field_dimensions, field_dimensions.center());
        for (index, value) in heatmap.map.indexed_iter() {
            if index == center {
                assert_eq!(*value, 1.0);
            } else {
                assert_eq!(*value, 0.0);
            }
        }
    }

    #[test]
    fn repeated_non_free_rule_ball_preserves_decayed_hypothesis_value() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let parameters = SearchSuggestorParameters::default();
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let filtered_game_controller_state = FilteredGameControllerState {
            game_state: FilteredGameState::Playing {
                ball_is_free: false,
                kick_off: true,
            },
            ..Default::default()
        };

        heatmap.update_with_rule_ball(
            &filtered_game_controller_state,
            &field_dimensions,
            &PrimaryState::Playing,
            &parameters,
        );
        let center = heatmap.field_to_heatmap(field_dimensions, field_dimensions.center());
        heatmap.map[center] = 0.25;

        heatmap.update_with_rule_ball(
            &filtered_game_controller_state,
            &field_dimensions,
            &PrimaryState::Playing,
            &parameters,
        );

        assert_eq!(heatmap.map[center], 0.25);
        assert!(
            heatmap
                .map
                .indexed_iter()
                .all(|(index, value)| index == center || *value == 0.0)
        );
    }

    #[test]
    fn unknown_non_free_rule_ball_hypothesis_does_not_clear_heatmap() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let parameters = SearchSuggestorParameters::default();
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.5);
        let filtered_game_controller_state = FilteredGameControllerState {
            game_state: FilteredGameState::Playing {
                ball_is_free: false,
                kick_off: false,
            },
            sub_state: Some(SubState::DirectFreeKick),
            ..Default::default()
        };

        heatmap.update_with_rule_ball(
            &filtered_game_controller_state,
            &field_dimensions,
            &PrimaryState::Playing,
            &parameters,
        );
        heatmap.recover_rule_ball_uncertainty(
            &filtered_game_controller_state,
            &field_dimensions,
            &PrimaryState::Playing,
            Duration::from_secs(5),
            Duration::from_secs(10),
        );

        assert!(heatmap.map.iter().all(|value| *value == 0.5));
    }

    #[test]
    fn changed_non_free_rule_ball_hypothesis_initializes_new_cells_to_full_value() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let parameters = SearchSuggestorParameters::default();
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let kick_off_state = FilteredGameControllerState {
            game_state: FilteredGameState::Playing {
                ball_is_free: false,
                kick_off: true,
            },
            ..Default::default()
        };
        let corner_state = FilteredGameControllerState {
            game_state: FilteredGameState::Playing {
                ball_is_free: false,
                kick_off: false,
            },
            sub_state: Some(SubState::CornerKick),
            kicking_team: Some(Team::Opponent),
            ..Default::default()
        };

        heatmap.update_with_rule_ball(
            &kick_off_state,
            &field_dimensions,
            &PrimaryState::Playing,
            &parameters,
        );
        let center = heatmap.field_to_heatmap(field_dimensions, field_dimensions.center());
        heatmap.map[center] = 0.25;

        heatmap.update_with_rule_ball(
            &corner_state,
            &field_dimensions,
            &PrimaryState::Playing,
            &parameters,
        );

        let left_corner = heatmap.field_to_heatmap(
            field_dimensions,
            field_dimensions.corner(Half::Own, Side::Left),
        );
        let right_corner = heatmap.field_to_heatmap(
            field_dimensions,
            field_dimensions.corner(Half::Own, Side::Right),
        );
        assert_eq!(heatmap.map[center], 0.0);
        assert_eq!(heatmap.map[left_corner], 1.0);
        assert_eq!(heatmap.map[right_corner], 1.0);
        assert!(heatmap.map.indexed_iter().all(|(index, value)| {
            index == left_corner || index == right_corner || *value == 0.0
        }));
    }

    #[test]
    fn non_free_rule_ball_recovery_increases_only_hypothesis_cells() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let parameters = SearchSuggestorParameters::default();
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let filtered_game_controller_state = FilteredGameControllerState {
            game_state: FilteredGameState::Playing {
                ball_is_free: false,
                kick_off: true,
            },
            ..Default::default()
        };

        heatmap.update_with_rule_ball(
            &filtered_game_controller_state,
            &field_dimensions,
            &PrimaryState::Playing,
            &parameters,
        );
        let center = heatmap.field_to_heatmap(field_dimensions, field_dimensions.center());
        let non_hypothesis = heatmap.field_to_heatmap(field_dimensions, point![1.0, 0.0]);
        heatmap.map[center] = 0.2;
        heatmap.map[non_hypothesis] = 0.8;

        heatmap.recover_rule_ball_uncertainty(
            &filtered_game_controller_state,
            &field_dimensions,
            &PrimaryState::Playing,
            Duration::from_secs(5),
            Duration::from_secs(10),
        );

        assert!((heatmap.map[center] - 0.7).abs() < 1e-6);
        assert_eq!(heatmap.map[non_hypothesis], 0.0);
    }

    #[test]
    fn rule_ball_only_increments_when_ball_is_free() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let parameters = SearchSuggestorParameters {
            rule_ball_weight_increment: 0.2,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.7);

        heatmap.update_with_rule_ball(
            &FilteredGameControllerState {
                game_state: FilteredGameState::Playing {
                    ball_is_free: true,
                    kick_off: false,
                },
                sub_state: Some(SubState::CornerKick),
                ..Default::default()
            },
            &field_dimensions,
            &PrimaryState::Playing,
            &parameters,
        );

        assert!(heatmap.map.iter().all(|value| *value >= 0.7));
        assert!(heatmap.map.iter().any(|value| (*value - 0.9).abs() < 1e-6));
    }

    #[test]
    fn perceived_ball_sets_only_ball_cell() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);

        heatmap.update_with_field_ball_position(field_dimensions, point![1.0, 0.0]);

        let ball_tile = heatmap.field_to_heatmap(field_dimensions, point![1.0, 0.0]);
        for (index, value) in heatmap.map.indexed_iter() {
            if index == ball_tile {
                assert_eq!(*value, 1.0);
            } else {
                assert_eq!(*value, 0.0);
            }
        }
    }

    #[test]
    fn team_ball_sets_only_ball_cell() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let parameters = SearchSuggestorParameters {
            team_ball_weight: 1.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);

        heatmap.update_with_team_ball(
            field_dimensions,
            TimeWrapper {
                time: ros_z::time::Time::zero(),
                inner: IncomingMessage::Hsl(HulkMessage::State(StateMessage {
                    player_number: PlayerNumber::Two,
                    pose: Pose2::new(point![0.0, 0.0], 0.0),
                    head_yaw: 0.0,
                    ball_position: Some(hsl_network_messages::BallPosition {
                        position: point![1.0, 0.0],
                        age: Duration::ZERO,
                    }),
                })),
            },
            &parameters,
        );

        let ball_tile = heatmap.field_to_heatmap(field_dimensions, point![1.0, 0.0]);
        for (index, value) in heatmap.map.indexed_iter() {
            if index == ball_tile {
                assert_eq!(*value, 1.0);
            } else {
                assert_eq!(*value, 0.0);
            }
        }
    }

    #[test]
    fn teammate_empty_head_fov_decays_visible_tiles() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let parameters = SearchSuggestorParameters {
            cells_per_meter: 2.0,
            decay_distance_factor: 0.5,
            heatmap_decay_range: 0.0..10.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap {
            map: Array2::ones((
                (field_dimensions.length * parameters.cells_per_meter).round() as usize,
                (field_dimensions.width * parameters.cells_per_meter).round() as usize,
            )),
            cells_per_meter: parameters.cells_per_meter,
            last_maximum_heatmap_position: None,
            has_decided_for_heatmap_tile: false,
            active_rule_ball_hypotheses: Vec::new(),
        };

        heatmap.decay_tiles_in_robot_fov(
            field_dimensions,
            vector![0.0, 0.0],
            FRAC_PI_2,
            parameters.decay_distance_factor,
            parameters.heatmap_decay_range.clone(),
            parameters.heatmap_full_decay_distance,
            parameters.heatmap_decay_falloff_distance,
        );

        let visible_tile = heatmap.field_to_heatmap(field_dimensions, point![0.0, 1.0]);
        let body_forward_tile = heatmap.field_to_heatmap(field_dimensions, point![1.0, 0.0]);

        assert!(heatmap.map[visible_tile] < 1.0);
        assert_eq!(heatmap.map[body_forward_tile], 1.0);
    }

    #[test]
    fn teammate_empty_head_fov_uses_obstacle_occlusion() {
        let field_dimensions = FieldDimensions {
            length: 20.0,
            width: 1.0,
            ..Default::default()
        };
        let parameters = SearchSuggestorParameters {
            decay_distance_factor: 0.5,
            heatmap_decay_range: 0.0..10.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let visible_tile = heatmap.field_to_heatmap(field_dimensions, point![1.0, 0.0]);
        let occluded_tile = heatmap.field_to_heatmap(field_dimensions, point![4.0, 0.0]);
        let obstacles = [Obstacle {
            kind: ObstacleKind::Unknown,
            position: point![2.0, 0.0],
            radius_at_foot_height: 0.4,
            radius_at_hip_height: 0.4,
        }];

        heatmap.decay_tiles_in_robot_fov_with_obstacles(
            field_dimensions,
            vector![0.0, 0.0],
            0.0,
            parameters.decay_distance_factor,
            parameters.heatmap_decay_range.clone(),
            parameters.heatmap_full_decay_distance,
            parameters.heatmap_decay_falloff_distance,
            &obstacles,
            Isometry2::identity(),
        );

        assert!(heatmap.map[visible_tile] < 1.0);
        assert_eq!(heatmap.map[occluded_tile], 1.0);
    }

    #[test]
    fn fov_decay_uses_full_strength_within_full_decay_distance() {
        let field_dimensions = FieldDimensions {
            length: 20.0,
            width: 1.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);

        heatmap.decay_tiles_in_robot_fov(
            field_dimensions,
            vector![0.0, 0.0],
            0.0,
            0.5,
            0.0..7.0,
            3.0,
            2.0,
        );

        let near_tile = heatmap.field_to_heatmap(field_dimensions, point![2.0, 0.0]);

        assert!((heatmap.map[near_tile] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fov_decay_visibility_filter_prevents_decay_for_occluded_tiles() {
        let field_dimensions = FieldDimensions {
            length: 20.0,
            width: 1.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let blocked_tile = heatmap.field_to_heatmap(field_dimensions, point![2.0, 0.0]);
        let blocked_tile_center = heatmap.heatmap_to_field(field_dimensions, blocked_tile);
        let visible_tile = heatmap.field_to_heatmap(field_dimensions, point![4.0, 0.0]);

        heatmap.decay_tiles_in_robot_fov_with_visibility_filter(
            field_dimensions,
            vector![0.0, 0.0],
            0.0,
            0.5,
            0.0..7.0,
            3.0,
            2.0,
            |tile_center| (tile_center - blocked_tile_center).norm() > 1e-6,
        );

        assert_eq!(heatmap.map[blocked_tile], 1.0);
        assert!(heatmap.map[visible_tile] < 1.0);
    }

    #[test]
    fn obstacle_occlusion_prevents_fov_decay_behind_obstacle() {
        let field_dimensions = FieldDimensions {
            length: 20.0,
            width: 1.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let visible_tile = heatmap.field_to_heatmap(field_dimensions, point![1.0, 0.0]);
        let occluded_tile = heatmap.field_to_heatmap(field_dimensions, point![4.0, 0.0]);
        let obstacles = [FieldObstacle::new(point![2.0, 0.0], 0.4)];

        heatmap.decay_tiles_in_robot_fov_with_field_obstacles(
            field_dimensions,
            vector![0.0, 0.0],
            0.0,
            0.5,
            0.0..7.0,
            3.0,
            2.0,
            &obstacles,
        );

        assert!(heatmap.map[visible_tile] < 1.0);
        assert_eq!(heatmap.map[occluded_tile], 1.0);
    }

    #[test]
    fn obstacle_occlusion_ignores_obstacle_containing_robot() {
        let field_dimensions = FieldDimensions {
            length: 20.0,
            width: 1.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let visible_tile = heatmap.field_to_heatmap(field_dimensions, point![2.0, 0.0]);
        let obstacles = [FieldObstacle::new(point![0.0, 0.0], 1.0)];

        heatmap.decay_tiles_in_robot_fov_with_field_obstacles(
            field_dimensions,
            vector![0.0, 0.0],
            0.0,
            0.5,
            0.0..7.0,
            3.0,
            2.0,
            &obstacles,
        );

        assert!(heatmap.map[visible_tile] < 1.0);
    }

    #[test]
    fn clear_obstacle_occupied_cells_clears_intersecting_heatmap_tiles() {
        let field_dimensions = FieldDimensions {
            length: 4.0,
            width: 4.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let obstacle_tile = heatmap.field_to_heatmap(field_dimensions, point![0.25, 0.25]);
        let clear_tile = heatmap.field_to_heatmap(field_dimensions, point![1.5, 1.5]);
        let obstacles = [FieldObstacle::new(point![0.0, 0.0], 0.35)];

        heatmap.clear_obstacle_occupied_cells_in_field(field_dimensions, &obstacles);

        assert_eq!(heatmap.map[obstacle_tile], 0.0);
        assert_eq!(heatmap.map[clear_tile], 1.0);
    }

    #[test]
    fn clear_obstacle_occupied_cells_ignores_ball_obstacles() {
        let field_dimensions = FieldDimensions {
            length: 4.0,
            width: 4.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let ball_tile = heatmap.field_to_heatmap(field_dimensions, point![0.25, 0.25]);
        let obstacles = [Obstacle::ball(point![0.0, 0.0], 0.35)];

        heatmap.clear_obstacle_occupied_cells(field_dimensions, &obstacles, Isometry2::identity());

        assert_eq!(heatmap.map[ball_tile], 1.0);
    }

    #[test]
    fn fov_decay_falls_off_after_full_decay_distance() {
        let field_dimensions = FieldDimensions {
            length: 20.0,
            width: 1.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);

        heatmap.decay_tiles_in_robot_fov(
            field_dimensions,
            vector![0.0, 0.0],
            0.0,
            0.5,
            0.0..7.0,
            3.0,
            2.0,
        );

        let near_tile = heatmap.field_to_heatmap(field_dimensions, point![2.0, 0.0]);
        let far_tile = heatmap.field_to_heatmap(field_dimensions, point![5.0, 0.0]);

        assert!(heatmap.map[far_tile] > heatmap.map[near_tile]);
        assert!(heatmap.map[far_tile] < 1.0);
    }

    #[test]
    fn fov_decay_does_not_apply_beyond_max_range() {
        let field_dimensions = FieldDimensions {
            length: 20.0,
            width: 1.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);

        heatmap.decay_tiles_in_robot_fov(
            field_dimensions,
            vector![0.0, 0.0],
            0.0,
            0.5,
            0.0..7.0,
            3.0,
            2.0,
        );

        let beyond_range_tile = heatmap.field_to_heatmap(field_dimensions, point![8.0, 0.0]);

        assert_eq!(heatmap.map[beyond_range_tile], 1.0);
    }

    #[test]
    fn fov_decay_does_not_decay_tile_under_robot_by_proximity() {
        let field_dimensions = FieldDimensions {
            length: 3.0,
            width: 3.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        let tile_under_robot = heatmap.field_to_heatmap(field_dimensions, point![0.0, 0.0]);

        heatmap.decay_tiles_in_robot_fov(
            field_dimensions,
            vector![0.0, 0.0],
            std::f32::consts::PI,
            0.5,
            0.0..7.0,
            3.0,
            2.0,
        );

        assert_eq!(heatmap.map[tile_under_robot], 1.0);
    }

    #[test]
    fn fov_decay_invalid_falloff_does_not_decay_beyond_full_distance() {
        let field_dimensions = FieldDimensions {
            length: 20.0,
            width: 1.0,
            ..Default::default()
        };

        for falloff_distance in [0.0, f32::NAN] {
            let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);

            heatmap.decay_tiles_in_robot_fov(
                field_dimensions,
                vector![0.0, 0.0],
                0.0,
                0.5,
                0.0..7.0,
                3.0,
                falloff_distance,
            );

            let near_tile = heatmap.field_to_heatmap(field_dimensions, point![2.0, 0.0]);
            let far_tile = heatmap.field_to_heatmap(field_dimensions, point![5.0, 0.0]);

            assert!((heatmap.map[near_tile] - 0.5).abs() < 1e-6);
            assert_eq!(heatmap.map[far_tile], 1.0);
            assert!(heatmap.map.iter().all(|value| !value.is_nan()));
        }
    }

    #[test]
    fn voronoi_filtered_search_position_prefers_owned_tile_over_global_maximum() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.0);

        let owned_tile = heatmap.field_to_heatmap(field_dimensions, point![-3.0, 0.0]);
        let teammate_tile = heatmap.field_to_heatmap(field_dimensions, point![3.0, 0.0]);
        heatmap.map[owned_tile] = 0.4;
        heatmap.map[teammate_tile] = 0.9;
        heatmap.update_selected_target(0.01, 0.5);

        let sites = [
            (Pose2::new(point![-4.0, 0.0], 0.0), PlayerNumber::Four),
            (Pose2::new(point![3.0, 0.0], 0.0), PlayerNumber::Five),
        ];

        let selected = heatmap
            .voronoi_filtered_search_position(field_dimensions, PlayerNumber::Four, &sites, 0.01)
            .expect("owned valid tile should be selected");

        assert_eq!(
            selected,
            heatmap.heatmap_to_field(field_dimensions, owned_tile)
        );
        assert_ne!(
            selected,
            heatmap.heatmap_to_field(field_dimensions, teammate_tile)
        );
    }

    #[test]
    fn voronoi_filtered_search_position_under_robot_returns_nearest_tile_corner() {
        let field_dimensions = FieldDimensions {
            length: 4.0,
            width: 4.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.0);

        let selected_tile = heatmap.field_to_heatmap(field_dimensions, point![0.25, 0.25]);
        heatmap.map[selected_tile] = 1.0;

        let sites = [(Pose2::new(point![0.25, 0.25], 0.0), PlayerNumber::Four)];

        let selected = heatmap
            .voronoi_filtered_search_position(field_dimensions, PlayerNumber::Four, &sites, 0.01)
            .expect("owned valid tile should be selected");

        assert_eq!(selected, point![0.0, 0.0]);
    }

    #[test]
    fn voronoi_filtered_search_position_under_robot_at_field_edge_stays_inside_field() {
        let field_dimensions = FieldDimensions {
            length: 4.0,
            width: 4.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.0);

        let selected_tile = heatmap.field_to_heatmap(field_dimensions, point![-1.75, 0.25]);
        heatmap.map[selected_tile] = 1.0;

        let sites = [(Pose2::new(point![-2.0, 0.25], 0.0), PlayerNumber::Four)];

        let target = heatmap
            .voronoi_filtered_search_target(field_dimensions, PlayerNumber::Four, &sites, 0.01)
            .expect("owned valid tile should be selected");

        assert_eq!(target.position, point![-1.75, 0.0]);
        assert_eq!(
            target.look_at,
            Some(heatmap.heatmap_to_field(field_dimensions, selected_tile))
        );
    }

    #[test]
    fn voronoi_filtered_search_target_under_robot_looks_at_tile_center() {
        let field_dimensions = FieldDimensions {
            length: 4.0,
            width: 4.0,
            ..Default::default()
        };
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.0);

        let selected_tile = heatmap.field_to_heatmap(field_dimensions, point![0.25, 0.25]);
        let selected_tile_center = heatmap.heatmap_to_field(field_dimensions, selected_tile);
        heatmap.map[selected_tile] = 1.0;

        let sites = [(Pose2::new(point![0.25, 0.25], 0.0), PlayerNumber::Four)];

        let target = heatmap
            .voronoi_filtered_search_target(field_dimensions, PlayerNumber::Four, &sites, 0.01)
            .expect("owned valid tile should be selected");

        assert_eq!(target.position, point![0.0, 0.0]);
        assert_eq!(target.look_at, Some(selected_tile_center));
    }

    #[test]
    fn voronoi_filtered_search_position_keeps_previous_tile_with_hysteresis() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.0);

        let previous_tile = heatmap.field_to_heatmap(field_dimensions, point![-4.0, 0.0]);
        let slightly_better_tile = heatmap.field_to_heatmap(field_dimensions, point![-2.0, 0.0]);
        heatmap.map[previous_tile] = 0.99;
        heatmap.map[slightly_better_tile] = 1.0;

        let sites = [
            (Pose2::new(point![-3.0, 0.0], 0.0), PlayerNumber::Four),
            (Pose2::new(point![3.0, 0.0], 0.0), PlayerNumber::Five),
        ];

        let previous_position = heatmap.heatmap_to_field(field_dimensions, previous_tile);
        let selected = heatmap
            .voronoi_filtered_search_position_with_hysteresis(
                field_dimensions,
                PlayerNumber::Four,
                &sites,
                Some(previous_position),
                0.5,
                0.8,
            )
            .expect("previous owned tile should be kept");

        assert_eq!(selected, previous_position);
    }

    #[test]
    fn voronoi_filtered_search_position_does_not_fallback_to_other_players_tile() {
        let field_dimensions = FieldDimensions::SPL_2025;
        let mut heatmap = Heatmap::new_uniform(field_dimensions, 1.0);
        heatmap.map.fill(0.0);

        let teammate_tile = heatmap.field_to_heatmap(field_dimensions, point![3.0, 0.0]);
        heatmap.map[teammate_tile] = 0.9;
        heatmap.update_selected_target(0.01, 0.5);

        let sites = [
            (Pose2::new(point![-3.0, 0.0], 0.0), PlayerNumber::Four),
            (Pose2::new(point![3.0, 0.0], 0.0), PlayerNumber::Five),
        ];

        assert!(
            heatmap
                .voronoi_filtered_search_position(
                    field_dimensions,
                    PlayerNumber::Four,
                    &sites,
                    0.01
                )
                .is_none()
        );
    }
}
