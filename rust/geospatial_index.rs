//! Geospatial Search over observations on and above the Earth, like photos, video frames, drone
//! footage and sensor fixes.
//!
//! Every entry is one [`GeospatialObservation`] of eleven doubles in natural units, a field no
//! device recorded left NaN. Every verb comes from [`GeospatialVerbs`], written once over the
//! receiver: a [`GeospatialIndex`] blocks on the calling thread, `index.on(&executor)` blocks on an
//! executor, and `index.deferred(&executor)` answers a [`Deferred`](crate::Deferred) from the same
//! verbs given owned arguments.
//!
//! File: rust/geospatial_index.rs
//! Author: Ash Vardanian
//! Date: November 16, 2026

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types)]

use core::borrow::BorrowMut;
use core::ffi::{c_int, c_void};
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr;
use std::path::Path;

use crate::dense_index::{
    DenseBatch, DenseReachability, DenseSampleConfig, DenseSaveBufferConfig, DenseSaveConfig, DenseTraversalStats,
    Monolith, RefineReport, Sample, usearch_phase_progress_t, usearch_reachability_t, usearch_refine_options_t,
    usearch_refine_report_t, usearch_traversal_stats_t,
};
use crate::types::{
    Accepts, Arena, Blocking, CPUClass, Completion, Deferring, Executor, GPUClass, Key, Matches, On, PhaseProgress,
    Predicate, ProgressCallback, QueryMatches, ReshardHandles, Result, Scalar, Status, Traversal, arena_slice, check,
    lower_predicate, lower_progress, nvme_options, output_slot, path_view, path_views, ram_options, spelling,
    usearch_arena_t, usearch_bytes_view_t, usearch_completion_blocking_k, usearch_completion_t, usearch_cpu_class_name,
    usearch_cpu_class_t, usearch_executor_t, usearch_gpu_class_name, usearch_gpu_class_t, usearch_key_t,
    usearch_keys_tape_view_t, usearch_keys_view_t, usearch_load_path_name, usearch_load_path_t,
    usearch_matches_tape_view_t, usearch_matches_view_t, usearch_nvme_options_t, usearch_outcome_t,
    usearch_predicate_t, usearch_progress_t, usearch_ram_options_t, usearch_rows_view_t, usearch_traversal_t,
    usearch_vram_options_t, vram_options,
};

// region: Raw Declarations

pub type usearch_geospatial_index_t = *mut c_void;
pub type usearch_geospatial_cursor_t = *mut c_void;
pub type usearch_geospatial_refine_plan_t = *mut c_void;
pub type usearch_geospatial_reshard_plan_t = *mut c_void;
pub type usearch_geospatial_relation_t = c_int;
pub type usearch_geospatial_observation_t = GeospatialObservation;

pub type usearch_geospatial_terrain_callback_t = Option<
    unsafe extern "C" fn(
        first_row: usize,
        first_column: usize,
        rows: usize,
        columns: usize,
        heights: *mut f32,
        heights_stride: usize,
        state: *mut c_void,
    ),
>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_geospatial_terrain_t {
    pub rows: usize,
    pub columns: usize,
    pub north_latitude_degrees: f64,
    pub west_longitude_degrees: f64,
    pub row_step_degrees: f64,
    pub column_step_degrees: f64,
    pub heights: *const f32,
    pub heights_stride: usize,
    pub callback: usearch_geospatial_terrain_callback_t,
    pub state: *mut c_void,
    pub path: usearch_bytes_view_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_geospatial_tolerance_t {
    pub metres: f64,
    pub seconds: f64,
}

#[repr(C)]
pub struct usearch_geospatial_init_ram_t {
    pub index: *mut usearch_geospatial_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub terrain: usearch_geospatial_terrain_t,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub ram: usearch_ram_options_t,
}

#[repr(C)]
pub struct usearch_geospatial_init_vram_t {
    pub index: *mut usearch_geospatial_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub terrain: usearch_geospatial_terrain_t,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub vram: usearch_vram_options_t,
}

#[repr(C)]
pub struct usearch_geospatial_load_ram_t {
    pub index: *mut usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
    pub values_per_key: usize,
    pub terrain: usearch_geospatial_terrain_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_geospatial_load_ram_buffer_t {
    pub index: *mut usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
    pub values_per_key: usize,
    pub terrain: usearch_geospatial_terrain_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_geospatial_load_vram_t {
    pub index: *mut usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub vram: usearch_vram_options_t,
    pub values_per_key: usize,
    pub terrain: usearch_geospatial_terrain_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_geospatial_load_vram_buffer_t {
    pub index: *mut usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub vram: usearch_vram_options_t,
    pub values_per_key: usize,
    pub terrain: usearch_geospatial_terrain_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_geospatial_view_ram_t {
    pub index: *mut usearch_geospatial_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub path: usearch_bytes_view_t,
    pub terrain: usearch_geospatial_terrain_t,
    pub ram: usearch_ram_options_t,
}

#[repr(C)]
pub struct usearch_geospatial_view_nvme_t {
    pub index: *mut usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub nvme: usearch_nvme_options_t,
    pub terrain: usearch_geospatial_terrain_t,
}

#[repr(C)]
pub struct usearch_geospatial_reserve_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub entries: usize,
}

#[repr(C)]
pub struct usearch_geospatial_clear_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
}

#[repr(C)]
pub struct usearch_geospatial_insert_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub keys: usearch_keys_view_t,
    pub observations: *const usearch_geospatial_observation_t,
    pub expansion_factor: usize,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_geospatial_refine_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
    pub plan: *mut usearch_geospatial_refine_plan_t,
    pub phase: *mut usearch_phase_progress_t,
}

#[repr(C)]
pub struct usearch_geospatial_refine_plan_finish_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub plan: usearch_geospatial_refine_plan_t,
    pub report: *mut usearch_refine_report_t,
}

#[repr(C)]
pub struct usearch_geospatial_health_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub reachability: *mut usearch_reachability_t,
}

#[repr(C)]
pub struct usearch_geospatial_search_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub queries: *const usearch_geospatial_observation_t,
    pub relation: usearch_geospatial_relation_t,
    pub tolerance: usearch_geospatial_tolerance_t,
    pub matches: *mut usearch_matches_view_t,
    pub arena: usearch_arena_t,
    pub wanted: usize,
    pub expansion_factor: usize,
    pub traversal: usearch_traversal_t,
    pub predicate: usearch_predicate_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_geospatial_within_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub queries: *const usearch_geospatial_observation_t,
    pub relation: usearch_geospatial_relation_t,
    pub tolerance: usearch_geospatial_tolerance_t,
    pub matches: *mut usearch_matches_tape_view_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
    pub traversal: usearch_traversal_t,
    pub predicate: usearch_predicate_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

pub type usearch_geospatial_key_stats_t = GeospatialKeyStats;

#[repr(C)]
pub struct usearch_geospatial_count_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub key: usearch_key_t,
    pub key_stats: *mut usearch_geospatial_key_stats_t,
}

#[repr(C)]
pub struct usearch_geospatial_get_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub key: usearch_key_t,
    pub observations: *mut *const usearch_geospatial_observation_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
}

#[repr(C)]
pub struct usearch_geospatial_reshard_t {
    pub sources_count: usize,
    pub sources: *const usearch_geospatial_index_t,
    pub targets_count: usize,
    pub targets: *const usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub seed: u64,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
    pub plan: *mut usearch_geospatial_reshard_plan_t,
    pub phase: *mut usearch_phase_progress_t,
}

#[repr(C)]
pub struct usearch_geospatial_reshard_plan_finish_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub plan: usearch_geospatial_reshard_plan_t,
    pub report: *mut usearch_refine_report_t,
}

#[repr(C)]
pub struct usearch_geospatial_cursor_init_t {
    pub index: usearch_geospatial_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub starts_count: usize,
    pub starts: *const usearch_key_t,
    pub cursor: *mut usearch_geospatial_cursor_t,
}

#[repr(C)]
pub struct usearch_geospatial_cursor_next_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub cursor: usearch_geospatial_cursor_t,
    pub keys: *mut usearch_keys_view_t,
    pub links: *mut usearch_keys_tape_view_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
    pub wanted: usize,
}

#[repr(C)]
pub struct usearch_geospatial_sample_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub shown_count: usize,
    pub shown: usearch_keys_view_t,
    pub shown_coordinates: usearch_rows_view_t,
    pub focus_count: usize,
    pub focus: usearch_keys_view_t,
    pub keys: *mut usearch_keys_view_t,
    pub entry_ordinals: *mut *const usize,
    pub parents: *mut usearch_matches_view_t,
    pub coordinates: *mut usearch_rows_view_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
    pub wanted: usize,
    pub seed: u64,
    pub expansion_factor: usize,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_geospatial_save_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_geospatial_save_buffer_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub monolith: *mut usearch_bytes_view_t,
    pub arena: usearch_arena_t,
    pub progress: usearch_progress_t,
}

pub type usearch_geospatial_metadata_t = GeospatialMetadata;

#[repr(C)]
pub struct usearch_geospatial_metadata_read_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub metadata: *mut usearch_geospatial_metadata_t,
}

#[repr(C)]
pub struct usearch_geospatial_metadata_read_buffer_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub metadata: *mut usearch_geospatial_metadata_t,
}

pub type usearch_geospatial_index_description_t = GeospatialIndexDescription;

#[repr(C)]
pub struct usearch_geospatial_describe_t {
    pub index: usearch_geospatial_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub description: *mut usearch_geospatial_index_description_t,
}

unsafe extern "C" {
    pub fn usearch_geospatial_init_ram(arguments: *mut usearch_geospatial_init_ram_t);
    pub fn usearch_geospatial_init_vram(arguments: *mut usearch_geospatial_init_vram_t);
    pub fn usearch_geospatial_load_ram(arguments: *mut usearch_geospatial_load_ram_t);
    pub fn usearch_geospatial_load_ram_buffer(arguments: *mut usearch_geospatial_load_ram_buffer_t);
    pub fn usearch_geospatial_load_vram(arguments: *mut usearch_geospatial_load_vram_t);
    pub fn usearch_geospatial_load_vram_buffer(arguments: *mut usearch_geospatial_load_vram_buffer_t);
    pub fn usearch_geospatial_view_ram(arguments: *mut usearch_geospatial_view_ram_t);
    pub fn usearch_geospatial_view_nvme(arguments: *mut usearch_geospatial_view_nvme_t);
    pub fn usearch_geospatial_free(index: usearch_geospatial_index_t);

    pub fn usearch_geospatial_reserve(arguments: *mut usearch_geospatial_reserve_t);
    pub fn usearch_geospatial_clear(arguments: *mut usearch_geospatial_clear_t);
    pub fn usearch_geospatial_insert(arguments: *mut usearch_geospatial_insert_t);
    pub fn usearch_geospatial_refine(arguments: *mut usearch_geospatial_refine_t);
    pub fn usearch_geospatial_refine_plan_finish(arguments: *mut usearch_geospatial_refine_plan_finish_t);
    pub fn usearch_geospatial_health(arguments: *mut usearch_geospatial_health_t);
    pub fn usearch_geospatial_search(arguments: *mut usearch_geospatial_search_t);
    pub fn usearch_geospatial_within(arguments: *mut usearch_geospatial_within_t);
    pub fn usearch_geospatial_count(arguments: *mut usearch_geospatial_count_t);
    pub fn usearch_geospatial_get(arguments: *mut usearch_geospatial_get_t);
    pub fn usearch_geospatial_reshard(arguments: *mut usearch_geospatial_reshard_t);
    pub fn usearch_geospatial_reshard_plan_finish(arguments: *mut usearch_geospatial_reshard_plan_finish_t);
    pub fn usearch_geospatial_sample(arguments: *mut usearch_geospatial_sample_t);

    pub fn usearch_geospatial_cursor_init(arguments: *mut usearch_geospatial_cursor_init_t);
    pub fn usearch_geospatial_cursor_next(arguments: *mut usearch_geospatial_cursor_next_t);
    pub fn usearch_geospatial_cursor_free(cursor: usearch_geospatial_cursor_t);

    pub fn usearch_geospatial_save(arguments: *mut usearch_geospatial_save_t);
    pub fn usearch_geospatial_save_buffer(arguments: *mut usearch_geospatial_save_buffer_t);
    pub fn usearch_geospatial_metadata_read(arguments: *mut usearch_geospatial_metadata_read_t);
    pub fn usearch_geospatial_metadata_read_buffer(arguments: *mut usearch_geospatial_metadata_read_buffer_t);
    pub fn usearch_geospatial_describe(arguments: *mut usearch_geospatial_describe_t);
}

// endregion: Raw Declarations

// region: Geometry

/// One observation in natural units: a position in WGS-84 degrees and metres above the ellipsoid,
/// its accuracy, the device's view cone, how far it saw, and the Unix seconds it covers.
///
/// A field no device recorded is NaN, never zero, and only widens the tests that read it.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GeospatialObservation {
    /// Degrees north of the equator.
    pub latitude_degrees: f64,
    /// Degrees east of the prime meridian.
    pub longitude_degrees: f64,
    /// Metres above the WGS-84 ellipsoid.
    pub altitude_metres: f64,
    /// One standard deviation of the horizontal position, in metres.
    pub accuracy_metres: f64,
    /// The view cone's axis, degrees clockwise from north.
    pub heading_degrees: f64,
    /// The view cone's axis, degrees above the horizon.
    pub pitch_degrees: f64,
    /// One standard deviation of the heading, in degrees.
    pub heading_accuracy_degrees: f64,
    /// The view cone's full opening angle, in degrees.
    pub field_of_view_degrees: f64,
    /// How far the device saw, in metres.
    pub range_metres: f64,
    /// The first Unix second the observation covers.
    pub begin_seconds: f64,
    /// The last Unix second the observation covers.
    pub end_seconds: f64,
}

impl GeospatialObservation {
    /// A position alone, every other field unrecorded.
    ///
    /// Rust-only.
    pub fn at(latitude_degrees: f64, longitude_degrees: f64) -> Self {
        GeospatialObservation { latitude_degrees, longitude_degrees, ..GeospatialObservation::default() }
    }
}

impl Default for GeospatialObservation {
    /// Every field unrecorded.
    fn default() -> Self {
        GeospatialObservation {
            latitude_degrees: f64::NAN,
            longitude_degrees: f64::NAN,
            altitude_metres: f64::NAN,
            accuracy_metres: f64::NAN,
            heading_degrees: f64::NAN,
            pitch_degrees: f64::NAN,
            heading_accuracy_degrees: f64::NAN,
            field_of_view_degrees: f64::NAN,
            range_metres: f64::NAN,
            begin_seconds: f64::NAN,
            end_seconds: f64::NAN,
        }
    }
}

/// What a query asks of an entry beside closeness.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum GeospatialRelation {
    /// Within the radius, widened by three standard deviations of both positions combined.
    #[default]
    Near = 1,
    /// Near, with a clear line of sight over the handle's terrain; refused by a handle without one.
    Visible = 2,
    /// Near, and the entry's view cone holds the query: which cameras see this place.
    Sees = 3,
    /// Near, and the query's view cone holds the entry: what this camera sees.
    Seen = 4,
}

/// Fills `rows` by `columns` heights of a terrain starting at `first_row` and `first_column`, in
/// metres above the ellipsoid, rows `stride` heights apart; a cell left NaN never blocks a sight
/// line. It runs on the executor's threads together, and a panic inside it aborts the process.
pub type GeospatialTerrainFill =
    fn(first_row: usize, first_column: usize, rows: usize, columns: usize, heights: &mut [f32], stride: usize);

/// # Safety
/// `state` must be a [`GeospatialTerrainFill`], and `heights` must address `rows` rows of
/// `heights_stride` bytes, the last holding `columns` heights.
unsafe extern "C" fn terrain_trampoline(
    first_row: usize,
    first_column: usize,
    rows: usize,
    columns: usize,
    heights: *mut f32,
    heights_stride: usize,
    state: *mut c_void,
) {
    unsafe {
        let fill: GeospatialTerrainFill = core::mem::transmute::<*mut c_void, GeospatialTerrainFill>(state);
        let stride = heights_stride / core::mem::size_of::<f32>();
        let length = match rows {
            0 => 0,
            rows => (rows - 1) * stride + columns,
        };
        fill(first_row, first_column, rows, columns, core::slice::from_raw_parts_mut(heights, length), stride);
    }
}

/// The elevation every line of sight is tested over: a north-up grid of heights in metres above
/// the WGS-84 ellipsoid, given by borrowed heights or a fill function, which the handle reads for
/// as long as it lives. A checkpoint never carries it, so every load and view names it again.
#[derive(Clone, Copy, Debug)]
pub struct GeospatialTerrain {
    /// The grid's row count; rows run south from the north edge.
    pub rows: usize,
    /// The grid's column count; columns run east from the west edge.
    pub columns: usize,
    /// The latitude of the centre of cell `[0, 0]`, in degrees.
    pub north_latitude_degrees: f64,
    /// The longitude of the centre of cell `[0, 0]`, in degrees.
    pub west_longitude_degrees: f64,
    /// The step between rows, in degrees, positive.
    pub row_step_degrees: f64,
    /// The step between columns, in degrees, positive.
    pub column_step_degrees: f64,
    /// The whole grid, row-major and packed.
    pub heights: Option<&'static [f32]>,
    /// Or a function filling windows of the grid on demand; VRAM handles refuse it.
    pub fill: Option<GeospatialTerrainFill>,
}

impl GeospatialTerrain {
    fn lower(&self) -> usearch_geospatial_terrain_t {
        usearch_geospatial_terrain_t {
            rows: self.rows,
            columns: self.columns,
            north_latitude_degrees: self.north_latitude_degrees,
            west_longitude_degrees: self.west_longitude_degrees,
            row_step_degrees: self.row_step_degrees,
            column_step_degrees: self.column_step_degrees,
            heights: self.heights.map_or(ptr::null(), <[f32]>::as_ptr),
            heights_stride: 0,
            callback: self.fill.map(|_| terrain_trampoline as _),
            state: self.fill.map_or(ptr::null_mut(), |fill| fill as *mut c_void),
            path: usearch_bytes_view_t::default(),
        }
    }
}

/// Lowers an optional terrain as the C interface reads it, zeroed meaning none.
fn lower_terrain(terrain: Option<&GeospatialTerrain>) -> usearch_geospatial_terrain_t {
    match terrain {
        Some(terrain) => terrain.lower(),
        None => usearch_geospatial_terrain_t {
            rows: 0,
            columns: 0,
            north_latitude_degrees: 0.0,
            west_longitude_degrees: 0.0,
            row_step_degrees: 0.0,
            column_step_degrees: 0.0,
            heights: ptr::null(),
            heights_stride: 0,
            callback: None,
            state: ptr::null_mut(),
            path: usearch_bytes_view_t::default(),
        },
    }
}

/// Construction options of a geospatial index; [`GeospatialIndex::init_ram`] and
/// [`GeospatialIndex::init_vram`] both take it.
#[derive(Clone, Debug, Default)]
pub struct GeospatialOptions {
    /// The elevation sight lines clear; `None` refuses [`GeospatialRelation::Visible`].
    pub terrain: Option<GeospatialTerrain>,
    /// Graph links per node; zero derives it.
    pub connectivity: usize,
    /// Upper bound on observations sharing one key; zero is unbounded.
    pub values_per_key: usize,
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; `None` means no ceiling.
    pub cpu_class_ceiling: Option<CPUClass>,
    /// The newest class dispatch may pick on the devices; `None` means no ceiling.
    pub gpu_class_ceiling: Option<GPUClass>,
}

/// The graph a checkpoint froze - everything a reader learns without opening it.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct GeospatialMetadata {
    /// Connections per node the graph was built with.
    pub connectivity: usize,
}

/// A live index's frozen metadata and current counters, each read once, so under concurrent
/// insertions they may come from slightly different moments.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct GeospatialIndexDescription {
    /// The frozen connectivity, as a checkpoint reports it.
    pub metadata: GeospatialMetadata,
    /// Stored entries.
    pub size: usize,
    /// Entries hosted without growing again.
    pub capacity: usize,
    /// Bytes the handle holds, payload storage included; traversal scratch is the executor's.
    pub memory_usage: usize,
    /// The CPU class dispatch picked, spelled by [`GeospatialIndexDescription::cpu_class`].
    cpu_class: usearch_cpu_class_t,
    /// The GPU class dispatch picked, spelled by [`GeospatialIndexDescription::gpu_class`].
    gpu_class: usearch_gpu_class_t,
    /// Live shards.
    pub shards_count: usize,
    /// How the last load read the parts, spelled by [`GeospatialIndexDescription::load_path`].
    load_path: usearch_load_path_t,
}

impl GeospatialIndexDescription {
    /// The CPU class dispatch picked on this machine, as [`CPUClass`] spells it.
    pub fn cpu_class(&self) -> &'static str {
        spelling(usearch_cpu_class_name, self.cpu_class)
    }

    /// The GPU class dispatch picked on the index's devices; "unknown" for a host index.
    pub fn gpu_class(&self) -> &'static str {
        spelling(usearch_gpu_class_name, self.gpu_class)
    }

    /// How the last load from a path read the parts: "none", "host-copy" or "cufile".
    pub fn load_path(&self) -> &'static str {
        spelling(usearch_load_path_name, self.load_path)
    }
}

/// What one key holds: its observations.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct GeospatialKeyStats {
    /// Observations stored under the key; zero means the index holds none.
    pub entries: usize,
}

// endregion: Geometry

// region: Configs

/// A checkpoint path and how it is copied into a new index in host memory; see
/// [`GeospatialIndex::load_ram`].
pub struct GeospatialLoadRamConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The executor the read fans out over; `None` reads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// NUMA node IDs to place the parts on; empty derives every node the process may use.
    pub memory_domains: &'config [i32],
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; `None` means no ceiling.
    pub cpu_class_ceiling: Option<CPUClass>,
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    pub values_per_key: usize,
    /// The elevation sight lines clear, which no checkpoint records.
    pub terrain: Option<GeospatialTerrain>,
    /// Reports the shards placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for GeospatialLoadRamConfig<'config> {
    fn from(path: &'config Location) -> Self {
        GeospatialLoadRamConfig {
            path: path.as_ref(),
            executor: None,
            memory_domains: &[],
            cpu_class_ceiling: None,
            values_per_key: 0,
            terrain: None,
            progress: None,
            progress_steps: 0,
        }
    }
}

/// A checkpoint path and how it is uploaded into a new index on devices; see
/// [`GeospatialIndex::load_vram`].
pub struct GeospatialLoadVramConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The executor the upload fans out over; `None` reads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// CUDA device ordinals to place the parts on; empty derives one free device.
    pub devices: &'config [i32],
    /// The newest class dispatch may pick on the devices; `None` means no ceiling.
    pub gpu_class_ceiling: Option<GPUClass>,
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    pub values_per_key: usize,
    /// The elevation sight lines clear, which no checkpoint records.
    pub terrain: Option<GeospatialTerrain>,
    /// Reports the shards placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for GeospatialLoadVramConfig<'config> {
    fn from(path: &'config Location) -> Self {
        GeospatialLoadVramConfig {
            path: path.as_ref(),
            executor: None,
            devices: &[],
            gpu_class_ceiling: None,
            values_per_key: 0,
            terrain: None,
            progress: None,
            progress_steps: 0,
        }
    }
}

/// A serialized monolith and how it is copied into a new index in host memory; see
/// [`GeospatialIndex::load_ram_buffer`].
pub struct GeospatialLoadRamBufferConfig<'config> {
    /// One serialized monolith, or a manifest leading its parts.
    pub buffer: &'config [u8],
    /// The executor the read fans out over; `None` reads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// NUMA node IDs to place shards on; empty derives every node the process may use.
    pub memory_domains: &'config [i32],
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; `None` means no ceiling.
    pub cpu_class_ceiling: Option<CPUClass>,
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    pub values_per_key: usize,
    /// The elevation sight lines clear, which no checkpoint records.
    pub terrain: Option<GeospatialTerrain>,
    /// Reports the bytes placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config> From<&'config [u8]> for GeospatialLoadRamBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        GeospatialLoadRamBufferConfig {
            buffer,
            executor: None,
            memory_domains: &[],
            cpu_class_ceiling: None,
            values_per_key: 0,
            terrain: None,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for GeospatialLoadRamBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self {
        GeospatialLoadRamBufferConfig::from(&**bytes)
    }
}

/// A serialized monolith and how it is uploaded into a new index on devices; see
/// [`GeospatialIndex::load_vram_buffer`].
pub struct GeospatialLoadVramBufferConfig<'config> {
    /// One serialized monolith, or a manifest leading its parts.
    pub buffer: &'config [u8],
    /// The executor the upload fans out over; `None` reads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// CUDA device ordinals to place shards on; empty derives one free device.
    pub devices: &'config [i32],
    /// The newest class dispatch may pick on the devices; `None` means no ceiling.
    pub gpu_class_ceiling: Option<GPUClass>,
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    pub values_per_key: usize,
    /// The elevation sight lines clear, which no checkpoint records.
    pub terrain: Option<GeospatialTerrain>,
    /// Reports the bytes placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config> From<&'config [u8]> for GeospatialLoadVramBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        GeospatialLoadVramBufferConfig {
            buffer,
            executor: None,
            devices: &[],
            gpu_class_ceiling: None,
            values_per_key: 0,
            terrain: None,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for GeospatialLoadVramBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self {
        GeospatialLoadVramBufferConfig::from(&**bytes)
    }
}

/// A checkpoint path and the drives it is served from in place; see [`GeospatialIndex::view_nvme`].
pub struct GeospatialViewNvmeConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The executor the open fans out over; `None` opens on the calling thread.
    pub executor: Option<&'config Executor>,
    /// Devices or directories to serve the parts from; empty serves each where its manifest points.
    pub paths: &'config [&'config Path],
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; `None` means no ceiling.
    pub cpu_class_ceiling: Option<CPUClass>,
    /// The elevation sight lines clear, which no checkpoint records.
    pub terrain: Option<GeospatialTerrain>,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for GeospatialViewNvmeConfig<'config> {
    fn from(path: &'config Location) -> Self {
        GeospatialViewNvmeConfig {
            path: path.as_ref(),
            executor: None,
            paths: &[],
            cpu_class_ceiling: None,
            terrain: None,
        }
    }
}

/// The observations an insertion adds and the knobs of its walks; see [`GeospatialVerbs::insert`].
pub struct GeospatialInsertConfig<'config, Observations = &'config [GeospatialObservation]> {
    /// One observation per key.
    pub observations: Observations,
    /// The breadth of the candidates the walk explores; zero derives it.
    pub expansion_factor: usize,
    /// Reports the entries inserted; every entry is claimed before the first report, so a `Break`
    /// is ignored.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl<Observations: AsRef<[GeospatialObservation]>> From<Observations> for GeospatialInsertConfig<'_, Observations> {
    fn from(observations: Observations) -> Self {
        GeospatialInsertConfig { observations, expansion_factor: 0, progress: None, progress_steps: 0, stats: None }
    }
}

/// The matches a search keeps and the knobs of its walk; see [`GeospatialVerbs::search`].
pub struct GeospatialSearchConfig<'config> {
    /// Matches kept per query.
    pub wanted: usize,
    /// What each match must be to the query beside close.
    pub relation: GeospatialRelation,
    /// The radius in metres, zero ranking by distance alone.
    pub metres: f64,
    /// Seconds widening the query's interval on both ends.
    pub seconds: f64,
    /// The breadth of the candidates the walk explores; zero derives it.
    pub expansion_factor: usize,
    /// Walks the graph by default; [`Traversal::Scan`] scores every entry exactly.
    pub traversal: Traversal,
    /// Admits only the keys it approves; the rest still route the walk.
    pub predicate: Option<&'config Predicate<'config>>,
    /// Reports the queries resolved and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl From<usize> for GeospatialSearchConfig<'_> {
    fn from(wanted: usize) -> Self {
        GeospatialSearchConfig {
            wanted,
            relation: GeospatialRelation::Near,
            metres: 0.0,
            seconds: 0.0,
            expansion_factor: 0,
            traversal: Traversal::Walk,
            predicate: None,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// Every match a relation admits inside a tolerance and the knobs of the walk; see
/// [`GeospatialVerbs::within`].
pub struct GeospatialWithinConfig<'config> {
    /// What each match must be to the query beside close.
    pub relation: GeospatialRelation,
    /// The radius in metres.
    pub metres: f64,
    /// Seconds widening the query's interval on both ends.
    pub seconds: f64,
    /// Walks the graph by default; [`Traversal::Scan`] scores every entry exactly.
    pub traversal: Traversal,
    /// Admits only the keys it approves; the rest still route the walk.
    pub predicate: Option<&'config Predicate<'config>>,
    /// Reports the queries resolved and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl From<f64> for GeospatialWithinConfig<'_> {
    /// Every entry near the query within `metres`.
    fn from(metres: f64) -> Self {
        GeospatialWithinConfig {
            relation: GeospatialRelation::Near,
            metres,
            seconds: 0.0,
            traversal: Traversal::Walk,
            predicate: None,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// How [`GeospatialVerbs::refine`] runs; a blocking call borrows its plan, a deferred one owns it.
pub struct GeospatialRefineConfig<'config, Plan = &'config mut GeospatialRefinePlan> {
    /// The breadth of any walk beside the round; zero derives it.
    pub expansion_factor: usize,
    /// Bytes a loaded NVMe handle holds for the one part it rewrites; zero derives it.
    pub memory_limit: usize,
    /// Reports every phase's units at its barriers and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
    /// A plan a stop leaves live and the next call continues, opened when fresh; `None` runs one
    /// round to its end.
    pub plan: Option<Plan>,
    /// Where the call stopped, [`Phase::Done`](crate::Phase::Done) once its round ended.
    pub phase: Option<&'config mut PhaseProgress>,
}

impl Default for GeospatialRefineConfig<'_> {
    fn default() -> Self {
        GeospatialRefineConfig::with_plan(None)
    }
}

impl From<GeospatialRefinePlan> for GeospatialRefineConfig<'_, GeospatialRefinePlan> {
    /// A deferred call owning `plan`, with every other knob derived.
    fn from(plan: GeospatialRefinePlan) -> Self {
        GeospatialRefineConfig::with_plan(Some(plan))
    }
}

impl<Plan> GeospatialRefineConfig<'_, Plan> {
    fn with_plan(plan: Option<Plan>) -> Self {
        GeospatialRefineConfig {
            expansion_factor: 0,
            memory_limit: 0,
            progress: None,
            progress_steps: 0,
            stats: None,
            plan,
            phase: None,
        }
    }
}

/// The handles a reshard moves between and its knobs; see [`GeospatialVerbs::reshard`].
pub struct GeospatialReshardConfig<'config, Plan = &'config mut GeospatialReshardPlan> {
    /// The handles whose entries move, each named once.
    pub sources: &'config [&'config GeospatialIndex],
    /// The handles the entries move into, each a source itself or empty.
    pub targets: &'config [&'config GeospatialIndex],
    /// The breadth of any walk beside the closing rounds; zero derives it.
    pub expansion_factor: usize,
    /// Bytes a loaded NVMe target holds for the one part it rewrites; zero derives it.
    pub memory_limit: usize,
    /// Picks the part roots of a source cut among several targets.
    pub seed: u64,
    /// Reports the closing rounds' phases and stops them wherever it says to, never a move.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
    /// A plan a stop leaves live and the next call continues, opened when fresh; `None` moves and
    /// runs to the end.
    pub plan: Option<Plan>,
    /// Where the call stopped, [`Phase::Done`](crate::Phase::Done) once its rounds ended.
    pub phase: Option<&'config mut PhaseProgress>,
}

impl Default for GeospatialReshardConfig<'_> {
    fn default() -> Self {
        GeospatialReshardConfig::with_plan(None)
    }
}

impl From<GeospatialReshardPlan> for GeospatialReshardConfig<'_, GeospatialReshardPlan> {
    /// A deferred call owning `plan`, naming no handle, since a live plan already holds them.
    fn from(plan: GeospatialReshardPlan) -> Self {
        GeospatialReshardConfig::with_plan(Some(plan))
    }
}

impl<Plan> GeospatialReshardConfig<'_, Plan> {
    fn with_plan(plan: Option<Plan>) -> Self {
        GeospatialReshardConfig {
            sources: &[],
            targets: &[],
            expansion_factor: 0,
            memory_limit: 0,
            seed: 0,
            progress: None,
            progress_steps: 0,
            stats: None,
            plan,
            phase: None,
        }
    }
}

/// A refine in flight between calls, which [`GeospatialVerbs::refine_plan_finish`] closes.
#[derive(Debug)]
pub struct GeospatialRefinePlan {
    handle: usearch_geospatial_refine_plan_t,
}

/// A reshard in flight between calls, which [`GeospatialVerbs::reshard_plan_finish`] closes.
#[derive(Debug)]
pub struct GeospatialReshardPlan {
    handle: usearch_geospatial_reshard_plan_t,
}

impl Default for GeospatialRefinePlan {
    /// A fresh plan, which the first call holding it opens.
    fn default() -> Self {
        GeospatialRefinePlan { handle: ptr::null_mut() }
    }
}

impl Default for GeospatialReshardPlan {
    /// A fresh plan, which the first call holding it opens.
    fn default() -> Self {
        GeospatialReshardPlan { handle: ptr::null_mut() }
    }
}

// Safety: a plan is driven by one call at a time, which the `&mut` holding it enforces.
unsafe impl Send for GeospatialRefinePlan {}
unsafe impl Send for GeospatialReshardPlan {}

impl<Observations> Accepts<GeospatialInsertConfig<'static, Observations>> for Deferring where
    Deferring: Accepts<Observations>
{
}
impl Accepts<Vec<GeospatialObservation>> for Deferring {}
impl<const COUNT: usize> Accepts<[GeospatialObservation; COUNT]> for Deferring {}
impl Accepts<GeospatialWithinConfig<'static>> for Deferring {}
impl Accepts<GeospatialSearchConfig<'static>> for Deferring {}
impl<Plan: 'static> Accepts<GeospatialRefineConfig<'static, Plan>> for Deferring {}
impl<Plan: 'static> Accepts<GeospatialReshardConfig<'static, Plan>> for Deferring {}

// endregion: Configs

// region: Geospatial Index

/// Geospatial Search Index over observations, whose verbs come from [`GeospatialVerbs`].
/// [`GeospatialIndex::clear`] alone takes `&mut self`.
///
/// ```no_run
/// use usearch::*;
///
/// let index = GeospatialIndex::init_ram(&GeospatialOptions::default(), &[])?;
/// let places = [
///     GeospatialObservation::at(48.8566, 2.3522),
///     GeospatialObservation::at(52.52, 13.405),
/// ];
/// index.insert([Key::from(1u64), Key::from(2u64)], &places)?;
/// let mut arena = Arena::new()?;
/// let matches = index.search(&places[1..], 1, &mut arena)?;
/// assert_eq!(matches.query(0)?.keys, [Key::from(2u64)]);
/// # Ok::<(), usearch::Error>(())
/// ```
///
/// Dropping it waits for every deferred call naming the index, then frees it.
#[derive(Debug)]
pub struct GeospatialIndex {
    handle: usearch_geospatial_index_t,
}

// Safety: the library documents a handle as shareable across threads.
unsafe impl Send for GeospatialIndex {}
unsafe impl Sync for GeospatialIndex {}

impl Drop for GeospatialIndex {
    fn drop(&mut self) {
        unsafe { usearch_geospatial_free(self.handle) }
    }
}

impl GeospatialIndex {
    /// An index whose handle a constructor is about to fill.
    fn null() -> Self {
        GeospatialIndex { handle: ptr::null_mut() }
    }

    /// The receiver whose verbs block on `executor`, contended while another call holds it.
    ///
    /// Rust-only.
    pub fn on<'index>(&'index self, executor: &'index Executor) -> On<'index, GeospatialIndex> {
        On { index: self, executor, completion: PhantomData }
    }

    /// The receiver whose verbs submit their calls to `executor`, answering a
    /// [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
    ///
    /// Rust-only.
    pub fn deferred<'index>(&'index self, executor: &'index Executor) -> On<'index, GeospatialIndex, Deferring> {
        On { index: self, executor, completion: PhantomData }
    }

    /// Erases all entries on the calling thread, keeping the reserved capacity and frozen geometry.
    pub fn clear(&mut self) -> Result<()> {
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_geospatial_clear(&mut usearch_geospatial_clear_t {
                index: self.handle,
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
            })
        };
        check(outcome)
    }

    /// Creates an empty, mutable index in host memory, one shard per NUMA node `memory_domains`
    /// names; empty derives every node the process may use.
    pub fn init_ram(options: &GeospatialOptions, memory_domains: &[i32]) -> Result<Self> {
        let mut index = GeospatialIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_geospatial_init_ram(&mut usearch_geospatial_init_ram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                terrain: lower_terrain(options.terrain.as_ref()),
                connectivity: options.connectivity,
                values_per_key: options.values_per_key,
                ram: ram_options(memory_domains, options.cpu_class_ceiling),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Creates an empty, mutable index in device memory, one shard per device; empty derives one
    /// free device.
    pub fn init_vram(options: &GeospatialOptions, devices: &[i32]) -> Result<Self> {
        let mut index = GeospatialIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_geospatial_init_vram(&mut usearch_geospatial_init_vram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                terrain: lower_terrain(options.terrain.as_ref()),
                connectivity: options.connectivity,
                values_per_key: options.values_per_key,
                vram: vram_options(devices, options.gpu_class_ceiling),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a checkpoint into a new, mutable index in host memory, shard by shard.
    pub fn load_ram<'config>(config: impl Into<GeospatialLoadRamConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let mut index = GeospatialIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        unsafe {
            usearch_geospatial_load_ram(&mut usearch_geospatial_load_ram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                ram: ram_options(config.memory_domains, config.cpu_class_ceiling),
                values_per_key: config.values_per_key,
                terrain: lower_terrain(config.terrain.as_ref()),
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a checkpoint into a new index on devices, part by part.
    pub fn load_vram<'config>(config: impl Into<GeospatialLoadVramConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let mut index = GeospatialIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        unsafe {
            usearch_geospatial_load_vram(&mut usearch_geospatial_load_vram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                vram: vram_options(config.devices, config.gpu_class_ceiling),
                values_per_key: config.values_per_key,
                terrain: lower_terrain(config.terrain.as_ref()),
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a serialized monolith into a new, mutable index in host memory.
    pub fn load_ram_buffer<'config>(config: impl Into<GeospatialLoadRamBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let mut index = GeospatialIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        unsafe {
            usearch_geospatial_load_ram_buffer(&mut usearch_geospatial_load_ram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                ram: ram_options(config.memory_domains, config.cpu_class_ceiling),
                values_per_key: config.values_per_key,
                terrain: lower_terrain(config.terrain.as_ref()),
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a serialized monolith into a new index on devices.
    pub fn load_vram_buffer<'config>(config: impl Into<GeospatialLoadVramBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let mut index = GeospatialIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        unsafe {
            usearch_geospatial_load_vram_buffer(&mut usearch_geospatial_load_vram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                vram: vram_options(config.devices, config.gpu_class_ceiling),
                values_per_key: config.values_per_key,
                terrain: lower_terrain(config.terrain.as_ref()),
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Serves a checkpoint in place through a memory mapping, read-only; the files must stay
    /// unmodified while it lives.
    pub fn view_ram(path: impl AsRef<Path>, terrain: Option<&GeospatialTerrain>) -> Result<Self> {
        let mut index = GeospatialIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_geospatial_view_ram(&mut usearch_geospatial_view_ram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                path: path_view(path.as_ref())?,
                terrain: lower_terrain(terrain),
                ram: ram_options(&[], None),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Serves a checkpoint in place from NVMe drives, read-only and sized past RAM.
    pub fn view_nvme<'config>(config: impl Into<GeospatialViewNvmeConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let views = path_views(config.paths)?;
        let mut index = GeospatialIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_geospatial_view_nvme(&mut usearch_geospatial_view_nvme_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                nvme: nvme_options(&views, config.cpu_class_ceiling),
                terrain: lower_terrain(config.terrain.as_ref()),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Reads a checkpoint's graph without loading it.
    pub fn metadata(path: impl AsRef<Path>) -> Result<GeospatialMetadata> {
        let path = path_view(path.as_ref())?;
        let mut metadata = GeospatialMetadata::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_geospatial_metadata_read(&mut usearch_geospatial_metadata_read_t {
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path,
                metadata: &mut metadata,
            })
        };
        check(outcome).map(|()| metadata)
    }

    /// Reads a checkpoint's graph out of a serialized monolith buffer.
    pub fn metadata_buffer(buffer: &[u8]) -> Result<GeospatialMetadata> {
        let mut metadata = GeospatialMetadata::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_geospatial_metadata_read_buffer(&mut usearch_geospatial_metadata_read_buffer_t {
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: buffer.into(),
                metadata: &mut metadata,
            })
        };
        check(outcome).map(|()| metadata)
    }
}

/// What a verb answers through `Receiver`: its output, or a [`Deferred`](crate::Deferred).
type Answer<'index, Receiver, Inputs, Output> =
    <<Receiver as GeospatialVerbs<'index>>::Completion as Completion>::Answer<'index, Inputs, Output>;

/// Every geospatial verb, written once for every receiver: `&GeospatialIndex` blocks on the calling
/// thread, and the [`On`] that [`GeospatialIndex::on`] or [`GeospatialIndex::deferred`] makes runs
/// on an executor.
///
/// Rust-only.
pub trait GeospatialVerbs<'index>: Copy {
    /// [`Blocking`] or [`Deferring`].
    type Completion: Completion;

    /// The index this receiver reaches and the executor its calls run on.
    #[doc(hidden)]
    fn handles(&self) -> (usearch_geospatial_index_t, usearch_executor_t);

    /// Reserves room for `entries` observations, a hint, since insertions grow the index on demand.
    fn reserve(self, entries: usize) -> Result<Answer<'index, Self, (), ()>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), (), |call, completion| {
            unsafe {
                usearch_geospatial_reserve(&mut usearch_geospatial_reserve_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    entries,
                })
            };
            Ok(())
        })
    }

    /// Reports the frozen metadata and the current counters in one call.
    fn describe(self) -> Result<Answer<'index, Self, (), GeospatialIndexDescription>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), GeospatialIndexDescription::default(), |call, completion| {
            unsafe {
                usearch_geospatial_describe(&mut usearch_geospatial_describe_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    description: &mut call.output,
                })
            };
            Ok(())
        })
    }

    /// The observations stored under `key`.
    fn count(self, key: Key) -> Result<Answer<'index, Self, (), GeospatialKeyStats>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), GeospatialKeyStats::default(), |call, completion| {
            unsafe {
                usearch_geospatial_count(&mut usearch_geospatial_count_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    key,
                    key_stats: &mut call.output,
                })
            };
            Ok(())
        })
    }

    /// Whether the index holds at least one entry under `key`.
    ///
    /// Rust-only.
    fn contains(self, key: Key) -> Result<bool>
    where
        Self: GeospatialVerbs<'index, Completion = Blocking>,
    {
        Ok(self.count(key)?.entries != 0)
    }

    /// Retrieves every observation stored under `key` into `arena`, within the stored rounding,
    /// which the result borrows; an absent key yields none.
    fn get<'arena>(
        self,
        key: Key,
        arena: &'arena mut Arena,
    ) -> Result<Answer<'index, Self, (), GeospatialObservations<'arena>>> {
        let (index, executor) = self.handles();
        let arena_handle = arena.handle;
        let observations = GeospatialObservations { arena: PhantomData, observations: ptr::null(), written: 0 };
        Self::Completion::answer((), observations, |call, completion| {
            let observations = &mut call.output;
            unsafe {
                usearch_geospatial_get(&mut usearch_geospatial_get_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    key,
                    observations: &mut observations.observations,
                    written: &mut observations.written,
                    arena: arena_handle,
                })
            };
            Ok(())
        })
    }

    /// Adds one observation per key, all-or-nothing and visible at once.
    fn insert<'config, Keys: AsRef<[Key]>, Observations: AsRef<[GeospatialObservation]>>(
        self,
        keys: Keys,
        config: impl Into<GeospatialInsertConfig<'config, Observations>>,
    ) -> Result<Answer<'index, Self, (Keys, GeospatialInsertConfig<'config, Observations>), ()>>
    where
        Self::Completion: Accepts<Keys> + Accepts<GeospatialInsertConfig<'config, Observations>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer((keys, config.into()), (), |call, completion| {
            let (keys, config) = &mut call.inputs;
            let keys = keys.as_ref();
            // The keys view carries no count, so only this side can see the two disagree.
            let observations = config.observations.as_ref();
            if observations.len() != keys.len() {
                return Err(crate::types::fail(
                    Status::ShapeMismatch,
                    "the key count does not match the observation count",
                ));
            }
            unsafe {
                usearch_geospatial_insert(&mut usearch_geospatial_insert_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: keys.len(),
                    keys: usearch_keys_view_t::of(keys),
                    observations: observations.as_ptr(),
                    expansion_factor: config.expansion_factor,
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                })
            };
            Ok(())
        })
    }

    /// Runs one refine round over the whole graph, answering a [`RefineReport`].
    fn refine<'config, Plan: BorrowMut<GeospatialRefinePlan>>(
        self,
        config: GeospatialRefineConfig<'config, Plan>,
    ) -> Result<Answer<'index, Self, GeospatialRefineConfig<'config, Plan>, RefineReport>>
    where
        Self::Completion: Accepts<GeospatialRefineConfig<'config, Plan>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer(config, RefineReport::default(), |call, completion| {
            let config = &mut call.inputs;
            unsafe {
                usearch_geospatial_refine(&mut usearch_geospatial_refine_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    report: &mut call.output,
                    refine: usearch_refine_options_t {
                        expansion_factor: config.expansion_factor,
                        memory_limit: config.memory_limit,
                    },
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                    plan: config.plan.as_mut().map_or(ptr::null_mut(), |plan| &mut plan.borrow_mut().handle),
                    phase: output_slot(&mut config.phase),
                })
            };
            Ok(())
        })
    }

    /// Closes a refine plan, whatever the call answers, and answers what the whole plan did.
    fn refine_plan_finish(self, plan: GeospatialRefinePlan) -> Result<Answer<'index, Self, (), RefineReport>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), RefineReport::default(), |call, completion| {
            unsafe {
                usearch_geospatial_refine_plan_finish(&mut usearch_geospatial_refine_plan_finish_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    plan: plan.handle,
                    report: &mut call.output,
                })
            };
            Ok(())
        })
    }

    /// Measures how much of the graph a search reaches from the root.
    fn health(self) -> Result<Answer<'index, Self, (), DenseReachability>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), DenseReachability::default(), |call, completion| {
            unsafe {
                usearch_geospatial_health(&mut usearch_geospatial_health_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    reachability: &mut call.output,
                })
            };
            Ok(())
        })
    }

    /// Ranks the `wanted` nearest entries of each query the relation admits inside the radius,
    /// writing them into `arena` as [`Matches`] scored in chord metres.
    fn search<'config, 'arena, Queries: AsRef<[GeospatialObservation]>>(
        self,
        queries: Queries,
        config: impl Into<GeospatialSearchConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Result<Answer<'index, Self, (Queries, GeospatialSearchConfig<'config>), Matches<'arena>>>
    where
        Self::Completion: Accepts<Queries> + Accepts<GeospatialSearchConfig<'config>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer((queries, config.into()), Matches::new(arena), |call, completion| {
            let (queries, config) = &mut call.inputs;
            let queries = queries.as_ref();
            call.output.queries = queries.len();
            unsafe {
                usearch_geospatial_search(&mut usearch_geospatial_search_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: queries.len(),
                    queries: queries.as_ptr(),
                    relation: config.relation as usearch_geospatial_relation_t,
                    tolerance: usearch_geospatial_tolerance_t { metres: config.metres, seconds: config.seconds },
                    matches: &mut call.output.view,
                    arena: call.output.arena.handle,
                    wanted: config.wanted,
                    expansion_factor: config.expansion_factor,
                    traversal: config.traversal as usearch_traversal_t,
                    predicate: lower_predicate(&config.predicate),
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                })
            };
            Ok(())
        })
    }

    /// Reports every entry each query's relation admits inside the tolerance, however many there
    /// turn out to be, as one ragged run per query in `arena`.
    fn within<'config, 'arena, Queries: AsRef<[GeospatialObservation]>>(
        self,
        queries: Queries,
        config: impl Into<GeospatialWithinConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Result<Answer<'index, Self, (Queries, GeospatialWithinConfig<'config>), GeospatialMatches<'arena>>>
    where
        Self::Completion: Accepts<Queries> + Accepts<GeospatialWithinConfig<'config>>,
    {
        let (index, executor) = self.handles();
        let arena_handle = arena.handle;
        let matches = GeospatialMatches {
            arena: PhantomData,
            view: usearch_matches_tape_view_t::default(),
            queries: 0,
            written: 0,
        };
        Self::Completion::answer((queries, config.into()), matches, |call, completion| {
            let (queries, config) = &mut call.inputs;
            let queries = queries.as_ref();
            call.output.queries = queries.len();
            unsafe {
                usearch_geospatial_within(&mut usearch_geospatial_within_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: queries.len(),
                    queries: queries.as_ptr(),
                    relation: config.relation as usearch_geospatial_relation_t,
                    tolerance: usearch_geospatial_tolerance_t { metres: config.metres, seconds: config.seconds },
                    matches: &mut call.output.view,
                    written: &mut call.output.written,
                    arena: arena_handle,
                    traversal: config.traversal as usearch_traversal_t,
                    predicate: lower_predicate(&config.predicate),
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                })
            };
            Ok(())
        })
    }

    /// Moves the entries of the sources into the targets, keeping their links, then runs a round
    /// on every target that took entries: several sources into one target gather, one source into
    /// several targets cuts, and one handle as its only source and target recuts in place.
    fn reshard<'config, Plan: BorrowMut<GeospatialReshardPlan>>(
        self,
        config: GeospatialReshardConfig<'config, Plan>,
    ) -> Result<Answer<'index, Self, (GeospatialReshardConfig<'config, Plan>, ReshardHandles), RefineReport>>
    where
        Self::Completion: Accepts<GeospatialReshardConfig<'config, Plan>>,
    {
        let (_, executor) = self.handles();
        let handles = |named: &[&GeospatialIndex]| named.iter().map(|named| named.handle).collect();
        let lists = ReshardHandles { sources: handles(config.sources), targets: handles(config.targets) };
        Self::Completion::answer((config, lists), RefineReport::default(), |call, completion| {
            let (config, lists) = &mut call.inputs;
            unsafe {
                usearch_geospatial_reshard(&mut usearch_geospatial_reshard_t {
                    sources_count: lists.sources.len(),
                    sources: lists.sources.as_ptr(),
                    targets_count: lists.targets.len(),
                    targets: lists.targets.as_ptr(),
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    report: &mut call.output,
                    refine: usearch_refine_options_t {
                        expansion_factor: config.expansion_factor,
                        memory_limit: config.memory_limit,
                    },
                    seed: config.seed,
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                    plan: config.plan.as_mut().map_or(ptr::null_mut(), |plan| &mut plan.borrow_mut().handle),
                    phase: output_slot(&mut config.phase),
                })
            };
            Ok(())
        })
    }

    /// Closes a reshard plan, whatever the call answers, and answers what the whole plan did.
    fn reshard_plan_finish(self, plan: GeospatialReshardPlan) -> Result<Answer<'index, Self, (), RefineReport>> {
        let (_, executor) = self.handles();
        Self::Completion::answer((), RefineReport::default(), |call, completion| {
            unsafe {
                usearch_geospatial_reshard_plan_finish(&mut usearch_geospatial_reshard_plan_finish_t {
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    plan: plan.handle,
                    report: &mut call.output,
                })
            };
            Ok(())
        })
    }

    /// Draws keys around the focus, or across the whole index, writing up to `wanted` of them
    /// into `arena` as a [`Sample`].
    fn sample<'config, 'arena>(
        self,
        wanted: usize,
        arena: &'arena mut Arena,
        config: DenseSampleConfig<'config>,
    ) -> Result<Answer<'index, Self, DenseSampleConfig<'config>, Sample<'arena>>>
    where
        Self::Completion: Accepts<DenseSampleConfig<'config>>,
    {
        // Two slices name the shown keys and their places, so only this side sees them disagree.
        if config.shown.len() != config.shown_coordinates.len() {
            return Err(crate::types::fail(Status::InvalidArgument, "every shown key needs its coordinates"));
        }
        let (index, executor) = self.handles();
        let sample = Sample {
            arena,
            keys: usearch_keys_view_t::default(),
            entry_ordinals: ptr::null(),
            parents: usearch_matches_view_t::default(),
            coordinates: usearch_rows_view_t::default(),
            written: 0,
        };
        Self::Completion::answer(config, sample, |call, completion| {
            let (config, sample) = (&mut call.inputs, &mut call.output);
            unsafe {
                usearch_geospatial_sample(&mut usearch_geospatial_sample_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    shown_count: config.shown.len(),
                    shown: usearch_keys_view_t::of(config.shown),
                    shown_coordinates: usearch_rows_view_t {
                        data: config.shown_coordinates.as_ptr().cast(),
                        stride: core::mem::size_of::<[f32; 3]>(),
                        columns: 3,
                        scalar: Scalar::F32 as crate::types::usearch_scalar_t,
                    },
                    focus_count: config.focus.len(),
                    focus: usearch_keys_view_t::of(config.focus),
                    keys: &mut sample.keys,
                    entry_ordinals: &mut sample.entry_ordinals,
                    parents: &mut sample.parents,
                    coordinates: &mut sample.coordinates,
                    written: &mut sample.written,
                    arena: sample.arena.handle,
                    wanted,
                    seed: config.seed,
                    expansion_factor: config.expansion_factor,
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                })
            };
            Ok(())
        })
    }

    /// Saves the index in the layout the path extension picks - one `.safegraphs` monolith, or a
    /// `.safegraphs.json` manifest beside its shards.
    fn save<'config, Location: AsRef<Path>>(
        self,
        config: impl Into<DenseSaveConfig<'config, Location>>,
    ) -> Result<Answer<'index, Self, DenseSaveConfig<'config, Location>, ()>>
    where
        Self::Completion: Accepts<DenseSaveConfig<'config, Location>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer(config.into(), (), |call, completion| {
            let config = &mut call.inputs;
            let path = path_view(config.path.as_ref())?;
            unsafe {
                usearch_geospatial_save(&mut usearch_geospatial_save_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    path,
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                })
            };
            Ok(())
        })
    }

    /// Saves the index as one monolith into the arena its config carries.
    #[allow(clippy::type_complexity)]
    fn save_buffer<'config, 'arena>(
        self,
        config: impl Into<DenseSaveBufferConfig<'config, 'arena>>,
    ) -> Result<Answer<'index, Self, (Option<&'config mut ProgressCallback<'config>>, usize), Monolith<'arena>>>
    where
        Self::Completion: Accepts<DenseSaveBufferConfig<'config, 'arena>>,
    {
        let (index, executor) = self.handles();
        let DenseSaveBufferConfig { arena, progress, progress_steps } = config.into();
        let arena_handle = arena.handle;
        let monolith = Monolith { arena: PhantomData, monolith: usearch_bytes_view_t::default() };
        Self::Completion::answer((progress, progress_steps), monolith, |call, completion| {
            let (progress, progress_steps) = &mut call.inputs;
            unsafe {
                usearch_geospatial_save_buffer(&mut usearch_geospatial_save_buffer_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    monolith: &mut call.output.monolith,
                    arena: arena_handle,
                    progress: lower_progress(progress, *progress_steps),
                })
            };
            Ok(())
        })
    }

    /// Opens a walk over the current contents, pulled blocking on this receiver's executor:
    /// without `starts` every entry as stored, with them the start keys' entries first, then the
    /// entries those link to breadth-first.
    fn cursor(self, starts: &[Key]) -> Result<GeospatialCursor<'index>>
    where
        Self: GeospatialVerbs<'index, Completion = Blocking>,
    {
        let (index, executor) = self.handles();
        let mut cursor = GeospatialCursor { handle: ptr::null_mut(), index, executor, receiver: PhantomData };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_geospatial_cursor_init(&mut usearch_geospatial_cursor_init_t {
                index,
                outcome: &mut outcome,
                starts_count: starts.len(),
                starts: starts.as_ptr(),
                cursor: &mut cursor.handle,
            })
        };
        check(outcome).map(|()| cursor)
    }
}

impl<'index> GeospatialVerbs<'index> for &'index GeospatialIndex {
    type Completion = Blocking;

    fn handles(&self) -> (usearch_geospatial_index_t, usearch_executor_t) {
        (self.handle, ptr::null_mut())
    }
}

impl<'index, Completion: crate::types::Completion> GeospatialVerbs<'index> for On<'index, GeospatialIndex, Completion> {
    type Completion = Completion;

    fn handles(&self) -> (usearch_geospatial_index_t, usearch_executor_t) {
        (self.index.handle, self.executor.handle)
    }
}

/// The observations [`GeospatialVerbs::get`] wrote into the arena it borrows.
///
/// Rust-only.
pub struct GeospatialObservations<'arena> {
    arena: PhantomData<&'arena Arena>,
    observations: *const GeospatialObservation,
    written: usize,
}

impl<'arena> GeospatialObservations<'arena> {
    /// Every observation under the key, in insertion order.
    pub fn view(&self) -> &'arena [GeospatialObservation] {
        // Safety: the arena holds a finished call's packed output.
        unsafe { arena_slice(self.observations, self.written) }
    }
}

/// The ragged matches [`GeospatialVerbs::within`] wrote into the arena it borrows: query `i` spans
/// its two offsets, scored in chord metres.
///
/// Rust-only.
pub struct GeospatialMatches<'arena> {
    arena: PhantomData<&'arena Arena>,
    view: usearch_matches_tape_view_t,
    queries: usize,
    written: usize,
}

impl<'arena> GeospatialMatches<'arena> {
    /// Queries the call resolved.
    pub fn queries(&self) -> usize {
        self.queries
    }

    /// Matches across every query.
    pub fn written(&self) -> usize {
        self.written
    }

    /// One query's matches, refused with [`Status::InvalidArgument`] when `query` is not below
    /// [`GeospatialMatches::queries`].
    pub fn query(&self, query: usize) -> Result<QueryMatches<'arena>> {
        match query < self.queries {
            // Safety: the arena holds a finished call's output.
            true => Ok(unsafe { self.view.query(query) }),
            false => Err(crate::types::fail(Status::InvalidArgument, "the query is not below the call's count")),
        }
    }
}

// Safety: every view points into the arena borrowed beside it, which is `Sync`.
unsafe impl Send for GeospatialObservations<'_> {}
unsafe impl Send for GeospatialMatches<'_> {}

/// A walk over the index contents; see [`GeospatialVerbs::cursor`].
///
/// Dropping it waits for every deferred call naming the cursor, then releases it.
#[derive(Debug)]
pub struct GeospatialCursor<'index> {
    handle: usearch_geospatial_cursor_t,
    index: usearch_geospatial_index_t,
    executor: usearch_executor_t,
    receiver: PhantomData<On<'index, GeospatialIndex>>,
}

impl GeospatialCursor<'_> {
    /// Copies up to `wanted` next keys into `arena`; an empty batch means the walk is exhausted.
    pub fn next_keys<'arena>(&mut self, wanted: usize, arena: &'arena mut Arena) -> Result<DenseBatch<'arena>> {
        self.next(false, wanted, arena)
    }

    /// Copies up to `wanted` next keys into `arena`, each with the keys it links to.
    pub fn next_links<'arena>(&mut self, wanted: usize, arena: &'arena mut Arena) -> Result<DenseBatch<'arena>> {
        self.next(true, wanted, arena)
    }

    fn next<'arena>(&mut self, linked: bool, wanted: usize, arena: &'arena mut Arena) -> Result<DenseBatch<'arena>> {
        let mut batch = DenseBatch {
            arena: PhantomData,
            keys: usearch_keys_view_t::default(),
            links: linked.then(usearch_keys_tape_view_t::default),
            written: 0,
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_geospatial_cursor_next(&mut usearch_geospatial_cursor_next_t {
                index: self.index,
                executor: self.executor,
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                cursor: self.handle,
                keys: &mut batch.keys,
                links: batch.links.as_mut().map_or(ptr::null_mut(), ptr::from_mut),
                written: &mut batch.written,
                arena: arena.handle,
                wanted,
            })
        };
        check(outcome).map(|()| batch)
    }
}

impl Drop for GeospatialCursor<'_> {
    fn drop(&mut self) {
        unsafe { usearch_geospatial_cursor_free(self.handle) }
    }
}

// endregion: Geospatial Index

// region: Tests

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;

    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::testing::{ScratchDirectory, assert_close, case_stream_key, executor, keys, refused};
    use crate::types::Phase;

    // region: Helpers

    /// Random positions between the 60th parallels, every other field unrecorded, the same for the
    /// same seed.
    fn positions(count: usize, seed: u64) -> Vec<GeospatialObservation> {
        let mut generator = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                let latitude = generator.random_range(-60.0..60.0);
                GeospatialObservation::at(latitude, generator.random_range(-180.0..180.0))
            })
            .collect()
    }

    /// A host index holding `count` random positions under consecutive keys from `first`.
    fn populated_index(count: usize, first: u64, seed: u64) -> Result<GeospatialIndex> {
        let index = GeospatialIndex::init_ram(&GeospatialOptions::default(), &[])?;
        if count != 0 {
            index.on(&executor()).insert(keys(first, count), positions(count, seed))?;
        }
        Ok(index)
    }

    /// Paris, Berlin and New York.
    fn capitals() -> [GeospatialObservation; 3] {
        [
            GeospatialObservation::at(48.8566, 2.3522),
            GeospatialObservation::at(52.52, 13.405),
            GeospatialObservation::at(40.7128, -74.006),
        ]
    }

    fn size(index: &GeospatialIndex) -> Result<usize> {
        Ok(index.describe()?.size)
    }

    // endregion: Helpers

    // region: Construction

    #[test]
    fn init_ram_starts_empty() {
        let described = GeospatialIndex::init_ram(&GeospatialOptions::default(), &[]).unwrap().describe().unwrap();
        assert_eq!(described.size, 0);
        assert_eq!(described.gpu_class(), "unknown", "a host index names no GPU class");
    }

    /// A VRAM index either opens on a GPU or answers the library's own refusal.
    #[test]
    fn init_vram_surfaces_the_library_status() {
        match GeospatialIndex::init_vram(&GeospatialOptions::default(), &[]) {
            Ok(index) => assert_ne!(index.describe().unwrap().gpu_class(), "unknown"),
            Err(error) => assert_ne!(error.status, Status::Deferred, "a refusal is final"),
        }
    }

    // endregion: Construction

    // region: Ingest and Retrieval

    /// Mirrors the Geospatial Indexes snippet of rust/README.md line by line; only the function
    /// around it and its closing `Ok(())` differ.
    #[test]
    fn readme_quickstart() -> Result<()> {
        let index = GeospatialIndex::init_ram(&GeospatialOptions::default(), &[])?;
        let stops = [
            GeospatialObservation::at(51.5308, -0.1238), // every other field unrecorded, NaN
            GeospatialObservation::at(51.5320, -0.1233),
            GeospatialObservation::at(51.5033, -0.1196),
        ];
        index.insert([Key::from(1u64), Key::from(2u64), Key::from(3u64)], &stops)?;

        let mut arena = Arena::new()?;
        let nearby = index.within(&stops[..1], 500.0, &mut arena)?;
        assert_eq!(nearby.query(0)?.keys.len(), 2); // both stops within 500 metres, in chord metres
        Ok(())
    }

    /// An observation searched for comes back first, at zero chord metres.
    #[test]
    fn finds_the_inserted_entry() {
        let index = GeospatialIndex::init_ram(&GeospatialOptions::default(), &[]).unwrap();
        let places = capitals();
        index.insert(keys(1, 3), &places).unwrap();
        assert_eq!(size(&index).unwrap(), 3);

        let mut arena = Arena::new().unwrap();
        let matches = index.search(&places[1..2], 1, &mut arena).unwrap();
        let best = matches.query(0).unwrap();
        assert_eq!((best.keys, best.distances[0]), (&[Key::from(2u64)][..], 0.0));
    }

    /// Two observations a kilometre apart fall inside a two-kilometre radius, and a far one does
    /// not.
    #[test]
    fn within_reports_only_the_entries_inside_the_radius() {
        let places = [
            GeospatialObservation::at(48.8566, 2.3522),
            GeospatialObservation::at(48.8656, 2.3522),
            GeospatialObservation::at(52.52, 13.405),
        ];
        let index = GeospatialIndex::init_ram(&GeospatialOptions::default(), &[]).unwrap();
        index.insert(keys(1, 3), &places).unwrap();

        let mut arena = Arena::new().unwrap();
        let report = index.within(&places[..1], 2000.0, &mut arena).unwrap();
        let mut found: Vec<u64> =
            report.query(0).unwrap().keys.iter().map(|&key| u64::try_from(key).unwrap()).collect();
        found.sort_unstable();
        assert_eq!(found, [1, 2]);
    }

    #[test]
    fn gets_and_counts_every_entry_under_a_key() {
        let index = GeospatialIndex::init_ram(&GeospatialOptions::default(), &[]).unwrap();
        let shared = Key::from(7u64);
        let places = capitals();
        index.insert([shared, shared], &places[..2]).unwrap();

        assert_eq!(index.count(shared).unwrap().entries, 2);
        assert!(index.contains(shared).unwrap() && !index.contains(Key::from(8u64)).unwrap());

        let mut arena = Arena::new().unwrap();
        let stored = index.get(shared, &mut arena).unwrap();
        assert_eq!(stored.view().len(), 2);
        for (restored, inserted) in stored.view().iter().zip(&places) {
            assert_close(restored.latitude_degrees, inserted.latitude_degrees, 1e-6);
            assert_close(restored.longitude_degrees, inserted.longitude_degrees, 1e-6);
        }
        assert!(index.get(Key::from(8u64), &mut arena).unwrap().view().is_empty(), "an absent key yields none");
    }

    #[test]
    fn rejects_mismatched_key_and_entry_counts() {
        let index = GeospatialIndex::init_ram(&GeospatialOptions::default(), &[]).unwrap();
        assert!(refused(index.insert(keys(0, 3), &capitals()[..1]), Status::ShapeMismatch));
        assert_eq!(size(&index).unwrap(), 0);
    }

    // endregion: Ingest and Retrieval

    // region: Serialization

    #[test]
    fn saves_and_loads_a_file() {
        let scratch = ScratchDirectory::new("usearch-geospatial-saves_and_loads_a_file");
        let path = scratch.path().join("index.safegraphs");
        let index = populated_index(32, 0, case_stream_key(0)).unwrap();
        index.on(&executor()).save(&path).unwrap();

        assert!(GeospatialIndex::metadata(&path).unwrap().connectivity > 0);
        let restored = GeospatialIndex::load_ram(&path).unwrap();
        assert_eq!(size(&restored).unwrap(), 32);
        assert!(restored.contains(Key::from(17u64)).unwrap());

        // A view serves the checkpoint in place and refuses changes, where RAM serves views at all.
        match GeospatialIndex::view_ram(&path, None) {
            Ok(viewed) => {
                assert_eq!(size(&viewed).unwrap(), 32);
                assert!(refused(viewed.reserve(64), Status::ReadOnly));
            }
            Err(error) => assert_eq!(error.status, Status::Unsupported),
        }
    }

    #[test]
    fn saves_and_loads_a_buffer() {
        let index = populated_index(32, 0, case_stream_key(0)).unwrap();
        let mut arena = Arena::new().unwrap();
        let saved = index.save_buffer(&mut arena).unwrap();
        let monolith = saved.view();
        assert!(!monolith.is_empty());

        let connectivity = index.describe().unwrap().metadata.connectivity;
        assert_eq!(GeospatialIndex::metadata_buffer(monolith).unwrap().connectivity, connectivity);
        assert_eq!(size(&GeospatialIndex::load_ram_buffer(monolith).unwrap()).unwrap(), 32);
        assert!(
            GeospatialIndex::load_ram_buffer(&monolith[..monolith.len() - 1]).is_err(),
            "a short buffer must not be read"
        );
    }

    // endregion: Serialization

    // region: Refinement

    /// A round reports what it did, its closing audit judging every entry, and leaves each entry
    /// findable.
    #[test]
    fn refine_reports_a_round_over_every_entry() {
        let index = populated_index(64, 0, case_stream_key(0)).unwrap();
        let report = index.on(&executor()).refine(GeospatialRefineConfig::default()).unwrap();
        assert!(report.changed <= report.reselected);
        assert_eq!(report.reachability.audited, 64);
        assert!(index.contains(Key::from(17u64)).unwrap());
    }

    /// A plan keeps a stopped round live for the next call, and its finish reports the whole plan.
    #[test]
    fn a_refine_plan_resumes_and_finishes() {
        let index = populated_index(64, 0, case_stream_key(0)).unwrap();
        let executor = executor();
        let (mut plan, mut phase) = (GeospatialRefinePlan::default(), PhaseProgress::default());
        let stopped = index.on(&executor).refine(GeospatialRefineConfig {
            progress: Some(&mut |_, _| ControlFlow::Break(())),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..GeospatialRefineConfig::default()
        });
        if let Err(failure) = stopped {
            assert_eq!(failure.status, Status::Cancelled);
            assert!(phase.phase != Phase::Done && phase.completed <= phase.total);
            index
                .on(&executor)
                .refine(GeospatialRefineConfig {
                    plan: Some(&mut plan),
                    phase: Some(&mut phase),
                    ..GeospatialRefineConfig::default()
                })
                .unwrap();
        }
        assert_eq!(phase.phase, Phase::Done);
        assert_eq!(index.on(&executor).refine_plan_finish(plan).unwrap().reachability.audited, 64);
    }

    /// A plan lives across calls on one executor, so a call on the calling thread cannot open one.
    #[test]
    fn a_refine_plan_needs_an_executor() {
        let index = populated_index(0, 0, case_stream_key(0)).unwrap();
        let mut plan = GeospatialRefinePlan::default();
        let refined =
            index.refine(GeospatialRefineConfig { plan: Some(&mut plan), ..GeospatialRefineConfig::default() });
        assert!(refused(refined, Status::InvalidArgument));
    }

    // endregion: Refinement

    // region: Resharding

    /// Several sources into one target gather every entry there, and the other sources end empty
    /// and live.
    #[test]
    fn gathers_several_sources_into_one_target() {
        let first = populated_index(32, 0, case_stream_key(0)).unwrap();
        let (second, third) = (
            populated_index(32, 32, case_stream_key(1)).unwrap(),
            populated_index(32, 64, case_stream_key(2)).unwrap(),
        );
        first
            .on(&executor())
            .reshard(GeospatialReshardConfig {
                sources: &[&first, &second, &third],
                targets: &[&first],
                ..GeospatialReshardConfig::default()
            })
            .unwrap();
        assert_eq!(size(&first).unwrap(), 96);
        assert!(first.contains(Key::from(40u64)).unwrap() && first.contains(Key::from(80u64)).unwrap());
        assert_eq!((size(&second).unwrap(), size(&third).unwrap()), (0, 0));
    }

    /// A stop cuts only the closing rounds, so every entry has moved; the plan it leaves live
    /// continues where it stopped, and a deferred receiver closes it.
    #[test]
    fn a_stopped_gather_continues_through_its_plan() {
        let first = populated_index(16, 0, case_stream_key(0)).unwrap();
        let second = populated_index(16, 16, case_stream_key(1)).unwrap();
        let executor = executor();
        let (mut plan, mut phase) = (GeospatialReshardPlan::default(), PhaseProgress::default());
        let stopped = first.on(&executor).reshard(GeospatialReshardConfig {
            sources: &[&first, &second],
            targets: &[&first],
            progress: Some(&mut |_, _| ControlFlow::Break(())),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..GeospatialReshardConfig::default()
        });
        if let Err(failure) = stopped {
            assert_eq!(failure.status, Status::Cancelled);
            assert_ne!(phase.phase, Phase::Done, "a stop names the phase it stopped in");
        }
        assert_eq!(size(&first).unwrap(), 32);

        // A deferred call owns the plan it continues and hands it back once final.
        let continued = first.deferred(&executor).reshard(GeospatialReshardConfig::from(plan)).unwrap();
        let (finished, (config, _)) = continued.wait_with_inputs();
        finished.unwrap();
        let plan = config.plan.expect("the call hands its plan back");
        let report = first.deferred(&executor).reshard_plan_finish(plan).unwrap().wait().unwrap();
        assert!(report.changed <= report.reselected);
    }

    /// One source into several targets cuts it between them, and the source stays live but empty.
    #[test]
    fn reshards_one_source_into_several_targets() {
        let source = populated_index(64, 0, case_stream_key(0)).unwrap();
        let targets =
            [populated_index(0, 0, case_stream_key(0)).unwrap(), populated_index(0, 0, case_stream_key(0)).unwrap()];
        let report = source
            .on(&executor())
            .reshard(GeospatialReshardConfig {
                sources: &[&source],
                targets: &[&targets[0], &targets[1]],
                seed: case_stream_key(1),
                ..GeospatialReshardConfig::default()
            })
            .unwrap();
        assert!(report.changed <= report.reselected);
        let (left, right) = (size(&targets[0]).unwrap(), size(&targets[1]).unwrap());
        assert!(left != 0 && right != 0 && left + right == 64);
        assert_eq!(size(&source).unwrap(), 0);
    }

    /// A cut a stop interrupts keeps every entry moved, the next call naming the plan ends its
    /// rounds, and a deferred receiver closes it.
    #[test]
    fn reshards_one_source_into_several_targets_through_a_plan() {
        let source = populated_index(64, 0, case_stream_key(0)).unwrap();
        let targets =
            [populated_index(0, 0, case_stream_key(0)).unwrap(), populated_index(0, 0, case_stream_key(0)).unwrap()];
        let executor = executor();
        let (mut plan, mut phase) = (GeospatialReshardPlan::default(), PhaseProgress::default());
        let stopped = source.on(&executor).reshard(GeospatialReshardConfig {
            sources: &[&source],
            targets: &[&targets[0], &targets[1]],
            progress: Some(&mut |_, _| ControlFlow::Break(())),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..GeospatialReshardConfig::default()
        });
        assert_eq!(size(&targets[0]).unwrap() + size(&targets[1]).unwrap(), 64);
        if stopped.is_err() {
            source
                .on(&executor)
                .reshard(GeospatialReshardConfig {
                    plan: Some(&mut plan),
                    phase: Some(&mut phase),
                    ..GeospatialReshardConfig::default()
                })
                .unwrap();
        }
        assert_eq!(phase.phase, Phase::Done);
        source.deferred(&executor).reshard_plan_finish(plan).unwrap().wait().unwrap();
    }

    /// One index as its only source and target has a single shard to recut, which the library
    /// refuses.
    #[test]
    fn a_recut_of_one_handle_is_unsupported() {
        let index = populated_index(16, 0, case_stream_key(0)).unwrap();
        let recut = index.on(&executor()).reshard(GeospatialReshardConfig {
            sources: &[&index],
            targets: &[&index],
            ..GeospatialReshardConfig::default()
        });
        assert!(refused(recut, Status::Unsupported));
        assert_eq!(size(&index).unwrap(), 16);
    }

    /// Empty or repeated lists are malformed and a target holding foreign entries mismatches;
    /// every refusal leaves the entries where they were.
    #[test]
    fn refuses_malformed_lists_and_moves_nothing() {
        let (first, second) =
            (populated_index(8, 0, case_stream_key(0)).unwrap(), populated_index(8, 8, case_stream_key(1)).unwrap());
        let executor = executor();
        let reshard = |sources: &[&GeospatialIndex], targets: &[&GeospatialIndex]| {
            first.on(&executor).reshard(GeospatialReshardConfig {
                sources,
                targets,
                ..GeospatialReshardConfig::default()
            })
        };
        assert!(refused(reshard(&[], &[&first]), Status::InvalidArgument));
        assert!(refused(reshard(&[&first, &first], &[&first]), Status::InvalidArgument));
        assert!(refused(reshard(&[&first], &[&second]), Status::ConfigMismatch));
        assert_eq!((size(&first).unwrap(), size(&second).unwrap()), (8, 8));
    }

    // endregion: Resharding
}

// endregion: Tests
