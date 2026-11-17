//! Shape Search over 3D point clouds of any size, like molecules, conformers and protein backbones.
//!
//! Shapes cross as one [`Tape`]: their points end to end, `x`, `y` and `z` together, and one more
//! offset than entries, counted in points, as an Apache Arrow `List` column lays them out. Every
//! verb comes from [`ShapesVerbs`], written once over the receiver: a [`ShapesIndex`] blocks on the
//! calling thread, `index.on(&executor)` blocks on an executor, and `index.deferred(&executor)`
//! answers a [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
//!
//! File: rust/shapes_index.rs
//! Author: Ash Vardanian
//! Date: November 16, 2026

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types)]

use core::borrow::BorrowMut;
use core::ffi::{c_char, c_int, c_void};
use core::fmt;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr;
use core::str::FromStr;
use std::path::Path;

use crate::dense_index::{
    DenseBatch, DenseReachability, DenseSampleConfig, DenseSaveBufferConfig, DenseSaveConfig, DenseTraversalStats,
    Monolith, RefineReport, Sample, usearch_phase_progress_t, usearch_reachability_t, usearch_refine_options_t,
    usearch_refine_report_t, usearch_traversal_stats_t,
};
use crate::types::{
    Accepts, Arena, Blocking, CPUClass, Completion, Deferring, Element, Error, Executor, GPUClass, Key, Matches, On,
    PhaseProgress, Predicate, ProgressCallback, ReshardHandles, Result, Scalar, Status, Tape, TapeEntries, TapeOffset,
    Traversal, check, lower_predicate, lower_progress, nvme_options, output_slot, path_view, path_views, ram_options,
    read_named, spelling, usearch_arena_t, usearch_bytes_view_t, usearch_completion_blocking_k, usearch_completion_t,
    usearch_cpu_class_name, usearch_cpu_class_t, usearch_executor_t, usearch_gpu_class_name, usearch_gpu_class_t,
    usearch_key_t, usearch_keys_tape_view_t, usearch_keys_view_t, usearch_load_path_name, usearch_load_path_t,
    usearch_matches_view_t, usearch_nvme_options_t, usearch_outcome_t, usearch_predicate_t, usearch_progress_t,
    usearch_ram_options_t, usearch_rows_view_t, usearch_scalar_t, usearch_status_t, usearch_tape_view_t,
    usearch_traversal_t, usearch_vram_options_t, vram_options,
};

// region: Raw Declarations

pub type usearch_shapes_index_t = *mut c_void;
pub type usearch_shapes_cursor_t = *mut c_void;
pub type usearch_shapes_refine_plan_t = *mut c_void;
pub type usearch_shapes_reshard_plan_t = *mut c_void;
pub type usearch_shapes_metric_t = c_int;

#[repr(C)]
pub struct usearch_shapes_init_ram_t {
    pub index: *mut usearch_shapes_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub metric: usearch_shapes_metric_t,
    pub quantization: usearch_scalar_t,
    pub points: usize,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub ram: usearch_ram_options_t,
}

#[repr(C)]
pub struct usearch_shapes_init_vram_t {
    pub index: *mut usearch_shapes_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub metric: usearch_shapes_metric_t,
    pub quantization: usearch_scalar_t,
    pub points: usize,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub vram: usearch_vram_options_t,
}

#[repr(C)]
pub struct usearch_shapes_load_ram_t {
    pub index: *mut usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
    pub values_per_key: usize,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_shapes_load_ram_buffer_t {
    pub index: *mut usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
    pub values_per_key: usize,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_shapes_load_vram_t {
    pub index: *mut usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub vram: usearch_vram_options_t,
    pub values_per_key: usize,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_shapes_load_vram_buffer_t {
    pub index: *mut usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub vram: usearch_vram_options_t,
    pub values_per_key: usize,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_shapes_view_ram_t {
    pub index: *mut usearch_shapes_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub path: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
}

#[repr(C)]
pub struct usearch_shapes_view_nvme_t {
    pub index: *mut usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub nvme: usearch_nvme_options_t,
}

#[repr(C)]
pub struct usearch_shapes_reserve_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub entries: usize,
    pub points: usize,
}

#[repr(C)]
pub struct usearch_shapes_clear_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
}

#[repr(C)]
pub struct usearch_shapes_insert_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub keys: usearch_keys_view_t,
    pub shapes: usearch_tape_view_t,
    pub expansion_factor: usize,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_shapes_refine_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
    pub plan: *mut usearch_shapes_refine_plan_t,
    pub phase: *mut usearch_phase_progress_t,
}

#[repr(C)]
pub struct usearch_shapes_refine_plan_finish_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub plan: usearch_shapes_refine_plan_t,
    pub report: *mut usearch_refine_report_t,
}

#[repr(C)]
pub struct usearch_shapes_health_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub reachability: *mut usearch_reachability_t,
}

#[repr(C)]
pub struct usearch_shapes_search_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub queries: usearch_tape_view_t,
    pub matches: *mut usearch_matches_view_t,
    pub arena: usearch_arena_t,
    pub wanted: usize,
    pub expansion_factor: usize,
    pub traversal: usearch_traversal_t,
    pub predicate: usearch_predicate_t,
    pub hints: *const usearch_matches_view_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

pub type usearch_shapes_key_stats_t = ShapesKeyStats;

#[repr(C)]
pub struct usearch_shapes_count_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub key: usearch_key_t,
    pub key_stats: *mut usearch_shapes_key_stats_t,
}

#[repr(C)]
pub struct usearch_shapes_get_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub key: usearch_key_t,
    pub shapes: *mut usearch_tape_view_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
    pub outputs_scalar: usearch_scalar_t,
}

#[repr(C)]
pub struct usearch_shapes_reshard_t {
    pub sources_count: usize,
    pub sources: *const usearch_shapes_index_t,
    pub targets_count: usize,
    pub targets: *const usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub seed: u64,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
    pub plan: *mut usearch_shapes_reshard_plan_t,
    pub phase: *mut usearch_phase_progress_t,
}

#[repr(C)]
pub struct usearch_shapes_reshard_plan_finish_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub plan: usearch_shapes_reshard_plan_t,
    pub report: *mut usearch_refine_report_t,
}

#[repr(C)]
pub struct usearch_shapes_cursor_init_t {
    pub index: usearch_shapes_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub starts_count: usize,
    pub starts: *const usearch_key_t,
    pub cursor: *mut usearch_shapes_cursor_t,
}

#[repr(C)]
pub struct usearch_shapes_cursor_next_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub cursor: usearch_shapes_cursor_t,
    pub keys: *mut usearch_keys_view_t,
    pub links: *mut usearch_keys_tape_view_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
    pub wanted: usize,
}

#[repr(C)]
pub struct usearch_shapes_sample_t {
    pub index: usearch_shapes_index_t,
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
pub struct usearch_shapes_save_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_shapes_save_buffer_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub monolith: *mut usearch_bytes_view_t,
    pub arena: usearch_arena_t,
    pub progress: usearch_progress_t,
}

pub type usearch_shapes_metadata_t = ShapesMetadata;

#[repr(C)]
pub struct usearch_shapes_metadata_read_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub metadata: *mut usearch_shapes_metadata_t,
}

#[repr(C)]
pub struct usearch_shapes_metadata_read_buffer_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub metadata: *mut usearch_shapes_metadata_t,
}

pub type usearch_shapes_index_description_t = ShapesIndexDescription;

#[repr(C)]
pub struct usearch_shapes_describe_t {
    pub index: usearch_shapes_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub description: *mut usearch_shapes_index_description_t,
}

unsafe extern "C" {
    pub fn usearch_shapes_metric_name(metric: usearch_shapes_metric_t) -> *const c_char;
    pub fn usearch_shapes_metric_from_name(
        name: usearch_bytes_view_t,
        metric: *mut usearch_shapes_metric_t,
    ) -> usearch_status_t;

    pub fn usearch_shapes_init_ram(arguments: *mut usearch_shapes_init_ram_t);
    pub fn usearch_shapes_init_vram(arguments: *mut usearch_shapes_init_vram_t);
    pub fn usearch_shapes_load_ram(arguments: *mut usearch_shapes_load_ram_t);
    pub fn usearch_shapes_load_ram_buffer(arguments: *mut usearch_shapes_load_ram_buffer_t);
    pub fn usearch_shapes_load_vram(arguments: *mut usearch_shapes_load_vram_t);
    pub fn usearch_shapes_load_vram_buffer(arguments: *mut usearch_shapes_load_vram_buffer_t);
    pub fn usearch_shapes_view_ram(arguments: *mut usearch_shapes_view_ram_t);
    pub fn usearch_shapes_view_nvme(arguments: *mut usearch_shapes_view_nvme_t);
    pub fn usearch_shapes_free(index: usearch_shapes_index_t);

    pub fn usearch_shapes_reserve(arguments: *mut usearch_shapes_reserve_t);
    pub fn usearch_shapes_clear(arguments: *mut usearch_shapes_clear_t);
    pub fn usearch_shapes_insert(arguments: *mut usearch_shapes_insert_t);
    pub fn usearch_shapes_refine(arguments: *mut usearch_shapes_refine_t);
    pub fn usearch_shapes_refine_plan_finish(arguments: *mut usearch_shapes_refine_plan_finish_t);
    pub fn usearch_shapes_health(arguments: *mut usearch_shapes_health_t);
    pub fn usearch_shapes_search(arguments: *mut usearch_shapes_search_t);
    pub fn usearch_shapes_count(arguments: *mut usearch_shapes_count_t);
    pub fn usearch_shapes_get(arguments: *mut usearch_shapes_get_t);
    pub fn usearch_shapes_reshard(arguments: *mut usearch_shapes_reshard_t);
    pub fn usearch_shapes_reshard_plan_finish(arguments: *mut usearch_shapes_reshard_plan_finish_t);
    pub fn usearch_shapes_sample(arguments: *mut usearch_shapes_sample_t);

    pub fn usearch_shapes_cursor_init(arguments: *mut usearch_shapes_cursor_init_t);
    pub fn usearch_shapes_cursor_next(arguments: *mut usearch_shapes_cursor_next_t);
    pub fn usearch_shapes_cursor_free(cursor: usearch_shapes_cursor_t);

    pub fn usearch_shapes_save(arguments: *mut usearch_shapes_save_t);
    pub fn usearch_shapes_save_buffer(arguments: *mut usearch_shapes_save_buffer_t);
    pub fn usearch_shapes_metadata_read(arguments: *mut usearch_shapes_metadata_read_t);
    pub fn usearch_shapes_metadata_read_buffer(arguments: *mut usearch_shapes_metadata_read_buffer_t);
    pub fn usearch_shapes_describe(arguments: *mut usearch_shapes_describe_t);
}

// endregion: Raw Declarations

// region: Geometry

/// The shape kinds, every one invariant to rotation, reflection, translation and point order.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum ShapesMetric {
    /// The mean distance from every point to its nearest partner, both ways, after aligning the
    /// principal axes under all eight sign flips.
    #[default]
    Chamfer = 1,
    /// The largest distance from any point to its nearest partner, aligned as Chamfer aligns.
    Hausdorff = 2,
    /// How far apart each side's internal distances are under the best transport plan.
    GromovWasserstein = 3,
    /// An exact transport plan between the points, alternated with the best rotation or reflection.
    WassersteinProcrustes = 4,
}

impl ShapesMetric {
    fn from_raw(value: usearch_shapes_metric_t) -> Option<Self> {
        let metric = match value {
            1 => ShapesMetric::Chamfer,
            2 => ShapesMetric::Hausdorff,
            3 => ShapesMetric::GromovWasserstein,
            4 => ShapesMetric::WassersteinProcrustes,
            _ => return None,
        };
        Some(metric)
    }
}

impl fmt::Display for ShapesMetric {
    /// Writes the metric's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_shapes_metric_name, *self as c_int))
    }
}

impl FromStr for ShapesMetric {
    type Err = Error;

    /// Reads a metric back from the name `Display` writes, refusing any other.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_shapes_metric_from_name, ShapesMetric::from_raw)
    }
}

/// Construction options of a shape index; [`ShapesIndex::init_ram`] and [`ShapesIndex::init_vram`]
/// both take it.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct ShapesOptions {
    /// The kind every pair is scored under.
    pub metric: ShapesMetric,
    /// The scalar type points are stored in, `f16` or `f32`.
    pub quantization: Scalar,
    /// The frozen point count, when every entry has one; zero admits any count.
    pub points: usize,
    /// Graph links per node; zero derives it.
    pub connectivity: usize,
    /// Upper bound on entries sharing one key; zero is unbounded.
    pub values_per_key: usize,
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; `None` means no ceiling.
    pub cpu_class_ceiling: Option<CPUClass>,
    /// The newest class dispatch may pick on the devices; `None` means no ceiling.
    pub gpu_class_ceiling: Option<GPUClass>,
}

impl Default for ShapesOptions {
    fn default() -> Self {
        ShapesOptions {
            metric: ShapesMetric::default(),
            quantization: Scalar::F32,
            points: 0,
            connectivity: 0,
            values_per_key: 0,
            cpu_class_ceiling: None,
            gpu_class_ceiling: None,
        }
    }
}

/// The kind and geometry a checkpoint froze - everything a reader learns without opening it.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct ShapesMetadata {
    /// The kind the checkpoint scores with.
    pub metric: ShapesMetric,
    /// The scalar type its points are stored in.
    pub quantization: Scalar,
    /// The frozen point count; zero admits any.
    pub points: usize,
    /// Connections per node the graph was built with.
    pub connectivity: usize,
}

impl Default for ShapesMetadata {
    fn default() -> Self {
        ShapesMetadata { metric: ShapesMetric::default(), quantization: Scalar::F32, points: 0, connectivity: 0 }
    }
}

/// A live index's frozen metadata and current counters, each read once, so under concurrent
/// insertions they may come from slightly different moments.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct ShapesIndexDescription {
    /// The frozen metric, quantization, point count and connectivity, as a checkpoint reports them.
    pub metadata: ShapesMetadata,
    /// Stored entries.
    pub size: usize,
    /// Entries hosted without growing again.
    pub capacity: usize,
    /// Points across every stored entry.
    pub total_points: usize,
    /// Bytes the handle holds, payload storage included; traversal scratch is the executor's.
    pub memory_usage: usize,
    /// The CPU class dispatch picked, spelled by [`ShapesIndexDescription::cpu_class`].
    cpu_class: usearch_cpu_class_t,
    /// The GPU class dispatch picked, spelled by [`ShapesIndexDescription::gpu_class`].
    gpu_class: usearch_gpu_class_t,
    /// Live shards.
    pub shards_count: usize,
    /// How the last load read the parts, spelled by [`ShapesIndexDescription::load_path`].
    load_path: usearch_load_path_t,
}

impl ShapesIndexDescription {
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

/// What one key holds: its entries, and their point counts in total and at most.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct ShapesKeyStats {
    /// Entries stored under the key; zero means the index holds none.
    pub entries: usize,
    /// Points across those entries.
    pub total_points: usize,
    /// Points of the largest of them.
    pub largest_points: usize,
}

// endregion: Geometry

// region: Configs

/// A checkpoint path and how it is copied into a new index in host memory; see
/// [`ShapesIndex::load_ram`].
pub struct ShapesLoadRamConfig<'config> {
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
    /// Reports the shards placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for ShapesLoadRamConfig<'config> {
    fn from(path: &'config Location) -> Self {
        ShapesLoadRamConfig {
            path: path.as_ref(),
            executor: None,
            memory_domains: &[],
            cpu_class_ceiling: None,
            values_per_key: 0,
            progress: None,
            progress_steps: 0,
        }
    }
}

/// A checkpoint path and how it is uploaded into a new index on devices; see
/// [`ShapesIndex::load_vram`].
pub struct ShapesLoadVramConfig<'config> {
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
    /// Reports the shards placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for ShapesLoadVramConfig<'config> {
    fn from(path: &'config Location) -> Self {
        ShapesLoadVramConfig {
            path: path.as_ref(),
            executor: None,
            devices: &[],
            gpu_class_ceiling: None,
            values_per_key: 0,
            progress: None,
            progress_steps: 0,
        }
    }
}

/// A serialized monolith and how it is copied into a new index in host memory; see
/// [`ShapesIndex::load_ram_buffer`].
pub struct ShapesLoadRamBufferConfig<'config> {
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
    /// Reports the bytes placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config> From<&'config [u8]> for ShapesLoadRamBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        ShapesLoadRamBufferConfig {
            buffer,
            executor: None,
            memory_domains: &[],
            cpu_class_ceiling: None,
            values_per_key: 0,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for ShapesLoadRamBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self {
        ShapesLoadRamBufferConfig::from(&**bytes)
    }
}

/// A serialized monolith and how it is uploaded into a new index on devices; see
/// [`ShapesIndex::load_vram_buffer`].
pub struct ShapesLoadVramBufferConfig<'config> {
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
    /// Reports the bytes placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config> From<&'config [u8]> for ShapesLoadVramBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        ShapesLoadVramBufferConfig {
            buffer,
            executor: None,
            devices: &[],
            gpu_class_ceiling: None,
            values_per_key: 0,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for ShapesLoadVramBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self {
        ShapesLoadVramBufferConfig::from(&**bytes)
    }
}

/// A checkpoint path and the drives it is served from in place; see [`ShapesIndex::view_nvme`].
pub struct ShapesViewNvmeConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The executor the open fans out over; `None` opens on the calling thread.
    pub executor: Option<&'config Executor>,
    /// Devices or directories to serve the parts from; empty serves each where its manifest points.
    pub paths: &'config [&'config Path],
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; `None` means no ceiling.
    pub cpu_class_ceiling: Option<CPUClass>,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for ShapesViewNvmeConfig<'config> {
    fn from(path: &'config Location) -> Self {
        ShapesViewNvmeConfig { path: path.as_ref(), executor: None, paths: &[], cpu_class_ceiling: None }
    }
}

/// The shapes an insertion adds and the knobs of its walks; see [`ShapesVerbs::insert`].
pub struct ShapesInsertConfig<'config, Value: Clone, Offset: Clone = u64> {
    /// One entry per key, its points end to end, converted on the way in where `Value` differs
    /// from the stored quantization.
    pub shapes: Tape<'config, [Value; 3], Offset>,
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

impl<'config, Value: Clone, Offset: Clone> From<Tape<'config, [Value; 3], Offset>>
    for ShapesInsertConfig<'config, Value, Offset>
{
    fn from(shapes: Tape<'config, [Value; 3], Offset>) -> Self {
        ShapesInsertConfig { shapes, expansion_factor: 0, progress: None, progress_steps: 0, stats: None }
    }
}

/// The matches a search keeps and the knobs of its walk; see [`ShapesVerbs::search`].
pub struct ShapesSearchConfig<'config> {
    /// Matches kept per query.
    pub wanted: usize,
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

impl From<usize> for ShapesSearchConfig<'_> {
    fn from(wanted: usize) -> Self {
        ShapesSearchConfig {
            wanted,
            expansion_factor: 0,
            traversal: Traversal::Walk,
            predicate: None,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// How [`ShapesVerbs::refine`] runs; a blocking call borrows its plan, a deferred one owns it.
pub struct ShapesRefineConfig<'config, Plan = &'config mut ShapesRefinePlan> {
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

impl Default for ShapesRefineConfig<'_> {
    fn default() -> Self {
        ShapesRefineConfig::with_plan(None)
    }
}

impl From<ShapesRefinePlan> for ShapesRefineConfig<'_, ShapesRefinePlan> {
    /// A deferred call owning `plan`, with every other knob derived.
    fn from(plan: ShapesRefinePlan) -> Self {
        ShapesRefineConfig::with_plan(Some(plan))
    }
}

impl<Plan> ShapesRefineConfig<'_, Plan> {
    fn with_plan(plan: Option<Plan>) -> Self {
        ShapesRefineConfig {
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

/// The handles a reshard moves between and its knobs; see [`ShapesVerbs::reshard`].
pub struct ShapesReshardConfig<'config, Plan = &'config mut ShapesReshardPlan> {
    /// The handles whose entries move, each named once.
    pub sources: &'config [&'config ShapesIndex],
    /// The handles the entries move into, each a source itself or empty.
    pub targets: &'config [&'config ShapesIndex],
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

impl Default for ShapesReshardConfig<'_> {
    fn default() -> Self {
        ShapesReshardConfig::with_plan(None)
    }
}

impl From<ShapesReshardPlan> for ShapesReshardConfig<'_, ShapesReshardPlan> {
    /// A deferred call owning `plan`, naming no handle, since a live plan already holds them.
    fn from(plan: ShapesReshardPlan) -> Self {
        ShapesReshardConfig::with_plan(Some(plan))
    }
}

impl<Plan> ShapesReshardConfig<'_, Plan> {
    fn with_plan(plan: Option<Plan>) -> Self {
        ShapesReshardConfig {
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

/// A refine in flight between calls, which [`ShapesVerbs::refine_plan_finish`] closes.
#[derive(Debug)]
pub struct ShapesRefinePlan {
    handle: usearch_shapes_refine_plan_t,
}

/// A reshard in flight between calls, which [`ShapesVerbs::reshard_plan_finish`] closes.
#[derive(Debug)]
pub struct ShapesReshardPlan {
    handle: usearch_shapes_reshard_plan_t,
}

impl Default for ShapesRefinePlan {
    /// A fresh plan, which the first call holding it opens.
    fn default() -> Self {
        ShapesRefinePlan { handle: ptr::null_mut() }
    }
}

impl Default for ShapesReshardPlan {
    /// A fresh plan, which the first call holding it opens.
    fn default() -> Self {
        ShapesReshardPlan { handle: ptr::null_mut() }
    }
}

// Safety: a plan is driven by one call at a time, which the `&mut` holding it enforces.
unsafe impl Send for ShapesRefinePlan {}
unsafe impl Send for ShapesReshardPlan {}

impl<Value: Clone + 'static, Offset: Clone + 'static> Accepts<ShapesInsertConfig<'static, Value, Offset>>
    for Deferring
{
}
impl Accepts<ShapesSearchConfig<'static>> for Deferring {}
impl<Plan: 'static> Accepts<ShapesRefineConfig<'static, Plan>> for Deferring {}
impl<Plan: 'static> Accepts<ShapesReshardConfig<'static, Plan>> for Deferring {}

// endregion: Configs

// region: Shapes Index

/// Shape Search Index over 3D point clouds, whose verbs come from [`ShapesVerbs`].
/// [`ShapesIndex::clear`] alone takes `&mut self`.
///
/// ```no_run
/// use usearch::*;
///
/// let index = ShapesIndex::init_ram(&ShapesOptions::default(), &[])?;
/// let pair: [[f32; 3]; 2] = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]];
/// let triple: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]];
/// index.insert([Key::from(1u64), Key::from(2u64)], Tape::of([&pair[..], &triple[..]]))?;
/// let mut arena = Arena::new()?;
/// let matches = index.search(Tape::of([&triple[..]]), 1, &mut arena)?;
/// assert_eq!(matches.query(0)?.keys, [Key::from(2u64)]);
/// # Ok::<(), usearch::Error>(())
/// ```
///
/// Dropping it waits for every deferred call naming the index, then frees it.
#[derive(Debug)]
pub struct ShapesIndex {
    handle: usearch_shapes_index_t,
}

// Safety: the library documents a handle as shareable across threads.
unsafe impl Send for ShapesIndex {}
unsafe impl Sync for ShapesIndex {}

impl Drop for ShapesIndex {
    fn drop(&mut self) {
        unsafe { usearch_shapes_free(self.handle) }
    }
}

impl ShapesIndex {
    /// An index whose handle a constructor is about to fill.
    fn null() -> Self {
        ShapesIndex { handle: ptr::null_mut() }
    }

    /// The receiver whose verbs block on `executor`, contended while another call holds it.
    ///
    /// Rust-only.
    pub fn on<'index>(&'index self, executor: &'index Executor) -> On<'index, ShapesIndex> {
        On { index: self, executor, completion: PhantomData }
    }

    /// The receiver whose verbs submit their calls to `executor`, answering a
    /// [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
    ///
    /// Rust-only.
    pub fn deferred<'index>(&'index self, executor: &'index Executor) -> On<'index, ShapesIndex, Deferring> {
        On { index: self, executor, completion: PhantomData }
    }

    /// Erases all entries on the calling thread, keeping the reserved capacity and frozen geometry.
    pub fn clear(&mut self) -> Result<()> {
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_shapes_clear(&mut usearch_shapes_clear_t {
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
    pub fn init_ram(options: &ShapesOptions, memory_domains: &[i32]) -> Result<Self> {
        let mut index = ShapesIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_shapes_init_ram(&mut usearch_shapes_init_ram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                metric: options.metric as usearch_shapes_metric_t,
                quantization: options.quantization as usearch_scalar_t,
                points: options.points,
                connectivity: options.connectivity,
                values_per_key: options.values_per_key,
                ram: ram_options(memory_domains, options.cpu_class_ceiling),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Creates an empty, mutable index in device memory, one shard per device; empty derives one
    /// free device.
    pub fn init_vram(options: &ShapesOptions, devices: &[i32]) -> Result<Self> {
        let mut index = ShapesIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_shapes_init_vram(&mut usearch_shapes_init_vram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                metric: options.metric as usearch_shapes_metric_t,
                quantization: options.quantization as usearch_scalar_t,
                points: options.points,
                connectivity: options.connectivity,
                values_per_key: options.values_per_key,
                vram: vram_options(devices, options.gpu_class_ceiling),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a checkpoint into a new, mutable index in host memory, shard by shard.
    pub fn load_ram<'config>(config: impl Into<ShapesLoadRamConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let mut index = ShapesIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        unsafe {
            usearch_shapes_load_ram(&mut usearch_shapes_load_ram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                ram: ram_options(config.memory_domains, config.cpu_class_ceiling),
                values_per_key: config.values_per_key,
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a checkpoint into a new index on devices, part by part.
    pub fn load_vram<'config>(config: impl Into<ShapesLoadVramConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let mut index = ShapesIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        unsafe {
            usearch_shapes_load_vram(&mut usearch_shapes_load_vram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                vram: vram_options(config.devices, config.gpu_class_ceiling),
                values_per_key: config.values_per_key,
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a serialized monolith into a new, mutable index in host memory.
    pub fn load_ram_buffer<'config>(config: impl Into<ShapesLoadRamBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let mut index = ShapesIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        unsafe {
            usearch_shapes_load_ram_buffer(&mut usearch_shapes_load_ram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                ram: ram_options(config.memory_domains, config.cpu_class_ceiling),
                values_per_key: config.values_per_key,
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a serialized monolith into a new index on devices.
    pub fn load_vram_buffer<'config>(config: impl Into<ShapesLoadVramBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let mut index = ShapesIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        unsafe {
            usearch_shapes_load_vram_buffer(&mut usearch_shapes_load_vram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                vram: vram_options(config.devices, config.gpu_class_ceiling),
                values_per_key: config.values_per_key,
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Serves a checkpoint in place through a memory mapping, read-only; the files must stay
    /// unmodified while it lives.
    pub fn view_ram(path: impl AsRef<Path>) -> Result<Self> {
        let mut index = ShapesIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_shapes_view_ram(&mut usearch_shapes_view_ram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                path: path_view(path.as_ref())?,
                ram: ram_options(&[], None),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Serves a checkpoint in place from NVMe drives, read-only and sized past RAM.
    pub fn view_nvme<'config>(config: impl Into<ShapesViewNvmeConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let views = path_views(config.paths)?;
        let mut index = ShapesIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_shapes_view_nvme(&mut usearch_shapes_view_nvme_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                nvme: nvme_options(&views, config.cpu_class_ceiling),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Reads a checkpoint's kind and geometry without loading it.
    pub fn metadata(path: impl AsRef<Path>) -> Result<ShapesMetadata> {
        let path = path_view(path.as_ref())?;
        let mut metadata = ShapesMetadata::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_shapes_metadata_read(&mut usearch_shapes_metadata_read_t {
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path,
                metadata: &mut metadata,
            })
        };
        check(outcome).map(|()| metadata)
    }

    /// Reads a checkpoint's kind and geometry out of a serialized monolith buffer.
    pub fn metadata_buffer(buffer: &[u8]) -> Result<ShapesMetadata> {
        let mut metadata = ShapesMetadata::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_shapes_metadata_read_buffer(&mut usearch_shapes_metadata_read_buffer_t {
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
    <<Receiver as ShapesVerbs<'index>>::Completion as Completion>::Answer<'index, Inputs, Output>;

/// Every shapes verb, written once for every receiver: `&ShapesIndex` blocks on the calling
/// thread, and the [`On`] that [`ShapesIndex::on`] or [`ShapesIndex::deferred`] makes runs on
/// an executor.
///
/// Rust-only.
pub trait ShapesVerbs<'index>: Copy {
    /// [`Blocking`] or [`Deferring`].
    type Completion: Completion;

    /// The index this receiver reaches and the executor its calls run on.
    #[doc(hidden)]
    fn handles(&self) -> (usearch_shapes_index_t, usearch_executor_t);

    /// Reserves room for `entries` entries and `points` points across them, a hint, since
    /// insertions grow the index on demand.
    fn reserve(self, entries: usize, points: usize) -> Result<Answer<'index, Self, (), ()>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), (), |call, completion| {
            unsafe {
                usearch_shapes_reserve(&mut usearch_shapes_reserve_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    entries,
                    points,
                })
            };
            Ok(())
        })
    }

    /// Reports the frozen metadata and the current counters in one call.
    fn describe(self) -> Result<Answer<'index, Self, (), ShapesIndexDescription>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), ShapesIndexDescription::default(), |call, completion| {
            unsafe {
                usearch_shapes_describe(&mut usearch_shapes_describe_t {
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

    /// The entries stored under `key`, with their total and largest point counts.
    fn count(self, key: Key) -> Result<Answer<'index, Self, (), ShapesKeyStats>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), ShapesKeyStats::default(), |call, completion| {
            unsafe {
                usearch_shapes_count(&mut usearch_shapes_count_t {
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
        Self: ShapesVerbs<'index, Completion = Blocking>,
    {
        Ok(self.count(key)?.entries != 0)
    }

    /// Retrieves every shape stored under `key` into `arena` as `Value` points, each in its
    /// inserted pose, which the result borrows; an absent key yields none.
    fn get<'arena, Value: Element>(
        self,
        key: Key,
        arena: &'arena mut Arena,
    ) -> Result<Answer<'index, Self, (), TapeEntries<'arena, [Value; 3]>>> {
        let (index, executor) = self.handles();
        let arena_handle = arena.handle;
        Self::Completion::answer((), TapeEntries::new(), |call, completion| {
            let shapes = &mut call.output;
            unsafe {
                usearch_shapes_get(&mut usearch_shapes_get_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    key,
                    shapes: &mut shapes.view,
                    written: &mut shapes.written,
                    arena: arena_handle,
                    outputs_scalar: Value::SCALAR as usearch_scalar_t,
                })
            };
            Ok(())
        })
    }

    /// Adds a tape of shapes under their keys, all-or-nothing and visible at once; the points'
    /// scalar is the call's own, converted to the stored quantization on the way in.
    fn insert<'config, Keys: AsRef<[Key]>, Value: Element, Offset: TapeOffset>(
        self,
        keys: Keys,
        config: impl Into<ShapesInsertConfig<'config, Value, Offset>>,
    ) -> Result<Answer<'index, Self, (Keys, ShapesInsertConfig<'config, Value, Offset>), ()>>
    where
        Self::Completion: Accepts<Keys> + Accepts<ShapesInsertConfig<'config, Value, Offset>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer((keys, config.into()), (), |call, completion| {
            let (keys, config) = &mut call.inputs;
            let keys = keys.as_ref();
            // The keys view carries no count, so only this side can see the two disagree.
            if config.shapes.count() != keys.len() {
                return Err(crate::types::fail(
                    Status::ShapeMismatch,
                    "the key count does not match the tape's entries",
                ));
            }
            unsafe {
                usearch_shapes_insert(&mut usearch_shapes_insert_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: keys.len(),
                    keys: usearch_keys_view_t::of(keys),
                    shapes: config.shapes.view(Value::SCALAR),
                    expansion_factor: config.expansion_factor,
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                })
            };
            Ok(())
        })
    }

    /// Runs one refine round over the whole graph, answering a [`RefineReport`].
    fn refine<'config, Plan: BorrowMut<ShapesRefinePlan>>(
        self,
        config: ShapesRefineConfig<'config, Plan>,
    ) -> Result<Answer<'index, Self, ShapesRefineConfig<'config, Plan>, RefineReport>>
    where
        Self::Completion: Accepts<ShapesRefineConfig<'config, Plan>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer(config, RefineReport::default(), |call, completion| {
            let config = &mut call.inputs;
            unsafe {
                usearch_shapes_refine(&mut usearch_shapes_refine_t {
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
    fn refine_plan_finish(self, plan: ShapesRefinePlan) -> Result<Answer<'index, Self, (), RefineReport>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), RefineReport::default(), |call, completion| {
            unsafe {
                usearch_shapes_refine_plan_finish(&mut usearch_shapes_refine_plan_finish_t {
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
                usearch_shapes_health(&mut usearch_shapes_health_t {
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

    /// Resolves a tape of query shapes into up to `wanted` nearest matches apiece, writing them
    /// into `arena` as [`Matches`].
    fn search<'config, 'arena, Value: Element, Offset: TapeOffset>(
        self,
        queries: Tape<'config, [Value; 3], Offset>,
        config: impl Into<ShapesSearchConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Result<Answer<'index, Self, (Tape<'config, [Value; 3], Offset>, ShapesSearchConfig<'config>), Matches<'arena>>>
    where
        Self::Completion: Accepts<Tape<'config, [Value; 3], Offset>> + Accepts<ShapesSearchConfig<'config>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer((queries, config.into()), Matches::new(arena), |call, completion| {
            let (queries, config) = &mut call.inputs;
            call.output.queries = queries.count();
            unsafe {
                usearch_shapes_search(&mut usearch_shapes_search_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: queries.count(),
                    queries: queries.view(Value::SCALAR),
                    matches: &mut call.output.view,
                    arena: call.output.arena.handle,
                    wanted: config.wanted,
                    expansion_factor: config.expansion_factor,
                    traversal: config.traversal as usearch_traversal_t,
                    predicate: lower_predicate(&config.predicate),
                    hints: ptr::null(),
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
    fn reshard<'config, Plan: BorrowMut<ShapesReshardPlan>>(
        self,
        config: ShapesReshardConfig<'config, Plan>,
    ) -> Result<Answer<'index, Self, (ShapesReshardConfig<'config, Plan>, ReshardHandles), RefineReport>>
    where
        Self::Completion: Accepts<ShapesReshardConfig<'config, Plan>>,
    {
        let (_, executor) = self.handles();
        let handles = |named: &[&ShapesIndex]| named.iter().map(|named| named.handle).collect();
        let lists = ReshardHandles { sources: handles(config.sources), targets: handles(config.targets) };
        Self::Completion::answer((config, lists), RefineReport::default(), |call, completion| {
            let (config, lists) = &mut call.inputs;
            unsafe {
                usearch_shapes_reshard(&mut usearch_shapes_reshard_t {
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
    fn reshard_plan_finish(self, plan: ShapesReshardPlan) -> Result<Answer<'index, Self, (), RefineReport>> {
        let (_, executor) = self.handles();
        Self::Completion::answer((), RefineReport::default(), |call, completion| {
            unsafe {
                usearch_shapes_reshard_plan_finish(&mut usearch_shapes_reshard_plan_finish_t {
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
                usearch_shapes_sample(&mut usearch_shapes_sample_t {
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
                usearch_shapes_save(&mut usearch_shapes_save_t {
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
                usearch_shapes_save_buffer(&mut usearch_shapes_save_buffer_t {
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
    fn cursor(self, starts: &[Key]) -> Result<ShapesCursor<'index>>
    where
        Self: ShapesVerbs<'index, Completion = Blocking>,
    {
        let (index, executor) = self.handles();
        let mut cursor = ShapesCursor { handle: ptr::null_mut(), index, executor, receiver: PhantomData };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_shapes_cursor_init(&mut usearch_shapes_cursor_init_t {
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

impl<'index> ShapesVerbs<'index> for &'index ShapesIndex {
    type Completion = Blocking;

    fn handles(&self) -> (usearch_shapes_index_t, usearch_executor_t) {
        (self.handle, ptr::null_mut())
    }
}

impl<'index, Completion: crate::types::Completion> ShapesVerbs<'index> for On<'index, ShapesIndex, Completion> {
    type Completion = Completion;

    fn handles(&self) -> (usearch_shapes_index_t, usearch_executor_t) {
        (self.index.handle, self.executor.handle)
    }
}

/// A walk over the index contents; see [`ShapesVerbs::cursor`].
///
/// Dropping it waits for every deferred call naming the cursor, then releases it.
#[derive(Debug)]
pub struct ShapesCursor<'index> {
    handle: usearch_shapes_cursor_t,
    index: usearch_shapes_index_t,
    executor: usearch_executor_t,
    receiver: PhantomData<On<'index, ShapesIndex>>,
}

impl ShapesCursor<'_> {
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
            usearch_shapes_cursor_next(&mut usearch_shapes_cursor_next_t {
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

impl Drop for ShapesCursor<'_> {
    fn drop(&mut self) {
        unsafe { usearch_shapes_cursor_free(self.handle) }
    }
}

// endregion: Shapes Index

// region: Tests

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;

    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::testing::{ScratchDirectory, assert_close_rows, case_stream_key, executor, keys, refused};
    use crate::types::Phase;

    // region: Helpers

    /// Random clouds of the given point counts inside the unit cube, the same for the same seed.
    fn clouds(points: impl IntoIterator<Item = usize>, seed: u64) -> Vec<Vec<[f32; 3]>> {
        let mut generator = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut coordinate = move || generator.random_range(0.0f32..1.0);
        points
            .into_iter()
            .map(|count| (0..count).map(|_| [coordinate(), coordinate(), coordinate()]).collect())
            .collect()
    }

    /// Random clouds of 4 to 12 points, the same for the same seed.
    fn random_clouds(count: usize, seed: u64) -> Vec<Vec<[f32; 3]>> {
        let mut generator = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let points: Vec<usize> = (0..count).map(|_| generator.random_range(4..13)).collect();
        clouds(points, generator.random())
    }

    /// A host index holding `count` random clouds under consecutive keys from `first`.
    fn populated_index(count: usize, first: u64, seed: u64) -> Result<ShapesIndex> {
        let index = ShapesIndex::init_ram(&ShapesOptions::default(), &[])?;
        if count != 0 {
            index.on(&executor()).insert(keys(first, count), Tape::of(random_clouds(count, seed)))?;
        }
        Ok(index)
    }

    fn size(index: &ShapesIndex) -> Result<usize> {
        Ok(index.describe()?.size)
    }

    // endregion: Helpers

    // region: Construction

    #[test]
    fn init_ram_starts_empty_and_keeps_the_metric() {
        let options = ShapesOptions { metric: ShapesMetric::Hausdorff, ..ShapesOptions::default() };
        let described = ShapesIndex::init_ram(&options, &[]).unwrap().describe().unwrap();
        assert_eq!((described.size, described.total_points), (0, 0));
        assert_eq!(described.metadata.metric, ShapesMetric::Hausdorff);
        assert_eq!(described.metadata.quantization, Scalar::F32);
        assert_eq!(described.gpu_class(), "unknown", "a host index names no GPU class");
    }

    #[test]
    fn spells_every_metric_back() {
        for metric in [
            ShapesMetric::Chamfer,
            ShapesMetric::Hausdorff,
            ShapesMetric::GromovWasserstein,
            ShapesMetric::WassersteinProcrustes,
        ] {
            assert_eq!(metric.to_string().parse(), Ok(metric), "the library spells it back");
        }
        assert_eq!(ShapesMetric::GromovWasserstein.to_string(), "gromov-wasserstein");
        assert!("gromov_wasserstein".parse::<ShapesMetric>().is_err());
        assert!("frechet".parse::<ShapesMetric>().is_err());
    }

    /// A VRAM index either opens on a GPU or answers the library's own refusal.
    #[test]
    fn init_vram_surfaces_the_library_status() {
        match ShapesIndex::init_vram(&ShapesOptions::default(), &[]) {
            Ok(index) => assert_ne!(index.describe().unwrap().gpu_class(), "unknown"),
            Err(error) => assert_ne!(error.status, Status::Deferred, "a refusal is final"),
        }
    }

    // endregion: Construction

    // region: Ingest and Retrieval

    /// Mirrors the Shape Indexes snippet of rust/README.md line by line; only the function
    /// around it and its closing `Ok(())` differ.
    #[test]
    fn readme_quickstart() -> Result<()> {
        let index = ShapesIndex::init_ram(&ShapesOptions::default(), &[])?; // Chamfer over f32 points
        let pair: [[f32; 3]; 2] = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]];
        let triple: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]];
        index.insert([Key::from(1u64), Key::from(2u64)], Tape::of([&pair[..], &triple[..]]))?;

        let probe: [[f32; 3]; 2] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut arena = Arena::new()?;
        let matches = index.search(Tape::of([&probe[..]]), 1, &mut arena)?;
        assert_eq!(matches.query(0)?.keys, [Key::from(1u64)]);
        Ok(())
    }

    /// Shapes of different point counts share one batch, and each finds itself first.
    #[test]
    fn finds_the_inserted_entry() {
        let shapes = clouds([5, 9, 13], case_stream_key(0));
        let index = ShapesIndex::init_ram(&ShapesOptions::default(), &[]).unwrap();
        index.insert(keys(1, 3), Tape::of(&shapes)).unwrap();
        let described = index.describe().unwrap();
        assert_eq!((described.size, described.total_points), (3, 27));

        let mut arena = Arena::new().unwrap();
        let matches = index.search(Tape::of([&shapes[2]]), 1, &mut arena).unwrap();
        assert_eq!(matches.query(0).unwrap().keys, [Key::from(3u64)]);
    }

    #[test]
    fn gets_and_counts_every_entry_under_a_key() {
        let shapes = clouds([5, 9], case_stream_key(0));
        let index = ShapesIndex::init_ram(&ShapesOptions::default(), &[]).unwrap();
        let shared = Key::from(7u64);
        index.insert([shared, shared], Tape::of(&shapes)).unwrap();

        let stats = index.count(shared).unwrap();
        assert_eq!((stats.entries, stats.total_points, stats.largest_points), (2, 14, 9));
        assert!(index.contains(shared).unwrap() && !index.contains(Key::from(8u64)).unwrap());

        let mut arena = Arena::new().unwrap();
        let stored = index.get::<f32>(shared, &mut arena).unwrap();
        assert_eq!(stored.written(), 2);
        for (restored, inserted) in stored.iter().zip(&shapes) {
            assert_eq!(restored.len(), inserted.len());
            let flat = |cloud: &[[f32; 3]]| cloud.iter().flatten().copied().collect::<Vec<f32>>();
            assert_close_rows(&flat(restored), &flat(inserted), 1e-4);
        }
        assert_eq!(index.get::<f32>(Key::from(8u64), &mut arena).unwrap().written(), 0, "an absent key yields none");
    }

    #[test]
    fn rejects_mismatched_key_and_entry_counts() {
        let index = ShapesIndex::init_ram(&ShapesOptions::default(), &[]).unwrap();
        assert!(refused(index.insert(keys(0, 3), Tape::of(clouds([5], case_stream_key(0)))), Status::ShapeMismatch));
        assert_eq!(size(&index).unwrap(), 0);
    }

    // endregion: Ingest and Retrieval

    // region: Serialization

    #[test]
    fn saves_and_loads_a_file() {
        let scratch = ScratchDirectory::new("usearch-shapes-saves_and_loads_a_file");
        let path = scratch.path().join("index.safegraphs");
        let index = populated_index(32, 0, case_stream_key(0)).unwrap();
        index.on(&executor()).save(&path).unwrap();

        assert_eq!(ShapesIndex::metadata(&path).unwrap().metric, ShapesMetric::Chamfer);
        let restored = ShapesIndex::load_ram(&path).unwrap();
        assert_eq!(size(&restored).unwrap(), 32);
        assert!(restored.contains(Key::from(17u64)).unwrap());

        // A view serves the checkpoint in place and refuses changes, where RAM serves views at all.
        match ShapesIndex::view_ram(&path) {
            Ok(viewed) => {
                assert_eq!(size(&viewed).unwrap(), 32);
                assert!(refused(viewed.reserve(64, 0), Status::ReadOnly));
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

        assert_eq!(ShapesIndex::metadata_buffer(monolith).unwrap().metric, ShapesMetric::Chamfer);
        assert_eq!(size(&ShapesIndex::load_ram_buffer(monolith).unwrap()).unwrap(), 32);
        assert!(
            ShapesIndex::load_ram_buffer(&monolith[..monolith.len() - 1]).is_err(),
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
        let report = index.on(&executor()).refine(ShapesRefineConfig::default()).unwrap();
        assert!(report.changed <= report.reselected);
        assert_eq!(report.reachability.audited, 64);
        assert!(index.contains(Key::from(17u64)).unwrap());
    }

    /// A plan keeps a stopped round live for the next call, and its finish reports the whole plan.
    #[test]
    fn a_refine_plan_resumes_and_finishes() {
        let index = populated_index(64, 0, case_stream_key(0)).unwrap();
        let executor = executor();
        let (mut plan, mut phase) = (ShapesRefinePlan::default(), PhaseProgress::default());
        let stopped = index.on(&executor).refine(ShapesRefineConfig {
            progress: Some(&mut |_, _| ControlFlow::Break(())),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..ShapesRefineConfig::default()
        });
        if let Err(failure) = stopped {
            assert_eq!(failure.status, Status::Cancelled);
            assert!(phase.phase != Phase::Done && phase.completed <= phase.total);
            index
                .on(&executor)
                .refine(ShapesRefineConfig {
                    plan: Some(&mut plan),
                    phase: Some(&mut phase),
                    ..ShapesRefineConfig::default()
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
        let mut plan = ShapesRefinePlan::default();
        let refined = index.refine(ShapesRefineConfig { plan: Some(&mut plan), ..ShapesRefineConfig::default() });
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
            .reshard(ShapesReshardConfig {
                sources: &[&first, &second, &third],
                targets: &[&first],
                ..ShapesReshardConfig::default()
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
        let (mut plan, mut phase) = (ShapesReshardPlan::default(), PhaseProgress::default());
        let stopped = first.on(&executor).reshard(ShapesReshardConfig {
            sources: &[&first, &second],
            targets: &[&first],
            progress: Some(&mut |_, _| ControlFlow::Break(())),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..ShapesReshardConfig::default()
        });
        if let Err(failure) = stopped {
            assert_eq!(failure.status, Status::Cancelled);
            assert_ne!(phase.phase, Phase::Done, "a stop names the phase it stopped in");
        }
        assert_eq!(size(&first).unwrap(), 32);

        // A deferred call owns the plan it continues and hands it back once final.
        let continued = first.deferred(&executor).reshard(ShapesReshardConfig::from(plan)).unwrap();
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
            .reshard(ShapesReshardConfig {
                sources: &[&source],
                targets: &[&targets[0], &targets[1]],
                seed: case_stream_key(1),
                ..ShapesReshardConfig::default()
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
        let (mut plan, mut phase) = (ShapesReshardPlan::default(), PhaseProgress::default());
        let stopped = source.on(&executor).reshard(ShapesReshardConfig {
            sources: &[&source],
            targets: &[&targets[0], &targets[1]],
            progress: Some(&mut |_, _| ControlFlow::Break(())),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..ShapesReshardConfig::default()
        });
        assert_eq!(size(&targets[0]).unwrap() + size(&targets[1]).unwrap(), 64);
        if stopped.is_err() {
            source
                .on(&executor)
                .reshard(ShapesReshardConfig {
                    plan: Some(&mut plan),
                    phase: Some(&mut phase),
                    ..ShapesReshardConfig::default()
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
        let recut = index.on(&executor()).reshard(ShapesReshardConfig {
            sources: &[&index],
            targets: &[&index],
            ..ShapesReshardConfig::default()
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
        let reshard = |sources: &[&ShapesIndex], targets: &[&ShapesIndex]| {
            first.on(&executor).reshard(ShapesReshardConfig { sources, targets, ..ShapesReshardConfig::default() })
        };
        assert!(refused(reshard(&[], &[&first]), Status::InvalidArgument));
        assert!(refused(reshard(&[&first, &first], &[&first]), Status::InvalidArgument));
        assert!(refused(reshard(&[&first], &[&second]), Status::ConfigMismatch));
        assert_eq!((size(&first).unwrap(), size(&second).unwrap()), (8, 8));
    }

    // endregion: Resharding
}

// endregion: Tests
