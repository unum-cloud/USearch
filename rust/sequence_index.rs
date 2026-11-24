//! Sequence Search over byte strings, UTF-8 text, DNA, RNA and proteins, under edit distances and
//! alignments.
//!
//! Sequences cross as one [`Tape`]: their bytes end to end, and one more offset than entries, as
//! an Apache Arrow `Binary` or `LargeBinary` column lays them out. Every verb comes from
//! [`SequenceVerbs`], written once over the receiver: a [`SequenceIndex`] blocks on the calling
//! thread, `index.on(&executor)` blocks on an executor, and `index.deferred(&executor)` answers a
//! [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
//!
//! File: rust/sequence_index.rs
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
    Accepts, Arena, Blocking, CPUClass, Calling, Completion, Deferring, Error, Executor, GPUClass, Handles, Key,
    Matches, On, PhaseProgress, Predicate, Progress, ReshardHandles, Result, Scalar, Status, Submitting, Tape,
    TapeEntries, TapeOffset, Traversal, check, fail, lower_predicate, lower_progress, nvme_options, output_slot,
    path_view, path_views, ram_options, read_named, spelling, usearch_arena_t, usearch_bytes_view_t,
    usearch_completion_blocking_k, usearch_completion_t, usearch_cpu_class_name, usearch_cpu_class_t,
    usearch_executor_t, usearch_gpu_class_name, usearch_gpu_class_t, usearch_key_t, usearch_keys_tape_view_t,
    usearch_keys_view_t, usearch_load_path_name, usearch_load_path_t, usearch_matches_view_t, usearch_nvme_options_t,
    usearch_outcome_t, usearch_predicate_t, usearch_progress_t, usearch_ram_options_t, usearch_rows_view_t,
    usearch_status_t, usearch_tape_view_t, usearch_traversal_t, usearch_vram_options_t, vram_options,
};

// region: Raw Declarations

pub type usearch_sequence_index_t = *mut c_void;
pub type usearch_sequence_cursor_t = *mut c_void;
pub type usearch_sequence_refine_plan_t = *mut c_void;
pub type usearch_sequence_reshard_plan_t = *mut c_void;
pub type usearch_sequence_metric_t = c_int;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_sequence_weights_t {
    pub substitution_matrix: *const i8,
    pub gap: i8,
}

#[repr(C)]
pub struct usearch_sequence_init_ram_t {
    pub index: *mut usearch_sequence_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub metric: usearch_sequence_metric_t,
    pub weights: *const usearch_sequence_weights_t,
    pub window_widths: *const usize,
    pub window_widths_count: usize,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub ram: usearch_ram_options_t,
}

#[repr(C)]
pub struct usearch_sequence_init_vram_t {
    pub index: *mut usearch_sequence_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub metric: usearch_sequence_metric_t,
    pub weights: *const usearch_sequence_weights_t,
    pub window_widths: *const usize,
    pub window_widths_count: usize,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub vram: usearch_vram_options_t,
}

#[repr(C)]
pub struct usearch_sequence_load_ram_t {
    pub index: *mut usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
    pub values_per_key: usize,
    pub weights: *const usearch_sequence_weights_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_sequence_load_ram_buffer_t {
    pub index: *mut usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
    pub values_per_key: usize,
    pub weights: *const usearch_sequence_weights_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_sequence_load_vram_t {
    pub index: *mut usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub vram: usearch_vram_options_t,
    pub values_per_key: usize,
    pub weights: *const usearch_sequence_weights_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_sequence_load_vram_buffer_t {
    pub index: *mut usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub vram: usearch_vram_options_t,
    pub values_per_key: usize,
    pub weights: *const usearch_sequence_weights_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_sequence_view_ram_t {
    pub index: *mut usearch_sequence_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub path: usearch_bytes_view_t,
    pub weights: *const usearch_sequence_weights_t,
    pub ram: usearch_ram_options_t,
}

#[repr(C)]
pub struct usearch_sequence_view_nvme_t {
    pub index: *mut usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub nvme: usearch_nvme_options_t,
    pub weights: *const usearch_sequence_weights_t,
}

#[repr(C)]
pub struct usearch_sequence_reserve_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub entries: usize,
    pub payload_bytes: usize,
}

#[repr(C)]
pub struct usearch_sequence_clear_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
}

#[repr(C)]
pub struct usearch_sequence_insert_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub keys: usearch_keys_view_t,
    pub sequences: usearch_tape_view_t,
    pub expansion_factor: usize,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_sequence_refine_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
    pub plan: *mut usearch_sequence_refine_plan_t,
    pub phase: *mut usearch_phase_progress_t,
}

#[repr(C)]
pub struct usearch_sequence_refine_plan_finish_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub plan: usearch_sequence_refine_plan_t,
    pub report: *mut usearch_refine_report_t,
}

#[repr(C)]
pub struct usearch_sequence_health_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub reachability: *mut usearch_reachability_t,
}

#[repr(C)]
pub struct usearch_sequence_search_t {
    pub index: usearch_sequence_index_t,
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
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

pub type usearch_sequence_key_stats_t = SequenceKeyStats;

#[repr(C)]
pub struct usearch_sequence_count_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub key: usearch_key_t,
    pub key_stats: *mut usearch_sequence_key_stats_t,
}

#[repr(C)]
pub struct usearch_sequence_get_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub key: usearch_key_t,
    pub sequences: *mut usearch_tape_view_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
}

#[repr(C)]
pub struct usearch_sequence_reshard_t {
    pub sources_count: usize,
    pub sources: *const usearch_sequence_index_t,
    pub targets_count: usize,
    pub targets: *const usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub seed: u64,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
    pub plan: *mut usearch_sequence_reshard_plan_t,
    pub phase: *mut usearch_phase_progress_t,
}

#[repr(C)]
pub struct usearch_sequence_reshard_plan_finish_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub plan: usearch_sequence_reshard_plan_t,
    pub report: *mut usearch_refine_report_t,
}

#[repr(C)]
pub struct usearch_sequence_cursor_init_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub starts_count: usize,
    pub starts: *const usearch_key_t,
    pub cursor: *mut usearch_sequence_cursor_t,
}

#[repr(C)]
pub struct usearch_sequence_cursor_next_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub cursor: usearch_sequence_cursor_t,
    pub keys: *mut usearch_keys_view_t,
    pub links: *mut usearch_keys_tape_view_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
    pub wanted: usize,
}

#[repr(C)]
pub struct usearch_sequence_sample_t {
    pub index: usearch_sequence_index_t,
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
pub struct usearch_sequence_save_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_sequence_save_buffer_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub monolith: *mut usearch_bytes_view_t,
    pub arena: usearch_arena_t,
    pub progress: usearch_progress_t,
}

pub type usearch_sequence_metadata_t = SequenceMetadata;

#[repr(C)]
pub struct usearch_sequence_metadata_read_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub metadata: *mut usearch_sequence_metadata_t,
}

#[repr(C)]
pub struct usearch_sequence_metadata_read_buffer_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub metadata: *mut usearch_sequence_metadata_t,
}

pub type usearch_sequence_index_description_t = SequenceIndexDescription;

#[repr(C)]
pub struct usearch_sequence_describe_t {
    pub index: usearch_sequence_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub description: *mut usearch_sequence_index_description_t,
}

unsafe extern "C" {
    pub fn usearch_sequence_metric_name(metric: usearch_sequence_metric_t) -> *const c_char;
    pub fn usearch_sequence_metric_from_name(
        name: usearch_bytes_view_t,
        metric: *mut usearch_sequence_metric_t,
    ) -> usearch_status_t;

    pub fn usearch_sequence_init_ram(arguments: *mut usearch_sequence_init_ram_t);
    pub fn usearch_sequence_init_vram(arguments: *mut usearch_sequence_init_vram_t);
    pub fn usearch_sequence_load_ram(arguments: *mut usearch_sequence_load_ram_t);
    pub fn usearch_sequence_load_ram_buffer(arguments: *mut usearch_sequence_load_ram_buffer_t);
    pub fn usearch_sequence_load_vram(arguments: *mut usearch_sequence_load_vram_t);
    pub fn usearch_sequence_load_vram_buffer(arguments: *mut usearch_sequence_load_vram_buffer_t);
    pub fn usearch_sequence_view_ram(arguments: *mut usearch_sequence_view_ram_t);
    pub fn usearch_sequence_view_nvme(arguments: *mut usearch_sequence_view_nvme_t);
    pub fn usearch_sequence_free(index: usearch_sequence_index_t);

    pub fn usearch_sequence_reserve(arguments: *mut usearch_sequence_reserve_t);
    pub fn usearch_sequence_clear(arguments: *mut usearch_sequence_clear_t);
    pub fn usearch_sequence_insert(arguments: *mut usearch_sequence_insert_t);
    pub fn usearch_sequence_refine(arguments: *mut usearch_sequence_refine_t);
    pub fn usearch_sequence_refine_plan_finish(arguments: *mut usearch_sequence_refine_plan_finish_t);
    pub fn usearch_sequence_health(arguments: *mut usearch_sequence_health_t);
    pub fn usearch_sequence_search(arguments: *mut usearch_sequence_search_t);
    pub fn usearch_sequence_count(arguments: *mut usearch_sequence_count_t);
    pub fn usearch_sequence_get(arguments: *mut usearch_sequence_get_t);
    pub fn usearch_sequence_reshard(arguments: *mut usearch_sequence_reshard_t);
    pub fn usearch_sequence_reshard_plan_finish(arguments: *mut usearch_sequence_reshard_plan_finish_t);
    pub fn usearch_sequence_sample(arguments: *mut usearch_sequence_sample_t);

    pub fn usearch_sequence_cursor_init(arguments: *mut usearch_sequence_cursor_init_t);
    pub fn usearch_sequence_cursor_next(arguments: *mut usearch_sequence_cursor_next_t);
    pub fn usearch_sequence_cursor_free(cursor: usearch_sequence_cursor_t);

    pub fn usearch_sequence_save(arguments: *mut usearch_sequence_save_t);
    pub fn usearch_sequence_save_buffer(arguments: *mut usearch_sequence_save_buffer_t);
    pub fn usearch_sequence_metadata_read(arguments: *mut usearch_sequence_metadata_read_t);
    pub fn usearch_sequence_metadata_read_buffer(arguments: *mut usearch_sequence_metadata_read_buffer_t);
    pub fn usearch_sequence_describe(arguments: *mut usearch_sequence_describe_t);
}

// endregion: Raw Declarations

// region: Geometry

/// The sequence kinds: edit distances over bytes or code points, a weighted alignment, and the
/// overlap of fixed-width windows.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum SequenceMetric {
    /// Edits between two byte strings, each insertion, deletion or substitution counting one.
    #[default]
    LevenshteinBytes = 1,
    /// Edits between two UTF-8 strings, counted over code points rather than bytes.
    LevenshteinUtf8 = 2,
    /// The global alignment score under [`SequenceWeights`], reported as a distance.
    NeedlemanWunsch = 3,
    /// The share of one side's fixed-width windows the other side misses.
    Overlap = 4,
}

impl SequenceMetric {
    fn from_raw(value: usearch_sequence_metric_t) -> Option<Self> {
        let metric = match value {
            1 => SequenceMetric::LevenshteinBytes,
            2 => SequenceMetric::LevenshteinUtf8,
            3 => SequenceMetric::NeedlemanWunsch,
            4 => SequenceMetric::Overlap,
            _ => return None,
        };
        Some(metric)
    }
}

impl fmt::Display for SequenceMetric {
    /// Writes the metric's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_sequence_metric_name, *self as c_int))
    }
}

impl FromStr for SequenceMetric {
    type Err = Error;

    /// Reads a metric back from the name `Display` writes, refusing any other.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_sequence_metric_from_name, SequenceMetric::from_raw)
    }
}

/// The weights [`SequenceMetric::NeedlemanWunsch`] scores with; the Levenshtein kinds refuse any.
#[derive(Clone, Copy, Debug)]
pub struct SequenceWeights<'matrix> {
    /// Signed scores of every byte against every byte, row-major.
    pub substitution_matrix: &'matrix [[i8; 256]; 256],
    /// Cost of one unaligned symbol.
    pub gap: i8,
}

impl SequenceWeights<'_> {
    fn lower(&self) -> usearch_sequence_weights_t {
        usearch_sequence_weights_t { substitution_matrix: self.substitution_matrix.as_ptr().cast(), gap: self.gap }
    }
}

/// Lowers optional weights into `slot`, answering the pointer a call reads them through.
fn weights_slot(
    weights: Option<&SequenceWeights>,
    slot: &mut usearch_sequence_weights_t,
) -> *const usearch_sequence_weights_t {
    match weights {
        Some(weights) => {
            *slot = weights.lower();
            slot
        }
        None => ptr::null(),
    }
}

/// Construction options of a sequence index; [`SequenceIndex::init_ram`] and
/// [`SequenceIndex::init_vram`] both take it.
#[derive(Clone, Debug, Default)]
pub struct SequenceOptions<'options> {
    /// The kind every pair is scored under.
    pub metric: SequenceMetric,
    /// The weights an alignment kind needs; `None` for the Levenshtein kinds, which refuse any.
    pub weights: Option<SequenceWeights<'options>>,
    /// Window widths in bytes for the overlap kind, which needs at least one; every other kind
    /// refuses any.
    pub window_widths: &'options [usize],
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

/// The kind and graph a checkpoint froze - everything a reader learns without opening it.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct SequenceMetadata {
    /// The kind the checkpoint scores with.
    pub metric: SequenceMetric,
    /// Connections per node the graph was built with.
    pub connectivity: usize,
}

/// A live index's frozen metadata and current counters, each read once, so under concurrent
/// insertions they may come from slightly different moments.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct SequenceIndexDescription {
    /// The frozen metric and connectivity, as a checkpoint reports them.
    pub metadata: SequenceMetadata,
    /// Stored entries.
    pub size: usize,
    /// Entries hosted without growing again.
    pub capacity: usize,
    /// Bytes the handle holds, payload storage included; traversal scratch is the executor's.
    pub memory_usage: usize,
    /// The CPU class dispatch picked, spelled by [`SequenceIndexDescription::cpu_class`].
    cpu_class: usearch_cpu_class_t,
    /// The GPU class dispatch picked, spelled by [`SequenceIndexDescription::gpu_class`].
    gpu_class: usearch_gpu_class_t,
    /// Live shards.
    pub shards_count: usize,
    /// How the last load read the parts, spelled by [`SequenceIndexDescription::load_path`].
    load_path: usearch_load_path_t,
}

impl SequenceIndexDescription {
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

/// What one key holds: its entries, and their byte lengths in total and at most.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct SequenceKeyStats {
    /// Entries stored under the key; zero means the index holds none.
    pub entries: usize,
    /// Bytes across those entries.
    pub total_bytes: usize,
    /// Bytes of the longest of them.
    pub largest_bytes: usize,
}

// endregion: Geometry

// region: Configs

/// A checkpoint path and how it is copied into a new index in host memory; see
/// [`SequenceIndex::load_ram`].
pub struct SequenceLoadRamConfig<'config> {
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
    /// The weights an alignment kind needs, which no checkpoint records.
    pub weights: Option<SequenceWeights<'config>>,
    /// Reports the shards placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<Progress<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for SequenceLoadRamConfig<'config> {
    fn from(path: &'config Location) -> Self {
        SequenceLoadRamConfig {
            path: path.as_ref(),
            executor: None,
            memory_domains: &[],
            cpu_class_ceiling: None,
            values_per_key: 0,
            weights: None,
            progress: None,
            progress_steps: 0,
        }
    }
}

/// A checkpoint path and how it is uploaded into a new index on devices; see
/// [`SequenceIndex::load_vram`].
pub struct SequenceLoadVramConfig<'config> {
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
    /// The weights an alignment kind needs, which no checkpoint records.
    pub weights: Option<SequenceWeights<'config>>,
    /// Reports the shards placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<Progress<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for SequenceLoadVramConfig<'config> {
    fn from(path: &'config Location) -> Self {
        SequenceLoadVramConfig {
            path: path.as_ref(),
            executor: None,
            devices: &[],
            gpu_class_ceiling: None,
            values_per_key: 0,
            weights: None,
            progress: None,
            progress_steps: 0,
        }
    }
}

/// A serialized monolith and how it is copied into a new index in host memory; see
/// [`SequenceIndex::load_ram_buffer`].
pub struct SequenceLoadRamBufferConfig<'config> {
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
    /// The weights an alignment kind needs, which no checkpoint records.
    pub weights: Option<SequenceWeights<'config>>,
    /// Reports the bytes placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<Progress<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config> From<&'config [u8]> for SequenceLoadRamBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        SequenceLoadRamBufferConfig {
            buffer,
            executor: None,
            memory_domains: &[],
            cpu_class_ceiling: None,
            values_per_key: 0,
            weights: None,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for SequenceLoadRamBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self {
        SequenceLoadRamBufferConfig::from(&**bytes)
    }
}

/// A serialized monolith and how it is uploaded into a new index on devices; see
/// [`SequenceIndex::load_vram_buffer`].
pub struct SequenceLoadVramBufferConfig<'config> {
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
    /// The weights an alignment kind needs, which no checkpoint records.
    pub weights: Option<SequenceWeights<'config>>,
    /// Reports the bytes placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<Progress<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config> From<&'config [u8]> for SequenceLoadVramBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        SequenceLoadVramBufferConfig {
            buffer,
            executor: None,
            devices: &[],
            gpu_class_ceiling: None,
            values_per_key: 0,
            weights: None,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for SequenceLoadVramBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self {
        SequenceLoadVramBufferConfig::from(&**bytes)
    }
}

/// A checkpoint path and the drives it is served from in place; see [`SequenceIndex::view_nvme`].
pub struct SequenceViewNvmeConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The executor the open fans out over; `None` opens on the calling thread.
    pub executor: Option<&'config Executor>,
    /// Devices or directories to serve the parts from; empty serves each where its manifest points.
    pub paths: &'config [&'config Path],
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; `None` means no ceiling.
    pub cpu_class_ceiling: Option<CPUClass>,
    /// The weights an alignment kind needs, which no checkpoint records.
    pub weights: Option<SequenceWeights<'config>>,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for SequenceViewNvmeConfig<'config> {
    fn from(path: &'config Location) -> Self {
        SequenceViewNvmeConfig {
            path: path.as_ref(),
            executor: None,
            paths: &[],
            cpu_class_ceiling: None,
            weights: None,
        }
    }
}

/// The sequences an insertion adds and the knobs of its walks; see [`SequenceVerbs::insert`].
pub struct SequenceInsertConfig<'config, Offset: Clone = u64> {
    /// One entry per key, its bytes end to end.
    pub sequences: Tape<'config, u8, Offset>,
    /// The breadth of the candidates the walk explores; zero derives it.
    pub expansion_factor: usize,
    /// Reports the entries inserted; every entry is claimed before the first report, so a `Break`
    /// is ignored.
    pub progress: Option<Progress<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl<'config, Offset: Clone> From<Tape<'config, u8, Offset>> for SequenceInsertConfig<'config, Offset> {
    fn from(sequences: Tape<'config, u8, Offset>) -> Self {
        SequenceInsertConfig { sequences, expansion_factor: 0, progress: None, progress_steps: 0, stats: None }
    }
}

/// The matches a search keeps and the knobs of its walk; see [`SequenceVerbs::search`].
pub struct SequenceSearchConfig<'config> {
    /// Matches kept per query.
    pub wanted: usize,
    /// The breadth of the candidates the walk explores; zero derives it.
    pub expansion_factor: usize,
    /// Walks the graph by default; [`Traversal::Scan`] scores every entry exactly.
    pub traversal: Traversal,
    /// Admits only the keys it approves; the rest still route the walk.
    pub predicate: Option<Predicate<'config>>,
    /// Reports the queries resolved and stops wherever it says to.
    pub progress: Option<Progress<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl From<usize> for SequenceSearchConfig<'_> {
    fn from(wanted: usize) -> Self {
        SequenceSearchConfig {
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

/// How [`SequenceVerbs::refine`] runs; a blocking call takes its plan by reference, a deferred
/// one owns it.
pub struct SequenceRefineConfig<'config, Plan = &'config mut SequenceRefinePlan> {
    /// The breadth of any walk beside the round; zero derives it.
    pub expansion_factor: usize,
    /// Bytes a loaded NVMe handle holds for the one part it rewrites; zero derives it.
    pub memory_limit: usize,
    /// Reports every phase's units at its barriers and stops wherever it says to.
    pub progress: Option<Progress<'config>>,
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

impl Default for SequenceRefineConfig<'_> {
    fn default() -> Self {
        SequenceRefineConfig::with_plan(None)
    }
}

impl From<SequenceRefinePlan> for SequenceRefineConfig<'_, SequenceRefinePlan> {
    /// A deferred call owning `plan`, with every other knob derived.
    fn from(plan: SequenceRefinePlan) -> Self {
        SequenceRefineConfig::with_plan(Some(plan))
    }
}

impl<Plan> SequenceRefineConfig<'_, Plan> {
    fn with_plan(plan: Option<Plan>) -> Self {
        SequenceRefineConfig {
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

/// The handles a reshard moves between and its knobs; see [`SequenceVerbs::reshard`].
pub struct SequenceReshardConfig<'config, Plan = &'config mut SequenceReshardPlan> {
    /// The handles whose entries move, each named once.
    pub sources: &'config [&'config SequenceIndex],
    /// The handles the entries move into, each a source itself or empty.
    pub targets: &'config [&'config SequenceIndex],
    /// The breadth of any walk beside the closing rounds; zero derives it.
    pub expansion_factor: usize,
    /// Bytes a loaded NVMe target holds for the one part it rewrites; zero derives it.
    pub memory_limit: usize,
    /// Picks the part roots of a source cut among several targets.
    pub seed: u64,
    /// Reports the closing rounds' phases and stops them wherever it says to, never a move.
    pub progress: Option<Progress<'config>>,
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

impl Default for SequenceReshardConfig<'_> {
    fn default() -> Self {
        SequenceReshardConfig::with_plan(None)
    }
}

impl From<SequenceReshardPlan> for SequenceReshardConfig<'_, SequenceReshardPlan> {
    /// A deferred call owning `plan`, naming no handle, since a live plan already holds them.
    fn from(plan: SequenceReshardPlan) -> Self {
        SequenceReshardConfig::with_plan(Some(plan))
    }
}

impl<Plan> SequenceReshardConfig<'_, Plan> {
    fn with_plan(plan: Option<Plan>) -> Self {
        SequenceReshardConfig {
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

/// A refine in flight between calls, which [`SequenceVerbs::refine_plan_finish`] closes.
#[derive(Debug)]
pub struct SequenceRefinePlan {
    handle: usearch_sequence_refine_plan_t,
}

/// A reshard in flight between calls, which [`SequenceVerbs::reshard_plan_finish`] closes.
#[derive(Debug)]
pub struct SequenceReshardPlan {
    handle: usearch_sequence_reshard_plan_t,
}

impl Default for SequenceRefinePlan {
    /// A fresh plan, which the first call holding it opens.
    fn default() -> Self {
        SequenceRefinePlan { handle: ptr::null_mut() }
    }
}

impl Default for SequenceReshardPlan {
    /// A fresh plan, which the first call holding it opens.
    fn default() -> Self {
        SequenceReshardPlan { handle: ptr::null_mut() }
    }
}

// Safety: a plan is driven by one call at a time, which the `&mut` holding it enforces.
unsafe impl Send for SequenceRefinePlan {}
unsafe impl Send for SequenceReshardPlan {}

impl<Offset: Clone + 'static> Accepts<SequenceInsertConfig<'static, Offset>> for Deferring {}
impl Accepts<SequenceSearchConfig<'static>> for Deferring {}
impl<Plan: 'static> Accepts<SequenceRefineConfig<'static, Plan>> for Deferring {}
impl<Plan: 'static> Accepts<SequenceReshardConfig<'static, Plan>> for Deferring {}

// endregion: Configs

// region: Sequence Index

/// Sequence Search Index over byte strings under edit distances and alignments, whose verbs come
/// from [`SequenceVerbs`]. [`SequenceIndex::clear`] alone takes `&mut self`.
///
/// ```no_run
/// use usearch::*;
///
/// let index = SequenceIndex::init_ram(&SequenceOptions::default(), &[])?;
/// index.insert([Key::from(1u64), Key::from(2u64)], Tape::of(["kitten", "sitting"]))?;
/// let mut arena = Arena::new()?;
/// let matches = index.search(Tape::of(["sitten"]), 1, &mut arena)?;
/// assert_eq!(matches.query(0)?.keys, [Key::from(1u64)]);
/// # Ok::<(), usearch::Error>(())
/// ```
///
/// Dropping it waits for every deferred call naming the index, then frees it.
#[derive(Debug)]
pub struct SequenceIndex {
    handle: usearch_sequence_index_t,
}

// Safety: the library documents a handle as shareable across threads.
unsafe impl Send for SequenceIndex {}
unsafe impl Sync for SequenceIndex {}

impl Drop for SequenceIndex {
    fn drop(&mut self) {
        unsafe { usearch_sequence_free(self.handle) }
    }
}

impl SequenceIndex {
    /// An index whose handle a constructor is about to fill.
    fn null() -> Self {
        SequenceIndex { handle: ptr::null_mut() }
    }

    /// The receiver whose verbs block on `executor`, contended while another call holds it.
    ///
    /// Rust-only.
    pub fn on<'index>(&'index self, executor: &'index Executor) -> On<'index, SequenceIndex> {
        On { index: self, executor, completion: PhantomData }
    }

    /// The receiver whose verbs submit their calls to `executor`, answering a
    /// [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
    ///
    /// Rust-only.
    pub fn deferred<'index>(&'index self, executor: &'index Executor) -> On<'index, SequenceIndex, Deferring> {
        On { index: self, executor, completion: PhantomData }
    }

    /// Erases all entries on the calling thread, keeping the reserved capacity and frozen kind.
    pub fn clear(&mut self) -> Result<()> {
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_sequence_clear(&mut usearch_sequence_clear_t {
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
    pub fn init_ram(options: &SequenceOptions, memory_domains: &[i32]) -> Result<Self> {
        let mut index = SequenceIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let mut weights = usearch_sequence_weights_t { substitution_matrix: ptr::null(), gap: 0 };
        unsafe {
            usearch_sequence_init_ram(&mut usearch_sequence_init_ram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                metric: options.metric as usearch_sequence_metric_t,
                weights: weights_slot(options.weights.as_ref(), &mut weights),
                window_widths: options.window_widths.as_ptr(),
                window_widths_count: options.window_widths.len(),
                connectivity: options.connectivity,
                values_per_key: options.values_per_key,
                ram: ram_options(memory_domains, options.cpu_class_ceiling),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Creates an empty, mutable index in device memory, one shard per device; empty derives one
    /// free device.
    pub fn init_vram(options: &SequenceOptions, devices: &[i32]) -> Result<Self> {
        let mut index = SequenceIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let mut weights = usearch_sequence_weights_t { substitution_matrix: ptr::null(), gap: 0 };
        unsafe {
            usearch_sequence_init_vram(&mut usearch_sequence_init_vram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                metric: options.metric as usearch_sequence_metric_t,
                weights: weights_slot(options.weights.as_ref(), &mut weights),
                window_widths: options.window_widths.as_ptr(),
                window_widths_count: options.window_widths.len(),
                connectivity: options.connectivity,
                values_per_key: options.values_per_key,
                vram: vram_options(devices, options.gpu_class_ceiling),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a checkpoint into a new, mutable index in host memory, shard by shard.
    pub fn load_ram<'config>(config: impl Into<SequenceLoadRamConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let mut index = SequenceIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let mut weights = usearch_sequence_weights_t { substitution_matrix: ptr::null(), gap: 0 };
        let progress = lower_progress(&config.progress, config.progress_steps);
        unsafe {
            usearch_sequence_load_ram(&mut usearch_sequence_load_ram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                ram: ram_options(config.memory_domains, config.cpu_class_ceiling),
                values_per_key: config.values_per_key,
                weights: weights_slot(config.weights.as_ref(), &mut weights),
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a checkpoint into a new index on devices, part by part.
    pub fn load_vram<'config>(config: impl Into<SequenceLoadVramConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let mut index = SequenceIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let mut weights = usearch_sequence_weights_t { substitution_matrix: ptr::null(), gap: 0 };
        let progress = lower_progress(&config.progress, config.progress_steps);
        unsafe {
            usearch_sequence_load_vram(&mut usearch_sequence_load_vram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                vram: vram_options(config.devices, config.gpu_class_ceiling),
                values_per_key: config.values_per_key,
                weights: weights_slot(config.weights.as_ref(), &mut weights),
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a serialized monolith into a new, mutable index in host memory.
    pub fn load_ram_buffer<'config>(config: impl Into<SequenceLoadRamBufferConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let mut index = SequenceIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let mut weights = usearch_sequence_weights_t { substitution_matrix: ptr::null(), gap: 0 };
        let progress = lower_progress(&config.progress, config.progress_steps);
        unsafe {
            usearch_sequence_load_ram_buffer(&mut usearch_sequence_load_ram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                ram: ram_options(config.memory_domains, config.cpu_class_ceiling),
                values_per_key: config.values_per_key,
                weights: weights_slot(config.weights.as_ref(), &mut weights),
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a serialized monolith into a new index on devices.
    pub fn load_vram_buffer<'config>(config: impl Into<SequenceLoadVramBufferConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let mut index = SequenceIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let mut weights = usearch_sequence_weights_t { substitution_matrix: ptr::null(), gap: 0 };
        let progress = lower_progress(&config.progress, config.progress_steps);
        unsafe {
            usearch_sequence_load_vram_buffer(&mut usearch_sequence_load_vram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                vram: vram_options(config.devices, config.gpu_class_ceiling),
                values_per_key: config.values_per_key,
                weights: weights_slot(config.weights.as_ref(), &mut weights),
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Serves a checkpoint in place through a memory mapping, read-only; the files must stay
    /// unmodified while it lives.
    pub fn view_ram(path: impl AsRef<Path>, weights: Option<&SequenceWeights>) -> Result<Self> {
        let mut index = SequenceIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let mut lowered = usearch_sequence_weights_t { substitution_matrix: ptr::null(), gap: 0 };
        unsafe {
            usearch_sequence_view_ram(&mut usearch_sequence_view_ram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                path: path_view(path.as_ref())?,
                weights: weights_slot(weights, &mut lowered),
                ram: ram_options(&[], None),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Serves a checkpoint in place from NVMe drives, read-only and sized past RAM.
    pub fn view_nvme<'config>(config: impl Into<SequenceViewNvmeConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let views = path_views(config.paths)?;
        let mut index = SequenceIndex::null();
        let mut outcome = usearch_outcome_t::default();
        let mut weights = usearch_sequence_weights_t { substitution_matrix: ptr::null(), gap: 0 };
        unsafe {
            usearch_sequence_view_nvme(&mut usearch_sequence_view_nvme_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                nvme: nvme_options(&views, config.cpu_class_ceiling),
                weights: weights_slot(config.weights.as_ref(), &mut weights),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Reads a checkpoint's kind and graph without loading it.
    pub fn metadata(path: impl AsRef<Path>) -> Result<SequenceMetadata> {
        let path = path_view(path.as_ref())?;
        let mut metadata = SequenceMetadata::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_sequence_metadata_read(&mut usearch_sequence_metadata_read_t {
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path,
                metadata: &mut metadata,
            })
        };
        check(outcome).map(|()| metadata)
    }

    /// Reads a checkpoint's kind and graph out of a serialized monolith buffer.
    pub fn metadata_buffer(buffer: &[u8]) -> Result<SequenceMetadata> {
        let mut metadata = SequenceMetadata::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_sequence_metadata_read_buffer(&mut usearch_sequence_metadata_read_buffer_t {
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
type Deferrable<'index, Receiver, Verb> =
    <<Receiver as SequenceVerbs<'index>>::Completion as Completion>::Deferrable<'index, Verb>;

/// Every sequence verb, written once for every receiver: `&SequenceIndex` blocks on the calling
/// thread, and the [`On`] that [`SequenceIndex::on`] or [`SequenceIndex::deferred`] makes runs on
/// an executor.
///
/// Rust-only.
pub trait SequenceVerbs<'index>: Copy {
    /// [`Blocking`] or [`Deferring`].
    type Completion: Completion;

    /// The index this receiver reaches and the executor its calls run on.
    #[doc(hidden)]
    fn handles(&self) -> (usearch_sequence_index_t, usearch_executor_t);

    /// Reserves room for `entries` entries and `payload_bytes` bytes across them, a hint, since
    /// insertions grow the index on demand.
    fn reserve(self, entries: usize, payload_bytes: usize) -> Result<Deferrable<'index, Self, SequenceReserve>> {
        let (index, executor) = self.handles();
        Self::Completion::answer(SequenceReserve { handles: Handles::of(index, executor), entries, payload_bytes })
    }

    /// Reports the frozen metadata and the current counters in one call.
    fn describe(self) -> Result<Deferrable<'index, Self, SequenceDescribe>> {
        let (index, executor) = self.handles();
        Self::Completion::answer(SequenceDescribe {
            handles: Handles::of(index, executor),
            description: SequenceIndexDescription::default(),
        })
    }

    /// The entries stored under `key`, with their total and largest byte lengths.
    fn count(self, key: Key) -> Result<Deferrable<'index, Self, SequenceCount>> {
        let (index, executor) = self.handles();
        Self::Completion::answer(SequenceCount {
            handles: Handles::of(index, executor),
            key,
            key_stats: SequenceKeyStats::default(),
        })
    }

    /// Whether the index holds at least one entry under `key`.
    ///
    /// Rust-only.
    fn contains(self, key: Key) -> Result<bool>
    where
        Self: SequenceVerbs<'index, Completion = Blocking>,
    {
        Ok(self.count(key)?.entries != 0)
    }

    /// Retrieves every sequence stored under `key` into `arena`, which the result points into; an
    /// absent key yields none.
    fn get<'arena>(
        self,
        key: Key,
        arena: &'arena mut Arena<'_>,
    ) -> Result<Deferrable<'index, Self, SequenceGet<'arena>>> {
        let (index, executor) = self.handles();
        Self::Completion::answer(SequenceGet {
            handles: Handles::of(index, executor),
            key,
            arena,
            sequences: TapeEntries::new(),
        })
    }

    /// Adds a tape of sequences under their keys, all-or-nothing and visible at once.
    fn insert<'config, Keys: AsRef<[Key]>, Offset: TapeOffset>(
        self,
        keys: Keys,
        config: impl Into<SequenceInsertConfig<'config, Offset>>,
    ) -> Result<Deferrable<'index, Self, SequenceInsert<'config, Keys, Offset>>>
    where
        Self::Completion: Accepts<Keys> + Accepts<SequenceInsertConfig<'config, Offset>>,
    {
        let (index, executor) = self.handles();
        let config = config.into();
        // The keys view holds no count, so only this side can see the two disagree.
        if config.sequences.count() != keys.as_ref().len() {
            return Err(fail(Status::ShapeMismatch, "the key count does not match the tape's entries"));
        }
        Self::Completion::answer(SequenceInsert { handles: Handles::of(index, executor), keys, config })
    }

    /// Runs one refine round over the whole graph, answering a [`RefineReport`].
    fn refine<'config, Plan: BorrowMut<SequenceRefinePlan>>(
        self,
        config: SequenceRefineConfig<'config, Plan>,
    ) -> Result<Deferrable<'index, Self, SequenceRefine<'config, Plan>>>
    where
        Self::Completion: Accepts<SequenceRefineConfig<'config, Plan>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer(SequenceRefine {
            handles: Handles::of(index, executor),
            config,
            report: RefineReport::default(),
        })
    }

    /// Closes a refine plan, whatever the call answers, and answers what the whole plan did.
    fn refine_plan_finish(
        self,
        plan: SequenceRefinePlan,
    ) -> Result<Deferrable<'index, Self, SequenceRefinePlanFinish>> {
        let (index, executor) = self.handles();
        Self::Completion::answer(SequenceRefinePlanFinish {
            handles: Handles::of(index, executor),
            plan,
            report: RefineReport::default(),
        })
    }

    /// Measures how much of the graph a search reaches from the root.
    fn health(self) -> Result<Deferrable<'index, Self, SequenceHealth>> {
        let (index, executor) = self.handles();
        Self::Completion::answer(SequenceHealth {
            handles: Handles::of(index, executor),
            reachability: DenseReachability::default(),
        })
    }

    /// Resolves a tape of queries into up to `wanted` nearest matches apiece, writing them into
    /// `arena` as [`Matches`].
    fn search<'config, 'arena, Offset: TapeOffset>(
        self,
        queries: Tape<'config, u8, Offset>,
        config: impl Into<SequenceSearchConfig<'config>>,
        arena: &'arena mut Arena<'_>,
    ) -> Result<Deferrable<'index, Self, SequenceSearch<'config, 'arena, Offset>>>
    where
        Self::Completion: Accepts<Tape<'config, u8, Offset>> + Accepts<SequenceSearchConfig<'config>>,
    {
        let (index, executor) = self.handles();
        let mut matches = Matches::new(arena);
        matches.queries = queries.count();
        Self::Completion::answer(SequenceSearch {
            handles: Handles::of(index, executor),
            queries,
            config: config.into(),
            matches,
        })
    }

    /// Moves the entries of the sources into the targets, keeping their links, then runs a round
    /// on every target that took entries: several sources into one target gather, one source into
    /// several targets cuts, and one handle as its only source and target recuts in place.
    fn reshard<'config, Plan: BorrowMut<SequenceReshardPlan>>(
        self,
        config: SequenceReshardConfig<'config, Plan>,
    ) -> Result<Deferrable<'index, Self, SequenceReshard<'config, Plan>>>
    where
        Self::Completion: Accepts<SequenceReshardConfig<'config, Plan>>,
    {
        let (_, executor) = self.handles();
        let handles = |named: &[&SequenceIndex]| named.iter().map(|named| named.handle).collect();
        let lists = ReshardHandles { sources: handles(config.sources), targets: handles(config.targets) };
        Self::Completion::answer(SequenceReshard {
            handles: Handles::of(ptr::null_mut(), executor),
            config,
            lists,
            report: RefineReport::default(),
        })
    }

    /// Closes a reshard plan, whatever the call answers, and answers what the whole plan did.
    fn reshard_plan_finish(
        self,
        plan: SequenceReshardPlan,
    ) -> Result<Deferrable<'index, Self, SequenceReshardPlanFinish>> {
        let (_, executor) = self.handles();
        Self::Completion::answer(SequenceReshardPlanFinish {
            handles: Handles::of(ptr::null_mut(), executor),
            plan,
            report: RefineReport::default(),
        })
    }

    /// Draws keys around the focus, or across the whole index, writing up to `wanted` of them
    /// into `arena` as a [`Sample`].
    fn sample<'config, 'arena>(
        self,
        wanted: usize,
        arena: &'arena mut Arena<'_>,
        config: DenseSampleConfig<'config>,
    ) -> Result<Deferrable<'index, Self, SequenceSample<'config, 'arena>>>
    where
        Self::Completion: Accepts<DenseSampleConfig<'config>>,
    {
        // Two slices name the shown keys and their places, so only this side sees them disagree.
        if config.shown.len() != config.shown_coordinates.len() {
            return Err(fail(Status::InvalidArgument, "every shown key needs its coordinates"));
        }
        let (index, executor) = self.handles();
        Self::Completion::answer(SequenceSample {
            handles: Handles::of(index, executor),
            wanted,
            config,
            sample: Sample::new(arena),
        })
    }

    /// Saves the index in the layout the path extension picks - one `.safegraphs` monolith, or a
    /// `.safegraphs.json` manifest beside its shards.
    fn save<'config, Location: AsRef<Path>>(
        self,
        config: impl Into<DenseSaveConfig<'config, Location>>,
    ) -> Result<Deferrable<'index, Self, SequenceSave<'config, Location>>>
    where
        Self::Completion: Accepts<DenseSaveConfig<'config, Location>>,
    {
        let (index, executor) = self.handles();
        let config = config.into();
        let path = path_view(config.path.as_ref())?;
        Self::Completion::answer(SequenceSave { handles: Handles::of(index, executor), path, config })
    }

    /// Saves the index as one monolith into the arena its config holds.
    fn save_buffer<'config, 'arena, 'executor>(
        self,
        config: impl Into<DenseSaveBufferConfig<'config, 'arena, 'executor>>,
    ) -> Result<Deferrable<'index, Self, SequenceSaveBuffer<'config, 'arena>>>
    where
        'executor: 'arena,
        Self::Completion: Accepts<DenseSaveBufferConfig<'config, 'arena, 'executor>>,
    {
        let (index, executor) = self.handles();
        let DenseSaveBufferConfig { arena, progress, progress_steps } = config.into();
        Self::Completion::answer(SequenceSaveBuffer {
            handles: Handles::of(index, executor),
            progress,
            progress_steps,
            monolith: Monolith::new(arena),
        })
    }
}

impl<'index> SequenceVerbs<'index> for &'index SequenceIndex {
    type Completion = Blocking;

    fn handles(&self) -> (usearch_sequence_index_t, usearch_executor_t) {
        (self.handle, ptr::null_mut())
    }
}

impl<'index, Completion: crate::types::Completion> SequenceVerbs<'index> for On<'index, SequenceIndex, Completion> {
    type Completion = Completion;

    fn handles(&self) -> (usearch_sequence_index_t, usearch_executor_t) {
        (self.index.handle, self.executor.handle)
    }
}

impl SequenceIndex {
    /// Opens a walk over the current contents: without `starts` every entry as stored, with them
    /// the start keys' entries first, then the entries those link to breadth-first.
    pub fn cursor(&self, starts: &[Key]) -> Result<SequenceCursor<'_>> {
        let mut cursor = SequenceCursor { handle: ptr::null_mut(), index: self.handle, owner: PhantomData };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_sequence_cursor_init(&mut usearch_sequence_cursor_init_t {
                index: self.handle,
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                starts_count: starts.len(),
                starts: starts.as_ptr(),
                cursor: &mut cursor.handle,
            })
        };
        check(outcome).map(|()| cursor)
    }
}

/// A walk over the index contents; see [`SequenceIndex::cursor`].
///
/// Its pulls run on the calling thread, or blocking on the executor [`SequenceCursor::on`] names
/// for them; the cursor keeps none.
///
/// Dropping it waits for every deferred call naming the cursor, then releases it.
#[derive(Debug)]
pub struct SequenceCursor<'index> {
    handle: usearch_sequence_cursor_t,
    index: usearch_sequence_index_t,
    owner: PhantomData<&'index SequenceIndex>,
}

impl<'index> SequenceCursor<'index> {
    /// The receiver whose pulls block on `executor`, contended while another call holds it.
    ///
    /// Rust-only.
    pub fn on<'cursor>(&'cursor mut self, executor: &'cursor Executor) -> On<'cursor, SequenceCursor<'index>> {
        On { index: self, executor, completion: PhantomData }
    }

    /// Copies up to `wanted` next keys into `arena` on the calling thread; an empty batch means the
    /// walk is exhausted.
    pub fn next_keys<'arena>(&mut self, wanted: usize, arena: &'arena mut Arena<'_>) -> Result<DenseBatch<'arena>> {
        self.pull(ptr::null_mut(), false, wanted, arena)
    }

    /// Copies up to `wanted` next keys into `arena` on the calling thread, each with the keys it
    /// links to.
    pub fn next_links<'arena>(&mut self, wanted: usize, arena: &'arena mut Arena<'_>) -> Result<DenseBatch<'arena>> {
        self.pull(ptr::null_mut(), true, wanted, arena)
    }

    /// Pulls a batch on `executor`, exporting each entry's links only when `linked` asks.
    fn pull<'arena>(
        &self,
        executor: usearch_executor_t,
        linked: bool,
        wanted: usize,
        arena: &'arena mut Arena<'_>,
    ) -> Result<DenseBatch<'arena>> {
        let mut batch = DenseBatch {
            arena: PhantomData,
            keys: usearch_keys_view_t::default(),
            links: linked.then(usearch_keys_tape_view_t::default),
            written: 0,
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_sequence_cursor_next(&mut usearch_sequence_cursor_next_t {
                index: self.index,
                executor,
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

impl On<'_, SequenceCursor<'_>> {
    /// Copies up to `wanted` next keys into `arena` on this receiver's executor.
    pub fn next_keys<'arena>(self, wanted: usize, arena: &'arena mut Arena<'_>) -> Result<DenseBatch<'arena>> {
        self.index.pull(self.executor.handle, false, wanted, arena)
    }

    /// Copies up to `wanted` next keys into `arena` on this receiver's executor, each with the keys
    /// it links to.
    pub fn next_links<'arena>(self, wanted: usize, arena: &'arena mut Arena<'_>) -> Result<DenseBatch<'arena>> {
        self.index.pull(self.executor.handle, true, wanted, arena)
    }
}

impl Drop for SequenceCursor<'_> {
    fn drop(&mut self) {
        unsafe { usearch_sequence_cursor_free(self.handle) }
    }
}

// endregion: Sequence Index

// region: Calls

/// What [`SequenceVerbs::reserve`] runs.
pub struct SequenceReserve {
    handles: Handles,
    entries: usize,
    payload_bytes: usize,
}

impl Calling for SequenceReserve {
    type Arguments = usearch_sequence_reserve_t;
    type Output = ();

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        usearch_sequence_reserve_t {
            index,
            executor,
            outcome,
            completion,
            entries: self.entries,
            payload_bytes: self.payload_bytes,
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_reserve(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<()> {
        status
    }
}

/// What [`SequenceVerbs::describe`] runs, and the description it fills.
pub struct SequenceDescribe {
    handles: Handles,
    description: SequenceIndexDescription,
}

impl Calling for SequenceDescribe {
    type Arguments = usearch_sequence_describe_t;
    type Output = SequenceIndexDescription;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        usearch_sequence_describe_t { index, executor, outcome, completion, description: &mut self.description }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_describe(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<SequenceIndexDescription> {
        status.map(|()| self.description)
    }
}

/// What [`SequenceVerbs::count`] runs, and the key's stats it fills.
pub struct SequenceCount {
    handles: Handles,
    key: Key,
    key_stats: SequenceKeyStats,
}

impl Calling for SequenceCount {
    type Arguments = usearch_sequence_count_t;
    type Output = SequenceKeyStats;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        usearch_sequence_count_t { index, executor, outcome, completion, key: self.key, key_stats: &mut self.key_stats }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_count(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<SequenceKeyStats> {
        status.map(|()| self.key_stats)
    }
}

/// What [`SequenceVerbs::get`] runs, and the tape it exports.
pub struct SequenceGet<'arena> {
    handles: Handles,
    key: Key,
    arena: &'arena Arena<'arena>,
    sequences: TapeEntries<'arena, u8>,
}

impl<'arena> Calling for SequenceGet<'arena> {
    type Arguments = usearch_sequence_get_t;
    type Output = TapeEntries<'arena, u8>;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        usearch_sequence_get_t {
            index,
            executor,
            outcome,
            completion,
            key: self.key,
            sequences: &mut self.sequences.view,
            written: &mut self.sequences.written,
            arena: self.arena.handle,
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_get(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<TapeEntries<'arena, u8>> {
        status.map(|()| self.sequences)
    }
}

/// What [`SequenceVerbs::insert`] runs: the keys and the config holding their tape.
pub struct SequenceInsert<'config, Keys, Offset: Clone> {
    handles: Handles,
    keys: Keys,
    config: SequenceInsertConfig<'config, Offset>,
}

impl<Keys: AsRef<[Key]>, Offset: TapeOffset> Calling for SequenceInsert<'_, Keys, Offset> {
    type Arguments = usearch_sequence_insert_t;
    type Output = ();

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        let (keys, config) = (self.keys.as_ref(), &mut self.config);
        usearch_sequence_insert_t {
            index,
            executor,
            outcome,
            completion,
            count: keys.len(),
            keys: usearch_keys_view_t::of(keys),
            sequences: config.sequences.view(Scalar::U8),
            expansion_factor: config.expansion_factor,
            progress: lower_progress(&config.progress, config.progress_steps),
            stats: output_slot(&mut config.stats),
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_insert(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<()> {
        status
    }
}

/// What [`SequenceVerbs::refine`] runs: its config, plan included, and the report it fills.
pub struct SequenceRefine<'config, Plan> {
    handles: Handles,
    config: SequenceRefineConfig<'config, Plan>,
    report: RefineReport,
}

impl<'config, Plan: BorrowMut<SequenceRefinePlan>> Calling for SequenceRefine<'config, Plan> {
    type Arguments = usearch_sequence_refine_t;
    type Output = RefineReport;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        let config = &mut self.config;
        usearch_sequence_refine_t {
            index,
            executor,
            outcome,
            completion,
            report: &mut self.report,
            refine: usearch_refine_options_t {
                expansion_factor: config.expansion_factor,
                memory_limit: config.memory_limit,
            },
            progress: lower_progress(&config.progress, config.progress_steps),
            stats: output_slot(&mut config.stats),
            plan: config.plan.as_mut().map_or(ptr::null_mut(), |plan| &mut plan.borrow_mut().handle),
            phase: output_slot(&mut config.phase),
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_refine(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<RefineReport> {
        status.map(|()| self.report)
    }
}

impl<'config, Plan: BorrowMut<SequenceRefinePlan>> Submitting for SequenceRefine<'config, Plan> {
    type Inputs = SequenceRefineConfig<'config, Plan>;

    fn finish_with_inputs(self, status: Result<()>) -> (Result<RefineReport>, Self::Inputs) {
        (status.map(|()| self.report), self.config)
    }
}

/// What [`SequenceVerbs::refine_plan_finish`] runs: the plan it closes and the report it fills.
pub struct SequenceRefinePlanFinish {
    handles: Handles,
    plan: SequenceRefinePlan,
    report: RefineReport,
}

impl Calling for SequenceRefinePlanFinish {
    type Arguments = usearch_sequence_refine_plan_finish_t;
    type Output = RefineReport;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        usearch_sequence_refine_plan_finish_t {
            index,
            executor,
            outcome,
            completion,
            plan: self.plan.handle,
            report: &mut self.report,
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_refine_plan_finish(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<RefineReport> {
        status.map(|()| self.report)
    }
}

/// What [`SequenceVerbs::health`] runs, and the reachability it fills.
pub struct SequenceHealth {
    handles: Handles,
    reachability: DenseReachability,
}

impl Calling for SequenceHealth {
    type Arguments = usearch_sequence_health_t;
    type Output = DenseReachability;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        usearch_sequence_health_t { index, executor, outcome, completion, reachability: &mut self.reachability }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_health(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<DenseReachability> {
        status.map(|()| self.reachability)
    }
}

/// What [`SequenceVerbs::search`] runs: its query tape and config, and the matches it writes.
pub struct SequenceSearch<'config, 'arena, Offset: Clone> {
    handles: Handles,
    queries: Tape<'config, u8, Offset>,
    config: SequenceSearchConfig<'config>,
    matches: Matches<'arena>,
}

impl<'arena, Offset: TapeOffset> Calling for SequenceSearch<'_, 'arena, Offset> {
    type Arguments = usearch_sequence_search_t;
    type Output = Matches<'arena>;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        let config = &mut self.config;
        usearch_sequence_search_t {
            index,
            executor,
            outcome,
            completion,
            count: self.queries.count(),
            queries: self.queries.view(Scalar::U8),
            matches: &mut self.matches.view,
            arena: self.matches.arena.handle,
            wanted: config.wanted,
            expansion_factor: config.expansion_factor,
            traversal: config.traversal as usearch_traversal_t,
            predicate: lower_predicate(&config.predicate),
            progress: lower_progress(&config.progress, config.progress_steps),
            stats: output_slot(&mut config.stats),
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_search(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<Matches<'arena>> {
        status.map(|()| self.matches)
    }
}

/// What [`SequenceVerbs::reshard`] runs: its config, plan included, the handles it passes and the
/// report it fills.
pub struct SequenceReshard<'config, Plan> {
    handles: Handles,
    config: SequenceReshardConfig<'config, Plan>,
    lists: ReshardHandles,
    report: RefineReport,
}

impl<'config, Plan: BorrowMut<SequenceReshardPlan>> Calling for SequenceReshard<'config, Plan> {
    type Arguments = usearch_sequence_reshard_t;
    type Output = RefineReport;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let (config, lists) = (&mut self.config, &self.lists);
        usearch_sequence_reshard_t {
            sources_count: lists.sources.len(),
            sources: lists.sources.as_ptr(),
            targets_count: lists.targets.len(),
            targets: lists.targets.as_ptr(),
            executor: self.handles.executor,
            outcome,
            completion,
            report: &mut self.report,
            refine: usearch_refine_options_t {
                expansion_factor: config.expansion_factor,
                memory_limit: config.memory_limit,
            },
            seed: config.seed,
            progress: lower_progress(&config.progress, config.progress_steps),
            stats: output_slot(&mut config.stats),
            plan: config.plan.as_mut().map_or(ptr::null_mut(), |plan| &mut plan.borrow_mut().handle),
            phase: output_slot(&mut config.phase),
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_reshard(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<RefineReport> {
        status.map(|()| self.report)
    }
}

impl<'config, Plan: BorrowMut<SequenceReshardPlan>> Submitting for SequenceReshard<'config, Plan> {
    type Inputs = SequenceReshardConfig<'config, Plan>;

    fn finish_with_inputs(self, status: Result<()>) -> (Result<RefineReport>, Self::Inputs) {
        (status.map(|()| self.report), self.config)
    }
}

/// What [`SequenceVerbs::reshard_plan_finish`] runs: the plan it closes and the report it fills.
pub struct SequenceReshardPlanFinish {
    handles: Handles,
    plan: SequenceReshardPlan,
    report: RefineReport,
}

impl Calling for SequenceReshardPlanFinish {
    type Arguments = usearch_sequence_reshard_plan_finish_t;
    type Output = RefineReport;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        usearch_sequence_reshard_plan_finish_t {
            executor: self.handles.executor,
            outcome,
            completion,
            plan: self.plan.handle,
            report: &mut self.report,
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_reshard_plan_finish(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<RefineReport> {
        status.map(|()| self.report)
    }
}

/// What [`SequenceVerbs::sample`] runs: its config and the draws it writes.
pub struct SequenceSample<'config, 'arena> {
    handles: Handles,
    wanted: usize,
    config: DenseSampleConfig<'config>,
    sample: Sample<'arena>,
}

impl<'arena> Calling for SequenceSample<'_, 'arena> {
    type Arguments = usearch_sequence_sample_t;
    type Output = Sample<'arena>;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        let (config, sample) = (&mut self.config, &mut self.sample);
        usearch_sequence_sample_t {
            index,
            executor,
            outcome,
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
            wanted: self.wanted,
            seed: config.seed,
            expansion_factor: config.expansion_factor,
            progress: lower_progress(&config.progress, config.progress_steps),
            stats: output_slot(&mut config.stats),
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_sample(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<Sample<'arena>> {
        status.map(|()| self.sample)
    }
}

/// What [`SequenceVerbs::save`] runs: its config and the path view into it.
pub struct SequenceSave<'config, Location> {
    handles: Handles,
    path: usearch_bytes_view_t,
    config: DenseSaveConfig<'config, Location>,
}

impl<Location> Calling for SequenceSave<'_, Location> {
    type Arguments = usearch_sequence_save_t;
    type Output = ();

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        usearch_sequence_save_t {
            index,
            executor,
            outcome,
            completion,
            path: self.path,
            progress: lower_progress(&self.config.progress, self.config.progress_steps),
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_save(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<()> {
        status
    }
}

/// What [`SequenceVerbs::save_buffer`] runs: its progress and the monolith it writes.
pub struct SequenceSaveBuffer<'config, 'arena> {
    handles: Handles,
    progress: Option<Progress<'config>>,
    progress_steps: usize,
    monolith: Monolith<'arena>,
}

impl<'arena> Calling for SequenceSaveBuffer<'_, 'arena> {
    type Arguments = usearch_sequence_save_buffer_t;
    type Output = Monolith<'arena>;

    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments {
        let Handles { handle: index, executor } = self.handles;
        usearch_sequence_save_buffer_t {
            index,
            executor,
            outcome,
            completion,
            monolith: &mut self.monolith.monolith,
            arena: self.monolith.arena.handle,
            progress: lower_progress(&self.progress, self.progress_steps),
        }
    }

    unsafe fn call(arguments: *mut Self::Arguments) {
        unsafe { usearch_sequence_save_buffer(arguments) }
    }

    fn finish(self, status: Result<()>) -> Result<Monolith<'arena>> {
        status.map(|()| self.monolith)
    }
}

// endregion: Calls

// region: Tests

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;

    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::testing::{ScratchDirectory, case_stream_key, executor, keys, refused};
    use crate::types::Phase;

    // region: Helpers

    /// Random lowercase words of 8 to 16 letters, the same for the same seed.
    fn words(count: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut generator = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                let length = generator.random_range(8..17);
                (0..length).map(|_| generator.random_range(b'a'..=b'z')).collect()
            })
            .collect()
    }

    /// A host index holding `count` random words under consecutive keys from `first`.
    fn populated_index(count: usize, first: u64, seed: u64) -> Result<SequenceIndex> {
        let index = SequenceIndex::init_ram(&SequenceOptions::default(), &[])?;
        if count != 0 {
            index.on(&executor()).insert(keys(first, count), Tape::of(words(count, seed)))?;
        }
        Ok(index)
    }

    fn size(index: &SequenceIndex) -> Result<usize> {
        Ok(index.describe()?.size)
    }

    // endregion: Helpers

    // region: Construction

    #[test]
    fn init_ram_starts_empty_and_keeps_the_metric() {
        let options = SequenceOptions { metric: SequenceMetric::LevenshteinUtf8, ..SequenceOptions::default() };
        let described = SequenceIndex::init_ram(&options, &[]).unwrap().describe().unwrap();
        assert_eq!(described.size, 0);
        assert_eq!(described.metadata.metric, SequenceMetric::LevenshteinUtf8);
        assert_eq!(described.gpu_class(), "unknown", "a host index names no GPU class");
    }

    #[test]
    fn spells_every_metric_back() {
        for metric in [
            SequenceMetric::LevenshteinBytes,
            SequenceMetric::LevenshteinUtf8,
            SequenceMetric::NeedlemanWunsch,
            SequenceMetric::Overlap,
        ] {
            assert_eq!(metric.to_string().parse(), Ok(metric), "the library spells it back");
        }
        assert!("hamming".parse::<SequenceMetric>().is_err());
    }

    /// A VRAM index either opens on a GPU or answers the library's own refusal.
    #[test]
    fn init_vram_surfaces_the_library_status() {
        match SequenceIndex::init_vram(&SequenceOptions::default(), &[]) {
            Ok(index) => assert_ne!(index.describe().unwrap().gpu_class(), "unknown"),
            Err(error) => assert_ne!(error.status, Status::Deferred, "a refusal is final"),
        }
    }

    // endregion: Construction

    // region: Ingest and Retrieval

    /// Mirrors the Sequence Indexes snippet of rust/README.md line by line; only the function
    /// around it and its closing `Ok(())` differ.
    #[test]
    fn readme_quickstart() -> Result<()> {
        let options = SequenceOptions { metric: SequenceMetric::LevenshteinUtf8, ..SequenceOptions::default() };
        let index = SequenceIndex::init_ram(&options, &[])?;
        let words = Tape::of(["kitten", "sitting", "mitten"]);
        index.insert([Key::from(1u64), Key::from(2u64), Key::from(3u64)], words)?;

        let mut arena = Arena::new()?;
        let matches = index.search(Tape::of(["sittin"]), 1, &mut arena)?;
        assert_eq!(matches.query(0)?.keys, [Key::from(2u64)]);
        assert_eq!(matches.query(0)?.distances[0], 1.0); // one edit away
        Ok(())
    }

    /// Every sequence searched for comes back first under its own key, at distance zero.
    #[test]
    fn finds_the_inserted_entry() {
        let index = SequenceIndex::init_ram(&SequenceOptions::default(), &[]).unwrap();
        index.insert(keys(1, 3), Tape::of(["kitten", "sitting", "mitten"])).unwrap();
        assert_eq!(size(&index).unwrap(), 3);

        let mut arena = Arena::new().unwrap();
        let matches = index.search(Tape::of(["sitting"]), 1, &mut arena).unwrap();
        let best = matches.query(0).unwrap();
        assert_eq!((best.keys, best.distances[0]), (&[Key::from(2u64)][..], 0.0));
    }

    #[test]
    fn gets_and_counts_every_entry_under_a_key() {
        let index = SequenceIndex::init_ram(&SequenceOptions::default(), &[]).unwrap();
        let shared = Key::from(7u64);
        index.insert([shared, shared], Tape::of(["kitten", "sitting"])).unwrap();

        let stats = index.count(shared).unwrap();
        assert_eq!((stats.entries, stats.total_bytes, stats.largest_bytes), (2, 13, 7));
        assert!(index.contains(shared).unwrap() && !index.contains(Key::from(8u64)).unwrap());

        let mut arena = Arena::new().unwrap();
        let stored = index.get(shared, &mut arena).unwrap();
        assert_eq!(stored.iter().collect::<Vec<_>>(), [&b"kitten"[..], &b"sitting"[..]]);
        assert_eq!(index.get(Key::from(8u64), &mut arena).unwrap().written(), 0, "an absent key yields none");
    }

    #[test]
    fn rejects_mismatched_key_and_entry_counts() {
        let index = SequenceIndex::init_ram(&SequenceOptions::default(), &[]).unwrap();
        assert!(refused(index.insert(keys(0, 3), Tape::of(["kitten"])), Status::ShapeMismatch));
        assert_eq!(size(&index).unwrap(), 0);
    }

    // endregion: Ingest and Retrieval

    // region: Serialization

    #[test]
    fn saves_and_loads_a_file() {
        let scratch = ScratchDirectory::new("usearch-sequence-saves_and_loads_a_file");
        let path = scratch.path().join("index.safegraphs");
        let index = populated_index(32, 0, case_stream_key(0)).unwrap();
        index.on(&executor()).save(&path).unwrap();

        assert_eq!(SequenceIndex::metadata(&path).unwrap().metric, SequenceMetric::LevenshteinBytes);
        let restored = SequenceIndex::load_ram(&path).unwrap();
        assert_eq!(size(&restored).unwrap(), 32);
        assert!(restored.contains(Key::from(17u64)).unwrap());

        // A view serves the checkpoint in place and refuses changes, where RAM serves views at all.
        match SequenceIndex::view_ram(&path, None) {
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

        assert_eq!(SequenceIndex::metadata_buffer(monolith).unwrap().metric, SequenceMetric::LevenshteinBytes);
        assert_eq!(size(&SequenceIndex::load_ram_buffer(monolith).unwrap()).unwrap(), 32);
        assert!(
            SequenceIndex::load_ram_buffer(&monolith[..monolith.len() - 1]).is_err(),
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
        let report = index.on(&executor()).refine(SequenceRefineConfig::default()).unwrap();
        assert!(report.changed <= report.reselected);
        assert_eq!(report.reachability.audited, 64);
        assert!(index.contains(Key::from(17u64)).unwrap());
    }

    /// A plan keeps a stopped round live for the next call, and its finish reports the whole plan.
    #[test]
    fn a_refine_plan_resumes_and_finishes() {
        let index = populated_index(64, 0, case_stream_key(0)).unwrap();
        let executor = executor();
        let (mut plan, mut phase) = (SequenceRefinePlan::default(), PhaseProgress::default());
        let stopped = index.on(&executor).refine(SequenceRefineConfig {
            progress: Some(Progress::new(&mut |_, _| ControlFlow::Break(()))),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..SequenceRefineConfig::default()
        });
        if let Err(failure) = stopped {
            assert_eq!(failure.status, Status::Cancelled);
            assert!(phase.phase != Phase::Done && phase.completed <= phase.total);
            index
                .on(&executor)
                .refine(SequenceRefineConfig {
                    plan: Some(&mut plan),
                    phase: Some(&mut phase),
                    ..SequenceRefineConfig::default()
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
        let mut plan = SequenceRefinePlan::default();
        let refined = index.refine(SequenceRefineConfig { plan: Some(&mut plan), ..SequenceRefineConfig::default() });
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
            .reshard(SequenceReshardConfig {
                sources: &[&first, &second, &third],
                targets: &[&first],
                ..SequenceReshardConfig::default()
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
        let (mut plan, mut phase) = (SequenceReshardPlan::default(), PhaseProgress::default());
        let stopped = first.on(&executor).reshard(SequenceReshardConfig {
            sources: &[&first, &second],
            targets: &[&first],
            progress: Some(Progress::new(&mut |_, _| ControlFlow::Break(()))),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..SequenceReshardConfig::default()
        });
        if let Err(failure) = stopped {
            assert_eq!(failure.status, Status::Cancelled);
            assert_ne!(phase.phase, Phase::Done, "a stop names the phase it stopped in");
        }
        assert_eq!(size(&first).unwrap(), 32);

        // A deferred call owns the plan it continues and hands it back once final.
        let continued = first.deferred(&executor).reshard(SequenceReshardConfig::from(plan)).unwrap();
        let (finished, config) = continued.wait_with_inputs();
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
            .reshard(SequenceReshardConfig {
                sources: &[&source],
                targets: &[&targets[0], &targets[1]],
                seed: case_stream_key(1),
                ..SequenceReshardConfig::default()
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
        let (mut plan, mut phase) = (SequenceReshardPlan::default(), PhaseProgress::default());
        let stopped = source.on(&executor).reshard(SequenceReshardConfig {
            sources: &[&source],
            targets: &[&targets[0], &targets[1]],
            progress: Some(Progress::new(&mut |_, _| ControlFlow::Break(()))),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..SequenceReshardConfig::default()
        });
        assert_eq!(size(&targets[0]).unwrap() + size(&targets[1]).unwrap(), 64);
        if stopped.is_err() {
            source
                .on(&executor)
                .reshard(SequenceReshardConfig {
                    plan: Some(&mut plan),
                    phase: Some(&mut phase),
                    ..SequenceReshardConfig::default()
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
        let recut = index.on(&executor()).reshard(SequenceReshardConfig {
            sources: &[&index],
            targets: &[&index],
            ..SequenceReshardConfig::default()
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
        let reshard = |sources: &[&SequenceIndex], targets: &[&SequenceIndex]| {
            first.on(&executor).reshard(SequenceReshardConfig { sources, targets, ..SequenceReshardConfig::default() })
        };
        assert!(refused(reshard(&[], &[&first]), Status::InvalidArgument));
        assert!(refused(reshard(&[&first, &first], &[&first]), Status::InvalidArgument));
        assert!(refused(reshard(&[&first], &[&second]), Status::ConfigMismatch));
        assert_eq!((size(&first).unwrap(), size(&second).unwrap()), (8, 8));
    }

    // endregion: Resharding
}

// endregion: Tests
