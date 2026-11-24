//! Vector Search over equidimensional dense embeddings.
//!
//! Every batch input is a NumKong matrix, so a memory-mapped dataset feeds the index with no
//! reshape and no copy: packed slabs go through [`crate::rows`], padded ones through
//! `MatrixView::from_raw_parts`. Handles are created against one residency - RAM, VRAM, or NVMe -
//! by the constructor named after it, which also says what reading a checkpoint means:
//! [`DenseIndex::load_ram`] materializes it, [`DenseIndex::load_vram`] uploads it to devices, and
//! [`DenseIndex::view_nvme`] opens it in place on disk.
//!
//! Every verb comes from [`DenseVerbs`], written once over the receiver: a [`DenseIndex`] blocks on
//! the calling thread, `index.on(&executor)` blocks on an executor, and `index.deferred(&executor)`
//! answers a [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
//!
//! File: rust/dense_index.rs
//! Author: Ash Vardanian
//! Date: August 15, 2026

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types)]

use core::borrow::BorrowMut;
use core::ffi::{c_char, c_int, c_void};
use core::fmt;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ptr::{self, NonNull};
use core::slice;
use core::str::FromStr;
use std::path::Path;

use numkong::MatrixView;

use crate::types::{
    Accepts, Arena, ArrowDeviceArray, ArrowExport, ArrowSchema, Blocking, CPUClass, Completion, Deferring, Element,
    Error, Executor, GPUClass, Key, Matches, On, PhaseProgress, Predicate, ProgressCallback, QueryMatches, Residency,
    Result, RowsView, Scalar, Status, Traversal, arena_slice, arrow_checked, check, fail, lower_predicate,
    lower_progress, nvme_options, output_slot, path_view, path_views, ram_options, read_named, spelling,
    usearch_arena_t, usearch_bytes_view_t, usearch_completion_blocking_k, usearch_completion_t, usearch_cpu_class_name,
    usearch_cpu_class_t, usearch_executor_t, usearch_gpu_class_name, usearch_gpu_class_t, usearch_key_t,
    usearch_keys_tape_view_t, usearch_keys_view_t, usearch_load_path_name, usearch_load_path_t,
    usearch_matches_tape_view_t, usearch_matches_tape_view_to_arrow, usearch_matches_view_t, usearch_nvme_options_t,
    usearch_outcome_t, usearch_predicate_t, usearch_progress_t, usearch_ram_options_t,
    usearch_refine_verdict_from_name, usearch_refine_verdict_name, usearch_rows_view_t, usearch_rows_view_to_arrow,
    usearch_scalar_t, usearch_status_t, usearch_traversal_t, usearch_vram_options_t, values_of, vram_options,
};

// region: Raw Declarations

pub type usearch_dense_index_t = *mut c_void;
pub type usearch_dense_cursor_t = *mut c_void;
pub type usearch_dense_search_stream_t = *mut c_void;
pub type usearch_dense_metric_t = c_int;

#[repr(C)]
pub struct usearch_dense_init_ram_t {
    pub index: *mut usearch_dense_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub metric: usearch_dense_metric_t,
    pub quantization: usearch_scalar_t,
    pub dimensions: usize,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub ram: usearch_ram_options_t,
}

#[repr(C)]
pub struct usearch_dense_init_vram_t {
    pub index: *mut usearch_dense_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub metric: usearch_dense_metric_t,
    pub quantization: usearch_scalar_t,
    pub dimensions: usize,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub vram: usearch_vram_options_t,
}

#[repr(C)]
pub struct usearch_dense_load_ram_t {
    pub index: *mut usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_dense_load_vram_t {
    pub index: *mut usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub vram: usearch_vram_options_t,
    pub values_per_key: usize,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_dense_load_ram_buffer_t {
    pub index: *mut usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
    pub connectivity: usize,
    pub values_per_key: usize,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_dense_load_vram_buffer_t {
    pub index: *mut usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub vram: usearch_vram_options_t,
    pub values_per_key: usize,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_dense_view_ram_t {
    pub index: *mut usearch_dense_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub path: usearch_bytes_view_t,
}

#[repr(C)]
pub struct usearch_dense_view_nvme_t {
    pub index: *mut usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub nvme: usearch_nvme_options_t,
    pub ram: *const usearch_ram_options_t,
}

#[repr(C)]
pub struct usearch_dense_load_nvme_t {
    pub index: *mut usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub nvme: usearch_nvme_options_t,
    pub ram: *const usearch_ram_options_t,
}

#[repr(C)]
pub struct usearch_dense_reserve_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub capacity: usize,
}

#[repr(C)]
pub struct usearch_dense_clear_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
}

pub type usearch_traversal_stats_t = DenseTraversalStats;

#[repr(C)]
pub struct usearch_dense_insert_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub keys: usearch_keys_view_t,
    pub vectors: usearch_rows_view_t,
    pub expansion_factor: usize,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_dense_refine_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
    pub plan: *mut usearch_dense_refine_plan_t,
    pub phase: *mut usearch_phase_progress_t,
}

pub type usearch_dense_refine_plan_t = *mut c_void;
pub type usearch_dense_reshard_plan_t = *mut c_void;
pub type usearch_phase_progress_t = PhaseProgress;

#[repr(C)]
pub struct usearch_dense_refine_plan_finish_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub plan: usearch_dense_refine_plan_t,
    pub report: *mut usearch_refine_report_t,
}

pub type usearch_reachability_t = DenseReachability;
pub type usearch_refine_report_t = RefineReport;

#[repr(C)]
pub struct usearch_refine_options_t {
    pub expansion_factor: usize,
    pub memory_limit: usize,
}

#[repr(C)]
pub struct usearch_dense_health_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub reachability: *mut usearch_reachability_t,
}

#[repr(C)]
pub struct usearch_dense_search_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub queries: usearch_rows_view_t,
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

#[repr(C)]
pub struct usearch_dense_count_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub key: usearch_key_t,
    pub entries: *mut usize,
}

#[repr(C)]
pub struct usearch_dense_get_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub key: usearch_key_t,
    pub vectors: *mut usearch_rows_view_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
    pub outputs_scalar: usearch_scalar_t,
}

#[repr(C)]
pub struct usearch_dense_reshard_t {
    pub sources_count: usize,
    pub sources: *const usearch_dense_index_t,
    pub targets_count: usize,
    pub targets: *const usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub ram: *const usearch_ram_options_t,
    pub vram: *const usearch_vram_options_t,
    pub nvme: *const usearch_nvme_options_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub seed: u64,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
    pub plan: *mut usearch_dense_reshard_plan_t,
    pub phase: *mut usearch_phase_progress_t,
}

#[repr(C)]
pub struct usearch_dense_reshard_plan_finish_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub plan: usearch_dense_reshard_plan_t,
    pub report: *mut usearch_refine_report_t,
}

#[repr(C)]
pub struct usearch_dense_sample_t {
    pub index: usearch_dense_index_t,
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
pub struct usearch_dense_cursor_init_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub starts_count: usize,
    pub starts: *const usearch_key_t,
    pub cursor: *mut usearch_dense_cursor_t,
}

#[repr(C)]
pub struct usearch_dense_cursor_next_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub cursor: usearch_dense_cursor_t,
    pub keys: *mut usearch_keys_view_t,
    pub links: *mut usearch_keys_tape_view_t,
    pub written: *mut usize,
    pub arena: usearch_arena_t,
    pub wanted: usize,
}

#[repr(C)]
pub struct usearch_dense_search_stream_init_t {
    pub stream: *mut usearch_dense_search_stream_t,
    pub index: usearch_dense_index_t,
    pub outcome: *mut usearch_outcome_t,
    pub query: usearch_rows_view_t,
    pub expansion_factor: usize,
}

#[repr(C)]
pub struct usearch_dense_search_stream_next_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub stream: usearch_dense_search_stream_t,
    pub matches: *mut usearch_matches_view_t,
    pub arena: usearch_arena_t,
    pub wanted: usize,
}

#[repr(C)]
pub struct usearch_dense_save_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub progress: usearch_progress_t,
}

#[repr(C)]
pub struct usearch_dense_save_buffer_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub monolith: *mut usearch_bytes_view_t,
    pub arena: usearch_arena_t,
    pub progress: usearch_progress_t,
}

pub type usearch_dense_metadata_t = DenseMetadata;

#[repr(C)]
pub struct usearch_dense_metadata_read_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub metadata: *mut usearch_dense_metadata_t,
}

#[repr(C)]
pub struct usearch_dense_metadata_read_buffer_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub metadata: *mut usearch_dense_metadata_t,
}

pub type usearch_dense_index_description_t = DenseIndexDescription;

#[repr(C)]
pub struct usearch_dense_describe_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub description: *mut usearch_dense_index_description_t,
}

#[repr(C)]
pub struct usearch_dense_multiply_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub left: usearch_rows_view_t,
    pub products: *mut usearch_matches_tape_view_t,
    pub written: *mut usize,
    pub required: *mut usize,
    pub arena: usearch_arena_t,
    pub wanted: usize,
    pub expansion_factor: usize,
    pub traversal: usearch_traversal_t,
    pub predicate: usearch_predicate_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

pub type usearch_dense_softmax_state_t = DenseSoftmaxState;

pub type usearch_value_lookup_t =
    Option<unsafe extern "C" fn(*mut c_void, *const usearch_key_t, usize, usearch_scalar_t, usize, *mut *const c_void)>;

#[repr(C)]
pub struct usearch_dense_attend_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub queries: usearch_rows_view_t,
    pub head_count: usize,
    pub value_lookup: usearch_value_lookup_t,
    pub value_lookup_state: *mut c_void,
    pub carry: *mut usearch_dense_softmax_state_t,
    pub outputs: *mut usearch_rows_view_t,
    pub arena: usearch_arena_t,
    pub outputs_scalar: usearch_scalar_t,
    pub expansion_factor: usize,
    pub score_scale: f32,
    pub traversal: usearch_traversal_t,
    pub predicate: usearch_predicate_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_dense_maxsim_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub queries: usearch_rows_view_t,
    pub query_offsets: *const usize,
    pub rows_per_query: usize,
    pub matches: *mut usearch_matches_tape_view_t,
    pub written: *mut usize,
    pub required: *mut usize,
    pub arena: usearch_arena_t,
    pub wanted: usize,
    pub expansion_factor: usize,
    pub traversal: usearch_traversal_t,
    pub predicate: usearch_predicate_t,
    pub hints: *const usearch_matches_view_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

unsafe extern "C" {
    pub fn usearch_dense_metric_name(metric: usearch_dense_metric_t) -> *const c_char;
    pub fn usearch_dense_metric_from_name(
        name: usearch_bytes_view_t,
        metric: *mut usearch_dense_metric_t,
    ) -> usearch_status_t;

    pub fn usearch_dense_init_ram(arguments: *mut usearch_dense_init_ram_t);
    pub fn usearch_dense_init_vram(arguments: *mut usearch_dense_init_vram_t);
    pub fn usearch_dense_load_ram(arguments: *mut usearch_dense_load_ram_t);
    pub fn usearch_dense_load_vram(arguments: *mut usearch_dense_load_vram_t);
    pub fn usearch_dense_load_ram_buffer(arguments: *mut usearch_dense_load_ram_buffer_t);
    pub fn usearch_dense_load_vram_buffer(arguments: *mut usearch_dense_load_vram_buffer_t);
    pub fn usearch_dense_view_ram(arguments: *mut usearch_dense_view_ram_t);
    pub fn usearch_dense_view_nvme(arguments: *mut usearch_dense_view_nvme_t);
    pub fn usearch_dense_load_nvme(arguments: *mut usearch_dense_load_nvme_t);
    pub fn usearch_dense_free(index: usearch_dense_index_t);

    pub fn usearch_dense_reserve(arguments: *mut usearch_dense_reserve_t);
    pub fn usearch_dense_clear(arguments: *mut usearch_dense_clear_t);
    pub fn usearch_dense_insert(arguments: *mut usearch_dense_insert_t);
    pub fn usearch_dense_refine(arguments: *mut usearch_dense_refine_t);
    pub fn usearch_dense_refine_plan_finish(arguments: *mut usearch_dense_refine_plan_finish_t);
    pub fn usearch_dense_health(arguments: *mut usearch_dense_health_t);
    pub fn usearch_dense_search(arguments: *mut usearch_dense_search_t);
    pub fn usearch_dense_count(arguments: *mut usearch_dense_count_t);
    pub fn usearch_dense_get(arguments: *mut usearch_dense_get_t);
    pub fn usearch_dense_reshard(arguments: *mut usearch_dense_reshard_t);
    pub fn usearch_dense_reshard_plan_finish(arguments: *mut usearch_dense_reshard_plan_finish_t);
    pub fn usearch_dense_sample(arguments: *mut usearch_dense_sample_t);

    pub fn usearch_dense_cursor_init(arguments: *mut usearch_dense_cursor_init_t);
    pub fn usearch_dense_cursor_next(arguments: *mut usearch_dense_cursor_next_t);
    pub fn usearch_dense_cursor_free(cursor: usearch_dense_cursor_t);

    pub fn usearch_dense_search_stream_init(arguments: *mut usearch_dense_search_stream_init_t);
    pub fn usearch_dense_search_stream_next(arguments: *mut usearch_dense_search_stream_next_t);
    pub fn usearch_dense_search_stream_free(stream: usearch_dense_search_stream_t);

    pub fn usearch_dense_save(arguments: *mut usearch_dense_save_t);
    pub fn usearch_dense_save_buffer(arguments: *mut usearch_dense_save_buffer_t);
    pub fn usearch_dense_metadata_read(arguments: *mut usearch_dense_metadata_read_t);
    pub fn usearch_dense_metadata_read_buffer(arguments: *mut usearch_dense_metadata_read_buffer_t);
    pub fn usearch_dense_describe(arguments: *mut usearch_dense_describe_t);

    pub fn usearch_dense_multiply(arguments: *mut usearch_dense_multiply_t);
    pub fn usearch_dense_attend(arguments: *mut usearch_dense_attend_t);
    pub fn usearch_dense_maxsim(arguments: *mut usearch_dense_maxsim_t);
}

// endregion: Raw Declarations

// region: Geometry

/// The dense metric kinds supported out of the box, all derived from a single inner-product
/// kernel with pre-computed magnitudes. Hamming and Jaccard are bit-level over [`Scalar::U1x8`]
/// alone, a kind the other three refuse.
///
/// Arbitrary metrics belong to the C++ baseline tier, which forfeits the SIMD kernel chains and
/// the magnitude cache both.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum DenseMetric {
    /// Sum of elementwise products, unnormalized and scale-sensitive; the only kind the linear-algebra verbs
    /// accept, since a product and a softmax both need logits rather than distances.
    Inner = 1,
    /// Cosine distance in [0, 2], zero for identical directions and blind to magnitude, which is precomputed once
    /// per vector at insertion rather than per comparison.
    #[default]
    Angular = 2,
    /// L2 distance, non-negative and zero only for identical vectors - the scale-sensitive counterpart to angular,
    /// where a doubled vector is far from its original rather than identical to it.
    Euclidean = 3,
    /// Count of differing bits, so its range is the dimensionality and every bit weighs the same.
    Hamming = 4,
    /// Differing bits over union bits in [0, 1], ignoring the zeros both vectors share - the kind for sparse sets,
    /// where Hamming would call two nearly-empty vectors identical.
    Jaccard = 5,
}

impl DenseMetric {
    fn from_raw(value: usearch_dense_metric_t) -> Option<Self> {
        let metric = match value {
            1 => DenseMetric::Inner,
            2 => DenseMetric::Angular,
            3 => DenseMetric::Euclidean,
            4 => DenseMetric::Hamming,
            5 => DenseMetric::Jaccard,
            _ => return None,
        };
        Some(metric)
    }
}

impl fmt::Display for DenseMetric {
    /// Writes the metric's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_dense_metric_name, *self as c_int))
    }
}

impl FromStr for DenseMetric {
    type Err = Error;

    /// Reads a metric back from the name `Display` writes, refusing any other.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_dense_metric_from_name, DenseMetric::from_raw)
    }
}

/// Construction options of a dense index, whose stored vectors round up to the SIMD step quantum so
/// every kernel runs tail-free.
///
/// [`DenseIndex::init_ram`] and [`DenseIndex::init_vram`] both take it.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct DenseOptions {
    /// The number of indexed dimensions, which init freezes; zero is refused with
    /// `Status::ConfigMismatch`.
    pub dimensions: usize,
    /// The metric kind used for distance calculations.
    pub metric: DenseMetric,
    /// The scalar type vectors are stored and quantized into.
    pub quantization: Scalar,
    /// The connections-per-node bound of the graph; zero derives.
    pub connectivity: usize,
    /// Upper bound on entries sharing one key; zero is unbounded.
    pub values_per_key: usize,
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; `None` means no ceiling.
    pub cpu_class_ceiling: Option<CPUClass>,
    /// The newest class dispatch may pick, capped below the best class every named device loads; a
    /// ceiling no named device loads is refused as unsupported; `None` means no ceiling.
    pub gpu_class_ceiling: Option<GPUClass>,
}

impl Default for DenseOptions {
    fn default() -> Self {
        DenseOptions {
            dimensions: 0,
            metric: DenseMetric::Angular,
            quantization: Scalar::BF16,
            connectivity: 0,
            values_per_key: 0,
            cpu_class_ceiling: None,
            gpu_class_ceiling: None,
        }
    }
}

/// The geometry and metric a checkpoint froze, readable without loading it.
///
/// `values_per_key` and the placement are the reader's to choose when loading, so a checkpoint
/// fixes neither.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct DenseMetadata {
    /// The metric kind the checkpoint scores with.
    pub metric: DenseMetric,
    /// The scalar type its vectors are stored and quantized into.
    pub quantization: Scalar,
    /// Scalars per stored vector.
    pub dimensions: usize,
    /// Connections per node the graph was built with.
    pub connectivity: usize,
}

impl Default for DenseMetadata {
    fn default() -> Self {
        DenseMetadata { metric: DenseMetric::default(), quantization: Scalar::BF16, dimensions: 0, connectivity: 0 }
    }
}

/// A live index's frozen metadata and current counters, each read once, so under concurrent
/// insertions they may come from slightly different moments.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct DenseIndexDescription {
    /// The frozen metric, quantization, dimensions and connectivity, as a checkpoint reports them.
    pub metadata: DenseMetadata,
    /// Stored entries.
    pub size: usize,
    /// Entries hosted without growing again.
    pub capacity: usize,
    /// Bytes the handle holds; traversal scratch is the executor's.
    pub memory_usage: usize,
    /// The CPU class dispatch picked, spelled by [`DenseIndexDescription::cpu_class`].
    cpu_class: usearch_cpu_class_t,
    /// The GPU class dispatch picked, spelled by [`DenseIndexDescription::gpu_class`].
    gpu_class: usearch_gpu_class_t,
    /// Live shards.
    pub shards_count: usize,
    /// How the last load read the parts, spelled by [`DenseIndexDescription::load_path`].
    load_path: usearch_load_path_t,
}

impl DenseIndexDescription {
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

// endregion: Geometry

// region: Configs

/// Traversal telemetry, fully overwritten by the call it is passed to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct DenseTraversalStats {
    /// Graph members visited by the call.
    pub visited_members: usize,
    /// Distances computed by the call, diversity re-selections included.
    pub computed_distances: usize,
    /// Starvation restarts - a graph-connectivity health signal, not a cost counter.
    pub restarts: usize,
    /// Hint keys no shard held an entry under; zero unless a search was handed hints.
    pub missing_hints: usize,
}

/// One reachability audit: the population it judged, and how much of it a search reaches.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct DenseReachability {
    /// Entries published when the audit started; compare `reachable` against this, never a later size.
    pub audited: usize,
    /// Of those, the entries a search reaches; the rest no search finds until a refinement re-links them.
    pub reachable: usize,
    /// Depths the walk from the root took to converge; the deepest shard's on a sharded handle.
    pub depth_from_root: usize,
    /// Depths the walk back toward the root took, where a health check walks both ways; the deepest
    /// shard's on a sharded handle.
    pub depth_to_root: usize,
    /// Entries no reached entry links to.
    pub unlinked: usize,
    /// Entries linked to by fewer entries than they link to.
    pub underlinked: usize,
    /// Entries deeper than the depth holding the most entries.
    pub beyond_most_populated_depth: usize,
    /// Islands no entry outside them links into.
    pub unreached_islands: usize,
    /// Islands with no link out.
    pub stranded_islands: usize,
    /// Components when direction is ignored.
    pub components: usize,
}

/// What a round concluded, so no caller picks a change rate.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum RefineVerdict {
    /// No round concluded: a report no round filled yet.
    #[default]
    None = 0,
    /// The round changed some entry's link set, so another round can help.
    Improved = 1,
    /// The round changed no entry's link set.
    Converged = 2,
    /// The round lowered nothing while something stands, as full entries leave no slot.
    Stalled = 3,
    /// The caller stopped the round; the phase names where.
    Stopped = 4,
}

impl RefineVerdict {
    fn from_raw(value: c_int) -> Option<Self> {
        match value {
            0 => Some(RefineVerdict::None),
            1 => Some(RefineVerdict::Improved),
            2 => Some(RefineVerdict::Converged),
            3 => Some(RefineVerdict::Stalled),
            4 => Some(RefineVerdict::Stopped),
            _ => None,
        }
    }
}

impl fmt::Display for RefineVerdict {
    /// Writes the verdict's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_refine_verdict_name, *self as c_int))
    }
}

impl FromStr for RefineVerdict {
    type Err = Error;

    /// Reads a verdict back from the name `Display` writes, refusing any other.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_refine_verdict_from_name, RefineVerdict::from_raw)
    }
}

/// What one refine-family call did to the graph: [`DenseVerbs::refine`], [`DenseVerbs::reshard`]
/// and their plans' finishes land it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct RefineReport {
    /// Entries whose neighbors were selected again, the islands of every heal round included.
    pub reselected: usize,
    /// Of those, the entries whose neighbor set gained a member it lacked; never above
    /// `reselected`.
    pub changed: usize,
    /// Entries the first audit found not reached both ways, counted once.
    pub islands_found: usize,
    /// Links written between an island and an entry the audit reached both ways.
    pub islands_reconnected: usize,
    /// Neighborhoods the round re-stitched.
    pub neighborhoods: usize,
    /// Entries of the widest neighborhood.
    pub widest_neighborhood: usize,
    /// Links the round added.
    pub added_links: usize,
    /// Links the round dropped.
    pub dropped_links: usize,
    /// What the round concluded.
    pub verdict: RefineVerdict,
    /// The health check opening the round.
    pub opening: DenseReachability,
    /// The health check closing the heal, as a [`DenseVerbs::health`] right after would take it.
    pub reachability: DenseReachability,
}

/// The online-softmax carry of one query row, which folds one attention call into another over
/// the same queries.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct DenseSoftmaxState {
    /// The largest logit seen so far - the recurrence's `m`.
    pub maximum_score: f32,
    /// The sum of exponentials taken against that maximum - the recurrence's `l`.
    pub partition: f32,
}

/// An attention's value lookup, run on the executor's threads together: it resolves one key to
/// its value row, stored in the index's own quantization and borrowed from storage outliving the
/// call, and `None` refuses the call; a panic inside it aborts the process.
pub type ValueLookup<'lookup, Value> = dyn Fn(Key) -> Option<&'lookup [Value]> + Sync + 'lookup;

/// A checkpoint path and how it is copied into a new index in host memory; see
/// [`DenseIndex::load_ram`].
pub struct DenseLoadRamConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The executor the read fans out over; `None` reads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// NUMA node IDs to place the parts on, part `p` loading whole into shard `p` modulo their
    /// count; empty derives every node the process may use.
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

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for DenseLoadRamConfig<'config> {
    fn from(path: &'config Location) -> Self {
        DenseLoadRamConfig {
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

/// A checkpoint path and how it is uploaded into a new index in device memory; see
/// [`DenseIndex::load_vram`].
pub struct DenseLoadVramConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The executor the upload runs on; `None` uploads on the default stream of each device.
    pub executor: Option<&'config Executor>,
    /// CUDA device ordinals to place the parts on, part `p` loading whole into device `p` modulo
    /// their count; empty derives one free device.
    pub devices: &'config [i32],
    /// The newest class dispatch may pick, capped below the best class every named device loads; a
    /// ceiling no named device loads is refused as unsupported; `None` means no ceiling.
    pub gpu_class_ceiling: Option<GPUClass>,
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    pub values_per_key: usize,
    /// Reports the shards placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for DenseLoadVramConfig<'config> {
    fn from(path: &'config Location) -> Self {
        DenseLoadVramConfig {
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
/// [`DenseIndex::load_ram_buffer`].
pub struct DenseLoadRamBufferConfig<'config> {
    /// The monolith's bytes.
    pub buffer: &'config [u8],
    /// The executor the copy fans out over; `None` copies on the calling thread.
    pub executor: Option<&'config Executor>,
    /// NUMA node IDs to place shards on, the monolith loading whole into the first; empty derives
    /// every node the process may use.
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

impl<'config> From<&'config [u8]> for DenseLoadRamBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        DenseLoadRamBufferConfig {
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

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for DenseLoadRamBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self {
        DenseLoadRamBufferConfig::from(&**bytes)
    }
}

/// A serialized monolith and how it is uploaded into a new index in device memory; see
/// [`DenseIndex::load_vram_buffer`].
pub struct DenseLoadVramBufferConfig<'config> {
    /// The monolith's bytes, in host or device memory.
    pub buffer: &'config [u8],
    /// The executor the upload runs on; `None` uploads on the default stream of each device.
    pub executor: Option<&'config Executor>,
    /// CUDA device ordinals to place shards on, the monolith loading whole into the first; empty
    /// derives one free device.
    pub devices: &'config [i32],
    /// The newest class dispatch may pick, capped below the best class every named device loads; a
    /// ceiling no named device loads is refused as unsupported; `None` means no ceiling.
    pub gpu_class_ceiling: Option<GPUClass>,
    /// Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    pub values_per_key: usize,
    /// Reports the bytes placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'config> From<&'config [u8]> for DenseLoadVramBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        DenseLoadVramBufferConfig {
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

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for DenseLoadVramBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self {
        DenseLoadVramBufferConfig::from(&**bytes)
    }
}

/// A checkpoint path and the drives serving it; see [`DenseIndex::view_nvme`] and
/// [`DenseIndex::load_nvme`].
pub struct DenseViewNvmeConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The NVMe executor whose rings read the checkpoint's headers; `None` reads on the calling
    /// thread.
    pub executor: Option<&'config Executor>,
    /// Devices or directories to serve the parts from; empty serves each where its manifest points,
    /// and [`DenseIndex::load_nvme`] admits no other.
    pub paths: &'config [&'config Path],
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; `None` means no ceiling.
    pub cpu_class_ceiling: Option<CPUClass>,
    /// Memory domains of a key table held in RAM, which `count`, `get`, `sample`, MaxSim and hinted
    /// searches read; `None` holds none.
    pub key_table_domains: Option<&'config [i32]>,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for DenseViewNvmeConfig<'config> {
    fn from(path: &'config Location) -> Self {
        DenseViewNvmeConfig {
            path: path.as_ref(),
            executor: None,
            paths: &[],
            cpu_class_ceiling: None,
            key_table_domains: None,
        }
    }
}

/// The rows of an insertion and the knobs of its walk; see [`DenseVerbs::insert`].
pub struct DenseInsertConfig<'config, Rows> {
    /// One row per key, borrowed or owned.
    pub vectors: Rows,
    /// The breadth of the candidates the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Reports the entries inserted; every entry is claimed before the first report, so a `Break` is
    /// ignored.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl<Rows: RowsView> From<Rows> for DenseInsertConfig<'_, Rows> {
    fn from(vectors: Rows) -> Self {
        DenseInsertConfig { vectors, expansion_factor: 0, progress: None, progress_steps: 0, stats: None }
    }
}

/// The knobs of [`DenseVerbs::refine`].
///
/// A blocking call borrows its plan; a deferred one owns it, and [`Deferred::wait_with_inputs`]
/// hands it back.
pub struct DenseRefineConfig<'config, Plan = &'config mut DenseRefinePlan> {
    /// The breadth of any walk beside the round; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Bytes a loaded NVMe handle holds for the one part it rewrites; zero derives what rounds the
    /// part in RAM, and RAM and VRAM handles never read it.
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

impl Default for DenseRefineConfig<'_> {
    fn default() -> Self {
        DenseRefineConfig::with_plan(None)
    }
}

impl From<DenseRefinePlan> for DenseRefineConfig<'_, DenseRefinePlan> {
    /// A deferred call owning `plan`, with every other knob derived.
    fn from(plan: DenseRefinePlan) -> Self {
        DenseRefineConfig::with_plan(Some(plan))
    }
}

impl<Plan> DenseRefineConfig<'_, Plan> {
    fn with_plan(plan: Option<Plan>) -> Self {
        DenseRefineConfig {
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

/// A refine in flight between calls, which [`DenseVerbs::refine_plan_finish`] closes.
///
/// While it lives the index admits no insertion and no save, and every call naming it runs on the
/// index and executor it was opened with.
#[derive(Debug)]
pub struct DenseRefinePlan {
    handle: usearch_dense_refine_plan_t,
}

/// A reshard in flight between calls, which [`DenseVerbs::reshard_plan_finish`] closes.
///
/// While it lives every handle it names admits no insertion and no save, and every call naming it
/// runs on the executor it was opened with.
#[derive(Debug)]
pub struct DenseReshardPlan {
    handle: usearch_dense_reshard_plan_t,
}

impl Default for DenseRefinePlan {
    /// A fresh plan, which the first call holding it opens.
    fn default() -> Self {
        DenseRefinePlan { handle: ptr::null_mut() }
    }
}

impl Default for DenseReshardPlan {
    /// A fresh plan, which the first call holding it opens.
    fn default() -> Self {
        DenseReshardPlan { handle: ptr::null_mut() }
    }
}

// Safety: a plan is driven by one call at a time, which the `&mut` holding it enforces.
unsafe impl Send for DenseRefinePlan {}
unsafe impl Send for DenseReshardPlan {}

/// The matches a search keeps and the knobs of its walk; see [`DenseVerbs::search`].
pub struct DenseSearchConfig<'config> {
    /// Matches kept per query, the "k" in "kANN".
    pub wanted: usize,
    /// The breadth of the candidates the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Walks the graph by default; [`Traversal::Scan`] scores every entry exactly, on RAM and NVMe
    /// handles, and a VRAM handle refuses it with [`Status::Unsupported`].
    pub traversal: Traversal,
    /// Admits only the keys it approves; the rest still route the walk, so recall does not
    /// collapse at low selectivity. A VRAM handle refuses one with [`Status::Unsupported`].
    pub predicate: Option<&'config Predicate<'config>>,
    /// Reports the queries resolved and stops wherever it says to; a stopped search reports
    /// [`Status::Cancelled`] and leaves no matches.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl From<usize> for DenseSearchConfig<'_> {
    fn from(wanted: usize) -> Self {
        DenseSearchConfig {
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

/// The documents a late-interaction search keeps and the knobs of its walk; see
/// [`DenseVerbs::maxsim`].
pub struct DenseMaxsimConfig<'config> {
    /// Documents kept per query, counted over keys rather than entries; zero keeps every key the
    /// query scored.
    pub wanted: usize,
    /// The breadth of the candidates the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Reports the queries resolved and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl From<usize> for DenseMaxsimConfig<'_> {
    fn from(wanted: usize) -> Self {
        DenseMaxsimConfig { wanted, expansion_factor: 0, progress: None, progress_steps: 0, stats: None }
    }
}

/// The pairs a sparse product keeps and the knobs of its walk; see [`DenseVerbs::multiply`].
pub struct DenseMultiplyConfig<'config> {
    /// Keeps the first `wanted` pairs each row's walk scored; zero keeps every pair.
    pub wanted: usize,
    /// The breadth of the candidates, which widens the emitted pattern; zero derives it.
    pub expansion_factor: usize,
    /// Reports the rows resolved and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl From<usize> for DenseMultiplyConfig<'_> {
    fn from(wanted: usize) -> Self {
        DenseMultiplyConfig { wanted, expansion_factor: 0, progress: None, progress_steps: 0, stats: None }
    }
}

/// The heads, carry and knobs of an attention's walk; see [`DenseVerbs::attend`].
pub struct DenseAttendConfig<'config> {
    /// The query heads sharing the index's key-value head, and with it one walk per token: row
    /// `t · head_count + h` is head `h` of token `t`, so it must divide the rows; zero means one.
    pub head_count: usize,
    /// Where the online-softmax state lands, one entry per query row, which folds this call into
    /// another over the same queries by partial-softmax rescaling.
    pub carry: Option<&'config mut [DenseSoftmaxState]>,
    /// The breadth of the candidates; zero derives it. Wider candidates walk further, folding more of the
    /// tail.
    pub expansion_factor: usize,
    /// The logit scale, conventionally the inverse square root of the head dimensionality; zero
    /// derives it from the index's frozen geometry.
    pub score_scale: f32,
    /// Reports the queries resolved and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl From<usize> for DenseAttendConfig<'_> {
    fn from(head_count: usize) -> Self {
        DenseAttendConfig {
            head_count,
            carry: None,
            expansion_factor: 0,
            score_scale: 0.0,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// The handles a reshard moves entries between, the placement a recut in place packs toward, and
/// the knobs of the closing rounds; see [`DenseVerbs::reshard`].
///
/// A blocking call borrows its plan; a deferred one owns it, and [`Deferred::wait_with_inputs`]
/// hands it back.
pub struct DenseReshardConfig<'config, Plan = &'config mut DenseReshardPlan> {
    /// The handles whose entries move, each named once.
    pub sources: &'config [&'config DenseIndex],
    /// The handles the entries move into, each a source itself or empty.
    pub targets: &'config [&'config DenseIndex],
    /// NUMA node IDs of a RAM placement recut in place; `None` keeps the placement.
    pub memory_domains: Option<&'config [i32]>,
    /// CUDA device ordinals of a VRAM placement recut in place; `None` keeps it.
    pub devices: Option<&'config [i32]>,
    /// Drives of an NVMe placement recut in place; `None` keeps it.
    pub paths: Option<&'config [&'config Path]>,
    /// Entries per shard of the RAM or VRAM placement; zero keeps the handle's own bound.
    pub shard_capacity: usize,
    /// The breadth of any walk beside the closing rounds; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Bytes a loaded NVMe target holds for the one part it rewrites; zero derives what rounds the
    /// part in RAM, and RAM and VRAM targets never read it.
    pub memory_limit: usize,
    /// Picks the part roots of a source cut among several targets: equal seeds over an unchanged
    /// source cut equally.
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

impl Default for DenseReshardConfig<'_> {
    fn default() -> Self {
        DenseReshardConfig::with_plan(None)
    }
}

impl From<DenseReshardPlan> for DenseReshardConfig<'_, DenseReshardPlan> {
    /// A deferred call owning `plan`, naming no handle, since a live plan already holds them.
    fn from(plan: DenseReshardPlan) -> Self {
        DenseReshardConfig::with_plan(Some(plan))
    }
}

impl<Plan> DenseReshardConfig<'_, Plan> {
    fn with_plan(plan: Option<Plan>) -> Self {
        DenseReshardConfig {
            sources: &[],
            targets: &[],
            memory_domains: None,
            devices: None,
            paths: None,
            shard_capacity: 0,
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

/// The view a sample grows and how it draws; see [`DenseVerbs::sample`].
#[derive(Default)]
pub struct DenseSampleConfig<'config> {
    /// The keys already shown: never drawn again, and the paths reach toward them.
    pub shown: &'config [Key],
    /// Where each of `shown` sits, the points later draws are placed against.
    pub shown_coordinates: &'config [[f32; 3]],
    /// The keys to look around; empty looks around the whole index.
    pub focus: &'config [Key],
    /// The draw: equal seeds, shown keys and focus over an unchanged index draw equally.
    pub seed: u64,
    /// The breadth of the walks behind the draws; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Reports the draws written against `wanted` and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

/// The path a checkpoint is written to and the handler counting its shards; see [`DenseVerbs::save`].
pub struct DenseSaveConfig<'config, Location> {
    /// A `.safegraphs` path for one monolith, a `.safegraphs.json` path for a manifest.
    pub path: Location,
    /// Reports the shards written and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<Location: AsRef<Path>> From<Location> for DenseSaveConfig<'_, Location> {
    fn from(path: Location) -> Self {
        DenseSaveConfig { path, progress: None, progress_steps: 0 }
    }
}

/// The arena a monolith is written into and the handler counting its bytes; see
/// [`DenseVerbs::save_buffer`].
pub struct DenseSaveBufferConfig<'config, 'arena> {
    /// The arena the monolith lands in, which the result borrows.
    pub arena: &'arena mut Arena,
    /// Reports the bytes written and stops wherever it says to; a stopped save leaves no monolith.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

impl<'arena> From<&'arena mut Arena> for DenseSaveBufferConfig<'_, 'arena> {
    fn from(arena: &'arena mut Arena) -> Self {
        DenseSaveBufferConfig { arena, progress: None, progress_steps: 0 }
    }
}

/// The query a stream walks and the breadth of its walk; see [`DenseVerbs::search_stream`].
pub struct DenseSearchStreamInitConfig<Query> {
    /// Exactly one query row.
    pub query: Query,
    /// The breadth of the candidates the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
}

impl<Query: RowsView> From<Query> for DenseSearchStreamInitConfig<Query> {
    fn from(query: Query) -> Self {
        DenseSearchStreamInitConfig { query, expansion_factor: 0 }
    }
}

impl<Rows> Accepts<DenseInsertConfig<'static, Rows>> for Deferring where Deferring: Accepts<Rows> {}
impl<Plan: 'static> Accepts<DenseRefineConfig<'static, Plan>> for Deferring {}
impl Accepts<DenseSearchConfig<'static>> for Deferring {}
impl Accepts<DenseMaxsimConfig<'static>> for Deferring {}
impl Accepts<DenseMultiplyConfig<'static>> for Deferring {}
impl Accepts<DenseAttendConfig<'static>> for Deferring {}
impl<Value> Accepts<&'static ValueLookup<'static, Value>> for Deferring {}
impl<Plan: 'static> Accepts<DenseReshardConfig<'static, Plan>> for Deferring {}
impl Accepts<DenseSampleConfig<'static>> for Deferring {}
impl<Location> Accepts<DenseSaveConfig<'static, Location>> for Deferring where Deferring: Accepts<Location> {}
impl Accepts<DenseSaveBufferConfig<'static, '_>> for Deferring {}

// endregion: Configs

// region: Results

/// The rows [`DenseVerbs::get`] or [`DenseVerbs::attend`] exported into the arena it borrows.
///
/// Rust-only.
pub struct DenseVectors<'arena, Value> {
    arena: &'arena Arena,
    vectors: usearch_rows_view_t,
    count: usize,
    value: PhantomData<Value>,
}

impl<'arena, Value: Element> DenseVectors<'arena, Value> {
    fn new(arena: &'arena Arena) -> Self {
        DenseVectors { arena, vectors: usearch_rows_view_t::default(), count: 0, value: PhantomData }
    }

    /// One row per entry under the key in insertion order, or per query, at the pitch the library
    /// reported; an absent key leaves a null base, which a zero-row view never reads.
    pub fn view(&self) -> Result<MatrixView<'arena, Value>> {
        let data = NonNull::new(self.vectors.data.cast::<Value>().cast_mut()).unwrap_or(NonNull::dangling());
        let shape = [self.count, self.vectors.columns];
        let strides = [self.vectors.stride as isize, core::mem::size_of::<Value>() as isize];
        unsafe { MatrixView::from_raw_parts(data.as_ptr(), &shape, &strides) }
            .map_err(|_| fail(Status::Internal, "the library returned an untraversable slab"))
    }

    /// Exports the rows as one Arrow `FixedSizeList` column of their scalar in place.
    ///
    /// The export borrows the arena, so no verb reaches it while the export lives:
    ///
    /// ```compile_fail
    /// use usearch::*;
    /// # fn main() -> Result<()> {
    /// let options = DenseOptions { dimensions: 2, ..DenseOptions::default() };
    /// let index = DenseIndex::init_ram(&options, &[])?;
    /// let mut arena = Arena::new()?;
    /// let vectors = index.get::<f32>(Key::from(1u64), &mut arena)?;
    /// let export = vectors.to_arrow()?;
    /// index.get::<f32>(Key::from(1u64), &mut arena)?;
    /// drop(export);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_arrow(&self) -> Result<ArrowExport<'arena>> {
        let mut schema = ArrowSchema::default();
        let mut array = ArrowDeviceArray::default();
        let status =
            unsafe { usearch_rows_view_to_arrow(self.vectors, self.count, self.arena.handle, &mut schema, &mut array) };
        arrow_checked(status)?;
        Ok(ArrowExport { arena: PhantomData, schema, array })
    }
}

/// The ragged run [`DenseVerbs::multiply`] or [`DenseVerbs::maxsim`] exported into the arena it
/// borrows: per left row, every pair its walk scored, in traversal order, or per query, its kept
/// keys, largest score first.
pub struct DenseProducts<'arena> {
    arena: &'arena Arena,
    products: usearch_matches_tape_view_t,
    count: usize,
    required: usize,
}

impl<'arena> DenseProducts<'arena> {
    fn new(arena: &'arena Arena) -> Self {
        DenseProducts { arena, products: usearch_matches_tape_view_t::default(), count: 0, required: 0 }
    }

    /// Left rows the call multiplied, or queries it scored.
    pub fn queries(&self) -> usize {
        self.count
    }

    /// Pairs the walks scored across every left row, or keys scored across every query, before
    /// the wanted bound cut any.
    pub fn required(&self) -> usize {
        self.required
    }

    /// The pairs of one left row, a key naming each indexed entry beside its product, or the keys
    /// of one query beside their summed maxima, refused with [`Status::InvalidArgument`] when
    /// `query` is not below [`DenseProducts::queries`].
    pub fn query(&self, query: usize) -> Result<QueryMatches<'arena>> {
        match query < self.count {
            // Safety: the arena holds a finished call's output.
            true => Ok(unsafe { self.products.query(query) }),
            false => Err(fail(Status::InvalidArgument, "the query is not below the call's count")),
        }
    }

    /// Every query's pairs or keys, in query order.
    pub fn iter(&self) -> impl Iterator<Item = QueryMatches<'arena>> {
        let products = self.products;
        // Safety: the arena holds a finished call's output.
        (0..self.count).map(move |query| unsafe { products.query(query) })
    }

    /// Exports every left row's pairs, or every query's keys, as one Arrow `LargeList` of
    /// `Struct<key, distance>` in place.
    ///
    /// The export borrows the arena, so no verb reaches it while the export lives:
    ///
    /// ```compile_fail
    /// use usearch::*;
    /// # fn main() -> Result<()> {
    /// let options = DenseOptions { dimensions: 2, metric: DenseMetric::Inner, ..DenseOptions::default() };
    /// let index = DenseIndex::init_ram(&options, &[])?;
    /// let slab = [1.0f32, 0.0];
    /// let mut arena = Arena::new()?;
    /// let products = index.multiply(&rows(&slab, 2)?, 1, &mut arena)?;
    /// let export = products.to_arrow()?;
    /// index.multiply(&rows(&slab, 2)?, 1, &mut arena)?;
    /// drop(export);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_arrow(&self) -> Result<ArrowExport<'arena>> {
        let mut schema = ArrowSchema::default();
        let mut array = ArrowDeviceArray::default();
        let status = unsafe {
            usearch_matches_tape_view_to_arrow(self.products, self.count, self.arena.handle, &mut schema, &mut array)
        };
        arrow_checked(status)?;
        Ok(ArrowExport { arena: PhantomData, schema, array })
    }
}

/// The bytes [`DenseVerbs::save_buffer`] exported into the arena it borrows.
///
/// Rust-only.
pub struct Monolith<'arena> {
    pub(crate) arena: PhantomData<&'arena Arena>,
    pub(crate) monolith: usearch_bytes_view_t,
}

impl<'arena> Monolith<'arena> {
    /// The monolith's bytes.
    pub fn view(&self) -> &'arena [u8] {
        // Safety: the arena holds a finished call's output.
        unsafe { arena_slice(self.monolith.data, self.monolith.length) }
    }
}

/// The draws [`DenseVerbs::sample`] exported into the arena it borrows.
///
/// Rust-only.
pub struct Sample<'arena> {
    pub(crate) arena: &'arena Arena,
    pub(crate) keys: usearch_keys_view_t,
    pub(crate) entry_ordinals: *const usize,
    pub(crate) parents: usearch_matches_view_t,
    pub(crate) coordinates: usearch_rows_view_t,
    pub(crate) written: usize,
}

impl<'arena> Sample<'arena> {
    /// Draws the call wrote; fewer than wanted only once the region is exhausted.
    pub fn written(&self) -> usize {
        self.written
    }

    /// The keys drawn.
    pub fn keys(&self) -> &'arena [Key] {
        // Safety: the arena holds a finished call's packed output.
        unsafe { arena_slice(self.keys.data, self.written) }
    }

    /// Which of its key's entries each draw is, in the order [`DenseVerbs::get`] writes them.
    pub fn entry_ordinals(&self) -> &'arena [usize] {
        // Safety: the arena holds a finished call's packed output.
        unsafe { arena_slice(self.entry_ordinals, self.written) }
    }

    /// The one shown or earlier draw each draw hangs from and its distance, one query per draw,
    /// a draw with none holding no match.
    pub fn parents(&self) -> Matches<'arena> {
        Matches { arena: self.arena, view: self.parents, queries: self.written }
    }

    /// Where each draw is placed, beside the shown keys.
    pub fn coordinates(&self) -> &'arena [[f32; 3]] {
        // Safety: the arena holds a finished call's packed output of three `f32` per row.
        unsafe { arena_slice(self.coordinates.data.cast::<[f32; 3]>(), self.written) }
    }
}

/// One cursor pull exported into the arena it borrows: the keys, and beside them the keys each
/// entry links to when the pull asked; the vectors themselves come from [`DenseVerbs::get`].
///
/// Rust-only.
pub struct DenseBatch<'arena> {
    pub(crate) arena: PhantomData<&'arena Arena>,
    pub(crate) keys: usearch_keys_view_t,
    pub(crate) links: Option<usearch_keys_tape_view_t>,
    pub(crate) written: usize,
}

impl<'arena> DenseBatch<'arena> {
    /// Entries the pull wrote; zero means the walk is exhausted.
    pub fn written(&self) -> usize {
        self.written
    }

    /// The keys of the entries.
    pub fn keys(&self) -> &'arena [Key] {
        // Safety: the arena holds a finished call's packed output.
        unsafe { arena_slice(self.keys.data, self.written) }
    }

    /// The keys entry `entry` links to, refused with [`Status::InvalidArgument`] when the pull did
    /// not ask for links or `entry` is not below [`DenseBatch::written`].
    pub fn links(&self, entry: usize) -> Result<&'arena [Key]> {
        match (&self.links, entry < self.written) {
            // Safety: the arena holds a finished call's output.
            (Some(links), true) => Ok(unsafe { links.entry(entry) }),
            _ => Err(fail(Status::InvalidArgument, "the pull exported no links, or the entry is not below its count")),
        }
    }
}

// Safety: every view points into the arena borrowed beside it, which is `Sync`.
unsafe impl<Value: Send> Send for DenseVectors<'_, Value> {}
unsafe impl Send for DenseProducts<'_> {}
unsafe impl Send for Monolith<'_> {}
unsafe impl Send for Sample<'_> {}
unsafe impl Send for DenseBatch<'_> {}

// endregion: Results

// region: Dense Index

/// Vector Search Index over equidimensional dense representations, whose verbs come from
/// [`DenseVerbs`]. [`DenseIndex::clear`] alone takes `&mut self`, since no other call may overlap it.
///
/// ```no_run
/// use usearch::*;
///
/// let options = DenseOptions {
///     dimensions: 4,
///     metric: DenseMetric::Angular,
///     quantization: Scalar::I8,
///     ..DenseOptions::default()
/// };
/// let index = DenseIndex::init_ram(&options, &[])?;
/// index.reserve(1_000)?;
///
/// let keys = [Key::from(0u64), Key::from(1u64)];
/// let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
/// index.insert(&keys, &rows(&slab, 4)?)?;
///
/// let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
/// let executor = Executor::for_ram(&[], None, 0)?;
/// let mut arena = Arena::new_in(&executor)?;
/// let matches = index.on(&executor).search(&rows(&query, 4)?, 10, &mut arena)?;
/// for query in matches.iter() {
///     println!("{} matches", query.keys.len());
/// }
/// # Ok::<(), usearch::Error>(())
/// ```
///
/// Dropping it waits for every deferred call naming the index, then frees it.
#[derive(Debug)]
pub struct DenseIndex {
    handle: usearch_dense_index_t,
}

// Safety: the library documents a handle as shareable across threads - searches never write to
// the graph, and insertions synchronize through per-node locks and publish in order.
unsafe impl Send for DenseIndex {}
unsafe impl Sync for DenseIndex {}

impl Drop for DenseIndex {
    fn drop(&mut self) {
        unsafe { usearch_dense_free(self.handle) }
    }
}

impl DenseIndex {
    /// An index whose handle a constructor is about to fill.
    fn null() -> Self {
        DenseIndex { handle: ptr::null_mut() }
    }

    /// The receiver whose verbs block on `executor`, contended while another call holds it.
    ///
    /// Rust-only.
    pub fn on<'index>(&'index self, executor: &'index Executor) -> On<'index, DenseIndex> {
        On { index: self, executor, completion: PhantomData }
    }

    /// The receiver whose verbs submit their calls to `executor`, answering a
    /// [`Deferred`](crate::Deferred) from the same verbs given owned arguments.
    ///
    /// The library reads a deferred call's inputs after the verb returns, so lending a borrowed
    /// query does not compile:
    ///
    /// ```compile_fail,E0277
    /// use usearch::*;
    ///
    /// let index = DenseIndex::init_ram(&DenseOptions { dimensions: 4, ..DenseOptions::default() }, &[])?;
    /// let executor = Executor::for_ram(&[], None, 0)?;
    /// let mut arena = Arena::new_in(&executor)?;
    /// let query = [0.0f32; 4];
    /// let call = index.deferred(&executor).search(&rows(&query, 4)?, 10, &mut arena)?;
    /// # Ok::<(), usearch::Error>(())
    /// ```
    ///
    /// The arena is borrowed as a blocking call borrows it, and so is the waited result:
    ///
    /// ```no_run
    /// use usearch::*;
    ///
    /// let index = DenseIndex::init_ram(&DenseOptions { dimensions: 4, ..DenseOptions::default() }, &[])?;
    /// let executor = Executor::for_ram(&[], None, 0)?;
    /// let mut arena = Arena::new_in(&executor)?;
    /// let call = index.deferred(&executor).get::<f32>(Key::from(1u64), &mut arena)?;
    /// let vectors = call.wait()?;
    /// # Ok::<(), usearch::Error>(())
    /// ```
    ///
    /// Rust-only.
    pub fn deferred<'index>(&'index self, executor: &'index Executor) -> On<'index, DenseIndex, Deferring> {
        On { index: self, executor, completion: PhantomData }
    }

    /// Erases all entries on the calling thread, keeping the reserved capacity and frozen geometry;
    /// `&mut self` keeps every other call on the handle from overlapping it.
    pub fn clear(&mut self) -> Result<()> {
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_clear(&mut usearch_dense_clear_t {
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
    pub fn init_ram(options: &DenseOptions, memory_domains: &[i32]) -> Result<Self> {
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_init_ram(&mut usearch_dense_init_ram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                metric: options.metric as usearch_dense_metric_t,
                quantization: options.quantization as usearch_scalar_t,
                dimensions: options.dimensions,
                connectivity: options.connectivity,
                values_per_key: options.values_per_key,
                ram: ram_options(memory_domains, options.cpu_class_ceiling),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Creates an empty, mutable index in device memory, sharded across `devices` and never
    /// replicated; empty derives one free device.
    pub fn init_vram(options: &DenseOptions, devices: &[i32]) -> Result<Self> {
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_init_vram(&mut usearch_dense_init_vram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                metric: options.metric as usearch_dense_metric_t,
                quantization: options.quantization as usearch_scalar_t,
                dimensions: options.dimensions,
                connectivity: options.connectivity,
                values_per_key: options.values_per_key,
                vram: vram_options(devices, options.gpu_class_ceiling),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a checkpoint into a new, mutable index in host memory; a bare path reads on the
    /// calling thread.
    pub fn load_ram<'config>(config: impl Into<DenseLoadRamConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_ram(&mut usearch_dense_load_ram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                ram: ram_options(config.memory_domains, config.cpu_class_ceiling),
                connectivity: 0,
                values_per_key: config.values_per_key,
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a checkpoint into a new, mutable index in device memory, placing its parts as
    /// [`DenseIndex::load_ram`] does.
    pub fn load_vram<'config>(config: impl Into<DenseLoadVramConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_vram(&mut usearch_dense_load_vram_t {
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

    /// Copies a monolith from a buffer into a new, mutable index in host memory.
    pub fn load_ram_buffer<'config>(config: impl Into<DenseLoadRamBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_ram_buffer(&mut usearch_dense_load_ram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                ram: ram_options(config.memory_domains, config.cpu_class_ceiling),
                connectivity: 0,
                values_per_key: config.values_per_key,
                progress,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a monolith from a buffer into a new, mutable index in device memory.
    pub fn load_vram_buffer<'config>(config: impl Into<DenseLoadVramBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let progress = lower_progress(&mut config.progress, config.progress_steps);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_vram_buffer(&mut usearch_dense_load_vram_buffer_t {
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

    /// Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
    /// verbs report [`Status::ReadOnly`]; the files must stay unmodified while it lives.
    pub fn view_ram(path: impl AsRef<Path>) -> Result<Self> {
        let path = path_view(path.as_ref())?;
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_view_ram(&mut usearch_dense_view_ram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                path,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Serves a checkpoint in place from NVMe drives, as a read-only index whose mutating verbs
    /// report [`Status::ReadOnly`].
    pub fn view_nvme<'config>(config: impl Into<DenseViewNvmeConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let views = path_views(config.paths)?;
        let key_table = config.key_table_domains.map(|domains| ram_options(domains, config.cpu_class_ceiling));
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_view_nvme(&mut usearch_dense_view_nvme_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                nvme: nvme_options(&views, config.cpu_class_ceiling),
                ram: key_table.as_ref().map_or(ptr::null(), |ram| ram),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Opens a checkpoint's manifest from NVMe drives as a writable index: it answers the verbs a
    /// view answers, [`DenseVerbs::refine`] rewrites its parts one at a time, and a lone monolith
    /// is refused with [`Status::Unsupported`].
    pub fn load_nvme<'config>(config: impl Into<DenseViewNvmeConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let views = path_views(config.paths)?;
        let key_table = config.key_table_domains.map(|domains| ram_options(domains, config.cpu_class_ceiling));
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_nvme(&mut usearch_dense_load_nvme_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                nvme: nvme_options(&views, config.cpu_class_ceiling),
                ram: key_table.as_ref().map_or(ptr::null(), |ram| ram),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Reads a checkpoint's geometry and metric without loading it; a manifest answers from its
    /// summary and its first part's header, so the cost does not grow with the index.
    pub fn metadata(path: impl AsRef<Path>) -> Result<DenseMetadata> {
        let path = path_view(path.as_ref())?;
        let mut metadata = DenseMetadata::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_metadata_read(&mut usearch_dense_metadata_read_t {
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path,
                metadata: &mut metadata,
            })
        };
        check(outcome).map(|()| metadata)
    }

    /// Reads geometry and metric out of a serialized monolith buffer.
    pub fn metadata_buffer(buffer: &[u8]) -> Result<DenseMetadata> {
        let mut metadata = DenseMetadata::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_metadata_read_buffer(&mut usearch_dense_metadata_read_buffer_t {
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

/// What a verb answers through `Receiver`: what the call produced, or a [`Deferred`](crate::Deferred).
type Answer<'index, Receiver, Inputs, Output> =
    <<Receiver as DenseVerbs<'index>>::Completion as Completion>::Answer<'index, Inputs, Output>;

/// Every dense verb, written once for every receiver: `&DenseIndex` blocks on the calling thread,
/// and the [`On`] that [`DenseIndex::on`] or [`DenseIndex::deferred`] makes runs on an executor.
///
/// A blocking receiver answers what the call produced; a deferring one answers a
/// [`Deferred`](crate::Deferred) that borrows the index and owns the inputs. Every configurable
/// verb's config converts into its `Dense*Config`, so the bare value is the whole call and
/// `..DenseSearchConfig::from(10)` reaches every knob.
/// Every verb whose output scales with the call borrows an [`Arena`] last as `&mut Arena`, blocking
/// or deferring alike, except `save_buffer`, whose config holds it, and its result borrows the arena
/// in turn.
///
/// Rust-only.
pub trait DenseVerbs<'index>: Copy {
    /// [`Blocking`] or [`Deferring`].
    type Completion: Completion;

    /// The index this receiver reaches and the executor its calls run on.
    #[doc(hidden)]
    fn handles(&self) -> (usearch_dense_index_t, usearch_executor_t);

    /// Reserves capacity for a number of incoming vectors ahead of insertion, a hint, since
    /// insertions grow the index on demand.
    fn reserve(self, capacity: usize) -> Result<Answer<'index, Self, (), ()>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), (), |call, completion| {
            unsafe {
                usearch_dense_reserve(&mut usearch_dense_reserve_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    capacity,
                })
            };
            Ok(())
        })
    }

    /// Reports the frozen metadata and the current counters in one call.
    fn describe(self) -> Result<Answer<'index, Self, (), DenseIndexDescription>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), DenseIndexDescription::default(), |call, completion| {
            unsafe {
                usearch_dense_describe(&mut usearch_dense_describe_t {
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

    /// The number of entries stored under `key`.
    fn count(self, key: Key) -> Result<Answer<'index, Self, (), usize>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), 0, |call, completion| {
            unsafe {
                usearch_dense_count(&mut usearch_dense_count_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    key,
                    entries: &mut call.output,
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
        Self: DenseVerbs<'index, Completion = Blocking>,
    {
        Ok(self.count(key)? != 0)
    }

    /// Retrieves every vector stored under `key` into `arena` as `Value` rows, which the result
    /// borrows; an absent key yields no rows.
    fn get<'arena, Value: Element>(
        self,
        key: Key,
        arena: &'arena mut Arena,
    ) -> Result<Answer<'index, Self, (), DenseVectors<'arena, Value>>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), DenseVectors::new(arena), |call, completion| {
            let vectors = &mut call.output;
            unsafe {
                usearch_dense_get(&mut usearch_dense_get_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    key,
                    vectors: &mut vectors.vectors,
                    written: &mut vectors.count,
                    arena: vectors.arena.handle,
                    outputs_scalar: Value::SCALAR as usearch_scalar_t,
                })
            };
            Ok(())
        })
    }

    /// Adds a matrix of vectors under their keys - the native batch operation.
    ///
    /// A batch is all-or-nothing and becomes visible at once. Every entry is claimed before the first
    /// progress report, so a `Break` is ignored. Overlaps safely with concurrent searches on the
    /// same handle, and the input scalar type is independent of the stored quantization - the
    /// library converts on ingest.
    fn insert<'config, Keys: AsRef<[Key]>, Rows: RowsView>(
        self,
        keys: Keys,
        config: impl Into<DenseInsertConfig<'config, Rows>>,
    ) -> Result<Answer<'index, Self, (Keys, DenseInsertConfig<'config, Rows>), ()>>
    where
        Self::Completion: Accepts<Keys> + Accepts<DenseInsertConfig<'config, Rows>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer((keys, config.into()), (), |call, completion| {
            let (keys, config) = &mut call.inputs;
            let (keys, batch) = (keys.as_ref(), config.vectors.batch()?);
            // The keys view carries no count, so only this side can see the two disagree.
            if batch.count != keys.len() {
                return Err(fail(Status::ShapeMismatch, "the key count does not match the vector row count"));
            }
            unsafe {
                usearch_dense_insert(&mut usearch_dense_insert_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: batch.count,
                    keys: usearch_keys_view_t::of(keys),
                    vectors: batch.rows,
                    expansion_factor: config.expansion_factor,
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                })
            };
            Ok(())
        })
    }

    /// Runs one refine round over the whole graph, answering a [`RefineReport`] of the entries
    /// re-selected and changed, the islands it re-linked, and whether another round can help.
    ///
    /// A cheap insertion pass followed by rounds beats one expensive pass. On a RAM handle searches
    /// may run throughout, on a VRAM one they may not, and insertions never may. A stop answers
    /// [`Status::Cancelled`] at the next barrier; without a plan it closes the round there, leaving
    /// the graph correct, and with one it leaves the plan live for the next call.
    fn refine<'config, Plan: BorrowMut<DenseRefinePlan>>(
        self,
        config: DenseRefineConfig<'config, Plan>,
    ) -> Result<Answer<'index, Self, DenseRefineConfig<'config, Plan>, RefineReport>>
    where
        Self::Completion: Accepts<DenseRefineConfig<'config, Plan>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer(config, RefineReport::default(), |call, completion| {
            let config = &mut call.inputs;
            unsafe {
                usearch_dense_refine(&mut usearch_dense_refine_t {
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

    /// Closes a refine plan, whatever the call answers, and answers what the whole plan did to the
    /// graph; a round a stop left closes as a stop at that barrier would.
    fn refine_plan_finish(self, plan: DenseRefinePlan) -> Result<Answer<'index, Self, (), RefineReport>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), RefineReport::default(), |call, completion| {
            unsafe {
                usearch_dense_refine_plan_finish(&mut usearch_dense_refine_plan_finish_t {
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

    /// Measures how much of the graph a search reaches from the root, swept level by level.
    ///
    /// Read-only, so concurrent insertions merely leave fresh entries uncounted.
    fn health(self) -> Result<Answer<'index, Self, (), DenseReachability>> {
        let (index, executor) = self.handles();
        Self::Completion::answer((), DenseReachability::default(), |call, completion| {
            unsafe {
                usearch_dense_health(&mut usearch_dense_health_t {
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

    /// Resolves a matrix of queries into up to `wanted` nearest matches apiece, writing them into
    /// `arena` as [`Matches`].
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    fn search<'config, 'arena, Queries: RowsView>(
        self,
        queries: Queries,
        config: impl Into<DenseSearchConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Result<Answer<'index, Self, (Queries, DenseSearchConfig<'config>), Matches<'arena>>>
    where
        Self::Completion: Accepts<Queries> + Accepts<DenseSearchConfig<'config>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer((queries, config.into()), Matches::new(arena), |call, completion| {
            let (queries, config) = &mut call.inputs;
            let batch = queries.batch()?;
            call.output.queries = batch.count;
            unsafe {
                usearch_dense_search(&mut usearch_dense_search_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: batch.count,
                    queries: batch.rows,
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

    /// Late-interaction scoring of multi-row queries against multi-row entries, writing up to
    /// `wanted` keys per query, largest score first, in `arena` as one ragged run; zero keeps every
    /// key a query scored.
    ///
    /// Every query row takes its best match within a key, and those maxima sum to that key's
    /// score - so `rows_per_query` is the token count each query occupies, `queries` holds them
    /// end to end, and `wanted` counts documents rather than rows. This is what `values_per_key`
    /// above one exists for. Runs on an inner-product index alone.
    ///
    /// See <https://arxiv.org/abs/2004.12832> - ColBERT, which introduced this scoring.
    fn maxsim<'config, 'arena, Queries: RowsView>(
        self,
        queries: Queries,
        rows_per_query: usize,
        config: impl Into<DenseMaxsimConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Result<Answer<'index, Self, (Queries, DenseMaxsimConfig<'config>), DenseProducts<'arena>>>
    where
        Self::Completion: Accepts<Queries> + Accepts<DenseMaxsimConfig<'config>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer((queries, config.into()), DenseProducts::new(arena), |call, completion| {
            let (queries, config) = &mut call.inputs;
            let batch = queries.batch()?;
            // The call counts queries, so only this side sees the matrix's rows fail to divide.
            if rows_per_query == 0 || !batch.count.is_multiple_of(rows_per_query) {
                return Err(fail(Status::ShapeMismatch, "the query rows do not divide into whole queries"));
            }
            let products = &mut call.output;
            products.count = batch.count / rows_per_query;
            unsafe {
                usearch_dense_maxsim(&mut usearch_dense_maxsim_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: products.count,
                    queries: batch.rows,
                    query_offsets: ptr::null(),
                    rows_per_query,
                    matches: &mut products.products,
                    written: ptr::null_mut(),
                    required: &mut products.required,
                    arena: products.arena.handle,
                    wanted: config.wanted,
                    expansion_factor: config.expansion_factor,
                    traversal: Traversal::Walk as usearch_traversal_t,
                    predicate: lower_predicate(&None),
                    hints: ptr::null(),
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                })
            };
            Ok(())
        })
    }

    /// Approximate sparse product of a matrix against the indexed vectors, writing its keys,
    /// products and offsets in `arena` as one ragged run.
    ///
    /// The linear-algebra sibling of [`DenseVerbs::search`], reporting every pair the traversal
    /// scored rather than only the nearest. Pairs arrive in traversal order rather than sorted.
    /// Runs on an inner-product index alone, since a product is a logit.
    fn multiply<'config, 'arena, Left: RowsView>(
        self,
        left: Left,
        config: impl Into<DenseMultiplyConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Result<Answer<'index, Self, (Left, DenseMultiplyConfig<'config>), DenseProducts<'arena>>>
    where
        Self::Completion: Accepts<Left> + Accepts<DenseMultiplyConfig<'config>>,
    {
        let (index, executor) = self.handles();
        Self::Completion::answer((left, config.into()), DenseProducts::new(arena), |call, completion| {
            let (left, config) = &mut call.inputs;
            let batch = left.batch()?;
            let products = &mut call.output;
            products.count = batch.count;
            unsafe {
                usearch_dense_multiply(&mut usearch_dense_multiply_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: batch.count,
                    left: batch.rows,
                    products: &mut products.products,
                    written: ptr::null_mut(),
                    required: &mut products.required,
                    arena: products.arena.handle,
                    wanted: config.wanted,
                    expansion_factor: config.expansion_factor,
                    traversal: Traversal::Walk as usearch_traversal_t,
                    predicate: lower_predicate(&None),
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                })
            };
            Ok(())
        })
    }

    /// Softmax attention of every query row over every indexed key its walk scores, folded as
    /// Flash Attention folds key tiles, with no top-k cut, writing into `arena` as rows of the
    /// queries' own scalar type.
    ///
    /// The index holds one layer's keys for one key-value head and no values: `value_lookup`
    /// resolves every scored key to its value row, as wide as the keys and stored in the index's
    /// own quantization. A key left without a row, or given one of another type or width, refuses the
    /// call with [`Status::ShapeMismatch`], since a missing row is a wrong softmax rather than a
    /// smaller one. A VRAM walk resolves values on the device, where no Rust closure runs, so an
    /// arena from a VRAM executor is refused with [`Status::Device`]. The outputs come back
    /// already divided by the partition. Runs on an inner-product index alone, since a softmax
    /// needs logits.
    ///
    /// See <https://arxiv.org/abs/2205.14135> - Flash Attention, whose recurrence this reuses.
    #[allow(clippy::type_complexity)]
    fn attend<'config, 'lookup, 'arena, Queries: RowsView, Value: Element>(
        self,
        queries: Queries,
        value_lookup: &'lookup ValueLookup<'lookup, Value>,
        config: impl Into<DenseAttendConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Result<
        Answer<
            'index,
            Self,
            (Queries, &'lookup ValueLookup<'lookup, Value>, DenseAttendConfig<'config>),
            DenseVectors<'arena, Queries::Value>,
        >,
    >
    where
        Self::Completion:
            Accepts<Queries> + Accepts<&'lookup ValueLookup<'lookup, Value>> + Accepts<DenseAttendConfig<'config>>,
    {
        let (index, executor) = self.handles();
        let inputs = (queries, value_lookup, config.into());
        Self::Completion::answer(inputs, DenseVectors::new(arena), |call, completion| {
            let (queries, value_lookup, config) = &mut call.inputs;
            let outputs = &mut call.output;
            // A VRAM walk calls its value lookup on the device, where no host trampoline runs.
            if outputs.arena.describe()?.residency == Residency::Vram {
                return Err(fail(Status::Device, "a VRAM walk needs a device-callable value lookup"));
            }
            let batch = queries.batch()?;
            let carry = match config.carry.as_deref_mut() {
                None => ptr::null_mut(),
                // The carry pointer has no length, so only this side can see it fall short.
                Some(carry) if carry.len() < batch.count => {
                    return Err(fail(
                        Status::ShapeMismatch,
                        "the softmax carry holds fewer entries than there are queries",
                    ));
                }
                Some(carry) => carry.as_mut_ptr(),
            };
            outputs.count = batch.count;
            unsafe {
                usearch_dense_attend(&mut usearch_dense_attend_t {
                    index,
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    count: batch.count,
                    queries: batch.rows,
                    head_count: config.head_count,
                    value_lookup: Some(value_lookup_trampoline::<Value>),
                    value_lookup_state: ptr::from_mut(value_lookup).cast(),
                    carry,
                    outputs: &mut outputs.vectors,
                    arena: outputs.arena.handle,
                    outputs_scalar: Queries::Value::SCALAR as usearch_scalar_t,
                    expansion_factor: config.expansion_factor,
                    score_scale: config.score_scale,
                    traversal: Traversal::Walk as usearch_traversal_t,
                    predicate: lower_predicate(&None),
                    progress: lower_progress(&mut config.progress, config.progress_steps),
                    stats: output_slot(&mut config.stats),
                })
            };
            Ok(())
        })
    }

    /// Moves the entries of the sources into the targets, keeping their links rather than
    /// re-inserting their vectors, then runs a round on every target that took entries.
    ///
    /// Several sources into one target among them gather into it; one source into several empty
    /// targets cuts into one graph-local part apiece, drawn from `seed`; one handle as its only
    /// source and target recuts its own shards in place under `shard_capacity`. Every handle stays
    /// the caller's: a source the targets do not name ends empty and live. Readers of a RAM target
    /// keep searching throughout, while a VRAM one admits no overlapping search. A stop cuts only
    /// the closing rounds, every entry moved, and answers [`Status::Cancelled`].
    fn reshard<'config, Plan: BorrowMut<DenseReshardPlan>>(
        self,
        config: DenseReshardConfig<'config, Plan>,
    ) -> Result<Answer<'index, Self, (DenseReshardConfig<'config, Plan>, ReshardLists), RefineReport>>
    where
        Self::Completion: Accepts<DenseReshardConfig<'config, Plan>>,
    {
        let (_, executor) = self.handles();
        let handles = |named: &[&DenseIndex]| named.iter().map(|named| named.handle).collect();
        let paths = config.paths.map(path_views).transpose()?.unwrap_or_default();
        let lists = ReshardLists {
            sources: handles(config.sources),
            targets: handles(config.targets),
            ram: config.memory_domains.map(|domains| usearch_ram_options_t {
                shard_capacity: config.shard_capacity,
                ..ram_options(domains, None)
            }),
            vram: config.devices.map(|devices| usearch_vram_options_t {
                shard_capacity: config.shard_capacity,
                ..vram_options(devices, None)
            }),
            nvme: config.paths.map(|_| nvme_options(&paths, None)),
            _paths: paths,
        };
        Self::Completion::answer((config, lists), RefineReport::default(), |call, completion| {
            let (config, lists) = &mut call.inputs;
            unsafe {
                usearch_dense_reshard(&mut usearch_dense_reshard_t {
                    sources_count: lists.sources.len(),
                    sources: lists.sources.as_ptr(),
                    targets_count: lists.targets.len(),
                    targets: lists.targets.as_ptr(),
                    executor,
                    outcome: &mut call.outcome,
                    completion,
                    ram: lists.ram.as_ref().map_or(ptr::null(), ptr::from_ref),
                    vram: lists.vram.as_ref().map_or(ptr::null(), ptr::from_ref),
                    nvme: lists.nvme.as_ref().map_or(ptr::null(), ptr::from_ref),
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

    /// Closes a reshard plan, whatever the call answers, and answers what the whole plan did to
    /// the graph; a plan whose entries have not all moved runs no round and reports zeros.
    fn reshard_plan_finish(self, plan: DenseReshardPlan) -> Result<Answer<'index, Self, (), RefineReport>> {
        let (_, executor) = self.handles();
        Self::Completion::answer((), RefineReport::default(), |call, completion| {
            unsafe {
                usearch_dense_reshard_plan_finish(&mut usearch_dense_reshard_plan_finish_t {
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

    /// Draws keys around the focus, or across the whole index, each hanging from a shown or
    /// earlier draw and placed in three dimensions beside the shown ones, writing up to `wanted`
    /// of them into `arena` as a [`Sample`]. A larger `wanted` under the same seed draws a
    /// superset; only a RAM index samples yet.
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
            return Err(fail(Status::InvalidArgument, "every shown key needs its coordinates"));
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
                usearch_dense_sample(&mut usearch_dense_sample_t {
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
                        scalar: Scalar::F32 as usearch_scalar_t,
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
    ///
    /// Demands a mutation-free window; concurrent readers keep searching throughout. A stopped save
    /// leaves leftovers every reader rejects and the caller is left to remove.
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
                usearch_dense_save(&mut usearch_dense_save_t {
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
                usearch_dense_save_buffer(&mut usearch_dense_save_buffer_t {
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

    /// Opens a walk over the current contents, pulled blocking on this receiver's executor; each
    /// shard's publication watermark is read as the walk enters it, so an entry once passed is never
    /// revisited.
    ///
    /// Without `starts` the walk yields every entry as stored. With them it yields the start keys'
    /// own entries first, in order, then the entries those link to breadth-first, each entry once.
    fn cursor(self, starts: &[Key]) -> Result<DenseCursor<'index>>
    where
        Self: DenseVerbs<'index, Completion = Blocking>,
    {
        let (index, executor) = self.handles();
        let mut cursor = DenseCursor { handle: ptr::null_mut(), index, executor, receiver: PhantomData };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_cursor_init(&mut usearch_dense_cursor_init_t {
                executor: ptr::null_mut(),
                cursor: &mut cursor.handle,
                index,
                outcome: &mut outcome,
                starts_count: starts.len(),
                starts: starts.as_ptr(),
            })
        };
        check(outcome).map(|()| cursor)
    }

    /// Opens a resumable nearest-neighbor stream for a one-row query, pulled blocking on this
    /// receiver's executor.
    ///
    /// Best-first traversal is naturally pausable, so callers that post-filter aggressively pull
    /// until satisfied instead of guessing "k" upfront. The stream keeps walk state rather than a
    /// thread.
    ///
    /// No residency implements the stream yet, so the engine answers [`Status::Unsupported`].
    fn search_stream<Query: RowsView>(
        self,
        config: impl Into<DenseSearchStreamInitConfig<Query>>,
    ) -> Result<DenseSearchStream<'index>>
    where
        Self: DenseVerbs<'index, Completion = Blocking>,
    {
        let config = config.into();
        // The stream takes no count, so only this side sees a matrix of several rows.
        let query = match config.query.batch()? {
            batch if batch.count == 1 => batch.rows,
            _ => return Err(fail(Status::ShapeMismatch, "a stream walks exactly one query row")),
        };
        let (index, executor) = self.handles();
        let mut stream = DenseSearchStream { handle: ptr::null_mut(), index, executor, receiver: PhantomData };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_search_stream_init(&mut usearch_dense_search_stream_init_t {
                stream: &mut stream.handle,
                index,
                outcome: &mut outcome,
                query,
                expansion_factor: config.expansion_factor,
            })
        };
        check(outcome).map(|()| stream)
    }
}

impl<'index> DenseVerbs<'index> for &'index DenseIndex {
    type Completion = Blocking;

    fn handles(&self) -> (usearch_dense_index_t, usearch_executor_t) {
        (self.handle, ptr::null_mut())
    }
}

impl<'index, Completion: crate::types::Completion> DenseVerbs<'index> for On<'index, DenseIndex, Completion> {
    type Completion = Completion;

    fn handles(&self) -> (usearch_dense_index_t, usearch_executor_t) {
        (self.index.handle, self.executor.handle)
    }
}

/// The handle lists and placements a reshard lends, laid out as the library reads them.
#[doc(hidden)]
pub struct ReshardLists {
    sources: Vec<usearch_dense_index_t>,
    targets: Vec<usearch_dense_index_t>,
    ram: Option<usearch_ram_options_t>,
    vram: Option<usearch_vram_options_t>,
    /// The path views `nvme` points into.
    _paths: Vec<usearch_bytes_view_t>,
    nvme: Option<usearch_nvme_options_t>,
}

// Safety: the handles belong to borrowed `DenseIndex`es, which are `Sync`, and the options point
// into the borrowed config and the paths beside them.
unsafe impl Send for ReshardLists {}

/// # Safety
/// `state` must address the `&ValueLookup` lent to the call, and `keys` and `rows` must hold
/// `count` entries each.
unsafe extern "C" fn value_lookup_trampoline<Value: Element>(
    state: *mut c_void,
    keys: *const Key,
    count: usize,
    scalar: usearch_scalar_t,
    columns: usize,
    rows: *mut *const c_void,
) {
    unsafe {
        let lookup = *state.cast::<&ValueLookup<'_, Value>>();
        // The library reads rows of its own shape, so a lookup typed otherwise answers no row.
        let row_length = match scalar == Value::SCALAR as usearch_scalar_t {
            true => values_of::<Value>(columns),
            false => usize::MAX,
        };
        let keys = slice::from_raw_parts(keys, count);
        for (key, row) in keys.iter().zip(slice::from_raw_parts_mut(rows, count)) {
            *row = match lookup(*key) {
                Some(value) if value.len() == row_length => value.as_ptr().cast(),
                _ => ptr::null(),
            };
        }
    }
}

// endregion: Dense Index

// region: Cursors and Streams

/// A walk over the index contents; see [`DenseVerbs::cursor`].
///
/// The walk is a sequential scan, so it stays a batch operation: export, migration, and
/// visualization all move whole slabs.
///
/// Dropping it waits for every deferred call naming the cursor, then releases it.
#[derive(Debug)]
pub struct DenseCursor<'index> {
    handle: usearch_dense_cursor_t,
    index: usearch_dense_index_t,
    executor: usearch_executor_t,
    receiver: PhantomData<On<'index, DenseIndex>>,
}

impl DenseCursor<'_> {
    /// Copies up to `wanted` next keys into `arena`; an empty batch means the walk is exhausted.
    pub fn next_keys<'arena>(&mut self, wanted: usize, arena: &'arena mut Arena) -> Result<DenseBatch<'arena>> {
        self.next(false, wanted, arena)
    }

    /// Copies up to `wanted` next keys into `arena`, each with the keys it links to; an empty
    /// batch means the walk is exhausted.
    pub fn next_links<'arena>(&mut self, wanted: usize, arena: &'arena mut Arena) -> Result<DenseBatch<'arena>> {
        self.next(true, wanted, arena)
    }

    /// Pulls a batch, exporting each entry's links only when `linked` asks.
    fn next<'arena>(&mut self, linked: bool, wanted: usize, arena: &'arena mut Arena) -> Result<DenseBatch<'arena>> {
        let mut batch = DenseBatch {
            arena: PhantomData,
            keys: usearch_keys_view_t::default(),
            links: linked.then(usearch_keys_tape_view_t::default),
            written: 0,
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_cursor_next(&mut usearch_dense_cursor_next_t {
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

impl Drop for DenseCursor<'_> {
    fn drop(&mut self) {
        unsafe { usearch_dense_cursor_free(self.handle) }
    }
}

/// A resumable nearest-neighbor stream; see [`DenseVerbs::search_stream`].
///
/// Ordering beyond the expansion horizon is best-effort nearest, not globally exact. Each pull
/// lands its matches in the arena it is given as one query of a search, so nothing allocates per
/// match. An empty pull means the walk is drained.
///
/// Dropping it waits for every deferred call naming the stream, then releases the walk state it
/// keeps.
#[derive(Debug)]
pub struct DenseSearchStream<'index> {
    handle: usearch_dense_search_stream_t,
    index: usearch_dense_index_t,
    executor: usearch_executor_t,
    receiver: PhantomData<On<'index, DenseIndex>>,
}

impl DenseSearchStream<'_> {
    /// Pulls up to `wanted` more matches into `arena`, as the one query of [`Matches`].
    pub fn next<'arena>(&mut self, wanted: usize, arena: &'arena mut Arena) -> Result<Matches<'arena>> {
        let mut matches = Matches::new(arena);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_search_stream_next(&mut usearch_dense_search_stream_next_t {
                index: self.index,
                executor: self.executor,
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                stream: self.handle,
                matches: &mut matches.view,
                arena: matches.arena.handle,
                wanted,
            })
        };
        matches.queries = 1;
        check(outcome).map(|()| matches)
    }
}

impl Drop for DenseSearchStream<'_> {
    fn drop(&mut self) {
        unsafe { usearch_dense_search_stream_free(self.handle) }
    }
}

// endregion: Cursors and Streams

// region: Tests

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use numkong::{Matrix, MatrixView, bf16, u1x8};
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::testing::{
        ScratchDirectory, assert_close, assert_close_rows, block_on, case_stream_key, executor, skipped,
    };
    use crate::types::{Deferred, Mood, Phase, rows};

    // region: Helpers

    /// Deterministic unit-length rows, so inner, angular and euclidean rank alike.
    fn seeded_rows(count: usize, dimensions: usize, seed: u64) -> Vec<f32> {
        let mut generator = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut slab = vec![0.0f32; count * dimensions];
        for row in slab.chunks_mut(dimensions) {
            loop {
                for value in row.iter_mut() {
                    *value = generator.random_range(-1.0f32..1.0);
                }
                let magnitude = row.iter().map(|value| value * value).sum::<f32>().sqrt();
                if magnitude > 1e-3 {
                    for value in row.iter_mut() {
                        *value /= magnitude;
                    }
                    break;
                }
            }
        }
        slab
    }

    fn sequential_keys(count: usize) -> Vec<Key> {
        (0..count as u64).map(Key::from).collect()
    }

    fn empty_index(dimensions: usize, metric: DenseMetric, quantization: Scalar) -> Result<DenseIndex> {
        let options = DenseOptions { dimensions, metric, quantization, ..DenseOptions::default() };
        let index = DenseIndex::init_ram(&options, &[])?;
        index.reserve(1024)?;
        Ok(index)
    }

    /// An empty index, or `None` when this build leaves its quantization out.
    fn empty_index_or_skip(dimensions: usize, metric: DenseMetric, quantization: Scalar) -> Result<Option<DenseIndex>> {
        match empty_index(dimensions, metric, quantization) {
            Err(error) if error.status == Status::Unsupported => {
                skipped(&format!("built without {quantization}"));
                Ok(None)
            }
            built => built.map(Some),
        }
    }

    fn populated_index(
        count: usize,
        dimensions: usize,
        metric: DenseMetric,
        quantization: Scalar,
    ) -> Result<(DenseIndex, Vec<Key>, Vec<f32>)> {
        let index = empty_index(dimensions, metric, quantization)?;
        index.reserve(count)?;
        let keys = sequential_keys(count);
        let slab = seeded_rows(count, dimensions, case_stream_key(0));
        index.on(&executor()).insert(&keys, &rows(&slab, dimensions)?)?;
        assert_eq!(index.describe()?.size, count);
        Ok((index, keys, slab))
    }

    /// How many matches one query reports - the shape most assertions here want.
    fn matches_of(index: &DenseIndex, query: &[f32], dimensions: usize) -> Result<usize> {
        let mut arena = Arena::new()?;
        let matches = index.search(&rows(query, dimensions)?, 1, &mut arena)?;
        Ok(matches.query(0)?.keys.len())
    }

    /// The rows a `get` exported, laid end to end.
    fn flattened<Value: Element>(rows: MatrixView<'_, Value>) -> Vec<Value> {
        (0..rows.shape()[0])
            .flat_map(|row| rows.slice_leading(row).unwrap().as_contiguous_slice().unwrap().to_vec())
            .collect()
    }

    /// The spacing of bf16 values at one, the suites' stored kind.
    const BF16_EPSILON: f64 = 1.0 / 128.0;

    fn requires_send_sync<Subject: Send + Sync>() {}

    fn requires_send<Subject: Send>() {}

    /// Wraps raw bytes as packed bit rows, eight dimensions to a byte.
    fn bit_rows(bytes: &[u8]) -> Vec<u1x8> {
        bytes.iter().map(|&byte| u1x8(byte)).collect()
    }

    /// Whether a watched call is let run to its end or stopped at its first report.
    enum Patience {
        /// Let every report through.
        Complete,
        /// Ask for a stop the first time the call reports.
        StopAtFirstReport,
    }

    /// What a progress handler saw: the last report, and whether the counts ever went backwards.
    ///
    /// The number of reports is deliberately not recorded - the engine ticks once per chunk where
    /// the stand-in ticks once per entry - and the assertions run after the call, since a panic
    /// inside a handler would abort.
    struct Watch {
        /// The finished and total counts of the last report.
        last: (usize, usize),
        /// Whether every report's finished count was at least its predecessor's.
        ordered: bool,
        /// What this handler answers with.
        patience: Patience,
    }

    impl Watch {
        fn new(patience: Patience) -> Self {
            Watch { last: (0, 0), ordered: true, patience }
        }

        fn saw(&mut self, finished: usize, total: usize) -> ControlFlow<()> {
            self.ordered &= finished >= self.last.0;
            self.last = (finished, total);
            match self.patience {
                Patience::Complete => ControlFlow::Continue(()),
                Patience::StopAtFirstReport => ControlFlow::Break(()),
            }
        }
    }

    // endregion: Helpers

    // region: Construction and Geometry

    /// The one place the Rust enum values are checked against the C ones: a renumbered
    /// constant reaches the library as a different kind, and is caught here and nowhere else.
    /// A storage kind the build's `USEARCH_DENSE_DTYPES` left out is refused as unsupported.
    #[test]
    fn constructs_every_metric_and_quantization() {
        for quantization in [
            Scalar::BF16,
            Scalar::F16,
            Scalar::E5M2,
            Scalar::E4M3,
            Scalar::E3M2,
            Scalar::E2M3,
            Scalar::I8,
            Scalar::I4x2,
            Scalar::U8,
            Scalar::U4x2,
            Scalar::U1x8,
        ] {
            let Some(index) = empty_index_or_skip(64, DenseMetric::Inner, quantization).unwrap() else {
                continue;
            };
            let described = index.describe().unwrap();
            assert_eq!(described.metadata.dimensions, 64, "{quantization:?}");
            assert_eq!(described.metadata.quantization, quantization);
            assert_eq!(described.size, 0, "{quantization:?}");
            assert!(described.metadata.connectivity > 0, "{quantization:?}");
        }

        // The bit-level metrics take packed rows, so each metric meets a kind it accepts.
        for (metric, quantization) in [
            (DenseMetric::Inner, Scalar::BF16),
            (DenseMetric::Angular, Scalar::BF16),
            (DenseMetric::Euclidean, Scalar::BF16),
            (DenseMetric::Hamming, Scalar::U1x8),
            (DenseMetric::Jaccard, Scalar::U1x8),
        ] {
            let Some(index) = empty_index_or_skip(64, metric, quantization).unwrap() else {
                continue;
            };
            let described = index.describe().unwrap();
            assert_eq!(described.metadata.metric, metric);
            assert_eq!(metric.to_string().parse(), Ok(metric), "the library spells it back");
        }
        assert!("cosine".parse::<DenseMetric>().is_err());
    }

    #[test]
    fn does_not_preallocate_members() {
        let options = DenseOptions { dimensions: 16, ..DenseOptions::default() };
        let index = DenseIndex::init_ram(&options, &[]).unwrap();
        let described = index.describe().unwrap();
        assert_eq!((described.size, described.capacity), (0, 0));

        index.reserve(1024).unwrap();
        let described = index.describe().unwrap();
        assert!(described.capacity >= 1024);
        assert_eq!(described.size, 0);
    }

    /// Init freezes the geometry, so a zeroed width is refused there rather than at the first batch.
    #[test]
    fn zero_dimensions_are_refused_at_init() {
        let options = DenseOptions { dimensions: 0, ..DenseOptions::default() };
        assert!(matches!(DenseIndex::init_ram(&options, &[]), Err(Error { status: Status::ConfigMismatch, .. })));
    }

    /// The handle is one pointer, so a move must not disturb it and two threads may share it.
    #[test]
    fn survives_box_and_arc_moves() {
        requires_send_sync::<DenseIndex>();
        let (index, _, _) = populated_index(4, 8, DenseMetric::Angular, Scalar::BF16).unwrap();

        let boxed = Box::new(index);
        assert_eq!(boxed.describe().unwrap().size, 4);
        let shared = std::sync::Arc::new(*boxed);
        assert_eq!(shared.describe().unwrap().size, 4);
    }

    #[test]
    fn clears_and_reuses() {
        const DIMENSIONS: usize = 8;
        let (mut index, keys, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let reserved = index.describe().unwrap().capacity;

        index.clear().unwrap();
        let cleared = index.describe().unwrap();
        assert_eq!(cleared.size, 0);
        assert_eq!(cleared.capacity, reserved, "clearing keeps the arena");
        assert_eq!(cleared.metadata.dimensions, DIMENSIONS, "and keeps the frozen geometry");

        index.insert(&keys, &rows(&slab, DIMENSIONS).unwrap()).unwrap();
        assert_eq!(index.describe().unwrap().size, 16);
    }

    // endregion: Construction and Geometry

    // region: Ingest and Retrieval

    #[test]
    fn adds_and_gets_vectors() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let keys = sequential_keys(2);
        let slab: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        index.insert(&keys, &rows(&slab, DIMENSIONS).unwrap()).unwrap();
        assert_eq!(index.describe().unwrap().size, 2);

        let mut arena = Arena::new().unwrap();
        let found = index.get::<f32>(keys[1], &mut arena).unwrap();
        assert_eq!(found.view().unwrap().shape(), &[1, DIMENSIONS]);
        assert_eq!(flattened(found.view().unwrap()), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            index.get::<f32>(Key::from(7u64), &mut arena).unwrap().view().unwrap().shape()[0],
            0,
            "an absent key yields no rows"
        );
    }

    /// An empty batch is a shape rather than a mistake, so it publishes nothing and reports no
    /// failure - what a caller filtering its own input relies on.
    #[test]
    fn accepts_an_empty_batch() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let empty: [f32; 0] = [];
        index.insert([], &rows(&empty, DIMENSIONS).unwrap()).unwrap();
        assert_eq!(index.describe().unwrap().size, 0);
    }

    #[test]
    fn rejects_mismatched_key_and_row_counts() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let slab = seeded_rows(2, DIMENSIONS, case_stream_key(0));
        assert!(matches!(
            index.insert(sequential_keys(3), &rows(&slab, DIMENSIONS).unwrap()),
            Err(Error { status: Status::ShapeMismatch, .. })
        ));
    }

    /// Every view carries the width the caller laid out, so the call refuses a narrow row before
    /// reading past it.
    #[test]
    fn rejects_wrong_dimension_matrices() {
        const DIMENSIONS: usize = 8;
        let (index, _, _) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let narrow = seeded_rows(1, DIMENSIONS - 1, case_stream_key(0));

        assert!(matches!(
            index.insert(sequential_keys(1), &rows(&narrow, DIMENSIONS - 1).unwrap()),
            Err(Error { status: Status::ShapeMismatch, .. })
        ));
        assert!(matches!(
            index.search(&rows(&narrow, DIMENSIONS - 1).unwrap(), 1, &mut Arena::new().unwrap()),
            Err(Error { status: Status::ShapeMismatch, .. })
        ));
        assert!(
            matches!(
                index.search_stream(rows(&narrow, DIMENSIONS - 1).unwrap()),
                Err(Error { status: Status::ShapeMismatch | Status::Unsupported, .. })
            ),
            "the query width is checked too, where a residency streams at all"
        );
    }

    #[test]
    fn rejects_non_contiguous_rows() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let slab = seeded_rows(2, DIMENSIONS * 2, case_stream_key(0));
        let element = core::mem::size_of::<f32>() as isize;
        // A column stride of two elements interleaves the rows, which the boundary forbids:
        // the ABI carries one row stride and no column stride.
        let interleaved: MatrixView<'_, f32> =
            unsafe { MatrixView::from_raw_parts(slab.as_ptr(), &[2, DIMENSIONS], &[element * 8, element * 2]) }
                .unwrap();
        assert!(matches!(
            index.insert(sequential_keys(2), interleaved),
            Err(Error { status: Status::UnsupportedLayout, .. })
        ));
    }

    #[test]
    fn rejects_descending_row_pitch() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let slab = seeded_rows(2, DIMENSIONS, case_stream_key(0));
        let element = core::mem::size_of::<f32>() as isize;
        let descending: MatrixView<'_, f32> = unsafe {
            MatrixView::from_raw_parts(
                slab.as_ptr().add(DIMENSIONS),
                &[2, DIMENSIONS],
                &[-(element * DIMENSIONS as isize), element],
            )
        }
        .unwrap();
        assert!(matches!(
            index.insert(sequential_keys(2), descending),
            Err(Error { status: Status::UnsupportedLayout, .. })
        ));
    }

    /// A pitch wider than the row is the layout a mapped dataset arrives in, and its rows must
    /// read back packed from the arena, padding left behind.
    #[test]
    fn accepts_strided_row_pitch() {
        const DIMENSIONS: usize = 4;
        const PITCH: usize = 8;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let mut padded = [0.0f32; 2 * PITCH];
        let rows_source = seeded_rows(2, DIMENSIONS, case_stream_key(0));
        padded[..DIMENSIONS].copy_from_slice(&rows_source[..DIMENSIONS]);
        padded[PITCH..PITCH + DIMENSIONS].copy_from_slice(&rows_source[DIMENSIONS..]);

        let element = core::mem::size_of::<f32>() as isize;
        let strided: MatrixView<'_, f32> = unsafe {
            MatrixView::from_raw_parts(padded.as_ptr(), &[2, DIMENSIONS], &[element * PITCH as isize, element])
        }
        .unwrap();
        index.insert(sequential_keys(2), strided).unwrap();

        let mut arena = Arena::new().unwrap();
        let walked_keys = index.cursor(&[]).unwrap().next_keys(2, &mut arena).unwrap().keys().to_vec();
        assert_eq!(walked_keys.len(), 2);
        for key in walked_keys {
            let source = u64::try_from(key).unwrap() as usize;
            let restored = flattened(index.get::<f32>(key, &mut arena).unwrap().view().unwrap());
            assert_close_rows(&restored, &rows_source[source * DIMENSIONS..][..DIMENSIONS], BF16_EPSILON);
        }
    }

    #[test]
    fn converts_scalar_kinds_on_ingest() {
        const DIMENSIONS: usize = 16;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let keys = sequential_keys(1);
        let slab = seeded_rows(1, DIMENSIONS, case_stream_key(0));
        index.insert(&keys, &rows(&slab, DIMENSIONS).unwrap()).unwrap();

        let wide = flattened(index.get::<f64>(keys[0], &mut Arena::new().unwrap()).unwrap().view().unwrap());
        assert_eq!(wide.len(), DIMENSIONS);
        for (left, right) in wide.iter().zip(&slab) {
            assert_close(*left, f64::from(*right), BF16_EPSILON);
        }
    }

    /// One call returns every entry under a key, however many there turn out to be.
    #[test]
    fn gets_every_entry_sharing_a_key() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let shared = Key::from(3u64);
        let slab = seeded_rows(3, DIMENSIONS, case_stream_key(0));
        index.insert([shared; 3], &rows(&slab, DIMENSIONS).unwrap()).unwrap();

        assert_eq!(index.count(shared).unwrap(), 3);
        assert!(index.contains(shared).unwrap());
        assert!(!index.contains(Key::from(8u64)).unwrap());

        let mut arena = Arena::new().unwrap();
        let restored = index.get::<f32>(shared, &mut arena).unwrap();
        assert_eq!(restored.view().unwrap().shape(), &[3, DIMENSIONS], "one row per entry");
        assert_close_rows(&flattened(restored.view().unwrap()), &slab, BF16_EPSILON);
    }

    /// An arena is reused once its outputs are no longer read, and keeping one call's outputs past
    /// the next takes a second one.
    #[test]
    fn a_second_arena_keeps_the_first_output_alive() {
        const DIMENSIONS: usize = 4;
        let (index, keys, slab) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let executor = executor();
        let receiver = index.on(&executor);
        let (mut first, mut second) = (Arena::new_in(&executor).unwrap(), Arena::new_in(&executor).unwrap());
        let held = receiver.get::<f32>(keys[0], &mut first).unwrap();
        let next = receiver.get::<f32>(keys[1], &mut second).unwrap();
        assert_close_rows(&flattened(held.view().unwrap()), &slab[..DIMENSIONS], BF16_EPSILON);
        assert_close_rows(&flattened(next.view().unwrap()), &slab[DIMENSIONS..DIMENSIONS * 2], BF16_EPSILON);

        let reused = receiver.get::<f32>(keys[3], &mut first).unwrap();
        assert_close_rows(&flattened(reused.view().unwrap()), &slab[DIMENSIONS * 3..], BF16_EPSILON);
    }

    #[test]
    fn uses_128_bit_keys() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let wide = Key::from(u128::MAX);
        let narrow = Key::from(9u64);
        let slab = seeded_rows(2, DIMENSIONS, case_stream_key(0));
        index.insert([wide, narrow], &rows(&slab, DIMENSIONS).unwrap()).unwrap();

        assert_eq!(index.count(wide).unwrap(), 1);
        assert_eq!(index.count(narrow).unwrap(), 1);
        assert!(u64::try_from(wide).is_err(), "a non-zero high half has no integer identity");
        assert_eq!(u64::try_from(narrow), Ok(9));
    }

    /// A sub-byte kind counts logical dimensions, so eight of them are one stored byte - the
    /// one place where the row width and the row extent part ways.
    #[test]
    fn packed_bit_rows_cross_as_one_byte_per_eight_dimensions() {
        const DIMENSIONS: usize = 8;
        let Some(index) = empty_index_or_skip(DIMENSIONS, DenseMetric::Hamming, Scalar::U1x8).unwrap() else {
            return skipped("built without u1x8");
        };
        let stored = bit_rows(&[0b0000_1111, 0b1111_0000]);
        let keys = [Key::from(42u64), Key::from(43u64)];
        index.insert(keys, &rows(&stored, DIMENSIONS).unwrap()).unwrap();
        assert_eq!(index.describe().unwrap().size, 2, "two bytes are two rows, not sixteen");

        let query = bit_rows(&[0b0111_1000]);
        let mut arena = Arena::new().unwrap();
        let matches = index.search(&rows(&query, DIMENSIONS).unwrap(), 2, &mut arena).unwrap();
        let found = matches.query(0).unwrap();
        assert_eq!(
            found.keys.iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
    }

    /// A sub-byte row comes back counted in dimensions, never in the bytes that hold them.
    #[test]
    fn gets_sub_byte_rows_at_their_logical_width() {
        const DIMENSIONS: usize = 16;
        let Some(index) = empty_index_or_skip(DIMENSIONS, DenseMetric::Hamming, Scalar::U1x8).unwrap() else {
            return skipped("built without u1x8");
        };
        let key = Key::from(5u64);
        index.insert([key], &rows(&bit_rows(&[0b1010_1010, 0b0000_0101]), DIMENSIONS).unwrap()).unwrap();

        let mut arena = Arena::new().unwrap();
        let found = index.get::<u1x8>(key, &mut arena).unwrap();
        assert_eq!(found.view().unwrap().shape(), &[1, DIMENSIONS]);
    }

    // endregion: Ingest and Retrieval

    // region: Search

    #[test]
    fn searches_an_empty_index() {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let query = seeded_rows(1, DIMENSIONS, case_stream_key(0));
        assert_eq!(matches_of(&index, &query, DIMENSIONS).unwrap(), 0);
    }

    /// A zero `wanted` walks nothing and lands only zeroed counts, with no keys to read.
    #[test]
    fn a_zero_wanted_lands_only_counts() {
        const DIMENSIONS: usize = 8;
        let (index, _, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let mut arena = Arena::new().unwrap();
        let matches = index.search(&rows(&slab[..4 * DIMENSIONS], DIMENSIONS).unwrap(), 0, &mut arena).unwrap();
        assert_eq!(matches.queries(), 4);
        assert!(matches.iter().all(|query| query.keys.is_empty()), "no query holds a match");
    }

    /// Every query's matches land packed in the arena, and its count in its own slot; a query past
    /// the call's count is refused rather than read.
    #[test]
    fn fills_every_query_of_a_batch() {
        const DIMENSIONS: usize = 64;
        const QUERIES: usize = 8;
        const WANTED: usize = 5;
        let (index, keys, slab) = populated_index(256, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let executor = executor();
        let mut arena = Arena::new_in(&executor).unwrap();
        let matches = index
            .on(&executor)
            .search(
                &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS).unwrap(),
                DenseSearchConfig { expansion_factor: 64, ..DenseSearchConfig::from(WANTED) },
                &mut arena,
            )
            .unwrap();
        assert_eq!(matches.queries(), QUERIES);
        assert!(matches.iter().all(|found| found.keys.len() == WANTED));
        for (query, found) in matches.iter().enumerate() {
            assert_eq!(found.keys[0], keys[query], "each query finds itself first");
        }
        assert_eq!(matches.query(QUERIES).unwrap_err().status, Status::InvalidArgument);
    }

    #[test]
    fn reports_traversal_stats() {
        const DIMENSIONS: usize = 32;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let keys = sequential_keys(64);
        let slab = seeded_rows(64, DIMENSIONS, case_stream_key(0));

        let mut inserting = DenseTraversalStats::default();
        index
            .insert(
                &keys,
                DenseInsertConfig {
                    stats: Some(&mut inserting),
                    ..DenseInsertConfig::from(&rows(&slab, DIMENSIONS).unwrap())
                },
            )
            .unwrap();
        assert!(inserting.computed_distances > 0);

        let query = rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap();
        let mut arena = Arena::new().unwrap();
        let mut searching = DenseTraversalStats::default();
        index
            .search(&query, DenseSearchConfig { stats: Some(&mut searching), ..DenseSearchConfig::from(4) }, &mut arena)
            .unwrap();
        let first = searching;
        assert!(first.computed_distances > 0);

        index
            .search(&query, DenseSearchConfig { stats: Some(&mut searching), ..DenseSearchConfig::from(4) }, &mut arena)
            .unwrap();
        assert_eq!(searching, first, "stats are overwritten by their call, never accumulated");
    }

    #[test]
    fn filters_with_a_stateless_predicate() {
        const DIMENSIONS: usize = 16;
        const WANTED: usize = 8;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let is_even = |key: Key| u64::try_from(key).is_ok_and(|value| value % 2 == 0);
        let mut arena = Arena::new().unwrap();
        let matches = index
            .search(
                &rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap(),
                DenseSearchConfig {
                    expansion_factor: 64,
                    predicate: Some(&is_even),
                    ..DenseSearchConfig::from(WANTED)
                },
                &mut arena,
            )
            .unwrap();

        let admitted = matches.query(0).unwrap();
        assert!(!admitted.keys.is_empty());
        for key in admitted.keys {
            assert!(is_even(*key), "{key} was not admitted by the predicate");
        }
    }

    /// The captured set crosses as the trampoline's context pointer, which is what separates
    /// this from the stateless case above.
    #[test]
    fn filters_with_a_stateful_predicate() {
        const DIMENSIONS: usize = 16;
        const WANTED: usize = 8;
        let (index, keys, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let allowed: HashSet<Key> = keys.iter().copied().take(5).collect();
        let executor = executor();
        let mut arena = Arena::new_in(&executor).unwrap();
        let matches = index
            .on(&executor)
            .search(
                &rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap(),
                DenseSearchConfig {
                    expansion_factor: 64,
                    predicate: Some(&|key: Key| allowed.contains(&key)),
                    ..DenseSearchConfig::from(WANTED)
                },
                &mut arena,
            )
            .unwrap();

        for key in matches.query(0).unwrap().keys {
            assert!(allowed.contains(key), "{key} is outside the captured set");
        }
    }

    #[test]
    fn filters_everything_out() {
        const DIMENSIONS: usize = 16;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let mut arena = Arena::new().unwrap();
        let matches = index
            .search(
                &rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap(),
                DenseSearchConfig { expansion_factor: 64, predicate: Some(&|_| false), ..DenseSearchConfig::from(4) },
                &mut arena,
            )
            .unwrap();
        assert!(matches.query(0).unwrap().keys.is_empty(), "a fully-rejecting predicate is not an error");
    }

    /// A scan scores every entry rather than the ones a walk reaches, so every query finds itself first.
    #[test]
    fn scans_find_every_query_itself() {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 64;
        let (index, keys, slab) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let mut arena = Arena::new().unwrap();
        let matches = index
            .search(
                &rows(&slab, DIMENSIONS).unwrap(),
                DenseSearchConfig { traversal: Traversal::Scan, ..DenseSearchConfig::from(1) },
                &mut arena,
            )
            .unwrap();
        assert_eq!(matches.queries(), COUNT);
        for (query, found) in matches.iter().enumerate() {
            assert_eq!(found.keys[0], keys[query]);
        }
    }

    // endregion: Search

    // region: Refinement and Linear Algebra

    /// The return is a report rather than a status, so a caller loops until the graph settles.
    #[test]
    fn refine_reports_its_round() {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, DenseMetric::Angular, Scalar::BF16).unwrap();
        let mut stats = DenseTraversalStats::default();
        let report = index
            .on(&executor())
            .refine(DenseRefineConfig { expansion_factor: 64, stats: Some(&mut stats), ..DenseRefineConfig::default() })
            .unwrap();
        assert!(report.changed <= report.reselected, "a sweep changes only the nodes it re-selected");
        assert_eq!(report.reachability.audited, COUNT, "the closing health check judges every entry");
        assert!(stats.computed_distances > 0, "the round reports its distances");
    }

    /// One source cut among empty targets leaves every entry in them, the source live and empty.
    #[test]
    fn reshards_one_source_into_several_targets() {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let targets = [
            empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap(),
            empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap(),
        ];
        let report = index
            .on(&executor())
            .reshard(DenseReshardConfig {
                sources: &[&index],
                targets: &[&targets[0], &targets[1]],
                seed: case_stream_key(1),
                ..DenseReshardConfig::default()
            })
            .unwrap();
        assert!(report.changed <= report.reselected);
        assert_eq!(targets[0].describe().unwrap().size + targets[1].describe().unwrap().size, COUNT);
        assert_eq!(index.describe().unwrap().size, 0);
    }

    /// Equal seeds over an unchanged index draw equally, so a larger draw extends a smaller one.
    #[test]
    fn a_larger_sample_draws_a_superset() {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, DenseMetric::Angular, Scalar::BF16).unwrap();
        let (mut smaller_arena, mut larger_arena) = (Arena::new().unwrap(), Arena::new().unwrap());
        let seed = case_stream_key(1);
        let seeded = || DenseSampleConfig { seed, ..Default::default() };
        let smaller = index.on(&executor()).sample(8, &mut smaller_arena, seeded()).unwrap();
        let larger = index.on(&executor()).sample(16, &mut larger_arena, seeded()).unwrap();
        assert_eq!((smaller.written(), larger.written()), (8, 16));
        assert_eq!(larger.coordinates().len(), 16);
        assert_eq!(larger.parents().queries(), 16);
        let drawn = larger.keys();
        assert!(smaller.keys().iter().all(|key| drawn.contains(key)));
    }

    /// The audit judges every entry published when it started, and no search reaches more.
    #[test]
    fn audits_how_much_of_the_graph_a_search_reaches() {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, DenseMetric::Angular, Scalar::BF16).unwrap();
        let reachability = index.on(&executor()).health().unwrap();
        assert_eq!(reachability.audited, COUNT);
        assert!(reachability.reachable != 0 && reachability.reachable <= COUNT);
        assert!(reachability.depth_from_root != 0, "a populated graph takes at least one depth");
    }

    /// Every row's pairs land in one ragged run, and `wanted` cuts each row while `required`
    /// still counts what the walks scored; the offsets land in the arena beside the run.
    #[test]
    fn multiplication_reports_every_scored_pair() {
        const DIMENSIONS: usize = 4;
        const ROWS: usize = 3;
        const WANTED: usize = 2;
        // A product is a logit, so the linear-algebra verbs take the inner kind alone.
        let index = empty_index(DIMENSIONS, DenseMetric::Inner, Scalar::BF16).unwrap();
        let slab = seeded_rows(ROWS, DIMENSIONS, case_stream_key(0));
        index.insert(sequential_keys(ROWS), &rows(&slab, DIMENSIONS).unwrap()).unwrap();

        let left = rows(&slab, DIMENSIONS).unwrap();
        let mut arena = Arena::new().unwrap();
        let every = index.multiply(&left, 0, &mut arena).unwrap();
        assert!(every.iter().all(|row| row.keys.len() == ROWS), "every row scores every column");

        let executor = executor();
        let config = DenseMultiplyConfig { expansion_factor: 2 * ROWS, ..DenseMultiplyConfig::from(WANTED) };
        let cut = index.multiply(&left, config, &mut arena).unwrap();
        assert_eq!(cut.queries(), ROWS);
        assert!(cut.iter().all(|row| row.keys.len() == WANTED));
        assert_eq!(cut.required(), ROWS * ROWS, "the count outruns the pairs kept");
        assert_eq!(cut.query(ROWS).unwrap_err().status, Status::InvalidArgument);

        let mut placed = Arena::new_in(&executor).unwrap();
        let same = index.on(&executor).multiply(&left, WANTED, &mut placed).unwrap();
        assert!(same.iter().all(|row| row.keys.len() == WANTED));
    }

    /// The outputs are sized over documents rather than rows, which is the one piece of
    /// arithmetic this door does that the batch search does not.
    #[test]
    fn late_interaction_scores_over_documents() {
        const DIMENSIONS: usize = 4;
        const TOKENS: usize = 2;
        const DOCUMENTS: usize = 3;
        const QUERIES: usize = 2;
        // A document is one key holding its token rows, which is what `values_per_key` is for.
        let options = DenseOptions {
            dimensions: DIMENSIONS,
            metric: DenseMetric::Inner,
            values_per_key: TOKENS,
            ..DenseOptions::default()
        };
        let index = DenseIndex::init_ram(&options, &[]).unwrap();
        index.reserve(DOCUMENTS * TOKENS).unwrap();

        let slab = seeded_rows(DOCUMENTS * TOKENS, DIMENSIONS, case_stream_key(0));
        let mut keys = Vec::with_capacity(DOCUMENTS * TOKENS);
        for document in 0..DOCUMENTS {
            keys.extend(core::iter::repeat_n(Key::from(document as u64), TOKENS));
        }
        index.insert(&keys, &rows(&slab, DIMENSIONS).unwrap()).unwrap();

        let tokens = rows(&slab[..QUERIES * TOKENS * DIMENSIONS], DIMENSIONS).unwrap();
        let executor = executor();
        let mut arena = Arena::new_in(&executor).unwrap();
        let matches = index.on(&executor).maxsim(&tokens, TOKENS, DOCUMENTS, &mut arena).unwrap();
        assert_eq!(matches.queries(), QUERIES);
        assert!(matches.iter().all(|query| query.keys.len() == DOCUMENTS), "every document is scored, once per query");
        let bounded: Vec<usize> = matches.iter().map(|query| query.keys.len()).collect();
        let every = index.maxsim(&tokens, TOKENS, 0, &mut arena).unwrap();
        assert!(
            every.iter().zip(bounded).all(|(query, kept)| query.keys.len() >= kept),
            "no bound lands every key a query scored"
        );
        assert_eq!(every.required(), every.iter().map(|query| query.keys.len()).sum::<usize>());
        assert!(matches!(
            index.maxsim(&tokens, TOKENS + 1, DOCUMENTS, &mut arena),
            Err(Error { status: Status::ShapeMismatch, .. })
        ));
    }

    /// A graph this small walks every key, so each head's output and carry are exact attention's,
    /// with one head per walk and with two sharing it; a key left without a row, or rows of another
    /// scalar type than the index stores, refuse.
    #[test]
    fn attention_is_exact_over_every_key() {
        const DIMENSIONS: usize = 4;
        const ENTRIES: usize = 16;
        const TOKENS: usize = 2;
        let index = empty_index(DIMENSIONS, DenseMetric::Inner, Scalar::BF16).unwrap();
        // Rounding every row through bf16 first lets the oracle score exactly what the index stores.
        let in_bf16 =
            |slab: Vec<f32>| -> Vec<f32> { slab.into_iter().map(|value| bf16::from_f32(value).to_f32()).collect() };
        let slab = in_bf16(seeded_rows(ENTRIES, DIMENSIONS, case_stream_key(0)));
        index.insert(sequential_keys(ENTRIES), &rows(&slab, DIMENSIONS).unwrap()).unwrap();
        let values = in_bf16(seeded_rows(ENTRIES, DIMENSIONS, case_stream_key(1)));
        let stored_values: Vec<bf16> = values.iter().map(|value| bf16::from_f32(*value)).collect();
        fn row_of<Value>(table: &[Value], entry: usize) -> &[Value] {
            &table[entry * DIMENSIONS..][..DIMENSIONS]
        }
        let value_lookup: &ValueLookup<bf16> =
            &|key| u64::try_from(key).ok().map(|entry| row_of(&stored_values, entry as usize));

        let executor = executor();
        let mut arena = Arena::new_in(&executor).unwrap();
        for head_count in [1, 2] {
            let queries = in_bf16(seeded_rows(TOKENS * head_count, DIMENSIONS, case_stream_key(2)));
            let mut carry = vec![DenseSoftmaxState::default(); TOKENS * head_count];
            let outputs = index
                .on(&executor)
                .attend(
                    &rows(&queries, DIMENSIONS).unwrap(),
                    value_lookup,
                    DenseAttendConfig {
                        expansion_factor: 2 * ENTRIES,
                        carry: Some(&mut carry),
                        ..DenseAttendConfig::from(head_count)
                    },
                    &mut arena,
                )
                .unwrap();
            assert_eq!(outputs.view().unwrap().shape(), &[TOKENS * head_count, DIMENSIONS]);
            let attended = flattened(outputs.view().unwrap());
            for query in 0..TOKENS * head_count {
                let logits: Vec<f64> = (0..ENTRIES)
                    .map(|entry| {
                        let products = row_of(&queries, query).iter().zip(row_of(&slab, entry));
                        products.map(|(left, right)| f64::from(left * right)).sum::<f64>() / (DIMENSIONS as f64).sqrt()
                    })
                    .collect();
                let maximum = logits.iter().copied().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = logits.iter().map(|logit| (logit - maximum).exp()).collect();
                let partition = weights.iter().sum::<f64>();
                for column in 0..DIMENSIONS {
                    let mixed = weights
                        .iter()
                        .enumerate()
                        .map(|(entry, weight)| weight * f64::from(row_of(&values, entry)[column]));
                    let expected = mixed.sum::<f64>() / partition;
                    assert_close(f64::from(attended[query * DIMENSIONS + column]), expected, 1e-3);
                }
                assert_close(f64::from(carry[query].maximum_score), maximum, 1e-3);
                assert_close(f64::from(carry[query].partition), partition, 1e-3);
            }
        }

        let refused =
            index.attend(&rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap(), &|_| None::<&[bf16]>, 1, &mut arena);
        assert!(matches!(refused, Err(Error { status: Status::ShapeMismatch, .. })));

        let wide = vec![0.0f64; DIMENSIONS];
        let mistyped =
            index.attend(&rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap(), &|_| Some(wide.as_slice()), 1, &mut arena);
        assert!(
            matches!(mistyped, Err(Error { status: Status::ShapeMismatch, .. })),
            "value rows are read in the index's own quantization"
        );
    }

    /// A product and a softmax both need logits, so a graph frozen under a distance refuses.
    #[test]
    fn linear_algebra_refuses_a_distance_kind() {
        const DIMENSIONS: usize = 4;
        let (index, _, slab) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        assert!(matches!(
            index
                .multiply(&rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap(), 0, &mut Arena::new().unwrap())
                .map(|products| products.queries()),
            Err(Error { status: Status::ConfigMismatch, .. })
        ));
    }

    // endregion: Refinement and Linear Algebra

    // region: Serialization

    /// The loaded index adopts its geometry from the checkpoint's own headers.
    #[test]
    fn saves_and_loads_a_file() {
        const DIMENSIONS: usize = 16;
        let scratch = ScratchDirectory::new("usearch-dense-saves_and_loads_a_file");
        let path = scratch.path().join("index.safegraphs");
        let (index, keys, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        index.on(&executor()).save(&path).unwrap();

        let restored = DenseIndex::load_ram(&path).unwrap().describe().unwrap();
        assert_eq!(restored.metadata.dimensions, DIMENSIONS);
        assert_eq!(restored.size, 32);

        // An NVMe view resolves keys only through the key table it was opened with.
        let storage = Executor::for_nvme(&[] as &[&Path], None, 1).unwrap();
        let keyed = DenseIndex::view_nvme(DenseViewNvmeConfig {
            executor: Some(&storage),
            key_table_domains: Some(&[]),
            ..DenseViewNvmeConfig::from(&path)
        })
        .unwrap();
        assert_eq!(keyed.on(&storage).count(keys[5]).unwrap(), 1);
        let keyless =
            DenseIndex::view_nvme(DenseViewNvmeConfig { executor: Some(&storage), ..DenseViewNvmeConfig::from(&path) })
                .unwrap();
        assert!(matches!(keyless.on(&storage).count(keys[5]), Err(Error { status: Status::Unsupported, .. })));

        // A view serves the same checkpoint in place and refuses to change it, where RAM serves views at all.
        match DenseIndex::view_ram(&path) {
            Ok(viewed) => {
                assert_eq!(viewed.describe().unwrap().size, 32);
                assert!(matches!(viewed.reserve(64), Err(Error { status: Status::ReadOnly, .. })));
            }
            Err(error) => assert_eq!(error.status, Status::Unsupported),
        }
    }

    #[test]
    fn saves_and_loads_a_buffer() {
        const DIMENSIONS: usize = 16;
        let (index, _, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let mut arena = Arena::new().unwrap();
        let saved = index.save_buffer(&mut arena).unwrap();
        let monolith = saved.view();
        assert!(!monolith.is_empty());

        let restored = DenseIndex::load_ram_buffer(monolith).unwrap();
        assert_eq!(restored.describe().unwrap().size, 32);

        let executor = executor();
        let truncated = DenseLoadRamBufferConfig {
            executor: Some(&executor),
            ..DenseLoadRamBufferConfig::from(&monolith[..monolith.len() - 1])
        };
        assert!(DenseIndex::load_ram_buffer(truncated).is_err(), "a short buffer must not be read");
    }

    #[test]
    fn reads_metadata_without_loading() {
        const DIMENSIONS: usize = 16;
        let scratch = ScratchDirectory::new("usearch-dense-reads_metadata_without_loading");
        let path = scratch.path().join("index.safegraphs");
        if empty_index_or_skip(DIMENSIONS, DenseMetric::Euclidean, Scalar::I8).unwrap().is_none() {
            return;
        }
        let (index, _, _) = populated_index(8, DIMENSIONS, DenseMetric::Euclidean, Scalar::I8).unwrap();
        index.save(&path).unwrap();

        let described = DenseIndex::metadata(&path).unwrap();
        assert_eq!(described.dimensions, DIMENSIONS);
        assert_eq!(described.metric, DenseMetric::Euclidean);
        assert_eq!(described.quantization, Scalar::I8);

        let mut arena = Arena::new().unwrap();
        let buffer = index
            .save_buffer(DenseSaveBufferConfig {
                progress: Some(&mut |_, _| ControlFlow::Continue(())),
                ..DenseSaveBufferConfig::from(&mut arena)
            })
            .unwrap();
        assert_eq!(DenseIndex::metadata_buffer(buffer.view()).unwrap().dimensions, DIMENSIONS);
    }

    #[test]
    #[cfg(unix)]
    fn rejects_a_non_utf8_path() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let index = empty_index(8, DenseMetric::Angular, Scalar::BF16).unwrap();
        let path = std::path::Path::new(OsStr::from_bytes(&[0xff, 0xfe]));
        assert!(
            matches!(index.save(path), Err(Error { status: Status::Io, .. })),
            "the C interface exchanges UTF-8 bytes"
        );
    }

    // endregion: Serialization

    // region: Resharding and Iteration

    /// Several sources gather into the one target among them, the others staying the caller's.
    #[test]
    fn gathers_several_sources_into_one_target() {
        const DIMENSIONS: usize = 16;
        let (index, keys, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let shard = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let shard_keys: Vec<Key> = (100..132u64).map(Key::from).collect();
        let shard_slab = seeded_rows(32, DIMENSIONS, case_stream_key(1));
        shard.insert(&shard_keys, &rows(&shard_slab, DIMENSIONS).unwrap()).unwrap();

        index
            .on(&executor())
            .reshard(DenseReshardConfig {
                sources: &[&index, &shard],
                targets: &[&index],
                ..DenseReshardConfig::default()
            })
            .unwrap();
        assert_eq!(index.describe().unwrap().size, 64);
        assert_eq!(shard.describe().unwrap().size, 0);
        assert!(index.contains(keys[0]).unwrap() && index.contains(shard_keys[0]).unwrap());
    }

    /// A reshard keeps every entry; a lone monolith has no manifest a loaded handle could replace.
    #[test]
    fn reshards_in_place_and_loads_only_manifests() {
        const DIMENSIONS: usize = 16;
        let scratch = ScratchDirectory::new("usearch-dense-reshards_in_place_and_loads_only_manifests");
        let path = scratch.path().join("index.safegraphs");
        let (index, _, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let executor = executor();
        index
            .on(&executor)
            .reshard(DenseReshardConfig {
                sources: &[&index],
                targets: &[&index],
                memory_domains: Some(&[]),
                shard_capacity: 64,
                ..DenseReshardConfig::default()
            })
            .unwrap();
        assert_eq!(index.describe().unwrap().size, 32);

        index.on(&executor).save(&path).unwrap();
        let storage = Executor::for_nvme(&[] as &[&Path], None, 1).unwrap();
        let loaded =
            DenseIndex::load_nvme(DenseViewNvmeConfig { executor: Some(&storage), ..DenseViewNvmeConfig::from(&path) });
        assert!(matches!(loaded, Err(Error { status: Status::Unsupported, .. })));
    }

    #[test]
    fn walks_entries_in_batches() {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 20;
        const BUFFERED: usize = 8;
        let (index, keys, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let executor = executor();
        let receiver = index.on(&executor);
        let mut cursor = receiver.cursor(&[]).unwrap();
        let mut arena = Arena::new_in(&executor).unwrap();
        let mut seen = HashSet::new();
        let mut batches = Vec::new();
        loop {
            let batch = cursor.next_keys(BUFFERED, &mut arena).unwrap();
            if batch.written() == 0 {
                break;
            }
            batches.push(batch.written());
            seen.extend(batch.keys().iter().copied());
        }

        assert_eq!(batches, vec![8, 8, 4], "the batch boundary is crossed twice");
        assert_eq!(seen, keys.iter().copied().collect::<HashSet<Key>>());
    }

    #[test]
    fn walks_entries_by_keys() {
        const DIMENSIONS: usize = 8;
        let (index, keys, _) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let mut cursor = index.cursor(&[]).unwrap();
        let mut arena = Arena::new().unwrap();
        let batch = cursor.next_keys(8, &mut arena).unwrap();
        assert_eq!(batch.written(), 4);
        assert_eq!(
            batch.keys().iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
        assert_eq!(cursor.next_keys(8, &mut arena).unwrap().written(), 0, "the walk is exhausted");
    }

    #[test]
    fn streams_matches_in_pulls() {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 200;
        const PULL: usize = 64;
        let (index, _, slab) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let query = rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap();
        let mut arena = Arena::new().unwrap();
        let mut stream = match index.search_stream(query) {
            Err(error) if error.status == Status::Unsupported => return skipped("needs search streams"),
            opened => opened.unwrap(),
        };
        let first = stream.next(5, &mut arena).unwrap();
        assert_eq!(first.queries(), 1, "a pull is one query of a search");
        assert_eq!(first.query(0).unwrap().keys.len(), 5, "a pull hands the caller as many as it asked for");

        let mut drained = 0;
        loop {
            let pulled = stream.next(PULL, &mut arena).unwrap().query(0).unwrap().keys.len();
            if pulled == 0 {
                break;
            }
            drained += pulled;
        }
        assert!(drained > PULL, "the walk spans several pulls");
        assert!(
            stream.next(PULL, &mut arena).unwrap().query(0).unwrap().keys.is_empty(),
            "a drained stream stays drained"
        );

        let mut widened = index
            .search_stream(DenseSearchStreamInitConfig {
                expansion_factor: 64,
                ..DenseSearchStreamInitConfig::from(query)
            })
            .unwrap();
        assert!(!widened.next(PULL, &mut arena).unwrap().query(0).unwrap().keys.is_empty());

        assert!(
            matches!(
                index.search_stream(rows(&slab[..2 * DIMENSIONS], DIMENSIONS).unwrap()),
                Err(Error { status: Status::ShapeMismatch, .. })
            ),
            "a stream walks one row, never the first of many"
        );
    }

    // endregion: Resharding and Iteration

    // region: Progress and Cancellation

    #[test]
    fn a_reported_batch_counts_up_to_the_rows_it_was_given() {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        index.reserve(COUNT).unwrap();
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, case_stream_key(0));

        let mut watch = Watch::new(Patience::Complete);
        index
            .on(&executor())
            .insert(
                &keys,
                DenseInsertConfig {
                    progress: Some(&mut |finished, total| watch.saw(finished, total)),
                    ..DenseInsertConfig::from(&rows(&slab, DIMENSIONS).unwrap())
                },
            )
            .unwrap();

        assert_eq!(index.describe().unwrap().size, COUNT);
        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (COUNT, COUNT), "the last report names the whole batch");
    }

    /// Every entry is claimed before the first report, so a stop is ignored and the batch publishes
    /// whole.
    #[test]
    fn a_stopped_batch_publishes_everything() {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        index.reserve(COUNT).unwrap();
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, case_stream_key(0));

        let mut watch = Watch::new(Patience::StopAtFirstReport);
        index
            .insert(
                &keys,
                DenseInsertConfig {
                    progress: Some(&mut |finished, total| watch.saw(finished, total)),
                    ..DenseInsertConfig::from(&rows(&slab, DIMENSIONS).unwrap())
                },
            )
            .unwrap();
        assert_eq!(index.describe().unwrap().size, COUNT);
    }

    #[test]
    fn a_reported_search_counts_up_to_the_queries_it_was_given() {
        const DIMENSIONS: usize = 8;
        const QUERIES: usize = 4;
        const WANTED: usize = 3;
        let (index, _, slab) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let mut watch = Watch::new(Patience::Complete);
        index
            .on(&executor())
            .search(
                &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS).unwrap(),
                DenseSearchConfig {
                    progress: Some(&mut |finished, total| watch.saw(finished, total)),
                    ..DenseSearchConfig::from(WANTED)
                },
                &mut Arena::new().unwrap(),
            )
            .unwrap();

        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (QUERIES, QUERIES));
    }

    /// Stopping a sweep is its time budget: the call still re-links its islands, so the graph it
    /// leaves stays whole, and answers `Cancelled`.
    #[test]
    fn a_stopped_refine_still_heals() {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 32;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let mut watch = Watch::new(Patience::StopAtFirstReport);
        let refined = index.on(&executor()).refine(DenseRefineConfig {
            progress: Some(&mut |finished, total| watch.saw(finished, total)),
            ..DenseRefineConfig::default()
        });

        match refined {
            Ok(report) => assert!(report.changed <= report.reselected, "an unstopped sweep reports it"),
            Err(failure) => assert_eq!(failure.status, Status::Cancelled),
        }
        assert!(watch.ordered, "a report never counts down");
        assert!(watch.last.1 != 0, "the handler saw the sweep start");
        assert_eq!(index.on(&executor()).health().unwrap().audited, COUNT);
    }

    #[test]
    fn a_reported_checkpoint_counts_what_it_wrote_and_placed() {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 32;
        let scratch = ScratchDirectory::new("usearch-dense-a_reported_checkpoint");
        let path = scratch.path().join("index.safegraphs");
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();

        let mut writing = Watch::new(Patience::Complete);
        index
            .on(&executor())
            .save(DenseSaveConfig {
                progress: Some(&mut |finished, total| writing.saw(finished, total)),
                ..DenseSaveConfig::from(&path)
            })
            .unwrap();
        assert!(writing.ordered, "a report never counts down");
        assert_eq!(writing.last.0, writing.last.1, "the last report names a finished write");
        assert!(writing.last.1 != 0);

        let mut placing = Watch::new(Patience::Complete);
        let restored = DenseIndex::load_ram(DenseLoadRamConfig {
            executor: Some(&executor()),
            progress: Some(&mut |finished, total| placing.saw(finished, total)),
            ..DenseLoadRamConfig::from(&path)
        })
        .unwrap();
        assert!(placing.ordered, "a report never counts down");
        assert_eq!(placing.last.0, placing.last.1, "the last report names a finished read");
        assert_eq!(restored.describe().unwrap().size, COUNT);
    }

    /// The handler reports the closing rounds, never the entries moved.
    #[test]
    fn a_reported_reshard_counts_its_closing_rounds() {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 16;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let first = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap().0;
        let second = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap().0;

        let mut watch = Watch::new(Patience::Complete);
        let report = index
            .on(&executor())
            .reshard(DenseReshardConfig {
                sources: &[&index, &first, &second],
                targets: &[&index],
                progress: Some(&mut |finished, total| watch.saw(finished, total)),
                ..DenseReshardConfig::default()
            })
            .unwrap();
        assert_eq!(watch.last.0, watch.last.1, "the last report names a finished phase");
        assert!(report.changed <= report.reselected);
        assert_eq!(index.describe().unwrap().size, COUNT * 3);
    }

    /// A stop cuts only the closing rounds, so every entry has moved; the plan it leaves live
    /// continues where it stopped, and its finish closes it.
    #[test]
    fn a_stopped_reshard_continues_through_its_plan() {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 16;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let first = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap().0;
        let second = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap().0;

        let executor = executor();
        let (mut plan, mut phase) = (DenseReshardPlan::default(), PhaseProgress::default());
        let mut watch = Watch::new(Patience::StopAtFirstReport);
        let stopped = index.on(&executor).reshard(DenseReshardConfig {
            sources: &[&index, &first, &second],
            targets: &[&index],
            progress: Some(&mut |finished, total| watch.saw(finished, total)),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..DenseReshardConfig::default()
        });
        if let Err(failure) = stopped {
            assert_eq!(failure.status, Status::Cancelled);
            assert_ne!(phase.phase, Phase::Done, "a stop names the phase it stopped in");
        }
        assert_eq!(index.describe().unwrap().size, COUNT * 3);

        // A deferred call owns the plan it continues and hands it back once final.
        let continued = index.deferred(&executor).reshard(DenseReshardConfig::from(plan)).unwrap();
        let (finished, (config, _)) = continued.wait_with_inputs();
        finished.unwrap();
        let plan = config.plan.expect("the call hands its plan back");
        let report = index.deferred(&executor).reshard_plan_finish(plan).unwrap().wait().unwrap();
        assert!(report.changed <= report.reselected);
    }

    /// A plan carries a refine across calls: a stop leaves it live, the next call ends the round,
    /// and the finish reports the whole plan.
    #[test]
    fn a_refine_plan_resumes_and_finishes() {
        const COUNT: usize = 32;
        let (index, _, _) = populated_index(COUNT, 8, DenseMetric::Angular, Scalar::BF16).unwrap();
        let executor = executor();
        let (mut plan, mut phase) = (DenseRefinePlan::default(), PhaseProgress::default());
        let mut watch = Watch::new(Patience::StopAtFirstReport);
        let stopped = index.on(&executor).refine(DenseRefineConfig {
            progress: Some(&mut |finished, total| watch.saw(finished, total)),
            plan: Some(&mut plan),
            phase: Some(&mut phase),
            ..DenseRefineConfig::default()
        });
        if let Err(failure) = stopped {
            assert_eq!(failure.status, Status::Cancelled);
            assert_ne!(phase.phase, Phase::Done);
            index
                .on(&executor)
                .refine(DenseRefineConfig {
                    plan: Some(&mut plan),
                    phase: Some(&mut phase),
                    ..DenseRefineConfig::default()
                })
                .unwrap();
        }
        assert_eq!(phase.phase, Phase::Done);
        let report = index.on(&executor).refine_plan_finish(plan).unwrap();
        assert_eq!(report.reachability.audited, COUNT);
    }

    // endregion: Progress and Cancellation

    // region: Concurrency

    const THREADS: usize = 4;

    /// The batch is the unit of parallelism: the executor fans one call out over its threads,
    /// so the caller never spawns any of its own.
    #[test]
    fn builds_and_searches_on_a_shared_executor() {
        const DIMENSIONS: usize = 128;
        const COUNT: usize = 1000;
        const WANTED: usize = 10;

        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        index.reserve(COUNT).unwrap();
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, case_stream_key(0));

        let executor = Executor::for_ram(&[], None, THREADS).unwrap();
        let receiver = index.on(&executor);
        let vectors = rows(&slab, DIMENSIONS).unwrap();
        receiver.insert(&keys, &vectors).unwrap();
        assert_eq!(receiver.describe().unwrap().size, COUNT);

        let config = DenseSearchConfig { expansion_factor: 64, ..DenseSearchConfig::from(WANTED) };
        let mut arena = Arena::new_in(&executor).unwrap();
        let matches = receiver.search(&vectors, config, &mut arena).unwrap();
        assert!(matches.iter().all(|query| query.keys.len() == WANTED), "every query fills its own output");
    }

    /// The index itself is the blocking receiver on the calling thread, with the same results.
    #[test]
    fn runs_on_the_calling_thread() {
        const DIMENSIONS: usize = 8;
        let (index, _, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let query = rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap();
        let (mut first, mut second) = (Arena::new().unwrap(), Arena::new().unwrap());
        let on_this_thread = index.search(&query, 4, &mut first).unwrap();
        let on_the_null_executor = index.on(&Executor::default()).search(&query, 4, &mut second).unwrap();
        assert_eq!(on_this_thread.query(0).unwrap().keys.len(), 4);
        assert_eq!(on_this_thread.query(0), on_the_null_executor.query(0));
    }

    #[test]
    fn searches_while_inserting() {
        const DIMENSIONS: usize = 64;
        const HALF: usize = 500;
        const WANTED: usize = 4;

        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        index.reserve(2 * HALF).unwrap();
        let keys = sequential_keys(2 * HALF);
        let slab = seeded_rows(2 * HALF, DIMENSIONS, case_stream_key(0));

        // The first half lands serially, so the reader below traverses a populated graph.
        index.insert(&keys[..HALF], &rows(&slab[..HALF * DIMENSIONS], DIMENSIONS).unwrap()).unwrap();

        // One call holds an executor at a time, so genuinely overlapping calls need one apiece -
        // the writer's and the reader's, on two threads sharing the handle.
        let writer = Executor::for_ram(&[], None, THREADS).unwrap();
        let reader = Executor::for_ram(&[], None, THREADS).unwrap();
        let known: HashSet<Key> = keys.iter().copied().collect();
        let strangers = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            scope.spawn(|| {
                let writer = writer;
                let vectors = rows(&slab[HALF * DIMENSIONS..], DIMENSIONS).unwrap();
                index.on(&writer).insert(&keys[HALF..], &vectors).expect("the racing batch must insert");
            });
            scope.spawn(|| {
                let reader = reader;
                let mut arena = Arena::new_in(&reader).expect("the arena must initialize");
                let queries = rows(&slab[..HALF * DIMENSIONS], DIMENSIONS).unwrap();
                let config = DenseSearchConfig { expansion_factor: 32, ..DenseSearchConfig::from(WANTED) };
                let matches =
                    index.on(&reader).search(&queries, config, &mut arena).expect("the racing search must resolve");
                // Whether a racing insert is visible is what the watermark leaves unspecified;
                // that no key outside the dataset appears is what it guarantees.
                for query in matches.iter() {
                    if query.keys.iter().any(|key| !known.contains(key)) {
                        strangers.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        });

        assert_eq!(strangers.load(Ordering::Relaxed), 0, "searches never see unpublished entries");
        assert_eq!(index.describe().unwrap().size, 2 * HALF);
    }

    // endregion: Concurrency

    // region: Deferred Calls

    /// A deferred search owns its queries, borrows its arena as the blocking one does, and answers
    /// what the blocking one does once waited on.
    #[test]
    fn a_deferred_search_matches_the_blocking_one() {
        const DIMENSIONS: usize = 16;
        const QUERIES: usize = 4;
        const WANTED: usize = 5;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let queries = Matrix::<f32>::from_slice(&slab[..QUERIES * DIMENSIONS], &[QUERIES, DIMENSIONS]).unwrap();

        let executor = executor();
        let mut arena = Arena::new_in(&executor).unwrap();
        let call = index.deferred(&executor).search(queries, WANTED, &mut arena).unwrap();
        requires_send::<Deferred<'static, (Matrix<f32>, DenseSearchConfig<'static>), Matches<'static>>>();
        requires_send::<Matches<'static>>();
        assert!(matches!(call.status(), Err(Status::Deferred) | Ok(())));
        let deferred = call.wait().unwrap();

        let mut blocking_arena = Arena::new().unwrap();
        let blocking = index
            .search(&rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS).unwrap(), WANTED, &mut blocking_arena)
            .unwrap();
        assert_eq!(deferred.queries(), QUERIES);
        for query in 0..QUERIES {
            assert_eq!(deferred.query(query), blocking.query(query));
        }

        let queries = Matrix::<f32>::from_slice(&slab[..DIMENSIONS], &[1, DIMENSIONS]).unwrap();
        let again = index.deferred(&executor).search(queries, WANTED, &mut arena).unwrap();
        assert_eq!(again.wait().unwrap().query(0), blocking.query(0));
    }

    /// A leaked deferred call leaves its arena busy rather than unsafe: a call naming it finishes
    /// contended, and dropping it waits for the leaked call.
    #[test]
    fn a_leaked_deferred_call_keeps_its_arena_busy() {
        use std::sync::atomic::AtomicBool;

        static ENTERED: AtomicBool = AtomicBool::new(false);
        static RELEASED: AtomicBool = AtomicBool::new(false);
        const DIMENSIONS: usize = 8;
        let (index, keys, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let queries = Matrix::<f32>::from_slice(&slab[..DIMENSIONS], &[1, DIMENSIONS]).unwrap();
        // The handler holds the call until released, so it is still running when the arena is named again.
        let gate: &'static mut ProgressCallback<'static> = Box::leak(Box::new(|_, _| {
            ENTERED.store(true, Ordering::Release);
            while !RELEASED.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            ControlFlow::Continue(())
        }));

        let executor = executor();
        let mut arena = Arena::new_in(&executor).unwrap();
        let config = DenseSearchConfig { progress: Some(gate), ..DenseSearchConfig::from(3) };
        core::mem::forget(index.deferred(&executor).search(queries, config, &mut arena).unwrap());
        while !ENTERED.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        assert!(matches!(index.get::<f32>(keys[0], &mut arena), Err(Error { status: Status::Contended, .. })));
        RELEASED.store(true, Ordering::Release);
        drop(arena);
    }

    /// A call meeting a running one on its executor finishes contended without running, and the
    /// executor serves the next call once the first is final.
    #[test]
    fn a_running_call_holds_its_executor() {
        use std::sync::atomic::AtomicBool;

        static ENTERED: AtomicBool = AtomicBool::new(false);
        static RELEASED: AtomicBool = AtomicBool::new(false);
        const DIMENSIONS: usize = 8;
        let (index, keys, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let queries = Matrix::<f32>::from_slice(&slab[..DIMENSIONS], &[1, DIMENSIONS]).unwrap();
        // A deferred call borrows its handler for as long as it may run, so the gate is leaked.
        let gate: &'static mut ProgressCallback<'static> = Box::leak(Box::new(|_, _| {
            ENTERED.store(true, Ordering::Release);
            while !RELEASED.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            ControlFlow::Continue(())
        }));

        let executor = executor();
        let mut arena = Arena::new_in(&executor).unwrap();
        let config = DenseSearchConfig { progress: Some(gate), ..DenseSearchConfig::from(3) };
        let held = index.deferred(&executor).search(queries, config, &mut arena).unwrap();
        while !ENTERED.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        assert!(matches!(index.on(&executor).count(keys[0]), Err(Error { status: Status::Contended, .. })));
        RELEASED.store(true, Ordering::Release);
        assert_eq!(held.wait().unwrap().query(0).unwrap().keys.len(), 3);
        assert_eq!(index.on(&executor).count(keys[0]).unwrap(), 1);
    }

    /// A drain leaves idle threads in the mood it names, and the next call runs in either.
    #[test]
    fn a_drain_sets_the_mood() {
        let (index, keys, _) = populated_index(4, 4, DenseMetric::Angular, Scalar::BF16).unwrap();
        let executor = executor();
        for mood in [Mood::Spinning, Mood::Parked] {
            executor.drain(mood).unwrap();
            assert_eq!(index.on(&executor).count(keys[0]).unwrap(), 1);
        }
    }

    /// Polling an awaited deferred call alone finishes it, since pool thread 0 runs it in the
    /// caller's place, and the answer is what `wait` would give.
    #[test]
    fn an_awaited_call_finishes_by_polling() {
        use core::future::{Future, IntoFuture};
        use core::pin::pin;
        use core::task::{Context, Poll, Waker};

        const DIMENSIONS: usize = 16;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let queries = Matrix::<f32>::from_slice(&slab[..DIMENSIONS], &[1, DIMENSIONS]).unwrap();
        let executor = executor();
        let mut arena = Arena::new_in(&executor).unwrap();
        let mut awaited = pin!(index.deferred(&executor).search(queries, 3, &mut arena).unwrap().into_future());
        let matches = block_on(awaited.as_mut());
        assert_eq!(
            matches.unwrap().query(0),
            index
                .search(&rows(&slab[..DIMENSIONS], DIMENSIONS).unwrap(), 3, &mut Arena::new().unwrap())
                .unwrap()
                .query(0)
        );
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(awaited.as_mut().poll(&mut context), Poll::Ready(Err(_))));
    }

    /// Deferred calls running together each borrow an arena, which their results borrow in turn.
    #[test]
    fn deferred_calls_borrow_their_arenas() {
        const DIMENSIONS: usize = 4;
        let (index, keys, slab) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        // One call holds an executor at a time, so calls running together take one apiece.
        let (finder, saver, counter) = (executor(), executor(), executor());

        let (mut first, mut second) = (Arena::new_in(&finder).unwrap(), Arena::new_in(&saver).unwrap());
        let found = index.deferred(&finder).get::<f32>(keys[2], &mut first).unwrap();
        let buffer = index.deferred(&saver).save_buffer(&mut second).unwrap();
        let counted = index.deferred(&counter).count(keys[3]).unwrap();
        assert_close_rows(
            &flattened(found.wait().unwrap().view().unwrap()),
            &slab[DIMENSIONS * 2..DIMENSIONS * 3],
            BF16_EPSILON,
        );
        assert_eq!(DenseIndex::metadata_buffer(buffer.wait().unwrap().view()).unwrap().dimensions, DIMENSIONS);
        assert_eq!(counted.wait().unwrap(), 1);
    }

    /// Dropping a call that is not yet final waits for it, then frees what it owned.
    #[test]
    fn an_unfinished_deferred_call_drops_by_waiting() {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16).unwrap();
        let slab = seeded_rows(COUNT, DIMENSIONS, case_stream_key(0));
        let vectors = Matrix::<f32>::from_slice(&slab, &[COUNT, DIMENSIONS]).unwrap();

        let executor = executor();
        let call = index.deferred(&executor).insert(sequential_keys(COUNT), vectors).unwrap();
        drop(call);
        assert_eq!(index.describe().unwrap().size, COUNT);
    }

    // endregion: Deferred Calls
}

// endregion: Tests
