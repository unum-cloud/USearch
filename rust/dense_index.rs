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

use core::ffi::{c_char, c_int, c_void};
use core::fmt;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::Deref;
use core::ptr::{self, NonNull};
use core::slice;
use core::str::FromStr;
use std::path::Path;

use numkong::MatrixView;

use crate::types::{
    arena_slice, arrow_checked, check, fail, lower_predicate, lower_progress, nvme_options, path_view, path_views,
    ram_options, read_spelling, spelling, static_string, usearch_arena_t, usearch_bytes_view_t,
    usearch_completion_blocking_k, usearch_completion_t, usearch_cpu_class_name, usearch_cpu_class_t,
    usearch_executor_t, usearch_gpu_class_name, usearch_gpu_class_t, usearch_key_t, usearch_keys_tape_view_t,
    usearch_keys_view_t, usearch_matches_tape_view_t, usearch_matches_tape_view_to_arrow, usearch_matches_view_t,
    usearch_nvme_options_t, usearch_outcome_t, usearch_predicate_t, usearch_progress_t, usearch_ram_options_t,
    usearch_rows_view_t, usearch_rows_view_to_arrow, usearch_scalar_t, usearch_traversal_t, usearch_vram_options_t,
    values_of, vram_options, Accepts, Arena, ArrowDeviceArray, ArrowExport, ArrowSchema, Blocking, CPUClass, Callbacks,
    Calling, Completion, Deferring, Element, Error, Executor, Exported, GPUClass, Handles, Key, Matches, On, Predicate,
    ProgressCallback, QueryMatches, Residency, Result, RowsView, Scalar, Status, Traversal,
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
pub struct usearch_dense_absorb_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub donors: *mut usearch_dense_index_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_dense_split_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub receivers: *const usearch_dense_index_t,
    pub report: *mut usearch_refine_report_t,
    pub refine: usearch_refine_options_t,
    pub seed: u64,
    pub progress: usearch_progress_t,
    pub stats: *mut usearch_traversal_stats_t,
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

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct usearch_dense_metadata_t {
    pub metric: usearch_dense_metric_t,
    pub quantization: usearch_scalar_t,
    pub dimensions: usize,
    pub connectivity: usize,
}

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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_dense_index_description_t {
    pub metadata: usearch_dense_metadata_t,
    pub size: usize,
    pub capacity: usize,
    pub memory_usage: usize,
    pub cpu_class: usearch_cpu_class_t,
    pub gpu_class: usearch_gpu_class_t,
    pub shards_count: usize,
}

impl Default for usearch_dense_index_description_t {
    fn default() -> Self {
        usearch_dense_index_description_t {
            metadata: usearch_dense_metadata_t::default(),
            size: 0,
            capacity: 0,
            memory_usage: 0,
            cpu_class: 0,
            gpu_class: 0,
            shards_count: 0,
        }
    }
}

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

extern "C" {
    pub fn usearch_dense_metric_name(metric: usearch_dense_metric_t) -> *const c_char;
    pub fn usearch_dense_metric_from_name(name: usearch_bytes_view_t) -> usearch_dense_metric_t;

    pub fn usearch_dense_init_ram(arguments: *mut usearch_dense_init_ram_t);
    pub fn usearch_dense_init_vram(arguments: *mut usearch_dense_init_vram_t);
    pub fn usearch_dense_load_ram(arguments: *mut usearch_dense_load_ram_t);
    pub fn usearch_dense_load_vram(arguments: *mut usearch_dense_load_vram_t);
    pub fn usearch_dense_load_ram_buffer(arguments: *mut usearch_dense_load_ram_buffer_t);
    pub fn usearch_dense_load_vram_buffer(arguments: *mut usearch_dense_load_vram_buffer_t);
    pub fn usearch_dense_view_ram(arguments: *mut usearch_dense_view_ram_t);
    pub fn usearch_dense_view_nvme(arguments: *mut usearch_dense_view_nvme_t);
    pub fn usearch_dense_free(index: usearch_dense_index_t);

    pub fn usearch_dense_reserve(arguments: *mut usearch_dense_reserve_t);
    pub fn usearch_dense_clear(arguments: *mut usearch_dense_clear_t);
    pub fn usearch_dense_insert(arguments: *mut usearch_dense_insert_t);
    pub fn usearch_dense_refine(arguments: *mut usearch_dense_refine_t);
    pub fn usearch_dense_health(arguments: *mut usearch_dense_health_t);
    pub fn usearch_dense_search(arguments: *mut usearch_dense_search_t);
    pub fn usearch_dense_count(arguments: *mut usearch_dense_count_t);
    pub fn usearch_dense_get(arguments: *mut usearch_dense_get_t);
    pub fn usearch_dense_absorb(arguments: *mut usearch_dense_absorb_t);
    pub fn usearch_dense_split(arguments: *mut usearch_dense_split_t);
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
        read_spelling(
            text,
            usearch_dense_metric_from_name,
            usearch_dense_metric_name,
            DenseMetric::from_raw,
        )
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
    /// The class a host index dispatches as; `None` dispatches as the machine.
    pub cpu_class: Option<CPUClass>,
    /// The class a device index dispatches as; `None` dispatches as the devices.
    pub gpu_class: Option<GPUClass>,
}

impl Default for DenseOptions {
    fn default() -> Self {
        DenseOptions {
            dimensions: 0,
            metric: DenseMetric::Angular,
            quantization: Scalar::BF16,
            connectivity: 0,
            values_per_key: 0,
            cpu_class: None,
            gpu_class: None,
        }
    }
}

/// The geometry and metric a checkpoint froze, readable without loading it.
///
/// `values_per_key` and the placement are the reader's to choose when loading, so a checkpoint
/// fixes neither.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct DenseMetadata {
    /// Scalars per stored vector.
    pub dimensions: usize,
    /// The metric kind the checkpoint scores with.
    pub metric: DenseMetric,
    /// The scalar type its vectors are stored and quantized into.
    pub quantization: Scalar,
    /// Connections per node the graph was built with.
    pub connectivity: usize,
}

impl DenseMetadata {
    /// Lifts the raw structure, refusing a kind this binding has no name for.
    fn from_raw(raw: &usearch_dense_metadata_t) -> Result<Self> {
        Ok(DenseMetadata {
            dimensions: raw.dimensions,
            metric: DenseMetric::from_raw(raw.metric)
                .ok_or(fail(Status::UnknownKind, "the checkpoint names an unknown metric"))?,
            quantization: Scalar::from_raw(raw.quantization)
                .ok_or(fail(Status::UnknownKind, "the checkpoint names an unknown scalar kind"))?,
            connectivity: raw.connectivity,
        })
    }
}

/// A live index's frozen metadata and current counters, each read once, so under concurrent
/// insertions they may come from slightly different moments.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
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
    /// The CPU class dispatch picked on this machine, as [`CPUClass`] spells it.
    pub cpu_class: &'static str,
    /// The GPU class dispatch picked on the index's devices; "unknown" for a host index.
    pub gpu_class: &'static str,
    /// Live shards.
    pub shards_count: usize,
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
    /// Levels the sweep took to converge; the deepest shard's on a sharded handle.
    pub sweeps: usize,
    /// Levels the sweep back toward the entry point took, where an audit sweeps both ways.
    pub back_sweeps: usize,
}

/// What one refine-family call did to the graph: [`DenseVerbs::refine`], [`DenseVerbs::absorb`]
/// and [`DenseVerbs::split`] land it.
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
    /// The audit closing the heal, as a fresh [`DenseVerbs::health`] right after would take it.
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
/// call, and `None` refuses the call, as a panic does before it resurfaces once the call is final.
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
    /// The class the index dispatches as; `None` dispatches as the machine.
    pub cpu_class: Option<CPUClass>,
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
            cpu_class: None,
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
    /// The class the index dispatches as; `None` dispatches as the devices.
    pub gpu_class: Option<GPUClass>,
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
            gpu_class: None,
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
    /// The class the index dispatches as; `None` dispatches as the machine.
    pub cpu_class: Option<CPUClass>,
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
            cpu_class: None,
            values_per_key: 0,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for DenseLoadRamBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self { DenseLoadRamBufferConfig::from(&**bytes) }
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
    /// The class the index dispatches as; `None` dispatches as the devices.
    pub gpu_class: Option<GPUClass>,
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
            gpu_class: None,
            values_per_key: 0,
            progress: None,
            progress_steps: 0,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for DenseLoadVramBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self { DenseLoadVramBufferConfig::from(&**bytes) }
}

/// A checkpoint path and the drives serving it in place; see [`DenseIndex::view_nvme`].
pub struct DenseViewNvmeConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The NVMe executor whose rings read the checkpoint's headers; `None` reads on the calling
    /// thread.
    pub executor: Option<&'config Executor>,
    /// Devices or directories to serve the parts from; empty serves each where its manifest points.
    pub paths: &'config [&'config Path],
    /// The class the index dispatches as; `None` dispatches as the machine.
    pub cpu_class: Option<CPUClass>,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for DenseViewNvmeConfig<'config> {
    fn from(path: &'config Location) -> Self {
        DenseViewNvmeConfig {
            path: path.as_ref(),
            executor: None,
            paths: &[],
            cpu_class: None,
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
        DenseInsertConfig {
            vectors,
            expansion_factor: 0,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
    }
}

/// The knobs of [`DenseVerbs::refine`].
#[derive(Default)]
pub struct DenseRefineConfig<'config> {
    /// The breadth of the candidates the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Reports the nodes swept and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

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
    /// collapse at low selectivity. A VRAM handle refuses one with [`Status::Device`].
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
        DenseMaxsimConfig {
            wanted,
            expansion_factor: 0,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
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
        DenseMultiplyConfig {
            wanted,
            expansion_factor: 0,
            progress: None,
            progress_steps: 0,
            stats: None,
        }
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

/// The donors an absorb consumes and the handler counting them; see [`DenseVerbs::absorb`].
pub struct DenseAbsorbConfig<'config, Donors> {
    /// The indexes absorbed, consumed whether or not the absorb succeeds.
    pub donors: Donors,
    /// Reports the nodes the closing refinement re-selected and stops it wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
}

/// The receivers a split fills and the knobs of its closing refinements; see [`DenseVerbs::split`].
pub struct DenseSplitConfig<'config> {
    /// At least two empty indexes of the source's residency, scalar, class, metric, geometry and
    /// shard count, each receiving one part.
    pub receivers: &'config [&'config DenseIndex],
    /// The breadth of the closing refinements' walks; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Picks the part roots: equal seeds over an unchanged index split equally.
    pub seed: u64,
    /// Reports the nodes the closing refinements re-selected and stops them wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    pub progress_steps: usize,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl<'config> From<&'config [&'config DenseIndex]> for DenseSplitConfig<'config> {
    fn from(receivers: &'config [&'config DenseIndex]) -> Self {
        DenseSplitConfig {
            receivers,
            expansion_factor: 0,
            seed: 0,
            progress: None,
            progress_steps: 0,
            stats: None,
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

impl<Donors: IntoIterator<Item = DenseIndex>> From<Donors> for DenseAbsorbConfig<'_, Donors> {
    fn from(donors: Donors) -> Self {
        DenseAbsorbConfig {
            donors,
            progress: None,
            progress_steps: 0,
        }
    }
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
        DenseSaveConfig {
            path,
            progress: None,
            progress_steps: 0,
        }
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
        DenseSaveBufferConfig {
            arena,
            progress: None,
            progress_steps: 0,
        }
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
        DenseSearchStreamInitConfig {
            query,
            expansion_factor: 0,
        }
    }
}

impl<Rows> Accepts<DenseInsertConfig<'static, Rows>> for Deferring where Deferring: Accepts<Rows> {}
impl Accepts<DenseRefineConfig<'static>> for Deferring {}
impl Accepts<DenseSearchConfig<'static>> for Deferring {}
impl Accepts<DenseMaxsimConfig<'static>> for Deferring {}
impl Accepts<DenseMultiplyConfig<'static>> for Deferring {}
impl Accepts<DenseAttendConfig<'static>> for Deferring {}
impl<Value> Accepts<&'static ValueLookup<'static, Value>> for Deferring {}
impl<Donors> Accepts<DenseAbsorbConfig<'static, Donors>> for Deferring {}
impl Accepts<DenseSplitConfig<'static>> for Deferring {}
impl Accepts<DenseSampleConfig<'static>> for Deferring {}
impl<Location> Accepts<DenseSaveConfig<'static, Location>> for Deferring where Deferring: Accepts<Location> {}
impl Accepts<DenseSaveBufferConfig<'static, '_>> for Deferring {}

// endregion: Configs

// region: Results

/// Borrows `count` rows a finished call exported into `arena`, refused over device memory; an
/// absent key leaves a null base, which a zero-row view never reads.
fn exported_rows<'arena, Value: Element>(
    arena: &'arena Arena,
    rows: &usearch_rows_view_t,
    count: usize,
) -> Result<MatrixView<'arena, Value>> {
    arena.host()?;
    let data = NonNull::new(rows.data.cast::<Value>().cast_mut()).unwrap_or(NonNull::dangling());
    let element = core::mem::size_of::<Value>() as isize;
    unsafe { MatrixView::from_raw_parts(data.as_ptr(), &[count, rows.columns], &[rows.stride as isize, element]) }
        .map_err(|_| fail(Status::Internal, "the library returned an untraversable slab"))
}

/// The rows [`DenseVerbs::get`] or [`DenseVerbs::attend`] exported into the arena it borrows.
///
/// Rust-only.
pub struct DenseVectors<'arena, Value> {
    arena: &'arena Arena,
    vectors: Exported<usearch_rows_view_t>,
    count: usize,
    value: PhantomData<Value>,
}

impl<'arena, Value: Element> DenseVectors<'arena, Value> {
    /// One row per entry under the key in insertion order, or per query, at the pitch the library
    /// reported; refused with [`Status::Device`] over device memory.
    pub fn view(&self) -> Result<MatrixView<'arena, Value>> { exported_rows(self.arena, &self.vectors.0, self.count) }

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
        let status = unsafe {
            usearch_rows_view_to_arrow(self.vectors.0, self.count, self.arena.handle, &mut schema, &mut array)
        };
        arrow_checked(status)?;
        Ok(ArrowExport {
            arena: self.arena,
            schema,
            array,
        })
    }
}

/// The ragged run [`DenseVerbs::multiply`] or [`DenseVerbs::maxsim`] exported into the arena it
/// borrows: per left row, every pair its walk scored, in traversal order, or per query, its kept
/// keys, largest score first.
pub struct DenseProducts<'arena> {
    arena: &'arena Arena,
    products: Exported<usearch_matches_tape_view_t>,
    count: usize,
    required: usize,
}

impl<'arena> DenseProducts<'arena> {
    /// Left rows the call multiplied, or queries it scored.
    pub fn queries(&self) -> usize { self.count }

    /// Pairs the walks scored across every left row, or keys scored across every query, before
    /// the wanted bound cut any.
    pub fn required(&self) -> usize { self.required }

    /// The pairs of one left row, a key naming each indexed entry beside its product, or the keys
    /// of one query beside their summed maxima, refused with [`Status::InvalidArgument`] when
    /// `query` is not below [`DenseProducts::queries`] and with [`Status::Device`] over device
    /// memory.
    pub fn query(&self, query: usize) -> Result<QueryMatches<'arena>> {
        self.arena.host()?;
        match query < self.count {
            // Safety: the arena holds a finished call's output in host memory.
            true => Ok(unsafe { self.products.0.query(query) }),
            false => Err(fail(Status::InvalidArgument, "the query is not below the call's count")),
        }
    }

    /// Every query's pairs or keys, in query order, refused with [`Status::Device`] over device memory.
    pub fn iter(&self) -> Result<impl Iterator<Item = QueryMatches<'arena>>> {
        self.arena.host()?;
        let products = self.products.0;
        // Safety: the arena holds a finished call's output in host memory.
        Ok((0..self.count).map(move |query| unsafe { products.query(query) }))
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
            usearch_matches_tape_view_to_arrow(self.products.0, self.count, self.arena.handle, &mut schema, &mut array)
        };
        arrow_checked(status)?;
        Ok(ArrowExport {
            arena: self.arena,
            schema,
            array,
        })
    }
}

/// The bytes [`DenseVerbs::save_buffer`] exported into the arena it borrows.
///
/// Rust-only.
pub struct Monolith<'arena> {
    arena: &'arena Arena,
    monolith: Exported<usearch_bytes_view_t>,
}

impl<'arena> Monolith<'arena> {
    /// The monolith's bytes, refused with [`Status::Device`] over device memory.
    pub fn view(&self) -> Result<&'arena [u8]> {
        self.arena.host()?;
        // Safety: the arena holds a finished call's output in host memory.
        Ok(unsafe { arena_slice(self.monolith.0.data, self.monolith.0.length) })
    }
}

/// The draws [`DenseVerbs::sample`] exported into the arena it borrows.
///
/// Rust-only.
pub struct Sample<'arena> {
    arena: &'arena Arena,
    keys: Exported<usearch_keys_view_t>,
    entry_ordinals: Exported<*const usize>,
    parents: Exported<usearch_matches_view_t>,
    coordinates: Exported<usearch_rows_view_t>,
    written: usize,
}

impl<'arena> Sample<'arena> {
    /// Draws the call wrote; fewer than wanted only once the region is exhausted.
    pub fn written(&self) -> usize { self.written }

    /// The keys drawn, refused with [`Status::Device`] over device memory.
    pub fn keys(&self) -> Result<&'arena [Key]> {
        self.arena.host()?;
        // Safety: the arena holds a finished call's packed output in host memory.
        Ok(unsafe { arena_slice(self.keys.0.data, self.written) })
    }

    /// Which of its key's entries each draw is, in the order [`DenseVerbs::get`] writes them.
    pub fn entry_ordinals(&self) -> Result<&'arena [usize]> {
        self.arena.host()?;
        // Safety: the arena holds a finished call's packed output in host memory.
        Ok(unsafe { arena_slice(self.entry_ordinals.0, self.written) })
    }

    /// The one shown or earlier draw each draw hangs from and its distance, one query per draw,
    /// a draw with none holding no match.
    pub fn parents(&self) -> Matches<'arena> {
        Matches {
            arena: self.arena,
            view: Exported(self.parents.0),
            queries: self.written,
        }
    }

    /// Where each draw is placed, beside the shown keys.
    pub fn coordinates(&self) -> Result<&'arena [[f32; 3]]> {
        self.arena.host()?;
        // Safety: the arena holds a finished call's packed output of three `f32` per row.
        Ok(unsafe { arena_slice(self.coordinates.0.data.cast::<[f32; 3]>(), self.written) })
    }
}

/// One cursor pull exported into the arena it borrows: the keys, and beside them the keys each
/// entry links to when the pull asked; the vectors themselves come from [`DenseVerbs::get`].
///
/// Rust-only.
pub struct DenseBatch<'arena> {
    arena: &'arena Arena,
    keys: Exported<usearch_keys_view_t>,
    links: Option<Exported<usearch_keys_tape_view_t>>,
    written: usize,
}

impl<'arena> DenseBatch<'arena> {
    /// Entries the pull wrote; zero means the walk is exhausted.
    pub fn written(&self) -> usize { self.written }

    /// The keys of the entries, refused with [`Status::Device`] over device memory.
    pub fn keys(&self) -> Result<&'arena [Key]> {
        self.arena.host()?;
        // Safety: the arena holds a finished call's packed output in host memory.
        Ok(unsafe { arena_slice(self.keys.0.data, self.written) })
    }

    /// The keys entry `entry` links to, refused with [`Status::InvalidArgument`] when the pull did
    /// not ask for links or `entry` is not below [`DenseBatch::written`], and with
    /// [`Status::Device`] over device memory.
    pub fn links(&self, entry: usize) -> Result<&'arena [Key]> {
        self.arena.host()?;
        match (&self.links, entry < self.written) {
            // Safety: the arena holds a finished call's output in host memory.
            (Some(links), true) => Ok(unsafe { links.0.entry(entry) }),
            _ => Err(fail(
                Status::InvalidArgument,
                "the pull exported no links, or the entry is not below its count",
            )),
        }
    }
}

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
/// let executor = Executor::for_ram(&[], None, 0, 0)?;
/// let mut arena = Arena::new_in(&executor)?;
/// let matches = index.on(&executor).search(&rows(&query, 4)?, 10, &mut arena)?;
/// for query in matches.iter()? {
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
    fn drop(&mut self) { unsafe { usearch_dense_free(self.handle) } }
}

impl DenseIndex {
    /// An index whose handle a constructor is about to fill.
    fn null() -> Self {
        DenseIndex {
            handle: ptr::null_mut(),
        }
    }

    /// The receiver whose verbs block on `executor`, after the calls submitted there before them.
    ///
    /// Rust-only.
    pub fn on<'index>(&'index self, executor: &'index Executor) -> On<'index, DenseIndex> {
        On {
            index: self,
            executor,
            completion: PhantomData,
        }
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
    /// let executor = Executor::for_ram(&[], None, 0, 0)?;
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
    /// let executor = Executor::for_ram(&[], None, 0, 0)?;
    /// let mut arena = Arena::new_in(&executor)?;
    /// let call = index.deferred(&executor).get::<f32>(Key::from(1u64), &mut arena)?;
    /// let vectors = call.wait()?;
    /// # Ok::<(), usearch::Error>(())
    /// ```
    ///
    /// Rust-only.
    pub fn deferred<'index>(&'index self, executor: &'index Executor) -> On<'index, DenseIndex, Deferring> {
        On {
            index: self,
            executor,
            completion: PhantomData,
        }
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
                ram: ram_options(memory_domains, options.cpu_class),
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
                vram: vram_options(devices, options.gpu_class),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a checkpoint into a new, mutable index in host memory; a bare path reads on the
    /// calling thread.
    pub fn load_ram<'config>(config: impl Into<DenseLoadRamConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let callbacks = Callbacks::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps, &callbacks);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_ram(&mut usearch_dense_load_ram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                ram: ram_options(config.memory_domains, config.cpu_class),
                connectivity: 0,
                values_per_key: config.values_per_key,
                progress,
            })
        };
        callbacks.raise();
        check(outcome).map(|()| index)
    }

    /// Uploads a checkpoint into a new, mutable index in device memory, placing its parts as
    /// [`DenseIndex::load_ram`] does.
    pub fn load_vram<'config>(config: impl Into<DenseLoadVramConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let callbacks = Callbacks::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps, &callbacks);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_vram(&mut usearch_dense_load_vram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                vram: vram_options(config.devices, config.gpu_class),
                values_per_key: config.values_per_key,
                progress,
            })
        };
        callbacks.raise();
        check(outcome).map(|()| index)
    }

    /// Copies a monolith from a buffer into a new, mutable index in host memory.
    pub fn load_ram_buffer<'config>(config: impl Into<DenseLoadRamBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let callbacks = Callbacks::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps, &callbacks);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_ram_buffer(&mut usearch_dense_load_ram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                ram: ram_options(config.memory_domains, config.cpu_class),
                connectivity: 0,
                values_per_key: config.values_per_key,
                progress,
            })
        };
        callbacks.raise();
        check(outcome).map(|()| index)
    }

    /// Uploads a monolith from a buffer into a new, mutable index in device memory.
    pub fn load_vram_buffer<'config>(config: impl Into<DenseLoadVramBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let callbacks = Callbacks::default();
        let progress = lower_progress(&mut config.progress, config.progress_steps, &callbacks);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_vram_buffer(&mut usearch_dense_load_vram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                vram: vram_options(config.devices, config.gpu_class),
                values_per_key: config.values_per_key,
                progress,
            })
        };
        callbacks.raise();
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
    ///
    /// Answers [`Status::Unsupported`] until NVMe checkpoints carry keys.
    pub fn view_nvme<'config>(config: impl Into<DenseViewNvmeConfig<'config>>) -> Result<Self> {
        let config = config.into();
        let views = path_views(config.paths)?;
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_view_nvme(&mut usearch_dense_view_nvme_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                nvme: nvme_options(&views, config.cpu_class),
                ram: ptr::null(),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Reads a checkpoint's geometry and metric without loading it; a manifest answers from its
    /// summary and its first part's header, so the cost does not grow with the index.
    pub fn metadata(path: impl AsRef<Path>) -> Result<DenseMetadata> {
        let path = path_view(path.as_ref())?;
        let mut raw = usearch_dense_metadata_t::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_metadata_read(&mut usearch_dense_metadata_read_t {
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path,
                metadata: &mut raw,
            })
        };
        check(outcome)?;
        DenseMetadata::from_raw(&raw)
    }

    /// Reads geometry and metric out of a serialized monolith buffer.
    pub fn metadata_buffer(buffer: &[u8]) -> Result<DenseMetadata> {
        let mut raw = usearch_dense_metadata_t::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_metadata_read_buffer(&mut usearch_dense_metadata_read_buffer_t {
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: buffer.into(),
                metadata: &mut raw,
            })
        };
        check(outcome)?;
        DenseMetadata::from_raw(&raw)
    }
}

/// What a verb answers through `Receiver`: what the call produced, or a [`Deferred`](crate::Deferred).
type Answer<'index, Receiver, Verb> =
    <<Receiver as DenseVerbs<'index>>::Completion as Completion>::Answer<'index, Verb>;

/// Every dense verb, written once for every receiver: `&DenseIndex` blocks on the calling thread,
/// and the [`On`] that [`DenseIndex::on`] or [`DenseIndex::deferred`] makes runs on an executor.
///
/// A blocking receiver answers what the call produced; a deferring one answers a
/// [`Deferred`](crate::Deferred) that borrows the index. Every configurable verb's config
/// converts into its `Dense*Config`, so the bare value is the whole call and
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
    fn handles(&self) -> Handles;

    /// Reserves capacity for a number of incoming vectors ahead of insertion, a hint, since
    /// insertions grow the index on demand.
    fn reserve(self, capacity: usize) -> Answer<'index, Self, DenseReserve> {
        <Self::Completion as Completion>::answer(DenseReserve {
            handles: self.handles(),
            capacity,
        })
    }

    /// Reports the frozen metadata and the current counters in one call.
    fn describe(self) -> Answer<'index, Self, DenseDescribe> {
        <Self::Completion as Completion>::answer(DenseDescribe {
            handles: self.handles(),
            description: usearch_dense_index_description_t::default(),
        })
    }

    /// The number of entries stored under `key`.
    fn count(self, key: Key) -> Answer<'index, Self, DenseCount> {
        <Self::Completion as Completion>::answer(DenseCount {
            handles: self.handles(),
            key,
            entries: 0,
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

    /// Retrieves every vector stored under `key` into `arena`, which the result borrows; an absent
    /// key yields no rows.
    fn get<'arena, Value: Element>(
        self,
        key: Key,
        arena: &'arena mut Arena,
    ) -> Answer<'index, Self, DenseGet<'arena, Value>> {
        <Self::Completion as Completion>::answer(DenseGet {
            handles: self.handles(),
            key,
            arena,
            vectors: Exported::default(),
            written: 0,
            value: PhantomData,
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
    ) -> Answer<'index, Self, DenseInsert<'config, Keys, Rows>>
    where
        Self::Completion: Accepts<Keys> + Accepts<DenseInsertConfig<'config, Rows>>,
    {
        <Self::Completion as Completion>::answer(DenseInsert {
            handles: self.handles(),
            keys,
            config: config.into(),
        })
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, answering a
    /// [`RefineReport`] of the nodes re-selected and changed and the islands it re-linked.
    ///
    /// A cheap insertion pass followed by refinement beats one expensive pass, so a wide expansion
    /// factor belongs here rather than in [`DenseVerbs::insert`]. On a RAM handle searches may run
    /// throughout, on a VRAM one they may not, and insertions never may. A stop is a time budget:
    /// the call still re-links its islands, leaving the graph correct, and answers
    /// [`Status::Cancelled`].
    fn refine<'config>(self, config: DenseRefineConfig<'config>) -> Answer<'index, Self, DenseRefine<'config>>
    where
        Self::Completion: Accepts<DenseRefineConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(DenseRefine {
            handles: self.handles(),
            report: RefineReport::default(),
            config,
        })
    }

    /// Audits how much of the graph a search reaches: forward reachability from as many landmarks
    /// as one search draws, swept level by level.
    ///
    /// Read-only, so concurrent insertions merely leave fresh entries unaudited.
    fn health(self) -> Answer<'index, Self, DenseHealth> {
        <Self::Completion as Completion>::answer(DenseHealth {
            handles: self.handles(),
            reachability: DenseReachability::default(),
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
    ) -> Answer<'index, Self, DenseSearch<'config, 'arena, Queries>>
    where
        Self::Completion: Accepts<Queries> + Accepts<DenseSearchConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(DenseSearch {
            handles: self.handles(),
            queries,
            config: config.into(),
            arena,
            matches: Exported::default(),
            count: 0,
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
    ) -> Answer<'index, Self, DenseMaxsim<'config, 'arena, Queries>>
    where
        Self::Completion: Accepts<Queries> + Accepts<DenseMaxsimConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(DenseMaxsim {
            handles: self.handles(),
            queries,
            rows_per_query,
            config: config.into(),
            arena,
            matches: Exported::default(),
            count: 0,
            required: 0,
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
    ) -> Answer<'index, Self, DenseMultiply<'config, 'arena, Left>>
    where
        Self::Completion: Accepts<Left> + Accepts<DenseMultiplyConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(DenseMultiply {
            handles: self.handles(),
            left,
            config: config.into(),
            arena,
            products: Exported::default(),
            count: 0,
            required: 0,
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
    fn attend<'config, 'lookup, 'arena, Queries: RowsView, Value: Element>(
        self,
        queries: Queries,
        value_lookup: &'lookup ValueLookup<'lookup, Value>,
        config: impl Into<DenseAttendConfig<'config>>,
        arena: &'arena mut Arena,
    ) -> Answer<'index, Self, DenseAttend<'config, 'lookup, 'arena, Queries, Value>>
    where
        Self::Completion:
            Accepts<Queries> + Accepts<&'lookup ValueLookup<'lookup, Value>> + Accepts<DenseAttendConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(DenseAttend {
            handles: self.handles(),
            queries,
            lookup: value_lookup,
            config: config.into(),
            arena,
            outputs: Exported::default(),
            count: 0,
        })
    }

    /// Absorbs every donor into this index, one after another, keeping their graphs rather than
    /// re-inserting their vectors and linking each in by walks across the appended part.
    ///
    /// Readers of a RAM index may keep searching throughout, while a VRAM one admits no
    /// overlapping search. The donors are consumed whether or not the absorb succeeds: a stop cuts
    /// only the closing refinement, every donor already moved, and answers [`Status::Cancelled`],
    /// while a failure frees the donors it never reached and may leave the failing donor's entries
    /// behind.
    fn absorb<'config, Donors: IntoIterator<Item = DenseIndex>>(
        self,
        config: impl Into<DenseAbsorbConfig<'config, Donors>>,
    ) -> Answer<'index, Self, DenseAbsorb<'config>>
    where
        Self::Completion: Accepts<DenseAbsorbConfig<'config, Donors>>,
    {
        let config = config.into();
        <Self::Completion as Completion>::answer(DenseAbsorb {
            handles: self.handles(),
            donors: DonorHandles::of(config.donors),
            report: RefineReport::default(),
            progress: config.progress,
            progress_steps: config.progress_steps,
        })
    }

    /// Splits this index into the empty receivers, the inverse of [`DenseVerbs::absorb`]: the
    /// entries are partitioned along the graph, each part lands in one receiver with its edges
    /// kept, and a closing refinement re-links every receiver, leaving this index empty and live.
    fn split<'config>(self, config: impl Into<DenseSplitConfig<'config>>) -> Answer<'index, Self, DenseSplit<'config>>
    where
        Self::Completion: Accepts<DenseSplitConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(DenseSplit {
            handles: self.handles(),
            receivers: ReceiverHandles(Vec::new()),
            report: RefineReport::default(),
            config: config.into(),
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
    ) -> Answer<'index, Self, DenseSample<'config, 'arena>>
    where
        Self::Completion: Accepts<DenseSampleConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(DenseSample {
            handles: self.handles(),
            wanted,
            config,
            arena,
            keys: Exported::default(),
            entry_ordinals: Exported(ptr::null()),
            parents: Exported::default(),
            coordinates: Exported::default(),
            written: 0,
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
    ) -> Answer<'index, Self, DenseSave<'config, Location>>
    where
        Self::Completion: Accepts<DenseSaveConfig<'config, Location>>,
    {
        <Self::Completion as Completion>::answer(DenseSave {
            handles: self.handles(),
            config: config.into(),
        })
    }

    /// Saves the index as one monolith into the arena its config carries.
    fn save_buffer<'config, 'arena>(
        self,
        config: impl Into<DenseSaveBufferConfig<'config, 'arena>>,
    ) -> Answer<'index, Self, DenseSaveBuffer<'config, 'arena>>
    where
        Self::Completion: Accepts<DenseSaveBufferConfig<'config, 'arena>>,
    {
        <Self::Completion as Completion>::answer(DenseSaveBuffer {
            handles: self.handles(),
            monolith: Exported::default(),
            config: config.into(),
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
        let mut cursor = DenseCursor {
            handle: ptr::null_mut(),
            handles: self.handles(),
            receiver: PhantomData,
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_cursor_init(&mut usearch_dense_cursor_init_t {
                cursor: &mut cursor.handle,
                index: cursor.handles.index,
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
        let query = match config.query.batch()? {
            batch if batch.count == 1 => batch.rows,
            _ => return Err(fail(Status::ShapeMismatch, "a stream walks exactly one query row")),
        };
        let mut stream = DenseSearchStream {
            handle: ptr::null_mut(),
            handles: self.handles(),
            receiver: PhantomData,
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_search_stream_init(&mut usearch_dense_search_stream_init_t {
                stream: &mut stream.handle,
                index: stream.handles.index,
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

    fn handles(&self) -> Handles {
        Handles {
            index: self.handle,
            executor: ptr::null_mut(),
        }
    }
}

impl<'index, Completion: crate::types::Completion> DenseVerbs<'index> for On<'index, DenseIndex, Completion> {
    type Completion = Completion;

    fn handles(&self) -> Handles {
        Handles {
            index: self.index.handle,
            executor: self.executor.handle,
        }
    }
}

// endregion: Dense Index

// region: Calls

/// A capacity reservation; see [`DenseVerbs::reserve`].
pub struct DenseReserve {
    handles: Handles,
    capacity: usize,
}

impl Calling for DenseReserve {
    type Arguments = usearch_dense_reserve_t;
    type Output = ();

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        _: &Callbacks,
    ) -> Result<Self::Arguments> {
        Ok(usearch_dense_reserve_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            capacity: self.capacity,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_reserve(arguments) }

    fn finish(self) -> Result<()> { Ok(()) }
}

/// A report of the metadata and counters; see [`DenseVerbs::describe`].
pub struct DenseDescribe {
    handles: Handles,
    description: usearch_dense_index_description_t,
}

// Safety: the description holds no pointers, only counters and class enumerators.
unsafe impl Send for DenseDescribe {}

impl Calling for DenseDescribe {
    type Arguments = usearch_dense_describe_t;
    type Output = DenseIndexDescription;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        _: &Callbacks,
    ) -> Result<Self::Arguments> {
        Ok(usearch_dense_describe_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            description: &mut self.description,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_describe(arguments) }

    fn finish(self) -> Result<DenseIndexDescription> {
        Ok(DenseIndexDescription {
            metadata: DenseMetadata::from_raw(&self.description.metadata)?,
            size: self.description.size,
            capacity: self.description.capacity,
            memory_usage: self.description.memory_usage,
            cpu_class: unsafe { static_string(usearch_cpu_class_name(self.description.cpu_class)) },
            gpu_class: unsafe { static_string(usearch_gpu_class_name(self.description.gpu_class)) },
            shards_count: self.description.shards_count,
        })
    }
}

/// A count of the entries under one key; see [`DenseVerbs::count`].
pub struct DenseCount {
    handles: Handles,
    key: Key,
    entries: usize,
}

impl Calling for DenseCount {
    type Arguments = usearch_dense_count_t;
    type Output = usize;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        _: &Callbacks,
    ) -> Result<Self::Arguments> {
        Ok(usearch_dense_count_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            key: self.key,
            entries: &mut self.entries,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_count(arguments) }

    fn finish(self) -> Result<usize> { Ok(self.entries) }
}

/// A retrieval of the vectors under one key; see [`DenseVerbs::get`].
pub struct DenseGet<'arena, Value> {
    handles: Handles,
    key: Key,
    arena: &'arena mut Arena,
    vectors: Exported<usearch_rows_view_t>,
    written: usize,
    value: PhantomData<Value>,
}

impl<'arena, Value: Element> Calling for DenseGet<'arena, Value> {
    type Arguments = usearch_dense_get_t;
    type Output = DenseVectors<'arena, Value>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        _: &Callbacks,
    ) -> Result<Self::Arguments> {
        Ok(usearch_dense_get_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            key: self.key,
            vectors: &mut self.vectors.0,
            written: &mut self.written,
            arena: self.arena.handle,
            outputs_scalar: Value::SCALAR as usearch_scalar_t,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_get(arguments) }

    fn finish(self) -> Result<DenseVectors<'arena, Value>> {
        Ok(DenseVectors {
            arena: self.arena,
            vectors: self.vectors,
            count: self.written,
            value: PhantomData,
        })
    }
}

/// An insertion; see [`DenseVerbs::insert`].
pub struct DenseInsert<'config, Keys, Rows> {
    handles: Handles,
    keys: Keys,
    config: DenseInsertConfig<'config, Rows>,
}

impl<Keys: AsRef<[Key]>, Rows: RowsView> Calling for DenseInsert<'_, Keys, Rows> {
    type Arguments = usearch_dense_insert_t;
    type Output = ();

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let (keys, batch) = (self.keys.as_ref(), self.config.vectors.batch()?);
        // The keys view carries no count, so only this side can see the two disagree.
        if batch.count != keys.len() {
            return Err(fail(
                Status::ShapeMismatch,
                "the key count does not match the vector row count",
            ));
        }
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(usearch_dense_insert_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            keys: usearch_keys_view_t::of(keys),
            vectors: batch.rows,
            expansion_factor: self.config.expansion_factor,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_insert(arguments) }

    fn finish(self) -> Result<()> { Ok(()) }
}

/// A refinement sweep; see [`DenseVerbs::refine`].
pub struct DenseRefine<'config> {
    handles: Handles,
    report: RefineReport,
    config: DenseRefineConfig<'config>,
}

impl Calling for DenseRefine<'_> {
    type Arguments = usearch_dense_refine_t;
    type Output = RefineReport;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(usearch_dense_refine_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            report: &mut self.report,
            refine: usearch_refine_options_t {
                expansion_factor: self.config.expansion_factor,
                memory_limit: 0,
            },
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_refine(arguments) }

    fn finish(self) -> Result<RefineReport> { Ok(self.report) }
}

/// A reachability audit; see [`DenseVerbs::health`].
pub struct DenseHealth {
    handles: Handles,
    reachability: DenseReachability,
}

impl Calling for DenseHealth {
    type Arguments = usearch_dense_health_t;
    type Output = DenseReachability;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        _: &Callbacks,
    ) -> Result<Self::Arguments> {
        Ok(usearch_dense_health_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            reachability: &mut self.reachability,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_health(arguments) }

    fn finish(self) -> Result<DenseReachability> { Ok(self.reachability) }
}

/// A k-nearest-neighbors search; see [`DenseVerbs::search`].
pub struct DenseSearch<'config, 'arena, Queries> {
    handles: Handles,
    queries: Queries,
    config: DenseSearchConfig<'config>,
    arena: &'arena mut Arena,
    matches: Exported<usearch_matches_view_t>,
    count: usize,
}

impl<'arena, Queries: RowsView> Calling for DenseSearch<'_, 'arena, Queries> {
    type Arguments = usearch_dense_search_t;
    type Output = Matches<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.queries.batch()?;
        self.count = batch.count;
        let predicate = lower_predicate(&self.config.predicate, callbacks);
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(usearch_dense_search_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            queries: batch.rows,
            matches: &mut self.matches.0,
            arena: self.arena.handle,
            wanted: self.config.wanted,
            expansion_factor: self.config.expansion_factor,
            traversal: self.config.traversal as usearch_traversal_t,
            predicate,
            hints: ptr::null(),
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_search(arguments) }

    fn finish(self) -> Result<Matches<'arena>> {
        Ok(Matches {
            arena: self.arena,
            view: self.matches,
            queries: self.count,
        })
    }
}

/// A late-interaction search; see [`DenseVerbs::maxsim`].
pub struct DenseMaxsim<'config, 'arena, Queries> {
    handles: Handles,
    queries: Queries,
    rows_per_query: usize,
    config: DenseMaxsimConfig<'config>,
    arena: &'arena mut Arena,
    matches: Exported<usearch_matches_tape_view_t>,
    count: usize,
    required: usize,
}

impl<'arena, Queries: RowsView> Calling for DenseMaxsim<'_, 'arena, Queries> {
    type Arguments = usearch_dense_maxsim_t;
    type Output = DenseProducts<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.queries.batch()?;
        // The call counts queries, so only this side sees the matrix's rows fail to divide.
        if self.rows_per_query == 0 || !batch.count.is_multiple_of(self.rows_per_query) {
            return Err(fail(
                Status::ShapeMismatch,
                "the query rows do not divide into whole queries",
            ));
        }
        self.count = batch.count / self.rows_per_query;
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(usearch_dense_maxsim_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: self.count,
            queries: batch.rows,
            query_offsets: ptr::null(),
            rows_per_query: self.rows_per_query,
            matches: &mut self.matches.0,
            written: ptr::null_mut(),
            required: &mut self.required,
            arena: self.arena.handle,
            wanted: self.config.wanted,
            expansion_factor: self.config.expansion_factor,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
            traversal: Traversal::Walk as usearch_traversal_t,
            predicate: lower_predicate(&None, callbacks),
            hints: ptr::null(),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_maxsim(arguments) }

    fn finish(self) -> Result<DenseProducts<'arena>> {
        Ok(DenseProducts {
            arena: self.arena,
            products: self.matches,
            count: self.count,
            required: self.required,
        })
    }
}

/// An approximate sparse product; see [`DenseVerbs::multiply`].
pub struct DenseMultiply<'config, 'arena, Left> {
    handles: Handles,
    left: Left,
    config: DenseMultiplyConfig<'config>,
    arena: &'arena mut Arena,
    products: Exported<usearch_matches_tape_view_t>,
    count: usize,
    required: usize,
}

impl<'arena, Left: RowsView> Calling for DenseMultiply<'_, 'arena, Left> {
    type Arguments = usearch_dense_multiply_t;
    type Output = DenseProducts<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let batch = self.left.batch()?;
        self.count = batch.count;
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(usearch_dense_multiply_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            left: batch.rows,
            products: &mut self.products.0,
            written: ptr::null_mut(),
            required: &mut self.required,
            arena: self.arena.handle,
            wanted: self.config.wanted,
            expansion_factor: self.config.expansion_factor,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
            traversal: Traversal::Walk as usearch_traversal_t,
            predicate: lower_predicate(&None, callbacks),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_multiply(arguments) }

    fn finish(self) -> Result<DenseProducts<'arena>> {
        Ok(DenseProducts {
            arena: self.arena,
            products: self.products,
            count: self.count,
            required: self.required,
        })
    }
}

/// # Safety
/// `state` must address the live [`Callbacks`] of the call the callback was passed to, lending a
/// `&ValueLookup`, and `keys` and `rows` must hold `count` entries each.
unsafe extern "C" fn value_lookup_trampoline<Value: Element>(
    state: *mut c_void,
    keys: *const Key,
    count: usize,
    scalar: usearch_scalar_t,
    columns: usize,
    rows: *mut *const c_void,
) {
    let callbacks = &*state.cast::<Callbacks>();
    let lookup = *callbacks.lookup.get().cast::<&ValueLookup<'_, Value>>();
    // The library reads rows of its own shape, so a lookup typed otherwise answers no row at all.
    let row_length = match scalar == Value::SCALAR as usearch_scalar_t {
        true => values_of::<Value>(columns),
        false => usize::MAX,
    };
    let keys = slice::from_raw_parts(keys, count);
    for (key, row) in keys.iter().zip(slice::from_raw_parts_mut(rows, count)) {
        // A panic leaves the row null, which refuses the call, and is re-raised once it is final.
        *row = match callbacks.guard(None, || lookup(*key)) {
            Some(value) if value.len() == row_length => value.as_ptr().cast(),
            _ => ptr::null(),
        };
    }
}

/// A graph-walk attention; see [`DenseVerbs::attend`].
pub struct DenseAttend<'config, 'lookup, 'arena, Queries, Value> {
    handles: Handles,
    queries: Queries,
    lookup: &'lookup ValueLookup<'lookup, Value>,
    config: DenseAttendConfig<'config>,
    arena: &'arena mut Arena,
    outputs: Exported<usearch_rows_view_t>,
    count: usize,
}

impl<'arena, Queries: RowsView, Value: Element> Calling for DenseAttend<'_, '_, 'arena, Queries, Value> {
    type Arguments = usearch_dense_attend_t;
    type Output = DenseVectors<'arena, Queries::Value>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        // A VRAM walk calls its value lookup on the device, which a host trampoline cannot serve.
        if self.arena.describe()?.residency == Residency::Vram {
            return Err(fail(Status::Device, "a VRAM walk needs a device-callable value lookup"));
        }
        let batch = self.queries.batch()?;
        self.count = batch.count;
        let carry = match self.config.carry.as_deref_mut() {
            None => ptr::null_mut(),
            // The carry pointer has no length, so only this side can see it fall short.
            Some(carry) if carry.len() < batch.count => {
                return Err(fail(
                    Status::ShapeMismatch,
                    "the softmax carry holds fewer entries than there are queries",
                ))
            }
            Some(carry) => carry.as_mut_ptr(),
        };
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        callbacks.lookup.set(ptr::from_ref(&self.lookup).cast());
        Ok(usearch_dense_attend_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            queries: batch.rows,
            head_count: self.config.head_count,
            value_lookup: Some(value_lookup_trampoline::<Value>),
            value_lookup_state: callbacks.state(),
            carry,
            outputs: &mut self.outputs.0,
            arena: self.arena.handle,
            outputs_scalar: Queries::Value::SCALAR as usearch_scalar_t,
            expansion_factor: self.config.expansion_factor,
            score_scale: self.config.score_scale,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
            traversal: Traversal::Walk as usearch_traversal_t,
            predicate: lower_predicate(&None, callbacks),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_attend(arguments) }

    fn finish(self) -> Result<DenseVectors<'arena, Queries::Value>> {
        Ok(DenseVectors {
            arena: self.arena,
            vectors: self.outputs,
            count: self.count,
            value: PhantomData,
        })
    }
}

/// Donor handles the library nulls as it absorbs them; the rest are freed with the call.
struct DonorHandles(Vec<usearch_dense_index_t>);

impl DonorHandles {
    /// Takes the handles out of the donors, which no longer free them.
    fn of(donors: impl IntoIterator<Item = DenseIndex>) -> Self {
        DonorHandles(
            donors
                .into_iter()
                .map(|donor| ManuallyDrop::new(donor).handle)
                .collect(),
        )
    }
}

// Safety: the handles were owned `DenseIndex`es, which are `Send`.
unsafe impl Send for DonorHandles {}

impl Drop for DonorHandles {
    fn drop(&mut self) {
        for &handle in &self.0 {
            unsafe { usearch_dense_free(handle) }
        }
    }
}

/// An absorb of donor indexes; see [`DenseVerbs::absorb`].
pub struct DenseAbsorb<'config> {
    handles: Handles,
    donors: DonorHandles,
    report: RefineReport,
    progress: Option<&'config mut ProgressCallback<'config>>,
    progress_steps: usize,
}

impl Calling for DenseAbsorb<'_> {
    type Arguments = usearch_dense_absorb_t;
    type Output = RefineReport;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let progress = lower_progress(&mut self.progress, self.progress_steps, callbacks);
        Ok(usearch_dense_absorb_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: self.donors.0.len(),
            donors: self.donors.0.as_mut_ptr(),
            report: &mut self.report,
            refine: usearch_refine_options_t {
                expansion_factor: 0,
                memory_limit: 0,
            },
            progress,
            stats: ptr::null_mut(),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_absorb(arguments) }

    fn finish(self) -> Result<RefineReport> { Ok(self.report) }
}

/// A split into receiver indexes; see [`DenseVerbs::split`].
pub struct DenseSplit<'config> {
    handles: Handles,
    receivers: ReceiverHandles,
    report: RefineReport,
    config: DenseSplitConfig<'config>,
}

/// The handles of the receivers a split borrows, laid out as the library reads them.
struct ReceiverHandles(Vec<usearch_dense_index_t>);

// Safety: the handles belong to borrowed `DenseIndex`es, which are `Sync`.
unsafe impl Send for ReceiverHandles {}

impl Calling for DenseSplit<'_> {
    type Arguments = usearch_dense_split_t;
    type Output = RefineReport;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        self.receivers = ReceiverHandles(self.config.receivers.iter().map(|receiver| receiver.handle).collect());
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(usearch_dense_split_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: self.receivers.0.len(),
            receivers: self.receivers.0.as_ptr(),
            report: &mut self.report,
            refine: usearch_refine_options_t {
                expansion_factor: self.config.expansion_factor,
                memory_limit: 0,
            },
            seed: self.config.seed,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_split(arguments) }

    fn finish(self) -> Result<RefineReport> { Ok(self.report) }
}

/// A sample of draws around a focus; see [`DenseVerbs::sample`].
pub struct DenseSample<'config, 'arena> {
    handles: Handles,
    wanted: usize,
    config: DenseSampleConfig<'config>,
    arena: &'arena mut Arena,
    keys: Exported<usearch_keys_view_t>,
    entry_ordinals: Exported<*const usize>,
    parents: Exported<usearch_matches_view_t>,
    coordinates: Exported<usearch_rows_view_t>,
    written: usize,
}

impl<'arena> Calling for DenseSample<'_, 'arena> {
    type Arguments = usearch_dense_sample_t;
    type Output = Sample<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        if self.config.shown.len() != self.config.shown_coordinates.len() {
            return Err(fail(Status::InvalidArgument, "every shown key needs its coordinates"));
        }
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(usearch_dense_sample_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            shown_count: self.config.shown.len(),
            shown: usearch_keys_view_t::of(self.config.shown),
            shown_coordinates: usearch_rows_view_t {
                data: self.config.shown_coordinates.as_ptr().cast(),
                stride: core::mem::size_of::<[f32; 3]>(),
                columns: 3,
                scalar: Scalar::F32 as usearch_scalar_t,
            },
            focus_count: self.config.focus.len(),
            focus: usearch_keys_view_t::of(self.config.focus),
            keys: &mut self.keys.0,
            entry_ordinals: &mut self.entry_ordinals.0,
            parents: &mut self.parents.0,
            coordinates: &mut self.coordinates.0,
            written: &mut self.written,
            arena: self.arena.handle,
            wanted: self.wanted,
            seed: self.config.seed,
            expansion_factor: self.config.expansion_factor,
            progress,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_sample(arguments) }

    fn finish(self) -> Result<Sample<'arena>> {
        Ok(Sample {
            arena: self.arena,
            keys: self.keys,
            entry_ordinals: self.entry_ordinals,
            parents: self.parents,
            coordinates: self.coordinates,
            written: self.written,
        })
    }
}

/// A checkpoint write; see [`DenseVerbs::save`].
pub struct DenseSave<'config, Location> {
    handles: Handles,
    config: DenseSaveConfig<'config, Location>,
}

impl<Location: AsRef<Path>> Calling for DenseSave<'_, Location> {
    type Arguments = usearch_dense_save_t;
    type Output = ();

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let path = path_view(self.config.path.as_ref())?;
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(usearch_dense_save_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            path,
            progress,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_save(arguments) }

    fn finish(self) -> Result<()> { Ok(()) }
}

/// A monolith written into an arena; see [`DenseVerbs::save_buffer`].
pub struct DenseSaveBuffer<'config, 'arena> {
    handles: Handles,
    monolith: Exported<usearch_bytes_view_t>,
    config: DenseSaveBufferConfig<'config, 'arena>,
}

impl<'arena> Calling for DenseSaveBuffer<'_, 'arena> {
    type Arguments = usearch_dense_save_buffer_t;
    type Output = Monolith<'arena>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
    ) -> Result<Self::Arguments> {
        let progress = lower_progress(&mut self.config.progress, self.config.progress_steps, callbacks);
        Ok(usearch_dense_save_buffer_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            monolith: &mut self.monolith.0,
            arena: self.config.arena.handle,
            progress,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_save_buffer(arguments) }

    fn finish(self) -> Result<Monolith<'arena>> {
        Ok(Monolith {
            arena: self.config.arena,
            monolith: self.monolith,
        })
    }
}

// endregion: Calls

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
    handles: Handles,
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
        let (mut keys, mut links, mut written) = (Exported::default(), Exported::default(), 0);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_cursor_next(&mut usearch_dense_cursor_next_t {
                index: self.handles.index,
                executor: self.handles.executor,
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                cursor: self.handle,
                keys: &mut keys.0,
                links: match linked {
                    true => ptr::from_mut(&mut links.0),
                    false => ptr::null_mut(),
                },
                written: &mut written,
                arena: arena.handle,
                wanted,
            })
        };
        check(outcome)?;
        Ok(DenseBatch {
            arena,
            keys,
            links: linked.then_some(links),
            written,
        })
    }
}

impl Drop for DenseCursor<'_> {
    fn drop(&mut self) { unsafe { usearch_dense_cursor_free(self.handle) } }
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
    handles: Handles,
    receiver: PhantomData<On<'index, DenseIndex>>,
}

impl DenseSearchStream<'_> {
    /// Pulls up to `wanted` more matches into `arena`, as the one query of [`Matches`].
    pub fn next<'arena>(&mut self, wanted: usize, arena: &'arena mut Arena) -> Result<Matches<'arena>> {
        let mut matches = Exported::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_search_stream_next(&mut usearch_dense_search_stream_next_t {
                index: self.handles.index,
                executor: self.handles.executor,
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                stream: self.handle,
                matches: &mut matches.0,
                arena: arena.handle,
                wanted,
            })
        };
        check(outcome)?;
        Ok(Matches {
            arena,
            view: matches,
            queries: 1,
        })
    }
}

impl Drop for DenseSearchStream<'_> {
    fn drop(&mut self) { unsafe { usearch_dense_search_stream_free(self.handle) } }
}

// endregion: Cursors and Streams

// region: Tests

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use numkong::{bf16, u1x8, Matrix, MatrixView};
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::types::{rows, Deferred, Mood};

    const SEED: u64 = 42;

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

    fn sequential_keys(count: usize) -> Vec<Key> { (0..count as u64).map(Key::from).collect() }

    /// One executor per test, since an executor orders the calls submitted to it.
    fn executor() -> Executor { Executor::for_ram(&[], None, 0, 0).expect("the executor must spawn") }

    fn empty_index(dimensions: usize, metric: DenseMetric, quantization: Scalar) -> Result<DenseIndex> {
        let options = DenseOptions {
            dimensions,
            metric,
            quantization,
            ..DenseOptions::default()
        };
        let index = DenseIndex::init_ram(&options, &[])?;
        index.reserve(1024)?;
        Ok(index)
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
        let slab = seeded_rows(count, dimensions, SEED);
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

    /// A checkpoint path scoped by test name and process, never the working directory.
    fn scratch_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("usearch-rust-{name}-{}.safegraphs", std::process::id()))
    }

    fn remove_quietly(path: &std::path::Path) { let _ = std::fs::remove_file(path); }

    /// The spacing of bf16 values at one, the suites' stored kind.
    const BF16_EPSILON: f64 = 1.0 / 128.0;

    fn close(left: f64, right: f64, epsilon: f64) -> bool {
        (left - right).abs() <= epsilon * left.abs().max(right.abs()).max(1.0)
    }

    /// Whether rows read back from bf16 storage match the rows written.
    fn close_rows(read: &[f32], written: &[f32]) -> bool {
        read.len() == written.len()
            && read
                .iter()
                .zip(written)
                .all(|(left, right)| close(f64::from(*left), f64::from(*right), BF16_EPSILON))
    }

    fn requires_send_sync<Subject: Send + Sync>() {}

    fn requires_send<Subject: Send>() {}

    /// Wraps raw bytes as packed bit rows, eight dimensions to a byte.
    fn bit_rows(bytes: &[u8]) -> Vec<u1x8> { bytes.iter().map(|&byte| u1x8(byte)).collect() }

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
    /// inside a handler stops the call first and resurfaces only once it is final.
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
            Watch {
                last: (0, 0),
                ordered: true,
                patience,
            }
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

    // region: Metadata Marshalling

    #[test]
    fn layouts() {
        use std::mem::{offset_of, size_of};

        let word = size_of::<usize>();
        assert_eq!(offset_of!(DenseTraversalStats, restarts), 2 * word);
        assert_eq!(size_of::<DenseTraversalStats>(), 4 * word);
        let cpu_class = offset_of!(usearch_dense_index_description_t, cpu_class);
        assert_eq!(offset_of!(usearch_dense_index_description_t, metadata), 0);
        assert_eq!(
            offset_of!(usearch_dense_index_description_t, size),
            size_of::<usearch_dense_metadata_t>()
        );
        assert_eq!(cpu_class, size_of::<usearch_dense_metadata_t>() + 3 * word);
        assert_eq!(offset_of!(usearch_dense_index_description_t, gpu_class), cpu_class + 4);
        assert_eq!(
            offset_of!(usearch_dense_index_description_t, shards_count),
            cpu_class + 8
        );
        assert_eq!(size_of::<usearch_dense_index_description_t>(), cpu_class + 8 + word);
    }

    #[test]
    fn metadata_lifts_the_four_fields_a_checkpoint_froze() -> Result<()> {
        let raw = usearch_dense_metadata_t {
            metric: DenseMetric::Euclidean as i32,
            quantization: Scalar::I8 as i32,
            dimensions: 768,
            connectivity: 32,
        };
        let lifted = DenseMetadata::from_raw(&raw)?;
        assert_eq!(lifted.dimensions, 768);
        assert_eq!(lifted.metric, DenseMetric::Euclidean);
        assert_eq!(lifted.quantization, Scalar::I8);
        assert_eq!(lifted.connectivity, 32);
        Ok(())
    }

    #[test]
    fn metadata_rejects_unknown_enum_tags() {
        let mut raw = usearch_dense_metadata_t {
            metric: DenseMetric::Angular as i32,
            quantization: Scalar::BF16 as i32,
            dimensions: 8,
            connectivity: 0,
        };
        raw.metric = 99;
        assert!(matches!(
            DenseMetadata::from_raw(&raw),
            Err(Error {
                status: Status::UnknownKind,
                ..
            })
        ));

        raw.metric = DenseMetric::Angular as i32;
        raw.quantization = 77;
        assert!(matches!(
            DenseMetadata::from_raw(&raw),
            Err(Error {
                status: Status::UnknownKind,
                ..
            })
        ));
    }

    // endregion: Metadata Marshalling

    // region: Construction and Geometry

    /// The one place the Rust enum values are checked against the C ones: a renumbered
    /// constant reaches the library as a different kind, and is caught here and nowhere else.
    /// A storage kind the build's `USEARCH_DENSE_DTYPES` left out is refused as unsupported.
    #[test]
    fn constructs_every_metric_and_quantization() -> Result<()> {
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
            let index = match empty_index(64, DenseMetric::Inner, quantization) {
                Err(error) if error.status == Status::Unsupported => continue,
                built => built?,
            };
            let described = index.describe()?;
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
            let described = empty_index(64, metric, quantization)?.describe()?;
            assert_eq!(described.metadata.metric, metric);
            assert_eq!(metric.to_string().parse(), Ok(metric), "the library spells it back");
        }
        assert!("cosine".parse::<DenseMetric>().is_err());
        Ok(())
    }

    #[test]
    fn does_not_preallocate_members() -> Result<()> {
        let options = DenseOptions {
            dimensions: 16,
            ..DenseOptions::default()
        };
        let index = DenseIndex::init_ram(&options, &[])?;
        let described = index.describe()?;
        assert_eq!((described.size, described.capacity), (0, 0));

        index.reserve(1024)?;
        let described = index.describe()?;
        assert!(described.capacity >= 1024);
        assert_eq!(described.size, 0);
        Ok(())
    }

    /// Init freezes the geometry, so a zeroed width is refused there rather than at the first batch.
    #[test]
    fn zero_dimensions_are_refused_at_init() {
        let options = DenseOptions {
            dimensions: 0,
            ..DenseOptions::default()
        };
        assert!(matches!(
            DenseIndex::init_ram(&options, &[]),
            Err(Error {
                status: Status::ConfigMismatch,
                ..
            })
        ));
    }

    /// The handle is one pointer, so a move must not disturb it and two threads may share it.
    #[test]
    fn survives_box_and_arc_moves() -> Result<()> {
        requires_send_sync::<DenseIndex>();
        let (index, _, _) = populated_index(4, 8, DenseMetric::Angular, Scalar::BF16)?;

        let boxed = Box::new(index);
        assert_eq!(boxed.describe()?.size, 4);
        let shared = std::sync::Arc::new(*boxed);
        assert_eq!(shared.describe()?.size, 4);
        Ok(())
    }

    #[test]
    fn clears_and_reuses() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (mut index, keys, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let reserved = index.describe()?.capacity;

        index.clear()?;
        let cleared = index.describe()?;
        assert_eq!(cleared.size, 0);
        assert_eq!(cleared.capacity, reserved, "clearing keeps the arena");
        assert_eq!(cleared.metadata.dimensions, DIMENSIONS, "and keeps the frozen geometry");

        index.insert(&keys, &rows(&slab, DIMENSIONS)?)?;
        assert_eq!(index.describe()?.size, 16);
        Ok(())
    }

    // endregion: Construction and Geometry

    // region: Ingest and Retrieval

    #[test]
    fn adds_and_gets_vectors() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let keys = sequential_keys(2);
        let slab: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        index.insert(&keys, &rows(&slab, DIMENSIONS)?)?;
        assert_eq!(index.describe()?.size, 2);

        let mut arena = Arena::new()?;
        let found = index.get::<f32>(keys[1], &mut arena)?;
        assert_eq!(found.view()?.shape(), &[1, DIMENSIONS]);
        assert_eq!(flattened(found.view()?), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            index.get::<f32>(Key::from(7u64), &mut arena)?.view()?.shape()[0],
            0,
            "an absent key yields no rows"
        );
        Ok(())
    }

    /// An empty batch is a shape rather than a mistake, so it publishes nothing and reports no
    /// failure - what a caller filtering its own input relies on.
    #[test]
    fn accepts_an_empty_batch() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let empty: [f32; 0] = [];
        index.insert([], &rows(&empty, DIMENSIONS)?)?;
        assert_eq!(index.describe()?.size, 0);
        Ok(())
    }

    #[test]
    fn rejects_mismatched_key_and_row_counts() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        assert!(matches!(
            index.insert(sequential_keys(3), &rows(&slab, DIMENSIONS)?),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        Ok(())
    }

    /// Every view carries the width the caller laid out, so the call refuses a narrow row before
    /// reading past it.
    #[test]
    fn rejects_wrong_dimension_matrices() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, _, _) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let narrow = seeded_rows(1, DIMENSIONS - 1, SEED);

        assert!(matches!(
            index.insert(sequential_keys(1), &rows(&narrow, DIMENSIONS - 1)?),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        assert!(matches!(
            index.search(&rows(&narrow, DIMENSIONS - 1)?, 1, &mut Arena::new()?),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        assert!(
            matches!(
                index.search_stream(rows(&narrow, DIMENSIONS - 1)?),
                Err(Error {
                    status: Status::ShapeMismatch | Status::Unsupported,
                    ..
                })
            ),
            "the query width is checked too, where a residency streams at all"
        );

        Ok(())
    }

    #[test]
    fn rejects_non_contiguous_rows() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let slab = seeded_rows(2, DIMENSIONS * 2, SEED);
        let element = core::mem::size_of::<f32>() as isize;
        // A column stride of two elements interleaves the rows, which the boundary forbids:
        // the ABI carries one row stride and no column stride.
        let interleaved: MatrixView<'_, f32> =
            unsafe { MatrixView::from_raw_parts(slab.as_ptr(), &[2, DIMENSIONS], &[element * 8, element * 2]) }
                .unwrap();
        assert!(matches!(
            index.insert(sequential_keys(2), interleaved),
            Err(Error {
                status: Status::UnsupportedLayout,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn rejects_descending_row_pitch() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let slab = seeded_rows(2, DIMENSIONS, SEED);
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
            Err(Error {
                status: Status::UnsupportedLayout,
                ..
            })
        ));
        Ok(())
    }

    /// A pitch wider than the row is the layout a mapped dataset arrives in, and its rows must
    /// read back packed from the arena, padding left behind.
    #[test]
    fn accepts_strided_row_pitch() -> Result<()> {
        const DIMENSIONS: usize = 4;
        const PITCH: usize = 8;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let mut padded = [0.0f32; 2 * PITCH];
        let rows_source = seeded_rows(2, DIMENSIONS, SEED);
        padded[..DIMENSIONS].copy_from_slice(&rows_source[..DIMENSIONS]);
        padded[PITCH..PITCH + DIMENSIONS].copy_from_slice(&rows_source[DIMENSIONS..]);

        let element = core::mem::size_of::<f32>() as isize;
        let strided: MatrixView<'_, f32> = unsafe {
            MatrixView::from_raw_parts(padded.as_ptr(), &[2, DIMENSIONS], &[element * PITCH as isize, element])
        }
        .unwrap();
        index.insert(sequential_keys(2), strided)?;

        let mut arena = Arena::new()?;
        let walked_keys = index.cursor(&[])?.next_keys(2, &mut arena)?.keys()?.to_vec();
        assert_eq!(walked_keys.len(), 2);
        for key in walked_keys {
            let source = u64::try_from(key).unwrap() as usize;
            let restored = flattened(index.get::<f32>(key, &mut arena)?.view()?);
            assert!(close_rows(&restored, &rows_source[source * DIMENSIONS..][..DIMENSIONS]));
        }
        Ok(())
    }

    #[test]
    fn converts_scalar_kinds_on_ingest() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let keys = sequential_keys(1);
        let slab = seeded_rows(1, DIMENSIONS, SEED);
        index.insert(&keys, &rows(&slab, DIMENSIONS)?)?;

        let wide = flattened(index.get::<f64>(keys[0], &mut Arena::new()?)?.view()?);
        assert_eq!(wide.len(), DIMENSIONS);
        for (left, right) in wide.iter().zip(&slab) {
            assert!(
                close(*left, *right as f64, BF16_EPSILON),
                "bf16 keeps eight mantissa bits"
            );
        }
        Ok(())
    }

    /// One call returns every entry under a key, however many there turn out to be.
    #[test]
    fn gets_every_entry_sharing_a_key() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let shared = Key::from(3u64);
        let slab = seeded_rows(3, DIMENSIONS, SEED);
        index.insert([shared; 3], &rows(&slab, DIMENSIONS)?)?;

        assert_eq!(index.count(shared)?, 3);
        assert!(index.contains(shared)?);
        assert!(!index.contains(Key::from(8u64))?);

        let mut arena = Arena::new()?;
        let restored = index.get::<f32>(shared, &mut arena)?;
        assert_eq!(restored.view()?.shape(), &[3, DIMENSIONS], "one row per entry");
        assert!(close_rows(&flattened(restored.view()?), &slab));
        Ok(())
    }

    /// An arena is reused once its outputs are no longer read, and keeping one call's outputs past
    /// the next takes a second one.
    #[test]
    fn a_second_arena_keeps_the_first_output_alive() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let (index, keys, slab) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let executor = executor();
        let receiver = index.on(&executor);
        let (mut first, mut second) = (Arena::new_in(&executor)?, Arena::new_in(&executor)?);
        let held = receiver.get::<f32>(keys[0], &mut first)?;
        let next = receiver.get::<f32>(keys[1], &mut second)?;
        assert!(close_rows(&flattened(held.view()?), &slab[..DIMENSIONS]));
        assert!(close_rows(&flattened(next.view()?), &slab[DIMENSIONS..DIMENSIONS * 2]));

        let reused = receiver.get::<f32>(keys[3], &mut first)?;
        assert!(close_rows(&flattened(reused.view()?), &slab[DIMENSIONS * 3..]));
        Ok(())
    }

    #[test]
    fn uses_128_bit_keys() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let wide = Key::from(u128::MAX);
        let narrow = Key::from(9u64);
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        index.insert([wide, narrow], &rows(&slab, DIMENSIONS)?)?;

        assert_eq!(index.count(wide)?, 1);
        assert_eq!(index.count(narrow)?, 1);
        assert!(
            u64::try_from(wide).is_err(),
            "a non-zero high half has no integer identity"
        );
        assert_eq!(u64::try_from(narrow), Ok(9));
        Ok(())
    }

    /// A sub-byte kind counts logical dimensions, so eight of them are one stored byte - the
    /// one place where the row width and the row extent part ways.
    #[test]
    fn packed_bit_rows_cross_as_one_byte_per_eight_dimensions() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, DenseMetric::Hamming, Scalar::U1x8)?;
        let stored = bit_rows(&[0b0000_1111, 0b1111_0000]);
        let keys = [Key::from(42u64), Key::from(43u64)];
        index.insert(keys, &rows(&stored, DIMENSIONS)?)?;
        assert_eq!(index.describe()?.size, 2, "two bytes are two rows, not sixteen");

        let query = bit_rows(&[0b0111_1000]);
        let mut arena = Arena::new()?;
        let matches = index.search(&rows(&query, DIMENSIONS)?, 2, &mut arena)?;
        let found = matches.query(0)?;
        assert_eq!(
            found.keys.iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
        Ok(())
    }

    /// A sub-byte row comes back counted in dimensions, never in the bytes that hold them.
    #[test]
    fn gets_sub_byte_rows_at_their_logical_width() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let index = empty_index(DIMENSIONS, DenseMetric::Hamming, Scalar::U1x8)?;
        let key = Key::from(5u64);
        index.insert([key], &rows(&bit_rows(&[0b1010_1010, 0b0000_0101]), DIMENSIONS)?)?;

        let mut arena = Arena::new()?;
        let found = index.get::<u1x8>(key, &mut arena)?;
        assert_eq!(found.view()?.shape(), &[1, DIMENSIONS]);
        Ok(())
    }

    // endregion: Ingest and Retrieval

    // region: Search

    #[test]
    fn searches_an_empty_index() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let query = seeded_rows(1, DIMENSIONS, SEED);
        assert_eq!(matches_of(&index, &query, DIMENSIONS)?, 0);
        Ok(())
    }

    /// A zero `wanted` walks nothing and lands only zeroed counts, with no keys to read.
    #[test]
    fn a_zero_wanted_lands_only_counts() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, _, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let mut arena = Arena::new()?;
        let matches = index.search(&rows(&slab[..4 * DIMENSIONS], DIMENSIONS)?, 0, &mut arena)?;
        assert_eq!(matches.queries(), 4);
        assert!(
            matches.iter()?.all(|query| query.keys.is_empty()),
            "no query holds a match"
        );
        Ok(())
    }

    /// Every query's matches land packed in the arena, and its count in its own slot; a query past
    /// the call's count is refused rather than read.
    #[test]
    fn fills_every_query_of_a_batch() -> Result<()> {
        const DIMENSIONS: usize = 64;
        const QUERIES: usize = 8;
        const WANTED: usize = 5;
        let (index, keys, slab) = populated_index(256, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let executor = executor();
        let mut arena = Arena::new_in(&executor)?;
        let matches = index.on(&executor).search(
            &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?,
            DenseSearchConfig {
                expansion_factor: 64,
                ..DenseSearchConfig::from(WANTED)
            },
            &mut arena,
        )?;
        assert_eq!(matches.queries(), QUERIES);
        assert!(matches.iter()?.all(|found| found.keys.len() == WANTED));
        for (query, found) in matches.iter()?.enumerate() {
            assert_eq!(found.keys[0], keys[query], "each query finds itself first");
        }
        assert_eq!(matches.query(QUERIES).unwrap_err().status, Status::InvalidArgument);
        Ok(())
    }

    #[test]
    fn reports_traversal_stats() -> Result<()> {
        const DIMENSIONS: usize = 32;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let keys = sequential_keys(64);
        let slab = seeded_rows(64, DIMENSIONS, SEED);

        let mut inserting = DenseTraversalStats::default();
        index.insert(
            &keys,
            DenseInsertConfig {
                stats: Some(&mut inserting),
                ..DenseInsertConfig::from(&rows(&slab, DIMENSIONS)?)
            },
        )?;
        assert!(inserting.computed_distances > 0);

        let query = rows(&slab[..DIMENSIONS], DIMENSIONS)?;
        let mut arena = Arena::new()?;
        let mut searching = DenseTraversalStats::default();
        index.search(
            &query,
            DenseSearchConfig {
                stats: Some(&mut searching),
                ..DenseSearchConfig::from(4)
            },
            &mut arena,
        )?;
        let first = searching;
        assert!(first.computed_distances > 0);

        index.search(
            &query,
            DenseSearchConfig {
                stats: Some(&mut searching),
                ..DenseSearchConfig::from(4)
            },
            &mut arena,
        )?;
        assert_eq!(
            searching, first,
            "stats are overwritten by their call, never accumulated"
        );
        Ok(())
    }

    #[test]
    fn filters_with_a_stateless_predicate() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const WANTED: usize = 8;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let is_even = |key: Key| u64::try_from(key).is_ok_and(|value| value % 2 == 0);
        let mut arena = Arena::new()?;
        let matches = index.search(
            &rows(&slab[..DIMENSIONS], DIMENSIONS)?,
            DenseSearchConfig {
                expansion_factor: 64,
                predicate: Some(&is_even),
                ..DenseSearchConfig::from(WANTED)
            },
            &mut arena,
        )?;

        let admitted = matches.query(0)?;
        assert!(!admitted.keys.is_empty());
        for key in admitted.keys {
            assert!(is_even(*key), "{key} was not admitted by the predicate");
        }
        Ok(())
    }

    /// The captured set crosses as the trampoline's context pointer, which is what separates
    /// this from the stateless case above.
    #[test]
    fn filters_with_a_stateful_predicate() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const WANTED: usize = 8;
        let (index, keys, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let allowed: HashSet<Key> = keys.iter().copied().take(5).collect();
        let executor = executor();
        let mut arena = Arena::new_in(&executor)?;
        let matches = index.on(&executor).search(
            &rows(&slab[..DIMENSIONS], DIMENSIONS)?,
            DenseSearchConfig {
                expansion_factor: 64,
                predicate: Some(&|key: Key| allowed.contains(&key)),
                ..DenseSearchConfig::from(WANTED)
            },
            &mut arena,
        )?;

        for key in matches.query(0)?.keys {
            assert!(allowed.contains(key), "{key} is outside the captured set");
        }
        Ok(())
    }

    /// A predicate that panics rejects the key rather than unwinding into the library, and its
    /// panic resurfaces once the call is final instead of passing for an empty answer.
    #[test]
    fn filters_everything_out() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let admitted = |predicate: &Predicate| -> Result<usize> {
            let mut arena = Arena::new()?;
            let matches = index.search(
                &rows(&slab[..DIMENSIONS], DIMENSIONS)?,
                DenseSearchConfig {
                    expansion_factor: 64,
                    predicate: Some(predicate),
                    ..DenseSearchConfig::from(4)
                },
                &mut arena,
            )?;
            Ok(matches.query(0)?.keys.len())
        };

        assert_eq!(admitted(&|_| false)?, 0, "a fully-rejecting predicate is not an error");
        let Err(fault) = std::panic::catch_unwind(|| admitted(&|_| panic!("a filter fault"))) else {
            panic!("a predicate's panic must resurface");
        };
        assert_eq!(fault.downcast_ref::<&str>(), Some(&"a filter fault"));
        Ok(())
    }

    /// A scan scores every entry rather than the ones a walk reaches, so every query finds itself first.
    #[test]
    fn scans_find_every_query_itself() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 64;
        let (index, keys, slab) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let mut arena = Arena::new()?;
        let matches = index.search(
            &rows(&slab, DIMENSIONS)?,
            DenseSearchConfig {
                traversal: Traversal::Scan,
                ..DenseSearchConfig::from(1)
            },
            &mut arena,
        )?;
        assert_eq!(matches.queries(), COUNT);
        for (query, found) in matches.iter()?.enumerate() {
            assert_eq!(found.keys[0], keys[query]);
        }
        Ok(())
    }

    // endregion: Search

    // region: Refinement and Linear Algebra

    /// The return is a report rather than a status, so a caller loops until the graph settles.
    #[test]
    fn refinement_reports_its_sweep() -> Result<()> {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, DenseMetric::Angular, Scalar::BF16)?;
        let mut stats = DenseTraversalStats::default();
        let report = index.on(&executor()).refine(DenseRefineConfig {
            expansion_factor: 64,
            stats: Some(&mut stats),
            ..DenseRefineConfig::default()
        })?;
        assert!(
            report.changed <= report.reselected,
            "a sweep changes only the nodes it re-selected"
        );
        assert_eq!(
            report.reachability.audited, COUNT,
            "the closing audit judges every entry"
        );
        assert!(stats.visited_members > 0, "the sweep reports the nodes it swept");
        Ok(())
    }

    /// The receivers hold every entry between them, and the source stays live but empty.
    #[test]
    fn splits_into_receivers() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let receivers = [
            empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?,
            empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?,
        ];
        for receiver in &receivers {
            receiver.reserve(COUNT)?;
        }
        let borrowed = [&receivers[0], &receivers[1]];
        let report = index.on(&executor()).split(DenseSplitConfig {
            seed: 7,
            ..DenseSplitConfig::from(&borrowed[..])
        })?;
        assert!(report.changed <= report.reselected);
        assert_eq!(receivers[0].describe()?.size + receivers[1].describe()?.size, COUNT);
        assert_eq!(index.describe()?.size, 0);
        Ok(())
    }

    /// Equal seeds over an unchanged index draw equally, so a larger draw extends a smaller one.
    #[test]
    fn a_larger_sample_draws_a_superset() -> Result<()> {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, DenseMetric::Angular, Scalar::BF16)?;
        let (mut smaller_arena, mut larger_arena) = (Arena::new()?, Arena::new()?);
        let seeded = || DenseSampleConfig {
            seed: 42,
            ..Default::default()
        };
        let smaller = index.on(&executor()).sample(8, &mut smaller_arena, seeded())?;
        let larger = index.on(&executor()).sample(16, &mut larger_arena, seeded())?;
        assert_eq!((smaller.written(), larger.written()), (8, 16));
        assert_eq!(larger.coordinates()?.len(), 16);
        assert_eq!(larger.parents().queries(), 16);
        let drawn = larger.keys()?;
        assert!(smaller.keys()?.iter().all(|key| drawn.contains(key)));
        Ok(())
    }

    /// The audit judges every entry published when it started, and no search reaches more.
    #[test]
    fn audits_how_much_of_the_graph_a_search_reaches() -> Result<()> {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, DenseMetric::Angular, Scalar::BF16)?;
        let reachability = index.on(&executor()).health()?;
        assert_eq!(reachability.audited, COUNT);
        assert!(reachability.reachable != 0 && reachability.reachable <= COUNT);
        assert!(reachability.sweeps != 0, "a populated graph takes at least one level");
        Ok(())
    }

    /// Every row's pairs land in one ragged run, and `wanted` cuts each row while `required`
    /// still counts what the walks scored; the offsets land in the arena beside the run.
    #[test]
    fn multiplication_reports_every_scored_pair() -> Result<()> {
        const DIMENSIONS: usize = 4;
        const ROWS: usize = 3;
        const WANTED: usize = 2;
        // A product is a logit, so the linear-algebra verbs take the inner kind alone.
        let index = empty_index(DIMENSIONS, DenseMetric::Inner, Scalar::BF16)?;
        let slab = seeded_rows(ROWS, DIMENSIONS, SEED);
        index.insert(sequential_keys(ROWS), &rows(&slab, DIMENSIONS)?)?;

        let left = rows(&slab, DIMENSIONS)?;
        let mut arena = Arena::new()?;
        let every = index.multiply(&left, 0, &mut arena)?;
        assert!(
            every.iter()?.all(|row| row.keys.len() == ROWS),
            "every row scores every column"
        );

        let executor = executor();
        let config = DenseMultiplyConfig {
            expansion_factor: 2 * ROWS,
            ..DenseMultiplyConfig::from(WANTED)
        };
        let cut = index.multiply(&left, config, &mut arena)?;
        assert_eq!(cut.queries(), ROWS);
        assert!(cut.iter()?.all(|row| row.keys.len() == WANTED));
        assert_eq!(cut.required(), ROWS * ROWS, "the count outruns the pairs kept");
        assert_eq!(cut.query(ROWS).unwrap_err().status, Status::InvalidArgument);

        let mut placed = Arena::new_in(&executor)?;
        let same = index.on(&executor).multiply(&left, WANTED, &mut placed)?;
        assert!(same.iter()?.all(|row| row.keys.len() == WANTED));
        Ok(())
    }

    /// The outputs are sized over documents rather than rows, which is the one piece of
    /// arithmetic this door does that the batch search does not.
    #[test]
    fn late_interaction_scores_over_documents() -> Result<()> {
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
        let index = DenseIndex::init_ram(&options, &[])?;
        index.reserve(DOCUMENTS * TOKENS)?;

        let slab = seeded_rows(DOCUMENTS * TOKENS, DIMENSIONS, SEED);
        let mut keys = Vec::with_capacity(DOCUMENTS * TOKENS);
        for document in 0..DOCUMENTS {
            keys.extend(core::iter::repeat_n(Key::from(document as u64), TOKENS));
        }
        index.insert(&keys, &rows(&slab, DIMENSIONS)?)?;

        let tokens = rows(&slab[..QUERIES * TOKENS * DIMENSIONS], DIMENSIONS)?;
        let executor = executor();
        let mut arena = Arena::new_in(&executor)?;
        let matches = index.on(&executor).maxsim(&tokens, TOKENS, DOCUMENTS, &mut arena)?;
        assert_eq!(matches.queries(), QUERIES);
        assert!(
            matches.iter()?.all(|query| query.keys.len() == DOCUMENTS),
            "every document is scored, once per query"
        );
        let bounded: Vec<usize> = matches.iter()?.map(|query| query.keys.len()).collect();
        let every = index.maxsim(&tokens, TOKENS, 0, &mut arena)?;
        assert!(
            every.iter()?.zip(bounded).all(|(query, kept)| query.keys.len() >= kept),
            "no bound lands every key a query scored"
        );
        assert_eq!(
            every.required(),
            every.iter()?.map(|query| query.keys.len()).sum::<usize>()
        );
        assert!(matches!(
            index.maxsim(&tokens, TOKENS + 1, DOCUMENTS, &mut arena),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        Ok(())
    }

    /// A graph this small walks every key, so each head's output and carry are exact attention's,
    /// with one head per walk and with two sharing it; a key left without a row, or rows of another
    /// scalar type than the index stores, refuse.
    #[test]
    fn attention_is_exact_over_every_key() -> Result<()> {
        const DIMENSIONS: usize = 4;
        const ENTRIES: usize = 16;
        const TOKENS: usize = 2;
        let index = empty_index(DIMENSIONS, DenseMetric::Inner, Scalar::BF16)?;
        // Rounding every row through bf16 first lets the oracle score exactly what the index stores.
        let in_bf16 =
            |slab: Vec<f32>| -> Vec<f32> { slab.into_iter().map(|value| bf16::from_f32(value).to_f32()).collect() };
        let slab = in_bf16(seeded_rows(ENTRIES, DIMENSIONS, SEED));
        index.insert(sequential_keys(ENTRIES), &rows(&slab, DIMENSIONS)?)?;
        let values = in_bf16(seeded_rows(ENTRIES, DIMENSIONS, SEED + 1));
        let stored_values: Vec<bf16> = values.iter().map(|value| bf16::from_f32(*value)).collect();
        fn row_of<Value>(table: &[Value], entry: usize) -> &[Value] { &table[entry * DIMENSIONS..][..DIMENSIONS] }
        let value_lookup: &ValueLookup<bf16> = &|key| {
            u64::try_from(key)
                .ok()
                .map(|entry| row_of(&stored_values, entry as usize))
        };

        let executor = executor();
        let mut arena = Arena::new_in(&executor)?;
        for head_count in [1, 2] {
            let queries = in_bf16(seeded_rows(TOKENS * head_count, DIMENSIONS, SEED + 2));
            let mut carry = vec![DenseSoftmaxState::default(); TOKENS * head_count];
            let outputs = index.on(&executor).attend(
                &rows(&queries, DIMENSIONS)?,
                value_lookup,
                DenseAttendConfig {
                    expansion_factor: 2 * ENTRIES,
                    carry: Some(&mut carry),
                    ..DenseAttendConfig::from(head_count)
                },
                &mut arena,
            )?;
            assert_eq!(outputs.view()?.shape(), &[TOKENS * head_count, DIMENSIONS]);
            let attended = flattened(outputs.view()?);
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
                    assert!(close(f64::from(attended[query * DIMENSIONS + column]), expected, 1e-3));
                }
                assert!(close(f64::from(carry[query].maximum_score), maximum, 1e-3));
                assert!(close(f64::from(carry[query].partition), partition, 1e-3));
            }
        }

        let refused = index.attend(
            &rows(&slab[..DIMENSIONS], DIMENSIONS)?,
            &|_| None::<&[bf16]>,
            1,
            &mut arena,
        );
        assert!(matches!(
            refused,
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));

        let wide = vec![0.0f64; DIMENSIONS];
        let mistyped = index.attend(
            &rows(&slab[..DIMENSIONS], DIMENSIONS)?,
            &|_| Some(wide.as_slice()),
            1,
            &mut arena,
        );
        assert!(
            matches!(
                mistyped,
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "value rows are read in the index's own quantization"
        );
        Ok(())
    }

    /// A product and a softmax both need logits, so a graph frozen under a distance refuses.
    #[test]
    fn linear_algebra_refuses_a_distance_kind() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let (index, _, slab) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        assert!(matches!(
            index
                .multiply(&rows(&slab[..DIMENSIONS], DIMENSIONS)?, 0, &mut Arena::new()?)
                .map(|products| products.queries()),
            Err(Error {
                status: Status::ConfigMismatch,
                ..
            })
        ));
        Ok(())
    }

    // endregion: Refinement and Linear Algebra

    // region: Serialization

    /// The loaded index adopts its geometry from the checkpoint's own headers.
    #[test]
    fn saves_and_loads_a_file() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let path = scratch_path("saves_and_loads_a_file");
        let (index, _, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        index.on(&executor()).save(&path)?;

        let restored = DenseIndex::load_ram(&path)?.describe()?;
        assert_eq!(restored.metadata.dimensions, DIMENSIONS);
        assert_eq!(restored.size, 32);

        // A view serves the same checkpoint in place and refuses to change it, where RAM serves views at all.
        match DenseIndex::view_ram(&path) {
            Ok(viewed) => {
                assert_eq!(viewed.describe()?.size, 32);
                assert!(matches!(
                    viewed.reserve(64),
                    Err(Error {
                        status: Status::ReadOnly,
                        ..
                    })
                ));
            }
            Err(error) => assert_eq!(error.status, Status::Unsupported),
        }
        remove_quietly(&path);
        Ok(())
    }

    #[test]
    fn saves_and_loads_a_buffer() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, _, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let mut arena = Arena::new()?;
        let saved = index.save_buffer(&mut arena)?;
        let monolith = saved.view()?;
        assert!(!monolith.is_empty());

        let restored = DenseIndex::load_ram_buffer(monolith)?;
        assert_eq!(restored.describe()?.size, 32);

        let executor = executor();
        let truncated = DenseLoadRamBufferConfig {
            executor: Some(&executor),
            ..DenseLoadRamBufferConfig::from(&monolith[..monolith.len() - 1])
        };
        assert!(
            DenseIndex::load_ram_buffer(truncated).is_err(),
            "a short buffer must not be read"
        );
        Ok(())
    }

    #[test]
    fn reads_metadata_without_loading() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let path = scratch_path("reads_metadata_without_loading");
        let (index, _, _) = populated_index(8, DIMENSIONS, DenseMetric::Euclidean, Scalar::I8)?;
        index.save(&path)?;

        let described = DenseIndex::metadata(&path)?;
        assert_eq!(described.dimensions, DIMENSIONS);
        assert_eq!(described.metric, DenseMetric::Euclidean);
        assert_eq!(described.quantization, Scalar::I8);

        let mut arena = Arena::new()?;
        let buffer = index.save_buffer(DenseSaveBufferConfig {
            progress: Some(&mut |_, _| ControlFlow::Continue(())),
            ..DenseSaveBufferConfig::from(&mut arena)
        })?;
        assert_eq!(DenseIndex::metadata_buffer(buffer.view()?)?.dimensions, DIMENSIONS);
        remove_quietly(&path);
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn rejects_a_non_utf8_path() -> Result<()> {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let index = empty_index(8, DenseMetric::Angular, Scalar::BF16)?;
        let path = std::path::Path::new(OsStr::from_bytes(&[0xff, 0xfe]));
        assert!(
            matches!(index.save(path), Err(Error { status: Status::Io, .. })),
            "the C interface exchanges UTF-8 bytes"
        );
        Ok(())
    }

    // endregion: Serialization

    // region: Absorbing and Iteration

    #[test]
    fn absorbs_a_donor_index() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, keys, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let donor = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let donor_keys: Vec<Key> = (100..132u64).map(Key::from).collect();
        let donor_slab = seeded_rows(32, DIMENSIONS, SEED + 1);
        donor.insert(&donor_keys, &rows(&donor_slab, DIMENSIONS)?)?;

        // The donors are consumed by value, so an absorbed handle cannot be used again.
        index.on(&executor()).absorb([donor])?;
        assert_eq!(index.describe()?.size, 64);
        assert!(index.contains(keys[0])? && index.contains(donor_keys[0])?);
        Ok(())
    }

    #[test]
    fn walks_entries_in_batches() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 20;
        const BUFFERED: usize = 8;
        let (index, keys, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let executor = executor();
        let receiver = index.on(&executor);
        let mut cursor = receiver.cursor(&[])?;
        let mut arena = Arena::new_in(&executor)?;
        let mut seen = HashSet::new();
        let mut batches = Vec::new();
        loop {
            let batch = cursor.next_keys(BUFFERED, &mut arena)?;
            if batch.written() == 0 {
                break;
            }
            batches.push(batch.written());
            seen.extend(batch.keys()?.iter().copied());
        }

        assert_eq!(batches, vec![8, 8, 4], "the batch boundary is crossed twice");
        assert_eq!(seen, keys.iter().copied().collect::<HashSet<Key>>());
        Ok(())
    }

    #[test]
    fn walks_entries_by_keys() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, keys, _) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let mut cursor = index.cursor(&[])?;
        let mut arena = Arena::new()?;
        let batch = cursor.next_keys(8, &mut arena)?;
        assert_eq!(batch.written(), 4);
        assert_eq!(
            batch.keys()?.iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
        assert_eq!(cursor.next_keys(8, &mut arena)?.written(), 0, "the walk is exhausted");
        Ok(())
    }

    #[test]
    fn streams_matches_in_pulls() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 200;
        const PULL: usize = 64;
        let (index, _, slab) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let query = rows(&slab[..DIMENSIONS], DIMENSIONS)?;
        let mut arena = Arena::new()?;
        let mut stream = match index.search_stream(query) {
            Err(error) if error.status == Status::Unsupported => return Ok(()),
            opened => opened?,
        };
        let first = stream.next(5, &mut arena)?;
        assert_eq!(first.queries(), 1, "a pull is one query of a search");
        assert_eq!(
            first.query(0)?.keys.len(),
            5,
            "a pull hands the caller as many as it asked for"
        );

        let mut drained = 0;
        loop {
            let pulled = stream.next(PULL, &mut arena)?.query(0)?.keys.len();
            if pulled == 0 {
                break;
            }
            drained += pulled;
        }
        assert!(drained > PULL, "the walk spans several pulls");
        assert!(
            stream.next(PULL, &mut arena)?.query(0)?.keys.is_empty(),
            "a drained stream stays drained"
        );

        let mut widened = index.search_stream(DenseSearchStreamInitConfig {
            expansion_factor: 64,
            ..DenseSearchStreamInitConfig::from(query)
        })?;
        assert!(!widened.next(PULL, &mut arena)?.query(0)?.keys.is_empty());

        assert!(
            matches!(
                index.search_stream(rows(&slab[..2 * DIMENSIONS], DIMENSIONS)?),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "a stream walks one row, never the first of many"
        );
        Ok(())
    }

    // endregion: Absorbing and Iteration

    // region: Progress and Cancellation

    #[test]
    fn a_reported_batch_counts_up_to_the_rows_it_was_given() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        index.reserve(COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let mut watch = Watch::new(Patience::Complete);
        index.on(&executor()).insert(
            &keys,
            DenseInsertConfig {
                progress: Some(&mut |finished, total| watch.saw(finished, total)),
                ..DenseInsertConfig::from(&rows(&slab, DIMENSIONS)?)
            },
        )?;

        assert_eq!(index.describe()?.size, COUNT);
        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (COUNT, COUNT), "the last report names the whole batch");
        Ok(())
    }

    /// Every entry is claimed before the first report, so a stop is ignored and the batch publishes
    /// whole.
    #[test]
    fn a_stopped_batch_publishes_everything() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        index.reserve(COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let mut watch = Watch::new(Patience::StopAtFirstReport);
        index.insert(
            &keys,
            DenseInsertConfig {
                progress: Some(&mut |finished, total| watch.saw(finished, total)),
                ..DenseInsertConfig::from(&rows(&slab, DIMENSIONS)?)
            },
        )?;
        assert_eq!(index.describe()?.size, COUNT);
        Ok(())
    }

    #[test]
    fn a_reported_search_counts_up_to_the_queries_it_was_given() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const QUERIES: usize = 4;
        const WANTED: usize = 3;
        let (index, _, slab) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let mut watch = Watch::new(Patience::Complete);
        index.on(&executor()).search(
            &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?,
            DenseSearchConfig {
                progress: Some(&mut |finished, total| watch.saw(finished, total)),
                ..DenseSearchConfig::from(WANTED)
            },
            &mut Arena::new()?,
        )?;

        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (QUERIES, QUERIES));
        Ok(())
    }

    /// Stopping a sweep is its time budget: the call still re-links its islands, so the graph it
    /// leaves stays whole, and answers `Cancelled`.
    #[test]
    fn a_stopped_refinement_still_heals() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 32;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

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
        assert_eq!(index.on(&executor()).health()?.audited, COUNT);
        Ok(())
    }

    #[test]
    fn a_reported_checkpoint_counts_what_it_wrote_and_placed() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 32;
        let path = scratch_path("a_reported_checkpoint");
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let mut writing = Watch::new(Patience::Complete);
        index.on(&executor()).save(DenseSaveConfig {
            progress: Some(&mut |finished, total| writing.saw(finished, total)),
            ..DenseSaveConfig::from(&path)
        })?;
        assert!(writing.ordered, "a report never counts down");
        assert_eq!(writing.last.0, writing.last.1, "the last report names a finished write");
        assert!(writing.last.1 != 0);

        let mut placing = Watch::new(Patience::Complete);
        let restored = DenseIndex::load_ram(DenseLoadRamConfig {
            executor: Some(&executor()),
            progress: Some(&mut |finished, total| placing.saw(finished, total)),
            ..DenseLoadRamConfig::from(&path)
        })?;
        assert!(placing.ordered, "a report never counts down");
        assert_eq!(placing.last.0, placing.last.1, "the last report names a finished read");
        assert_eq!(restored.describe()?.size, COUNT);

        remove_quietly(&path);
        Ok(())
    }

    /// The handler counts the nodes the closing refinement re-selected, never the donors moved.
    #[test]
    fn a_reported_absorb_counts_its_closing_refinement() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 16;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let donors = [
            populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?.0,
            populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?.0,
        ];
        let mut watch = Watch::new(Patience::Complete);
        let report = index.on(&executor()).absorb(DenseAbsorbConfig {
            progress: Some(&mut |finished, total| watch.saw(finished, total)),
            ..DenseAbsorbConfig::from(donors)
        })?;

        assert!(watch.ordered, "a report never counts down");
        assert_eq!(
            watch.last.0, watch.last.1,
            "the last report names a finished refinement"
        );
        assert!(report.changed <= report.reselected);
        assert_eq!(index.describe()?.size, COUNT * 3);
        Ok(())
    }

    /// A stop cuts only the closing refinement, so every donor has moved - they were handed over
    /// by value, so a leak or a double free would surface here too.
    #[test]
    fn a_stopped_absorb_keeps_the_donors_it_absorbed() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 16;
        const DONORS: usize = 2;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;

        let donors = vec![
            populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?.0,
            populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?.0,
        ];
        let mut watch = Watch::new(Patience::StopAtFirstReport);
        let absorbed = index.absorb(DenseAbsorbConfig {
            progress: Some(&mut |finished, total| watch.saw(finished, total)),
            ..DenseAbsorbConfig::from(donors)
        });

        match absorbed {
            Ok(report) => assert!(report.changed <= report.reselected, "an unstopped absorb reports it"),
            Err(failure) => assert_eq!(failure.status, Status::Cancelled),
        }
        assert_eq!(index.describe()?.size, COUNT * (1 + DONORS));
        Ok(())
    }

    // endregion: Progress and Cancellation

    // region: Concurrency

    const THREADS: usize = 4;

    /// The batch is the unit of parallelism: the executor fans one call out over its threads,
    /// so the caller never spawns any of its own.
    #[test]
    fn builds_and_searches_on_a_shared_executor() -> Result<()> {
        const DIMENSIONS: usize = 128;
        const COUNT: usize = 1000;
        const WANTED: usize = 10;

        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        index.reserve(COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let executor = Executor::for_ram(&[], None, THREADS, 0)?;
        let receiver = index.on(&executor);
        let vectors = rows(&slab, DIMENSIONS)?;
        receiver.insert(&keys, &vectors)?;
        assert_eq!(receiver.describe()?.size, COUNT);

        let config = DenseSearchConfig {
            expansion_factor: 64,
            ..DenseSearchConfig::from(WANTED)
        };
        let mut arena = Arena::new_in(&executor)?;
        let matches = receiver.search(&vectors, config, &mut arena)?;
        assert!(
            matches.iter()?.all(|query| query.keys.len() == WANTED),
            "every query fills its own output"
        );
        Ok(())
    }

    /// The index itself is the blocking receiver on the calling thread, with the same results.
    #[test]
    fn runs_on_the_calling_thread() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, _, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let query = rows(&slab[..DIMENSIONS], DIMENSIONS)?;
        let (mut first, mut second) = (Arena::new()?, Arena::new()?);
        let on_this_thread = index.search(&query, 4, &mut first)?;
        let on_the_null_executor = index.on(&Executor::default()).search(&query, 4, &mut second)?;
        assert_eq!(on_this_thread.query(0)?.keys.len(), 4);
        assert_eq!(on_this_thread.query(0), on_the_null_executor.query(0));
        Ok(())
    }

    #[test]
    fn searches_while_inserting() -> Result<()> {
        const DIMENSIONS: usize = 64;
        const HALF: usize = 500;
        const WANTED: usize = 4;

        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        index.reserve(2 * HALF)?;
        let keys = sequential_keys(2 * HALF);
        let slab = seeded_rows(2 * HALF, DIMENSIONS, SEED);

        // The first half lands serially, so the reader below traverses a populated graph.
        index.insert(&keys[..HALF], &rows(&slab[..HALF * DIMENSIONS], DIMENSIONS)?)?;

        // Calls on one executor run in order, so genuinely overlapping calls need one apiece -
        // the writer's and the reader's, on two threads sharing the handle.
        let writer = Executor::for_ram(&[], None, THREADS, 0)?;
        let reader = Executor::for_ram(&[], None, THREADS, 0)?;
        let known: HashSet<Key> = keys.iter().copied().collect();
        let strangers = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            scope.spawn(|| {
                let writer = writer;
                let vectors = rows(&slab[HALF * DIMENSIONS..], DIMENSIONS).unwrap();
                index
                    .on(&writer)
                    .insert(&keys[HALF..], &vectors)
                    .expect("the racing batch must insert");
            });
            scope.spawn(|| {
                let reader = reader;
                let mut arena = Arena::new_in(&reader).expect("the arena must initialize");
                let queries = rows(&slab[..HALF * DIMENSIONS], DIMENSIONS).unwrap();
                let config = DenseSearchConfig {
                    expansion_factor: 32,
                    ..DenseSearchConfig::from(WANTED)
                };
                let matches = index
                    .on(&reader)
                    .search(&queries, config, &mut arena)
                    .expect("the racing search must resolve");
                // Whether a racing insert is visible is what the watermark leaves unspecified;
                // that no key outside the dataset appears is what it guarantees.
                for query in matches.iter().expect("the arena is host memory") {
                    if query.keys.iter().any(|key| !known.contains(key)) {
                        strangers.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        });

        assert_eq!(
            strangers.load(Ordering::Relaxed),
            0,
            "searches never see unpublished entries"
        );
        assert_eq!(index.describe()?.size, 2 * HALF);
        Ok(())
    }

    // endregion: Concurrency

    // region: Deferred Calls

    /// A deferred search owns its queries, borrows its arena as the blocking one does, and answers
    /// what the blocking one does once waited on.
    #[test]
    fn a_deferred_search_matches_the_blocking_one() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const QUERIES: usize = 4;
        const WANTED: usize = 5;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let queries = Matrix::<f32>::from_slice(&slab[..QUERIES * DIMENSIONS], &[QUERIES, DIMENSIONS]).unwrap();

        let executor = executor();
        let mut arena = Arena::new_in(&executor)?;
        let call = index.deferred(&executor).search(queries, WANTED, &mut arena)?;
        requires_send::<Deferred<'static, DenseSearch<'static, 'static, Matrix<f32>>>>();
        requires_send::<Matches<'static>>();
        assert_eq!(call.status(), Err(Status::Deferred));
        let deferred = call.wait()?;

        let mut blocking_arena = Arena::new()?;
        let blocking = index.search(
            &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?,
            WANTED,
            &mut blocking_arena,
        )?;
        assert_eq!(deferred.queries(), QUERIES);
        for query in 0..QUERIES {
            assert_eq!(deferred.query(query), blocking.query(query));
        }

        let queries = Matrix::<f32>::from_slice(&slab[..DIMENSIONS], &[1, DIMENSIONS]).unwrap();
        let again = index.deferred(&executor).search(queries, WANTED, &mut arena)?;
        assert_eq!(again.wait()?.query(0), blocking.query(0));
        Ok(())
    }

    /// A leaked deferred call leaves its arena busy rather than unsafe: a call naming it finishes
    /// contended, and dropping it waits for the leaked call.
    #[test]
    fn a_leaked_deferred_call_keeps_its_arena_busy() -> Result<()> {
        use std::sync::atomic::AtomicBool;

        static RELEASED: AtomicBool = AtomicBool::new(false);
        const DIMENSIONS: usize = 8;
        let (index, keys, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let queries = Matrix::<f32>::from_slice(&slab[..DIMENSIONS], &[1, DIMENSIONS]).unwrap();
        // The handler holds the call until released, so it is still running when the arena is named again.
        let gate: &'static mut ProgressCallback<'static> = Box::leak(Box::new(|_, _| {
            while !RELEASED.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            ControlFlow::Continue(())
        }));

        let executor = executor();
        let mut arena = Arena::new_in(&executor)?;
        let config = DenseSearchConfig {
            progress: Some(gate),
            ..DenseSearchConfig::from(3)
        };
        core::mem::forget(index.deferred(&executor).search(queries, config, &mut arena)?);
        assert!(matches!(
            index.get::<f32>(keys[0], &mut arena),
            Err(Error {
                status: Status::Contended,
                ..
            })
        ));
        RELEASED.store(true, Ordering::Release);
        drop(arena);
        Ok(())
    }

    /// Awaiting a deferred call stays pending while it is not final and answers what `wait` would.
    #[test]
    fn an_awaited_call_is_pending_until_final() -> Result<()> {
        use core::future::{Future, IntoFuture};
        use core::pin::pin;
        use core::task::{Context, Poll, Waker};

        const DIMENSIONS: usize = 16;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let queries = Matrix::<f32>::from_slice(&slab[..DIMENSIONS], &[1, DIMENSIONS]).unwrap();
        let executor = executor();
        let mut arena = Arena::new_in(&executor)?;
        let mut awaited = pin!(index.deferred(&executor).search(queries, 3, &mut arena)?.into_future());
        let mut context = Context::from_waker(Waker::noop());
        assert!(awaited.as_mut().poll(&mut context).is_pending());
        executor.drain(Mood::Parked)?;
        let Poll::Ready(matches) = awaited.as_mut().poll(&mut context) else {
            panic!("a drained call must be final");
        };
        assert_eq!(
            matches?.query(0),
            index
                .search(&rows(&slab[..DIMENSIONS], DIMENSIONS)?, 3, &mut Arena::new()?)?
                .query(0)
        );
        assert!(matches!(awaited.as_mut().poll(&mut context), Poll::Ready(Err(_))));
        Ok(())
    }

    /// Deferred calls in flight together each borrow their own arena, which their results borrow in turn.
    #[test]
    fn deferred_calls_borrow_their_arenas() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let (index, keys, slab) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let executor = executor();
        let receiver = index.deferred(&executor);

        let (mut first, mut second) = (Arena::new_in(&executor)?, Arena::new_in(&executor)?);
        let found = receiver.get::<f32>(keys[2], &mut first)?;
        let buffer = receiver.save_buffer(&mut second)?;
        let counted = receiver.count(keys[3])?;
        assert!(close_rows(
            &flattened(found.wait()?.view()?),
            &slab[DIMENSIONS * 2..DIMENSIONS * 3]
        ));
        assert_eq!(
            DenseIndex::metadata_buffer(buffer.wait()?.view()?)?.dimensions,
            DIMENSIONS
        );
        assert_eq!(counted.wait()?, 1);
        Ok(())
    }

    /// Dropping a call that is not yet final waits for it, then frees what it owned.
    #[test]
    fn an_unfinished_deferred_call_drops_by_waiting() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);
        let vectors = Matrix::<f32>::from_slice(&slab, &[COUNT, DIMENSIONS]).unwrap();

        let executor = executor();
        let call = index.deferred(&executor).insert(sequential_keys(COUNT), vectors)?;
        assert_eq!(call.status(), Err(Status::Deferred));
        drop(call);
        assert_eq!(index.describe()?.size, COUNT);
        Ok(())
    }

    /// A submission finding the executor's deferred table full is refused rather than waited for.
    #[test]
    fn a_full_deferred_table_answers_contended() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 16;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);
        let vectors = || Matrix::<f32>::from_slice(&slab, &[COUNT, DIMENSIONS]).unwrap();

        let executor = Executor::for_ram(&[], None, 1, 1)?;
        let receiver = index.deferred(&executor);
        let first = receiver.insert(sequential_keys(COUNT), vectors())?;
        let second = receiver.insert(sequential_keys(COUNT), vectors())?;
        assert_eq!(second.status(), Err(Status::Contended));
        first.wait()?;
        assert_eq!(index.describe()?.size, COUNT);
        Ok(())
    }

    // endregion: Deferred Calls
}

// endregion: Tests
