//! Vector Search over equidimensional dense embeddings - the Rust twin of
//! `include/usearch/dense_index.h`.
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
use std::borrow::BorrowMut;
use std::path::Path;

use numkong::{MatrixView, TensorMut};

use crate::types::{
    arena_slice, check, fail, lower_predicate, lower_progress, matches_span, nvme_options, path_view, path_views,
    ram_options, read_spelling, span_of, spelling, static_string, usearch_arena_t, usearch_bytes_view_t,
    usearch_completion_blocking_k, usearch_completion_t, usearch_distance_t, usearch_executor_t, usearch_key_t,
    usearch_keys_span_t, usearch_keys_view_t, usearch_matches_span_t, usearch_nvme_options_t, usearch_outcome_t,
    usearch_predicate_t, usearch_progress_callback_t, usearch_ram_options_t, usearch_rows_span_t, usearch_rows_view_t,
    usearch_scalar_t, usearch_vram_options_t, values_of, vram_options, Accepts, Arena, Blocking, Calling, Completion,
    Deferring, Distance, Element, Error, Executor, Exported, Handles, Key, Match, Matches, MatchesRow, MatchesSpan, On,
    Predicate, ProgressCallback, Ranked, Result, RowsSpan, RowsView, Scalar, Status,
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
    pub values_per_key: usize,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
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
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
}

#[repr(C)]
pub struct usearch_dense_load_ram_buffer_t {
    pub index: *mut usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub buffer: usearch_bytes_view_t,
    pub ram: usearch_ram_options_t,
    pub values_per_key: usize,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
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
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
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

pub type usearch_dense_traversal_stats_t = DenseTraversalStats;

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
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
    pub stats: *mut usearch_dense_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_dense_refine_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub changed: *mut usize,
    pub expansion_factor: usize,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
    pub stats: *mut usearch_dense_traversal_stats_t,
}

pub type usearch_dense_reachability_t = DenseReachability;

#[repr(C)]
pub struct usearch_dense_health_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub reachability: *mut usearch_dense_reachability_t,
}

#[repr(C)]
pub struct usearch_dense_search_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub queries: usearch_rows_view_t,
    pub matches: usearch_matches_span_t,
    pub expansion_factor: usize,
    pub predicate: usearch_predicate_t,
    pub predicate_state: *mut c_void,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
    pub stats: *mut usearch_dense_traversal_stats_t,
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
    pub scalar: usearch_scalar_t,
    pub vectors: *mut *mut c_void,
    pub written: *mut usize,
    pub stride: *mut usize,
    pub columns: *mut usize,
    pub arena: *mut usearch_arena_t,
}

#[repr(C)]
pub struct usearch_dense_merge_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub donors: *mut usearch_dense_index_t,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
}

#[repr(C)]
pub struct usearch_dense_cursor_init_t {
    pub cursor: *mut usearch_dense_cursor_t,
    pub index: usearch_dense_index_t,
    pub outcome: *mut usearch_outcome_t,
}

#[repr(C)]
pub struct usearch_dense_cursor_next_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub cursor: usearch_dense_cursor_t,
    pub keys: usearch_keys_span_t,
    pub vectors: usearch_rows_span_t,
    pub written: *mut usize,
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
    pub matches: usearch_matches_span_t,
}

#[repr(C)]
pub struct usearch_dense_save_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub path: usearch_bytes_view_t,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
}

#[repr(C)]
pub struct usearch_dense_save_buffer_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub bytes: *mut *mut u8,
    pub length: *mut usize,
    pub arena: *mut usearch_arena_t,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
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
    pub capabilities: *const c_char,
}

impl Default for usearch_dense_index_description_t {
    fn default() -> Self {
        usearch_dense_index_description_t {
            metadata: usearch_dense_metadata_t::default(),
            size: 0,
            capacity: 0,
            memory_usage: 0,
            capabilities: ptr::null(),
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
    pub offsets: *mut usize,
    pub keys: *mut *mut usearch_key_t,
    pub scores: *mut *mut usearch_distance_t,
    pub arena: *mut usearch_arena_t,
    pub written: *mut usize,
    pub required: *mut usize,
    pub wanted: usize,
    pub expansion_factor: usize,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
    pub stats: *mut usearch_dense_traversal_stats_t,
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
    pub outputs: usearch_rows_span_t,
    pub carry: *mut usearch_dense_softmax_state_t,
    pub expansion_factor: usize,
    pub score_scale: f32,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
    pub stats: *mut usearch_dense_traversal_stats_t,
}

#[repr(C)]
pub struct usearch_dense_maxsim_t {
    pub index: usearch_dense_index_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub completion: usearch_completion_t,
    pub count: usize,
    pub queries: usearch_rows_view_t,
    pub rows_per_query: usize,
    pub matches: usearch_matches_span_t,
    pub expansion_factor: usize,
    pub progress: usearch_progress_callback_t,
    pub progress_state: *mut c_void,
    pub stats: *mut usearch_dense_traversal_stats_t,
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
    pub fn usearch_dense_merge(arguments: *mut usearch_dense_merge_t);

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
///
/// Mirrors `usearch_dense_metric_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum DenseMetric {
    Inner = 1,
    #[default]
    Angular = 2,
    Euclidean = 3,
    Hamming = 4,
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
    /// Calls `usearch_dense_metric_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_dense_metric_name, *self as c_int))
    }
}

impl FromStr for DenseMetric {
    type Err = Error;

    /// Calls `usearch_dense_metric_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(
            text,
            usearch_dense_metric_from_name,
            usearch_dense_metric_name,
            DenseMetric::from_raw,
        )
    }
}

/// Construction options of a dense index, whose stored rows round up to the SIMD step quantum so
/// every kernel runs tail-free.
///
/// Rust-only; the fields `usearch_dense_init_ram_t` and `usearch_dense_init_vram_t` share.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct DenseOptions {
    /// The number of indexed dimensions; zero leaves the geometry unfrozen, so the index admits
    /// no rows.
    pub dimensions: usize,
    /// The metric kind used for distance calculations.
    pub metric: DenseMetric,
    /// The scalar type vectors are stored and quantized into.
    pub quantization: Scalar,
    /// The connections-per-node bound of the graph; zero derives.
    pub connectivity: usize,
    /// Upper bound on entries sharing one key; zero derives to one - unique keys.
    pub values_per_key: usize,
}

impl Default for DenseOptions {
    fn default() -> Self {
        DenseOptions {
            dimensions: 0,
            metric: DenseMetric::Angular,
            quantization: Scalar::BF16,
            connectivity: 0,
            values_per_key: 0,
        }
    }
}

/// The geometry and metric a checkpoint froze, readable without loading it.
///
/// `values_per_key` and the placement are the reader's to choose when loading, so a checkpoint
/// fixes neither.
///
/// Mirrors `usearch_dense_metadata_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct DenseMetadata {
    /// Scalars per stored vector.
    pub dimensions: usize,
    /// The metric kind the checkpoint scores with.
    pub metric: DenseMetric,
    /// The scalar type its rows are stored and quantized into.
    pub quantization: Scalar,
    /// Connections per node the graph was built with.
    pub connectivity: usize,
}

impl DenseMetadata {
    /// Lifts the C structure, refusing a kind this binding has no name for.
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
///
/// Mirrors `usearch_dense_index_description_t`.
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
    /// The capability codename the index dispatched to on this machine.
    pub capabilities: &'static str,
}

// endregion: Geometry

// region: Configs

/// Traversal telemetry, fully overwritten by the call it is passed to.
///
/// Mirrors `usearch_dense_traversal_stats_t`.
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
}

/// One reachability audit: the population it judged, and how much of it a search reaches.
///
/// Mirrors `usearch_dense_reachability_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct DenseReachability {
    /// Entries published when the audit started; compare `reachable` against this, never a later size.
    pub audited: usize,
    /// Of those, the entries a search reaches; the rest no search finds until a refinement re-links them.
    pub reachable: usize,
    /// Levels the sweep took to converge, a diameter proxy; the deepest shard's on a sharded handle.
    pub sweeps: usize,
}

/// The online-softmax carry of one query row, which folds one attention call into another over
/// the same queries.
///
/// Mirrors `usearch_dense_softmax_state_t`.
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
/// call, and `None` refuses the call.
///
/// Mirrors `usearch_value_lookup_t`.
pub type ValueLookup<'lookup, Value> = dyn Fn(Key) -> Option<&'lookup [Value]> + Sync + 'lookup;

/// A checkpoint path and how it is copied into a new index in host memory; see
/// [`DenseIndex::load_ram`].
///
/// Mirrors `usearch_dense_load_ram_t`.
pub struct DenseLoadRamConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The executor the read fans out over; `None` reads on the calling thread.
    pub executor: Option<&'config Executor>,
    /// NUMA node IDs to place the parts on, part `p` landing whole on shard `p` modulo their count;
    /// empty derives every node the process may use.
    pub memory_domains: &'config [i32],
    /// Upper bound on entries sharing one key, which no checkpoint records; zero derives to one.
    pub values_per_key: usize,
    /// Reports the shards placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for DenseLoadRamConfig<'config> {
    fn from(path: &'config Location) -> Self {
        DenseLoadRamConfig {
            path: path.as_ref(),
            executor: None,
            memory_domains: &[],
            values_per_key: 0,
            progress: None,
        }
    }
}

/// A checkpoint path and how it is uploaded into a new index in device memory; see
/// [`DenseIndex::load_vram`].
///
/// Mirrors `usearch_dense_load_vram_t`.
pub struct DenseLoadVramConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The executor the upload runs on; `None` uploads on the default stream of each device.
    pub executor: Option<&'config Executor>,
    /// CUDA device ordinals to place the parts on, part `p` landing whole on device `p` modulo
    /// their count; empty derives one free device.
    pub devices: &'config [i32],
    /// Upper bound on entries sharing one key, which no checkpoint records; zero derives to one.
    pub values_per_key: usize,
    /// Reports the shards placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for DenseLoadVramConfig<'config> {
    fn from(path: &'config Location) -> Self {
        DenseLoadVramConfig {
            path: path.as_ref(),
            executor: None,
            devices: &[],
            values_per_key: 0,
            progress: None,
        }
    }
}

/// A serialized monolith and how it is copied into a new index in host memory; see
/// [`DenseIndex::load_ram_buffer`].
///
/// Mirrors `usearch_dense_load_ram_buffer_t`.
pub struct DenseLoadRamBufferConfig<'config> {
    /// The monolith's bytes.
    pub buffer: &'config [u8],
    /// The executor the copy fans out over; `None` copies on the calling thread.
    pub executor: Option<&'config Executor>,
    /// NUMA node IDs to place shards on, the monolith landing whole on the first; empty derives
    /// every node the process may use.
    pub memory_domains: &'config [i32],
    /// Upper bound on entries sharing one key, which no checkpoint records; zero derives to one.
    pub values_per_key: usize,
    /// Reports the bytes placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
}

impl<'config> From<&'config [u8]> for DenseLoadRamBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        DenseLoadRamBufferConfig {
            buffer,
            executor: None,
            memory_domains: &[],
            values_per_key: 0,
            progress: None,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for DenseLoadRamBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self { DenseLoadRamBufferConfig::from(&**bytes) }
}

/// A serialized monolith and how it is uploaded into a new index in device memory; see
/// [`DenseIndex::load_vram_buffer`].
///
/// Mirrors `usearch_dense_load_vram_buffer_t`.
pub struct DenseLoadVramBufferConfig<'config> {
    /// The monolith's bytes, in host or device memory.
    pub buffer: &'config [u8],
    /// The executor the upload runs on; `None` uploads on the default stream of each device.
    pub executor: Option<&'config Executor>,
    /// CUDA device ordinals to place shards on, the monolith landing whole on the first; empty
    /// derives one free device.
    pub devices: &'config [i32],
    /// Upper bound on entries sharing one key, which no checkpoint records; zero derives to one.
    pub values_per_key: usize,
    /// Reports the bytes placed and stops wherever it says to; a stopped load creates no index.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
}

impl<'config> From<&'config [u8]> for DenseLoadVramBufferConfig<'config> {
    fn from(buffer: &'config [u8]) -> Self {
        DenseLoadVramBufferConfig {
            buffer,
            executor: None,
            devices: &[],
            values_per_key: 0,
            progress: None,
        }
    }
}

impl<'config, Bytes: Deref<Target = [u8]>> From<&'config Bytes> for DenseLoadVramBufferConfig<'config> {
    fn from(bytes: &'config Bytes) -> Self { DenseLoadVramBufferConfig::from(&**bytes) }
}

/// A checkpoint path and the drives serving it in place; see [`DenseIndex::view_nvme`].
///
/// Mirrors `usearch_dense_view_nvme_t`.
pub struct DenseViewNvmeConfig<'config> {
    /// A `.safegraphs` monolith or a `.safegraphs.json` manifest.
    pub path: &'config Path,
    /// The NVMe executor whose rings read the checkpoint's headers; `None` reads on the calling
    /// thread.
    pub executor: Option<&'config Executor>,
    /// Devices or directories to serve the parts from; empty serves each where its manifest points.
    pub paths: &'config [&'config Path],
}

impl<'config, Location: AsRef<Path> + ?Sized> From<&'config Location> for DenseViewNvmeConfig<'config> {
    fn from(path: &'config Location) -> Self {
        DenseViewNvmeConfig {
            path: path.as_ref(),
            executor: None,
            paths: &[],
        }
    }
}

/// The rows of an insertion and the knobs of its walk; see [`DenseVerbs::insert`].
///
/// Mirrors `usearch_dense_insert_t`.
pub struct DenseInsertConfig<'config, Rows> {
    /// One row per key, borrowed or owned.
    pub vectors: Rows,
    /// The candidate-pool breadth the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Reports the rows inserted; every row is claimed before the first report, so a `Break` is
    /// ignored.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl<Rows: RowsView> From<Rows> for DenseInsertConfig<'_, Rows> {
    fn from(vectors: Rows) -> Self {
        DenseInsertConfig {
            vectors,
            expansion_factor: 0,
            progress: None,
            stats: None,
        }
    }
}

/// The knobs of [`DenseVerbs::refine`].
///
/// Mirrors `usearch_dense_refine_t`.
#[derive(Default)]
pub struct DenseRefineConfig<'config> {
    /// The candidate-pool breadth the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Reports the nodes swept and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

/// Where a search lands its matches and the knobs of its walk; see [`DenseVerbs::search`].
///
/// Mirrors `usearch_dense_search_t`.
pub struct DenseSearchConfig<'config, Destination = Matches> {
    /// [`Matches`] the call allocates, or a `&mut MatchesSpan` of caller-owned buffers.
    pub matches: Destination,
    /// Matches kept per query, at most the destination's width.
    pub wanted: usize,
    /// The candidate-pool breadth the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Admits only the keys it approves; the rest still route the walk, so recall does not
    /// collapse at low selectivity. A VRAM handle refuses one with [`Status::Device`].
    pub predicate: Option<&'config Predicate<'config>>,
    /// Reports the queries resolved and stops wherever it says to; a stopped search reports
    /// [`Status::Cancelled`] and leaves every output row indeterminate.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl From<usize> for DenseSearchConfig<'_, Matches> {
    fn from(wanted: usize) -> Self {
        DenseSearchConfig {
            matches: Matches::default(),
            wanted,
            expansion_factor: 0,
            predicate: None,
            progress: None,
            stats: None,
        }
    }
}

impl<'span, 'buffers> From<&'span mut MatchesSpan<'buffers>>
    for DenseSearchConfig<'_, &'span mut MatchesSpan<'buffers>>
{
    fn from(matches: &'span mut MatchesSpan<'buffers>) -> Self {
        DenseSearchConfig {
            wanted: matches.wanted,
            matches,
            expansion_factor: 0,
            predicate: None,
            progress: None,
            stats: None,
        }
    }
}

/// Where a late-interaction search lands its matches and the knobs of its walk; see
/// [`DenseVerbs::maxsim`].
///
/// Mirrors `usearch_dense_maxsim_t`.
pub struct DenseMaxsimConfig<'config, Destination = Matches> {
    /// [`Matches`] the call allocates, or a `&mut MatchesSpan` of caller-owned buffers.
    pub matches: Destination,
    /// Documents kept per query, at most the destination's width.
    pub wanted: usize,
    /// The candidate-pool breadth the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// Reports the queries resolved and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl From<usize> for DenseMaxsimConfig<'_, Matches> {
    fn from(wanted: usize) -> Self {
        DenseMaxsimConfig {
            matches: Matches::default(),
            wanted,
            expansion_factor: 0,
            progress: None,
            stats: None,
        }
    }
}

impl<'span, 'buffers> From<&'span mut MatchesSpan<'buffers>>
    for DenseMaxsimConfig<'_, &'span mut MatchesSpan<'buffers>>
{
    fn from(matches: &'span mut MatchesSpan<'buffers>) -> Self {
        DenseMaxsimConfig {
            wanted: matches.wanted,
            matches,
            expansion_factor: 0,
            progress: None,
            stats: None,
        }
    }
}

/// The arena a sparse product lands in and the knobs of its walk; see [`DenseVerbs::multiply`].
///
/// Mirrors `usearch_dense_multiply_t`.
pub struct DenseMultiplyConfig<'config, Holder = Arena> {
    /// An owned [`Arena`] the result keeps, or a `&mut Arena` it borrows.
    pub arena: Holder,
    /// Keeps the first `wanted` pairs each row's walk scored; zero keeps every pair.
    pub wanted: usize,
    /// The candidate-pool breadth, which widens the emitted pattern; zero derives it.
    pub expansion_factor: usize,
    /// Reports the rows resolved and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl<Holder: BorrowMut<Arena>> From<Holder> for DenseMultiplyConfig<'_, Holder> {
    fn from(arena: Holder) -> Self {
        DenseMultiplyConfig {
            arena,
            wanted: 0,
            expansion_factor: 0,
            progress: None,
            stats: None,
        }
    }
}

/// Where an attention lands its outputs and the knobs of its walk; see [`DenseVerbs::attend`].
///
/// Mirrors `usearch_dense_attend_t`.
pub struct DenseAttendConfig<'config, Outputs> {
    /// One row per query, borrowed or owned, already divided by the partition when the call ends.
    pub outputs: Outputs,
    /// The query heads sharing the index's key-value head, and with it one walk per token: row
    /// `t · head_count + h` is head `h` of token `t`, so it must divide the rows; zero means one.
    pub head_count: usize,
    /// Where the online-softmax state lands, one entry per query row, which folds this call into
    /// another over the same queries by partial-softmax rescaling.
    pub carry: Option<&'config mut [DenseSoftmaxState]>,
    /// The candidate-pool breadth; zero derives it. A wider pool walks further, folding more of the
    /// tail.
    pub expansion_factor: usize,
    /// The logit scale, conventionally the inverse square root of the head dimensionality; zero
    /// derives it from the index's frozen geometry.
    pub score_scale: f32,
    /// Reports the queries resolved and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
    /// Where the traversal telemetry lands, overwritten by the call.
    pub stats: Option<&'config mut DenseTraversalStats>,
}

impl<Outputs: RowsSpan> From<Outputs> for DenseAttendConfig<'_, Outputs> {
    fn from(outputs: Outputs) -> Self {
        DenseAttendConfig {
            outputs,
            head_count: 0,
            carry: None,
            expansion_factor: 0,
            score_scale: 0.0,
            progress: None,
            stats: None,
        }
    }
}

/// The donors a merge absorbs and the handler counting them; see [`DenseVerbs::merge`].
///
/// Mirrors `usearch_dense_merge_t`.
pub struct DenseMergeConfig<'config, Donors> {
    /// The indexes absorbed, consumed whether or not the merge succeeds.
    pub donors: Donors,
    /// Reports the donors absorbed and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
}

impl<Donors: IntoIterator<Item = DenseIndex>> From<Donors> for DenseMergeConfig<'_, Donors> {
    fn from(donors: Donors) -> Self { DenseMergeConfig { donors, progress: None } }
}

/// The path a checkpoint is written to and the handler counting its shards; see [`DenseVerbs::save`].
///
/// Mirrors `usearch_dense_save_t`.
pub struct DenseSaveConfig<'config, Location> {
    /// A `.safegraphs` path for one monolith, a `.safegraphs.json` path for a manifest.
    pub path: Location,
    /// Reports the shards written and stops wherever it says to.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
}

impl<Location: AsRef<Path>> From<Location> for DenseSaveConfig<'_, Location> {
    fn from(path: Location) -> Self { DenseSaveConfig { path, progress: None } }
}

/// The arena a monolith is written into and the handler counting its bytes; see
/// [`DenseVerbs::save_buffer`].
///
/// Mirrors `usearch_dense_save_buffer_t`.
pub struct DenseSaveBufferConfig<'config, Holder = Arena> {
    /// An owned [`Arena`] the result keeps, or a `&mut Arena` it borrows.
    pub arena: Holder,
    /// Reports the bytes written and stops wherever it says to; a stopped save leaves no monolith.
    pub progress: Option<&'config mut ProgressCallback<'config>>,
}

impl<Holder: BorrowMut<Arena>> From<Holder> for DenseSaveBufferConfig<'_, Holder> {
    fn from(arena: Holder) -> Self { DenseSaveBufferConfig { arena, progress: None } }
}

/// The query a stream walks and the breadth of its walk; see [`DenseVerbs::search_stream`].
///
/// Mirrors `usearch_dense_search_stream_init_t`.
pub struct DenseSearchStreamInitConfig<Query> {
    /// Exactly one query row.
    pub query: Query,
    /// The candidate-pool breadth the walk explores; zero derives it from the geometry.
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
impl<Destination> Accepts<DenseSearchConfig<'static, Destination>> for Deferring where Deferring: Accepts<Destination> {}
impl<Destination> Accepts<DenseMaxsimConfig<'static, Destination>> for Deferring where Deferring: Accepts<Destination> {}
impl<Holder> Accepts<DenseMultiplyConfig<'static, Holder>> for Deferring where Deferring: Accepts<Holder> {}
impl<Outputs> Accepts<DenseAttendConfig<'static, Outputs>> for Deferring where Deferring: Accepts<Outputs> {}
impl<Value> Accepts<&'static ValueLookup<'static, Value>> for Deferring {}
impl<Donors> Accepts<DenseMergeConfig<'static, Donors>> for Deferring {}
impl<Location> Accepts<DenseSaveConfig<'static, Location>> for Deferring where Deferring: Accepts<Location> {}
impl<Holder> Accepts<DenseSaveBufferConfig<'static, Holder>> for Deferring where Deferring: Accepts<Holder> {}

// endregion: Configs

// region: Results

/// The rows [`DenseVerbs::get`] exported into an arena, which `Holder` owns or borrows.
///
/// Rust-only.
pub struct Vectors<Value: Element, Holder = Arena> {
    holder: Holder,
    /// Lives in the arena `holder` keeps, so it is only lent out for as long as `self` is borrowed.
    rows: MatrixView<'static, Value>,
}

impl<Value: Element, Holder> Vectors<Value, Holder> {
    /// One row per entry under the key, in insertion order, at the pitch the library reported.
    pub fn view(&self) -> MatrixView<'_, Value> { self.rows }

    /// The arena, for the next call to reuse.
    pub fn into_arena(self) -> Holder { self.holder }
}

/// The ragged sparse product [`DenseVerbs::multiply`] exported into an arena, which `Holder` owns or
/// borrows: per left row, every pair its walk scored, in traversal order.
///
/// Rust-only.
pub struct Products<Holder = Arena> {
    holder: Holder,
    offsets: Vec<usize>,
    keys: Exported<Key>,
    scores: Exported<Distance>,
    written: usize,
    required: usize,
}

impl<Holder> Products<Holder> {
    /// Left rows the call multiplied.
    pub fn queries(&self) -> usize { self.offsets.len() - 1 }

    /// Pairs the walks scored across every row, before the wanted bound cut any.
    pub fn required(&self) -> usize { self.required }

    /// The pairs of one left row - a key naming each indexed row and its product - or `None`
    /// when `row` is not below [`Products::queries`].
    pub fn row(&self, row: usize) -> Option<MatchesRow<'_>> {
        let offsets = self.offsets.get(row..row + 2)?;
        let keys = unsafe { arena_slice(self.keys.0, self.written) };
        let scores = unsafe { arena_slice(self.scores.0, self.written) };
        Some(MatchesRow {
            keys: keys.get(offsets[0]..offsets[1])?,
            distances: scores.get(offsets[0]..offsets[1])?,
        })
    }

    /// Every left row's pairs, in row order.
    pub fn rows(&self) -> impl Iterator<Item = MatchesRow<'_>> { (0..self.queries()).filter_map(|row| self.row(row)) }

    /// The arena, for the next call to reuse.
    pub fn into_arena(self) -> Holder { self.holder }
}

/// The bytes [`DenseVerbs::save_buffer`] exported into an arena, which `Holder` owns or borrows.
///
/// Rust-only.
pub struct Monolith<Holder = Arena> {
    holder: Holder,
    bytes: Exported<u8>,
    length: usize,
}

impl<Holder> Monolith<Holder> {
    /// The arena, for the next call to reuse.
    pub fn into_arena(self) -> Holder { self.holder }
}

impl<Holder> Deref for Monolith<Holder> {
    type Target = [u8];

    fn deref(&self) -> &[u8] { unsafe { arena_slice(self.bytes.0, self.length) } }
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
/// let executor = Executor::for_ram(&[], 0, 0)?;
/// let matches = index.on(&executor).search(&rows(&query, 4)?, 10)?;
/// for row in matches.rows() {
///     println!("{} matches", row.keys.len());
/// }
/// # Ok::<(), usearch::Error>(())
/// ```
///
/// Mirrors `usearch_dense_index_t`; dropping it calls `usearch_dense_free`.
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
    /// The library reads a deferred call's inputs after the verb returns, so lending a borrow does
    /// not compile:
    ///
    /// ```compile_fail,E0277
    /// use usearch::*;
    ///
    /// let index = DenseIndex::init_ram(&DenseOptions { dimensions: 4, ..DenseOptions::default() }, &[])?;
    /// let executor = Executor::for_ram(&[], 0, 0)?;
    /// let mut arena = Arena::default();
    /// let call = index.deferred(&executor).get::<f32, _>(Key::from(1u64), &mut arena)?;
    /// # Ok::<(), usearch::Error>(())
    /// ```
    ///
    /// An owned arena compiles, and the rows it holds come back with it:
    ///
    /// ```no_run
    /// use usearch::*;
    ///
    /// let index = DenseIndex::init_ram(&DenseOptions { dimensions: 4, ..DenseOptions::default() }, &[])?;
    /// let executor = Executor::for_ram(&[], 0, 0)?;
    /// let call = index.deferred(&executor).get::<f32, _>(Key::from(1u64), Arena::default())?;
    /// let rows = call.wait()?;
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
    ///
    /// Calls `usearch_dense_clear`.
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
    ///
    /// Calls `usearch_dense_init_ram`.
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
                ram: ram_options(memory_domains),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Creates an empty, mutable index in device memory, sharded across `devices` and never
    /// replicated; empty derives one free device.
    ///
    /// Calls `usearch_dense_init_vram`.
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
                vram: vram_options(devices),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a checkpoint into a new, mutable index in host memory; a bare path reads on the
    /// calling thread.
    ///
    /// Calls `usearch_dense_load_ram`.
    pub fn load_ram<'config>(config: impl Into<DenseLoadRamConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let (progress, progress_state) = lower_progress(&mut config.progress);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_ram(&mut usearch_dense_load_ram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                ram: ram_options(config.memory_domains),
                values_per_key: config.values_per_key,
                progress,
                progress_state,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a checkpoint into a new, mutable index in device memory, placing its parts as
    /// [`DenseIndex::load_ram`] does.
    ///
    /// Calls `usearch_dense_load_vram`.
    pub fn load_vram<'config>(config: impl Into<DenseLoadVramConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let (progress, progress_state) = lower_progress(&mut config.progress);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_vram(&mut usearch_dense_load_vram_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                path: path_view(config.path)?,
                vram: vram_options(config.devices),
                values_per_key: config.values_per_key,
                progress,
                progress_state,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Copies a monolith from a buffer into a new, mutable index in host memory.
    ///
    /// Calls `usearch_dense_load_ram_buffer`.
    pub fn load_ram_buffer<'config>(config: impl Into<DenseLoadRamBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let (progress, progress_state) = lower_progress(&mut config.progress);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_ram_buffer(&mut usearch_dense_load_ram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                ram: ram_options(config.memory_domains),
                values_per_key: config.values_per_key,
                progress,
                progress_state,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Uploads a monolith from a buffer into a new, mutable index in device memory.
    ///
    /// Calls `usearch_dense_load_vram_buffer`.
    pub fn load_vram_buffer<'config>(config: impl Into<DenseLoadVramBufferConfig<'config>>) -> Result<Self> {
        let mut config = config.into();
        let (progress, progress_state) = lower_progress(&mut config.progress);
        let mut index = DenseIndex::null();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_load_vram_buffer(&mut usearch_dense_load_vram_buffer_t {
                index: &mut index.handle,
                executor: config.executor.map_or(ptr::null_mut(), |executor| executor.handle),
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                buffer: config.buffer.into(),
                vram: vram_options(config.devices),
                values_per_key: config.values_per_key,
                progress,
                progress_state,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
    /// verbs report [`Status::ReadOnly`]; the files must stay unmodified while it lives.
    ///
    /// Calls `usearch_dense_view_ram`.
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
    ///
    /// Calls `usearch_dense_view_nvme`.
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
                nvme: nvme_options(&views),
            })
        };
        check(outcome).map(|()| index)
    }

    /// Reads a checkpoint's geometry and metric without loading it; a manifest answers from its
    /// summary and its first part's header, so the cost does not grow with the index.
    ///
    /// Calls `usearch_dense_metadata_read`.
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
    ///
    /// Calls `usearch_dense_metadata_read_buffer`.
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
/// [`Deferred`](crate::Deferred) that borrows the index. Every configurable verb's last argument
/// converts into its `Dense*Config`, so the bare value is the whole call and
/// `..DenseSearchConfig::from(10)` reaches every knob.
///
/// Rust-only.
pub trait DenseVerbs<'index>: Copy {
    /// [`Blocking`] or [`Deferring`].
    type Completion: Completion;

    /// The index this receiver reaches and the executor its calls run on.
    fn handles(&self) -> Handles;

    /// Reserves capacity for a number of incoming vectors ahead of insertion.
    ///
    /// Calls `usearch_dense_reserve`.
    fn reserve(self, capacity: usize) -> Answer<'index, Self, DenseReserve> {
        <Self::Completion as Completion>::answer(DenseReserve {
            handles: self.handles(),
            capacity,
        })
    }

    /// Reports the frozen metadata and the current counters in one call.
    ///
    /// Calls `usearch_dense_describe`.
    fn describe(self) -> Answer<'index, Self, DenseDescribe> {
        <Self::Completion as Completion>::answer(DenseDescribe {
            handles: self.handles(),
            description: usearch_dense_index_description_t::default(),
        })
    }

    /// The number of entries stored under `key`.
    ///
    /// Calls `usearch_dense_count`.
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

    /// Retrieves every vector stored under `key` into `arena`, which the result owns when handed an
    /// [`Arena`] and borrows when lent a `&mut Arena`; an absent key yields no rows.
    ///
    /// Calls `usearch_dense_get`.
    fn get<Value: Element, Holder: BorrowMut<Arena>>(
        self,
        key: Key,
        arena: Holder,
    ) -> Answer<'index, Self, DenseGet<Value, Holder>>
    where
        Self::Completion: Accepts<Holder>,
    {
        <Self::Completion as Completion>::answer(DenseGet {
            handles: self.handles(),
            key,
            holder: arena,
            vectors: Exported::default(),
            written: 0,
            stride: 0,
            columns: 0,
        })
    }

    /// Adds a matrix of vectors under their keys - the native batch operation.
    ///
    /// A batch is all-or-nothing and becomes visible at once. Every row is claimed before the first
    /// progress report, so a `Break` is ignored. Overlaps safely with concurrent searches on the
    /// same handle, and the input scalar type is independent of the stored quantization - the
    /// library converts on ingest.
    ///
    /// Calls `usearch_dense_insert`.
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

    /// Re-selects the neighbors of every stored entry against the finished graph, answering the
    /// number of nodes whose edges changed.
    ///
    /// A cheap insertion pass followed by refinement beats one expensive pass, so a wide expansion
    /// factor belongs here rather than in [`DenseVerbs::insert`]. On a RAM handle searches may run
    /// throughout, on a VRAM one they may not, and insertions never may. A stop is a time budget
    /// rather than a failure, so it reports no status.
    ///
    /// Calls `usearch_dense_refine`.
    fn refine<'config>(self, config: DenseRefineConfig<'config>) -> Answer<'index, Self, DenseRefine<'config>>
    where
        Self::Completion: Accepts<DenseRefineConfig<'config>>,
    {
        <Self::Completion as Completion>::answer(DenseRefine {
            handles: self.handles(),
            changed: 0,
            config,
        })
    }

    /// Audits how much of the graph a search reaches: forward reachability from as many landmarks
    /// as one search draws, swept level by level.
    ///
    /// Read-only, so concurrent insertions merely leave fresh entries unaudited.
    ///
    /// Calls `usearch_dense_health`.
    fn health(self) -> Answer<'index, Self, DenseHealth> {
        <Self::Completion as Completion>::answer(DenseHealth {
            handles: self.handles(),
            reachability: DenseReachability::default(),
        })
    }

    /// Resolves a matrix of queries into up to `wanted` nearest matches apiece - a bare count
    /// allocates [`Matches`], a `&mut MatchesSpan` lands them in caller-owned buffers.
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    ///
    /// Calls `usearch_dense_search`.
    fn search<'config, Queries: RowsView, Destination: Ranked>(
        self,
        queries: Queries,
        config: impl Into<DenseSearchConfig<'config, Destination>>,
    ) -> Answer<'index, Self, DenseSearch<'config, Queries, Destination>>
    where
        Self::Completion: Accepts<Queries> + Accepts<DenseSearchConfig<'config, Destination>>,
    {
        <Self::Completion as Completion>::answer(DenseSearch {
            handles: self.handles(),
            queries,
            config: config.into(),
        })
    }

    /// Late-interaction scoring of multi-row queries against multi-row entries, keeping up to
    /// `wanted` keys per query, largest score first.
    ///
    /// Every query row takes its best match within a key, and those maxima sum to that key's
    /// score - so `rows_per_query` is the token count each query occupies, `queries` holds them
    /// end to end, and `wanted` counts documents rather than rows. This is what `values_per_key`
    /// above one exists for. Runs on an inner-product index alone.
    ///
    /// See <https://arxiv.org/abs/2004.12832> - ColBERT, which introduced this scoring.
    ///
    /// Calls `usearch_dense_maxsim`.
    fn maxsim<'config, Queries: RowsView, Destination: Ranked>(
        self,
        queries: Queries,
        rows_per_query: usize,
        config: impl Into<DenseMaxsimConfig<'config, Destination>>,
    ) -> Answer<'index, Self, DenseMaxsim<'config, Queries, Destination>>
    where
        Self::Completion: Accepts<Queries> + Accepts<DenseMaxsimConfig<'config, Destination>>,
    {
        <Self::Completion as Completion>::answer(DenseMaxsim {
            handles: self.handles(),
            queries,
            rows_per_query,
            config: config.into(),
        })
    }

    /// Approximate sparse product of a matrix against the indexed rows, landing in the arena its
    /// config carries.
    ///
    /// The linear-algebra sibling of [`DenseVerbs::search`], reporting every pair the traversal
    /// scored rather than only the nearest. Pairs arrive in traversal order rather than sorted.
    /// Runs on an inner-product index alone, since a product is a logit.
    ///
    /// Calls `usearch_dense_multiply`.
    fn multiply<'config, Left: RowsView, Holder: BorrowMut<Arena>>(
        self,
        left: Left,
        config: impl Into<DenseMultiplyConfig<'config, Holder>>,
    ) -> Answer<'index, Self, DenseMultiply<'config, Left, Holder>>
    where
        Self::Completion: Accepts<Left> + Accepts<DenseMultiplyConfig<'config, Holder>>,
    {
        <Self::Completion as Completion>::answer(DenseMultiply {
            handles: self.handles(),
            left,
            offsets: Vec::new(),
            keys: Exported::default(),
            scores: Exported::default(),
            written: 0,
            required: 0,
            config: config.into(),
        })
    }

    /// Softmax attention of every query row over every indexed key its walk scores, folded as
    /// Flash Attention folds key tiles, with no top-k cut, landing in its config's outputs.
    ///
    /// The index holds one layer's keys for one key-value head and no values: `value_lookup`
    /// resolves every scored key to its value row, as wide as the keys and stored in the index's
    /// own quantization. A key left without a row, or given one of another type or width, refuses the
    /// call with [`Status::ShapeMismatch`], since a missing row is a wrong softmax rather than a
    /// smaller one, and a VRAM handle refuses the host lookup with
    /// [`Status::InvalidArgument`]. The outputs come back already divided by the partition, and
    /// handed back when owned. Runs on an inner-product index alone, since a softmax needs logits.
    ///
    /// See <https://arxiv.org/abs/2205.14135> - Flash Attention, whose recurrence this reuses.
    ///
    /// Calls `usearch_dense_attend`.
    fn attend<'config, 'lookup, Queries: RowsView, Value: Element, Outputs: RowsSpan>(
        self,
        queries: Queries,
        value_lookup: &'lookup ValueLookup<'lookup, Value>,
        config: impl Into<DenseAttendConfig<'config, Outputs>>,
    ) -> Answer<'index, Self, DenseAttend<'config, 'lookup, Queries, Value, Outputs>>
    where
        Self::Completion: Accepts<Queries>
            + Accepts<&'lookup ValueLookup<'lookup, Value>>
            + Accepts<DenseAttendConfig<'config, Outputs>>,
    {
        <Self::Completion as Completion>::answer(DenseAttend {
            handles: self.handles(),
            queries,
            lookup: value_lookup,
            config: config.into(),
        })
    }

    /// Absorbs every donor into this index, wiring cross-links via landmark stitching.
    ///
    /// Cheaper than re-inserting the donors' vectors one by one, and readers may keep searching
    /// throughout. The donors are consumed whether or not the merge succeeds; a stopped merge keeps
    /// the donors already absorbed, and the count the handler last saw names the boundary.
    ///
    /// No residency absorbs donors yet, so the engine answers [`Status::Unsupported`] and frees the
    /// donors.
    ///
    /// Calls `usearch_dense_merge`.
    fn merge<'config, Donors: IntoIterator<Item = DenseIndex>>(
        self,
        config: impl Into<DenseMergeConfig<'config, Donors>>,
    ) -> Answer<'index, Self, DenseMerge<'config>>
    where
        Self::Completion: Accepts<DenseMergeConfig<'config, Donors>>,
    {
        let config = config.into();
        <Self::Completion as Completion>::answer(DenseMerge {
            handles: self.handles(),
            donors: DonorHandles::of(config.donors),
            progress: config.progress,
        })
    }

    /// Saves the index in the layout the path extension picks - one `.safegraphs` monolith, or a
    /// `.safegraphs.json` manifest beside its shards.
    ///
    /// Demands a mutation-free window; concurrent readers keep searching throughout. A stopped save
    /// leaves leftovers every reader rejects and the caller is left to remove.
    ///
    /// Calls `usearch_dense_save`.
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
    ///
    /// Calls `usearch_dense_save_buffer`.
    fn save_buffer<'config, Holder: BorrowMut<Arena>>(
        self,
        config: impl Into<DenseSaveBufferConfig<'config, Holder>>,
    ) -> Answer<'index, Self, DenseSaveBuffer<'config, Holder>>
    where
        Self::Completion: Accepts<DenseSaveBufferConfig<'config, Holder>>,
    {
        <Self::Completion as Completion>::answer(DenseSaveBuffer {
            handles: self.handles(),
            bytes: Exported::default(),
            length: 0,
            config: config.into(),
        })
    }

    /// Opens a row scan over the current contents, pulled blocking on this receiver's executor;
    /// each shard's publication watermark is read as the walk enters it, so a row once passed is
    /// never revisited.
    ///
    /// Calls `usearch_dense_cursor_init`.
    fn cursor(self) -> Result<DenseCursor<'index>>
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
    ///
    /// Calls `usearch_dense_search_stream_init`.
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
            keys: [Key::NIL; MATCHES_PER_PULL],
            distances: [0.0; MATCHES_PER_PULL],
            filled: 0,
            position: 0,
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
///
/// Calls `usearch_dense_reserve`.
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
///
/// Calls `usearch_dense_describe`.
pub struct DenseDescribe {
    handles: Handles,
    description: usearch_dense_index_description_t,
}

// Safety: the only pointer the description holds is the static capabilities name.
unsafe impl Send for DenseDescribe {}

impl Calling for DenseDescribe {
    type Arguments = usearch_dense_describe_t;
    type Output = DenseIndexDescription;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
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
            capabilities: unsafe { static_string(self.description.capabilities) },
        })
    }
}

/// A count of the entries under one key; see [`DenseVerbs::count`].
///
/// Calls `usearch_dense_count`.
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

/// A retrieval of the rows under one key; see [`DenseVerbs::get`].
///
/// Calls `usearch_dense_get`.
pub struct DenseGet<Value, Holder> {
    handles: Handles,
    key: Key,
    holder: Holder,
    vectors: Exported<Value>,
    written: usize,
    stride: usize,
    columns: usize,
}

impl<Value: Element, Holder: BorrowMut<Arena>> Calling for DenseGet<Value, Holder> {
    type Arguments = usearch_dense_get_t;
    type Output = Vectors<Value, Holder>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
    ) -> Result<Self::Arguments> {
        Ok(usearch_dense_get_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            key: self.key,
            scalar: Value::SCALAR as usearch_scalar_t,
            vectors: ptr::addr_of_mut!(self.vectors.0).cast(),
            written: &mut self.written,
            stride: &mut self.stride,
            columns: &mut self.columns,
            arena: &mut self.holder.borrow_mut().handle,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_get(arguments) }

    fn finish(self) -> Result<Vectors<Value, Holder>> {
        // An absent key leaves no rows and a null pointer, which a zero-row view never reads.
        let data = NonNull::new(self.vectors.0).unwrap_or(NonNull::dangling()).as_ptr();
        let element = core::mem::size_of::<Value>() as isize;
        let rows = unsafe {
            MatrixView::from_raw_parts(data, &[self.written, self.columns], &[self.stride as isize, element])
        }
        .map_err(|_| fail(Status::Internal, "the library returned an untraversable slab"))?;
        Ok(Vectors {
            holder: self.holder,
            rows,
        })
    }
}

/// An insertion; see [`DenseVerbs::insert`].
///
/// Calls `usearch_dense_insert`.
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
    ) -> Result<Self::Arguments> {
        let (keys, batch) = (self.keys.as_ref(), self.config.vectors.batch()?);
        // The keys view carries no count, so only this side can see the two disagree.
        if batch.count != keys.len() {
            return Err(fail(
                Status::ShapeMismatch,
                "the key count does not match the vector row count",
            ));
        }
        let (progress, progress_state) = lower_progress(&mut self.config.progress);
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
            progress_state,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_insert(arguments) }

    fn finish(self) -> Result<()> { Ok(()) }
}

/// A refinement sweep; see [`DenseVerbs::refine`].
///
/// Calls `usearch_dense_refine`.
pub struct DenseRefine<'config> {
    handles: Handles,
    changed: usize,
    config: DenseRefineConfig<'config>,
}

impl Calling for DenseRefine<'_> {
    type Arguments = usearch_dense_refine_t;
    type Output = usize;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
    ) -> Result<Self::Arguments> {
        let (progress, progress_state) = lower_progress(&mut self.config.progress);
        Ok(usearch_dense_refine_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            changed: &mut self.changed,
            expansion_factor: self.config.expansion_factor,
            progress,
            progress_state,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_refine(arguments) }

    fn finish(self) -> Result<usize> { Ok(self.changed) }
}

/// A reachability audit; see [`DenseVerbs::health`].
///
/// Calls `usearch_dense_health`.
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
///
/// Calls `usearch_dense_search`.
pub struct DenseSearch<'config, Queries, Destination> {
    handles: Handles,
    queries: Queries,
    config: DenseSearchConfig<'config, Destination>,
}

impl<Queries: RowsView, Destination: Ranked> Calling for DenseSearch<'_, Queries, Destination> {
    type Arguments = usearch_dense_search_t;
    type Output = Destination::Output;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
    ) -> Result<Self::Arguments> {
        let batch = self.queries.batch()?;
        let (predicate, predicate_state) = lower_predicate(&self.config.predicate);
        let (progress, progress_state) = lower_progress(&mut self.config.progress);
        Ok(usearch_dense_search_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            queries: batch.rows,
            matches: self.config.matches.span(batch.count, self.config.wanted)?,
            expansion_factor: self.config.expansion_factor,
            predicate,
            predicate_state,
            progress,
            progress_state,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_search(arguments) }

    fn finish(self) -> Result<Destination::Output> { Ok(self.config.matches.finish()) }
}

/// A late-interaction search; see [`DenseVerbs::maxsim`].
///
/// Calls `usearch_dense_maxsim`.
pub struct DenseMaxsim<'config, Queries, Destination> {
    handles: Handles,
    queries: Queries,
    rows_per_query: usize,
    config: DenseMaxsimConfig<'config, Destination>,
}

impl<Queries: RowsView, Destination: Ranked> Calling for DenseMaxsim<'_, Queries, Destination> {
    type Arguments = usearch_dense_maxsim_t;
    type Output = Destination::Output;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
    ) -> Result<Self::Arguments> {
        let batch = self.queries.batch()?;
        // The call counts queries, so only this side sees the matrix's rows fail to divide.
        if self.rows_per_query == 0 || !batch.count.is_multiple_of(self.rows_per_query) {
            return Err(fail(
                Status::ShapeMismatch,
                "the query rows do not divide into whole queries",
            ));
        }
        let count = batch.count / self.rows_per_query;
        let (progress, progress_state) = lower_progress(&mut self.config.progress);
        Ok(usearch_dense_maxsim_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count,
            queries: batch.rows,
            rows_per_query: self.rows_per_query,
            matches: self.config.matches.span(count, self.config.wanted)?,
            expansion_factor: self.config.expansion_factor,
            progress,
            progress_state,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_maxsim(arguments) }

    fn finish(self) -> Result<Destination::Output> { Ok(self.config.matches.finish()) }
}

/// An approximate sparse product; see [`DenseVerbs::multiply`].
///
/// Calls `usearch_dense_multiply`.
pub struct DenseMultiply<'config, Left, Holder> {
    handles: Handles,
    left: Left,
    offsets: Vec<usize>,
    keys: Exported<Key>,
    scores: Exported<Distance>,
    written: usize,
    required: usize,
    config: DenseMultiplyConfig<'config, Holder>,
}

impl<Left: RowsView, Holder: BorrowMut<Arena>> Calling for DenseMultiply<'_, Left, Holder> {
    type Arguments = usearch_dense_multiply_t;
    type Output = Products<Holder>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
    ) -> Result<Self::Arguments> {
        let batch = self.left.batch()?;
        self.offsets = vec![0; batch.count + 1];
        let (progress, progress_state) = lower_progress(&mut self.config.progress);
        Ok(usearch_dense_multiply_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            left: batch.rows,
            offsets: self.offsets.as_mut_ptr(),
            keys: &mut self.keys.0,
            scores: &mut self.scores.0,
            arena: &mut self.config.arena.borrow_mut().handle,
            written: &mut self.written,
            required: &mut self.required,
            wanted: self.config.wanted,
            expansion_factor: self.config.expansion_factor,
            progress,
            progress_state,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_multiply(arguments) }

    fn finish(self) -> Result<Products<Holder>> {
        Ok(Products {
            holder: self.config.arena,
            offsets: self.offsets,
            keys: self.keys,
            scores: self.scores,
            written: self.written,
            required: self.required,
        })
    }
}

/// # Safety
/// `state` must address a live `(&ValueLookup, row_length)` that outlives the call the callback was
/// passed to, and `keys` and `rows` must hold `count` entries each.
unsafe extern "C" fn value_lookup_trampoline<Value: Element>(
    state: *mut c_void,
    keys: *const Key,
    count: usize,
    scalar: usearch_scalar_t,
    columns: usize,
    rows: *mut *const c_void,
) {
    let lookup = *state.cast::<&ValueLookup<'_, Value>>();
    // The library reads rows of its own shape, so a lookup typed otherwise answers no row at all.
    let row_length = match scalar == Value::SCALAR as usearch_scalar_t {
        true => values_of::<Value>(columns),
        false => usize::MAX,
    };
    let keys = slice::from_raw_parts(keys, count);
    for (key, row) in keys.iter().zip(slice::from_raw_parts_mut(rows, count)) {
        // Unwinding out of an `extern "C"` frame aborts, so a panic leaves the row null and refuses the call.
        *row = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lookup(*key))) {
            Ok(Some(value)) if value.len() == row_length => value.as_ptr().cast(),
            _ => ptr::null(),
        };
    }
}

/// A graph-walk attention; see [`DenseVerbs::attend`].
///
/// Calls `usearch_dense_attend`.
pub struct DenseAttend<'config, 'lookup, Queries, Value, Outputs> {
    handles: Handles,
    queries: Queries,
    lookup: &'lookup ValueLookup<'lookup, Value>,
    config: DenseAttendConfig<'config, Outputs>,
}

impl<Queries: RowsView, Value: Element, Outputs: RowsSpan> Calling for DenseAttend<'_, '_, Queries, Value, Outputs> {
    type Arguments = usearch_dense_attend_t;
    type Output = Outputs::Output;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
    ) -> Result<Self::Arguments> {
        let (batch, outputs) = (self.queries.batch()?, self.config.outputs.span()?);
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
        let (progress, progress_state) = lower_progress(&mut self.config.progress);
        Ok(usearch_dense_attend_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: batch.count,
            queries: batch.rows,
            head_count: self.config.head_count,
            value_lookup: Some(value_lookup_trampoline::<Value>),
            value_lookup_state: ptr::addr_of_mut!(self.lookup).cast(),
            outputs,
            carry,
            expansion_factor: self.config.expansion_factor,
            score_scale: self.config.score_scale,
            progress,
            progress_state,
            stats: self.config.stats.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut),
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_attend(arguments) }

    fn finish(self) -> Result<Outputs::Output> { Ok(self.config.outputs.finish()) }
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

/// A merge of donor indexes; see [`DenseVerbs::merge`].
///
/// Calls `usearch_dense_merge`.
pub struct DenseMerge<'config> {
    handles: Handles,
    donors: DonorHandles,
    progress: Option<&'config mut ProgressCallback<'config>>,
}

impl Calling for DenseMerge<'_> {
    type Arguments = usearch_dense_merge_t;
    type Output = ();

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
    ) -> Result<Self::Arguments> {
        let (progress, progress_state) = lower_progress(&mut self.progress);
        Ok(usearch_dense_merge_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            count: self.donors.0.len(),
            donors: self.donors.0.as_mut_ptr(),
            progress,
            progress_state,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_merge(arguments) }

    fn finish(self) -> Result<()> { Ok(()) }
}

/// A checkpoint write; see [`DenseVerbs::save`].
///
/// Calls `usearch_dense_save`.
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
    ) -> Result<Self::Arguments> {
        let path = path_view(self.config.path.as_ref())?;
        let (progress, progress_state) = lower_progress(&mut self.config.progress);
        Ok(usearch_dense_save_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            path,
            progress,
            progress_state,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_save(arguments) }

    fn finish(self) -> Result<()> { Ok(()) }
}

/// A monolith written into an arena; see [`DenseVerbs::save_buffer`].
///
/// Calls `usearch_dense_save_buffer`.
pub struct DenseSaveBuffer<'config, Holder> {
    handles: Handles,
    bytes: Exported<u8>,
    length: usize,
    config: DenseSaveBufferConfig<'config, Holder>,
}

impl<Holder: BorrowMut<Arena>> Calling for DenseSaveBuffer<'_, Holder> {
    type Arguments = usearch_dense_save_buffer_t;
    type Output = Monolith<Holder>;

    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
    ) -> Result<Self::Arguments> {
        let (progress, progress_state) = lower_progress(&mut self.config.progress);
        Ok(usearch_dense_save_buffer_t {
            index: self.handles.index,
            executor: self.handles.executor,
            outcome,
            completion,
            bytes: &mut self.bytes.0,
            length: &mut self.length,
            arena: &mut self.config.arena.borrow_mut().handle,
            progress,
            progress_state,
        })
    }

    unsafe fn call(arguments: *mut Self::Arguments) { usearch_dense_save_buffer(arguments) }

    fn finish(self) -> Result<Monolith<Holder>> {
        Ok(Monolith {
            holder: self.config.arena,
            bytes: self.bytes,
            length: self.length,
        })
    }
}

// endregion: Calls

// region: Cursors and Streams

/// A row scan over the index contents; see [`DenseVerbs::cursor`].
///
/// The walk is a sequential row scan, so it stays a batch operation: export, migration, and
/// visualization all move whole slabs.
///
/// Mirrors `usearch_dense_cursor_t`; dropping it calls `usearch_dense_cursor_free`.
#[derive(Debug)]
pub struct DenseCursor<'index> {
    handle: usearch_dense_cursor_t,
    handles: Handles,
    receiver: PhantomData<On<'index, DenseIndex>>,
}

impl DenseCursor<'_> {
    /// Copies the next keys into `keys`, returning how many were written; zero means the walk is
    /// exhausted.
    ///
    /// Calls `usearch_dense_cursor_next`.
    pub fn next_keys(&mut self, keys: &mut [Key]) -> Result<usize> {
        let skipped = usearch_rows_span_t {
            data: ptr::null_mut(),
            stride: 0,
            columns: 0,
            capacity: 0,
            scalar: 0,
        };
        self.next(keys, skipped)
    }

    /// Copies the next entries into `keys` and `vectors`, which needs room for as many rows as
    /// there are keys, returning the rows written; zero means the walk is exhausted.
    ///
    /// Calls `usearch_dense_cursor_next`.
    pub fn next_batch<Value: Element>(
        &mut self,
        keys: &mut [Key],
        vectors: &mut impl TensorMut<Value, 2>,
    ) -> Result<usize> {
        let span = span_of(vectors)?;
        self.next(keys, span)
    }

    fn next(&mut self, keys: &mut [Key], vectors: usearch_rows_span_t) -> Result<usize> {
        let mut written = 0;
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_cursor_next(&mut usearch_dense_cursor_next_t {
                index: self.handles.index,
                executor: self.handles.executor,
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                cursor: self.handle,
                keys: usearch_keys_span_t::of(keys),
                vectors,
                written: &mut written,
            })
        };
        check(outcome)?;
        Ok(written)
    }
}

impl Drop for DenseCursor<'_> {
    fn drop(&mut self) { unsafe { usearch_dense_cursor_free(self.handle) } }
}

/// Matches a stream pulls per crossing into the library, amortising the call over a run of them.
const MATCHES_PER_PULL: usize = 64;

/// A resumable nearest-neighbor stream; see [`DenseVerbs::search_stream`].
///
/// Ordering beyond the expansion horizon is best-effort nearest, not globally exact. Matches
/// arrive in batches and are yielded one at a time, so nothing allocates per item. The walk ends
/// at its first empty pull or failure, after which the stream yields nothing more.
///
/// Mirrors `usearch_dense_search_stream_t`; dropping it calls `usearch_dense_search_stream_free`.
#[derive(Debug)]
pub struct DenseSearchStream<'index> {
    /// Null once the walk ended.
    handle: usearch_dense_search_stream_t,
    handles: Handles,
    keys: [Key; MATCHES_PER_PULL],
    distances: [Distance; MATCHES_PER_PULL],
    filled: usize,
    position: usize,
    receiver: PhantomData<On<'index, DenseIndex>>,
}

impl DenseSearchStream<'_> {
    /// Refills the batch straight from the library, ending the walk once it is exhausted or fails.
    fn pull(&mut self) -> Result<()> {
        let mut written = 0;
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_dense_search_stream_next(&mut usearch_dense_search_stream_next_t {
                index: self.handles.index,
                executor: self.handles.executor,
                outcome: &mut outcome,
                completion: usearch_completion_blocking_k,
                stream: self.handle,
                matches: matches_span(
                    &mut self.keys,
                    &mut self.distances,
                    core::slice::from_mut(&mut written),
                    MATCHES_PER_PULL,
                    MATCHES_PER_PULL,
                ),
            })
        };
        (self.filled, self.position) = (written, 0);
        let pulled = check(outcome);
        if pulled.is_err() || written == 0 {
            unsafe { usearch_dense_search_stream_free(core::mem::replace(&mut self.handle, ptr::null_mut())) }
        }
        pulled
    }
}

impl Iterator for DenseSearchStream<'_> {
    type Item = Result<Match>;

    /// Calls `usearch_dense_search_stream_next` once per batch of matches.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.filled {
            if self.handle.is_null() {
                return None;
            }
            if let Err(failure) = self.pull() {
                return Some(Err(failure));
            }
        }
        let key = *self.keys[..self.filled].get(self.position)?;
        let distance = self.distances[self.position];
        self.position += 1;
        Some(Ok(Match { key, distance }))
    }
}

impl core::iter::FusedIterator for DenseSearchStream<'_> {}

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

    use numkong::{u1x8, Matrix, MatrixSpan, MatrixView};
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::types::{rows, rows_mut, Deferred, Mood};

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
    fn executor() -> Executor { Executor::for_ram(&[], 0, 0).expect("the executor must spawn") }

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
        let matches = index.search(&rows(query, dimensions)?, 1)?;
        Ok(matches.row(0).map_or(0, |row| row.keys.len()))
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

    fn close(left: f64, right: f64, tolerance: f64) -> bool { (left - right).abs() <= tolerance }

    fn requires_send_sync<Subject: Send + Sync>() {}

    fn requires_send<Subject: Send>() {}

    /// Packs one byte per element, LSB first, as NumKong's bit vectors are laid out.
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
    /// the stand-in ticks once per row - and the assertions run after the call, since a panic
    /// inside a handler is caught and read as a stop rather than as a failed test.
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
            quantization: Scalar::F32 as i32,
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
    #[test]
    fn constructs_every_metric_and_quantization() -> Result<()> {
        for quantization in [
            Scalar::F64,
            Scalar::F32,
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
            let described = empty_index(64, DenseMetric::Angular, quantization)?.describe()?;
            assert_eq!(described.metadata.dimensions, 64, "{quantization:?}");
            assert_eq!(described.metadata.quantization, quantization);
            assert_eq!(described.size, 0, "{quantization:?}");
            assert!(described.metadata.connectivity > 0, "{quantization:?}");
        }

        // The bit-level metrics take packed rows, so each metric meets a kind it accepts.
        for (metric, quantization) in [
            (DenseMetric::Inner, Scalar::F32),
            (DenseMetric::Angular, Scalar::F32),
            (DenseMetric::Euclidean, Scalar::F32),
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

    /// A zeroed width freezes nothing, and the first batch has no width the index would trust.
    #[test]
    fn an_unfrozen_geometry_admits_no_rows() -> Result<()> {
        let options = DenseOptions {
            dimensions: 0,
            quantization: Scalar::F32,
            ..DenseOptions::default()
        };
        let index = DenseIndex::init_ram(&options, &[])?;
        assert_eq!(index.describe()?.metadata.dimensions, 0);

        let slab = seeded_rows(1, 12, SEED);
        assert!(matches!(
            index.insert(sequential_keys(1), &rows(&slab, 12)?),
            Err(Error {
                status: Status::Uninitialized,
                ..
            })
        ));
        Ok(())
    }

    /// The handle is one pointer, so a move must not disturb it and two threads may share it.
    #[test]
    fn survives_box_and_arc_moves() -> Result<()> {
        requires_send_sync::<DenseIndex>();
        let (index, _, _) = populated_index(4, 8, DenseMetric::Angular, Scalar::F32)?;

        let boxed = Box::new(index);
        assert_eq!(boxed.describe()?.size, 4);
        let shared = std::sync::Arc::new(*boxed);
        assert_eq!(shared.describe()?.size, 4);
        Ok(())
    }

    #[test]
    fn clears_and_reuses() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (mut index, keys, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
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
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let keys = sequential_keys(2);
        let slab: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        index.insert(&keys, &rows(&slab, DIMENSIONS)?)?;
        assert_eq!(index.describe()?.size, 2);

        let found = index.get::<f32, _>(keys[1], Arena::default())?;
        assert_eq!(found.view().shape(), &[1, DIMENSIONS]);
        assert_eq!(flattened(found.view()), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            index.get::<f32, _>(Key::from(7u64), Arena::default())?.view().shape()[0],
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
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let empty: [f32; 0] = [];
        index.insert([], &rows(&empty, DIMENSIONS)?)?;
        assert_eq!(index.describe()?.size, 0);
        Ok(())
    }

    #[test]
    fn rejects_mismatched_key_and_row_counts() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
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
        let (index, _, _) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let narrow = seeded_rows(1, DIMENSIONS - 1, SEED);

        assert!(matches!(
            index.insert(sequential_keys(1), &rows(&narrow, DIMENSIONS - 1)?),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        assert!(matches!(
            index.search(&rows(&narrow, DIMENSIONS - 1)?, 1),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        assert!(
            matches!(
                index.search_stream(rows(&narrow, DIMENSIONS - 1)?),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "the query width is checked too"
        );

        let mut cursor = index.cursor()?;
        let mut keys = [Key::NIL; 1];
        let mut buffer = [0.0f32; DIMENSIONS - 1];
        assert!(matches!(
            cursor.next_batch(&mut keys, &mut rows_mut(&mut buffer, DIMENSIONS - 1)?),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn rejects_non_contiguous_rows() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
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
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
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

    /// A pitch wider than the row is the layout a mapped dataset arrives in, and it must cross
    /// both ways - leaving the padding of the row it writes through untouched.
    #[test]
    fn accepts_strided_row_pitch() -> Result<()> {
        const DIMENSIONS: usize = 4;
        const PITCH: usize = 8;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
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

        let mut restored = [-1.0f32; 2 * PITCH];
        let mut span: MatrixSpan<'_, f32> = unsafe {
            MatrixSpan::from_raw_parts(
                restored.as_mut_ptr(),
                &[2, DIMENSIONS],
                &[element * PITCH as isize, element],
            )
        }
        .unwrap();
        let mut walked = [Key::NIL; 2];
        assert_eq!(index.cursor()?.next_batch(&mut walked, &mut span)?, 2);
        for (left, right) in restored[PITCH..PITCH + DIMENSIONS]
            .iter()
            .zip(&rows_source[DIMENSIONS..])
        {
            assert!(close(*left as f64, *right as f64, 1e-6));
        }
        assert!(
            restored[DIMENSIONS..PITCH].iter().all(|value| *value == -1.0),
            "the padding past the row is never written"
        );
        Ok(())
    }

    #[test]
    fn converts_scalar_kinds_on_ingest() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::BF16)?;
        let keys = sequential_keys(1);
        let slab = seeded_rows(1, DIMENSIONS, SEED);
        index.insert(&keys, &rows(&slab, DIMENSIONS)?)?;

        let wide = flattened(index.get::<f64, _>(keys[0], Arena::default())?.view());
        assert_eq!(wide.len(), DIMENSIONS);
        for (left, right) in wide.iter().zip(&slab) {
            assert!(close(*left, *right as f64, 1e-2), "bf16 keeps eight mantissa bits");
        }
        Ok(())
    }

    /// One call returns every entry under a key, however many there turn out to be.
    #[test]
    fn gets_every_entry_sharing_a_key() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let shared = Key::from(3u64);
        let slab = seeded_rows(3, DIMENSIONS, SEED);
        index.insert([shared; 3], &rows(&slab, DIMENSIONS)?)?;

        assert_eq!(index.count(shared)?, 3);
        assert!(index.contains(shared)?);
        assert!(!index.contains(Key::from(8u64))?);

        let restored = index.get::<f32, _>(shared, Arena::default())?;
        assert_eq!(restored.view().shape(), &[3, DIMENSIONS], "one row per entry");
        for (left, right) in flattened(restored.view()).iter().zip(&slab) {
            assert!(close(*left as f64, *right as f64, 1e-6));
        }
        Ok(())
    }

    /// A borrowed arena is reused, and keeping one call's rows past the next takes a second one.
    #[test]
    fn a_second_arena_keeps_the_first_output_alive() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let (index, keys, slab) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let executor = executor();
        let receiver = index.on(&executor);
        let (mut first, mut second) = (Arena::default(), Arena::default());
        let held = receiver.get::<f32, _>(keys[0], &mut first)?;
        let next = receiver.get::<f32, _>(keys[1], &mut second)?;
        assert_eq!(flattened(held.view()), &slab[..DIMENSIONS]);
        assert_eq!(flattened(next.view()), &slab[DIMENSIONS..DIMENSIONS * 2]);

        // An owned arena comes back with the result, ready for the next call.
        let owned = index.get::<f32, _>(keys[2], Arena::default())?.into_arena();
        let reused = index.get::<f32, _>(keys[3], owned)?;
        assert_eq!(flattened(reused.view()), &slab[DIMENSIONS * 3..]);
        Ok(())
    }

    #[test]
    fn uses_128_bit_keys() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
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
        let matches = index.search(&rows(&query, DIMENSIONS)?, 2)?;
        let row = matches.row(0).unwrap();
        assert_eq!(
            row.keys.iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
        Ok(())
    }

    /// A sub-byte row comes back as wide as it went in, never rounded up to its whole bytes.
    #[test]
    fn gets_sub_byte_rows_at_their_logical_width() -> Result<()> {
        const DIMENSIONS: usize = 12;
        let index = empty_index(DIMENSIONS, DenseMetric::Hamming, Scalar::U1x8)?;
        let key = Key::from(5u64);
        index.insert([key], &rows(&bit_rows(&[0b1010_1010, 0b0000_0101]), DIMENSIONS)?)?;

        let found = index.get::<u1x8, _>(key, Arena::default())?;
        assert_eq!(found.view().shape(), &[1, DIMENSIONS]);
        Ok(())
    }

    // endregion: Ingest and Retrieval

    // region: Search

    #[test]
    fn searches_an_empty_index() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let query = seeded_rows(1, DIMENSIONS, SEED);
        assert_eq!(matches_of(&index, &query, DIMENSIONS)?, 0);
        Ok(())
    }

    /// A batch is where the output row stride earns its keep: every query's matches land in
    /// their own row of both matrices, and its count in its own slot.
    #[test]
    fn fills_every_output_row_of_a_batch() -> Result<()> {
        const DIMENSIONS: usize = 64;
        const QUERIES: usize = 8;
        const WANTED: usize = 5;
        let (index, _, slab) = populated_index(256, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        // The index names its rows zero upwards, so the untouched marker has to sit above them.
        let untouched = Key::from(u128::MAX);
        let mut found = vec![untouched; QUERIES * WANTED];
        let mut scores = vec![0.0 as Distance; QUERIES * WANTED];
        let mut counts = vec![0usize; QUERIES];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index.on(&executor()).search(
            &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?,
            DenseSearchConfig {
                expansion_factor: 64,
                ..DenseSearchConfig::from(&mut matches)
            },
        )?;
        assert!(matches.rows().all(|row| row.keys.len() == WANTED));
        assert!(
            matches.rows().all(|row| !row.keys.contains(&untouched)),
            "every slot of every row was written"
        );
        Ok(())
    }

    /// A span wider than the wanted count lands every row inside a wider matrix.
    #[test]
    fn lands_matches_inside_a_wider_span() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const QUERIES: usize = 2;
        let (index, _, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let mut found = vec![Key::NIL; QUERIES * 4];
        let mut scores = vec![0.0 as Distance; QUERIES * 4];
        let mut counts = vec![0usize; QUERIES];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index.search(
            &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?,
            DenseSearchConfig {
                wanted: 2,
                ..DenseSearchConfig::from(&mut matches)
            },
        )?;
        assert!(matches.rows().all(|row| row.keys.len() == 2));
        Ok(())
    }

    #[test]
    fn rejects_undersized_outputs() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const QUERIES: usize = 4;
        let (index, _, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let queries = rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?;

        let mut found = vec![Key::NIL; (QUERIES - 1) * 2];
        let mut scores = vec![0.0 as Distance; (QUERIES - 1) * 2];
        let mut counts = vec![0usize; QUERIES - 1];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        assert!(
            matches!(
                index.search(&queries, &mut matches),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "fewer output rows than there are queries"
        );

        let mut found = vec![Key::NIL; QUERIES * 2];
        let mut scores = vec![0.0 as Distance; QUERIES * 2];
        let mut counts = vec![0usize; QUERIES];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        assert!(
            matches!(
                index.search(
                    &queries,
                    DenseSearchConfig {
                        wanted: 3,
                        ..DenseSearchConfig::from(&mut matches)
                    }
                ),
                Err(Error {
                    status: Status::UnsupportedLayout,
                    ..
                })
            ),
            "fewer output columns than wanted"
        );
        Ok(())
    }

    #[test]
    fn reports_traversal_stats() -> Result<()> {
        const DIMENSIONS: usize = 32;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
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
        let mut searching = DenseTraversalStats::default();
        index.search(
            &query,
            DenseSearchConfig {
                stats: Some(&mut searching),
                ..DenseSearchConfig::from(4)
            },
        )?;
        let first = searching;
        assert!(first.computed_distances > 0);

        index.search(
            &query,
            DenseSearchConfig {
                stats: Some(&mut searching),
                ..DenseSearchConfig::from(4)
            },
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
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let is_even = |key: Key| u64::try_from(key).is_ok_and(|value| value % 2 == 0);
        let matches = index.search(
            &rows(&slab[..DIMENSIONS], DIMENSIONS)?,
            DenseSearchConfig {
                expansion_factor: 64,
                predicate: Some(&is_even),
                ..DenseSearchConfig::from(WANTED)
            },
        )?;

        let admitted = matches.row(0).unwrap();
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
        let (index, keys, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let allowed: HashSet<Key> = keys.iter().copied().take(5).collect();
        let matches = index.on(&executor()).search(
            &rows(&slab[..DIMENSIONS], DIMENSIONS)?,
            DenseSearchConfig {
                expansion_factor: 64,
                predicate: Some(&|key: Key| allowed.contains(&key)),
                ..DenseSearchConfig::from(WANTED)
            },
        )?;

        for key in matches.row(0).unwrap().keys {
            assert!(allowed.contains(key), "{key} is outside the captured set");
        }
        Ok(())
    }

    /// A predicate that panics rejects the key rather than unwinding into the library.
    #[test]
    fn filters_everything_out() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let predicates: [&Predicate; 2] = [&|_| false, &|_| panic!("a filter fault")];
        for predicate in predicates {
            let matches = index.search(
                &rows(&slab[..DIMENSIONS], DIMENSIONS)?,
                DenseSearchConfig {
                    expansion_factor: 64,
                    predicate: Some(predicate),
                    ..DenseSearchConfig::from(4)
                },
            )?;
            assert!(
                matches.row(0).unwrap().keys.is_empty(),
                "a fully-rejecting predicate is not an error"
            );
        }
        Ok(())
    }

    // endregion: Search

    // region: Refinement and Linear Algebra

    /// The return is a change count rather than a status, so a caller loops until the graph
    /// settles.
    #[test]
    fn refinement_reports_its_sweep() -> Result<()> {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, DenseMetric::Angular, Scalar::F32)?;
        let mut stats = DenseTraversalStats::default();
        let changed = index.on(&executor()).refine(DenseRefineConfig {
            expansion_factor: 64,
            stats: Some(&mut stats),
            ..DenseRefineConfig::default()
        })?;
        assert!(
            changed <= COUNT,
            "a sweep cannot change more nodes than the index holds"
        );
        assert!(stats.visited_members > 0, "the sweep reports the nodes it swept");
        Ok(())
    }

    /// The audit judges every entry published when it started, and no search reaches more.
    #[test]
    fn audits_how_much_of_the_graph_a_search_reaches() -> Result<()> {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, DenseMetric::Angular, Scalar::F32)?;
        let reachability = index.on(&executor()).health()?;
        assert_eq!(reachability.audited, COUNT);
        assert!(reachability.reachable != 0 && reachability.reachable <= COUNT);
        assert!(reachability.sweeps != 0, "a populated graph takes at least one level");
        Ok(())
    }

    /// Every row's pairs land in one ragged run, and `wanted` cuts each row while `required`
    /// still counts what the walks scored.
    #[test]
    fn multiplication_reports_every_scored_pair() -> Result<()> {
        const DIMENSIONS: usize = 4;
        const ROWS: usize = 3;
        const WANTED: usize = 2;
        // A product is a logit, so the linear-algebra verbs take the inner kind alone.
        let index = empty_index(DIMENSIONS, DenseMetric::Inner, Scalar::F32)?;
        let slab = seeded_rows(ROWS, DIMENSIONS, SEED);
        index.insert(sequential_keys(ROWS), &rows(&slab, DIMENSIONS)?)?;

        let left = rows(&slab, DIMENSIONS)?;
        let every = index.multiply(&left, Arena::default())?;
        assert!(
            every.rows().all(|row| row.keys.len() == ROWS),
            "every row scores every column"
        );

        let mut arena = every.into_arena();
        let config = DenseMultiplyConfig {
            wanted: WANTED,
            ..DenseMultiplyConfig::from(&mut arena)
        };
        let cut = index.on(&executor()).multiply(&left, config)?;
        assert_eq!(cut.queries(), ROWS);
        assert!(cut.rows().all(|row| row.keys.len() == WANTED));
        assert_eq!(cut.required(), ROWS * ROWS, "the count outruns the room it was given");
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
            quantization: Scalar::F32,
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
        let matches = index.on(&executor()).maxsim(&tokens, TOKENS, DOCUMENTS)?;
        assert_eq!(matches.queries(), QUERIES);
        assert!(
            matches.rows().all(|row| row.keys.len() == DOCUMENTS),
            "every document is scored, once per query"
        );
        assert!(matches!(
            index.maxsim(&tokens, TOKENS + 1, DOCUMENTS),
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
        let index = empty_index(DIMENSIONS, DenseMetric::Inner, Scalar::F32)?;
        let slab = seeded_rows(ENTRIES, DIMENSIONS, SEED);
        index.insert(sequential_keys(ENTRIES), &rows(&slab, DIMENSIONS)?)?;
        let values = seeded_rows(ENTRIES, DIMENSIONS, SEED + 1);
        fn row_of(table: &[f32], entry: usize) -> &[f32] { &table[entry * DIMENSIONS..][..DIMENSIONS] }
        let value_lookup: &ValueLookup<f32> =
            &|key| u64::try_from(key).ok().map(|entry| row_of(&values, entry as usize));

        for head_count in [1, 2] {
            let queries = seeded_rows(TOKENS * head_count, DIMENSIONS, SEED + 2);
            let mut attended = vec![0.0f32; TOKENS * head_count * DIMENSIONS];
            let mut carry = vec![DenseSoftmaxState::default(); TOKENS * head_count];
            index.on(&executor()).attend(
                &rows(&queries, DIMENSIONS)?,
                value_lookup,
                DenseAttendConfig {
                    head_count,
                    expansion_factor: 2 * ENTRIES,
                    carry: Some(&mut carry),
                    ..DenseAttendConfig::from(&mut rows_mut(&mut attended, DIMENSIONS)?)
                },
            )?;
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
                    assert!((f64::from(attended[query * DIMENSIONS + column]) - expected).abs() < 1e-4);
                }
                assert!((f64::from(carry[query].maximum_score) - maximum).abs() < 1e-4);
                assert!((f64::from(carry[query].partition) - partition).abs() < 1e-4);
            }
        }

        let mut attended = vec![0.0f32; DIMENSIONS];
        let refused = index.attend(
            &rows(&slab[..DIMENSIONS], DIMENSIONS)?,
            &|_| None::<&[f32]>,
            &mut rows_mut(&mut attended, DIMENSIONS)?,
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
            &mut rows_mut(&mut attended, DIMENSIONS)?,
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
        let (index, _, slab) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        assert!(matches!(
            index
                .multiply(&rows(&slab[..DIMENSIONS], DIMENSIONS)?, Arena::default())
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
        let (index, _, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        index.on(&executor()).save(&path)?;

        let restored = DenseIndex::load_ram(&path)?.describe()?;
        assert_eq!(restored.metadata.dimensions, DIMENSIONS);
        assert_eq!(restored.size, 32);

        // A view serves the same checkpoint in place and refuses to change it.
        let viewed = DenseIndex::view_ram(&path)?;
        assert_eq!(viewed.describe()?.size, 32);
        assert!(matches!(
            viewed.reserve(64),
            Err(Error {
                status: Status::ReadOnly,
                ..
            })
        ));
        drop(viewed);
        remove_quietly(&path);
        Ok(())
    }

    #[test]
    fn saves_and_loads_a_buffer() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, _, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let monolith = index.save_buffer(Arena::default())?;
        assert!(!monolith.is_empty());

        let restored = DenseIndex::load_ram_buffer(&monolith)?;
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

        let mut arena = Arena::default();
        let buffer = index.save_buffer(DenseSaveBufferConfig {
            progress: Some(&mut |_, _| ControlFlow::Continue(())),
            ..DenseSaveBufferConfig::from(&mut arena)
        })?;
        assert_eq!(DenseIndex::metadata_buffer(&buffer)?.dimensions, DIMENSIONS);
        remove_quietly(&path);
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn rejects_a_non_utf8_path() -> Result<()> {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let index = empty_index(8, DenseMetric::Angular, Scalar::F32)?;
        let path = std::path::Path::new(OsStr::from_bytes(&[0xff, 0xfe]));
        assert!(
            matches!(index.save(path), Err(Error { status: Status::Io, .. })),
            "the C interface exchanges UTF-8 bytes"
        );
        Ok(())
    }

    // endregion: Serialization

    // region: Merging and Iteration

    #[test]
    fn merges_a_donor_index() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, keys, _) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let donor = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let donor_keys: Vec<Key> = (100..132u64).map(Key::from).collect();
        let donor_slab = seeded_rows(32, DIMENSIONS, SEED + 1);
        donor.insert(&donor_keys, &rows(&donor_slab, DIMENSIONS)?)?;

        // The donors are consumed by value, so a merged handle cannot be used again.
        index.on(&executor()).merge([donor])?;
        assert_eq!(index.describe()?.size, 64);
        assert!(index.contains(keys[0])? && index.contains(donor_keys[0])?);
        Ok(())
    }

    #[test]
    fn walks_entries_in_batches() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 20;
        const BUFFERED: usize = 8;
        let (index, keys, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let executor = executor();
        let receiver = index.on(&executor);
        let mut cursor = receiver.cursor()?;
        let mut seen = HashSet::new();
        let mut batches = Vec::new();
        let mut batch_keys = [Key::NIL; BUFFERED];
        let mut batch_vectors = Matrix::<f32>::zeros(&[BUFFERED, DIMENSIONS]).unwrap();
        loop {
            let written = cursor.next_batch(&mut batch_keys, &mut batch_vectors)?;
            if written == 0 {
                break;
            }
            batches.push(written);
            seen.extend(batch_keys[..written].iter().copied());
        }

        assert_eq!(batches, vec![8, 8, 4], "the batch boundary is crossed twice");
        assert_eq!(seen, keys.iter().copied().collect::<HashSet<Key>>());
        Ok(())
    }

    #[test]
    fn walks_entries_without_vectors() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, keys, _) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let mut cursor = index.cursor()?;
        let mut batch_keys = [Key::NIL; 8];
        let written = cursor.next_keys(&mut batch_keys)?;
        assert_eq!(written, 4);
        assert_eq!(
            batch_keys[..written].iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
        Ok(())
    }

    #[test]
    fn streams_matches_in_pulls() -> Result<()> {
        const DIMENSIONS: usize = 16;
        // Past the stream's internal refill boundary, so the batch seam is exercised.
        const COUNT: usize = 200;
        let (index, _, slab) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let query = rows(&slab[..DIMENSIONS], DIMENSIONS)?;
        let mut stream = index.search_stream(query)?;
        let first: Vec<Match> = stream
            .by_ref()
            .take(5)
            .map(|found| found.expect("the stream must not fault"))
            .collect();
        assert_eq!(first.len(), 5, "a pull hands the caller as many as it asked for");

        let drained = stream.count();
        assert!(drained > MATCHES_PER_PULL, "the stream crosses its refill boundary");

        let mut exhausted = index.search_stream(DenseSearchStreamInitConfig {
            expansion_factor: 64,
            ..DenseSearchStreamInitConfig::from(query)
        })?;
        while exhausted.next().is_some() {}
        assert!(exhausted.next().is_none(), "a drained stream stays drained");

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

    // endregion: Merging and Iteration

    // region: Progress and Cancellation

    #[test]
    fn a_reported_batch_counts_up_to_the_rows_it_was_given() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
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

    /// Every row is claimed before the first report, so a stop is ignored and the batch publishes
    /// whole.
    #[test]
    fn a_stopped_batch_publishes_everything() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
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
        let (index, _, slab) = populated_index(32, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let mut watch = Watch::new(Patience::Complete);
        index.on(&executor()).search(
            &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?,
            DenseSearchConfig {
                progress: Some(&mut |finished, total| watch.saw(finished, total)),
                ..DenseSearchConfig::from(WANTED)
            },
        )?;

        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (QUERIES, QUERIES));
        Ok(())
    }

    /// Stopping a sweep is the time budget rather than a failure, so nothing is reported.
    #[test]
    fn a_stopped_refinement_is_not_a_failure() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 32;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let mut watch = Watch::new(Patience::StopAtFirstReport);
        index.on(&executor()).refine(DenseRefineConfig {
            progress: Some(&mut |finished, total| watch.saw(finished, total)),
            ..DenseRefineConfig::default()
        })?;

        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last.1, COUNT, "the sweep is denominated in stored entries");
        assert!(watch.last.0 != 0, "the handler saw the sweep start");
        Ok(())
    }

    #[test]
    fn a_reported_checkpoint_counts_what_it_wrote_and_placed() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 32;
        let path = scratch_path("a_reported_checkpoint");
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

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

    #[test]
    fn a_reported_merge_counts_its_donors() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 16;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let donors = [
            populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?.0,
            populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?.0,
        ];
        let mut watch = Watch::new(Patience::Complete);
        index.on(&executor()).merge(DenseMergeConfig {
            progress: Some(&mut |finished, total| watch.saw(finished, total)),
            ..DenseMergeConfig::from(donors)
        })?;

        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (2, 2), "one report per donor absorbed");
        assert_eq!(index.describe()?.size, COUNT * 3);
        Ok(())
    }

    /// A stopped merge keeps what it absorbed, and the donors past the boundary are freed with the
    /// call - they were handed over by value, so a leak or a double free would surface here.
    #[test]
    fn a_stopped_merge_keeps_the_donors_it_absorbed() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 16;
        const DONORS: usize = 2;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;

        let donors = vec![
            populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?.0,
            populated_index(COUNT, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?.0,
        ];
        let mut watch = Watch::new(Patience::StopAtFirstReport);
        let merged = index.merge(DenseMergeConfig {
            progress: Some(&mut |finished, total| watch.saw(finished, total)),
            ..DenseMergeConfig::from(donors)
        });

        match merged {
            Ok(()) => assert_eq!(watch.last.0, DONORS, "an unstopped merge absorbed them all"),
            Err(failure) => assert_eq!(failure.status, Status::Cancelled),
        }
        assert_eq!(index.describe()?.size, COUNT * (1 + watch.last.0));
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

        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        index.reserve(COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let executor = Executor::for_ram(&[], THREADS, 0)?;
        let receiver = index.on(&executor);
        let vectors = rows(&slab, DIMENSIONS)?;
        receiver.insert(&keys, &vectors)?;
        assert_eq!(receiver.describe()?.size, COUNT);

        let config = DenseSearchConfig {
            expansion_factor: 64,
            ..DenseSearchConfig::from(WANTED)
        };
        let matches = receiver.search(&vectors, config)?;
        assert!(
            matches.rows().all(|row| row.keys.len() == WANTED),
            "every query fills its own output row"
        );
        Ok(())
    }

    /// The index itself is the blocking receiver on the calling thread, with the same results.
    #[test]
    fn runs_on_the_calling_thread() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, _, slab) = populated_index(16, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let query = rows(&slab[..DIMENSIONS], DIMENSIONS)?;
        let on_this_thread = index.search(&query, 4)?;
        let on_the_null_executor = index.on(&Executor::default()).search(&query, 4)?;
        assert_eq!(on_this_thread.row(0).unwrap().keys.len(), 4);
        assert_eq!(on_this_thread.row(0), on_the_null_executor.row(0));
        Ok(())
    }

    #[test]
    fn searches_while_inserting() -> Result<()> {
        const DIMENSIONS: usize = 64;
        const HALF: usize = 500;
        const WANTED: usize = 4;

        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        index.reserve(2 * HALF)?;
        let keys = sequential_keys(2 * HALF);
        let slab = seeded_rows(2 * HALF, DIMENSIONS, SEED);

        // The first half lands serially, so the reader below traverses a populated graph.
        index.insert(&keys[..HALF], &rows(&slab[..HALF * DIMENSIONS], DIMENSIONS)?)?;

        // Calls on one executor run in order, so genuinely overlapping calls need one apiece -
        // the writer's and the reader's, on two threads sharing the handle.
        let writer = Executor::for_ram(&[], THREADS, 0)?;
        let reader = Executor::for_ram(&[], THREADS, 0)?;
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
                let mut found = vec![Key::NIL; HALF * WANTED];
                let mut scores = vec![0.0 as Distance; HALF * WANTED];
                let mut counts = vec![0usize; HALF];
                let mut matches =
                    MatchesSpan::new(&mut found, &mut scores, &mut counts).expect("the output buffers are one shape");
                let queries = rows(&slab[..HALF * DIMENSIONS], DIMENSIONS).unwrap();
                let config = DenseSearchConfig {
                    expansion_factor: 32,
                    ..DenseSearchConfig::from(&mut matches)
                };
                index
                    .on(&reader)
                    .search(&queries, config)
                    .expect("the racing search must resolve");
                // Whether a racing insert is visible is what the watermark leaves unspecified;
                // that no key outside the dataset appears is what it guarantees.
                for row in matches.rows() {
                    if row.keys.iter().any(|key| !known.contains(key)) {
                        strangers.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        });

        assert_eq!(
            strangers.load(Ordering::Relaxed),
            0,
            "searches never see unpublished rows"
        );
        assert_eq!(index.describe()?.size, 2 * HALF);
        Ok(())
    }

    // endregion: Concurrency

    // region: Deferred Calls

    /// A deferred search owns its queries and answers what the blocking one does, once waited on.
    #[test]
    fn a_deferred_search_matches_the_blocking_one() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const QUERIES: usize = 4;
        const WANTED: usize = 5;
        let (index, _, slab) = populated_index(64, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let queries = Matrix::<f32>::from_slice(&slab[..QUERIES * DIMENSIONS], &[QUERIES, DIMENSIONS]).unwrap();

        let executor = executor();
        let call = index.deferred(&executor).search(queries, WANTED)?;
        requires_send::<Deferred<'static, DenseSearch<'static, Matrix<f32>, Matches>>>();
        assert_eq!(call.status(), Err(Status::Deferred));
        executor.drain(Mood::Parked)?;
        let deferred = call.wait()?;

        let blocking = index.search(&rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?, WANTED)?;
        assert_eq!(deferred.queries(), QUERIES);
        for query in 0..QUERIES {
            assert_eq!(deferred.row(query), blocking.row(query));
        }
        Ok(())
    }

    /// An owned arena and owned outputs travel with the call and come back with its result.
    #[test]
    fn a_deferred_call_hands_back_what_it_owned() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let (index, keys, slab) = populated_index(4, DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let executor = executor();
        let receiver = index.deferred(&executor);

        let found = receiver.get::<f32, _>(keys[2], Arena::default())?;
        let buffer = receiver.save_buffer(Arena::default())?;
        let counted = receiver.count(keys[3])?;
        executor.drain(Mood::Parked)?;
        assert_eq!(flattened(found.wait()?.view()), &slab[DIMENSIONS * 2..DIMENSIONS * 3]);
        assert_eq!(DenseIndex::metadata_buffer(&buffer.wait()?)?.dimensions, DIMENSIONS);
        assert_eq!(counted.wait()?, 1);
        Ok(())
    }

    /// A call the executor finished needs no wait: dropping it frees what it owned.
    #[test]
    fn a_drained_deferred_call_drops_without_a_wait() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);
        let vectors = Matrix::<f32>::from_slice(&slab, &[COUNT, DIMENSIONS]).unwrap();

        let executor = executor();
        let call = index.deferred(&executor).insert(sequential_keys(COUNT), vectors)?;
        executor.drain(Mood::Parked)?;
        drop(call);
        assert_eq!(index.describe()?.size, COUNT);
        Ok(())
    }

    /// A submission finding the executor's deferred table full is refused rather than waited for.
    #[test]
    fn a_full_deferred_table_answers_contended() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 16;
        let index = empty_index(DIMENSIONS, DenseMetric::Angular, Scalar::F32)?;
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);
        let vectors = || Matrix::<f32>::from_slice(&slab, &[COUNT, DIMENSIONS]).unwrap();

        let executor = Executor::for_ram(&[], 1, 1)?;
        let receiver = index.deferred(&executor);
        let first = receiver.insert(sequential_keys(COUNT), vectors())?;
        let second = receiver.insert(sequential_keys(COUNT), vectors())?;
        assert_eq!(second.status(), Err(Status::Contended));
        executor.drain(Mood::Parked)?;
        first.wait()?;
        assert_eq!(index.describe()?.size, COUNT);
        Ok(())
    }

    // endregion: Deferred Calls
}

// endregion: Tests
