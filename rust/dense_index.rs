//! Vector Search over equidimensional dense embeddings - the Rust twin of
//! `include/usearch/dense_index.h`.
//!
//! Every batch input is a NumKong matrix, so a memory-mapped dataset feeds the index with no
//! reshape and no copy: packed slabs go through [`crate::rows`], padded ones through
//! `MatrixView::from_raw_parts`. Handles are created against one residency - RAM,
//! VRAM, or NVMe - picked by [`Residency`], which also decides what loading a checkpoint
//! means: materialize into RAM, upload to a device, or open in place on disk.
//!
//! Every verb with optional arguments is a builder that runs when `run` is called, blocking until
//! the call is done. Outputs whose size only the call discovers borrow the caller's [`Arena`],
//! which the next such call rewinds; ranked outputs can land in caller-owned buffers instead.
//!
//! A handle may be shared across threads, which is why the mutating verbs take `&self`.

use core::ffi::{c_int, c_void};
use core::marker::PhantomData;
use core::ptr;
use std::path::Path;

use numkong::{Allocator, MatrixSpan, MatrixView, StorageElement, TensorRef};

use crate::types::ffi::usearch_outcome_t;
use crate::types::{
    arena_slice, check, fail, lower_progress, nvme_options, path_view, path_views, ram_options, static_string,
    vram_options, Arena, Distance, Executor, Key, Progress, Residency, Result, ScalarKind, Status,
};

// region: Raw Declarations

/// One-to-one mirror of `include/usearch/dense_index.h`.
mod ffi {
    #![allow(non_camel_case_types)]
    // A one-to-one mirror declares every verb, including the ones no method drives yet.
    #![allow(dead_code)]

    use core::ffi::{c_char, c_int, c_void};

    pub use crate::types::ffi::{
        usearch_arena_t, usearch_bytes_view_t, usearch_completion_t, usearch_distance_t, usearch_executor_t,
        usearch_key_t, usearch_keys_span_t, usearch_keys_view_t, usearch_matches_span_t, usearch_matrix_view_t,
        usearch_nvme_options_t, usearch_outcome_t, usearch_predicate_t, usearch_progress_callback_t,
        usearch_ram_options_t, usearch_rows_span_t, usearch_rows_view_t, usearch_scalar_t, usearch_vram_options_t,
    };

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

    pub type usearch_dense_traversal_stats_t = super::Stats;

    #[repr(C)]
    pub struct usearch_dense_add_t {
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

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct usearch_dense_reachability_t {
        pub audited: usize,
        pub reachable: usize,
        pub sweeps: usize,
    }

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

    pub type usearch_dense_softmax_state_t = super::SoftmaxState;

    #[repr(C)]
    pub struct usearch_dense_attend_t {
        pub index: usearch_dense_index_t,
        pub executor: usearch_executor_t,
        pub outcome: *mut usearch_outcome_t,
        pub completion: usearch_completion_t,
        pub count: usize,
        pub queries: usearch_rows_view_t,
        pub values: usearch_matrix_view_t,
        pub outputs: usearch_rows_span_t,
        pub carry: *mut usearch_dense_softmax_state_t,
        pub expansion_factor: usize,
        pub score_scale: f32,
        pub mass_tolerance: f32,
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
        pub fn usearch_dense_add(arguments: *mut usearch_dense_add_t);
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
}

// endregion: Raw Declarations

// region: Options

/// The dense metric kinds supported out of the box, all derived from a single inner-product
/// kernel with pre-computed magnitudes. Hamming and Jaccard are bit-level, over the packed
/// sub-byte kinds.
///
/// Arbitrary metrics belong to the C++ baseline tier, which forfeits the SIMD kernel chains and
/// the magnitude cache both.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum Metric {
    Inner = 1,
    #[default]
    Angular = 2,
    Euclidean = 3,
    Hamming = 4,
    Jaccard = 5,
}

impl Metric {
    fn from_raw(value: i32) -> Option<Self> {
        let metric = match value {
            1 => Metric::Inner,
            2 => Metric::Angular,
            3 => Metric::Euclidean,
            4 => Metric::Hamming,
            5 => Metric::Jaccard,
            _ => return None,
        };
        Some(metric)
    }
}

/// Construction options of a dense index.
///
/// Stored rows always round up to the SIMD step quantum, so rows stay cache-line aligned and
/// every kernel runs tail-free; the arena is slightly larger for it, and that is not a knob.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct Options {
    /// The number of indexed dimensions; zero leaves the geometry unfrozen, so the index admits
    /// no rows.
    pub dimensions: usize,
    /// The metric kind used for distance calculations.
    pub metric: Metric,
    /// The scalar type vectors are stored and quantized into.
    pub quantization: ScalarKind,
    /// The connections-per-node bound of the graph; zero derives.
    pub connectivity: usize,
    /// Upper bound on entries sharing one key; zero derives to one - unique keys.
    pub values_per_key: usize,
    /// Where the payload lives.
    pub residency: Residency,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            dimensions: 0,
            metric: Metric::Angular,
            quantization: ScalarKind::BF16,
            connectivity: 0,
            values_per_key: 0,
            residency: Residency::default(),
        }
    }
}

/// The geometry and metric a checkpoint froze - everything a reader learns without opening it.
///
/// `values_per_key` is absent because it bounds a live handle's mutable state rather than the
/// serialized form, so a checkpoint records nothing about it. Placement is absent for the same
/// kind of reason: a checkpoint says where it was written, never where a reader must keep it.
/// Both stay the caller's to choose when loading.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct Metadata {
    /// Scalars per stored vector.
    pub dimensions: usize,
    /// The metric kind the checkpoint scores with.
    pub metric: Metric,
    /// The scalar type its rows are stored and quantized into.
    pub quantization: ScalarKind,
    /// Connections per node the graph was built with.
    pub connectivity: usize,
}

impl Metadata {
    /// Lifts the C structure, refusing a kind this binding has no name for.
    fn from_raw(raw: &ffi::usearch_dense_metadata_t) -> Result<Self> {
        Ok(Metadata {
            dimensions: raw.dimensions,
            metric: Metric::from_raw(raw.metric)
                .ok_or(fail(Status::UnknownKind, "the checkpoint names an unknown metric"))?,
            quantization: ScalarKind::from_raw(raw.quantization)
                .ok_or(fail(Status::UnknownKind, "the checkpoint names an unknown scalar kind"))?,
            connectivity: raw.connectivity,
        })
    }
}

/// A live index's frozen metadata and current counters, each read once, so under concurrent
/// insertions they may come from slightly different moments.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct Description {
    /// The frozen metric, quantization, dimensions and connectivity, as a checkpoint reports them.
    pub metadata: Metadata,
    /// Stored entries.
    pub size: usize,
    /// Entries hosted without growing again.
    pub capacity: usize,
    /// Bytes the handle holds; traversal scratch is the executor's.
    pub memory_usage: usize,
    /// The capability codename the index dispatched to on this machine.
    pub capabilities: &'static str,
}

// endregion: Options

// region: Telemetry

/// The online-softmax carry of one query row - Flash Attention's running maximum and partition.
///
/// The two are one concept: a maximum without its partition is a number nothing can rescale, so
/// they travel as one element rather than as two arrays a caller could supply half of. Carrying
/// them is what folds one attention call into another over the same queries.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
pub struct SoftmaxState {
    /// The largest logit seen so far - the recurrence's `m`.
    pub maximum_score: f32,
    /// The sum of exponentials taken against that maximum - the recurrence's `l`.
    pub partition: f32,
}

/// Traversal telemetry, fully overwritten by the call it is passed to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(C)]
#[non_exhaustive]
pub struct Stats {
    /// Graph members visited by the call.
    pub visited_members: usize,
    /// Distances computed by the call, diversity re-selections included.
    pub computed_distances: usize,
    /// Starvation restarts - a graph-connectivity health signal, not a cost counter.
    pub restarts: usize,
}

// Every call overwrites its stats rather than accumulating, so a sweep over many batches
// sums them itself.
impl core::ops::AddAssign for Stats {
    fn add_assign(&mut self, other: Stats) {
        self.visited_members += other.visited_members;
        self.computed_distances += other.computed_distances;
        self.restarts += other.restarts;
    }
}

// endregion: Telemetry

// region: Results

/// One retrieved entry.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Match {
    /// The matched entry's key.
    pub key: Key,
    /// Its distance to the query, in the index's metric.
    pub distance: Distance,
}

/// One query's matches, trimmed to the cells the call wrote.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MatchesRow<'matches> {
    /// The keys written for this query, nearest first.
    pub keys: &'matches [Key],
    /// Their distances, positionally paired with the keys.
    pub distances: &'matches [Distance],
}

/// Reads one row of a `wanted`-wide ranked slab.
fn row_of<'slab>(
    keys: &'slab [Key],
    distances: &'slab [Distance],
    counts: &[usize],
    wanted: usize,
    query: usize,
) -> Option<MatchesRow<'slab>> {
    let start = query * wanted;
    let end = start + (*counts.get(query)?).min(wanted);
    Some(MatchesRow {
        keys: keys.get(start..end)?,
        distances: distances.get(start..end)?,
    })
}

/// Lends three packed `wanted`-wide slabs as the span a ranked call writes, keeping up to
/// `wanted` matches per row.
fn matches_span(
    keys: &mut [Key],
    distances: &mut [Distance],
    counts: &mut [usize],
    columns: usize,
    wanted: usize,
) -> ffi::usearch_matches_span_t {
    ffi::usearch_matches_span_t {
        keys: keys.as_mut_ptr(),
        keys_stride: columns * core::mem::size_of::<Key>(),
        scores: distances.as_mut_ptr(),
        scores_stride: columns * core::mem::size_of::<Distance>(),
        counts: counts.as_mut_ptr(),
        capacity: counts.len(),
        wanted,
    }
}

/// Where a ranked call writes its results - three caller-owned buffers, sized together.
///
/// The buffer lengths decide the row width: `keys.len() / counts.len()` columns per query, which
/// may exceed the call's wanted count to land results inside a wider matrix.
pub struct MatchesSpan<'buffers> {
    /// The `queries × wanted` key slab, rows contiguous.
    keys: &'buffers mut [Key],
    /// The `queries × wanted` distance slab, rows contiguous.
    distances: &'buffers mut [Distance],
    /// One match count per query.
    counts: &'buffers mut [usize],
    /// Columns per row.
    wanted: usize,
}

impl<'buffers> MatchesSpan<'buffers> {
    /// Lends three packed slabs: `counts.len() × wanted` keys and distances, one count per query.
    ///
    /// The row width follows from the lengths rather than being asked for, so the space reserved
    /// and the width assumed cannot disagree.
    pub fn new(
        keys: &'buffers mut [Key],
        distances: &'buffers mut [Distance],
        counts: &'buffers mut [usize],
    ) -> Result<Self> {
        if counts.is_empty() {
            return Err(fail(Status::ShapeMismatch, "a run must hold at least one query"));
        }
        if keys.len() != distances.len() {
            return Err(fail(
                Status::ShapeMismatch,
                "the key and distance outputs are not the same shape",
            ));
        }
        if keys.is_empty() || !keys.len().is_multiple_of(counts.len()) {
            return Err(fail(
                Status::ShapeMismatch,
                "the outputs do not divide into one row per query",
            ));
        }
        let wanted = keys.len() / counts.len();
        Ok(MatchesSpan {
            keys,
            distances,
            counts,
            wanted,
        })
    }

    /// Columns per row.
    pub fn wanted(&self) -> usize { self.wanted }

    /// Query rows the buffers were sized for.
    pub fn queries(&self) -> usize { self.counts.len() }

    /// Matches written for one query, keys and distances read together, or `None` when `query`
    /// is not below [`MatchesSpan::queries`].
    pub fn row(&self, query: usize) -> Option<MatchesRow<'_>> {
        row_of(self.keys, self.distances, self.counts, self.wanted, query)
    }

    /// Every query's row, in query order.
    pub fn rows(&self) -> impl Iterator<Item = MatchesRow<'_>> {
        (0..self.queries()).filter_map(|query| self.row(query))
    }

    /// Lends the three outputs to a call keeping up to `wanted` matches per row.
    fn lower(&mut self, wanted: usize) -> ffi::usearch_matches_span_t {
        matches_span(self.keys, self.distances, self.counts, self.wanted, wanted)
    }
}

/// The matches a ranked call returns when no span was lent to it; see [`DenseIndex::search`].
pub struct Matches {
    keys: Vec<Key>,
    distances: Vec<Distance>,
    counts: Vec<usize>,
    wanted: usize,
}

impl Matches {
    /// Sizes the slabs for `queries` rows of `wanted` columns.
    fn new(queries: usize, wanted: usize) -> Result<Self> {
        let cells = queries.checked_mul(wanted).ok_or(fail(
            Status::Overflow,
            "the outputs would not fit an addressable buffer",
        ))?;
        Ok(Matches {
            keys: vec![Key::NIL; cells],
            distances: vec![0.0; cells],
            counts: vec![0; queries],
            wanted,
        })
    }

    /// The most matches any row holds.
    pub fn wanted(&self) -> usize { self.wanted }

    /// Query rows the call resolved.
    pub fn queries(&self) -> usize { self.counts.len() }

    /// Matches written for one query, or `None` when `query` is not below [`Matches::queries`].
    pub fn row(&self, query: usize) -> Option<MatchesRow<'_>> {
        row_of(&self.keys, &self.distances, &self.counts, self.wanted, query)
    }

    /// Every query's row, in query order.
    pub fn rows(&self) -> impl Iterator<Item = MatchesRow<'_>> {
        (0..self.queries()).filter_map(|query| self.row(query))
    }

    fn lower(&mut self) -> ffi::usearch_matches_span_t {
        matches_span(
            &mut self.keys,
            &mut self.distances,
            &mut self.counts,
            self.wanted,
            self.wanted,
        )
    }
}

/// The ragged sparse product [`Multiply::run`] lands in an arena: per left row, every pair its
/// walk scored, in traversal order.
pub struct Products<'arena> {
    offsets: Vec<usize>,
    keys: &'arena [Key],
    scores: &'arena [Distance],
    required: usize,
}

impl Products<'_> {
    /// Left rows the call multiplied.
    pub fn queries(&self) -> usize { self.offsets.len() - 1 }

    /// Pairs the walks scored across every row, before the wanted bound cut any.
    pub fn required(&self) -> usize { self.required }

    /// The pairs of one left row - a key naming each indexed row and its product - or `None`
    /// when `row` is not below [`Products::queries`].
    pub fn row(&self, row: usize) -> Option<MatchesRow<'_>> {
        let offsets = self.offsets.get(row..row + 2)?;
        Some(MatchesRow {
            keys: self.keys.get(offsets[0]..offsets[1])?,
            distances: self.scores.get(offsets[0]..offsets[1])?,
        })
    }

    /// Every left row's pairs, in row order.
    pub fn rows(&self) -> impl Iterator<Item = MatchesRow<'_>> { (0..self.queries()).filter_map(|row| self.row(row)) }
}

// endregion: Results

// region: Dense Index

/// Vector Search Index over equidimensional dense representations.
///
/// ```no_run
/// use usearch::dense::{DenseIndex, Metric, Options};
/// use usearch::{rows, Executor, Key, ScalarKind};
///
/// let mut options = Options::default();
/// options.dimensions = 4;
/// options.metric = Metric::Angular;
/// options.quantization = ScalarKind::I8;
///
/// let index = DenseIndex::new(&options)?;
/// let executor = Executor::for_ram(&[], 0, 0)?;
/// index.reserve(&executor, 1_000)?;
///
/// let keys = [Key::from(0u64), Key::from(1u64)];
/// let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
/// index.add(&executor, &keys, &rows(&slab, 4)?).run()?;
///
/// let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
/// let matches = index.search(&executor, &rows(&query, 4)?, 10).run()?;
/// for row in matches.rows() {
///     println!("{} matches", row.keys.len());
/// }
/// # Ok::<(), usearch::Error>(())
/// ```
#[derive(Debug)]
pub struct DenseIndex {
    handle: ffi::usearch_dense_index_t,
}

// Safety: the library documents a handle as shareable across threads - searches never write to
// the graph, and insertions synchronize through per-node locks and publish in order.
unsafe impl Send for DenseIndex {}
unsafe impl Sync for DenseIndex {}

impl Drop for DenseIndex {
    fn drop(&mut self) { unsafe { ffi::usearch_dense_free(self.handle) } }
}

impl DenseIndex {
    /// Initializes a new, empty index for the residency the options select.
    ///
    /// An NVMe index is never initialized, only opened in place through [`DenseIndex::load`].
    pub fn new(options: &Options) -> Result<Self> {
        let mut index = DenseIndex {
            handle: ptr::null_mut(),
        };
        let mut outcome = usearch_outcome_t::default();
        let (metric, quantization) = (options.metric as i32, options.quantization as i32);
        match &options.residency {
            Residency::Ram { memory_domains } => unsafe {
                ffi::usearch_dense_init_ram(&mut ffi::usearch_dense_init_ram_t {
                    index: &mut index.handle,
                    outcome: &mut outcome,
                    metric,
                    quantization,
                    dimensions: options.dimensions,
                    connectivity: options.connectivity,
                    values_per_key: options.values_per_key,
                    ram: ram_options(memory_domains),
                })
            },
            Residency::Vram { devices } => unsafe {
                ffi::usearch_dense_init_vram(&mut ffi::usearch_dense_init_vram_t {
                    index: &mut index.handle,
                    outcome: &mut outcome,
                    metric,
                    quantization,
                    dimensions: options.dimensions,
                    connectivity: options.connectivity,
                    values_per_key: options.values_per_key,
                    vram: vram_options(devices),
                })
            },
            Residency::Nvme { .. } => {
                return Err(fail(
                    Status::Unsupported,
                    "an NVMe index serves a checkpoint in place, so it is loaded, never initialized",
                ))
            }
        }
        check(outcome).map(|()| index)
    }

    /// Copies a checkpoint into a new index, or opens it in place, as the residency picks.
    pub fn load<'a>(executor: &'a Executor, path: &'a (impl AsRef<Path> + ?Sized)) -> Load<'a> {
        Load {
            executor,
            source: Source::Path(path.as_ref()),
            residency: Residency::default(),
            values_per_key: 0,
            progress: None,
        }
    }

    /// Copies a monolith from a buffer into a new index, as [`DenseIndex::load`] does a path.
    pub fn load_buffer<'a>(executor: &'a Executor, buffer: &'a [u8]) -> Load<'a> {
        Load {
            executor,
            source: Source::Buffer(buffer),
            residency: Residency::default(),
            values_per_key: 0,
            progress: None,
        }
    }

    /// Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
    /// verbs report [`Status::ReadOnly`]; the files must stay unmodified while it lives.
    pub fn view(path: impl AsRef<Path>) -> Result<Self> {
        let path = path_view(path.as_ref())?;
        let mut index = DenseIndex {
            handle: ptr::null_mut(),
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_view_ram(&mut ffi::usearch_dense_view_ram_t {
                index: &mut index.handle,
                outcome: &mut outcome,
                path,
            })
        };
        check(outcome).map(|()| index)
    }

    /// Reads a checkpoint's geometry and metric without loading it; a manifest path answers
    /// from its summary alone, so no shard is opened.
    pub fn metadata(executor: &Executor, path: impl AsRef<Path>) -> Result<Metadata> {
        let path = path_view(path.as_ref())?;
        let mut raw = ffi::usearch_dense_metadata_t::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_metadata_read(&mut ffi::usearch_dense_metadata_read_t {
                executor: executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                path,
                metadata: &mut raw,
            })
        };
        check(outcome)?;
        Metadata::from_raw(&raw)
    }

    /// Reads geometry and metric out of a serialized monolith buffer.
    pub fn metadata_buffer(executor: &Executor, buffer: &[u8]) -> Result<Metadata> {
        let mut raw = ffi::usearch_dense_metadata_t::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_metadata_read_buffer(&mut ffi::usearch_dense_metadata_read_buffer_t {
                executor: executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                buffer: buffer.into(),
                metadata: &mut raw,
            })
        };
        check(outcome)?;
        Metadata::from_raw(&raw)
    }

    /// Reserves capacity for a number of incoming vectors ahead of insertion.
    pub fn reserve(&self, executor: &Executor, capacity: usize) -> Result<()> {
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_reserve(&mut ffi::usearch_dense_reserve_t {
                index: self.handle,
                executor: executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                capacity,
            })
        };
        check(outcome)
    }

    /// Erases all entries, keeping the reserved capacity and frozen geometry.
    pub fn clear(&self, executor: &Executor) -> Result<()> {
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_clear(&mut ffi::usearch_dense_clear_t {
                index: self.handle,
                executor: executor.handle(),
                outcome: &mut outcome,
                completion: 0,
            })
        };
        check(outcome)
    }

    /// Reports the frozen metadata and the current counters in one call.
    pub fn describe(&self, executor: &Executor) -> Result<Description> {
        let mut raw = ffi::usearch_dense_index_description_t {
            metadata: ffi::usearch_dense_metadata_t::default(),
            size: 0,
            capacity: 0,
            memory_usage: 0,
            capabilities: ptr::null(),
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_describe(&mut ffi::usearch_dense_describe_t {
                index: self.handle,
                executor: executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                description: &mut raw,
            })
        };
        check(outcome)?;
        Ok(Description {
            metadata: Metadata::from_raw(&raw.metadata)?,
            size: raw.size,
            capacity: raw.capacity,
            memory_usage: raw.memory_usage,
            capabilities: unsafe { static_string(raw.capabilities) },
        })
    }

    /// Adds a matrix of vectors under their keys - the native batch operation.
    ///
    /// A batch is all-or-nothing, a cancelled one included, since every fallible step runs before
    /// the first entry identity is claimed. Overlaps safely with concurrent searches on the same
    /// handle, and the input scalar type is independent of the stored quantization - the library
    /// converts on ingest.
    pub fn add<'a, Scalar: crate::types::Scalar>(
        &'a self,
        executor: &'a Executor,
        keys: &'a [Key],
        vectors: &'a MatrixView<'_, Scalar, impl Allocator>,
    ) -> Add<'a> {
        // The keys view carries no count, so only this side can see the two disagree.
        let batch = batch_of(vectors).and_then(|batch| match batch.count == keys.len() {
            true => Ok(batch),
            false => Err(fail(
                Status::ShapeMismatch,
                "the key count does not match the vector row count",
            )),
        });
        Add {
            index: self,
            executor,
            keys,
            batch,
            expansion_factor: 0,
            progress: None,
            stats: None,
        }
    }

    /// Resolves a matrix of queries into up to `wanted` nearest matches apiece.
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    pub fn search<'a, Scalar: crate::types::Scalar>(
        &'a self,
        executor: &'a Executor,
        queries: &'a MatrixView<'_, Scalar, impl Allocator>,
        wanted: usize,
    ) -> Search<'a> {
        Search {
            index: self,
            executor,
            batch: batch_of(queries),
            wanted,
            expansion_factor: 0,
            predicate: None,
            progress: None,
            stats: None,
        }
    }

    /// The number of entries stored under `key`.
    pub fn count(&self, executor: &Executor, key: Key) -> Result<usize> {
        let mut entries = 0;
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_count(&mut ffi::usearch_dense_count_t {
                index: self.handle,
                executor: executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                key,
                entries: &mut entries,
            })
        };
        check(outcome)?;
        Ok(entries)
    }

    /// Whether the index holds at least one entry under `key`.
    pub fn contains(&self, executor: &Executor, key: Key) -> Result<bool> { Ok(self.count(executor, key)? != 0) }

    /// Retrieves every vector stored under `key` into `arena`, packed one row per entry in
    /// insertion order; an absent key yields an empty slab.
    pub fn get<'arena, Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        key: Key,
        arena: &'arena mut Arena,
    ) -> Result<&'arena [Scalar]> {
        let mut vectors = ptr::null_mut();
        let (mut written, mut stride) = (0, 0);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_get(&mut ffi::usearch_dense_get_t {
                index: self.handle,
                executor: executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                key,
                scalar: Scalar::KIND as i32,
                vectors: &mut vectors,
                written: &mut written,
                stride: &mut stride,
                arena: arena.handle(),
            })
        };
        check(outcome)?;
        let length = written * stride / core::mem::size_of::<Scalar>();
        Ok(unsafe { arena_slice(vectors.cast_const().cast(), length) })
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, returning from
    /// `run` the number of nodes whose edges changed.
    ///
    /// A build step rather than a repair: a cheap insertion pass followed by one refinement
    /// reaches a better graph, in less total time, than a single expensive insertion pass - so a
    /// wide expansion factor belongs here rather than in [`DenseIndex::add`]. A caller loops until
    /// the count falls below whatever change rate they consider settled. Searches may run
    /// throughout; insertions may not.
    pub fn refine<'a>(&'a self, executor: &'a Executor) -> Refine<'a> {
        Refine {
            index: self,
            executor,
            expansion_factor: 0,
            progress: None,
            stats: None,
        }
    }

    /// Approximate sparse product of a matrix against the indexed rows.
    ///
    /// The linear-algebra sibling of [`DenseIndex::search`], reporting every pair the traversal
    /// scored rather than only the nearest. Pairs arrive in traversal order rather than sorted.
    /// Runs on an inner-product index alone, since a product is a logit.
    pub fn multiply<'a, Scalar: crate::types::Scalar>(
        &'a self,
        executor: &'a Executor,
        left: &'a MatrixView<'_, Scalar, impl Allocator>,
    ) -> Multiply<'a> {
        Multiply {
            index: self,
            executor,
            batch: batch_of(left),
            wanted: 0,
            expansion_factor: 0,
            progress: None,
            stats: None,
        }
    }

    /// Late-interaction scoring of multi-row queries against multi-row entries, keeping up to
    /// `wanted` keys per query.
    ///
    /// Every query row takes its best match within a key, and those maxima sum to that key's
    /// score - so `rows_per_query` is the token count each query occupies, `queries` holds them
    /// end to end, and `wanted` counts documents rather than rows. This is what `values_per_key`
    /// above one exists for. Runs on an inner-product index alone.
    ///
    /// See <https://arxiv.org/abs/2004.12832> - ColBERT, which introduced this scoring.
    pub fn maxsim<'a, Scalar: crate::types::Scalar>(
        &'a self,
        executor: &'a Executor,
        queries: &'a MatrixView<'_, Scalar, impl Allocator>,
        rows_per_query: usize,
        wanted: usize,
    ) -> Maxsim<'a> {
        // The call counts queries, so only this side sees the matrix's rows fail to divide.
        let batch = batch_of(queries).and_then(|batch| {
            match rows_per_query != 0 && batch.count.is_multiple_of(rows_per_query) {
                true => Ok(batch),
                false => Err(fail(
                    Status::ShapeMismatch,
                    "the query rows do not divide into whole queries",
                )),
            }
        });
        Maxsim {
            index: self,
            executor,
            batch,
            rows_per_query,
            wanted,
            expansion_factor: 0,
            progress: None,
            stats: None,
        }
    }

    /// Softmax-weighted attention over the indexed keys, folded into the graph walk in one pass.
    ///
    /// The index holds the keys and `values` holds the value rows, addressed by key: an entry's
    /// key is its row offset into `values`, so a context inserted in position order needs no
    /// mapping. No match list is materialized, and each value row is gathered exactly once.
    /// `outputs` comes back already divided by its partition. Runs on an inner-product index
    /// alone, since a softmax needs logits.
    ///
    /// See <https://arxiv.org/abs/2205.14135> - Flash Attention, whose recurrence this reuses.
    pub fn attend<'a, Query: crate::types::Scalar, Value: crate::types::Scalar>(
        &'a self,
        executor: &'a Executor,
        queries: &'a MatrixView<'_, Query, impl Allocator>,
        values: &'a MatrixView<'_, Value, impl Allocator>,
        outputs: &'a mut MatrixSpan<'_, Value, impl Allocator>,
    ) -> Attend<'a> {
        let lowered = batch_of(queries).and_then(|batch| {
            let values_layout = MatrixLayout::of(values)?;
            let outputs_layout = MatrixLayout::of(outputs)?;
            let values = ffi::usearch_matrix_view_t {
                data: values.as_ptr().cast(),
                rows: values_layout.rows,
                columns: values_layout.columns,
                stride: values_layout.stride_bytes,
                scalar: Value::KIND as i32,
            };
            let outputs = ffi::usearch_rows_span_t {
                data: outputs.as_mut_ptr().cast(),
                stride: outputs_layout.stride_bytes,
                columns: outputs_layout.columns,
                capacity: outputs_layout.rows,
                scalar: Value::KIND as i32,
            };
            Ok((batch, values, outputs))
        });
        Attend {
            index: self,
            executor,
            lowered,
            carry: None,
            expansion_factor: 0,
            score_scale: 0.0,
            mass_tolerance: 0.0,
            progress: None,
            stats: None,
        }
    }

    /// Absorbs every donor into this index, wiring cross-links via landmark stitching.
    ///
    /// Cheaper than re-inserting the donors' vectors one by one, and readers may keep searching
    /// throughout. The donors are consumed whether or not the merge succeeds.
    pub fn merge<'a>(&'a self, executor: &'a Executor, donors: Vec<DenseIndex>) -> Merge<'a> {
        Merge {
            index: self,
            executor,
            donors,
            progress: None,
        }
    }

    /// Opens a row scan over the current contents; each shard's publication watermark is read as
    /// the walk enters it, so a row once passed is never revisited.
    pub fn cursor<Scalar: crate::types::Scalar>(&self) -> Result<Cursor<'_, Scalar>> {
        let mut cursor = Cursor {
            cursor: ptr::null_mut(),
            index: self,
            scalar: PhantomData,
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_cursor_init(&mut ffi::usearch_dense_cursor_init_t {
                cursor: &mut cursor.cursor,
                index: self.handle,
                outcome: &mut outcome,
            })
        };
        check(outcome).map(|()| cursor)
    }

    /// Opens a resumable nearest-neighbor stream for the first row of `query`.
    ///
    /// Best-first traversal is naturally pausable, so callers that post-filter aggressively pull
    /// until satisfied instead of guessing "k" upfront. The stream keeps walk state rather than a
    /// thread, and every pull is an ordinary call on `executor`.
    pub fn search_stream<'index, Scalar: crate::types::Scalar>(
        &'index self,
        executor: &'index Executor,
        query: &MatrixView<'_, Scalar, impl Allocator>,
    ) -> Result<SearchStream<'index>> {
        let query = batch_of(query)?.rows;
        let mut stream = SearchStream {
            stream: ptr::null_mut(),
            executor,
            keys: [Key::NIL; MATCHES_PER_PULL],
            distances: [0.0; MATCHES_PER_PULL],
            filled: 0,
            position: 0,
            index: self,
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_search_stream_init(&mut ffi::usearch_dense_search_stream_init_t {
                stream: &mut stream.stream,
                index: self.handle,
                outcome: &mut outcome,
                query,
                expansion_factor: 0,
            })
        };
        check(outcome).map(|()| stream)
    }

    /// Saves the index in the layout the path extension picks - one `.safegraphs` monolith, or a
    /// `.safegraphs.json` manifest beside its shards.
    ///
    /// Demands a mutation-free window; concurrent readers keep searching throughout.
    pub fn save<'a>(&'a self, executor: &'a Executor, path: &'a (impl AsRef<Path> + ?Sized)) -> Save<'a> {
        Save {
            index: self,
            executor,
            path: path.as_ref(),
            progress: None,
        }
    }

    /// Saves the index into `arena` as one monolith.
    pub fn save_buffer<'arena>(&self, executor: &Executor, arena: &'arena mut Arena) -> Result<&'arena [u8]> {
        let mut bytes = ptr::null_mut();
        let mut length = 0;
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_save_buffer(&mut ffi::usearch_dense_save_buffer_t {
                index: self.handle,
                executor: executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                bytes: &mut bytes,
                length: &mut length,
                arena: arena.handle(),
                progress: None,
                progress_state: ptr::null_mut(),
            })
        };
        check(outcome)?;
        Ok(unsafe { arena_slice(bytes.cast_const(), length) })
    }
}

// endregion: Dense Index

// region: Builders

/// A borrowed run of rows and the batch axis it spans.
#[derive(Clone, Copy)]
struct Batch {
    rows: ffi::usearch_rows_view_t,
    count: usize,
}

/// Borrows a matrix as the batch of rows a verb reads, its width passed through for the call to
/// check.
fn batch_of<Scalar: crate::types::Scalar>(matrix: &MatrixView<'_, Scalar, impl Allocator>) -> Result<Batch> {
    let layout = MatrixLayout::of(matrix)?;
    Ok(Batch {
        rows: ffi::usearch_rows_view_t {
            data: matrix.as_ptr().cast(),
            stride: layout.stride_bytes,
            columns: layout.columns,
            scalar: Scalar::KIND as i32,
        },
        count: layout.rows,
    })
}

/// A search filter, run on the executor's threads.
type Predicate<'a> = dyn Fn(Key) -> bool + Sync + 'a;

/// # Safety
/// `state` must address a live `&Predicate` that outlives the call the callback was passed to.
unsafe extern "C" fn admission_trampoline(key: Key, state: *mut c_void) -> c_int {
    (*(state as *const &Predicate))(key) as c_int
}

/// An insertion; see [`DenseIndex::add`].
#[must_use = "a builder does nothing until `run`"]
pub struct Add<'a> {
    index: &'a DenseIndex,
    executor: &'a Executor,
    keys: &'a [Key],
    batch: Result<Batch>,
    expansion_factor: usize,
    progress: Option<&'a mut Progress<'a>>,
    stats: Option<&'a mut Stats>,
}

impl<'a> Add<'a> {
    /// The candidate-pool breadth the build walk explores; zero derives it from the geometry.
    pub fn expansion_factor(mut self, expansion_factor: usize) -> Self {
        self.expansion_factor = expansion_factor;
        self
    }

    /// Where the traversal telemetry lands.
    pub fn stats(mut self, stats: &'a mut Stats) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Reports the rows admitted and stops wherever `progress` says to; a cancelled batch reports
    /// [`Status::Cancelled`] and publishes nothing.
    pub fn progress(mut self, progress: &'a mut Progress<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Runs the insertion to completion.
    pub fn run(mut self) -> Result<()> {
        let batch = self.batch?;
        let (progress, progress_state) = lower_progress(&mut self.progress);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_add(&mut ffi::usearch_dense_add_t {
                index: self.index.handle,
                executor: self.executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                count: batch.count,
                keys: ffi::usearch_keys_view_t::of(self.keys),
                vectors: batch.rows,
                expansion_factor: self.expansion_factor,
                progress,
                progress_state,
                stats: self.stats.map_or(ptr::null_mut(), ptr::from_mut),
            })
        };
        check(outcome)
    }
}

/// A k-nearest-neighbors search; see [`DenseIndex::search`].
#[must_use = "a builder does nothing until `run` or `into`"]
pub struct Search<'a> {
    index: &'a DenseIndex,
    executor: &'a Executor,
    batch: Result<Batch>,
    wanted: usize,
    expansion_factor: usize,
    predicate: Option<&'a Predicate<'a>>,
    progress: Option<&'a mut Progress<'a>>,
    stats: Option<&'a mut Stats>,
}

impl<'a> Search<'a> {
    /// The candidate-pool breadth the walk explores; zero derives it from the geometry.
    pub fn expansion_factor(mut self, expansion_factor: usize) -> Self {
        self.expansion_factor = expansion_factor;
        self
    }

    /// Where the traversal telemetry lands.
    pub fn stats(mut self, stats: &'a mut Stats) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Admits only the keys `predicate` approves.
    ///
    /// Non-matching entries still route the traversal, so recall does not collapse at low
    /// selectivity. A VRAM handle refuses a predicate with [`Status::Device`].
    pub fn predicate(mut self, predicate: &'a (dyn Fn(Key) -> bool + Sync + 'a)) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Reports the queries resolved and stops wherever `progress` says to; a cancelled search
    /// reports [`Status::Cancelled`] and leaves every output row indeterminate.
    pub fn progress(mut self, progress: &'a mut Progress<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Runs the search, returning the matches in buffers it allocates.
    pub fn run(self) -> Result<Matches> {
        let mut matches = Matches::new(self.batch?.count, self.wanted)?;
        self.call(matches.lower())?;
        Ok(matches)
    }

    /// Runs the search, landing the matches in caller-owned buffers - the allocation-free path.
    pub fn into(self, matches: &mut MatchesSpan<'_>) -> Result<()> {
        let lowered = matches.lower(self.wanted);
        self.call(lowered)
    }

    fn call(mut self, matches: ffi::usearch_matches_span_t) -> Result<()> {
        let batch = self.batch?;
        let (predicate, predicate_state): (ffi::usearch_predicate_t, *mut c_void) = match &self.predicate {
            Some(predicate) => (
                Some(admission_trampoline),
                predicate as *const &Predicate as *mut c_void,
            ),
            None => (None, ptr::null_mut()),
        };
        let (progress, progress_state) = lower_progress(&mut self.progress);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_search(&mut ffi::usearch_dense_search_t {
                index: self.index.handle,
                executor: self.executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                count: batch.count,
                queries: batch.rows,
                matches,
                expansion_factor: self.expansion_factor,
                predicate,
                predicate_state,
                progress,
                progress_state,
                stats: self.stats.map_or(ptr::null_mut(), ptr::from_mut),
            })
        };
        check(outcome)
    }
}

/// A refinement sweep; see [`DenseIndex::refine`].
#[must_use = "a builder does nothing until `run`"]
pub struct Refine<'a> {
    index: &'a DenseIndex,
    executor: &'a Executor,
    expansion_factor: usize,
    progress: Option<&'a mut Progress<'a>>,
    stats: Option<&'a mut Stats>,
}

impl<'a> Refine<'a> {
    /// The candidate-pool breadth the re-selection walk explores; zero derives it.
    pub fn expansion_factor(mut self, expansion_factor: usize) -> Self {
        self.expansion_factor = expansion_factor;
        self
    }

    /// Where the traversal telemetry lands.
    pub fn stats(mut self, stats: &'a mut Stats) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Reports the nodes swept and stops wherever `progress` says to.
    ///
    /// Cancelling here is a time budget rather than a failure: re-selecting a node's edges is
    /// idempotent and claims no identity, so a stopped sweep leaves the graph correct and merely
    /// less improved, and reports no cancelled status. A deadline check is the handler this asks
    /// for.
    pub fn progress(mut self, progress: &'a mut Progress<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Runs the sweep, returning the number of nodes whose edges changed.
    pub fn run(mut self) -> Result<usize> {
        let (progress, progress_state) = lower_progress(&mut self.progress);
        let mut changed = 0;
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_refine(&mut ffi::usearch_dense_refine_t {
                index: self.index.handle,
                executor: self.executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                changed: &mut changed,
                expansion_factor: self.expansion_factor,
                progress,
                progress_state,
                stats: self.stats.map_or(ptr::null_mut(), ptr::from_mut),
            })
        };
        check(outcome)?;
        Ok(changed)
    }
}

/// An approximate sparse product; see [`DenseIndex::multiply`].
#[must_use = "a builder does nothing until `run`"]
pub struct Multiply<'a> {
    index: &'a DenseIndex,
    executor: &'a Executor,
    batch: Result<Batch>,
    wanted: usize,
    expansion_factor: usize,
    progress: Option<&'a mut Progress<'a>>,
    stats: Option<&'a mut Stats>,
}

impl<'a> Multiply<'a> {
    /// Keeps the first `wanted` pairs each row's walk scored; zero keeps every pair.
    pub fn wanted(mut self, wanted: usize) -> Self {
        self.wanted = wanted;
        self
    }

    /// The candidate-pool breadth, which widens the emitted pattern; zero derives it.
    pub fn expansion_factor(mut self, expansion_factor: usize) -> Self {
        self.expansion_factor = expansion_factor;
        self
    }

    /// Where the traversal telemetry lands.
    pub fn stats(mut self, stats: &'a mut Stats) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Reports the rows resolved and stops wherever `progress` says to.
    pub fn progress(mut self, progress: &'a mut Progress<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Runs the product, landing the pairs in `arena`.
    pub fn run<'arena>(mut self, arena: &'arena mut Arena) -> Result<Products<'arena>> {
        let batch = self.batch?;
        let mut offsets = vec![0; batch.count + 1];
        let (progress, progress_state) = lower_progress(&mut self.progress);
        let (mut keys, mut scores) = (ptr::null_mut(), ptr::null_mut());
        let (mut written, mut required) = (0, 0);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_multiply(&mut ffi::usearch_dense_multiply_t {
                index: self.index.handle,
                executor: self.executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                count: batch.count,
                left: batch.rows,
                offsets: offsets.as_mut_ptr(),
                keys: &mut keys,
                scores: &mut scores,
                arena: arena.handle(),
                written: &mut written,
                required: &mut required,
                wanted: self.wanted,
                expansion_factor: self.expansion_factor,
                progress,
                progress_state,
                stats: self.stats.map_or(ptr::null_mut(), ptr::from_mut),
            })
        };
        check(outcome)?;
        Ok(Products {
            offsets,
            keys: unsafe { arena_slice(keys.cast_const(), written) },
            scores: unsafe { arena_slice(scores.cast_const(), written) },
            required,
        })
    }
}

/// A late-interaction search; see [`DenseIndex::maxsim`].
#[must_use = "a builder does nothing until `run` or `into`"]
pub struct Maxsim<'a> {
    index: &'a DenseIndex,
    executor: &'a Executor,
    batch: Result<Batch>,
    rows_per_query: usize,
    wanted: usize,
    expansion_factor: usize,
    progress: Option<&'a mut Progress<'a>>,
    stats: Option<&'a mut Stats>,
}

impl<'a> Maxsim<'a> {
    /// The candidate-pool breadth; zero derives it from the geometry.
    pub fn expansion_factor(mut self, expansion_factor: usize) -> Self {
        self.expansion_factor = expansion_factor;
        self
    }

    /// Where the traversal telemetry lands.
    pub fn stats(mut self, stats: &'a mut Stats) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Reports the queries resolved and stops wherever `progress` says to.
    pub fn progress(mut self, progress: &'a mut Progress<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Runs the scoring, returning the best keys largest score first, in buffers it allocates.
    pub fn run(self) -> Result<Matches> {
        let mut matches = Matches::new(self.batch?.count / self.rows_per_query, self.wanted)?;
        self.call(matches.lower())?;
        Ok(matches)
    }

    /// Runs the scoring, landing the best keys in caller-owned buffers.
    pub fn into(self, matches: &mut MatchesSpan<'_>) -> Result<()> {
        let lowered = matches.lower(self.wanted);
        self.call(lowered)
    }

    fn call(mut self, matches: ffi::usearch_matches_span_t) -> Result<()> {
        let batch = self.batch?;
        let (progress, progress_state) = lower_progress(&mut self.progress);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_maxsim(&mut ffi::usearch_dense_maxsim_t {
                index: self.index.handle,
                executor: self.executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                count: batch.count / self.rows_per_query,
                queries: batch.rows,
                rows_per_query: self.rows_per_query,
                matches,
                expansion_factor: self.expansion_factor,
                progress,
                progress_state,
                stats: self.stats.map_or(ptr::null_mut(), ptr::from_mut),
            })
        };
        check(outcome)
    }
}

/// A graph-walk attention; see [`DenseIndex::attend`].
#[must_use = "a builder does nothing until `run`"]
pub struct Attend<'a> {
    index: &'a DenseIndex,
    executor: &'a Executor,
    lowered: Result<(Batch, ffi::usearch_matrix_view_t, ffi::usearch_rows_span_t)>,
    carry: Option<&'a mut [SoftmaxState]>,
    expansion_factor: usize,
    score_scale: f32,
    mass_tolerance: f32,
    progress: Option<&'a mut Progress<'a>>,
    stats: Option<&'a mut Stats>,
}

impl<'a> Attend<'a> {
    /// Where the online-softmax state lands, one entry per query row, which folds this branch
    /// into another over the same queries by partial-softmax rescaling.
    pub fn carry(mut self, carry: &'a mut [SoftmaxState]) -> Self {
        self.carry = Some(carry);
        self
    }

    /// The candidate-pool breadth; zero derives it from the geometry.
    pub fn expansion_factor(mut self, expansion_factor: usize) -> Self {
        self.expansion_factor = expansion_factor;
        self
    }

    /// The logit scale, conventionally the inverse square root of the head dimensionality; zero
    /// derives it from the index's frozen geometry.
    pub fn score_scale(mut self, score_scale: f32) -> Self {
        self.score_scale = score_scale;
        self
    }

    /// The residual-mass threshold the walk stops at - the mass-denominated sibling of the radius
    /// a search terminates on; zero derives it.
    pub fn mass_tolerance(mut self, mass_tolerance: f32) -> Self {
        self.mass_tolerance = mass_tolerance;
        self
    }

    /// Where the traversal telemetry lands.
    pub fn stats(mut self, stats: &'a mut Stats) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Reports the queries resolved and stops wherever `progress` says to.
    pub fn progress(mut self, progress: &'a mut Progress<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Runs the attention to completion.
    pub fn run(mut self) -> Result<()> {
        let (batch, values, outputs) = self.lowered?;
        let carry = match self.carry.take() {
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
        let (progress, progress_state) = lower_progress(&mut self.progress);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_attend(&mut ffi::usearch_dense_attend_t {
                index: self.index.handle,
                executor: self.executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                count: batch.count,
                queries: batch.rows,
                values,
                outputs,
                carry,
                expansion_factor: self.expansion_factor,
                score_scale: self.score_scale,
                mass_tolerance: self.mass_tolerance,
                progress,
                progress_state,
                stats: self.stats.map_or(ptr::null_mut(), ptr::from_mut),
            })
        };
        check(outcome)
    }
}

/// A merge of donor indexes; see [`DenseIndex::merge`].
#[must_use = "a builder does nothing until `run`"]
pub struct Merge<'a> {
    index: &'a DenseIndex,
    executor: &'a Executor,
    donors: Vec<DenseIndex>,
    progress: Option<&'a mut Progress<'a>>,
}

impl<'a> Merge<'a> {
    /// Reports the donors absorbed and stops wherever `progress` says to.
    ///
    /// Cancellation stops between donors, so the ones already absorbed stay absorbed and the count
    /// the handler last saw names the boundary. The donors past it are freed here, the caller
    /// having handed them over by value.
    pub fn progress(mut self, progress: &'a mut Progress<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Runs the merge to completion.
    pub fn run(mut self) -> Result<()> {
        // The library consumes and nulls every donor it absorbs, so the slots left live after the
        // call are the ones handed over by value and never taken.
        let mut handles: Vec<ffi::usearch_dense_index_t> = self
            .donors
            .drain(..)
            .map(|mut donor| core::mem::replace(&mut donor.handle, ptr::null_mut()))
            .collect();
        let (progress, progress_state) = lower_progress(&mut self.progress);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_merge(&mut ffi::usearch_dense_merge_t {
                index: self.index.handle,
                executor: self.executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                count: handles.len(),
                donors: handles.as_mut_ptr(),
                progress,
                progress_state,
            })
        };
        for handle in handles.into_iter().filter(|handle| !handle.is_null()) {
            unsafe { ffi::usearch_dense_free(handle) }
        }
        check(outcome)
    }
}

/// A checkpoint write; see [`DenseIndex::save`].
#[must_use = "a builder does nothing until `run`"]
pub struct Save<'a> {
    index: &'a DenseIndex,
    executor: &'a Executor,
    path: &'a Path,
    progress: Option<&'a mut Progress<'a>>,
}

impl<'a> Save<'a> {
    /// Reports the shards written and stops wherever `progress` says to.
    ///
    /// A cancelled save leaves a monolith without its header, or the parts of a manifest layout
    /// without the manifest - leftovers every reader rejects and the caller is left to remove.
    pub fn progress(mut self, progress: &'a mut Progress<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Runs the write to completion.
    pub fn run(mut self) -> Result<()> {
        let path = path_view(self.path)?;
        let (progress, progress_state) = lower_progress(&mut self.progress);
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_save(&mut ffi::usearch_dense_save_t {
                index: self.index.handle,
                executor: self.executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                path,
                progress,
                progress_state,
            })
        };
        check(outcome)
    }
}

/// Where a load reads its checkpoint from.
enum Source<'a> {
    Path(&'a Path),
    Buffer(&'a [u8]),
}

/// A checkpoint read into a new index; see [`DenseIndex::load`].
#[must_use = "a builder does nothing until `run`"]
pub struct Load<'a> {
    executor: &'a Executor,
    source: Source<'a>,
    residency: Residency,
    values_per_key: usize,
    progress: Option<&'a mut Progress<'a>>,
}

impl<'a> Load<'a> {
    /// Where the parts land - RAM or VRAM copy them in, NVMe serves a path in place, read-only.
    /// Part `p` lands whole on shard `p` modulo the shard count, wherever it was written from.
    pub fn residency(mut self, residency: Residency) -> Self {
        self.residency = residency;
        self
    }

    /// Upper bound on entries sharing one key, which no checkpoint records; zero derives to one.
    pub fn values_per_key(mut self, values_per_key: usize) -> Self {
        self.values_per_key = values_per_key;
        self
    }

    /// Reports the shards placed and stops wherever `progress` says to; a cancelled load creates
    /// no index.
    pub fn progress(mut self, progress: &'a mut Progress<'a>) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Runs the read, returning the new index.
    pub fn run(mut self) -> Result<DenseIndex> {
        let (progress, progress_state) = lower_progress(&mut self.progress);
        let executor = self.executor.handle();
        let values_per_key = self.values_per_key;
        let mut index = DenseIndex {
            handle: ptr::null_mut(),
        };
        let mut outcome = usearch_outcome_t::default();
        match (&self.source, &self.residency) {
            (Source::Path(path), Residency::Ram { memory_domains }) => unsafe {
                ffi::usearch_dense_load_ram(&mut ffi::usearch_dense_load_ram_t {
                    index: &mut index.handle,
                    executor,
                    outcome: &mut outcome,
                    completion: 0,
                    path: path_view(path)?,
                    ram: ram_options(memory_domains),
                    values_per_key,
                    progress,
                    progress_state,
                })
            },
            (Source::Path(path), Residency::Vram { devices }) => unsafe {
                ffi::usearch_dense_load_vram(&mut ffi::usearch_dense_load_vram_t {
                    index: &mut index.handle,
                    executor,
                    outcome: &mut outcome,
                    completion: 0,
                    path: path_view(path)?,
                    vram: vram_options(devices),
                    values_per_key,
                    progress,
                    progress_state,
                })
            },
            (Source::Path(path), Residency::Nvme { paths }) => {
                let views = path_views(paths)?;
                unsafe {
                    ffi::usearch_dense_view_nvme(&mut ffi::usearch_dense_view_nvme_t {
                        index: &mut index.handle,
                        executor,
                        outcome: &mut outcome,
                        completion: 0,
                        path: path_view(path)?,
                        nvme: nvme_options(&views),
                    })
                }
            }
            (Source::Buffer(buffer), Residency::Ram { memory_domains }) => unsafe {
                ffi::usearch_dense_load_ram_buffer(&mut ffi::usearch_dense_load_ram_buffer_t {
                    index: &mut index.handle,
                    executor,
                    outcome: &mut outcome,
                    completion: 0,
                    buffer: (*buffer).into(),
                    ram: ram_options(memory_domains),
                    values_per_key,
                    progress,
                    progress_state,
                })
            },
            (Source::Buffer(buffer), Residency::Vram { devices }) => unsafe {
                ffi::usearch_dense_load_vram_buffer(&mut ffi::usearch_dense_load_vram_buffer_t {
                    index: &mut index.handle,
                    executor,
                    outcome: &mut outcome,
                    completion: 0,
                    buffer: (*buffer).into(),
                    vram: vram_options(devices),
                    values_per_key,
                    progress,
                    progress_state,
                })
            },
            (Source::Buffer(_), Residency::Nvme { .. }) => {
                return Err(fail(
                    Status::Unsupported,
                    "an NVMe index serves a path in place, never a buffer",
                ))
            }
        }
        check(outcome).map(|()| index)
    }
}

// endregion: Builders

// region: Cursors & Streams

/// A row scan over the index contents; see [`DenseIndex::cursor`].
///
/// The walk is a sequential row scan, so it stays a batch operation: export, migration, and
/// visualization all move whole slabs.
#[derive(Debug)]
pub struct Cursor<'index, Scalar: crate::types::Scalar> {
    cursor: ffi::usearch_dense_cursor_t,
    index: &'index DenseIndex,
    scalar: PhantomData<Scalar>,
}

impl<'index, Scalar: crate::types::Scalar> Cursor<'index, Scalar> {
    /// Copies the next entries into caller-owned outputs, returning the rows written; zero
    /// means the walk is exhausted.
    ///
    /// Passing `None` for the vectors walks the key column alone, which needs the matrix type
    /// spelled, as in `None::<&mut MatrixSpan<f32>>`; otherwise the matrix needs room for as many
    /// rows as there are keys.
    pub fn next_batch(
        &mut self,
        keys: &mut [Key],
        vectors: Option<&mut MatrixSpan<'_, Scalar, impl Allocator>>,
    ) -> Result<usize> {
        let span = match vectors {
            None => ffi::usearch_rows_span_t {
                data: ptr::null_mut(),
                stride: 0,
                columns: 0,
                capacity: 0,
                scalar: Scalar::KIND as i32,
            },
            Some(matrix) => {
                let layout = MatrixLayout::of(matrix)?;
                ffi::usearch_rows_span_t {
                    data: matrix.as_mut_ptr().cast(),
                    stride: layout.stride_bytes,
                    columns: layout.columns,
                    capacity: layout.rows,
                    scalar: Scalar::KIND as i32,
                }
            }
        };
        let mut written = 0;
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_cursor_next(&mut ffi::usearch_dense_cursor_next_t {
                index: self.index.handle,
                executor: ptr::null_mut(),
                outcome: &mut outcome,
                completion: 0,
                cursor: self.cursor,
                keys: ffi::usearch_keys_span_t::of(keys),
                vectors: span,
                written: &mut written,
            })
        };
        check(outcome)?;
        Ok(written)
    }
}

impl<'index, Scalar: crate::types::Scalar> Drop for Cursor<'index, Scalar> {
    fn drop(&mut self) { unsafe { ffi::usearch_dense_cursor_free(self.cursor) } }
}

/// Matches a stream pulls per crossing into the library, amortising the call over a run of them.
const MATCHES_PER_PULL: usize = 64;

/// A resumable nearest-neighbor stream; see [`DenseIndex::search_stream`].
///
/// Ordering beyond the expansion horizon is best-effort nearest, not globally exact. Matches
/// arrive in batches and are yielded one at a time, so nothing allocates per item.
#[derive(Debug)]
pub struct SearchStream<'index> {
    stream: ffi::usearch_dense_search_stream_t,
    executor: &'index Executor,
    keys: [Key; MATCHES_PER_PULL],
    distances: [Distance; MATCHES_PER_PULL],
    filled: usize,
    position: usize,
    index: &'index DenseIndex,
}

impl<'index> SearchStream<'index> {
    /// Refills the batch straight from the library, leaving it empty once the walk is exhausted.
    fn pull(&mut self) -> Result<()> {
        let mut written = 0;
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            ffi::usearch_dense_search_stream_next(&mut ffi::usearch_dense_search_stream_next_t {
                index: self.index.handle,
                executor: self.executor.handle(),
                outcome: &mut outcome,
                completion: 0,
                stream: self.stream,
                matches: matches_span(
                    &mut self.keys,
                    &mut self.distances,
                    core::slice::from_mut(&mut written),
                    MATCHES_PER_PULL,
                    MATCHES_PER_PULL,
                ),
            })
        };
        check(outcome)?;
        (self.filled, self.position) = (written, 0);
        Ok(())
    }
}

impl<'index> Iterator for SearchStream<'index> {
    type Item = Result<Match>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.filled {
            if let Err(failure) = self.pull() {
                return Some(Err(failure));
            }
            if self.filled == 0 {
                return None;
            }
        }
        let key = *self.keys.get(self.position)?;
        let distance = *self.distances.get(self.position)?;
        self.position += 1;
        Some(Ok(Match { key, distance }))
    }
}

impl<'index> core::iter::FusedIterator for SearchStream<'index> {}

impl<'index> Drop for SearchStream<'index> {
    fn drop(&mut self) { unsafe { ffi::usearch_dense_search_stream_free(self.stream) } }
}

// endregion: Cursors & Streams

// region: Helpers

/// What a matrix supplies to a descriptor, once the layout it must satisfy is checked.
struct MatrixLayout {
    /// Rows the matrix spans, which a span reports as its capacity.
    rows: usize,
    /// Logical scalars per row, which the call checks against the frozen geometry.
    columns: usize,
    /// Bytes between the starts of consecutive rows.
    stride_bytes: usize,
}

impl MatrixLayout {
    /// Reads a matrix's shape, refusing the layouts the C interface cannot traverse.
    ///
    /// Only this side can answer whether rows are contiguous and ascend in memory, which NumKong
    /// knows and a bare pointer does not.
    fn of<Element: StorageElement>(matrix: &impl TensorRef<Element, 2>) -> Result<Self> {
        let &[rows, columns] = matrix.shape() else {
            return Err(fail(Status::ShapeMismatch, "a matrix holds two axes"));
        };
        if !matrix.has_contiguous_rows() {
            return Err(fail(
                Status::UnsupportedLayout,
                "expected a matrix whose rows are contiguous; only the pitch may stride",
            ));
        }
        let pitch = matrix.stride_bytes(0);
        if pitch < 0 {
            return Err(fail(Status::UnsupportedLayout, "matrix rows must ascend in memory"));
        }
        Ok(MatrixLayout {
            rows,
            columns,
            stride_bytes: pitch as usize,
        })
    }
}

// endregion: Helpers

// region: Tests

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use numkong::{u1x8, MatrixView};
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::types::{rows, rows_mut, Continuation, Error};

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

    fn empty_index(dimensions: usize, metric: Metric, quantization: ScalarKind) -> Result<DenseIndex> {
        let index = DenseIndex::new(&Options {
            dimensions,
            metric,
            quantization,
            ..Options::default()
        })?;
        index.reserve(&Executor::default(), 1024)?;
        Ok(index)
    }

    fn populated_index(
        count: usize,
        dimensions: usize,
        metric: Metric,
        quantization: ScalarKind,
    ) -> Result<(DenseIndex, Vec<Key>, Vec<f32>)> {
        let index = empty_index(dimensions, metric, quantization)?;
        index.reserve(&Executor::default(), count)?;
        let keys = sequential_keys(count);
        let slab = seeded_rows(count, dimensions, SEED);
        index.add(&executor(), &keys, &rows(&slab, dimensions)?).run()?;
        assert_eq!(index.describe(&Executor::default())?.size, count);
        Ok((index, keys, slab))
    }

    /// How many matches one query reports - the shape most assertions here want.
    fn matches_of(index: &DenseIndex, query: &[f32], dimensions: usize) -> Result<usize> {
        let matches = index.search(&executor(), &rows(query, dimensions)?, 1).run()?;
        let found = matches.row(0).map_or(0, |row| row.keys.len());
        Ok(found)
    }

    /// A checkpoint path scoped by test name and process, never the working directory.
    fn scratch_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("usearch-rust-{name}-{}.safegraphs", std::process::id()))
    }

    fn remove_quietly(path: &std::path::Path) { let _ = std::fs::remove_file(path); }

    fn close(left: f64, right: f64, tolerance: f64) -> bool { (left - right).abs() <= tolerance }

    fn requires_send_sync<Subject: Send + Sync>() {}

    /// Packs one byte per element, LSB first, as NumKong's bit vectors are laid out.
    fn bit_rows(bytes: &[u8]) -> Vec<u1x8> { bytes.iter().map(|&byte| u1x8(byte)).collect() }

    /// Whether a watched call is let run to completion or stopped at its first report.
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
    /// inside a handler is caught and read as a cancellation rather than as a failed test.
    struct Watch {
        /// The completed and total counts of the last report.
        last: (usize, usize),
        /// Whether every report's completed count was at least its predecessor's.
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

        fn saw(&mut self, completed: usize, total: usize) -> Continuation {
            self.ordered &= completed >= self.last.0;
            self.last = (completed, total);
            match self.patience {
                Patience::Complete => Continuation::Continue,
                Patience::StopAtFirstReport => Continuation::Cancel,
            }
        }
    }

    // endregion: Helpers

    // region: Metadata Marshalling

    #[test]
    fn metadata_lifts_the_four_fields_a_checkpoint_froze() -> Result<()> {
        let raw = ffi::usearch_dense_metadata_t {
            metric: Metric::Euclidean as i32,
            quantization: ScalarKind::I8 as i32,
            dimensions: 768,
            connectivity: 32,
        };
        let lifted = Metadata::from_raw(&raw)?;
        assert_eq!(lifted.dimensions, 768);
        assert_eq!(lifted.metric, Metric::Euclidean);
        assert_eq!(lifted.quantization, ScalarKind::I8);
        assert_eq!(lifted.connectivity, 32);
        Ok(())
    }

    #[test]
    fn metadata_rejects_unknown_enum_tags() {
        let mut raw = ffi::usearch_dense_metadata_t {
            metric: Metric::Angular as i32,
            quantization: ScalarKind::F32 as i32,
            dimensions: 8,
            connectivity: 0,
        };
        raw.metric = 99;
        assert!(matches!(
            Metadata::from_raw(&raw),
            Err(Error {
                status: Status::UnknownKind,
                ..
            })
        ));

        raw.metric = Metric::Angular as i32;
        raw.quantization = 77;
        assert!(matches!(
            Metadata::from_raw(&raw),
            Err(Error {
                status: Status::UnknownKind,
                ..
            })
        ));
    }

    // endregion: Metadata Marshalling

    // region: Construction & Geometry

    /// The one place the Rust enum values are checked against the C ones: a renumbered
    /// constant reaches the library as a different kind, and is caught here and nowhere else.
    #[test]
    fn constructs_every_metric_and_quantization() -> Result<()> {
        for quantization in [
            ScalarKind::F64,
            ScalarKind::F32,
            ScalarKind::BF16,
            ScalarKind::F16,
            ScalarKind::E5M2,
            ScalarKind::E4M3,
            ScalarKind::E3M2,
            ScalarKind::E2M3,
            ScalarKind::I8,
            ScalarKind::I4x2,
            ScalarKind::U8,
            ScalarKind::U4x2,
            ScalarKind::U1x8,
        ] {
            let described = empty_index(64, Metric::Angular, quantization)?.describe(&Executor::default())?;
            assert_eq!(described.metadata.dimensions, 64, "{quantization:?}");
            assert_eq!(described.metadata.quantization, quantization);
            assert_eq!(described.size, 0, "{quantization:?}");
            assert!(described.metadata.connectivity > 0, "{quantization:?}");
        }

        // The bit-level metrics take packed rows, so each metric meets a kind it accepts.
        for (metric, quantization) in [
            (Metric::Inner, ScalarKind::F32),
            (Metric::Angular, ScalarKind::F32),
            (Metric::Euclidean, ScalarKind::F32),
            (Metric::Hamming, ScalarKind::U1x8),
            (Metric::Jaccard, ScalarKind::U1x8),
        ] {
            let described = empty_index(64, metric, quantization)?.describe(&Executor::default())?;
            assert_eq!(described.metadata.metric, metric);
        }
        Ok(())
    }

    #[test]
    fn does_not_preallocate_members() -> Result<()> {
        let index = DenseIndex::new(&Options {
            dimensions: 16,
            ..Options::default()
        })?;
        let described = index.describe(&Executor::default())?;
        assert_eq!((described.size, described.capacity), (0, 0));

        index.reserve(&Executor::default(), 1024)?;
        let described = index.describe(&Executor::default())?;
        assert!(described.capacity >= 1024);
        assert_eq!(described.size, 0);
        Ok(())
    }

    /// A zeroed width freezes nothing, and the first batch has no width the index would trust.
    #[test]
    fn an_unfrozen_geometry_admits_no_rows() -> Result<()> {
        let index = DenseIndex::new(&Options {
            dimensions: 0,
            quantization: ScalarKind::F32,
            ..Options::default()
        })?;
        assert_eq!(index.describe(&Executor::default())?.metadata.dimensions, 0);

        let slab = seeded_rows(1, 12, SEED);
        assert!(matches!(
            index.add(&executor(), &sequential_keys(1), &rows(&slab, 12)?).run(),
            Err(Error {
                status: Status::Uninitialized,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn an_nvme_index_is_never_initialized() {
        let options = Options {
            dimensions: 8,
            residency: Residency::Nvme { paths: Vec::new() },
            ..Options::default()
        };
        assert!(matches!(
            DenseIndex::new(&options),
            Err(Error {
                status: Status::Unsupported,
                ..
            })
        ));
    }

    /// The handle is one pointer, so a move must not disturb it and two threads may share it.
    #[test]
    fn survives_box_and_arc_moves() -> Result<()> {
        requires_send_sync::<DenseIndex>();
        let (index, _, _) = populated_index(4, 8, Metric::Angular, ScalarKind::F32)?;

        let boxed = Box::new(index);
        assert_eq!(boxed.describe(&Executor::default())?.size, 4);
        let shared = std::sync::Arc::new(*boxed);
        assert_eq!(shared.describe(&Executor::default())?.size, 4);
        Ok(())
    }

    // endregion: Construction & Geometry

    // region: Ingest & Retrieval

    #[test]
    fn adds_and_gets_vectors() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let keys = sequential_keys(2);
        let slab: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        index.add(&executor(), &keys, &rows(&slab, DIMENSIONS)?).run()?;
        assert_eq!(index.describe(&Executor::default())?.size, 2);

        let (executor, mut arena) = (executor(), Arena::default());
        assert_eq!(index.get::<f32>(&executor, keys[1], &mut arena)?, &[0.0, 1.0, 0.0, 0.0]);
        assert!(
            index.get::<f32>(&executor, Key::from(7u64), &mut arena)?.is_empty(),
            "an absent key yields nothing"
        );
        Ok(())
    }

    /// An empty batch is a shape rather than a mistake, so it publishes nothing and reports no
    /// failure - what a caller filtering its own input relies on.
    #[test]
    fn accepts_an_empty_batch() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let empty: [f32; 0] = [];
        index.add(&executor(), &[], &rows(&empty, DIMENSIONS)?).run()?;
        assert_eq!(index.describe(&Executor::default())?.size, 0);
        Ok(())
    }

    #[test]
    fn rejects_mismatched_key_and_row_counts() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        assert!(matches!(
            index
                .add(&executor(), &sequential_keys(3), &rows(&slab, DIMENSIONS)?)
                .run(),
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
        let (index, _, _) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let narrow = seeded_rows(1, DIMENSIONS - 1, SEED);

        assert!(matches!(
            index
                .add(&executor(), &sequential_keys(1), &rows(&narrow, DIMENSIONS - 1)?)
                .run(),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        assert!(matches!(
            index.search(&executor(), &rows(&narrow, DIMENSIONS - 1)?, 1).run(),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        assert!(
            matches!(
                index.search_stream(&executor(), &rows(&narrow, DIMENSIONS - 1)?),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "the query width is checked too"
        );

        let mut cursor = index.cursor::<f32>()?;
        let mut keys = [Key::NIL; 1];
        let mut buffer = [0.0f32; DIMENSIONS - 1];
        assert!(matches!(
            cursor.next_batch(&mut keys, Some(&mut rows_mut(&mut buffer, DIMENSIONS - 1)?)),
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
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let slab = seeded_rows(2, DIMENSIONS * 2, SEED);
        let element = core::mem::size_of::<f32>() as isize;
        // A column stride of two elements interleaves the rows, which the boundary forbids:
        // the ABI carries one row stride and no column stride.
        let interleaved: MatrixView<'_, f32> =
            unsafe { MatrixView::from_raw_parts(slab.as_ptr(), &[2, DIMENSIONS], &[element * 8, element * 2]) }
                .unwrap();
        assert!(matches!(
            index.add(&executor(), &sequential_keys(2), &interleaved).run(),
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
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
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
            index.add(&executor(), &sequential_keys(2), &descending).run(),
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
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let mut padded = [0.0f32; 2 * PITCH];
        let rows_source = seeded_rows(2, DIMENSIONS, SEED);
        padded[..DIMENSIONS].copy_from_slice(&rows_source[..DIMENSIONS]);
        padded[PITCH..PITCH + DIMENSIONS].copy_from_slice(&rows_source[DIMENSIONS..]);

        let element = core::mem::size_of::<f32>() as isize;
        let strided: MatrixView<'_, f32> = unsafe {
            MatrixView::from_raw_parts(padded.as_ptr(), &[2, DIMENSIONS], &[element * PITCH as isize, element])
        }
        .unwrap();
        let keys = sequential_keys(2);
        index.add(&executor(), &keys, &strided).run()?;

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
        assert_eq!(index.cursor::<f32>()?.next_batch(&mut walked, Some(&mut span))?, 2);
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
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::BF16)?;
        let keys = sequential_keys(1);
        let slab = seeded_rows(1, DIMENSIONS, SEED);
        index.add(&executor(), &keys, &rows(&slab, DIMENSIONS)?).run()?;

        let mut arena = Arena::default();
        let wide = index.get::<f64>(&executor(), keys[0], &mut arena)?;
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
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let shared = Key::from(3u64);
        let slab = seeded_rows(3, DIMENSIONS, SEED);
        index.add(&executor(), &[shared; 3], &rows(&slab, DIMENSIONS)?).run()?;

        assert_eq!(index.count(&Executor::default(), shared)?, 3);
        assert!(index.contains(&Executor::default(), shared)?);
        assert!(!index.contains(&Executor::default(), Key::from(8u64))?);

        let mut arena = Arena::default();
        let restored = index.get::<f32>(&executor(), shared, &mut arena)?;
        assert_eq!(restored.len(), DIMENSIONS * 3);
        for (left, right) in restored.iter().zip(&slab) {
            assert!(close(*left as f64, *right as f64, 1e-6));
        }
        Ok(())
    }

    /// An output borrows its arena, so keeping one call's rows past the next takes a second arena.
    #[test]
    fn a_second_arena_keeps_the_first_output_alive() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let (index, keys, slab) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let executor = executor();
        let (mut first, mut second) = (Arena::default(), Arena::default());
        let held = index.get::<f32>(&executor, keys[0], &mut first)?;
        let next = index.get::<f32>(&executor, keys[1], &mut second)?;
        assert_eq!(held, &slab[..DIMENSIONS]);
        assert_eq!(next, &slab[DIMENSIONS..DIMENSIONS * 2]);
        Ok(())
    }

    #[test]
    fn uses_128_bit_keys() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let wide = Key::from(u128::MAX);
        let narrow = Key::from(9u64);
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        index
            .add(&executor(), &[wide, narrow], &rows(&slab, DIMENSIONS)?)
            .run()?;

        assert_eq!(index.count(&Executor::default(), wide)?, 1);
        assert_eq!(index.count(&Executor::default(), narrow)?, 1);
        assert_eq!(wide.as_u64(), None, "a non-zero high half has no integer identity");
        assert_eq!(narrow.as_u64(), Some(9));
        Ok(())
    }

    // endregion: Ingest & Retrieval

    // region: Refinement & Linear Algebra

    /// The return is a change count rather than a status, so a caller loops until the graph
    /// settles.
    #[test]
    fn refinement_reports_its_sweep() -> Result<()> {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, Metric::Angular, ScalarKind::F32)?;
        let mut stats = Stats::default();
        let changed = index.refine(&executor()).expansion_factor(64).stats(&mut stats).run()?;
        assert!(
            changed <= COUNT,
            "a sweep cannot change more nodes than the index holds"
        );
        assert!(stats.visited_members > 0, "the sweep reports the nodes it swept");
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
        let index = empty_index(DIMENSIONS, Metric::Inner, ScalarKind::F32)?;
        let slab = seeded_rows(ROWS, DIMENSIONS, SEED);
        index
            .add(&executor(), &sequential_keys(ROWS), &rows(&slab, DIMENSIONS)?)
            .run()?;

        let executor = executor();
        let left = rows(&slab, DIMENSIONS)?;
        let mut arena = Arena::default();
        let every = index.multiply(&executor, &left).run(&mut arena)?;
        assert!(
            every.rows().all(|row| row.keys.len() == ROWS),
            "every row scores every column"
        );

        let cut = index.multiply(&executor, &left).wanted(WANTED).run(&mut arena)?;
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
        let index = DenseIndex::new(&Options {
            dimensions: DIMENSIONS,
            metric: Metric::Inner,
            quantization: ScalarKind::F32,
            values_per_key: TOKENS,
            ..Options::default()
        })?;
        index.reserve(&Executor::default(), DOCUMENTS * TOKENS)?;

        let slab = seeded_rows(DOCUMENTS * TOKENS, DIMENSIONS, SEED);
        let mut keys = Vec::with_capacity(DOCUMENTS * TOKENS);
        for document in 0..DOCUMENTS {
            keys.extend(core::iter::repeat_n(Key::from(document as u64), TOKENS));
        }
        index.add(&executor(), &keys, &rows(&slab, DIMENSIONS)?).run()?;

        let tokens = rows(&slab[..QUERIES * TOKENS * DIMENSIONS], DIMENSIONS)?;
        let matches = index.maxsim(&executor(), &tokens, TOKENS, DOCUMENTS).run()?;
        assert_eq!(matches.queries(), QUERIES);
        assert!(
            matches.rows().all(|row| row.keys.len() == DOCUMENTS),
            "every document is scored, once per query"
        );
        Ok(())
    }

    /// One context row and one carry entry per query, both written through their own stride.
    #[test]
    fn attention_writes_one_context_row_per_query() -> Result<()> {
        const DIMENSIONS: usize = 4;
        const VALUE_WIDTH: usize = 2;
        const ENTRIES: usize = 4;
        const QUERIES: usize = 2;
        let index = empty_index(DIMENSIONS, Metric::Inner, ScalarKind::F32)?;
        let slab = seeded_rows(ENTRIES, DIMENSIONS, SEED);
        // An entry's key is its row offset into the value table, so positions insert in order.
        index
            .add(&executor(), &sequential_keys(ENTRIES), &rows(&slab, DIMENSIONS)?)
            .run()?;

        let values = seeded_rows(ENTRIES, VALUE_WIDTH, SEED + 1);
        let mut attended = vec![0.0f32; QUERIES * VALUE_WIDTH];
        let mut carry = vec![SoftmaxState::default(); QUERIES];
        let mut context = rows_mut(&mut attended, VALUE_WIDTH)?;
        index
            .attend(
                &executor(),
                &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?,
                &rows(&values, VALUE_WIDTH)?,
                &mut context,
            )
            .carry(&mut carry)
            .run()?;
        assert!(
            carry.iter().all(|state| state.partition > 0.0),
            "every query reports the softmax denominator it accumulated"
        );
        Ok(())
    }

    /// A product and a softmax both need logits, so a graph frozen under a distance refuses.
    #[test]
    fn linear_algebra_refuses_a_distance_kind() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let (index, _, slab) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        assert!(matches!(
            index
                .multiply(&executor(), &rows(&slab[..DIMENSIONS], DIMENSIONS)?)
                .run(&mut Arena::default())
                .map(|products| products.queries()),
            Err(Error {
                status: Status::ConfigMismatch,
                ..
            })
        ));
        Ok(())
    }

    // endregion: Refinement & Linear Algebra

    // region: Search

    #[test]
    fn searches_an_empty_index() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
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
        let (index, _, slab) = populated_index(256, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        // The index names its rows zero upwards, so the untouched marker has to sit above them.
        let untouched = Key::from(u128::MAX);
        let mut found = vec![untouched; QUERIES * WANTED];
        let mut scores = vec![0.0 as Distance; QUERIES * WANTED];
        let mut counts = vec![0usize; QUERIES];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index
            .search(&executor(), &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?, WANTED)
            .expansion_factor(64)
            .into(&mut matches)?;
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
        let (index, _, slab) = populated_index(16, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let mut found = vec![Key::NIL; QUERIES * 4];
        let mut scores = vec![0.0 as Distance; QUERIES * 4];
        let mut counts = vec![0usize; QUERIES];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index
            .search(&executor(), &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?, 2)
            .into(&mut matches)?;
        assert!(matches.rows().all(|row| row.keys.len() == 2));
        Ok(())
    }

    /// The row width follows from the buffer lengths, so lengths that cannot be read as one
    /// shape are refused before any verb sees them.
    #[test]
    fn rejects_outputs_that_are_not_one_shape() {
        let mut found = vec![Key::NIL; 8];
        let mut scores = vec![0.0 as Distance; 8];
        let mut counts = vec![0usize; 3];
        let mut queryless: [usize; 0] = [];
        assert!(
            matches!(
                MatchesSpan::new(&mut found, &mut scores, &mut queryless),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "no queries to write for"
        );
        assert!(
            matches!(
                MatchesSpan::new(&mut found, &mut scores[..6], &mut counts),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "the keys and distances disagree"
        );
        assert!(
            matches!(
                MatchesSpan::new(&mut found, &mut scores, &mut counts),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "eight cells do not divide into three rows"
        );
    }

    #[test]
    fn rejects_undersized_outputs() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const QUERIES: usize = 4;
        let (index, _, slab) = populated_index(16, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let executor = executor();
        let queries = rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?;

        let mut found = vec![Key::NIL; (QUERIES - 1) * 2];
        let mut scores = vec![0.0 as Distance; (QUERIES - 1) * 2];
        let mut counts = vec![0usize; QUERIES - 1];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        assert!(
            matches!(
                index.search(&executor, &queries, 2).into(&mut matches),
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
                index.search(&executor, &queries, 3).into(&mut matches),
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
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let keys = sequential_keys(64);
        let slab = seeded_rows(64, DIMENSIONS, SEED);

        let mut inserting = Stats::default();
        index
            .add(&executor(), &keys, &rows(&slab, DIMENSIONS)?)
            .stats(&mut inserting)
            .run()?;
        assert!(inserting.computed_distances > 0);

        let executor = executor();
        let query = rows(&slab[..DIMENSIONS], DIMENSIONS)?;
        let mut searching = Stats::default();
        index.search(&executor, &query, 4).stats(&mut searching).run()?;
        let first = searching;
        assert!(first.computed_distances > 0);

        index.search(&executor, &query, 4).stats(&mut searching).run()?;
        assert_eq!(
            searching, first,
            "stats are overwritten by their call, never accumulated"
        );

        // Accumulating is the caller's job, and the operator exists for exactly that.
        let mut total = first;
        total += first;
        assert_eq!(total.computed_distances, first.computed_distances * 2);
        Ok(())
    }

    #[test]
    fn filters_with_a_stateless_predicate() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const WANTED: usize = 8;
        let (index, _, slab) = populated_index(64, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let is_even = |key: Key| key.as_u64().is_some_and(|value| value % 2 == 0);
        let matches = index
            .search(&executor(), &rows(&slab[..DIMENSIONS], DIMENSIONS)?, WANTED)
            .expansion_factor(64)
            .predicate(&is_even)
            .run()?;

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
        let (index, keys, slab) = populated_index(64, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let allowed: HashSet<Key> = keys.iter().copied().take(5).collect();
        let matches = index
            .search(&executor(), &rows(&slab[..DIMENSIONS], DIMENSIONS)?, WANTED)
            .expansion_factor(64)
            .predicate(&|key: Key| allowed.contains(&key))
            .run()?;

        for key in matches.row(0).unwrap().keys {
            assert!(allowed.contains(key), "{key} is outside the captured set");
        }
        Ok(())
    }

    #[test]
    fn filters_everything_out() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, _, slab) = populated_index(64, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let matches = index
            .search(&executor(), &rows(&slab[..DIMENSIONS], DIMENSIONS)?, 4)
            .expansion_factor(64)
            .predicate(&|_: Key| false)
            .run()?;
        assert!(
            matches.row(0).unwrap().keys.is_empty(),
            "a fully-rejecting predicate is not an error"
        );
        Ok(())
    }

    // endregion: Search

    // region: Metrics

    /// A sub-byte kind counts logical dimensions, so eight of them are one stored byte - the
    /// one place where the row width and the row extent part ways.
    #[test]
    fn packed_bit_rows_cross_as_one_byte_per_eight_dimensions() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, Metric::Hamming, ScalarKind::U1x8)?;
        let stored = bit_rows(&[0b0000_1111, 0b1111_0000]);
        let keys = [Key::from(42u64), Key::from(43u64)];
        index.add(&executor(), &keys, &rows(&stored, DIMENSIONS)?).run()?;
        assert_eq!(
            index.describe(&Executor::default())?.size,
            2,
            "two bytes are two rows, not sixteen"
        );

        let query = bit_rows(&[0b0111_1000]);
        let matches = index.search(&executor(), &rows(&query, DIMENSIONS)?, 2).run()?;
        let row = matches.row(0).unwrap();
        assert_eq!(
            row.keys.iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
        Ok(())
    }

    // endregion: Metrics

    // region: Serialization

    /// The loaded index adopts its geometry from the checkpoint's own headers.
    #[test]
    fn saves_and_loads_a_file() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let path = scratch_path("saves_and_loads_a_file");
        let (index, _, _) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        index.save(&executor(), &path).run()?;

        let restored = DenseIndex::load(&executor(), &path)
            .run()?
            .describe(&Executor::default())?;
        assert_eq!(restored.metadata.dimensions, DIMENSIONS);
        assert_eq!(restored.size, 32);

        // A view serves the same checkpoint in place and refuses to change it.
        let viewed = DenseIndex::view(&path)?;
        assert_eq!(viewed.describe(&Executor::default())?.size, 32);
        assert!(matches!(
            viewed.reserve(&Executor::default(), 64),
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
        let (index, _, _) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let buffer = index.save_buffer(&executor(), &mut Arena::default())?.to_vec();
        assert!(!buffer.is_empty());

        let restored = DenseIndex::load_buffer(&executor(), &buffer).run()?;
        assert_eq!(restored.describe(&Executor::default())?.size, 32);
        assert!(
            DenseIndex::load_buffer(&executor(), &buffer[..buffer.len() - 1])
                .run()
                .is_err(),
            "a short buffer must not be read"
        );
        Ok(())
    }

    #[test]
    fn reads_metadata_without_loading() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let path = scratch_path("reads_metadata_without_loading");
        let (index, _, _) = populated_index(8, DIMENSIONS, Metric::Euclidean, ScalarKind::I8)?;
        index.save(&executor(), &path).run()?;

        let executor = executor();
        let described = DenseIndex::metadata(&executor, &path)?;
        assert_eq!(described.dimensions, DIMENSIONS);
        assert_eq!(described.metric, Metric::Euclidean);
        assert_eq!(described.quantization, ScalarKind::I8);

        let mut arena = Arena::default();
        let buffer = index.save_buffer(&executor, &mut arena)?;
        assert_eq!(DenseIndex::metadata_buffer(&executor, buffer)?.dimensions, DIMENSIONS);
        remove_quietly(&path);
        Ok(())
    }

    #[test]
    #[cfg(unix)]
    fn rejects_a_non_utf8_path() -> Result<()> {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let index = empty_index(8, Metric::Angular, ScalarKind::F32)?;
        let path = std::path::Path::new(OsStr::from_bytes(&[0xff, 0xfe]));
        assert!(
            matches!(
                index.save(&executor(), path).run(),
                Err(Error { status: Status::Io, .. })
            ),
            "the C interface exchanges UTF-8 bytes"
        );
        Ok(())
    }

    #[test]
    fn clears_and_reuses() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, keys, slab) = populated_index(16, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let reserved = index.describe(&Executor::default())?.capacity;

        index.clear(&Executor::default())?;
        let cleared = index.describe(&Executor::default())?;
        assert_eq!(cleared.size, 0);
        assert_eq!(cleared.capacity, reserved, "clearing keeps the arena");
        assert_eq!(cleared.metadata.dimensions, DIMENSIONS, "and keeps the frozen geometry");

        index.add(&executor(), &keys, &rows(&slab, DIMENSIONS)?).run()?;
        assert_eq!(index.describe(&Executor::default())?.size, 16);
        Ok(())
    }

    // endregion: Serialization

    // region: Merge, Cursor & Stream

    #[test]
    fn merges_a_donor_index() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, keys, _) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let donor = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let donor_keys: Vec<Key> = (100..132u64).map(Key::from).collect();
        let donor_slab = seeded_rows(32, DIMENSIONS, SEED + 1);
        donor
            .add(&executor(), &donor_keys, &rows(&donor_slab, DIMENSIONS)?)
            .run()?;

        // The donors are consumed by value, so a merged handle cannot be used again.
        index.merge(&executor(), vec![donor]).run()?;
        assert_eq!(index.describe(&Executor::default())?.size, 64);
        assert!(index.contains(&Executor::default(), keys[0])? && index.contains(&Executor::default(), donor_keys[0])?);
        Ok(())
    }

    #[test]
    fn walks_entries_in_batches() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 20;
        const BUFFERED: usize = 8;
        let (index, keys, _) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let mut cursor = index.cursor::<f32>()?;
        let mut seen = HashSet::new();
        let mut batches = Vec::new();
        let mut batch_keys = [Key::NIL; BUFFERED];
        let mut batch_vectors = vec![0.0f32; BUFFERED * DIMENSIONS];
        loop {
            let mut span = rows_mut(&mut batch_vectors, DIMENSIONS)?;
            let written = cursor.next_batch(&mut batch_keys, Some(&mut span))?;
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
        let (index, keys, _) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let mut cursor = index.cursor::<f32>()?;
        let mut batch_keys = [Key::NIL; 8];
        let written = cursor.next_batch(&mut batch_keys, None::<&mut MatrixSpan<'_, f32>>)?;
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
        let (index, _, slab) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let executor = executor();
        let query = rows(&slab[..DIMENSIONS], DIMENSIONS)?;
        let mut stream = index.search_stream(&executor, &query)?;
        let first: Vec<Match> = stream
            .by_ref()
            .take(5)
            .map(|found| found.expect("the stream must not fault"))
            .collect();
        assert_eq!(first.len(), 5, "a pull hands the caller as many as it asked for");

        let drained = stream.count();
        assert!(drained > MATCHES_PER_PULL, "the stream crosses its refill boundary");

        let mut exhausted = index.search_stream(&executor, &query)?;
        while exhausted.next().is_some() {}
        assert!(exhausted.next().is_none(), "a drained stream stays drained");
        Ok(())
    }

    // endregion: Merge, Cursor & Stream

    // region: Progress & Cancellation

    #[test]
    fn a_reported_batch_counts_up_to_the_rows_it_was_given() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        index.reserve(&Executor::default(), COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let mut watch = Watch::new(Patience::Complete);
        index
            .add(&executor(), &keys, &rows(&slab, DIMENSIONS)?)
            .progress(&mut |completed, total| watch.saw(completed, total))
            .run()?;

        assert_eq!(index.describe(&Executor::default())?.size, COUNT);
        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (COUNT, COUNT), "the last report names the whole batch");
        Ok(())
    }

    /// A batch is all-or-nothing even when stopped, so a cancelled one leaves no row behind.
    #[test]
    fn a_stopped_batch_publishes_nothing() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        index.reserve(&Executor::default(), COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let mut watch = Watch::new(Patience::StopAtFirstReport);
        let added = index
            .add(&executor(), &keys, &rows(&slab, DIMENSIONS)?)
            .progress(&mut |completed, total| watch.saw(completed, total))
            .run();

        match added {
            Ok(()) => assert_eq!(
                index.describe(&Executor::default())?.size,
                COUNT,
                "an uncancelled batch published it all"
            ),
            Err(failure) => {
                assert_eq!(failure.status, Status::Cancelled);
                assert_eq!(
                    index.describe(&Executor::default())?.size,
                    0,
                    "a stopped batch publishes no row"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn a_reported_search_counts_up_to_the_queries_it_was_given() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const QUERIES: usize = 4;
        const WANTED: usize = 3;
        let (index, _, slab) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let mut watch = Watch::new(Patience::Complete);
        index
            .search(&executor(), &rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS)?, WANTED)
            .progress(&mut |completed, total| watch.saw(completed, total))
            .run()?;

        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (QUERIES, QUERIES));
        Ok(())
    }

    /// Stopping a sweep is the time budget rather than a failure, so nothing is reported.
    #[test]
    fn a_stopped_refinement_is_not_a_failure() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 32;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let mut watch = Watch::new(Patience::StopAtFirstReport);
        index
            .refine(&executor())
            .progress(&mut |completed, total| watch.saw(completed, total))
            .run()?;

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
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let mut writing = Watch::new(Patience::Complete);
        index
            .save(&executor(), &path)
            .progress(&mut |completed, total| writing.saw(completed, total))
            .run()?;
        assert!(writing.ordered, "a report never counts down");
        assert_eq!(writing.last.0, writing.last.1, "the last report names a finished write");
        assert!(writing.last.1 != 0);

        let mut placing = Watch::new(Patience::Complete);
        let restored = DenseIndex::load(&executor(), &path)
            .progress(&mut |completed, total| placing.saw(completed, total))
            .run()?;
        assert!(placing.ordered, "a report never counts down");
        assert_eq!(placing.last.0, placing.last.1, "the last report names a finished read");
        assert_eq!(restored.describe(&Executor::default())?.size, COUNT);

        remove_quietly(&path);
        Ok(())
    }

    #[test]
    fn a_reported_merge_counts_its_donors() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 16;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let donors = vec![
            populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?.0,
            populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?.0,
        ];
        let mut watch = Watch::new(Patience::Complete);
        index
            .merge(&executor(), donors)
            .progress(&mut |completed, total| watch.saw(completed, total))
            .run()?;

        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (2, 2), "one report per donor absorbed");
        assert_eq!(index.describe(&Executor::default())?.size, COUNT * 3);
        Ok(())
    }

    /// A stopped merge keeps what it absorbed, and the donors past the boundary are freed here -
    /// they were handed over by value, so a leak or a double free would surface in this test.
    #[test]
    fn a_stopped_merge_keeps_the_donors_it_absorbed() -> Result<()> {
        const DIMENSIONS: usize = 16;
        const COUNT: usize = 16;
        const DONORS: usize = 2;
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let donors = vec![
            populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?.0,
            populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?.0,
        ];
        let mut watch = Watch::new(Patience::StopAtFirstReport);
        let merged = index
            .merge(&executor(), donors)
            .progress(&mut |completed, total| watch.saw(completed, total))
            .run();

        match merged {
            Ok(()) => assert_eq!(watch.last.0, DONORS, "an uncancelled merge absorbed them all"),
            Err(failure) => assert_eq!(failure.status, Status::Cancelled),
        }
        assert_eq!(index.describe(&Executor::default())?.size, COUNT * (1 + watch.last.0));
        Ok(())
    }

    // endregion: Progress & Cancellation

    // region: Concurrency

    const THREADS: usize = 4;

    /// The batch is the unit of parallelism: the executor fans one call out over its threads,
    /// so the caller never spawns any of its own.
    #[test]
    fn builds_and_searches_on_a_shared_executor() -> Result<()> {
        const DIMENSIONS: usize = 128;
        const COUNT: usize = 1000;
        const WANTED: usize = 10;

        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        index.reserve(&Executor::default(), COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let executor = Executor::for_ram(&[], THREADS, 0)?;
        let vectors = rows(&slab, DIMENSIONS)?;
        index.add(&executor, &keys, &vectors).run()?;
        assert_eq!(index.describe(&Executor::default())?.size, COUNT);

        let matches = index.search(&executor, &vectors, WANTED).expansion_factor(64).run()?;
        assert!(
            matches.rows().all(|row| row.keys.len() == WANTED),
            "every query fills its own output row"
        );
        Ok(())
    }

    /// The null executor runs the call on the calling thread, with the same results.
    #[test]
    fn runs_on_the_calling_thread() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, _, slab) = populated_index(16, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let matches = index
            .search(&Executor::default(), &rows(&slab[..DIMENSIONS], DIMENSIONS)?, 4)
            .run()?;
        assert_eq!(matches.row(0).unwrap().keys.len(), 4);
        Ok(())
    }

    #[test]
    fn searches_while_inserting() -> Result<()> {
        const DIMENSIONS: usize = 64;
        const HALF: usize = 500;
        const WANTED: usize = 4;

        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        index.reserve(&Executor::default(), 2 * HALF)?;
        let keys = sequential_keys(2 * HALF);
        let slab = seeded_rows(2 * HALF, DIMENSIONS, SEED);

        // The first half lands serially, so the reader below traverses a populated graph.
        index
            .add(
                &executor(),
                &keys[..HALF],
                &rows(&slab[..HALF * DIMENSIONS], DIMENSIONS)?,
            )
            .run()?;

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
                    .add(&writer, &keys[HALF..], &vectors)
                    .run()
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
                index
                    .search(&reader, &queries, WANTED)
                    .expansion_factor(32)
                    .into(&mut matches)
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
        assert_eq!(index.describe(&Executor::default())?.size, 2 * HALF);
        Ok(())
    }

    // endregion: Concurrency
}

// endregion: Tests
