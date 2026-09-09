//! Vector Search over equidimensional dense embeddings - the Rust twin of
//! `include/usearch/dense_index.h`.
//!
//! Every batch input is a NumKong matrix, so a memory-mapped dataset feeds the index with no
//! reshape and no copy: packed slabs go through [`crate::rows`], padded ones through
//! `MatrixView::from_raw_parts`. Handles are created against one residency - RAM,
//! VRAM, or NVMe - picked by [`Residency`], which also decides what loading a checkpoint
//! means: materialize into RAM, upload to a device, or open in place on disk.
//!
//! A handle may be shared across threads, which is why the mutating verbs take `&self`.

use core::ffi::{c_int, c_void};
use core::marker::PhantomData;
use core::ptr;
use std::path::Path;

use numkong::{MatrixSpan, MatrixView, StorageElement, TensorRef};

use crate::types::ffi::{usearch_distance_t, usearch_error_t, usearch_key_t};
use crate::types::{
    check, fail, lower_progress, static_string, Continuation, Distance, Executor, Key, Residency, Result, ScalarKind,
    Status, PATH_NOT_UTF8,
};

// region: Raw Declarations

/// One-to-one mirror of `include/usearch/dense_index.h`.
mod ffi {
    #![allow(non_camel_case_types)]
    // A one-to-one mirror declares every verb, including the ones no method drives yet.
    #![allow(dead_code)]

    use core::ffi::{c_char, c_void};

    pub use crate::types::ffi::{
        usearch_bytes_span_t, usearch_bytes_view_t, usearch_distance_t, usearch_enum_t, usearch_error_t,
        usearch_executor_t, usearch_key_t, usearch_keys_span_t, usearch_keys_view_t, usearch_matches_span_t,
        usearch_matrix_span_t, usearch_matrix_view_t, usearch_predicate_t, usearch_progress_callback_t,
        usearch_residency_options_t,
    };

    pub type usearch_dense_index_t = *mut c_void;
    pub type usearch_dense_cursor_t = *mut c_void;
    pub type usearch_dense_finder_t = *mut c_void;

    pub type usearch_dense_metric_callback_t =
        unsafe extern "C" fn(*const c_void, *const c_void, usize, *mut c_void) -> usearch_distance_t;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_dense_options_t {
        pub metric: usearch_enum_t,
        pub metric_callback: Option<usearch_dense_metric_callback_t>,
        pub metric_callback_state: *mut c_void,
        pub quantization: usearch_enum_t,
        pub dimensions: usize,
        pub connectivity: usize,
        pub values_per_key: usize,
        pub residency: usearch_enum_t,
        pub residency_options: usearch_residency_options_t,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct usearch_dense_metadata_t {
        pub metric: usearch_enum_t,
        pub quantization: usearch_enum_t,
        pub dimensions: usize,
        pub connectivity: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct usearch_dense_softmax_state_t {
        pub maximum_score: f32,
        pub partition: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_dense_insertion_options_t {
        pub expansion_factor: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_dense_search_options_t {
        pub expansion_factor: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_dense_attention_options_t {
        pub expansion_factor: usize,
        pub score_scale: f32,
        pub mass_tolerance: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct usearch_dense_traversal_stats_t {
        pub visited_members: usize,
        pub computed_distances: usize,
        pub restarts: usize,
    }

    extern "C" {
        pub fn usearch_dense_init(
            options: *const usearch_dense_options_t,
            error: *mut usearch_error_t,
        ) -> usearch_dense_index_t;
        pub fn usearch_dense_free(index: usearch_dense_index_t);
        pub fn usearch_dense_reserve(index: usearch_dense_index_t, capacity: usize, error: *mut usearch_error_t);
        pub fn usearch_dense_clear(index: usearch_dense_index_t, error: *mut usearch_error_t);

        pub fn usearch_dense_size(index: usearch_dense_index_t, error: *mut usearch_error_t) -> usize;
        pub fn usearch_dense_capacity(index: usearch_dense_index_t, error: *mut usearch_error_t) -> usize;
        pub fn usearch_dense_dimensions(index: usearch_dense_index_t, error: *mut usearch_error_t) -> usize;
        pub fn usearch_dense_connectivity(index: usearch_dense_index_t, error: *mut usearch_error_t) -> usize;
        pub fn usearch_dense_memory_usage(index: usearch_dense_index_t, error: *mut usearch_error_t) -> usize;
        pub fn usearch_dense_capabilities(index: usearch_dense_index_t, error: *mut usearch_error_t) -> *const c_char;

        pub fn usearch_dense_add(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            keys: usearch_keys_view_t,
            vectors: *const usearch_matrix_view_t,
            options: *const usearch_dense_insertion_options_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        ) -> usize;

        pub fn usearch_dense_search(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            queries: *const usearch_matrix_view_t,
            matches: *const usearch_matches_span_t,
            options: *const usearch_dense_search_options_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_search_if(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            queries: *const usearch_matrix_view_t,
            matches: *const usearch_matches_span_t,
            options: *const usearch_dense_search_options_t,
            predicate: usearch_predicate_t,
            predicate_state: *mut c_void,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_refine(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            options: *const usearch_dense_insertion_options_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        ) -> usize;

        pub fn usearch_dense_multiply(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            left: *const usearch_matrix_view_t,
            products: *const usearch_matches_span_t,
            options: *const usearch_dense_search_options_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_maxsim(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            queries: *const usearch_matrix_view_t,
            rows_per_query: usize,
            matches: *const usearch_matches_span_t,
            options: *const usearch_dense_search_options_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_attend(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            queries: *const usearch_matrix_view_t,
            values: *const usearch_matrix_view_t,
            outputs: *const usearch_matrix_span_t,
            carry: *mut usearch_dense_softmax_state_t,
            options: *const usearch_dense_attention_options_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_count(
            index: usearch_dense_index_t,
            key: usearch_key_t,
            error: *mut usearch_error_t,
        ) -> usize;
        pub fn usearch_dense_get(
            index: usearch_dense_index_t,
            key: usearch_key_t,
            vectors: *const usearch_matrix_span_t,
            error: *mut usearch_error_t,
        ) -> usize;

        pub fn usearch_dense_merge(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            donors: *mut usearch_dense_index_t,
            donors_count: usize,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_cursor(
            index: usearch_dense_index_t,
            error: *mut usearch_error_t,
        ) -> usearch_dense_cursor_t;
        pub fn usearch_dense_cursor_next(
            cursor: usearch_dense_cursor_t,
            keys: usearch_keys_span_t,
            vectors: *const usearch_matrix_span_t,
            error: *mut usearch_error_t,
        ) -> usize;
        pub fn usearch_dense_cursor_free(cursor: usearch_dense_cursor_t);

        pub fn usearch_dense_find(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            query: *const usearch_matrix_view_t,
            error: *mut usearch_error_t,
        ) -> usearch_dense_finder_t;
        pub fn usearch_dense_find_next(
            finder: usearch_dense_finder_t,
            keys: usearch_keys_span_t,
            distances: *mut usearch_distance_t,
            error: *mut usearch_error_t,
        ) -> usize;
        pub fn usearch_dense_find_free(finder: usearch_dense_finder_t);

        pub fn usearch_dense_storage_requirements(index: usearch_dense_index_t, error: *mut usearch_error_t) -> usize;
        pub fn usearch_dense_save(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            path: usearch_bytes_view_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            error: *mut usearch_error_t,
        );
        pub fn usearch_dense_load(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            path: usearch_bytes_view_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            error: *mut usearch_error_t,
        );
        pub fn usearch_dense_metadata(
            path: usearch_bytes_view_t,
            metadata: *mut usearch_dense_metadata_t,
            error: *mut usearch_error_t,
        );
        pub fn usearch_dense_save_buffer(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            buffer: usearch_bytes_span_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            error: *mut usearch_error_t,
        );
        pub fn usearch_dense_load_buffer(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            buffer: usearch_bytes_view_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            error: *mut usearch_error_t,
        );
        pub fn usearch_dense_metadata_buffer(
            buffer: usearch_bytes_view_t,
            metadata: *mut usearch_dense_metadata_t,
            error: *mut usearch_error_t,
        );
    }
}

// endregion: Raw Declarations

// region: Options

/// The dense metric kinds supported out of the box, all derived from a single inner-product
/// kernel with pre-computed magnitudes. Hamming and Jaccard are bit-level, over the packed
/// sub-byte kinds.
///
/// This binding does not expose the C layer's custom-callback escape hatch: a caller-supplied
/// metric forfeits the SIMD kernel chains and the magnitude cache both.
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

/// Construction and loading options of a dense index.
///
/// Stored rows always round up to the SIMD step quantum, so rows stay cache-line aligned and
/// every kernel runs tail-free; the arena is slightly larger for it, and that is not a knob.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct Options {
    /// The number of indexed dimensions; zero defers the geometry to a later load, which
    /// reads it off the checkpoint.
    pub dimensions: usize,
    /// The metric kind used for distance calculations.
    pub metric: Metric,
    /// The scalar type vectors are stored and quantized into.
    pub quantization: ScalarKind,
    /// The connections-per-node bound of the graph; zero derives.
    pub connectivity: usize,
    /// Upper bound on entries sharing one key; zero derives to one - unique keys.
    pub values_per_key: usize,
    /// Where the payload lives, and what loading a checkpoint means.
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

impl Options {
    /// Marshals into the C structure, borrowing this one's placement lists.
    ///
    /// `path_views` receives the byte views an NVMe placement needs, so it must outlive the
    /// call the result is passed to.
    fn to_raw(&self, path_views: &mut Vec<crate::types::ffi::usearch_bytes_view_t>) -> ffi::usearch_dense_options_t {
        ffi::usearch_dense_options_t {
            metric: self.metric as i32,
            metric_callback: None,
            metric_callback_state: ptr::null_mut(),
            quantization: self.quantization as i32,
            dimensions: self.dimensions,
            connectivity: self.connectivity,
            values_per_key: self.values_per_key,
            residency: self.residency.tag(),
            residency_options: self.residency.to_raw(path_views),
        }
    }
}

/// The geometry and metric a checkpoint froze - everything a reader learns without opening it.
///
/// `values_per_key` is absent because it bounds a live handle's mutable state rather than the
/// serialized form, so a checkpoint records nothing about it. Placement is absent for the same
/// kind of reason: a checkpoint says where it was written, never where a reader must keep it.
/// Both stay the caller's to choose in [`Options`] before [`DenseIndex::new`].
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

// endregion: Options

// region: Per-Call Settings & Telemetry

/// Optional per-call settings of [`DenseIndex::add`] and [`DenseIndex::refine`].
///
/// A default derives every setting, which is what a null options pointer means to the C
/// interface - so there is one spelling of "decide for me" rather than an `Option` around a type
/// that already has one. Extensible, so construct from [`Default`] and set what matters.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct InsertionOptions {
    /// Candidate-pool breadth the build walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
}

impl InsertionOptions {
    /// The settings a given candidate-pool breadth names, with everything else derived.
    pub fn with_expansion_factor(expansion_factor: usize) -> Self {
        InsertionOptions { expansion_factor }
    }

    /// Lowers into the struct the C interface reads.
    fn lower(self) -> ffi::usearch_dense_insertion_options_t {
        ffi::usearch_dense_insertion_options_t {
            expansion_factor: self.expansion_factor,
        }
    }
}

/// Optional per-call settings of the read-only walks - [`DenseIndex::search`],
/// [`DenseIndex::search_if`], [`DenseIndex::multiply`] and [`DenseIndex::maxsim`].
///
/// A default derives every setting, and the type stays distinct from [`InsertionOptions`] even
/// where the two agree field for field, so a knob either side gains reaches that side alone.
/// Extensible, so construct from [`Default`] and set what matters.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct SearchOptions {
    /// Candidate-pool breadth the walk explores; zero derives it from the geometry.
    pub expansion_factor: usize,
}

impl SearchOptions {
    /// The settings a given candidate-pool breadth names, with everything else derived.
    pub fn with_expansion_factor(expansion_factor: usize) -> Self {
        SearchOptions { expansion_factor }
    }

    /// Lowers into the struct the C interface reads.
    fn lower(self) -> ffi::usearch_dense_search_options_t {
        ffi::usearch_dense_search_options_t {
            expansion_factor: self.expansion_factor,
        }
    }
}

/// Optional per-call settings of [`DenseIndex::attend`].
///
/// A default derives every setting. Extensible, so construct from [`Default`] and set what
/// matters rather than naming every field.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[non_exhaustive]
pub struct AttentionOptions {
    /// Candidate-pool breadth; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// The logit scale, conventionally the inverse square root of the head dimensionality;
    /// zero derives it from the index's frozen geometry.
    pub score_scale: f32,
    /// The residual-mass threshold the walk stops at - the mass-denominated sibling of the
    /// radius a search terminates on; zero derives.
    pub mass_tolerance: f32,
}

impl AttentionOptions {
    /// Lowers into the struct the C interface reads.
    fn lower(self) -> ffi::usearch_dense_attention_options_t {
        ffi::usearch_dense_attention_options_t {
            expansion_factor: self.expansion_factor,
            score_scale: self.score_scale,
            mass_tolerance: self.mass_tolerance,
        }
    }
}

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

// endregion: Per-Call Settings & Telemetry

// region: Results

/// One retrieved entry.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Match {
    /// The matched entry's key.
    pub key: Key,
    /// Its distance to the query, in the index's metric.
    pub distance: Distance,
}

/// One query's matches, trimmed to the cells the walk wrote.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MatchesRow<'matches> {
    /// The keys written for this query, nearest first.
    pub keys: &'matches [Key],
    /// Their distances, positionally paired with the keys.
    pub distances: &'matches [Distance],
    /// Matches the walk found, above the row's width when more were found than fit.
    pub found: usize,
}

impl MatchesRow<'_> {
    /// Whether the walk found more than the row could hold.
    pub fn truncated(&self) -> bool {
        self.found > self.keys.len()
    }
}

/// Where a batched search writes its results - three caller-owned buffers, sized together.
///
/// The buffer lengths decide how many matches each query may report: `keys.len() / counts.len()`
/// is the "k" in "kANN", so the wanted count cannot disagree with the space reserved for it. A
/// sparse product and late interaction land in the same shape: there a key names each pair's
/// indexed row and its distance carries the product or the summed maxima.
pub struct MatchesSpan<'buffers> {
    /// The `queries × wanted` key slab, rows contiguous.
    keys: &'buffers mut [Key],
    /// The `queries × wanted` distance slab, rows contiguous.
    distances: &'buffers mut [Distance],
    /// One match count per query.
    counts: &'buffers mut [usize],
    /// Columns per row, and the upper bound on matches - the "k" in "kANN".
    wanted: usize,
}

impl<'buffers> MatchesSpan<'buffers> {
    /// Lends three packed slabs: `counts.len() × wanted` keys and distances, one count per query.
    ///
    /// The row width follows from the lengths rather than being asked for, so the space reserved
    /// and the count asked for cannot disagree.
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

    /// Columns per row, and the upper bound on matches - the "k" in "kANN".
    pub fn wanted(&self) -> usize {
        self.wanted
    }

    /// Query rows the buffers were sized for.
    pub fn queries(&self) -> usize {
        self.counts.len()
    }

    /// Matches written for one query, keys and distances read together.
    ///
    /// # Panics
    /// Panics when `query` is not below [`MatchesSpan::queries`].
    pub fn row(&self, query: usize) -> MatchesRow<'_> {
        let found = self.counts[query];
        let start = query * self.wanted;
        let end = start + found.min(self.wanted);
        MatchesRow {
            keys: &self.keys[start..end],
            distances: &self.distances[start..end],
            found,
        }
    }

    /// Every query's row, in query order.
    pub fn rows(&self) -> impl Iterator<Item = MatchesRow<'_>> {
        (0..self.queries()).map(|query| self.row(query))
    }
}

// endregion: Results

// region: Dense Index

/// Vector Search Index over equidimensional dense representations.
///
/// ```no_run
/// use usearch::dense::{DenseIndex, InsertionOptions, MatchesSpan, Metric, Options, SearchOptions};
/// use usearch::{rows, Distance, Executor, Key, ScalarKind};
///
/// let mut options = Options::default();
/// options.dimensions = 4;
/// options.metric = Metric::Angular;
/// options.quantization = ScalarKind::I8;
///
/// let index = DenseIndex::new(&options)?;
/// index.reserve(1_000)?;
/// let executor = Executor::for_ram(&[], 0)?;
///
/// let keys = [Key::from(0u64), Key::from(1u64)];
/// let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
/// index.add(&executor, &keys, rows(&slab, 4), InsertionOptions::default(), None)?;
///
/// let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
/// let (mut found, mut scores, mut counts) = ([Key::NIL; 10], [0.0 as Distance; 10], [0usize; 1]);
/// let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
/// index.search(&executor, rows(&query, 4), SearchOptions::default(), &mut matches, None)?;
/// println!("{} matches", matches.row(0).found);
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
    fn drop(&mut self) {
        unsafe { ffi::usearch_dense_free(self.handle) }
    }
}

impl DenseIndex {
    /// Initializes a new index for the residency the options select.
    pub fn new(options: &Options) -> Result<Self> {
        let mut path_views = Vec::new();
        let raw = options.to_raw(&mut path_views);
        let mut error = usearch_error_t::default();
        let handle = unsafe { ffi::usearch_dense_init(&raw, &mut error) };
        check(error)?;
        Ok(DenseIndex { handle: owned(handle)? })
    }

    /// Reads a checkpoint's geometry and metric without loading it; a manifest path answers
    /// from its summary alone, so no shard is opened.
    pub fn metadata(path: impl AsRef<Path>) -> Result<Metadata> {
        let path = path.as_ref().to_str().ok_or(fail(Status::Io, PATH_NOT_UTF8))?;
        let mut raw = ffi::usearch_dense_metadata_t::default();
        let mut error = usearch_error_t::default();
        unsafe { ffi::usearch_dense_metadata(path.into(), &mut raw, &mut error) };
        check(error)?;
        Metadata::from_raw(&raw)
    }

    /// Reads geometry and metric out of a serialized monolith buffer.
    pub fn metadata_buffer(buffer: &[u8]) -> Result<Metadata> {
        let mut raw = ffi::usearch_dense_metadata_t::default();
        let mut error = usearch_error_t::default();
        unsafe { ffi::usearch_dense_metadata_buffer(view_of(buffer), &mut raw, &mut error) };
        check(error)?;
        Metadata::from_raw(&raw)
    }

    /// Reserves capacity for a number of incoming vectors ahead of insertion.
    pub fn reserve(&self, capacity: usize) -> Result<()> {
        let mut error = usearch_error_t::default();
        unsafe { ffi::usearch_dense_reserve(self.handle, capacity, &mut error) };
        check(error)
    }

    /// Erases all entries, keeping the reserved capacity and frozen geometry.
    pub fn clear(&self) -> Result<()> {
        let mut error = usearch_error_t::default();
        unsafe { ffi::usearch_dense_clear(self.handle, &mut error) };
        check(error)
    }

    /// The number of stored entries.
    pub fn size(&self) -> Result<usize> {
        self.read_integer_property(ffi::usearch_dense_size)
    }

    /// The reserved capacity in entries.
    pub fn capacity(&self) -> Result<usize> {
        self.read_integer_property(ffi::usearch_dense_capacity)
    }

    /// The frozen dimensionality of the stored vectors, zero while the geometry is deferred.
    pub fn dimensions(&self) -> Result<usize> {
        self.read_integer_property(ffi::usearch_dense_dimensions)
    }

    /// The frozen connections-per-node bound of the graph.
    pub fn connectivity(&self) -> Result<usize> {
        self.read_integer_property(ffi::usearch_dense_connectivity)
    }

    /// The memory usage of the index in bytes.
    pub fn memory_usage(&self) -> Result<usize> {
        self.read_integer_property(ffi::usearch_dense_memory_usage)
    }

    /// The bytes one monolith of the index takes under the default layout - what
    /// [`DenseIndex::save_buffer`] needs.
    pub fn storage_requirements(&self) -> Result<usize> {
        self.read_integer_property(ffi::usearch_dense_storage_requirements)
    }

    /// The capability codename the index dispatched to on this machine.
    pub fn capabilities(&self) -> Result<&'static str> {
        let mut error = usearch_error_t::default();
        let raw = unsafe { ffi::usearch_dense_capabilities(self.handle, &mut error) };
        check(error)?;
        match raw.is_null() {
            true => Err(fail(
                Status::Internal,
                "the library reported no capabilities and no error",
            )),
            false => Ok(unsafe { static_string(raw) }),
        }
    }

    /// Reads one of the handle's integer properties, which every geometry and footprint getter
    /// above shares the shape of; `capabilities` is the one property that answers text instead.
    fn read_integer_property(
        &self,
        entry_point: unsafe extern "C" fn(ffi::usearch_dense_index_t, *mut usearch_error_t) -> usize,
    ) -> Result<usize> {
        let mut error = usearch_error_t::default();
        let reading = unsafe { entry_point(self.handle, &mut error) };
        check(error)?;
        Ok(reading)
    }

    /// Adds a matrix of vectors under their keys - the native batch operation.
    ///
    /// Returns the row count on success and zero on failure: a batch is all-or-nothing, since
    /// every fallible step runs before the first entry identity is claimed. Overlaps safely
    /// with concurrent searches on the same handle, and the input scalar type is independent
    /// of the stored quantization - the library converts on ingest.
    pub fn add<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        keys: &[Key],
        vectors: MatrixView<'_, Scalar>,
        options: InsertionOptions,
        stats: Option<&mut Stats>,
    ) -> Result<usize> {
        self.add_dispatch(executor, keys, vectors, options, stats, None, ptr::null_mut())
    }

    /// Adds a matrix of vectors like [`DenseIndex::add`], reporting the rows admitted as they
    /// publish and stopping wherever `progress` says to.
    ///
    /// A handler makes the batch incremental rather than all-or-nothing: a cancellation stops at
    /// an increment boundary, the rows already published stand, and the call reports
    /// [`Status::Cancelled`] - so the count the handler last saw is what the index kept.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so a
    /// counter or a deadline needs no synchronization of its own. A panic inside it is caught and
    /// read as a cancellation.
    pub fn add_reporting<Scalar, Handler>(
        &self,
        executor: &Executor,
        keys: &[Key],
        vectors: MatrixView<'_, Scalar>,
        options: InsertionOptions,
        stats: Option<&mut Stats>,
        progress: &mut Handler,
    ) -> Result<usize>
    where
        Scalar: crate::types::Scalar,
        Handler: FnMut(usize, usize) -> Continuation + Send,
    {
        let (reports, state) = lower_progress(progress);
        self.add_dispatch(executor, keys, vectors, options, stats, reports, state)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_dispatch<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        keys: &[Key],
        vectors: MatrixView<'_, Scalar>,
        options: InsertionOptions,
        stats: Option<&mut Stats>,
        progress: crate::types::ffi::usearch_progress_callback_t,
        progress_state: *mut c_void,
    ) -> Result<usize> {
        let view = ffi::usearch_matrix_view_t::of(&vectors)?;
        if keys.len() != view.rows {
            return Err(fail(
                Status::ShapeMismatch,
                "the key count does not match the vector row count",
            ));
        }
        let lowered = options.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        let added = unsafe {
            ffi::usearch_dense_add(
                self.handle,
                executor.handle(),
                ffi::usearch_keys_view_t::of(keys),
                &view,
                &lowered,
                progress,
                progress_state,
                &mut raw_stats,
                &mut error,
            )
        };
        check(error)?;
        report(stats, raw_stats);
        Ok(added)
    }

    /// Resolves a matrix of queries into caller-owned outputs - the allocation-free batch path.
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    pub fn search<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Scalar>,
        options: SearchOptions,
        matches: &mut MatchesSpan<'_>,
        stats: Option<&mut Stats>,
    ) -> Result<()> {
        self.search_dispatch(
            executor,
            queries,
            options,
            matches,
            stats,
            None,
            ptr::null_mut(),
            None,
            ptr::null_mut(),
        )
    }

    /// Resolves a matrix of queries like [`DenseIndex::search`], reporting the queries resolved
    /// and stopping wherever `progress` says to.
    ///
    /// A cancelled batch reports [`Status::Cancelled`] and leaves every output row indeterminate,
    /// a query resolved before the stop being indistinguishable from one that never ran.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so a
    /// counter or a deadline needs no synchronization of its own. A panic inside it is caught and
    /// read as a cancellation.
    pub fn search_reporting<Scalar, Handler>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Scalar>,
        options: SearchOptions,
        matches: &mut MatchesSpan<'_>,
        stats: Option<&mut Stats>,
        progress: &mut Handler,
    ) -> Result<()>
    where
        Scalar: crate::types::Scalar,
        Handler: FnMut(usize, usize) -> Continuation + Send,
    {
        let (reports, state) = lower_progress(progress);
        self.search_dispatch(
            executor,
            queries,
            options,
            matches,
            stats,
            None,
            ptr::null_mut(),
            reports,
            state,
        )
    }

    /// The filtered sibling of [`DenseIndex::search`], admitting only keys the predicate approves.
    ///
    /// Non-matching entries still route the traversal, so recall does not collapse at low
    /// selectivity. The predicate runs on the executor's threads, hence `Sync`.
    pub fn search_if<Scalar: crate::types::Scalar, Predicate: Fn(Key) -> bool + Sync>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Scalar>,
        options: SearchOptions,
        matches: &mut MatchesSpan<'_>,
        stats: Option<&mut Stats>,
        predicate: &Predicate,
    ) -> Result<()> {
        unsafe extern "C" fn trampoline<Predicate: Fn(Key) -> bool + Sync>(
            key: usearch_key_t,
            state: *mut c_void,
        ) -> c_int {
            (*(state as *const Predicate))(Key::from_raw(key)) as c_int
        }
        self.search_dispatch(
            executor,
            queries,
            options,
            matches,
            stats,
            Some(trampoline::<Predicate>),
            predicate as *const Predicate as *mut c_void,
            None,
            ptr::null_mut(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn search_dispatch<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Scalar>,
        options: SearchOptions,
        matches: &mut MatchesSpan<'_>,
        stats: Option<&mut Stats>,
        predicate: crate::types::ffi::usearch_predicate_t,
        predicate_state: *mut c_void,
        progress: crate::types::ffi::usearch_progress_callback_t,
        progress_state: *mut c_void,
    ) -> Result<()> {
        let queries_view = ffi::usearch_matrix_view_t::of(&queries)?;
        let lowered_matches = ffi::usearch_matches_span_t::of(matches, queries_view.rows)?;
        let lowered = options.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();

        // One method per C verb: an unfiltered search is its own entry point, not the filtered
        // one handed a null predicate.
        match predicate {
            None => unsafe {
                ffi::usearch_dense_search(
                    self.handle,
                    executor.handle(),
                    &queries_view,
                    &lowered_matches,
                    &lowered,
                    progress,
                    progress_state,
                    &mut raw_stats,
                    &mut error,
                )
            },
            admits => unsafe {
                ffi::usearch_dense_search_if(
                    self.handle,
                    executor.handle(),
                    &queries_view,
                    &lowered_matches,
                    &lowered,
                    admits,
                    predicate_state,
                    progress,
                    progress_state,
                    &mut raw_stats,
                    &mut error,
                )
            },
        }
        check(error)?;
        report(stats, raw_stats);
        Ok(())
    }

    /// The number of entries stored under `key`.
    pub fn count(&self, key: Key) -> Result<usize> {
        let mut error = usearch_error_t::default();
        let counted = unsafe { ffi::usearch_dense_count(self.handle, key.to_raw(), &mut error) };
        check(error)?;
        Ok(counted)
    }

    /// Whether the index holds at least one entry under `key`.
    pub fn contains(&self, key: Key) -> Result<bool> {
        Ok(self.count(key)? != 0)
    }

    /// Retrieves the vectors stored under `key`, one row per entry, in insertion order.
    ///
    /// Returns the true number of entries, never the number written: a count above the
    /// matrix's row capacity means the copy was truncated, so allocate that many and call
    /// again. [`DenseIndex::count`] is the same number without the copy.
    pub fn get<Scalar: crate::types::Scalar>(&self, key: Key, vectors: &mut MatrixSpan<'_, Scalar>) -> Result<usize> {
        let span = ffi::usearch_matrix_span_t::of(vectors)?;
        let mut error = usearch_error_t::default();
        let entries = unsafe { ffi::usearch_dense_get(self.handle, key.to_raw(), &span, &mut error) };
        check(error)?;
        Ok(entries)
    }

    /// Re-selects the neighbors of every stored entry against the finished graph.
    ///
    /// A build step rather than a repair: a cheap insertion pass followed by one refinement
    /// reaches a better graph, in less total time, than a single expensive insertion pass - so a
    /// wide expansion factor belongs here rather than in [`DenseIndex::add`]. Returns the number
    /// of nodes whose edges changed, so a caller loops until that falls below whatever change
    /// rate they consider settled. Searches may run throughout; insertions may not.
    pub fn refine(&self, executor: &Executor, options: InsertionOptions, stats: Option<&mut Stats>) -> Result<usize> {
        self.refine_dispatch(executor, options, stats, None, ptr::null_mut())
    }

    /// Re-selects neighbors like [`DenseIndex::refine`], reporting the nodes swept and stopping
    /// wherever `progress` says to.
    ///
    /// Cancelling here is a time budget rather than a failure: re-selecting a node's edges is
    /// idempotent and claims no identity, so a stopped sweep leaves the graph correct and merely
    /// less improved, reports no cancelled status, and resumes where it stopped on the next call.
    /// A deadline check is the handler this asks for.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so a
    /// counter or a deadline needs no synchronization of its own. A panic inside it is caught and
    /// read as a cancellation.
    pub fn refine_reporting<Handler: FnMut(usize, usize) -> Continuation + Send>(
        &self,
        executor: &Executor,
        options: InsertionOptions,
        stats: Option<&mut Stats>,
        progress: &mut Handler,
    ) -> Result<usize> {
        let (reports, state) = lower_progress(progress);
        self.refine_dispatch(executor, options, stats, reports, state)
    }

    fn refine_dispatch(
        &self,
        executor: &Executor,
        options: InsertionOptions,
        stats: Option<&mut Stats>,
        progress: crate::types::ffi::usearch_progress_callback_t,
        progress_state: *mut c_void,
    ) -> Result<usize> {
        let lowered = options.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        let changed = unsafe {
            ffi::usearch_dense_refine(
                self.handle,
                executor.handle(),
                &lowered,
                progress,
                progress_state,
                &mut raw_stats,
                &mut error,
            )
        };
        check(error)?;
        report(stats, raw_stats);
        Ok(changed)
    }

    /// Approximate sparse product of a matrix against the indexed rows.
    ///
    /// The linear-algebra sibling of [`DenseIndex::search`], reporting every pair the traversal
    /// scored rather than only the nearest, so a row's keys name each pair's indexed row and its
    /// distances carry their product. Pairs arrive in traversal order rather than sorted. Runs on
    /// an inner-product index alone, since a product is a logit.
    pub fn multiply<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        left: MatrixView<'_, Scalar>,
        options: SearchOptions,
        products: &mut MatchesSpan<'_>,
        stats: Option<&mut Stats>,
    ) -> Result<()> {
        let view = ffi::usearch_matrix_view_t::of(&left)?;
        let lowered_products = ffi::usearch_matches_span_t::of(products, view.rows)?;
        let lowered = options.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_multiply(
                self.handle,
                executor.handle(),
                &view,
                &lowered_products,
                &lowered,
                None,
                ptr::null_mut(),
                &mut raw_stats,
                &mut error,
            )
        };
        check(error)?;
        report(stats, raw_stats);
        Ok(())
    }

    /// Late-interaction scoring of multi-row queries against multi-row entries.
    ///
    /// Every query row takes its best match within a key, and those maxima sum to that key's
    /// score - so `rows_per_query` is the token count each query occupies, `queries` holds them
    /// end to end, and the "k" the outputs size is over documents rather than rows. This is what
    /// `values_per_key` above one exists for. Runs on an inner-product index alone.
    ///
    /// See <https://arxiv.org/abs/2004.12832> - ColBERT, which introduced this scoring.
    pub fn maxsim<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Scalar>,
        rows_per_query: usize,
        options: SearchOptions,
        matches: &mut MatchesSpan<'_>,
        stats: Option<&mut Stats>,
    ) -> Result<()> {
        let view = ffi::usearch_matrix_view_t::of(&queries)?;
        if rows_per_query == 0 || view.rows % rows_per_query != 0 {
            return Err(fail(
                Status::ShapeMismatch,
                "the query rows do not divide into whole queries",
            ));
        }
        let lowered_matches = ffi::usearch_matches_span_t::of(matches, view.rows / rows_per_query)?;
        let lowered = options.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_maxsim(
                self.handle,
                executor.handle(),
                &view,
                rows_per_query,
                &lowered_matches,
                &lowered,
                None,
                ptr::null_mut(),
                &mut raw_stats,
                &mut error,
            )
        };
        check(error)?;
        report(stats, raw_stats);
        Ok(())
    }

    /// Softmax-weighted attention over the indexed keys, folded into the graph walk in one pass.
    ///
    /// The index holds the keys and `values` holds the value rows, addressed by key: an entry's
    /// key is its row offset into `values`, so a context inserted in position order needs no
    /// mapping. No match list is materialized, and each value row is gathered exactly once.
    /// `outputs` comes back already divided by its partition.
    ///
    /// Passing a `carry` asks for the online-softmax state, one entry per query row, which folds
    /// this branch into another over the same queries by partial-softmax rescaling. Runs on an
    /// inner-product index alone, since a softmax needs logits.
    ///
    /// See <https://arxiv.org/abs/2205.14135> - Flash Attention, whose recurrence this reuses.
    #[allow(clippy::too_many_arguments)] // The verb takes what the C one takes.
    pub fn attend<Query: crate::types::Scalar, Value: crate::types::Scalar>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Query>,
        values: MatrixView<'_, Value>,
        options: AttentionOptions,
        outputs: &mut MatrixSpan<'_, Value>,
        carry: Option<&mut [SoftmaxState]>,
        stats: Option<&mut Stats>,
    ) -> Result<()> {
        let queries_view = ffi::usearch_matrix_view_t::of(&queries)?;
        let values_view = ffi::usearch_matrix_view_t::of(&values)?;
        let outputs_span = ffi::usearch_matrix_span_t::of(outputs)?;
        if outputs_span.capacity < queries_view.rows || outputs_span.columns != values_view.columns {
            return Err(fail(
                Status::ShapeMismatch,
                "the outputs hold fewer rows than there are queries, or are not as wide as a value",
            ));
        }
        let carry_pointer = match carry {
            None => ptr::null_mut(),
            Some(state) if state.len() < queries_view.rows => {
                return Err(fail(
                    Status::ShapeMismatch,
                    "the softmax carry holds fewer entries than there are queries",
                ))
            }
            Some(state) => state.as_mut_ptr() as *mut ffi::usearch_dense_softmax_state_t,
        };
        let lowered = options.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_attend(
                self.handle,
                executor.handle(),
                &queries_view,
                &values_view,
                &outputs_span,
                carry_pointer,
                &lowered,
                None,
                ptr::null_mut(),
                &mut raw_stats,
                &mut error,
            )
        };
        check(error)?;
        report(stats, raw_stats);
        Ok(())
    }

    /// Absorbs every donor into this index, wiring cross-links via landmark stitching.
    ///
    /// Cheaper than re-inserting the donors' vectors one by one, and readers may keep searching
    /// throughout. The donors are consumed whether or not the merge succeeds.
    pub fn merge(&self, executor: &Executor, donors: Vec<DenseIndex>) -> Result<()> {
        let mut handles = surrendered(donors);
        self.merge_dispatch(executor, &mut handles, None, ptr::null_mut())
    }

    /// Absorbs every donor like [`DenseIndex::merge`], reporting the donors absorbed and stopping
    /// wherever `progress` says to.
    ///
    /// Cancellation stops between donors, so the ones already absorbed stay absorbed and the count
    /// the handler last saw names the boundary. The donors past it are freed here, the caller
    /// having handed them over by value.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so a
    /// counter or a deadline needs no synchronization of its own. A panic inside it is caught and
    /// read as a cancellation.
    pub fn merge_reporting<Handler: FnMut(usize, usize) -> Continuation + Send>(
        &self,
        executor: &Executor,
        donors: Vec<DenseIndex>,
        progress: &mut Handler,
    ) -> Result<()> {
        let mut handles = surrendered(donors);
        // The boundary a stop falls on is the caller's to read from their own reports, and this
        // side needs it too, to free the donors it was handed and the merge never took.
        let mut absorbed = 0usize;
        let merged = {
            let mut watch = |completed: usize, total: usize| {
                absorbed = completed;
                progress(completed, total)
            };
            let (reports, state) = lower_progress(&mut watch);
            self.merge_dispatch(executor, &mut handles, reports, state)
        };
        if let Err(failure) = merged {
            if failure.status == Status::Cancelled {
                let live = handles.get(absorbed..).unwrap_or_default();
                live.iter().copied().for_each(free_donor);
            }
            return Err(failure);
        }
        Ok(())
    }

    fn merge_dispatch(
        &self,
        executor: &Executor,
        handles: &mut [ffi::usearch_dense_index_t],
        progress: crate::types::ffi::usearch_progress_callback_t,
        progress_state: *mut c_void,
    ) -> Result<()> {
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_merge(
                self.handle,
                executor.handle(),
                handles.as_mut_ptr(),
                handles.len(),
                progress,
                progress_state,
                &mut error,
            )
        };
        check(error)
    }

    /// Opens a snapshot cursor over the current contents, capturing the publication watermark
    /// so concurrent inserts never appear mid-iteration.
    pub fn cursor<Scalar: crate::types::Scalar>(&self) -> Result<Cursor<'_, Scalar>> {
        let mut error = usearch_error_t::default();
        let cursor = unsafe { ffi::usearch_dense_cursor(self.handle, &mut error) };
        check(error)?;
        Ok(Cursor {
            cursor: owned(cursor)?,
            index: PhantomData,
        })
    }

    /// Opens a resumable nearest-neighbor stream for one query.
    ///
    /// Best-first traversal is naturally pausable, so callers that post-filter aggressively pull
    /// until satisfied instead of guessing "k" upfront. Each open finder pins one traversal
    /// context's memory - drop it promptly.
    pub fn find<'borrow, Scalar: crate::types::Scalar>(
        &'borrow self,
        executor: &'borrow Executor,
        query: &[Scalar],
    ) -> Result<Finder<'borrow>> {
        let view = ffi::usearch_matrix_view_t::of_row(query);
        let mut error = usearch_error_t::default();
        let finder = unsafe { ffi::usearch_dense_find(self.handle, executor.handle(), &view, &mut error) };
        check(error)?;
        Ok(Finder {
            finder: owned(finder)?,
            batch: [Match {
                key: Key::NIL,
                distance: 0.0,
            }; MATCHES_PER_PULL],
            filled: 0,
            position: 0,
            drained: false,
            index: PhantomData,
        })
    }

    /// Saves the index in the layout the path extension picks - one `.safegraphs` monolith, or a
    /// `.safegraphs.json` manifest beside its shards.
    ///
    /// Demands a mutation-free window; concurrent readers keep searching throughout.
    pub fn save(&self, executor: &Executor, path: impl AsRef<Path>) -> Result<()> {
        self.save_dispatch(executor, path, None, ptr::null_mut())
    }

    /// Saves the index like [`DenseIndex::save`], reporting the shards written and stopping
    /// wherever `progress` says to.
    ///
    /// A cancelled save leaves a monolith without its header, or the parts of a manifest layout
    /// without the manifest - leftovers every reader rejects and the caller is left to remove.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so a
    /// counter or a deadline needs no synchronization of its own. A panic inside it is caught and
    /// read as a cancellation.
    pub fn save_reporting<Handler: FnMut(usize, usize) -> Continuation + Send>(
        &self,
        executor: &Executor,
        path: impl AsRef<Path>,
        progress: &mut Handler,
    ) -> Result<()> {
        let (reports, state) = lower_progress(progress);
        self.save_dispatch(executor, path, reports, state)
    }

    fn save_dispatch(
        &self,
        executor: &Executor,
        path: impl AsRef<Path>,
        progress: crate::types::ffi::usearch_progress_callback_t,
        progress_state: *mut c_void,
    ) -> Result<()> {
        let path = path.as_ref().to_str().ok_or(fail(Status::Io, PATH_NOT_UTF8))?;
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_save(
                self.handle,
                executor.handle(),
                path.into(),
                progress,
                progress_state,
                &mut error,
            )
        };
        check(error)
    }

    /// Loads a checkpoint, with this handle's residency deciding the behavior: materialize into
    /// RAM, upload to the device, or open in place on disk.
    ///
    /// A handle whose geometry was left derived freezes it here, from the checkpoint's headers.
    pub fn load(&self, executor: &Executor, path: impl AsRef<Path>) -> Result<()> {
        self.load_dispatch(executor, path, None, ptr::null_mut())
    }

    /// Loads a checkpoint like [`DenseIndex::load`], reporting the shards placed and stopping
    /// wherever `progress` says to.
    ///
    /// A cancelled load publishes nothing: the handle is empty afterwards and loads cleanly again.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so a
    /// counter or a deadline needs no synchronization of its own. A panic inside it is caught and
    /// read as a cancellation.
    pub fn load_reporting<Handler: FnMut(usize, usize) -> Continuation + Send>(
        &self,
        executor: &Executor,
        path: impl AsRef<Path>,
        progress: &mut Handler,
    ) -> Result<()> {
        let (reports, state) = lower_progress(progress);
        self.load_dispatch(executor, path, reports, state)
    }

    fn load_dispatch(
        &self,
        executor: &Executor,
        path: impl AsRef<Path>,
        progress: crate::types::ffi::usearch_progress_callback_t,
        progress_state: *mut c_void,
    ) -> Result<()> {
        let path = path.as_ref().to_str().ok_or(fail(Status::Io, PATH_NOT_UTF8))?;
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_load(
                self.handle,
                executor.handle(),
                path.into(),
                progress,
                progress_state,
                &mut error,
            )
        };
        check(error)
    }

    /// Saves the index into a caller-owned buffer as one monolith.
    pub fn save_buffer(&self, executor: &Executor, buffer: &mut [u8]) -> Result<()> {
        let span = ffi::usearch_bytes_span_t {
            data: buffer.as_mut_ptr(),
            length: buffer.len(),
        };
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_save_buffer(self.handle, executor.handle(), span, None, ptr::null_mut(), &mut error)
        };
        check(error)
    }

    /// Loads a monolith from a buffer; a residency that opens in place demands the buffer
    /// outlive the handle.
    pub fn load_buffer(&self, executor: &Executor, buffer: &[u8]) -> Result<()> {
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_load_buffer(
                self.handle,
                executor.handle(),
                view_of(buffer),
                None,
                ptr::null_mut(),
                &mut error,
            )
        };
        check(error)
    }
}

// endregion: Dense Index

// region: Cursors & Finders

/// A snapshot cursor over the index contents; see [`DenseIndex::cursor`].
///
/// The walk is a sequential row scan, so it stays a batch operation: export, migration, and
/// visualization all move whole slabs.
#[derive(Debug)]
pub struct Cursor<'index, Scalar: crate::types::Scalar> {
    cursor: ffi::usearch_dense_cursor_t,
    index: PhantomData<(&'index DenseIndex, Scalar)>,
}

impl<'index, Scalar: crate::types::Scalar> Cursor<'index, Scalar> {
    /// Copies the next entries into caller-owned outputs, returning the rows written; zero
    /// means the snapshot is exhausted.
    ///
    /// Passing `None` for the vectors walks the key column alone.
    pub fn next_batch(&mut self, keys: &mut [Key], vectors: Option<&mut MatrixSpan<'_, Scalar>>) -> Result<usize> {
        // A key-only walk names no matrix, and the row count is then the key slab's own length.
        let span = match vectors {
            None => None,
            Some(matrix) => Some(ffi::usearch_matrix_span_t::of(matrix)?),
        };
        let mut error = usearch_error_t::default();
        let written = unsafe {
            ffi::usearch_dense_cursor_next(
                self.cursor,
                ffi::usearch_keys_span_t::of(keys),
                span.as_ref().map_or(ptr::null(), |span| span as *const _),
                &mut error,
            )
        };
        check(error)?;
        Ok(written)
    }
}

impl<'index, Scalar: crate::types::Scalar> Drop for Cursor<'index, Scalar> {
    fn drop(&mut self) {
        unsafe { ffi::usearch_dense_cursor_free(self.cursor) }
    }
}

/// Matches a finder pulls per crossing into the library, amortising the call over a run of them.
const MATCHES_PER_PULL: usize = 64;

/// A resumable nearest-neighbor stream; see [`DenseIndex::find`].
///
/// Ordering beyond the expansion horizon is best-effort nearest, not globally exact. Matches
/// arrive in batches and are yielded one at a time, so nothing allocates per item.
#[derive(Debug)]
pub struct Finder<'index> {
    finder: ffi::usearch_dense_finder_t,
    batch: [Match; MATCHES_PER_PULL],
    filled: usize,
    position: usize,
    drained: bool,
    index: PhantomData<(&'index DenseIndex, &'index Executor)>,
}

impl<'index> Finder<'index> {
    fn refill(&mut self) -> Result<()> {
        let mut keys = [Key::NIL; MATCHES_PER_PULL];
        let mut distances = [0.0; MATCHES_PER_PULL];
        let mut error = usearch_error_t::default();
        let written = unsafe {
            ffi::usearch_dense_find_next(
                self.finder,
                ffi::usearch_keys_span_t::of(&mut keys),
                distances.as_mut_ptr(),
                &mut error,
            )
        };
        check(error)?;
        for position in 0..written {
            self.batch[position] = Match {
                key: keys[position],
                distance: distances[position],
            };
        }
        self.filled = written;
        self.position = 0;
        self.drained = written == 0;
        Ok(())
    }
}

impl<'index> Iterator for Finder<'index> {
    type Item = Result<Match>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position == self.filled {
            if self.drained {
                return None;
            }
            if let Err(failure) = self.refill() {
                self.drained = true;
                return Some(Err(failure));
            }
            if self.filled == 0 {
                return None;
            }
        }
        self.position += 1;
        Some(Ok(self.batch[self.position - 1]))
    }
}

impl<'index> core::iter::FusedIterator for Finder<'index> {}

impl<'index> Drop for Finder<'index> {
    fn drop(&mut self) {
        unsafe { ffi::usearch_dense_find_free(self.finder) }
    }
}

// endregion: Cursors & Finders

// region: Helpers

/// What a matrix supplies to a descriptor, once the layout it must satisfy is checked.
struct MatrixLayout {
    /// Rows the matrix spans, which a span reports as its capacity.
    rows: usize,
    /// Logical scalars per row, which the engine checks against the frozen geometry.
    columns: usize,
    /// Bytes between the starts of consecutive rows.
    stride_bytes: usize,
}

impl MatrixLayout {
    /// Reads a matrix's shape, refusing the layouts the C interface cannot traverse.
    ///
    /// Only the layout question stays on this side - whether rows are contiguous and ascend in
    /// memory, which NumKong can answer and a bare pointer cannot. The width crosses in the
    /// descriptor and is checked against the frozen geometry by the engine.
    fn of<Element: StorageElement>(matrix: &impl TensorRef<Element, 2>) -> Result<Self> {
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
            rows: matrix.shape()[0],
            columns: matrix.shape()[1],
            stride_bytes: pitch as usize,
        })
    }
}

impl ffi::usearch_matrix_view_t {
    /// Borrows a matrix as the input descriptor every batch verb reads.
    fn of<Scalar: crate::types::Scalar>(matrix: &MatrixView<'_, Scalar>) -> Result<Self> {
        let layout = MatrixLayout::of(matrix)?;
        Ok(ffi::usearch_matrix_view_t {
            data: matrix.as_ptr() as *const c_void,
            rows: layout.rows,
            columns: layout.columns,
            stride_bytes: layout.stride_bytes,
            scalar_kind: Scalar::KIND as i32,
        })
    }

    /// The one-row descriptor a finder takes, whose width is the slab's whole length.
    fn of_row<Scalar: crate::types::Scalar>(row: &[Scalar]) -> Self {
        ffi::usearch_matrix_view_t {
            data: row.as_ptr() as *const c_void,
            rows: 1,
            columns: crate::types::dimensions_of::<Scalar>(row.len()),
            stride_bytes: 0,
            scalar_kind: Scalar::KIND as i32,
        }
    }
}

impl ffi::usearch_matrix_span_t {
    /// Lends a caller-owned matrix as the output descriptor a copying verb writes into.
    fn of<Scalar: crate::types::Scalar>(matrix: &mut MatrixSpan<'_, Scalar>) -> Result<Self> {
        let layout = MatrixLayout::of(matrix)?;
        Ok(ffi::usearch_matrix_span_t {
            data: matrix.as_mut_ptr() as *mut c_void,
            capacity: layout.rows,
            columns: layout.columns,
            stride_bytes: layout.stride_bytes,
            scalar_kind: Scalar::KIND as i32,
        })
    }
}

impl ffi::usearch_matches_span_t {
    /// Lends the three ranked outputs, refusing a span that cannot hold @p queries rows.
    fn of(matches: &mut MatchesSpan<'_>, queries: usize) -> Result<Self> {
        if matches.queries() < queries {
            return Err(fail(
                Status::ShapeMismatch,
                "the outputs hold fewer rows than there are queries",
            ));
        }
        let wanted = matches.wanted();
        Ok(ffi::usearch_matches_span_t {
            keys: matches.keys.as_mut_ptr() as *mut usearch_key_t,
            keys_stride_bytes: 0,
            scores: matches.distances.as_mut_ptr() as *mut usearch_distance_t,
            scores_stride_bytes: 0,
            counts: matches.counts.as_mut_ptr(),
            wanted,
        })
    }
}

/// Borrows a byte slab as the view the C interface accepts.
fn view_of(bytes: &[u8]) -> ffi::usearch_bytes_view_t {
    ffi::usearch_bytes_view_t {
        data: bytes.as_ptr(),
        length: bytes.len(),
    }
}

/// Rejects a null handle the library returned without reporting an error.
fn owned(handle: *mut c_void) -> Result<*mut c_void> {
    match handle.is_null() {
        true => Err(fail(Status::Internal, "the library returned no handle and no error")),
        false => Ok(handle),
    }
}

/// Copies the donors' handles into the contiguous array the C interface reads, forgetting the
/// wrappers, since a merge consumes every donor it absorbs.
fn surrendered(donors: Vec<DenseIndex>) -> Vec<ffi::usearch_dense_index_t> {
    let handles: Vec<ffi::usearch_dense_index_t> = donors.iter().map(|donor| donor.handle).collect();
    donors.into_iter().for_each(core::mem::forget);
    handles
}

/// Frees a donor a cancelled merge left live, the caller having handed it over by value.
fn free_donor(handle: ffi::usearch_dense_index_t) {
    if !handle.is_null() {
        unsafe { ffi::usearch_dense_free(handle) }
    }
}

fn report(sink: Option<&mut Stats>, raw: ffi::usearch_dense_traversal_stats_t) {
    if let Some(sink) = sink {
        *sink = Stats {
            visited_members: raw.visited_members,
            computed_distances: raw.computed_distances,
            restarts: raw.restarts,
        };
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
    use crate::types::{rows, rows_mut, Error};

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

    fn sequential_keys(count: usize) -> Vec<Key> {
        (0..count as u64).map(Key::from).collect()
    }

    /// One executor per test, since an executor admits a single driver at a time.
    fn executor() -> Executor {
        Executor::for_ram(&[], 0).expect("the executor must spawn")
    }

    fn empty_index(dimensions: usize, metric: Metric, quantization: ScalarKind) -> Result<DenseIndex> {
        let index = DenseIndex::new(&Options {
            dimensions,
            metric,
            quantization,
            ..Options::default()
        })?;
        index.reserve(1024)?;
        Ok(index)
    }

    fn populated_index(
        count: usize,
        dimensions: usize,
        metric: Metric,
        quantization: ScalarKind,
    ) -> Result<(DenseIndex, Vec<Key>, Vec<f32>)> {
        let index = empty_index(dimensions, metric, quantization)?;
        index.reserve(count)?;
        let keys = sequential_keys(count);
        let slab = seeded_rows(count, dimensions, SEED);
        let added = index.add(
            &executor(),
            &keys,
            rows(&slab, dimensions),
            InsertionOptions::default(),
            None,
        )?;
        assert_eq!(added, count);
        Ok((index, keys, slab))
    }

    /// How many matches one query reports - the shape most assertions here want.
    fn matches_of(index: &DenseIndex, query: &[f32], dimensions: usize) -> Result<usize> {
        let mut keys = [Key::NIL; 1];
        let mut distances = [0.0 as Distance; 1];
        let mut counts = [0usize; 1];
        let mut matches = MatchesSpan::new(&mut keys, &mut distances, &mut counts)?;
        index.search(
            &executor(),
            rows(query, dimensions),
            SearchOptions::default(),
            &mut matches,
            None,
        )?;
        Ok(matches.row(0).found)
    }

    /// A checkpoint path scoped by test name and process, never the working directory.
    fn scratch_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("usearch-rust-{name}-{}.safegraphs", std::process::id()))
    }

    fn remove_quietly(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
    }

    fn close(left: f64, right: f64, tolerance: f64) -> bool {
        (left - right).abs() <= tolerance
    }

    fn requires_send_sync<Subject: Send + Sync>() {}

    /// Packs one byte per element, LSB first, as NumKong's bit vectors are laid out.
    fn bit_rows(bytes: &[u8]) -> Vec<u1x8> {
        bytes.iter().map(|&byte| u1x8(byte)).collect()
    }

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

    // region: Options & Metadata Marshalling

    #[test]
    fn options_lower_every_field() {
        let mut path_views = Vec::new();
        let pinned = Options {
            dimensions: 768,
            metric: Metric::Euclidean,
            quantization: ScalarKind::I8,
            connectivity: 32,
            values_per_key: 4,
            residency: Residency::default(),
        };
        let raw = pinned.to_raw(&mut path_views);
        assert_eq!(raw.dimensions, 768);
        assert_eq!(raw.metric, Metric::Euclidean as i32);
        assert_eq!(raw.quantization, ScalarKind::I8 as i32);
        assert_eq!(raw.connectivity, 32);
        assert_eq!(raw.values_per_key, 4);
        assert_eq!(raw.residency, Residency::default().tag());
    }

    #[test]
    fn options_carry_the_placement_a_residency_names() {
        let mut path_views = Vec::new();
        // Every arm and the ABI tag that selects its union member.
        let arms = [
            (
                Residency::Ram {
                    memory_domains: vec![0, 1],
                },
                0,
            ),
            (
                Residency::Vram {
                    devices: vec![0, 1, 2, 3],
                },
                1,
            ),
            (
                Residency::Nvme {
                    paths: vec!["/mnt/nvme0".to_string(), "/mnt/nvme1".to_string()],
                },
                2,
            ),
        ];
        for (residency, tag) in arms {
            let raw = Options {
                residency,
                ..Options::default()
            }
            .to_raw(&mut path_views);
            assert_eq!(raw.residency, tag);
        }

        // The list travels inside the arm the tag selects, not beside it.
        let devices = Options {
            residency: Residency::Vram {
                devices: vec![0, 1, 2, 3],
            },
            ..Options::default()
        };
        assert_eq!(
            unsafe { devices.to_raw(&mut path_views).residency_options.vram.devices_count },
            4
        );
    }

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

    #[test]
    fn options_carry_no_metric_callback() {
        let mut path_views = Vec::new();
        let raw = Options::default().to_raw(&mut path_views);
        assert!(raw.metric_callback.is_none(), "this binding exposes no custom metric");
        assert!(raw.metric_callback_state.is_null());
    }

    // endregion: Options & Metadata Marshalling

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
            let index = empty_index(64, Metric::Angular, quantization)?;
            assert_eq!(index.dimensions()?, 64, "{quantization:?}");
            assert_eq!(index.size()?, 0, "{quantization:?}");
            assert!(index.connectivity()? > 0, "{quantization:?}");
        }

        // The bit-level metrics take packed rows, so each metric meets a kind it accepts.
        for (metric, quantization) in [
            (Metric::Inner, ScalarKind::F32),
            (Metric::Angular, ScalarKind::F32),
            (Metric::Euclidean, ScalarKind::F32),
            (Metric::Hamming, ScalarKind::U1x8),
            (Metric::Jaccard, ScalarKind::U1x8),
        ] {
            assert_eq!(empty_index(64, metric, quantization)?.dimensions()?, 64, "{metric:?}");
        }
        Ok(())
    }

    #[test]
    fn does_not_preallocate_members() -> Result<()> {
        let index = DenseIndex::new(&Options {
            dimensions: 16,
            ..Options::default()
        })?;
        assert_eq!(index.size()?, 0);
        assert_eq!(index.capacity()?, 0);

        index.reserve(1024)?;
        assert!(index.capacity()? >= 1024);
        assert_eq!(index.size()?, 0);
        Ok(())
    }

    /// A zeroed width defers the geometry to a later load, never to the first batch - which
    /// has no width to freeze from that the index would trust.
    #[test]
    fn defers_geometry_to_a_checkpoint() -> Result<()> {
        let index = DenseIndex::new(&Options {
            dimensions: 0,
            quantization: ScalarKind::F32,
            ..Options::default()
        })?;
        assert_eq!(index.dimensions()?, 0);

        index.reserve(4)?;
        let slab = seeded_rows(1, 12, SEED);
        assert!(
            matches!(
                index.add(
                    &executor(),
                    &sequential_keys(1),
                    rows(&slab, 12),
                    InsertionOptions::default(),
                    None
                ),
                Err(Error {
                    status: Status::Uninitialized,
                    ..
                })
            ),
            "an unfrozen geometry admits no rows"
        );

        let written = populated_index(4, 12, Metric::Angular, ScalarKind::F32)?.0;
        let checkpoint = scratch_path("defers_geometry_to_a_checkpoint");
        written.save(&executor(), &checkpoint)?;
        index.load(&executor(), &checkpoint)?;
        remove_quietly(&checkpoint);
        assert_eq!(index.dimensions()?, 12);
        Ok(())
    }

    /// The handle is one pointer, so a move must not disturb it and two threads may share it.
    #[test]
    fn survives_box_and_arc_moves() -> Result<()> {
        requires_send_sync::<DenseIndex>();
        let (index, _, _) = populated_index(4, 8, Metric::Angular, ScalarKind::F32)?;

        let boxed = Box::new(index);
        assert_eq!(boxed.size()?, 4);
        let shared = std::sync::Arc::new(*boxed);
        assert_eq!(shared.size()?, 4);
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
        let added = index.add(
            &executor(),
            &keys,
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;
        assert_eq!(added, 2);
        assert_eq!(index.size()?, 2);

        let mut restored = [0.0f32; DIMENSIONS];
        let mut span = rows_mut(&mut restored, DIMENSIONS);
        assert_eq!(index.get(keys[1], &mut span)?, 1);
        assert_eq!(restored, [0.0, 1.0, 0.0, 0.0]);
        Ok(())
    }

    /// An empty batch is a shape rather than a mistake, so it publishes nothing and reports no
    /// failure - what a caller filtering its own input relies on.
    #[test]
    fn accepts_an_empty_batch() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let empty: [f32; 0] = [];
        let added = index.add(
            &executor(),
            &[],
            rows(&empty, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;
        assert_eq!(added, 0);
        assert_eq!(index.size()?, 0);
        Ok(())
    }

    #[test]
    fn rejects_mismatched_key_and_row_counts() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        assert!(matches!(
            index.add(
                &executor(),
                &sequential_keys(3),
                rows(&slab, DIMENSIONS),
                InsertionOptions::default(),
                None,
            ),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn rejects_wrong_dimension_matrices() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, keys, _) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let narrow = seeded_rows(1, DIMENSIONS - 1, SEED);

        assert!(matches!(
            index.add(
                &executor(),
                &sequential_keys(1),
                rows(&narrow, DIMENSIONS - 1),
                InsertionOptions::default(),
                None,
            ),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));

        let mut buffer = [0.0f32; DIMENSIONS - 1];
        let mut span = rows_mut(&mut buffer, DIMENSIONS - 1);
        assert!(matches!(
            index.get(keys[0], &mut span),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));

        let mut found = [Key::NIL; 1];
        let mut scores = [0.0 as Distance; 1];
        let mut counts = [0usize; 1];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        assert!(matches!(
            index.search(
                &executor(),
                rows(&narrow, DIMENSIONS - 1),
                SearchOptions::default(),
                &mut matches,
                None,
            ),
            Err(Error {
                status: Status::ShapeMismatch,
                ..
            })
        ));

        assert!(
            matches!(
                index.find(&executor(), &narrow[..]),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "the query width is checked too"
        );
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
            unsafe { MatrixView::from_raw_parts(slab.as_ptr(), &[2, DIMENSIONS], &[element * 8, element * 2]) };
        assert!(matches!(
            index.add(
                &executor(),
                &sequential_keys(2),
                interleaved,
                InsertionOptions::default(),
                None,
            ),
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
        };
        assert!(matches!(
            index.add(
                &executor(),
                &sequential_keys(2),
                descending,
                InsertionOptions::default(),
                None,
            ),
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
        };
        let keys = sequential_keys(2);
        assert_eq!(
            index.add(&executor(), &keys, strided, InsertionOptions::default(), None)?,
            2
        );

        let mut restored = [-1.0f32; PITCH];
        let mut span: MatrixSpan<'_, f32> = unsafe {
            MatrixSpan::from_raw_parts(
                restored.as_mut_ptr(),
                &[1, DIMENSIONS],
                &[element * PITCH as isize, element],
            )
        };
        assert_eq!(index.get(keys[1], &mut span)?, 1);
        for (left, right) in restored[..DIMENSIONS].iter().zip(&rows_source[DIMENSIONS..]) {
            assert!(close(*left as f64, *right as f64, 1e-6));
        }
        assert!(
            restored[DIMENSIONS..].iter().all(|value| *value == -1.0),
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
        index.add(
            &executor(),
            &keys,
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;

        let mut wide = [0.0f64; DIMENSIONS];
        let mut span = rows_mut(&mut wide, DIMENSIONS);
        assert_eq!(index.get(keys[0], &mut span)?, 1);
        for (left, right) in wide.iter().zip(&slab) {
            assert!(close(*left, *right as f64, 1e-2), "bf16 keeps eight mantissa bits");
        }
        Ok(())
    }

    /// Get reports what exists, never what it wrote, so a caller sizes from the answer and
    /// calls again.
    #[test]
    fn reports_the_true_extent_when_truncated() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let shared = Key::from(3u64);
        let slab = seeded_rows(3, DIMENSIONS, SEED);
        index.add(
            &executor(),
            &[shared; 3],
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;

        assert_eq!(index.count(shared)?, 3);
        assert!(index.contains(shared)?);
        assert!(!index.contains(Key::from(8u64))?);

        // A one-row span cannot hold them, and the count says so rather than reporting one.
        let mut cramped = [0.0f32; DIMENSIONS];
        let mut span = rows_mut(&mut cramped, DIMENSIONS);
        assert_eq!(index.get(shared, &mut span)?, 3);

        // Sized from that answer, the second call copies everything.
        let mut roomy = [0.0f32; DIMENSIONS * 3];
        let mut span = rows_mut(&mut roomy, DIMENSIONS);
        assert_eq!(index.get(shared, &mut span)?, 3);
        for (left, right) in roomy.iter().zip(&slab) {
            assert!(close(*left as f64, *right as f64, 1e-6));
        }
        Ok(())
    }

    #[test]
    fn uses_128_bit_keys() -> Result<()> {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let wide = Key::from(u128::MAX);
        let narrow = Key::from(9u64);
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        index.add(
            &executor(),
            &[wide, narrow],
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;

        assert_eq!(index.count(wide)?, 1);
        assert_eq!(index.count(narrow)?, 1);
        assert_eq!(wide.as_u64(), None, "a non-zero high half has no integer identity");
        assert_eq!(narrow.as_u64(), Some(9));
        Ok(())
    }

    // region: Refinement & Linear Algebra

    /// The return is a change count rather than a status, so a caller loops until the graph
    /// settles.
    #[test]
    fn refinement_reports_its_sweep() -> Result<()> {
        const COUNT: usize = 64;
        let (index, _, _) = populated_index(COUNT, 8, Metric::Angular, ScalarKind::F32)?;
        let mut stats = Stats::default();
        let changed = index.refine(
            &executor(),
            InsertionOptions::with_expansion_factor(64),
            Some(&mut stats),
        )?;
        assert!(
            changed <= COUNT,
            "a sweep cannot change more nodes than the index holds"
        );
        assert!(stats.visited_members > 0, "the sweep reports the nodes it swept");
        Ok(())
    }

    /// The count is the true number of scored pairs, which the header lets exceed the column
    /// capacity to signal truncation.
    #[test]
    fn multiplication_reports_every_scored_pair() -> Result<()> {
        const DIMENSIONS: usize = 4;
        const ROWS: usize = 3;
        const WANTED: usize = 2;
        // A product is a logit, so the linear-algebra verbs take the inner kind alone.
        let index = empty_index(DIMENSIONS, Metric::Inner, ScalarKind::F32)?;
        let slab = seeded_rows(ROWS, DIMENSIONS, SEED);
        index.add(
            &executor(),
            &sequential_keys(ROWS),
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;

        let (mut columns, mut products, mut counts) = (
            vec![Key::NIL; ROWS * WANTED],
            vec![0.0; ROWS * WANTED],
            vec![0usize; ROWS],
        );
        let mut pairs = MatchesSpan::new(&mut columns, &mut products, &mut counts)?;
        index.multiply(
            &executor(),
            rows(&slab, DIMENSIONS),
            SearchOptions::default(),
            &mut pairs,
            None,
        )?;
        assert_eq!(pairs.wanted(), WANTED, "the row width follows from the buffer lengths");
        assert!(
            pairs.rows().all(|row| row.found == ROWS),
            "every row scores every column"
        );
        assert!(pairs.row(0).truncated(), "the count outruns the room it was given");
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
        index.reserve(DOCUMENTS * TOKENS)?;

        let slab = seeded_rows(DOCUMENTS * TOKENS, DIMENSIONS, SEED);
        let mut keys = Vec::with_capacity(DOCUMENTS * TOKENS);
        for document in 0..DOCUMENTS {
            keys.extend(core::iter::repeat_n(Key::from(document as u64), TOKENS));
        }
        index.add(
            &executor(),
            &keys,
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;

        let (mut found, mut scores, mut counts) = (
            vec![Key::NIL; QUERIES * DOCUMENTS],
            vec![0.0; QUERIES * DOCUMENTS],
            vec![0usize; QUERIES],
        );
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index.maxsim(
            &executor(),
            rows(&slab[..QUERIES * TOKENS * DIMENSIONS], DIMENSIONS),
            TOKENS,
            SearchOptions::default(),
            &mut matches,
            None,
        )?;
        assert!(
            matches.rows().all(|row| row.found == DOCUMENTS),
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
        index.add(
            &executor(),
            &sequential_keys(ENTRIES),
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;

        let values = seeded_rows(ENTRIES, VALUE_WIDTH, SEED + 1);
        let mut attended = vec![0.0f32; QUERIES * VALUE_WIDTH];
        let mut carry = vec![SoftmaxState::default(); QUERIES];
        let mut context = rows_mut(&mut attended, VALUE_WIDTH);
        index.attend(
            &executor(),
            rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS),
            rows(&values, VALUE_WIDTH),
            AttentionOptions::default(),
            &mut context,
            Some(&mut carry),
            None,
        )?;
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
        let (mut columns, mut products, mut counts) = (vec![Key::NIL; 2], vec![0.0; 2], vec![0usize; 1]);
        let mut pairs = MatchesSpan::new(&mut columns, &mut products, &mut counts)?;
        assert!(matches!(
            index.multiply(
                &executor(),
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                SearchOptions::default(),
                &mut pairs,
                None,
            ),
            Err(Error {
                status: Status::ConfigMismatch,
                ..
            })
        ));
        Ok(())
    }

    // endregion: Refinement & Linear Algebra

    // endregion: Ingest & Retrieval

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
        index.search(
            &executor(),
            rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS),
            SearchOptions::with_expansion_factor(64),
            &mut matches,
            None,
        )?;
        assert!(matches.rows().all(|row| row.found == WANTED));
        assert!(
            matches.rows().all(|row| !row.keys.contains(&untouched)),
            "every slot of every row was written"
        );
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

        let mut found = vec![Key::NIL; (QUERIES - 1) * 2];
        let mut scores = vec![0.0 as Distance; (QUERIES - 1) * 2];
        let mut counts = vec![0usize; QUERIES - 1];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        assert!(
            matches!(
                index.search(
                    &executor,
                    rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS),
                    SearchOptions::default(),
                    &mut matches,
                    None,
                ),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "fewer output rows than there are queries"
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
        index.add(
            &executor(),
            &keys,
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            Some(&mut inserting),
        )?;
        assert!(inserting.computed_distances > 0);

        let mut searching = Stats::default();
        let mut found = [Key::NIL; 4];
        let mut scores = [0.0 as Distance; 4];
        let mut counts = [0usize; 1];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index.search(
            &executor(),
            rows(&slab[..DIMENSIONS], DIMENSIONS),
            SearchOptions::default(),
            &mut matches,
            Some(&mut searching),
        )?;
        let first = searching;
        assert!(first.computed_distances > 0);

        index.search(
            &executor(),
            rows(&slab[..DIMENSIONS], DIMENSIONS),
            SearchOptions::default(),
            &mut matches,
            Some(&mut searching),
        )?;
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
        let mut found = [Key::NIL; WANTED];
        let mut scores = [0.0 as Distance; WANTED];
        let mut counts = [0usize; 1];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index.search_if(
            &executor(),
            rows(&slab[..DIMENSIONS], DIMENSIONS),
            SearchOptions::with_expansion_factor(64),
            &mut matches,
            None,
            &is_even,
        )?;

        let admitted = matches.row(0);
        assert!(admitted.found > 0);
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
        let is_allowed = |key: Key| allowed.contains(&key);
        let mut found = [Key::NIL; WANTED];
        let mut scores = [0.0 as Distance; WANTED];
        let mut counts = [0usize; 1];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index.search_if(
            &executor(),
            rows(&slab[..DIMENSIONS], DIMENSIONS),
            SearchOptions::with_expansion_factor(64),
            &mut matches,
            None,
            &is_allowed,
        )?;

        for key in matches.row(0).keys {
            assert!(allowed.contains(key), "{key} is outside the captured set");
        }
        Ok(())
    }

    #[test]
    fn filters_everything_out() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, _, slab) = populated_index(64, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let admits_nothing = |_: Key| false;
        let mut found = [Key::NIL; 4];
        let mut scores = [0.0 as Distance; 4];
        let mut counts = [0usize; 1];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index.search_if(
            &executor(),
            rows(&slab[..DIMENSIONS], DIMENSIONS),
            SearchOptions::with_expansion_factor(64),
            &mut matches,
            None,
            &admits_nothing,
        )?;
        assert_eq!(matches.row(0).found, 0, "a fully-rejecting predicate is not an error");
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
        index.add(
            &executor(),
            &keys,
            rows(&stored, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;
        assert_eq!(index.size()?, 2, "two bytes are two rows, not sixteen");

        let query = bit_rows(&[0b0111_1000]);
        let mut found = [Key::NIL; 2];
        let mut scores = [0.0 as Distance; 2];
        let mut counts = [0usize; 1];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index.search(
            &executor(),
            rows(&query, DIMENSIONS),
            SearchOptions::default(),
            &mut matches,
            None,
        )?;
        let row = matches.row(0);
        assert_eq!(row.found, 2);
        assert_eq!(
            row.keys.iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
        Ok(())
    }

    // endregion: Metrics

    // region: Serialization

    /// A derived geometry freezes from the checkpoint's own headers.
    #[test]
    fn saves_and_loads_a_file() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let path = scratch_path("saves_and_loads_a_file");
        let (index, _, _) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        index.save(&executor(), &path)?;

        let restored = DenseIndex::new(&Options {
            dimensions: 0,
            quantization: ScalarKind::F32,
            ..Options::default()
        })?;
        restored.load(&executor(), &path)?;
        assert_eq!(restored.dimensions()?, DIMENSIONS);
        assert_eq!(restored.size()?, 32);
        remove_quietly(&path);
        Ok(())
    }

    #[test]
    fn saves_and_loads_a_buffer() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, _, _) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let length = index.storage_requirements()?;
        assert!(length > 0);

        let mut buffer = vec![0u8; length];
        index.save_buffer(&executor(), &mut buffer)?;

        let restored = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        restored.load_buffer(&executor(), &buffer)?;
        assert_eq!(restored.size()?, index.size()?);

        let truncated = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        assert!(
            truncated.load_buffer(&executor(), &buffer[..length - 1]).is_err(),
            "a short buffer must not be read"
        );
        Ok(())
    }

    #[test]
    fn reads_metadata_without_loading() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let path = scratch_path("reads_metadata_without_loading");
        let (index, _, _) = populated_index(8, DIMENSIONS, Metric::Euclidean, ScalarKind::I8)?;
        index.save(&executor(), &path)?;

        let described = DenseIndex::metadata(&path)?;
        assert_eq!(described.dimensions, DIMENSIONS);
        assert_eq!(described.metric, Metric::Euclidean);
        assert_eq!(described.quantization, ScalarKind::I8);

        let mut buffer = vec![0u8; index.storage_requirements()?];
        index.save_buffer(&executor(), &mut buffer)?;
        assert_eq!(DenseIndex::metadata_buffer(&buffer)?.dimensions, DIMENSIONS);
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
            matches!(index.save(&executor(), path), Err(Error { status: Status::Io, .. })),
            "the C interface exchanges UTF-8 bytes"
        );
        Ok(())
    }

    #[test]
    fn clears_and_reuses() -> Result<()> {
        const DIMENSIONS: usize = 8;
        let (index, keys, slab) = populated_index(16, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let reserved = index.capacity()?;

        index.clear()?;
        assert_eq!(index.size()?, 0);
        assert_eq!(index.capacity()?, reserved, "clearing keeps the arena");
        assert_eq!(index.dimensions()?, DIMENSIONS, "and keeps the frozen geometry");

        index.add(
            &executor(),
            &keys,
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;
        assert_eq!(index.size()?, 16);
        Ok(())
    }

    // endregion: Serialization

    // region: Merge, Cursor & Finder

    #[test]
    fn merges_a_donor_index() -> Result<()> {
        const DIMENSIONS: usize = 16;
        let (index, keys, _) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let donor = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let donor_keys: Vec<Key> = (100..132u64).map(Key::from).collect();
        let donor_slab = seeded_rows(32, DIMENSIONS, SEED + 1);
        donor.add(
            &executor(),
            &donor_keys,
            rows(&donor_slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;

        // The donors are consumed by value, so a merged handle cannot be used again.
        index.merge(&executor(), vec![donor])?;
        assert_eq!(index.size()?, 64);
        assert!(index.contains(keys[0])? && index.contains(donor_keys[0])?);
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
            let mut span = rows_mut(&mut batch_vectors, DIMENSIONS);
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
        let written = cursor.next_batch(&mut batch_keys, None)?;
        assert_eq!(written, 4);
        assert_eq!(
            batch_keys[..written].iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
        Ok(())
    }

    #[test]
    fn streams_matches_from_a_finder() -> Result<()> {
        const DIMENSIONS: usize = 16;
        // Past the finder's internal refill boundary, so the batch seam is exercised.
        const COUNT: usize = 200;
        let (index, _, slab) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let executor = executor();
        let mut finder = index.find(&executor, &slab[..DIMENSIONS])?;
        let first: Vec<Match> = finder
            .by_ref()
            .take(5)
            .map(|found| found.expect("the stream must not fault"))
            .collect();
        assert_eq!(first.len(), 5, "a pull hands the caller as many as it asked for");

        let drained = finder.count();
        assert!(drained > MATCHES_PER_PULL, "the stream crosses its refill boundary");

        let mut exhausted = index.find(&executor, &slab[..DIMENSIONS])?;
        while exhausted.next().is_some() {}
        assert!(exhausted.next().is_none(), "a drained finder stays drained");
        Ok(())
    }

    // endregion: Merge, Cursor & Finder

    // region: Progress & Cancellation

    #[test]
    fn a_reported_batch_counts_up_to_the_rows_it_was_given() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        index.reserve(COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let mut watch = Watch::new(Patience::Complete);
        let added = index.add_reporting(
            &executor(),
            &keys,
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
            &mut |completed, total| watch.saw(completed, total),
        )?;

        assert_eq!(added, COUNT);
        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (COUNT, COUNT), "the last report names the whole batch");
        Ok(())
    }

    #[test]
    fn a_stopped_batch_keeps_the_rows_it_published() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 64;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        index.reserve(COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let mut watch = Watch::new(Patience::StopAtFirstReport);
        let added = index.add_reporting(
            &executor(),
            &keys,
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
            &mut |completed, total| watch.saw(completed, total),
        );

        // How much of the batch a stop leaves behind is the increment's size, not this test's to
        // name, so the count the handler last saw is what the index is checked against.
        match added {
            Ok(published) => assert_eq!(published, watch.last.0, "an uncancelled batch published it all"),
            Err(failure) => assert_eq!(failure.status, Status::Cancelled),
        }
        assert_eq!(
            index.size()?,
            watch.last.0,
            "the rows the handler saw are the rows that stand"
        );
        Ok(())
    }

    #[test]
    fn a_reported_search_counts_up_to_the_queries_it_was_given() -> Result<()> {
        const DIMENSIONS: usize = 8;
        const QUERIES: usize = 4;
        const WANTED: usize = 3;
        let (index, _, slab) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let mut keys = [Key::NIL; QUERIES * WANTED];
        let mut scores = [0.0 as Distance; QUERIES * WANTED];
        let mut counts = [0usize; QUERIES];
        let mut matches = MatchesSpan::new(&mut keys, &mut scores, &mut counts)?;

        let mut watch = Watch::new(Patience::Complete);
        index.search_reporting(
            &executor(),
            rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS),
            SearchOptions::default(),
            &mut matches,
            None,
            &mut |completed, total| watch.saw(completed, total),
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
        let (index, _, _) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32)?;

        let mut watch = Watch::new(Patience::StopAtFirstReport);
        index.refine_reporting(
            &executor(),
            InsertionOptions::default(),
            None,
            &mut |completed, total| watch.saw(completed, total),
        )?;

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
        index.save_reporting(&executor(), &path, &mut |completed, total| {
            writing.saw(completed, total)
        })?;
        assert!(writing.ordered, "a report never counts down");
        assert_eq!(writing.last.0, writing.last.1, "the last report names a finished write");
        assert!(writing.last.1 != 0);

        let restored = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        let mut placing = Watch::new(Patience::Complete);
        restored.load_reporting(&executor(), &path, &mut |completed, total| {
            placing.saw(completed, total)
        })?;
        assert!(placing.ordered, "a report never counts down");
        assert_eq!(placing.last.0, placing.last.1, "the last report names a finished read");
        assert_eq!(restored.size()?, COUNT);

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
        index.merge_reporting(&executor(), donors, &mut |completed, total| watch.saw(completed, total))?;

        assert!(watch.ordered, "a report never counts down");
        assert_eq!(watch.last, (2, 2), "one report per donor absorbed");
        assert_eq!(index.size()?, COUNT * 3);
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
        let merged = index.merge_reporting(&executor(), donors, &mut |completed, total| watch.saw(completed, total));

        match merged {
            Ok(()) => assert_eq!(watch.last.0, DONORS, "an uncancelled merge absorbed them all"),
            Err(failure) => assert_eq!(failure.status, Status::Cancelled),
        }
        assert_eq!(index.size()?, COUNT * (1 + watch.last.0));
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
        index.reserve(COUNT)?;
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let executor = Executor::for_ram(&[], THREADS)?;
        let added = index.add(
            &executor,
            &keys,
            rows(&slab, DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;
        assert_eq!(added, COUNT);
        assert_eq!(index.size()?, COUNT);

        let mut found = vec![Key::NIL; COUNT * WANTED];
        let mut scores = vec![0.0 as Distance; COUNT * WANTED];
        let mut counts = vec![0usize; COUNT];
        let mut matches = MatchesSpan::new(&mut found, &mut scores, &mut counts)?;
        index.search(
            &executor,
            rows(&slab, DIMENSIONS),
            SearchOptions::with_expansion_factor(64),
            &mut matches,
            None,
        )?;
        assert!(
            matches.rows().all(|row| row.found == WANTED),
            "every query fills its own output row"
        );
        Ok(())
    }

    #[test]
    fn searches_while_inserting() -> Result<()> {
        const DIMENSIONS: usize = 64;
        const HALF: usize = 500;
        const WANTED: usize = 4;

        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32)?;
        index.reserve(2 * HALF)?;
        let keys = sequential_keys(2 * HALF);
        let slab = seeded_rows(2 * HALF, DIMENSIONS, SEED);

        // The first half lands serially, so the reader below traverses a populated graph.
        let seeding = executor();
        index.add(
            &seeding,
            &keys[..HALF],
            rows(&slab[..HALF * DIMENSIONS], DIMENSIONS),
            InsertionOptions::default(),
            None,
        )?;
        drop(seeding);

        // An executor admits one driver at a time, so genuinely overlapping calls need one
        // apiece - the writer's and the reader's, on two threads sharing the handle.
        let writer = Executor::for_ram(&[], THREADS)?;
        let reader = Executor::for_ram(&[], THREADS)?;
        let known: HashSet<Key> = keys.iter().copied().collect();
        let strangers = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            scope.spawn(|| {
                let writer = writer;
                index
                    .add(
                        &writer,
                        &keys[HALF..],
                        rows(&slab[HALF * DIMENSIONS..], DIMENSIONS),
                        InsertionOptions::default(),
                        None,
                    )
                    .expect("the racing batch must insert");
            });
            scope.spawn(|| {
                let reader = reader;
                let mut found = vec![Key::NIL; HALF * WANTED];
                let mut scores = vec![0.0 as Distance; HALF * WANTED];
                let mut counts = vec![0usize; HALF];
                let mut matches =
                    MatchesSpan::new(&mut found, &mut scores, &mut counts).expect("the output buffers are one shape");
                index
                    .search(
                        &reader,
                        rows(&slab[..HALF * DIMENSIONS], DIMENSIONS),
                        SearchOptions::with_expansion_factor(32),
                        &mut matches,
                        None,
                    )
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
        assert_eq!(index.size()?, 2 * HALF);
        Ok(())
    }

    // endregion: Concurrency
}

// endregion: Tests
