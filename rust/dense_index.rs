//! Vector Search over equidimensional dense embeddings - the Rust twin of
//! `include/usearch/dense_index.h`.
//!
//! Every batch input and output is a NumKong matrix, so a memory-mapped dataset feeds the
//! index with no reshape and no copy: packed slabs go through [`crate::rows`], padded ones
//! through `MatrixView::from_raw_parts`. Handles are created against one residency - RAM,
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
    check, fail, static_string, Distance, Executor, Key, Residency, Result, ScalarKind, Status, PATH_NOT_UTF8,
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
        usearch_executor_t, usearch_key_t, usearch_predicate_t, usearch_progress_callback_t,
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
    #[derive(Clone, Copy)]
    pub struct usearch_dense_matrix_view_t {
        pub data: *const c_void,
        pub rows: usize,
        pub columns: usize,
        pub stride_bytes: usize,
        pub scalar_kind: usearch_enum_t,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_dense_matrix_span_t {
        pub data: *mut c_void,
        pub capacity: usize,
        pub columns: usize,
        pub stride_bytes: usize,
        pub scalar_kind: usearch_enum_t,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_dense_matches_span_t {
        pub keys: *mut usearch_key_t,
        pub keys_stride_bytes: usize,
        pub scores: *mut usearch_distance_t,
        pub scores_stride_bytes: usize,
        pub counts: *mut usize,
        pub wanted: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct usearch_dense_softmax_state_t {
        pub maximum_score: f32,
        pub partition: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_dense_expansion_config_t {
        pub expansion_factor: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_dense_attention_config_t {
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
            keys: *const usearch_key_t,
            vectors: *const usearch_dense_matrix_view_t,
            config: *const usearch_dense_expansion_config_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        ) -> usize;

        pub fn usearch_dense_search(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            queries: *const usearch_dense_matrix_view_t,
            matches: *const usearch_dense_matches_span_t,
            config: *const usearch_dense_expansion_config_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_search_if(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            queries: *const usearch_dense_matrix_view_t,
            matches: *const usearch_dense_matches_span_t,
            config: *const usearch_dense_expansion_config_t,
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
            config: *const usearch_dense_expansion_config_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        ) -> usize;

        pub fn usearch_dense_multiply(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            left: *const usearch_dense_matrix_view_t,
            products: *const usearch_dense_matches_span_t,
            config: *const usearch_dense_expansion_config_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_maxsim(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            queries: *const usearch_dense_matrix_view_t,
            rows_per_query: usize,
            matches: *const usearch_dense_matches_span_t,
            config: *const usearch_dense_expansion_config_t,
            progress: usearch_progress_callback_t,
            progress_state: *mut c_void,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_attend(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            queries: *const usearch_dense_matrix_view_t,
            values: *const usearch_dense_matrix_view_t,
            outputs: *const usearch_dense_matrix_span_t,
            carry: *mut usearch_dense_softmax_state_t,
            config: *const usearch_dense_attention_config_t,
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
            vectors: *const usearch_dense_matrix_span_t,
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
            keys: *mut usearch_key_t,
            vectors: *const usearch_dense_matrix_span_t,
            error: *mut usearch_error_t,
        ) -> usize;
        pub fn usearch_dense_cursor_free(cursor: usearch_dense_cursor_t);

        pub fn usearch_dense_find(
            index: usearch_dense_index_t,
            executor: usearch_executor_t,
            query: *const usearch_dense_matrix_view_t,
            error: *mut usearch_error_t,
        ) -> usearch_dense_finder_t;
        pub fn usearch_dense_find_next(
            finder: usearch_dense_finder_t,
            keys: *mut usearch_key_t,
            distances: *mut usearch_distance_t,
            wanted: usize,
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
            options: *mut usearch_dense_options_t,
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
            options: *mut usearch_dense_options_t,
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

    /// Reads back the geometry and metric a checkpoint froze; the placement comes back empty,
    /// since a checkpoint records where it was written rather than where a reader must keep it.
    fn from_raw(raw: &ffi::usearch_dense_options_t) -> Result<Self> {
        Ok(Options {
            dimensions: raw.dimensions,
            metric: Metric::from_raw(raw.metric)
                .ok_or(fail(Status::UnknownKind, "the checkpoint names an unknown metric"))?,
            quantization: ScalarKind::from_raw(raw.quantization)
                .ok_or(fail(Status::UnknownKind, "the checkpoint names an unknown scalar kind"))?,
            connectivity: raw.connectivity,
            values_per_key: raw.values_per_key,
            residency: Residency::from_raw(raw.residency)?,
        })
    }
}

// endregion: Options

// region: Per-Call Settings & Telemetry

/// Optional per-call settings, shared by insertion and both search forms.
///
/// A default derives every setting, which is what a null config means to the C interface - so
/// there is one spelling of "decide for me" rather than an `Option` around a type that already
/// has one. Extensible, so construct from [`Default`] and set what matters.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct ExpansionConfig {
    /// Candidate-pool breadth; zero derives it from the geometry.
    pub expansion_factor: usize,
}

impl ExpansionConfig {
    /// The settings a given candidate-pool breadth names, with everything else derived.
    pub fn with_expansion_factor(expansion_factor: usize) -> Self {
        ExpansionConfig { expansion_factor }
    }

    /// Lowers into the struct the C interface reads.
    fn lower(self) -> ffi::usearch_dense_expansion_config_t {
        ffi::usearch_dense_expansion_config_t {
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
pub struct AttentionConfig {
    /// Candidate-pool breadth; zero derives it from the geometry.
    pub expansion_factor: usize,
    /// The logit scale, conventionally the inverse square root of the head dimensionality;
    /// zero derives it from the index's frozen geometry.
    pub score_scale: f32,
    /// The residual-mass threshold the walk stops at - the mass-denominated sibling of the
    /// radius a search terminates on; zero derives.
    pub mass_tolerance: f32,
}

impl AttentionConfig {
    /// Lowers into the struct the C interface reads.
    fn lower(self) -> ffi::usearch_dense_attention_config_t {
        ffi::usearch_dense_attention_config_t {
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

/// Where a batched search writes its results - three caller-owned outputs, sized together.
///
/// The keys matrix decides how many matches each query may report: its width is the "k" in
/// "kANN", so the wanted count cannot disagree with the space reserved for it.
pub struct Outputs<'buffers> {
    /// Matrix of `queries × wanted` keys.
    pub keys: MatrixSpan<'buffers, Key>,
    /// Matrix of `queries × wanted` distances.
    pub distances: MatrixSpan<'buffers, Distance>,
    /// One match count per query.
    pub counts: &'buffers mut [usize],
}

impl<'buffers> Outputs<'buffers> {
    /// Wraps three packed slabs: `queries × wanted` keys and distances, one count per query.
    pub fn packed(
        keys: &'buffers mut [Key],
        distances: &'buffers mut [Distance],
        counts: &'buffers mut [usize],
        wanted: usize,
    ) -> Self {
        Outputs {
            keys: crate::types::rows_mut(keys, wanted),
            distances: crate::types::rows_mut(distances, wanted),
            counts,
        }
    }
}

// endregion: Results

// region: Dense Index

/// Vector Search Index over equidimensional dense representations.
///
/// ```no_run
/// use usearch::dense::{DenseIndex, ExpansionConfig, Metric, Options, Outputs};
/// use usearch::{rows, Distance, Executor, Key, Residency, ScalarKind};
///
/// let mut options = Options::default();
/// options.dimensions = 4;
/// options.metric = Metric::Angular;
/// options.quantization = ScalarKind::I8;
///
/// let index = DenseIndex::new(&options)?;
/// index.reserve(1_000)?;
/// let executor = Executor::new(&Residency::default(), 0)?;
///
/// let keys = [Key::from(0u64), Key::from(1u64)];
/// let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
/// index.add(&executor, &keys, rows(&slab, 4), ExpansionConfig::default(), None)?;
///
/// let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
/// let (mut found, mut scores, mut counts) = ([Key::NIL; 10], [0.0 as Distance; 10], [0usize; 1]);
/// let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 10);
/// index.search_into(&executor, rows(&query, 4), ExpansionConfig::default(), &mut outputs, None)?;
/// println!("{} matches", counts[0]);
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

    /// Reads a checkpoint's geometry and metric metadata without loading it; a manifest path
    /// answers from its summary alone, so no shard is opened.
    pub fn metadata(path: impl AsRef<Path>) -> Result<Options> {
        let path = path.as_ref().to_str().ok_or(fail(Status::Io, PATH_NOT_UTF8))?;
        let mut raw = unsafe { core::mem::zeroed() };
        let mut error = usearch_error_t::default();
        unsafe { ffi::usearch_dense_metadata(path.into(), &mut raw, &mut error) };
        check(error)?;
        Options::from_raw(&raw)
    }

    /// Reads geometry and metric metadata from a serialized monolith buffer.
    pub fn metadata_buffer(buffer: &[u8]) -> Result<Options> {
        let mut raw = unsafe { core::mem::zeroed() };
        let mut error = usearch_error_t::default();
        unsafe { ffi::usearch_dense_metadata_buffer(view_of(buffer), &mut raw, &mut error) };
        check(error)?;
        Options::from_raw(&raw)
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
        self.measure(ffi::usearch_dense_size)
    }

    /// The reserved capacity in entries.
    pub fn capacity(&self) -> Result<usize> {
        self.measure(ffi::usearch_dense_capacity)
    }

    /// The frozen dimensionality of the stored vectors, zero while the geometry is deferred.
    pub fn dimensions(&self) -> Result<usize> {
        self.measure(ffi::usearch_dense_dimensions)
    }

    /// The frozen connections-per-node bound of the graph.
    pub fn connectivity(&self) -> Result<usize> {
        self.measure(ffi::usearch_dense_connectivity)
    }

    /// The memory usage of the index in bytes.
    pub fn memory_usage(&self) -> Result<usize> {
        self.measure(ffi::usearch_dense_memory_usage)
    }

    /// The bytes one monolith of the index takes under the default layout - what
    /// [`DenseIndex::save_buffer`] needs.
    pub fn storage_requirements(&self) -> Result<usize> {
        self.measure(ffi::usearch_dense_storage_requirements)
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

    fn measure(
        &self,
        probe: unsafe extern "C" fn(ffi::usearch_dense_index_t, *mut usearch_error_t) -> usize,
    ) -> Result<usize> {
        let mut error = usearch_error_t::default();
        let measured = unsafe { probe(self.handle, &mut error) };
        check(error)?;
        Ok(measured)
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
        config: ExpansionConfig,
        stats: Option<&mut Stats>,
    ) -> Result<usize> {
        let view = ffi::usearch_dense_matrix_view_t::of(&vectors)?;
        if keys.len() != view.rows {
            return Err(fail(
                Status::ShapeMismatch,
                "the key count does not match the vector row count",
            ));
        }
        let config = config.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        let added = unsafe {
            ffi::usearch_dense_add(
                self.handle,
                executor.handle(),
                keys.as_ptr() as *const usearch_key_t,
                &view,
                &config,
                None,
                ptr::null_mut(),
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
    pub fn search_into<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Scalar>,
        config: ExpansionConfig,
        outputs: &mut Outputs<'_>,
        stats: Option<&mut Stats>,
    ) -> Result<()> {
        self.search_dispatch(executor, queries, config, outputs, stats, None, ptr::null_mut())
    }

    /// The filtered sibling of [`DenseIndex::search_into`], admitting only keys the predicate
    /// approves.
    ///
    /// Non-matching entries still route the traversal, so recall does not collapse at low
    /// selectivity. The predicate runs on the executor's threads, hence `Sync`.
    pub fn search_if_into<Scalar: crate::types::Scalar, Predicate: Fn(Key) -> bool + Sync>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Scalar>,
        config: ExpansionConfig,
        outputs: &mut Outputs<'_>,
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
            config,
            outputs,
            stats,
            Some(trampoline::<Predicate>),
            predicate as *const Predicate as *mut c_void,
        )
    }

    fn search_dispatch<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Scalar>,
        config: ExpansionConfig,
        outputs: &mut Outputs<'_>,
        stats: Option<&mut Stats>,
        predicate: crate::types::ffi::usearch_predicate_t,
        predicate_state: *mut c_void,
    ) -> Result<()> {
        let queries_view = ffi::usearch_dense_matrix_view_t::of(&queries)?;
        let matches = ffi::usearch_dense_matches_span_t::of(outputs, queries_view.rows)?;
        let config = config.lower();
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
                    &matches,
                    &config,
                    None,
                    ptr::null_mut(),
                    &mut raw_stats,
                    &mut error,
                )
            },
            admits => unsafe {
                ffi::usearch_dense_search_if(
                    self.handle,
                    executor.handle(),
                    &queries_view,
                    &matches,
                    &config,
                    admits,
                    predicate_state,
                    None,
                    ptr::null_mut(),
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
        let span = ffi::usearch_dense_matrix_span_t::of(vectors)?;
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
    pub fn refine(&self, executor: &Executor, config: ExpansionConfig, stats: Option<&mut Stats>) -> Result<usize> {
        let lowered = config.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        let changed = unsafe {
            ffi::usearch_dense_refine(
                self.handle,
                executor.handle(),
                &lowered,
                None,
                ptr::null_mut(),
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
    /// The linear-algebra sibling of [`DenseIndex::search_into`], reporting every pair the
    /// traversal scored rather than only the nearest, so `outputs.keys` names each pair's indexed
    /// row and `outputs.distances` carries their product. Pairs arrive in traversal order rather
    /// than sorted. Runs on an inner-product index alone, since a product is a logit.
    pub fn multiply_into<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        left: MatrixView<'_, Scalar>,
        config: ExpansionConfig,
        outputs: &mut Outputs<'_>,
        stats: Option<&mut Stats>,
    ) -> Result<()> {
        let view = ffi::usearch_dense_matrix_view_t::of(&left)?;
        let products = ffi::usearch_dense_matches_span_t::of(outputs, view.rows)?;
        let lowered = config.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_multiply(
                self.handle,
                executor.handle(),
                &view,
                &products,
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
    pub fn maxsim_into<Scalar: crate::types::Scalar>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Scalar>,
        rows_per_query: usize,
        config: ExpansionConfig,
        outputs: &mut Outputs<'_>,
        stats: Option<&mut Stats>,
    ) -> Result<()> {
        let view = ffi::usearch_dense_matrix_view_t::of(&queries)?;
        if rows_per_query == 0 || view.rows % rows_per_query != 0 {
            return Err(fail(
                Status::ShapeMismatch,
                "the query rows do not divide into whole queries",
            ));
        }
        let matches = ffi::usearch_dense_matches_span_t::of(outputs, view.rows / rows_per_query)?;
        let lowered = config.lower();
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_maxsim(
                self.handle,
                executor.handle(),
                &view,
                rows_per_query,
                &matches,
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
    pub fn attend_into<Query: crate::types::Scalar, Value: crate::types::Scalar>(
        &self,
        executor: &Executor,
        queries: MatrixView<'_, Query>,
        values: MatrixView<'_, Value>,
        config: AttentionConfig,
        outputs: &mut MatrixSpan<'_, Value>,
        carry: Option<&mut [SoftmaxState]>,
        stats: Option<&mut Stats>,
    ) -> Result<()> {
        let queries_view = ffi::usearch_dense_matrix_view_t::of(&queries)?;
        let values_view = ffi::usearch_dense_matrix_view_t::of(&values)?;
        let outputs_span = ffi::usearch_dense_matrix_span_t::of(outputs)?;
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
        let lowered = config.lower();
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
        // The array the C interface reads has to be contiguous, and the donors are consumed by
        // the call, so their Rust destructors are dropped once the handles are copied out.
        let mut handles: Vec<ffi::usearch_dense_index_t> = donors.iter().map(|donor| donor.handle).collect();
        donors.into_iter().for_each(core::mem::forget);
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_merge(
                self.handle,
                executor.handle(),
                handles.as_mut_ptr(),
                handles.len(),
                None,
                ptr::null_mut(),
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
        let view = ffi::usearch_dense_matrix_view_t::of_row(query);
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
        let path = path.as_ref().to_str().ok_or(fail(Status::Io, PATH_NOT_UTF8))?;
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_save(
                self.handle,
                executor.handle(),
                path.into(),
                None,
                ptr::null_mut(),
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
        let path = path.as_ref().to_str().ok_or(fail(Status::Io, PATH_NOT_UTF8))?;
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_load(
                self.handle,
                executor.handle(),
                path.into(),
                None,
                ptr::null_mut(),
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
            Some(matrix) => Some(ffi::usearch_dense_matrix_span_t::of(matrix)?),
        };
        let mut error = usearch_error_t::default();
        let written = unsafe {
            ffi::usearch_dense_cursor_next(
                self.cursor,
                keys.as_mut_ptr() as *mut usearch_key_t,
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
                keys.as_mut_ptr() as *mut usearch_key_t,
                distances.as_mut_ptr(),
                MATCHES_PER_PULL,
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

impl ffi::usearch_dense_matrix_view_t {
    /// Borrows a matrix as the input descriptor every batch verb reads.
    fn of<Scalar: crate::types::Scalar>(matrix: &MatrixView<'_, Scalar>) -> Result<Self> {
        let layout = MatrixLayout::of(matrix)?;
        Ok(ffi::usearch_dense_matrix_view_t {
            data: matrix.as_ptr() as *const c_void,
            rows: layout.rows,
            columns: layout.columns,
            stride_bytes: layout.stride_bytes,
            scalar_kind: Scalar::KIND as i32,
        })
    }

    /// The one-row descriptor a finder takes, whose width is the slab's whole length.
    fn of_row<Scalar: crate::types::Scalar>(row: &[Scalar]) -> Self {
        ffi::usearch_dense_matrix_view_t {
            data: row.as_ptr() as *const c_void,
            rows: 1,
            columns: crate::types::dimensions_of::<Scalar>(row.len()),
            stride_bytes: 0,
            scalar_kind: Scalar::KIND as i32,
        }
    }
}

impl ffi::usearch_dense_matrix_span_t {
    /// Lends a caller-owned matrix as the output descriptor a copying verb writes into.
    fn of<Scalar: crate::types::Scalar>(matrix: &mut MatrixSpan<'_, Scalar>) -> Result<Self> {
        let layout = MatrixLayout::of(matrix)?;
        Ok(ffi::usearch_dense_matrix_span_t {
            data: matrix.as_mut_ptr() as *mut c_void,
            capacity: layout.rows,
            columns: layout.columns,
            stride_bytes: layout.stride_bytes,
            scalar_kind: Scalar::KIND as i32,
        })
    }
}

impl ffi::usearch_dense_matches_span_t {
    /// Lends the three ranked outputs, refusing a set that cannot hold @p queries rows.
    fn of(outputs: &mut Outputs<'_>, queries: usize) -> Result<Self> {
        let keys = MatrixLayout::of(&outputs.keys)?;
        let distances = MatrixLayout::of(&outputs.distances)?;
        if distances.columns != keys.columns {
            return Err(fail(
                Status::ShapeMismatch,
                "the key and distance outputs disagree on how many matches a row holds",
            ));
        }
        if keys.rows < queries || distances.rows < queries || outputs.counts.len() < queries {
            return Err(fail(
                Status::ShapeMismatch,
                "the outputs hold fewer rows than there are queries",
            ));
        }
        Ok(ffi::usearch_dense_matches_span_t {
            keys: outputs.keys.as_mut_ptr() as *mut usearch_key_t,
            keys_stride_bytes: keys.stride_bytes,
            scores: outputs.distances.as_mut_ptr() as *mut usearch_distance_t,
            scores_stride_bytes: distances.stride_bytes,
            counts: outputs.counts.as_mut_ptr(),
            wanted: keys.columns,
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
    use crate::types::{rows, rows_mut};

    const SEED: u64 = 42;

    // region: Helpers

    /// Deterministic unit-length rows. Normalizing is not cosmetic: on the unit sphere inner,
    /// angular, and euclidean induce the same ranking, so one assertion covers all three.
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
        Executor::new(&Residency::default(), 0).expect("the executor must spawn")
    }

    fn empty_index(dimensions: usize, metric: Metric, quantization: ScalarKind) -> DenseIndex {
        let index = DenseIndex::new(&Options {
            dimensions,
            metric,
            quantization,
            ..Options::default()
        })
        .expect("the index must initialize");
        index.reserve(1024).expect("the reservation must succeed");
        index
    }

    fn populated_index(
        count: usize,
        dimensions: usize,
        metric: Metric,
        quantization: ScalarKind,
    ) -> (DenseIndex, Vec<Key>, Vec<f32>) {
        let index = empty_index(dimensions, metric, quantization);
        index.reserve(count).expect("the reservation must succeed");
        let keys = sequential_keys(count);
        let slab = seeded_rows(count, dimensions, SEED);
        let added = index
            .add(
                &executor(),
                &keys,
                rows(&slab, dimensions),
                ExpansionConfig::default(),
                None,
            )
            .expect("the batch must insert");
        assert_eq!(added, count);
        (index, keys, slab)
    }

    /// The three-buffer dance collapsed into one call - the shape most assertions want.
    fn top_one(index: &DenseIndex, query: &[f32], dimensions: usize) -> Option<Match> {
        let mut keys = [Key::NIL; 1];
        let mut distances = [0.0 as Distance; 1];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut keys, &mut distances, &mut counts, 1);
        index
            .search_into(
                &executor(),
                rows(query, dimensions),
                ExpansionConfig::default(),
                &mut outputs,
                None,
            )
            .expect("the search must run");
        match counts[0] {
            0 => None,
            _ => Some(Match {
                key: keys[0],
                distance: distances[0],
            }),
        }
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

    fn check_construction(quantization: ScalarKind) {
        let index = empty_index(64, Metric::Angular, quantization);
        assert_eq!(
            index.dimensions().expect("the geometry must report"),
            64,
            "{quantization:?}"
        );
        assert_eq!(index.size().expect("the size must report"), 0, "{quantization:?}");
        assert!(
            index.connectivity().expect("the connectivity must report") > 0,
            "{quantization:?}"
        );
    }

    fn check_metric_quantization(metric: Metric, quantization: ScalarKind) {
        const DIMENSIONS: usize = 64;
        let (index, keys, slab) = populated_index(32, DIMENSIONS, metric, quantization);
        let found = top_one(&index, &slab[..DIMENSIONS], DIMENSIONS).expect("a match must come back");
        assert_eq!(found.key, keys[0], "{metric:?} over {quantization:?}");
    }

    /// Packs one byte per element, LSB first, as NumKong's bit vectors are laid out.
    fn bit_rows(bytes: &[u8]) -> Vec<u1x8> {
        bytes.iter().map(|&byte| u1x8(byte)).collect()
    }

    // endregion: Helpers

    // region: Options Round Trip

    #[test]
    fn options_survive_the_raw_round_trip() {
        let mut path_views = Vec::new();
        let pinned = Options {
            dimensions: 768,
            metric: Metric::Euclidean,
            quantization: ScalarKind::I8,
            connectivity: 32,
            values_per_key: 4,
            residency: Residency::default(),
        };
        assert_eq!(
            Options::from_raw(&pinned.to_raw(&mut path_views)).expect("the tags must be known"),
            pinned
        );

        let derived = Options::default();
        assert_eq!(
            Options::from_raw(&derived.to_raw(&mut path_views)).expect("the tags must be known"),
            derived
        );
    }

    #[test]
    fn options_carry_the_placement_a_residency_names() {
        let mut path_views = Vec::new();

        let sockets = Options {
            residency: Residency::Ram {
                memory_domains: vec![0, 1],
            },
            ..Options::default()
        };
        let raw = sockets.to_raw(&mut path_views);
        assert_eq!(raw.residency, 0);
        assert_eq!(unsafe { raw.residency_options.ram.memory_domains_count }, 2);

        let devices = Options {
            residency: Residency::Vram {
                devices: vec![0, 1, 2, 3],
            },
            ..Options::default()
        };
        let raw = devices.to_raw(&mut path_views);
        assert_eq!(raw.residency, 1);
        assert_eq!(unsafe { raw.residency_options.vram.devices_count }, 4);

        // A checkpoint records where it was written, never where a reader must keep it, so the
        // placement comes back empty while the tag survives.
        assert_eq!(
            Options::from_raw(&raw).expect("the tags must be known").residency,
            Residency::Vram { devices: Vec::new() }
        );
    }

    #[test]
    fn options_reject_unknown_enum_tags() {
        let mut path_views = Vec::new();

        let mut raw = Options::default().to_raw(&mut path_views);
        raw.residency = 9;
        assert_eq!(
            Options::from_raw(&raw).unwrap_err().status,
            Status::UnknownKind,
            "an unknown residency must not be guessed"
        );

        let mut raw = Options::default().to_raw(&mut path_views);
        raw.metric = 99;
        assert_eq!(
            Options::from_raw(&raw).unwrap_err().status,
            Status::UnknownKind,
            "an unknown metric must not be guessed"
        );

        let mut raw = Options::default().to_raw(&mut path_views);
        raw.quantization = 77;
        assert_eq!(
            Options::from_raw(&raw).unwrap_err().status,
            Status::UnknownKind,
            "an unknown scalar kind must not be guessed"
        );
    }

    #[test]
    fn options_carry_no_metric_callback() {
        let mut path_views = Vec::new();
        let raw = Options::default().to_raw(&mut path_views);
        assert!(raw.metric_callback.is_none(), "this binding exposes no custom metric");
        assert!(raw.metric_callback_state.is_null());
    }

    // endregion: Options Round Trip

    // region: Construction & Geometry

    #[test]
    fn constructs_every_quantization() {
        check_construction(ScalarKind::F64);
        check_construction(ScalarKind::F32);
        check_construction(ScalarKind::BF16);
        check_construction(ScalarKind::F16);
        check_construction(ScalarKind::E5M2);
        check_construction(ScalarKind::E4M3);
        check_construction(ScalarKind::E3M2);
        check_construction(ScalarKind::E2M3);
        check_construction(ScalarKind::I8);
        check_construction(ScalarKind::I4x2);
        check_construction(ScalarKind::U8);
        check_construction(ScalarKind::U4x2);
        check_construction(ScalarKind::U1x8);
    }

    #[test]
    fn does_not_preallocate_members() {
        let index = DenseIndex::new(&Options {
            dimensions: 16,
            ..Options::default()
        })
        .expect("the index must initialize");
        assert_eq!(index.size().expect("the size must report"), 0);
        assert_eq!(index.capacity().expect("the capacity must report"), 0);

        index.reserve(1024).expect("the reservation must succeed");
        assert!(index.capacity().expect("the capacity must report") >= 1024);
        assert_eq!(index.size().expect("the size must report"), 0);
    }

    #[test]
    fn defers_geometry_to_a_checkpoint() {
        // A zeroed width defers the geometry to a later load, which reads it off the checkpoint -
        // never to the first batch, which has no width to freeze from that the index would trust.
        let index = DenseIndex::new(&Options {
            dimensions: 0,
            quantization: ScalarKind::F32,
            ..Options::default()
        })
        .expect("the index must initialize");
        assert_eq!(index.dimensions().expect("the geometry must report"), 0);

        index.reserve(4).expect("the reservation must succeed");
        let slab = seeded_rows(1, 12, SEED);
        assert_eq!(
            index
                .add(
                    &executor(),
                    &sequential_keys(1),
                    rows(&slab, 12),
                    ExpansionConfig::default(),
                    None
                )
                .unwrap_err()
                .status,
            Status::Uninitialized,
            "an unfrozen geometry admits no rows"
        );

        let written = populated_index(4, 12, Metric::Angular, ScalarKind::F32).0;
        let checkpoint = scratch_path("defers_geometry_to_a_checkpoint");
        written
            .save(&executor(), &checkpoint)
            .expect("the checkpoint must write");
        index.load(&executor(), &checkpoint).expect("the checkpoint must load");
        let _ = std::fs::remove_file(&checkpoint);
        assert_eq!(index.dimensions().expect("the geometry must report"), 12);
    }

    #[test]
    fn survives_box_and_arc_moves() {
        requires_send_sync::<DenseIndex>();
        const DIMENSIONS: usize = 8;
        let (index, keys, slab) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32);

        let boxed = Box::new(index);
        assert_eq!(
            top_one(&boxed, &slab[..DIMENSIONS], DIMENSIONS).map(|found| found.key),
            Some(keys[0])
        );

        let shared = std::sync::Arc::new(*boxed);
        assert_eq!(
            top_one(&shared, &slab[..DIMENSIONS], DIMENSIONS).map(|found| found.key),
            Some(keys[0])
        );
    }

    // endregion: Construction & Geometry

    // region: Ingest & Retrieval

    #[test]
    fn adds_and_gets_vectors() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let keys = sequential_keys(2);
        let slab: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(
            index
                .add(
                    &executor(),
                    &keys,
                    rows(&slab, DIMENSIONS),
                    ExpansionConfig::default(),
                    None
                )
                .expect("the batch must insert"),
            2
        );
        assert_eq!(index.size().expect("the size must report"), 2);

        let mut restored = [0.0f32; DIMENSIONS];
        let mut span = rows_mut(&mut restored, DIMENSIONS);
        assert_eq!(index.get(keys[1], &mut span).expect("the row must come back"), 1);
        assert_eq!(restored, [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn rejects_mismatched_key_and_row_counts() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        let failure = index
            .add(
                &executor(),
                &sequential_keys(3),
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .unwrap_err();
        assert_eq!(failure.status, Status::ShapeMismatch);
    }

    #[test]
    fn rejects_wrong_dimension_matrices() {
        const DIMENSIONS: usize = 8;
        let (index, keys, _) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let narrow = seeded_rows(1, DIMENSIONS - 1, SEED);

        let failure = index
            .add(
                &executor(),
                &sequential_keys(1),
                rows(&narrow, DIMENSIONS - 1),
                ExpansionConfig::default(),
                None,
            )
            .unwrap_err();
        assert_eq!(failure.status, Status::ShapeMismatch);

        let mut buffer = [0.0f32; DIMENSIONS - 1];
        let mut span = rows_mut(&mut buffer, DIMENSIONS - 1);
        assert_eq!(index.get(keys[0], &mut span).unwrap_err().status, Status::ShapeMismatch);

        let mut found = [Key::NIL; 1];
        let mut scores = [0.0 as Distance; 1];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 1);
        let failure = index
            .search_into(
                &executor(),
                rows(&narrow, DIMENSIONS - 1),
                ExpansionConfig::default(),
                &mut outputs,
                None,
            )
            .unwrap_err();
        assert_eq!(failure.status, Status::ShapeMismatch);

        assert_eq!(
            index.find(&executor(), &narrow[..]).unwrap_err().status,
            Status::ShapeMismatch,
            "the query width is checked too"
        );
    }

    #[test]
    fn rejects_non_contiguous_rows() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let slab = seeded_rows(2, DIMENSIONS * 2, SEED);
        let element = core::mem::size_of::<f32>() as isize;
        // A column stride of two elements interleaves the rows, which the boundary forbids.
        let interleaved: MatrixView<'_, f32> =
            unsafe { MatrixView::from_raw_parts(slab.as_ptr(), &[2, DIMENSIONS], &[element * 8, element * 2]) };
        let failure = index
            .add(
                &executor(),
                &sequential_keys(2),
                interleaved,
                ExpansionConfig::default(),
                None,
            )
            .unwrap_err();
        assert_eq!(
            failure.status,
            Status::UnsupportedLayout,
            "the ABI carries one row stride and no column stride"
        );
    }

    #[test]
    fn rejects_descending_row_pitch() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        let element = core::mem::size_of::<f32>() as isize;
        let descending: MatrixView<'_, f32> = unsafe {
            MatrixView::from_raw_parts(
                slab.as_ptr().add(DIMENSIONS),
                &[2, DIMENSIONS],
                &[-(element * DIMENSIONS as isize), element],
            )
        };
        let failure = index
            .add(
                &executor(),
                &sequential_keys(2),
                descending,
                ExpansionConfig::default(),
                None,
            )
            .unwrap_err();
        assert_eq!(failure.status, Status::UnsupportedLayout);
    }

    #[test]
    fn accepts_strided_row_pitch() {
        const DIMENSIONS: usize = 4;
        const PITCH: usize = 8;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
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
            index
                .add(&executor(), &keys, strided, ExpansionConfig::default(), None)
                .expect("the padded batch must insert"),
            2
        );

        let mut restored = [0.0f32; DIMENSIONS];
        let mut span = rows_mut(&mut restored, DIMENSIONS);
        assert_eq!(index.get(keys[1], &mut span).expect("the row must come back"), 1);
        for (left, right) in restored.iter().zip(&rows_source[DIMENSIONS..]) {
            assert!(close(*left as f64, *right as f64, 1e-6));
        }
    }

    #[test]
    fn converts_scalar_kinds_on_ingest() {
        const DIMENSIONS: usize = 16;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::BF16);
        let keys = sequential_keys(1);
        let slab = seeded_rows(1, DIMENSIONS, SEED);
        index
            .add(
                &executor(),
                &keys,
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("an f32 batch lands in a bf16 index");

        let mut wide = [0.0f64; DIMENSIONS];
        let mut span = rows_mut(&mut wide, DIMENSIONS);
        assert_eq!(index.get(keys[0], &mut span).expect("the row must come back"), 1);
        for (left, right) in wide.iter().zip(&slab) {
            assert!(close(*left, *right as f64, 1e-2), "bf16 keeps eight mantissa bits");
        }
    }

    #[test]
    fn counts_duplicate_keys() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let shared = Key::from(7u64);
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        index
            .add(
                &executor(),
                &[shared, shared],
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("two rows may share one key");

        assert_eq!(index.count(shared).expect("the count must report"), 2);
        assert!(index.contains(shared).expect("the membership must report"));
        assert!(!index.contains(Key::from(8u64)).expect("the membership must report"));

        let mut both = [0.0f32; DIMENSIONS * 2];
        let mut span = rows_mut(&mut both, DIMENSIONS);
        assert_eq!(index.get(shared, &mut span).expect("both rows must come back"), 2);
    }

    #[test]
    fn reports_the_true_extent_when_truncated() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let shared = Key::from(3u64);
        let slab = seeded_rows(3, DIMENSIONS, SEED);
        index
            .add(
                &executor(),
                &[shared; 3],
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("three rows may share one key");

        // A one-row span cannot hold them, and the count says so rather than reporting one.
        let mut cramped = [0.0f32; DIMENSIONS];
        let mut span = rows_mut(&mut cramped, DIMENSIONS);
        assert_eq!(index.get(shared, &mut span).expect("the extent must report"), 3);
        assert_eq!(
            index.count(shared).expect("the count must report"),
            3,
            "the probe agrees"
        );

        // Sized from that answer, the second call copies everything.
        let mut roomy = [0.0f32; DIMENSIONS * 3];
        let mut span = rows_mut(&mut roomy, DIMENSIONS);
        assert_eq!(index.get(shared, &mut span).expect("the extent must report"), 3);
        for (left, right) in roomy.iter().zip(&slab) {
            assert!(close(*left as f64, *right as f64, 1e-6));
        }
    }

    #[test]
    fn uses_128_bit_keys() {
        const DIMENSIONS: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let wide = Key::from(u128::MAX);
        let narrow = Key::from(9u64);
        let slab = seeded_rows(2, DIMENSIONS, SEED);
        index
            .add(
                &executor(),
                &[wide, narrow],
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("wide keys insert like any other");

        assert_eq!(index.count(wide).expect("the count must report"), 1);
        assert_eq!(wide.as_u64(), None, "a non-zero high half has no integer identity");
        assert_eq!(narrow.as_u64(), Some(9));
        assert_eq!(
            top_one(&index, &slab[..DIMENSIONS], DIMENSIONS).map(|found| found.key),
            Some(wide)
        );
    }

    // region: Refinement & Linear Algebra

    #[test]
    fn refinement_settles() {
        const DIMENSIONS: usize = 8;
        let (index, _, _) = populated_index(64, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let executor = executor();
        let mut stats = Stats::default();

        // The return is a change count, not a status, so a caller loops until the graph settles.
        let changed = index
            .refine(&executor, ExpansionConfig::with_expansion_factor(64), Some(&mut stats))
            .expect("the sweep must run");
        assert!(changed <= 64, "a sweep cannot change more nodes than the index holds");
        assert!(stats.visited_members > 0, "the sweep reports the nodes it swept");
    }

    #[test]
    fn multiplication_reports_every_scored_pair() {
        const DIMENSIONS: usize = 4;
        const ROWS: usize = 3;
        const WANTED: usize = 2;
        // A product is a logit, so the linear-algebra verbs take the inner kind alone.
        let index = empty_index(DIMENSIONS, Metric::Inner, ScalarKind::F32);
        let slab = seeded_rows(ROWS, DIMENSIONS, SEED);
        let keys = sequential_keys(ROWS);
        index
            .add(
                &executor(),
                &keys,
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("the rows must insert");

        let (mut columns, mut products, mut counts) = (
            vec![Key::NIL; ROWS * WANTED],
            vec![0.0; ROWS * WANTED],
            vec![0usize; ROWS],
        );
        let mut outputs = Outputs::packed(&mut columns, &mut products, &mut counts, WANTED);
        index
            .multiply_into(
                &executor(),
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                &mut outputs,
                None,
            )
            .expect("the product must resolve");

        // The count is the true number of scored pairs, which may exceed the column capacity.
        assert!(counts.iter().all(|&scored| scored > 0), "every row scores something");
        // A row against itself is its own squared norm, which is the largest product it can make.
        let self_product: f64 = slab[..DIMENSIONS]
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum();
        assert!(products[..counts[0].min(WANTED)]
            .iter()
            .any(|scored| close(*scored, self_product, 1e-3)));
    }

    #[test]
    fn late_interaction_sums_per_token_maxima() {
        const DIMENSIONS: usize = 4;
        const TOKENS: usize = 2;
        const DOCUMENTS: usize = 3;
        let mut options = Options::default();
        options.dimensions = DIMENSIONS;
        options.metric = Metric::Inner;
        options.quantization = ScalarKind::F32;
        // A document is one key holding its token rows, which is what `values_per_key` is for.
        options.values_per_key = TOKENS;
        let index = DenseIndex::new(&options).expect("the index must initialize");
        index.reserve(DOCUMENTS * TOKENS).expect("the reservation must succeed");

        let slab = seeded_rows(DOCUMENTS * TOKENS, DIMENSIONS, SEED);
        let mut keys = Vec::with_capacity(DOCUMENTS * TOKENS);
        for document in 0..DOCUMENTS {
            keys.extend(core::iter::repeat_n(Key::from(document as u64), TOKENS));
        }
        index
            .add(
                &executor(),
                &keys,
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("the token rows must insert");

        let (mut found, mut scores, mut counts) = (vec![Key::NIL; DOCUMENTS], vec![0.0; DOCUMENTS], vec![0usize; 1]);
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, DOCUMENTS);
        index
            .maxsim_into(
                &executor(),
                rows(&slab[..TOKENS * DIMENSIONS], DIMENSIONS),
                TOKENS,
                ExpansionConfig::default(),
                &mut outputs,
                None,
            )
            .expect("the late interaction must resolve");

        assert_eq!(counts[0], DOCUMENTS, "every document is scored");
        // The query is document zero's own tokens, so it sums the largest maxima of any document.
        assert_eq!(found[0], Key::from(0u64), "a document scores highest against itself");
        assert!(scores[0] >= scores[1], "summed maxima arrive best-first");
    }

    #[test]
    fn attention_weights_values_by_softmax() {
        const DIMENSIONS: usize = 4;
        const VALUE_WIDTH: usize = 2;
        const ENTRIES: usize = 4;
        let index = empty_index(DIMENSIONS, Metric::Inner, ScalarKind::F32);
        let slab = seeded_rows(ENTRIES, DIMENSIONS, SEED);
        // An entry's key is its row offset into the value table, so positions insert in order.
        let keys = sequential_keys(ENTRIES);
        index
            .add(
                &executor(),
                &keys,
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("the rows must insert");

        let values = seeded_rows(ENTRIES, VALUE_WIDTH, SEED + 1);
        let mut attended = vec![0.0f32; VALUE_WIDTH];
        let mut carry = vec![SoftmaxState::default(); 1];
        let mut outputs = rows_mut(&mut attended, VALUE_WIDTH);
        index
            .attend_into(
                &executor(),
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                rows(&values, VALUE_WIDTH),
                AttentionConfig::default(),
                &mut outputs,
                Some(&mut carry),
                None,
            )
            .expect("the attention must resolve");

        // The output is a convex combination of the value rows, so it lies within their range.
        for column in 0..VALUE_WIDTH {
            let column_values: Vec<f32> = (0..ENTRIES).map(|row| values[row * VALUE_WIDTH + column]).collect();
            let low = column_values.iter().cloned().fold(f32::INFINITY, f32::min);
            let high = column_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(attended[column] >= low - 1e-3 && attended[column] <= high + 1e-3);
        }
        assert!(carry[0].partition > 0.0, "the carry reports the softmax denominator");
    }

    #[test]
    fn linear_algebra_refuses_a_distance_kind() {
        const DIMENSIONS: usize = 4;
        // A product and a softmax both need logits, so a graph frozen under a distance refuses.
        let (index, _, slab) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let (mut columns, mut products, mut counts) = (vec![Key::NIL; 2], vec![0.0; 2], vec![0usize; 1]);
        let mut outputs = Outputs::packed(&mut columns, &mut products, &mut counts, 2);
        assert_eq!(
            index
                .multiply_into(
                    &executor(),
                    rows(&slab[..DIMENSIONS], DIMENSIONS),
                    ExpansionConfig::default(),
                    &mut outputs,
                    None,
                )
                .unwrap_err()
                .status,
            Status::ConfigMismatch
        );
    }

    // endregion: Refinement & Linear Algebra

    // endregion: Ingest & Retrieval

    // region: Search

    #[test]
    fn searches_an_empty_index() {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let query = seeded_rows(1, DIMENSIONS, SEED);
        assert_eq!(top_one(&index, &query, DIMENSIONS), None);
    }

    #[test]
    fn finds_each_row_as_its_own_nearest() {
        const DIMENSIONS: usize = 64;
        const QUERIES: usize = 8;
        const WANTED: usize = 5;
        let (index, keys, slab) = populated_index(256, DIMENSIONS, Metric::Angular, ScalarKind::F32);

        let mut found = vec![Key::NIL; QUERIES * WANTED];
        let mut scores = vec![0.0 as Distance; QUERIES * WANTED];
        let mut counts = vec![0usize; QUERIES];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, WANTED);
        index
            .search_into(
                &executor(),
                rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS),
                ExpansionConfig::with_expansion_factor(64),
                &mut outputs,
                None,
            )
            .expect("the batch must resolve");

        for query in 0..QUERIES {
            assert_eq!(counts[query], WANTED);
            assert_eq!(found[query * WANTED], keys[query], "row {query} is its own nearest");
            assert!(scores[query * WANTED] <= scores[query * WANTED + 1]);
        }
    }

    #[test]
    fn returns_zero_distances_for_identical_rows() {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let row = [0.5f32; DIMENSIONS];
        let slab: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        index
            .add(
                &executor(),
                &sequential_keys(2),
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("identical rows insert");

        let found = top_one(&index, &row, DIMENSIONS).expect("a match must come back");
        assert!(close(found.distance, 0.0, 1e-6));
    }

    #[test]
    fn rejects_undersized_outputs() {
        const DIMENSIONS: usize = 8;
        const QUERIES: usize = 4;
        let (index, _, slab) = populated_index(16, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let executor = executor();

        let mut found = vec![Key::NIL; (QUERIES - 1) * 2];
        let mut scores = vec![0.0 as Distance; (QUERIES - 1) * 2];
        let mut counts = vec![0usize; QUERIES];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 2);
        assert_eq!(
            index
                .search_into(
                    &executor,
                    rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS),
                    ExpansionConfig::default(),
                    &mut outputs,
                    None,
                )
                .unwrap_err()
                .status,
            Status::ShapeMismatch,
            "too few output rows"
        );

        let mut found = vec![Key::NIL; QUERIES * 2];
        let mut scores = vec![0.0 as Distance; QUERIES * 2];
        let mut counts = vec![0usize; QUERIES - 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 2);
        assert_eq!(
            index
                .search_into(
                    &executor,
                    rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS),
                    ExpansionConfig::default(),
                    &mut outputs,
                    None,
                )
                .unwrap_err()
                .status,
            Status::ShapeMismatch,
            "too few counts"
        );
    }

    #[test]
    fn reports_traversal_stats() {
        const DIMENSIONS: usize = 32;
        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let keys = sequential_keys(64);
        let slab = seeded_rows(64, DIMENSIONS, SEED);

        let mut inserting = Stats::default();
        index
            .add(
                &executor(),
                &keys,
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                Some(&mut inserting),
            )
            .expect("the batch must insert");
        assert!(inserting.computed_distances > 0);

        let mut searching = Stats::default();
        let mut found = [Key::NIL; 4];
        let mut scores = [0.0 as Distance; 4];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 4);
        index
            .search_into(
                &executor(),
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                ExpansionConfig::default(),
                &mut outputs,
                Some(&mut searching),
            )
            .expect("the search must run");
        let first = searching;
        assert!(first.computed_distances > 0);

        index
            .search_into(
                &executor(),
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                ExpansionConfig::default(),
                &mut outputs,
                Some(&mut searching),
            )
            .expect("the search must run");
        assert_eq!(
            searching, first,
            "stats are overwritten by their call, never accumulated"
        );

        // Accumulating is the caller's job, and the operator exists for exactly that.
        let mut total = first;
        total += first;
        assert_eq!(total.computed_distances, first.computed_distances * 2);
    }

    #[test]
    fn filters_with_a_stateless_predicate() {
        const DIMENSIONS: usize = 16;
        const WANTED: usize = 8;
        let (index, _, slab) = populated_index(64, DIMENSIONS, Metric::Angular, ScalarKind::F32);

        let is_even = |key: Key| key.as_u64().is_some_and(|value| value % 2 == 0);
        let mut found = [Key::NIL; WANTED];
        let mut scores = [0.0 as Distance; WANTED];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, WANTED);
        index
            .search_if_into(
                &executor(),
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                ExpansionConfig::with_expansion_factor(64),
                &mut outputs,
                None,
                &is_even,
            )
            .expect("the filtered search must run");

        let admitted = counts[0];
        assert!(admitted > 0);
        for key in &found[..admitted] {
            assert!(is_even(*key), "{key} was not admitted by the predicate");
        }
    }

    #[test]
    fn filters_with_a_stateful_predicate() {
        const DIMENSIONS: usize = 16;
        const WANTED: usize = 8;
        let (index, keys, slab) = populated_index(64, DIMENSIONS, Metric::Angular, ScalarKind::F32);

        let allowed: HashSet<Key> = keys.iter().copied().take(5).collect();
        let is_allowed = |key: Key| allowed.contains(&key);
        let mut found = [Key::NIL; WANTED];
        let mut scores = [0.0 as Distance; WANTED];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, WANTED);
        index
            .search_if_into(
                &executor(),
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                ExpansionConfig::with_expansion_factor(64),
                &mut outputs,
                None,
                &is_allowed,
            )
            .expect("the filtered search must run");

        let admitted = counts[0];
        for key in &found[..admitted] {
            assert!(allowed.contains(key), "{key} is outside the captured set");
        }
    }

    #[test]
    fn filters_everything_out() {
        const DIMENSIONS: usize = 16;
        let (index, _, slab) = populated_index(64, DIMENSIONS, Metric::Angular, ScalarKind::F32);

        let admits_nothing = |_: Key| false;
        let mut found = [Key::NIL; 4];
        let mut scores = [0.0 as Distance; 4];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 4);
        index
            .search_if_into(
                &executor(),
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                ExpansionConfig::with_expansion_factor(64),
                &mut outputs,
                None,
                &admits_nothing,
            )
            .expect("a fully-rejecting predicate is not an error");
        assert_eq!(counts[0], 0);
    }

    // endregion: Search

    // region: Metrics

    #[test]
    fn searches_every_quantization_at_angular() {
        check_metric_quantization(Metric::Angular, ScalarKind::F64);
        check_metric_quantization(Metric::Angular, ScalarKind::F32);
        check_metric_quantization(Metric::Angular, ScalarKind::BF16);
        check_metric_quantization(Metric::Angular, ScalarKind::F16);
        check_metric_quantization(Metric::Angular, ScalarKind::E5M2);
        check_metric_quantization(Metric::Angular, ScalarKind::E4M3);
        check_metric_quantization(Metric::Angular, ScalarKind::I8);
        check_metric_quantization(Metric::Angular, ScalarKind::U8);
    }

    #[test]
    fn searches_every_metric_at_four_quantizations() {
        check_metric_quantization(Metric::Inner, ScalarKind::F32);
        check_metric_quantization(Metric::Inner, ScalarKind::BF16);
        check_metric_quantization(Metric::Inner, ScalarKind::I8);
        check_metric_quantization(Metric::Inner, ScalarKind::U8);
        check_metric_quantization(Metric::Angular, ScalarKind::F32);
        check_metric_quantization(Metric::Angular, ScalarKind::BF16);
        check_metric_quantization(Metric::Angular, ScalarKind::I8);
        check_metric_quantization(Metric::Angular, ScalarKind::U8);
        check_metric_quantization(Metric::Euclidean, ScalarKind::F32);
        check_metric_quantization(Metric::Euclidean, ScalarKind::BF16);
        check_metric_quantization(Metric::Euclidean, ScalarKind::I8);
        check_metric_quantization(Metric::Euclidean, ScalarKind::U8);
    }

    #[test]
    fn measures_hamming_over_packed_bits() {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, Metric::Hamming, ScalarKind::U1x8);
        let stored = bit_rows(&[0b0000_1111, 0b1111_0000]);
        let keys = [Key::from(42u64), Key::from(43u64)];
        index
            .add(
                &executor(),
                &keys,
                rows(&stored, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("bit rows insert");

        let query = bit_rows(&[0b0111_1000]);
        let mut found = [Key::NIL; 2];
        let mut scores = [0.0 as Distance; 2];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 2);
        index
            .search_into(
                &executor(),
                rows(&query, DIMENSIONS),
                ExpansionConfig::default(),
                &mut outputs,
                None,
            )
            .expect("the search must run");

        assert_eq!(counts[0], 2);
        assert_eq!(found[0], keys[1]);
        assert!(close(scores[0], 2.0, 1e-9), "two bits differ from 0b11110000");
        assert_eq!(found[1], keys[0]);
        assert!(close(scores[1], 6.0, 1e-9), "six bits differ from 0b00001111");
    }

    #[test]
    fn measures_jaccard_over_packed_bits() {
        const DIMENSIONS: usize = 8;
        let index = empty_index(DIMENSIONS, Metric::Jaccard, ScalarKind::U1x8);
        let stored = bit_rows(&[0b0000_1111]);
        let keys = [Key::from(1u64)];
        index
            .add(
                &executor(),
                &keys,
                rows(&stored, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("bit rows insert");

        // 0b00001111 against 0b00000111: three bits shared, four in the union.
        let query = bit_rows(&[0b0000_0111]);
        let found = top_one_bits(&index, &query, DIMENSIONS).expect("a match must come back");
        assert!(close(found.distance, 1.0 - 3.0 / 4.0, 1e-6));
    }

    /// The bit-vector sibling of [`top_one`], which is written against `f32` queries.
    fn top_one_bits(index: &DenseIndex, query: &[u1x8], dimensions: usize) -> Option<Match> {
        let mut keys = [Key::NIL; 1];
        let mut distances = [0.0 as Distance; 1];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut keys, &mut distances, &mut counts, 1);
        index
            .search_into(
                &executor(),
                rows(query, dimensions),
                ExpansionConfig::default(),
                &mut outputs,
                None,
            )
            .expect("the search must run");
        match counts[0] {
            0 => None,
            _ => Some(Match {
                key: keys[0],
                distance: distances[0],
            }),
        }
    }

    // endregion: Metrics

    // region: Serialization

    #[test]
    fn saves_and_loads_a_file() {
        const DIMENSIONS: usize = 16;
        let path = scratch_path("saves_and_loads_a_file");
        let (index, keys, slab) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        index.save(&executor(), &path).expect("the checkpoint must write");

        // A derived geometry freezes from the checkpoint's own headers.
        let restored = DenseIndex::new(&Options {
            dimensions: 0,
            quantization: ScalarKind::F32,
            ..Options::default()
        })
        .expect("the index must initialize");
        restored.load(&executor(), &path).expect("the checkpoint must load");
        assert_eq!(restored.dimensions().expect("the geometry must report"), DIMENSIONS);
        assert_eq!(restored.size().expect("the size must report"), 32);
        assert_eq!(
            top_one(&restored, &slab[..DIMENSIONS], DIMENSIONS).map(|found| found.key),
            Some(keys[0])
        );
        remove_quietly(&path);
    }

    #[test]
    fn saves_and_loads_a_buffer() {
        const DIMENSIONS: usize = 16;
        let (index, keys, slab) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let length = index.storage_requirements().expect("the length must report");
        assert!(length > 0);

        let mut buffer = vec![0u8; length];
        index
            .save_buffer(&executor(), &mut buffer)
            .expect("the monolith must write");

        let restored = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        restored
            .load_buffer(&executor(), &buffer)
            .expect("the monolith must load");
        assert_eq!(
            restored.size().expect("the size must report"),
            index.size().expect("the size must report")
        );
        assert_eq!(
            top_one(&restored, &slab[..DIMENSIONS], DIMENSIONS).map(|found| found.key),
            Some(keys[0])
        );

        let truncated = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        assert!(
            truncated.load_buffer(&executor(), &buffer[..length - 1]).is_err(),
            "a short buffer must not be read"
        );
    }

    #[test]
    fn reads_metadata_without_loading() {
        const DIMENSIONS: usize = 16;
        let path = scratch_path("reads_metadata_without_loading");
        let (index, _, _) = populated_index(8, DIMENSIONS, Metric::Euclidean, ScalarKind::I8);
        index.save(&executor(), &path).expect("the checkpoint must write");

        let described = DenseIndex::metadata(&path).expect("the metadata must read");
        assert_eq!(described.dimensions, DIMENSIONS);
        assert_eq!(described.metric, Metric::Euclidean);
        assert_eq!(described.quantization, ScalarKind::I8);

        let mut buffer = vec![0u8; index.storage_requirements().expect("the length must report")];
        index
            .save_buffer(&executor(), &mut buffer)
            .expect("the monolith must write");
        let from_buffer = DenseIndex::metadata_buffer(&buffer).expect("the metadata must read");
        assert_eq!(from_buffer.dimensions, DIMENSIONS);
        remove_quietly(&path);
    }

    #[test]
    #[cfg(unix)]
    fn rejects_a_non_utf8_path() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let index = empty_index(8, Metric::Angular, ScalarKind::F32);
        let path = std::path::Path::new(OsStr::from_bytes(&[0xff, 0xfe]));
        assert_eq!(
            index.save(&executor(), path).unwrap_err().status,
            Status::Io,
            "the C interface exchanges UTF-8 bytes"
        );
    }

    #[test]
    fn clears_and_reuses() {
        const DIMENSIONS: usize = 8;
        let (index, keys, slab) = populated_index(16, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let reserved = index.capacity().expect("the capacity must report");

        index.clear().expect("the index must clear");
        assert_eq!(index.size().expect("the size must report"), 0);
        assert_eq!(index.capacity().expect("the capacity must report"), reserved);
        assert_eq!(index.dimensions().expect("the geometry must report"), DIMENSIONS);

        index
            .add(
                &executor(),
                &keys,
                rows(&slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("a cleared index takes the batch again");
        assert_eq!(index.size().expect("the size must report"), 16);
    }

    // endregion: Serialization

    // region: Merge, Cursor & Finder

    #[test]
    fn merges_a_donor_index() {
        const DIMENSIONS: usize = 16;
        let (index, keys, slab) = populated_index(32, DIMENSIONS, Metric::Angular, ScalarKind::F32);

        let donor = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let donor_keys: Vec<Key> = (100..132u64).map(Key::from).collect();
        let donor_slab = seeded_rows(32, DIMENSIONS, SEED + 1);
        donor
            .add(
                &executor(),
                &donor_keys,
                rows(&donor_slab, DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("the donor must fill");

        index
            .merge(&executor(), vec![donor])
            .expect("the donor must be absorbed");
        assert_eq!(index.size().expect("the size must report"), 64);
        assert_eq!(
            top_one(&index, &slab[..DIMENSIONS], DIMENSIONS).map(|found| found.key),
            Some(keys[0])
        );
        assert_eq!(
            top_one(&index, &donor_slab[..DIMENSIONS], DIMENSIONS).map(|found| found.key),
            Some(donor_keys[0])
        );
    }

    #[test]
    fn walks_entries_in_batches() {
        const DIMENSIONS: usize = 8;
        const COUNT: usize = 20;
        const BUFFERED: usize = 8;
        let (index, keys, _) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32);

        let mut cursor = index.cursor::<f32>().expect("the cursor must open");
        let mut seen = HashSet::new();
        let mut batches = Vec::new();
        let mut batch_keys = [Key::NIL; BUFFERED];
        let mut batch_vectors = vec![0.0f32; BUFFERED * DIMENSIONS];
        loop {
            let mut span = rows_mut(&mut batch_vectors, DIMENSIONS);
            let written = cursor
                .next_batch(&mut batch_keys, Some(&mut span))
                .expect("the cursor must advance");
            if written == 0 {
                break;
            }
            batches.push(written);
            seen.extend(batch_keys[..written].iter().copied());
        }

        assert_eq!(batches, vec![8, 8, 4], "the batch boundary is crossed twice");
        assert_eq!(seen, keys.iter().copied().collect::<HashSet<Key>>());
    }

    #[test]
    fn walks_entries_without_vectors() {
        const DIMENSIONS: usize = 8;
        let (index, keys, _) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32);

        let mut cursor = index.cursor::<f32>().expect("the cursor must open");
        let mut batch_keys = [Key::NIL; 8];
        let written = cursor
            .next_batch(&mut batch_keys, None)
            .expect("the key column alone walks");
        assert_eq!(written, 4);
        assert_eq!(
            batch_keys[..written].iter().copied().collect::<HashSet<Key>>(),
            keys.iter().copied().collect::<HashSet<Key>>()
        );
    }

    #[test]
    fn streams_matches_from_a_finder() {
        const DIMENSIONS: usize = 16;
        // Past the finder's internal refill boundary, so the batch seam is exercised.
        const COUNT: usize = 200;
        let (index, keys, slab) = populated_index(COUNT, DIMENSIONS, Metric::Angular, ScalarKind::F32);

        let executor = executor();
        let mut finder = index
            .find(&executor, &slab[..DIMENSIONS])
            .expect("the finder must open");
        let first: Vec<Match> = finder
            .by_ref()
            .take(5)
            .map(|found| found.expect("the stream must not fault"))
            .collect();
        assert_eq!(first[0].key, keys[0], "the query's own row comes first");
        for pair in first.windows(2) {
            assert!(pair[0].distance <= pair[1].distance, "matches arrive nearest first");
        }

        let drained = finder.count();
        assert!(drained > MATCHES_PER_PULL, "the stream crosses its refill boundary");

        let mut exhausted = index
            .find(&executor, &slab[..DIMENSIONS])
            .expect("the finder must open");
        while exhausted.next().is_some() {}
        assert!(exhausted.next().is_none(), "a drained finder stays drained");
    }

    // endregion: Merge, Cursor & Finder

    // region: Concurrency

    const THREADS: usize = 4;

    #[test]
    fn builds_and_searches_on_a_shared_executor() {
        const DIMENSIONS: usize = 128;
        const COUNT: usize = 1000;
        const WANTED: usize = 10;

        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        index.reserve(COUNT).expect("the reservation must succeed");
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        // The batch is the unit of parallelism: the executor fans one call out over its threads,
        // so the caller never spawns any of its own.
        let executor = Executor::new(&Residency::default(), THREADS).expect("the executor must spawn");
        assert_eq!(
            index
                .add(
                    &executor,
                    &keys,
                    rows(&slab, DIMENSIONS),
                    ExpansionConfig::default(),
                    None
                )
                .expect("the batch must insert"),
            COUNT
        );
        assert_eq!(index.size().expect("the size must report"), COUNT);

        let mut found = vec![Key::NIL; COUNT * WANTED];
        let mut scores = vec![0.0 as Distance; COUNT * WANTED];
        let mut counts = vec![0usize; COUNT];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, WANTED);
        index
            .search_into(
                &executor,
                rows(&slab, DIMENSIONS),
                ExpansionConfig::with_expansion_factor(64),
                &mut outputs,
                None,
            )
            .expect("the batch must resolve");
        for row in 0..COUNT {
            assert_ne!(counts[row], 0, "every query reports a match");
            assert_eq!(found[row * WANTED], keys[row], "every row is its own nearest");
        }
    }

    #[test]
    fn searches_while_inserting() {
        const DIMENSIONS: usize = 64;
        const HALF: usize = 500;
        const WANTED: usize = 4;

        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        index.reserve(2 * HALF).expect("the reservation must succeed");
        let keys = sequential_keys(2 * HALF);
        let slab = seeded_rows(2 * HALF, DIMENSIONS, SEED);

        // The first half lands serially, so the reader below traverses a populated graph.
        let seeding = executor();
        index
            .add(
                &seeding,
                &keys[..HALF],
                rows(&slab[..HALF * DIMENSIONS], DIMENSIONS),
                ExpansionConfig::default(),
                None,
            )
            .expect("the seed batch must insert");
        drop(seeding);

        // An executor admits one driver at a time, so genuinely overlapping calls need one
        // apiece - the writer's and the reader's, on two threads sharing the handle.
        let writer = Executor::new(&Residency::default(), THREADS).expect("the executor must spawn");
        let reader = Executor::new(&Residency::default(), THREADS).expect("the executor must spawn");
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
                        ExpansionConfig::default(),
                        None,
                    )
                    .expect("the racing batch must insert");
            });
            scope.spawn(|| {
                let reader = reader;
                let mut found = vec![Key::NIL; HALF * WANTED];
                let mut scores = vec![0.0 as Distance; HALF * WANTED];
                let mut counts = vec![0usize; HALF];
                let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, WANTED);
                index
                    .search_into(
                        &reader,
                        rows(&slab[..HALF * DIMENSIONS], DIMENSIONS),
                        ExpansionConfig::with_expansion_factor(32),
                        &mut outputs,
                        None,
                    )
                    .expect("the racing search must resolve");
                // Whether a racing insert is visible is what the watermark leaves unspecified;
                // that no key outside the dataset appears is what it guarantees.
                for row in 0..HALF {
                    if found[row * WANTED..row * WANTED + counts[row]]
                        .iter()
                        .any(|key| !known.contains(key))
                    {
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
        assert_eq!(index.size().expect("the size must report"), 2 * HALF);

        for row in (0..2 * HALF).step_by(64) {
            let query = &slab[row * DIMENSIONS..(row + 1) * DIMENSIONS];
            assert_eq!(
                top_one(&index, query, DIMENSIONS).map(|found| found.key),
                Some(keys[row])
            );
        }
    }

    // endregion: Concurrency
}

// endregion: Tests
