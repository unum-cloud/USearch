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
use core::mem::ManuallyDrop;
use core::ptr;
use std::path::Path;

use numkong::{MatrixSpan, MatrixView, StorageElement, TensorRef};

use crate::types::ffi::{usearch_distance_t, usearch_error_t, usearch_key_t};
use crate::types::{check, fail, static_string, Distance, Key, Residency, Result, ScalarKind, Status, PATH_NOT_UTF8};

// region: Raw Declarations

/// One-to-one mirror of `include/usearch/dense_index.h`.
mod ffi {
    #![allow(non_camel_case_types)]

    use core::ffi::{c_char, c_void};

    pub use crate::types::ffi::{
        usearch_distance_t, usearch_enum_t, usearch_error_t, usearch_key_t, usearch_predicate_t,
        usearch_residency_options_t, usearch_string_t,
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
        pub pad_dimensions: bool,
        pub connectivity: usize,
        pub residency: usearch_enum_t,
        pub residency_options: usearch_residency_options_t,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_dense_expansion_config_t {
        pub expansion_factor: usize,
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
            keys: *const usearch_key_t,
            vectors: *const c_void,
            count: usize,
            stride: usize,
            scalar_kind: usearch_enum_t,
            config: *const usearch_dense_expansion_config_t,
            stats: *mut usearch_dense_traversal_stats_t,
            error: *mut usearch_error_t,
        ) -> usize;

        pub fn usearch_dense_search(
            index: usearch_dense_index_t,
            queries: *const c_void,
            queries_count: usize,
            queries_stride: usize,
            scalar_kind: usearch_enum_t,
            wanted: usize,
            config: *const usearch_dense_expansion_config_t,
            stats: *mut usearch_dense_traversal_stats_t,
            keys: *mut usearch_key_t,
            keys_stride: usize,
            distances: *mut usearch_distance_t,
            distances_stride: usize,
            counts: *mut usize,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_search_if(
            index: usearch_dense_index_t,
            queries: *const c_void,
            queries_count: usize,
            queries_stride: usize,
            scalar_kind: usearch_enum_t,
            wanted: usize,
            config: *const usearch_dense_expansion_config_t,
            stats: *mut usearch_dense_traversal_stats_t,
            predicate: usearch_predicate_t,
            predicate_state: *mut c_void,
            keys: *mut usearch_key_t,
            keys_stride: usize,
            distances: *mut usearch_distance_t,
            distances_stride: usize,
            counts: *mut usize,
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
            vectors: *mut c_void,
            wanted: usize,
            stride: usize,
            scalar_kind: usearch_enum_t,
            error: *mut usearch_error_t,
        ) -> usize;

        pub fn usearch_dense_merge(
            index: usearch_dense_index_t,
            donor: usearch_dense_index_t,
            error: *mut usearch_error_t,
        );

        pub fn usearch_dense_cursor(
            index: usearch_dense_index_t,
            error: *mut usearch_error_t,
        ) -> usearch_dense_cursor_t;
        pub fn usearch_dense_cursor_next(
            cursor: usearch_dense_cursor_t,
            keys: *mut usearch_key_t,
            vectors: *mut c_void,
            wanted: usize,
            stride: usize,
            scalar_kind: usearch_enum_t,
            error: *mut usearch_error_t,
        ) -> usize;
        pub fn usearch_dense_cursor_free(cursor: usearch_dense_cursor_t);

        pub fn usearch_dense_find(
            index: usearch_dense_index_t,
            query: *const c_void,
            scalar_kind: usearch_enum_t,
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

        pub fn usearch_dense_serialized_length(index: usearch_dense_index_t, error: *mut usearch_error_t) -> usize;
        pub fn usearch_dense_save(index: usearch_dense_index_t, path: usearch_string_t, error: *mut usearch_error_t);
        pub fn usearch_dense_load(index: usearch_dense_index_t, path: usearch_string_t, error: *mut usearch_error_t);
        pub fn usearch_dense_metadata(
            path: usearch_string_t,
            options: *mut usearch_dense_options_t,
            error: *mut usearch_error_t,
        );
        pub fn usearch_dense_save_buffer(
            index: usearch_dense_index_t,
            buffer: *mut c_void,
            length: usize,
            error: *mut usearch_error_t,
        );
        pub fn usearch_dense_load_buffer(
            index: usearch_dense_index_t,
            buffer: *const c_void,
            length: usize,
            error: *mut usearch_error_t,
        );
        pub fn usearch_dense_metadata_buffer(
            buffer: *const c_void,
            length: usize,
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

/// Whether stored rows are rounded up to the SIMD step quantum.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum DimensionPadding {
    /// Rows keep their natural width.
    #[default]
    Natural,
    /// Cache-line-aligned rows and tail-free unrolled kernels, at slightly larger arenas.
    SimdQuantum,
}

/// Construction and loading options of a dense index.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Options {
    /// The number of indexed dimensions; zero defers the geometry to a later load, which
    /// reads it off the checkpoint.
    pub dimensions: usize,
    /// Whether stored rows round up to the SIMD step quantum - a property of the layout, so
    /// it applies whichever residency traverses it.
    pub dimension_padding: DimensionPadding,
    /// The metric kind used for distance calculations.
    pub metric: Metric,
    /// The scalar type vectors are stored and quantized into.
    pub quantization: ScalarKind,
    /// The connections-per-node bound of the graph; zero derives.
    pub connectivity: usize,
    /// Where the payload lives, and what loading a checkpoint means.
    pub residency: Residency,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            dimensions: 0,
            dimension_padding: DimensionPadding::Natural,
            metric: Metric::Angular,
            quantization: ScalarKind::BF16,
            connectivity: 0,
            residency: Residency::default(),
        }
    }
}

impl Options {
    fn to_raw(self) -> ffi::usearch_dense_options_t {
        let (residency, residency_options) = self.residency.to_raw();
        ffi::usearch_dense_options_t {
            metric: self.metric as i32,
            metric_callback: None,
            metric_callback_state: ptr::null_mut(),
            quantization: self.quantization as i32,
            dimensions: self.dimensions,
            pad_dimensions: self.dimension_padding == DimensionPadding::SimdQuantum,
            connectivity: self.connectivity,
            residency,
            residency_options,
        }
    }

    fn from_raw(raw: &ffi::usearch_dense_options_t) -> Result<Self> {
        Ok(Options {
            dimensions: raw.dimensions,
            dimension_padding: match raw.pad_dimensions {
                true => DimensionPadding::SimdQuantum,
                false => DimensionPadding::Natural,
            },
            metric: Metric::from_raw(raw.metric)
                .ok_or(fail(Status::UnknownKind, "the checkpoint names an unknown metric"))?,
            quantization: ScalarKind::from_raw(raw.quantization)
                .ok_or(fail(Status::UnknownKind, "the checkpoint names an unknown scalar kind"))?,
            connectivity: raw.connectivity,
            residency: unsafe { Residency::from_raw(raw.residency, &raw.residency_options) }?,
        })
    }
}

// endregion: Options

// region: Per-Call Settings & Telemetry

/// Traversal telemetry, fully overwritten by the call it is passed to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
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
#[derive(Debug)]
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

// region: Index

/// Vector Search Index over equidimensional dense representations.
///
/// ```no_run
/// use usearch::dense::{Index, Metric, Options, Outputs};
/// use usearch::{rows, Distance, Key, ScalarKind};
///
/// let index = Index::new(&Options {
///     dimensions: 4,
///     metric: Metric::Angular,
///     quantization: ScalarKind::I8,
///     ..Options::default()
/// })?;
/// index.reserve(1_000)?;
///
/// let keys = [Key::from(0u64), Key::from(1u64)];
/// let slab: [f32; 8] = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2];
/// index.add(&keys, rows(&slab, 4), 0, None)?;
///
/// let query: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
/// let (mut found, mut scores, mut counts) = ([Key::NIL; 10], [0.0 as Distance; 10], [0usize; 1]);
/// let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 10);
/// index.search_into(rows(&query, 4), 0, &mut outputs, None)?;
/// println!("{} matches", counts[0]);
/// # Ok::<(), usearch::Error>(())
/// ```
#[derive(Debug)]
pub struct Index {
    handle: ffi::usearch_dense_index_t,
}

// Safety: the library documents a handle as shareable across threads - searches never write to
// the graph, and insertions synchronize through per-node locks and publish in order.
unsafe impl Send for Index {}
unsafe impl Sync for Index {}

impl Drop for Index {
    fn drop(&mut self) { unsafe { ffi::usearch_dense_free(self.handle) } }
}

impl Index {
    /// Initializes a new index for the residency the options select.
    pub fn new(options: &Options) -> Result<Self> {
        let raw = options.to_raw();
        let mut error = usearch_error_t::default();
        let handle = unsafe { ffi::usearch_dense_init(&raw, &mut error) };
        check(error)?;
        Ok(Index { handle: owned(handle)? })
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
        unsafe {
            ffi::usearch_dense_metadata_buffer(buffer.as_ptr() as *const c_void, buffer.len(), &mut raw, &mut error)
        };
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
    pub fn size(&self) -> Result<usize> { self.measure(ffi::usearch_dense_size) }

    /// The reserved capacity in entries.
    pub fn capacity(&self) -> Result<usize> { self.measure(ffi::usearch_dense_capacity) }

    /// The frozen dimensionality of the stored vectors, zero while the geometry is deferred.
    pub fn dimensions(&self) -> Result<usize> { self.measure(ffi::usearch_dense_dimensions) }

    /// The frozen connections-per-node bound of the graph.
    pub fn connectivity(&self) -> Result<usize> { self.measure(ffi::usearch_dense_connectivity) }

    /// The memory usage of the index in bytes.
    pub fn memory_usage(&self) -> Result<usize> { self.measure(ffi::usearch_dense_memory_usage) }

    /// The expected serialized size in bytes: every shard file, plus the manifest when the
    /// index spans more than one.
    pub fn serialized_length(&self) -> Result<usize> { self.measure(ffi::usearch_dense_serialized_length) }

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

    /// Adds a matrix of vectors under their keys - the native batch operation, returning the
    /// number of rows inserted.
    ///
    /// Overlaps safely with concurrent searches on the same handle. The input scalar type is
    /// independent of the stored quantization; the library converts on ingest.
    pub fn add<Scalar: crate::types::Scalar>(
        &self,
        keys: &[Key],
        vectors: MatrixView<'_, Scalar>,
        expansion_factor: usize,
        stats: Option<&mut Stats>,
    ) -> Result<usize> {
        let (count, stride) = rows_of(&vectors, self.dimensions()?)?;
        if keys.len() != count {
            return Err(fail(
                Status::ShapeMismatch,
                "the key count does not match the vector row count",
            ));
        }
        let config = ffi::usearch_dense_expansion_config_t { expansion_factor };
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        let added = unsafe {
            ffi::usearch_dense_add(
                self.handle,
                keys.as_ptr() as *const usearch_key_t,
                vectors.as_ptr() as *const c_void,
                count,
                stride,
                Scalar::KIND as i32,
                &config,
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
        queries: MatrixView<'_, Scalar>,
        expansion_factor: usize,
        outputs: &mut Outputs<'_>,
        stats: Option<&mut Stats>,
    ) -> Result<()> {
        self.search_dispatch(queries, expansion_factor, outputs, stats, None, ptr::null_mut())
    }

    /// The filtered sibling of [`Index::search_into`], admitting only keys the predicate approves.
    ///
    /// Non-matching entries still route the traversal, so recall does not collapse at low
    /// selectivity. The predicate runs on the library's worker threads, hence `Sync`.
    pub fn search_if_into<Scalar: crate::types::Scalar, Predicate: Fn(Key) -> bool + Sync>(
        &self,
        queries: MatrixView<'_, Scalar>,
        expansion_factor: usize,
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
            queries,
            expansion_factor,
            outputs,
            stats,
            Some(trampoline::<Predicate>),
            predicate as *const Predicate as *mut c_void,
        )
    }

    fn search_dispatch<Scalar: crate::types::Scalar>(
        &self,
        queries: MatrixView<'_, Scalar>,
        expansion_factor: usize,
        outputs: &mut Outputs<'_>,
        stats: Option<&mut Stats>,
        predicate: crate::types::ffi::usearch_predicate_t,
        predicate_state: *mut c_void,
    ) -> Result<()> {
        let (queries_count, queries_stride) = rows_of(&queries, self.dimensions()?)?;
        let (keys_rows, keys_stride) = rows_of(&outputs.keys, 0)?;
        let wanted = outputs.keys.shape()[1];
        let (distances_rows, distances_stride) = rows_of(&outputs.distances, wanted)?;
        if keys_rows < queries_count || distances_rows < queries_count || outputs.counts.len() < queries_count {
            return Err(fail(
                Status::ShapeMismatch,
                "the outputs hold fewer rows than there are queries",
            ));
        }

        let config = ffi::usearch_dense_expansion_config_t { expansion_factor };
        let mut raw_stats = ffi::usearch_dense_traversal_stats_t::default();
        let mut error = usearch_error_t::default();
        let queries_pointer = queries.as_ptr() as *const c_void;
        let keys_pointer = outputs.keys.as_mut_ptr() as *mut usearch_key_t;
        let distances_pointer = outputs.distances.as_mut_ptr() as *mut usearch_distance_t;
        let counts_pointer = outputs.counts.as_mut_ptr();
        match predicate {
            None => unsafe {
                ffi::usearch_dense_search(
                    self.handle,
                    queries_pointer,
                    queries_count,
                    queries_stride,
                    Scalar::KIND as i32,
                    wanted,
                    &config,
                    &mut raw_stats,
                    keys_pointer,
                    keys_stride,
                    distances_pointer,
                    distances_stride,
                    counts_pointer,
                    &mut error,
                )
            },
            admits => unsafe {
                ffi::usearch_dense_search_if(
                    self.handle,
                    queries_pointer,
                    queries_count,
                    queries_stride,
                    Scalar::KIND as i32,
                    wanted,
                    &config,
                    &mut raw_stats,
                    admits,
                    predicate_state,
                    keys_pointer,
                    keys_stride,
                    distances_pointer,
                    distances_stride,
                    counts_pointer,
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
    pub fn contains(&self, key: Key) -> Result<bool> { Ok(self.count(key)? != 0) }

    /// Retrieves the vectors stored under `key`, one row per entry, in insertion order.
    ///
    /// Returns the true number of entries, never the number written: a count above the
    /// matrix's row capacity means the copy was truncated, so allocate that many and call
    /// again. [`Index::count`] is the same number without the copy.
    pub fn get<Scalar: crate::types::Scalar>(&self, key: Key, vectors: &mut MatrixSpan<'_, Scalar>) -> Result<usize> {
        let (wanted, stride) = rows_of(vectors, self.dimensions()?)?;
        let mut error = usearch_error_t::default();
        let entries = unsafe {
            ffi::usearch_dense_get(
                self.handle,
                key.to_raw(),
                vectors.as_mut_ptr() as *mut c_void,
                wanted,
                stride,
                Scalar::KIND as i32,
                &mut error,
            )
        };
        check(error)?;
        Ok(entries)
    }

    /// Absorbs `donor` into this index, wiring cross-links via landmark stitching.
    ///
    /// Cheaper than re-inserting the donor's vectors one by one, and readers may keep searching
    /// throughout. The donor is consumed whether or not the merge succeeds.
    pub fn merge(&self, donor: Index) -> Result<()> {
        let donor = ManuallyDrop::new(donor);
        let mut error = usearch_error_t::default();
        unsafe { ffi::usearch_dense_merge(self.handle, donor.handle, &mut error) };
        check(error)
    }

    /// Opens a snapshot cursor over the current contents, capturing the publication watermark
    /// so concurrent inserts never appear mid-iteration.
    pub fn cursor<Scalar: crate::types::Scalar>(&self) -> Result<Cursor<'_, Scalar>> {
        let dimensions = self.dimensions()?;
        let mut error = usearch_error_t::default();
        let cursor = unsafe { ffi::usearch_dense_cursor(self.handle, &mut error) };
        check(error)?;
        Ok(Cursor {
            cursor: owned(cursor)?,
            dimensions,
            index: PhantomData,
        })
    }

    /// Opens a resumable nearest-neighbor stream for one query.
    ///
    /// Best-first traversal is naturally pausable, so callers that post-filter aggressively pull
    /// until satisfied instead of guessing "k" upfront. Each open finder pins one traversal
    /// context's memory - drop it promptly.
    pub fn find<Scalar: crate::types::Scalar>(&self, query: &[Scalar]) -> Result<Finder<'_>> {
        let dimensions = self.dimensions()?;
        if dimensions != 0 && query.len() != Scalar::dims_to_values(dimensions) {
            return Err(fail(
                Status::ShapeMismatch,
                "the query width does not match the index dimensionality",
            ));
        }
        let mut error = usearch_error_t::default();
        let finder = unsafe {
            ffi::usearch_dense_find(
                self.handle,
                query.as_ptr() as *const c_void,
                Scalar::KIND as i32,
                &mut error,
            )
        };
        check(error)?;
        Ok(Finder {
            finder: owned(finder)?,
            batch: [Match {
                key: Key::NIL,
                distance: 0.0,
            }; BATCH],
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
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref().to_str().ok_or(fail(Status::Io, PATH_NOT_UTF8))?;
        let mut error = usearch_error_t::default();
        unsafe { ffi::usearch_dense_save(self.handle, path.into(), &mut error) };
        check(error)
    }

    /// Loads a checkpoint, with this handle's residency deciding the behavior: materialize into
    /// RAM, upload to the device, or open in place on disk.
    ///
    /// A handle whose geometry was left derived freezes it here, from the checkpoint's headers.
    pub fn load(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref().to_str().ok_or(fail(Status::Io, PATH_NOT_UTF8))?;
        let mut error = usearch_error_t::default();
        unsafe { ffi::usearch_dense_load(self.handle, path.into(), &mut error) };
        check(error)
    }

    /// Saves the index into a caller-owned buffer as one monolith.
    pub fn save_buffer(&self, buffer: &mut [u8]) -> Result<()> {
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_save_buffer(
                self.handle,
                buffer.as_mut_ptr() as *mut c_void,
                buffer.len(),
                &mut error,
            )
        };
        check(error)
    }

    /// Loads a monolith from a buffer; a residency that opens in place demands the buffer
    /// outlive the handle.
    pub fn load_buffer(&self, buffer: &[u8]) -> Result<()> {
        let mut error = usearch_error_t::default();
        unsafe {
            ffi::usearch_dense_load_buffer(self.handle, buffer.as_ptr() as *const c_void, buffer.len(), &mut error)
        };
        check(error)
    }
}

// endregion: Index

// region: Cursors & Finders

/// How many matches a finder pulls per crossing into the library.
const BATCH: usize = 64;

/// A snapshot cursor over the index contents; see [`Index::cursor`].
///
/// The walk is a sequential row scan, so it stays a batch operation: export, migration, and
/// visualization all move whole slabs.
#[derive(Debug)]
pub struct Cursor<'index, Scalar: crate::types::Scalar> {
    cursor: ffi::usearch_dense_cursor_t,
    dimensions: usize,
    index: PhantomData<(&'index Index, Scalar)>,
}

impl<'index, Scalar: crate::types::Scalar> Cursor<'index, Scalar> {
    /// Copies the next entries into caller-owned outputs, returning the rows written; zero
    /// means the snapshot is exhausted.
    ///
    /// Passing `None` for the vectors walks the key column alone.
    pub fn next_batch(&mut self, keys: &mut [Key], vectors: Option<&mut MatrixSpan<'_, Scalar>>) -> Result<usize> {
        let (pointer, wanted, stride) = match vectors {
            None => (ptr::null_mut(), keys.len(), 0),
            Some(matrix) => {
                let (rows, stride) = rows_of(matrix, self.dimensions)?;
                (matrix.as_mut_ptr() as *mut c_void, rows.min(keys.len()), stride)
            }
        };
        let mut error = usearch_error_t::default();
        let written = unsafe {
            ffi::usearch_dense_cursor_next(
                self.cursor,
                keys.as_mut_ptr() as *mut usearch_key_t,
                pointer,
                wanted,
                stride,
                Scalar::KIND as i32,
                &mut error,
            )
        };
        check(error)?;
        Ok(written)
    }
}

impl<'index, Scalar: crate::types::Scalar> Drop for Cursor<'index, Scalar> {
    fn drop(&mut self) { unsafe { ffi::usearch_dense_cursor_free(self.cursor) } }
}

/// A resumable nearest-neighbor stream; see [`Index::find`].
///
/// Ordering beyond the expansion horizon is best-effort nearest, not globally exact. Matches
/// arrive in batches and are yielded one at a time, so nothing allocates per item.
#[derive(Debug)]
pub struct Finder<'index> {
    finder: ffi::usearch_dense_finder_t,
    batch: [Match; BATCH],
    filled: usize,
    position: usize,
    drained: bool,
    index: PhantomData<&'index Index>,
}

impl<'index> Finder<'index> {
    fn refill(&mut self) -> Result<()> {
        let mut keys = [Key::NIL; BATCH];
        let mut distances = [0.0; BATCH];
        let mut error = usearch_error_t::default();
        let written = unsafe {
            ffi::usearch_dense_find_next(
                self.finder,
                keys.as_mut_ptr() as *mut usearch_key_t,
                distances.as_mut_ptr(),
                BATCH,
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
    fn drop(&mut self) { unsafe { ffi::usearch_dense_find_free(self.finder) } }
}

// endregion: Cursors & Finders

// region: Helpers

/// Row count and byte pitch of a matrix crossing the boundary, validated on the way.
///
/// A `dimensions` of zero skips the width check, for the outputs whose width is the wanted
/// count rather than the geometry.
fn rows_of<Element: StorageElement>(matrix: &impl TensorRef<Element, 2>, dimensions: usize) -> Result<(usize, usize)> {
    if !matrix.has_contiguous_rows() {
        return Err(fail(
            Status::UnsupportedLayout,
            "expected a matrix whose rows are contiguous; only the pitch may stride",
        ));
    }
    if dimensions != 0 && matrix.shape()[1] != dimensions {
        return Err(fail(
            Status::ShapeMismatch,
            "the matrix row width does not match the index dimensionality",
        ));
    }
    let pitch = matrix.stride_bytes(0);
    if pitch < 0 {
        return Err(fail(Status::UnsupportedLayout, "matrix rows must ascend in memory"));
    }
    Ok((matrix.shape()[0], pitch as usize))
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
    use crate::types::{rows, rows_mut, Addressing, Device, HugePages, IoMode};

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

    fn sequential_keys(count: usize) -> Vec<Key> { (0..count as u64).map(Key::from).collect() }

    fn empty_index(dimensions: usize, metric: Metric, quantization: ScalarKind) -> Index {
        let index = Index::new(&Options {
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
    ) -> (Index, Vec<Key>, Vec<f32>) {
        let index = empty_index(dimensions, metric, quantization);
        index.reserve(count).expect("the reservation must succeed");
        let keys = sequential_keys(count);
        let slab = seeded_rows(count, dimensions, SEED);
        let added = index
            .add(&keys, rows(&slab, dimensions), 0, None)
            .expect("the batch must insert");
        assert_eq!(added, count);
        (index, keys, slab)
    }

    /// The three-buffer dance collapsed into one call - the shape most assertions want.
    fn top_one(index: &Index, query: &[f32], dimensions: usize) -> Option<Match> {
        let mut keys = [Key::NIL; 1];
        let mut distances = [0.0 as Distance; 1];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut keys, &mut distances, &mut counts, 1);
        index
            .search_into(rows(query, dimensions), 0, &mut outputs, None)
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

    fn remove_quietly(path: &std::path::Path) { let _ = std::fs::remove_file(path); }

    fn close(left: f64, right: f64, tolerance: f64) -> bool { (left - right).abs() <= tolerance }

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
    fn bit_rows(bytes: &[u8]) -> Vec<u1x8> { bytes.iter().map(|&byte| u1x8(byte)).collect() }

    // endregion: Helpers

    // region: Options Round Trip

    #[test]
    fn options_survive_the_raw_round_trip_for_ram() {
        let pinned = Options {
            dimensions: 768,
            dimension_padding: DimensionPadding::SimdQuantum,
            metric: Metric::Euclidean,
            quantization: ScalarKind::I8,
            connectivity: 32,
            residency: Residency::Ram {
                threads: 7,
                huge_pages: HugePages::RequireHugePages,
            },
        };
        assert_eq!(
            Options::from_raw(&pinned.to_raw()).expect("the tags must be known"),
            pinned
        );

        let derived = Options::default();
        assert_eq!(
            Options::from_raw(&derived.to_raw()).expect("the tags must be known"),
            derived
        );
    }

    #[test]
    fn options_survive_the_raw_round_trip_for_vram() {
        // Ordinal zero is the trap: it must not come back as `FirstFree`, which marshals as -1.
        for device in [Device::FirstFree, Device::Ordinal(0), Device::Ordinal(3)] {
            for addressing in [Addressing::DeviceOnly, Addressing::Unified] {
                let options = Options {
                    residency: Residency::Vram { device, addressing },
                    ..Options::default()
                };
                assert_eq!(
                    Options::from_raw(&options.to_raw()).expect("the tags must be known"),
                    options
                );
            }
        }
    }

    #[test]
    fn options_survive_the_raw_round_trip_for_nvme() {
        for (sector_bytes, queue_depth) in [(0u32, 0u32), (512, 128), (4096, 1)] {
            for io in [IoMode::Direct, IoMode::PageCached] {
                let options = Options {
                    residency: Residency::Nvme {
                        sector_bytes,
                        queue_depth,
                        io,
                    },
                    ..Options::default()
                };
                assert_eq!(
                    Options::from_raw(&options.to_raw()).expect("the tags must be known"),
                    options
                );
            }
        }
    }

    #[test]
    fn options_reject_unknown_enum_tags() {
        let mut raw = Options::default().to_raw();
        raw.residency = 9;
        assert_eq!(
            Options::from_raw(&raw).unwrap_err().status,
            Status::UnknownKind,
            "an unknown residency must not be guessed"
        );

        let mut raw = Options::default().to_raw();
        raw.metric = 99;
        assert_eq!(
            Options::from_raw(&raw).unwrap_err().status,
            Status::UnknownKind,
            "an unknown metric must not be guessed"
        );

        let mut raw = Options::default().to_raw();
        raw.quantization = 77;
        assert_eq!(
            Options::from_raw(&raw).unwrap_err().status,
            Status::UnknownKind,
            "an unknown scalar kind must not be guessed"
        );
    }

    #[test]
    fn options_carry_no_metric_callback() {
        let raw = Options::default().to_raw();
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
        let index = Index::new(&Options {
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
    fn defers_geometry_until_frozen() {
        let index = Index::new(&Options {
            dimensions: 0,
            quantization: ScalarKind::F32,
            ..Options::default()
        })
        .expect("the index must initialize");
        assert_eq!(index.dimensions().expect("the geometry must report"), 0);

        index.reserve(4).expect("the reservation must succeed");
        let slab = seeded_rows(1, 12, SEED);
        index
            .add(&sequential_keys(1), rows(&slab, 12), 0, None)
            .expect("the first batch freezes the geometry");
        assert_eq!(index.dimensions().expect("the geometry must report"), 12);
    }

    #[test]
    fn survives_box_and_arc_moves() {
        requires_send_sync::<Index>();
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

    #[test]
    fn reports_growing_memory_usage() {
        const DIMENSIONS: usize = 32;
        let empty = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let baseline = empty.memory_usage().expect("the usage must report");
        let (full, _, _) = populated_index(256, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        assert!(full.memory_usage().expect("the usage must report") > baseline);
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
                .add(&keys, rows(&slab, DIMENSIONS), 0, None)
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
            .add(&sequential_keys(3), rows(&slab, DIMENSIONS), 0, None)
            .unwrap_err();
        assert_eq!(failure.status, Status::ShapeMismatch);
    }

    #[test]
    fn rejects_wrong_dimension_matrices() {
        const DIMENSIONS: usize = 8;
        let (index, keys, _) = populated_index(4, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let narrow = seeded_rows(1, DIMENSIONS - 1, SEED);

        let failure = index
            .add(&sequential_keys(1), rows(&narrow, DIMENSIONS - 1), 0, None)
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
            .search_into(rows(&narrow, DIMENSIONS - 1), 0, &mut outputs, None)
            .unwrap_err();
        assert_eq!(failure.status, Status::ShapeMismatch);

        assert_eq!(
            index.find(&narrow[..]).unwrap_err().status,
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
        let failure = index.add(&sequential_keys(2), interleaved, 0, None).unwrap_err();
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
        let failure = index.add(&sequential_keys(2), descending, 0, None).unwrap_err();
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
                .add(&keys, strided, 0, None)
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
            .add(&keys, rows(&slab, DIMENSIONS), 0, None)
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
            .add(&[shared, shared], rows(&slab, DIMENSIONS), 0, None)
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
            .add(&[shared; 3], rows(&slab, DIMENSIONS), 0, None)
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
            .add(&[wide, narrow], rows(&slab, DIMENSIONS), 0, None)
            .expect("wide keys insert like any other");

        assert_eq!(index.count(wide).expect("the count must report"), 1);
        assert_eq!(wide.as_u64(), None, "a non-zero high half has no integer identity");
        assert_eq!(narrow.as_u64(), Some(9));
        assert_eq!(
            top_one(&index, &slab[..DIMENSIONS], DIMENSIONS).map(|found| found.key),
            Some(wide)
        );
    }

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
            .search_into(rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS), 64, &mut outputs, None)
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
            .add(&sequential_keys(2), rows(&slab, DIMENSIONS), 0, None)
            .expect("identical rows insert");

        let found = top_one(&index, &row, DIMENSIONS).expect("a match must come back");
        assert!(close(found.distance, 0.0, 1e-6));
    }

    #[test]
    fn rejects_undersized_outputs() {
        const DIMENSIONS: usize = 8;
        const QUERIES: usize = 4;
        let (index, _, slab) = populated_index(16, DIMENSIONS, Metric::Angular, ScalarKind::F32);
        let queries = rows(&slab[..QUERIES * DIMENSIONS], DIMENSIONS);

        let mut found = vec![Key::NIL; (QUERIES - 1) * 2];
        let mut scores = vec![0.0 as Distance; (QUERIES - 1) * 2];
        let mut counts = vec![0usize; QUERIES];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 2);
        assert_eq!(
            index.search_into(queries, 0, &mut outputs, None).unwrap_err().status,
            Status::ShapeMismatch,
            "too few output rows"
        );

        let mut found = vec![Key::NIL; QUERIES * 2];
        let mut scores = vec![0.0 as Distance; QUERIES * 2];
        let mut counts = vec![0usize; QUERIES - 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 2);
        assert_eq!(
            index.search_into(queries, 0, &mut outputs, None).unwrap_err().status,
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
            .add(&keys, rows(&slab, DIMENSIONS), 0, Some(&mut inserting))
            .expect("the batch must insert");
        assert!(inserting.computed_distances > 0);

        let mut searching = Stats::default();
        let mut found = [Key::NIL; 4];
        let mut scores = [0.0 as Distance; 4];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 4);
        index
            .search_into(
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                0,
                &mut outputs,
                Some(&mut searching),
            )
            .expect("the search must run");
        let first = searching;
        assert!(first.computed_distances > 0);

        index
            .search_into(
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                0,
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
            .search_if_into(rows(&slab[..DIMENSIONS], DIMENSIONS), 64, &mut outputs, None, &is_even)
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
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                64,
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
                rows(&slab[..DIMENSIONS], DIMENSIONS),
                64,
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
            .add(&keys, rows(&stored, DIMENSIONS), 0, None)
            .expect("bit rows insert");

        let query = bit_rows(&[0b0111_1000]);
        let mut found = [Key::NIL; 2];
        let mut scores = [0.0 as Distance; 2];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, 2);
        index
            .search_into(rows(&query, DIMENSIONS), 0, &mut outputs, None)
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
            .add(&keys, rows(&stored, DIMENSIONS), 0, None)
            .expect("bit rows insert");

        // 0b00001111 against 0b00000111: three bits shared, four in the union.
        let query = bit_rows(&[0b0000_0111]);
        let found = top_one_bits(&index, &query, DIMENSIONS).expect("a match must come back");
        assert!(close(found.distance, 1.0 - 3.0 / 4.0, 1e-6));
    }

    /// The bit-vector sibling of [`top_one`], which is written against `f32` queries.
    fn top_one_bits(index: &Index, query: &[u1x8], dimensions: usize) -> Option<Match> {
        let mut keys = [Key::NIL; 1];
        let mut distances = [0.0 as Distance; 1];
        let mut counts = [0usize; 1];
        let mut outputs = Outputs::packed(&mut keys, &mut distances, &mut counts, 1);
        index
            .search_into(rows(query, dimensions), 0, &mut outputs, None)
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
        index.save(&path).expect("the checkpoint must write");

        // A derived geometry freezes from the checkpoint's own headers.
        let restored = Index::new(&Options {
            dimensions: 0,
            quantization: ScalarKind::F32,
            ..Options::default()
        })
        .expect("the index must initialize");
        restored.load(&path).expect("the checkpoint must load");
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
        let length = index.serialized_length().expect("the length must report");
        assert!(length > 0);

        let mut buffer = vec![0u8; length];
        index.save_buffer(&mut buffer).expect("the monolith must write");

        let restored = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        restored.load_buffer(&buffer).expect("the monolith must load");
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
            truncated.load_buffer(&buffer[..length - 1]).is_err(),
            "a short buffer must not be read"
        );
    }

    #[test]
    fn reads_metadata_without_loading() {
        const DIMENSIONS: usize = 16;
        let path = scratch_path("reads_metadata_without_loading");
        let (index, _, _) = populated_index(8, DIMENSIONS, Metric::Euclidean, ScalarKind::I8);
        index.save(&path).expect("the checkpoint must write");

        let described = Index::metadata(&path).expect("the metadata must read");
        assert_eq!(described.dimensions, DIMENSIONS);
        assert_eq!(described.metric, Metric::Euclidean);
        assert_eq!(described.quantization, ScalarKind::I8);

        let mut buffer = vec![0u8; index.serialized_length().expect("the length must report")];
        index.save_buffer(&mut buffer).expect("the monolith must write");
        let from_buffer = Index::metadata_buffer(&buffer).expect("the metadata must read");
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
            index.save(path).unwrap_err().status,
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
            .add(&keys, rows(&slab, DIMENSIONS), 0, None)
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
            .add(&donor_keys, rows(&donor_slab, DIMENSIONS), 0, None)
            .expect("the donor must fill");

        index.merge(donor).expect("the donor must be absorbed");
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

        let mut finder = index.find(&slab[..DIMENSIONS]).expect("the finder must open");
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
        assert!(drained > BATCH, "the stream crosses its refill boundary");

        let mut exhausted = index.find(&slab[..DIMENSIONS]).expect("the finder must open");
        while exhausted.next().is_some() {}
        assert!(exhausted.next().is_none(), "a drained finder stays drained");
    }

    // endregion: Merge, Cursor & Finder

    // region: Concurrency

    const THREADS: usize = 4;

    #[test]
    fn builds_and_searches_in_parallel() {
        const DIMENSIONS: usize = 128;
        const COUNT: usize = 1000;
        const WANTED: usize = 10;

        let index = empty_index(DIMENSIONS, Metric::Angular, ScalarKind::F32);
        index.reserve(COUNT).expect("the reservation must succeed");
        let keys = sequential_keys(COUNT);
        let slab = seeded_rows(COUNT, DIMENSIONS, SEED);

        let topology = forkunion::Topology::new().expect("the topology must enumerate");
        let mut pool = forkunion::spawn(&topology, THREADS);
        let failures = AtomicUsize::new(0);

        // A prong never panics: unwinding here would cross the C fork-join boundary, so every
        // failure is folded into a counter and asserted after the join.
        forkunion::for_n(&mut pool, COUNT, |prong| {
            let row = prong.task_index;
            let inserted = index.add(
                &keys[row..row + 1],
                rows(&slab[row * DIMENSIONS..(row + 1) * DIMENSIONS], DIMENSIONS),
                0,
                None,
            );
            if inserted != Ok(1) {
                failures.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(
            failures.load(Ordering::Relaxed),
            0,
            "every concurrent insert must succeed"
        );
        assert_eq!(index.size().expect("the size must report"), COUNT);

        let mismatches = AtomicUsize::new(0);
        forkunion::for_n(&mut pool, COUNT, |prong| {
            let row = prong.task_index;
            // Per-task locals, not per-thread state: each call owns its own stack frame, so the
            // closure captures nothing mutable and stays `Fn + Sync` for free.
            let mut restored = [0.0f32; DIMENSIONS];
            let mut span = rows_mut(&mut restored, DIMENSIONS);
            if index.get(keys[row], &mut span) != Ok(1) {
                mismatches.fetch_add(1, Ordering::Relaxed);
                return;
            }
            let source = &slab[row * DIMENSIONS..(row + 1) * DIMENSIONS];
            if restored
                .iter()
                .zip(source)
                .any(|(left, right)| (left - right).abs() > 1e-6)
            {
                mismatches.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(mismatches.load(Ordering::Relaxed), 0, "every row must come back intact");

        let misses = AtomicUsize::new(0);
        forkunion::for_n(&mut pool, COUNT, |prong| {
            let row = prong.task_index;
            let mut found = [Key::NIL; WANTED];
            let mut scores = [0.0 as Distance; WANTED];
            let mut counts = [0usize; 1];
            let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, WANTED);
            let query = &slab[row * DIMENSIONS..(row + 1) * DIMENSIONS];
            if index
                .search_into(rows(query, DIMENSIONS), 64, &mut outputs, None)
                .is_err()
            {
                misses.fetch_add(1, Ordering::Relaxed);
                return;
            }
            if counts[0] == 0 || found[0] != keys[row] {
                misses.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(
            misses.load(Ordering::Relaxed),
            0,
            "every row is its own nearest under contention"
        );
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

        // The first half lands serially, so the readers below traverse a populated graph.
        index
            .add(&keys[..HALF], rows(&slab[..HALF * DIMENSIONS], DIMENSIONS), 0, None)
            .expect("the seed batch must insert");

        let topology = forkunion::Topology::new().expect("the topology must enumerate");
        let mut pool = forkunion::spawn(&topology, THREADS);
        let failures = AtomicUsize::new(0);
        let strangers = AtomicUsize::new(0);
        let known: HashSet<Key> = keys.iter().copied().collect();

        // The range splits contiguously across threads, so the low tasks are writers and the
        // high tasks are readers - genuinely overlapping, not interleaved on one core.
        forkunion::for_n(&mut pool, 2 * HALF, |prong| match prong.task_index < HALF {
            true => {
                let row = HALF + prong.task_index;
                let inserted = index.add(
                    &keys[row..row + 1],
                    rows(&slab[row * DIMENSIONS..(row + 1) * DIMENSIONS], DIMENSIONS),
                    0,
                    None,
                );
                if inserted != Ok(1) {
                    failures.fetch_add(1, Ordering::Relaxed);
                }
            }
            false => {
                let row = prong.task_index - HALF;
                let mut found = [Key::NIL; WANTED];
                let mut scores = [0.0 as Distance; WANTED];
                let mut counts = [0usize; 1];
                let mut outputs = Outputs::packed(&mut found, &mut scores, &mut counts, WANTED);
                let query = &slab[row * DIMENSIONS..(row + 1) * DIMENSIONS];
                if index
                    .search_into(rows(query, DIMENSIONS), 32, &mut outputs, None)
                    .is_err()
                {
                    failures.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                // Whether a racing insert is visible is what the watermark leaves unspecified;
                // that no key outside the dataset appears is what it guarantees.
                if found[..counts[0]].iter().any(|key| !known.contains(key)) {
                    strangers.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        assert_eq!(failures.load(Ordering::Relaxed), 0, "no call may fail under contention");
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
