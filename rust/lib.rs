//! USearch - Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices.
//!
//! The crate links the pre-instantiated C library and consumes it through its stable C headers,
//! so no index template is instantiated here and nothing but the vocabulary below crosses the
//! boundary. Each module mirrors one header and is named after it.
//!
//! ```no_run
//! use usearch::*;
//!
//! let options = DenseOptions {
//!     dimensions: 768,
//!     metric: DenseMetric::Angular,
//!     quantization: Scalar::I8,
//!     ..DenseOptions::default()
//! };
//! let index = DenseIndex::init_ram(&options, &[])?;
//! index.reserve(1_000_000)?;
//!
//! let executor = Executor::for_ram(&[], 0, 0)?;
//! let slab = vec![0.0f32; 768];
//! index.on(&executor).insert([Key::from(42u64)], &rows(&slab, 768)?)?;
//! # Ok::<(), usearch::Error>(())
//! ```

mod dense_index; // mirrors include/usearch/dense_index.h
mod types; // mirrors include/usearch/types.h

// The C mirror items are public within their modules, so only these names leave the crate.
pub use dense_index::{
    DenseAttend, DenseAttendConfig, DenseCount, DenseCursor, DenseDescribe, DenseGet, DenseHealth, DenseIndex,
    DenseIndexDescription, DenseInsert, DenseInsertConfig, DenseLoadRamBufferConfig, DenseLoadRamConfig,
    DenseLoadVramBufferConfig, DenseLoadVramConfig, DenseMaxsim, DenseMaxsimConfig, DenseMerge, DenseMergeConfig,
    DenseMetadata, DenseMetric, DenseMultiply, DenseMultiplyConfig, DenseOptions, DenseReachability, DenseRefine,
    DenseRefineConfig, DenseReserve, DenseSave, DenseSaveBuffer, DenseSaveBufferConfig, DenseSaveConfig, DenseSearch,
    DenseSearchConfig, DenseSearchStream, DenseSearchStreamInitConfig, DenseSoftmaxState, DenseTraversalStats,
    DenseVerbs, DenseViewNvmeConfig, Monolith, Products, ValueLookup, Vectors,
};
pub use types::{
    capabilities_available, capabilities_compiled, rows, rows_mut, version, Arena, Blocking, Deferred, Deferring,
    Distance, Element, Error, Executor, Key, Match, Matches, MatchesRow, MatchesSpan, Mood, On, Predicate,
    ProgressCallback, Result, Scalar, Status, Wait,
};
