//! USearch - Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices.
//!
//! The crate links the precompiled engine, so building it compiles no index.
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
//! let mut arena = Arena::new_in(&executor)?;
//! let slab = vec![0.0f32; 768];
//! index.on(&executor).insert([Key::from(42u64)], &rows(&slab, 768)?)?;
//! let matches = index.on(&executor).search(&rows(&slab, 768)?, 10, &mut arena)?;
//! # Ok::<(), usearch::Error>(())
//! ```

mod dense_index;
mod types;

// The C mirror items are public within their modules, so only these names leave the crate.
pub use dense_index::{
    DenseAbsorb, DenseAbsorbConfig, DenseAttend, DenseAttendConfig, DenseBatch, DenseCount, DenseCursor, DenseDescribe,
    DenseGet, DenseHealth, DenseIndex, DenseIndexDescription, DenseInsert, DenseInsertConfig, DenseLoadRamBufferConfig,
    DenseLoadRamConfig, DenseLoadVramBufferConfig, DenseLoadVramConfig, DenseMaxsim, DenseMaxsimConfig, DenseMetadata,
    DenseMetric, DenseMultiply, DenseMultiplyConfig, DenseOptions, DenseProducts, DenseReachability, DenseRefine,
    DenseRefineConfig, DenseReserve, DenseSave, DenseSaveBuffer, DenseSaveBufferConfig, DenseSaveConfig, DenseSearch,
    DenseSearchConfig, DenseSearchStream, DenseSearchStreamInitConfig, DenseSoftmaxState, DenseTraversalStats,
    DenseVectors, DenseVerbs, DenseViewNvmeConfig, Monolith, ValueLookup,
};
pub use types::{
    capabilities_available, capabilities_compiled, rows, version, Arena, Blocking, Deferred, Deferring, Distance,
    Element, Error, Executor, Key, Matches, MatchesRow, Mood, On, Predicate, ProgressCallback, Residency, Result,
    Scalar, Status, Traversal, Wait,
};
