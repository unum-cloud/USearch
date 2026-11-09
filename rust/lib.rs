//! USearch - One Proximity Graph for Search, Sparse Products & Attention, across RAM, VRAM & NVMe.
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
//! index.reserve(1_000_000)?; // Optional capacity hint; inserts grow on demand
//!
//! let executor = Executor::for_ram(&[], None, 0)?;
//! let mut arena = Arena::new_in(&executor)?;
//! let slab = vec![0.0f32; 768];
//! index.on(&executor).insert([Key::from(42u64)], &rows(&slab, 768)?)?;
//! let matches = index.on(&executor).search(&rows(&slab, 768)?, 10, &mut arena)?;
//! # Ok::<(), usearch::Error>(())
//! ```
//!
//! File: rust/lib.rs
//! Author: Ash Vardanian
//! Date: April 29, 2023

#![warn(missing_docs)]

mod dense_index;
mod types;

// The C mirror items are public within their modules, so only these names leave the crate.
pub use dense_index::{
    DenseAbsorbConfig, DenseAttendConfig, DenseBatch, DenseCursor, DenseIndex, DenseIndexDescription,
    DenseInsertConfig, DenseLoadRamBufferConfig, DenseLoadRamConfig, DenseLoadVramBufferConfig, DenseLoadVramConfig,
    DenseMaxsimConfig, DenseMetadata, DenseMetric, DenseMultiplyConfig, DenseOptions, DenseProducts, DenseReachability,
    DenseRefineConfig, DenseSampleConfig, DenseSaveBufferConfig, DenseSaveConfig, DenseSearchConfig, DenseSearchStream,
    DenseSearchStreamInitConfig, DenseSoftmaxState, DenseSplitConfig, DenseTraversalStats, DenseVectors, DenseVerbs,
    DenseViewNvmeConfig, Monolith, RefineReport, Sample, ValueLookup,
};
pub use types::{
    capabilities_available, capabilities_compiled, keys_from_arrow, rows, version, Arena, ArenaDescription, ArrowArray,
    ArrowColumn, ArrowDeviceArray, ArrowExport, ArrowRows, ArrowSchema, Blocking, CPUClass, Deferred, Deferring,
    Distance, Element, Error, Executor, GPUClass, Key, Matches, Mood, On, Predicate, ProgressCallback, QueryMatches,
    Residency, Result, Scalar, Status, Traversal, Wait,
};
