//! USearch - One Proximity Graph for Search, Sparse Products & Attention across RAM, VRAM & NVMe.
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
mod geospatial_index;
mod sealed;
mod sequence_index;
mod shapes_index;
#[cfg(test)]
mod testing;
mod types;

// The C mirror items are public within their modules, so only these names leave the crate.
pub use dense_index::{
    DenseAttend, DenseAttendConfig, DenseBatch, DenseCount, DenseCursor, DenseDescribe, DenseGet, DenseHealth,
    DenseIndex, DenseIndexDescription, DenseInsert, DenseInsertConfig, DenseLoadRamBufferConfig, DenseLoadRamConfig,
    DenseLoadVramBufferConfig, DenseLoadVramConfig, DenseMaxsim, DenseMaxsimConfig, DenseMetadata, DenseMetric,
    DenseMultiply, DenseMultiplyConfig, DenseOptions, DenseProducts, DenseReachability, DenseRefine, DenseRefineConfig,
    DenseRefinePlan, DenseRefinePlanFinish, DenseReserve, DenseReshard, DenseReshardConfig, DenseReshardPlan,
    DenseReshardPlanFinish, DenseSample, DenseSampleConfig, DenseSave, DenseSaveBuffer, DenseSaveBufferConfig,
    DenseSaveConfig, DenseSearch, DenseSearchConfig, DenseSearchStream, DenseSearchStreamInitConfig, DenseSoftmaxState,
    DenseTraversalStats, DenseVectors, DenseVerbs, DenseViewNvmeConfig, Monolith, RefineReport, RefineVerdict, Sample,
    ValueLookup,
};
pub use geospatial_index::{
    GeospatialCount, GeospatialCursor, GeospatialDescribe, GeospatialGet, GeospatialHealth, GeospatialIndex,
    GeospatialIndexDescription, GeospatialInsert, GeospatialInsertConfig, GeospatialKeyStats,
    GeospatialLoadRamBufferConfig, GeospatialLoadRamConfig, GeospatialLoadVramBufferConfig, GeospatialLoadVramConfig,
    GeospatialMatches, GeospatialMetadata, GeospatialObservation, GeospatialObservations, GeospatialOptions,
    GeospatialRefine, GeospatialRefineConfig, GeospatialRefinePlan, GeospatialRefinePlanFinish, GeospatialRelation,
    GeospatialReserve, GeospatialReshard, GeospatialReshardConfig, GeospatialReshardPlan, GeospatialReshardPlanFinish,
    GeospatialSample, GeospatialSave, GeospatialSaveBuffer, GeospatialSearch, GeospatialSearchConfig,
    GeospatialTerrain, GeospatialTerrainFill, GeospatialVerbs, GeospatialViewNvmeConfig, GeospatialWithin,
    GeospatialWithinConfig,
};
pub use sequence_index::{
    SequenceCount, SequenceCursor, SequenceDescribe, SequenceGet, SequenceHealth, SequenceIndex,
    SequenceIndexDescription, SequenceInsert, SequenceInsertConfig, SequenceKeyStats, SequenceLoadRamBufferConfig,
    SequenceLoadRamConfig, SequenceLoadVramBufferConfig, SequenceLoadVramConfig, SequenceMetadata, SequenceMetric,
    SequenceOptions, SequenceRefine, SequenceRefineConfig, SequenceRefinePlan, SequenceRefinePlanFinish,
    SequenceReserve, SequenceReshard, SequenceReshardConfig, SequenceReshardPlan, SequenceReshardPlanFinish,
    SequenceSample, SequenceSave, SequenceSaveBuffer, SequenceSearch, SequenceSearchConfig, SequenceVerbs,
    SequenceViewNvmeConfig, SequenceWeights,
};
pub use shapes_index::{
    ShapesCount, ShapesCursor, ShapesDescribe, ShapesGet, ShapesHealth, ShapesIndex, ShapesIndexDescription,
    ShapesInsert, ShapesInsertConfig, ShapesKeyStats, ShapesLoadRamBufferConfig, ShapesLoadRamConfig,
    ShapesLoadVramBufferConfig, ShapesLoadVramConfig, ShapesMetadata, ShapesMetric, ShapesOptions, ShapesRefine,
    ShapesRefineConfig, ShapesRefinePlan, ShapesRefinePlanFinish, ShapesReserve, ShapesReshard, ShapesReshardConfig,
    ShapesReshardPlan, ShapesReshardPlanFinish, ShapesSample, ShapesSave, ShapesSaveBuffer, ShapesSearch,
    ShapesSearchConfig, ShapesVerbs, ShapesViewNvmeConfig,
};
pub use types::{
    Admission, Arena, ArenaDescription, ArrowArray, ArrowColumn, ArrowDeviceArray, ArrowExport, ArrowRows, ArrowSchema,
    Blocking, CPUClass, Deferred, Deferring, Distance, Element, Error, Executor, GPUClass, Key, Matches, Mood, On,
    Phase, PhaseProgress, Predicate, PredicateCost, Progress, QueryMatches, Residency, Result, Scalar, Status, Tape,
    TapeEntries, TapeOffset, Traversal, capabilities_available, capabilities_compiled, keys_from_arrow, rows, version,
};
