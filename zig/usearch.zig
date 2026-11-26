//! Zig binding over the USearch v3 C ABI - Vector Search over dense embeddings, in RAM, in VRAM,
//! or on NVMe.
//!
//! The engine is C++ behind a stable C interface, and this module binds that interface with
//! hand-written `extern fn` declarations rather than `@cImport`. Two reasons: a dependent then
//! needs no include path of its own, only the archive; and the declarations already say what the
//! call sites need - `enum(c_int)` for a tag, `extern struct` for a descriptor, `[*]Key` for an
//! output run - where a translated header hands back `c_uint` and `?*anyopaque` and asks every
//! call site to cast.
//!
//! Only the dense modality is bound. The receiver chooses how a call completes, and the verb
//! stays the same: the index runs it on the calling thread, `index.on(executor)` blocks on an
//! executor, and `index.deferred(executor, &call)` submits it into a `Deferred` to wait on. Every
//! optional knob - the progress handler and the search predicate among them - rides one config
//! struct per verb, so a verb has one entry point however it is configured.
//!
//! Basic usage:
//! ```zig
//! const usearch = @import("usearch");
//!
//! var index = try usearch.DenseIndex.initRam(.{ .dimensions = 256 }, &.{});
//! defer index.deinit();
//! try index.reserve(1_000);
//!
//! const keys = [_]usearch.Key{ .fromInteger(0), .fromInteger(1) };
//! try index.insert(&keys, try .of(f32, &slab, 256), .{});
//!
//! // An executor's kind must match the index's residency.
//! var executor = try usearch.Executor.forRam(.{});
//! defer executor.deinit();
//!
//! // Outputs land in an arena the executor places, reused by every call that takes it.
//! var arena = try usearch.Arena.init(executor);
//! defer arena.deinit();
//! const matches = try index.on(executor).search(try .of(f32, &slab, 256), 10, &arena, .{});
//! const nearest = matches.query(0).keys[0];
//! ```
//!
//! This root re-exports every public symbol from the modules that mirror the C headers - `types`
//! and `dense` - so `@import("usearch").X` resolves whichever header X came from. Each module
//! holds its own FFI declarations at the top and its tests at the bottom.
//!
//! File: zig/usearch.zig
//! Author: Ash Vardanian
//! Date: September 7, 2026

const types = @import("usearch/types.zig");
const dense = @import("usearch/dense.zig");
const basic = @import("usearch/basic.zig");
const sequence = @import("usearch/sequence.zig");
const shapes = @import("usearch/shapes.zig");
const geospatial = @import("usearch/geospatial.zig");

// types
pub const Status = types.Status;
pub const Error = types.Error;
pub const Outcome = types.Outcome;
pub const Key = types.Key;
pub const Distance = types.Distance;
pub const Scalar = types.Scalar;
pub const Residency = types.Residency;
pub const Traversal = types.Traversal;
pub const BytesView = types.BytesView;
pub const RowsView = types.RowsView;
pub const TapeLayout = basic.TapeLayout;
pub const TapeView = basic.TapeView;
pub const TapeEntries = basic.TapeEntries;
pub const Matches = types.Matches;
pub const QueryMatches = types.QueryMatches;
pub const Admission = types.Admission;
pub const Continuation = types.Continuation;
pub const ProgressCallback = types.ProgressCallback;
pub const Predicate = types.Predicate;
pub const Mood = types.Mood;
pub const Phase = types.Phase;
pub const PhaseProgress = types.PhaseProgress;
pub const ExecutorForRamConfig = types.ExecutorForRamConfig;
pub const ExecutorForVramConfig = types.ExecutorForVramConfig;
pub const ExecutorForNvmeConfig = types.ExecutorForNvmeConfig;
pub const Executor = types.Executor;
pub const Arena = types.Arena;
pub const ArenaDescription = types.ArenaDescription;
pub const version = types.version;
pub const capabilitiesCompiled = types.capabilitiesCompiled;
pub const capabilitiesAvailable = types.capabilitiesAvailable;

// dense
pub const ValueLookup = dense.ValueLookup;
pub const DenseMetric = dense.DenseMetric;
pub const DenseOptions = dense.DenseOptions;
pub const DenseMetadata = dense.DenseMetadata;
pub const DenseIndexDescription = dense.DenseIndexDescription;
pub const DenseTraversalStats = dense.DenseTraversalStats;
pub const DenseReachability = dense.DenseReachability;
pub const DenseRefineVerdict = dense.DenseRefineVerdict;
pub const DenseRefineReport = dense.DenseRefineReport;
pub const DenseRefinePlan = dense.DenseRefinePlan;
pub const DenseReshardPlan = dense.DenseReshardPlan;
pub const DenseSoftmaxState = dense.DenseSoftmaxState;
pub const DenseLoadRamConfig = dense.DenseLoadRamConfig;
pub const DenseLoadVramConfig = dense.DenseLoadVramConfig;
pub const DenseLoadRamBufferConfig = dense.DenseLoadRamBufferConfig;
pub const DenseLoadVramBufferConfig = dense.DenseLoadVramBufferConfig;
pub const DenseViewNvmeConfig = dense.DenseViewNvmeConfig;
pub const DenseMetadataReadConfig = dense.DenseMetadataReadConfig;
pub const DenseMetadataReadBufferConfig = dense.DenseMetadataReadBufferConfig;
pub const DenseInsertConfig = dense.DenseInsertConfig;
pub const DenseRefineConfig = dense.DenseRefineConfig;
pub const DenseSearchConfig = dense.DenseSearchConfig;
pub const DenseReshardConfig = dense.DenseReshardConfig;
pub const DenseSampleConfig = dense.DenseSampleConfig;
pub const DenseMultiplyConfig = dense.DenseMultiplyConfig;
pub const DenseAttendConfig = dense.DenseAttendConfig;
pub const DenseMaxsimConfig = dense.DenseMaxsimConfig;
pub const DenseSaveConfig = dense.DenseSaveConfig;
pub const DenseSaveBufferConfig = dense.DenseSaveBufferConfig;
pub const DenseSearchStreamInitConfig = dense.DenseSearchStreamInitConfig;
pub const DenseIndex = dense.DenseIndex;
pub const Deferred = dense.Deferred;
pub const DenseVectors = dense.DenseVectors;
pub const DenseProducts = dense.DenseProducts;
pub const DenseCursorBatch = dense.DenseCursorBatch;
pub const DenseDraws = dense.DenseDraws;
pub const Monolith = dense.Monolith;
pub const DenseCursor = dense.DenseCursor;
pub const DenseSearchStream = dense.DenseSearchStream;

// sequence
pub const SequenceMetric = sequence.SequenceMetric;
pub const SequenceCosts = sequence.SequenceCosts;
pub const SequenceOptions = sequence.SequenceOptions;
pub const SequenceMetadata = sequence.SequenceMetadata;
pub const SequenceIndexDescription = sequence.SequenceIndexDescription;
pub const SequenceKeyStats = sequence.SequenceKeyStats;
pub const SequenceRefinePlan = sequence.SequenceRefinePlan;
pub const SequenceReshardPlan = sequence.SequenceReshardPlan;
pub const SequenceLoadRamConfig = sequence.SequenceLoadRamConfig;
pub const SequenceLoadVramConfig = sequence.SequenceLoadVramConfig;
pub const SequenceLoadRamBufferConfig = sequence.SequenceLoadRamBufferConfig;
pub const SequenceLoadVramBufferConfig = sequence.SequenceLoadVramBufferConfig;
pub const SequenceViewNvmeConfig = sequence.SequenceViewNvmeConfig;
pub const SequenceInsertConfig = sequence.SequenceInsertConfig;
pub const SequenceRefineConfig = sequence.SequenceRefineConfig;
pub const SequenceSearchConfig = sequence.SequenceSearchConfig;
pub const SequenceReshardConfig = sequence.SequenceReshardConfig;
pub const SequenceIndex = sequence.SequenceIndex;
pub const SequenceCursor = sequence.SequenceCursor;

// shapes
pub const ShapesMetric = shapes.ShapesMetric;
pub const ShapesOptions = shapes.ShapesOptions;
pub const ShapesMetadata = shapes.ShapesMetadata;
pub const ShapesIndexDescription = shapes.ShapesIndexDescription;
pub const ShapesKeyStats = shapes.ShapesKeyStats;
pub const ShapesRefinePlan = shapes.ShapesRefinePlan;
pub const ShapesReshardPlan = shapes.ShapesReshardPlan;
pub const ShapesLoadRamConfig = shapes.ShapesLoadRamConfig;
pub const ShapesLoadVramConfig = shapes.ShapesLoadVramConfig;
pub const ShapesLoadRamBufferConfig = shapes.ShapesLoadRamBufferConfig;
pub const ShapesLoadVramBufferConfig = shapes.ShapesLoadVramBufferConfig;
pub const ShapesViewNvmeConfig = shapes.ShapesViewNvmeConfig;
pub const ShapesInsertConfig = shapes.ShapesInsertConfig;
pub const ShapesRefineConfig = shapes.ShapesRefineConfig;
pub const ShapesSearchConfig = shapes.ShapesSearchConfig;
pub const ShapesReshardConfig = shapes.ShapesReshardConfig;
pub const ShapesIndex = shapes.ShapesIndex;
pub const ShapesCursor = shapes.ShapesCursor;

// geospatial
pub const GeospatialObservation = geospatial.GeospatialObservation;
pub const GeospatialRelation = geospatial.GeospatialRelation;
pub const TerrainFunction = geospatial.TerrainFunction;
pub const GeospatialTerrainFill = geospatial.GeospatialTerrainFill;
pub const GeospatialTerrain = geospatial.GeospatialTerrain;
pub const GeospatialOptions = geospatial.GeospatialOptions;
pub const GeospatialMetadata = geospatial.GeospatialMetadata;
pub const GeospatialIndexDescription = geospatial.GeospatialIndexDescription;
pub const GeospatialKeyStats = geospatial.GeospatialKeyStats;
pub const GeospatialRefinePlan = geospatial.GeospatialRefinePlan;
pub const GeospatialReshardPlan = geospatial.GeospatialReshardPlan;
pub const GeospatialLoadRamConfig = geospatial.GeospatialLoadRamConfig;
pub const GeospatialLoadVramConfig = geospatial.GeospatialLoadVramConfig;
pub const GeospatialLoadRamBufferConfig = geospatial.GeospatialLoadRamBufferConfig;
pub const GeospatialLoadVramBufferConfig = geospatial.GeospatialLoadVramBufferConfig;
pub const GeospatialViewNvmeConfig = geospatial.GeospatialViewNvmeConfig;
pub const GeospatialInsertConfig = geospatial.GeospatialInsertConfig;
pub const GeospatialRefineConfig = geospatial.GeospatialRefineConfig;
pub const GeospatialSearchConfig = geospatial.GeospatialSearchConfig;
pub const GeospatialWithinConfig = geospatial.GeospatialWithinConfig;
pub const GeospatialReshardConfig = geospatial.GeospatialReshardConfig;
pub const GeospatialIndex = geospatial.GeospatialIndex;
pub const GeospatialObservations = geospatial.GeospatialObservations;
pub const GeospatialMatches = geospatial.GeospatialMatches;
pub const GeospatialCursor = geospatial.GeospatialCursor;

test { // pull each module's tests into `zig build test`
    _ = @import("usearch/types.zig");
    _ = @import("usearch/dense.zig");
    _ = @import("usearch/basic.zig");
    _ = @import("usearch/sequence.zig");
    _ = @import("usearch/shapes.zig");
    _ = @import("usearch/geospatial.zig");
    // Every re-export above, so one that outlives the declaration it names fails here rather
    // than in a dependent - an unreferenced `pub const` is otherwise never analyzed.
    @import("std").testing.refAllDecls(@This());
}
