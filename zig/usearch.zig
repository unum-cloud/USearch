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
//! const nearest = (try matches.row(0)).keys[0];
//! ```
//!
//! This root re-exports every public symbol from the modules that mirror the C headers - `types`
//! and `dense` - so `@import("usearch").X` resolves whichever header X came from. Each module
//! carries its own FFI declarations at the top and its tests at the bottom.

const types = @import("usearch/types.zig");
const dense = @import("usearch/dense.zig");

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
pub const Matches = types.Matches;
pub const MatchesRow = types.MatchesRow;
pub const Admission = types.Admission;
pub const Continuation = types.Continuation;
pub const ProgressCallback = types.ProgressCallback;
pub const Predicate = types.Predicate;
pub const Mood = types.Mood;
pub const ExecutorForRamConfig = types.ExecutorForRamConfig;
pub const ExecutorOnPoolConfig = types.ExecutorOnPoolConfig;
pub const ExecutorForVramConfig = types.ExecutorForVramConfig;
pub const ExecutorForNvmeConfig = types.ExecutorForNvmeConfig;
pub const Executor = types.Executor;
pub const Arena = types.Arena;
pub const Deferred = types.Deferred;
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
pub const DenseMergeConfig = dense.DenseMergeConfig;
pub const DenseMultiplyConfig = dense.DenseMultiplyConfig;
pub const DenseAttendConfig = dense.DenseAttendConfig;
pub const DenseMaxsimConfig = dense.DenseMaxsimConfig;
pub const DenseSaveConfig = dense.DenseSaveConfig;
pub const DenseSaveBufferConfig = dense.DenseSaveBufferConfig;
pub const DenseSearchStreamInitConfig = dense.DenseSearchStreamInitConfig;
pub const DenseIndex = dense.DenseIndex;
pub const DenseVectors = dense.DenseVectors;
pub const DenseProducts = dense.DenseProducts;
pub const DenseCursorBatch = dense.DenseCursorBatch;
pub const Monolith = dense.Monolith;
pub const DenseCursor = dense.DenseCursor;
pub const DenseSearchStream = dense.DenseSearchStream;
pub const DenseReserve = dense.DenseReserve;
pub const DenseClear = dense.DenseClear;
pub const DenseDescribe = dense.DenseDescribe;
pub const DenseCount = dense.DenseCount;
pub const DenseGet = dense.DenseGet;
pub const DenseInsert = dense.DenseInsert;
pub const DenseRefine = dense.DenseRefine;
pub const DenseHealth = dense.DenseHealth;
pub const DenseSearch = dense.DenseSearch;
pub const DenseMerge = dense.DenseMerge;
pub const DenseCursorNext = dense.DenseCursorNext;
pub const DenseSearchStreamNext = dense.DenseSearchStreamNext;
pub const DenseSave = dense.DenseSave;
pub const DenseSaveBuffer = dense.DenseSaveBuffer;
pub const DenseMultiply = dense.DenseMultiply;
pub const DenseAttend = dense.DenseAttend;
pub const DenseMaxsim = dense.DenseMaxsim;

test { // pull each module's tests into `zig build test`
    _ = @import("usearch/types.zig");
    _ = @import("usearch/dense.zig");
    // Every re-export above, so one that outlives the declaration it names fails here rather
    // than in a dependent - an unreferenced `pub const` is otherwise never analyzed.
    @import("std").testing.refAllDecls(@This());
}
