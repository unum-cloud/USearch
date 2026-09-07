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
//! Only the dense modality is bound. Progress reporting and cancellation are not exposed: every
//! verb passes the ABI a null handler.
//!
//! Basic usage:
//! ```zig
//! const usearch = @import("usearch");
//!
//! const index = try usearch.DenseIndex.init(&.{ .dimensions = 256, .quantization = .f32 });
//! defer index.deinit();
//! try index.reserve(1_000);
//!
//! // Every fanned-out verb needs an executor, and its kind must match the index's residency.
//! const executor = try usearch.Executor.forRam(&.{}, 0);
//! defer executor.deinit();
//!
//! const keys = [_]usearch.Key{ .fromInteger(0), .fromInteger(1) };
//! _ = try index.add(executor, &keys, try .of(f32, &slab, 256), .{}, null);
//!
//! var found: [20]usearch.Key = undefined;
//! var scores: [20]usearch.Distance = undefined;
//! var counts: [2]usize = undefined;
//! try index.search(executor, try .of(f32, &slab, 256), try .of(&found, &scores, &counts), .{}, null);
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
pub const Key = types.Key;
pub const Distance = types.Distance;
pub const Scalar = types.Scalar;
pub const Residency = types.Residency;
pub const BytesView = types.BytesView;
pub const BytesSpan = types.BytesSpan;
pub const KeysView = types.KeysView;
pub const KeysSpan = types.KeysSpan;
pub const MatrixView = types.MatrixView;
pub const MatrixSpan = types.MatrixSpan;
pub const MatchesSpan = types.MatchesSpan;
pub const RamOptions = types.RamOptions;
pub const VramOptions = types.VramOptions;
pub const NvmeOptions = types.NvmeOptions;
pub const PlacementOptions = types.PlacementOptions;
pub const Placement = types.Placement;
pub const Executor = types.Executor;
pub const version = types.version;
pub const capabilitiesCompiled = types.capabilitiesCompiled;
pub const capabilitiesAvailable = types.capabilitiesAvailable;

// dense
pub const Admission = dense.Admission;
pub const MetricCallback = dense.MetricCallback;
pub const Metric = dense.Metric;
pub const Options = dense.Options;
pub const ExpansionConfig = dense.ExpansionConfig;
pub const AttentionConfig = dense.AttentionConfig;
pub const TraversalStats = dense.TraversalStats;
pub const SoftmaxState = dense.SoftmaxState;
pub const DenseIndex = dense.DenseIndex;
pub const Cursor = dense.Cursor;
pub const Finder = dense.Finder;
pub const metadata = dense.metadata;
pub const metadataBuffer = dense.metadataBuffer;

test { // pull each module's tests into `zig build test`
    _ = @import("usearch/types.zig");
    _ = @import("usearch/dense.zig");
    // Every re-export above, so one that outlives the declaration it names fails here rather
    // than in a dependent - an unreferenced `pub const` is otherwise never analyzed.
    @import("std").testing.refAllDecls(@This());
}
