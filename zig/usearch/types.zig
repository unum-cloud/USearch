//! Shared vocabulary of the USearch C interface - the Zig twin of `include/usearch/types.h`.
//!
//! Holds the `Status`/`Error` pair every call reports through, the 128-bit `Key` and the
//! `Distance` it is scored by, the `Scalar` encodings rows travel in, the `MatrixView`,
//! `MatrixSpan` and `MatchesSpan` descriptors every modality's batches ride, the `Placement` that
//! says which hardware an index lives on, and the `Executor` that drives every fanned-out verb.
//!
//! The C ABI is declared by hand rather than pulled through `@cImport`, so the module needs no
//! include path from its dependents and no cast at a call site: an `extern struct` already has
//! the ABI's layout, and an `enum(c_int)` already has its width.

const std = @import("std");

extern fn usearch_executor_for_ram(ram: *const RamOptions, threads: usize, outcome: *Outcome) ?*anyopaque;
extern fn usearch_executor_for_nvme(nvme: *const NvmeOptions, threads: usize, outcome: *Outcome) ?*anyopaque;
extern fn usearch_executor_for_vram(vram: *const VramOptions, outcome: *Outcome) ?*anyopaque;
extern fn usearch_executor_drain(executor: *anyopaque, outcome: *Outcome) void;
extern fn usearch_executor_free(executor: *anyopaque) void;

extern fn usearch_version() [*:0]const u8;
extern fn usearch_capabilities_compiled() [*:0]const u8;
extern fn usearch_capabilities_available() [*:0]const u8;

/// What went wrong, mirroring `usearch_status_t` value-for-value.
///
/// Non-exhaustive, so a status a newer engine reports arrives intact rather than coerced.
pub const Status = enum(c_int) {
    /// The call completed; the outcome's message carries nothing.
    success = 0,
    /// An allocation or reservation could not be satisfied.
    out_of_memory = 1,
    /// Dimensionality, row counts, or output capacity disagree.
    shape_mismatch = 2,
    /// A stride or packing the engine cannot traverse.
    unsupported_layout = 3,
    /// A scalar, metric, or residency tag this build does not know.
    unknown_kind = 4,
    /// A mutating verb on a handle that opens its checkpoint in place.
    read_only = 5,
    /// A file could not be opened, read, written, or parsed.
    io = 6,
    /// A device could not be selected, or its memory could not be claimed.
    device = 7,
    /// A caller-supplied representation could not be decoded.
    malformed_input = 8,
    /// The library reported neither success nor a failure - a defect, not a state.
    internal = 9,
    /// A hard design ceiling was reached, like the 32-bit id space.
    overflow = 10,
    /// The verb needs a prior successful reservation that never happened.
    uninitialized = 11,
    /// The request contradicts an already-frozen configuration.
    config_mismatch = 12,
    /// A synchronization contract was violated by a concurrent operation.
    contended = 13,
    /// A progress handler asked the operation to stop.
    cancelled = 14,
    _,
};

/// One error per status, so `@errorName` carries as much as the status itself does.
///
/// Zig error values hold no payload, so neither the static message nor the symbol that failed can
/// travel with the error. Read `Outcome.message` where that detail matters.
///
/// One set serves every verb rather than a narrower one per verb: the engine answers each call
/// through the same `Outcome`, and nothing in the ABI bounds which status a given verb may report,
/// so a narrower signature would be a promise this boundary cannot keep.
pub const Error = error{
    OutOfMemory,
    ShapeMismatch,
    UnsupportedLayout,
    UnknownKind,
    ReadOnly,
    Io,
    Device,
    MalformedInput,
    Internal,
    Overflow,
    Uninitialized,
    ConfigMismatch,
    Contended,
    Cancelled,
};

/// Outcome of one call: a status to branch on, and a static message carrying what the status
/// cannot. Mirrors `usearch_error_t`, and is what every verb passes as its trailing out-parameter.
///
/// Internal to the boundary rather than public vocabulary: every verb answers `Error!T`, so a
/// caller reads the error and never the outcome that produced it. Messages are static and need
/// no deallocation.
pub const Outcome = extern struct {
    /// What the call reported; `.success` is the only non-error.
    status: Status = .success,
    /// Static, English detail, null exactly when the status is `.success`.
    message: ?[*:0]const u8 = null,

    /// Lifts the status into the error set; `.success` is the only non-error.
    pub fn check(self: Outcome) Error!void {
        if (self.status == .success) return;
        return asError(self.status);
    }

    /// The static detail the status could not carry; empty where the call reported none.
    pub fn describe(self: Outcome) [:0]const u8 {
        return if (self.message) |text| std.mem.span(text) else "";
    }
};

/// Fails a call the binding refused before it crossed the ABI, reporting the status the engine
/// would have reported had it seen the same thing.
///
/// Every shape the binding checks itself funnels through here, so a caller cannot tell a
/// locally-rejected call from an engine-rejected one by its error value.
pub fn refuse(status: Status) Error {
    return asError(status);
}

/// The error one status names. `.success` is not one, so it answers `Internal` - a call that
/// succeeded has no business reaching here.
fn asError(status: Status) Error {
    return switch (status) {
        .out_of_memory => Error.OutOfMemory,
        .shape_mismatch => Error.ShapeMismatch,
        .unsupported_layout => Error.UnsupportedLayout,
        .unknown_kind => Error.UnknownKind,
        .read_only => Error.ReadOnly,
        .io => Error.Io,
        .device => Error.Device,
        .malformed_input => Error.MalformedInput,
        .overflow => Error.Overflow,
        .uninitialized => Error.Uninitialized,
        .config_mismatch => Error.ConfigMismatch,
        .contended => Error.Contended,
        .cancelled => Error.Cancelled,
        else => Error.Internal,
    };
}

/// Distance or similarity score between two entries, in the metric of the index that produced it.
pub const Distance = f64;

/// A 128-bit entry identity, rendered canonically as a lowercase dashed UUID.
///
/// Keys need not be unique: several entries may share one, and `DenseIndex.count` reports how
/// many. Integer identities park in the low eight bytes, so `Key.fromInteger(42)` renders as
/// `00000000-0000-0000-0000-00000000002a`.
pub const Key = extern struct {
    /// The sixteen identity bytes, most significant first.
    bytes: [16]u8,

    /// The all-zero key.
    pub const nil: Key = .{ .bytes = @splat(0) };

    /// Parks an integer identity in the low eight bytes, zeroing the rest.
    pub fn fromInteger(value: u64) Key {
        var key: Key = nil;
        std.mem.writeInt(u64, key.bytes[8..16], value, .big);
        return key;
    }

    /// The integer identity in the low eight bytes, or null when the high eight are not zero.
    pub fn asInteger(self: Key) ?u64 {
        for (self.bytes[0..8]) |byte| if (byte != 0) return null;
        return std.mem.readInt(u64, self.bytes[8..16], .big);
    }

    /// Renders the canonical lowercase dashed UUID, which is what `{f}` prints.
    pub fn format(self: Key, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        for (self.bytes, 0..) |byte, position| {
            if (position == 4 or position == 6 or position == 8 or position == 10) try writer.writeByte('-');
            try writer.print("{x:0>2}", .{byte});
        }
    }

    /// Reads the rendering `format` produces, where dashes may sit anywhere or nowhere.
    ///
    /// Anything but thirty-two hexadecimal digits, of either case, is `Error.MalformedInput`.
    pub fn parse(text: []const u8) Error!Key {
        var digits: [32]u8 = undefined;
        var written: usize = 0;
        for (text) |character| {
            if (character == '-') continue;
            if (written == digits.len) return refuse(.malformed_input);
            digits[written] = std.fmt.charToDigit(character, 16) catch return refuse(.malformed_input);
            written += 1;
        }
        if (written != digits.len) return refuse(.malformed_input);

        var key: Key = undefined;
        for (&key.bytes, 0..) |*byte, position| byte.* = digits[position * 2] << 4 | digits[position * 2 + 1];
        return key;
    }
};

/// Scalar types the indexes store and exchange, named by width and encoding. Sub-byte types pack
/// several dimensions per byte, as their `xN` suffixes spell.
///
/// Non-exhaustive, so an encoding a newer engine reports arrives intact rather than coerced.
pub const Scalar = enum(c_int) {
    /// No scalar named - an unset field, or a custom metric owning its own storage.
    unknown = 0,

    /// IEEE-754 binary64.
    f64 = 1,
    /// IEEE-754 binary32.
    f32 = 2,
    /// Truncated f32 - 8-bit exponent, 7-bit mantissa.
    bf16 = 3,
    /// IEEE-754 binary16.
    f16 = 4,
    /// FP8 with a 5-bit exponent.
    e5m2 = 5,
    /// FP8 with a 4-bit exponent.
    e4m3 = 6,
    /// FP6 with a 3-bit exponent, byte-aligned with two padding bits.
    e3m2 = 7,
    /// FP6 with a 2-bit exponent, byte-aligned with two padding bits.
    e2m3 = 8,

    /// 8-bit signed integer.
    i8 = 10,
    /// Two signed nibbles per byte, even dimensions low.
    i4x2 = 11,

    /// 8-bit unsigned integer.
    u8 = 20,
    /// Two unsigned nibbles per byte, even dimensions low.
    u4x2 = 21,
    /// Eight single-bit dimensions per byte - what Hamming and Jaccard score.
    u1x8 = 22,
    _,

    /// The encoding a Zig primitive stores as. The narrow and sub-byte kinds have no primitive
    /// to name them, so a matrix of those is built with its `scalar_kind` spelled out.
    pub fn of(comptime Element: type) Scalar {
        return switch (Element) {
            f64 => .f64,
            f32 => .f32,
            f16 => .f16,
            i8 => .i8,
            u8 => .u8,
            else => @compileError("`" ++ @typeName(Element) ++ "` names no scalar kind; spell the kind out"),
        };
    }
};

/// Where an index's payload lives and which engine traverses it. Tags `Placement`, and decides
/// what "load" means: materialize into RAM, upload to a device, or open in place on disk.
pub const Residency = enum(c_int) {
    /// Volatile-memory index traversed by CPU threads.
    ram = 0,
    /// Accelerator-memory index traversed by GPU warps.
    vram = 1,
    /// Persisted index traversed through pluggable I/O engines.
    nvme = 2,
};

/// A borrowed run of bytes and its length, never null-terminated by contract - the shape every
/// string argument arrives in. Mirrors `usearch_bytes_view_t`.
pub const BytesView = extern struct {
    /// Base pointer of the run; null names an absent argument.
    data: ?[*]const u8,
    /// Bytes the run covers.
    length: usize,

    /// Borrows a slice as the view the C interface accepts - no terminator, no copy.
    pub fn of(bytes: []const u8) BytesView {
        return .{ .data = bytes.ptr, .length = bytes.len };
    }
};

/// A caller-owned, writable run of bytes - the mutable twin of `BytesView`, carried by every verb
/// that fills a buffer the caller allocated. Mirrors `usearch_bytes_span_t`.
pub const BytesSpan = extern struct {
    /// Base pointer of the run; null names an absent argument.
    data: ?[*]u8,
    /// Bytes the run can hold.
    length: usize,

    /// Borrows a mutable slice as the span the C interface accepts.
    pub fn of(bytes: []u8) BytesSpan {
        return .{ .data = bytes.ptr, .length = bytes.len };
    }
};

/// A borrowed run of keys and its length - how a batch names the identities it carries. Mirrors
/// `usearch_keys_view_t`.
pub const KeysView = extern struct {
    /// Base pointer of the run; null names an absent argument.
    data: ?[*]const Key,
    /// Keys the run covers, checked against the row count of the matrix it accompanies.
    count: usize,

    /// Borrows a slice as the view the C interface accepts.
    pub fn of(keys: []const Key) KeysView {
        return .{ .data = keys.ptr, .count = keys.len };
    }
};

/// A caller-owned, writable run of keys - the mutable twin of `KeysView`, and the only bound a
/// streaming verb has on how much it may write. Mirrors `usearch_keys_span_t`.
pub const KeysSpan = extern struct {
    /// Base pointer of the run; null names an absent argument.
    data: ?[*]Key,
    /// Keys the run can hold, which is the batch size every such verb writes up to.
    capacity: usize,

    /// Lends a mutable slice as the span the C interface accepts.
    pub fn of(keys: []Key) KeysSpan {
        return .{ .data = keys.ptr, .capacity = keys.len };
    }
};

/// A borrowed strided matrix - the batch currency every modality's row inputs take. Mirrors
/// `usearch_matrix_view_t`.
///
/// One vector is a one-row matrix, so a caller never packs a special case. `columns` is what the
/// engine checks the frozen geometry against, and a zero `stride_bytes` means rows sit
/// contiguously at their natural width.
pub const MatrixView = extern struct {
    /// Base pointer of the first row; null names an absent matrix.
    data: ?*const anyopaque,
    /// Rows the matrix carries.
    rows: usize,
    /// Logical scalars per row, as the caller counts them.
    columns: usize,
    /// Bytes between the starts of consecutive rows; zero means contiguous.
    stride_bytes: usize,
    /// The scalar type the rows hold, converted on the way in where it differs.
    scalar_kind: Scalar,

    /// Names no matrix - what a verb reads as "skip this input".
    pub const none: MatrixView = .{ .data = null, .rows = 0, .columns = 0, .stride_bytes = 0, .scalar_kind = .unknown };

    /// Reads a contiguous slab as `data.len / columns` rows of `columns` scalars.
    ///
    /// A zero column count, or a slab that is not a whole number of rows, is a shape the engine
    /// would refuse too, so it is refused here rather than divided by.
    pub fn of(comptime Element: type, data: []const Element, columns: usize) Error!MatrixView {
        if (columns == 0 or data.len % columns != 0) return refuse(.shape_mismatch);
        return .{
            .data = data.ptr,
            .rows = data.len / columns,
            .columns = columns,
            .stride_bytes = 0,
            .scalar_kind = Scalar.of(Element),
        };
    }

    /// Reads one contiguous vector as a one-row matrix; an empty vector is no row at all.
    pub fn ofRow(comptime Element: type, data: []const Element) Error!MatrixView {
        return of(Element, data, data.len);
    }
};

/// The writable twin of `MatrixView`, carried by every verb that lands rows in a matrix the
/// caller allocated. Mirrors `usearch_matrix_span_t`.
pub const MatrixSpan = extern struct {
    /// Base pointer of the first row; null names an absent matrix.
    data: ?*anyopaque,
    /// Rows the matrix can hold, which a verb writes up to and reports back.
    capacity: usize,
    /// Logical scalars per row, as the caller counts them.
    columns: usize,
    /// Bytes between the starts of consecutive rows; zero means contiguous.
    stride_bytes: usize,
    /// The scalar type to write, converted on the way out where it differs.
    scalar_kind: Scalar,

    /// Names no matrix - what `DenseIndex.get` probes a count with, and what skips a
    /// cursor's vector export.
    pub const none: MatrixSpan = .{
        .data = null,
        .capacity = 0,
        .columns = 0,
        .stride_bytes = 0,
        .scalar_kind = .unknown,
    };

    /// Writes a contiguous slab as `data.len / columns` rows of `columns` scalars.
    ///
    /// A zero column count, or a slab that is not a whole number of rows, is a shape the engine
    /// would refuse too, so it is refused here rather than divided by.
    pub fn of(comptime Element: type, data: []Element, columns: usize) Error!MatrixSpan {
        if (columns == 0 or data.len % columns != 0) return refuse(.shape_mismatch);
        return .{
            .data = data.ptr,
            .capacity = data.len / columns,
            .columns = columns,
            .stride_bytes = 0,
            .scalar_kind = Scalar.of(Element),
        };
    }

    /// Writes one contiguous vector as a one-row matrix; an empty vector is no row at all.
    pub fn ofRow(comptime Element: type, data: []Element) Error!MatrixSpan {
        return of(Element, data, data.len);
    }
};

/// Where a ranked run lands - three caller-owned outputs, sized together. Mirrors
/// `usearch_matches_span_t`.
///
/// `wanted` is the "k" in "kANN", and it is the keys matrix that decides it, so the space
/// reserved and the count asked for cannot disagree. A sparse product and late interaction land
/// in the same shape: there `keys` names each pair's indexed row and `scores` carries its product
/// or its summed maxima.
pub const MatchesSpan = extern struct {
    /// First row of the key output.
    keys: [*]Key,
    /// Bytes between the starts of consecutive key rows; zero packs them.
    keys_stride_bytes: usize,
    /// First row of the score output, positionally paired with `keys`.
    scores: [*]Distance,
    /// Bytes between the starts of consecutive score rows; zero packs them.
    scores_stride_bytes: usize,
    /// Matches written per query row - one entry per row, never strided.
    counts: [*]usize,
    /// Columns per row, and the upper bound on matches - the "k" in "kANN".
    wanted: usize,

    /// Lends three packed slabs: `counts.len × wanted` keys and scores, one count per query.
    ///
    /// The strides and `wanted` follow from the lengths, so they are derived here rather than
    /// asked for - three numbers a caller could contradict become one shape the lengths fix.
    /// Lengths that cannot be read as one shape are refused with `Error.ShapeMismatch`.
    pub fn of(keys: []Key, scores: []Distance, counts: []usize) Error!MatchesSpan {
        if (counts.len == 0 or keys.len != scores.len or keys.len % counts.len != 0)
            return refuse(.shape_mismatch);
        return .{
            .keys = keys.ptr,
            .keys_stride_bytes = 0,
            .scores = scores.ptr,
            .scores_stride_bytes = 0,
            .counts = counts.ptr,
            .wanted = keys.len / counts.len,
        };
    }
};

/// The host memory domains a RAM-resident index places its shards in. Mirrors
/// `usearch_ram_options_t`; build one through `Placement.asOptions`.
pub const RamOptions = extern struct {
    /// NUMA node ids to place shards on; null derives every node the process may allocate on.
    memory_domains: ?[*]const i32 = null,
    /// Entries in `memory_domains`; zero derives.
    memory_domains_count: usize = 0,
};

/// The devices a VRAM-resident index places its shards in. Mirrors `usearch_vram_options_t`;
/// build one through `Placement.asOptions`.
pub const VramOptions = extern struct {
    /// CUDA device ordinals to place shards on; null derives one free device.
    devices: ?[*]const i32 = null,
    /// Entries in `devices`; zero derives.
    devices_count: usize = 0,
};

/// The devices an NVMe-resident index places its shards on. Mirrors `usearch_nvme_options_t`;
/// build one through `Placement.asOptions`.
pub const NvmeOptions = extern struct {
    /// Devices or directories to place shards on; null leaves every shard where its manifest points.
    paths: ?[*]const BytesView = null,
    /// Entries in `paths`; zero derives.
    paths_count: usize = 0,
};

/// The residency-tagged union of placement options the ABI reads, whose active member
/// `Residency` selects. Mirrors the anonymous union inside `usearch_dense_options_t`.
pub const PlacementOptions = extern union {
    /// Read when the residency is `.ram`.
    ram: RamOptions,
    /// Read when the residency is `.vram`.
    vram: VramOptions,
    /// Read when the residency is `.nvme`.
    nvme: NvmeOptions,
};

/// Where an index lives and the hardware it is placed across - one shard per named domain,
/// device, or path, sharded and never replicated.
///
/// Names where the index lives, never who walks it: threads and streams belong to the `Executor`
/// the bulk verbs take. An empty slice derives the placement from the machine.
pub const Placement = union(Residency) {
    /// NUMA node ids to place shards on; empty derives every node the process may allocate on.
    ram: []const i32,
    /// CUDA device ordinals to place shards on; empty derives one free device.
    vram: []const i32,
    /// Devices or directories to place shards on; empty leaves every shard where its manifest points.
    nvme: []const BytesView,

    /// Places a RAM index across every domain the process may allocate on.
    pub const anywhere: Placement = .{ .ram = &.{} };

    /// The C union the ABI reads, borrowing this placement's slices for the duration of the call.
    pub fn asOptions(self: Placement) PlacementOptions {
        return switch (self) {
            .ram => |domains| .{ .ram = .{ .memory_domains = domains.ptr, .memory_domains_count = domains.len } },
            .vram => |devices| .{ .vram = .{ .devices = devices.ptr, .devices_count = devices.len } },
            .nvme => |paths| .{ .nvme = .{ .paths = paths.ptr, .paths_count = paths.len } },
        };
    }
};

/// The host threads or device streams that advance a call, and the traversal scratch each of them
/// walks with - everything a verb needs to run but an index does not own.
///
/// One executor serves any number of index handles, and its kind is fixed by the constructor that
/// produced it: it must match the residency of every handle it drives, or the first call reports
/// `Error.Device`. Every fanned-out verb demands one - there is no null to pass.
///
/// An executor admits one driver at a time: a second concurrent call reports `Error.Contended`
/// rather than corrupting the dispatch already in flight. Handles stay freely concurrent - it is
/// the executor that serializes, never the index.
pub const Executor = struct {
    /// The opaque `usearch_executor_t` the ABI hands back.
    handle: *anyopaque,

    /// Spawns host workers for RAM-resident indexes across the domains local to `memory_domains`,
    /// which must be the very slice the indexes were initialized with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains carry, because full occupancy starves
    /// the ordered publication path.
    pub fn forRam(memory_domains: []const i32, threads: usize) Error!Executor {
        const ram: RamOptions = .{ .memory_domains = memory_domains.ptr, .memory_domains_count = memory_domains.len };
        var outcome: Outcome = .{};
        const handle = usearch_executor_for_ram(&ram, threads, &outcome);
        errdefer if (handle) |live| usearch_executor_free(live);
        try outcome.check();
        return .{ .handle = handle orelse return refuse(.internal) };
    }

    /// Spawns host workers for NVMe-resident indexes, one submission ring apiece, across the
    /// domains local to the devices `paths` names.
    ///
    /// Zero `threads` derives from the devices' queue capabilities.
    pub fn forNvme(paths: []const BytesView, threads: usize) Error!Executor {
        const nvme: NvmeOptions = .{ .paths = paths.ptr, .paths_count = paths.len };
        var outcome: Outcome = .{};
        const handle = usearch_executor_for_nvme(&nvme, threads, &outcome);
        errdefer if (handle) |live| usearch_executor_free(live);
        try outcome.check();
        return .{ .handle = handle orelse return refuse(.internal) };
    }

    /// Opens one submission stream on every named device, all driven by the one host thread that
    /// calls the verbs - which is why this constructor takes no thread count.
    pub fn forVram(devices: []const i32) Error!Executor {
        const vram: VramOptions = .{ .devices = devices.ptr, .devices_count = devices.len };
        var outcome: Outcome = .{};
        const handle = usearch_executor_for_vram(&vram, &outcome);
        errdefer if (handle) |live| usearch_executor_free(live);
        try outcome.check();
        return .{ .handle = handle orelse return refuse(.internal) };
    }

    /// Waits for every deferred call on this executor to finish, reporting the first failure a
    /// drained call carried.
    pub fn drain(self: Executor) Error!void {
        var outcome: Outcome = .{};
        usearch_executor_drain(self.handle, &outcome);
        try outcome.check();
    }

    /// Drains every thread or stream, releases the scratch arena, and frees the executor.
    pub fn deinit(self: Executor) void {
        usearch_executor_free(self.handle);
    }
};

// Every descriptor this module declares by hand, field by field, against the layout the C
// compiler gives its twin - the check that makes writing the interface out as safe as translating
// it. The offsets are written in pointer widths, so a 32-bit target holds them too.
comptime {
    const pointer = @sizeOf(usize);
    const tag = @sizeOf(c_int);

    std.debug.assert(@sizeOf(Key) == 16);
    std.debug.assert(@alignOf(Key) == 1);
    std.debug.assert(@offsetOf(Key, "bytes") == 0);

    std.debug.assert(@sizeOf(Status) == tag);
    std.debug.assert(@sizeOf(Scalar) == tag);
    std.debug.assert(@sizeOf(Residency) == tag);
    std.debug.assert(@sizeOf(Distance) == 8);

    std.debug.assert(@sizeOf(Outcome) == 2 * pointer);
    std.debug.assert(@offsetOf(Outcome, "status") == 0);
    std.debug.assert(@offsetOf(Outcome, "message") == pointer);

    std.debug.assert(@sizeOf(BytesView) == 2 * pointer);
    std.debug.assert(@offsetOf(BytesView, "data") == 0);
    std.debug.assert(@offsetOf(BytesView, "length") == pointer);

    std.debug.assert(@sizeOf(BytesSpan) == 2 * pointer);
    std.debug.assert(@offsetOf(BytesSpan, "data") == 0);
    std.debug.assert(@offsetOf(BytesSpan, "length") == pointer);

    std.debug.assert(@sizeOf(KeysView) == 2 * pointer);
    std.debug.assert(@offsetOf(KeysView, "data") == 0);
    std.debug.assert(@offsetOf(KeysView, "count") == pointer);

    std.debug.assert(@sizeOf(KeysSpan) == 2 * pointer);
    std.debug.assert(@offsetOf(KeysSpan, "data") == 0);
    std.debug.assert(@offsetOf(KeysSpan, "capacity") == pointer);

    std.debug.assert(@sizeOf(MatrixView) == 5 * pointer);
    std.debug.assert(@offsetOf(MatrixView, "data") == 0);
    std.debug.assert(@offsetOf(MatrixView, "rows") == pointer);
    std.debug.assert(@offsetOf(MatrixView, "columns") == 2 * pointer);
    std.debug.assert(@offsetOf(MatrixView, "stride_bytes") == 3 * pointer);
    std.debug.assert(@offsetOf(MatrixView, "scalar_kind") == 4 * pointer);

    std.debug.assert(@sizeOf(MatrixSpan) == 5 * pointer);
    std.debug.assert(@offsetOf(MatrixSpan, "data") == 0);
    std.debug.assert(@offsetOf(MatrixSpan, "capacity") == pointer);
    std.debug.assert(@offsetOf(MatrixSpan, "columns") == 2 * pointer);
    std.debug.assert(@offsetOf(MatrixSpan, "stride_bytes") == 3 * pointer);
    std.debug.assert(@offsetOf(MatrixSpan, "scalar_kind") == 4 * pointer);

    std.debug.assert(@sizeOf(MatchesSpan) == 6 * pointer);
    std.debug.assert(@offsetOf(MatchesSpan, "keys") == 0);
    std.debug.assert(@offsetOf(MatchesSpan, "keys_stride_bytes") == pointer);
    std.debug.assert(@offsetOf(MatchesSpan, "scores") == 2 * pointer);
    std.debug.assert(@offsetOf(MatchesSpan, "scores_stride_bytes") == 3 * pointer);
    std.debug.assert(@offsetOf(MatchesSpan, "counts") == 4 * pointer);
    std.debug.assert(@offsetOf(MatchesSpan, "wanted") == 5 * pointer);

    std.debug.assert(@sizeOf(RamOptions) == 2 * pointer);
    std.debug.assert(@offsetOf(RamOptions, "memory_domains") == 0);
    std.debug.assert(@offsetOf(RamOptions, "memory_domains_count") == pointer);

    std.debug.assert(@sizeOf(VramOptions) == 2 * pointer);
    std.debug.assert(@offsetOf(VramOptions, "devices") == 0);
    std.debug.assert(@offsetOf(VramOptions, "devices_count") == pointer);

    std.debug.assert(@sizeOf(NvmeOptions) == 2 * pointer);
    std.debug.assert(@offsetOf(NvmeOptions, "paths") == 0);
    std.debug.assert(@offsetOf(NvmeOptions, "paths_count") == pointer);

    std.debug.assert(@sizeOf(PlacementOptions) == 2 * pointer);
    std.debug.assert(@alignOf(PlacementOptions) == pointer);
}

/// The library's version, as a static null-terminated string.
pub fn version() [:0]const u8 {
    return std.mem.span(usearch_version());
}

/// The hardware capabilities this binary was compiled with, comma-separated.
pub fn capabilitiesCompiled() [:0]const u8 {
    return std.mem.span(usearch_capabilities_compiled());
}

/// The hardware capabilities the current machine offers at runtime, comma-separated.
pub fn capabilitiesAvailable() [:0]const u8 {
    return std.mem.span(usearch_capabilities_available());
}

test "statuses lift into the error set one for one" {
    // Every named status but `success` must map to its own error, and an unnamed one must not
    // vanish into success.
    const success: Outcome = .{};
    try success.check();
    try std.testing.expectEqualStrings("", success.describe());

    const contended: Outcome = .{ .status = .contended, .message = "another thread is driving it" };
    try std.testing.expectError(Error.Contended, contended.check());
    try std.testing.expectEqualStrings("another thread is driving it", contended.describe());

    const unnamed: Outcome = .{ .status = @enumFromInt(9999) };
    try std.testing.expectError(Error.Internal, unnamed.check());
}

test "integer identities park in the low eight bytes" {
    const key = Key.fromInteger(42);
    try std.testing.expectEqual(@as(?u64, 42), key.asInteger());
    try std.testing.expectEqual(@as(u8, 0x2a), key.bytes[15]);
    for (key.bytes[0..15]) |byte| try std.testing.expectEqual(@as(u8, 0), byte);

    try std.testing.expectEqual(@as(?u64, 0), Key.nil.asInteger());

    // A key whose high half is set is no integer identity, and says so rather than truncating.
    var wide: Key = Key.fromInteger(7);
    wide.bytes[0] = 1;
    try std.testing.expectEqual(@as(?u64, null), wide.asInteger());
}

test "keys render as dashed UUIDs and read back" {
    var buffer: [64]u8 = undefined;
    try std.testing.expectEqualStrings(
        "00000000-0000-0000-0000-00000000002a",
        try std.fmt.bufPrint(&buffer, "{f}", .{Key.fromInteger(42)}),
    );
    try std.testing.expectEqualStrings(
        "00000000-0000-0000-0000-000000000000",
        try std.fmt.bufPrint(&buffer, "{f}", .{Key.nil}),
    );

    // Every one of the sixteen bytes survives the round trip, in order.
    var wide: Key = undefined;
    for (&wide.bytes, 0..) |*byte, position| byte.* = @intCast(position * 16 + position);
    const rendered = try std.fmt.bufPrint(&buffer, "{f}", .{wide});
    try std.testing.expectEqualStrings("00112233-4455-6677-8899-aabbccddeeff", rendered);
    try std.testing.expectEqual(wide, try Key.parse(rendered));

    // Dashes may sit anywhere or nowhere, and either case of digit reads the same.
    try std.testing.expectEqual(wide, try Key.parse("00112233445566778899AABBCCDDEEFF"));
    try std.testing.expectEqual(wide, try Key.parse("0-0-1-12233445566778899aabbccddeeff"));

    try std.testing.expectError(Error.MalformedInput, Key.parse(""));
    try std.testing.expectError(Error.MalformedInput, Key.parse("00112233-4455-6677-8899-aabbccddeef"));
    try std.testing.expectError(Error.MalformedInput, Key.parse("00112233-4455-6677-8899-aabbccddeeff0"));
    try std.testing.expectError(Error.MalformedInput, Key.parse("00112233-4455-6677-8899-aabbccddeefg"));
}

test "placements borrow their slices into the ABI union" {
    const domains = [_]i32{ 0, 1 };
    const placement: Placement = .{ .ram = &domains };
    try std.testing.expectEqual(Residency.ram, std.meta.activeTag(placement));
    const options = placement.asOptions();
    try std.testing.expectEqual(@as(usize, 2), options.ram.memory_domains_count);
    try std.testing.expectEqual(@as(i32, 1), options.ram.memory_domains.?[1]);

    // Deriving is an empty slice rather than a second spelling of the same thing.
    try std.testing.expectEqual(@as(usize, 0), Placement.anywhere.asOptions().ram.memory_domains_count);
}

test "scalar kinds keep the ABI's values" {
    try std.testing.expectEqual(@as(c_int, 2), @intFromEnum(Scalar.of(f32)));
    try std.testing.expectEqual(@as(c_int, 4), @intFromEnum(Scalar.of(f16)));
    try std.testing.expectEqual(@as(c_int, 10), @intFromEnum(Scalar.of(i8)));
    try std.testing.expectEqual(@as(c_int, 22), @intFromEnum(Scalar.u1x8));
}

test { // analyze every declaration, the methods on each handle included
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Executor);
    std.testing.refAllDecls(Outcome);
    std.testing.refAllDecls(Key);
    std.testing.refAllDecls(Scalar);
    std.testing.refAllDecls(Placement);
    std.testing.refAllDecls(BytesView);
    std.testing.refAllDecls(BytesSpan);
}
