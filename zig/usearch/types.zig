//! Shared vocabulary of the USearch C interface - the Zig twin of `include/usearch/types.h`.
//!
//! Holds the `Status`/`Error` pair every call reports through, the 128-bit `Key` and the
//! `Distance` it is scored by, the `Scalar` encodings rows travel in, the `RowsView` every
//! modality's batches ride, the `Executor` that drives every verb, the `Arena` every output whose
//! length scales with a call lands in, the `Matches` a ranked run comes back as, and the
//! `Deferred` a submitted call is held in.
//!
//! The C ABI is declared by hand rather than pulled through `@cImport`, so the module needs no
//! include path from its dependents and no cast at a call site: an `extern struct` already has
//! the ABI's layout, and an `enum(c_int)` already has its width.

const std = @import("std");

extern fn usearch_outcome_wait(outcome: *Outcome) Status;
extern fn usearch_outcome_status(outcome: *Outcome) Status;
extern fn usearch_executor_for_ram(arguments: *usearch_executor_for_ram_t) void;
extern fn usearch_executor_on_pool(arguments: *usearch_executor_on_pool_t) void;
extern fn usearch_executor_for_nvme(arguments: *usearch_executor_for_nvme_t) void;
extern fn usearch_executor_for_vram(arguments: *usearch_executor_for_vram_t) void;
extern fn usearch_executor_drain(arguments: *usearch_executor_drain_t) void;
extern fn usearch_executor_free(executor: ?*anyopaque) void;
extern fn usearch_arena_init(arguments: *usearch_arena_init_t) void;
extern fn usearch_arena_free(arena: ?*anyopaque) void;

extern fn usearch_status_name(status: Status) [*:0]const u8;
extern fn usearch_status_from_name(name: BytesView) Status;
extern fn usearch_scalar_name(scalar: Scalar) [*:0]const u8;
extern fn usearch_scalar_from_name(name: BytesView) Scalar;
extern fn usearch_residency_name(residency: Residency) [*:0]const u8;
extern fn usearch_residency_from_name(name: BytesView) Residency;
extern fn usearch_mood_name(mood: Mood) [*:0]const u8;
extern fn usearch_mood_from_name(name: BytesView) Mood;

extern fn usearch_version_name() [*:0]const u8;
extern fn usearch_capabilities_compiled() [*:0]const u8;
extern fn usearch_capabilities_available() [*:0]const u8;

/// What went wrong, mirroring `usearch_status_t` value-for-value.
///
/// Non-exhaustive, so a status a newer engine reports arrives intact rather than coerced.
pub const Status = enum(c_int) {
    /// The call completed; the outcome's message carries nothing.
    success = 0,
    /// The call was submitted deferred and is not final yet; the one non-final status.
    deferred = 1,
    /// An allocation or reservation could not be satisfied.
    out_of_memory = 2,
    /// Dimensionality or row counts disagree.
    shape_mismatch = 3,
    /// A stride or packing the engine cannot traverse.
    unsupported_layout = 4,
    /// A scalar, metric, or residency tag this build does not know.
    unknown_kind = 5,
    /// A mutating verb on a handle that serves its checkpoint in place.
    read_only = 6,
    /// A file could not be opened, read, written, or parsed.
    io = 7,
    /// A device could not be selected, or its memory could not be claimed.
    device = 8,
    /// A caller-supplied representation could not be decoded.
    malformed = 9,
    /// The library reported neither success nor a failure - a defect, not a state.
    internal = 10,
    /// A hard design ceiling was reached, like the 32-bit ID space.
    overflow = 11,
    /// The verb needs a prior successful reservation that never happened.
    uninitialized = 12,
    /// The request contradicts an already-frozen configuration.
    config_mismatch = 13,
    /// The call was refused rather than waited for - a full deferred table, or an arena serving another call.
    contended = 14,
    /// A progress handler asked the operation to stop.
    cancelled = 15,
    /// The verb is declared for every residency but this one does not implement it.
    unsupported = 16,
    /// An argument breaks the verb's contract.
    invalid_argument = 17,
    _,

    /// The canonical spelling, "unrecognized" for a value no status carries. Calls `usearch_status_name`.
    pub fn name(self: Status) [:0]const u8 {
        return std.mem.span(usearch_status_name(self));
    }

    /// Reads a spelling back, or null when no status spells it. Calls `usearch_status_from_name`.
    pub fn fromName(text: []const u8) ?Status {
        return spelled(Status, usearch_status_from_name(.of(text)), text);
    }
};

/// One error per final failure status, so `@errorName` carries as much as the status itself does.
///
/// Zig error values hold no payload, so neither the static message nor the symbol that failed can
/// travel with the error. Lend an `Outcome` as a call's diagnostics, or read `Deferred.message`,
/// where that detail matters.
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
    Malformed,
    Internal,
    Overflow,
    Uninitialized,
    ConfigMismatch,
    Contended,
    Cancelled,
    Unsupported,
    InvalidArgument,
};

/// Outcome of one call: a status to branch on, and a static message carrying what the status
/// cannot. Mirrors `usearch_outcome_t`, which every verb's argument structure points at.
///
/// Every verb answers `Error!T`, and a caller wanting the message too lends an `Outcome` as the
/// call's diagnostics, which the call's final outcome is copied into. Messages are static and need
/// no deallocation.
pub const Outcome = extern struct {
    /// What the call reported; `.success` is the only non-error.
    status: Status = .success,
    /// Static, English detail, null exactly when the status is `.success`.
    message: ?[*:0]const u8 = null,

    /// Lifts the status into the error set; `.success` is the only non-error.
    pub fn check(self: Outcome) Error!void {
        if (self.status == .success) return;
        return refuse(self.status);
    }

    /// The static detail the status could not carry; empty where the call reported none.
    pub fn text(self: Outcome) [:0]const u8 {
        return if (self.message) |message| std.mem.span(message) else "";
    }

    /// Copies the outcome into `diagnostics` where one was lent, then lifts it as `check` does.
    pub fn checkInto(self: Outcome, diagnostics: ?*Outcome) Error!void {
        if (diagnostics) |lent| lent.* = self;
        return self.check();
    }
};

/// The error one status names. Fails a call the binding refused before it crossed the ABI too,
/// reporting the status the engine would have reported had it seen the same thing.
///
/// Every shape the binding checks itself funnels through here, so a caller cannot tell a
/// locally-rejected call from an engine-rejected one by its error value. A `.deferred` status
/// reaching a final check is a defect and answers `Internal`; a value this build has no name for
/// answers `InvalidArgument`, as C names it.
pub fn refuse(status: Status) Error {
    return switch (status) {
        .out_of_memory => Error.OutOfMemory,
        .shape_mismatch => Error.ShapeMismatch,
        .unsupported_layout => Error.UnsupportedLayout,
        .unknown_kind => Error.UnknownKind,
        .read_only => Error.ReadOnly,
        .io => Error.Io,
        .device => Error.Device,
        .malformed => Error.Malformed,
        .overflow => Error.Overflow,
        .uninitialized => Error.Uninitialized,
        .config_mismatch => Error.ConfigMismatch,
        .contended => Error.Contended,
        .cancelled => Error.Cancelled,
        .unsupported => Error.Unsupported,
        .success, .internal, .deferred => Error.Internal,
        else => Error.InvalidArgument,
    };
}

/// `refuse`, leaving the binding's own `message` in `diagnostics` where one was lent.
pub fn refuseInto(diagnostics: ?*Outcome, status: Status, message: [*:0]const u8) Error {
    if (diagnostics) |lent| lent.* = .{ .status = status, .message = message };
    return refuse(status);
}

/// The value a C `_from_name` table read back, or null when its own spelling of it differs, since
/// the tables answer a fallback rather than a refusal for text they do not know.
fn spelled(comptime Enum: type, value: Enum, text: []const u8) ?Enum {
    return if (std.mem.eql(u8, value.name(), text)) value else null;
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
        if (std.mem.readInt(u64, self.bytes[0..8], .big) != 0) return null;
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
    /// Anything but thirty-two hexadecimal digits, of either case, is `Error.Malformed`.
    pub fn parse(text: []const u8) Error!Key {
        var digits: [32]u8 = undefined;
        var written: usize = 0;
        for (text) |character| {
            if (character == '-') continue;
            if (written == digits.len) return refuse(.malformed);
            digits[written] = std.fmt.charToDigit(character, 16) catch return refuse(.malformed);
            written += 1;
        }
        if (written != digits.len) return refuse(.malformed);

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
    /// 32-bit signed integer - the coordinates of geospatial points.
    i32 = 12,

    /// 8-bit unsigned integer.
    u8 = 20,
    /// Two unsigned nibbles per byte, even dimensions low.
    u4x2 = 21,
    /// Eight single-bit dimensions per byte - what Hamming and Jaccard score.
    u1x8 = 22,
    _,

    /// The raw bits of one `.bf16` value, for the encodings Zig has no primitive for.
    pub const BF16 = extern struct { bits: u16 };
    /// The raw bits of one `.e5m2` value.
    pub const E5M2 = extern struct { bits: u8 };
    /// The raw bits of one `.e4m3` value.
    pub const E4M3 = extern struct { bits: u8 };
    /// The raw bits of one `.e3m2` value, two padding bits high.
    pub const E3M2 = extern struct { bits: u8 };
    /// The raw bits of one `.e2m3` value, two padding bits high.
    pub const E2M3 = extern struct { bits: u8 };
    /// Two signed nibbles of `.i4x2`, the even dimension low.
    pub const I4x2 = extern struct { bits: u8 };
    /// Two unsigned nibbles of `.u4x2`, the even dimension low.
    pub const U4x2 = extern struct { bits: u8 };
    /// Eight single-bit dimensions of `.u1x8`, the first one lowest.
    pub const U1x8 = extern struct { bits: u8 };

    /// The encoding a Zig element type stores as: a primitive, or one of the raw-bit types above.
    pub fn of(comptime Element: type) Scalar {
        return switch (Element) {
            f64 => .f64,
            f32 => .f32,
            BF16 => .bf16,
            f16 => .f16,
            E5M2 => .e5m2,
            E4M3 => .e4m3,
            E3M2 => .e3m2,
            E2M3 => .e2m3,
            i8 => .i8,
            I4x2 => .i4x2,
            i32 => .i32,
            u8 => .u8,
            U4x2 => .u4x2,
            U1x8 => .u1x8,
            else => @compileError("`" ++ @typeName(Element) ++ "` names no scalar kind"),
        };
    }

    /// Logical dimensions one stored value carries: more than one for the packed kinds. Binding-only.
    pub fn dimensionsPerValue(self: Scalar) usize {
        return switch (self) {
            .i4x2, .u4x2 => 2,
            .u1x8 => 8,
            else => 1,
        };
    }

    /// The canonical spelling, as `.f32` spells "f32". Calls `usearch_scalar_name`.
    pub fn name(self: Scalar) [:0]const u8 {
        return std.mem.span(usearch_scalar_name(self));
    }

    /// Reads a spelling back, or null when no kind spells it. Calls `usearch_scalar_from_name`.
    pub fn fromName(text: []const u8) ?Scalar {
        return spelled(Scalar, usearch_scalar_from_name(.of(text)), text);
    }
};

/// Where an index's payload lives and which engine traverses it - the vocabulary the lifecycle
/// verbs spell in their names, as `DenseIndex.initVram` does. Kept for its name table.
pub const Residency = enum(c_int) {
    /// Volatile-memory index traversed by CPU threads.
    ram = 0,
    /// Accelerator-memory index traversed by GPU warps.
    vram = 1,
    /// Persisted index traversed through pluggable I/O engines.
    nvme = 2,

    /// The canonical spelling, as `.ram` spells "ram". Calls `usearch_residency_name`.
    pub fn name(self: Residency) [:0]const u8 {
        return std.mem.span(usearch_residency_name(self));
    }

    /// Reads a spelling back, or null when no residency spells it. Calls `usearch_residency_from_name`.
    pub fn fromName(text: []const u8) ?Residency {
        return spelled(Residency, usearch_residency_from_name(.of(text)), text);
    }
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

/// A strided run of keys - how a batch names the identity of each of its rows, or a cursor pull
/// the keys it wrote into its arena. Mirrors `usearch_keys_view_t`; the call's own count bounds it.
pub const usearch_keys_view_t = extern struct {
    /// First key of the run; null names an absent argument.
    data: ?[*]const Key = null,
    /// Bytes between consecutive keys; zero repeats the first.
    stride: usize = 0,

    /// Borrows a packed slice as the view the C interface accepts.
    pub fn of(keys: []const Key) usearch_keys_view_t {
        return .{ .data = keys.ptr, .stride = @sizeOf(Key) };
    }
};

/// A run of rows along a call's batch axis, as the C interface reads an input or writes an output
/// into an arena. Mirrors `usearch_rows_view_t`; an input is built from a `RowsView`, whose row
/// count becomes the call's count.
pub const usearch_rows_view_t = extern struct {
    data: ?*const anyopaque = null,
    stride: usize = 0,
    columns: usize = 0,
    scalar: Scalar = .unknown,

    /// The rows the C interface reads; the row count travels separately, as the call's count.
    pub fn of(rows: RowsView) usearch_rows_view_t {
        return .{ .data = rows.data, .stride = rows.stride, .columns = rows.columns, .scalar = rows.scalar };
    }
};

/// A borrowed run of rows - the batch currency every modality's row inputs take, mirroring
/// `usearch_rows_view_t` with the row count it leaves to the call.
///
/// One vector is a one-row run, so a caller never packs a special case. `rows` is the batch axis a
/// verb passes on as its count, and `columns` travels with the rows for the engine to check
/// against the frozen geometry. A zero `stride` repeats the first row.
pub const RowsView = struct {
    /// Base pointer of the first row.
    data: ?*const anyopaque,
    /// Rows the run carries.
    rows: usize,
    /// Logical scalars per row, as the caller counts them - dimensions, never bytes.
    columns: usize,
    /// Bytes between the starts of consecutive rows; zero repeats the first row.
    stride: usize,
    /// The scalar type the rows hold, converted on the way in where it differs.
    scalar: Scalar,

    /// Reads a contiguous slab as rows of `columns` dimensions, which a packed kind counts in
    /// dimensions rather than values, so 128 dimensions of `Scalar.U1x8` take sixteen values a row.
    ///
    /// A zero column count, a width that does not fill whole values, or a slab that is not a whole
    /// number of rows is a shape the engine would refuse too, so it is refused here rather than
    /// divided by.
    pub fn of(comptime Element: type, data: []const Element, columns: usize) Error!RowsView {
        const values = try valuesPerRow(Element, data.len, columns);
        return .{
            .data = data.ptr,
            .rows = data.len / values,
            .columns = columns,
            .stride = values * @sizeOf(Element),
            .scalar = Scalar.of(Element),
        };
    }

    /// Reads one contiguous vector as a one-row run; an empty vector is no row at all.
    pub fn ofRow(comptime Element: type, data: []const Element) Error!RowsView {
        return of(Element, data, data.len * comptime Scalar.of(Element).dimensionsPerValue());
    }
};

/// The values of `Element` one row of `columns` dimensions takes, refusing a slab of `length`
/// values that is not a whole number of such rows.
fn valuesPerRow(comptime Element: type, length: usize, columns: usize) Error!usize {
    const per_value = comptime Scalar.of(Element).dimensionsPerValue();
    if (columns == 0 or columns % per_value != 0) return refuse(.shape_mismatch);
    const values = columns / per_value;
    if (length % values != 0) return refuse(.shape_mismatch);
    return values;
}

/// A ranked run a call wrote into its arena: rows of `columns` packed columns, at most the call's
/// `wanted`, the first `counts[i]` of row `i` written. Mirrors `usearch_matches_view_t`.
pub const usearch_matches_view_t = extern struct {
    keys: ?[*]const Key = null,
    scores: ?[*]const Distance = null,
    counts: ?[*]const usize = null,
    columns: usize = 0,
};

/// A ragged run of matches a call wrote into its arena, row `i` spanning `offsets[i]` up to
/// `offsets[i + 1]`. Mirrors `usearch_matches_tape_view_t`.
pub const usearch_matches_tape_view_t = extern struct {
    keys: ?[*]const Key = null,
    scores: ?[*]const Distance = null,
    offsets: ?[*]const usize = null,
};

/// The matches of one row: nearest first for a search, largest first for a `maxsim`, and in
/// traversal order for a `multiply`. Binding-only.
pub const MatchesRow = struct {
    /// The keys written for the row.
    keys: []const Key,
    /// Positionally paired with `keys`: distances for a search, similarities for the linear-algebra verbs.
    scores: []const Distance,
};

/// The ranked rows a search, a `maxsim` or a stream pull wrote into an arena, one per query, each
/// holding up to the `wanted` the call asked for. Mirrors `usearch_matches_view_t`.
///
/// It reads the arena, so every read is refused once that arena serves another call or is freed.
pub const Matches = struct {
    stamp: Arena.Stamp,
    view: usearch_matches_view_t,
    /// Query rows the call resolved.
    queries: usize,

    /// The matches written for one query, refusing a `query` past `queries` with
    /// `Error.InvalidArgument`, a stale view with `Error.Uninitialized`, and device memory with `Error.Device`.
    pub fn row(self: Matches, query: usize) Error!MatchesRow {
        try self.stamp.check();
        if (query >= self.queries) return refuse(.invalid_argument);
        const first = query * self.view.columns;
        const written = self.view.counts.?[query];
        return .{ .keys = landed(Key, self.view.keys, first, written), .scores = landed(Distance, self.view.scores, first, written) };
    }

    /// Where the matches live, which the host reads only as `.ram` or `.nvme`.
    pub fn residency(self: Matches) Residency {
        return self.stamp.arena.residency;
    }
};

/// Borrows `length` elements from `first` on, of a run a call wrote into an arena; a failed or
/// empty call leaves the run null. Binding-only.
pub fn landed(comptime Element: type, data: ?[*]const Element, first: usize, length: usize) []const Element {
    return if (length == 0) &.{} else data.?[first..][0..length];
}

/// The host memory domains a RAM-resident index places its shards in. Mirrors
/// `usearch_ram_options_t`.
pub const usearch_ram_options_t = extern struct {
    /// NUMA node IDs to place shards on; null derives every node the process may allocate on.
    memory_domains: ?[*]const i32 = null,
    /// Entries in `memory_domains`; zero derives.
    memory_domains_count: usize = 0,

    pub fn of(memory_domains: []const i32) usearch_ram_options_t {
        return .{ .memory_domains = memory_domains.ptr, .memory_domains_count = memory_domains.len };
    }
};

/// The devices a VRAM-resident index places its shards in. Mirrors `usearch_vram_options_t`.
pub const usearch_vram_options_t = extern struct {
    /// CUDA device ordinals to place shards on; null derives one free device.
    devices: ?[*]const i32 = null,
    /// Entries in `devices`; zero derives.
    devices_count: usize = 0,

    pub fn of(devices: []const i32) usearch_vram_options_t {
        return .{ .devices = devices.ptr, .devices_count = devices.len };
    }
};

/// The devices an NVMe-resident index serves its shards from. Mirrors `usearch_nvme_options_t`.
pub const usearch_nvme_options_t = extern struct {
    /// Devices or directories to serve shards from; null leaves every shard where its manifest points.
    paths: ?[*]const BytesView = null,
    /// Entries in `paths`; zero derives.
    paths_count: usize = 0,

    pub fn of(paths: []const BytesView) usearch_nvme_options_t {
        return .{ .paths = paths.ptr, .paths_count = paths.len };
    }
};

/// Whether a filtered search admits one candidate key or passes over it. Mirrors `usearch_admission_t`.
pub const Admission = enum(c_int) {
    /// Skip the key; the walk still routes through it.
    reject = 0,
    /// Report the key among the matches.
    admit = 1,
};

/// Whether the operation a handler is watching carries on or unwinds. Mirrors `usearch_continuation_t`.
pub const Continuation = enum(c_int) {
    /// Carry on to the next unit of work.
    @"continue" = 0,
    /// Stop; what the finished increments leave behind is each verb's own contract.
    cancel = 1,
};

/// Mirrors `usearch_progress_callback_t`.
pub const ProgressFunction = *const fn (usize, usize, ?*anyopaque) callconv(.c) Continuation;
/// Mirrors `usearch_predicate_t`.
pub const PredicateFunction = *const fn (Key, ?*anyopaque) callconv(.c) Admission;

/// A progress handler as the C interface carries it: a `usearch_progress_callback_t` and the state
/// forwarded into it. Build one with `ProgressCallback.of`; the default reports nowhere.
pub const ProgressCallback = struct {
    callback: ?ProgressFunction = null,
    state: ?*anyopaque = null,

    /// Wraps `reports` in a trampoline generated per call site, so it reads a typed context rather
    /// than an `anyopaque`. Pass `{}` for a handler needing no context, and a single-item pointer
    /// otherwise.
    ///
    /// The handler arrives on whichever thread advanced the work and never on two at once, so it
    /// needs no synchronization of its own; answering `.cancel` stops the call.
    pub fn of(context: anytype, comptime reports: fn (@TypeOf(context), usize, usize) Continuation) ProgressCallback {
        const Context = @TypeOf(context);
        checkContext(Context);
        return .{
            .callback = struct {
                fn advance(completed: usize, total: usize, state: ?*anyopaque) callconv(.c) Continuation {
                    if (Context == void) return reports({}, completed, total);
                    return reports(@ptrCast(@alignCast(state.?)), completed, total);
                }
            }.advance,
            .state = contextPointer(Context, context),
        };
    }
};

/// A per-key admission filter as the C interface carries it: a `usearch_predicate_t` and the
/// state forwarded into it. Build one with `Predicate.of`; the default admits everything.
pub const Predicate = struct {
    callback: ?PredicateFunction = null,
    state: ?*anyopaque = null,

    /// Wraps `admits` in a trampoline generated per call site. Pass `{}` for a predicate needing
    /// no context, and a single-item pointer otherwise.
    ///
    /// It runs once per scored candidate, far more often than `wanted`, and concurrently from the
    /// executor's threads - so it belongs on the caller's fast path and must tolerate that.
    pub fn of(context: anytype, comptime admits: fn (@TypeOf(context), Key) Admission) Predicate {
        const Context = @TypeOf(context);
        checkContext(Context);
        return .{
            .callback = struct {
                fn admit(key: Key, state: ?*anyopaque) callconv(.c) Admission {
                    if (Context == void) return admits({}, key);
                    return admits(@ptrCast(@alignCast(state.?)), key);
                }
            }.admit,
            .state = contextPointer(Context, context),
        };
    }
};

/// Rejects a context that cannot cross the FFI boundary. A callback's state travels as one
/// `*anyopaque`, so a slice or a many-item pointer has no single address to pass.
pub inline fn checkContext(comptime Context: type) void {
    if (Context == void) return;
    const info = @typeInfo(Context);
    if (info != .pointer or info.pointer.size != .one)
        @compileError(std.fmt.comptimePrint(
            "Context must be `void` or a single-item pointer like `&my_context`, got `{s}`",
            .{@typeName(Context)},
        ));
}

/// The pointer a callback travels with. A `void` context travels as null and is never read.
pub inline fn contextPointer(comptime Context: type, context: Context) ?*anyopaque {
    return if (Context == void) null else @ptrCast(@constCast(context));
}

/// Whether a call returns with its final status, or once submitted. Mirrors `usearch_completion_t`;
/// a receiver's type picks it, so no verb takes it as an argument.
pub const Completion = enum(c_int) {
    blocking = 0,
    deferred = 1,
};

/// What an executor does with its threads or streams once it goes idle. Mirrors `usearch_mood_t`.
pub const Mood = enum(c_int) {
    /// Park them; the next call pays a wake-up.
    parked = 0,
    /// Keep them spinning, trading power for the next call's latency.
    spinning = 1,

    /// The canonical spelling, as `.parked` spells "parked". Calls `usearch_mood_name`.
    pub fn name(self: Mood) [:0]const u8 {
        return std.mem.span(usearch_mood_name(self));
    }

    /// Reads a spelling back, or null when no mood spells it. Calls `usearch_mood_from_name`.
    pub fn fromName(text: []const u8) ?Mood {
        return spelled(Mood, usearch_mood_from_name(.of(text)), text);
    }
};

const usearch_executor_for_ram_t = extern struct {
    executor: *?*anyopaque,
    outcome: *Outcome,
    ram: usearch_ram_options_t,
    threads: usize,
    deferred_limit: usize,
};

const usearch_executor_on_pool_t = extern struct {
    executor: *?*anyopaque,
    outcome: *Outcome,
    ram: usearch_ram_options_t,
    pool: ?*anyopaque,
    deferred_limit: usize,
};

const usearch_executor_for_vram_t = extern struct {
    executor: *?*anyopaque,
    outcome: *Outcome,
    vram: usearch_vram_options_t,
    streams: ?[*]const ?*anyopaque,
    scratch_limit: usize,
    deferred_limit: usize,
};

const usearch_executor_for_nvme_t = extern struct {
    executor: *?*anyopaque,
    outcome: *Outcome,
    nvme: usearch_nvme_options_t,
    threads: usize,
    deferred_limit: usize,
};

const usearch_arena_init_t = extern struct {
    arena: *?*anyopaque,
    executor: ?*anyopaque,
    outcome: *Outcome,
};

const usearch_executor_drain_t = extern struct {
    executor: ?*anyopaque,
    outcome: *Outcome,
    mood: Mood,
};

/// Optional settings of `Executor.forRam`, mirroring `usearch_executor_for_ram_t`.
pub const ExecutorForRamConfig = struct {
    /// The memory domains of the indexes it will drive; empty derives every node the process may use.
    memory_domains: []const i32 = &.{},
    /// Threads to spawn; zero takes 7/8 of the cores those domains carry.
    threads: usize = 0,
    /// Deferred calls it tracks at once, sizing its deferred table; zero takes a default.
    deferred_limit: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `Executor.onPool`, mirroring `usearch_executor_on_pool_t`.
pub const ExecutorOnPoolConfig = struct {
    /// The memory domains of the indexes it will drive; empty derives every node the process may use.
    memory_domains: []const i32 = &.{},
    /// Deferred calls it tracks at once, sizing its deferred table; zero takes a default.
    deferred_limit: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `Executor.forVram`, mirroring `usearch_executor_for_vram_t`.
pub const ExecutorForVramConfig = struct {
    /// The devices of the indexes it will drive; empty derives one free device.
    devices: []const i32 = &.{},
    /// Borrowed CUDA streams, one per device in `devices` order; null creates its own.
    streams: ?[]const ?*anyopaque = null,
    /// The ceiling on scratch bytes per device; zero lets scratch grow as geometry asks.
    scratch_limit: usize = 0,
    /// Deferred calls it tracks at once, sizing its deferred table; zero takes a default.
    deferred_limit: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// Optional settings of `Executor.forNvme`, mirroring `usearch_executor_for_nvme_t`.
pub const ExecutorForNvmeConfig = struct {
    /// The devices of the indexes it will drive; empty leaves every shard where its manifest points.
    paths: []const BytesView = &.{},
    /// Threads to spawn, one submission ring apiece; zero derives from the devices' queue capabilities.
    threads: usize = 0,
    /// Deferred calls it tracks at once, sizing its deferred table; zero takes a default.
    deferred_limit: usize = 0,
    /// Where the call's final outcome, its static message included, is copied; null copies nothing.
    diagnostics: ?*Outcome = null,
};

/// The host threads or device streams that advance a call, and the traversal scratch each of them
/// walks with - everything a verb needs to run but an index does not own.
///
/// One executor serves any number of index handles, and its kind is fixed by the constructor that
/// produced it: it must match the residency of every handle it drives, or the call reports
/// `Error.Device`. A verb called on the index itself takes no executor and runs on the calling thread.
///
/// Calls on one executor run in submission order; handles stay freely concurrent - it is the
/// executor that orders, never the index. An executor without threads of its own holds a deferred
/// call until its wait, its next `drain` or blocking call, or the free of a handle it names.
pub const Executor = struct {
    /// The opaque `usearch_executor_t` the ABI hands back.
    handle: ?*anyopaque,
    /// The residency of the constructor that produced it, which places the arenas it makes.
    residency: Residency,

    /// Spawns host threads for RAM-resident indexes across the domains local to `memory_domains`,
    /// which should be the very slice the indexes were placed with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains carry, because full occupancy starves
    /// the ordered publication path. Calls `usearch_executor_for_ram`.
    pub fn forRam(config: ExecutorForRamConfig) Error!Executor {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_executor_for_ram_t = .{
            .executor = &handle,
            .outcome = &outcome,
            .ram = .of(config.memory_domains),
            .threads = config.threads,
            .deferred_limit = config.deferred_limit,
        };
        usearch_executor_for_ram(&arguments);
        return adopt(handle, .ram, outcome, config.diagnostics);
    }

    /// Creates a RAM executor over a borrowed ForkUnion `fu_pool_t`, so several libraries in one
    /// process share one set of threads; the pool must outlive the executor, and keeps its lender's
    /// mood. Calls `usearch_executor_on_pool`.
    pub fn onPool(pool: *anyopaque, config: ExecutorOnPoolConfig) Error!Executor {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_executor_on_pool_t = .{
            .executor = &handle,
            .outcome = &outcome,
            .ram = .of(config.memory_domains),
            .pool = pool,
            .deferred_limit = config.deferred_limit,
        };
        usearch_executor_on_pool(&arguments);
        return adopt(handle, .ram, outcome, config.diagnostics);
    }

    /// Spawns host threads for NVMe-resident indexes, one submission ring apiece, across the
    /// domains local to the devices `paths` names. Calls `usearch_executor_for_nvme`.
    pub fn forNvme(config: ExecutorForNvmeConfig) Error!Executor {
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_executor_for_nvme_t = .{
            .executor = &handle,
            .outcome = &outcome,
            .nvme = .of(config.paths),
            .threads = config.threads,
            .deferred_limit = config.deferred_limit,
        };
        usearch_executor_for_nvme(&arguments);
        return adopt(handle, .nvme, outcome, config.diagnostics);
    }

    /// Opens one submission stream on every named device, or borrows `streams`, one CUDA stream
    /// per device in `devices` order, all driven by whichever host thread submits.
    /// Calls `usearch_executor_for_vram`.
    pub fn forVram(config: ExecutorForVramConfig) Error!Executor {
        // C reads one stream per device and has no count of its own to check them against.
        if (config.streams) |borrowed| if (borrowed.len != config.devices.len)
            return refuseInto(config.diagnostics, .shape_mismatch, "The streams must number one per device");
        var handle: ?*anyopaque = null;
        var outcome: Outcome = .{};
        var arguments: usearch_executor_for_vram_t = .{
            .executor = &handle,
            .outcome = &outcome,
            .vram = .of(config.devices),
            .streams = if (config.streams) |borrowed| borrowed.ptr else null,
            .scratch_limit = config.scratch_limit,
            .deferred_limit = config.deferred_limit,
        };
        usearch_executor_for_vram(&arguments);
        return adopt(handle, .vram, outcome, config.diagnostics);
    }

    fn adopt(handle: ?*anyopaque, residency: Residency, outcome: Outcome, diagnostics: ?*Outcome) Error!Executor {
        errdefer usearch_executor_free(handle);
        try outcome.checkInto(diagnostics);
        return .{ .handle = handle, .residency = residency };
    }

    /// Waits for every call submitted before the drain began, then sets the mood the executor
    /// adopts whenever it next goes idle. Reports only what befell the executor itself.
    /// Calls `usearch_executor_drain`.
    pub fn drain(self: Executor, mood: Mood) Error!void {
        var outcome: Outcome = .{};
        var arguments: usearch_executor_drain_t = .{ .executor = self.handle, .outcome = &outcome, .mood = mood };
        usearch_executor_drain(&arguments);
        try outcome.check();
    }

    /// Drains every call, releases the scratch and the streams it owns, and frees the executor,
    /// which is invalid afterwards; the arenas it made keep their memory. Calls `usearch_executor_free`.
    pub fn deinit(self: *Executor) void {
        usearch_executor_free(self.handle);
        self.* = undefined;
    }
};

/// Memory the library writes every output whose length scales with a call into - matches, rows,
/// products, monoliths and cursor batches - mirroring `usearch_arena_t`.
///
/// Every verb landing such outputs borrows one as `*Arena`, blocking or deferred alike. Each call
/// rewinds it, so the views of the previous call refuse every read afterwards, and refills it,
/// growing only past its high-water mark, so a steady loop allocates once. An arena serves one call
/// at a time: a second use while a call holds it fails with `Error.Contended` and leaves that call's
/// views alone. A view finds its arena where the call left it, so an arena stays put, never copied,
/// while views over it live.
pub const Arena = struct {
    /// The opaque `usearch_arena_t` the ABI hands back; null once freed.
    handle: ?*anyopaque,
    /// The residency of the executor that made it: host pages, or a GPU's memory as `.vram`.
    residency: Residency,
    /// Calls it has served, which every view compares against its own. Binding-only.
    generation: usize = 0,

    /// Makes an arena where `executor` computes, decided once, or host pages for calls on the
    /// calling thread when null; it may outlive the executor. Calls `usearch_arena_init`.
    pub fn init(executor: ?Executor) Error!Arena {
        var arena: Arena = .{ .handle = null, .residency = if (executor) |chosen| chosen.residency else .ram };
        var outcome: Outcome = .{};
        var arguments: usearch_arena_init_t = .{
            .arena = &arena.handle,
            .executor = if (executor) |chosen| chosen.handle else null,
            .outcome = &outcome,
        };
        usearch_arena_init(&arguments);
        errdefer usearch_arena_free(arena.handle);
        try outcome.check();
        return arena;
    }

    /// Frees every output in it, waiting for every deferred call naming it, and leaves every view
    /// over it refusing reads. Calls `usearch_arena_free`.
    pub fn deinit(self: *Arena) void {
        usearch_arena_free(self.handle);
        self.handle = null;
        self.generation +%= 1;
    }

    /// The arena and call a view was written by. Binding-only.
    pub const Stamp = struct {
        arena: *const Arena,
        generation: usize,

        /// Refuses a read once the arena served another call or was freed with `Error.Uninitialized`,
        /// and a read of device memory with `Error.Device`.
        pub fn check(self: Stamp) Error!void {
            if (self.arena.generation != self.generation) return refuse(.uninitialized);
            if (self.arena.residency == .vram) return refuse(.device);
        }
    };
};

/// The stamp a call C has taken from `arena` carries, once a blocking call returns or a deferred one
/// is submitted: every call but a contended one rewound the arena, killing the previous call's views.
/// Binding-only.
pub fn served(arena: *Arena, status: Status) Arena.Stamp {
    if (status != .contended) arena.generation +%= 1;
    return .{ .arena = arena, .generation = arena.generation };
}

/// A submitted call, held in storage the caller places - on the stack, in a struct, or from an
/// allocator - which owns its outcome until the call is final, while the arena, the handles and the
/// slices it was handed stay borrowed until then.
///
/// `status` polls without blocking, `wait` blocks until the call is final and answers its result
/// or error - the same one however often it is asked - and `deinit` waits the same way. A submitted
/// `Deferred` stays put until `deinit`, and has one owner: `wait` and `deinit` never run on two
/// threads at once, while `status` reads from any. Binding-only; `wait` and `deinit` call
/// `usearch_outcome_wait`.
pub fn Deferred(comptime Call: type) type {
    return struct {
        const Self = @This();

        call: Call,
        answer: ?(Error!Call.Result) = null,

        /// Copies `call` into `self`, which stays put until final, and submits it. Binding-only.
        pub fn submit(self: *Self, call: Call, index: ?*anyopaque, executor: ?*anyopaque) void {
            self.* = .{ .call = call };
            self.call.launch(index, executor, .deferred);
        }

        /// `.deferred` until the call is final, then its final status. Calls `usearch_outcome_status`.
        pub fn status(self: *Self) Status {
            return usearch_outcome_status(&self.call.outcome);
        }

        /// The static detail of the final status; empty until the call is final, and on success.
        pub fn message(self: *Self) [:0]const u8 {
            return if (self.status() == .deferred) "" else self.call.outcome.text();
        }

        /// Blocks until the call is final, then answers its result, the same views a blocking call
        /// answers over the arena it borrowed. One thread waits at a time. Calls `usearch_outcome_wait`.
        pub fn wait(self: *Self) Error!Call.Result {
            if (self.answer) |answer| return answer;
            _ = usearch_outcome_wait(&self.call.outcome);
            self.answer = self.call.finish();
            return self.answer.?;
        }

        /// Blocks until the call is final; the storage is reusable afterwards, and the result's
        /// views stay as valid as the arena keeps them. Calls `usearch_outcome_wait`.
        pub fn deinit(self: *Self) void {
            _ = self.wait() catch {};
            self.call = undefined;
            self.answer = null;
        }
    };
}

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
    std.debug.assert(@sizeOf(Mood) == tag);
    std.debug.assert(@sizeOf(Completion) == tag);
    std.debug.assert(@sizeOf(Admission) == tag);
    std.debug.assert(@sizeOf(Continuation) == tag);
    std.debug.assert(@sizeOf(Distance) == 8);

    std.debug.assert(@sizeOf(Scalar.BF16) == 2);
    std.debug.assert(@sizeOf(Scalar.U1x8) == 1);

    std.debug.assert(@sizeOf(Outcome) == 2 * pointer);
    std.debug.assert(@offsetOf(Outcome, "status") == 0);
    std.debug.assert(@offsetOf(Outcome, "message") == pointer);

    std.debug.assert(@sizeOf(BytesView) == 2 * pointer);
    std.debug.assert(@offsetOf(BytesView, "data") == 0);
    std.debug.assert(@offsetOf(BytesView, "length") == pointer);

    std.debug.assert(@sizeOf(usearch_keys_view_t) == 2 * pointer);
    std.debug.assert(@offsetOf(usearch_keys_view_t, "stride") == pointer);

    std.debug.assert(@sizeOf(usearch_rows_view_t) == 4 * pointer);
    std.debug.assert(@offsetOf(usearch_rows_view_t, "stride") == pointer);
    std.debug.assert(@offsetOf(usearch_rows_view_t, "columns") == 2 * pointer);
    std.debug.assert(@offsetOf(usearch_rows_view_t, "scalar") == 3 * pointer);

    std.debug.assert(@sizeOf(usearch_matches_view_t) == 4 * pointer);
    std.debug.assert(@offsetOf(usearch_matches_view_t, "scores") == pointer);
    std.debug.assert(@offsetOf(usearch_matches_view_t, "counts") == 2 * pointer);
    std.debug.assert(@offsetOf(usearch_matches_view_t, "columns") == 3 * pointer);

    std.debug.assert(@sizeOf(usearch_matches_tape_view_t) == 3 * pointer);
    std.debug.assert(@offsetOf(usearch_matches_tape_view_t, "offsets") == 2 * pointer);

    std.debug.assert(@sizeOf(usearch_ram_options_t) == 2 * pointer);
    std.debug.assert(@sizeOf(usearch_vram_options_t) == 2 * pointer);
    std.debug.assert(@sizeOf(usearch_nvme_options_t) == 2 * pointer);

    std.debug.assert(@sizeOf(usearch_executor_for_ram_t) == 6 * pointer);
    std.debug.assert(@offsetOf(usearch_executor_for_ram_t, "ram") == 2 * pointer);
    std.debug.assert(@offsetOf(usearch_executor_for_ram_t, "threads") == 4 * pointer);
    std.debug.assert(@offsetOf(usearch_executor_for_ram_t, "deferred_limit") == 5 * pointer);
    std.debug.assert(@sizeOf(usearch_executor_on_pool_t) == 6 * pointer);
    std.debug.assert(@offsetOf(usearch_executor_on_pool_t, "pool") == 4 * pointer);
    std.debug.assert(@sizeOf(usearch_executor_for_vram_t) == 7 * pointer);
    std.debug.assert(@offsetOf(usearch_executor_for_vram_t, "streams") == 4 * pointer);
    std.debug.assert(@sizeOf(usearch_executor_for_nvme_t) == 6 * pointer);
    std.debug.assert(@sizeOf(usearch_executor_drain_t) == 3 * pointer);
    std.debug.assert(@sizeOf(usearch_arena_init_t) == 3 * pointer);
    std.debug.assert(@offsetOf(usearch_arena_init_t, "outcome") == 2 * pointer);
    std.debug.assert(@offsetOf(usearch_executor_drain_t, "mood") == 2 * pointer);
}

/// The library's version, as a static null-terminated string. Calls `usearch_version_name`.
pub fn version() [:0]const u8 {
    return std.mem.span(usearch_version_name());
}

/// The hardware capabilities this binary was compiled with, comma-separated.
/// Calls `usearch_capabilities_compiled`.
pub fn capabilitiesCompiled() [:0]const u8 {
    return std.mem.span(usearch_capabilities_compiled());
}

/// The hardware capabilities the current machine offers at runtime, comma-separated.
/// Calls `usearch_capabilities_available`.
pub fn capabilitiesAvailable() [:0]const u8 {
    return std.mem.span(usearch_capabilities_available());
}

/// Every named tag of `Enum` with its underscores spelled as dashes, which is how the C tables spell them.
fn expectSpellings(comptime Enum: type) !void {
    inline for (@typeInfo(Enum).@"enum".fields) |field| {
        const value: Enum = @enumFromInt(field.value);
        var dashed: [field.name.len]u8 = field.name[0..].*;
        std.mem.replaceScalar(u8, &dashed, '_', '-');
        try std.testing.expectEqualStrings(&dashed, value.name());
        try std.testing.expectEqual(@as(?Enum, value), Enum.fromName(&dashed));
    }
}

test "statuses lift into the error set one for one" {
    const success: Outcome = .{};
    try success.check();
    try std.testing.expectEqualStrings("", success.text());

    const contended: Outcome = .{ .status = .contended, .message = "another thread is driving it" };
    try std.testing.expectError(Error.Contended, contended.check());
    try std.testing.expectEqualStrings("another thread is driving it", contended.text());
    try std.testing.expectError(Error.InvalidArgument, (Outcome{ .status = .invalid_argument }).check());
    try std.testing.expectError(Error.Unsupported, (Outcome{ .status = .unsupported }).check());

    // A deferred status reaching a final check is a defect rather than a state.
    try std.testing.expectError(Error.Internal, (Outcome{ .status = .deferred }).check());

    // A status this build has no name for is still a failure, named as C names it.
    const unnamed: Outcome = .{ .status = @enumFromInt(9999) };
    try std.testing.expectError(Error.InvalidArgument, unnamed.check());
}

test "every enumerator keeps the value the linked library spells it by" {
    // A renumbered header shifts the spellings, which is what this catches and a size check cannot.
    try expectSpellings(Status);
    try expectSpellings(Scalar);
    try expectSpellings(Residency);
    try expectSpellings(Mood);
    try std.testing.expectEqualStrings("unrecognized", (@as(Status, @enumFromInt(9999))).name());
    try std.testing.expectEqual(@as(?Status, null), Status.fromName("pending"));
    try std.testing.expectEqual(@as(?Mood, null), Mood.fromName("chill"));
    try std.testing.expectEqual(@as(?Residency, null), Residency.fromName("disk"));
    try std.testing.expectEqual(@as(?Scalar, null), Scalar.fromName("f128"));
}

test "integer identities park in the low eight bytes" {
    const key = Key.fromInteger(42);
    try std.testing.expectEqual(@as(?u64, 42), key.asInteger());
    try std.testing.expectEqualSlices(u8, &(.{0} ** 15 ++ .{0x2a}), &key.bytes);

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

    // Every one of the sixteen bytes survives the round trip, in order.
    var wide: Key = undefined;
    for (&wide.bytes, 0..) |*byte, position| byte.* = @intCast(position * 16 + position);
    const rendered = try std.fmt.bufPrint(&buffer, "{f}", .{wide});
    try std.testing.expectEqualStrings("00112233-4455-6677-8899-aabbccddeeff", rendered);
    try std.testing.expectEqual(wide, try Key.parse(rendered));

    // Dashes may sit anywhere or nowhere, and either case of digit reads the same.
    try std.testing.expectEqual(wide, try Key.parse("00112233445566778899AABBCCDDEEFF"));
    try std.testing.expectEqual(wide, try Key.parse("0-0-1-12233445566778899aabbccddeeff"));

    try std.testing.expectError(Error.Malformed, Key.parse(""));
    try std.testing.expectError(Error.Malformed, Key.parse("00112233-4455-6677-8899-aabbccddeef"));
    try std.testing.expectError(Error.Malformed, Key.parse("00112233-4455-6677-8899-aabbccddeeff0"));
    try std.testing.expectError(Error.Malformed, Key.parse("00112233-4455-6677-8899-aabbccddeefg"));
}

test "packed kinds count dimensions, never bytes" {
    const bits = [_]Scalar.U1x8{ .{ .bits = 0x0f }, .{ .bits = 0xf0 }, .{ .bits = 0xff }, .{ .bits = 0x00 } };
    const rows = try RowsView.of(Scalar.U1x8, &bits, 16);
    try std.testing.expectEqual(RowsView{ .data = &bits, .rows = 2, .columns = 16, .stride = 2, .scalar = .u1x8 }, rows);
    try std.testing.expectEqual(@as(usize, 32), (try RowsView.ofRow(Scalar.U1x8, &bits)).columns);

    // A width that does not fill whole packed values is no shape at all.
    try std.testing.expectError(Error.ShapeMismatch, RowsView.of(Scalar.U1x8, &bits, 12));
    try std.testing.expectError(Error.ShapeMismatch, RowsView.of(Scalar.U1x8, &bits, 24));
}

test { // analyze every declaration, the methods on each handle included
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Executor);
    std.testing.refAllDecls(Arena);
    std.testing.refAllDecls(Outcome);
    std.testing.refAllDecls(Key);
    std.testing.refAllDecls(Scalar);
    std.testing.refAllDecls(BytesView);
}
