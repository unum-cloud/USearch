//! The ragged tapes the sequence and shape indexes read and write: the layout of their offsets, the
//! viewed input batch, and the packed entries a `get` writes into an arena.

const types = @import("types.zig");

const Error = types.Error;
const Scalar = types.Scalar;
const outputSlice = types.outputSlice;
const refuse = types.refuse;

/// How wide a tape's offsets are, naming the member of its offsets to read.
/// Mirrors `usearch_tape_layout_t`.
pub const TapeLayout = enum(c_int) {
    /// Apache Arrow `Binary` or `List`, with 32-bit offsets.
    u32 = 0,
    /// Apache Arrow `LargeBinary` or `LargeList`, the same with 64-bit offsets.
    u64 = 1,
};

/// A ragged run as Apache Arrow lays one out, as the C interface reads an input or writes an output
/// into an arena. Mirrors `usearch_tape_view_t`; an input is built from a `TapeView`, whose entry
/// count becomes the call's count.
pub const usearch_tape_view_t = extern struct {
    data: ?*const anyopaque = null,
    layout: TapeLayout = .u64,
    offsets: extern union { u32: ?[*]const u32, u64: ?[*]const u64 } = .{ .u64 = null },
    scalar: Scalar = .unknown,

    /// The tape the C interface reads; the entry count travels separately, as the call's count.
    pub fn of(tape: TapeView) usearch_tape_view_t {
        return .{
            .data = tape.data,
            .layout = tape.layout,
            .offsets = switch (tape.layout) {
                .u32 => .{ .u32 = @ptrCast(@alignCast(tape.offsets)) },
                .u64 => .{ .u64 = @ptrCast(@alignCast(tape.offsets)) },
            },
            .scalar = tape.scalar,
        };
    }
};

/// A viewed ragged batch - the currency the sequence and shape verbs read, mirroring
/// `usearch_tape_view_t` with the entry count it leaves to the call.
///
/// Elements lie end to end, and one more offset than entries surrounds them, so entry `i` spans
/// elements `offsets[i]` up to `offsets[i + 1]`, as an Arrow column holds them. What an element is
/// belongs to the modality: a byte of a sequence, or a point of a shape.
pub const TapeView = struct {
    /// First element, which every offset counts from.
    data: ?*const anyopaque,
    /// Entries the tape holds, one fewer than its offsets.
    entries: usize,
    /// Which width `offsets` holds.
    layout: TapeLayout,
    /// The `entries + 1` non-decreasing offsets, in elements.
    offsets: ?*const anyopaque,
    /// The scalar type the elements hold, converted on the way in where it differs.
    scalar: Scalar,

    /// Views an Arrow-shaped pair of runs, `Offset` being `u32` or `u64` - `[]const u8` with its
    /// offsets for sequences, `[]const [3]f32` with point offsets for shapes.
    ///
    /// No offsets, or offsets reaching past the elements, are refused with `Error.ShapeMismatch`,
    /// since the interface reads no element count to catch the overrun.
    pub fn of(comptime Element: type, comptime Offset: type, elements: []const Element, offsets: []const Offset) Error!TapeView {
        const Value = switch (@typeInfo(Element)) {
            .array => |array| array.child,
            else => Element,
        };
        if (offsets.len == 0 or offsets[offsets.len - 1] > elements.len) return refuse(.shape_mismatch);
        return .{
            .data = elements.ptr,
            .entries = offsets.len - 1,
            .layout = switch (Offset) {
                u32 => .u32,
                u64 => .u64,
                else => @compileError("A tape's offsets are `u32` or `u64`, got `" ++ @typeName(Offset) ++ "`"),
            },
            .offsets = offsets.ptr,
            .scalar = comptime Scalar.of(Value),
        };
    }
};

/// The entries a `get` wrote packed into an arena as one tape of `Element`, its offsets as wide
/// as `usize`, readable until the arena's next call. Mirrors `usearch_tape_view_t`.
pub fn TapeEntries(comptime Element: type) type {
    return struct {
        const Self = @This();

        view: usearch_tape_view_t = .{},
        /// Entries written; zero for an absent key.
        written: usize = 0,

        /// The `written + 1` offsets, in elements; empty for an absent key.
        pub fn offsets(self: Self) []const usize {
            if (self.written == 0) return &.{};
            return outputSlice(usize, @ptrCast(@alignCast(self.view.offsets.u64)), 0, self.written + 1);
        }

        /// One entry's elements, `position` below `written`.
        pub fn entry(self: Self, position: usize) []const Element {
            const tape = self.offsets();
            return outputSlice(Element, @ptrCast(@alignCast(self.view.data)), tape[position], tape[position + 1] - tape[position]);
        }

        /// Every entry's elements, end to end.
        pub fn elements(self: Self) []const Element {
            const tape = self.offsets();
            return if (tape.len == 0) &.{} else outputSlice(Element, @ptrCast(@alignCast(self.view.data)), 0, tape[tape.len - 1]);
        }
    };
}
