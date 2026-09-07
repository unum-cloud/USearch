# USearch for Zig

The Zig SDK binds the stable C interface with hand-written `extern fn` declarations rather than `@cImport`, so a dependent needs no include path of its own - only the library.
It exposes the dense modality alone, and every verb it binds is one verb of the C interface, named the same way.
A `comptime` table checks every descriptor's size and field offsets against the layout the ABI fixes, so writing the interface out by hand is checked rather than trusted.

## Installation

```sh
zig fetch --save git+https://github.com/unum-cloud/USearch
```

Then wire the module into your own `build.zig`:

```zig
const usearch = b.dependency("usearch", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("usearch", usearch.module("usearch"));
```

No C or C++ is compiled here: the engine is CMake's to build, and this module consumes it through the C headers alone.
`-Dlib-dir` names a directory holding the library and may be repeated; left alone it points at `build_release/c`, where an in-tree CMake build leaves it.
The shared `usearch_c` is what gets linked, since it resolves its own dependencies and one `-l` finishes the link.

A bare `zig build` compiles every declaration without the engine on disk, which makes it a compile check; `zig build test` needs the library.

## Quickstart

Every batch input and output is a strided matrix - a base pointer, a row count, a column count, and a byte stride - so a memory-mapped dataset feeds the index with no reshape and no copy.
An index owns rows; an `Executor` owns the threads or streams a fanned-out verb runs on, and every batch verb takes one.

```zig
const usearch = @import("usearch");

const index = try usearch.DenseIndex.init(&.{
    .dimensions = 4,
    .metric = .angular,
    .quantization = .i8,
});
defer index.deinit();
try index.reserve(1_000);

const executor = try usearch.Executor.forRam(&.{}, 0);
defer executor.deinit();

const keys = [_]usearch.Key{ .fromInteger(42), .fromInteger(43) };
const slab = [_]f32{ 0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2 };
_ = try index.add(executor, &keys, try .of(f32, &slab, 4), .{}, null);

const query = [_]f32{ 0.2, 0.1, 0.2, 0.1 };
var found: [10]usearch.Key = undefined;
var scores: [10]usearch.Distance = undefined;
var counts: [1]usize = undefined;
try index.search(executor, try .of(f32, &query, 4), try .of(&found, &scores, &counts), .{}, null);
```

One executor serves any number of indexes, and its kind must match the residency their handles were built for.
Zero threads takes 7/8 of the cores, since full occupancy starves the ordered publication path; `forVram` takes no thread count at all, driving one stream per device from the calling thread.

The scalar type of the input is independent of the stored quantization - the `f32` slab above lands in an `i8` index - so nothing needs converting before insertion.
`MatrixView.of` reads a packed slab; a dataset whose rows are padded to a wider pitch fills the struct directly, setting `stride_bytes` to the pitch.
Lengths that cannot be read as a whole number of rows are refused with `Error.ShapeMismatch` rather than asserted, so the check survives `ReleaseFast`.

## Search Allocates Nothing

There is one search verb per form and both write into buffers you own, so a benchmark loop allocates once and never again.
`MatchesSpan.of` derives the strides and the "k" from the slab lengths, so the space reserved and the count asked for cannot disagree.

```zig
var keys = try allocator.alloc(usearch.Key, queries_count * 10);
var scores = try allocator.alloc(usearch.Distance, queries_count * 10);
var counts = try allocator.alloc(usize, queries_count);
var stats: usearch.TraversalStats = .{};

try index.search(executor, try .of(f32, queries, 768), try .of(keys, scores, counts),
    .{ .expansion_factor = 128 }, &stats);
std.debug.print("{d} distances computed\n", .{stats.computed_distances});
```

The candidate-pool breadth rides the call rather than the handle, so a recall sweep is a loop over expansion factors against one immutable index, with no mutation and no synchronization between points.

## Filtering With Predicates

`searchIf` admits only the keys a predicate approves, and non-matching entries still route the traversal, so recall does not collapse at low selectivity.
The predicate is a `comptime` function rather than a closure, so the trampoline that crosses the boundary is generated per call site and nothing is boxed.
Its context travels as one pointer: pass `{}` for a predicate needing none, and a single-item pointer otherwise.

```zig
const admits = struct {
    fn call(_: void, key: usearch.Key) usearch.Admission {
        const value = key.asInteger() orelse return .reject;
        return if (value % 2 == 0) .admit else .reject;
    }
}.call;

try index.searchIf(executor, try .of(f32, queries, 768), try .of(keys, scores, counts), .{}, null, {}, admits);
```

It runs once per scored candidate, far more often than the "k" asked for, and concurrently from the executor's threads - so it belongs on your fast path and must tolerate that.

## Residency

Where the payload lives is a tagged union, not a flag, and it decides what loading a checkpoint means: materialize into RAM, upload to a device, or open in place on disk.
It also names the hardware its shards are placed across - one shard per memory domain, device, or path - and an empty slice derives one from the machine.

```zig
const geometry = usearch.Options{ .dimensions = 768 };

// One graph per socket.
const in_memory = geometry.placed(.{ .ram = &.{ 0, 1 } });
// Sharded across the devices, never replicated.
const on_device = geometry.placed(.{ .vram = &.{ 0, 1 } });
// Opened in place, read-only, and sized past RAM.
const on_disk = geometry.placed(.{ .nvme = &.{.of("/mnt/nvme0")} });
```

`placed` sets the tag and the union arm together, so the two cannot disagree.
An NVMe handle opens a checkpoint in place and is read-only, so every mutating verb on it fails with `Error.ReadOnly`.
Bigger indexes come from merging, never mutating: shards build in RAM, serialize, and join a manifest.

## Serialization

The path extension picks the layout - a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.

```zig
try index.save(executor, "index.safegraphs");
try index.load(executor, "index.safegraphs");

const geometry = try usearch.metadata("index.safegraphs"); // without loading a shard
```

Checkpoints are residency-portable: one file materializes into RAM, uploads to a device, or opens in place on disk, decided by the handle that loads it.
In-memory buffers hold single monoliths only, through `saveBuffer` and `loadBuffer`.

## Metrics

Every metric derives from a single inner-product kernel with pre-computed magnitudes.

- `.inner` - inner product, and the only kind the linear-algebra verbs accept.
- `.angular` - cosine distance, over pre-computed magnitudes.
- `.euclidean` - squared euclidean distance.
- `.hamming` - bit-level, over the packed sub-byte kinds.
- `.jaccard` - bit-level, over the packed sub-byte kinds.

`scoredBy` points the options at a metric of your own, setting the kind, the quantization, and the callback together so the four cannot disagree.
Its trampoline is generated per call site like `searchIf`'s, so the scoring function reads two typed slices rather than two `anyopaque` pointers, and its context travels as one pointer.

```zig
const score = struct {
    fn call(_: void, left: []const f32, right: []const f32) usearch.Distance {
        var total: f32 = 0;
        for (left, right) |first, second| total += (first - second) * (first - second);
        return total;
    }
}.call;

const options = (usearch.Options{ .dimensions = 768 }).scoredBy(f32, {}, score);
```

It is called concurrently from the executor's threads, and forfeits the SIMD kernel chains and the magnitude cache both, so reach for a built-in kind wherever one fits.

## Linear Algebra Over The Graph

Three verbs treat the index as a sparse operand rather than as a kNN store, each folding its arithmetic into the graph walk instead of materializing a match list.
All three read logits rather than distances, so they refuse anything but an inner-product index with `Error.ConfigMismatch`.

`multiply` is a sparse matrix product: every left row is walked against the index, and the products land in the same `MatchesSpan` a search writes, `keys` naming each pair's indexed row.

```zig
try index.multiply(executor, try .of(f32, left, 768), try .of(keys, scores, counts), .{}, null);
```

`maxsim` is late interaction - every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `values_per_key` above one exists for, and the outputs are sized over documents rather than rows.

```zig
try index.maxsim(executor, try .of(f32, tokens, 768), tokens_per_query, try .of(keys, scores, counts), .{}, null);
```

`attend` is softmax-weighted attention, gathering each value row exactly once and returning outputs already divided by their partition.
An entry's key is its row offset into the value table, so a context inserted in position order needs no mapping, and a carry of one `SoftmaxState` per query row folds this branch into another over the same queries.

```zig
try index.attend(executor, try .of(f32, queries, 768), try .of(f32, values, 768),
    try .of(f32, context, 768), carry, .{}, null);
```

## Refinement

`refine` re-selects the neighbors of every stored entry against the finished graph and returns how many nodes changed edges, so a cheap insertion pass followed by one refinement reaches a better graph in less total time than one expensive insertion pass.
Zero means the graph has settled, and searches may run throughout.

```zig
while (true) {
    const changed = try index.refine(executor, .{ .expansion_factor = 128 }, null);
    if (changed == 0) break;
}
```

## Streaming and Iteration

A finder is a resumable best-first traversal, so callers that post-filter aggressively pull until satisfied instead of guessing the "k" upfront.

```zig
const finder = try index.find(executor, try .ofRow(f32, &query));
defer finder.deinit();

var keys: [16]usearch.Key = undefined;
var distances: [16]usearch.Distance = undefined;
while (true) {
    const pulled = try finder.next(&keys, &distances);
    if (pulled == 0) break;
    for (keys[0..pulled], distances[0..pulled]) |key, distance| std.debug.print("{any} at {d}\n", .{ key, distance });
}
```

A cursor walks the index contents as a sequential row scan behind a publication watermark, so concurrent inserts never appear mid-iteration.

```zig
const cursor = try index.cursor();
defer cursor.deinit();

var keys: [1024]usearch.Key = undefined;
var vectors: [1024 * 768]f32 = undefined;
while (true) {
    const written = try cursor.next(&keys, try .of(f32, &vectors, 768));
    if (written == 0) break;
}
```

## Keys

A key is 128 bits of opaque identity, and integer identities park in the low eight bytes, so a UUID rendering and an integer identity sort alike.
Keys need not be unique: several entries may share one, and `count` reports how many.

```zig
const key: usearch.Key = .fromInteger(42);
const value = key.asInteger(); // 42, or null when the high eight bytes are set

std.debug.print("{f}\n", .{key}); // 00000000-0000-0000-0000-00000000002a
const parsed = try usearch.Key.parse("00000000-0000-0000-0000-00000000002a");
```

`format` renders the canonical lowercase dashed UUID, so a key prints through `{f}` and through any writer.
`parse` reads that rendering back, taking dashes anywhere or nowhere and either case of digit, and refusing anything else with `Error.MalformedInput`.

## Errors

Every verb answers `Error!T`, and a shape this binding refuses before the boundary reports the status the engine would have reported for the same fault - so a locally-rejected call is indistinguishable from an engine-rejected one.
The engine's static detail message rides `Outcome`, which stays internal to the boundary.
One set covers every verb rather than a narrower one per verb, since nothing in the ABI bounds which status a given call may report.

## Introspection

```zig
std.debug.print("{d} entries in {d} bytes\n", .{ try index.size(), try index.memoryUsage() });
std.debug.print("dispatched to {s}\n", .{try index.capabilities()});
std.debug.print("compiled for {s}\n", .{usearch.capabilitiesCompiled()});
std.debug.print("available here {s}\n", .{usearch.capabilitiesAvailable()});
```
