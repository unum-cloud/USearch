# USearch for Zig

The Zig SDK binds the stable C ABI in `c/usearch/` with hand-written `extern fn` declarations rather than `@cImport`, so a dependent needs no include path of its own, only the library.
It requires Zig 0.16 or later and links the engine's shared library, built by CMake.
It exposes dense indexes, and the sequence, shape and geospatial indexes beside them, each with the same receivers, plans and arenas.

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

## Quickstart

An index owns entries; an `Executor` owns the threads or streams a verb runs on; an `Arena` holds every output whose length scales with the call.

```zig
const usearch = @import("usearch");

var index = try usearch.DenseIndex.initRam(.{
    .dimensions = 4,
    .metric = .angular,
    .quantization = .i8,
}, &.{}); // one shard per NUMA node, or the memory domains you name
defer index.deinit();
try index.reserve(1_000);

const keys = [_]usearch.Key{ .fromInteger(42), .fromInteger(43) };
const slab = [_]f32{ 0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2 };
try index.insert(&keys, try .of(f32, &slab, 4), .{});

const query = [_]f32{ 0.2, 0.1, 0.2, 0.1 };
var arena = try usearch.Arena.init(null); // host pages, for calls on the calling thread
defer arena.deinit();
const matches = try index.search(try .of(f32, &query, 4), 10, &arena, .{}); // up to 10 matches per query
for (matches.query(0).keys) |key| std.debug.print("{f}\n", .{key});

var executor = try usearch.Executor.forRam(.{});
defer executor.deinit();
_ = try index.on(executor).search(try .of(f32, &query, 4), 10, &arena, .{}); // the same verb, on an executor
```

## Receivers & Deferred Calls

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
The index runs it on the calling thread, `index.on(executor)` blocks on an executor, refused while another call holds it, and `index.deferred(executor, &call)` submits it there into a `usearch.Deferred` you hold:

```zig
const here = try index.search(queries, 10, &arena, .{});               // the calling thread
const there = try index.on(executor).search(queries, 10, &other, .{}); // blocking, on the executor
var call: usearch.Deferred(usearch.Matches) = undefined;
try index.deferred(executor, &call).search(queries, 10, &spare, .{});  // submitted, into `call`
defer call.deinit();
```

`DenseIndex.On` and `DenseIndex.DeferredOn(Result)` offer every verb the index does, with the same arguments, the `*Arena` included, because one generic writes each verb once for both; a deferred receiver compiles only the verbs answering the result its storage holds.
Constructors address no index yet, so no receiver offers them; the ones the C interface lets fan out take an `executor` in their config, and run on the calling thread without one.

A `Deferred` owns what the library reports into until its call is final, the outcome and the result slot the call fills, and reads the rest in place from memory the caller keeps alive, as a blocking call does.
The handles it names, such as the index, the executor, the arena, a cursor or a stream, need nothing more: freeing one waits for every deferred call naming it.
The plain memory it reads, such as the rows, the keys, the path, the carry and the handlers' contexts, stays yours, and must outlive the call.

```zig
var call: usearch.Deferred(usearch.DenseVectors(f32)) = undefined;
try index.deferred(executor, &call).get(f32, key, &arena);
defer call.deinit();
if (call.status() == .deferred) {
    // other work
}
const vectors = try call.wait(); // the view a blocking `get` answers, valid until the next call given `arena`
```

The storage is yours to place, so submitting allocates nothing: a stack slot, a field, or an array of them for a batch of submitted calls.
The library reports into the storage, so a submitted `Deferred` stays put until `deinit`; a heap slot comes from `allocator.create` and is destroyed after `deinit`.
Zig has no async keyword left to lean on, so a deferred call is waited on and nothing else: `status` reads the outcome with acquire ordering and never blocks, and `wait` blocks in the library until the call is final and answers its result or error, read from the final outcome however often it is asked.
`deinit` waits the same way, and the slot can then hold another call.
A `Deferred` has one owner: `wait` and `deinit` never run on two threads at once, while `status` reads from any.
`message` answers the static detail of the final status, empty until the call is final and on success.
One call holds an executor at a time, so a second call finding it held finishes at once with `Error.Contended`, and nothing queues.
A deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling `status` alone finishes it, and its progress handler and predicate run on the pool's threads; a single thread, WebAssembly or a VRAM executor runs it at submission.
Calls depending on each other are ordered by waiting on the first before submitting the second.
A shape only the binding can check, like a key run longer than its rows, is refused before anything is submitted, so the submitting call itself fails and the slot holds nothing to `deinit`.

## Executors

One executor serves any number of indexes, and its kind must match the residency their handles were built for.
There is a constructor per C one, each taking a config of exactly what its C structure takes:

```zig
var sockets = try usearch.Executor.forRam(.{ .memory_domains = &.{ 0, 1 }, .threads = 0 });
var gpus = try usearch.Executor.forVram(.{ .devices = &.{ 0, 1 }, .streams = null, .scratch_limit = 0 });
var drives = try usearch.Executor.forNvme(.{ .paths = &.{.of("/mnt/nvme0")}, .threads = 0 });
```

Zero threads takes 7/8 of the cores, since full occupancy starves the ordered publication path; `forVram` takes no thread count at all, driving from the calling thread one stream per device, which the executor opens and destroys with itself; your own CUDA streams answer `unsupported` until they are wired.
`drain(mood)` waits until no call holds the executor, then lets the threads go `.parked` or stay `.spinning` once idle.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`Arena.init(executor)` places it where that executor computes, decided once, and `Arena.init(null)` makes host pages for the calling thread.
Each call rewinds its arena and grows it only past its high-water mark, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
A view read after a later call took its arena fails as `Contended`.
A slice or segment an accessor returned is valid only until the arena's next call.
Every output is a host slice, so `Arena.init` refuses with `error.Device` an executor whose memory the host may not read; `describe` reports where an arena lives, its vendor and whether the host may read it.
It grows and frees on that executor's stream, so it serves only calls on that executor, any other answering `Error.Device`, and is freed before it, like every arena before its executor.

```zig
var arena = try usearch.Arena.init(executor); // `Arena.init(null)` is host pages for the calling thread
defer arena.deinit();
for (batches) |batch| {
    const matches = try index.on(executor).search(batch, 10, &arena, .{}); // rewinds: the last batch's matches are gone
    total += matches.query(0).scores[0];
}

var found = try usearch.Arena.init(null); // two live results, two arenas
defer found.deinit();
var fetched = try usearch.Arena.init(null);
defer fetched.deinit();
const nearest = try index.search(query, 10, &found, .{});
for (nearest.query(0).keys) |key| rerank(key, try index.get(f32, key, &fetched));

var calls: [shards.len]usearch.Deferred(usearch.Matches) = undefined;
for (&calls, shards, executors, &arenas) |*call, shard, pool, *own|
    try shard.deferred(pool, call).search(queries, 10, own, .{}); // one executor and one arena per submitted call
defer for (&calls) |*call| call.deinit();
for (&calls) |*call| consume(try call.wait());
```

A call that finds its arena serving another call finishes with `Error.Contended` and rewinds nothing, so the holder's views stay readable.
A call that fails for any other reason still rewinds it.
Only fixed-size reports like `DenseTraversalStats` stay your fields.

## Keys

A key is 128 bits of opaque identity, and integer identities park in the low eight bytes, so a UUID rendering and an integer identity sort alike.
Keys need not be unique: several entries may share one, `count` reports how many, `contains` whether any, and `get` returns their vectors.

```zig
const key: usearch.Key = .fromInteger(42);
const value = key.asInteger(); // 42, or null when the high eight bytes are set

std.debug.print("{f}\n", .{key}); // 00000000-0000-0000-0000-00000000002a
const parsed = try usearch.Key.parse("00000000-0000-0000-0000-00000000002a");
```

`format` renders the canonical lowercase dashed UUID, so a key prints through `{f}` and through any writer.
`parse` reads that rendering back, taking dashes anywhere or nowhere and either case of digit, and refusing anything else with `Error.Malformed`.

## Errors

Every verb answers `Error!T`, one error per final status of the C interface.
The engine checks every shape it can see; the binding checks only what never crosses the boundary, like a key run longer than its rows, and reports the status the engine would have.
The engine's static detail message rides an `Outcome` you pass: `diagnosed(&outcome)` on the index or a blocking receiver copies every call's final outcome into it, a constructor's config takes one as `diagnostics`, and a `Deferred` answers its own through `message`.
The binding's own refusals land there too, with a message of their own.
One set covers every verb rather than a narrower one per verb, since nothing in the ABI bounds which status a given call may report.

```zig
var outcome: usearch.Outcome = .{};
_ = index.on(executor).diagnosed(&outcome).search(queries, 10, &arena, .{}) catch |failure| {
    std.log.err("{s}: {s}", .{ @errorName(failure), outcome.text() });
    return failure;
};
var copy = try usearch.DenseIndex.loadRam("index.safegraphs", .{ .diagnostics = &outcome });
```

`Status`, `Scalar`, `Mood`, `Residency` and `DenseMetric` carry the C enumerators' values, and their `name` and `fromName` spell them through the C name tables, `fromName` answering null for text no table spells.

```zig
std.debug.print("{s}\n", .{usearch.Status.contended.name()}); // contended
const metric = usearch.DenseMetric.fromName("inner") orelse return error.UnknownMetric;
```

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each names the hardware its shards are placed across, one shard per memory domain, device or path, and an empty slice derives one from the machine.

```zig
// One graph per socket.
var in_memory = try usearch.DenseIndex.initRam(.{ .dimensions = 768 }, &.{ 0, 1 });
// Sharded across the devices, never replicated.
var on_device = try usearch.DenseIndex.initVram(.{ .dimensions = 768 }, &.{ 0, 1 });
// Served in place, read-only, and sized past RAM.
var on_disk = try usearch.DenseIndex.viewNvme("index.safegraphs.json", .{
    .executor = nvme_executor,
    .paths = &.{.of("/mnt/nvme0")},
    .key_table_domains = &.{0},
});
```

An NVMe index is never initialized, only viewed in place, and it is read-only, so every mutating verb on it fails with `Error.ReadOnly`.
`key_table_domains` holds a key table in RAM beside the view; without one, `count`, `get`, `sample`, MaxSim and hinted searches answer `Error.Unsupported`.
`loadNvme` takes the same config and opens a manifest writable, so `refine` rewrites its parts one at a time, while a lone monolith answers `Error.Unsupported`.

## Serialization

The path extension picks the layout: a `.safegraphs` path is one monolith, and a `.safegraphs.json` path is a manifest cataloging many.

```zig
try index.on(executor).save("index.safegraphs", .{});
var copy = try usearch.DenseIndex.loadRam("index.safegraphs", .{ .executor = executor, .values_per_key = 1 });
var uploaded = try usearch.DenseIndex.loadVram("index.safegraphs", .{ .executor = gpu, .devices = &.{0} });
var mapped = try usearch.DenseIndex.viewRam("index.safegraphs"); // read-only, served in place

const metadata = try usearch.DenseIndex.metadataRead("index.safegraphs", .{}); // without loading a shard
```

The result holds only the four fields a checkpoint froze, so `values_per_key` and the placement stay yours to set in the load config.

Checkpoints are residency-portable: one file copies into RAM, uploads to a device, or is served in place on disk, decided by the constructor that reads it.
In-memory buffers hold single monoliths only.
`saveBuffer` lands one in an `Arena`, since only the call knows its length, and `loadRamBuffer` or `loadVramBuffer` reads it back.

```zig
const monolith = try index.saveBuffer(&arena, .{});
var restored = try usearch.DenseIndex.loadRamBuffer(monolith.bytes(), .{});
```

## Dense Indexes

### Rows & Scalars

Every batch input is a run of rows given by a base pointer, a row count, a column count, and a byte stride, so a memory-mapped dataset feeds the index with no reshape and no copy.
The scalar type of the input is independent of the stored quantization, so the `f32` slab of the Quickstart lands in an `i8` index and nothing needs converting before insertion.
`RowsView.of` reads a packed slab; a dataset whose rows are padded to a wider pitch fills the struct directly, setting `stride` to the pitch; a zero `stride` repeats the first row for every one of them.
The encodings Zig has no primitive for, `usearch.Scalar.BF16`, `E5M2`, `E4M3`, `E3M2`, `E2M3`, and the packed `I4x2`, `U4x2` and `U1x8`, travel as their raw bits, and a packed kind counts columns in dimensions, so 128 dimensions of `Scalar.U1x8` take sixteen values a row.
Lengths that cannot be read as a whole number of rows are refused with `Error.ShapeMismatch` rather than asserted, so the check survives `ReleaseFast`.
The column count travels with the rows, and the engine refuses one that differs from the frozen geometry with the same error.

Every metric derives from a single inner-product kernel with pre-computed magnitudes.

- `.inner`: inner product, and the only kind the linear-algebra verbs accept.
- `.angular`: cosine distance, over pre-computed magnitudes.
- `.euclidean`: L2 distance.
- `.hamming`: bit-level, over `.u1x8` vectors alone.
- `.jaccard`: bit-level, over `.u1x8` vectors alone.

Arbitrary metrics belong to the C++ baseline tier; the C interface, and so this module, takes the built-in kinds alone.

### Search

Every verb has one entry point, its optional knobs riding one config struct named after its C argument structure, such as `DenseSearchConfig`, `DenseInsertConfig` or `DenseMultiplyConfig`.

```zig
var arena = try usearch.Arena.init(executor);
defer arena.deinit();
var stats: usearch.DenseTraversalStats = .{};

const matches = try index.on(executor).search(try .of(f32, queries, 768), 10, &arena, .{
    .expansion_factor = 128,
    .stats = &stats,
});
for (0..matches.queries) |query| consume(matches.query(query));
std.debug.print("{d} distances computed\n", .{stats.computed_distances});
```

`Matches.query` answers one query's keys and distances, nearest first, as many as the walk found up to the "k" asked for.

The breadth of the candidates rides the call rather than the handle, so a recall sweep is a loop over expansion factors against one immutable index, with no mutation and no synchronization between points.
`.traversal = .scan` scores every entry instead of walking, on RAM and NVMe handles, which gives that sweep its exact ground truth; a VRAM handle answers it with `Error.Unsupported`.

### Filtered Search

A `Predicate` in the search config admits only the keys it approves, and non-matching entries still route the traversal, so recall does not collapse at low selectivity.
`Predicate.of` takes a `comptime` function rather than a closure, so the trampoline that crosses the boundary is generated per call site and nothing is boxed.
Its context travels as one pointer: pass `{}` for a predicate needing none, and a single-item pointer otherwise.

```zig
const admits = struct {
    fn call(_: void, key: usearch.Key) usearch.Admission {
        const value = key.asInteger() orelse return .reject;
        return if (value % 2 == 0) .admit else .reject;
}
}.call;

const even = try index.on(executor).search(try .of(f32, queries, 768), 10, &arena, .{
    .predicate = .of({}, admits),
});
```

It runs once per scored candidate, far more often than the "k" asked for, and concurrently from the executor's threads, so it belongs on your fast path and must tolerate that.
A VRAM index refuses a predicate with `Error.Device`, since a host callback cannot narrow a device walk.

Every long verb, namely `insert`, `refine`, `search`, `reshard`, `multiply`, `attend`, `maxsim`, `save`, `saveBuffer` and every load, takes a `ProgressCallback` in its config, called with the units finished so far and the units the call finishes if it runs to the end.
Answering `.cancel` stops the call at the next increment boundary, except an `insert`, which ignores it.

```zig
const Deadline = struct {
    io: std.Io,
    stop_at: i64,
    fn reached(self: *const @This(), _: usize, _: usize) usearch.Continuation {
        const now = std.Io.Clock.awake.now(self.io).toMilliseconds();
        return if (now < self.stop_at) .@"continue" else .cancel;
}
};

const deadline: Deadline = .{ .io = io, .stop_at = std.Io.Clock.awake.now(io).toMilliseconds() + 30_000 };
_ = try index.on(executor).refine(.{
    .expansion_factor = 128,
    .progress = .of(&deadline, Deadline.reached),
});
```

Like a predicate, the handler is a `comptime` function whose trampoline is generated per call site, and its context travels as one pointer, `{}` where none is needed.
It arrives on whichever thread advanced the work and never on two at once, so a counter or a deadline needs no synchronization of its own.
Reports are denominated in whatever the verb consumes, such as entries for an insertion, queries for a search, a phase's entries or islands for a round and shards for a checkpoint, and how often they arrive is the increment's size rather than a guarantee, so read the counts and never the number of calls.

An `insert` cannot be stopped: it claims every entry before its first report and publishes the whole batch at once, so a batch is all-or-nothing with or without a handler.
A stopped `refine` answers `Error.Cancelled` at a phase barrier, leaving the graph correct, and a plan keeps the round to continue later, as the Refinement section shows.

### Retrieval and Streaming

`get` lands every vector stored under a key in the arena, one packed row per entry in insertion order, as the element type you name; an absent key yields no rows.

```zig
const vectors = try index.get(f32, key, &arena);
for (0..vectors.rows) |row| consume(vectors.row(row));
```

`DenseVectors(Element).view` holds the byte pitch of each row as `stride` and its dimensions as `columns`, as the call reported them, which is how a sub-byte kind like `Scalar.U1x8` is read without arithmetic on its packing.

A cursor walks the index contents, reading each shard's publication watermark as it enters that shard, so an entry once passed is never revisited.
Without start keys it yields every entry as stored; with them, the starts' own entries first, then the entries those link to breadth-first, each once.
Each pull lands up to `wanted` keys in the arena, and the keys each links to when `links` is true; the vectors themselves come from `get`.

```zig
var cursor = try usearch.DenseCursor.init(index, &.{});
defer cursor.deinit();

while (true) {
    const batch = try index.on(executor).cursorNext(cursor, 1024, true, &arena);
    if (batch.written == 0) break; // exhausted
    consume(batch.keys(), batch.links(0));
}
```

A search stream is a resumable best-first traversal, so callers that post-filter aggressively pull until satisfied instead of guessing the "k" upfront.
It keeps walk state rather than a thread, so any number may stay open on one executor, and every pull is an ordinary verb naming the index again, so it runs on whichever receiver pulls.

```zig
var stream = try usearch.DenseSearchStream.init(index, try .ofRow(f32, &query), .{});
defer stream.deinit();

while (true) {
    const found = (try index.on(executor).searchStreamNext(stream, 16, &arena)).query(0);
    if (found.keys.len == 0) break; // drained
    for (found.keys, found.scores) |key, distance| std.debug.print("{f} at {d}\n", .{ key, distance });
}
```

A drained walk answers an empty query, as the C interface does, so a loop ends on the first pull that lands nothing.

### Refinement

`refine` runs one round over the finished graph and returns a `DenseRefineReport`: the entries re-selected and changed, the islands found and reconnected, the neighborhoods re-stitched with the links added and dropped, the `verdict`, and the health checks opening the round and closing the heal.
A cheap insertion pass followed by refinement reaches a better graph in less total time than one expensive insertion pass.
A `verdict` other than `.improved` means the round settled, and on a RAM index searches may run throughout.

```zig
while ((try index.on(executor).refine(.{ .expansion_factor = 128 })).verdict == .improved) {}
```

On a loaded NVMe handle `memory_limit` bounds the bytes a round holds for the one part it rewrites: zero rounds the part in RAM, and a smaller budget streams its rounds from the drives.

A cancel stops the round at a phase barrier and answers `Error.Cancelled`.
A `DenseRefinePlan` in the config keeps it: an empty plan opens one a cancel leaves live, the next call naming it continues where it stopped, and `refinePlanFinish` closes it with the whole plan's report.
A `PhaseProgress` in the config reports where the call stopped, `.done` once its round ended.

```zig
var plan: usearch.DenseRefinePlan = .{};
var phase: usearch.PhaseProgress = .{};
while (true) {
    _ = index.on(executor).refine(.{ .plan = &plan, .phase = &phase, .progress = budget }) catch |failure| switch (failure) {
        error.Cancelled => continue, // resumes at `phase.phase`
        else => return failure,
    };
    if (phase.phase == .done) break;
}
const whole = try index.on(executor).refinePlanFinish(&plan);
```

### Resharding

`reshard` moves the entries of its sources into its targets, keeping their links rather than re-inserting their vectors, then runs a round on every target that took entries.
Several sources into one target, itself among them, gather into it; one source into several targets cuts it into graph-local parts, its part roots drawn from `seed`; one handle as its only source and target recuts its own shards in place.
Every handle must share the first source's residency, scalar, metric and geometry, or the call answers `Error.ConfigMismatch`, and a target is a source itself or empty.
A source the targets do not name ends empty and live, and every handle stays yours to `deinit`.

```zig
_ = try index.on(executor).reshard(&.{ index, first, second }, &.{index}, .{}); // gather
_ = try index.on(executor).reshard(&.{index}, &receivers, .{ .seed = 42 }); // cut
_ = try index.on(executor).reshard(&.{index}, &.{index}, .{ .memory_domains = &.{}, .shard_capacity = 10_000_000 }); // recut
```

The handles come from the two lists and the executor from the receiver, and `DenseIndex.reshard` runs it on the calling thread.
A cancel stops the closing rounds, never a move, and a `DenseReshardPlan` with `reshardPlanFinish` keeps and closes them as `refine` does.

### Sampling

`sample` draws keys around `focus`, or across the whole index, each hanging from a shown or earlier draw and placed in three dimensions beside the `shown` keys already on screen, so a view grows one connected sample as it zooms.
Equal seeds, shown keys and focus draw equally, and a larger `wanted` draws a superset.

```zig
const draws = try index.on(executor).sample(256, &arena, .{ .shown = shown, .shown_coordinates = places, .focus = near, .seed = 42 });
for (draws.keys(), draws.entryOrdinals(), draws.coordinates(), 0..) |key, entry_ordinal, point, position| {
    const parent = draws.parents().query(position); // empty for a root
    _ = .{ key, entry_ordinal, parent, point };
}
```

### Sparse Multiplication

Three verbs, `multiply`, `maxsim` and `attend`, treat the index as a sparse operand rather than as a kNN store, each folding its arithmetic into the graph walk instead of materializing a match list.
All three read logits rather than distances, so they refuse anything but an inner-product index with `Error.ConfigMismatch`.

`multiply` is a sparse matrix product: every left row is walked against the index, and how many pairs it scores is known only once the walk ends, so the products come back as a ragged `DenseProducts`: keys, scores and their Apache Arrow offsets, all in the arena.

```zig
const products = try index.on(executor).multiply(try .of(f32, left, 768), &arena, .{ .wanted = 64 });
for (0..products.queries) |row| {
    const pairs = products.query(row);
    accumulate(row, pairs.keys, pairs.scores);
}
```

`offsets`, `keys` and `scores` answer the whole tape at once, for a consumer that takes Arrow buffers as they are.

### Late Interaction

`maxsim` is late interaction: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `values_per_key` above one exists for, and the outputs are sized over documents rather than rows.
The keys land as a `DenseProducts`, one run per query, largest score first, and a zero `wanted` keeps every key a query scored.

```zig
const documents = try index.on(executor).maxsim(try .of(f32, tokens, 768), tokens_per_query, 10, &arena, .{});
for (0..documents.queries) |query| {
    const ranked = documents.query(query);
    rerank(query, ranked.keys, ranked.scores);
}
```

### Attention

`attend` is Flash Attention folded into the walk: every key it scores enters an online softmax, with no top-k cut, and the outputs come back as wide as the keys and divided by the partition.
The index stores no values, so a `ValueLookup` resolves each scored key to its row, as wide as the keys and stored in the index's quantization.
Its trampoline checks the kind and width every batch states, so a missing row, or one of another kind or width, refuses the call with `Error.ShapeMismatch`.
It runs on the executor's threads, so it must tolerate concurrent calls.
That trampoline is a host function, which serves RAM and NVMe indexes; a VRAM walk calls the lookup on the device, so a VRAM index takes a `ValueLookup` whose `callback` is device-callable, set directly.
The `head_count` query heads of grouped-query attention share one walk per token, row `t · head_count + h` being head `h` of token `t`, and a `carry` of one `DenseSoftmaxState` per query row, in the caller's memory rather than in the arena, folds this branch into another over the same queries.
The attended rows land in the arena as the element type the call names.

```zig
const Cache = struct {
    rows: []const usearch.Scalar.BF16,
    fn resolve(self: *const @This(), key: usearch.Key) ?[]const usearch.Scalar.BF16 {
        const entry: usize = @intCast(key.asInteger() orelse return null);
        return self.rows[entry * 128 ..][0..128];
}
};
const attended = try index.on(executor).attend(f32, try .of(f32, queries, 128), .of(usearch.Scalar.BF16, &cache, Cache.resolve), &arena, .{
    .head_count = 8,
    .carry = carry,
});
const first_head = attended.row(0);
```

### Introspection

```zig
const description = try index.describe();
std.debug.print("{d} entries in {d} bytes\n", .{ description.size, description.memory_usage });
std.debug.print("dispatched to {s} across {d} shards\n", .{ description.cpuClassName(), description.shards_count });
std.debug.print("compiled for {s}\n", .{usearch.capabilitiesCompiled()});
std.debug.print("available here {s}\n", .{usearch.capabilitiesAvailable()});

const reachability = try index.on(executor).health();
std.debug.print("{d} of {d} entries reachable\n", .{ reachability.reachable, reachability.audited });
```

`health` measures how much of the graph a search reaches from the root; entries it misses stay unfound until a `refine` re-links them.

## Sequence Indexes

A `SequenceIndex` searches proteins under alignment, DNA and RNA under shifted overlap, and UTF-8 text under window containment.
Its sequences cross as one `TapeView`, the Apache Arrow `Binary` layout of bytes end to end around one more offset than entries, so the two buffers of an Arrow column feed it as they are, with `u32` or `u64` offsets.

```zig
var index = try usearch.SequenceIndex.initRam(.{ .metric = .utf8_containment }, &.{});
defer index.deinit();
const keys = [_]usearch.Key{ .fromInteger(1), .fromInteger(2), .fromInteger(3) };
const offsets = [_]u64{ 0, 19, 29, 40 };
try index.insert(&keys, try .of(u8, u64, "The quick brown foxA lazy dogPack my box", &offsets), .{});

var arena = try usearch.Arena.init(null);
defer arena.deinit();
const query = [_]u64{ 0, 9 };
const matches = try index.search(try .of(u8, u64, "BROWN FOX", &query), 1, &arena, .{});
std.debug.assert(matches.query(0).scores[0] < 1.0 / 32768.0); // holds every query window, whatever the case
```

The `.metric` is `.protein_alignment`, `.dna_overlap` or `.utf8_containment`.
Each metric stores a sequence folded, so `get` returns the stored form rather than the inserted bytes: packed class codes for a protein, one class code a byte for DNA, and case-folded UTF-8 for text.
Optional `SequenceCosts` give 256 × 256 similarities between raw bytes and a gap cost in place of the metric's own, and a containment refuses them.
`SequenceOptions.sketch_bytes` sets the bytes of the sketch in front of each stored sequence, a multiple of four, where zero takes the metric's default of 512 for protein alignment and UTF-8 containment.
A DNA overlap keeps no sketch and refuses any.

```zig
var similarities = std.mem.zeroes([256][256]i8);
for ("ACGT") |base| similarities[base][base] = 2;
const costs: usearch.SequenceCosts = .{ .similarities = &similarities, .gap = 2 };
var reads = try usearch.SequenceIndex.initRam(.{ .metric = .dna_overlap, .costs = &costs }, &.{});
defer reads.deinit();
```

Text folds its case inside the engine, so inputs are raw bytes and there is nothing to pack.
A checkpoint records the costs, so loads and views take none.
`get` writes a key's sequences into an arena as `TapeEntries(u8)`, and `count` answers a `SequenceKeyStats` with the key's entries and their total and largest byte lengths.
Offsets that reach past the elements are refused with `Error.ShapeMismatch` before the call crosses, since the C interface reads no element count to catch them.
The receivers, reports and plans work as the dense ones do: `refine` and `reshard` take a `SequenceRefinePlan` or a `SequenceReshardPlan` and answer a `DenseRefineReport`.
`reshard` gathers several sources into one target or cuts one source into several, while one handle as its only source and target answers `Error.Unsupported`, as these indexes take no placement to recut into.

## Shape Indexes

A `ShapesIndex` searches 3D point clouds of any size, like molecules, conformers and protein backbones, invariant to rotation, reflection, translation and point order.
Its tape holds `[3]` points end to end, with offsets that count points, as an Arrow `List` column lays them out:

```zig
var index = try usearch.ShapesIndex.initRam(.{}, &.{}); // Chamfer over f32 points
defer index.deinit();
const points = [_][3]f32{ .{ 0, 0, 0 }, .{ 1.1, 0, 0 }, .{ 0, 0, 0 }, .{ 1.5, 0, 0 }, .{ 0, 1.5, 0 } };
const offsets = [_]u64{ 0, 2, 5 };
try index.insert(&.{ .fromInteger(1), .fromInteger(2) }, try .of([3]f32, u64, &points, &offsets), .{});

const probe = [_][3]f32{ .{ 0, 0, 0 }, .{ 1, 0, 0 } };
const matches = try index.search(try .of([3]f32, u64, &probe, &.{ 0, 2 }), 1, &arena, .{});
const pair = try index.get(f32, matches.query(0).keys[0], &arena); // `TapeEntries([3]f32)`
```

The metrics are `.chamfer`, `.hausdorff`, `.gromov_wasserstein` and `.wasserstein_procrustes`, over `.f16` or `.f32` storage, whatever scalar the points arrive in.
A non-zero `points` in `ShapesOptions` freezes every shape's point count, and `reserve` takes the points to expect beside the entries.
`get` converts each shape to the element type it names and returns it in its inserted pose, within the stored rounding, and `count` reports a key's total and largest point counts.

## Geospatial Indexes

A `GeospatialIndex` searches observations on and above the Earth, like photos, video frames, drone footage and sensor fixes.
`GeospatialObservation` is the library's own eleven doubles, so a slice of them crosses with no copy; a field no device recorded stays NaN, which only widens the tests that read it, and `GeospatialObservation.at` leaves every field but the position that way:

```zig
var index = try usearch.GeospatialIndex.initRam(.{}, &.{});
defer index.deinit();
const stops = [_]usearch.GeospatialObservation{ .at(51.5308, -0.1238), .at(51.5320, -0.1233), .at(51.5033, -0.1196) };
try index.insert(&.{ .fromInteger(1), .fromInteger(2), .fromInteger(3) }, &stops, .{});

const nearby = try index.within(stops[0..1], 500, &arena, .{}); // every stop within 500 metres
std.debug.assert(nearby.query(0).keys.len == 2);
const ranked = try index.search(stops[0..1], 1, &arena, .{ .relation = .near }); // chord metres
```

`within` reports every entry the relation admits inside the radius, however many, as a ragged `GeospatialMatches` with Arrow offsets, while `search` ranks the `wanted` nearest, a zero `metres` ranking by distance alone.
The relation is a field of each call's config: `.near` widens the radius by both positions' accuracies, `.sees` and `.seen` test the view cones, and `.visible` tests a line of sight over the `GeospatialTerrain` the handle was built with, refused with `Error.Unsupported` on a handle without one.
A terrain points at heights that outlive the handle, or fills windows of them on demand through `GeospatialTerrainFill.of`, a trampoline generated per call site that runs on the executor's threads; no checkpoint holds it, so every load and view names it again.
`get` returns a key's observations within the stored rounding.

## Left Unbound

- The `hints` of a dense or shape search, which the C argument structures hold and this module leaves null.
- An NVMe `load`, search streams, and the sparse verbs, for the sequence, shape and geospatial indexes, since the C library declares none for them.
- The Arrow converters of `c/usearch/arrow.h`, which view an Arrow column as an input and export an arena's output as one.
- The numeric `usearch_version` triple; `usearch.version()` answers the library's version string instead.

## Design

- __A thin layer over the C ABI.__
  Every name follows `c/usearch/`: a C verb `usearch_dense_<verb>` is a `<verb>` here, in camelCase, and enums mirror C enums, never strings.
  Every type is its C name without `usearch_` and `_t`, a verb's argument structure becomes its `Dense*Config`, and every wrapper's documentation names the C symbol it calls or says it is binding-only.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor, &call)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `usearch_outcome_status`, `wait` blocks through `usearch_outcome_wait`, and with no `async` left in Zig, waiting is how a result is collected.
- __No life of its own.__
  The binding starts no threads and keeps no queues; releasing a `Deferred` rides `defer call.deinit()`, which waits for the call first.
- __Arenas belong to the caller and are reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes, and its views read the arena directly until the arena's next call.
- __Nothing costs anything until it is used.__
  Submitting a call allocates nothing, and an arena allocates only when a call writes into it.
- __Layouts are declared, not translated.__
  An `extern struct` already has the ABI's layout and an `enum(c_int)` its width, so no call site casts what a translated header would hand back as `c_uint` or `?*anyopaque`.
- __Callbacks are `comptime` functions.__
  Predicates, progress handlers and value lookups wrap a function known at compile time, so the trampoline crossing the boundary is generated per call site and nothing is boxed.
