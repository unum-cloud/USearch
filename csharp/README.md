# USearch for C#

The C# binding calls the flat C interface in `include/usearch/` through source-generated `[LibraryImport]` stubs over `libusearch_shared`, so it compiles no native shim of its own.
It requires .NET 10, and the package carries the native library for Linux x64 and macOS Arm64.
It exposes dense indexes; sequence, matrix and geospatial indexes follow as the C library ships them.

## Installation

```sh
dotnet add package Cloud.Unum.USearch
```

The package targets `net10.0`, the long-term-support release supported through November 2028.
Building from source compiles the C library first, then places it where the package looks for its native assets:

```sh
cmake --preset relwithdebinfo -D USEARCH_BUILD_SHARED=1
cmake --build --preset relwithdebinfo
mkdir -p csharp/lib/runtimes/osx-arm64/native                   # Or linux-x64 with the .so
cp build_relwithdebinfo/libusearch_shared.dylib csharp/lib/runtimes/osx-arm64/native
cd csharp && dotnet test
```

## Quickstart

```csharp
using Cloud.Unum.USearch;

DenseOptions options = DenseOptions.Default with  // Angular metric, BF16 quantization
{
    Metric = DenseMetric.Angular,                 // Or Inner, Euclidean, Hamming, Jaccard
    Quantization = Scalar.BF16,                   // Or F16, E5M2, E4M3, E3M2, E2M3, I8, U8, U1x8
    Dimensions = 3,                               // Frozen at init; zero is refused
    Connectivity = 16,                            // Connections per node in the graph, optional
    ValuesPerKey = 1,                             // Upper bound on entries sharing one key, optional
};

using DenseIndex index = DenseIndex.InitRam(options);
index.Reserve(1_000);                             // Optional capacity hint; inserts grow on demand

Key[] keys = [42UL];
float[] vector = [0.2f, 0.6f, 0.4f];
index.Insert(keys, RowsView.RowsOf<float>(vector, dimensions: 3));                // On the caller's thread

using Executor pool = Executor.ForRam();                                         // The threads a receiver names
using Arena arena = new(pool);                                                   // Where outputs land, reused
Matches found = index.On(pool).Search(RowsView.RowsOf<float>(vector, 3), wanted: 10, arena); // The same verb, on the pool
Key nearest = found[0].Keys[0];                                                  // 00000000-0000-0000-0000-00000000002a
double distance = found[0].Distances[0];
```

## Receivers & Deferred Calls

The receiver chooses where a verb runs, and the verb stays the same.
Called on the index, a verb runs on the caller's own thread, single-threaded on RAM and on the handle's device and default stream on VRAM.
On `index.On(executor)` it blocks on that executor, refused while another call holds it.
On `index.Deferred(executor)` it submits there and answers a `Deferred<TResult>` at once.

```csharp
Matches local = index.Search(queries, wanted: 10, arena);                 // The caller's thread
Matches pooled = index.On(pool).Search(queries, wanted: 10, arena);       // Blocking, on the pool
Matches awaited = await index.Deferred(pool).Search(queries, 10, arena);  // Or: .Wait() in place of await
```

A `Deferred<TResult>` owns its outcome and the pins on every input it lends the library until the call is final, and borrows its index and arena exactly as a blocking call does, so `Wait()` answers the same view a blocking call would, over the same arena.
`Status` reads `Status.Deferred` until then, `Wait()` blocks and answers the result or raises the failure, and `await` runs that wait on the thread pool.
Verbs that produce nothing answer a plain `Deferred`, as `Task` stands beside `Task<TResult>`.

```csharp
using Deferred<Matches> call = index.Deferred(pool).Search(queries, wanted: 10, arena);
if (call.Status == Status.Deferred)
{
    // other work
}
Matches matches = await call;                                       // Or: call.Wait()
```

Waiting again answers the same result or raises the same failure.
Disposing a call nobody waited for waits for it, discards what it produced, and lets its arena serve the next call.
One left unfinished and unreachable keeps its outputs and pins alive until a collection finds it final, rather than freeing memory the library still writes.
Disposing an index or an arena that a deferred call names waits for that call first.
One call holds an executor at a time, so a second call finding it held finishes at once with `Status.Contended`, and nothing queues.
A deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling alone finishes it, and its progress handler and predicate run on the pool's threads; a single thread or a VRAM executor runs it at submission.
Calls depending on each other are ordered by waiting on the first before submitting the second.

## Executors

An executor owns host threads or device streams; an index owns entries.
Keeping them apart is what lets one pool serve several indexes, and what makes the cost of spawning visible at the line that spawns.

One factory per C constructor, each taking only what its hardware has: `ForVram` has no thread count because a single host thread drives one stream per device.

```csharp
using Executor threads = Executor.ForRam(threads: 8);             // Zero derives 7/8 of the cores
using Executor rings = Executor.ForNvme(threads: 4);              // One submission ring apiece
using Executor streams = Executor.ForVram(scratchLimit: 1 << 30); // One stream per device
using Executor shared = Executor.OnPool(forkUnionPool);           // Borrows another library's threads

threads.Drain(Mood.Spinning);                                     // Waits out the holding call, then keeps spinning
```

Each factory also names where its shards go, which is how a dual-socket or multi-GPU box is reached.

```csharp
using Executor domains = Executor.ForRam(16, memoryDomains: [0, 1]);  // One shard per NUMA node
using Executor devices = Executor.ForVram([0, 1], cudaStreams);        // Borrowed streams, one per device
using Executor striped = Executor.ForNvme(paths: ["/mnt/nvme0", "/mnt/nvme1"]);
```

One call holds an executor at a time, while handles stay freely concurrent, so a caller wanting genuine concurrency gives each thread its own executor.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`new Arena(executor)` places it where that executor computes, decided once, and `new Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own, and a call naming an arena that serves another call finishes with `Status.Contended` and rewinds nothing, so the holder's views stay readable.
A stale view throws `USearchException` with `Status.Uninitialized`; a `ReadOnlySpan` an accessor already returned cannot outlive its stack frame, so read it before the next call.
An arena's `Describe()` says where it lives, and one from a VRAM executor holds device memory, which host accessors refuse with `Status.Device`.
It grows, fills and frees on that executor's first stream, so it serves only calls on that executor and holds it open until disposed; a host arena may outlive its executor.

```csharp
using Arena arena = new(pool);                                     // new Arena() is host pages for the calling thread
foreach (RowsView batch in batches)
{
    Matches matches = index.On(pool).Search(batch, 10, arena);   // Rewinds: the last batch's matches are gone
    foreach (double distance in matches[0].Distances) total += distance;
}

using Arena found = new(), fetched = new();                        // Two live results, two arenas
foreach (Key key in index.Search(query, 10, found)[0].Keys)
{
    Rerank(key, index.Get(key, fetched));
}

Executor[] pools = [.. shards.Select(_ => Executor.ForRam(threads: 8))];  // One executor and one arena per submitted call
Arena[] arenas = [.. pools.Select(own => new Arena(own))];
Deferred<Matches>[] calls = [.. shards.Select((shard, at) => shard.Deferred(pools[at]).Search(queries, 10, arenas[at]))];
foreach (Deferred<Matches> call in calls) Consume(await call);   // Or: call.Wait()
```

## Keys

Keys are 128 bits wide and render as lowercase dashed UUIDs, with integer identities parked in the low eight bytes, so `42UL` converts implicitly and `TryGetUInt64` reads one back.
A full identity comes from `Key.Parse("0123abcd-4567-89ef-fedc-ba9876543210")`, or `Key.TryParse` when the text may be malformed.
Parsing reads the dashes as separators rather than positions, so they may sit anywhere or nowhere, and what they leave behind must be thirty-two hexadecimal digits of either case.
`Key.FromGuid` and `Key.ToGuid` carry an identity across to `System.Guid`, which is where a time-ordered `Guid.CreateVersion7()` key comes from.

## Errors

Every failure arrives as a `USearchException` carrying a `Status`, one enumerator per `usearch_status_*_k`, such as `ShapeMismatch`, `Io`, `Unsupported`, or `InvalidArgument`.
Width and geometry checks belong to the library, which sees every row view's `columns`; the binding checks only what the C interface cannot see, like a key count against the rows it accompanies, a softmax carry against the queries, or a negative `wanted`.

The closed sets spell themselves through the library's own name tables, so a message reads `usearch: shape-mismatch: ...` and parsing user text never restates a table:

```csharp
string spelled = Status.ShapeMismatch.Name();          // "shape-mismatch"
bool known = Names.TryParse("bf16", out Scalar scalar);
```

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has, one shard per memory domain, device or path, and an empty list derives the placement.

```csharp
using DenseIndex inRam = DenseIndex.InitRam(options, memoryDomains: [0, 1]);      // One graph per socket
using DenseIndex inVram = DenseIndex.InitVram(options, devices: [0, 1, 2, 3]);    // Sharded, never replicated
using DenseIndex uploaded = DenseIndex.LoadVram("index.safegraphs", devices: [0, 1]);
using DenseIndex onDisk = DenseIndex.ViewNvme("index.safegraphs.json", paths: ["/mnt/nvme0"]);
```

An NVMe handle serves a checkpoint in place, and every mutating verb fails on it with `Status.ReadOnly`, as on a `DenseIndex.ViewRam`.
Until NVMe checkpoints carry keys, `ViewNvme` answers `Status.Unsupported`.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout: a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.
Checkpoints are residency-portable: one file materializes into RAM, uploads to the devices, or opens in place on disk, decided by the constructor that opens it.

```csharp
await index.Deferred(pool).Save("index.safegraphs");                        // Or: index.On(pool).Save(path)

DenseMetadata metadata = DenseIndex.MetadataRead("index.safegraphs", pool);  // Geometry without loading

using DenseIndex restored = DenseIndex.LoadRam("index.safegraphs", executor: pool); // Checkpoint freezes the geometry
using DenseIndex served = DenseIndex.ViewRam("index.safegraphs");                    // Mapped in place, read-only

Monolith buffer = index.On(pool).SaveBuffer(arena);                         // Or one monolith in memory
using DenseIndex copied = DenseIndex.LoadRamBuffer(buffer.Bytes, executor: pool);
```

`DenseMetadata` carries the four fields a checkpoint froze: dimensions, metric, quantization and connectivity.
Residency is absent because a checkpoint records where it was written rather than where a reader must keep it, and `ValuesPerKey` is absent because it bounds a live handle's mutable state rather than the serialized form.
A monolith reports its `Length` as a `long`, and one past 2 GiB refuses to lend its `Bytes` as one run of memory with `Status.Overflow`.

## Dense Indexes

`DenseOptions` is immutable, so a variant comes from `DenseOptions.Default with { ... }` rather than from mutating a value another handle may already have been built from.

### Rows & Scalars

Every batch argument is a `RowsView`: a caller-owned slab viewed as rows of equal width, carrying its row count, its logical width, its byte pitch, and the scalar kind its values hold.
A single vector is a one-row matrix, and the input scalar type is independent of the stored quantization, since the library converts on ingest.

```csharp
float[] rows = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
RowsView packed = RowsView.RowsOf<float>(rows, dimensions: 3);              // Three rows, pitch 12 bytes

byte[] bits = new byte[16];
RowsView bitsets = RowsView.RowsOf<byte>(bits, dimensions: 64, Scalar.U1x8);   // Two rows, pitch 8 bytes

float[] padded = new float[2 * 8];
RowsView strided = RowsView.StridedRowsOf<float>(                          // The layout a mapped dataset arrives in
    padded, rows: 2, dimensions: 3, strideBytes: 8 * sizeof(float), Scalar.F32);
```

The kind travels with the slab rather than being derived from its element type, which is what makes the packed sub-byte kinds reachable: sixty-four `Scalar.U1x8` dimensions occupy eight bytes, not sixty-four.

A `RowsView` borrows and never owns: it holds the `Memory<T>` it was built from, so the slab stays reachable for as long as the view is, and a deferred call pins it until final.

### Search

```csharp
Key[] keys = [1UL, 2UL, 3UL];
DenseTraversalStats stats = new();
index.On(pool).Insert(keys, packed, expansionFactor: 128, stats: stats);

Matches found = index.On(pool).Search(packed, wanted: 4, arena, expansionFactor: 64);
for (int query = 0; query < found.Queries; query++)
{
    ReadOnlySpan<Key> matched = found[query].Keys;              // Already trimmed to the true count
    ReadOnlySpan<double> distances = found[query].Distances;
}
```

The breadth of the candidates rides each call rather than the handle, so a recall sweep runs against one immutable index with no mutation between points.
`traversal: Traversal.Scan` scores every entry instead of walking, on RAM and NVMe handles, which gives that sweep its exact ground truth; a VRAM handle refuses a scan with `Status.Unsupported`.
`DenseTraversalStats` is a pure output, overwritten by every call it is passed to, and `Accumulate` folds one call's counters into another's to total a run of batches.

A sweep reuses one arena across every point, which keeps its high-water mark, so nothing is allocated between them:

```csharp
foreach (int breadth in new[] { 16, 32, 64, 128 })
{
    Matches point = index.On(pool).Search(packed, wanted: 10, arena, expansionFactor: (ulong)breadth);
    // Read `point` here: the next point's call rewinds the arena
}
```

### Filtered Search

```csharp
Matches admitted = index.On(pool).Search(
    RowsView.RowsOf<float>(vector, dimensions: 3), wanted: 10, arena,
    predicate: key => key.TryGetUInt64(out ulong identity) && identity % 2 == 0);
```

Rejected entries still route the traversal, so recall does not collapse at low selectivity.
The predicate runs on the executor's worker threads, so it must be safe for concurrent use, and it is called once per scored candidate rather than once per reported match.
An exception it throws is captured, every later candidate is rejected, and the exception is rethrown with its own type once the call is final, never unwound through the native frames.
A host predicate cannot narrow a device walk, so a VRAM handle refuses one with `Status.Device`.

### Retrieval and Streaming

```csharp
DenseVectors vectors = index.On(pool).Get(key, arena, Scalar.F32);  // Every entry under the key, zero-copy
ReadOnlySpan<float> first = vectors.Row<float>(0);                  // As wide as vectors.Columns

using DenseCursor cursor = index.On(pool).Cursor();    // Each entry once, in storage order; `Cursor([key])` walks out from a key
for (DenseBatch batch = cursor.Next(128, arena); batch.Count != 0; batch = cursor.Next(128, arena))
{
    ReadOnlySpan<Key> page = batch.Keys;               // Pass `links: true` for `batch.Links(entry)`; vectors come from `Get`
}

using DenseSearchStream stream = index.On(pool).SearchStream(RowsView.RowsOf<float>(vector, dimensions: 3));
QueryMatches nearer = stream.Next(8, arena);             // Resumable nearest-neighbor stream
```

A cursor or stream keeps walk state rather than a thread, pulls on the executor of the receiver that opened it, and keeps its index alive until disposed.
Every pull takes its arena, so one arena serves a scan and any other verb between pulls, and a drained pull answers an empty view.

### Refinement

```csharp
RefineReport report = index.On(pool).Refine(expansionFactor: 256);
```

Refinement re-selects the neighbors of every stored entry against the finished graph, improving edges that were chosen when the graph around them was still incomplete.
It is a build step rather than a repair: a cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass, so a wide expansion factor belongs here rather than in `Insert`.
The return is a `RefineReport` of the nodes re-selected and `Changed`, the islands found and reconnected, and the `Reachability` audit closing the heal; `Changed` at zero means the graph has settled, so a caller under a time budget loops until it falls below whatever rate it considers settled.
Searches may run throughout.

### Absorbing & Splitting

```csharp
index.On(pool).Absorb(donor);
index.On(pool).Split([left, right], seed: 7); // two empty receivers take a part each
```

Bigger indexes come from absorbing, never mutating: `Absorb` appends each donor in turn, keeping its graph and linking it in across the appended part.
Every donor it absorbs is consumed and then refuses every verb with `Status.Uninitialized`, while one it did not absorb, like a donor of another scalar refused with `Status.ConfigMismatch`, stays a live index you still own, though its entries may already have moved.
`Split` is its inverse, partitioning the entries along the graph into empty receivers and leaving the source empty and live; both answer a `RefineReport` of their closing refinements.

### Sampling

```csharp
Sample drawn = index.On(pool).Sample(
    256, arena, new DenseSampleConfig { Shown = shown, ShownCoordinates = shownXyz, Focus = near, Seed = 42 });
```

`Sample` draws keys around a focus, each hanging from a shown or earlier draw, and places them in three dimensions beside the keys already shown, so a view grows one connected sample as it zooms.
Its `Keys`, `EntryOrdinals`, `Parents` and `Coordinates` read in place from the arena.

### Sparse Multiplication

```csharp
DenseProducts products = index.On(pool).Multiply(RowsView.RowsOf<float>(left, dimensions: 3), arena, wanted: 64);
ReadOnlySpan<Key> reached = products.Keys(0);         // Every indexed entry the first walk scored
ReadOnlySpan<double> dots = products.Scores(0);       // Paired with those keys, in traversal order
```

Every pair a traversal scored is an exact product the caller would otherwise recompute, so `Multiply` hands them all back, ragged per left row, with their offsets in the arena beside them.
It runs on the inner-product kinds alone, and the pairs a walk never reaches are the low and negative products.

### Late Interaction

```csharp
DenseOptions colbert = DenseOptions.Default with
{
    Dimensions = 128,
    Metric = DenseMetric.Inner,      // Only the inner-product kinds score logits
    Quantization = Scalar.BF16,
    ValuesPerKey = 32,               // A document is one key holding its token rows
};

DenseProducts scored = index.On(pool).Maxsim(RowsView.RowsOf<float>(queryTokens, dimensions: 128), rowsPerQuery: 32, wanted: 10, arena);
ReadOnlySpan<Key> documents = scored.Keys(0);         // The first query's best documents, largest first
ReadOnlySpan<double> totals = scored.Scores(0);       // Each document's summed maxima
```

Every query row takes its best match within a key, and those maxima sum to that key's score, so the wanted count is the "k" over documents rather than over rows.
Each query lands a ragged run in the arena, the same `DenseProducts` that `Multiply` returns, and a wanted count of zero keeps every document the query scored.
That is what a values-per-key bound above one exists for, and a graph frozen under a distance kind refuses with `Status.ConfigMismatch`.

### Attention

`Attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the outputs come back already divided by the partition.
The index stores no values, so every scored key resolves to a row of a values `RowsView` the caller lends, as wide as the keys and stored in the index's own quantization.

```csharp
DenseSoftmaxState[] carry = new DenseSoftmaxState[queryRows];
DenseVectors attended = index.On(pool).Attend(
    RowsView.RowsOf<float>(queryHeads, 128), RowsView.RowsOf<ushort>(valueCache, 128, Scalar.BF16),
    rowOf: key => positionOf[key],                       // Runs on the executor's threads together
    arena, headCount: 8, carry: carry);                  // One row per query row, as the queries' kind
ReadOnlySpan<float> first = attended.Row<float>(0);
```

With `headCount: 8`, row `t * 8 + h` of the queries is head `h` of token `t`, and the heads of one token share its walk.
A key resolving outside the values, or values of another kind or width than the library reads, refuses the call with `Status.ShapeMismatch`, since a missing row is a wrong softmax rather than a smaller one.
The `carry` holds one `DenseSoftmaxState` per query row, which folds this branch into another over the same queries, such as a recent window kept out of the index; it is state the caller carries between calls, so it stays lent rather than written into the arena.
An `outputsScalar` argument lands the attended rows as another kind than the queries' own.
`rowOf` is a host function, so it serves RAM handles alone: a VRAM walk resolves rows through a device function, which this binding does not build, and an `Attend` into a VRAM arena refuses with `Status.Device`.

### Introspection

```csharp
DenseIndexDescription description = index.Describe();    // Metadata, size, capacity, memory, capabilities
DenseReachability reachability = index.On(pool).Health();   // Audited, Reachable, Sweeps
ulong stored = index.On(pool).Count(key);                 // Entries under the key; Contains(key) tests for any
string kernels = USearchCapabilities.CapabilitiesAvailable();
```

`Health` audits how much of the graph a search reaches from its landmarks; entries it misses stay unfound until a refinement re-links them.
`USearchCapabilities` also reports the linked library's `Version()` and the `CapabilitiesCompiled()` into it, tied to no index.

## Interop

Every entry point is declared with `[LibraryImport]`, so the marshalling stubs are source-generated and compiled in rather than reflected at runtime, and the assembly stays trimmable and ahead-of-time compilable.
Every verb takes one argument struct, as the C interface does, and holds a reference on each `SafeHandle` the struct names for the length of the C call, so a finalizer can never free a handle out from under a running verb, and the free of one a deferred call still names waits for that call.

## What This Binding Does Not Expose

- __Progress handlers.__
  Every verb passes the C `progress` callback as null, so a long `Insert`, `Refine`, `Absorb`, `Split`, `Sample` or `Save` runs to its end, and `Status.Cancelled` never arises.
- __Deferred cursors and streams.__
  `Cursor` and `SearchStream` pull blocking, so a `DenseIndexDeferredOn` lacks them.
- __Constructors on receivers.__
  The constructors address no index yet, so no receiver offers them; the ones the C interface lets fan out take an `executor` argument and run on the caller's thread without one.
- __Version numbers and class names.__
  `usearch_version` and the CPU and GPU class spellings stay unbound; `USearchCapabilities.Version()` answers the version as text.

## Design

- __A thin layer over the C ABI.__
  Every name follows `include/usearch/`: a C verb `usearch_dense_<verb>` is a `<Verb>` method here, and enums mirror C enums, never strings.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling thread, `On(executor)` blocks on an executor, and `Deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `Status` polls through `usearch_outcome_status`, `Wait()` blocks through `usearch_outcome_wait`, and `await` hands that wait to the thread pool.
- __No life of its own.__
  The binding starts no threads and keeps no queues; releasing a `Deferred` rides `Dispose` and `using`, which waits for the call first.
- __Arenas are lent and reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes; a view the arena's next call outdated throws rather than reading rewound memory.
- __Nothing costs anything until it is used.__
  Submitting a call allocates nothing, and an arena allocates only when a call writes into it.
- __Blocking calls stay on the stack.__
  A blocking call keeps its outputs on the stack, fixes its buffers in place without a pinning handle, and answers a view by value; a deferred one allocates its handle and one pinned outputs struct.
- __Exceptions never cross the native boundary.__
  A predicate or `rowOf` delegate that throws is captured on the worker thread and rethrown with its own type once the call is final.
