# USearch for C#

## Installation

```sh
dotnet add package Cloud.Unum.USearch
```

The package targets `net10.0`, the long-term-support release supported through November 2028.

## Quickstart

```csharp
using Cloud.Unum.USearch;

DenseOptions options = DenseOptions.Default with  // Angular metric, BF16 quantization, RAM residency
{
    Metric = DenseMetric.Angular,                 // Or Inner, Euclidean, Hamming, Jaccard
    Quantization = ScalarKind.F32,                // Or F64, BF16, F16, E5M2, E4M3, E3M2, E2M3, I8, U8, U1x8
    Dimensions = 3,                               // Zero defers the geometry to a later `Load`
    Connectivity = 16,                            // Connections per node in the graph, optional
    ValuesPerKey = 1,                             // Upper bound on entries sharing one key, optional
};

using Executor pool = Executor.ForRam();          // The threads every fanned-out verb runs on
using DenseIndex index = new(options);
index.Reserve(1_000);

float[] vector = [0.2f, 0.6f, 0.4f];
index.Add(pool, [42UL], Matrix.RowsOf<float>(vector, dimensions: 3));

MatchesSpan found = MatchesSpan.Packed(queries: 1, wanted: 10);
index.Search(pool, Matrix.RowsOf<float>(vector, dimensions: 3), found);
USearchKey nearest = found[0].Keys[0];            // 00000000-0000-0000-0000-00000000002a
double distance = found[0].Distances[0];
```

`DenseOptions` is immutable, so a variant comes from `DenseOptions.Default with { ... }` rather than from mutating a value another handle may already have been built from.

Keys are 128 bits wide and render as lowercase dashed UUIDs, with integer identities parked in the low eight bytes - so `42UL` converts implicitly, and a full identity comes from `USearchKey.Parse("0123abcd-4567-89ef-fedc-ba9876543210")`.
Parsing reads the dashes as separators rather than positions, so they may sit anywhere or nowhere, and what they leave behind must be thirty-two hexadecimal digits of either case.
`USearchKey.FromGuid` and `USearchKey.ToGuid` carry an identity across to `System.Guid`, which is where a time-ordered `Guid.CreateVersion7()` key comes from.

## Executors

An executor owns host threads or device streams; an index owns rows.
Keeping them apart is what lets one pool serve several indexes, and what makes the cost of spawning visible at the line that spawns.

One factory per residency, each taking only what its residency has — `ForVram` has no thread count because a single host thread drives one stream per device.

```csharp
using Executor threads = Executor.ForRam(threads: 8);             // Zero derives 7/8 of the cores
using Executor rings = Executor.ForNvme(threads: 4);              // One submission ring apiece
using Executor streams = Executor.ForVram();                      // One stream per device

threads.Drain();                                                  // Awaits every deferred call
```

Each factory also names where its shards go, which is how a dual-socket or multi-GPU box is reached.

```csharp
using Executor domains = Executor.ForRam([0, 1], threads: 16);    // One shard per NUMA node
using Executor devices = Executor.ForVram([0, 1]);                // One stream per named device
using Executor striped = Executor.ForNvme(["/mnt/nvme0", "/mnt/nvme1"]);
```

Every fanned-out verb takes the executor it runs on, and there is no overload that leaves it out: the engine reads a missing executor as a residency mismatch, so the absence is a compile-time one instead.
An executor admits one driver at a time, reporting `Status.Contended` to the second - handles stay freely concurrent, so a caller wanting genuine concurrency gives each thread its own executor.

## Matrices

Every batch argument and every batch output is a `Matrix`: a caller-owned slab viewed as rows of equal width, carrying its row count, its logical width, its byte pitch, and the scalar kind its values hold.
A single vector is a one-row matrix, and the input scalar type is independent of the stored quantization - the library converts on ingest.

```csharp
float[] rows = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
Matrix packed = Matrix.RowsOf<float>(rows, dimensions: 3);              // Three rows, pitch 12 bytes

byte[] bits = new byte[16];
Matrix bitsets = Matrix.RowsOf<byte>(bits, dimensions: 64, ScalarKind.U1x8);   // Two rows, pitch 8 bytes

float[] padded = new float[2 * 8];
Matrix strided = Matrix.StridedRowsOf<float>(                          // The layout a mapped dataset arrives in
    padded, rows: 2, dimensions: 3, strideBytes: 8 * sizeof(float), ScalarKind.F32);
```

The kind travels with the slab rather than being derived from its element type, which is what makes the packed sub-byte kinds reachable: sixty-four `ScalarKind.U1x8` dimensions occupy eight bytes, not sixty-four.
`ScalarKind.U1x8.RowBytes(64)` reports that pitch, and a width leaving a value half filled is refused rather than rounded up.

A `Matrix` borrows and never owns, so it lives only as long as the span it was built from.

## Batch Operations

```csharp
USearchKey[] keys = [1UL, 2UL, 3UL];
TraversalStats stats = new();
index.Add(pool, keys, packed, new DenseInsertionOptions { ExpansionFactor = 128 }, stats);

MatchesSpan found = MatchesSpan.Packed(queries: 3, wanted: 4);
index.Search(pool, packed, found, new DenseSearchOptions { ExpansionFactor = 64 });
for (int query = 0; query < found.Queries; query++)
{
    ReadOnlySpan<USearchKey> matched = found[query].Keys;       // Already trimmed to the true count
    ReadOnlySpan<double> distances = found[query].Distances;
}
```

The candidate-pool breadth rides each call rather than the handle, so a recall sweep runs against one immutable index with no mutation between points.
`TraversalStats` is a pure output, overwritten by every call it is passed to.

A sweep that reuses one set of buffers lends them instead of asking for fresh ones, so nothing is allocated between points, and the `wanted` width follows from the slab lengths rather than being stated a second time:

```csharp
USearchKey[] keySlab = new USearchKey[queries * wanted];
double[] distanceSlab = new double[queries * wanted];
nuint[] countSlab = new nuint[queries];
MatchesSpan lent = MatchesSpan.Of(keySlab, distanceSlab, countSlab);

foreach (int breadth in new[] { 16, 32, 64, 128 })
{
    index.Search(pool, packed, lent, new DenseSearchOptions { ExpansionFactor = (ulong)breadth });
}
```

## Refinement

```csharp
ulong changed = index.Refine(pool, new DenseInsertionOptions { ExpansionFactor = 256 });
```

Refinement re-selects the neighbors of every stored entry against the finished graph, improving edges that were chosen when the graph around them was still incomplete.
It is a build step rather than a repair: a cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass - so a wide expansion factor belongs here rather than in `Add`.
The return is the number of nodes whose edges changed, and zero means the graph has settled, so a caller under a time budget loops until the return falls below whatever rate it considers settled.
Searches may run throughout.

## Filtered Search

```csharp
MatchesSpan admitted = MatchesSpan.Packed(queries: 1, wanted: 10);
index.SearchIf(
    pool, Matrix.RowsOf<float>(vector, dimensions: 3), admitted,
    key => key.TryGetUInt64(out ulong identity) && identity % 2 == 0
        ? Admission.Admit
        : Admission.Reject);
```

The predicate answers with an `Admission` rather than a boolean, matching the two states the C interface names.
It runs on the executor's worker threads, so it must be safe for concurrent use, and it is called once per scored candidate rather than once per reported match.
An exception it throws is captured and raised once the call returns, never unwound through the native frames.

## Late Interaction

```csharp
DenseOptions colbert = DenseOptions.Default with
{
    Dimensions = 128,
    Metric = DenseMetric.Inner,      // Only the inner-product kinds score logits
    Quantization = ScalarKind.F32,
    ValuesPerKey = 32,               // A document is one key holding its token rows
};

MatchesSpan scored = MatchesSpan.Packed(queries: 4, wanted: 10);
index.MaxSim(pool, Matrix.RowsOf<float>(queryTokens, dimensions: 128), rowsPerQuery: 32, scored);
```

Every query row takes its best match within a key, and those maxima sum to that key's score, so the wanted count is the "k" over documents rather than over rows.
That is what a values-per-key bound above one exists for, and a graph frozen under a distance kind refuses with `Status.ConfigMismatch`.

## Retrieval, Cursors, and Streams

```csharp
ulong stored = index.Get(key);                         // Probes the count without copying
float[] vectors = new float[stored * index.Dimensions()];
index.Get(key, Matrix.RowsOf<float>(vectors, dimensions: 3));   // Returns the true count, even when truncated

using USearchCursor cursor = index.Cursor();           // Consistent snapshot, concurrent inserts hidden
USearchKey[] page = new USearchKey[128];
for (int written = cursor.Next(page); written != 0; written = cursor.Next(page)) { }

using USearchFinder finder = index.Find(pool, Matrix.RowsOf<float>(vector, dimensions: 3));
finder.Next(page.AsSpan(0, 8), new double[8]);         // Resumable nearest-neighbor stream
```

A live finder pins one of the executor's threads until it is disposed, so an executor with few threads and many open finders starves.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout: a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.
Checkpoints are residency-portable - one file materializes into RAM, uploads to the devices, or opens in place on disk, decided by the handle that loads it.

```csharp
index.Save(pool, "index.safegraphs");

DenseMetadata metadata = DenseIndex.Metadata("index.safegraphs");  // Geometry without loading

DenseOptions deferred = DenseOptions.Default with { Quantization = ScalarKind.F32 };
using DenseIndex restored = new(deferred);          // Dimensions left zero: the checkpoint freezes them
restored.Load(pool, "index.safegraphs");

byte[] buffer = new byte[index.StorageRequirements()];  // Or one monolith in memory
index.SaveToBuffer(pool, buffer);
restored.LoadFromBuffer(pool, buffer);
```

`DenseMetadata` carries the four fields a checkpoint froze - dimensions, metric, quantization and connectivity.
Residency is absent because a checkpoint records where it was written rather than where a reader must keep it, and `ValuesPerKey` is absent because it bounds a live handle's mutable state rather than the serialized form.

## Residency

```csharp
options = options with { Residency = ResidencyKind.Ram };    // Volatile memory traversed by CPU threads
options = options with { Residency = ResidencyKind.Vram };   // Accelerator memory, sharded across the devices
options = options with { Residency = ResidencyKind.Nvme };   // Persisted, traversed through pluggable I/O engines
```

Placement within a kind - the memory domains, the device ordinals, the device paths - is derived, so an index spreads over whatever the machine offers.
An NVMe handle opens its checkpoint in place, so it is read-only and every mutating verb fails on it with `Status.ReadOnly`.
Bigger indexes come from serializing shards as monoliths and cataloging them in a manifest, which one handle then loads across its own memory domains.

## Failures

Every failure arrives as a `USearchException` carrying a `Status` - `ShapeMismatch`, `UnknownKind`, `Overflow`, `Uninitialized`, and the rest of the vocabulary shared with the engine.

## Interop

Every entry point is declared with `[LibraryImport]`, so the marshalling stubs are source-generated and compiled in rather than reflected at runtime, and the assembly stays trimmable and ahead-of-time compilable.
Engine handles are `SafeHandle` types, which the marshaller reference-counts across every call they cross, so a finalizer can never free a handle out from under a running verb.
