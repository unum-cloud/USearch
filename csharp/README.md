# USearch for C#

## Installation

```sh
dotnet add package Cloud.Unum.USearch
```

## Quickstart

```csharp
using Cloud.Unum.USearch;

DenseOptions options = DenseOptions.Default;   // Angular metric, BF16 quantization, RAM residency
options.Metric = DenseMetric.Angular;          // Or Inner, Euclidean, Hamming, Jaccard
options.Quantization = ScalarKind.F32;         // Or F64, BF16, F16, E5M2, E4M3, E3M2, E2M3, I8, U8, U1x8
options.Dimensions = 3;                        // Zero defers the geometry to a later `Load`
options.Connectivity = 16;                     // Connections per node in the graph, optional

using USearchIndex index = new(options);
index.Reserve(1_000);

float[] vector = [0.2f, 0.6f, 0.4f];
index.Add<float>(42UL, vector);

SearchResults found = index.Search<float>(vector, wanted: 10);
USearchKey nearest = found.KeysOf(0)[0];       // 00000000-0000-0000-0000-00000000002a
double distance = found.DistancesOf(0)[0];
```

Keys are 128 bits wide and render as lowercase dashed UUIDs, with integer identities parked in the low eight bytes - so `42UL` converts implicitly, and a full identity comes from `USearchKey.Parse("0123abcd-4567-89ef-fedc-ba9876543210")`.

## Batch Operations

Every verb is a batch over a row-major matrix; a single vector is a one-row matrix.
Rows are packed, so the byte pitch is the row's own width, and the input scalar type is independent of the stored quantization - the library converts on ingest.

```csharp
USearchKey[] keys = [1UL, 2UL, 3UL];
float[] rows = [.. new float[] { 0.1f, 0.2f, 0.3f }, .. new float[] { 0.4f, 0.5f, 0.6f },
                .. new float[] { 0.7f, 0.8f, 0.9f }];

TraversalStats stats = new();
index.Add<float>(keys, rows, expansionFactor: 128, stats: stats);

SearchResults found = index.Search<float>(rows, wanted: 4, expansionFactor: 64);
for (int query = 0; query < found.Queries; query++)
{
    ReadOnlySpan<USearchKey> matched = found.KeysOf(query);   // Already trimmed to the true count
    ReadOnlySpan<double> distances = found.DistancesOf(query);
}
```

The candidate-pool breadth rides each call rather than the handle, so a recall sweep runs against one immutable index with no mutation between points.
`TraversalStats` is a pure output, overwritten by every call it is passed to.

## Filtered Search

```csharp
SearchResults admitted = index.SearchIf<float>(
    vector, wanted: 10, admits: key => key.TryGetUInt64(out ulong identity) && identity % 2 == 0);
```

The predicate runs on the library's worker threads, so it must be safe for concurrent use.
An exception it throws is captured and raised once the call returns, never unwound through the native frames.

## Retrieval, Cursors, and Streams

```csharp
ulong stored = index.Count(key);                       // Several entries may share one key
float[] vectors = new float[stored * index.Dimensions()];
index.Get<float>(key, vectors);                        // Returns the true count, even when truncated

using USearchCursor cursor = index.Snapshot();         // Consistent snapshot, concurrent inserts hidden
USearchKey[] page = new USearchKey[128];
for (int written = cursor.Next(page); written != 0; written = cursor.Next(page)) { }

using USearchFinder finder = index.Find<float>(vector);  // Resumable nearest-neighbor stream
finder.Next(page.AsSpan(0, 8), new double[8]);
```

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout: a `.safegraphs` path is one monolith, a `.safegraphs.json` path is a manifest cataloging many.
Checkpoints are residency-portable - one file materializes into RAM, uploads to a device, or opens in place on disk, decided by the handle that loads it.

```csharp
index.Save("index.safegraphs");

DenseOptions metadata = USearchIndex.Metadata("index.safegraphs");  // Geometry without loading

DenseOptions deferred = DenseOptions.Default;
deferred.Quantization = ScalarKind.F32;             // Dimensions left zero: the checkpoint freezes them
using USearchIndex restored = new(deferred);
restored.Load("index.safegraphs");

byte[] buffer = new byte[index.SerializedLength()];  // Or one monolith in memory
index.SaveToBuffer(buffer);
restored.LoadFromBuffer(buffer);
```

## Residency

```csharp
options.Residency = new Residency { Kind = ResidencyKind.Ram, Threads = 0, HugePages = HugePages.BasePages };
options.Residency = Residency.OnDevice(0, Addressing.UnifiedAddressing);
options.Residency = new Residency { Kind = ResidencyKind.Nvme, SectorBytes = 512, IOMode = IOMode.DirectIO };
```

An NVMe handle opens its checkpoint in place, so it is read-only and every mutating verb fails on it with `Status.ReadOnly`.
Bigger indexes come from merging, never mutating: shards build in RAM, serialize as monoliths, and join a manifest.

```csharp
index.Merge(donor);   // The donor is consumed - its handle is emptied and must not be used again
```

## Failures

Every failure arrives as a `USearchException` carrying a `Status` - `ShapeMismatch`, `UnknownKind`, `Overflow`, `Uninitialized`, and the rest of the vocabulary shared with the engine.
