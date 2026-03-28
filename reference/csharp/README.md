# USearch for C#

## Installation

```sh
dotnet add package Cloud.Unum.USearch
```

## Quickstart

```csharp
using System.Diagnostics;
using Cloud.Unum.USearch;

using var index = new USearchIndex(
    metricKind: MetricKind.Cos, // Choose cosine metric
    quantization: ScalarKind.Float32, // Only quantization to Float32, Float64, Int8 is currently supported
    dimensions: 3,  // Define the number of dimensions in input vectors
    connectivity: 16, // How frequent should the connections in the graph be, optional
    expansionAdd: 128, // Control the recall of indexing, optional
    expansionSearch: 64 // Control the quality of search, optional
);

var vector = new float[] { 0.2f, 0.6f, 0.4f };
index.Add(42, vector);
int matches = index.Search(vector, 10, out ulong[] keys, out float[] distances);

Trace.Assert(index.Size() == 1);
Trace.Assert(matches == 1);
Trace.Assert(keys[0] == 42);
Trace.Assert(distances[0] <= 0.001f);
```

## Serialization

```csharp
index.Save("index.usearch")

// Copy the whole index into memory
using var indexLoaded = new USearchIndex("index.usearch");

// Or view from disk without loading in memory
// using var indexLoaded = new USearchIndex("index.usearch", view: true);

Trace.Assert(indexLoaded.Size() == 1);
Trace.Assert(indexLoaded.Dimensions() == 3);
Trace.Assert(indexLoaded.Connectivity() == 16);
Trace.Assert(indexLoaded.Contains(42));
```

## Batch Operations

Adding a batch of entries is identical to adding a single vector.

```csharp
using var index = new USearchIndex(MetricKind.Cos, ScalarKind.Float32, dimensions: 3);

// Generate keys and random vectors
int n = 100;
ulong[] keys = Enumerable.Range(0, n).Select(i => (ulong)i).ToArray();
int dims = checked((int)index.Dimensions());
float[][] vectors = Enumerable.Range(0, n)
    .Select(_ => Enumerable.Range(0, dims)
        .Select(__ => (float)new Random().NextDouble() * 0.3f)
        .ToArray())
    .ToArray();

index.Add(keys, vectors);
int matches = index.Search(vectors[0], 10, out ulong[] foundKeys, out float[] foundDistances);

Trace.Assert(matches <= 10);
Trace.Assert(foundKeys[0] == 0);
Trace.Assert(foundDistances[0] <= 0.001f);
```
