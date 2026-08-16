# USearch for Python

## Installation

```sh
pip install usearch
```

## Quickstart

```python
import numpy as np
from usearch import DenseIndex

index = DenseIndex(
    dimensions=256,         # Scalars per vector
    metric="angular",       # 'inner', 'angular', 'euclidean', 'hamming', or 'jaccard'
    quantization="bf16",    # The stored scalar type: 'f64'...'e2m3', 'i8', 'u8', 'u1x8'
    connectivity=16,        # Connections per node; 0 derives it from the geometry
)

vectors = np.random.rand(10_000, 256).astype(np.float32)
index.reserve(len(vectors))
index.add(np.arange(len(vectors)), vectors)

matches = index.search(vectors[:100], 10)
assert len(index) == 10_000
assert np.asarray(matches.distances).shape == (100, 10)
assert matches[0][0].distance < 0.001
```

The bindings are pure CPython, sitting directly on the library's stable C ABI - no PyBind11, no NanoBind, and no SWIG.
Every verb is a batch, and one vector is a one-row matrix, so a memory-mapped dataset reaches the index with no reshape and no copy.

## Modalities

One index type per modality, each mirroring the C header of the same name:

| Python | C header | Rust |
|---|---|---|
| `usearch.DenseIndex` | `usearch/dense_index.h` | `usearch::dense::Index` |
| `usearch.SequenceIndex` | `usearch/sequence_index.h` | `usearch::sequence::Index` |
| `usearch.MatrixIndex` | `usearch/matrix_index.h` | `usearch::matrix::Index` |
| `usearch.GeospatialIndex` | `usearch/geospatial_index.h` | `usearch::geospatial::Index` |

`DenseIndex` is the one instantiated today; the rest announce themselves with a `NotImplementedError` until their engines land.

## Identities

Keys are 128 bits wide and render as lowercase dashed UUIDs.
Integer identities park in the low eight bytes, so `Key(42)` prints as `00000000-0000-0000-0000-00000000002a`, and keys need not be unique - several entries may share one, and `count` reports how many.

```python
from usearch import Key

Key(42)                                         # from an integer
Key("00000000-0000-0000-0000-00000000002a")     # from the dashed rendering
Key(uuid.uuid4())                               # from a `uuid.UUID`
```

No NumPy dtype expresses a 128-bit integer, so batch identities travel as big-endian sixteen-byte rows - an `(N, 16)` buffer of `uint8` - in both directions.
`usearch.keys` carries `integers_to_keys` and `keys_to_integers` for the common case where identities really are just row numbers.

## Inputs & Outputs

Inputs arrive through PEP 3118 first and `__array_interface__` second, so NumPy is never required and a PyTorch or TensorFlow tensor needs no round-trip through it.
Only the row pitch may stride; a strided column would force a gather the SIMD kernels have no form for.

Results come back as one `Matches` object whose fields are zero-copy views over arenas it owns:

```python
matches = index.search(queries, 10)
matches.keys          # memoryview, shape (queries, 10, 16), format 'B'
matches.distances     # memoryview, shape (queries, 10),     format 'd'
matches.counts        # memoryview, shape (queries,)
matches.visited_members, matches.computed_distances, matches.restarts
matches[0]            # a tuple of `Match` objects, one per hit for query zero
np.asarray(matches.distances)   # zero-copy; NumPy never entered the binding
```

When you already own the output buffers, `search_into` writes into them and allocates nothing:

```python
keys = np.empty((len(queries), 10, 16), dtype=np.uint8)
distances = np.empty((len(queries), 10), dtype=np.float64)
counts = np.empty(len(queries), dtype=np.uint64)
index.search_into(queries, 10, keys, distances, counts)
```

## Expansion Rides The Call

The candidate-pool breadth is a per-call setting rather than a property of the handle, so a recall sweep runs against one immutable index with no mutation between points:

```python
for expansion_factor in (16, 32, 64, 128, 256):
    matches = index.search(queries, 10, expansion_factor=expansion_factor)
    print(expansion_factor, matches.computed_distances)
```

## Filtered Search

A predicate admits keys one at a time.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.
The callback re-enters Python, so it runs under the GIL and serializes the walk - every other verb releases the GIL and lets the library parallelize internally.

```python
matches = index.search(queries, 10, predicate=lambda key: int(key) % 2 == 0)
```

## Retrieval & Iteration

```python
index.count(42)             # entries sharing this key
42 in index                 # the same question, as a membership test
index[42]                   # a (count, dimensions) memoryview of what is stored under it

for keys in index.batches(1024):                    # snapshot walk, keys only
    ...
for keys, vectors in index.batches(1024, dtype="f32"):   # snapshot walk with payloads
    ...
for match in index.find(query):                     # resumable best-first stream
    if match.distance > 0.3:
        break
```

## Serialization

```python
index.save("index.safegraphs")
index.load("index.safegraphs")
```

A handle constructed with `dimensions=0` freezes its geometry from the checkpoint's own headers on `load`.
To learn a checkpoint's shape without opening a shard, ask for its metadata - the keys are this class's own constructor keywords, so it round-trips:

```python
index = DenseIndex(**DenseIndex.metadata("index.safegraphs"))
index.load("index.safegraphs")
```

Both verbs also accept a buffer of `serialized_length` bytes in place of a path.

## Residency

Where the payload lives and which engine traverses it is a constructor keyword, and it also decides what "load" means: materialize into RAM, upload to a device, or open in place on disk.

```python
DenseIndex(dimensions=256, residency="ram", threads=0, pages="huge")
DenseIndex(dimensions=256, residency="vram", device=0, addressing="unified")
DenseIndex(dimensions=256, residency="nvme", sector_bytes=512, queue_depth=0, io="direct")
```

External-memory graphs are never modified in place, so an NVMe handle is read-only and every mutating verb fails on it.
Bigger indexes come from merging:

```python
first.merge(second)   # `second` is consumed and must not be used afterwards
```

## Linear Algebra

Two verbs read the same walk as arithmetic rather than as retrieval.
Both are into-only: the width of what they emit is the caller's to choose, so there is no allocating door to choose it for them.

`multiply_into` reports __every__ pair the traversal scored rather than only the nearest - an approximate `A × Bᵀ`, dense in and sparse out.
A walk computes many times the products it reports and each of them is exact, so the caller stops recomputing what the index already knew.
It returns how many rows truncated, which makes the retry loop a `while`:

```python
while index.multiply_into(rows, wanted, columns, products, counts):
    wanted *= 2
    columns, products, counts = grow(wanted)
```

A count above `wanted` is the truth about that row, not a slice bound - unlike `search_into`, where it can never exceed the width.

`attend_into` folds a softmax over that same walk, the online-softmax recurrence of Flash Attention with the graph replacing the linear sweep over key tiles.
The index holds one layer's keys for one key-value head; `values` holds the value rows, addressed by key, so a context inserted in token order needs no mapping and the publication watermark doubles as the causal mask.

```python
index.attend_into(queries, values, outputs, outputs_dtype="bf16")
```

Ask for `maximum_scores` and `partitions` together to recombine this branch with another over the same queries - a dense recent window, or a second shard of one context - by partial-softmax rescaling.
One alone is a number nothing can use, so passing one without the other is refused.

## Composition & Concurrency

A handle may be shared across threads.
Searches never write to the graph, so any number may overlap with insertions, and saving needs a mutation-free window.
Extensions declare `Py_MOD_GIL_NOT_USED`, so free-threaded builds run without the interpreter lock.

## Tooling

`usearch.io` reads and writes the `fbin` / `ibin` / `ivecs` matrix formats, and `usearch.eval` carries the recall harness:

```python
from usearch.eval import self_recall, random_vectors

stats = self_recall(index)
print(stats.mean_recall, stats.mean_efficiency)
```
