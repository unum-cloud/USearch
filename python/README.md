# USearch for Python

## Installation

```sh
pip install usearch
```

## Quickstart

```python
import numpy as np
import usearch
from usearch import DenseIndex

index = DenseIndex.for_ram(
    dimensions=256,         # Scalars per vector
    metric="angular",       # 'inner', 'angular', 'euclidean', 'hamming', or 'jaccard'
    quantization="bf16",    # The stored scalar type: 'f64'...'e2m3', 'i8', 'u8', 'u1x8'
    connectivity=16,        # Connections per node; 0 derives it from the geometry
)

pool = usearch.Executor.for_ram()   # The workers every fanned-out verb runs on

vectors = np.random.rand(10_000, 256).astype(np.float32)
index.reserve(len(vectors))
index.add(np.arange(len(vectors)), vectors, executor=pool)

matches = index.search(vectors[:100], 10, executor=pool)
assert len(index) == 10_000
assert np.asarray(matches.distances).shape == (100, 10)
assert matches[0][0].distance < 0.001
```

The bindings are pure CPython, sitting directly on the library's stable C ABI - no PyBind11, no NanoBind, and no SWIG.
Every verb is a batch, and one vector is a one-row matrix, so a memory-mapped dataset reaches the index with no reshape and no copy.

## Executors

An executor owns host threads or device streams; an index owns rows.
Keeping them apart is what lets one pool serve several indexes, and what makes the cost of spawning visible at the line that spawns it rather than hidden in a constructor.
Every verb that fans out - `add`, `refine`, `search`, `maxsim`, `merge`, `find`, `save`, `load` and their kin - takes one, and its residency must match the handle's.

One factory per residency, since each spawns something different and takes different arguments:

```python
pool = usearch.Executor.for_ram(threads=8)                          # 0 derives from the hardware
numa_pool = usearch.Executor.for_ram(threads=8, memory_domains=[0, 1])
device_pool = usearch.Executor.for_vram(devices=[0, 1])             # One stream per CUDA ordinal
disk_pool = usearch.Executor.for_nvme(threads=8, paths=["/mnt/nvme0", "/mnt/nvme1"])
pool.drain()                                                        # Waits for every deferred call
```

`for_ram` spawns host worker threads, and its `memory_domains` are the NUMA node ids those threads stay local to.
`for_nvme` opens one submission ring per worker, striping shards across the `paths` it is given.
`for_vram` takes no thread count at all: one host thread drives one stream per device, so the width of that pool is the device list rather than a worker count.
Calling `usearch.Executor()` raises a `TypeError` naming the three factories, since what a pool spawns cannot be derived from nothing.

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
Only the row stride may vary; a strided column would force a gather the SIMD kernels have no form for.

Results come back as one `Matches` object whose fields are zero-copy views over arenas it owns:

```python
matches = index.search(queries, 10, executor=pool)
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
index.search_into(queries, 10, keys, distances, counts, executor=pool)
```

## Expansion Rides The Call

The candidate-pool breadth is a per-call setting rather than a property of the handle, so a recall sweep runs against one immutable index with no mutation between points:

```python
for expansion_factor in (16, 32, 64, 128, 256):
    matches = index.search(queries, 10, executor=pool, expansion_factor=expansion_factor)
    print(expansion_factor, matches.computed_distances)
```

## Filtered Search

A predicate admits keys one at a time.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.
The callback re-enters Python, so it runs under the GIL and serializes the walk - every other verb releases the GIL and lets the library parallelize internally.

```python
matches = index.search(queries, 10, executor=pool, predicate=lambda key: int(key) % 2 == 0)
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
for match in index.find(query, executor=pool):      # resumable best-first stream
    if match.distance > 0.3:
        break
```

## Serialization

```python
index.save("index.safegraphs", executor=pool)
index.load("index.safegraphs", executor=pool)
```

A handle constructed with `dimensions=0` freezes its geometry from the checkpoint's own headers on `load`.
To learn a checkpoint's shape without opening a shard, ask for its metadata - the keys are the five geometry keywords every factory takes, so it round-trips through whichever factory you want the payload in:

```python
index = DenseIndex.for_ram(**DenseIndex.metadata("index.safegraphs"))
index.load("index.safegraphs", executor=pool)
```

Residency is never among those keys, because where the payload lives stays the caller's choice.

Both verbs also accept a buffer of `storage_requirements` bytes in place of a path, holding one monolith of every shard.

## Residency

Where the payload lives and which engine traverses it is the factory you call rather than a keyword you pass, and it also decides what `load` means: materialize into RAM, upload to a device, or open in place on disk.

```python
DenseIndex.for_ram(256, memory_domains=[0, 1])    # One shard per NUMA node
DenseIndex.for_vram(256, devices=[0, 1])          # One shard per CUDA ordinal
DenseIndex.for_nvme(256, paths=["/mnt/nvme0"])    # One shard per path
```

`dimensions`, `metric` and `quantization` stay positional in every factory, and everything after them is keyword-only: `connectivity`, `values_per_key`, and the one residency arm that factory accepts.
Handing a factory another residency's arm - `devices` to `for_ram`, say - raises a `TypeError` rather than being quietly ignored.

Shards are never replicated, so eight devices host eight slices of one graph rather than eight copies, and an index outgrowing them fails rather than paging to the host - the answer is to name more devices.
An NVMe path is whatever identifies a device to an engine that can reach it: a filesystem path names a kernel device, a transport identifier selects a userspace engine, and a FUSE mount is how object storage stages in without an HTTP stack.

External-memory graphs are never modified in place, so an NVMe handle is read-only and every mutating verb fails on it.
Bigger indexes come from merging:

```python
first.merge(second, executor=pool)   # `second` is consumed and must not be used afterwards
```

## Refinement

`refine` re-selects the neighbors of every stored entry against the finished graph, improving edges that were chosen when the graph around them was still incomplete.
It is a build step rather than a repair: a cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass - so a wide `expansion_factor` belongs here rather than in `add`.

```python
while index.refine(executor=pool, expansion_factor=128) > len(index) // 100:
    pass    # Loop until fewer than one node in a hundred still moves an edge
```

The return is how many nodes changed edges, so a caller loops until it falls below whatever change rate they consider settled.
Searches may run throughout, and a stopped sweep resumes where it left off, which is what makes a whole-graph pass affordable under a time budget.

## Linear Algebra

Three verbs read the same walk as arithmetic rather than as retrieval.
All of them need an inner-product index, since a product, a softmax, and a maximum over logits are all things only those kinds score.

`multiply_into` reports __every__ pair the traversal scored rather than only the nearest - an approximate `A × Bᵀ`, dense in and sparse out.
A walk computes many times the products it reports and each of them is exact, so the caller stops recomputing what the index already knew.
It is into-only: the width of the emitted pattern is the caller's to choose, so there is no allocating door to choose it for them.
It returns how many rows truncated, which makes the retry loop a `while`:

```python
while index.multiply_into(rows, wanted, columns, products, counts, executor=pool):
    wanted *= 2
    columns, products, counts = grow(wanted)
```

A count above `wanted` is the truth about that row, not a slice bound - unlike `search_into`, where it can never exceed the width.

`attend_into` folds a softmax over that same walk, the online-softmax recurrence of Flash Attention with the graph replacing the linear sweep over key tiles.
The index holds one layer's keys for one key-value head; `values` holds the value rows, addressed by key, so a context inserted in token order needs no mapping and the publication watermark doubles as the causal mask.

```python
carry = np.zeros((len(queries), 2), dtype=np.float32)
index.attend_into(queries, values, outputs, executor=pool, outputs_dtype="bf16", carry=carry)
```

Ask for the `carry` to recombine this branch with another over the same queries - a dense recent window, or a second shard of one context - by partial-softmax rescaling.
Each of its rows holds one query's running maximum and the softmax denominator taken against it; a maximum without its partition is a number nothing can rescale, which is why they travel as one row rather than as two arrays.

`maxsim` is late interaction: multi-row queries scored against multi-row entries, where every query row takes its best match within a key and those maxima sum to that key's score.
A document is one key holding its token rows, which is what `values_per_key` above one exists for, and the outputs are sized over documents rather than over rows.

```python
index = DenseIndex.for_ram(dimensions=128, metric="inner", values_per_key=32)
...
matches = index.maxsim(query_tokens, 32, 10, executor=pool)   # 32 token rows per query
```

## Composition & Concurrency

A handle may be shared across threads.
Searches never write to the graph, so any number may overlap with insertions, and saving needs a mutation-free window.
Extensions declare `Py_MOD_GIL_NOT_USED`, so free-threaded builds run without the interpreter lock.

## Tooling

`usearch.io` reads and writes the `fbin` / `ibin` / `ivecs` matrix formats, and `usearch.eval` carries the recall harness:

```python
from usearch.eval import self_recall

stats = self_recall(index, executor=pool)
print(stats.mean_recall, stats.mean_efficiency)
```
