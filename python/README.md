# USearch for Python

The bindings are pure CPython in C11, sitting directly on the library's stable C ABI - no PyBind11, no NanoBind, and no SWIG.
Every verb is a batch, and one vector is a one-row matrix, so a memory-mapped dataset reaches the index with no reshape and no copy.

## Installation

```sh
pip install usearch
```

## Quickstart

```python
import numpy as np
import usearch
from usearch import DenseIndex, DenseMetric, Scalar

index = DenseIndex.init_ram(
    dimensions=256,                 # Scalars per vector
    metric=DenseMetric.ANGULAR,     # INNER, ANGULAR, EUCLIDEAN, HAMMING or JACCARD
    quantization=Scalar.BF16,       # The stored scalar type: F64...E2M3, I8, U8, U1X8
    connectivity=16,                # Connections per node; 0 derives it from the geometry
)

vectors = np.random.rand(10_000, 256).astype(np.float32)
index.reserve(len(vectors))
index.insert(np.arange(len(vectors)), vectors)          # On the calling thread

pool = usearch.Executor.for_ram()                       # The workers a receiver names
matches = index.on(pool).search(vectors[:100], 10)      # Blocking, on the pool
assert len(index) == 10_000
assert np.asarray(matches.distances).shape == (100, 10)
key, distance = matches[0][0]
assert distance < 0.001
```

## Receivers and Deferred Calls

A receiver carries what every C argument struct puts before its inputs: the executor and the completion.
The index runs a verb on the calling thread, `index.on(executor)` answers a `usearch.On` whose verbs block on that executor, and `index.deferred(executor)` answers a `usearch.DeferredOn` whose verbs submit there and answer a `usearch.Deferred`.
In a coroutine, `await` a deferred call; anywhere else, `wait()` blocks the thread with its state detached:

```python
import asyncio

async def nearest(pool):
    matches = index.search(queries, 10)                 # the calling thread
    matches = index.on(pool).search(queries, 10)        # blocking, after every earlier submission
    call = index.deferred(pool).search(queries, 10)     # usearch.Deferred, submitted now
    call.status                                         # usearch.Status.DEFERRED until final
    pool.drain()                                        # waits for every earlier submission
    return await call                                   # or: call.wait()

matches = asyncio.run(nearest(pool))
```

`await` runs `wait()` on the event loop's default executor, so the thread it blocks belongs to asyncio.
The index and the receivers it derives never change after they are made, so threads share them freely.
Arguments the binding refuses raise at the call, while the engine's refusals raise at `await` or `wait()`, as many times as they are asked.
A submission that finds the executor's deferred table full, sized by `deferred_limit`, finishes with `Status.CONTENDED` and raises `ContendedError` once awaited.
`cursor` and `search_stream` answer iterators whose every step blocks, so a `DeferredOn` lacks them.
Constructors address no index yet, so no receiver offers them; the ones the C ABI lets fan out take `executor=`, and run on the calling thread without one.

## Executors

An executor owns host threads or device streams; an index owns rows.
Keeping them apart is what lets one pool serve several indexes, and what makes the cost of spawning visible at the line that spawns it rather than hidden in a constructor.
Its residency must match that of every index it drives.

One factory per residency, since each spawns something different and takes different arguments:

```python
pool = usearch.Executor.for_ram(threads=8)                          # 0 derives from the hardware
numa_pool = usearch.Executor.for_ram(threads=8, memory_domains=[0, 1])
device_pool = usearch.Executor.for_vram(devices=[0, 1])             # One stream per CUDA ordinal
disk_pool = usearch.Executor.for_nvme(threads=8, paths=["/mnt/nvme0", "/mnt/nvme1"])
shared_pool = usearch.Executor.on_pool(fu_pool_address)             # Borrows a ForkUnion pool by address
pool.drain(usearch.Mood.SPINNING)                                   # Waits, then keeps idle workers hot
```

`for_ram` spawns host worker threads, and its `memory_domains` are the NUMA node IDs those threads stay local to.
`for_nvme` opens one submission ring per worker, striping shards across the `paths` it is given.
`for_vram` takes no thread count at all: one host thread drives one stream per device, so the width of that pool is the device list rather than a worker count.
`on_pool` borrows a ForkUnion `fu_pool_t` another library spawned, passed as an integer address the way `ctypes` and `cffi` hand pointers over, and the pool must outlive the executor.
Every factory takes `deferred_limit=`, the deferred calls it tracks at once, with zero taking a default.
`drain` waits for everything submitted before it, then lets idle workers sleep with `Mood.PARKED` or keeps them hot with `Mood.SPINNING`.
Used as a context manager, an executor waits for its calls and closes on exit, after which a receiver naming it raises `UninitializedError`.
Calling `usearch.Executor()` raises a `TypeError` naming the factories, since what a pool spawns cannot be derived from nothing.

## Modalities

One index type per modality, each mirroring the C header of the same name:

| Python                    | C header                     | Rust                       |
| :------------------------ | :--------------------------- | :------------------------- |
| `usearch.DenseIndex`      | `usearch/dense_index.h`      | `usearch::DenseIndex`      |
| `usearch.SequenceIndex`   | `usearch/sequence_index.h`   | `usearch::SequenceIndex`   |
| `usearch.MatrixIndex`     | `usearch/matrix_index.h`     | `usearch::MatrixIndex`     |
| `usearch.GeospatialIndex` | `usearch/geospatial_index.h` | `usearch::GeospatialIndex` |

`DenseIndex` is the one instantiated today; the rest announce themselves with a `NotImplementedError` until their engines land.

## Identities

Keys are 128 bits wide and render as lowercase dashed UUIDs.
Integer identities sit in the low eight bytes, so `Key(42)` prints as `00000000-0000-0000-0000-00000000002a`, and keys need not be unique - several entries may share one, and `count` reports how many.

```python
from usearch import Key

Key(42)                                         # from an integer
Key("00000000-0000-0000-0000-00000000002a")     # from the dashed rendering
Key(uuid.uuid4())                               # from a `uuid.UUID`
```

A key equals and hashes as the integer or `uuid.UUID` it reads as, never as the text or bytes it parses from, and pickles as its sixteen bytes.
No NumPy dtype expresses a 128-bit integer, so batch identities travel as big-endian sixteen-byte rows - an `(N, 16)` buffer of `uint8` - in both directions.
`usearch.keys` carries `integers_to_keys` and `keys_to_integers` for the common case where identities really are just row numbers.

## Inputs & Outputs

Inputs arrive through PEP 3118 first and `__array_interface__` second, so NumPy is never required and a PyTorch or TensorFlow tensor needs no round-trip through it.
Only the row stride may vary; a strided column would force a gather the SIMD kernels have no form for.
A `dtype=` names the scalar type a buffer cannot spell, such as `Scalar.BF16` over raw bytes, and must be as wide as the buffer's items.
An `outputs_dtype=` names the scalar type a verb exports as - in `get`, `cursor` and `attend` - converted from storage.

Every verb returns its results rather than filling buffers the caller sized.
A ranked verb returns a `Matches` owning three fresh buffers, and reads through to them without copying:

```python
matches = index.on(pool).search(queries, 10)
matches.keys          # memoryview, shape (queries, 10, 16), format 'B'
matches.distances     # memoryview, shape (queries, 10),     format 'd'
matches.counts        # memoryview, shape (queries,)
matches[0]            # a tuple of `Match` struct sequences, one per hit for query zero
np.asarray(matches.distances)   # zero-copy; NumPy never entered the binding
```

A count bounds its own row, and the cells past it are zeros in a fresh run.
A sweep that must not allocate passes a run back as `out=`, which the call lands in and returns, or wraps buffers it already owns:

```python
matches = index.on(pool).search(batches[0], 10)
for batch in batches[1:]:
    index.on(pool).search(batch, 10, out=matches)     # Equal batches, so no allocation

keys = np.zeros((len(queries), 10, 16), dtype=np.uint8)
distances = np.zeros((len(queries), 10), dtype=np.float64)
counts = np.zeros(len(queries), dtype=np.uint64)
index.on(pool).search(queries, 10, out=usearch.Matches(keys, distances, counts))
```

Outputs only the call can size land in a `usearch.Arena` and come back as zero-copy results that keep it alive: `get` answers `usearch.DenseVectors`, `multiply` answers `usearch.DenseProducts`, and `save_buffer` answers `usearch.Monolith`.
Pass `arena=` to reuse one across calls, so a steady loop allocates once; a call refuses with `ContendedError` while any result over the previous outputs is still alive, and a deferred call holds its arena until it is final:

```python
arena, total_bytes = usearch.Arena(), 0
for key in range(10):
    rows = index.get(key, arena=arena)
    if rows is not None:
        total_bytes += memoryview(rows).nbytes
        del rows   # Frees the arena for the next call
```

A call on a VRAM executor leaves device memory in its arena, so its results export `__cuda_array_interface__` for CuPy, PyTorch or Numba and refuse the host buffer protocol.

Traversal telemetry rides a `DenseTraversalStats` the call fully overwrites, so one reading always belongs to one call and never to the sweep around it:

```python
stats = usearch.DenseTraversalStats()
index.on(pool).search(queries, 10, stats=stats)
stats.visited_members, stats.computed_distances, stats.restarts
```

`+=` adds one reading into another when the totals across a sweep are what is wanted:

```python
total = usearch.DenseTraversalStats()
for batch in batches:
    index.on(pool).search(batch, 10, stats=stats)
    total += stats
```

## Errors

Every failure raises a subclass of `usearch.Error`, itself a `RuntimeError`, one per status the C ABI reports, and each carries that status as its `status` attribute, a `usearch.Status` member:

```python
try:
    index.on(pool).search(queries, 10, progress=lambda completed, total: completed < 100)
except usearch.CancelledError:
    ...                         # The handler stopped the call, which published nothing
except usearch.Error as error:
    print(error.status)         # Status.SHAPE_MISMATCH, Status.READ_ONLY, Status.DEVICE, ...
```

`OutOfMemoryError`, `ShapeMismatchError`, `UnsupportedLayoutError`, `UnknownKindError`, `ReadOnlyError`, `InputOutputError`, `DeviceError`, `MalformedError`, `InternalError`, `CapacityOverflowError`, `UninitializedError`, `ConfigMismatchError`, `ContendedError`, `CancelledError`, `UnsupportedError` and `InvalidArgumentError` mirror the C statuses one to one.
Each also derives the built-in naming the same fault, so `except ValueError` catches a `ShapeMismatchError`, `except OSError` an `InputOutputError`, and `except MemoryError` an `OutOfMemoryError`.
A failure the binding catches before the boundary raises the class the engine would have raised for the same fault.

## Expansion Rides The Call

The candidate-pool breadth is a per-call setting rather than a property of the handle, so a recall sweep runs against one immutable index with no mutation between points:

```python
for expansion_factor in (16, 32, 64, 128, 256):
    stats = usearch.DenseTraversalStats()
    index.on(pool).search(queries, 10, expansion_factor=expansion_factor, stats=stats)
    print(expansion_factor, stats.computed_distances)
```

## Callbacks

A predicate admits keys one at a time.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.
The callback re-enters Python from the library's threads, so it runs under the GIL and serializes the walk - every other verb releases the GIL and lets the library parallelize internally.

```python
index.on(pool).search(queries, 10, predicate=lambda key: int(key) % 2 == 0)
```

A `progress` handler is called with `(completed, total)` units; `False` stops the call, and `None`, what a bare `print` answers, carries on.
`insert` is the exception: every row is claimed before the first report, so its handler reports and cannot stop the batch.
The first exception any callback raises outranks the outcome it turned into, whether the call blocked or was deferred.

## Retrieval & Iteration

```python
index.count(42)             # entries sharing this key
42 in index                 # the same question, as a membership test
index[42]                   # a (count, dimensions) `DenseVectors` over a copy of what is stored under it
index.describe()            # a DenseIndexDescription: .metadata, .size, .capacity, .memory_usage, .capabilities
index.health()              # a DenseReachability: .audited, .reachable, .sweeps

for keys in index.cursor(1024):                         # row scan, keys only
    ...
for keys, vectors in index.cursor(1024, outputs_dtype=Scalar.F32, columns=index.dimensions):
    ...                                                 # row scan with payloads
with index.on(pool).search_stream(query) as stream:     # resumable best-first stream
    for key, distance in stream:
        if distance > 0.3:
            break
```

A cursor exporting payloads takes their `columns` from the caller rather than asking the index, so it is given exactly when `outputs_dtype` is.
A cursor or a stream releases its walk on `close()` or at the end of a `with` block, so a `merge` waiting on it need not wait for the collector.

## Serialization

```python
index.on(pool).save("index.safegraphs")
copy = DenseIndex.load_ram("index.safegraphs", executor=pool)  # A new, mutable index in host memory
view = DenseIndex.view_ram("index.safegraphs")      # Served in place, read-only
```

Loading creates a handle rather than filling one, its geometry read from the checkpoint's own headers, and a cancelled or failed load creates none.
The executor a load fans out over is its `executor=` keyword, and without one it reads on the calling thread.
A view maps the file instead of copying it, so opening costs the headers alone, and every mutating verb on it raises `ReadOnlyError`.
To learn a checkpoint's shape without opening a shard, ask for its metadata - a `DenseMetadata` struct sequence whose fields are the geometry every empty constructor takes:

```python
metadata = DenseIndex.metadata_read("index.safegraphs")
empty = DenseIndex.init_ram(
    metadata.dimensions, metadata.metric, metadata.quantization, connectivity=metadata.connectivity
)
```

Residency is never among those fields, because where the payload lives stays the caller's choice.

`save_buffer` returns one monolith of every shard as a `Monolith` over its arena, and `load_ram_buffer` and `metadata_read_buffer` read one back from any buffer.

## Residency

Where the payload lives and which engine traverses it is the constructor you call rather than a keyword you pass, one per C lifecycle verb and residency:

```python
DenseIndex.init_ram(256, memory_domains=[0, 1])                         # Empty, one shard per NUMA node
DenseIndex.init_vram(256, devices=[0, 1])                               # Empty, one shard per CUDA ordinal
DenseIndex.load_vram("index.safegraphs", devices=[0, 1])                # Uploaded from a checkpoint
DenseIndex.view_nvme("index.safegraphs.json", paths=["/mnt/nvme0"])     # Served in place from disk
```

`dimensions`, `metric` and `quantization` stay positional in both empty constructors, and everything after them is keyword-only: `connectivity`, `values_per_key`, and the one residency arm that constructor accepts.
Handing a constructor another residency's arm - `devices` to `init_ram`, say - raises a `TypeError` rather than being quietly ignored.

Shards are never replicated, so eight devices host eight slices of one graph rather than eight copies, and an index outgrowing them fails rather than paging to the host - the answer is to name more devices.
An NVMe path is whatever identifies a device to an engine that can reach it: a filesystem path names a kernel device, a transport identifier selects a userspace engine, and a FUSE mount is how object storage stages in without an HTTP stack.

External-memory graphs are never modified in place, so an NVMe handle is a view and every mutating verb fails on it.
Bigger indexes come from merging:

```python
first.on(pool).merge(second)   # `second` is consumed and must not be used afterwards
```

## Refinement

`refine` re-selects the neighbors of every stored entry against the finished graph, improving edges that were chosen when the graph around them was still incomplete.
It is a build step rather than a repair: a cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass - so a wide `expansion_factor` belongs here rather than in `insert`.

```python
while index.on(pool).refine(expansion_factor=128) > len(index) // 100:
    pass    # Loop until fewer than one node in a hundred still moves an edge
```

The return is how many nodes changed edges, so a caller loops until it falls below whatever change rate they consider settled.
Searches may run throughout on a RAM handle but not on a VRAM one, and a stopped sweep leaves the graph correct for the next call to sweep again, which is what makes a whole-graph pass affordable under a time budget.

## Linear Algebra

Three verbs read the same walk as arithmetic rather than as retrieval.
All of them need an inner-product index, since a product, a softmax, and a maximum over logits are all things only those kinds score.

`multiply` reports __every__ pair the traversal scored rather than only the nearest - an approximate A × Bᵀ, dense in and sparse out.
A walk computes many times the products it reports and each of them is exact, so the caller stops recomputing what the index already knew.
How many pairs a walk scores is known only once it ends, so they come back ragged, with Apache Arrow offsets around them:

```python
products = index.on(pool).multiply(rows, 0)             # 0 keeps every pair; a bound keeps the first ones
offsets = np.asarray(products.offsets)                   # (rows + 1,) - row `i` spans offsets[i] to offsets[i + 1]
columns = np.asarray(products.keys)                      # (pairs, 16) - which indexed row each pair names
scores = np.asarray(products.scores)                     # (pairs,)    - the exact products
products.required                                        # pairs scored before the bound cut any
row_keys, row_scores = products.row(0)                   # one row's pairs, for each of the len(products) rows
```

`attend` is Flash Attention folded into that same walk: every key the walk scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, so the long tail it scores brings the result toward exact dense attention.
The index holds one layer's keys for one key-value head and no values, so `value_lookup` resolves the keys of every scored batch to their value rows, as wide as the keys and stored in the index's quantization.
Grouped-query heads share one walk per token: with `head_count=8`, row `t * 8 + h` of the queries is head `h` of token `t`.

```python
carry = np.zeros((len(queries), 2), dtype=np.float32)
value_lookup = lambda keys: values[keys_to_integers(keys)]  # (count, 16) keys in, (count, dimensions) rows out
outputs = index.on(pool).attend(queries, value_lookup, head_count=8, outputs_dtype=Scalar.BF16, carry=carry)
```

A lookup answering `None` refuses the call with `ShapeMismatchError`, since a missing row is a wrong softmax rather than a smaller one.
It re-enters Python from the library's threads, so it runs under the GIL.
The attended rows come back already divided by the partition; pass `out=` a buffer of the same bytes to land them there instead.

Ask for the `carry` to recombine this branch with another over the same queries - attention sinks, a dense recent window, or a second index of one context - by partial-softmax rescaling.
Keep the recent window out of the index, inserting a token once it leaves the window, so the two branches never count a key twice.
Each of its rows holds one query's largest logit and the softmax denominator taken against it; a maximum without its partition is a number nothing can rescale, which is why they travel as one row rather than as two arrays.

`maxsim` is late interaction: multi-row queries scored against multi-row entries, where every query row takes its best match within a key and those maxima sum to that key's score.
A document is one key holding its token rows, which is what `values_per_key` above one exists for, and the result is sized over documents rather than over rows.

```python
index = DenseIndex.init_ram(dimensions=128, metric=DenseMetric.INNER, values_per_key=32)
...
documents = index.on(pool).maxsim(query_tokens, 32, 10)   # 32 token rows per query, 10 documents each
```

## Composition & Concurrency

A handle may be shared across threads, and so may every receiver derived from it.
Searches never write to the graph, so any number may overlap with insertions, and saving needs a mutation-free window.
The extension declares `Py_MOD_GIL_NOT_USED`, so free-threaded builds run without the interpreter lock; the few fields the binding mutates - a stream's buffer, a latched callback exception, a `DenseTraversalStats` total, a merged donor's handle - are fenced by critical sections or atomics.

## Tooling

`usearch.io` reads and writes the `fbin` / `ibin` / `i8bin` matrix formats, and `usearch.eval` carries the recall harness, which takes the index or any blocking receiver:

```python
from usearch.eval import self_recall

stats = self_recall(index.on(pool))
print(stats.mean_recall, stats.mean_efficiency)
```

## Design

The binding is a thin layer over the C ABI in `include/usearch/`, and every rule below follows from that.

- __Names follow the header.__
  Every type is its C name without `usearch_` and `_t` - `DenseMetric`, `DenseTraversalStats`, `DenseIndexDescription` - every constructor is a C lifecycle verb, and every docstring names the C symbol it calls, or says the method is Python-only.
- __Blocking calls stay as simple as the language allows.__
  `index.search(queries, 10)` takes no executor, completion or receiver argument and returns its result; nothing about asynchrony reaches a caller who never asks for it.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling thread, a `usearch.On` blocks on its executor, and a `usearch.DeferredOn` submits the call there and answers a `usearch.Deferred`.
  The receiver's type alone chooses the completion, so all three offer the same verbs with the same arguments, and no verb takes an `executor=` keyword - only constructors, which no receiver offers, do.
- __A `Deferred` owns its call.__
  It holds the outcome, the arena, the result it lands in and every input buffer the library borrows, so nothing the call reads ever moves.
  `await` hands `wait()` to `loop.run_in_executor`, `wait()` blocks until the call is final, and `status` polls the outcome.
  Dropping an unfinished one waits for its call, which is the C contract rather than binding logic.
- __The binding has no life of its own.__
  It starts no threads and counts no calls; the C executor owns every thread, and every other wait belongs to the C library.
- __Statuses and modes are enums, reports are struct sequences.__
  `usearch.Status`, `usearch.Mood`, `usearch.DenseMetric`, `usearch.Scalar` and `usearch.Residency` are `enum.IntEnum` classes built from tables the extension publishes, valued as the C enumerators, so nothing is ever compared as a string.
  `describe`, `health` and `metadata_read` answer the C structs as struct sequences, never as dicts.
- __Arenas are reused, never shared.__
  A result owns a fresh `usearch.Arena` by default, passing `arena=` reuses one, and a call is refused while any result over the previous outputs is still alive.
- __Language idioms over parallel APIs.__
  The buffer protocol, mapping sugar, struct sequences, context managers and `+=` take the place of helper methods.
