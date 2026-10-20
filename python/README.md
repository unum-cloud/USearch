# USearch for Python

The bindings are pure CPython, with no PyBind11, no NanoBind, and no SWIG.
They need CPython 3.12 or newer on Linux, macOS or Windows, and the extension itself never requires NumPy.
Every verb is a batch, and one vector is a one-row matrix, so a memory-mapped dataset reaches the index with no reshape and no copy.
They expose dense indexes; the other modalities are declared and raise `NotImplementedError` until the library ships their engines.

- `usearch.DenseIndex` searches equidimensional dense embeddings, and is the one instantiated today.
- `usearch.SequenceIndex` searches variable-length sequences.
- `usearch.ShapesIndex` searches 3D point clouds of any size.
- `usearch.GeospatialIndex` searches space and time.

## Installation

```sh
pip install usearch
```

The wheels carry every dense storage type.
A build from source takes its count from `USEARCH_DENSE_DTYPES` in `pyproject.toml`, eleven, and a leaner one can override it:

```sh
pip install . -C cmake.define.USEARCH_DENSE_DTYPES=4  # `bf16`, `u1x8`, `i8` & `e4m3` alone
```

## Quickstart

```python
import numpy as np
import usearch
from usearch import DenseIndex, DenseMetric, Scalar

index = DenseIndex.init_ram(
    dimensions=256,  # Scalars per vector
    metric=DenseMetric.ANGULAR,  # INNER, ANGULAR, EUCLIDEAN, HAMMING or JACCARD
    quantization=Scalar.BF16,  # The stored scalar type: BF16...E2M3, I8, U8, U1X8
    connectivity=16,  # Connections per node; 0 derives it from the geometry
)

vectors = np.random.rand(10_000, 256).astype(np.float32)
index.reserve(len(vectors))
index.insert(np.arange(len(vectors)), vectors)  # On the calling thread

pool = usearch.Executor.for_ram()  # The workers a receiver names
arena = usearch.Arena(pool)  # The memory results land in, made where the pool computes
matches = index.on(pool).search(vectors[:100], 10, arena)  # Blocking, on the pool
assert len(index) == 10_000
assert np.asarray(matches.distances).shape == (100, 10)
key, distance = matches[0][0]
assert distance < 0.001
```

## Receivers & Deferred Calls

A receiver carries what every C argument struct puts before its inputs: the executor and the completion.
The index runs a verb on the calling thread, `index.on(executor)` answers a `usearch.On` whose verbs block on that executor, and `index.deferred(executor)` answers a `usearch.DeferredOn` whose verbs submit there and answer a `usearch.Deferred`.
The receiver's type alone chooses the completion, so all three offer the same verbs with the same arguments, and no verb takes an `executor=` keyword.
In a coroutine, `await` a deferred call; anywhere else, `wait()` blocks the thread with its state detached:

```python
import asyncio


async def nearest(pool):
    here, there = usearch.Arena(), usearch.Arena(pool)
    matches = index.search(queries, 10, here)  # the calling thread
    matches = index.on(pool).search(queries, 10, there)  # blocking, after every earlier submission
    call = index.deferred(pool).search(queries, 10, there)  # a usearch.Deferred, submitted now
    call.status  # usearch.Status.DEFERRED until final
    return await call  # or: call.wait()


matches = asyncio.run(nearest(pool))
```

`await` runs `wait()` on the running event loop's default executor, so the thread it blocks belongs to asyncio.
A `Deferred` holds the outcome and every input buffer the library reads, so nothing the call reads ever moves.
It holds its arena as a blocking call does, until it is waited on or dropped: meanwhile another call naming that arena raises `ContendedError`, and its result views the arena like the blocking verb's.
Dropping an unfinished one waits for its call, which is the C contract rather than binding logic.
The collector leaves a cycle through an unfinished one for a later collection, never clearing what the library still reads.
Constructors address no index yet, so no receiver offers them; the ones the C ABI lets fan out take `executor=`, and run on the calling thread without one.

## Executors

An executor owns host threads or device streams; an index owns rows.
Keeping them apart is what lets one pool serve several indexes, and what makes the cost of spawning visible at the line that spawns it rather than hidden in a constructor.
Its residency must match that of every index it drives.

One factory per residency, since each spawns something different and takes different arguments:

```python
pool = usearch.Executor.for_ram(threads=8)  # 0 derives from the hardware
numa_pool = usearch.Executor.for_ram(threads=8, memory_domains=[0, 1])
device_pool = usearch.Executor.for_vram(devices=[0, 1])  # One stream per CUDA ordinal
disk_pool = usearch.Executor.for_nvme(threads=8, paths=["/mnt/nvme0", "/mnt/nvme1"])
shared_pool = usearch.Executor.on_pool(fu_pool_address)  # Borrows a ForkUnion pool by address
pool.drain(usearch.Mood.SPINNING)  # Waits, then keeps idle workers hot
```

`for_ram` spawns host worker threads, and its `memory_domains` are the NUMA node IDs those threads stay local to.
`for_nvme` opens one submission ring per worker, striping shards across the `paths` it is given.
`for_vram` takes no thread count at all: one host thread drives one stream per device, so the width of that pool is the device list rather than a worker count.
`on_pool` borrows a ForkUnion `fu_pool_t` another library spawned, passed as an integer address the way `ctypes` and `cffi` hand pointers over, and the pool must outlive the executor.
Every factory takes `deferred_limit=`, the deferred calls it tracks at once, with zero taking a default.
A submission that finds that table full finishes with `Status.CONTENDED` and raises `ContendedError` once awaited.
`drain` waits for everything submitted before it, then lets idle workers sleep with `Mood.PARKED` or keeps them hot with `Mood.SPINNING`.
Used as a context manager, an executor waits for its calls and closes on exit, after which a receiver naming it raises `UninitializedError`.
Calling `usearch.Executor()` raises a `TypeError` naming the factories, since what a pool spawns cannot be derived from nothing.

### Concurrency

The index and every receiver derived from it never change after they are made, so threads share them freely.
Searches never write to the graph, so any number may overlap with insertions, and saving needs a mutation-free window.
The extension declares `Py_MOD_GIL_NOT_USED`, so free-threaded builds run without the interpreter lock.
Critical sections or atomics fence the few fields the binding mutates: an arena's exports and generation, the handle a cursor or stream lends its running pull, a closed executor's handle, a latched callback exception, a `DenseTraversalStats` total and an absorbed donor's handle.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in a `usearch.Arena` you pass, and the verb answers a view over it without copying.
`usearch.Arena(executor)` places it where that executor computes, decided once, and `usearch.Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
A NumPy array or `memoryview` over an output pins its arena as a `memoryview` pins a `bytearray`, so the next call naming that arena raises `ContendedError` until the export is released.
A stale output raises `UninitializedError` from `memoryview`, from indexing a `Matches` and from `DenseProducts.row`, but `np.asarray` swallows that error into a 0-d object array, so read through those or copy with `np.array(...)` while the output is fresh.
An arena from a VRAM executor holds device memory, which host accessors refuse with `DeviceError`.
It serves only calls on that executor, any other or none raising `DeviceError`, and holds the executor until it is freed, so an executor closed first keeps its streams until then.
Its views export `__cuda_array_interface__` instead, for CuPy, PyTorch or Numba, and `arena.residency` tells the two apart.

```python
hits, arena = 0, usearch.Arena(pool)
for batch in batches:
    matches = index.on(pool).search(batch, 10, arena)  # rewinds: last batch's matches are gone
    hits += int(np.asarray(matches.counts).sum())  # an array kept past here would pin the arena
found, rows = usearch.Arena(), usearch.Arena()  # two live results, two arenas
for key, _ in index.search(query, 10, found)[0]:
    rerank(key, index.get(key, rows))
arenas = [usearch.Arena(pool) for _ in shards]  # one arena per submitted call
calls = [shard.deferred(pool).search(queries, 10, arena) for shard, arena in zip(shards, arenas)]
results = [call.wait() for call in calls]  # or, in a coroutine: await call
```

`search` answers a `usearch.Matches`, `get` and `attend` answer `usearch.DenseVectors`, a cursor's batches `usearch.DenseBatch`, `multiply` and `maxsim` answer `usearch.DenseProducts`, and `save_buffer` answers `usearch.Monolith`.
Their `len()` counts rows, or bytes of a monolith.

## Keys

Keys are 128 bits wide and render as lowercase dashed UUIDs.
Integer keys sit in the low eight bytes, so `Key(42)` prints as `00000000-0000-0000-0000-00000000002a`, and keys need not be unique: several entries may share one, and `count` reports how many.

```python
import uuid
from usearch import Key

Key(42)  # from an integer
Key("00000000-0000-0000-0000-00000000002a")  # from the dashed rendering
Key(uuid.uuid4())  # from a `uuid.UUID`
```

A key equals and hashes as the integer or `uuid.UUID` it reads as, never as the text or bytes it parses from, and pickles as its sixteen bytes.
No NumPy dtype expresses a 128-bit integer, so batch keys travel in both directions as big-endian sixteen-byte rows, an `(N, 16)` buffer of `uint8`.
`usearch.keys` carries `integers_to_keys` and `keys_to_integers` for the common case where keys really are just row numbers.

## Errors

Every failure raises a subclass of `usearch.Error`, itself a `RuntimeError`, and each carries the status the C ABI reported as its `status` attribute, a `usearch.Status` member:

```python
try:
    index.on(pool).search(queries, 10, arena, progress=lambda completed, total: completed < 100)
except usearch.CancelledError:
    ...  # The handler stopped the call, which published nothing
except usearch.Error as error:
    print(error.status)  # Status.SHAPE_MISMATCH, Status.READ_ONLY, Status.DEVICE, ...
```

`usearch.Status` is an `enum.IntEnum` valued as `usearch_status_t`.
`SUCCESS` and `DEFERRED` raise nothing, and every other member has its own class, which also derives the built-in naming the same fault where one exists:

- `OUT_OF_MEMORY` → `OutOfMemoryError`, a `MemoryError`
- `SHAPE_MISMATCH` → `ShapeMismatchError`, a `ValueError`
- `UNSUPPORTED_LAYOUT` → `UnsupportedLayoutError`, a `ValueError`
- `UNKNOWN_KIND` → `UnknownKindError`, a `ValueError`
- `READ_ONLY` → `ReadOnlyError`
- `IO` → `InputOutputError`, an `OSError`
- `DEVICE` → `DeviceError`
- `MALFORMED` → `MalformedError`, a `ValueError`
- `INTERNAL` → `InternalError`
- `OVERFLOW` → `CapacityOverflowError`, an `OverflowError`
- `UNINITIALIZED` → `UninitializedError`
- `CONFIG_MISMATCH` → `ConfigMismatchError`, a `ValueError`
- `CONTENDED` → `ContendedError`
- `CANCELLED` → `CancelledError`
- `UNSUPPORTED` → `UnsupportedError`, a `NotImplementedError`
- `INVALID_ARGUMENT` → `InvalidArgumentError`, a `ValueError`

So `except ValueError` catches a `ShapeMismatchError`, `except OSError` an `InputOutputError`, and `except MemoryError` an `OutOfMemoryError`.
A failure the binding catches before the boundary raises the class the engine would have raised for the same fault.
Arguments the binding refuses raise at the call, while the engine's refusals of a deferred call raise at `await` or `wait()`, as many times as they are asked.

## Residency

Where the payload lives and which engine traverses it is the constructor you call rather than a keyword you pass, one per C lifecycle verb and residency:

```python
DenseIndex.init_ram(256, memory_domains=[0, 1])  # Empty, one shard per NUMA node
DenseIndex.init_vram(256, devices=[0, 1])  # Empty, one shard per CUDA ordinal
DenseIndex.load_vram("index.safegraphs", devices=[0, 1])  # Uploaded from a checkpoint
DenseIndex.view_nvme("index.safegraphs.json", paths=["/mnt/nvme0"])  # Served in place from disk
```

`dimensions`, `metric` and `quantization` stay positional in both empty constructors, and everything after them is keyword-only: `connectivity`, `values_per_key`, and the one residency arm that constructor accepts.
Handing a constructor another residency's arm, such as `devices` to `init_ram`, raises a `TypeError` rather than being quietly ignored.

Shards are never replicated, so eight devices host eight slices of one graph rather than eight copies, and an index outgrowing them fails rather than paging to the host, so the answer is to name more devices.
An NVMe path is whatever identifies a device to an engine that can reach it: a filesystem path names a kernel device, a transport identifier selects a userspace engine, and a FUSE mount is how object storage stages in without an HTTP stack.
External-memory graphs are never modified in place, so an NVMe handle is a view and every mutating verb fails on it; bigger ones come from [absorbing](#absorbing--splitting).

## Serialization

```python
index.on(pool).save("index.safegraphs")
copy = DenseIndex.load_ram("index.safegraphs", executor=pool)  # A new, mutable index in host memory
view = DenseIndex.view_ram("index.safegraphs")  # Served in place, read-only
```

Loading creates a handle rather than filling one, its geometry read from the checkpoint's own headers, and a cancelled or failed load creates none.
The executor a load fans out over is its `executor=` keyword, and without one it reads on the calling thread.
A view maps the file instead of copying it, so opening costs the headers alone, and every mutating verb on it raises `ReadOnlyError`.
To learn a checkpoint's shape without opening a shard, ask for its metadata, a `DenseMetadata` struct sequence whose fields are the geometry every empty constructor takes:

```python
metadata = DenseIndex.metadata_read("index.safegraphs")
empty = DenseIndex.init_ram(
    metadata.dimensions, metadata.metric, metadata.quantization, connectivity=metadata.connectivity
)
```

Residency is never among those fields, because where the payload lives stays the caller's choice.

`save_buffer` returns one monolith of every shard as a `Monolith` over the arena it is handed, and `load_ram_buffer` and `metadata_read_buffer` read one back from any buffer.

## Dense Indexes

### Rows & Scalars

Inputs arrive through PEP 3118 first and `__array_interface__` second, so NumPy is never required and a PyTorch or TensorFlow tensor needs no round-trip through it.
Only the row stride may vary; a strided column would force a gather the SIMD kernels have no form for.
A `dtype=` names the scalar type a buffer cannot spell, such as `Scalar.BF16` over raw bytes, and must be as wide as the buffer's items.
In `get` and `attend`, an `outputs_dtype=` names the scalar type the verb exports as, converted from storage.

### Search

A ranked verb answers a `Matches` that reads through to three outputs in the arena:

```python
arena = usearch.Arena(pool)
matches = index.on(pool).search(queries, 10, arena)
matches.keys  # shape (queries, columns, 16), format 'B', columns at most 10
matches.distances  # shape (queries, columns),     format 'd'
matches.counts  # shape (queries,)
matches[0]  # a tuple of `Match` struct sequences, one per hit for query zero
np.asarray(matches.distances)  # zero-copy; NumPy never entered the binding
```

A count bounds its own row, and the cells past it hold nothing a call wrote.

The breadth of the candidates is a per-call setting rather than a property of the handle, so a recall sweep runs against one immutable index with no mutation between points.
Traversal telemetry rides a `DenseTraversalStats` the call fully overwrites, so one reading of `visited_members`, `computed_distances` and `restarts` always belongs to one call and never to the sweep around it:

```python
for expansion_factor in (16, 32, 64, 128, 256):
    stats = usearch.DenseTraversalStats()
    index.on(pool).search(queries, 10, arena, expansion_factor=expansion_factor, stats=stats)
    print(expansion_factor, stats.computed_distances)
```

`traversal=usearch.Traversal.SCAN` skips the walk and scores every row exactly, on RAM and NVMe handles, which gives such a sweep its ground truth.
`+=` adds one reading into another when the totals across a sweep are what is wanted:

```python
total = usearch.DenseTraversalStats()
for batch in batches:
    index.on(pool).search(batch, 10, arena, stats=stats)
    total += stats
```

### Filtering & Progress

A predicate admits keys one at a time.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.
The callback re-enters Python from the library's threads, so it runs under the GIL and serializes the walk, while every other verb releases the GIL and lets the library parallelize internally.

```python
index.on(pool).search(queries, 10, arena, predicate=lambda key: int(key) % 2 == 0)
```

A `progress` handler is called with `(completed, total)` units; `False` stops the call, and `None`, what a bare `print` answers, carries on.
`insert` is the exception: every row is claimed before the first report, so its handler reports and cannot stop the batch.
The first exception any callback raises outranks the outcome it turned into, whether the call blocked or was deferred.

### Retrieval: Get, Cursors & Streams

```python
index.get(42, arena)  # a (count, dimensions) `DenseVectors` of what is stored under it
cursor = index.on(pool).cursor()  # row scan, pulled in batches; `cursor([42])` walks out from key 42
while batch := cursor.next(1024, arena, links=True):
    ...  # batch.keys, and batch.links as (offsets, keys) when asked; rows come from `get`
with index.on(pool).search_stream(query) as stream:  # resumable best-first stream
    while pulled := stream.next(16, arena)[0]:  # one row of `Matches`, empty once drained
        if pulled[-1].distance > 0.3:
            break
```

`get` for an absent key answers an empty, falsy `DenseVectors`.
Every pull names its arena and rewinds it like any other call, so one arena serves a scan and any other verb in turn.
A drained cursor answers an empty, falsy `DenseBatch`, and a drained stream a row with no matches.
A cursor or a stream releases its walk on `close()` or at the end of a `with` block, so an `absorb` waiting on it need not wait for the collector, and a later pull raises `UninitializedError`.

### Refinement

`refine` re-selects the neighbors of every stored entry against the finished graph, improving edges that were chosen when the graph around them was still incomplete.
It is a build step rather than a repair: a cheap insertion pass followed by one refinement reaches a better graph, in less total time, than a single expensive insertion pass, so a wide `expansion_factor` belongs here rather than in `insert`.

```python
while index.on(pool).refine(expansion_factor=128) > len(index) // 100:
    pass  # Loop until fewer than one node in a hundred still moves an edge
```

The return is how many nodes changed edges, so a caller loops until it falls below whatever change rate they consider settled.
Searches may run throughout on a RAM handle but not on a VRAM one, and a stopped sweep leaves the graph correct for the next call to sweep again, which is what makes a whole-graph pass affordable under a time budget.

### Absorbing & Splitting

```python
first.on(pool).absorb(second, third)  # both donors are consumed and must not be used afterwards
```

Donors are absorbed one after another, each keeping its graph and linked in by walks across the appended part.
A later verb on a consumed donor raises `UninitializedError`, and a stopped `absorb` hands every donor it never reached back to its wrapper.

### Sparse Multiplication

`multiply`, `maxsim` and `attend` read the walk as arithmetic rather than as retrieval.
All of them need an inner-product index, since a product, a maximum over logits, and a softmax are all things only those kinds score.

`multiply` reports __every__ pair the traversal scored rather than only the nearest: an approximate A × Bᵀ, dense in and sparse out.
A walk computes many times the products it reports and each of them is exact, so the caller stops recomputing what the index already knew.
How many pairs a walk scores is known only once it ends, so they come back ragged, with Apache Arrow offsets around them:

```python
products = index.on(pool).multiply(rows, 0, arena)  # 0 keeps every pair; a bound keeps the first ones
offsets = np.asarray(products.offsets)  # (rows + 1,): row `i` spans offsets[i] to offsets[i + 1]
columns = np.asarray(products.keys)  # (pairs, 16): which indexed row each pair names
scores = np.asarray(products.scores)  # (pairs,): the exact products
products.required  # pairs scored before the bound cut any
row_keys, row_scores = products.row(0)  # one row's pairs, for each of the len(products) rows
```

### Late Interaction

`maxsim` scores multi-row queries against multi-row entries, where every query row takes its best match within a key and those maxima sum to that key's score.
A document is one key holding its token rows, which is what `values_per_key` above one exists for, and the result is the ragged run `multiply` answers, one run of documents per query, largest first.
A zero `wanted` keeps every document a query scored.

```python
tokens = DenseIndex.init_ram(dimensions=128, metric=DenseMetric.INNER, values_per_key=32)
...
documents = tokens.on(pool).maxsim(query_tokens, 32, 10, arena)  # 32 token rows per query, up to 10 documents each
offsets = np.asarray(documents.offsets)  # (queries + 1,): query `i` spans offsets[i] to offsets[i + 1]
```

### Attention

`attend` is Flash Attention folded into the walk: every key the walk scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, so the long tail it scores brings the result toward exact dense attention.
The index holds one layer's keys for one key-value head and no values, so `row_of` names the row of `values` each scored key reads, the rows as wide as the keys and stored in the index's quantization.
Grouped-query heads share one walk per token: with `head_count=8`, row `t * 8 + h` of the queries is head `h` of token `t`.

```python
carry = np.zeros((len(queries), 2), dtype=np.float32)
row_of = {key: row for row, key in enumerate(context_keys)}.get  # a scored Key in, its row of `values` out
outputs = index.on(pool).attend(queries, values, row_of, arena, head_count=8, outputs_dtype=Scalar.BF16, carry=carry)
```

A `row_of` answering `None`, or a row outside `values`, refuses the call with `ShapeMismatchError`, since a missing row is a wrong softmax rather than a smaller one; so do values of another width or scalar type, which `values_dtype=` names when the buffer cannot.
It re-enters Python from the library's threads, so it runs under the GIL.
A VRAM walk resolves values on the device, where no Python callable runs, so `attend` with a VRAM arena raises `DeviceError`.
The attended rows come back already divided by the partition, as a `DenseVectors` over the arena.

Ask for the `carry` to recombine this branch by partial-softmax rescaling with another over the same queries, such as attention sinks, a dense recent window, or a second index of one context.
Keep the recent window out of the index, inserting a token once it leaves the window, so the two branches never count a key twice.
Each of its rows holds one query's largest logit and the softmax denominator taken against it; a maximum without its partition is a number nothing can rescale, which is why they travel as one row rather than as two arrays.

### Introspection

```python
index.count(42)  # entries sharing this key
42 in index  # the same question, as a membership test
len(index)  # stored entries
index.describe()  # a DenseIndexDescription: .metadata, .size, .capacity, .memory_usage, .capabilities
index.health()  # a DenseReachability: .audited, .reachable, .sweeps
index.capabilities  # the kernel family dispatch picked on this machine
usearch.capabilities_compiled()  # the kernel families this build carries
usearch.capabilities_available()  # the ones this machine can run
```

`capacity`, `dimensions`, `connectivity` and `memory_usage` are properties read through `describe` as well.
`health` audits how much of the graph a search reaches, and the entries it cannot reach wait for a `refine` to re-link them.
Every docstring names the C symbol it calls, or says the method is Python-only.

## Tooling

`usearch.io` reads and writes the `fbin` / `ibin` / `i8bin` matrix formats, and `usearch.eval` carries the recall harness, which takes the index or any blocking receiver:

```python
from usearch.eval import self_recall

stats = self_recall(index.on(pool), usearch.Arena(pool))
print(stats.mean_recall, stats.mean_efficiency)
```

## What This Binding Does Not Expose

- `cursor` and `search_stream` on a `DeferredOn`: they open walks whose every pull blocks, so there is nothing to submit.
- `usearch_dense_metric_from_name`: `DenseMetric` is an `enum.IntEnum`, so `DenseMetric["ANGULAR"]` parses a name and no string ever reaches a verb.

## Design

- __A thin layer over the C ABI.__
  Every name follows `c/usearch/`: a C verb `usearch_dense_<verb>` is a `<verb>` here, a type drops `usearch_` and `_t`, and enums mirror C enums as `enum.IntEnum` classes, never strings.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `usearch_outcome_status`, `wait` blocks through `usearch_outcome_wait`, and `await` hands `wait()` to the running event loop's default executor.
- __No life of its own.__
  The binding starts no threads and keeps no queues; releasing a `Deferred` rides reference counting, which waits for the call first.
- __Arenas are lent and reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes; an exported buffer pins its arena until released, and a view over a rewound one raises `UninitializedError`.
- __Nothing costs anything until it is used.__
  Blocking calls take no deferred-table slot, and an arena allocates only when a call writes into it.
- __Blocking calls stay as simple as the language allows.__
  `index.search(queries, 10, arena)` takes no executor, completion or receiver argument and returns its result; nothing about asynchrony reaches a caller who never asks for it.
- __Language idioms over parallel APIs.__
  The buffer protocol, `in`, struct sequences, context managers and `+=` take the place of helper methods, and `describe`, `health` and `metadata_read` answer the C structs as struct sequences, never as dicts.
