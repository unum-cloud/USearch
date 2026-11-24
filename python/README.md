# USearch for Python

The bindings are pure CPython, with no PyBind11, no NanoBind, and no SWIG.
They need CPython 3.12 or newer on Linux, macOS or Windows, and the extension itself never requires NumPy.
Every verb is a batch, and one vector is a one-row matrix, so a memory-mapped dataset reaches the index with no reshape and no copy.
Every modality forwards to the C library, and a residency or verb a build cannot serve raises the status the library answers.

- `usearch.DenseIndex` searches equidimensional dense embeddings.
- `usearch.SequenceIndex` searches variable-length sequences, a batch being a list of `str` or bytes-like objects.
- `usearch.ShapesIndex` searches 3D point clouds of any size, a batch being a list of `(points, 3)` matrices.
- `usearch.GeospatialIndex` searches observations in space and time, a batch being an `(N, 11)` float64 matrix.

Every modality takes the same receivers, `refine` and `reshard`, each with its own pair of plan types, such as `usearch.ShapesRefinePlan` and `usearch.ShapesReshardPlan`.

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
index.reserve(len(vectors))  # Optional capacity hint; inserts grow on demand
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

The index runs a verb on the calling thread, `index.on(executor)` answers a `usearch.On` whose verbs block on that executor, and `index.deferred(executor)` a `usearch.DeferredOn` whose verbs submit there and answer a `usearch.Deferred`.
The receiver's type alone chooses the completion, so all three offer the same verbs with the same arguments, and no verb takes an `executor=` keyword.

```python
async def nearest(pool, arena):
    call = index.deferred(pool).search(queries, 10, arena)  # a usearch.Deferred, submitted now
    call.status  # usearch.Status.DEFERRED until final
    return await call  # or, outside a coroutine: call.wait()
```

`await` polls the status between sleeps that double from 1 µs to 1 ms, so the coroutine suspends and no thread blocks.
A `Deferred` holds its outcome, its arena and every input buffer the library reads until it is waited on or dropped, and dropping an unfinished one waits for its call.
The full rules live in the [C README](../c/README.md#receivers--deferred-calls).

## Executors

An executor owns host threads or device streams, and an index owns entries, so one pool serves several indexes and the cost of spawning shows at the line that spawns.
Its residency must match that of every index it drives:

```python
pool = usearch.Executor.for_ram(threads=8, memory_domains=[0, 1])  # 0 threads derives from the hardware
device_pool = usearch.Executor.for_vram(devices=[0, 1])  # One host thread, one stream per device
disk_pool = usearch.Executor.for_nvme(threads=8, paths=["/mnt/nvme0", "/mnt/nvme1"])
pool.drain(usearch.Mood.SPINNING)  # Waits, then keeps idle workers hot
```

`memory_domains` are the NUMA nodes the threads stay local to, and `for_nvme` opens one submission ring per worker.
One call holds an executor at a time and nothing queues, so a second call raises `ContendedError`.
A deferred call on a pool of several threads runs on its thread 0, so polling `status` alone finishes it, and its callbacks run on the pool's threads under the GIL.
Used as a context manager, an executor waits for its calls and closes on exit.

### Concurrency

The index and its receivers never change after they are made, so threads share them freely, and the extension declares `Py_MOD_GIL_NOT_USED` for free-threaded builds.
Searches never write to the graph, so any number may overlap with insertions, while saving needs a mutation-free window.
An exception escaping a `progress`, `predicate` or `row_of` callback goes to `sys.unraisablehook`, and the callback answers cancel, or rejects its key.

## Arenas

Every output whose length only the call knows lands in a `usearch.Arena` you pass, and the verb answers a view over it without copying.
`usearch.Arena(executor)` places it where that executor computes, and `usearch.Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so a steady loop allocates once and the previous call's views go stale; copy a result with `np.array(...)` or pass the next call another arena.
An arena serves one call at a time, so each deferred call needs its own.
A VRAM arena holds device memory and serves only its executor's calls, its views exporting `__cuda_array_interface__`, while every view exports `__dlpack__`, so `torch.from_dlpack(vectors)` takes it in place.

```python
found, fetched = usearch.Arena(), usearch.Arena()  # Two live results, two arenas
for key, _ in index.search(query, 10, found)[0]:
    rerank(key, index.get(key, fetched))
```

`search` answers a `usearch.Matches`, `get` and `attend` a `usearch.DenseVectors`, a cursor's batches a `usearch.DenseBatch`, `multiply` and `maxsim` a `usearch.DenseProducts`, and `save_buffer` a `usearch.Monolith`.

## Keys

Keys are 128 bits wide and render as lowercase dashed UUIDs.
Integer keys sit in the low eight bytes, so `Key(42)` prints as `00000000-0000-0000-0000-00000000002a`, and several entries may share one key, as `count` reports.

```python
import uuid
from usearch import Key

Key(42)  # from an integer
Key("00000000-0000-0000-0000-00000000002a")  # from the dashed rendering
Key(uuid.uuid4())  # from a `uuid.UUID`
```

A key equals and hashes as the integer or `uuid.UUID` it reads as, and pickles as its sixteen bytes.
No NumPy dtype holds a 128-bit integer, so batch keys travel as big-endian sixteen-byte rows, an `(N, 16)` buffer of `uint8`, and `usearch.keys` carries `integers_to_keys` and `keys_to_integers` for keys that are entry numbers.

## Errors

Every failure raises a subclass of `usearch.Error`, a `RuntimeError` carrying the C status as its `status` attribute, a `usearch.Status` member:

```python
try:
    index.on(pool).search(queries, 10, arena, progress=lambda completed, total: completed < 100)
except usearch.CancelledError:
    ...  # The handler stopped the call, which published nothing
except usearch.Error as error:
    print(error.status)  # Status.SHAPE_MISMATCH, Status.READ_ONLY, Status.DEVICE, ...
```

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

Arguments the binding refuses raise at the call, while the engine's refusals of a deferred call raise at `await` or `wait()`, as many times as they are asked.

## Residency

Where the payload lives is the constructor you call rather than a keyword you pass, one per C lifecycle verb and residency:

```python
DenseIndex.init_ram(256, memory_domains=[0, 1])  # Empty, one shard per NUMA node
DenseIndex.init_vram(256, devices=[0, 1])  # Empty, one shard per device
DenseIndex.load_vram("index.safegraphs", devices=[0, 1])  # Uploaded from a checkpoint
DenseIndex.view_nvme("index.safegraphs.json", paths=["/mnt/nvme0"])  # Served in place from disk
```

`dimensions`, `metric` and `quantization` stay positional, and the rest is keyword-only, including the one residency arm a constructor accepts; another's, such as `devices` to `init_ram`, raises a `TypeError`.
Shards are never replicated, so eight devices host eight slices of one graph, and an index outgrowing them fails rather than paging to the host.
An NVMe path is a filesystem path for a kernel engine, a transport identifier for a userspace one, or a FUSE mount over object storage.
An NVMe view refuses every mutating verb, and `key_table_domains` holds its key table in RAM; without one, `count`, `get`, `sample`, `maxsim` and hinted searches raise `UnsupportedError`.
`load_nvme` opens a manifest writable instead, so `refine` rewrites its parts one at a time, while a lone monolith raises `UnsupportedError`.

## Serialization

```python
index.on(pool).save("index.safegraphs")
copy = DenseIndex.load_ram("index.safegraphs", executor=pool)  # A new, mutable index in host memory
view = DenseIndex.view_ram("index.safegraphs")  # Served in place, read-only
```

Loading creates a handle, its geometry read from the checkpoint's headers, fanning out over its `executor=` or reading on the calling thread, and a cancelled or failed load creates none.
A view maps the file, so opening costs the headers alone, and every mutating verb on it raises `ReadOnlyError`.
`metadata_read` answers a `DenseMetadata` whose fields are the geometry every empty constructor takes, never the residency, which stays the caller's choice.
`save_buffer` writes one monolith of every shard into its arena, and `load_ram_buffer` and `metadata_read_buffer` read one back from any buffer.

## Dense Indexes

### Rows & Scalars

Inputs arrive through DLPack first, then PEP 3118 and `__array_interface__`, so a PyTorch, JAX or CuPy tensor needs no round-trip through NumPy, on the host or on its GPU.
Only the row stride may vary, and a `dtype=` names the scalar type a buffer cannot spell, such as `Scalar.BF16` over raw bytes.
In `get` and `attend`, an `outputs_dtype=` names the scalar type the verb exports as, converted from storage.

### Search

A ranked verb answers a `Matches` that reads through to three outputs in the arena, each row bounded by its count:

```python
stats = usearch.DenseTraversalStats()  # Overwritten by every call, summed with `+=`
matches = index.on(pool).search(queries, 10, arena, expansion_factor=64, stats=stats)
matches.keys  # shape (queries, columns, 16), format 'B', columns at most 10
matches.distances  # shape (queries, columns), format 'd'
matches.counts  # shape (queries,)
matches[0]  # a tuple of `Match` struct sequences, one per hit for query zero
```

The breadth of the candidates is a per-call setting, so a recall sweep runs against one immutable index, and `stats` reports each call's `visited_members`, `computed_distances` and `restarts`.
`traversal=usearch.Traversal.SCAN` scores every entry exactly on RAM and NVMe handles, which gives such a sweep its ground truth.

### Filtered Search

A predicate admits keys one at a time, while non-matching entries still route the traversal, so recall does not collapse at low selectivity.
It re-enters Python from the library's threads, so it runs under the GIL and serializes the walk.

```python
index.on(pool).search(queries, 10, arena, predicate=lambda key: int(key) % 2 == 0)
```

A `progress` handler is called with `(completed, total)` units, and `False` stops the call, except in `insert`, which claims every entry before its first report.

### Retrieval and Streaming

```python
index.get(42, arena)  # a (count, dimensions) `DenseVectors` of what is stored under it
cursor = index.on(pool).cursor()  # entry scan, pulled in batches; `cursor([42])` walks out from key 42
while batch := cursor.next(1024, arena, links=True):
    ...  # batch.keys, and batch.links as (offsets, keys) when asked; vectors come from `get`
with index.on(pool).search_stream(query) as stream:  # resumable best-first stream
    while pulled := stream.next(16, arena)[0]:  # one query of `Matches`, empty once drained
        if pulled[-1].distance > 0.3:
            break
```

An absent key, a drained cursor and a drained stream all answer empty, falsy results.
A cursor or a stream releases its walk on `close()` or at the end of a `with` block.

### Refinement

`refine` re-selects the neighbors of every stored entry against the finished graph, so a cheap insertion pass and one refinement reach a better graph, in less total time, than one expensive insertion pass.

```python
while index.on(pool).refine(expansion_factor=128).verdict == usearch.RefineVerdict.IMPROVED:
    pass  # Loop until a round changes no entry's links, stalls or is stopped
```

The `RefineReport` counts the nodes re-selected and `changed`, the islands reconnected, the `neighborhoods` re-stitched and the links added and dropped, beside the `verdict` and the audits around the round.
A stopped round still re-links its islands before raising `CancelledError`, and a `usearch.DenseRefinePlan` passed as `plan=` continues it in the next call until `refine_plan_finish` closes it.

### Resharding

```python
first.on(pool).reshard([first, second, third], [first])  # gathers every entry into `first`
first.on(pool).reshard([first], [left, right], seed=7)  # cuts `first` between two empty targets
first.on(pool).reshard([first], [first], memory_domains=[], shard_capacity=10_000_000)  # recuts in place
```

Entries move with their links, then every target that took entries runs one round, answering its `RefineReport`; a source the targets do not name ends empty and stays the caller's.
A `usearch.DenseReshardPlan` resumes a stopped reshard as a refine plan does.

### Sampling

```python
drawn = index.on(pool).sample(256, arena, shown=shown, shown_coordinates=xyz, focus=near, seed=42)
```

`sample` draws keys around `focus`, each hanging from a shown or earlier draw, and places them in three dimensions beside the `shown` keys.
The `Sample` holds `keys`, `entry_ordinals`, `parents` and `coordinates`, and a larger `wanted` under the same seed draws a superset.

### Sparse Multiplication

`multiply` reports __every__ pair the traversal scored rather than only the nearest: an approximate A × Bᵀ, dense in and sparse out, each product exact.
It needs an inner-product index, as `maxsim` and `attend` do, and the pairs come back ragged, with Apache Arrow offsets around them:

```python
products = index.on(pool).multiply(rows, 0, arena)  # 0 keeps every pair; a bound keeps the first ones
offsets = np.asarray(products.offsets)  # (rows + 1,): left row `i` spans offsets[i] to offsets[i + 1]
columns = np.asarray(products.keys)  # (pairs, 16): which indexed entry each pair names
scores = np.asarray(products.scores)  # (pairs,): the exact products
row_keys, row_scores = products.query(0)  # one left row's pairs
```

### Late Interaction

`maxsim` scores multi-row queries against documents, each a key holding its token rows, summing every query row's best match within a key.
It answers the ragged run `multiply` does, one run of documents per query, largest first, and a zero `wanted` keeps every document scored.

```python
tokens = DenseIndex.init_ram(dimensions=128, metric=DenseMetric.INNER, values_per_key=32)
documents = tokens.on(pool).maxsim(query_tokens, 32, 10, arena)  # 32 token rows per query, up to 10 documents
```

### Attention

`attend` is Flash Attention folded into the walk: every key the walk scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut.
The index holds one layer's keys for one key-value head, `row_of` names the row of `values` each scored key reads, and with `head_count=8`, query row `t * 8 + h` is head `h` of token `t`.

```python
carry = np.zeros((len(queries), 2), dtype=np.float32)
row_of = {key: row for row, key in enumerate(context_keys)}.get  # a scored Key in, its row of `values` out
outputs = index.on(pool).attend(queries, values, row_of, arena, head_count=8, outputs_dtype=Scalar.BF16, carry=carry)
```

A `row_of` answering `None` raises `ShapeMismatchError`, and it runs under the GIL, while a VRAM arena, whose walk no Python callable can join, raises `DeviceError`.
The `carry` rows hold each query's largest logit and partition, so this branch recombines with another over the same queries, such as attention sinks or a dense recent window.

### Introspection

```python
index.count(42)  # entries sharing this key
42 in index  # the same question, as a membership test
len(index)  # stored entries
index.describe()  # a DenseIndexDescription: .metadata, .size, .capacity, .memory_usage, .cpu_class, .gpu_class, .shards_count
index.health()  # a DenseReachability: .audited, .reachable, .depth_from_root
usearch.capabilities_available()  # the kernel families this machine can run
```

`cpu_class_ceiling=` and `gpu_class_ceiling=` on a constructor, `view_*` or `load_*` name the newest class dispatch may pick, and the description reports the class picked.
Every docstring names the C symbol it calls, or says the method is Python-only.

## Sequence Indexes

`SequenceIndex` searches byte strings, UTF-8 text, DNA, RNA and proteins under edit distances and alignments.
Its verbs and receivers are the dense ones, and a batch is a list of `str` or bytes-like objects, each `str` read as UTF-8.

```python
import numpy as np
from usearch import Arena, SequenceIndex, SequenceMetric
from usearch.keys import keys_to_integers

words = SequenceIndex.init_ram(SequenceMetric.LEVENSHTEIN_UTF8)
words.insert([1, 2, 3], ["kitten", b"sitting", bytearray(b"mitten")])
matches = words.search(["sitting"], 1, Arena())
assert keys_to_integers(np.asarray(matches.keys)[0]).tolist() == [2]
assert np.asarray(matches.distances)[0, 0] == 0
```

`SequenceMetric.NEEDLEMAN_WUNSCH` takes a `substitution_matrix`, any buffer of 256 × 256 signed bytes, and a `gap` cost; `SequenceMetric.OVERLAP` takes `window_widths` in bytes.
A checkpoint records no substitution matrix, so every `load_*` and `view_*` takes it again.
`count` answers the entries under a key, and `get` answers them as an Apache Arrow tape in the arena, offsets beside the bytes end to end.

## Shape Indexes

`ShapesIndex` searches 3D point clouds of any size, like molecules, conformers and protein backbones, with Chamfer, Hausdorff, Gromov-Wasserstein and Wasserstein-Procrustes, all invariant to rotation, reflection, translation and point order.
A batch is a list of `(points, 3)` matrices, which may differ in their point counts.

```python
import numpy as np
from usearch import Arena, ShapesIndex, ShapesMetric
from usearch.keys import keys_to_integers

generator = np.random.default_rng(42)
shapes = [generator.random((points, 3), dtype=np.float32) for points in (5, 9, 13)]
conformers = ShapesIndex.init_ram(ShapesMetric.HAUSDORFF)
conformers.insert([1, 2, 3], shapes)
assert conformers.describe().total_points == 27
matches = conformers.search(shapes[2], 1, Arena())
assert keys_to_integers(np.asarray(matches.keys)[0]).tolist() == [3]
```

A non-zero `points` freezes the count every shape holds, and `quantization` picks `Scalar.F16` or `Scalar.F32` storage.
`get` converts the points to its `outputs_dtype`, `Scalar.F32` by default, and returns each shape in the pose it was inserted in, as point offsets beside a `(points, 3)` matrix.

## Geospatial Indexes

`GeospatialIndex` searches observations on and above the Earth, like photos, video frames and sensor fixes, under nearness and view cones.
A batch is an `(N, 11)` float64 matrix: latitude and longitude in degrees, altitude and accuracy in metres, the heading, pitch, heading accuracy and field of view of the view cone, its range, and the first and last second it covers, NaN wherever nothing was recorded.

```python
import numpy as np
from usearch import Arena, GeospatialIndex
from usearch.keys import keys_to_integers

stops = np.full((3, 11), np.nan)  # Positions alone, every other field unrecorded
stops[:, 0] = [48.8566, 48.8656, 52.52]  # Latitudes
stops[:, 1] = [2.3522, 2.3522, 13.405]  # Longitudes
places = GeospatialIndex.init_ram()
places.insert([1, 2, 3], stops)
keys, _ = places.within(stops[0], Arena(), metres=2000.0).query(0)
assert sorted(keys_to_integers(np.asarray(keys)).tolist()) == [1, 2]  # The two Paris stops
```

`search` ranks the `wanted` nearest entries a `GeospatialRelation` admits, in chord metres, and `within` reports every entry inside the radius as one ragged run per query; `seconds=` widens each query's time interval.
No handle here carries a terrain, so `GeospatialRelation.VISIBLE` raises `UnsupportedError`, and `get` returns a key's observations within the stored rounding.

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

- __Pure CPython over the C ABI.__
  A C verb `usearch_dense_<verb>` is a `<verb>` here, a type drops `usearch_` and `_t`, and enums are `enum.IntEnum` classes, never strings.
- __Deferred calls are awaitables.__
  `await` polls `usearch_outcome_status` between doubling sleeps on the running loop, and `wait` blocks in `usearch_outcome_wait`.
- __Reference counting is the lifetime.__
  Releasing a `Deferred` waits for its call, and a view over an arena lasts until the next call rewinds it.
- __Language idioms over parallel APIs.__
  The buffer protocol, DLPack, `in`, struct sequences, context managers and `+=` take the place of helper methods.
