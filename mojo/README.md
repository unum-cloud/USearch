# USearch for Mojo

The Mojo package binds the C ABI in `c/usearch/`, opening the engine's shared library with `dlopen` rather than linking it, so a consumer builds with no linker flags of its own.
It needs Mojo 1.1 or a later 1.x release through pixi, on Linux x86-64, Linux AArch64 or macOS on Apple silicon.
It exposes dense indexes, Vector Search over equidimensional embeddings; sequence, matrix and geospatial indexes follow as the C library ships them.

## Installation

Name the package by `path` or `git` in a Pixi manifest; the engine arrives as its run dependency, in the environment prefix `mojo build` bakes as runpath.

```toml
[dependencies]
usearch_mojo = { path = "path/to/USearch/mojo" }
```

From a checkout, `pixi run test` builds the engine through the tree's own CMake and runs `mojo/test.mojo` against it.
On Linux the environment pins a glibc 2.34 sysroot, which the MAX runtime `mojo build` links against needs.

## Quickstart

```mojo
from usearch import Arena, DenseIndex, DenseMetric, DenseOptions, Executor, Key, Library, RowsView, ScalarKind

def main() raises:
    var library = Library()
    var options = DenseOptions(3, DenseMetric.ANGULAR, ScalarKind.BF16)  # the defaults, spelled out
    var index = DenseIndex.init_ram(library, options)  # one shard on every NUMA node
    index.reserve(1_000)

    var keys = [Key(integer=42), Key(integer=43)]
    var vectors = [Float32(0.2), 0.6, 0.4, Float32(0.1), 0.3, 0.9]
    index.insert(keys, RowsView.of(vectors, 3))  # on the calling thread

    var executor = Executor.for_ram(library, threads=4)
    var arena = Arena(executor)
    var query = [Float32(0.2), 0.6, 0.4]
    var matches = index.on(executor).search(RowsView.of(query, 3), 10, arena)  # the same verb, on an executor
    print(matches.row_keys(0)[0], matches.row_distances(0)[0])  # 00000000-0000-0000-0000-00000000002a 0.0
```

An index owns rows; an `Executor` owns the threads or streams a call runs on; an `Arena` owns the outputs a call writes.
`Library()` opens the engine once, and every handle created through it keeps it mapped.

## Receivers & Deferred Calls

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
The index runs it on the calling thread, `index.on(executor)` blocks on an executor, and `index.deferred(executor)` submits it there and answers a `Deferred`:

```mojo
index.search(queries, 10, arena)                               # the calling thread
index.on(executor).search(queries, 10, arena)                  # blocking, after every earlier submission
var call = index.deferred(executor).search(queries, 10, arena)  # Deferred[...], final once waited
```

`DenseIndexOn` and `DenseIndexDeferredOn` offer the same verbs with the same arguments.
The blocking verbs are written once, in the `DenseVerbs` trait both the index and `DenseIndexOn` conform to.
Constructors address no index yet, so no receiver offers them; the ones whose C structure names an executor take one.

A `Deferred` owns its outcome and outputs until the call is final, and borrows the index, the executor, the arena and every input it was handed, which Mojo keeps alive for as long as the `Deferred` is.

```mojo
var call = index.deferred(executor).search(queries, 10, arena)
if call.status() == Status.DEFERRED:
    pass  # other work
print(call.wait().row_keys(0)[0])  # the matches, read in place from your arena
```

Mojo has no runtime of its own to await on, so `wait` is the whole interface: it blocks in `usearch_outcome_wait` until the call is final, then answers the result or raises the failure, as many times as it is asked.
The result is the very `Matches` a blocking search answers, borrowing the same arena, and `status` reads the outcome without blocking.
A failure this binding catches before the call crosses is final at once, and arrives through the same `status` and `wait`.
Mojo destroys a value at its last use, and destroying an unfinished `Deferred` waits for it.
Calls depending on each other are ordered by waiting on the first before submitting the second.

## Executors

One executor serves any number of indexes, and its residency must match the one their handles were built for.
`Executor(library)` owns nothing and runs every call handed it on the calling thread, whatever the handle's residency.
There is a constructor per C one, each taking every argument its own hardware has and ending with the `deferred_limit` sizing its table of deferred calls, where zero takes a default:

```mojo
var threads = Executor.for_ram(library, RamOptions.on(domains), threads=8)  # zero takes 7/8 of the cores
var shared = Executor.on_pool(library, fork_union_pool)                       # borrows a fu_pool_t
var streams = Executor.for_vram(library, VramOptions.on(devices))             # one host thread, a stream per device
var rings = Executor.for_nvme(library, threads=4)                             # one submission ring per thread
threads.drain(Mood.SPINNING)                                                # waits for every call, then keeps spinning
```

Calls on one executor start in submission order, while index handles stay freely concurrent, so threads that genuinely overlap take one executor apiece.
A submission finding the executor's deferred table full finishes at once with `Status.CONTENDED`.
An executor without threads of its own holds a deferred call until its `wait`, the executor's next `drain` or blocking call, or the free of a handle it names.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`Arena(executor)` places it where that executor computes, decided once, and `Arena(library)` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
A view's origin keeps its arena alive, and reading a stale one raises `Status.UNINITIALIZED`.
An arena from a VRAM executor holds device memory, which host accessors refuse with `Status.DEVICE`, and serves only that executor's calls, keeping the executor alive until the arena goes.

```mojo
var arena = Arena(executor)  # Arena(library) makes host pages for the calling thread
for position in range(len(batches)):
    var matches = index.on(executor).search(batches[position], 10, arena)  # rewinds: the last batch's matches are gone
    total += matches.row_distances(0)[0]

var found = Arena(executor)
var rows = Arena(executor)  # two live results, two arenas
var hits = index.on(executor).search(query, 10, found)
for key in hits.row_keys(0):
    rerank(key, index.on(executor).get[DType.float32](key, rows))

var east_arena = Arena(executor)
var west_arena = Arena(executor)  # one arena per submitted call
var east = east_shard.deferred(executor).search(queries, 10, east_arena)
var west = west_shard.deferred(executor).search(queries, 10, west_arena)
print(east.wait().row_keys(0)[0], west.wait().row_keys(0)[0])
```

The arena is the verb's last required argument, before the optional config, on every receiver.
A call that finds its arena serving another call finishes with `Status.CONTENDED` and rewinds nothing, so the holder's views stay readable.
A call that fails for any other reason still rewinds it.

## Keys

A key is 128 bits of opaque identity, rendered as a lowercase dashed UUID.
Keys need not be unique: several entries may share one, and `count` reports how many.

```mojo
var key = Key(integer=42)
print(key)                     # 00000000-0000-0000-0000-00000000002a
print(key.as_integer().value()) # 42
```

## Errors

Every fallible entry point raises `Error` and nothing else, since Mojo allows one error type per function.
An `Error` carries a `Status` shared with the C interface, so a fault this binding catches before the boundary reports the status the engine would have reported for the same fault:

```mojo
try:
    index.insert(keys, vectors)
except error:
    if error.status == Status.SHAPE_MISMATCH:
        pass  # the keys and the rows disagree
    print(error)  # USearch: shape-mismatch [usearch_dense_insert]
```

An `Error` keeps the engine mapped while it lives, since its name and message are the engine's own static text.
`library.status_name` and `status_from_name` spell and parse a `Status` through the engine's own name table, as their twins do for scalars, residencies, traversals, moods and dense metrics:

```mojo
print(library.status_name(Status.READ_ONLY), library.scalar_name(ScalarKind.BF16))  # read-only bf16
```

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has, one shard per entry, and leaving it out derives one.

```mojo
var domains = [Int32(0), Int32(1)]
var in_ram = DenseIndex.init_ram(library, options, RamOptions.on(domains))  # one graph per socket
var devices = [Int32(0), Int32(1), Int32(2), Int32(3)]
var in_vram = DenseIndex.init_vram(library, options, VramOptions.on(devices))  # sharded, never replicated
var on_nvme = DenseIndex.view_nvme(Executor.for_nvme(library), "index.safegraphs")
```

An NVMe index is only ever viewed: external-memory graphs are never modified in place.
Until NVMe checkpoints carry keys, `view_nvme` answers `Status.UNSUPPORTED`.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout.
A `.safegraphs` path is one self-describing monolith; a `.safegraphs.json` path is a manifest cataloging many, with every shard path resolved relative to it.
In-memory buffers hold single monoliths only.

```mojo
index.on(executor).save("index.safegraphs")
var restored = DenseIndex.load_ram(executor, "index.safegraphs")
var viewed = DenseIndex.view_ram(library, "index.safegraphs")  # read-only, served from the mapping
var metadata = dense_metadata_read(executor, "index.safegraphs")  # geometry without loading a shard

var monolith = index.save_buffer(arena)
var copied = DenseIndex.load_ram_buffer(executor, monolith.bytes())
```

Loading creates the index, freezing its geometry from the checkpoint's own headers.
The metadata carries only the four fields a checkpoint froze, so `values_per_key` and the placement stay yours to pass to the load.

## Dense Indexes

`DenseOptions` freezes an index's geometry: its dimensions, metric, quantization, connectivity and `values_per_key`, angular over `BF16` rows unless told otherwise, with every zeroed optional field derived.
`DenseMetric` names `INNER`, `ANGULAR`, `EUCLIDEAN`, `HAMMING` and `JACCARD`; the last two read rows as `ScalarKind.U1X8` bit sets and refuse any other quantization.
Insertions and searches overlap freely on one handle: a search never writes to the graph, and an insertion publishes a row only once it is reachable.
`multiply`, `maxsim` and `attend` treat the index as a sparse operand, folding their arithmetic into the walk instead of materializing a match list, and need logits rather than distances, so they run on `DenseMetric.INNER` alone and refuse other metrics with `Status.CONFIG_MISMATCH`.

### Rows & Scalars

Every verb is a batch over strided rows, so a memory-mapped dataset reaches the index with no reshape and no copy, and a single vector is a one-row view.
A `RowsView` borrows a caller-owned run as rows of equal width, and it is the shape every batch argument takes.
`RowsView.of` derives the row count and the byte pitch from a packed run, and the scalar kind from its element type.

```mojo
var vectors = RowsView.of(embeddings, 768)  # Float32 elements, so ScalarKind.F32
```

The library converts on ingest and on export, so the scalar kind of the input is independent of the index's own quantization, and a `Float32` query searches an `I8` index without a conversion pass of your own.
Sub-byte kinds count logical dimensions rather than bytes and name their kind explicitly, so 128 dimensions of `U1X8` occupy sixteen bytes per row:

```mojo
var packed = List[UInt8](unsafe_uninit_length=rows * 16)  # filled before the call
var bits = RowsView.of(packed, 128, ScalarKind.U1X8)
```

A dataset whose rows are padded to a wider pitch builds the view field by field, with its own `stride` in bytes.

### Search

Every verb with optional knobs takes a config named after its C argument structure, such as `DenseInsertConfig`, `DenseSearchConfig` or `DenseMultiplyConfig`, where every field left out derives what the C interface derives.
So the bare call takes none, and naming a field reaches a knob:

```mojo
var stats = DenseTraversalStats()
_ = index.on(executor).search(queries, 10, arena, DenseSearchConfig(expansion_factor=128, stats=Pointer(to=stats)))
print(stats.computed_distances, "distances computed")
```

A `stats` slot is overwritten whole by every call it is lent to, so a reused one reads the last call's figures rather than a running total.

A search answers `Matches`: a queries-by-`wanted` grid of keys, the distances paired with them, and one match count per query, all written into the arena.
`row_keys` and `row_distances` borrow one query's results, nearest first, trimmed to the count that query reported:

```mojo
var matches = index.search(queries, 10, arena)
for query in range(len(matches)):
    for rank in range(len(matches.row_keys(query))):
        print(query, matches.row_keys(query)[rank], matches.row_distances(query)[rank])
```

The candidate-pool breadth rides each call as `expansion_factor` rather than the handle, so a recall sweep runs against one immutable index with no mutation between points.
`DenseSearchConfig(traversal=Traversal.SCAN)` scores every row instead of walking, on RAM and NVMe handles, which gives that sweep its exact ground truth.

### Filtering & Progress

A `predicate` in the `DenseSearchConfig` admits only the keys it approves, while non-matching entries still route the traversal, so recall does not collapse at low selectivity.
It is the address of a `usearch_predicate_callback_t` compiled in C, judging a batch of keys per call, and its state crosses beside it as an address.

```mojo
var admitted = index.search(queries, 10, arena, DenseSearchConfig(predicate=admits_even, predicate_state=tenant))
```

The predicate runs on the executor's threads together, so it must be safe for concurrent use.
This binding installs no progress handler, so an `insert` batch is all-or-nothing: every fallible step runs before the first entry is claimed, and a rejected batch publishes no row.

### Retrieval: Get, Cursors & Streams

`get` exports every entry stored under one key, one row per entry in insertion order, as `DenseVectors`, whose every row is exactly `columns()` values wide, as the engine reported them, however far apart the engine laid the rows.

```mojo
var stored = index.get[DType.float32](Key(integer=42), arena)  # empty when the key is absent
```

A cursor walks the contents as a sequential row scan, landing each batch in the arena, its rows converted to the kind you name; `next_keys` walks the key column alone.

```mojo
var cursor = index.on(executor).cursor()
while True:
    var batch = cursor.next_batch(1024, ScalarKind.F32, arena)  # empty once exhausted
    if len(batch) == 0:
        break
    print(batch.keys()[0], batch.row[DType.float32](0)[0])
```

A search stream is a resumable best-first traversal, so callers that post-filter aggressively pull until satisfied instead of guessing "k" upfront.
No residency of the engine implements the stream yet, so it answers `Status.UNSUPPORTED`.

```mojo
var stream = index.on(executor).search_stream(RowsView.of(query, 768))
while True:
    var pulled = stream.next(16, arena)  # one row of Matches, empty once drained
    if len(pulled.row_keys(0)) == 0:
        break
```

Cursors and streams open on the blocking receivers alone, the index and `index.on(executor)`, and every pull blocks.
Both borrow the index and the executor, so neither can be released while a cursor or a stream lives, and a drained one answers an empty view rather than raising.

### Refinement

`refine` re-runs the neighbor selection over entries already in the graph, returning how many nodes had their neighbors changed.
A batch built at a low expansion factor reaches the quality of a slower build afterwards, and the count reaching zero is what settled looks like:

```mojo
while index.on(executor).refine(DenseRefineConfig(expansion_factor=128)) != 0:
    pass
```

### Absorbing & Splitting

Bigger indexes come from absorbing, never mutating: `absorb` takes its donors by value, appends each one's rows with its graph, then refines across the appended part, and the engine frees each donor it absorbs.
Splitting is not in the C library yet.

```mojo
var donors = List[DenseIndex]()
donors.append(shard^)
index.on(executor).absorb(donors^)
```

### Sparse Multiplication

`multiply` is a sparse matrix product: every pair the walk scored for each left row, in traversal order, lands in an arena as one ragged run, with one more offset than left rows, as Apache Arrow lays a ragged batch out.

```mojo
var products = index.on(executor).multiply(left, arena, DenseMultiplyConfig(wanted=64))
var offsets = products.offsets()
for row in range(left_rows):
    print(row, Int(offsets[row + 1] - offsets[row]), "pairs")
```

### Late Interaction

`maxsim` is late interaction: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `values_per_key` above one exists for, and the wanted count is over documents rather than rows.

```mojo
var documents = index.on(executor).maxsim(tokens, tokens_per_query, 10, arena)  # Matches, as search answers
```

### Attention

`attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the attended rows land in the arena already divided by the partition.
The index stores no values, so a `ValueLookup` resolves every scored batch of keys to value rows, `columns` values of `scalar` apiece, the index's own width and kind, which it checks before answering any row.
A key without a row refuses the call with `Status.SHAPE_MISMATCH`, since a missing row is a wrong softmax rather than a smaller one.
A VRAM index calls its lookup inside the device walk, so there `value_lookup` must be a `__device__` function's address, read with `cudaMemcpyFromSymbol` on NVIDIA or `hipMemcpyFromSymbol` on AMD; a host function there is undefined, never refused.

```mojo
def value_rows(
    state: Address, keys: Pointer[Key, ImmUntrackedOrigin], count: c_size_t,
    scalar: ScalarKind, columns: c_size_t, rows: Pointer[Address, MutAnyOrigin],
) abi("C"):
    ...  # write one row address per key, or NULL_ADDRESS when it has none

var carry = List[DenseSoftmaxState](length=queries_count, fill=DenseSoftmaxState())
var context = index.on(executor).attend[DType.float32](
    queries, value_rows, cache_address, arena, DenseAttendConfig(head_count=8, carry=carry)
)
```

With `head_count=8`, row `t * 8 + h` of the queries is head `h` of token `t`, and the heads of one token share its walk.
A `carry` of one `DenseSoftmaxState` per query row, lent by you, folds this branch into another over the same queries, such as a recent window kept out of the index; a shorter one is refused, since the C structure carries no count for it.

### Introspection

```mojo
var description = index.describe()
print(description.size, description.memory_usage, description.capabilities)  # the kernels dispatch picked
print(index.count(Key(integer=42)))  # entries sharing one key

var reachability = index.health()
print(reachability.reachable, "of", reachability.audited, "entries reachable")
print(library.version_name(), library.capabilities_available())
```

`health` audits how much of the graph a search reaches from its landmarks; entries it misses stay unfound until a `refine` re-links them.

## What This Binding Does Not Expose

- __Progress handlers and cancellation.__
  A handler returns a verdict, and a Mojo 1.1 `def ... abi("C")` that returns a value does not lower to the C calling convention, so every `progress` slot crosses as `NULL_ADDRESS`.
- __Predicates written in Mojo.__
  `predicate` takes the address of a `usearch_predicate_callback_t` compiled in C.
- __Deferred cursors and streams.__
  `DenseIndexDeferredOn` offers neither, since their pulls block.

## Design

- __A thin layer over the C ABI.__
  Every name follows `c/usearch/`: a C verb `usearch_dense_<verb>` is a `<verb>` here, and enums mirror C enums, never strings.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `usearch_outcome_status`, `wait` blocks through `usearch_outcome_wait`, and the origins a `Deferred` carries keep everything it borrows alive until it is gone.
- __No life of its own.__
  The binding starts no threads and keeps no queues; releasing a `Deferred` rides Mojo's destruction at last use, which waits for the call first.
- __Arenas are lent and reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes; the binding stamps each arena with a counter that every view checks before reading, and keeps no call counts.
- __Nothing costs anything until it is used.__
  Blocking calls take no deferred-table slot, and an arena allocates only when a call writes into it.
- __Blocking calls live on the stack.__
  A blocking call keeps its outcome and outputs in stack slots and allocates nothing of its own; a deferred one moves those slots to one heap allocation it frees once final.
- __`ScalarKind`, not `Scalar`.__
  Types drop `usearch_` and `_t` from their C names, except `ScalarKind`, which keeps clear of Mojo's built-in `Scalar`.
