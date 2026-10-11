# USearch for Mojo

The Mojo SDK reaches the engine through its stable C headers, opened with `dlopen` rather than linked, so a consumer builds with no linker flags of its own.
It exposes the dense modality, Vector Search over equidimensional embeddings, and the three linear-algebra verbs that read products, attention and late-interaction scores off the same walk.

Every verb is a batch over strided rows, so a memory-mapped dataset reaches the index with no reshape and no copy, and a single vector is a one-row view.

## Installation

Name the package by `path` or `git` in a Pixi manifest; the engine arrives as its run dependency, in the environment prefix `mojo build` bakes as runpath.

```toml
[dependencies]
usearch_mojo = { path = "path/to/USearch/mojo" }
```

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

## Receivers

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
The index runs it on the calling thread, `index.on(executor)` blocks on an executor, and `index.deferred(executor)` submits it there and answers a `Deferred`:

```mojo
index.search(queries, 10, arena)                               # the calling thread
index.on(executor).search(queries, 10, arena)                  # blocking, after every earlier submission
var call = index.deferred(executor).search(queries, 10, arena)  # Deferred[...], final once waited
```

`DenseIndexOn` and `DenseIndexDeferredOn` offer the same verbs with the same arguments; only the blocking receivers add `cursor` and `search_stream`, whose every pull blocks.
The blocking verbs are written once, in the `DenseVerbs` trait both the index and `DenseIndexOn` conform to.
Constructors address no index yet, so no receiver offers them; the ones whose C structure names an executor take one.

## Deferred Calls

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

An executor without threads of its own holds a deferred call until its `wait`, the executor's next `drain` or blocking call, or the free of a handle it names.
Mojo destroys a value at its last use, and destroying an unfinished `Deferred` waits for it.
A submission finding the executor's deferred table full, sized by `deferred_limit`, finishes at once with `Status.CONTENDED`.
Calls depending on each other are ordered by waiting on the first before submitting the second.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`Arena(executor)` places it where that executor computes, decided once, and `Arena(library)` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
A view's origin keeps its arena alive, and reading a stale one raises `Status.UNINITIALIZED`.
An arena from a VRAM executor holds device memory, which host accessors refuse with `Status.DEVICE`.

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
var west_arena = Arena(executor)  # one arena per call in flight
var east = east_shard.deferred(executor).search(queries, 10, east_arena)
var west = west_shard.deferred(executor).search(queries, 10, west_arena)
print(east.wait().row_keys(0)[0], west.wait().row_keys(0)[0])
```

The arena is the verb's last required argument, before the optional config, on every receiver.
A call that finds its arena busy finishes with `Status.CONTENDED` and rewinds nothing, so the holder's views stay readable.
A call that fails for any other reason still rewinds it.
A drained cursor or stream answers an empty view rather than raising.

## Rows

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

## Configs and Outputs

Every verb with optional knobs takes a config named after its C argument structure, such as `DenseInsertConfig`, `DenseSearchConfig` or `DenseMultiplyConfig`, where every field left out derives what the C interface derives.
So the bare call takes none, and naming a field reaches a knob:

```mojo
var stats = DenseTraversalStats()
_ = index.on(executor).search(queries, 10, arena, DenseSearchConfig(expansion_factor=128, stats=Pointer(to=stats)))
print(stats.computed_distances, "distances computed")
```

A search answers `Matches`: a queries-by-`wanted` grid of keys, the distances paired with them, and one match count per query, all written into the arena.
`row_keys` and `row_distances` borrow one query's results, trimmed to the count that query reported:

```mojo
var matches = index.search(queries, 10, arena)
for query in range(len(matches)):
    for rank in range(len(matches.row_keys(query))):
        print(query, matches.row_keys(query)[rank], matches.row_distances(query)[rank])
```

A `get` or `attend` answers `DenseVectors`, whose every row is exactly `columns()` values wide, as the engine reported them, however far apart the engine laid the rows.

The candidate-pool breadth rides each call as `expansion_factor` rather than the handle, so a recall sweep runs against one immutable index with no mutation between points.
`DenseSearchConfig(traversal=Traversal.SCAN)` scores every row instead of walking, on RAM and NVMe handles, which gives that sweep its exact ground truth.

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
`Executor(library)` owns nothing and runs a load on the calling thread.

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

## Executors

One executor serves any number of indexes, and its residency must match the one their handles were built for.
There is a constructor per C one, each taking every argument its own hardware has and ending with the `deferred_limit` sizing its table of deferred calls, where zero takes a default:

```mojo
var threads = Executor.for_ram(library, RamOptions.on(domains), threads=8)  # zero takes 7/8 of the cores
var shared = Executor.on_pool(library, fork_union_pool)                       # borrows a fu_pool_t
var streams = Executor.for_vram(library, VramOptions.on(devices))             # one host thread, a stream per device
var rings = Executor.for_nvme(library, threads=4)                             # one submission ring per thread
threads.drain(Mood.SPINNING)                                                  # awaits every call, then keeps spinning
```

Calls on one executor start in submission order, while index handles stay freely concurrent, so threads that genuinely overlap take one executor apiece.

## Introspection

```mojo
var description = index.describe()
print(description.size, description.memory_usage, description.capabilities)  # the kernels dispatch picked
print(index.count(Key(integer=42)))  # entries sharing one key

var reachability = index.health()
print(reachability.reachable, "of", reachability.audited, "entries reachable")

print(library.version_name(), library.capabilities_available())
print(library.status_name(Status.READ_ONLY), library.scalar_name(ScalarKind.BF16))  # read-only bf16
```

`health` audits how much of the graph a search reaches from its landmarks; entries it misses stay unfound until a `refine` re-links them.
Every enum spells and parses through the engine's own name tables: `status_name` and `status_from_name`, and their twins for scalars, residencies, moods and dense metrics.

## Filtering

A `predicate` in the `DenseSearchConfig` admits only the keys it approves, while non-matching entries still route the traversal, so recall does not collapse at low selectivity.
It is the address of a `usearch_predicate_t` compiled in C, since a Mojo function returning a value does not lower to the C calling convention in Mojo 1.1; its state crosses beside it as an address.

```mojo
var admitted = index.search(queries, 10, arena, DenseSearchConfig(predicate=admits_even, predicate_state=tenant))
```

The predicate runs on the executor's threads together, so it must be safe for concurrent use.

## Linear Algebra Over The Graph

Three verbs treat the index as a sparse operand rather than as a kNN store, and each folds its arithmetic into the graph walk instead of materializing a match list.
All three need logits rather than distances, so they run on an inner-product index alone.

`multiply` is a sparse matrix product: every pair the walk scored for each left row, in traversal order, lands in an arena as one ragged run, with one more offset than left rows, as Apache Arrow lays a ragged batch out.

```mojo
var products = index.on(executor).multiply(left, arena, DenseMultiplyConfig(wanted=64))
var offsets = products.offsets()
for row in range(left_rows):
    print(row, Int(offsets[row + 1] - offsets[row]), "pairs")
```

`maxsim` is late interaction: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `values_per_key` above one exists for, and the wanted count is over documents rather than rows.

```mojo
var documents = index.on(executor).maxsim(tokens, tokens_per_query, 10, arena)  # Matches, as search answers
```

`attend` is Flash Attention folded into the walk: every key it scores folds into each head's running maximum, partition and weighted value sum, with no top-k cut, and the attended rows land in the arena already divided by the partition.
The index stores no values, so a `ValueLookup` resolves every scored batch of keys to value rows, `columns` values of `scalar` apiece, the index's own width and kind, which it checks before answering any row.
A key without a row refuses the call with `Status.SHAPE_MISMATCH`, since a missing row is a wrong softmax rather than a smaller one.

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

## Streaming and Iteration

A search stream is a resumable best-first traversal, so callers that post-filter aggressively pull until satisfied instead of guessing "k" upfront.
No residency of the engine implements the stream yet, so it answers `Status.UNSUPPORTED`.

```mojo
var stream = index.on(executor).search_stream(RowsView.of(query, 768))
while True:
    var pulled = stream.next(16, arena)  # one row of Matches, empty once drained
    if len(pulled.row_keys(0)) == 0:
        break
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

Both borrow the index and the executor, so neither can be released while a cursor or a stream lives.

## Refinement and Merging

`refine` re-runs the neighbor selection over entries already in the graph, returning how many nodes had their neighbors changed.
A batch built at a low expansion factor reaches the quality of a slower build afterwards, and the count reaching zero is what settled looks like:

```mojo
while index.on(executor).refine(DenseRefineConfig(expansion_factor=128)) != 0:
    pass
```

Bigger indexes come from merging, never mutating: `merge` takes its donors by value, and the engine frees each one it absorbs.
No residency of the engine absorbs donors yet, so it answers `Status.UNSUPPORTED`.

```mojo
var donors = List[DenseIndex]()
donors.append(shard^)
index.on(executor).merge(donors^)
```

## Design

The package is a thin layer over the C ABI in `include/usearch/`, and every rule below follows from that.

- __Names follow the header.__
  Every type is its C name without `usearch_` and `_t`, such as `DenseMetric`, `DenseTraversalStats` and `DenseIndexDescription`; every constructor is a C lifecycle verb such as `init_ram` or `for_ram`, a verb's argument structure becomes its `Dense*Config`, and every wrapper's documentation names the C symbol it calls or says it is binding-only.
  The one exception is `ScalarKind`, which keeps clear of Mojo's built-in `Scalar`.
- __Blocking calls stay simple.__
  `index.search(queries, 10, arena)` takes no executor, completion or receiver argument and writes its outputs before returning, so nothing about asynchrony reaches a caller who never asks for it.
- __Outputs land in arenas.__
  Every output whose length scales with the call lands in the `Arena` the verb requires, one verb per operation; only fixed-size reports like stats and descriptions, and state you carry like attention's `carry`, stay yours.
- __Deferred calls are symmetric, through Mojo's own idioms.__
  The receiver's type alone chooses the completion, so the index, `DenseIndexOn` and `DenseIndexDeferredOn` offer the same verbs with the same arguments, each one a single crossing all three route.
  A `Deferred` carries the origins of everything it borrows, so the compiler keeps the index, the executor and the inputs alive until it is gone.
- __Nothing costs anything until it is used.__
  A blocking call keeps its outcome and outputs in stack slots and allocates nothing of its own; a deferred one moves those slots to one heap allocation it frees once final.
- __A `Deferred` has no life of its own.__
  `status` polls through `usearch_outcome_status`, `wait` blocks in `usearch_outcome_wait`, and destroying an unfinished one waits the same way; the binding starts no thread, counts no calls and schedules nothing.
- __Statuses and modes are enums.__
  `Status`, `Mood`, `Residency`, `ScalarKind` and `DenseMetric` carry the C enumerators' values, and the `Library` spells and parses them through the C name tables, so nothing is compared as a string.
- __Arenas are reused, never shared.__
  Every call borrows its arena and reuses it, every result refuses to read an arena a later call rewound, and no index owns an arena or a lock.
  Contention is the engine's to refuse, so the binding claims nothing and only counts the calls the engine took.
