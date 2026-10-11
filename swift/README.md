# USearch for Swift

## Installation

USearch is available through the Swift Package Manager.

```swift
dependencies: [
    .package(url: "https://github.com/unum-cloud/USearch", .upToNextMajor(from: "3.0.0"))
]
```

The package compiles the Swift surface only, and depends on NumKong for the matrix views and the low-precision scalars.
The engine itself is `libusearch_static`, built by CMake and linked by the package.
Swift 6.4 or newer is required, on macOS 15, iOS 18, tvOS 18, watchOS 11, visionOS 2, or Linux.

## Quickstart

Every verb is a batch over a `RowsView`: a base pointer, a row count, the values per row, the byte pitch between consecutive rows, and the scalar kind they hold.
A single vector is a one-row matrix, so nothing is reshaped and nothing is copied.

```swift
import NumKong
import USearch

let index = try DenseIndex.initRam(DenseOptions(dimensions: 3, metric: .angular, quantization: .bf16))
try index.reserve(2)

let vectors = try Tensor.fromArray([Float](arrayLiteral: 0.3, 0.5, 1.2, 0.4, 0.2, 1.2), rows: 2, columns: 3)
try index.insert(keys: [42, 43], vectors: RowsView(vectors))                // the calling thread

let query = RowsView(try Tensor.fromArray([Float](arrayLiteral: 0.3, 0.5, 1.2), rows: 1, columns: 3))
let executor = try Executor.forRam()
let arena = try Arena(on: executor)                                        // where the matches land
let matches = try index.on(executor).search(query, wanted: 10, arena: arena) // the same verb, on an executor
assert(matches[0][0].key == Key(42))
```

Every verb whose output grows with the call writes it into an `Arena` and answers a view reading it in place, so a steady loop allocates once.
There is one verb per operation: no allocating twin, no caller buffers, and no default arena.

The lifecycle verbs name where the payload lives, as the C ones do: `initRam` and `initVram` start empty, `loadRam`, `loadVram` and their `Buffer` forms copy a checkpoint in, and `viewRam` and `viewNvme` serve one in place.
The library checks every row view's width against the index geometry itself, so a mismatch surfaces as its own `shapeMismatch`.

## Receivers and Deferred Calls

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
Called on the index, a verb runs on the calling thread; on `index.on(executor)` it blocks on that executor, after every call submitted there before it; on `index.deferred(executor)` it submits there and answers a `Deferred` at once.

```swift
let here = try index.search(queries, wanted: 10, arena: arena)                  // the calling thread
let blocking = try index.on(executor).search(queries, wanted: 10, arena: arena) // blocking, on the executor
let call = try index.deferred(executor).search(queries, wanted: 10, arena: arena) // Deferred<Matches>
let matches = try await call.value  // or: try call.wait()
```

A `Deferred` owns its outcome and its outputs, and keeps alive every object whose memory the call reads, until the call is final.
It borrows the index, the executor and the arena as a blocking call does, since releasing any of them waits for the calls naming it, and its waited result is the same view a blocking call answers.
`status` reads `deferred` until then, and `wait()` blocks and answers the result or throws the failure, the same every time it is asked.
`value` waits on a Dispatch thread and then reads the result in the caller's isolation, so an `async` caller suspends rather than blocks, and the cooperative pool keeps its threads.
Releasing an unfinished one waits for it first.

```swift
let call = try index.deferred(executor).search(queries, wanted: 10, arena: arena)
if call.status == .deferred {
    // other work
}
let matches = try await call.value  // or: try call.wait()
```

A failure this binding catches before the call crosses throws at submission, and one the library catches arrives through `status` and `wait`.
A submission finding the executor's deferred table full, sized by `deferredLimit`, finishes at once with `contended`.
Calls depending on each other are ordered by waiting on the first before submitting the second.
Rows built over a `Tensor` stay alive with the call, while memory borrowed through a `MatrixView` or a softmax carry must outlive it.
`cursor` and `searchStream` pull blocking, so a `DenseIndexDeferredOn` lacks them, and the constructors address no index yet, so no receiver offers them; the ones the C interface lets fan out take an `executor` in their config and run on the calling thread without one.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
`Arena(on: executor)` places it where that executor computes, decided once, and `Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
Views stay on the arena's thread, which the compiler checks, and reading a stale one traps as an out-of-bounds index does.
An arena from a VRAM executor holds device memory, which host accessors refuse with a trap.

```swift
let arena = try Arena(on: executor)                                              // Arena() is host pages for the calling thread
for batch in batches {
    let matches = try index.on(executor).search(batch, wanted: 10, arena: arena) // rewinds: the last batch's matches are gone
    total += matches.joined().reduce(0) { $0 + $1.distance }
}
let found = try Arena(), rows = try Arena()                                      // two live results, two arenas
for match in try index.search(query, wanted: 10, arena: found)[0] {
    rerank(match.key, try index.get(key: match.key, as: Float.self, arena: rows))
}
let calls = try zip(shards, arenas).map { shard, arena in
    try shard.deferred(executor).search(queries, wanted: 10, arena: arena)       // one arena per call in flight
}
var results: [Matches] = []
for call in calls { results.append(try await call.value) }                       // or: try call.wait()
```

A result keeps the arena it reads alive, so it outlives the index and the caller's reference alike.

## Rows

A `RowsView` borrows a NumKong `MatrixView`, or keeps a NumKong `Tensor` alive.
A matrix view borrows and never owns, so it stays valid only for as long as the storage it points into, which is why a call over one sits inside a `withUnsafeBufferPointer`.

```swift
let padded: [Float] = load()                                               // 768 dimensions pitched at 1024
try padded.withUnsafeBufferPointer { buffer in
    let rows = MatrixView(baseAddress: buffer.baseAddress!, rows: 1_000, columns: 768, rowStrideBytes: 1024 * 4)
    try index.on(executor).insert(keys: keys, vectors: RowsView(rows))
}
```

Sub-byte kinds count logical dimensions rather than stored values, so a `U1x8` view one value wide carries eight dimensions.

## Results

A search writes up to `wanted` matches per query into its arena, and answers `Matches` reading them in place: `matches[query]` is one query's row, so neither level copies a row out.
A row's own `count` stops at the matches the call wrote, however many were wanted.

```swift
for query in 0 ..< matches.count {
    for match in matches[query] { print(match.key, match.distance) }
}
```

Both levels are `RandomAccessCollection`, so a row walks, maps and slices like any other sequence.
A sweep passes the same arena to every call and never allocates again once it reaches its high-water mark.

```swift
let arena = try Arena(on: executor)
for batch in batches {
    let matches = try index.on(executor).search(batch, wanted: 10, arena: arena)
    consume(matches) // read before the next call given `arena`
}
```

A view cannot be `~Escapable` while its lifetime annotation is still behind an experimental feature flag as of Swift 6.4, so it checks its arena at runtime instead.

## Errors

`USearchError` is the only error the SDK throws, and every method says so with `throws(USearchError)`, so a `catch` reads `error.status` with no cast.
`Status` has one constant per `usearch_status_*_k` of the C interface, and its `description` spells it as the library does, like `invalid-argument`.

```swift
do {
    try index.reserve(1 << 40)
} catch {
    print(error.status, error.message)
}
```

NumKong exports a `Status` of its own, so a file importing both spells this one `USearch.Status`.

## Executors

An index says where its payload lives; an `Executor` says who walks it.
One executor serves any number of handles, so several categories of index share one machine's cores or devices instead of each claiming a private set.
Its kind is fixed by the factory that spawned it and must match every handle it drives; a mismatch reports `device`.

```swift
let hostThreads = try Executor.forRam(memoryDomains: [0, 1], threads: 0, deferredLimit: 0)
let submissionRings = try Executor.forNvme(paths: ["/mnt/nvme0"], threads: 8)
let deviceStreams = try Executor.forVram(devices: [0, 1], streams: nil, scratchLimit: 1 << 30)
```

Each factory takes every argument its C constructor has, and only those.
`deferredLimit` sizes the table of deferred calls an executor tracks, and `forVram` can borrow your CUDA streams, one per device, or cap per-device scratch.
A thread count of zero derives from the hardware: 7/8 of the cores those memory domains carry, since full occupancy starves the ordered publication path, and for NVMe the devices' queue capabilities.
`forVram` takes no thread count at all: one host thread drives one stream per device, so there is no count to name.

Calls on one executor run in submission order, and index handles stay freely shareable.
`Executor.onPool(memoryDomains:pool:deferredLimit:)` borrows an existing ForkUnion pool instead of spawning threads, and `drain(mood:)` waits for the executor's calls and picks whether its threads park or spin once idle.

## Keys

A key is sixteen opaque bytes, rendered as a lowercase dashed UUID.
Integer identities park in the low eight bytes, so `Key(42)` prints as `00000000-0000-0000-0000-00000000002a` and an integer literal is a key wherever one is expected.
Keys need not be unique: several entries may share one, and `count(key:)` reports how many.

A key is `Codable` as that same string, so a document written by any SDK reads back here unchanged.

```swift
let key: Key = 42
assert(key.integerIdentity == 42)
let parsed = try Key(parsing: key.description)                             // equal to `key`
let encoded = String(decoding: try JSONEncoder().encode(key), as: UTF8.self) // "\"\(key)\""
```

## Scalars & Quantization

The scalar type of what you hand the index is independent of the scalar type it stores: the library converts on ingest and on export.
`Double`, `Float`, `Int8`, and `UInt8` are native, and the low-precision encodings come from NumKong: `BFloat16`, `E5M2`, `E4M3`, `E3M2`, `E2M3` and `U1x8`, each of which converts from a `Float` rather than only naming a bit pattern.

```swift
let row = [BFloat16(float: 0.3), BFloat16(float: 0.5), BFloat16(float: 1.2)]
```

`Scalar`, `DenseMetric` and `Mood` spell themselves through the library's own name tables, and `Scalar(name: "bf16")` reads one back.

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
Each takes only the placement its own hardware has, one shard per named memory domain, device, or path, and an empty list derives the placement from the machine:

```swift
let inRam = try DenseIndex.initRam(options, memoryDomains: [0, 1])          // one graph per socket
let inVram = try DenseIndex.initVram(options, devices: [0, 1])              // sharded, never replicated
let onNvme = try DenseIndex.viewNvme("index.safegraphs.json", config: DenseViewNvmeConfig(paths: ["/mnt/nvme0"]))
```

An NVMe index serves a checkpoint in place, so it is only ever viewed, and until NVMe checkpoints carry keys, `viewNvme` reports `unsupported`.
Placement never comes back out of a checkpoint, which records where it was written, never where a reader must keep it, so `metadataRead` answers a `DenseMetadata` naming dimensions, metric, quantization and connectivity alone.
`valuesPerKey` is absent from it too, bounding a live handle's mutable state rather than the serialized form.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout: a `.safegraphs` monolith, or a `.safegraphs.json` manifest beside its shards.

```swift
try index.on(executor).save(path: "index.safegraphs")
let copied = try DenseIndex.loadRam("index.safegraphs", config: DenseLoadRamConfig(executor: executor))
let mapped = try DenseIndex.viewRam("index.safegraphs")

let described = try DenseIndex.metadataRead("index.safegraphs")
print(described.dimensions, described.metric, described.quantization)
```

`loadRam` and `loadVram` copy the checkpoint into a new mutable index; `viewRam` serves it in place through a memory mapping, and every mutating verb on a viewed index reports `readOnly`.
Every lifecycle verb that reports takes an optional `progress` handler in its config, as do the batch verbs; returning `.cancel` stops the call with `cancelled`.

An in-memory checkpoint is the same bytes reached another way.
`saveBuffer` returns them as a `Monolith` read in place from an `Arena`, since only the call knows how many there are, and `loadRamBuffer` and `metadataReadBuffer` read them back through a span.

```swift
let monolith = try index.on(executor).saveBuffer(arena: arena)
let restored = try monolith.withUnsafeBytes { bytes in
    try DenseIndex.loadRamBuffer(bytes.bytes, config: DenseLoadRamBufferConfig(executor: executor))
}
```

## Filtered Search and Progress

The predicate runs on the executor's threads, so it must be `@Sendable` and safe for concurrent use; an error it throws rejects that key.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.
`DenseSearchConfig(traversal: .scan)` skips the walk and scores every row exactly, on RAM and NVMe handles, the fallback for a filter too selective for a walk to cross.

```swift
let even = try index.on(executor).search(queries, wanted: 10, arena: arena, config: DenseSearchConfig(
    predicate: { key in key.integerIdentity.map { $0 % 2 == 0 } ?? false }))
```

Every long verb's config takes a `progress` handler, called with the units finished so far and the units the call finishes if it runs to the end; returning `.cancel` stops it wherever the verb can stop, which an `insert` cannot.
A handler never runs on two threads at once, so it may carry a counter or a deadline without a lock, but it may run on whichever thread advanced the work, hence `@Sendable`.
A predicate, handler or value lookup that throws never unwinds into C; the first error stops the call where it can and is thrown once the call is final, a `USearchError` as it was and any other as the `cause` of a `cancelled` one.

```swift
let deadline = ContinuousClock.now + .seconds(30)
try index.on(executor).refine(config: DenseRefineConfig(
    expansionFactor: 128,
    progress: { _, _ in ContinuousClock.now < deadline ? .continue : .cancel }))
```

## Streaming

`cursor()` opens a row scan over the contents, walking in batches, which is what export and migration want.
Each shard's publication watermark is read as the walk enters it, so a row once passed is never revisited.
Every pull lands up to `wanted` keys in its arena, and the rows beside them when `as:` names their scalar kind.

```swift
let cursor = try index.on(executor).cursor()
while true {
    let batch = try cursor.next(wanted: 1024, as: Float.self, arena: arena)
    guard !batch.keys.isEmpty else { break }
    print(Array(batch.keys), batch.vectors.count)
}
```

`searchStream` opens a resumable nearest-neighbor stream over one query row, so a caller that post-filters aggressively pulls until satisfied rather than guessing a `wanted` count upfront.
The stream keeps walk state rather than a thread, every pull is an ordinary call on its receiver's executor, and a failed pull throws.
Ordering beyond the expansion horizon is best-effort nearest rather than globally exact.
The engine declares the stream before implementing it, so opening one reports `unsupported` for now.

```swift
let stream = try index.on(executor).searchStream(query)
while true {
    let pulled = try stream.next(wanted: 64, arena: arena)[0]              // nearest first, empty once drained
    let close = pulled.prefix { $0.distance < 0.25 }
    for match in close { print(match.key) }
    guard !close.isEmpty, close.count == pulled.count else { break }
}
```

Cursors and streams keep their index alive until they go.

## Linear Algebra Over The Graph

Three verbs treat the index as a sparse operand rather than as a kNN store, each folding its arithmetic into the graph walk instead of materializing a match list.
All three read logits rather than distances, so they run on an inner-product index alone.

`multiply` is a sparse matrix product, reporting every pair the traversal scored rather than only the nearest.
How many pairs a walk scores is known only once it ends, so the call returns them as `DenseProducts` read in place from an arena: one row of pairs per left row, in traversal order, and `required` counting every pair scored before a non-zero `wanted` cut any.

```swift
let products = try index.on(executor).multiply(left, arena: arena, config: DenseMultiplyConfig(wanted: 64))
for pair in products[0] { print(pair.key, pair.distance) }
```

`maxsim` is late interaction: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `valuesPerKey` above one exists for, and the outputs are sized over documents rather than rows.

```swift
let documents = try index.on(executor).maxsim(tokens, rowsPerQuery: tokensPerQuery, wanted: 10, arena: arena) // one row per whole query
```

`attend` is softmax attention over every key the graph walk scores, folded as Flash Attention folds key tiles, with outputs already divided by the partition.
Row `t · headCount + h` of the queries, outputs and carry is head `h` of token `t`, so the query heads of grouped-query attention share the index's one key-value head and one walk per token.
The index stores no values: a `ValueLookup` names, for every scored key, its row of a values `RowsView` as wide as the keys and stored in the index's own quantization.
A key without a row, or values of another width or scalar kind than the library reads, refuses the call with `shapeMismatch`, since a missing row is a wrong softmax rather than a smaller one.
The lookup runs on the executor's threads together, so it must be `@Sendable` and safe for concurrent use.
The outputs land in the arena as rows of the scalar kind `as:` names, and a `carry` of one `DenseSoftmaxState` per query row folds this branch into another over the same queries.

```swift
let lookup = ValueLookup(values: RowsView(valueCache)) { key in rowOf[key] }
let context = try carry.withUnsafeMutableBufferPointer { carry in
    try index.on(executor).attend(
        queries, lookup: lookup, as: Float.self, arena: arena,
        config: DenseAttendConfig(headCount: 8, carry: carry))
}
```

## Refinement and Merging

`refine` re-selects the neighbors of every stored entry against the finished graph and reports how many nodes changed edges, so a cheap insertion pass followed by one refinement reaches a better graph in less total time than one expensive insertion pass.
Zero means the graph has settled, and searches may run throughout.

```swift
while try index.on(executor).refine(config: DenseRefineConfig(expansionFactor: 128)) != 0 {}
```

Bigger indexes come from merging, never mutating: `merge` consumes every donor it absorbs, and every later verb on one reports `uninitialized`; a merge stopped early hands the rest back to their indexes.
No residency absorbs donors yet, so the engine answers `unsupported`.

```swift
try index.on(executor).merge([shard])
```

## Metadata & Telemetry

```swift
let description = try index.describe()
print(description.metadata.dimensions, description.metadata.connectivity)
print(description.size, description.capacity, description.memoryUsage, description.capabilities)

let rows = try index.get(key: 42, as: Float.self, arena: arena)
print(rows.count, rows.columns, rows.stride, Array(rows[0]))

let reachability = try index.on(executor).health()
print(reachability.reachable, "of", reachability.audited, "entries reachable")

let stats = DenseTraversalStats()
_ = try index.on(executor).search(queries, wanted: 10, arena: arena, config: DenseSearchConfig(stats: stats))
print(stats.visitedMembers, stats.computedDistances, stats.restarts)

print(version(), capabilitiesCompiled(), capabilitiesAvailable())
```

`health` audits how much of the graph a search reaches from its landmarks; entries it misses stay unfound until a `refine` re-links them.
Every call overwrites the stats it is handed rather than accumulating, so a sweep sums them itself.
Insertions and searches overlap freely on one handle: searches read only entries published behind the visibility watermark, while insertions publish in order.
Serialization is the one exception, demanding a mutation-free window.

## Design

The package is a thin layer over the C ABI in `include/usearch/`, and every rule below follows from that.

- __Names follow the header.__
  Every type is its C name without `usearch_` and `_t`, such as `DenseMetric`, `DenseTraversalStats` and `DenseIndexDescription`; every constructor is a C lifecycle verb such as `DenseIndex.initRam` or `Executor.forRam`, a verb's argument structure becomes its `Dense*Config`, and every wrapper's documentation names the C symbol it calls or says it is binding-only.
- __Arenas for outputs only the call can size.__
  Every verb whose output grows with the call requires an `Arena` and answers a view over it, one verb per operation, and nothing else complicates memory.
- __Blocking calls stay simple.__
  `index.search(queries, wanted: 10, arena: arena)` takes no executor or completion, and a default config derives every knob; nothing about asynchrony reaches a caller who never asks for it.
- __Deferred calls are symmetric, through Swift's own idioms.__
  The receiver's type alone chooses the completion, so `DenseIndex`, `DenseIndexOn` and `DenseIndexDeferredOn` offer the same verbs with the same arguments, and a `Deferred` is awaited with `try await call.value` or waited with `try call.wait()`.
- __Nothing costs anything until it is used.__
  A blocking call lends its inputs in place and frees its outcome, slots and callbacks on return; a deferred one copies only the keys and paths it borrows, and keeps the rest in its `Deferred` until final.
- __A `Deferred` has no life of its own.__
  `status` reads the outcome word with acquire ordering, `wait` blocks in `usearch_outcome_wait`, `value` runs that wait on a Dispatch thread and reads the result back in the caller's isolation, and releasing an unfinished call waits the same way; the binding starts no thread, counts no calls and queues nothing.
- __Statuses and modes carry the C enumerators.__
  `Status`, `Mood`, `DenseMetric` and `Scalar` hold the C values themselves, so nothing is compared as a string, and their `description` and `init(name:)` spell the names the C tables own.
- __Arenas are reused, never shared.__
  Each call rewinds the arena it is given, a view remembers which call wrote it and traps once a later one did, and no index owns an arena or a lock.
