# USearch for Swift

The Swift package binds the C ABI in `c/usearch/` through `USearchC`, a Clang module map over its headers, and links the `libusearch_static` engine CMake builds.
It requires Swift 6.4 or newer, on macOS 26, iOS 26, tvOS 26, watchOS 26, visionOS 26, or Linux, and takes its matrix views and low-precision scalars from NumKong.
It exposes dense indexes over vectors, sequence indexes over strings, shape indexes over 3D point clouds and geospatial indexes over observations of the Earth, each with the same receivers, plans and reports.

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/unum-cloud/USearch", .upToNextMajor(from: "3.0.0"))
]
```

The package compiles the Swift surface only, and depends on NumKong as a source package.
SwiftPM downloads the engine prebuilt, as the release's `USearchC` XCFramework on Apple platforms and as its artifact bundle on Linux, so no C++ toolchain is needed.
A checkout builds its own with CMake and points `USEARCH_SWIFT_ARTIFACT` at it, relative to the package root, as `NUMKONG_SWIFT_ARTIFACT` points the `numkong/` dependency at its own build:

```sh
cmake --preset swift
cmake --build --preset swift
USEARCH_SWIFT_ARTIFACT=build_swift/USearchC.xcframework swift build # USearchC.artifactbundle off Apple platforms
```

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

## Receivers & Deferred Calls

The receiver chooses where a verb runs and how it completes, and the verb stays the same.
Called on the index, a verb runs on the calling thread; on `index.on(executor)` it blocks on that executor, refused while another call holds it; on `index.deferred(executor)` it submits there and answers a `Deferred` at once.
`DenseIndex`, `DenseIndexOn` and `DenseIndexDeferredOn` take each verb with the same arguments, and a blocking call takes no executor or completion, with a default config deriving every knob.

```swift
let here = try index.search(queries, wanted: 10, arena: arena)                  // the calling thread
let blocking = try index.on(executor).search(queries, wanted: 10, arena: arena) // blocking, on the executor
let call = try index.deferred(executor).search(queries, wanted: 10, arena: arena) // Deferred<Matches>
let matches = try await call.value  // or: try call.wait()
```

A `Deferred` owns its outcome and its outputs, and keeps alive every object whose memory the call reads, until the call is final; one owner polls or waits on it, and nothing locks.
It uses the index, the executor and the arena as a blocking call does, since releasing any of them waits for the calls naming it, and its waited result is the same view a blocking call answers.
A blocking call passes its inputs in place and frees its outcome, slots and callbacks on return; a deferred one copies only the keys and paths it views.
`status` reads `deferred` until then, through the outcome word with acquire ordering, and `wait()` blocks and answers the result or throws the failure, the same every time it is asked.
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
One call holds an executor at a time, so a second call finding it held finishes at once with `contended`, and nothing queues.
A deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling `status` alone finishes it, and its progress handler and predicate run on the pool's threads; a single thread or a VRAM executor runs it at submission.
Calls depending on each other are ordered by waiting on the first before submitting the second.
Rows built over a `Tensor` stay alive with the call, while the memory a `MatrixView` points into, or a softmax carry, must outlive it.
The constructors that the C interface lets fan out take an `executor` in their config and run on the calling thread without one.

## Executors

An index says where its payload lives; an `Executor` says who walks it.
One executor serves any number of handles, so several categories of index share one machine's cores or devices instead of each claiming a private set.
Its kind is fixed by the factory that spawned it and must match every handle it drives; a mismatch reports `device`.

```swift
let hostThreads = try Executor.forRam(memoryDomains: [0, 1], threads: 0)
let submissionRings = try Executor.forNvme(paths: ["/mnt/nvme0"], threads: 8)
let deviceStreams = try Executor.forVram(devices: [0, 1], streams: nil, scratchLimit: 1 << 30)
```

Each factory takes every argument its C constructor has, and only those.
`forVram` opens one stream per device, created and destroyed with the executor, answers `unsupported` to your own CUDA streams, since the executor drives only its own, and can cap per-device scratch.
A thread count of zero derives from the hardware: 7/8 of the cores those memory domains hold, since full occupancy starves the ordered publication path, and for NVMe the devices' queue capabilities.
`forVram` takes no thread count at all: one host thread drives one stream per device, so there is no count to name.

One call holds an executor at a time, and index handles stay freely shareable.
`drain(mood:)` waits until no call holds the executor and picks whether its threads park or spin once idle.

## Arenas

Every output whose length only the call knows, such as matches, rows, products, monoliths and cursor or stream batches, lands in an `Arena` you pass, and the verb answers a view over it without copying.
There is one verb per operation: no allocating twin, no caller buffers, and no default arena.
`Arena(on: executor)` places it where that executor computes, decided once, and `Arena()` makes host pages for the calling thread.
Each call rewinds its arena, so the previous call's views die and a steady loop allocates once it reaches its high-water mark.
To keep a result past the next call, or to read one while making the next, pass that call another arena.
An arena serves one call at a time, so each deferred call needs its own.
Views stay on the arena's thread, which the compiler checks, and hold the arena alive.
A view read after a later call took its arena fails as `Contended`.
A slice or segment an accessor returned is valid only until the arena's next call.
A view cannot be `~Escapable` while its lifetime annotation is still behind an experimental feature flag as of Swift 6.4, so the engine's generation, not the compiler, tells which call wrote it.
An arena from a VRAM executor serves only that executor's calls, keeping the executor alive until the arena goes. `describe()` reports its device, vendor and `hostAccess`: reading a result in memory the host may not read, as CUDA and ROCm device memory, traps with "the outputs are memory the host may not read", and `description.host()` throws `device` for it first, while Apple's shared memory reads like host pages.

```swift
let arena = try Arena(on: executor)                                              // Arena() is host pages for the calling thread
for batch in batches {
    let matches = try index.on(executor).search(batch, wanted: 10, arena: arena) // rewinds: last batch's matches are gone
    total += matches.joined().reduce(0) { $0 + $1.distance }
}
let found = try Arena(), fetched = try Arena()                                   // two live results, two arenas
for match in try index.search(query, wanted: 10, arena: found)[0] {
    rerank(match.key, try index.get(key: match.key, as: Float.self, arena: fetched))
}
let calls = try shards.indices.map { at in
    try shards[at].deferred(executors[at]).search(queries, wanted: 10, arena: arenas[at]) // one executor and one arena per submitted call
}
var results: [Matches] = []
for call in calls { results.append(try await call.value) }                       // or: try call.wait()
```

A result keeps the arena it reads alive, so it outlives the index and the caller's reference alike.

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

## Residency

Where the payload lives is the constructor's name, as in the C lifecycle verbs, so a residency an index lacks has no constructor to call.
`initRam` and `initVram` start empty, `loadRam`, `loadVram` and their `Buffer` forms copy a checkpoint in, and `viewRam` and `viewNvme` serve one in place.
Each takes only the placement its own hardware has, one shard per named memory domain, device, or path, and an empty list derives the placement from the machine:

```swift
let inRam = try DenseIndex.initRam(options, memoryDomains: [0, 1])          // one graph per socket
let inVram = try DenseIndex.initVram(options, devices: [0, 1])              // sharded, never replicated
let onNvme = try DenseIndex.viewNvme("index.safegraphs.json", config: DenseViewNvmeConfig(paths: ["/mnt/nvme0"]))
```

An NVMe view serves a checkpoint in place, and `keyTableDomains` holds a key table in RAM beside it; without one, `count`, `get`, `sample`, MaxSim and hinted searches report `unsupported`.
`loadNvme` takes the same config and opens a manifest writable, so `refine` rewrites its parts one at a time, while a lone monolith reports `unsupported`.
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
Saving demands a mutation-free window, while concurrent readers keep searching throughout.

An in-memory checkpoint is the same bytes reached another way.
`saveBuffer` returns them as a `Monolith` read in place from an `Arena`, since only the call knows how many there are, and `loadRamBuffer` and `metadataReadBuffer` read them back through a span.

```swift
let monolith = try index.on(executor).saveBuffer(arena: arena)
let restored = try monolith.withUnsafeBytes { bytes in
    try DenseIndex.loadRamBuffer(bytes.bytes, config: DenseLoadRamBufferConfig(executor: executor))
}
```

## Dense Indexes

A `DenseIndex` holds equidimensional vectors under keys, and its verbs below take every receiver above.

### Rows & Scalars

A `RowsView` views a NumKong `MatrixView`, or keeps a NumKong `Tensor` alive.
A matrix view only points at memory it never owns, so it stays valid only for as long as the storage it points into, which is why a call over one sits inside a `withUnsafeBufferPointer`.

```swift
let padded: [Float] = load()                                               // 768 dimensions pitched at 1024
try padded.withUnsafeBufferPointer { buffer in
    let rows = MatrixView(baseAddress: buffer.baseAddress!, rows: 1_000, columns: 768, rowStrideBytes: 1024 * 4)
    try index.on(executor).insert(keys: keys, vectors: RowsView(rows))
}
```

The library checks every row view's width against the index geometry itself, so a mismatch surfaces as its own `shapeMismatch`.
Sub-byte kinds count logical dimensions rather than stored values, so a `U1x8` view one value wide holds eight dimensions.

The scalar type of what you hand the index is independent of the scalar type it stores: the library converts on ingest and on export.
`Double`, `Float`, `Int8`, and `UInt8` are native, and the low-precision encodings come from NumKong: `BFloat16`, `E5M2`, `E4M3`, `E3M2`, `E2M3` and `U1x8`, each of which converts from a `Float` rather than only naming a bit pattern.

```swift
let row = [BFloat16(float: 0.3), BFloat16(float: 0.5), BFloat16(float: 1.2)]
```

`Scalar`, `DenseMetric` and `Mood` spell themselves through the library's own name tables, and `Scalar(name: "bf16")` reads one back.

### Search

A search writes up to `wanted` matches per query into its arena, ranked nearest first, and answers `Matches` reading them in place: `matches[query]` is one query's `QueryMatches`, so neither level copies a match out.
A query's own `count` stops at the matches the call wrote, however many were wanted.
Both levels are `RandomAccessCollection`, so a query walks, maps and slices like any other sequence.

```swift
let stats = DenseTraversalStats()
let matches = try index.on(executor).search(queries, wanted: 10, arena: arena, config: DenseSearchConfig(
    expansionFactor: 64, stats: stats))
for query in 0 ..< matches.count {
    for match in matches[query] { print(match.key, match.distance) }
}
print(stats.visitedMembers, stats.computedDistances, stats.restarts)
```

`expansionFactor` is the breadth of the candidates the walk explores, and zero derives it from the geometry.
The `insert`, `refine`, `search`, `maxsim`, `multiply` and `attend` configs take a `DenseTraversalStats`, and every call overwrites the stats it is handed rather than accumulating, so a sweep sums them itself.
Insertions and searches overlap freely on one handle: searches read only entries published behind the visibility watermark, while insertions publish in order.

### Filtered Search

The predicate runs on the executor's threads, so it must be `@Sendable` and safe for concurrent use.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.
`DenseSearchConfig(traversal: .scan)` skips the walk and scores every entry exactly, on RAM and NVMe handles, the fallback for a filter too selective for a walk to cross.

```swift
let even = try index.on(executor).search(queries, wanted: 10, arena: arena, config: DenseSearchConfig(
    predicate: { key in key.integerIdentity.map { $0 % 2 == 0 } ?? false }))
```

Every long verb's config, and every lifecycle config that reports, takes a `progress` handler, called with the units finished so far and the units the call finishes if it runs to the end; returning `.cancel` stops it with `cancelled` wherever the verb can stop, which an `insert` cannot.
A handler never runs on two threads at once, so it may hold a counter or a deadline without a lock, but it may run on whichever thread advanced the work, hence `@Sendable`.
A predicate, handler or value lookup never throws, as its type says, so nothing unwinds into C; a handler stops a call by returning `.cancel`.

```swift
let deadline = ContinuousClock.now + .seconds(30)
try index.on(executor).refine(config: DenseRefineConfig(
    expansionFactor: 128,
    progress: { _, _ in ContinuousClock.now < deadline ? .continue : .cancel }))
```

### Retrieval and Streaming

`get` lands every vector stored under a key in its arena, one row per entry sharing the key, converted to the scalar kind `as:` names.

```swift
let vectors = try index.get(key: 42, as: Float.self, arena: arena)
print(vectors.count, vectors.columns, vectors.stride, Array(vectors[0]))
```

`cursor()` opens a walk over the contents in batches, which is what export and migration want.
Each shard's publication watermark is read as the walk enters it, so an entry once passed is never revisited.
Without `starts:` it yields every entry as stored; with them, the starts' own entries first, then the entries those link to breadth-first, each once.
Every pull lands up to `wanted` keys in its arena, and `nextLinks` lands the keys each links to beside them; the vectors themselves come from `get`.
The cursor keeps no executor: a pull on the cursor runs on the calling thread, and one through `cursor.on(executor)` blocks on that executor.

```swift
let cursor = try index.cursor()
while true {
    let batch = try cursor.on(executor).nextLinks(wanted: 1024, arena: arena) // each pull names its executor
    guard !batch.keys.isEmpty else { break }
    print(Array(batch.keys), batch.links[0])
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

### Refinement

`refine` re-selects the neighbors of every stored entry against the finished graph and returns a `DenseRefineReport`: the entries re-selected and changed, the islands found and reconnected, the neighborhoods re-stitched with the links added and dropped, the `verdict`, and the health checks opening the round and closing the heal.
A cheap insertion pass followed by one refinement reaches a better graph in less total time than one expensive insertion pass.
A caller loops until the `verdict` is no longer `.improved`, and searches may run throughout.

```swift
while try index.on(executor).refine(config: DenseRefineConfig(expansionFactor: 128)).verdict == .improved {}
```

On a loaded NVMe handle `memoryLimit` bounds the bytes a round holds for the one part it rewrites: zero rounds the part in RAM, and a smaller budget streams its rounds from the drives.

### Resharding

`reshard` moves the entries of its `sources` into its `targets`, keeping their links rather than re-inserting their vectors, then runs a round on every target that took entries; every handle is named, the index itself included.
Several sources gather into one target, itself among them, and each other source ends empty and live.

```swift
try index.on(executor).reshard(sources: [index, shard], targets: [index])
```

One source cut into several empty targets hands each a graph-local part, keeping their edges and leaving the source empty but live.
Every reshard reports its closing rounds as `refine` does.

```swift
let report = try index.on(executor).reshard(sources: [index], targets: [left, right], config: DenseReshardConfig(seed: 42))
```

The index as its only source and target recuts its own shards in place into as few as fit `shardCapacity`, keeping every entry.

```swift
try index.on(executor).reshard(
    sources: [index],
    targets: [index],
    config: DenseReshardConfig(memoryDomains: [], shardCapacity: 10_000_000)
)
```

A `DenseRefinePlan` or `DenseReshardPlan` in the config keeps a stopped call live: the next call naming it continues where the stop came, its `phase` says where each call stopped, and `finish(_:)` closes it and reports what the whole plan did.

```swift
let plan = DenseRefinePlan()
while plan.phase.phase != .done {
    do {
        _ = try index.on(executor).refine(config: DenseRefineConfig(progress: withinBudget, plan: plan))
    } catch where error.status == .cancelled {}
}
let report = try index.on(executor).finish(plan)
```

### Sampling

`sample` draws keys around `focus`, or across the whole index, each hanging from a shown or earlier draw and placed in three dimensions beside the `shown` keys already on screen, so a view grows one connected sample as it zooms.
Equal seeds, shown keys and focus draw equally, and a larger `wanted` draws a superset.

```swift
let draws = try index.on(executor).sample(wanted: 256, arena: arena, config: DenseSampleConfig(
    shown: shown, shownCoordinates: places, focus: near, seed: 42))
for draw in 0 ..< draws.count {
    let (key, entryOrdinal, parent, point) = (draws.keys[draw], draws.entryOrdinal(draw), draws.parents[draw], draws.coordinates[draw])
}
```

### Sparse Multiplication

Three verbs, `multiply`, `maxsim` and `attend`, treat the index as a sparse operand rather than as a kNN store, each folding its arithmetic into the graph walk instead of materializing a match list.
All three read logits rather than distances, so they run on an inner-product index alone.

`multiply` is a sparse matrix product, reporting every pair the traversal scored rather than only the nearest.
How many pairs a walk scores is known only once it ends, so the call returns them as `DenseProducts` read in place from an arena: one run of pairs per left row, in traversal order, and `required` counting every pair scored before a non-zero `wanted` cut any.

```swift
let products = try index.on(executor).multiply(left, arena: arena, config: DenseMultiplyConfig(wanted: 64))
for pair in products[0] { print(pair.key, pair.distance) }
```

### Late Interaction

`maxsim` is late interaction: every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `valuesPerKey` above one exists for, and the outputs are sized over documents rather than rows.
The call returns the same `DenseProducts` as `multiply`, one ragged run per whole query with its documents largest score first, and a zero `wanted` keeps every document the query scored.

```swift
let documents = try index.on(executor).maxsim(tokens, rowsPerQuery: tokensPerQuery, wanted: 10, arena: arena)
for document in documents[0] { print(document.key, document.distance) }
```

### Attention

`attend` is softmax attention over every key the graph walk scores, folded as Flash Attention folds key tiles, with outputs already divided by the partition.
Row `t · headCount + h` of the queries, outputs and carry is head `h` of token `t`, so the query heads of grouped-query attention share the index's one key-value head and one walk per token.
The index stores no values: a `ValueLookup` names, for every scored key, its row of a values `RowsView` as wide as the keys and stored in the index's own quantization.
A key without a row, or values of another width or scalar kind than the library reads, refuses the call with `shapeMismatch`, since a missing row is a wrong softmax rather than a smaller one.
A VRAM index walks on the device and resolves values through a device function, which a Swift closure cannot be, so `attend` on it reports `unsupported`.
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

### Introspection

```swift
let description = try index.describe()
print(description.metadata.dimensions, description.metadata.connectivity)
print(description.size, description.capacity, description.memoryUsage, description.cpuClass, description.shardsCount)
print(try index.count(key: 42), try index.contains(key: 42))

let reachability = try index.on(executor).health()
print(reachability.reachable, "of", reachability.audited, "entries reachable at depth", reachability.depthFromRoot)

print(version(), capabilitiesCompiled(), capabilitiesAvailable())
```

`health` measures how much of the graph a search reaches from the root; entries it misses stay unfound until a `refine` re-links them.

## Sequence Indexes

A `SequenceIndex` searches proteins under `.proteinAlignment`, DNA and RNA under `.dnaOverlap`, and UTF-8 text under `.utf8Containment`.
A sequence is a `[UInt8]` of raw bytes, and a batch crosses as one Apache Arrow tape the binding packs:

```swift
let index = try SequenceIndex.initRam(SequenceOptions(metric: .utf8Containment))
let texts = ["The quick brown fox", "A lazy dog", "Pack my box"].map { Array($0.utf8) }
try index.insert(keys: [1, 2, 3], sequences: texts)

let matches = try index.search([Array("BROWN FOX".utf8)], wanted: 3, arena: try Arena())
print(matches[0][0].key, matches[0][0].distance) // key 1, whatever the case
```

Each metric stores a sequence folded, so `get` returns the stored form rather than the inserted bytes: packed class codes for a protein, one class code a byte for DNA, and case-folded UTF-8 for text.
Text folds its case inside the engine, so there is nothing to pack.
Optional `SequenceCosts` give 256 × 256 similarities between raw bytes and a gap cost in place of the metric's own, and a containment refuses them.
`SequenceOptions.sketchBytes` sets the bytes of the sketch in front of each stored sequence, a multiple of four, where zero takes the metric's default of 512 for protein alignment and UTF-8 containment.
A DNA overlap keeps no sketch and refuses any.
A checkpoint records the costs, so loads and views take none.
The verbs, receivers and plans are the dense ones: `refine` and `reshard` take a `SequenceRefinePlan` or `SequenceReshardPlan` in their configs, `get` answers each stored sequence in place as `TapeEntries<UInt8>`, and `count` reports a key's entries with their total and largest byte lengths.

## Shape Indexes

A `ShapesIndex` searches 3D point clouds of any size, like molecules, conformers and protein backbones, under Chamfer, Hausdorff, Gromov-Wasserstein or Wasserstein-Procrustes, each blind to rotation, reflection, translation and point order.
A shape is its coordinates, `x`, `y` and `z` of each point in turn, in any `ScalarValue`, and point counts may differ between shapes:

```swift
let index = try ShapesIndex.initRam(ShapesOptions(metric: .hausdorff))
let water: [Float] = [0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0]
let methane: [Float] = [0, 0, 0, 0.63, 0.63, 0.63, -0.63, -0.63, 0.63, -0.63, 0.63, -0.63, 0.63, -0.63, -0.63]
try index.insert(keys: [1, 2], shapes: [water, methane])

let matches = try index.search([methane], wanted: 1, arena: try Arena())
print(matches[0][0].key) // methane finds itself
```

`ShapesOptions` freezes `f16` or `f32` storage, and a non-zero `points` refuses shapes of any other count with `shapeMismatch`.
`get(key:as:arena:)` converts to the scalar asked for and returns each shape in its inserted pose, and `count` reports a key's total and largest point counts.

## Geospatial Indexes

A `GeospatialIndex` searches observations on and above the Earth, like photos, video frames, drone footage and sensor fixes.
A `GeospatialObservation` holds eleven doubles in natural units, and every field left out is NaN, which only widens the tests that read it:

```swift
let index = try GeospatialIndex.initRam()
let stops = [
    GeospatialObservation(latitudeDegrees: 51.5308, longitudeDegrees: -0.1238, accuracyMetres: 5),
    GeospatialObservation(latitudeDegrees: 51.5320, longitudeDegrees: -0.1233, accuracyMetres: 5),
    GeospatialObservation(latitudeDegrees: 51.5033, longitudeDegrees: -0.1196, accuracyMetres: 5),
]
try index.insert(keys: [1, 2, 3], observations: stops)

let station = GeospatialObservation(latitudeDegrees: 51.5308, longitudeDegrees: -0.1238)
let nearby = try index.within([station], tolerance: GeospatialTolerance(metres: 500), arena: try Arena())
print(nearby[0].map(\.key)) // the two stops within 500 metres
```

`within` reports every entry a `GeospatialRelation` admits, however many, while `search` ranks the `wanted` nearest, both scored by the chord between positions in metres.
`.near` widens the radius by both positions' accuracies, `.visible` needs a clear line of sight over the handle's `GeospatialTerrain`, and `.sees` and `.seen` test the entry's or the query's view cone.
The terrain is a grid of `f32` heights in a `RowsView`, which the index reads in place for its lifetime and keeps alive when built over a `Tensor`; a checkpoint never holds it, so every load and view names it again.

## Left Unbound

- `cursor` and `searchStream` pull blocking, so a `DenseIndexDeferredOn` lacks them; `contains` is a test over `count`, so the deferred receiver offers `count` alone.
- The constructors address no index yet, so no receiver offers them.
- A geospatial terrain filled on demand by a callback: the library calls it from the executor's threads for every sight line, so the binding passes only a grid of heights or a dataset path.
- `searchStream`, `maxsim`, `multiply` and `attend` exist for dense indexes alone, as in the C library.
- The Arrow converters of `c/usearch/arrow.h`, which view an Arrow column as an input and export an arena's output as one.

## Design

- __A thin layer over the C ABI.__
  Every name follows `c/usearch/`: a C verb `usearch_dense_<verb>` is a camel-cased `<verb>` here, such as `saveBuffer`, a type is its C name without `usearch_` and `_t`, and enums mirror C enums, never strings.
- __The receiver chooses, the verb stays.__
  The index runs a verb on the calling thread, `on(executor)` blocks on an executor, and `deferred(executor)` submits it, all with the same arguments.
- __Deferred calls use the language's own idioms.__
  `status` polls through `usearch_outcome_status`, `wait()` blocks through `usearch_outcome_wait`, and `try await call.value` polls `status` between `Task.sleep` pauses growing from 1 µs to 1 ms, so an `async` caller suspends and no thread parks.
- __No life of its own.__
  The binding starts no threads and keeps no queues; releasing a `Deferred` rides its `deinit`, which waits for the call first.
- __Arenas belong to the caller and are reused, never shared.__
  Every output whose length scales with the call lands in an arena the caller passes, and its views end at the arena's next call.
- __Nothing costs anything until it is used.__
  Submitting a call allocates nothing, and an arena allocates only when a call writes into it.
- __Every wrapper names its C symbol.__
  Each wrapper's documentation names the C function it calls or says it is binding-only, and a verb's argument structure becomes its `Dense*Config`, or a `Basic*Config` shared by the other three modalities.
