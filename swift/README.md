# USearch for Swift

## Installation

USearch is available through the Swift Package Manager.

```swift
dependencies: [
    .package(url: "https://github.com/unum-cloud/USearch", .upToNextMajor(from: "3.0.0"))
]
```

The package compiles the Swift surface only, and depends on NumKong for the matrix views and the low-precision scalars.
The engine itself is `libusearch_c`, built by CMake and linked by the package.

## Quickstart

Every verb is a batch over a NumKong `MatrixView`: a base pointer, a row count, a column count, and the byte pitch between consecutive rows.
A single vector is a one-row matrix, so nothing is reshaped and nothing is copied.

```swift
import NumKong
import USearch

let index = try DenseIndex(options: DenseOptions(
    dimensions: 3, metric: .angular, quantization: .f32))
let executor = try Executor.forRam()
try index.reserve(2)

let vectors: [Float] = [0.3, 0.5, 1.2, 0.4, 0.2, 1.2]
try vectors.withUnsafeBufferPointer { buffer in
    _ = try index.add(
        executor: executor, keys: [42, 43],
        vectors: MatrixView(baseAddress: buffer.baseAddress!, rows: 2, cols: 3))
}

let queries: [Float] = [0.3, 0.5, 1.2]
var keys = [Key](repeating: .nilKey, count: 10)
var distances = [Distance](repeating: 0, count: 10)
var counts = [Int](repeating: 0, count: 1)

try keys.withUnsafeMutableBufferPointer { keyBuffer in
    try distances.withUnsafeMutableBufferPointer { distanceBuffer in
        try counts.withUnsafeMutableBufferPointer { countBuffer in
            let matches = try MatchesSpan(keys: keyBuffer, distances: distanceBuffer, counts: countBuffer)
            try queries.withUnsafeBufferPointer { buffer in
                _ = try index.search(
                    executor: executor,
                    queries: MatrixView(baseAddress: buffer.baseAddress!, rows: 1, cols: 3),
                    into: matches)
            }
            assert(matches[0][0].key == Key(42))
        }
    }
}
```

A search writes into buffers you own and returns only its telemetry, so a sweep allocates once rather than once per query.
The width of the key buffer against the count buffer _is_ the "k" in "kANN", so the space reserved and the count asked for cannot disagree.

A matrix borrows and never owns, so it stays valid only for as long as the storage it points into — which is why every call site above sits inside a `withUnsafeBufferPointer`.
`MatchesSpan` is `~Escapable` for the same reason, so the compiler refuses a span that outlives the closure holding its buffers rather than leaving it to a comment.

## Errors

`Error` is the only error the SDK throws, and every method says so with `throws(Error)`, so a `catch` reads `error.status` with no cast.

```swift
do {
    try index.reserve(1 << 40)
} catch {
    print(error.status.name, error.message)
}
```

## Results

A run counts its queries, and `matches[query]` reads one query's row through the same buffers, so neither level copies a row out.
A row's own `count` stops at the cells the engine wrote, however wide the buffers are.

```swift
for query in 0 ..< matches.count {
    let row = matches[query]
    if row.wasTruncated { print("the engine had more to report than `wanted` could hold") }
    for position in 0 ..< row.count { print(row[position].key, row[position].distance) }
}
```

Neither level is a `Collection`: a `~Escapable` type cannot conform to one, and that is the price of having the compiler check the borrow.

## Executors

An index says where its payload lives; an `Executor` says who walks it.
One executor serves any number of handles, so several categories of index share one machine's cores or devices instead of each claiming a private set.
Its kind is fixed by the factory that spawned it and must match every handle it drives; a mismatch reports `deviceFailure` on the first call.

```swift
let hostThreads = try Executor.forRam(memoryDomains: [0, 1], threads: 0)
let submissionRings = try Executor.forNvme(paths: ["/mnt/nvme0"], threads: 8)
let deviceStreams = try Executor.forVram(devices: [0, 1])
```

Each factory takes only what its residency has.
A thread count of zero derives from the hardware — 7/8 of the cores those memory domains carry, since full occupancy starves the ordered publication path, and for NVMe the devices' queue capabilities.
`forVram` takes no thread count at all: one host thread drives one stream per device, so there is no count to name.

An executor admits one driver at a time, reporting `contended` to the second, so a thread that wants to drive its own calls spawns its own.
Index handles stay freely shareable.

## Keys

A key is sixteen opaque bytes, rendered as a lowercase dashed UUID.
Integer identities park in the low eight bytes, so `Key(42)` prints as `00000000-0000-0000-0000-00000000002a` and an integer literal is a key wherever one is expected.
Keys need not be unique: several entries may share one, and `count(key:)` reports how many.

A key is `Codable` as that same string, so a document written by any SDK reads back here unchanged.

```swift
let key: Key = 42
assert(key.integerIdentity == 42)
assert(try Key(parsing: key.description) == key)
assert(String(decoding: try JSONEncoder().encode(key), as: UTF8.self) == "\"\(key)\"")
```

## Scalars & Quantization

The scalar type of what you hand the index is independent of the scalar type it stores:
the library converts on ingest and on export.
`Double`, `Float`, `Int8`, and `UInt8` are native, and the low-precision encodings come from NumKong — `BFloat16`, `E5M2`, `E4M3`, `E3M2`, `E2M3`, `U1x8` — each of which converts from a `Float` rather than only naming a bit pattern.

```swift
let row = [BFloat16(float: 0.3), BFloat16(float: 0.5), BFloat16(float: 1.2)]
```

Three kinds have no NumKong counterpart and stay here: `Float16Bits`, because Swift's own `Float16` is absent on x86_64 at the type level, and `Int4x2Bits` and `UInt4x2Bits`, whose packing is USearch's own quantization.

## Residency

Where the payload lives, and what loading a checkpoint means, is one enum — one shard per named memory domain, device, or path, and an empty list derives the placement from the machine:

```swift
DenseOptions(dimensions: 768, residency: .ram(memoryDomains: [0, 1]))
DenseOptions(dimensions: 768, residency: .vram(devices: [0, 1]))
DenseOptions(dimensions: 768, residency: .nvme(paths: ["/mnt/nvme0", "/mnt/nvme1"]))
```

A dimensionality of zero is legal: the geometry defers to a later `load`, which reads it off the checkpoint.
Placement never comes back out of one — a checkpoint records where it was written, never where a reader must keep it — so `metadata` answers a `DenseMetadata` naming dimensions, metric, quantization and connectivity alone.
`valuesPerKey` is absent from it too, bounding a live handle's mutable state rather than the serialized form.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout —
a `.safegraphs` monolith, or a `.safegraphs.json` manifest beside its shards.

```swift
try index.save(executor: executor, path: "index.safegraphs")
try index.load(executor: executor, path: "index.safegraphs")

let described = try DenseIndex.metadata(path: "index.safegraphs")
print(described.dimensions, described.metric.name, described.quantization.name)
```

An in-memory checkpoint is the same bytes reached another way; size the buffer with `storageRequirements`.

```swift
var buffer = [UInt8](repeating: 0, count: try index.storageRequirements)
try buffer.withUnsafeMutableBytes { try index.save(executor: executor, buffer: $0) }
```

## Filtered Search

The predicate runs on the executor's threads, so it must be safe for concurrent use and cannot throw.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```swift
try queries.withUnsafeBufferPointer { buffer in
    _ = try index.search(
        executor: executor,
        queries: MatrixView(baseAddress: buffer.baseAddress!, rows: 1, cols: 3),
        into: matches,
        admitting: { key in key.integerIdentity.map { $0 % 2 == 0 } ?? false })
}
```

## Streaming

`cursor()` opens a consistent snapshot over the contents — the publication watermark is captured when it opens, so concurrent inserts never appear mid-iteration — and it walks in batches, which is what export and migration want.

```swift
var keys = [Key](repeating: .nilKey, count: 1024)
let cursor = try index.cursor()
while true {
    let written = try keys.withUnsafeMutableBufferPointer { try cursor.next(into: $0) }
    guard written != 0 else { break }
    print(keys.prefix(written))
}
```

`find(executor:query:)` opens a resumable nearest-neighbor stream over one query row, so a caller that post-filters aggressively pulls until satisfied rather than guessing a `wanted` count upfront.
A live finder pins one of the executor's threads for as long as it exists, so an executor with few threads and many open finders starves — drop each one the moment its caller stops pulling.
Ordering beyond the expansion horizon is best-effort nearest rather than globally exact, and because an iterator cannot throw, a failed pull ends the stream and lands in `failure`.

```swift
let finder = try index.find(executor: executor, query: queryView)
for match in finder where match.distance < 0.25 { print(match.key) }
assert(finder.failure == nil)
```

## Linear Algebra Over The Graph

Three verbs treat the index as a sparse operand rather than as a kNN store, each folding its arithmetic into the graph walk instead of materializing a match list.
All three read logits rather than distances, so they run on an inner-product index alone.

`multiply(executor:left:into:options:)` is a sparse matrix product, reporting every pair the traversal scored rather than only the nearest, and landing them in the same `MatchesSpan` a search fills.

```swift
try left.withUnsafeBufferPointer { buffer in
    _ = try index.multiply(
        executor: executor,
        left: MatrixView(baseAddress: buffer.baseAddress!, rows: rows, cols: 768),
        into: products)
}
```

`maxsim(executor:queries:rowsPerQuery:into:options:)` is late interaction — every query row takes its best match within a key, and those maxima sum to that key's score.
A document is one key holding its token rows, which is what a `valuesPerKey` above one exists for, and the outputs are sized over documents rather than rows.

```swift
try tokens.withUnsafeBufferPointer { buffer in
    _ = try index.maxsim(
        executor: executor,
        queries: MatrixView(baseAddress: buffer.baseAddress!, rows: tokens.count / 768, cols: 768),
        rowsPerQuery: tokensPerQuery, into: documents)
}
```

`attend(executor:queries:values:into:carry:options:)` is softmax-weighted attention, gathering each value row exactly once and writing outputs already divided by their partition.
An entry's key is its row offset into `values`, so a context inserted in position order needs no mapping, and a `carry` of one `SoftmaxState` per query row folds this branch into another over the same queries.
The outputs arrive in a NumKong `MatrixSpan`, the written-through twin of a view, which is also what `get(key:into:)` fills.

```swift
_ = try index.attend(
    executor: executor, queries: queryView, values: valueView,
    into: contextSpan, carry: carryBuffer)
```

## Refinement

`refine(executor:expansionFactor:)` re-selects the neighbors of every stored entry against the finished graph and reports how many nodes changed edges, so a cheap insertion pass followed by one refinement reaches a better graph in less total time than one expensive insertion pass.
Zero means the graph has settled, and searches may run throughout.

```swift
while try index.refine(executor: executor, expansionFactor: 128).changed != 0 {}
```

## Metadata & Telemetry

```swift
let dimensions = try index.dimensions
let connectivity = try index.connectivity
let entries = try index.size
let bytes = try index.memoryUsage
let dispatched = try index.capabilities

let stats = try index.search(executor: executor, queries: queryView, into: matches)
print(stats.visitedMembers, stats.computedDistances, stats.restarts)
```

Insertions and searches overlap freely on one handle: searches read only entries published behind the visibility watermark, while insertions publish in order.
Serialization is the one exception, demanding a mutation-free window.
