# USearch for Swift

## Installation

USearch is available through the Swift Package Manager.

```swift
dependencies: [
    .package(url: "https://github.com/unum-cloud/USearch", .upToNextMajor(from: "3.0.0"))
]
```

The package compiles the Swift surface only.
The engine itself is `libusearch_c`, built by CMake, so point the linker at it — `swift build -Xlinker -L<build directory>` — or install it where the system linker already looks.

```sh
cmake -B build -D USEARCH_BUILD_LIB_C=1 && cmake --build build --config Release
swift build -Xlinker -Lbuild
```

## Quickstart

Every verb is a batch over a strided matrix: a base pointer, a row count, and the byte pitch between consecutive rows.
A single vector is a one-row matrix, so nothing is reshaped and nothing is copied.

```swift
import USearch

let index = try USearchIndex(options: USearchDenseOptions(
    dimensions: 3, metric: .angular, quantization: .f32))
try index.reserve(2)

let vectors: [Float] = [0.3, 0.5, 1.2, 0.4, 0.2, 1.2]
try vectors.withUnsafeBufferPointer { buffer in
    _ = try index.add(keys: [42, 43], vectors: USearchMatrixView.rows(of: buffer, dimensions: 3))
}

let queries: [Float] = [0.3, 0.5, 1.2]
let matches = try queries.withUnsafeBufferPointer { buffer in
    try index.search(queries: USearchMatrixView.rows(of: buffer, dimensions: 3), wanted: 10)
}
assert(matches.matches(query: 0).first?.key == USearchKey(42))
```

## Keys

A key is sixteen opaque bytes, rendered as a lowercase dashed UUID.
Integer identities park in the low eight bytes, so `USearchKey(42)` prints as `00000000-0000-0000-0000-00000000002a` and an integer literal is a key wherever one is expected.
Keys need not be unique: several entries may share one, and `count(key:)` reports how many.

```swift
let key: USearchKey = 42
assert(key.integerIdentity == 42)
assert(try USearchKey(parsing: key.description) == key)
```

## Scalars & Quantization

The scalar type of what you hand the index is independent of the scalar type it stores:
the library converts on ingest and on export.
`Double`, `Float`, `Int8`, and `UInt8` are native; the encodings Swift has no builtin for arrive as bit-holding structs — `USearchBFloat16`, `USearchFloat16`, `USearchE5M2`, `USearchE4M3`, `USearchE3M2`, `USearchE2M3`, `USearchI4x2`, `USearchU4x2`, `USearchU1x8`.

## Residency

Where the payload lives, and what loading a checkpoint means, is one enum:

```swift
USearchDenseOptions(dimensions: 768, residency: .ram(threads: 0, pages: .basePages))
USearchDenseOptions(dimensions: 768, residency: .vram(device: -1, addressing: .deviceOnly))
USearchDenseOptions(dimensions: 768, residency: .nvme(sectorBytes: 512, queueDepth: 64, io: .directIO))
```

A dimensionality of zero is legal: the geometry defers to a later `load`, which reads it off the checkpoint.

## Serialization

Checkpoints ride the SafeGraphs interchange format, and the path extension picks the layout —
a `.safegraphs` monolith, or a `.safegraphs.json` manifest beside its shards.

```swift
try index.save(path: "index.safegraphs")
try index.load(path: "index.safegraphs")

let described = try USearchIndex.metadata(path: "index.safegraphs")
print(described.dimensions, described.metric.name, described.quantization.name)
```

## Filtered Search

The predicate runs on the library's own worker threads, so it must be safe for concurrent use and cannot throw.
Non-matching entries still route the traversal, so recall does not collapse at low selectivity.

```swift
let matches = try queries.withUnsafeBufferPointer { buffer in
    try index.search(
        queries: USearchMatrixView.rows(of: buffer, dimensions: 3), wanted: 10,
        admitting: { key in key.integerIdentity.map { $0 % 2 == 0 } ?? false })
}
```

## Metadata & Telemetry

```swift
let dimensions = try index.dimensions
let connectivity = try index.connectivity
let entries = try index.size
let bytes = try index.memoryUsage
let dispatched = try index.capabilities

print(matches.stats.visitedMembers, matches.stats.computedDistances, matches.stats.restarts)
```

Insertions and searches overlap freely on one handle: searches read only entries published behind the visibility watermark, while insertions publish in order.
Serialization is the one exception, demanding a mutation-free window.
`merge(donor:)` consumes the donor outright — the absorbing index takes its handle, and any concurrent verb on the donor is a use-after-free.
