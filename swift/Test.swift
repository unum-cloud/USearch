//
//  Test.swift
//  USearch
//
//  The boundary suite: what the binding itself decides - key byte order, row lowering, handle
//  lifetime, the rows a call wrote, the streaming handles, the callback trampolines, the arena
//  views, and how blocking and deferred calls complete. That vector search returns near neighbors
//  is the engine's contract and is tested where the engine lives.
//
//  Suites run in parallel and every test gets a fresh instance, so nothing here is shared: each
//  test spawns the one executor it drives, and the helpers below are free functions rather than
//  members for the same reason.
//

import Foundation
import Testing

import NumKong

import USearch

/// One executor per test, so no test waits behind another's calls.
private func executor() throws(USearchError) -> Executor { try Executor.forRam() }

/// The error a call failed with, and nil when it succeeded. `USearchError` is the only error the
/// SDK throws, so the catch needs no cast.
private func failure<Value>(of body: @autoclosure () throws(USearchError) -> Value) -> USearchError? {
    do {
        _ = try body()
        return nil
    }
    catch {
        return error
    }
}

/// The status a call failed with, and nil when it succeeded.
private func failureStatus<Value>(of body: @autoclosure () throws(USearchError) -> Value) -> USearch.Status? {
    failure(of: try body())?.status
}

/// The status a body failed with, for bodies that also build tensors and so throw untyped.
private func failureStatus(_ body: () throws -> Void) -> USearch.Status? {
    do {
        try body()
        return nil
    }
    catch {
        return (error as? USearchError)?.status
    }
}

/// Rows over an owning tensor, which a deferred call keeps alive until final.
private func rows<Element: ScalarValue>(_ values: [Element], columns: Int) throws -> RowsView {
    RowsView(try Tensor.fromArray(values, rows: values.count / columns, columns: columns))
}

/// Views an array as rows of the given width for the duration of `body`, which is what keeps the
/// borrow and the storage the same length.
private func withMatrix<Element: ScalarValue, Result, Failure: Swift.Error>(
    _ values: [Element],
    columns: Int,
    pitchBytes: Int? = nil,
    rows: Int? = nil,
    _ body: (RowsView) throws(Failure) -> Result
) throws(Failure) -> Result {
    try values.withUnsafeBufferPointer { (buffer) throws(Failure) -> Result in
        try body(
            RowsView(
                MatrixView(
                    baseAddress: buffer.baseAddress!,
                    rows: rows ?? buffer.count / columns,
                    columns: columns,
                    rowStrideBytes: pitchBytes
                )
            )
        )
    }
}

/// A two-dimensional bf16 index holding `[1, 0]`, `[0, 1]` and `[1, 1]` under keys 1, 2 and 3.
private func triangle(metric: DenseMetric = .euclidean) throws -> DenseIndex {
    let index = try DenseIndex.initRam(DenseOptions(dimensions: 2, metric: metric))
    try index.reserve(3)
    try index.insert(keys: [1, 2, 3], vectors: try rows([Float](arrayLiteral: 1, 0, 0, 1, 1, 1), columns: 2))
    return index
}

/// The two-row inner-product index over the identity basis the linear-algebra verbs score against.
private func basis() throws -> DenseIndex {
    let index = try DenseIndex.initRam(DenseOptions(dimensions: 2, metric: .inner))
    try index.reserve(2)
    try index.insert(keys: [0, 1], vectors: try rows([Float](arrayLiteral: 1, 0, 0, 1), columns: 2))
    return index
}

/// Whether two scores agree to a thousandth of the larger magnitude, or of one when both are smaller.
private func nearly(_ left: Double, _ right: Double) -> Bool {
    abs(left - right) <= 1e-3 * max(abs(left), abs(right), 1)
}

/// Counts reports from the executor's threads, which never call a handler on two at once.
private final class Reports: @unchecked Sendable {
    var count = 0
    var completed = 0
}

/// What a caller's closure throws to stop a call.
private struct Refusal: Swift.Error, Equatable {}

@Suite("Keys")
struct KeyTests {

    @Test func byteOrder() throws {
        #expect(Key(42).description == "00000000-0000-0000-0000-00000000002a")
        #expect(Key.nilKey.description == "00000000-0000-0000-0000-000000000000")
        #expect(Key(42).byteArray == Array(repeating: 0, count: 15) + [42])
        #expect(Key(UInt64.max).description == "00000000-0000-0000-ffff-ffffffffffff")

        #expect(Key(42).integerIdentity == 42)
        #expect(Key.nilKey.integerIdentity == 0)
        #expect(try Key(bytes: [1] + Array(repeating: 0, count: 15)).integerIdentity == nil)

        // A literal is the same key as the constructed one, and equality agrees with hashing.
        let literal: Key = 42
        #expect(literal == Key(42))
        #expect(Set([literal, Key(42), Key(43)]).count == 2)
        #expect(Key(42) < Key(43))
    }

    /// Every rendering reads back as the key that produced it, whatever the identity's magnitude.
    @Test(arguments: [0, 1, 42, 1 << 32, UInt64.max] as [UInt64])
    func renderingRoundTrips(value: UInt64) throws {
        let key = Key(value)
        #expect(try Key(parsing: key.description) == key)
    }

    @Test func parsingRefusesMalformedText() throws {
        #expect(try Key(parsing: "0000000000000000000000000000002a") == Key(42))
        #expect(failureStatus(of: try Key(parsing: "")) == .malformed)
        #expect(failureStatus(of: try Key(parsing: "00000000-0000-0000-0000-00000000002")) == .malformed)
        #expect(failureStatus(of: try Key(parsing: "00000000-0000-0000-0000-00000000002ab")) == .malformed)
        #expect(failureStatus(of: try Key(parsing: "0000000g-0000-0000-0000-00000000002a")) == .malformed)
        #expect(failureStatus(of: try Key(bytes: [1, 2, 3])) == .malformed)
    }

    /// A key crosses a JSON document as its rendering, so a stored document reads back the same
    /// string whichever SDK wrote it.
    @Test func codesAsItsRendering() throws {
        let keys = [Key(42), Key.nilKey, try Key(bytes: [UInt8](1 ... 16))]
        let encoded = try JSONEncoder().encode(keys)
        #expect(
            try JSONDecoder().decode([String].self, from: encoded) == [
                "00000000-0000-0000-0000-00000000002a",
                "00000000-0000-0000-0000-000000000000",
                "01020304-0506-0708-090a-0b0c0d0e0f10",
            ]
        )
        #expect(try JSONDecoder().decode([Key].self, from: encoded) == keys)
        #expect(throws: (any Swift.Error).self) {
            try JSONDecoder().decode([Key].self, from: Data(#"["not-a-key"]"#.utf8))
        }
    }
}

@Suite("Vocabulary")
struct VocabularyTests {

    /// Every closed set spells itself through the library's own tables, and reads only those
    /// spellings back.
    @Test func namesAreTheLibrarys() {
        #expect(USearch.Status.invalidArgument.description == "invalid-argument")
        #expect(USearch.Status.deferred.description == "deferred")
        #expect(USearch.Status(name: "read-only") == .readOnly)
        #expect(USearch.Status(name: "not-a-status") == nil)
        #expect(Scalar.bf16.description == "bf16")
        #expect(Scalar(name: "u1x8") == .u1x8)
        #expect(Scalar(name: "i32") == .i32)
        #expect(Scalar(name: "f128") == nil)
        #expect(DenseMetric.inner.description == "inner")
        #expect(DenseMetric(name: "jaccard") == .jaccard)
        #expect(Mood.spinning.description == "spinning")
        #expect(Mood(name: "parked") == .parked)
        #expect(Mood(name: "asleep") == nil)
        #expect(Residency.vram.description == "vram")
        #expect(Residency(name: "nvme") == .nvme)
        #expect(Residency(name: "disk") == nil)
        #expect(Traversal.scan.description == "scan")
        #expect(Traversal(name: "walk") == .walk)
        #expect(Traversal(name: "sweep") == nil)
    }

    @Test func errorsRenderTheirStatusAndMessage() {
        do {
            _ = try Key(parsing: "")
            Issue.record("an empty rendering parsed")
        }
        catch {
            #expect(error.description == "usearch: malformed: the key is not thirty-two hexadecimal digits")
        }
    }

    @Test func executorFactoriesTakeOnlyTheirOwnPlacement() throws {
        let executor = try Executor.forRam(memoryDomains: [], threads: 2, deferredLimit: 4)
        try executor.drain(mood: .spinning)
        // The streams cross behind a pointer that carries no count, so the binding counts them.
        #expect(failureStatus(of: try Executor.forVram(devices: [0, 1], streams: [nil])) == .shapeMismatch)
        // An arena lives where its executor computes, and without one in host pages.
        #expect(try Arena(on: executor).residency == .ram)
        #expect(try Arena().residency == .ram)
    }

    @Test func theLibraryNamesItselfAndItsCapabilities() {
        #expect(version() == "3.0.0")
        #expect(!capabilitiesCompiled().isEmpty)
        #expect(!capabilitiesAvailable().isEmpty)
    }
}

@Suite("Marshalling")
struct MarshallingTests {

    /// A zeroed width is legal: the geometry defers to a later load. Every verb that needs the
    /// width refuses until it freezes, rather than deriving a zero-byte row.
    @Test func unfrozenGeometryIsRefused() throws {
        let index = try DenseIndex.initRam(DenseOptions())
        #expect(try index.describe().metadata.dimensions == 0)
        let vector = try rows([Float](arrayLiteral: 0.1, 0.2, 0.3, 0.4), columns: 4)
        #expect(failureStatus(of: try index.insert(keys: [1], vectors: vector)) == .uninitialized)
        #expect(failureStatus(of: try index.search(vector, wanted: 1, arena: try Arena())) == .uninitialized)
    }

    @Test func strideArithmetic() throws {
        let index = try DenseIndex.initRam(DenseOptions(dimensions: 4, metric: .euclidean))
        let executor = try executor()
        try index.reserve(4)

        // Three rows of four dimensions inside a buffer pitched at six floats per row - the padded
        // layout a memory-mapped dataset arrives in. Only the first four of every six are read.
        let padded: [Float] = [
            1, 0, 0, 0, 9, 9,
            0, 1, 0, 0, 9, 9,
            0, 0, 1, 0, 9, 9,
        ]
        let pitch = 6 * MemoryLayout<Float>.stride
        try withMatrix(padded, columns: 4, pitchBytes: pitch, rows: 3) { vectors in
            #expect(vectors.rows == 3)
            #expect(vectors.stride == pitch)
            try index.on(executor).insert(keys: [1, 2, 3], vectors: vectors)
        }
        #expect(try index.describe().size == 3)

        // A pitch too narrow to hold its own row is a layout the library refuses.
        #expect(
            try withMatrix(padded, columns: 4, pitchBytes: 8, rows: 3) { vectors in
                failureStatus(of: try index.insert(keys: [4, 5, 6], vectors: vectors))
            } == .unsupportedLayout
        )

        // The row width crosses as the view's `columns`, so the library sees the disagreement.
        #expect(
            try withMatrix(padded, columns: 6) { vectors in
                failureStatus(of: try index.insert(keys: [4, 5, 6], vectors: vectors))
            } == .shapeMismatch
        )

        // The keys cross with no count of their own, so the binding counts them against the rows.
        #expect(
            try withMatrix(padded, columns: 6) { vectors in
                failureStatus(of: try index.insert(keys: [4, 5], vectors: vectors))
            } == .shapeMismatch
        )

        // A row reads back packed at its logical width, without the padding it arrived in.
        let exported = try index.on(executor).get(key: 2, as: Float.self, arena: try Arena(on: executor))
        #expect(Array(exported.joined()) == [0, 1, 0, 0])
    }

    /// A sub-byte kind counts logical dimensions, so eight of them are one stored byte - the one
    /// place where the row width and the row extent part ways.
    @Test func packedBitsCrossAsOneBytePerEightDimensions() throws {
        let index = try DenseIndex.initRam(DenseOptions(dimensions: 8, metric: .hamming, quantization: .u1x8))
        try index.reserve(1)
        let packed = [U1x8(bitPattern: 0b0000_1111)]
        let vectors = try rows(packed, columns: 1)
        #expect(vectors.columns == 8)
        #expect(vectors.stride == 1)
        try index.insert(keys: [1], vectors: vectors)
        let exported = try index.get(key: 1, as: U1x8.self, arena: try Arena())
        #expect(exported.count == 1)
        #expect(exported.columns == 8)
        #expect(exported.stride == 1)
        #expect(Array(exported[0]) == packed)
    }
}

@Suite("Results")
struct ResultTests {

    @Test func rowsStopAtTheCellsWritten() throws {
        let index = try DenseIndex.initRam(
            DenseOptions(dimensions: 2, metric: .euclidean, valuesPerKey: 3)
        )
        let arena = try Arena()
        try index.reserve(3)

        // Three entries share one key, and an export returns every one of them in insertion order.
        try index.insert(keys: [7, 7, 7], vectors: try rows([Float](arrayLiteral: 1, 0, 2, 0, 3, 0), columns: 2))
        #expect(try index.count(key: 7) == 3)
        #expect(try index.contains(key: 7))
        #expect(try !index.contains(key: 8))

        #expect(Array(try index.get(key: 7, as: Float.self, arena: arena).joined()) == [1, 0, 2, 0, 3, 0])
        #expect(try index.get(key: 8, as: Float.self, arena: arena).isEmpty)

        // A search asking for more than the index holds reports what it wrote and nothing past it,
        // so the untouched cells never reach a caller.
        let stats = DenseTraversalStats()
        let matches = try index.search(
            try rows([Float](arrayLiteral: 1, 0), columns: 2),
            wanted: 10,
            arena: arena,
            config: DenseSearchConfig(stats: stats)
        )
        #expect(matches.count == 1)
        #expect((matches[0].count ... 10).contains(matches.columns))
        #expect(stats.computedDistances > 0)
        #expect(matches[0].map(\.key) == [7, 7, 7])
        let distances = matches[0].map(\.distance)
        #expect(distances.count == 3 && zip(distances, [0, 1, 2]).allSatisfy { nearly($0, $1) })

        // Wanting nothing lands only the counts, with no keys or scores behind them.
        let none = try index.search(try rows([Float](arrayLiteral: 1, 0), columns: 2), wanted: 0, arena: arena)
        #expect(none.count == 1 && none[0].isEmpty)
    }

    /// The arena's next call rewinds it, so a view an earlier call answered traps on its next read
    /// rather than reading what the later call wrote there.
    #if os(macOS) || os(Linux)
        @Test func aLaterCallEndsTheEarlierViews() async {
            let exit = await #expect(processExitsWith: .failure, observing: [\.standardErrorContent]) {
                let index = try DenseIndex.initRam(DenseOptions(dimensions: 2))
                try index.reserve(2)
                let vectors = RowsView(try Tensor.fromArray([Float](arrayLiteral: 1, 2, 3, 4), rows: 2, columns: 2))
                try index.insert(keys: [1, 2], vectors: vectors)
                let arena = try Arena()
                let first = try index.get(key: 1, as: Float.self, arena: arena)
                _ = try index.get(key: 2, as: Float.self, arena: arena)
                _ = first[0]
            }
            let reported = exit.map { String(decoding: $0.standardErrorContent, as: UTF8.self) } ?? ""
            #expect(reported.contains("the arena was rewound since this view was written: uninitialized"))
        }
    #endif

    @Test func filteredSearchTrampoline() throws {
        let index = try triangle()
        let query = try rows([Float](arrayLiteral: 1, 0), columns: 2)

        let arena = try Arena()

        let admitted = DenseSearchConfig(predicate: { $0 == Key(2) })
        #expect(try index.search(query, wanted: 3, arena: arena, config: admitted)[0].map(\.key) == [2])

        let rejected = DenseSearchConfig(predicate: { _ in false })
        #expect(try index.search(query, wanted: 3, arena: arena, config: rejected)[0].isEmpty)
    }

    /// A scan scores every row exactly, so each row queried against its own index finds itself first.
    @Test func scanFindsEveryRowItself() throws {
        let index = try triangle(metric: .angular)
        let queries = try rows([Float](arrayLiteral: 1, 0, 0, 1, 1, 1), columns: 2)
        let scan = DenseSearchConfig(traversal: .scan)
        let matches = try index.search(queries, wanted: 3, arena: try Arena(), config: scan)
        for (query, key) in [Key(1), Key(2), Key(3)].enumerated() {
            #expect(matches[query].first?.key == key)
        }
    }

    /// A handler hears every unit of work, and a stop it asks for comes back as `cancelled`.
    @Test func progressReportsAndStops() throws {
        let index = try triangle()
        let queries = try rows([Float](arrayLiteral: 1, 0, 0, 1), columns: 2)
        let reports = Reports()
        let arena = try Arena()
        _ = try index.search(
            queries,
            wanted: 1,
            arena: arena,
            config: DenseSearchConfig(progress: { completed, _ in
                reports.count += 1
                reports.completed = completed
                return .continue
            })
        )
        #expect(reports.count >= 1)
        #expect(reports.completed == 2)
        let stopped = DenseSearchConfig(progress: { _, _ in .cancel })
        #expect(failureStatus(of: try index.search(queries, wanted: 1, arena: arena, config: stopped)) == .cancelled)
    }

    /// An error a predicate or handler throws never unwinds into C, and is what the call throws
    /// once final, blocking or deferred.
    @Test func closureErrorsArriveOnceFinal() throws {
        let index = try triangle()
        let query = try rows([Float](arrayLiteral: 1, 0), columns: 2)
        let arena = try Arena()
        let filtered = DenseSearchConfig(predicate: { _ in throw Refusal() })
        let thrown = failure(of: try index.search(query, wanted: 3, arena: arena, config: filtered))
        #expect(thrown?.cause as? Refusal == Refusal())

        let reporting = DenseSearchConfig(progress: { _, _ in throw Refusal() })
        let call = try index.deferred(try executor()).search(query, wanted: 3, arena: arena, config: reporting)
        #expect(failure(of: try call.wait())?.cause as? Refusal == Refusal())
        #expect(call.status == .cancelled)
    }
}

@Suite("Streaming")
struct StreamingTests {

    @Test func cursorWalksASnapshot() throws {
        let index = try triangle()
        let executor = try executor()
        let arena = try Arena(on: executor)

        let cursor = try index.on(executor).cursor()
        var walked: [Key] = []
        while true {
            let batch = try cursor.nextLinks(wanted: 2, arena: arena)
            guard !batch.keys.isEmpty else { break }
            #expect(batch.links.count == batch.keys.count)
            walked.append(contentsOf: batch.keys)
        }
        #expect(walked == [1, 2, 3])

        // A key-only walk needs no links, and reads the same snapshot from a fresh cursor.
        #expect(Array(try index.cursor().next(wanted: 2, arena: arena)) == [1, 2])
    }

    /// The engine declares the stream before implementing it, so an `unsupported` open is the one
    /// failure tolerated; an open stream pulls in batches and yields every match nearest first.
    @Test func searchStreamYieldsMatchesLazily() throws {
        let index = try triangle()
        let executor = try executor()
        let arena = try Arena(on: executor)
        let query = try rows([Float](arrayLiteral: 1, 0), columns: 2)
        do {
            let stream = try index.on(executor).searchStream(query)
            var streamed: [Key] = []
            while true {
                let pulled = try stream.next(wanted: 2, arena: arena)
                guard !pulled[0].isEmpty else { break }
                streamed.append(contentsOf: pulled[0].map(\.key))
            }
            #expect(streamed == [1, 3, 2])
        }
        catch {
            #expect(error.status == .unsupported)
        }

        // A stream walks one query row, and refuses rows carrying more before asking.
        let pair = try rows([Float](arrayLiteral: 1, 0, 0, 1), columns: 2)
        #expect(failureStatus(of: try index.searchStream(pair)) == .shapeMismatch)
    }
}

@Suite("Lifetime")
struct LifetimeTests {

    /// The index frees its handle when the last reference drops, so a deallocation that never
    /// happens is the only visible shape of a leak.
    @Test func handleReleasesWithItsLastReference() throws {
        weak var observed: DenseIndex?
        var index: DenseIndex? = try DenseIndex.initRam(DenseOptions(dimensions: 2))
        observed = index
        #expect(observed != nil)
        index = nil
        #expect(observed == nil)
    }

    /// A cursor counts as a reference to the index it walks, since its own handle would dangle past
    /// the index's - so the snapshot outlives the caller's last mention of the index.
    @Test func cursorHoldsTheIndexItWalks() throws {
        weak var observed: DenseIndex?
        var index: DenseIndex? = try DenseIndex.initRam(DenseOptions(dimensions: 2))
        try index?.reserve(1)
        observed = index
        var cursor: DenseCursor? = try #require(index).cursor()
        index = nil
        #expect(cursor != nil)
        #expect(observed != nil, "an open cursor holds the index it walks")
        cursor = nil
        #expect(observed == nil)
    }

    // Raw spans over a buffer arrive with the macOS 26 runtime.
    @available(macOS 26.0, iOS 26.0, tvOS 26.0, watchOS 26.0, visionOS 26.0, *)
    @Test func serializationRoundTrip() throws {
        let index = try triangle()
        let executor = try executor()

        let monolith = try index.on(executor).saveBuffer(arena: try Arena(on: executor))
        let restored = try monolith.withUnsafeBytes { (bytes) throws(USearchError) in
            try DenseIndex.loadRamBuffer(bytes.bytes, config: DenseLoadRamBufferConfig(executor: executor))
        }
        #expect(try restored.describe().size == 3)
        #expect(try restored.describe().metadata.dimensions == 2)
        #expect(try restored.count(key: 1) == 1)

        let described = try monolith.withUnsafeBytes { (bytes) throws(USearchError) in
            try DenseIndex.metadataReadBuffer(bytes.bytes)
        }
        #expect(described.dimensions == 2)
        #expect(described.metric == .euclidean)
        #expect(described.quantization == .bf16)
    }

    /// A result holds its arena, so it reads fine after the index and the caller's arena are gone.
    @Test func resultOutlivesItsIndexAndArena() throws {
        var index: DenseIndex? = try DenseIndex.initRam(DenseOptions(dimensions: 2))
        try index?.reserve(1)
        try index?.insert(keys: [1], vectors: try rows([Float](arrayLiteral: 3, 4), columns: 2))
        let exported = try #require(index).get(key: 1, as: Float.self, arena: try Arena())
        index = nil
        #expect(Array(exported[0]) == [3, 4])
    }

    /// A failed call leaves the arena serving the next one.
    @Test func arenaSurvivesFailures() throws {
        let index = try triangle()
        let arena = try Arena()
        // A product needs logits, so a euclidean index refuses it after the arena was lent.
        let left = try rows([Float](arrayLiteral: 1, 0), columns: 2)
        #expect(failureStatus(of: try index.multiply(left, arena: arena)) == .configMismatch)
        let first = try index.get(key: 1, as: Float.self, arena: arena)
        #expect(Array(first[0]) == [1, 0])
        #expect(Array(try index.get(key: 3, as: Float.self, arena: arena)[0]) == [1, 1])
    }

    /// A viewed checkpoint is served in place, so every mutating verb on it is refused.
    @Test func viewedCheckpointIsReadOnly() throws {
        let index = try triangle()
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("usearch-swift-\(UUID().uuidString).safegraphs").path
        defer { try? FileManager.default.removeItem(atPath: path) }
        try index.save(path: path)

        // The engine declares views before it serves them, so only the refusal is checked there.
        guard let viewed = try? DenseIndex.viewRam(path) else {
            #expect(failureStatus(of: try DenseIndex.viewRam(path)) == .unsupported)
            return
        }
        #expect(try viewed.count(key: 1) == 1)
        #expect(try DenseIndex.metadataRead(path).dimensions == 2)
        #expect(try DenseIndex.loadRam(path).describe().size == 3)
        let vector = try rows([Float](arrayLiteral: 1, 0), columns: 2)
        #expect(failureStatus(of: try viewed.insert(keys: [4], vectors: vector)) == .readOnly)
    }

    /// An absorb consumes every donor it takes in, which then refuses every verb rather than reaching
    /// a freed handle, and one stopped early hands the rest back.
    @Test func absorbConsumesItsDonors() throws {
        let index = try triangle()
        func donor(_ key: Key) throws -> DenseIndex {
            let donor = try DenseIndex.initRam(DenseOptions(dimensions: 2, metric: .euclidean))
            try donor.reserve(1)
            try donor.insert(keys: [key], vectors: try rows([Float](arrayLiteral: 2, 2), columns: 2))
            return donor
        }
        let (first, second) = (try donor(4), try donor(5))
        #expect(failureStatus(of: try index.absorb([first, first])) == .invalidArgument)
        #expect(failureStatus(of: try index.absorb([index])) == .configMismatch)

        let stopped = DenseAbsorbConfig(progress: { _, _ in .cancel })
        #expect(failureStatus(of: try index.absorb([first, second], config: stopped)) == .cancelled)
        #expect(try index.count(key: 4) == 1)
        #expect(failureStatus(of: try first.describe()) == .uninitialized)
        #expect(try second.count(key: 5) == 1)

        try index.absorb([second])
        #expect(try index.describe().size == 5)
        #expect(failureStatus(of: try second.describe()) == .uninitialized)
        #expect(failureStatus(of: try index.absorb([second])) == .uninitialized)
    }
}

@Suite("Deferred")
struct DeferredTests {

    /// A deferred call answers what the blocking one does, once waited on, and answers it again
    /// every time it is asked.
    @Test func deferredSearchMatchesTheBlockingOne() throws {
        let index = try triangle()
        let executor = try executor()
        let queries = try rows([Float](arrayLiteral: 1, 0, 0, 1), columns: 2)
        let blocking = try index.on(executor).search(queries, wanted: 3, arena: try Arena(on: executor))

        let call = try index.deferred(executor).search(queries, wanted: 3, arena: try Arena(on: executor))
        #expect(call.status == .deferred)
        let deferred = try call.wait()
        #expect(call.status == .success)
        #expect(deferred.map { $0.map(\.key) } == blocking.map { $0.map(\.key) })
        #expect(try call.wait().map { $0.map(\.distance) } == blocking.map { $0.map(\.distance) })
    }

    /// A blocking call on the same executor runs the deferred calls submitted before it first.
    @Test func aBlockingCallRunsTheDeferredOnesBeforeIt() throws {
        let index = try triangle()
        let executor = try executor()
        let call = try index.deferred(executor).count(key: 3)
        #expect(call.status == .deferred)
        #expect(try index.on(executor).count(key: 1) == 1)
        #expect(call.status == .success)
        #expect(try call.wait() == 1)
    }

    /// Awaiting suspends rather than blocks, and reads the views back in the awaiting actor's isolation.
    @MainActor @Test func awaitingSuspendsOnTheConcurrencyRuntime() async throws {
        let index = try triangle()
        let executor = try executor()
        let query = try rows([Float](arrayLiteral: 1, 0), columns: 2)
        let call = try index.deferred(executor).search(query, wanted: 1, arena: try Arena(on: executor))
        #expect(call.status == .deferred)
        let matches = try await call.value
        #expect(matches[0].map(\.key) == [1])
    }

    /// A failure this binding catches throws at submission; one the library catches arrives
    /// through the call's status and wait.
    @Test func failuresArriveWhereTheyAreCaught() throws {
        let index = try triangle()
        let executor = try executor()
        let vectors = try rows([Float](arrayLiteral: 1, 0), columns: 2)
        #expect(
            failureStatus(of: try index.deferred(executor).insert(keys: [4, 5], vectors: vectors)) == .shapeMismatch
        )

        let call = try index.deferred(executor).multiply(vectors, arena: try Arena(on: executor))
        #expect(failureStatus(of: try call.wait()) == .configMismatch)
        #expect(call.status == .configMismatch)
    }

    /// A submission finding the deferred table full finishes at once, without waiting for a slot.
    @Test func aFullDeferredTableRefusesAtOnce() throws {
        let index = try triangle()
        let executor = try Executor.forRam(deferredLimit: 1)
        let held = try index.deferred(executor).count(key: 1)
        let refused = try index.deferred(executor).count(key: 2)
        #expect(held.status == .deferred)
        #expect(refused.status == .contended)
        #expect(failureStatus(of: try refused.wait()) == .contended)
        #expect(try held.wait() == 1)
    }

    /// A deferred call borrows its arena as a blocking one does, so a call on the same arena
    /// meanwhile is refused without ending the holder's views, and the arena serves again once the
    /// call is final.
    @Test func aDeferredCallBorrowsItsArena() throws {
        let index = try triangle()
        let executor = try executor()
        let arena = try Arena(on: executor)
        let gate = DispatchSemaphore(value: 0)
        let held = DenseSearchConfig(progress: { _, _ in
            gate.wait()
            return .continue
        })
        let call = try index.deferred(executor).search(
            try rows([Float](arrayLiteral: 1, 0), columns: 2),
            wanted: 1,
            arena: arena,
            config: held
        )
        #expect(failureStatus(of: try index.get(key: 2, as: Float.self, arena: arena)) == .contended)
        #expect(try index.deferred(executor).get(key: 2, as: Float.self, arena: arena).status == .contended)
        gate.signal()
        #expect(try call.wait()[0].map(\.key) == [1])
        #expect(Array(try index.get(key: 2, as: Float.self, arena: arena)[0]) == [0, 1])
    }

    /// The rows a deferred insertion reads stay alive with the call, whatever the caller drops.
    @Test func aDeferredCallKeepsItsInputs() throws {
        let index = try DenseIndex.initRam(DenseOptions(dimensions: 2, metric: .euclidean))
        let executor = try executor()
        try index.reserve(2)
        let call = try index.deferred(executor).insert(
            keys: [8, 9],
            vectors: try rows([Float](arrayLiteral: 5, 6, 7, 8), columns: 2)
        )
        try call.wait()
        #expect(Array(try index.get(key: 9, as: Float.self, arena: try Arena()).joined()) == [7, 8])
    }
}

@Suite("Linear Algebra")
struct LinearAlgebraTests {

    /// A wanted bound keeps the first pairs each walk scored, and `required` still counts them all.
    @Test func multiplyReportsEveryScoredPair() throws {
        let index = try basis()
        let left = try rows([Float](arrayLiteral: 1, 0, 0, 1), columns: 2)
        let arena = try Arena()
        let every = try index.multiply(left, arena: arena)
        #expect(every.count == 2)
        #expect(every.map(\.count) == [2, 2])
        #expect(Set(every[1].map(\.key)) == [Key(0), Key(1)])

        let bounded = try index.multiply(left, arena: arena, config: DenseMultiplyConfig(wanted: 1))
        #expect(bounded.map(\.count) == [1, 1], "the bound cuts each row")
        #expect(bounded.required == 4, "though every pair was scored")
    }

    @Test func maxSimScoresOverDocumentsRatherThanRows() throws {
        let index = try basis()
        // Two rows per query, so one query of two tokens scores against both indexed documents,
        // written as one row.
        let tokens = try rows([Float](arrayLiteral: 1, 0, 0, 1), columns: 2)
        let arena = try Arena()
        let documents = try index.maxsim(tokens, rowsPerQuery: 2, wanted: 2, arena: arena)
        #expect(documents.count == 1)
        #expect(Set(documents[0].map(\.key)) == [Key(0), Key(1)])
        let bounded = documents[0].count
        let every = try index.maxsim(tokens, rowsPerQuery: 2, wanted: 0, arena: arena)
        #expect(every[0].count >= bounded, "a zero bound keeps every scored document")

        // A token count the rows do not divide by is a shape the binding refuses on its own.
        #expect(failureStatus(of: try index.maxsim(tokens, rowsPerQuery: 3, wanted: 2, arena: arena)) == .shapeMismatch)
        #expect(failureStatus(of: try index.maxsim(tokens, rowsPerQuery: 0, wanted: 2, arena: arena)) == .shapeMismatch)
    }

    /// A graph this small walks every key, so attention is the exact softmax over all of them, with
    /// one head per walk and with two sharing it; a key the lookup leaves without a row refuses.
    @Test func attendIsExactAttentionOverEveryKey() throws {
        let executor = try executor()
        let index = try DenseIndex.initRam(DenseOptions(dimensions: 2, metric: .inner))
        try index.reserve(4)
        // Every key and value is one a bf16 holds exactly, so the stored rows are these.
        let keys: [Float] = [1, 0, 0.5, 0.75, 0, 1, -0.75, 0.5]
        try index.insert(keys: [0, 1, 2, 3], vectors: try rows(keys, columns: 2))
        let queries: [Float] = [1, 0, 0, 1]
        let floats: [Float] = [10, 20, 30, 40, 50, 60, 70, 80]
        let values = floats.map(BFloat16.init(float:))
        func logit(_ query: Int, _ key: Int) -> Double {
            let product = queries[query * 2] * keys[key * 2] + queries[query * 2 + 1] * keys[key * 2 + 1]
            return Double(product) / 2.0.squareRoot()
        }
        // Keys at or past `resolvable` get no value row.
        func attend<Value: ScalarValue>(
            headCount: Int,
            values: [Value],
            resolvable: Int = 4
        ) throws -> ([Float], [DenseSoftmaxState]) {
            var carry = [DenseSoftmaxState](repeating: DenseSoftmaxState(), count: 2)
            let lookup = ValueLookup(values: try rows(values, columns: 2)) { key in
                key.integerIdentity.map(Int.init).flatMap { $0 < resolvable ? $0 : nil }
            }
            let queryRows = try rows(queries, columns: 2)
            let context = try carry.withUnsafeMutableBufferPointer { carry in
                try index.on(executor).attend(
                    queryRows,
                    lookup: lookup,
                    as: Float.self,
                    arena: try Arena(on: executor),
                    config: DenseAttendConfig(headCount: headCount, carry: carry)
                )
            }
            return (Array(context.joined()), carry)
        }

        for headCount in [1, 2] {
            let (context, carry) = try attend(headCount: headCount, values: values)
            for query in 0 ..< 2 {
                let logits = (0 ..< 4).map { logit(query, $0) }
                let maximum = logits.max()!
                let weights = logits.map { exp($0 - maximum) }
                let partition = weights.reduce(0, +)
                for column in 0 ..< 2 {
                    let mixed = weights.indices.map { Double(floats[$0 * 2 + column]) * weights[$0] }.reduce(0, +)
                    #expect(nearly(Double(context[query * 2 + column]), mixed / partition))
                }
                #expect(nearly(Double(carry[query].maximumScore), maximum))
                #expect(nearly(Double(carry[query].partition), partition))
            }
        }
        #expect(failureStatus { _ = try attend(headCount: 2, values: values, resolvable: 3) } == .shapeMismatch)
        // A lookup that throws leaves its key without a row, and the call throws what it threw.
        let refusing = ValueLookup(values: try rows(values, columns: 2)) { _ in throw Refusal() }
        let queryRows = try rows(queries, columns: 2)
        let arena = try Arena(on: executor)
        let refused = failure(
            of: try index.on(executor).attend(queryRows, lookup: refusing, as: Float.self, arena: arena)
        )
        #expect(refused?.cause as? Refusal == Refusal())
        // Values of another kind than the index stores answer no row at all.
        #expect(failureStatus { _ = try attend(headCount: 1, values: floats) } == .shapeMismatch)
    }

    @Test func refinementSettlesAndHealthAuditsEveryEntry() throws {
        let index = try triangle()
        let executor = try executor()
        #expect(try index.refine(config: DenseRefineConfig(expansionFactor: 64)) == 0)
        let reachability = try index.on(executor).health()
        #expect(reachability.audited == 3)
        #expect(reachability.reachable == 3)
        #expect(try index.health() == reachability)
    }
}
