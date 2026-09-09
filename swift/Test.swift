//
//  Test.swift
//  USearch
//
//  The boundary suite: what the binding itself decides - key byte order, options marshalling
//  through the residency union, matrix lowering, handle lifetime, truncation reporting, the
//  streaming handles, and the predicate trampoline. That vector search returns near neighbors is
//  the engine's contract and is tested where the engine lives.
//
//  Suites run in parallel and every test gets a fresh instance, so nothing here is shared: each
//  test spawns the one executor it drives, and the helpers below are free functions rather than
//  members for the same reason.
//

import Foundation
import Testing

import NumKong

import USearch

/// One executor per test, since an executor admits a single driver at a time.
private func executor() throws(Error) -> Executor { try Executor.forRam() }

/// The status a call failed with, and nil when it succeeded. `Error` is the only error the SDK
/// throws, so the catch needs no cast.
private func failureStatus<Value>(of body: @autoclosure () throws(Error) -> Value) -> Status? {
    do {
        _ = try body()
        return nil
    }
    catch {
        return error.status
    }
}

/// Views an array as rows of the given width for the duration of `body`, which is what keeps the
/// borrow and the storage the same length.
private func withMatrix<Element: ScalarValue, Result, Failure: Swift.Error>(
    _ values: [Element],
    columns: Int,
    pitchBytes: Int? = nil,
    rows: Int? = nil,
    _ body: (MatrixView<Element>) throws(Failure) -> Result
) throws(Failure) -> Result {
    try values.withUnsafeBufferPointer { (buffer) throws(Failure) -> Result in
        try body(
            MatrixView(
                baseAddress: buffer.baseAddress!,
                rows: rows ?? buffer.count / columns,
                cols: columns,
                rowStrideBytes: pitchBytes
            )
        )
    }
}

/// The written-through twin of `withMatrix`, for the verbs that export rows.
private func withMutableMatrix<Element: ScalarValue, Result, Failure: Swift.Error>(
    _ values: inout [Element],
    columns: Int,
    pitchBytes: Int? = nil,
    rows: Int? = nil,
    _ body: (MatrixSpan<Element>) throws(Failure) -> Result
) throws(Failure) -> Result {
    try values.withUnsafeMutableBufferPointer { (buffer) throws(Failure) -> Result in
        try body(
            MatrixSpan(
                baseAddress: buffer.baseAddress!,
                rows: rows ?? buffer.count / columns,
                cols: columns,
                rowStrideBytes: pitchBytes
            )
        )
    }
}

/// The keys, distances and counts one ranked run writes through, allocated once and freed when the
/// test ends.
private final class MatchesStorage {
    let keys: UnsafeMutableBufferPointer<Key>
    let distances: UnsafeMutableBufferPointer<Distance>
    let counts: UnsafeMutableBufferPointer<Int>

    init(queries: Int, wanted: Int) {
        keys = .allocate(capacity: queries * wanted)
        distances = .allocate(capacity: queries * wanted)
        counts = .allocate(capacity: queries)
        keys.initialize(repeating: .nilKey)
        distances.initialize(repeating: 0)
        counts.initialize(repeating: 0)
    }

    deinit {
        keys.deallocate()
        distances.deallocate()
        counts.deallocate()
    }

    /// The span the ranking verbs write through.
    var span: MatchesSpan {
        get throws(Error) { try MatchesSpan(keys: keys, distances: distances, counts: counts) }
    }
}

/// The two-row inner-product index over the identity basis the linear-algebra verbs score against.
private func basis(executor: Executor) throws -> DenseIndex {
    let index = try DenseIndex(options: DenseOptions(dimensions: 2, metric: .inner, quantization: .f32))
    try index.reserve(2)
    let rows: [Float] = [1, 0, 0, 1]
    try withMatrix(rows, columns: 2) { matrix in
        _ = try index.add(executor: executor, keys: [0, 1], vectors: matrix)
    }
    return index
}

@Suite("Keys")
struct KeyTests {

    @Test func byteOrder() throws {
        #expect(MemoryLayout<Key>.size == 16)
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
        #expect(failureStatus(of: try Key(parsing: "")) == .malformedInput)
        #expect(failureStatus(of: try Key(parsing: "00000000-0000-0000-0000-00000000002")) != nil)
        #expect(failureStatus(of: try Key(parsing: "00000000-0000-0000-0000-00000000002ab")) != nil)
        #expect(failureStatus(of: try Key(parsing: "0000000g-0000-0000-0000-00000000002a")) != nil)
        #expect(failureStatus(of: try Key(bytes: [1, 2, 3])) != nil)
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

@Suite("Options")
struct OptionTests {

    /// A checkpoint records geometry and metric, never placement, so the arm survives the round
    /// trip and the list it names comes back empty for the caller to fill.
    @Test(arguments: [
        Residency.ram(memoryDomains: []),
        .ram(memoryDomains: [0, 1]),
        .vram(devices: []),
        .vram(devices: [0, 3]),
        .nvme(paths: []),
        .nvme(paths: ["/mnt/first", "/mnt/second"]),
    ])
    func roundTripThroughResidencyUnion(residency: Residency) throws {
        let options = DenseOptions(
            dimensions: 96,
            metric: .euclidean,
            quantization: .i8,
            connectivity: 24,
            valuesPerKey: 4,
            residency: residency
        )
        let reloaded = try options.reloaded()
        #expect(reloaded.dimensions == 96)
        #expect(reloaded.metric == .euclidean)
        #expect(reloaded.quantization == .i8)
        #expect(reloaded.connectivity == 24)
        #expect(reloaded.valuesPerKey == 4)
        switch (reloaded.residency, residency) {
        case (.ram(let memoryDomains), .ram): #expect(memoryDomains.isEmpty)
        case (.vram(let devices), .vram): #expect(devices.isEmpty)
        case (.nvme(let paths), .nvme): #expect(paths.isEmpty)
        default: Issue.record("the residency arm did not survive the round trip")
        }
    }

    @Test func executorFactoriesTakeOnlyTheirOwnPlacement() throws {
        let executor = try Executor.forRam(memoryDomains: [], threads: 2)
        try executor.drain()
    }
}

@Suite("Marshalling")
struct MarshallingTests {

    /// A zeroed width is legal: the geometry defers to a later load. Every verb that needs the
    /// width refuses until it freezes, rather than deriving a zero-byte row.
    @Test func unfrozenGeometryIsRefused() throws {
        let index = try DenseIndex(options: DenseOptions())
        let executor = try executor()
        #expect(try index.dimensions == 0)
        let vector: [Float] = [0.1, 0.2, 0.3, 0.4]
        try withMatrix(vector, columns: 4) { matrix in
            #expect(
                failureStatus(of: try index.add(executor: executor, keys: [1], vectors: matrix))
                    == .uninitialized
            )
            let storage = MatchesStorage(queries: 1, wanted: 1)
            #expect(
                failureStatus(of: try index.search(executor: executor, queries: matrix, into: storage.span))
                    != nil
            )
        }
    }

    @Test func strideArithmetic() throws {
        let index = try DenseIndex(
            options: DenseOptions(dimensions: 4, metric: .euclidean, quantization: .f32)
        )
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
        try withMatrix(padded, columns: 4, pitchBytes: pitch, rows: 3) { matrix in
            #expect(matrix.rows == 3)
            #expect(matrix.rowStrideBytes == pitch)
            let inserted = try index.add(executor: executor, keys: [1, 2, 3], vectors: matrix)
            #expect(inserted.rows == 3)
        }
        #expect(try index.size == 3)

        // A pitch too narrow to hold its own row is a layout the binding refuses before the call
        // crosses.
        #expect(
            try withMatrix(padded, columns: 4, pitchBytes: 8, rows: 3) { matrix in
                failureStatus(of: try index.add(executor: executor, keys: [4, 5, 6], vectors: matrix))
            } == .unsupportedLayout
        )

        // A row width the frozen geometry disagrees with is refused the same way.
        #expect(
            try withMatrix(padded, columns: 6) { matrix in
                failureStatus(of: try index.add(executor: executor, keys: [4, 5, 6], vectors: matrix))
            } == .shapeMismatch
        )

        // A row read back through a wider pitch lands where the stride says, leaving the padding
        // untouched.
        var exported = [Float](repeating: -1, count: 6)
        let entries = try withMutableMatrix(&exported, columns: 4, pitchBytes: pitch, rows: 1) { span in
            try index.get(key: 2, into: span)
        }
        #expect(entries == 1)
        #expect(Array(exported[0 ..< 4]) == [0, 1, 0, 0])
        #expect(Array(exported[4 ..< 6]) == [-1, -1])
    }

    /// A sub-byte kind counts logical dimensions, so eight of them are one stored byte - the one
    /// place where the row width and the row extent part ways.
    @Test func packedBitsCrossAsOneBytePerEightDimensions() throws {
        let index = try DenseIndex(options: DenseOptions(dimensions: 8, quantization: .u1x8))
        let executor = try executor()
        try index.reserve(1)
        let packed = [U1x8(bitPattern: 0b0000_1111)]
        try withMatrix(packed, columns: 1) { matrix in
            #expect(matrix.rows == 1)
            #expect(matrix.rowStrideBytes == 1)
            let added = try index.add(executor: executor, keys: [1], vectors: matrix)
            #expect(added.rows == 1)
        }
    }
}

@Suite("Results")
struct ResultTests {

    @Test func truncationReporting() throws {
        let index = try DenseIndex(
            options: DenseOptions(dimensions: 2, metric: .euclidean, quantization: .f32, valuesPerKey: 3)
        )
        let executor = try executor()
        try index.reserve(3)

        // Three entries share one key, so a two-row export truncates and reports the true total.
        let vectors: [Float] = [1, 0, 2, 0, 3, 0]
        try withMatrix(vectors, columns: 2) { matrix in
            _ = try index.add(executor: executor, keys: [7, 7, 7], vectors: matrix)
        }
        #expect(try index.count(key: 7) == 3)
        #expect(try index.contains(key: 7))
        #expect(try !index.contains(key: 8))

        var exported = [Float](repeating: -1, count: 4)
        let entries = try withMutableMatrix(&exported, columns: 2) { span in
            try index.get(key: 7, into: span)
        }
        #expect(entries == 3)
        #expect(exported == [1, 0, 2, 0])

        // A search asking for more than the index holds reports what it wrote and nothing past it,
        // so the untouched cells never reach a caller.
        let queries: [Float] = [1, 0]
        let storage = MatchesStorage(queries: 1, wanted: 10)
        let matches = try storage.span
        let stats = try withMatrix(queries, columns: 2) { matrix in
            try index.search(executor: executor, queries: matrix, into: matches)
        }
        #expect(matches.count == 1)
        #expect(matches.wanted == 10)
        #expect(stats.computedDistances > 0)

        // A row reads through the run rather than copying it out, and stops at the cells the engine
        // wrote.
        let row = matches[0]
        #expect(row.count == 3)
        #expect(!row.wasTruncated)
        let walked = row.map(\.key)
        #expect(walked.count == 3)
        #expect(walked.first == row.first?.key)
    }

    @Test func filteredSearchTrampoline() throws {
        let index = try DenseIndex(
            options: DenseOptions(dimensions: 2, metric: .euclidean, quantization: .f32)
        )
        let executor = try executor()
        try index.reserve(3)
        let vectors: [Float] = [1, 0, 0, 1, 1, 1]
        try withMatrix(vectors, columns: 2) { matrix in
            _ = try index.add(executor: executor, keys: [1, 2, 3], vectors: matrix)
        }

        let queries: [Float] = [1, 0]
        let storage = MatchesStorage(queries: 1, wanted: 3)
        let matches = try storage.span
        try withMatrix(queries, columns: 2) { matrix in
            _ = try index.search(
                executor: executor,
                queries: matrix,
                into: matches,
                admitting: { $0 == Key(2) }
            )
        }
        let admitted = matches[0]
        #expect(admitted.count == 1)
        #expect(admitted[0].key == Key(2))

        try withMatrix(queries, columns: 2) { matrix in
            _ = try index.search(
                executor: executor,
                queries: matrix,
                into: matches,
                admitting: { _ in false }
            )
        }
        #expect(matches[0].count == 0)
    }
}

@Suite("Streaming")
struct StreamingTests {

    @Test func cursorWalksASnapshot() throws {
        let index = try DenseIndex(
            options: DenseOptions(dimensions: 2, metric: .euclidean, quantization: .f32)
        )
        let executor = try executor()
        try index.reserve(3)
        let vectors: [Float] = [1, 0, 0, 1, 1, 1]
        try withMatrix(vectors, columns: 2) { matrix in
            _ = try index.add(executor: executor, keys: [1, 2, 3], vectors: matrix)
        }

        var keys = [Key](repeating: .nilKey, count: 2)
        var exported = [Float](repeating: -1, count: 4)
        let cursor = try index.cursor()
        var walked: [Key] = []
        while true {
            var written = 0
            do {
                var keySpan = keys.mutableSpan
                written = try withMutableMatrix(&exported, columns: 2) { (span) throws(Error) -> Int in
                    try cursor.next(into: &keySpan, vectors: span)
                }
            }
            guard written != 0 else { break }
            walked.append(contentsOf: keys.prefix(written))
        }
        #expect(Set(walked) == Set([Key(1), Key(2), Key(3)]))

        // A key-only walk needs no matrix, and reads the same snapshot from a fresh cursor. The
        // span borrows `keys`, so it is scoped to end that borrow before the array is read again.
        let keysOnly = try index.cursor()
        var keysWritten = 0
        do {
            var keySpan = keys.mutableSpan
            keysWritten = try keysOnly.next(into: &keySpan)
        }
        #expect(keysWritten > 0)
    }

    @Test func finderStreamsMatchesLazily() throws {
        let index = try DenseIndex(
            options: DenseOptions(dimensions: 2, metric: .euclidean, quantization: .f32)
        )
        let executor = try executor()
        try index.reserve(3)
        let vectors: [Float] = [1, 0, 0, 1, 1, 1]
        try withMatrix(vectors, columns: 2) { matrix in
            _ = try index.add(executor: executor, keys: [1, 2, 3], vectors: matrix)
        }

        let queries: [Float] = [1, 0]
        let streamed = try withMatrix(queries, columns: 2) { matrix -> [Key] in
            let finder = try index.find(executor: executor, query: matrix)
            defer { #expect(finder.failure == nil) }
            return finder.prefix(2).map(\.key)
        }
        #expect(streamed.count == 2, "a prefix pulls exactly what it asked for")

        // A finder streams one query row, and refuses a matrix carrying more.
        let pair: [Float] = [1, 0, 0, 1]
        #expect(
            try withMatrix(pair, columns: 2) { matrix in
                failureStatus(of: try index.find(executor: executor, query: matrix))
            } == .shapeMismatch
        )
    }
}

@Suite("Lifetime")
struct LifetimeTests {

    /// The index frees its handle when the last reference drops, so a deallocation that never
    /// happens is the only visible shape of a leak.
    @Test func handleReleasesWithItsLastReference() throws {
        weak var observed: DenseIndex?
        var index: DenseIndex? = try DenseIndex(options: DenseOptions(dimensions: 2, quantization: .f32))
        observed = index
        #expect(observed != nil)
        index = nil
        #expect(observed == nil)
    }

    /// A cursor counts as a reference to the index it walks, since its own handle would dangle past
    /// the index's - so the snapshot outlives the caller's last mention of the index.
    @Test func cursorHoldsTheIndexItWalks() throws {
        weak var observed: DenseIndex?
        var index: DenseIndex? = try DenseIndex(options: DenseOptions(dimensions: 2, quantization: .f32))
        try index?.reserve(1)
        observed = index
        var cursor: DenseCursor? = try #require(index).cursor()
        index = nil
        #expect(cursor != nil)
        #expect(observed != nil, "an open cursor holds the index it walks")
        cursor = nil
        #expect(observed == nil)
    }

    @Test func serializationRoundTrip() throws {
        let options = DenseOptions(dimensions: 2, metric: .euclidean, quantization: .f32)
        let index = try DenseIndex(options: options)
        let executor = try executor()
        try index.reserve(2)
        let vectors: [Float] = [1, 0, 0, 1]
        try withMatrix(vectors, columns: 2) { matrix in
            _ = try index.add(executor: executor, keys: [11, 12], vectors: matrix)
        }

        var buffer = [UInt8](repeating: 0, count: try index.storageRequirements)
        do {
            var typed = buffer.mutableSpan
            var span = typed.mutableBytes
            try index.save(executor: executor, buffer: &span)
        }

        let restored = try DenseIndex(options: DenseOptions())
        try restored.load(executor: executor, buffer: buffer.span.bytes)
        #expect(try restored.size == 2)
        #expect(try restored.dimensions == 2)
        #expect(try restored.count(key: 11) == 1)

        let described = try DenseIndex.metadata(buffer: buffer.span.bytes)
        #expect(described.dimensions == 2)
        #expect(described.metric == .euclidean)
        #expect(described.quantization == .f32)
    }
}

@Suite("Linear Algebra")
struct LinearAlgebraTests {

    /// The count is the true number of scored pairs, which the ABI lets exceed the room reserved
    /// for it - the one path that reaches `wasTruncated`.
    @Test func multiplyReportsEveryScoredPair() throws {
        let executor = try executor()
        let index = try basis(executor: executor)
        let left: [Float] = [1, 0]
        let storage = MatchesStorage(queries: 1, wanted: 1)
        let products = try storage.span
        try withMatrix(left, columns: 2) { matrix in
            _ = try index.multiply(executor: executor, left: matrix, into: products)
        }
        #expect(products.count == 1)
        #expect(products[0].count == 1, "the borrow stops at the room it was given")
        #expect(products[0].wasTruncated, "though both columns were scored")
    }

    @Test func maxSimScoresOverDocumentsRatherThanRows() throws {
        let executor = try executor()
        let index = try basis(executor: executor)
        // Two rows per query, so one query of two tokens scores against both indexed documents.
        let tokens: [Float] = [1, 0, 0, 1]
        let storage = MatchesStorage(queries: 1, wanted: 2)
        let scored = try storage.span
        try withMatrix(tokens, columns: 2) { matrix in
            _ = try index.maxsim(executor: executor, queries: matrix, rowsPerQuery: 2, into: scored)
        }
        #expect(scored.count == 1)
        #expect(scored[0].count == 2)

        // A token count the rows do not divide by is a shape the binding refuses on its own.
        #expect(
            try withMatrix(tokens, columns: 2) { matrix in
                failureStatus(
                    of: try index.maxsim(executor: executor, queries: matrix, rowsPerQuery: 3, into: scored)
                )
            } == .shapeMismatch
        )
    }

    /// One context row and one carry entry per query, each written through its own stride.
    @Test func attendWritesOneContextRowPerQuery() throws {
        let executor = try executor()
        let index = try basis(executor: executor)
        let queries: [Float] = [1, 0, 0, 1]
        let values: [Float] = [10, 20, 30, 40]
        var context = [Float](repeating: 0, count: 4)
        var carry = [SoftmaxState](repeating: SoftmaxState(), count: 2)

        try withMatrix(queries, columns: 2) { queryMatrix in
            try withMatrix(values, columns: 2) { valueMatrix in
                try withMutableMatrix(&context, columns: 2) { outputSpan in
                    try carry.withUnsafeMutableBufferPointer { (carryBuffer) throws(Error) -> Void in
                        _ = try index.attend(
                            executor: executor,
                            queries: queryMatrix,
                            values: valueMatrix,
                            into: outputSpan,
                            carry: carryBuffer
                        )
                    }
                }
            }
        }
        #expect(carry[0].partition > 0)
        #expect(carry[1].partition > 0)
    }
}

@Suite("Introspection")
struct IntrospectionTests {

    @Test func theLibraryNamesItselfAndItsCapabilities() {
        #expect(!usearchVersion().isEmpty)
        #expect(!usearchCapabilitiesCompiled().isEmpty)
        #expect(!usearchCapabilitiesAvailable().isEmpty)
    }
}
