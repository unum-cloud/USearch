//
//  Test.swift
//  USearch
//
//  The boundary suite: what the binding itself decides - key byte order, options marshalling
//  through the residency union, stride arithmetic, handle lifetime, truncation reporting, and
//  the predicate trampoline. That vector search returns near neighbors is the engine's contract
//  and is tested where the engine lives.
//

import USearch
import XCTest

final class Test: XCTestCase {

    // region: Keys

    func testKeyByteOrder() throws {
        XCTAssertEqual(MemoryLayout<USearchKey>.size, 16)
        XCTAssertEqual(USearchKey(42).description, "00000000-0000-0000-0000-00000000002a")
        XCTAssertEqual(USearchKey.nilKey.description, "00000000-0000-0000-0000-000000000000")
        XCTAssertEqual(USearchKey(42).byteArray, Array(repeating: 0, count: 15) + [42])
        XCTAssertEqual(USearchKey(UInt64.max).description, "00000000-0000-0000-ffff-ffffffffffff")

        XCTAssertEqual(USearchKey(42).integerIdentity, 42)
        XCTAssertEqual(USearchKey.nilKey.integerIdentity, 0)
        XCTAssertNil(try USearchKey(bytes: [1] + Array(repeating: 0, count: 15)).integerIdentity)

        // A literal is the same key as the constructed one, and equality agrees with hashing.
        let literal: USearchKey = 42
        XCTAssertEqual(literal, USearchKey(42))
        XCTAssertEqual(Set([literal, USearchKey(42), USearchKey(43)]).count, 2)
        XCTAssertTrue(USearchKey(42) < USearchKey(43))
    }

    func testKeyParsing() throws {
        for value: UInt64 in [0, 1, 42, 1 << 32, UInt64.max] {
            let key = USearchKey(value)
            XCTAssertEqual(try USearchKey(parsing: key.description), key)
        }
        XCTAssertEqual(try USearchKey(parsing: "0000000000000000000000000000002a"), USearchKey(42))
        XCTAssertThrowsError(try USearchKey(parsing: "")) { failure in
            XCTAssertEqual((failure as? USearchError)?.status, .malformedInput)
        }
        XCTAssertThrowsError(try USearchKey(parsing: "00000000-0000-0000-0000-00000000002"))
        XCTAssertThrowsError(try USearchKey(parsing: "00000000-0000-0000-0000-00000000002ab"))
        XCTAssertThrowsError(try USearchKey(parsing: "0000000g-0000-0000-0000-00000000002a"))
        XCTAssertThrowsError(try USearchKey(bytes: [1, 2, 3]))
    }

    // endregion: Keys

    // region: Options

    func testOptionsRoundTripThroughResidencyUnion() throws {
        let residencies: [USearchResidency] = [
            .ram(threads: 7, pages: .requireHugePages),
            .ram(threads: 0, pages: .basePages),
            .vram(device: -1, addressing: .unifiedAddressing),
            .vram(device: 3, addressing: .deviceOnly),
            .nvme(sectorBytes: 512, queueDepth: 64, io: .directIO),
            .nvme(sectorBytes: 0, queueDepth: 0, io: .pageCachedIO),
        ]
        for residency in residencies {
            let options = USearchDenseOptions(
                dimensions: 96,
                metric: .euclidean,
                quantization: .i8,
                connectivity: 24,
                dimensionPadding: .simdQuantum,
                residency: residency
            )
            let reloaded = try options.reloaded()
            XCTAssertEqual(reloaded.dimensions, 96)
            XCTAssertEqual(reloaded.metric, .euclidean)
            XCTAssertEqual(reloaded.quantization, .i8)
            XCTAssertEqual(reloaded.connectivity, 24)
            XCTAssertEqual(reloaded.dimensionPadding, .simdQuantum)
            switch (reloaded.residency, residency) {
            case (.ram(let threads, let pages), .ram(let wantedThreads, let wantedPages)):
                XCTAssertEqual(threads, wantedThreads)
                XCTAssertEqual(pages, wantedPages)
            case (.vram(let device, let addressing), .vram(let wantedDevice, let wantedAddressing)):
                XCTAssertEqual(device, wantedDevice)
                XCTAssertEqual(addressing, wantedAddressing)
            case (.nvme(let sector, let depth, let io), .nvme(let wantedSector, let wantedDepth, let wantedIO)):
                XCTAssertEqual(sector, wantedSector)
                XCTAssertEqual(depth, wantedDepth)
                XCTAssertEqual(io, wantedIO)
            default:
                XCTFail("the residency arm did not survive the round trip")
            }
        }
    }

    // endregion: Options

    // region: Marshalling

    func testUnfrozenGeometryIsRefused() throws {
        // A zeroed width is legal: the geometry defers to a later load. Every verb that needs
        // the width refuses until it freezes, rather than deriving a zero-byte row.
        let index = try USearchIndex(options: USearchDenseOptions())
        XCTAssertEqual(try index.dimensions, 0)
        let vector: [Float] = [0.1, 0.2, 0.3, 0.4]
        try vector.withUnsafeBufferPointer { buffer in
            let matrix = try USearchMatrixView.rows(of: buffer, dimensions: 4)
            XCTAssertThrowsError(try index.add(keys: [1], vectors: matrix)) { failure in
                XCTAssertEqual((failure as? USearchError)?.status, .uninitialized)
            }
            XCTAssertThrowsError(try index.search(queries: matrix, wanted: 1))
        }
    }

    func testStrideArithmetic() throws {
        let index = try USearchIndex(
            options: USearchDenseOptions(dimensions: 4, metric: .euclidean, quantization: .f32)
        )
        try index.reserve(4)

        // Three rows of four dimensions inside a slab pitched at six floats per row - the padded
        // layout a memory-mapped dataset arrives in. Only the first four of every six are read.
        let padded: [Float] = [
            1, 0, 0, 0, 9, 9,
            0, 1, 0, 0, 9, 9,
            0, 0, 1, 0, 9, 9,
        ]
        let pitch = 6 * MemoryLayout<Float>.stride
        try padded.withUnsafeBufferPointer { buffer in
            let matrix = try USearchMatrixView.strided(buffer, rows: 3, dimensions: 4, stride: pitch)
            XCTAssertEqual(matrix.rows, 3)
            XCTAssertEqual(matrix.stride, pitch)
            let inserted = try index.add(keys: [1, 2, 3], vectors: matrix)
            XCTAssertEqual(inserted.rows, 3)
        }
        XCTAssertEqual(try index.size, 3)

        // The packed constructor derives the row count, and rejects a partial trailing row.
        try padded.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(try USearchMatrixView.rows(of: buffer, dimensions: 6).rows, 3)
            XCTAssertThrowsError(try USearchMatrixView.rows(of: buffer, dimensions: 4))
            XCTAssertThrowsError(try USearchMatrixView.rows(of: buffer, dimensions: 0))
            XCTAssertThrowsError(try USearchMatrixView.strided(buffer, rows: 3, dimensions: 4, stride: 8))
            XCTAssertThrowsError(try USearchMatrixView.strided(buffer, rows: 99, dimensions: 4, stride: pitch))
        }

        // A row read back through a wider pitch lands where the stride says, leaving the padding
        // untouched.
        var exported = [Float](repeating: -1, count: 6)
        let entries = try exported.withUnsafeMutableBufferPointer { buffer in
            try index.get(key: 2, into: USearchMutableMatrixView.strided(buffer, rows: 1, dimensions: 4, stride: pitch))
        }
        XCTAssertEqual(entries, 1)
        XCTAssertEqual(Array(exported[0 ..< 4]), [0, 1, 0, 0])
        XCTAssertEqual(Array(exported[4 ..< 6]), [-1, -1])
    }

    func testUnknownScalarKindIsReported() throws {
        // The stub implements five scalar kinds and rejects the rest by tag rather than guessing
        // at their width, so a mismatched row size surfaces as a status instead of a bad read.
        let index = try USearchIndex(options: USearchDenseOptions(dimensions: 8, quantization: .f32))
        let packed = [USearchU1x8](repeating: USearchU1x8(bits: 0), count: 1)
        try packed.withUnsafeBufferPointer { buffer in
            let matrix = try USearchMatrixView.rows(of: buffer, dimensions: 8)
            XCTAssertEqual(matrix.rows, 1)
            XCTAssertEqual(matrix.stride, 1)
            XCTAssertThrowsError(try index.add(keys: [1], vectors: matrix)) { failure in
                XCTAssertEqual((failure as? USearchError)?.status, .unknownKind)
            }
        }
    }

    // endregion: Marshalling

    // region: Results

    func testTruncationReporting() throws {
        let index = try USearchIndex(
            options: USearchDenseOptions(dimensions: 2, metric: .euclidean, quantization: .f32)
        )
        try index.reserve(3)

        // Three entries share one key, so a two-row export truncates and reports the true total.
        let vectors: [Float] = [1, 0, 2, 0, 3, 0]
        try vectors.withUnsafeBufferPointer { buffer in
            _ = try index.add(keys: [7, 7, 7], vectors: USearchMatrixView.rows(of: buffer, dimensions: 2))
        }
        XCTAssertEqual(try index.count(key: 7), 3)
        XCTAssertTrue(try index.contains(key: 7))
        XCTAssertFalse(try index.contains(key: 8))

        var exported = [Float](repeating: -1, count: 4)
        let entries = try exported.withUnsafeMutableBufferPointer { buffer in
            try index.get(key: 7, into: USearchMutableMatrixView.rows(of: buffer, dimensions: 2))
        }
        XCTAssertEqual(entries, 3)
        XCTAssertEqual(exported, [1, 0, 2, 0])

        // A search asking for more than the index holds reports what it wrote and nothing past
        // it, so the untouched cells never reach a caller.
        let queries: [Float] = [1, 0]
        let matches = try queries.withUnsafeBufferPointer { buffer in
            try index.search(queries: USearchMatrixView.rows(of: buffer, dimensions: 2), wanted: 10)
        }
        XCTAssertEqual(matches.queries, 1)
        XCTAssertEqual(matches.wanted, 10)
        XCTAssertEqual(matches.keys.count, 10)
        XCTAssertEqual(matches.matches(query: 0).count, 3)
        XCTAssertEqual(matches.matches(query: 1).count, 0)
        XCTAssertFalse(matches.wasTruncated(query: 0))
        XCTAssertTrue(matches.stats.computedDistances > 0)
    }

    func testFilteredSearchTrampoline() throws {
        let index = try USearchIndex(
            options: USearchDenseOptions(dimensions: 2, metric: .euclidean, quantization: .f32)
        )
        try index.reserve(3)
        let vectors: [Float] = [1, 0, 0, 1, 1, 1]
        try vectors.withUnsafeBufferPointer { buffer in
            _ = try index.add(keys: [1, 2, 3], vectors: USearchMatrixView.rows(of: buffer, dimensions: 2))
        }

        let queries: [Float] = [1, 0]
        let admitted = try queries.withUnsafeBufferPointer { buffer in
            try index.search(
                queries: USearchMatrixView.rows(of: buffer, dimensions: 2),
                wanted: 3,
                admitting: { $0 == USearchKey(2) }
            )
        }
        XCTAssertEqual(admitted.matches(query: 0).map(\.key), [USearchKey(2)])

        let refused = try queries.withUnsafeBufferPointer { buffer in
            try index.search(
                queries: USearchMatrixView.rows(of: buffer, dimensions: 2),
                wanted: 3,
                admitting: { _ in false }
            )
        }
        XCTAssertEqual(refused.matches(query: 0).count, 0)
    }

    // endregion: Results

    // region: Lifetime

    func testHandleLifetime() throws {
        // The index frees its handle when the last reference drops - the leak this binding used
        // to carry is visible only as a deallocation that never happens.
        weak var observed: USearchIndex?
        var index: USearchIndex? = try USearchIndex(options: USearchDenseOptions(dimensions: 2, quantization: .f32))
        observed = index
        XCTAssertNotNil(observed)
        index = nil
        XCTAssertNil(observed)
    }

    func testMergeConsumesTheDonor() throws {
        let options = USearchDenseOptions(dimensions: 2, metric: .euclidean, quantization: .f32)
        let index = try USearchIndex(options: options)
        let donor = try USearchIndex(options: options)
        try index.reserve(2)
        try donor.reserve(2)

        let first: [Float] = [1, 0]
        let second: [Float] = [0, 1]
        try first.withUnsafeBufferPointer { buffer in
            _ = try index.add(keys: [1], vectors: USearchMatrixView.rows(of: buffer, dimensions: 2))
        }
        try second.withUnsafeBufferPointer { buffer in
            _ = try donor.add(keys: [2], vectors: USearchMatrixView.rows(of: buffer, dimensions: 2))
        }

        try index.merge(donor: donor)
        XCTAssertEqual(try index.size, 2)

        // The donor is inert afterwards: its handle belongs to the absorbing index, so every
        // verb on it reports an uninitialized handle rather than reaching freed memory.
        XCTAssertThrowsError(try donor.size) { failure in
            XCTAssertEqual((failure as? USearchError)?.status, .uninitialized)
        }
        XCTAssertThrowsError(try index.merge(donor: donor))
    }

    func testSerializationRoundTrip() throws {
        let options = USearchDenseOptions(dimensions: 2, metric: .euclidean, quantization: .f32)
        let index = try USearchIndex(options: options)
        try index.reserve(2)
        let vectors: [Float] = [1, 0, 0, 1]
        try vectors.withUnsafeBufferPointer { buffer in
            _ = try index.add(keys: [11, 12], vectors: USearchMatrixView.rows(of: buffer, dimensions: 2))
        }

        var buffer = [UInt8](repeating: 0, count: try index.serializedLength)
        try buffer.withUnsafeMutableBytes { try index.save(buffer: $0) }

        let restored = try USearchIndex(options: USearchDenseOptions())
        try buffer.withUnsafeBytes { try restored.load(buffer: $0) }
        XCTAssertEqual(try restored.size, 2)
        XCTAssertEqual(try restored.dimensions, 2)
        XCTAssertEqual(try restored.count(key: 11), 1)

        let described = try buffer.withUnsafeBytes { try USearchIndex.metadata(buffer: $0) }
        XCTAssertEqual(described.dimensions, 2)
        XCTAssertEqual(described.metric, .euclidean)
        XCTAssertEqual(described.quantization, .f32)
    }

    // endregion: Lifetime

    func testIntrospection() throws {
        XCTAssertFalse(usearchVersion().isEmpty)
        XCTAssertFalse(usearchCapabilitiesCompiled().isEmpty)
        XCTAssertFalse(usearchCapabilitiesAvailable().isEmpty)
    }
}
