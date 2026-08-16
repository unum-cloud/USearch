//
//  DenseIndex.swift
//  USearch
//
//  Vector Search over equidimensional dense embeddings - the Swift twin of
//  include/usearch/dense_index.h.
//
//  Every verb is a batch over strided matrices, so a memory-mapped dataset reaches the index
//  with no reshape and no copy, and a single vector is a one-row matrix. The candidate-pool
//  breadth rides each call rather than the handle, which lets a recall sweep run against one
//  immutable index with no mutation between points.
//

import USearchC

// region: Options

/// A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
/// capabilities. All of them derive from a single inner-product kernel with pre-computed
/// magnitudes; hamming and jaccard are bit-level, over the packed sub-byte kinds.
///
/// The C layer's custom-callback escape hatch stays unexposed here, as in the Rust and Go
/// bindings: a caller-supplied metric forfeits the SIMD kernel chains and the magnitude cache
/// both.
public enum USearchMetric: Int32, Sendable {
    /// A metric the library could not name.
    case unknown = 0
    /// The inner product.
    case inner = 1
    /// The cosine distance.
    case angular = 2
    /// The squared euclidean distance.
    case euclidean = 3
    /// The count of differing bits.
    case hamming = 4
    /// The bit-level jaccard distance.
    case jaccard = 5

    /// The metric rendered as its C enumerator's stem.
    public var name: String {
        switch self {
        case .unknown: return "unknown"
        case .inner: return "inner"
        case .angular: return "angular"
        case .euclidean: return "euclidean"
        case .hamming: return "hamming"
        case .jaccard: return "jaccard"
        }
    }

    var raw: usearch_dense_metric_t { usearch_dense_metric_t(rawValue: UInt32(bitPattern: rawValue)) }

    /// Lifts a tag the library wrote, refusing to guess at one this build cannot name.
    static func lifted(_ raw: usearch_dense_metric_t) throws -> USearchMetric {
        guard let metric = USearchMetric(rawValue: Int32(bitPattern: raw.rawValue)), metric != .unknown else {
            throw fail(.unknownKind, "the checkpoint names an unknown metric")
        }
        return metric
    }
}

/// Whether stored rows round up to the SIMD step quantum. A property of the stored layout, so
/// it applies whichever residency traverses it.
public enum USearchDimensionPadding: Sendable {
    /// Rows keep their natural width.
    case natural
    /// Cache-line-aligned rows and tail-free unrolled kernels, at slightly larger arenas.
    case simdQuantum
}

/// The construction and loading options of a dense index.
public struct USearchDenseOptions: Sendable {
    /// The width of the indexed vectors; zero defers the geometry to a later load, which reads
    /// it off the checkpoint.
    public var dimensions: Int
    /// The distance kind used between vectors.
    public var metric: USearchMetric
    /// The scalar type vectors are stored and quantized into.
    public var quantization: USearchScalar
    /// The bound on connections per node in the graph; zero derives.
    public var connectivity: Int
    /// Whether stored rows round up to the SIMD step quantum.
    public var dimensionPadding: USearchDimensionPadding
    /// Where the payload lives, and what loading a checkpoint means.
    public var residency: USearchResidency

    /// An angular bf16 index in RAM, with every optional field derived.
    public init(
        dimensions: Int = 0,
        metric: USearchMetric = .angular,
        quantization: USearchScalar = .bf16,
        connectivity: Int = 0,
        dimensionPadding: USearchDimensionPadding = .natural,
        residency: USearchResidency = .default
    ) {
        self.dimensions = dimensions
        self.metric = metric
        self.quantization = quantization
        self.connectivity = connectivity
        self.dimensionPadding = dimensionPadding
        self.residency = residency
    }

    /// Lowers the options, leaving the custom-metric callback null.
    func lowered() -> usearch_dense_options_t {
        var raw = usearch_dense_options_t()
        raw.metric = metric.raw
        raw.quantization = quantization.raw
        raw.dimensions = dimensions
        raw.pad_dimensions = dimensionPadding == .simdQuantum
        raw.connectivity = connectivity
        residency.write(into: &raw)
        return raw
    }

    /// Lifts the options a checkpoint's headers describe.
    static func lifted(_ raw: usearch_dense_options_t) throws -> USearchDenseOptions {
        USearchDenseOptions(
            dimensions: raw.dimensions,
            metric: try USearchMetric.lifted(raw.metric),
            quantization: try USearchScalar.lifted(raw.quantization),
            connectivity: raw.connectivity,
            dimensionPadding: raw.pad_dimensions ? .simdQuantum : .natural,
            residency: try USearchResidency.read(from: raw)
        )
    }

    /// Lowers the options and lifts them straight back - the marshalling round trip, and the
    /// only thing that exercises the residency union without an index behind it.
    public func reloaded() throws -> USearchDenseOptions { try USearchDenseOptions.lifted(lowered()) }
}

// endregion: Options

// region: Per-Call Settings & Telemetry

/// The traversal telemetry one call reports. Every call overwrites it rather than accumulating,
/// so a sweep over many batches sums them itself.
public struct USearchTraversalStats: Equatable, Sendable {
    /// The graph members the call visited.
    public var visitedMembers: Int = 0
    /// The distances the call computed, diversity re-selections included.
    public var computedDistances: Int = 0
    /// The starvation re-seedings - a graph-connectivity health signal rather than a cost
    /// counter.
    public var restarts: Int = 0

    /// Folds another call's telemetry into this one.
    public mutating func accumulate(_ other: USearchTraversalStats) {
        visitedMembers += other.visitedMembers
        computedDistances += other.computedDistances
        restarts += other.restarts
    }

    init() {}

    init(_ raw: usearch_dense_traversal_stats_t) {
        visitedMembers = raw.visited_members
        computedDistances = raw.computed_distances
        restarts = raw.restarts
    }
}

// endregion: Per-Call Settings & Telemetry

// region: Results

/// One retrieved entry.
public struct USearchMatch: Equatable, Sendable {
    /// The matched entry's key.
    public let key: USearchKey
    /// Its distance to the query, in the index's metric.
    public let distance: USearchDistance
}

/// What a batched search wrote: a queries-by-wanted matrix of keys, the distances positionally
/// paired with them, one match count per query, and the telemetry of the walk.
///
/// A count may exceed the wanted width, and the cells past it are memory the engine never
/// wrote, so every read goes through `matches(query:)` rather than into the slabs directly.
public struct USearchMatches: Sendable {
    /// The `queries × wanted` key matrix, rows contiguous.
    public let keys: [USearchKey]
    /// The `queries × wanted` distance matrix, rows contiguous.
    public let distances: [USearchDistance]
    /// The per-query match counts, as the engine reported them.
    public let counts: [Int]
    /// The upper bound on matches per query - the "k" in "kANN" and the row width of both
    /// matrices.
    public let wanted: Int
    /// The traversal telemetry of the call that produced these.
    public let stats: USearchTraversalStats

    /// How many queries the slabs were sized for.
    public var queries: Int { counts.count }

    /// One query's matches, trimmed to the cells the engine actually wrote. An out-of-range
    /// query yields nothing.
    public func matches(query: Int) -> [USearchMatch] {
        guard query >= 0, query < counts.count else { return [] }
        let written = min(counts[query], wanted)
        let from = query * wanted
        return (0 ..< written).map { USearchMatch(key: keys[from + $0], distance: distances[from + $0]) }
    }

    /// Whether the engine had more to report for a query than the wanted width could hold.
    public func wasTruncated(query: Int) -> Bool {
        guard query >= 0, query < counts.count else { return false }
        return counts[query] > wanted
    }
}

// endregion: Results

// region: Index

/// A Vector Search Index over equidimensional dense representations.
///
/// A handle may be shared across threads, so no method here takes a lock. On a RAM handle,
/// insertions and searches overlap freely: searches read only entries published behind the
/// visibility watermark and never write to the graph, while insertions synchronize through
/// per-node locks and publish in order. Serialization is the one exception, demanding a
/// mutation-free window, though concurrent readers keep searching throughout.
public final class USearchIndex {
    /// The native handle, released by `deinit` and emptied when a merge consumes it.
    private var handle: usearch_dense_index_t?

    /// Initializes an index for the residency its options select.
    public init(options: USearchDenseOptions) throws {
        var raw = options.lowered()
        let created = try throwing { usearch_dense_init(&raw, $0) }
        guard let created else { throw fail(.internalFailure, "the library returned no handle and no error") }
        handle = created
    }

    deinit {
        guard let handle else { return }
        usearch_dense_free(handle)
    }

    /// Reads a checkpoint's geometry and metric metadata without loading it - from the
    /// manifest's summary when the path names one, so no shard is opened.
    public static func metadata(path: String) throws -> USearchDenseOptions {
        var raw = usearch_dense_options_t()
        try withBorrowed(path) { borrowed in try throwing { usearch_dense_metadata(borrowed, &raw, $0) } }
        return try USearchDenseOptions.lifted(raw)
    }

    /// Reads geometry and metric metadata from a serialized monolith buffer.
    public static func metadata(buffer: UnsafeRawBufferPointer) throws -> USearchDenseOptions {
        guard let base = buffer.baseAddress, buffer.count != 0 else {
            throw fail(.malformedInput, "the buffer is empty")
        }
        var raw = usearch_dense_options_t()
        try throwing { usearch_dense_metadata_buffer(base, buffer.count, &raw, $0) }
        return try USearchDenseOptions.lifted(raw)
    }

    /// Reserves capacity for a total number of entries, the current size included.
    public func reserve(_ capacity: Int) throws {
        let handle = try liveHandle()
        try throwing { usearch_dense_reserve(handle, capacity, $0) }
    }

    /// Erases all entries, keeping the reserved capacity and the frozen geometry.
    public func clear() throws {
        let handle = try liveHandle()
        try throwing { usearch_dense_clear(handle, $0) }
    }

    /// The number of stored entries.
    public var size: Int { get throws { try measure(usearch_dense_size) } }

    /// The reserved capacity in entries.
    public var capacity: Int { get throws { try measure(usearch_dense_capacity) } }

    /// The frozen dimensionality of the stored vectors; zero while the geometry still defers to
    /// a later load.
    public var dimensions: Int { get throws { try measure(usearch_dense_dimensions) } }

    /// The frozen connections-per-node bound of the graph.
    public var connectivity: Int { get throws { try measure(usearch_dense_connectivity) } }

    /// The memory usage of the index in bytes.
    public var memoryUsage: Int { get throws { try measure(usearch_dense_memory_usage) } }

    /// The expected serialized size in bytes: every shard file, plus the manifest when the index
    /// spans more than one.
    public var serializedLength: Int { get throws { try measure(usearch_dense_serialized_length) } }

    /// The capability codename the index dispatched to on this machine.
    public var capabilities: String {
        get throws {
            let handle = try liveHandle()
            let raw = try throwing { usearch_dense_capabilities(handle, $0) }
            guard let raw else { throw fail(.internalFailure, "the library reported no capabilities and no error") }
            return String(cString: raw)
        }
    }

    /// Inserts a matrix of vectors under their keys - the native batch operation, where a single
    /// vector is a one-row matrix. Overlaps safely with concurrent searches on the same handle.
    ///
    /// The input scalar type is independent of the stored quantization; the library converts on
    /// ingest. An expansion factor of zero derives the candidate-pool breadth.
    ///
    /// A batch is all-or-nothing: every fallible step runs before the first entry identity is
    /// claimed, so a thrown failure leaves the index untouched. An accelerator fault severe
    /// enough to poison the device context is the sole exception, reporting `deviceFailure` and
    /// latching the index unusable rather than partially applied.
    @discardableResult
    public func add(
        keys: [USearchKey],
        vectors: USearchMatrixView,
        expansionFactor: Int = 0
    ) throws -> (rows: Int, stats: USearchTraversalStats) {
        let handle = try liveHandle()
        try vectors.validate(dimensions: try frozenDimensions())
        guard keys.count == vectors.rows else {
            throw fail(.shapeMismatch, "the key count does not match the vector row count")
        }
        guard !keys.isEmpty else { return (0, USearchTraversalStats()) }

        var config = usearch_dense_expansion_config_t(expansion_factor: expansionFactor)
        var rawStats = usearch_dense_traversal_stats_t()
        let added = try keys.withUnsafeBufferPointer { keysBuffer in
            try keysBuffer.withMemoryRebound(to: usearch_key_t.self) { rawKeys in
                try throwing {
                    usearch_dense_add(
                        handle,
                        rawKeys.baseAddress,
                        vectors.base,
                        vectors.rows,
                        vectors.stride,
                        vectors.scalar.raw,
                        &config,
                        &rawStats,
                        $0
                    )
                }
            }
        }
        return (added, USearchTraversalStats(rawStats))
    }

    /// Resolves a matrix of queries - the native batch operation, parallelized internally.
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    public func search(
        queries: USearchMatrixView,
        wanted: Int,
        expansionFactor: Int = 0
    ) throws -> USearchMatches {
        try searchDispatch(queries: queries, wanted: wanted, expansionFactor: expansionFactor, admits: nil)
    }

    /// The filtered sibling of `search(queries:wanted:expansionFactor:)`, admitting only the keys
    /// the predicate approves. Non-matching entries still route the traversal, so recall does not
    /// collapse at low selectivity.
    ///
    /// The predicate runs on the library's own worker threads, so it must be safe for concurrent
    /// use, and it cannot throw: a Swift error unwinding into a C frame would tear the engine's
    /// traversal down mid-walk.
    public func search(
        queries: USearchMatrixView,
        wanted: Int,
        expansionFactor: Int = 0,
        admitting admits: @escaping (USearchKey) -> Bool
    ) throws -> USearchMatches {
        try searchDispatch(queries: queries, wanted: wanted, expansionFactor: expansionFactor, admits: admits)
    }

    /// The number of entries stored under a key; zero means the index holds none.
    public func count(key: USearchKey) throws -> Int {
        let handle = try liveHandle()
        return try throwing { usearch_dense_count(handle, key.storage, $0) }
    }

    /// Whether the index holds at least one entry under a key.
    public func contains(key: USearchKey) throws -> Bool { try count(key: key) != 0 }

    /// Retrieves the vectors stored under a key into the matrix, one row per entry sharing the
    /// key, in insertion order.
    ///
    /// Returns the true number of entries, never the number written: a count above the matrix's
    /// row capacity means the copy was truncated, so size for that many and call again.
    /// `count(key:)` is the same number without the copy.
    public func get(key: USearchKey, into vectors: USearchMutableMatrixView) throws -> Int {
        let handle = try liveHandle()
        try vectors.validate(dimensions: try frozenDimensions())
        return try throwing {
            usearch_dense_get(
                handle,
                key.storage,
                vectors.base,
                vectors.rows,
                vectors.stride,
                vectors.scalar.raw,
                $0
            )
        }
    }

    /// Absorbs a donor index - cheaper than re-inserting its vectors one by one, wiring
    /// cross-links via landmark stitching. Readers of this index may keep searching throughout,
    /// since unpublished rows hide behind the watermark.
    ///
    /// The donor demands exclusive ownership: the C side consumes it whether the call succeeds
    /// or fails, so its handle is emptied here and any concurrent verb on it is a use-after-free.
    /// The emptied donor is inert afterwards - its own deallocation frees nothing.
    public func merge(donor: USearchIndex) throws {
        let handle = try liveHandle()
        let consumed = try donor.surrenderHandle()
        try throwing { usearch_dense_merge(handle, consumed, $0) }
    }

    /// Writes the index under a path, in the layout its extension picks - one `.safegraphs`
    /// monolith, or a `.safegraphs.json` manifest beside its shards. Demands a mutation-free
    /// window; concurrent readers keep searching throughout.
    public func save(path: String) throws {
        let handle = try liveHandle()
        try withBorrowed(path) { borrowed in try throwing { usearch_dense_save(handle, borrowed, $0) } }
    }

    /// Reads a checkpoint into this handle, shard by shard, with the handle's residency deciding
    /// the behavior: materialize into RAM, upload to the device, or open in place on disk. A
    /// handle whose geometry was left zeroed freezes it here, from the checkpoint's own headers.
    public func load(path: String) throws {
        let handle = try liveHandle()
        try withBorrowed(path) { borrowed in try throwing { usearch_dense_load(handle, borrowed, $0) } }
    }

    /// Writes the index into a caller-owned buffer as one monolith. Size the buffer with
    /// `serializedLength`.
    public func save(buffer: UnsafeMutableRawBufferPointer) throws {
        let handle = try liveHandle()
        guard let base = buffer.baseAddress, buffer.count != 0 else {
            throw fail(.shapeMismatch, "the buffer is empty")
        }
        try throwing { usearch_dense_save_buffer(handle, base, buffer.count, $0) }
    }

    /// Reads a monolith from a buffer into this handle, with the residency deciding the behavior
    /// like `load(path:)`. A residency that opens in place demands the buffer outlive the handle.
    public func load(buffer: UnsafeRawBufferPointer) throws {
        let handle = try liveHandle()
        guard let base = buffer.baseAddress, buffer.count != 0 else {
            throw fail(.malformedInput, "the buffer is empty")
        }
        try throwing { usearch_dense_load_buffer(handle, base, buffer.count, $0) }
    }

    // The filtered search hands the engine a pointer to this box rather than to a Swift closure,
    // which has no stable C representation.
    private final class AdmissionBox {
        let admits: (USearchKey) -> Bool
        init(_ admits: @escaping (USearchKey) -> Bool) { self.admits = admits }
    }

    private func searchDispatch(
        queries: USearchMatrixView,
        wanted: Int,
        expansionFactor: Int,
        admits: ((USearchKey) -> Bool)?
    ) throws -> USearchMatches {
        let handle = try liveHandle()
        try queries.validate(dimensions: try frozenDimensions())
        guard wanted > 0 else { throw fail(.shapeMismatch, "the search reserves no room for a single match") }

        // The wanted count has no ceiling, so the two output matrices are sized through the
        // overflow-checked extent before a single byte is claimed.
        let keyBytes = MemoryLayout<usearch_key_t>.stride
        let distanceBytes = MemoryLayout<usearch_distance_t>.stride
        let keysExtent = try checkedExtent(rows: queries.rows, columns: wanted, elementBytes: keyBytes)
        _ = try checkedExtent(rows: queries.rows, columns: wanted, elementBytes: distanceBytes)
        let cells = keysExtent / keyBytes
        guard queries.rows != 0 else {
            return USearchMatches(keys: [], distances: [], counts: [], wanted: wanted, stats: USearchTraversalStats())
        }

        var keys = [USearchKey](repeating: .nilKey, count: cells)
        var distances = [USearchDistance](repeating: 0, count: cells)
        var counts = [Int](repeating: 0, count: queries.rows)
        var config = usearch_dense_expansion_config_t(expansion_factor: expansionFactor)
        var rawStats = usearch_dense_traversal_stats_t()
        var failure = usearch_error_t()

        keys.withUnsafeMutableBufferPointer { keysBuffer in
            keysBuffer.withMemoryRebound(to: usearch_key_t.self) { rawKeys in
                distances.withUnsafeMutableBufferPointer { distancesBuffer in
                    counts.withUnsafeMutableBufferPointer { countsBuffer in
                        guard let admits else {
                            usearch_dense_search(
                                handle,
                                queries.base,
                                queries.rows,
                                queries.stride,
                                queries.scalar.raw,
                                wanted,
                                &config,
                                &rawStats,
                                rawKeys.baseAddress,
                                wanted * keyBytes,
                                distancesBuffer.baseAddress,
                                wanted * distanceBytes,
                                countsBuffer.baseAddress,
                                &failure
                            )
                            return
                        }
                        let box = AdmissionBox(admits)
                        withExtendedLifetime(box) {
                            let state = Unmanaged.passUnretained(box).toOpaque()
                            usearch_dense_search_if(
                                handle,
                                queries.base,
                                queries.rows,
                                queries.stride,
                                queries.scalar.raw,
                                wanted,
                                &config,
                                &rawStats,
                                USearchIndex.usearchDenseAdmits,
                                state,
                                rawKeys.baseAddress,
                                wanted * keyBytes,
                                distancesBuffer.baseAddress,
                                wanted * distanceBytes,
                                countsBuffer.baseAddress,
                                &failure
                            )
                        }
                    }
                }
            }
        }
        try check(failure)
        return USearchMatches(
            keys: keys,
            distances: distances,
            counts: counts,
            wanted: wanted,
            stats: USearchTraversalStats(rawStats)
        )
    }

    /// The admission trampoline the engine calls from its worker threads. The boxed closure
    /// cannot throw, so nothing unwinds out of this frame and into C.
    private static let usearchDenseAdmits: usearch_predicate_t = { key, state in
        guard let state else { return 1 }
        return Unmanaged<AdmissionBox>.fromOpaque(state).takeUnretainedValue().admits(USearchKey(raw: key)) ? 1 : 0
    }

    /// The handle, or a thrown failure once a merge consumed it or a deallocation released it.
    private func liveHandle() throws -> usearch_dense_index_t {
        guard let handle else { throw fail(.uninitialized, "the index is consumed or was never initialized") }
        return handle
    }

    /// Hands the handle to a caller that takes ownership, leaving this index inert.
    private func surrenderHandle() throws -> usearch_dense_index_t {
        guard let taken = handle else { throw fail(.uninitialized, "the donor index is consumed or uninitialized") }
        handle = nil
        return taken
    }

    /// The frozen row width, refusing the verbs whose geometry a later load still decides.
    private func frozenDimensions() throws -> Int {
        let dimensions = try self.dimensions
        guard dimensions != 0 else {
            throw fail(.uninitialized, "the index geometry is unfrozen; reserve or load a checkpoint first")
        }
        return dimensions
    }

    /// Pairs a probe's reading with its error slot.
    private func measure(
        _ probe: (usearch_dense_index_t?, UnsafeMutablePointer<usearch_error_t>?) -> Int
    ) throws -> Int {
        let handle = try liveHandle()
        return try throwing { probe(handle, $0) }
    }
}

// endregion: Index

// region: Helpers

/// Views a Swift string as the pointer-and-length pair every path argument takes. The string is
/// never terminated, by contract, so nothing is copied past the UTF-8 encoding itself.
func withBorrowed<Value>(_ text: String, _ body: (usearch_string_t) throws -> Value) rethrows -> Value {
    var text = text
    return try text.withUTF8 { buffer in
        let data = buffer.baseAddress.map { UnsafeRawPointer($0).assumingMemoryBound(to: CChar.self) }
        return try body(usearch_string_t(data: data, length: buffer.count))
    }
}

// endregion: Helpers
