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

import NumKong
import USearchC

// MARK: - Options

/// A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
/// capabilities. All of them derive from a single inner-product kernel with pre-computed
/// magnitudes; hamming and jaccard are bit-level, over the packed sub-byte kinds.
///
/// The C layer's custom-callback escape hatch stays unexposed: a caller-supplied metric forfeits
/// the SIMD kernel chains and the magnitude cache both.
public enum Metric: Int32, Sendable {
    /// A metric the library could not name.
    case unknown = 0
    /// The inner product.
    case inner = 1
    /// The cosine distance.
    case angular = 2
    /// The L2 distance.
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
    static func lifted(_ raw: usearch_dense_metric_t) throws(Error) -> Metric {
        guard let metric = Metric(rawValue: Int32(bitPattern: raw.rawValue)), metric != .unknown else {
            throw fail(.unknownKind, "the checkpoint names an unknown metric")
        }
        return metric
    }
}

/// The construction and loading options of a dense index.
///
/// Stored rows always round up to the SIMD step quantum, so rows stay cache-line aligned and
/// every kernel runs tail-free; the arena is slightly larger for it, and that is not a knob.
public struct DenseOptions: Sendable {
    /// The width of the indexed vectors; zero defers the geometry to a later load, which reads
    /// it off the checkpoint.
    public var dimensions: Int
    /// The distance kind used between vectors.
    public var metric: Metric
    /// The scalar type vectors are stored and quantized into.
    public var quantization: Scalar
    /// The bound on connections per node in the graph; zero derives.
    public var connectivity: Int
    /// The bound on entries sharing one key; zero derives to one - unique keys.
    public var valuesPerKey: Int
    /// Where the payload lives, and what loading a checkpoint means.
    public var residency: Residency

    /// An angular bf16 index in RAM, with every optional field derived.
    public init(
        dimensions: Int = 0,
        metric: Metric = .angular,
        quantization: Scalar = .bf16,
        connectivity: Int = 0,
        valuesPerKey: Int = 0,
        residency: Residency = .default
    ) {
        self.dimensions = dimensions
        self.metric = metric
        self.quantization = quantization
        self.connectivity = connectivity
        self.valuesPerKey = valuesPerKey
        self.residency = residency
    }

    /// Lowers the options against an already-copied placement, leaving the custom-metric
    /// callback null. The placement must outlive every call the options are passed to.
    func lowered(_ placement: LoweredResidency) -> usearch_dense_options_t {
        var raw = usearch_dense_options_t()
        raw.metric = metric.raw
        raw.quantization = quantization.raw
        raw.dimensions = dimensions
        raw.connectivity = connectivity
        raw.values_per_key = valuesPerKey
        placement.write(into: &raw)
        return raw
    }

    /// Lifts a lowered struct back into options, the residency arriving with an empty placement
    /// the caller fills.
    static func lifted(_ raw: usearch_dense_options_t) throws(Error) -> DenseOptions {
        DenseOptions(
            dimensions: raw.dimensions,
            metric: try Metric.lifted(raw.metric),
            quantization: try Scalar.lifted(raw.quantization),
            connectivity: raw.connectivity,
            valuesPerKey: raw.values_per_key,
            residency: try Residency.lifted(raw.residency)
        )
    }

    /// Lowers the options and lifts them straight back - the marshalling round trip, and the
    /// only thing that exercises the residency union without an index behind it. The placement
    /// lists do not survive it, since lifting never reads them back.
    public func reloaded() throws(Error) -> DenseOptions {
        let placement = LoweredResidency(residency)
        return try withExtendedLifetime(placement) { () throws(Error) -> DenseOptions in
            try DenseOptions.lifted(lowered(placement))
        }
    }
}

/// The geometry and metric a checkpoint froze - everything a reader learns without opening it,
/// and nothing a reader must choose for itself.
///
/// A checkpoint records neither the residency nor the bound on entries sharing one key: the first
/// says where it was written rather than where a reader must keep it, and the second bounds a live
/// handle's mutable state rather than the serialized form. Both stay the caller's to choose in the
/// `DenseOptions` a handle is built from.
public struct DenseMetadata: Sendable {
    /// The scalars per stored vector.
    public var dimensions: Int
    /// The distance kind the checkpoint scores with.
    public var metric: Metric
    /// The scalar type its rows are stored and quantized into.
    public var quantization: Scalar
    /// The connections-per-node bound the graph was built with.
    public var connectivity: Int

    /// Lifts the geometry and metric a checkpoint's headers describe.
    static func lifted(_ raw: usearch_dense_metadata_t) throws(Error) -> DenseMetadata {
        DenseMetadata(
            dimensions: raw.dimensions,
            metric: try Metric.lifted(raw.metric),
            quantization: try Scalar.lifted(raw.quantization),
            connectivity: raw.connectivity
        )
    }
}

// MARK: - Per-Call Settings & Telemetry

/// The traversal telemetry one call reports. Every call overwrites it rather than accumulating,
/// so a sweep over many batches sums them itself.
public struct TraversalStats: Equatable, Sendable {
    /// The graph members the call visited.
    public var visitedMembers: Int = 0
    /// The distances the call computed, diversity re-selections included.
    public var computedDistances: Int = 0
    /// The starvation re-seedings - a graph-connectivity health signal rather than a cost
    /// counter.
    public var restarts: Int = 0

    /// Folds another call's telemetry into this one.
    public mutating func accumulate(_ other: TraversalStats) {
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

/// The per-call settings of insertion and refinement, the two verbs that write to the graph.
/// Every zeroed field derives, which is what almost every caller wants.
///
/// Extensible by design: construct one and set what matters rather than naming every field, so a
/// setting added later leaves existing call sites compiling.
public struct InsertionOptions: Equatable, Sendable {
    /// The candidate-pool breadth the insertion walk searches for neighbors with; zero derives it
    /// from the geometry.
    public var expansionFactor: Int = 0

    public init(expansionFactor: Int = 0) {
        self.expansionFactor = expansionFactor
    }

    /// Lowers into the struct the C interface reads.
    var lowered: usearch_dense_insertion_options_t {
        usearch_dense_insertion_options_t(expansion_factor: expansionFactor)
    }
}

/// The per-call settings of every read-only walk - both search forms, the sparse product, and
/// late interaction. Every zeroed field derives.
///
/// Extensible by design, like `InsertionOptions`: construct one and set what matters rather than
/// naming every field, so a setting added later leaves existing call sites compiling.
public struct SearchOptions: Equatable, Sendable {
    /// The candidate-pool breadth the walk explores; zero derives it from the geometry.
    public var expansionFactor: Int = 0

    public init(expansionFactor: Int = 0) {
        self.expansionFactor = expansionFactor
    }

    /// Lowers into the struct the C interface reads.
    var lowered: usearch_dense_search_options_t {
        usearch_dense_search_options_t(expansion_factor: expansionFactor)
    }
}

/// The per-call settings of `attend(...)`. Every zeroed field derives.
///
/// Extensible by design, like `SearchOptions`: construct one and set what matters.
public struct AttentionOptions: Equatable, Sendable {
    /// The candidate-pool breadth; zero derives it from the geometry.
    public var expansionFactor: Int = 0
    /// The logit scale, conventionally the inverse square root of the head dimensionality; zero
    /// derives it from the index's frozen geometry.
    public var scoreScale: Float = 0
    /// The residual-mass threshold the walk stops at - the mass-denominated sibling of the radius
    /// a search terminates on; zero derives.
    public var massTolerance: Float = 0

    public init(expansionFactor: Int = 0, scoreScale: Float = 0, massTolerance: Float = 0) {
        self.expansionFactor = expansionFactor
        self.scoreScale = scoreScale
        self.massTolerance = massTolerance
    }

    /// Lowers into the struct the C interface reads.
    var lowered: usearch_dense_attention_options_t {
        usearch_dense_attention_options_t(
            expansion_factor: expansionFactor,
            score_scale: scoreScale,
            mass_tolerance: massTolerance
        )
    }
}

/// Flash Attention's running maximum and partition for one query row.
///
/// The two are one concept: a maximum without its partition is a number nothing can rescale, so
/// they share an element rather than travelling as two arrays a caller could supply half of.
/// Carrying them is what folds one attention call into another over the same queries.
public struct SoftmaxState: Equatable, Sendable {
    /// The largest logit seen so far - the recurrence's `m`.
    public var maximumScore: Float = 0
    /// The sum of exponentials taken against that maximum - the recurrence's `l`.
    public var partition: Float = 0

    public init(maximumScore: Float = 0, partition: Float = 0) {
        self.maximumScore = maximumScore
        self.partition = partition
    }
}

// MARK: - Results

/// One retrieved entry.
public struct Match: Equatable, Sendable {
    /// The matched entry's key.
    public let key: Key
    /// Its distance to the query, in the index's metric.
    public let distance: Distance

    /// Pairs a key with its distance. Written out so the inlinable readers below may call it.
    @usableFromInline init(key: Key, distance: Distance) {
        self.key = key
        self.distance = distance
    }
}

/// Where a ranked run lands - three caller-owned buffers, sized together, and the shape every
/// ranking verb fills. Mirrors `usearch_matches_span_t`.
///
/// It borrows and never owns, so a benchmark loop allocates once and never again, and it stays
/// valid only as long as the storage it was built from. `span[query]` reads one query's row and
/// stops at the cells the engine wrote - indexing the buffers directly walks into memory the
/// engine never touched.
public struct MatchesSpan: RandomAccessCollection {
    /// The `queries × wanted` key matrix, rows contiguous.
    public let keys: UnsafeMutableBufferPointer<Key>
    /// The `queries × wanted` distance matrix, rows contiguous.
    public let distances: UnsafeMutableBufferPointer<Distance>
    /// The per-query match counts, one entry per query.
    public let counts: UnsafeMutableBufferPointer<Int>
    /// The upper bound on matches per query - the "k" in "kANN" and the row width of both
    /// matrices.
    public let wanted: Int

    /// Lends three buffers the caller already sized to `counts.count × wanted`.
    ///
    /// The row width is derived from the lengths rather than asked for, so the space reserved and
    /// the count asked for cannot disagree.
    public init(
        keys: UnsafeMutableBufferPointer<Key>,
        distances: UnsafeMutableBufferPointer<Distance>,
        counts: UnsafeMutableBufferPointer<Int>
    ) throws(Error) {
        guard !counts.isEmpty else { throw fail(.shapeMismatch, "a run must hold at least one query") }
        guard keys.count == distances.count else {
            throw fail(.shapeMismatch, "the key and distance buffers are not the same shape")
        }
        guard !keys.isEmpty, keys.count % counts.count == 0 else {
            throw fail(.shapeMismatch, "the buffers do not divide into one row per query")
        }
        self.keys = keys
        self.distances = distances
        self.counts = counts
        wanted = keys.count / counts.count
    }

    /// The first query's position, which a collection walk starts from.
    @inlinable public var startIndex: Int { 0 }

    /// One past the last query's position.
    @inlinable public var endIndex: Int { counts.count }

    /// One query's matches, trimmed to the cells the engine actually wrote.
    @inlinable public subscript(query: Int) -> QueryMatches { QueryMatches(run: self, query: query) }

    /// One query's row of a run, walked without copying it out.
    public struct QueryMatches: RandomAccessCollection {
        /// The run this row belongs to, whose buffers it reads through.
        public let run: MatchesSpan
        /// Which query's row it is.
        public let query: Int

        /// Names one row of a run. Written out so the inlinable subscript above may call it.
        @usableFromInline init(run: MatchesSpan, query: Int) {
            self.run = run
            self.query = query
        }

        /// Whether the engine had more to report than the wanted width could hold.
        @inlinable public var wasTruncated: Bool { run.counts[query] > run.wanted }

        /// The first match's position, which a collection walk starts from.
        @inlinable public var startIndex: Int { 0 }

        /// One past the last match the engine wrote, capped at the wanted width.
        @inlinable public var endIndex: Int { Swift.min(run.counts[query], run.wanted) }

        /// The match at a position, key and distance read from their paired buffers.
        @inlinable public subscript(position: Int) -> Match {
            let cell = query * run.wanted + position
            return Match(key: run.keys[cell], distance: run.distances[cell])
        }
    }

    /// Lowers into the descriptor the C interface fills.
    var lowered: usearch_matches_span_t {
        usearch_matches_span_t(
            keys: UnsafeMutableRawPointer(keys.baseAddress!).assumingMemoryBound(to: usearch_key_t.self),
            keys_stride_bytes: wanted * MemoryLayout<usearch_key_t>.stride,
            scores: distances.baseAddress,
            scores_stride_bytes: wanted * MemoryLayout<usearch_distance_t>.stride,
            counts: counts.baseAddress,
            wanted: wanted
        )
    }
}

// MARK: - Index

/// A Vector Search Index over equidimensional dense representations.
///
/// A handle may be shared across threads, so no method here takes a lock. On a RAM handle,
/// insertions and searches overlap freely: searches read only entries published behind the
/// visibility watermark and never write to the graph, while insertions synchronize through
/// per-node locks and publish in order. Serialization is the one exception, demanding a
/// mutation-free window, though concurrent readers keep searching throughout.
public final class DenseIndex: @unchecked Sendable {
    /// The native handle, released by `deinit` and never reassigned, so every verb sees the one
    /// the initializer claimed and the engine's own synchronization is all that sharing relies on.
    private let handle: usearch_dense_index_t

    /// Initializes an index for the residency its options select.
    public init(options: borrowing DenseOptions) throws(Error) {
        let placement = LoweredResidency(options.residency)
        let created = try withExtendedLifetime(placement) { () throws(Error) -> usearch_dense_index_t? in
            var raw = options.lowered(placement)
            return try throwing { usearch_dense_init(&raw, $0) }
        }
        guard let created else { throw fail(.internalFailure, "the library returned no handle and no error") }
        handle = created
    }

    deinit { usearch_dense_free(handle) }

    /// Reads a checkpoint's geometry and metric metadata without loading it - from the
    /// manifest's summary when the path names one, so no shard is opened.
    public static func metadata(path: borrowing String) throws(Error) -> DenseMetadata {
        var raw = usearch_dense_metadata_t()
        try withBorrowed(path) { (borrowed) throws(Error) in
            try throwing { usearch_dense_metadata(borrowed, &raw, $0) }
        }
        return try DenseMetadata.lifted(raw)
    }

    /// Reads geometry and metric metadata from a serialized monolith buffer.
    public static func metadata(buffer: UnsafeRawBufferPointer) throws(Error) -> DenseMetadata {
        guard let base = buffer.baseAddress, buffer.count != 0 else {
            throw fail(.malformedInput, "the buffer is empty")
        }
        var raw = usearch_dense_metadata_t()
        let view = usearch_bytes_view_t(data: base.assumingMemoryBound(to: UInt8.self), length: buffer.count)
        try throwing { usearch_dense_metadata_buffer(view, &raw, $0) }
        return try DenseMetadata.lifted(raw)
    }

    /// Reserves capacity for a total number of entries, the current size included.
    public func reserve(_ capacity: Int) throws(Error) {
        try throwing { usearch_dense_reserve(handle, capacity, $0) }
    }

    /// Erases all entries, keeping the reserved capacity and the frozen geometry.
    public func clear() throws(Error) {
        try throwing { usearch_dense_clear(handle, $0) }
    }

    /// The number of stored entries.
    public var size: Int { get throws(Error) { try measure(usearch_dense_size) } }

    /// The reserved capacity in entries.
    public var capacity: Int { get throws(Error) { try measure(usearch_dense_capacity) } }

    /// The frozen dimensionality of the stored vectors; zero while the geometry still defers to
    /// a later load.
    public var dimensions: Int { get throws(Error) { try measure(usearch_dense_dimensions) } }

    /// The frozen connections-per-node bound of the graph.
    public var connectivity: Int { get throws(Error) { try measure(usearch_dense_connectivity) } }

    /// The memory usage of the index in bytes.
    public var memoryUsage: Int { get throws(Error) { try measure(usearch_dense_memory_usage) } }

    /// The bytes one monolith of the index takes under the default layout - what `save(executor:buffer:)`
    /// needs; a manifest checkpoint's parts sum to about the same, plus the manifest.
    public var storageRequirements: Int {
        get throws(Error) { try measure(usearch_dense_storage_requirements) }
    }

    /// The capability codename the index dispatched to on this machine.
    public var capabilities: String {
        get throws(Error) {
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
    public func add<Element: ScalarValue>(
        executor: Executor,
        keys: borrowing [Key],
        vectors: borrowing MatrixView<Element>,
        options: InsertionOptions = InsertionOptions()
    ) throws(Error) -> (rows: Int, stats: TraversalStats) {
        try vectors.validate(dimensions: try frozenDimensions())
        guard !keys.isEmpty else { return (0, TraversalStats()) }

        var view = vectors.lowered
        var lowered = options.lowered
        var rawStats = usearch_dense_traversal_stats_t()
        let added = try keys.withUnsafeBufferPointer { (keysBuffer) throws(Error) -> Int in
            try keysBuffer.withMemoryRebound(to: usearch_key_t.self) { (rawKeys) throws(Error) -> Int in
                let run = usearch_keys_view_t(data: rawKeys.baseAddress, count: rawKeys.count)
                return try throwing {
                    usearch_dense_add(handle, executor.raw, run, &view, &lowered, nil, nil, &rawStats, $0)
                }
            }
        }
        return (added, TraversalStats(rawStats))
    }

    /// Resolves a matrix of queries into caller-owned outputs - the allocation-free batch path,
    /// parallelized internally.
    ///
    /// Never writes to the graph, so any number of these may overlap with concurrent insertions.
    @discardableResult
    public func search<Element: ScalarValue>(
        executor: Executor,
        queries: borrowing MatrixView<Element>,
        into matches: MatchesSpan,
        options: SearchOptions = SearchOptions()
    ) throws(Error) -> TraversalStats {
        try searchDispatch(
            executor: executor,
            queries: queries,
            matches: matches,
            options: options,
            admits: nil
        )
    }

    /// The filtered sibling of `search(executor:queries:into:options:)`, admitting only the keys
    /// the predicate approves. Non-matching entries still route the traversal, so recall does not
    /// collapse at low selectivity.
    ///
    /// The predicate runs on the executor's threads, so it must be safe for concurrent use, and
    /// it cannot throw: a Swift error unwinding into a C frame would tear the engine's traversal
    /// down mid-walk.
    @discardableResult
    public func search<Element: ScalarValue>(
        executor: Executor,
        queries: borrowing MatrixView<Element>,
        into matches: MatchesSpan,
        options: SearchOptions = SearchOptions(),
        admitting admits: @escaping (Key) -> Bool
    ) throws(Error) -> TraversalStats {
        try searchDispatch(
            executor: executor,
            queries: queries,
            matches: matches,
            options: options,
            admits: admits
        )
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, returning the
    /// nodes whose edges changed; zero means the graph has settled.
    ///
    /// A build step rather than a repair: a cheap insertion pass followed by one refinement
    /// reaches a better graph, in less total time, than one expensive insertion pass - so a wide
    /// expansion factor belongs here rather than in `add`. Searches may run throughout.
    @discardableResult
    public func refine(
        executor: Executor,
        options: InsertionOptions = InsertionOptions()
    ) throws(Error) -> (changed: Int, stats: TraversalStats) {
        var lowered = options.lowered
        var rawStats = usearch_dense_traversal_stats_t()
        let changed = try throwing {
            usearch_dense_refine(handle, executor.raw, &lowered, nil, nil, &rawStats, $0)
        }
        return (changed, TraversalStats(rawStats))
    }

    /// The approximate sparse product of a matrix against the indexed rows - the linear-algebra
    /// sibling of `search`, reporting every pair the traversal scored rather than only the
    /// nearest of them.
    ///
    /// Pairs arrive in traversal order rather than sorted, and carry no duplicates. A count above
    /// the wanted width means that row was truncated to its nearest pairs, which
    /// each query's `wasTruncated` reports. Runs on an inner-product index alone, since a
    /// product needs logits rather than distances.
    @discardableResult
    public func multiply<Element: ScalarValue>(
        executor: Executor,
        left: borrowing MatrixView<Element>,
        into products: MatchesSpan,
        options: SearchOptions = SearchOptions()
    ) throws(Error) -> TraversalStats {
        try left.validate(dimensions: try frozenDimensions())
        guard products.count == left.rows else {
            throw fail(.shapeMismatch, "the outputs are not sized for one row per left row")
        }
        var view = left.lowered
        var lowered = options.lowered
        var span = products.lowered
        var rawStats = usearch_dense_traversal_stats_t()
        try throwing { failure in
            usearch_dense_multiply(handle, executor.raw, &view, &span, &lowered, nil, nil, &rawStats, failure)
        }
        return TraversalStats(rawStats)
    }

    /// Late-interaction scoring of multi-row queries against multi-row entries: every query row
    /// takes its best match within a key, and those maxima sum to that key's score.
    ///
    /// A document is one key holding its token rows, which is what a `valuesPerKey` above one
    /// exists for. `rowsPerQuery` is the token count each query occupies, `queries` holds them end
    /// to end, and the outputs are sized over documents rather than rows. Approximate downward
    /// only: a token whose neighborhood the walk never entered contributes no maximum, so a key
    /// can score below its exact value but never above it. Runs on an inner-product index alone.
    ///
    /// See <https://arxiv.org/abs/2004.12832> - ColBERT, which introduced this scoring.
    @discardableResult
    public func maxsim<Element: ScalarValue>(
        executor: Executor,
        queries: borrowing MatrixView<Element>,
        rowsPerQuery: Int,
        into documents: MatchesSpan,
        options: SearchOptions = SearchOptions()
    ) throws(Error) -> TraversalStats {
        try queries.validate(dimensions: try frozenDimensions())
        guard rowsPerQuery > 0, queries.rows % rowsPerQuery == 0 else {
            throw fail(.shapeMismatch, "the query rows do not divide into whole queries")
        }
        guard documents.count == queries.rows / rowsPerQuery else {
            throw fail(.shapeMismatch, "the outputs are not sized for one row per whole query")
        }
        var view = queries.lowered
        var lowered = options.lowered
        var span = documents.lowered
        var rawStats = usearch_dense_traversal_stats_t()
        try throwing { failure in
            usearch_dense_maxsim(
                handle,
                executor.raw,
                &view,
                rowsPerQuery,
                &span,
                &lowered,
                nil,
                nil,
                &rawStats,
                failure
            )
        }
        return TraversalStats(rawStats)
    }

    /// Softmax-weighted attention of a query matrix over the indexed keys, folded into the graph
    /// walk in a single pass - no match list is materialized, and each value row is gathered
    /// exactly once, at the moment its key was scored.
    ///
    /// The index holds the keys and `values` holds the value rows, addressed by key: an entry's
    /// key is its row offset into `values`, so a context inserted in position order needs no
    /// mapping. `outputs` comes back already divided by its partition.
    ///
    /// Passing a `carry` asks for the online-softmax state, one entry per query row, which folds
    /// this branch into another over the same queries by partial-softmax rescaling. Runs on an
    /// inner-product index alone, since a softmax needs logits.
    ///
    /// See <https://arxiv.org/abs/2205.14135> - Flash Attention, whose recurrence this reuses.
    @discardableResult
    public func attend<Query: ScalarValue, Value: ScalarValue, Output: ScalarValue>(
        executor: Executor,
        queries: borrowing MatrixView<Query>,
        values: borrowing MatrixView<Value>,
        into outputs: borrowing MatrixSpan<Output>,
        carry: UnsafeMutableBufferPointer<SoftmaxState>? = nil,
        options: AttentionOptions = AttentionOptions()
    ) throws(Error) -> TraversalStats {
        try queries.validate(dimensions: try frozenDimensions())
        guard outputs.rows >= queries.rows else {
            throw fail(.shapeMismatch, "the outputs hold fewer rows than there are queries")
        }
        guard outputs.dimensions == values.dimensions else {
            throw fail(.shapeMismatch, "the outputs are not as wide as the value rows")
        }
        if let carry, carry.count < queries.rows {
            throw fail(.shapeMismatch, "the carry holds fewer states than there are queries")
        }

        var queriesView = queries.lowered
        var valuesView = values.lowered
        var outputsSpan = outputs.lowered
        var lowered = options.lowered
        var rawStats = usearch_dense_traversal_stats_t()
        var failure = usearch_error_t()
        let attend = { (states: UnsafeMutablePointer<usearch_dense_softmax_state_t>?) -> Void in
            usearch_dense_attend(
                self.handle,
                executor.raw,
                &queriesView,
                &valuesView,
                &outputsSpan,
                states,
                &lowered,
                nil,
                nil,
                &rawStats,
                &failure
            )
        }
        if let carry {
            carry.withMemoryRebound(to: usearch_dense_softmax_state_t.self) { attend($0.baseAddress) }
        }
        else {
            attend(nil)
        }
        try check(failure)
        return TraversalStats(rawStats)
    }

    /// The number of entries stored under a key; zero means the index holds none.
    public func count(key: Key) throws(Error) -> Int {
        return try throwing { usearch_dense_count(handle, key.storage, $0) }
    }

    /// Whether the index holds at least one entry under a key.
    public func contains(key: Key) throws(Error) -> Bool { try count(key: key) != 0 }

    /// Retrieves the vectors stored under a key into the matrix, one row per entry sharing the
    /// key, in insertion order.
    ///
    /// Returns the true number of entries, never the number written: a count above the matrix's
    /// row capacity means the copy was truncated, so size for that many and call again.
    /// `count(key:)` is the same number without the copy.
    public func get<Element: ScalarValue>(
        key: Key,
        into vectors: borrowing MatrixSpan<Element>
    ) throws(Error) -> Int {
        try vectors.validate(dimensions: try frozenDimensions())
        var span = vectors.lowered
        return try throwing { usearch_dense_get(handle, key.storage, &span, $0) }
    }

    /// Writes the index under a path, in the layout its extension picks - one `.safegraphs`
    /// monolith, or a `.safegraphs.json` manifest beside its shards. Demands a mutation-free
    /// window; concurrent readers keep searching throughout.
    public func save(executor: Executor, path: borrowing String) throws(Error) {
        try withBorrowed(path) { (borrowed) throws(Error) in
            try throwing { usearch_dense_save(handle, executor.raw, borrowed, nil, nil, $0) }
        }
    }

    /// Reads a checkpoint into this handle, shard by shard, with the handle's residency deciding
    /// the behavior: materialize into RAM, upload to the devices, or open in place on disk. A
    /// handle whose geometry was left zeroed freezes it here, from the checkpoint's own headers.
    public func load(executor: Executor, path: borrowing String) throws(Error) {
        try withBorrowed(path) { (borrowed) throws(Error) in
            try throwing { usearch_dense_load(handle, executor.raw, borrowed, nil, nil, $0) }
        }
    }

    /// Writes the index into a caller-owned buffer as one monolith. Size the buffer with
    /// `storageRequirements`.
    public func save(executor: Executor, buffer: UnsafeMutableRawBufferPointer) throws(Error) {
        guard let base = buffer.baseAddress, buffer.count != 0 else {
            throw fail(.shapeMismatch, "the buffer is empty")
        }
        let span = usearch_bytes_span_t(data: base.assumingMemoryBound(to: UInt8.self), length: buffer.count)
        try throwing { usearch_dense_save_buffer(handle, executor.raw, span, nil, nil, $0) }
    }

    /// Reads a monolith from a buffer into this handle, with the residency deciding the behavior
    /// like `load(executor:path:)`. A residency that opens in place demands the buffer outlive
    /// the handle.
    public func load(executor: Executor, buffer: UnsafeRawBufferPointer) throws(Error) {
        guard let base = buffer.baseAddress, buffer.count != 0 else {
            throw fail(.malformedInput, "the buffer is empty")
        }
        let view = usearch_bytes_view_t(data: base.assumingMemoryBound(to: UInt8.self), length: buffer.count)
        try throwing { usearch_dense_load_buffer(handle, executor.raw, view, nil, nil, $0) }
    }

    /// Opens a consistent snapshot cursor over the current contents, for export or migration.
    ///
    /// The watermark is captured here, so entries published after this call never appear
    /// mid-iteration.
    public func cursor() throws(Error) -> DenseCursor {
        let opened = try throwing { usearch_dense_cursor(handle, $0) }
        guard let opened else { throw fail(.internalFailure, "the library returned no cursor and no error") }
        return DenseCursor(handle: opened, index: self)
    }

    /// Opens a resumable nearest-neighbor stream over one query row, so a caller that post-filters
    /// aggressively pulls until satisfied rather than guessing a wanted count upfront.
    ///
    /// A live finder pins one of the executor's threads until it is released, so an executor with
    /// few threads and many open finders starves; drop each one the moment its caller stops
    /// pulling rather than at the end of the enclosing scope.
    public func find<Element: ScalarValue>(
        executor: Executor,
        query: borrowing MatrixView<Element>
    ) throws(Error) -> DenseFinder {
        try query.validate(dimensions: try frozenDimensions())
        guard query.rows == 1 else { throw fail(.shapeMismatch, "a finder streams one query row") }
        var view = query.lowered
        let opened = try throwing { usearch_dense_find(handle, executor.raw, &view, $0) }
        guard let opened else { throw fail(.internalFailure, "the library returned no finder and no error") }
        return DenseFinder(handle: opened, index: self, executor: executor)
    }

    // The filtered search hands the engine a pointer to this box rather than to a Swift closure,
    // which has no stable C representation.
    private final class AdmissionBox {
        let admits: (Key) -> Bool
        init(_ admits: @escaping (Key) -> Bool) { self.admits = admits }
    }

    private func searchDispatch<Element: ScalarValue>(
        executor: Executor,
        queries: borrowing MatrixView<Element>,
        matches: MatchesSpan,
        options: SearchOptions,
        admits: ((Key) -> Bool)?
    ) throws(Error) -> TraversalStats {
        try queries.validate(dimensions: try frozenDimensions())
        guard matches.count == queries.rows else {
            throw fail(.shapeMismatch, "the outputs are not sized for one row per query")
        }
        var view = queries.lowered
        var lowered = options.lowered
        var span = matches.lowered
        var rawStats = usearch_dense_traversal_stats_t()
        var failure = usearch_error_t()
        if let admits {
            let box = AdmissionBox(admits)
            withExtendedLifetime(box) {
                let state = Unmanaged.passUnretained(box).toOpaque()
                usearch_dense_search_if(
                    handle,
                    executor.raw,
                    &view,
                    &span,
                    &lowered,
                    DenseIndex.usearchDenseAdmits,
                    state,
                    nil,
                    nil,
                    &rawStats,
                    &failure
                )
            }
        }
        else {
            usearch_dense_search(handle, executor.raw, &view, &span, &lowered, nil, nil, &rawStats, &failure)
        }
        try check(failure)
        return TraversalStats(rawStats)
    }

    /// The admission trampoline the engine calls from the executor's threads. The boxed closure
    /// cannot throw, so nothing unwinds out of this frame and into C.
    private static let usearchDenseAdmits: usearch_predicate_t = { key, state in
        // A lost box means the trampoline is broken. Admitting would turn a filtered search into
        // an unfiltered one and answer wrong rows without saying so.
        guard let state else { return usearch_reject_k }
        let box = Unmanaged<AdmissionBox>.fromOpaque(state).takeUnretainedValue()
        return box.admits(Key(raw: key)) ? usearch_admit_k : usearch_reject_k
    }

    /// The frozen row width, refusing the verbs whose geometry a later load still decides.
    private func frozenDimensions() throws(Error) -> Int {
        let dimensions = try self.dimensions
        guard dimensions != 0 else {
            throw fail(.uninitialized, "the index geometry is unfrozen; reserve or load a checkpoint first")
        }
        return dimensions
    }

    /// Pairs a probe's reading with its error slot.
    private func measure(
        _ probe: (usearch_dense_index_t?, UnsafeMutablePointer<usearch_error_t>?) -> Int
    ) throws(Error) -> Int {
        return try throwing { probe(handle, $0) }
    }
}

// MARK: - Cursors & Finders

/// A consistent snapshot cursor over an index's contents; see `DenseIndex.cursor()`.
///
/// The walk is a sequential row scan, so it stays a batch operation - export, migration, and
/// visualization all move whole batches.
public final class DenseCursor {
    /// The native handle, released by `deinit` and never reassigned.
    private let handle: usearch_dense_cursor_t
    /// The index the snapshot was taken of, held so the cursor cannot outlive it.
    private let index: DenseIndex

    init(handle: usearch_dense_cursor_t, index: DenseIndex) {
        self.handle = handle
        self.index = index
    }

    deinit { usearch_dense_cursor_free(handle) }

    /// Copies the next batch of keys into `keys`, walking the key column alone; the rows written
    /// come back, and zero means the snapshot is exhausted.
    public func next(into keys: UnsafeMutableBufferPointer<Key>) throws(Error) -> Int {
        try advance(keys: keys, vectors: nil)
    }

    /// Copies the next batch of keys and the vectors they identify. The key run bounds the batch,
    /// so the matrix must have room for at least as many rows.
    public func next<Element: ScalarValue>(
        into keys: UnsafeMutableBufferPointer<Key>,
        vectors: borrowing MatrixSpan<Element>
    ) throws(Error) -> Int {
        try advance(keys: keys, vectors: vectors.lowered)
    }

    /// Advances the snapshot by one batch, with or without the vector export.
    private func advance(
        keys: UnsafeMutableBufferPointer<Key>,
        vectors: usearch_matrix_span_t?
    ) throws(Error) -> Int {
        let handle = handle
        return try keys.withMemoryRebound(to: usearch_key_t.self) { (rawKeys) throws(Error) -> Int in
            let run = usearch_keys_span_t(data: rawKeys.baseAddress, capacity: rawKeys.count)
            guard var span = vectors else {
                return try throwing { usearch_dense_cursor_next(handle, run, nil, $0) }
            }
            return try throwing { usearch_dense_cursor_next(handle, run, &span, $0) }
        }
    }
}

/// A resumable nearest-neighbor stream over one query; see `DenseIndex.find(executor:query:)`.
///
/// Best-first traversal is naturally pausable, so the stream owns a live frontier and visited set
/// and yields matches lazily - at the price of pinning one of the executor's threads for as long
/// as it lives. Ordering beyond the expansion horizon is best-effort nearest rather than globally
/// exact, so pulling far past that breadth degrades the ranking rather than ending the stream.
public final class DenseFinder: Sequence, IteratorProtocol {
    /// Matches pulled per crossing into the library, amortising the call over a run of them.
    private static let matchesPerPull = 64

    /// The native handle, released by `deinit` and never reassigned.
    private let handle: usearch_dense_finder_t
    /// The index being walked, held so the stream cannot outlive it.
    private let index: DenseIndex
    /// The executor whose thread the stream pins, held for the same reason.
    private let executor: Executor
    /// The keys of the batch last pulled, sized once at construction.
    private var keys: [Key]
    /// The distances of that batch, positionally paired with `keys`.
    private var distances: [Distance]
    /// How many of those slots the last pull filled.
    private var filled: Int
    /// How many of the filled slots `next()` has already yielded.
    private var position: Int
    /// Whether the traversal reported itself drained, or a pull failed.
    private var drained: Bool
    /// The failure that ended iteration early, and nil when the stream simply ran out.
    public private(set) var failure: Error?

    init(handle: usearch_dense_finder_t, index: DenseIndex, executor: Executor) {
        self.handle = handle
        self.index = index
        self.executor = executor
        keys = [Key](repeating: .nilKey, count: DenseFinder.matchesPerPull)
        distances = [Distance](repeating: 0, count: DenseFinder.matchesPerPull)
        filled = 0
        position = 0
        drained = false
    }

    deinit { usearch_dense_find_free(handle) }

    /// The next-nearest match, or nil once the traversal is drained. A failed pull ends iteration
    /// and lands in `failure`, since an iterator cannot throw.
    public func next() -> Match? {
        if position == filled {
            guard !drained, refill() else { return nil }
        }
        position += 1
        return Match(key: keys[position - 1], distance: distances[position - 1])
    }

    /// Pulls one batch into the buffers, reporting whether anything landed.
    private func refill() -> Bool {
        let handle = handle
        do {
            filled = try keys.withUnsafeMutableBufferPointer { (keysBuffer) throws(Error) -> Int in
                try keysBuffer.withMemoryRebound(to: usearch_key_t.self) { (rawKeys) throws(Error) -> Int in
                    try distances.withUnsafeMutableBufferPointer { (scores) throws(Error) -> Int in
                        let run = usearch_keys_span_t(data: rawKeys.baseAddress, capacity: rawKeys.count)
                        return try throwing { usearch_dense_find_next(handle, run, scores.baseAddress, $0) }
                    }
                }
            }
        }
        catch {
            failure = error
            filled = 0
        }
        position = 0
        drained = filled == 0
        return filled != 0
    }
}

// MARK: - Helpers

/// Views a Swift string as the pointer-and-length pair every path argument takes. The string is
/// never terminated, by contract, so nothing is copied past the UTF-8 encoding itself.
func withBorrowed<Value>(
    _ text: borrowing String,
    _ body: (usearch_bytes_view_t) throws(Error) -> Value
) throws(Error) -> Value {
    var text = copy text
    var outcome: Result<Value, Error>?
    text.withUTF8 { buffer in
        outcome = Result { () throws(Error) -> Value in
            try body(usearch_bytes_view_t(data: buffer.baseAddress, length: buffer.count))
        }
    }
    guard let outcome else { throw fail(.internalFailure, "the path was never presented to the call") }
    return try outcome.get()
}
