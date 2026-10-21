//
//  DenseIndex.swift
//  USearch
//
//  Vector Search over equidimensional dense embeddings.
//
//  Every verb is a batch over strided rows, so a memory-mapped dataset reaches the index
//  with no reshape and no copy, and a single vector is a one-row matrix. The breadth of the
//  candidates rides each call rather than the handle, which lets a recall sweep run against one
//  immutable index with no mutation between points.
//

import NumKong

import USearchC

// MARK: - Options

/// A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
/// capabilities, carrying the C enumerator itself. All of them derive from a single inner-product
/// kernel with pre-computed magnitudes; hamming and jaccard are bit-level, over the packed
/// sub-byte kinds.
///
/// Arbitrary metrics belong to the C++ baseline tier, which forfeits the SIMD kernel chains and
/// the magnitude cache both.
public struct DenseMetric: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: usearch_dense_metric_t

    /// No kind named - an unset field, refused by every constructor.
    public static let unknown = DenseMetric(raw: usearch_dense_metric_unknown_k)
    /// The inner product.
    public static let inner = DenseMetric(raw: usearch_dense_metric_inner_k)
    /// The cosine distance.
    public static let angular = DenseMetric(raw: usearch_dense_metric_angular_k)
    /// The L2 distance.
    public static let euclidean = DenseMetric(raw: usearch_dense_metric_euclidean_k)
    /// The count of differing bits.
    public static let hamming = DenseMetric(raw: usearch_dense_metric_hamming_k)
    /// The bit-level jaccard distance.
    public static let jaccard = DenseMetric(raw: usearch_dense_metric_jaccard_k)

    /// The metric rendered as the library spells it. Calls `usearch_dense_metric_name`.
    public var description: String { String(cString: usearch_dense_metric_name(raw)) }

    /// Reads a spelling back, and nil for one no metric carries. Calls
    /// `usearch_dense_metric_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_dense_metric_from_name, usearch_dense_metric_name) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_dense_metric_t) { self.raw = raw }
}

/// The construction options of a dense index, as `initRam` and `initVram` freeze them.
///
/// Stored rows always round up to the SIMD step quantum, so rows stay cache-line aligned and
/// every kernel runs tail-free; the arena is slightly larger for it, and that is not a knob.
public struct DenseOptions: Sendable {
    /// The width of the indexed vectors; zero leaves the geometry unfrozen, so the index admits
    /// no rows.
    public var dimensions: Int
    /// The distance kind used between vectors.
    public var metric: DenseMetric
    /// The scalar type vectors are stored and quantized into.
    public var quantization: Scalar
    /// The bound on connections per node in the graph; zero derives.
    public var connectivity: Int
    /// The bound on entries sharing one key; zero derives to one - unique keys.
    public var valuesPerKey: Int

    /// An angular bf16 index, with every optional field derived.
    public init(
        dimensions: Int = 0,
        metric: DenseMetric = .angular,
        quantization: Scalar = .bf16,
        connectivity: Int = 0,
        valuesPerKey: Int = 0
    ) {
        self.dimensions = dimensions
        self.metric = metric
        self.quantization = quantization
        self.connectivity = connectivity
        self.valuesPerKey = valuesPerKey
    }
}

/// The geometry and metric a checkpoint froze - everything a reader learns without opening it,
/// and nothing a reader must choose for itself.
///
/// A checkpoint records neither the placement nor the bound on entries sharing one key: the first
/// says where it was written rather than where a reader must keep it, and the second bounds a live
/// handle's mutable state rather than the serialized form. Both stay the caller's to choose when
/// loading.
public struct DenseMetadata: Sendable {
    /// The scalars per stored vector.
    public var dimensions: Int
    /// The distance kind the checkpoint scores with.
    public var metric: DenseMetric
    /// The scalar type its rows are stored and quantized into.
    public var quantization: Scalar
    /// The connections-per-node bound the graph was built with.
    public var connectivity: Int

    init(_ raw: usearch_dense_metadata_t) {
        dimensions = raw.dimensions
        metric = DenseMetric(raw: raw.metric)
        quantization = Scalar(raw: raw.quantization)
        connectivity = raw.connectivity
    }
}

/// A live index's frozen metadata and current counters, each read once, so under concurrent
/// insertions they may come from slightly different moments.
public struct DenseIndexDescription: Sendable {
    /// The frozen metric, quantization, dimensions and connectivity, as a checkpoint reports them.
    public var metadata: DenseMetadata
    /// The stored entries.
    public var size: Int
    /// The entries hosted without growing again.
    public var capacity: Int
    /// The bytes the handle holds; traversal scratch is the executor's.
    public var memoryUsage: Int
    /// The capability codename the index dispatched to on this machine.
    public var capabilities: String

    init(_ raw: usearch_dense_index_description_t) {
        metadata = DenseMetadata(raw.metadata)
        size = raw.size
        capacity = raw.capacity
        memoryUsage = raw.memory_usage
        capabilities = raw.capabilities.map { String(cString: $0) } ?? ""
    }
}

/// One reachability audit: the population it judged, and how much of it a search reaches.
public struct DenseReachability: Equatable, Sendable {
    /// The entries published when the audit started; compare `reachable` against this, never a
    /// later size.
    public var audited: Int
    /// Of those, the entries a search reaches from the landmarks; the rest no search finds until
    /// a refinement re-links them.
    public var reachable: Int
    /// The levels the sweep took to converge.
    public var sweeps: Int
    /// The levels the sweep back toward the entry point took, where an audit sweeps both ways.
    public var backSweeps: Int

    init(_ raw: usearch_reachability_t) {
        audited = raw.audited
        reachable = raw.reachable
        sweeps = raw.sweeps
        backSweeps = raw.back_sweeps
    }
}

// MARK: - Per-Call Configs & Telemetry

/// The traversal telemetry one call reports, passed in a config and written by the call. Every
/// call overwrites it rather than accumulating, so a sweep over many batches sums them itself,
/// and it belongs to one call until that call is final.
public final class DenseTraversalStats: @unchecked Sendable {
    /// The C counters the call writes, at an address that holds for a deferred call.
    let raw: UnsafeMutablePointer<usearch_traversal_stats_t>

    public init() {
        raw = .allocate(capacity: 1)
        raw.initialize(to: usearch_traversal_stats_t())
    }

    deinit { raw.deallocate() }

    /// The graph members the call visited.
    public var visitedMembers: Int { raw.pointee.visited_members }
    /// The distances the call computed, diversity re-selections included.
    public var computedDistances: Int { raw.pointee.computed_distances }
    /// The starvation re-seedings - a graph-connectivity health signal rather than a cost
    /// counter.
    public var restarts: Int { raw.pointee.restarts }
}

/// Flash Attention's running maximum and partition for one query row.
///
/// The two are one concept: a maximum without its partition is a number nothing can rescale, so
/// they share an element rather than travelling as two arrays a caller could supply half of.
/// Carrying them is what folds one attention call into another over the same queries.
public struct DenseSoftmaxState: Equatable, Sendable {
    /// The largest logit seen so far - the recurrence's `m`.
    public var maximumScore: Float = 0
    /// The sum of exponentials taken against that maximum - the recurrence's `l`.
    public var partition: Float = 0

    public init(maximumScore: Float = 0, partition: Float = 0) {
        self.maximumScore = maximumScore
        self.partition = partition
    }
}

/// How `attend` resolves scored keys to value rows: a key names its row of `values`, whose rows
/// must be as wide as the keys and stored in the index's own quantization.
///
/// The index stores no values, so only the keys a walk scores are ever resolved. A key without a
/// row, or values of another width or scalar kind than the library reads, refuses the call with
/// `shapeMismatch`, since a missing row is a wrong softmax rather than a smaller one.
public struct ValueLookup: Sendable {
    /// The value rows, which must outlive the call.
    public var values: RowsView
    /// The row of `values` holding a key's value, or nil for none. It runs on the executor's
    /// threads together, so it must be safe for concurrent use; the first error it throws leaves
    /// that key without a row and is what the call throws.
    public var row: @Sendable (Key) throws -> Int?

    public init(values: RowsView, row: @escaping @Sendable (Key) throws -> Int?) {
        self.values = values
        self.row = row
    }
}

/// The knobs of `insert`. Every zeroed field derives, which is what almost every caller wants.
public struct DenseInsertConfig {
    /// The breadth of the candidates the insertion walk searches for neighbors with; zero derives it
    /// from the geometry.
    public var expansionFactor: Int
    /// Called with rows admitted; it reports and cannot cancel.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?

    public init(
        expansionFactor: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil
    ) {
        self.expansionFactor = expansionFactor
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
    }
}

/// The knobs of `refine`. Every zeroed field derives.
public struct DenseRefineConfig {
    /// The breadth of the candidates the re-selection walk explores; zero derives it from the geometry.
    public var expansionFactor: Int
    /// Called with nodes swept; a stop leaves the graph correct and merely less improved.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?

    public init(
        expansionFactor: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil
    ) {
        self.expansionFactor = expansionFactor
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
    }
}

/// The knobs of `search`. Every zeroed field derives.
public struct DenseSearchConfig {
    /// The breadth of the candidates the walk explores; zero derives it from the geometry, and a
    /// scan ignores it.
    public var expansionFactor: Int
    /// Walks the graph by default; `.scan` scores every row exactly, on RAM and NVMe handles, and
    /// a VRAM handle refuses it with `unsupported`.
    public var traversal: Traversal
    /// Narrows the matches to the keys it approves. Non-matching entries still route the
    /// traversal, so recall does not collapse at low selectivity. It runs on the executor's
    /// threads together, so it must be safe for concurrent use; an error it throws never unwinds
    /// into a C frame, but rejects that key, and the first one is what the call throws. A VRAM
    /// handle refuses one with `device`, since host code cannot narrow a device walk.
    public var predicate: (@Sendable (Key) throws -> Bool)?
    /// Called with queries resolved.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?

    public init(
        expansionFactor: Int = 0,
        traversal: Traversal = .walk,
        predicate: (@Sendable (Key) throws -> Bool)? = nil,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil
    ) {
        self.expansionFactor = expansionFactor
        self.traversal = traversal
        self.predicate = predicate
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
    }
}

/// The knobs of `maxsim`. Every zeroed field derives.
public struct DenseMaxsimConfig {
    /// The breadth of the candidates; zero derives it from the geometry.
    public var expansionFactor: Int
    /// Called with queries resolved.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?

    public init(
        expansionFactor: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil
    ) {
        self.expansionFactor = expansionFactor
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
    }
}

/// The knobs of `multiply`. Every zeroed field derives.
public struct DenseMultiplyConfig {
    /// The bound on pairs kept per row, the first ones its walk scored; zero keeps every pair.
    public var wanted: Int
    /// The breadth of the candidates, which widens the emitted pattern; zero derives.
    public var expansionFactor: Int
    /// Called with rows resolved.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?

    public init(
        wanted: Int = 0,
        expansionFactor: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil
    ) {
        self.wanted = wanted
        self.expansionFactor = expansionFactor
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
    }
}

/// The knobs of `attend`. Every zeroed field derives.
public struct DenseAttendConfig {
    /// The query heads sharing the index's key-value head, and with it one walk per token; it must
    /// divide the query rows, and zero means one.
    public var headCount: Int
    /// The softmax state to write, one per query row, which must outlive the call; nil skips it.
    public var carry: UnsafeMutableBufferPointer<DenseSoftmaxState>?
    /// The breadth of the candidates; zero derives it from the geometry. Wider candidates walk further,
    /// folding more of the tail.
    public var expansionFactor: Int
    /// The logit scale, conventionally the inverse square root of the head dimensionality; zero
    /// derives it from the index's frozen geometry.
    public var scoreScale: Float
    /// Called with queries resolved.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?

    public init(
        headCount: Int = 0,
        carry: UnsafeMutableBufferPointer<DenseSoftmaxState>? = nil,
        expansionFactor: Int = 0,
        scoreScale: Float = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil
    ) {
        self.headCount = headCount
        self.carry = carry
        self.expansionFactor = expansionFactor
        self.scoreScale = scoreScale
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
    }
}

/// The knobs of `absorb`.
public struct DenseAbsorbConfig {
    /// Called with donors absorbed.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(progress: Progress? = nil, progressSteps: Int = 0) {
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// The knobs of `save`.
public struct DenseSaveConfig {
    /// Called with shards written.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(progress: Progress? = nil, progressSteps: Int = 0) {
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// The knobs of `saveBuffer`.
public struct DenseSaveBufferConfig {
    /// Called with bytes written.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(progress: Progress? = nil, progressSteps: Int = 0) {
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// The knobs of `searchStream`.
public struct DenseSearchStreamInitConfig {
    /// The breadth of the candidates the walk explores; zero derives it from the geometry.
    public var expansionFactor: Int

    public init(expansionFactor: Int = 0) { self.expansionFactor = expansionFactor }
}

/// The knobs of `loadRam`.
public struct DenseLoadRamConfig {
    /// The executor the read fans out over, one shard per thread; nil reads on the calling thread.
    public var executor: Executor?
    /// The memory domains to place the parts on, part `p` landing on shard `p` modulo their count;
    /// empty derives every node the process may allocate on.
    public var memoryDomains: [Int32]
    /// The bound on entries sharing one key, which no checkpoint records; zero derives to one.
    public var valuesPerKey: Int
    /// Called with shards placed.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(
        executor: Executor? = nil,
        memoryDomains: [Int32] = [],
        valuesPerKey: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) {
        self.executor = executor
        self.memoryDomains = memoryDomains
        self.valuesPerKey = valuesPerKey
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// The knobs of `loadVram`.
public struct DenseLoadVramConfig {
    /// The executor the upload runs on, one stream per device; nil uploads from the calling thread.
    public var executor: Executor?
    /// The devices to place the parts on, part `p` landing on device `p` modulo their count; empty
    /// derives one free device.
    public var devices: [Int32]
    /// The bound on entries sharing one key, which no checkpoint records; zero derives to one.
    public var valuesPerKey: Int
    /// Called with shards placed.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(
        executor: Executor? = nil,
        devices: [Int32] = [],
        valuesPerKey: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) {
        self.executor = executor
        self.devices = devices
        self.valuesPerKey = valuesPerKey
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// The knobs of `loadRamBuffer`.
public struct DenseLoadRamBufferConfig {
    /// The executor the copy fans out over; nil copies on the calling thread.
    public var executor: Executor?
    /// The memory domains to place shards on; the monolith lands whole on the first.
    public var memoryDomains: [Int32]
    /// The bound on entries sharing one key, which no checkpoint records; zero derives to one.
    public var valuesPerKey: Int
    /// Called with bytes placed.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(
        executor: Executor? = nil,
        memoryDomains: [Int32] = [],
        valuesPerKey: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) {
        self.executor = executor
        self.memoryDomains = memoryDomains
        self.valuesPerKey = valuesPerKey
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// The knobs of `loadVramBuffer`.
public struct DenseLoadVramBufferConfig {
    /// The executor the upload runs on, one stream per device; nil uploads from the calling thread.
    public var executor: Executor?
    /// The devices to place shards on; the monolith lands whole on the first.
    public var devices: [Int32]
    /// The bound on entries sharing one key, which no checkpoint records; zero derives to one.
    public var valuesPerKey: Int
    /// Called with bytes placed.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(
        executor: Executor? = nil,
        devices: [Int32] = [],
        valuesPerKey: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) {
        self.executor = executor
        self.devices = devices
        self.valuesPerKey = valuesPerKey
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// The knobs of `viewNvme`.
public struct DenseViewNvmeConfig {
    /// The NVMe executor whose rings read the checkpoint's headers; nil reads on the calling thread.
    public var executor: Executor?
    /// The devices the parts are served from - a filesystem path, a transport identifier, or a
    /// FUSE mount over object storage; empty serves each part where its manifest points.
    public var paths: [String]

    public init(executor: Executor? = nil, paths: [String] = []) {
        self.executor = executor
        self.paths = paths
    }
}

// MARK: - Results

/// The rows `get` or `attend` wrote into an arena, one per entry sharing the key or per query, read
/// in place as `Value`. Mirrors `usearch_rows_view_t`.
public struct DenseVectors<Value: ScalarValue>: RandomAccessCollection {
    /// The arena the rows live in, and which call wrote them.
    let write: ArenaWrite
    /// The packed rows the call wrote.
    let vectors: usearch_rows_view_t
    /// One past the last row the call wrote.
    public let endIndex: Int

    /// The first row's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// The bytes between consecutive rows, as the library reported them.
    public var stride: Int { vectors.stride }

    /// The logical values in each row, as the library reported them.
    public var columns: Int { vectors.columns }

    /// One row's stored values, packed at their natural width.
    public subscript(row: Int) -> Row {
        write.check(row, below: endIndex)
        let perValue = Value.scalarKind.dimensionsPerValue
        return Row(
            write: write,
            base: vectors.data! + row * vectors.stride,
            endIndex: (columns + perValue - 1) / perValue
        )
    }

    /// One row of the arena's values, read in place.
    public struct Row: RandomAccessCollection {
        /// The arena the row lives in, and which call wrote it.
        let write: ArenaWrite
        /// The row's first value.
        let base: UnsafeRawPointer
        /// One past the last value in the row.
        public let endIndex: Int

        /// The first value's position, which a collection walk starts from.
        public var startIndex: Int { 0 }

        /// The value at a position.
        public subscript(position: Int) -> Value {
            write.check(position, below: endIndex)
            return base.load(fromByteOffset: position * MemoryLayout<Value>.stride, as: Value.self)
        }
    }
}

/// The keys a cursor pull wrote into an arena, in walk order. Mirrors `usearch_keys_view_t`.
public struct DenseKeys: RandomAccessCollection {
    /// The arena the keys live in, and which call wrote them.
    let write: ArenaWrite
    /// The keys the call wrote.
    let keys: usearch_keys_view_t
    /// One past the last key the call wrote.
    public let endIndex: Int

    /// The first key's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// The key at a position.
    public subscript(position: Int) -> Key {
        write.check(position, below: endIndex)
        return Key(
            raw: UnsafeRawPointer(keys.data!).load(fromByteOffset: position * keys.stride, as: usearch_key_t.self)
        )
    }
}

/// The keys each entry of a cursor pull links to, wrote into an arena as a ragged run and read in
/// place. Mirrors `usearch_keys_tape_view_t`.
public struct DenseLinks: RandomAccessCollection {
    /// The arena the keys live in, and which call wrote them.
    let write: ArenaWrite
    /// The linked keys and offsets the call wrote.
    let links: usearch_keys_tape_view_t
    /// One past the last entry the pull wrote.
    public let endIndex: Int

    /// The first entry's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// The keys one entry links to, as the graph stored them when the pull ran.
    public subscript(entry: Int) -> [Key] {
        write.check(entry, below: endIndex)
        guard let offsets = links.offsets, let keys = links.keys else { return [] }
        return (offsets[entry] ..< offsets[entry + 1]).map { Key(raw: keys[$0]) }
    }
}

/// The ragged sparse product `multiply` wrote into an arena: per left row, every pair its walk
/// scored, in traversal order; or the keys `maxsim` kept per query, largest score first; read in
/// place. Mirrors `usearch_matches_tape_view_t`.
public struct DenseProducts: RandomAccessCollection {
    /// The arena the pairs live in, and which call wrote them.
    let write: ArenaWrite
    /// The keys, products and offsets the call wrote.
    let products: usearch_matches_tape_view_t
    /// One past the last left row or query.
    public let endIndex: Int
    /// Pairs the walks scored across every row, or keys scored across every query, before the
    /// wanted bound cut any.
    public let required: Int

    /// The first row's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// One left row's pairs, in traversal order.
    public subscript(row: Int) -> MatchesRow {
        write.check(row, below: endIndex)
        let cells = products.offsets[row] ..< products.offsets[row + 1]
        return MatchesRow(write: write, keys: products.keys, scores: products.scores, cells: cells)
    }
}

/// The monolith `saveBuffer` wrote into an arena, read in place. Mirrors `usearch_bytes_view_t`.
public struct Monolith {
    /// The arena the bytes live in, and which call wrote them.
    let write: ArenaWrite
    /// The bytes the call wrote.
    let monolith: usearch_bytes_view_t

    /// The bytes in the monolith.
    public var count: Int { monolith.length }

    /// Lends the serialized checkpoint to `body`, which must not keep the buffer past its return.
    public func withUnsafeBytes<Result, Failure: Swift.Error>(
        _ body: (UnsafeRawBufferPointer) throws(Failure) -> Result
    ) throws(Failure) -> Result {
        write.check()
        return try body(UnsafeRawBufferPointer(start: monolith.data, count: monolith.length))
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
///
/// Every verb called on the index runs on the calling thread and blocks until its outcome is
/// final; `on(_:)` runs the same verbs on an executor, and `deferred(_:)` submits them there.
public final class DenseIndex: @unchecked Sendable {
    /// The native handle, released by `deinit`, and null once an absorb consumed it, so every verb
    /// sees the one the lifecycle verb claimed and the engine's own synchronization is all that
    /// sharing relies on.
    var raw: usearch_dense_index_t?
    /// Where the payload lives, as the lifecycle verb that created the handle named it.
    let residency: Residency

    /// Adopts a handle a lifecycle verb just created.
    private init(raw: usearch_dense_index_t, residency: Residency) {
        self.raw = raw
        self.residency = residency
    }

    /// Calls `usearch_dense_free`.
    deinit { usearch_dense_free(raw) }

    /// Initializes an empty, mutable index in host memory, one shard per memory domain; an empty
    /// list derives every node the process may allocate on. Calls `usearch_dense_init_ram`.
    public static func initRam(
        _ options: DenseOptions,
        memoryDomains: [Int32] = []
    ) throws(USearchError) -> DenseIndex {
        let landing = Landing()
        let index = landing.slots(usearch_dense_index_t?.none)
        try landing.lend(memoryDomains) { (domains) throws(USearchError) in
            var arguments = usearch_dense_init_ram_t(
                index: index,
                outcome: landing.outcome,
                metric: options.metric.raw,
                quantization: options.quantization.raw,
                dimensions: options.dimensions,
                connectivity: options.connectivity,
                values_per_key: options.valuesPerKey,
                ram: usearch_ram_options_t(memory_domains: domains, memory_domains_count: memoryDomains.count)
            )
            usearch_dense_init_ram(&arguments)
        }
        return try adopted(index, landing, .ram)
    }

    /// Initializes an empty, mutable index in device memory, sharded across the named devices and
    /// never replicated; an empty list derives one free device. Calls `usearch_dense_init_vram`.
    public static func initVram(_ options: DenseOptions, devices: [Int32] = []) throws(USearchError) -> DenseIndex {
        let landing = Landing()
        let index = landing.slots(usearch_dense_index_t?.none)
        try landing.lend(devices) { (lentDevices) throws(USearchError) in
            var arguments = usearch_dense_init_vram_t(
                index: index,
                outcome: landing.outcome,
                metric: options.metric.raw,
                quantization: options.quantization.raw,
                dimensions: options.dimensions,
                connectivity: options.connectivity,
                values_per_key: options.valuesPerKey,
                vram: usearch_vram_options_t(devices: lentDevices, devices_count: devices.count)
            )
            usearch_dense_init_vram(&arguments)
        }
        return try adopted(index, landing, .vram)
    }

    /// Copies a checkpoint into a new, mutable index in host memory. The geometry and metric come
    /// from the checkpoint's own headers, and part `p` lands whole on shard `p` modulo the shard
    /// count, wherever it was written from.
    ///
    /// A cancelled or failed load creates no index. Calls `usearch_dense_load_ram`.
    public static func loadRam(
        _ path: String,
        config: DenseLoadRamConfig = DenseLoadRamConfig()
    ) throws(USearchError) -> DenseIndex {
        let landing = Landing()
        let index = landing.slots(usearch_dense_index_t?.none)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        try landing.lend(path) { (path) throws(USearchError) in
            try landing.lend(config.memoryDomains) { (domains) throws(USearchError) in
                var arguments = usearch_dense_load_ram_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: landing.outcome,
                    completion: landing.completion,
                    path: path,
                    ram: usearch_ram_options_t(
                        memory_domains: domains,
                        memory_domains_count: config.memoryDomains.count
                    ),
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_dense_load_ram(&arguments)
            }
        }
        return try adopted(index, landing, .ram)
    }

    /// Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
    /// `loadRam`. Calls `usearch_dense_load_vram`.
    public static func loadVram(
        _ path: String,
        config: DenseLoadVramConfig = DenseLoadVramConfig()
    ) throws(USearchError) -> DenseIndex {
        let landing = Landing()
        let index = landing.slots(usearch_dense_index_t?.none)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        try landing.lend(path) { (path) throws(USearchError) in
            try landing.lend(config.devices) { (devices) throws(USearchError) in
                var arguments = usearch_dense_load_vram_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: landing.outcome,
                    completion: landing.completion,
                    path: path,
                    vram: usearch_vram_options_t(devices: devices, devices_count: config.devices.count),
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_dense_load_vram(&arguments)
            }
        }
        return try adopted(index, landing, .vram)
    }

    /// Copies a monolith from a buffer into a new index in host memory, as `loadRam` does a path.
    /// Calls `usearch_dense_load_ram_buffer`.
    public static func loadRamBuffer(
        _ buffer: RawSpan,
        config: DenseLoadRamBufferConfig = DenseLoadRamBufferConfig()
    ) throws(USearchError) -> DenseIndex {
        let landing = Landing()
        let index = landing.slots(usearch_dense_index_t?.none)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        try buffer.withUnsafeBytes { (bytes) throws(USearchError) in
            try landing.lend(config.memoryDomains) { (domains) throws(USearchError) in
                var arguments = usearch_dense_load_ram_buffer_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: landing.outcome,
                    completion: landing.completion,
                    buffer: usearch_bytes_view_t(bytes),
                    ram: usearch_ram_options_t(
                        memory_domains: domains,
                        memory_domains_count: config.memoryDomains.count
                    ),
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_dense_load_ram_buffer(&arguments)
            }
        }
        return try adopted(index, landing, .ram)
    }

    /// Copies a monolith from a buffer into a new index in device memory, as `loadVram` does a
    /// path. Calls `usearch_dense_load_vram_buffer`.
    public static func loadVramBuffer(
        _ buffer: RawSpan,
        config: DenseLoadVramBufferConfig = DenseLoadVramBufferConfig()
    ) throws(USearchError) -> DenseIndex {
        let landing = Landing()
        let index = landing.slots(usearch_dense_index_t?.none)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        try buffer.withUnsafeBytes { (bytes) throws(USearchError) in
            try landing.lend(config.devices) { (devices) throws(USearchError) in
                var arguments = usearch_dense_load_vram_buffer_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: landing.outcome,
                    completion: landing.completion,
                    buffer: usearch_bytes_view_t(bytes),
                    vram: usearch_vram_options_t(devices: devices, devices_count: config.devices.count),
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_dense_load_vram_buffer(&arguments)
            }
        }
        return try adopted(index, landing, .vram)
    }

    /// Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
    /// verbs report `readOnly`; the files must stay unmodified while it lives. Calls
    /// `usearch_dense_view_ram`.
    public static func viewRam(_ path: String) throws(USearchError) -> DenseIndex {
        let landing = Landing()
        let index = landing.slots(usearch_dense_index_t?.none)
        try landing.lend(path) { (path) throws(USearchError) in
            var arguments = usearch_dense_view_ram_t(index: index, outcome: landing.outcome, path: path)
            usearch_dense_view_ram(&arguments)
        }
        return try adopted(index, landing, .ram)
    }

    /// Serves a checkpoint in place from NVMe drives, read through the executor's submission rings,
    /// as a read-only index. Until NVMe checkpoints carry keys, it reports `unsupported`. Calls
    /// `usearch_dense_view_nvme`.
    public static func viewNvme(
        _ path: String,
        config: DenseViewNvmeConfig = DenseViewNvmeConfig()
    ) throws(USearchError) -> DenseIndex {
        let landing = Landing()
        let index = landing.slots(usearch_dense_index_t?.none)
        try landing.lend(path) { (path) throws(USearchError) in
            withPaths(config.paths) { nvme in
                var arguments = usearch_dense_view_nvme_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: landing.outcome,
                    completion: landing.completion,
                    path: path,
                    nvme: nvme
                )
                usearch_dense_view_nvme(&arguments)
            }
        }
        return try adopted(index, landing, .nvme)
    }

    /// Reads a checkpoint's geometry and metric without loading it - from the manifest's summary
    /// when the path names one, so no shard is opened. Calls `usearch_dense_metadata_read`.
    public static func metadataRead(_ path: String, executor: Executor? = nil) throws(USearchError) -> DenseMetadata {
        let landing = Landing()
        let metadata = landing.slots(usearch_dense_metadata_t())
        try landing.lend(path) { (path) throws(USearchError) in
            var arguments = usearch_dense_metadata_read_t(
                executor: executor?.raw,
                outcome: landing.outcome,
                completion: landing.completion,
                path: path,
                metadata: metadata
            )
            usearch_dense_metadata_read(&arguments)
        }
        return try landing.read { DenseMetadata(metadata.pointee) }
    }

    /// Reads geometry and metric metadata from a serialized monolith buffer. Calls
    /// `usearch_dense_metadata_read_buffer`.
    public static func metadataReadBuffer(
        _ buffer: RawSpan,
        executor: Executor? = nil
    ) throws(USearchError) -> DenseMetadata {
        let landing = Landing()
        let metadata = landing.slots(usearch_dense_metadata_t())
        buffer.withUnsafeBytes { bytes in
            var arguments = usearch_dense_metadata_read_buffer_t(
                executor: executor?.raw,
                outcome: landing.outcome,
                completion: landing.completion,
                buffer: usearch_bytes_view_t(bytes),
                metadata: metadata
            )
            usearch_dense_metadata_read_buffer(&arguments)
        }
        return try landing.read { DenseMetadata(metadata.pointee) }
    }

    /// Adopts the handle a blocking lifecycle verb wrote, or throws its failure.
    private static func adopted(
        _ index: UnsafeMutablePointer<usearch_dense_index_t?>,
        _ landing: Landing,
        _ residency: Residency
    ) throws(USearchError) -> DenseIndex {
        try landing.read { () throws(USearchError) in DenseIndex(raw: try created(index), residency: residency) }
    }

    /// The same verbs, blocking on `executor` after every call submitted there before them.
    /// Binding-only: names the executor every call passes.
    public func on(_ executor: Executor) -> DenseIndexOn { DenseIndexOn(index: self, executor: executor) }

    /// The same verbs, submitted to `executor` and answered at once as a `Deferred`. Binding-only:
    /// names the executor and the deferred completion every call passes.
    public func deferred(_ executor: Executor) -> DenseIndexDeferredOn {
        DenseIndexDeferredOn(index: self, executor: executor)
    }

    /// The verbs on the calling thread.
    private var here: DenseIndexOn { DenseIndexOn(index: self, executor: nil) }

    /// Reserves capacity for a total number of entries, the current size included. Calls
    /// `usearch_dense_reserve`.
    public func reserve(_ capacity: Int) throws(USearchError) { try here.reserve(capacity) }

    /// Erases all entries, keeping the reserved capacity and the frozen geometry. Calls
    /// `usearch_dense_clear`.
    public func clear() throws(USearchError) { try here.clear() }

    /// Reports the frozen metadata and the current counters in one call. Calls
    /// `usearch_dense_describe`.
    public func describe() throws(USearchError) -> DenseIndexDescription { try here.describe() }

    /// The number of entries stored under a key; zero means the index holds none. Calls
    /// `usearch_dense_count`.
    public func count(key: Key) throws(USearchError) -> Int { try here.count(key: key) }

    /// Whether the index holds at least one entry under a key. Binding-only: calls `count`.
    public func contains(key: Key) throws(USearchError) -> Bool { try here.contains(key: key) }

    /// Retrieves every vector stored under a key, one row per entry sharing the key, in insertion
    /// order, converted to `Value`; an absent key yields no rows. The rows land in `arena`, read in
    /// place by the result until the arena's next call. Calls `usearch_dense_get`.
    public func get<Value: ScalarValue>(
        key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> DenseVectors<Value> {
        try here.get(key: key, as: value, arena: arena)
    }

    /// Inserts rows of vectors under their keys - the native batch operation, where a single
    /// vector is a one-row matrix. Overlaps safely with concurrent searches on the same handle.
    ///
    /// The input scalar type is independent of the stored quantization; the library converts on
    /// ingest. A batch is all-or-nothing, and its rows become visible at once: every fallible step
    /// runs before the first entry identity is claimed, so a thrown failure leaves the index
    /// untouched. Every row is claimed before the handler's first report, so it reports but cannot
    /// cancel. Calls `usearch_dense_insert`.
    public func insert(
        keys: [Key],
        vectors: RowsView,
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) {
        try here.insert(keys: keys, vectors: vectors, config: config)
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, returning the
    /// nodes whose edges changed; zero means the graph has settled.
    ///
    /// A build step rather than a repair: a cheap insertion pass followed by one refinement
    /// reaches a better graph, in less total time, than one expensive insertion pass - so a wide
    /// expansion factor belongs here rather than in `insert`. Searches may run throughout. Calls
    /// `usearch_dense_refine`.
    @discardableResult
    public func refine(config: DenseRefineConfig = DenseRefineConfig()) throws(USearchError) -> Int {
        try here.refine(config: config)
    }

    /// Audits how much of the graph a search can reach from its landmarks; entries it misses stay
    /// unfound until a refinement re-links them. Calls `usearch_dense_health`.
    public func health() throws(USearchError) -> DenseReachability { try here.health() }

    /// Resolves rows of queries into up to `wanted` matches apiece, landing in `arena` - the native
    /// batch operation, parallelized internally on an executor. The result reads the arena in place
    /// until its next call. Never writes to the graph, so any number of these may overlap with
    /// concurrent insertions. Calls `usearch_dense_search`.
    public func search(
        _ queries: RowsView,
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Matches {
        try here.search(queries, wanted: wanted, arena: arena, config: config)
    }

    /// Late-interaction scoring of multi-row queries against multi-row entries: every query row
    /// takes its best match within a key, and those maxima sum to that key's score.
    ///
    /// A document is one key holding its token rows, which is what a `valuesPerKey` above one
    /// exists for. `rowsPerQuery` is the token count each query occupies, `queries` holds them end
    /// to end, and up to `wanted` documents land in `arena` as one ragged run, one row per whole
    /// query rather than per token, largest score first; a zero `wanted` keeps every document the
    /// query scored. Approximate downward only: a token whose neighborhood the walk never entered
    /// contributes no maximum, so a key can score below its exact value but never above it. Runs on
    /// an inner-product index alone.
    ///
    /// See <https://arxiv.org/abs/2004.12832> - ColBERT, which introduced this scoring. Calls
    /// `usearch_dense_maxsim`.
    public func maxsim(
        _ queries: RowsView,
        rowsPerQuery: Int,
        wanted: Int,
        arena: Arena,
        config: DenseMaxsimConfig = DenseMaxsimConfig()
    ) throws(USearchError) -> DenseProducts {
        try here.maxsim(queries, rowsPerQuery: rowsPerQuery, wanted: wanted, arena: arena, config: config)
    }

    /// The approximate sparse product of rows against the indexed rows - the linear-algebra
    /// sibling of `search`, reporting every pair the traversal scored rather than only the
    /// nearest of them.
    ///
    /// Pairs arrive in traversal order rather than sorted, and carry no duplicates. A non-zero
    /// `wanted` keeps the first pairs each row's walk scored, and `required` says how many there
    /// were. The keys, products and row offsets land in `arena` as one ragged run. Runs on an
    /// inner-product index alone, since a product needs logits rather than distances. Calls
    /// `usearch_dense_multiply`.
    public func multiply(
        _ left: RowsView,
        arena: Arena,
        config: DenseMultiplyConfig = DenseMultiplyConfig()
    ) throws(USearchError) -> DenseProducts {
        try here.multiply(left, arena: arena, config: config)
    }

    /// Softmax attention of a run of query heads over every indexed key their graph walks score,
    /// folded as Flash Attention folds key tiles, with no top-k cut.
    ///
    /// Row `t · headCount + h` of `queries`, `outputs` and `carry` is head `h` of token `t`: the
    /// `headCount` query heads of grouped-query attention share the index's one key-value head and
    /// one walk per token. The index stores no values, so `lookup` resolves the keys of every
    /// scored batch to rows of its values.
    ///
    /// The outputs land in `arena` as `Value` rows, and come back divided by the partition. Passing
    /// a `carry` asks for the softmax state, one entry per query row, which folds this branch into
    /// another over the same queries by partial-softmax rescaling. Runs on an inner-product index
    /// alone, and refuses a VRAM one with `device`, whose walk needs a device function as its lookup.
    ///
    /// See <https://arxiv.org/abs/2205.14135> - Flash Attention, whose recurrence this reuses.
    /// Calls `usearch_dense_attend`.
    public func attend<Value: ScalarValue>(
        _ queries: RowsView,
        lookup: ValueLookup,
        as value: Value.Type,
        arena: Arena,
        config: DenseAttendConfig = DenseAttendConfig()
    ) throws(USearchError) -> DenseVectors<Value> {
        try here.attend(queries, lookup: lookup, as: value, arena: arena, config: config)
    }

    /// Absorbs every donor into this index, keeping their graphs rather than re-inserting their
    /// vectors. Every donor absorbed is consumed, and every later verb on one reports
    /// `uninitialized`; an absorb stopped early hands the rest back to their indexes once final. A
    /// donor named twice refuses with `invalidArgument`, and the receiver itself with
    /// `configMismatch`, as the library does. Calls `usearch_dense_absorb`.
    public func absorb(_ donors: [DenseIndex], config: DenseAbsorbConfig = DenseAbsorbConfig()) throws(USearchError) {
        try here.absorb(donors, config: config)
    }

    /// Writes the index under a path, in the layout its extension picks - one `.safegraphs`
    /// monolith, or a `.safegraphs.json` manifest beside its shards. Demands a mutation-free
    /// window; concurrent readers keep searching throughout. Calls `usearch_dense_save`.
    public func save(path: String, config: DenseSaveConfig = DenseSaveConfig()) throws(USearchError) {
        try here.save(path: path, config: config)
    }

    /// Writes the index as one monolith, the same bytes `save(path:)` writes to a `.safegraphs`
    /// path, into `arena`. Calls `usearch_dense_save_buffer`.
    public func saveBuffer(
        arena: Arena,
        config: DenseSaveBufferConfig = DenseSaveBufferConfig()
    ) throws(USearchError) -> Monolith {
        try here.saveBuffer(arena: arena, config: config)
    }

    /// Opens a walk over the current contents, for export or migration. Each shard's publication
    /// watermark is read as the walk enters it, so a row once passed is never revisited. Without
    /// `starts` it yields every row as stored; with them, the start keys' own rows first in order,
    /// then the rows those link to breadth-first, each once. Calls `usearch_dense_cursor_init`.
    public func cursor(starts: [Key] = []) throws(USearchError) -> DenseCursor { try here.cursor(starts: starts) }

    /// Opens a resumable nearest-neighbor stream over one query row, so a caller that post-filters
    /// aggressively pulls until satisfied rather than guessing a wanted count upfront. Calls
    /// `usearch_dense_search_stream_init`.
    public func searchStream(
        _ query: RowsView,
        config: DenseSearchStreamInitConfig = DenseSearchStreamInitConfig()
    ) throws(USearchError) -> DenseSearchStream {
        try here.searchStream(query, config: config)
    }
}

// MARK: - Receivers

/// An index's verbs on an executor, each blocking until final after every call submitted there
/// before it; see `DenseIndex.on(_:)` and the index's own verbs for what each one does.
///
/// A blocking call borrows the arena it is given, and its result reads it until the arena's next
/// call.
public struct DenseIndexOn: Sendable {
    /// The index every verb addresses.
    let index: DenseIndex
    /// The executor every verb runs on, or nil for the calling thread.
    let executor: Executor?

    /// Runs one verb, blocking until it is final.
    private func run<Value>(
        _ verb: (DenseCall) throws(USearchError) -> Submission<Value>
    ) throws(USearchError) -> Value {
        try verb(DenseCall(index: index, executor: executor, landing: Landing())).wait()
    }

    /// Calls `usearch_dense_reserve`.
    public func reserve(_ capacity: Int) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.reserve(capacity) }
    }

    /// Calls `usearch_dense_clear`.
    public func clear() throws(USearchError) { try run { (call) throws(USearchError) in try call.clear() } }

    /// Calls `usearch_dense_describe`.
    public func describe() throws(USearchError) -> DenseIndexDescription {
        try run { (call) throws(USearchError) in try call.describe() }
    }

    /// Calls `usearch_dense_count`.
    public func count(key: Key) throws(USearchError) -> Int {
        try run { (call) throws(USearchError) in try call.count(key) }
    }

    /// Binding-only: calls `count`.
    public func contains(key: Key) throws(USearchError) -> Bool { try count(key: key) != 0 }

    /// Calls `usearch_dense_get`.
    public func get<Value: ScalarValue>(
        key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> DenseVectors<Value> {
        try run { (call) throws(USearchError) in try call.get(key, as: value, arena: arena) }
    }

    /// Calls `usearch_dense_insert`.
    public func insert(
        keys: [Key],
        vectors: RowsView,
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.insert(keys, vectors, config: config) }
    }

    /// Calls `usearch_dense_refine`.
    @discardableResult
    public func refine(config: DenseRefineConfig = DenseRefineConfig()) throws(USearchError) -> Int {
        try run { (call) throws(USearchError) in try call.refine(config: config) }
    }

    /// Calls `usearch_dense_health`.
    public func health() throws(USearchError) -> DenseReachability {
        try run { (call) throws(USearchError) in try call.health() }
    }

    /// Calls `usearch_dense_search`.
    public func search(
        _ queries: RowsView,
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Matches {
        try run { (call) throws(USearchError) in try call.search(queries, wanted: wanted, arena: arena, config: config)
        }
    }

    /// Calls `usearch_dense_maxsim`.
    public func maxsim(
        _ queries: RowsView,
        rowsPerQuery: Int,
        wanted: Int,
        arena: Arena,
        config: DenseMaxsimConfig = DenseMaxsimConfig()
    ) throws(USearchError) -> DenseProducts {
        try run { (call) throws(USearchError) in
            try call.maxsim(queries, rowsPerQuery: rowsPerQuery, wanted: wanted, arena: arena, config: config)
        }
    }

    /// Calls `usearch_dense_multiply`.
    public func multiply(
        _ left: RowsView,
        arena: Arena,
        config: DenseMultiplyConfig = DenseMultiplyConfig()
    ) throws(USearchError) -> DenseProducts {
        try run { (call) throws(USearchError) in try call.multiply(left, arena: arena, config: config) }
    }

    /// Calls `usearch_dense_attend`.
    public func attend<Value: ScalarValue>(
        _ queries: RowsView,
        lookup: ValueLookup,
        as value: Value.Type,
        arena: Arena,
        config: DenseAttendConfig = DenseAttendConfig()
    ) throws(USearchError) -> DenseVectors<Value> {
        try run { (call) throws(USearchError) in
            try call.attend(queries, lookup: lookup, as: value, arena: arena, config: config)
        }
    }

    /// Calls `usearch_dense_absorb`.
    public func absorb(_ donors: [DenseIndex], config: DenseAbsorbConfig = DenseAbsorbConfig()) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.absorb(donors, config: config) }
    }

    /// Calls `usearch_dense_save`.
    public func save(path: String, config: DenseSaveConfig = DenseSaveConfig()) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.save(path, config: config) }
    }

    /// Calls `usearch_dense_save_buffer`.
    public func saveBuffer(
        arena: Arena,
        config: DenseSaveBufferConfig = DenseSaveBufferConfig()
    ) throws(USearchError) -> Monolith {
        try run { (call) throws(USearchError) in try call.saveBuffer(arena: arena, config: config) }
    }

    /// Opens a cursor whose every pull blocks on this executor. Calls `usearch_dense_cursor_init`.
    public func cursor(starts: [Key] = []) throws(USearchError) -> DenseCursor {
        try run { (call) throws(USearchError) in try call.cursor(starts: starts) }
    }

    /// Opens a stream whose every pull blocks on this executor. Calls
    /// `usearch_dense_search_stream_init`.
    public func searchStream(
        _ query: RowsView,
        config: DenseSearchStreamInitConfig = DenseSearchStreamInitConfig()
    ) throws(USearchError) -> DenseSearchStream {
        try run { (call) throws(USearchError) in try call.searchStream(query, config: config) }
    }
}

/// An index's verbs submitted to an executor, each answering a `Deferred` at once; see
/// `DenseIndex.deferred(_:)` and the index's own verbs for what each one does.
///
/// A failure this binding catches before the call crosses throws at submission. Every object whose
/// memory a call reads, like rows built over a `Tensor`, stays alive until it is final, and memory
/// borrowed from elsewhere must outlive it too. The index, the executor and the arena are borrowed
/// as a blocking call borrows them, since releasing any of them waits for the calls naming it.
public struct DenseIndexDeferredOn: Sendable {
    /// The index every verb addresses.
    let index: DenseIndex
    /// The executor every verb is submitted to.
    let executor: Executor

    /// Submits one verb, whose landing its `Deferred` keeps until final.
    private func submit<Value>(
        _ verb: (DenseCall) throws(USearchError) -> Submission<Value>
    ) throws(USearchError) -> Deferred<Value> {
        Deferred(try verb(DenseCall(index: index, executor: executor, landing: Landing(usearch_completion_deferred_k))))
    }

    /// Calls `usearch_dense_reserve`.
    public func reserve(_ capacity: Int) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.reserve(capacity) }
    }

    /// Calls `usearch_dense_clear`.
    public func clear() throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.clear() }
    }

    /// Calls `usearch_dense_describe`.
    public func describe() throws(USearchError) -> Deferred<DenseIndexDescription> {
        try submit { (call) throws(USearchError) in try call.describe() }
    }

    /// Calls `usearch_dense_count`.
    public func count(key: Key) throws(USearchError) -> Deferred<Int> {
        try submit { (call) throws(USearchError) in try call.count(key) }
    }

    /// Calls `usearch_dense_get`.
    public func get<Value: ScalarValue>(
        key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> Deferred<DenseVectors<Value>> {
        try submit { (call) throws(USearchError) in try call.get(key, as: value, arena: arena) }
    }

    /// Calls `usearch_dense_insert`.
    public func insert(
        keys: [Key],
        vectors: RowsView,
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.insert(keys, vectors, config: config) }
    }

    /// Calls `usearch_dense_refine`.
    public func refine(config: DenseRefineConfig = DenseRefineConfig()) throws(USearchError) -> Deferred<Int> {
        try submit { (call) throws(USearchError) in try call.refine(config: config) }
    }

    /// Calls `usearch_dense_health`.
    public func health() throws(USearchError) -> Deferred<DenseReachability> {
        try submit { (call) throws(USearchError) in try call.health() }
    }

    /// Calls `usearch_dense_search`.
    public func search(
        _ queries: RowsView,
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Deferred<Matches> {
        try submit { (call) throws(USearchError) in
            try call.search(queries, wanted: wanted, arena: arena, config: config)
        }
    }

    /// Calls `usearch_dense_maxsim`.
    public func maxsim(
        _ queries: RowsView,
        rowsPerQuery: Int,
        wanted: Int,
        arena: Arena,
        config: DenseMaxsimConfig = DenseMaxsimConfig()
    ) throws(USearchError) -> Deferred<DenseProducts> {
        try submit { (call) throws(USearchError) in
            try call.maxsim(queries, rowsPerQuery: rowsPerQuery, wanted: wanted, arena: arena, config: config)
        }
    }

    /// Calls `usearch_dense_multiply`.
    public func multiply(
        _ left: RowsView,
        arena: Arena,
        config: DenseMultiplyConfig = DenseMultiplyConfig()
    ) throws(USearchError) -> Deferred<DenseProducts> {
        try submit { (call) throws(USearchError) in try call.multiply(left, arena: arena, config: config) }
    }

    /// Calls `usearch_dense_attend`.
    public func attend<Value: ScalarValue>(
        _ queries: RowsView,
        lookup: ValueLookup,
        as value: Value.Type,
        arena: Arena,
        config: DenseAttendConfig = DenseAttendConfig()
    ) throws(USearchError) -> Deferred<DenseVectors<Value>> {
        try submit { (call) throws(USearchError) in
            try call.attend(queries, lookup: lookup, as: value, arena: arena, config: config)
        }
    }

    /// Calls `usearch_dense_absorb`.
    public func absorb(
        _ donors: [DenseIndex],
        config: DenseAbsorbConfig = DenseAbsorbConfig()
    ) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.absorb(donors, config: config) }
    }

    /// Calls `usearch_dense_save`.
    public func save(path: String, config: DenseSaveConfig = DenseSaveConfig()) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.save(path, config: config) }
    }

    /// Calls `usearch_dense_save_buffer`.
    public func saveBuffer(
        arena: Arena,
        config: DenseSaveBufferConfig = DenseSaveBufferConfig()
    ) throws(USearchError) -> Deferred<Monolith> {
        try submit { (call) throws(USearchError) in try call.saveBuffer(arena: arena, config: config) }
    }
}

/// The whole queries the rows hold, which only this side can count, since the C call takes the
/// query count rather than the rows.
private func wholeQueries(_ queries: RowsView, _ rowsPerQuery: Int) throws(USearchError) -> Int {
    guard rowsPerQuery > 0, queries.rows % rowsPerQuery == 0 else {
        throw USearchError(status: .shapeMismatch, message: "the query rows do not divide into whole queries")
    }
    return queries.rows / rowsPerQuery
}

// MARK: - Calls

/// The index, executor and landing one verb lowers into its argument structure, and the one place
/// each verb is written.
///
/// Every verb reads its result through a local `read`, called at once for a blocking call and kept
/// only by a deferred one, so the blocking path forms no escaping closure.
struct DenseCall {
    /// The index the call addresses.
    let index: DenseIndex
    /// The executor the call runs on, or nil for the calling thread.
    let executor: Executor?
    /// Where the call's outcome and slots live until it is final.
    let landing: Landing

    /// Whether the call blocks or returns once submitted.
    private var completion: usearch_completion_t { landing.completion }

    /// The handle the call names, refused once an absorb consumed it.
    private func open() throws(USearchError) -> usearch_dense_index_t {
        guard let handle = index.raw else {
            throw USearchError(status: .uninitialized, message: "an absorb consumed this index")
        }
        return handle
    }

    func reserve(_ capacity: Int) throws(USearchError) -> Submission<Void> {
        var arguments = usearch_dense_reserve_t(
            index: try open(),
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            capacity: capacity
        )
        usearch_dense_reserve(&arguments)
        return try landing.submitted()
    }

    func clear() throws(USearchError) -> Submission<Void> {
        var arguments = usearch_dense_clear_t(
            index: try open(),
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion
        )
        usearch_dense_clear(&arguments)
        return try landing.submitted()
    }

    func describe() throws(USearchError) -> Submission<DenseIndexDescription> {
        let handle = try open()
        let description = landing.slots(usearch_dense_index_description_t())
        var arguments = usearch_dense_describe_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            description: description
        )
        usearch_dense_describe(&arguments)
        func read() -> DenseIndexDescription { DenseIndexDescription(description.pointee) }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }

    func count(_ key: Key) throws(USearchError) -> Submission<Int> {
        let handle = try open()
        let entries = landing.slots(0)
        var arguments = usearch_dense_count_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            key: key.raw,
            entries: entries
        )
        usearch_dense_count(&arguments)
        func read() -> Int { entries.pointee }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }

    func get<Value: ScalarValue>(
        _ key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> Submission<DenseVectors<Value>> {
        let handle = try open()
        let vectors = landing.slots(usearch_rows_view_t())
        let written = landing.slots(0)
        let write = landing.lend(arena)
        var arguments = usearch_dense_get_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            key: key.raw,
            vectors: vectors,
            written: written,
            arena: arena.raw,
            outputs_scalar: Value.scalarKind.raw
        )
        usearch_dense_get(&arguments)
        // A generic local function is boxed even where it does not escape, so the reader is a literal.
        if landing.isDeferred {
            return .pending(landing) { DenseVectors(write: write, vectors: vectors.pointee, endIndex: written.pointee) }
        }
        return .final(
            try landing.read { DenseVectors(write: write, vectors: vectors.pointee, endIndex: written.pointee) }
        )
    }

    func insert(
        _ keys: [Key],
        _ vectors: RowsView,
        config: DenseInsertConfig
    ) throws(USearchError) -> Submission<Void> {
        // The key run carries no count of its own, so only this side can see it disagree.
        guard keys.count == vectors.rows else {
            throw USearchError(status: .shapeMismatch, message: "the key count does not match the vector row count")
        }
        let handle = try open()
        landing.anchor(vectors.owner)
        landing.anchor(config.stats)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        try landing.lend(keys) { (keys) throws(USearchError) in
            var arguments = usearch_dense_insert_t(
                index: handle,
                executor: executor?.raw,
                outcome: landing.outcome,
                completion: completion,
                count: vectors.rows,
                keys: usearch_keys_view_t(
                    data: UnsafeRawPointer(keys)?.assumingMemoryBound(to: usearch_key_t.self),
                    stride: MemoryLayout<usearch_key_t>.stride
                ),
                vectors: vectors.raw,
                expansion_factor: config.expansionFactor,
                progress: progress,
                stats: config.stats?.raw
            )
            usearch_dense_insert(&arguments)
        }
        return try landing.submitted()
    }

    func refine(config: DenseRefineConfig) throws(USearchError) -> Submission<Int> {
        let handle = try open()
        landing.anchor(config.stats)
        let changed = landing.slots(0)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        var arguments = usearch_dense_refine_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            changed: changed,
            reachability: nil,
            expansion_factor: config.expansionFactor,
            progress: progress,
            stats: config.stats?.raw
        )
        usearch_dense_refine(&arguments)
        func read() -> Int { changed.pointee }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }

    func health() throws(USearchError) -> Submission<DenseReachability> {
        let handle = try open()
        let reachability = landing.slots(usearch_reachability_t())
        var arguments = usearch_dense_health_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            reachability: reachability
        )
        usearch_dense_health(&arguments)
        func read() -> DenseReachability { DenseReachability(reachability.pointee) }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }

    func search(
        _ queries: RowsView,
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig
    ) throws(USearchError) -> Submission<Matches> {
        let handle = try open()
        landing.anchor(queries.owner)
        landing.anchor(config.stats)
        let matches = landing.slots(usearch_matches_view_t())
        let write = landing.lend(arena)
        let predicate = landing.lend(config.predicate)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        var arguments = usearch_dense_search_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            count: queries.rows,
            queries: queries.raw,
            matches: matches,
            arena: arena.raw,
            wanted: wanted,
            expansion_factor: config.expansionFactor,
            traversal: config.traversal.raw,
            predicate: predicate,
            progress: progress,
            stats: config.stats?.raw
        )
        usearch_dense_search(&arguments)
        func read() -> Matches { Matches(write: write, view: matches.pointee, endIndex: queries.rows) }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }

    func maxsim(
        _ queries: RowsView,
        rowsPerQuery: Int,
        wanted: Int,
        arena: Arena,
        config: DenseMaxsimConfig
    ) throws(USearchError) -> Submission<DenseProducts> {
        let count = try wholeQueries(queries, rowsPerQuery)
        let handle = try open()
        landing.anchor(queries.owner)
        landing.anchor(config.stats)
        let matches = landing.slots(usearch_matches_tape_view_t())
        let required = landing.slots(0)
        let write = landing.lend(arena)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        var arguments = usearch_dense_maxsim_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            count: count,
            queries: queries.raw,
            query_offsets: nil,
            rows_per_query: rowsPerQuery,
            matches: matches,
            written: nil,
            required: required,
            arena: arena.raw,
            wanted: wanted,
            expansion_factor: config.expansionFactor,
            progress: progress,
            stats: config.stats?.raw,
            traversal: Traversal.walk.raw,
            predicate: usearch_predicate_t()
        )
        usearch_dense_maxsim(&arguments)
        func read() -> DenseProducts {
            DenseProducts(write: write, products: matches.pointee, endIndex: count, required: required.pointee)
        }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }

    func multiply(
        _ left: RowsView,
        arena: Arena,
        config: DenseMultiplyConfig
    ) throws(USearchError) -> Submission<DenseProducts> {
        let handle = try open()
        landing.anchor(left.owner)
        landing.anchor(config.stats)
        let products = landing.slots(usearch_matches_tape_view_t())
        let required = landing.slots(0)
        let write = landing.lend(arena)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        var arguments = usearch_dense_multiply_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            count: left.rows,
            left: left.raw,
            products: products,
            written: nil,
            required: required,
            arena: arena.raw,
            wanted: config.wanted,
            expansion_factor: config.expansionFactor,
            progress: progress,
            stats: config.stats?.raw,
            traversal: Traversal.walk.raw,
            predicate: usearch_predicate_t()
        )
        usearch_dense_multiply(&arguments)
        func read() -> DenseProducts {
            DenseProducts(write: write, products: products.pointee, endIndex: left.rows, required: required.pointee)
        }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }

    func attend<Value: ScalarValue>(
        _ queries: RowsView,
        lookup: ValueLookup,
        as value: Value.Type,
        arena: Arena,
        config: DenseAttendConfig
    ) throws(USearchError) -> Submission<DenseVectors<Value>> {
        // The carry pointer has no length, so only this side can see it fall short.
        if let carry = config.carry, carry.count < queries.rows {
            throw USearchError(status: .shapeMismatch, message: "the carry holds fewer states than there are queries")
        }
        // A VRAM walk calls its lookup on the device, and the trampoline below is host code.
        guard index.residency != .vram else {
            throw USearchError(status: .device, message: "a VRAM index resolves values through a device function")
        }
        let handle = try open()
        landing.anchor(queries.owner)
        landing.anchor(lookup.values.owner)
        landing.anchor(config.stats)
        let outputs = landing.slots(usearch_rows_view_t())
        let write = landing.lend(arena)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        var arguments = usearch_dense_attend_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            count: queries.rows,
            queries: queries.raw,
            head_count: config.headCount,
            value_lookup: valueLookupTrampoline,
            value_lookup_state: landing.place(LookupState(lookup: lookup, landing: landing)),
            carry: UnsafeMutableRawPointer(config.carry?.baseAddress)?
                .assumingMemoryBound(to: usearch_dense_softmax_state_t.self),
            outputs: outputs,
            arena: arena.raw,
            outputs_scalar: Value.scalarKind.raw,
            expansion_factor: config.expansionFactor,
            score_scale: config.scoreScale,
            progress: progress,
            stats: config.stats?.raw,
            traversal: Traversal.walk.raw,
            predicate: usearch_predicate_t()
        )
        usearch_dense_attend(&arguments)
        let rows = queries.rows
        if landing.isDeferred {
            return .pending(landing) { DenseVectors(write: write, vectors: outputs.pointee, endIndex: rows) }
        }
        return .final(try landing.read { DenseVectors(write: write, vectors: outputs.pointee, endIndex: rows) })
    }

    func absorb(_ donors: [DenseIndex], config: DenseAbsorbConfig) throws(USearchError) -> Submission<Void> {
        // The library would absorb one handle twice, so only distinct donors other than the receiver cross.
        let distinct = Set(donors.map(ObjectIdentifier.init))
        guard distinct.count == donors.count else {
            throw USearchError(status: .invalidArgument, message: "a donor is named twice")
        }
        guard !distinct.contains(ObjectIdentifier(index)) else {
            throw USearchError(status: .configMismatch, message: "an index cannot absorb itself")
        }
        let handle = try open()
        // The library nulls each slot it consumes, so the slots are the landing's own.
        let slots = landing.slots(usearch_dense_index_t?.none, count: donors.count)
        for (position, donor) in donors.enumerated() {
            guard let raw = donor.raw else {
                throw USearchError(status: .uninitialized, message: "an absorb consumed this donor")
            }
            slots[position] = raw
        }
        for donor in donors { donor.raw = nil }
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        var arguments = usearch_dense_absorb_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            count: donors.count,
            donors: slots,
            changed: nil,
            reachability: nil,
            expansion_factor: 0,
            progress: progress,
            stats: nil
        )
        usearch_dense_absorb(&arguments)
        // The handles belong to the call until final, then whichever it left go back to their donors.
        landing.onFinal {
            for (position, donor) in donors.enumerated() {
                if let survivor = slots[position] { donor.raw = survivor }
            }
        }
        return try landing.submitted()
    }

    func save(_ path: String, config: DenseSaveConfig) throws(USearchError) -> Submission<Void> {
        let handle = try open()
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        try landing.lend(path) { (path) throws(USearchError) in
            var arguments = usearch_dense_save_t(
                index: handle,
                executor: executor?.raw,
                outcome: landing.outcome,
                completion: completion,
                path: path,
                progress: progress
            )
            usearch_dense_save(&arguments)
        }
        return try landing.submitted()
    }

    func saveBuffer(arena: Arena, config: DenseSaveBufferConfig) throws(USearchError) -> Submission<Monolith> {
        let handle = try open()
        let monolith = landing.slots(usearch_bytes_view_t())
        let write = landing.lend(arena)
        let progress = landing.lend(config.progress, steps: config.progressSteps)
        var arguments = usearch_dense_save_buffer_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: completion,
            monolith: monolith,
            arena: arena.raw,
            progress: progress
        )
        usearch_dense_save_buffer(&arguments)
        func read() -> Monolith { Monolith(write: write, monolith: monolith.pointee) }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }

    func cursor(starts: [Key]) throws(USearchError) -> Submission<DenseCursor> {
        let handle = try open()
        let cursor = landing.slots(usearch_dense_cursor_t?.none)
        try landing.lend(starts) { (run) throws(USearchError) in
            var arguments = usearch_dense_cursor_init_t(
                cursor: cursor,
                index: handle,
                outcome: landing.outcome,
                starts_count: starts.count,
                starts: UnsafeRawPointer(run)?.assumingMemoryBound(to: usearch_key_t.self)
            )
            usearch_dense_cursor_init(&arguments)
        }
        func read() throws(USearchError) -> DenseCursor {
            DenseCursor(raw: try created(cursor), index: index, executor: executor)
        }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }

    func searchStream(
        _ query: RowsView,
        config: DenseSearchStreamInitConfig
    ) throws(USearchError) -> Submission<DenseSearchStream> {
        // The C call takes one row and no count, so only this side can see a matrix carrying more.
        guard query.rows == 1 else {
            throw USearchError(status: .shapeMismatch, message: "a search stream walks one query row")
        }
        let handle = try open()
        let stream = landing.slots(usearch_dense_search_stream_t?.none)
        var arguments = usearch_dense_search_stream_init_t(
            stream: stream,
            index: handle,
            outcome: landing.outcome,
            query: query.raw,
            expansion_factor: config.expansionFactor
        )
        usearch_dense_search_stream_init(&arguments)
        func read() throws(USearchError) -> DenseSearchStream {
            DenseSearchStream(raw: try created(stream), index: index, executor: executor)
        }
        return try landing.isDeferred ? .pending(landing, read) : .final(landing.read(read))
    }
}

/// A filtered search's predicate and the landing its first thrown error goes to, placed in that
/// landing, since a Swift closure has no stable C representation.
private final class AdmissionState {
    let admits: @Sendable (Key) throws -> Bool
    /// The landing that keeps this state, so it outlives every call of the trampoline.
    unowned let landing: Landing

    init(admits: @escaping @Sendable (Key) throws -> Bool, landing: Landing) {
        self.admits = admits
        self.landing = landing
    }
}

/// The admission trampoline the engine calls from the executor's threads with a batch of keys. A
/// thrown error is latched and rejects its key, so nothing unwinds out of this frame and into C.
private let admissionTrampoline: usearch_predicate_callback_t = { keys, admissions, count, state in
    guard let keys, let admissions else { return }
    // A lost state means the trampoline is broken. Admitting would turn a filtered search into
    // an unfiltered one and answer wrong rows without saying so.
    guard let state else { return admissions.update(repeating: usearch_reject_k, count: count) }
    let admission = Unmanaged<AdmissionState>.fromOpaque(state).takeUnretainedValue()
    for index in 0 ..< count {
        do {
            admissions[index] = try admission.admits(Key(raw: keys[index])) ? usearch_admit_k : usearch_reject_k
        }
        catch {
            admission.landing.latch(error)
            admissions[index] = usearch_reject_k
        }
    }
}

extension Landing {
    /// Lends an optional predicate as the struct a search takes; nil lends nulls.
    fileprivate func lend(_ predicate: (@Sendable (Key) throws -> Bool)?) -> usearch_predicate_t {
        guard let predicate else { return usearch_predicate_t() }
        let state = place(AdmissionState(admits: predicate, landing: self))
        return usearch_predicate_t(callback: admissionTrampoline, state: state, cost: usearch_predicate_cost_cheap_k)
    }
}

/// Attention's value lookup and the landing its first thrown error goes to, placed in that
/// landing, since a C trampoline cannot capture the lookup itself.
private final class LookupState {
    let lookup: ValueLookup
    /// The landing that keeps this state, so it outlives every call of the trampoline.
    unowned let landing: Landing

    init(lookup: ValueLookup, landing: Landing) {
        self.lookup = lookup
        self.landing = landing
    }
}

/// The value-lookup trampoline the engine calls from the executor's threads, once per scored batch.
private let valueLookupTrampoline: usearch_value_lookup_t = { state, keys, count, scalar, columns, rows in
    guard let rows else { return }
    // A row left unresolved, or a lost state, refuses the call rather than reading garbage.
    guard let state, let keys else { return rows.update(repeating: nil, count: count) }
    let lookup = Unmanaged<LookupState>.fromOpaque(state).takeUnretainedValue()
    let values = lookup.lookup.values
    // The library reads rows of its own shape, so values of another one answer no row at all.
    guard scalar == values.scalar.raw, columns == values.columns else {
        return rows.update(repeating: nil, count: count)
    }
    for position in 0 ..< count {
        do {
            let row = try lookup.lookup.row(Key(raw: keys[position]))
            rows[position] = row.flatMap { $0 >= 0 && $0 < values.rows ? values.data + $0 * values.stride : nil }
        }
        catch {
            lookup.landing.latch(error)
            rows[position] = nil
        }
    }
}

// MARK: - Cursors & Streams

/// A walk over an index's contents; see `DenseIndex.cursor()`.
///
/// The walk is sequential, so it stays a batch operation - export, migration, and
/// visualization all move whole batches. Every pull blocks on the executor of the receiver that
/// opened it, and the cursor keeps its index alive.
public final class DenseCursor {
    /// The native handle, released by `deinit` and never reassigned.
    private let raw: usearch_dense_cursor_t
    /// The index the walk covers, held so the cursor cannot outlive it.
    private let index: DenseIndex
    /// The executor every pull runs on, or nil for the calling thread.
    private let executor: Executor?

    init(raw: usearch_dense_cursor_t, index: DenseIndex, executor: Executor?) {
        self.raw = raw
        self.index = index
        self.executor = executor
    }

    /// Calls `usearch_dense_cursor_free`.
    deinit { usearch_dense_cursor_free(raw) }

    /// Copies up to `wanted` next keys into `arena`, walking the key column alone; an empty batch
    /// means the walk is exhausted. Calls `usearch_dense_cursor_next`.
    public func next(wanted: Int, arena: Arena) throws(USearchError) -> DenseKeys {
        try pull(wanted, false, arena).keys
    }

    /// Copies up to `wanted` next keys into `arena`, each with the keys it links to; an empty batch
    /// means the walk is exhausted. The rows themselves come from `get`. Calls
    /// `usearch_dense_cursor_next`.
    public func nextLinks(wanted: Int, arena: Arena) throws(USearchError) -> (keys: DenseKeys, links: DenseLinks) {
        let (keys, links) = try pull(wanted, true, arena)
        return (keys, DenseLinks(write: keys.write, links: links, endIndex: keys.count))
    }

    /// Advances the walk by one batch, exporting each entry's links only when `linked` asks.
    private func pull(
        _ wanted: Int,
        _ linked: Bool,
        _ arena: Arena
    ) throws(USearchError) -> (keys: DenseKeys, links: usearch_keys_tape_view_t) {
        guard let handle = index.raw else {
            throw USearchError(status: .uninitialized, message: "an absorb consumed this index")
        }
        let landing = Landing()
        let keys = landing.slots(usearch_keys_view_t())
        let links = landing.slots(usearch_keys_tape_view_t())
        let written = landing.slots(0)
        let write = landing.lend(arena)
        var arguments = usearch_dense_cursor_next_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: landing.completion,
            cursor: raw,
            keys: keys,
            links: linked ? links : nil,
            written: written,
            arena: arena.raw,
            wanted: wanted
        )
        usearch_dense_cursor_next(&arguments)
        return try landing.read {
            (DenseKeys(write: write, keys: keys.pointee, endIndex: written.pointee), links.pointee)
        }
    }
}

/// A resumable nearest-neighbor stream over one query; see `DenseIndex.searchStream`.
///
/// Best-first traversal is naturally pausable, so the stream keeps a live frontier and visited set
/// and yields matches lazily. Ordering beyond the expansion horizon is best-effort nearest rather
/// than globally exact, so pulling far past that breadth degrades the ranking rather than ending
/// the stream. Every pull blocks on the executor of the receiver that opened it.
public final class DenseSearchStream {
    /// The native handle, released by `deinit` and never reassigned.
    private let raw: usearch_dense_search_stream_t
    /// The index being walked, held so the stream cannot outlive it.
    private let index: DenseIndex
    /// The executor every pull runs on, or nil for the calling thread.
    private let executor: Executor?

    init(raw: usearch_dense_search_stream_t, index: DenseIndex, executor: Executor?) {
        self.raw = raw
        self.index = index
        self.executor = executor
    }

    /// Calls `usearch_dense_search_stream_free`.
    deinit { usearch_dense_search_stream_free(raw) }

    /// Pulls up to `wanted` more matches into `arena`, as the one row of a search, nearest first;
    /// an empty row means the traversal is drained. Calls `usearch_dense_search_stream_next`.
    public func next(wanted: Int, arena: Arena) throws(USearchError) -> Matches {
        guard let handle = index.raw else {
            throw USearchError(status: .uninitialized, message: "an absorb consumed this index")
        }
        let landing = Landing()
        let matches = landing.slots(usearch_matches_view_t())
        let write = landing.lend(arena)
        var arguments = usearch_dense_search_stream_next_t(
            index: handle,
            executor: executor?.raw,
            outcome: landing.outcome,
            completion: landing.completion,
            stream: raw,
            matches: matches,
            arena: arena.raw,
            wanted: wanted
        )
        usearch_dense_search_stream_next(&arguments)
        return try landing.read { Matches(write: write, view: matches.pointee, endIndex: 1) }
    }
}

extension usearch_bytes_view_t {
    /// Views a byte buffer as the pointer-and-length pair every buffer argument takes.
    fileprivate init(_ bytes: UnsafeRawBufferPointer) {
        self.init(data: bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), length: bytes.count)
    }
}
