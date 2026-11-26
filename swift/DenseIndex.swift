/**
 *  @file swift/DenseIndex.swift
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Vector Search over equidimensional dense embeddings.
 *
 *  Every verb is a batch over strided rows, so a memory-mapped dataset reaches the index
 *  with no reshape and no copy, and a single vector is a one-row matrix. The breadth of the
 *  candidates rides each call rather than the handle, which lets a recall sweep run against one
 *  immutable index with no mutation between points.
 */

import NumKong

import USearchC

// MARK: - Options

/// A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
/// capabilities. All of them derive from a single inner-product kernel with pre-computed
/// magnitudes; hamming and jaccard are bit-level, over the packed sub-byte kinds.
///
/// Arbitrary metrics belong to the baseline tier, which forfeits the SIMD kernel chains and the
/// magnitude cache both.
public struct DenseMetric: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
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

    /// The metric rendered as the library spells it.
    public var description: String { String(cString: usearch_dense_metric_name(raw)) }

    /// Reads a spelling back, and nil for one no metric holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_dense_metric_from_name, usearch_dense_metric_unknown_k) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_dense_metric_t) { self.raw = raw }
}

/// The construction options of a dense index, as `initRam` and `initVram` freeze them.
///
/// Stored vectors always round up to the SIMD step quantum, so they stay cache-line aligned and
/// every kernel runs tail-free; the arena is slightly larger for it, and that is not a knob.
public struct DenseOptions: Sendable {
    /// The width of the indexed vectors; zero leaves the geometry unfrozen, so the index admits
    /// no entries.
    public var dimensions: Int
    /// The distance kind used between vectors.
    public var metric: DenseMetric
    /// The scalar type vectors are stored and quantized into.
    public var quantization: Scalar
    /// The bound on connections per node in the graph; zero derives.
    public var connectivity: Int
    /// The bound on entries sharing one key; zero is unbounded.
    public var valuesPerKey: Int
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; nil means no ceiling.
    public var cpuClassCeiling: CPUClass?
    /// The newest class dispatch may pick, capped below the best class every named device loads; a
    /// ceiling no named device loads is refused as `unsupported`; nil means no ceiling.
    public var gpuClassCeiling: GPUClass?

    /// An angular bf16 index, with every optional field derived.
    public init(
        dimensions: Int = 0,
        metric: DenseMetric = .angular,
        quantization: Scalar = .bf16,
        connectivity: Int = 0,
        valuesPerKey: Int = 0,
        cpuClassCeiling: CPUClass? = nil,
        gpuClassCeiling: GPUClass? = nil
    ) {
        self.dimensions = dimensions
        self.metric = metric
        self.quantization = quantization
        self.connectivity = connectivity
        self.valuesPerKey = valuesPerKey
        self.cpuClassCeiling = cpuClassCeiling
        self.gpuClassCeiling = gpuClassCeiling
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
    /// The library's own record.
    var raw = usearch_dense_metadata_t()

    /// The scalars per stored vector.
    public var dimensions: Int { raw.dimensions }
    /// The distance kind the checkpoint scores with.
    public var metric: DenseMetric { DenseMetric(raw: raw.metric) }
    /// The scalar type its vectors are stored and quantized into.
    public var quantization: Scalar { Scalar(raw: raw.quantization) }
    /// The connections-per-node bound the graph was built with.
    public var connectivity: Int { raw.connectivity }
}

/// A live index's frozen metadata and current counters, each read once, so under concurrent
/// insertions they may come from slightly different moments.
public struct DenseIndexDescription: Sendable {
    /// The library's own record.
    var raw = usearch_dense_index_description_t()

    /// The frozen metric, quantization, dimensions and connectivity, as a checkpoint reports them.
    public var metadata: DenseMetadata { DenseMetadata(raw: raw.metadata) }
    /// The stored entries.
    public var size: Int { raw.size }
    /// The entries hosted without growing again.
    public var capacity: Int { raw.capacity }
    /// The bytes the handle holds; traversal scratch is the executor's.
    public var memoryUsage: Int { raw.memory_usage }
    /// The CPU class whose kits the index dispatched to on RAM and NVMe; nil on VRAM.
    public var cpuClass: CPUClass? {
        raw.cpu_class == usearch_cpu_class_unknown_k ? nil : CPUClass(raw: raw.cpu_class)
    }
    /// The GPU class whose kernels the index dispatched to on VRAM; nil on RAM and NVMe.
    public var gpuClass: GPUClass? {
        raw.gpu_class == usearch_gpu_class_unknown_k ? nil : GPUClass(raw: raw.gpu_class)
    }
    /// The live shards, each a graph of its own.
    public var shardsCount: Int { raw.shards_count }
}

/// One reachability audit: the population it judged, and how much of it a search reaches.
public struct DenseReachability: Equatable, Sendable {
    /// The library's own record.
    var raw = usearch_reachability_t()

    /// The entries published when the audit started; compare `reachable` against this, never a
    /// later size.
    public var audited: Int { raw.audited }
    /// Of those, the entries a search reaches from the root; the rest no search finds until
    /// a refinement re-links them.
    public var reachable: Int { raw.reachable }
    /// The depths the walk from the root took to converge.
    public var depthFromRoot: Int { raw.depth_from_root }
    /// The depths the walk back toward the root took, where a health check walks both ways.
    public var depthToRoot: Int { raw.depth_to_root }

    /// Equal in every count the audit published.
    public static func == (left: DenseReachability, right: DenseReachability) -> Bool {
        (left.audited, left.reachable, left.depthFromRoot, left.depthToRoot)
            == (right.audited, right.reachable, right.depthFromRoot, right.depthToRoot)
    }
}

/// What a refine round concluded, so no caller picks a change rate.
public struct DenseRefineVerdict: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_refine_verdict_t

    /// No round concluded: a report no round filled yet.
    public static let none = DenseRefineVerdict(raw: usearch_refine_none_k)
    /// The round changed some entry's link set, so another round can help.
    public static let improved = DenseRefineVerdict(raw: usearch_refine_improved_k)
    /// The round changed no entry's link set.
    public static let converged = DenseRefineVerdict(raw: usearch_refine_converged_k)
    /// The round lowered nothing while something stands, as full entries leave no slot.
    public static let stalled = DenseRefineVerdict(raw: usearch_refine_stalled_k)
    /// The caller stopped the round; the phase names where.
    public static let stopped = DenseRefineVerdict(raw: usearch_refine_stopped_k)

    /// The verdict rendered as the library spells it.
    public var description: String { String(cString: usearch_refine_verdict_name(raw)) }

    /// Reads a spelling back, and nil for one no verdict holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_refine_verdict_from_name, usearch_refine_none_k) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_refine_verdict_t) { self.raw = raw }
}

/// What one `refine` or `reshard` did to the graph, so far when its plan stays live, and what the
/// whole plan did once its `finish` lands it.
public struct DenseRefineReport: Sendable {
    /// The library's own record.
    var raw = usearch_refine_report_t()

    /// The entries whose neighbors were selected again, the islands of every heal round included.
    public var reselected: Int { raw.reselected }
    /// Of those, the entries whose neighbor set changed, any link added or dropped.
    public var changed: Int { raw.changed }
    /// The entries the first audit found not reached both ways.
    public var islandsFound: Int { raw.islands_found }
    /// The links written between an island and an entry the audit reached both ways.
    public var islandsReconnected: Int { raw.islands_reconnected }
    /// The neighborhoods the round re-stitched.
    public var neighborhoods: Int { raw.neighborhoods }
    /// The entries of the widest neighborhood.
    public var widestNeighborhood: Int { raw.widest_neighborhood }
    /// The links the round added.
    public var addedLinks: Int { raw.added_links }
    /// The links the round dropped.
    public var droppedLinks: Int { raw.dropped_links }
    /// What the round concluded.
    public var verdict: DenseRefineVerdict { DenseRefineVerdict(raw: raw.verdict) }
    /// The health check opening the round.
    public var opening: DenseReachability { DenseReachability(raw: raw.opening) }
    /// The health check closing the heal, as a fresh `health` right after would take it.
    public var reachability: DenseReachability { DenseReachability(raw: raw.reachability) }
}

// MARK: - Per-Call Configs & Telemetry

/// The traversal telemetry one call reports, passed in a config and written by the call. Every
/// call overwrites it rather than accumulating, so a sweep over many batches sums them itself,
/// and it belongs to one call until that call is final.
public final class DenseTraversalStats: @unchecked Sendable {
    /// The counters the call writes, at an address that holds for a deferred call.
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
    /// The hint keys no shard held an entry under; zero unless a search was handed hints.
    public var missingHints: Int { raw.pointee.missing_hints }
    /// Of the distances computed, the sketch distances a sequence search's seeding walk spent;
    /// zero for every other walk.
    public var seedingDistances: Int { raw.pointee.seeding_distances }
}

/// Flash Attention's running maximum and partition for one query row.
///
/// The two are one concept: a maximum without its partition is a number nothing can rescale, so
/// they share an element rather than travelling as two arrays a caller could supply half of.
/// Holding them is what folds one attention call into another over the same queries.
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
    /// threads together, so it must be safe for concurrent use.
    public var row: @Sendable (Key) -> Int?

    public init(values: RowsView, row: @escaping @Sendable (Key) -> Int?) {
        self.values = values
        self.row = row
    }
}

/// The knobs of `insert`. Every zeroed field derives, which is what almost every caller wants.
public struct DenseInsertConfig {
    /// The breadth of the candidates the insertion walk searches for neighbors with; zero derives it
    /// from the geometry.
    public var expansionFactor: Int
    /// Called with entries admitted; it reports and cannot cancel.
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
    /// Bytes a loaded NVMe handle holds for the one part it rewrites; zero derives what rounds the
    /// part in RAM, and RAM and VRAM handles never read it.
    public var memoryLimit: Int
    /// Called with each phase's units at its barriers; a stop leaves the graph correct and merely
    /// less improved.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?
    /// Keeps the round live across calls: nil runs one round to its end, a fresh plan opens one
    /// that a stop leaves live, and a live one continues where it stopped, or starts the next
    /// round.
    public var plan: DenseRefinePlan?

    public init(
        expansionFactor: Int = 0,
        memoryLimit: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil,
        plan: DenseRefinePlan? = nil
    ) {
        self.expansionFactor = expansionFactor
        self.memoryLimit = memoryLimit
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
        self.plan = plan
    }
}

/// The knobs of `search`. Every zeroed field derives.
public struct DenseSearchConfig {
    /// The breadth of the candidates the walk explores; zero derives it from the geometry, and a
    /// scan ignores it.
    public var expansionFactor: Int
    /// Walks the graph by default; `.scan` scores every entry exactly, on RAM and NVMe handles, and
    /// a VRAM handle refuses it with `unsupported`.
    public var traversal: Traversal
    /// Narrows the matches to the keys it approves. Non-matching entries still route the
    /// traversal, so recall does not collapse at low selectivity. It runs on the executor's
    /// threads together, so it must be safe for concurrent use. A VRAM handle refuses one, since
    /// host code cannot narrow a device walk.
    public var predicate: (@Sendable (Key) -> Bool)?
    /// Called with queries resolved.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?

    public init(
        expansionFactor: Int = 0,
        traversal: Traversal = .walk,
        predicate: (@Sendable (Key) -> Bool)? = nil,
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
    /// The bound on pairs kept per left row, the first ones its walk scored; zero keeps every pair.
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

/// The placements and knobs of `reshard`; a nil placement keeps the handle's own.
public struct DenseReshardConfig {
    /// NUMA node IDs of a RAM placement of a handle recut in place, one shard per node at least;
    /// empty derives them.
    public var memoryDomains: [Int32]?
    /// CUDA ordinals of a VRAM placement, one shard per device at least.
    public var devices: [Int32]?
    /// Drives of an NVMe placement, one shard per drive at least.
    public var paths: [String]?
    /// Entries per shard of the RAM or VRAM placement; zero keeps the handle's own bound.
    public var shardCapacity: Int
    /// Picks the part roots of a source split among several targets: equal seeds over an unchanged
    /// source split equally.
    public var seed: UInt64
    /// The breadth of the candidates of the closing rounds; zero derives.
    public var expansionFactor: Int
    /// Bytes a loaded NVMe target holds for the one part it rewrites; zero derives what rounds the
    /// part in RAM, and RAM and VRAM targets never read it.
    public var memoryLimit: Int
    /// Called with the closing rounds' units at their barriers; a stop never undoes a move.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?
    /// Keeps the reshard live across calls: nil moves and runs to the end, a fresh plan opens one
    /// that a stop leaves live, and a live one continues where it stopped.
    public var plan: DenseReshardPlan?

    public init(
        memoryDomains: [Int32]? = nil,
        devices: [Int32]? = nil,
        paths: [String]? = nil,
        shardCapacity: Int = 0,
        seed: UInt64 = 0,
        expansionFactor: Int = 0,
        memoryLimit: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil,
        plan: DenseReshardPlan? = nil
    ) {
        self.memoryDomains = memoryDomains
        self.devices = devices
        self.paths = paths
        self.shardCapacity = shardCapacity
        self.seed = seed
        self.expansionFactor = expansionFactor
        self.memoryLimit = memoryLimit
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
        self.plan = plan
    }
}

/// One refine kept live across calls: the round in hand, and the root every round's health
/// checks share. A fresh plan opens on the first `refine` naming it, and `finish` closes it;
/// while it lives the index admits no insertion and no save, and each call names the index and
/// executor it was opened with.
public final class DenseRefinePlan: @unchecked Sendable {
    /// The handle the library opens, and the phase its last call stopped in, at addresses that hold
    /// for a deferred call.
    let raw: UnsafeMutablePointer<(handle: usearch_dense_refine_plan_t?, phase: PhaseProgress)>

    public init() {
        raw = .allocate(capacity: 1)
        raw.initialize(to: (nil, PhaseProgress()))
    }

    deinit { raw.deallocate() }

    /// Where the last call naming the plan stopped, `done` once its round ended.
    public var phase: PhaseProgress { raw.pointee.phase }
}

/// One reshard kept live across calls: its own sources and targets, the entries still to move,
/// then the rounds of every target that took entries. A fresh plan opens on the first `reshard`
/// naming it, and `finish` closes it; while it lives every handle it names admits no insertion
/// and no save, and each call names the executor it was opened with.
public final class DenseReshardPlan: @unchecked Sendable {
    /// The handle the library opens, and the phase its last call stopped in, at addresses that hold
    /// for a deferred call.
    let raw: UnsafeMutablePointer<(handle: usearch_dense_reshard_plan_t?, phase: PhaseProgress)>

    public init() {
        raw = .allocate(capacity: 1)
        raw.initialize(to: (nil, PhaseProgress()))
    }

    deinit { raw.deallocate() }

    /// Where the last call naming the plan stopped, `done` once its rounds ended.
    public var phase: PhaseProgress { raw.pointee.phase }
}

/// The knobs of `sample`. Left alone, it draws across the whole index with nothing shown.
public struct DenseSampleConfig {
    /// The keys already shown: never drawn again, and the paths reach toward them.
    public var shown: [Key]
    /// Where each `shown` key sits, three values apiece, end to end.
    public var shownCoordinates: [Float]
    /// The keys to look around; empty looks around the whole index.
    public var focus: [Key]
    /// Equal seeds, shown keys and focus over an unchanged index draw equally.
    public var seed: UInt64
    /// The breadth of the candidates of the walks behind the draws; zero derives.
    public var expansionFactor: Int
    /// Called with draws made; a stop answers `cancelled`.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?

    public init(
        shown: [Key] = [],
        shownCoordinates: [Float] = [],
        focus: [Key] = [],
        seed: UInt64 = 0,
        expansionFactor: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil
    ) {
        self.shown = shown
        self.shownCoordinates = shownCoordinates
        self.focus = focus
        self.seed = seed
        self.expansionFactor = expansionFactor
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
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
    /// The memory domains to place the parts on, part `p` loading into shard `p` modulo their
    /// count; empty derives every node the process may allocate on.
    public var memoryDomains: [Int32]
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; nil means no ceiling.
    public var cpuClassCeiling: CPUClass?
    /// The bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    public var valuesPerKey: Int
    /// Called with shards placed.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(
        executor: Executor? = nil,
        memoryDomains: [Int32] = [],
        cpuClassCeiling: CPUClass? = nil,
        valuesPerKey: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) {
        self.executor = executor
        self.memoryDomains = memoryDomains
        self.cpuClassCeiling = cpuClassCeiling
        self.valuesPerKey = valuesPerKey
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// The knobs of `loadVram`.
public struct DenseLoadVramConfig {
    /// The executor the upload runs on, one stream per device; nil uploads from the calling thread.
    public var executor: Executor?
    /// The devices to place the parts on, part `p` loading onto device `p` modulo their count;
    /// empty derives one free device.
    public var devices: [Int32]
    /// The newest class dispatch may pick, capped below the best class every named device loads; a
    /// ceiling no named device loads is refused as `unsupported`; nil means no ceiling.
    public var gpuClassCeiling: GPUClass?
    /// The bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    public var valuesPerKey: Int
    /// Called with shards placed.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(
        executor: Executor? = nil,
        devices: [Int32] = [],
        gpuClassCeiling: GPUClass? = nil,
        valuesPerKey: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) {
        self.executor = executor
        self.devices = devices
        self.gpuClassCeiling = gpuClassCeiling
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
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; nil means no ceiling.
    public var cpuClassCeiling: CPUClass?
    /// The bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    public var valuesPerKey: Int
    /// Called with bytes placed.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(
        executor: Executor? = nil,
        memoryDomains: [Int32] = [],
        cpuClassCeiling: CPUClass? = nil,
        valuesPerKey: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) {
        self.executor = executor
        self.memoryDomains = memoryDomains
        self.cpuClassCeiling = cpuClassCeiling
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
    /// The newest class dispatch may pick, capped below the best class every named device loads; a
    /// ceiling no named device loads is refused as `unsupported`; nil means no ceiling.
    public var gpuClassCeiling: GPUClass?
    /// The bound on entries sharing one key, which no checkpoint records; zero is unbounded.
    public var valuesPerKey: Int
    /// Called with bytes placed.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int

    public init(
        executor: Executor? = nil,
        devices: [Int32] = [],
        gpuClassCeiling: GPUClass? = nil,
        valuesPerKey: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0
    ) {
        self.executor = executor
        self.devices = devices
        self.gpuClassCeiling = gpuClassCeiling
        self.valuesPerKey = valuesPerKey
        self.progress = progress
        self.progressSteps = progressSteps
    }
}

/// The knobs of `viewNvme` and `loadNvme`.
public struct DenseViewNvmeConfig {
    /// The NVMe executor whose rings read the checkpoint's headers; nil reads on the calling thread.
    public var executor: Executor?
    /// The devices the parts are served from - a filesystem path, a transport identifier, or a
    /// FUSE mount over object storage; empty serves each part where its manifest points.
    public var paths: [String]
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; nil means no ceiling.
    public var cpuClassCeiling: CPUClass?
    /// Memory domains of a key table held in RAM, which `count`, `get`, `sample`, MaxSim and hinted
    /// searches read; empty derives them, and nil holds none.
    public var keyTableDomains: [Int32]?

    public init(
        executor: Executor? = nil,
        paths: [String] = [],
        cpuClassCeiling: CPUClass? = nil,
        keyTableDomains: [Int32]? = nil
    ) {
        self.executor = executor
        self.paths = paths
        self.cpuClassCeiling = cpuClassCeiling
        self.keyTableDomains = keyTableDomains
    }
}

// MARK: - Results

/// The rows `get` or `attend` wrote into an arena, one per entry sharing the key or per query, read
/// in place as `Value` until the arena's next call, and holding the arena alive.
public struct DenseVectors<Value: ScalarValue>: RandomAccessCollection {
    /// The arena the rows live in, held so they outlive no release of it.
    let arena: ArenaWrite
    /// The packed rows the call wrote.
    var vectors = usearch_rows_view_t()
    /// One past the last row the call wrote.
    public internal(set) var endIndex: Int

    /// The first row's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// The bytes between consecutive rows, as the library reported them.
    public var stride: Int { vectors.stride }

    /// The logical values in each row, as the library reported them.
    public var columns: Int { vectors.columns }

    /// One row's stored values, packed at their natural width.
    public subscript(row: Int) -> Row {
        arena.read(row, below: endIndex) {
            let perValue = Value.scalarKind.dimensionsPerValue
            return Row(
                arena: arena,
                base: vectors.data! + row * vectors.stride,
                endIndex: (columns + perValue - 1) / perValue
            )
        }
    }

    /// One row of the arena's values, read in place.
    public struct Row: RandomAccessCollection {
        /// The arena the row lives in, held so it outlives no release of it.
        let arena: ArenaWrite
        /// The row's first value.
        let base: UnsafeRawPointer
        /// One past the last value in the row.
        public let endIndex: Int

        /// The first value's position, which a collection walk starts from.
        public var startIndex: Int { 0 }

        /// The value at a position.
        public subscript(position: Int) -> Value {
            arena.read(position, below: endIndex) {
                return base.load(fromByteOffset: position * MemoryLayout<Value>.stride, as: Value.self)
            }
        }
    }
}

/// The keys a cursor pull wrote into an arena, in walk order.
public struct DenseKeys: RandomAccessCollection {
    /// The arena the keys live in, held so they outlive no release of it.
    let arena: ArenaWrite
    /// The keys the call wrote.
    var keys = usearch_keys_view_t()
    /// One past the last key the call wrote.
    public internal(set) var endIndex: Int

    /// The first key's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// The key at a position.
    public subscript(position: Int) -> Key {
        arena.read(position, below: endIndex) {
            return Key(
                raw: UnsafeRawPointer(keys.data!).load(fromByteOffset: position * keys.stride, as: usearch_key_t.self)
            )
        }
    }
}

/// The draws `sample` wrote into an arena, one per draw: its key, which of that key's entries it
/// is, the shown or earlier draw it hangs from, and where it is placed, read in place.
public struct DenseDraws {
    /// The arena the draws live in, held so they outlive no release of it.
    let arena: ArenaWrite
    /// The drawn keys the call wrote.
    var drawnKeys = usearch_keys_view_t()
    /// Which of its key's entries each draw is.
    var entryOrdinals: UnsafePointer<Int>?
    /// Each draw's parent and its distance.
    var drawnParents = usearch_matches_view_t()
    /// Where each draw is placed.
    var drawnCoordinates = usearch_rows_view_t()
    /// The draws written; fewer than wanted only once the region is exhausted.
    public internal(set) var count = 0

    /// The drawn keys.
    public var keys: DenseKeys { DenseKeys(arena: arena, keys: drawnKeys, endIndex: count) }
    /// Each draw's parent and its distance, one column per draw, a root holding none.
    public var parents: Matches { Matches(arena: arena, view: drawnParents, endIndex: count) }
    /// Where each draw is placed, one row of three values apiece.
    public var coordinates: DenseVectors<Float> {
        DenseVectors(arena: arena, vectors: drawnCoordinates, endIndex: count)
    }

    /// Which of its key's entries one draw is, in the order `get` writes them.
    public func entryOrdinal(_ draw: Int) -> Int {
        arena.read(draw, below: count) {
            return entryOrdinals![draw]
        }
    }
}

/// The keys each entry of a cursor pull links to, wrote into an arena as a ragged run and read in
/// place.
public struct DenseLinks: RandomAccessCollection {
    /// The arena the keys live in, held so they outlive no release of it.
    let arena: ArenaWrite
    /// The linked keys and offsets the call wrote.
    let links: usearch_keys_tape_view_t
    /// One past the last entry the pull wrote.
    public let endIndex: Int

    /// The first entry's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// The keys one entry links to, as the graph stored them when the pull ran.
    public subscript(entry: Int) -> [Key] {
        arena.read(entry, below: endIndex) {
            guard let offsets = links.offsets, let keys = links.keys else { return [] }
            return (offsets[entry] ..< offsets[entry + 1]).map { Key(raw: keys[$0]) }
        }
    }
}

/// The ragged sparse product `multiply` wrote into an arena: per left row, every pair its walk
/// scored, in traversal order; or the keys `maxsim` kept per query, largest score first; read in
/// place.
public struct DenseProducts: RandomAccessCollection {
    /// The arena the pairs live in, held so they outlive no release of it.
    let arena: ArenaWrite
    /// The keys, products and offsets the call wrote.
    var products = usearch_matches_tape_view_t()
    /// One past the last left row or query.
    public let endIndex: Int
    /// Pairs the walks scored across every left row, or keys scored across every query, before the
    /// wanted bound cut any.
    public internal(set) var required = 0

    /// The first query's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// One left row's pairs, in traversal order, or one query's keys, largest score first.
    public subscript(query: Int) -> QueryMatches {
        arena.read(query, below: endIndex) {
            let cells = products.offsets[query] ..< products.offsets[query + 1]
            return QueryMatches(arena: arena, keys: products.keys, scores: products.scores, cells: cells)
        }
    }
}

/// The monolith `saveBuffer` wrote into an arena, read in place.
public struct Monolith {
    /// The arena the bytes live in, held so they outlive no release of it.
    let arena: ArenaWrite
    /// The bytes the call wrote.
    var monolith = usearch_bytes_view_t()

    /// The bytes in the monolith.
    public var count: Int { monolith.length }

    /// Passes the serialized checkpoint to `body`, which must not keep the buffer past its return.
    public func withUnsafeBytes<Result, Failure: Swift.Error>(
        _ body: (UnsafeRawBufferPointer) throws(Failure) -> Result
    ) throws(Failure) -> Result {
        try arena.read { () throws(Failure) in
            try body(UnsafeRawBufferPointer(start: monolith.data, count: monolith.length))
        }
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
    /// The native handle, released by `deinit`. Immutable for the object's whole life, so the
    /// engine's own synchronization is all that sharing relies on.
    let raw: usearch_dense_index_t

    /// Adopts a handle a lifecycle verb just created.
    private init(raw: usearch_dense_index_t) { self.raw = raw }

    /// Frees the index once every deferred call naming it is final.
    deinit { usearch_dense_free(raw) }

    /// Initializes an empty, mutable index in host memory, one shard per memory domain; an empty
    /// list derives every node the process may allocate on.
    public static func initRam(
        _ options: DenseOptions,
        memoryDomains: [Int32] = []
    ) throws(USearchError) -> DenseIndex {
        let callState = CallState()
        let index = callState.slots(usearch_dense_index_t?.none)
        try callState.pass(memoryDomains) { (domains) throws(USearchError) in
            var arguments = usearch_dense_init_ram_t(
                index: index,
                outcome: callState.outcome,
                metric: options.metric.raw,
                quantization: options.quantization.raw,
                dimensions: options.dimensions,
                connectivity: options.connectivity,
                values_per_key: options.valuesPerKey,
                ram: usearch_ram_options_t(
                    memory_domains: domains,
                    memory_domains_count: memoryDomains.count,
                    shard_capacity: 0,
                    cpu_class_ceiling: options.cpuClassCeiling?.raw ?? usearch_cpu_class_unknown_k,
                    memory_limit: 0
                )
            )
            usearch_dense_init_ram(&arguments)
        }
        return try adopted(index, callState)
    }

    /// Initializes an empty, mutable index in device memory, sharded across the named devices and
    /// never replicated; an empty list derives one free device.
    public static func initVram(_ options: DenseOptions, devices: [Int32] = []) throws(USearchError) -> DenseIndex {
        let callState = CallState()
        let index = callState.slots(usearch_dense_index_t?.none)
        try callState.pass(devices) { (passedDevices) throws(USearchError) in
            var arguments = usearch_dense_init_vram_t(
                index: index,
                outcome: callState.outcome,
                metric: options.metric.raw,
                quantization: options.quantization.raw,
                dimensions: options.dimensions,
                connectivity: options.connectivity,
                values_per_key: options.valuesPerKey,
                vram: usearch_vram_options_t(
                    devices: passedDevices,
                    devices_count: devices.count,
                    shard_capacity: 0,
                    gpu_class_ceiling: options.gpuClassCeiling?.raw ?? usearch_gpu_class_unknown_k
                )
            )
            usearch_dense_init_vram(&arguments)
        }
        return try adopted(index, callState)
    }

    /// Copies a checkpoint into a new, mutable index in host memory. The geometry and metric come
    /// from the checkpoint's own headers, and part `p` lands whole on shard `p` modulo the shard
    /// count, wherever it was written from.
    ///
    /// A cancelled or failed load creates no index.
    public static func loadRam(
        _ path: String,
        config: DenseLoadRamConfig = DenseLoadRamConfig()
    ) throws(USearchError) -> DenseIndex {
        let callState = CallState()
        let index = callState.slots(usearch_dense_index_t?.none)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try callState.pass(path) { (path) throws(USearchError) in
            try callState.pass(config.memoryDomains) { (domains) throws(USearchError) in
                var arguments = usearch_dense_load_ram_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: callState.outcome,
                    completion: callState.completion,
                    path: path,
                    ram: usearch_ram_options_t(
                        memory_domains: domains,
                        memory_domains_count: config.memoryDomains.count,
                        shard_capacity: 0,
                        cpu_class_ceiling: config.cpuClassCeiling?.raw ?? usearch_cpu_class_unknown_k,
                        memory_limit: 0
                    ),
                    connectivity: 0,
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_dense_load_ram(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
    /// `loadRam`.
    public static func loadVram(
        _ path: String,
        config: DenseLoadVramConfig = DenseLoadVramConfig()
    ) throws(USearchError) -> DenseIndex {
        let callState = CallState()
        let index = callState.slots(usearch_dense_index_t?.none)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try callState.pass(path) { (path) throws(USearchError) in
            try callState.pass(config.devices) { (devices) throws(USearchError) in
                var arguments = usearch_dense_load_vram_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: callState.outcome,
                    completion: callState.completion,
                    path: path,
                    vram: usearch_vram_options_t(
                        devices: devices,
                        devices_count: config.devices.count,
                        shard_capacity: 0,
                        gpu_class_ceiling: config.gpuClassCeiling?.raw ?? usearch_gpu_class_unknown_k
                    ),
                    connectivity: 0,
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_dense_load_vram(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Copies a monolith from a buffer into a new index in host memory, as `loadRam` does a path.
    public static func loadRamBuffer(
        _ buffer: RawSpan,
        config: DenseLoadRamBufferConfig = DenseLoadRamBufferConfig()
    ) throws(USearchError) -> DenseIndex {
        let callState = CallState()
        let index = callState.slots(usearch_dense_index_t?.none)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try buffer.withUnsafeBytes { (bytes) throws(USearchError) in
            try callState.pass(config.memoryDomains) { (domains) throws(USearchError) in
                var arguments = usearch_dense_load_ram_buffer_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: callState.outcome,
                    completion: callState.completion,
                    buffer: usearch_bytes_view_t(bytes),
                    ram: usearch_ram_options_t(
                        memory_domains: domains,
                        memory_domains_count: config.memoryDomains.count,
                        shard_capacity: 0,
                        cpu_class_ceiling: config.cpuClassCeiling?.raw ?? usearch_cpu_class_unknown_k,
                        memory_limit: 0
                    ),
                    connectivity: 0,
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_dense_load_ram_buffer(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Copies a monolith from a buffer into a new index in device memory, as `loadVram` does a
    /// path.
    public static func loadVramBuffer(
        _ buffer: RawSpan,
        config: DenseLoadVramBufferConfig = DenseLoadVramBufferConfig()
    ) throws(USearchError) -> DenseIndex {
        let callState = CallState()
        let index = callState.slots(usearch_dense_index_t?.none)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try buffer.withUnsafeBytes { (bytes) throws(USearchError) in
            try callState.pass(config.devices) { (devices) throws(USearchError) in
                var arguments = usearch_dense_load_vram_buffer_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: callState.outcome,
                    completion: callState.completion,
                    buffer: usearch_bytes_view_t(bytes),
                    vram: usearch_vram_options_t(
                        devices: devices,
                        devices_count: config.devices.count,
                        shard_capacity: 0,
                        gpu_class_ceiling: config.gpuClassCeiling?.raw ?? usearch_gpu_class_unknown_k
                    ),
                    connectivity: 0,
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_dense_load_vram_buffer(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
    /// verbs report `readOnly`; the files must stay unmodified while it lives.
    public static func viewRam(_ path: String) throws(USearchError) -> DenseIndex {
        let callState = CallState()
        let index = callState.slots(usearch_dense_index_t?.none)
        try callState.pass(path) { (path) throws(USearchError) in
            var arguments = usearch_dense_view_ram_t(index: index, outcome: callState.outcome, path: path)
            usearch_dense_view_ram(&arguments)
        }
        return try adopted(index, callState)
    }

    /// Serves a checkpoint in place from NVMe drives, read through the executor's submission rings,
    /// as a read-only index.
    public static func viewNvme(
        _ path: String,
        config: DenseViewNvmeConfig = DenseViewNvmeConfig()
    ) throws(USearchError) -> DenseIndex {
        let callState = CallState()
        let index = callState.slots(usearch_dense_index_t?.none)
        let keyTableDomains = config.keyTableDomains ?? []
        try callState.pass(path) { (path) throws(USearchError) in
            try callState.pass(keyTableDomains) { (domains) throws(USearchError) in
                let keyTable = usearch_ram_options_t(
                    memory_domains: domains,
                    memory_domains_count: keyTableDomains.count,
                    shard_capacity: 0,
                    cpu_class_ceiling: config.cpuClassCeiling?.raw ?? usearch_cpu_class_unknown_k,
                    memory_limit: 0
                )
                withUnsafePointer(to: keyTable) { keyTable in
                    withPaths(config.paths, config.cpuClassCeiling) { nvme in
                        var arguments = usearch_dense_view_nvme_t(
                            index: index,
                            executor: config.executor?.raw,
                            outcome: callState.outcome,
                            completion: callState.completion,
                            path: path,
                            nvme: nvme,
                            ram: config.keyTableDomains == nil ? nil : keyTable
                        )
                        usearch_dense_view_nvme(&arguments)
                    }
                }
            }
        }
        return try adopted(index, callState)
    }

    /// Opens a checkpoint's manifest from NVMe drives as a writable index: it answers every read a
    /// view answers, and `refine` rewrites its parts one at a time. A lone monolith has no manifest
    /// to replace and refuses with `unsupported`, as do `paths` naming any.
    public static func loadNvme(
        _ path: String,
        config: DenseViewNvmeConfig = DenseViewNvmeConfig()
    ) throws(USearchError) -> DenseIndex {
        let callState = CallState()
        let index = callState.slots(usearch_dense_index_t?.none)
        let keyTableDomains = config.keyTableDomains ?? []
        try callState.pass(path) { (path) throws(USearchError) in
            try callState.pass(keyTableDomains) { (domains) throws(USearchError) in
                let keyTable = usearch_ram_options_t(
                    memory_domains: domains,
                    memory_domains_count: keyTableDomains.count,
                    shard_capacity: 0,
                    cpu_class_ceiling: config.cpuClassCeiling?.raw ?? usearch_cpu_class_unknown_k,
                    memory_limit: 0
                )
                withUnsafePointer(to: keyTable) { keyTable in
                    withPaths(config.paths, config.cpuClassCeiling) { nvme in
                        var arguments = usearch_dense_load_nvme_t(
                            index: index,
                            executor: config.executor?.raw,
                            outcome: callState.outcome,
                            completion: callState.completion,
                            path: path,
                            nvme: nvme,
                            ram: config.keyTableDomains == nil ? nil : keyTable
                        )
                        usearch_dense_load_nvme(&arguments)
                    }
                }
            }
        }
        return try adopted(index, callState)
    }

    /// Reads a checkpoint's geometry and metric without loading it - from the manifest's summary
    /// when the path names one, so no shard is opened.
    public static func metadataRead(_ path: String, executor: Executor? = nil) throws(USearchError) -> DenseMetadata {
        let callState = CallState()
        let metadata = callState.slots(DenseMetadata())
        try callState.pass(path) { (path) throws(USearchError) in
            var arguments = usearch_dense_metadata_read_t(
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: callState.completion,
                path: path,
                metadata: metadata.pointer(to: \.raw)!
            )
            usearch_dense_metadata_read(&arguments)
        }
        try callState.wait()
        return metadata.pointee
    }

    /// Reads geometry and metric metadata from a serialized monolith buffer.
    public static func metadataReadBuffer(
        _ buffer: RawSpan,
        executor: Executor? = nil
    ) throws(USearchError) -> DenseMetadata {
        let callState = CallState()
        let metadata = callState.slots(DenseMetadata())
        buffer.withUnsafeBytes { bytes in
            var arguments = usearch_dense_metadata_read_buffer_t(
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: callState.completion,
                buffer: usearch_bytes_view_t(bytes),
                metadata: metadata.pointer(to: \.raw)!
            )
            usearch_dense_metadata_read_buffer(&arguments)
        }
        try callState.wait()
        return metadata.pointee
    }

    /// Adopts the handle a blocking lifecycle verb wrote, or throws its failure.
    private static func adopted(
        _ index: UnsafeMutablePointer<usearch_dense_index_t?>,
        _ callState: CallState
    ) throws(USearchError) -> DenseIndex {
        try callState.wait()
        return DenseIndex(raw: try created(index))
    }

    /// The same verbs, blocking on `executor`, each refused while another call holds it.
    /// Binding-only: names the executor every call passes.
    public func on(_ executor: Executor) -> DenseIndexOn { DenseIndexOn(index: self, executor: executor) }

    /// The same verbs, submitted to `executor` and answered at once as a `Deferred`. Binding-only:
    /// names the executor and the deferred completion every call passes.
    public func deferred(_ executor: Executor) -> DenseIndexDeferredOn {
        DenseIndexDeferredOn(index: self, executor: executor)
    }

    /// The verbs on the calling thread.
    private var here: DenseIndexOn { DenseIndexOn(index: self, executor: nil) }

    /// Reserves capacity for a total number of entries, the current size included.
    public func reserve(_ capacity: Int) throws(USearchError) { try here.reserve(capacity) }

    /// Erases all entries, keeping the reserved capacity and the frozen geometry.
    public func clear() throws(USearchError) { try here.clear() }

    /// Reports the frozen metadata and the current counters in one call.
    public func describe() throws(USearchError) -> DenseIndexDescription { try here.describe() }

    /// The number of entries stored under a key; zero means the index holds none.
    public func count(key: Key) throws(USearchError) -> Int { try here.count(key: key) }

    /// Whether the index holds at least one entry under a key. Binding-only: calls `count`.
    public func contains(key: Key) throws(USearchError) -> Bool { try here.contains(key: key) }

    /// Retrieves every vector stored under a key, one row per entry sharing the key, in insertion
    /// order, converted to `Value`; an absent key yields no rows. The rows land in `arena`, read in
    /// place by the result until the arena's next call.
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
    /// ingest. A batch is all-or-nothing, and its entries become visible at once: every fallible step
    /// runs before the first entry identity is claimed, so a thrown failure leaves the index
    /// untouched. Every entry is claimed before the handler's first report, so it reports but cannot
    /// cancel.
    public func insert(
        keys: [Key],
        vectors: RowsView,
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) {
        try here.insert(keys: keys, vectors: vectors, config: config)
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, reporting the
    /// entries re-selected and changed and the islands healed; zero `changed` means settled.
    ///
    /// A build step rather than a repair: a cheap insertion pass followed by one refinement
    /// reaches a better graph, in less total time, than one expensive insertion pass - so a wide
    /// expansion factor belongs here rather than in `insert`. Searches may run throughout.
    @discardableResult
    public func refine(config: DenseRefineConfig = DenseRefineConfig()) throws(USearchError) -> DenseRefineReport {
        try here.refine(config: config)
    }

    /// Measures how much of the graph a search reaches from the root; entries it misses stay
    /// unfound until a refinement re-links them.
    public func health() throws(USearchError) -> DenseReachability { try here.health() }

    /// Resolves rows of queries into up to `wanted` matches apiece, written in `arena` - the native
    /// batch operation, parallelized internally on an executor. The result reads the arena in place
    /// until its next call. Never writes to the graph, so any number of these may overlap with
    /// concurrent insertions.
    public func search(
        _ queries: RowsView,
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Matches {
        try here.search(queries, wanted: wanted, arena: arena, config: config)
    }

    /// Late-interaction scoring of multi-row queries against multi-vector documents: every query
    /// row takes its best match within a key, and those maxima sum to that key's score.
    ///
    /// A document is one key holding its token rows, which is what a `valuesPerKey` above one
    /// exists for. `rowsPerQuery` is the token count each query occupies, `queries` holds them end
    /// to end, and up to `wanted` documents land in `arena` as one ragged run, one run per whole
    /// query rather than per token, largest score first; a zero `wanted` keeps every document the
    /// query scored. Approximate downward only: a token whose neighborhood the walk never entered
    /// contributes no maximum, so a key can score below its exact value but never above it. Runs on
    /// an inner-product index alone.
    ///
    /// See <https://arxiv.org/abs/2004.12832> - ColBERT, which introduced this scoring.
    public func maxsim(
        _ queries: RowsView,
        rowsPerQuery: Int,
        wanted: Int,
        arena: Arena,
        config: DenseMaxsimConfig = DenseMaxsimConfig()
    ) throws(USearchError) -> DenseProducts {
        try here.maxsim(queries, rowsPerQuery: rowsPerQuery, wanted: wanted, arena: arena, config: config)
    }

    /// The approximate sparse product of rows against the indexed vectors - the linear-algebra
    /// sibling of `search`, reporting every pair the traversal scored rather than only the
    /// nearest of them.
    ///
    /// Pairs arrive in traversal order rather than sorted, and hold no duplicates. A non-zero
    /// `wanted` keeps the first pairs each row's walk scored, and `required` says how many there
    /// were. The keys, products and row offsets land in `arena` as one ragged run. Runs on an
    /// inner-product index alone, since a product needs logits rather than distances.
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
    /// alone, and on RAM and NVMe handles alone, since a VRAM walk needs a device function as its
    /// lookup.
    ///
    /// See <https://arxiv.org/abs/2205.14135> - Flash Attention, whose recurrence this reuses.
    public func attend<Value: ScalarValue>(
        _ queries: RowsView,
        lookup: ValueLookup,
        as value: Value.Type,
        arena: Arena,
        config: DenseAttendConfig = DenseAttendConfig()
    ) throws(USearchError) -> DenseVectors<Value> {
        try here.attend(queries, lookup: lookup, as: value, arena: arena, config: config)
    }

    /// Moves the entries of `sources` into `targets`, keeping their links rather than
    /// re-inserting their vectors, then runs a round on every target that took entries. Several
    /// sources gather into one target, itself among them; one source cuts into one graph-local
    /// part per target; and one index as its only source and target recuts its own shards in
    /// place into as few as fit the placement's shard capacity.
    ///
    /// Every handle is checked before the first entry moves, so a refusal moves nothing, and
    /// every handle stays the caller's: a source the targets do not name ends empty and live.
    /// Readers of a RAM target keep searching throughout; a VRAM one admits no overlapping
    /// search.
    @discardableResult
    public func reshard(
        sources: [DenseIndex],
        targets: [DenseIndex],
        config: DenseReshardConfig = DenseReshardConfig()
    ) throws(USearchError) -> DenseRefineReport {
        try here.reshard(sources: sources, targets: targets, config: config)
    }

    /// Draws up to `wanted` keys around `config.focus`, or across the whole index, each hanging
    /// from a shown or earlier draw and placed in three dimensions beside the shown ones, written
    /// in `arena`; a larger `wanted` draws a superset. Only RAM samples yet.
    public func sample(
        wanted: Int,
        arena: Arena,
        config: DenseSampleConfig = DenseSampleConfig()
    ) throws(USearchError) -> DenseDraws {
        try here.sample(wanted: wanted, arena: arena, config: config)
    }

    /// Writes the index under a path, in the layout its extension picks - one `.safegraphs`
    /// monolith, or a `.safegraphs.json` manifest beside its shards. Demands a mutation-free
    /// window; concurrent readers keep searching throughout.
    public func save(path: String, config: DenseSaveConfig = DenseSaveConfig()) throws(USearchError) {
        try here.save(path: path, config: config)
    }

    /// Writes the index as one monolith, the same bytes `save(path:)` writes to a `.safegraphs`
    /// path, into `arena`.
    public func saveBuffer(
        arena: Arena,
        config: DenseSaveBufferConfig = DenseSaveBufferConfig()
    ) throws(USearchError) -> Monolith {
        try here.saveBuffer(arena: arena, config: config)
    }

    /// Opens a walk over the current contents, for export or migration. Each shard's publication
    /// watermark is read as the walk enters it, so an entry once passed is never revisited. Without
    /// `starts` it yields every entry as stored; with them, the start keys' own entries first in
    /// order, then the entries those link to breadth-first, each once.
    public func cursor(starts: [Key] = []) throws(USearchError) -> DenseCursor {
        try DenseCall(index: self, executor: nil, callState: CallState()).cursor(starts: starts)
    }

    /// Opens a resumable nearest-neighbor stream over one query row, so a caller that post-filters
    /// aggressively pulls until satisfied rather than guessing a wanted count upfront.
    public func searchStream(
        _ query: RowsView,
        config: DenseSearchStreamInitConfig = DenseSearchStreamInitConfig()
    ) throws(USearchError) -> DenseSearchStream {
        try here.searchStream(query, config: config)
    }
}

// MARK: - Receivers

/// An index's verbs on an executor, each blocking until final and refused while another call holds
/// it; see `DenseIndex.on(_:)` and the index's own verbs for what each one does.
///
/// A blocking call uses the arena it is given, and its result reads it until the arena's next
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
        try verb(DenseCall(index: index, executor: executor, callState: CallState())).wait()
    }

    /// Reserves capacity for a total number of entries, the current size included.
    public func reserve(_ capacity: Int) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.reserve(capacity) }
    }

    /// Erases all entries, keeping the reserved capacity and the frozen geometry.
    public func clear() throws(USearchError) { try run { (call) throws(USearchError) in try call.clear() } }

    /// Reports the frozen metadata and the current counters in one call.
    public func describe() throws(USearchError) -> DenseIndexDescription {
        try run { (call) throws(USearchError) in try call.describe() }
    }

    /// The number of entries stored under a key; zero means the index holds none.
    public func count(key: Key) throws(USearchError) -> Int {
        try run { (call) throws(USearchError) in try call.count(key) }
    }

    /// Binding-only: calls `count`.
    public func contains(key: Key) throws(USearchError) -> Bool { try count(key: key) != 0 }

    /// Retrieves every vector stored under a key into `arena`, one row per entry sharing the key.
    public func get<Value: ScalarValue>(
        key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> DenseVectors<Value> {
        try run { (call) throws(USearchError) in try call.get(key, as: value, arena: arena) }
    }

    /// Inserts rows of vectors under their keys, all or nothing.
    public func insert(
        keys: [Key],
        vectors: RowsView,
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.insert(keys, vectors, config: config) }
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, healing islands.
    @discardableResult
    public func refine(config: DenseRefineConfig = DenseRefineConfig()) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.refine(config: config) }
    }

    /// Measures how much of the graph a search reaches from the root.
    public func health() throws(USearchError) -> DenseReachability {
        try run { (call) throws(USearchError) in try call.health() }
    }

    /// Resolves rows of queries into up to `wanted` matches apiece, written in `arena`.
    public func search(
        _ queries: RowsView,
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Matches {
        try run { (call) throws(USearchError) in try call.search(queries, wanted: wanted, arena: arena, config: config)
        }
    }

    /// Scores multi-row queries against multi-vector documents by late interaction, into `arena`.
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

    /// The approximate sparse product of rows against the indexed vectors, written in `arena`.
    public func multiply(
        _ left: RowsView,
        arena: Arena,
        config: DenseMultiplyConfig = DenseMultiplyConfig()
    ) throws(USearchError) -> DenseProducts {
        try run { (call) throws(USearchError) in try call.multiply(left, arena: arena, config: config) }
    }

    /// Softmax attention of query heads over every indexed key their walks score, into `arena`.
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

    /// Moves the entries of `sources` into `targets`, then runs a round on every target that took
    /// entries.
    @discardableResult
    public func reshard(
        sources: [DenseIndex],
        targets: [DenseIndex],
        config: DenseReshardConfig = DenseReshardConfig()
    ) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.reshard(sources: sources, targets: targets, config: config) }
    }

    /// Closes a live refine plan and frees it whatever the call answers, reporting what the whole
    /// plan did to the graph; a round a stop left in hand closes as that stop would have.
    @discardableResult
    public func finish(_ plan: DenseRefinePlan) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Closes a live reshard plan and frees it whatever the call answers, reporting what the whole
    /// plan did; a plan whose entries have not all moved runs no round and reports zeros.
    @discardableResult
    public func finish(_ plan: DenseReshardPlan) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Draws up to `wanted` keys around `config.focus`, or across the whole index, into `arena`.
    public func sample(
        wanted: Int,
        arena: Arena,
        config: DenseSampleConfig = DenseSampleConfig()
    ) throws(USearchError) -> DenseDraws {
        try run { (call) throws(USearchError) in try call.sample(wanted: wanted, arena: arena, config: config) }
    }

    /// Writes the index under a path, in the layout its extension picks.
    public func save(path: String, config: DenseSaveConfig = DenseSaveConfig()) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.save(path, config: config) }
    }

    /// Writes the index into `arena` as one monolith, the bytes a `.safegraphs` path holds.
    public func saveBuffer(
        arena: Arena,
        config: DenseSaveBufferConfig = DenseSaveBufferConfig()
    ) throws(USearchError) -> Monolith {
        try run { (call) throws(USearchError) in try call.saveBuffer(arena: arena, config: config) }
    }

    /// Opens a stream whose every pull blocks on this executor.
    public func searchStream(
        _ query: RowsView,
        config: DenseSearchStreamInitConfig = DenseSearchStreamInitConfig()
    ) throws(USearchError) -> DenseSearchStream {
        try DenseCall(index: index, executor: executor, callState: CallState()).searchStream(query, config: config)
    }
}

/// An index's verbs submitted to an executor, each answering a `Deferred` at once; see
/// `DenseIndex.deferred(_:)` and the index's own verbs for what each one does.
///
/// A failure this binding catches before the call crosses throws at submission. Every object whose
/// memory a call reads, like rows built over a `Tensor`, stays alive until it is final, and memory
/// viewed from elsewhere must outlive it too. The index, the executor and the arena are viewed as a
/// blocking call views them, since releasing any of them waits for the calls naming it.
public struct DenseIndexDeferredOn: Sendable {
    /// The index every verb addresses.
    let index: DenseIndex
    /// The executor every verb is submitted to.
    let executor: Executor

    /// Submits one verb, whose call state its `Deferred` keeps until final.
    private func submit<Value>(
        _ verb: (DenseCall) throws(USearchError) -> Submission<Value>
    ) throws(USearchError) -> Deferred<Value> {
        Deferred(
            try verb(DenseCall(index: index, executor: executor, callState: CallState(usearch_completion_deferred_k)))
        )
    }

    /// Reserves capacity for a total number of entries, the current size included.
    public func reserve(_ capacity: Int) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.reserve(capacity) }
    }

    /// Erases all entries, keeping the reserved capacity and the frozen geometry.
    public func clear() throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.clear() }
    }

    /// Reports the frozen metadata and the current counters in one call.
    public func describe() throws(USearchError) -> Deferred<DenseIndexDescription> {
        try submit { (call) throws(USearchError) in try call.describe() }
    }

    /// The number of entries stored under a key; zero means the index holds none.
    public func count(key: Key) throws(USearchError) -> Deferred<Int> {
        try submit { (call) throws(USearchError) in try call.count(key) }
    }

    /// Retrieves every vector stored under a key into `arena`, one row per entry sharing the key.
    public func get<Value: ScalarValue>(
        key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> Deferred<DenseVectors<Value>> {
        try submit { (call) throws(USearchError) in try call.get(key, as: value, arena: arena) }
    }

    /// Inserts rows of vectors under their keys, all or nothing.
    public func insert(
        keys: [Key],
        vectors: RowsView,
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.insert(keys, vectors, config: config) }
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, healing islands.
    public func refine(
        config: DenseRefineConfig = DenseRefineConfig()
    ) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in try call.refine(config: config) }
    }

    /// Measures how much of the graph a search reaches from the root.
    public func health() throws(USearchError) -> Deferred<DenseReachability> {
        try submit { (call) throws(USearchError) in try call.health() }
    }

    /// Resolves rows of queries into up to `wanted` matches apiece, written in `arena`.
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

    /// Scores multi-row queries against multi-vector documents by late interaction, into `arena`.
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

    /// The approximate sparse product of rows against the indexed vectors, written in `arena`.
    public func multiply(
        _ left: RowsView,
        arena: Arena,
        config: DenseMultiplyConfig = DenseMultiplyConfig()
    ) throws(USearchError) -> Deferred<DenseProducts> {
        try submit { (call) throws(USearchError) in try call.multiply(left, arena: arena, config: config) }
    }

    /// Softmax attention of query heads over every indexed key their walks score, into `arena`.
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

    /// Moves the entries of `sources` into `targets`, then runs a round on every target that took
    /// entries.
    public func reshard(
        sources: [DenseIndex],
        targets: [DenseIndex],
        config: DenseReshardConfig = DenseReshardConfig()
    ) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in
            try call.reshard(sources: sources, targets: targets, config: config)
        }
    }

    /// Closes a live refine plan and frees it whatever the call answers.
    public func finish(_ plan: DenseRefinePlan) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Closes a live reshard plan and frees it whatever the call answers.
    public func finish(_ plan: DenseReshardPlan) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Draws up to `wanted` keys around `config.focus`, or across the whole index, into `arena`.
    public func sample(
        wanted: Int,
        arena: Arena,
        config: DenseSampleConfig = DenseSampleConfig()
    ) throws(USearchError) -> Deferred<DenseDraws> {
        try submit { (call) throws(USearchError) in try call.sample(wanted: wanted, arena: arena, config: config) }
    }

    /// Writes the index under a path, in the layout its extension picks.
    public func save(path: String, config: DenseSaveConfig = DenseSaveConfig()) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.save(path, config: config) }
    }

    /// Writes the index into `arena` as one monolith, the bytes a `.safegraphs` path holds.
    public func saveBuffer(
        arena: Arena,
        config: DenseSaveBufferConfig = DenseSaveBufferConfig()
    ) throws(USearchError) -> Deferred<Monolith> {
        try submit { (call) throws(USearchError) in try call.saveBuffer(arena: arena, config: config) }
    }
}

/// The whole queries the rows hold, which only this side can count, since the engine takes the
/// query count rather than the rows.
private func wholeQueries(_ queries: RowsView, _ rowsPerQuery: Int) throws(USearchError) -> Int {
    guard rowsPerQuery > 0, queries.rows % rowsPerQuery == 0 else {
        throw USearchError(status: .shapeMismatch, message: "the query rows do not divide into whole queries")
    }
    return queries.rows / rowsPerQuery
}

// MARK: - Calls

/// The index, executor and call state one verb lowers into its argument structure, and the one
/// place each verb is written. Every verb points the library at a slot holding its result, which
/// a blocking call reads at once and a deferred one once final.
struct DenseCall {
    /// The index the call addresses.
    let index: DenseIndex
    /// The executor the call runs on, or nil for the calling thread.
    let executor: Executor?
    /// Where the call's outcome and slots live until it is final.
    let callState: CallState

    /// Whether the call blocks or returns once submitted.
    private var completion: usearch_completion_t { callState.completion }

    func reserve(_ capacity: Int) throws(USearchError) -> Submission<Void> {
        var arguments = usearch_dense_reserve_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            capacity: capacity
        )
        usearch_dense_reserve(&arguments)
        return try callState.submitted()
    }

    func clear() throws(USearchError) -> Submission<Void> {
        var arguments = usearch_dense_clear_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion
        )
        usearch_dense_clear(&arguments)
        return try callState.submitted()
    }

    func describe() throws(USearchError) -> Submission<DenseIndexDescription> {
        let description = callState.slots(DenseIndexDescription())
        var arguments = usearch_dense_describe_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            description: description.pointer(to: \.raw)!
        )
        usearch_dense_describe(&arguments)
        return try callState.submitted(description)
    }

    func count(_ key: Key) throws(USearchError) -> Submission<Int> {
        let entries = callState.slots(0)
        var arguments = usearch_dense_count_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            key: key.raw,
            entries: entries
        )
        usearch_dense_count(&arguments)
        return try callState.submitted(entries)
    }

    func get<Value: ScalarValue>(
        _ key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> Submission<DenseVectors<Value>> {
        let vectors = callState.slots(DenseVectors<Value>(arena: ArenaWrite(arena), endIndex: 0))
        var arguments = usearch_dense_get_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            key: key.raw,
            vectors: vectors.pointer(to: \.vectors)!,
            written: vectors.pointer(to: \.endIndex)!,
            arena: arena.raw,
            outputs_scalar: Value.scalarKind.raw
        )
        usearch_dense_get(&arguments)
        return try callState.submitted(vectors)
    }

    func insert(
        _ keys: [Key],
        _ vectors: RowsView,
        config: DenseInsertConfig
    ) throws(USearchError) -> Submission<Void> {
        // The key run holds no count of its own, so only this side can see it disagree.
        guard keys.count == vectors.rows else {
            throw USearchError(status: .shapeMismatch, message: "the key count does not match the vector row count")
        }
        callState.anchor(vectors.owner)
        callState.anchor(config.stats)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try callState.pass(keys) { (keys) throws(USearchError) in
            var arguments = usearch_dense_insert_t(
                index: index.raw,
                executor: executor?.raw,
                outcome: callState.outcome,
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
        return try callState.submitted()
    }

    func refine(config: DenseRefineConfig) throws(USearchError) -> Submission<DenseRefineReport> {
        callState.anchor(config.stats)
        callState.anchor(config.plan)
        let report = callState.slots(DenseRefineReport())
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        var arguments = usearch_dense_refine_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            report: report.pointer(to: \.raw)!,
            refine: usearch_refine_options_t(
                expansion_factor: config.expansionFactor,
                memory_limit: config.memoryLimit
            ),
            progress: progress,
            stats: config.stats?.raw,
            plan: config.plan?.raw.pointer(to: \.handle),
            phase: config.plan?.raw.pointer(to: \.phase.raw)
        )
        usearch_dense_refine(&arguments)
        return try callState.submitted(report)
    }

    func finish(_ plan: DenseRefinePlan) throws(USearchError) -> Submission<DenseRefineReport> {
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_dense_refine_plan_finish_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            plan: plan.raw.pointee.handle,
            report: report.pointer(to: \.raw)!
        )
        usearch_dense_refine_plan_finish(&arguments)
        plan.raw.pointee.handle = nil
        return try callState.submitted(report)
    }

    func health() throws(USearchError) -> Submission<DenseReachability> {
        let reachability = callState.slots(DenseReachability())
        var arguments = usearch_dense_health_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            reachability: reachability.pointer(to: \.raw)!
        )
        usearch_dense_health(&arguments)
        return try callState.submitted(reachability)
    }

    func search(
        _ queries: RowsView,
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig
    ) throws(USearchError) -> Submission<Matches> {
        callState.anchor(queries.owner)
        callState.anchor(config.stats)
        let matches = callState.slots(Matches(arena: ArenaWrite(arena), endIndex: queries.rows))
        var arguments = usearch_dense_search_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            count: queries.rows,
            queries: queries.raw,
            matches: matches.pointer(to: \.view)!,
            arena: arena.raw,
            wanted: wanted,
            expansion_factor: config.expansionFactor,
            traversal: config.traversal.raw,
            predicate: callState.pass(config.predicate),
            hints: nil,
            progress: callState.pass(config.progress, steps: config.progressSteps),
            stats: config.stats?.raw
        )
        usearch_dense_search(&arguments)
        return try callState.submitted(matches)
    }

    func maxsim(
        _ queries: RowsView,
        rowsPerQuery: Int,
        wanted: Int,
        arena: Arena,
        config: DenseMaxsimConfig
    ) throws(USearchError) -> Submission<DenseProducts> {
        let count = try wholeQueries(queries, rowsPerQuery)
        callState.anchor(queries.owner)
        callState.anchor(config.stats)
        let products = callState.slots(DenseProducts(arena: ArenaWrite(arena), endIndex: count))
        var arguments = usearch_dense_maxsim_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            count: count,
            queries: queries.raw,
            query_offsets: nil,
            rows_per_query: rowsPerQuery,
            matches: products.pointer(to: \.products)!,
            written: nil,
            required: products.pointer(to: \.required)!,
            arena: arena.raw,
            wanted: wanted,
            expansion_factor: config.expansionFactor,
            traversal: Traversal.walk.raw,
            predicate: usearch_predicate_t(),
            hints: nil,
            progress: callState.pass(config.progress, steps: config.progressSteps),
            stats: config.stats?.raw
        )
        usearch_dense_maxsim(&arguments)
        return try callState.submitted(products)
    }

    func multiply(
        _ left: RowsView,
        arena: Arena,
        config: DenseMultiplyConfig
    ) throws(USearchError) -> Submission<DenseProducts> {
        callState.anchor(left.owner)
        callState.anchor(config.stats)
        let products = callState.slots(DenseProducts(arena: ArenaWrite(arena), endIndex: left.rows))
        var arguments = usearch_dense_multiply_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            count: left.rows,
            left: left.raw,
            products: products.pointer(to: \.products)!,
            written: nil,
            required: products.pointer(to: \.required)!,
            arena: arena.raw,
            wanted: config.wanted,
            expansion_factor: config.expansionFactor,
            traversal: Traversal.walk.raw,
            predicate: usearch_predicate_t(),
            progress: callState.pass(config.progress, steps: config.progressSteps),
            stats: config.stats?.raw
        )
        usearch_dense_multiply(&arguments)
        return try callState.submitted(products)
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
        callState.anchor(queries.owner)
        callState.anchor(lookup.values.owner)
        callState.anchor(config.stats)
        let outputs = callState.slots(DenseVectors<Value>(arena: ArenaWrite(arena), endIndex: queries.rows))
        var arguments = usearch_dense_attend_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            count: queries.rows,
            queries: queries.raw,
            head_count: config.headCount,
            value_lookup: valueLookupTrampoline,
            value_lookup_state: callState.place(Handler(lookup)),
            carry: UnsafeMutableRawPointer(config.carry?.baseAddress)?
                .assumingMemoryBound(to: usearch_dense_softmax_state_t.self),
            outputs: outputs.pointer(to: \.vectors)!,
            arena: arena.raw,
            outputs_scalar: Value.scalarKind.raw,
            expansion_factor: config.expansionFactor,
            score_scale: config.scoreScale,
            traversal: Traversal.walk.raw,
            predicate: usearch_predicate_t(),
            progress: callState.pass(config.progress, steps: config.progressSteps),
            stats: config.stats?.raw
        )
        usearch_dense_attend(&arguments)
        return try callState.submitted(outputs)
    }

    func reshard(
        sources: [DenseIndex],
        targets: [DenseIndex],
        config: DenseReshardConfig
    ) throws(USearchError) -> Submission<DenseRefineReport> {
        for handle in sources + targets { callState.anchor(handle) }
        callState.anchor(config.stats)
        callState.anchor(config.plan)
        let report = callState.slots(DenseRefineReport())
        let ram = config.memoryDomains.map { domains in
            callState.slots(
                usearch_ram_options_t(
                    memory_domains: callState.copied(domains),
                    memory_domains_count: domains.count,
                    shard_capacity: config.shardCapacity,
                    cpu_class_ceiling: usearch_cpu_class_unknown_k,
                    memory_limit: 0
                )
            )
        }
        let vram = config.devices.map { devices in
            callState.slots(
                usearch_vram_options_t(
                    devices: callState.copied(devices),
                    devices_count: devices.count,
                    shard_capacity: config.shardCapacity,
                    gpu_class_ceiling: usearch_gpu_class_unknown_k
                )
            )
        }
        let nvme = config.paths.map { paths in
            let bytes = callState.copied(paths.flatMap(\.utf8))
            var offset = 0
            let views = paths.map { path in
                defer { offset += path.utf8.count }
                return usearch_bytes_view_t(data: bytes.map { UnsafePointer($0 + offset) }, length: path.utf8.count)
            }
            return callState.slots(
                usearch_nvme_options_t(
                    paths: callState.copied(views),
                    paths_count: views.count,
                    cpu_class_ceiling: usearch_cpu_class_unknown_k
                )
            )
        }
        var arguments = usearch_dense_reshard_t(
            sources_count: sources.count,
            sources: callState.copied(sources.map { Optional($0.raw) }),
            targets_count: targets.count,
            targets: callState.copied(targets.map { Optional($0.raw) }),
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            ram: ram,
            vram: vram,
            nvme: nvme,
            report: report.pointer(to: \.raw)!,
            refine: usearch_refine_options_t(
                expansion_factor: config.expansionFactor,
                memory_limit: config.memoryLimit
            ),
            seed: config.seed,
            progress: callState.pass(config.progress, steps: config.progressSteps),
            stats: config.stats?.raw,
            plan: config.plan?.raw.pointer(to: \.handle),
            phase: config.plan?.raw.pointer(to: \.phase.raw)
        )
        usearch_dense_reshard(&arguments)
        return try callState.submitted(report)
    }

    func finish(_ plan: DenseReshardPlan) throws(USearchError) -> Submission<DenseRefineReport> {
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_dense_reshard_plan_finish_t(
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            plan: plan.raw.pointee.handle,
            report: report.pointer(to: \.raw)!
        )
        usearch_dense_reshard_plan_finish(&arguments)
        plan.raw.pointee.handle = nil
        return try callState.submitted(report)
    }

    func sample(wanted: Int, arena: Arena, config: DenseSampleConfig) throws(USearchError) -> Submission<DenseDraws> {
        // The call reads one count for both, so only this side can see the keys and places
        // disagree.
        guard config.shownCoordinates.count == 3 * config.shown.count else {
            throw USearchError(status: .shapeMismatch, message: "the shown keys need three coordinates apiece")
        }
        callState.anchor(config.stats)
        let draws = callState.slots(DenseDraws(arena: ArenaWrite(arena)))
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        func view(_ keys: UnsafePointer<Key>?) -> usearch_keys_view_t {
            usearch_keys_view_t(
                data: UnsafeRawPointer(keys)?.assumingMemoryBound(to: usearch_key_t.self),
                stride: MemoryLayout<usearch_key_t>.stride
            )
        }
        try callState.pass(config.shown) { (shown) throws(USearchError) in
            try callState.pass(config.focus) { (focus) throws(USearchError) in
                try callState.pass(config.shownCoordinates) { (shownCoordinates) throws(USearchError) in
                    var arguments = usearch_dense_sample_t(
                        index: index.raw,
                        executor: executor?.raw,
                        outcome: callState.outcome,
                        completion: completion,
                        shown_count: config.shown.count,
                        shown: view(shown),
                        shown_coordinates: usearch_rows_view_t(
                            data: UnsafeRawPointer(shownCoordinates),
                            stride: 3 * MemoryLayout<Float>.stride,
                            columns: 3,
                            scalar: usearch_scalar_f32_k
                        ),
                        focus_count: config.focus.count,
                        focus: view(focus),
                        keys: draws.pointer(to: \.drawnKeys)!,
                        entry_ordinals: draws.pointer(to: \.entryOrdinals)!,
                        parents: draws.pointer(to: \.drawnParents)!,
                        coordinates: draws.pointer(to: \.drawnCoordinates)!,
                        written: draws.pointer(to: \.count)!,
                        arena: arena.raw,
                        wanted: wanted,
                        seed: config.seed,
                        expansion_factor: config.expansionFactor,
                        progress: progress,
                        stats: config.stats?.raw
                    )
                    usearch_dense_sample(&arguments)
                }
            }
        }
        return try callState.submitted(draws)
    }

    func save(_ path: String, config: DenseSaveConfig) throws(USearchError) -> Submission<Void> {
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try callState.pass(path) { (path) throws(USearchError) in
            var arguments = usearch_dense_save_t(
                index: index.raw,
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: completion,
                path: path,
                progress: progress
            )
            usearch_dense_save(&arguments)
        }
        return try callState.submitted()
    }

    func saveBuffer(arena: Arena, config: DenseSaveBufferConfig) throws(USearchError) -> Submission<Monolith> {
        let monolith = callState.slots(Monolith(arena: ArenaWrite(arena)))
        var arguments = usearch_dense_save_buffer_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            monolith: monolith.pointer(to: \.monolith)!,
            arena: arena.raw,
            progress: callState.pass(config.progress, steps: config.progressSteps)
        )
        usearch_dense_save_buffer(&arguments)
        return try callState.submitted(monolith)
    }

    func cursor(starts: [Key]) throws(USearchError) -> DenseCursor {
        let cursor = callState.slots(usearch_dense_cursor_t?.none)
        try callState.pass(starts) { (run) throws(USearchError) in
            var arguments = usearch_dense_cursor_init_t(
                index: index.raw,
                executor: nil,
                outcome: callState.outcome,
                starts_count: starts.count,
                starts: UnsafeRawPointer(run)?.assumingMemoryBound(to: usearch_key_t.self),
                cursor: cursor
            )
            usearch_dense_cursor_init(&arguments)
        }
        try callState.wait()
        return DenseCursor(raw: try created(cursor), index: index)
    }

    func searchStream(
        _ query: RowsView,
        config: DenseSearchStreamInitConfig
    ) throws(USearchError) -> DenseSearchStream {
        // The C call takes one row and no count, so only this side can see a matrix holding more.
        guard query.rows == 1 else {
            throw USearchError(status: .shapeMismatch, message: "a search stream walks one query row")
        }
        let stream = callState.slots(usearch_dense_search_stream_t?.none)
        var arguments = usearch_dense_search_stream_init_t(
            stream: stream,
            index: index.raw,
            outcome: callState.outcome,
            query: query.raw,
            expansion_factor: config.expansionFactor
        )
        usearch_dense_search_stream_init(&arguments)
        try callState.wait()
        return DenseSearchStream(raw: try created(stream), index: index, executor: executor)
    }
}

/// The admission trampoline the engine calls from the executor's threads with a batch of keys.
private let admissionTrampoline: usearch_predicate_callback_t = { keys, admissions, count, state in
    let admits = Unmanaged<Handler<@Sendable (Key) -> Bool>>.fromOpaque(state!).takeUnretainedValue().body
    for index in 0 ..< count {
        admissions![index] = admits(Key(raw: keys![index])) ? usearch_admit_k : usearch_reject_k
    }
}

extension CallState {
    /// Passes an optional predicate as the struct a search takes; nil passes nulls.
    func pass(_ predicate: (@Sendable (Key) -> Bool)?) -> usearch_predicate_t {
        guard let predicate else { return usearch_predicate_t() }
        return usearch_predicate_t(
            callback: admissionTrampoline,
            state: place(Handler(predicate)),
            cost: usearch_predicate_cost_cheap_k
        )
    }
}

/// The value-lookup trampoline the engine calls from the executor's threads, once per scored batch.
private let valueLookupTrampoline: usearch_value_lookup_t = { state, keys, count, scalar, columns, rows in
    let lookup = Unmanaged<Handler<ValueLookup>>.fromOpaque(state!).takeUnretainedValue().body
    let values = lookup.values
    // The library reads rows of its own shape, so values of another one answer no row at all.
    guard scalar == values.scalar.raw, columns == values.columns else {
        return rows!.update(repeating: nil, count: count)
    }
    for position in 0 ..< count {
        let row = lookup.row(Key(raw: keys![position]))
        rows![position] = row.flatMap { $0 >= 0 && $0 < values.rows ? values.data + $0 * values.stride : nil }
    }
}

// MARK: - Cursors & Streams

/// A walk over an index's contents; see `DenseIndex.cursor()`.
///
/// The walk is sequential, so it stays a batch operation - export, migration, and visualization all
/// move whole batches. The cursor keeps no executor: its pulls run on the calling thread, or block
/// on the executor `on(_:)` names for each of them, and it keeps its index alive.
public final class DenseCursor {
    /// The native handle, released by `deinit` and never reassigned.
    private let raw: usearch_dense_cursor_t
    /// The index the walk covers, held so the cursor cannot outlive it.
    private let index: DenseIndex

    init(raw: usearch_dense_cursor_t, index: DenseIndex) {
        self.raw = raw
        self.index = index
    }

    /// Releases the cursor once every deferred call naming it is final.
    deinit { usearch_dense_cursor_free(raw) }

    /// Copies up to `wanted` next keys into `arena`, walking the key column alone; an empty batch
    /// means the walk is exhausted.
    public func next(wanted: Int, arena: Arena) throws(USearchError) -> DenseKeys {
        try pull(wanted, false, arena, nil).keys
    }

    /// Copies up to `wanted` next keys into `arena`, each with the keys it links to; an empty batch
    /// means the walk is exhausted. The vectors themselves come from `get`.
    public func nextLinks(wanted: Int, arena: Arena) throws(USearchError) -> (keys: DenseKeys, links: DenseLinks) {
        let (keys, links) = try pull(wanted, true, arena, nil)
        return (keys, DenseLinks(arena: ArenaWrite(arena), links: links, endIndex: keys.count))
    }

    /// The receiver whose pulls block on `executor`, named again for every pull. Binding-only.
    public func on(_ executor: Executor) -> DenseCursorOn { DenseCursorOn(cursor: self, executor: executor) }

    /// Advances the walk by one batch on `executor`, or the calling thread when nil, exporting each
    /// entry's links only when `linked` asks.
    fileprivate func pull(
        _ wanted: Int,
        _ linked: Bool,
        _ arena: Arena,
        _ executor: Executor?
    ) throws(USearchError) -> (keys: DenseKeys, links: usearch_keys_tape_view_t) {
        let callState = CallState()
        let keys = callState.slots(DenseKeys(arena: ArenaWrite(arena), endIndex: 0))
        let links = callState.slots(usearch_keys_tape_view_t())
        var arguments = usearch_dense_cursor_next_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: callState.completion,
            cursor: raw,
            keys: keys.pointer(to: \.keys)!,
            links: linked ? links : nil,
            written: keys.pointer(to: \.endIndex)!,
            arena: arena.raw,
            wanted: wanted
        )
        usearch_dense_cursor_next(&arguments)
        try callState.wait()
        return (keys.pointee, links.pointee)
    }
}

/// A cursor's pulls blocking on one executor, as `DenseCursor.on(_:)` makes them. Binding-only.
public struct DenseCursorOn {
    /// The cursor the pulls advance.
    let cursor: DenseCursor
    /// The executor every pull blocks on.
    let executor: Executor

    /// Copies up to `wanted` next keys into `arena` on the executor; an empty batch means the walk
    /// is exhausted.
    public func next(wanted: Int, arena: Arena) throws(USearchError) -> DenseKeys {
        try cursor.pull(wanted, false, arena, executor).keys
    }

    /// Copies up to `wanted` next keys into `arena` on the executor, with the keys each links to.
    public func nextLinks(wanted: Int, arena: Arena) throws(USearchError) -> (keys: DenseKeys, links: DenseLinks) {
        let (keys, links) = try cursor.pull(wanted, true, arena, executor)
        return (keys, DenseLinks(arena: ArenaWrite(arena), links: links, endIndex: keys.count))
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

    /// Releases the stream and the walk state it keeps, once every deferred call naming it is
    /// final.
    deinit { usearch_dense_search_stream_free(raw) }

    /// Pulls up to `wanted` more matches into `arena`, as the one query of a search, nearest first;
    /// an empty query means the traversal is drained.
    public func next(wanted: Int, arena: Arena) throws(USearchError) -> Matches {
        let callState = CallState()
        let matches = callState.slots(Matches(arena: ArenaWrite(arena), endIndex: 1))
        var arguments = usearch_dense_search_stream_next_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: callState.completion,
            stream: raw,
            matches: matches.pointer(to: \.view)!,
            arena: arena.raw,
            wanted: wanted
        )
        usearch_dense_search_stream_next(&arguments)
        try callState.wait()
        return matches.pointee
    }
}

extension usearch_bytes_view_t {
    /// Views a byte buffer as the pointer-and-length pair every buffer argument takes.
    init(_ bytes: UnsafeRawBufferPointer) {
        self.init(data: bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), length: bytes.count)
    }
}
