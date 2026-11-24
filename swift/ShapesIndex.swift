/**
 *  @file swift/ShapesIndex.swift
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Similarity search over 3D point clouds of any size - molecules, conformers and protein
 *      backbones - under Chamfer, Hausdorff, Gromov-Wasserstein and Wasserstein-Procrustes.
 *
 *  Every batch crosses as one tape of points, so a shape is its coordinates, `x`, `y` and `z` of
 *  each point in turn, and a batch an array of them; the scalar is each call's own type parameter.
 *  The verbs, receivers, plans and reports are the dense index's.
 */

import USearchC

// MARK: - Options

/// A shape distance kind, spelled as the library spells it. Each is invariant to rotation,
/// reflection, translation and point order.
public struct ShapesMetric: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_shapes_metric_t

    /// No kind named - an unset field, refused by every constructor.
    public static let unknown = ShapesMetric(raw: usearch_shapes_metric_unknown_k)
    /// The mean distance from every point to its nearest partner, after aligning principal axes.
    public static let chamfer = ShapesMetric(raw: usearch_shapes_metric_chamfer_k)
    /// The largest such distance.
    public static let hausdorff = ShapesMetric(raw: usearch_shapes_metric_hausdorff_k)
    /// The mismatch between each side's own internal distances, needing no superposition.
    public static let gromovWasserstein = ShapesMetric(raw: usearch_shapes_metric_gromov_wasserstein_k)
    /// An exact transport plan alternated with the best rotation or reflection.
    public static let wassersteinProcrustes = ShapesMetric(raw: usearch_shapes_metric_wasserstein_procrustes_k)

    /// The metric rendered as the library spells it.
    public var description: String { String(cString: usearch_shapes_metric_name(raw)) }

    /// Reads a spelling back, and nil for one no metric holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_shapes_metric_from_name, usearch_shapes_metric_unknown_k) else {
            return nil
        }
        self.raw = raw
    }

    init(raw: usearch_shapes_metric_t) { self.raw = raw }
}

/// The construction options of a shapes index, as `initRam` and `initVram` freeze them.
public struct ShapesOptions: Sendable {
    /// The distance kind.
    public var metric: ShapesMetric
    /// The scalar points are stored in, `f16` or `f32`.
    public var quantization: Scalar
    /// The point count every entry must have; zero admits any.
    public var points: Int
    /// The bound on connections per node in the graph; zero derives.
    public var connectivity: Int
    /// The bound on entries sharing one key; zero is unbounded.
    public var valuesPerKey: Int
    /// The newest class dispatch may pick, intersected with what the machine runs; nil means no
    /// ceiling.
    public var cpuClassCeiling: CPUClass?
    /// The newest class dispatch may pick on the named devices; nil means no ceiling.
    public var gpuClassCeiling: GPUClass?

    /// A Chamfer index over `f32` points of any count, with every optional field derived.
    public init(
        metric: ShapesMetric = .chamfer,
        quantization: Scalar = .f32,
        points: Int = 0,
        connectivity: Int = 0,
        valuesPerKey: Int = 0,
        cpuClassCeiling: CPUClass? = nil,
        gpuClassCeiling: GPUClass? = nil
    ) {
        self.metric = metric
        self.quantization = quantization
        self.points = points
        self.connectivity = connectivity
        self.valuesPerKey = valuesPerKey
        self.cpuClassCeiling = cpuClassCeiling
        self.gpuClassCeiling = gpuClassCeiling
    }
}

/// The metric and geometry a checkpoint froze - everything a reader learns without opening it.
public struct ShapesMetadata: Sendable {
    /// The library's own record.
    var raw = usearch_shapes_metadata_t()

    /// The distance kind the checkpoint scores with.
    public var metric: ShapesMetric { ShapesMetric(raw: raw.metric) }
    /// The scalar its points are stored in.
    public var quantization: Scalar { Scalar(raw: raw.quantization) }
    /// The point count every entry has, or zero for any.
    public var points: Int { raw.points }
    /// The connections-per-node bound the graph was built with.
    public var connectivity: Int { raw.connectivity }
}

/// A live index's frozen metadata and current counters, each read once.
public struct ShapesIndexDescription: Sendable {
    /// The library's own record.
    var raw = usearch_shapes_index_description_t()

    /// The frozen metric, quantization, point count and connectivity, as a checkpoint reports them.
    public var metadata: ShapesMetadata { ShapesMetadata(raw: raw.metadata) }
    /// The stored entries.
    public var size: Int { raw.size }
    /// The entries hosted without growing again.
    public var capacity: Int { raw.capacity }
    /// The points across every stored entry.
    public var totalPoints: Int { raw.total_points }
    /// The bytes the handle holds, point storage included.
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

/// The entries one key holds, and the extent of their points.
public struct ShapesKeyStats: Equatable, Sendable {
    /// The library's own record.
    var raw = usearch_shapes_key_stats_t()

    /// The entries stored under the key; zero means the index holds none.
    public var entries: Int { raw.entries }
    /// Their points summed.
    public var totalPoints: Int { raw.total_points }
    /// The points of the largest of them.
    public var largestPoints: Int { raw.largest_points }

    /// Equal in every count the call reported.
    public static func == (left: ShapesKeyStats, right: ShapesKeyStats) -> Bool {
        (left.entries, left.totalPoints, left.largestPoints) == (right.entries, right.totalPoints, right.largestPoints)
    }
}

/// The knobs of `refine` on a shapes index.
public typealias ShapesRefineConfig = BasicRefineConfig<ShapesRefinePlan>

/// The knobs of `reshard` on shapes indexes.
public typealias ShapesReshardConfig = BasicReshardConfig<ShapesReshardPlan>

/// One shapes refine kept live across calls; see `DenseRefinePlan`, whose rules it follows.
public final class ShapesRefinePlan: @unchecked Sendable {
    /// The handle the library opens, and the phase its last call stopped in, at addresses that hold
    /// for a deferred call.
    let raw: UnsafeMutablePointer<(handle: usearch_shapes_refine_plan_t?, phase: PhaseProgress)>

    public init() {
        raw = .allocate(capacity: 1)
        raw.initialize(to: (nil, PhaseProgress()))
    }

    deinit { raw.deallocate() }

    /// Where the last call naming the plan stopped, `done` once its round ended.
    public var phase: PhaseProgress { raw.pointee.phase }
}

/// One shapes reshard kept live across calls; see `DenseReshardPlan`, whose rules it follows.
public final class ShapesReshardPlan: @unchecked Sendable {
    /// The handle the library opens, and the phase its last call stopped in, at addresses that hold
    /// for a deferred call.
    let raw: UnsafeMutablePointer<(handle: usearch_shapes_reshard_plan_t?, phase: PhaseProgress)>

    public init() {
        raw = .allocate(capacity: 1)
        raw.initialize(to: (nil, PhaseProgress()))
    }

    deinit { raw.deallocate() }

    /// Where the last call naming the plan stopped, `done` once its rounds ended.
    public var phase: PhaseProgress { raw.pointee.phase }
}

// MARK: - Index

/// A Similarity Search Index over 3D point clouds, in RAM or VRAM, or viewed from NVMe.
///
/// A handle may be shared across threads, so no method here takes a lock: insertions and searches
/// overlap freely, and saving demands a mutation-free window. Every verb called on the index runs
/// on the calling thread; `on(_:)` runs the same verbs on an executor, and `deferred(_:)` submits
/// them there.
public final class ShapesIndex: @unchecked Sendable {
    /// The native handle, released by `deinit`.
    let raw: usearch_shapes_index_t

    private init(raw: usearch_shapes_index_t) { self.raw = raw }

    /// Frees the index once every deferred call naming it is final.
    deinit { usearch_shapes_free(raw) }

    /// Initializes an empty, mutable index in host memory, one shard per memory domain; an empty
    /// list derives every node the process may allocate on.
    public static func initRam(
        _ options: ShapesOptions = ShapesOptions(),
        memoryDomains: [Int32] = []
    ) throws(USearchError) -> ShapesIndex {
        let callState = CallState()
        let index = callState.slots(usearch_shapes_index_t?.none)
        try callState.pass(memoryDomains) { (domains) throws(USearchError) in
            var arguments = usearch_shapes_init_ram_t(
                index: index,
                outcome: callState.outcome,
                metric: options.metric.raw,
                quantization: options.quantization.raw,
                points: options.points,
                connectivity: options.connectivity,
                values_per_key: options.valuesPerKey,
                ram: ramOptions(domains, memoryDomains.count, options.cpuClassCeiling)
            )
            usearch_shapes_init_ram(&arguments)
        }
        return try adopted(index, callState)
    }

    /// Initializes an empty, mutable index in device memory; an empty list derives one free device.
    public static func initVram(
        _ options: ShapesOptions = ShapesOptions(),
        devices: [Int32] = []
    ) throws(USearchError) -> ShapesIndex {
        let callState = CallState()
        let index = callState.slots(usearch_shapes_index_t?.none)
        try callState.pass(devices) { (passedDevices) throws(USearchError) in
            var arguments = usearch_shapes_init_vram_t(
                index: index,
                outcome: callState.outcome,
                metric: options.metric.raw,
                quantization: options.quantization.raw,
                points: options.points,
                connectivity: options.connectivity,
                values_per_key: options.valuesPerKey,
                vram: vramOptions(passedDevices, devices.count, options.gpuClassCeiling)
            )
            usearch_shapes_init_vram(&arguments)
        }
        return try adopted(index, callState)
    }

    /// Copies a checkpoint into a new, mutable index in host memory. A cancelled or failed load
    /// creates no index.
    public static func loadRam(
        _ path: String,
        config: DenseLoadRamConfig = DenseLoadRamConfig()
    ) throws(USearchError) -> ShapesIndex {
        let callState = CallState()
        let index = callState.slots(usearch_shapes_index_t?.none)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try callState.pass(path) { (path) throws(USearchError) in
            try callState.pass(config.memoryDomains) { (domains) throws(USearchError) in
                var arguments = usearch_shapes_load_ram_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: callState.outcome,
                    completion: callState.completion,
                    path: path,
                    ram: ramOptions(domains, config.memoryDomains.count, config.cpuClassCeiling),
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_shapes_load_ram(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Copies a checkpoint into a new, mutable index in device memory.
    public static func loadVram(
        _ path: String,
        config: DenseLoadVramConfig = DenseLoadVramConfig()
    ) throws(USearchError) -> ShapesIndex {
        let callState = CallState()
        let index = callState.slots(usearch_shapes_index_t?.none)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try callState.pass(path) { (path) throws(USearchError) in
            try callState.pass(config.devices) { (devices) throws(USearchError) in
                var arguments = usearch_shapes_load_vram_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: callState.outcome,
                    completion: callState.completion,
                    path: path,
                    vram: vramOptions(devices, config.devices.count, config.gpuClassCeiling),
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_shapes_load_vram(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Copies a checkpoint from a buffer into a new index in host memory, as `loadRam` does a path.
    public static func loadRamBuffer(
        _ buffer: RawSpan,
        config: DenseLoadRamBufferConfig = DenseLoadRamBufferConfig()
    ) throws(USearchError) -> ShapesIndex {
        let callState = CallState()
        let index = callState.slots(usearch_shapes_index_t?.none)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try buffer.withUnsafeBytes { (bytes) throws(USearchError) in
            try callState.pass(config.memoryDomains) { (domains) throws(USearchError) in
                var arguments = usearch_shapes_load_ram_buffer_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: callState.outcome,
                    completion: callState.completion,
                    buffer: usearch_bytes_view_t(bytes),
                    ram: ramOptions(domains, config.memoryDomains.count, config.cpuClassCeiling),
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_shapes_load_ram_buffer(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Copies a checkpoint from a buffer into a new index in device memory, as `loadVram` does a
    /// path.
    public static func loadVramBuffer(
        _ buffer: RawSpan,
        config: DenseLoadVramBufferConfig = DenseLoadVramBufferConfig()
    ) throws(USearchError) -> ShapesIndex {
        let callState = CallState()
        let index = callState.slots(usearch_shapes_index_t?.none)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try buffer.withUnsafeBytes { (bytes) throws(USearchError) in
            try callState.pass(config.devices) { (devices) throws(USearchError) in
                var arguments = usearch_shapes_load_vram_buffer_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: callState.outcome,
                    completion: callState.completion,
                    buffer: usearch_bytes_view_t(bytes),
                    vram: vramOptions(devices, config.devices.count, config.gpuClassCeiling),
                    values_per_key: config.valuesPerKey,
                    progress: progress
                )
                usearch_shapes_load_vram_buffer(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Serves a checkpoint in place through a memory mapping, as a read-only index.
    public static func viewRam(
        _ path: String,
        memoryDomains: [Int32] = [],
        cpuClassCeiling: CPUClass? = nil
    ) throws(USearchError) -> ShapesIndex {
        let callState = CallState()
        let index = callState.slots(usearch_shapes_index_t?.none)
        try callState.pass(path) { (path) throws(USearchError) in
            try callState.pass(memoryDomains) { (domains) throws(USearchError) in
                var arguments = usearch_shapes_view_ram_t(
                    index: index,
                    outcome: callState.outcome,
                    path: path,
                    ram: ramOptions(domains, memoryDomains.count, cpuClassCeiling)
                )
                usearch_shapes_view_ram(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Serves a checkpoint in place from NVMe drives, read through the executor's submission rings,
    /// as a read-only index.
    public static func viewNvme(
        _ path: String,
        config: BasicViewNvmeConfig = BasicViewNvmeConfig()
    ) throws(USearchError) -> ShapesIndex {
        let callState = CallState()
        let index = callState.slots(usearch_shapes_index_t?.none)
        try callState.pass(path) { (path) throws(USearchError) in
            withPaths(config.paths, config.cpuClassCeiling) { nvme in
                var arguments = usearch_shapes_view_nvme_t(
                    index: index,
                    executor: config.executor?.raw,
                    outcome: callState.outcome,
                    completion: callState.completion,
                    path: path,
                    nvme: nvme
                )
                usearch_shapes_view_nvme(&arguments)
            }
        }
        return try adopted(index, callState)
    }

    /// Reads a checkpoint's metric and geometry without loading it.
    public static func metadataRead(
        _ path: String,
        executor: Executor? = nil
    ) throws(USearchError) -> ShapesMetadata {
        let callState = CallState()
        let metadata = callState.slots(ShapesMetadata())
        try callState.pass(path) { (path) throws(USearchError) in
            var arguments = usearch_shapes_metadata_read_t(
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: callState.completion,
                path: path,
                metadata: metadata.pointer(to: \.raw)!
            )
            usearch_shapes_metadata_read(&arguments)
        }
        try callState.wait()
        return metadata.pointee
    }

    /// Reads the metric and geometry a checkpoint in a buffer froze, without loading it.
    public static func metadataReadBuffer(
        _ buffer: RawSpan,
        executor: Executor? = nil
    ) throws(USearchError) -> ShapesMetadata {
        let callState = CallState()
        let metadata = callState.slots(ShapesMetadata())
        buffer.withUnsafeBytes { bytes in
            var arguments = usearch_shapes_metadata_read_buffer_t(
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: callState.completion,
                buffer: usearch_bytes_view_t(bytes),
                metadata: metadata.pointer(to: \.raw)!
            )
            usearch_shapes_metadata_read_buffer(&arguments)
        }
        try callState.wait()
        return metadata.pointee
    }

    /// Adopts the handle a blocking lifecycle verb wrote, or throws its failure.
    private static func adopted(
        _ index: UnsafeMutablePointer<usearch_shapes_index_t?>,
        _ callState: CallState
    ) throws(USearchError) -> ShapesIndex {
        try callState.wait()
        return ShapesIndex(raw: try created(index))
    }

    /// The same verbs, blocking on `executor`. Binding-only: names the executor every call passes.
    public func on(_ executor: Executor) -> ShapesIndexOn { ShapesIndexOn(index: self, executor: executor) }

    /// The same verbs, submitted to `executor` and answered at once as a `Deferred`. Binding-only:
    /// names the executor and the deferred completion every call passes.
    public func deferred(_ executor: Executor) -> ShapesIndexDeferredOn {
        ShapesIndexDeferredOn(index: self, executor: executor)
    }

    /// The verbs on the calling thread.
    private var here: ShapesIndexOn { ShapesIndexOn(index: self, executor: nil) }

    /// Reserves room for a total number of entries and points, the current ones included.
    public func reserve(entries: Int, points: Int = 0) throws(USearchError) {
        try here.reserve(entries: entries, points: points)
    }

    /// Erases all entries, keeping the reserved capacity and the frozen metric.
    public func clear() throws(USearchError) { try here.clear() }

    /// Reports the frozen metadata and the current counters in one call.
    public func describe() throws(USearchError) -> ShapesIndexDescription { try here.describe() }

    /// The entries stored under a key, with the extent of their points.
    public func count(key: Key) throws(USearchError) -> ShapesKeyStats { try here.count(key: key) }

    /// Retrieves every shape stored under a key into `arena`, converted to `Value` and in its
    /// inserted pose, in insertion order; an absent key yields none.
    public func get<Value: ScalarValue>(
        key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> TapeEntries<Value> {
        try here.get(key: key, as: value, arena: arena)
    }

    /// Inserts shapes under their keys, all or nothing, visible at once: each shape is its points
    /// `x`, `y` and `z` in turn, and point counts may differ between shapes.
    public func insert<Value: ScalarValue>(
        keys: [Key],
        shapes: [[Value]],
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) {
        try here.insert(keys: keys, shapes: shapes, config: config)
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, healing islands.
    @discardableResult
    public func refine(config: ShapesRefineConfig = ShapesRefineConfig()) throws(USearchError) -> DenseRefineReport {
        try here.refine(config: config)
    }

    /// Measures how much of the graph a search reaches from the root.
    public func health() throws(USearchError) -> DenseReachability { try here.health() }

    /// Resolves query shapes into up to `wanted` matches apiece, written in `arena`.
    public func search<Value: ScalarValue>(
        _ queries: [[Value]],
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Matches {
        try here.search(queries, wanted: wanted, arena: arena, config: config)
    }

    /// Moves the entries of `sources` into `targets`, keeping their links, then runs a round on
    /// every target that took entries: several sources gather into one, one source cuts into
    /// several. A refusal moves nothing.
    @discardableResult
    public func reshard(
        sources: [ShapesIndex],
        targets: [ShapesIndex],
        config: ShapesReshardConfig = ShapesReshardConfig()
    ) throws(USearchError) -> DenseRefineReport {
        try here.reshard(sources: sources, targets: targets, config: config)
    }

    /// Draws up to `wanted` keys around `config.focus`, or across the whole index, into `arena`.
    public func sample(
        wanted: Int,
        arena: Arena,
        config: DenseSampleConfig = DenseSampleConfig()
    ) throws(USearchError) -> DenseDraws {
        try here.sample(wanted: wanted, arena: arena, config: config)
    }

    /// Writes the index under a path: one monolith for a lone shard, or a manifest beside the
    /// parts.
    public func save(path: String, config: DenseSaveConfig = DenseSaveConfig()) throws(USearchError) {
        try here.save(path: path, config: config)
    }

    /// Writes the index into `arena`, the bytes `save(path:)` writes for a lone shard.
    public func saveBuffer(
        arena: Arena,
        config: DenseSaveBufferConfig = DenseSaveBufferConfig()
    ) throws(USearchError) -> Monolith {
        try here.saveBuffer(arena: arena, config: config)
    }

    /// Opens a snapshot walk over the current contents; with `starts`, from those keys
    /// breadth-first.
    public func cursor(starts: [Key] = []) throws(USearchError) -> ShapesCursor {
        try ShapesCall(index: self, executor: nil, callState: CallState()).cursor(starts: starts)
    }
}

// MARK: - Receivers

/// A shapes index's verbs on an executor, each blocking until final; see `ShapesIndex`.
public struct ShapesIndexOn: Sendable {
    /// The index every verb addresses.
    let index: ShapesIndex
    /// The executor every verb runs on, or nil for the calling thread.
    let executor: Executor?

    /// Runs one verb, blocking until it is final.
    private func run<Value>(
        _ verb: (ShapesCall) throws(USearchError) -> Submission<Value>
    ) throws(USearchError) -> Value {
        try verb(ShapesCall(index: index, executor: executor, callState: CallState())).wait()
    }

    /// Reserves room for a total number of entries and points.
    public func reserve(entries: Int, points: Int = 0) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.reserve(entries, points) }
    }

    /// Erases all entries.
    public func clear() throws(USearchError) { try run { (call) throws(USearchError) in try call.clear() } }

    /// Reports the frozen metadata and the current counters.
    public func describe() throws(USearchError) -> ShapesIndexDescription {
        try run { (call) throws(USearchError) in try call.describe() }
    }

    /// The entries stored under a key, with the extent of their points.
    public func count(key: Key) throws(USearchError) -> ShapesKeyStats {
        try run { (call) throws(USearchError) in try call.count(key) }
    }

    /// Retrieves every shape stored under a key into `arena`, converted to `Value`.
    public func get<Value: ScalarValue>(
        key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> TapeEntries<Value> {
        try run { (call) throws(USearchError) in try call.get(key, as: value, arena: arena) }
    }

    /// Inserts shapes under their keys, all or nothing.
    public func insert<Value: ScalarValue>(
        keys: [Key],
        shapes: [[Value]],
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.insert(keys, shapes, config: config) }
    }

    /// Re-selects the neighbors of every stored entry, healing islands.
    @discardableResult
    public func refine(config: ShapesRefineConfig = ShapesRefineConfig()) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.refine(config: config) }
    }

    /// Measures how much of the graph a search reaches from the root.
    public func health() throws(USearchError) -> DenseReachability {
        try run { (call) throws(USearchError) in try call.health() }
    }

    /// Resolves query shapes into up to `wanted` matches apiece.
    public func search<Value: ScalarValue>(
        _ queries: [[Value]],
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Matches {
        try run { (call) throws(USearchError) in try call.search(queries, wanted: wanted, arena: arena, config: config)
        }
    }

    /// Moves the entries of `sources` into `targets`, then runs a round on every target that took
    /// entries.
    @discardableResult
    public func reshard(
        sources: [ShapesIndex],
        targets: [ShapesIndex],
        config: ShapesReshardConfig = ShapesReshardConfig()
    ) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.reshard(sources: sources, targets: targets, config: config) }
    }

    /// Closes a live refine plan and frees it whatever the call answers.
    @discardableResult
    public func finish(_ plan: ShapesRefinePlan) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Closes a live reshard plan and frees it whatever the call answers.
    @discardableResult
    public func finish(_ plan: ShapesReshardPlan) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Draws up to `wanted` keys into `arena`.
    public func sample(
        wanted: Int,
        arena: Arena,
        config: DenseSampleConfig = DenseSampleConfig()
    ) throws(USearchError) -> DenseDraws {
        try run { (call) throws(USearchError) in try call.sample(wanted: wanted, arena: arena, config: config) }
    }

    /// Writes the index under a path.
    public func save(path: String, config: DenseSaveConfig = DenseSaveConfig()) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.save(path, config: config) }
    }

    /// Writes the index into `arena`.
    public func saveBuffer(
        arena: Arena,
        config: DenseSaveBufferConfig = DenseSaveBufferConfig()
    ) throws(USearchError) -> Monolith {
        try run { (call) throws(USearchError) in try call.saveBuffer(arena: arena, config: config) }
    }

}

/// A shapes index's verbs submitted to an executor, each answering a `Deferred` at once; see
/// `ShapesIndex` and `DenseIndexDeferredOn`, whose viewing rules these follow.
public struct ShapesIndexDeferredOn: Sendable {
    /// The index every verb addresses.
    let index: ShapesIndex
    /// The executor every verb is submitted to.
    let executor: Executor

    /// Submits one verb, whose call state its `Deferred` keeps until final.
    private func submit<Value>(
        _ verb: (ShapesCall) throws(USearchError) -> Submission<Value>
    ) throws(USearchError) -> Deferred<Value> {
        Deferred(
            try verb(
                ShapesCall(index: index, executor: executor, callState: CallState(usearch_completion_deferred_k))
            )
        )
    }

    /// Reserves room for a total number of entries and points.
    public func reserve(entries: Int, points: Int = 0) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.reserve(entries, points) }
    }

    /// Erases all entries.
    public func clear() throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.clear() }
    }

    /// Reports the frozen metadata and the current counters.
    public func describe() throws(USearchError) -> Deferred<ShapesIndexDescription> {
        try submit { (call) throws(USearchError) in try call.describe() }
    }

    /// The entries stored under a key, with the extent of their points.
    public func count(key: Key) throws(USearchError) -> Deferred<ShapesKeyStats> {
        try submit { (call) throws(USearchError) in try call.count(key) }
    }

    /// Retrieves every shape stored under a key into `arena`, converted to `Value`.
    public func get<Value: ScalarValue>(
        key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> Deferred<TapeEntries<Value>> {
        try submit { (call) throws(USearchError) in try call.get(key, as: value, arena: arena) }
    }

    /// Inserts shapes under their keys, all or nothing.
    public func insert<Value: ScalarValue>(
        keys: [Key],
        shapes: [[Value]],
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.insert(keys, shapes, config: config) }
    }

    /// Re-selects the neighbors of every stored entry, healing islands.
    public func refine(
        config: ShapesRefineConfig = ShapesRefineConfig()
    ) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in try call.refine(config: config) }
    }

    /// Measures how much of the graph a search reaches from the root.
    public func health() throws(USearchError) -> Deferred<DenseReachability> {
        try submit { (call) throws(USearchError) in try call.health() }
    }

    /// Resolves query shapes into up to `wanted` matches apiece.
    public func search<Value: ScalarValue>(
        _ queries: [[Value]],
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Deferred<Matches> {
        try submit { (call) throws(USearchError) in
            try call.search(queries, wanted: wanted, arena: arena, config: config)
        }
    }

    /// Moves the entries of `sources` into `targets`, then runs a round on every target that took
    /// entries.
    public func reshard(
        sources: [ShapesIndex],
        targets: [ShapesIndex],
        config: ShapesReshardConfig = ShapesReshardConfig()
    ) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in
            try call.reshard(sources: sources, targets: targets, config: config)
        }
    }

    /// Closes a live refine plan and frees it whatever the call answers.
    public func finish(_ plan: ShapesRefinePlan) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Closes a live reshard plan and frees it whatever the call answers.
    public func finish(_ plan: ShapesReshardPlan) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Draws up to `wanted` keys into `arena`.
    public func sample(
        wanted: Int,
        arena: Arena,
        config: DenseSampleConfig = DenseSampleConfig()
    ) throws(USearchError) -> Deferred<DenseDraws> {
        try submit { (call) throws(USearchError) in try call.sample(wanted: wanted, arena: arena, config: config) }
    }

    /// Writes the index under a path.
    public func save(path: String, config: DenseSaveConfig = DenseSaveConfig()) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.save(path, config: config) }
    }

    /// Writes the index into `arena`.
    public func saveBuffer(
        arena: Arena,
        config: DenseSaveBufferConfig = DenseSaveBufferConfig()
    ) throws(USearchError) -> Deferred<Monolith> {
        try submit { (call) throws(USearchError) in try call.saveBuffer(arena: arena, config: config) }
    }
}

// MARK: - Calls

/// The index, executor and call state one verb lowers into its argument structure, and the one
/// place each shapes verb is written.
struct ShapesCall {
    /// The index the call addresses.
    let index: ShapesIndex
    /// The executor the call runs on, or nil for the calling thread.
    let executor: Executor?
    /// Where the call's outcome and slots live until it is final.
    let callState: CallState

    /// Whether the call blocks or returns once submitted.
    private var completion: usearch_completion_t { callState.completion }

    func reserve(_ entries: Int, _ points: Int) throws(USearchError) -> Submission<Void> {
        var arguments = usearch_shapes_reserve_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            entries: entries,
            points: points
        )
        usearch_shapes_reserve(&arguments)
        return try callState.submitted()
    }

    func clear() throws(USearchError) -> Submission<Void> {
        var arguments = usearch_shapes_clear_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion
        )
        usearch_shapes_clear(&arguments)
        return try callState.submitted()
    }

    func describe() throws(USearchError) -> Submission<ShapesIndexDescription> {
        let description = callState.slots(ShapesIndexDescription())
        var arguments = usearch_shapes_describe_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            description: description.pointer(to: \.raw)!
        )
        usearch_shapes_describe(&arguments)
        return try callState.submitted(description)
    }

    func count(_ key: Key) throws(USearchError) -> Submission<ShapesKeyStats> {
        let stats = callState.slots(ShapesKeyStats())
        var arguments = usearch_shapes_count_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            key: key.raw,
            key_stats: stats.pointer(to: \.raw)!
        )
        usearch_shapes_count(&arguments)
        return try callState.submitted(stats)
    }

    func get<Value: ScalarValue>(
        _ key: Key,
        as value: Value.Type,
        arena: Arena
    ) throws(USearchError) -> Submission<TapeEntries<Value>> {
        let entries = callState.slots(TapeEntries<Value>(arena: arena, endIndex: 0, valuesPerElement: 3))
        var arguments = usearch_shapes_get_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            key: key.raw,
            shapes: entries.pointer(to: \.tape)!,
            written: entries.pointer(to: \.endIndex)!,
            arena: arena.raw,
            outputs_scalar: Value.scalarKind.raw
        )
        usearch_shapes_get(&arguments)
        return try callState.submitted(entries)
    }

    func insert<Value: ScalarValue>(
        _ keys: [Key],
        _ shapes: [[Value]],
        config: DenseInsertConfig
    ) throws(USearchError) -> Submission<Void> {
        // The key run holds no count of its own, so only this side can see it disagree.
        guard keys.count == shapes.count else {
            throw USearchError(status: .shapeMismatch, message: "the key count does not match the shape count")
        }
        callState.anchor(config.stats)
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try callState.pass(keys) { (keys) throws(USearchError) in
            try callState.pass(tape: shapes, valuesPerElement: 3) { (tape) throws(USearchError) in
                var arguments = usearch_shapes_insert_t(
                    index: index.raw,
                    executor: executor?.raw,
                    outcome: callState.outcome,
                    completion: completion,
                    count: shapes.count,
                    keys: keysView(keys),
                    shapes: tape,
                    expansion_factor: config.expansionFactor,
                    progress: progress,
                    stats: config.stats?.raw
                )
                usearch_shapes_insert(&arguments)
            }
        }
        return try callState.submitted()
    }

    func refine(config: ShapesRefineConfig) throws(USearchError) -> Submission<DenseRefineReport> {
        callState.anchor(config.stats)
        callState.anchor(config.plan)
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_shapes_refine_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            report: report.pointer(to: \.raw)!,
            refine: config.options,
            progress: callState.pass(config.progress, steps: config.progressSteps),
            stats: config.stats?.raw,
            plan: config.plan?.raw.pointer(to: \.handle),
            phase: config.plan?.raw.pointer(to: \.phase.raw)
        )
        usearch_shapes_refine(&arguments)
        return try callState.submitted(report)
    }

    func finish(_ plan: ShapesRefinePlan) throws(USearchError) -> Submission<DenseRefineReport> {
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_shapes_refine_plan_finish_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            plan: plan.raw.pointee.handle,
            report: report.pointer(to: \.raw)!
        )
        usearch_shapes_refine_plan_finish(&arguments)
        plan.raw.pointee.handle = nil
        return try callState.submitted(report)
    }

    func health() throws(USearchError) -> Submission<DenseReachability> {
        let reachability = callState.slots(DenseReachability())
        var arguments = usearch_shapes_health_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            reachability: reachability.pointer(to: \.raw)!
        )
        usearch_shapes_health(&arguments)
        return try callState.submitted(reachability)
    }

    func search<Value: ScalarValue>(
        _ queries: [[Value]],
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig
    ) throws(USearchError) -> Submission<Matches> {
        callState.anchor(config.stats)
        let matches = callState.slots(Matches(arena: arena, endIndex: queries.count))
        try callState.pass(tape: queries, valuesPerElement: 3) { (tape) throws(USearchError) in
            var arguments = usearch_shapes_search_t(
                index: index.raw,
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: completion,
                count: queries.count,
                queries: tape,
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
            usearch_shapes_search(&arguments)
        }
        return try callState.submitted(matches)
    }

    func reshard(
        sources: [ShapesIndex],
        targets: [ShapesIndex],
        config: ShapesReshardConfig
    ) throws(USearchError) -> Submission<DenseRefineReport> {
        for handle in sources + targets { callState.anchor(handle) }
        callState.anchor(config.stats)
        callState.anchor(config.plan)
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_shapes_reshard_t(
            sources_count: sources.count,
            sources: callState.copied(sources.map { Optional($0.raw) }),
            targets_count: targets.count,
            targets: callState.copied(targets.map { Optional($0.raw) }),
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            report: report.pointer(to: \.raw)!,
            refine: config.options,
            seed: config.seed,
            progress: callState.pass(config.progress, steps: config.progressSteps),
            stats: config.stats?.raw,
            plan: config.plan?.raw.pointer(to: \.handle),
            phase: config.plan?.raw.pointer(to: \.phase.raw)
        )
        usearch_shapes_reshard(&arguments)
        return try callState.submitted(report)
    }

    func finish(_ plan: ShapesReshardPlan) throws(USearchError) -> Submission<DenseRefineReport> {
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_shapes_reshard_plan_finish_t(
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            plan: plan.raw.pointee.handle,
            report: report.pointer(to: \.raw)!
        )
        usearch_shapes_reshard_plan_finish(&arguments)
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
        let draws = callState.slots(DenseDraws(arena: arena))
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try callState.pass(config.shown) { (shown) throws(USearchError) in
            try callState.pass(config.focus) { (focus) throws(USearchError) in
                try callState.pass(config.shownCoordinates) { (shownCoordinates) throws(USearchError) in
                    var arguments = usearch_shapes_sample_t(
                        index: index.raw,
                        executor: executor?.raw,
                        outcome: callState.outcome,
                        completion: completion,
                        shown_count: config.shown.count,
                        shown: keysView(shown),
                        shown_coordinates: shownPlaces(shownCoordinates),
                        focus_count: config.focus.count,
                        focus: keysView(focus),
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
                    usearch_shapes_sample(&arguments)
                }
            }
        }
        return try callState.submitted(draws)
    }

    func save(_ path: String, config: DenseSaveConfig) throws(USearchError) -> Submission<Void> {
        let progress = callState.pass(config.progress, steps: config.progressSteps)
        try callState.pass(path) { (path) throws(USearchError) in
            var arguments = usearch_shapes_save_t(
                index: index.raw,
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: completion,
                path: path,
                progress: progress
            )
            usearch_shapes_save(&arguments)
        }
        return try callState.submitted()
    }

    func saveBuffer(arena: Arena, config: DenseSaveBufferConfig) throws(USearchError) -> Submission<Monolith> {
        let monolith = callState.slots(Monolith(arena: arena))
        var arguments = usearch_shapes_save_buffer_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            monolith: monolith.pointer(to: \.monolith)!,
            arena: arena.raw,
            progress: callState.pass(config.progress, steps: config.progressSteps)
        )
        usearch_shapes_save_buffer(&arguments)
        return try callState.submitted(monolith)
    }

    func cursor(starts: [Key]) throws(USearchError) -> ShapesCursor {
        let cursor = callState.slots(usearch_shapes_cursor_t?.none)
        try callState.pass(starts) { (run) throws(USearchError) in
            var arguments = usearch_shapes_cursor_init_t(
                index: index.raw,
                executor: nil,
                outcome: callState.outcome,
                starts_count: starts.count,
                starts: UnsafeRawPointer(run)?.assumingMemoryBound(to: usearch_key_t.self),
                cursor: cursor
            )
            usearch_shapes_cursor_init(&arguments)
        }
        try callState.wait()
        return ShapesCursor(raw: try created(cursor), index: index)
    }
}

// MARK: - Cursors

/// A snapshot walk over a shapes index's contents; see `DenseCursor`, whose rules it follows.
public final class ShapesCursor {
    /// The native handle, released by `deinit` and never reassigned.
    private let raw: usearch_shapes_cursor_t
    /// The index the walk covers, held so the cursor cannot outlive it.
    private let index: ShapesIndex

    init(raw: usearch_shapes_cursor_t, index: ShapesIndex) {
        self.raw = raw
        self.index = index
    }

    deinit { usearch_shapes_cursor_free(raw) }

    /// Copies up to `wanted` next keys into `arena`; an empty batch means the walk is exhausted.
    public func next(wanted: Int, arena: Arena) throws(USearchError) -> DenseKeys {
        try pull(wanted, false, arena, nil).keys
    }

    /// Copies up to `wanted` next keys into `arena`, each with the keys it links to.
    public func nextLinks(wanted: Int, arena: Arena) throws(USearchError) -> (keys: DenseKeys, links: DenseLinks) {
        let (keys, links) = try pull(wanted, true, arena, nil)
        return (keys, DenseLinks(arena: arena, links: links, endIndex: keys.count))
    }

    /// The receiver whose pulls block on `executor`, named again for every pull. Binding-only.
    public func on(_ executor: Executor) -> ShapesCursorOn { ShapesCursorOn(cursor: self, executor: executor) }

    /// Advances the walk by one batch on `executor`, or the calling thread when nil, exporting each
    /// entry's links only when `linked` asks.
    fileprivate func pull(
        _ wanted: Int,
        _ linked: Bool,
        _ arena: Arena,
        _ executor: Executor?
    ) throws(USearchError) -> (keys: DenseKeys, links: usearch_keys_tape_view_t) {
        let callState = CallState()
        let keys = callState.slots(DenseKeys(arena: arena, endIndex: 0))
        let links = callState.slots(usearch_keys_tape_view_t())
        var arguments = usearch_shapes_cursor_next_t(
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
        usearch_shapes_cursor_next(&arguments)
        try callState.wait()
        return (keys.pointee, links.pointee)
    }
}

/// A cursor's pulls blocking on one executor, as `ShapesCursor.on(_:)` makes them. Binding-only.
public struct ShapesCursorOn {
    /// The cursor the pulls advance.
    let cursor: ShapesCursor
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
        return (keys, DenseLinks(arena: arena, links: links, endIndex: keys.count))
    }
}
