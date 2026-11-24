/**
 *  @file swift/GeospatialIndex.swift
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Search over observations on and above the Earth - photos, video frames, drone footage
 *      and sensor fixes - by distance, line of sight and view cones.
 *
 *  Each entry is one observation of eleven doubles in natural units, NaN wherever nothing was
 *  recorded, and a missing field only ever widens the tests that read it. The verbs, receivers,
 *  plans and reports are the dense index's, and `within` reports every entry a relation admits.
 */

import USearchC

// MARK: - Observations

/// One observation as devices record it: every field in its natural unit, and NaN wherever nothing
/// was recorded. The index stores a copy rounded outward - centimetres, milliseconds and finer
/// turns of heading - and `get` reads it back within that rounding.
public struct GeospatialObservation: Equatable, Sendable {
    /// The library's own record, which batches reinterpret rather than copy.
    var raw: usearch_geospatial_observation_t

    /// Places an observation at a position, every field the caller leaves out unrecorded.
    public init(
        latitudeDegrees: Double,
        longitudeDegrees: Double,
        altitudeMetres: Double = .nan,
        accuracyMetres: Double = .nan,
        headingDegrees: Double = .nan,
        pitchDegrees: Double = .nan,
        headingAccuracyDegrees: Double = .nan,
        fieldOfViewDegrees: Double = .nan,
        rangeMetres: Double = .nan,
        beginSeconds: Double = .nan,
        endSeconds: Double = .nan
    ) {
        raw = usearch_geospatial_observation_t(
            latitude_degrees: latitudeDegrees,
            longitude_degrees: longitudeDegrees,
            altitude_metres: altitudeMetres,
            accuracy_metres: accuracyMetres,
            heading_degrees: headingDegrees,
            pitch_degrees: pitchDegrees,
            heading_accuracy_degrees: headingAccuracyDegrees,
            field_of_view_degrees: fieldOfViewDegrees,
            range_metres: rangeMetres,
            begin_seconds: beginSeconds,
            end_seconds: endSeconds
        )
    }

    init(raw: usearch_geospatial_observation_t) { self.raw = raw }

    /// Degrees north on WGS-84, south negative.
    public var latitudeDegrees: Double { raw.latitude_degrees }
    /// Degrees east on WGS-84, west negative.
    public var longitudeDegrees: Double { raw.longitude_degrees }
    /// Metres above the WGS-84 ellipsoid.
    public var altitudeMetres: Double { raw.altitude_metres }
    /// One standard deviation of the horizontal position, in metres.
    public var accuracyMetres: Double { raw.accuracy_metres }
    /// Where the device looked, in degrees clockwise from true north.
    public var headingDegrees: Double { raw.heading_degrees }
    /// Degrees above the horizon.
    public var pitchDegrees: Double { raw.pitch_degrees }
    /// One standard deviation of the heading, in degrees.
    public var headingAccuracyDegrees: Double { raw.heading_accuracy_degrees }
    /// The full field of view, in degrees.
    public var fieldOfViewDegrees: Double { raw.field_of_view_degrees }
    /// How far the device saw, in metres along the slant.
    public var rangeMetres: Double { raw.range_metres }
    /// The first instant the observation covers, in Unix seconds.
    public var beginSeconds: Double { raw.begin_seconds }
    /// The last instant the observation covers, in Unix seconds.
    public var endSeconds: Double { raw.end_seconds }

    /// Equal field by field, NaN equal to NaN, since an unrecorded field is one value.
    public static func == (left: GeospatialObservation, right: GeospatialObservation) -> Bool {
        withUnsafeBytes(of: left.raw) { left in withUnsafeBytes(of: right.raw) { right in left.elementsEqual(right) } }
    }
}

/// Views an observation run as the array of records every batch verb takes.
func observationsView(
    _ observations: UnsafePointer<GeospatialObservation>?
) -> UnsafePointer<usearch_geospatial_observation_t>? {
    UnsafeRawPointer(observations)?.assumingMemoryBound(to: usearch_geospatial_observation_t.self)
}

/// Which entries a query admits.
public struct GeospatialRelation: Hashable, Sendable {
    /// The engine's own value.
    let raw: usearch_geospatial_relation_t

    /// No relation named, refused by every query.
    public static let unknown = GeospatialRelation(raw: usearch_geospatial_relation_unknown_k)
    /// Within the radius, widened by three standard deviations of both positions combined.
    public static let near = GeospatialRelation(raw: usearch_geospatial_relation_near_k)
    /// Near, with a clear line of sight over the handle's terrain; a handle without one refuses it.
    public static let visible = GeospatialRelation(raw: usearch_geospatial_relation_visible_k)
    /// Near, and the entry's view cone holds the query: which cameras see this place.
    public static let sees = GeospatialRelation(raw: usearch_geospatial_relation_sees_k)
    /// Near, and the query's view cone holds the entry: what this camera sees.
    public static let seen = GeospatialRelation(raw: usearch_geospatial_relation_seen_k)

    init(raw: usearch_geospatial_relation_t) { self.raw = raw }
}

/// How far a query reaches in space and time.
public struct GeospatialTolerance: Sendable {
    /// The radius in metres; zero means none.
    public var metres: Double
    /// The slack added to both ends of the query's time interval, in seconds.
    public var seconds: Double

    public init(metres: Double = 0, seconds: Double = 0) {
        self.metres = metres
        self.seconds = seconds
    }

    /// The tolerance as the library reads it.
    var raw: usearch_geospatial_tolerance_t { usearch_geospatial_tolerance_t(metres: metres, seconds: seconds) }
}

/// The elevation every line of sight is tested over: a north-up grid of `f32` heights in metres
/// above the WGS-84 ellipsoid, or a dataset only an NVMe residency reads. A checkpoint never
/// carries it, so every load and view names it again.
public struct GeospatialTerrain: Sendable {
    /// The grid, rows running south and columns east, borrowed for the index's lifetime; an index
    /// built over a `Tensor` keeps it alive itself.
    public var heights: RowsView?
    /// The latitude of the centre of cell [0,0], in degrees.
    public var northLatitudeDegrees: Double
    /// The longitude of the centre of cell [0,0], in degrees.
    public var westLongitudeDegrees: Double
    /// The step between rows, in degrees, positive.
    public var rowStepDegrees: Double
    /// The step between columns, in degrees, positive.
    public var columnStepDegrees: Double
    /// Or a dataset on disk, which RAM and VRAM handles refuse.
    public var path: String?

    public init(
        heights: RowsView? = nil,
        northLatitudeDegrees: Double = 0,
        westLongitudeDegrees: Double = 0,
        rowStepDegrees: Double = 0,
        columnStepDegrees: Double = 0,
        path: String? = nil
    ) {
        self.heights = heights
        self.northLatitudeDegrees = northLatitudeDegrees
        self.westLongitudeDegrees = westLongitudeDegrees
        self.rowStepDegrees = rowStepDegrees
        self.columnStepDegrees = columnStepDegrees
        self.path = path
    }
}

extension CallState {
    /// Lends an optional terrain as the record every geospatial lifecycle verb takes; nil lends the
    /// zeroed one that means none.
    func lend<Value>(
        _ terrain: GeospatialTerrain?,
        _ body: (usearch_geospatial_terrain_t) throws(USearchError) -> Value
    ) throws(USearchError) -> Value {
        guard let terrain else { return try body(usearch_geospatial_terrain_t()) }
        // The record carries no scalar of its own, so only this side can see heights of another.
        if let heights = terrain.heights, heights.scalar != .f32 {
            throw USearchError(status: .invalidArgument, message: "terrain heights are f32")
        }
        return try lend(terrain.path ?? "") { (path) throws(USearchError) -> Value in
            var raw = usearch_geospatial_terrain_t()
            raw.rows = terrain.heights?.rows ?? 0
            raw.columns = terrain.heights?.columns ?? 0
            raw.north_latitude_degrees = terrain.northLatitudeDegrees
            raw.west_longitude_degrees = terrain.westLongitudeDegrees
            raw.row_step_degrees = terrain.rowStepDegrees
            raw.column_step_degrees = terrain.columnStepDegrees
            raw.heights = terrain.heights.map { $0.data.assumingMemoryBound(to: Float.self) }
            raw.heights_stride = terrain.heights?.stride ?? 0
            raw.path = terrain.path == nil ? usearch_bytes_view_t() : path
            return try body(raw)
        }
    }
}

// MARK: - Options

/// The construction options of a geospatial index, as `initRam` and `initVram` freeze them.
public struct GeospatialOptions: Sendable {
    /// The terrain lines of sight are tested over; nil means none.
    public var terrain: GeospatialTerrain?
    /// The bound on connections per node in the graph; zero derives to twelve.
    public var connectivity: Int
    /// The bound on observations sharing one key; zero is unbounded.
    public var valuesPerKey: Int
    /// The newest class dispatch may pick, intersected with what the machine runs; nil means no
    /// ceiling.
    public var cpuClassCeiling: CPUClass?
    /// The newest class dispatch may pick on the named devices; nil means no ceiling.
    public var gpuClassCeiling: GPUClass?

    /// An index without terrain, with every optional field derived.
    public init(
        terrain: GeospatialTerrain? = nil,
        connectivity: Int = 0,
        valuesPerKey: Int = 0,
        cpuClassCeiling: CPUClass? = nil,
        gpuClassCeiling: GPUClass? = nil
    ) {
        self.terrain = terrain
        self.connectivity = connectivity
        self.valuesPerKey = valuesPerKey
        self.cpuClassCeiling = cpuClassCeiling
        self.gpuClassCeiling = gpuClassCeiling
    }
}

/// What a checkpoint froze - everything a reader learns without opening it.
public struct GeospatialMetadata: Sendable {
    /// The library's own record.
    var raw = usearch_geospatial_metadata_t()

    /// The connections-per-node bound the graph was built with.
    public var connectivity: Int { raw.connectivity }
}

/// A live index's frozen metadata and current counters, each read once.
public struct GeospatialIndexDescription: Sendable {
    /// The library's own record.
    var raw = usearch_geospatial_index_description_t()

    /// The frozen connectivity, as a checkpoint reports it.
    public var metadata: GeospatialMetadata { GeospatialMetadata(raw: raw.metadata) }
    /// The stored entries.
    public var size: Int { raw.size }
    /// The entries hosted without growing again.
    public var capacity: Int { raw.capacity }
    /// The bytes the handle holds.
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

/// The knobs of `within`. Every zeroed field derives.
public struct GeospatialWithinConfig {
    /// Walks the graph by default; `.scan` tests every entry exactly.
    public var traversal: Traversal
    /// Narrows the entries to the keys it approves. It runs on the executor's threads together, so
    /// it must be safe for concurrent use.
    public var predicate: (@Sendable (Key) -> Bool)?
    /// Called with queries resolved.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?

    public init(
        traversal: Traversal = .walk,
        predicate: (@Sendable (Key) -> Bool)? = nil,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil
    ) {
        self.traversal = traversal
        self.predicate = predicate
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
    }
}

/// The observations `get` wrote into an arena, in insertion order, read in place until the arena's
/// next call.
public struct GeospatialObservations: RandomAccessCollection {
    /// The arena the observations live in, held so they outlive no release of it.
    let arena: Arena
    /// The observations the call wrote.
    var observations: UnsafePointer<usearch_geospatial_observation_t>?
    /// One past the last observation the call wrote.
    public internal(set) var endIndex: Int

    /// The first observation's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// The observation at a position.
    public subscript(position: Int) -> GeospatialObservation {
        boundsCheck(position, below: endIndex)
        return GeospatialObservation(raw: observations![position])
    }
}

/// The knobs of `refine` on a geospatial index.
public typealias GeospatialRefineConfig = BasicRefineConfig<GeospatialRefinePlan>

/// The knobs of `reshard` on geospatial indexes.
public typealias GeospatialReshardConfig = BasicReshardConfig<GeospatialReshardPlan>

/// One geospatial refine kept live across calls; see `DenseRefinePlan`, whose rules it follows.
public final class GeospatialRefinePlan: @unchecked Sendable {
    /// The handle the library opens, and the phase its last call stopped in, at addresses that hold
    /// for a deferred call.
    let raw: UnsafeMutablePointer<(handle: usearch_geospatial_refine_plan_t?, phase: PhaseProgress)>

    public init() {
        raw = .allocate(capacity: 1)
        raw.initialize(to: (nil, PhaseProgress()))
    }

    deinit { raw.deallocate() }

    /// Where the last call naming the plan stopped, `done` once its round ended.
    public var phase: PhaseProgress { raw.pointee.phase }
}

/// One geospatial reshard kept live across calls; see `DenseReshardPlan`, whose rules it follows.
public final class GeospatialReshardPlan: @unchecked Sendable {
    /// The handle the library opens, and the phase its last call stopped in, at addresses that hold
    /// for a deferred call.
    let raw: UnsafeMutablePointer<(handle: usearch_geospatial_reshard_plan_t?, phase: PhaseProgress)>

    public init() {
        raw = .allocate(capacity: 1)
        raw.initialize(to: (nil, PhaseProgress()))
    }

    deinit { raw.deallocate() }

    /// Where the last call naming the plan stopped, `done` once its rounds ended.
    public var phase: PhaseProgress { raw.pointee.phase }
}

// MARK: - Index

/// A Search Index over geospatial observations, in RAM or VRAM, or viewed from NVMe.
///
/// A handle may be shared across threads, so no method here takes a lock: insertions and searches
/// overlap freely, and saving demands a mutation-free window. Every verb called on the index runs
/// on the calling thread; `on(_:)` runs the same verbs on an executor, and `deferred(_:)` submits
/// them there.
public final class GeospatialIndex: @unchecked Sendable {
    /// The native handle, released by `deinit`.
    let raw: usearch_geospatial_index_t
    /// The storage of the terrain heights the handle borrows, held for its lifetime.
    private let terrainOwner: AnyObject?

    private init(raw: usearch_geospatial_index_t, terrainOwner: AnyObject?) {
        self.raw = raw
        self.terrainOwner = terrainOwner
    }

    /// Frees the index once every deferred call naming it is final.
    deinit { usearch_geospatial_free(raw) }

    /// Initializes an empty, mutable index in host memory, one shard per memory domain; an empty
    /// list derives every node the process may allocate on.
    public static func initRam(
        _ options: GeospatialOptions = GeospatialOptions(),
        memoryDomains: [Int32] = []
    ) throws(USearchError) -> GeospatialIndex {
        let callState = CallState()
        let index = callState.slots(usearch_geospatial_index_t?.none)
        try callState.lend(options.terrain) { (terrain) throws(USearchError) in
            try callState.lend(memoryDomains) { (domains) throws(USearchError) in
                var arguments = usearch_geospatial_init_ram_t(
                    index: index,
                    outcome: callState.outcome,
                    terrain: terrain,
                    connectivity: options.connectivity,
                    values_per_key: options.valuesPerKey,
                    ram: ramOptions(domains, memoryDomains.count, options.cpuClassCeiling)
                )
                usearch_geospatial_init_ram(&arguments)
            }
        }
        return try adopted(index, callState, options.terrain)
    }

    /// Initializes an empty, mutable index in device memory; an empty list derives one free device.
    public static func initVram(
        _ options: GeospatialOptions = GeospatialOptions(),
        devices: [Int32] = []
    ) throws(USearchError) -> GeospatialIndex {
        let callState = CallState()
        let index = callState.slots(usearch_geospatial_index_t?.none)
        try callState.lend(options.terrain) { (terrain) throws(USearchError) in
            try callState.lend(devices) { (lentDevices) throws(USearchError) in
                var arguments = usearch_geospatial_init_vram_t(
                    index: index,
                    outcome: callState.outcome,
                    terrain: terrain,
                    connectivity: options.connectivity,
                    values_per_key: options.valuesPerKey,
                    vram: vramOptions(lentDevices, devices.count, options.gpuClassCeiling)
                )
                usearch_geospatial_init_vram(&arguments)
            }
        }
        return try adopted(index, callState, options.terrain)
    }

    /// Copies a checkpoint into a new, mutable index in host memory, naming its terrain again. A
    /// cancelled or failed load creates no index.
    public static func loadRam(
        _ path: String,
        terrain: GeospatialTerrain? = nil,
        config: DenseLoadRamConfig = DenseLoadRamConfig()
    ) throws(USearchError) -> GeospatialIndex {
        let callState = CallState()
        let index = callState.slots(usearch_geospatial_index_t?.none)
        let progress = callState.lend(config.progress, steps: config.progressSteps)
        try callState.lend(path) { (path) throws(USearchError) in
            try callState.lend(terrain) { (lentTerrain) throws(USearchError) in
                try callState.lend(config.memoryDomains) { (domains) throws(USearchError) in
                    var arguments = usearch_geospatial_load_ram_t(
                        index: index,
                        executor: config.executor?.raw,
                        outcome: callState.outcome,
                        completion: callState.completion,
                        path: path,
                        ram: ramOptions(domains, config.memoryDomains.count, config.cpuClassCeiling),
                        values_per_key: config.valuesPerKey,
                        terrain: lentTerrain,
                        progress: progress
                    )
                    usearch_geospatial_load_ram(&arguments)
                }
            }
        }
        return try adopted(index, callState, terrain)
    }

    /// Copies a checkpoint into a new, mutable index in device memory, naming its terrain again.
    public static func loadVram(
        _ path: String,
        terrain: GeospatialTerrain? = nil,
        config: DenseLoadVramConfig = DenseLoadVramConfig()
    ) throws(USearchError) -> GeospatialIndex {
        let callState = CallState()
        let index = callState.slots(usearch_geospatial_index_t?.none)
        let progress = callState.lend(config.progress, steps: config.progressSteps)
        try callState.lend(path) { (path) throws(USearchError) in
            try callState.lend(terrain) { (lentTerrain) throws(USearchError) in
                try callState.lend(config.devices) { (devices) throws(USearchError) in
                    var arguments = usearch_geospatial_load_vram_t(
                        index: index,
                        executor: config.executor?.raw,
                        outcome: callState.outcome,
                        completion: callState.completion,
                        path: path,
                        vram: vramOptions(devices, config.devices.count, config.gpuClassCeiling),
                        values_per_key: config.valuesPerKey,
                        terrain: lentTerrain,
                        progress: progress
                    )
                    usearch_geospatial_load_vram(&arguments)
                }
            }
        }
        return try adopted(index, callState, terrain)
    }

    /// Copies a checkpoint from a buffer into a new index in host memory, as `loadRam` does a path.
    public static func loadRamBuffer(
        _ buffer: RawSpan,
        terrain: GeospatialTerrain? = nil,
        config: DenseLoadRamBufferConfig = DenseLoadRamBufferConfig()
    ) throws(USearchError) -> GeospatialIndex {
        let callState = CallState()
        let index = callState.slots(usearch_geospatial_index_t?.none)
        let progress = callState.lend(config.progress, steps: config.progressSteps)
        try buffer.withUnsafeBytes { (bytes) throws(USearchError) in
            try callState.lend(terrain) { (lentTerrain) throws(USearchError) in
                try callState.lend(config.memoryDomains) { (domains) throws(USearchError) in
                    var arguments = usearch_geospatial_load_ram_buffer_t(
                        index: index,
                        executor: config.executor?.raw,
                        outcome: callState.outcome,
                        completion: callState.completion,
                        buffer: usearch_bytes_view_t(bytes),
                        ram: ramOptions(domains, config.memoryDomains.count, config.cpuClassCeiling),
                        values_per_key: config.valuesPerKey,
                        terrain: lentTerrain,
                        progress: progress
                    )
                    usearch_geospatial_load_ram_buffer(&arguments)
                }
            }
        }
        return try adopted(index, callState, terrain)
    }

    /// Copies a checkpoint from a buffer into a new index in device memory, as `loadVram` does a
    /// path.
    public static func loadVramBuffer(
        _ buffer: RawSpan,
        terrain: GeospatialTerrain? = nil,
        config: DenseLoadVramBufferConfig = DenseLoadVramBufferConfig()
    ) throws(USearchError) -> GeospatialIndex {
        let callState = CallState()
        let index = callState.slots(usearch_geospatial_index_t?.none)
        let progress = callState.lend(config.progress, steps: config.progressSteps)
        try buffer.withUnsafeBytes { (bytes) throws(USearchError) in
            try callState.lend(terrain) { (lentTerrain) throws(USearchError) in
                try callState.lend(config.devices) { (devices) throws(USearchError) in
                    var arguments = usearch_geospatial_load_vram_buffer_t(
                        index: index,
                        executor: config.executor?.raw,
                        outcome: callState.outcome,
                        completion: callState.completion,
                        buffer: usearch_bytes_view_t(bytes),
                        vram: vramOptions(devices, config.devices.count, config.gpuClassCeiling),
                        values_per_key: config.valuesPerKey,
                        terrain: lentTerrain,
                        progress: progress
                    )
                    usearch_geospatial_load_vram_buffer(&arguments)
                }
            }
        }
        return try adopted(index, callState, terrain)
    }

    /// Serves a checkpoint in place through a memory mapping, as a read-only index.
    public static func viewRam(
        _ path: String,
        terrain: GeospatialTerrain? = nil,
        memoryDomains: [Int32] = [],
        cpuClassCeiling: CPUClass? = nil
    ) throws(USearchError) -> GeospatialIndex {
        let callState = CallState()
        let index = callState.slots(usearch_geospatial_index_t?.none)
        try callState.lend(path) { (path) throws(USearchError) in
            try callState.lend(terrain) { (lentTerrain) throws(USearchError) in
                try callState.lend(memoryDomains) { (domains) throws(USearchError) in
                    var arguments = usearch_geospatial_view_ram_t(
                        index: index,
                        outcome: callState.outcome,
                        path: path,
                        terrain: lentTerrain,
                        ram: ramOptions(domains, memoryDomains.count, cpuClassCeiling)
                    )
                    usearch_geospatial_view_ram(&arguments)
                }
            }
        }
        return try adopted(index, callState, terrain)
    }

    /// Serves a checkpoint in place from NVMe drives, read through the executor's submission rings,
    /// as a read-only index; only this residency reads a terrain `path`.
    public static func viewNvme(
        _ path: String,
        terrain: GeospatialTerrain? = nil,
        config: BasicViewNvmeConfig = BasicViewNvmeConfig()
    ) throws(USearchError) -> GeospatialIndex {
        let callState = CallState()
        let index = callState.slots(usearch_geospatial_index_t?.none)
        try callState.lend(path) { (path) throws(USearchError) in
            try callState.lend(terrain) { (lentTerrain) throws(USearchError) in
                withPaths(config.paths, config.cpuClassCeiling) { nvme in
                    var arguments = usearch_geospatial_view_nvme_t(
                        index: index,
                        executor: config.executor?.raw,
                        outcome: callState.outcome,
                        completion: callState.completion,
                        path: path,
                        nvme: nvme,
                        terrain: lentTerrain
                    )
                    usearch_geospatial_view_nvme(&arguments)
                }
            }
        }
        return try adopted(index, callState, terrain)
    }

    /// Reads a checkpoint's connectivity without loading it.
    public static func metadataRead(
        _ path: String,
        executor: Executor? = nil
    ) throws(USearchError) -> GeospatialMetadata {
        let callState = CallState()
        let metadata = callState.slots(GeospatialMetadata())
        try callState.lend(path) { (path) throws(USearchError) in
            var arguments = usearch_geospatial_metadata_read_t(
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: callState.completion,
                path: path,
                metadata: metadata.pointer(to: \.raw)!
            )
            usearch_geospatial_metadata_read(&arguments)
        }
        try callState.wait()
        return metadata.pointee
    }

    /// Reads the connectivity a checkpoint in a buffer froze, without loading it.
    public static func metadataReadBuffer(
        _ buffer: RawSpan,
        executor: Executor? = nil
    ) throws(USearchError) -> GeospatialMetadata {
        let callState = CallState()
        let metadata = callState.slots(GeospatialMetadata())
        buffer.withUnsafeBytes { bytes in
            var arguments = usearch_geospatial_metadata_read_buffer_t(
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: callState.completion,
                buffer: usearch_bytes_view_t(bytes),
                metadata: metadata.pointer(to: \.raw)!
            )
            usearch_geospatial_metadata_read_buffer(&arguments)
        }
        try callState.wait()
        return metadata.pointee
    }

    /// Adopts the handle a blocking lifecycle verb wrote, holding the terrain storage it borrows,
    /// or throws its failure.
    private static func adopted(
        _ index: UnsafeMutablePointer<usearch_geospatial_index_t?>,
        _ callState: CallState,
        _ terrain: GeospatialTerrain?
    ) throws(USearchError) -> GeospatialIndex {
        try callState.wait()
        return GeospatialIndex(raw: try created(index), terrainOwner: terrain?.heights?.owner)
    }

    /// The same verbs, blocking on `executor`. Binding-only: names the executor every call passes.
    public func on(_ executor: Executor) -> GeospatialIndexOn { GeospatialIndexOn(index: self, executor: executor) }

    /// The same verbs, submitted to `executor` and answered at once as a `Deferred`. Binding-only:
    /// names the executor and the deferred completion every call passes.
    public func deferred(_ executor: Executor) -> GeospatialIndexDeferredOn {
        GeospatialIndexDeferredOn(index: self, executor: executor)
    }

    /// The verbs on the calling thread.
    private var here: GeospatialIndexOn { GeospatialIndexOn(index: self, executor: nil) }

    /// Reserves room for a total number of entries, the current ones included.
    public func reserve(entries: Int) throws(USearchError) { try here.reserve(entries: entries) }

    /// Erases all entries, keeping the reserved capacity and the frozen metric.
    public func clear() throws(USearchError) { try here.clear() }

    /// Reports the frozen metadata and the current counters in one call.
    public func describe() throws(USearchError) -> GeospatialIndexDescription { try here.describe() }

    /// The observations stored under a key; zero means the index holds none.
    public func count(key: Key) throws(USearchError) -> Int { try here.count(key: key) }

    /// Retrieves every observation stored under a key into `arena`, in insertion order and within
    /// the stored rounding; an absent key yields none.
    public func get(key: Key, arena: Arena) throws(USearchError) -> GeospatialObservations {
        try here.get(key: key, arena: arena)
    }

    /// Inserts observations under their keys, all or nothing, visible at once.
    public func insert(
        keys: [Key],
        observations: [GeospatialObservation],
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) {
        try here.insert(keys: keys, observations: observations, config: config)
    }

    /// Re-selects the neighbors of every stored entry against the finished graph, healing islands.
    @discardableResult
    public func refine(config: GeospatialRefineConfig = GeospatialRefineConfig()) throws(USearchError)
        -> DenseRefineReport
    {
        try here.refine(config: config)
    }

    /// Measures how much of the graph a search reaches from the root.
    public func health() throws(USearchError) -> DenseReachability { try here.health() }

    /// Ranks the `wanted` nearest entries of each query `relation` admits within `tolerance`,
    /// written in `arena`, scored by the chord between the positions in metres; a zero radius
    /// ranks by distance alone.
    public func search(
        _ queries: [GeospatialObservation],
        relation: GeospatialRelation = .near,
        tolerance: GeospatialTolerance = GeospatialTolerance(),
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Matches {
        try here.search(
            queries,
            relation: relation,
            tolerance: tolerance,
            wanted: wanted,
            arena: arena,
            config: config
        )
    }

    /// Reports every entry of each query `relation` admits within `tolerance`, however many there
    /// turn out to be, as one run per query in `arena`, scored by the chord in metres.
    public func within(
        _ queries: [GeospatialObservation],
        relation: GeospatialRelation = .near,
        tolerance: GeospatialTolerance,
        arena: Arena,
        config: GeospatialWithinConfig = GeospatialWithinConfig()
    ) throws(USearchError) -> DenseProducts {
        try here.within(queries, relation: relation, tolerance: tolerance, arena: arena, config: config)
    }

    /// Moves the entries of `sources` into `targets`, keeping their links, then runs a round on
    /// every target that took entries: several sources gather into one, one source cuts into
    /// several. A refusal moves nothing.
    @discardableResult
    public func reshard(
        sources: [GeospatialIndex],
        targets: [GeospatialIndex],
        config: GeospatialReshardConfig = GeospatialReshardConfig()
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
    public func cursor(starts: [Key] = []) throws(USearchError) -> GeospatialCursor { try here.cursor(starts: starts) }
}

// MARK: - Receivers

/// A geospatial index's verbs on an executor, each blocking until final; see `GeospatialIndex`.
public struct GeospatialIndexOn: Sendable {
    /// The index every verb addresses.
    let index: GeospatialIndex
    /// The executor every verb runs on, or nil for the calling thread.
    let executor: Executor?

    /// Runs one verb, blocking until it is final.
    private func run<Value>(
        _ verb: (GeospatialCall) throws(USearchError) -> Submission<Value>
    ) throws(USearchError) -> Value {
        try verb(GeospatialCall(index: index, executor: executor, callState: CallState())).wait()
    }

    /// Reserves room for a total number of entries.
    public func reserve(entries: Int) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.reserve(entries) }
    }

    /// Erases all entries.
    public func clear() throws(USearchError) { try run { (call) throws(USearchError) in try call.clear() } }

    /// Reports the frozen metadata and the current counters.
    public func describe() throws(USearchError) -> GeospatialIndexDescription {
        try run { (call) throws(USearchError) in try call.describe() }
    }

    /// The observations stored under a key; zero means the index holds none.
    public func count(key: Key) throws(USearchError) -> Int {
        try run { (call) throws(USearchError) in try call.count(key) }
    }

    /// Retrieves every observation stored under a key into `arena`.
    public func get(key: Key, arena: Arena) throws(USearchError) -> GeospatialObservations {
        try run { (call) throws(USearchError) in try call.get(key, arena: arena) }
    }

    /// Inserts observations under their keys, all or nothing.
    public func insert(
        keys: [Key],
        observations: [GeospatialObservation],
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) {
        try run { (call) throws(USearchError) in try call.insert(keys, observations, config: config) }
    }

    /// Re-selects the neighbors of every stored entry, healing islands.
    @discardableResult
    public func refine(config: GeospatialRefineConfig = GeospatialRefineConfig()) throws(USearchError)
        -> DenseRefineReport
    {
        try run { (call) throws(USearchError) in try call.refine(config: config) }
    }

    /// Measures how much of the graph a search reaches from the root.
    public func health() throws(USearchError) -> DenseReachability {
        try run { (call) throws(USearchError) in try call.health() }
    }

    /// Ranks the `wanted` nearest entries each query admits.
    public func search(
        _ queries: [GeospatialObservation],
        relation: GeospatialRelation = .near,
        tolerance: GeospatialTolerance = GeospatialTolerance(),
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Matches {
        try run { (call) throws(USearchError) in
            try call.search(
                queries,
                relation: relation,
                tolerance: tolerance,
                wanted: wanted,
                arena: arena,
                config: config
            )
        }
    }

    /// Reports every entry of each query `relation` admits within `tolerance`, one run per query.
    public func within(
        _ queries: [GeospatialObservation],
        relation: GeospatialRelation = .near,
        tolerance: GeospatialTolerance,
        arena: Arena,
        config: GeospatialWithinConfig = GeospatialWithinConfig()
    ) throws(USearchError) -> DenseProducts {
        try run { (call) throws(USearchError) in
            try call.within(queries, relation: relation, tolerance: tolerance, arena: arena, config: config)
        }
    }

    /// Moves the entries of `sources` into `targets`, then runs a round on every target that took
    /// entries.
    @discardableResult
    public func reshard(
        sources: [GeospatialIndex],
        targets: [GeospatialIndex],
        config: GeospatialReshardConfig = GeospatialReshardConfig()
    ) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.reshard(sources: sources, targets: targets, config: config) }
    }

    /// Closes a live refine plan and frees it whatever the call answers.
    @discardableResult
    public func finish(_ plan: GeospatialRefinePlan) throws(USearchError) -> DenseRefineReport {
        try run { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Closes a live reshard plan and frees it whatever the call answers.
    @discardableResult
    public func finish(_ plan: GeospatialReshardPlan) throws(USearchError) -> DenseRefineReport {
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

    /// Opens a cursor whose every pull blocks on this executor.
    public func cursor(starts: [Key] = []) throws(USearchError) -> GeospatialCursor {
        try GeospatialCall(index: index, executor: executor, callState: CallState()).cursor(starts: starts)
    }
}

/// A geospatial index's verbs submitted to an executor, each answering a `Deferred` at once; see
/// `GeospatialIndex` and `DenseIndexDeferredOn`, whose borrowing rules these follow.
public struct GeospatialIndexDeferredOn: Sendable {
    /// The index every verb addresses.
    let index: GeospatialIndex
    /// The executor every verb is submitted to.
    let executor: Executor

    /// Submits one verb, whose call state its `Deferred` keeps until final.
    private func submit<Value>(
        _ verb: (GeospatialCall) throws(USearchError) -> Submission<Value>
    ) throws(USearchError) -> Deferred<Value> {
        Deferred(
            try verb(
                GeospatialCall(index: index, executor: executor, callState: CallState(usearch_completion_deferred_k))
            )
        )
    }

    /// Reserves room for a total number of entries.
    public func reserve(entries: Int) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.reserve(entries) }
    }

    /// Erases all entries.
    public func clear() throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.clear() }
    }

    /// Reports the frozen metadata and the current counters.
    public func describe() throws(USearchError) -> Deferred<GeospatialIndexDescription> {
        try submit { (call) throws(USearchError) in try call.describe() }
    }

    /// The observations stored under a key; zero means the index holds none.
    public func count(key: Key) throws(USearchError) -> Deferred<Int> {
        try submit { (call) throws(USearchError) in try call.count(key) }
    }

    /// Retrieves every observation stored under a key into `arena`.
    public func get(key: Key, arena: Arena) throws(USearchError) -> Deferred<GeospatialObservations> {
        try submit { (call) throws(USearchError) in try call.get(key, arena: arena) }
    }

    /// Inserts observations under their keys, all or nothing.
    public func insert(
        keys: [Key],
        observations: [GeospatialObservation],
        config: DenseInsertConfig = DenseInsertConfig()
    ) throws(USearchError) -> Deferred<Void> {
        try submit { (call) throws(USearchError) in try call.insert(keys, observations, config: config) }
    }

    /// Re-selects the neighbors of every stored entry, healing islands.
    public func refine(
        config: GeospatialRefineConfig = GeospatialRefineConfig()
    ) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in try call.refine(config: config) }
    }

    /// Measures how much of the graph a search reaches from the root.
    public func health() throws(USearchError) -> Deferred<DenseReachability> {
        try submit { (call) throws(USearchError) in try call.health() }
    }

    /// Ranks the `wanted` nearest entries each query admits.
    public func search(
        _ queries: [GeospatialObservation],
        relation: GeospatialRelation = .near,
        tolerance: GeospatialTolerance = GeospatialTolerance(),
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig = DenseSearchConfig()
    ) throws(USearchError) -> Deferred<Matches> {
        try submit { (call) throws(USearchError) in
            try call.search(
                queries,
                relation: relation,
                tolerance: tolerance,
                wanted: wanted,
                arena: arena,
                config: config
            )
        }
    }

    /// Reports every entry of each query `relation` admits within `tolerance`, one run per query.
    public func within(
        _ queries: [GeospatialObservation],
        relation: GeospatialRelation = .near,
        tolerance: GeospatialTolerance,
        arena: Arena,
        config: GeospatialWithinConfig = GeospatialWithinConfig()
    ) throws(USearchError) -> Deferred<DenseProducts> {
        try submit { (call) throws(USearchError) in
            try call.within(queries, relation: relation, tolerance: tolerance, arena: arena, config: config)
        }
    }

    /// Moves the entries of `sources` into `targets`, then runs a round on every target that took
    /// entries.
    public func reshard(
        sources: [GeospatialIndex],
        targets: [GeospatialIndex],
        config: GeospatialReshardConfig = GeospatialReshardConfig()
    ) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in
            try call.reshard(sources: sources, targets: targets, config: config)
        }
    }

    /// Closes a live refine plan and frees it whatever the call answers.
    public func finish(_ plan: GeospatialRefinePlan) throws(USearchError) -> Deferred<DenseRefineReport> {
        try submit { (call) throws(USearchError) in try call.finish(plan) }
    }

    /// Closes a live reshard plan and frees it whatever the call answers.
    public func finish(_ plan: GeospatialReshardPlan) throws(USearchError) -> Deferred<DenseRefineReport> {
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
/// place each geospatial verb is written.
struct GeospatialCall {
    /// The index the call addresses.
    let index: GeospatialIndex
    /// The executor the call runs on, or nil for the calling thread.
    let executor: Executor?
    /// Where the call's outcome and slots live until it is final.
    let callState: CallState

    /// Whether the call blocks or returns once submitted.
    private var completion: usearch_completion_t { callState.completion }

    func reserve(_ entries: Int) throws(USearchError) -> Submission<Void> {
        var arguments = usearch_geospatial_reserve_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            entries: entries
        )
        usearch_geospatial_reserve(&arguments)
        return try callState.submitted()
    }

    func clear() throws(USearchError) -> Submission<Void> {
        var arguments = usearch_geospatial_clear_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion
        )
        usearch_geospatial_clear(&arguments)
        return try callState.submitted()
    }

    func describe() throws(USearchError) -> Submission<GeospatialIndexDescription> {
        let description = callState.slots(GeospatialIndexDescription())
        var arguments = usearch_geospatial_describe_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            description: description.pointer(to: \.raw)!
        )
        usearch_geospatial_describe(&arguments)
        return try callState.submitted(description)
    }

    func count(_ key: Key) throws(USearchError) -> Submission<Int> {
        let stats = callState.slots(usearch_geospatial_key_stats_t())
        var arguments = usearch_geospatial_count_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            key: key.raw,
            key_stats: stats
        )
        usearch_geospatial_count(&arguments)
        return try callState.submitted(stats.pointer(to: \.entries)!)
    }

    func get(_ key: Key, arena: Arena) throws(USearchError) -> Submission<GeospatialObservations> {
        let entries = callState.slots(GeospatialObservations(arena: arena, endIndex: 0))
        var arguments = usearch_geospatial_get_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            key: key.raw,
            observations: entries.pointer(to: \.observations)!,
            written: entries.pointer(to: \.endIndex)!,
            arena: arena.raw
        )
        usearch_geospatial_get(&arguments)
        return try callState.submitted(entries)
    }

    func insert(
        _ keys: [Key],
        _ observations: [GeospatialObservation],
        config: DenseInsertConfig
    ) throws(USearchError) -> Submission<Void> {
        // The key run carries no count of its own, so only this side can see it disagree.
        guard keys.count == observations.count else {
            throw USearchError(status: .shapeMismatch, message: "the key count does not match the observation count")
        }
        callState.anchor(config.stats)
        let progress = callState.lend(config.progress, steps: config.progressSteps)
        try callState.lend(keys) { (keys) throws(USearchError) in
            try callState.lend(observations) { (lentObservations) throws(USearchError) in
                var arguments = usearch_geospatial_insert_t(
                    index: index.raw,
                    executor: executor?.raw,
                    outcome: callState.outcome,
                    completion: completion,
                    count: observations.count,
                    keys: keysView(keys),
                    observations: observationsView(lentObservations),
                    expansion_factor: config.expansionFactor,
                    progress: progress,
                    stats: config.stats?.raw
                )
                usearch_geospatial_insert(&arguments)
            }
        }
        return try callState.submitted()
    }

    func refine(config: GeospatialRefineConfig) throws(USearchError) -> Submission<DenseRefineReport> {
        callState.anchor(config.stats)
        callState.anchor(config.plan)
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_geospatial_refine_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            report: report.pointer(to: \.raw)!,
            refine: config.options,
            progress: callState.lend(config.progress, steps: config.progressSteps),
            stats: config.stats?.raw,
            plan: config.plan?.raw.pointer(to: \.handle),
            phase: config.plan?.raw.pointer(to: \.phase.raw)
        )
        usearch_geospatial_refine(&arguments)
        return try callState.submitted(report)
    }

    func finish(_ plan: GeospatialRefinePlan) throws(USearchError) -> Submission<DenseRefineReport> {
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_geospatial_refine_plan_finish_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            plan: plan.raw.pointee.handle,
            report: report.pointer(to: \.raw)!
        )
        usearch_geospatial_refine_plan_finish(&arguments)
        plan.raw.pointee.handle = nil
        return try callState.submitted(report)
    }

    func health() throws(USearchError) -> Submission<DenseReachability> {
        let reachability = callState.slots(DenseReachability())
        var arguments = usearch_geospatial_health_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            reachability: reachability.pointer(to: \.raw)!
        )
        usearch_geospatial_health(&arguments)
        return try callState.submitted(reachability)
    }

    func search(
        _ queries: [GeospatialObservation],
        relation: GeospatialRelation = .near,
        tolerance: GeospatialTolerance = GeospatialTolerance(),
        wanted: Int,
        arena: Arena,
        config: DenseSearchConfig
    ) throws(USearchError) -> Submission<Matches> {
        callState.anchor(config.stats)
        let matches = callState.slots(Matches(arena: arena, endIndex: queries.count))
        try callState.lend(queries) { (lentQueries) throws(USearchError) in
            var arguments = usearch_geospatial_search_t(
                index: index.raw,
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: completion,
                count: queries.count,
                queries: observationsView(lentQueries),
                relation: relation.raw,
                tolerance: tolerance.raw,
                matches: matches.pointer(to: \.view)!,
                arena: arena.raw,
                wanted: wanted,
                expansion_factor: config.expansionFactor,
                traversal: config.traversal.raw,
                predicate: callState.lend(config.predicate),
                progress: callState.lend(config.progress, steps: config.progressSteps),
                stats: config.stats?.raw
            )
            usearch_geospatial_search(&arguments)
        }
        return try callState.submitted(matches)
    }

    func within(
        _ queries: [GeospatialObservation],
        relation: GeospatialRelation,
        tolerance: GeospatialTolerance,
        arena: Arena,
        config: GeospatialWithinConfig
    ) throws(USearchError) -> Submission<DenseProducts> {
        callState.anchor(config.stats)
        let products = callState.slots(DenseProducts(arena: arena, endIndex: queries.count))
        try callState.lend(queries) { (lentQueries) throws(USearchError) in
            var arguments = usearch_geospatial_within_t(
                index: index.raw,
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: completion,
                count: queries.count,
                queries: observationsView(lentQueries),
                relation: relation.raw,
                tolerance: tolerance.raw,
                matches: products.pointer(to: \.products)!,
                written: nil,
                arena: arena.raw,
                traversal: config.traversal.raw,
                predicate: callState.lend(config.predicate),
                progress: callState.lend(config.progress, steps: config.progressSteps),
                stats: config.stats?.raw
            )
            usearch_geospatial_within(&arguments)
        }
        return try callState.submitted(products)
    }

    func reshard(
        sources: [GeospatialIndex],
        targets: [GeospatialIndex],
        config: GeospatialReshardConfig
    ) throws(USearchError) -> Submission<DenseRefineReport> {
        for handle in sources + targets { callState.anchor(handle) }
        callState.anchor(config.stats)
        callState.anchor(config.plan)
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_geospatial_reshard_t(
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
            progress: callState.lend(config.progress, steps: config.progressSteps),
            stats: config.stats?.raw,
            plan: config.plan?.raw.pointer(to: \.handle),
            phase: config.plan?.raw.pointer(to: \.phase.raw)
        )
        usearch_geospatial_reshard(&arguments)
        return try callState.submitted(report)
    }

    func finish(_ plan: GeospatialReshardPlan) throws(USearchError) -> Submission<DenseRefineReport> {
        let report = callState.slots(DenseRefineReport())
        var arguments = usearch_geospatial_reshard_plan_finish_t(
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            plan: plan.raw.pointee.handle,
            report: report.pointer(to: \.raw)!
        )
        usearch_geospatial_reshard_plan_finish(&arguments)
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
        let progress = callState.lend(config.progress, steps: config.progressSteps)
        try callState.lend(config.shown) { (shown) throws(USearchError) in
            try callState.lend(config.focus) { (focus) throws(USearchError) in
                try callState.lend(config.shownCoordinates) { (shownCoordinates) throws(USearchError) in
                    var arguments = usearch_geospatial_sample_t(
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
                    usearch_geospatial_sample(&arguments)
                }
            }
        }
        return try callState.submitted(draws)
    }

    func save(_ path: String, config: DenseSaveConfig) throws(USearchError) -> Submission<Void> {
        let progress = callState.lend(config.progress, steps: config.progressSteps)
        try callState.lend(path) { (path) throws(USearchError) in
            var arguments = usearch_geospatial_save_t(
                index: index.raw,
                executor: executor?.raw,
                outcome: callState.outcome,
                completion: completion,
                path: path,
                progress: progress
            )
            usearch_geospatial_save(&arguments)
        }
        return try callState.submitted()
    }

    func saveBuffer(arena: Arena, config: DenseSaveBufferConfig) throws(USearchError) -> Submission<Monolith> {
        let monolith = callState.slots(Monolith(arena: arena))
        var arguments = usearch_geospatial_save_buffer_t(
            index: index.raw,
            executor: executor?.raw,
            outcome: callState.outcome,
            completion: completion,
            monolith: monolith.pointer(to: \.monolith)!,
            arena: arena.raw,
            progress: callState.lend(config.progress, steps: config.progressSteps)
        )
        usearch_geospatial_save_buffer(&arguments)
        return try callState.submitted(monolith)
    }

    func cursor(starts: [Key]) throws(USearchError) -> GeospatialCursor {
        let cursor = callState.slots(usearch_geospatial_cursor_t?.none)
        try callState.pass(starts) { (run) throws(USearchError) in
            var arguments = usearch_geospatial_cursor_init_t(
                index: index.raw,
                executor: nil,
                outcome: callState.outcome,
                starts_count: starts.count,
                starts: UnsafeRawPointer(run)?.assumingMemoryBound(to: usearch_key_t.self),
                cursor: cursor
            )
            usearch_geospatial_cursor_init(&arguments)
        }
        try callState.wait()
        return GeospatialCursor(raw: try created(cursor), index: index, executor: executor)
    }
}

// MARK: - Cursors

/// A snapshot walk over a geospatial index's contents; see `DenseCursor`, whose rules it follows.
public final class GeospatialCursor {
    /// The native handle, released by `deinit` and never reassigned.
    private let raw: usearch_geospatial_cursor_t
    /// The index the walk covers, held so the cursor cannot outlive it.
    private let index: GeospatialIndex
    /// The executor every pull runs on, or nil for the calling thread.
    private let executor: Executor?

    init(raw: usearch_geospatial_cursor_t, index: GeospatialIndex, executor: Executor?) {
        self.raw = raw
        self.index = index
        self.executor = executor
    }

    deinit { usearch_geospatial_cursor_free(raw) }

    /// Copies up to `wanted` next keys into `arena`; an empty batch means the walk is exhausted.
    public func next(wanted: Int, arena: Arena) throws(USearchError) -> DenseKeys {
        try pull(wanted, false, arena).keys
    }

    /// Copies up to `wanted` next keys into `arena`, each with the keys it links to.
    public func nextLinks(wanted: Int, arena: Arena) throws(USearchError) -> (keys: DenseKeys, links: DenseLinks) {
        let (keys, links) = try pull(wanted, true, arena)
        return (keys, DenseLinks(arena: arena, links: links, endIndex: keys.count))
    }

    /// Advances the walk by one batch, exporting each entry's links only when `linked` asks.
    private func pull(
        _ wanted: Int,
        _ linked: Bool,
        _ arena: Arena
    ) throws(USearchError) -> (keys: DenseKeys, links: usearch_keys_tape_view_t) {
        let callState = CallState()
        let keys = callState.slots(DenseKeys(arena: arena, endIndex: 0))
        let links = callState.slots(usearch_keys_tape_view_t())
        var arguments = usearch_geospatial_cursor_next_t(
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
        usearch_geospatial_cursor_next(&arguments)
        try callState.wait()
        return (keys.pointee, links.pointee)
    }
}
