/**
 *  @file swift/BasicIndexes.swift
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief What the sequence, shapes and geospatial indexes share: tapes, plan configs and the
 *      entries `get` writes.
 *
 *  Variable-length entries cannot ride a run of equal rows, so the sequence and shapes verbs take
 *  and answer Apache Arrow tapes: the elements of every entry end to end, and one more offset than
 *  entries around them. Everything else these indexes report - matches, draws, keys, links, refine
 *  reports and monoliths - is the dense index's vocabulary, reused as it stands.
 */

import USearchC

// MARK: - Tapes

extension CallState {
    /// Passes `entries` as one tape, each entry's values end to end and the 64-bit offsets counting
    /// them in units of `valuesPerElement` values, so a shape's offsets count points rather than
    /// coordinates.
    func pass<Element: ScalarValue, Value>(
        tape entries: [[Element]],
        valuesPerElement: Int,
        _ body: (usearch_tape_view_t) throws(USearchError) -> Value
    ) throws(USearchError) -> Value {
        // The tape holds elements, so only this side can see an entry holding part of one.
        guard entries.allSatisfy({ $0.count % valuesPerElement == 0 }) else {
            throw USearchError(status: .shapeMismatch, message: "an entry holds a partial element")
        }
        var offsets: [UInt64] = [0]
        offsets.reserveCapacity(entries.count + 1)
        for entry in entries { offsets.append(offsets.last! + UInt64(entry.count / valuesPerElement)) }
        return try pass(entries.flatMap { $0 }) { (values) throws(USearchError) -> Value in
            try pass(offsets) { (offsets) throws(USearchError) -> Value in
                var tape = usearch_tape_view_t()
                tape.data = UnsafeRawPointer(values)
                tape.layout = usearch_tape_u64_k
                tape.offsets.u64 = offsets
                tape.scalar = Element.scalarKind.raw
                return try body(tape)
            }
        }
    }
}

/// The entries `get` wrote into an arena as one tape, read in place as `Value` until the arena's
/// next call: a sequence's bytes, or a shape's coordinates, `x`, `y` and `z` of each point in turn.
public struct TapeEntries<Value: ScalarValue>: RandomAccessCollection {
    /// The arena the tape lives in, held so it outlives no release of it.
    let arena: ArenaWrite
    /// The tape the call wrote.
    var tape = usearch_tape_view_t()
    /// One past the last entry the call wrote.
    public internal(set) var endIndex: Int
    /// The values one tape element spans: one byte of a sequence, three coordinates of a point.
    let valuesPerElement: Int

    /// The first entry's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// One entry's values, as many as its elements span.
    public subscript(entry: Int) -> DenseVectors<Value>.Row {
        arena.read(entry, below: endIndex) {
            let (first, last) = (offset(entry), offset(entry + 1))
            return DenseVectors<Value>.Row(
                arena: arena,
                base: tape.data! + first * valuesPerElement * MemoryLayout<Value>.stride,
                endIndex: (last - first) * valuesPerElement
            )
        }
    }

    /// The element an entry starts at, in whichever width the library wrote the offsets.
    private func offset(_ entry: Int) -> Int {
        tape.layout == usearch_tape_u32_k ? Int(tape.offsets.u32[entry]) : Int(tape.offsets.u64[entry])
    }
}

// MARK: - Configs

/// The knobs of `refine` on a sequence, shapes or geospatial index. Every zeroed field derives.
public struct BasicRefineConfig<Plan: AnyObject> {
    /// The breadth of the candidates the re-selection walk explores; zero derives it from the
    /// geometry.
    public var expansionFactor: Int
    /// Bytes a loaded NVMe handle holds for the one part it rewrites; zero derives, and RAM and
    /// VRAM handles never read it.
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
    public var plan: Plan?

    public init(
        expansionFactor: Int = 0,
        memoryLimit: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil,
        plan: Plan? = nil
    ) {
        self.expansionFactor = expansionFactor
        self.memoryLimit = memoryLimit
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
        self.plan = plan
    }

    /// The spending of the round, as the library reads it.
    var options: usearch_refine_options_t {
        usearch_refine_options_t(expansion_factor: expansionFactor, memory_limit: memoryLimit)
    }
}

/// The knobs of `reshard` on a sequence, shapes or geospatial index, whose handles keep the
/// placements they were made with.
public struct BasicReshardConfig<Plan: AnyObject> {
    /// Picks the part roots of a source cut among several targets: equal seeds over an unchanged
    /// source cut equally.
    public var seed: UInt64
    /// The breadth of the candidates of the closing rounds; zero derives.
    public var expansionFactor: Int
    /// Bytes a loaded NVMe target holds for the one part it rewrites; zero derives.
    public var memoryLimit: Int
    /// Called with the closing rounds' units at their barriers; a stop never undoes a move.
    public var progress: Progress?
    /// Reports the call makes at most, the last at completion; zero derives to ten.
    public var progressSteps: Int
    /// The traversal telemetry to write.
    public var stats: DenseTraversalStats?
    /// Keeps the reshard live across calls: nil moves and runs to the end, a fresh plan opens one
    /// that a stop leaves live, and a live one continues where it stopped.
    public var plan: Plan?

    public init(
        seed: UInt64 = 0,
        expansionFactor: Int = 0,
        memoryLimit: Int = 0,
        progress: Progress? = nil,
        progressSteps: Int = 0,
        stats: DenseTraversalStats? = nil,
        plan: Plan? = nil
    ) {
        self.seed = seed
        self.expansionFactor = expansionFactor
        self.memoryLimit = memoryLimit
        self.progress = progress
        self.progressSteps = progressSteps
        self.stats = stats
        self.plan = plan
    }

    /// The spending of the closing rounds, as the library reads it.
    var options: usearch_refine_options_t {
        usearch_refine_options_t(expansion_factor: expansionFactor, memory_limit: memoryLimit)
    }
}

/// The knobs of `viewNvme` on a sequence, shapes or geospatial index.
public struct BasicViewNvmeConfig {
    /// The NVMe executor whose rings read the checkpoint's headers; nil reads on the calling
    /// thread.
    public var executor: Executor?
    /// The devices the parts are served from; empty serves each part where its manifest points.
    public var paths: [String]
    /// The newest class dispatch may pick, intersected with what the machine runs, so a class the
    /// machine lacks narrows rather than refuses; nil means no ceiling.
    public var cpuClassCeiling: CPUClass?

    public init(executor: Executor? = nil, paths: [String] = [], cpuClassCeiling: CPUClass? = nil) {
        self.executor = executor
        self.paths = paths
        self.cpuClassCeiling = cpuClassCeiling
    }
}

// MARK: - Placements

/// The host placement every RAM lifecycle verb takes, viewing `domains` for the call.
func ramOptions(
    _ domains: UnsafePointer<Int32>?,
    _ count: Int,
    _ cpuClassCeiling: CPUClass?
) -> usearch_ram_options_t {
    usearch_ram_options_t(
        memory_domains: domains,
        memory_domains_count: count,
        shard_capacity: 0,
        cpu_class_ceiling: cpuClassCeiling?.raw ?? usearch_cpu_class_unknown_k,
        memory_limit: 0
    )
}

/// The device placement every VRAM lifecycle verb takes, viewing `devices` for the call.
func vramOptions(
    _ devices: UnsafePointer<Int32>?,
    _ count: Int,
    _ gpuClassCeiling: GPUClass?
) -> usearch_vram_options_t {
    usearch_vram_options_t(
        devices: devices,
        devices_count: count,
        shard_capacity: 0,
        gpu_class_ceiling: gpuClassCeiling?.raw ?? usearch_gpu_class_unknown_k
    )
}

/// Views a key run as the strided keys every batch verb takes.
func keysView(_ keys: UnsafePointer<Key>?) -> usearch_keys_view_t {
    usearch_keys_view_t(
        data: UnsafeRawPointer(keys)?.assumingMemoryBound(to: usearch_key_t.self),
        stride: MemoryLayout<usearch_key_t>.stride
    )
}

/// The three-column `f32` places of the shown keys a `sample` lays its draws against.
func shownPlaces(_ coordinates: UnsafePointer<Float>?) -> usearch_rows_view_t {
    usearch_rows_view_t(
        data: UnsafeRawPointer(coordinates),
        stride: 3 * MemoryLayout<Float>.stride,
        columns: 3,
        scalar: usearch_scalar_f32_k
    )
}
