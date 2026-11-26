/**
 *  @file swift/ModalitiesTest.swift
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The sequence, shapes and geospatial suites: what the binding lowers and what it forwards.
 *
 *  Each case mirrors `test/modalities.py`: an index starts empty, finds what it holds, keeps it
 *  through a checkpoint, runs refine and reshard plans to their finish, and forwards the library's
 *  refusals. The cases shared by every modality are written once over `Modality`, which each index
 *  type conforms to below with the one verb call per step.
 */

import Foundation
import Testing

import USearch

/// One executor per test, so no test waits behind another's calls.
private func executor() throws(USearchError) -> Executor { try Executor.forRam() }

/// The status a call failed with, and nil when it succeeded.
private func failureStatus<Value>(of body: @autoclosure () throws(USearchError) -> Value) -> USearch.Status? {
    do {
        _ = try body()
        return nil
    }
    catch {
        return error.status
    }
}

/// A seeded draw, so every run builds the same entries.
private struct SplitMix: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var mixed = state
        mixed = (mixed ^ (mixed >> 30)) &* 0xBF58_476D_1CE4_E5B9
        mixed = (mixed ^ (mixed >> 27)) &* 0x94D0_49BB_1331_11EB
        return mixed ^ (mixed >> 31)
    }
}

/// Consecutive integer keys from `first`.
private func keys(_ count: Int, from first: Int = 0) -> [Key] { (first ..< first + count).map { Key(UInt64($0)) } }

/// A temporary checkpoint path, removed by the caller.
private func temporaryPath() -> String {
    FileManager.default.temporaryDirectory.appendingPathComponent("usearch-swift-\(UUID().uuidString).safegraphs").path
}

/// What the shared cases need of a modality: an empty index, one holding random entries, its size,
/// and each long operation through its own plan type.
private protocol Modality: AnyObject {
    associatedtype RefinePlan: AnyObject
    associatedtype ReshardPlan: AnyObject
    static func makeRefinePlan() -> RefinePlan
    static func makeReshardPlan() -> ReshardPlan
    static func empty() throws(USearchError) -> Self
    func fill(_ count: Int, from first: Int, seed: UInt64, on executor: Executor) throws(USearchError)
    func size() throws(USearchError) -> Int
    func holds(_ key: Key) throws(USearchError) -> Bool
    func refine(on executor: Executor?, stop: Bool, plan: RefinePlan?) throws(USearchError) -> DenseRefineReport
    func finish(_ plan: RefinePlan, on executor: Executor) throws(USearchError) -> DenseRefineReport
    func reshard(
        _ sources: [Self],
        _ targets: [Self],
        on executor: Executor,
        seed: UInt64,
        stop: Bool,
        plan: ReshardPlan?
    ) throws(USearchError) -> DenseRefineReport
    func finishDeferred(_ plan: ReshardPlan, on executor: Executor) throws(USearchError) -> DenseRefineReport
    static func phase(of plan: RefinePlan) -> PhaseProgress
    static func phase(of plan: ReshardPlan) -> PhaseProgress
}

extension Modality {
    /// An index holding `count` random entries under consecutive keys from `first`.
    static func filled(
        _ count: Int,
        from first: Int = 0,
        seed: UInt64 = 0,
        on executor: Executor
    ) throws(USearchError) -> Self {
        let index = try empty()
        try index.fill(count, from: first, seed: seed, on: executor)
        return index
    }
}

/// A handler that stops the operation it watches at the first barrier.
private let stopping: USearch.Progress = { _, _ in .cancel }

extension SequenceIndex: Modality {
    static func makeRefinePlan() -> SequenceRefinePlan { SequenceRefinePlan() }
    static func makeReshardPlan() -> SequenceReshardPlan { SequenceReshardPlan() }
    static func empty() throws(USearchError) -> SequenceIndex { try initRam() }

    func fill(_ count: Int, from first: Int, seed: UInt64, on executor: Executor) throws(USearchError) {
        var draw = SplitMix(state: seed)
        let sequences = (0 ..< count).map { _ in
            (0 ..< Int.random(in: 8 ... 16, using: &draw)).map { _ in UInt8.random(in: 97 ... 122, using: &draw) }
        }
        try on(executor).insert(keys: keys(count, from: first), sequences: sequences)
    }

    func size() throws(USearchError) -> Int { try describe().size }
    func holds(_ key: Key) throws(USearchError) -> Bool { try count(key: key).entries != 0 }

    func refine(on executor: Executor?, stop: Bool, plan: SequenceRefinePlan?) throws(USearchError) -> DenseRefineReport
    {
        let config = SequenceRefineConfig(progress: stop ? stopping : nil, plan: plan)
        guard let executor else { return try refine(config: config) }
        return try on(executor).refine(config: config)
    }

    func finish(_ plan: SequenceRefinePlan, on executor: Executor) throws(USearchError) -> DenseRefineReport {
        try on(executor).finish(plan)
    }

    func reshard(
        _ sources: [SequenceIndex],
        _ targets: [SequenceIndex],
        on executor: Executor,
        seed: UInt64,
        stop: Bool,
        plan: SequenceReshardPlan?
    ) throws(USearchError) -> DenseRefineReport {
        let config = SequenceReshardConfig(seed: seed, progress: stop ? stopping : nil, plan: plan)
        return try on(executor).reshard(sources: sources, targets: targets, config: config)
    }

    func finishDeferred(_ plan: SequenceReshardPlan, on executor: Executor) throws(USearchError) -> DenseRefineReport {
        try deferred(executor).finish(plan).wait()
    }

    static func phase(of plan: SequenceRefinePlan) -> PhaseProgress { plan.phase }
    static func phase(of plan: SequenceReshardPlan) -> PhaseProgress { plan.phase }
}

extension ShapesIndex: Modality {
    static func makeRefinePlan() -> ShapesRefinePlan { ShapesRefinePlan() }
    static func makeReshardPlan() -> ShapesReshardPlan { ShapesReshardPlan() }
    static func empty() throws(USearchError) -> ShapesIndex { try initRam() }

    func fill(_ count: Int, from first: Int, seed: UInt64, on executor: Executor) throws(USearchError) {
        var draw = SplitMix(state: seed)
        let shapes = (0 ..< count).map { _ in
            (0 ..< 3 * Int.random(in: 4 ... 12, using: &draw)).map { _ in Float.random(in: 0 ..< 1, using: &draw) }
        }
        try on(executor).insert(keys: keys(count, from: first), shapes: shapes)
    }

    func size() throws(USearchError) -> Int { try describe().size }
    func holds(_ key: Key) throws(USearchError) -> Bool { try count(key: key).entries != 0 }

    func refine(on executor: Executor?, stop: Bool, plan: ShapesRefinePlan?) throws(USearchError) -> DenseRefineReport {
        let config = ShapesRefineConfig(progress: stop ? stopping : nil, plan: plan)
        guard let executor else { return try refine(config: config) }
        return try on(executor).refine(config: config)
    }

    func finish(_ plan: ShapesRefinePlan, on executor: Executor) throws(USearchError) -> DenseRefineReport {
        try on(executor).finish(plan)
    }

    func reshard(
        _ sources: [ShapesIndex],
        _ targets: [ShapesIndex],
        on executor: Executor,
        seed: UInt64,
        stop: Bool,
        plan: ShapesReshardPlan?
    ) throws(USearchError) -> DenseRefineReport {
        let config = ShapesReshardConfig(seed: seed, progress: stop ? stopping : nil, plan: plan)
        return try on(executor).reshard(sources: sources, targets: targets, config: config)
    }

    func finishDeferred(_ plan: ShapesReshardPlan, on executor: Executor) throws(USearchError) -> DenseRefineReport {
        try deferred(executor).finish(plan).wait()
    }

    static func phase(of plan: ShapesRefinePlan) -> PhaseProgress { plan.phase }
    static func phase(of plan: ShapesReshardPlan) -> PhaseProgress { plan.phase }
}

extension GeospatialIndex: Modality {
    static func makeRefinePlan() -> GeospatialRefinePlan { GeospatialRefinePlan() }
    static func makeReshardPlan() -> GeospatialReshardPlan { GeospatialReshardPlan() }
    static func empty() throws(USearchError) -> GeospatialIndex { try initRam() }

    func fill(_ count: Int, from first: Int, seed: UInt64, on executor: Executor) throws(USearchError) {
        var draw = SplitMix(state: seed)
        let observations = (0 ..< count).map { _ in
            GeospatialObservation(
                latitudeDegrees: Double.random(in: -60 ... 60, using: &draw),
                longitudeDegrees: Double.random(in: -180 ... 180, using: &draw)
            )
        }
        try on(executor).insert(keys: keys(count, from: first), observations: observations)
    }

    func size() throws(USearchError) -> Int { try describe().size }
    func holds(_ key: Key) throws(USearchError) -> Bool { try count(key: key) != 0 }

    func refine(
        on executor: Executor?,
        stop: Bool,
        plan: GeospatialRefinePlan?
    ) throws(USearchError) -> DenseRefineReport {
        let config = GeospatialRefineConfig(progress: stop ? stopping : nil, plan: plan)
        guard let executor else { return try refine(config: config) }
        return try on(executor).refine(config: config)
    }

    func finish(_ plan: GeospatialRefinePlan, on executor: Executor) throws(USearchError) -> DenseRefineReport {
        try on(executor).finish(plan)
    }

    func reshard(
        _ sources: [GeospatialIndex],
        _ targets: [GeospatialIndex],
        on executor: Executor,
        seed: UInt64,
        stop: Bool,
        plan: GeospatialReshardPlan?
    ) throws(USearchError) -> DenseRefineReport {
        let config = GeospatialReshardConfig(seed: seed, progress: stop ? stopping : nil, plan: plan)
        return try on(executor).reshard(sources: sources, targets: targets, config: config)
    }

    func finishDeferred(_ plan: GeospatialReshardPlan, on executor: Executor) throws(USearchError) -> DenseRefineReport
    {
        try deferred(executor).finish(plan).wait()
    }

    static func phase(of plan: GeospatialRefinePlan) -> PhaseProgress { plan.phase }
    static func phase(of plan: GeospatialReshardPlan) -> PhaseProgress { plan.phase }
}

// MARK: - Cases Shared by Every Modality

/// A round reports what it did, its closing audit judging every entry, and leaves each entry
/// findable.
private func refineReportsARoundOverEveryEntry<Index: Modality>(_ modality: Index.Type) throws {
    let executor = try executor()
    let index = try Index.filled(64, on: executor)
    let report = try index.refine(on: executor, stop: false, plan: nil)
    #expect(report.changed <= report.reselected)
    #expect(report.reachability.audited == 64)
    #expect(try index.holds(17))
}

/// A plan keeps a stopped round live for the next call, and its finish closes it once, with the
/// whole plan's report.
private func refinePlanResumesAndFinishes<Index: Modality>(_ modality: Index.Type) throws {
    let executor = try executor()
    let index = try Index.filled(64, on: executor)
    let plan = Index.makeRefinePlan()
    let stopped = failureStatus(of: try index.refine(on: executor, stop: true, plan: plan))
    #expect(stopped == nil || stopped == .cancelled)
    #expect(Index.phase(of: plan).phase != .done)
    #expect(Index.phase(of: plan).completed <= Index.phase(of: plan).total)
    _ = try index.refine(on: executor, stop: false, plan: plan)
    #expect(Index.phase(of: plan).phase == .done)
    #expect(try index.finish(plan, on: executor).reachability.audited == 64)
    #expect(failureStatus(of: try index.finish(plan, on: executor)) == .invalidArgument)
}

/// A plan lives across calls on one executor, so a call on the calling thread cannot open one.
private func refinePlanNeedsAnExecutor<Index: Modality>(_ modality: Index.Type) throws {
    let index = try Index.empty()
    #expect(failureStatus(of: try index.refine(on: nil, stop: false, plan: Index.makeRefinePlan())) == .invalidArgument)
}

/// Several sources into one target gather every entry there, and the other sources end empty and
/// live.
private func reshardGathersEverySource<Index: Modality>(_ modality: Index.Type) throws {
    let executor = try executor()
    let first = try Index.filled(32, on: executor)
    let donors = try [1, 2].map { (seed) throws(USearchError) in
        try Index.filled(32, from: 32 * seed, seed: UInt64(seed), on: executor)
    }
    _ = try first.reshard([first] + donors, [first], on: executor, seed: 0, stop: false, plan: nil)
    #expect(try first.size() == 96)
    #expect(try first.holds(40) && first.holds(80))
    #expect(try donors.map { (donor) throws(USearchError) in try donor.size() } == [0, 0])
}

/// One source into several targets cuts it between them, and the source stays live but empty.
private func reshardCutsOneSourceIntoSeveralTargets<Index: Modality>(_ modality: Index.Type) throws {
    let executor = try executor()
    let source = try Index.filled(64, on: executor)
    let targets = [try Index.empty(), try Index.empty()]
    _ = try source.reshard([source], targets, on: executor, seed: 7, stop: false, plan: nil)
    let sizes = try targets.map { (target) throws(USearchError) in try target.size() }
    #expect(sizes.reduce(0, +) == 64)
    #expect(sizes.allSatisfy { $0 != 0 })
    #expect(try source.size() == 0)
}

/// A cut a stop interrupts keeps every entry moved, the next call naming the plan ends its rounds,
/// and a deferred receiver closes it.
private func reshardCutRunsThroughAPlan<Index: Modality>(_ modality: Index.Type) throws {
    let executor = try executor()
    let source = try Index.filled(64, on: executor)
    let targets = [try Index.empty(), try Index.empty()]
    let plan = Index.makeReshardPlan()
    let stopped = failureStatus(
        of: try source.reshard([source], targets, on: executor, seed: 0, stop: true, plan: plan)
    )
    #expect(stopped == nil || stopped == .cancelled)
    #expect(try targets.map { (target) throws(USearchError) in try target.size() }.reduce(0, +) == 64)
    #expect(Index.phase(of: plan).phase != .done)
    _ = try source.reshard([source], targets, on: executor, seed: 0, stop: false, plan: plan)
    #expect(Index.phase(of: plan).phase == .done)
    _ = try source.finishDeferred(plan, on: executor)
}

/// One index as its only source and target has a single shard to recut, which the library refuses.
private func reshardRecutOfOneHandleIsUnsupported<Index: Modality>(_ modality: Index.Type) throws {
    let executor = try executor()
    let index = try Index.filled(16, on: executor)
    #expect(
        failureStatus(of: try index.reshard([index], [index], on: executor, seed: 0, stop: false, plan: nil))
            == .unsupported
    )
    #expect(try index.size() == 16)
}

/// Empty or repeated lists are malformed and a target holding foreign entries mismatches; every
/// refusal leaves the entries where they were.
private func reshardRefusesMalformedListsAndMovesNothing<Index: Modality>(_ modality: Index.Type) throws {
    let executor = try executor()
    let first = try Index.filled(8, on: executor)
    let second = try Index.filled(8, from: 8, seed: 1, on: executor)
    func refusal(_ sources: [Index], _ targets: [Index]) -> USearch.Status? {
        failureStatus(of: try first.reshard(sources, targets, on: executor, seed: 0, stop: false, plan: nil))
    }
    #expect(refusal([], [first]) == .invalidArgument)
    #expect(refusal([first, first], [first]) == .invalidArgument)
    #expect(refusal([first], [second]) == .configMismatch)
    #expect(try [first.size(), second.size()] == [8, 8])
}

@Suite("Modality Plans")
struct ModalityPlanTests {
    @Test func sequenceRefine() throws {
        try refineReportsARoundOverEveryEntry(SequenceIndex.self)
        try refinePlanResumesAndFinishes(SequenceIndex.self)
        try refinePlanNeedsAnExecutor(SequenceIndex.self)
    }

    @Test func shapesRefine() throws {
        try refineReportsARoundOverEveryEntry(ShapesIndex.self)
        try refinePlanResumesAndFinishes(ShapesIndex.self)
        try refinePlanNeedsAnExecutor(ShapesIndex.self)
    }

    @Test func geospatialRefine() throws {
        try refineReportsARoundOverEveryEntry(GeospatialIndex.self)
        try refinePlanResumesAndFinishes(GeospatialIndex.self)
        try refinePlanNeedsAnExecutor(GeospatialIndex.self)
    }

    @Test func sequenceReshard() throws {
        try reshardGathersEverySource(SequenceIndex.self)
        try reshardCutsOneSourceIntoSeveralTargets(SequenceIndex.self)
        try reshardCutRunsThroughAPlan(SequenceIndex.self)
        try reshardRecutOfOneHandleIsUnsupported(SequenceIndex.self)
        try reshardRefusesMalformedListsAndMovesNothing(SequenceIndex.self)
    }

    @Test func shapesReshard() throws {
        try reshardGathersEverySource(ShapesIndex.self)
        try reshardCutsOneSourceIntoSeveralTargets(ShapesIndex.self)
        try reshardCutRunsThroughAPlan(ShapesIndex.self)
        try reshardRecutOfOneHandleIsUnsupported(ShapesIndex.self)
        try reshardRefusesMalformedListsAndMovesNothing(ShapesIndex.self)
    }

    @Test func geospatialReshard() throws {
        try reshardGathersEverySource(GeospatialIndex.self)
        try reshardCutsOneSourceIntoSeveralTargets(GeospatialIndex.self)
        try reshardCutRunsThroughAPlan(GeospatialIndex.self)
        try reshardRecutOfOneHandleIsUnsupported(GeospatialIndex.self)
        try reshardRefusesMalformedListsAndMovesNothing(GeospatialIndex.self)
    }
}

// MARK: - Cases of One Modality

@Suite("Sequence Indexes")
struct SequenceIndexTests {

    /// Mirrors the snippet under `## Sequence Indexes` in `swift/README.md`, line for line but for
    /// the `#expect` lines standing in for its `print`.
    @Test func readmeSnippet() throws {
        let index = try SequenceIndex.initRam(SequenceOptions(metric: .utf8Containment))
        let texts = ["The quick brown fox", "A lazy dog", "Pack my box"].map { Array($0.utf8) }
        try index.insert(keys: [1, 2, 3], sequences: texts)

        let matches = try index.search([Array("BROWN FOX".utf8)], wanted: 3, arena: try Arena())
        #expect(matches[0][0].key == 1)
        #expect(matches[0][0].distance < 1.0 / 32768)
    }

    /// A fresh index describes itself as empty, on a CPU class, with the metric it froze.
    @Test func initRamStartsEmpty() throws {
        let described = try SequenceIndex.initRam(SequenceOptions(metric: .dnaOverlap)).describe()
        #expect(described.size == 0)
        #expect(described.cpuClass != nil && described.gpuClass == nil)
        #expect(described.metadata.metric == .dnaOverlap)
    }

    /// A VRAM index either opens on a GPU or forwards the status the library answered.
    @Test func initVramForwardsTheLibrarysStatus() throws {
        let status = failureStatus(of: try SequenceIndex.initVram())
        #expect(status == nil || (status != .success && status != .deferred))
    }

    /// Every text searched for comes back first under its own key, at distance zero, and reads
    /// back as it was stored.
    @Test func searchFindsTheInsertedEntry() throws {
        let index = try SequenceIndex.initRam(SequenceOptions(metric: .utf8Containment))
        try index.insert(keys: [1, 2, 3], sequences: ["kitten", "sitting", "mitten"].map { Array($0.utf8) })
        #expect(try index.describe().size == 3)
        let arena = try Arena()
        let matches = try index.search([Array("sitting".utf8)], wanted: 1, arena: arena)
        #expect(matches[0][0].key == 2)
        #expect(matches[0][0].distance == 0)
        let stored = try index.get(key: 2, arena: arena)
        #expect(stored.count == 1 && Array(stored[0]) == Array("sitting".utf8))
        let stats = try index.count(key: 2)
        #expect(stats.entries == 1 && stats.totalBytes == 7 && stats.largestBytes == 7)
        #expect(try index.count(key: 4).entries == 0)
    }

    /// A checkpoint keeps every entry and the metric, and its metadata reads without loading it.
    @Test func checkpointRoundTrips() throws {
        let index = try SequenceIndex.initRam(SequenceOptions(metric: .utf8Containment))
        try index.insert(keys: [1, 2], sequences: [Array("kitten".utf8), Array("sitting".utf8)])
        let path = temporaryPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        try index.save(path: path)
        let metadata = try SequenceIndex.metadataRead(path)
        #expect(metadata.metric == .utf8Containment)
        let loaded = try SequenceIndex.loadRam(path)
        #expect(try loaded.describe().size == 2)
        #expect(try loaded.count(key: 2).entries == 1)
    }

    /// Similarities over `ACGT` alone: every base scores `match` against itself, `A` and `G` score
    /// `near` against each other, and every other pair scores zero.
    static func nucleotideSimilarities(match: Int8, near: Int8) -> [Int8] {
        var similarities = [Int8](repeating: 0, count: 256 * 256)
        for base in "ACGT".utf8 { similarities[Int(base) * 256 + Int(base)] = match }
        let adenine = Int(UInt8(ascii: "A")), guanine = Int(UInt8(ascii: "G"))
        similarities[adenine * 256 + guanine] = near
        similarities[guanine * 256 + adenine] = near
        return similarities
    }

    /// The library refuses the costs no metric reads, and the binding counts what has no count.
    @Test func costsReachTheLibrary() throws {
        let similarities = Self.nucleotideSimilarities(match: 2, near: 0)
        let costs = SequenceCosts(similarities: similarities, gap: 2)
        let containment = SequenceOptions(metric: .utf8Containment, costs: costs)
        #expect(failureStatus(of: try SequenceIndex.initRam(containment)) == .configMismatch)
        // The table crosses with no count, so the binding counts it.
        let short = SequenceCosts(similarities: [0], gap: 1)
        #expect(failureStatus(of: try SequenceIndex.initRam(SequenceOptions(costs: short))) == .shapeMismatch)
    }

    /// A query that is the reverse complement of a read finds that read first, at distance zero.
    @Test func overlapFindsAReverseComplement() throws {
        let index = try SequenceIndex.initRam(SequenceOptions(metric: .dnaOverlap))
        let reads = [
            "ACGTTGCAAGGCTTAACCGGTTACGATCGA", "GGCTTAACCGGTTACGATCGATTTGCAGCA", "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
        ]
        try index.insert(keys: [1, 2, 3], sequences: reads.map { Array($0.utf8) })
        let query = Array("TGCTGCAAATCGATCGTAACCGGTTAAGCC".utf8)
        let matches = try index.search([query], wanted: 2, arena: try Arena())
        #expect(matches[0][0].key == 2)
        #expect(matches[0][0].distance == 0)
    }

    /// Under costs placing `A` close to `G`, one `A` to `G` substitution ranks above one
    /// `A` to `C`.
    @Test func alignmentWeighsCosts() throws {
        let costs = SequenceCosts(similarities: Self.nucleotideSimilarities(match: 4, near: 3), gap: 4)
        let index = try SequenceIndex.initRam(SequenceOptions(costs: costs))
        try index.insert(keys: [1, 2], sequences: ["GCGTTGCAACGT", "CCGTTGCAACGT"].map { Array($0.utf8) })
        let matches = try index.search([Array("ACGTTGCAACGT".utf8)], wanted: 2, arena: try Arena())
        #expect(matches[0][0].key == 1)
        #expect(matches[0][0].distance < matches[0][1].distance)
    }

    /// Containment needs no costs: a query held by one text finds it first, whatever the case.
    @Test func containmentFindsTheHoldingText() throws {
        let index = try SequenceIndex.initRam(SequenceOptions(metric: .utf8Containment))
        let texts = [
            "the quick brown fox jumps over the lazy dog", "a journey of a thousand miles begins with a single step",
        ]
        try index.insert(keys: [1, 2], sequences: texts.map { Array($0.utf8) })
        let matches = try index.search([Array("BROWN FOX jumps".utf8)], wanted: 2, arena: try Arena())
        #expect(matches[0][0].key == 1)
    }

    /// Every metric spells itself through the library's own table.
    @Test func namesAreTheLibrarys() {
        #expect(SequenceMetric(name: SequenceMetric.utf8Containment.description) == .utf8Containment)
        #expect(SequenceMetric(name: "not-a-metric") == nil)
    }
}

@Suite("Shape Indexes")
struct ShapesIndexTests {

    /// Mirrors the snippet under `## Shape Indexes` in `swift/README.md`, line for line but for the
    /// `#expect` lines standing in for its `print`.
    @Test func readmeSnippet() throws {
        let index = try ShapesIndex.initRam(ShapesOptions(metric: .hausdorff))
        let water: [Float] = [0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0]
        let methane: [Float] = [0, 0, 0, 0.63, 0.63, 0.63, -0.63, -0.63, 0.63, -0.63, 0.63, -0.63, 0.63, -0.63, -0.63]
        try index.insert(keys: [1, 2], shapes: [water, methane])

        let matches = try index.search([methane], wanted: 1, arena: try Arena())
        #expect(matches[0][0].key == 2)
    }

    /// A fresh index describes itself as empty with the metric it froze.
    @Test func initRamStartsEmpty() throws {
        let described = try ShapesIndex.initRam(ShapesOptions(metric: .hausdorff)).describe()
        #expect(described.size == 0 && described.totalPoints == 0)
        #expect(described.cpuClass != nil && described.gpuClass == nil)
        #expect(described.metadata.metric == .hausdorff)
    }

    /// A VRAM index either opens on a GPU or forwards the status the library answered.
    @Test func initVramForwardsTheLibrarysStatus() throws {
        let status = failureStatus(of: try ShapesIndex.initVram())
        #expect(status == nil || (status != .success && status != .deferred))
    }

    /// Shapes of different point counts share one batch, each finds itself first, and each reads
    /// back with its point count.
    @Test func searchFindsTheInsertedEntry() throws {
        var draw = SplitMix(state: 42)
        let shapes = [5, 9, 13].map { points in (0 ..< 3 * points).map { _ in Float.random(in: 0 ..< 1, using: &draw) }
        }
        let index = try ShapesIndex.initRam()
        try index.insert(keys: [1, 2, 3], shapes: shapes)
        #expect(try index.describe().size == 3)
        #expect(try index.describe().totalPoints == 27)
        let arena = try Arena()
        #expect(try index.search([shapes[2]], wanted: 1, arena: arena)[0][0].key == 3)
        #expect(try index.get(key: 3, as: Float.self, arena: arena)[0].count == 39)
        let stats = try index.count(key: 2)
        #expect(stats.entries == 1 && stats.totalPoints == 9 && stats.largestPoints == 9)
        // A tape counts points, so only the binding sees a shape holding part of one.
        #expect(failureStatus(of: try index.insert(keys: [4], shapes: [[Float](arrayLiteral: 0, 1)])) == .shapeMismatch)
    }

    /// A checkpoint keeps every entry and the geometry, and its metadata reads without loading it.
    @Test func checkpointRoundTrips() throws {
        let index = try ShapesIndex.initRam(ShapesOptions(metric: .hausdorff))
        try index.insert(keys: [1], shapes: [[0, 0, 0, 1, 0, 0, 0, 1, 0] as [Float]])
        let path = temporaryPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        try index.save(path: path)
        let metadata = try ShapesIndex.metadataRead(path)
        #expect(metadata.metric == .hausdorff && metadata.quantization == .f32)
        #expect(try ShapesIndex.loadRam(path).count(key: 1).totalPoints == 3)
    }

    /// Every metric spells itself through the library's own table.
    @Test func namesAreTheLibrarys() {
        #expect(ShapesMetric(name: ShapesMetric.chamfer.description) == .chamfer)
        #expect(ShapesMetric(name: "not-a-metric") == nil)
        #expect(ShapesMetric.gromovWasserstein.description == "gromov-wasserstein")
        #expect(ShapesMetric(name: "wasserstein-procrustes") == .wassersteinProcrustes)
        #expect(ShapesMetric(name: "gromov_wasserstein") == nil)
    }
}

@Suite("Geospatial Indexes")
struct GeospatialIndexTests {

    /// Mirrors the snippet under `## Geospatial Indexes` in `swift/README.md`, line for line but
    /// for the `#expect` lines standing in for its `print`.
    @Test func readmeSnippet() throws {
        let index = try GeospatialIndex.initRam()
        let stops = [
            GeospatialObservation(latitudeDegrees: 51.5308, longitudeDegrees: -0.1238, accuracyMetres: 5),
            GeospatialObservation(latitudeDegrees: 51.5320, longitudeDegrees: -0.1233, accuracyMetres: 5),
            GeospatialObservation(latitudeDegrees: 51.5033, longitudeDegrees: -0.1196, accuracyMetres: 5),
        ]
        try index.insert(keys: [1, 2, 3], observations: stops)

        let station = GeospatialObservation(latitudeDegrees: 51.5308, longitudeDegrees: -0.1238)
        let nearby = try index.within([station], tolerance: GeospatialTolerance(metres: 500), arena: try Arena())
        #expect(Set(nearby[0].map(\.key)) == [1, 2])
    }

    /// A fresh index describes itself as empty, on a CPU class.
    @Test func initRamStartsEmpty() throws {
        let described = try GeospatialIndex.initRam().describe()
        #expect(described.size == 0)
        #expect(described.cpuClass != nil && described.gpuClass == nil)
    }

    /// A VRAM index either opens on a GPU or forwards the status the library answered.
    @Test func initVramForwardsTheLibrarysStatus() throws {
        let status = failureStatus(of: try GeospatialIndex.initVram())
        #expect(status == nil || (status != .success && status != .deferred))
    }

    /// An observation searched for comes back first at zero chord metres, and reads back in place.
    @Test func searchFindsTheInsertedEntry() throws {
        let places = [(48.8566, 2.3522), (52.52, 13.405), (40.7128, -74.006)].map {
            GeospatialObservation(latitudeDegrees: $0.0, longitudeDegrees: $0.1)
        }
        let index = try GeospatialIndex.initRam()
        try index.insert(keys: [1, 2, 3], observations: places)
        #expect(try index.describe().size == 3)
        let arena = try Arena()
        let matches = try index.search([places[1]], relation: .near, wanted: 1, arena: arena)
        #expect(matches[0][0].key == 2 && matches[0][0].distance == 0)
        let stored = try index.get(key: 2, arena: arena)
        #expect(stored.count == 1 && abs(stored[0].latitudeDegrees - 52.52) < 1e-6)
        #expect(try index.count(key: 2) == 1 && index.count(key: 4) == 0)
    }

    /// Two observations a kilometre apart fall inside a two-kilometre radius, and a far one does
    /// not.
    @Test func withinReportsOnlyTheEntriesInsideTheRadius() throws {
        let places = [(48.8566, 2.3522), (48.8656, 2.3522), (52.52, 13.405)].map {
            GeospatialObservation(latitudeDegrees: $0.0, longitudeDegrees: $0.1)
        }
        let index = try GeospatialIndex.initRam()
        try index.insert(keys: [1, 2, 3], observations: places)
        let report = try index.within([places[0]], tolerance: GeospatialTolerance(metres: 2000), arena: try Arena())
        #expect(Set(report[0].map(\.key)) == [1, 2])
    }

    /// A checkpoint keeps every observation.
    @Test func checkpointRoundTrips() throws {
        let index = try GeospatialIndex.initRam()
        try index.insert(keys: [1], observations: [GeospatialObservation(latitudeDegrees: 0, longitudeDegrees: 0)])
        let path = temporaryPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        try index.save(path: path)
        #expect(try GeospatialIndex.metadataRead(path).connectivity != 0)
        #expect(try GeospatialIndex.loadRam(path).count(key: 1) == 1)
    }
}
