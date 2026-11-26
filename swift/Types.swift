/**
 *  @file swift/Types.swift
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Shared vocabulary of the USearch Swift SDK.
 *
 *  Statuses, keys, scalar kinds, rows, executors, arenas, and deferred calls are
 *  modality-agnostic, so they live here and the dense modality builds on them in DenseIndex.swift.
 *
 *  Batches travel in a `RowsView` over NumKong's `MatrixView` or `Tensor`, and its low-precision
 *  types are the scalars a caller builds rows out of, and this file only tags them by `Scalar`.
 */

import NumKong

import USearchC

// MARK: - Errors

/// What went wrong, in the vocabulary the engine and this binding share - one constant per
/// engine status. Validation performed here before a call crosses reports the status the engine
/// would have reported for the same fault.
///
/// A `USearchError` never holds `success` or `deferred`; a `Deferred` call reports both.
public struct Status: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_status_t

    /// The call succeeded and holds no message.
    public static let success = Status(raw: usearch_status_success_k)
    /// The call was submitted deferred and is not final yet.
    public static let deferred = Status(raw: usearch_status_deferred_k)
    /// An allocation or reservation could not be satisfied.
    public static let outOfMemory = Status(raw: usearch_status_out_of_memory_k)
    /// Dimensionality or row counts disagree.
    public static let shapeMismatch = Status(raw: usearch_status_shape_mismatch_k)
    /// A stride or packing the engine cannot traverse.
    public static let unsupportedLayout = Status(raw: usearch_status_unsupported_layout_k)
    /// A scalar, metric, or residency tag this build does not know.
    public static let unknownKind = Status(raw: usearch_status_unknown_kind_k)
    /// A mutating verb reached a handle that serves its checkpoint in place.
    public static let readOnly = Status(raw: usearch_status_read_only_k)
    /// A file could not be opened, read, written, or parsed.
    public static let io = Status(raw: usearch_status_io_k)
    /// A device could not be selected, or its memory could not be claimed.
    public static let device = Status(raw: usearch_status_device_k)
    /// A caller-supplied representation could not be decoded.
    public static let malformed = Status(raw: usearch_status_malformed_k)
    /// The library reported neither success nor a failure.
    public static let `internal` = Status(raw: usearch_status_internal_k)
    /// A hard design ceiling was reached, like the 32-bit ID space.
    public static let overflow = Status(raw: usearch_status_overflow_k)
    /// The verb needs a prior successful reservation that never happened.
    public static let uninitialized = Status(raw: usearch_status_uninitialized_k)
    /// The request contradicts an already-frozen configuration.
    public static let configMismatch = Status(raw: usearch_status_config_mismatch_k)
    /// A call was refused rather than waited for - an executor another call holds, a busy arena, or
    /// a blocking call from inside a callback of the executor it would wait on.
    public static let contended = Status(raw: usearch_status_contended_k)
    /// A progress handler asked the operation to stop; what the finished increments leave behind
    /// is each verb's own contract.
    public static let cancelled = Status(raw: usearch_status_cancelled_k)
    /// The verb is declared for every residency but this one does not implement it.
    public static let unsupported = Status(raw: usearch_status_unsupported_k)
    /// An argument breaks the verb's contract.
    public static let invalidArgument = Status(raw: usearch_status_invalid_argument_k)

    /// The status rendered as the library spells it, like `invalid-argument`.
    public var description: String { String(cString: usearch_status_name(raw)) }

    /// Reads a spelling back, and nil for one no status holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_status_from_name, usearch_status_success_k) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_status_t) { self.raw = raw }
}

/// One failed call: a status every caller can branch on, and the detail the status cannot
/// carry.
public struct USearchError: Swift.Error, CustomStringConvertible, Equatable, Sendable {
    /// The failure identity, shared with every other SDK.
    public let status: Status
    /// The library's static detail, or this binding's own when it rejected the call.
    public let message: String

    public var description: String { "usearch: \(status): \(message)" }

    init(status: Status, message: String) {
        self.status = status
        self.message = message
    }

    /// Lifts a final outcome. Messages are static library storage and need no deallocation, so the
    /// copy is the only cost.
    init(_ outcome: usearch_outcome_t) {
        status = Status(raw: outcome.status)
        message = outcome.message.map { String(cString: $0) } ?? ""
    }
}

/// Reads `name` through a C reader that answers a status, and nil for a spelling it refuses.
func parsed<Raw>(
    _ name: String,
    _ read: (usearch_bytes_view_t, UnsafeMutablePointer<Raw>?) -> usearch_status_t,
    _ unread: Raw
) -> Raw? {
    var raw = unread
    let status = Array(name.utf8).withUnsafeBufferPointer {
        read(usearch_bytes_view_t(data: $0.baseAddress, length: $0.count), &raw)
    }
    return status == usearch_status_success_k ? raw : nil
}

// MARK: - Keys & Distances

/// A score between two entries, in the metric of the index that produced it.
public typealias Distance = Double

/// A 128-bit entry identity, most significant byte first, rendered as a lowercase dashed UUID.
///
/// Keys need not be unique: several entries may share one, and `DenseIndex.count(key:)`
/// reports how many. Integer identities park in the low eight bytes, so `Key(42)`
/// prints as `00000000-0000-0000-0000-00000000002a`.
public struct Key: Hashable, Comparable, CustomStringConvertible, ExpressibleByIntegerLiteral, Sendable {
    /// The sixteen identity bytes, most significant first - the engine's own key, so batches
    /// reinterpret rather than copy.
    var raw: usearch_key_t

    /// The all-zero key.
    public static let nilKey = Key()

    /// Builds the all-zero key.
    public init() { raw = usearch_key_t() }

    /// Parks an integer identity in the low eight bytes, zeroing the rest.
    public init(_ value: UInt64) {
        raw = usearch_key_t()
        for position in 0 ..< 8 { raw[15 - position] = UInt8(truncatingIfNeeded: value >> (8 * position)) }
    }

    public init(integerLiteral value: UInt64) { self.init(value) }

    /// Adopts sixteen bytes, most significant first.
    public init(bytes: [UInt8]) throws(USearchError) {
        guard bytes.count == 16 else {
            throw USearchError(status: .malformed, message: "a key is sixteen bytes, most significant first")
        }
        raw = usearch_key_t()
        for position in 0 ..< 16 { raw[position] = bytes[position] }
    }

    /// Reads the rendering `description` produces. Dashes may sit anywhere, or nowhere.
    public init(parsing text: String) throws(USearchError) {
        let malformed = USearchError(status: .malformed, message: "the key is not thirty-two hexadecimal digits")
        var digits: [UInt8] = []
        digits.reserveCapacity(32)
        for character in text.unicodeScalars where character != "-" {
            guard digits.count < 32, let digit = character.hexadecimalDigit else { throw malformed }
            digits.append(digit)
        }
        guard digits.count == 32 else { throw malformed }
        raw = usearch_key_t()
        for position in 0 ..< 16 { raw[position] = digits[position * 2] << 4 | digits[position * 2 + 1] }
    }

    /// Adopts the engine's key.
    init(raw: usearch_key_t) { self.raw = raw }

    /// The sixteen identity bytes, most significant first.
    public var byteArray: [UInt8] { (0 ..< 16).map { raw[$0] } }

    /// The integer identity in the low eight bytes, and nil when the high eight hold anything.
    public var integerIdentity: UInt64? {
        for position in 0 ..< 8 where raw[position] != 0 { return nil }
        var value: UInt64 = 0
        for position in 8 ..< 16 { value = value << 8 | UInt64(raw[position]) }
        return value
    }

    public var description: String {
        let hexadecimal = Array("0123456789abcdef".utf8)
        var rendered: [UInt8] = []
        rendered.reserveCapacity(36)
        for position in 0 ..< 16 {
            if position == 4 || position == 6 || position == 8 || position == 10 {
                rendered.append(UInt8(ascii: "-"))
            }
            rendered.append(hexadecimal[Int(raw[position] >> 4)])
            rendered.append(hexadecimal[Int(raw[position] & 0x0f)])
        }
        return String(decoding: rendered, as: UTF8.self)
    }

    public func hash(into hasher: inout Hasher) {
        for position in 0 ..< 16 { hasher.combine(raw[position]) }
    }

    public static func == (first: Key, second: Key) -> Bool {
        for position in 0 ..< 16 where first.raw[position] != second.raw[position] { return false }
        return true
    }

    public static func < (first: Key, second: Key) -> Bool {
        for position in 0 ..< 16 where first.raw[position] != second.raw[position] {
            return first.raw[position] < second.raw[position]
        }
        return false
    }
}

extension usearch_key_t {
    /// One identity byte, read through the tuple Swift imports the fixed-size array as.
    fileprivate subscript(position: Int) -> UInt8 {
        get { withUnsafeBytes(of: bytes) { $0[position] } }
        set { withUnsafeMutableBytes(of: &bytes) { $0[position] = newValue } }
    }
}

extension Key: Codable {
    /// Reads the key from the lowercase dashed rendering `description` produces.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let text = try container.decode(String.self)
        do {
            try self.init(parsing: text)
        }
        catch {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: error.message)
        }
    }

    /// Writes the key as that same rendering.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(description)
    }
}

extension Unicode.Scalar {
    /// The value of one hexadecimal digit, in either case, or nil for anything else.
    fileprivate var hexadecimalDigit: UInt8? {
        switch self {
        case "0" ... "9": return UInt8(value - 48)
        case "a" ... "f": return UInt8(value - 87)
        case "A" ... "F": return UInt8(value - 55)
        default: return nil
        }
    }
}

// MARK: - Scalars

/// A stored or exchanged scalar type, named by width and encoding. Sub-byte kinds pack several
/// dimensions per byte, as their `xN` suffixes spell.
public struct Scalar: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_scalar_t

    /// No scalar named - an unset field.
    public static let unknown = Scalar(raw: usearch_scalar_unknown_k)
    /// 64-bit double precision.
    public static let f64 = Scalar(raw: usearch_scalar_f64_k)
    /// 32-bit single precision.
    public static let f32 = Scalar(raw: usearch_scalar_f32_k)
    /// The brain floating-point format.
    public static let bf16 = Scalar(raw: usearch_scalar_bf16_k)
    /// IEEE half precision.
    public static let f16 = Scalar(raw: usearch_scalar_f16_k)
    /// 8-bit floating point with five exponent bits.
    public static let e5m2 = Scalar(raw: usearch_scalar_e5m2_k)
    /// 8-bit floating point with four exponent bits.
    public static let e4m3 = Scalar(raw: usearch_scalar_e4m3_k)
    /// 6-bit floating point with three exponent bits.
    public static let e3m2 = Scalar(raw: usearch_scalar_e3m2_k)
    /// 6-bit floating point with two exponent bits.
    public static let e2m3 = Scalar(raw: usearch_scalar_e2m3_k)
    /// An 8-bit signed integer.
    public static let i8 = Scalar(raw: usearch_scalar_i8_k)
    /// Two 4-bit signed integers per byte.
    public static let i4x2 = Scalar(raw: usearch_scalar_i4x2_k)
    /// A 32-bit signed integer, the coordinates of geospatial points.
    public static let i32 = Scalar(raw: usearch_scalar_i32_k)
    /// An 8-bit unsigned integer.
    public static let u8 = Scalar(raw: usearch_scalar_u8_k)
    /// Two 4-bit unsigned integers per byte.
    public static let u4x2 = Scalar(raw: usearch_scalar_u4x2_k)
    /// Eight single-bit dimensions per byte.
    public static let u1x8 = Scalar(raw: usearch_scalar_u1x8_k)

    /// The kind rendered as the library spells it.
    public var description: String { String(cString: usearch_scalar_name(raw)) }

    /// Reads a spelling back, and nil for one no kind holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_scalar_from_name, usearch_scalar_unknown_k) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_scalar_t) { self.raw = raw }

    /// How many logical dimensions one stored value holds: one for the byte-and-wider kinds,
    /// more for the packed ones.
    var dimensionsPerValue: Int {
        switch self {
        case .i4x2, .u4x2: return 2
        case .u1x8: return 8
        default: return 1
        }
    }
}

/// An element type a matrix may hold. Every conformance below names a type Swift or NumKong
/// already owns, so a caller converts a `Float` into any stored kind rather than only naming it,
/// and this SDK defines no scalar of its own.
public protocol ScalarValue {
    /// The scalar kind this element type crosses the boundary as.
    static var scalarKind: Scalar { get }
}

extension Double: ScalarValue {
    public static var scalarKind: Scalar { .f64 }
}
extension Float: ScalarValue {
    public static var scalarKind: Scalar { .f32 }
}
extension BFloat16: ScalarValue {
    public static var scalarKind: Scalar { .bf16 }
}
extension Float16: ScalarValue {
    public static var scalarKind: Scalar { .f16 }
}
extension E5M2: ScalarValue {
    public static var scalarKind: Scalar { .e5m2 }
}
extension E4M3: ScalarValue {
    public static var scalarKind: Scalar { .e4m3 }
}
extension E3M2: ScalarValue {
    public static var scalarKind: Scalar { .e3m2 }
}
extension E2M3: ScalarValue {
    public static var scalarKind: Scalar { .e2m3 }
}
extension Int8: ScalarValue {
    public static var scalarKind: Scalar { .i8 }
}
extension I4x2: ScalarValue {
    public static var scalarKind: Scalar { .i4x2 }
}
extension UInt8: ScalarValue {
    public static var scalarKind: Scalar { .u8 }
}
extension U4x2: ScalarValue {
    public static var scalarKind: Scalar { .u4x2 }
}
extension U1x8: ScalarValue {
    public static var scalarKind: Scalar { .u1x8 }
}

// MARK: - Rows

/// A viewed run of rows along a call's batch axis - the input of every verb whose rows share one
/// width - with the row count that sets the call's batch size.
///
/// Built over a NumKong `MatrixView`, it views and stays valid only as long as the storage the
/// view points into, so a memory-mapped dataset reaches the index with no reshape and no copy.
/// Built over a `Tensor`, it keeps the tensor alive, which is what a deferred call needs.
public struct RowsView: @unchecked Sendable {
    /// The first row.
    public let data: UnsafeRawPointer
    /// The rows the run holds, which the call takes as its batch axis.
    public let rows: Int
    /// Bytes between the starts of consecutive rows; zero repeats the first row.
    public let stride: Int
    /// Logical values per row, wider than a view's `columns` for the packed sub-byte kinds.
    public let columns: Int
    /// The scalar kind the rows hold.
    public let scalar: Scalar
    /// The storage the rows live in, held when the view keeps it alive.
    let owner: AnyObject?

    /// Views the rows of a NumKong view, which must outlive every call reading them.
    public init<Element: ScalarValue>(_ matrix: MatrixView<Element>) { self.init(matrix, owner: nil) }

    /// Views the rows of a NumKong tensor, keeping it alive as long as the view is.
    public init<Element: ScalarValue>(_ tensor: Tensor<Element>) { self.init(tensor.view(), owner: tensor) }

    private init<Element: ScalarValue>(_ matrix: MatrixView<Element>, owner: AnyObject?) {
        data = UnsafeRawPointer(matrix.baseAddress)
        rows = matrix.rows
        stride = matrix.rowStrideBytes
        columns = matrix.columns * Element.scalarKind.dimensionsPerValue
        scalar = Element.scalarKind
        self.owner = owner
    }

    /// The descriptor the engine reads.
    var raw: usearch_rows_view_t {
        usearch_rows_view_t(data: data, stride: stride, columns: columns, scalar: scalar.raw)
    }
}

// MARK: - Matches

/// One retrieved entry.
public struct Match: Equatable, Sendable {
    /// The matched entry's key.
    public let key: Key
    /// Its distance to the query, in the index's metric.
    public let distance: Distance
}

/// The ranked matches a search or a stream pull wrote into an arena, one list per query, read
/// in place until the arena's next call.
public struct Matches: RandomAccessCollection {
    /// The arena the matches live in, held so they outlive no release of it.
    let arena: Arena
    /// The packed matches the call wrote.
    var view = usearch_matches_view_t()
    /// One past the last query the call resolved.
    public let endIndex: Int

    /// The first query's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// The columns per query the call wrote, at most its `wanted` - the "k" in "kANN".
    public var columns: Int { view.columns }

    /// One query's matches, as many as the call wrote.
    public subscript(query: Int) -> QueryMatches {
        boundsCheck(query, below: endIndex, in: arena)
        let start = query * view.columns
        return QueryMatches(
            arena: arena,
            keys: view.keys,
            scores: view.scores,
            cells: start ..< start + view.counts[query]
        )
    }
}

/// One query's matches, or one left row's pairs, read in place from the arena they live in.
public struct QueryMatches: RandomAccessCollection {
    /// The arena the matches live in, held so they outlive no release of it.
    let arena: Arena
    /// The keys of every query of the call.
    let keys: UnsafePointer<usearch_key_t>?
    /// The scores of every query of the call, positionally paired with `keys`.
    let scores: UnsafePointer<usearch_distance_t>?
    /// The cells of `keys` and `scores` this query spans.
    let cells: Range<Int>

    /// The first match's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// One past the last match the call wrote.
    public var endIndex: Int { cells.count }

    /// The match at a position, key and score read from their paired arrays.
    public subscript(position: Int) -> Match {
        boundsCheck(position, below: cells.count, in: arena)
        let cell = cells.lowerBound + position
        return Match(key: Key(raw: keys![cell]), distance: scores![cell])
    }
}

// MARK: - Progress

/// Whether the operation a progress handler is watching continues or unwinds.
public enum Continuation: Sendable {
    /// Carry on to the next unit of work.
    case `continue`
    /// Stop; what the finished increments leave behind is each verb's own contract.
    case cancel
}

/// A periodic report of a long call's advance - units finished and units in total, in whatever
/// the verb consumes - and the caller's chance to stop it.
///
/// Never invoked concurrently, however many threads perform the work, so the handler needs no
/// synchronization of its own. It runs on whichever thread advanced the work, hence `@Sendable`.
public typealias Progress = @Sendable (_ completed: Int, _ total: Int) -> Continuation

/// A handler placed where its trampoline reads it, since a Swift closure has no stable
/// representation the engine can call.
final class Handler<Body> {
    let body: Body

    init(_ body: Body) { self.body = body }
}

/// The trampoline every reporting verb hands the engine.
private let progressTrampoline: usearch_progress_callback_t = { completed, total, state in
    let progress = Unmanaged<Handler<Progress>>.fromOpaque(state!).takeUnretainedValue()
    return progress.body(completed, total) == .cancel ? usearch_cancel_k : usearch_continue_k
}

/// The kind of work a refine or reshard plan is doing, the same in every residency.
public struct Phase: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_phase_t

    /// No plan, or one that has not started.
    public static let none = Phase(raw: usearch_phase_none_k)
    /// A reshard moving keys and values into a target, with the links it keeps.
    public static let copy = Phase(raw: usearch_phase_copy_k)
    /// A reshard on NVMe building blocks of a target's rows in RAM and writing their links.
    public static let build = Phase(raw: usearch_phase_build_k)
    /// A reshard cutting a source into one part per target.
    public static let partition = Phase(raw: usearch_phase_partition_k)
    /// A reshard moving each part into its target.
    public static let scatter = Phase(raw: usearch_phase_scatter_k)
    /// A round's health check, opening or closing it.
    public static let health = Phase(raw: usearch_phase_health_k)
    /// A round linking each island with the root's component.
    public static let heal = Phase(raw: usearch_phase_heal_k)
    /// A round drawing the hubs its gather streams entries past.
    public static let hubs = Phase(raw: usearch_phase_hubs_k)
    /// A round gathering every entry into a neighborhood.
    public static let gather = Phase(raw: usearch_phase_gather_k)
    /// A round re-stitching each neighborhood.
    public static let restitch = Phase(raw: usearch_phase_restitch_k)
    /// A round on NVMe rewriting each tile of links from what its restitch changed.
    public static let write = Phase(raw: usearch_phase_write_k)
    /// The plan ended its round, and nothing more is pending.
    public static let done = Phase(raw: usearch_phase_done_k)

    /// The phase rendered as the library spells it.
    public var description: String { String(cString: usearch_phase_name(raw)) }

    /// Reads a spelling back, and nil for one no phase holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_phase_from_name, usearch_phase_none_k) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_phase_t) { self.raw = raw }
}

/// Where a plan stands: its phase, that phase's position in the plan, and its own count.
public struct PhaseProgress: Sendable {
    /// The library's own record.
    var raw = usearch_phase_progress_t()

    /// The kind of work in hand.
    public var phase: Phase { Phase(raw: raw.phase) }
    /// This phase's position among the ones the plan runs.
    public var ordinal: Int { raw.ordinal }
    /// The units of this phase done, entries or islands.
    public var completed: Int { raw.completed }
    /// The units of this phase in all.
    public var total: Int { raw.total }
}

// MARK: - Call State

/// Everything one call points the library at, held where it cannot move until the call is final.
///
/// Submission copies the argument structure, so only its pointees live here: the outcome, the slots
/// the call writes, the handlers it calls back, and for a deferred call the copies of what it views
/// and the objects whose memory it reads. A blocking call releases it on return, and a deferred one
/// is kept by its `Deferred` until final.
final class CallState {
    /// Whether the call blocks or returns once submitted.
    let completion: usearch_completion_t
    /// The outcome the call stores its final status in.
    let outcome = UnsafeMutablePointer<usearch_outcome_t>.allocate(capacity: 1)
    /// The values the call reads or writes through, released with the call state.
    private var slots: [any Allocation] = []
    /// The handlers the call reads, and the objects whose memory a deferred call reads.
    private var anchors: [AnyObject] = []

    init(_ completion: usearch_completion_t = usearch_completion_blocking_k) {
        self.completion = completion
        outcome.initialize(to: usearch_outcome_t())
    }

    /// Runs only once the call is final: it returned blocking, its `Deferred` waited, or it never crossed.
    deinit {
        for slot in slots { slot.release() }
        outcome.deallocate()
    }

    /// Whether the call returns once submitted, so its result is read on a later wait.
    var isDeferred: Bool { completion == usearch_completion_deferred_k }

    /// Allocates `count` values, initialized to `initial`, which the call reads or writes through.
    func slots<Value>(_ initial: Value, count: Int = 1) -> UnsafeMutablePointer<Value> {
        let values = UnsafeMutablePointer<Value>.allocate(capacity: count)
        values.initialize(repeating: initial, count: count)
        slots.append(Values(start: values, count: count))
        return values
    }

    /// Keeps a handler alive until the call is final, answering the pointer its trampoline reads it
    /// through.
    func place(_ handler: AnyObject) -> UnsafeMutableRawPointer {
        anchors.append(handler)
        return Unmanaged.passUnretained(handler).toOpaque()
    }

    /// Keeps an object alive until a deferred call is final; a blocking call's caller already does.
    func anchor(_ object: AnyObject?) {
        if isDeferred, let object { anchors.append(object) }
    }

    /// Passes trivial `values` to the call: in place while a blocking call runs, or as a copy the
    /// call state owns, since a deferred call reads them after the verb returns.
    func pass<Element, Value>(
        _ values: [Element],
        _ body: (UnsafePointer<Element>?) throws(USearchError) -> Value
    ) throws(USearchError) -> Value {
        guard isDeferred, !values.isEmpty else {
            return try values.withUnsafeBufferPointer { (values) throws(USearchError) -> Value in
                try body(values.baseAddress)
            }
        }
        return try body(copied(values))
    }

    /// Copies trivial `values` into slots the call state owns until final; empty answers nil.
    func copied<Element>(_ values: [Element]) -> UnsafeMutablePointer<Element>? {
        guard let first = values.first else { return nil }
        let copy = slots(first, count: values.count)
        _ = UnsafeMutableBufferPointer(start: copy, count: values.count).update(fromContentsOf: values)
        return copy
    }

    /// Passes the UTF-8 bytes of a path, never terminated, by contract.
    func pass<Value>(
        _ text: String,
        _ body: (usearch_bytes_view_t) throws(USearchError) -> Value
    ) throws(USearchError) -> Value {
        let bytes = Array(text.utf8)
        return try pass(bytes) { (data) throws(USearchError) -> Value in
            try body(usearch_bytes_view_t(data: data, length: bytes.count))
        }
    }

    /// Passes an optional handler and its report count as the progress every reporting verb takes,
    /// so the trampoline can never be paired with the wrong state; nil passes a null callback.
    func pass(_ progress: Progress?, steps: Int) -> usearch_progress_t {
        guard let progress else { return usearch_progress_t(callback: nil, state: nil, steps: steps) }
        return usearch_progress_t(callback: progressTrampoline, state: place(Handler(progress)), steps: steps)
    }

    /// Waits until the call is final, then throws its failure.
    func wait() throws(USearchError) {
        guard usearch_outcome_wait(outcome) == usearch_status_success_k else { throw USearchError(outcome.pointee) }
    }

    /// The submission of a verb whose result the library wrote into `result`.
    func submitted<Value>(_ result: UnsafeMutablePointer<Value>) throws(USearchError) -> Submission<Value> {
        guard isDeferred else {
            try wait()
            return .final(result.pointee)
        }
        return .pending(self, result)
    }

    /// The submission of a verb with nothing to read but its status.
    func submitted() throws(USearchError) -> Submission<Void> { try submitted(slots(())) }
}

/// A run of values a call state allocated, released with it.
private protocol Allocation {
    func release()
}

private struct Values<Value>: Allocation {
    let start: UnsafeMutablePointer<Value>
    let count: Int

    func release() { start.deinitialize(count: count).deallocate() }
}

/// A call the library accepted, and the slot its result lands in.
enum Submission<Value> {
    /// A blocking call's result, read once the call returned final.
    case final(Value)
    /// A deferred call's state, and the slot its result lands in once final.
    case pending(CallState, UnsafeMutablePointer<Value>)

    /// Blocks until the call is final, then answers its result or throws its failure.
    func wait() throws(USearchError) -> Value {
        switch self {
        case .final(let value): return value
        case .pending(let callState, let result):
            try callState.wait()
            return result.pointee
        }
    }
}

/// The handle a lifecycle call wrote, which its success promises.
func created<Handle>(_ slot: UnsafeMutablePointer<Handle?>) throws(USearchError) -> Handle {
    guard let handle = slot.pointee else {
        throw USearchError(status: .internal, message: "the library reported success without a handle")
    }
    return handle
}

/// Traps on a position past `count`, as a collection does.
func boundsCheck(_ position: Int, below count: Int) {
    precondition(position >= 0 && position < count, "the position is out of bounds")
}

/// Traps on a position past `count`, or on outputs in memory the host may not read.
func boundsCheck(_ position: Int, below count: Int, in arena: Arena) {
    precondition(arena.hostAccess == .readable, hostRefusal)
    boundsCheck(position, below: count)
}

// MARK: - Deferred Calls

/// A call submitted to an executor, which finishes there while its one owner polls, waits, or does
/// other work.
///
/// It owns the call's outcome and outputs, and keeps alive every object whose memory the call
/// reads, until the call is final. The handles it names, the arena among them, it views as a
/// blocking call does. Releasing an unfinished one waits for it first.
public final class Deferred<Value> {
    /// The call state and the result slot.
    private let submission: Submission<Value>

    init(_ submission: Submission<Value>) { self.submission = submission }

    deinit {
        if case .pending(let callState, _) = submission { _ = usearch_outcome_wait(callState.outcome) }
    }

    /// `deferred` until the call is final, then `success` or its failure, read without blocking.
    /// Binding-only: reads the status word `wait` would, with its acquire ordering.
    public var status: Status {
        guard case .pending(let callState, _) = submission else { return .success }
        return Status(raw: usearch_outcome_status(callState.outcome))
    }

    /// Blocks until the call is final, then answers its result or throws its failure, the same
    /// every time it is asked.
    public func wait() throws(USearchError) -> Value { try submission.wait() }
}

// A call reading arena views stays on the arena's thread, so only a Sendable result may travel.
extension Deferred: @unchecked Sendable where Value: Sendable {}

extension Deferred {
    /// Suspends until the call is final, polling its status between sleeps that double from 1 µs
    /// to 1 ms, so no thread is parked on it; a cancelled task falls through to the blocking wait.
    /// The result is read back in the caller's isolation. Binding-only: polls `status`, runs
    /// `wait`.
    nonisolated(nonsending) public var value: Value {
        get async throws(USearchError) {
            var pause = Duration.microseconds(1)
            while status == .deferred {
                do { try await Task.sleep(for: pause) }
                catch { break }
                pause = min(pause * 2, .milliseconds(1))
            }
            return try wait()
        }
    }
}

// MARK: - Classes

/// A CPU family USearch ships tuned variants for; dispatch reports its pick in this vocabulary, and
/// a RAM or NVMe placement or executor may cap dispatch at one.
public struct CPUClass: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_cpu_class_t

    /// The portable floor, with no SIMD assumptions.
    public static let serial = CPUClass(raw: usearch_cpu_class_serial_k)
    /// Haswell-class x86: AVX2 and FMA.
    public static let intelHaswell = CPUClass(raw: usearch_cpu_class_intel_haswell_k)
    /// Sierra Forest-class x86: AVX-VNNI without AVX-512.
    public static let intelSierra = CPUClass(raw: usearch_cpu_class_intel_sierra_k)
    /// Skylake-class x86: the AVX-512 foundation.
    public static let intelSkylake = CPUClass(raw: usearch_cpu_class_intel_skylake_k)
    /// Icelake-class x86: AVX-512 VNNI and VPOPCNT.
    public static let intelIcelake = CPUClass(raw: usearch_cpu_class_intel_icelake_k)
    /// Genoa-class x86: native bf16 dot products over AVX-512.
    public static let amdGenoa = CPUClass(raw: usearch_cpu_class_amd_genoa_k)
    /// Sapphire Rapids-class x86: AVX-512 FP16 and the AMX tiles.
    public static let intelSapphire = CPUClass(raw: usearch_cpu_class_intel_sapphire_k)
    /// Granite Rapids-class x86: the AMX-FP16 tiles.
    public static let intelGranite = CPUClass(raw: usearch_cpu_class_intel_granite_k)
    /// Diamond Rapids-class x86: AVX10.2's FP8 conversions.
    public static let intelDiamond = CPUClass(raw: usearch_cpu_class_intel_diamond_k)
    /// Graviton2-class Arm: the bare NEON floor.
    public static let awsG2 = CPUClass(raw: usearch_cpu_class_aws_g2_k)
    /// Armv8.6 NEON with dot products and bf16.
    public static let appleM2 = CPUClass(raw: usearch_cpu_class_apple_m2_k)
    /// Apple M4-class Arm: the SME outer-product tiles.
    public static let appleM4 = CPUClass(raw: usearch_cpu_class_apple_m4_k)
    /// Graviton5-class Arm: Armv9.2.
    public static let awsG5 = CPUClass(raw: usearch_cpu_class_aws_g5_k)
    /// Vera-class Arm: FP8 dot products.
    public static let nvidiaVera = CPUClass(raw: usearch_cpu_class_nvidia_vera_k)
    /// Alibaba C930-class RISC-V: RVA23 vectors.
    public static let alibabaC930 = CPUClass(raw: usearch_cpu_class_alibaba_c930_k)
    /// IBM POWER10-class Power: VSX vectors and MMA accumulators.
    public static let ibmPower10 = CPUClass(raw: usearch_cpu_class_ibm_power10_k)
    /// WebAssembly with fixed-width simd128.
    public static let wasm32V128 = CPUClass(raw: usearch_cpu_class_wasm32_v128_k)
    /// WebAssembly with relaxed SIMD, threads and 64-bit addressing.
    public static let wasm64Extended = CPUClass(raw: usearch_cpu_class_wasm64_extended_k)
    /// The one class of a monoversioned build.
    public static let monoversioned = CPUClass(raw: usearch_cpu_class_monoversioned_k)

    /// The class as the library spells it.
    public var description: String { String(cString: usearch_cpu_class_name(raw)) }

    /// Reads a spelling back, and nil for one no class holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_cpu_class_from_name, usearch_cpu_class_unknown_k),
            raw != usearch_cpu_class_unknown_k
        else { return nil }
        self.raw = raw
    }

    init(raw: usearch_cpu_class_t) { self.raw = raw }
}

/// A GPU family USearch ships tuned variants for; a VRAM placement or executor may cap dispatch
/// at one.
public struct GPUClass: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_gpu_class_t

    /// Ampere-class NVIDIA, and every NVIDIA GPU without a class of its own.
    public static let nvidiaAmpere = GPUClass(raw: usearch_gpu_class_nvidia_ampere_k)
    /// Ada-class NVIDIA: the fp8 conversions.
    public static let nvidiaAda = GPUClass(raw: usearch_gpu_class_nvidia_ada_k)
    /// Hopper-class NVIDIA.
    public static let nvidiaHopper = GPUClass(raw: usearch_gpu_class_nvidia_hopper_k)
    /// Blackwell-class NVIDIA.
    public static let nvidiaBlackwell = GPUClass(raw: usearch_gpu_class_nvidia_blackwell_k)
    /// Consumer Blackwell NVIDIA.
    public static let nvidiaBlackwellRTX = GPUClass(raw: usearch_gpu_class_nvidia_blackwellrtx_k)
    /// Rubin-class NVIDIA.
    public static let nvidiaRubin = GPUClass(raw: usearch_gpu_class_nvidia_rubin_k)
    /// CDNA3 AMD: gfx942.
    public static let amdCDNA3 = GPUClass(raw: usearch_gpu_class_amd_cdna3_k)
    /// CDNA4 AMD: gfx950.
    public static let amdCDNA4 = GPUClass(raw: usearch_gpu_class_amd_cdna4_k)
    /// CDNA5 AMD: gfx1250.
    public static let amdCDNA5 = GPUClass(raw: usearch_gpu_class_amd_cdna5_k)
    /// Apple7-family GPUs, and every Apple GPU without a class of its own.
    public static let appleMetal7 = GPUClass(raw: usearch_gpu_class_apple_metal7_k)
    /// Apple9-family GPUs.
    public static let appleMetal9 = GPUClass(raw: usearch_gpu_class_apple_metal9_k)
    /// Apple10-family GPUs.
    public static let appleMetal10 = GPUClass(raw: usearch_gpu_class_apple_metal10_k)

    /// The class as the library spells it.
    public var description: String { String(cString: usearch_gpu_class_name(raw)) }

    /// Reads a spelling back, and nil for one no class holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_gpu_class_from_name, usearch_gpu_class_unknown_k) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_gpu_class_t) { self.raw = raw }
}

// MARK: - Executors

/// Where an executor computes, and with it where the arenas it makes keep their memory.
public struct Residency: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_residency_t

    /// Host memory, walked by host threads.
    public static let ram = Residency(raw: usearch_residency_ram_k)
    /// Device memory, walked by device streams; no host read reaches it.
    public static let vram = Residency(raw: usearch_residency_vram_k)
    /// NVMe drives, read through host submission rings into host memory.
    public static let nvme = Residency(raw: usearch_residency_nvme_k)

    /// The residency rendered as the library spells it.
    public var description: String { String(cString: usearch_residency_name(raw)) }

    /// Reads a spelling back, and nil for one no residency holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_residency_from_name, usearch_residency_unknown_k) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_residency_t) { self.raw = raw }
}

/// How a search reaches the entries it scores.
public struct Traversal: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_traversal_t

    /// Walks the graph, scoring the entries the expansion reaches - approximate, the default.
    public static let walk = Traversal(raw: usearch_traversal_walk_k)
    /// Scores every entry published when the query's scan starts - exact, on RAM and NVMe handles.
    public static let scan = Traversal(raw: usearch_traversal_scan_k)

    /// The traversal rendered as the library spells it.
    public var description: String { String(cString: usearch_traversal_name(raw)) }

    /// Reads a spelling back, and nil for one no traversal holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_traversal_from_name, usearch_traversal_unknown_k) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_traversal_t) { self.raw = raw }
}

/// What an executor does with its threads or streams once it goes idle.
public struct Mood: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_mood_t

    /// Idle threads and streams sleep; the next call pays a wake-up.
    public static let parked = Mood(raw: usearch_mood_parked_k)
    /// Idle threads and streams stay hot, trading power for the next call's latency.
    public static let spinning = Mood(raw: usearch_mood_spinning_k)

    /// The mood rendered as the library spells it.
    public var description: String { String(cString: usearch_mood_name(raw)) }

    /// Reads a spelling back, and nil for one no mood holds.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_mood_from_name, usearch_mood_parked_k) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_mood_t) { self.raw = raw }
}

/// The host threads or device streams a fanned-out verb runs on, and the traversal scratch each
/// of them walks with.
///
/// One executor serves any number of index handles, so separate categories of indexes share one
/// machine's cores or devices instead of each claiming a private set. Its kind is fixed by the
/// factory that spawned it and must match the residency of every handle it drives; a mismatch
/// reports `device`.
///
/// A verb called on the index itself runs on the calling thread - single-threaded on RAM, on the
/// handle's device and default stream on VRAM. One call holds an executor at a time, and a second
/// finding it held finishes `contended`, as nothing queues; index handles stay freely shareable. A
/// deferred call on a caller-exclusive pool runs on the pool's thread 0, so polling alone finishes
/// it, while a single thread or a VRAM executor runs it at submission.
public final class Executor: @unchecked Sendable {
    /// The native handle, released by `deinit`. Immutable for the object's whole life, so the
    /// engine's own synchronization is the only thing sharing this across threads relies on.
    let raw: usearch_executor_t

    /// Adopts a handle a factory below spawned.
    private init(raw: usearch_executor_t) { self.raw = raw }

    /// Creates an executor of host threads for RAM-resident indexes, interleaved across the compute
    /// domains local to `memoryDomains` - the very list the indexes were initialized with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains hold, since full occupancy starves the
    /// ordered publication path. A `cpuClassCeiling` intersects with what the machine runs, so a
    /// class the machine lacks narrows rather than refuses; nil means no ceiling.
    public static func forRam(
        memoryDomains: [Int32] = [],
        cpuClassCeiling: CPUClass? = nil,
        threads: Int = 0
    ) throws(USearchError) -> Executor {
        let callState = CallState()
        let executor = callState.slots(usearch_executor_t?.none)
        try callState.pass(memoryDomains) { (domains) throws(USearchError) in
            var arguments = usearch_executor_for_ram_t(
                executor: executor,
                outcome: callState.outcome,
                ram: usearch_ram_options_t(
                    memory_domains: domains,
                    memory_domains_count: memoryDomains.count,
                    shard_capacity: 0,
                    cpu_class_ceiling: cpuClassCeiling?.raw ?? usearch_cpu_class_unknown_k,
                    memory_limit: 0
                ),
                threads: threads,
                scratch_limit: 0
            )
            usearch_executor_for_ram(&arguments)
        }
        try callState.wait()
        return Executor(raw: try created(executor))
    }

    /// Creates an executor of host threads for NVMe-resident indexes, one submission ring apiece,
    /// across the compute domains local to the devices `paths` names.
    ///
    /// Zero `threads` derives from those devices' queue capabilities.
    public static func forNvme(
        paths: [String] = [],
        cpuClassCeiling: CPUClass? = nil,
        threads: Int = 0
    ) throws(USearchError) -> Executor {
        let callState = CallState()
        let executor = callState.slots(usearch_executor_t?.none)
        withPaths(paths, cpuClassCeiling) { nvme in
            var arguments = usearch_executor_for_nvme_t(
                executor: executor,
                outcome: callState.outcome,
                nvme: nvme,
                ram: usearch_ram_options_t(),
                threads: threads,
                nvme_class: usearch_nvme_class_unknown_k,
                scratch_limit: 0
            )
            usearch_executor_for_nvme(&arguments)
        }
        try callState.wait()
        return Executor(raw: try created(executor))
    }

    /// Opens one non-blocking stream on every named device, created and destroyed with the
    /// executor, all of them driven by the one host thread that calls the verbs - which is why this
    /// factory takes no thread count.
    ///
    /// `streams` is declared for the caller's CUDA streams, one per device in `devices` order,
    /// which answer `unsupported`, since the executor drives only its own; nil uses the executor's
    /// own. A zero `scratchLimit` lets per-device scratch grow as geometry asks. A
    /// `gpuClassCeiling` no named device loads is refused with `unsupported`; nil means no ceiling.
    public static func forVram(
        devices: [Int32] = [],
        gpuClassCeiling: GPUClass? = nil,
        streams: [UnsafeMutableRawPointer?]? = nil,
        scratchLimit: Int = 0
    ) throws(USearchError) -> Executor {
        // The C side reads one stream per device behind a pointer that holds no count.
        if let streams, streams.count != devices.count {
            throw USearchError(status: .shapeMismatch, message: "the streams must name one per device")
        }
        let callState = CallState()
        let executor = callState.slots(usearch_executor_t?.none)
        try callState.pass(devices) { (passedDevices) throws(USearchError) in
            try callState.pass(streams ?? []) { (passedStreams) throws(USearchError) in
                var arguments = usearch_executor_for_vram_t(
                    executor: executor,
                    outcome: callState.outcome,
                    vram: usearch_vram_options_t(
                        devices: passedDevices,
                        devices_count: devices.count,
                        shard_capacity: 0,
                        gpu_class_ceiling: gpuClassCeiling?.raw ?? usearch_gpu_class_unknown_k
                    ),
                    streams: streams == nil ? nil : passedStreams,
                    scratch_limit: scratchLimit
                )
                usearch_executor_for_vram(&arguments)
            }
        }
        try callState.wait()
        return Executor(raw: try created(executor))
    }

    deinit { usearch_executor_free(raw) }

    /// Waits until no call holds this executor, then adopts `mood` until the next call; a failure
    /// is the executor's own, such as a lost device, never one of a drained call.
    public func drain(mood: Mood = .parked) throws(USearchError) {
        let callState = CallState()
        var arguments = usearch_executor_drain_t(executor: raw, outcome: callState.outcome, mood: mood.raw)
        usearch_executor_drain(&arguments)
        try callState.wait()
    }
}

/// Passes NVMe paths as the byte views the engine's NVMe options point at, beside the class
/// ceiling, for one blocking call.
func withPaths<Value>(
    _ paths: [String],
    _ cpuClassCeiling: CPUClass?,
    _ body: (usearch_nvme_options_t) -> Value
) -> Value {
    let bytes = paths.flatMap(\.utf8)
    return bytes.withUnsafeBufferPointer { bytes in
        var offset = 0
        let views = paths.map { path in
            defer { offset += path.utf8.count }
            return usearch_bytes_view_t(data: bytes.baseAddress.map { $0 + offset }, length: path.utf8.count)
        }
        return views.withUnsafeBufferPointer { views in
            body(
                usearch_nvme_options_t(
                    paths: views.baseAddress,
                    paths_count: views.count,
                    cpu_class_ceiling: cpuClassCeiling?.raw ?? usearch_cpu_class_unknown_k
                )
            )
        }
    }
}

// MARK: - Arenas

/// Memory the library writes every output whose length scales with a call into - matches, rows,
/// products, monoliths, cursor and stream batches - reused across calls so a steady loop allocates
/// once.
///
/// Every such verb requires one and answers a view reading it in place. Each call given the
/// arena rewinds it, so the views an earlier call answered end there, and reading one after
/// reads what the next call wrote. To keep a result past the next call, give that call another
/// arena. Every call uses the arena, deferred ones included, and one naming it while another
/// call writes it is refused with `contended`. An arena serves one call at a time and is not
/// `Sendable`. One in GPU memory, as `describe` reports, serves only calls on the executor that
/// made it, keeps that executor alive, and answers views no host read may reach.
public final class Arena {
    /// The native handle, released by `deinit`.
    let raw: usearch_arena_t
    /// The VRAM executor that made it, held so the arena is freed first, on that executor's first stream.
    private var executor: Executor?
    /// Whether host code may read the outputs, as the executor that made it decided once.
    private(set) var hostAccess = HostAccess.refused

    /// Makes an arena where `executor` computes, host pages or its GPU's memory, decided once; nil
    /// makes host pages for calls on the calling thread.
    public init(on executor: Executor? = nil) throws(USearchError) {
        let callState = CallState()
        let arena = callState.slots(usearch_arena_t?.none)
        var arguments = usearch_arena_init_t(arena: arena, executor: executor?.raw, outcome: callState.outcome)
        usearch_arena_init(&arguments)
        try callState.wait()
        raw = try created(arena)
        let description = try describe()
        hostAccess = description.hostAccess
        if description.residency == .vram { self.executor = executor }
    }

    /// Frees the arena once every deferred call naming it is final. The free refuses only while an
    /// Arrow export of the outputs is open, and no Swift view opens one, so its status is dropped.
    deinit { _ = usearch_arena_free(raw) }

    /// Reports where the arena's outputs live, as the executor that made it decided.
    public func describe() throws(USearchError) -> ArenaDescription {
        let callState = CallState()
        let description = callState.slots(usearch_arena_description_t())
        var arguments = usearch_arena_describe_t(arena: raw, outcome: callState.outcome, description: description)
        usearch_arena_describe(&arguments)
        try callState.wait()
        return ArenaDescription(description.pointee)
    }
}

/// Where an arena's outputs live, which decides whether a host read reaches them.
public struct ArenaDescription: Sendable {
    /// `ram` for host pages, `vram` for one device's memory.
    public var residency: Residency
    /// The position of the GPU holding the outputs among the library's devices; nil for host pages.
    public var device: Int32?
    /// The vendor of that GPU; `unknown` for host pages.
    public var vendor: GpuVendor
    /// The vendor's own number for that GPU, which Arrow and DLPack name; nil for host pages.
    public var vendorOrdinal: Int32?
    /// Whether host code may read the outputs: host, pinned, managed and Apple's shared memory.
    public var hostAccess: HostAccess

    /// A description built by hand, as a test of a host read uses.
    public init(
        residency: Residency, device: Int32?, vendor: GpuVendor, vendorOrdinal: Int32?, hostAccess: HostAccess
    ) {
        self.residency = residency
        self.device = device
        self.vendor = vendor
        self.vendorOrdinal = vendorOrdinal
        self.hostAccess = hostAccess
    }

    init(_ raw: usearch_arena_description_t) {
        residency = Residency(raw: raw.residency)
        device = raw.device < 0 ? nil : raw.device
        vendor = GpuVendor(raw: raw.vendor)
        vendorOrdinal = raw.vendor_ordinal < 0 ? nil : raw.vendor_ordinal
        hostAccess = HostAccess(raw: raw.host_access)
    }

    /// Succeeds when host code may read the outputs, and throws `device` otherwise.
    public func host() throws(USearchError) {
        guard hostAccess == .readable else { throw USearchError(status: .device, message: hostRefusal) }
    }
}

/// Whether host code may read an arena's outputs.
public struct HostAccess: Hashable, Sendable, CustomStringConvertible {
    /// The engine's own value.
    let raw: usearch_host_access_t

    /// Only the device may read them.
    public static let refused = HostAccess(raw: usearch_host_access_refused_k)
    /// Host code may read them: host, pinned, managed and Apple's shared memory.
    public static let readable = HostAccess(raw: usearch_host_access_readable_k)

    /// The host access rendered as the library spells it.
    public var description: String { String(cString: usearch_host_access_name(raw)) }

    init(raw: usearch_host_access_t) { self.raw = raw }
}

/// The vendor whose runtime drives a GPU.
public enum GpuVendor: Sendable {
    /// No vendor, as host pages report.
    case unknown
    /// NVIDIA, through the CUDA driver.
    case nvidia
    /// AMD, through the HIP runtime.
    case amd
    /// Apple, through Metal.
    case apple

    /// Lifts the engine's value, `unknown` for none it names.
    init(raw: usearch_gpu_vendor_t) {
        switch raw {
        case usearch_gpu_vendor_nvidia_k: self = .nvidia
        case usearch_gpu_vendor_amd_k: self = .amd
        case usearch_gpu_vendor_apple_k: self = .apple
        default: self = .unknown
        }
    }
}

/// Why a read of an arena's outputs was refused.
let hostRefusal = "the outputs are memory the host may not read"

// MARK: - Introspection

/// The version of the linked library, like `3.0.0`.
public func version() -> String { String(cString: usearch_version_name()) }

/// The hardware capabilities the linked binary was compiled with, comma separated.
public func capabilitiesCompiled() -> String { String(cString: usearch_capabilities_compiled()) }

/// The hardware capabilities the current machine offers at runtime, comma separated.
public func capabilitiesAvailable() -> String { String(cString: usearch_capabilities_available()) }
