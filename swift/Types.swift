//
//  Types.swift
//  USearch
//
//  Shared vocabulary of the USearch Swift SDK.
//
//  Statuses, keys, scalar kinds, rows, executors, arenas, and deferred calls are
//  modality-agnostic, so they live here and the dense modality builds on them in DenseIndex.swift.
//
//  Batches travel in a `RowsView` over NumKong's `MatrixView` or `Tensor`, and its low-precision
//  types are the scalars a caller builds rows out of, and this file only tags them by `Scalar`.
//

import Dispatch
import NumKong
import Synchronization

import USearchC

// MARK: - Errors

/// What went wrong, in the vocabulary the engine and this binding share - one constant per
/// `usearch_status_*_k`, carrying the C enumerator itself. Validation performed here before a call
/// crosses reports the status the engine would have reported for the same fault.
///
/// A `USearchError` never carries `success` or `deferred`; a `Deferred` call reports both.
public struct Status: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: usearch_status_t

    /// The call succeeded and carries no message.
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
    /// A call was refused rather than waited for - a full deferred table, a busy arena, or a
    /// blocking call from inside a callback of the executor it would wait on.
    public static let contended = Status(raw: usearch_status_contended_k)
    /// A progress handler asked the operation to stop; what the finished increments leave behind
    /// is each verb's own contract.
    public static let cancelled = Status(raw: usearch_status_cancelled_k)
    /// The verb is declared for every residency but this one does not implement it.
    public static let unsupported = Status(raw: usearch_status_unsupported_k)
    /// An argument breaks the verb's contract.
    public static let invalidArgument = Status(raw: usearch_status_invalid_argument_k)

    /// The status rendered as the library spells it, like `invalid-argument`. Calls
    /// `usearch_status_name`.
    public var description: String { String(cString: usearch_status_name(raw)) }

    /// Reads a spelling back, and nil for one no status carries. Calls `usearch_status_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_status_from_name, usearch_status_name) else { return nil }
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
    /// What a caller's closure threw to stop the call, and nil for every other failure.
    public let cause: (any Swift.Error)?

    public var description: String { "usearch: \(status): \(message)" }

    init(status: Status, message: String) {
        self.status = status
        self.message = message
        cause = nil
    }

    /// Lifts a final outcome. Messages are static library storage and need no deallocation, so the
    /// copy is the only cost.
    init(_ outcome: usearch_outcome_t) {
        status = Status(raw: outcome.status)
        message = outcome.message.map { String(cString: $0) } ?? ""
        cause = nil
    }

    /// Carries what a caller's closure threw: a `USearchError` as it was, any other as `cancelled`.
    init(thrown: any Swift.Error) {
        if let thrown = thrown as? USearchError {
            self = thrown
            return
        }
        status = .cancelled
        message = String(describing: thrown)
        cause = thrown
    }

    /// Equal in status and message, which is as far as an arbitrary cause compares.
    public static func == (left: USearchError, right: USearchError) -> Bool {
        left.status == right.status && left.message == right.message
    }
}

/// Reads `name` through a C `from_name` table, keeping only a value whose own spelling echoes it,
/// since every table answers some default for a name it does not know.
func parsed<Raw>(
    _ name: String,
    _ read: (usearch_bytes_view_t) -> Raw,
    _ spell: (Raw) -> UnsafePointer<CChar>?
) -> Raw? {
    let raw = Array(name.utf8).withUnsafeBufferPointer {
        read(usearch_bytes_view_t(data: $0.baseAddress, length: $0.count))
    }
    return spell(raw).map { String(cString: $0) } == name ? raw : nil
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
    /// The sixteen identity bytes, most significant first - the C key itself, so batches
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

    /// Adopts the C key.
    init(raw: usearch_key_t) { self.raw = raw }

    /// The sixteen identity bytes, most significant first.
    public var byteArray: [UInt8] { (0 ..< 16).map { raw[$0] } }

    /// The integer identity in the low eight bytes, and nil when the high eight carry anything.
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
    /// One identity byte, read through the tuple Swift imports the C array member as.
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

/// A stored or exchanged scalar type, named by width and encoding, carrying the C enumerator
/// itself. Sub-byte kinds pack several dimensions per byte, as their `xN` suffixes spell.
public struct Scalar: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
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

    /// The kind rendered as the library spells it. Calls `usearch_scalar_name`.
    public var description: String { String(cString: usearch_scalar_name(raw)) }

    /// Reads a spelling back, and nil for one no kind carries. Calls `usearch_scalar_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_scalar_from_name, usearch_scalar_name) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_scalar_t) { self.raw = raw }

    /// How many logical dimensions one stored value carries: one for the byte-and-wider kinds,
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

/// A borrowed run of rows along a call's batch axis - the input of every verb whose rows share one
/// width - mirroring `usearch_rows_view_t`, plus the row count the call takes as its `count`.
///
/// Built over a NumKong `MatrixView`, it borrows and stays valid only as long as the storage the
/// view points into, so a memory-mapped dataset reaches the index with no reshape and no copy.
/// Built over a `Tensor`, it keeps the tensor alive, which is what a deferred call needs.
public struct RowsView: @unchecked Sendable {
    /// The first row.
    public let data: UnsafeRawPointer
    /// The rows the run carries, which the call takes as its batch axis.
    public let rows: Int
    /// Bytes between the starts of consecutive rows; zero repeats the first row.
    public let stride: Int
    /// Logical values per row, wider than a view's `columns` for the packed sub-byte kinds.
    public let columns: Int
    /// The scalar kind the rows hold.
    public let scalar: Scalar
    /// The storage the rows live in, held when the view keeps it alive.
    let owner: AnyObject?

    /// Borrows the rows of a NumKong view, which must outlive every call reading them.
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

    /// The descriptor the C interface reads.
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

/// The ranked rows a search, a `maxsim` or a stream pull wrote into an arena, one per query, read
/// in place. Mirrors `usearch_matches_view_t`.
public struct Matches: RandomAccessCollection {
    /// The arena the rows live in, and which call wrote them.
    let write: ArenaWrite
    /// The packed rows the call wrote.
    let view: usearch_matches_view_t
    /// One past the last query the call resolved.
    public let endIndex: Int

    /// The first query's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// The columns per query the call wrote, at most its `wanted` - the "k" in "kANN".
    public var columns: Int { view.columns }

    /// One query's matches, as many as the call wrote.
    public subscript(query: Int) -> MatchesRow {
        write.check(query, below: endIndex)
        let start = query * view.columns
        return MatchesRow(
            write: write,
            keys: view.keys,
            scores: view.scores,
            cells: start ..< start + view.counts[query]
        )
    }
}

/// One query's matches, or one left row's pairs, read in place from the arena they live in.
public struct MatchesRow: RandomAccessCollection {
    /// The arena the row lives in, and which call wrote it.
    let write: ArenaWrite
    /// The keys of every row of the call.
    let keys: UnsafePointer<usearch_key_t>?
    /// The scores of every row of the call, positionally paired with `keys`.
    let scores: UnsafePointer<usearch_distance_t>?
    /// The cells of `keys` and `scores` this row spans.
    let cells: Range<Int>

    /// The first match's position, which a collection walk starts from.
    public var startIndex: Int { 0 }

    /// One past the last match the call wrote.
    public var endIndex: Int { cells.count }

    /// The match at a position, key and score read from their paired arrays.
    public subscript(position: Int) -> Match {
        write.check(position, below: cells.count)
        let cell = cells.lowerBound + position
        return Match(key: Key(raw: keys![cell]), distance: scores![cell])
    }
}

// MARK: - Progress

/// Whether the operation a progress handler is watching carries on or unwinds.
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
/// synchronization of its own; an error it throws never unwinds into a C frame, but stops the call
/// as `.cancel` would and is thrown once the call is final. It runs on whichever thread advanced the
/// work, hence `@Sendable`.
public typealias Progress = @Sendable (_ completed: Int, _ total: Int) throws -> Continuation

/// A progress handler and the landing its first thrown error goes to, placed in that landing,
/// since a Swift closure has no stable C representation.
private final class ProgressState {
    let progress: Progress
    /// The landing that keeps this state, so it outlives every call of the trampoline.
    unowned let landing: Landing

    init(progress: @escaping Progress, landing: Landing) {
        self.progress = progress
        self.landing = landing
    }
}

/// The trampoline every reporting verb hands the engine.
private let progressTrampoline: usearch_progress_callback_t = { completed, total, state in
    guard let state else { return usearch_cancel_k }
    let progress = Unmanaged<ProgressState>.fromOpaque(state).takeUnretainedValue()
    do {
        return try progress.progress(completed, total) == .cancel ? usearch_cancel_k : usearch_continue_k
    }
    catch {
        progress.landing.latch(error)
        return usearch_cancel_k
    }
}

// MARK: - Landings

/// Everything one call points the library at, held where it cannot move until the call is final.
///
/// Submission copies the argument structure, so only its pointees live here: the outcome, the slots
/// the call writes, the closures it calls back, and for a deferred call the copies of what it
/// borrows and the objects whose memory it reads. A blocking call releases it on return, and a
/// deferred one is kept by its `Deferred` until final.
final class Landing {
    /// Whether the call blocks or returns once submitted.
    let completion: usearch_completion_t
    /// The outcome the call stores its final status in.
    let outcome = UnsafeMutablePointer<usearch_outcome_t>.allocate(capacity: 1)
    /// The first error a lent closure threw, from whichever thread threw it.
    private let thrown = Mutex<(any Swift.Error)?>(nil)
    /// The trivial values the call reads or writes through, freed with the landing.
    private var slots: [UnsafeMutableRawPointer] = []
    /// The callback states the call reads, and the objects whose memory a deferred call reads.
    private var anchors: [AnyObject] = []
    /// What the call hands back once final, like the donors a merge did not consume, taken by the
    /// first wait to settle, since several threads may wait at once.
    private let finals = Mutex<[@Sendable () -> Void]>([])
    /// The arena lent to the call, until `rewind` learns whether the call took it.
    private var lent: ArenaWrite?

    init(_ completion: usearch_completion_t = usearch_completion_blocking_k) {
        self.completion = completion
        outcome.initialize(to: usearch_outcome_t())
    }

    /// Runs only once the call is final: it returned blocking, its `Deferred` waited, or it never crossed.
    deinit {
        settle()
        for slot in slots { slot.deallocate() }
        outcome.deallocate()
    }

    /// Whether the call returns once submitted, so its result is read on a later wait.
    var isDeferred: Bool { completion == usearch_completion_deferred_k }

    /// Allocates `count` trivial values, initialized to `initial`, which the call reads or writes
    /// through.
    func slots<Value>(_ initial: Value, count: Int = 1) -> UnsafeMutablePointer<Value> {
        let values = UnsafeMutablePointer<Value>.allocate(capacity: count)
        values.initialize(repeating: initial, count: count)
        slots.append(UnsafeMutableRawPointer(values))
        return values
    }

    /// Keeps a callback's state alive until the call is final, answering the pointer its trampoline
    /// reads it through.
    func place(_ state: AnyObject) -> UnsafeMutableRawPointer {
        anchors.append(state)
        return Unmanaged.passUnretained(state).toOpaque()
    }

    /// Keeps `error` unless another thread's got there first, so the first failure wins.
    func latch(_ error: any Swift.Error) {
        thrown.withLock { if $0 == nil { $0 = error } }
    }

    /// The error a lent closure threw, read once the call is final.
    var failure: USearchError? {
        thrown.withLock { $0 }.map { USearchError(thrown: $0) }
    }

    /// Keeps an object alive until a deferred call is final; a blocking call's caller already does.
    func anchor(_ object: AnyObject?) {
        if isDeferred, let object { anchors.append(object) }
    }

    /// Lends trivial `values` to the call: in place while a blocking call runs, or as a copy the
    /// landing owns, since a deferred call reads them after the verb returns.
    func lend<Element, Value>(
        _ values: [Element],
        _ body: (UnsafePointer<Element>?) throws(USearchError) -> Value
    ) throws(USearchError) -> Value {
        guard isDeferred, let first = values.first else {
            return try values.withUnsafeBufferPointer { (values) throws(USearchError) -> Value in
                try body(values.baseAddress)
            }
        }
        let copy = slots(first, count: values.count)
        _ = UnsafeMutableBufferPointer(start: copy, count: values.count).update(fromContentsOf: values)
        return try body(copy)
    }

    /// Lends the UTF-8 bytes of a path, never terminated, by contract.
    func lend<Value>(
        _ text: String,
        _ body: (usearch_bytes_view_t) throws(USearchError) -> Value
    ) throws(USearchError) -> Value {
        let bytes = Array(text.utf8)
        return try lend(bytes) { (data) throws(USearchError) -> Value in
            try body(usearch_bytes_view_t(data: data, length: bytes.count))
        }
    }

    /// Lends an optional handler as the callback and state pair every reporting verb takes, so the
    /// trampoline can never be paired with the wrong state; nil lends nulls.
    func lend(_ progress: Progress?) -> (usearch_progress_callback_t?, UnsafeMutableRawPointer?) {
        guard let progress else { return (nil, nil) }
        return (progressTrampoline, place(ProgressState(progress: progress, landing: self)))
    }

    /// Lends `arena` to the call, answering the stamp its views carry once `rewind` advances the
    /// arena to it.
    func lend(_ arena: Arena) -> ArenaWrite {
        let write = ArenaWrite(arena: arena, generation: arena.generation &+ 1)
        lent = write
        return write
    }

    /// Advances the lent arena once the library took the call, after a blocking one returns or right
    /// after a deferred one is submitted, so the views earlier calls wrote there stop reading. Failed
    /// calls advance it too, and only a `contended` one, which never touched the arena, does not.
    func rewind() {
        guard let write = lent else { return }
        lent = nil
        if usearch_outcome_status(outcome) != usearch_status_contended_k { write.arena.generation &+= 1 }
    }

    /// Hands back what the call may leave behind once it is final: now for a blocking call, which is
    /// final on return, or on every wait of a deferred one, so each must hand back at most once.
    func onFinal(_ body: @escaping () -> Void) {
        guard isDeferred else { return body() }
        // What the closure captures belongs to the call until final, so the first settler may run it.
        nonisolated(unsafe) let body = body
        finals.withLock { $0.append { body() } }
    }

    private func settle() {
        var actions: [@Sendable () -> Void] = []
        finals.withLock { swap(&$0, &actions) }
        for action in actions { action() }
    }

    /// Waits until the call is final, hands back what it left, then answers its result or throws its
    /// failure, a closure's before the library's.
    func read<Value>(_ result: () throws(USearchError) -> Value) throws(USearchError) -> Value {
        let status = usearch_outcome_wait(outcome)
        rewind()
        settle()
        if let failure { throw failure }
        guard status == usearch_status_success_k else { throw USearchError(outcome.pointee) }
        return try result()
    }

    /// The submission of a verb with nothing to read but its status.
    func submitted() throws(USearchError) -> Submission<Void> {
        try isDeferred ? .pending(self) {} : .final(read {})
    }
}

/// A call the library accepted, and how to read what it produced.
///
/// A verb reads a blocking call's result at once, through a reader it borrows, and keeps the reader
/// only for a deferred one - so only a deferred call pays for the closure.
enum Submission<Value> {
    /// A blocking call's result, read once the call returned final.
    case final(Value)
    /// A deferred call's landing, and how to read its result once final.
    case pending(Landing, () throws(USearchError) -> Value)

    /// Blocks until the call is final, then answers its result or throws its failure.
    func wait() throws(USearchError) -> Value {
        switch self {
        case .final(let value): return value
        case .pending(let landing, let result): return try landing.read(result)
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

// MARK: - Deferred Calls

/// A call submitted to an executor, which finishes there while the caller does other work.
///
/// It owns the call's outcome and outputs, and keeps alive every object whose memory the call reads,
/// until the call is final. The handles it names, the arena among them, it borrows as a blocking
/// call does. Releasing an unfinished one waits for it first.
public final class Deferred<Value> {
    /// The landing and the result reader.
    private let submission: Submission<Value>

    init(_ submission: Submission<Value>) {
        self.submission = submission
        if case .pending(let landing, _) = submission { landing.rewind() }
    }

    deinit {
        if case .pending(let landing, _) = submission { _ = usearch_outcome_wait(landing.outcome) }
    }

    /// `deferred` until the call is final, then `success` or its failure, read without blocking.
    /// Binding-only: reads the status word `usearch_outcome_wait` would, with its acquire ordering.
    public var status: Status {
        guard case .pending(let landing, _) = submission else { return .success }
        let raw = usearch_outcome_status(landing.outcome)
        guard raw != usearch_status_deferred_k else { return .deferred }
        return landing.failure?.status ?? Status(raw: raw)
    }

    /// Blocks until the call is final, then answers its result or throws its failure, the same
    /// every time it is asked. Calls `usearch_outcome_wait`.
    public func wait() throws(USearchError) -> Value { try submission.wait() }
}

// A call reading arena views stays on the arena's thread, so only a Sendable result may travel.
extension Deferred: @unchecked Sendable where Value: Sendable {}

extension Deferred {
    /// Waits for the call on a Dispatch thread, so an `async` caller suspends rather than blocks,
    /// and the cooperative pool keeps every thread; the result is read back in the caller's
    /// isolation. Binding-only: runs `wait`.
    nonisolated(nonsending) public var value: Value {
        get async throws(USearchError) {
            if case .pending(let landing, _) = submission {
                nonisolated(unsafe) let outcome = landing.outcome
                await withCheckedContinuation { continuation in
                    DispatchQueue.global().async {
                        _ = usearch_outcome_wait(outcome)
                        continuation.resume()
                    }
                }
            }
            return try wait()
        }
    }
}

// MARK: - Executors

/// Where an executor computes, and with it where the arenas it makes keep their memory, carrying
/// the C enumerator.
public struct Residency: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: usearch_residency_t

    /// Host memory, walked by host threads.
    public static let ram = Residency(raw: usearch_residency_ram_k)
    /// Device memory, walked by device streams; no host read reaches it.
    public static let vram = Residency(raw: usearch_residency_vram_k)
    /// NVMe drives, read through host submission rings into host memory.
    public static let nvme = Residency(raw: usearch_residency_nvme_k)

    /// The residency rendered as the library spells it. Calls `usearch_residency_name`.
    public var description: String { String(cString: usearch_residency_name(raw)) }

    /// Reads a spelling back, and nil for one no residency carries. Calls
    /// `usearch_residency_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_residency_from_name, usearch_residency_name) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_residency_t) { self.raw = raw }
}

/// How a search reaches the rows it scores, carrying the C enumerator.
public struct Traversal: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: usearch_traversal_t

    /// Walks the graph, scoring the rows the expansion reaches - approximate, the default.
    public static let walk = Traversal(raw: usearch_traversal_walk_k)
    /// Scores every row published when the query's scan starts - exact, on RAM and NVMe handles.
    public static let scan = Traversal(raw: usearch_traversal_scan_k)

    /// The traversal rendered as the library spells it. Calls `usearch_traversal_name`.
    public var description: String { String(cString: usearch_traversal_name(raw)) }

    /// Reads a spelling back, and nil for one no traversal carries. Calls
    /// `usearch_traversal_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_traversal_from_name, usearch_traversal_name) else { return nil }
        self.raw = raw
    }

    init(raw: usearch_traversal_t) { self.raw = raw }
}

/// What an executor does with its threads or streams once it goes idle, carrying the C enumerator.
public struct Mood: Hashable, Sendable, CustomStringConvertible {
    /// The C enumerator.
    let raw: usearch_mood_t

    /// Idle threads and streams sleep; the next call pays a wake-up.
    public static let parked = Mood(raw: usearch_mood_parked_k)
    /// Idle threads and streams stay hot, trading power for the next call's latency.
    public static let spinning = Mood(raw: usearch_mood_spinning_k)

    /// The mood rendered as the library spells it. Calls `usearch_mood_name`.
    public var description: String { String(cString: usearch_mood_name(raw)) }

    /// Reads a spelling back, and nil for one no mood carries. Calls `usearch_mood_from_name`.
    public init?(name: String) {
        guard let raw = parsed(name, usearch_mood_from_name, usearch_mood_name) else { return nil }
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
/// handle's device and default stream on VRAM. Calls on one executor run in submission order, and
/// index handles stay freely shareable.
public final class Executor: @unchecked Sendable {
    /// The native handle, released by `deinit`. Immutable for the object's whole life, so the
    /// engine's own synchronization is the only thing sharing this across threads relies on.
    let raw: usearch_executor_t
    /// Where it computes, which the factory that spawned it named.
    public let residency: Residency

    /// Adopts a handle a factory below spawned.
    private init(raw: usearch_executor_t, residency: Residency) {
        self.raw = raw
        self.residency = residency
    }

    /// Spawns host threads for RAM-resident indexes, interleaved across the compute domains local
    /// to `memoryDomains` - the very list the indexes were initialized with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains carry, since full occupancy starves the
    /// ordered publication path. Every factory's `deferredLimit` sizes the table of deferred calls
    /// it tracks at once, and zero takes a default. Calls `usearch_executor_for_ram`.
    public static func forRam(
        memoryDomains: [Int32] = [],
        threads: Int = 0,
        deferredLimit: Int = 0
    ) throws(USearchError) -> Executor {
        let landing = Landing()
        let executor = landing.slots(usearch_executor_t?.none)
        try landing.lend(memoryDomains) { (domains) throws(USearchError) in
            var arguments = usearch_executor_for_ram_t(
                executor: executor,
                outcome: landing.outcome,
                ram: usearch_ram_options_t(memory_domains: domains, memory_domains_count: memoryDomains.count),
                threads: threads,
                deferred_limit: deferredLimit
            )
            usearch_executor_for_ram(&arguments)
        }
        return try landing.read { () throws(USearchError) in Executor(raw: try created(executor), residency: .ram) }
    }

    /// Creates a RAM executor over a borrowed ForkUnion `fu_pool_t`, so several libraries in one
    /// process share one set of threads while each keeps its own scratch. The pool must outlive
    /// the executor, and keeps its lender's mood. Calls `usearch_executor_on_pool`.
    public static func onPool(
        memoryDomains: [Int32] = [],
        pool: UnsafeMutableRawPointer,
        deferredLimit: Int = 0
    ) throws(USearchError) -> Executor {
        let landing = Landing()
        let executor = landing.slots(usearch_executor_t?.none)
        try landing.lend(memoryDomains) { (domains) throws(USearchError) in
            var arguments = usearch_executor_on_pool_t(
                executor: executor,
                outcome: landing.outcome,
                ram: usearch_ram_options_t(memory_domains: domains, memory_domains_count: memoryDomains.count),
                pool: pool,
                deferred_limit: deferredLimit
            )
            usearch_executor_on_pool(&arguments)
        }
        return try landing.read { () throws(USearchError) in Executor(raw: try created(executor), residency: .ram) }
    }

    /// Spawns host threads for NVMe-resident indexes, one submission ring apiece, across the
    /// compute domains local to the devices `paths` names.
    ///
    /// Zero `threads` derives from those devices' queue capabilities. Calls
    /// `usearch_executor_for_nvme`.
    public static func forNvme(
        paths: [String] = [],
        threads: Int = 0,
        deferredLimit: Int = 0
    ) throws(USearchError) -> Executor {
        let landing = Landing()
        let executor = landing.slots(usearch_executor_t?.none)
        withPaths(paths) { nvme in
            var arguments = usearch_executor_for_nvme_t(
                executor: executor,
                outcome: landing.outcome,
                nvme: nvme,
                threads: threads,
                deferred_limit: deferredLimit
            )
            usearch_executor_for_nvme(&arguments)
        }
        return try landing.read { () throws(USearchError) in Executor(raw: try created(executor), residency: .nvme) }
    }

    /// Opens one submission stream on every named device, all of them driven by the one host thread
    /// that calls the verbs - which is why this factory takes no thread count.
    ///
    /// `streams` borrows the caller's CUDA streams, one per device in `devices` order, and nil
    /// creates the executor's own. A zero `scratchLimit` lets per-device scratch grow as geometry
    /// asks. Calls `usearch_executor_for_vram`.
    public static func forVram(
        devices: [Int32] = [],
        streams: [UnsafeMutableRawPointer?]? = nil,
        scratchLimit: Int = 0,
        deferredLimit: Int = 0
    ) throws(USearchError) -> Executor {
        // The C side reads one stream per device behind a pointer that carries no count.
        if let streams, streams.count != devices.count {
            throw USearchError(status: .shapeMismatch, message: "the streams must name one per device")
        }
        let landing = Landing()
        let executor = landing.slots(usearch_executor_t?.none)
        try landing.lend(devices) { (lentDevices) throws(USearchError) in
            try landing.lend(streams ?? []) { (lentStreams) throws(USearchError) in
                var arguments = usearch_executor_for_vram_t(
                    executor: executor,
                    outcome: landing.outcome,
                    vram: usearch_vram_options_t(devices: lentDevices, devices_count: devices.count),
                    streams: streams == nil ? nil : lentStreams,
                    scratch_limit: scratchLimit,
                    deferred_limit: deferredLimit
                )
                usearch_executor_for_vram(&arguments)
            }
        }
        return try landing.read { () throws(USearchError) in Executor(raw: try created(executor), residency: .vram) }
    }

    deinit { usearch_executor_free(raw) }

    /// Waits for every call submitted to this executor, then adopts `mood` whenever it next goes
    /// idle; a failure is the executor's own, such as a lost device, never one of a drained call.
    /// Calls `usearch_executor_drain`.
    public func drain(mood: Mood = .parked) throws(USearchError) {
        let landing = Landing()
        var arguments = usearch_executor_drain_t(executor: raw, outcome: landing.outcome, mood: mood.raw)
        usearch_executor_drain(&arguments)
        try landing.read {}
    }
}

/// Lends NVMe paths as the byte views `usearch_nvme_options_t` points at, for one blocking call.
func withPaths<Value>(_ paths: [String], _ body: (usearch_nvme_options_t) -> Value) -> Value {
    let bytes = paths.flatMap(\.utf8)
    return bytes.withUnsafeBufferPointer { bytes in
        var offset = 0
        let views = paths.map { path in
            defer { offset += path.utf8.count }
            return usearch_bytes_view_t(data: bytes.baseAddress.map { $0 + offset }, length: path.utf8.count)
        }
        return views.withUnsafeBufferPointer { views in
            body(usearch_nvme_options_t(paths: views.baseAddress, paths_count: views.count))
        }
    }
}

// MARK: - Arenas

/// Memory the library writes every output whose length scales with a call into - matches, rows,
/// products, monoliths, cursor and stream batches - reused across calls so a steady loop allocates
/// once.
///
/// Every such verb requires one and answers a view reading it in place. Each call given the arena
/// rewinds it, so the views an earlier call answered stop reading: any read of them traps, as an
/// out-of-bounds index does. To keep a result past the next call, give that call another arena.
/// Every call borrows the arena, deferred ones included, and one naming it while another call
/// writes it is refused with `contended`. An arena serves one call at a time and is not `Sendable`.
public final class Arena {
    /// The native handle, released by `deinit`.
    let raw: usearch_arena_t
    /// Where its memory lives, which the executor that made it decided.
    public let residency: Residency
    /// Advanced by every call given the arena, so a view remembers which call wrote it.
    var generation: UInt64 = 0

    /// Makes an arena where `executor` computes, host pages or its GPU's memory, decided once; nil
    /// makes host pages for calls on the calling thread. Calls `usearch_arena_init`.
    public init(on executor: Executor? = nil) throws(USearchError) {
        let landing = Landing()
        let arena = landing.slots(usearch_arena_t?.none)
        var arguments = usearch_arena_init_t(arena: arena, executor: executor?.raw, outcome: landing.outcome)
        usearch_arena_init(&arguments)
        raw = try landing.read { () throws(USearchError) in try created(arena) }
        residency = executor?.residency ?? .ram
    }

    /// Calls `usearch_arena_free`.
    deinit { usearch_arena_free(raw) }
}

/// The arena a view reads, and which of the calls given it wrote the view.
struct ArenaWrite {
    let arena: Arena
    let generation: UInt64

    /// Refuses a read the arena no longer backs, or one reaching device memory.
    func check() {
        precondition(arena.generation == generation, "the arena was rewound since this view was written: uninitialized")
        precondition(arena.residency != .vram, "the view is device memory, which no host read reaches")
    }

    /// Refuses as `check()` does, then a `position` past `count`, as a collection does.
    func check(_ position: Int, below count: Int) {
        check()
        precondition(position >= 0 && position < count, "the position is out of bounds")
    }
}

// MARK: - Introspection

/// The version of the linked library, like `3.0.0`. Calls `usearch_version_name`.
public func version() -> String { String(cString: usearch_version_name()) }

/// The hardware capabilities the linked binary was compiled with, comma separated. Calls
/// `usearch_capabilities_compiled`.
public func capabilitiesCompiled() -> String { String(cString: usearch_capabilities_compiled()) }

/// The hardware capabilities the current machine offers at runtime, comma separated. Calls
/// `usearch_capabilities_available`.
public func capabilitiesAvailable() -> String { String(cString: usearch_capabilities_available()) }
