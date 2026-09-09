//
//  Types.swift
//  USearch
//
//  Shared vocabulary of the USearch Swift SDK - the twin of include/usearch/types.h.
//
//  Statuses, keys, scalar kinds, matrix lowering, and residency are modality-agnostic, so they
//  live here and the dense modality builds on them in DenseIndex.swift.
//
//  Batches travel in NumKong's `MatrixView` and `MatrixSpan`, and its low-precision types are the
//  scalars a caller builds rows out of; this file only tags them with the kind the C descriptors
//  carry.
//

import NumKong
import USearchC

// MARK: - Errors

/// What went wrong, in the vocabulary the engine and this binding share. Validation performed
/// here before a call crosses reports the status the engine would have reported for the same
/// fault.
public enum Status: Int32, Sendable {
    /// The call completed and carries no message.
    case success = 0
    /// An allocation or reservation could not be satisfied.
    case outOfMemory = 1
    /// Dimensionality, row counts, or output capacity disagree.
    case shapeMismatch = 2
    /// A stride or packing the engine cannot traverse.
    case unsupportedLayout = 3
    /// A scalar, metric, or residency tag this build does not know.
    case unknownKind = 4
    /// A mutating verb reached a handle that opens its checkpoint in place.
    case readOnly = 5
    /// A file could not be opened, read, written, or parsed.
    case ioFailure = 6
    /// A device could not be selected, or its memory could not be claimed.
    case deviceFailure = 7
    /// A caller-supplied representation could not be decoded.
    case malformedInput = 8
    /// The library reported neither success nor a failure.
    case internalFailure = 9
    /// A hard design ceiling was reached, like the 32-bit identity space.
    case overflow = 10
    /// The verb needs a prior successful reservation that never happened.
    case uninitialized = 11
    /// The request contradicts an already-frozen configuration.
    case configMismatch = 12
    /// A synchronization contract was violated by a concurrent operation; discard any produced
    /// output and retry in a quieter phase.
    case contended = 13
    /// A progress handler asked the operation to stop; it unwound and published nothing.
    case cancelled = 14
    /// A status a newer library reported and this build cannot name; never success.
    case unrecognized = -1

    /// The status rendered as its C enumerator's stem.
    public var name: String {
        switch self {
        case .success: return "success"
        case .outOfMemory: return "out_of_memory"
        case .shapeMismatch: return "shape_mismatch"
        case .unsupportedLayout: return "unsupported_layout"
        case .unknownKind: return "unknown_kind"
        case .readOnly: return "read_only"
        case .ioFailure: return "io"
        case .deviceFailure: return "device"
        case .malformedInput: return "malformed_input"
        case .internalFailure: return "internal"
        case .overflow: return "overflow"
        case .uninitialized: return "uninitialized"
        case .configMismatch: return "config_mismatch"
        case .contended: return "contended"
        case .cancelled: return "cancelled"
        case .unrecognized: return "unrecognized"
        }
    }
}

/// One failed call: a status every caller can branch on, and the detail the status cannot
/// carry.
public struct Error: Swift.Error, CustomStringConvertible, Equatable, Sendable {
    /// The failure identity, shared with every other SDK.
    public let status: Status
    /// The library's static detail, or this binding's own when it rejected the call.
    public let message: String

    public var description: String { "usearch: \(status.name): \(message)" }
}

/// Builds the failure this binding reports for a fault it caught before the boundary.
func fail(_ status: Status, _ message: String) -> Error {
    Error(status: status, message: message)
}

/// Turns a populated C error slot into a Swift error. Messages are static library storage and
/// need no deallocation, so the copy is the only cost.
func check(_ raw: usearch_error_t) throws(Error) {
    guard raw.status != usearch_success_k else { return }
    let named = Status(rawValue: Int32(bitPattern: raw.status.rawValue)) ?? .unrecognized
    let status = named == .success ? .unrecognized : named
    throw Error(status: status, message: raw.message.map { String(cString: $0) } ?? "")
}

/// Runs one C call against a zeroed error slot and lifts a populated slot into a thrown error.
func throwing<Value>(_ body: (UnsafeMutablePointer<usearch_error_t>) -> Value) throws(Error) -> Value {
    var failure = usearch_error_t()
    let value = body(&failure)
    try check(failure)
    return value
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
    /// The sixteen identity bytes, laid out exactly as the C key so batches reinterpret rather
    /// than copy.
    var storage: usearch_key_t

    /// The all-zero key.
    public static let nilKey = Key()

    /// Builds the all-zero key.
    public init() { storage = usearch_key_t() }

    /// Parks an integer identity in the low eight bytes, zeroing the rest.
    public init(_ value: UInt64) {
        storage = usearch_key_t()
        withUnsafeMutableBytes(of: &storage.bytes) { target in
            for position in 0 ..< 8 { target[15 - position] = UInt8(truncatingIfNeeded: value >> (8 * position)) }
        }
    }

    public init(integerLiteral value: UInt64) { self.init(value) }

    /// Adopts sixteen bytes, most significant first.
    public init(bytes: borrowing [UInt8]) throws(Error) {
        guard bytes.count == 16 else { throw fail(.malformedInput, "a key is sixteen bytes, most significant first") }
        storage = usearch_key_t()
        withUnsafeMutableBytes(of: &storage.bytes) { target in
            for position in 0 ..< 16 { target[position] = bytes[position] }
        }
    }

    /// Reads the rendering `description` produces. Dashes may sit anywhere, or nowhere.
    public init(parsing text: borrowing String) throws(Error) {
        let malformed = "the key is not thirty-two hexadecimal digits"
        var digits: [UInt8] = []
        digits.reserveCapacity(32)
        for character in text.unicodeScalars where character != "-" {
            guard digits.count < 32, let digit = character.hexadecimalDigit else {
                throw fail(.malformedInput, malformed)
            }
            digits.append(digit)
        }
        guard digits.count == 32 else { throw fail(.malformedInput, malformed) }
        storage = usearch_key_t()
        withUnsafeMutableBytes(of: &storage.bytes) { target in
            for position in 0 ..< 16 { target[position] = digits[position * 2] << 4 | digits[position * 2 + 1] }
        }
    }

    init(raw: usearch_key_t) { storage = raw }

    /// The sixteen identity bytes, most significant first.
    public var byteArray: [UInt8] { withUnsafeBytes(of: storage.bytes) { Array($0) } }

    /// The integer identity in the low eight bytes, and nil when the high eight carry anything.
    public var integerIdentity: UInt64? {
        withUnsafeBytes(of: storage.bytes) { source in
            for position in 0 ..< 8 where source[position] != 0 { return nil }
            var value: UInt64 = 0
            for position in 8 ..< 16 { value = value << 8 | UInt64(source[position]) }
            return value
        }
    }

    public var description: String {
        withUnsafeBytes(of: storage.bytes) { source in
            let hexadecimal = Array("0123456789abcdef".utf8)
            var rendered: [UInt8] = []
            rendered.reserveCapacity(36)
            for position in 0 ..< 16 {
                if position == 4 || position == 6 || position == 8 || position == 10 {
                    rendered.append(UInt8(ascii: "-"))
                }
                rendered.append(hexadecimal[Int(source[position] >> 4)])
                rendered.append(hexadecimal[Int(source[position] & 0x0f)])
            }
            return String(decoding: rendered, as: UTF8.self)
        }
    }

    public func hash(into hasher: inout Hasher) {
        withUnsafeBytes(of: storage.bytes) { hasher.combine(bytes: $0) }
    }

    public static func == (first: Key, second: Key) -> Bool {
        withUnsafeBytes(of: first.storage.bytes) { left in
            withUnsafeBytes(of: second.storage.bytes) { right in left.elementsEqual(right) }
        }
    }

    public static func < (first: Key, second: Key) -> Bool {
        withUnsafeBytes(of: first.storage.bytes) { left in
            withUnsafeBytes(of: second.storage.bytes) { right in
                for position in 0 ..< 16 where left[position] != right[position] {
                    return left[position] < right[position]
                }
                return false
            }
        }
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
public enum Scalar: Int32, Sendable {
    /// A type the library could not name.
    case unknown = 0
    /// 64-bit double precision.
    case f64 = 1
    /// 32-bit single precision.
    case f32 = 2
    /// The brain floating-point format.
    case bf16 = 3
    /// IEEE half precision.
    case f16 = 4
    /// 8-bit floating point with five exponent bits.
    case e5m2 = 5
    /// 8-bit floating point with four exponent bits.
    case e4m3 = 6
    /// 6-bit floating point with three exponent bits.
    case e3m2 = 7
    /// 6-bit floating point with two exponent bits.
    case e2m3 = 8
    /// An 8-bit signed integer.
    case i8 = 10
    /// Two 4-bit signed integers per byte.
    case i4x2 = 11
    /// An 8-bit unsigned integer.
    case u8 = 20
    /// Two 4-bit unsigned integers per byte.
    case u4x2 = 21
    /// Eight single-bit dimensions per byte.
    case u1x8 = 22

    /// The kind rendered as its C enumerator's stem.
    public var name: String {
        switch self {
        case .unknown: return "unknown"
        case .f64: return "f64"
        case .f32: return "f32"
        case .bf16: return "bf16"
        case .f16: return "f16"
        case .e5m2: return "e5m2"
        case .e4m3: return "e4m3"
        case .e3m2: return "e3m2"
        case .e2m3: return "e2m3"
        case .i8: return "i8"
        case .i4x2: return "i4x2"
        case .u8: return "u8"
        case .u4x2: return "u4x2"
        case .u1x8: return "u1x8"
        }
    }

    /// How many logical dimensions one stored value carries: one for the byte-and-wider kinds,
    /// more for the packed ones.
    @inlinable public var dimensionsPerValue: Int {
        switch self {
        case .i4x2, .u4x2: return 2
        case .u1x8: return 8
        default: return 1
        }
    }

    var raw: usearch_scalar_t { usearch_scalar_t(rawValue: UInt32(bitPattern: rawValue)) }

    /// Lifts a tag the library wrote, refusing to guess at the numbering gaps it reserves.
    static func lifted(_ raw: usearch_scalar_t) throws(Error) -> Scalar {
        guard let kind = Scalar(rawValue: Int32(bitPattern: raw.rawValue)), kind != .unknown else {
            throw fail(.unknownKind, "the checkpoint names an unknown scalar kind")
        }
        return kind
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

// MARK: - Matrices

/// Bytes spanned by a rectangle of `rows` by `columns` elements, refusing the products that do
/// not fit an `Int`.
///
/// The wanted count of a search has no ceiling, so the outputs it implies are the one place a
/// caller's arithmetic can silently wrap before a single byte is allocated.
func checkedExtent(rows: Int, columns: Int, elementBytes: Int) throws(Error) -> Int {
    guard rows >= 0, columns >= 0, elementBytes > 0 else {
        throw fail(.shapeMismatch, "a matrix extent cannot be negative")
    }
    let (cells, cellsOverflowed) = rows.multipliedReportingOverflow(by: columns)
    guard !cellsOverflowed else { throw fail(.overflow, "the matrix has more cells than an index can address") }
    let (bytes, bytesOverflowed) = cells.multipliedReportingOverflow(by: elementBytes)
    guard !bytesOverflowed else { throw fail(.overflow, "the matrix spans more bytes than an index can address") }
    return bytes
}

/// Every batch input takes a NumKong matrix - a base pointer, a row count, and a byte pitch
/// between consecutive row starts - so a memory-mapped dataset reaches the index with no reshape
/// and no copy, and a single vector is a one-row matrix.
///
/// A matrix borrows and never owns; it stays valid only as long as the storage it was built from,
/// which is why every call site here builds one inside a `withUnsafeBufferPointer`.
extension MatrixView where Element: ScalarValue {
    /// The logical width of one row, which is wider than `cols` for the sub-byte kinds that pack
    /// several dimensions into one stored value.
    var dimensions: Int { cols * Element.scalarKind.dimensionsPerValue }

    /// Lowers into the descriptor the C interface reads, borrowing the same storage this matrix
    /// already borrows and tagging it with the kind its element type names.
    var lowered: usearch_matrix_view_t {
        usearch_matrix_view_t(
            data: UnsafeRawPointer(baseAddress),
            rows: rows,
            columns: dimensions,
            stride_bytes: rowStrideBytes,
            scalar_kind: Element.scalarKind.raw
        )
    }

    /// Checks the matrix against a frozen geometry. A width of zero skips the geometry check, for
    /// the handles whose geometry a later load decides.
    func validate(dimensions frozen: Int) throws(Error) {
        guard rows >= 0, cols > 0 else { throw fail(.shapeMismatch, "a matrix row cannot be empty") }
        guard rowStrideBytes >= cols * MemoryLayout<Element>.stride else {
            throw fail(.unsupportedLayout, "the row pitch cannot hold a whole row")
        }
        guard frozen == 0 || dimensions == frozen else {
            throw fail(.shapeMismatch, "the matrix rows are not as wide as the index geometry")
        }
    }
}

extension MatrixSpan where Element: ScalarValue {
    /// The logical width of one row, counted like its read-only twin's.
    var dimensions: Int { cols * Element.scalarKind.dimensionsPerValue }

    /// Lowers into the descriptor the C interface writes through, whose `capacity` is the rows
    /// this matrix can receive rather than a count it promises to fill.
    var lowered: usearch_matrix_span_t {
        usearch_matrix_span_t(
            data: UnsafeMutableRawPointer(baseAddress),
            capacity: rows,
            columns: dimensions,
            stride_bytes: rowStrideBytes,
            scalar_kind: Element.scalarKind.raw
        )
    }

    /// Checks the matrix against a frozen geometry, like its read-only twin.
    func validate(dimensions frozen: Int) throws(Error) {
        guard rows >= 0, cols > 0 else { throw fail(.shapeMismatch, "a matrix row cannot be empty") }
        guard rowStrideBytes >= cols * MemoryLayout<Element>.stride else {
            throw fail(.unsupportedLayout, "the row pitch cannot hold a whole row")
        }
        guard frozen == 0 || dimensions == frozen else {
            throw fail(.shapeMismatch, "the matrix rows are not as wide as the index geometry")
        }
    }
}

// MARK: - Residency

/// Where an index's payload lives, which engine traverses it, and the hardware its shards are
/// placed across - one shard per named memory domain, device, or path.
///
/// Names where the index lives, never who walks it: threads and streams belong to the
/// `Executor` the fanned-out verbs take. An empty placement derives one from the machine.
public enum Residency: Sendable {
    /// A volatile-memory index traversed by CPU threads - the concurrent baseline. The NUMA
    /// node ids name where shards are placed; an empty list derives every node the process may
    /// allocate on.
    case ram(memoryDomains: [Int32])
    /// An accelerator-memory index traversed by GPU warps, sharded across the named device
    /// ordinals and never replicated; an empty list derives one free device.
    case vram(devices: [Int32])
    /// A persisted index traversed through pluggable I/O engines, sized past RAM. A path is
    /// whatever identifies a device to the engine that can reach it - a filesystem path, a
    /// transport identifier, or a FUSE mount over object storage; an empty list leaves every
    /// shard where its manifest points.
    case nvme(paths: [String])

    /// A RAM index whose memory domains the machine derives.
    public static let `default` = Residency.ram(memoryDomains: [])

    /// The tag naming the active arm of the C options union.
    var tag: usearch_residency_t {
        switch self {
        case .ram: return usearch_residency_ram_k
        case .vram: return usearch_residency_vram_k
        case .nvme: return usearch_residency_nvme_k
        }
    }

    /// The residency a tag names, with an empty placement.
    ///
    /// A checkpoint records where it was written, never where a reader must keep it, so the
    /// placement lists come back empty for the caller to fill before initialization.
    static func lifted(_ tag: usearch_residency_t) throws(Error) -> Residency {
        switch tag {
        case usearch_residency_ram_k: return .ram(memoryDomains: [])
        case usearch_residency_vram_k: return .vram(devices: [])
        case usearch_residency_nvme_k: return .nvme(paths: [])
        default: throw fail(.unknownKind, "the checkpoint names a residency this binding does not know")
        }
    }
}

/// The C-side copies of a residency's placement lists, keeping every pointer the options union
/// carries valid for as long as this holder is alive.
///
/// Swift array storage cannot outlive the closure that borrowed it, and an NVMe placement needs
/// one byte view per path pointing into storage of its own, so both are copied out once here
/// rather than nested closure by closure.
final class LoweredResidency {
    /// The tag naming which of the three arms below the options union reads.
    let tag: usearch_residency_t
    /// The host-memory arm, zeroed unless the residency is RAM.
    let ram: usearch_ram_options_t
    /// The accelerator arm, zeroed unless the residency is VRAM.
    let vram: usearch_vram_options_t
    /// The persisted arm, zeroed unless the residency is NVMe.
    let nvme: usearch_nvme_options_t

    /// The NUMA node ids or device ordinals the two integer arms point at.
    private let ordinals: UnsafeMutableBufferPointer<Int32>
    /// One borrowed view per NVMe path, pointing into `text`.
    private let views: UnsafeMutableBufferPointer<usearch_bytes_view_t>
    /// The UTF-8 bytes of every NVMe path, laid end to end and never terminated.
    private let text: UnsafeMutableBufferPointer<UInt8>

    /// Copies whichever placement list the residency names, leaving the other two arms zeroed.
    init(_ residency: Residency) {
        tag = residency.tag
        switch residency {
        case .ram(let memoryDomains):
            ordinals = LoweredResidency.copied(memoryDomains)
            views = UnsafeMutableBufferPointer(start: nil, count: 0)
            text = UnsafeMutableBufferPointer(start: nil, count: 0)
            ram = usearch_ram_options_t(
                memory_domains: ordinals.baseAddress.map { UnsafePointer($0) },
                memory_domains_count: ordinals.count
            )
            vram = usearch_vram_options_t()
            nvme = usearch_nvme_options_t()
        case .vram(let devices):
            ordinals = LoweredResidency.copied(devices)
            views = UnsafeMutableBufferPointer(start: nil, count: 0)
            text = UnsafeMutableBufferPointer(start: nil, count: 0)
            ram = usearch_ram_options_t()
            vram = usearch_vram_options_t(
                devices: ordinals.baseAddress.map { UnsafePointer($0) },
                devices_count: ordinals.count
            )
            nvme = usearch_nvme_options_t()
        case .nvme(let paths):
            ordinals = UnsafeMutableBufferPointer(start: nil, count: 0)
            (views, text) = LoweredResidency.copied(paths)
            ram = usearch_ram_options_t()
            vram = usearch_vram_options_t()
            nvme = usearch_nvme_options_t(
                paths: views.baseAddress.map { UnsafePointer($0) },
                paths_count: views.count
            )
        }
    }

    deinit {
        ordinals.deallocate()
        views.deallocate()
        text.deallocate()
    }

    /// Stamps the tag and the arm it selects into a zeroed options structure.
    func write(into options: inout usearch_dense_options_t) {
        options.residency = tag
        switch tag {
        case usearch_residency_vram_k: options.vram = vram
        case usearch_residency_nvme_k: options.nvme = nvme
        default: options.ram = ram
        }
    }

    /// Copies a placement list out of Swift storage, reporting an empty list as the null pointer
    /// the C interface derives from.
    private static func copied(_ ordinals: [Int32]) -> UnsafeMutableBufferPointer<Int32> {
        guard !ordinals.isEmpty else { return UnsafeMutableBufferPointer(start: nil, count: 0) }
        let buffer = UnsafeMutableBufferPointer<Int32>.allocate(capacity: ordinals.count)
        _ = buffer.initialize(fromContentsOf: ordinals)
        return buffer
    }

    /// Copies every path's UTF-8 bytes into one run and builds the views naming them.
    private static func copied(
        _ paths: [String]
    ) -> (UnsafeMutableBufferPointer<usearch_bytes_view_t>, UnsafeMutableBufferPointer<UInt8>) {
        let total = paths.reduce(0) { $0 + $1.utf8.count }
        guard !paths.isEmpty, total != 0 else {
            return (UnsafeMutableBufferPointer(start: nil, count: 0), UnsafeMutableBufferPointer(start: nil, count: 0))
        }
        let text = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: total)
        let views = UnsafeMutableBufferPointer<usearch_bytes_view_t>.allocate(capacity: paths.count)
        var offset = 0
        for (position, path) in paths.enumerated() {
            let start = offset
            for byte in path.utf8 {
                text[offset] = byte
                offset += 1
            }
            views[position] = usearch_bytes_view_t(
                data: UnsafePointer(text.baseAddress! + start),
                length: offset - start
            )
        }
        return (views, text)
    }
}

// MARK: - Executors

/// The host threads or device streams a fanned-out verb runs on, and the traversal scratch each
/// of them walks with.
///
/// One executor serves any number of index handles, so separate categories of indexes share one
/// machine's cores or devices instead of each claiming a private set. Its kind is fixed by the
/// factory that spawned it and must match the residency of every handle it drives; a mismatch
/// reports `deviceFailure` on the first call.
///
/// An executor admits one driver at a time - a second concurrent call reports `contended` - so a
/// thread that wants to drive its own calls spawns its own executor. Index handles stay freely
/// shareable.
public final class Executor: @unchecked Sendable {
    /// The native handle, released by `deinit`. Immutable for the object's whole life, so the
    /// engine's own synchronization is the only thing sharing this across threads relies on.
    private let handle: usearch_executor_t

    /// Adopts a handle a factory below spawned.
    private init(handle: usearch_executor_t) {
        self.handle = handle
    }

    /// Spawns host workers for RAM-resident indexes, interleaved across the compute domains local
    /// to `memoryDomains` - the very list the indexes were initialized with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains carry, since full occupancy starves the
    /// ordered publication path.
    public static func forRam(memoryDomains: [Int32] = [], threads: Int = 0) throws(Error) -> Executor {
        let placement = LoweredResidency(.ram(memoryDomains: memoryDomains))
        return try spawned(placement) { failure in
            withUnsafePointer(to: placement.ram) { usearch_executor_for_ram($0, threads, failure) }
        }
    }

    /// Spawns host workers for NVMe-resident indexes, one submission ring apiece, across the
    /// compute domains local to the devices `paths` names.
    ///
    /// Zero `threads` derives from those devices' queue capabilities.
    public static func forNvme(paths: [String] = [], threads: Int = 0) throws(Error) -> Executor {
        let placement = LoweredResidency(.nvme(paths: paths))
        return try spawned(placement) { failure in
            withUnsafePointer(to: placement.nvme) { usearch_executor_for_nvme($0, threads, failure) }
        }
    }

    /// Opens one submission stream on every named device, all of them driven by the one host thread
    /// that calls the verbs - which is why this factory takes no thread count.
    public static func forVram(devices: [Int32] = []) throws(Error) -> Executor {
        let placement = LoweredResidency(.vram(devices: devices))
        return try spawned(placement) { failure in
            withUnsafePointer(to: placement.vram) { usearch_executor_for_vram($0, failure) }
        }
    }

    /// Runs one factory against a zeroed error slot, keeping the placement copies alive across it
    /// and refusing a null handle the library reported no failure for.
    private static func spawned(
        _ placement: LoweredResidency,
        _ open: (UnsafeMutablePointer<usearch_error_t>) -> usearch_executor_t?
    ) throws(Error) -> Executor {
        let created = try withExtendedLifetime(placement) { () throws(Error) -> usearch_executor_t? in
            try throwing { failure in open(failure) }
        }
        guard let created else { throw fail(.internalFailure, "the library returned no executor and no error") }
        return Executor(handle: created)
    }

    deinit { usearch_executor_free(handle) }

    /// Waits for every deferred call on this executor to finish, reporting the first failure a
    /// drained call held back.
    public func drain() throws(Error) {
        try throwing { usearch_executor_drain(handle, $0) }
    }

    /// The raw handle the modality verbs pass across the boundary.
    var raw: usearch_executor_t { handle }
}

// MARK: - Introspection

/// The version of the linked library.
public func usearchVersion() -> String { String(cString: usearch_version()) }

/// The hardware capabilities the linked binary was compiled with, comma separated.
public func usearchCapabilitiesCompiled() -> String { String(cString: usearch_capabilities_compiled()) }

/// The hardware capabilities the current machine offers at runtime, comma separated.
public func usearchCapabilitiesAvailable() -> String { String(cString: usearch_capabilities_available()) }
