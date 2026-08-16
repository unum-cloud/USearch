//
//  Types.swift
//  USearch
//
//  Shared vocabulary of the USearch Swift SDK - the twin of include/usearch/types.h.
//
//  Statuses, keys, scalar kinds, matrix views, and residency are modality-agnostic, so they
//  live here and the dense modality builds on them in DenseIndex.swift.
//

import USearchC

// region: Errors

/// What went wrong, in the vocabulary the engine and this binding share. Validation performed
/// here before a call crosses reports the status the engine would have reported for the same
/// fault.
public enum USearchStatus: Int32, Sendable {
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
        case .unrecognized: return "unrecognized"
        }
    }
}

/// One failed call: a status every caller can branch on, and the detail the status cannot
/// carry.
public struct USearchError: Error, CustomStringConvertible, Equatable, Sendable {
    /// The failure identity, shared with every other SDK.
    public let status: USearchStatus
    /// The library's static detail, or this binding's own when it rejected the call.
    public let message: String

    public var description: String { "usearch: \(status.name): \(message)" }
}

/// Builds the failure this binding reports for a fault it caught before the boundary.
func fail(_ status: USearchStatus, _ message: String) -> USearchError {
    USearchError(status: status, message: message)
}

/// Turns a populated C error slot into a Swift error. Messages are static library storage and
/// need no deallocation, so the copy is the only cost.
func check(_ raw: usearch_error_t) throws {
    guard raw.status != usearch_success_k else { return }
    let named = USearchStatus(rawValue: Int32(bitPattern: raw.status.rawValue)) ?? .unrecognized
    let status = named == .success ? .unrecognized : named
    throw USearchError(status: status, message: raw.message.map { String(cString: $0) } ?? "")
}

/// Runs one C call against a zeroed error slot and lifts a populated slot into a thrown error.
func throwing<Value>(_ body: (UnsafeMutablePointer<usearch_error_t>) -> Value) throws -> Value {
    var failure = usearch_error_t()
    let value = body(&failure)
    try check(failure)
    return value
}

// endregion: Errors

// region: Keys & Distances

/// A score between two entries, in the metric of the index that produced it.
public typealias USearchDistance = Double

/// A 128-bit entry identity, most significant byte first, rendered as a lowercase dashed UUID.
///
/// Keys need not be unique: several entries may share one, and `USearchIndex.count(key:)`
/// reports how many. Integer identities park in the low eight bytes, so `USearchKey(42)`
/// prints as `00000000-0000-0000-0000-00000000002a`.
public struct USearchKey: Hashable, Comparable, CustomStringConvertible, ExpressibleByIntegerLiteral, Sendable {
    /// The sixteen identity bytes, laid out exactly as the C key so batches reinterpret rather
    /// than copy.
    var storage: usearch_key_t

    /// The all-zero key.
    public static let nilKey = USearchKey()

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
    public init(bytes: [UInt8]) throws {
        guard bytes.count == 16 else { throw fail(.malformedInput, "a key is sixteen bytes, most significant first") }
        storage = usearch_key_t()
        withUnsafeMutableBytes(of: &storage.bytes) { target in
            for position in 0 ..< 16 { target[position] = bytes[position] }
        }
    }

    /// Reads the rendering `description` produces. Dashes may sit anywhere, or nowhere.
    public init(parsing text: String) throws {
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

    public static func == (first: USearchKey, second: USearchKey) -> Bool {
        withUnsafeBytes(of: first.storage.bytes) { left in
            withUnsafeBytes(of: second.storage.bytes) { right in left.elementsEqual(right) }
        }
    }

    public static func < (first: USearchKey, second: USearchKey) -> Bool {
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

// endregion: Keys & Distances

// region: Scalars

/// A stored or exchanged scalar type, named by width and encoding. Sub-byte kinds pack several
/// dimensions per byte, as their `xN` suffixes spell.
public enum USearchScalar: Int32, Sendable {
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
    public var dimensionsPerValue: Int {
        switch self {
        case .i4x2, .u4x2: return 2
        case .u1x8: return 8
        default: return 1
        }
    }

    var raw: usearch_scalar_t { usearch_scalar_t(rawValue: UInt32(bitPattern: rawValue)) }

    /// Lifts a tag the library wrote, refusing to guess at the numbering gaps it reserves.
    static func lifted(_ raw: usearch_scalar_t) throws -> USearchScalar {
        guard let kind = USearchScalar(rawValue: Int32(bitPattern: raw.rawValue)), kind != .unknown else {
            throw fail(.unknownKind, "the checkpoint names an unknown scalar kind")
        }
        return kind
    }
}

/// An element type a matrix view may hold. The conformances below cover every scalar kind,
/// with a bit-holding struct standing in wherever Swift has no builtin.
public protocol USearchScalarValue {
    /// The scalar kind this element type crosses the boundary as.
    static var scalarKind: USearchScalar { get }
}

/// The sixteen bits of a brain floating-point value.
public struct USearchBFloat16: Equatable, Sendable {
    /// The raw bits, unconverted.
    public var bits: UInt16
    public init(bits: UInt16) { self.bits = bits }
}

/// The sixteen bits of an IEEE half-precision value.
public struct USearchFloat16: Equatable, Sendable {
    /// The raw bits, unconverted.
    public var bits: UInt16
    public init(bits: UInt16) { self.bits = bits }
}

/// The eight bits of a five-exponent minifloat.
public struct USearchE5M2: Equatable, Sendable {
    /// The raw bits, unconverted.
    public var bits: UInt8
    public init(bits: UInt8) { self.bits = bits }
}

/// The eight bits of a four-exponent minifloat.
public struct USearchE4M3: Equatable, Sendable {
    /// The raw bits, unconverted.
    public var bits: UInt8
    public init(bits: UInt8) { self.bits = bits }
}

/// The six occupied bits of a three-exponent minifloat.
public struct USearchE3M2: Equatable, Sendable {
    /// The raw bits, unconverted.
    public var bits: UInt8
    public init(bits: UInt8) { self.bits = bits }
}

/// The six occupied bits of a two-exponent minifloat.
public struct USearchE2M3: Equatable, Sendable {
    /// The raw bits, unconverted.
    public var bits: UInt8
    public init(bits: UInt8) { self.bits = bits }
}

/// Two 4-bit signed integers.
public struct USearchI4x2: Equatable, Sendable {
    /// The raw bits, unconverted.
    public var bits: UInt8
    public init(bits: UInt8) { self.bits = bits }
}

/// Two 4-bit unsigned integers.
public struct USearchU4x2: Equatable, Sendable {
    /// The raw bits, unconverted.
    public var bits: UInt8
    public init(bits: UInt8) { self.bits = bits }
}

/// Eight single-bit dimensions.
public struct USearchU1x8: Equatable, Sendable {
    /// The raw bits, unconverted.
    public var bits: UInt8
    public init(bits: UInt8) { self.bits = bits }
}

extension Double: USearchScalarValue {
    public static var scalarKind: USearchScalar { .f64 }
}
extension Float: USearchScalarValue {
    public static var scalarKind: USearchScalar { .f32 }
}
extension USearchBFloat16: USearchScalarValue {
    public static var scalarKind: USearchScalar { .bf16 }
}
extension USearchFloat16: USearchScalarValue {
    public static var scalarKind: USearchScalar { .f16 }
}
extension USearchE5M2: USearchScalarValue {
    public static var scalarKind: USearchScalar { .e5m2 }
}
extension USearchE4M3: USearchScalarValue {
    public static var scalarKind: USearchScalar { .e4m3 }
}
extension USearchE3M2: USearchScalarValue {
    public static var scalarKind: USearchScalar { .e3m2 }
}
extension USearchE2M3: USearchScalarValue {
    public static var scalarKind: USearchScalar { .e2m3 }
}
extension Int8: USearchScalarValue {
    public static var scalarKind: USearchScalar { .i8 }
}
extension USearchI4x2: USearchScalarValue {
    public static var scalarKind: USearchScalar { .i4x2 }
}
extension UInt8: USearchScalarValue {
    public static var scalarKind: USearchScalar { .u8 }
}
extension USearchU4x2: USearchScalarValue {
    public static var scalarKind: USearchScalar { .u4x2 }
}
extension USearchU1x8: USearchScalarValue {
    public static var scalarKind: USearchScalar { .u1x8 }
}

// endregion: Scalars

// region: Matrix Views

/// Bytes spanned by a rectangle of `rows` by `columns` elements, refusing the products that do
/// not fit an `Int`.
///
/// The wanted count of a search has no ceiling, so the outputs it implies are the one place a
/// caller's arithmetic can silently wrap before a single byte is allocated.
func checkedExtent(rows: Int, columns: Int, elementBytes: Int) throws -> Int {
    guard rows >= 0, columns >= 0, elementBytes > 0 else {
        throw fail(.shapeMismatch, "a matrix extent cannot be negative")
    }
    let (cells, cellsOverflowed) = rows.multipliedReportingOverflow(by: columns)
    guard !cellsOverflowed else { throw fail(.overflow, "the matrix has more cells than an index can address") }
    let (bytes, bytesOverflowed) = cells.multipliedReportingOverflow(by: elementBytes)
    guard !bytesOverflowed else { throw fail(.overflow, "the matrix spans more bytes than an index can address") }
    return bytes
}

/// A borrowed run of equal-width rows: a base pointer, a row count, and a byte pitch between
/// consecutive row starts. Every batch input takes this shape, so a memory-mapped dataset
/// reaches the index with no reshape and no copy, and a single vector is a one-row matrix.
///
/// A view borrows and never owns; it stays valid only as long as the storage it was built from.
public struct USearchMatrixView {
    /// The first row's first byte.
    public let base: UnsafeRawPointer
    /// How many rows the view spans.
    public let rows: Int
    /// The logical width of one row.
    public let dimensions: Int
    /// The byte pitch between consecutive row starts.
    public let stride: Int
    /// The scalar type the rows hold.
    public let scalar: USearchScalar

    /// Views a packed run of values as rows of the given width.
    public static func rows<Element: USearchScalarValue>(
        of buffer: UnsafeBufferPointer<Element>,
        dimensions: Int
    ) throws -> USearchMatrixView {
        let shape = try packedShape(count: buffer.count, dimensions: dimensions, of: Element.self)
        guard let base = buffer.baseAddress else { throw fail(.shapeMismatch, "the matrix borrows no storage") }
        return USearchMatrixView(
            base: UnsafeRawPointer(base),
            rows: shape.rows,
            dimensions: dimensions,
            stride: shape.stride,
            scalar: Element.scalarKind
        )
    }

    /// Views a run whose pitch exceeds its row width - the padded layout a memory-mapped
    /// dataset arrives in.
    public static func strided<Element: USearchScalarValue>(
        _ buffer: UnsafeBufferPointer<Element>,
        rows: Int,
        dimensions: Int,
        stride: Int
    ) throws -> USearchMatrixView {
        try validateStride(count: buffer.count, rows: rows, dimensions: dimensions, stride: stride, of: Element.self)
        guard let base = buffer.baseAddress else { throw fail(.shapeMismatch, "the matrix borrows no storage") }
        return USearchMatrixView(
            base: UnsafeRawPointer(base),
            rows: rows,
            dimensions: dimensions,
            stride: stride,
            scalar: Element.scalarKind
        )
    }

    /// Checks the view against a frozen geometry. A width of zero skips the geometry check, for
    /// the handles whose geometry a later load decides.
    func validate(dimensions frozen: Int) throws {
        guard frozen == 0 || dimensions == frozen else {
            throw fail(.shapeMismatch, "the matrix rows are not as wide as the index geometry")
        }
    }
}

/// The written-through twin of `USearchMatrixView`, for the verbs that export rows.
public struct USearchMutableMatrixView {
    /// The first row's first byte.
    public let base: UnsafeMutableRawPointer
    /// How many rows the view spans.
    public let rows: Int
    /// The logical width of one row.
    public let dimensions: Int
    /// The byte pitch between consecutive row starts.
    public let stride: Int
    /// The scalar type the rows hold.
    public let scalar: USearchScalar

    /// Views a packed run of values as rows of the given width.
    public static func rows<Element: USearchScalarValue>(
        of buffer: UnsafeMutableBufferPointer<Element>,
        dimensions: Int
    ) throws -> USearchMutableMatrixView {
        let shape = try packedShape(count: buffer.count, dimensions: dimensions, of: Element.self)
        guard let base = buffer.baseAddress else { throw fail(.shapeMismatch, "the matrix borrows no storage") }
        return USearchMutableMatrixView(
            base: UnsafeMutableRawPointer(base),
            rows: shape.rows,
            dimensions: dimensions,
            stride: shape.stride,
            scalar: Element.scalarKind
        )
    }

    /// Views a run whose pitch exceeds its row width.
    public static func strided<Element: USearchScalarValue>(
        _ buffer: UnsafeMutableBufferPointer<Element>,
        rows: Int,
        dimensions: Int,
        stride: Int
    ) throws -> USearchMutableMatrixView {
        try validateStride(count: buffer.count, rows: rows, dimensions: dimensions, stride: stride, of: Element.self)
        guard let base = buffer.baseAddress else { throw fail(.shapeMismatch, "the matrix borrows no storage") }
        return USearchMutableMatrixView(
            base: UnsafeMutableRawPointer(base),
            rows: rows,
            dimensions: dimensions,
            stride: stride,
            scalar: Element.scalarKind
        )
    }

    /// Checks the view against a frozen geometry, like its read-only twin.
    func validate(dimensions frozen: Int) throws {
        guard frozen == 0 || dimensions == frozen else {
            throw fail(.shapeMismatch, "the matrix rows are not as wide as the index geometry")
        }
    }
}

/// Row count and byte pitch of a packed run, and the divisibility both depend on.
private func packedShape<Element: USearchScalarValue>(
    count: Int,
    dimensions: Int,
    of element: Element.Type
) throws -> (rows: Int, stride: Int) {
    let valuesPerRow = try valuesPerRow(dimensions: dimensions, of: element)
    guard count % valuesPerRow == 0 else {
        throw fail(.shapeMismatch, "the values do not divide into whole rows of the given width")
    }
    return (count / valuesPerRow, valuesPerRow * MemoryLayout<Element>.stride)
}

/// Rejects a stride that cannot hold a row, and rows that overrun the borrowed storage.
private func validateStride<Element: USearchScalarValue>(
    count: Int,
    rows: Int,
    dimensions: Int,
    stride: Int,
    of element: Element.Type
) throws {
    let rowBytes = try valuesPerRow(dimensions: dimensions, of: element) * MemoryLayout<Element>.stride
    guard stride >= rowBytes else { throw fail(.unsupportedLayout, "the stride cannot hold a whole row") }
    guard rows >= 0 else { throw fail(.shapeMismatch, "a matrix cannot hold a negative row count") }
    guard rows != 0 else { return }
    let span = try checkedExtent(rows: rows - 1, columns: stride, elementBytes: 1) + rowBytes
    guard span <= count * MemoryLayout<Element>.stride else {
        throw fail(.shapeMismatch, "the rows overrun the borrowed storage")
    }
}

/// How many stored values one row of the given logical width occupies.
private func valuesPerRow<Element: USearchScalarValue>(dimensions: Int, of element: Element.Type) throws -> Int {
    guard dimensions > 0 else { throw fail(.shapeMismatch, "a matrix row cannot be zero dimensions wide") }
    let perValue = Element.scalarKind.dimensionsPerValue
    guard dimensions % perValue == 0 else {
        throw fail(.shapeMismatch, "the dimensions do not fill whole \(Element.scalarKind.name) values")
    }
    return dimensions / perValue
}

// endregion: Matrix Views

// region: Residency

/// Whether the RAM arenas demand a provisioned huge-page pool.
public enum USearchPageBacking: Sendable {
    /// Ordinary pages, whatever the kernel provides.
    case basePages
    /// Fails loudly when the pool is missing, rather than degrading silently.
    case requireHugePages
}

/// Whether a VRAM index may spill past the device's memory.
public enum USearchAddressing: Sendable {
    /// Everything stays in device memory.
    case deviceOnly
    /// Cold shards spill into host memory through the pager, trading traversal latency for
    /// datasets past the device.
    case unifiedAddressing
}

/// How an NVMe index reaches the block device.
public enum USearchIOMode: Sendable {
    /// The O_DIRECT-class page-cache bypass - the default and the point.
    case directIO
    /// Ordinary buffered reads through the page cache.
    case pageCachedIO
}

/// Where an index's payload lives and which engine traverses it. The case picks the arm of the
/// C options union, and decides what loading a checkpoint means for the handle.
public enum USearchResidency: Sendable {
    /// A volatile-memory index traversed by CPU threads - the concurrent baseline. A thread
    /// count of zero takes 7/8 of the logical cores, since full occupancy starves the ordered
    /// publication path.
    case ram(threads: Int, pages: USearchPageBacking)
    /// An accelerator-memory index traversed by GPU warps. A negative device ordinal picks the
    /// first free device.
    case vram(device: Int32, addressing: USearchAddressing)
    /// A persisted index traversed through pluggable I/O engines, sized past RAM. Zeroed sector
    /// bytes probe the device, and a zeroed queue depth derives from it; the handle is
    /// read-only, so every mutating verb fails on it with `readOnly`.
    case nvme(sectorBytes: UInt32, queueDepth: UInt32, io: USearchIOMode)

    /// A RAM index with a derived thread count and ordinary pages.
    public static let `default` = USearchResidency.ram(threads: 0, pages: .basePages)

    /// Stamps the residency tag and its arm of the options union.
    func write(into options: inout usearch_dense_options_t) {
        switch self {
        case .ram(let threads, let pages):
            options.residency = usearch_residency_ram_k
            options.ram = usearch_ram_options_t(threads: threads, huge_pages: pages == .requireHugePages)
        case .vram(let device, let addressing):
            options.residency = usearch_residency_vram_k
            options.vram = usearch_vram_options_t(device: device, unified_addressing: addressing == .unifiedAddressing)
        case .nvme(let sectorBytes, let queueDepth, let io):
            options.residency = usearch_residency_nvme_k
            options.nvme = usearch_nvme_options_t(
                sector_bytes: sectorBytes,
                queue_depth: queueDepth,
                direct_io: io == .directIO
            )
        }
    }

    /// Reconstructs a residency from the tag and the arm it selects.
    static func read(from options: usearch_dense_options_t) throws -> USearchResidency {
        switch options.residency {
        case usearch_residency_ram_k:
            return .ram(
                threads: options.ram.threads,
                pages: options.ram.huge_pages ? .requireHugePages : .basePages
            )
        case usearch_residency_vram_k:
            return .vram(
                device: options.vram.device,
                addressing: options.vram.unified_addressing ? .unifiedAddressing : .deviceOnly
            )
        case usearch_residency_nvme_k:
            return .nvme(
                sectorBytes: options.nvme.sector_bytes,
                queueDepth: options.nvme.queue_depth,
                io: options.nvme.direct_io ? .directIO : .pageCachedIO
            )
        default:
            throw fail(.unknownKind, "the checkpoint names an unknown residency")
        }
    }
}

// endregion: Residency

// region: Introspection

/// The version of the linked library.
public func usearchVersion() -> String { String(cString: usearch_version()) }

/// The hardware capabilities the linked binary was compiled with, comma separated.
public func usearchCapabilitiesCompiled() -> String { String(cString: usearch_capabilities_compiled()) }

/// The hardware capabilities the current machine offers at runtime, comma separated.
public func usearchCapabilitiesAvailable() -> String { String(cString: usearch_capabilities_available()) }

// endregion: Introspection
