//
//  USearchIndex.swift
//
//
//  Created by Dan Palmer on 9/2/25.
//

import Foundation
import USearchC

public enum USearchScalar: UInt {
    case f32
    case f16
    case f64
    case i8
    case b1
}

public enum USearchMetric: UInt {
    case ip
    case cos
    case l2sq
    case pearson
    case haversine
    case divergence
    case jaccard
    case hamming
    case tanimoto
    case sorensen
}

public typealias USearchKey = UInt64
public typealias USearchFilterFn = (USearchKey) -> Bool

extension USearchMetric {
    func toNative() -> usearch_metric_kind_t {
        switch self {
        case .ip:
            return usearch_metric_ip_k
        case .cos:
            return usearch_metric_cos_k
        case .l2sq:
            return usearch_metric_l2sq_k
        case .hamming:
            return usearch_metric_hamming_k
        case .haversine:
            return usearch_metric_haversine_k
        case .divergence:
            return usearch_metric_divergence_k
        case .jaccard:
            return usearch_metric_jaccard_k
        case .pearson:
            return usearch_metric_pearson_k
        case .sorensen:
            return usearch_metric_sorensen_k
        case .tanimoto:
            return usearch_metric_tanimoto_k
        }
    }
}

extension USearchScalar {
    func toNative() -> usearch_scalar_kind_t {
        switch self {
        case .i8:
            return usearch_scalar_i8_k
        case .f16:
            return usearch_scalar_f16_k
        case .b1:
            return usearch_scalar_b1_k
        case .f32:
            return usearch_scalar_f32_k
        case .f64:
            return usearch_scalar_f64_k
        }
    }
}

public enum USearchError: Error {
    case outOfMemory
    case unknownScalarKind
    case unknownMetricKind
    case fileReadError
    case fileTypeError
    case keyLookupsDisabled
    case keyMissing
    case serializationError
    case deserializationError
    case immutableError
    case keyCountError
    case renameCollisionError
    case indexTooSmallToClusterError
    case duplicateKeysError
    case reservationError

    case pathNotUTF8Encodable

    case unknownError(String)

    static func fromErrorString(_ errorString: String) -> USearchError {
        switch errorString {
        case "Out of memory!",
            "Out of memory",
            "Out of memory when preparing contexts!",
            "Out of memory, allocating a temporary buffer for batch results",
            "Failed to allocate memory for the index!",
            "Failed to allocate memory for the casts",
            "Failed to reserve memory for the index",
            "Failed to allocate memory for the available threads!",
            "Failed to allocate memory for the index",
            "Can't allocate memory for a free-list",
            "Failed to allocate memory for the casts!":
            return .outOfMemory
        case "Unknown scalar kind!":
            return .unknownScalarKind
        case "Unknown metric kind!":
            return .unknownMetricKind
        case "End of file reached!", "Can't infer file size":
            return .fileReadError
        case "Not a dense USearch index!":
            return .fileTypeError
        case "Key lookups are disabled!":
            return .keyLookupsDisabled
        case "Key missing!":
            return .keyMissing
        case "Failed to serialize into stream":
            return .serializationError
        case "Failed to read 32-bit dimensions of the matrix",
            "Failed to read 64-bit dimensions of the matrix",
            "Failed to allocate memory to address vectors",
            "Failed to read vectors",
            "Failed to read the index ",  // space left intentionally blank
            "Magic header mismatch - the file isn't an index",
            "File format may be different, please rebuild",
            "Key type doesn't match, consider rebuilding",
            "Slot type doesn't match, consider rebuilding",
            "Index size and the number of vectors doesn't match",
            "File is corrupted and lacks matrix dimensions",
            "File is corrupted and lacks a header":
            return .deserializationError
        case "Can't add to an immutable index",
            "Can't remove from an immutable index":
            return .immutableError
        case "Free keys count mismatch":
            return .keyCountError
        case "Renaming impossible, the key is already in use":
            return .renameCollisionError
        case "Index too small to cluster!":
            return .indexTooSmallToClusterError
        case "Duplicate keys not allowed in high-level wrappers":
            return .duplicateKeysError
        case "Reserve capacity ahead of insertions!":
            return .reservationError
        default:
            return .unknownError(errorString)
        }
    }
}

@available(iOS 13.0, macOS 11.0, tvOS 13.0, watchOS 6.0, visionOS 1.0, *)
public class USearchIndex: NSObject {
    private var nativeIndex: usearch_index_t

    private init(native: consuming usearch_index_t) throws {
        nativeIndex = native
        super.init()
    }

    var dimensions: UInt32 {
        get throws {
            return try UInt32(throwing { usearch_dimensions(nativeIndex, $0) })
        }
    }

    var connectivity: UInt32 {
        get throws {
            return try UInt32(throwing { usearch_connectivity(nativeIndex, $0) })
        }
    }

    var expansionAdd: UInt32 {
        get throws {
            return try UInt32(throwing { usearch_expansion_add(nativeIndex, $0) })
        }
    }

    var expansionSearch: UInt32 {
        get throws {
            return try UInt32(throwing { usearch_expansion_search(nativeIndex, $0) })
        }
    }

    var length: UInt32 {
        get throws {
            return try UInt32(throwing { usearch_size(nativeIndex, $0) })
        }
    }

    var capacity: UInt32 {
        get throws {
            return try UInt32(throwing { usearch_capacity(nativeIndex, $0) })
        }
    }

    /**
     * @brief Initializes a new index.
     * @param metric The distance function to compare the dis-similarity of vectors.
     * @param dimensions The number of dimensions planned for this index.
     * @param connectivity Number of connections per node in the proximity graph.
     * Higher connectivity improves quantization, increases memory usage, and reduces construction speed.
     * @param quantization Quantization of internal vector representations. Lower quantization means higher speed.
     */
    public static func make(
        metric: USearchMetric,
        dimensions: UInt32,
        connectivity: UInt32,
        quantization: USearchScalar
    ) throws -> USearchIndex {
        return try make(
            metric: metric,
            dimensions: dimensions,
            connectivity: connectivity,
            quantization: quantization,
            multi: false
        )
    }

    /**
     * @brief Initializes a new index.
     * @param metric The distance function to compare the dis-similarity of vectors.
     * @param dimensions The number of dimensions planned for this index.
     * @param connectivity Number of connections per node in the proximity graph.
     * Higher connectivity improves quantization, increases memory usage, and reduces construction speed.
     * @param quantization Quantization of internal vector representations. Lower quantization means higher speed.
     * @param multi Enables indexing multiple vectors per key when true.
     */
    public static func make(
        metric: USearchMetric,
        dimensions: UInt32,
        connectivity: UInt32,
        quantization: USearchScalar,
        multi: Bool
    ) throws -> USearchIndex {
        let options = usearch_init_options_t(
            metric_kind: metric.toNative(),
            metric: nil,
            quantization: quantization.toNative(),
            dimensions: Int(dimensions),
            connectivity: Int(connectivity),
            expansion_add: 0,
            expansion_search: 0,
            multi: multi
        )

        let optionsPtr = pointer(options)
        let index = try throwing { usearch_init(optionsPtr, $0) }
        guard let index else {
            throw USearchError.unknownError("No index created, no error returned.")
        }

        return try USearchIndex(native: index)
    }

    public var isEmpty: Bool {
        get throws {
            return (try throwing { usearch_size(nativeIndex, $0) } != 0)
        }
    }

    override public var description: String {
        do {
            return try
                "USearchIndex(dimensions: \(dimensions), connectivity: \(connectivity), length: \(length), capacity: \(capacity), isEmpty: \(isEmpty))"
        }
        catch {
            return "USearchIndex(error: \(error))"
        }
    }

    /**
     * @brief Pre-allocates space in the index for the given number of vectors.
     */
    public func reserve(_ count: UInt32) throws {
        try throwing { usearch_reserve(nativeIndex, Int(count), $0) }
    }

    /**
     * @brief Adds a labeled vector to the index.
     * @param vector Single-precision vector.
     */
    public func addSingle(key: USearchKey, vector: UnsafePointer<Float32>) throws {
        try throwing { usearch_add(nativeIndex, key, vector, USearchScalar.f32.toNative(), $0) }
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Single-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    public func searchSingle(
        vector: UnsafePointer<Float32>,
        count: UInt32,
        keys: UnsafeMutablePointer<USearchKey>?,
        distances: UnsafeMutablePointer<Float32>?
    ) throws -> UInt32 {
        let found = try throwing {
            usearch_search(nativeIndex, vector, USearchScalar.f32.toNative(), Int(count), keys, distances, $0)
        }
        return UInt32(found)
    }

    /**
    * @brief Retrieves a labeled single-precision vector from the index.
    * @param vector A buffer to store the vector.
    * @param count For multi-indexes, the number of vectors to retrieve.
    * @return Number of vectors exported to `vector`.
    */
    public func getSingle(key: USearchKey, vector: UnsafeMutablePointer<Float32>, count: UInt32) throws -> UInt32 {
        let result = try throwing {
            usearch_get(nativeIndex, key, Int(count), vector, USearchScalar.f32.toNative(), $0)
        }
        return UInt32(result)
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Single-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param filter Closure called for each key, determining whether to include or
     *               skip key in the results.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    public func filteredSearchSingle(
        vector: UnsafePointer<Float32>,
        count: UInt32,
        filter: @escaping USearchFilterFn,
        keys: UnsafeMutablePointer<USearchKey>?,
        distances: UnsafeMutablePointer<Float32>?
    ) throws -> UInt32 {
        return try filteredSearchGeneric(
            nativeIndex,
            vector: vector,
            count: count,
            quantization: .f32,
            filter: filter,
            keys: keys,
            distances: distances
        )
    }

    /**
     * @brief Adds a labeled vector to the index.
     * @param vector Double-precision vector.
     */
    public func addDouble(key: USearchKey, vector: UnsafePointer<Float64>) throws {
        try throwing { usearch_add(nativeIndex, key, vector, USearchScalar.f64.toNative(), $0) }
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Double-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    public func searchDouble(
        vector: UnsafePointer<Float64>,
        count: UInt32,
        keys: UnsafeMutablePointer<USearchKey>?,
        distances: UnsafeMutablePointer<Float32>?
    ) throws -> UInt32 {
        let found = try throwing {
            usearch_search(nativeIndex, vector, USearchScalar.f64.toNative(), Int(count), keys, distances, $0)
        }
        return UInt32(found)
    }

    /**
    * @brief Retrieves a labeled double-precision vector from the index.
    * @param vector A buffer to store the vector.
    * @param count For multi-indexes, the number of vectors to retrieve.
    * @return Number of vectors exported to `vector`.
    */
    public func getDouble(key: USearchKey, vector: UnsafeMutablePointer<Float64>, count: UInt32) throws -> UInt32 {
        let result = try throwing {
            usearch_get(nativeIndex, key, Int(count), vector, USearchScalar.f64.toNative(), $0)
        }
        return UInt32(result)
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Double-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param filter Closure called for each key, determining whether to include or
     *               skip key in the results.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    public func filteredSearchDouble(
        vector: UnsafePointer<Float64>,
        count: UInt32,
        filter: @escaping USearchFilterFn,
        keys: UnsafeMutablePointer<USearchKey>?,
        distances: UnsafeMutablePointer<Float32>?
    ) throws -> UInt32 {
        return try filteredSearchGeneric(
            nativeIndex,
            vector: vector,
            count: count,
            quantization: .f64,
            filter: filter,
            keys: keys,
            distances: distances
        )
    }

    /**
     * @brief Adds a labeled vector to the index.
     * @param vector Half-precision vector.
     */
    @available(iOS 14.0, macOS 11.0, tvOS 14.0, watchOS 7.0, visionOS 1.0, *)
    public func addHalf(key: USearchKey, vector: UnsafePointer<Float16>) throws {
        try throwing { usearch_add(nativeIndex, key, vector, USearchScalar.f16.toNative(), $0) }
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Half-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    @available(iOS 14.0, macOS 11.0, tvOS 14.0, watchOS 7.0, visionOS 1.0, *)
    public func searchHalf(
        vector: UnsafePointer<Float16>,
        count: UInt32,
        keys: UnsafeMutablePointer<USearchKey>?,
        distances: UnsafeMutablePointer<Float32>?
    ) throws -> UInt32 {
        let found = try throwing {
            usearch_search(nativeIndex, vector, USearchScalar.f16.toNative(), Int(count), keys, distances, $0)
        }
        return UInt32(found)
    }

    /**
    * @brief Retrieves a labeled half-precision vector from the index.
    * @param vector A buffer to store the vector.
    * @param count For multi-indexes, the number of vectors to retrieve.
    * @return Number of vectors exported to `vector`.
    */
    @available(iOS 14.0, macOS 11.0, tvOS 14.0, watchOS 7.0, visionOS 1.0, *)
    public func getHalf(key: USearchKey, vector: UnsafeMutablePointer<Float16>, count: UInt32) throws -> UInt32 {
        let result = try throwing {
            usearch_get(nativeIndex, key, Int(count), vector, USearchScalar.f16.toNative(), $0)
        }
        return UInt32(result)
    }

    /**
     * @brief Approximate nearest neighbors search.
     * @param vector Double-precision query vector.
     * @param count Upper limit on the number of matches to retrieve.
     * @param filter Closure called for each key, determining whether to include or
     *               skip key in the results.
     * @param keys Optional output buffer for keys of approximate neighbors.
     * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
     * @return Number of matches exported to `keys` and `distances`.
     */
    @available(iOS 14.0, macOS 11.0, tvOS 14.0, watchOS 7.0, visionOS 1.0, *)
    public func filteredSearchHalf(
        vector: UnsafePointer<Float16>,
        count: UInt32,
        filter: @escaping USearchFilterFn,
        keys: UnsafeMutablePointer<USearchKey>?,
        distances: UnsafeMutablePointer<Float32>?
    ) throws -> UInt32 {
        return try filteredSearchGeneric(
            nativeIndex,
            vector: vector,
            count: count,
            quantization: .f16,
            filter: filter,
            keys: keys,
            distances: distances
        )
    }

    public func contains(key: USearchKey) throws -> Bool {
        return try throwing { usearch_contains(nativeIndex, key, $0) }
    }

    public func count(key: USearchKey) throws -> UInt32 {
        return UInt32(try throwing { usearch_count(nativeIndex, key, $0) })
    }

    public func remove(key: USearchKey) throws -> UInt32 {
        return try UInt32(throwing { usearch_remove(nativeIndex, key, $0) })
    }

    public func rename(from key: USearchKey, to newKey: USearchKey) throws -> UInt32 {
        return try UInt32(throwing { usearch_rename(nativeIndex, key, newKey, $0) })
    }

    /**
     * @brief Saves pre-constructed index to disk.
     */
    public func save(path: String) throws {
        guard let cPath = path.cString(using: .utf8) else {
            throw USearchError.pathNotUTF8Encodable
        }
        try throwing { usearch_save(nativeIndex, cPath, $0) }
    }

    /**
     * @brief Loads a pre-constructed index from index.
     */
    public func load(path: String) throws {
        guard let cPath = path.cString(using: .utf8) else {
            throw USearchError.pathNotUTF8Encodable
        }
        try throwing { usearch_load(nativeIndex, cPath, $0) }
    }

    /**
     * @brief Views a pre-constructed index from disk without loading it into RAM.
     *        Allows working with larger-than memory indexes and saving scarce
     *        memory on device in read-only workloads.
     */
    public func view(path: String) throws {
        guard let cPath = path.cString(using: .utf8) else {
            throw USearchError.pathNotUTF8Encodable
        }
        try throwing { usearch_view(nativeIndex, cPath, $0) }
    }

    /**
     * @brief Removes all the data from index, while preserving the settings.
     */
    public func clear() throws {
        try throwing { usearch_clear(nativeIndex, $0) }
    }
}
