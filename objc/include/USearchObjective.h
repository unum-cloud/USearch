#pragma once

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, USearchScalar) {
    USearchScalarF64,
    USearchScalarF32,
    USearchScalarBF16,
    USearchScalarF16,
    USearchScalarE5M2,
    USearchScalarE4M3,
    USearchScalarI8,
    USearchScalarB1,
};

typedef NS_ENUM(NSUInteger, USearchMetric) {
    USearchMetricUnknown,
    USearchMetricIP,
    USearchMetricCos,
    USearchMetricL2sq,
    USearchMetricPearson,
    USearchMetricHaversine,
    USearchMetricDivergence,
    USearchMetricJaccard,
    USearchMetricHamming,
    USearchMetricTanimoto,
    USearchMetricSorensen
};

typedef UInt64 USearchKey;

typedef bool (^USearchFilterFn)(USearchKey key);

API_AVAILABLE(ios(13.0), macos(10.15), tvos(13.0), watchos(6.0))
@interface USearchIndex : NSObject

@property (readonly) UInt32 dimensions;
@property (readonly) UInt32 connectivity;
@property (readonly) UInt32 expansionAdd;
@property (readonly) UInt32 expansionSearch;

@property (readonly) UInt32 length;
@property (readonly) UInt32 capacity;
@property (readonly) Boolean isEmpty;

- (instancetype)init NS_UNAVAILABLE;

/**
 * @brief Initializes a new index.
 * @param metric The distance function to compare the dis-similarity of vectors.
 * @param dimensions The number of dimensions planned for this index.
 * @param connectivity Number of connections per node in the proximity graph.
 * Higher connectivity improves quantization, increases memory usage, and reduces construction speed.
 * @param quantization Quantization of internal vector representations. Lower quantization means higher speed.
 */
+ (instancetype)make:(USearchMetric)metric dimensions:(UInt32)dimensions connectivity:(UInt32)connectivity quantization:(USearchScalar)quantization NS_SWIFT_NAME(make(metric:dimensions:connectivity:quantization:));

/**
 * @brief Initializes a new index.
 * @param metric The distance function to compare the dis-similarity of vectors.
 * @param dimensions The number of dimensions planned for this index.
 * @param connectivity Number of connections per node in the proximity graph.
 * Higher connectivity improves quantization, increases memory usage, and reduces construction speed.
 * @param quantization Quantization of internal vector representations. Lower quantization means higher speed.
 * @param multi Enables indexing multiple vectors per key when true.
 */
+ (instancetype)make:(USearchMetric)metricKind
          dimensions:(UInt32)dimensions
        connectivity:(UInt32)connectivity
        quantization:(USearchScalar)quantization
               multi:(BOOL)multi NS_SWIFT_NAME(make(metric:dimensions:connectivity:quantization:multi:));


/**
 * @brief Pre-allocates space in the index for the given number of vectors.
 */
- (void)reserve:(UInt32)count NS_SWIFT_NAME(reserve(_:));

/**
 * @brief Adds a labeled vector to the index.
 * @param vector Single-precision vector.
 */
- (void)addSingle:(USearchKey)key
           vector:(Float32 const *_Nonnull)vector NS_SWIFT_NAME(addSingle(key:vector:));

/**
 * @brief Approximate nearest neighbors search.
 * @param vector Single-precision query vector.
 * @param count Upper limit on the number of matches to retrieve.
 * @param keys Optional output buffer for keys of approximate neighbors.
 * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
 * @return Number of matches exported to `keys` and `distances`.
 */
- (UInt32)searchSingle:(Float32 const *_Nonnull)vector
                 count:(UInt32)count
                  keys:(USearchKey *_Nullable)keys
             distances:(Float32 *_Nullable)distances NS_SWIFT_NAME(searchSingle(vector:count:keys:distances:));

/**
* @brief Retrieves a labeled single-precision vector from the index.
* @param vector A buffer to store the vector.
* @param count For multi-indexes, the number of vectors to retrieve.
* @return Number of vectors exported to `vector`.
*/
- (UInt32)getSingle:(USearchKey)key
    vector:(void *_Nonnull)vector
    count:(UInt32)count NS_SWIFT_NAME(getSingle(key:vector:count:));

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
- (UInt32)filteredSearchSingle:(Float32 const *_Nonnull)vector
                 count:(UInt32)count
                filter:(USearchFilterFn)filter
                  keys:(USearchKey *_Nullable)keys
             distances:(Float32 *_Nullable)distances NS_SWIFT_NAME(filteredSearchSingle(vector:count:filter:keys:distances:));

/**
 * @brief Adds a labeled vector to the index.
 * @param vector Double-precision vector.
 */
- (void)addDouble:(USearchKey)key
           vector:(Float64 const *_Nonnull)vector NS_SWIFT_NAME(addDouble(key:vector:));

/**
 * @brief Approximate nearest neighbors search.
 * @param vector Double-precision query vector.
 * @param count Upper limit on the number of matches to retrieve.
 * @param keys Optional output buffer for keys of approximate neighbors.
 * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
 * @return Number of matches exported to `keys` and `distances`.
 */
- (UInt32)searchDouble:(Float64 const *_Nonnull)vector
                 count:(UInt32)count
                  keys:(USearchKey *_Nullable)keys
             distances:(Float32 *_Nullable)distances NS_SWIFT_NAME(searchDouble(vector:count:keys:distances:));

/**
* @brief Retrieves a labeled double-precision vector from the index.
* @param vector A buffer to store the vector.
* @param count For multi-indexes, the number of vectors to retrieve.
* @return Number of vectors exported to `vector`.
*/
- (UInt32)getDouble:(USearchKey)key
    vector:(void *_Nonnull)vector
    count:(UInt32)count NS_SWIFT_NAME(getDouble(key:vector:count:));

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
- (UInt32)filteredSearchDouble:(Float64 const *_Nonnull)vector
                 count:(UInt32)wanted
                filter:(USearchFilterFn)predicate
                  keys:(USearchKey *_Nullable)keys
             distances:(Float32 *_Nullable)distances NS_SWIFT_NAME(filteredSearchDouble(vector:count:filter:keys:distances:));
/**
 * @brief Adds a labeled vector to the index.
 * @param vector Half-precision vector.
 */
- (void)addHalf:(USearchKey)key
         vector:(void const *_Nonnull)vector NS_SWIFT_NAME(addHalf(key:vector:));

/**
 * @brief Approximate nearest neighbors search.
 * @param vector Half-precision query vector.
 * @param count Upper limit on the number of matches to retrieve.
 * @param keys Optional output buffer for keys of approximate neighbors.
 * @param distances Optional output buffer for (increasing) distances to approximate neighbors.
 * @return Number of matches exported to `keys` and `distances`.
 */
- (UInt32)searchHalf:(void const *_Nonnull)vector
               count:(UInt32)count
                keys:(USearchKey *_Nullable)keys
           distances:(Float32 *_Nullable)distances NS_SWIFT_NAME(searchHalf(vector:count:keys:distances:));

/**
* @brief Retrieves a labeled half-precision vector from the index.
* @param vector A buffer to store the vector.
* @param count For multi-indexes, the number of vectors to retrieve.
* @return Number of vectors exported to `vector`.
*/
- (UInt32)getHalf:(USearchKey)key
        vector:(void *_Nonnull)vector
        count:(UInt32)count NS_SWIFT_NAME(getHalf(key:vector:count:));

- (Boolean)contains:(USearchKey)key NS_SWIFT_NAME(contains(key:));

- (UInt32)count:(USearchKey)key NS_SWIFT_NAME(count(key:));

- (void)remove:(USearchKey)key NS_SWIFT_NAME(remove(key:));

- (void)rename:(USearchKey)key to:(USearchKey)key NS_SWIFT_NAME(rename(from:to:));


/**
 * @brief Saves pre-constructed index to disk.
 */
- (void)save:(NSString *)path NS_SWIFT_NAME(save(path:));

/**
 * @brief Loads a pre-constructed index from index.
 */
- (void)load:(NSString *)path NS_SWIFT_NAME(load(path:));

/**
 * @brief Views a pre-constructed index from disk without loading it into RAM.
 *        Allows working with larger-than memory indexes and saving scarce
 *        memory on device in read-only workloads.
 */
- (void)view:(NSString *)path NS_SWIFT_NAME(view(path:));

/**
 * @brief Removes all the data from index, while preserving the settings.
 */
- (void)clear NS_SWIFT_NAME(clear());

@end

NS_ASSUME_NONNULL_END
