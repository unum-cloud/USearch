#ifndef UNUM_USEARCH_H
#define UNUM_USEARCH_H

#include <stdbool.h> // `bool`
#include <stddef.h>  // `size_t`
#include <stdint.h>  // `uint64_t`

#ifdef __cplusplus
extern "C" {
#endif

#ifndef USEARCH_EXPORT
#if defined(_WIN32) && !defined(__MINGW32__)
#define USEARCH_EXPORT __declspec(dllexport)
#else
#define USEARCH_EXPORT
#endif
#endif

USEARCH_EXPORT typedef void* usearch_index_t;
USEARCH_EXPORT typedef uint64_t usearch_key_t;
USEARCH_EXPORT typedef float usearch_distance_t;

/**
 *  @brief  Pointer to a null-terminated error message.
 *          Returned error messages @b don't need to be deallocated.
 */
USEARCH_EXPORT typedef char const* usearch_error_t;

/**
 *  @brief  Type-punned callback for "metrics" or "distance functions",
 *          that accepts pointers to two vectors and measures their @b dis-similarity.
 */
USEARCH_EXPORT typedef usearch_distance_t (*usearch_metric_t)(void const*, void const*);

/**
 *  @brief  Enumerator for the most common kinds of `usearch_metric_t`.
 *          Those are supported out of the box, with SIMD-optimizations for most common hardware.
 */
USEARCH_EXPORT typedef enum usearch_metric_kind_t {
    usearch_metric_unknown_k = 0,
    usearch_metric_cos_k = 1,
    usearch_metric_ip_k = 2,
    usearch_metric_l2sq_k = 3,
    usearch_metric_haversine_k = 4,
    usearch_metric_divergence_k = 5,
    usearch_metric_pearson_k = 6,
    usearch_metric_jaccard_k = 7,
    usearch_metric_hamming_k = 8,
    usearch_metric_tanimoto_k = 9,
    usearch_metric_sorensen_k = 10,
} usearch_metric_kind_t;

USEARCH_EXPORT typedef enum usearch_scalar_kind_t {
    usearch_scalar_unknown_k = 0,
    usearch_scalar_f32_k = 1,
    usearch_scalar_f64_k = 2,
    usearch_scalar_f16_k = 3,
    usearch_scalar_i8_k = 4,
    usearch_scalar_b1_k = 5,
    usearch_scalar_bf16_k = 6,
} usearch_scalar_kind_t;

USEARCH_EXPORT typedef struct usearch_init_options_t {
    /**
     *  @brief The metric kind used for distance calculation between vectors.
     */
    usearch_metric_kind_t metric_kind;
    /**
     *  @brief The @b optional custom distance metric function used for distance calculation between vectors.
     *  If the `metric_kind` is set to `usearch_metric_unknown_k`, this function pointer mustn't be `NULL`.
     */
    usearch_metric_t metric;
    /**
     *  @brief The scalar kind used for quantization of vector data during indexing.
     *  In most cases, on modern hardware, it's recommended to use half-precision floating-point numbers.
     *  When quantization is enabled, the "get"-like functions won't be able to recover the original data,
     *  so you may want to replicate the original vectors elsewhere.
     *
     *  Quantizing to integers is also possible, but it's important to note that it's only valid for cosine-like
     *  metrics. As part of the quantization process, the vectors are normalized to unit length and later scaled
     *  to @b [-127,127] range to occupy the full 8-bit range.
     *
     *  Quantizing to 1-bit booleans is also possible, but it's only valid for binary metrics like Jaccard, Hamming,
     *  etc. As part of the quantization process, the scalar components greater than zero are set to `true`, and the
     *  rest to `false`.
     */
    usearch_scalar_kind_t quantization;
    /**
     *  @brief The number of dimensions in the vectors to be indexed.
     *  Must be defined for most metrics, but can be avoided for `usearch_metric_haversine_k`.
     */
    size_t dimensions;
    /**
     *  @brief The @b optional connectivity parameter that limits connections-per-node in graph.
     */
    size_t connectivity;
    /**
     *  @brief The @b optional expansion factor used for index construction when adding vectors.
     */
    size_t expansion_add;
    /**
     *  @brief The @b optional expansion factor used for index construction during search operations.
     */
    size_t expansion_search;
    /**
     *  @brief When set allows multiple vectors to map to the same key.
     */
    bool multi;
} usearch_init_options_t;

/**
 *  @brief Retrieves the version of the library.
 *  @return The version of the library.
 */
USEARCH_EXPORT char const* usearch_version(void);

/**
 *  @brief Initializes a new instance of the index.
 *  @param options Pointer to the `usearch_init_options_t` structure containing initialization options.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return A handle to the initialized USearch index, or `NULL` on failure.
 */
USEARCH_EXPORT usearch_index_t usearch_init(usearch_init_options_t* options, usearch_error_t* error);

/**
 *  @brief Frees the resources associated with the index.
 *  @param[inout] index The handle to the USearch index to be freed.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_free(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Reports the memory usage of the index.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of bytes used by the index.
 */
USEARCH_EXPORT size_t usearch_memory_usage(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Reports the SIMD capabilities used by the index on the current CPU.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return The codename of the SIMD instruction set used by the index.
 */
USEARCH_EXPORT char const* usearch_hardware_acceleration(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Reports expected file size after serialization.
 *  @param[in] index The handle to the USearch index to be serialized.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT size_t usearch_serialized_length(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Saves the index to a file.
 *  @param[in] index The handle to the USearch index to be serialized.
 *  @param[in] path The file path where the index will be saved.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_save(usearch_index_t index, char const* path, usearch_error_t* error);

/**
 *  @brief Loads the index from a file.
 *  @param[inout] index The handle to the USearch index to be populated from path.
 *  @param[in] path The file path from where the index will be loaded.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_load(usearch_index_t index, char const* path, usearch_error_t* error);

/**
 *  @brief Creates a view of the index from a file without copying it into memory.
 *  @param[inout] index The handle to the USearch index to be populated with a file view.
 *  @param[in] path The file path from where the view will be created.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_view(usearch_index_t index, char const* path, usearch_error_t* error);

/**
 *  @brief Loads index metadata from a file.
 *  @param[in] path The file path from where the index will be loaded.
 *  @param[out] options Pointer to the `usearch_init_options_t` structure to be populated.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_metadata(char const* path, usearch_init_options_t* options, usearch_error_t* error);

/**
 *  @brief Saves the index to an in-memory buffer.
 *  @param[in] index The handle to the USearch index to be serialized.
 *  @param[in] buffer The in-memory continuous buffer where the index will be saved.
 *  @param[in] length The length of the buffer in bytes.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_save_buffer(usearch_index_t index, void* buffer, size_t length, usearch_error_t* error);

/**
 *  @brief Loads the index from an in-memory buffer.
 *  @param[inout] index The handle to the USearch index to be populated from buffer.
 *  @param[in] buffer The in-memory continuous buffer from where the index will be loaded.
 *  @param[in] length The length of the buffer in bytes.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_load_buffer(usearch_index_t index, void const* buffer, size_t length,
                                        usearch_error_t* error);

/**
 *  @brief Creates a view of the index from an in-memory buffer without copying it into memory.
 *  @param[inout] index The handle to the USearch index to be populated with a buffer view.
 *  @param[in] buffer The in-memory continuous buffer from where the view will be created.
 *  @param[in] length The length of the buffer in bytes.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_view_buffer(usearch_index_t index, void const* buffer, size_t length,
                                        usearch_error_t* error);

/**
 *  @brief Loads index metadata from an in-memory buffer.
 *  @param[in] buffer The in-memory continuous buffer from where the view will be created.
 *  @param[out] options Pointer to the `usearch_init_options_t` structure to be populated.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_metadata_buffer(void const* buffer, size_t length, usearch_init_options_t* options,
                                            usearch_error_t* error);

/**
 *  @brief Reports the current size (number of vectors) of the index.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT size_t usearch_size(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Reports the current capacity (number of vectors) of the index.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT size_t usearch_capacity(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Reports the current dimensions of the vectors in the index.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT size_t usearch_dimensions(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Reports the current connectivity of the index.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT size_t usearch_connectivity(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Reserves memory for a specified number of incoming vectors.
 *  @param[inout] index The handle to the USearch index to be resized.
 *  @param[in] capacity The desired total capacity including current size.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_reserve(usearch_index_t index, size_t capacity, usearch_error_t* error);

/**
 *  @brief Retrieves the expansion value used during index creation.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return The expansion value used during index creation.
 */
USEARCH_EXPORT size_t usearch_expansion_add(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Retrieves the expansion value used during search.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return The expansion value used during search.
 */
USEARCH_EXPORT size_t usearch_expansion_search(usearch_index_t index, usearch_error_t* error);

/**
 *  @brief Updates the expansion value used during index creation. Rarely used.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] expansion The new expansion value.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_change_expansion_add(usearch_index_t index, size_t expansion, usearch_error_t* error);

/**
 *  @brief Updates the expansion value used during search. Rarely used.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] expansion The new expansion value.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_change_expansion_search(usearch_index_t index, size_t expansion, usearch_error_t* error);

/**
 *  @brief Updates the number of threads that would be used to construct the index. Rarely used.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] threads The new limit for the number of concurrent threads.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_change_threads_add(usearch_index_t index, size_t threads, usearch_error_t* error);

/**
 *  @brief Updates the number of threads that will be performing concurrent traversals. Rarely used.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] threads The new limit for the number of concurrent threads.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_change_threads_search(usearch_index_t index, size_t threads, usearch_error_t* error);

/**
 *  @brief Updates the metric kind used for distance calculation between vectors.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] kind The metric kind used for distance calculation between vectors.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_change_metric_kind(usearch_index_t index, usearch_metric_kind_t kind,
                                               usearch_error_t* error);

/**
 *  @brief Updates the custom metric function used for distance calculation between vectors.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] metric The custom metric function used for distance calculation between vectors.
 *  @param[in] state The @b optional state pointer to be passed to the custom metric function.
 *  @param[in] kind The metric kind used for distance calculation between vectors. Needed for serialization.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_change_metric(usearch_index_t index, usearch_metric_t metric, void* state,
                                          usearch_metric_kind_t kind, usearch_error_t* error);

/**
 *  @brief Adds a vector with a key to the index.
 *  @param[inout] index The handle to the USearch index to be populated.
 *  @param[in] key The key associated with the vector.
 *  @param[in] vector Pointer to the vector data.
 *  @param[in] vector_kind The scalar type used in the vector data.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_add(              //
    usearch_index_t index, usearch_key_t key, //
    void const* vector, usearch_scalar_kind_t vector_kind, usearch_error_t* error);

/**
 *  @brief Checks if the index contains a vector with a specific key.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] key The key to be checked.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return `true` if the index contains the vector with the given key, `false` otherwise.
 */
USEARCH_EXPORT bool usearch_contains(usearch_index_t index, usearch_key_t key, usearch_error_t* error);

/**
 *  @brief Counts the number of entries in the index under a specific key.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] key The key to be checked.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of vectors found under that key.
 */
USEARCH_EXPORT size_t usearch_count(usearch_index_t index, usearch_key_t key, usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors (kANN) Search for closest vectors to query.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] query_vector Pointer to the query vector data.
 *  @param[in] query_kind The scalar type used in the query vector data.
 *  @param[in] count Upper bound on the number of neighbors to search, the "k" in "kANN".
 *  @param[out] keys Output buffer for up to `count` nearest neighbors keys.
 *  @param[out] distances Output buffer for up to `count` distances to nearest neighbors.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of found matches.
 */
USEARCH_EXPORT size_t usearch_search(                                         //
    usearch_index_t index,                                                    //
    void const* query_vector, usearch_scalar_kind_t query_kind, size_t count, //
    usearch_key_t* keys, usearch_distance_t* distances, usearch_error_t* error);

/**
 *  @brief  Performs k-Approximate Nearest Neighbors (kANN) Search for closest vectors to query,
 *          predicated on a custom function that returns `true` for vectors to be included.
 *
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] query_vector Pointer to the query vector data.
 *  @param[in] query_kind The scalar type used in the query vector data.
 *  @param[in] count Upper bound on the number of neighbors to search, the "k" in "kANN".
 *  @param[in] filter The custom filter function that returns `true` for vectors to be included.
 *  @param[in] filter_state The @b optional state pointer to be passed to the custom filter function.
 *  @param[out] keys Output buffer for up to `count` nearest neighbors keys.
 *  @param[out] distances Output buffer for up to `count` distances to nearest neighbors.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of found matches.
 */
USEARCH_EXPORT size_t usearch_filtered_search(                                //
    usearch_index_t index,                                                    //
    void const* query_vector, usearch_scalar_kind_t query_kind, size_t count, //
    int (*filter)(usearch_key_t key, void* filter_state), void* filter_state, //
    usearch_key_t* keys, usearch_distance_t* distances, usearch_error_t* error);

/**
 *  @brief Retrieves the vector associated with the given key from the index.
 *  @param[in] index The handle to the USearch index to be queried.
 *  @param[in] key The key of the vector to retrieve.
 *  @param[out] vector Pointer to the memory where the vector data will be copied.
 *  @param[in] count Number of vectors that can be fitted into `vector` for multi-vector entries.
 *  @param[in] vector_kind The scalar type used in the vector data.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of vectors found under that name and exported to `vector`.
 */
USEARCH_EXPORT size_t usearch_get(                          //
    usearch_index_t index, usearch_key_t key, size_t count, //
    void* vector, usearch_scalar_kind_t vector_kind, usearch_error_t* error);

/**
 *  @brief Removes the vector associated with the given key from the index.
 *  @param[inout] index The handle to the USearch index to be modified.
 *  @param[in] key The key of the vector to be removed.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of vectors found under that name and dropped from the index.
 */
USEARCH_EXPORT size_t usearch_remove(usearch_index_t index, usearch_key_t key, usearch_error_t* error);

/**
 *  @brief Renames the vector to map to a different key.
 *  @param[inout] index The handle to the USearch index to be modified.
 *  @param[in] from The key of the vector to be renamed.
 *  @param[in] to New key for found entry.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Number of vectors found under that name and renamed.
 */
USEARCH_EXPORT size_t usearch_rename(usearch_index_t index, usearch_key_t from, usearch_key_t to,
                                     usearch_error_t* error);

/**
 *  @brief Computes the distance between two equi-dimensional vectors.
 *  @param[in] vector_first The first vector for comparison.
 *  @param[in] vector_second The second vector for comparison.
 *  @param[in] scalar_kind The scalar type used in the vectors.
 *  @param[in] dimensions The number of dimensions in each vector.
 *  @param[in] metric_kind The metric kind used for distance calculation between vectors.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 *  @return Distance between given vectors.
 */
USEARCH_EXPORT usearch_distance_t usearch_distance(       //
    void const* vector_first, void const* vector_second,  //
    usearch_scalar_kind_t scalar_kind, size_t dimensions, //
    usearch_metric_kind_t metric_kind, usearch_error_t* error);

/**
 *  @brief Multi-threaded many-to-many exact nearest neighbors search for equi-dimensional vectors.
 *  @param[in] dataset Pointer to the first scalar of the dataset matrix.
 *  @param[in] queries Pointer to the first scalar of the queries matrix.
 *  @param[in] dataset_size Number of vectors in the `dataset`.
 *  @param[in] queries_size Number of vectors in the `queries` set.
 *  @param[in] dataset_stride Number of bytes between starts of consecutive vectors in `dataset`.
 *  @param[in] queries_stride Number of bytes between starts of consecutive vectors in `queries`.
 *  @param[in] scalar_kind The scalar type used in the vectors.
 *  @param[in] dimensions The number of dimensions in each vector.
 *  @param[in] metric_kind The metric kind used for distance calculation between vectors.
 *  @param[in] count Upper bound on the number of neighbors to search, the "k" in "kANN".
 *  @param[in] threads Upper bound for the number of CPU threads to use.
 *  @param[out] keys Output matrix for `queries_size * count` nearest neighbors keys. Each row of the
 *              matrix must be contiguous in memory, but different rows can be separated by `keys_stride` bytes.
 *  @param[in] keys_stride Number of bytes between starts of consecutive rows od scalars in `keys`.
 *  @param[out] distances Output matrix for `queries_size * count` distances to nearest neighbors. Each row of the
 *              matrix must be contiguous in memory, but different rows can be separated by `keys_stride` bytes.
 *  @param[in] distances_stride Number of bytes between starts of consecutive rows od scalars in `distances`.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_exact_search(                            //
    void const* dataset, size_t dataset_size, size_t dataset_stride, //
    void const* queries, size_t queries_size, size_t queries_stride, //
    usearch_scalar_kind_t scalar_kind, size_t dimensions,            //
    usearch_metric_kind_t metric_kind, size_t count, size_t threads, //
    usearch_key_t* keys, size_t keys_stride,                         //
    usearch_distance_t* distances, size_t distances_stride,          //
    usearch_error_t* error);

/**
 * @brief Erases all the vectors from the index.
 *  @param[inout] index The handle to the USearch index to be modified.
 *  @param[out] error Pointer to a string where the error message will be stored, if an error occurs.
 */
USEARCH_EXPORT void usearch_clear(usearch_index_t index, usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_H
