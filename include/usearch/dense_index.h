/**
 *  @brief The dense-modality C interface: Vector Search over equidimensional embeddings
 *      with inner-product, angular, euclidean, hamming, and jaccard metrics - all derived
 *      from a single inner-product kernel with pre-computed magnitudes.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/dense_index.h
 *
 *  Every batch input and output is a strided matrix: a base pointer, a row count, and a
 *  byte stride between consecutive rows - packed layouts pass their natural row width, and
 *  a single row passes a count of one with any stride. Handles are created against one
 *  residency - RAM, VRAM, or NVMe - picked by the tagged options union, which also decides
 *  how loading behaves.
 */
#ifndef UNUM_USEARCH_DENSE_INDEX_H
#define UNUM_USEARCH_DENSE_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Vector Search Index over equidimensional dense representations.
 *
 *  @see usearch_dense_init
 *  @see usearch_dense_add
 *  @see usearch_dense_search
 */
USEARCH_EXPORT typedef void* usearch_dense_t;

/**
 *  @brief A consistent snapshot cursor over the index contents. Captures the publication
 *      watermark at creation, so concurrent inserts never appear mid-iteration, and the
 *      walk is a sequential row scan - export, migration, and visualization feed off it.
 */
USEARCH_EXPORT typedef void* usearch_dense_cursor_t;

/**
 *  @brief A resumable nearest-neighbor stream: best-first traversal is naturally pausable,
 *      so the finder owns a live frontier & visited set and yields matches lazily. Callers
 *      that post-filter aggressively pull until satisfied instead of guessing "k" upfront.
 *      Each open finder pins one traversal context's memory - free it promptly.
 */
USEARCH_EXPORT typedef void* usearch_dense_finder_t;

/** @brief The dense metric kinds supported out of the box, with SIMD kernels dispatched
 *      by the machine's capabilities. */
USEARCH_EXPORT typedef enum usearch_dense_metric_t {
    usearch_dense_metric_unknown_k = 0,
    usearch_dense_metric_inner_k = 1,
    usearch_dense_metric_angular_k = 2,
    usearch_dense_metric_euclidean_k = 3,
    usearch_dense_metric_hamming_k = 4,
    usearch_dense_metric_jaccard_k = 5,
} usearch_dense_metric_t;

/**
 *  @brief Type-punned callback for user-defined dense metrics, measuring the dissimilarity
 *      of two equidimensional vectors. Custom metrics forfeit the SIMD kernel chains.
 *
 *  Arguments, in order:
 *  - @c first: pointer to the first vector's scalars, in the index's stored type.
 *  - @c second: pointer to the second vector's scalars, same type and length.
 *  - @c dimensions: the number of logical dimensions in each vector.
 *  - @c state: the opaque pointer registered alongside the callback, forwarded verbatim.
 */
USEARCH_EXPORT typedef usearch_distance_t (*usearch_dense_metric_callback_t)( //
    void const* first, void const* second, size_t dimensions, void* state);

/** @brief RAM-residency behavior: CPU threading & memory backing of the volatile index. */
USEARCH_EXPORT typedef struct usearch_dense_ram_options_t {
    /** @brief Worker threads for construction & batched search; zero takes 7/8 of the
     *      logical cores - full occupancy starves the ordered publication path. */
    size_t threads;
    /** @brief Rounds stored rows up to the SIMD step quantum: cache-line-aligned rows and
     *      tail-free unrolled kernels, at the price of slightly larger arenas. */
    bool pad_dimensions;
    /** @brief Expects a provisioned huge-page pool and fails loudly when it is missing,
     *      rather than silently degrading to base pages. */
    bool huge_pages;
} usearch_dense_ram_options_t;

/** @brief VRAM-residency behavior: device selection for the GPU-traversed index. */
USEARCH_EXPORT typedef struct usearch_dense_vram_options_t {
    /** @brief CUDA device ordinal; negative picks the first free device. */
    int32_t device;
    /** @brief Allows spilling cold shards into host memory over unified addressing. */
    bool unified_addressing;
} usearch_dense_vram_options_t;

/** @brief NVMe-residency behavior: the on-disk arena and its I/O engine. */
USEARCH_EXPORT typedef struct usearch_dense_nvme_options_t {
    /** @brief Directory hosting the index arena & checkpoints. Must outlive the handle. */
    char const* directory;
    /** @brief I/O granularity in bytes; zero probes the device. 512 fits one 256-d i8 row. */
    uint32_t sector_bytes;
    /** @brief Submission ring depth; zero derives from the device's queue capabilities. */
    uint32_t queue_depth;
    /** @brief O_DIRECT-class page-cache bypass - the default and the point. */
    bool direct_io;
} usearch_dense_nvme_options_t;

/**
 *  @brief Construction & loading options of a dense index. The @c residency field tags
 *      which union member is read; zeroed optional fields derive from the geometry.
 */
USEARCH_EXPORT typedef struct usearch_dense_options_t {
    /** @brief The metric kind used for distance calculations between vectors. */
    usearch_dense_metric_t metric;
    /** @brief The @b optional custom metric; must be set when @c metric is unknown. */
    usearch_dense_metric_callback_t metric_callback;
    /** @brief The @b optional opaque state forwarded into every @c metric_callback call. */
    void* metric_callback_state;
    /** @brief The scalar type vectors are stored & quantized into. */
    usearch_scalar_t quantization;
    /** @brief The number of dimensions in the indexed vectors. */
    size_t dimensions;
    /** @brief The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;
    /** @brief Selects the active member of the union below, and the meaning of "load". */
    usearch_residency_t residency;
    union {
        usearch_dense_ram_options_t ram;
        usearch_dense_vram_options_t vram;
        usearch_dense_nvme_options_t nvme;
    };
} usearch_dense_options_t;

/**
 *  @brief Initializes a new dense index for the residency the options select.
 *  @param[in] options Pointer to the populated options structure.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_dense_t usearch_dense_init(usearch_dense_options_t const* options, usearch_error_t* error);

/**
 *  @brief Frees the resources associated with the index.
 *  @param[inout] index The handle to be freed; invalid afterwards.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 */
USEARCH_EXPORT void usearch_dense_free(usearch_dense_t index, usearch_error_t* error);

/**
 *  @brief Reserves capacity for a number of incoming vectors ahead of insertion.
 *  @param[inout] index The handle to be resized.
 *  @param[in] capacity The desired total capacity, including the current size.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 */
USEARCH_EXPORT void usearch_dense_reserve(usearch_dense_t index, size_t capacity, usearch_error_t* error);

/**
 *  @brief Erases all entries, keeping the reserved capacity and frozen geometry.
 *  @param[inout] index The handle to be cleared.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 */
USEARCH_EXPORT void usearch_dense_clear(usearch_dense_t index, usearch_error_t* error);

/** @brief Reports the number of stored entries. */
USEARCH_EXPORT size_t usearch_dense_size(usearch_dense_t index, usearch_error_t* error);
/** @brief Reports the reserved capacity in entries. */
USEARCH_EXPORT size_t usearch_dense_capacity(usearch_dense_t index, usearch_error_t* error);
/** @brief Reports the frozen dimensionality of the stored vectors. */
USEARCH_EXPORT size_t usearch_dense_dimensions(usearch_dense_t index, usearch_error_t* error);
/** @brief Reports the frozen connections-per-node bound of the graph. */
USEARCH_EXPORT size_t usearch_dense_connectivity(usearch_dense_t index, usearch_error_t* error);
/** @brief Reports the memory usage of the index in bytes. */
USEARCH_EXPORT size_t usearch_dense_memory_usage(usearch_dense_t index, usearch_error_t* error);
/** @brief Reports the capability codename the index dispatched to on this machine. */
USEARCH_EXPORT char const* usearch_dense_capabilities(usearch_dense_t index, usearch_error_t* error);

/**
 *  @brief Optional per-call configuration & telemetry for insertions. Zero-initialize before
 *      use: the setting fields are read as the batch starts, the @b reported fields are
 *      overwritten as it ends. Passing @c NULL derives every setting and skips reporting.
 */
USEARCH_EXPORT typedef struct usearch_dense_add_config_t {
    /** @brief The @b optional construction candidate-pool breadth; zero derives. */
    size_t expansion;
    /** @brief @b Reported: graph nodes visited while placing the batch. */
    size_t visited_nodes;
    /** @brief @b Reported: distances computed while placing the batch. */
    size_t computed_distances;
} usearch_dense_add_config_t;

/**
 *  @brief Adds a strided matrix of vectors under their keys - the native batch operation;
 *      a single vector is a one-row matrix.
 *
 *  @param[inout] index The handle to be populated.
 *  @param[in] keys Array of @p count keys, one per row.
 *  @param[in] vectors Base pointer of the first row of vector data.
 *  @param[in] count Number of rows in @p vectors and entries in @p keys.
 *  @param[in] stride Number of bytes between starts of consecutive rows in @p vectors.
 *  @param[in] kind The scalar type of @p vectors, converted on ingest when it differs
 *      from the stored quantization.
 *  @param[inout] config The @b optional per-call configuration & telemetry; @c NULL derives
 *      every setting and skips reporting.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 *  @return Number of successfully inserted rows - equal to @p count unless an error stopped
 *      the batch early.
 */
USEARCH_EXPORT size_t usearch_dense_add(              //
    usearch_dense_t index, usearch_key_t const* keys, //
    void const* vectors, size_t count, size_t stride, //
    usearch_scalar_t kind,                            //
    usearch_dense_add_config_t* config, usearch_error_t* error);

/**
 *  @brief Optional per-call configuration & telemetry for searches, shared by the plain and
 *      the filtered form. Zero-initialize before use: the setting fields are read as the
 *      batch starts, the @b reported fields are overwritten as it ends. Passing @c NULL
 *      derives every setting and skips reporting.
 */
USEARCH_EXPORT typedef struct usearch_dense_search_config_t {
    /** @brief The @b optional search candidate-pool breadth; zero derives. */
    size_t expansion;
    /** @brief @b Reported: graph nodes visited while resolving the batch. */
    size_t visited_nodes;
    /** @brief @b Reported: distances computed while resolving the batch. */
    size_t computed_distances;
} usearch_dense_search_config_t;

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a strided matrix of queries -
 *      the native batch operation, parallelized internally; one query is a one-row matrix.
 *
 *  @param[in] index The handle to be queried.
 *  @param[in] queries Base pointer of the first query row.
 *  @param[in] queries_count Number of query rows.
 *  @param[in] queries_stride Number of bytes between starts of consecutive query rows.
 *  @param[in] queries_kind The scalar type of the query rows.
 *  @param[in] wanted Upper bound on matches per query - the "k" in "kANN".
 *  @param[inout] config The @b optional per-call configuration & telemetry; @c NULL derives
 *      every setting and skips reporting.
 *  @param[out] keys Output matrix of `queries_count × wanted` keys; rows contiguous.
 *  @param[in] keys_stride Number of bytes between starts of consecutive rows in @p keys.
 *  @param[out] distances Output matrix of `queries_count × wanted` distances; rows contiguous.
 *  @param[in] distances_stride Number of bytes between starts of consecutive @p distances rows.
 *  @param[out] counts Output array of @p queries_count per-query match counts.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 */
USEARCH_EXPORT void usearch_dense_search(                             //
    usearch_dense_t index,                                            //
    void const* queries, size_t queries_count, size_t queries_stride, //
    usearch_scalar_t queries_kind, size_t wanted,                     //
    usearch_dense_search_config_t* config,                            //
    usearch_key_t* keys, size_t keys_stride,                          //
    usearch_distance_t* distances, size_t distances_stride,           //
    size_t* counts, usearch_error_t* error);

/**
 *  @brief The filtered sibling of @c usearch_dense_search: identical strided-matrix shape,
 *      admitting only entries the @p predicate approves. Non-matching entries still route
 *      the traversal, so recall does not collapse at low selectivity.
 *
 *  @param[in] predicate Per-key admission callback; returns non-zero to admit.
 *  @param[in] predicate_state Opaque pointer forwarded into every @p predicate call.
 */
USEARCH_EXPORT void usearch_dense_search_if(                          //
    usearch_dense_t index,                                            //
    void const* queries, size_t queries_count, size_t queries_stride, //
    usearch_scalar_t queries_kind, size_t wanted,                     //
    usearch_dense_search_config_t* config,                            //
    usearch_predicate_t predicate, void* predicate_state,             //
    usearch_key_t* keys, size_t keys_stride,                          //
    usearch_distance_t* distances, size_t distances_stride,           //
    size_t* counts, usearch_error_t* error);

/**
 *  @brief Checks if the index holds at least one entry under @p key.
 */
USEARCH_EXPORT bool usearch_dense_contains(usearch_dense_t index, usearch_key_t key, usearch_error_t* error);

/**
 *  @brief Counts the entries stored under @p key.
 */
USEARCH_EXPORT size_t usearch_dense_count(usearch_dense_t index, usearch_key_t key, usearch_error_t* error);

/**
 *  @brief Retrieves up to @p count vectors stored under @p key into a strided matrix.
 *  @param[out] vectors Base pointer of the output matrix; rows contiguous.
 *  @param[in] stride Number of bytes between starts of consecutive output rows.
 *  @param[in] kind The scalar type to export as, converted from storage when different.
 *  @return Number of rows written.
 */
USEARCH_EXPORT size_t usearch_dense_get(        //
    usearch_dense_t index, usearch_key_t key,   //
    void* vectors, size_t count, size_t stride, //
    usearch_scalar_t kind, usearch_error_t* error);

/**
 *  @brief Absorbs @p donor into @p index - cheaper than re-inserting its vectors one by
 *      one, wiring cross-links via landmark stitching. The donor handle is consumed and
 *      must not be used afterwards, not even for @c usearch_dense_free. Readers of
 *      @p index may keep searching throughout: unpublished rows hide behind the watermark.
 */
USEARCH_EXPORT void usearch_dense_merge(usearch_dense_t index, usearch_dense_t donor, usearch_error_t* error);

/**
 *  @brief Opens a snapshot cursor over the current contents; see @c usearch_dense_cursor_t.
 *  @return The cursor, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_dense_cursor_t usearch_dense_entries(usearch_dense_t index, usearch_error_t* error);

/**
 *  @brief Copies the next batch of entries out of the cursor into strided outputs.
 *  @param[out] keys Output array of up to @p count keys.
 *  @param[out] vectors Output matrix of up to @p count rows; @c NULL skips vector export.
 *  @param[in] stride Number of bytes between starts of consecutive output vector rows.
 *  @param[in] kind The scalar type to export vectors as.
 *  @return Number of rows written; zero when the snapshot is exhausted.
 */
USEARCH_EXPORT size_t usearch_dense_entries_next(       //
    usearch_dense_cursor_t cursor, usearch_key_t* keys, //
    void* vectors, size_t count, size_t stride,         //
    usearch_scalar_t kind, usearch_error_t* error);

/** @brief Releases the cursor. */
USEARCH_EXPORT void usearch_dense_entries_free(usearch_dense_cursor_t cursor);

/**
 *  @brief Opens a resumable nearest-neighbor stream for one query; see @c usearch_dense_finder_t.
 *  @return The finder, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_dense_finder_t usearch_dense_find( //
    usearch_dense_t index, void const* query, usearch_scalar_t kind, usearch_error_t* error);

/**
 *  @brief Pulls up to @p count next-nearest matches from the stream. Ordering beyond the
 *      expansion horizon is best-effort nearest, not globally exact.
 *  @return Number of matches written; zero when the traversal is drained.
 */
USEARCH_EXPORT size_t usearch_dense_find_next(                        //
    usearch_dense_finder_t finder,                                    //
    usearch_key_t* keys, usearch_distance_t* distances, size_t count, //
    usearch_error_t* error);

/** @brief Releases the finder and the traversal context it pinned. */
USEARCH_EXPORT void usearch_dense_find_free(usearch_dense_finder_t finder);

/**
 *  @brief Reports the expected serialized size of the index in bytes.
 */
USEARCH_EXPORT size_t usearch_dense_serialized_length(usearch_dense_t index, usearch_error_t* error);

/**
 *  @brief Saves the index under a checkpoint path.
 */
USEARCH_EXPORT void usearch_dense_save(usearch_dense_t index, char const* path, usearch_error_t* error);

/**
 *  @brief Loads a checkpoint into an initialized handle. The handle's residency decides the
 *      behavior: materialize into RAM, upload to the device, or open in place on disk.
 */
USEARCH_EXPORT void usearch_dense_load(usearch_dense_t index, char const* path, usearch_error_t* error);

/**
 *  @brief Creates a read-only view of a checkpoint without copying it into memory.
 */
USEARCH_EXPORT void usearch_dense_view(usearch_dense_t index, char const* path, usearch_error_t* error);

/**
 *  @brief Reads a checkpoint's geometry & metric metadata into @p options without loading it.
 */
USEARCH_EXPORT void usearch_dense_metadata(char const* path, usearch_dense_options_t* options, usearch_error_t* error);

/**
 *  @brief Saves the index into a caller-owned memory buffer of @p length bytes.
 */
USEARCH_EXPORT void usearch_dense_save_buffer( //
    usearch_dense_t index, void* buffer, size_t length, usearch_error_t* error);

/**
 *  @brief Loads an index from a memory buffer into an initialized handle.
 */
USEARCH_EXPORT void usearch_dense_load_buffer( //
    usearch_dense_t index, void const* buffer, size_t length, usearch_error_t* error);

/**
 *  @brief Creates a read-only view over a serialized index in a caller-owned buffer, which
 *      must outlive the handle.
 */
USEARCH_EXPORT void usearch_dense_view_buffer( //
    usearch_dense_t index, void const* buffer, size_t length, usearch_error_t* error);

/**
 *  @brief Reads geometry & metric metadata from a serialized buffer into @p options.
 */
USEARCH_EXPORT void usearch_dense_metadata_buffer( //
    void const* buffer, size_t length, usearch_dense_options_t* options, usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_DENSE_INDEX_H
