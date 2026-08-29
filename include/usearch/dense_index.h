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
 *
 *  @section dense_index_serialization Serialization
 *
 *  Checkpoints ride the SafeGraphs interchange format, and the path extension picks the
 *  layout. A @c .safegraphs path is one monolith: a self-describing file holding one shard's
 *  vectors, neighbor lists, and its key column. A @c .safegraphs.json path is a manifest: a
 *  small JSON catalog of many such monoliths - each with shard-local node ids, so edges
 *  never cross files - with every shard path resolved relative to the manifest. A manifest
 *  records where its shards were written, never where a reader must keep them: loading into a
 *  handle that names memory domains places them across those domains as it reads, and loading
 *  into one that names none opens every shard where it already sits and moves no bytes. So a
 *  checkpoint written from a mounted bucket stripes onto local drives by being loaded, and a
 *  symlinked manifest entry still works for anyone who would rather arrange files than name
 *  domains. In-memory buffers hold single monoliths only, since a manifest's relative
 *  paths have nothing to resolve against in memory. Checkpoints are residency-portable: one
 *  file materializes into RAM, uploads to a device, or opens in place on disk.
 *
 *  @section dense_index_gpu_acceleration GPU Acceleration & Constraints
 *
 *  A VRAM handle builds and searches on the devices its options name, one shard per device
 *  and never a replica, so capacity is what those devices host together: the geometry freezes
 *  at the first reservation, and rows grow only up to that. Planes stay device-resident and
 *  never migrate, so an index outgrowing its devices fails rather than paging to the host -
 *  the answer to a dataset past one accelerator is to name more of them.
 *
 *  @section dense_index_nvme_traversals NVME Traversals & Mergers
 *
 *  An NVMe handle opens a checkpoint and traverses it through whichever I/O engine can reach
 *  the devices its paths name - a filesystem path selects a kernel engine, a transport
 *  identifier a userspace one, and a FUSE mount over a bucket reads as an ordinary path.
 *  External-memory graphs are never modified in place, so the handle is read-only,
 *  and every mutating verb fails on it. Bigger indexes come from merging, never mutating:
 *  shards build in RAM, serialize as monoliths, and join a manifest - the compaction shape
 *  of LSM storage.
 *
 *  @section dense_index_concurrency Concurrency & Thread Safety
 *
 *  A batch insertion is all-or-nothing: every fallible step precedes the claim of the first
 *  entry identity, so a failed batch publishes nothing and leaves the reported size untouched.
 *
 *  A handle may be shared across threads. On a RAM handle, insertions and searches overlap
 *  freely: searches read only entries published behind the visibility watermark and never
 *  write to the graph, while insertions synchronize through per-node locks and publish in
 *  order. A VRAM handle is driven by one host thread through streams - operations on
 *  distinct streams overlap, host threading adds nothing. An NVMe handle admits arbitrarily
 *  many concurrent searches, having nothing to mutate. Serialization is the one exception:
 *  saving demands a mutation-free window, though concurrent readers keep searching
 *  throughout. Per-call outputs - stats, counts, the error pointer - belong to their call
 *  and must not be shared between in-flight calls.
 *
 *  @section dense_index_linear_algebra Approximate Products & Attention
 *
 *  Two opt-in verbs at the end of this header read the graph as linear algebra rather than as a
 *  search. @c usearch_dense_multiply is an approximate `A × Bᵀ` in the shape of BLAS' `?GEMM`,
 *  with the indexed rows as `B`, and a sparse output: every pair the traversal scored, not only
 *  the nearest. Unlike a sparse `?GEMM`, the output pattern is not implied by the operands' own
 *  pattern - it is whatever the walk touched - so the product is dense-in, sparse-out, and its
 *  sparsity is magnitude-driven.
 *
 *  @c usearch_dense_attend folds a softmax over that same walk. A running maximum, partition,
 *  and weighted value sum update per scored batch and rescale whenever the maximum moves - the
 *  online-softmax recurrence of Flash Attention, with the graph replacing the linear sweep over
 *  key tiles. Softmax weights decay exponentially, so scoring a superset of the true top matches
 *  costs accuracy nothing, and neither verb pays for the ranking a search would. One index serves
 *  one layer's one key-value head: grouped-query and multi-head-latent attention already share
 *  their keys across query heads, so every query head is just another row of the query matrix.
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
USEARCH_EXPORT typedef void* usearch_dense_index_t;

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

/** @brief The dense metric kinds supported out of the box, with SIMD kernels dispatched by the machine's
 *      capabilities. The last two read their rows as @c usearch_scalar_u1x8_k bit sets and refuse any other
 *      quantization; the first four refuse that one. */
USEARCH_EXPORT typedef enum usearch_dense_metric_t {
/** @brief No built-in kind - @c metric_callback supplies the scoring and forfeits the SIMD kernel chains. */
    usearch_dense_metric_unknown_k = 0,
/** @brief Sum of elementwise products, unnormalized and scale-sensitive; the only kind the linear-algebra
     *      verbs accept, since a product and a softmax both need logits rather than distances. */
    usearch_dense_metric_inner_k = 1,
/** @brief Cosine distance in [0, 2], zero for identical directions and blind to magnitude, which is
     *      precomputed once per row at insertion rather than per comparison. */
    usearch_dense_metric_angular_k = 2,
/** @brief L2 distance, non-negative and zero only for identical rows - the scale-sensitive counterpart to
     *      angular, where a doubled vector is far from its original rather than identical to it. */
    usearch_dense_metric_euclidean_k = 3,
/** @brief Count of differing bits, so its range is the dimensionality and every bit weighs the same. */
    usearch_dense_metric_hamming_k = 4,
/** @brief Differing bits over union bits in [0, 1], ignoring the zeros both rows share - the kind for sparse
     *      sets, where Hamming would call two nearly-empty rows identical. */
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
    /** @brief The number of dimensions in the indexed vectors; zero defers the geometry to
     *      a later @c usearch_dense_load, which reads it off the checkpoint. */
    size_t dimensions;
    /** @brief Rounds stored rows up to the SIMD step quantum: cache-line-aligned rows and
     *      tail-free unrolled kernels, at the price of slightly larger arenas. A property
     *      of the stored layout, so it applies whichever residency traverses it. */
    bool pad_dimensions;
    /** @brief The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;
    /** @brief Selects the active member of the union below, and the meaning of "load". */
    usearch_residency_t residency;
    union {
        usearch_ram_options_t ram;
        usearch_vram_options_t vram;
        usearch_nvme_options_t nvme;
    };
} usearch_dense_options_t;

/**
 *  @brief Initializes a new dense index for the residency the options select.
 *  @param[in] options Pointer to the populated options structure.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_dense_index_t usearch_dense_init(usearch_dense_options_t const* options, usearch_error_t* error);

/**
 *  @brief Frees the resources associated with the index; cannot fail, so takes no error.
 *  @param[inout] index The handle to be freed; invalid afterwards.
 */
USEARCH_EXPORT void usearch_dense_free(usearch_dense_index_t index);

/**
 *  @brief Reserves capacity for a number of incoming vectors ahead of insertion.
 *  @param[inout] index The handle to be resized.
 *  @param[in] capacity The desired total capacity, including the current size.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 */
USEARCH_EXPORT void usearch_dense_reserve(usearch_dense_index_t index, size_t capacity, usearch_error_t* error);

/**
 *  @brief Erases all entries, keeping the reserved capacity and frozen geometry.
 *  @param[inout] index The handle to be cleared.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 */
USEARCH_EXPORT void usearch_dense_clear(usearch_dense_index_t index, usearch_error_t* error);

/** @brief Reports the number of stored entries. */
USEARCH_EXPORT size_t usearch_dense_size(usearch_dense_index_t index, usearch_error_t* error);
/** @brief Reports the reserved capacity in entries. */
USEARCH_EXPORT size_t usearch_dense_capacity(usearch_dense_index_t index, usearch_error_t* error);
/** @brief Reports the frozen dimensionality of the stored vectors. */
USEARCH_EXPORT size_t usearch_dense_dimensions(usearch_dense_index_t index, usearch_error_t* error);
/** @brief Reports the frozen connections-per-node bound of the graph. */
USEARCH_EXPORT size_t usearch_dense_connectivity(usearch_dense_index_t index, usearch_error_t* error);
/** @brief Reports the memory usage of the index in bytes. */
USEARCH_EXPORT size_t usearch_dense_memory_usage(usearch_dense_index_t index, usearch_error_t* error);
/** @brief Reports the capability codename the index dispatched to on this machine. */
USEARCH_EXPORT char const* usearch_dense_capabilities(usearch_dense_index_t index, usearch_error_t* error);

/**
 *  @brief Optional per-call settings, shared by insertion and both search forms - a pure
 *      input, never written. Passing @c NULL derives every setting.
 */
USEARCH_EXPORT typedef struct usearch_dense_expansion_config_t {
    /** @brief The @b optional candidate-pool breadth; zero derives. */
    size_t expansion_factor;
} usearch_dense_expansion_config_t;

/**
 *  @brief Traversal telemetry, shared by insertion and both search forms - a pure output,
 *      fully overwritten by the call it is passed to. Passing @c NULL skips reporting.
 */
USEARCH_EXPORT typedef struct usearch_dense_traversal_stats_t {
    /** @brief Graph members visited while resolving the batch. */
    size_t visited_members;
    /** @brief Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;
    /** @brief Starvation restarts - re-seedings after the frontier drained prematurely,
     *      and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;
} usearch_dense_traversal_stats_t;

/**
 *  @brief Adds a strided matrix of vectors under their keys - the native batch operation;
 *      a single vector is a one-row matrix. Overlaps safely with concurrent searches on
 *      the same handle. @ref dense_index_concurrency
 *
 *  @param[inout] index The handle to be populated.
 *  @param[in] keys Array of @p count keys, one per row.
 *  @param[in] vectors Base pointer of the first row of vector data.
 *  @param[in] count Number of rows in @p vectors and entries in @p keys.
 *  @param[in] stride Number of bytes between starts of consecutive rows in @p vectors.
 *  @param[in] scalar_kind The scalar type of @p vectors, converted on ingest when it differs
 *      from the stored quantization.
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return @p count on success, zero on failure - a batch is all-or-nothing. Every step that
 *      can fail runs before the first entry identity is claimed, so a rejected batch leaves
 *      the index untouched and publishes no row. An accelerator fault severe enough to poison
 *      the device context is the sole exception: it reports @c usearch_error_device_k and
 *      latches the index unusable rather than partially applied.
 */
USEARCH_EXPORT size_t usearch_dense_add(                    //
    usearch_dense_index_t index, usearch_key_t const* keys, //
    void const* vectors, size_t count, size_t stride,       //
    usearch_scalar_t scalar_kind,                           //
    usearch_dense_expansion_config_t const* config, usearch_dense_traversal_stats_t* stats, usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a strided matrix of queries -
 *      the native batch operation, parallelized internally; one query is a one-row matrix.
 *      Never writes to the graph, so any number may overlap with concurrent insertions.
 *  @ref dense_index_concurrency
 *
 *  @param[in] index The handle to be queried.
 *  @param[in] queries Base pointer of the first query row.
 *  @param[in] queries_count Number of query rows.
 *  @param[in] queries_stride Number of bytes between starts of consecutive query rows.
 *  @param[in] scalar_kind The scalar type of the query rows.
 *  @param[in] wanted Upper bound on matches per query - the "k" in "kANN".
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] keys Output matrix of `queries_count × wanted` keys; rows contiguous.
 *  @param[in] keys_stride Number of bytes between starts of consecutive rows in @p keys.
 *  @param[out] distances Output matrix of `queries_count × wanted` distances; rows contiguous.
 *  @param[in] distances_stride Number of bytes between starts of consecutive @p distances rows.
 *  @param[out] counts Output array of @p queries_count per-query match counts.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 */
USEARCH_EXPORT void usearch_dense_search(                             //
    usearch_dense_index_t index,                                      //
    void const* queries, size_t queries_count, size_t queries_stride, //
    usearch_scalar_t scalar_kind, size_t wanted,                      //
    usearch_dense_expansion_config_t const* config,                   //
    usearch_dense_traversal_stats_t* stats,                           //
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
    usearch_dense_index_t index,                                      //
    void const* queries, size_t queries_count, size_t queries_stride, //
    usearch_scalar_t scalar_kind, size_t wanted,                      //
    usearch_dense_expansion_config_t const* config,                   //
    usearch_dense_traversal_stats_t* stats,                           //
    usearch_predicate_t predicate, void* predicate_state,             //
    usearch_key_t* keys, size_t keys_stride,                          //
    usearch_distance_t* distances, size_t distances_stride,           //
    size_t* counts, usearch_error_t* error);

/**
 *  @brief Counts the entries stored under @p key; zero means the index holds none.
 *
 *  Every entry is one row of @c dimensions, so the count is the whole extent and there is
 *  nothing further to report - the sibling modalities carry a stats out-parameter here only
 *  because their entries vary in size. Pairing this with @c usearch_dense_get is not
 *  atomic, but harmlessly so: entries are only ever added, so a stale count under-reports
 *  and @c usearch_dense_get returns the true total.
 */
USEARCH_EXPORT size_t usearch_dense_count(usearch_dense_index_t index, usearch_key_t key, usearch_error_t* error);

/**
 *  @brief Retrieves the vectors stored under @p key into a strided matrix, one row per
 *      entry sharing the key, in insertion order.
 *  @param[out] vectors Base pointer of the output matrix; @c NULL probes the count without
 *      copying.
 *  @param[in] wanted Upper bound on rows to write - the row capacity of @p vectors.
 *  @param[in] stride Number of bytes between starts of consecutive output rows.
 *  @param[in] scalar_kind The scalar type to export as, converted from storage when different.
 *  @return The @b true number of entries under @p key, never the number written - a value
 *      above @p wanted means the copy was truncated, so allocate that many and call again.
 *      Zero when the key is absent.
 */
USEARCH_EXPORT size_t usearch_dense_get(            //
    usearch_dense_index_t index, usearch_key_t key, //
    void* vectors, size_t wanted, size_t stride,    //
    usearch_scalar_t scalar_kind, usearch_error_t* error);

/**
 *  @brief Absorbs @p donor into @p index - cheaper than re-inserting its vectors one by
 *      one, wiring cross-links via landmark stitching. The donor handle is consumed and
 *      must not be used afterwards, not even for @c usearch_dense_free. Readers of
 *      @p index may keep searching throughout: unpublished rows hide behind the watermark.
 */
USEARCH_EXPORT void usearch_dense_merge(usearch_dense_index_t index, usearch_dense_index_t donor,
                                        usearch_error_t* error);

/**
 *  @brief Opens a snapshot cursor over the current contents; see @c usearch_dense_cursor_t.
 *  @return The cursor, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_dense_cursor_t usearch_dense_cursor(usearch_dense_index_t index, usearch_error_t* error);

/**
 *  @brief Copies the next batch of entries out of the cursor into strided outputs.
 *  @param[out] keys Output array of up to @p wanted keys.
 *  @param[out] vectors Output matrix of up to @p wanted rows; @c NULL skips vector export.
 *  @param[in] wanted Upper bound on rows to write - the capacity of @p keys and @p vectors.
 *  @param[in] stride Number of bytes between starts of consecutive output vector rows.
 *  @param[in] scalar_kind The scalar type to export vectors as.
 *  @return Number of rows written; zero when the snapshot is exhausted.
 */
USEARCH_EXPORT size_t usearch_dense_cursor_next(        //
    usearch_dense_cursor_t cursor, usearch_key_t* keys, //
    void* vectors, size_t wanted, size_t stride,        //
    usearch_scalar_t scalar_kind, usearch_error_t* error);

/** @brief Releases the cursor. */
USEARCH_EXPORT void usearch_dense_cursor_free(usearch_dense_cursor_t cursor);

/**
 *  @brief Opens a resumable nearest-neighbor stream for one query; see @c usearch_dense_finder_t.
 *  @return The finder, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_dense_finder_t usearch_dense_find( //
    usearch_dense_index_t index, void const* query, usearch_scalar_t scalar_kind, usearch_error_t* error);

/**
 *  @brief Pulls up to @p wanted next-nearest matches from the stream. Ordering beyond the
 *      expansion horizon is best-effort nearest, not globally exact.
 *  @param[in] wanted Upper bound on matches to write - the capacity of @p keys and @p distances.
 *  @return Number of matches written; zero when the traversal is drained.
 */
USEARCH_EXPORT size_t usearch_dense_find_next(                         //
    usearch_dense_finder_t finder,                                     //
    usearch_key_t* keys, usearch_distance_t* distances, size_t wanted, //
    usearch_error_t* error);

/** @brief Releases the finder and the traversal context it pinned. */
USEARCH_EXPORT void usearch_dense_find_free(usearch_dense_finder_t finder);

/**
 *  @brief Reports the expected serialized size in bytes: every shard file, plus the
 *      manifest when the index spans more than one.
 */
USEARCH_EXPORT size_t usearch_dense_serialized_length(usearch_dense_index_t index, usearch_error_t* error);

/**
 *  @brief Saves the index under @p path in the layout its extension picks - one monolith,
 *      or a manifest beside its shards. Demands a mutation-free window; concurrent readers
 *      keep searching throughout. @ref dense_index_serialization
 */
USEARCH_EXPORT void usearch_dense_save(usearch_dense_index_t index, usearch_string_t path, usearch_error_t* error);

/**
 *  @brief Loads a checkpoint into an initialized handle, shard by shard, with the handle's
 *      residency deciding the behavior: materialize into RAM, upload to the device, or open
 *      in place on disk - the read-only form, where mutating verbs fail. A handle whose
 *      geometry was left zeroed freezes it here, from the checkpoint's own headers.
 *  @ref dense_index_serialization
 */
USEARCH_EXPORT void usearch_dense_load(usearch_dense_index_t index, usearch_string_t path, usearch_error_t* error);

/**
 *  @brief Reads a checkpoint's geometry & metric metadata into @p options without loading
 *      it - from the manifest's summary when @p path names one, so no shard is opened.
 */
USEARCH_EXPORT void usearch_dense_metadata( //
    usearch_string_t path, usearch_dense_options_t* options, usearch_error_t* error);

/**
 *  @brief Saves the index into a caller-owned memory buffer of @p length bytes, as one
 *      monolith. @ref dense_index_serialization
 */
USEARCH_EXPORT void usearch_dense_save_buffer( //
    usearch_dense_index_t index, void* buffer, size_t length, usearch_error_t* error);

/**
 *  @brief Loads a monolith from a memory buffer into an initialized handle, with the
 *      handle's residency deciding the behavior like @c usearch_dense_load; a residency
 *      that opens in place demands the buffer outlive the handle.
 */
USEARCH_EXPORT void usearch_dense_load_buffer( //
    usearch_dense_index_t index, void const* buffer, size_t length, usearch_error_t* error);

/**
 *  @brief Reads geometry & metric metadata from a serialized monolith buffer into @p options.
 */
USEARCH_EXPORT void usearch_dense_metadata_buffer( //
    void const* buffer, size_t length, usearch_dense_options_t* options, usearch_error_t* error);

/**
 *  @brief Approximate sparse product of a strided matrix against the indexed rows - the linear
 *      algebra sibling of @c usearch_dense_search, reporting @b every pair the traversal scored
 *      instead of only the nearest. @ref dense_index_linear_algebra
 *
 *  A walk scores many times the pairs a search reports, and each of them is an exact product the
 *  caller would otherwise recompute - this verb hands them all back. Pairs arrive in traversal
 *  order rather than sorted, and carry no duplicates: the traversal's visited set admits each
 *  indexed row at most once per query row.
 *
 *  @param[in] index The handle supplying the right-hand rows.
 *  @param[in] rows Base pointer of the first left-hand row.
 *  @param[in] rows_count Number of left-hand rows.
 *  @param[in] rows_stride Number of bytes between starts of consecutive left-hand rows.
 *  @param[in] scalar_kind The scalar type of the left-hand rows.
 *  @param[in] wanted Upper bound on pairs emitted per row - the column capacity of the outputs.
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting. A wider
 *      expansion factor widens the emitted pattern, being what the walk's breadth decides.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] columns Output matrix of `rows_count × wanted` keys, naming the indexed row of
 *      each pair; rows contiguous.
 *  @param[in] columns_stride Number of bytes between starts of consecutive @p columns rows.
 *  @param[out] products Output matrix of `rows_count × wanted` products, positionally paired
 *      with @p columns; rows contiguous.
 *  @param[in] products_stride Number of bytes between starts of consecutive @p products rows.
 *  @param[out] counts Output array of @p rows_count per-row pair counts. Each reports the
 *      @b true number of pairs scored, never the number written - a value above @p wanted means
 *      that row was truncated to its nearest pairs, so raise @p wanted and call again.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 */
USEARCH_EXPORT void usearch_dense_multiply(                  //
    usearch_dense_index_t index,                             //
    void const* rows, size_t rows_count, size_t rows_stride, //
    usearch_scalar_t scalar_kind, size_t wanted,             //
    usearch_dense_expansion_config_t const* config,          //
    usearch_dense_traversal_stats_t* stats,                  //
    usearch_key_t* columns, size_t columns_stride,           //
    usearch_distance_t* products, size_t products_stride,    //
    size_t* counts, usearch_error_t* error);

/**
 *  @brief Optional per-call settings of @c usearch_dense_attend - a pure input, never written.
 *      Passing @c NULL derives every setting.
 */
USEARCH_EXPORT typedef struct usearch_dense_attention_config_t {
    /** @brief The @b optional candidate-pool breadth; zero derives. */
    size_t expansion_factor;
    /** @brief The @b optional logit scale, conventionally the inverse square root of the head
     *      dimensionality; zero derives it from the index's frozen geometry. */
    float score_scale;
    /** @brief The @b optional residual-mass threshold the walk stops at, the mass-denominated
     *      sibling of the radius a search terminates on; zero derives. */
    float mass_tolerance;
} usearch_dense_attention_config_t;

/**
 *  @brief Softmax-weighted attention of a strided query matrix over the indexed keys, folded
 *      into the graph walk in a single pass - no match list is materialized, and each value row
 *      is gathered exactly once, at the moment its key was scored.
 *  @ref dense_index_linear_algebra
 *
 *  The index holds the keys and @p values holds the value rows, addressed by key: an entry's key
 *  is its row offset into @p values, parked in the low bytes with the rest zeroed. Token
 *  positions already are such keys, so a context inserted in position order needs no mapping,
 *  and the publication watermark becomes the causal mask - a chunk of keys is attendable only
 *  once its insertion published.
 *
 *  Never writes to the graph, so any number of these may overlap with concurrent insertions -
 *  which is what streaming a growing context through this verb amounts to.
 *  @ref dense_index_concurrency
 *
 *  @param[in] index The handle holding one layer's keys for one key-value head.
 *  @param[in] queries Base pointer of the first query row; every query head sharing those keys
 *      is one more row here.
 *  @param[in] queries_count Number of query rows.
 *  @param[in] queries_stride Number of bytes between starts of consecutive query rows.
 *  @param[in] queries_scalar_kind The scalar type of the query rows.
 *  @param[in] values Base pointer of the first value row.
 *  @param[in] value_dimensions Number of dimensions per value row, independent of the key
 *      dimensionality the index froze.
 *  @param[in] values_stride Number of bytes between starts of consecutive value rows.
 *  @param[in] values_scalar_kind The scalar type of the value rows.
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] outputs Output matrix of `queries_count × value_dimensions` attended rows,
 *      already divided by their partitions; rows contiguous.
 *  @param[in] outputs_stride Number of bytes between starts of consecutive @p outputs rows.
 *  @param[in] outputs_scalar_kind The scalar type to write @p outputs as.
 *  @param[out] maximum_scores The @b optional array of @p queries_count running maxima;
 *      @c NULL skips reporting.
 *  @param[out] partitions The @b optional array of @p queries_count softmax denominators;
 *      @c NULL skips reporting. Reported together, the pair recombines this branch with another
 *      over the same queries - a dense recent window, or a second shard of one context - by the
 *      partial-softmax rescaling, which is the only reason to ask for either.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 */
USEARCH_EXPORT void usearch_dense_attend(                                       //
    usearch_dense_index_t index,                                                //
    void const* queries, size_t queries_count, size_t queries_stride,           //
    usearch_scalar_t queries_scalar_kind,                                       //
    void const* values, size_t value_dimensions, size_t values_stride,          //
    usearch_scalar_t values_scalar_kind,                                        //
    usearch_dense_attention_config_t const* config,                             //
    usearch_dense_traversal_stats_t* stats,                                     //
    void* outputs, size_t outputs_stride, usearch_scalar_t outputs_scalar_kind, //
    float* maximum_scores, float* partitions,                                   //
    usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_DENSE_INDEX_H
