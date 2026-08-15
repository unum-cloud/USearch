/**
 *  @brief The sequence-modality C interface: similarity search over variable-length
 *      sequences - textual documents, DNA, RNA, and protein strings - under Levenshtein,
 *      Needleman-Wunsch, Smith-Waterman, and Gotoh-affine scoring.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/sequence_index.h
 *
 *  Variable-length payloads cannot ride a strided matrix, so every batch input is a
 *  @b tape: one array of @c usearch_string_t views, each carrying its own base pointer and
 *  byte length - the sequence-modality twin of the dense header's strided-matrix
 *  convention. Two constructors split the metric families: uniform edit distances need no
 *  parameters, weighted alignments take a substitution matrix and gap costs.
 *
 *  @section sequence_index_concurrency Concurrency & Thread Safety
 *
 *  A handle may be shared across threads. On a RAM handle, insertions and searches overlap
 *  freely: searches read only entries published behind the visibility watermark and never
 *  write to the graph, while insertions synchronize through per-node locks and publish in
 *  order. An NVMe handle admits arbitrarily many concurrent searches, having nothing to
 *  mutate. Serialization is the one exception: saving demands a mutation-free window,
 *  though concurrent readers keep searching throughout. Per-call outputs - counts and the
 *  error pointer - belong to their call and must not be shared between in-flight calls.
 */
#ifndef UNUM_USEARCH_SEQUENCE_INDEX_H
#define UNUM_USEARCH_SEQUENCE_INDEX_H

#include "usearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Similarity Search Index over variable-length sequences.
 *
 *  @see usearch_sequence_init_uniform For byte-level and UTF-8 @b Levenshtein distances.
 *  @see usearch_sequence_init_weighted For @b Needleman-Wunsch, @b Smith-Waterman, and Gotoh variants.
 */
USEARCH_EXPORT typedef void* usearch_sequence_index_t;

/**
 *  @brief A consistent snapshot cursor over the index contents. Captures the publication
 *      watermark at creation, so concurrent inserts never appear mid-iteration, and the
 *      walk is a sequential entry scan - export, migration, and visualization feed off it.
 */
USEARCH_EXPORT typedef void* usearch_sequence_cursor_t;

/** @brief The sequence scoring kinds supported out of the box. */
USEARCH_EXPORT typedef enum usearch_sequence_metric_t {
    usearch_sequence_metric_unknown_k = 0,
    /** @brief Levenshtein edit distance over raw bytes. */
    usearch_sequence_metric_uniform_bytes_k = 1,
    /** @brief Levenshtein edit distance over UTF-8 code points. */
    usearch_sequence_metric_uniform_utf8_k = 2,
    /** @brief Global alignment with linear gap costs - Needleman-Wunsch. */
    usearch_sequence_metric_weighted_linear_k = 3,
    /** @brief Local alignment with affine gap costs - Smith-Waterman-Gotoh. */
    usearch_sequence_metric_weighted_affine_k = 4,
} usearch_sequence_metric_t;

/**
 *  @brief Construction & loading options of a sequence index. The @c residency field tags
 *      which union member is read; zeroed optional fields derive from the geometry.
 */
USEARCH_EXPORT typedef struct usearch_sequence_options_t {
    /** @brief The scoring kind; the constructor overload must match its family. */
    usearch_sequence_metric_t metric;
    /** @brief The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;
    /** @brief Selects the active member of the union below, and the meaning of "load". */
    usearch_residency_t residency;
    union {
        usearch_ram_options_t ram;
        usearch_nvme_options_t nvme;
    };
} usearch_sequence_options_t;

/**
 *  @brief Initializes a sequence index for the @b uniform edit-distance metrics - the
 *      byte-level and UTF-8 Levenshtein kinds, needing no scoring parameters.
 *  @param[in] options Pointer to the populated options; its metric must be a uniform kind.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_sequence_index_t usearch_sequence_init_uniform( //
    usearch_sequence_options_t const* options, usearch_error_t* error);

/**
 *  @brief Initializes a sequence index for the @b weighted alignment metrics - global
 *      Needleman-Wunsch with linear gaps or local Smith-Waterman-Gotoh with affine gaps.
 *  @param[in] options Pointer to the populated options; its metric must be a weighted kind.
 *  @param[in] substitution_matrix Row-major 256×256 signed scores over byte alphabets.
 *  @param[in] gap_open Cost of opening a gap; the linear kind uses it as the flat gap cost.
 *  @param[in] gap_extend Cost of extending an open gap; ignored by the linear kind.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_sequence_index_t usearch_sequence_init_weighted( //
    usearch_sequence_options_t const* options,                          //
    int8_t const* substitution_matrix,                                  //
    int8_t gap_open, int8_t gap_extend, usearch_error_t* error);

/**
 *  @brief Frees the resources associated with the index; cannot fail, so takes no error.
 */
USEARCH_EXPORT void usearch_sequence_free(usearch_sequence_index_t index);

/**
 *  @brief Reserves capacity for a number of incoming sequences ahead of insertion.
 *  @param[in] entries The desired total entry capacity, including the current size.
 *  @param[in] payload_bytes Expected total payload volume; zero lets the arena grow lazily.
 */
USEARCH_EXPORT void usearch_sequence_reserve( //
    usearch_sequence_index_t index, size_t entries, size_t payload_bytes, usearch_error_t* error);

/**
 *  @brief Erases all entries, keeping the reserved capacity.
 */
USEARCH_EXPORT void usearch_sequence_clear(usearch_sequence_index_t index, usearch_error_t* error);

/** @brief Reports the number of stored entries. */
USEARCH_EXPORT size_t usearch_sequence_size(usearch_sequence_index_t index, usearch_error_t* error);
/** @brief Reports the reserved capacity in entries. */
USEARCH_EXPORT size_t usearch_sequence_capacity(usearch_sequence_index_t index, usearch_error_t* error);
/** @brief Reports the frozen connections-per-node bound of the graph. */
USEARCH_EXPORT size_t usearch_sequence_connectivity(usearch_sequence_index_t index, usearch_error_t* error);
/** @brief Reports the memory usage of the index in bytes, payload arenas included. */
USEARCH_EXPORT size_t usearch_sequence_memory_usage(usearch_sequence_index_t index, usearch_error_t* error);
/** @brief Reports the capability codename the index dispatched to on this machine. */
USEARCH_EXPORT char const* usearch_sequence_capabilities(usearch_sequence_index_t index, usearch_error_t* error);

/**
 *  @brief Optional per-call settings, shared by insertion and both search forms - a pure
 *      input, never written. Passing @c NULL derives every setting.
 */
USEARCH_EXPORT typedef struct usearch_sequence_expansion_config_t {
    /** @brief The @b optional candidate-pool breadth - the `ef` of the literature; zero
     *      derives. */
    size_t expansion_factor;
} usearch_sequence_expansion_config_t;

/**
 *  @brief Traversal telemetry, shared by insertion and both search forms - a pure output,
 *      fully overwritten by the call it is passed to. Passing @c NULL skips reporting.
 */
USEARCH_EXPORT typedef struct usearch_sequence_traversal_stats_t {
    /** @brief Graph members visited while resolving the batch. */
    size_t visited_members;
    /** @brief Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;
    /** @brief Starvation restarts - re-seedings after the frontier drained prematurely,
     *      and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;
} usearch_sequence_traversal_stats_t;

/**
 *  @brief Adds a tape of sequences under their keys - the native batch operation; a single
 *      sequence is a tape of one.
 *
 *  @param[inout] index The handle to be populated.
 *  @param[in] keys Array of @p count keys, one per sequence.
 *  @param[in] sequences Array of @p count borrowed views, one per sequence.
 *  @param[in] count Number of sequences in the tape.
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 *  @return Number of successfully inserted sequences - equal to @p count unless an error
 *      stopped the batch early.
 */
USEARCH_EXPORT size_t usearch_sequence_add(                    //
    usearch_sequence_index_t index, usearch_key_t const* keys, //
    usearch_string_t const* sequences, size_t count,           //
    usearch_sequence_expansion_config_t const* config,         //
    usearch_sequence_traversal_stats_t* stats, usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a tape of queries - the
 *      native batch operation, parallelized internally; one query is a tape of one.
 *      Never writes to the graph, so any number may overlap with concurrent insertions.
 *  @ref sequence_index_concurrency
 *
 *  @param[in] index The handle to be queried.
 *  @param[in] queries Array of @p queries_count borrowed query views.
 *  @param[in] queries_count Number of queries in the tape.
 *  @param[in] wanted Upper bound on matches per query - the "k" in "kANN".
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] keys Output matrix of `queries_count × wanted` keys; rows contiguous.
 *  @param[in] keys_stride Number of bytes between starts of consecutive rows in @p keys.
 *  @param[out] distances Output matrix of `queries_count × wanted` scores; rows contiguous.
 *  @param[in] distances_stride Number of bytes between starts of consecutive @p distances rows.
 *  @param[out] counts Output array of @p queries_count per-query match counts.
 *  @param[out] error Populated with a status and a static message; the status is
 *      @c usearch_success_k when the call succeeded.
 */
USEARCH_EXPORT void usearch_sequence_search(                //
    usearch_sequence_index_t index,                         //
    usearch_string_t const* queries, size_t queries_count,  //
    size_t wanted,                                          //
    usearch_sequence_expansion_config_t const* config,      //
    usearch_sequence_traversal_stats_t* stats,              //
    usearch_key_t* keys, size_t keys_stride,                //
    usearch_distance_t* distances, size_t distances_stride, //
    size_t* counts, usearch_error_t* error);

/**
 *  @brief The filtered sibling of @c usearch_sequence_search: identical tape shape,
 *      admitting only entries the @p predicate approves. Non-matching entries still route
 *      the traversal, so recall does not collapse at low selectivity.
 *
 *  @param[in] predicate Per-key admission callback; returns non-zero to admit.
 *  @param[in] predicate_state Opaque pointer forwarded into every @p predicate call.
 */
USEARCH_EXPORT void usearch_sequence_search_if(             //
    usearch_sequence_index_t index,                         //
    usearch_string_t const* queries, size_t queries_count,  //
    size_t wanted,                                          //
    usearch_sequence_expansion_config_t const* config,      //
    usearch_sequence_traversal_stats_t* stats,              //
    usearch_predicate_t predicate, void* predicate_state,   //
    usearch_key_t* keys, size_t keys_stride,                //
    usearch_distance_t* distances, size_t distances_stride, //
    size_t* counts, usearch_error_t* error);

/** @brief What one key holds, enough to size a buffer before retrieving any of it. */
USEARCH_EXPORT typedef struct usearch_sequence_key_stats_t {
    /** @brief Entries sharing the key; keys need not be unique. */
    size_t entries;
    /** @brief Bytes summed across those entries. */
    size_t total_bytes;
    /** @brief Bytes of the longest single entry - the per-entry stride one @c get needs. */
    size_t largest_bytes;
} usearch_sequence_key_stats_t;

/**
 *  @brief Counts the entries stored under @p key; zero means the index holds none.
 *
 *  Pairing this with @c usearch_sequence_get is not atomic: an insertion between the two
 *  calls leaves the count stale. That is harmless, because entries are only ever added, so
 *  a stale count under-reports and @c usearch_sequence_get returns the true total - a caller
 *  seeing more than it asked for allocates again and converges.
 *
 *  @param[out] stats The @b optional extent of the key's payload; @c NULL asks for the
 *      count alone.
 *  @return The number of entries under @p key, mirroring @c stats->entries.
 */
USEARCH_EXPORT size_t usearch_sequence_count(          //
    usearch_sequence_index_t index, usearch_key_t key, //
    usearch_sequence_key_stats_t* stats, usearch_error_t* error);

/**
 *  @brief Retrieves the sequences stored under @p key into a caller-owned buffer.
 *
 *  Entries are written at @p stride spacing in insertion order and are never
 *  null-terminated, so sizing the buffer takes @c usearch_sequence_count first:
 *  @c entries slots of @c largest_bytes each.
 *
 *  @param[out] sequences Destination for up to @p wanted entries; @c NULL probes the count
 *      without copying.
 *  @param[in] wanted Upper bound on entries to write - the entry capacity of @p sequences.
 *  @param[in] stride Number of bytes between starts of consecutive entries in @p sequences.
 *  @param[out] lengths The @b optional per-entry true byte length, up to @p wanted of them;
 *      @c NULL skips it.
 *  @return The @b true number of entries under @p key, never the number written - a value
 *      above @p wanted means the copy was truncated, so allocate that many and call again.
 *      Zero when the key is absent.
 */
USEARCH_EXPORT size_t usearch_sequence_get(            //
    usearch_sequence_index_t index, usearch_key_t key, //
    char* sequences, size_t wanted, size_t stride, size_t* lengths, usearch_error_t* error);

/**
 *  @brief Absorbs @p donor into @p index - cheaper than re-inserting its sequences one by
 *      one, wiring cross-links via landmark stitching. The donor handle is consumed and
 *      must not be used afterwards, not even for @c usearch_sequence_free. Readers of
 *      @p index may keep searching throughout: unpublished entries hide behind the watermark.
 */
USEARCH_EXPORT void usearch_sequence_merge(usearch_sequence_index_t index, usearch_sequence_index_t donor,
                                           usearch_error_t* error);

/**
 *  @brief Opens a snapshot cursor over the current contents; see @c usearch_sequence_cursor_t.
 *  @return The cursor, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_sequence_cursor_t usearch_sequence_cursor(usearch_sequence_index_t index,
                                                                 usearch_error_t* error);

/**
 *  @brief Copies the next batch of entries out of the cursor.
 *  @param[out] keys Output array of up to @p wanted keys.
 *  @param[out] sequences Output array of up to @p wanted views borrowed from the index's
 *      arena, valid until the cursor moves or closes; @c NULL skips payload export.
 *  @param[in] wanted Upper bound on entries to write - the capacity of @p keys and
 *      @p sequences.
 *  @return Number of entries written; zero when the snapshot is exhausted.
 */
USEARCH_EXPORT size_t usearch_sequence_cursor_next(        //
    usearch_sequence_cursor_t cursor, usearch_key_t* keys, //
    usearch_string_t* sequences, size_t wanted, usearch_error_t* error);

/** @brief Releases the cursor. */
USEARCH_EXPORT void usearch_sequence_cursor_free(usearch_sequence_cursor_t cursor);

/**
 *  @brief Reports the expected serialized size of the index in bytes.
 */
USEARCH_EXPORT size_t usearch_sequence_serialized_length(usearch_sequence_index_t index, usearch_error_t* error);

/**
 *  @brief Saves the index under @p path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 */
USEARCH_EXPORT void usearch_sequence_save(usearch_sequence_index_t index, usearch_string_t path,
                                          usearch_error_t* error);

/**
 *  @brief Loads a checkpoint into an initialized handle, with the handle's residency
 *      deciding the behavior: materialize into RAM, or open in place on disk - the
 *      read-only form, where mutating verbs fail.
 */
USEARCH_EXPORT void usearch_sequence_load(usearch_sequence_index_t index, usearch_string_t path,
                                          usearch_error_t* error);

/**
 *  @brief Reads a checkpoint's metric & geometry metadata into @p options without loading it.
 */
USEARCH_EXPORT void usearch_sequence_metadata( //
    usearch_string_t path, usearch_sequence_options_t* options, usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_SEQUENCE_INDEX_H
