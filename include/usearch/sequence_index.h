/**
 *  @brief The sequence-modality C interface: similarity search over variable-length
 *      sequences - textual documents, DNA, RNA, and protein strings - under Levenshtein,
 *      Needleman-Wunsch, Smith-Waterman, and Gotoh-affine scoring.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/sequence_index.h
 *
 *  Variable-length payloads cannot ride a strided matrix, so every batch input is a
 *  @b tape: one array of base pointers and one of byte lengths, index-aligned - the
 *  sequence-modality twin of the dense header's strided-matrix convention. Two
 *  constructors split the metric families: uniform edit distances need no parameters,
 *  weighted alignments take a substitution matrix and gap costs.
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
USEARCH_EXPORT typedef void* usearch_sequence_t;

/**
 *  @brief A consistent snapshot cursor over the index contents, capturing the publication
 *      watermark at creation - concurrent inserts never appear mid-iteration.
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

/** @brief RAM-residency behavior of the sequence index; the only residency shipping first. */
USEARCH_EXPORT typedef struct usearch_sequence_ram_options_t {
    /** @brief Worker threads for construction & batched search; zero takes 7/8 of the
     *      logical cores. */
    size_t threads;
    /** @brief Expects a provisioned huge-page pool and fails loudly when it is missing. */
    bool huge_pages;
} usearch_sequence_ram_options_t;

/** @brief NVMe-residency behavior of the sequence index. */
USEARCH_EXPORT typedef struct usearch_sequence_nvme_options_t {
    /** @brief Directory hosting the index arena & checkpoints. Must outlive the handle. */
    char const* directory;
    /** @brief I/O granularity in bytes; zero probes the device. */
    uint32_t sector_bytes;
    /** @brief Submission ring depth; zero derives from the device's queue capabilities. */
    uint32_t queue_depth;
    /** @brief O_DIRECT-class page-cache bypass. */
    bool direct_io;
} usearch_sequence_nvme_options_t;

/**
 *  @brief Construction & loading options of a sequence index. The @c residency field tags
 *      which union member is read; zeroed optional fields derive from the geometry.
 */
USEARCH_EXPORT typedef struct usearch_sequence_options_t {
    /** @brief The scoring kind; the constructor overload must match its family. */
    usearch_sequence_metric_t metric;
    /** @brief The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;
    /** @brief The @b optional construction candidate-pool breadth; zero derives. */
    size_t expansion_add;
    /** @brief The @b optional search candidate-pool breadth; zero derives. */
    size_t expansion_search;
    /** @brief Selects the active member of the union below, and the meaning of "load". */
    usearch_residency_t residency;
    union {
        usearch_sequence_ram_options_t ram;
        usearch_sequence_nvme_options_t nvme;
    };
} usearch_sequence_options_t;

/**
 *  @brief Initializes a sequence index for the @b uniform edit-distance metrics - the
 *      byte-level and UTF-8 Levenshtein kinds, needing no scoring parameters.
 *  @param[in] options Pointer to the populated options; its metric must be a uniform kind.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_sequence_t usearch_sequence_init_uniform( //
    usearch_sequence_options_t const* options, usearch_error_t* error);

/**
 *  @brief Initializes a sequence index for the @b weighted alignment metrics - global
 *      Needleman-Wunsch with linear gaps or local Smith-Waterman-Gotoh with affine gaps.
 *  @param[in] options Pointer to the populated options; its metric must be a weighted kind.
 *  @param[in] substitution_matrix Row-major 256×256 signed scores over byte alphabets.
 *  @param[in] gap_open Cost of opening a gap; the linear kind uses it as the flat gap cost.
 *  @param[in] gap_extend Cost of extending an open gap; ignored by the linear kind.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_sequence_t usearch_sequence_init_weighted( //
    usearch_sequence_options_t const* options,                    //
    int8_t const* substitution_matrix,                            //
    int8_t gap_open, int8_t gap_extend, usearch_error_t* error);

/**
 *  @brief Frees the resources associated with the index.
 */
USEARCH_EXPORT void usearch_sequence_free(usearch_sequence_t index, usearch_error_t* error);

/**
 *  @brief Reserves capacity for a number of incoming sequences ahead of insertion.
 *  @param[in] entries The desired total entry capacity, including the current size.
 *  @param[in] payload_bytes Expected total payload volume; zero lets the arena grow lazily.
 */
USEARCH_EXPORT void usearch_sequence_reserve( //
    usearch_sequence_t index, size_t entries, size_t payload_bytes, usearch_error_t* error);

/**
 *  @brief Erases all entries, keeping the reserved capacity.
 */
USEARCH_EXPORT void usearch_sequence_clear(usearch_sequence_t index, usearch_error_t* error);

/** @brief Reports the number of stored entries. */
USEARCH_EXPORT size_t usearch_sequence_size(usearch_sequence_t index, usearch_error_t* error);
/** @brief Reports the reserved capacity in entries. */
USEARCH_EXPORT size_t usearch_sequence_capacity(usearch_sequence_t index, usearch_error_t* error);
/** @brief Reports the memory usage of the index in bytes, payload arenas included. */
USEARCH_EXPORT size_t usearch_sequence_memory_usage(usearch_sequence_t index, usearch_error_t* error);
/** @brief Reports the capability codename the index dispatched to on this machine. */
USEARCH_EXPORT char const* usearch_sequence_capabilities(usearch_sequence_t index, usearch_error_t* error);

/**
 *  @brief Adds a tape of sequences under their keys - the native batch operation; a single
 *      sequence is a tape of one.
 *
 *  @param[inout] index The handle to be populated.
 *  @param[in] keys Array of @p count keys, one per sequence.
 *  @param[in] sequences Array of @p count base pointers, one per sequence.
 *  @param[in] lengths Array of @p count byte lengths, index-aligned with @p sequences.
 *  @param[in] count Number of sequences in the tape.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 *  @return Number of successfully inserted sequences.
 */
USEARCH_EXPORT size_t usearch_sequence_add(              //
    usearch_sequence_t index, usearch_key_t const* keys, //
    char const* const* sequences, size_t const* lengths, //
    size_t count, usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a tape of queries - the
 *      native batch operation, parallelized internally; one query is a tape of one.
 *
 *  @param[in] index The handle to be queried.
 *  @param[in] queries Array of @p queries_count query base pointers.
 *  @param[in] lengths Array of @p queries_count query byte lengths.
 *  @param[in] queries_count Number of queries in the tape.
 *  @param[in] wanted Upper bound on matches per query - the "k" in "kANN".
 *  @param[out] keys Output matrix of `queries_count × wanted` keys; rows contiguous.
 *  @param[in] keys_stride Number of bytes between starts of consecutive rows in @p keys.
 *  @param[out] distances Output matrix of `queries_count × wanted` scores; rows contiguous.
 *  @param[in] distances_stride Number of bytes between starts of consecutive @p distances rows.
 *  @param[out] counts Output array of @p queries_count per-query match counts.
 *  @param[out] error Pointer to a string where an error message will be stored, on failure.
 */
USEARCH_EXPORT void usearch_sequence_search(                //
    usearch_sequence_t index,                               //
    char const* const* queries, size_t const* lengths,      //
    size_t queries_count, size_t wanted,                    //
    usearch_key_t* keys, size_t keys_stride,                //
    usearch_distance_t* distances, size_t distances_stride, //
    size_t* counts, usearch_error_t* error);

/**
 *  @brief The filtered sibling of @c usearch_sequence_search: identical tape shape,
 *      admitting only entries the @p predicate approves.
 */
USEARCH_EXPORT void usearch_sequence_search_if(             //
    usearch_sequence_t index,                               //
    char const* const* queries, size_t const* lengths,      //
    size_t queries_count, size_t wanted,                    //
    usearch_predicate_t predicate, void* predicate_state,   //
    usearch_key_t* keys, size_t keys_stride,                //
    usearch_distance_t* distances, size_t distances_stride, //
    size_t* counts, usearch_error_t* error);

/**
 *  @brief Checks if the index holds at least one entry under @p key.
 */
USEARCH_EXPORT bool usearch_sequence_contains(usearch_sequence_t index, usearch_key_t key, usearch_error_t* error);

/**
 *  @brief Counts the entries stored under @p key.
 */
USEARCH_EXPORT size_t usearch_sequence_count(usearch_sequence_t index, usearch_key_t key, usearch_error_t* error);

/**
 *  @brief Retrieves one sequence stored under @p key into a caller-owned buffer.
 *  @param[out] buffer Destination for the sequence bytes; not null-terminated.
 *  @param[in] capacity Size of @p buffer in bytes.
 *  @return The full byte length of the stored sequence - retry with a larger buffer when
 *      it exceeds @p capacity; zero when the key is absent.
 */
USEARCH_EXPORT size_t usearch_sequence_get(      //
    usearch_sequence_t index, usearch_key_t key, //
    char* buffer, size_t capacity, usearch_error_t* error);

/**
 *  @brief Absorbs @p donor into @p index without re-inserting its sequences one by one.
 *      The donor handle is consumed and must not be used afterwards. Readers of @p index
 *      may keep searching throughout.
 */
USEARCH_EXPORT void usearch_sequence_merge(usearch_sequence_t index, usearch_sequence_t donor, usearch_error_t* error);

/**
 *  @brief Opens a snapshot cursor over the current contents.
 *  @return The cursor, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_sequence_cursor_t usearch_sequence_entries(usearch_sequence_t index, usearch_error_t* error);

/**
 *  @brief Steps the cursor to the next entry, exposing its key and payload without copying.
 *  @param[out] key The entry's key.
 *  @param[out] sequence Borrowed pointer into the index's arena; valid until the cursor
 *      moves or closes.
 *  @param[out] length The sequence's byte length.
 *  @return Non-zero while an entry was produced; zero when the snapshot is exhausted.
 */
USEARCH_EXPORT int usearch_sequence_entries_next(         //
    usearch_sequence_cursor_t cursor, usearch_key_t* key, //
    char const** sequence, size_t* length, usearch_error_t* error);

/** @brief Releases the cursor. */
USEARCH_EXPORT void usearch_sequence_entries_free(usearch_sequence_cursor_t cursor);

/**
 *  @brief Reports the expected serialized size of the index in bytes.
 */
USEARCH_EXPORT size_t usearch_sequence_serialized_length(usearch_sequence_t index, usearch_error_t* error);

/**
 *  @brief Saves the index under a checkpoint path.
 */
USEARCH_EXPORT void usearch_sequence_save(usearch_sequence_t index, char const* path, usearch_error_t* error);

/**
 *  @brief Loads a checkpoint into an initialized handle, per the handle's residency.
 */
USEARCH_EXPORT void usearch_sequence_load(usearch_sequence_t index, char const* path, usearch_error_t* error);

/**
 *  @brief Creates a read-only view of a checkpoint without copying it into memory.
 */
USEARCH_EXPORT void usearch_sequence_view(usearch_sequence_t index, char const* path, usearch_error_t* error);

/**
 *  @brief Reads a checkpoint's metric & geometry metadata into @p options without loading.
 */
USEARCH_EXPORT void usearch_sequence_metadata( //
    char const* path, usearch_sequence_options_t* options, usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_SEQUENCE_INDEX_H
