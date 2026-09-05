/**
 *  @brief The sequence-modality C interface: similarity search over variable-length
 *      sequences - textual documents, DNA, RNA, and protein strings - under Levenshtein,
 *      Needleman-Wunsch, Smith-Waterman, and Gotoh-affine scoring.
 *
 *  @author Ash Vardanian
 *  @file include/usearch/sequence_index.h
 *
 *  Variable-length payloads cannot ride a strided matrix, so every batch input is a
 *  @b tape: one array of @c usearch_bytes_view_t views, each carrying its own base pointer and
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
 *
 *  Entries are permanent. No verb removes one, so an index only ever grows until it is
 *  cleared wholesale, and a key's entries accumulate up to the @c values_per_key bound its
 *  options froze.
 *
 *  The executor, not the handle, is what serializes: a handle admits any number of
 *  concurrent callers, while an executor admits @b one driver at a time and reports
 *  @c usearch_error_contended_k to the second.
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
 *  @see usearch_sequence_init_weighted For @b Needleman-Wunsch, @b Smith-Waterman and Gotoh.
 */
USEARCH_EXPORT typedef void* usearch_sequence_index_t;

/**
 *  @brief A consistent snapshot cursor over the index contents. Captures the publication
 *      watermark at creation, so concurrent inserts never appear mid-iteration, and the
 *      walk is a sequential entry scan - export, migration, and visualization feed off it.
 */
USEARCH_EXPORT typedef void* usearch_sequence_cursor_t;

/**
 *  @brief The sequence scoring kinds supported out of the box.
 *
 *  The alignment kinds form a grid: global or local, against linear or affine gaps. All of
 *  them fill the same @b [|a|+1,|b|+1] table of cell @c D[i,j], the best score for the two
 *  prefixes ending at @c i and @c j, so every kind here is quadratic in time. Only the width
 *  in flight differs: the linear kinds carry one row, the affine kinds three.
 *
 *  Only the global half can weigh the index's own edges. A local score is blind to everything
 *  outside its matched window, so every superstring of a sequence scores as that sequence
 *  does and an index built on one would collapse distinct entries onto a single point. Each
 *  local kind is therefore served from an index built under its own global counterpart, which
 *  shares its costs and bounds it - picked by the library and never named by the caller.
 *
 *  The global kinds minimize a cost, writing @c c for the substitution cost and @c g, @c o,
 *  @c e for the flat, opening and extending gap costs. The local kinds maximize a similarity,
 *  writing @c w for the substitution score, so a matrix admissible for one is not admissible
 *  for the other with its signs unchanged.
 */
USEARCH_EXPORT typedef enum usearch_sequence_metric_t {
    usearch_sequence_metric_unknown_k = 0,
    /**
     *  @brief Levenshtein edit distance over raw bytes.
     *
     *  The number of single-byte insertions, deletions and substitutions turning one sequence
     *  into the other, every edit costing one. A true metric, and the cheapest kind here.
     *
     *  D[i,j] = min( D[i-1,j] + 1, D[i,j-1] + 1, D[i-1,j-1] + (a[i] ≠ b[j]) )
     *
     *  @code{.py}
     *  row = list(range(len(b) + 1))                # one [|b|+1] row is the whole working set
     *  for i, x in enumerate(a, 1):
     *      row, prev = [i] + [0] * len(b), row
     *      for j, y in enumerate(b, 1):
     *          row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (x != y))
     *  levenshtein = row[-1]
     *  @endcode
     */
    usearch_sequence_metric_levenshtein_bytes_k = 1,
    /**
     *  @brief Levenshtein edit distance over UTF-8 code points.
     *
     *  The same recurrence, run over decoded code points rather than bytes, so one accented
     *  character costs one edit instead of the two or three its encoding occupies.
     */
    usearch_sequence_metric_levenshtein_utf8_k = 2,
    /**
     *  @brief Global alignment with linear gap costs - Needleman-Wunsch.
     *
     *  Levenshtein with the unit costs replaced by a substitution matrix and a flat gap cost,
     *  so a mismatch between similar symbols can be cheaper than one between dissimilar ones.
     *  Both sequences are aligned end to end, and every unaligned symbol is charged.
     *
     *  D[i,j] = min( D[i-1,j] + g, D[i,j-1] + g, D[i-1,j-1] + c(a[i], b[j]) )
     *
     *  @code{.py}
     *  row = [j * g for j in range(len(b) + 1)]
     *  for i, x in enumerate(a, 1):
     *      row, prev = [i * g] + [0] * len(b), row
     *      for j, y in enumerate(b, 1):
     *          row[j] = min(prev[j] + g, row[j - 1] + g, prev[j - 1] + c[x][y])
     *  needleman_wunsch = row[-1]
     *  @endcode
     */
    usearch_sequence_metric_needleman_wunsch_k = 3,
    /**
     *  @brief Local alignment with affine gap costs - Smith-Waterman-Gotoh. Ranks only.
     *
     *  The affine recurrence under a zero floor, reporting the best cell anywhere in the
     *  table rather than the corner. The floor is the blindness: a prefix or suffix that
     *  aligns badly is dropped at no cost instead of being charged.
     *
     *  S[i,j] = max( 0, X[i,j], Y[i,j], S[i-1,j-1] + w(a[i], b[j]) ),  score = max_{i,j} S[i,j]
     *
     *  @code{.py}
     *  # the needleman_wunsch_gotoh loop with max for min, w for c, and a 0 floor on S[i][j],
     *  # returning max(S) over every cell rather than the last one
     *  @endcode
     */
    usearch_sequence_metric_smith_waterman_gotoh_k = 4,
    /**
     *  @brief Global alignment with affine gap costs - Needleman-Wunsch-Gotoh.
     *
     *  Charges a gap run @c o for opening and @c e for each symbol after, so one long gap is
     *  cheaper than several short ones - the realistic model for an indel. Needs three tracks
     *  in flight: the aligned cell and one per gap direction. Both gap tracks reopen from the
     *  combined cell rather than from the aligned one, which is what lets a deletion become
     *  an insertion directly and keeps the kind a metric.
     *
     *  X[i,j] = min( D[i-1,j] + o, X[i-1,j] + e )
     *  Y[i,j] = min( D[i,j-1] + o, Y[i,j-1] + e )
     *  D[i,j] = min( D[i-1,j-1] + c(a[i], b[j]), X[i,j], Y[i,j] )
     *
     *  @code{.py}
     *  D = [0 if j == 0 else o + (j - 1) * e for j in range(len(b) + 1)]
     *  X = [inf] * (len(b) + 1)                     # gap in b, carried down the column
     *  for i, x in enumerate(a, 1):
     *      D, prev, Y = [o + (i - 1) * e] + [0] * len(b), D, inf
     *      for j, y in enumerate(b, 1):
     *          X[j] = min(prev[j] + o, X[j] + e)
     *          Y = min(D[j - 1] + o, Y + e)         # gap in a, carried along the row
     *          D[j] = min(prev[j - 1] + c[x][y], X[j], Y)
     *  gotoh = D[-1]
     *  @endcode
     */
    usearch_sequence_metric_needleman_wunsch_gotoh_k = 5,
    /**
     *  @brief Local alignment with linear gap costs - Smith-Waterman. Ranks only.
     *
     *  Needleman-Wunsch under a zero floor, reporting the best cell anywhere in the table.
     *  Finds the single best-matching window and charges nothing for what surrounds it.
     *
     *  S[i,j] = max( 0, S[i-1,j] − g, S[i,j-1] − g, S[i-1,j-1] + w(a[i], b[j]) )
     *
     *  @code{.py}
     *  best, row = 0, [0] * (len(b) + 1)
     *  for i, x in enumerate(a, 1):
     *      row, prev = [0] * (len(b) + 1), row
     *      for j, y in enumerate(b, 1):
     *          row[j] = max(0, prev[j] - g, row[j - 1] - g, prev[j - 1] + w[x][y])
     *          best = max(best, row[j])             # the 0 floor is the blindness
     *  smith_waterman = best
     *  @endcode
     */
    usearch_sequence_metric_smith_waterman_k = 6,
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
    /** @brief Upper bound on entries sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;
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
 *  @param[in] options Pointer to the populated options; its metric must be a Levenshtein kind.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return A handle to the initialized index, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_sequence_index_t usearch_sequence_init_uniform( //
    usearch_sequence_options_t const* options, usearch_error_t* error);

/**
 *  @brief Initializes a sequence index for the @b weighted alignment metrics - global
 *      Needleman-Wunsch with linear gaps or local Smith-Waterman-Gotoh with affine gaps.
 *  @param[in] options Pointer to the populated options; its metric must be an alignment kind.
 *  @param[in] substitution_matrix Row-major 256×256 signed scores over byte alphabets.
 *  @param[in] gap_open Cost of opening a gap; the linear kind uses it as the flat gap cost.
 *  @param[in] gap_extend Cost of extending an open gap; ignored by the linear kind.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
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
 *  @param[in] executor Lanes and scratch to fan out over; its kind must match the residency.
 *  @param[in] keys Array of @p count keys, one per sequence.
 *  @param[in] sequences Array of @p count borrowed views, one per sequence.
 *  @param[in] count Number of sequences in the tape.
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[in] progress The @b optional handler, called with entries admitted and @p count.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return @p count on success, zero on failure.
 *
 *  A batch is all-or-nothing. Every fallible step runs before the first entry identity is
 *  claimed, so a rejected batch publishes nothing - a cancelled one included, which reports
 *  @c usearch_error_cancelled_k. A second thread driving @p executor reports
 *  @c usearch_error_contended_k before any work begins.
 *
 *  A key already holding @c values_per_key entries rejects the whole batch with
 *  @c usearch_error_overflow_k, since a smaller retry cannot help.
 */
USEARCH_EXPORT size_t usearch_sequence_add(                      //
    usearch_sequence_index_t index, usearch_executor_t executor, //
    usearch_key_t const* keys,                                   //
    usearch_bytes_view_t const* sequences, size_t count,         //
    usearch_sequence_expansion_config_t const* config,           //
    usearch_progress_callback_t progress, void* progress_state,  //
    usearch_sequence_traversal_stats_t* stats,                   //
    usearch_error_t* error);

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      edges that were chosen when the graph around them was still incomplete.
 *
 *  @param[inout] index The handle to be refined.
 *  @param[in] executor Lanes and scratch to fan out over; its kind must match the residency.
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[in] progress The @b optional handler, called with entries swept; @c NULL skips it.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return Number of entries whose edges changed; zero means the graph has settled.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c expansion_factor belongs here rather than in @c usearch_sequence_add. It pays most in this
 *  modality, where a single alignment is quadratic and a stingy insertion pass is tempting.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting an entry's
 *  edges is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved - nothing unwinds, and no @c usearch_error_cancelled_k is reported. The
 *  sweep resumes where it stopped, so calling under a time budget converges across calls.
 *
 *  Searches may run throughout. A read-only handle refuses with @c usearch_error_read_only_k, like
 *  every other mutating verb.
 */
USEARCH_EXPORT size_t usearch_sequence_refine(                   //
    usearch_sequence_index_t index, usearch_executor_t executor, //
    usearch_sequence_expansion_config_t const* config,           //
    usearch_progress_callback_t progress, void* progress_state,  //
    usearch_sequence_traversal_stats_t* stats,                   //
    usearch_error_t* error);

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a tape of queries - the
 *      native batch operation, parallelized internally; one query is a tape of one.
 *      Never writes to the graph, so any number may overlap with concurrent insertions.
 *  @ref sequence_index_concurrency
 *
 *  @param[in] index The handle to be queried.
 *  @param[in] executor Lanes and scratch to fan out over; its kind must match the residency.
 *  @param[in] queries Array of @p queries_count borrowed query views.
 *  @param[in] queries_count Number of queries in the tape.
 *  @param[in] wanted Upper bound on matches per query - the "k" in "kANN".
 *  @param[in] config The @b optional per-call settings; @c NULL derives every setting.
 *  @param[in] progress The @b optional handler, called with queries resolved and their count.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] keys Output matrix of `queries_count × wanted` keys; rows contiguous.
 *  @param[in] keys_stride Number of bytes between starts of consecutive rows in @p keys.
 *  @param[out] distances Output matrix of `queries_count × wanted` scores; rows contiguous.
 *  @param[in] distances_stride Number of bytes between starts of consecutive @p distances rows.
 *  @param[out] counts Output array of @p queries_count per-query match counts.
 *  @param[out] stats The @b optional traversal telemetry; @c NULL skips reporting.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *
 *  A cancelled search reports @c usearch_error_cancelled_k and leaves every output row
 *  indeterminate. A second thread driving @p executor reports @c usearch_error_contended_k
 *  before any work begins.
 */
USEARCH_EXPORT void usearch_sequence_search(                     //
    usearch_sequence_index_t index, usearch_executor_t executor, //
    usearch_bytes_view_t const* queries, size_t queries_count,   //
    size_t wanted,                                               //
    usearch_sequence_expansion_config_t const* config,           //
    usearch_progress_callback_t progress, void* progress_state,  //
    usearch_key_t* keys, size_t keys_stride,                     //
    usearch_distance_t* distances, size_t distances_stride,      //
    size_t* counts,                                              //
    usearch_sequence_traversal_stats_t* stats,                   //
    usearch_error_t* error);

/**
 *  @brief The filtered sibling of @c usearch_sequence_search: identical tape shape, admitting
 *      only entries the @p predicate approves. Every parameter it shares carries the meaning
 *      documented there.
 *
 *  @param[in] predicate Per-key admission callback; returns non-zero to admit.
 *  @param[in] predicate_state Opaque pointer forwarded into every @p predicate call.
 *
 *  Non-matching entries still route the traversal rather than terminating it, so recall does
 *  not collapse at low selectivity - the walk crosses rejected regions to reach admitted ones
 *  behind them. The cost is that @p predicate is called far more often than @p wanted, once
 *  per scored candidate, so it belongs on the caller's fast path.
 */
USEARCH_EXPORT void usearch_sequence_search_if(                  //
    usearch_sequence_index_t index, usearch_executor_t executor, //
    usearch_bytes_view_t const* queries, size_t queries_count,   //
    size_t wanted,                                               //
    usearch_sequence_expansion_config_t const* config,           //
    usearch_predicate_t predicate, void* predicate_state,        //
    usearch_progress_callback_t progress, void* progress_state,  //
    usearch_key_t* keys, size_t keys_stride,                     //
    usearch_distance_t* distances, size_t distances_stride,      //
    size_t* counts,                                              //
    usearch_sequence_traversal_stats_t* stats,                   //
    usearch_error_t* error);

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
 *  @param[out] stats The @b optional payload extent; @c NULL asks for the count alone.
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
 *  @param[out] sequences Destination for up to @p wanted entries; @c NULL probes only.
 *  @param[in] wanted Upper bound on entries to write - the entry capacity of @p sequences.
 *  @param[in] stride Number of bytes between starts of consecutive entries in @p sequences.
 *  @param[out] lengths The @b optional per-entry true byte length; @c NULL skips it.
 *  @return The @b true number of entries under @p key; zero when absent.
 *
 *  The return is never the number written. A value above @p wanted means the copy was
 *  truncated, so allocate that many and call again.
 */
USEARCH_EXPORT size_t usearch_sequence_get(            //
    usearch_sequence_index_t index, usearch_key_t key, //
    char* sequences, size_t wanted, size_t stride, size_t* lengths, usearch_error_t* error);

/**
 *  @brief Absorbs every handle in @p donors into @p index - cheaper than re-inserting their
 *      sequences one by one, wiring cross-links via landmark stitching. Donor handles are
 *      consumed and must not be used afterwards, not even for @c usearch_sequence_free.
 *
 *  @param[inout] index The handle absorbing the donors.
 *  @param[in] executor The threads and scratch the merge fans out over, one donor per thread.
 *  @param[in] donors Array of @p donors_count handles, every one of them consumed.
 *  @param[in] donors_count Number of entries in @p donors.
 *  @param[in] progress The @b optional handler, called with donors absorbed and their count.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *
 *  Readers of @p index may keep searching throughout, since unpublished entries hide behind
 *  the watermark. Absorbing a group costs the wall clock of its largest source rather than
 *  the sum of them, because the threads advance in parallel.
 *
 *  Cancellation stops @b between donors, which makes this the one verb where it is not
 *  all-or-nothing: donors already absorbed stay absorbed, and the rest stay live handles the
 *  caller still owns and must free.
 */
USEARCH_EXPORT void usearch_sequence_merge(                      //
    usearch_sequence_index_t index, usearch_executor_t executor, //
    usearch_sequence_index_t const* donors, size_t donors_count, //
    usearch_progress_callback_t progress, void* progress_state,  //
    usearch_error_t* error);

/**
 *  @brief Opens a snapshot cursor over the current contents; see @c usearch_sequence_cursor_t.
 *  @return The cursor, or @c NULL on failure.
 */
USEARCH_EXPORT usearch_sequence_cursor_t usearch_sequence_cursor(usearch_sequence_index_t index,
                                                                 usearch_error_t* error);

/**
 *  @brief Copies the next batch of entries out of the cursor.
 *
 *  @param[inout] cursor The cursor to advance.
 *  @param[out] keys Output array of up to @p wanted keys.
 *  @param[out] sequences Output array of up to @p wanted borrowed views; @c NULL skips.
 *  @param[in] wanted Upper bound on entries - the capacity of @p keys and @p sequences.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *  @return Number of entries written; zero when the snapshot is exhausted.
 *
 *  The views written into @p sequences borrow the index's own arena rather than copying, so
 *  they stay valid only until the cursor next advances or is freed. A caller keeping any of
 *  them past that point must copy the bytes out first.
 */
USEARCH_EXPORT size_t usearch_sequence_cursor_next(        //
    usearch_sequence_cursor_t cursor, usearch_key_t* keys, //
    usearch_bytes_view_t* sequences, size_t wanted, usearch_error_t* error);

/** @brief Releases the cursor. */
USEARCH_EXPORT void usearch_sequence_cursor_free(usearch_sequence_cursor_t cursor);

/** @brief Bytes one monolith takes under the default layout; a manifest's parts sum to about the same. */
USEARCH_EXPORT size_t usearch_sequence_storage_requirements(usearch_sequence_index_t index, usearch_error_t* error);

/**
 *  @brief Saves the index under @p path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  @param[in] index The handle to be written.
 *  @param[in] executor The threads and scratch the write fans out over, one shard per thread.
 *  @param[in] path Destination path, whose extension picks the layout.
 *  @param[in] progress The @b optional handler, called with parts written and their count.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 */
USEARCH_EXPORT void usearch_sequence_save(                       //
    usearch_sequence_index_t index, usearch_executor_t executor, //
    usearch_bytes_view_t path,                                   //
    usearch_progress_callback_t progress, void* progress_state,  //
    usearch_error_t* error);

/**
 *  @brief Loads a checkpoint into an initialized handle, with the handle's residency
 *      deciding the behavior: materialize into RAM, or open in place on disk - the
 *      read-only form, where mutating verbs fail.
 *
 *  @param[inout] index The handle to be populated.
 *  @param[in] executor The threads and scratch the read fans out over, one shard per thread.
 *  @param[in] path Checkpoint path - a monolith, or a manifest naming many.
 *  @param[in] progress The @b optional handler, called with parts placed and their count.
 *  @param[in] progress_state Opaque pointer forwarded into every @p progress call.
 *  @param[out] error Status and a static message; @c usearch_success_k on success.
 *
 *  Where the parts end up is the handle's own memory domains, never the manifest's. A part
 *  carries its own length in its header, so a second load skips whatever already arrived
 *  whole and cancelling costs the part in flight rather than everything behind it.
 */
USEARCH_EXPORT void usearch_sequence_load(                       //
    usearch_sequence_index_t index, usearch_executor_t executor, //
    usearch_bytes_view_t path,                                   //
    usearch_progress_callback_t progress, void* progress_state,  //
    usearch_error_t* error);

/**
 *  @brief Reads a checkpoint's metric & geometry metadata into @p options without loading it.
 */
USEARCH_EXPORT void usearch_sequence_metadata( //
    usearch_bytes_view_t path, usearch_sequence_options_t* options, usearch_error_t* error);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_SEQUENCE_INDEX_H
