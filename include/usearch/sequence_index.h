/**
 *  @file include/usearch/sequence_index.h
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The sequence-modality C interface: similarity search over variable-length sequences -
 *      textual documents, DNA, RNA, and protein strings - under Levenshtein, Needleman-Wunsch,
 *      Smith-Waterman, and Gotoh-affine scoring.
 *
 *  Variable-length payloads cannot ride a run of equal rows, so every batch input is a @b tape:
 *  the bytes of every sequence end to end, and @c count + 1 Apache Arrow offsets around them - the
 *  sequence-modality twin of the dense header's rows. One constructor serves both metric
 *  families: uniform edit distances need no parameters, weighted alignments take a substitution
 *  matrix and gap costs. Handles live in RAM, created empty by @c usearch_sequence_init_ram,
 *  copied from a checkpoint by @c usearch_sequence_load_ram, or served from one in place by
 *  @c usearch_sequence_view_ram, which is read-only.
 *
 *  @section sequence_index_concurrency Concurrency & Thread Safety
 *
 *  A handle may be shared across threads. On a mutable handle, insertions and searches overlap
 *  freely: searches read only entries published behind the visibility watermark and never
 *  write to the graph, while insertions synchronize through per-node locks and publish in
 *  order. A viewed handle admits arbitrarily many concurrent searches, having nothing to
 *  mutate. Serialization is the one exception: saving demands a mutation-free window,
 *  though concurrent readers keep searching throughout. Per-call outputs - counts and the
 *  outcome - belong to their call and must not be shared between calls that are not final.
 *
 *  An insertion is all-or-nothing per call, and its entries become visible at once, so a
 *  concurrent search sees every entry of the call or none of them.
 *
 *  Entries are permanent. No verb removes one, so an index only ever grows until it is
 *  cleared wholesale, and a key's entries accumulate up to the @c values_per_key bound its
 *  creation froze.
 *
 *  The executor, not the handle, is what orders: a handle admits any number of concurrent
 *  callers, while calls on one executor run in the order they were submitted.
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
 *  @see usearch_sequence_init_ram
 *  @see usearch_sequence_insert
 *  @see usearch_sequence_search
 */
typedef struct usearch_sequence_index_handle_t* usearch_sequence_index_t;

/** A consistent snapshot cursor over the index contents. Captures the publication
 *  watermark at creation, so concurrent inserts never appear mid-iteration, and the
 *  walk is a sequential entry scan - export, migration, and visualization feed off it.
 *  Each pull names the index again; an overlapping pull reports @c usearch_status_contended_k. */
typedef struct usearch_sequence_cursor_handle_t* usearch_sequence_cursor_t;

/**
 *  @brief The sequence scoring kinds supported out of the box.
 *
 *  The alignment kinds form a grid: global or local, against linear or affine gaps. All of
 *  them fill the same @b [|a|+1,|b|+1] table of cell @c D[i,j], the best score for the two
 *  prefixes ending at @c i and @c j, so every kind here is quadratic in time. Only the live width
 *  differs: the linear kinds carry one row, the affine kinds three.
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
typedef enum usearch_sequence_metric_t {
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
     *  cheaper than several short ones - the realistic model for an indel. Needs three live tracks:
     *  the aligned cell and one per gap direction. Both gap tracks reopen from the
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

/** The substitution matrix and gap costs an alignment kind scores with; the Levenshtein kinds
 *  score with unit costs and take none. */
typedef struct usearch_sequence_weights_t {

    /** Row-major 256×256 signed scores over byte alphabets. */
    int8_t const* substitution_matrix;

    /** Cost of opening a gap; the linear kinds use it as the flat gap cost. */
    int8_t gap_open;

    /** Cost of extending an open gap; ignored by the linear kinds. */
    int8_t gap_extend;
} usearch_sequence_weights_t;

/** The arguments of @c usearch_sequence_init_ram. */
typedef struct usearch_sequence_init_ram_t {

    /** Output for the new index. */
    usearch_sequence_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The scoring kind. */
    usearch_sequence_metric_t metric;

    /** The weights an alignment kind needs; @c NULL for the Levenshtein kinds, which refuse any. */
    usearch_sequence_weights_t const* weights;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Upper bound on entries sharing one key; zero derives to one - unique keys. */
    size_t values_per_key;

    /** The memory domains to place one shard on apiece. */
    usearch_ram_options_t ram;
} usearch_sequence_init_ram_t;

/** Creates an empty, mutable sequence index in host memory, for the uniform edit distances or
 *  the weighted alignments alike. */
USEARCH_EXPORT void usearch_sequence_init_ram(usearch_sequence_init_ram_t* arguments);

/** The arguments of @c usearch_sequence_load_ram. */
typedef struct usearch_sequence_load_ram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_sequence_index_t* index;

    /** The executor the read fans out over, one shard per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** The memory domains to place the parts on. */
    usearch_ram_options_t ram;

    /** Upper bound on entries sharing one key, which no checkpoint records; zero derives to one. */
    size_t values_per_key;

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_sequence_load_ram_t;

/**
 *  @brief Copies a checkpoint into a new, mutable index in host memory.
 *
 *  Where the parts end up is the call's own memory domains, never the manifest's. A part
 *  carries its own length in its header, so a second load skips whatever already arrived
 *  whole and cancelling costs the part being read rather than everything behind it.
 */
USEARCH_EXPORT void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments);

/** The arguments of @c usearch_sequence_view_ram. */
typedef struct usearch_sequence_view_ram_t {

    /** Output for the new, read-only index. */
    usearch_sequence_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;
} usearch_sequence_view_ram_t;

/** Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
 *  verbs report @c usearch_status_read_only_k. */
USEARCH_EXPORT void usearch_sequence_view_ram(usearch_sequence_view_ram_t* arguments);

/** Frees the resources associated with the index; @c NULL is safe, and it cannot fail, so takes
 *  no outcome.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_sequence_free(usearch_sequence_index_t index);

/** The arguments of @c usearch_sequence_reserve. */
typedef struct usearch_sequence_reserve_t {

    /** The handle to be resized. */
    usearch_sequence_index_t index;

    /** The executor the growth runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The desired total entry capacity, including the current size. */
    size_t entries;

    /** Expected total payload volume; zero lets the payload storage grow lazily. */
    size_t payload_bytes;
} usearch_sequence_reserve_t;

/** Reserves capacity for a number of incoming sequences ahead of insertion. */
USEARCH_EXPORT void usearch_sequence_reserve(usearch_sequence_reserve_t* arguments);

/** The arguments of @c usearch_sequence_clear. */
typedef struct usearch_sequence_clear_t {

    /** The handle to be cleared. */
    usearch_sequence_index_t index;

    /** The executor the erasure runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;
} usearch_sequence_clear_t;

/** Erases all entries, keeping the reserved capacity. */
USEARCH_EXPORT void usearch_sequence_clear(usearch_sequence_clear_t* arguments);

/** Traversal telemetry, shared by insertion and search - a pure output, fully overwritten by the
 *  call it is passed to. Passing @c NULL skips reporting. */
typedef struct usearch_sequence_traversal_stats_t {

    /** Graph members visited while resolving the batch. */
    size_t visited_members;

    /** Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;

    /** Starvation restarts - re-seedings after the frontier drained prematurely,
     *  and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;
} usearch_sequence_traversal_stats_t;

/** The arguments of @c usearch_sequence_insert. */
typedef struct usearch_sequence_insert_t {

    /** The handle to be populated. */
    usearch_sequence_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Sequences to admit - the batch axis of @c keys and @c sequences. */
    size_t count;

    /** The identities to admit the sequences under, one per sequence. */
    usearch_keys_view_t keys;

    /** The sequences, as bytes end to end around @c count + 1 offsets. */
    usearch_tape_view_t sequences;

    /** The @b optional candidate-pool breadth the build walk explores - the @c ef of the
     *  literature; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries admitted and the batch length. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_sequence_traversal_stats_t* stats;
} usearch_sequence_insert_t;

/**
 *  @brief Adds a tape of sequences under their keys - the native batch operation; a single
 *      sequence is a tape of one.
 *  @retval usearch_status_read_only_k The handle is a view.
 *  @retval usearch_status_overflow_k A key already holds @c values_per_key entries.
 *
 *  An insertion is all-or-nothing per call. Every fallible step runs before the first entry
 *  identity is claimed, so a rejected call publishes nothing - a cancelled one included, which
 *  reports @c usearch_status_cancelled_k - and the entries of an accepted one become visible at
 *  once.
 *
 *  A key already holding @c values_per_key entries rejects the whole call, since a smaller retry
 *  cannot help.
 */
USEARCH_EXPORT void usearch_sequence_insert(usearch_sequence_insert_t* arguments);

/** The arguments of @c usearch_sequence_refine. */
typedef struct usearch_sequence_refine_t {

    /** The handle to be refined. */
    usearch_sequence_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional output for the number of entries whose edges changed; zero means settled. */
    size_t* changed;

    /** The @b optional candidate-pool breadth the re-selection walk explores; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries swept; @c NULL skips it. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_sequence_traversal_stats_t* stats;
} usearch_sequence_refine_t;

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      edges that were chosen when the graph around them was still incomplete.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c expansion_factor belongs here rather than in @c usearch_sequence_insert. It pays most in this
 *  modality, where a single alignment is quadratic and a stingy insertion pass is tempting.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting an entry's
 *  edges is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved - nothing unwinds, and no @c usearch_status_cancelled_k is reported. The
 *  sweep resumes where it stopped, so calling under a time budget converges across calls.
 *
 *  Searches may run throughout. A viewed handle refuses with @c usearch_status_read_only_k, like
 *  every other mutating verb.
 */
USEARCH_EXPORT void usearch_sequence_refine(usearch_sequence_refine_t* arguments);

/** The arguments of @c usearch_sequence_search. */
typedef struct usearch_sequence_search_t {

    /** The handle to be queried. */
    usearch_sequence_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Queries to resolve; the outputs have @c count rows of @c matches.wanted entries. */
    size_t count;

    /** The queries, as bytes end to end around @c count + 1 offsets. */
    usearch_tape_view_t queries;

    /** Where the run lands. */
    usearch_matches_span_t matches;

    /** The @b optional candidate-pool breadth the walk explores - the @c ef of the
     *  literature; zero derives. */
    size_t expansion_factor;

    /** The @b optional per-key admission callback; @c NULL admits everything. */
    usearch_predicate_t predicate;

    /** Opaque pointer forwarded into every @c predicate call. */
    void* predicate_state;

    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_sequence_traversal_stats_t* stats;
} usearch_sequence_search_t;

/**
 *  @brief Performs k-Approximate Nearest Neighbors Search for a tape of queries - the native batch
 *      operation, parallelized internally; one query is a tape of one. Never writes to the graph,
 *      so any number may overlap with concurrent insertions. @ref sequence_index_concurrency
 *
 *  A @c predicate narrows the matches to the entries it admits. Non-matching entries still route
 *  the traversal rather than terminating it, so recall does not collapse at low selectivity - the
 *  walk crosses rejected regions to reach admitted ones behind them. The cost is that the
 *  predicate is called far more often than @c matches.wanted, once per scored candidate, so it
 *  belongs on the caller's fast path.
 *
 *  A cancelled search reports @c usearch_status_cancelled_k and leaves every output row
 *  indeterminate.
 */
USEARCH_EXPORT void usearch_sequence_search(usearch_sequence_search_t* arguments);

/** What one key holds, enough to size a buffer before retrieving any of it. */
typedef struct usearch_sequence_key_stats_t {

    /** Entries sharing the key; keys need not be unique. */
    size_t entries;

    /** Bytes summed across those entries. */
    size_t total_bytes;

    /** Bytes of the longest single entry. */
    size_t largest_bytes;
} usearch_sequence_key_stats_t;

/** The arguments of @c usearch_sequence_count. */
typedef struct usearch_sequence_count_t {

    /** The handle to be queried. */
    usearch_sequence_index_t index;

    /** The executor the lookup runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are counted. */
    usearch_key_t key;

    /** Output for what @c key holds; zero @c entries means the index holds none. */
    usearch_sequence_key_stats_t* key_stats;
} usearch_sequence_count_t;

/** Counts the entries stored under one key, with their payload extent. */
USEARCH_EXPORT void usearch_sequence_count(usearch_sequence_count_t* arguments);

/** The arguments of @c usearch_sequence_get. */
typedef struct usearch_sequence_get_t {

    /** The handle to be queried. */
    usearch_sequence_index_t index;

    /** The executor whose thread copies the entries. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The key whose entries are exported. */
    usearch_key_t key;

    /** Output for the tape's bytes inside @c arena, entries end to end, never terminated. */
    uint8_t** sequences;

    /** Output for @c written + 1 offsets into @c sequences, inside @c arena. */
    size_t** offsets;

    /** The @b optional output for the entries written; zero when the key is absent. */
    size_t* written;

    /** Memory for @c sequences and @c offsets: reused when it points to an arena, created when it
     *  points to @c NULL. */
    usearch_arena_t* arena;
} usearch_sequence_get_t;

/**
 *  @brief Retrieves every sequence stored under one key as an Apache Arrow tape, in insertion order -
 *      one call, however many and however long they turn out to be.
 *
 *  An absent key leaves both outputs @c NULL.
 */
USEARCH_EXPORT void usearch_sequence_get(usearch_sequence_get_t* arguments);

/** The arguments of @c usearch_sequence_merge. */
typedef struct usearch_sequence_merge_t {

    /** The handle absorbing the donors. */
    usearch_sequence_index_t index;

    /** The executor the merge fans out over, one donor per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Donor handles to absorb. */
    size_t count;

    /** Array of @c count handles, every one of them consumed and nulled as it goes. */
    usearch_sequence_index_t* donors;

    /** The @b optional handler, called with donors absorbed and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_sequence_merge_t;

/**
 *  @brief Absorbs every handle in @c donors into @c index - cheaper than re-inserting their
 *      sequences one by one, wiring cross-links via landmark stitching. Donor handles are
 *      consumed and must not be used afterwards, not even for @c usearch_sequence_free.
 *
 *  Readers of @c index may keep searching throughout, since unpublished entries hide behind
 *  the watermark. Absorbing a group costs the wall clock of its largest source rather than
 *  the sum of them, because the threads advance in parallel.
 *
 *  Cancellation stops @b between donors, which makes this the one verb where it is not
 *  all-or-nothing: donors already absorbed stay absorbed, and the rest stay live handles the
 *  caller still owns and must free. Every consumed slot is nulled as it goes, so the array itself
 *  says which handles survived.
 */
USEARCH_EXPORT void usearch_sequence_merge(usearch_sequence_merge_t* arguments);

/** The arguments of @c usearch_sequence_cursor_init. */
typedef struct usearch_sequence_cursor_init_t {

    /** Output for the new cursor; @c NULL on failure. */
    usearch_sequence_cursor_t* cursor;

    /** The handle to be scanned. */
    usearch_sequence_index_t index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;
} usearch_sequence_cursor_init_t;

/** Opens a snapshot cursor over the current contents; see @c usearch_sequence_cursor_t. */
USEARCH_EXPORT void usearch_sequence_cursor_init(usearch_sequence_cursor_init_t* arguments);

/** The arguments of @c usearch_sequence_cursor_next. */
typedef struct usearch_sequence_cursor_next_t {

    /** The handle the cursor was opened on. */
    usearch_sequence_index_t index;

    /** The executor the pull runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The cursor to advance. */
    usearch_sequence_cursor_t cursor;

    /** Where the identities land; its capacity bounds the batch. */
    usearch_keys_span_t keys;

    /** The @b optional output for up to @c keys.capacity borrowed views; @c NULL skips them. */
    usearch_bytes_view_t* sequences;

    /** Output for the entries written; zero when the snapshot is exhausted. */
    size_t* written;
} usearch_sequence_cursor_next_t;

/**
 *  @brief Copies the next batch of entries out of the cursor.
 *  @retval usearch_status_invalid_argument_k @c index is not the handle the cursor was opened on.
 *
 *  The views written into @c sequences borrow the index's own storage rather than copying, so
 *  they stay valid only until the cursor next advances or is freed. A caller keeping any of
 *  them past that point must copy the bytes out first.
 */
USEARCH_EXPORT void usearch_sequence_cursor_next(usearch_sequence_cursor_next_t* arguments);

/** Releases the cursor; @c NULL is safe.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_sequence_cursor_free(usearch_sequence_cursor_t cursor);

/** The arguments of @c usearch_sequence_save. */
typedef struct usearch_sequence_save_t {

    /** The handle to be written. */
    usearch_sequence_index_t index;

    /** The executor the write fans out over, one shard per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Destination path, whose extension picks the layout. */
    usearch_bytes_view_t path;

    /** The @b optional handler, called with parts written and their count. */
    usearch_progress_callback_t progress;

    /** Opaque pointer forwarded into every @c progress call. */
    void* progress_state;
} usearch_sequence_save_t;

/**
 *  @brief Saves a mutable index under @c path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 */
USEARCH_EXPORT void usearch_sequence_save(usearch_sequence_save_t* arguments);

/** The metric and geometry a checkpoint froze - everything a reader learns without opening it. */
typedef struct usearch_sequence_metadata_t {

    /** The scoring kind the checkpoint was built under. */
    usearch_sequence_metric_t metric;

    /** Connections per node the graph was built with. */
    size_t connectivity;
} usearch_sequence_metadata_t;

/** The arguments of @c usearch_sequence_metadata_read. */
typedef struct usearch_sequence_metadata_read_t {

    /** The executor the read runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** Output for every field the checkpoint froze. */
    usearch_sequence_metadata_t* metadata;
} usearch_sequence_metadata_read_t;

/** Reads a checkpoint's metric & geometry without loading it. */
USEARCH_EXPORT void usearch_sequence_metadata_read(usearch_sequence_metadata_read_t* arguments);

/** A live index's state. Each counter is read once, so under concurrent insertions they may come
 *  from slightly different moments. */
typedef struct usearch_sequence_index_description_t {

    /** The frozen metric and connectivity, as a checkpoint reports them. */
    usearch_sequence_metadata_t metadata;

    /** Stored entries. */
    size_t size;

    /** Entries hosted without growing again. */
    size_t capacity;

    /** Bytes the handle holds, payload storage included; traversal scratch is the executor's. */
    size_t memory_usage;

    /** Static codename of the kernels the index dispatched to on this machine. */
    char const* capabilities;
} usearch_sequence_index_description_t;

/** The arguments of @c usearch_sequence_describe. */
typedef struct usearch_sequence_describe_t {

    /** The handle to be described. */
    usearch_sequence_index_t index;

    /** The executor the report runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output, filled whole. */
    usearch_sequence_index_description_t* description;
} usearch_sequence_describe_t;

/** Reports a live index's frozen metadata and current counters in one call. */
USEARCH_EXPORT void usearch_sequence_describe(usearch_sequence_describe_t* arguments);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_SEQUENCE_INDEX_H
