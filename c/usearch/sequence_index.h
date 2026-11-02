/**
 *  @file c/usearch/sequence_index.h
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The sequence-modality C interface: similarity search over variable-length sequences -
 *      textual documents, DNA, RNA, and protein strings - under Levenshtein, window overlap, and
 *      Needleman-Wunsch scoring.
 *
 *  Variable-length payloads cannot ride a run of equal rows, so every batch input is a @b tape:
 *  the bytes of every sequence end to end, and @c count + 1 Apache Arrow offsets around them - the
 *  sequence-modality twin of the dense header's rows. One constructor serves every metric family:
 *  uniform edit distances need no parameters, window overlap takes its window widths, weighted
 *  alignments take a substitution matrix and gap costs. Handles are created by a lifecycle verb
 *  naming both an action and a residency:
 *
 *      Action                           RAM          VRAM         NVMe
 *      init: start empty, mutable       init_ram     init_vram    none
 *      load: copy a checkpoint in       load_ram     load_vram    none
 *      view: serve one in place         view_ram     none         view_nvme
 *      save: write a checkpoint         save         save         none
 *
 *  Each is spelled @c usearch_sequence_ and the suffix above. A VRAM handle serves byte and UTF-8
 *  Levenshtein, window overlap and Needleman-Wunsch, on one GPU, through the insertion, search,
 *  count and description verbs, refusing the rest with @c usearch_status_unsupported_k, and a
 *  search there takes no @c predicate.
 *
 *  A viewed handle is read-only, refusing every mutating verb with @c usearch_status_read_only_k.
 *  An NVMe handle answers every read verb as a RAM handle does, its entries read through the
 *  executor's engines, and saving answers @c usearch_status_unsupported_k, since a view already is
 *  its checkpoint.
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
 *  concurrent search sees every entry of the call or none of them. Their keys land right after,
 *  so a key lookup only finds entries a search already sees.
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
 *  @sa usearch_sequence_init_ram
 *  @sa usearch_sequence_insert
 *  @sa usearch_sequence_search
 */
typedef struct usearch_sequence_index_handle_t* usearch_sequence_index_t;

/** A consistent snapshot cursor over the graph. Captures the publication watermark at creation,
 *  so concurrent inserts never appear mid-iteration, and yields each entry's key with the keys it
 *  links to - every entry as stored, or breadth-first from start keys - which export, migration,
 *  and visualization feed off. Each pull names the index again; an overlapping pull reports
 *  @c usearch_status_contended_k. */
typedef struct usearch_sequence_cursor_handle_t* usearch_sequence_cursor_t;

/**
 *  @brief One refinement in flight, optionally absorbing donors or splitting into receivers first.
 *
 *  It holds everything the pass needs between calls, so a cancelled
 *  @c usearch_sequence_refinement_next continues where it stopped, and nothing about it stays in
 *  the index once it finishes. The index admits no insertion and no save while the handle lives;
 *  searches proceed as during @c usearch_sequence_refine. Each call names the index and the
 *  executor it was opened with.
 */
typedef struct usearch_sequence_refinement_handle_t* usearch_sequence_refinement_t;

/**
 *  @brief The sequence scoring kinds supported out of the box.
 *
 *  The edit kinds fill a @b [|a|+1,|b|+1] table of cell @c D[i,j], the least cost of turning the
 *  prefix of @c a ending at @c i into the prefix of @c b ending at @c j, carrying one row, so each
 *  is quadratic in time. They write @c c for the substitution cost and @c g for the gap cost.
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
     *  @brief Window overlap: the share of the larger window set the other side misses.
     *
     *  Every window of every configured width is taken from both sequences, and a window of the
     *  stored @c b is a hit when the query @c a holds the same bytes at the same width. The
     *  distance is the misses over whichever side has more windows: zero where the windows
     *  coincide at every width, one where they share none. Exact at every radius; the window
     *  widths are in bytes, fixed at creation and recorded in the checkpoint.
     *
     *  d(a, b) = (max(|W(a)|, |W(b)|) − |{ v ∈ W(b) : v ∈ W(a) }|) / max(|W(a)|, |W(b)|)
     *
     *  @code{.py}
     *  windows = lambda s: [(w, s[i:i + w]) for w in widths for i in range(len(s) - w + 1)]
     *  query, stored = windows(a), windows(b)       # repeats kept, every width at once
     *  hits = sum(window in set(query) for window in stored)
     *  longer = max(len(query), len(stored))
     *  overlap = (longer - hits) / longer if longer else 1
     *  @endcode
     */
    usearch_sequence_metric_overlap_k = 4,
} usearch_sequence_metric_t;

/** The spelling a checkpoint records for a kind, as
 *  @c usearch_sequence_metric_levenshtein_bytes_k spells "levenshtein"; "unknown" for a value
 *  outside the set. */
USEARCH_EXPORT char const* usearch_sequence_metric_name(usearch_sequence_metric_t metric);

/** Reads a kind back from its spelling; @c usearch_sequence_metric_unknown_k when unrecognized. */
USEARCH_EXPORT usearch_sequence_metric_t usearch_sequence_metric_from_name(usearch_bytes_view_t name);

/**
 *  @brief The substitution matrix and gap costs an alignment kind scores with; the Levenshtein
 *      kinds score with unit costs and take none.
 *
 *  Needleman-Wunsch reads the matrix as similarities @c s between class codes, the classes ending
 *  at the last code scoring anything, at most 32 of them, and its sequences as those codes packed
 *  two bits apiece up to four classes and five past, little-endian behind one byte counting the
 *  slots the last bytes leave unused. It scores with @c c(a,b) = (s(a,a) + s(b,b)) / 2 − s(a,b),
 *  costs that must not be negative, rounded to integers of at most 15 together with the gap cost,
 *  all scaled down by one factor when either passes it.
 */
typedef struct usearch_sequence_weights_t {

    /** Row-major 256×256 signed scores over byte alphabets. */
    int8_t const* substitution_matrix;

    /** Cost of one unaligned symbol. */
    int8_t gap;
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

    /** Window widths in bytes for the overlap kind, which needs at least one; every other kind
     *  refuses any. */
    size_t const* window_widths;

    /** Number of entries in @c window_widths. */
    size_t window_widths_count;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Upper bound on entries sharing one key; zero is unbounded. */
    size_t values_per_key;

    /** The memory domains to place one shard on apiece. */
    usearch_ram_options_t ram;
} usearch_sequence_init_ram_t;

/**
 *  Creates an empty, mutable sequence index in host memory, for the uniform edit distances or
 *  the weighted alignments alike.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c NULL @c window_widths for a nonzero count, a zero
 *    window width, or @c weights without a substitution matrix.
 *  - @c usearch_status_unknown_kind_k An unknown @c metric.
 *  - @c usearch_status_config_mismatch_k @c weights for a kind other than Needleman-Wunsch,
 *    Needleman-Wunsch without them, window widths missing for the overlap kind or given to
 *    another, or weights folding into more than 32 classes, a negative gap or a pair above its
 *    mean self-score.
 *  - @c usearch_status_unsupported_k A pinned class the CPU does not admit.
 */
USEARCH_EXPORT void usearch_sequence_init_ram(usearch_sequence_init_ram_t* arguments);

/** The arguments of @c usearch_sequence_init_vram, which match @c usearch_sequence_init_ram_t but
 *  for the placement. */
typedef struct usearch_sequence_init_vram_t {

    /** Output for the new index. */
    usearch_sequence_index_t* index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The scoring kind; byte and UTF-8 Levenshtein, window overlap and Needleman-Wunsch serve a
     *  GPU. */
    usearch_sequence_metric_t metric;

    /** The weights an alignment kind needs; @c NULL for the Levenshtein kinds, which refuse any. */
    usearch_sequence_weights_t const* weights;

    /** Window widths in bytes for the overlap kind; every other kind refuses any. */
    size_t const* window_widths;

    /** Number of entries in @c window_widths. */
    size_t window_widths_count;

    /** The @b optional connections-per-node bound; zero derives. */
    size_t connectivity;

    /** Zero alone: a GPU index keeps no key lookup, so it bounds no key's entries and refuses any
     *  bound it is given. */
    size_t values_per_key;

    /** The devices to place one shard on apiece; naming none picks device zero. */
    usearch_vram_options_t vram;
} usearch_sequence_init_vram_t;

/**
 *  @brief Creates an empty, mutable sequence index in device memory.
 *
 *  Insertions and searches on it run on an executor driving its device, and a search writes its
 *  matches in device memory in an arena made for that executor, or on host pages in a host one.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c NULL window widths for a nonzero count, a @c NULL
 *    substitution matrix, or an empty window.
 *  - @c usearch_status_unknown_kind_k A kind other than byte or UTF-8 Levenshtein, window overlap
 *    or Needleman-Wunsch.
 *  - @c usearch_status_config_mismatch_k Weights for a kind that does not align, an alignment
 *    kind without weights, window widths for any kind but overlap or none for overlap, or devices
 *    of two classes.
 *  - @c usearch_status_unsupported_k A @c values_per_key bound, or devices that do not run the
 *    pinned class.
 *  - @c usearch_status_device_k A device past the listing, one whose vendor no kernel serves, or
 *    a build without GPU kernels.
 */
USEARCH_EXPORT void usearch_sequence_init_vram(usearch_sequence_init_vram_t* arguments);

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

    /** Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_sequence_load_ram_t;

/**
 *  @brief Copies a checkpoint into a new, mutable index in host memory.
 *
 *  Where the parts end up is the call's own memory domains, never the manifest's: part @c p is
 *  written whole on shard @c p, beside any placed shards the parts leave empty, and the shards past
 *  the placed ones open over those domains in turn.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, a checkpoint or part path that does
 *    not parse, or @c weights without a substitution matrix.
 *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part, its header, columns,
 *    sums or recorded costs do not parse.
 *  - @c usearch_status_unknown_kind_k A checkpoint of an unknown metric.
 *  - @c usearch_status_unsupported_k A pinned class the CPU does not admit.
 *  - @c usearch_status_uninitialized_k The executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k @c weights for a checkpoint of another kind than
 *    Needleman-Wunsch or folding into more than 32 classes, a negative gap or a pair above its
 *    mean self-score, a Needleman-Wunsch checkpoint recording no costs and given no weights, or
 *    the checkpoint's columns differing from the shards' in dtype, shape or key width.
 *  - @c usearch_status_device_k The executor is not a RAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor, or its pool runs another
 *    dispatch.
 *  - @c usearch_status_overflow_k A part passes a shard's bound or its key multimap.
 */
USEARCH_EXPORT void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments);

/** The arguments of @c usearch_sequence_load_ram_buffer. */
typedef struct usearch_sequence_load_ram_buffer_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_sequence_index_t* index;

    /** The executor the read fans out over, one shard per thread. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Source holding one serialized monolith, or the manifest of several parts leading them. */
    usearch_bytes_view_t buffer;

    /** The memory domains to place the parts on. */
    usearch_ram_options_t ram;

    /** Upper bound on entries sharing one key, which no checkpoint records; zero is unbounded. */
    size_t values_per_key;

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_sequence_load_ram_buffer_t;

/**
 *  @brief Copies a checkpoint from a memory buffer into a new, mutable index in host memory.
 *
 *  This places, reports and cancels like @c usearch_sequence_load_ram, which reaches the same
 *  bytes through a mapping, so @c buffer needs no file behind it, and a mapped file no copy.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c buffer, or @c weights without a
 *    substitution matrix.
 *  - @c usearch_status_malformed_k The manifest lists no part, or a part, its header, columns,
 *    sums or recorded costs do not parse.
 *  - @c usearch_status_unknown_kind_k A checkpoint of an unknown metric.
 *  - @c usearch_status_unsupported_k A pinned class the CPU does not admit.
 *  - @c usearch_status_uninitialized_k The executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k @c weights or recorded costs @c usearch_sequence_load_ram
 *    refuses for the same reasons, or the checkpoint's columns differing from the shards' in
 *    dtype, shape or key width.
 *  - @c usearch_status_device_k The executor is not a RAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor, or its pool runs another
 *    dispatch.
 *  - @c usearch_status_overflow_k A part passes a shard's bound or its key multimap.
 */
USEARCH_EXPORT void usearch_sequence_load_ram_buffer(usearch_sequence_load_ram_buffer_t* arguments);

/** The arguments of @c usearch_sequence_load_vram. */
typedef struct usearch_sequence_load_vram_t {

    /** Output for the new index, written by the time the call is final. */
    usearch_sequence_index_t* index;

    /** The executor the upload runs on, one stream per device. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** The devices to place the parts on. */
    usearch_vram_options_t vram;

    /** Zero alone: a GPU index keeps no key lookup, so it bounds no key's entries and refuses any
     *  bound it is given. */
    size_t values_per_key;

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;

    /** The @b optional handler, called with parts placed and their count. */
    usearch_progress_t progress;
} usearch_sequence_load_vram_t;

/**
 *  Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
 *  @c usearch_sequence_load_ram.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, a checkpoint or part path that does
 *    not parse, or @c weights without a substitution matrix.
 *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part, its header, columns,
 *    sums or recorded costs do not parse.
 *  - @c usearch_status_unknown_kind_k A checkpoint of an unknown metric.
 *  - @c usearch_status_unsupported_k A @c values_per_key bound, or devices that do not run the
 *    pinned class.
 *  - @c usearch_status_config_mismatch_k Devices of two classes, @c weights for a checkpoint of
 *    another kind than Needleman-Wunsch or folding into more than 32 classes, a negative gap or a
 *    pair above its mean self-score, a Needleman-Wunsch checkpoint recording no costs and given
 *    no weights, the executor driving no stream on some shard's device, or the checkpoint's
 *    columns differing from the shards' in dtype, shape or key width.
 *  - @c usearch_status_device_k An ordinal past the listing, devices this build carries no kernels
 *    for, a build without GPU kernels, or an executor that is not a VRAM executor.
 *  - @c usearch_status_contended_k Another call holds the executor.
 *  - @c usearch_status_overflow_k A part passes a shard's bound or its key multimap.
 */
USEARCH_EXPORT void usearch_sequence_load_vram(usearch_sequence_load_vram_t* arguments);

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

    /** The class to dispatch as; a view reads its shards from the checkpoint, so only
     *  @c usearch_ram_options_t::cpu_class is read. */
    usearch_ram_options_t ram;
} usearch_sequence_view_ram_t;

/**
 *  Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
 *  verbs report @c usearch_status_read_only_k.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, a checkpoint or part path that does
 *    not parse, or @c weights without a substitution matrix.
 *  - @c usearch_status_malformed_k The checkpoint lists no part, or a part, its header or its
 *    recorded costs do not parse.
 *  - @c usearch_status_unknown_kind_k A checkpoint of an unknown metric.
 *  - @c usearch_status_config_mismatch_k @c weights for a checkpoint of another kind than
 *    Needleman-Wunsch or folding into more than 32 classes, a negative gap or a pair above its
 *    mean self-score, or a Needleman-Wunsch checkpoint recording no costs and given no weights.
 *  - @c usearch_status_unsupported_k Every other checkpoint, as RAM shards cannot serve a mapping
 *    in place yet.
 */
USEARCH_EXPORT void usearch_sequence_view_ram(usearch_sequence_view_ram_t* arguments);

/** The arguments of @c usearch_sequence_view_nvme. */
typedef struct usearch_sequence_view_nvme_t {

    /** Output for the new, read-only index, written by the time the call is final. */
    usearch_sequence_index_t* index;

    /** The NVMe executor the call is ordered on; the headers come through blocking reads. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Checkpoint path - a monolith, or a manifest naming many. */
    usearch_bytes_view_t path;

    /** The devices the parts are served from; naming none serves each where its manifest points. */
    usearch_nvme_options_t nvme;

    /** The weights an alignment checkpoint was built with; @c NULL for the Levenshtein kinds. */
    usearch_sequence_weights_t const* weights;
} usearch_sequence_view_nvme_t;

/**
 *  @brief Serves a checkpoint in place from NVMe drives as a read-only index, every later call
 *      reading it through its executor's engines.
 *
 *  Opening reads the headers alone. Every mutating verb on the handle reports
 *  @c usearch_status_read_only_k, and the parts must stay unmodified until the handle is freed.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k @c executor is not an NVMe executor.
 *  - @c usearch_status_invalid_argument_k An empty @c path, @c NULL paths for a nonzero count, a
 *    path that does not parse or names a base off its medium's block, or @c weights without a
 *    substitution matrix.
 *  - @c usearch_status_malformed_k The manifest names no part, or a manifest, header, row
 *    directory or the recorded costs do not parse.
 *  - @c usearch_status_unknown_kind_k A checkpoint of an unknown metric.
 *  - @c usearch_status_config_mismatch_k Parts disagreeing with this index's values, keys, node
 *    IDs or metric, or with @c nvme, @c weights for a checkpoint of another kind than
 *    Needleman-Wunsch or folding into more than 32 classes, a negative gap or a pair above its
 *    mean self-score, or a Needleman-Wunsch checkpoint recording no costs and given no weights.
 *  - @c usearch_status_unsupported_k A pinned class the CPU does not admit.
 *  - @c usearch_status_uninitialized_k The executor holds no threads or pool workers.
 *  - @c usearch_status_contended_k Another call holds the executor, or its pool runs another
 *    dispatch.
 */
USEARCH_EXPORT void usearch_sequence_view_nvme(usearch_sequence_view_nvme_t* arguments);

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

/**
 *  Grows the shards ahead of incoming entries - a capacity hint, since insertions grow them on
 *  demand.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is an NVMe view.
 *  - @c usearch_status_overflow_k @c entries passes a shard's ceiling, its ID space or its key
 *    column.
 */
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

/**
 *  Erases all entries, keeping the reserved capacity.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is an NVMe view.
 *  - @c usearch_status_contended_k A call in flight, a batch or load holding claims, or a plan
 *    holding the handle closed.
 */
USEARCH_EXPORT void usearch_sequence_clear(usearch_sequence_clear_t* arguments);

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

    /** The @b optional breadth of the candidates the build walk explores - the @c ef of the
     *  literature; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries admitted and the batch length. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_insert_t;

/**
 *  @brief Adds a tape of sequences under their keys - the native batch operation; a single
 *      sequence is a tape of one.
 *
 *  An insertion is all-or-nothing per call. Every fallible step runs before the first entry
 *  identity is claimed, so a rejected call publishes nothing - a cancelled one included, which
 *  reports @c usearch_status_cancelled_k - and the entries of an accepted one become visible at
 *  once, their keys right after.
 *
 *  A key already holding @c values_per_key entries rejects the whole call, since a smaller retry
 *  cannot help.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is a view.
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

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional breadth of the candidates the re-selection walk explores; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries re-selected; a cancel stops the re-selection,
     *  still heals, and answers @c usearch_status_cancelled_k. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_refine_t;

/**
 *  @brief Re-selects the neighbors of every stored entry against the finished graph, improving
 *      neighbors chosen when the graph around them was still incomplete.
 *
 *  This is a build step rather than a repair. A cheap insertion pass followed by one refinement
 *  reaches a better graph, in less total time, than a single expensive insertion pass - so a wide
 *  @c expansion_factor belongs here rather than in @c usearch_sequence_insert. It pays most in this
 *  modality, where a single alignment is quadratic and a stingy insertion pass is tempting.
 *
 *  Cancelling is safe, which is what makes a whole-graph sweep affordable. Re-selecting an entry's
 *  neighbors is idempotent and claims no identity, so a stopped pass leaves the graph correct and
 *  merely less improved: nothing unwinds, and the call answers @c usearch_status_cancelled_k. The
 *  index keeps nothing of a stopped pass, so the next call starts at the first entry again; a
 *  caller wanting to continue where a cancel stopped holds a @c usearch_sequence_refinement_t
 *  instead.
 *
 *  Re-selection alone can leave an island, an entry no search reaches, so every call closes by
 *  re-linking them: unreachable entries re-select from scratch, and whatever the selection still
 *  drops is reconnected to a reachable one. A cancelled call skips the rest of the re-selection but
 *  still re-links, so it never leaves entries unreachable.
 *
 *  Searches may run throughout: re-selection rewrites one entry at a time and never removes one,
 *  so a concurrent walk sees either the old neighbors or the new ones. Insertions never overlap:
 *  one in flight refuses the call with @c usearch_status_contended_k, and one arriving meanwhile
 *  is refused the same way, as is a save.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k The handle is an NVMe view.
 *  - @c usearch_status_unsupported_k The handle lives on a GPU, whose index refines nothing yet.
 *  - @c usearch_status_contended_k An insertion or another refinement holds the handle.
 */
USEARCH_EXPORT void usearch_sequence_refine(usearch_sequence_refine_t* arguments);

/** The arguments of @c usearch_sequence_health. */
typedef struct usearch_sequence_health_t {

    /** The handle to be audited. */
    usearch_sequence_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output for the audit; zeroed on failure. */
    usearch_reachability_t* reachability;
} usearch_sequence_health_t;

/**
 *  @brief Audits how much of the graph a search can reach: every shard swept level by level from
 *      its first entry, forward and back. Read-only, so concurrent insertions merely leave fresh
 *      entries out of the audit.
 *
 *  An NVMe view sweeps forward alone, from entries spaced through each part, so its
 *  @c back_sweeps stays zero.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k The executor's kind does not match the residency.
 *  - @c usearch_status_uninitialized_k No shard is placed, or the executor holds no pool workers
 *    or streams.
 *  - @c usearch_status_contended_k Another call holds the executor or its pool, or a destructive
 *    window overlapped the audit twice.
 *  - @c usearch_status_config_mismatch_k The executor drives no stream on some VRAM shard's
 *    device, or an NVMe medium the engines cannot read.
 *  - @c usearch_status_io_k A read from an NVMe view's drives failed.
 */
USEARCH_EXPORT void usearch_sequence_health(usearch_sequence_health_t* arguments);

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

    /** Queries to resolve; the outputs have @c count rows of @c wanted entries. */
    size_t count;

    /** The queries, as bytes end to end around @c count + 1 offsets. */
    usearch_tape_view_t queries;

    /** Output for the run, inside @c arena. */
    usearch_matches_view_t* matches;

    /** Memory for @c matches, rewound by the call. */
    usearch_arena_t arena;

    /** Columns per row, and the upper bound on matches - the "k" in "kANN". */
    size_t wanted;

    /** The @b optional breadth of the candidates a walk explores - the @c ef of the literature;
     *  zero derives, a scan ignores it. */
    size_t expansion_factor;

    /** How entries are reached: a walk, the default, or an exact scan of every entry, on RAM and
     *  NVMe. */
    usearch_traversal_t traversal;

    /** The @b optional predicate, asked about a batch of candidates at a time. */
    usearch_predicate_t predicate;

    /** The @b optional handler, called with queries resolved and their count. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_search_t;

/**
 *  @brief Performs k-Nearest Neighbors Search for a tape of queries, approximate by walking the
 *      graph or exact by scanning every entry; one query is a tape of one.
 *
 *  The native batch operation, parallelized internally. Never writes to the graph, so any number
 *  may overlap with concurrent insertions. @ref sequence_index_concurrency
 *
 *  A @c predicate narrows the matches to the entries it admits. Non-matching entries still route
 *  the traversal rather than terminating it, so recall does not collapse at low selectivity - the
 *  walk crosses rejected regions to reach admitted ones behind them. The cost is that the
 *  predicate judges far more keys than @c wanted, every scored candidate, asked about a batch of
 *  them at a time, so it belongs on the caller's fast path.
 *
 *  A cancelled search reports @c usearch_status_cancelled_k and leaves @c matches @c NULL.
 *
 *  @par Statuses:
 *  - @c usearch_status_unknown_kind_k A @c traversal other than a walk or a scan.
 *  - @c usearch_status_unsupported_k A scan of a VRAM handle, or a @c predicate on one.
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

/**
 *  @brief Counts the entries stored under one key, with their payload extent.
 *
 *  A GPU handle counts the entries alone and leaves the extents zero, because its payloads stay on
 *  the devices themselves.
 *
 *  @par Statuses:
 *  - @c usearch_status_device_k An NVMe view's executor is not an NVMe executor.
 *  - @c usearch_status_uninitialized_k An NVMe view's executor holds no pool workers.
 *  - @c usearch_status_contended_k Another call holds an NVMe view's executor, or its pool runs
 *    another dispatch.
 *  - @c usearch_status_config_mismatch_k An NVMe medium the engines cannot read.
 */
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

    /** Output for the tape inside @c arena: entries' bytes end to end, never terminated, around
     *  @c written + 1 offsets. */
    usearch_tape_view_t* sequences;

    /** The @b optional output for the entries written; zero when the key is absent. */
    size_t* written;

    /** Memory for @c sequences, rewound by the call. */
    usearch_arena_t arena;
} usearch_sequence_get_t;

/**
 *  @brief Retrieves every sequence stored under one key as an Apache Arrow tape, in insertion order -
 *      one call, however many and however long they turn out to be.
 *
 *  An absent key leaves the tape's pointers @c NULL.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c NULL @c arena.
 *  - @c usearch_status_device_k An arena or an executor on another residency or device.
 *  - @c usearch_status_contended_k The arena serves another call, or another call holds an NVMe
 *    view's executor or its pool.
 *  - @c usearch_status_uninitialized_k An NVMe view's executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k An NVMe medium the engines cannot read.
 */
USEARCH_EXPORT void usearch_sequence_get(usearch_sequence_get_t* arguments);

/** The arguments of @c usearch_sequence_absorb. */
typedef struct usearch_sequence_absorb_t {

    /** The handle absorbing the donors. */
    usearch_sequence_index_t index;

    /** The executor each donor's move and the closing refinement fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Donor handles to absorb. */
    size_t count;

    /** Array of @c count handles, each consumed and nulled once its entries are published here. */
    usearch_sequence_index_t* donors;

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional breadth of the candidates of the closing refine's walks; zero derives. */
    size_t expansion_factor;

    /** The @b optional handler, called with entries the closing refine re-selected; a cancel stops
     *  it as it stops @c usearch_sequence_refine, never a move. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_absorb_t;

/**
 *  @brief Absorbs every handle in @c donors into @c index, one after another, keeping their graphs
 *      rather than re-inserting their sequences, and links each in by walks across. Donor handles
 *      are consumed and must not be used afterwards, not even for @c usearch_sequence_free.
 *
 *  Every donor is checked and held before the first one enters, so a refusal consumes none. The
 *  donors then enter one after another, each all-or-nothing: its entries append to @c index with
 *  their edges rebased, publish at once, and the donor is consumed, so at most one donor's entries
 *  are ever held twice. One refinement then walks every entry across the parts the donors appended,
 *  round by round, so a group of donors costs one refinement rather than one per donor. Readers of
 *  @c index may keep searching throughout, since unpublished entries hide behind the watermark,
 *  while nothing may search a donor. Insertions into @c index arriving meanwhile are refused with
 *  @c usearch_status_contended_k.
 *
 *  The handler reports and cancels the closing refinement alone, exactly as in
 *  @c usearch_sequence_refine: a cancel skips the rest of the walks across, still re-links, and
 *  answers @c usearch_status_cancelled_k with every donor consumed. A caller wanting to continue
 *  those walks later holds a @c usearch_sequence_refinement_t instead.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k @c index is an NVMe view.
 *  - @c usearch_status_unsupported_k @c index lives on a GPU, whose index absorbs nothing yet.
 *  - @c usearch_status_invalid_argument_k A @c NULL slot in @c donors, or @c NULL @c donors
 *    for a nonzero @c count.
 *  - @c usearch_status_config_mismatch_k A donor of another residency, class, kind or geometry,
 *    or @c index itself.
 *  - @c usearch_status_contended_k An insertion or refinement in flight on @c index or on a
 *    donor; every donor stays a live handle.
 */
USEARCH_EXPORT void usearch_sequence_absorb(usearch_sequence_absorb_t* arguments);

/** The arguments of @c usearch_sequence_split. */
typedef struct usearch_sequence_split_t {

    /** The handle split into the receivers, left empty and live. */
    usearch_sequence_index_t index;

    /** The executor the partition, each part's move and the closing refinements fan out over. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Receiver handles to split into, at least two. */
    size_t count;

    /** Array of @c count empty handles of the source's residency, class, kind and geometry, each
     *  receiving one part. */
    usearch_sequence_index_t const* receivers;

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional breadth of the candidates of the closing refines' walks; zero derives. */
    size_t expansion_factor;

    /** Picks the part roots: equal seeds over an unchanged index split equally. */
    uint64_t seed;

    /** The @b optional handler, called with entries the closing refines re-selected; a cancel stops
     *  them as it stops @c usearch_sequence_refine, never a move. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_split_t;

/**
 *  @brief Splits @c index into the handles in @c receivers, the inverse of
 *      @c usearch_sequence_absorb: graph-local parts keeping their edges rather than re-inserting
 *      their sequences, one part per receiver, and @c index left empty but live.
 *
 *  Every receiver is checked and held before the first part moves, so a refusal moves nothing. The
 *  part roots are drawn from @c seed without a scan, and each part grows breadth-first over the
 *  links from its root, capped at an equal share of the entries, while entries no root reaches
 *  enter the smallest part. Each part then enters its receiver all-or-nothing, its edges rebased,
 *  its edges into other parts dropped, and the entries that lost one placed first; @c index clears
 *  once every part moved. Each receiver closes by re-selecting the entries that lost an edge and
 *  re-linking the islands that leaves, as @c usearch_sequence_refine does. Insertions into @c index
 *  or a receiver arriving meanwhile are refused with @c usearch_status_contended_k, and nothing may
 *  search either until the call is final.
 *
 *  The handler reports and cancels the closing refinements alone, exactly as in
 *  @c usearch_sequence_refine: a cancel skips the rest of the re-selection, still re-links, and
 *  answers @c usearch_status_cancelled_k with every part moved. A caller wanting to continue that
 *  re-selection later holds a @c usearch_sequence_refinement_t opened with @c receivers instead.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k @c index is a view.
 *  - @c usearch_status_invalid_argument_k Fewer than two receivers, @c NULL @c receivers, or a
 *    @c NULL slot in them.
 *  - @c usearch_status_config_mismatch_k A receiver of another residency, class, kind or
 *    geometry, one holding entries or another shard count, one named twice, or @c index
 *    itself.
 *  - @c usearch_status_contended_k An insertion or refinement in flight on @c index or on a
 *    receiver; nothing moves.
 *  - @c usearch_status_overflow_k A part passes a receiver shard's ceiling.
 *  - @c usearch_status_unsupported_k @c index lives on a GPU, whose index splits nothing yet.
 */
USEARCH_EXPORT void usearch_sequence_split(usearch_sequence_split_t* arguments);

/** The arguments of @c usearch_sequence_refinement_init. */
typedef struct usearch_sequence_refinement_init_t {

    /** The handle to be refined. */
    usearch_sequence_index_t index;

    /** The executor every call on the refinement runs on; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The @b optional donors to absorb first, each consumed and nulled once its entries land; the
     *  array must outlive the refinement. */
    usearch_sequence_index_t* donors;

    /** Donor handles in @c donors. */
    size_t count;

    /** The @b optional receivers to split @c index into first, as @c usearch_sequence_split fills
     *  them, each refined in turn; the array must outlive the refinement. */
    usearch_sequence_index_t const* receivers;

    /** Receiver handles in @c receivers, zero or at least two. */
    size_t receivers_count;

    /** Output for the new handle; @c NULL on failure. The index admits no insertion or save until
     *  @c usearch_sequence_refinement_finish frees it. */
    usearch_sequence_refinement_t* refinement;

    /** The @b optional breadth of the candidates the re-selection walk explores; zero derives. */
    size_t expansion_factor;

    /** Picks the part roots when @c receivers are named, as in @c usearch_sequence_split. */
    uint64_t seed;
} usearch_sequence_refinement_init_t;

/**
 *  @brief Opens a refinement of @c index, absorbing @c donors first when any are named, as
 *      @c usearch_sequence_absorb moves them, or splitting it into @c receivers, as
 *      @c usearch_sequence_split fills them. Every allocation the refinement needs in the index
 *      happens here, never in a step.
 *
 *  @par Statuses:
 *  - @c usearch_status_read_only_k @c index is an NVMe view.
 *  - @c usearch_status_unsupported_k @c index lives on a GPU, whose index refines nothing yet.
 *  - @c usearch_status_invalid_argument_k No executor, a @c NULL slot in @c donors or
 *    @c receivers, @c NULL @c donors for a nonzero @c count, one receiver alone, or donors and
 *    receivers both named.
 *  - @c usearch_status_config_mismatch_k A donor or receiver of another residency, class, kind
 *    or geometry, a receiver holding entries, or @c index itself.
 *  - @c usearch_status_contended_k An insertion or another refinement in flight on @c index, on
 *    a donor or on a receiver.
 */
USEARCH_EXPORT void usearch_sequence_refinement_init(usearch_sequence_refinement_init_t* arguments);

/** The arguments of @c usearch_sequence_refinement_next. */
typedef struct usearch_sequence_refinement_next_t {

    /** The handle the refinement was opened on. */
    usearch_sequence_index_t index;

    /** The executor the refinement was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The refinement to continue. */
    usearch_sequence_refinement_t refinement;

    /** The @b optional handler, called with entries re-selected across every call so far. */
    usearch_progress_t progress;

    /** The @b optional outputs for the entries re-selected so far and, of those, the ones whose
     *  neighbors changed; their ratio falling is what says another call buys little. */
    size_t* reselected;
    size_t* changed;

    /** The @b optional traversal telemetry of every call on the refinement so far; @c NULL skips
     *  reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_refinement_next_t;

/**
 *  @brief Continues the refinement until it ends or @c progress cancels, moving any donors still
 *      pending first. Answers @c usearch_status_cancelled_k when a later call has more to do; that
 *      call continues exactly where this one stopped. The executor is held only for the call.
 *
 *  Every call reaching the refinement writes the running totals, a cancelled or failed one too, so
 *  they only grow from one call to the next and @c changed never exceeds @c reselected; all stay
 *  zero while donors are pending.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c index or @c executor is not the one the refinement
 *    was opened with.
 */
USEARCH_EXPORT void usearch_sequence_refinement_next(usearch_sequence_refinement_next_t* arguments);

/** The arguments of @c usearch_sequence_refinement_finish. */
typedef struct usearch_sequence_refinement_finish_t {

    /** The handle the refinement was opened on. */
    usearch_sequence_index_t index;

    /** The executor the refinement was opened with. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** The refinement to finish; freed by the call, whatever it answers. */
    usearch_sequence_refinement_t refinement;

    /** The @b optional output for what the call did to the graph, as @c usearch_refine_report_t
     *  counts it; @c NULL skips reporting. */
    usearch_refine_report_t* report;

    /** The @b optional traversal telemetry of every call on the refinement; @c NULL skips
     *  reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_refinement_finish_t;

/**
 *  @brief Skips whatever re-selection a stopped refinement left, re-links the islands it leaves as
 *      @c usearch_sequence_refine does, reports, reopens the index, and frees the handle. A
 *      refinement whose donors have not all moved re-links nothing and reports zeros.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c index or @c executor is not the one the refinement
 *    was opened with; the handle is still freed.
 */
USEARCH_EXPORT void usearch_sequence_refinement_finish(usearch_sequence_refinement_finish_t* arguments);

/** The arguments of @c usearch_sequence_cursor_init. */
typedef struct usearch_sequence_cursor_init_t {

    /** The handle to be scanned. */
    usearch_sequence_index_t index;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The @b optional count of @c starts; zero walks every entry as stored. */
    size_t starts_count;

    /** The keys to walk from, in the order the walk begins with them. */
    usearch_key_t const* starts;

    /** Output for the new cursor; @c NULL on failure. */
    usearch_sequence_cursor_t* cursor;
} usearch_sequence_cursor_init_t;

/**
 *  @brief Opens a snapshot cursor over the current contents; see @c usearch_sequence_cursor_t.
 *
 *  Without @c starts the cursor yields every entry as stored. With them it yields the start keys'
 *  own entries first, in the order given, then the entries those link to, breadth-first, each
 *  entry once, so a caller wanting only the starts' immediate links reads the links of the first
 *  pulled entries. A key the index lacks starts nothing, and a key several entries share starts
 *  from the first stored.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c NULL @c starts for a nonzero @c starts_count.
 *  - @c usearch_status_unsupported_k The handle lives on a GPU, which walks no links yet.
 */
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

    /** Output for the identities, inside @c arena. */
    usearch_keys_view_t* keys;

    /** The @b optional output for each entry's links, the keys of the entries it links to in the
     *  graph, entry @c i spanning @c offsets[i] up to @c offsets[i+1], inside @c arena; @c NULL
     *  skips them. */
    usearch_keys_tape_view_t* links;

    /** Output for the entries written; zero when the snapshot is exhausted. */
    size_t* written;

    /** Memory for @c keys and @c links, rewound by the call. */
    usearch_arena_t arena;

    /** The upper bound on entries this pull writes. */
    size_t wanted;
} usearch_sequence_cursor_next_t;

/**
 *  @brief Copies the next batch of entries out of the cursor into @c arena, each with the keys it
 *      links to, in the order @c usearch_sequence_cursor_init describes; the sequences themselves
 *      come from @c usearch_sequence_get.
 *
 *  Links name only entries the snapshot holds, as the graph stored them when the pull ran. The
 *  keys are copied rather than borrowed from the index, so the batch stays valid until the next
 *  call given the same arena, however the cursor moves meanwhile.
 *
 *  @par Statuses:
 *  - @c usearch_status_io_k A read from an NVMe view's drives failed.
 *  - @c usearch_status_invalid_argument_k @c index is not the handle the cursor was opened on.
 */
USEARCH_EXPORT void usearch_sequence_cursor_next(usearch_sequence_cursor_next_t* arguments);

/** Releases the cursor; @c NULL is safe.
 *  Waits until every deferred call naming the handle is final. */
USEARCH_EXPORT void usearch_sequence_cursor_free(usearch_sequence_cursor_t cursor);

/** The arguments of @c usearch_sequence_sample. */
typedef struct usearch_sequence_sample_t {

    /* Call */
    /** The handle to be sampled. */
    usearch_sequence_index_t index;

    /** The executor the draws run on; its kind must match the handle's residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /* Inputs */

    /** The @b optional count of @c shown; zero shows nothing yet. */
    size_t shown_count;

    /** The keys already shown: never drawn again, and the paths reach toward them. */
    usearch_keys_view_t shown;

    /** Where the @c shown keys sit, @c shown_count rows of three @c usearch_scalar_f32_k values,
     *  the points later draws are placed against. */
    usearch_rows_view_t shown_coordinates;

    /** The @b optional count of @c focus; zero looks around the whole index. */
    size_t focus_count;

    /** The keys to look around, the roots the draws grow from. */
    usearch_keys_view_t focus;

    /* Outputs */

    /** Output for the keys drawn, inside @c arena. */
    usearch_keys_view_t* keys;

    /** Output for which of its key's entries each draw is, in the order @c usearch_sequence_get
     *  writes them, inside @c arena. */
    size_t const** entry_ordinals;

    /** Output for the one shown or earlier draw each draw hangs from and its distance, one column,
     *  a draw with none counting zero, inside @c arena. */
    usearch_matches_view_t* parents;

    /** Output for where each draw is placed, @c written rows of three @c usearch_scalar_f32_k
     *  values, inside @c arena. */
    usearch_rows_view_t* coordinates;

    /** Output for the draws written; fewer than @c wanted only once the region is exhausted. */
    size_t* written;

    /** Memory for every output, rewound by the call. */
    usearch_arena_t arena;

    /* Knobs */
    /** The upper bound on draws this call writes. */
    size_t wanted;

    /** The draw: equal seeds, shown keys and focus over an unchanged index draw equally, and a
     *  larger @c wanted draws a superset. */
    uint64_t seed;

    /** The @b optional breadth of the candidates of the walks behind the draws; zero derives. */
    size_t expansion_factor;

    /* Observers */
    /** The @b optional handler, called with draws written and @c wanted. */
    usearch_progress_t progress;

    /** The @b optional traversal telemetry; @c NULL skips reporting. */
    usearch_traversal_stats_t* stats;
} usearch_sequence_sample_t;

/**
 *  @brief Draws keys around @c focus, or across the whole index, each hanging from a shown or
 *      earlier draw and placed in three dimensions beside the shown ones - so a view grows one
 *      connected sample as it zooms, never scanning the index.
 *
 *  The draws follow the graph: along paths joining the focus to each other and toward the nearest
 *  shown key, then around each focus key as far as its nearest neighbor on those paths, ranked by
 *  @c seed so a closer view draws a superset of a farther one. A key holding several entries draws
 *  each as its own, named by its entry ordinal. Each draw is placed by its distances to its nearest
 *  shown keys, the shown keys keeping their coordinates; with nothing shown, the draws are scaled
 *  against one another.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c NULL @c shown or @c focus for a nonzero count, or
 *    @c shown_coordinates not three @c usearch_scalar_f32_k columns.
 *  - @c usearch_status_unsupported_k The handle lives on a GPU, which does not sample yet.
 *  - @c usearch_status_io_k A read from an NVMe view's drives failed.
 */
USEARCH_EXPORT void usearch_sequence_sample(usearch_sequence_sample_t* arguments);

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

    /** Destination path: the monolith of a lone shard, or the manifest beside the parts of many. */
    usearch_bytes_view_t path;

    /** The @b optional handler, called with parts written and their count. */
    usearch_progress_t progress;
} usearch_sequence_save_t;

/**
 *  @brief Saves a mutable index under @c path. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  The shard count picks the layout: one shard writes one monolith at @c path, and several write
 *  the manifest at @c path beside one part per shard, each with its shard's own node IDs.
 *  A cancelled save leaves the parts it already wrote and no manifest referencing them, so the
 *  checkpoint reads as absent rather than half-valid. Those leftovers are the caller's to remove.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c path, or a part or manifest path that does
 *    not parse.
 *  - @c usearch_status_device_k The executor's kind does not match the residency.
 *  - @c usearch_status_unsupported_k The handle is an NVMe view, which already is its checkpoint.
 *  - @c usearch_status_contended_k Another call holds the executor or its pool, a dispensed batch
 *    has yet to publish, or a call, batch, load or plan holds the handle.
 *  - @c usearch_status_uninitialized_k The executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k The executor drives no stream on some VRAM shard's
 *    device.
 *  - @c usearch_status_overflow_k @c path names a device that ends before a part or the manifest.
 */
USEARCH_EXPORT void usearch_sequence_save(usearch_sequence_save_t* arguments);

/** The arguments of @c usearch_sequence_save_buffer. */
typedef struct usearch_sequence_save_buffer_t {

    /** The handle to be written. */
    usearch_sequence_index_t index;

    /** The executor to fan out over; its kind must match the residency. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Output for the checkpoint's bytes and length, inside @c arena. */
    usearch_bytes_view_t* monolith;

    /** Memory for @c monolith, rewound by the call. */
    usearch_arena_t arena;

    /** The @b optional handler, called with parts written and their count. */
    usearch_progress_t progress;
} usearch_sequence_save_buffer_t;

/**
 *  @brief Saves a mutable index into @c arena: one monolith for a handle of one shard, or a
 *      manifest leading one part per shard. Demands a mutation-free window; concurrent readers
 *      keep searching throughout.
 *
 *  A lone shard's monolith is the same bytes @c usearch_sequence_save writes. Several shards write
 *  the manifest first, listing each part by its offset inside the buffer, then every part on the
 *  header's alignment, each with its shard's own node IDs, so a load opens one shard per part
 *  again. A cancelled call leaves @c monolith->data @c NULL.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c NULL @c arena.
 *  - @c usearch_status_device_k An arena or an executor on another residency or device.
 *  - @c usearch_status_contended_k The arena serves another call, another call holds the
 *    executor or its pool, a dispensed batch has yet to publish, or a call, batch, load or plan
 *    holds the handle.
 *  - @c usearch_status_unsupported_k The handle is an NVMe view, which already is its checkpoint,
 *    or lives on a GPU, which writes no checkpoint into memory yet.
 *  - @c usearch_status_uninitialized_k No shard is live, or the executor holds no pool workers.
 *  - @c usearch_status_config_mismatch_k A stored value disagrees with the shape its column
 *    records.
 *  - @c usearch_status_out_of_memory_k The arena cannot grow to the checkpoint's length.
 */
USEARCH_EXPORT void usearch_sequence_save_buffer(usearch_sequence_save_buffer_t* arguments);

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

/**
 *  Reads a checkpoint's metric & geometry without loading it.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k The checkpoint or a part path does not parse.
 *  - @c usearch_status_malformed_k The checkpoint or a part is empty, its manifest lists no part,
 *    or a header does not parse or spans more bytes than its part.
 */
USEARCH_EXPORT void usearch_sequence_metadata_read(usearch_sequence_metadata_read_t* arguments);

/** The arguments of @c usearch_sequence_metadata_read_buffer. */
typedef struct usearch_sequence_metadata_read_buffer_t {

    /** The executor the read runs on. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Blocking, the default, or deferred. */
    usearch_completion_t completion;

    /** Source holding one serialized monolith, or the manifest of several parts leading them. */
    usearch_bytes_view_t buffer;

    /** Output for every field the checkpoint froze. */
    usearch_sequence_metadata_t* metadata;
} usearch_sequence_metadata_read_buffer_t;

/**
 *  @brief Reads the fields a checkpoint in a memory buffer froze without loading it, as
 *      @c usearch_sequence_metadata_read reads them from a path.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k An empty @c buffer.
 *  - @c usearch_status_malformed_k The manifest lists no part, or a header does not parse or
 *    spans more bytes than its part.
 */
USEARCH_EXPORT void usearch_sequence_metadata_read_buffer(usearch_sequence_metadata_read_buffer_t* arguments);

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

    /** The CPU class whose kits the index dispatched to on RAM and NVMe;
     *  @c usearch_cpu_class_unknown_k on VRAM. */
    usearch_cpu_class_t cpu_class;

    /** The GPU class whose kernels the index dispatched to on VRAM; @c usearch_gpu_class_unknown_k
     *  on RAM and NVMe. */
    usearch_gpu_class_t gpu_class;

    /** Live shards, each a graph of its own; a fresh index has one per memory domain or device it
     *  was placed on. */
    size_t shards_count;
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
