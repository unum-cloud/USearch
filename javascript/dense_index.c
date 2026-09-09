/**
 *  @brief Vector search over equidimensional dense embeddings - the Node-API twin of
 *      @c include/usearch/dense_index.h.
 *  @file javascript/dense_index.c
 *  @author Ash Vardanian
 *
 *  The @c CompiledIndex class, the snapshot cursor and the resumable finder it opens, and the
 *  checkpoint metadata readers that need no index at all.
 */
#include <stdlib.h> // `malloc`, `free`
#include <string.h> // `memcpy`, `memset`

#include "lib.h"

#pragma region Handles

/** @brief A snapshot cursor and the geometry its vector export is sized from. */
typedef struct usearch_js_cursor_t {
    /** @brief The engine handle, @c NULL once released. */
    usearch_dense_cursor_t handle;
    /** @brief The frozen dimensionality, captured so a pull needs no index argument. */
    size_t dimensions;
} usearch_js_cursor_t;

/** @brief A resumable nearest-neighbor stream, holding one of the executor's threads until released. */
typedef struct usearch_js_finder_t {
    /** @brief The engine handle, @c NULL once released. */
    usearch_dense_finder_t handle;
} usearch_js_finder_t;

/** @brief The three outputs of a ranked run, allocated where the engine writes them. */
/**
 *  @brief Borrows the three output arrays a ranked verb fills, deriving the row width.
 *
 *  @c wanted follows from the key array's length over the count array's, so the space reserved
 *  and the width asked for cannot disagree. Counts land in a scratch run rather than the
 *  caller's array, since the engine writes @c size_t where a @c BigUint64Array holds @c uint64_t
 *  and the two part company off 64-bit targets; @c usearch_js_matches_publish copies them back.
 */
static bool usearch_js_matches_from_value(napi_env env, napi_value keys_value, napi_value scores_value,
                                          napi_value counts_value, size_t queries_count, usearch_matches_span_t* span) {
    usearch_js_matrix_t keys, scores;
    napi_typedarray_type counts_type = napi_int8_array;
    napi_value counts_buffer = NULL;
    size_t counts_offset = 0, counts_length = 0;
    void* counts_data = NULL;
    bool is_typedarray = false;

    if (!usearch_js_matrix_from_value(env, keys_value, "keys", usearch_scalar_u8_k, &keys) ||
        !usearch_js_matrix_from_value(env, scores_value, "distances", usearch_scalar_f64_k, &scores))
        return false;
    if (napi_is_typedarray(env, counts_value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, counts_value, &counts_type, &counts_length, &counts_data, &counts_buffer,
                                 &counts_offset) != napi_ok ||
        counts_type != napi_biguint64_array)
        return usearch_js_throw_field(env, "counts", "must be a BigUint64Array");

    if (counts_length != queries_count)
        return usearch_js_throw(env, "shape-mismatch", "The counts must hold one entry per query");
    if (keys.bytes % (queries_count * usearch_js_key_bytes_k) != 0)
        return usearch_js_throw(env, "shape-mismatch", "The keys do not divide into whole rows per query");

    size_t const wanted = keys.bytes / (queries_count * usearch_js_key_bytes_k);
    if (!wanted) return usearch_js_throw_field(env, "keys", "must hold at least one match per query");
    if (scores.bytes != queries_count * wanted * sizeof(usearch_distance_t))
        return usearch_js_throw(env, "shape-mismatch", "The keys and distances are not the same shape");

    span->counts = (size_t*)malloc(queries_count * sizeof(size_t));
    if (!span->counts) return usearch_js_throw(env, "out-of-memory", "Could not allocate the count scratch");
    span->keys = (usearch_key_t*)keys.data;
    span->keys_stride_bytes = wanted * usearch_js_key_bytes_k;
    span->scores = (usearch_distance_t*)scores.data;
    span->scores_stride_bytes = wanted * sizeof(usearch_distance_t);
    span->wanted = wanted;
    return true;
}

/** @brief Copies the counts the engine wrote into the caller's array and frees the scratch. */
static napi_value usearch_js_matches_publish(napi_env env, napi_value counts_value, size_t queries_count,
                                             usearch_matches_span_t* span) {
    napi_typedarray_type counts_type = napi_int8_array;
    napi_value counts_buffer = NULL;
    size_t counts_offset = 0, counts_length = 0;
    void* counts_data = NULL;
    if (napi_get_typedarray_info(env, counts_value, &counts_type, &counts_length, &counts_data, &counts_buffer,
                                 &counts_offset) == napi_ok)
        for (size_t query_index = 0; query_index != queries_count; ++query_index)
            ((uint64_t*)counts_data)[query_index] = (uint64_t)span->counts[query_index];
    free(span->counts);
    span->counts = NULL;
    return usearch_js_undefined(env);
}

/** @brief Releases the count scratch when a verb fails before publishing. */
static void usearch_js_matches_release(usearch_matches_span_t* span) {
    free(span->counts);
    span->counts = NULL;
}

#pragma endregion Ranked Runs

#pragma region Index Construction

static void usearch_js_index_finalize(napi_env env, void* data, void* hint) {
    usearch_js_index_t* const index = (usearch_js_index_t*)data;
    (void)env;
    (void)hint;
    if (!index) return;
    usearch_dense_free(index->handle);
    free(index);
}

/**
 *  @brief @c CompiledIndex(dimensions, metric, quantization, connectivity, valuesPerKey, residency, placement).
 *
 *  The candidate-pool breadth is not here: it rides every call, which lets a recall sweep run
 *  against one immutable index with no mutation between points. The @p placement is the arm the
 *  residency selects - NUMA nodes, CUDA ordinals, or device paths - and the three public
 *  factories live in TypeScript, where each can take only its own.
 */
static napi_value usearch_js_index_construct(napi_env env, napi_callback_info info) {
    size_t arguments_count = 7;
    napi_value arguments[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    napi_value self = NULL;
    if (napi_get_cb_info(env, info, &arguments_count, arguments, &self, NULL) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not read the CompiledIndex arguments");
        return NULL;
    }
    if (arguments_count != 7) {
        usearch_js_throw(env, "ERR_INVALID_ARG_TYPE",                                            //
                         "CompiledIndex expects 7 arguments: dimensions, metric, quantization, " //
                         "connectivity, valuesPerKey, residency, placement");
        return NULL;
    }

    usearch_dense_options_t options;
    memset(&options, 0, sizeof(options));
    if (!usearch_js_size_from_value(env, arguments[0], "dimensions", &options.dimensions) ||
        !usearch_js_metric_from_value(env, arguments[1], "metric", &options.metric) ||
        !usearch_js_scalar_from_value(env, arguments[2], "quantization", &options.quantization) ||
        !usearch_js_size_from_value(env, arguments[3], "connectivity", &options.connectivity) ||
        !usearch_js_size_from_value(env, arguments[4], "valuesPerKey", &options.values_per_key) ||
        !usearch_js_residency_from_value(env, arguments[5], "residency", &options.residency))
        return NULL;

    usearch_js_paths_arena_t paths = {NULL, 0};
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    if (options.residency == usearch_residency_nvme_k) {
        if (!usearch_js_paths_from_value(env, arguments[6], "paths", &paths)) return NULL;
        options.nvme.paths = paths.views;
        options.nvme.paths_count = paths.count;
    }
    else {
        char const* const field = options.residency == usearch_residency_vram_k ? "devices" : "memoryDomains";
        if (!usearch_js_ordinals_from_value(env, arguments[6], field, &ordinals, &ordinals_count)) return NULL;
        if (options.residency == usearch_residency_vram_k) {
            options.vram.devices = ordinals;
            options.vram.devices_count = ordinals_count;
        }
        else {
            options.ram.memory_domains = ordinals;
            options.ram.memory_domains_count = ordinals_count;
        }
    }

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_index_t const handle = usearch_dense_init(&options, &error);
    usearch_js_paths_release(&paths);
    free(ordinals);
    if (!usearch_js_succeeded(env, &error)) return NULL;

    usearch_js_index_t* const index = (usearch_js_index_t*)malloc(sizeof(usearch_js_index_t));
    if (!index) {
        usearch_dense_free(handle);
        usearch_js_throw(env, "out-of-memory", "Could not allocate the CompiledIndex");
        return NULL;
    }
    index->handle = handle;
    if (napi_wrap(env, self, index, usearch_js_index_finalize, NULL, NULL) != napi_ok) {
        usearch_dense_free(handle);
        free(index);
        usearch_js_throw(env, "internal", "Could not attach the CompiledIndex");
        return NULL;
    }
    return self;
}

/** @brief Reads the receiver and, when @p arguments is non-NULL, the verb's arguments beside it. */
static usearch_js_index_t* usearch_js_index_self(napi_env env, napi_callback_info info, size_t* arguments_count,
                                                 napi_value* arguments) {
    napi_value self = NULL;
    if (napi_get_cb_info(env, info, arguments_count, arguments, &self, NULL) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not read the verb arguments");
        return NULL;
    }
    usearch_js_index_t* index = NULL;
    if (napi_unwrap(env, self, (void**)&index) != napi_ok || !index) {
        usearch_js_throw(env, "internal", "The CompiledIndex is not attached");
        return NULL;
    }
    return index;
}

/** @brief Refuses a verb whose argument count is not the one its signature spells. */
static bool usearch_js_arity(napi_env env, size_t taken, size_t wanted, char const* signature) {
    if (taken == wanted) return true;
    return usearch_js_throw(env, "ERR_INVALID_ARG_TYPE", signature);
}

#pragma endregion Index Construction

#pragma region Index Probes

static napi_value usearch_js_index_dimensions(napi_env env, napi_callback_info info) {
    usearch_js_index_t* const index = usearch_js_index_self(env, info, NULL, NULL);
    if (!index) return NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const value = usearch_dense_dimensions(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_number(env, value);
}

static napi_value usearch_js_index_connectivity(napi_env env, napi_callback_info info) {
    usearch_js_index_t* const index = usearch_js_index_self(env, info, NULL, NULL);
    if (!index) return NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const value = usearch_dense_connectivity(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_number(env, value);
}

static napi_value usearch_js_index_size(napi_env env, napi_callback_info info) {
    usearch_js_index_t* const index = usearch_js_index_self(env, info, NULL, NULL);
    if (!index) return NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const value = usearch_dense_size(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_number(env, value);
}

static napi_value usearch_js_index_capacity(napi_env env, napi_callback_info info) {
    usearch_js_index_t* const index = usearch_js_index_self(env, info, NULL, NULL);
    if (!index) return NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const value = usearch_dense_capacity(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_number(env, value);
}

static napi_value usearch_js_index_memory_usage(napi_env env, napi_callback_info info) {
    usearch_js_index_t* const index = usearch_js_index_self(env, info, NULL, NULL);
    if (!index) return NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const value = usearch_dense_memory_usage(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_number(env, value);
}

static napi_value usearch_js_index_storage_requirements(napi_env env, napi_callback_info info) {
    usearch_js_index_t* const index = usearch_js_index_self(env, info, NULL, NULL);
    if (!index) return NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const value = usearch_dense_storage_requirements(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_number(env, value);
}

static napi_value usearch_js_index_capabilities(napi_env env, napi_callback_info info) {
    usearch_js_index_t* const index = usearch_js_index_self(env, info, NULL, NULL);
    if (!index) return NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    char const* const value = usearch_dense_capabilities(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_static_string(env, value);
}

/** @brief @c reserve(capacity) - claims room for a total entry count ahead of insertion. */
static napi_value usearch_js_index_reserve(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value arguments[1] = {NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 1, "`reserve` expects 1 argument: capacity")) return NULL;

    size_t capacity = 0;
    if (!usearch_js_size_from_value(env, arguments[0], "capacity", &capacity)) return NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_reserve(index->handle, capacity, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_undefined(env);
}

/** @brief @c clear() - erases every entry, keeping the reserved capacity and frozen geometry. */
static napi_value usearch_js_index_clear(napi_env env, napi_callback_info info) {
    usearch_js_index_t* const index = usearch_js_index_self(env, info, NULL, NULL);
    if (!index) return NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_clear(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_undefined(env);
}

#pragma endregion Index Probes

#pragma region Insertion and Search

/** @brief Slots a telemetry sink carries, in the order a report writes them. */
enum { usearch_js_stats_slots_k = 3 };

/**
 *  @brief Borrows the @b optional telemetry sink every walking verb takes as its last argument.
 *
 *  @param[in] env The environment the call runs in.
 *  @param[in] value The @c Float64Array to report into; @c undefined asks for no telemetry.
 *  @param[out] sink Where the counters land, or @c NULL when none were asked for.
 *  @return @c true on success, @c false with an exception pending otherwise.
 */
static bool usearch_js_stats_from_value(napi_env env, napi_value value, double** sink) {
    *sink = NULL;
    napi_valuetype kind = napi_undefined;
    if (napi_typeof(env, value, &kind) != napi_ok)
        return usearch_js_throw_field(env, "stats", "could not be inspected");
    if (kind == napi_undefined || kind == napi_null) return true;

    napi_typedarray_type type = napi_int8_array;
    napi_value arraybuffer = NULL;
    size_t offset = 0, elements = 0;
    void* data = NULL;
    bool is_typedarray = false;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &type, &elements, &data, &arraybuffer, &offset) != napi_ok ||
        type != napi_float64_array || elements < usearch_js_stats_slots_k)
        return usearch_js_throw_field(env, "stats", "must be a Float64Array of three slots");
    *sink = (double*)data;
    return true;
}

/** @brief Writes one call's counters into @p sink, which the caller's accumulator folds in. */
static void usearch_js_stats_report(double* sink, usearch_dense_traversal_stats_t const* stats) {
    if (!sink) return;
    sink[0] = (double)stats->visited_members;
    sink[1] = (double)stats->computed_distances;
    sink[2] = (double)stats->restarts;
}

/** @brief @c add(keys, vectors, executor, expansion, scalarKind, stats) → rows published. */
static napi_value usearch_js_index_add(napi_env env, napi_callback_info info) {
    size_t arguments_count = 6;
    napi_value arguments[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 6,
                          "`add` expects 6 arguments: keys, vectors, executor, expansion, scalarKind, stats"))
        return NULL;

    usearch_js_matrix_t vectors;
    usearch_executor_t executor = NULL;
    usearch_scalar_t declared = usearch_scalar_unknown_k;
    usearch_dense_insertion_options_t options = {0};
    double* sink = NULL;
    if (!usearch_js_scalar_from_value(env, arguments[4], "scalarKind", &declared) ||
        !usearch_js_matrix_from_value(env, arguments[1], "vectors", declared, &vectors) ||
        !usearch_js_executor_from_value(env, arguments[2], &executor) ||
        !usearch_js_size_from_value(env, arguments[3], "expansion", &options.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[5], &sink))
        return NULL;

    size_t count = 0;
    usearch_key_t* const keys = usearch_js_keys_from_value(env, arguments[0], "keys", &count);
    if (!keys) return NULL;

    napi_value result = NULL;
    usearch_error_t error = {usearch_success_k, NULL};
    // The index defines the row, never the caller's array: a narrower row read at the frozen
    // width is an out-of-bounds read inside the engine.
    size_t const dimensions = usearch_dense_dimensions(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) goto cleanup;
    size_t const row_bytes = usearch_js_scalar_row_bytes(dimensions, vectors.scalar_kind);
    if (!row_bytes || vectors.bytes != row_bytes * count) {
        usearch_js_throw(env, "shape-mismatch", "The vectors do not split into one row per key");
        goto cleanup;
    }

    size_t const size = usearch_dense_size(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) goto cleanup;
    size_t const capacity = usearch_dense_capacity(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) goto cleanup;
    if (size + count > capacity) {
        usearch_dense_reserve(index->handle, size + count, &error);
        if (!usearch_js_succeeded(env, &error)) goto cleanup;
    }

    usearch_keys_view_t const identities = {keys, count};
    usearch_matrix_view_t const rows = {vectors.data, count, dimensions, 0, vectors.scalar_kind};
    usearch_dense_traversal_stats_t stats = {0};
    size_t const published = usearch_dense_add(index->handle, executor, identities, &rows, &options, NULL, NULL, &stats,
                                               &error);
    if (!usearch_js_succeeded(env, &error)) goto cleanup;
    usearch_js_stats_report(sink, &stats);
    result = usearch_js_number(env, published);

cleanup:
    free(keys);
    return result;
}

/** @brief @c refine(executor, expansion, stats) → nodes whose edges changed. */
static napi_value usearch_js_index_refine(napi_env env, napi_callback_info info) {
    size_t arguments_count = 3;
    napi_value arguments[3] = {NULL, NULL, NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 3, "`refine` expects 3 arguments: executor, expansion, stats"))
        return NULL;

    usearch_executor_t executor = NULL;
    usearch_dense_insertion_options_t options = {0};
    double* sink = NULL;
    if (!usearch_js_executor_from_value(env, arguments[0], &executor) ||
        !usearch_js_size_from_value(env, arguments[1], "expansion", &options.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[2], &sink))
        return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_traversal_stats_t stats = {0};
    size_t const changed = usearch_dense_refine(index->handle, executor, &options, NULL, NULL, &stats, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    usearch_js_stats_report(sink, &stats);
    return usearch_js_number(env, changed);
}

/**
 *  @brief @c search(queries, keys, distances, counts, executor, expansion, scalarKind, stats) →
 *      @c undefined, filling the three arrays the caller lends.
 *
 *  Answers with nothing: the outputs are the caller's arrays, which is what lets a recall sweep
 *  allocate once. The matches wanted per query follow from the key array's own length.
 */
static napi_value usearch_js_index_search(napi_env env, napi_callback_info info) {
    size_t arguments_count = 8;
    napi_value arguments[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 8,                                           //
                          "`search` expects 8 arguments: queries, keys, distances, counts, " //
                          "executor, expansion, scalarKind, stats"))
        return NULL;

    usearch_js_matrix_t queries;
    usearch_executor_t executor = NULL;
    usearch_scalar_t declared = usearch_scalar_unknown_k;
    usearch_dense_search_options_t options = {0};
    double* sink = NULL;
    if (!usearch_js_scalar_from_value(env, arguments[6], "scalarKind", &declared) ||
        !usearch_js_matrix_from_value(env, arguments[0], "queries", declared, &queries) ||
        !usearch_js_executor_from_value(env, arguments[4], &executor) ||
        !usearch_js_size_from_value(env, arguments[5], "expansion", &options.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[7], &sink))
        return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    size_t const dimensions = usearch_dense_dimensions(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    size_t const row_bytes = usearch_js_scalar_row_bytes(dimensions, queries.scalar_kind);
    if (!row_bytes || queries.bytes % row_bytes != 0) {
        usearch_js_throw(env, "shape-mismatch", "The queries do not split into rows of the index geometry");
        return NULL;
    }

    size_t const queries_count = queries.bytes / row_bytes;
    usearch_matches_span_t matches;
    if (!usearch_js_matches_from_value(env, arguments[1], arguments[2], arguments[3], queries_count, &matches))
        return NULL;

    usearch_matrix_view_t const rows = {queries.data, queries_count, dimensions, 0, queries.scalar_kind};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_dense_search(index->handle, executor, &rows, &matches, &options, NULL, NULL, &stats, &error);
    if (!usearch_js_succeeded(env, &error)) {
        usearch_js_matches_release(&matches);
        return NULL;
    }
    usearch_js_stats_report(sink, &stats);
    return usearch_js_matches_publish(env, arguments[3], queries_count, &matches);
}

/**
 *  @brief @c attend(queries, values, valueDimensions, outputs, carry, executor, expansion,
 *      scoreScale, massTolerance, queriesKind, valuesKind, stats) → @c undefined.
 *
 *  Writes into the @p outputs the caller lends rather than answering with a fresh array, so a
 *  sweep allocates once. The @p carry is a @c Float32Array of two floats per query row - the
 *  running maximum and the partition taken against it - and folding a second branch over the
 *  same queries needs both, which is why they share a row rather than travelling apart.
 */
static napi_value usearch_js_index_attend(napi_env env, napi_callback_info info) {
    size_t arguments_count = 12;
    napi_value arguments[12] = {NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 12,                                           //
                          "`attend` expects 12 arguments: queries, values, valueDimensions, " //
                          "outputs, carry, executor, expansion, scoreScale, massTolerance, "  //
                          "queriesKind, valuesKind, stats"))
        return NULL;

    usearch_js_matrix_t queries, values, outputs;
    size_t value_dimensions = 0;
    usearch_executor_t executor = NULL;
    usearch_scalar_t queries_declared = usearch_scalar_unknown_k, values_declared = usearch_scalar_unknown_k;
    usearch_dense_attention_options_t options = {0};
    double score_scale = 0, mass_tolerance = 0;
    double* sink = NULL;
    if (!usearch_js_scalar_from_value(env, arguments[9], "queriesKind", &queries_declared) ||
        !usearch_js_scalar_from_value(env, arguments[10], "valuesKind", &values_declared) ||
        !usearch_js_matrix_from_value(env, arguments[0], "queries", queries_declared, &queries) ||
        !usearch_js_matrix_from_value(env, arguments[1], "values", values_declared, &values) ||
        !usearch_js_size_from_value(env, arguments[2], "valueDimensions", &value_dimensions) ||
        !usearch_js_matrix_from_value(env, arguments[3], "outputs", usearch_scalar_unknown_k, &outputs) ||
        !usearch_js_executor_from_value(env, arguments[5], &executor) ||
        !usearch_js_size_from_value(env, arguments[6], "expansion", &options.expansion_factor) ||
        !usearch_js_double_from_value(env, arguments[7], "scoreScale", &score_scale) ||
        !usearch_js_double_from_value(env, arguments[8], "massTolerance", &mass_tolerance) ||
        !usearch_js_stats_from_value(env, arguments[11], &sink))
        return NULL;
    options.score_scale = (float)score_scale;
    options.mass_tolerance = (float)mass_tolerance;

    usearch_error_t error = {usearch_success_k, NULL};
    size_t const dimensions = usearch_dense_dimensions(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    if (!value_dimensions) {
        usearch_js_throw_field(env, "valueDimensions", "must be a positive integer");
        return NULL;
    }

    size_t const queries_row_bytes = usearch_js_scalar_row_bytes(dimensions, queries.scalar_kind);
    if (!queries_row_bytes || queries.bytes % queries_row_bytes != 0) {
        usearch_js_throw(env, "shape-mismatch", "The queries do not span whole rows of the index width");
        return NULL;
    }
    size_t const queries_count = queries.bytes / queries_row_bytes;

    size_t const values_row_bytes = usearch_js_scalar_row_bytes(value_dimensions, values.scalar_kind);
    if (!values_row_bytes || values.bytes % values_row_bytes != 0) {
        usearch_js_throw(env, "shape-mismatch", "The value rows do not span whole rows of `valueDimensions`");
        return NULL;
    }

    size_t const outputs_row_bytes = usearch_js_scalar_row_bytes(value_dimensions, outputs.scalar_kind);
    if (!outputs_row_bytes || outputs.bytes != outputs_row_bytes * queries_count) {
        usearch_js_throw(env, "shape-mismatch", "The outputs must hold one row of `valueDimensions` per query");
        return NULL;
    }

    // A caller folding this branch into another lends the carry; one omitting it starts fresh.
    // The element type is checked rather than declared: a Float64Array of one element per query
    // spans the same bytes as a Float32Array of two, and would be read as pairs it does not hold.
    usearch_dense_softmax_state_t* carried = NULL;
    if (!usearch_js_value_absent(env, arguments[4])) {
        napi_typedarray_type carry_type = napi_int8_array;
        napi_value carry_buffer = NULL;
        size_t carry_offset = 0, carry_elements = 0;
        void* carry_data = NULL;
        bool is_typedarray = false;
        if (napi_is_typedarray(env, arguments[4], &is_typedarray) != napi_ok || !is_typedarray ||
            napi_get_typedarray_info(env, arguments[4], &carry_type, &carry_elements, &carry_data, &carry_buffer,
                                     &carry_offset) != napi_ok ||
            carry_type != napi_float32_array) {
            usearch_js_throw_field(env, "carry", "must be a Float32Array");
            return NULL;
        }
        if (carry_elements != queries_count * 2) {
            usearch_js_throw(env, "shape-mismatch", "The carry must hold a maximum and a partition per query");
            return NULL;
        }
        carried = (usearch_dense_softmax_state_t*)carry_data;
    }

    usearch_matrix_view_t const query_rows = {queries.data, queries_count, dimensions, 0, queries.scalar_kind};
    usearch_matrix_view_t const value_rows = {values.data, values.bytes / values_row_bytes, value_dimensions, 0,
                                              values.scalar_kind};
    usearch_matrix_span_t const output_rows = {outputs.data, queries_count, value_dimensions, 0, outputs.scalar_kind};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_dense_attend(index->handle, executor, &query_rows, &value_rows, &output_rows, carried, &options, NULL, NULL,
                         &stats, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    usearch_js_stats_report(sink, &stats);
    return usearch_js_undefined(env);
}

/**
 *  @brief @c multiply(left, columns, products, counts, executor, expansion, scalarKind, stats) →
 *      @c undefined, filling the three arrays the caller lends.
 *
 *  One emitted pair per column rather than one ranked match, so the scores are unnormalized
 *  logits and the index must hold the inner-product kind. The pairs emitted per row follow from
 *  the column array's own length.
 */
static napi_value usearch_js_index_multiply(napi_env env, napi_callback_info info) {
    size_t arguments_count = 8;
    napi_value arguments[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 8,                                            //
                          "`multiply` expects 8 arguments: left, columns, products, counts, " //
                          "executor, expansion, scalarKind, stats"))
        return NULL;

    usearch_js_matrix_t left;
    usearch_executor_t executor = NULL;
    usearch_scalar_t declared = usearch_scalar_unknown_k;
    usearch_dense_search_options_t options = {0};
    double* sink = NULL;
    if (!usearch_js_scalar_from_value(env, arguments[6], "scalarKind", &declared) ||
        !usearch_js_matrix_from_value(env, arguments[0], "left", declared, &left) ||
        !usearch_js_executor_from_value(env, arguments[4], &executor) ||
        !usearch_js_size_from_value(env, arguments[5], "expansion", &options.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[7], &sink))
        return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    size_t const dimensions = usearch_dense_dimensions(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    size_t const row_bytes = usearch_js_scalar_row_bytes(dimensions, left.scalar_kind);
    if (!row_bytes || left.bytes % row_bytes != 0) {
        usearch_js_throw(env, "shape-mismatch", "The left rows do not span whole rows of the index width");
        return NULL;
    }

    size_t const rows_count = left.bytes / row_bytes;
    usearch_matches_span_t products;
    if (!usearch_js_matches_from_value(env, arguments[1], arguments[2], arguments[3], rows_count, &products))
        return NULL;

    usearch_matrix_view_t const rows = {left.data, rows_count, dimensions, 0, left.scalar_kind};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_dense_multiply(index->handle, executor, &rows, &products, &options, NULL, NULL, &stats, &error);
    if (!usearch_js_succeeded(env, &error)) {
        usearch_js_matches_release(&products);
        return NULL;
    }
    usearch_js_stats_report(sink, &stats);
    return usearch_js_matches_publish(env, arguments[3], rows_count, &products);
}

/**
 *  @brief @c maxsim(queries, rowsPerQuery, keys, distances, counts, executor, expansion,
 *      scalarKind, stats) → @c undefined, filling the three arrays the caller lends.
 *
 *  Every query row takes its best match within a key and those maxima sum to that key's score,
 *  so the run is over keys rather than rows.
 */
static napi_value usearch_js_index_maxsim(napi_env env, napi_callback_info info) {
    size_t arguments_count = 9;
    napi_value arguments[9] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 9,                                                 //
                          "`maxsim` expects 9 arguments: queries, rowsPerQuery, keys, distances, " //
                          "counts, executor, expansion, scalarKind, stats"))
        return NULL;

    usearch_js_matrix_t queries;
    size_t rows_per_query = 0;
    usearch_executor_t executor = NULL;
    usearch_scalar_t declared = usearch_scalar_unknown_k;
    usearch_dense_search_options_t options = {0};
    double* sink = NULL;
    if (!usearch_js_scalar_from_value(env, arguments[7], "scalarKind", &declared) ||
        !usearch_js_matrix_from_value(env, arguments[0], "queries", declared, &queries) ||
        !usearch_js_size_from_value(env, arguments[1], "rowsPerQuery", &rows_per_query) ||
        !usearch_js_executor_from_value(env, arguments[5], &executor) ||
        !usearch_js_size_from_value(env, arguments[6], "expansion", &options.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[8], &sink))
        return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    size_t const dimensions = usearch_dense_dimensions(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    if (!rows_per_query) {
        usearch_js_throw_field(env, "rowsPerQuery", "must be a positive integer");
        return NULL;
    }
    size_t const row_bytes = usearch_js_scalar_row_bytes(dimensions, queries.scalar_kind);
    if (!row_bytes || queries.bytes % (row_bytes * rows_per_query) != 0) {
        usearch_js_throw(env, "shape-mismatch", "The token rows do not split into whole queries");
        return NULL;
    }

    size_t const rows_count = queries.bytes / row_bytes;
    size_t const queries_count = rows_count / rows_per_query;
    usearch_matches_span_t matches;
    if (!usearch_js_matches_from_value(env, arguments[2], arguments[3], arguments[4], queries_count, &matches))
        return NULL;

    usearch_matrix_view_t const rows = {queries.data, rows_count, dimensions, 0, queries.scalar_kind};
    usearch_dense_traversal_stats_t stats = {0};
    usearch_dense_maxsim(index->handle, executor, &rows, rows_per_query, &matches, &options, NULL, NULL, &stats,
                         &error);
    if (!usearch_js_succeeded(env, &error)) {
        usearch_js_matches_release(&matches);
        return NULL;
    }
    usearch_js_stats_report(sink, &stats);
    return usearch_js_matches_publish(env, arguments[4], queries_count, &matches);
}

#pragma endregion Insertion and Search

#pragma region Membership

/** @brief @c count(keys) → one entry count per identity. */
static napi_value usearch_js_index_count(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value arguments[1] = {NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 1, "`count` expects 1 argument: keys")) return NULL;

    size_t count = 0;
    usearch_key_t* const keys = usearch_js_keys_from_value(env, arguments[0], "keys", &count);
    if (!keys) return NULL;

    napi_value result = NULL;
    if (napi_create_array_with_length(env, count, &result) != napi_ok) {
        usearch_js_throw(env, "out-of-memory", "Could not allocate the count array");
        result = NULL;
        goto cleanup;
    }
    for (size_t entry_index = 0; entry_index != count; ++entry_index) {
        usearch_error_t error = {usearch_success_k, NULL};
        size_t const entries = usearch_dense_count(index->handle, keys[entry_index], &error);
        if (!usearch_js_succeeded(env, &error)) {
            result = NULL;
            goto cleanup;
        }
        napi_set_element(env, result, (uint32_t)entry_index, usearch_js_number(env, entries));
    }

cleanup:
    free(keys);
    return result;
}

/** @brief @c contains(keys) → whether the index holds any entry under each identity. */
static napi_value usearch_js_index_contains(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value arguments[1] = {NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 1, "`contains` expects 1 argument: keys")) return NULL;

    size_t count = 0;
    usearch_key_t* const keys = usearch_js_keys_from_value(env, arguments[0], "keys", &count);
    if (!keys) return NULL;

    napi_value result = NULL;
    if (napi_create_array_with_length(env, count, &result) != napi_ok) {
        usearch_js_throw(env, "out-of-memory", "Could not allocate the membership array");
        result = NULL;
        goto cleanup;
    }
    for (size_t entry_index = 0; entry_index != count; ++entry_index) {
        usearch_error_t error = {usearch_success_k, NULL};
        size_t const entries = usearch_dense_count(index->handle, keys[entry_index], &error);
        if (!usearch_js_succeeded(env, &error)) {
            result = NULL;
            goto cleanup;
        }
        napi_value present = NULL;
        napi_get_boolean(env, entries != 0, &present);
        napi_set_element(env, result, (uint32_t)entry_index, present);
    }

cleanup:
    free(keys);
    return result;
}

#pragma endregion Membership

#pragma region Retrieval

/**
 *  @brief @c get(key) → the rows stored under one identity, laid end to end.
 *
 *  Rows come back as @c f32 whatever the index quantized them into, since that is the widest
 *  type a JavaScript number carries without a second representation.
 */
static napi_value usearch_js_index_get(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value arguments[1] = {NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 1, "`get` expects 1 argument: key")) return NULL;

    usearch_key_t key;
    if (!usearch_js_key_from_value(env, arguments[0], "key", &key)) return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    size_t const dimensions = usearch_dense_dimensions(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    size_t const entries = usearch_dense_get(index->handle, key, NULL, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;

    void* data = NULL;
    size_t const scalars = entries * dimensions;
    napi_value const buffer = usearch_js_buffer(env, scalars * sizeof(float), &data);
    if (!buffer) return NULL;
    if (entries) {
        usearch_matrix_span_t const rows = {data, entries, dimensions, 0, usearch_scalar_f32_k};
        usearch_dense_get(index->handle, key, &rows, &error);
        if (!usearch_js_succeeded(env, &error)) return NULL;
    }
    napi_value array = NULL;
    if (napi_create_typedarray(env, napi_float32_array, scalars, buffer, 0, &array) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not publish the vector array");
        return NULL;
    }
    return array;
}

#pragma endregion Retrieval

#pragma region Serialization

/** @brief @c save(path, executor) - writes the checkpoint the path's extension picks. */
static napi_value usearch_js_index_save(napi_env env, napi_callback_info info) {
    size_t arguments_count = 2;
    napi_value arguments[2] = {NULL, NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 2, "`save` expects 2 arguments: path, executor")) return NULL;

    char path[usearch_js_path_capacity_k];
    size_t path_length = 0;
    usearch_executor_t executor = NULL;
    if (!usearch_js_string_from_value(env, arguments[0], "path", path, sizeof(path), &path_length) ||
        !usearch_js_executor_from_value(env, arguments[1], &executor))
        return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_bytes_view_t const view = {(uint8_t const*)path, path_length};
    usearch_dense_save(index->handle, executor, view, NULL, NULL, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_undefined(env);
}

/** @brief @c load(path, executor) - materializes a checkpoint into this handle. */
static napi_value usearch_js_index_load(napi_env env, napi_callback_info info) {
    size_t arguments_count = 2;
    napi_value arguments[2] = {NULL, NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 2, "`load` expects 2 arguments: path, executor")) return NULL;

    char path[usearch_js_path_capacity_k];
    size_t path_length = 0;
    usearch_executor_t executor = NULL;
    if (!usearch_js_string_from_value(env, arguments[0], "path", path, sizeof(path), &path_length) ||
        !usearch_js_executor_from_value(env, arguments[1], &executor))
        return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_bytes_view_t const view = {(uint8_t const*)path, path_length};
    usearch_dense_load(index->handle, executor, view, NULL, NULL, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_undefined(env);
}

/** @brief @c saveBuffer(executor) → one monolith, sized by the storage requirements. */
static napi_value usearch_js_index_save_buffer(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value arguments[1] = {NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 1, "`saveBuffer` expects 1 argument: executor")) return NULL;

    usearch_executor_t executor = NULL;
    if (!usearch_js_executor_from_value(env, arguments[0], &executor)) return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    size_t const required = usearch_dense_storage_requirements(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;

    void* data = NULL;
    napi_value const buffer = usearch_js_buffer(env, required, &data);
    if (!buffer) return NULL;
    usearch_bytes_span_t const span = {(uint8_t*)data, required};
    usearch_dense_save_buffer(index->handle, executor, span, NULL, NULL, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;

    napi_value array = NULL;
    if (napi_create_typedarray(env, napi_uint8_array, required, buffer, 0, &array) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not publish the checkpoint buffer");
        return NULL;
    }
    return array;
}

/** @brief Borrows the bytes of a @c Uint8Array, which is how a checkpoint arrives without a file. */
static bool usearch_js_bytes_from_value(napi_env env, napi_value value, char const* field, usearch_bytes_view_t* view) {
    napi_typedarray_type type = napi_int8_array;
    napi_value arraybuffer = NULL;
    size_t offset = 0, elements = 0;
    void* data = NULL;
    bool is_typedarray = false;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &type, &elements, &data, &arraybuffer, &offset) != napi_ok ||
        type != napi_uint8_array)
        return usearch_js_throw_field(env, field, "must be a Uint8Array");
    view->data = (uint8_t const*)data;
    view->length = elements;
    return true;
}

/** @brief @c loadBuffer(buffer, executor) - materializes a monolith held in memory. */
static napi_value usearch_js_index_load_buffer(napi_env env, napi_callback_info info) {
    size_t arguments_count = 2;
    napi_value arguments[2] = {NULL, NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 2, "`loadBuffer` expects 2 arguments: buffer, executor")) return NULL;

    usearch_bytes_view_t buffer = {NULL, 0};
    usearch_executor_t executor = NULL;
    if (!usearch_js_bytes_from_value(env, arguments[0], "buffer", &buffer) ||
        !usearch_js_executor_from_value(env, arguments[1], &executor))
        return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_load_buffer(index->handle, executor, buffer, NULL, NULL, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_undefined(env);
}

/** @brief Wraps the geometry and metric a checkpoint froze; placement is never recorded. */
static napi_value usearch_js_metadata_object(napi_env env, usearch_dense_options_t const* options) {
    napi_value object = NULL;
    if (napi_create_object(env, &object) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not publish the checkpoint metadata");
        return NULL;
    }
    napi_set_named_property(env, object, "dimensions", usearch_js_number(env, options->dimensions));
    napi_set_named_property(env, object, "metric",
                            usearch_js_static_string(env, usearch_dense_metric_name(options->metric)));
    napi_set_named_property(env, object, "quantization",
                            usearch_js_static_string(env, usearch_scalar_name(options->quantization)));
    napi_set_named_property(env, object, "connectivity", usearch_js_number(env, options->connectivity));
    napi_set_named_property(env, object, "valuesPerKey", usearch_js_number(env, options->values_per_key));
    return object;
}

/** @brief @c CompiledIndex.metadata(path) → the geometry a checkpoint froze, without loading it. */
static napi_value usearch_js_index_metadata(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value arguments[1] = {NULL};
    if (napi_get_cb_info(env, info, &arguments_count, arguments, NULL, NULL) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not read the verb arguments");
        return NULL;
    }
    if (!usearch_js_arity(env, arguments_count, 1, "`metadata` expects 1 argument: path")) return NULL;

    char path[usearch_js_path_capacity_k];
    size_t path_length = 0;
    if (!usearch_js_string_from_value(env, arguments[0], "path", path, sizeof(path), &path_length)) return NULL;

    usearch_dense_options_t options;
    memset(&options, 0, sizeof(options));
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_bytes_view_t const view = {(uint8_t const*)path, path_length};
    usearch_dense_metadata(view, &options, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_metadata_object(env, &options);
}

/** @brief @c CompiledIndex.metadataBuffer(buffer) → the geometry a monolith in memory froze. */
static napi_value usearch_js_index_metadata_buffer(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value arguments[1] = {NULL};
    if (napi_get_cb_info(env, info, &arguments_count, arguments, NULL, NULL) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not read the verb arguments");
        return NULL;
    }
    if (!usearch_js_arity(env, arguments_count, 1, "`metadataBuffer` expects 1 argument: buffer")) return NULL;

    usearch_bytes_view_t buffer = {NULL, 0};
    if (!usearch_js_bytes_from_value(env, arguments[0], "buffer", &buffer)) return NULL;

    usearch_dense_options_t options;
    memset(&options, 0, sizeof(options));
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_metadata_buffer(buffer, &options, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_metadata_object(env, &options);
}

#pragma endregion Serialization

#pragma region Cursors

static void usearch_js_cursor_finalize(napi_env env, void* data, void* hint) {
    usearch_js_cursor_t* const cursor = (usearch_js_cursor_t*)data;
    (void)env;
    (void)hint;
    if (!cursor) return;
    if (cursor->handle) usearch_dense_cursor_free(cursor->handle);
    free(cursor);
}

/** @brief Reads the cursor behind the receiver, refusing one that has already been released. */
static usearch_js_cursor_t* usearch_js_cursor_self(napi_env env, napi_callback_info info, size_t* arguments_count,
                                                   napi_value* arguments) {
    napi_value self = NULL;
    usearch_js_cursor_t* cursor = NULL;
    if (napi_get_cb_info(env, info, arguments_count, arguments, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&cursor) != napi_ok || !cursor) {
        usearch_js_throw(env, "internal", "The cursor is not attached");
        return NULL;
    }
    if (!cursor->handle) {
        usearch_js_throw(env, "uninitialized", "The cursor was already released");
        return NULL;
    }
    return cursor;
}

/** @brief @c next(batch) → @c [keys, vectors] of up to @p batch rows, or @c null when drained. */
static napi_value usearch_js_cursor_next(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value arguments[1] = {NULL};
    usearch_js_cursor_t* const cursor = usearch_js_cursor_self(env, info, &arguments_count, arguments);
    if (!cursor) return NULL;
    if (!usearch_js_arity(env, arguments_count, 1, "`next` expects 1 argument: batch")) return NULL;

    size_t batch = 0;
    if (!usearch_js_size_from_value(env, arguments[0], "batch", &batch)) return NULL;
    if (!batch) {
        usearch_js_throw_field(env, "batch", "must be a positive integer");
        return NULL;
    }

    void* keys_data = NULL;
    void* vectors_data = NULL;
    napi_value const keys_buffer = usearch_js_buffer(env, batch * usearch_js_key_bytes_k, &keys_data);
    if (!keys_buffer) return NULL;
    napi_value const vectors_buffer = usearch_js_buffer(env, batch * cursor->dimensions * sizeof(float), &vectors_data);
    if (!vectors_buffer) return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_keys_span_t const identities = {(usearch_key_t*)keys_data, batch};
    usearch_matrix_span_t const rows = {vectors_data, batch, cursor->dimensions, 0, usearch_scalar_f32_k};
    size_t const written = usearch_dense_cursor_next(cursor->handle, identities, &rows, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    if (!written) {
        napi_value null_value = NULL;
        napi_get_null(env, &null_value);
        return null_value;
    }

    napi_value keys_array = NULL, vectors_array = NULL, result = NULL;
    if (napi_create_typedarray(env, napi_uint8_array, written * usearch_js_key_bytes_k, keys_buffer, 0, &keys_array) !=
            napi_ok ||
        napi_create_typedarray(env, napi_float32_array, written * cursor->dimensions, vectors_buffer, 0,
                               &vectors_array) != napi_ok ||
        napi_create_array_with_length(env, 2, &result) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not publish the cursor batch");
        return NULL;
    }
    napi_set_element(env, result, 0, keys_array);
    napi_set_element(env, result, 1, vectors_array);
    return result;
}

/** @brief @c free() - releases the snapshot; a second call is a no-op. */
static napi_value usearch_js_cursor_release(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    usearch_js_cursor_t* cursor = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&cursor) != napi_ok || !cursor) {
        usearch_js_throw(env, "internal", "The cursor is not attached");
        return NULL;
    }
    if (cursor->handle) usearch_dense_cursor_free(cursor->handle);
    cursor->handle = NULL;
    return usearch_js_undefined(env);
}

/** @brief @c snapshot() → a consistent cursor over the entries published so far. */
static napi_value usearch_js_index_snapshot(napi_env env, napi_callback_info info) {
    usearch_js_index_t* const index = usearch_js_index_self(env, info, NULL, NULL);
    if (!index) return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    size_t const dimensions = usearch_dense_dimensions(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    usearch_dense_cursor_t const handle = usearch_dense_cursor(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;

    static napi_property_descriptor const properties[] = {
        {"next", NULL, usearch_js_cursor_next, NULL, NULL, NULL, napi_default_method, NULL},
        {"free", NULL, usearch_js_cursor_release, NULL, NULL, NULL, napi_default_method, NULL},
    };
    napi_value object = NULL;
    usearch_js_cursor_t* const cursor = (usearch_js_cursor_t*)malloc(sizeof(usearch_js_cursor_t));
    if (!cursor) {
        usearch_dense_cursor_free(handle);
        usearch_js_throw(env, "out-of-memory", "Could not allocate the cursor");
        return NULL;
    }
    cursor->handle = handle;
    cursor->dimensions = dimensions;
    if (napi_create_object(env, &object) != napi_ok ||
        napi_define_properties(env, object, usearch_js_table_size_(properties), properties) != napi_ok ||
        napi_wrap(env, object, cursor, usearch_js_cursor_finalize, NULL, NULL) != napi_ok) {
        usearch_dense_cursor_free(handle);
        free(cursor);
        usearch_js_throw(env, "internal", "Could not attach the cursor");
        return NULL;
    }
    return object;
}

#pragma endregion Cursors

#pragma region Finders

static void usearch_js_finder_finalize(napi_env env, void* data, void* hint) {
    usearch_js_finder_t* const finder = (usearch_js_finder_t*)data;
    (void)env;
    (void)hint;
    if (!finder) return;
    if (finder->handle) usearch_dense_find_free(finder->handle);
    free(finder);
}

/** @brief Reads the finder behind the receiver, refusing one that has already been released. */
static usearch_js_finder_t* usearch_js_finder_self(napi_env env, napi_callback_info info, size_t* arguments_count,
                                                   napi_value* arguments) {
    napi_value self = NULL;
    usearch_js_finder_t* finder = NULL;
    if (napi_get_cb_info(env, info, arguments_count, arguments, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&finder) != napi_ok || !finder) {
        usearch_js_throw(env, "internal", "The finder is not attached");
        return NULL;
    }
    if (!finder->handle) {
        usearch_js_throw(env, "uninitialized", "The finder was already released");
        return NULL;
    }
    return finder;
}

/** @brief @c next(wanted) → @c [keys, scores] of up to @p wanted matches, or @c null when drained. */
static napi_value usearch_js_finder_next(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value arguments[1] = {NULL};
    usearch_js_finder_t* const finder = usearch_js_finder_self(env, info, &arguments_count, arguments);
    if (!finder) return NULL;
    if (!usearch_js_arity(env, arguments_count, 1, "`next` expects 1 argument: wanted")) return NULL;

    size_t wanted = 0;
    if (!usearch_js_size_from_value(env, arguments[0], "wanted", &wanted)) return NULL;
    if (!wanted) {
        usearch_js_throw_field(env, "wanted", "must be a positive integer");
        return NULL;
    }

    void* keys_data = NULL;
    void* scores_data = NULL;
    napi_value const keys_buffer = usearch_js_buffer(env, wanted * usearch_js_key_bytes_k, &keys_data);
    if (!keys_buffer) return NULL;
    napi_value const scores_buffer = usearch_js_buffer(env, wanted * sizeof(usearch_distance_t), &scores_data);
    if (!scores_buffer) return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_keys_span_t const identities = {(usearch_key_t*)keys_data, wanted};
    size_t const written = usearch_dense_find_next(finder->handle, identities, (usearch_distance_t*)scores_data,
                                                   &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    if (!written) {
        napi_value null_value = NULL;
        napi_get_null(env, &null_value);
        return null_value;
    }

    napi_value keys_array = NULL, scores_array = NULL, result = NULL;
    if (napi_create_typedarray(env, napi_uint8_array, written * usearch_js_key_bytes_k, keys_buffer, 0, &keys_array) !=
            napi_ok ||
        napi_create_typedarray(env, napi_float64_array, written, scores_buffer, 0, &scores_array) != napi_ok ||
        napi_create_array_with_length(env, 2, &result) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not publish the finder batch");
        return NULL;
    }
    napi_set_element(env, result, 0, keys_array);
    napi_set_element(env, result, 1, scores_array);
    return result;
}

/** @brief @c free() - releases the stream and the executor thread it pinned; a second call is a no-op. */
static napi_value usearch_js_finder_release(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    usearch_js_finder_t* finder = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&finder) != napi_ok || !finder) {
        usearch_js_throw(env, "internal", "The finder is not attached");
        return NULL;
    }
    if (finder->handle) usearch_dense_find_free(finder->handle);
    finder->handle = NULL;
    return usearch_js_undefined(env);
}

/** @brief @c find(query, executor, scalarKind) → a resumable stream over one query row. */
static napi_value usearch_js_index_find(napi_env env, napi_callback_info info) {
    size_t arguments_count = 3;
    napi_value arguments[3] = {NULL, NULL, NULL};
    usearch_js_index_t* const index = usearch_js_index_self(env, info, &arguments_count, arguments);
    if (!index) return NULL;
    if (!usearch_js_arity(env, arguments_count, 3, "`find` expects 3 arguments: query, executor, scalarKind"))
        return NULL;

    usearch_js_matrix_t query;
    usearch_executor_t executor = NULL;
    usearch_scalar_t declared = usearch_scalar_unknown_k;
    if (!usearch_js_scalar_from_value(env, arguments[2], "scalarKind", &declared) ||
        !usearch_js_matrix_from_value(env, arguments[0], "query", declared, &query) ||
        !usearch_js_executor_from_value(env, arguments[1], &executor))
        return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    size_t const dimensions = usearch_dense_dimensions(index->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    size_t const row_bytes = usearch_js_scalar_row_bytes(dimensions, query.scalar_kind);
    if (!row_bytes || query.bytes != row_bytes) {
        usearch_js_throw(env, "shape-mismatch", "The finder takes exactly one row of the index geometry");
        return NULL;
    }

    usearch_matrix_view_t const rows = {query.data, 1, dimensions, 0, query.scalar_kind};
    usearch_dense_finder_t const handle = usearch_dense_find(index->handle, executor, &rows, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;

    static napi_property_descriptor const properties[] = {
        {"next", NULL, usearch_js_finder_next, NULL, NULL, NULL, napi_default_method, NULL},
        {"free", NULL, usearch_js_finder_release, NULL, NULL, NULL, napi_default_method, NULL},
    };
    napi_value object = NULL;
    usearch_js_finder_t* const finder = (usearch_js_finder_t*)malloc(sizeof(usearch_js_finder_t));
    if (!finder) {
        usearch_dense_find_free(handle);
        usearch_js_throw(env, "out-of-memory", "Could not allocate the finder");
        return NULL;
    }
    finder->handle = handle;
    if (napi_create_object(env, &object) != napi_ok ||
        napi_define_properties(env, object, usearch_js_table_size_(properties), properties) != napi_ok ||
        napi_wrap(env, object, finder, usearch_js_finder_finalize, NULL, NULL) != napi_ok) {
        usearch_dense_find_free(handle);
        free(finder);
        usearch_js_throw(env, "internal", "Could not attach the finder");
        return NULL;
    }
    return object;
}

#pragma endregion Finders

#pragma region Registration

napi_value usearch_js_index_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"dimensions", NULL, usearch_js_index_dimensions, NULL, NULL, NULL, napi_default_method, NULL},
        {"connectivity", NULL, usearch_js_index_connectivity, NULL, NULL, NULL, napi_default_method, NULL},
        {"size", NULL, usearch_js_index_size, NULL, NULL, NULL, napi_default_method, NULL},
        {"capacity", NULL, usearch_js_index_capacity, NULL, NULL, NULL, napi_default_method, NULL},
        {"memoryUsage", NULL, usearch_js_index_memory_usage, NULL, NULL, NULL, napi_default_method, NULL},
        {"storageRequirements", NULL, usearch_js_index_storage_requirements, NULL, NULL, NULL, napi_default_method,
         NULL},
        {"capabilities", NULL, usearch_js_index_capabilities, NULL, NULL, NULL, napi_default_method, NULL},
        {"reserve", NULL, usearch_js_index_reserve, NULL, NULL, NULL, napi_default_method, NULL},
        {"clear", NULL, usearch_js_index_clear, NULL, NULL, NULL, napi_default_method, NULL},
        {"add", NULL, usearch_js_index_add, NULL, NULL, NULL, napi_default_method, NULL},
        {"refine", NULL, usearch_js_index_refine, NULL, NULL, NULL, napi_default_method, NULL},
        {"search", NULL, usearch_js_index_search, NULL, NULL, NULL, napi_default_method, NULL},
        {"maxsim", NULL, usearch_js_index_maxsim, NULL, NULL, NULL, napi_default_method, NULL},
        {"multiply", NULL, usearch_js_index_multiply, NULL, NULL, NULL, napi_default_method, NULL},
        {"attend", NULL, usearch_js_index_attend, NULL, NULL, NULL, napi_default_method, NULL},
        {"count", NULL, usearch_js_index_count, NULL, NULL, NULL, napi_default_method, NULL},
        {"contains", NULL, usearch_js_index_contains, NULL, NULL, NULL, napi_default_method, NULL},
        {"get", NULL, usearch_js_index_get, NULL, NULL, NULL, napi_default_method, NULL},
        {"snapshot", NULL, usearch_js_index_snapshot, NULL, NULL, NULL, napi_default_method, NULL},
        {"find", NULL, usearch_js_index_find, NULL, NULL, NULL, napi_default_method, NULL},
        {"save", NULL, usearch_js_index_save, NULL, NULL, NULL, napi_default_method, NULL},
        {"load", NULL, usearch_js_index_load, NULL, NULL, NULL, napi_default_method, NULL},
        {"saveBuffer", NULL, usearch_js_index_save_buffer, NULL, NULL, NULL, napi_default_method, NULL},
        {"loadBuffer", NULL, usearch_js_index_load_buffer, NULL, NULL, NULL, napi_default_method, NULL},
        {"metadata", NULL, usearch_js_index_metadata, NULL, NULL, NULL, napi_static, NULL},
        {"metadataBuffer", NULL, usearch_js_index_metadata_buffer, NULL, NULL, NULL, napi_static, NULL},
    };
    napi_value index_class = NULL;
    if (napi_define_class(env, "CompiledIndex", NAPI_AUTO_LENGTH, usearch_js_index_construct, NULL,
                          usearch_js_table_size_(properties), properties, &index_class) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not define the CompiledIndex class");
        return NULL;
    }
    return index_class;
}

#pragma endregion Registration
