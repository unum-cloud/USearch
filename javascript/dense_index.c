/**
 *  @file javascript/dense_index.c
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief Node-API vector search over equidimensional dense embeddings.
 *
 *  The @c CompiledIndex class and the lifecycle verbs that instantiate it, the cursor and the
 *  search stream it opens, and the checkpoint metadata readers that need no index at all.
 *
 *  Every instance verb takes the executor and the completion first, so one native function serves
 *  the calling thread, a blocking receiver and a deferred one alike.
 */
#include <stdlib.h> // `calloc`, `free`
#include <string.h> // `memcpy`, `memset`

#include "lib.h"

#pragma region Calls

/** A dense call: the shared header, then every output a dense verb lands before its answer. */
typedef struct usearch_js_dense_call_t {

    /** The outcome, the answer builder and the held values, shared by every modality. */
    usearch_js_call_t call;

    /** Traversal telemetry of a walking verb. */
    usearch_traversal_stats_t stats;

    /** The audit @c health reports. */
    usearch_reachability_t reachability;

    /** The report @c describe fills. */
    usearch_dense_index_description_t description;

    /** Donors an absorb names, which its settlement walks, or the rows on a verb's batch axis. */
    size_t count;

    /** Entries under a key, nodes a refinement changed, or the rows or pairs a call landed. */
    size_t written;

    /** Pairs a product scored before its bound cut any, or keys a late interaction landed. */
    size_t required;

    /** The ranked run a search or a stream pull landed in its arena. */
    usearch_matches_view_t matches;

    /** The pairs a product, or the keys a late-interaction scoring, landed in its arena. */
    usearch_matches_tape_view_t products;

    /** The rows a retrieval, a cursor pull or an attention landed in its arena. */
    usearch_rows_view_t vectors;

    /** The identities a cursor pull landed in its arena. */
    usearch_keys_view_t keys;

    /** The keys each of a cursor pull's entries links to, inside its arena. */
    usearch_keys_tape_view_t links;

    /** A monolith inside its arena. */
    usearch_bytes_view_t monolith;

    /** The kind the rows were asked in, which picks their element type even when none landed. */
    usearch_scalar_t scalar;
} usearch_js_dense_call_t;

/** Points the engine at @p call's telemetry when the caller passed a reading, nowhere otherwise. */
static bool usearch_js_stats_from_value(napi_env env, napi_value sink, usearch_js_dense_call_t* call,
                                        usearch_traversal_stats_t** stats) {
    napi_valuetype kind = napi_undefined;
    *stats = NULL;
    if (usearch_js_value_absent(env, sink)) return true;
    if (napi_typeof(env, sink, &kind) != napi_ok || kind != napi_object)
        return usearch_js_throw_field(env, "stats", "must be a DenseTraversalStats");
    *stats = &call->stats;
    return true;
}

/** Overwrites the reading @p sink with the counters one call reported; an absent one takes none. */
static bool usearch_js_stats_report(napi_env env, napi_value sink, usearch_traversal_stats_t const* stats) {
    if (usearch_js_value_absent(env, sink)) return true;
    if (napi_set_named_property(env, sink, "visitedMembers", usearch_js_number(env, stats->visited_members)) !=
            napi_ok ||
        napi_set_named_property(env, sink, "computedDistances", usearch_js_number(env, stats->computed_distances)) !=
            napi_ok ||
        napi_set_named_property(env, sink, "restarts", usearch_js_number(env, stats->restarts)) != napi_ok)
        return usearch_js_throw(env, usearch_status_internal_k, "Could not publish the telemetry");
    return true;
}

/** Wraps the geometry and metric a checkpoint froze; placement is never recorded. */
static napi_value usearch_js_metadata_object(napi_env env, usearch_dense_metadata_t const* metadata) {
    napi_value object = NULL;
    if (napi_create_object(env, &object) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not publish the checkpoint metadata");
        return NULL;
    }
    napi_set_named_property(env, object, "metric",
                            usearch_js_static_string(env, usearch_dense_metric_name(metadata->metric)));
    napi_set_named_property(env, object, "quantization",
                            usearch_js_static_string(env, usearch_scalar_name(metadata->quantization)));
    napi_set_named_property(env, object, "dimensions", usearch_js_number(env, metadata->dimensions));
    napi_set_named_property(env, object, "connectivity", usearch_js_number(env, metadata->connectivity));
    return object;
}

static napi_value usearch_js_finish_nothing(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    (void)call;
    (void)slots;
    return usearch_js_undefined(env);
}

static napi_value usearch_js_finish_stats(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    if (!usearch_js_stats_report(env, slots[usearch_js_slot_stats_k], &((usearch_js_dense_call_t*)call)->stats))
        return NULL;
    return usearch_js_undefined(env);
}

static napi_value usearch_js_finish_written(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    (void)slots;
    return usearch_js_number(env, ((usearch_js_dense_call_t*)call)->written);
}

static napi_value usearch_js_finish_refine(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_js_dense_call_t* const dense = (usearch_js_dense_call_t*)call;
    if (!usearch_js_stats_report(env, slots[usearch_js_slot_stats_k], &dense->stats)) return NULL;
    return usearch_js_number(env, dense->written);
}

static napi_value usearch_js_finish_health(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_reachability_t const* const reachability = &((usearch_js_dense_call_t*)call)->reachability;
    napi_value object = NULL;
    (void)slots;
    if (napi_create_object(env, &object) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not publish the reachability audit");
        return NULL;
    }
    napi_set_named_property(env, object, "audited", usearch_js_number(env, reachability->audited));
    napi_set_named_property(env, object, "reachable", usearch_js_number(env, reachability->reachable));
    napi_set_named_property(env, object, "sweeps", usearch_js_number(env, reachability->sweeps));
    napi_set_named_property(env, object, "backSweeps", usearch_js_number(env, reachability->back_sweeps));
    return object;
}

static napi_value usearch_js_finish_describe(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_dense_index_description_t const* const description = &((usearch_js_dense_call_t*)call)->description;
    napi_value const metadata = usearch_js_metadata_object(env, &description->metadata);
    napi_value object = NULL;
    (void)slots;
    if (!metadata || napi_create_object(env, &object) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not publish the index description");
        return NULL;
    }
    napi_set_named_property(env, object, "metadata", metadata);
    napi_set_named_property(env, object, "size", usearch_js_number(env, description->size));
    napi_set_named_property(env, object, "capacity", usearch_js_number(env, description->capacity));
    napi_set_named_property(env, object, "memoryUsage", usearch_js_number(env, description->memory_usage));
    napi_set_named_property(env, object, "capabilities", usearch_js_static_string(env, description->capabilities));
    return object;
}

#pragma endregion Calls

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
 *  @brief @c CompiledIndex() - an empty shell that a lifecycle verb fills with its handle.
 *
 *  Every handle comes from one of the @c init, @c load and @c view verbs, so construction takes
 *  nothing but the addon's seal, and JavaScript cannot make a shell of its own.
 */
static napi_value usearch_js_index_construct(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    if (!usearch_js_sealed(env, info, &self)) {
        usearch_js_throw(env, usearch_status_invalid_argument_k, "A CompiledIndex is made by a lifecycle verb");
        return NULL;
    }
    return self;
}

/** Wraps a handle a lifecycle verb just made, or throws its outcome and frees the handle. */
static napi_value usearch_js_index_adopt(napi_env env, usearch_outcome_t const* outcome, usearch_dense_index_t handle) {
    if (!usearch_js_succeeded(env, outcome)) return NULL;
    usearch_js_addon_t* addon = NULL;
    napi_value instance = NULL;
    usearch_js_index_t* const index = (usearch_js_index_t*)calloc(1, sizeof(usearch_js_index_t));
    if (napi_get_instance_data(env, (void**)&addon) == napi_ok && addon)
        instance = usearch_js_instantiate(env, addon->index_class);
    if (!instance || !index || napi_wrap(env, instance, index, usearch_js_index_finalize, NULL, NULL) != napi_ok) {
        free(index);
        usearch_dense_free(handle);
        usearch_js_throw(env, usearch_status_internal_k, "Could not instantiate the CompiledIndex");
        return NULL;
    }
    index->handle = handle;
    return instance;
}

/** Reads the receiver and @p wanted arguments beside it, missing ones reading as @c undefined. */
static usearch_js_index_t* usearch_js_index_self(napi_env env, napi_callback_info info, size_t wanted,
                                                 napi_value* arguments, napi_value* self) {
    size_t arguments_count = wanted;
    usearch_js_index_t* index = NULL;
    if (napi_get_cb_info(env, info, &arguments_count, arguments, self, NULL) != napi_ok ||
        napi_unwrap(env, *self, (void**)&index) != napi_ok || !index) {
        usearch_js_throw(env, usearch_status_internal_k, "The CompiledIndex is not attached");
        return NULL;
    }
    for (size_t position = arguments_count; position < wanted; ++position)
        arguments[position] = usearch_js_undefined(env);
    if (!index->handle) {
        usearch_js_throw(env, usearch_status_uninitialized_k, "The index was freed or absorbed into another");
        return NULL;
    }
    return index;
}

/** Reads @p wanted arguments of a static verb, which has no receiver to unwrap. */
static bool usearch_js_static_arguments(napi_env env, napi_callback_info info, size_t wanted, napi_value* arguments) {
    size_t arguments_count = wanted;
    if (napi_get_cb_info(env, info, &arguments_count, arguments, NULL, NULL) != napi_ok)
        return usearch_js_throw(env, usearch_status_internal_k, "Could not read the verb arguments");
    for (size_t position = arguments_count; position < wanted; ++position)
        arguments[position] = usearch_js_undefined(env);
    return true;
}

/**
 *  @brief @c initRam(dimensions, metric, quantization, connectivity, valuesPerKey, memoryDomains)
 *      and @c initVram(..., devices) → an empty, mutable index.
 *
 *  The breadth of the candidates is not here: it rides every call, which lets a recall sweep run
 *  against one immutable index with no mutation between points.
 */
static napi_value usearch_js_index_init(napi_env env, napi_callback_info info, usearch_residency_t residency) {
    napi_value arguments[6];
    size_t dimensions = 0, connectivity = 0, values_per_key = 0;
    usearch_dense_metric_t metric = usearch_dense_metric_unknown_k;
    usearch_scalar_t quantization = usearch_scalar_unknown_k;
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    char const* const field = residency == usearch_residency_vram_k ? "devices" : "memoryDomains";
    if (!usearch_js_static_arguments(env, info, 6, arguments) ||
        !usearch_js_size_from_value(env, arguments[0], "dimensions", &dimensions) ||
        !usearch_js_metric_from_value(env, arguments[1], "metric", &metric) ||
        !usearch_js_scalar_from_value(env, arguments[2], "quantization", &quantization) ||
        !usearch_js_size_from_value(env, arguments[3], "connectivity", &connectivity) ||
        !usearch_js_size_from_value(env, arguments[4], "valuesPerKey", &values_per_key) ||
        !usearch_js_ordinals_from_value(env, arguments[5], field, &ordinals, &ordinals_count))
        return NULL;

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    if (residency == usearch_residency_vram_k) {
        usearch_dense_init_vram_t init = {.index = &handle,
                                          .outcome = &outcome,
                                          .metric = metric,
                                          .quantization = quantization,
                                          .dimensions = dimensions,
                                          .connectivity = connectivity,
                                          .values_per_key = values_per_key,
                                          .vram = {ordinals, ordinals_count}};
        usearch_dense_init_vram(&init);
    }
    else {
        usearch_dense_init_ram_t init = {.index = &handle,
                                         .outcome = &outcome,
                                         .metric = metric,
                                         .quantization = quantization,
                                         .dimensions = dimensions,
                                         .connectivity = connectivity,
                                         .values_per_key = values_per_key,
                                         .ram = {ordinals, ordinals_count}};
        usearch_dense_init_ram(&init);
    }
    free(ordinals);
    return usearch_js_index_adopt(env, &outcome, handle);
}

static napi_value usearch_js_index_init_ram(napi_env env, napi_callback_info info) {
    return usearch_js_index_init(env, info, usearch_residency_ram_k);
}

static napi_value usearch_js_index_init_vram(napi_env env, napi_callback_info info) {
    return usearch_js_index_init(env, info, usearch_residency_vram_k);
}

/** Where a load reads its checkpoint from. */
typedef enum usearch_js_source_t {
    usearch_js_source_path_k,
    usearch_js_source_buffer_k,
} usearch_js_source_t;

/** @c loadRam(path, executor, memoryDomains, valuesPerKey), @c loadVram(path, executor, devices,
 *  valuesPerKey), and their @c Buffer twins taking any @c TypedArray → a new, mutable index. */
static napi_value usearch_js_index_load(napi_env env, napi_callback_info info, usearch_residency_t residency,
                                        usearch_js_source_t source) {
    napi_value arguments[4];
    if (!usearch_js_static_arguments(env, info, 4, arguments)) return NULL;

    char* path = NULL;
    void* buffer = NULL;
    usearch_bytes_view_t bytes = {NULL, 0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0;
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    char const* const field = residency == usearch_residency_vram_k ? "devices" : "memoryDomains";
    if (source == usearch_js_source_path_k) path = usearch_js_string_from_value(env, arguments[0], "path", &bytes);
    else if (usearch_js_bytes_from_value(env, arguments[0], "buffer", &buffer, &bytes.length))
        bytes.data = (uint8_t const*)buffer;
    bool const read = source == usearch_js_source_path_k ? path != NULL : bytes.data != NULL || !bytes.length;
    if (!read || !usearch_js_executor_from_value(env, arguments[1], &executor) ||
        !usearch_js_size_from_value(env, arguments[3], "valuesPerKey", &values_per_key) ||
        !usearch_js_ordinals_from_value(env, arguments[2], field, &ordinals, &ordinals_count)) {
        free(path);
        return NULL;
    }

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_ram_options_t const ram = {ordinals, ordinals_count};
    usearch_vram_options_t const vram = {ordinals, ordinals_count};
    if (residency == usearch_residency_vram_k && source == usearch_js_source_path_k) {
        usearch_dense_load_vram_t load = {.index = &handle,
                                          .executor = executor,
                                          .outcome = &outcome,
                                          .path = bytes,
                                          .vram = vram,
                                          .values_per_key = values_per_key};
        usearch_dense_load_vram(&load);
    }
    else if (residency == usearch_residency_vram_k) {
        usearch_dense_load_vram_buffer_t load = {.index = &handle,
                                                 .executor = executor,
                                                 .outcome = &outcome,
                                                 .buffer = bytes,
                                                 .vram = vram,
                                                 .values_per_key = values_per_key};
        usearch_dense_load_vram_buffer(&load);
    }
    else if (source == usearch_js_source_path_k) {
        usearch_dense_load_ram_t load = {.index = &handle,
                                         .executor = executor,
                                         .outcome = &outcome,
                                         .path = bytes,
                                         .ram = ram,
                                         .values_per_key = values_per_key};
        usearch_dense_load_ram(&load);
    }
    else {
        usearch_dense_load_ram_buffer_t load = {.index = &handle,
                                                .executor = executor,
                                                .outcome = &outcome,
                                                .buffer = bytes,
                                                .ram = ram,
                                                .values_per_key = values_per_key};
        usearch_dense_load_ram_buffer(&load);
    }
    free(ordinals);
    free(path);
    return usearch_js_index_adopt(env, &outcome, handle);
}

static napi_value usearch_js_index_load_ram(napi_env env, napi_callback_info info) {
    return usearch_js_index_load(env, info, usearch_residency_ram_k, usearch_js_source_path_k);
}

static napi_value usearch_js_index_load_vram(napi_env env, napi_callback_info info) {
    return usearch_js_index_load(env, info, usearch_residency_vram_k, usearch_js_source_path_k);
}

static napi_value usearch_js_index_load_ram_buffer(napi_env env, napi_callback_info info) {
    return usearch_js_index_load(env, info, usearch_residency_ram_k, usearch_js_source_buffer_k);
}

static napi_value usearch_js_index_load_vram_buffer(napi_env env, napi_callback_info info) {
    return usearch_js_index_load(env, info, usearch_residency_vram_k, usearch_js_source_buffer_k);
}

/** @c viewRam(path) → a read-only index serving the checkpoint through a memory mapping. */
static napi_value usearch_js_index_view_ram(napi_env env, napi_callback_info info) {
    napi_value argument = NULL;
    usearch_dense_view_ram_t view_ram = {0};
    if (!usearch_js_static_arguments(env, info, 1, &argument)) return NULL;
    char* const path = usearch_js_string_from_value(env, argument, "path", &view_ram.path);
    if (!path) return NULL;

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    view_ram.index = &handle;
    view_ram.outcome = &outcome;
    usearch_dense_view_ram(&view_ram);
    free(path);
    return usearch_js_index_adopt(env, &outcome, handle);
}

/** @c viewNvme(path, executor, paths) → a read-only index served from NVMe drives. */
static napi_value usearch_js_index_view_nvme(napi_env env, napi_callback_info info) {
    napi_value arguments[3];
    usearch_dense_view_nvme_t view_nvme = {0};
    usearch_js_paths_arena_t paths = {NULL, 0};
    if (!usearch_js_static_arguments(env, info, 3, arguments)) return NULL;
    char* const path = usearch_js_string_from_value(env, arguments[0], "path", &view_nvme.path);
    if (!path || !usearch_js_executor_from_value(env, arguments[1], &view_nvme.executor) ||
        !usearch_js_paths_from_value(env, arguments[2], "paths", &paths)) {
        free(path);
        return NULL;
    }

    usearch_dense_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    view_nvme.index = &handle;
    view_nvme.outcome = &outcome;
    view_nvme.nvme = (usearch_nvme_options_t) {paths.views, paths.count};
    usearch_dense_view_nvme(&view_nvme);
    usearch_js_paths_release(&paths);
    free(path);
    return usearch_js_index_adopt(env, &outcome, handle);
}

/** @c metadataRead(path, executor) and @c metadataReadBuffer(buffer, executor) → what a checkpoint
 *  froze. */
static napi_value usearch_js_index_metadata_read(napi_env env, napi_callback_info info, usearch_js_source_t source) {
    napi_value arguments[2];
    usearch_dense_metadata_t metadata;
    memset(&metadata, 0, sizeof(metadata));
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t executor = NULL;
    if (!usearch_js_static_arguments(env, info, 2, arguments) ||
        !usearch_js_executor_from_value(env, arguments[1], &executor))
        return NULL;
    if (source == usearch_js_source_path_k) {
        usearch_dense_metadata_read_t read = {.executor = executor, .outcome = &outcome, .metadata = &metadata};
        char* const path = usearch_js_string_from_value(env, arguments[0], "path", &read.path);
        if (!path) return NULL;
        usearch_dense_metadata_read(&read);
        free(path);
    }
    else {
        void* buffer = NULL;
        usearch_dense_metadata_read_buffer_t read = {.executor = executor, .outcome = &outcome, .metadata = &metadata};
        if (!usearch_js_bytes_from_value(env, arguments[0], "buffer", &buffer, &read.buffer.length)) return NULL;
        read.buffer.data = (uint8_t const*)buffer;
        usearch_dense_metadata_read_buffer(&read);
    }
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    return usearch_js_metadata_object(env, &metadata);
}

static napi_value usearch_js_index_metadata_read_path(napi_env env, napi_callback_info info) {
    return usearch_js_index_metadata_read(env, info, usearch_js_source_path_k);
}

static napi_value usearch_js_index_metadata_read_buffer(napi_env env, napi_callback_info info) {
    return usearch_js_index_metadata_read(env, info, usearch_js_source_buffer_k);
}

/** @c free() - releases the handle now, waiting for every deferred call naming it; idempotent. */
static napi_value usearch_js_index_release(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    usearch_js_index_t* index = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&index) != napi_ok || !index) {
        usearch_js_throw(env, usearch_status_internal_k, "The CompiledIndex is not attached");
        return NULL;
    }
    usearch_dense_free(index->handle);
    index->handle = NULL;
    return usearch_js_undefined(env);
}

#pragma endregion Index Construction

#pragma region Index Verbs

/**
 *  @brief Opens the call every instance verb runs through, reading the executor and the completion.
 *  @return The call, or @c NULL with an exception pending.
 */
static usearch_js_dense_call_t* usearch_js_dense_open(napi_env env, napi_value const* arguments,
                                                      usearch_js_dense_call_t* local, usearch_executor_t* executor,
                                                      usearch_completion_t* completion) {
    if (!usearch_js_executor_from_value(env, arguments[0], executor) ||
        !usearch_js_completion_from_value(env, arguments[1], completion))
        return NULL;
    return (usearch_js_dense_call_t*)usearch_js_call_open(env, *completion, &local->call, sizeof(*local));
}

/** @c describe(executor, completion) → the frozen metadata and current counters, each read once. */
static napi_value usearch_js_index_describe(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_describe_t describe = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &describe.executor, &describe.completion) : NULL;
    if (!call) return NULL;
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_describe;
    describe.outcome = &call->call.outcome;
    describe.description = &call->description;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_describe(&describe);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** @c health(executor, completion) → how much of the graph a search reaches from the landmarks. */
static napi_value usearch_js_index_health(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_health_t health = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &health.executor, &health.completion) : NULL;
    if (!call) return NULL;
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_health;
    health.outcome = &call->call.outcome;
    health.reachability = &call->reachability;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_health(&health);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** @c reserve(executor, completion, capacity) - claims room for a total entry count up front. */
static napi_value usearch_js_index_reserve(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_reserve_t reserve = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &reserve.executor, &reserve.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_size_from_value(env, arguments[2], "capacity", &reserve.capacity)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_nothing;
    reserve.outcome = &call->call.outcome;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_reserve(&reserve);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** @c clear(executor, completion) - erases every entry, keeping capacity and frozen geometry. */
static napi_value usearch_js_index_clear(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_clear_t clear = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &clear.executor, &clear.completion) : NULL;
    if (!call) return NULL;
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_nothing;
    clear.outcome = &call->call.outcome;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_clear(&clear);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** @c count(executor, completion, key) → the entries stored under one identity. */
static napi_value usearch_js_index_count(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_count_t count = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &count.executor, &count.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_key_from_value(env, arguments[2], "key", &count.key)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_written;
    count.outcome = &call->call.outcome;
    count.entries = &call->written;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_count(&count);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** @c insert(executor, completion, keys, data, rows, columns, stride, scalar, expansion, stats). */
static napi_value usearch_js_index_insert(napi_env env, napi_callback_info info) {
    napi_value arguments[10], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 10, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_insert_t insert = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &insert.executor, &insert.completion) : NULL;
    if (!call) return NULL;
    size_t keys_count = 0;
    usearch_key_t** const gathered = (usearch_key_t**)&call->call.owned[0];
    if (!usearch_js_rows_from_value(env, arguments + 3, "vectors", &insert.count, &insert.vectors) ||
        !usearch_js_size_from_value(env, arguments[8], "expansionFactor", &insert.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[9], call, &insert.stats) ||
        !usearch_js_keys_from_value(env, arguments[2], "keys", &insert.keys.data, &keys_count, gathered)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    // The keys view carries no count, so only this side can see the two disagree; one key
    // broadcasts.
    if (keys_count != insert.count && keys_count != 1) {
        usearch_js_call_abandon(&call->call, &local.call);
        usearch_js_throw(env, usearch_status_shape_mismatch_k, "The keys name neither one key nor one per row");
        return NULL;
    }
    insert.keys.stride = keys_count == 1 ? 0 : sizeof(usearch_key_t);
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], NULL, arguments[3], arguments[2], arguments[9]};
    call->call.finish = usearch_js_finish_stats;
    insert.outcome = &call->call.outcome;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_insert(&insert);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** @c refine(executor, completion, expansion, stats) → the nodes whose edges changed. */
static napi_value usearch_js_index_refine(napi_env env, napi_callback_info info) {
    napi_value arguments[4], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 4, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_refine_t refine = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &refine.executor, &refine.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_size_from_value(env, arguments[2], "expansionFactor", &refine.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[3], call, &refine.stats)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], NULL, NULL, NULL, arguments[3]};
    call->call.finish = usearch_js_finish_refine;
    refine.outcome = &call->call.outcome;
    refine.changed = &call->written;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_refine(&refine);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** Builds an object of @p count named values, or answers @c NULL with an exception pending. */
static napi_value usearch_js_object(napi_env env, size_t count, char const* const* names, napi_value const* values) {
    napi_value object = NULL;
    bool built = napi_create_object(env, &object) == napi_ok;
    for (size_t position = 0; built && position != count; ++position)
        built = napi_set_named_property(env, object, names[position], values[position]) == napi_ok;
    if (built) return object;
    usearch_js_throw(env, usearch_status_internal_k, "Could not publish the outputs");
    return NULL;
}

/** Views a ranked run of @p count rows inside @p arena as @c {keys, scores, counts, columns}. */
static napi_value usearch_js_matches_object(napi_env env, napi_value arena, usearch_matches_view_t const* matches,
                                            size_t count) {
    size_t const cells = count * matches->columns;
    napi_value const keys = usearch_js_arena_view(env, arena, napi_uint8_array, (void*)matches->keys,
                                                  cells * sizeof(usearch_key_t), cells * sizeof(usearch_key_t));
    napi_value const scores = keys ? usearch_js_arena_view(env, arena, napi_float64_array, (void*)matches->scores,
                                                           cells * sizeof(usearch_distance_t), cells)
                                   : NULL;
    napi_value const counts = scores ? usearch_js_arena_view(env, arena, napi_biguint64_array, (void*)matches->counts,
                                                             count * sizeof(size_t), count)
                                     : NULL;
    if (!counts) return NULL;
    static char const* const names[] = {"keys", "scores", "counts", "columns"};
    napi_value const values[] = {keys, scores, counts, usearch_js_number(env, matches->columns)};
    return usearch_js_object(env, usearch_js_table_size_(names), names, values);
}

/** The @c TypedArray a kind reads back as; kinds JavaScript spells no element for stay bytes. */
static napi_typedarray_type usearch_js_typedarray_of(usearch_scalar_t scalar, size_t* unit_bytes) {
    switch (scalar) {
    case usearch_scalar_f64_k: *unit_bytes = 8; return napi_float64_array;
    case usearch_scalar_f32_k: *unit_bytes = 4; return napi_float32_array;
#ifdef NODE_API_HAS_FLOAT16_ARRAY
    case usearch_scalar_f16_k: *unit_bytes = 2; return napi_float16_array;
#else
    case usearch_scalar_f16_k: *unit_bytes = 2; return napi_uint16_array;
#endif
    case usearch_scalar_bf16_k: *unit_bytes = 2; return napi_uint16_array;
    case usearch_scalar_i8_k: *unit_bytes = 1; return napi_int8_array;
    default: *unit_bytes = 1; return napi_uint8_array;
    }
}

/** Views @p count rows a call landed inside @p arena as @c {data, rows, columns, stride}, typed by
 *  @p call's kind. */
static napi_value usearch_js_rows_object(napi_env env, napi_value arena, usearch_js_dense_call_t const* call,
                                         size_t count) {
    size_t unit_bytes = 1;
    napi_typedarray_type const type = usearch_js_typedarray_of(call->scalar, &unit_bytes);
    size_t const bytes = count * call->vectors.stride;
    napi_value const data = usearch_js_arena_view(env, arena, type, (void*)call->vectors.data, bytes,
                                                  bytes / unit_bytes);
    if (!data) return NULL;
    static char const* const names[] = {"data", "rows", "columns", "stride"};
    napi_value const values[] = {data, usearch_js_number(env, count), usearch_js_number(env, call->vectors.columns),
                                 usearch_js_number(env, call->vectors.stride)};
    return usearch_js_object(env, usearch_js_table_size_(names), names, values);
}

static napi_value usearch_js_finish_matches(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_js_dense_call_t const* const dense = (usearch_js_dense_call_t const*)call;
    if (!usearch_js_stats_report(env, slots[usearch_js_slot_stats_k], &dense->stats)) return NULL;
    return usearch_js_matches_object(env, slots[usearch_js_slot_arena_k], &dense->matches, dense->count);
}

/** @c search(executor, completion, data, rows, columns, stride, scalar, arena, wanted, expansion,
 *  traversal, stats) → @c {keys, scores, counts, columns}, viewed inside @p arena. */
static napi_value usearch_js_index_search(napi_env env, napi_callback_info info) {
    napi_value arguments[12], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 12, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_search_t search = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &search.executor, &search.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_rows_from_value(env, arguments + 2, "queries", &search.count, &search.queries) ||
        !usearch_js_size_from_value(env, arguments[8], "wanted", &search.wanted) ||
        !usearch_js_size_from_value(env, arguments[9], "expansionFactor", &search.expansion_factor) ||
        !usearch_js_traversal_from_value(env, arguments[10], "traversal", &search.traversal) ||
        !usearch_js_stats_from_value(env, arguments[11], call, &search.stats) ||
        !usearch_js_arena_from_value(env, arguments[7], &call->call, &search.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[7], arguments[2], NULL, arguments[11]};
    call->call.finish = usearch_js_finish_matches;
    call->count = search.count;
    search.outcome = &call->call.outcome;
    search.matches = &call->matches;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_search(&search);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** Views the pairs and their Arrow offsets inside the arena. */
static napi_value usearch_js_finish_multiply(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_js_dense_call_t const* const dense = (usearch_js_dense_call_t const*)call;
    napi_value const arena = slots[usearch_js_slot_arena_k];
    size_t const pairs = dense->written;
    if (!usearch_js_stats_report(env, slots[usearch_js_slot_stats_k], &dense->stats)) return NULL;
    napi_value const offsets = usearch_js_arena_view(env, arena, napi_biguint64_array, (void*)dense->products.offsets,
                                                     (dense->count + 1) * sizeof(size_t), dense->count + 1);
    napi_value const keys = offsets
                                ? usearch_js_arena_view(env, arena, napi_uint8_array, (void*)dense->products.keys,
                                                        pairs * sizeof(usearch_key_t), pairs * sizeof(usearch_key_t))
                                : NULL;
    napi_value const scores = keys
                                  ? usearch_js_arena_view(env, arena, napi_float64_array, (void*)dense->products.scores,
                                                          pairs * sizeof(usearch_distance_t), pairs)
                                  : NULL;
    if (!scores) return NULL;
    static char const* const names[] = {"offsets", "keys", "scores", "required"};
    napi_value const values[] = {offsets, keys, scores, usearch_js_number(env, dense->required)};
    return usearch_js_object(env, usearch_js_table_size_(names), names, values);
}

/**
 *  @brief @c maxsim(executor, completion, data, rows, columns, stride, scalar, rowsPerQuery, arena,
 *      wanted, expansion, stats) → @c {offsets, keys, scores, required}, viewed inside @p arena.
 *
 *  Every query row takes its best match within a key and those maxima sum to that key's score,
 *  so query @c i spans keys @c offsets[i] up to @c offsets[i+1], largest first; a zero @p wanted
 *  keeps every key it scored.
 */
static napi_value usearch_js_index_maxsim(napi_env env, napi_callback_info info) {
    napi_value arguments[12], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 12, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_maxsim_t maxsim = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &maxsim.executor, &maxsim.completion) : NULL;
    if (!call) return NULL;
    size_t rows = 0;
    if (!usearch_js_rows_from_value(env, arguments + 2, "queries", &rows, &maxsim.queries) ||
        !usearch_js_size_from_value(env, arguments[7], "rowsPerQuery", &maxsim.rows_per_query)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    // The engine sees a query count, never the total rows it was derived from.
    if (!maxsim.rows_per_query || rows % maxsim.rows_per_query != 0) {
        usearch_js_call_abandon(&call->call, &local.call);
        usearch_js_throw(env, usearch_status_shape_mismatch_k, "The rows do not split into whole queries");
        return NULL;
    }
    maxsim.count = rows / maxsim.rows_per_query;
    if (!usearch_js_size_from_value(env, arguments[9], "wanted", &maxsim.wanted) ||
        !usearch_js_size_from_value(env, arguments[10], "expansionFactor", &maxsim.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[11], call, &maxsim.stats) ||
        !usearch_js_arena_from_value(env, arguments[8], &call->call, &maxsim.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[8], arguments[2], NULL, arguments[11]};
    call->call.finish = usearch_js_finish_multiply;
    call->count = maxsim.count;
    maxsim.outcome = &call->call.outcome;
    maxsim.matches = &call->products;
    maxsim.written = &call->written;
    maxsim.required = &call->required;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_maxsim(&maxsim);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

static napi_value usearch_js_finish_get(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_js_dense_call_t const* const dense = (usearch_js_dense_call_t const*)call;
    return usearch_js_rows_object(env, slots[usearch_js_slot_arena_k], dense, dense->written);
}

/** @c get(executor, completion, key, scalar, arena) → @c {data, rows, columns, stride}, viewed
 *  inside @p arena. */
static napi_value usearch_js_index_get(napi_env env, napi_callback_info info) {
    napi_value arguments[5], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 5, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_get_t get = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &get.executor, &get.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_key_from_value(env, arguments[2], "key", &get.key) ||
        !usearch_js_scalar_from_value(env, arguments[3], "scalar", &get.outputs_scalar) ||
        !usearch_js_arena_from_value(env, arguments[4], &call->call, &get.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[4]};
    call->call.finish = usearch_js_finish_get;
    call->scalar = get.outputs_scalar;
    get.outcome = &call->call.outcome;
    get.vectors = &call->vectors;
    get.written = &call->written;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_get(&get);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/**
 *  @brief @c multiply(executor, completion, data, rows, columns, stride, scalar, arena, wanted,
 *      expansion, stats) → @c {offsets, keys, scores, required}, viewed inside @p arena.
 *
 *  How many pairs a walk scores is known only once it ends, so the pairs land in @p arena as a
 *  ragged run with Apache Arrow offsets, row @c i spanning @c offsets[i] up to @c offsets[i+1].
 */
static napi_value usearch_js_index_multiply(napi_env env, napi_callback_info info) {
    napi_value arguments[11], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 11, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_multiply_t multiply = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &multiply.executor, &multiply.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_rows_from_value(env, arguments + 2, "left", &multiply.count, &multiply.left) ||
        !usearch_js_size_from_value(env, arguments[8], "wanted", &multiply.wanted) ||
        !usearch_js_size_from_value(env, arguments[9], "expansionFactor", &multiply.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[10], call, &multiply.stats) ||
        !usearch_js_arena_from_value(env, arguments[7], &call->call, &multiply.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[7], arguments[2], NULL, arguments[10]};
    call->call.finish = usearch_js_finish_multiply;
    call->count = multiply.count;
    multiply.outcome = &call->call.outcome;
    multiply.products = &call->products;
    multiply.written = &call->written;
    multiply.required = &call->required;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_multiply(&multiply);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** Hands every donor the library left in its slot back to its wrapper; a nulled one is consumed. */
static void usearch_js_conclude_absorb(usearch_js_call_t* call) {
    usearch_dense_index_t const* const handles = (usearch_dense_index_t const*)call->owned[0];
    usearch_js_index_t** const donors = (usearch_js_index_t**)call->owned[1];
    usearch_js_dense_call_t const* const dense = (usearch_js_dense_call_t const*)call;
    for (size_t donor = 0; donor != dense->count; ++donor)
        if (handles[donor]) donors[donor]->handle = handles[donor];
}

/**
 *  @brief @c absorb(executor, completion, donors) - absorbs every donor, each consumed as it nulls.
 *
 *  Every donor's handle moves into the call at submission, so until it is final a donor refuses
 *  every verb and no second absorb can name it; @c usearch_js_conclude_absorb hands back the rest.
 */
static napi_value usearch_js_index_absorb(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_absorb_t absorb = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &absorb.executor, &absorb.completion) : NULL;
    if (!call) return NULL;
    bool is_array = false;
    uint32_t length = 0;
    if (napi_is_array(env, arguments[2], &is_array) != napi_ok || !is_array ||
        napi_get_array_length(env, arguments[2], &length) != napi_ok) {
        usearch_js_call_abandon(&call->call, &local.call);
        usearch_js_throw_field(env, "donors", "must be an array of indexes");
        return NULL;
    }
    usearch_dense_index_t* const handles = (usearch_dense_index_t*)calloc(length ? length : 1, sizeof(void*));
    usearch_js_index_t** const donors = (usearch_js_index_t**)calloc(length ? length : 1, sizeof(void*));
    call->call.owned[0] = handles;
    call->call.owned[1] = donors;
    if (!handles || !donors) {
        usearch_js_call_abandon(&call->call, &local.call);
        usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not gather the donors");
        return NULL;
    }
    usearch_js_addon_t* addon = NULL;
    napi_get_instance_data(env, (void**)&addon);
    for (uint32_t position = 0; position != length; ++position) {
        napi_value entry = NULL;
        usearch_js_index_t* const donor = addon && napi_get_element(env, arguments[2], position, &entry) == napi_ok
                                              ? (usearch_js_index_t*)usearch_js_unwrap_instance(
                                                    env, entry, addon->index_class, "donors", "must hold indexes")
                                              : NULL;
        if (!donor || !donor->handle) {
            call->count = position;
            usearch_js_conclude_absorb(&call->call);
            usearch_js_call_abandon(&call->call, &local.call);
            if (donor) usearch_js_throw(env, usearch_status_uninitialized_k, "A donor was freed or absorbed already");
            return NULL;
        }
        donors[position] = donor;
        handles[position] = donor->handle;
        donor->handle = NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], NULL, arguments[2]};
    call->call.finish = usearch_js_finish_nothing;
    call->call.conclude = usearch_js_conclude_absorb;
    call->count = length;
    absorb.outcome = &call->call.outcome;
    absorb.count = length;
    absorb.donors = handles;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_absorb(&absorb);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

#pragma endregion Index Verbs

#pragma region Serialization

/** @c save(executor, completion, path) - writes the checkpoint the path's extension picks. */
static napi_value usearch_js_index_save(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_save_t save = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &save.executor, &save.completion) : NULL;
    if (!call) return NULL;
    call->call.owned[0] = usearch_js_string_from_value(env, arguments[2], "path", &save.path);
    if (!call->call.owned[0]) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_nothing;
    save.outcome = &call->call.outcome;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_save(&save);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** Views the monolith inside its arena. */
static napi_value usearch_js_finish_save_buffer(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_bytes_view_t const* const monolith = &((usearch_js_dense_call_t const*)call)->monolith;
    return usearch_js_arena_view(env, slots[usearch_js_slot_arena_k], napi_uint8_array, (void*)monolith->data,
                                 monolith->length, monolith->length);
}

/** @c saveBuffer(executor, completion, arena) → one monolith's bytes, viewed inside @p arena. */
static napi_value usearch_js_index_save_buffer(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_dense_call_t local;
    usearch_dense_save_buffer_t save = {.index = index ? index->handle : NULL};
    usearch_js_dense_call_t* const call =
        index ? usearch_js_dense_open(env, arguments, &local, &save.executor, &save.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_arena_from_value(env, arguments[2], &call->call, &save.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[2]};
    call->call.finish = usearch_js_finish_save_buffer;
    save.outcome = &call->call.outcome;
    save.monolith = &call->monolith;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_dense_save_buffer(&save);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

#pragma endregion Serialization

#pragma region Attention

/** What one @c attend call hands its value lookup, which runs on the calling thread. */
typedef struct usearch_js_value_rows_t {

    /** The environment the lookup re-enters. */
    napi_env env;

    /** The call a thrown exception latches into. */
    usearch_js_call_t* call;

    /** The caller's function, reachable for as long as the verb runs. */
    napi_value row_of;

    /** The value rows, borrowed for the whole call. */
    usearch_rows_view_t values;

    /** Rows @c values holds, which bounds every row @c row_of answers. */
    size_t rows;
} usearch_js_value_rows_t;

/** Takes the pending exception as the call's failure, unless an earlier one was latched already. */
static void usearch_js_value_rows_latch(usearch_js_value_rows_t* lookup) {
    napi_value error = NULL;
    bool pending = false;
    if (napi_is_exception_pending(lookup->env, &pending) != napi_ok || !pending ||
        napi_get_and_clear_last_exception(lookup->env, &error) != napi_ok)
        return;
    if (!lookup->call->failure) napi_create_reference(lookup->env, error, 1, &lookup->call->failure);
}

/**
 *  @brief The @c usearch_value_lookup_t of every attention, re-entering JavaScript once per scored
 *      key and pointing it at the value row @c row_of answers.
 *
 *  A host function, so it serves RAM handles alone: a VRAM walk resolves rows through a device
 *  function, and @c attend refuses a VRAM arena with @c usearch_status_device_k rather than hand
 *  this one to it.
 *
 *  A key answered by anything but a row the values hold, or asked for in another kind or width than
 *  the values are stored in, resolves to @c NULL, which refuses the call. Once @c row_of throws,
 *  every later key resolves to @c NULL without calling it, and the call throws that exception
 *  rather than the refusal.
 */
static void usearch_js_value_rows(void* state, usearch_key_t const* keys, size_t count, usearch_scalar_t scalar,
                                  size_t columns, void const** rows) {
    usearch_js_value_rows_t* const lookup = (usearch_js_value_rows_t*)state;
    napi_env const env = lookup->env;
    napi_handle_scope scope = NULL;
    napi_value buffer = NULL, receiver = NULL;
    void* copied = NULL;
    bool const readable = scalar == lookup->values.scalar && columns == lookup->values.columns;
    bool const opened = readable && !lookup->call->failure && napi_open_handle_scope(env, &scope) == napi_ok;
    // The keys are the library's only for this batch, so JavaScript reads a copy it may keep.
    if (opened && napi_get_undefined(env, &receiver) == napi_ok &&
        (buffer = usearch_js_buffer(env, count * sizeof(usearch_key_t), &copied)) != NULL)
        memcpy(copied, keys, count * sizeof(usearch_key_t));
    else if (opened) usearch_js_value_rows_latch(lookup);

    for (size_t position = 0; position != count; ++position) {
        napi_value key = NULL, answer = NULL;
        double row = -1;
        rows[position] = NULL;
        if (!buffer || lookup->call->failure ||
            napi_create_typedarray(env, napi_uint8_array, sizeof(usearch_key_t), buffer,
                                   position * sizeof(usearch_key_t), &key) != napi_ok)
            continue;
        if (napi_call_function(env, receiver, lookup->row_of, 1, &key, &answer) != napi_ok) {
            usearch_js_value_rows_latch(lookup);
            continue;
        }
        if (napi_get_value_double(env, answer, &row) == napi_ok && row >= 0 && row < (double)lookup->rows &&
            row == (double)(size_t)row)
            rows[position] = (char const*)lookup->values.data + (size_t)row * lookup->values.stride;
    }
    if (opened) napi_close_handle_scope(env, scope);
}

/** A softmax carry: a @c Float32Array of two floats per query row, or nothing at all. */
static bool usearch_js_carry_from_value(napi_env env, napi_value value, size_t count,
                                        usearch_dense_softmax_state_t** carry) {
    // The element type is checked rather than declared: a Float64Array of one element per query
    // spans the same bytes as a Float32Array of two, and would be read as pairs it does not hold.
    napi_typedarray_type type = napi_int8_array;
    size_t elements = 0;
    void* data = NULL;
    bool is_typedarray = false;
    *carry = NULL;
    if (usearch_js_value_absent(env, value)) return true;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &type, &elements, &data, NULL, NULL) != napi_ok ||
        type != napi_float32_array)
        return usearch_js_throw_field(env, "carry", "must be a Float32Array");
    if (elements != count * 2)
        return usearch_js_throw(env, usearch_status_shape_mismatch_k,
                                "The carry must hold a maximum and a partition per query");
    *carry = (usearch_dense_softmax_state_t*)data;
    return true;
}

static napi_value usearch_js_finish_attend(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_js_dense_call_t const* const dense = (usearch_js_dense_call_t const*)call;
    if (!usearch_js_stats_report(env, slots[usearch_js_slot_stats_k], &dense->stats)) return NULL;
    return usearch_js_rows_object(env, slots[usearch_js_slot_arena_k], dense, dense->count);
}

/**
 *  @brief @c attend(data, rows, columns, stride, scalar, values..5, rowOf, arena, scalar, carry,
 *      headCount, expansion, scoreScale, stats) → @c {data, rows, columns, stride}, one attended
 *      row per query in @p arena.
 *
 *  The @p carry is a @c Float32Array of two floats per query row - the largest logit and the
 *  partition taken against it - lent by the caller, since folding a second branch over the same
 *  queries needs both, which is why they share a row rather than travelling apart. Every lookup
 *  re-enters JavaScript, which no executor thread may, so the verb takes no executor and no
 *  completion and runs on the calling thread.
 */
static napi_value usearch_js_index_attend(napi_env env, napi_callback_info info) {
    napi_value arguments[18], self = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 18, arguments, &self);
    if (!index) return NULL;
    usearch_js_dense_call_t call;
    memset(&call, 0, sizeof(call));
    usearch_js_value_rows_t lookup = {.env = env, .call = &call.call, .row_of = arguments[10]};
    usearch_dense_attend_t attend = {.index = index->handle};
    napi_valuetype row_of_type = napi_undefined;
    double score_scale = 0;
    if (!usearch_js_rows_from_value(env, arguments, "queries", &attend.count, &attend.queries) ||
        !usearch_js_rows_from_value(env, arguments + 5, "values", &lookup.rows, &lookup.values) ||
        (!usearch_js_value_absent(env, arguments[12]) &&
         !usearch_js_scalar_from_value(env, arguments[12], "scalar", &attend.outputs_scalar)) ||
        !usearch_js_size_from_value(env, arguments[14], "headCount", &attend.head_count) ||
        !usearch_js_size_from_value(env, arguments[15], "expansionFactor", &attend.expansion_factor) ||
        !usearch_js_double_from_value(env, arguments[16], "scoreScale", &score_scale) ||
        !usearch_js_stats_from_value(env, arguments[17], &call, &attend.stats) ||
        ((napi_typeof(env, arguments[10], &row_of_type) != napi_ok || row_of_type != napi_function) &&
         !usearch_js_throw_field(env, "rowOf", "must be a function")) ||
        !usearch_js_carry_from_value(env, arguments[13], attend.count, &attend.carry) ||
        !usearch_js_arena_from_value(env, arguments[11], &call.call, &attend.arena))
        return NULL;
    // A VRAM arena means a device walk, which would call this host lookup on the device.
    if (call.call.arena->residency == usearch_residency_vram_k) {
        usearch_js_throw(env, usearch_status_device_k, "A VRAM walk needs a device-callable value lookup");
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, NULL, arguments[11], arguments[0], NULL, arguments[17]};
    call.call.finish = usearch_js_finish_attend;
    call.count = attend.count;
    call.scalar = attend.outputs_scalar ? attend.outputs_scalar : attend.queries.scalar;
    attend.score_scale = (float)score_scale;
    attend.outcome = &call.call.outcome;
    attend.outputs = &call.vectors;
    attend.value_lookup = usearch_js_value_rows;
    attend.value_lookup_state = &lookup;
    usearch_dense_attend(&attend);
    return usearch_js_call_answer(env, &call.call, &call.call, slots, NULL);
}

#pragma endregion Attention

#pragma region Cursors and Streams

/** A row scan or a resumable search, and the index and executor every pull names again. */
typedef struct usearch_js_walk_t {

    /** The engine cursor or stream, @c NULL once released. */
    void* handle;

    /** A strong reference to the @c CompiledIndex, so the index outlives the walk it borrows. */
    napi_ref index;

    /** A strong reference to the @c Executor every pull runs on; @c NULL for the calling thread. */
    napi_ref executor;
} usearch_js_walk_t;

/** Releases the engine walk through @p release, then the references keeping its index and executor
 *  alive. */
static void usearch_js_walk_release(napi_env env, usearch_js_walk_t* walk, void (*release)(void*)) {
    release(walk->handle);
    walk->handle = NULL;
    if (walk->index) napi_delete_reference(env, walk->index);
    if (walk->executor) napi_delete_reference(env, walk->executor);
    walk->index = NULL;
    walk->executor = NULL;
}

static void usearch_js_cursor_free(void* handle) { usearch_dense_cursor_free((usearch_dense_cursor_t)handle); }

static void usearch_js_search_stream_free(void* handle) {
    usearch_dense_search_stream_free((usearch_dense_search_stream_t)handle);
}

static void usearch_js_cursor_finalize(napi_env env, void* data, void* hint) {
    (void)hint;
    usearch_js_walk_release(env, (usearch_js_walk_t*)data, usearch_js_cursor_free);
    free(data);
}

static void usearch_js_search_stream_finalize(napi_env env, void* data, void* hint) {
    (void)hint;
    usearch_js_walk_release(env, (usearch_js_walk_t*)data, usearch_js_search_stream_free);
    free(data);
}

/** Reads the walk behind the receiver, and the index handle and executor its pull names. */
static usearch_js_walk_t* usearch_js_walk_self(napi_env env, napi_callback_info info, size_t wanted,
                                               napi_value* arguments, usearch_dense_index_t* index,
                                               usearch_executor_t* executor) {
    size_t arguments_count = wanted;
    napi_value self = NULL, index_value = NULL, executor_value = NULL;
    usearch_js_walk_t* walk = NULL;
    usearch_js_index_t* wrapper = NULL;
    if (napi_get_cb_info(env, info, &arguments_count, arguments, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&walk) != napi_ok || !walk) {
        usearch_js_throw(env, usearch_status_internal_k, "The walk is not attached");
        return NULL;
    }
    for (size_t position = arguments_count; position < wanted; ++position)
        arguments[position] = usearch_js_undefined(env);
    if (!walk->handle) {
        usearch_js_throw(env, usearch_status_uninitialized_k, "The walk was already released");
        return NULL;
    }
    if (napi_get_reference_value(env, walk->index, &index_value) != napi_ok ||
        napi_unwrap(env, index_value, (void**)&wrapper) != napi_ok || !wrapper || !wrapper->handle) {
        usearch_js_throw(env, usearch_status_uninitialized_k, "The index the walk borrows was freed");
        return NULL;
    }
    if (walk->executor && napi_get_reference_value(env, walk->executor, &executor_value) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "The executor the walk pulls on is missing");
        return NULL;
    }
    if (!usearch_js_executor_from_value(env, executor_value, executor)) return NULL;
    *index = wrapper->handle;
    return walk;
}

/**
 *  @brief Wraps an engine walk into an object exposing @p next, @c free and @c [Symbol.dispose],
 *      holding the index the walk borrows and the executor every pull runs on.
 *  @return The object, or @c NULL with an exception pending and the walk released.
 */
static napi_value usearch_js_walk_object(napi_env env, void* handle, napi_value index, napi_value executor,
                                         napi_callback next, napi_callback free_method, napi_finalize finalize,
                                         void (*release)(void*)) {
    napi_property_descriptor const properties[] = {
        {"next", NULL, next, NULL, NULL, NULL, napi_default_method, NULL},
        {"free", NULL, free_method, NULL, NULL, NULL, napi_default_method, NULL},
    };
    napi_value object = NULL;
    usearch_js_walk_t* const walk = (usearch_js_walk_t*)calloc(1, sizeof(usearch_js_walk_t));
    if (!walk) {
        release(handle);
        usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not allocate the walk");
        return NULL;
    }
    walk->handle = handle;
    bool const held = napi_create_reference(env, index, 1, &walk->index) == napi_ok &&
                      (usearch_js_value_absent(env, executor) ||
                       napi_create_reference(env, executor, 1, &walk->executor) == napi_ok);
    if (!held || napi_create_object(env, &object) != napi_ok ||
        napi_wrap(env, object, walk, finalize, NULL, NULL) != napi_ok) {
        usearch_js_walk_release(env, walk, release);
        free(walk);
        usearch_js_throw(env, usearch_status_internal_k, "Could not attach the walk");
        return NULL;
    }
    // Wrapped already, so a failure from here on is the finalizer's to clean up.
    if (napi_define_properties(env, object, usearch_js_table_size_(properties), properties) != napi_ok ||
        !usearch_js_disposable(env, object, "free")) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not attach the walk");
        return NULL;
    }
    return object;
}

/** Views the keys a cursor pull landed as @c {keys, rows}. */
static napi_value usearch_js_finish_cursor_keys(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_js_dense_call_t const* const dense = (usearch_js_dense_call_t const*)call;
    size_t const bytes = dense->written * sizeof(usearch_key_t);
    napi_value const keys = usearch_js_arena_view(env, slots[usearch_js_slot_arena_k], napi_uint8_array,
                                                  (void*)dense->keys.data, bytes, bytes);
    if (!keys) return NULL;
    static char const* const names[] = {"keys", "rows"};
    napi_value const values[] = {keys, usearch_js_number(env, dense->written)};
    return usearch_js_object(env, usearch_js_table_size_(names), names, values);
}

/** Views the keys and links a cursor pull landed as @c {keys, rows, links: {offsets, keys}}. */
static napi_value usearch_js_finish_cursor_links(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_js_dense_call_t const* const dense = (usearch_js_dense_call_t const*)call;
    napi_value const arena = slots[usearch_js_slot_arena_k];
    napi_value const batch = usearch_js_finish_cursor_keys(env, call, slots);
    size_t const linked = dense->links.offsets ? dense->links.offsets[dense->written] : 0;
    napi_value const offsets = batch ? usearch_js_arena_view(env, arena, napi_biguint64_array,
                                                             (void*)dense->links.offsets,
                                                             (dense->written + 1) * sizeof(size_t), dense->written + 1)
                                     : NULL;
    napi_value const keys = offsets
                                ? usearch_js_arena_view(env, arena, napi_uint8_array, (void*)dense->links.keys,
                                                        linked * sizeof(usearch_key_t), linked * sizeof(usearch_key_t))
                                : NULL;
    if (!keys) return NULL;
    static char const* const names[] = {"offsets", "keys"};
    napi_value const values[] = {offsets, keys};
    napi_value const links = usearch_js_object(env, usearch_js_table_size_(names), names, values);
    if (!links || napi_set_named_property(env, batch, "links", links) != napi_ok) return NULL;
    return batch;
}

/** @c next(wanted, arena, links) → @c {keys, rows, links}, the next batch viewed inside @p arena;
 *  @c links is present when the truthy flag asked for it, and zero rows mean the walk ended. */
static napi_value usearch_js_cursor_next(napi_env env, napi_callback_info info) {
    napi_value arguments[3];
    usearch_dense_cursor_next_t next = {0};
    usearch_js_walk_t* const walk = usearch_js_walk_self(env, info, 3, arguments, &next.index, &next.executor);
    if (!walk) return NULL;
    usearch_js_dense_call_t call;
    memset(&call, 0, sizeof(call));
    bool exports_links = false;
    if (!usearch_js_value_absent(env, arguments[2]) &&
        napi_get_value_bool(env, arguments[2], &exports_links) != napi_ok) {
        usearch_js_throw_field(env, "links", "must be a boolean");
        return NULL;
    }
    if (!usearch_js_size_from_value(env, arguments[0], "wanted", &next.wanted) ||
        !usearch_js_arena_from_value(env, arguments[1], &call.call, &next.arena))
        return NULL;
    napi_value const slots[usearch_js_slots_k] = {NULL, NULL, arguments[1]};
    call.call.finish = exports_links ? usearch_js_finish_cursor_links : usearch_js_finish_cursor_keys;
    next.outcome = &call.call.outcome;
    next.cursor = (usearch_dense_cursor_t)walk->handle;
    next.keys = &call.keys;
    next.links = exports_links ? &call.links : NULL;
    next.written = &call.written;
    usearch_dense_cursor_next(&next);
    return usearch_js_call_answer(env, &call.call, &call.call, slots, NULL);
}

/** @c next(wanted, arena) → @c {keys, scores, counts, columns}, one ranked row viewed inside
 *  @p arena; its count is zero once the walk is drained. */
static napi_value usearch_js_search_stream_next(napi_env env, napi_callback_info info) {
    napi_value arguments[2];
    usearch_dense_search_stream_next_t next = {0};
    usearch_js_walk_t* const walk = usearch_js_walk_self(env, info, 2, arguments, &next.index, &next.executor);
    if (!walk) return NULL;
    usearch_js_dense_call_t call;
    memset(&call, 0, sizeof(call));
    if (!usearch_js_size_from_value(env, arguments[0], "wanted", &next.wanted) ||
        !usearch_js_arena_from_value(env, arguments[1], &call.call, &next.arena))
        return NULL;
    napi_value const slots[usearch_js_slots_k] = {NULL, NULL, arguments[1]};
    call.call.finish = usearch_js_finish_matches;
    call.count = 1;
    next.outcome = &call.call.outcome;
    next.stream = (usearch_dense_search_stream_t)walk->handle;
    next.matches = &call.matches;
    usearch_dense_search_stream_next(&next);
    return usearch_js_call_answer(env, &call.call, &call.call, slots, NULL);
}

/** @c free() - releases the scan; a second call is a no-op. */
static napi_value usearch_js_cursor_release(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    usearch_js_walk_t* walk = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&walk) != napi_ok || !walk) {
        usearch_js_throw(env, usearch_status_internal_k, "The cursor is not attached");
        return NULL;
    }
    usearch_js_walk_release(env, walk, usearch_js_cursor_free);
    return usearch_js_undefined(env);
}

/** @c free() - releases the walk state; a second call is a no-op. */
static napi_value usearch_js_search_stream_release(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    usearch_js_walk_t* walk = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&walk) != napi_ok || !walk) {
        usearch_js_throw(env, usearch_status_internal_k, "The search stream is not attached");
        return NULL;
    }
    usearch_js_walk_release(env, walk, usearch_js_search_stream_free);
    return usearch_js_undefined(env);
}

/** @c cursor(executor, starts) → a walk over the entries published so far, pulling on @p executor;
 *  @p starts is absent for every row as stored, else the keys to walk out from. */
static napi_value usearch_js_index_cursor(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_executor_t executor = NULL;
    usearch_dense_cursor_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_cursor_init_t init = {.cursor = &handle, .outcome = &outcome};
    usearch_key_t* gathered = NULL;
    if (!index || !usearch_js_executor_from_value(env, arguments[0], &executor)) return NULL;
    if (!usearch_js_value_absent(env, arguments[1]) &&
        !usearch_js_keys_from_value(env, arguments[1], "starts", &init.starts, &init.starts_count, &gathered))
        return NULL;

    init.index = index->handle;
    usearch_dense_cursor_init(&init);
    free(gathered);
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    return usearch_js_walk_object(env, handle, self, arguments[0], usearch_js_cursor_next, usearch_js_cursor_release,
                                  usearch_js_cursor_finalize, usearch_js_cursor_free);
}

/** @c searchStream(executor, data, rows, columns, stride, scalar, expansion) → a resumable
 *  walk over one row. */
static napi_value usearch_js_index_search_stream(napi_env env, napi_callback_info info) {
    napi_value arguments[7], self = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 7, arguments, &self);
    usearch_executor_t executor = NULL;
    usearch_dense_search_stream_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_search_stream_init_t init = {.stream = &handle, .outcome = &outcome};
    size_t rows = 0;
    if (!index || !usearch_js_executor_from_value(env, arguments[0], &executor) ||
        !usearch_js_rows_from_value(env, arguments + 1, "query", &rows, &init.query) ||
        !usearch_js_size_from_value(env, arguments[6], "expansionFactor", &init.expansion_factor))
        return NULL;
    // The query view carries no count, so a second row would pass unseen.
    if (rows != 1) {
        usearch_js_throw(env, usearch_status_shape_mismatch_k, "A search stream takes exactly one query row");
        return NULL;
    }
    init.index = index->handle;
    usearch_dense_search_stream_init(&init);
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    return usearch_js_walk_object(env, handle, self, arguments[0], usearch_js_search_stream_next,
                                  usearch_js_search_stream_release, usearch_js_search_stream_finalize,
                                  usearch_js_search_stream_free);
}

#pragma endregion Cursors and Streams

#pragma region Registration

napi_value usearch_js_index_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"free", NULL, usearch_js_index_release, NULL, NULL, NULL, napi_default_method, NULL},
        {"describe", NULL, usearch_js_index_describe, NULL, NULL, NULL, napi_default_method, NULL},
        {"health", NULL, usearch_js_index_health, NULL, NULL, NULL, napi_default_method, NULL},
        {"reserve", NULL, usearch_js_index_reserve, NULL, NULL, NULL, napi_default_method, NULL},
        {"clear", NULL, usearch_js_index_clear, NULL, NULL, NULL, napi_default_method, NULL},
        {"count", NULL, usearch_js_index_count, NULL, NULL, NULL, napi_default_method, NULL},
        {"insert", NULL, usearch_js_index_insert, NULL, NULL, NULL, napi_default_method, NULL},
        {"refine", NULL, usearch_js_index_refine, NULL, NULL, NULL, napi_default_method, NULL},
        {"search", NULL, usearch_js_index_search, NULL, NULL, NULL, napi_default_method, NULL},
        {"maxsim", NULL, usearch_js_index_maxsim, NULL, NULL, NULL, napi_default_method, NULL},
        {"get", NULL, usearch_js_index_get, NULL, NULL, NULL, napi_default_method, NULL},
        {"multiply", NULL, usearch_js_index_multiply, NULL, NULL, NULL, napi_default_method, NULL},
        {"absorb", NULL, usearch_js_index_absorb, NULL, NULL, NULL, napi_default_method, NULL},
        {"save", NULL, usearch_js_index_save, NULL, NULL, NULL, napi_default_method, NULL},
        {"saveBuffer", NULL, usearch_js_index_save_buffer, NULL, NULL, NULL, napi_default_method, NULL},
        {"attend", NULL, usearch_js_index_attend, NULL, NULL, NULL, napi_default_method, NULL},
        {"cursor", NULL, usearch_js_index_cursor, NULL, NULL, NULL, napi_default_method, NULL},
        {"searchStream", NULL, usearch_js_index_search_stream, NULL, NULL, NULL, napi_default_method, NULL},
        {"initRam", NULL, usearch_js_index_init_ram, NULL, NULL, NULL, napi_static, NULL},
        {"initVram", NULL, usearch_js_index_init_vram, NULL, NULL, NULL, napi_static, NULL},
        {"loadRam", NULL, usearch_js_index_load_ram, NULL, NULL, NULL, napi_static, NULL},
        {"loadVram", NULL, usearch_js_index_load_vram, NULL, NULL, NULL, napi_static, NULL},
        {"loadRamBuffer", NULL, usearch_js_index_load_ram_buffer, NULL, NULL, NULL, napi_static, NULL},
        {"loadVramBuffer", NULL, usearch_js_index_load_vram_buffer, NULL, NULL, NULL, napi_static, NULL},
        {"viewRam", NULL, usearch_js_index_view_ram, NULL, NULL, NULL, napi_static, NULL},
        {"viewNvme", NULL, usearch_js_index_view_nvme, NULL, NULL, NULL, napi_static, NULL},
        {"metadataRead", NULL, usearch_js_index_metadata_read_path, NULL, NULL, NULL, napi_static, NULL},
        {"metadataReadBuffer", NULL, usearch_js_index_metadata_read_buffer, NULL, NULL, NULL, napi_static, NULL},
    };
    napi_value index_class = NULL;
    if (napi_define_class(env, "CompiledIndex", NAPI_AUTO_LENGTH, usearch_js_index_construct, NULL,
                          usearch_js_table_size_(properties), properties, &index_class) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not define the CompiledIndex class");
        return NULL;
    }
    return index_class;
}

#pragma endregion Registration
