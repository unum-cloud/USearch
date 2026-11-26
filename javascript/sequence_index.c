/**
 *  @file javascript/sequence_index.c
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Node-API search over sequences: protein alignments, DNA overlaps and UTF-8 containments.
 *
 *  The @c CompiledSequenceIndex class and the lifecycle verbs that instantiate it, the cursor it
 *  opens, and the checkpoint metadata readers that need no index at all. Sequences cross as one
 *  Apache Arrow tape of bytes and offsets, viewed in place.
 *
 *  Every instance verb takes the executor and the completion first, so one native function serves
 *  the calling thread, a blocking receiver and a deferred one alike.
 */
#include <stdlib.h> // `calloc`, `free`
#include <string.h> // `memset`

#include "lib.h"

#pragma region Calls

/** Wraps the metric, connectivity, classes and sketch bytes a checkpoint froze. */
static napi_value usearch_js_sequence_metadata_object(napi_env env, usearch_sequence_metadata_t const* metadata) {
    static char const* const names[] = {"metric", "connectivity", "classes", "sketchBytes"};
    napi_value const values[] = {usearch_js_static_string(env, usearch_sequence_metric_name(metadata->metric)),
                                 usearch_js_number(env, metadata->connectivity),
                                 usearch_js_number(env, metadata->classes),
                                 usearch_js_number(env, metadata->sketch_bytes)};
    return usearch_js_object(env, usearch_js_table_size_(names), names, values);
}

static napi_value usearch_js_sequence_finish_describe(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_sequence_index_description_t const* const description =
        &((usearch_js_index_call_t const*)call)->description.sequence;
    napi_value const metadata = usearch_js_sequence_metadata_object(env, &description->metadata);
    (void)slots;
    if (!metadata) return NULL;
    static char const* const names[] = {"metadata", "size",     "capacity",    "memoryUsage",
                                        "cpuClass", "gpuClass", "shardsCount", "loadPath"};
    napi_value const values[] = {metadata,
                                 usearch_js_number(env, description->size),
                                 usearch_js_number(env, description->capacity),
                                 usearch_js_number(env, description->memory_usage),
                                 usearch_js_static_string(env, usearch_cpu_class_name(description->cpu_class)),
                                 usearch_js_static_string(env, usearch_gpu_class_name(description->gpu_class)),
                                 usearch_js_number(env, description->shards_count),
                                 usearch_js_static_string(env, usearch_load_path_name(description->load_path))};
    return usearch_js_object(env, usearch_js_table_size_(names), names, values);
}

static napi_value usearch_js_sequence_finish_count(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_sequence_key_stats_t const* const stats = &((usearch_js_index_call_t const*)call)->key_stats.sequence;
    (void)slots;
    static char const* const names[] = {"entries", "totalBytes", "largestBytes"};
    napi_value const values[] = {usearch_js_number(env, stats->entries), usearch_js_number(env, stats->total_bytes),
                                 usearch_js_number(env, stats->largest_bytes)};
    return usearch_js_object(env, usearch_js_table_size_(names), names, values);
}

#pragma endregion Calls

#pragma region Index Construction

static void usearch_js_sequence_finalize(napi_env env, void* data, void* hint) {
    usearch_js_index_t* const index = (usearch_js_index_t*)data;
    (void)env;
    (void)hint;
    if (!index) return;
    usearch_sequence_free(index->handle);
    free(index);
}

/** @c CompiledSequenceIndex() - an empty shell that a lifecycle verb fills with its handle. */
static napi_value usearch_js_sequence_construct(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    if (!usearch_js_sealed(env, info, &self)) {
        usearch_js_throw(env, usearch_status_invalid_argument_k, "A CompiledSequenceIndex is made by a lifecycle verb");
        return NULL;
    }
    return self;
}

/** Wraps a handle a lifecycle verb just made, or throws its outcome and frees the handle. */
static napi_value usearch_js_sequence_adopt(napi_env env, usearch_outcome_t const* outcome,
                                            usearch_sequence_index_t handle) {
    if (!usearch_js_succeeded(env, outcome)) return NULL;
    usearch_js_addon_t* addon = NULL;
    napi_value instance = NULL;
    usearch_js_index_t* const index = (usearch_js_index_t*)calloc(1, sizeof(usearch_js_index_t));
    if (napi_get_instance_data(env, (void**)&addon) == napi_ok && addon)
        instance = usearch_js_instantiate(env, addon->sequence_index_class);
    if (!instance || !index || napi_wrap(env, instance, index, usearch_js_sequence_finalize, NULL, NULL) != napi_ok) {
        free(index);
        usearch_sequence_free(handle);
        usearch_js_throw(env, usearch_status_internal_k, "Could not instantiate the CompiledSequenceIndex");
        return NULL;
    }
    index->handle = handle;
    return instance;
}

/** Reads the costs from two optional values: an @c Int8Array of 256 × 256 similarities between raw
 *  bytes and the gap cost. The array is viewed, neither copied nor kept past the call.
 *  @p costs is left @c NULL when neither is named, for the metric's own. */
static bool usearch_js_sequence_costs_from_values(napi_env env, napi_value const* arguments,
                                                  usearch_sequence_costs_t* values,
                                                  usearch_sequence_costs_t const** costs) {
    napi_typedarray_type type = napi_uint8_array;
    size_t elements = 0;
    void* data = NULL;
    bool is_typedarray = false;
    int32_t gap = 0;
    memset(values, 0, sizeof(*values));
    *costs = NULL;
    if (!usearch_js_value_absent(env, arguments[0])) {
        if (napi_is_typedarray(env, arguments[0], &is_typedarray) != napi_ok || !is_typedarray ||
            napi_get_typedarray_info(env, arguments[0], &type, &elements, &data, NULL, NULL) != napi_ok ||
            type != napi_int8_array || elements != 256 * 256)
            return usearch_js_throw_field(env, "similarities", "must be an Int8Array of 256 by 256 scores");
        values->similarities = (int8_t const*)data;
        *costs = values;
    }
    if (!usearch_js_value_absent(env, arguments[1])) {
        if (napi_get_value_int32(env, arguments[1], &gap) != napi_ok || gap < INT8_MIN || gap > INT8_MAX)
            return usearch_js_throw_field(env, "gap", "must fit a signed byte");
        values->gap = (int8_t)gap;
        *costs = values;
    }
    return true;
}

/** <tt>initRam(metric, similarities, gap, connectivity, sketchBytes, valuesPerKey, memoryDomains,
 *  cpuClassCeiling)</tt> and <tt>initVram(..., devices, gpuClassCeiling)</tt> → an empty, mutable index.
 *  The costs are read during the call alone. */
static napi_value usearch_js_sequence_init(napi_env env, napi_callback_info info, usearch_residency_t residency) {
    napi_value arguments[8];
    size_t connectivity = 0, sketch_bytes = 0, values_per_key = 0;
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;
    usearch_sequence_costs_t values;
    usearch_sequence_costs_t const* costs;
    usearch_cpu_class_t cpu_class_ceiling = usearch_cpu_class_unknown_k;
    usearch_gpu_class_t gpu_class_ceiling = usearch_gpu_class_unknown_k;
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    bool const vram = residency == usearch_residency_vram_k;
    if (!usearch_js_static_arguments(env, info, 8, arguments) ||
        !(vram ? usearch_js_gpu_class_from_value(env, arguments[7], &gpu_class_ceiling)
               : usearch_js_cpu_class_from_value(env, arguments[7], &cpu_class_ceiling)) ||
        !usearch_js_sequence_metric_from_value(env, arguments[0], "metric", &metric) ||
        !usearch_js_sequence_costs_from_values(env, arguments + 1, &values, &costs) ||
        !usearch_js_size_from_value(env, arguments[3], "connectivity", &connectivity) ||
        !usearch_js_size_from_value(env, arguments[4], "sketchBytes", &sketch_bytes) ||
        !usearch_js_size_from_value(env, arguments[5], "valuesPerKey", &values_per_key) ||
        !usearch_js_ordinals_from_value(env, arguments[6], vram ? "devices" : "memoryDomains", &ordinals,
                                        &ordinals_count))
        return NULL;

    usearch_sequence_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    if (vram) {
        usearch_sequence_init_vram_t init = {.index = &handle,
                                             .outcome = &outcome,
                                             .metric = metric,
                                             .costs = costs,
                                             .connectivity = connectivity,
                                             .sketch_bytes = sketch_bytes,
                                             .values_per_key = values_per_key,
                                             .vram = {ordinals, ordinals_count, 0, gpu_class_ceiling}};
        usearch_sequence_init_vram(&init);
    }
    else {
        usearch_sequence_init_ram_t init = {.index = &handle,
                                            .outcome = &outcome,
                                            .metric = metric,
                                            .costs = costs,
                                            .connectivity = connectivity,
                                            .sketch_bytes = sketch_bytes,
                                            .values_per_key = values_per_key,
                                            .ram = {ordinals, ordinals_count, 0, cpu_class_ceiling}};
        usearch_sequence_init_ram(&init);
    }
    free(ordinals);
    return usearch_js_sequence_adopt(env, &outcome, handle);
}

static napi_value usearch_js_sequence_init_ram(napi_env env, napi_callback_info info) {
    return usearch_js_sequence_init(env, info, usearch_residency_ram_k);
}

static napi_value usearch_js_sequence_init_vram(napi_env env, napi_callback_info info) {
    return usearch_js_sequence_init(env, info, usearch_residency_vram_k);
}

/** <tt>loadRam(path, executor, memoryDomains, valuesPerKey, cpuClassCeiling)</tt> and
 *  <tt>loadVram(path, executor, devices, valuesPerKey, gpuClassCeiling)</tt> → a new, mutable index;
 *  @c Buffer twins take any @c TypedArray. */
static napi_value usearch_js_sequence_load(napi_env env, napi_callback_info info, usearch_residency_t residency,
                                           usearch_js_source_t source) {
    napi_value arguments[5];
    if (!usearch_js_static_arguments(env, info, 5, arguments)) return NULL;

    char* path = NULL;
    void* buffer = NULL;
    usearch_bytes_view_t bytes = {NULL, 0};
    usearch_executor_t executor = NULL;
    size_t values_per_key = 0;
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    usearch_cpu_class_t cpu_class_ceiling = usearch_cpu_class_unknown_k;
    usearch_gpu_class_t gpu_class_ceiling = usearch_gpu_class_unknown_k;
    bool const vram = residency == usearch_residency_vram_k;
    if (source == usearch_js_source_path_k) path = usearch_js_string_from_value(env, arguments[0], "path", &bytes);
    else if (usearch_js_bytes_from_value(env, arguments[0], "buffer", &buffer, &bytes.length))
        bytes.data = (uint8_t const*)buffer;
    bool const read = source == usearch_js_source_path_k ? path != NULL : bytes.data != NULL || !bytes.length;
    if (!read || !usearch_js_executor_from_value(env, arguments[1], &executor) ||
        !usearch_js_size_from_value(env, arguments[3], "valuesPerKey", &values_per_key) ||
        !(vram ? usearch_js_gpu_class_from_value(env, arguments[4], &gpu_class_ceiling)
               : usearch_js_cpu_class_from_value(env, arguments[4], &cpu_class_ceiling)) ||
        !usearch_js_ordinals_from_value(env, arguments[2], vram ? "devices" : "memoryDomains", &ordinals,
                                        &ordinals_count)) {
        free(path);
        return NULL;
    }

    usearch_sequence_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_ram_options_t const ram = {ordinals, ordinals_count, 0, cpu_class_ceiling};
    usearch_vram_options_t const vram_options = {ordinals, ordinals_count, 0, gpu_class_ceiling};
    if (vram && source == usearch_js_source_path_k) {
        usearch_sequence_load_vram_t load = {.index = &handle,
                                             .executor = executor,
                                             .outcome = &outcome,
                                             .path = bytes,
                                             .vram = vram_options,
                                             .values_per_key = values_per_key};
        usearch_sequence_load_vram(&load);
    }
    else if (vram) {
        usearch_sequence_load_vram_buffer_t load = {.index = &handle,
                                                    .executor = executor,
                                                    .outcome = &outcome,
                                                    .buffer = bytes,
                                                    .vram = vram_options,
                                                    .values_per_key = values_per_key};
        usearch_sequence_load_vram_buffer(&load);
    }
    else if (source == usearch_js_source_path_k) {
        usearch_sequence_load_ram_t load = {.index = &handle,
                                            .executor = executor,
                                            .outcome = &outcome,
                                            .path = bytes,
                                            .ram = ram,
                                            .values_per_key = values_per_key};
        usearch_sequence_load_ram(&load);
    }
    else {
        usearch_sequence_load_ram_buffer_t load = {.index = &handle,
                                                   .executor = executor,
                                                   .outcome = &outcome,
                                                   .buffer = bytes,
                                                   .ram = ram,
                                                   .values_per_key = values_per_key};
        usearch_sequence_load_ram_buffer(&load);
    }
    free(ordinals);
    free(path);
    return usearch_js_sequence_adopt(env, &outcome, handle);
}

static napi_value usearch_js_sequence_load_ram(napi_env env, napi_callback_info info) {
    return usearch_js_sequence_load(env, info, usearch_residency_ram_k, usearch_js_source_path_k);
}

static napi_value usearch_js_sequence_load_vram(napi_env env, napi_callback_info info) {
    return usearch_js_sequence_load(env, info, usearch_residency_vram_k, usearch_js_source_path_k);
}

static napi_value usearch_js_sequence_load_ram_buffer(napi_env env, napi_callback_info info) {
    return usearch_js_sequence_load(env, info, usearch_residency_ram_k, usearch_js_source_buffer_k);
}

static napi_value usearch_js_sequence_load_vram_buffer(napi_env env, napi_callback_info info) {
    return usearch_js_sequence_load(env, info, usearch_residency_vram_k, usearch_js_source_buffer_k);
}

/** <tt>viewRam(path, cpuClassCeiling)</tt> → a read-only index serving the
 *  checkpoint through a memory mapping. */
static napi_value usearch_js_sequence_view_ram(napi_env env, napi_callback_info info) {
    napi_value arguments[2];
    usearch_sequence_view_ram_t view_ram = {0};
    if (!usearch_js_static_arguments(env, info, 2, arguments) ||
        !usearch_js_cpu_class_from_value(env, arguments[1], &view_ram.ram.cpu_class_ceiling))
        return NULL;
    char* const path = usearch_js_string_from_value(env, arguments[0], "path", &view_ram.path);
    if (!path) return NULL;

    usearch_sequence_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    view_ram.index = &handle;
    view_ram.outcome = &outcome;
    usearch_sequence_view_ram(&view_ram);
    free(path);
    return usearch_js_sequence_adopt(env, &outcome, handle);
}

/** <tt>viewNvme(path, executor, paths, cpuClassCeiling)</tt> → a read-only
 *  index served from NVMe drives. */
static napi_value usearch_js_sequence_view_nvme(napi_env env, napi_callback_info info) {
    napi_value arguments[4];
    usearch_sequence_view_nvme_t view_nvme = {0};
    usearch_js_paths_arena_t paths = {NULL, 0};
    usearch_cpu_class_t cpu_class_ceiling = usearch_cpu_class_unknown_k;
    if (!usearch_js_static_arguments(env, info, 4, arguments) ||
        !usearch_js_executor_from_value(env, arguments[1], &view_nvme.executor) ||
        !usearch_js_cpu_class_from_value(env, arguments[3], &cpu_class_ceiling))
        return NULL;
    char* const path = usearch_js_string_from_value(env, arguments[0], "path", &view_nvme.path);
    if (!path || !usearch_js_paths_from_value(env, arguments[2], "paths", &paths)) {
        free(path);
        return NULL;
    }

    usearch_sequence_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    view_nvme.index = &handle;
    view_nvme.outcome = &outcome;
    view_nvme.nvme = (usearch_nvme_options_t) {paths.views, paths.count, cpu_class_ceiling};
    usearch_sequence_view_nvme(&view_nvme);
    usearch_js_paths_release(&paths);
    free(path);
    return usearch_js_sequence_adopt(env, &outcome, handle);
}

/** <tt>metadataRead(path, executor)</tt> and <tt>metadataReadBuffer(buffer, executor)</tt> → what a checkpoint
 *  froze. */
static napi_value usearch_js_sequence_metadata_read(napi_env env, napi_callback_info info, usearch_js_source_t source) {
    napi_value arguments[2];
    usearch_sequence_metadata_t metadata;
    memset(&metadata, 0, sizeof(metadata));
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t executor = NULL;
    if (!usearch_js_static_arguments(env, info, 2, arguments) ||
        !usearch_js_executor_from_value(env, arguments[1], &executor))
        return NULL;
    if (source == usearch_js_source_path_k) {
        usearch_sequence_metadata_read_t read = {.executor = executor, .outcome = &outcome, .metadata = &metadata};
        char* const path = usearch_js_string_from_value(env, arguments[0], "path", &read.path);
        if (!path) return NULL;
        usearch_sequence_metadata_read(&read);
        free(path);
    }
    else {
        void* buffer = NULL;
        usearch_sequence_metadata_read_buffer_t read = {.executor = executor,
                                                        .outcome = &outcome,
                                                        .metadata = &metadata};
        if (!usearch_js_bytes_from_value(env, arguments[0], "buffer", &buffer, &read.buffer.length)) return NULL;
        read.buffer.data = (uint8_t const*)buffer;
        usearch_sequence_metadata_read_buffer(&read);
    }
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    return usearch_js_sequence_metadata_object(env, &metadata);
}

static napi_value usearch_js_sequence_metadata_read_path(napi_env env, napi_callback_info info) {
    return usearch_js_sequence_metadata_read(env, info, usearch_js_source_path_k);
}

static napi_value usearch_js_sequence_metadata_read_buffer(napi_env env, napi_callback_info info) {
    return usearch_js_sequence_metadata_read(env, info, usearch_js_source_buffer_k);
}

/** @c free() - releases the handle now, waiting for every deferred call naming it; idempotent. */
static napi_value usearch_js_sequence_release(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    usearch_js_index_t* index = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&index) != napi_ok || !index) {
        usearch_js_throw(env, usearch_status_internal_k, "The CompiledSequenceIndex is not attached");
        return NULL;
    }
    usearch_sequence_free(index->handle);
    index->handle = NULL;
    return usearch_js_undefined(env);
}

#pragma endregion Index Construction

#pragma region Index Verbs

/** <tt>describe(executor, completion)</tt> → the frozen metadata and current counters, each read once. */
static napi_value usearch_js_sequence_describe(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_describe_t describe = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &describe.executor, &describe.completion) : NULL;
    if (!call) return NULL;
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->header.finish = usearch_js_sequence_finish_describe;
    describe.outcome = &call->header.outcome;
    describe.description = &call->description.sequence;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_describe(&describe);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>health(executor, completion)</tt> → how much of the graph a search reaches from the root. */
static napi_value usearch_js_sequence_health(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_health_t health = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &health.executor, &health.completion) : NULL;
    if (!call) return NULL;
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->header.finish = usearch_js_finish_health;
    health.outcome = &call->header.outcome;
    health.reachability = &call->reachability;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_health(&health);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>reserve(executor, completion, entries, payloadBytes)</tt> - claims room for a total entry count and
 *  the bytes they hold up front. */
static napi_value usearch_js_sequence_reserve(napi_env env, napi_callback_info info) {
    napi_value arguments[4], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 4, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_reserve_t reserve = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &reserve.executor, &reserve.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_size_from_value(env, arguments[2], "entries", &reserve.entries) ||
        !usearch_js_size_from_value(env, arguments[3], "payloadBytes", &reserve.payload_bytes)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->header.finish = usearch_js_finish_nothing;
    reserve.outcome = &call->header.outcome;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_reserve(&reserve);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>clear(executor, completion)</tt> - erases every entry, keeping capacity. */
static napi_value usearch_js_sequence_clear(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_clear_t clear = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &clear.executor, &clear.completion) : NULL;
    if (!call) return NULL;
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->header.finish = usearch_js_finish_nothing;
    clear.outcome = &call->header.outcome;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_clear(&clear);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>count(executor, completion, key)</tt> → @c {entries, totalBytes, largestBytes} under one key. */
static napi_value usearch_js_sequence_count(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_count_t count = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &count.executor, &count.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_key_from_value(env, arguments[2], "key", &count.key)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->header.finish = usearch_js_sequence_finish_count;
    count.outcome = &call->header.outcome;
    count.key_stats = &call->key_stats.sequence;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_count(&count);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>insert(executor, completion, keys, data, offsets, scalar, expansion, stats)</tt>. */
static napi_value usearch_js_sequence_insert(napi_env env, napi_callback_info info) {
    napi_value arguments[8], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 8, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_insert_t insert = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &insert.executor, &insert.completion) : NULL;
    if (!call) return NULL;
    size_t keys_count = 0;
    usearch_key_t** const gathered = (usearch_key_t**)&call->header.owned[0];
    if (!usearch_js_tape_from_value(env, arguments + 3, "sequences", 1, &insert.count, &insert.sequences) ||
        !usearch_js_size_from_value(env, arguments[6], "expansionFactor", &insert.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[7], &call->header, &insert.stats) ||
        !usearch_js_keys_from_value(env, arguments[2], "keys", &insert.keys.data, &keys_count, gathered)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    // The keys view holds no count, so only this side can see the two disagree.
    if (keys_count != insert.count) {
        usearch_js_call_abandon(&call->header, &local.header);
        usearch_js_throw(env, usearch_status_shape_mismatch_k, "The keys name one key per sequence");
        return NULL;
    }
    insert.keys.stride = sizeof(usearch_key_t);
    napi_value const slots[usearch_js_slots_k] = {
        self, arguments[0], NULL, arguments[3], arguments[2], arguments[7], [usearch_js_slot_focus_k] = arguments[4]};
    call->header.finish = usearch_js_finish_nothing;
    insert.outcome = &call->header.outcome;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_insert(&insert);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>refine(executor, completion, expansion, memoryLimit, stats, plan)</tt> → what the round did
 *  to the graph; a @p plan continues across calls until it is finished. */
static napi_value usearch_js_sequence_refine(napi_env env, napi_callback_info info) {
    napi_value arguments[6], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 6, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_refine_t refine = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &refine.executor, &refine.completion) : NULL;
    if (!call) return NULL;
    usearch_js_plan_t* plan = NULL;
    if (!usearch_js_size_from_value(env, arguments[2], "expansionFactor", &refine.refine.expansion_factor) ||
        !usearch_js_size_from_value(env, arguments[3], "memoryLimit", &refine.refine.memory_limit) ||
        !usearch_js_stats_from_value(env, arguments[4], &call->header, &refine.stats) ||
        !usearch_js_plan_from_value(env, arguments[5], &plan)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {
        self, arguments[0], NULL, NULL, NULL, arguments[4], [usearch_js_slot_plan_k] = arguments[5]};
    call->header.finish = usearch_js_finish_report;
    refine.plan = plan ? &plan->refine.sequence : NULL;
    refine.phase = plan ? &plan->progress : NULL;
    refine.outcome = &call->header.outcome;
    refine.report = &call->report;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_refine(&refine);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>refinePlanFinish(executor, completion, plan)</tt> → what the whole plan did to the graph,
 *  closing and freeing it whatever the call answers. */
static napi_value usearch_js_sequence_refine_plan_finish(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_refine_plan_finish_t finish = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &finish.executor, &finish.completion) : NULL;
    if (!call) return NULL;
    usearch_js_plan_t* plan = NULL;
    if (!usearch_js_plan_from_value(env, arguments[2], &plan)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->header.finish = usearch_js_finish_report;
    finish.outcome = &call->header.outcome;
    finish.report = &call->report;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    // The library frees the plan whatever the call answers.
    finish.plan = plan ? plan->refine.sequence : NULL;
    if (plan) plan->refine.sequence = NULL;
    usearch_sequence_refine_plan_finish(&finish);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>search(executor, completion, data, offsets, scalar, arena, wanted, expansion, traversal,
 *  stats)</tt> → @c {keys, scores, counts, columns}, viewed inside @p arena. */
static napi_value usearch_js_sequence_search(napi_env env, napi_callback_info info) {
    napi_value arguments[10], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 10, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_search_t search = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &search.executor, &search.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_tape_from_value(env, arguments + 2, "queries", 1, &search.count, &search.queries) ||
        !usearch_js_size_from_value(env, arguments[6], "wanted", &search.wanted) ||
        !usearch_js_size_from_value(env, arguments[7], "expansionFactor", &search.expansion_factor) ||
        !usearch_js_traversal_from_value(env, arguments[8], "traversal", &search.traversal) ||
        !usearch_js_stats_from_value(env, arguments[9], &call->header, &search.stats) ||
        !usearch_js_arena_from_value(env, arguments[5], &search.arena)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self,         arguments[0], arguments[5],
                                                  arguments[2], arguments[3], arguments[9]};
    call->header.finish = usearch_js_finish_matches;
    call->count = search.count;
    search.outcome = &call->header.outcome;
    search.matches = &call->matches;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_search(&search);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>get(executor, completion, key, arena)</tt> → @c {data, offsets, entries}, every sequence under the
 *  key as a tape viewed inside @p arena. */
static napi_value usearch_js_sequence_get(napi_env env, napi_callback_info info) {
    napi_value arguments[4], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 4, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_get_t get = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &get.executor, &get.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_key_from_value(env, arguments[2], "key", &get.key) ||
        !usearch_js_arena_from_value(env, arguments[3], &get.arena)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[3]};
    call->header.finish = usearch_js_finish_tape;
    // One byte per element.
    call->count = 1;
    call->scalar = usearch_scalar_u8_k;
    get.outcome = &call->header.outcome;
    get.sequences = &call->tape;
    get.written = &call->written;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_get(&get);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>sample(executor, completion, arena, wanted, shown, data, rows, columns, stride, scalar, focus,
 *  seed, expansion)</tt> → @c {keys, entryOrdinals, parents, coordinates, draws} inside @p arena. */
static napi_value usearch_js_sequence_sample(napi_env env, napi_callback_info info) {
    napi_value arguments[13], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 13, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_sample_t sample = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &sample.executor, &sample.completion) : NULL;
    if (!call) return NULL;
    size_t seed = 0, placed = 0;
    sample.shown_coordinates = (usearch_rows_view_t) {NULL, 3 * sizeof(float), 3, usearch_scalar_f32_k};
    if (!usearch_js_size_from_value(env, arguments[3], "wanted", &sample.wanted) ||
        !usearch_js_optional_keys(env, arguments[4], "shown", &call->header, 0, &sample.shown, &sample.shown_count) ||
        (!usearch_js_value_absent(env, arguments[5]) &&
         !usearch_js_rows_from_value(env, arguments + 5, "shownCoordinates", &placed, &sample.shown_coordinates)) ||
        !usearch_js_optional_keys(env, arguments[10], "focus", &call->header, 1, &sample.focus, &sample.focus_count) ||
        !usearch_js_size_from_value(env, arguments[11], "seed", &seed) ||
        !usearch_js_size_from_value(env, arguments[12], "expansionFactor", &sample.expansion_factor) ||
        !usearch_js_arena_from_value(env, arguments[2], &sample.arena)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    // The coordinates hold no count of their own, so only this side can see them disagree.
    if (placed != sample.shown_count) {
        usearch_js_call_abandon(&call->header, &local.header);
        usearch_js_throw(env, usearch_status_shape_mismatch_k, "The shown keys need one place apiece");
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {
        self, arguments[0], arguments[2], arguments[5], arguments[4], [usearch_js_slot_focus_k] = arguments[10]};
    call->header.finish = usearch_js_finish_sample;
    call->scalar = usearch_scalar_f32_k;
    sample.outcome = &call->header.outcome;
    sample.keys = &call->keys;
    sample.entry_ordinals = &call->entry_ordinals;
    sample.parents = &call->matches;
    sample.coordinates = &call->vectors;
    sample.written = &call->written;
    sample.seed = seed;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_sample(&sample);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>reshard(executor, completion, sources, targets, seed, expansion, memoryLimit, stats, plan)</tt>
 *  → what the closing rounds did to the graph; the call copies the handles before it starts. */
static napi_value usearch_js_sequence_reshard(napi_env env, napi_callback_info info) {
    napi_value arguments[9], deferred = NULL;
    usearch_js_index_call_t local;
    usearch_sequence_reshard_t reshard = {0};
    if (!usearch_js_static_arguments(env, info, 9, arguments)) return NULL;
    usearch_js_index_call_t* const call = usearch_js_index_open(env, arguments, &local, &reshard.executor,
                                                                &reshard.completion);
    if (!call) return NULL;
    uint32_t sources_count = 0, targets_count = 0;
    size_t seed = 0;
    usearch_js_plan_t* plan = NULL;
    usearch_js_addon_t* addon = NULL;
    napi_get_instance_data(env, (void**)&addon);
    bool const counted = addon && usearch_js_handles_count(env, arguments[2], "sources", &sources_count) &&
                         usearch_js_handles_count(env, arguments[3], "targets", &targets_count);
    usearch_sequence_index_t* const handles =
        counted ? (usearch_sequence_index_t*)calloc((size_t)sources_count + targets_count + 1, sizeof(void*)) : NULL;
    call->header.owned[0] = handles;
    if (counted && !handles) usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not gather the handles");
    if (!handles ||
        !usearch_js_handles_from_value(env, arguments[2], "sources", sources_count, addon->sequence_index_class,
                                       (void**)handles) ||
        !usearch_js_handles_from_value(env, arguments[3], "targets", targets_count, addon->sequence_index_class,
                                       (void**)handles + sources_count) ||
        !usearch_js_size_from_value(env, arguments[4], "seed", &seed) ||
        !usearch_js_size_from_value(env, arguments[5], "expansionFactor", &reshard.refine.expansion_factor) ||
        !usearch_js_size_from_value(env, arguments[6], "memoryLimit", &reshard.refine.memory_limit) ||
        !usearch_js_stats_from_value(env, arguments[7], &call->header, &reshard.stats) ||
        !usearch_js_plan_from_value(env, arguments[8], &plan)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {
        NULL, arguments[0], NULL, arguments[2], arguments[3], arguments[7], [usearch_js_slot_plan_k] = arguments[8]};
    call->header.finish = usearch_js_finish_report;
    reshard.outcome = &call->header.outcome;
    reshard.sources_count = sources_count;
    reshard.sources = handles;
    reshard.targets_count = targets_count;
    reshard.targets = handles + sources_count;
    reshard.seed = seed;
    reshard.report = &call->report;
    reshard.plan = plan ? &plan->reshard.sequence : NULL;
    reshard.phase = plan ? &plan->progress : NULL;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_reshard(&reshard);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>reshardPlanFinish(executor, completion, plan)</tt> → what the whole plan did to the graph,
 *  closing and freeing it whatever the call answers. */
static napi_value usearch_js_sequence_reshard_plan_finish(napi_env env, napi_callback_info info) {
    napi_value arguments[3], deferred = NULL;
    usearch_js_index_call_t local;
    usearch_sequence_reshard_plan_finish_t finish = {0};
    if (!usearch_js_static_arguments(env, info, 3, arguments)) return NULL;
    usearch_js_index_call_t* const call = usearch_js_index_open(env, arguments, &local, &finish.executor,
                                                                &finish.completion);
    if (!call) return NULL;
    usearch_js_plan_t* plan = NULL;
    if (!usearch_js_plan_from_value(env, arguments[2], &plan)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {NULL, arguments[0]};
    call->header.finish = usearch_js_finish_report;
    finish.outcome = &call->header.outcome;
    finish.report = &call->report;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    finish.plan = plan ? plan->reshard.sequence : NULL;
    if (plan) plan->reshard.sequence = NULL;
    usearch_sequence_reshard_plan_finish(&finish);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

#pragma endregion Index Verbs

#pragma region Serialization

/** <tt>save(executor, completion, path)</tt> - writes the checkpoint the path's extension picks. */
static napi_value usearch_js_sequence_save(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_save_t save = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &save.executor, &save.completion) : NULL;
    if (!call) return NULL;
    call->header.owned[0] = usearch_js_string_from_value(env, arguments[2], "path", &save.path);
    if (!call->header.owned[0]) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->header.finish = usearch_js_finish_nothing;
    save.outcome = &call->header.outcome;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_save(&save);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

/** <tt>saveBuffer(executor, completion, arena)</tt> → one monolith's bytes, viewed inside @p arena. */
static napi_value usearch_js_sequence_save_buffer(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_index_call_t local;
    usearch_sequence_save_buffer_t save = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &save.executor, &save.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_arena_from_value(env, arguments[2], &save.arena)) {
        usearch_js_call_abandon(&call->header, &local.header);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[2]};
    call->header.finish = usearch_js_finish_save_buffer;
    save.outcome = &call->header.outcome;
    save.monolith = &call->monolith;
    if (!usearch_js_call_hold(env, &call->header, &local.header, slots, &deferred)) return NULL;
    usearch_sequence_save_buffer(&save);
    return usearch_js_call_answer(env, &call->header, &local.header, slots, deferred);
}

#pragma endregion Serialization

#pragma region Cursors

static void usearch_js_sequence_cursor_free(void* handle) {
    usearch_sequence_cursor_free((usearch_sequence_cursor_t)handle);
}

static void usearch_js_sequence_cursor_finalize(napi_env env, void* data, void* hint) {
    (void)hint;
    usearch_js_walk_release(env, (usearch_js_walk_t*)data, usearch_js_sequence_cursor_free);
    free(data);
}

/** <tt>next(executor, wanted, arena, links)</tt> → @c {keys, entries, links}, the next batch viewed
 *  inside @p arena, pulled on @p executor or the calling thread; zero entries end the walk. */
static napi_value usearch_js_sequence_cursor_next(napi_env env, napi_callback_info info) {
    napi_value arguments[4];
    usearch_sequence_cursor_next_t next = {0};
    void* index = NULL;
    usearch_js_walk_t* const walk = usearch_js_walk_self(env, info, 4, arguments, &index, &next.executor);
    if (!walk || !usearch_js_executor_from_value(env, arguments[0], &next.executor)) return NULL;
    usearch_js_index_call_t call;
    memset(&call, 0, sizeof(call));
    bool exports_links = false;
    if (!usearch_js_value_absent(env, arguments[3]) &&
        napi_get_value_bool(env, arguments[3], &exports_links) != napi_ok) {
        usearch_js_throw_field(env, "links", "must be a boolean");
        return NULL;
    }
    if (!usearch_js_size_from_value(env, arguments[1], "wanted", &next.wanted) ||
        !usearch_js_arena_from_value(env, arguments[2], &next.arena))
        return NULL;
    napi_value const slots[usearch_js_slots_k] = {NULL, NULL, arguments[2]};
    call.header.finish = exports_links ? usearch_js_finish_cursor_links : usearch_js_finish_cursor_keys;
    next.index = index;
    next.outcome = &call.header.outcome;
    next.cursor = (usearch_sequence_cursor_t)walk->handle;
    next.keys = &call.keys;
    next.links = exports_links ? &call.links : NULL;
    next.written = &call.written;
    usearch_sequence_cursor_next(&next);
    return usearch_js_call_answer(env, &call.header, &call.header, slots, NULL);
}

/** @c free() - releases the walk; a second call is a no-op. */
static napi_value usearch_js_sequence_cursor_release(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    usearch_js_walk_t* walk = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&walk) != napi_ok || !walk) {
        usearch_js_throw(env, usearch_status_internal_k, "The cursor is not attached");
        return NULL;
    }
    usearch_js_walk_release(env, walk, usearch_js_sequence_cursor_free);
    return usearch_js_undefined(env);
}

/** <tt>cursor(starts)</tt> → a walk over the entries published so far, keeping no executor;
 *  @p starts is absent for every entry as stored, else the keys to walk out from. */
static napi_value usearch_js_sequence_cursor(napi_env env, napi_callback_info info) {
    napi_value arguments[1], self = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 1, arguments, &self);
    usearch_sequence_cursor_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_sequence_cursor_init_t init = {.cursor = &handle, .outcome = &outcome};
    usearch_key_t* gathered = NULL;
    if (!index) return NULL;
    if (!usearch_js_value_absent(env, arguments[0]) &&
        !usearch_js_keys_from_value(env, arguments[0], "starts", &init.starts, &init.starts_count, &gathered))
        return NULL;
    init.index = index->handle;
    usearch_sequence_cursor_init(&init);
    free(gathered);
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    return usearch_js_walk_object(env, handle, self, usearch_js_undefined(env), usearch_js_sequence_cursor_next,
                                  usearch_js_sequence_cursor_release, usearch_js_sequence_cursor_finalize,
                                  usearch_js_sequence_cursor_free);
}

#pragma endregion Cursors

#pragma region Registration

napi_value usearch_js_sequence_index_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"free", NULL, usearch_js_sequence_release, NULL, NULL, NULL, napi_default_method, NULL},
        {"describe", NULL, usearch_js_sequence_describe, NULL, NULL, NULL, napi_default_method, NULL},
        {"health", NULL, usearch_js_sequence_health, NULL, NULL, NULL, napi_default_method, NULL},
        {"reserve", NULL, usearch_js_sequence_reserve, NULL, NULL, NULL, napi_default_method, NULL},
        {"clear", NULL, usearch_js_sequence_clear, NULL, NULL, NULL, napi_default_method, NULL},
        {"count", NULL, usearch_js_sequence_count, NULL, NULL, NULL, napi_default_method, NULL},
        {"insert", NULL, usearch_js_sequence_insert, NULL, NULL, NULL, napi_default_method, NULL},
        {"refine", NULL, usearch_js_sequence_refine, NULL, NULL, NULL, napi_default_method, NULL},
        {"refinePlanFinish", NULL, usearch_js_sequence_refine_plan_finish, NULL, NULL, NULL, napi_default_method, NULL},
        {"search", NULL, usearch_js_sequence_search, NULL, NULL, NULL, napi_default_method, NULL},
        {"get", NULL, usearch_js_sequence_get, NULL, NULL, NULL, napi_default_method, NULL},
        {"sample", NULL, usearch_js_sequence_sample, NULL, NULL, NULL, napi_default_method, NULL},
        {"save", NULL, usearch_js_sequence_save, NULL, NULL, NULL, napi_default_method, NULL},
        {"saveBuffer", NULL, usearch_js_sequence_save_buffer, NULL, NULL, NULL, napi_default_method, NULL},
        {"cursor", NULL, usearch_js_sequence_cursor, NULL, NULL, NULL, napi_default_method, NULL},
        {"initRam", NULL, usearch_js_sequence_init_ram, NULL, NULL, NULL, napi_static, NULL},
        {"initVram", NULL, usearch_js_sequence_init_vram, NULL, NULL, NULL, napi_static, NULL},
        {"loadRam", NULL, usearch_js_sequence_load_ram, NULL, NULL, NULL, napi_static, NULL},
        {"loadVram", NULL, usearch_js_sequence_load_vram, NULL, NULL, NULL, napi_static, NULL},
        {"loadRamBuffer", NULL, usearch_js_sequence_load_ram_buffer, NULL, NULL, NULL, napi_static, NULL},
        {"loadVramBuffer", NULL, usearch_js_sequence_load_vram_buffer, NULL, NULL, NULL, napi_static, NULL},
        {"viewRam", NULL, usearch_js_sequence_view_ram, NULL, NULL, NULL, napi_static, NULL},
        {"viewNvme", NULL, usearch_js_sequence_view_nvme, NULL, NULL, NULL, napi_static, NULL},
        {"metadataRead", NULL, usearch_js_sequence_metadata_read_path, NULL, NULL, NULL, napi_static, NULL},
        {"metadataReadBuffer", NULL, usearch_js_sequence_metadata_read_buffer, NULL, NULL, NULL, napi_static, NULL},
        {"reshard", NULL, usearch_js_sequence_reshard, NULL, NULL, NULL, napi_static, NULL},
        {"reshardPlanFinish", NULL, usearch_js_sequence_reshard_plan_finish, NULL, NULL, NULL, napi_static, NULL},
    };
    napi_value index_class = NULL;
    if (napi_define_class(env, "CompiledSequenceIndex", NAPI_AUTO_LENGTH, usearch_js_sequence_construct, NULL,
                          usearch_js_table_size_(properties), properties, &index_class) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not define the CompiledSequenceIndex class");
        return NULL;
    }
    return index_class;
}

#pragma endregion Registration
