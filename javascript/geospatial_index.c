/**
 *  @file javascript/geospatial_index.c
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Node-API search over observations in space and time under nearness, sight and view cones.
 *
 *  The @c CompiledGeospatialIndex class and the lifecycle verbs that instantiate it, the cursor it
 *  opens, and the checkpoint metadata readers that need no index at all. Observations cross as a
 *  @c Float64Array of eleven fields apiece, NaN where nothing was recorded, borrowed in place, and
 *  no handle carries a terrain.
 *
 *  Every instance verb takes the executor and the completion first, so one native function serves
 *  the calling thread, a blocking receiver and a deferred one alike.
 */
#include <stdlib.h> // `calloc`, `free`, `malloc`
#include <string.h> // `memset`

#include "lib.h"

#pragma region Calls

/** Fields one observation spans, the doubles of @c usearch_geospatial_observation_t. */
enum { usearch_js_geospatial_fields_k = sizeof(usearch_geospatial_observation_t) / sizeof(double) };

/** Borrows a @c Float64Array of whole observations in place. */
static bool usearch_js_observations_from_value(napi_env env, napi_value value, char const* field,
                                               usearch_geospatial_observation_t const** observations, size_t* count) {
    napi_typedarray_type type = napi_int8_array;
    size_t elements = 0;
    void* data = NULL;
    bool is_typedarray = false;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &type, &elements, &data, NULL, NULL) != napi_ok ||
        type != napi_float64_array)
        return usearch_js_throw_field(env, field, "must be a Float64Array of eleven fields per observation");
    // The call has one count and the array none, so only this side sees a partial observation.
    if (elements % usearch_js_geospatial_fields_k)
        return usearch_js_throw(env, usearch_status_shape_mismatch_k, "The observations span a partial one");
    *observations = (usearch_geospatial_observation_t const*)data;
    *count = elements / usearch_js_geospatial_fields_k;
    return true;
}

/** Reads the relation as the C enumerator's value, which the library refuses when it knows none,
 *  and the radius and time slack beside it. */
static bool usearch_js_relation_from_values(napi_env env, napi_value const* arguments,
                                            usearch_geospatial_relation_t* relation,
                                            usearch_geospatial_tolerance_t* tolerance) {
    uint32_t code = 0;
    if (napi_get_value_uint32(env, arguments[0], &code) != napi_ok)
        return usearch_js_throw_field(env, "relation", "must be a GeospatialRelation");
    *relation = (usearch_geospatial_relation_t)code;
    return usearch_js_double_from_value(env, arguments[1], "metres", &tolerance->metres) &&
           usearch_js_double_from_value(env, arguments[2], "seconds", &tolerance->seconds);
}

/** Wraps the connectivity a checkpoint froze. */
static napi_value usearch_js_geospatial_metadata_object(napi_env env, usearch_geospatial_metadata_t const* metadata) {
    static char const* const names[] = {"connectivity"};
    napi_value const values[] = {usearch_js_number(env, metadata->connectivity)};
    return usearch_js_object(env, usearch_js_table_size_(names), names, values);
}

static napi_value usearch_js_geospatial_finish_describe(napi_env env, usearch_js_call_t* call,
                                                        napi_value const* slots) {
    usearch_geospatial_index_description_t const* const description =
        &((usearch_js_index_call_t const*)call)->description.geospatial;
    napi_value const metadata = usearch_js_geospatial_metadata_object(env, &description->metadata);
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

static napi_value usearch_js_geospatial_finish_count(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_geospatial_key_stats_t const* const stats = &((usearch_js_index_call_t const*)call)->key_stats.geospatial;
    (void)slots;
    static char const* const names[] = {"entries"};
    napi_value const values[] = {usearch_js_number(env, stats->entries)};
    return usearch_js_object(env, usearch_js_table_size_(names), names, values);
}

/** Views the observations a retrieval wrote as a @c Float64Array of eleven fields apiece. */
static napi_value usearch_js_geospatial_finish_get(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    usearch_js_index_call_t const* const index_call = (usearch_js_index_call_t const*)call;
    size_t const fields = index_call->written * usearch_js_geospatial_fields_k;
    return usearch_js_arena_view(env, slots[usearch_js_slot_arena_k], napi_float64_array,
                                 (void*)index_call->observations, fields * sizeof(double), fields);
}

#pragma endregion Calls

#pragma region Index Construction

static void usearch_js_geospatial_finalize(napi_env env, void* data, void* hint) {
    usearch_js_index_t* const index = (usearch_js_index_t*)data;
    (void)env;
    (void)hint;
    if (!index) return;
    usearch_geospatial_free(index->handle);
    free(index);
}

/** @c CompiledGeospatialIndex() - an empty shell that a lifecycle verb fills with its handle. */
static napi_value usearch_js_geospatial_construct(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    if (!usearch_js_sealed(env, info, &self)) {
        usearch_js_throw(env, usearch_status_invalid_argument_k,
                         "A CompiledGeospatialIndex is made by a lifecycle verb");
        return NULL;
    }
    return self;
}

/** Wraps a handle a lifecycle verb just made, or throws its outcome and frees the handle. */
static napi_value usearch_js_geospatial_adopt(napi_env env, usearch_outcome_t const* outcome,
                                              usearch_geospatial_index_t handle) {
    if (!usearch_js_succeeded(env, outcome)) return NULL;
    usearch_js_addon_t* addon = NULL;
    napi_value instance = NULL;
    usearch_js_index_t* const index = (usearch_js_index_t*)calloc(1, sizeof(usearch_js_index_t));
    if (napi_get_instance_data(env, (void**)&addon) == napi_ok && addon)
        instance = usearch_js_instantiate(env, addon->geospatial_index_class);
    if (!instance || !index || napi_wrap(env, instance, index, usearch_js_geospatial_finalize, NULL, NULL) != napi_ok) {
        free(index);
        usearch_geospatial_free(handle);
        usearch_js_throw(env, usearch_status_internal_k, "Could not instantiate the CompiledGeospatialIndex");
        return NULL;
    }
    index->handle = handle;
    return instance;
}

/** <tt>initRam(connectivity, valuesPerKey, memoryDomains, cpuClassCeiling)</tt> and <tt>initVram(...,
 *  devices, gpuClassCeiling)</tt> → an empty, mutable index without terrain. */
static napi_value usearch_js_geospatial_init(napi_env env, napi_callback_info info, usearch_residency_t residency) {
    napi_value arguments[4];
    size_t connectivity = 0, values_per_key = 0;
    usearch_cpu_class_t cpu_class_ceiling = usearch_cpu_class_unknown_k;
    usearch_gpu_class_t gpu_class_ceiling = usearch_gpu_class_unknown_k;
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    bool const vram = residency == usearch_residency_vram_k;
    if (!usearch_js_static_arguments(env, info, 4, arguments) ||
        !(vram ? usearch_js_gpu_class_from_value(env, arguments[3], &gpu_class_ceiling)
               : usearch_js_cpu_class_from_value(env, arguments[3], &cpu_class_ceiling)) ||
        !usearch_js_size_from_value(env, arguments[0], "connectivity", &connectivity) ||
        !usearch_js_size_from_value(env, arguments[1], "valuesPerKey", &values_per_key) ||
        !usearch_js_ordinals_from_value(env, arguments[2], vram ? "devices" : "memoryDomains", &ordinals,
                                        &ordinals_count))
        return NULL;

    usearch_geospatial_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    if (vram) {
        usearch_geospatial_init_vram_t init = {.index = &handle,
                                               .outcome = &outcome,
                                               .connectivity = connectivity,
                                               .values_per_key = values_per_key,
                                               .vram = {ordinals, ordinals_count, 0, gpu_class_ceiling}};
        usearch_geospatial_init_vram(&init);
    }
    else {
        usearch_geospatial_init_ram_t init = {.index = &handle,
                                              .outcome = &outcome,
                                              .connectivity = connectivity,
                                              .values_per_key = values_per_key,
                                              .ram = {ordinals, ordinals_count, 0, cpu_class_ceiling}};
        usearch_geospatial_init_ram(&init);
    }
    free(ordinals);
    return usearch_js_geospatial_adopt(env, &outcome, handle);
}

static napi_value usearch_js_geospatial_init_ram(napi_env env, napi_callback_info info) {
    return usearch_js_geospatial_init(env, info, usearch_residency_ram_k);
}

static napi_value usearch_js_geospatial_init_vram(napi_env env, napi_callback_info info) {
    return usearch_js_geospatial_init(env, info, usearch_residency_vram_k);
}

/** <tt>loadRam(path, executor, memoryDomains, valuesPerKey, cpuClassCeiling)</tt>,
 *  <tt>loadVram(path, executor, devices, valuesPerKey, gpuClassCeiling)</tt>, and their @c Buffer
 *  twins taking any @c TypedArray → a new, mutable index. */
static napi_value usearch_js_geospatial_load(napi_env env, napi_callback_info info, usearch_residency_t residency,
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

    usearch_geospatial_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_ram_options_t const ram = {ordinals, ordinals_count, 0, cpu_class_ceiling};
    usearch_vram_options_t const vram_options = {ordinals, ordinals_count, 0, gpu_class_ceiling};
    if (vram && source == usearch_js_source_path_k) {
        usearch_geospatial_load_vram_t load = {.index = &handle,
                                               .executor = executor,
                                               .outcome = &outcome,
                                               .path = bytes,
                                               .vram = vram_options,
                                               .values_per_key = values_per_key};
        usearch_geospatial_load_vram(&load);
    }
    else if (vram) {
        usearch_geospatial_load_vram_buffer_t load = {.index = &handle,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .buffer = bytes,
                                                      .vram = vram_options,
                                                      .values_per_key = values_per_key};
        usearch_geospatial_load_vram_buffer(&load);
    }
    else if (source == usearch_js_source_path_k) {
        usearch_geospatial_load_ram_t load = {.index = &handle,
                                              .executor = executor,
                                              .outcome = &outcome,
                                              .path = bytes,
                                              .ram = ram,
                                              .values_per_key = values_per_key};
        usearch_geospatial_load_ram(&load);
    }
    else {
        usearch_geospatial_load_ram_buffer_t load = {.index = &handle,
                                                     .executor = executor,
                                                     .outcome = &outcome,
                                                     .buffer = bytes,
                                                     .ram = ram,
                                                     .values_per_key = values_per_key};
        usearch_geospatial_load_ram_buffer(&load);
    }
    free(ordinals);
    free(path);
    return usearch_js_geospatial_adopt(env, &outcome, handle);
}

static napi_value usearch_js_geospatial_load_ram(napi_env env, napi_callback_info info) {
    return usearch_js_geospatial_load(env, info, usearch_residency_ram_k, usearch_js_source_path_k);
}

static napi_value usearch_js_geospatial_load_vram(napi_env env, napi_callback_info info) {
    return usearch_js_geospatial_load(env, info, usearch_residency_vram_k, usearch_js_source_path_k);
}

static napi_value usearch_js_geospatial_load_ram_buffer(napi_env env, napi_callback_info info) {
    return usearch_js_geospatial_load(env, info, usearch_residency_ram_k, usearch_js_source_buffer_k);
}

static napi_value usearch_js_geospatial_load_vram_buffer(napi_env env, napi_callback_info info) {
    return usearch_js_geospatial_load(env, info, usearch_residency_vram_k, usearch_js_source_buffer_k);
}

/** <tt>viewRam(path, cpuClassCeiling)</tt> → a read-only index serving the checkpoint through a memory
 *  mapping. */
static napi_value usearch_js_geospatial_view_ram(napi_env env, napi_callback_info info) {
    napi_value arguments[2];
    usearch_geospatial_view_ram_t view_ram = {0};
    if (!usearch_js_static_arguments(env, info, 2, arguments) ||
        !usearch_js_cpu_class_from_value(env, arguments[1], &view_ram.ram.cpu_class_ceiling))
        return NULL;
    char* const path = usearch_js_string_from_value(env, arguments[0], "path", &view_ram.path);
    if (!path) return NULL;

    usearch_geospatial_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    view_ram.index = &handle;
    view_ram.outcome = &outcome;
    usearch_geospatial_view_ram(&view_ram);
    free(path);
    return usearch_js_geospatial_adopt(env, &outcome, handle);
}

/** <tt>viewNvme(path, executor, paths, cpuClassCeiling)</tt> → a read-only index served from NVMe
 *  drives. */
static napi_value usearch_js_geospatial_view_nvme(napi_env env, napi_callback_info info) {
    napi_value arguments[4];
    usearch_geospatial_view_nvme_t view_nvme = {0};
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

    usearch_geospatial_index_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    view_nvme.index = &handle;
    view_nvme.outcome = &outcome;
    view_nvme.nvme = (usearch_nvme_options_t) {paths.views, paths.count, cpu_class_ceiling};
    usearch_geospatial_view_nvme(&view_nvme);
    usearch_js_paths_release(&paths);
    free(path);
    return usearch_js_geospatial_adopt(env, &outcome, handle);
}

/** <tt>metadataRead(path, executor)</tt> and <tt>metadataReadBuffer(buffer, executor)</tt> → what a checkpoint
 *  froze. */
static napi_value usearch_js_geospatial_metadata_read(napi_env env, napi_callback_info info,
                                                      usearch_js_source_t source) {
    napi_value arguments[2];
    usearch_geospatial_metadata_t metadata;
    memset(&metadata, 0, sizeof(metadata));
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t executor = NULL;
    if (!usearch_js_static_arguments(env, info, 2, arguments) ||
        !usearch_js_executor_from_value(env, arguments[1], &executor))
        return NULL;
    if (source == usearch_js_source_path_k) {
        usearch_geospatial_metadata_read_t read = {.executor = executor, .outcome = &outcome, .metadata = &metadata};
        char* const path = usearch_js_string_from_value(env, arguments[0], "path", &read.path);
        if (!path) return NULL;
        usearch_geospatial_metadata_read(&read);
        free(path);
    }
    else {
        void* buffer = NULL;
        usearch_geospatial_metadata_read_buffer_t read = {.executor = executor,
                                                          .outcome = &outcome,
                                                          .metadata = &metadata};
        if (!usearch_js_bytes_from_value(env, arguments[0], "buffer", &buffer, &read.buffer.length)) return NULL;
        read.buffer.data = (uint8_t const*)buffer;
        usearch_geospatial_metadata_read_buffer(&read);
    }
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    return usearch_js_geospatial_metadata_object(env, &metadata);
}

static napi_value usearch_js_geospatial_metadata_read_path(napi_env env, napi_callback_info info) {
    return usearch_js_geospatial_metadata_read(env, info, usearch_js_source_path_k);
}

static napi_value usearch_js_geospatial_metadata_read_buffer(napi_env env, napi_callback_info info) {
    return usearch_js_geospatial_metadata_read(env, info, usearch_js_source_buffer_k);
}

/** @c free() - releases the handle now, waiting for every deferred call naming it; idempotent. */
static napi_value usearch_js_geospatial_release(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    usearch_js_index_t* index = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&index) != napi_ok || !index) {
        usearch_js_throw(env, usearch_status_internal_k, "The CompiledGeospatialIndex is not attached");
        return NULL;
    }
    usearch_geospatial_free(index->handle);
    index->handle = NULL;
    return usearch_js_undefined(env);
}

#pragma endregion Index Construction

#pragma region Index Verbs

/** <tt>describe(executor, completion)</tt> → the frozen metadata and current counters, each read once. */
static napi_value usearch_js_geospatial_describe(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_describe_t describe = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &describe.executor, &describe.completion) : NULL;
    if (!call) return NULL;
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_geospatial_finish_describe;
    describe.outcome = &call->call.outcome;
    describe.description = &call->description.geospatial;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_describe(&describe);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>health(executor, completion)</tt> → how much of the graph a search reaches from the root. */
static napi_value usearch_js_geospatial_health(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_health_t health = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &health.executor, &health.completion) : NULL;
    if (!call) return NULL;
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_health;
    health.outcome = &call->call.outcome;
    health.reachability = &call->reachability;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_health(&health);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>reserve(executor, completion, entries)</tt> - claims room for a total entry count up front. */
static napi_value usearch_js_geospatial_reserve(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_reserve_t reserve = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &reserve.executor, &reserve.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_size_from_value(env, arguments[2], "entries", &reserve.entries)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_nothing;
    reserve.outcome = &call->call.outcome;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_reserve(&reserve);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>clear(executor, completion)</tt> - erases every entry, keeping capacity. */
static napi_value usearch_js_geospatial_clear(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_clear_t clear = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &clear.executor, &clear.completion) : NULL;
    if (!call) return NULL;
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_nothing;
    clear.outcome = &call->call.outcome;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_clear(&clear);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>count(executor, completion, key)</tt> → @c {entries} under one identity. */
static napi_value usearch_js_geospatial_count(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_count_t count = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &count.executor, &count.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_key_from_value(env, arguments[2], "key", &count.key)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_geospatial_finish_count;
    count.outcome = &call->call.outcome;
    count.key_stats = &call->key_stats.geospatial;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_count(&count);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>insert(executor, completion, keys, observations, expansion, stats)</tt>. */
static napi_value usearch_js_geospatial_insert(napi_env env, napi_callback_info info) {
    napi_value arguments[6], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 6, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_insert_t insert = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &insert.executor, &insert.completion) : NULL;
    if (!call) return NULL;
    size_t keys_count = 0;
    usearch_key_t** const gathered = (usearch_key_t**)&call->call.owned[0];
    if (!usearch_js_observations_from_value(env, arguments[3], "observations", &insert.observations, &insert.count) ||
        !usearch_js_size_from_value(env, arguments[4], "expansionFactor", &insert.expansion_factor) ||
        !usearch_js_stats_from_value(env, arguments[5], &call->call, &insert.stats) ||
        !usearch_js_keys_from_value(env, arguments[2], "keys", &insert.keys.data, &keys_count, gathered)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    // The keys view carries no count, so only this side can see the two disagree.
    if (keys_count != insert.count) {
        usearch_js_call_abandon(&call->call, &local.call);
        usearch_js_throw(env, usearch_status_shape_mismatch_k, "The keys name one key per observation");
        return NULL;
    }
    insert.keys.stride = sizeof(usearch_key_t);
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], NULL, arguments[3], arguments[2], arguments[5]};
    call->call.finish = usearch_js_finish_nothing;
    insert.outcome = &call->call.outcome;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_insert(&insert);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>refine(executor, completion, expansion, memoryLimit, stats, plan)</tt> → what the round did
 *  to the graph; a @p plan continues across calls until it is finished. */
static napi_value usearch_js_geospatial_refine(napi_env env, napi_callback_info info) {
    napi_value arguments[6], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 6, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_refine_t refine = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &refine.executor, &refine.completion) : NULL;
    if (!call) return NULL;
    usearch_js_plan_t* plan = NULL;
    if (!usearch_js_size_from_value(env, arguments[2], "expansionFactor", &refine.refine.expansion_factor) ||
        !usearch_js_size_from_value(env, arguments[3], "memoryLimit", &refine.refine.memory_limit) ||
        !usearch_js_stats_from_value(env, arguments[4], &call->call, &refine.stats) ||
        !usearch_js_plan_from_value(env, arguments[5], &plan)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {
        self, arguments[0], NULL, NULL, NULL, arguments[4], [usearch_js_slot_plan_k] = arguments[5]};
    call->call.finish = usearch_js_finish_report;
    refine.plan = plan ? &plan->refine.geospatial : NULL;
    refine.phase = plan ? &plan->progress : NULL;
    refine.outcome = &call->call.outcome;
    refine.report = &call->report;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_refine(&refine);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>refinePlanFinish(executor, completion, plan)</tt> → what the whole plan did to the graph,
 *  closing and freeing it whatever the call answers. */
static napi_value usearch_js_geospatial_refine_plan_finish(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_refine_plan_finish_t finish = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &finish.executor, &finish.completion) : NULL;
    if (!call) return NULL;
    usearch_js_plan_t* plan = NULL;
    if (!usearch_js_plan_from_value(env, arguments[2], &plan)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0]};
    call->call.finish = usearch_js_finish_report;
    finish.outcome = &call->call.outcome;
    finish.report = &call->report;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    // The library frees the plan whatever the call answers.
    finish.plan = plan ? plan->refine.geospatial : NULL;
    if (plan) plan->refine.geospatial = NULL;
    usearch_geospatial_refine_plan_finish(&finish);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>search(executor, completion, observations, relation, metres, seconds, arena, wanted, expansion,
 *  traversal, stats)</tt> → @c {keys, scores, counts, columns}, chord metres viewed inside @p arena. */
static napi_value usearch_js_geospatial_search(napi_env env, napi_callback_info info) {
    napi_value arguments[11], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 11, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_search_t search = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &search.executor, &search.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_observations_from_value(env, arguments[2], "queries", &search.queries, &search.count) ||
        !usearch_js_relation_from_values(env, arguments + 3, &search.relation, &search.tolerance) ||
        !usearch_js_size_from_value(env, arguments[7], "wanted", &search.wanted) ||
        !usearch_js_size_from_value(env, arguments[8], "expansionFactor", &search.expansion_factor) ||
        !usearch_js_traversal_from_value(env, arguments[9], "traversal", &search.traversal) ||
        !usearch_js_stats_from_value(env, arguments[10], &call->call, &search.stats) ||
        !usearch_js_arena_from_value(env, arguments[6], &search.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[6], arguments[2], NULL, arguments[10]};
    call->call.finish = usearch_js_finish_matches;
    call->count = search.count;
    search.outcome = &call->call.outcome;
    search.matches = &call->matches;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_search(&search);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>within(executor, completion, observations, relation, metres, seconds, arena, traversal, stats)</tt>
 *  → @c {offsets, keys, scores, required}, every entry inside each query's radius viewed inside
 *  @p arena, query @c i spanning @c offsets[i] up to @c offsets[i+1]. */
static napi_value usearch_js_geospatial_within(napi_env env, napi_callback_info info) {
    napi_value arguments[9], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 9, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_within_t within = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &within.executor, &within.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_observations_from_value(env, arguments[2], "queries", &within.queries, &within.count) ||
        !usearch_js_relation_from_values(env, arguments + 3, &within.relation, &within.tolerance) ||
        !usearch_js_traversal_from_value(env, arguments[7], "traversal", &within.traversal) ||
        !usearch_js_stats_from_value(env, arguments[8], &call->call, &within.stats) ||
        !usearch_js_arena_from_value(env, arguments[6], &within.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[6], arguments[2], NULL, arguments[8]};
    call->call.finish = usearch_js_finish_multiply;
    call->count = within.count;
    within.outcome = &call->call.outcome;
    within.matches = &call->products;
    within.written = &call->written;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_within(&within);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>get(executor, completion, key, arena)</tt> → every observation under the key, a @c Float64Array of
 *  eleven fields apiece viewed inside @p arena. */
static napi_value usearch_js_geospatial_get(napi_env env, napi_callback_info info) {
    napi_value arguments[4], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 4, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_get_t get = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &get.executor, &get.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_key_from_value(env, arguments[2], "key", &get.key) ||
        !usearch_js_arena_from_value(env, arguments[3], &get.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[3]};
    call->call.finish = usearch_js_geospatial_finish_get;
    get.outcome = &call->call.outcome;
    get.observations = &call->observations;
    get.written = &call->written;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_get(&get);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>sample(executor, completion, arena, wanted, shown, data, rows, columns, stride, scalar, focus,
 *  seed, expansion)</tt> → @c {keys, entryOrdinals, parents, coordinates, draws} inside @p arena. */
static napi_value usearch_js_geospatial_sample(napi_env env, napi_callback_info info) {
    napi_value arguments[13], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 13, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_sample_t sample = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &sample.executor, &sample.completion) : NULL;
    if (!call) return NULL;
    size_t seed = 0, placed = 0;
    sample.shown_coordinates = (usearch_rows_view_t) {NULL, 3 * sizeof(float), 3, usearch_scalar_f32_k};
    if (!usearch_js_size_from_value(env, arguments[3], "wanted", &sample.wanted) ||
        !usearch_js_optional_keys(env, arguments[4], "shown", &call->call, 0, &sample.shown, &sample.shown_count) ||
        (!usearch_js_value_absent(env, arguments[5]) &&
         !usearch_js_rows_from_value(env, arguments + 5, "shownCoordinates", &placed, &sample.shown_coordinates)) ||
        !usearch_js_optional_keys(env, arguments[10], "focus", &call->call, 1, &sample.focus, &sample.focus_count) ||
        !usearch_js_size_from_value(env, arguments[11], "seed", &seed) ||
        !usearch_js_size_from_value(env, arguments[12], "expansionFactor", &sample.expansion_factor) ||
        !usearch_js_arena_from_value(env, arguments[2], &sample.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    // The coordinates carry no count of their own, so only this side can see them disagree.
    if (placed != sample.shown_count) {
        usearch_js_call_abandon(&call->call, &local.call);
        usearch_js_throw(env, usearch_status_shape_mismatch_k, "The shown keys need one place apiece");
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {
        self, arguments[0], arguments[2], arguments[5], arguments[4], [usearch_js_slot_focus_k] = arguments[10]};
    call->call.finish = usearch_js_finish_sample;
    call->scalar = usearch_scalar_f32_k;
    sample.outcome = &call->call.outcome;
    sample.keys = &call->keys;
    sample.entry_ordinals = &call->entry_ordinals;
    sample.parents = &call->matches;
    sample.coordinates = &call->vectors;
    sample.written = &call->written;
    sample.seed = seed;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_sample(&sample);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>reshard(executor, completion, sources, targets, seed, expansion, memoryLimit, stats, plan)</tt>
 *  → what the closing rounds did to the graph; the call copies the handles before it starts. */
static napi_value usearch_js_geospatial_reshard(napi_env env, napi_callback_info info) {
    napi_value arguments[9], deferred = NULL;
    usearch_js_index_call_t local;
    usearch_geospatial_reshard_t reshard = {0};
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
    usearch_geospatial_index_t* const handles =
        counted ? (usearch_geospatial_index_t*)calloc((size_t)sources_count + targets_count + 1, sizeof(void*)) : NULL;
    call->call.owned[0] = handles;
    if (counted && !handles) usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not gather the handles");
    if (!handles ||
        !usearch_js_handles_from_value(env, arguments[2], "sources", sources_count, addon->geospatial_index_class,
                                       (void**)handles) ||
        !usearch_js_handles_from_value(env, arguments[3], "targets", targets_count, addon->geospatial_index_class,
                                       (void**)handles + sources_count) ||
        !usearch_js_size_from_value(env, arguments[4], "seed", &seed) ||
        !usearch_js_size_from_value(env, arguments[5], "expansionFactor", &reshard.refine.expansion_factor) ||
        !usearch_js_size_from_value(env, arguments[6], "memoryLimit", &reshard.refine.memory_limit) ||
        !usearch_js_stats_from_value(env, arguments[7], &call->call, &reshard.stats) ||
        !usearch_js_plan_from_value(env, arguments[8], &plan)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {
        NULL, arguments[0], NULL, arguments[2], arguments[3], arguments[7], [usearch_js_slot_plan_k] = arguments[8]};
    call->call.finish = usearch_js_finish_report;
    reshard.outcome = &call->call.outcome;
    reshard.sources_count = sources_count;
    reshard.sources = handles;
    reshard.targets_count = targets_count;
    reshard.targets = handles + sources_count;
    reshard.seed = seed;
    reshard.report = &call->report;
    reshard.plan = plan ? &plan->reshard.geospatial : NULL;
    reshard.phase = plan ? &plan->progress : NULL;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_reshard(&reshard);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>reshardPlanFinish(executor, completion, plan)</tt> → what the whole plan did to the graph,
 *  closing and freeing it whatever the call answers. */
static napi_value usearch_js_geospatial_reshard_plan_finish(napi_env env, napi_callback_info info) {
    napi_value arguments[3], deferred = NULL;
    usearch_js_index_call_t local;
    usearch_geospatial_reshard_plan_finish_t finish = {0};
    if (!usearch_js_static_arguments(env, info, 3, arguments)) return NULL;
    usearch_js_index_call_t* const call = usearch_js_index_open(env, arguments, &local, &finish.executor,
                                                                &finish.completion);
    if (!call) return NULL;
    usearch_js_plan_t* plan = NULL;
    if (!usearch_js_plan_from_value(env, arguments[2], &plan)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {NULL, arguments[0]};
    call->call.finish = usearch_js_finish_report;
    finish.outcome = &call->call.outcome;
    finish.report = &call->report;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    finish.plan = plan ? plan->reshard.geospatial : NULL;
    if (plan) plan->reshard.geospatial = NULL;
    usearch_geospatial_reshard_plan_finish(&finish);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

#pragma endregion Index Verbs

#pragma region Serialization

/** <tt>save(executor, completion, path)</tt> - writes the checkpoint the path's extension picks. */
static napi_value usearch_js_geospatial_save(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_save_t save = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &save.executor, &save.completion) : NULL;
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
    usearch_geospatial_save(&save);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

/** <tt>saveBuffer(executor, completion, arena)</tt> → one monolith's bytes, viewed inside @p arena. */
static napi_value usearch_js_geospatial_save_buffer(napi_env env, napi_callback_info info) {
    napi_value arguments[3], self = NULL, deferred = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 3, arguments, &self);
    usearch_js_index_call_t local;
    usearch_geospatial_save_buffer_t save = {.index = index ? index->handle : NULL};
    usearch_js_index_call_t* const call =
        index ? usearch_js_index_open(env, arguments, &local, &save.executor, &save.completion) : NULL;
    if (!call) return NULL;
    if (!usearch_js_arena_from_value(env, arguments[2], &save.arena)) {
        usearch_js_call_abandon(&call->call, &local.call);
        return NULL;
    }
    napi_value const slots[usearch_js_slots_k] = {self, arguments[0], arguments[2]};
    call->call.finish = usearch_js_finish_save_buffer;
    save.outcome = &call->call.outcome;
    save.monolith = &call->monolith;
    if (!usearch_js_call_hold(env, &call->call, &local.call, slots, &deferred)) return NULL;
    usearch_geospatial_save_buffer(&save);
    return usearch_js_call_answer(env, &call->call, &local.call, slots, deferred);
}

#pragma endregion Serialization

#pragma region Cursors

static void usearch_js_geospatial_cursor_free(void* handle) {
    usearch_geospatial_cursor_free((usearch_geospatial_cursor_t)handle);
}

static void usearch_js_geospatial_cursor_finalize(napi_env env, void* data, void* hint) {
    (void)hint;
    usearch_js_walk_release(env, (usearch_js_walk_t*)data, usearch_js_geospatial_cursor_free);
    free(data);
}

/** <tt>next(wanted, arena, links)</tt> → @c {keys, entries, links}, the next batch viewed inside @p arena;
 *  zero entries mean the walk ended. */
static napi_value usearch_js_geospatial_cursor_next(napi_env env, napi_callback_info info) {
    napi_value arguments[3];
    usearch_geospatial_cursor_next_t next = {0};
    void* index = NULL;
    usearch_js_walk_t* const walk = usearch_js_walk_self(env, info, 3, arguments, &index, &next.executor);
    if (!walk) return NULL;
    usearch_js_index_call_t call;
    memset(&call, 0, sizeof(call));
    bool exports_links = false;
    if (!usearch_js_value_absent(env, arguments[2]) &&
        napi_get_value_bool(env, arguments[2], &exports_links) != napi_ok) {
        usearch_js_throw_field(env, "links", "must be a boolean");
        return NULL;
    }
    if (!usearch_js_size_from_value(env, arguments[0], "wanted", &next.wanted) ||
        !usearch_js_arena_from_value(env, arguments[1], &next.arena))
        return NULL;
    napi_value const slots[usearch_js_slots_k] = {NULL, NULL, arguments[1]};
    call.call.finish = exports_links ? usearch_js_finish_cursor_links : usearch_js_finish_cursor_keys;
    next.index = index;
    next.outcome = &call.call.outcome;
    next.cursor = (usearch_geospatial_cursor_t)walk->handle;
    next.keys = &call.keys;
    next.links = exports_links ? &call.links : NULL;
    next.written = &call.written;
    usearch_geospatial_cursor_next(&next);
    return usearch_js_call_answer(env, &call.call, &call.call, slots, NULL);
}

/** @c free() - releases the walk; a second call is a no-op. */
static napi_value usearch_js_geospatial_cursor_release(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    usearch_js_walk_t* walk = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok ||
        napi_unwrap(env, self, (void**)&walk) != napi_ok || !walk) {
        usearch_js_throw(env, usearch_status_internal_k, "The cursor is not attached");
        return NULL;
    }
    usearch_js_walk_release(env, walk, usearch_js_geospatial_cursor_free);
    return usearch_js_undefined(env);
}

/** <tt>cursor(executor, starts)</tt> → a walk over the entries published so far, pulling on @p executor;
 *  @p starts is absent for every entry as stored, else the keys to walk out from. */
static napi_value usearch_js_geospatial_cursor(napi_env env, napi_callback_info info) {
    napi_value arguments[2], self = NULL;
    usearch_js_index_t* const index = usearch_js_index_self(env, info, 2, arguments, &self);
    usearch_executor_t executor = NULL;
    usearch_geospatial_cursor_t handle = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_geospatial_cursor_init_t init = {.cursor = &handle, .outcome = &outcome};
    usearch_key_t* gathered = NULL;
    if (!index || !usearch_js_executor_from_value(env, arguments[0], &executor)) return NULL;
    if (!usearch_js_value_absent(env, arguments[1]) &&
        !usearch_js_keys_from_value(env, arguments[1], "starts", &init.starts, &init.starts_count, &gathered))
        return NULL;
    init.index = index->handle;
    usearch_geospatial_cursor_init(&init);
    free(gathered);
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    return usearch_js_walk_object(env, handle, self, arguments[0], usearch_js_geospatial_cursor_next,
                                  usearch_js_geospatial_cursor_release, usearch_js_geospatial_cursor_finalize,
                                  usearch_js_geospatial_cursor_free);
}

#pragma endregion Cursors

#pragma region Registration

napi_value usearch_js_geospatial_index_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"free", NULL, usearch_js_geospatial_release, NULL, NULL, NULL, napi_default_method, NULL},
        {"describe", NULL, usearch_js_geospatial_describe, NULL, NULL, NULL, napi_default_method, NULL},
        {"health", NULL, usearch_js_geospatial_health, NULL, NULL, NULL, napi_default_method, NULL},
        {"reserve", NULL, usearch_js_geospatial_reserve, NULL, NULL, NULL, napi_default_method, NULL},
        {"clear", NULL, usearch_js_geospatial_clear, NULL, NULL, NULL, napi_default_method, NULL},
        {"count", NULL, usearch_js_geospatial_count, NULL, NULL, NULL, napi_default_method, NULL},
        {"insert", NULL, usearch_js_geospatial_insert, NULL, NULL, NULL, napi_default_method, NULL},
        {"refine", NULL, usearch_js_geospatial_refine, NULL, NULL, NULL, napi_default_method, NULL},
        {"refinePlanFinish", NULL, usearch_js_geospatial_refine_plan_finish, NULL, NULL, NULL, napi_default_method,
         NULL},
        {"search", NULL, usearch_js_geospatial_search, NULL, NULL, NULL, napi_default_method, NULL},
        {"within", NULL, usearch_js_geospatial_within, NULL, NULL, NULL, napi_default_method, NULL},
        {"get", NULL, usearch_js_geospatial_get, NULL, NULL, NULL, napi_default_method, NULL},
        {"sample", NULL, usearch_js_geospatial_sample, NULL, NULL, NULL, napi_default_method, NULL},
        {"save", NULL, usearch_js_geospatial_save, NULL, NULL, NULL, napi_default_method, NULL},
        {"saveBuffer", NULL, usearch_js_geospatial_save_buffer, NULL, NULL, NULL, napi_default_method, NULL},
        {"cursor", NULL, usearch_js_geospatial_cursor, NULL, NULL, NULL, napi_default_method, NULL},
        {"initRam", NULL, usearch_js_geospatial_init_ram, NULL, NULL, NULL, napi_static, NULL},
        {"initVram", NULL, usearch_js_geospatial_init_vram, NULL, NULL, NULL, napi_static, NULL},
        {"loadRam", NULL, usearch_js_geospatial_load_ram, NULL, NULL, NULL, napi_static, NULL},
        {"loadVram", NULL, usearch_js_geospatial_load_vram, NULL, NULL, NULL, napi_static, NULL},
        {"loadRamBuffer", NULL, usearch_js_geospatial_load_ram_buffer, NULL, NULL, NULL, napi_static, NULL},
        {"loadVramBuffer", NULL, usearch_js_geospatial_load_vram_buffer, NULL, NULL, NULL, napi_static, NULL},
        {"viewRam", NULL, usearch_js_geospatial_view_ram, NULL, NULL, NULL, napi_static, NULL},
        {"viewNvme", NULL, usearch_js_geospatial_view_nvme, NULL, NULL, NULL, napi_static, NULL},
        {"metadataRead", NULL, usearch_js_geospatial_metadata_read_path, NULL, NULL, NULL, napi_static, NULL},
        {"metadataReadBuffer", NULL, usearch_js_geospatial_metadata_read_buffer, NULL, NULL, NULL, napi_static, NULL},
        {"reshard", NULL, usearch_js_geospatial_reshard, NULL, NULL, NULL, napi_static, NULL},
        {"reshardPlanFinish", NULL, usearch_js_geospatial_reshard_plan_finish, NULL, NULL, NULL, napi_static, NULL},
    };
    napi_value index_class = NULL;
    if (napi_define_class(env, "CompiledGeospatialIndex", NAPI_AUTO_LENGTH, usearch_js_geospatial_construct, NULL,
                          usearch_js_table_size_(properties), properties, &index_class) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not define the CompiledGeospatialIndex class");
        return NULL;
    }
    return index_class;
}

#pragma endregion Registration
