/**
 *  @brief The vocabulary every modality shares - the Node-API twin of @c include/usearch/types.h.
 *  @file javascript/types.c
 *  @author Ash Vardanian
 *
 *  Closed-set naming, argument parsing, identities, matrix borrowing, and the @c Executor whose
 *  threads or streams every fanned-out verb runs on.
 */
#include <stdio.h>  // `snprintf`
#include <stdlib.h> // `malloc`, `free`
#include <string.h> // `memcpy`, `memset`, `strcmp`

#include "lib.h"

#pragma region Naming

/** @brief One spelling of a closed-set option and the value it selects. */
typedef struct usearch_js_choice_naming_t {
    /** @brief The spelling, as a caller writes it. */
    char const* name;
    /** @brief The enumerator the spelling selects. */
    int value;
} usearch_js_choice_naming_t;

/**
 *  @brief The metric kinds, one name each.
 *
 *  No aliases: two spellings of one enumerator drift apart the moment one of them acquires a
 *  meaning the other never had.
 */
static usearch_js_choice_naming_t const usearch_js_metric_namings[] = {
    {"inner", usearch_dense_metric_inner_k},         {"angular", usearch_dense_metric_angular_k},
    {"euclidean", usearch_dense_metric_euclidean_k}, {"hamming", usearch_dense_metric_hamming_k},
    {"jaccard", usearch_dense_metric_jaccard_k},
};

/**
 *  @brief The scalar types, ordered by shrinking dynamic range - floats widest exponent first,
 *      then signed integers descending, then unsigned.
 */
static usearch_js_choice_naming_t const usearch_js_scalar_namings[] = {
    {"f64", usearch_scalar_f64_k},   {"f32", usearch_scalar_f32_k},   {"bf16", usearch_scalar_bf16_k},
    {"f16", usearch_scalar_f16_k},   {"e5m2", usearch_scalar_e5m2_k}, {"e4m3", usearch_scalar_e4m3_k},
    {"e3m2", usearch_scalar_e3m2_k}, {"e2m3", usearch_scalar_e2m3_k}, {"i8", usearch_scalar_i8_k},
    {"i4x2", usearch_scalar_i4x2_k}, {"u8", usearch_scalar_u8_k},     {"u4x2", usearch_scalar_u4x2_k},
    {"u1x8", usearch_scalar_u1x8_k},
};

/** @brief Where an index's payload lives, and which executor factory matches it. */
static usearch_js_choice_naming_t const usearch_js_residency_namings[] = {
    {"ram", usearch_residency_ram_k},
    {"vram", usearch_residency_vram_k},
    {"nvme", usearch_residency_nvme_k},
};

#pragma endregion Naming

#pragma region Error Reporting

bool usearch_js_throw(napi_env env, char const* code, char const* message) {
    napi_throw_error(env, code, message);
    return false;
}

bool usearch_js_throw_field(napi_env env, char const* field, char const* expectation) {
    char message[usearch_js_message_capacity_k];
    snprintf(message, sizeof(message), "The '%s' argument %s", field, expectation);
    return usearch_js_throw(env, "ERR_INVALID_ARG_TYPE", message);
}

bool usearch_js_succeeded(napi_env env, usearch_error_t const* error) {
    if (error->status == usearch_success_k) return true;
    return usearch_js_throw(env, usearch_status_name(error->status),
                            error->message ? error->message : "The operation failed");
}

#pragma endregion Error Reporting

#pragma region Argument Parsing

bool usearch_js_value_absent(napi_env env, napi_value value) {
    napi_valuetype kind = napi_undefined;
    if (!value || napi_typeof(env, value, &kind) != napi_ok) return true;
    return kind == napi_undefined || kind == napi_null;
}

bool usearch_js_double_from_value(napi_env env, napi_value value, char const* field, double* result) {
    *result = 0;
    if (usearch_js_value_absent(env, value)) return true;
    if (napi_get_value_double(env, value, result) != napi_ok)
        return usearch_js_throw_field(env, field, "must be a finite real");
    return true;
}

bool usearch_js_size_from_value(napi_env env, napi_value value, char const* field, size_t* result) {
    napi_valuetype kind = napi_undefined;
    if (napi_typeof(env, value, &kind) != napi_ok) return usearch_js_throw_field(env, field, "could not be inspected");
    if (kind == napi_undefined || kind == napi_null) {
        *result = 0;
        return true;
    }
    if (kind == napi_bigint) {
        uint64_t wide = 0;
        bool lossless = false;
        if (napi_get_value_bigint_uint64(env, value, &wide, &lossless) != napi_ok || !lossless)
            return usearch_js_throw_field(env, field, "must be a non-negative integer");
        *result = (size_t)wide;
        return true;
    }
    double number = 0;
    if (kind != napi_number || napi_get_value_double(env, value, &number) != napi_ok)
        return usearch_js_throw_field(env, field, "must be a number or a BigInt");
    if (!(number >= 0)) return usearch_js_throw_field(env, field, "must be a non-negative integer");
    *result = (size_t)number;
    return true;
}

bool usearch_js_string_from_value(napi_env env, napi_value value, char const* field, char* buffer, size_t capacity,
                                  size_t* length) {
    napi_valuetype kind = napi_undefined;
    if (napi_typeof(env, value, &kind) != napi_ok || kind != napi_string)
        return usearch_js_throw_field(env, field, "must be a string");
    size_t written = 0;
    if (napi_get_value_string_utf8(env, value, buffer, capacity, &written) != napi_ok)
        return usearch_js_throw_field(env, field, "must be a shorter string");
    *length = written;
    return true;
}

/**
 *  @brief Reads one of a closed set of names into the value it selects.
 *
 *  The value travels in the table rather than being re-derived from the match position, so
 *  reordering a table cannot silently change what a name means.
 */
static bool usearch_js_choice_from_value(napi_env env, napi_value value, char const* field,
                                         usearch_js_choice_naming_t const* namings, size_t count, int* result) {
    char name[usearch_js_name_capacity_k];
    size_t length = 0;
    if (!usearch_js_string_from_value(env, value, field, name, sizeof(name), &length)) return false;
    for (size_t naming_index = 0; naming_index != count; ++naming_index)
        if (strcmp(namings[naming_index].name, name) == 0) {
            *result = namings[naming_index].value;
            return true;
        }
    char message[usearch_js_message_capacity_k];
    int offset = snprintf(message, sizeof(message), "Unknown '%s': '%s'. Expected one of:", field, name);
    for (size_t naming_index = 0; naming_index != count && offset > 0 && (size_t)offset < sizeof(message);
         ++naming_index)
        offset += snprintf(message + offset, sizeof(message) - (size_t)offset, " %s", namings[naming_index].name);
    return usearch_js_throw(env, "ERR_INVALID_ARG_VALUE", message);
}

bool usearch_js_metric_from_value(napi_env env, napi_value value, char const* field, usearch_dense_metric_t* result) {
    int choice = usearch_dense_metric_angular_k;
    if (!usearch_js_choice_from_value(env, value, field, usearch_js_metric_namings,
                                      usearch_js_table_size_(usearch_js_metric_namings), &choice))
        return false;
    *result = (usearch_dense_metric_t)choice;
    return true;
}

bool usearch_js_scalar_from_value(napi_env env, napi_value value, char const* field, usearch_scalar_t* result) {
    napi_valuetype kind = napi_undefined;
    if (napi_typeof(env, value, &kind) != napi_ok) return usearch_js_throw_field(env, field, "could not be inspected");
    if (kind == napi_undefined || kind == napi_null) {
        *result = usearch_scalar_unknown_k;
        return true;
    }
    int choice = usearch_scalar_unknown_k;
    if (!usearch_js_choice_from_value(env, value, field, usearch_js_scalar_namings,
                                      usearch_js_table_size_(usearch_js_scalar_namings), &choice))
        return false;
    *result = (usearch_scalar_t)choice;
    return true;
}

bool usearch_js_residency_from_value(napi_env env, napi_value value, char const* field, usearch_residency_t* result) {
    int choice = usearch_residency_ram_k;
    if (!usearch_js_choice_from_value(env, value, field, usearch_js_residency_namings,
                                      usearch_js_table_size_(usearch_js_residency_namings), &choice))
        return false;
    *result = (usearch_residency_t)choice;
    return true;
}

bool usearch_js_ordinals_from_value(napi_env env, napi_value value, char const* field, int32_t** ordinals,
                                    size_t* count) {
    *ordinals = NULL;
    *count = 0;
    if (usearch_js_value_absent(env, value)) return true;

    bool is_array = false;
    uint32_t length = 0;
    if (napi_is_array(env, value, &is_array) != napi_ok || !is_array ||
        napi_get_array_length(env, value, &length) != napi_ok)
        return usearch_js_throw_field(env, field, "must be an array of integers");
    if (!length) return true;

    int32_t* const claimed = (int32_t*)malloc((size_t)length * sizeof(int32_t));
    if (!claimed) return usearch_js_throw(env, "out-of-memory", "Could not allocate the residency arm");
    for (uint32_t position = 0; position != length; ++position) {
        napi_value entry = NULL;
        if (napi_get_element(env, value, position, &entry) != napi_ok ||
            napi_get_value_int32(env, entry, &claimed[position]) != napi_ok) {
            free(claimed);
            return usearch_js_throw_field(env, field, "must hold integers");
        }
    }
    *ordinals = claimed;
    *count = length;
    return true;
}

bool usearch_js_paths_from_value(napi_env env, napi_value value, char const* field, usearch_js_paths_arena_t* arena) {
    arena->views = NULL;
    arena->count = 0;
    if (usearch_js_value_absent(env, value)) return true;

    bool is_array = false;
    uint32_t length = 0;
    if (napi_is_array(env, value, &is_array) != napi_ok || !is_array ||
        napi_get_array_length(env, value, &length) != napi_ok)
        return usearch_js_throw_field(env, field, "must be an array of strings");
    if (!length) return true;

    // One block: the views the ABI reads contiguously, then the bytes they point at.
    size_t bytes_total = 0;
    size_t* const lengths = (size_t*)malloc((size_t)length * sizeof(size_t));
    if (!lengths) return usearch_js_throw(env, "out-of-memory", "Could not measure the residency arm");
    for (uint32_t position = 0; position != length; ++position) {
        napi_value entry = NULL;
        size_t spelled = 0;
        if (napi_get_element(env, value, position, &entry) != napi_ok ||
            napi_get_value_string_utf8(env, entry, NULL, 0, &spelled) != napi_ok) {
            free(lengths);
            return usearch_js_throw_field(env, field, "must hold strings");
        }
        lengths[position] = spelled;
        bytes_total += spelled + 1; //! One terminator apiece, which the views never span.
    }

    size_t const views_bytes = (size_t)length * sizeof(usearch_bytes_view_t);
    arena->views = (usearch_bytes_view_t*)malloc(views_bytes + bytes_total);
    if (!arena->views) {
        free(lengths);
        return usearch_js_throw(env, "out-of-memory", "Could not allocate the residency arm");
    }

    char* spelling = (char*)arena->views + views_bytes;
    for (uint32_t position = 0; position != length; ++position) {
        napi_value entry = NULL;
        size_t written = 0;
        if (napi_get_element(env, value, position, &entry) != napi_ok ||
            napi_get_value_string_utf8(env, entry, spelling, lengths[position] + 1, &written) != napi_ok) {
            free(lengths);
            usearch_js_paths_release(arena);
            return usearch_js_throw_field(env, field, "must hold strings");
        }
        arena->views[position].data = (uint8_t const*)spelling;
        arena->views[position].length = written;
        spelling += written + 1;
    }
    free(lengths);
    arena->count = length;
    return true;
}

void usearch_js_paths_release(usearch_js_paths_arena_t* arena) {
    free(arena->views);
    arena->views = NULL;
    arena->count = 0;
}

size_t usearch_js_scalar_row_bytes(size_t columns, usearch_scalar_t scalar_kind) {
    switch (scalar_kind) {
    case usearch_scalar_f64_k: return columns * 8;
    case usearch_scalar_f32_k: return columns * 4;
    case usearch_scalar_bf16_k:
    case usearch_scalar_f16_k: return columns * 2;
    case usearch_scalar_e5m2_k:
    case usearch_scalar_e4m3_k:
    case usearch_scalar_e3m2_k:
    case usearch_scalar_e2m3_k:
    case usearch_scalar_i8_k:
    case usearch_scalar_u8_k: return columns;
    case usearch_scalar_i4x2_k:
    case usearch_scalar_u4x2_k: return (columns + 1) / 2;
    case usearch_scalar_u1x8_k: return (columns + 7) / 8;
    case usearch_scalar_unknown_k: return 0;
    }
    return 0;
}

/** @brief Bytes one element of a @c TypedArray occupies; zero for the types no scalar kind spells. */
static size_t usearch_js_typedarray_unit_bytes(napi_typedarray_type type) {
    switch (type) {
    case napi_int8_array:
    case napi_uint8_array:
    case napi_uint8_clamped_array: return 1;
    case napi_int16_array:
    case napi_uint16_array: return 2;
    case napi_int32_array:
    case napi_uint32_array:
    case napi_float32_array: return 4;
    case napi_float64_array:
    case napi_bigint64_array:
    case napi_biguint64_array: return 8;
    }
    return 0;
}

/** @brief The scalar kind a @c TypedArray spells on its own, before any declared kind overrides it. */
static usearch_scalar_t usearch_js_scalar_from_typedarray(napi_typedarray_type type) {
    switch (type) {
    case napi_float64_array: return usearch_scalar_f64_k;
    case napi_float32_array: return usearch_scalar_f32_k;
    case napi_int8_array: return usearch_scalar_i8_k;
    case napi_uint8_array:
    case napi_uint8_clamped_array: return usearch_scalar_u8_k;
    default: return usearch_scalar_unknown_k;
    }
}

bool usearch_js_matrix_from_value(napi_env env, napi_value value, char const* field, usearch_scalar_t declared,
                                  usearch_js_matrix_t* matrix) {
    bool is_typedarray = false;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray)
        return usearch_js_throw_field(env, field, "must be a numeric TypedArray");

    napi_typedarray_type type = napi_int8_array;
    napi_value arraybuffer = NULL;
    size_t offset = 0, elements = 0;
    void* data = NULL;
    if (napi_get_typedarray_info(env, value, &type, &elements, &data, &arraybuffer, &offset) != napi_ok)
        return usearch_js_throw_field(env, field, "could not be inspected");

    size_t const unit_bytes = usearch_js_typedarray_unit_bytes(type);
    if (!unit_bytes) return usearch_js_throw_field(env, field, "must be a numeric TypedArray");

    matrix->scalar_kind = declared != usearch_scalar_unknown_k ? declared : usearch_js_scalar_from_typedarray(type);
    if (matrix->scalar_kind == usearch_scalar_unknown_k)
        return usearch_js_throw_field(env, field,                                            //
                                      "must name its scalar kind, since a BigInt64Array, "   //
                                      "Int16Array, Int32Array, Uint16Array, or Uint32Array " //
                                      "spells none on its own");
    matrix->data = data;
    matrix->bytes = elements * unit_bytes;
    return true;
}

#pragma endregion Argument Parsing

#pragma region Identities

/** @brief Parks an integer identity in the trailing eight bytes, most significant first. */
static void usearch_js_key_from_integer(uint64_t integer, usearch_key_t* key) {
    memset(key->bytes, 0, sizeof(key->bytes));
    for (size_t position = 0; position != 8; ++position)
        key->bytes[15 - position] = (uint8_t)(integer >> (position * 8));
}

usearch_key_t* usearch_js_keys_from_value(napi_env env, napi_value value, char const* field, size_t* count) {
    napi_typedarray_type type = napi_int8_array;
    napi_value arraybuffer = NULL;
    size_t offset = 0, elements = 0;
    void* data = NULL;
    bool is_typedarray = false;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &type, &elements, &data, &arraybuffer, &offset) != napi_ok ||
        (type != napi_biguint64_array && type != napi_uint8_array)) {
        usearch_js_throw_field(env, field, "must be a Uint8Array of sixteen bytes per key, or a BigUint64Array");
        return NULL;
    }
    if (type == napi_uint8_array && elements % usearch_js_key_bytes_k != 0) {
        usearch_js_throw_field(env, field, "must span a whole number of sixteen-byte keys");
        return NULL;
    }

    size_t const keys_count = type == napi_uint8_array ? elements / usearch_js_key_bytes_k : elements;
    usearch_key_t* const keys = (usearch_key_t*)malloc((keys_count ? keys_count : 1) * sizeof(usearch_key_t));
    if (!keys) {
        usearch_js_throw(env, "out-of-memory", "Could not allocate the identity batch");
        return NULL;
    }
    if (type == napi_uint8_array) { memcpy(keys, data, keys_count * usearch_js_key_bytes_k); }
    else {
        uint64_t const* const integers = (uint64_t const*)data;
        for (size_t entry_index = 0; entry_index != keys_count; ++entry_index)
            usearch_js_key_from_integer(integers[entry_index], &keys[entry_index]);
    }
    *count = keys_count;
    return keys;
}

bool usearch_js_key_from_value(napi_env env, napi_value value, char const* field, usearch_key_t* key) {
    size_t count = 0;
    usearch_key_t* const keys = usearch_js_keys_from_value(env, value, field, &count);
    if (!keys) return false;
    bool const single = count == 1;
    if (single) *key = keys[0];
    free(keys);
    return single ? true : usearch_js_throw_field(env, field, "must name exactly one key");
}

napi_value usearch_js_buffer(napi_env env, size_t bytes, void** data) {
    napi_value buffer = NULL;
    if (napi_create_arraybuffer(env, bytes ? bytes : 1, data, &buffer) != napi_ok) {
        usearch_js_throw(env, "out-of-memory", "Could not allocate the output buffer");
        return NULL;
    }
    return buffer;
}

napi_value usearch_js_number(napi_env env, size_t value) {
    napi_value number = NULL;
    napi_create_double(env, (double)value, &number);
    return number;
}

napi_value usearch_js_static_string(napi_env env, char const* text) {
    napi_value string = NULL;
    napi_create_string_utf8(env, text ? text : "", NAPI_AUTO_LENGTH, &string);
    return string;
}

napi_value usearch_js_undefined(napi_env env) {
    napi_value undefined = NULL;
    napi_get_undefined(env, &undefined);
    return undefined;
}

#pragma endregion Identities

#pragma region Executor

static void usearch_js_executor_finalize(napi_env env, void* data, void* hint) {
    usearch_js_executor_t* const executor = (usearch_js_executor_t*)data;
    (void)env;
    (void)hint;
    if (!executor) return;
    usearch_executor_free(executor->handle);
    free(executor);
}

/**
 *  @brief @c CompiledExecutor(residency, threads, placement) - the workers a fanned-out verb runs on.
 *
 *  The residency picks the factory, and the factory decides what the other two mean: host
 *  workers over NUMA nodes for RAM, one submission ring apiece over device paths for NVMe, and
 *  for VRAM no thread count at all, a single host thread driving one stream per named device.
 *  The three public factories live in TypeScript, where each can take only its own arm.
 */
static napi_value usearch_js_executor_construct(napi_env env, napi_callback_info info) {
    size_t arguments_count = 3;
    napi_value arguments[3] = {NULL, NULL, NULL};
    napi_value self = NULL;
    if (napi_get_cb_info(env, info, &arguments_count, arguments, &self, NULL) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not read the CompiledExecutor arguments");
        return NULL;
    }
    if (arguments_count != 3) {
        usearch_js_throw(env, "ERR_INVALID_ARG_TYPE",
                         "CompiledExecutor expects 3 arguments: residency, threads, placement");
        return NULL;
    }

    size_t threads = 0;
    usearch_residency_t residency = usearch_residency_ram_k;
    if (!usearch_js_residency_from_value(env, arguments[0], "residency", &residency) ||
        !usearch_js_size_from_value(env, arguments[1], "threads", &threads))
        return NULL;

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_executor_t handle = NULL;
    if (residency == usearch_residency_nvme_k) {
        usearch_js_paths_arena_t paths;
        if (!usearch_js_paths_from_value(env, arguments[2], "paths", &paths)) return NULL;
        usearch_nvme_options_t const nvme = {paths.views, paths.count};
        handle = usearch_executor_for_nvme(&nvme, threads, &error);
        usearch_js_paths_release(&paths);
    }
    else {
        int32_t* ordinals = NULL;
        size_t ordinals_count = 0;
        char const* const field = residency == usearch_residency_vram_k ? "devices" : "memoryDomains";
        if (!usearch_js_ordinals_from_value(env, arguments[2], field, &ordinals, &ordinals_count)) return NULL;
        if (residency == usearch_residency_vram_k) {
            usearch_vram_options_t const vram = {ordinals, ordinals_count};
            handle = usearch_executor_for_vram(&vram, &error);
        }
        else {
            usearch_ram_options_t const ram = {ordinals, ordinals_count};
            handle = usearch_executor_for_ram(&ram, threads, &error);
        }
        free(ordinals);
    }
    if (!usearch_js_succeeded(env, &error)) return NULL;

    usearch_js_executor_t* const executor = (usearch_js_executor_t*)malloc(sizeof(usearch_js_executor_t));
    if (!executor) {
        usearch_executor_free(handle);
        usearch_js_throw(env, "out-of-memory", "Could not allocate the Executor");
        return NULL;
    }
    executor->handle = handle;
    if (napi_wrap(env, self, executor, usearch_js_executor_finalize, NULL, NULL) != napi_ok) {
        usearch_executor_free(handle);
        free(executor);
        usearch_js_throw(env, "internal", "Could not attach the Executor");
        return NULL;
    }
    return self;
}

/** @brief @c drain() - waits for every deferred call on this executor to finish. */
static napi_value usearch_js_executor_drain(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    if (napi_get_cb_info(env, info, NULL, NULL, &self, NULL) != napi_ok) return NULL;
    usearch_js_executor_t* executor = NULL;
    if (napi_unwrap(env, self, (void**)&executor) != napi_ok || !executor) {
        usearch_js_throw(env, "internal", "The Executor is not attached");
        return NULL;
    }
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_executor_drain(executor->handle, &error);
    if (!usearch_js_succeeded(env, &error)) return NULL;
    return usearch_js_undefined(env);
}

bool usearch_js_executor_from_value(napi_env env, napi_value value, usearch_executor_t* handle) {
    usearch_js_addon_t* addon = NULL;
    napi_value constructor = NULL;
    bool is_executor = false;
    if (napi_get_instance_data(env, (void**)&addon) != napi_ok || !addon ||
        napi_get_reference_value(env, addon->executor_class, &constructor) != napi_ok ||
        napi_instanceof(env, value, constructor, &is_executor) != napi_ok || !is_executor)
        return usearch_js_throw_field(env, "executor", "must be an Executor");
    usearch_js_executor_t* executor = NULL;
    if (napi_unwrap(env, value, (void**)&executor) != napi_ok || !executor)
        return usearch_js_throw_field(env, "executor", "must be an attached Executor");
    *handle = executor->handle;
    return true;
}

napi_value usearch_js_executor_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"drain", NULL, usearch_js_executor_drain, NULL, NULL, NULL, napi_default_method, NULL},
    };
    napi_value executor_class = NULL;
    if (napi_define_class(env, "Executor", NAPI_AUTO_LENGTH, usearch_js_executor_construct, NULL,
                          usearch_js_table_size_(properties), properties, &executor_class) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not define the Executor class");
        return NULL;
    }
    return executor_class;
}

#pragma endregion Executor

#pragma region Library Probes

napi_value usearch_js_version(napi_env env, napi_callback_info info) {
    (void)info;
    return usearch_js_static_string(env, usearch_version());
}

napi_value usearch_js_capabilities_compiled(napi_env env, napi_callback_info info) {
    (void)info;
    return usearch_js_static_string(env, usearch_capabilities_compiled());
}

napi_value usearch_js_capabilities_available(napi_env env, napi_callback_info info) {
    (void)info;
    return usearch_js_static_string(env, usearch_capabilities_available());
}

#pragma endregion Library Probes
