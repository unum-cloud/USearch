/**
 *  @file javascript/types.c
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief The Node-API vocabulary every modality shares.
 *
 *  Closed-set naming, argument parsing, identities, row borrowing, the @c Executor whose threads or
 *  streams every fanned-out verb runs on, the @c Arena every scaling output lands in, and the
 *  @c CompiledDeferred a deferred call answers.
 */
#include <stdio.h>  // `snprintf`
#include <stdlib.h> // `malloc`, `calloc`, `realloc`, `free`
#include <string.h> // `memset`, `strcmp`, `strlen`

#include "lib.h"

#pragma region Naming

/**
 *  @brief Reads one of a closed set of names through the C name table that spells it.
 *
 *  The C reader answers a default for a name it does not know, so a name counts as read only when
 *  the C writer spells the answer back the same way - which is what refuses a misspelling rather
 *  than reading it as the default.
 */
static bool usearch_js_name_from_value(napi_env env, napi_value value, char const* field, int (*parse)(char const*),
                                       char const* (*spell)(int), int* result) {
    char name[usearch_js_name_capacity_k];
    size_t length = 0;
    napi_valuetype kind = napi_undefined;
    if (napi_typeof(env, value, &kind) != napi_ok || kind != napi_string ||
        napi_get_value_string_utf8(env, value, name, sizeof(name), &length) != napi_ok)
        return usearch_js_throw_field(env, field, "must be a string");
    *result = parse(name);
    if (strcmp(spell(*result), name) == 0) return true;
    char message[usearch_js_message_capacity_k];
    snprintf(message, sizeof(message), "The '%s' argument names nothing: '%s'", field, name);
    return usearch_js_throw(env, usearch_status_invalid_argument_k, message);
}

static usearch_bytes_view_t usearch_js_name_view(char const* name) {
    return (usearch_bytes_view_t) {(uint8_t const*)name, strlen(name)};
}

static int usearch_js_metric_parse(char const* name) {
    return (int)usearch_dense_metric_from_name(usearch_js_name_view(name));
}

static char const* usearch_js_metric_spell(int metric) {
    return usearch_dense_metric_name((usearch_dense_metric_t)metric);
}

static int usearch_js_mood_parse(char const* name) { return (int)usearch_mood_from_name(usearch_js_name_view(name)); }

static char const* usearch_js_mood_spell(int mood) { return usearch_mood_name((usearch_mood_t)mood); }

static int usearch_js_traversal_parse(char const* name) {
    return (int)usearch_traversal_from_name(usearch_js_name_view(name));
}

static char const* usearch_js_traversal_spell(int traversal) {
    return usearch_traversal_name((usearch_traversal_t)traversal);
}

static int usearch_js_cpu_class_parse(char const* name) {
    return (int)usearch_cpu_class_from_name(usearch_js_name_view(name));
}

static char const* usearch_js_cpu_class_spell(int cpu_class) {
    return usearch_cpu_class_name((usearch_cpu_class_t)cpu_class);
}

static int usearch_js_gpu_class_parse(char const* name) {
    return (int)usearch_gpu_class_from_name(usearch_js_name_view(name));
}

static char const* usearch_js_gpu_class_spell(int gpu_class) {
    return usearch_gpu_class_name((usearch_gpu_class_t)gpu_class);
}

bool usearch_js_traversal_from_value(napi_env env, napi_value value, char const* field, usearch_traversal_t* result) {
    int traversal = usearch_traversal_walk_k;
    if (!usearch_js_value_absent(env, value) &&
        !usearch_js_name_from_value(env, value, field, usearch_js_traversal_parse, usearch_js_traversal_spell,
                                    &traversal))
        return false;
    *result = (usearch_traversal_t)traversal;
    return true;
}

bool usearch_js_cpu_class_from_value(napi_env env, napi_value value, usearch_cpu_class_t* result) {
    int cpu_class = usearch_cpu_class_unknown_k;
    if (!usearch_js_value_absent(env, value) &&
        !usearch_js_name_from_value(env, value, "cpuClass", usearch_js_cpu_class_parse, usearch_js_cpu_class_spell,
                                    &cpu_class))
        return false;
    *result = (usearch_cpu_class_t)cpu_class;
    return true;
}

bool usearch_js_gpu_class_from_value(napi_env env, napi_value value, usearch_gpu_class_t* result) {
    int gpu_class = usearch_gpu_class_unknown_k;
    if (!usearch_js_value_absent(env, value) &&
        !usearch_js_name_from_value(env, value, "gpuClass", usearch_js_gpu_class_parse, usearch_js_gpu_class_spell,
                                    &gpu_class))
        return false;
    *result = (usearch_gpu_class_t)gpu_class;
    return true;
}

bool usearch_js_metric_from_value(napi_env env, napi_value value, char const* field, usearch_dense_metric_t* result) {
    int metric = usearch_dense_metric_unknown_k;
    if (!usearch_js_name_from_value(env, value, field, usearch_js_metric_parse, usearch_js_metric_spell, &metric))
        return false;
    if (metric == usearch_dense_metric_unknown_k) return usearch_js_throw_field(env, field, "must name a metric");
    *result = (usearch_dense_metric_t)metric;
    return true;
}

bool usearch_js_scalar_from_value(napi_env env, napi_value value, char const* field, usearch_scalar_t* result) {
    uint32_t code = 0;
    if (napi_get_value_uint32(env, value, &code) != napi_ok || !usearch_js_row_bytes((usearch_scalar_t)code, 1))
        return usearch_js_throw_field(env, field, "must name a scalar kind");
    *result = (usearch_scalar_t)code;
    return true;
}

napi_value usearch_js_scalars(napi_env env) {
    static usearch_scalar_t const scalars[] = {
        usearch_scalar_f64_k,  usearch_scalar_f32_k,  usearch_scalar_bf16_k, usearch_scalar_f16_k,
        usearch_scalar_e5m2_k, usearch_scalar_e4m3_k, usearch_scalar_e3m2_k, usearch_scalar_e2m3_k,
        usearch_scalar_i8_k,   usearch_scalar_i4x2_k, usearch_scalar_i32_k,  usearch_scalar_u8_k,
        usearch_scalar_u4x2_k, usearch_scalar_u1x8_k,
    };
    napi_value table = NULL;
    bool mapped = napi_create_object(env, &table) == napi_ok;
    for (size_t position = 0; mapped && position != usearch_js_table_size_(scalars); ++position)
        mapped = napi_set_named_property(env, table, usearch_scalar_name(scalars[position]),
                                         usearch_js_number(env, scalars[position])) == napi_ok;
    if (!mapped || napi_object_freeze(env, table) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not map the scalar kinds");
        return NULL;
    }
    return table;
}

size_t usearch_js_row_bytes(usearch_scalar_t scalar, size_t columns) {
    switch (scalar) {
    case usearch_scalar_f64_k: return columns * 8;
    case usearch_scalar_f32_k:
    case usearch_scalar_i32_k: return columns * 4;
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
    default: return 0;
    }
}

#pragma endregion Naming

#pragma region Argument Parsing

bool usearch_js_throw(napi_env env, usearch_status_t status, char const* message) {
    napi_throw_error(env, usearch_status_name(status), message);
    return false;
}

bool usearch_js_throw_field(napi_env env, char const* field, char const* expectation) {
    char message[usearch_js_message_capacity_k];
    snprintf(message, sizeof(message), "The '%s' argument %s", field, expectation);
    return usearch_js_throw(env, usearch_status_invalid_argument_k, message);
}

bool usearch_js_succeeded(napi_env env, usearch_outcome_t const* outcome) {
    if (outcome->status == usearch_status_success_k) return true;
    return usearch_js_throw(env, outcome->status, outcome->message ? outcome->message : "The operation failed");
}

bool usearch_js_value_absent(napi_env env, napi_value value) {
    napi_valuetype kind = napi_undefined;
    if (!value || napi_typeof(env, value, &kind) != napi_ok) return true;
    return kind == napi_undefined || kind == napi_null;
}

napi_value usearch_js_property(napi_env env, napi_value object, char const* name) {
    napi_value property = NULL;
    napi_valuetype kind = napi_undefined;
    if (!object || napi_typeof(env, object, &kind) != napi_ok || kind != napi_object ||
        napi_get_named_property(env, object, name, &property) != napi_ok)
        return usearch_js_undefined(env);
    return property;
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
        if (napi_get_value_bigint_uint64(env, value, &wide, &lossless) != napi_ok || !lossless || wide > SIZE_MAX)
            return usearch_js_throw_field(env, field, "must be a non-negative integer");
        *result = (size_t)wide;
        return true;
    }
    double number = 0;
    if (kind != napi_number || napi_get_value_double(env, value, &number) != napi_ok)
        return usearch_js_throw_field(env, field, "must be a number or a BigInt");
    // The range check comes first: casting a negative, infinite or oversized double is undefined.
    if (!(number >= 0 && number < 18446744073709551616.0) || (double)(uint64_t)number != number ||
        (uint64_t)number > SIZE_MAX)
        return usearch_js_throw_field(env, field, "must be a non-negative integer");
    *result = (size_t)number;
    return true;
}

char* usearch_js_string_from_value(napi_env env, napi_value value, char const* field, usearch_bytes_view_t* view) {
    napi_valuetype kind = napi_undefined;
    size_t length = 0;
    if (napi_typeof(env, value, &kind) != napi_ok || kind != napi_string ||
        napi_get_value_string_utf8(env, value, NULL, 0, &length) != napi_ok) {
        usearch_js_throw_field(env, field, "must be a string");
        return NULL;
    }
    // Measured first, since a buffer too short is truncated silently rather than refused.
    char* const text = (char*)malloc(length + 1);
    if (!text) {
        usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not copy the string");
        return NULL;
    }
    if (napi_get_value_string_utf8(env, value, text, length + 1, &length) != napi_ok) {
        free(text);
        usearch_js_throw_field(env, field, "must be a string");
        return NULL;
    }
    view->data = (uint8_t const*)text;
    view->length = length;
    return text;
}

bool usearch_js_completion_from_value(napi_env env, napi_value value, usearch_completion_t* result) {
    size_t completion = 0;
    if (!usearch_js_size_from_value(env, value, "completion", &completion)) return false;
    if (completion != usearch_completion_blocking_k && completion != usearch_completion_deferred_k)
        return usearch_js_throw_field(env, "completion", "must be blocking or deferred");
    *result = (usearch_completion_t)completion;
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
    if (!claimed) return usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not allocate the residency arm");
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
    if (!lengths) return usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not measure the residency arm");
    for (uint32_t position = 0; position != length; ++position) {
        napi_value entry = NULL;
        size_t spelled = 0;
        if (napi_get_element(env, value, position, &entry) != napi_ok ||
            napi_get_value_string_utf8(env, entry, NULL, 0, &spelled) != napi_ok) {
            free(lengths);
            return usearch_js_throw_field(env, field, "must hold strings");
        }
        lengths[position] = spelled;
        bytes_total += spelled + 1; // One terminator apiece, which the views never span.
    }

    size_t const views_bytes = (size_t)length * sizeof(usearch_bytes_view_t);
    arena->views = (usearch_bytes_view_t*)malloc(views_bytes + bytes_total);
    if (!arena->views) {
        free(lengths);
        return usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not allocate the residency arm");
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

/** Bytes one element of a @c TypedArray occupies; zero for a type this binding does not know. */
static size_t usearch_js_typedarray_unit_bytes(napi_typedarray_type type) {
    switch (type) {
    case napi_int8_array:
    case napi_uint8_array:
    case napi_uint8_clamped_array: return 1;
#ifdef NODE_API_HAS_FLOAT16_ARRAY
    case napi_float16_array:
#endif
    case napi_int16_array:
    case napi_uint16_array: return 2;
    case napi_int32_array:
    case napi_uint32_array:
    case napi_float32_array: return 4;
    case napi_float64_array:
    case napi_bigint64_array:
    case napi_biguint64_array: return 8;
    default: return 0;
    }
}

usearch_scalar_t usearch_js_scalar_from_typedarray(napi_typedarray_type type) {
    switch (type) {
    case napi_float64_array: return usearch_scalar_f64_k;
    case napi_float32_array: return usearch_scalar_f32_k;
#ifdef NODE_API_HAS_FLOAT16_ARRAY
    case napi_float16_array: return usearch_scalar_f16_k;
#endif
    case napi_int8_array: return usearch_scalar_i8_k;
    default: return usearch_scalar_unknown_k;
    }
}

bool usearch_js_bytes_from_value(napi_env env, napi_value value, char const* field, void** data, size_t* bytes) {
    napi_typedarray_type type = napi_int8_array;
    napi_value arraybuffer = NULL;
    size_t offset = 0, elements = 0;
    bool is_typedarray = false;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &type, &elements, data, &arraybuffer, &offset) != napi_ok ||
        !usearch_js_typedarray_unit_bytes(type))
        return usearch_js_throw_field(env, field, "must be a numeric TypedArray");
    *bytes = elements * usearch_js_typedarray_unit_bytes(type);
    return true;
}

/** Whether @p rows rows of @p row_bytes, @p stride bytes apart, stay inside @p bytes. */
static bool usearch_js_rows_fit(size_t rows, size_t stride, size_t row_bytes, size_t bytes) {
    if (!rows) return true;
    if (row_bytes > bytes) return false;
    return !stride || rows - 1 <= (bytes - row_bytes) / stride;
}

bool usearch_js_rows_from_value(napi_env env, napi_value const* arguments, char const* field, size_t* count,
                                usearch_rows_view_t* view) {
    void* data = NULL;
    size_t bytes = 0;
    if (!usearch_js_bytes_from_value(env, arguments[0], field, &data, &bytes) ||
        !usearch_js_size_from_value(env, arguments[1], "rows", count) ||
        !usearch_js_size_from_value(env, arguments[2], "columns", &view->columns) ||
        !usearch_js_size_from_value(env, arguments[3], "stride", &view->stride) ||
        !usearch_js_scalar_from_value(env, arguments[4], "scalar", &view->scalar))
        return false;
    if (!usearch_js_rows_fit(*count, view->stride, usearch_js_row_bytes(view->scalar, view->columns), bytes))
        return usearch_js_throw(env, usearch_status_shape_mismatch_k, "The rows reach past the end of their array");
    view->data = data;
    return true;
}

#pragma endregion Argument Parsing

#pragma region Identities

/** Parks an integer identity in the trailing eight bytes, most significant first. */
static void usearch_js_key_from_integer(uint64_t integer, usearch_key_t* key) {
    memset(key->bytes, 0, sizeof(key->bytes));
    for (size_t position = 0; position != 8; ++position)
        key->bytes[15 - position] = (uint8_t)(integer >> (position * 8));
}

bool usearch_js_keys_from_value(napi_env env, napi_value value, char const* field, usearch_key_t const** keys,
                                size_t* count, usearch_key_t** gathered) {
    napi_typedarray_type type = napi_int8_array;
    napi_value arraybuffer = NULL;
    size_t offset = 0, elements = 0;
    void* data = NULL;
    bool is_typedarray = false;
    *gathered = NULL;
    if (napi_is_typedarray(env, value, &is_typedarray) != napi_ok || !is_typedarray ||
        napi_get_typedarray_info(env, value, &type, &elements, &data, &arraybuffer, &offset) != napi_ok ||
        (type != napi_biguint64_array && type != napi_uint8_array))
        return usearch_js_throw_field(env, field, "must be a Uint8Array of sixteen bytes per key, or a BigUint64Array");
    if (type == napi_uint8_array) {
        if (elements % usearch_js_key_bytes_k != 0)
            return usearch_js_throw_field(env, field, "must span a whole number of sixteen-byte keys");
        *keys = (usearch_key_t const*)data;
        *count = elements / usearch_js_key_bytes_k;
        return true;
    }
    usearch_key_t* const integers = (usearch_key_t*)malloc((elements ? elements : 1) * sizeof(usearch_key_t));
    if (!integers) return usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not gather the identities");
    for (size_t entry_index = 0; entry_index != elements; ++entry_index)
        usearch_js_key_from_integer(((uint64_t const*)data)[entry_index], &integers[entry_index]);
    *keys = integers;
    *count = elements;
    *gathered = integers;
    return true;
}

bool usearch_js_key_from_value(napi_env env, napi_value value, char const* field, usearch_key_t* key) {
    usearch_key_t const* keys = NULL;
    usearch_key_t* gathered = NULL;
    size_t count = 0;
    if (!usearch_js_keys_from_value(env, value, field, &keys, &count, &gathered)) return false;
    bool const single = count == 1;
    if (single) *key = keys[0];
    free(gathered);
    return single ? true : usearch_js_throw_field(env, field, "must name exactly one key");
}

napi_value usearch_js_buffer(napi_env env, size_t bytes, void** data) {
    napi_value buffer = NULL;
    if (napi_create_arraybuffer(env, bytes ? bytes : 1, data, &buffer) != napi_ok) {
        usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not allocate the output buffer");
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

#pragma region Classes

/** The addon state holding every class constructor; @c NULL with an exception pending. */
static usearch_js_addon_t* usearch_js_addon(napi_env env) {
    usearch_js_addon_t* addon = NULL;
    if (napi_get_instance_data(env, (void**)&addon) != napi_ok || !addon) {
        usearch_js_throw(env, usearch_status_internal_k, "The addon state is missing");
        return NULL;
    }
    return addon;
}

bool usearch_js_sealed(napi_env env, napi_callback_info info, napi_value* self) {
    size_t arguments_count = 1;
    napi_value argument = NULL, seal = NULL;
    bool equal = false;
    usearch_js_addon_t* const addon = usearch_js_addon(env);
    return addon && napi_get_cb_info(env, info, &arguments_count, &argument, self, NULL) == napi_ok &&
           arguments_count == 1 && napi_get_reference_value(env, addon->seal, &seal) == napi_ok &&
           napi_strict_equals(env, argument, seal, &equal) == napi_ok && equal;
}

napi_value usearch_js_instantiate(napi_env env, napi_ref constructor) {
    napi_value constructor_value = NULL, seal = NULL, instance = NULL;
    usearch_js_addon_t* const addon = usearch_js_addon(env);
    if (!addon || napi_get_reference_value(env, constructor, &constructor_value) != napi_ok ||
        napi_get_reference_value(env, addon->seal, &seal) != napi_ok ||
        napi_new_instance(env, constructor_value, 1, &seal, &instance) != napi_ok)
        return NULL;
    return instance;
}

bool usearch_js_disposable(napi_env env, napi_value holder, char const* method) {
    napi_value global = NULL, symbols = NULL, dispose = NULL, function = NULL;
    if (napi_get_global(env, &global) != napi_ok ||
        napi_get_named_property(env, global, "Symbol", &symbols) != napi_ok ||
        napi_get_named_property(env, symbols, "dispose", &dispose) != napi_ok)
        return false;
    // A runtime without explicit resource management has no symbol to key the method by.
    if (usearch_js_value_absent(env, dispose)) return true;
    return napi_get_named_property(env, holder, method, &function) == napi_ok &&
           napi_set_property(env, holder, dispose, function) == napi_ok;
}

/** Gives the instances of @p constructor a @c [Symbol.dispose] calling their @c free. */
static bool usearch_js_class_disposable(napi_env env, napi_value constructor) {
    napi_value prototype = NULL;
    return napi_get_named_property(env, constructor, "prototype", &prototype) == napi_ok &&
           usearch_js_disposable(env, prototype, "free");
}

void* usearch_js_unwrap_instance(napi_env env, napi_value value, napi_ref reference, char const* field,
                                 char const* expectation) {
    napi_value constructor = NULL;
    bool is_instance = false;
    void* wrapped = NULL;
    if (napi_get_reference_value(env, reference, &constructor) != napi_ok ||
        napi_instanceof(env, value, constructor, &is_instance) != napi_ok || !is_instance ||
        napi_unwrap(env, value, &wrapped) != napi_ok || !wrapped) {
        usearch_js_throw_field(env, field, expectation);
        return NULL;
    }
    return wrapped;
}

/** Reads the native state behind the receiver of a method. */
static void* usearch_js_self(napi_env env, napi_callback_info info, size_t* arguments_count, napi_value* arguments,
                             napi_value* self) {
    napi_value receiver = NULL;
    void* wrapped = NULL;
    if (napi_get_cb_info(env, info, arguments_count, arguments, &receiver, NULL) != napi_ok ||
        napi_unwrap(env, receiver, &wrapped) != napi_ok || !wrapped) {
        usearch_js_throw(env, usearch_status_internal_k, "The receiver is not attached");
        return NULL;
    }
    if (self) *self = receiver;
    return wrapped;
}

#pragma endregion Classes

#pragma region Executor

static void usearch_js_executor_finalize(napi_env env, void* data, void* hint) {
    usearch_js_executor_t* const executor = (usearch_js_executor_t*)data;
    (void)env;
    (void)hint;
    if (!executor) return;
    usearch_executor_free(executor->handle);
    free(executor);
}

/** @c Executor() - refused unless one of the four factories calls it with the addon's seal. */
static napi_value usearch_js_executor_construct(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    if (!usearch_js_sealed(env, info, &self)) {
        usearch_js_throw(
            env, usearch_status_invalid_argument_k,
            "An Executor is made by Executor.forRam, Executor.onPool, Executor.forVram or Executor.forNvme");
        return NULL;
    }
    return self;
}

/** Wraps the handle a factory just made, or throws its outcome; frees the handle on failure. */
static napi_value usearch_js_executor_adopt(napi_env env, usearch_outcome_t const* outcome, usearch_executor_t handle) {
    if (!usearch_js_succeeded(env, outcome)) return NULL;
    usearch_js_addon_t* const addon = usearch_js_addon(env);
    napi_value const instance = addon ? usearch_js_instantiate(env, addon->executor_class) : NULL;
    usearch_js_executor_t* const executor = (usearch_js_executor_t*)calloc(1, sizeof(usearch_js_executor_t));
    if (!instance || !executor ||
        napi_wrap(env, instance, executor, usearch_js_executor_finalize, NULL, NULL) != napi_ok) {
        free(executor);
        usearch_executor_free(handle);
        usearch_js_throw(env, usearch_status_internal_k, "Could not instantiate the Executor");
        return NULL;
    }
    executor->handle = handle;
    return instance;
}

/** Reads a foreign pointer, crossing as the @c BigInt of its address; absent reads as @c NULL. */
static bool usearch_js_pointer_from_value(napi_env env, napi_value value, char const* field, void** pointer) {
    *pointer = NULL;
    if (usearch_js_value_absent(env, value)) return true;
    uint64_t address = 0;
    bool lossless = false;
    if (napi_get_value_bigint_uint64(env, value, &address, &lossless) != napi_ok || !lossless)
        return usearch_js_throw_field(env, field, "must be a BigInt address");
    *pointer = (void*)(uintptr_t)address;
    return true;
}

/** Reads up to @p wanted arguments of a static function; missing ones read as @c undefined. */
static bool usearch_js_arguments(napi_env env, napi_callback_info info, size_t wanted, napi_value* arguments) {
    size_t arguments_count = wanted;
    if (napi_get_cb_info(env, info, &arguments_count, arguments, NULL, NULL) != napi_ok)
        return usearch_js_throw(env, usearch_status_internal_k, "Could not read the arguments");
    for (size_t position = arguments_count; position < wanted; ++position)
        arguments[position] = usearch_js_undefined(env);
    return true;
}

/** @c forRam({memoryDomains, threads, deferredLimit, cpuClass}) → host workers over NUMA nodes. */
static napi_value usearch_js_executor_for_ram(napi_env env, napi_callback_info info) {
    napi_value config = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t handle = NULL;
    usearch_executor_for_ram_t for_ram = {.executor = &handle, .outcome = &outcome};
    int32_t* ordinals = NULL;
    if (!usearch_js_arguments(env, info, 1, &config) ||
        !usearch_js_size_from_value(env, usearch_js_property(env, config, "threads"), "threads", &for_ram.threads) ||
        !usearch_js_size_from_value(env, usearch_js_property(env, config, "deferredLimit"), "deferredLimit",
                                    &for_ram.deferred_limit) ||
        !usearch_js_cpu_class_from_value(env, usearch_js_property(env, config, "cpuClass"), &for_ram.ram.cpu_class) ||
        !usearch_js_ordinals_from_value(env, usearch_js_property(env, config, "memoryDomains"), "memoryDomains",
                                        &ordinals, &for_ram.ram.memory_domains_count))
        return NULL;
    for_ram.ram.memory_domains = ordinals;
    usearch_executor_for_ram(&for_ram);
    free(ordinals);
    return usearch_js_executor_adopt(env, &outcome, handle);
}

/** @c onPool(pool, {memoryDomains, deferredLimit, cpuClass}) → an executor borrowing a ForkUnion
 *  pool. */
static napi_value usearch_js_executor_on_pool(napi_env env, napi_callback_info info) {
    napi_value arguments[2] = {NULL, NULL};
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t handle = NULL;
    usearch_executor_on_pool_t on_pool = {.executor = &handle, .outcome = &outcome};
    int32_t* ordinals = NULL;
    if (!usearch_js_arguments(env, info, 2, arguments) ||
        !usearch_js_pointer_from_value(env, arguments[0], "pool", &on_pool.pool) ||
        !usearch_js_size_from_value(env, usearch_js_property(env, arguments[1], "deferredLimit"), "deferredLimit",
                                    &on_pool.deferred_limit) ||
        !usearch_js_cpu_class_from_value(env, usearch_js_property(env, arguments[1], "cpuClass"),
                                         &on_pool.ram.cpu_class) ||
        !usearch_js_ordinals_from_value(env, usearch_js_property(env, arguments[1], "memoryDomains"), "memoryDomains",
                                        &ordinals, &on_pool.ram.memory_domains_count))
        return NULL;
    on_pool.ram.memory_domains = ordinals;
    usearch_executor_on_pool(&on_pool);
    free(ordinals);
    return usearch_js_executor_adopt(env, &outcome, handle);
}

/** @c forVram({devices, streams, scratchLimit, deferredLimit, gpuClass}) → one stream per
 *  device. */
static napi_value usearch_js_executor_for_vram(napi_env env, napi_callback_info info) {
    napi_value config = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t handle = NULL;
    usearch_executor_for_vram_t for_vram = {.executor = &handle, .outcome = &outcome};
    int32_t* ordinals = NULL;
    if (!usearch_js_arguments(env, info, 1, &config) ||
        !usearch_js_size_from_value(env, usearch_js_property(env, config, "scratchLimit"), "scratchLimit",
                                    &for_vram.scratch_limit) ||
        !usearch_js_size_from_value(env, usearch_js_property(env, config, "deferredLimit"), "deferredLimit",
                                    &for_vram.deferred_limit) ||
        !usearch_js_gpu_class_from_value(env, usearch_js_property(env, config, "gpuClass"), &for_vram.vram.gpu_class) ||
        !usearch_js_ordinals_from_value(env, usearch_js_property(env, config, "devices"), "devices", &ordinals,
                                        &for_vram.vram.devices_count))
        return NULL;
    for_vram.vram.devices = ordinals;

    // The engine reads one stream per device, a length only this side can check.
    napi_value const streams_value = usearch_js_property(env, config, "streams");
    void** streams = NULL;
    uint32_t streams_count = 0;
    if (!usearch_js_value_absent(env, streams_value)) {
        bool is_array = false;
        if (napi_is_array(env, streams_value, &is_array) != napi_ok || !is_array ||
            napi_get_array_length(env, streams_value, &streams_count) != napi_ok) {
            free(ordinals);
            usearch_js_throw_field(env, "streams", "must be an array of BigInt addresses");
            return NULL;
        }
        if (streams_count != for_vram.vram.devices_count) {
            free(ordinals);
            usearch_js_throw(env, usearch_status_shape_mismatch_k, "The streams must name one per device");
            return NULL;
        }
        streams = (void**)calloc(streams_count ? streams_count : 1, sizeof(void*));
        if (!streams) {
            free(ordinals);
            usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not allocate the streams");
            return NULL;
        }
        for (uint32_t position = 0; position != streams_count; ++position) {
            napi_value entry = NULL;
            if (napi_get_element(env, streams_value, position, &entry) != napi_ok ||
                !usearch_js_pointer_from_value(env, entry, "streams", &streams[position])) {
                free(streams);
                free(ordinals);
                return NULL;
            }
        }
        for_vram.streams = (void* const*)streams;
    }
    usearch_executor_for_vram(&for_vram);
    free(streams);
    free(ordinals);
    return usearch_js_executor_adopt(env, &outcome, handle);
}

/** @c forNvme({paths, threads, deferredLimit, cpuClass}) → one submission ring per worker. */
static napi_value usearch_js_executor_for_nvme(napi_env env, napi_callback_info info) {
    napi_value config = NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t handle = NULL;
    usearch_executor_for_nvme_t for_nvme = {.executor = &handle, .outcome = &outcome};
    usearch_js_paths_arena_t paths = {NULL, 0};
    if (!usearch_js_arguments(env, info, 1, &config) ||
        !usearch_js_size_from_value(env, usearch_js_property(env, config, "threads"), "threads", &for_nvme.threads) ||
        !usearch_js_size_from_value(env, usearch_js_property(env, config, "deferredLimit"), "deferredLimit",
                                    &for_nvme.deferred_limit) ||
        !usearch_js_cpu_class_from_value(env, usearch_js_property(env, config, "cpuClass"), &for_nvme.nvme.cpu_class) ||
        !usearch_js_paths_from_value(env, usearch_js_property(env, config, "paths"), "paths", &paths))
        return NULL;
    for_nvme.nvme = (usearch_nvme_options_t) {paths.views, paths.count, for_nvme.nvme.cpu_class};
    usearch_executor_for_nvme(&for_nvme);
    usearch_js_paths_release(&paths);
    return usearch_js_executor_adopt(env, &outcome, handle);
}

/** @c drain(mood) - waits for every call submitted so far, then idles in @p mood; absent parks. */
static napi_value usearch_js_executor_drain(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value argument = NULL;
    usearch_js_executor_t* const executor = (usearch_js_executor_t*)usearch_js_self(env, info, &arguments_count,
                                                                                    &argument, NULL);
    if (!executor) return NULL;
    int mood = usearch_mood_parked_k;
    if (arguments_count && !usearch_js_value_absent(env, argument) &&
        !usearch_js_name_from_value(env, argument, "mood", usearch_js_mood_parse, usearch_js_mood_spell, &mood))
        return NULL;
    if (!executor->handle) {
        usearch_js_throw(env, usearch_status_uninitialized_k, "The Executor was already freed");
        return NULL;
    }
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_drain_t drain = {.executor = executor->handle, .outcome = &outcome, .mood = (usearch_mood_t)mood};
    usearch_executor_drain(&drain);
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    return usearch_js_undefined(env);
}

/** @c free() - runs every deferred call and joins the workers now; a second call is a no-op. */
static napi_value usearch_js_executor_release(napi_env env, napi_callback_info info) {
    usearch_js_executor_t* const executor = (usearch_js_executor_t*)usearch_js_self(env, info, NULL, NULL, NULL);
    if (!executor) return NULL;
    if (executor->arenas) {
        usearch_js_throw(env, usearch_status_contended_k, "Free the GPU arenas made on the Executor before it");
        return NULL;
    }
    usearch_executor_free(executor->handle);
    executor->handle = NULL;
    return usearch_js_undefined(env);
}

/** Unwraps an @c Executor that was not freed yet; @c NULL with an exception pending. */
static usearch_js_executor_t* usearch_js_executor_live(napi_env env, napi_value value) {
    usearch_js_addon_t* const addon = usearch_js_addon(env);
    usearch_js_executor_t* const executor = addon ? (usearch_js_executor_t*)usearch_js_unwrap_instance(
                                                        env, value, addon->executor_class, "executor",
                                                        "must be an Executor")
                                                  : NULL;
    // A freed executor would otherwise read as NULL, silently running on the caller's thread.
    if (executor && !executor->handle) {
        usearch_js_throw(env, usearch_status_uninitialized_k, "The Executor was already freed");
        return NULL;
    }
    return executor;
}

bool usearch_js_executor_from_value(napi_env env, napi_value value, usearch_executor_t* handle) {
    *handle = NULL;
    if (usearch_js_value_absent(env, value)) return true;
    usearch_js_executor_t const* const executor = usearch_js_executor_live(env, value);
    if (!executor) return false;
    *handle = executor->handle;
    return true;
}

napi_value usearch_js_executor_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"drain", NULL, usearch_js_executor_drain, NULL, NULL, NULL, napi_default_method, NULL},
        {"free", NULL, usearch_js_executor_release, NULL, NULL, NULL, napi_default_method, NULL},
        {"forRam", NULL, usearch_js_executor_for_ram, NULL, NULL, NULL, napi_static, NULL},
        {"forVram", NULL, usearch_js_executor_for_vram, NULL, NULL, NULL, napi_static, NULL},
        {"forNvme", NULL, usearch_js_executor_for_nvme, NULL, NULL, NULL, napi_static, NULL},
        {"onPool", NULL, usearch_js_executor_on_pool, NULL, NULL, NULL, napi_static, NULL},
    };
    napi_value executor_class = NULL;
    if (napi_define_class(env, "Executor", NAPI_AUTO_LENGTH, usearch_js_executor_construct, NULL,
                          usearch_js_table_size_(properties), properties, &executor_class) != napi_ok ||
        !usearch_js_class_disposable(env, executor_class)) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not define the Executor class");
        return NULL;
    }
    return executor_class;
}

#pragma endregion Executor

#pragma region Arena

/** Forgets every view handed out so far, detaching the ones still alive when @p detach asks. */
static void usearch_js_arena_forget(napi_env env, usearch_js_arena_t* arena,
                                    napi_status (*detach)(napi_env, napi_value)) {
    for (size_t view_index = 0; view_index != arena->views_count; ++view_index) {
        napi_value buffer = NULL;
        if (detach && napi_get_reference_value(env, arena->views[view_index], &buffer) == napi_ok && buffer)
            detach(env, buffer);
        napi_delete_reference(env, arena->views[view_index]);
    }
    arena->views_count = 0;
}

/** Lets go of the executor a GPU arena held, once the arena's memory is freed. */
static void usearch_js_arena_unbind(napi_env env, usearch_js_arena_t* arena) {
    if (!arena->executor) return;
    --arena->executor->arenas;
    napi_delete_reference(env, arena->executor_reference);
    arena->executor = NULL;
}

static void usearch_js_arena_finalize(napi_env env, void* data, void* hint) {
    usearch_js_arena_t* const arena = (usearch_js_arena_t*)data;
    (void)hint;
    if (!arena) return;
    usearch_js_arena_forget(env, arena, NULL);
    free(arena->views);
    usearch_arena_free(arena->handle);
    usearch_js_arena_unbind(env, arena);
    free(arena);
}

/** @c Arena(executor) - empty memory where @p executor computes; host pages for the calling thread
 *  when absent. */
static napi_value usearch_js_arena_construct(napi_env env, napi_callback_info info) {
    size_t arguments_count = 1;
    napi_value self = NULL, argument = NULL;
    if (napi_get_cb_info(env, info, &arguments_count, &argument, &self, NULL) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not read the Arena arguments");
        return NULL;
    }
    usearch_js_executor_t* executor = NULL;
    if (arguments_count && !usearch_js_value_absent(env, argument) &&
        !(executor = usearch_js_executor_live(env, argument)))
        return NULL;
    usearch_js_arena_t* const arena = (usearch_js_arena_t*)calloc(1, sizeof(usearch_js_arena_t));
    if (!arena) {
        usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not allocate the Arena");
        return NULL;
    }
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_arena_init_t init = {.arena = &arena->handle,
                                 .executor = executor ? executor->handle : NULL,
                                 .outcome = &outcome};
    usearch_arena_init(&init);
    if (!usearch_js_succeeded(env, &outcome)) {
        free(arena);
        return NULL;
    }
    usearch_arena_description_t description = {usearch_residency_ram_k, 0};
    usearch_arena_describe_t describe = {.arena = arena->handle, .outcome = &outcome, .description = &description};
    usearch_arena_describe(&describe);
    if (!usearch_js_succeeded(env, &outcome)) {
        usearch_arena_free(arena->handle);
        free(arena);
        return NULL;
    }
    // A GPU arena frees on its executor's first stream, so it holds that executor until freed.
    if (description.residency == usearch_residency_vram_k) {
        if (napi_create_reference(env, argument, 1, &arena->executor_reference) != napi_ok) {
            usearch_arena_free(arena->handle);
            free(arena);
            usearch_js_throw(env, usearch_status_internal_k, "Could not hold the Executor");
            return NULL;
        }
        arena->executor = executor;
        ++executor->arenas;
    }
    if (napi_wrap(env, self, arena, usearch_js_arena_finalize, NULL, NULL) != napi_ok) {
        usearch_arena_free(arena->handle);
        usearch_js_arena_unbind(env, arena);
        free(arena);
        usearch_js_throw(env, usearch_status_internal_k, "Could not attach the Arena");
        return NULL;
    }
    return self;
}

/** @c describe() → @c {residency, device}: the canonical residency name and the device ordinal,
 *  @c null for host pages. */
static napi_value usearch_js_arena_describe(napi_env env, napi_callback_info info) {
    usearch_js_arena_t const* const arena = (usearch_js_arena_t const*)usearch_js_self(env, info, NULL, NULL, NULL);
    if (!arena) return NULL;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_arena_description_t description = {usearch_residency_ram_k, 0};
    usearch_arena_describe_t describe = {.arena = arena->handle, .outcome = &outcome, .description = &description};
    usearch_arena_describe(&describe);
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    napi_value object = NULL, device = NULL;
    if (napi_create_object(env, &object) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not publish the arena description");
        return NULL;
    }
    if (description.residency != usearch_residency_vram_k) napi_get_null(env, &device);
    else device = usearch_js_number(env, description.device);
    napi_set_named_property(env, object, "residency",
                            usearch_js_static_string(env, usearch_residency_name(description.residency)));
    napi_set_named_property(env, object, "device", device);
    return object;
}

/** @c free() - releases the memory now, once every deferred call writing it is final, detaching
 *  every view still held into it; a second call is a no-op. */
static napi_value usearch_js_arena_release(napi_env env, napi_callback_info info) {
    usearch_js_arena_t* const arena = (usearch_js_arena_t*)usearch_js_self(env, info, NULL, NULL, NULL);
    if (!arena) return NULL;
    if (!arena->handle) return usearch_js_undefined(env);
    usearch_js_arena_forget(env, arena, napi_detach_arraybuffer);
    usearch_arena_free(arena->handle);
    usearch_js_arena_unbind(env, arena);
    arena->handle = NULL;
    arena->holder = NULL;
    return usearch_js_undefined(env);
}

bool usearch_js_arena_from_value(napi_env env, napi_value value, usearch_js_call_t* call, usearch_arena_t* handle) {
    usearch_js_addon_t* const addon = usearch_js_addon(env);
    if (!addon) return false;
    usearch_js_arena_t* const arena = (usearch_js_arena_t*)usearch_js_unwrap_instance(env, value, addon->arena_class,
                                                                                      "arena", "must be an Arena");
    if (!arena) return false;
    if (!arena->handle) return usearch_js_throw(env, usearch_status_uninitialized_k, "The Arena was already freed");
    call->arena = arena;
    *handle = arena->handle;
    return true;
}

/** Views @p buffer as @p elements of @p type; half-precision rows read as their bits on a runtime
 *  without @c Float16Array, which a build against newer headers can still be loaded into. */
static napi_status usearch_js_typedarray(napi_env env, napi_typedarray_type type, size_t elements, napi_value buffer,
                                         napi_value* array) {
    napi_status const created = napi_create_typedarray(env, type, elements, buffer, 0, array);
#ifdef NODE_API_HAS_FLOAT16_ARRAY
    if (created != napi_ok && type == napi_float16_array)
        return napi_create_typedarray(env, napi_uint16_array, elements, buffer, 0, array);
#endif
    return created;
}

napi_value usearch_js_arena_view(napi_env env, napi_value arena_value, napi_typedarray_type type, void* data,
                                 size_t bytes, size_t elements) {
    napi_value buffer = NULL, array = NULL;
    usearch_js_arena_t* arena = NULL;
    if (napi_unwrap(env, arena_value, (void**)&arena) != napi_ok || !arena) {
        usearch_js_throw(env, usearch_status_internal_k, "The Arena is not attached");
        return NULL;
    }
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_arena_description_t description = {usearch_residency_ram_k, 0};
    usearch_arena_describe_t describe = {.arena = arena->handle, .outcome = &outcome, .description = &description};
    usearch_arena_describe(&describe);
    if (!usearch_js_succeeded(env, &outcome)) return NULL;
    if (description.residency == usearch_residency_vram_k) {
        napi_get_null(env, &array);
        return array;
    }
    if (arena->views_count == arena->views_capacity) {
        size_t const capacity = arena->views_capacity ? arena->views_capacity * 2 : 4;
        napi_ref* const views = (napi_ref*)realloc(arena->views, capacity * sizeof(napi_ref));
        if (!views) {
            usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not track the arena view");
            return NULL;
        }
        arena->views = views;
        arena->views_capacity = capacity;
    }
    // The non-enumerable `arena` property is what keeps the memory alive while any view of it is.
    napi_property_descriptor const owner = {"arena", NULL, NULL, NULL, NULL, arena_value, napi_default, NULL};
    // An empty result owns a buffer of its own, tracked alike, so it goes stale with the rest.
    bool const empty = !data || !bytes;
    napi_status const created = empty ? napi_create_arraybuffer(env, 0, NULL, &buffer)
                                      : napi_create_external_arraybuffer(env, data, bytes, NULL, NULL, &buffer);
    if (created != napi_ok || napi_define_properties(env, buffer, 1, &owner) != napi_ok ||
        usearch_js_typedarray(env, type, empty ? 0 : elements, buffer, &array) != napi_ok ||
        napi_create_reference(env, buffer, 0, &arena->views[arena->views_count]) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not view the arena");
        return NULL;
    }
    ++arena->views_count;
    return array;
}

napi_value usearch_js_arena_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"describe", NULL, usearch_js_arena_describe, NULL, NULL, NULL, napi_default_method, NULL},
        {"free", NULL, usearch_js_arena_release, NULL, NULL, NULL, napi_default_method, NULL},
    };
    napi_value arena_class = NULL;
    if (napi_define_class(env, "Arena", NAPI_AUTO_LENGTH, usearch_js_arena_construct, NULL,
                          usearch_js_table_size_(properties), properties, &arena_class) != napi_ok ||
        !usearch_js_class_disposable(env, arena_class)) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not define the Arena class");
        return NULL;
    }
    return arena_class;
}

#pragma endregion Arena

#pragma region Deferred Calls and Library Probes

/** Drops what a final call holds and owns, keeping its outcome readable; safe to repeat. */
static void usearch_js_call_release(napi_env env, usearch_js_call_t* call) {
    if (call->conclude) call->conclude(call);
    call->conclude = NULL;
    if (call->failure) napi_delete_reference(env, call->failure);
    call->failure = NULL;
    for (size_t slot = 0; slot != usearch_js_slots_k; ++slot) {
        if (call->held[slot]) napi_delete_reference(env, call->held[slot]);
        call->held[slot] = NULL;
    }
    for (size_t block = 0; block != usearch_js_owned_k; ++block) {
        free(call->owned[block]);
        call->owned[block] = NULL;
    }
}

/**
 *  @brief Frees a deferred call once its wrapper is collected.
 *
 *  The wrapper stays reachable until its call is final - the TypeScript @c Deferred re-registers it
 *  with the collector until then - so an unfinished call here is a contract breach, and it is left
 *  to leak rather than be freed under the library's feet.
 */
static void usearch_js_deferred_finalize(napi_env env, void* data, void* hint) {
    usearch_js_call_t* const call = (usearch_js_call_t*)data;
    (void)hint;
    if (!call || usearch_outcome_status(&call->outcome) == usearch_status_deferred_k) return;
    usearch_js_call_release(env, call);
    free(call);
}

/** @c CompiledDeferred() - refused unless a deferred submission calls it with the addon's seal. */
static napi_value usearch_js_deferred_construct(napi_env env, napi_callback_info info) {
    napi_value self = NULL;
    if (!usearch_js_sealed(env, info, &self)) {
        usearch_js_throw(env, usearch_status_invalid_argument_k, "A deferred call is made by a deferred receiver");
        return NULL;
    }
    return self;
}

/** @c status() → the canonical name of the call's status, @c deferred until it is final. */
static napi_value usearch_js_deferred_status(napi_env env, napi_callback_info info) {
    usearch_js_call_t* const call = (usearch_js_call_t*)usearch_js_self(env, info, NULL, NULL, NULL);
    if (!call) return NULL;
    return usearch_js_static_string(env, usearch_status_name(usearch_outcome_status(&call->outcome)));
}

/** An outcome wait handed to the libuv pool, which a promise resolves once it returns. */
typedef struct usearch_js_settling_t {

    /** The work item the pool runs. */
    napi_async_work work;

    /** The promise the loop resolves once the wait returns. */
    napi_deferred promise;

    /** A strong reference keeping the call alive through the wait. */
    napi_ref wrapper;

    /** The outcome the pool waits on. */
    usearch_outcome_t* outcome;
} usearch_js_settling_t;

static void usearch_js_settling_execute(napi_env env, void* data) {
    (void)env;
    usearch_outcome_wait(((usearch_js_settling_t*)data)->outcome);
}

static void usearch_js_settling_complete(napi_env env, napi_status status, void* data) {
    usearch_js_settling_t* const settling = (usearch_js_settling_t*)data;
    (void)status;
    napi_resolve_deferred(env, settling->promise, usearch_js_undefined(env));
    napi_delete_reference(env, settling->wrapper);
    napi_delete_async_work(env, settling->work);
    free(settling);
}

/** @c wait() - blocks the calling thread until the call is final. */
static napi_value usearch_js_deferred_wait(napi_env env, napi_callback_info info) {
    usearch_js_call_t* const call = (usearch_js_call_t*)usearch_js_self(env, info, NULL, NULL, NULL);
    if (!call) return NULL;
    usearch_outcome_wait(&call->outcome);
    return usearch_js_undefined(env);
}

/** @c settled() → a promise resolved once the call is final, its wait run on the libuv pool. */
static napi_value usearch_js_deferred_settled(napi_env env, napi_callback_info info) {
    napi_value self = NULL, promise = NULL, name = NULL;
    usearch_js_call_t* const call = (usearch_js_call_t*)usearch_js_self(env, info, NULL, NULL, &self);
    if (!call) return NULL;
    usearch_js_settling_t* const settling = (usearch_js_settling_t*)calloc(1, sizeof(usearch_js_settling_t));
    if (!settling) {
        usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not allocate the wait");
        return NULL;
    }
    settling->outcome = &call->outcome;
    if (napi_create_string_utf8(env, "usearch:deferred", NAPI_AUTO_LENGTH, &name) != napi_ok ||
        napi_create_reference(env, self, 1, &settling->wrapper) != napi_ok ||
        napi_create_async_work(env, NULL, name, usearch_js_settling_execute, usearch_js_settling_complete, settling,
                               &settling->work) != napi_ok ||
        napi_create_promise(env, &settling->promise, &promise) != napi_ok ||
        napi_queue_async_work(env, settling->work) != napi_ok) {
        if (settling->work) napi_delete_async_work(env, settling->work);
        if (settling->wrapper) napi_delete_reference(env, settling->wrapper);
        free(settling);
        usearch_js_throw(env, usearch_status_internal_k, "Could not queue the wait");
        return NULL;
    }
    return promise;
}

/** Builds a final call's answer, or throws its callback's exception, failure or stale arena. */
static napi_value usearch_js_call_finish(napi_env env, usearch_js_call_t* call, napi_value const* slots) {
    napi_value failure = NULL;
    if (call->failure && napi_get_reference_value(env, call->failure, &failure) == napi_ok) {
        napi_throw(env, failure);
        return NULL;
    }
    if (!usearch_js_succeeded(env, &call->outcome)) return NULL;
    if (call->arena && call->arena->holder != call) {
        usearch_js_throw(env, usearch_status_uninitialized_k,
                         "The arena was rewound or freed since this call wrote into it");
        return NULL;
    }
    return call->finish(env, call, slots);
}

/** @c result() → the verb's answer once the call is final, or its failure thrown; answers once. */
static napi_value usearch_js_deferred_result(napi_env env, napi_callback_info info) {
    usearch_js_call_t* const call = (usearch_js_call_t*)usearch_js_self(env, info, NULL, NULL, NULL);
    if (!call) return NULL;
    if (usearch_outcome_status(&call->outcome) == usearch_status_deferred_k) {
        usearch_js_throw(env, usearch_status_internal_k, "The call is not final yet");
        return NULL;
    }
    if (!call->finish) {
        usearch_js_throw(env, usearch_status_internal_k, "The call already answered");
        return NULL;
    }
    napi_value slots[usearch_js_slots_k] = {NULL};
    for (size_t slot = 0; slot != usearch_js_slots_k; ++slot)
        if (call->held[slot]) napi_get_reference_value(env, call->held[slot], &slots[slot]);
    napi_value const result = usearch_js_call_finish(env, call, slots);
    call->finish = NULL;
    usearch_js_call_release(env, call);
    return result;
}

napi_value usearch_js_deferred_define(napi_env env) {
    static napi_property_descriptor const properties[] = {
        {"status", NULL, usearch_js_deferred_status, NULL, NULL, NULL, napi_default_method, NULL},
        {"wait", NULL, usearch_js_deferred_wait, NULL, NULL, NULL, napi_default_method, NULL},
        {"settled", NULL, usearch_js_deferred_settled, NULL, NULL, NULL, napi_default_method, NULL},
        {"result", NULL, usearch_js_deferred_result, NULL, NULL, NULL, napi_default_method, NULL},
    };
    napi_value deferred_class = NULL;
    if (napi_define_class(env, "CompiledDeferred", NAPI_AUTO_LENGTH, usearch_js_deferred_construct, NULL,
                          usearch_js_table_size_(properties), properties, &deferred_class) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not define the CompiledDeferred class");
        return NULL;
    }
    return deferred_class;
}

usearch_js_call_t* usearch_js_call_open(napi_env env, usearch_completion_t completion, usearch_js_call_t* local,
                                        size_t bytes) {
    memset(local, 0, bytes);
    if (completion == usearch_completion_blocking_k) return local;
    usearch_js_call_t* const call = (usearch_js_call_t*)calloc(1, bytes);
    if (!call) usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not allocate the deferred call");
    return call;
}

bool usearch_js_call_hold(napi_env env, usearch_js_call_t* call, usearch_js_call_t const* local,
                          napi_value const* slots, napi_value* deferred) {
    if (call == local) return true;
    usearch_js_addon_t* const addon = usearch_js_addon(env);
    napi_value const instance = addon ? usearch_js_instantiate(env, addon->deferred_class) : NULL;
    bool held = instance != NULL;
    for (size_t slot = 0; held && slot != usearch_js_slots_k; ++slot)
        if (!usearch_js_value_absent(env, slots[slot]))
            held = napi_create_reference(env, slots[slot], 1, &call->held[slot]) == napi_ok;
    if (held && napi_wrap(env, instance, call, usearch_js_deferred_finalize, NULL, NULL) == napi_ok) {
        *deferred = instance;
        return true;
    }
    usearch_js_call_release(env, call);
    free(call);
    usearch_js_throw(env, usearch_status_internal_k, "Could not hold the deferred call");
    return false;
}

napi_value usearch_js_call_answer(napi_env env, usearch_js_call_t* call, usearch_js_call_t const* local,
                                  napi_value const* slots, napi_value deferred) {
    // A contended call never rewound the arena, so the holder's views stay readable.
    usearch_js_arena_t* const arena = call->arena;
    if (arena && usearch_outcome_status(&call->outcome) != usearch_status_contended_k) {
        usearch_js_arena_forget(env, arena, napi_detach_arraybuffer);
        arena->holder = call;
    }
    if (call != local) return deferred;
    napi_value const result = usearch_js_call_finish(env, call, slots);
    usearch_js_call_release(env, call);
    return result;
}

void usearch_js_call_abandon(usearch_js_call_t* call, usearch_js_call_t const* local) {
    if (!call) return;
    for (size_t block = 0; block != usearch_js_owned_k; ++block) free(call->owned[block]);
    if (call != local) free(call);
}

napi_value usearch_js_version(napi_env env, napi_callback_info info) {
    (void)info;
    return usearch_js_static_string(env, usearch_version_name());
}

napi_value usearch_js_capabilities_compiled(napi_env env, napi_callback_info info) {
    (void)info;
    return usearch_js_static_string(env, usearch_capabilities_compiled());
}

napi_value usearch_js_capabilities_available(napi_env env, napi_callback_info info) {
    (void)info;
    return usearch_js_static_string(env, usearch_capabilities_available());
}

#pragma endregion Deferred Calls and Library Probes
