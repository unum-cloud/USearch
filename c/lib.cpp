#include <cassert>

#include <usearch/index_dense.hpp>

extern "C" {
#include "usearch.h"
}

using namespace unum::usearch;
using namespace unum;

using add_result_t = typename index_dense_t::add_result_t;
using search_result_t = typename index_dense_t::search_result_t;
using labeling_result_t = typename index_dense_t::labeling_result_t;

static_assert(std::is_same<usearch_key_t, index_dense_t::vector_key_t>::value, "Type mismatch between C and C++");
static_assert(std::is_same<usearch_distance_t, index_dense_t::distance_t>::value, "Type mismatch between C and C++");

metric_kind_t metric_kind_to_cpp(usearch_metric_kind_t kind) {
    switch (kind) {
    case usearch_metric_ip_k: return metric_kind_t::ip_k;
    case usearch_metric_l2sq_k: return metric_kind_t::l2sq_k;
    case usearch_metric_cos_k: return metric_kind_t::cos_k;
    case usearch_metric_haversine_k: return metric_kind_t::haversine_k;
    case usearch_metric_divergence_k: return metric_kind_t::divergence_k;
    case usearch_metric_pearson_k: return metric_kind_t::pearson_k;
    case usearch_metric_jaccard_k: return metric_kind_t::jaccard_k;
    case usearch_metric_hamming_k: return metric_kind_t::hamming_k;
    case usearch_metric_tanimoto_k: return metric_kind_t::tanimoto_k;
    case usearch_metric_sorensen_k: return metric_kind_t::sorensen_k;
    default: return metric_kind_t::unknown_k;
    }
}

usearch_metric_kind_t metric_kind_to_c(metric_kind_t kind) {
    switch (kind) {
    case metric_kind_t::ip_k: return usearch_metric_ip_k;
    case metric_kind_t::l2sq_k: return usearch_metric_l2sq_k;
    case metric_kind_t::cos_k: return usearch_metric_cos_k;
    case metric_kind_t::haversine_k: return usearch_metric_haversine_k;
    case metric_kind_t::divergence_k: return usearch_metric_divergence_k;
    case metric_kind_t::pearson_k: return usearch_metric_pearson_k;
    case metric_kind_t::jaccard_k: return usearch_metric_jaccard_k;
    case metric_kind_t::hamming_k: return usearch_metric_hamming_k;
    case metric_kind_t::tanimoto_k: return usearch_metric_tanimoto_k;
    case metric_kind_t::sorensen_k: return usearch_metric_sorensen_k;
    default: return usearch_metric_unknown_k;
    }
}
scalar_kind_t scalar_kind_to_cpp(usearch_scalar_kind_t kind) {
    switch (kind) {
    case usearch_scalar_f32_k: return scalar_kind_t::f32_k;
    case usearch_scalar_f64_k: return scalar_kind_t::f64_k;
    case usearch_scalar_f16_k: return scalar_kind_t::f16_k;
    case usearch_scalar_i8_k: return scalar_kind_t::i8_k;
    case usearch_scalar_b1_k: return scalar_kind_t::b1x8_k;
    default: return scalar_kind_t::unknown_k;
    }
}

usearch_scalar_kind_t scalar_kind_to_c(scalar_kind_t kind) {
    switch (kind) {
    case scalar_kind_t::f32_k: return usearch_scalar_f32_k;
    case scalar_kind_t::f64_k: return usearch_scalar_f64_k;
    case scalar_kind_t::f16_k: return usearch_scalar_f16_k;
    case scalar_kind_t::i8_k: return usearch_scalar_i8_k;
    case scalar_kind_t::b1x8_k: return usearch_scalar_b1_k;
    default: return usearch_scalar_unknown_k;
    }
}

add_result_t add_(index_dense_t* index, usearch_key_t key, void const* vector, scalar_kind_t kind) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->add(key, (f32_t const*)vector);
    case scalar_kind_t::f64_k: return index->add(key, (f64_t const*)vector);
    case scalar_kind_t::f16_k: return index->add(key, (f16_t const*)vector);
    case scalar_kind_t::i8_k: return index->add(key, (i8_t const*)vector);
    case scalar_kind_t::b1x8_k: return index->add(key, (b1x8_t const*)vector);
    default: return add_result_t{}.failed("Unknown scalar kind!");
    }
}

std::size_t get_(index_dense_t* index, usearch_key_t key, size_t count, void* vector, scalar_kind_t kind) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->get(key, (f32_t*)vector, count);
    case scalar_kind_t::f64_k: return index->get(key, (f64_t*)vector, count);
    case scalar_kind_t::f16_k: return index->get(key, (f16_t*)vector, count);
    case scalar_kind_t::i8_k: return index->get(key, (i8_t*)vector, count);
    case scalar_kind_t::b1x8_k: return index->get(key, (b1x8_t*)vector, count);
    default: return search_result_t().failed("Unknown scalar kind!");
    }
}

search_result_t search_(index_dense_t* index, void const* vector, scalar_kind_t kind, size_t n) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->search((f32_t const*)vector, n);
    case scalar_kind_t::f64_k: return index->search((f64_t const*)vector, n);
    case scalar_kind_t::f16_k: return index->search((f16_t const*)vector, n);
    case scalar_kind_t::i8_k: return index->search((i8_t const*)vector, n);
    case scalar_kind_t::b1x8_k: return index->search((b1x8_t const*)vector, n);
    default: return search_result_t().failed("Unknown scalar kind!");
    }
}

extern "C" {

__attribute__((export_name("usearch_init")))
USEARCH_EXPORT usearch_index_t usearch_init(usearch_init_options_t* options, usearch_error_t* error) {

    assert(options && error && "Missing arguments");

    index_dense_config_t config(options->connectivity, options->expansion_add, options->expansion_search);
    config.multi = options->multi;
    metric_kind_t metric_kind = metric_kind_to_cpp(options->metric_kind);
    scalar_kind_t scalar_kind = scalar_kind_to_cpp(options->quantization);

    metric_punned_t metric = //
        !options->metric ? metric_punned_t(options->dimensions, metric_kind, scalar_kind)
                         : metric_punned_t(options->dimensions,                               //
                                           reinterpret_cast<std::uintptr_t>(options->metric), //
                                           metric_punned_signature_t::array_array_k,          //
                                           metric_kind, scalar_kind);

    index_dense_t index = index_dense_t::make(metric, config);
    index_dense_t* result_ptr = new index_dense_t(std::move(index));
    if (!result_ptr || !*result_ptr)
        *error = "Out of memory!";
    return result_ptr;
}

__attribute__((export_name("usearch_free")))
USEARCH_EXPORT void usearch_free(usearch_index_t index, usearch_error_t*) {
    delete reinterpret_cast<index_dense_t*>(index);
}

__attribute__((export_name("usearch_serialized_length")))
USEARCH_EXPORT size_t usearch_serialized_length(usearch_index_t index, usearch_error_t*) {
    assert(index && "Missing arguments");
    return reinterpret_cast<index_dense_t*>(index)->serialized_length();
}

__attribute__((export_name("usearch_save")))
USEARCH_EXPORT void usearch_save(usearch_index_t index, char const* path, usearch_error_t* error) {

    assert(index && path && error && "Missing arguments");
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->save(path);
    if (!result)
        *error = result.error.release();
}

__attribute__((export_name("usearch_load")))
USEARCH_EXPORT void usearch_load(usearch_index_t index, char const* path, usearch_error_t* error) {

    assert(index && path && error && "Missing arguments");
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->load(path);
    if (!result)
        *error = result.error.release();
}

__attribute__((export_name("usearch_view")))
USEARCH_EXPORT void usearch_view(usearch_index_t index, char const* path, usearch_error_t* error) {

    assert(index && path && error && "Missing arguments");
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->view(path);
    if (!result)
        *error = result.error.release();
}

__attribute__((export_name("usearch_metadata")))
USEARCH_EXPORT void usearch_metadata(char const* path, usearch_init_options_t* options, usearch_error_t* error) {

    assert(path && options && error && "Missing arguments");
    index_dense_metadata_result_t result = index_dense_metadata_from_path(path);
    if (!result)
        *error = result.error.release();

    options->metric_kind = metric_kind_to_c(result.head.kind_metric);
    options->quantization = scalar_kind_to_c(result.head.kind_scalar);
    options->dimensions = result.head.dimensions;
    options->multi = result.head.multi;

    options->connectivity = 0;
    options->expansion_add = 0;
    options->expansion_search = 0;
    options->metric = NULL;
}

__attribute__((export_name("usearch_save_buffer")))
USEARCH_EXPORT void usearch_save_buffer(usearch_index_t index, void* buffer, size_t length, usearch_error_t* error) {

    assert(index && buffer && length && error && "Missing arguments");
    memory_mapped_file_t memory_map((byte_t*)buffer, length);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->save(std::move(memory_map));
    if (!result)
        *error = result.error.release();
}

__attribute__((export_name("usearch_load_buffer")))
USEARCH_EXPORT void usearch_load_buffer(usearch_index_t index, void const* buffer, size_t length,
                                        usearch_error_t* error) {

    assert(index && buffer && length && error && "Missing arguments");
    memory_mapped_file_t memory_map((byte_t*)buffer, length);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->load(std::move(memory_map));
    if (!result)
        *error = result.error.release();
}

__attribute__((export_name("usearch_view_buffer")))
USEARCH_EXPORT void usearch_view_buffer(usearch_index_t index, void const* buffer, size_t length,
                                        usearch_error_t* error) {

    assert(index && buffer && length && error && "Missing arguments");
    memory_mapped_file_t memory_map((byte_t*)buffer, length);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(index)->view(std::move(memory_map));
    if (!result)
        *error = result.error.release();
}

__attribute__((export_name("usearch_metadata_buffer")))
USEARCH_EXPORT void usearch_metadata_buffer(void const* buffer, size_t length, usearch_init_options_t* options,
                                            usearch_error_t* error) {

    assert(buffer && length && options && error && "Missing arguments");
    index_dense_metadata_result_t result =
        index_dense_metadata_from_buffer(memory_mapped_file_t((byte_t*)(buffer), length));
    if (!result)
        *error = result.error.release();

    options->metric_kind = metric_kind_to_c(result.head.kind_metric);
    options->quantization = scalar_kind_to_c(result.head.kind_scalar);
    options->dimensions = result.head.dimensions;
    options->multi = result.head.multi;

    options->connectivity = 0;
    options->expansion_add = 0;
    options->expansion_search = 0;
    options->metric = NULL;
}

__attribute__((export_name("usearch_size")))
USEARCH_EXPORT size_t usearch_size(usearch_index_t index, usearch_error_t*) { //
    return reinterpret_cast<index_dense_t*>(index)->size();
}

__attribute__((export_name("usearch_capacity")))
USEARCH_EXPORT size_t usearch_capacity(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_dense_t*>(index)->capacity();
}

__attribute__((export_name("usearch_dimensions")))
USEARCH_EXPORT size_t usearch_dimensions(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_dense_t*>(index)->dimensions();
}

__attribute__((export_name("usearch_connectivity")))
USEARCH_EXPORT size_t usearch_connectivity(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_dense_t*>(index)->connectivity();
}

__attribute__((export_name("usearch_reserve")))
USEARCH_EXPORT void usearch_reserve(usearch_index_t index, size_t capacity, usearch_error_t* error) {
    assert(index && error && "Missing arguments");
    if (!reinterpret_cast<index_dense_t*>(index)->reserve(capacity))
        *error = "Out of memory!";
}

__attribute__((export_name("usearch_add")))
USEARCH_EXPORT void usearch_add(                                                              //
    usearch_index_t index, usearch_key_t key, void const* vector, usearch_scalar_kind_t kind, //
    usearch_error_t* error) {

    assert(index && vector && error && "Missing arguments");
    add_result_t result = add_(reinterpret_cast<index_dense_t*>(index), key, vector, scalar_kind_to_cpp(kind));
    if (!result)
        *error = result.error.release();
}

__attribute__((export_name("usearch_contains")))
USEARCH_EXPORT bool usearch_contains(usearch_index_t index, usearch_key_t key, usearch_error_t*) {
    assert(index && "Missing arguments");
    return reinterpret_cast<index_dense_t*>(index)->contains(key);
}

__attribute__((export_name("usearch_count")))
USEARCH_EXPORT size_t usearch_count(usearch_index_t index, usearch_key_t key, usearch_error_t*) {
    assert(index && "Missing arguments");
    return reinterpret_cast<index_dense_t*>(index)->count(key);
}

__attribute__((export_name("usearch_search")))
USEARCH_EXPORT size_t usearch_search(                                                            //
    usearch_index_t index, void const* vector, usearch_scalar_kind_t kind, size_t results_limit, //
    usearch_key_t* found_keys, usearch_distance_t* found_distances, usearch_error_t* error) {

    assert(index && vector && error && "Missing arguments");
    search_result_t result =
        search_(reinterpret_cast<index_dense_t*>(index), vector, scalar_kind_to_cpp(kind), results_limit);
    if (!result) {
        *error = result.error.release();
        return 0;
    }

    return result.dump_to(found_keys, found_distances);
}

__attribute__((export_name("usearch_get")))
USEARCH_EXPORT size_t usearch_get(                          //
    usearch_index_t index, usearch_key_t key, size_t count, //
    void* vectors, usearch_scalar_kind_t kind, usearch_error_t*) {

    assert(index && vectors);
    return get_(reinterpret_cast<index_dense_t*>(index), key, count, vectors, scalar_kind_to_cpp(kind));
}

__attribute__((export_name("usearch_remove")))
USEARCH_EXPORT size_t usearch_remove(usearch_index_t index, usearch_key_t key, usearch_error_t* error) {

    assert(index && error && "Missing arguments");
    labeling_result_t result = reinterpret_cast<index_dense_t*>(index)->remove(key);
    if (!result)
        *error = result.error.release();
    return result.completed;
}

__attribute__((export_name("usearch_rename")))
USEARCH_EXPORT size_t usearch_rename( //
    usearch_index_t index, usearch_key_t from, usearch_key_t to, usearch_error_t* error) {

    assert(index && error && "Missing arguments");
    labeling_result_t result = reinterpret_cast<index_dense_t*>(index)->rename(from, to);
    if (!result)
        *error = result.error.release();
    return result.completed;
}

__attribute__((export_name("usearch_distance")))
USEARCH_EXPORT usearch_distance_t usearch_distance(       //
    void const* vector_first, void const* vector_second,  //
    usearch_scalar_kind_t scalar_kind, size_t dimensions, //
    usearch_metric_kind_t metric_kind, usearch_error_t* error) {

    (void)error;
    metric_punned_t metric(dimensions, metric_kind_to_cpp(metric_kind), scalar_kind_to_cpp(scalar_kind));
    return metric((byte_t const*)vector_first, (byte_t const*)vector_second);
}

__attribute__((export_name("usearch_exact_search")))
USEARCH_EXPORT void usearch_exact_search(                             //
    void const* dataset, size_t dataset_count, size_t dataset_stride, //
    void const* queries, size_t queries_count, size_t queries_stride, //
    usearch_scalar_kind_t scalar_kind, size_t dimensions,             //
    usearch_metric_kind_t metric_kind, size_t count, size_t threads,  //
    usearch_key_t* keys, size_t keys_stride,                          //
    usearch_distance_t* distances, size_t distances_stride,           //
    usearch_error_t* error) {

    assert(dataset && queries && keys && distances && error && "Missing arguments");

    metric_punned_t metric(dimensions, metric_kind_to_cpp(metric_kind), scalar_kind_to_cpp(scalar_kind));
    executor_default_t executor(threads);
    static exact_search_t search;
    exact_search_results_t result = search(                    //
        (byte_t const*)dataset, dataset_count, dataset_stride, //
        (byte_t const*)queries, queries_count, queries_stride, //
        count, metric);

    if (!result) {
        *error = "Out of memory, allocating a temporary buffer for batch results";
        return;
    }

    // Export results into the output buffer
    for (std::size_t query_idx = 0; query_idx != queries_count; ++query_idx) {
        auto query_result = result.at(query_idx);
        auto query_keys = (usearch_key_t*)((byte_t*)keys + query_idx * keys_stride);
        auto query_distances = (usearch_distance_t*)((byte_t*)distances + query_idx * distances_stride);
        for (std::size_t i = 0; i != count; ++i)
            query_keys[i] = static_cast<usearch_key_t>(query_result[i].offset),
            query_distances[i] = static_cast<usearch_distance_t>(query_result[i].distance);
    }
}
}
