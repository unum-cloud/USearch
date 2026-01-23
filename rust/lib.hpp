#pragma once
#include "rust/cxx.h"

// We don't have to forward declare all of those:
struct Matches;
struct IndexOptions;
enum class MetricKind;
enum class ScalarKind;

#include <usearch/index_dense.hpp> // `unum::usearch::index_dense_t`

#include <memory> // `std::unique_ptr`

using uptr_t = size_t;

class NativeIndex {
  public:
    using metric_t = unum::usearch::metric_punned_t;
    using distance_t = unum::usearch::distance_punned_t;
    using index_dense_t = unum::usearch::index_dense_t;
    using add_result_t = typename index_dense_t::add_result_t;
    using search_result_t = typename index_dense_t::search_result_t;
    using vector_key_t = typename index_dense_t::vector_key_t;

    NativeIndex(std::unique_ptr<index_dense_t> index);

    void reserve(size_t) const;
    void reserve_capacity_and_threads(size_t, size_t) const;

    void add_b1x8(vector_key_t key, rust::Slice<uint8_t const> vector) const;
    void add_i8(vector_key_t key, rust::Slice<int8_t const> vector) const;
    void add_f16(vector_key_t key, rust::Slice<int16_t const> vector) const;
    void add_f32(vector_key_t key, rust::Slice<float const> vector) const;
    void add_f64(vector_key_t key, rust::Slice<double const> vector) const;

    Matches search_b1x8(rust::Slice<uint8_t const> query, size_t count) const;
    Matches search_i8(rust::Slice<int8_t const> query, size_t count) const;
    Matches search_f16(rust::Slice<int16_t const> query, size_t count) const;
    Matches search_f32(rust::Slice<float const> query, size_t count) const;
    Matches search_f64(rust::Slice<double const> query, size_t count) const;

    Matches exact_search_b1x8(rust::Slice<uint8_t const> query, size_t count) const;
    Matches exact_search_i8(rust::Slice<int8_t const> query, size_t count) const;
    Matches exact_search_f16(rust::Slice<int16_t const> query, size_t count) const;
    Matches exact_search_f32(rust::Slice<float const> query, size_t count) const;
    Matches exact_search_f64(rust::Slice<double const> query, size_t count) const;

    // clang-format off
    Matches filtered_search_b1x8(rust::Slice<uint8_t const> query, size_t count, uptr_t filter_function, uptr_t filter_state) const;
    Matches filtered_search_i8(rust::Slice<int8_t const> query, size_t count, uptr_t filter_function, uptr_t filter_state) const;
    Matches filtered_search_f16(rust::Slice<int16_t const> query, size_t count, uptr_t filter_function, uptr_t filter_state) const;
    Matches filtered_search_f32(rust::Slice<float const> query, size_t count, uptr_t filter_function, uptr_t filter_state) const;
    Matches filtered_search_f64(rust::Slice<double const> query, size_t count, uptr_t filter_function, uptr_t filter_state) const;
    // clang-format on

    size_t get_b1x8(vector_key_t key, rust::Slice<uint8_t> vector) const;
    size_t get_i8(vector_key_t key, rust::Slice<int8_t> vector) const;
    size_t get_f16(vector_key_t key, rust::Slice<int16_t> vector) const;
    size_t get_f32(vector_key_t key, rust::Slice<float> vector) const;
    size_t get_f64(vector_key_t key, rust::Slice<double> vector) const;

    size_t expansion_add() const;
    size_t expansion_search() const;
    void change_expansion_add(size_t n) const;
    void change_expansion_search(size_t n) const;

    void change_metric(uptr_t metric, uptr_t state) const;
    void change_metric_kind(MetricKind metric) const;

    size_t count(vector_key_t key) const;
    size_t remove(vector_key_t key) const;
    size_t rename(vector_key_t from, vector_key_t to) const;
    bool contains(vector_key_t key) const;

    size_t dimensions() const;
    size_t connectivity() const;
    size_t size() const;
    size_t capacity() const;
    size_t serialized_length() const;

    void save(rust::Str path) const;
    void load(rust::Str path) const;
    void view(rust::Str path) const;
    void reset() const;
    size_t memory_usage() const;
    char const* hardware_acceleration() const;

    void save_to_buffer(rust::Slice<uint8_t> buffer) const;
    void load_from_buffer(rust::Slice<uint8_t const> buffer) const;
    void view_from_buffer(rust::Slice<uint8_t const> buffer) const;

  private:
    std::unique_ptr<index_dense_t> index_;
};

std::unique_ptr<NativeIndex> new_native_index(IndexOptions const& options);
