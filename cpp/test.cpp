/**
 *  @file cpp/test.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The smoke test of the C++ wrappers: one dense round trip through every handle they own,
 *      against either library.
 */
#include <cstdint> // `std::uint8_t`
#include <cstdio>  // `std::remove`
#include <cstdlib> // `std::exit`
#include <cstring> // `std::memcmp`, `std::strlen`

#include <random> // `std::mt19937`, `std::normal_distribution`
#include <vector> // `std::vector`

#include <fmt/format.h>

#include "usearch/dense_index.hpp"

namespace us = unum::usearch;

template <typename outcome_type_>
static void expect_success(outcome_type_ const& reported, char const* context) {
    if (reported) return;
    fmt::println("Failed: {} - {}", context,
                 reported.message ? reported.message : usearch_status_name(reported.status));
    std::exit(1);
}

static void expect(bool must_be_true, char const* context) {
    if (must_be_true) return;
    fmt::println("Failed: {}", context);
    std::exit(1);
}

/** The key spelling @p offset as big-endian bytes, so keys made from offsets order as they do. */
static usearch_key_t key_of(std::size_t offset) {
    usearch_key_t key {};
    for (std::size_t byte = 0; byte != sizeof(offset); ++byte)
        key.bytes[sizeof(key.bytes) - 1 - byte] = static_cast<std::uint8_t>(offset >> (byte * 8));
    return key;
}

static bool same_key(usearch_key_t const& first, usearch_key_t const& second) {
    return std::memcmp(first.bytes, second.bytes, sizeof(first.bytes)) == 0;
}

int main() {
    std::size_t const rows = 1000, dimensions = 64, probe = 7;
    std::vector<float> vectors(rows * dimensions);
    std::mt19937 generator {42};
    std::normal_distribution<float> distribution;
    for (float& value : vectors) value = distribution(generator);
    std::vector<usearch_key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = key_of(row);
    usearch_rows_view_t const all_rows {vectors.data(), dimensions * sizeof(float), dimensions, usearch_scalar_f32_k};

    us::outcome<us::executor_t> const executor = us::executor_t::for_ram();
    expect_success(executor, "spawning an executor");
    us::outcome<us::arena_t> const arena = us::arena_t::make({.executor = executor.value.handle()});
    expect_success(arena, "making an arena");
    us::outcome<us::dense_index_t> index = us::dense_index_t::init_ram({.metric = usearch_dense_metric_angular_k,
                                                                        .quantization = usearch_scalar_bf16_k,
                                                                        .dimensions = dimensions,
                                                                        .connectivity = 16});
    expect_success(index, "creating an index");

    expect_success(index.value.reserve({.capacity = rows}, executor.value), "reserving the entries");
    expect_success(
        index.value.insert({.count = rows, .keys = {keys.data(), sizeof(usearch_key_t)}, .vectors = all_rows},
                           executor.value),
        "inserting the rows");
    us::outcome<usearch_dense_index_description_t> const described = index.value.describe({}, executor.value);
    expect_success(described, "describing the index");
    expect(described.value.size == rows, "An insertion is all-or-nothing, so every entry must be published");

    us::outcome<usearch_matches_view_t> const matches = index.value.search(
        {.count = rows, .queries = all_rows, .wanted = 1}, arena.value, executor.value);
    expect_success(matches, "searching every row");
    std::size_t recalled = 0;
    for (std::size_t row = 0; row != rows; ++row) recalled += same_key(matches.value.keys[row], keys[row]);
    expect(recalled * 10 >= rows * 9, "Self-recall collapsed below 90%");

    us::outcome<std::size_t> const counted = index.value.count({.key = keys[probe]}, executor.value);
    expect_success(counted, "counting a key");
    expect(counted.value == 1, "Unique keys hold one entry apiece");
    std::size_t written = 0;
    us::outcome<usearch_rows_view_t> const fetched = index.value.get(
        {.key = keys[probe], .written = &written, .outputs_scalar = usearch_scalar_f32_k}, arena.value, executor.value);
    expect_success(fetched, "getting a key");
    expect(written == 1 && fetched.value.columns == dimensions, "A key's one row must come back whole");

    char const* const path = "usearch_test_cpp.safegraphs";
    usearch_bytes_view_t const path_view {reinterpret_cast<std::uint8_t const*>(path), std::strlen(path)};
    expect_success(index.value.save({.path = path_view}, executor.value), "saving a monolith");
    // RAM copies a checkpoint in, so a view in place serves it from storage.
    us::outcome<us::executor_t> const storage = us::executor_t::for_nvme({.threads = 1});
    expect_success(storage, "spawning a storage executor");
    us::outcome<us::dense_index_t> viewed = us::dense_index_t::view_nvme({.path = path_view}, storage.value);
    expect_success(viewed, "viewing the monolith");
    us::outcome<> const refused = viewed.value.insert(
        {.count = 1, .keys = {keys.data(), sizeof(usearch_key_t)}, .vectors = all_rows}, storage.value);
    expect(refused.status == usearch_status_read_only_k, "A viewed handle must refuse every mutation");
    us::outcome<usearch_matches_view_t> const found = viewed.value.search(
        {.count = 1,
         .queries = {vectors.data() + probe * dimensions, 0, dimensions, usearch_scalar_f32_k},
         .wanted = 1},
        arena.value, storage.value);
    expect_success(found, "searching the view");
    expect(same_key(found.value.keys[0], keys[probe]), "The view must find a stored entry at itself");

    us::outcome<us::dense_cursor_t> const cursor = index.value.cursor_init();
    expect_success(cursor, "opening a cursor");
    us::outcome<usearch_keys_view_t> const pulled = index.value.cursor_next(
        cursor.value, {.written = &written, .wanted = rows}, arena.value, executor.value);
    expect_success(pulled, "pulling every entry through the cursor");
    expect(written == rows, "One pull wide enough must export every entry");

    std::remove(path);
    fmt::println("All C++ wrapper checks passed.");
    return 0;
}
