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
    us::rows_view<float> const all_rows {vectors.data(), rows, dimensions};

    us::outcome<us::executor_t> const executor = us::executor_t::for_ram();
    expect_success(executor, "spawning an executor");
    us::outcome<us::arena_t> arena = us::arena_t::make({.executor = executor.value.handle()});
    expect_success(arena, "making an arena");
    usearch_dense_init_ram_t const geometry {.metric = usearch_dense_metric_angular_k,
                                             .quantization = usearch_scalar_bf16_k,
                                             .dimensions = dimensions,
                                             .connectivity = 16};
    us::outcome<us::dense_index_t> index = us::dense_index_t::init_ram(geometry);
    expect_success(index, "creating an index");
    auto const on = index.value.on(executor.value);

    expect_success(on.reserve(rows), "reserving the entries");
    expect_success(on.insert<float>(keys, all_rows), "inserting the rows");
    us::outcome<usearch_dense_index_description_t> const described = on.describe();
    expect_success(described, "describing the index");
    expect(described.value.size == rows, "An insertion is all-or-nothing, so every entry must be published");
    expect(on.insert<float>({keys.data(), 1}, all_rows).status == usearch_status_shape_mismatch_k,
           "Keys and rows of different counts must be refused before the call crosses");

    us::outcome<usearch_matches_view_t> const matches = on.search<float>(all_rows, 1, arena.value);
    expect_success(matches, "searching every row");
    std::size_t recalled = 0;
    for (std::size_t row = 0; row != rows; ++row) recalled += same_key(matches.value.keys[row], keys[row]);
    expect(recalled * 10 >= rows * 9, "Self-recall collapsed below 90%");

    us::outcome<std::size_t> const counted = on.count(keys[probe]);
    expect_success(counted, "counting a key");
    expect(counted.value == 1, "Unique keys hold one entry apiece");
    us::outcome<us::dense_vectors<float>> const fetched = on.get<float>(keys[probe], arena.value);
    expect_success(fetched, "getting a key");
    float const* const fetched_row = fetched.value.row(0);
    expect(fetched.value.written == 1 && fetched.value.view.columns == dimensions && fetched_row[0] != 0,
           "A key's one row must come back whole");

    // The receiver's type picks the completion, and the deferred call's owner waits on it.
    {
        us::deferred<usearch_reachability_t> audit = index.value.deferred(executor.value).health();
        us::outcome<usearch_reachability_t> const audited = audit.wait();
        expect_success(audited, "auditing deferred");
        expect(audited.value.audited == rows, "A deferred audit must see every entry");
    }

    // A plan keeps its round between calls, and its finish on the receiver closes it.
    us::dense_refine_plan_t refine_plan;
    std::size_t reports = 0;
    auto cancel_first = [&](std::size_t, std::size_t) noexcept {
        return reports++ == 0 ? usearch_cancel_k : usearch_continue_k;
    };
    us::outcome<usearch_refine_report_t> const stopped = on.refine(
        {.refine = {.memory_limit = 0}, .progress = {cancel_first}, .plan = &refine_plan});
    expect(stopped.status == usearch_status_cancelled_k, "A cancel must stop the planned round");
    expect(refine_plan.handle() != nullptr, "A cancel must leave the plan live");
    expect_success(on.refine({.plan = &refine_plan}), "continuing the planned round");
    expect(refine_plan.phase().phase == usearch_phase_done_k, "A continued round must run to its end");
    us::outcome<usearch_refine_report_t> const finished = on.refine_plan_finish(std::move(refine_plan));
    expect_success(finished, "finishing the refine plan");
    expect(std::strlen(usearch_refine_verdict_name(finished.value.verdict)) != 0, "Every verdict has a name");
    fmt::println("Refined: {} reselected, verdict \"{}\", phase \"{}\"", finished.value.reselected,
                 usearch_refine_verdict_name(finished.value.verdict), usearch_phase_name(usearch_phase_done_k));

    char const* const path = "usearch_test_cpp.safegraphs";
    usearch_bytes_view_t const path_view {reinterpret_cast<std::uint8_t const*>(path), std::strlen(path)};
    expect_success(on.save(path_view), "saving a monolith");
    // RAM copies a checkpoint in, so a view in place serves it from storage.
    us::outcome<us::executor_t> const storage = us::executor_t::for_nvme({.threads = 1});
    expect_success(storage, "spawning a storage executor");
    us::outcome<us::dense_index_t> viewed = us::dense_index_t::view_nvme({.path = path_view}, storage.value);
    expect_success(viewed, "viewing the monolith");
    auto const on_storage = viewed.value.on(storage.value);
    us::outcome<> const refused = on_storage.insert<float>({keys.data(), 1}, {vectors.data(), 1, dimensions});
    expect(refused.status == usearch_status_read_only_k, "A viewed handle must refuse every mutation");
    us::outcome<usearch_matches_view_t> const found = on_storage.search<float>(
        {vectors.data() + probe * dimensions, 1, dimensions}, 1, arena.value);
    expect_success(found, "searching the view");
    expect(same_key(found.value.keys[0], keys[probe]), "The view must find a stored entry at itself");

    us::outcome<us::dense_cursor_t> const cursor = index.value.cursor_init();
    expect_success(cursor, "opening a cursor");
    us::outcome<us::dense_cursor_batch_t> const pulled = on.cursor_next(cursor.value, rows, arena.value);
    expect_success(pulled, "pulling every entry through the cursor");
    expect(pulled.value.written == rows, "One pull wide enough must export every entry");

    constexpr std::size_t gathered_rows = 10;
    usearch_key_t gathered_keys[gathered_rows];
    for (std::size_t row = 0; row != gathered_rows; ++row) gathered_keys[row] = key_of(rows + row);
    us::outcome<us::dense_index_t> other = us::dense_index_t::init_ram(geometry);
    expect_success(other, "creating a second index");
    expect_success(
        other.value.on(executor.value).insert<float>(gathered_keys, {vectors.data(), gathered_rows, dimensions}),
        "inserting into the second index");
    usearch_dense_index_t const gathered[2] = {index.value.handle(), other.value.handle()};
    expect_success(on.reshard({gathered, 2}, {gathered, 1}), "gathering two sources into one");
    expect(on.describe().value.size == rows + gathered_rows, "A gather keeps every entry of its sources");

    // A cut moves one source into several targets, emptying it.
    us::outcome<us::dense_index_t> first_part = us::dense_index_t::init_ram(geometry);
    us::outcome<us::dense_index_t> second_part = us::dense_index_t::init_ram(geometry);
    expect_success(first_part, "creating the first part");
    expect_success(second_part, "creating the second part");
    usearch_dense_index_t const source[1] = {index.value.handle()};
    usearch_dense_index_t const parts[2] = {first_part.value.handle(), second_part.value.handle()};
    expect_success(on.reshard(source, parts, {.seed = 7}), "cutting one source into two targets");
    std::size_t const first_size = first_part.value.on(executor.value).describe().value.size;
    std::size_t const second_size = second_part.value.on(executor.value).describe().value.size;
    expect(on.describe().value.size == 0, "A cut source must end empty");
    expect(first_size + second_size == rows + gathered_rows, "A cut keeps every entry of its source");
    expect(first_size != 0 && second_size != 0, "A cut must give every target a part");

    std::remove(path);
    fmt::println("All C++ wrapper checks passed.");
    return 0;
}
