/**
 *  @file test/dense_vram.cpp
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The one VRAM runner: resolves @c USEARCH_DEVICES against the process listing and runs
 *      each named device through the suites its vendor's test object registered, then through the
 *      library's own arm for its class.
 *
 *  One object library carries the vendor-neutral shards, composed over the cell of every vendor
 *  this build carries, and each carried vendor's entry registers with @c vram_suites before
 *  @c main runs. The runner names no vendor: a device whose vendor registered nothing is reported
 *  and skipped, and an unset @c USEARCH_DEVICES runs the first device of every vendor present.
 *  The library's arms are reached as users reach them, through the C interface, which picks the
 *  arm of the device's class, so every class the library compiles is covered on a device of that
 *  class without instantiating its index again.
 */
#include <cstdint> // `std::uintptr_t` - the checkpoint path's per-run salt
#include <cstdio>  // `std::remove` - the reloaded checkpoint's files
#include <cstring> // `std::memcmp` - a match against its own key

#include <vector> // `std::vector` - the selected devices and the keys

#include <fmt/format.h>

#include "harness.hpp"
#include "usearch/dense_index.h"             // `usearch_dense_init_vram` - the library's arms
#include "usearch/utilities/vram_device.hpp" // `gpu_devices` - the listing `USEARCH_DEVICES` indexes

using namespace unum::usearch;
using namespace unum::usearch::test;

/** The positions in the listing @p settings names, or of the first listed device of every vendor
 *  when it names none. An ordinal past the listing exits with status 1, after @c main printed the
 *  listing. */
static std::vector<std::uint32_t> selected_devices(settings_t const& settings) {
    std::span<gpu_device_t const> const listed = gpu_devices();
    std::vector<std::uint32_t> selected;
    for (std::uint32_t const ordinal : settings.devices) {
        if (ordinal < listed.size()) {
            selected.push_back(ordinal);
            continue;
        }
        fmt::println(stderr, "USEARCH_DEVICES names device {}, past the {} listed", ordinal, listed.size());
        std::exit(1);
    }
    if (settings.devices.empty())
        for (std::uint32_t ordinal = 0; ordinal != listed.size(); ++ordinal)
            if (selected.empty() || listed[selected.back()].vendor != listed[ordinal].vendor)
                selected.push_back(ordinal);
    return selected;
}

/** One stored scalar of the dtype ladder. */
struct vram_arm_dtype_t {
    usearch_scalar_t scalar;
    char const* name;
};

/** The ladder's scalars in its order, of which the library compiles the first
 *  @c USEARCH_DENSE_DTYPES. */
inline constexpr vram_arm_dtype_t vram_arm_dtypes_k[] = {
    {usearch_scalar_bf16_k, "bf16"}, {usearch_scalar_u1x8_k, "u1"},   {usearch_scalar_i8_k, "i8"},
    {usearch_scalar_e4m3_k, "e4m3"}, {usearch_scalar_f16_k, "f16"},   {usearch_scalar_u8_k, "u8"},
    {usearch_scalar_e5m2_k, "e5m2"}, {usearch_scalar_i4x2_k, "i4"},   {usearch_scalar_u4x2_k, "u4"},
    {usearch_scalar_e3m2_k, "e3m2"}, {usearch_scalar_e2m3_k, "e2m3"},
};

/** The metrics each class arm compiles for the numeric dtypes, and for the bit-packed one. */
inline constexpr usearch_dense_metric_t vram_arm_numeric_metrics_k[] = {
    usearch_dense_metric_inner_k, usearch_dense_metric_angular_k, usearch_dense_metric_euclidean_k};
inline constexpr usearch_dense_metric_t vram_arm_binary_metrics_k[] = {
    usearch_dense_metric_inner_k, usearch_dense_metric_hamming_k, usearch_dense_metric_jaccard_k};

/** The 128-bit key of row @p row, its identity in the low bytes. */
static usearch_key_t vram_arm_key(std::size_t row) noexcept {
    usearch_key_t key {};
    for (std::size_t byte = 0; byte != 8; ++byte) key.bytes[15 - byte] = static_cast<std::uint8_t>(row >> (byte * 8));
    return key;
}

/** Device @p ordinal 's arm of one dtype and metric through the C interface, as @p shards shards
 *  on that device: every row goes in under its key, and nine in ten find themselves first. The
 *  inner product has no such guarantee, so every row of it must instead return the requested
 *  count, nearest first. The matches land in the executor's device arena and come back through
 *  @p suite. Three shards are then saved as three parts and loaded back onto the device, a shard
 *  per part, and judged again. */
static std::size_t check_vram_arm(vram_arm_dtype_t const& dtype, usearch_dense_metric_t metric, std::int32_t ordinal,
                                  std::size_t shards, vram_suite_t const& suite, u64_t seed) {
    constexpr std::size_t rows_count = 512, dimensions = 256, wanted = 10;
    auto const vectors = make_sketched_vectors<nk::f32_t>(rows_count, dimensions, stream_key(seed, dtype.name));
    std::vector<usearch_key_t> keys(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = vram_arm_key(row);
    usearch_rows_view_t const rows {vectors.data(), static_cast<std::size_t>(vectors.stride_bytes(0)), dimensions,
                                    usearch_scalar_f32_k};
    std::int32_t const devices[] = {ordinal, ordinal, ordinal};
    usearch_vram_options_t const vram {devices, shards};
    std::size_t failures = 0;

    usearch_outcome_t outcome {};
    usearch_dense_index_t index = nullptr;
    usearch_dense_init_vram_t init {};
    init.index = &index;
    init.outcome = &outcome;
    init.metric = metric;
    init.quantization = dtype.scalar;
    init.dimensions = dimensions;
    init.vram = vram;
    usearch_dense_init_vram(&init);
    failures += expect(!outcome.message);

    usearch_executor_t executor = nullptr;
    usearch_executor_for_vram_t placement {};
    placement.executor = &executor;
    placement.outcome = &outcome;
    placement.vram = vram;
    if (!failures) usearch_executor_for_vram(&placement);
    failures += expect(!outcome.message);

    usearch_dense_reserve_t reserve {};
    reserve.index = index;
    reserve.executor = executor;
    reserve.outcome = &outcome;
    reserve.capacity = rows_count;
    if (!failures) usearch_dense_reserve(&reserve);
    failures += expect(!outcome.message);

    usearch_dense_insert_t insert {};
    insert.index = index;
    insert.executor = executor;
    insert.outcome = &outcome;
    insert.count = rows_count;
    insert.keys = {keys.data(), sizeof(usearch_key_t)};
    insert.vectors = rows;
    if (!failures) usearch_dense_insert(&insert);
    failures += expect(!outcome.message);

    usearch_arena_t arena = nullptr;
    usearch_arena_init_t arena_init {};
    arena_init.arena = &arena;
    arena_init.executor = executor;
    arena_init.outcome = &outcome;
    if (!failures) usearch_arena_init(&arena_init);
    failures += expect(!outcome.message);

    bool const is_inner = metric == usearch_dense_metric_inner_k;
    auto search_every_row = [&] {
        usearch_matches_view_t matches {};
        usearch_dense_search_t search {};
        search.index = index;
        search.executor = executor;
        search.outcome = &outcome;
        search.count = rows_count;
        search.queries = rows;
        search.matches = &matches;
        search.arena = arena;
        search.wanted = wanted;
        if (!failures) usearch_dense_search(&search);
        failures += expect(!outcome.message);

        std::vector<usearch_key_t> found_keys(rows_count * matches.columns);
        std::vector<usearch_distance_t> found_scores(rows_count * matches.columns);
        std::vector<std::size_t> found_counts(rows_count);
        if (!failures)
            failures += expect(
                suite.download(found_keys.data(), matches.keys, found_keys.size() * sizeof(usearch_key_t)) &&
                suite.download(found_scores.data(), matches.scores, found_scores.size() * sizeof(usearch_distance_t)) &&
                suite.download(found_counts.data(), matches.counts, found_counts.size() * sizeof(std::size_t)));

        std::size_t passed = 0;
        if (!failures)
            for (std::size_t row = 0; row != rows_count; ++row) {
                std::size_t const found = found_counts[row];
                usearch_distance_t const* scores = &found_scores[row * matches.columns];
                bool ascending = true;
                for (std::size_t column = 1; column < found; ++column)
                    ascending &= scores[column - 1] <= scores[column];
                bool const found_itself = found && std::memcmp(&found_keys[row * matches.columns], &keys[row],
                                                               sizeof(usearch_key_t)) == 0;
                passed += ascending && (is_inner ? found == wanted : found_itself);
            }
        failures += expect(failures || (is_inner ? passed == rows_count : passed * 10 >= rows_count * 9));
    };
    search_every_row();

    if (shards == 3) {
        std::string const stem = fmt::format("/tmp/usearch-vram-arm-{:x}", reinterpret_cast<std::uintptr_t>(&vectors));
        std::string const manifest = stem + ".json";
        usearch_dense_save_t save {};
        save.index = index;
        save.executor = executor;
        save.outcome = &outcome;
        save.path = {reinterpret_cast<std::uint8_t const*>(manifest.data()), manifest.size()};
        if (!failures) usearch_dense_save(&save);
        failures += expect(!outcome.message);

        usearch_dense_free(index);
        index = nullptr;
        usearch_dense_load_vram_t load {};
        load.index = &index;
        load.executor = executor;
        load.outcome = &outcome;
        load.path = save.path;
        load.vram = {devices, 1};
        if (!failures) usearch_dense_load_vram(&load);
        failures += expect(!outcome.message);
        search_every_row();

        std::remove(manifest.c_str());
        for (std::size_t part = 1; part <= shards; ++part)
            std::remove(fmt::format("{}-{:05}-of-{:05}.safegraphs", stem, part, shards).c_str());
    }

    usearch_arena_free(arena);
    usearch_executor_free(executor);
    usearch_dense_free(index);
    return failures;
}

/** Device @p ordinal 's arm of every compiled dtype under every metric of its family, which the C
 *  interface picks by the device's class, as one shard and as three, reading the matches back
 *  through @p suite. */
static std::size_t check_vram_arms(settings_t const& settings, std::uint32_t ordinal, vram_suite_t const& suite) {
    std::size_t failures_count = 0;
    if (!settings.selects("vram_arms")) return failures_count;
    for (std::size_t position = 0; position != USEARCH_DENSE_DTYPES; ++position) {
        vram_arm_dtype_t const& dtype = vram_arm_dtypes_k[position];
        std::span<usearch_dense_metric_t const> const metrics = dtype.scalar == usearch_scalar_u1x8_k
                                                                    ? std::span {vram_arm_binary_metrics_k}
                                                                    : std::span {vram_arm_numeric_metrics_k};
        for (usearch_dense_metric_t const metric : metrics)
            for (std::size_t const shards : {std::size_t {1}, std::size_t {3}}) {
                std::string const name = fmt::format("arm {}+{} {}-shard", dtype.name,
                                                     usearch_dense_metric_name(metric), shards);
                failures_count += report(name.c_str(), check_vram_arm(dtype, metric, static_cast<std::int32_t>(ordinal),
                                                                      shards, suite, settings.seed));
            }
    }
    return failures_count;
}

int main() {
    settings_t const settings = read_settings();
    print(probe_machine());
    print(settings);
    std::span<gpu_device_t const> const listed = gpu_devices();
    for (std::size_t position = 0; position != listed.size(); ++position)
        fmt::println("- GPU {}: {}, {} #{}, {} compute units, {} MiB", position, listed[position].name,
                     usearch_gpu_class_name(static_cast<usearch_gpu_class_t>(listed[position].gpu_class)),
                     listed[position].vendor_ordinal, listed[position].compute_units,
                     listed[position].memory_bytes >> 20);
    std::vector<std::uint32_t> const ordinals = selected_devices(settings);
    if (ordinals.empty()) return fmt::println("- GPU: none listed"), 0;
    std::size_t failures_count = 0;
    for (std::uint32_t const ordinal : ordinals) {
        gpu_device_t const& device = listed[ordinal];
        vram_suite_t const* suite = nullptr;
        for (vram_suite_t const& registered : vram_suites())
            if (registered.vendor == device.vendor) suite = &registered;
        if (!suite) {
            fmt::println("- GPU: {} #{} has no suite linked", device.name, device.vendor_ordinal);
            continue;
        }
        failures_count += suite->check(settings, device);
        failures_count += test::sequence::check_sequence_vram(settings, device);
        failures_count += test::shapes::check_shapes_vram(settings, device);
        failures_count += test::geospatial::check_geospatial_vram(settings, device);
        failures_count += check_vram_arms(settings, ordinal, *suite);
    }
    return report("total", failures_count) ? 1 : 0;
}
