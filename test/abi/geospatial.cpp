/**
 *  @file test/abi/geospatial.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The geospatial suite over the C interface, swept on the diagonal: the nearest relation at
 *      the best class and at every admitted class, over each compiled residency at one shard and at
 *      three.
 *
 *  Observations are points spread over a city's square kilometre on flat ground. Each cell checks
 *  the verbs a caller reaches: the points land under their keys across the shards and each counts
 *  once and reads back, a walk keeps most of the exact scan's top ten, a containment walk reports
 *  every entry of the scan's top ten inside its radius, a RAM index refines, through a plan too,
 *  and gathers a second source through a reshard plan while a GPU one refuses all three, an audit
 *  reaches every entry, cursors walk every entry and a sample draws distinct ones, the index saves
 *  into memory and back and into every residency and as a view byte for byte, an NVMe view answers,
 *  audits, walks and samples and refuses every write, a RAM index reshards into two targets through
 *  a plan with every key in one, and a cleared index holds nothing and ends any cursor it opened
 *  before the clear.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <algorithm> // `std::find`, `std::nth_element` - the contained keys, the radius
#include <array>     // `std::array` - every query's tenth-nearest distance
#include <limits>    // `std::numeric_limits` - unrecorded fields
#include <span>      // `std::span` - the queries' tenths that the oracle filled
#include <string>    // `std::string` - the cells' names
#include <vector>    // `std::vector` - the observations

#include "usearch/geospatial_index.h"

#include "harness.hpp"

namespace unum::usearch::test::abi {

/** Observations each cell builds, and the observations a second source holds; the queries held out
 *  past them number @c recall_samples_k. */
static constexpr std::size_t geospatial_entries_k = 2048, geospatial_packed_entries_k = 256;

/** The breadth every walk takes. */
static constexpr std::size_t geospatial_expansion_k = 64;

/** @p count points over a city's square kilometre, every other field unrecorded. */
static std::vector<usearch_geospatial_observation_t> generate_points(std::size_t count, util::u64_t seed) {
    constexpr util::f64_t nan_k = std::numeric_limits<util::f64_t>::quiet_NaN();
    split_mix_generator_t random_generator {seed};
    uniform_between<util::f64_t> const across {-1, 1};
    std::vector<usearch_geospatial_observation_t> points(count);
    for (usearch_geospatial_observation_t& point : points)
        point = {51.5 + across(random_generator) * 0.0045,
                 -0.1 + across(random_generator) * 0.0072,
                 nan_k,
                 nan_k,
                 nan_k,
                 nan_k,
                 nan_k,
                 nan_k,
                 nan_k,
                 nan_k,
                 nan_k};
    return points;
}

/** Every key of the first @p count entries reads one observation back, on host residencies. */
static void expect_gets(handle<geospatial_verbs_t>& index, std::size_t count) {
    if (index.residency == usearch_residency_vram_k) return;
    std::size_t misread = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        usearch_geospatial_observation_t const* observations = nullptr;
        std::size_t written = 0;
        usearch_geospatial_get_t get {};
        get.index = index.get(), get.executor = index.executor(), get.outcome = &outcome;
        get.key = key_of_offset(index.session->seed, offset);
        get.observations = &observations, get.written = &written, get.arena = index.arena();
        usearch_geospatial_get(&get);
        misread += outcome.status != usearch_status_success_k || written != 1;
    }
    expect_true(misread == 0);
}

/** A containment walk on the refined graph, at the median tenth-nearest distance, reports every
 *  entry of the exact scan's top ten inside that radius, where the residency serves a scan. */
static void expect_within(handle<geospatial_verbs_t>& index, geospatial_queries_t queries) {
    copied_matches_t oracle;
    usearch_outcome_t const scanned = scan_reference(index, queries, recall_samples_k, cell_wanted_k, cell_batch_k,
                                                     oracle);
    if (scanned.status == usearch_status_unsupported_k) return;
    if (!expect_success(scanned)) return;
    st::vector<std::size_t> const offsets = match_offsets(oracle);
    std::array<util::f64_t, recall_samples_k> tenths;
    std::size_t tenths_count = 0;
    for (std::size_t query = 0; query != recall_samples_k; ++query)
        if (oracle.counts[query]) tenths[tenths_count++] = oracle.scores[offsets[query + 1] - 1];
    std::span<util::f64_t> const tenth = std::span {tenths}.first(tenths_count);
    if (tenth.empty()) return void(expect_true(false));
    std::nth_element(tenth.begin(), tenth.begin() + tenth.size() / 2, tenth.end());
    util::f64_t const radius_metres = tenth[tenth.size() / 2];

    usearch_outcome_t outcome {};
    usearch_matches_tape_view_t matches {};
    usearch_geospatial_within_t within {};
    within.index = index.get(), within.executor = index.executor(), within.outcome = &outcome;
    within.count = recall_samples_k, within.queries = queries.observations, within.relation = queries.relation;
    within.tolerance = {radius_metres, 0}, within.matches = &matches, within.arena = index.arena();
    usearch_geospatial_within(&within);
    if (!expect_success(outcome)) return;
    std::vector<std::size_t> contained_offsets(recall_samples_k + 1);
    expect_true(download(contained_offsets.data(), matches.offsets, contained_offsets.size() * sizeof(std::size_t)));
    std::vector<usearch_key_t> contained(contained_offsets.back());
    expect_true(download(contained.data(), matches.keys, contained.size() * sizeof(usearch_key_t)));
    std::size_t unminted = 0;
    std::vector<util::u64_t> contained_entries(contained.size());
    for (std::size_t match = 0; match != contained.size(); ++match) {
        std::optional<util::u64_t> const entry = offset_of_key(index.session->seed, contained[match]);
        unminted += !entry, contained_entries[match] = entry.value_or(key_offsets_k);
    }
    std::size_t missed = 0;
    for (std::size_t query = 0; query != recall_samples_k; ++query)
        for (std::size_t match = offsets[query]; match != offsets[query + 1]; ++match) {
            // Scan scores round to f32, so an entry at the radius may land on either side of it.
            if (oracle.scores[match] >= radius_metres * (1 - 1e-4)) continue;
            std::optional<util::u64_t> const entry = offset_of_key(index.session->seed,
                                                                   std::bit_cast<usearch_key_t>(oracle.keys[match]));
            unminted += !entry;
            missed += std::find(contained_entries.begin() + contained_offsets[query],
                                contained_entries.begin() + contained_offsets[query + 1],
                                entry.value_or(key_offsets_k)) ==
                      contained_entries.begin() + contained_offsets[query + 1];
        }
    if (missed) fmt::println(stderr, "  {} entries inside {:.1f} m left out", missed, radius_metres);
    expect_true(unminted == 0);
    expect_true(missed == 0);
}

/** One cell, as the file's docblock describes. */
static void test_geospatial_behaviors(sweep_cell_t const& cell, util::u64_t seed) {
    session_t session = open_session(settings_of(session_settings(), cell), cell.cpu_class);
    if (!expect_true(bool(session))) return;
    handle<geospatial_verbs_t> index = open_index<geospatial_verbs_t>(session, geospatial_geometry_t {}, cell.shards);
    if (!expect_true(bool(index))) return;
    std::vector<usearch_geospatial_observation_t> const points = generate_points(
        geospatial_entries_k + recall_samples_k, seed);
    std::vector<usearch_geospatial_observation_t> const second_source = generate_points(geospatial_packed_entries_k,
                                                                                        ~seed);
    geospatial_queries_t const queries {points.data() + geospatial_entries_k, usearch_geospatial_relation_near_k, {}};

    if (!check_built(cell, index, points.data(), geospatial_entries_k)) return;
    expect_gets(index, geospatial_entries_k);
    expect_walk_recall(index, queries, recall_samples_k, geospatial_expansion_k, 0.8);
    expect_refined(index, queries, recall_samples_k, geospatial_expansion_k, usearch_status_success_k);
    expect_within(index, queries);
    expect_health(index, geospatial_entries_k);
    expect_cursor(index, geospatial_entries_k);
    expect_sample(index, geospatial_entries_k);
    expect_buffer_round_trip(index, queries, recall_samples_k, geospatial_expansion_k, 0.8,
                             usearch_status_unsupported_k);
    scratch_directory_t const scratch;
    std::string const stem = scratch.path("checkpoint");
    expect_widened(index, stem + "-widened");
    expect_persistence(cell, index, points.data(), queries, recall_samples_k, geospatial_expansion_k, 0.8, stem,
                       saved_bytes_t::identical_k, 0.0);
    if (cell.residency == usearch_residency_nvme_k) return;
    expect_refine_plan(index, points.data(), geospatial_entries_k, usearch_status_success_k);
    expect_reshard_gather(index, second_source.data(), geospatial_packed_entries_k, geospatial_entries_k,
                          usearch_status_success_k);
    expect_reshard_cut(index, usearch_status_success_k);
    expect_cleared(index, points.data());
}

/** A VRAM index's exact scan judges the same matches as the index reloaded on RAM scans, a
 *  centimetre apart at most, the resolution the entries keep. */
static void test_geospatial_device_scan(sweep_cell_t const& cell, util::u64_t seed) {
    session_t session = open_session(settings_of(session_settings(), cell), cell.cpu_class);
    if (!expect_true(bool(session))) return;
    handle<geospatial_verbs_t> index = open_index<geospatial_verbs_t>(session, geospatial_geometry_t {}, cell.shards);
    if (!expect_true(bool(index))) return;
    std::vector<usearch_geospatial_observation_t> const points = generate_points(
        geospatial_entries_k + recall_samples_k, seed);
    geospatial_queries_t const queries {points.data() + geospatial_entries_k, usearch_geospatial_relation_near_k, {}};
    usearch_traversal_stats_t stats {};
    if (!expect_success(insert_entries(index, points.data(), geospatial_entries_k, 0, cell_batch_k, 0, stats))) return;
    scratch_directory_t const scratch;
    expect_device_scan(index, queries, recall_samples_k, scratch.path("scanned"), 0.01);
}

void run_geospatial_abi_cases() {
    // Each stream keeps the name its cell had before the label spelled the constants.
    for (sweep_cell_t const& cell : diagonal_sweep(usearch_scalar_unknown_k)) {
        std::string const name = fmt::format("geospatial {} near", name_of(cell));
        std::string const label = fmt::format("test_geospatial_behaviors({}, {}, {})", constant_of(cell.residency),
                                              constant_of(cell.cpu_class), cell.shards);
        run_test_case(label.c_str(), test_geospatial_behaviors, cell, stream_key(test_run.settings.seed, name));
        if (cell.residency != usearch_residency_vram_k || cell.cpu_class != usearch_cpu_class_unknown_k) continue;
        std::string const scan_label = fmt::format("test_geospatial_device_scan({}, {})", constant_of(cell.cpu_class),
                                                   cell.shards);
        run_test_case(scan_label.c_str(), test_geospatial_device_scan, cell,
                      stream_key(test_run.settings.seed, fmt::format("geospatial {} device-scan", name_of(cell))));
    }
}

} // namespace unum::usearch::test::abi
