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
 *  every entry of the scan's top ten inside its radius, a RAM index refines, through a handle too,
 *  and absorbs while a GPU one refuses all three, an audit reaches every entry, cursors walk every
 *  entry and a sample draws distinct ones, the index saves into memory and back and into every
 *  residency and as a view byte for byte, an NVMe view answers, audits, walks and samples and
 *  refuses every write, a RAM index splits into two with every key in one part, and a cleared index
 *  holds nothing and ends a cursor opened before it.
 */
#include <algorithm> // `std::find`, `std::nth_element` - the contained keys, the radius
#include <limits>    // `std::numeric_limits` - unrecorded fields
#include <string>    // `std::string` - the cells' names
#include <vector>    // `std::vector` - the observations

#include "usearch/geospatial_index.h"

#include "harness.hpp"

namespace unum::usearch::test::abi {

/** Observations each cell builds, the queries held out past them, and the donor's observations. */
static constexpr std::size_t geospatial_entries_k = 2048, geospatial_queries_k = 128, geospatial_donor_entries_k = 256;

/** The breadth every walk takes. */
static constexpr std::size_t geospatial_expansion_k = 64;

/** @p count points over a city's square kilometre, every other field unrecorded. */
static std::vector<usearch_geospatial_observation_t> generate_points(std::size_t count, u64_t seed) {
    constexpr f64_t nan_k = std::numeric_limits<f64_t>::quiet_NaN();
    splitmix64_t random_generator {seed};
    uniform_between<f64_t> const across {-1, 1};
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
static std::size_t expect_gets(handle<geospatial_verbs_t>& index, std::size_t count) {
    if (index.residency == usearch_residency_vram_k) return 0;
    std::size_t misread = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        usearch_geospatial_observation_t const* observations = nullptr;
        std::size_t written = 0;
        usearch_geospatial_get_t get {};
        get.index = index.get(), get.executor = index.executor(), get.outcome = &outcome;
        get.key = key_of_offset(index.session->seed, offset);
        get.observations = &observations, get.written = &written, get.arena = index.session->arena;
        usearch_geospatial_get(&get);
        misread += outcome.status != usearch_status_success_k || written != 1;
    }
    return check(misread == 0);
}

/** A containment walk on the refined graph, at the median tenth-nearest distance, reports every
 *  entry of the exact scan's top ten inside that radius, where the residency serves a scan. */
static std::size_t expect_within(handle<geospatial_verbs_t>& index, geospatial_queries_t queries) {
    copied_matches_t oracle;
    usearch_outcome_t const scanned = scan_oracle(index, queries, geospatial_queries_k, cell_wanted_k, cell_batch_k,
                                                  oracle);
    if (scanned.status == usearch_status_unsupported_k) return 0;
    std::size_t failures = check(scanned);
    if (failures) return failures;
    std::vector<std::size_t> const offsets = match_offsets(oracle);
    std::vector<f64_t> tenth;
    for (std::size_t query = 0; query != geospatial_queries_k; ++query)
        if (oracle.counts[query]) tenth.push_back(oracle.scores[offsets[query + 1] - 1]);
    if (tenth.empty()) return check(false);
    std::nth_element(tenth.begin(), tenth.begin() + tenth.size() / 2, tenth.end());
    f64_t const radius_metres = tenth[tenth.size() / 2];

    usearch_outcome_t outcome {};
    usearch_matches_tape_view_t matches {};
    usearch_geospatial_within_t within {};
    within.index = index.get(), within.executor = index.executor(), within.outcome = &outcome;
    within.count = geospatial_queries_k, within.queries = queries.observations, within.relation = queries.relation;
    within.tolerance = {radius_metres, 0}, within.matches = &matches, within.arena = index.session->arena;
    usearch_geospatial_within(&within);
    failures += check(outcome);
    if (failures) return failures;
    std::vector<std::size_t> contained_offsets(geospatial_queries_k + 1);
    failures += check(
        download(contained_offsets.data(), matches.offsets, contained_offsets.size() * sizeof(std::size_t)));
    std::vector<usearch_key_t> contained(contained_offsets.back());
    failures += check(download(contained.data(), matches.keys, contained.size() * sizeof(usearch_key_t)));
    std::size_t unminted = 0;
    std::vector<u64_t> contained_entries(contained.size());
    for (std::size_t match = 0; match != contained.size(); ++match) {
        std::optional<u64_t> const entry = offset_of_key(index.session->seed, contained[match]);
        unminted += !entry, contained_entries[match] = entry.value_or(key_offsets_k);
    }
    std::size_t missed = 0;
    for (std::size_t query = 0; query != geospatial_queries_k; ++query)
        for (std::size_t match = offsets[query]; match != offsets[query + 1]; ++match) {
            // Scan scores round to f32, so an entry at the radius may land on either side of it.
            if (oracle.scores[match] >= radius_metres * (1 - 1e-4)) continue;
            std::optional<u64_t> const entry = offset_of_key(index.session->seed,
                                                             std::bit_cast<usearch_key_t>(oracle.keys[match]));
            unminted += !entry;
            missed += std::find(contained_entries.begin() + contained_offsets[query],
                                contained_entries.begin() + contained_offsets[query + 1],
                                entry.value_or(key_offsets_k)) ==
                      contained_entries.begin() + contained_offsets[query + 1];
        }
    if (missed) fmt::println(stderr, "  {} entries inside {:.1f} m left out", missed, radius_metres);
    return failures + check(unminted == 0) + check(missed == 0);
}

/** One cell, named @p name, as the file's docblock describes. */
static std::size_t check_geospatial_behaviors(sweep_cell_t const& cell, session_t& session, std::string const& name,
                                              u64_t seed) {
    handle<geospatial_verbs_t> index = open_index<geospatial_verbs_t>(session, geospatial_geometry_t {}, cell.shards);
    if (!index) return 1;
    std::vector<usearch_geospatial_observation_t> const points = generate_points(
        geospatial_entries_k + geospatial_queries_k, seed);
    std::vector<usearch_geospatial_observation_t> const donated = generate_points(geospatial_donor_entries_k, ~seed);
    geospatial_queries_t const queries {points.data() + geospatial_entries_k, usearch_geospatial_relation_near_k, {}};

    std::size_t failures = expect_built(cell, index, points.data(), geospatial_entries_k);
    if (failures) return failures;
    failures += expect_gets(index, geospatial_entries_k);
    failures += expect_walk_recall(index, queries, geospatial_queries_k, geospatial_expansion_k, 0.8);
    failures += expect_refined(index, queries, geospatial_queries_k, geospatial_expansion_k,
                               usearch_status_unsupported_k);
    failures += expect_within(index, queries);
    failures += expect_health(index, geospatial_entries_k);
    failures += expect_cursor(index, geospatial_entries_k);
    failures += expect_sample(index, geospatial_entries_k);
    failures += expect_buffer_round_trip(index, queries, geospatial_queries_k, geospatial_expansion_k, 0.8,
                                         usearch_status_unsupported_k);
    std::string const stem = fmt::format("/tmp/usearch-abi-geospatial-{:016x}", stream_key(seed, name));
    failures += expect_persistence(cell, index, points.data(), queries, geospatial_queries_k, geospatial_expansion_k,
                                   0.8, stem, saved_bytes_t::identical_k, 0.0);
    if (cell.residency == usearch_residency_nvme_k) return failures;
    failures += expect_refinement(index, points.data(), geospatial_entries_k, usearch_status_unsupported_k);
    failures += expect_absorbed(index, donated.data(), geospatial_donor_entries_k, geospatial_entries_k,
                                usearch_status_unsupported_k);
    failures += expect_split(index, usearch_status_unsupported_k);
    return failures + expect_cleared(index, points.data());
}

std::size_t check_geospatial_abi(settings_t const& settings) {
    std::size_t failures = 0;
    for (sweep_cell_t const& cell : diagonal_sweep(usearch_scalar_unknown_k)) {
        std::string const name = fmt::format("geospatial {} near", name_of(cell));
        if (!settings.filter.selects(name)) continue;
        session_t session = open_session(settings_of(settings, cell), cell.cpu_class);
        failures += report(
            name.c_str(),
            session ? check_geospatial_behaviors(cell, session, name, stream_key(settings.seed, name)) : 1);
    }
    return failures;
}

} // namespace unum::usearch::test::abi
