/**
 *  @file test/abi/dense.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The dense suite over the C interface, swept on the diagonal: every scalar at the best
 *      class, every admitted class at bf16, and serial at every scalar, over each compiled
 *      residency at one shard and at three.
 *
 *  Each cell builds one index the way a binding does, through a session of its residency, and
 *  checks what a caller observes. The vectors land under their keys across the shards; every key
 *  counts once and reads back; nine in ten vectors find themselves first, and a walk keeps nine in
 *  ten of the exact scan's top ten; a predicate admits only what it admits; one search's matches
 *  hint the next, counting the keys no entry holds; refinements settle, an audit reaches every
 *  entry, cursors walk every entry and a sample draws distinct ones; the index saves into memory
 *  and back, and into every residency and as a view, and an NVMe view serves the same searches,
 *  audits, walks and samples and refuses every write; a refinement handle holds the index closed
 *  while its cancelled pulls carry their totals on; a donor absorbs whole; a RAM index compacts
 *  into one shard; the index splits into two with every key in one part; a cleared index holds
 *  nothing and ends a cursor opened before it; every metric's name reads back.
 */
#include <cstring> // `std::memcmp` - the keys' order

#include <algorithm> // `std::binary_search`, `std::sort` - the admitted keys
#include <string>    // `std::string` - the cells' names and checkpoint stems
#include <vector>    // `std::vector` - the keys

#include "usearch/dense_index.h"

#include "harness.hpp"

namespace unum::usearch::test::abi {

/** Rows each cell builds, rows its donor adds, and the columns of each. */
static constexpr std::size_t dense_rows_k = 1024, dense_donor_rows_k = 256, dense_dimensions_k = 64;

/** The breadth every walk takes, and the queries a recall judges. */
static constexpr std::size_t dense_expansion_k = 128, dense_queries_k = 128;

/** Every row of @p vectors as the C interface takes them, @c f32 converted on the way in. */
static usearch_rows_view_t rows_of(nk::matrix<nk::f32_t> const& vectors) noexcept {
    return {vectors.data(), static_cast<std::size_t>(vectors.stride_bytes(0)), dense_dimensions_k,
            usearch_scalar_f32_k};
}

/** @p count rows from @p seed: sign sketches for bits, small integers for every other scalar. */
static nk::matrix<nk::f32_t> dense_rows(usearch_scalar_t scalar, std::size_t count, u64_t seed) {
    return scalar == usearch_scalar_u1x8_k ? make_sketched_vectors<nk::f32_t>(count, dense_dimensions_k, seed)
                                           : make_vectors<nk::f32_t>(count, dense_dimensions_k, seed);
}

/** Every row searches for itself: nine in ten must land first, every written key was minted, and
 *  every query's matches ascend. */
static std::size_t expect_self_recall(handle<dense_verbs_t>& index, usearch_rows_view_t rows, std::size_t count) {
    copied_matches_t found;
    usearch_traversal_stats_t stats {};
    std::size_t failures = check(search_queries(index, rows, count, cell_batch_k, cell_wanted_k, dense_expansion_k,
                                                usearch_traversal_walk_k, stats, found));
    if (failures) return failures;
    std::vector<std::size_t> const offsets = match_offsets(found);
    std::size_t found_first = 0, unminted = 0, descending = 0;
    for (std::size_t query = 0; query != count; ++query) {
        std::size_t const first = offsets[query], last = offsets[query + 1];
        for (std::size_t match = first; match != last; ++match) {
            std::optional<u64_t> const minted = offset_of_key(index.session->seed,
                                                              std::bit_cast<usearch_key_t>(found.keys[match]));
            unminted += !minted;
            found_first += match == first && minted == query;
            if (match != first) descending += found.scores[match] < found.scores[match - 1];
        }
    }
    return check(found_first * 10 >= count * 9) + check(unminted == 0) + check(descending == 0);
}

/** Every key of the first @p count entries reads one vector back. */
static std::size_t expect_gets(handle<dense_verbs_t>& index, std::size_t count) {
    std::size_t misread = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        usearch_rows_view_t vectors {};
        std::size_t written = 0;
        usearch_dense_get_t get {};
        get.index = index.get(), get.executor = index.executor(), get.outcome = &outcome;
        get.key = key_of_offset(index.session->seed, offset);
        get.vectors = &vectors, get.written = &written, get.arena = index.session->arena;
        get.outputs_scalar = usearch_scalar_f32_k;
        usearch_dense_get(&get);
        misread += outcome.status != usearch_status_success_k || written != 1;
    }
    return check(misread == 0);
}

/** Whether @p first sorts before @p second, byte by byte. */
static bool key_precedes(usearch_key_t const& first, usearch_key_t const& second) noexcept {
    return std::memcmp(first.bytes, second.bytes, sizeof(first.bytes)) < 0;
}

/** Admits the keys the sorted run of keys at @p admitted holds. */
static void admit_listed(usearch_key_t const* keys, usearch_admission_t* admissions, std::size_t count,
                         void* admitted) noexcept {
    std::vector<usearch_key_t> const& listed = *static_cast<std::vector<usearch_key_t> const*>(admitted);
    for (std::size_t position = 0; position != count; ++position)
        admissions[position] = std::binary_search(listed.begin(), listed.end(), keys[position], key_precedes)
                                   ? usearch_admit_k
                                   : usearch_reject_k;
}

/** A host walk writes only the keys of the even entries among the first @p count , chosen up front
 *  for the predicate to look up; a device walk refuses a host predicate as unsupported. */
static std::size_t expect_predicate(handle<dense_verbs_t>& index, usearch_rows_view_t rows, std::size_t count) {
    std::vector<usearch_key_t> admitted;
    for (std::size_t offset = 0; offset < count; offset += 2)
        admitted.push_back(key_of_offset(index.session->seed, offset));
    std::sort(admitted.begin(), admitted.end(), key_precedes);
    usearch_outcome_t outcome {};
    usearch_matches_view_t matches {};
    usearch_dense_search_t search {};
    search.index = index.get(), search.executor = index.executor(), search.outcome = &outcome;
    search.count = dense_queries_k, search.queries = rows, search.matches = &matches;
    search.arena = index.session->arena, search.wanted = cell_wanted_k, search.expansion_factor = dense_expansion_k;
    search.predicate = {admit_listed, &admitted, usearch_predicate_cost_cheap_k};
    usearch_dense_search(&search);
    if (index.residency == usearch_residency_vram_k) return check(outcome.status == usearch_status_unsupported_k);
    std::size_t failures = check(outcome);
    if (failures) return failures;
    copied_matches_t const found = copy_matches(matches, dense_queries_k);
    std::size_t outside = 0;
    for (uuid_t const& key : found.keys)
        outside += !std::binary_search(admitted.begin(), admitted.end(), std::bit_cast<usearch_key_t>(key),
                                       key_precedes);
    return check(outside == 0 && !found.keys.empty());
}

/** A RAM index compacts in place under a capacity of every entry and keeps every entry; the shards
 *  it emptied stay as idle slots, which the description counts among the live ones. */
static std::size_t expect_compact(handle<dense_verbs_t>& index, std::size_t count) {
    usearch_outcome_t outcome {};
    usearch_dense_index_t sources[1] = {index.get()};
    usearch_dense_compact_t compact {};
    compact.target = index.get(), compact.executor = index.executor(), compact.outcome = &outcome;
    compact.count = 1, compact.sources = sources, compact.shard_capacity = count;
    usearch_dense_compact(&compact);
    std::size_t const failures = check(outcome);
    return failures + check(describe(index).size == count);
}

/** One cell of the dense diagonal, named @p name, as the file's docblock describes. */
static std::size_t check_dense_behaviors(sweep_cell_t const& cell, settings_t const& settings,
                                         std::string const& name) {
    // A view holds a key table, so its key verbs and hints probe it rather than scan.
    settings_t cell_settings = settings_of(settings, cell);
    if (cell.residency == usearch_residency_nvme_k)
        cell_settings.nvme_ram.value = std::numeric_limits<std::size_t>::max();
    session_t session = open_session(cell_settings, cell.cpu_class);
    if (!session) return 1;
    u64_t const seed = stream_key(settings.seed, name);
    bool const binary = cell.scalar == usearch_scalar_u1x8_k;
    usearch_dense_metadata_t const geometry {binary ? usearch_dense_metric_hamming_k : usearch_dense_metric_angular_k,
                                             cell.scalar, dense_dimensions_k, 0};
    handle<dense_verbs_t> index = open_index<dense_verbs_t>(session, geometry, cell.shards);
    if (!index) return 1;
    nk::matrix<nk::f32_t> const vectors = dense_rows(cell.scalar, dense_rows_k, seed);
    usearch_rows_view_t const rows = rows_of(vectors);

    std::size_t failures = expect_built(cell, index, rows, dense_rows_k);
    if (failures) return failures;
    failures += expect_gets(index, dense_rows_k);
    failures += expect_self_recall(index, rows, dense_rows_k);
    failures += expect_walk_recall(index, rows, dense_queries_k, dense_expansion_k, 0.9);
    failures += expect_predicate(index, rows, dense_rows_k);
    failures += expect_hints(
        index, rows, dense_queries_k, dense_expansion_k,
        cell.residency == usearch_residency_vram_k ? usearch_status_unsupported_k : usearch_status_success_k);
    failures += expect_refined(index, rows, dense_queries_k, dense_expansion_k, usearch_status_success_k);
    failures += expect_health(index, dense_rows_k);
    failures += expect_cursor(index, dense_rows_k);
    failures += expect_sample(index, dense_rows_k);
    failures += expect_buffer_round_trip(index, rows, dense_queries_k, dense_expansion_k, 0.9,
                                         usearch_status_success_k);
    std::string const stem = fmt::format("/tmp/usearch-abi-dense-{:016x}", seed);
    failures += expect_persistence(cell, index, rows, rows, dense_queries_k, dense_expansion_k, 0.9, stem,
                                   saved_bytes_t::identical_k, 0.0);
    if (cell.residency == usearch_residency_nvme_k) return failures;
    failures += expect_refinement(index, rows, dense_rows_k, usearch_status_success_k);
    nk::matrix<nk::f32_t> const donated = dense_rows(cell.scalar, dense_donor_rows_k, ~seed);
    failures += expect_absorbed(index, rows_of(donated), dense_donor_rows_k, dense_rows_k, usearch_status_success_k);
    if (cell.residency == usearch_residency_ram_k) failures += expect_compact(index, dense_rows_k + dense_donor_rows_k);
    failures += expect_split(index, usearch_status_success_k);
    return failures + expect_cleared(index, rows);
}

std::size_t check_dense_abi(settings_t const& settings) {
    std::size_t failures = 0;
    if (settings.filter.selects("dense metric names"))
        failures += report("dense metric names", expect_metric_names<dense_verbs_t>());
    for (sweep_cell_t const& cell : diagonal_sweep(usearch_scalar_bf16_k)) {
        std::string const name = "dense " + name_of(cell);
        if (settings.filter.selects(name))
            failures += report(name.c_str(), check_dense_behaviors(cell, settings, name));
    }
    return failures;
}

} // namespace unum::usearch::test::abi
