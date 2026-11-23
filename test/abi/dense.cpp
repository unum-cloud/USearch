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
 *  audits, walks and samples and refuses every write; a refine plan holds the index closed while
 *  calls its handler cancels hold its round to the end; a second source gathers whole into the
 *  index, a RAM index reshards into one shard in place, and the index reshards into two targets
 *  with every key in one, each through a plan its cancelled calls hold to the end; a cleared index
 *  holds nothing and ends a cursor opened before it; every metric's name reads back.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstring> // `std::memcmp` - the keys' order

#include <algorithm> // `std::binary_search`, `std::sort` - the admitted keys
#include <string>    // `std::string` - the cells' names and checkpoint stems
#include <vector>    // `std::vector` - the keys

#include "usearch/dense_index.h"

#include "harness.hpp"

namespace unum::usearch::test::abi {

/** Rows each cell builds, rows a second source adds, and the columns of each. */
static constexpr std::size_t dense_rows_k = 1024, dense_packed_rows_k = 256, dense_dimensions_k = 64;

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
static void expect_self_recall(handle<dense_verbs_t>& index, usearch_rows_view_t rows, std::size_t count) {
    copied_matches_t found;
    usearch_traversal_stats_t stats {};
    if (!expect_success(search_queries(index, rows, count, cell_batch_k, cell_wanted_k, dense_expansion_k,
                                       usearch_traversal_walk_k, stats, found)))
        return;
    st::vector<std::size_t> const offsets = match_offsets(found);
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
    expect_true(found_first * 10 >= count * 9);
    expect_true(unminted == 0);
    expect_true(descending == 0);
}

/** Every key of the first @p count entries reads one vector back. */
static void expect_gets(handle<dense_verbs_t>& index, std::size_t count) {
    std::size_t misread = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        usearch_rows_view_t vectors {};
        std::size_t written = 0;
        usearch_dense_get_t get {};
        get.index = index.get(), get.executor = index.executor(), get.outcome = &outcome;
        get.key = key_of_offset(index.session->seed, offset);
        get.vectors = &vectors, get.written = &written, get.arena = index.arena();
        get.outputs_scalar = usearch_scalar_f32_k;
        usearch_dense_get(&get);
        misread += outcome.status != usearch_status_success_k || written != 1;
    }
    expect_true(misread == 0);
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
static void expect_predicate(handle<dense_verbs_t>& index, usearch_rows_view_t rows, std::size_t count) {
    std::vector<usearch_key_t> admitted;
    for (std::size_t offset = 0; offset < count; offset += 2)
        admitted.push_back(key_of_offset(index.session->seed, offset));
    std::sort(admitted.begin(), admitted.end(), key_precedes);
    usearch_outcome_t outcome {};
    usearch_matches_view_t matches {};
    usearch_dense_search_t search {};
    search.index = index.get(), search.executor = index.executor(), search.outcome = &outcome;
    search.count = dense_queries_k, search.queries = rows, search.matches = &matches;
    search.arena = index.arena(), search.wanted = cell_wanted_k, search.expansion_factor = dense_expansion_k;
    search.predicate = {admit_listed, &admitted, usearch_predicate_cost_cheap_k};
    usearch_dense_search(&search);
    if (index.residency == usearch_residency_vram_k) return void(expect_status(outcome, usearch_status_unsupported_k));
    if (!expect_success(outcome)) return;
    copied_matches_t const found = copy_matches(matches, dense_queries_k);
    std::size_t outside = 0;
    for (uuid128_t const& key : found.keys)
        outside += !std::binary_search(admitted.begin(), admitted.end(), std::bit_cast<usearch_key_t>(key),
                                       key_precedes);
    expect_true(outside == 0 && !found.keys.empty());
}

/** A RAM index reshards in place, as its own only source and target, under a capacity of every
 *  entry through a plan every call of which is cancelled, and keeps every entry, which the plan's
 *  finish audits; the description then counts each shard it emptied among the live ones, as an
 *  idle slot of its own. */
static void expect_reshard_in_place(handle<dense_verbs_t>& index, std::size_t count) {
    usearch_outcome_t outcome {};
    usearch_ram_options_t ram {};
    ram.shard_capacity = count;
    usearch_refine_report_t report {};
    usearch_dense_reshard_plan_t plan = nullptr;
    usearch_dense_index_t const self = index.get();
    drive_cancelled_plan(plan, outcome, count, [&](usearch_progress_t progress, usearch_phase_progress_t* phase) {
        usearch_dense_reshard_t reshard {};
        reshard.sources_count = 1, reshard.sources = &self, reshard.targets_count = 1, reshard.targets = &self;
        reshard.executor = index.executor(), reshard.outcome = &outcome;
        reshard.ram = &ram, reshard.report = &report, reshard.progress = progress;
        reshard.plan = &plan, reshard.phase = phase;
        usearch_dense_reshard(&reshard);
    });
    expect_success(outcome);
    expect_true(plan != nullptr);
    usearch_dense_reshard_plan_finish_t finish {};
    finish.executor = index.executor(), finish.outcome = &outcome;
    finish.plan = plan, finish.report = &report;
    usearch_dense_reshard_plan_finish(&finish);
    expect_success(outcome);
    expect_true(report.reachability.audited == count);
    expect_true(describe(index).size == count);
}

/** Walks as narrow as their rankings answer the first @p count queries alike on the session's
 *  executor and on the calling thread alone, since a walk's landmarks follow its row. */
static void expect_repeatable_walks(handle<dense_verbs_t>& index, usearch_rows_view_t rows, std::size_t count) {
    copied_matches_t pooled, alone;
    usearch_traversal_stats_t stats {};
    expect_success(search_queries(index, rows, count, count, cell_wanted_k, cell_wanted_k, usearch_traversal_walk_k,
                                  stats, pooled));
    usearch_outcome_t outcome {};
    usearch_matches_view_t matches {};
    dense_verbs_t::search(index.get(), nullptr, index.arena(), count, rows, cell_wanted_k, cell_wanted_k,
                          usearch_traversal_walk_k, &matches, &stats, &outcome);
    if (!expect_success(outcome) || !expect_true(append_matches(matches, count, alone))) return;
    expect_true(std::ranges::equal(pooled.keys, alone.keys));
}

/** An executor whose threads may hold a page of scratch refuses an insert that needs more as out
 *  of memory, rather than growing past it. */
static void test_scratch_limit(u64_t seed) {
    usearch_outcome_t outcome {};
    usearch_executor_t executor = nullptr;
    usearch_executor_for_ram_t spawn {};
    spawn.executor = &executor, spawn.outcome = &outcome, spawn.threads = 1, spawn.scratch_limit = 4096;
    usearch_executor_for_ram(&spawn);
    if (!expect_success(outcome)) return;
    usearch_dense_index_t index = nullptr;
    usearch_dense_init_ram_t init {};
    init.index = &index, init.outcome = &outcome, init.metric = usearch_dense_metric_angular_k;
    init.quantization = usearch_scalar_bf16_k, init.dimensions = dense_dimensions_k;
    usearch_dense_init_ram(&init);
    nk::matrix<nk::f32_t> const vectors = dense_rows(usearch_scalar_bf16_k, dense_queries_k, seed);
    std::vector<usearch_key_t> keys(dense_queries_k);
    for (std::size_t offset = 0; offset != dense_queries_k; ++offset) keys[offset] = key_of_offset(seed, offset);
    usearch_dense_insert_t insert {};
    insert.index = index, insert.executor = executor, insert.outcome = &outcome;
    insert.count = dense_queries_k, insert.keys = {keys.data(), sizeof(usearch_key_t)};
    insert.vectors = rows_of(vectors);
    if (expect_success(outcome)) usearch_dense_insert(&insert);
    expect_status(outcome, usearch_status_out_of_memory_k);
    usearch_dense_free(index);
    usearch_executor_free(executor);
}

/** One cell of the dense diagonal, as the file's docblock describes. */
static void test_dense_behaviors(sweep_cell_t const& cell, u64_t seed) {
    // A view holds a key table, so its key verbs and hints probe it rather than scan.
    settings_t cell_settings = settings_of(session_settings(), cell);
    if (cell.residency == usearch_residency_nvme_k)
        cell_settings.nvme_ram.value = (std::numeric_limits<std::size_t>::max)();
    session_t session = open_session(cell_settings, cell.cpu_class);
    if (!expect_true(bool(session))) return;
    bool const binary = cell.scalar == usearch_scalar_u1x8_k;
    usearch_dense_metadata_t const geometry {binary ? usearch_dense_metric_hamming_k : usearch_dense_metric_angular_k,
                                             cell.scalar, dense_dimensions_k, 0};
    handle<dense_verbs_t> index = open_index<dense_verbs_t>(session, geometry, cell.shards);
    if (!expect_true(bool(index))) return;
    nk::matrix<nk::f32_t> const vectors = dense_rows(cell.scalar, dense_rows_k, seed);
    usearch_rows_view_t const rows = rows_of(vectors);

    if (!check_built(cell, index, rows, dense_rows_k)) return;
    expect_gets(index, dense_rows_k);
    expect_self_recall(index, rows, dense_rows_k);
    expect_walk_recall(index, rows, dense_queries_k, dense_expansion_k, 0.9);
    if (cell.residency == usearch_residency_ram_k) expect_repeatable_walks(index, rows, dense_queries_k);
    expect_predicate(index, rows, dense_rows_k);
    expect_hints(index, rows, dense_queries_k, dense_expansion_k,
                 cell.residency == usearch_residency_vram_k ? usearch_status_unsupported_k : usearch_status_success_k);
    expect_refined(index, rows, dense_queries_k, dense_expansion_k, usearch_status_success_k);
    expect_health(index, dense_rows_k);
    expect_cursor(index, dense_rows_k);
    expect_sample(index, dense_rows_k);
    expect_buffer_round_trip(index, rows, dense_queries_k, dense_expansion_k, 0.9, usearch_status_success_k);
    scratch_directory_t const scratch;
    std::string const stem = scratch.path("checkpoint");
    expect_persistence(cell, index, rows, rows, dense_queries_k, dense_expansion_k, 0.9, stem,
                       saved_bytes_t::identical_k, 0.0);
    if (cell.residency == usearch_residency_nvme_k) return;
    expect_refine_plan(index, rows, dense_rows_k, usearch_status_success_k);
    nk::matrix<nk::f32_t> const second_source = dense_rows(cell.scalar, dense_packed_rows_k, ~seed);
    expect_reshard_gather(index, rows_of(second_source), dense_packed_rows_k, dense_rows_k, usearch_status_success_k);
    if (cell.residency == usearch_residency_ram_k) expect_reshard_in_place(index, dense_rows_k + dense_packed_rows_k);
    expect_reshard_cut(index, usearch_status_success_k);
    expect_cleared(index, rows);
}

void run_dense_abi_cases() {
    run_test_case("test_metric_names<dense_verbs_t>", test_metric_names<dense_verbs_t>);
    run_test_case("usearch_executor_for_ram_t::scratch_limit", test_scratch_limit,
                  stream_key(test_run.settings.seed, "usearch_executor_for_ram_t::scratch_limit"));
    for (sweep_cell_t const& cell : diagonal_sweep(usearch_scalar_bf16_k)) {
        std::string const name = fmt::format("dense {}", name_of(cell));
        // The stream keeps the name the cell had before its label spelled the constants.
        std::string const label = fmt::format("test_dense_behaviors({}, {}, {}, {})", constant_of(cell.residency),
                                              constant_of(cell.scalar), constant_of(cell.cpu_class), cell.shards);
        run_test_case(label.c_str(), test_dense_behaviors, cell, stream_key(test_run.settings.seed, name));
    }
}

} // namespace unum::usearch::test::abi
