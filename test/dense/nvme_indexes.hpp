/**
 *  @file test/dense/nvme_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The suite composers for @c dense::nvme_indexes and its rewrites: two-shard
 *      @c dense::ram_indexes saved as SafeGraphs parts and viewed back from storage, once per
 *      I/O engine the host admits.
 *
 *  Parts go under @c TMPDIR, so a run that points it at a volume enforcing direct-I/O alignment
 *  checks every covering read against it. A walk as wide as the index scores every entry, so the
 *  RAM group the parts came from judges every verb, as do the @c f64 oracles of @c dense.cuh.
 *
 *  Reshard, refine and build view each target back and judge it against a fresh RAM build of the
 *  same entries. Recall is the share of every query's results whose exact distance is within its
 *  exact k-th nearest, so ties never count against a ranking. A rewrite passes when its target,
 *  searched with the default walk, recalls within @ref rewrite_recall_margin_k of a fresh RAM build
 *  of the same entries searched alike.
 */
#pragma once

#include <cstdint> // `std::uintptr_t`, for the checkpoint paths' per-run salt
#include <cstdio>  // `std::remove`, `std::rename`, `std::snprintf`, for the checkpoint's files

#include <algorithm>    // `std::is_sorted`, `std::nth_element`
#include <array>        // `std::array`
#include <bit>          // `std::bit_cast`, for `uuid128_t` keys from numbers
#include <filesystem>   // `std::filesystem::temp_directory_path`, the directory parts go into
#include <memory>       // `std::unique_ptr`, for the rewrites' checkpoints
#include <string>       // `std::string`, for the relocation's directory
#include <string_view>  // `std::string_view`
#include <system_error> // `std::error_code`, for the temporary directory
#include <vector>       // `std::vector`

#include <fmt/format.h>

#include "usearch/dense/nvme_reshard.hpp"
#include "usearch/dense/ram_indexes.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** The dense NVMe view over the kits the ladders pick for @p dtype_ under @p caps_, as @c src/
 *  spells them for a CPU class. */
template <typename key_type_, nk_dtype_t dtype_, nk_capability_t caps_,
          typename engine_type_ = default_io_engine<fu::standard_yield_t>>
using nvme_indexes_for = nvme_indexes<
    key_type_, typename quads_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
    typename tiles_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t, typename elementwise_ladder<caps_>::kit_t,
    typename rmscast_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
    typename visits_hash_set_ladder<caps_>::kit_t, engine_type_, fu::standard_atomic_ref, fu::standard_yield_t>;

/** Whether a sharded index reads its parts from storage: the NVMe residency. */
template <typename indexes_type_>
concept nvme_collection =
    std::is_same_v<typename indexes_type_::executor_t, nvme_executor<typename indexes_type_::executor_t::engine_t>>;

/** Reads every engine of an NVMe scenario keeps outstanding: small enough that a walk's flights
 *  outnumber it, so the staged reads wait for completions. */
constexpr std::size_t nvme_queue_depth_k = 32;

/** A queue a lone query's walks leave mostly idle at one hop each, so they keep several in flight,
 *  while a batch of a few dozen rows on two threads fills it at one hop per walk. */
constexpr std::size_t nvme_deep_queue_depth_k = 8 * nvme_queue_depth_k;

/** Where one NVMe scenario's checkpoint goes, inside a scratch directory removed with it. */
struct checkpoint_files_t {
    scratch_directory_t directory;

    /** Where a save of several shards writes its manifest, its parts beside it. */
    std::string manifest;

    /** Where a save of one shard writes its monolith. */
    std::string monolith;

    /** Parts the manifest names. */
    std::size_t parts = 0;

    explicit checkpoint_files_t(std::string_view scenario)
        : manifest(directory.path(fmt::format("{}-parts.safegraphs", scenario))),
          monolith(directory.path(fmt::format("{}.safegraphs", scenario))) {}

    /** The path of part @p part beside the manifest, formatted into @p buffer. */
    std::string_view part_path(growing_buffer<char>& buffer, std::size_t part) const noexcept {
        auto const [path, status] = safegraphs_part_path(buffer, safegraphs_stem(manifest.data()), part, parts);
        return succeeded(status) ? path : std::string_view {};
    }
};

/** One batch's rankings as a search writes them: @c wanted keys and distances per query, and each
 *  query's count. */
template <typename nvme_type_>
struct nvme_rankings {
    using key_t = typename nvme_type_::key_t;
    using distance_t = typename nvme_type_::distance_t;

    std::size_t wanted = 0;
    std::vector<key_t> keys;
    std::vector<distance_t> distances;
    std::vector<std::size_t> counts;

    nvme_rankings(std::size_t queries_count, std::size_t wanted)
        : wanted(wanted), keys(queries_count * wanted), distances(queries_count * wanted), counts(queries_count) {}

    /** Ranks @p queries on @p indexes through @p executor, each walk @p expansion_factor wide, zero
     *  meaning the index's default; a RAM group also takes its @p traversals, one per thread. */
    template <typename indexes_type_, typename executor_type_, typename... traversals_types_>
    expected<typename indexes_type_::search_result_t> search(indexes_type_& indexes,
                                                             typename nvme_type_::vectors_view_t queries,
                                                             executor_type_& executor, std::size_t expansion_factor,
                                                             traversals_types_... traversals) {
        typename indexes_type_::search_config_t config {};
        config.expansion_factor = static_cast<decltype(config.expansion_factor)>(expansion_factor);
        return indexes.search(queries, matches(), config, executor, traversals...);
    }

    nk::matrix_span<key_t> keys_span() noexcept { return {keys.data(), counts.size(), wanted}; }
    nk::matrix_span<distance_t> distances_span() noexcept { return {distances.data(), counts.size(), wanted}; }
    ranked_matches_span<key_t, distance_t> matches() noexcept {
        return {keys_span(), distances_span(), std::span {counts}};
    }

    std::span<key_t const> keys_of(std::size_t query) const noexcept {
        return {keys.data() + query * wanted, counts[query]};
    }
    std::span<distance_t const> distances_of(std::size_t query) const noexcept {
        return {distances.data() + query * wanted, counts[query]};
    }
};

/** The RAM group an NVMe view is judged against: @p rows rows keyed @c 1+row unless @p key_of says
 *  otherwise, built under @p kernel over two shards and saved under @c TMPDIR as the @p scenario
 *  checkpoint by @ref view, or once more over one shard as a monolith by @ref view_monolith. */
template <typename nvme_type_>
struct nvme_reference {
    using nvme_t = nvme_type_;
    using key_t = typename nvme_t::key_t;
    using scalar_t = typename nvme_t::scalar_t;
    using ram_t =
        ram_indexes<key_t, typename nvme_t::quads_kit_t, typename nvme_t::tiles_kit_t,
                    typename nvme_t::elementwise_kit_t, typename nvme_t::rmscast_kit_t, typename nvme_t::visits_kit_t>;

    /** Whether every check since the fixture was built held. */
    checks_held_t held;
    dense_indexes_executor<ram_t> builder;
    ram_t built;
    nk::matrix<scalar_t> vectors;
    std::vector<key_t> keys;
    checkpoint_files_t checkpoint;

    template <typename key_of_type_ = key_t (*)(std::size_t) noexcept>
    nvme_reference(std::string_view scenario, nk_kernel_kind_t kernel, std::size_t rows, std::size_t dimensions,
                   u64_t seed, key_of_type_&& key_of = &default_key_) noexcept
        : built(kernel, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}),
          vectors(make_vectors<scalar_t>(rows, dimensions, stream_key(seed, scenario))), keys(rows),
          checkpoint(scenario),
          whole_(kernel, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}) {
        for (std::size_t row = 0; row != rows; ++row) keys[row] = key_of(row);
        if (failed(builder.status)) {
            expect_true(false);
            return;
        }
        dense_indexes_build(built, builder.executor, vectors, keys);
    }

    /** Saves the group's manifest and parts at @p alignment and views them through @p viewed. */
    void view(nvme_t& viewed, std::size_t alignment = safegraphs_alignment_default_k) noexcept {
        save_config_t config;
        config.alignment = alignment;
        checkpoint.parts = built.shards_count();
        expect_true(succeeded(built.save(checkpoint.manifest.data(), config, builder.executor)));
        expect_equal(viewed.view(checkpoint.manifest.data()), status_t::success_k);
        hold(viewed);
    }

    /** Holds @p viewed 's key table, unbounded, so its key verbs probe it rather than refuse. */
    template <typename viewed_type_>
    void hold(viewed_type_& viewed) noexcept {
        dense_indexes_executor<viewed_type_> holding;
        expect_true(succeeded(holding.status) &&
                    succeeded(viewed.hold_keys(forkunion_allocator_t {no_memory_domain_k}, 0, holding.executor)));
    }

    /** The rows once more as one shard, whose save is a lone monolith, built on first use. */
    ram_t& whole() noexcept {
        if (whole_.shards_count()) return whole_;
        expect_true(succeeded(whole_.emplace(no_memory_domain_k)));
        expect_true(succeeded(whole_.reserve({keys.size()}).status));
        expect_true(succeeded(whole_.insert(std::span<key_t const> {keys}, vectors.view(), builder.executor).status));
        return whole_;
    }

    /** Saves @ref whole as a monolith and views it through @p viewed. */
    void view_monolith(nvme_t& viewed) noexcept {
        expect_true(succeeded(whole().save(checkpoint.monolith.data(), save_config_t {}, builder.executor)));
        expect_equal(viewed.view(checkpoint.monolith.data()), status_t::success_k);
        hold(viewed);
    }

    /** The RAM group's rankings of @p queries under a walk as wide as the index, which ranks every
     *  row. */
    nvme_rankings<nvme_t> exhaustive_rankings(typename nvme_t::vectors_view_t queries, std::size_t wanted) {
        nvme_rankings<nvme_t> ranked {queries.extent(0), wanted};
        expect_true(succeeded(
            ranked.search(built, queries, builder.executor, vectors.extent(0), to_writable_span(builder.traversals))
                .status));
        return ranked;
    }

    /** The row a key names. */
    static std::size_t row_of(key_t key) noexcept { return static_cast<std::size_t>(dense_key_number(key)) - 1; }

    /** The first @p count rows, which the scenarios query with. */
    nk::matrix_view<scalar_t> leading_rows(std::size_t count) const noexcept {
        nk::matrix_view<scalar_t> const rows = vectors.view();
        return rows(nk::range(std::size_t {0}, count), nk::slice);
    }

  private:
    ram_t whole_;

    static key_t default_key_(std::size_t row) noexcept { return dense_key<key_t>(1 + row); }
};

/** Every query of @p found ranks like @p wanted: each key a distinct entry at its own exact
 *  distance from the query, so ties may swap keys but never distances. */
template <typename reference_type_, typename rankings_type_>
void nvme_expect_rankings(reference_type_ const& reference, typename reference_type_::nvme_t::vectors_view_t queries,
                          rankings_type_ const& found, rankings_type_ const& wanted) {
    using nvme_t = typename reference_type_::nvme_t;
    using scalar_t = typename nvme_t::scalar_t;
    error_stats_t exact_errors, wanted_errors;
    for (std::size_t query = 0; query != found.counts.size(); ++query) {
        auto const found_keys = found.keys_of(query);
        auto const found_distances = found.distances_of(query), wanted_distances = wanted.distances_of(query);
        expect_equal(found_keys.size(), wanted_distances.size());
        std::vector<bool> found_rows(reference.keys.size());
        auto const query_vector = queries.row(query).as_vector();
        auto const query_norms = norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(query_vector);
        for (std::size_t rank = 0; rank != min_of(found_keys.size(), wanted_distances.size()); ++rank) {
            std::size_t const row = reference_type_::row_of(found_keys[rank]);
            expect_true(row < found_rows.size() && !found_rows[row]);
            if (row >= found_rows.size() || found_rows[row]) continue;
            found_rows[row] = true;
            auto const stored = reference.vectors.view().row(row).as_vector();
            auto const stored_norms = norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(stored);
            f32_t const exact = distance<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(
                reference.built.kernel(), query_vector, query_norms, stored, stored_norms);
            exact_errors.add(found_distances[rank], exact);
            wanted_errors.add(found_distances[rank], wanted_distances[rank]);
        }
    }
    expect_errors("ranked distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k, exact_errors,
                  0);
    expect_errors("ranked against wanted", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                  wanted_errors, 0);
}

/** The NVMe audit of parts reports what the RAM group they came from reports: the same root's
 *  component, as many levels, the same in-link tallies and depth, and the same islands. */
inline void nvme_expect_reachability(reachability_t const& nvme, reachability_t const& ram) {
    expect_equal(nvme.audited_nodes, ram.audited_nodes);
    expect_equal(nvme.reachable_nodes, ram.reachable_nodes);
    expect_equal(nvme.depth_from_root, ram.depth_from_root);
    expect_equal(nvme.unlinked, ram.unlinked);
    expect_equal(nvme.underlinked, ram.underlinked);
    expect_equal(nvme.beyond_most_populated_depth, ram.beyond_most_populated_depth);
    expect_equal(nvme.unreached_islands, ram.unreached_islands);
    expect_equal(nvme.stranded_islands, ram.stranded_islands);
    expect_equal(nvme.components, ram.components);
}

/** Search on NVMe against the RAM group its parts came from: under a walk as wide as the index both
 *  rank every entry, so every query's ranking matches rank for rank, each key at its exact
 *  distance; a default walk keeps its candidates ascending over distinct entries at their exact
 *  distances. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_search(u64_t seed) {
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 8, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_search", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed);
    dense_indexes_executor<nvme_type_> driver;
    expect_equal(viewed.size(), rows_count);
    expect_equal(viewed.shards_count(), reference.built.shards_count());
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    auto const [wide_result, wide_status] = nvme.search(viewed, queries, driver.executor, rows_count);
    expect_equal(wide_status, status_t::success_k);
    expect_equal(wide_result.computed_distances, queries_count * rows_count);
    expect_equal(wide_result.value_reads.count, wide_result.computed_distances);
    expect_true(nvme.counts == ram.counts);
    nvme_expect_rankings(reference, queries, nvme, ram);

    // A default walk reaches far fewer entries, yet whatever it ranks it ranks exactly.
    auto const [narrow_result, narrow_status] = nvme.search(viewed, queries, driver.executor, 0);
    expect_equal(narrow_status, status_t::success_k);
    expect_true(narrow_result.computed_distances < queries_count * rows_count);
    nvme_expect_rankings(reference, queries, nvme, nvme);
    for (std::size_t query = 0; query != queries_count; ++query) {
        expect_equal(nvme.counts[query], wanted);
        expect_true(std::is_sorted(nvme.distances_of(query).begin(), nvme.distances_of(query).end()));
    }
}

/** Hints on NVMe seed the walks as on RAM once the view holds a key table: without one, hints and
 *  key lookups refuse, and a limit short of the table refuses to build it. With it, lookups count
 *  every key exactly, both residencies count the same keys held nowhere, a hint of each query's
 *  exact nearest ranks it first on both at a narrow breadth, a warm second call ranks alike, and
 *  MaxSim rows take their query's hints alike. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_hints(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using distance_t = typename nvme_type_::distance_t;
    using ram_t = typename nvme_reference<nvme_type_>::ram_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 8, wanted = 4, hint_columns = 2;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_hints", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed);
    dense_indexes_executor<nvme_type_> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    // Each query's exact nearest, then a key held nowhere.
    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> const exact = reference.exhaustive_rankings(queries, 1);
    auto hint_keys = nk::matrix<key_t>::zeros({queries_count, hint_columns}).value;
    std::vector<std::size_t> const hint_counts(queries_count, hint_columns);
    for (std::size_t query = 0; query != queries_count; ++query) {
        std::span<key_t const> const nearest_keys = exact.keys_of(query);
        hint_keys(query, 0) = nearest_keys[0];
        hint_keys(query, 1) = dense_key<key_t>(rows_count + 1 + query);
    }
    ranked_hints_view<key_t> const hints {hint_keys.view(), std::span<std::size_t const> {hint_counts}};
    typename ram_t::search_config_t ram_config {};
    typename nvme_type_::search_config_t nvme_config {};
    ram_config.expansion_factor = nvme_config.expansion_factor = 16;
    nvme_rankings<nvme_type_> ram {queries_count, wanted}, nvme {queries_count, wanted}, warm {queries_count, wanted};
    std::span<key_t const> const probed = hints_of(hints, 0);
    std::array<std::size_t, hint_columns> entries {};
    nvme_type_ keyless {forkunion_allocator_t {no_memory_domain_k}};
    expect_equal(keyless.view(reference.checkpoint.manifest.data()), status_t::success_k);
    expect_equal(keyless.search(queries, nvme.matches(), nvme_config, driver.executor, {}, hints).status,
                 status_t::unsupported_k);
    expect_equal(keyless.count(probed, std::span {entries}, driver.executor), status_t::unsupported_k);
    expect_equal(keyless.hold_keys(forkunion_allocator_t {no_memory_domain_k}, 1, driver.executor),
                 status_t::bad_alloc_k);
    expect_equal(viewed.hold_keys(forkunion_allocator_t {no_memory_domain_k}, 0, driver.executor), status_t::success_k);
    expect_equal(viewed.count(probed, std::span {entries}, driver.executor), status_t::success_k);
    expect_equal(entries[0], 1u);
    expect_equal(entries[1], 0u);

    auto const [ram_result, ram_status] = reference.built.search(queries, ram.matches(), ram_config,
                                                                 reference.builder.executor,
                                                                 to_writable_span(reference.builder.traversals), hints);
    auto const [nvme_result, nvme_status] = viewed.search(queries, nvme.matches(), nvme_config, driver.executor, {},
                                                          hints);
    auto const [warm_result, warm_status] = viewed.search(queries, warm.matches(), nvme_config, driver.executor, {},
                                                          hints);
    expect_equal(ram_status, status_t::success_k);
    expect_equal(nvme_status, status_t::success_k);
    expect_equal(warm_status, status_t::success_k);
    expect_equal(ram_result.missing_hints, queries_count);
    expect_equal(nvme_result.missing_hints, queries_count);
    expect_equal(warm_result.missing_hints, queries_count);
    expect_true(warm.counts == nvme.counts);
    nvme_expect_rankings(reference, queries, warm, nvme);
    for (std::size_t query = 0; query != queries_count; ++query) {
        expect_equal(nvme.counts[query], wanted);
        expect_equal(ram.counts[query], wanted);
        if (!nvme.counts[query] || !ram.counts[query]) continue;
        std::span<distance_t const> const exact_distances = exact.distances_of(query),
                                          nvme_distances = nvme.distances_of(query),
                                          ram_distances = ram.distances_of(query);
        f64_t const nearest = exact_distances[0], tolerance = 1e-3 * max_of(std::fabs(nearest), 1.0);
        expect_true(nvme_distances[0] <= nearest + tolerance);
        expect_true(ram_distances[0] <= nearest + tolerance);
    }

    // MaxSim queries of two rows each take the first row's hints for both rows' walks, and count
    // the key held nowhere once per query on both residencies.
    if constexpr (kernel_ == nk_kernel_dot_k) {
        constexpr std::size_t grouped_count = queries_count / 2;
        std::array<std::size_t, grouped_count + 1> query_offsets;
        for (std::size_t query = 0; query != query_offsets.size(); ++query) query_offsets[query] = 2 * query;
        auto grouped_keys = nk::matrix<key_t>::zeros({grouped_count, hint_columns}).value;
        for (std::size_t query = 0; query != grouped_count; ++query)
            grouped_keys(query, 0) = hint_keys(2 * query, 0), grouped_keys(query, 1) = hint_keys(2 * query, 1);
        std::vector<std::size_t> const grouped_counts(grouped_count, hint_columns);
        ranked_hints_view<key_t> const grouped {grouped_keys.view(), std::span<std::size_t const> {grouped_counts}};
        std::vector<key_t> keys(grouped_count * wanted);
        std::vector<f64_t> scores(grouped_count * wanted);
        std::array<std::size_t, grouped_count + 1> offsets {};
        std::array<std::size_t, grouped_count> scored {};
        products_tape<key_t, f64_t> const matches {std::span {keys}, std::span {scores}, std::span {offsets}, wanted};
        auto const [ram_maxsim, ram_maxsim_status] = reference.built.maxsim(
            queries, std::span<std::size_t const> {query_offsets}, std::span {scored}, matches, {ram_config, wanted},
            reference.builder.executor, to_writable_span(reference.builder.traversals), grouped);
        expect_equal(ram_maxsim_status, status_t::success_k);
        expect_equal(ram_maxsim.missing_hints, grouped_count);
        offsets[0] = 0;
        auto const [nvme_maxsim, nvme_maxsim_status] = viewed.maxsim(
            queries, std::span<std::size_t const> {query_offsets}, std::span {scored}, matches, {nvme_config, wanted},
            driver.executor, {}, grouped);
        expect_equal(nvme_maxsim_status, status_t::success_k);
        expect_equal(nvme_maxsim.missing_hints, grouped_count);
        for (std::size_t query = 0; query != grouped_count; ++query)
            expect_true(offsets[query + 1] - offsets[query] == min_of(scored[query], wanted) && scored[query]);
    }
}

/** Expects @p index to rank @p queries under one gathering predicate per thread of @p executor,
 *  each carrying @p payload_type_ and admitting what @p per_key admits, as under @p per_key asked a
 *  quad at a time; a RAM group also takes its @p traversals, one per thread. */
template <typename payload_type_, typename reference_type_, typename index_type_, typename executor_type_,
          typename per_key_type_, typename... traversals_types_>
void nvme_expect_gathered(reference_type_ const& reference, index_type_& index, executor_type_& executor,
                          typename index_type_::search_config_t config,
                          typename reference_type_::nvme_t::vectors_view_t queries, per_key_type_ per_key,
                          std::size_t wanted, traversals_types_... traversals) {
    using nvme_t = typename reference_type_::nvme_t;
    std::size_t const queries_count = queries.extent(0);
    nvme_rankings<nvme_t> immediate {queries_count, wanted}, gathered {queries_count, wanted};
    auto family = gathering_per_thread<typename nvme_t::key_t, payload_type_>(per_key, executor.size());
    expect_true(succeeded(
        index.search_if(queries, immediate.matches(), config, quad_predicate(per_key), executor, traversals...)
            .status));
    expect_true(
        succeeded(index.search_if(queries, gathered.matches(), config, family, executor, traversals...).status));
    expect_true(gathered.counts == immediate.counts);
    nvme_expect_rankings(reference, queries, gathered, immediate);
    for (std::size_t query = 0; query != queries_count; ++query)
        for (typename nvme_t::key_t const key : gathered.keys_of(query))
            expect_true(per_key(key) == admission_t::admit_k);
}

/** Filtered search on NVMe against the RAM group: a per-key predicate reads every scored entry's
 *  key, and under a walk as wide as the index both rank every admitted entry alike, whether it
 *  answers each quad or gathers per thread until each round settles. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_filtered(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using candidates_count_t = typename nvme_type_::index_t::candidates_count_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 4, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_filtered", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed);
    dense_indexes_executor<nvme_type_> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    auto const is_odd = [](key_t key) noexcept {
        return dense_key_number(key) % 2 ? admission_t::admit_k : admission_t::reject_k;
    };
    auto const odd = quad_predicate(is_odd);
    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> ram {queries_count, wanted}, nvme {queries_count, wanted};
    expect_true(succeeded(reference.built
                              .search_if(queries, ram.matches(), {static_cast<candidates_count_t>(rows_count)}, odd,
                                         reference.builder.executor, to_writable_span(reference.builder.traversals))
                              .status));
    expect_true(succeeded(
        viewed.search_if(queries, nvme.matches(), {static_cast<candidates_count_t>(rows_count)}, odd, driver.executor)
            .status));
    expect_true(nvme.counts == ram.counts);
    nvme_expect_rankings(reference, queries, nvme, ram);
    for (std::size_t query = 0; query != queries_count; ++query) {
        expect_equal(nvme.counts[query], wanted);
        for (key_t const key : nvme.keys_of(query)) expect_true(dense_key_number(key) % 2);
    }
    using ram_t = typename nvme_reference<nvme_type_>::ram_t;
    typename ram_t::search_config_t ram_config {static_cast<candidates_count_t>(rows_count)};
    typename nvme_type_::search_config_t nvme_config {static_cast<candidates_count_t>(rows_count)};
    nvme_expect_gathered<typename ram_t::walk_payload_t>(reference, reference.built, reference.builder.executor,
                                                         ram_config, queries, is_odd, wanted,
                                                         to_writable_span(reference.builder.traversals));
    nvme_expect_gathered<typename nvme_type_::walk_payload_t>(reference, viewed, driver.executor, nvme_config, queries,
                                                              is_odd, wanted);
    ram_config.traversal = traversal_t::scan_k, nvme_config.traversal = traversal_t::scan_k;
    nvme_expect_gathered<typename ram_t::scan_payload_t>(reference, reference.built, reference.builder.executor,
                                                         ram_config, queries, is_odd, wanted,
                                                         to_writable_span(reference.builder.traversals));
    nvme_expect_gathered<typename nvme_type_::scan_payload_t>(reference, viewed, driver.executor, nvme_config, queries,
                                                              is_odd, wanted);

    // One key in eight still fills every query at the default breadth, as each walk stops on what
    // it admitted rather than on what it scored.
    auto const one_in_eight = quad_predicate(
        [](key_t key) noexcept { return dense_key_number(key) % 8 ? admission_t::reject_k : admission_t::admit_k; });
    expect_true(succeeded(viewed.search_if(queries, nvme.matches(), {}, one_in_eight, driver.executor).status));
    for (std::size_t query = 0; query != queries_count; ++query) {
        expect_equal(nvme.counts[query], wanted);
        for (key_t const key : nvme.keys_of(query)) expect_true(dense_key_number(key) % 8 == 0);
    }
}

/** The key scans against the RAM group's lookups, a key shared by every third entry and absent keys
 *  included: every count, verdict and true total agrees, every key's exported entries are its
 *  earliest in placement order, as @c read_entries reads them, and a narrower output is refused. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_key_scans(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, wanted = 2;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    key_t const shared = dense_key<key_t>(100000);

    auto const key_of = [&](std::size_t row) noexcept { return row % 3 ? dense_key<key_t>(1 + row) : shared; };
    nvme_reference<nvme_type_> reference {"nvme_scans", kernel_, rows_count, dimensions, seed, key_of};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed);
    dense_indexes_executor<nvme_type_> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    std::array<key_t, 6> const probes {shared,
                                       dense_key<key_t>(2),
                                       dense_key<key_t>(1),
                                       shared,
                                       dense_key<key_t>(rows_count + 7),
                                       dense_key<key_t>(rows_count)};
    std::span<key_t const> const probed {probes};
    std::array<std::size_t, probes.size()> ram_counted {}, nvme_counted {}, ram_contained {}, nvme_contained {},
        ram_gotten {}, nvme_gotten {};
    auto ram_rows = make_vectors<scalar_t>(probes.size() * wanted, dimensions, 0);
    auto nvme_rows = make_vectors<scalar_t>(probes.size() * wanted, dimensions, 0);
    auto& ram_executor = reference.builder.executor;
    expect_equal(reference.built.count(probed, std::span {ram_counted}, ram_executor), status_t::success_k);
    expect_equal(viewed.count(probed, std::span {nvme_counted}, driver.executor), status_t::success_k);
    expect_equal(reference.built.contains(probed, std::span {ram_contained}, ram_executor), status_t::success_k);
    expect_equal(viewed.contains(probed, std::span {nvme_contained}, driver.executor), status_t::success_k);
    expect_equal(reference.built.get(probed, ram_rows.span(), std::span {ram_gotten}, wanted, ram_executor),
                 status_t::success_k);
    expect_equal(viewed.get(probed, nvme_rows.span(), std::span {nvme_gotten}, wanted, driver.executor),
                 status_t::success_k);
    // The whole view in placement order, which ranks every key's entries by part and node.
    std::vector<key_t> placed_keys(rows_count);
    auto placed_rows = make_vectors<scalar_t>(rows_count, dimensions, 0);
    expect_equal(viewed.read_entries(0, std::span {placed_keys}, placed_rows.span(), driver.executor).status,
                 status_t::success_k);
    std::size_t const row_bytes = dense_rows_bytes(reference.vectors.view(), 1);
    for (std::size_t probe = 0; probe != probes.size(); ++probe) {
        expect_equal(nvme_counted[probe], ram_counted[probe]);
        expect_equal(nvme_contained[probe], ram_contained[probe]);
        expect_equal(nvme_gotten[probe], ram_gotten[probe]);
        std::size_t rank = 0;
        for (std::size_t placed = 0; placed != rows_count && rank != min_of(nvme_gotten[probe], wanted); ++placed) {
            if (!(placed_keys[placed] == probes[probe])) continue;
            expect_true(std::memcmp(nvme_rows.view().row(probe * wanted + rank).as_vector().byte_data(),
                                    placed_rows.view().row(placed).as_vector().byte_data(), row_bytes) == 0);
            ++rank;
        }
        expect_equal(rank, min_of(nvme_gotten[probe], wanted));
    }
    expect_equal(nvme_counted[0], divide_round_up(rows_count, std::size_t {3}));
    expect_equal(nvme_contained[4], 0u);
    expect_equal(viewed.get(probed, nvme_rows.span(), std::span {nvme_gotten}, wanted + 1, driver.executor),
                 status_t::config_mismatch_k);
    auto narrow_rows = make_vectors<scalar_t>(probes.size() * wanted, dense_dimensions_for<scalar_t>(1), 0);
    expect_equal(viewed.get(probed, narrow_rows.span(), std::span {nvme_gotten}, wanted, driver.executor),
                 status_t::config_mismatch_k);
    expect_equal(
        reference.built.get(probed, narrow_rows.span(), std::span {ram_gotten}, wanted, reference.builder.executor),
        status_t::config_mismatch_k);
}

/** A file medium whose reads round to at least @p block_bytes_, so a host probing 512-byte blocks
 *  checks 4096-byte covering reads as well. */
template <u32_t block_bytes_>
struct nvme_reblocked_medium {
    file_medium_t file;
};

template <u32_t block_bytes_>
u32_t block_bytes_of(nvme_reblocked_medium<block_bytes_> const& medium) noexcept {
    return max_of(medium.file.block_bytes, block_bytes_);
}

template <u32_t block_bytes_>
u64_t byte_base_of(nvme_reblocked_medium<block_bytes_> const&) noexcept {
    return 0;
}

template <u32_t block_bytes_>
io_device_queues_t device_queues_of(nvme_reblocked_medium<block_bytes_> const& medium) noexcept {
    return medium.file.queues;
}

/** The file a kernel engine reads a @ref nvme_reblocked_medium through. */
template <typename wanted_type_, u32_t block_bytes_>
    requires std::is_same_v<wanted_type_, file_medium_t>
file_medium_t const* medium_as(nvme_reblocked_medium<block_bytes_> const& medium) noexcept {
    return &medium.file;
}

/** @p engine_type_ reading @p medium_type_ media, every member forwarded, whose submits fail on
 *  request: the one that brings @ref submits_to_failure down to zero completes what it submitted,
 *  then drops it, as a refused submit does. */
template <typename engine_type_, typename medium_type_ = typename engine_type_::medium_t>
struct nvme_composed_engine {
    using wait_t = typename engine_type_::wait_t;
    using allocator_t = typename engine_type_::allocator_t;
    using medium_t = medium_type_;
    using config_t = typename engine_type_::config_t;

    /** Submits left until one fails, across every engine of this type; zero never fails. */
    inline static std::size_t submits_to_failure = 0;

    engine_type_ engine;
    std::size_t outstanding = 0;

    explicit nvme_composed_engine(allocator_t const& allocator = {}) noexcept : engine(allocator) {}

    static constexpr io_transport_t transport() noexcept { return engine_type_::transport(); }
    status_t open(config_t const& config) noexcept { return engine.open(config); }
    status_t reserve(std::size_t buffers, std::size_t buffer_bytes) noexcept {
        return engine.reserve(buffers, buffer_bytes);
    }
    status_t attach(std::span<medium_t const> media) noexcept { return engine.attach(media); }
    void detach() noexcept { engine.detach(); }
    status_t enqueue_read(io_request_t request) noexcept {
        status_t const enqueued = engine.enqueue_read(request);
        outstanding += succeeded(enqueued);
        return enqueued;
    }
    status_t submit(io_wait_t wait) noexcept {
        std::size_t const left = fu::standard_atomic_ref<std::size_t>(submits_to_failure)
                                     .fetch_sub_if_at_least(1, 0, std::memory_order_relaxed);
        if (left != 1) return engine.submit(wait);
        std::array<io_completion_t, 16> dropped;
        while (outstanding && succeeded(engine.submit(io_wait_t::blocking_k)))
            outstanding -= engine.poll_completions(dropped);
        outstanding = 0;
        return status_t::io_k;
    }
    std::size_t poll_completions(std::span<io_completion_t> completions) noexcept {
        std::size_t const polled = engine.poll_completions(completions);
        outstanding -= polled;
        return polled;
    }
    std::byte* buffer(io_buffer_id_t buffer) noexcept { return engine.buffer(buffer); }
};

/** Covering reads around rows that straddle blocks: value rows of 100 storage values in columns
 *  aligned to only 64 bytes, read at 512- and 4096-byte blocks, rank every entry exactly like the
 *  RAM group. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_, u32_t block_bytes_>
void test_nvme_indexes_straddling(u64_t seed) {
    using engine_t = nvme_composed_engine<engine_type_, nvme_reblocked_medium<block_bytes_>>;
    using nvme_t = nvme_indexes_for<uuid128_t, dtype_, nk_cap_serial_k, engine_t>;
    constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 4, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_t::scalar_t>(100);

    nvme_reference<nvme_t> reference {"nvme_straddling", kernel_, rows_count, dimensions, seed};
    nvme_t viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, 64);
    dense_indexes_executor<nvme_t> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_t> const ram = reference.exhaustive_rankings(queries, wanted);
    nvme_rankings<nvme_t> nvme {queries_count, wanted};
    expect_true(succeeded(nvme.search(viewed, queries, driver.executor, rows_count).status));
    nvme_expect_rankings(reference, queries, nvme, ram);

    // Neighbors rows straddle blocks too, so the audit's merged reads must still fetch every row
    // whole.
    auto const [ram_health, ram_status] = reference.built.health(reference.builder.executor);
    auto const [nvme_health, nvme_status] = viewed.health(driver.executor);
    expect_equal(nvme_status, ram_status);
    nvme_expect_reachability(nvme_health, ram_health);
}

/** The cursor walks the entries the RAM group's cursor does, in placement order, then answers no
 *  more, and @c read_entries reads their vectors back through reads crossing the parts' ends. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_cursor(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, chunk = 37;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_cursor", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed);
    dense_indexes_executor<nvme_type_> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    // The RAM cursor hands out keys alone, so its rows come back through `get`, one per key.
    typename nvme_reference<nvme_type_>::ram_t::cursor_t ram_cursor {forkunion_allocator_t {no_memory_domain_k}};
    expect_equal(ram_cursor.open(reference.built, {}), status_t::success_k);
    typename nvme_type_::cursor_t nvme_cursor {forkunion_allocator_t {no_memory_domain_k}};
    expect_equal(nvme_cursor.open(viewed, {}), status_t::success_k);
    std::vector<key_t> ram_keys(chunk), nvme_keys(chunk), placed_keys(chunk);
    std::vector<std::size_t> ram_entries(chunk);
    auto ram_rows = make_vectors<scalar_t>(chunk, dimensions, 0),
         nvme_rows = make_vectors<scalar_t>(chunk, dimensions, 0);
    std::size_t const row_bytes = dense_rows_bytes(reference.vectors.view(), 1);
    std::size_t exported = 0;
    while (true) {
        auto const [ram_read, ram_status] = ram_cursor.pull(reference.built, std::span {ram_keys}, nullptr, nullptr);
        expect_equal(ram_status, status_t::success_k);
        expect_equal(reference.built.get(std::span<key_t const> {ram_keys.data(), ram_read},
                                         rows_span_t {ram_rows.span()}.slice(0, ram_read),
                                         std::span {ram_entries.data(), ram_read}, 1, reference.builder.executor),
                     status_t::success_k);
        auto const [nvme_pulled, status] = nvme_cursor.pull(viewed, std::span {nvme_keys}, nullptr, nullptr,
                                                            driver.executor);
        expect_equal(status, status_t::success_k);
        expect_equal(nvme_pulled, ram_read);
        auto const [placed, placed_status] = viewed.read_entries(exported, std::span {placed_keys}, nvme_rows.span(),
                                                                 driver.executor);
        expect_equal(placed_status, status_t::success_k);
        expect_equal(placed.count, ram_read);
        if (!ram_read || nvme_pulled != ram_read || placed.count != ram_read) break;
        expect_true(placed.scan_reads.count >= 2);
        for (std::size_t row = 0; row != ram_read; ++row) {
            expect_true(nvme_keys[row] == ram_keys[row]);
            expect_true(placed_keys[row] == ram_keys[row]);
            expect_true(std::memcmp(nvme_rows.view().row(row).as_vector().byte_data(),
                                    ram_rows.view().row(row).as_vector().byte_data(), row_bytes) == 0);
        }
        exported += ram_read;
    }
    expect_equal(exported, rows_count);
}

/** The reachability audit agrees with the RAM group's on the same parts: the same nodes audited,
 *  the same reached. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_health(u64_t seed) {
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_health", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed);
    dense_indexes_executor<nvme_type_> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    auto const [ram, ram_status] = reference.built.health(reference.builder.executor);
    auto const [nvme, nvme_status] = viewed.health(driver.executor);
    expect_equal(ram_status, status_t::success_k);
    expect_equal(nvme_status, status_t::success_k);
    expect_equal(nvme.audited_nodes, rows_count);
    expect_true(nvme.depth_from_root > 0);
    nvme_expect_reachability(nvme, ram);
}

/** Per-query latencies and reads by purpose: every query's latency is reported, positive; a search
 *  reads one value row per distance, one neighbors row per expansion and one key per written
 *  result; a filtered search reads a key beside every value row and one more per written result;
 *  attention and a filtered product read a key beside every value row and resolve none, and over
 *  parts of another kernel attention, multiply and maxsim refuse with @c config_mismatch_k. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_telemetry(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    using vector_view_t = typename nvme_type_::vector_view_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 8, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_telemetry", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed);
    dense_indexes_executor<nvme_type_> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    std::vector<u64_t> latencies(queries_count);
    auto expect_latencies = [&] {
        for (u64_t& latency : latencies) {
            expect_true(latency > 0);
            latency = 0;
        }
    };
    std::size_t const value_bytes = dense_rows_bytes(reference.vectors.view(), 1);
    std::size_t const neighbor_bytes = viewed.connectivity() * sizeof(node_id_t);

    auto const [searched, search_status] = viewed.search(queries, nvme.matches(), {}, driver.executor,
                                                         std::span {latencies});
    expect_equal(search_status, status_t::success_k);
    expect_latencies();
    expect_equal(searched.neighbor_reads.count, searched.visited_members);
    expect_equal(searched.value_reads.count, searched.computed_distances);
    expect_equal(searched.key_reads.count, queries_count * wanted);
    expect_equal(searched.scan_reads.count, 0u);
    expect_true(searched.neighbor_reads.bytes >= searched.neighbor_reads.count * neighbor_bytes);
    expect_true(searched.value_reads.bytes >= searched.value_reads.count * value_bytes);
    expect_true(searched.key_reads.bytes >= searched.key_reads.count * sizeof(key_t));

    auto const odd = quad_predicate(
        [](key_t key) noexcept { return dense_key_number(key) % 2 ? admission_t::admit_k : admission_t::reject_k; });
    auto const [filtered, filtered_status] = viewed.search_if(queries, nvme.matches(), {}, odd, driver.executor,
                                                              std::span {latencies});
    expect_equal(filtered_status, status_t::success_k);
    expect_latencies();
    expect_true(filtered.count > 0);
    expect_equal(filtered.key_reads.count, filtered.value_reads.count + filtered.count);

    auto lookup = [&](std::span<key_t const> batch, std::span<vector_view_t> rows) noexcept {
        for (std::size_t lane = 0; lane != batch.size(); ++lane)
            rows[lane] = reference.vectors.view().row(reference.row_of(batch[lane])).as_vector();
    };
    auto outputs = nk::matrix<f32_t>::zeros({queries_count, dimensions}).value;
    std::vector<softmax_state_t> carry(queries_count);
    auto const [attended, attend_status] = viewed.attend(queries, 1, lookup, 1.0f / f32_t(dimensions), outputs.span(),
                                                         std::span {carry}, {}, driver.executor, std::span {latencies});
    // Every query row is a query of its own.
    std::vector<std::size_t> query_offsets(queries_count + 1);
    for (std::size_t query = 0; query != query_offsets.size(); ++query) query_offsets[query] = query;
    if constexpr (kernel_ == nk_kernel_dot_k) {
        expect_equal(attend_status, status_t::success_k);
        expect_latencies();
        expect_true(attended.value_reads.count > 0);
        expect_equal(attended.key_reads.count, attended.value_reads.count);
        expect_equal(attended.value_reads.count, attended.computed_distances);

        // Filtered MaxSim ranks odd keys alone, whether walks or a scan nominate them.
        std::vector<key_t> keys(queries_count * wanted);
        std::vector<f64_t> scores(queries_count * wanted);
        std::vector<std::size_t> offsets(queries_count + 1), scored(queries_count);
        products_tape<key_t, f64_t> const matches {std::span {keys}, std::span {scores}, std::span {offsets}, wanted};
        typename nvme_type_::search_config_t nominating;
        for (traversal_t const traversal : {traversal_t::walk_k, traversal_t::scan_k}) {
            nominating.traversal = traversal;
            offsets[0] = 0;
            expect_equal(viewed
                             .maxsim_if(queries, std::span<std::size_t const> {query_offsets}, std::span {scored},
                                        matches, {nominating, wanted}, odd, driver.executor)
                             .status,
                         status_t::success_k);
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const first = offsets[query], kept = offsets[query + 1] - first;
                expect_true(kept > 0 && scored[query] >= kept);
                for (std::size_t rank = 0; rank != kept; ++rank)
                    expect_true(dense_key_number(keys[first + rank]) % 2 == 1);
            }
        }

        // Filtered products write odd keys alone, read beside their values with none resolved
        // after, and refuse a scan; attention folds no key a predicate rejects.
        std::vector<typename nvme_type_::dot_result_t> products(queries_count * wanted);
        products_tape<key_t, typename nvme_type_::dot_result_t> const pairs {std::span {keys}, std::span {products},
                                                                             std::span {offsets}, wanted};
        offsets[0] = 0;
        auto const [multiplied, multiply_status] = viewed.multiply_if(queries, pairs, std::span {scored}, {}, odd,
                                                                      driver.executor);
        expect_equal(multiply_status, status_t::success_k);
        expect_equal(multiplied.key_reads.count, multiplied.value_reads.count);
        for (std::size_t query = 0; query != queries_count; ++query) {
            std::size_t const first = offsets[query], kept = offsets[query + 1] - first;
            expect_true(scored[query] > 0 && kept == min_of(scored[query], wanted));
            for (std::size_t pair = 0; pair != kept; ++pair) expect_true(dense_key_number(keys[first + pair]) % 2 == 1);
        }
        typename nvme_type_::search_config_t scan;
        scan.traversal = traversal_t::scan_k;
        expect_equal(viewed.multiply_if(queries, pairs, std::span {scored}, scan, odd, driver.executor).status,
                     status_t::unsupported_k);
        auto const no_rows = [](std::span<key_t const>, std::span<vector_view_t>) noexcept {};
        auto const reject_all = quad_predicate([](key_t) noexcept { return admission_t::reject_k; });
        expect_equal(viewed
                         .attend_if(queries, 1, no_rows, 1.0f, outputs.span(), std::span {carry}, {}, reject_all,
                                    driver.executor)
                         .status,
                     status_t::success_k);
        for (softmax_state_t const& state : carry) expect_equal(state.partition, 0.0f);
    }
    else {
        // Products are dots, so parts of another kernel refuse every algebra verb before any read.
        std::vector<key_t> keys(queries_count * wanted);
        std::vector<typename nvme_type_::dot_result_t> products(queries_count * wanted);
        std::vector<f64_t> scores(queries_count * wanted);
        std::vector<std::size_t> offsets(queries_count + 1), scored(queries_count);
        expect_equal(attend_status, status_t::config_mismatch_k);
        expect_equal(viewed
                         .multiply(queries,
                                   products_tape<key_t, typename nvme_type_::dot_result_t> {
                                       std::span {keys}, std::span {products}, std::span {offsets}, wanted},
                                   std::span {scored}, {}, driver.executor)
                         .status,
                     status_t::config_mismatch_k);
        expect_equal(
            viewed
                .maxsim(queries, std::span<std::size_t const> {query_offsets}, std::span {scored},
                        products_tape<key_t, f64_t> {std::span {keys}, std::span {scores}, std::span {offsets}, wanted},
                        {{}, wanted}, driver.executor)
                .status,
            status_t::config_mismatch_k);
        expect_equal(attended.value_reads.count, 0u);
    }
}

/** Views keep no executor: one executor serves a manifest view and a monolith view of the same rows
 *  in turn, and a second executor serves the first view alike; a claimed executor refuses a second
 *  caller, and one asked for a transport its engine does not drive refuses to spawn. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_executors(u64_t seed) {
    using executor_t = typename nvme_type_::executor_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 4, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_executors", kernel_, rows_count, dimensions, seed};
    nvme_type_ parts {forkunion_allocator_t {no_memory_domain_k}}, whole {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(parts);
    reference.view_monolith(whole);
    dense_indexes_executor<nvme_type_> first, second;
    expect_equal(parts.shards_count(), reference.built.shards_count());
    expect_equal(whole.shards_count(), 1u);
    if (!expect_true(succeeded(first.status) && succeeded(second.status)) || !reference.held()) return;

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    auto expect_exact = [&](nvme_type_& viewed, executor_t& executor) {
        expect_true(succeeded(nvme.search(viewed, queries, executor, rows_count).status));
        nvme_expect_rankings(reference, queries, nvme, ram);
    };
    expect_exact(parts, first.executor);
    expect_exact(whole, first.executor);
    expect_exact(parts, first.executor);
    expect_exact(parts, second.executor);

    auto [claim, claim_status] = first.executor.claim();
    expect_equal(claim_status, status_t::success_k);
    expect_equal(nvme.search(parts, queries, first.executor, 0).status, status_t::contended_k);

    io_engine_config_t foreign;
    foreign.queue_depth = nvme_queue_depth_k;
    foreign.transport = executor_t::engine_t::transport() == io_transport_t::blocking_k ? io_transport_t::linux_aio_k
                                                                                        : io_transport_t::blocking_k;
    expect_equal(executor_t::spawn(first.topology, 1, foreign).status, status_t::unsupported_k);
}

/** A read that comes back short of its row stops the call: a part truncated under a live view fails
 *  search, filtered search and the key scans with @c io_k, again on every retry, and the very
 *  executor then serves an intact view exactly, so every engine drained and every buffer
 *  returned. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_failed_read(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 8, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_failed_read", kernel_, rows_count, dimensions, seed};
    nvme_type_ broken {forkunion_allocator_t {no_memory_domain_k}}, intact {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(broken);
    reference.view_monolith(intact);
    dense_indexes_executor<nvme_type_> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;
    {
        growing_buffer<char> part_path {global_allocator};
        auto [file, status] = seekable_file_t::open(reference.checkpoint.part_path(part_path, 1),
                                                    file_access_t::truncate_read_write_k);
        expect_equal(status, status_t::success_k);
    }

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    auto const any = quad_predicate([](key_t) noexcept { return admission_t::admit_k; });
    for (std::size_t attempt = 0; attempt != 2; ++attempt) {
        expect_equal(nvme.search(broken, queries, driver.executor, 0).status, status_t::io_k);
        expect_equal(broken.search_if(queries, nvme.matches(), {}, any, driver.executor).status, status_t::io_k);
        // Every key, so some confirm their candidates in the truncated part.
        std::vector<std::size_t> entries(reference.keys.size());
        expect_equal(broken.count(std::span<key_t const> {reference.keys}, std::span {entries}, driver.executor),
                     status_t::io_k);
    }

    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    expect_true(succeeded(nvme.search(intact, queries, driver.executor, rows_count).status));
    nvme_expect_rankings(reference, queries, nvme, ram);
}

/** A submit failing midway, its reads issued and then dropped, fails a walk or a scan with @c io_k
 *  wherever it strikes, first submit or later, several hops of a walk in flight or one, and the
 *  very executors then serve both exactly, nothing stranded in an engine. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_failed_submit(u64_t seed) {
    using engine_t = nvme_composed_engine<engine_type_>;
    using nvme_t = nvme_indexes_for<uuid128_t, dtype_, nk_cap_serial_k, engine_t>;
    using key_t = typename nvme_t::key_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 8, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_t::scalar_t>(dense_values_k);

    nvme_reference<nvme_t> reference {"nvme_failed_submit", kernel_, rows_count, dimensions, seed};
    nvme_t viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed);
    dense_indexes_executor<nvme_t> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_t> nvme {queries_count, wanted};
    std::span<key_t const> const probed = std::span<key_t const> {reference.keys}.first(4);
    std::array<std::size_t, 4> entries {};
    for (std::size_t const countdown : {1, 4}) {
        engine_t::submits_to_failure = countdown;
        expect_equal(nvme.search(viewed, queries, driver.executor, 0).status, status_t::io_k);
        expect_equal(engine_t::submits_to_failure, 0u);
    }
    engine_t::submits_to_failure = 1;
    expect_equal(viewed.count(probed, std::span {entries}, driver.executor), status_t::io_k);
    expect_equal(engine_t::submits_to_failure, 0u);

    // A lone query on a deep queue has several hops of its walks in flight by its second and third
    // submits.
    dense_indexes_executor<nvme_t> deep {nvme_deep_queue_depth_k};
    expect_equal(deep.status, status_t::success_k);
    nvme_rankings<nvme_t> lone {1, wanted};
    for (std::size_t const countdown : {2, 3}) {
        engine_t::submits_to_failure = countdown;
        expect_equal(lone.search(viewed, reference.leading_rows(1), deep.executor, 0).status, status_t::io_k);
        expect_equal(engine_t::submits_to_failure, 0u);
    }

    nvme_rankings<nvme_t> const ram = reference.exhaustive_rankings(queries, wanted);
    expect_true(succeeded(nvme.search(viewed, queries, driver.executor, rows_count).status));
    nvme_expect_rankings(reference, queries, nvme, ram);
    expect_true(succeeded(nvme.search(viewed, queries, deep.executor, rows_count).status));
    nvme_expect_rankings(reference, queries, nvme, ram);
    expect_equal(viewed.count(probed, std::span {entries}, driver.executor), status_t::success_k);
    for (std::size_t const entry : entries) expect_equal(entry, 1u);
}

/** Lone queries on a queue their walks leave idle keep several hops in flight, and find at least as
 *  many true neighbors as on a queue a hop fills, their candidates only as wide as their results.
 *  Over blocking reads they also take fewer round trips, and a full batch on the deep queue, which
 *  leaves no depth idle, ranks exactly as the shallow queue does. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_pipelined(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    // Parts of thousands of entries, so a narrow walk scores a fraction of them and misses some
    // true neighbors.
    constexpr std::size_t rows_count = 16 * dense_thread_nodes_k, queries_count = 32, wanted = 10;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_pipelined", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed);
    dense_indexes_executor<nvme_type_> shallow, deep {nvme_deep_queue_depth_k};
    if (!expect_true(succeeded(shallow.status) && succeeded(deep.status)) || !reference.held()) return;

    auto const owned = make_vectors<scalar_t>(queries_count, dimensions, stream_key(seed, "nvme_pipelined_queries"));
    auto const queries = owned.view();
    nvme_rankings<nvme_type_> const exact = reference.exhaustive_rankings(queries, wanted);
    nvme_rankings<nvme_type_> lone {1, wanted};
    auto hits = [&](std::size_t query) {
        std::size_t found = 0;
        for (key_t const& key : lone.keys_of(0))
            for (key_t const& truth : exact.keys_of(query)) found += key == truth;
        return found;
    };
    std::size_t hop_by_hop_hits = 0, pipelined_hits = 0, hop_by_hop_trips = 0, pipelined_trips = 0;
    for (std::size_t query = 0; query != queries_count; ++query) {
        auto const row = queries(nk::range(query, query + 1), nk::slice);
        auto const [hop_by_hop, hop_by_hop_status] = lone.search(viewed, row, shallow.executor, wanted);
        expect_equal(hop_by_hop_status, status_t::success_k);
        hop_by_hop_hits += hits(query), hop_by_hop_trips += hop_by_hop.round_trips;
        auto const [pipelined, pipelined_status] = lone.search(viewed, row, deep.executor, wanted);
        expect_equal(pipelined_status, status_t::success_k);
        nvme_expect_rankings(reference, row, lone, lone);
        pipelined_hits += hits(query), pipelined_trips += pipelined.round_trips;
    }
    expect_true(pipelined_hits >= hop_by_hop_hits);
    // Kernel engines finish a submit's reads over several waits, in any order, and walks break
    // distance ties by completion order, so only blocking reads count one round trip per hop and
    // replay a walk exactly.
    if constexpr (nvme_type_::engine_t::transport() != io_transport_t::blocking_k) return;
    expect_true(pipelined_trips < hop_by_hop_trips);

    nvme_rankings<nvme_type_> hop_by_hop {queries_count, wanted}, full {queries_count, wanted};
    auto const [shallow_result, shallow_status] = hop_by_hop.search(viewed, queries, shallow.executor, wanted);
    auto const [deep_result, deep_status] = full.search(viewed, queries, deep.executor, wanted);
    expect_equal(shallow_status, status_t::success_k);
    expect_equal(deep_status, status_t::success_k);
    expect_true(full.counts == hop_by_hop.counts && full.keys == hop_by_hop.keys &&
                full.distances == hop_by_hop.distances);
    // Past the batch's end, deep queues let walks inherit idle flights and hold hops a walk
    // advancing hop by hop never reaches, so they may only do more work, never less.
    expect_true(deep_result.visited_members >= shallow_result.visited_members);
    expect_true(deep_result.computed_distances >= shallow_result.computed_distances);
    expect_true(deep_result.neighbor_reads.count >= shallow_result.neighbor_reads.count);
}

/** Parts served from elsewhere than the manifest points: one through the directory holding it, one
 *  moved under another name and named itself. The view ranks exactly like the RAM group, while one
 *  naming too few parts is refused and one following the manifest no longer opens. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_relocated(u64_t seed) {
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 4, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_relocated", kernel_, rows_count, dimensions, seed};
    nvme_type_ original {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(original);
    dense_indexes_executor<nvme_type_> driver;
    expect_equal(reference.checkpoint.parts, 2u);
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    char const* const manifest = reference.checkpoint.manifest.data();
    char const* const moved = reference.checkpoint.monolith.data();
    growing_buffer<char> part_path {global_allocator};
    expect_equal(std::rename(reference.checkpoint.part_path(part_path, 1).data(), moved), 0);
    std::string_view const manifest_path {manifest};
    std::string const directory {manifest_path.substr(0, manifest_path.find_last_of("/\\"))};
    std::array<std::string_view, 2> const paths {directory, moved};

    nvme_type_ relocated {forkunion_allocator_t {no_memory_domain_k}};
    expect_equal(relocated.view(manifest, std::span {paths}.first(1)), status_t::config_mismatch_k);
    expect_true(failed(relocated.view(manifest)));
    expect_equal(relocated.view(manifest, std::span {paths}), status_t::success_k);
    expect_equal(relocated.size(), rows_count);

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    expect_true(succeeded(nvme.search(relocated, queries, driver.executor, rows_count).status));
    nvme_expect_rankings(reference, queries, nvme, ram);
}

/** A monolith saved past byte zero of a larger file views through its base like the plain monolith
 *  and leaves the bytes before it as they were, while a base off the file's block is refused, as is
 *  an address in no form @ref file_address_t lists. The base is a multiple of every page a mapping
 *  may start on. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_indexes_byte_base(u64_t seed) {
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 4, wanted = 8;
    constexpr std::size_t byte_base = 65536;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_byte_base", kernel_, rows_count, dimensions, seed};
    nvme_type_ plain {forkunion_allocator_t {no_memory_domain_k}};
    reference.view_monolith(plain);
    dense_indexes_executor<nvme_type_> driver;
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    struct address_case_t {
        std::string_view path, location;
        u32_t namespace_id;
        u64_t byte_base;
    };
    for (address_case_t const& address : {
             address_case_t {"/data/sift.safegraphs", "/data/sift.safegraphs", 0, 0},
             address_case_t {"/dev/nvme1n1@64GiB", "/dev/nvme1n1", 0, 64ull << 30},
             address_case_t {"/dev/nvme1n1@4096", "/dev/nvme1n1", 0, 4096},
             address_case_t {"/data/run@64GiB@0", "/data/run@64GiB", 0, 0},
             address_case_t {"spdk://0000:5e:00.0@64GiB", "0000:5e:00.0", 1, 64ull << 30},
             address_case_t {"spdk://0000:5e:00.0/2@64GiB", "0000:5e:00.0", 2, 64ull << 30},
             address_case_t {"trtype:PCIe traddr:0000:5e:00.0 ns:2 base:4096", "trtype:PCIe traddr:0000:5e:00.0", 2,
                             4096},
         }) {
        auto const [parsed, parse_status] = parse_file_address(address.path);
        expect_equal(parse_status, status_t::success_k);
        expect_equal(parsed.location, address.location);
        expect_equal(parsed.namespace_id, address.namespace_id);
        expect_equal(parsed.byte_base, address.byte_base);
    }
    expect_equal(parse_file_address("").status, status_t::invalid_argument_k);
    expect_equal(parse_file_address("/dev/nvme1n1@64GB").status, status_t::invalid_argument_k);

    // The manifest's path, which a monolith view leaves unused, holds the larger file.
    char const* const medium = reference.checkpoint.manifest.data();
    std::vector<byte_t> const leading(byte_base, byte_t {0xA5});
    std::vector<byte_t> kept(byte_base);
    {
        auto [file, open_status] = seekable_file_t::open(medium, file_access_t::truncate_read_write_k);
        expect_true(succeeded(open_status) && succeeded(file.write_at(0, leading)));
    }
    std::string const based = fmt::format("{}@{}KiB", medium, byte_base / 1024);
    expect_true(succeeded(reference.whole().save(based, save_config_t {}, reference.builder.executor)));
    {
        auto [file, open_status] = seekable_file_t::open(medium, file_access_t::read_only_k);
        expect_true(succeeded(open_status) && succeeded(file.read_at(0, kept)) && kept == leading);
    }
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    expect_equal(viewed.view(based), status_t::success_k);
    expect_equal(viewed.size(), rows_count);
    std::string const misaligned = fmt::format("{}@{}", medium, byte_base + 1);
    if (!expect_equal(file_medium_t::open(misaligned).status, status_t::invalid_argument_k)) return;

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> plain_rankings {queries_count, wanted}, based_rankings {queries_count, wanted};
    expect_true(succeeded(plain_rankings.search(plain, queries, driver.executor, rows_count).status));
    expect_true(succeeded(based_rankings.search(viewed, queries, driver.executor, rows_count).status));
    expect_true(based_rankings.counts == plain_rankings.counts);
    nvme_expect_rankings(reference, queries, based_rankings, plain_rankings);
}

/** The single-part index over a monolith: under a walk as wide as the part it ranks every entry
 *  like the RAM group, each node ID a distinct entry at its exact distance, the entries named
 *  through the cursor's placement order. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_index_search(u64_t seed) {
    using index_t = typename nvme_type_::index_t;
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    using distance_t = typename nvme_type_::distance_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 4, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_part", kernel_, rows_count, dimensions, seed};
    nvme_type_ whole {forkunion_allocator_t {no_memory_domain_k}};
    reference.view_monolith(whole);
    dense_indexes_executor<nvme_type_> driver;
    std::unique_ptr<index_t> const part_ptr {new (std::nothrow) index_t(kernel_)};
    index_t& part = *part_ptr;
    expect_equal(part.view(reference.checkpoint.monolith.data()), status_t::success_k);
    expect_equal(part.size(), rows_count);
    if (!expect_true(succeeded(driver.status)) || !reference.held()) return;

    std::vector<key_t> placed_keys(rows_count);
    expect_equal(whole.read_entries(0, std::span {placed_keys}, {}, driver.executor).status, status_t::success_k);

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    std::vector<node_id_t> ids(queries_count * wanted);
    std::vector<distance_t> distances(queries_count * wanted);
    std::vector<std::size_t> counts(queries_count);
    typename index_t::search_config_t const config {static_cast<typename index_t::candidates_count_t>(rows_count)};
    auto const [searched, status] = part.search(queries, nk::matrix_span<node_id_t> {ids.data(), queries_count, wanted},
                                                nk::matrix_span<distance_t> {distances.data(), queries_count, wanted},
                                                std::span {counts}, config, driver.executor);
    expect_equal(status, status_t::success_k);
    expect_equal(searched.computed_distances, queries_count * rows_count);
    expect_true(counts == ram.counts);
    error_stats_t exact_errors, ram_errors;
    for (std::size_t query = 0; query != queries_count; ++query) {
        std::vector<bool> found(rows_count);
        auto const query_vector = queries.row(query).as_vector();
        auto const query_norms = norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(query_vector);
        for (std::size_t rank = 0; rank != min_of(counts[query], wanted); ++rank) {
            node_id_t const node = ids[query * wanted + rank];
            expect_true(node < rows_count && !found[node]);
            if (node >= rows_count || found[node]) continue;
            found[node] = true;
            auto const stored = reference.vectors.view().row(reference.row_of(placed_keys[node])).as_vector();
            auto const stored_norms = norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(stored);
            f32_t const exact = distance<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(
                kernel_, query_vector, query_norms, stored, stored_norms);
            exact_errors.add(distances[query * wanted + rank], exact);
            ram_errors.add(distances[query * wanted + rank], ram.distances[query * wanted + rank]);
        }
    }
    expect_errors("part distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k, exact_errors,
                  0);
    expect_errors("part against ram", comparison_family_t::approximate_k, comparison_verdict_t::assert_k, ram_errors,
                  0);
}

/** Queries every recall is measured over, and results each one keeps. */
constexpr std::size_t rewrite_queries_k = 64, rewrite_wanted_k = 10;

/** Entries the reshard and refine scenarios rewrite. */
constexpr std::size_t rewrite_entries_k = 4 * dense_thread_nodes_k;

/** The rows one rewrite scenario draws from, the RAM groups it saves them through, the checkpoints
 *  it removes, and every query's exact k-th distance. */
template <typename nvme_type_>
struct rewrite_fixture {
    using nvme_t = nvme_type_;
    using key_t = typename nvme_t::key_t;
    using scalar_t = typename nvme_t::scalar_t;
    using ram_t = typename nvme_reference<nvme_t>::ram_t;

    /** Whether every check since the fixture was built held. */
    checks_held_t held;
    std::string_view scenario;
    nk_kernel_kind_t kernel;
    nk::matrix<scalar_t> vectors;
    nk::matrix<scalar_t> queries;
    std::vector<key_t> keys;
    std::vector<f32_t> thresholds;
    dense_indexes_executor<ram_t> builder;
    dense_indexes_executor<nvme_t> driver;
    std::vector<std::unique_ptr<checkpoint_files_t>> checkpoints;

    rewrite_fixture(std::string_view scenario, nk_kernel_kind_t kernel, std::size_t rows, std::size_t dimensions,
                    u64_t seed)
        : scenario(scenario), kernel(kernel),
          vectors(make_vectors<scalar_t>(rows, dimensions, stream_key(seed, scenario))),
          queries(make_vectors<scalar_t>(rewrite_queries_k, dimensions, stream_key(seed, scenario, 1))), keys(rows),
          thresholds(rewrite_queries_k) {
        for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
        expect_equal(builder.status, status_t::success_k);
        expect_equal(driver.status, status_t::success_k);
        std::vector<f32_t> distances(rows);
        for (std::size_t query = 0; query != rewrite_queries_k; ++query) {
            for (std::size_t row = 0; row != rows; ++row) distances[row] = exact_(query, row);
            std::nth_element(distances.begin(), distances.begin() + (rewrite_wanted_k - 1), distances.end());
            thresholds[query] = distances[rewrite_wanted_k - 1];
        }
    }

    /** A fresh checkpoint in a scratch directory, removed with the fixture. */
    checkpoint_files_t& checkpoint() {
        checkpoints.push_back(std::make_unique<checkpoint_files_t>(scenario));
        return *checkpoints.back();
    }

    /** Builds rows [first, first + count) into a fresh RAM group of @p shards shards. */
    void build(ram_t& group, std::size_t first, std::size_t count, std::size_t shards = 2) {
        for (std::size_t shard = 0; shard != shards; ++shard) expect_true(succeeded(group.emplace(no_memory_domain_k)));
        expect_true(succeeded(group.reserve({count}).status));
        auto const all = vectors.view();
        auto const rows = all(nk::range(first, first + count), nk::slice);
        expect_true(succeeded(
            group.insert(std::span<key_t const> {keys.data() + first, count}, rows, builder.executor).status));
    }

    /** Saves rows [first, first + count) as a RAM group of @p shards shards and answers the path:
     *  a lone shard's monolith, or a manifest beside one part per shard. */
    char const* save(std::size_t first, std::size_t count, std::size_t shards) {
        ram_t group {kernel, {vectors.extent(1), connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        return save(group, first, count, shards);
    }

    /** Builds rows [first, first + count) into @p group of @p shards shards, saves it and answers
     *  the path. */
    char const* save(ram_t& group, std::size_t first, std::size_t count, std::size_t shards) {
        build(group, first, count, shards);
        checkpoint_files_t& saved = checkpoint();
        saved.parts = group.shards_count();
        char const* const path = shards == 1 ? saved.monolith.data() : saved.manifest.data();
        expect_equal(group.save(path, save_config_t {}, builder.executor), status_t::success_k);
        return path;
    }

    /** The share of the results @p indexes ranks for every query, through @p executor under the
     *  default walk, that lie within the query's exact k-th distance; a RAM group also takes its
     *  @p traversals, one per thread. */
    template <typename indexes_type_, typename executor_type_, typename... traversals_types_>
    f64_t recall(indexes_type_& indexes, executor_type_& executor, traversals_types_... traversals) {
        nvme_rankings<nvme_t> ranked {rewrite_queries_k, rewrite_wanted_k};
        expect_equal(ranked.search(indexes, queries.view(), executor, 0, traversals...).status, status_t::success_k);
        std::size_t hits = 0;
        for (std::size_t query = 0; query != rewrite_queries_k; ++query)
            for (key_t const key : ranked.keys_of(query)) {
                f32_t const exact = exact_(query, dense_key_number(key) - 1);
                hits += exact <= thresholds[query] + 1e-5f * (1 + std::fabs(thresholds[query]));
            }
        return static_cast<f64_t>(hits) / static_cast<f64_t>(rewrite_queries_k * rewrite_wanted_k);
    }

    /** The recall of a fresh RAM build of every entry into @p shards shards, so its searches walk
     *  as many graphs as a target of as many parts. */
    f64_t fresh_recall(std::size_t shards) {
        ram_t fresh {kernel, {vectors.extent(1), connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        build(fresh, 0, keys.size(), shards);
        return recall(fresh, builder.executor, to_writable_span(builder.traversals));
    }

    /** Views @p path as an NVMe index, which must open. */
    nvme_t view(char const* path) {
        nvme_t viewed {forkunion_allocator_t {no_memory_domain_k}};
        expect_equal(viewed.view(path), status_t::success_k);
        dense_indexes_executor<nvme_t> holding;
        expect_true(succeeded(holding.status) &&
                    succeeded(viewed.hold_keys(forkunion_allocator_t {no_memory_domain_k}, 0, holding.executor)));
        return viewed;
    }

    /** Every key names exactly one entry of @p viewed. */
    void expect_keys(nvme_t& viewed) {
        std::vector<std::size_t> entries(keys.size());
        expect_equal(viewed.count(std::span<key_t const> {keys}, std::span {entries}, driver.executor),
                     status_t::success_k);
        std::size_t miscounted = 0;
        for (std::size_t const entry : entries) miscounted += entry != 1;
        expect_equal(miscounted, 0u);
    }

    /** Recalls within the margin of a fresh RAM build of as many shards as it has parts, and
     *  holds every node in the root's component. */
    void expect_recall(nvme_t& viewed, char const* label) {
        expect_recall(viewed, label, fresh_recall(viewed.shards_count()));
    }

    /** Recalls within the margin of @p fresh and holds every node in the root's component. */
    void expect_recall(nvme_t& viewed, char const* label, f64_t fresh) {
        f64_t const rewritten = recall(viewed, driver.executor);
        if (rewritten + rewrite_recall_margin_k < fresh) {
            fmt::println(stderr, "  {} {}: recall {:.3f} trails {:.3f} of a fresh RAM build past the margin", scenario,
                         label, rewritten, fresh);
            expect_true(false);
        }
        expect_reachable(viewed);
    }

    /** At least the suite's share of @p count stored entries from @p first find themselves in
     *  @p viewed, judged by distance band, so the entries of one part cannot hide behind
     *  another's. */
    void expect_self_recall(nvme_t& viewed, std::size_t first, std::size_t count, char const* label) {
        nvme_rankings<nvme_t> ranked {count, rewrite_wanted_k};
        expect_equal(
            ranked.search(viewed, vectors(nk::range(first, first + count), nk::slice), driver.executor, 0).status,
            status_t::success_k);
        std::size_t hits = 0;
        for (std::size_t row = 0; row != count; ++row)
            for (auto const distance : ranked.distances_of(row))
                if (distance < dense_self_match_tolerance_k && distance > -dense_self_match_tolerance_k) {
                    ++hits;
                    break;
                }
        if (hits * 100 < count * dense_minimal_recall_percent_k) {
            fmt::println(stderr, "  {} {}: {} of {} rows find themselves", scenario, label, hits, count);
            expect_true(false);
        }
    }

    /** Every node of @p viewed is in the root's component. */
    void expect_reachable(nvme_t& viewed) {
        auto const [health, health_status] = viewed.health(driver.executor);
        expect_equal(health_status, status_t::success_k);
        expect_equal(health.reachable_nodes, viewed.size());
    }

  private:
    f32_t exact_(std::size_t query, std::size_t row) const noexcept {
        auto const query_vector = queries.view().row(query).as_vector();
        auto const stored = vectors.view().row(row).as_vector();
        return distance<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(
            kernel, query_vector, norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(query_vector), stored,
            norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(stored));
    }
};

/** A lent workspace of @p bytes. */
inline std::vector<byte_t> rewrite_workspace(std::size_t bytes) { return std::vector<byte_t>(bytes); }

/** Resharding the two parts one RAM group saved yields one part over every entry, every key once,
 *  that a RAM group loads back whole, and that recalls within the margin of a fresh RAM build. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_rewrites_reshard(u64_t seed) {
    using ram_t = typename rewrite_fixture<nvme_type_>::ram_t;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_reshard", kernel_, rewrite_entries_k, dimensions, seed};
    nvme_type_ source = fixture.view(fixture.save(0, rewrite_entries_k, 2));
    if (!fixture.held()) return;
    std::vector<byte_t> workspace = rewrite_workspace(4u << 20);
    checkpoint_files_t& target = fixture.checkpoint();
    std::array<nvme_type_*, 1> const sources {&source};
    auto const [resharded, status] = reshard(std::span<nvme_type_* const>(sources), target.manifest.data(),
                                             fixture.driver.executor, std::span {workspace});
    expect_equal(status, status_t::success_k);
    target.parts = resharded.target_shards;
    expect_equal(resharded.source_shards, 2u);
    expect_equal(resharded.target_shards, 1u);
    expect_equal(resharded.moved_entries, rewrite_entries_k / 2);
    expect_equal(resharded.cut_edges, 0u);
    expect_true(resharded.graph.reselected_nodes > 0);
    if (failed(status)) return;

    nvme_type_ viewed = fixture.view(target.manifest.data());
    expect_equal(viewed.size(), rewrite_entries_k);
    expect_equal(viewed.shards_count(), 1u);
    fixture.expect_keys(viewed);
    fixture.expect_recall(viewed, "of two parts");

    ram_t loaded {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(loaded)));
    expect_equal(loaded.load(target.manifest.data(), load_config_t {}, fixture.builder.executor), status_t::success_k);
    expect_equal(loaded.size(), rewrite_entries_k);
}

/** Three parts from two sources, a manifest of two and a monolith, reshard task by task, a search
 *  of a source running between every two tasks on the same executor, and the target recalls within
 *  the margin. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_rewrites_reshard_tasks(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    std::size_t const first_rows = rewrite_entries_k * 2 / 3;
    rewrite_fixture<nvme_type_> fixture {"rewrite_reshard_tasks", kernel_, rewrite_entries_k, dimensions, seed};
    nvme_type_ first = fixture.view(fixture.save(0, first_rows, 2));
    nvme_type_ second = fixture.view(fixture.save(first_rows, rewrite_entries_k - first_rows, 1));
    if (!fixture.held()) return;
    expect_equal(first.shards_count() + second.shards_count(), 3u);
    std::vector<byte_t> workspace = rewrite_workspace(4u << 20);
    checkpoint_files_t& target = fixture.checkpoint();
    std::array<nvme_type_*, 2> const sources {&first, &second};
    auto [plan, status] = plan_reshard(std::span<nvme_type_* const>(sources), target.manifest.data(),
                                       std::span {workspace});
    expect_equal(status, status_t::success_k);
    if (failed(status)) return;

    nvme_rankings<nvme_type_> interleaved {rewrite_queries_k, rewrite_wanted_k};
    std::size_t phases = 0, tasks = 0;
    do {
        ++phases;
        for (auto const task : plan.tasks()) {
            expect_equal(plan.run(task, fixture.driver.executor), status_t::success_k);
            expect_equal(interleaved.search(first, fixture.queries.view(), fixture.driver.executor, 0).status,
                         status_t::success_k);
            ++tasks;
        }
    } while (plan.advance());
    expect_true(plan.phase() == phase_t::done_k);
    expect_true(phases >= 2 && tasks >= phases);
    auto const [resharded, finish_status] = plan.finish(fixture.driver.executor);
    expect_equal(finish_status, status_t::success_k);
    target.parts = resharded.target_shards;
    expect_equal(resharded.source_shards, 3u);
    expect_equal(resharded.target_shards, 1u);
    if (failed(finish_status)) return;

    nvme_type_ viewed = fixture.view(target.manifest.data());
    fixture.expect_keys(viewed);
    fixture.expect_recall(viewed, "of three parts");
    // The second source's monolith is the largest part and stays resident;
    // the first source's two parts are appended.
    fixture.expect_self_recall(viewed, 0, rewrite_queries_k, "appended rows");
    fixture.expect_self_recall(viewed, first_rows, rewrite_queries_k, "resident rows");
}

/** Four parts reshard into two under a capacity of two parts' entries, each appended part linking
 *  in, then back into four under a capacity of one part's entries: every part is cut in half, the
 *  cut edges are counted, and each half, repaired by the RAM refine the workspace holds, recalls
 *  within the margin. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_rewrites_reshard_capacity(u64_t seed) {
    using ram_t = typename rewrite_fixture<nvme_type_>::ram_t;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    std::size_t const part_entries = rewrite_entries_k / 4;
    rewrite_fixture<nvme_type_> fixture {"rewrite_reshard_capacity", kernel_, rewrite_entries_k, dimensions, seed};
    ram_t group {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    nvme_type_ source = fixture.view(fixture.save(group, 0, rewrite_entries_k, 4));
    if (!fixture.held()) return;
    expect_equal(source.shards_count(), 4u);
    std::vector<byte_t> workspace = rewrite_workspace(8u << 20);

    checkpoint_files_t& halves = fixture.checkpoint();
    std::array<nvme_type_*, 1> const quartered {&source};
    auto const [packed, pack_status] = reshard(std::span<nvme_type_* const>(quartered), halves.manifest.data(),
                                               fixture.driver.executor, std::span {workspace},
                                               reshard_config_t {{}, 2 * part_entries});
    expect_equal(pack_status, status_t::success_k);
    halves.parts = packed.target_shards;
    expect_equal(packed.source_shards, 4u);
    expect_equal(packed.target_shards, 2u);
    expect_equal(packed.moved_entries, 2 * part_entries);
    expect_equal(packed.cut_edges, 0u);
    if (failed(pack_status)) return;
    nvme_type_ halved = fixture.view(halves.manifest.data());
    expect_equal(halved.shards_count(), 2u);
    expect_equal(halved.size(), rewrite_entries_k);
    fixture.expect_keys(halved);
    fixture.expect_recall(halved, "of four parts into two");

    checkpoint_files_t& quarters = fixture.checkpoint();
    std::array<nvme_type_*, 1> const halved_sources {&halved};
    auto const [cut, cut_status] = reshard(std::span<nvme_type_* const>(halved_sources), quarters.manifest.data(),
                                           fixture.driver.executor, std::span {workspace},
                                           reshard_config_t {{}, part_entries});
    expect_equal(cut_status, status_t::success_k);
    quarters.parts = cut.target_shards;
    expect_equal(cut.target_shards, 4u);
    expect_equal(cut.moved_entries, 2 * part_entries);
    expect_true(cut.cut_edges > 0);
    expect_true(cut.graph.reselected_nodes >= rewrite_entries_k);
    if (failed(cut_status)) return;
    nvme_type_ viewed = fixture.view(quarters.manifest.data());
    expect_equal(viewed.shards_count(), 4u);
    expect_equal(viewed.size(), rewrite_entries_k);
    fixture.expect_keys(viewed);
    fixture.expect_recall(viewed, "of two parts into four, repaired in RAM");
}

/**
 *  @brief One streamed round over every node of a part, in a workspace too small for a RAM index,
 *      re-stitches every node in neighborhoods no wider than the workspace holds.
 *
 *  Over the part as built it keeps every node reached and recalls within the margin of it. The
 *  part thinned to one neighbor per node, node zero for every other node, fills node zero's row
 *  after a few islands, so the heal links those alone, as the RAM heal does, and the restitch
 *  reaches every neighborhood holding one: the round recalls better and reaches more nodes than
 *  its opening check found. No task writes any file beside the target part but the restitch tape,
 *  gone once the plan finishes.
 */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_rewrites_refine(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_refine", kernel_, rewrite_entries_k, dimensions, seed};
    char const* const saved = fixture.save(0, rewrite_entries_k, 1);
    if (!fixture.held()) return;

    // The thinned copy keeps every node's first neighbor and nothing else.
    checkpoint_files_t& thinned = fixture.checkpoint();
    {
        auto [mapping, map_status] = mapped_file_t::read(saved);
        expect_equal(map_status, status_t::success_k);
        auto const [header, header_status] = safegraphs_read_header(mapping.view());
        if (!expect_equal(header_status, status_t::success_k)) return;
        std::vector<byte_t> bytes(mapping.view().begin(), mapping.view().end());
        std::size_t const connectivity = regular_width(header.neighbors.shape).value_or(0);
        auto* const neighbors = reinterpret_cast<node_id_t*>(bytes.data() + header.data_offset +
                                                             header.neighbors.data.first);
        for (std::size_t row = 0; row != rewrite_entries_k; ++row) {
            if (row) neighbors[row * connectivity] = 0;
            for (std::size_t slot = 1; slot != connectivity; ++slot)
                neighbors[row * connectivity + slot] = node_id_unknown_k;
        }
        auto [file, open_status] = seekable_file_t::open(thinned.monolith.data(), file_access_t::truncate_read_write_k);
        expect_equal(open_status, status_t::success_k);
        expect_equal(file.write_at(0, bytes_view_t {bytes}), status_t::success_k);
    }
    std::vector<byte_t> workspace = rewrite_workspace(64u << 10);
    {
        nvme_type_ built = fixture.view(saved);
        if (!fixture.held()) return;
        f64_t const built_recall = fixture.recall(built, fixture.driver.executor);
        checkpoint_files_t& rebuilt = fixture.checkpoint();
        auto const [refined, status] = refine(built, iota_view<std::size_t> {0, rewrite_entries_k},
                                              rebuilt.manifest.data(), fixture.driver.executor, std::span {workspace});
        expect_equal(status, status_t::success_k);
        rebuilt.parts = refined.target_shards;
        refine_report_t const& round = refined.graph;
        expect_equal(round.reselected_nodes, rewrite_entries_k);
        expect_true(round.neighborhoods > 1);
        expect_equal(round.reachability.reachable_nodes, rewrite_entries_k);
        if (failed(status)) return;
        nvme_type_ viewed = fixture.view(rebuilt.manifest.data());
        fixture.expect_keys(viewed);
        fixture.expect_recall(viewed, "streamed", built_recall);
    }

    nvme_type_ source = fixture.view(thinned.monolith.data());
    if (!fixture.held()) return;
    f64_t const before = fixture.recall(source, fixture.driver.executor);
    checkpoint_files_t& target = fixture.checkpoint();
    auto [plan, plan_status] = plan_refine(source, iota_view<std::size_t> {0, rewrite_entries_k},
                                           target.manifest.data(), std::span {workspace});
    expect_equal(plan_status, status_t::success_k);
    if (failed(plan_status)) return;
    // Every file named after the target's stem: its one part, and the restitch tape during a round.
    std::filesystem::path const manifest_path {target.manifest.data()};
    std::string const stem {safegraphs_stem(target.manifest.data())};
    auto stem_files = [&] {
        std::size_t files = 0;
        for (std::filesystem::directory_entry const& entry :
             std::filesystem::directory_iterator {manifest_path.parent_path()})
            files += entry.path().string().starts_with(stem);
        return files;
    };
    target.parts = 1;
    std::size_t restitches = 0;
    do {
        if (plan.phase() == phase_t::restitch_k) restitches += plan.tasks().size();
        for (auto const task : plan.tasks()) {
            expect_equal(plan.run(task, fixture.driver.executor), status_t::success_k);
            expect_true(stem_files() <= 2u);
        }
    } while (plan.advance());
    auto const [refined, status] = plan.finish(fixture.driver.executor);
    expect_equal(status, status_t::success_k);
    target.parts = refined.target_shards;
    refine_report_t const& round = refined.graph;
    expect_equal(refined.moved_entries, 0u);
    expect_equal(round.reselected_nodes, rewrite_entries_k);
    expect_true(round.changed_nodes > 0 && round.changed_nodes <= round.reselected_nodes);
    expect_true(round.islands_found > 0);
    expect_true(round.neighborhoods > 1 && round.neighborhoods == restitches);
    expect_true(round.widest_neighborhood < rewrite_entries_k);
    expect_true(round.reachability.reachable_nodes > round.opening.reachable_nodes);
    expect_equal(stem_files(), 2u);
    if (!fixture.held()) return;

    nvme_type_ viewed = fixture.view(target.manifest.data());
    f64_t const after = fixture.recall(viewed, fixture.driver.executor);
    if (!(after > before)) {
        fmt::println(stderr, "  {}: recall {:.3f} after the refine, {:.3f} before it", fixture.scenario, after, before);
        expect_true(false);
    }
    fixture.expect_keys(viewed);
}

/**
 *  @brief Building rows four times what the workspace holds as one RAM index runs at least four
 *      blocks and stitches them by streamed rounds until one stops improving, reaching every entry
 *      and recalling within the margin of a RAM build of every entry.
 *
 *  A RAM index takes its chunk directories and its first chunk before it holds an entry, 1.8 MB
 *  with 128-byte cache lines, so the workspace is measured on this host: that fixed cost, plus a
 *  quarter of the rows' staging and RAM rows. A literal four times the rows' bytes would take tens
 *  of thousands of rows to outgrow the fixed cost.
 */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_rewrites_build(u64_t seed) {
    using scalar_t = typename nvme_type_::scalar_t;
    using ram_t = typename rewrite_fixture<nvme_type_>::ram_t;
    using workspace_allocator_t = arena_allocator<byte_t, lent_allocator<byte_t>>;
    using block_index_t = ram_index<typename nvme_type_::quads_kit_t, typename nvme_type_::tiles_kit_t,
                                    typename nvme_type_::elementwise_kit_t, typename nvme_type_::rmscast_kit_t,
                                    typename nvme_type_::visits_kit_t, fu::standard_yield_t, fu::standard_atomic_ref,
                                    dummy_cache_hints_t, workspace_allocator_t, workspace_allocator_t,
                                    arena_allocator<byte_t, forkunion_allocator_t>, node_id_t>;
    constexpr std::size_t rows_count = 4 * rewrite_entries_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_build", kernel_, rows_count, dimensions, seed};
    // One RAM shard over every entry is both the rows the build reads and the graph it is judged
    // against.
    ram_t group {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    char const* const saved = fixture.save(group, 0, rows_count, 1);
    if (!fixture.held()) return;

    std::vector<byte_t> probed(16u << 20);
    typename workspace_allocator_t::arena_t probe_arena {lent_allocator<byte_t> {probed}};
    expect_equal(probe_arena.reserve_exactly(probed.size()), status_t::success_k);
    {
        workspace_allocator_t const allocator {probe_arena};
        std::unique_ptr<block_index_t> const probe {
            new (std::nothrow) block_index_t(kernel_, {dimensions, connectivity_16_k}, allocator, allocator)};
        expect_true(probe && succeeded(probe->reserve({1}).status));
    }
    std::size_t const fixed = probe_arena.size();
    std::size_t const value_bytes = dense_rows_bytes(fixture.vectors.view(), 1), neighbors_bytes = 16 * sizeof(u32_t);
    std::size_t const row_bytes = 2 * value_bytes + 2 * neighbors_bytes + 2 * sizeof(u32_t) + 16;
    std::vector<byte_t> workspace = rewrite_workspace(fixed + rows_count / 4 * row_bytes);
    checkpoint_files_t& target = fixture.checkpoint();
    std::array<std::string_view, 1> const sources {saved};
    auto [plan, status] = plan_build<nvme_type_>(std::span<std::string_view const>(sources), target.manifest.data(),
                                                 std::span {workspace});
    expect_equal(status, status_t::success_k);
    if (failed(status)) return;
    std::size_t blocks = 0, rounds = 0;
    do {
        if (plan.phase() == phase_t::build_k) blocks += plan.tasks().size();
        rounds += plan.ordinal() == 1;
        for (auto const task : plan.tasks()) expect_equal(plan.run(task, fixture.driver.executor), status_t::success_k);
    } while (plan.advance());
    auto const [built, finish_status] = plan.finish(fixture.driver.executor);
    expect_equal(finish_status, status_t::success_k);
    target.parts = built.target_shards;
    expect_true(blocks >= 4);
    expect_equal(built.target_shards, 1u);
    if (failed(finish_status)) return;

    expect_true(built.graph.reselected_nodes >= rows_count);
    expect_true(built.graph.neighborhoods > 1);
    expect_true(rounds > 1);
    expect_true(built.graph.verdict != refine_verdict_t::stopped_k);
    expect_equal(built.graph.reachability.reachable_nodes, rows_count);
    nvme_type_ viewed = fixture.view(target.manifest.data());
    fixture.expect_keys(viewed);
    fixture.expect_recall(
        viewed, "in blocks",
        fixture.recall(group, fixture.builder.executor, to_writable_span(fixture.builder.traversals)));
    fixture.expect_self_recall(viewed, 0, rewrite_queries_k, "first block");
    fixture.expect_self_recall(viewed, rows_count - rewrite_queries_k, rewrite_queries_k, "last block");
}

/** A reshard dropped after its first restitch task leaves no manifest, no part and no restitch
 *  tape behind, so nothing views. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_rewrites_dropped(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_dropped", kernel_, rewrite_entries_k, dimensions, seed};
    nvme_type_ source = fixture.view(fixture.save(0, rewrite_entries_k, 2));
    if (!fixture.held()) return;
    std::vector<byte_t> workspace = rewrite_workspace(512u << 10);
    checkpoint_files_t& target = fixture.checkpoint();
    target.parts = 1;
    {
        std::array<nvme_type_*, 1> const sources {&source};
        auto [plan, status] = plan_reshard(std::span<nvme_type_* const>(sources), target.manifest.data(),
                                           std::span {workspace});
        expect_equal(status, status_t::success_k);
        while (succeeded(status) && plan.phase() != phase_t::restitch_k) {
            for (auto const task : plan.tasks())
                expect_equal(plan.run(task, fixture.driver.executor), status_t::success_k);
            if (!plan.advance()) break;
        }
        expect_true(plan.phase() == phase_t::restitch_k);
        auto const tasks = plan.tasks();
        expect_equal(plan.run(tasks[0], fixture.driver.executor), status_t::success_k);
        expect_equal(plan.finish(fixture.driver.executor).status, status_t::cancelled_k);
    }
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    expect_true(failed(viewed.view(target.manifest.data())));
    growing_buffer<char> part_path {global_allocator};
    std::string_view const part = target.part_path(part_path, 0);
    expect_true(failed(seekable_file_t::open(part, file_access_t::read_only_k).status));
    std::string const tape = fmt::format("{}.tape", part);
    expect_true(failed(seekable_file_t::open(tape, file_access_t::read_only_k).status));
}

/** The bytes of the file at @p path, empty when it cannot be read. */
inline std::vector<byte_t> nvme_file_bytes(char const* path) {
    auto [mapping, map_status] = mapped_file_t::read(path);
    if (failed(map_status)) return {};
    return std::vector<byte_t>(mapping.view().begin(), mapping.view().end());
}

/** A manifest of two parts loads as a writable index, removing a part a crashed refine left, and a
 *  second load of the same manifest and a view refuse to refine through it, the one with
 *  @c contended_k before, between and after the refine's publishes, the other with @c read_only_k.
 *  A refine stopped ahead of its first commit leaves the manifest's bytes and every part intact, so
 *  a fresh view ranks exactly what the first view ranks; a whole refine runs one RAM round per part
 *  over its neighborhoods, replaces both parts, recalls within the margin and keeps every key,
 *  reaches every node, and a view of the replaced manifest agrees with the loaded index. Releasing
 *  the loaded index removes the superseded parts. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_load_refine(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"load_refine", kernel_, rewrite_entries_k, dimensions, seed};
    char const* const saved = fixture.save(0, rewrite_entries_k, 2);
    if (!fixture.held()) return;
    std::vector<byte_t> const manifest_before = nvme_file_bytes(saved);
    growing_buffer<char> old_part {global_allocator}, new_part {global_allocator};
    std::string const generation_stem = fmt::format("{}.1", safegraphs_stem(saved));
    auto replaced_part_exists = [&](std::size_t part) {
        auto const [path, status] = safegraphs_part_path(new_part, generation_stem, part, 2);
        return succeeded(status) && succeeded(seekable_file_t::open(path, file_access_t::read_only_k).status);
    };

    nvme_type_ view = fixture.view(saved);
    f64_t const before = fixture.recall(view, fixture.driver.executor);
    expect_equal(view.refuses_writes(), status_t::read_only_k);
    expect_equal(refine(view, refine_config_t {}, fixture.driver.executor).status, status_t::read_only_k);
    // A part a crashed refine left behind, which the next writable load removes.
    {
        auto const [orphan, orphan_status] = safegraphs_part_path(new_part, generation_stem, 0, 2);
        auto [planted, planted_status] = seekable_file_t::open(orphan, file_access_t::truncate_read_write_k);
        expect_true(succeeded(orphan_status) && succeeded(planted_status));
    }
    {
        nvme_type_ loaded {forkunion_allocator_t {no_memory_domain_k}};
        if (!expect_equal(loaded.load(saved), status_t::success_k)) return;
        expect_true(!replaced_part_exists(0));
        expect_equal(loaded.refuses_writes(), status_t::unsupported_k);
        nvme_type_ second {forkunion_allocator_t {no_memory_domain_k}};
        expect_equal(second.load(saved), status_t::contended_k);
        nvme_type_ monolith {forkunion_allocator_t {no_memory_domain_k}};
        expect_equal(monolith.load(fixture.checkpoints[0]->part_path(old_part, 0)), status_t::unsupported_k);

        // Stopped at its first task, the refine commits nothing and leaves no part behind.
        auto const stop = [](std::size_t, std::size_t) noexcept { return continuation_t::cancel_k; };
        expect_equal(refine(loaded, refine_config_t {}, fixture.driver.executor, stop).status, status_t::cancelled_k);
        expect_true(nvme_file_bytes(saved) == manifest_before);
        expect_true(!replaced_part_exists(0) && !replaced_part_exists(1));
        nvme_type_ untouched = fixture.view(saved);
        expect_equal(fixture.recall(untouched, fixture.driver.executor), before);

        // Between its tasks and across both publishes, a second writer stays refused.
        std::size_t admitted = 0;
        auto const probe = [&](std::size_t, std::size_t) noexcept {
            admitted += second.load(saved) != status_t::contended_k;
            return continuation_t::continue_k;
        };
        auto const [refined, refined_status] = refine(loaded, refine_config_t {}, fixture.driver.executor, probe);
        expect_equal(admitted, 0u);
        expect_equal(second.load(saved), status_t::contended_k);
        expect_equal(refined_status, status_t::success_k);
        expect_equal(refined.reselected_nodes, rewrite_entries_k);
        expect_true(refined.neighborhoods > 0);
        expect_equal(refined.reachability.reachable_nodes, rewrite_entries_k);
        expect_true(replaced_part_exists(0) && replaced_part_exists(1));
        expect_true(nvme_file_bytes(saved) != manifest_before);
        f64_t const after = fixture.recall(loaded, fixture.driver.executor);
        if (after + rewrite_recall_margin_k < before) {
            fmt::println(stderr, "  {}: recall {:.3f} after the refine trails {:.3f} before it past the margin",
                         fixture.scenario, after, before);
            expect_true(false);
        }
        // A view opened before the refine keeps reading the superseded parts it holds open.
        expect_equal(fixture.recall(view, fixture.driver.executor), before);
        nvme_type_ reopened = fixture.view(saved);
        fixture.expect_keys(reopened);
        fixture.expect_reachable(reopened);
        expect_equal(fixture.recall(reopened, fixture.driver.executor), after);
        loaded.release();
    }
    for (std::size_t part = 0; part != 2; ++part) {
        std::string_view const path = fixture.checkpoints[0]->part_path(old_part, part);
        expect_true(failed(seekable_file_t::open(path, file_access_t::read_only_k).status));
        auto const [replaced, status] = safegraphs_part_path(new_part, generation_stem, part, 2);
        if (succeeded(status)) std::remove(replaced.data());
    }
}

/** Loads @p path as a writable NVMe index with its key table, which must succeed. */
template <typename nvme_type_>
void nvme_load_writable(nvme_type_& loaded, char const* path, rewrite_fixture<nvme_type_>& fixture) {
    expect_equal(loaded.load(path), status_t::success_k);
    expect_true(succeeded(loaded.hold_keys(forkunion_allocator_t {no_memory_domain_k}, 0, fixture.driver.executor)));
}

/** Releases the loaded @p loaded and removes its manifest and the parts it serves, whose names a
 *  rewrite chose beside the checkpoint's own. */
template <typename nvme_type_>
void nvme_remove_loaded(nvme_type_& loaded) {
    std::vector<std::string> paths {std::string {loaded.manifest_path()}};
    typename nvme_type_::paths_t buffer {forkunion_allocator_t {no_memory_domain_k}};
    for (std::size_t shard = 0; shard != loaded.shards_count(); ++shard)
        if (auto const [path, status] = loaded.part_path(shard, buffer); succeeded(status)) paths.emplace_back(path);
    loaded.release();
    for (std::string const& path : paths) std::remove(path.c_str());
}

/** Every entry of @p loaded is in the root's component, and each of its keys counts once. */
template <typename nvme_type_>
void nvme_expect_whole(nvme_type_& loaded, rewrite_fixture<nvme_type_>& fixture, std::size_t first, std::size_t count) {
    auto const [health, health_status] = loaded.health(fixture.driver.executor);
    expect_equal(health_status, status_t::success_k);
    expect_equal(health.reachable_nodes, loaded.size());
    std::vector<std::size_t> entries(count);
    expect_equal(
        loaded.count(std::span {fixture.keys.data() + first, count}, std::span {entries}, fixture.driver.executor),
        status_t::success_k);
    std::size_t miscounted = 0;
    for (std::size_t const entry : entries) miscounted += entry != 1;
    expect_equal(miscounted, 0u);
}

/** A loaded index of two parts gathers a viewed source of one: it serves every entry of both, each
 *  key once, reached from its root, and the target parts ran their rounds. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_reshard_gathers(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    std::size_t const half = rewrite_entries_k / 2;
    rewrite_fixture<nvme_type_> fixture {"reshard_gathers", kernel_, rewrite_entries_k, dimensions, seed};
    char const* const gathering = fixture.save(0, half, 2);
    nvme_type_ source = fixture.view(fixture.save(half, rewrite_entries_k - half, 1));
    nvme_type_ loaded {forkunion_allocator_t {no_memory_domain_k}};
    nvme_load_writable(loaded, gathering, fixture);
    if (!fixture.held()) return;
    std::array<nvme_type_*, 2> const sources {&loaded, &source};
    std::array<nvme_type_*, 1> const targets {&loaded};
    std::array<nvme_type_*, 2> const repeated {&source, &source}, holding_null {&source, nullptr};
    for (std::span<nvme_type_* const> malformed :
         {std::span<nvme_type_* const> {repeated}, std::span<nvme_type_* const> {holding_null},
          std::span<nvme_type_* const> {}})
        expect_equal(reshard(malformed, std::span<nvme_type_* const> {targets}, fixture.driver.executor).status,
                     status_t::invalid_argument_k);
    expect_equal(
        reshard(std::span<nvme_type_* const> {sources}, std::span<nvme_type_* const> {}, fixture.driver.executor)
            .status,
        status_t::invalid_argument_k);
    // Cutting into targets that name a view answers read-only before the source among them.
    expect_equal(
        reshard(std::span<nvme_type_* const> {sources}, std::span<nvme_type_* const> {sources}, fixture.driver.executor)
            .status,
        status_t::read_only_k);
    auto const [gathered, status] = reshard(std::span<nvme_type_* const> {sources},
                                            std::span<nvme_type_* const> {targets}, fixture.driver.executor);
    expect_equal(status, status_t::success_k);
    expect_true(gathered.graph.neighborhoods > 0 && gathered.graph.reselected_nodes > 0);
    expect_equal(loaded.size(), rewrite_entries_k);
    nvme_expect_whole(loaded, fixture, 0, rewrite_entries_k);
    nvme_remove_loaded(loaded);
}

/** A refine plan over a loaded index of two parts commits every part and reaches every entry;
 *  past its finish it advances no further, and a second finish answers @c uninitialized_k. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_refine_finishes(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"refine_finishes", kernel_, rewrite_entries_k, dimensions, seed};
    char const* const saved = fixture.save(0, rewrite_entries_k, 2);
    nvme_type_ loaded {forkunion_allocator_t {no_memory_domain_k}};
    nvme_load_writable(loaded, saved, fixture);
    if (!fixture.held()) return;
    {
        auto [plan, plan_status] = plan_refine(loaded, refine_config_t {}, fixture.driver.executor);
        expect_equal(plan_status, status_t::success_k);
        if (failed(plan_status)) return;
        auto drive_barrier = [&] {
            for (auto const task : plan.tasks())
                expect_equal(plan.run(task, fixture.driver.executor), status_t::success_k);
            return plan.advance();
        };
        while (drive_barrier()) {}
        auto const [refined, status] = plan.finish(fixture.driver.executor);
        expect_equal(status, status_t::success_k);
        expect_equal(refined.reachability.reachable_nodes, rewrite_entries_k);
        expect_true(!plan.advance());
        expect_equal(plan.finish(fixture.driver.executor).status, status_t::uninitialized_k);
    }
    nvme_expect_whole(loaded, fixture, 0, rewrite_entries_k);
    nvme_remove_loaded(loaded);
}

/** A loaded index of four parts reshards into one, which serves every entry, each key once, all
 *  reached from its root. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_reshard_recuts(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"reshard_recuts", kernel_, rewrite_entries_k, dimensions, seed};
    char const* const saved = fixture.save(0, rewrite_entries_k, 4);
    nvme_type_ loaded {forkunion_allocator_t {no_memory_domain_k}};
    nvme_load_writable(loaded, saved, fixture);
    if (!fixture.held()) return;
    expect_equal(loaded.shards_count(), 4u);
    std::array<nvme_type_*, 1> const itself {&loaded};
    auto const [resharded, status] = reshard(std::span<nvme_type_* const> {itself},
                                             std::span<nvme_type_* const> {itself}, fixture.driver.executor);
    expect_equal(status, status_t::success_k);
    expect_equal(resharded.source_shards, 4u);
    expect_equal(resharded.target_shards, 1u);
    expect_equal(loaded.shards_count(), 1u);
    expect_equal(loaded.size(), rewrite_entries_k);
    nvme_expect_whole(loaded, fixture, 0, rewrite_entries_k);
    nvme_remove_loaded(loaded);
}

/** Two loaded indexes of two parts each are cut into two empty targets, several sources into
 *  several targets: each target loads four parts of its own, one per source part, together
 *  they serve every entry, each key once, every entry reached from its target's root, and both
 *  sources are left empty. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
void test_nvme_reshard_cuts(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    std::size_t const half = rewrite_entries_k / 2;
    rewrite_fixture<nvme_type_> fixture {"reshard_cuts", kernel_, rewrite_entries_k, dimensions, seed};
    char const* const saved = fixture.save(0, half, 2);
    char const* const other_saved = fixture.save(half, rewrite_entries_k - half, 2);
    nvme_type_ loaded {forkunion_allocator_t {no_memory_domain_k}}, other {forkunion_allocator_t {no_memory_domain_k}};
    nvme_load_writable(loaded, saved, fixture);
    nvme_load_writable(other, other_saved, fixture);
    if (!fixture.held()) return;
    std::array<nvme_type_, 2> targets {nvme_type_ {forkunion_allocator_t {no_memory_domain_k}},
                                       nvme_type_ {forkunion_allocator_t {no_memory_domain_k}}};
    std::array<nvme_type_*, 2> const slots {&targets[0], &targets[1]};
    std::array<nvme_type_*, 2> const sources {&loaded, &other};
    auto const [cut, status] = reshard(std::span<nvme_type_* const> {sources}, std::span<nvme_type_* const> {slots},
                                       fixture.driver.executor);
    expect_equal(status, status_t::success_k);
    expect_true(cut.graph.neighborhoods > 0);
    expect_equal(loaded.shards_count() + other.shards_count(), 0u);
    expect_equal(targets[0].size() + targets[1].size(), rewrite_entries_k);
    std::vector<std::size_t> totals(rewrite_entries_k, 0), entries(rewrite_entries_k);
    for (nvme_type_& target : targets) {
        expect_equal(target.shards_count(), 4u);
        expect_true(
            succeeded(target.hold_keys(forkunion_allocator_t {no_memory_domain_k}, 0, fixture.driver.executor)));
        auto const [health, health_status] = target.health(fixture.driver.executor);
        expect_equal(health_status, status_t::success_k);
        expect_equal(health.reachable_nodes, target.size());
        expect_equal(target.count(std::span<typename nvme_type_::key_t const> {fixture.keys}, std::span {entries},
                                  fixture.driver.executor),
                     status_t::success_k);
        for (std::size_t key = 0; key != rewrite_entries_k; ++key) totals[key] += entries[key];
    }
    std::size_t miscounted = 0;
    for (std::size_t const total : totals) miscounted += total != 1;
    expect_equal(miscounted, 0u);
    for (nvme_type_& target : targets) nvme_remove_loaded(target);
}

/** A @c dense::nvme_indexes the shared scenarios build like any other sharded index: entries go
 *  in a two-shard @c dense::ram_indexes, and every insertion saves its manifest and views the
 *  parts. The one-key lookups the NVMe view lacks run as batches of one on an executor of its
 *  own. */
template <typename nvme_type_>
struct nvme_staged_indexes {
    using viewed_t = nvme_type_;
    using key_t = typename viewed_t::key_t;
    using scalar_t = typename viewed_t::scalar_t;
    using dot_result_t = typename viewed_t::dot_result_t;
    using distance_t = typename viewed_t::distance_t;
    using vector_view_t = typename viewed_t::vector_view_t;
    using vectors_view_t = typename viewed_t::vectors_view_t;
    using vectors_span_t = typename viewed_t::vectors_span_t;
    using search_config_t = typename viewed_t::search_config_t;
    using maxsim_config_t = typename viewed_t::maxsim_config_t;
    using search_result_t = typename viewed_t::search_result_t;
    using executor_t = typename viewed_t::executor_t;
    using index_t = typename viewed_t::index_t;
    using ram_t = typename nvme_reference<viewed_t>::ram_t;
    using reserve_capacity_t = typename ram_t::reserve_capacity_t;
    using geometry_t = typename ram_t::geometry_t;

    dense_indexes_executor<ram_t> builder;
    dense_indexes_executor<viewed_t> reader;
    ram_t built;
    viewed_t viewed;
    checkpoint_files_t checkpoint {"nvme_staged"};

    /** The RAM group walks by @p kernel in @p geometry; the view reads both back from the parts'
     *  headers. */
    nvme_staged_indexes(nk_kernel_kind_t kernel, geometry_t geometry, forkunion_allocator_t allocator) noexcept
        : built(kernel, geometry, allocator), viewed(allocator) {}

    status_t emplace(fu_memory_domain_id_t memory_domain) noexcept { return built.emplace(memory_domain); }
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return built.reserve(config); }
    std::size_t size() const noexcept { return viewed.size(); }

    /** Inserts into the RAM group, then saves it as a manifest and views the parts. */
    expected<typename ram_t::insert_result_t> insert(std::span<key_t const> keys, vectors_view_t vectors,
                                                     executor_t&) noexcept {
        expected<typename ram_t::insert_result_t> inserted = built.insert(keys, vectors, builder.executor);
        if (!inserted) return inserted;
        return {inserted.value, save_and_view_()};
    }

    /** Inserts one row into the RAM group through @p traversal, then saves and views the parts
     *  again. */
    template <typename traversal_type_>
    auto insert(key_t const& key, vector_view_t vector, traversal_type_& traversal) noexcept {
        auto inserted = built.insert(key, vector, traversal);
        if (inserted) inserted.status = save_and_view_();
        return inserted;
    }

    std::size_t count(key_t const& key) noexcept {
        std::size_t entries = 0;
        return succeeded(viewed.count({&key, 1}, {&entries, 1}, reader.executor)) ? entries : 0;
    }
    bool contains(key_t const& key) noexcept { return count(key) != 0; }
    std::size_t get(key_t const& key, vectors_span_t vectors, std::size_t wanted, executor_t& executor) noexcept {
        std::size_t entries = 0;
        std::size_t const rows = min_of(wanted, vectors.extent(0));
        status_t const gotten = viewed.get({&key, 1}, vectors(nk::range(std::size_t {0}, rows), nk::slice),
                                           {&entries, 1}, rows, executor);
        return succeeded(gotten) ? entries : 0;
    }
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) noexcept {
        return viewed.count(keys, entries, executor);
    }
    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) noexcept {
        return viewed.contains(keys, entries, executor);
    }
    status_t get(std::span<key_t const> keys, vectors_span_t vectors, std::span<std::size_t> entries,
                 std::size_t wanted, executor_t& executor) noexcept {
        return viewed.get(keys, vectors, entries, wanted, executor);
    }

    expected<search_result_t> search(vectors_view_t queries, ranked_matches_span<key_t, distance_t> matches,
                                     search_config_t config, executor_t& executor) noexcept {
        return viewed.search(queries, matches, config, executor);
    }

    template <typename predicate_type_>
    expected<search_result_t> search_if(vectors_view_t queries, ranked_matches_span<key_t, distance_t> matches,
                                        search_config_t config, predicate_type_&& predicate,
                                        executor_t& executor) noexcept {
        return viewed.search_if(queries, matches, config, predicate, executor);
    }

    expected<search_result_t> search(vectors_view_t queries, indexer_heads<scalar_t> const& heads,
                                     ranked_matches_span<key_t, distance_t> matches, search_config_t config,
                                     executor_t& executor) noexcept {
        return viewed.search(queries, heads, matches, config, executor);
    }

    template <typename product_type_>
    expected<search_result_t> multiply(vectors_view_t left, products_tape<key_t, product_type_> products,
                                       std::span<std::size_t> scored, search_config_t config,
                                       executor_t& executor) noexcept {
        return viewed.multiply(left, products, scored, config, executor);
    }

    template <typename score_type_>
    expected<search_result_t> maxsim(vectors_view_t queries, std::span<std::size_t const> query_offsets,
                                     std::span<std::size_t> scored, products_tape<key_t, score_type_> matches,
                                     maxsim_config_t config, executor_t& executor) noexcept {
        return viewed.maxsim(queries, query_offsets, scored, matches, config, executor);
    }

    template <typename value_lookup_type_>
    expected<search_result_t> attend(vectors_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, nk::matrix_span<f32_t> outputs,
                                     std::span<softmax_state_t> carry, search_config_t config,
                                     executor_t& executor) noexcept {
        return viewed.attend(queries, head_count, value_lookup, score_scale, outputs, carry, config, executor);
    }

  private:
    status_t save_and_view_() noexcept {
        checkpoint.parts = built.shards_count();
        if (status_t const saved = built.save(checkpoint.manifest.data(), save_config_t {}, builder.executor);
            failed(saved))
            return saved;
        if (status_t const opened = viewed.view(checkpoint.manifest.data()); failed(opened)) return opened;
        dense_indexes_executor<viewed_t> holding;
        if (failed(holding.status)) return holding.status;
        return viewed.hold_keys(forkunion_allocator_t {no_memory_domain_k}, 0, holding.executor);
    }
};

/** Whether a sharded index is the NVMe view the shared scenarios build through its RAM group. */
template <typename indexes_type_>
concept nvme_staged = nvme_collection<indexes_type_> && requires { typename indexes_type_::ram_t; };

/** A monolith keyed @c u32, as the benches write one, viewed by a @c uuid collection: the cursor,
 *  the key scans and a ranking's resolved keys read every key zero-extended, big-endian as the
 *  bindings render integer identities, at the distances the @c u32 view of the file ranks. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
void test_nvme_widened_keys(u64_t seed) {
    using narrow_t = nvme_indexes_for<u32_t, dtype_, nk_cap_serial_k, engine_type_>;
    using wide_t = nvme_indexes_for<uuid128_t, dtype_, nk_cap_serial_k, engine_type_>;
    using scalar_t = typename wide_t::scalar_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 8, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    auto const number_of = [](uuid128_t const& key) noexcept {
        u64_t number = 0;
        for (std::size_t byte = sizeof(key.bytes) - sizeof(number); byte != sizeof(key.bytes); ++byte)
            number = number << 8 | std::to_integer<u64_t>(key.bytes[byte]);
        return number;
    };

    nvme_reference<narrow_t> reference {"nvme_widened", kernel_, rows_count, dimensions, seed};
    narrow_t narrow {forkunion_allocator_t {no_memory_domain_k}};
    reference.view_monolith(narrow);
    wide_t wide {forkunion_allocator_t {no_memory_domain_k}};
    expect_equal(wide.view(reference.checkpoint.monolith.data()), status_t::success_k);
    reference.hold(wide);
    dense_indexes_executor<narrow_t> narrow_driver;
    dense_indexes_executor<wide_t> wide_driver;
    if (!expect_true(succeeded(narrow_driver.status) && succeeded(wide_driver.status)) || !reference.held()) return;
    expect_equal(wide.size(), rows_count);

    std::vector<uuid128_t> placed_keys(rows_count);
    auto placed_rows = make_vectors<scalar_t>(rows_count, dimensions, 0);
    expect_equal(wide.read_entries(0, std::span {placed_keys}, placed_rows.span(), wide_driver.executor).status,
                 status_t::success_k);
    std::vector<bool> found(rows_count);
    for (uuid128_t const& key : placed_keys) {
        u64_t const number = number_of(key);
        bool const fresh = key == uuid_of(number) && number >= 1 && number <= rows_count && !found[number - 1];
        expect_true(fresh);
        if (fresh) found[number - 1] = true;
    }

    std::array<uuid128_t, 2> const probes {uuid_of(1), uuid_of(rows_count + 7)};
    std::array<std::size_t, probes.size()> counted {};
    expect_equal(wide.count(std::span<uuid128_t const> {probes}, std::span {counted}, wide_driver.executor),
                 status_t::success_k);
    expect_equal(counted[0], 1u);
    expect_equal(counted[1], 0u);

    // A walk as wide as the file ranks every entry, so only ties may swap keys between the two
    // views.
    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<narrow_t> narrow_rankings {queries_count, wanted};
    nvme_rankings<wide_t> wide_rankings {queries_count, wanted};
    expect_true(succeeded(narrow_rankings.search(narrow, queries, narrow_driver.executor, rows_count).status));
    expect_true(succeeded(wide_rankings.search(wide, queries, wide_driver.executor, rows_count).status));
    expect_true(wide_rankings.counts == narrow_rankings.counts);
    for (std::size_t slot = 0; slot != wide_rankings.keys.size(); ++slot) {
        uuid128_t const& key = wide_rankings.keys[slot];
        expect_equal(wide_rankings.distances[slot], narrow_rankings.distances[slot]);
        expect_true(key == uuid_of(number_of(key)) && number_of(key) <= rows_count);
    }
}

/** Composes one engine's NVMe cell: every read verb against the RAM group its parts came from,
 *  every rewrite under the angular kernel against a fresh RAM build, since a dot graph neither
 *  stays reachable through a reshard nor finds itself at distance zero, and a @c u32 monolith
 *  viewed through @c uuid128_t keys, skipped where the host refuses the engine. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
case_result_t test_nvme_configuration(u64_t seed, char const* refused) {
    using indexes_t = nvme_indexes_for<uuid128_t, dtype_, nk_cap_serial_k, engine_type_>;
    using staged_t = nvme_staged_indexes<indexes_t>;
    {
        dense_indexes_executor<indexes_t> probe;
        if (failed(probe.status)) return case_result_t::skipped(refused);
    }
    test_nvme_indexes_search<indexes_t, kernel_>(seed);
    test_nvme_indexes_hints<indexes_t, kernel_>(seed);
    test_nvme_indexes_filtered<indexes_t, kernel_>(seed);
    test_nvme_indexes_key_scans<indexes_t, kernel_>(seed);
    test_nvme_indexes_cursor<indexes_t, kernel_>(seed);
    test_nvme_indexes_health<indexes_t, kernel_>(seed);
    test_nvme_indexes_telemetry<indexes_t, kernel_>(seed);
    test_nvme_indexes_executors<indexes_t, kernel_>(seed);
    test_nvme_indexes_failed_read<indexes_t, kernel_>(seed);
    test_nvme_indexes_failed_submit<engine_type_, dtype_, kernel_>(seed);
    test_nvme_indexes_pipelined<indexes_t, kernel_>(seed);
    test_nvme_indexes_relocated<indexes_t, kernel_>(seed);
    test_nvme_indexes_byte_base<indexes_t, kernel_>(seed);
    test_nvme_index_search<indexes_t, kernel_>(seed);
    test_nvme_indexes_straddling<engine_type_, dtype_, kernel_, 512>(seed);
    test_nvme_indexes_straddling<engine_type_, dtype_, kernel_, 4096>(seed);
    test_dense_indexes_scan<staged_t, kernel_>(seed);
    test_dense_indexes_products<staged_t, kernel_>(seed);
    test_dense_indexes_indexer_search<staged_t, kernel_>(seed);
    test_dense_indexes_maxsim<staged_t, kernel_>(seed);
    test_dense_indexes_attend<staged_t, kernel_>(seed);
    test_dense_indexes_duplicate_keys<staged_t, kernel_>(seed);
    dense_expect_duplicates_build<staged_t, kernel_>(seed);
    test_nvme_rewrites_reshard<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_rewrites_reshard_tasks<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_rewrites_reshard_capacity<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_rewrites_refine<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_rewrites_build<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_rewrites_dropped<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_load_refine<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_refine_finishes<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_reshard_gathers<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_reshard_recuts<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_reshard_cuts<indexes_t, nk_kernel_angular_k>(seed);
    test_nvme_widened_keys<engine_type_, dtype_, kernel_>(seed);
    return {};
}

#if USEARCH_WITH_LINUX_SPDK

/**
 *  @brief The SPDK engine over a monolith copied onto an NVMe namespace, judged against the
 *      blocking engine over the monolith's file, every verb alike.
 *
 *  Runs in two passes around a rig: with no @p nvme_paths it saves the monolith as
 *  @c usearch-nvme-spdk.safegraphs under @c TMPDIR for the rig to copy onto a namespace, and skips
 *  the rest; with @c USEARCH_NVME_PATHS naming that namespace's transport ID it views the copy
 *  directly and through a relocation of the file's view onto it.
 */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
case_result_t test_nvme_spdk_configuration(u64_t seed, std::span<std::string_view const> nvme_paths) {
    using files_t = nvme_indexes_for<uuid128_t, dtype_, nk_cap_serial_k, blocking_files_engine<fu::standard_yield_t>>;
    using spdk_t = nvme_indexes_for<uuid128_t, dtype_, nk_cap_serial_k, linux_spdk_engine<fu::standard_yield_t>>;
    using key_t = typename files_t::key_t;
    using scalar_t = typename files_t::scalar_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 8, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    // The rig copies the part between the passes, so it outlives a run's scratch directory.
    std::error_code error;
    std::string const part = (std::filesystem::temp_directory_path(error) / "usearch-nvme-spdk.safegraphs").string();
    if (nvme_paths.empty()) {
        nvme_reference<files_t> reference {"nvme_spdk", kernel_, rows_count, dimensions, seed};
        expect_true(succeeded(reference.whole().save(part, save_config_t {}, reference.builder.executor)));
        return case_result_t::skipped("needs USEARCH_NVME_PATHS");
    }

    files_t files {forkunion_allocator_t {no_memory_domain_k}};
    spdk_t direct {forkunion_allocator_t {no_memory_domain_k}}, relocated {forkunion_allocator_t {no_memory_domain_k}};
    bool const viewed = expect_equal(files.view(part), status_t::success_k) &
                        expect_equal(direct.view(nvme_paths.front()), status_t::success_k) &
                        expect_equal(relocated.view(part, nvme_paths.first(1)), status_t::success_k);
    expect_equal(direct.size(), rows_count);
    expect_equal(relocated.size(), rows_count);
    dense_indexes_executor<files_t> files_driver;
    dense_indexes_executor<spdk_t> spdk_driver;
    auto const released = [&]() noexcept {
        direct.release();
        relocated.release();
        return case_result_t {};
    };
    if (!viewed || !expect_true(succeeded(files_driver.status) && succeeded(spdk_driver.status))) return released();
    // Key lookups need the key tables, which neither view holds until asked.
    expect_equal(files.hold_keys(forkunion_allocator_t {no_memory_domain_k}, 0, files_driver.executor),
                 status_t::success_k);
    expect_equal(direct.hold_keys(forkunion_allocator_t {no_memory_domain_k}, 0, spdk_driver.executor),
                 status_t::success_k);

    // Under a walk as wide as the part both rank every entry, so only ties may swap keys.
    auto const queries = make_vectors<scalar_t>(queries_count, dimensions, stream_key(seed, "nvme_spdk_queries"));
    nvme_rankings<files_t> expected_rankings {queries_count, wanted};
    expect_true(succeeded(expected_rankings.search(files, queries.view(), files_driver.executor, rows_count).status));
    for (spdk_t* const viewed : {&direct, &relocated}) {
        nvme_rankings<files_t> found {queries_count, wanted};
        expect_true(succeeded(found.search(*viewed, queries.view(), spdk_driver.executor, rows_count).status));
        expect_true(found.counts == expected_rankings.counts);
        for (std::size_t slot = 0; slot != found.distances.size(); ++slot)
            expect_equal(found.distances[slot], expected_rankings.distances[slot]);
    }

    std::array<key_t, 4> const probes {dense_key<key_t>(1), dense_key<key_t>(2), dense_key<key_t>(rows_count),
                                       dense_key<key_t>(rows_count + 7)};
    std::array<std::size_t, probes.size()> files_gotten {}, spdk_gotten {};
    auto files_rows = make_vectors<scalar_t>(probes.size(), dimensions, 0);
    auto spdk_rows = make_vectors<scalar_t>(probes.size(), dimensions, 0);
    expect_equal(files.get(probes, files_rows.span(), std::span {files_gotten}, 1, files_driver.executor),
                 status_t::success_k);
    expect_equal(direct.get(probes, spdk_rows.span(), std::span {spdk_gotten}, 1, spdk_driver.executor),
                 status_t::success_k);
    expect_true(files_gotten == spdk_gotten);
    std::size_t const row_bytes = dense_rows_bytes(files_rows.view(), 1);
    auto same_rows = [&](auto const& first, auto const& second, std::size_t rows) {
        std::size_t differing = 0;
        for (std::size_t row = 0; row != rows; ++row)
            differing += std::memcmp(first.view().row(row).as_vector().byte_data(),
                                     second.view().row(row).as_vector().byte_data(), row_bytes) != 0;
        return differing == 0;
    };
    expect_true(same_rows(files_rows, spdk_rows, probes.size()));

    std::vector<key_t> files_keys(rows_count), spdk_keys(rows_count);
    auto files_placed = make_vectors<scalar_t>(rows_count, dimensions, 0);
    auto spdk_placed = make_vectors<scalar_t>(rows_count, dimensions, 0);
    expect_equal(files.read_entries(0, std::span {files_keys}, files_placed.span(), files_driver.executor).status,
                 status_t::success_k);
    expect_equal(direct.read_entries(0, std::span {spdk_keys}, spdk_placed.span(), spdk_driver.executor).status,
                 status_t::success_k);
    expect_true(files_keys == spdk_keys);
    expect_true(same_rows(files_placed, spdk_placed, rows_count));

    auto const [files_health, files_status] = files.health(files_driver.executor);
    auto const [spdk_health, spdk_status] = direct.health(spdk_driver.executor);
    expect_equal(spdk_status, files_status);
    expect_equal(spdk_health.reachable_nodes, files_health.reachable_nodes);
    return released();
}

#endif

} // namespace unum::usearch::test::dense

/*  The residency forms specialize templates of the enclosing @c test namespace, which only a scope
 *  enclosing it may do, so they follow the namespace at file scope and qualify what they name. */

/** The NVMe form: two threads over a topology of its own, each opening the engine of the index's
 *  type, its queue @p queue_depth reads deep. */
template <us::test::dense::nvme_collection indexes_type_>
struct us::test::dense_indexes_executor<indexes_type_> {
    using executor_t = typename indexes_type_::executor_t;

    fu_topology_t topology = nullptr;
    executor_t executor;
    us::utilities::status_t status = us::utilities::status_t::uninitialized_k;

    explicit dense_indexes_executor(std::size_t queue_depth = us::test::dense::nvme_queue_depth_k) noexcept {
        if (fu_topology_new(&topology) != fu_success_k) return;
        us::utilities::io_engine_config_t config;
        config.queue_depth = queue_depth;
#if USEARCH_WITH_LINUX_SPDK
        // The SPDK engine shares the environment the views opened, over plain pages for fabrics.
        if constexpr (std::is_same_v<typename executor_t::medium_t, us::utilities::spdk_medium_t>) {
            us::utilities::linux_spdk_process_t::config_t environment;
            environment.memory = us::utilities::linux_spdk_memory_t::plain_pages_k;
            // A refused start leaves the engines to answer why at spawn.
            [[maybe_unused]] us::utilities::status_t const acquired =
                us::utilities::linux_spdk_process_t::acquire<fu::standard_yield_t>(environment);
        }
#endif
        auto [spawn_executor, spawn_status] = executor_t::spawn(topology, 2, config);
        status = spawn_status;
        executor = std::move(spawn_executor);
    }
    ~dense_indexes_executor() noexcept {
        executor.release();
        executor = executor_t {};
        if (topology) fu_topology_delete(topology);
    }
    dense_indexes_executor(dense_indexes_executor const&) = delete;
    dense_indexes_executor& operator=(dense_indexes_executor const&) = delete;

    /** Places the shards one sharded scenario drives: two RAM shards on the heap domain, saved
     *  before any view. */
    static us::utilities::status_t place_shards(indexes_type_& indexes) noexcept {
        if (us::utilities::status_t const placed = indexes.emplace(us::utilities::no_memory_domain_k); failed(placed))
            return placed;
        return indexes.emplace(us::utilities::no_memory_domain_k);
    }
};

/** The staged NVMe form: one-row insertions go into the RAM group, through a context over its own
 *  arena. */
template <us::test::dense::nvme_staged indexes_type_>
struct us::test::dense_indexes_row_driver<indexes_type_> {
    us::test::dense_arena_context<typename indexes_type_::ram_t> arena;
    us::utilities::status_t status;

    dense_indexes_row_driver(indexes_type_& indexes, us::test::dense_indexes_executor<indexes_type_>&) noexcept
        : arena(indexes.built), status(arena.status) {}
    typename indexes_type_::ram_t::traversal_context_t& handle() noexcept { return arena.context; }
};
