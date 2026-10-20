/**
 *  @file test/dense/nvme_indexes.hpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The suite composers for @c dense::nvme_indexes and its rewrites: two-shard
 *      @c dense::ram_indexes saved as SafeGraphs parts and viewed back from storage, once per
 *      I/O engine the host admits.
 *
 *  Parts land under @c TMPDIR, so a run that points it at a volume enforcing direct-I/O alignment
 *  checks every covering read against it. A walk as wide as the index scores every row, so the
 *  RAM group the parts came from judges every verb, as do the @c f64 oracles of @c dense.cuh.
 *
 *  Merge, split, refine and build view each target back and judge it against a fresh RAM build of
 *  the same rows. Recall is the share of every query's results whose exact distance is within its
 *  exact k-th nearest, so ties never count against a ranking. A rewrite passes when its target,
 *  searched with the default walk, recalls within @ref rewrite_recall_margin_k of a fresh RAM build
 *  of the same rows searched alike.
 */
#pragma once

#include <cstdint> // `std::uintptr_t`, for the checkpoint paths' per-run salt
#include <cstdio>  // `std::remove`, `std::rename`, `std::snprintf`, for the checkpoint's files

#include <algorithm>    // `std::is_sorted`, `std::nth_element`, `std::upper_bound`
#include <array>        // `std::array`
#include <atomic>       // `std::atomic`, for the failing engines' countdown
#include <bit>          // `std::bit_cast`, for `uuid_t` keys from numbers
#include <filesystem>   // `std::filesystem::temp_directory_path`, the directory parts land in
#include <memory>       // `std::unique_ptr`, for the rewrites' checkpoints
#include <string>       // `std::string`, for the relocation's directory
#include <string_view>  // `std::string_view`
#include <system_error> // `std::error_code`, for the temporary directory
#include <vector>       // `std::vector`

#include <fmt/format.h>

#include "usearch/dense/nvme_indexes.hpp"
#include "usearch/dense/ram_indexes.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** The dense NVMe view over the kits the ladders pick for @p dtype_ under @p caps_, as @c src/
 *  spells them for a CPU class. */
template <typename key_type_, nk_dtype_t dtype_, nk_capability_t caps_, typename engine_type_ = default_io_engine_t>
using nvme_indexes_for =
    nvme_indexes<key_type_, typename quads_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                 typename tiles_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                 typename elementwise_ladder<caps_>::kit_t,
                 typename rmscast_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                 typename visits_hash_set_ladder<caps_>::kit_t, engine_type_>;

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

/** The key numbered @p number: the number itself, or a @c uuid_t holding it in its low word and its
 *  complement in the high one, so both halves of a 16-byte key differ between rows. */
template <typename key_type_>
key_type_ nvme_key(u64_t number) noexcept {
    if constexpr (std::is_same_v<key_type_, uuid_t>)
        return std::bit_cast<uuid_t>(std::array<u64_t, 2> {number, ~number});
    else return static_cast<key_type_>(number);
}

/** The number @ref nvme_key made @p key from. */
template <typename key_type_>
u64_t nvme_key_number(key_type_ const& key) noexcept {
    if constexpr (std::is_same_v<key_type_, uuid_t>) {
        std::array<u64_t, 2> const words = std::bit_cast<std::array<u64_t, 2>>(key);
        return words[0];
    }
    else return static_cast<u64_t>(key);
}

/** The file @p name in the temporary directory, which follows @c TMPDIR. */
inline std::string temporary_path(std::string_view name) {
    std::error_code error;
    return (std::filesystem::temp_directory_path(error) / name).string();
}

/** Where one NVMe scenario's checkpoint lands under @c TMPDIR, named per process and scenario, and
 *  removed with it. */
struct nvme_checkpoint_t {

    /** Where a distributed save writes its manifest, its parts beside it. */
    std::array<char, 256> manifest {};

    /** Where a monolith save writes its one file. */
    std::array<char, 256> monolith {};

    /** Parts the manifest names, which the removal deletes beside it. */
    std::size_t parts = 0;

    explicit nvme_checkpoint_t(std::string_view scenario) noexcept {
        u64_t const salt = mix_bits(reinterpret_cast<std::uintptr_t>(this) ^ fnv1a64(scenario)) & 0xFFFFFFFFu;
        std::string const stem = temporary_path(fmt::format("usearch-nvme-{:08x}.safegraphs", salt));
        fmt::format_to_n(manifest.data(), manifest.size() - 1, "{}.json", stem);
        fmt::format_to_n(monolith.data(), monolith.size() - 1, "{}", stem);
    }
    nvme_checkpoint_t(nvme_checkpoint_t const&) = delete;
    nvme_checkpoint_t& operator=(nvme_checkpoint_t const&) = delete;
    ~nvme_checkpoint_t() noexcept { remove(); }

    /** The path of part @p part beside the manifest, formatted into @p buffer. */
    std::string_view part_path(growing_buffer<char>& buffer, std::size_t part) const noexcept {
        auto const [path, status] = safegraphs_part_path(buffer, safegraphs_stem(manifest.data()), part, parts);
        return succeeded(status) ? path : std::string_view {};
    }

    /** Removes the manifest, its parts, and the monolith. */
    void remove() const noexcept {
        std::remove(manifest.data());
        std::remove(monolith.data());
        growing_buffer<char> buffer {global_allocator};
        for (std::size_t part = 0; part != parts; ++part)
            if (std::string_view const path = part_path(buffer, part); !path.empty()) std::remove(path.data());
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
        return indexes.search(queries, executor, traversals..., config, keys_span(), distances_span(),
                              std::span {counts});
    }

    nk::matrix_span<key_t> keys_span() noexcept { return {keys.data(), counts.size(), wanted}; }
    nk::matrix_span<distance_t> distances_span() noexcept { return {distances.data(), counts.size(), wanted}; }

    std::span<key_t const> keys_of(std::size_t query) const noexcept {
        return {keys.data() + query * wanted, counts[query]};
    }
    std::span<distance_t const> distances_of(std::size_t query) const noexcept {
        return {distances.data() + query * wanted, counts[query]};
    }
};

/** The RAM group an NVMe view is judged against: @p rows rows keyed @c 1+row unless @p key_of says
 *  otherwise, built under @p kernel and saved under @c TMPDIR as the @p scenario checkpoint by
 *  @ref view. */
template <typename nvme_type_>
struct nvme_reference {
    using nvme_t = nvme_type_;
    using key_t = typename nvme_t::key_t;
    using scalar_t = typename nvme_t::scalar_t;
    using ram_t =
        ram_indexes<key_t, typename nvme_t::quads_kit_t, typename nvme_t::tiles_kit_t,
                    typename nvme_t::elementwise_kit_t, typename nvme_t::rmscast_kit_t, typename nvme_t::visits_kit_t>;

    dense_indexes_executor<ram_t> builder;
    ram_t built;
    nk::matrix<scalar_t> vectors;
    std::vector<key_t> keys;
    nvme_checkpoint_t checkpoint;
    std::size_t failures = 0;

    template <typename key_of_type_ = key_t (*)(std::size_t) noexcept>
    nvme_reference(std::string_view scenario, nk_kernel_kind_t kernel, std::size_t rows, std::size_t dimensions,
                   u64_t seed, key_of_type_&& key_of = &default_key_) noexcept
        : built(kernel, forkunion_allocator_t {no_memory_domain_k}),
          vectors(make_vectors<scalar_t>(rows, dimensions, stream_key(seed, scenario))), keys(rows),
          checkpoint(scenario) {
        for (std::size_t row = 0; row != rows; ++row) keys[row] = key_of(row);
        if (failed(builder.status)) {
            failures += expect(false);
            return;
        }
        failures += dense_indexes_build(built, builder.executor, vectors, keys, dimensions);
    }

    /** Saves the group in @p layout at @p alignment and views the checkpoint through @p viewed. */
    void view(nvme_t& viewed, checkpoint_layout_t layout,
              std::size_t alignment = safegraphs_alignment_default_k) noexcept {
        save_config_t config;
        config.layout = layout;
        config.alignment = alignment;
        char const* const path = layout == checkpoint_layout_t::distributed_k ? checkpoint.manifest.data()
                                                                              : checkpoint.monolith.data();
        checkpoint.parts = built.shards_count();
        failures += expect(succeeded(built.save(path, config, builder.executor)));
        failures += expect_eq(viewed.view(path), status_t::success_k);
    }

    /** The RAM group's rankings of @p queries under a walk as wide as the index, which ranks every
     *  row. */
    nvme_rankings<nvme_t> exhaustive_rankings(typename nvme_t::vectors_view_t queries, std::size_t wanted) {
        nvme_rankings<nvme_t> ranked {queries.extent(0), wanted};
        failures += expect(succeeded(
            ranked.search(built, queries, builder.executor, vectors.extent(0), to_writable_span(builder.traversals))
                .status));
        return ranked;
    }

    /** The row a key names. */
    static std::size_t row_of(key_t key) noexcept { return static_cast<std::size_t>(nvme_key_number(key)) - 1; }

    /** The first @p count rows, which the scenarios query with. */
    nk::matrix_view<scalar_t> leading_rows(std::size_t count) const noexcept {
        nk::matrix_view<scalar_t> const rows = vectors.view();
        return rows(nk::range(std::size_t {0}, count), nk::slice);
    }

  private:
    static key_t default_key_(std::size_t row) noexcept { return nvme_key<key_t>(1 + row); }
};

/** Every query of @p found ranks like @p wanted: each key a distinct row at its own exact distance
 *  from the query, so ties may swap keys but never distances. */
template <typename reference_type_, typename rankings_type_>
std::size_t nvme_expect_rankings(reference_type_ const& reference,
                                 typename reference_type_::nvme_t::vectors_view_t queries, rankings_type_ const& found,
                                 rankings_type_ const& wanted) {
    using nvme_t = typename reference_type_::nvme_t;
    using scalar_t = typename nvme_t::scalar_t;
    std::size_t failures = 0;
    for (std::size_t query = 0; query != found.counts.size(); ++query) {
        auto const found_keys = found.keys_of(query);
        auto const found_distances = found.distances_of(query), wanted_distances = wanted.distances_of(query);
        failures += expect_eq(found_keys.size(), wanted_distances.size());
        std::vector<bool> seen(reference.keys.size());
        auto const query_vector = queries.row(query).as_vector();
        auto const query_norms = norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(query_vector);
        for (std::size_t rank = 0; rank != min_of(found_keys.size(), wanted_distances.size()); ++rank) {
            std::size_t const row = reference_type_::row_of(found_keys[rank]);
            failures += expect(row < seen.size() && !seen[row]);
            if (row >= seen.size() || seen[row]) continue;
            seen[row] = true;
            auto const stored = reference.vectors.view().row(row).as_vector();
            auto const stored_norms = norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(stored);
            f32_t const exact = distance<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(
                reference.built.kernel(), query_vector, query_norms, stored, stored_norms);
            failures += expect_near(found_distances[rank], exact);
            failures += expect_near(found_distances[rank], wanted_distances[rank]);
        }
    }
    return failures;
}

/** Search on NVMe against the RAM group its parts came from: under a walk as wide as the index both
 *  rank every row, so every query's ranking matches rank for rank, each key at its exact distance;
 *  a default walk keeps its candidates ascending over distinct rows at their exact distances. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_search(u64_t seed) {
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 8, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_search", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, checkpoint_layout_t::distributed_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    failures += expect_eq(viewed.size(), rows_count);
    failures += expect_eq(viewed.shards_count(), reference.built.shards_count());
    if (failed(driver.status) || failures) return failures + expect(false);

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    auto const [wide_result, wide_status] = nvme.search(viewed, queries, driver.executor, rows_count);
    failures += expect_eq(wide_status, status_t::success_k);
    failures += expect_eq(wide_result.computed_distances, queries_count * rows_count);
    failures += expect_eq(wide_result.value_reads.count, wide_result.computed_distances);
    failures += expect(nvme.counts == ram.counts);
    failures += nvme_expect_rankings(reference, queries, nvme, ram);

    // A default walk reaches far fewer rows, yet whatever it ranks it ranks exactly.
    auto const [narrow_result, narrow_status] = nvme.search(viewed, queries, driver.executor, 0);
    failures += expect_eq(narrow_status, status_t::success_k);
    failures += expect(narrow_result.computed_distances < queries_count * rows_count);
    failures += nvme_expect_rankings(reference, queries, nvme, nvme);
    for (std::size_t query = 0; query != queries_count; ++query) {
        failures += expect_eq(nvme.counts[query], wanted);
        failures += expect(std::is_sorted(nvme.distances_of(query).begin(), nvme.distances_of(query).end()));
    }
    return failures;
}

/** Expects @p index to rank @p queries under one gathering predicate per thread of @p executor,
 *  each carrying @p payload_type_ and admitting what @p per_key admits, as under @p per_key asked a
 *  quad at a time; a RAM group also takes its @p traversals, one per thread. */
template <typename payload_type_, typename reference_type_, typename index_type_, typename executor_type_,
          typename per_key_type_, typename... traversals_types_>
std::size_t nvme_expect_gathered(reference_type_ const& reference, index_type_& index, executor_type_& executor,
                                 typename index_type_::search_config_t config,
                                 typename reference_type_::nvme_t::vectors_view_t queries, per_key_type_ per_key,
                                 std::size_t wanted, traversals_types_... traversals) {
    using nvme_t = typename reference_type_::nvme_t;
    std::size_t const queries_count = queries.extent(0);
    nvme_rankings<nvme_t> immediate {queries_count, wanted}, gathered {queries_count, wanted};
    auto family = gathering_per_thread<typename nvme_t::key_t, payload_type_>(per_key, executor.size());
    std::size_t failures = 0;
    failures += expect(
        succeeded(index
                      .search_if(queries, executor, traversals..., config, immediate.keys_span(),
                                 immediate.distances_span(), std::span {immediate.counts}, quad_predicate(per_key))
                      .status));
    failures += expect(succeeded(index
                                     .search_if(queries, executor, traversals..., config, gathered.keys_span(),
                                                gathered.distances_span(), std::span {gathered.counts}, family)
                                     .status));
    failures += expect(gathered.counts == immediate.counts);
    failures += nvme_expect_rankings(reference, queries, gathered, immediate);
    for (std::size_t query = 0; query != queries_count; ++query)
        for (typename nvme_t::key_t const key : gathered.keys_of(query))
            failures += expect(per_key(key) == admission_t::admit_k);
    return failures;
}

/** Filtered search on NVMe against the RAM group: a per-key predicate reads every scored row's
 *  key, and under a walk as wide as the index both rank every admitted row alike, whether it
 *  answers each quad or gathers per thread until each round settles. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_filtered(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using small_size_t = typename nvme_type_::index_t::small_size_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 4, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_filtered", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, checkpoint_layout_t::distributed_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    if (failed(driver.status) || failures) return failures + expect(false);

    auto const is_odd = [](key_t key) noexcept {
        return nvme_key_number(key) % 2 ? admission_t::admit_k : admission_t::reject_k;
    };
    auto const odd = quad_predicate(is_odd);
    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> ram {queries_count, wanted}, nvme {queries_count, wanted};
    failures += expect(
        succeeded(reference.built
                      .search_if(queries, reference.builder.executor, to_writable_span(reference.builder.traversals),
                                 {static_cast<small_size_t>(rows_count)}, ram.keys_span(), ram.distances_span(),
                                 std::span {ram.counts}, odd)
                      .status));
    failures += expect(succeeded(viewed
                                     .search_if(queries, driver.executor, {static_cast<small_size_t>(rows_count)},
                                                nvme.keys_span(), nvme.distances_span(), std::span {nvme.counts}, odd)
                                     .status));
    failures += expect(nvme.counts == ram.counts);
    failures += nvme_expect_rankings(reference, queries, nvme, ram);
    for (std::size_t query = 0; query != queries_count; ++query) {
        failures += expect_eq(nvme.counts[query], wanted);
        for (key_t const key : nvme.keys_of(query)) failures += expect(nvme_key_number(key) % 2);
    }
    using ram_t = typename nvme_reference<nvme_type_>::ram_t;
    typename ram_t::search_config_t ram_config {static_cast<small_size_t>(rows_count)};
    typename nvme_type_::search_config_t nvme_config {static_cast<small_size_t>(rows_count)};
    failures += nvme_expect_gathered<typename ram_t::walk_payload_t>(
        reference, reference.built, reference.builder.executor, ram_config, queries, is_odd, wanted,
        to_writable_span(reference.builder.traversals));
    failures += nvme_expect_gathered<typename nvme_type_::walk_payload_t>(reference, viewed, driver.executor,
                                                                          nvme_config, queries, is_odd, wanted);
    ram_config.traversal = traversal_t::scan_k, nvme_config.traversal = traversal_t::scan_k;
    failures += nvme_expect_gathered<typename ram_t::scan_payload_t>(
        reference, reference.built, reference.builder.executor, ram_config, queries, is_odd, wanted,
        to_writable_span(reference.builder.traversals));
    failures += nvme_expect_gathered<typename nvme_type_::scan_payload_t>(reference, viewed, driver.executor,
                                                                          nvme_config, queries, is_odd, wanted);

    // One key in eight still fills every row at the default breadth, as each walk stops on what it
    // admitted rather than on what it scored.
    auto const one_in_eight = quad_predicate(
        [](key_t key) noexcept { return nvme_key_number(key) % 8 ? admission_t::reject_k : admission_t::admit_k; });
    failures += expect(succeeded(viewed
                                     .search_if(queries, driver.executor, {}, nvme.keys_span(), nvme.distances_span(),
                                                std::span {nvme.counts}, one_in_eight)
                                     .status));
    for (std::size_t query = 0; query != queries_count; ++query) {
        failures += expect_eq(nvme.counts[query], wanted);
        for (key_t const key : nvme.keys_of(query)) failures += expect(nvme_key_number(key) % 8 == 0);
    }
    return failures;
}

/** The key scans against the RAM group's lookups, a key shared by every third row and absent keys
 *  included: every count, verdict and true total agrees, and every key's exported rows are its
 *  earliest in placement order, as the cursor reads them; an output narrower than a stored row is
 *  refused. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_key_scans(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, wanted = 2;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    key_t const shared = nvme_key<key_t>(100000);

    auto const key_of = [&](std::size_t row) noexcept { return row % 3 ? nvme_key<key_t>(1 + row) : shared; };
    nvme_reference<nvme_type_> reference {"nvme_scans", kernel_, rows_count, dimensions, seed, key_of};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, checkpoint_layout_t::distributed_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    if (failed(driver.status) || failures) return failures + expect(false);

    std::array<key_t, 6> const probes {shared,
                                       nvme_key<key_t>(2),
                                       nvme_key<key_t>(1),
                                       shared,
                                       nvme_key<key_t>(rows_count + 7),
                                       nvme_key<key_t>(rows_count)};
    std::span<key_t const> const probed {probes};
    std::array<std::size_t, probes.size()> ram_counted {}, nvme_counted {}, ram_contained {}, nvme_contained {},
        ram_gotten {}, nvme_gotten {};
    auto ram_rows = make_vectors<scalar_t>(probes.size() * wanted, dimensions, 0);
    auto nvme_rows = make_vectors<scalar_t>(probes.size() * wanted, dimensions, 0);
    auto& ram_executor = reference.builder.executor;
    failures += expect_eq(reference.built.count(probed, std::span {ram_counted}, ram_executor), status_t::success_k);
    failures += expect_eq(viewed.count(probed, std::span {nvme_counted}, driver.executor), status_t::success_k);
    failures += expect_eq(reference.built.contains(probed, std::span {ram_contained}, ram_executor),
                          status_t::success_k);
    failures += expect_eq(viewed.contains(probed, std::span {nvme_contained}, driver.executor), status_t::success_k);
    failures += expect_eq(reference.built.get(probed, ram_rows.span(), wanted, std::span {ram_gotten}, ram_executor),
                          status_t::success_k);
    failures += expect_eq(viewed.get(probed, nvme_rows.span(), wanted, std::span {nvme_gotten}, driver.executor),
                          status_t::success_k);
    // The whole view in placement order, which ranks every key's rows by part and row.
    std::vector<key_t> placed_keys(rows_count);
    auto placed_rows = make_vectors<scalar_t>(rows_count, dimensions, 0);
    typename nvme_type_::cursor_t cursor {viewed};
    failures += expect_eq(cursor.read(viewed, std::span {placed_keys}, placed_rows.span(), driver.executor).status,
                          status_t::success_k);
    std::size_t const row_bytes = dense_rows_bytes(reference.vectors.view(), 1);
    for (std::size_t probe = 0; probe != probes.size(); ++probe) {
        failures += expect_eq(nvme_counted[probe], ram_counted[probe]);
        failures += expect_eq(nvme_contained[probe], ram_contained[probe]);
        failures += expect_eq(nvme_gotten[probe], ram_gotten[probe]);
        std::size_t rank = 0;
        for (std::size_t placed = 0; placed != rows_count && rank != min_of(nvme_gotten[probe], wanted); ++placed) {
            if (!(placed_keys[placed] == probes[probe])) continue;
            failures += expect(std::memcmp(nvme_rows.view().row(probe * wanted + rank).as_vector().byte_data(),
                                           placed_rows.view().row(placed).as_vector().byte_data(), row_bytes) == 0);
            ++rank;
        }
        failures += expect_eq(rank, min_of(nvme_gotten[probe], wanted));
    }
    failures += expect_eq(nvme_counted[0], divide_round_up(rows_count, std::size_t {3}));
    failures += expect_eq(nvme_contained[4], 0u);
    failures += expect_eq(viewed.get(probed, nvme_rows.span(), wanted + 1, std::span {nvme_gotten}, driver.executor),
                          status_t::config_mismatch_k);
    auto narrow_rows = make_vectors<scalar_t>(probes.size() * wanted, dense_dimensions_for<scalar_t>(1), 0);
    failures += expect_eq(viewed.get(probed, narrow_rows.span(), wanted, std::span {nvme_gotten}, driver.executor),
                          status_t::config_mismatch_k);
    failures += expect_eq(
        reference.built.get(probed, narrow_rows.span(), wanted, std::span {ram_gotten}, reference.builder.executor),
        status_t::config_mismatch_k);
    return failures;
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

/** The file a kernel engine reads a @ref nvme_reblocked_medium through. */
template <typename wanted_type_, u32_t block_bytes_>
    requires std::is_same_v<wanted_type_, file_medium_t>
file_medium_t const* medium_as(nvme_reblocked_medium<block_bytes_> const& medium) noexcept {
    return &medium.file;
}

/** @p engine_type_ reading @p medium_type_ media, every member forwarded, whose submits fail on
 *  request: the one that brings @ref submits_to_failure down to zero lands what it submitted, then
 *  drops it, as a refused submit does. */
template <typename engine_type_, typename medium_type_ = typename engine_type_::medium_t>
struct nvme_composed_engine {
    using wait_t = typename engine_type_::wait_t;
    using allocator_t = typename engine_type_::allocator_t;
    using medium_t = medium_type_;
    using config_t = typename engine_type_::config_t;

    /** Submits left until one fails, across every engine of this type; zero never fails. */
    inline static std::atomic<std::size_t> submits_to_failure {0};

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
        std::size_t left = submits_to_failure.load(std::memory_order_relaxed);
        while (left && !submits_to_failure.compare_exchange_weak(left, left - 1, std::memory_order_relaxed)) {}
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
 *  aligned to only 64 bytes, read at 512- and 4096-byte blocks, rank every row exactly like the RAM
 *  group. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_, u32_t block_bytes_>
std::size_t test_nvme_indexes_straddling(u64_t seed) {
    using engine_t = nvme_composed_engine<engine_type_, nvme_reblocked_medium<block_bytes_>>;
    using nvme_t = nvme_indexes_for<u32_t, dtype_, nk_cap_serial_k, engine_t>;
    constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 4, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_t::scalar_t>(100);

    nvme_reference<nvme_t> reference {"nvme_straddling", kernel_, rows_count, dimensions, seed};
    nvme_t viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, checkpoint_layout_t::distributed_k, 64);
    dense_indexes_executor<nvme_t> driver;
    std::size_t failures = reference.failures;
    if (failed(driver.status) || failures) return failures + expect(false);

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_t> const ram = reference.exhaustive_rankings(queries, wanted);
    nvme_rankings<nvme_t> nvme {queries_count, wanted};
    failures += expect(succeeded(nvme.search(viewed, queries, driver.executor, rows_count).status));
    failures += nvme_expect_rankings(reference, queries, nvme, ram);

    // Neighbors rows straddle blocks too, so the audit's merged reads must still land every row
    // whole.
    auto const [ram_health, ram_status] = reference.built.health(reference.builder.executor);
    auto const [nvme_health, nvme_status] = viewed.health(driver.executor);
    failures += expect_eq(nvme_status, ram_status);
    failures += expect_eq(nvme_health.reachable_nodes, ram_health.reachable_nodes);
    return failures;
}

/** The cursor exports the rows the RAM group's cursor does, keys and vectors alike and in placement
 *  order, through reads that cross the parts' ends; then it answers no more rows. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_cursor(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, chunk = 37;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_cursor", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, checkpoint_layout_t::distributed_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    if (failed(driver.status) || failures) return failures + expect(false);

    typename nvme_reference<nvme_type_>::ram_t::cursor_t ram_cursor {reference.built};
    typename nvme_type_::cursor_t nvme_cursor {viewed};
    std::vector<key_t> ram_keys(chunk), nvme_keys(chunk);
    auto ram_rows = make_vectors<scalar_t>(chunk, dimensions, 0),
         nvme_rows = make_vectors<scalar_t>(chunk, dimensions, 0);
    std::size_t const row_bytes = dense_rows_bytes(reference.vectors.view(), 1);
    std::size_t exported = 0;
    while (true) {
        std::size_t const ram_read = ram_cursor.read(reference.built, std::span {ram_keys}, ram_rows.span());
        auto const [nvme_read, status] = nvme_cursor.read(viewed, std::span {nvme_keys}, nvme_rows.span(),
                                                          driver.executor);
        failures += expect_eq(status, status_t::success_k);
        failures += expect_eq(nvme_read.count, ram_read);
        if (!ram_read || nvme_read.count != ram_read) break;
        failures += expect(nvme_read.scan_reads.count >= 2);
        for (std::size_t row = 0; row != ram_read; ++row) {
            failures += expect(nvme_keys[row] == ram_keys[row]);
            failures += expect(std::memcmp(nvme_rows.view().row(row).as_vector().byte_data(),
                                           ram_rows.view().row(row).as_vector().byte_data(), row_bytes) == 0);
        }
        exported += ram_read;
    }
    failures += expect_eq(exported, rows_count);
    return failures;
}

/** The reachability audit agrees with the RAM group's on the same parts: the same rows audited, the
 *  same reached. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_health(u64_t seed) {
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_health", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, checkpoint_layout_t::distributed_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    if (failed(driver.status) || failures) return failures + expect(false);

    auto const [ram, ram_status] = reference.built.health(reference.builder.executor);
    auto const [nvme, nvme_status] = viewed.health(driver.executor);
    failures += expect_eq(ram_status, status_t::success_k);
    failures += expect_eq(nvme_status, status_t::success_k);
    failures += expect_eq(nvme.audited_nodes, rows_count);
    failures += expect_eq(nvme.audited_nodes, ram.audited_nodes);
    failures += expect_eq(nvme.reachable_nodes, ram.reachable_nodes);
    failures += expect(nvme.sweeps > 0);
    return failures;
}

/** Per-row latencies and reads by purpose: every row's latency lands, positive; a search reads one
 *  value row per distance, one neighbors row per expansion and one key per written result; a
 *  filtered search reads a key beside every value row and one more per written result; attention
 *  and a filtered product read a key beside every value row and resolve none, and over parts of
 *  another kernel attention, multiply and maxsim refuse with @c config_mismatch_k. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_telemetry(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    using vector_view_t = typename nvme_type_::vector_view_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 8, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_telemetry", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, checkpoint_layout_t::distributed_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    if (failed(driver.status) || failures) return failures + expect(false);

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    std::vector<u64_t> latencies(queries_count);
    auto expect_latencies = [&] {
        std::size_t missing = 0;
        for (u64_t& latency : latencies) missing += expect(latency > 0), latency = 0;
        return missing;
    };
    std::size_t const value_bytes = dense_rows_bytes(reference.vectors.view(), 1);
    std::size_t const neighbor_bytes = viewed.connectivity() * sizeof(node_id_t);

    auto const [searched, search_status] = viewed.search(queries, driver.executor, {}, nvme.keys_span(),
                                                         nvme.distances_span(), std::span {nvme.counts},
                                                         std::span {latencies});
    failures += expect_eq(search_status, status_t::success_k);
    failures += expect_latencies();
    failures += expect_eq(searched.neighbor_reads.count, searched.visited_members);
    failures += expect_eq(searched.value_reads.count, searched.computed_distances);
    failures += expect_eq(searched.key_reads.count, queries_count * wanted);
    failures += expect_eq(searched.scan_reads.count, 0u);
    failures += expect(searched.neighbor_reads.bytes >= searched.neighbor_reads.count * neighbor_bytes);
    failures += expect(searched.value_reads.bytes >= searched.value_reads.count * value_bytes);
    failures += expect(searched.key_reads.bytes >= searched.key_reads.count * sizeof(key_t));

    auto const odd = quad_predicate(
        [](key_t key) noexcept { return nvme_key_number(key) % 2 ? admission_t::admit_k : admission_t::reject_k; });
    auto const [filtered, filtered_status] = viewed.search_if(queries, driver.executor, {}, nvme.keys_span(),
                                                              nvme.distances_span(), std::span {nvme.counts}, odd,
                                                              std::span {latencies});
    failures += expect_eq(filtered_status, status_t::success_k);
    failures += expect_latencies();
    failures += expect(filtered.count > 0);
    failures += expect_eq(filtered.key_reads.count, filtered.value_reads.count + filtered.count);

    auto lookup = [&](std::span<key_t const> batch, std::span<vector_view_t> rows) noexcept {
        for (std::size_t lane = 0; lane != batch.size(); ++lane)
            rows[lane] = reference.vectors.view().row(reference.row_of(batch[lane])).as_vector();
    };
    auto outputs = nk::matrix<f32_t>::zeros({queries_count, dimensions}).value;
    std::vector<softmax_state_t> carry(queries_count);
    auto const [attended, attend_status] = viewed.attend(queries, 1, lookup, 1.0f / f32_t(dimensions), outputs.span(),
                                                         std::span {carry}, driver.executor, {}, std::span {latencies});
    // Every query row is a query of its own.
    std::vector<std::size_t> query_offsets(queries_count + 1);
    for (std::size_t query = 0; query != query_offsets.size(); ++query) query_offsets[query] = query;
    if constexpr (kernel_ == nk_kernel_dot_k) {
        failures += expect_eq(attend_status, status_t::success_k);
        failures += expect_latencies();
        failures += expect(attended.value_reads.count > 0);
        failures += expect_eq(attended.key_reads.count, attended.value_reads.count);
        failures += expect_eq(attended.value_reads.count, attended.computed_distances);

        // Filtered MaxSim ranks odd keys alone, whether walks or a scan nominate them.
        auto keys = nk::matrix<key_t>::zeros({queries_count, wanted}).value;
        auto scores = nk::matrix<f64_t>::zeros({queries_count, wanted}).value;
        std::vector<std::size_t> counts(queries_count);
        typename nvme_type_::search_config_t nominating;
        for (traversal_t const traversal : {traversal_t::walk_k, traversal_t::scan_k}) {
            nominating.traversal = traversal;
            failures += expect_eq(viewed
                                      .maxsim_if(queries, std::span<std::size_t const> {query_offsets}, driver.executor,
                                                 nominating, keys.span(), scores.span(), std::span {counts}, odd)
                                      .status,
                                  status_t::success_k);
            for (std::size_t query = 0; query != queries_count; ++query) {
                failures += expect(counts[query] > 0);
                for (std::size_t rank = 0; rank != counts[query]; ++rank)
                    failures += expect(nvme_key_number(keys(query, rank)) % 2 == 1);
            }
        }

        // Filtered products write odd keys alone, read beside their values with none resolved
        // after, and refuse a scan; attention folds no key a predicate rejects.
        auto products = nk::matrix<typename nvme_type_::dot_result_t>::zeros({queries_count, wanted}).value;
        auto const [multiplied, multiply_status] = viewed.multiply_if(queries, driver.executor, {}, keys.span(),
                                                                      products.span(), std::span {counts}, odd);
        failures += expect_eq(multiply_status, status_t::success_k);
        failures += expect_eq(multiplied.key_reads.count, multiplied.value_reads.count);
        for (std::size_t query = 0; query != queries_count; ++query) {
            failures += expect(counts[query] > 0);
            for (std::size_t pair = 0; pair != min_of(counts[query], wanted); ++pair)
                failures += expect(nvme_key_number(keys(query, pair)) % 2 == 1);
        }
        typename nvme_type_::search_config_t scan;
        scan.traversal = traversal_t::scan_k;
        failures += expect_eq(
            viewed.multiply_if(queries, driver.executor, scan, keys.span(), products.span(), std::span {counts}, odd)
                .status,
            status_t::unsupported_k);
        auto const no_rows = [](std::span<key_t const>, std::span<vector_view_t>) noexcept {};
        auto const reject_all = quad_predicate([](key_t) noexcept { return admission_t::reject_k; });
        failures += expect_eq(viewed
                                  .attend_if(queries, 1, no_rows, 1.0f, outputs.span(), std::span {carry},
                                             driver.executor, {}, reject_all)
                                  .status,
                              status_t::success_k);
        for (softmax_state_t const& state : carry) failures += expect_eq(state.partition, 0.0f);
    }
    else {
        // Products are dots, so parts of another kernel refuse every algebra verb before any read.
        auto keys = nk::matrix<key_t>::zeros({queries_count, wanted}).value;
        auto products = nk::matrix<typename nvme_type_::dot_result_t>::zeros({queries_count, wanted}).value;
        auto scores = nk::matrix<f64_t>::zeros({queries_count, wanted}).value;
        std::vector<std::size_t> counts(queries_count);
        failures += expect_eq(attend_status, status_t::config_mismatch_k);
        failures += expect_eq(
            viewed.multiply(queries, driver.executor, {}, keys.span(), products.span(), std::span {counts}).status,
            status_t::config_mismatch_k);
        failures += expect_eq(viewed
                                  .maxsim(queries, std::span<std::size_t const> {query_offsets}, driver.executor, {},
                                          keys.span(), scores.span(), std::span {counts})
                                  .status,
                              status_t::config_mismatch_k);
        failures += expect_eq(attended.value_reads.count, 0u);
    }
    return failures;
}

/** Views keep no executor: one executor serves a manifest view and a monolith view of the same rows
 *  in turn, and a second executor serves the first view alike; a claimed executor refuses a second
 *  caller, and one asked for a transport its engine does not drive refuses to spawn. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_executors(u64_t seed) {
    using executor_t = typename nvme_type_::executor_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 4, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_executors", kernel_, rows_count, dimensions, seed};
    nvme_type_ parts {forkunion_allocator_t {no_memory_domain_k}}, whole {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(parts, checkpoint_layout_t::distributed_k);
    reference.view(whole, checkpoint_layout_t::monolith_k);
    dense_indexes_executor<nvme_type_> first, second;
    std::size_t failures = reference.failures;
    failures += expect_eq(parts.shards_count(), reference.built.shards_count());
    failures += expect_eq(whole.shards_count(), 1u);
    if (failed(first.status) || failed(second.status) || failures) return failures + expect(false);

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    auto expect_exact = [&](nvme_type_& viewed, executor_t& executor) {
        std::size_t const refused = expect(succeeded(nvme.search(viewed, queries, executor, rows_count).status));
        return refused + nvme_expect_rankings(reference, queries, nvme, ram);
    };
    failures += expect_exact(parts, first.executor);
    failures += expect_exact(whole, first.executor);
    failures += expect_exact(parts, first.executor);
    failures += expect_exact(parts, second.executor);

    auto [claim, claim_status] = first.executor.claim();
    failures += expect_eq(claim_status, status_t::success_k);
    failures += expect_eq(nvme.search(parts, queries, first.executor, 0).status, status_t::contended_k);

    io_engine_config_t foreign;
    foreign.queue_depth = nvme_queue_depth_k;
    foreign.transport = executor_t::engine_t::transport() == io_transport_t::blocking_k ? io_transport_t::linux_aio_k
                                                                                        : io_transport_t::blocking_k;
    failures += expect_eq(executor_t::spawn(first.topology, 1, foreign).status, status_t::unsupported_k);
    return failures;
}

/** A read that lands short of its row stops the call: a part truncated under a live view fails
 *  search, filtered search and the key scans with @c io_k, again on every retry, and the very
 *  executor then serves an intact view exactly, so every engine drained and every buffer
 *  returned. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_failed_read(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 8, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_failed_read", kernel_, rows_count, dimensions, seed};
    nvme_type_ broken {forkunion_allocator_t {no_memory_domain_k}}, intact {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(broken, checkpoint_layout_t::distributed_k);
    reference.view(intact, checkpoint_layout_t::monolith_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    if (failed(driver.status) || failures) return failures + expect(false);
    {
        growing_buffer<char> part_path {global_allocator};
        auto [file, status] = seekable_file_t::open(reference.checkpoint.part_path(part_path, 1), file_mode_t::write_k);
        failures += expect_eq(status, status_t::success_k);
    }

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    auto const any = quad_predicate([](key_t) noexcept { return admission_t::admit_k; });
    for (std::size_t attempt = 0; attempt != 2; ++attempt) {
        failures += expect_eq(nvme.search(broken, queries, driver.executor, 0).status, status_t::io_k);
        failures += expect_eq(broken
                                  .search_if(queries, driver.executor, {}, nvme.keys_span(), nvme.distances_span(),
                                             std::span {nvme.counts}, any)
                                  .status,
                              status_t::io_k);
        std::array<std::size_t, 1> entries {};
        failures += expect_eq(
            broken.count(std::span<key_t const> {reference.keys}.first(1), std::span {entries}, driver.executor),
            status_t::io_k);
    }

    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    failures += expect(succeeded(nvme.search(intact, queries, driver.executor, rows_count).status));
    return failures + nvme_expect_rankings(reference, queries, nvme, ram);
}

/** A submit failing midway, its reads issued and then dropped, fails a walk or a scan with @c io_k
 *  wherever it strikes, first submit or later, several hops of a walk in flight or one, and the
 *  very executors then serve both exactly, nothing stranded in an engine. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_failed_submit(u64_t seed) {
    using engine_t = nvme_composed_engine<engine_type_>;
    using nvme_t = nvme_indexes_for<u32_t, dtype_, nk_cap_serial_k, engine_t>;
    using key_t = typename nvme_t::key_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 8, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_t::scalar_t>(dense_values_k);

    nvme_reference<nvme_t> reference {"nvme_failed_submit", kernel_, rows_count, dimensions, seed};
    nvme_t viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, checkpoint_layout_t::distributed_k);
    dense_indexes_executor<nvme_t> driver;
    std::size_t failures = reference.failures;
    if (failed(driver.status) || failures) return failures + expect(false);

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_t> nvme {queries_count, wanted};
    std::span<key_t const> const probed = std::span<key_t const> {reference.keys}.first(4);
    std::array<std::size_t, 4> entries {};
    for (std::size_t const countdown : {1, 4}) {
        engine_t::submits_to_failure = countdown;
        failures += expect_eq(nvme.search(viewed, queries, driver.executor, 0).status, status_t::io_k);
        failures += expect_eq(engine_t::submits_to_failure.load(), 0u);
    }
    engine_t::submits_to_failure = 1;
    failures += expect_eq(viewed.count(probed, std::span {entries}, driver.executor), status_t::io_k);
    failures += expect_eq(engine_t::submits_to_failure.load(), 0u);

    // A lone query on a deep queue has several hops of its walks in flight by its second and third
    // submits.
    dense_indexes_executor<nvme_t> deep {nvme_deep_queue_depth_k};
    failures += expect_eq(deep.status, status_t::success_k);
    nvme_rankings<nvme_t> lone {1, wanted};
    for (std::size_t const countdown : {2, 3}) {
        engine_t::submits_to_failure = countdown;
        failures += expect_eq(lone.search(viewed, reference.leading_rows(1), deep.executor, 0).status, status_t::io_k);
        failures += expect_eq(engine_t::submits_to_failure.load(), 0u);
    }

    nvme_rankings<nvme_t> const ram = reference.exhaustive_rankings(queries, wanted);
    failures += expect(succeeded(nvme.search(viewed, queries, driver.executor, rows_count).status));
    failures += nvme_expect_rankings(reference, queries, nvme, ram);
    failures += expect(succeeded(nvme.search(viewed, queries, deep.executor, rows_count).status));
    failures += nvme_expect_rankings(reference, queries, nvme, ram);
    failures += expect_eq(viewed.count(probed, std::span {entries}, driver.executor), status_t::success_k);
    for (std::size_t const entry : entries) failures += expect_eq(entry, 1u);
    return failures;
}

/** Lone queries on a queue their walks leave idle keep several hops in flight, and find at least as
 *  many true neighbors as on a queue a hop fills, their candidates only as wide as their results.
 *  Over blocking reads they also take fewer round trips, and a full batch on the deep queue, which
 *  leaves no depth idle, ranks exactly as the shallow queue does. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_pipelined(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    // Parts of thousands of rows, so a narrow walk scores a fraction of them and misses some true
    // neighbors.
    constexpr std::size_t rows_count = 16 * dense_thread_nodes_k, queries_count = 32, wanted = 10;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_pipelined", kernel_, rows_count, dimensions, seed};
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(viewed, checkpoint_layout_t::distributed_k);
    dense_indexes_executor<nvme_type_> shallow, deep {nvme_deep_queue_depth_k};
    std::size_t failures = reference.failures;
    if (failed(shallow.status) || failed(deep.status) || failures) return failures + expect(false);

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
        failures += expect_eq(hop_by_hop_status, status_t::success_k);
        hop_by_hop_hits += hits(query), hop_by_hop_trips += hop_by_hop.round_trips;
        auto const [pipelined, pipelined_status] = lone.search(viewed, row, deep.executor, wanted);
        failures += expect_eq(pipelined_status, status_t::success_k);
        failures += nvme_expect_rankings(reference, row, lone, lone);
        pipelined_hits += hits(query), pipelined_trips += pipelined.round_trips;
    }
    failures += expect(pipelined_hits >= hop_by_hop_hits);
    // Kernel engines land a submit's reads over several waits, in any order, and walks break
    // distance ties by landing order, so only blocking reads count one round trip per hop and
    // replay a walk exactly.
    if constexpr (nvme_type_::engine_t::transport() != io_transport_t::blocking_k) return failures;
    failures += expect(pipelined_trips < hop_by_hop_trips);

    nvme_rankings<nvme_type_> hop_by_hop {queries_count, wanted}, full {queries_count, wanted};
    auto const [shallow_result, shallow_status] = hop_by_hop.search(viewed, queries, shallow.executor, wanted);
    auto const [deep_result, deep_status] = full.search(viewed, queries, deep.executor, wanted);
    failures += expect_eq(shallow_status, status_t::success_k);
    failures += expect_eq(deep_status, status_t::success_k);
    failures += expect(full.counts == hop_by_hop.counts && full.keys == hop_by_hop.keys &&
                       full.distances == hop_by_hop.distances);
    // Past the batch's end, deep queues let walks inherit idle flights and hold hops a walk
    // advancing hop by hop never reaches, so they may only do more work, never less.
    failures += expect(deep_result.visited_members >= shallow_result.visited_members);
    failures += expect(deep_result.computed_distances >= shallow_result.computed_distances);
    failures += expect(deep_result.neighbor_reads.count >= shallow_result.neighbor_reads.count);
    return failures;
}

/** Parts served from elsewhere than the manifest points: one through the directory holding it, one
 *  moved under another name and named itself. The view ranks exactly like the RAM group, while one
 *  naming too few parts is refused and one following the manifest no longer opens. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_relocated(u64_t seed) {
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 4, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_relocated", kernel_, rows_count, dimensions, seed};
    nvme_type_ original {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(original, checkpoint_layout_t::distributed_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    failures += expect_eq(reference.checkpoint.parts, 2u);
    if (failed(driver.status) || failures) return failures + expect(false);

    char const* const manifest = reference.checkpoint.manifest.data();
    char const* const moved = reference.checkpoint.monolith.data();
    growing_buffer<char> part_path {global_allocator};
    failures += expect_eq(std::rename(reference.checkpoint.part_path(part_path, 1).data(), moved), 0);
    std::string_view const manifest_path {manifest};
    std::string const directory {manifest_path.substr(0, manifest_path.rfind('/'))};
    std::array<std::string_view, 2> const paths {directory, moved};

    nvme_type_ relocated {forkunion_allocator_t {no_memory_domain_k}};
    failures += expect_eq(relocated.view(manifest, std::span {paths}.first(1)), status_t::config_mismatch_k);
    failures += expect(failed(relocated.view(manifest)));
    failures += expect_eq(relocated.view(manifest, std::span {paths}), status_t::success_k);
    failures += expect_eq(relocated.size(), rows_count);

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    nvme_rankings<nvme_type_> nvme {queries_count, wanted};
    failures += expect(succeeded(nvme.search(relocated, queries, driver.executor, rows_count).status));
    return failures + nvme_expect_rankings(reference, queries, nvme, ram);
}

/** A monolith saved past byte zero of a larger file views through its base like the plain monolith
 *  and leaves the bytes before it as they were, while a base off the file's block is refused, as is
 *  an address in no form @ref file_address_t lists. The base is a multiple of every page a mapping
 *  may start on. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_indexes_byte_base(u64_t seed) {
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 4, wanted = 8;
    constexpr std::size_t byte_base = 65536;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_byte_base", kernel_, rows_count, dimensions, seed};
    nvme_type_ plain {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(plain, checkpoint_layout_t::monolith_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    if (failed(driver.status) || failures) return failures + expect(false);

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
        failures += expect_eq(parse_status, status_t::success_k);
        failures += expect_eq(parsed.location, address.location);
        failures += expect_eq(parsed.namespace_id, address.namespace_id);
        failures += expect_eq(parsed.byte_base, address.byte_base);
    }
    failures += expect_eq(parse_file_address("").status, status_t::invalid_argument_k);
    failures += expect_eq(parse_file_address("/dev/nvme1n1@64GB").status, status_t::invalid_argument_k);

    // The manifest's path, which a monolith save leaves unused, holds the larger file.
    char const* const medium = reference.checkpoint.manifest.data();
    std::vector<byte_t> const leading(byte_base, byte_t {0xA5});
    std::vector<byte_t> kept(byte_base);
    {
        auto [file, open_status] = seekable_file_t::open(medium, file_mode_t::write_k);
        failures += expect(succeeded(open_status) && succeeded(file.write_at(0, leading)));
    }
    std::string const based = fmt::format("{}@{}KiB", medium, byte_base / 1024);
    save_config_t config;
    config.layout = checkpoint_layout_t::monolith_k;
    failures += expect(succeeded(reference.built.save(based, config, reference.builder.executor)));
    {
        auto [file, open_status] = seekable_file_t::open(medium, file_mode_t::read_k);
        failures += expect(succeeded(open_status) && succeeded(file.read_at(0, kept)) && kept == leading);
    }
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    failures += expect_eq(viewed.view(based), status_t::success_k);
    failures += expect_eq(viewed.size(), rows_count);
    std::string const misaligned = fmt::format("{}@{}", medium, byte_base + 1);
    failures += expect_eq(file_medium_t::open(misaligned).status, status_t::invalid_argument_k);
    if (failures) return failures;

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> plain_rankings {queries_count, wanted}, based_rankings {queries_count, wanted};
    failures += expect(succeeded(plain_rankings.search(plain, queries, driver.executor, rows_count).status));
    failures += expect(succeeded(based_rankings.search(viewed, queries, driver.executor, rows_count).status));
    failures += expect(based_rankings.counts == plain_rankings.counts);
    return failures + nvme_expect_rankings(reference, queries, based_rankings, plain_rankings);
}

/** The single-part index over a monolith: under a walk as wide as the part it ranks every row like
 *  the RAM group, each node ID a distinct row at its exact distance, the rows named through the
 *  cursor's placement order. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_index_search(u64_t seed) {
    using index_t = typename nvme_type_::index_t;
    using key_t = typename nvme_type_::key_t;
    using scalar_t = typename nvme_type_::scalar_t;
    using distance_t = typename nvme_type_::distance_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 4, wanted = 8;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_type_> reference {"nvme_part", kernel_, rows_count, dimensions, seed};
    nvme_type_ whole {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(whole, checkpoint_layout_t::monolith_k);
    dense_indexes_executor<nvme_type_> driver;
    std::size_t failures = reference.failures;
    std::unique_ptr<index_t> const part_ptr {new (std::nothrow) index_t(kernel_)};
    index_t& part = *part_ptr;
    failures += expect_eq(part.view(reference.checkpoint.monolith.data()), status_t::success_k);
    failures += expect_eq(part.size(), rows_count);
    if (failed(driver.status) || failures) return failures + expect(false);

    std::vector<key_t> placed_keys(rows_count);
    typename nvme_type_::cursor_t cursor {whole};
    failures += expect_eq(cursor.read(whole, std::span {placed_keys}, {}, driver.executor).status, status_t::success_k);

    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<nvme_type_> const ram = reference.exhaustive_rankings(queries, wanted);
    std::vector<node_id_t> ids(queries_count * wanted);
    std::vector<distance_t> distances(queries_count * wanted);
    std::vector<std::size_t> counts(queries_count);
    typename index_t::search_config_t const config {static_cast<typename index_t::small_size_t>(rows_count)};
    auto const [searched, status] = part.search(
        queries, driver.executor, config, nk::matrix_span<node_id_t> {ids.data(), queries_count, wanted},
        nk::matrix_span<distance_t> {distances.data(), queries_count, wanted}, std::span {counts});
    failures += expect_eq(status, status_t::success_k);
    failures += expect_eq(searched.computed_distances, queries_count * rows_count);
    failures += expect(counts == ram.counts);
    for (std::size_t query = 0; query != queries_count; ++query) {
        std::vector<bool> seen(rows_count);
        auto const query_vector = queries.row(query).as_vector();
        auto const query_norms = norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(query_vector);
        for (std::size_t rank = 0; rank != min_of(counts[query], wanted); ++rank) {
            node_id_t const node = ids[query * wanted + rank];
            failures += expect(node < rows_count && !seen[node]);
            if (node >= rows_count || seen[node]) continue;
            seen[node] = true;
            auto const stored = reference.vectors.view().row(reference.row_of(placed_keys[node])).as_vector();
            auto const stored_norms = norms<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(stored);
            f32_t const exact = distance<typename quads_ladder<scalar_t, nk_cap_serial_k>::kit_t>(
                kernel_, query_vector, query_norms, stored, stored_norms);
            failures += expect_near(distances[query * wanted + rank], exact);
            failures += expect_near(distances[query * wanted + rank], ram.distances[query * wanted + rank]);
        }
    }
    return failures;
}

/** Recall a rewrite may lose against a fresh RAM build of the same rows, as a share of the
 *  results. */
constexpr f64_t rewrite_recall_margin_k = 0.05;

/** Queries every recall is measured over, and results each one keeps. */
constexpr std::size_t rewrite_queries_k = 64, rewrite_wanted_k = 10;

/** Rows the merge, split and refine scenarios rewrite. */
constexpr std::size_t rewrite_rows_k = 4 * dense_thread_nodes_k;

/** The rows one rewrite scenario draws from, the RAM groups it saves them through, the checkpoints
 *  it removes, and every query's exact k-th distance. */
template <typename nvme_type_>
struct rewrite_fixture {
    using nvme_t = nvme_type_;
    using key_t = typename nvme_t::key_t;
    using scalar_t = typename nvme_t::scalar_t;
    using ram_t = typename nvme_reference<nvme_t>::ram_t;

    std::string_view scenario;
    nk_kernel_kind_t kernel;
    nk::matrix<scalar_t> vectors;
    nk::matrix<scalar_t> queries;
    std::vector<key_t> keys;
    std::vector<f32_t> thresholds;
    dense_indexes_executor<ram_t> builder;
    dense_indexes_executor<nvme_t> driver;
    std::vector<std::unique_ptr<nvme_checkpoint_t>> checkpoints;
    std::size_t failures = 0;

    rewrite_fixture(std::string_view scenario, nk_kernel_kind_t kernel, std::size_t rows, std::size_t dimensions,
                    u64_t seed)
        : scenario(scenario), kernel(kernel),
          vectors(make_vectors<scalar_t>(rows, dimensions, stream_key(seed, scenario))),
          queries(make_vectors<scalar_t>(rewrite_queries_k, dimensions, stream_key(seed, scenario, 1))), keys(rows),
          thresholds(rewrite_queries_k) {
        for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);
        failures += expect_eq(builder.status, status_t::success_k);
        failures += expect_eq(driver.status, status_t::success_k);
        std::vector<f32_t> distances(rows);
        for (std::size_t query = 0; query != rewrite_queries_k; ++query) {
            for (std::size_t row = 0; row != rows; ++row) distances[row] = exact_(query, row);
            std::nth_element(distances.begin(), distances.begin() + (rewrite_wanted_k - 1), distances.end());
            thresholds[query] = distances[rewrite_wanted_k - 1];
        }
    }

    /** A fresh checkpoint under @c TMPDIR, removed with the fixture. */
    nvme_checkpoint_t& checkpoint() {
        checkpoints.push_back(std::make_unique<nvme_checkpoint_t>(scenario));
        return *checkpoints.back();
    }

    /** Builds rows [first, first + count) into a fresh RAM group of @p shards shards. */
    void build(ram_t& group, std::size_t first, std::size_t count, std::size_t shards = 2) {
        for (std::size_t shard = 0; shard != shards; ++shard)
            failures += expect(succeeded(group.emplace(no_memory_domain_k)));
        failures += expect(succeeded(group.reserve({count, connectivity_16_k, vectors.extent(1)}).status));
        auto const all = vectors.view();
        auto const rows = all(nk::range(first, first + count), nk::slice);
        failures += expect(succeeded(
            group.insert(std::span<key_t const> {keys.data() + first, count}, rows, builder.executor).status));
    }

    /** Saves rows [first, first + count) as a two-shard RAM group in @p layout and answers the
     *  path. */
    char const* save(std::size_t first, std::size_t count, checkpoint_layout_t layout) {
        ram_t group {kernel, forkunion_allocator_t {no_memory_domain_k}};
        return save(group, first, count, layout);
    }

    /** Builds rows [first, first + count) into @p group of @p shards shards, saves it in @p layout
     *  and answers the path. */
    char const* save(ram_t& group, std::size_t first, std::size_t count, checkpoint_layout_t layout,
                     std::size_t shards = 2) {
        build(group, first, count, shards);
        nvme_checkpoint_t& saved = checkpoint();
        saved.parts = group.shards_count();
        save_config_t config;
        config.layout = layout;
        char const* const path = layout == checkpoint_layout_t::distributed_k ? saved.manifest.data()
                                                                              : saved.monolith.data();
        failures += expect_eq(group.save(path, config, builder.executor), status_t::success_k);
        return path;
    }

    /** The share of the results @p indexes ranks for every query, through @p executor under the
     *  default walk, that lie within the query's exact k-th distance; a RAM group also takes its
     *  @p traversals, one per thread. */
    template <typename indexes_type_, typename executor_type_, typename... traversals_types_>
    f64_t recall(indexes_type_& indexes, executor_type_& executor, traversals_types_... traversals) {
        nvme_rankings<nvme_t> ranked {rewrite_queries_k, rewrite_wanted_k};
        failures += expect_eq(ranked.search(indexes, queries.view(), executor, 0, traversals...).status,
                              status_t::success_k);
        std::size_t hits = 0;
        for (std::size_t query = 0; query != rewrite_queries_k; ++query)
            for (key_t const key : ranked.keys_of(query)) {
                f32_t const exact = exact_(query, static_cast<std::size_t>(key) - 1);
                hits += exact <= thresholds[query] + 1e-5f * (1 + std::fabs(thresholds[query]));
            }
        return static_cast<f64_t>(hits) / static_cast<f64_t>(rewrite_queries_k * rewrite_wanted_k);
    }

    /** The recall of a fresh RAM build of every row into @p shards shards, so its searches walk as
     *  many graphs as a target of as many parts. */
    f64_t fresh_recall(std::size_t shards) {
        ram_t fresh {kernel, forkunion_allocator_t {no_memory_domain_k}};
        build(fresh, 0, keys.size(), shards);
        return recall(fresh, builder.executor, to_writable_span(builder.traversals));
    }

    /** Views @p path as an NVMe index, which must open. */
    nvme_t view(char const* path) {
        nvme_t viewed {forkunion_allocator_t {no_memory_domain_k}};
        failures += expect_eq(viewed.view(path), status_t::success_k);
        return viewed;
    }

    /** Every key names exactly one row of @p viewed. */
    void expect_keys(nvme_t& viewed) {
        std::vector<std::size_t> entries(keys.size());
        failures += expect_eq(viewed.count(std::span<key_t const> {keys}, std::span {entries}, driver.executor),
                              status_t::success_k);
        std::size_t miscounted = 0;
        for (std::size_t const entry : entries) miscounted += entry != 1;
        failures += expect_eq(miscounted, 0u);
    }

    /** Recalls within the margin of a fresh RAM build of as many shards as it has parts, and
     *  reaches every row from the audit's landmarks. */
    void expect_recall(nvme_t& viewed, char const* label) {
        expect_recall(viewed, label, fresh_recall(viewed.shards_count()));
    }

    /** Recalls within the margin of @p fresh, and reaches every row from the audit's landmarks. */
    void expect_recall(nvme_t& viewed, char const* label, f64_t fresh) {
        f64_t const rewritten = recall(viewed, driver.executor);
        if (rewritten + rewrite_recall_margin_k < fresh) {
            fmt::println(stderr, "  {} {}: recall {:.3f} trails {:.3f} of a fresh RAM build past the margin", scenario,
                         label, rewritten, fresh);
            ++failures;
        }
        expect_reachable(viewed);
    }

    /** At least the suite's share of @p count stored rows from @p first find themselves in
     *  @p viewed, judged by distance band, so the rows of one part cannot hide behind another's. */
    void expect_self_recall(nvme_t& viewed, std::size_t first, std::size_t count, char const* label) {
        nvme_rankings<nvme_t> ranked {count, rewrite_wanted_k};
        failures += expect_eq(
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
            ++failures;
        }
    }

    /** Every row of @p viewed is reachable from the audit's landmarks. */
    void expect_reachable(nvme_t& viewed) {
        auto const [health, health_status] = viewed.health(driver.executor);
        failures += expect_eq(health_status, status_t::success_k);
        failures += expect_eq(health.reachable_nodes, viewed.size());
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

/** Merging the two parts one RAM group saved yields one part over every row, every key once, that a
 *  RAM group loads back whole, and that recalls within the margin of a fresh RAM build. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_rewrites_merge(u64_t seed) {
    using ram_t = typename rewrite_fixture<nvme_type_>::ram_t;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_merge", kernel_, rewrite_rows_k, dimensions, seed};
    nvme_type_ source = fixture.view(fixture.save(0, rewrite_rows_k, checkpoint_layout_t::distributed_k));
    if (fixture.failures) return fixture.failures;
    std::vector<byte_t> workspace = rewrite_workspace(4u << 20);
    nvme_checkpoint_t& target = fixture.checkpoint();
    std::array<nvme_type_ const*, 1> const sources {&source};
    auto const [merged, status] = merge(std::span<nvme_type_ const* const>(sources), target.manifest.data(),
                                        fixture.driver.executor, std::span {workspace});
    fixture.failures += expect_eq(status, status_t::success_k);
    target.parts = merged.parts;
    fixture.failures += expect_eq(merged.rows, rewrite_rows_k);
    fixture.failures += expect_eq(merged.parts, 1u);
    fixture.failures += expect_eq(merged.cut_edges, 0u);
    fixture.failures += expect(merged.graph.visited_nodes > 0);
    fixture.failures += expect(merged.value_reads.count > 0 && merged.written_bytes > 0);
    if (failed(status)) return fixture.failures;

    nvme_type_ viewed = fixture.view(target.manifest.data());
    fixture.failures += expect_eq(viewed.size(), rewrite_rows_k);
    fixture.failures += expect_eq(viewed.shards_count(), 1u);
    fixture.expect_keys(viewed);
    fixture.expect_recall(viewed, "of two parts");

    ram_t loaded {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    fixture.failures += expect(succeeded(place_shards(loaded)));
    fixture.failures += expect_eq(loaded.load(target.manifest.data(), load_config_t {}, fixture.builder.executor),
                                  status_t::success_k);
    fixture.failures += expect_eq(loaded.size(), rewrite_rows_k);
    return fixture.failures;
}

/** Three parts from two sources, a manifest of two and a monolith, merge step by step, a search of
 *  a source running between every two steps on the same executor, and the merge recalls within the
 *  margin. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_rewrites_merge_steps(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    std::size_t const first_rows = rewrite_rows_k * 2 / 3;
    rewrite_fixture<nvme_type_> fixture {"rewrite_merge_steps", kernel_, rewrite_rows_k, dimensions, seed};
    nvme_type_ first = fixture.view(fixture.save(0, first_rows, checkpoint_layout_t::distributed_k));
    nvme_type_ second = fixture.view(
        fixture.save(first_rows, rewrite_rows_k - first_rows, checkpoint_layout_t::monolith_k));
    if (fixture.failures) return fixture.failures;
    fixture.failures += expect_eq(first.shards_count() + second.shards_count(), 3u);
    std::vector<byte_t> workspace = rewrite_workspace(4u << 20);
    nvme_checkpoint_t& target = fixture.checkpoint();
    std::array<nvme_type_ const*, 2> const sources {&first, &second};
    auto [plan, status] = plan_merge(std::span<nvme_type_ const* const>(sources), target.manifest.data(),
                                     std::span {workspace});
    fixture.failures += expect_eq(status, status_t::success_k);
    if (failed(status)) return fixture.failures;

    nvme_rankings<nvme_type_> interleaved {rewrite_queries_k, rewrite_wanted_k};
    std::size_t phases = 0, steps = 0;
    do {
        ++phases;
        for (auto const step : plan.steps()) {
            fixture.failures += expect_eq(plan.run(step, fixture.driver.executor), status_t::success_k);
            fixture.failures += expect_eq(
                interleaved.search(first, fixture.queries.view(), fixture.driver.executor, 0).status,
                status_t::success_k);
            ++steps;
        }
    } while (plan.advance());
    fixture.failures += expect(plan.phase() == nvme_rewrite_phase_t::finish_k);
    fixture.failures += expect(phases >= 4 && steps >= phases);
    auto const [merged, finish_status] = plan.finish();
    fixture.failures += expect_eq(finish_status, status_t::success_k);
    target.parts = merged.parts;
    fixture.failures += expect_eq(merged.rows, rewrite_rows_k);
    if (failed(finish_status)) return fixture.failures;

    nvme_type_ viewed = fixture.view(target.manifest.data());
    fixture.expect_keys(viewed);
    fixture.expect_recall(viewed, "of three parts");
    // The second source's monolith is the largest part and stays resident;
    // the first source's two parts are appended.
    fixture.expect_self_recall(viewed, 0, rewrite_queries_k, "appended rows");
    fixture.expect_self_recall(viewed, first_rows, rewrite_queries_k, "resident rows");
    return fixture.failures;
}

/** A split by rows of one part into thirds: the parts and rows add up, the cut edges are counted,
 *  and each third, repaired by the RAM refine the workspace holds, recalls within the margin. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_rewrites_split_rows(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_split_rows", kernel_, rewrite_rows_k, dimensions, seed};
    nvme_type_ source = fixture.view(fixture.save(0, rewrite_rows_k, checkpoint_layout_t::monolith_k));
    if (fixture.failures) return fixture.failures;
    std::vector<byte_t> workspace = rewrite_workspace(8u << 20);
    nvme_checkpoint_t& target = fixture.checkpoint();
    // The monolith of two RAM shards has no edge across its halves, so thirds cut inside them.
    auto const [cut, status] = split(source, split_by_rows_t {divide_round_up<std::size_t>(rewrite_rows_k, 3)},
                                     target.manifest.data(), fixture.driver.executor, std::span {workspace});
    fixture.failures += expect_eq(status, status_t::success_k);
    target.parts = cut.parts;
    fixture.failures += expect_eq(cut.parts, 3u);
    fixture.failures += expect_eq(cut.rows, rewrite_rows_k);
    fixture.failures += expect(cut.cut_edges > 0);
    fixture.failures += expect(cut.graph.visited_nodes >= rewrite_rows_k);
    if (failed(status)) return fixture.failures;

    nvme_type_ viewed = fixture.view(target.manifest.data());
    fixture.failures += expect_eq(viewed.shards_count(), 3u);
    fixture.failures += expect_eq(viewed.size(), rewrite_rows_k);
    fixture.expect_keys(viewed);
    fixture.expect_recall(viewed, "by rows, repaired in RAM");
    return fixture.failures;
}

/** A split by keys of two parts into three key ranges, in a workspace too small for any RAM index,
 *  so every part is repaired by walks: the parts and rows add up, each part holds only its range,
 *  the cut edges are counted, and the parts recall within the margin. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_rewrites_split_keys(u64_t seed) {
    using key_t = typename nvme_type_::key_t;
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_split_keys", kernel_, rewrite_rows_k, dimensions, seed};
    nvme_type_ source = fixture.view(fixture.save(0, rewrite_rows_k, checkpoint_layout_t::distributed_k));
    if (fixture.failures) return fixture.failures;
    std::vector<byte_t> workspace = rewrite_workspace(512u << 10);
    nvme_checkpoint_t& target = fixture.checkpoint();
    std::array<key_t, 2> const fences {static_cast<key_t>(1 + rewrite_rows_k / 3),
                                       static_cast<key_t>(1 + rewrite_rows_k * 2 / 3)};
    auto const [cut, status] = split(source, split_by_keys<key_t> {std::span<key_t const>(fences)},
                                     target.manifest.data(), fixture.driver.executor, std::span {workspace});
    fixture.failures += expect_eq(status, status_t::success_k);
    target.parts = cut.parts;
    fixture.failures += expect_eq(cut.parts, 3u);
    fixture.failures += expect_eq(cut.rows, rewrite_rows_k);
    fixture.failures += expect(cut.cut_edges > 0);
    if (failed(status)) return fixture.failures;

    nvme_type_ viewed = fixture.view(target.manifest.data());
    fixture.failures += expect_eq(viewed.shards_count(), 3u);
    fixture.expect_keys(viewed);
    // In placement order the parts follow the fences, so the ranges of the keys never fall back.
    typename nvme_type_::cursor_t cursor {viewed};
    std::vector<key_t> read(rewrite_rows_k);
    auto const [exported, read_status] = cursor.read(viewed, std::span {read}, {}, fixture.driver.executor);
    fixture.failures += expect_eq(read_status, status_t::success_k);
    fixture.failures += expect_eq(exported.count, rewrite_rows_k);
    auto range_of = [&](key_t key) noexcept {
        return std::upper_bound(fences.begin(), fences.end(), key) - fences.begin();
    };
    std::size_t fallbacks = 0;
    for (std::size_t row = 1; row < exported.count; ++row) fallbacks += range_of(read[row]) < range_of(read[row - 1]);
    fixture.failures += expect_eq(fallbacks, 0u);
    fixture.expect_recall(viewed, "by keys, repaired by walks");
    return fixture.failures;
}

/** A part thinned to one neighbor per row recalls poorly; refining all its rows by walks, in a
 *  workspace too small for a RAM index, recalls better and links back every row nothing links to.
 *  The thinning leaves mutual nearest pairs linking only each other, which the heal leaves for
 *  @c health to report, so not every row is reachable. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_rewrites_refine(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_refine", kernel_, rewrite_rows_k, dimensions, seed};
    char const* const saved = fixture.save(0, rewrite_rows_k, checkpoint_layout_t::monolith_k);
    if (fixture.failures) return fixture.failures;

    // The thinned copy keeps every row's first neighbor and nothing else.
    nvme_checkpoint_t& thinned = fixture.checkpoint();
    {
        auto [mapping, map_status] = mapped_file_t::read(saved);
        fixture.failures += expect_eq(map_status, status_t::success_k);
        auto const [header, header_status] = safegraphs_read_header(mapping.view());
        fixture.failures += expect_eq(header_status, status_t::success_k);
        if (fixture.failures) return fixture.failures;
        std::vector<byte_t> bytes(mapping.view().begin(), mapping.view().end());
        std::size_t const connectivity = regular_width(header.neighbors.shape).value_or(0);
        auto* const neighbors = reinterpret_cast<node_id_t*>(bytes.data() + header.data_offset +
                                                             header.neighbors.data.first);
        for (std::size_t row = 0; row != rewrite_rows_k; ++row)
            for (std::size_t slot = 1; slot != connectivity; ++slot)
                neighbors[row * connectivity + slot] = node_id_unknown_k;
        auto [file, open_status] = seekable_file_t::open(thinned.monolith.data(), file_mode_t::write_k);
        fixture.failures += expect_eq(open_status, status_t::success_k);
        fixture.failures += expect_eq(file.write_at(0, bytes_view_t {bytes}), status_t::success_k);
    }
    nvme_type_ source = fixture.view(thinned.monolith.data());
    if (fixture.failures) return fixture.failures;
    f64_t const before = fixture.recall(source, fixture.driver.executor);

    std::vector<byte_t> workspace = rewrite_workspace(64u << 10);
    nvme_checkpoint_t& target = fixture.checkpoint();
    auto const [refined, status] = refine(source, iota_view<std::size_t> {0, rewrite_rows_k}, target.manifest.data(),
                                          fixture.driver.executor, std::span {workspace});
    fixture.failures += expect_eq(status, status_t::success_k);
    target.parts = refined.parts;
    fixture.failures += expect_eq(refined.rows, rewrite_rows_k);
    fixture.failures += expect_eq(refined.graph.visited_nodes, rewrite_rows_k);
    fixture.failures += expect(refined.graph.changed_nodes > 0);
    fixture.failures += expect_eq(refined.graph.islands_reconnected, refined.graph.islands_found);
    if (failed(status)) return fixture.failures;

    nvme_type_ viewed = fixture.view(target.manifest.data());
    f64_t const after = fixture.recall(viewed, fixture.driver.executor);
    if (!(after > before)) {
        fmt::println(stderr, "  {}: recall {:.3f} after the refine, {:.3f} before it", fixture.scenario, after, before);
        ++fixture.failures;
    }
    fixture.expect_keys(viewed);
    return fixture.failures;
}

/** Building rows four times what the workspace holds as one RAM index runs at least four blocks,
 *  links them across, and recalls within the margin of a RAM build of every row.
 *
 *  A RAM index takes its chunk directories and its first chunk before it holds a row, 1.8 MB with
 *  128-byte cache lines, so the workspace is measured on this host: that fixed cost, plus a quarter
 *  of the rows' staging and RAM rows. A literal four times the rows' bytes would take tens of
 *  thousands of rows to outgrow the fixed cost. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_rewrites_build(u64_t seed) {
    using scalar_t = typename nvme_type_::scalar_t;
    using ram_t = typename rewrite_fixture<nvme_type_>::ram_t;
    using block_index_t = ram_index<typename nvme_type_::quads_kit_t, typename nvme_type_::tiles_kit_t,
                                    typename nvme_type_::elementwise_kit_t, typename nvme_type_::rmscast_kit_t,
                                    typename nvme_type_::visits_kit_t, fu::standard_yield_t, fu::standard_atomic_ref,
                                    dummy_cache_hints_t, nvme_workspace_allocator<byte_t>,
                                    nvme_workspace_allocator<byte_t>, arena_allocator<byte_t, forkunion_allocator_t>>;
    constexpr std::size_t rows_count = 4 * rewrite_rows_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_build", kernel_, rows_count, dimensions, seed};
    // One RAM shard over every row is both the rows the build reads and the graph it is judged
    // against.
    ram_t group {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    char const* const saved = fixture.save(group, 0, rows_count, checkpoint_layout_t::distributed_k, 1);
    if (fixture.failures) return fixture.failures;

    std::vector<byte_t> probed(16u << 20);
    bytes_span_t left {probed};
    {
        std::unique_ptr<block_index_t> const probe {new (std::nothrow) block_index_t(
            kernel_, nvme_workspace_allocator<byte_t>(left), nvme_workspace_allocator<byte_t>(left))};
        fixture.failures += expect(probe && succeeded(probe->reserve({1, connectivity_16_k, dimensions}).status));
    }
    std::size_t const fixed = probed.size() - left.size();
    std::size_t const value_bytes = dense_rows_bytes(fixture.vectors.view(), 1), neighbors_bytes = 16 * sizeof(u32_t);
    std::size_t const row_bytes = 2 * value_bytes + 2 * neighbors_bytes + 2 * sizeof(u32_t) + 16;
    std::vector<byte_t> workspace = rewrite_workspace(fixed + rows_count / 4 * row_bytes);
    nvme_checkpoint_t& target = fixture.checkpoint();
    std::array<std::string_view, 1> const sources {saved};
    auto [plan, status] = plan_build<nvme_type_>(std::span<std::string_view const>(sources), target.manifest.data(),
                                                 std::span {workspace});
    fixture.failures += expect_eq(status, status_t::success_k);
    if (failed(status)) return fixture.failures;
    std::size_t blocks = 0;
    do {
        if (plan.phase() == nvme_rewrite_phase_t::build_k) blocks += plan.steps().size();
        for (auto const step : plan.steps())
            fixture.failures += expect_eq(plan.run(step, fixture.driver.executor), status_t::success_k);
    } while (plan.advance());
    auto const [built, finish_status] = plan.finish();
    fixture.failures += expect_eq(finish_status, status_t::success_k);
    target.parts = built.parts;
    fixture.failures += expect(blocks >= 4);
    fixture.failures += expect_eq(built.rows, rows_count);
    fixture.failures += expect_eq(built.parts, 1u);
    if (failed(finish_status)) return fixture.failures;

    fixture.failures += expect(built.graph.visited_nodes >= rows_count);
    nvme_type_ viewed = fixture.view(target.manifest.data());
    fixture.expect_keys(viewed);
    fixture.expect_recall(
        viewed, "in blocks",
        fixture.recall(group, fixture.builder.executor, to_writable_span(fixture.builder.traversals)));
    fixture.expect_self_recall(viewed, 0, rewrite_queries_k, "first block");
    fixture.expect_self_recall(viewed, rows_count - rewrite_queries_k, rewrite_queries_k, "last block");
    return fixture.failures;
}

/** A merge dropped after its first re-select step leaves no manifest, no part and no spill behind,
 *  so nothing views. */
template <typename nvme_type_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_rewrites_dropped(u64_t seed) {
    std::size_t const dimensions = dense_dimensions_for<typename nvme_type_::scalar_t>(dense_values_k);
    rewrite_fixture<nvme_type_> fixture {"rewrite_dropped", kernel_, rewrite_rows_k, dimensions, seed};
    nvme_type_ source = fixture.view(fixture.save(0, rewrite_rows_k, checkpoint_layout_t::distributed_k));
    if (fixture.failures) return fixture.failures;
    std::vector<byte_t> workspace = rewrite_workspace(512u << 10);
    nvme_checkpoint_t& target = fixture.checkpoint();
    target.parts = 1;
    {
        std::array<nvme_type_ const*, 1> const sources {&source};
        auto [plan, status] = plan_merge(std::span<nvme_type_ const* const>(sources), target.manifest.data(),
                                         std::span {workspace});
        fixture.failures += expect_eq(status, status_t::success_k);
        while (succeeded(status) && plan.phase() != nvme_rewrite_phase_t::reselect_k) {
            for (auto const step : plan.steps())
                fixture.failures += expect_eq(plan.run(step, fixture.driver.executor), status_t::success_k);
            if (!plan.advance()) break;
        }
        fixture.failures += expect(plan.phase() == nvme_rewrite_phase_t::reselect_k);
        auto const steps = plan.steps();
        fixture.failures += expect_eq(plan.run(steps[0], fixture.driver.executor), status_t::success_k);
        fixture.failures += expect_eq(plan.finish().status, status_t::cancelled_k);
    }
    nvme_type_ viewed {forkunion_allocator_t {no_memory_domain_k}};
    fixture.failures += expect(failed(viewed.view(target.manifest.data())));
    growing_buffer<char> part_path {global_allocator};
    std::string_view const part = target.part_path(part_path, 0);
    fixture.failures += expect(failed(seekable_file_t::open(part, file_mode_t::read_k).status));
    char spill[256] = {};
    std::string_view const stem = safegraphs_stem(target.manifest.data());
    std::snprintf(spill, sizeof(spill), "%.*s-spill-00000.bin", static_cast<int>(stem.size()), stem.data());
    fixture.failures += expect(failed(seekable_file_t::open(spill, file_mode_t::read_k).status));
    return fixture.failures;
}

/** A @c dense::nvme_indexes the shared scenarios build like any other sharded index: rows land in a
 *  two-shard @c dense::ram_indexes, and every insertion saves its manifest and views the parts. The
 *  one-key lookups the NVMe view lacks run as batches of one on an executor of its own. */
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
    using search_result_t = typename viewed_t::search_result_t;
    using executor_t = typename viewed_t::executor_t;
    using index_t = typename viewed_t::index_t;
    using ram_t = typename nvme_reference<viewed_t>::ram_t;
    using reserve_capacity_t = typename ram_t::reserve_capacity_t;

    dense_indexes_executor<ram_t> builder;
    dense_indexes_executor<viewed_t> reader;
    ram_t built;
    viewed_t viewed;
    nvme_checkpoint_t checkpoint {"nvme_staged"};

    /** The RAM group walks by @p kernel; the view reads it back from the parts' headers. */
    nvme_staged_indexes(nk_kernel_kind_t kernel, forkunion_allocator_t allocator) noexcept
        : built(kernel, allocator), viewed(allocator) {}

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
        status_t const gotten = viewed.get({&key, 1}, vectors(nk::range(std::size_t {0}, rows), nk::slice), rows,
                                           {&entries, 1}, executor);
        return succeeded(gotten) ? entries : 0;
    }
    status_t count(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) noexcept {
        return viewed.count(keys, entries, executor);
    }
    status_t contains(std::span<key_t const> keys, std::span<std::size_t> entries, executor_t& executor) noexcept {
        return viewed.contains(keys, entries, executor);
    }
    status_t get(std::span<key_t const> keys, vectors_span_t vectors, std::size_t wanted,
                 std::span<std::size_t> entries, executor_t& executor) noexcept {
        return viewed.get(keys, vectors, wanted, entries, executor);
    }

    expected<search_result_t> search(vectors_view_t queries, executor_t& executor, search_config_t config,
                                     nk::matrix_span<key_t> keys, nk::matrix_span<distance_t> distances,
                                     std::span<std::size_t> counts) noexcept {
        return viewed.search(queries, executor, config, keys, distances, counts);
    }

    template <typename predicate_type_>
    expected<search_result_t> search_if(vectors_view_t queries, executor_t& executor, search_config_t config,
                                        nk::matrix_span<key_t> keys, nk::matrix_span<distance_t> distances,
                                        std::span<std::size_t> counts, predicate_type_&& predicate) noexcept {
        return viewed.search_if(queries, executor, config, keys, distances, counts, predicate);
    }

    expected<search_result_t> search(vectors_view_t queries, indexer_heads<scalar_t> const& heads, executor_t& executor,
                                     search_config_t config, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<distance_t> distances, std::span<std::size_t> counts) noexcept {
        return viewed.search(queries, heads, executor, config, keys, distances, counts);
    }

    expected<search_result_t> multiply(vectors_view_t left, executor_t& executor, search_config_t config,
                                       nk::matrix_span<key_t> keys, nk::matrix_span<dot_result_t> products,
                                       std::span<std::size_t> counts) noexcept {
        return viewed.multiply(left, executor, config, keys, products, counts);
    }

    expected<search_result_t> maxsim(vectors_view_t queries, std::span<std::size_t const> query_offsets,
                                     executor_t& executor, search_config_t config, nk::matrix_span<key_t> keys,
                                     nk::matrix_span<f64_t> scores, std::span<std::size_t> counts) noexcept {
        return viewed.maxsim(queries, query_offsets, executor, config, keys, scores, counts);
    }

    template <typename value_lookup_type_>
    expected<search_result_t> attend(vectors_view_t queries, std::size_t head_count, value_lookup_type_&& value_lookup,
                                     f32_t score_scale, nk::matrix_span<f32_t> outputs,
                                     std::span<softmax_state_t> carry, executor_t& executor,
                                     search_config_t config = {}) noexcept {
        return viewed.attend(queries, head_count, value_lookup, score_scale, outputs, carry, executor, config);
    }

  private:
    status_t save_and_view_() noexcept {
        save_config_t config;
        config.layout = checkpoint_layout_t::distributed_k;
        checkpoint.parts = built.shards_count();
        if (status_t const saved = built.save(checkpoint.manifest.data(), config, builder.executor); failed(saved))
            return saved;
        return viewed.view(checkpoint.manifest.data());
    }
};

/** Whether a sharded index is the NVMe view the shared scenarios build through its RAM group. */
template <typename indexes_type_>
concept nvme_staged = nvme_collection<indexes_type_> && requires { typename indexes_type_::ram_t; };

/** Composes one engine's NVMe suite over one (dtype, kernel) cell, skipping it where the host
 *  refuses the engine. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_configuration(char const* name, u64_t seed) {
    using indexes_t = nvme_indexes_for<u32_t, dtype_, nk_cap_serial_k, engine_type_>;
    using staged_t = nvme_staged_indexes<indexes_t>;
    {
        dense_indexes_executor<indexes_t> probe;
        if (failed(probe.status)) return report(name, "engine refused by this host");
    }
    std::size_t failures_count = 0;
    failures_count += test_nvme_indexes_search<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_filtered<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_key_scans<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_cursor<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_health<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_telemetry<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_executors<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_failed_read<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_failed_submit<engine_type_, dtype_, kernel_>(seed);
    failures_count += test_nvme_indexes_pipelined<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_relocated<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_byte_base<indexes_t, kernel_>(seed);
    failures_count += test_nvme_index_search<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_straddling<engine_type_, dtype_, kernel_, 512>(seed);
    failures_count += test_nvme_indexes_straddling<engine_type_, dtype_, kernel_, 4096>(seed);
    failures_count += test_dense_indexes_scan<staged_t, kernel_>(seed);
    failures_count += test_dense_indexes_products<staged_t, kernel_>(seed);
    failures_count += test_dense_indexes_indexer_search<staged_t, kernel_>(seed);
    failures_count += test_dense_indexes_maxsim<staged_t, kernel_>(seed);
    failures_count += test_dense_indexes_attend<staged_t, kernel_>(seed);
    failures_count += test_dense_indexes_duplicate_keys<staged_t, kernel_>(seed);
    return report(name, failures_count);
}

/** The exact scan alone over parts a matrix engine streams under @p caps_, skipped where the build
 *  lacks the kit, the host its capabilities, or the engine. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_, nk_capability_t caps_>
std::size_t test_nvme_scan_configuration(char const* name, u64_t seed) {
    if constexpr (!dense_cell_testable<dtype_, kernel_, caps_>) return report(name, "kit not compiled for this target");
    else {
        using indexes_t = nvme_indexes_for<u32_t, dtype_, caps_, engine_type_>;
        if (!dense_cell_admitted<caps_, fu::standard_atomic_ref>())
            return report(name, "capabilities missing at runtime");
        {
            dense_indexes_executor<indexes_t> probe;
            if (failed(probe.status)) return report(name, "engine refused by this host");
        }
        return report(name, test_dense_indexes_scan<nvme_staged_indexes<indexes_t>, kernel_>(seed));
    }
}

/** The cases that read keys, once more over the C layer's 16-byte @c uuid_t keys, skipped where the
 *  host refuses the engine. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_uuid_configuration(char const* name, u64_t seed) {
    using indexes_t = nvme_indexes_for<uuid_t, dtype_, nk_cap_serial_k, engine_type_>;
    {
        dense_indexes_executor<indexes_t> probe;
        if (failed(probe.status)) return report(name, "engine refused by this host");
    }
    std::size_t failures_count = 0;
    failures_count += test_nvme_indexes_search<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_filtered<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_key_scans<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_cursor<indexes_t, kernel_>(seed);
    failures_count += test_nvme_indexes_telemetry<indexes_t, kernel_>(seed);
    return report(name, failures_count);
}

/** A monolith keyed @c u32, as the benches write one, viewed by a @c uuid collection: the cursor,
 *  the key scans and a ranking's resolved keys read every key zero-extended, big-endian as the
 *  bindings render integer identities, at the distances the @c u32 view of the file ranks. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_widened_keys(char const* name, u64_t seed) {
    using narrow_t = nvme_indexes_for<u32_t, dtype_, nk_cap_serial_k, engine_type_>;
    using wide_t = nvme_indexes_for<uuid_t, dtype_, nk_cap_serial_k, engine_type_>;
    using scalar_t = typename wide_t::scalar_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 8, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    auto const number_of = [](uuid_t const& key) noexcept {
        u64_t number = 0;
        for (std::size_t byte = sizeof(key.bytes) - sizeof(number); byte != sizeof(key.bytes); ++byte)
            number = number << 8 | std::to_integer<u64_t>(key.bytes[byte]);
        return number;
    };

    nvme_reference<narrow_t> reference {"nvme_widened", kernel_, rows_count, dimensions, seed};
    narrow_t narrow {forkunion_allocator_t {no_memory_domain_k}};
    reference.view(narrow, checkpoint_layout_t::monolith_k);
    wide_t wide {forkunion_allocator_t {no_memory_domain_k}};
    std::size_t failures = reference.failures;
    failures += expect_eq(wide.view(reference.checkpoint.monolith.data()), status_t::success_k);
    dense_indexes_executor<narrow_t> narrow_driver;
    dense_indexes_executor<wide_t> wide_driver;
    if (failed(narrow_driver.status) || failed(wide_driver.status) || failures)
        return report(name, failures + expect(false));
    failures += expect_eq(wide.size(), rows_count);

    std::vector<uuid_t> placed_keys(rows_count);
    auto placed_rows = make_vectors<scalar_t>(rows_count, dimensions, 0);
    typename wide_t::cursor_t cursor {wide};
    failures += expect_eq(cursor.read(wide, std::span {placed_keys}, placed_rows.span(), wide_driver.executor).status,
                          status_t::success_k);
    std::vector<bool> seen(rows_count);
    for (uuid_t const& key : placed_keys) {
        u64_t const number = number_of(key);
        bool const fresh = key == uuid_of(number) && number >= 1 && number <= rows_count && !seen[number - 1];
        failures += expect(fresh);
        if (fresh) seen[number - 1] = true;
    }

    std::array<uuid_t, 2> const probes {uuid_of(1), uuid_of(rows_count + 7)};
    std::array<std::size_t, probes.size()> counted {};
    failures += expect_eq(wide.count(std::span<uuid_t const> {probes}, std::span {counted}, wide_driver.executor),
                          status_t::success_k);
    failures += expect_eq(counted[0], 1u);
    failures += expect_eq(counted[1], 0u);

    // A walk as wide as the file ranks every row, so only ties may swap keys between the two views.
    auto const queries = reference.leading_rows(queries_count);
    nvme_rankings<narrow_t> narrow_rankings {queries_count, wanted};
    nvme_rankings<wide_t> wide_rankings {queries_count, wanted};
    failures += expect(succeeded(narrow_rankings.search(narrow, queries, narrow_driver.executor, rows_count).status));
    failures += expect(succeeded(wide_rankings.search(wide, queries, wide_driver.executor, rows_count).status));
    failures += expect(wide_rankings.counts == narrow_rankings.counts);
    for (std::size_t slot = 0; slot != wide_rankings.keys.size(); ++slot) {
        uuid_t const& key = wide_rankings.keys[slot];
        failures += expect_eq(wide_rankings.distances[slot], narrow_rankings.distances[slot]);
        failures += expect(key == uuid_of(number_of(key)) && number_of(key) <= rows_count);
    }
    return report(name, failures);
}

/** Composes one engine's rewrite suite over one (dtype, kernel) cell, skipping it where the host
 *  refuses the engine. */
template <typename engine_type_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_rewrites_configuration(char const* name, u64_t seed) {
    using indexes_t = nvme_indexes_for<u32_t, dtype_, nk_cap_serial_k, engine_type_>;
    {
        dense_indexes_executor<indexes_t> probe;
        if (failed(probe.status)) return report(name, "engine refused by this host");
    }
    std::size_t failures_count = 0;
    failures_count += test_nvme_rewrites_merge<indexes_t, kernel_>(seed);
    failures_count += test_nvme_rewrites_merge_steps<indexes_t, kernel_>(seed);
    failures_count += test_nvme_rewrites_split_rows<indexes_t, kernel_>(seed);
    failures_count += test_nvme_rewrites_split_keys<indexes_t, kernel_>(seed);
    failures_count += test_nvme_rewrites_refine<indexes_t, kernel_>(seed);
    failures_count += test_nvme_rewrites_build<indexes_t, kernel_>(seed);
    failures_count += test_nvme_rewrites_dropped<indexes_t, kernel_>(seed);
    return report(name, failures_count);
}

#if USEARCH_WITH_LINUX_SPDK

/**
 *  @brief The SPDK engine over a monolith copied onto an NVMe namespace, judged against the
 *      blocking engine over the monolith's file, every verb alike.
 *
 *  Runs in two passes around a rig: with no @p nvme_paths it saves the monolith as
 *  `usearch-nvme-spdk.safegraphs` under @c TMPDIR for the rig to copy onto a namespace, and with
 *  @c USEARCH_NVME_PATHS naming that namespace's transport ID it views the copy directly and
 *  through a relocation of the file's view onto it.
 */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
std::size_t test_nvme_spdk_configuration(char const* name, u64_t seed, std::span<std::string_view const> nvme_paths) {
    using files_t = nvme_indexes_for<u32_t, dtype_, nk_cap_serial_k, blocking_files_engine<>>;
    using spdk_t = nvme_indexes_for<u32_t, dtype_, nk_cap_serial_k, linux_spdk_engine<>>;
    using key_t = typename files_t::key_t;
    using scalar_t = typename files_t::scalar_t;
    constexpr std::size_t rows_count = 2 * dense_thread_nodes_k, queries_count = 8, wanted = 16;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::string const part = temporary_path("usearch-nvme-spdk.safegraphs");
    if (nvme_paths.empty()) {
        nvme_reference<files_t> reference {"nvme_spdk", kernel_, rows_count, dimensions, seed};
        save_config_t config;
        config.layout = checkpoint_layout_t::monolith_k;
        std::size_t const failures = reference.failures +
                                     expect(succeeded(reference.built.save(part, config, reference.builder.executor)));
        return failures ? report(name, failures) : report(name, "saved the part for the rig to copy");
    }

    std::size_t failures = 0;
    files_t files {forkunion_allocator_t {no_memory_domain_k}};
    spdk_t direct {forkunion_allocator_t {no_memory_domain_k}}, relocated {forkunion_allocator_t {no_memory_domain_k}};
    failures += expect_eq(files.view(part), status_t::success_k);
    failures += expect_eq(direct.view(nvme_paths.front()), status_t::success_k);
    failures += expect_eq(relocated.view(part, nvme_paths.first(1)), status_t::success_k);
    failures += expect_eq(direct.size(), rows_count);
    failures += expect_eq(relocated.size(), rows_count);
    dense_indexes_executor<files_t> files_driver;
    dense_indexes_executor<spdk_t> spdk_driver;
    failures += expect_eq(files_driver.status, status_t::success_k);
    failures += expect_eq(spdk_driver.status, status_t::success_k);
    if (failures) return report(name, failures);

    // Under a walk as wide as the part both rank every row, so only ties may swap keys.
    auto const queries = make_vectors<scalar_t>(queries_count, dimensions, stream_key(seed, "nvme_spdk_queries"));
    nvme_rankings<files_t> expected_rankings {queries_count, wanted};
    failures += expect(
        succeeded(expected_rankings.search(files, queries.view(), files_driver.executor, rows_count).status));
    for (spdk_t* const viewed : {&direct, &relocated}) {
        nvme_rankings<files_t> found {queries_count, wanted};
        failures += expect(succeeded(found.search(*viewed, queries.view(), spdk_driver.executor, rows_count).status));
        failures += expect(found.counts == expected_rankings.counts);
        for (std::size_t slot = 0; slot != found.distances.size(); ++slot)
            failures += expect_eq(found.distances[slot], expected_rankings.distances[slot]);
    }

    std::array<key_t, 4> const probes {1, 2, static_cast<key_t>(rows_count), static_cast<key_t>(rows_count + 7)};
    std::array<std::size_t, probes.size()> files_gotten {}, spdk_gotten {};
    auto files_rows = make_vectors<scalar_t>(probes.size(), dimensions, 0);
    auto spdk_rows = make_vectors<scalar_t>(probes.size(), dimensions, 0);
    failures += expect_eq(files.get(probes, files_rows.span(), 1, std::span {files_gotten}, files_driver.executor),
                          status_t::success_k);
    failures += expect_eq(direct.get(probes, spdk_rows.span(), 1, std::span {spdk_gotten}, spdk_driver.executor),
                          status_t::success_k);
    failures += expect(files_gotten == spdk_gotten);
    std::size_t const row_bytes = dense_rows_bytes(files_rows.view(), 1);
    auto same_rows = [&](auto const& first, auto const& second, std::size_t rows) {
        std::size_t differing = 0;
        for (std::size_t row = 0; row != rows; ++row)
            differing += std::memcmp(first.view().row(row).as_vector().byte_data(),
                                     second.view().row(row).as_vector().byte_data(), row_bytes) != 0;
        return differing == 0;
    };
    failures += expect(same_rows(files_rows, spdk_rows, probes.size()));

    std::vector<key_t> files_keys(rows_count), spdk_keys(rows_count);
    auto files_placed = make_vectors<scalar_t>(rows_count, dimensions, 0);
    auto spdk_placed = make_vectors<scalar_t>(rows_count, dimensions, 0);
    typename files_t::cursor_t files_cursor {files};
    typename spdk_t::cursor_t spdk_cursor {direct};
    failures += expect_eq(
        files_cursor.read(files, std::span {files_keys}, files_placed.span(), files_driver.executor).status,
        status_t::success_k);
    failures += expect_eq(
        spdk_cursor.read(direct, std::span {spdk_keys}, spdk_placed.span(), spdk_driver.executor).status,
        status_t::success_k);
    failures += expect(files_keys == spdk_keys);
    failures += expect(same_rows(files_placed, spdk_placed, rows_count));

    auto const [files_health, files_status] = files.health(files_driver.executor);
    auto const [spdk_health, spdk_status] = direct.health(spdk_driver.executor);
    failures += expect_eq(spdk_status, files_status);
    failures += expect_eq(spdk_health.reachable_nodes, files_health.reachable_nodes);
    return report(name, failures);
}

#endif

/** The NVMe sharded-index suite entry its translation unit defines and @c dense.cpp composes
 *  explicitly; returns the suite's failure count. */
std::size_t check_nvme_indexes(settings_t const& settings);

/** The rewrite suite entry, defined and composed beside @ref check_nvme_indexes; returns the
 *  suite's failure count. */
std::size_t check_nvme_rewrites(settings_t const& settings);

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
            us::utilities::linux_spdk_driver_t::config_t environment;
            environment.memory = us::utilities::linux_spdk_memory_t::plain_pages_k;
            config.spdk_driver = us::utilities::linux_spdk_driver_t::open(environment).value;
        }
#endif
        auto [spawn_executor, spawn_status] = executor_t::spawn(topology, 2, config);
        status = spawn_status;
        executor = std::move(spawn_executor);
    }
    ~dense_indexes_executor() noexcept {
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

/** The staged NVMe form: one-row insertions land in the RAM group, through a context over its own
 *  arena. */
template <us::test::dense::nvme_staged indexes_type_>
struct us::test::dense_indexes_row_driver<indexes_type_> {
    us::test::dense_arena_context<typename indexes_type_::ram_t> arena;
    us::utilities::status_t status;

    dense_indexes_row_driver(indexes_type_& indexes, us::test::dense_indexes_executor<indexes_type_>&) noexcept
        : arena(indexes.built), status(arena.status) {}
    typename indexes_type_::ram_t::traversal_context_t& handle() noexcept { return arena.context; }
};
