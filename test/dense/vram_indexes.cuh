/**
 *  @file test/dense/vram_indexes.cuh
 *  @author Ash Vardanian
 *  @date September 1, 2026
 *  @brief The shared-memory cell of the sharded VRAM index - one shard per device - instantiated
 *      per vendor, as @c vram_index.cuh is.
 *
 *  The scenarios live in @c dense.cuh, templated over the sharded-index type; the VRAM forms of
 *  the placement and executor helpers live here, over whichever executor the cell's index names.
 */
#pragma once

#include "usearch/dense/vram_amd_compaction.hpp"    // `plan_compact` of AMD shards into RAM
#include "usearch/dense/vram_nvidia_compaction.hpp" // `plan_compact` of NVIDIA shards into RAM

#include "dense.cuh"
#include "nvme_indexes.hpp" // `nvme_reference`, `nvme_checkpoint_t` - the round trip through storage
#include "vram_index.cuh"   // the vendor's `vram_cell`

namespace unum::usearch::test::dense {

/** Copies @p bytes a @p vendor_ device wrote at @p source into host @p destination; false when the
 *  copy failed. Apple's unified memory is the host's own. */
template <gpu_vendor_t vendor_>
bool dense_download(void* destination, void const* source, std::size_t bytes) noexcept {
    if constexpr (vendor_ == gpu_vendor_t::nvidia_k)
        return cuda_copy(test::gpu_drivers().cuda, destination, source, bytes) == cuda_success_k;
    else if constexpr (vendor_ == gpu_vendor_t::amd_k)
        return test::gpu_drivers().hip.copy(destination, source, bytes, hip_copy_device_to_host_k) == hip_success_k;
    else return std::memcpy(destination, source, bytes), true;
}

/** Whether a sharded index places its shards on GPUs - the VRAM residency, whichever vendor's. */
template <typename indexes_type_>
concept vram_collection = requires { indexes_type_::vendor_k; };

/** The VRAM reading of the search race: the index admits no search during a mutation, so the
 *  exclusion is the executor's - a search on a claimed executor is refused, and the reader walks
 *  the published entries only once the writer has returned. Two shards take the batch with nothing
 *  reserved, and every key counts exactly once across them. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_search_race(u64_t seed, typename indexes_type_::executor_t& placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;

    std::size_t const nodes_count = 4 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement)) && succeeded(place_shards(indexes, placement)));

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;

    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_race"));
    std::vector<key_t> keys(nodes_count);
    for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = dense_key<key_t>(1 + row);
    std::array<key_t, 8> found_keys {};
    std::array<typename indexes_type_::distance_t, 8> found_distances {};

    // A claimed executor refuses the search outright, before any scratch is carved.
    {
        auto [claim, claim_status] = executor.claim();
        failures += expect(succeeded(claim_status));
        failures += expect_eq(indexes
                                  .search(vectors.view().row(0).as_vector(), std::span<key_t> {found_keys},
                                          std::span {found_distances}, {}, executor)
                                  .status,
                              status_t::contended_k);
    }

    auto [insert_result, insert_status] = indexes.insert(std::span<key_t const> {keys.data(), nodes_count},
                                                         vectors.view(), executor);
    failures += expect_eq(insert_status, status_t::success_k);
    failures += expect_eq(indexes.size(), nodes_count);
    failures += expect(indexes.shard(0).size() && indexes.shard(1).size());
    std::size_t miscounted = 0;
    for (key_t const& key : keys) miscounted += indexes.count(key) != 1;
    failures += expect_eq(miscounted, 0u);
    failures += expect(!indexes.contains(dense_key<key_t>(0)) && indexes.count(dense_key<key_t>(0)) == 0);

    // Every published entry resolves to a key inside the assigned range - never the key column's
    // zero-fill.
    std::size_t mislabeled = 0;
    for (std::size_t searches = 0; searches != dense_reader_minimal_searches_k; ++searches) {
        auto const [search_result, search_status] = indexes.search(
            vectors.view().row(searches % nodes_count).as_vector(), std::span<key_t> {found_keys},
            std::span {found_distances}, {}, executor);
        failures += expect_eq(search_status, status_t::success_k);
        for (std::size_t match = 0; match != search_result.count; ++match)
            mislabeled += dense_key_number(found_keys[match]) < 1 || dense_key_number(found_keys[match]) > nodes_count;
    }
    failures += expect_eq(mislabeled, 0u);
    return failures;
}

/** The planned insertion: launched entries stay out of the size, the key lookups and a search until
 *  @c finish publishes every one, a plan short of entries refuses to finish and stays whole, its
 *  gate turns a second batch away, and a plan dying unfinished publishes nothing and leaves its
 *  shard refusing every later batch. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_planned_insert(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, half = rows_count / 2;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> writer {placement}, reader {placement};
    if (failed(writer.status) || failed(reader.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement)));
    failures += expect(succeeded(indexes.reserve({rows_count}).status));
    auto const vectors = make_vectors<scalar_t>(2 * rows_count, dimensions, stream_key(seed, "indexes_planned"));
    std::vector<key_t> keys(2 * rows_count);
    for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = dense_key<key_t>(1 + row);
    auto rows_of = [&](std::size_t first, std::size_t count) noexcept {
        return vectors(nk::range(first, first + count), nk::slice);
    };
    auto keys_of = [&](std::size_t first, std::size_t count) noexcept {
        return std::span<key_t const> {keys.data() + first, count};
    };
    std::array<key_t, 8> found_keys {};
    std::array<typename indexes_type_::distance_t, 8> found_distances {};

    {
        auto [plan, plan_status] = indexes.plan_insert(rows_count, writer.executor);
        failures += expect_eq(plan_status, status_t::success_k);
        failures += expect_eq(plan.insert(half, keys_of(half, half), rows_of(half, half)), status_t::success_k);
        failures += expect_eq(indexes.size(), 0u);
        failures += expect(!indexes.contains(keys[half]));
        auto const [search_result, search_status] = indexes.search(vectors.view().row(half).as_vector(),
                                                                   std::span<key_t> {found_keys},
                                                                   std::span {found_distances}, {}, reader.executor);
        failures += expect(succeeded(search_status) && search_result.count == 0);
        failures += expect_eq(indexes.insert(keys_of(0, 1), rows_of(0, 1), reader.executor).status,
                              status_t::contended_k);
        failures += expect_eq(plan.finish().status, status_t::config_mismatch_k);
        failures += expect_eq(plan.insert(0, keys_of(0, half), rows_of(0, half)), status_t::success_k);
        auto const [insert_result, finish_status] = plan.finish();
        failures += expect_eq(finish_status, status_t::success_k);
        failures += expect_eq(insert_result.new_size, rows_count);
    }
    failures += expect_eq(indexes.size(), rows_count);
    failures += expect(indexes.contains(keys[0]) && indexes.contains(keys[rows_count - 1]));

    {
        auto [plan, plan_status] = indexes.plan_insert(rows_count, writer.executor);
        failures += expect_eq(plan_status, status_t::success_k);
        failures += expect_eq(plan.insert(0, keys_of(rows_count, half), rows_of(rows_count, half)),
                              status_t::success_k);
    }
    failures += expect_eq(indexes.size(), rows_count);
    failures += expect_eq(indexes.status(), status_t::cancelled_k);
    failures += expect(!indexes.contains(keys[rows_count]));
    failures += expect_eq(indexes.insert(keys_of(rows_count, 1), rows_of(rows_count, 1), writer.executor).status,
                          status_t::cancelled_k);
    return failures;
}

/** The VRAM reading of the sharded indexer search, keys equal to their rows: every shard walks the same queries
 *  under the same heads with candidates as wide as the index, so the merged top of every query is
 *  the oracle's. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_indexer_search(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    if constexpr (kernel_ != nk_kernel_dot_k || nk::is_integral_dtype<scalar_t>()) return 0;
    else {
        using key_t = typename indexes_type_::key_t;
        constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 4, wanted = 16;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::size_t failures = 0;

        dense_indexes_executor<indexes_type_> driver {placement};
        if (failed(driver.status)) return failures + expect(false);
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(indexes, placement)));
        failures += expect(succeeded(indexes.reserve({rows_count}).status));
        auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "indexes_indexer"));
        std::array<key_t, rows_count> keys;
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(row);
        failures += expect(
            succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));

        auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
        dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
        typename indexes_type_::search_config_t const exhaustive {rows_count, wanted};
        auto found_keys = nk::matrix<key_t>::zeros({queries_count, wanted}).value;
        auto found_distances = nk::matrix<distance_t>::zeros({queries_count, wanted}).value;
        std::array<std::size_t, queries_count> counts;
        failures += expect(succeeded(indexes
                                         .search(queries, inputs.heads(),
                                                 {found_keys.span(), found_distances.span(), std::span {counts}},
                                                 exhaustive, driver.executor)
                                         .status));
        nk::matrix_view<key_t> const ranked_keys = found_keys.view();
        nk::matrix_view<distance_t> const ranked_distances = found_distances.view();
        for (std::size_t query = 0; query != queries_count; ++query) {
            nk::range_t const kept = nk::range(std::size_t {0}, counts[query]);
            failures += dense_expect_indexer_matches(inputs, query, ranked_keys(query, kept, nk::slice).as_vector(),
                                                     ranked_distances(query, kept, nk::slice).as_vector());
        }
        return failures;
    }
}

/** The device outputs of the sharded verbs, over two shards on the device under test, agree with
 *  their host forms. The same queries ranked into host spans, merged on the host, and into spans on
 *  the first shard's device, merged there, agree on every count, on every distance, and on the keys
 *  of every run of equal distances, which the merges may order differently. An odd row count splits
 *  the shards unevenly, and a width past either shard's rows leaves both rankings short and the
 *  merged ones full. A gather into device rows matches the one copied home byte for byte, and the
 *  products packed on the device, widened to f64, name distinct rows at their exact dots, each run
 *  a stride long. With every row ranking every key, the exact device MaxSim keeps every key at its
 *  f64 total of native dots, within rounding. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_device_outputs(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t rows_count = 255, queries_count = 8, wanted = 200;
    constexpr std::size_t ranks_count = queries_count * wanted;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(indexes.emplace(driver.executor.stream(0))));
    failures += expect(succeeded(indexes.emplace(driver.executor.stream(0))));
    failures += expect(succeeded(indexes.reserve({rows_count}).status));
    auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "indexes_device_merge"));
    std::array<key_t, rows_count> keys;
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(1 + row);
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));

    typename indexes_type_::executor_t::device_arena_t arena {driver.executor.stream(0)};
    key_t* const device_keys = arena.template allocate<key_t>(ranks_count);
    distance_t* const device_distances = arena.template allocate<distance_t>(ranks_count);
    std::size_t* const device_counts = arena.template allocate<std::size_t>(queries_count);
    if (failures || !device_keys || !device_distances || !device_counts) return failures + expect(false);

    auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
    typename indexes_type_::search_config_t const exhaustive {rows_count, wanted};
    std::vector<key_t> host_keys(ranks_count), merged_keys(ranks_count);
    std::vector<distance_t> host_distances(ranks_count), merged_distances(ranks_count);
    std::array<std::size_t, queries_count> host_counts, merged_counts;
    failures += expect(
        succeeded(indexes
                      .search(queries,
                              ranked_matches_span<key_t, distance_t> {
                                  nk::matrix_span<key_t> {host_keys.data(), queries_count, wanted},
                                  nk::matrix_span<distance_t> {host_distances.data(), queries_count, wanted},
                                  std::span {host_counts}},
                              exhaustive, driver.executor)
                      .status));
    failures += expect(succeeded(indexes
                                     .search(queries,
                                             ranked_matches_span<key_t, distance_t> {
                                                 nk::matrix_span<key_t> {device_keys, queries_count, wanted},
                                                 nk::matrix_span<distance_t> {device_distances, queries_count, wanted},
                                                 std::span {device_counts, queries_count}},
                                             exhaustive, driver.executor)
                                     .status));
    failures += expect(
        dense_download<indexes_type_::vendor_k>(merged_keys.data(), device_keys, ranks_count * sizeof(key_t)) &&
        dense_download<indexes_type_::vendor_k>(merged_distances.data(), device_distances,
                                                ranks_count * sizeof(distance_t)) &&
        dense_download<indexes_type_::vendor_k>(merged_counts.data(), device_counts,
                                                queries_count * sizeof(std::size_t)));
    if (failures) return failures;

    // Both calls run the same walks and differ only in where they merge, so the distances match
    // exactly, bit for bit.
    error_stats_t merge_errors;
    for (std::size_t query = 0; query != queries_count; ++query) {
        std::size_t const found = host_counts[query];
        failures += expect_eq(found, wanted);
        failures += expect_eq(merged_counts[query], found);
        if (merged_counts[query] != found) continue;
        std::size_t const first = query * wanted;
        for (std::size_t rank = first; rank != first + found; ++rank)
            merge_errors.add(merged_distances[rank], host_distances[rank]);
        for (std::size_t begin = first, end = first; begin != first + found; begin = end) {
            while (end != first + found && host_distances[end] == host_distances[begin]) ++end;
            // A run the width cuts may keep different ties on either side.
            if (end == first + wanted) break;
            auto const by_number = [](key_t const& first, key_t const& second) noexcept {
                return dense_key_number(first) < dense_key_number(second);
            };
            std::sort(host_keys.begin() + begin, host_keys.begin() + end, by_number);
            std::sort(merged_keys.begin() + begin, merged_keys.begin() + end, by_number);
            failures += expect(
                std::equal(host_keys.begin() + begin, host_keys.begin() + end, merged_keys.begin() + begin));
        }
    }
    failures += expect_errors("device merge distances", comparison_family_t::exact_k, comparison_verdict_t::assert_k,
                              merge_errors, 0);

    // One row per key, gathered home through the first shard's staging and in place on the device.
    constexpr std::size_t gathered_keys = 5;
    std::size_t const row_values = divide_round_up<std::size_t>(dimensions, nk::dimensions_per_value<scalar_t>());
    std::size_t const row_bytes = row_values * sizeof(scalar_t);
    nk_dtype_t const dtype = dtype_of<scalar_t>();
    std::vector<scalar_t> host_rows(gathered_keys * row_values), gathered_rows(gathered_keys * row_values);
    std::array<std::size_t, gathered_keys> host_entries {}, gathered_entries {};
    scalar_t* const device_rows = arena.template allocate<scalar_t>(gathered_keys * row_values);
    std::span<key_t const> const asked {keys.data(), gathered_keys};
    failures += expect(
        device_rows &&
        succeeded(indexes.get(
            asked,
            rows_span_t {reinterpret_cast<byte_t*>(host_rows.data()), gathered_keys, dimensions, row_bytes, dtype},
            std::span {host_entries}, 1, driver.executor)) &&
        succeeded(indexes.get(
            asked, rows_span_t {reinterpret_cast<byte_t*>(device_rows), gathered_keys, dimensions, row_bytes, dtype},
            std::span {gathered_entries}, 1, driver.executor)) &&
        dense_download<indexes_type_::vendor_k>(gathered_rows.data(), device_rows, gathered_keys * row_bytes));
    failures += expect(host_entries == gathered_entries &&
                       std::all_of(host_entries.begin(), host_entries.end(), [](std::size_t n) { return n == 1; }));
    failures += expect(std::memcmp(host_rows.data(), gathered_rows.data(), gathered_keys * row_bytes) == 0);

    if constexpr (kernel_ == nk_kernel_dot_k) {
        constexpr std::size_t pairs = 16, pairs_count = queries_count * pairs;
        std::vector<key_t> packed_keys(pairs_count);
        std::vector<f64_t> packed_products(pairs_count);
        std::array<std::size_t, queries_count> packed_scored {};
        std::array<std::size_t, queries_count + 1> packed_offsets {};
        std::size_t* const device_offsets = arena.template allocate<std::size_t>(queries_count + 1);
        key_t* const tape_keys = arena.template allocate<key_t>(pairs_count);
        f64_t* const tape_products = arena.template allocate<f64_t>(pairs_count);
        std::size_t const zero = 0;
        if (!device_offsets || !tape_keys || !tape_products ||
            failed(arena.upload(device_offsets, &zero, sizeof(zero))) || failed(arena.wait()))
            return failures + expect(false);
        products_tape<key_t, f64_t> const tape {{tape_keys, pairs_count},
                                                {tape_products, pairs_count},
                                                {device_offsets, queries_count + 1},
                                                pairs};
        failures += expect(
            succeeded(indexes.multiply(queries, tape, std::span {packed_scored}, exhaustive, driver.executor).status));
        failures += expect(
            dense_download<indexes_type_::vendor_k>(packed_offsets.data(), device_offsets,
                                                    (queries_count + 1) * sizeof(std::size_t)) &&
            dense_download<indexes_type_::vendor_k>(packed_keys.data(), tape_keys, pairs_count * sizeof(key_t)) &&
            dense_download<indexes_type_::vendor_k>(packed_products.data(), tape_products,
                                                    pairs_count * sizeof(f64_t)));
        if (failures) return failures;
        for (std::size_t query = 0; query != queries_count; ++query) {
            std::size_t const first = packed_offsets[query], kept = min_of(packed_scored[query], pairs);
            failures += expect(packed_scored[query] > pairs);
            failures += expect_eq(packed_offsets[query + 1] - first, kept);
            if (packed_offsets[query + 1] - first != kept) continue;
            failures += dense_expect_product_row<rows_count>(
                queries.row(query).as_vector(), vectors.view(),
                nk::vector_view<key_t> {std::span {packed_keys}.subspan(first, kept)},
                nk::vector_view<f64_t> {std::span {packed_products}.subspan(first, kept)},
                [](key_t key) noexcept { return dense_key_number(key) - 1; });
        }

        // Every row ranks every key, so every key is nominated and kept, and the device's f64 sum
        // of its native dots over the group's rows differs from the host's only by rounding.
        constexpr std::size_t tokens_per_query = 4, groups = queries_count / tokens_per_query, kept = rows_count;
        typename indexes_type_::maxsim_config_t const maxsim {exhaustive, kept};
        std::vector<key_t> host_maxsim_keys(groups * kept);
        std::vector<f64_t> host_maxsim_scores(groups * kept);
        std::array<std::size_t, groups> maxsim_scored {};
        std::array<std::size_t, groups + 1> offsets {}, maxsim_offsets {};
        for (std::size_t group = 0; group <= groups; ++group) offsets[group] = group * tokens_per_query;
        std::size_t* const tape_offsets = arena.template allocate<std::size_t>(groups + 1);
        key_t* const maxsim_keys = arena.template allocate<key_t>(groups * kept);
        f64_t* const maxsim_scores = arena.template allocate<f64_t>(groups * kept);
        if (!tape_offsets || !maxsim_keys || !maxsim_scores ||
            failed(arena.upload(tape_offsets, &zero, sizeof(zero))) || failed(arena.wait()))
            return failures + expect(false);
        products_tape<key_t, f64_t> const maxsim_tape {{maxsim_keys, groups * kept},
                                                       {maxsim_scores, groups * kept},
                                                       {tape_offsets, groups + 1},
                                                       kept};
        failures += expect(succeeded(indexes
                                         .maxsim(queries, std::span<std::size_t const> {offsets},
                                                 std::span {maxsim_scored}, maxsim_tape, maxsim, driver.executor)
                                         .status));
        failures += expect(dense_download<indexes_type_::vendor_k>(maxsim_offsets.data(), tape_offsets,
                                                                   (groups + 1) * sizeof(std::size_t)) &&
                           dense_download<indexes_type_::vendor_k>(host_maxsim_keys.data(), maxsim_keys,
                                                                   groups * kept * sizeof(key_t)) &&
                           dense_download<indexes_type_::vendor_k>(host_maxsim_scores.data(), maxsim_scores,
                                                                   groups * kept * sizeof(f64_t)));
        if (failures) return failures;
        error_stats_t maxsim_errors;
        for (std::size_t group = 0; group != groups; ++group) {
            std::size_t const first = maxsim_offsets[group], count = maxsim_offsets[group + 1] - first;
            failures += expect_eq(maxsim_scored[group], rows_count) + expect_eq(count, rows_count);
            if (count != rows_count) continue;
            std::array<bool, rows_count> found {};
            for (std::size_t rank = 0; rank != count; ++rank) {
                std::size_t const row = dense_key_number(host_maxsim_keys[first + rank]) - 1;
                failures += expect(row < rows_count && !found[row]);
                if (row >= rows_count || found[row]) continue;
                found[row] = true;
                f64_t exact = 0;
                for (std::size_t token = 0; token != tokens_per_query; ++token)
                    exact += static_cast<f64_t>(
                        dense_native_dot(queries.row(group * tokens_per_query + token).as_vector(),
                                         vectors.view().row(row).as_vector()));
                maxsim_errors.add(host_maxsim_scores[first + rank], exact);
            }
        }
        failures += expect_errors("device maxsim scores", comparison_family_t::approximate_k,
                                  comparison_verdict_t::assert_k, maxsim_errors, 0);
    }
    return failures;
}

/** A @c get on an executor that has run nothing finds a key only one of two shards holds: the shard
 *  holding none of the asked keys carves nothing instead of failing, and the row comes back as the
 *  inserting executor reads it. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_get_cold(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> writer {placement}, reader {placement};
    if (failed(writer.status) || failed(reader.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement)) && succeeded(place_shards(indexes, placement)));
    failures += expect(succeeded(indexes.reserve({1}).status));
    auto const vectors = make_vectors<scalar_t>(1, dimensions, stream_key(seed, "indexes_get_cold"));
    std::array<key_t, 1> const keys {dense_key<key_t>(1)};
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), writer.executor).status));
    failures += expect_eq(indexes.size(), 1u);

    std::size_t const row_values = divide_round_up<std::size_t>(dimensions, nk::dimensions_per_value<scalar_t>());
    std::size_t const row_bytes = row_values * sizeof(scalar_t);
    std::vector<scalar_t> cold_row(row_values), warm_row(row_values);
    std::array<std::size_t, 1> cold_entries {}, warm_entries {};
    auto get_row = [&](std::vector<scalar_t>& row, std::array<std::size_t, 1>& entries,
                       typename indexes_type_::executor_t& executor) noexcept {
        return indexes.get(
            std::span<key_t const> {keys},
            rows_span_t {reinterpret_cast<byte_t*>(row.data()), 1, dimensions, row_bytes, dtype_of<scalar_t>()},
            std::span {entries}, 1, executor);
    };
    failures += expect_eq(get_row(cold_row, cold_entries, reader.executor), status_t::success_k);
    failures += expect_eq(get_row(warm_row, warm_entries, writer.executor), status_t::success_k);
    failures += expect(cold_entries[0] == 1 && warm_entries[0] == 1);
    failures += expect(std::memcmp(cold_row.data(), warm_row.data(), row_bytes) == 0);
    return failures;
}

/** A RAM group's monolith loaded into the VRAM collection and saved again comes back byte for byte,
 *  norms included, and the NVMe tier views the VRAM save as it views the RAM one. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_nvme_roundtrip(u64_t seed, typename indexes_type_::executor_t& placement) {
    using key_t = typename indexes_type_::key_t;
    using nvme_t = nvme_indexes_for<key_t, indexes_type_::vector_dtype_k, nk_cap_serial_k, blocking_files_engine<>>;
    using scalar_t = typename nvme_t::scalar_t;
    std::size_t const rows_count = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_t> reference {"vram_nvme_roundtrip", kernel_, rows_count, dimensions, seed};
    std::size_t failures = reference.failures;
    failures += expect(succeeded(
        reference.whole().save(reference.checkpoint.monolith.data(), save_config_t {}, reference.builder.executor)));

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement)));
    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status) || failures) return failures + expect(false);
    nvme_checkpoint_t const resaved {"vram_nvme_resaved"};
    failures += expect(
        succeeded(indexes.load(reference.checkpoint.monolith.data(), load_config_t {}, driver.executor)));
    failures += expect(
        succeeded(indexes.save(resaved.monolith.data(), dense_indexes_save_config<indexes_type_>(), driver.executor)));

    auto [ram_file, ram_status] = mapped_file_t::read(reference.checkpoint.monolith.data());
    auto [vram_file, vram_status] = mapped_file_t::read(resaved.monolith.data());
    failures += expect(succeeded(ram_status) && succeeded(vram_status));
    if (failures) return failures;
    bytes_view_t const ram_bytes = ram_file.view(), vram_bytes = vram_file.view();
    failures += expect(ram_bytes.size() == vram_bytes.size() &&
                       std::memcmp(ram_bytes.data(), vram_bytes.data(), ram_bytes.size()) == 0);

    nvme_t viewed {forkunion_allocator_t {no_memory_domain_k}};
    failures += expect_eq(viewed.view(resaved.monolith.data()), status_t::success_k);
    failures += expect_eq(viewed.size(), rows_count);
    return failures;
}

/** Device shards compact into one host shard: every entry crosses with its key exactly once, the
 *  source ends empty, nodes of every source shard link into the others', and the merged graph walks
 *  within noise of the recall at ten one built from every entry at once reaches. Past
 *  @c connectivity+1 tiny shards the fold splits them into bins, each member keeping a slot of a
 *  node from every other, and the run stops once the re-selection finished and resumes. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_compact(u64_t seed, typename indexes_type_::executor_t& placement,
                                            std::size_t shards_count) {
    using key_t = typename indexes_type_::key_t;
    using scalar_t = typename indexes_type_::scalar_t;
    using nvme_t = nvme_indexes_for<key_t, indexes_type_::vector_dtype_k, nk_cap_serial_k, blocking_files_engine<>>;
    using ram_t = typename nvme_reference<nvme_t>::ram_t;
    using node_id_t = typename ram_t::index_t::node_id_t;
    constexpr std::size_t samples_k = 32;
    constexpr std::size_t noise_percent_k = 10;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver {placement};
    dense_indexes_executor<ram_t> host;
    if (failed(driver.status) || failed(host.status)) return failures + expect(false);
    auto const vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_compact"));
    auto const queries = make_vectors<scalar_t>(samples_k, dimensions, stream_key(seed, "indexes_compact", 1));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ source {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    for (std::size_t shard = 0; shard != shards_count; ++shard)
        failures += expect(succeeded(place_shards(source, placement)));
    failures += expect_eq(source.reserve({rows}).status, status_t::success_k);
    failures += expect(
        succeeded(source.insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), driver.executor).status));
    // The donor shards are appended in their order, so each one's nodes follow the counts of the
    // shards before it.
    std::vector<std::size_t> shard_ends(shards_count);
    for (std::size_t shard = 0, end = 0; shard != shards_count; ++shard)
        shard_ends[shard] = end += source.shard(shard).size();
    ram_t target {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(target.emplace(no_memory_domain_k)));

    indexes_type_* const sources[1] = {&source};
    std::span<indexes_type_* const> const compacted {sources};
    // The run stops once the re-selection finished, moving nothing, and the next call resumes it.
    auto resumed = [&] {
        auto [plan, plan_status] = plan_compact(compacted, target, driver.executor, host.executor);
        failures += expect_eq(plan_status, status_t::success_k);
        auto cancel = [](std::size_t, std::size_t) noexcept { return continuation_t::cancel_k; };
        failures += expect_eq(plan.run(driver.executor, host.executor, to_writable_span(host.traversals), cancel),
                              status_t::cancelled_k);
        failures += expect_eq(source.size(), rows);
        failures += expect_eq(plan.run(driver.executor, host.executor, to_writable_span(host.traversals)),
                              status_t::success_k);
        return plan.finish(host.executor, to_writable_span(host.traversals));
    };
    auto const [result, status] = shards_count <= connectivity_16_k + 1
                                      ? compact(compacted, target, driver.executor, host.executor)
                                      : resumed();
    failures += expect_eq(status, status_t::success_k);
    failures += expect(result.source_shards == shards_count && result.target_shards == 1 &&
                       result.moved_entries == rows);
    failures += expect(result.graph.reselected_nodes > 0);
    failures += expect_eq(source.size(), 0u);
    failures += expect_eq(target.size(), rows);
    failures += expect(target.dimensions() == dimensions && target.connectivity() == connectivity_16_k);
    std::size_t miscounted = 0;
    for (key_t const key : keys) miscounted += target.count(key) != 1;
    failures += expect_eq(miscounted, 0u);

    // A neighbor list holding a node of another source shard is a link the fold made, and none
    // points past the nodes.
    ram_cursor<typename ram_t::index_t, synchronization_t::immutable_k, forkunion_allocator_t> cursor {target.shard(0)};
    failures += expect_eq(cursor.reserve(forkunion_allocator_t {no_memory_domain_k}), status_t::success_k);
    auto shard_of = [&](node_id_t row) noexcept {
        return std::upper_bound(shard_ends.begin(), shard_ends.end(), std::size_t {row}) - shard_ends.begin();
    };
    std::size_t crossing = 0, stray = 0;
    for (node_id_t row = 0; row != rows; ++row) {
        if (failed(cursor.seek(row))) continue;
        bool crosses = false;
        for (node_id_t const neighbor : cursor.neighbors()) {
            stray += neighbor >= rows;
            crosses |= neighbor < rows && shard_of(neighbor) != shard_of(row);
        }
        crossing += crosses;
    }
    failures += expect(crossing > 0);
    failures += expect_eq(stray, 0u);

    ram_t built {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += dense_indexes_build(built, host.executor, vectors, keys);
    std::size_t const baseline = dense_recall_at_ten(built.shard(0), 0, queries.view());
    std::size_t const merged = dense_recall_at_ten(target.shard(0), 0, queries.view());
    failures += expect(merged * 100 + noise_percent_k * samples_k * 10 >= baseline * 100);
    return failures;
}

/** A two-shard index splits into two receivers of two shards each: the sizes add up, every key
 *  counts once across the receivers, the source ends empty, and the receivers' merged recall at
 *  ten against an exact reference stays within noise of the source's. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_split(u64_t seed, typename indexes_type_::executor_t& placement) {
    using key_t = typename indexes_type_::key_t;
    using scalar_t = typename indexes_type_::scalar_t;
    using distance_t = typename indexes_type_::distance_t;
    using nvme_t = nvme_indexes_for<key_t, indexes_type_::vector_dtype_k, nk_cap_serial_k, blocking_files_engine<>>;
    using ram_t = typename nvme_reference<nvme_t>::ram_t;
    using reference_t = typename ram_t::index_t;
    constexpr std::size_t samples_k = 32;
    constexpr std::size_t wanted_k = 10;
    constexpr std::size_t noise_percent_k = 10;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver {placement};
    dense_indexes_executor<ram_t> host;
    if (failed(driver.status) || failed(host.status)) return failures + expect(false);
    auto const vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_split"));
    auto const queries = make_vectors<scalar_t>(samples_k, dimensions, stream_key(seed, "indexes_split", 1));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ source {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ first {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ second {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    for (indexes_type_* placed : {&source, &first, &second})
        failures += expect(succeeded(place_shards(*placed, placement)) && succeeded(place_shards(*placed, placement)));
    failures += expect_eq(source.reserve({rows}).status, status_t::success_k);
    failures += expect(
        succeeded(source.insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), driver.executor).status));

    // The exact tenth distance of every query over all rows, from a host reference's scan.
    ram_t built {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += dense_indexes_build(built, host.executor, vectors, keys);
    reference_t& reference = built.shard(0);
    typename reference_t::traversal_context_t context = dense_make_context<reference_t>(0);
    typename reference_t::search_config_t const scan {
        .expansion_factor = static_cast<typename reference_t::candidates_count_t>(wanted_k),
        .traversal = traversal_t::scan_k};
    std::vector<f64_t> tenths(samples_k, 0);
    for (std::size_t query = 0; query != samples_k; ++query)
        if (succeeded(reference.search(queries.view().row(query).as_vector(), scan, context).status) &&
            !context.results.empty())
            tenths[query] = context.results[context.results.size() - 1].distance;

    // The ten nearest the walks of every index in `parts` return, merged, within each tenth.
    auto recall = [&](std::initializer_list<indexes_type_*> parts) {
        std::size_t found = 0;
        for (std::size_t query = 0; query != samples_k; ++query) {
            std::vector<distance_t> merged;
            for (indexes_type_* part : parts) {
                std::array<key_t, wanted_k> found_keys {};
                std::array<distance_t, wanted_k> found_distances {};
                auto const [result, status] = part->search(
                    queries.view().row(query).as_vector(), std::span<key_t> {found_keys},
                    std::span<distance_t> {found_distances}, {}, driver.executor);
                if (succeeded(status))
                    merged.insert(merged.end(), found_distances.begin(), found_distances.begin() + result.count);
            }
            std::sort(merged.begin(), merged.end());
            f64_t const tenth = tenths[query];
            for (std::size_t rank = 0; rank != min_of(wanted_k, merged.size()); ++rank)
                found += merged[rank] <= tenth + 1e-3 * max_of(std::fabs(tenth), 1.0);
        }
        return found;
    };
    std::size_t const source_recall = recall({&source});

    indexes_type_* refused[2] = {&first, &source};
    failures += expect_eq(source.plan_split(std::span<indexes_type_*> {refused}, {}, driver.executor).status,
                          status_t::config_mismatch_k);
    indexes_type_* receivers[2] = {&first, &second};
    auto const [split, split_status] = source.split(std::span<indexes_type_*> {receivers}, {{}, seed}, driver.executor);
    failures += expect_eq(split_status, status_t::success_k);
    failures += expect_eq(split.reachability.audited_nodes, rows);
    failures += expect_eq(source.size(), 0u);
    failures += expect_eq(first.size() + second.size(), rows);
    failures += expect(first.size() && second.size());

    std::vector<std::size_t> in_first(rows), in_second(rows);
    failures += expect(succeeded(first.count(keys, in_first, driver.executor)) &&
                       succeeded(second.count(keys, in_second, driver.executor)));
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != rows; ++row) miscounted += in_first[row] + in_second[row] != 1;
    failures += expect_eq(miscounted, 0u);

    std::size_t const split_recall = recall({&first, &second});
    failures += expect(split_recall * 100 + noise_percent_k * samples_k * wanted_k >= source_recall * 100);
    return failures;
}

/** A split refused for want of a second receiver leaves the donor whole: its size and the count of
 *  every key stay as inserted. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_split_refused(u64_t seed, typename indexes_type_::executor_t& placement) {
    using key_t = typename indexes_type_::key_t;
    using scalar_t = typename indexes_type_::scalar_t;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return failures + expect(false);
    auto const vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_split_refused"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ donor {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ lone {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    for (indexes_type_* placed : {&donor, &lone})
        failures += expect(succeeded(place_shards(*placed, placement)) && succeeded(place_shards(*placed, placement)));
    failures += expect_eq(donor.reserve({rows}).status, status_t::success_k);
    failures += expect(
        succeeded(donor.insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), driver.executor).status));

    indexes_type_* receivers[1] = {&lone};
    failures += expect_eq(donor.plan_split(std::span<indexes_type_*> {receivers}, {}, driver.executor).status,
                          status_t::config_mismatch_k);
    failures += expect_eq(donor.size(), rows);
    std::size_t miscounted = 0;
    for (key_t const& key : keys) miscounted += donor.count(key) != 1;
    failures += expect_eq(miscounted, 0u);
    return failures;
}

/** A vendor this build leaves out composes nothing, so its shard instantiates empty. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
std::size_t test_vram_indexes_configuration(char const*, u64_t, executor_type_&) {
    return 0;
}

/** Past @c shard_capacity, entries open generations of device shards on the same placements: one
 *  batch fills as many as it needs with every key resolving once, and a batch crossing the bound
 *  tops the open generation up and goes in whole, its newest entries reachable by a search. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_generations(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t bound = 1000, rows = 5000;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return failures + expect(false);
    auto const vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_generations"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    auto insert = [&](indexes_type_& indexes, std::size_t first, std::size_t count) {
        return indexes
            .insert(std::span<key_t const> {keys.data() + first, count},
                    vectors(nk::range(first, first + count), nk::slice), driver.executor)
            .status;
    };
    auto shard_sizes = [](indexes_type_ const& indexes) {
        std::vector<std::size_t> sizes;
        for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard)
            sizes.push_back(indexes.shard(shard).size());
        return sizes;
    };
    auto miscounted = [&](indexes_type_ const& indexes, std::size_t count) {
        std::size_t wrong = 0;
        for (std::size_t row = 0; row != count; ++row) wrong += indexes.count(keys[row]) != 1;
        return wrong;
    };
    auto bounded = [&](indexes_type_& indexes) {
        indexes.bound_shards(bound);
        return succeeded(place_shards(indexes, placement)) && succeeded(place_shards(indexes, placement)) &&
               succeeded(indexes.reserve({bound}).status) && indexes.shard_capacity() == bound;
    };

    // One batch over two empty shards fills two generations and half a third.
    indexes_type_ filled {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(bounded(filled));
    failures += expect_eq(insert(filled, 0, rows), status_t::success_k);
    failures += expect_eq(filled.size(), rows);
    failures += expect((shard_sizes(filled) == std::vector<std::size_t> {1000, 1000, 1000, 1000, 500, 500}));
    failures += expect_eq(miscounted(filled, rows), 0u);

    // 750 entries a shard, then a batch of 1000 tops both up to the bound and opens a generation
    // for the other 500, whole, and a search reaches the new generation's entries.
    indexes_type_ crossed {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(bounded(crossed));
    failures += expect_eq(insert(crossed, 0, 1500), status_t::success_k);
    failures += expect_eq(crossed.shards_count(), 2u);
    failures += expect_eq(insert(crossed, 1500, 1000), status_t::success_k);
    failures += expect((shard_sizes(crossed) == std::vector<std::size_t> {1000, 1000, 250, 250}));
    failures += expect_eq(miscounted(crossed, 2500), 0u);
    if constexpr (kernel_ != nk_kernel_dot_k) {
        std::array<key_t, 8> found_keys {};
        std::array<typename indexes_type_::distance_t, 8> found_distances {};
        auto const [result, status] = crossed.search(vectors.view().row(2499).as_vector(),
                                                     std::span<key_t> {found_keys}, std::span {found_distances}, {},
                                                     driver.executor);
        bool reached = false;
        for (std::size_t match = 0; match != result.count; ++match) reached |= found_keys[match] == keys[2499];
        failures += expect(succeeded(status) && reached);
    }
    return failures;
}

/** Composes one (dtype, kernel) cell's sharded suite over the collection of @p executor 's vendor
 *  on its device, reported under @p name prefixed by the vendor's toolkit. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
    requires vram_compiled<executor_type_::vendor_k>
std::size_t test_vram_indexes_configuration(char const* name, u64_t seed, executor_type_& executor) {
    using cell_t = vram_cell<executor_type_::vendor_k, dtype_, kernel_>;
    using indexes_t = typename cell_t::indexes_t;
    std::string const vendor_name = fmt::format("{}-{}", cell_t::vendor_prefix_k, name);

    // The suite is composed here, not reflected: every call below is a capability claim
    // this sharded, executor-driven category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_indexes_guards<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_duplicate_keys<indexes_t, kernel_>(seed, executor);
    failures_count += dense_expect_duplicates_build<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_save_load<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_part_load<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_buffer_roundtrip<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_cancelled_load<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_concurrent_add<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_vram_indexes_search_race<indexes_t, kernel_>(seed, executor);
    if constexpr (executor_type_::vendor_k == gpu_vendor_t::nvidia_k)
        failures_count += test_dense_indexes_scan<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_planned_insert<indexes_t, kernel_>(seed, executor);
    // Apple's tower opens no generations of shards.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        failures_count += test_dense_vram_indexes_generations<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_refine<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_absorb<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_absorb_many<indexes_t, kernel_>(seed, executor);
    // Apple's split is unverified until an Apple run takes this cell.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        failures_count += test_dense_vram_indexes_split<indexes_t, kernel_>(seed, executor);
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        failures_count += test_dense_vram_indexes_split_refused<indexes_t, kernel_>(seed, executor);
    // Apple's tower has no compaction into RAM.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        for (std::size_t const shards : {std::size_t {2}, std::size_t {2} * (connectivity_16_k + 1)})
            failures_count += test_dense_vram_indexes_compact<indexes_t, kernel_>(seed, executor, shards);
    failures_count += test_dense_indexes_products<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_maxsim<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_vram_indexes_indexer_search<indexes_t, kernel_>(seed, executor);
    // Apple merges every ranking on the host, so its two merges are one path.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        failures_count += test_dense_vram_indexes_device_outputs<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_vram_indexes_get_cold<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_vram_indexes_nvme_roundtrip<indexes_t, kernel_>(seed, executor);
    return report(vendor_name.c_str(), failures_count);
}

/** The sharded VRAM suite entry its translation unit defines once per vendor, for its executor, and
 *  each carried vendor's entry composes explicitly over the executor it opened; returns the suite's
 *  failure count. */
template <typename executor_type_>
std::size_t check_vram_indexes(settings_t const& settings, executor_type_& executor);

} // namespace unum::usearch::test::dense

/*  The residency forms specialize templates of the enclosing @c test namespace, which only a
 *  scope enclosing it may do, so they follow the namespace at file scope and qualify what they name. */

/** The VRAM form: one stream or queue of its own on the device of @p placement, the executor the
 *  vendor's entry opened. */
template <us::test::dense::vram_collection indexes_type_>
struct us::test::dense_indexes_executor<indexes_type_> {
    using executor_t = typename indexes_type_::executor_t;

    executor_t executor;
    us::utilities::status_t status = us::utilities::status_t::uninitialized_k;

    explicit dense_indexes_executor(executor_t const& placement) noexcept {
        int const device = placement.device(0);
        auto [executor, spawn_status] = spawn_(std::span<int const> {&device, 1});
        status = spawn_status;
        this->executor = std::move(executor);
    }
    dense_indexes_executor(dense_indexes_executor const&) = delete;
    dense_indexes_executor& operator=(dense_indexes_executor const&) = delete;

    /** Opens @p devices through the vendor's driver table the harness loaded. */
    static us::utilities::expected<executor_t> spawn_(std::span<int const> devices) noexcept {
        us::utilities::gpu_drivers_t& drivers = us::test::gpu_drivers();
        if constexpr (executor_t::vendor_k == us::utilities::gpu_vendor_t::nvidia_k)
            return executor_t::spawn(drivers.cuda, devices);
        else if constexpr (executor_t::vendor_k == us::utilities::gpu_vendor_t::amd_k)
            return executor_t::spawn(drivers.hip, devices);
        else return executor_t::spawn(drivers.metal_contexts, devices);
    }

    /** Places the shards one sharded scenario drives: one VRAM shard on the device of
     *  @p placement; each further call places one more beside it. */
    static us::utilities::status_t place_shards(indexes_type_& indexes, executor_t& placement) noexcept {
        return indexes.emplace(placement.stream(0));
    }

    /** Runs @p verb over a copy of the host @p tape on the executor's device, where the tape verbs
     *  write, its first offset uploaded, and copies the written keys, scores and offsets home. */
    template <typename key_type_, typename score_type_, typename verb_type_>
    us::utilities::expected<typename indexes_type_::search_result_t> on_device(
        us::dense::products_tape<key_type_, score_type_> tape, verb_type_&& verb) noexcept {
        using us::utilities::status_t;
        typename executor_t::device_arena_t arena {executor.stream(0)};
        key_type_* const keys = arena.template allocate<key_type_>(tape.keys.size());
        score_type_* const scores = arena.template allocate<score_type_>(tape.scores.size());
        std::size_t* const offsets = arena.template allocate<std::size_t>(tape.offsets.size());
        if (!keys || !scores || !offsets) return {{}, status_t::bad_alloc_k};
        if (failed(arena.upload(offsets, tape.offsets.data(), sizeof(std::size_t))) || failed(arena.wait()))
            return {{}, status_t::device_error_k};
        us::utilities::expected<typename indexes_type_::search_result_t> const run = verb(
            us::dense::products_tape<key_type_, score_type_> {{keys, tape.keys.size()},
                                                              {scores, tape.scores.size()},
                                                              {offsets, tape.offsets.size()},
                                                              tape.stride});
        if (!us::test::dense::dense_download<executor_t::vendor_k>(tape.keys.data(), keys, tape.keys.size_bytes()) ||
            !us::test::dense::dense_download<executor_t::vendor_k>(tape.scores.data(), scores,
                                                                   tape.scores.size_bytes()) ||
            !us::test::dense::dense_download<executor_t::vendor_k>(tape.offsets.data(), offsets,
                                                                   tape.offsets.size_bytes()))
            return {{}, status_t::device_error_k};
        return run;
    }
};

/** The VRAM form: the one-row verbs take the executor itself, as no single context can
 *  walk every device. */
template <us::test::dense::vram_collection indexes_type_>
struct us::test::dense_indexes_row_driver<indexes_type_> {
    typename indexes_type_::executor_t& executor;
    us::utilities::status_t status = us::utilities::status_t::success_k;

    dense_indexes_row_driver(indexes_type_&, us::test::dense_indexes_executor<indexes_type_>& driver) noexcept
        : executor(driver.executor) {}
    typename indexes_type_::executor_t& handle() noexcept { return executor; }
};
