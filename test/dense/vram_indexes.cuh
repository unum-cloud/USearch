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

#include "usearch/dense/vram_amd_reshard.hpp"    // `plan_reshard` of AMD shards into RAM
#include "usearch/dense/vram_nvidia_reshard.hpp" // `plan_reshard` of NVIDIA shards into RAM

#include "dense.cuh"
#include "nvme_indexes.hpp" // `nvme_reference`, `checkpoint_files_t` - the round trip through storage
#include "vram_index.cuh"   // the vendor's `vram_cell`

namespace util = unum::usearch::utilities;

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
void test_dense_vram_indexes_search_race(u64_t seed, typename indexes_type_::executor_t& placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;

    std::size_t const nodes_count = 4 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement)) && succeeded(place_shards(indexes, placement)));

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;

    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_race"));
    nk::vector<key_t> keys = make_vector<key_t>(nodes_count);
    for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = dense_key<key_t>(1 + row);
    nk::vector<key_t> found_keys = make_vector<key_t>(8);
    nk::vector<typename indexes_type_::distance_t> found_distances = make_vector<typename indexes_type_::distance_t>(8);

    // A claimed executor refuses the search outright, before any scratch is carved.
    {
        auto [claim, claim_status] = executor.claim();
        expect_true(succeeded(claim_status));
        expect_equal(indexes
                         .search(vectors.view().row(0).as_vector(), to_writable_span(found_keys),
                                 to_writable_span(found_distances), {}, executor)
                         .status,
                     status_code_t::contended_k);
    }

    auto [insert_result, insert_status] = indexes.insert(std::span<key_t const> {keys.data(), nodes_count},
                                                         vectors.view(), executor);
    expect_equal(insert_status, status_code_t::success_k);
    expect_equal(indexes.size(), nodes_count);
    expect_true(indexes.shard(0).size() && indexes.shard(1).size());
    std::size_t miscounted = 0;
    for (key_t const& key : keys) miscounted += indexes.count(key) != 1;
    expect_equal(miscounted, 0u);
    expect_true(!indexes.contains(dense_key<key_t>(0)) && indexes.count(dense_key<key_t>(0)) == 0);

    // Every published entry resolves to a key inside the assigned range - never the key column's
    // zero-fill.
    std::size_t mislabeled = 0;
    for (std::size_t searches = 0; searches != dense_reader_minimal_searches_k; ++searches) {
        auto const [search_result, search_status] = indexes.search(
            vectors.view().row(searches % nodes_count).as_vector(), to_writable_span(found_keys),
            to_writable_span(found_distances), {}, executor);
        expect_equal(search_status, status_code_t::success_k);
        for (std::size_t match = 0; match != search_result.count; ++match)
            mislabeled += dense_key_number(found_keys[match]) < 1 || dense_key_number(found_keys[match]) > nodes_count;
    }
    expect_equal(mislabeled, 0u);
}

/** The planned insertion: launched entries stay out of the size, the key lookups and a search until
 *  @c finish publishes every one, a plan short of entries refuses to finish and stays whole, its
 *  gate turns a second batch away, and a plan dying unfinished publishes nothing and leaves its
 *  shard refusing every later batch. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_indexes_planned_insert(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, half = rows_count / 2;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> writer {placement}, reader {placement};
    if (failed(writer.status) || failed(reader.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement)));
    expect_true(succeeded(indexes.reserve({rows_count}).status));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(2 * rows_count, dimensions,
                                                                stream_key(seed, "indexes_planned"));
    nk::vector<key_t> keys = make_vector<key_t>(2 * rows_count);
    for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = dense_key<key_t>(1 + row);
    auto rows_of = [&](std::size_t first, std::size_t count) noexcept {
        return vectors(nk::range(first, first + count), nk::slice);
    };
    auto keys_of = [&](std::size_t first, std::size_t count) noexcept {
        return std::span<key_t const> {keys.data() + first, count};
    };
    nk::vector<key_t> found_keys = make_vector<key_t>(8);
    nk::vector<typename indexes_type_::distance_t> found_distances = make_vector<typename indexes_type_::distance_t>(8);

    {
        auto [plan, plan_status] = indexes.plan_insert(rows_count, writer.executor);
        expect_equal(plan_status, status_code_t::success_k);
        expect_equal(plan.insert(half, keys_of(half, half), rows_of(half, half)), status_code_t::success_k);
        expect_equal(indexes.size(), 0u);
        expect_true(!indexes.contains(keys[half]));
        auto const [search_result, search_status] = indexes.search(
            vectors.view().row(half).as_vector(), to_writable_span(found_keys), to_writable_span(found_distances), {},
            reader.executor);
        expect_true(succeeded(search_status) && search_result.count == 0);
        expect_equal(indexes.insert(keys_of(0, 1), rows_of(0, 1), reader.executor).status, status_code_t::contended_k);
        expect_equal(plan.finish().status, status_code_t::config_mismatch_k);
        expect_equal(plan.insert(0, keys_of(0, half), rows_of(0, half)), status_code_t::success_k);
        auto const [insert_result, finish_status] = plan.finish();
        expect_equal(finish_status, status_code_t::success_k);
        expect_equal(insert_result.new_size, rows_count);
    }
    expect_equal(indexes.size(), rows_count);
    expect_true(indexes.contains(keys[0]) && indexes.contains(keys[rows_count - 1]));

    {
        auto [plan, plan_status] = indexes.plan_insert(rows_count, writer.executor);
        expect_equal(plan_status, status_code_t::success_k);
        expect_equal(plan.insert(0, keys_of(rows_count, half), rows_of(rows_count, half)), status_code_t::success_k);
    }
    expect_equal(indexes.size(), rows_count);
    expect_equal(indexes.status(), status_code_t::cancelled_k);
    expect_true(!indexes.contains(keys[rows_count]));
    expect_equal(indexes.insert(keys_of(rows_count, 1), rows_of(rows_count, 1), writer.executor).status,
                 status_code_t::cancelled_k);
}

/** The VRAM reading of the sharded indexer search, keys equal to their rows: every shard walks the same queries
 *  under the same heads with candidates as wide as the index, so the merged top of every query is
 *  the oracle's. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_vram_indexes_indexer_search(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    if constexpr (kernel_ != nk_kernel_dot_k || nk::is_integral_dtype<scalar_t>()) return;
    else {
        using key_t = typename indexes_type_::key_t;
        constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 4, wanted = 16;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

        dense_indexes_executor<indexes_type_> driver {placement};
        if (failed(driver.status)) return void(expect_true(false));
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(indexes, placement)));
        expect_true(succeeded(indexes.reserve({rows_count}).status));
        nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions,
                                                                    stream_key(seed, "indexes_indexer"));
        nk::vector<key_t> keys = make_vector<key_t>(rows_count);
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(row);
        expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));

        auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
        dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
        typename indexes_type_::search_config_t const exhaustive {rows_count, wanted};
        nk::matrix<key_t> found_keys = make_matrix<key_t>(queries_count, wanted);
        nk::matrix<distance_t> found_distances = make_matrix<distance_t>(queries_count, wanted);
        std::array<std::size_t, queries_count> counts;
        expect_true(succeeded(indexes
                                  .search(queries, inputs.heads(),
                                          {found_keys.span(), found_distances.span(), std::span {counts}}, exhaustive,
                                          driver.executor)
                                  .status));
        nk::matrix_view<key_t> const ranked_keys = found_keys.view();
        nk::matrix_view<distance_t> const ranked_distances = found_distances.view();
        nk::vector<f64_t> ranked = make_vector<f64_t>(rows_count);
        for (std::size_t query = 0; query != queries_count; ++query) {
            nk::range_t const kept = nk::range(std::size_t {0}, counts[query]);
            dense_expect_indexer_matches(inputs, query, ranked_keys(query, kept, nk::slice).as_vector(),
                                         ranked_distances(query, kept, nk::slice).as_vector(),
                                         to_writable_span(ranked));
        }
        return;
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
void test_dense_vram_indexes_device_outputs(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t rows_count = 255, queries_count = 8, wanted = 200;
    constexpr std::size_t ranks_count = queries_count * wanted;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(indexes.emplace(driver.executor.stream(0))));
    expect_true(succeeded(indexes.emplace(driver.executor.stream(0))));
    expect_true(succeeded(indexes.reserve({rows_count}).status));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions,
                                                                stream_key(seed, "indexes_device_merge"));
    nk::vector<key_t> keys = make_vector<key_t>(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(1 + row);
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));

    typename indexes_type_::executor_t::device_arena_t arena {driver.executor.stream(0)};
    key_t* const device_keys = arena.template allocate<key_t>(ranks_count);
    distance_t* const device_distances = arena.template allocate<distance_t>(ranks_count);
    std::size_t* const device_counts = arena.template allocate<std::size_t>(queries_count);
    if (!expect_true(device_keys && device_distances && device_counts)) return;

    auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
    typename indexes_type_::search_config_t const exhaustive {rows_count, wanted};
    nk::matrix<key_t> host_keys = make_matrix<key_t>(queries_count, wanted);
    nk::matrix<key_t> merged_keys = make_matrix<key_t>(queries_count, wanted);
    nk::matrix<distance_t> host_distances = make_matrix<distance_t>(queries_count, wanted);
    nk::matrix<distance_t> merged_distances = make_matrix<distance_t>(queries_count, wanted);
    std::array<std::size_t, queries_count> host_counts, merged_counts;
    expect_true(succeeded(indexes
                              .search(queries,
                                      ranked_matches_span<key_t, distance_t> {host_keys.span(), host_distances.span(),
                                                                              std::span {host_counts}},
                                      exhaustive, driver.executor)
                              .status));
    expect_true(succeeded(indexes
                              .search(queries,
                                      ranked_matches_span<key_t, distance_t> {
                                          nk::matrix_span<key_t> {device_keys, queries_count, wanted},
                                          nk::matrix_span<distance_t> {device_distances, queries_count, wanted},
                                          std::span {device_counts, queries_count}},
                                      exhaustive, driver.executor)
                              .status));
    if (!expect_true(
            dense_download<indexes_type_::vendor_k>(merged_keys.data(), device_keys, ranks_count * sizeof(key_t)) &&
            dense_download<indexes_type_::vendor_k>(merged_distances.data(), device_distances,
                                                    ranks_count * sizeof(distance_t)) &&
            dense_download<indexes_type_::vendor_k>(merged_counts.data(), device_counts,
                                                    queries_count * sizeof(std::size_t))))
        return;

    // Both calls run the same walks and differ only in where they merge, so the distances match
    // exactly, bit for bit.
    error_stats_t merge_errors;
    for (std::size_t query = 0; query != queries_count; ++query) {
        std::size_t const found = host_counts[query];
        expect_equal(found, wanted);
        expect_equal(merged_counts[query], found);
        if (merged_counts[query] != found) continue;
        for (std::size_t rank = 0; rank != found; ++rank)
            merge_errors.add(merged_distances(query, rank), host_distances(query, rank));
        key_t* const host_row = host_keys.row(query).data();
        key_t* const merged_row = merged_keys.row(query).data();
        for (std::size_t begin = 0, end = 0; begin != found; begin = end) {
            while (end != found && host_distances(query, end) == host_distances(query, begin)) ++end;
            // A run the width cuts may keep different ties on either side.
            if (end == wanted) break;
            auto const by_number = [](key_t const& first, key_t const& second) noexcept {
                return dense_key_number(first) < dense_key_number(second);
            };
            std::sort(host_row + begin, host_row + end, by_number);
            std::sort(merged_row + begin, merged_row + end, by_number);
            expect_true(std::equal(host_row + begin, host_row + end, merged_row + begin));
        }
    }
    expect_errors("device merge distances", comparison_family_t::exact_k, comparison_verdict_t::assert_k, merge_errors,
                  0);

    // One row per key, gathered home through the first shard's staging and in place on the device.
    constexpr std::size_t gathered_keys = 5;
    std::size_t const row_values = divide_round_up<std::size_t>(dimensions, nk::dimensions_per_value<scalar_t>());
    std::size_t const row_bytes = row_values * sizeof(scalar_t);
    nk_dtype_t const dtype = dtype_of<scalar_t>();
    nk::matrix<scalar_t> host_rows = make_matrix<scalar_t>(gathered_keys, dimensions);
    nk::matrix<scalar_t> gathered_rows = make_matrix<scalar_t>(gathered_keys, dimensions);
    std::array<std::size_t, gathered_keys> host_entries {}, gathered_entries {};
    scalar_t* const device_rows = arena.template allocate<scalar_t>(gathered_keys * row_values);
    std::span<key_t const> const asked {keys.data(), gathered_keys};
    expect_true(
        device_rows &&
        succeeded(indexes.get(
            asked,
            rows_span_t {reinterpret_cast<byte_t*>(host_rows.data()), gathered_keys, dimensions, row_bytes, dtype},
            std::span {host_entries}, 1, driver.executor)) &&
        succeeded(indexes.get(
            asked, rows_span_t {reinterpret_cast<byte_t*>(device_rows), gathered_keys, dimensions, row_bytes, dtype},
            std::span {gathered_entries}, 1, driver.executor)) &&
        dense_download<indexes_type_::vendor_k>(gathered_rows.data(), device_rows, gathered_keys * row_bytes));
    expect_true(host_entries == gathered_entries &&
                std::all_of(host_entries.begin(), host_entries.end(), [](std::size_t n) { return n == 1; }));
    expect_true(std::memcmp(host_rows.data(), gathered_rows.data(), gathered_keys * row_bytes) == 0);

    if constexpr (kernel_ == nk_kernel_dot_k) {
        constexpr std::size_t pairs = 16, pairs_count = queries_count * pairs;
        nk::vector<key_t> packed_keys = make_vector<key_t>(pairs_count);
        nk::vector<f64_t> packed_products = make_vector<f64_t>(pairs_count);
        std::array<std::size_t, queries_count> packed_scored {};
        std::array<std::size_t, queries_count + 1> packed_offsets {};
        std::size_t* const device_offsets = arena.template allocate<std::size_t>(queries_count + 1);
        key_t* const tape_keys = arena.template allocate<key_t>(pairs_count);
        f64_t* const tape_products = arena.template allocate<f64_t>(pairs_count);
        std::size_t const zero = 0;
        if (!device_offsets || !tape_keys || !tape_products ||
            failed(arena.upload(device_offsets, &zero, sizeof(zero))) || failed(arena.wait()))
            return void(expect_true(false));
        products_tape<key_t, f64_t> const tape {{tape_keys, pairs_count},
                                                {tape_products, pairs_count},
                                                {device_offsets, queries_count + 1},
                                                pairs};
        if (!expect_true(succeeded(
                indexes.multiply(queries, tape, std::span {packed_scored}, exhaustive, driver.executor).status)) ||
            !expect_true(
                dense_download<indexes_type_::vendor_k>(packed_offsets.data(), device_offsets,
                                                        (queries_count + 1) * sizeof(std::size_t)) &&
                dense_download<indexes_type_::vendor_k>(packed_keys.data(), tape_keys, pairs_count * sizeof(key_t)) &&
                dense_download<indexes_type_::vendor_k>(packed_products.data(), tape_products,
                                                        pairs_count * sizeof(f64_t))))
            return;
        nk::vector_view<key_t> const packed_keys_view = packed_keys.view();
        nk::vector_view<f64_t> const products = packed_products.view();
        for (std::size_t query = 0; query != queries_count; ++query) {
            std::size_t const first = packed_offsets[query], kept = min_of(packed_scored[query], pairs);
            expect_true(packed_scored[query] > pairs);
            expect_equal(packed_offsets[query + 1] - first, kept);
            if (packed_offsets[query + 1] - first != kept) continue;
            dense_expect_product_row<rows_count>(
                queries.row(query).as_vector(), vectors.view(), packed_keys_view[nk::range(first, first + kept)],
                products[nk::range(first, first + kept)], [](key_t key) noexcept { return dense_key_number(key) - 1; });
        }

        // Every row ranks every key, so every key is nominated and kept, and the device's f64 sum
        // of its native dots over the group's rows differs from the host's only by rounding.
        constexpr std::size_t tokens_per_query = 4, groups = queries_count / tokens_per_query, kept = rows_count;
        typename indexes_type_::maxsim_config_t const maxsim {exhaustive, kept};
        nk::vector<key_t> host_maxsim_keys = make_vector<key_t>(groups * kept);
        nk::vector<f64_t> host_maxsim_scores = make_vector<f64_t>(groups * kept);
        std::array<std::size_t, groups> maxsim_scored {};
        std::array<std::size_t, groups + 1> offsets {}, maxsim_offsets {};
        for (std::size_t group = 0; group <= groups; ++group) offsets[group] = group * tokens_per_query;
        std::size_t* const tape_offsets = arena.template allocate<std::size_t>(groups + 1);
        key_t* const maxsim_keys = arena.template allocate<key_t>(groups * kept);
        f64_t* const maxsim_scores = arena.template allocate<f64_t>(groups * kept);
        if (!tape_offsets || !maxsim_keys || !maxsim_scores ||
            failed(arena.upload(tape_offsets, &zero, sizeof(zero))) || failed(arena.wait()))
            return void(expect_true(false));
        products_tape<key_t, f64_t> const maxsim_tape {{maxsim_keys, groups * kept},
                                                       {maxsim_scores, groups * kept},
                                                       {tape_offsets, groups + 1},
                                                       kept};
        if (!expect_true(succeeded(indexes
                                       .maxsim(queries, std::span<std::size_t const> {offsets},
                                               std::span {maxsim_scored}, maxsim_tape, maxsim, driver.executor)
                                       .status)) ||
            !expect_true(dense_download<indexes_type_::vendor_k>(maxsim_offsets.data(), tape_offsets,
                                                                 (groups + 1) * sizeof(std::size_t)) &&
                         dense_download<indexes_type_::vendor_k>(host_maxsim_keys.data(), maxsim_keys,
                                                                 groups * kept * sizeof(key_t)) &&
                         dense_download<indexes_type_::vendor_k>(host_maxsim_scores.data(), maxsim_scores,
                                                                 groups * kept * sizeof(f64_t))))
            return;
        error_stats_t maxsim_errors;
        for (std::size_t group = 0; group != groups; ++group) {
            std::size_t const first = maxsim_offsets[group], count = maxsim_offsets[group + 1] - first;
            expect_equal(maxsim_scored[group], rows_count);
            expect_equal(count, rows_count);
            if (count != rows_count) continue;
            std::array<bool, rows_count> found {};
            for (std::size_t rank = 0; rank != count; ++rank) {
                std::size_t const row = dense_key_number(host_maxsim_keys[first + rank]) - 1;
                expect_true(row < rows_count && !found[row]);
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
        expect_errors("device maxsim scores", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                      maxsim_errors, 0);
    }
}

/** A @c get on an executor that has run nothing finds a key only one of two shards holds: the shard
 *  holding none of the asked keys carves nothing instead of failing, and the row comes back as the
 *  inserting executor reads it. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_vram_indexes_get_cold(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> writer {placement}, reader {placement};
    if (failed(writer.status) || failed(reader.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement)) && succeeded(place_shards(indexes, placement)));
    expect_true(succeeded(indexes.reserve({1}).status));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(1, dimensions, stream_key(seed, "indexes_get_cold"));
    std::array<key_t, 1> const keys {dense_key<key_t>(1)};
    expect_true(succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), writer.executor).status));
    expect_equal(indexes.size(), 1u);

    std::size_t const row_values = divide_round_up<std::size_t>(dimensions, nk::dimensions_per_value<scalar_t>());
    std::size_t const row_bytes = row_values * sizeof(scalar_t);
    nk::vector<scalar_t> cold_row = make_vector<scalar_t>(dimensions), warm_row = make_vector<scalar_t>(dimensions);
    std::array<std::size_t, 1> cold_entries {}, warm_entries {};
    auto get_row = [&](nk::vector<scalar_t>& row, std::array<std::size_t, 1>& entries,
                       typename indexes_type_::executor_t& executor) noexcept {
        return indexes.get(
            std::span<key_t const> {keys},
            rows_span_t {reinterpret_cast<byte_t*>(row.data()), 1, dimensions, row_bytes, dtype_of<scalar_t>()},
            std::span {entries}, 1, executor);
    };
    expect_equal(get_row(cold_row, cold_entries, reader.executor), status_code_t::success_k);
    expect_equal(get_row(warm_row, warm_entries, writer.executor), status_code_t::success_k);
    expect_true(cold_entries[0] == 1 && warm_entries[0] == 1);
    expect_true(std::memcmp(cold_row.data(), warm_row.data(), row_bytes) == 0);
}

/** A RAM group's monolith loaded into the VRAM collection and saved again comes back byte for byte,
 *  norms included, whether loaded by path, which reads through cuFile where it can, or from its
 *  mapped bytes, and the NVMe tier views the VRAM save like the RAM one. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_vram_indexes_nvme_round_trip(u64_t seed, typename indexes_type_::executor_t& placement) {
    using key_t = typename indexes_type_::key_t;
    using nvme_t = nvme_indexes_for<key_t, indexes_type_::vector_dtype_k, nk_cap_serial_k,
                                    blocking_files_engine<fu::standard_yield_t>>;
    using scalar_t = typename nvme_t::scalar_t;
    std::size_t const rows_count = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    nvme_reference<nvme_t> reference {"vram_nvme_roundtrip", kernel_, rows_count, dimensions, seed};
    expect_true(succeeded(
        reference.whole().save(reference.checkpoint.monolith.data(), save_config_t {}, reference.builder.executor)));
    auto [ram_file, ram_status] = mapped_file_t::read(reference.checkpoint.monolith.data());
    dense_indexes_executor<indexes_type_> driver {placement};
    if (!expect_true(succeeded(ram_status)) || !expect_true(succeeded(driver.status)) || !reference.held()) return;
    bytes_view_t const ram_bytes = ram_file.view();
    checkpoint_files_t const resaved {"vram_nvme_resaved"};
    for (std::size_t source = 0; source != 2; ++source) {
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(indexes, placement)));
        expect_true(
            succeeded(source ? indexes.load(ram_bytes, load_config_t {}, driver.executor)
                             : indexes.load(reference.checkpoint.monolith.data(), load_config_t {}, driver.executor)));
        expect_true(succeeded(
            indexes.save(resaved.monolith.data(), dense_indexes_save_config<indexes_type_>(), driver.executor)));
        auto [vram_file, vram_status] = mapped_file_t::read(resaved.monolith.data());
        if (!expect_true(succeeded(vram_status))) return;
        bytes_view_t const vram_bytes = vram_file.view();
        expect_true(ram_bytes.size() == vram_bytes.size() &&
                    std::memcmp(ram_bytes.data(), vram_bytes.data(), ram_bytes.size()) == 0);
    }

    nvme_t viewed {forkunion_allocator_t {no_memory_domain_k}};
    expect_equal(viewed.view(resaved.monolith.data()), status_code_t::success_k);
    expect_equal(viewed.size(), rows_count);
}

/** Device shards reshard into one host shard: every entry crosses with its key exactly once, the
 *  source ends empty, nodes of every source shard link into the others', and the merged graph walks
 *  within noise of the recall at ten one built from every entry at once reaches. Past
 *  @c connectivity+1 tiny shards the fold splits them into bins, each member keeping a slot of a
 *  node from every other, and the run stops once the re-selection finished and resumes. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_vram_indexes_reshard(u64_t seed, typename indexes_type_::executor_t& placement,
                                     std::size_t shards_count) {
    using key_t = typename indexes_type_::key_t;
    using scalar_t = typename indexes_type_::scalar_t;
    using nvme_t = nvme_indexes_for<key_t, indexes_type_::vector_dtype_k, nk_cap_serial_k,
                                    blocking_files_engine<fu::standard_yield_t>>;
    using ram_t = typename nvme_reference<nvme_t>::ram_t;
    using node_id_t = typename ram_t::index_t::node_id_t;
    constexpr std::size_t samples_k = 32;
    constexpr std::size_t noise_percent_k = static_cast<std::size_t>(rewrite_recall_margin_k * 100);
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement};
    dense_indexes_executor<ram_t> host;
    if (failed(driver.status) || failed(host.status)) return void(expect_true(false));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_reshard"));
    nk::matrix<scalar_t> const queries = make_vectors<scalar_t>(samples_k, dimensions,
                                                                stream_key(seed, "indexes_reshard", 1));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ source {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    for (std::size_t shard = 0; shard != shards_count; ++shard) expect_true(succeeded(place_shards(source, placement)));
    expect_equal(source.reserve({rows}).status, status_code_t::success_k);
    expect_true(
        succeeded(source.insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), driver.executor).status));
    // The source shards are appended in their order, so each one's nodes follow the counts of the
    // shards before it.
    std::vector<std::size_t> shard_ends(shards_count);
    for (std::size_t shard = 0, end = 0; shard != shards_count; ++shard)
        shard_ends[shard] = end += source.shard(shard).size();
    ram_t target {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(target.emplace(no_memory_domain_k)));

    indexes_type_* const sources[1] = {&source};
    std::span<indexes_type_* const> const resharded {sources};
    // The run stops once the re-selection finished, moving nothing, and the next call resumes it.
    auto resumed = [&] {
        auto [plan, plan_status] = plan_reshard(resharded, target, driver.executor, host.executor);
        expect_equal(plan_status, status_code_t::success_k);
        auto cancel = [](std::size_t, std::size_t) noexcept { return continuation_t::cancel_k; };
        expect_equal(plan.run(driver.executor, host.executor, to_writable_span(host.traversals), cancel),
                     status_code_t::cancelled_k);
        expect_equal(source.size(), rows);
        expect_equal(plan.run(driver.executor, host.executor, to_writable_span(host.traversals)),
                     status_code_t::success_k);
        return plan.finish(host.executor, to_writable_span(host.traversals));
    };
    auto const [result, status] = shards_count <= connectivity_16_k + 1
                                      ? reshard(resharded, target, driver.executor, host.executor)
                                      : resumed();
    expect_equal(status, status_code_t::success_k);
    expect_true(result.source_shards == shards_count && result.target_shards == 1 && result.moved_entries == rows);
    expect_true(result.graph.reselected_nodes > 0);
    expect_equal(source.size(), 0u);
    expect_equal(target.size(), rows);
    expect_true(target.dimensions() == dimensions && target.connectivity() == connectivity_16_k);
    std::size_t miscounted = 0;
    for (key_t const key : keys) miscounted += target.count(key) != 1;
    expect_equal(miscounted, 0u);

    // A neighbor list holding a node of another source shard is a link the fold made, and none
    // points past the nodes.
    ram_cursor<typename ram_t::index_t, synchronization_t::immutable_k, forkunion_allocator_t> cursor {target.shard(0)};
    expect_equal(cursor.reserve(forkunion_allocator_t {no_memory_domain_k}), status_code_t::success_k);
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
    expect_true(crossing > 0);
    expect_equal(stray, 0u);

    ram_t built {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    dense_indexes_build(built, host.executor, vectors, keys);
    std::size_t const baseline = dense_recall_at_ten(built.shard(0), 0, queries.view());
    std::size_t const merged = dense_recall_at_ten(target.shard(0), 0, queries.view());
    expect_true(merged * 100 + noise_percent_k * samples_k * 10 >= baseline * 100);
}

/** A two-shard index splits into two targets of two shards each: the sizes add up, every key
 *  counts once across the targets, the source ends empty, and the targets' merged recall at
 *  ten against an exact reference stays within noise of the source's. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_vram_indexes_cut(u64_t seed, typename indexes_type_::executor_t& placement) {
    using key_t = typename indexes_type_::key_t;
    using scalar_t = typename indexes_type_::scalar_t;
    using distance_t = typename indexes_type_::distance_t;
    using nvme_t = nvme_indexes_for<key_t, indexes_type_::vector_dtype_k, nk_cap_serial_k,
                                    blocking_files_engine<fu::standard_yield_t>>;
    using ram_t = typename nvme_reference<nvme_t>::ram_t;
    using reference_t = typename ram_t::index_t;
    constexpr std::size_t samples_k = 32;
    constexpr std::size_t wanted_k = 10;
    constexpr std::size_t noise_percent_k = static_cast<std::size_t>(rewrite_recall_margin_k * 100);
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement};
    dense_indexes_executor<ram_t> host;
    if (failed(driver.status) || failed(host.status)) return void(expect_true(false));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_split"));
    nk::matrix<scalar_t> const queries = make_vectors<scalar_t>(samples_k, dimensions,
                                                                stream_key(seed, "indexes_split", 1));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ source {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ first {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ second {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    for (indexes_type_* placed : {&source, &first, &second})
        expect_true(succeeded(place_shards(*placed, placement)) && succeeded(place_shards(*placed, placement)));
    expect_equal(source.reserve({rows}).status, status_code_t::success_k);
    expect_true(
        succeeded(source.insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), driver.executor).status));

    // The exact tenth distance of every query over all rows, from a host reference's scan.
    ram_t built {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    dense_indexes_build(built, host.executor, vectors, keys);
    reference_t& reference = built.shard(0);
    typename reference_t::traversal_context_t context = dense_make_context<reference_t>(0);
    typename reference_t::search_config_t const scan {
        .expansion_factor = static_cast<typename reference_t::candidates_count_t>(wanted_k),
        .traversal = traversal_t::scan_k};
    nk::vector<f64_t> tenths = make_vector<f64_t>(samples_k);
    for (std::size_t query = 0; query != samples_k; ++query)
        if (succeeded(reference.search(queries.view().row(query).as_vector(), scan, context).status) &&
            !context.results.empty())
            tenths[query] = context.results[context.results.size() - 1].distance;

    // The ten nearest the walks of every index in `parts` return, merged, within each tenth.
    auto recall = [&](std::initializer_list<indexes_type_*> parts) {
        std::size_t found = 0;
        nk::vector<distance_t> merged_distances = make_vector<distance_t>(parts.size() * wanted_k);
        nk::vector<key_t> found_keys = make_vector<key_t>(wanted_k);
        nk::vector<distance_t> found_distances = make_vector<distance_t>(wanted_k);
        for (std::size_t query = 0; query != samples_k; ++query) {
            std::size_t merged_count = 0;
            for (indexes_type_* part : parts) {
                auto const [result, status] = part->search(queries.view().row(query).as_vector(),
                                                           to_writable_span(found_keys),
                                                           to_writable_span(found_distances), {}, driver.executor);
                if (!succeeded(status)) continue;
                std::memcpy(merged_distances.data() + merged_count, found_distances.data(),
                            result.count * sizeof(distance_t));
                merged_count += result.count;
            }
            std::span<distance_t> const merged = to_writable_span(merged_distances).first(merged_count);
            std::sort(merged.begin(), merged.end());
            f64_t const tenth = tenths[query];
            for (std::size_t rank = 0; rank != min_of(wanted_k, merged.size()); ++rank)
                found += merged[rank] <= tenth + 1e-3 * max_of(std::fabs(tenth), 1.0);
        }
        return found;
    };
    std::size_t const source_recall = recall({&source});

    indexes_type_* const sources[1] = {&source};
    indexes_type_* const refused[2] = {&first, &source};
    expect_equal(plan_reshard(std::span<indexes_type_* const> {sources}, std::span<indexes_type_* const> {refused},
                              driver.executor)
                     .status,
                 status_code_t::config_mismatch_k);
    indexes_type_* const targets[2] = {&first, &second};
    reshard_config_t config;
    config.seed = seed;
    auto [plan, plan_status] = plan_reshard(std::span<indexes_type_* const> {sources},
                                            std::span<indexes_type_* const> {targets}, driver.executor, config);
    expect_equal(plan_status, status_code_t::success_k);
    expect_equal(plan.run(driver.executor), status_code_t::success_k);
    // A second round restarts at the first target and runs on every one, never past the last.
    expect_true(plan.phase() == phase_t::done_k);
    plan.next_round();
    expect_true(plan.phase() != phase_t::done_k);
    expect_equal(plan.run(driver.executor), status_code_t::success_k);
    expect_true(plan.phase() == phase_t::done_k);
    auto const [cut, cut_status] = plan.finish(driver.executor);
    expect_equal(cut_status, status_code_t::success_k);
    expect_equal(cut.moved_entries, rows);
    expect_equal(cut.graph.reachability.audited_nodes, rows);
    expect_equal(source.size(), 0u);
    expect_equal(first.size() + second.size(), rows);
    expect_true(first.size() && second.size());

    nk::vector<std::size_t> in_first = make_vector<std::size_t>(rows), in_second = make_vector<std::size_t>(rows);
    expect_true(succeeded(first.count(to_span(keys), to_writable_span(in_first), driver.executor)) &&
                succeeded(second.count(to_span(keys), to_writable_span(in_second), driver.executor)));
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != rows; ++row) miscounted += in_first[row] + in_second[row] != 1;
    expect_equal(miscounted, 0u);

    std::size_t const cut_recall = recall({&first, &second});
    expect_true(cut_recall * 100 + noise_percent_k * samples_k * wanted_k >= source_recall * 100);
}

/** A reshard of several sources into targets naming one of them, or one naming a shard capacity, is
 *  refused and leaves the source whole: its size and the count of every key stay as inserted. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_vram_indexes_cut_refused(u64_t seed, typename indexes_type_::executor_t& placement) {
    using key_t = typename indexes_type_::key_t;
    using scalar_t = typename indexes_type_::scalar_t;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return void(expect_true(false));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows, dimensions,
                                                                stream_key(seed, "indexes_split_refused"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);

    indexes_type_ source {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ lone {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ first {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    for (indexes_type_* placed : {&source, &lone, &first})
        expect_true(succeeded(place_shards(*placed, placement)) && succeeded(place_shards(*placed, placement)));
    expect_equal(source.reserve({rows}).status, status_code_t::success_k);
    expect_true(
        succeeded(source.insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), driver.executor).status));

    indexes_type_* const sources[2] = {&source, &lone};
    indexes_type_* const targets[2] = {&first, &lone};
    expect_equal(plan_reshard(std::span<indexes_type_* const> {sources}, std::span<indexes_type_* const> {targets},
                              driver.executor)
                     .status,
                 status_code_t::config_mismatch_k);
    reshard_config_t capped;
    capped.shard_capacity = rows;
    expect_equal(plan_reshard(std::span<indexes_type_* const> {sources}.first(1),
                              std::span<indexes_type_* const> {sources}.first(1), driver.executor, capped)
                     .status,
                 status_code_t::unsupported_k);
    expect_equal(source.size(), rows);
    std::size_t miscounted = 0;
    for (key_t const& key : keys) miscounted += source.count(key) != 1;
    expect_equal(miscounted, 0u);
}

/** Two shards walk as one: a cursor yields every entry once as stored, and from the first key
 *  breadth-first that key first and each reached entry once, every link a key the index holds once
 *  and the offsets closing each entry's run in order; a shard copies its neighbor rows out only
 *  whole and inside its watermark; and a cursor opened before a clear answers @c contended_k after
 *  it, reading nothing. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_vram_indexes_cursor(u64_t seed, typename indexes_type_::executor_t& placement) {
    using key_t = typename indexes_type_::key_t;
    using scalar_t = typename indexes_type_::scalar_t;
    using cursor_t = typename indexes_type_::cursor_t;
    using node_id_t = typename indexes_type_::index_t::node_id_t;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return void(expect_true(false));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_cursor"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes, placement)) && succeeded(place_shards(indexes, placement)));
    expect_equal(indexes.reserve({rows}).status, status_code_t::success_k);
    expect_true(
        succeeded(indexes.insert(std::span<key_t const> {keys.data(), rows}, vectors.view(), driver.executor).status));
    expect_true(indexes.shard(0).size() && indexes.shard(1).size());

    expect_sharded_cursor(indexes, keys[0], rows, connectivity_16_k + 1);

    typename indexes_type_::index_t& shard = indexes.shard(0);
    std::size_t const row_ids = shard.connectivity();
    nk::vector<node_id_t> staged = make_vector<node_id_t>(row_ids + 1);
    std::span<node_id_t> const whole = to_writable_span(staged).first(row_ids);
    expect_equal(shard.neighbors(0, whole), status_code_t::success_k);
    expect_equal(shard.neighbors(0, to_writable_span(staged)), status_code_t::config_mismatch_k);
    expect_equal(shard.neighbors(static_cast<node_id_t>(shard.size()), whole), status_code_t::config_mismatch_k);

    cursor_t cursor {forkunion_allocator_t {no_memory_domain_k}};
    expect_equal(cursor.open(indexes, {}), status_code_t::success_k);
    nk::vector<key_t> read_keys = make_vector<key_t>(4);
    auto const [before, before_status] = cursor.pull(indexes, to_writable_span(read_keys), nullptr, nullptr);
    expect_true(succeeded(before_status) && before == read_keys.size());
    expect_equal(indexes.clear(), status_code_t::success_k);
    auto const [after, after_status] = cursor.pull(indexes, to_writable_span(read_keys), nullptr, nullptr);
    expect_true(after_status == status_code_t::contended_k && after == 0);
}

/** A vendor this build leaves out composes nothing, so its shard instantiates empty. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
void test_vram_indexes_configuration(u64_t, executor_type_&) {}

/** Past @c shard_capacity, entries open generations of device shards on the same placements: one
 *  batch fills as many as it needs with every key resolving once, and a batch crossing the bound
 *  tops the open generation up and goes in whole, its newest entries reachable by a search. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
void test_dense_vram_indexes_generations(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t bound = 1000, rows = 5000;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return void(expect_true(false));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows, dimensions,
                                                                stream_key(seed, "indexes_generations"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
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
    expect_true(bounded(filled));
    expect_equal(insert(filled, 0, rows), status_code_t::success_k);
    expect_equal(filled.size(), rows);
    expect_true((shard_sizes(filled) == std::vector<std::size_t> {1000, 1000, 1000, 1000, 500, 500}));
    expect_equal(miscounted(filled, rows), 0u);

    // 750 entries a shard, then a batch of 1000 tops both up to the bound and opens a generation
    // for the other 500, whole, and a search reaches the new generation's entries.
    indexes_type_ crossed {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(bounded(crossed));
    expect_equal(insert(crossed, 0, 1500), status_code_t::success_k);
    expect_equal(crossed.shards_count(), 2u);
    expect_equal(insert(crossed, 1500, 1000), status_code_t::success_k);
    expect_true((shard_sizes(crossed) == std::vector<std::size_t> {1000, 1000, 250, 250}));
    expect_equal(miscounted(crossed, 2500), 0u);
    if constexpr (kernel_ != nk_kernel_dot_k) {
        nk::vector<key_t> found_keys = make_vector<key_t>(8);
        nk::vector<typename indexes_type_::distance_t> found_distances =
            make_vector<typename indexes_type_::distance_t>(8);
        auto const [result, status] = crossed.search(vectors.view().row(2499).as_vector(), to_writable_span(found_keys),
                                                     to_writable_span(found_distances), {}, driver.executor);
        bool reached = false;
        for (std::size_t match = 0; match != result.count; ++match) reached |= found_keys[match] == keys[2499];
        expect_true(succeeded(status) && reached);
    }
}

/** Composes one (dtype, kernel) cell's sharded suite over the collection of @p executor 's vendor
 *  on its device. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
    requires vram_compiled<executor_type_::vendor_k>
void test_vram_indexes_configuration(u64_t seed, executor_type_& executor) {
    using cell_t = vram_cell<executor_type_::vendor_k, dtype_, kernel_>;
    using indexes_t = typename cell_t::indexes_t;

    // The suite is composed here, not reflected: every call below is a capability claim
    // this sharded, executor-driven category makes about itself.
    test_dense_indexes_guards<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_duplicate_keys<indexes_t, kernel_>(seed, executor);
    dense_expect_duplicates_build<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_save_load<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_part_load<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_buffer_round_trip<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_cancelled_load<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_concurrent_add<indexes_t, kernel_>(seed, executor);
    test_dense_vram_indexes_search_race<indexes_t, kernel_>(seed, executor);
    if constexpr (executor_type_::vendor_k == gpu_vendor_t::nvidia_k)
        test_dense_indexes_scan<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_planned_insert<indexes_t, kernel_>(seed, executor);
    // Apple's tower opens no generations of shards.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        test_dense_vram_indexes_generations<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_refine<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_gather<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_gather_many<indexes_t, kernel_>(seed, executor);
    // Apple's cut is unverified until an Apple run takes this cell.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        test_dense_vram_indexes_cut<indexes_t, kernel_>(seed, executor);
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        test_dense_vram_indexes_cut_refused<indexes_t, kernel_>(seed, executor);
    test_dense_vram_indexes_cursor<indexes_t, kernel_>(seed, executor);
    // Apple's tower has no reshard into RAM.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        for (std::size_t const shards : {std::size_t {2}, std::size_t {2} * (connectivity_16_k + 1)})
            test_dense_vram_indexes_reshard<indexes_t, kernel_>(seed, executor, shards);
    test_dense_indexes_products<indexes_t, kernel_>(seed, executor);
    test_dense_indexes_maxsim<indexes_t, kernel_>(seed, executor);
    test_dense_vram_indexes_indexer_search<indexes_t, kernel_>(seed, executor);
    // Apple merges every ranking on the host, so its two merges are one path.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        test_dense_vram_indexes_device_outputs<indexes_t, kernel_>(seed, executor);
    test_dense_vram_indexes_get_cold<indexes_t, kernel_>(seed, executor);
    test_dense_vram_indexes_nvme_round_trip<indexes_t, kernel_>(seed, executor);
}

/** The sharded VRAM cases its translation unit defines once per vendor, for its executor, which
 *  each carried vendor's entry runs over the executor it opened. */
template <typename executor_type_>
void run_vram_indexes_cases(executor_type_& executor);

} // namespace unum::usearch::test::dense

/*  The residency forms specialize templates of the enclosing @c test namespace, which only a
 *  scope enclosing it may do, so they follow the namespace at file scope and qualify what they name. */

/** The VRAM form: one stream or queue of its own on the device of @p placement, the executor the
 *  vendor's entry opened. */
template <us::test::dense::vram_collection indexes_type_>
struct us::test::dense_indexes_executor<indexes_type_> {
    using executor_t = typename indexes_type_::executor_t;

    executor_t executor;
    util::status_t status = util::status_code_t::uninitialized_k;

    explicit dense_indexes_executor(executor_t const& placement) noexcept {
        int const device = placement.device(0);
        auto [executor, spawn_status] = spawn_(std::span<int const> {&device, 1});
        status = spawn_status;
        this->executor = std::move(executor);
    }
    dense_indexes_executor(dense_indexes_executor const&) = delete;
    dense_indexes_executor& operator=(dense_indexes_executor const&) = delete;

    /** Opens @p devices through the vendor's driver table the harness loaded. */
    static util::expected<executor_t> spawn_(std::span<int const> devices) noexcept {
        util::gpu_drivers_t& drivers = us::test::gpu_drivers();
        if constexpr (executor_t::vendor_k == util::gpu_vendor_t::nvidia_k)
            return executor_t::spawn(drivers.cuda, devices);
        else if constexpr (executor_t::vendor_k == util::gpu_vendor_t::amd_k)
            return executor_t::spawn(drivers.hip, devices);
        else return executor_t::spawn(drivers.metal_contexts, devices);
    }

    /** Places the shards one sharded scenario drives: one VRAM shard on the device of
     *  @p placement; each further call places one more beside it. */
    static util::status_t place_shards(indexes_type_& indexes, executor_t& placement) noexcept {
        return indexes.emplace(placement.stream(0));
    }

    /** Runs @p verb over a copy of the host @p tape on the executor's device, where the tape verbs
     *  write, its first offset uploaded, and copies the written keys, scores and offsets home. */
    template <typename key_type_, typename score_type_, typename verb_type_>
    util::expected<typename indexes_type_::search_result_t> on_device(
        us::dense::products_tape<key_type_, score_type_> tape, verb_type_&& verb) noexcept {
        using util::status_t;
        typename executor_t::device_arena_t arena {executor.stream(0)};
        key_type_* const keys = arena.template allocate<key_type_>(tape.keys.size());
        score_type_* const scores = arena.template allocate<score_type_>(tape.scores.size());
        std::size_t* const offsets = arena.template allocate<std::size_t>(tape.offsets.size());
        if (!keys || !scores || !offsets) return {{}, status_code_t::out_of_memory_k};
        if (failed(arena.upload(offsets, tape.offsets.data(), sizeof(std::size_t))) || failed(arena.wait()))
            return {{}, status_code_t::device_k};
        auto const [run,
                    run_status] = verb(us::dense::products_tape<key_type_, score_type_> {{keys, tape.keys.size()},
                                                                                         {scores, tape.scores.size()},
                                                                                         {offsets, tape.offsets.size()},
                                                                                         tape.stride});
        if (!us::test::dense::dense_download<executor_t::vendor_k>(tape.keys.data(), keys, tape.keys.size_bytes()) ||
            !us::test::dense::dense_download<executor_t::vendor_k>(tape.scores.data(), scores,
                                                                   tape.scores.size_bytes()) ||
            !us::test::dense::dense_download<executor_t::vendor_k>(tape.offsets.data(), offsets,
                                                                   tape.offsets.size_bytes()))
            return {{}, status_code_t::device_k};
        return {run, run_status};
    }
};

/** The VRAM form: the one-row verbs take the executor itself, as no single context can
 *  walk every device. */
template <us::test::dense::vram_collection indexes_type_>
struct us::test::dense_indexes_row_driver<indexes_type_> {
    typename indexes_type_::executor_t& executor;
    util::status_t status = util::status_code_t::success_k;

    dense_indexes_row_driver(indexes_type_&, us::test::dense_indexes_executor<indexes_type_>& driver) noexcept
        : executor(driver.executor) {}
    typename indexes_type_::executor_t& handle() noexcept { return executor; }
};
