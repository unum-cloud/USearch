/**
 *  @file test/dense/vram_indexes.cuh
 *  @author Ash Vardanian
 *  @date September 1, 2026
 *  @brief The suite composer for the sharded VRAM index - one shard per device - shared by its
 *      translation unit and instantiated per vendor, as @c vram_index.cuh is.
 *
 *  The scenarios live in @c dense.cuh, templated over the sharded-index type; the VRAM forms of
 *  the placement and executor helpers live here, over whichever executor the cell's index names.
 */
#pragma once

#include "dense.cuh"
#include "vram_index.cuh" // `vram_cell`, `selected_device`

namespace unum::usearch::test::dense {

/** Whether a sharded index places its shards on GPUs - the VRAM residency, whichever vendor's. */
template <typename indexes_type_>
concept vram_indexes = requires { indexes_type_::vendor_k; };

/** The VRAM reading of the search race: the index admits no search during a mutation,
 *  so the exclusion is the executor's - a search on a claimed executor is refused, and the
 *  reader walks the published rows only once the writer has returned. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_search_race(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;

    std::size_t const nodes_count = 4 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions}).status));

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;

    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_race"));
    std::vector<key_t> keys(nodes_count);
    for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = static_cast<key_t>(1 + row);
    std::array<key_t, 8> found_keys {};
    std::array<typename indexes_type_::distance_t, 8> found_distances {};

    // A claimed executor refuses the search outright, before any scratch is carved.
    {
        auto [claim, claim_status] = executor.claim();
        failures += expect(succeeded(claim_status));
        failures += expect_eq(indexes
                                  .search(vectors.view().row(0).as_vector(), executor, {},
                                          std::span<key_t> {found_keys}, std::span {found_distances})
                                  .status,
                              status_t::contended_k);
    }

    auto [insert_result, insert_status] = indexes.insert(std::span<key_t const> {keys.data(), nodes_count},
                                                         vectors.view(), executor);
    failures += expect_eq(insert_status, status_t::success_k);
    failures += expect_eq(indexes.size(), nodes_count);

    // Every published row resolves to a key inside the assigned range - never the key column's zero-fill.
    std::size_t mislabeled = 0;
    for (std::size_t searches = 0; searches != dense_reader_minimal_searches_k; ++searches) {
        auto const [search_result, search_status] = indexes.search(
            vectors.view().row(searches % nodes_count).as_vector(), executor, {}, std::span<key_t> {found_keys},
            std::span {found_distances});
        failures += expect_eq(search_status, status_t::success_k);
        for (std::size_t match = 0; match != search_result.count; ++match)
            mislabeled += found_keys[match] < key_t {1} || found_keys[match] > static_cast<key_t>(nodes_count);
    }
    failures += expect_eq(mislabeled, 0u);
    return failures;
}

/** The planned insertion: launched rows stay out of the size, the locators and a search until @c finish publishes
 *  every one, a plan short of rows refuses to finish and stays whole, its gate turns a second batch away, and a plan
 *  dying unfinished publishes nothing and leaves its shard refusing every later batch. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_indexes_planned_insert(u64_t seed) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, half = rows_count / 2;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> writer, reader;
    if (failed(writer.status) || failed(reader.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(writer.place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows_count, connectivity_16_k, dimensions}).status));
    auto const vectors = make_vectors<scalar_t>(2 * rows_count, dimensions, stream_key(seed, "indexes_planned"));
    std::vector<key_t> keys(2 * rows_count);
    for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = static_cast<key_t>(1 + row);
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
                                                                   reader.executor, {}, std::span<key_t> {found_keys},
                                                                   std::span {found_distances});
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
 *  under the same heads with a pool as wide as the index, so the merged top of every query is the oracle's. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_indexer_search(u64_t seed) {
    using scalar_t = typename indexes_type_::scalar_t;
    if constexpr (kernel_ != nk_kernel_dot_k || nk::is_integral_dtype<scalar_t>()) return 0;
    else {
        using key_t = typename indexes_type_::key_t;
        constexpr std::size_t rows_count = dense_thread_nodes_k, queries_count = 4, wanted = 16;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::size_t failures = 0;

        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return failures + expect(false);
        indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(driver.place_shards(indexes)));
        failures += expect(succeeded(indexes.reserve({rows_count, connectivity_16_k, dimensions}).status));
        auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "indexes_indexer"));
        std::array<key_t, rows_count> keys;
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = static_cast<key_t>(row);
        failures += expect(
            succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));

        auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
        dense_indexer_inputs<scalar_t> const inputs = dense_make_indexer_inputs(vectors.view(), queries_count);
        typename indexes_type_::search_config_t const exhaustive {rows_count, wanted};
        auto found_keys = nk::matrix<key_t>::zeros({queries_count, wanted}).value;
        auto found_distances = nk::matrix<dense_distance_t>::zeros({queries_count, wanted}).value;
        std::array<std::size_t, queries_count> counts;
        failures += expect(succeeded(indexes
                                         .search(queries, inputs.heads(), driver.executor, exhaustive,
                                                 found_keys.span(), found_distances.span(), std::span {counts})
                                         .status));
        nk::matrix_view<key_t> const ranked_keys = found_keys.view();
        nk::matrix_view<dense_distance_t> const ranked_distances = found_distances.view();
        for (std::size_t query = 0; query != queries_count; ++query) {
            nk::range_t const kept = nk::range(std::size_t {0}, counts[query]);
            failures += dense_expect_indexer_matches(inputs, query, ranked_keys(query, kept, nk::slice).as_vector(),
                                                     ranked_distances(query, kept, nk::slice).as_vector());
        }
        return failures;
    }
}

/** Composes one (dtype, kernel) cell's sharded suite over @p vendor_ 's collection on the device
 *  under test, reported under @p name prefixed by the vendor's toolkit. */
template <gpu_vendor_t vendor_, nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
std::size_t test_vram_indexes_configuration(char const* name, u64_t seed) {
    using cell_t = vram_cell<vendor_, dtype_, kernel_>;
    using indexes_t = typename cell_t::indexes_t;
    std::string const vendor_name = fmt::format("{}-{}", cell_t::vendor_prefix_k, name);

    // The suite is composed here, not reflected: every call below is a capability claim
    // this sharded, executor-driven category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_indexes_guards<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_duplicate_keys<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_save_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_split_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_buffer_roundtrip<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_cancelled_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_concurrent_add<indexes_t, kernel_>(seed);
    failures_count += test_dense_vram_indexes_search_race<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_planned_insert<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_refine<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_products<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_maxsim<indexes_t, kernel_>(seed);
    failures_count += test_dense_vram_indexes_indexer_search<indexes_t, kernel_>(seed);
    return report(vendor_name.c_str(), failures_count);
}

/** The sharded VRAM suite entry its translation unit defines for the vendor of its object library
 *  and that vendor's entry composes explicitly; returns the suite's failure count. */
template <gpu_vendor_t vendor_>
std::size_t check_vram_indexes(settings_t const& settings);

} // namespace unum::usearch::test::dense

/*  The residency forms specialize templates of the enclosing @c test namespace, which only a
 *  scope enclosing it may do, so they follow the namespace at file scope and qualify what they name. */

/** The VRAM form: one stream or queue on the device under test. */
template <us::test::dense::vram_indexes indexes_type_>
struct us::test::dense_indexes_executor<indexes_type_> {
    using executor_t = typename indexes_type_::executor_t;

    executor_t executor;
    us::status_t status = us::status_t::uninitialized_k;

    dense_indexes_executor() noexcept {
        int const device = us::test::dense::selected_device;
        auto [executor, spawn_status] = executor_t::spawn(std::span<int const> {&device, 1});
        status = spawn_status;
        this->executor = std::move(executor);
    }
    dense_indexes_executor(dense_indexes_executor const&) = delete;
    dense_indexes_executor& operator=(dense_indexes_executor const&) = delete;

    /** Places the shards one sharded scenario drives: one VRAM shard on the device under test,
     *  since the index refuses two on one device. */
    static us::status_t place_shards(indexes_type_& indexes) noexcept {
        return indexes.emplace(us::test::dense::selected_device);
    }
};

/** The VRAM form: the one-row verbs take the executor itself, as no single context can
 *  walk every device. */
template <us::test::dense::vram_indexes indexes_type_>
struct us::test::dense_indexes_row_driver<indexes_type_> {
    typename indexes_type_::executor_t& executor;
    us::status_t status = us::status_t::success_k;

    dense_indexes_row_driver(indexes_type_&, us::test::dense_indexes_executor<indexes_type_>& driver) noexcept
        : executor(driver.executor) {}
    typename indexes_type_::executor_t& handle() noexcept { return executor; }
};
