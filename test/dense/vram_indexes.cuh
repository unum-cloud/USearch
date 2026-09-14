/**
 *  @brief The CUDA-side suite composer for @c dense_vram_indexes - the sharded index over
 *      one shard per device - shared by its translation unit.
 *  @author Ash Vardanian
 *  @file test/dense/vram_indexes.cuh
 *  @date September 1, 2026
 *
 *  The scenarios live in @c dense.cuh, templated over the sharded-index type; the VRAM forms
 *  of the placement and executor helpers live here, since they name the CUDA executor.
 */
#pragma once

#include "usearch/dense/vram_indexes.cuh"

#include "dense.cuh"

namespace unum {
namespace usearch {

/** @brief Whether a sharded index runs on CUDA streams - the VRAM residency. */
template <typename indexes_type_>
concept vram_indexes = std::is_same_v<typename indexes_type_::executor_t, cuda_executor_t>;

/** @brief The VRAM form: one stream on device zero. */
template <vram_indexes indexes_type_>
struct dense_indexes_executor<indexes_type_> {
    using executor_t = cuda_executor_t;

    executor_t executor;
    status_t status = status_t::uninitialized_k;

    dense_indexes_executor() noexcept {
        int const device = 0;
        auto [executor, spawn_status] = executor_t::spawn(std::span<int const> {&device, 1});
        status = spawn_status;
        this->executor = std::move(executor);
    }
    dense_indexes_executor(dense_indexes_executor const&) = delete;
    dense_indexes_executor& operator=(dense_indexes_executor const&) = delete;
};

/** @brief Places the shards one sharded scenario drives: one VRAM shard on device zero, since
 *      the index refuses two on one device. */
template <vram_indexes indexes_type_>
status_t place_shards(indexes_type_& indexes) noexcept {
    return indexes.emplace(0);
}

/** @brief The VRAM form: the one-row verbs take the executor itself, as no single context can
 *      walk every device. */
template <vram_indexes indexes_type_>
struct dense_indexes_row_driver<indexes_type_> {
    cuda_executor_t& executor;
    status_t status = status_t::success_k;

    dense_indexes_row_driver(indexes_type_&, dense_indexes_executor<indexes_type_>& driver) noexcept
        : executor(driver.executor) {}
    cuda_executor_t& handle() noexcept { return executor; }
};

/**
 *  @brief The VRAM reading of the search race: the index admits no search during a mutation,
 *      so the exclusion is the executor's - a search on a claimed executor is refused, and the
 *      reader walks the published rows only once the writer has returned.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
    requires vram_indexes<indexes_type_>
std::size_t test_dense_indexes_search_race() {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;

    std::size_t const nodes_count = 4 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes {forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions}).status));

    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;

    auto vectors = make_vectors<scalar_t>(nodes_count, dimensions, dense_seed_indexes_race_k);
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

/** @brief Composes one (dtype, kernel) cell's sharded suite over device zero, skipping it
 *      where no device answers. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
std::size_t test_cuda_indexes_configuration(char const* name) {
    if (failed(cuda_device_specs_t::query(0).status)) return report(name, "no CUDA device reachable");

    using indexes_t = dense_vram_indexes<u32_t, dtype_, kernel_, cuda_arch_blackwell_k>;

    // The suite is composed here, not reflected: every call below is a capability claim
    // this sharded, executor-driven category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_indexes_guards<indexes_t>();
    failures_count += test_dense_indexes_duplicate_keys<indexes_t>();
    failures_count += test_dense_indexes_save_load<indexes_t, kernel_>();
    failures_count += test_dense_indexes_split_load<indexes_t, kernel_>();
    failures_count += test_dense_indexes_buffer_roundtrip<indexes_t, kernel_>();
    failures_count += test_dense_indexes_cancelled_load<indexes_t, kernel_>();
    failures_count += test_dense_indexes_concurrent_add<indexes_t, kernel_>();
    failures_count += test_dense_indexes_search_race<indexes_t, kernel_>();
    return report(name, failures_count);
}

} // namespace usearch
} // namespace unum

#pragma region Suite Entry Points

/** @brief The sharded CUDA suite entry its translation unit defines and the suite's @c main
 *      composes explicitly; returns the suite's failure count. */
std::size_t test_dense_indexes_cuda();

#pragma endregion Suite Entry Points
