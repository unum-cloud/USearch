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
#include "nvme_indexes.hpp" // `nvme_reference`, `nvme_checkpoint_t` - the round trip through storage
#include "vram_index.cuh"   // the vendor's `vram_cell`

namespace unum::usearch::test::dense {

/** Whether a sharded index places its shards on GPUs - the VRAM residency, whichever vendor's. */
template <typename indexes_type_>
concept vram_collection = requires { indexes_type_::vendor_k; };

/** The VRAM reading of the search race: the index admits no search during a mutation,
 *  so the exclusion is the executor's - a search on a claimed executor is refused, and the
 *  reader walks the published rows only once the writer has returned. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_search_race(u64_t seed, typename indexes_type_::executor_t& placement) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;

    std::size_t const nodes_count = 4 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement)));
    failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions}).status));

    dense_indexes_executor<indexes_type_> driver {placement};
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
std::size_t test_dense_indexes_planned_insert(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t rows_count = dense_thread_nodes_k, half = rows_count / 2;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> writer {placement}, reader {placement};
    if (failed(writer.status) || failed(reader.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement)));
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
        indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(indexes, placement)));
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
        auto found_distances = nk::matrix<distance_t>::zeros({queries_count, wanted}).value;
        std::array<std::size_t, queries_count> counts;
        failures += expect(succeeded(indexes
                                         .search(queries, inputs.heads(), driver.executor, exhaustive,
                                                 found_keys.span(), found_distances.span(), std::span {counts})
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

/** The two merges of the sharded search, over two shards on the device under test: the same queries
 *  ranked into host spans, merged on the host, and into spans on the first shard's device, merged
 *  there, agree on every count, on every distance, and on the keys of every run of equal distances,
 *  which the merges may order differently. An odd row count splits the shards unevenly, and a width
 *  past either shard's rows leaves both rankings short and the merged ones full. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
std::size_t test_dense_vram_indexes_device_merge(u64_t seed, typename indexes_type_::executor_t& placement) {
    using scalar_t = typename indexes_type_::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t rows_count = 255, queries_count = 8, wanted = 200;
    constexpr std::size_t ranks_count = queries_count * wanted;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(indexes.emplace(driver.executor.stream(0))));
    failures += expect(succeeded(indexes.emplace(driver.executor.stream(0))));
    failures += expect(succeeded(indexes.reserve({rows_count, connectivity_16_k, dimensions}).status));
    auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "indexes_device_merge"));
    std::array<key_t, rows_count> keys;
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = static_cast<key_t>(1 + row);
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));

    vram_suite_t const* suite = nullptr;
    for (vram_suite_t const& registered : vram_suites())
        if (registered.vendor == indexes_type_::vendor_k) suite = &registered;
    typename indexes_type_::executor_t::device_arena_t arena {driver.executor.stream(0)};
    key_t* const device_keys = arena.template allocate<key_t>(ranks_count);
    distance_t* const device_distances = arena.template allocate<distance_t>(ranks_count);
    std::size_t* const device_counts = arena.template allocate<std::size_t>(queries_count);
    if (failures || !suite || !device_keys || !device_distances || !device_counts) return failures + expect(false);

    auto const queries = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
    typename indexes_type_::search_config_t const exhaustive {rows_count, wanted};
    std::vector<key_t> host_keys(ranks_count), merged_keys(ranks_count);
    std::vector<distance_t> host_distances(ranks_count), merged_distances(ranks_count);
    std::array<std::size_t, queries_count> host_counts, merged_counts;
    failures += expect(succeeded(indexes
                                     .search(queries, driver.executor, exhaustive,
                                             nk::matrix_span<key_t> {host_keys.data(), queries_count, wanted},
                                             nk::matrix_span<distance_t> {host_distances.data(), queries_count, wanted},
                                             std::span {host_counts})
                                     .status));
    failures += expect(succeeded(indexes
                                     .search(queries, driver.executor, exhaustive,
                                             nk::matrix_span<key_t> {device_keys, queries_count, wanted},
                                             nk::matrix_span<distance_t> {device_distances, queries_count, wanted},
                                             std::span {device_counts, queries_count})
                                     .status));
    failures += expect(suite->download(merged_keys.data(), device_keys, ranks_count * sizeof(key_t)) &&
                       suite->download(merged_distances.data(), device_distances, ranks_count * sizeof(distance_t)) &&
                       suite->download(merged_counts.data(), device_counts, queries_count * sizeof(std::size_t)));
    if (failures) return failures;

    for (std::size_t query = 0; query != queries_count; ++query) {
        std::size_t const found = host_counts[query];
        failures += expect_eq(found, wanted);
        failures += expect_eq(merged_counts[query], found);
        if (merged_counts[query] != found) continue;
        std::size_t const first = query * wanted;
        for (std::size_t rank = first; rank != first + found; ++rank)
            failures += expect_near(merged_distances[rank], host_distances[rank]);
        for (std::size_t begin = first, end = first; begin != first + found; begin = end) {
            while (end != first + found && host_distances[end] == host_distances[begin]) ++end;
            // A run the width cuts may keep different ties on either side.
            if (end == first + wanted) break;
            std::sort(host_keys.begin() + begin, host_keys.begin() + end);
            std::sort(merged_keys.begin() + begin, merged_keys.begin() + end);
            failures += expect(
                std::equal(host_keys.begin() + begin, host_keys.begin() + end, merged_keys.begin() + begin));
        }
    }
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
        reference.built.save(reference.checkpoint.monolith.data(), save_config_t {}, reference.builder.executor)));

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement)));
    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status) || failures) return failures + expect(false);
    nvme_checkpoint_t const resaved {"vram_nvme_resaved"};
    failures += expect(
        succeeded(indexes.load(reference.checkpoint.monolith.data(), load_config_t {}, driver.executor)));
    failures += expect(succeeded(indexes.save(resaved.monolith.data(),
                                              dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::monolith_k),
                                              driver.executor)));

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

/** The sharded attention, keys equal to their rows and value rows looked up by key in a device
 *  table the vendor's @c value_table_type_ uploads: every shard walks every node, so each head's
 *  merged output and carry match exact full attention, with one head per token and with four
 *  sharing its walk, while a narrow walk stays within the oracle's bounds; a key without a value
 *  row, mismatched outputs, and a head count that does not divide the rows refuse. */
template <typename indexes_type_, typename value_table_type_>
std::size_t dense_expect_vram_indexes_attend(u64_t seed, typename indexes_type_::executor_t& placement) {
    using index_t = typename indexes_type_::index_t;
    using lookup_t = typename value_table_type_::lookup_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t nodes_count = dense_thread_nodes_k, tokens_count = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t failures = 0;

    dense_indexes_executor<indexes_type_> driver {placement};
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {nk_kernel_dot_k, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes, placement)));
    failures += expect(succeeded(indexes.reserve({nodes_count, connectivity_16_k, dimensions}).status));
    auto const vectors = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_attend"));
    auto const values = make_vectors<scalar_t>(nodes_count, dimensions, stream_key(seed, "indexes_attend", 1));
    std::vector<key_t> keys(nodes_count);
    for (std::size_t row = 0; row != nodes_count; ++row) keys[row] = static_cast<key_t>(row);
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));
    auto const table = value_table_type_::upload(values.view(), placement.stream(0));
    if (!table.rows) return failures + expect(false);

    typename indexes_type_::search_config_t const exhaustive {nodes_count};
    auto const queries = make_vectors<scalar_t>(tokens_count * 4, dimensions, stream_key(seed, "indexes_attend", 2));
    auto outputs = nk::matrix<f32_t>::zeros({queries.extent(0), dimensions}).value;
    std::vector<softmax_state_t> carry(queries.extent(0));
    auto attend = [&](nk::matrix_view<scalar_t> heads, std::size_t head_count, lookup_t lookup, f32_t score_scale,
                      typename indexes_type_::search_config_t config) noexcept {
        auto const [attended, status] = indexes.attend(
            heads, head_count, lookup, score_scale, outputs(nk::range(std::size_t {0}, heads.extent(0)), nk::slice),
            std::span {carry}.first(heads.extent(0)), driver.executor, config);
        failures += expect(failed(status) || attended.count == heads.extent(0));
        return status;
    };
    for (std::size_t const head_count : {std::size_t {1}, std::size_t {4}}) {
        auto const heads = queries(nk::range(std::size_t {0}, tokens_count * head_count), nk::slice);
        f32_t const score_scale = dense_attend_scale(
            dense_oracle_dots(dense_widened_rows(heads, index_t::vector_dtype_k),
                              dense_widened_rows(vectors.view(), index_t::vector_dtype_k), dimensions));
        nk::matrix_view<f32_t> const attended = outputs(nk::range(std::size_t {0}, heads.extent(0)), nk::slice);
        std::span<softmax_state_t const> const carried {carry.data(), heads.extent(0)};
        failures += expect(succeeded(attend(heads, head_count, table.lookup(nodes_count), score_scale, exhaustive)));
        failures += dense_expect_full_attention(
            vectors.view(), heads, head_count,
            [&](std::size_t row) noexcept { return values.view().row(row).as_vector(); }, score_scale, attended,
            carried);
        failures += expect(succeeded(attend(heads, head_count, table.lookup(nodes_count), score_scale, {})));
        failures += dense_expect_attention_bounds(vectors.view(), values.view(), heads, score_scale, attended, carried);
    }

    // Every shard walks every node, so the key past a shortened table refuses.
    failures += expect_eq(attend(queries.view(), 4, table.lookup(nodes_count - 1), 1, exhaustive),
                          status_t::shape_mismatch_k);
    failures += expect_eq(attend(queries.view(), 3, table.lookup(nodes_count), 1, exhaustive),
                          status_t::shape_mismatch_k);
    failures += expect_eq(indexes
                              .attend(queries.view(), 4, table.lookup(nodes_count), 1,
                                      outputs(nk::range(std::size_t {0}, std::size_t {1}), nk::slice),
                                      std::span {carry}, driver.executor, exhaustive)
                              .status,
                          status_t::config_mismatch_k);
    return failures;
}

/** A vendor this build leaves out composes nothing, so its shard instantiates empty. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename executor_type_>
std::size_t test_vram_indexes_configuration(char const*, u64_t, executor_type_&) {
    return 0;
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
    failures_count += test_dense_indexes_save_load<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_split_load<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_buffer_roundtrip<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_cancelled_load<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_concurrent_add<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_vram_indexes_search_race<indexes_t, kernel_>(seed, executor);
    if constexpr (executor_type_::vendor_k == gpu_vendor_t::nvidia_k)
        failures_count += test_dense_indexes_scan<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_planned_insert<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_refine<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_absorb<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_absorb_many<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_products<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_indexes_maxsim<indexes_t, kernel_>(seed, executor);
    failures_count += test_dense_vram_indexes_indexer_search<indexes_t, kernel_>(seed, executor);
    // Apple merges every ranking on the host, so its two merges are one path.
    if constexpr (executor_type_::vendor_k != gpu_vendor_t::apple_k)
        failures_count += test_dense_vram_indexes_device_merge<indexes_t, kernel_>(seed, executor);
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
        auto [executor, spawn_status] = executor_t::spawn(std::span<int const> {&device, 1});
        status = spawn_status;
        this->executor = std::move(executor);
    }
    dense_indexes_executor(dense_indexes_executor const&) = delete;
    dense_indexes_executor& operator=(dense_indexes_executor const&) = delete;

    /** Places the shards one sharded scenario drives: one VRAM shard on the device of
     *  @p placement; each further call places one more beside it. */
    static us::utilities::status_t place_shards(indexes_type_& indexes, executor_t& placement) noexcept {
        return indexes.emplace(placement.stream(0));
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
