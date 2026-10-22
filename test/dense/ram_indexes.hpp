/**
 *  @file test/dense/ram_indexes.hpp
 *  @author Ash Vardanian
 *  @date August 31, 2026
 *  @brief The CPU-side suite composer for @c dense::ram_indexes - the sharded index over one
 *      shard per memory domain - shared by its translation unit.
 *
 *  The scenarios live in @c dense.cuh, templated over the sharded-index type; this composer
 *  names a curated handful of cells rather than the per-dtype grid, since the routing, key
 *  columns, and executor logic under test are dtype-independent. The counting-allocator
 *  ledger stays with the single-index suite: the sharded index bakes its own placement
 *  allocators, so its leak discipline is the sanitizers' job rather than the ledger's.
 */
#pragma once

#include <fmt/format.h>
#include <numkong/numkong.hpp>

#include "usearch/dense/ram_indexes.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** The sharded RAM index over the kits the ladders pick for @p dtype_ under @p caps_, as @c src/
 *  spells them for a CPU class. */
template <nk_dtype_t dtype_, nk_capability_t caps_, typename waiter_type_ = fu::standard_yield_t,
          template <typename> class atomic_ref_ = fu::standard_atomic_ref>
using ram_indexes_for = ram_indexes<u32_t, typename quads_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                                    typename tiles_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                                    typename elementwise_ladder<caps_>::kit_t,
                                    typename rmscast_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                                    typename visits_hash_set_ladder<caps_>::kit_t, waiter_type_, atomic_ref_>;

/** No-op sibling picked when the cell cannot be compiled on this target: its serial
 *  kernels are not adapted yet, or the requested kit is absent from the build. */
template <                                        //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,  //
    typename waiter_type_ = fu::standard_yield_t, //
    nk_capability_t caps_ = nk_cap_serial_k,      //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
static std::size_t test_ram_indexes_configuration(char const* name, u64_t) {
    if constexpr (!quads_supported<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>)
        return report(name, "serial kernels not adapted yet");
    else return report(name, "kit not compiled for this target");
}

/**
 *  @brief An insertion arriving while an absorb into the index runs answers @c contended_k and
 *      lands nothing, and an absorb meeting a batch in flight answers the same and leaves the
 *      donor whole; every row that did land resolves exactly once.
 *
 *  One thread absorbs, retrying while a racing insertion holds a batch in flight, while another
 *  inserts single rows until the absorb returns. A round that sees no refusal refills the donor and
 *  races again.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_absorb_refusals(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const part = 2 * dense_thread_nodes_k, rounds = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    auto vectors = make_vectors<scalar_t>(part * (rounds + 1), dimensions, stream_key(seed, "indexes_absorb_refusals"));
    auto rows_of = [&](std::size_t first) { return vectors(nk::range(first, first + part), nk::slice); };
    std::vector<key_t> keys(part * (rounds + 1));
    for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = static_cast<key_t>(1 + row);
    auto keys_of = [&](std::size_t first) { return std::span<key_t const> {keys.data() + first, part}; };

    std::array<dense_indexes_executor<indexes_type_>, 2> drivers;
    for (dense_indexes_executor<indexes_type_> const& driver : drivers)
        if (failed(driver.status)) return failures + expect(false);
    indexes_type_ recipient {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ donor {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(recipient)) && succeeded(place_shards(donor)));
    for (indexes_type_* built : {&recipient, &donor})
        failures += expect(succeeded(built->reserve({part * (rounds + 1), connectivity_16_k, dimensions}).status));
    failures += expect(succeeded(recipient.insert(keys_of(0), rows_of(0), drivers[0].executor).status));
    failures += expect(succeeded(donor.insert(keys_of(part), rows_of(part), drivers[1].executor).status));

    // Racing rows take keys past every part's, so each resolves to the one row it landed.
    std::vector<key_t> landed_keys;
    key_t next_key = static_cast<key_t>(keys.size() + 1);
    {
        auto [plan, plan_status] = recipient.plan_insert(1, drivers[0].executor);
        failures += expect(succeeded(plan_status));
        failures += expect_eq(plan.insert(0, std::span<key_t const> {&next_key, 1}, vectors(nk::range(0, 1), nk::slice),
                                          to_writable_span(drivers[0].traversals)),
                              status_t::success_k);
        failures += expect_eq(recipient.absorb(donor, drivers[1].executor).status, status_t::contended_k);
        failures += expect_eq(donor.size(), part);
        failures += expect(succeeded(plan.finish().status));
        landed_keys.push_back(next_key++);
    }

    dense_indexes_row_driver<indexes_type_> rows_driver {recipient, drivers[0]};
    failures += expect(succeeded(rows_driver.status));
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(2))) return failures + expect(false);
    std::size_t refused = 0, unexpected = 0, absorbed_rows = 0;
    for (std::size_t round = 0; round != rounds && !refused; ++round) {
        if (round)
            failures += expect(succeeded(
                donor.insert(keys_of(part * (round + 1)), rows_of(part * (round + 1)), drivers[1].executor).status));
        std::atomic<bool> absorbing {false}, absorbed {false};
        status_t absorb_status = status_t::success_k;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                absorbing.store(true, std::memory_order_release);
                do absorb_status = recipient.absorb(donor, drivers[1].executor).status;
                while (absorb_status == status_t::contended_k);
                absorbed.store(true, std::memory_order_release);
            }
            else if (thread == 1) {
                while (!absorbing.load(std::memory_order_acquire)) std::this_thread::yield();
                for (std::size_t row = 0; !absorbed.load(std::memory_order_acquire); row = (row + 1) % part) {
                    status_t const status =
                        recipient.insert(next_key, vectors.view().row(row).as_vector(), rows_driver.handle()).status;
                    if (status == status_t::success_k) landed_keys.push_back(next_key++);
                    else if (status == status_t::contended_k) ++refused;
                    else ++unexpected;
                    std::this_thread::yield();
                }
            }
        });
        failures += expect_eq(absorb_status, status_t::success_k);
        failures += expect_eq(donor.size(), 0u);
        absorbed_rows += part;
    }
    failures += expect(refused > 0);
    failures += expect_eq(unexpected, 0u);
    failures += expect_eq(recipient.size(), part + absorbed_rows + landed_keys.size());
    std::size_t miscounted = 0;
    for (std::size_t row = 0; row != part + absorbed_rows; ++row) miscounted += recipient.count(keys[row]) != 1;
    for (key_t const key : landed_keys) miscounted += recipient.count(key) != 1;
    failures += expect_eq(miscounted, 0u);
    return failures;
}

/** A batch planned and dropped before its first step gives back the executor and every shard's
 *  claim, so planning the same batch again grows no shard and the rows then land. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_abandoned_plan(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({dense_thread_nodes_k, connectivity_16_k, dimensions}).status));

    // A whole number of rows per shard, so every rotation deals each shard the same share.
    std::size_t const rows = indexes.capacity() / indexes.shards_count() * indexes.shards_count();
    std::size_t const capacity = indexes.capacity();
    for (std::size_t attempt = 0; attempt != 2; ++attempt) {
        auto [abandoned, abandoned_status] = indexes.plan_insert(rows, driver.executor);
        failures += expect(succeeded(abandoned_status));
    }
    failures += expect_eq(indexes.capacity(), capacity);

    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_abandoned_plan"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);
    failures += expect(succeeded(indexes.insert(keys, vectors.view(), driver.executor).status));
    failures += expect_eq(indexes.size(), rows);
    failures += expect_eq(indexes.capacity(), capacity);
    return failures;
}

/** Past @c shard_capacity, rows open generations of shards on the same placements: one batch fills
 *  as many as it needs, a batch crossing the bound tops the open generation up and lands whole, one
 *  row past a full generation opens the next, and a load striping a checkpoint over the bound opens
 *  as many as its rows need. A monolith past the 32-bit IDs refuses to frame. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_generations(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t bound = 1000, rows = 5000;
    std::size_t failures = 0;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_generations"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = static_cast<key_t>(1 + row);
    auto rows_of = [&](std::size_t first, std::size_t count) {
        return vectors(nk::range(first, first + count), nk::slice);
    };
    auto keys_of = [&](std::size_t first, std::size_t count) {
        return std::span<key_t const> {keys.data() + first, count};
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
        return succeeded(place_shards(indexes)) &&
               succeeded(indexes.reserve({bound, connectivity_16_k, dimensions}).status) &&
               indexes.shard_capacity() == bound;
    };

    // One batch over two empty shards fills two generations and half a third.
    indexes_type_ filled {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(bounded(filled));
    failures += expect(succeeded(filled.insert(keys_of(0, rows), rows_of(0, rows), driver.executor).status));
    failures += expect_eq(filled.size(), rows);
    failures += expect((shard_sizes(filled) == std::vector<std::size_t> {1000, 1000, 1000, 1000, 500, 500}));
    failures += expect_eq(miscounted(filled, rows), 0u);

    // 750 rows a shard, then a batch of 1000 tops both up to the bound and opens a generation for
    // the other 500, whole, and a search reaches the new generation's rows.
    indexes_type_ crossed {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(bounded(crossed));
    failures += expect(succeeded(crossed.insert(keys_of(0, 1500), rows_of(0, 1500), driver.executor).status));
    failures += expect_eq(crossed.shards_count(), 2u);
    failures += expect(succeeded(crossed.insert(keys_of(1500, 1000), rows_of(1500, 1000), driver.executor).status));
    failures += expect((shard_sizes(crossed) == std::vector<std::size_t> {1000, 1000, 250, 250}));
    failures += expect_eq(miscounted(crossed, 2500), 0u);
    dense_indexes_row_driver<indexes_type_> rows_driver {crossed, driver};
    failures += expect(succeeded(rows_driver.status));
    if constexpr (kernel_ != nk_kernel_dot_k) {
        std::array<key_t, 8> found_keys {};
        std::array<typename indexes_type_::distance_t, 8> found_distances {};
        auto const [result, status] = crossed.search(vectors.view().row(2499).as_vector(), rows_driver.handle(), {},
                                                     std::span<key_t> {found_keys}, std::span {found_distances});
        failures += expect(succeeded(status) && result.count > 0);
        failures += expect(std::find(found_keys.begin(), found_keys.begin() + result.count, keys[2499]) !=
                           found_keys.begin() + result.count);
    }

    // A batch filling the second generation exactly opens none, and one more row opens a third.
    failures += expect(succeeded(crossed.insert(keys_of(2500, 1500), rows_of(2500, 1500), driver.executor).status));
    failures += expect_eq(crossed.shards_count(), 4u);
    failures += expect(
        succeeded(crossed.insert(keys[4000], vectors.view().row(4000).as_vector(), rows_driver.handle()).status));
    failures += expect_eq(crossed.shards_count(), 6u);
    failures += expect_eq(crossed.size(), 4001u);
    failures += expect_eq(miscounted(crossed, 4001), 0u);

    // A monolith of the filled index striped back over two bounded shards opens three generations.
    save_config_t const monolith = dense_indexes_save_config<indexes_type_>(checkpoint_layout_t::monolith_k);
    auto const [bytes, measure_status] = filled.storage_requirements(monolith);
    failures += expect(succeeded(measure_status));
    std::vector<std::byte> buffer(bytes);
    failures += expect(succeeded(filled.save(bytes_span_t {buffer}, monolith, driver.executor)));
    indexes_type_ restored {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    restored.bound_shards(bound);
    failures += expect(succeeded(place_shards(restored)));
    load_config_t striped;
    striped.placement = checkpoint_placement_t::split_k;
    failures += expect(succeeded(restored.load(bytes_view_t {buffer}, striped, driver.executor)));
    failures += expect_eq(restored.size(), rows);
    failures += expect((shard_sizes(restored) == std::vector<std::size_t> {834, 834, 833, 833, 833, 833}));
    failures += expect_eq(miscounted(restored, rows), 0u);

    // The guard every monolith save frames through, reached without four billion rows.
    typename indexes_type_::shards_t const unplaced {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect_eq(shard_monolith_header(unplaced, indexes_type_::index_t::max_node_count_k + 1,
                                                save_config_t {}, std::span<safegraphs_custom_field_t const> {})
                              .status,
                          status_t::overflow_k);
    return failures;
}

/** The filtered algebra verbs keep only what their predicate admits: products name even documents
 *  alone and refuse a scan, MaxSim ranks even documents at their exact scores under a walk and a
 *  scan alike, and attention folds no key a predicate rejects. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_filtered_algebra(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using vector_view_t = typename index_t::vector_view_t;
    using search_config_t = typename indexes_type_::search_config_t;
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else {
        constexpr std::size_t rows_count = dense_thread_nodes_k, tokens_per_document = 4;
        constexpr std::size_t documents_count = rows_count / tokens_per_document;
        constexpr std::size_t rows_per_query = 4, queries_count = 2, wanted = 5;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::size_t failures = 0;
        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return failures + expect(false);
        indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(indexes)));
        failures += expect(succeeded(indexes.reserve({rows_count, connectivity_16_k, dimensions}).status));
        auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "filtered_algebra"));
        std::array<key_t, rows_count> keys;
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = static_cast<key_t>(row / tokens_per_document);
        failures += expect(
            succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));
        auto const admit_even = quad_predicate(
            [](key_t key) noexcept { return key % 2 ? admission_t::reject_k : admission_t::admit_k; });
        search_config_t const exhaustive {static_cast<typename index_t::small_size_t>(rows_count)};

        auto const left = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
        auto product_keys = nk::matrix<key_t>::zeros({queries_count, rows_count}).value;
        auto products = nk::matrix<typename indexes_type_::dot_result_t>::zeros({queries_count, rows_count}).value;
        std::array<std::size_t, queries_count> counts;
        failures += expect(
            succeeded(indexes
                          .multiply_if(left, driver.executor, to_writable_span(driver.traversals), exhaustive,
                                       product_keys.span(), products.span(), std::span {counts}, admit_even)
                          .status));
        for (std::size_t row = 0; row != queries_count; ++row) {
            failures += expect(counts[row] > 0 && counts[row] <= rows_count / 2);
            for (std::size_t pair = 0; pair != min_of(counts[row], rows_count); ++pair)
                failures += expect(product_keys(row, pair) % 2 == 0);
        }
        // No exact product scan exists, so a scan refuses rather than walk.
        search_config_t scan;
        scan.traversal = traversal_t::scan_k;
        failures += expect(indexes
                               .multiply_if(left, driver.executor, to_writable_span(driver.traversals), scan,
                                            product_keys.span(), products.span(), std::span {counts}, admit_even)
                               .status == status_t::unsupported_k);

        // Every even document's exact score: each query row's best dot among its tokens, summed.
        auto const queries = vectors(nk::range(std::size_t {0}, queries_count * rows_per_query), nk::slice);
        std::array<std::array<f64_t, documents_count>, queries_count> exact {};
        for (std::size_t row = 0; row != queries_count * rows_per_query; ++row)
            for (std::size_t document = 0; document != documents_count; ++document) {
                f64_t best = 0;
                for (std::size_t token = 0; token != tokens_per_document; ++token)
                    best = max_of(best, static_cast<f64_t>(dense_native_dot(
                                            queries.row(row).as_vector(),
                                            vectors.view().row(document * tokens_per_document + token).as_vector())));
                exact[row / rows_per_query][document] += best;
            }
        auto found_keys = nk::matrix<key_t>::zeros({queries_count, wanted}).value;
        auto found_scores = nk::matrix<f64_t>::zeros({queries_count, wanted}).value;
        std::array<std::size_t, queries_count + 1> query_offsets;
        for (std::size_t query = 0; query != query_offsets.size(); ++query)
            query_offsets[query] = query * rows_per_query;
        // An exhaustive walk and a scan nominate every even document alike.
        for (search_config_t const config : {exhaustive, scan}) {
            counts.fill(0);
            std::array<std::size_t, queries_count> scored;
            maxsim_rows<key_t, f64_t> const land {found_keys.span(), found_scores.span(), std::span {counts}};
            failures += expect(succeeded(indexes
                                             .maxsim_if(queries, std::span<std::size_t const> {query_offsets},
                                                        driver.executor, to_writable_span(driver.traversals), config,
                                                        wanted, std::span {scored}, land, admit_even)
                                             .status));
            for (std::size_t query = 0; query != queries_count; ++query) {
                failures += expect_eq(counts[query], wanted) + expect_eq(scored[query], (documents_count + 1) / 2);
                for (std::size_t rank = 0; rank != counts[query]; ++rank) {
                    std::size_t const document = static_cast<std::size_t>(found_keys(query, rank));
                    failures += expect(document < documents_count && document % 2 == 0);
                    if (document < documents_count)
                        failures += expect_near(found_scores(query, rank), exact[query][document]);
                }
            }
        }

        // A lookup that never resolves would refuse any key that folds.
        auto const no_rows = [](std::span<key_t const>, std::span<vector_view_t>) noexcept {};
        auto const reject_all = quad_predicate([](key_t) noexcept { return admission_t::reject_k; });
        auto outputs = nk::matrix<f32_t>::zeros({queries_count, dimensions}).value;
        std::array<softmax_state_t, queries_count> carry;
        failures += expect(
            succeeded(indexes
                          .attend_if(left, 1, no_rows, 1.0f, outputs.span(), std::span {carry}, driver.executor,
                                     to_writable_span(driver.traversals), exhaustive, reject_all)
                          .status));
        for (softmax_state_t const& state : carry) failures += expect_eq(state.partition, 0.0f);
        return failures;
    }
}

template <                                        //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,  //
    typename waiter_type_ = fu::standard_yield_t, //
    nk_capability_t caps_ = nk_cap_serial_k,      //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
    requires dense_cell_testable<dtype_, kernel_, caps_>
static std::size_t test_ram_indexes_configuration(char const* name, u64_t seed) {

    using indexes_t = ram_indexes_for<dtype_, caps_, waiter_type_, atomic_ref_>;
    if (!dense_cell_admitted<caps_, atomic_ref_>()) return report(name, "capabilities missing at runtime");

    // The suite is composed here, not reflected: every call below is a capability claim
    // this sharded, executor-driven category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_indexes_guards<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_duplicate_keys<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_save_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_split_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_pooled_save<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_buffer_roundtrip<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_cancelled_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_concurrent_add<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_search_race<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_scan<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_arena_growth<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_refine<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_absorb<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_absorb_many<indexes_t, kernel_>(seed);
    failures_count += test_ram_indexes_absorb_refusals<indexes_t, kernel_>(seed);
    failures_count += test_ram_indexes_abandoned_plan<indexes_t, kernel_>(seed);
    failures_count += test_ram_indexes_generations<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_split<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_growth<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_products<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_indexer_search<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_maxsim<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_attend<indexes_t, kernel_>(seed);
    failures_count += test_ram_indexes_filtered_algebra<indexes_t, kernel_>(seed);
    return report(name, failures_count);
}

/** The exact scan and MaxSim alone over shards a matrix engine streams under @p caps_, skipped
 *  where the build lacks the kit or the host its capabilities; only dot kernels offer MaxSim. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, nk_capability_t caps_>
std::size_t test_ram_scan_configuration(char const* name, u64_t seed) {
    if constexpr (!dense_cell_testable<dtype_, kernel_, caps_>) return report(name, "kit not compiled for this target");
    else if (!dense_cell_admitted<caps_, fu::standard_atomic_ref>())
        return report(name, "capabilities missing at runtime");
    else
        return report(name, test_dense_indexes_scan<ram_indexes_for<dtype_, caps_>, kernel_>(seed) +
                                test_dense_indexes_maxsim<ram_indexes_for<dtype_, caps_>, kernel_>(seed));
}

/** The sharded-index suite entry its translation unit defines and @c dense.cpp
 *  composes explicitly; returns the suite's failure count. */
std::size_t check_ram_indexes(settings_t const& settings);

} // namespace unum::usearch::test::dense
