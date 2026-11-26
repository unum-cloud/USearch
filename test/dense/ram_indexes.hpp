/**
 *  @file test/dense/ram_indexes.hpp
 *  @author Ash Vardanian
 *  @date August 31, 2026
 *  @brief The CPU-side suite composer for @c dense::ram_indexes - the sharded index over one
 *      shard per memory domain - shared by its translation unit.
 *
 *  The scenarios live in @c dense.cuh, templated over the sharded-index type; this composer names a
 *  curated handful of cells rather than the per-dtype grid, since the routing, keys, and executor
 *  logic under test are dtype-independent. The counting-allocator ledger stays with the
 *  single-index suite: the sharded index bakes its own placement allocators, so its leak discipline
 *  is the sanitizers' job rather than the ledger's.
 */
#pragma once

#include <cmath>   // `std::isfinite` - every draw's coordinates
#include <cstring> // `std::memcmp` - the repeated draws' coordinates

#include <algorithm> // `std::find` - a draw among the shown keys
#include <vector>    // `std::vector` - a draw's outputs

#include <fmt/format.h>

#include <numkong/numkong.hpp>

#include "usearch/dense/ram_indexes.hpp"
#include "usearch/dense/ram_reshard.hpp"

#include "dense.cuh"
#include "dense_specializations.cuh" // `quads_kit_of` & kin - the kits a CPU class runs a dtype on

namespace unum::usearch::test::dense {

/** A behavior cell: the kits of @p cpu_class_ under a waiter or atomic it would not pick. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_, typename waiter_type_ = fu::standard_yield_t,
          template <typename> class atomic_ref_ = fu::standard_atomic_ref>
using ram_indexes_for = ram_indexes<uuid128_t, quads_kit_of<dtype_, cpu_class_>, tiles_kit_of<dtype_, cpu_class_>,
                                    elementwise_kit_of<cpu_class_>, rmscast_kit_of<dtype_, cpu_class_>,
                                    visits_kit_of<cpu_class_>, waiter_type_, atomic_ref_>;

/** Reshards @p source into @p target, which leads its own sources and keeps its shards. */
template <typename indexes_type_>
static expected<reshard_report_t> ram_indexes_gather(indexes_type_& target, indexes_type_& source,
                                                     typename indexes_type_::executor_t& executor) {
    std::array<indexes_type_*, 2> const sources {&target, &source};
    std::array<indexes_type_*, 1> const targets {&target};
    return reshard(std::span<indexes_type_* const> {sources}, std::span<indexes_type_* const> {targets}, executor);
}

/** No-op sibling picked when the cell cannot be compiled on this target: its serial
 *  kernels are not adapted yet, or the requested kit is absent from the build. */
template <                                          //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,    //
    typename waiter_type_ = fu::standard_yield_t,   //
    cpu_class_t cpu_class_ = cpu_class_t::serial_k, //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
static case_result_t test_ram_indexes_configuration(u64_t) {
    if constexpr (!a_quads_supported_kernel<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>)
        return case_result_t::skipped(needs_serial_kernels_k);
    else return case_result_t::skipped(needs_kit_target_k);
}

/**
 *  @brief An insertion arriving while a gather into the index runs answers @c contended_k and
 *      places nothing, and a gather meeting a batch in flight answers the same and leaves the
 *      source whole; every entry that was placed resolves exactly once.
 *
 *  One thread gathers, retrying while a racing insertion holds a batch in flight, while another
 *  inserts single entries until the gather returns. A round that sees no refusal refills the source
 *  and races again.
 */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_gather_refusals(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const part = 2 * dense_thread_nodes_k, rounds = 4;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(part * (rounds + 1), dimensions,
                                                          stream_key(seed, "indexes_gather_refusals"));
    auto rows_of = [&](std::size_t first) { return vectors(nk::range(first, first + part), nk::slice); };
    nk::vector<key_t> keys = make_vector<key_t>(part * (rounds + 1));
    for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = dense_key<key_t>(1 + row);
    auto keys_of = [&](std::size_t first) { return std::span<key_t const> {keys.data() + first, part}; };

    std::array<dense_indexes_executor<indexes_type_>, 2> drivers;
    for (dense_indexes_executor<indexes_type_> const& driver : drivers)
        if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ target {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ source {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(target)) && succeeded(place_shards(source)));
    for (indexes_type_* built : {&target, &source})
        expect_true(succeeded(built->reserve({part * (rounds + 1)}).status));
    expect_true(succeeded(target.insert(keys_of(0), rows_of(0), drivers[0].executor).status));
    expect_true(succeeded(source.insert(keys_of(part), rows_of(part), drivers[1].executor).status));

    // Racing entries take keys past every part's, so each resolves to the one entry it placed.
    std::vector<key_t> inserted_keys;
    u64_t next_number = keys.size() + 1;
    key_t next_key = dense_key<key_t>(next_number);
    {
        auto [plan, plan_status] = target.plan_insert(1, drivers[0].executor);
        expect_true(succeeded(plan_status));
        expect_equal(plan.insert(0, std::span<key_t const> {&next_key, 1}, vectors(nk::range(0, 1), nk::slice),
                                 to_writable_span(drivers[0].traversals)),
                     status_code_t::success_k);
        expect_equal(ram_indexes_gather(target, source, drivers[1].executor).status, status_code_t::contended_k);
        expect_equal(source.size(), part);
        expect_true(succeeded(plan.finish().status));
        inserted_keys.push_back(next_key);
        next_key = dense_key<key_t>(++next_number);
    }

    dense_indexes_row_driver<indexes_type_> rows_driver {target, drivers[0]};
    expect_true(succeeded(rows_driver.status));
    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) || fu::failed(pool.spawn(topology, 2, fu::caller_inclusive_k)))
        return void(expect_true(false));
    std::size_t refused = 0, unexpected = 0, gathered_entries = 0;
    for (std::size_t round = 0; round != rounds && !refused; ++round) {
        // A consumed source stays closed and sealed, so every round gathers a fresh one.
        if (round) {
            source =
                indexes_type_ {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
            expect_true(succeeded(place_shards(source)));
            expect_true(succeeded(source.reserve({part}).status));
            expect_true(succeeded(
                source.insert(keys_of(part * (round + 1)), rows_of(part * (round + 1)), drivers[1].executor).status));
        }
        bool gathering = false, gathered = false;
        status_t gather_status = status_code_t::success_k;
        expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                fu::standard_atomic_ref<bool>(gathering).store(true, std::memory_order_release);
                do gather_status = ram_indexes_gather(target, source, drivers[1].executor).status;
                while (gather_status == status_code_t::contended_k);
                fu::standard_atomic_ref<bool>(gathered).store(true, std::memory_order_release);
            }
            else if (thread == 1) {
                while (!fu::standard_atomic_ref<bool>(gathering).load(std::memory_order_acquire))
                    std::this_thread::yield();
                for (std::size_t row = 0; !fu::standard_atomic_ref<bool>(gathered).load(std::memory_order_acquire);
                     row = (row + 1) % part) {
                    status_t const status =
                        target.insert(next_key, vectors.view().row(row).as_vector(), rows_driver.handle()).status;
                    if (status == status_code_t::success_k) {
                        inserted_keys.push_back(next_key);
                        next_key = dense_key<key_t>(++next_number);
                    }
                    else if (status == status_code_t::contended_k) ++refused;
                    else ++unexpected;
                    std::this_thread::yield();
                }
            }
        })));
        expect_equal(gather_status, status_code_t::success_k);
        expect_equal(source.size(), 0u);
        gathered_entries += part;
    }
    expect_true(refused > 0);
    expect_equal(unexpected, 0u);
    expect_equal(target.size(), part + gathered_entries + inserted_keys.size());
    std::size_t miscounted = 0;
    for (std::size_t entry = 0; entry != part + gathered_entries; ++entry) miscounted += target.count(keys[entry]) != 1;
    for (key_t const key : inserted_keys) miscounted += target.count(key) != 1;
    expect_equal(miscounted, 0u);
}

/** A batch planned and dropped before its first step gives back the executor and every shard's
 *  claim, so planning the same batch again grows no shard and the entries then are placed. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_abandoned_plan(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes)));
    expect_true(succeeded(indexes.reserve({dense_thread_nodes_k}).status));

    // A whole number of entries per shard, so every batch split gives each shard the same share.
    std::size_t const rows = indexes.capacity() / indexes.shards_count() * indexes.shards_count();
    std::size_t const capacity = indexes.capacity();
    for (std::size_t attempt = 0; attempt != 2; ++attempt) {
        auto [abandoned, abandoned_status] = indexes.plan_insert(rows, driver.executor);
        expect_true(succeeded(abandoned_status));
    }
    expect_equal(indexes.capacity(), capacity);

    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_abandoned_plan"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));
    expect_equal(indexes.size(), rows);
    expect_equal(indexes.capacity(), capacity);
}

/** A batch narrower than the executor fills an empty index: its entries wire at once, each while
 *  the ones dispensed ahead of it may be unwired, and the audit still reaches every entry. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_wide_executor(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k), rows = 64;
    dense_indexes_executor<indexes_type_> driver {2 * rows};
    if (failed(driver.status)) return void(expect_true(false));
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_wide_executor"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    // Each build races differently, so a few of them catch a herd that one may miss.
    for (std::size_t build = 0; build != 4; ++build) {
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(indexes.emplace(no_memory_domain_k)));
        expect_true(succeeded(indexes.reserve({rows}).status));
        expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));
        auto const [health, status] = indexes.health(driver.executor);
        expect_true(succeeded(status));
        if constexpr (kernel_ != nk_kernel_dot_k) expect_equal(health.reachable_nodes, rows);
        else expect_true(health.reachable_nodes * 100 >= rows * dense_minimal_recall_percent_k);
    }
}

/** SAMPLE over shards: one seed draws the same keys at the same bits twice; a focus key draws
 *  first; a call wanting fewer draws a prefix of one wanting more, parents included; and a call
 *  shown the first one's draws never draws them again, hanging every draw from a shown point or
 *  from an earlier draw. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_sample(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t const rows = 192;
    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes)));
    expect_true(succeeded(indexes.reserve({rows}).status));
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_sample"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));

    struct drawn_t {
        std::vector<key_t> keys, parents;
        std::vector<std::size_t> entry_ordinals;
        std::vector<f32_t> distances, coordinates;
        std::size_t written = 0;
    };
    u64_t const sample_seed = seed ^ 0x5A5Au;
    auto draw = [&](std::size_t wanted, std::span<key_t const> shown, std::span<f32_t const> shown_coordinates,
                    std::span<key_t const> focus) {
        drawn_t drawn {std::vector<key_t>(wanted), std::vector<key_t>(wanted),     std::vector<std::size_t>(wanted),
                       std::vector<f32_t>(wanted), std::vector<f32_t>(wanted * 3), 0};
        sample_outputs<key_t> const outputs {drawn.keys, drawn.entry_ordinals, drawn.parents, drawn.distances,
                                             drawn.coordinates};
        auto const [result, status] = indexes.sample({shown, shown_coordinates, focus}, outputs, {sample_seed, 0},
                                                     driver.executor);
        expect_true(succeeded(status));
        drawn.written = result.written;
        for (std::size_t draw = 0; draw != drawn.written; ++draw) {
            expect_equal(drawn.entry_ordinals[draw], 0u);
            for (std::size_t column = 0; column != 3; ++column)
                expect_true(std::isfinite(drawn.coordinates[draw * 3 + column]));
        }
        return drawn;
    };
    auto among = [](std::span<key_t const> haystack, key_t const& key) {
        return std::find(haystack.begin(), haystack.end(), key) != haystack.end();
    };

    drawn_t const first = draw(64, {}, {}, {}), again = draw(64, {}, {}, {});
    expect_true(first.written > 0);
    expect_equal(again.written, first.written);
    expect_true(first.keys == again.keys && first.parents == again.parents);
    expect_true(
        std::memcmp(first.coordinates.data(), again.coordinates.data(), first.coordinates.size() * sizeof(f32_t)) == 0);

    key_t const focus = keys[7];
    drawn_t const narrow = draw(16, {}, {}, std::span<key_t const>(&focus, 1));
    drawn_t const wide = draw(48, {}, {}, std::span<key_t const>(&focus, 1));
    expect_true(narrow.written > 0 && narrow.keys[0] == focus && wide.keys[0] == focus);
    expect_true(narrow.written <= wide.written);
    for (std::size_t draw = 0; draw != min_of(narrow.written, wide.written); ++draw)
        expect_true(narrow.keys[draw] == wide.keys[draw] && narrow.parents[draw] == wide.parents[draw] &&
                    narrow.distances[draw] == wide.distances[draw]);

    std::size_t const shown_count = min_of<std::size_t>(16, first.written);
    std::span<key_t const> const shown_keys(first.keys.data(), shown_count);
    std::span<f32_t const> const shown_coordinates(first.coordinates.data(), shown_count * 3);
    drawn_t const zoomed = draw(32, shown_keys, shown_coordinates, {});
    expect_true(zoomed.written > 0);
    for (std::size_t draw = 0; draw != zoomed.written; ++draw) {
        expect_true(!among(shown_keys, zoomed.keys[draw]));
        std::span<key_t const> const earlier(zoomed.keys.data(), draw);
        expect_true(among(shown_keys, zoomed.parents[draw]) || among(earlier, zoomed.parents[draw]));
    }
}

/** Past @c shard_capacity, entries open generations of shards on the same placements: one batch
 *  fills as many as it needs, a batch crossing the bound tops the open generation up and is placed
 *  whole, one entry past a full generation opens the next, and a load of the filled index places
 *  each part on a shard of its own, as many generations as it saved. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_generations(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    constexpr std::size_t bound = 1000, rows = 5000;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return void(expect_true(false));
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_generations"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
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
        return succeeded(place_shards(indexes)) && succeeded(indexes.reserve({bound}).status) &&
               indexes.shard_capacity() == bound;
    };

    // One batch over two empty shards fills two generations and half a third.
    indexes_type_ filled {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(bounded(filled));
    expect_true(succeeded(filled.insert(keys_of(0, rows), rows_of(0, rows), driver.executor).status));
    expect_equal(filled.size(), rows);
    expect_true((shard_sizes(filled) == std::vector<std::size_t> {1000, 1000, 1000, 1000, 500, 500}));
    expect_equal(miscounted(filled, rows), 0u);

    // 750 entries a shard, then a batch of 1000 tops both up to the bound and opens a generation
    // for the other 500, whole, and a search reaches the new generation's entries.
    indexes_type_ crossed {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(bounded(crossed));
    expect_true(succeeded(crossed.insert(keys_of(0, 1500), rows_of(0, 1500), driver.executor).status));
    expect_equal(crossed.shards_count(), 2u);
    expect_true(succeeded(crossed.insert(keys_of(1500, 1000), rows_of(1500, 1000), driver.executor).status));
    expect_true((shard_sizes(crossed) == std::vector<std::size_t> {1000, 1000, 250, 250}));
    expect_equal(miscounted(crossed, 2500), 0u);
    dense_indexes_row_driver<indexes_type_> rows_driver {crossed, driver};
    expect_true(succeeded(rows_driver.status));
    if constexpr (kernel_ != nk_kernel_dot_k) {
        nk::vector<key_t> found_keys = make_vector<key_t>(8);
        nk::vector<typename indexes_type_::distance_t> found_distances =
            make_vector<typename indexes_type_::distance_t>(8);
        auto const [result, status] = crossed.search(vectors.view().row(2499).as_vector(), to_writable_span(found_keys),
                                                     to_writable_span(found_distances), {}, rows_driver.handle());
        expect_true(succeeded(status) && result.count > 0);
        expect_true(std::find(found_keys.data(), found_keys.data() + result.count, keys[2499]) !=
                    found_keys.data() + result.count);
    }

    // A batch filling the second generation exactly opens none, and one more entry opens a third.
    expect_true(succeeded(crossed.insert(keys_of(2500, 1500), rows_of(2500, 1500), driver.executor).status));
    expect_equal(crossed.shards_count(), 4u);
    expect_true(
        succeeded(crossed.insert(keys[4000], vectors.view().row(4000).as_vector(), rows_driver.handle()).status));
    expect_equal(crossed.shards_count(), 6u);
    expect_equal(crossed.size(), 4001u);
    expect_equal(miscounted(crossed, 4001), 0u);

    // Batches of 3500 over two shards bounded at 2500 entries: the second tops up the first
    // generation, fills a second and starts a third, and the third grows a fresh shard holding 250
    // entries by 1750, more than the chunk the bound truncates adds, though only its deficit is
    // missing.
    {
        constexpr std::size_t straddled_bound = 2500, batch = 3500, straddled_rows = 4 * batch;
        nk::matrix<scalar_t> straddled_vectors = make_vectors<scalar_t>(straddled_rows, dimensions,
                                                                        stream_key(seed, "straddled"));
        nk::vector<key_t> straddled_keys = make_vector<key_t>(straddled_rows);
        for (std::size_t row = 0; row != straddled_rows; ++row) straddled_keys[row] = dense_key<key_t>(1 + row);
        indexes_type_ straddled {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        straddled.bound_shards(straddled_bound);
        expect_true(succeeded(place_shards(straddled)));
        expect_true(succeeded(straddled.reserve({straddled_bound}).status));
        for (std::size_t const first : {std::size_t {0}, batch, 3 * batch}) {
            std::size_t const count = first == batch ? 2 * batch : batch;
            expect_true(
                succeeded(straddled
                              .insert(std::span<key_t const> {straddled_keys.data() + first, count},
                                      straddled_vectors(nk::range(first, first + count), nk::slice), driver.executor)
                              .status));
        }
        expect_true((shard_sizes(straddled) == std::vector<std::size_t> {2500, 2500, 2500, 2500, 2000, 2000}));
        std::size_t wrong = 0;
        for (key_t const& key : straddled_keys) wrong += straddled.count(key) != 1;
        expect_equal(wrong, 0u);
    }

    // The filled index restores into a bounded one part by part, each generation a shard again.
    save_config_t const config = dense_indexes_save_config<indexes_type_>();
    auto const [bytes, measure_status] = filled.storage_requirements(config);
    expect_true(succeeded(measure_status));
    std::vector<std::byte> buffer(bytes);
    expect_true(succeeded(filled.save(bytes_span_t {buffer}, config, driver.executor)));
    indexes_type_ restored {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    restored.bound_shards(bound);
    expect_true(succeeded(place_shards(restored)));
    expect_true(succeeded(restored.load(bytes_view_t {buffer}, load_config_t {}, driver.executor)));
    expect_equal(restored.size(), rows);
    expect_true(shard_sizes(restored) == shard_sizes(filled));
    expect_equal(miscounted(restored, rows), 0u);
}

/** Four even shards recut in place under a bound of half the entries into two, the kept ones
 *  keeping their IDs and the others staying as idle slots, while a looker counts every key once,
 *  then out of place into a fresh single shard, which empties the source. Every key resolves once
 *  with its vector intact after each, and searches find sampled vectors themselves within noise of
 *  before. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_recut(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using distance_t = typename indexes_type_::distance_t;
    std::size_t const rows = dense_growth_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return void(expect_true(false));
    auto& executor = driver.executor;
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_recut"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    auto filled_shards = [](indexes_type_ const& indexes) {
        std::size_t filled = 0;
        for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard)
            filled += indexes.shard(shard).size() != 0;
        return filled;
    };
    // Sampled vectors whose search returns their own key; a dot graph's hubs make self-search moot.
    auto self_hits = [&](indexes_type_& indexes) {
        std::size_t hits = 0;
        if constexpr (kernel_ != nk_kernel_dot_k) {
            dense_indexes_row_driver<indexes_type_> walker {indexes, driver};
            expect_true(succeeded(walker.status));
            nk::vector<key_t> found_keys = make_vector<key_t>(8);
            nk::vector<distance_t> found_distances = make_vector<distance_t>(8);
            for (std::size_t sample = 0; sample != recall_samples_k; ++sample) {
                std::size_t const row = sample * (rows / recall_samples_k);
                auto const [result, status] = indexes.search(vectors.view().row(row).as_vector(),
                                                             to_writable_span(found_keys),
                                                             to_writable_span(found_distances), {}, walker.handle());
                expect_true(succeeded(status));
                hits += std::find(found_keys.data(), found_keys.data() + result.count, keys[row]) !=
                        found_keys.data() + result.count;
            }
        }
        return hits;
    };

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes)) && succeeded(place_shards(indexes)));
    expect_true(succeeded(indexes.reserve({rows}).status));
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), executor).status));
    expect_equal(filled_shards(indexes), 4u);
    std::size_t const hits_before = self_hits(indexes);

    // In place: each kept shard takes one other, the only entries that move, and then runs a round
    // sized by the executor's cache. A looker counts keys throughout, and each moved entry's key
    // moves and its old shard forgets it inside one window, so every key counts exactly once.
    std::array<indexes_type_*, 1> const in_place {&indexes};
    reshard_config_t halves;
    halves.shard_capacity = rows / 2;
    expected<reshard_report_t> recut {};
    bool recutting = true;
    std::size_t miscounted = 0;
    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) || fu::failed(pool.spawn(topology, 2, fu::caller_inclusive_k)))
        return void(expect_true(false));
    expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
        if (thread == 0) {
            recut = reshard(std::span<indexes_type_* const> {in_place}, std::span<indexes_type_* const> {in_place},
                            executor, halves);
            fu::standard_atomic_ref<bool>(recutting).store(false, std::memory_order_release);
        }
        else if (thread == 1)
            for (std::size_t row = 0; fu::standard_atomic_ref<bool>(recutting).load(std::memory_order_acquire);
                 row = (row + 1) % rows)
                miscounted += indexes.count(keys[row]) != 1;
    })));
    expect_equal(miscounted, 0u);
    auto const [packed, packed_status] = recut;
    expect_equal(packed_status, status_code_t::success_k);
    expect_true(packed.source_shards == 4 && packed.target_shards == 2 && packed.moved_entries == rows / 2);
    expect_true(packed.graph.reselected_nodes >= rows);
    expect_true(packed.graph.neighborhoods > 1 && packed.graph.traversal_stats.visited_members == 0);
    expect_equal(indexes.shards_count(), 4u);
    expect_equal(filled_shards(indexes), 2u);
    dense_indexes_expect_restored(indexes, vectors, keys, dimensions, executor, seed);
    expect_true(self_hits(indexes) + recall_margin_k * recall_samples_k >= hits_before);

    // A source of another geometry refuses the plan before anything closes, so every index stays
    // open.
    indexes_type_ mismatched {kernel_, {dimensions, connectivity_32_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(mismatched)));
    expect_true(succeeded(mismatched.reserve({rows}).status));
    std::array<indexes_type_*, 2> const mixed {&indexes, &mismatched};
    expect_equal(
        reshard(std::span<indexes_type_* const> {mixed}, std::span<indexes_type_* const> {in_place}, executor, halves)
            .status,
        status_code_t::config_mismatch_k);
    expect_true(succeeded(indexes.plan_insert(1, executor).status));
    expect_true(succeeded(mismatched.plan_insert(1, executor).status));

    // Out of place into a fresh single shard: every entry moves, and the source ends empty.
    indexes_type_ single {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(single.emplace(no_memory_domain_k)));
    std::array<indexes_type_*, 1> const into_single {&single};
    auto const [moved, moved_status] = reshard(std::span<indexes_type_* const> {in_place},
                                               std::span<indexes_type_* const> {into_single}, executor);
    expect_equal(moved_status, status_code_t::success_k);
    expect_true(moved.source_shards == 4 && moved.target_shards == 1 && moved.moved_entries == rows);
    expect_equal(indexes.size(), 0u);
    dense_indexes_expect_restored(single, vectors, keys, dimensions, executor, seed);
    expect_true(self_hits(single) + recall_margin_k * recall_samples_k >= hits_before);
}

/** Every verb is safe beside a clear: searches racing one return their own rows or
 *  @c contended_k, an insertion plan holding claims refuses a clear and a gather with
 *  @c contended_k until it dies, and a cursor opened before a clear answers @c contended_k. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_clear_race(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using distance_t = typename indexes_type_::distance_t;
    constexpr std::size_t searchers_k = 4, rounds_k = 4;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    nk::matrix<scalar_t> vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_clear_race"));
    nk::vector<key_t> keys = make_vector<key_t>(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    std::array<dense_indexes_executor<indexes_type_>, 2> drivers;
    for (dense_indexes_executor<indexes_type_> const& driver : drivers)
        if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes)));
    expect_true(succeeded(indexes.reserve({rows}).status));

    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) || fu::failed(pool.spawn(topology, searchers_k + 1, fu::caller_inclusive_k)))
        return void(expect_true(false));
    std::size_t foreign_count = 0, refused_count = 0;
    fu::standard_atomic_ref<std::size_t> const foreign {foreign_count}, refused {refused_count};
    for (std::size_t round = 0; round != rounds_k; ++round) {
        expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), drivers[0].executor).status));
        bool cleared = false;
        expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                while (indexes.clear() == status_code_t::contended_k) std::this_thread::yield();
                fu::standard_atomic_ref<bool>(cleared).store(true, std::memory_order_release);
                return;
            }
            dense_indexes_row_driver<indexes_type_> walker {indexes, drivers[0]};
            if (failed(walker.status)) return foreign.store_add(1);
            nk::vector<key_t> found_keys = make_vector<key_t>(8);
            nk::vector<distance_t> found_distances = make_vector<distance_t>(8);
            for (std::size_t row = thread; !fu::standard_atomic_ref<bool>(cleared).load(std::memory_order_acquire);
                 row = (row + searchers_k) % rows) {
                auto const [result, status] = indexes.search(vectors.view().row(row).as_vector(),
                                                             to_writable_span(found_keys),
                                                             to_writable_span(found_distances), {}, walker.handle());
                if (status == status_code_t::contended_k) refused.store_add(1);
                else if (failed(status)) foreign.store_add(1);
                else
                    for (std::size_t found = 0; found != result.count; ++found)
                        foreign.store_add(dense_key_number(found_keys[found]) < 1 ||
                                          dense_key_number(found_keys[found]) > rows);
            }
        })));
        expect_equal(indexes.size(), 0u);
        expect_equal(indexes.count(keys[0]), 0u);
    }
    expect_equal(foreign_count, 0u);

    // Claims a plan holds refuse every close, so neither a clear nor a gather frees them.
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), drivers[0].executor).status));
    indexes_type_ source {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(source)));
    {
        auto [plan, plan_status] = indexes.plan_insert(1, drivers[0].executor);
        expect_true(succeeded(plan_status));
        expect_equal(indexes.clear(), status_code_t::contended_k);
        expect_equal(ram_indexes_gather(indexes, source, drivers[1].executor).status, status_code_t::contended_k);
    }
    expect_equal(indexes.size(), rows);

    // A cursor reads under the epoch it opened in, and a clear since ends it.
    typename indexes_type_::cursor_t cursor {forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(cursor.open(indexes, {})));
    nk::vector<key_t> read_keys = make_vector<key_t>(4);
    auto const [before, before_status] = cursor.pull(indexes, to_writable_span(read_keys), nullptr, nullptr);
    expect_true(succeeded(before_status) && before == read_keys.size());
    expect_equal(indexes.clear(), status_code_t::success_k);
    auto const [after, after_status] = cursor.pull(indexes, to_writable_span(read_keys), nullptr, nullptr);
    expect_true(after_status == status_code_t::contended_k && after == 0);
}

/** The filtered algebra verbs keep only what their predicate admits: products name even documents
 *  alone under a walk and a scan alike, MaxSim ranks even documents at their exact scores under a
 *  walk and a scan alike, and attention folds no key a predicate rejects. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_filtered_algebra(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using dot_result_t = typename indexes_type_::dot_result_t;
    using vector_view_t = typename index_t::vector_view_t;
    using search_config_t = typename indexes_type_::search_config_t;
    if constexpr (kernel_ != nk_kernel_dot_k) return;
    else {
        constexpr std::size_t rows_count = dense_thread_nodes_k, tokens_per_document = 4;
        constexpr std::size_t documents_count = rows_count / tokens_per_document;
        constexpr std::size_t tokens_per_query = 4, queries_count = 2, wanted = 5;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return void(expect_true(false));
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(indexes)));
        expect_true(succeeded(indexes.reserve({rows_count}).status));
        nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions,
                                                                    stream_key(seed, "filtered_algebra"));
        nk::vector<key_t> keys = make_vector<key_t>(rows_count);
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(row / tokens_per_document);
        expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));
        auto const admit_even = quad_predicate([](key_t key) noexcept {
            return dense_key_number(key) % 2 ? admission_t::reject_k : admission_t::admit_k;
        });
        search_config_t const exhaustive {static_cast<typename index_t::candidates_count_t>(rows_count)};
        search_config_t scan;
        scan.traversal = traversal_t::scan_k;

        auto const left = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
        nk::vector<key_t> product_keys = make_vector<key_t>(queries_count * rows_count);
        nk::vector<dot_result_t> products = make_vector<dot_result_t>(queries_count * rows_count);
        std::array<std::size_t, queries_count + 1> offsets {};
        std::array<std::size_t, queries_count> scored {};
        products_tape<key_t, dot_result_t> const pairs {to_writable_span(product_keys), to_writable_span(products),
                                                        std::span {offsets}, rows_count};
        for (search_config_t const config : {exhaustive, scan}) {
            offsets[0] = 0;
            expect_true(succeeded(indexes
                                      .multiply_if(left, pairs, std::span {scored}, config, admit_even, driver.executor,
                                                   to_writable_span(driver.traversals))
                                      .status));
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const first = offsets[query], kept = offsets[query + 1] - first;
                expect_true(scored[query] > 0 && scored[query] <= rows_count / 2);
                expect_equal(kept, min_of(scored[query], rows_count));
                for (std::size_t pair = 0; pair != kept; ++pair)
                    expect_true(dense_key_number(product_keys[first + pair]) % 2 == 0);
            }
        }

        // Every even document's exact score: each query row's best dot among its tokens, summed.
        auto const queries = vectors(nk::range(std::size_t {0}, queries_count * tokens_per_query), nk::slice);
        nk::matrix<f64_t> exact = make_matrix<f64_t>(queries_count, documents_count);
        for (std::size_t row = 0; row != queries_count * tokens_per_query; ++row)
            for (std::size_t document = 0; document != documents_count; ++document) {
                f64_t best = 0;
                for (std::size_t token = 0; token != tokens_per_document; ++token)
                    best = max_of(best, static_cast<f64_t>(dense_native_dot(
                                            queries.row(row).as_vector(),
                                            vectors.view().row(document * tokens_per_document + token).as_vector())));
                exact(row / tokens_per_query, document) += best;
            }
        nk::vector<key_t> found_keys = make_vector<key_t>(queries_count * wanted);
        nk::vector<f64_t> found_scores = make_vector<f64_t>(queries_count * wanted);
        std::array<std::size_t, queries_count + 1> query_offsets;
        for (std::size_t query = 0; query != query_offsets.size(); ++query)
            query_offsets[query] = query * tokens_per_query;
        products_tape<key_t, f64_t> const matches {to_writable_span(found_keys), to_writable_span(found_scores),
                                                   std::span {offsets}, wanted};
        // An exhaustive walk and a scan nominate every even document alike.
        error_stats_t score_errors;
        for (search_config_t const config : {exhaustive, scan}) {
            offsets[0] = 0;
            expect_true(succeeded(indexes
                                      .maxsim_if(queries, std::span<std::size_t const> {query_offsets},
                                                 std::span {scored}, matches, {config, wanted}, admit_even,
                                                 driver.executor, to_writable_span(driver.traversals))
                                      .status));
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const first = offsets[query], kept = offsets[query + 1] - first;
                expect_equal(kept, wanted);
                expect_equal(scored[query], (documents_count + 1) / 2);
                for (std::size_t rank = 0; rank != kept; ++rank) {
                    std::size_t const document = dense_key_number(found_keys[first + rank]);
                    expect_true(document < documents_count && document % 2 == 0);
                    if (document < documents_count)
                        score_errors.add(found_scores[first + rank], exact(query, document));
                }
            }
        }
        expect_errors("filtered maxsim scores", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                      score_errors, 0);

        // A lookup that never resolves would refuse any key that folds.
        auto const no_rows = [](std::span<key_t const>, std::span<vector_view_t>) noexcept {};
        auto const reject_all = quad_predicate([](key_t) noexcept { return admission_t::reject_k; });
        nk::matrix<f32_t> outputs = make_matrix<f32_t>(queries_count, dimensions);
        nk::vector<softmax_state_t> carry = make_vector<softmax_state_t>(queries_count);
        expect_true(
            succeeded(indexes
                          .attend_if(left, 1, no_rows, 1.0f, outputs.span(), to_writable_span(carry), exhaustive,
                                     reject_all, driver.executor, to_writable_span(driver.traversals))
                          .status));
        for (softmax_state_t const& state : carry) expect_equal(state.partition, 0.0f);
        return;
    }
}

/** A MaxSim scan scores every key exactly once though its rows span shards: keeping every key, it
 *  writes each document once at its exact total and counts each once; a hint of the best document
 *  keeps it at the narrowest walk, its rows on both shards seeding every row's walk. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_maxsim_spanning(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using search_config_t = typename indexes_type_::search_config_t;
    if constexpr (kernel_ != nk_kernel_dot_k) return;
    else {
        constexpr std::size_t rows_count = dense_thread_nodes_k, tokens_per_document = 4, tokens_per_query = 4;
        constexpr std::size_t documents_count = rows_count / tokens_per_document;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return void(expect_true(false));
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(indexes)));
        nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions,
                                                                    stream_key(seed, "maxsim_spanning"));
        nk::vector<key_t> keys = make_vector<key_t>(rows_count);
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(1 + row / tokens_per_document);
        expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));
        // One batch rotates over both shards, so every document keeps rows in each.
        std::size_t spanning = 0;
        for (std::size_t row = 0; row != rows_count; row += tokens_per_document)
            spanning += indexes.shard(0).count(keys[row]) && indexes.shard(1).count(keys[row]);
        expect_equal(spanning, documents_count);

        nk::vector<f64_t> exact = make_vector<f64_t>(documents_count);
        auto const queries = vectors(nk::range(std::size_t {0}, tokens_per_query), nk::slice);
        for (std::size_t row = 0; row != tokens_per_query; ++row)
            for (std::size_t document = 0; document != documents_count; ++document) {
                f64_t best = std::numeric_limits<f64_t>::lowest();
                for (std::size_t token = 0; token != tokens_per_document; ++token)
                    best = max_of(best, static_cast<f64_t>(dense_native_dot(
                                            queries.row(row).as_vector(),
                                            vectors.view().row(document * tokens_per_document + token).as_vector())));
                exact[document] += best;
            }
        search_config_t scan;
        scan.traversal = traversal_t::scan_k;
        nk::vector<key_t> found_keys = make_vector<key_t>(rows_count);
        nk::vector<f64_t> found_scores = make_vector<f64_t>(rows_count);
        std::array<std::size_t, 1> scored {};
        std::array<std::size_t, 2> offsets {};
        std::array<std::size_t, 2> const query_offsets {0, tokens_per_query};
        products_tape<key_t, f64_t> const matches {to_writable_span(found_keys), to_writable_span(found_scores),
                                                   std::span {offsets}, rows_count};
        expect_true(succeeded(indexes
                                  .maxsim(queries, std::span<std::size_t const> {query_offsets}, std::span {scored},
                                          matches, {scan, 0}, driver.executor, to_writable_span(driver.traversals))
                                  .status));
        expect_equal(scored[0], documents_count);
        expect_equal(offsets[1], documents_count);
        std::array<std::size_t, documents_count> placed {};
        error_stats_t score_errors;
        for (std::size_t rank = 0; rank != min_of(offsets[1], documents_count); ++rank) {
            std::size_t const document = dense_key_number(found_keys[rank]) - 1;
            expect_true(document < documents_count);
            if (document >= documents_count) continue;
            ++placed[document];
            score_errors.add(found_scores[rank], exact[document]);
        }
        for (std::size_t const times : placed) expect_equal(times, 1u);

        // A hint of the best document seeds every row's walk with all its rows across both shards,
        // so the narrowest walk keeps it at its exact total, and a key held nowhere is counted.
        std::size_t best = 0;
        for (std::size_t document = 0; document != documents_count; ++document)
            if (exact[document] > exact[best]) best = document;
        nk::matrix<key_t> hint_keys = make_matrix<key_t>(1, 2);
        hint_keys(0, 0) = dense_key<key_t>(1 + best);
        hint_keys(0, 1) = dense_key<key_t>(1 + documents_count);
        std::array<std::size_t, 1> const hint_counts {2};
        search_config_t narrow {};
        narrow.expansion_factor = 1;
        offsets[0] = 0;
        auto const [hinted, hinted_status] = indexes.maxsim(
            queries, std::span<std::size_t const> {query_offsets}, std::span {scored}, matches, {narrow, 1},
            driver.executor, to_writable_span(driver.traversals),
            ranked_hints_view<key_t> {hint_keys.view(), std::span<std::size_t const> {hint_counts}});
        expect_true(succeeded(hinted_status));
        expect_equal(hinted.missing_hints, 1u);
        expect_equal(offsets[1], 1u);
        if (offsets[1] == 1) {
            expect_equal(dense_key_number(found_keys[0]) - 1, best);
            score_errors.add(found_scores[0], exact[best]);
        }
        return void(expect_errors("spanning maxsim scores", comparison_family_t::approximate_k,
                                  comparison_verdict_t::assert_k, score_errors, 0));
    }
}

/** Exact scans of the algebra verbs against brute force: a scan's products keep the widest dots,
 *  each exactly the native dot of a distinct stored row, largest first, and a scan's attention
 *  matches exact full attention, every query row its own head; only the dot kernel offers both. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_exact_scans(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using dot_result_t = typename indexes_type_::dot_result_t;
    using vector_view_t = typename index_t::vector_view_t;
    using search_config_t = typename indexes_type_::search_config_t;
    if constexpr (kernel_ != nk_kernel_dot_k) return;
    else {
        constexpr std::size_t rows_count = dense_thread_nodes_k, left_count = 16, wanted = 8, attended_count = 8;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return void(expect_true(false));
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(place_shards(indexes)));
        expect_true(succeeded(indexes.reserve({rows_count}).status));
        nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions,
                                                                    stream_key(seed, "exact_scans"));
        nk::vector<key_t> keys = make_vector<key_t>(rows_count);
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(row);
        expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));
        search_config_t scan;
        scan.traversal = traversal_t::scan_k;

        // As wide as a matrix engine's tile, so a tiled class streams the batch through it.
        auto const left = vectors(nk::range(std::size_t {0}, left_count), nk::slice);
        nk::vector<key_t> product_keys = make_vector<key_t>(left_count * wanted);
        nk::vector<dot_result_t> products = make_vector<dot_result_t>(left_count * wanted);
        std::array<std::size_t, left_count + 1> offsets {};
        std::array<std::size_t, left_count> scored {};
        products_tape<key_t, dot_result_t> const pairs {to_writable_span(product_keys), to_writable_span(products),
                                                        std::span {offsets}, wanted};
        expect_true(succeeded(
            indexes
                .multiply(left, pairs, std::span {scored}, scan, driver.executor, to_writable_span(driver.traversals))
                .status));
        nk::vector_view<key_t> const product_keys_view = product_keys.view();
        nk::vector_view<dot_result_t> const products_view = products.view();
        nk::vector<f64_t> exact = make_vector<f64_t>(rows_count);
        for (std::size_t row = 0; row != left_count; ++row) {
            std::size_t const first = offsets[row];
            expect_equal(scored[row], wanted);
            expect_equal(offsets[row + 1] - first, wanted);
            if (offsets[row + 1] - first != wanted) continue;
            dense_expect_product_row<rows_count>(left.row(row).as_vector(), vectors.view(),
                                                 product_keys_view[nk::range(first, first + wanted)],
                                                 products_view[nk::range(first, first + wanted)],
                                                 [](key_t key) noexcept { return dense_key_number(key); });
            // The ranking reads the finalized distance, so only a near tie at the cut may swap.
            for (std::size_t stored = 0; stored != rows_count; ++stored)
                exact[stored] = static_cast<f64_t>(
                    dense_native_dot(left.row(row).as_vector(), vectors.view().row(stored).as_vector()));
            std::sort(exact.data(), exact.data() + rows_count, std::greater<f64_t> {});
            for (std::size_t rank = 0; rank != wanted; ++rank) {
                f64_t const kept = static_cast<f64_t>(products[first + rank]);
                expect_true(kept + 1e-3 * max_of(std::fabs(exact[wanted - 1]), 1.0) >= exact[wanted - 1]);
                if (rank) expect_true(static_cast<f64_t>(products[first + rank - 1]) >= kept);
            }
        }

        nk::matrix<scalar_t> const values = make_vectors<scalar_t>(rows_count, dimensions,
                                                                   stream_key(seed, "exact_scans", 1));
        nk::matrix<scalar_t> const queries = make_vectors<scalar_t>(attended_count, dimensions,
                                                                    stream_key(seed, "exact_scans", 2));
        auto value_of = [&](std::size_t row) noexcept { return values.view().row(row).as_vector(); };
        auto lookup = [&](std::span<key_t const> batch, std::span<vector_view_t> rows) noexcept {
            for (std::size_t lane = 0; lane != batch.size(); ++lane)
                rows[lane] = value_of(dense_key_number(batch[lane]));
        };
        nk::matrix<f32_t> widened_queries = make_matrix<f32_t>(attended_count, dimensions);
        nk::matrix<f32_t> widened_vectors = make_matrix<f32_t>(rows_count, dimensions);
        nk::matrix<f32_t> widened_values = make_matrix<f32_t>(rows_count, dimensions);
        nk::matrix<dense_oracle_dot_t> dots = make_matrix<dense_oracle_dot_t>(attended_count, rows_count);
        dense_widened_rows(queries.view(), index_t::vector_dtype_k, widened_queries.span());
        dense_widened_rows(vectors.view(), index_t::vector_dtype_k, widened_vectors.span());
        dense_widened_rows(values.view(), index_t::vector_dtype_k, widened_values.span());
        dense_oracle_dots(widened_queries.view(), widened_vectors.view(), dots.span());
        f32_t const score_scale = dense_attend_scale(dots.view());
        nk::matrix<f32_t> outputs = make_matrix<f32_t>(attended_count, dimensions);
        nk::vector<softmax_state_t> carry = make_vector<softmax_state_t>(attended_count);
        auto const [attended, attend_status] = drive_attend(indexes, queries.view(), 1, lookup, score_scale,
                                                            outputs.span(), to_writable_span(carry), scan, driver);
        expect_true(succeeded(attend_status));
        expect_equal(attended.count, attended_count);
        dense_expect_full_attention(dots.view(), widened_values.view(), 1, score_scale, outputs.view(), to_span(carry));
        return;
    }
}

/** Hints seed every shard's walk: zero-count hints rank exactly as none, a hint of each query's
 *  exact nearest ranks it first at a narrow breadth, keys no shard holds are counted exactly, a
 *  warm second call ranks alike in the hint slots the first reserved, a query counting more hints
 *  than they are wide refuses, and a key stored under several nodes stages every one of them. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static void test_ram_indexes_hints(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using distance_t = typename indexes_type_::distance_t;
    using node_id_t = typename index_t::node_id_t;
    using search_config_t = typename indexes_type_::search_config_t;
    using search_result_t = typename indexes_type_::search_result_t;
    constexpr std::size_t rows_count = 2000, queries_count = 64, wanted = 10, shared_count = 8, hint_columns = 3;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    // One thread walks the queries in order from one seed, so two calls rank alike.
    dense_indexes_executor<indexes_type_> driver {1};
    if (failed(driver.status)) return void(expect_true(false));
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    expect_true(succeeded(place_shards(indexes)));
    expect_true(succeeded(indexes.reserve({rows_count}).status));
    nk::matrix<scalar_t> const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "hints"));
    // The leading rows share key one, so the keys from two up to their count are held nowhere.
    nk::vector<key_t> keys = make_vector<key_t>(rows_count);
    for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(row < shared_count ? 1 : 1 + row);
    expect_true(succeeded(indexes.insert(to_span(keys), vectors.view(), driver.executor).status));

    auto const queries = vectors(nk::range(rows_count - queries_count, rows_count), nk::slice);
    nk::matrix<key_t> exact_keys = make_matrix<key_t>(queries_count, 1);
    nk::matrix<distance_t> exact_distances = make_matrix<distance_t>(queries_count, 1);
    nk::vector<std::size_t> exact_counts = make_vector<std::size_t>(queries_count);
    search_config_t const scan {.traversal = traversal_t::scan_k};
    expect_true(succeeded(drive_search(indexes, queries, exact_keys.span(), exact_distances.span(),
                                       to_writable_span(exact_counts), scan, driver)
                              .status));

    // Each query's exact nearest, then two keys held nowhere.
    nk::matrix<key_t> hint_keys = make_matrix<key_t>(queries_count, hint_columns);
    nk::vector<std::size_t> hint_counts = make_vector<std::size_t>(queries_count);
    nk::vector<std::size_t> zero_counts = make_vector<std::size_t>(queries_count);
    std::fill_n(hint_counts.data(), queries_count, hint_columns);
    for (std::size_t query = 0; query != queries_count; ++query) {
        hint_keys(query, 0) = exact_keys(query, 0);
        hint_keys(query, 1) = dense_key<key_t>(2);
        hint_keys(query, 2) = dense_key<key_t>(rows_count + 1 + query);
    }
    search_config_t narrow {};
    narrow.expansion_factor = 16;
    struct ranked_t {
        nk::matrix<key_t> keys = make_matrix<key_t>(queries_count, wanted);
        nk::matrix<distance_t> distances = make_matrix<distance_t>(queries_count, wanted);
        nk::vector<std::size_t> counts = make_vector<std::size_t>(queries_count);
        expected<search_result_t> result {};
    };
    auto search = [&](ranked_t& ranked, ranked_hints_view<key_t> hints) {
        ranked_matches_span<key_t, distance_t> const matches {ranked.keys.span(), ranked.distances.span(),
                                                              to_writable_span(ranked.counts)};
        ranked.result = indexes.search(queries, matches, narrow, driver.executor, to_writable_span(driver.traversals),
                                       hints);
        expect_true(succeeded(ranked.result.status));
    };
    ranked_t plain, zero, hinted, warm;
    search(plain, {});
    search(zero, {hint_keys.view(), to_span(zero_counts)});
    search(hinted, {hint_keys.view(), to_span(hint_counts)});
    node_id_t const* const hinted_slots = driver.traversals[0].hinted_ids.data();
    std::byte const* const held_slots = driver.traversals[0].hinted_held.data();
    search(warm, {hint_keys.view(), to_span(hint_counts)});
    expect_true(driver.traversals[0].hinted_ids.data() == hinted_slots);
    expect_true(driver.traversals[0].hinted_held.data() == held_slots);
    expect_true(std::memcmp(warm.counts.data(), hinted.counts.data(), queries_count * sizeof(std::size_t)) == 0);
    expect_true(std::memcmp(warm.keys.data(), hinted.keys.data(), queries_count * wanted * sizeof(key_t)) == 0);

    expect_true(std::memcmp(zero.counts.data(), plain.counts.data(), queries_count * sizeof(std::size_t)) == 0);
    expect_true(std::memcmp(zero.keys.data(), plain.keys.data(), queries_count * wanted * sizeof(key_t)) == 0);
    expect_true(
        std::memcmp(zero.distances.data(), plain.distances.data(), queries_count * wanted * sizeof(distance_t)) == 0);
    expect_equal(zero.result.value.computed_distances, plain.result.value.computed_distances);
    expect_equal(zero.result.value.missing_hints, 0u);
    expect_equal(plain.result.value.missing_hints, 0u);
    expect_equal(hinted.result.value.missing_hints, 2 * queries_count);

    // A query counting more hints than the matrix is wide refuses before any walk.
    nk::vector<std::size_t> too_many = make_vector<std::size_t>(queries_count);
    std::fill_n(too_many.data(), queries_count, hint_columns);
    too_many[queries_count - 1] = hint_columns + 1;
    ranked_t refused;
    ranked_matches_span<key_t, distance_t> const refused_matches {refused.keys.span(), refused.distances.span(),
                                                                  to_writable_span(refused.counts)};
    expect_equal(indexes
                     .search(queries, refused_matches, narrow, driver.executor, to_writable_span(driver.traversals),
                             {hint_keys.view(), to_span(too_many)})
                     .status,
                 status_code_t::shape_mismatch_k);

    std::size_t plain_hits = 0, hinted_hits = 0;
    for (std::size_t query = 0; query != queries_count; ++query) {
        f64_t const exact = exact_distances(query, 0), tolerance = 1e-3 * max_of(std::fabs(exact), 1.0);
        plain_hits += plain.counts[query] && plain.distances(query, 0) <= exact + tolerance;
        hinted_hits += hinted.counts[query] && hinted.distances(query, 0) <= exact + tolerance;
    }
    expect_true(hinted_hits >= plain_hits);
    expect_equal(hinted_hits, queries_count);

    key_t const shared_key = dense_key<key_t>(1);
    std::size_t staged_count = 0;
    nk::vector<node_id_t> slots = make_vector<node_id_t>(shared_count);
    std::byte held {0};
    for (std::size_t shard = 0; shard != indexes.shards_count(); ++shard) {
        index_t const& shard_index = indexes.shard(shard);
        std::span<node_id_t const> const staged = to_span(slots).first(shard_index.stage_hints_(
            std::span<key_t const> {&shared_key, 1}, static_cast<node_id_t>(shard_index.size()),
            to_writable_span(slots), bytes_span_t {&held, 1}));
        staged_count += staged.size();
        typename index_t::keyed_t const& shard_keys = shard_index.keys();
        for (node_id_t const node : staged) expect_equal(dense_key_number(shard_keys[node]), 1u);
    }
    expect_equal(staged_count, shared_count);
    expect_true(held == std::byte {1});
}

template <                                          //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,    //
    typename waiter_type_ = fu::standard_yield_t,   //
    cpu_class_t cpu_class_ = cpu_class_t::serial_k, //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
    requires a_testable_dense_cell<dtype_, kernel_, numkong_capabilities_of(cpu_class_)>
static case_result_t test_ram_indexes_configuration(u64_t seed) {

    using indexes_t = ram_indexes_for<dtype_, cpu_class_, waiter_type_, atomic_ref_>;
    if (!dense_cell_admitted<numkong_capabilities_of(cpu_class_), atomic_ref_>())
        return case_result_t::skipped(needs_capabilities_k);

    // The suite is composed here, not reflected: every call below is a capability claim
    // this sharded, executor-driven category makes about itself.
    test_dense_indexes_guards<indexes_t, kernel_>(seed);
    test_dense_indexes_duplicate_keys<indexes_t, kernel_>(seed);
    dense_expect_duplicates_build<indexes_t, kernel_>(seed);
    test_dense_indexes_save_load<indexes_t, kernel_>(seed);
    test_dense_indexes_part_load<indexes_t, kernel_>(seed);
    test_dense_indexes_pooled_save<indexes_t, kernel_>(seed);
    test_dense_indexes_buffer_round_trip<indexes_t, kernel_>(seed);
    test_dense_indexes_cancelled_load<indexes_t, kernel_>(seed);
    test_dense_indexes_concurrent_add<indexes_t, kernel_>(seed);
    test_dense_indexes_search_race<indexes_t, kernel_>(seed);
    test_dense_indexes_scan<indexes_t, kernel_>(seed);
    test_dense_indexes_arena_growth<indexes_t, kernel_>(seed);
    test_dense_indexes_refine<indexes_t, kernel_>(seed);
    test_dense_indexes_gather<indexes_t, kernel_>(seed);
    test_dense_indexes_gather_many<indexes_t, kernel_>(seed);
    test_ram_indexes_gather_refusals<indexes_t, kernel_>(seed);
    test_ram_indexes_abandoned_plan<indexes_t, kernel_>(seed);
    test_ram_indexes_wide_executor<indexes_t, kernel_>(seed);
    test_ram_indexes_sample<indexes_t, kernel_>(seed);
    test_ram_indexes_generations<indexes_t, kernel_>(seed);
    test_ram_indexes_recut<indexes_t, kernel_>(seed);
    test_ram_indexes_clear_race<indexes_t, kernel_>(seed);
    test_dense_indexes_cut<indexes_t, kernel_>(seed);
    test_dense_indexes_growth<indexes_t, kernel_>(seed);
    test_dense_indexes_products<indexes_t, kernel_>(seed);
    test_dense_indexes_indexer_search<indexes_t, kernel_>(seed);
    test_dense_indexes_maxsim<indexes_t, kernel_>(seed);
    test_dense_indexes_attend<indexes_t, kernel_>(seed);
    test_ram_indexes_filtered_algebra<indexes_t, kernel_>(seed);
    test_ram_indexes_maxsim_spanning<indexes_t, kernel_>(seed);
    test_ram_indexes_exact_scans<indexes_t, kernel_>(seed);
    test_ram_indexes_hints<indexes_t, kernel_>(seed);
    return {};
}

/** The exact scans, attention and MaxSim alone over shards the kits of @p cpu_class_ stream,
 *  skipped where the build lacks the kit or the host its capabilities, and attention and MaxSim
 *  without a dot kernel. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_>
case_result_t test_ram_scan_configuration(u64_t seed) {
    constexpr nk_capability_t caps_k = numkong_capabilities_of(cpu_class_);
    using indexes_t = ram_indexes_for<dtype_, cpu_class_>;
    if constexpr (!a_testable_dense_cell<dtype_, kernel_, caps_k>) return case_result_t::skipped(needs_kit_target_k);
    else if (!dense_cell_admitted<caps_k, fu::standard_atomic_ref>())
        return case_result_t::skipped(needs_capabilities_k);
    else {
        test_dense_indexes_scan<indexes_t, kernel_>(seed);
        test_dense_indexes_maxsim<indexes_t, kernel_>(seed);
        test_ram_indexes_exact_scans<indexes_t, kernel_>(seed);
        return {};
    }
}

} // namespace unum::usearch::test::dense
