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
#include "usearch/dense/ram_compaction.hpp"

#include "dense.cuh"
#include "dense_specializations.cuh" // `quads_kit_of` & kin - the kits a CPU class runs a dtype on

namespace unum::usearch::test::dense {

/** A behavior cell: the kits of @p cpu_class_ under a waiter or atomic it would not pick. */
template <nk_dtype_t dtype_, cpu_class_t cpu_class_, typename waiter_type_ = fu::standard_yield_t,
          template <typename> class atomic_ref_ = fu::standard_atomic_ref>
using ram_indexes_for = ram_indexes<uuid_t, quads_kit_of<dtype_, cpu_class_>, tiles_kit_of<dtype_, cpu_class_>,
                                    elementwise_kit_of<cpu_class_>, rmscast_kit_of<dtype_, cpu_class_>,
                                    visits_kit_of<cpu_class_>, waiter_type_, atomic_ref_>;

/** No-op sibling picked when the cell cannot be compiled on this target: its serial
 *  kernels are not adapted yet, or the requested kit is absent from the build. */
template <                                          //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,    //
    typename waiter_type_ = fu::standard_yield_t,   //
    cpu_class_t cpu_class_ = cpu_class_t::serial_k, //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
static std::size_t test_ram_indexes_configuration(char const* name, u64_t) {
    if constexpr (!quads_supported<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>)
        return report(name, "serial kernels not adapted yet");
    else return report(name, "kit not compiled for this target");
}

/**
 *  @brief An insertion arriving while an absorb into the index runs answers @c contended_k and
 *      lands nothing, and an absorb meeting a batch in flight answers the same and leaves the
 *      donor whole; every entry that did land resolves exactly once.
 *
 *  One thread absorbs, retrying while a racing insertion holds a batch in flight, while another
 *  inserts single entries until the absorb returns. A round that sees no refusal refills the donor
 *  and races again.
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
    for (std::size_t row = 0; row != keys.size(); ++row) keys[row] = dense_key<key_t>(1 + row);
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

    // Racing entries take keys past every part's, so each resolves to the one entry it landed.
    std::vector<key_t> landed_keys;
    u64_t next_number = keys.size() + 1;
    key_t next_key = dense_key<key_t>(next_number);
    {
        auto [plan, plan_status] = recipient.plan_insert(1, drivers[0].executor);
        failures += expect(succeeded(plan_status));
        failures += expect_eq(plan.insert(0, std::span<key_t const> {&next_key, 1}, vectors(nk::range(0, 1), nk::slice),
                                          to_writable_span(drivers[0].traversals)),
                              status_t::success_k);
        failures += expect_eq(recipient.absorb(donor, drivers[1].executor).status, status_t::contended_k);
        failures += expect_eq(donor.size(), part);
        failures += expect(succeeded(plan.finish().status));
        landed_keys.push_back(next_key);
        next_key = dense_key<key_t>(++next_number);
    }

    dense_indexes_row_driver<indexes_type_> rows_driver {recipient, drivers[0]};
    failures += expect(succeeded(rows_driver.status));
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(2))) return failures + expect(false);
    std::size_t refused = 0, unexpected = 0, absorbed_rows = 0;
    for (std::size_t round = 0; round != rounds && !refused; ++round) {
        // A consumed donor stays closed and sealed, so every round absorbs a fresh one.
        if (round) {
            donor = indexes_type_ {kernel_, forkunion_allocator_t {no_memory_domain_k}};
            failures += expect(succeeded(place_shards(donor)));
            failures += expect(succeeded(donor.reserve({part, connectivity_16_k, dimensions}).status));
            failures += expect(succeeded(
                donor.insert(keys_of(part * (round + 1)), rows_of(part * (round + 1)), drivers[1].executor).status));
        }
        bool absorbing = false, absorbed = false;
        status_t absorb_status = status_t::success_k;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                fu::standard_atomic_ref<bool>(absorbing).store(true, std::memory_order_release);
                do absorb_status = recipient.absorb(donor, drivers[1].executor).status;
                while (absorb_status == status_t::contended_k);
                fu::standard_atomic_ref<bool>(absorbed).store(true, std::memory_order_release);
            }
            else if (thread == 1) {
                while (!fu::standard_atomic_ref<bool>(absorbing).load(std::memory_order_acquire))
                    std::this_thread::yield();
                for (std::size_t row = 0; !fu::standard_atomic_ref<bool>(absorbed).load(std::memory_order_acquire);
                     row = (row + 1) % part) {
                    status_t const status =
                        recipient.insert(next_key, vectors.view().row(row).as_vector(), rows_driver.handle()).status;
                    if (status == status_t::success_k) {
                        landed_keys.push_back(next_key);
                        next_key = dense_key<key_t>(++next_number);
                    }
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
 *  claim, so planning the same batch again grows no shard and the entries then land. */
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

    // A whole number of entries per shard, so every batch split gives each shard the same share.
    std::size_t const rows = indexes.capacity() / indexes.shards_count() * indexes.shards_count();
    std::size_t const capacity = indexes.capacity();
    for (std::size_t attempt = 0; attempt != 2; ++attempt) {
        auto [abandoned, abandoned_status] = indexes.plan_insert(rows, driver.executor);
        failures += expect(succeeded(abandoned_status));
    }
    failures += expect_eq(indexes.capacity(), capacity);

    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_abandoned_plan"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    failures += expect(succeeded(indexes.insert(keys, vectors.view(), driver.executor).status));
    failures += expect_eq(indexes.size(), rows);
    failures += expect_eq(indexes.capacity(), capacity);
    return failures;
}

/** Past @c shard_capacity, entries open generations of shards on the same placements: one batch
 *  fills as many as it needs, a batch crossing the bound tops the open generation up and lands
 *  whole, one entry past a full generation opens the next, and a load of the filled index lands
 *  each part on a shard of its own, as many generations as it saved. */
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

    // 750 entries a shard, then a batch of 1000 tops both up to the bound and opens a generation
    // for the other 500, whole, and a search reaches the new generation's entries.
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

    // A batch filling the second generation exactly opens none, and one more entry opens a third.
    failures += expect(succeeded(crossed.insert(keys_of(2500, 1500), rows_of(2500, 1500), driver.executor).status));
    failures += expect_eq(crossed.shards_count(), 4u);
    failures += expect(
        succeeded(crossed.insert(keys[4000], vectors.view().row(4000).as_vector(), rows_driver.handle()).status));
    failures += expect_eq(crossed.shards_count(), 6u);
    failures += expect_eq(crossed.size(), 4001u);
    failures += expect_eq(miscounted(crossed, 4001), 0u);

    // Batches of 3500 over two shards bounded at 2500 entries: the second tops up the first
    // generation, fills a second and starts a third, and the third grows a fresh shard holding 250
    // entries by 1750, more than the chunk the bound truncates adds, though only its deficit is
    // missing.
    {
        constexpr std::size_t straddled_bound = 2500, batch = 3500, straddled_rows = 4 * batch;
        auto straddled_vectors = make_vectors<scalar_t>(straddled_rows, dimensions, stream_key(seed, "straddled"));
        std::vector<key_t> straddled_keys(straddled_rows);
        for (std::size_t row = 0; row != straddled_rows; ++row) straddled_keys[row] = dense_key<key_t>(1 + row);
        indexes_type_ straddled {kernel_, forkunion_allocator_t {no_memory_domain_k}};
        straddled.bound_shards(straddled_bound);
        failures += expect(succeeded(place_shards(straddled)));
        failures += expect(succeeded(straddled.reserve({straddled_bound, connectivity_16_k, dimensions}).status));
        for (std::size_t const first : {std::size_t {0}, batch, 3 * batch}) {
            std::size_t const count = first == batch ? 2 * batch : batch;
            failures += expect(
                succeeded(straddled
                              .insert(std::span<key_t const> {straddled_keys.data() + first, count},
                                      straddled_vectors(nk::range(first, first + count), nk::slice), driver.executor)
                              .status));
        }
        failures += expect((shard_sizes(straddled) == std::vector<std::size_t> {2500, 2500, 2500, 2500, 2000, 2000}));
        std::size_t wrong = 0;
        for (key_t const& key : straddled_keys) wrong += straddled.count(key) != 1;
        failures += expect_eq(wrong, 0u);
    }

    // The filled index restores into a bounded one part by part, each generation a shard again.
    save_config_t const config = dense_indexes_save_config<indexes_type_>();
    auto const [bytes, measure_status] = filled.storage_requirements(config);
    failures += expect(succeeded(measure_status));
    std::vector<std::byte> buffer(bytes);
    failures += expect(succeeded(filled.save(bytes_span_t {buffer}, config, driver.executor)));
    indexes_type_ restored {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    restored.bound_shards(bound);
    failures += expect(succeeded(place_shards(restored)));
    failures += expect(succeeded(restored.load(bytes_view_t {buffer}, load_config_t {}, driver.executor)));
    failures += expect_eq(restored.size(), rows);
    failures += expect(shard_sizes(restored) == shard_sizes(filled));
    failures += expect_eq(miscounted(restored, rows), 0u);
    return failures;
}

/** Four even shards compact in place under a bound of half the entries into two, the kept ones
 *  keeping their IDs and the others staying as idle slots, while a looker counts every key once,
 *  then out of place into a fresh single shard, which empties the source. Every key resolves once
 *  with its vector intact after each, and searches find sampled vectors themselves within noise of
 *  before. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_compact(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using distance_t = typename indexes_type_::distance_t;
    constexpr std::size_t samples_k = 64, noise_percent_k = 10;
    std::size_t failures = 0;
    std::size_t const rows = dense_growth_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    auto& executor = driver.executor;
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_compact"));
    std::vector<key_t> keys(rows);
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
            failures += expect(succeeded(walker.status));
            std::array<key_t, 8> found_keys {};
            std::array<distance_t, 8> found_distances {};
            for (std::size_t sample = 0; sample != samples_k; ++sample) {
                std::size_t const row = sample * (rows / samples_k);
                auto const [result, status] = indexes.search(vectors.view().row(row).as_vector(), walker.handle(), {},
                                                             std::span<key_t> {found_keys},
                                                             std::span {found_distances});
                failures += expect(succeeded(status));
                hits += std::find(found_keys.begin(), found_keys.begin() + result.count, keys[row]) !=
                        found_keys.begin() + result.count;
            }
        }
        return hits;
    };

    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)) && succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions}).status));
    failures += expect(succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), executor).status));
    failures += expect_eq(filled_shards(indexes), 4u);
    std::size_t const hits_before = self_hits(indexes);

    // In place: each kept shard takes one other, the only entries that move and re-select. A looker
    // counts keys throughout, and each moved entry's one locator swaps whole, so every key counts
    // once.
    std::array<indexes_type_*, 1> const in_place {&indexes};
    compact_config_t halves;
    halves.shard_capacity = rows / 2;
    expected<compact_result_t> compacted {};
    bool compacting = true;
    std::size_t miscounted = 0;
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(2))) return failures + expect(false);
    pool.for_threads([&](std::size_t thread) noexcept {
        if (thread == 0) {
            compacted = compact(std::span<indexes_type_* const> {in_place}, indexes, executor, halves);
            fu::standard_atomic_ref<bool>(compacting).store(false, std::memory_order_release);
        }
        else if (thread == 1)
            for (std::size_t row = 0; fu::standard_atomic_ref<bool>(compacting).load(std::memory_order_acquire);
                 row = (row + 1) % rows)
                miscounted += indexes.count(keys[row]) != 1;
    });
    failures += expect_eq(miscounted, 0u);
    auto const [packed, packed_status] = compacted;
    failures += expect_eq(packed_status, status_t::success_k);
    failures += expect(packed.source_shards == 4 && packed.target_shards == 2 && packed.moved_entries == rows / 2);
    failures += expect(packed.graph.reselected_nodes >= rows / 2);
    failures += expect_eq(indexes.shards_count(), 4u);
    failures += expect_eq(filled_shards(indexes), 2u);
    failures += dense_indexes_expect_restored(indexes, vectors, keys, dimensions, executor, seed);
    failures += expect(self_hits(indexes) * 100 + noise_percent_k * samples_k >= hits_before * 100);

    // Geometry is checked once every source is closed, so a mismatch refuses only then, and the
    // dying plan reopens the target and every source it closed.
    indexes_type_ mismatched {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(mismatched)));
    failures += expect(succeeded(mismatched.reserve({rows, connectivity_32_k, dimensions}).status));
    std::array<indexes_type_*, 2> const mixed {&indexes, &mismatched};
    failures += expect_eq(compact(std::span<indexes_type_* const> {mixed}, indexes, executor, halves).status,
                          status_t::config_mismatch_k);
    failures += expect(succeeded(indexes.plan_insert(1, executor).status));
    failures += expect(succeeded(mismatched.plan_insert(1, executor).status));

    // Out of place into a fresh single shard: every entry moves, and the source ends empty.
    indexes_type_ single {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(single.emplace(no_memory_domain_k)));
    auto const [moved, moved_status] = compact(std::span<indexes_type_* const> {in_place}, single, executor);
    failures += expect_eq(moved_status, status_t::success_k);
    failures += expect(moved.source_shards == 4 && moved.target_shards == 1 && moved.moved_entries == rows);
    failures += expect_eq(indexes.size(), 0u);
    failures += dense_indexes_expect_restored(single, vectors, keys, dimensions, executor, seed);
    failures += expect(self_hits(single) * 100 + noise_percent_k * samples_k >= hits_before * 100);
    return failures;
}

/** Every verb is safe beside a clear: searches racing one return their own rows or
 *  @c contended_k, an insertion plan holding claims refuses a clear and an absorb with
 *  @c contended_k until it dies, and a cursor opened before a clear answers @c contended_k. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_clear_race(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using distance_t = typename indexes_type_::distance_t;
    constexpr std::size_t searchers_k = 4, rounds_k = 4;
    std::size_t failures = 0;
    std::size_t const rows = 2 * dense_thread_nodes_k;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_clear_race"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    std::array<dense_indexes_executor<indexes_type_>, 2> drivers;
    for (dense_indexes_executor<indexes_type_> const& driver : drivers)
        if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows, connectivity_16_k, dimensions}).status));

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(searchers_k + 1))) return failures + expect(false);
    std::size_t foreign_count = 0, refused_count = 0;
    fu::standard_atomic_ref<std::size_t> const foreign {foreign_count}, refused {refused_count};
    for (std::size_t round = 0; round != rounds_k; ++round) {
        failures += expect(
            succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), drivers[0].executor).status));
        bool cleared = false;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                while (indexes.clear() == status_t::contended_k) std::this_thread::yield();
                fu::standard_atomic_ref<bool>(cleared).store(true, std::memory_order_release);
                return;
            }
            dense_indexes_row_driver<indexes_type_> walker {indexes, drivers[0]};
            if (failed(walker.status)) return foreign.add(1);
            std::array<key_t, 8> found_keys {};
            std::array<distance_t, 8> found_distances {};
            for (std::size_t row = thread; !fu::standard_atomic_ref<bool>(cleared).load(std::memory_order_acquire);
                 row = (row + searchers_k) % rows) {
                auto const [result, status] = indexes.search(vectors.view().row(row).as_vector(), walker.handle(), {},
                                                             std::span<key_t> {found_keys},
                                                             std::span {found_distances});
                if (status == status_t::contended_k) refused.add(1);
                else if (failed(status)) foreign.add(1);
                else
                    for (std::size_t found = 0; found != result.count; ++found)
                        foreign.add(dense_key_number(found_keys[found]) < 1 ||
                                    dense_key_number(found_keys[found]) > rows);
            }
        });
        failures += expect_eq(indexes.size(), 0u);
        failures += expect_eq(indexes.count(keys[0]), 0u);
    }
    failures += expect_eq(foreign_count, 0u);

    // Claims a plan holds refuse every close, so neither a clear nor an absorb frees them.
    failures += expect(
        succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), drivers[0].executor).status));
    indexes_type_ donor {kernel_, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(donor)));
    {
        auto [plan, plan_status] = indexes.plan_insert(1, drivers[0].executor);
        failures += expect(succeeded(plan_status));
        failures += expect_eq(indexes.clear(), status_t::contended_k);
        failures += expect_eq(indexes.absorb(donor, drivers[1].executor).status, status_t::contended_k);
    }
    failures += expect_eq(indexes.size(), rows);

    // A cursor reads under the epoch it opened in, and a clear since ends it.
    typename indexes_type_::cursor_t cursor {indexes};
    std::array<key_t, 4> read_keys {};
    auto const [before, before_status] = cursor.read(indexes, std::span<key_t> {read_keys}, rows_span_t {});
    failures += expect(succeeded(before_status) && before == read_keys.size());
    failures += expect_eq(indexes.clear(), status_t::success_k);
    auto const [after, after_status] = cursor.read(indexes, std::span<key_t> {read_keys}, rows_span_t {});
    failures += expect(after_status == status_t::contended_k && after == 0);
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
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(row / tokens_per_document);
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

template <                                          //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,    //
    typename waiter_type_ = fu::standard_yield_t,   //
    cpu_class_t cpu_class_ = cpu_class_t::serial_k, //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
    requires dense_cell_testable<dtype_, kernel_, numkong_capabilities_of(cpu_class_)>
static std::size_t test_ram_indexes_configuration(char const* name, u64_t seed) {

    using indexes_t = ram_indexes_for<dtype_, cpu_class_, waiter_type_, atomic_ref_>;
    if (!dense_cell_admitted<numkong_capabilities_of(cpu_class_), atomic_ref_>())
        return report(name, "capabilities missing at runtime");

    // The suite is composed here, not reflected: every call below is a capability claim
    // this sharded, executor-driven category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_dense_indexes_guards<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_duplicate_keys<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_save_load<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_part_load<indexes_t, kernel_>(seed);
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
    failures_count += test_ram_indexes_compact<indexes_t, kernel_>(seed);
    failures_count += test_ram_indexes_clear_race<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_split<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_growth<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_products<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_indexer_search<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_maxsim<indexes_t, kernel_>(seed);
    failures_count += test_dense_indexes_attend<indexes_t, kernel_>(seed);
    failures_count += test_ram_indexes_filtered_algebra<indexes_t, kernel_>(seed);
    return report(name, failures_count);
}

/** The exact scan and MaxSim alone over shards the matrix engine of @p cpu_class_ streams, skipped
 *  where the build lacks the kit or the host its capabilities; only dot kernels offer MaxSim. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, cpu_class_t cpu_class_>
std::size_t test_ram_scan_configuration(char const* name, u64_t seed) {
    constexpr nk_capability_t caps_k = numkong_capabilities_of(cpu_class_);
    using indexes_t = ram_indexes_for<dtype_, cpu_class_>;
    if constexpr (!dense_cell_testable<dtype_, kernel_, caps_k>)
        return report(name, "kit not compiled for this target");
    else if (!dense_cell_admitted<caps_k, fu::standard_atomic_ref>())
        return report(name, "capabilities missing at runtime");
    else
        return report(name, test_dense_indexes_scan<indexes_t, kernel_>(seed) +
                                test_dense_indexes_maxsim<indexes_t, kernel_>(seed));
}

} // namespace unum::usearch::test::dense
