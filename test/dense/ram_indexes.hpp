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
 *      places nothing, and an absorb meeting a batch in flight answers the same and leaves the
 *      donor whole; every entry that was placed resolves exactly once.
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
    indexes_type_ recipient {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    indexes_type_ donor {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(recipient)) && succeeded(place_shards(donor)));
    for (indexes_type_* built : {&recipient, &donor})
        failures += expect(succeeded(built->reserve({part * (rounds + 1)}).status));
    failures += expect(succeeded(recipient.insert(keys_of(0), rows_of(0), drivers[0].executor).status));
    failures += expect(succeeded(donor.insert(keys_of(part), rows_of(part), drivers[1].executor).status));

    // Racing entries take keys past every part's, so each resolves to the one entry it placed.
    std::vector<key_t> inserted_keys;
    u64_t next_number = keys.size() + 1;
    key_t next_key = dense_key<key_t>(next_number);
    {
        auto [plan, plan_status] = recipient.plan_insert(1, drivers[0].executor);
        failures += expect(succeeded(plan_status));
        failures += expect_eq(plan.insert(0, std::span<key_t const> {&next_key, 1}, vectors(nk::range(0, 1), nk::slice),
                                          to_writable_span(drivers[0].traversals)),
                              status_t::success_k);
        failures += expect_eq(recipient.absorb(donor, {}, drivers[1].executor).status, status_t::contended_k);
        failures += expect_eq(donor.size(), part);
        failures += expect(succeeded(plan.finish().status));
        inserted_keys.push_back(next_key);
        next_key = dense_key<key_t>(++next_number);
    }

    dense_indexes_row_driver<indexes_type_> rows_driver {recipient, drivers[0]};
    failures += expect(succeeded(rows_driver.status));
    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(2))) return failures + expect(false);
    std::size_t refused = 0, unexpected = 0, absorbed_entries = 0;
    for (std::size_t round = 0; round != rounds && !refused; ++round) {
        // A consumed donor stays closed and sealed, so every round absorbs a fresh one.
        if (round) {
            donor =
                indexes_type_ {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
            failures += expect(succeeded(place_shards(donor)));
            failures += expect(succeeded(donor.reserve({part}).status));
            failures += expect(succeeded(
                donor.insert(keys_of(part * (round + 1)), rows_of(part * (round + 1)), drivers[1].executor).status));
        }
        bool absorbing = false, absorbed = false;
        status_t absorb_status = status_t::success_k;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                fu::standard_atomic_ref<bool>(absorbing).store(true, std::memory_order_release);
                do absorb_status = recipient.absorb(donor, {}, drivers[1].executor).status;
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
                        inserted_keys.push_back(next_key);
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
        absorbed_entries += part;
    }
    failures += expect(refused > 0);
    failures += expect_eq(unexpected, 0u);
    failures += expect_eq(recipient.size(), part + absorbed_entries + inserted_keys.size());
    std::size_t miscounted = 0;
    for (std::size_t entry = 0; entry != part + absorbed_entries; ++entry)
        miscounted += recipient.count(keys[entry]) != 1;
    for (key_t const key : inserted_keys) miscounted += recipient.count(key) != 1;
    failures += expect_eq(miscounted, 0u);
    return failures;
}

/** A batch planned and dropped before its first step gives back the executor and every shard's
 *  claim, so planning the same batch again grows no shard and the entries then are placed. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_abandoned_plan(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({dense_thread_nodes_k}).status));

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

/** SAMPLE over shards: one seed draws the same keys at the same bits twice; a focus key draws
 *  first; a call wanting fewer draws a prefix of one wanting more, parents included; and a call
 *  shown the first one's draws never draws them again, hanging every draw from a shown point or
 *  from an earlier draw. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_sample(u64_t seed) {
    using scalar_t = typename indexes_type_::index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    std::size_t failures = 0;
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    std::size_t const rows = 192;
    dense_indexes_executor<indexes_type_> driver;
    if (failed(driver.status)) return failures + expect(false);
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows}).status));
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "indexes_sample"));
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row);
    failures += expect(succeeded(indexes.insert(keys, vectors.view(), driver.executor).status));

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
        failures += expect(succeeded(status));
        drawn.written = result.written;
        for (std::size_t draw = 0; draw != drawn.written; ++draw) {
            failures += expect_eq(drawn.entry_ordinals[draw], 0u);
            for (std::size_t column = 0; column != 3; ++column)
                failures += expect(std::isfinite(drawn.coordinates[draw * 3 + column]));
        }
        return drawn;
    };
    auto among = [](std::span<key_t const> haystack, key_t const& key) {
        return std::find(haystack.begin(), haystack.end(), key) != haystack.end();
    };

    drawn_t const first = draw(64, {}, {}, {}), again = draw(64, {}, {}, {});
    failures += expect(first.written > 0);
    failures += expect_eq(again.written, first.written);
    failures += expect(first.keys == again.keys && first.parents == again.parents);
    failures += expect(
        std::memcmp(first.coordinates.data(), again.coordinates.data(), first.coordinates.size() * sizeof(f32_t)) == 0);

    key_t const focus = keys[7];
    drawn_t const narrow = draw(16, {}, {}, std::span<key_t const>(&focus, 1));
    drawn_t const wide = draw(48, {}, {}, std::span<key_t const>(&focus, 1));
    failures += expect(narrow.written > 0 && narrow.keys[0] == focus && wide.keys[0] == focus);
    failures += expect(narrow.written <= wide.written);
    for (std::size_t draw = 0; draw != min_of(narrow.written, wide.written); ++draw)
        failures += expect(narrow.keys[draw] == wide.keys[draw] && narrow.parents[draw] == wide.parents[draw] &&
                           narrow.distances[draw] == wide.distances[draw]);

    std::size_t const shown_count = min_of<std::size_t>(16, first.written);
    std::span<key_t const> const shown_keys(first.keys.data(), shown_count);
    std::span<f32_t const> const shown_coordinates(first.coordinates.data(), shown_count * 3);
    drawn_t const zoomed = draw(32, shown_keys, shown_coordinates, {});
    failures += expect(zoomed.written > 0);
    for (std::size_t draw = 0; draw != zoomed.written; ++draw) {
        failures += expect(!among(shown_keys, zoomed.keys[draw]));
        std::span<key_t const> const earlier(zoomed.keys.data(), draw);
        failures += expect(among(shown_keys, zoomed.parents[draw]) || among(earlier, zoomed.parents[draw]));
    }
    return failures;
}

/** Past @c shard_capacity, entries open generations of shards on the same placements: one batch
 *  fills as many as it needs, a batch crossing the bound tops the open generation up and is placed
 *  whole, one entry past a full generation opens the next, and a load of the filled index places
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
        return succeeded(place_shards(indexes)) && succeeded(indexes.reserve({bound}).status) &&
               indexes.shard_capacity() == bound;
    };

    // One batch over two empty shards fills two generations and half a third.
    indexes_type_ filled {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(bounded(filled));
    failures += expect(succeeded(filled.insert(keys_of(0, rows), rows_of(0, rows), driver.executor).status));
    failures += expect_eq(filled.size(), rows);
    failures += expect((shard_sizes(filled) == std::vector<std::size_t> {1000, 1000, 1000, 1000, 500, 500}));
    failures += expect_eq(miscounted(filled, rows), 0u);

    // 750 entries a shard, then a batch of 1000 tops both up to the bound and opens a generation
    // for the other 500, whole, and a search reaches the new generation's entries.
    indexes_type_ crossed {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
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
        auto const [result, status] = crossed.search(vectors.view().row(2499).as_vector(),
                                                     std::span<key_t> {found_keys}, std::span {found_distances}, {},
                                                     rows_driver.handle());
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
        indexes_type_ straddled {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        straddled.bound_shards(straddled_bound);
        failures += expect(succeeded(place_shards(straddled)));
        failures += expect(succeeded(straddled.reserve({straddled_bound}).status));
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
    indexes_type_ restored {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
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
                auto const [result, status] = indexes.search(vectors.view().row(row).as_vector(),
                                                             std::span<key_t> {found_keys}, std::span {found_distances},
                                                             {}, walker.handle());
                failures += expect(succeeded(status));
                hits += std::find(found_keys.begin(), found_keys.begin() + result.count, keys[row]) !=
                        found_keys.begin() + result.count;
            }
        }
        return hits;
    };

    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)) && succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows}).status));
    failures += expect(succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), executor).status));
    failures += expect_eq(filled_shards(indexes), 4u);
    std::size_t const hits_before = self_hits(indexes);

    // In place: each kept shard takes one other, the only entries that move and re-select. A looker
    // counts keys throughout, and each moved entry's key moves and its old shard forgets it inside
    // one window, so every key counts exactly once.
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

    // A source of another geometry refuses the plan before anything closes, so every index stays
    // open.
    indexes_type_ mismatched {kernel_, {dimensions, connectivity_32_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(mismatched)));
    failures += expect(succeeded(mismatched.reserve({rows}).status));
    std::array<indexes_type_*, 2> const mixed {&indexes, &mismatched};
    failures += expect_eq(compact(std::span<indexes_type_* const> {mixed}, indexes, executor, halves).status,
                          status_t::config_mismatch_k);
    failures += expect(succeeded(indexes.plan_insert(1, executor).status));
    failures += expect(succeeded(mismatched.plan_insert(1, executor).status));

    // Out of place into a fresh single shard: every entry moves, and the source ends empty.
    indexes_type_ single {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
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
    indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(indexes)));
    failures += expect(succeeded(indexes.reserve({rows}).status));

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
                auto const [result, status] = indexes.search(vectors.view().row(row).as_vector(),
                                                             std::span<key_t> {found_keys}, std::span {found_distances},
                                                             {}, walker.handle());
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
    indexes_type_ donor {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(place_shards(donor)));
    {
        auto [plan, plan_status] = indexes.plan_insert(1, drivers[0].executor);
        failures += expect(succeeded(plan_status));
        failures += expect_eq(indexes.clear(), status_t::contended_k);
        failures += expect_eq(indexes.absorb(donor, {}, drivers[1].executor).status, status_t::contended_k);
    }
    failures += expect_eq(indexes.size(), rows);

    // A cursor reads under the epoch it opened in, and a clear since ends it.
    typename indexes_type_::cursor_t cursor {forkunion_allocator_t {no_memory_domain_k}};
    failures += expect(succeeded(cursor.open(indexes, {})));
    std::array<key_t, 4> read_keys {};
    auto const [before, before_status] = cursor.pull(indexes, std::span<key_t> {read_keys}, nullptr, nullptr);
    failures += expect(succeeded(before_status) && before == read_keys.size());
    failures += expect_eq(indexes.clear(), status_t::success_k);
    auto const [after, after_status] = cursor.pull(indexes, std::span<key_t> {read_keys}, nullptr, nullptr);
    failures += expect(after_status == status_t::contended_k && after == 0);
    return failures;
}

/** The filtered algebra verbs keep only what their predicate admits: products name even documents
 *  alone under a walk and a scan alike, MaxSim ranks even documents at their exact scores under a
 *  walk and a scan alike, and attention folds no key a predicate rejects. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_filtered_algebra(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using dot_result_t = typename indexes_type_::dot_result_t;
    using vector_view_t = typename index_t::vector_view_t;
    using search_config_t = typename indexes_type_::search_config_t;
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else {
        constexpr std::size_t rows_count = dense_thread_nodes_k, tokens_per_document = 4;
        constexpr std::size_t documents_count = rows_count / tokens_per_document;
        constexpr std::size_t tokens_per_query = 4, queries_count = 2, wanted = 5;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::size_t failures = 0;
        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return failures + expect(false);
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(indexes)));
        failures += expect(succeeded(indexes.reserve({rows_count}).status));
        auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "filtered_algebra"));
        std::array<key_t, rows_count> keys;
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(row / tokens_per_document);
        failures += expect(
            succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));
        auto const admit_even = quad_predicate(
            [](key_t key) noexcept { return key % 2 ? admission_t::reject_k : admission_t::admit_k; });
        search_config_t const exhaustive {static_cast<typename index_t::candidates_count_t>(rows_count)};
        search_config_t scan;
        scan.traversal = traversal_t::scan_k;

        auto const left = vectors(nk::range(std::size_t {0}, queries_count), nk::slice);
        std::vector<key_t> product_keys(queries_count * rows_count);
        std::vector<dot_result_t> products(queries_count * rows_count);
        std::array<std::size_t, queries_count + 1> offsets {};
        std::array<std::size_t, queries_count> scored {};
        products_tape<key_t, dot_result_t> const pairs {std::span {product_keys}, std::span {products},
                                                        std::span {offsets}, rows_count};
        for (search_config_t const config : {exhaustive, scan}) {
            offsets[0] = 0;
            failures += expect(succeeded(indexes
                                             .multiply_if(left, pairs, std::span {scored}, config, admit_even,
                                                          driver.executor, to_writable_span(driver.traversals))
                                             .status));
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const first = offsets[query], kept = offsets[query + 1] - first;
                failures += expect(scored[query] > 0 && scored[query] <= rows_count / 2);
                failures += expect_eq(kept, min_of(scored[query], rows_count));
                for (std::size_t pair = 0; pair != kept; ++pair)
                    failures += expect(product_keys[first + pair] % 2 == 0);
            }
        }

        // Every even document's exact score: each query row's best dot among its tokens, summed.
        auto const queries = vectors(nk::range(std::size_t {0}, queries_count * tokens_per_query), nk::slice);
        std::array<std::array<f64_t, documents_count>, queries_count> exact {};
        for (std::size_t row = 0; row != queries_count * tokens_per_query; ++row)
            for (std::size_t document = 0; document != documents_count; ++document) {
                f64_t best = 0;
                for (std::size_t token = 0; token != tokens_per_document; ++token)
                    best = max_of(best, static_cast<f64_t>(dense_native_dot(
                                            queries.row(row).as_vector(),
                                            vectors.view().row(document * tokens_per_document + token).as_vector())));
                exact[row / tokens_per_query][document] += best;
            }
        std::vector<key_t> found_keys(queries_count * wanted);
        std::vector<f64_t> found_scores(queries_count * wanted);
        std::array<std::size_t, queries_count + 1> query_offsets;
        for (std::size_t query = 0; query != query_offsets.size(); ++query)
            query_offsets[query] = query * tokens_per_query;
        products_tape<key_t, f64_t> const matches {std::span {found_keys}, std::span {found_scores},
                                                   std::span {offsets}, wanted};
        // An exhaustive walk and a scan nominate every even document alike.
        error_stats_t score_errors;
        for (search_config_t const config : {exhaustive, scan}) {
            offsets[0] = 0;
            failures += expect(succeeded(indexes
                                             .maxsim_if(queries, std::span<std::size_t const> {query_offsets},
                                                        std::span {scored}, matches, {config, wanted}, admit_even,
                                                        driver.executor, to_writable_span(driver.traversals))
                                             .status));
            for (std::size_t query = 0; query != queries_count; ++query) {
                std::size_t const first = offsets[query], kept = offsets[query + 1] - first;
                failures += expect_eq(kept, wanted) + expect_eq(scored[query], (documents_count + 1) / 2);
                for (std::size_t rank = 0; rank != kept; ++rank) {
                    std::size_t const document = static_cast<std::size_t>(found_keys[first + rank]);
                    failures += expect(document < documents_count && document % 2 == 0);
                    if (document < documents_count)
                        score_errors.add(found_scores[first + rank], exact[query][document]);
                }
            }
        }
        failures += expect_errors("filtered maxsim scores", comparison_family_t::approximate_k,
                                  comparison_verdict_t::assert_k, score_errors, 0);

        // A lookup that never resolves would refuse any key that folds.
        auto const no_rows = [](std::span<key_t const>, std::span<vector_view_t>) noexcept {};
        auto const reject_all = quad_predicate([](key_t) noexcept { return admission_t::reject_k; });
        auto outputs = nk::matrix<f32_t>::zeros({queries_count, dimensions}).value;
        std::array<softmax_state_t, queries_count> carry;
        failures += expect(
            succeeded(indexes
                          .attend_if(left, 1, no_rows, 1.0f, outputs.span(), std::span {carry}, exhaustive, reject_all,
                                     driver.executor, to_writable_span(driver.traversals))
                          .status));
        for (softmax_state_t const& state : carry) failures += expect_eq(state.partition, 0.0f);
        return failures;
    }
}

/** A MaxSim scan scores every key exactly once though its rows span shards: keeping every key, it
 *  writes each document once at its exact total and counts each once. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_maxsim_spanning(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using search_config_t = typename indexes_type_::search_config_t;
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else {
        constexpr std::size_t rows_count = dense_thread_nodes_k, tokens_per_document = 4, tokens_per_query = 4;
        constexpr std::size_t documents_count = rows_count / tokens_per_document;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::size_t failures = 0;
        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return failures + expect(false);
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(indexes)));
        auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "maxsim_spanning"));
        std::array<key_t, rows_count> keys;
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(1 + row / tokens_per_document);
        failures += expect(
            succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));
        // One batch rotates over both shards, so every document keeps rows in each.
        std::size_t spanning = 0;
        for (std::size_t row = 0; row != rows_count; row += tokens_per_document)
            spanning += indexes.shard(0).count(keys[row]) && indexes.shard(1).count(keys[row]);
        failures += expect_eq(spanning, documents_count);

        std::array<f64_t, documents_count> exact {};
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
        std::vector<key_t> found_keys(rows_count);
        std::vector<f64_t> found_scores(rows_count);
        std::array<std::size_t, 1> scored {};
        std::array<std::size_t, 2> offsets {};
        std::array<std::size_t, 2> const query_offsets {0, tokens_per_query};
        products_tape<key_t, f64_t> const matches {std::span {found_keys}, std::span {found_scores},
                                                   std::span {offsets}, rows_count};
        failures += expect(
            succeeded(indexes
                          .maxsim(queries, std::span<std::size_t const> {query_offsets}, std::span {scored}, matches,
                                  {scan, 0}, driver.executor, to_writable_span(driver.traversals))
                          .status));
        failures += expect_eq(scored[0], documents_count) + expect_eq(offsets[1], documents_count);
        std::array<std::size_t, documents_count> placed {};
        error_stats_t score_errors;
        for (std::size_t rank = 0; rank != min_of(offsets[1], documents_count); ++rank) {
            std::size_t const document = dense_key_number(found_keys[rank]) - 1;
            failures += expect(document < documents_count);
            if (document >= documents_count) continue;
            ++placed[document];
            score_errors.add(found_scores[rank], exact[document]);
        }
        for (std::size_t const times : placed) failures += expect_eq(times, 1u);
        return failures + expect_errors("spanning maxsim scores", comparison_family_t::approximate_k,
                                        comparison_verdict_t::assert_k, score_errors, 0);
    }
}

/** Exact scans of the algebra verbs against brute force: a scan's products keep the widest dots,
 *  each exactly the native dot of a distinct stored row, largest first, and a scan's attention
 *  matches exact full attention, every query row its own head; only the dot kernel offers both. */
template <typename indexes_type_, nk_kernel_kind_t kernel_>
static std::size_t test_ram_indexes_exact_scans(u64_t seed) {
    using index_t = typename indexes_type_::index_t;
    using scalar_t = typename index_t::scalar_t;
    using key_t = typename indexes_type_::key_t;
    using dot_result_t = typename indexes_type_::dot_result_t;
    using vector_view_t = typename index_t::vector_view_t;
    using search_config_t = typename indexes_type_::search_config_t;
    if constexpr (kernel_ != nk_kernel_dot_k) return 0;
    else {
        constexpr std::size_t rows_count = dense_thread_nodes_k, left_count = 16, wanted = 8, attended_count = 8;
        std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
        std::size_t failures = 0;
        dense_indexes_executor<indexes_type_> driver;
        if (failed(driver.status)) return failures + expect(false);
        indexes_type_ indexes {kernel_, {dimensions, connectivity_16_k}, forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(place_shards(indexes)));
        failures += expect(succeeded(indexes.reserve({rows_count}).status));
        auto const vectors = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "exact_scans"));
        std::array<key_t, rows_count> keys;
        for (std::size_t row = 0; row != rows_count; ++row) keys[row] = dense_key<key_t>(row);
        failures += expect(
            succeeded(indexes.insert(std::span<key_t const> {keys}, vectors.view(), driver.executor).status));
        search_config_t scan;
        scan.traversal = traversal_t::scan_k;

        // As wide as a matrix engine's tile, so a tiled class streams the batch through it.
        auto const left = vectors(nk::range(std::size_t {0}, left_count), nk::slice);
        std::vector<key_t> product_keys(left_count * wanted);
        std::vector<dot_result_t> products(left_count * wanted);
        std::array<std::size_t, left_count + 1> offsets {};
        std::array<std::size_t, left_count> scored {};
        products_tape<key_t, dot_result_t> const pairs {std::span {product_keys}, std::span {products},
                                                        std::span {offsets}, wanted};
        failures += expect(succeeded(
            indexes
                .multiply(left, pairs, std::span {scored}, scan, driver.executor, to_writable_span(driver.traversals))
                .status));
        for (std::size_t row = 0; row != left_count; ++row) {
            std::size_t const first = offsets[row];
            failures += expect_eq(scored[row], wanted) + expect_eq(offsets[row + 1] - first, wanted);
            if (offsets[row + 1] - first != wanted) continue;
            failures += dense_expect_product_row<rows_count>(
                left.row(row).as_vector(), vectors.view(),
                nk::vector_view<key_t> {std::span {product_keys}.subspan(first, wanted)},
                nk::vector_view<dot_result_t> {std::span {products}.subspan(first, wanted)},
                [](key_t key) noexcept { return dense_key_number(key); });
            // The ranking reads the finalized distance, so only a near tie at the cut may swap.
            std::array<f64_t, rows_count> exact;
            for (std::size_t stored = 0; stored != rows_count; ++stored)
                exact[stored] = static_cast<f64_t>(
                    dense_native_dot(left.row(row).as_vector(), vectors.view().row(stored).as_vector()));
            std::sort(exact.begin(), exact.end(), std::greater<f64_t> {});
            for (std::size_t rank = 0; rank != wanted; ++rank) {
                f64_t const kept = static_cast<f64_t>(products[first + rank]);
                failures += expect(kept + 1e-3 * max_of(std::fabs(exact[wanted - 1]), 1.0) >= exact[wanted - 1]);
                if (rank) failures += expect(static_cast<f64_t>(products[first + rank - 1]) >= kept);
            }
        }

        auto const values = make_vectors<scalar_t>(rows_count, dimensions, stream_key(seed, "exact_scans", 1));
        auto const queries = make_vectors<scalar_t>(attended_count, dimensions, stream_key(seed, "exact_scans", 2));
        auto value_of = [&](std::size_t row) noexcept { return values.view().row(row).as_vector(); };
        auto lookup = [&](std::span<key_t const> batch, std::span<vector_view_t> rows) noexcept {
            for (std::size_t lane = 0; lane != batch.size(); ++lane)
                rows[lane] = value_of(dense_key_number(batch[lane]));
        };
        f32_t const score_scale = dense_attend_scale(
            dense_oracle_dots(dense_widened_rows(queries.view(), index_t::vector_dtype_k),
                              dense_widened_rows(vectors.view(), index_t::vector_dtype_k), dimensions));
        auto outputs = nk::matrix<f32_t>::zeros({attended_count, dimensions}).value;
        std::vector<softmax_state_t> carry(attended_count);
        auto const [attended, attend_status] = drive_attend(indexes, queries.view(), 1, lookup, score_scale,
                                                            outputs.span(), std::span {carry}, scan, driver);
        failures += expect(succeeded(attend_status));
        failures += expect_eq(attended.count, attended_count);
        failures += dense_expect_full_attention(vectors.view(), queries.view(), 1, value_of, score_scale,
                                                outputs.view(), std::span<softmax_state_t const> {carry});
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
    failures_count += dense_expect_duplicates_build<indexes_t, kernel_>(seed);
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
    failures_count += test_ram_indexes_sample<indexes_t, kernel_>(seed);
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
    failures_count += test_ram_indexes_maxsim_spanning<indexes_t, kernel_>(seed);
    failures_count += test_ram_indexes_exact_scans<indexes_t, kernel_>(seed);
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
