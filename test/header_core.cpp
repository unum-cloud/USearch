/**
 *  @file test/header_core.cpp
 *  @author Ash Vardanian
 *  @date November 5, 2026
 *  @brief The dependency-free core's proof: the basic RAM index and its sharded index build, fill,
 *      search and recut with no sibling library on the include path.
 *
 *  Values are single floats scored by their absolute difference, the smallest metric the concepts
 *  admit, and every container takes @c std::allocator, so the unit builds with only @c include/
 *  and @c c/ on its path. The suites hold the behavioral depth, while this unit shows the tier
 *  builds and runs on the standard library alone.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstring> // `std::memcpy` - the codec's bytes

#include <array>       // `std::array` - the entries, keys and results
#include <memory>      // `std::allocator`, `std::unique_ptr` - the indexes never move
#include <new>         // `std::nothrow` - the indexes live on the heap
#include <span>        // `std::span` - the metric and codec surfaces
#include <type_traits> // `std::is_same_v` - the sharded index's allocator

#include "usearch/basic/ram_index.hpp"
#include "usearch/basic/ram_indexes.hpp"

using namespace unum::usearch;
namespace util = unum::usearch::utilities;

/** The absolute difference of two floats, staged one round at a time. */
struct scalar_metric_t {
    util::f32_t query_ = 0;
    std::array<util::f64_t, 64> differences_ {};
    std::size_t staged_ = 0;

    void reset(util::f32_t query) noexcept { query_ = query, staged_ = 0; }
    basic::metric_staging_t push(util::f32_t value) noexcept {
        util::f64_t const difference = static_cast<util::f64_t>(value) - query_;
        differences_[staged_++] = difference < 0 ? -difference : difference;
        return staged_ != differences_.size() ? basic::metric_staging_t::staged_k : basic::metric_staging_t::refused_k;
    }
    void drain(std::span<util::f64_t> distances, util::f64_t) noexcept {
        for (std::size_t lane = 0; lane != distances.size(); ++lane) distances[lane] = differences_[lane];
        staged_ = 0;
    }
};

/** A float's four bytes as stored. */
struct scalar_codec_t {
    std::size_t storage_requirements(util::f32_t const&) const noexcept { return sizeof(util::f32_t); }
    util::status_t save(util::f32_t const& value, util::bytes_span_t output) const noexcept {
        if (output.size() < sizeof(value)) return util::status_code_t::overflow_k;
        std::memcpy(output.data(), &value, sizeof(value));
        return util::status_code_t::success_k;
    }
    util::status_t load(util::bytes_view_t input, util::f32_t& value) const noexcept {
        if (input.size() != sizeof(value)) return util::status_code_t::malformed_k;
        std::memcpy(&value, input.data(), sizeof(value));
        return util::status_code_t::success_k;
    }
};

using shard_t = basic::ram_index<util::f32_t, util::u64_t, util::u32_t>;
using indexes_t = basic::ram_indexes<shard_t>;

static_assert(basic::a_metric<scalar_metric_t, util::f32_t, util::f64_t>);
static_assert(basic::a_value_codec<scalar_codec_t, util::f32_t>);
static_assert(std::is_same_v<indexes_t::allocator_t, std::allocator<util::byte_t>>);

int main() {
    constexpr std::size_t entries_k = 64, connectivity_k = 8;
    std::array<util::f32_t, entries_k> values {};
    std::array<util::u64_t, entries_k> keys {};
    for (std::size_t entry = 0; entry != entries_k; ++entry)
        values[entry] = static_cast<util::f32_t>(entry), keys[entry] = 1000 + entry;
    scalar_metric_t metric;
    scalar_codec_t const codec;
    basic::keep_diverse_t const selector {connectivity_k};
    shard_t::traversal_context_t context {7};
    if (util::failed(shard_t::reserve_traversal(context, connectivity_k, {}))) return 1;

    // A lone shard, filled and searched.
    std::unique_ptr<shard_t> const lone {new (std::nothrow) shard_t};
    if (!lone || util::failed(lone->reserve(entries_k).status)) return 1;
    std::array<util::f32_t, entries_k> lone_values = values;
    if (util::failed(lone->insert({lone_values}, {keys}, metric, {}, context, selector).status)) return 1;
    auto const [lone_result, lone_status] = lone->search(util::f32_t {5}, {}, metric, context);
    if (util::failed(lone_status) || !lone_result.count) return 1;

    // Two shards on the standard allocator, filled through one plan.
    std::unique_ptr<indexes_t> const sharded {new (std::nothrow) indexes_t {std::allocator<util::byte_t> {}}};
    if (!sharded) return 1;
    indexes_t& indexes = *sharded;
    for (std::size_t shard = 0; shard != 2; ++shard)
        if (util::failed(indexes.emplace(std::allocator<util::byte_t> {}))) return 1;
    auto [plan, plan_status] = indexes.plan_insert(entries_k);
    if (util::failed(plan_status)) return 1;
    plan.insert(0, std::span<util::f32_t> {values}, std::span<util::u64_t const> {keys}, metric, context, selector);
    if (util::failed(plan.finish()) || indexes.size() != entries_k) return 1;

    // A merged search below one snapshot finds the exact entry first.
    std::array<util::u32_t, 2> watermarks {};
    if (util::failed(indexes.snapshot(watermarks).status)) return 1;
    std::array<util::u64_t, 4> found_keys {};
    std::array<util::f64_t, 4> found_distances {};
    util::dummy_predicate_t predicate;
    auto const [found, found_status] = indexes.search_if(util::f32_t {17}, watermarks, found_keys, found_distances, {},
                                                         predicate, metric, context);
    if (util::failed(found_status) || !found || found_keys[0] != 1017) return 1;

    // A recut of the index into one shard keeps every entry.
    std::array<indexes_t*, 1> const alone {&indexes};
    indexes_t::reshard_config_t one_shard;
    one_shard.shard_capacity = entries_k;
    auto const [recut, recut_status] = indexes.reshard(alone, alone, one_shard, selector, codec, metric, context);
    if (util::failed(recut_status) || indexes.size() != entries_k) return 1;
    std::size_t filled = 0;
    for (shard_t const& one : indexes.live_shards()) filled += one.size() != 0;
    return filled == 1 ? 0 : 1;
}
