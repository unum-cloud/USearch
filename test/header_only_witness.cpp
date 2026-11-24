/**
 *  @file test/header_only_witness.cpp
 *  @author Ash Vardanian
 *  @date August 27, 2026
 *  @brief The header-only tier's proof.
 *
 *  Everything under @c utilities/, @c basic/ & @c geospatial/ compiles & runs from headers alone,
 *  and this translation unit instantiates the whole public surface to witness it. The RAM tier of
 *  @c basic/ needs the standard library and the OS headers only, which @c header_core.cpp proves,
 *  while the rest of the tier also requires the header-only ForkUnion & NumKong.
 *
 *  @c utilities/serialization.hpp is deliberately absent: it consumes yyjson for the
 *  SafeGraphs JSON header, so it sits outside the tier this file vouches for. Checks are
 *  deliberately minimal - instantiate, insert, search, round-trip - existence is the
 *  assertion; behavioral depth lives in the fmt-linked suite.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <array>       // `std::array` - stored tags & batch overloads
#include <bit>         // `std::bit_cast` - uuid keys from words
#include <memory>      // `std::allocator` - every container names its allocator, `std::unique_ptr` - the indexes
#include <new>         // `std::nothrow` - indexes live on the heap
#include <span>        // `std::span` - metric & codec surfaces
#include <string_view> // `std::string_view` - the status vocabulary probe
#include <type_traits> // `std::is_same_v` - the columns' witness
#include <vector>      // `std::vector` - the five snapshot columns

#include <forkunion/capabilities.hpp> // `fu::preferred_cache_hints_t` - the read hints

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_sorts.hpp"
#include "usearch/utilities/ram_tables.hpp"
#include "usearch/dense/types.hpp"
#include "usearch/basic/ram_index.hpp"
#include "usearch/geospatial/ram_metric.hpp"

using namespace unum::usearch;
namespace util = unum::usearch::utilities;

/** The geospatial tier's witnesses: its metric, codec & index alias satisfy the basic concepts. */
static_assert(basic::metric<geospatial::chord_metric_t, geospatial::earth_observation_t, nk_u64_t>);
static_assert(basic::value_codec<geospatial::earth_observation_codec_t, geospatial::earth_observation_t>);
static_assert(basic::selector<geospatial::surface_keep_octants_t, geospatial::ram_index_t, geospatial::chord_metric_t>);

/** Hamming distance over eight-byte tags - the smallest metric the concept admits. */
struct header_only_metric_t {
    std::array<char, 8> query_ {};
    std::array<util::u64_t, 64> differences_ {};
    std::size_t staged_ = 0;

    void reset(std::array<char, 8> const& query) noexcept { query_ = query, staged_ = 0; }
    basic::metric_staging_t push(std::array<char, 8> const& value) noexcept {
        util::u64_t difference = 0;
        for (std::size_t position = 0; position != query_.size(); ++position)
            difference += query_[position] != value[position];
        differences_[staged_++] = difference;
        return staged_ != differences_.size() ? basic::metric_staging_t::staged_k : basic::metric_staging_t::refused_k;
    }
    void drain(std::span<util::u64_t> distances, util::u64_t) noexcept {
        for (std::size_t lane = 0; lane != distances.size(); ++lane) distances[lane] = differences_[lane];
        staged_ = 0;
    }
};

/** The same metric with the transparency tag - a compile-only concept witness. */
struct header_only_transparent_metric_t : header_only_metric_t {
    using is_transparent = void;
};

/** Identity-bytes codec for the fixed-width tags. */
struct header_only_codec_t {
    std::size_t storage_requirements(std::array<char, 8> const&) const noexcept { return 8; }
    util::status_t save(std::array<char, 8> const& value, util::bytes_span_t output) const noexcept {
        if (output.size() < value.size()) return util::status_code_t::overflow_k;
        for (std::size_t position = 0; position != value.size(); ++position)
            output[position] = static_cast<std::byte>(value[position]);
        return util::status_code_t::success_k;
    }
    util::status_t load(util::bytes_view_t input, std::array<char, 8>& value) const noexcept {
        if (input.size() != value.size()) return util::status_code_t::malformed_k;
        for (std::size_t position = 0; position != value.size(); ++position)
            value[position] = static_cast<char>(input[position]);
        return util::status_code_t::success_k;
    }
};

using narrow_index_t = basic::ram_index<std::array<char, 8>, util::u64_t, util::u32_t, util::u64_t>;
using wide_index_t = basic::ram_index<std::array<char, 8>, util::uuid128_t, util::u64_t, util::u64_t>;

static_assert(basic::value<std::array<char, 8>>);
static_assert(basic::metric<header_only_metric_t, std::array<char, 8>, util::u64_t>);
static_assert(!basic::transparent_metric<header_only_metric_t>);
static_assert(basic::transparent_metric<header_only_transparent_metric_t>);
static_assert(basic::query<std::array<char, 8>, header_only_metric_t, std::array<char, 8>>);
static_assert(basic::value_codec<header_only_codec_t, std::array<char, 8>>);
static_assert(basic::selector<basic::keep_closest_t, narrow_index_t, header_only_metric_t>);
static_assert(basic::selector<basic::keep_diverse_t, wide_index_t, header_only_metric_t>);
static_assert(util::walk_step_t::exhausted_k != util::walk_step_t::advanced_k);
static_assert(
    std::is_same_v<narrow_index_t::columns_t, basic::columns<util::u64_t, util::u32_t, util::byte_t, util::u64_t>>);
static_assert(std::is_move_constructible_v<narrow_index_t::save_plan_t> &&
              !std::is_copy_constructible_v<narrow_index_t::save_plan_t> &&
              !std::is_copy_assignable_v<narrow_index_t::save_plan_t> &&
              !std::is_move_assignable_v<narrow_index_t::save_plan_t>);
static_assert(std::is_move_constructible_v<narrow_index_t::load_plan_t> &&
              !std::is_copy_constructible_v<narrow_index_t::load_plan_t> &&
              !std::is_copy_assignable_v<narrow_index_t::load_plan_t> &&
              !std::is_move_assignable_v<narrow_index_t::load_plan_t>);
static_assert(
    std::is_same_v<wide_index_t::columns_view_t,
                   basic::columns<util::uuid128_t const, util::u64_t const, util::byte_t const, util::u64_t const>>);

/** The tags the index probes store: distinct leading letters, shared tails. */
static std::array<char, 8> make_tag(std::size_t ordinal) noexcept {
    return {static_cast<char>('a' + ordinal % 26), static_cast<char>('a' + ordinal / 26), 0, 0, 0, 0, 0, 0};
}

int main() {
    // The type vocabulary: statuses, expected & its viewing face, hashes & mixers.
    if (std::string_view {util::status_to_string(util::status_code_t::success_k)}.empty()) return 1;
    auto const [answered, answered_status] = util::expected<util::u32_t> {42, util::status_code_t::success_k};
    if (util::failed(answered_status) || answered != 42) return 1;
    util::u64_t seven = 7;
    auto const [viewed, viewed_status] = util::expected<util::u64_t const&> {&seven, util::status_code_t::success_k};
    if (util::failed(viewed_status) || *viewed != 7) return 1;
    util::uuid128_t const uuid {};
    if (std::hash<util::uuid128_t> {}(uuid) != std::hash<util::uuid128_t> {}(uuid)) return 1;
    if (util::mix_bits(1) == util::mix_bits(2)) return 1;
    if (util::multiply_high(util::u64_t {1} << 32, util::u64_t {1} << 32) != 1) return 1;
    if (util::dtype_named("u64") != util::dtype_t::u64_k) return 1;
    if (std::string_view {dtype_name(util::dtype_t::uuid_k)} != "uuid") return 1;

    // The engine & distribution split, at both widths.
    util::splitmix64_t random_generator {7};
    if (util::uniform_below<util::u32_t> {10}(random_generator) >= 10) return 1;
    if (util::uniform_below<util::u64_t> {10}(random_generator) >= 10) return 1;
    if (util::ceil2(3) != 4) return 1;

    // Sequential containers: the growing buffer's whole accessor family.
    util::growing_buffer<util::u32_t> sequence {util::global_allocator};
    if (util::failed(sequence.reserve(4))) return 1;
    sequence.push_back_reserved(3), sequence.push_back_reserved(1), sequence.push_back_reserved(2);
    if (sequence.front() != 3 || sequence.back() != 2) return 1;
    if (sequence.pop_back() != 2 || sequence.size() != 2) return 1;
    if (util::failed(sequence.push_back(9)) || sequence.back() != 9) return 1;

    std::array<util::u32_t, 4> quad {4, 1, 3, 2};
    util::network_sort(std::span<util::u32_t, 4> {quad});
    if (quad[0] != 1 || quad[3] != 4) return 1;

    util::max_heap<dense::scored_node_t, dense::further_scored_node_t> frontier {util::global_allocator};
    if (util::failed(frontier.reserve(64))) return 1;
    if (util::failed(frontier.insert({dense::node_id_t {1}, dense::distance_t {0.5f}}))) return 1;
    if (util::failed(frontier.insert({dense::node_id_t {2}, dense::distance_t {0.25f}}))) return 1;
    if (frontier.top().node_id != 2) return 1; // The furthest-first comparator surfaces the nearest.
    frontier.pop();

    util::max_heap_dary<8, dense::scored_node_t, dense::further_scored_node_t> wide_frontier {util::global_allocator};
    if (util::failed(wide_frontier.reserve(64))) return 1;
    if (util::failed(wide_frontier.insert({dense::node_id_t {3}, dense::distance_t {0.75f}}))) return 1;
    if (wide_frontier.top().node_id != 3) return 1;

    util::sorted_buffer<dense::scored_node_t, dense::closer_scored_node_t> results {util::global_allocator};
    if (util::failed(results.reserve(4))) return 1;
    results.insert({dense::node_id_t {3}, dense::distance_t {0.75f}}, 4);
    if (results.top().node_id != 3) return 1;

    // Associative containers: the visited set & the concurrent multimap.
    util::growing_hash_set<dense::node_id_t> visited {util::global_allocator};
    if (util::failed(visited.reserve(64))) return 1;
    if (visited.insert(42) != util::membership_t::absent_k) return 1;
    if (!visited.contains(42) || visited.contains(43)) return 1;
    if (visited.insert(std::array<dense::node_id_t, 4> {42, 43, 44, 45}) != 0b1110u) return 1;

    util::chunked_multimap<util::u64_t, util::u32_t> multimap {util::global_allocator};
    if (util::failed(multimap.reserve(2))) return 1;
    multimap.insert(7, 0);
    multimap.insert(7, 1);
    if (multimap.count(7, 2) != 2 || !multimap.contains(7, 2) || multimap.contains(7, 0)) return 1;
    if (util::failed(multimap.reserve(4096)) || multimap.count(7, 2) != 2) return 1; // Growing keeps entries.
    multimap.clear();
    if (multimap.contains(7, 2)) return 1;

    fu::preferred_cache_hints_t const hints;
    hints(&random_generator, fu::prefetch_line_t {});
    hints(&random_generator, fu::demote_line_t {});

    // The basic index, narrow instantiation: the whole public surface, existence-level.
    std::unique_ptr<narrow_index_t> const index_ptr {new (std::nothrow) narrow_index_t};
    if (!index_ptr) return 1;
    narrow_index_t& index = *index_ptr;
    if (util::failed(index.reserve(16).status) || index.capacity() < 16 || !index.memory_usage()) return 1;
    std::array<std::array<char, 8>, 8> tags {};
    std::array<util::u64_t, 8> keys {};
    for (std::size_t tag = 0; tag != tags.size(); ++tag) tags[tag] = make_tag(tag), keys[tag] = 100 + tag;
    header_only_metric_t metric;
    header_only_codec_t const codec;
    narrow_index_t::traversal_context_t context {7};
    std::array<util::u32_t, 8> assigned_ids {};
    if (util::failed(index.insert({tags}, {keys}, metric, {assigned_ids}, context).status)) return 1;
    std::array<std::array<char, 8>, 1> one_more {{make_tag(60)}};
    std::array<util::u64_t, 1> const one_more_key {200};
    if (util::failed(index
                         .template insert<util::synchronization_t::exclusive_k>({one_more}, {one_more_key}, metric, {},
                                                                                context, basic::keep_closest_t {2})
                         .status))
        return 1;
    if (index.size() != 9) return 1;

    // A planned batch stepped on this thread, its context over another allocator than the index's.
    std::array<std::array<char, 8>, 2> planned_tags {{make_tag(61), make_tag(62)}};
    std::array<util::u64_t, 2> const planned_keys {300, 301};
    narrow_index_t::traversal_context_over<std::allocator<char>> planned_context {11};
    if (util::failed(narrow_index_t::reserve_traversal(planned_context, 2, {}))) return 1;
    auto [plan, plan_status] = index.plan_insert(planned_tags.size());
    if (util::failed(plan_status)) return 1;
    plan.insert(0, planned_tags, planned_keys, metric, planned_context, basic::keep_closest_t {2});
    if (plan.finish() != util::status_code_t::success_k || index.size() != 11 || !index.contains(301)) return 1;

    // All three search modes answer, ranked & settled-equal.
    auto const query = make_tag(0);
    auto const [concurrent_result, concurrent_status] = index.search(query, {}, metric, context);
    auto const [immutable_result, immutable_status] = index.template search<util::synchronization_t::immutable_k>(
        query, {}, metric, context);
    auto const [exclusive_result, exclusive_status] = index.template search<util::synchronization_t::exclusive_k>(
        query, {}, metric, context);
    if (util::failed(concurrent_status) || util::failed(immutable_status) || util::failed(exclusive_status)) return 1;
    if (!concurrent_result.count) return 1;
    if (immutable_result.count != concurrent_result.count || exclusive_result.count != concurrent_result.count)
        return 1;

    // Manual wiring, the checked readers, and the keyed lookups.
    auto const [membership, link_status] = index.link(0, 1);
    if (util::failed(link_status)) return 1;
    if (index.unlink(0, 1) != util::status_code_t::success_k) return 1;
    if (index.unlink(0, 1) != util::status_code_t::malformed_k) return 1;
    auto const [first_key, key_status] = index.key_of(0);
    if (util::failed(key_status) || first_key != 100) return 1;
    auto const [first_value, value_status] = index.value_of(0);
    if (util::failed(value_status) || *first_value != index.published_value_of(0)) return 1;
    if (!index.degree_of(0)) return 1;
    if (index.key_of(static_cast<util::u32_t>(index.size())).status != util::status_code_t::malformed_k) return 1;
    auto const [found_node, find_status] = index.find(101);
    if (util::failed(find_status) || found_node != 1) return 1;
    if (!index.contains(200) || index.count(999) != 0) return 1;
    std::size_t found_holders = 0;
    index.find_each(100, [&](util::u32_t) noexcept { ++found_holders; });
    if (found_holders != 1) return 1;

    auto const health = index.health(context);
    if (util::failed(health.status) || !health.connected_components) return 1;

    // The snapshot round-trip needs nothing outside the tier either: five columns in vectors.
    auto const [extents, measure_status] = index.columns_extents(codec);
    if (util::failed(measure_status)) return 1;
    std::vector<util::u64_t> keys_column(extents.nodes);
    std::vector<util::u64_t> neighbor_offsets(extents.nodes + 1);
    std::vector<util::u32_t> neighbor_ids(extents.neighbors);
    std::vector<util::u64_t> value_offsets(extents.nodes + 1);
    std::vector<std::byte> values(extents.value_bytes);
    narrow_index_t::columns_t const columns {keys_column, neighbor_offsets, neighbor_ids, value_offsets, values};
    if (index.save(columns, codec) != util::status_code_t::success_k) return 1;
    std::unique_ptr<narrow_index_t> const restored_ptr {new (std::nothrow) narrow_index_t};
    if (!restored_ptr) return 1;
    narrow_index_t& restored = *restored_ptr;
    if (util::failed(restored.reserve(16).status)) return 1;
    if (restored.load(columns.view(), codec) != util::status_code_t::success_k) return 1;
    if (restored.size() != index.size()) return 1;
    auto const [restored_key, restored_key_status] = restored.key_of(0);
    if (util::failed(restored_key_status) || restored_key != first_key) return 1;
    auto const [restored_degree, restored_degree_status] = restored.degree_of(3);
    auto const [saved_degree, saved_degree_status] = index.degree_of(3);
    if (util::failed(restored_degree_status) || util::failed(saved_degree_status) || restored_degree != saved_degree)
        return 1;

    // The wide instantiation: uuid keys through `std::hash<utilities::uuid128_t>`, 64-bit IDs.
    std::unique_ptr<wide_index_t> const wide_index_ptr {new (std::nothrow) wide_index_t};
    if (!wide_index_ptr) return 1;
    wide_index_t& wide_index = *wide_index_ptr;
    if (util::failed(wide_index.reserve(4).status)) return 1;
    std::array<util::uuid128_t, 2> const wide_keys {std::bit_cast<util::uuid128_t>(std::array<util::u64_t, 2> {1, 2}),
                                                    std::bit_cast<util::uuid128_t>(std::array<util::u64_t, 2> {3, 4})};
    std::array<std::array<char, 8>, 2> wide_tags {{make_tag(1), make_tag(2)}};
    wide_index_t::traversal_context_t wide_context {9};
    if (util::failed(
            wide_index.insert({wide_tags}, {wide_keys}, metric, {}, wide_context, basic::keep_diverse_t {2}).status))
        return 1;
    auto const [wide_node, wide_find_status] = wide_index.find(wide_keys[1]);
    if (util::failed(wide_find_status) || wide_node != 1) return 1;
    if (util::failed(wide_index.search(make_tag(1), {}, metric, wide_context).status)) return 1;

    // Clearing forgets the nodes and keys and keeps the covered chunks.
    std::size_t const wide_capacity = wide_index.capacity();
    if (wide_index.clear() != util::status_code_t::success_k || wide_index.size() ||
        wide_index.capacity() != wide_capacity)
        return 1;
    if (wide_index.contains(wide_keys[1]) || wide_index.get_allocator() != std::allocator<util::byte_t> {}) return 1;
    return 0;
}
