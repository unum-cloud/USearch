/**
 *  @brief The header-only tier's proof: everything under @c utilities/, @c basic/ &
 *      @c geospatial/ compiles & runs with the standard library alone - only @c -Iinclude,
 *      no submodule paths, no third-party targets - and this translation unit instantiates
 *      the whole public surface to witness it.
 *  @author Ash Vardanian
 *  @file test/header_only_witness.cpp
 *  @date August 27, 2026
 *
 *  @c utilities/serialization.hpp is deliberately absent: it consumes yyjson for the
 *  SafeGraphs JSON header, so it sits outside the tier this file vouches for. Checks are
 *  deliberately minimal - instantiate, insert, search, round-trip - existence is the
 *  assertion; behavioral depth lives in the fmt-linked suite.
 */
#include <array>       // `std::array` - stored tags & batch overloads
#include <bit>         // `std::bit_cast` - uuid keys from words
#include <memory>      // `std::allocator` - every container names its allocator
#include <span>        // `std::span` - metric & codec surfaces
#include <string_view> // `std::string_view` - the status vocabulary probe
#include <type_traits> // `std::is_same_v` - the columns' witness
#include <vector>      // `std::vector` - the five snapshot columns

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_tables.hpp"
#include "usearch/utilities/ram_hints.hpp"
#include "usearch/basic/ram_index.hpp"
#include "usearch/geospatial/ram_index.hpp"

using namespace unum::usearch;

/** @brief The geospatial tier's witnesses: its metric, codec & index alias satisfy the basic concepts. */
static_assert(basic_metric<chord_metric_t, earth_fixed_t, i64_t>);
static_assert(basic_value_codec<earth_fixed_codec_t, earth_fixed_t>);
static_assert(basic_selector<surface_keep_octants_t, geospatial_ram_index_t, chord_metric_t>);

/** @brief Hamming distance over eight-byte tags - the smallest metric the concept admits. */
struct header_only_metric_t {
    std::array<char, 8> query_ {};
    std::array<u64_t, 64> differences_ {};
    std::size_t staged_ = 0;

    void reset(std::array<char, 8> const& query) noexcept { query_ = query, staged_ = 0; }
    basic_metric_staging_t push(std::array<char, 8> const& value) noexcept {
        u64_t difference = 0;
        for (std::size_t position = 0; position != query_.size(); ++position)
            difference += query_[position] != value[position];
        differences_[staged_++] = difference;
        return staged_ != differences_.size() ? basic_metric_staging_t::staged_k : basic_metric_staging_t::refused_k;
    }
    void drain(std::span<u64_t> distances) noexcept {
        for (std::size_t lane = 0; lane != distances.size(); ++lane) distances[lane] = differences_[lane];
        staged_ = 0;
    }
};

/** @brief The same metric with the transparency tag - a compile-only concept witness. */
struct header_only_transparent_metric_t : header_only_metric_t {
    using is_transparent = void;
};

/** @brief Identity-bytes codec for the fixed-width tags. */
struct header_only_codec_t {
    std::size_t storage_requirements(std::array<char, 8> const&) const noexcept { return 8; }
    status_t save(std::array<char, 8> const& value, bytes_span_t output) const noexcept {
        if (output.size() < value.size()) return status_t::overflow_k;
        for (std::size_t position = 0; position != value.size(); ++position)
            output[position] = static_cast<std::byte>(value[position]);
        return status_t::success_k;
    }
    status_t load(bytes_view_t input, std::array<char, 8>& value) const noexcept {
        if (input.size() != value.size()) return status_t::malformed_k;
        for (std::size_t position = 0; position != value.size(); ++position)
            value[position] = static_cast<char>(input[position]);
        return status_t::success_k;
    }
};

using narrow_index_t = basic_ram_index<std::array<char, 8>, u64_t, u32_t, u64_t>;
using wide_index_t = basic_ram_index<std::array<char, 8>, uuid_t, u64_t, u64_t>;

static_assert(basic_value<std::array<char, 8>>);
static_assert(basic_metric<header_only_metric_t, std::array<char, 8>, u64_t>);
static_assert(!transparent_metric<header_only_metric_t>);
static_assert(transparent_metric<header_only_transparent_metric_t>);
static_assert(basic_query<std::array<char, 8>, header_only_metric_t, std::array<char, 8>>);
static_assert(basic_value_codec<header_only_codec_t, std::array<char, 8>>);
static_assert(basic_selector<basic_keep_closest_t, narrow_index_t, header_only_metric_t>);
static_assert(basic_selector<basic_keep_diverse_t, wide_index_t, header_only_metric_t>);
static_assert(basic_expansion_round_t::converged_k != basic_expansion_round_t::advanced_k);
static_assert(std::is_same_v<narrow_index_t::columns_t, basic_columns<u64_t, u32_t, byte_t, u64_t>>);
static_assert(std::is_move_constructible_v<narrow_index_t::save_plan_t> &&
              !std::is_copy_constructible_v<narrow_index_t::save_plan_t> &&
              !std::is_copy_assignable_v<narrow_index_t::save_plan_t> &&
              !std::is_move_assignable_v<narrow_index_t::save_plan_t>);
static_assert(std::is_move_constructible_v<narrow_index_t::load_plan_t> &&
              !std::is_copy_constructible_v<narrow_index_t::load_plan_t> &&
              !std::is_copy_assignable_v<narrow_index_t::load_plan_t> &&
              !std::is_move_assignable_v<narrow_index_t::load_plan_t>);
static_assert(
    std::is_same_v<wide_index_t::columns_view_t, basic_columns<uuid_t const, u64_t const, byte_t const, u64_t const>>);

/** @brief The tags the index probes store: distinct leading letters, shared tails. */
static std::array<char, 8> make_tag(std::size_t ordinal) noexcept {
    return {static_cast<char>('a' + ordinal % 26), static_cast<char>('a' + ordinal / 26), 0, 0, 0, 0, 0, 0};
}

int main() {
    // The type vocabulary: statuses, expected & its borrowing face, hashes & mixers.
    if (std::string_view {status_to_string(status_t::success_k)}.empty()) return 1;
    expected<u32_t> const answered {42, status_t::success_k};
    if (!answered || answered.value != 42) return 1;
    u64_t seven = 7;
    expected<u64_t const&> const borrowed {&seven, status_t::success_k};
    if (!borrowed || borrowed.value() != 7) return 1;
    uuid_t const uuid {};
    if (std::hash<uuid_t> {}(uuid) != std::hash<uuid_t> {}(uuid)) return 1;
    if (mix_bits(1) == mix_bits(2)) return 1;
    if (multiply_high(u64_t {1} << 32, u64_t {1} << 32) != 1) return 1;
    if (dtype_named("u64") != dtype_t::u64_k) return 1;
    if (std::string_view {dtype_name(dtype_t::uuid_k)} != "uuid") return 1;

    // The engine & distribution split, at both widths.
    splitmix64_t random_generator {7};
    if (uniform_below<u32_t> {10}(random_generator) >= 10) return 1;
    if (uniform_below<u64_t> {10}(random_generator) >= 10) return 1;
    if (ceil2(3) != 4) return 1;

    // Sequential containers: the growing buffer's whole accessor family.
    growing_buffer<u32_t> sequence {global_allocator};
    if (failed(sequence.reserve(4))) return 1;
    sequence.push_back_reserved(3), sequence.push_back_reserved(1), sequence.push_back_reserved(2);
    if (sequence.front() != 3 || sequence.back() != 2) return 1;
    if (sequence.pop_back() != 2 || sequence.size() != 2) return 1;
    if (failed(sequence.push_back(9)) || sequence.back() != 9) return 1;

    std::array<u32_t, 4> quad {4, 1, 3, 2};
    network_sort(std::span<u32_t, 4> {quad});
    if (quad[0] != 1 || quad[3] != 4) return 1;

    max_heap<dense_scored_node_t, dense_further_scored_node_t> frontier {global_allocator};
    if (failed(frontier.reserve(64))) return 1;
    if (failed(frontier.insert({dense_node_id_t {1}, dense_distance_t {0.5f}}))) return 1;
    if (failed(frontier.insert({dense_node_id_t {2}, dense_distance_t {0.25f}}))) return 1;
    if (frontier.top().node_id != 2) return 1; // The furthest-first comparator surfaces the nearest.
    frontier.pop();

    max_heap_dary<8, dense_scored_node_t, dense_further_scored_node_t> wide_frontier {global_allocator};
    if (failed(wide_frontier.reserve(64))) return 1;
    if (failed(wide_frontier.insert({dense_node_id_t {3}, dense_distance_t {0.75f}}))) return 1;
    if (wide_frontier.top().node_id != 3) return 1;

    sorted_buffer<dense_scored_node_t, dense_closer_scored_node_t> results {global_allocator};
    if (failed(results.reserve(4))) return 1;
    results.insert({dense_node_id_t {3}, dense_distance_t {0.75f}}, 4);
    if (results.top().node_id != 3) return 1;

    // Associative containers: the visited set & the concurrent multimap.
    growing_hash_set<dense_node_id_t> visited {global_allocator};
    if (failed(visited.reserve(64))) return 1;
    if (visited.insert(42) != membership_t::absent_k) return 1;
    if (!visited.contains(42) || visited.contains(43)) return 1;
    if (visited.insert(std::array<dense_node_id_t, 4> {42, 43, 44, 45}) != 0b1110u) return 1;

    growing_multimap<u64_t, u32_t> multimap {global_allocator};
    if (failed(multimap.reserve(8))) return 1;
    if (failed(multimap.insert(7, 42)) || failed(multimap.insert(7, 43))) return 1;
    if (multimap.count(7) != 2 || !multimap.contains(7)) return 1;
    if (failed(multimap.reserve(64)) || multimap.count(7) != 2) return 1; // Rehash keeps entries.
    multimap.clear();
    if (multimap.size() != 0 || multimap.contains(7)) return 1;

    prefetch_query_sse_t const prefetch_query;
    prefetch_query(&random_generator);
    prefetch_candidate_sse_t const prefetch_candidate;
    prefetch_candidate(&random_generator);

    // The basic index, narrow instantiation: the whole public surface, existence-level.
    narrow_index_t index;
    if (failed(index.reserve(16).status) || index.capacity() != 16) return 1;
    std::array<std::array<char, 8>, 8> tags {};
    std::array<u64_t, 8> keys {};
    for (std::size_t tag = 0; tag != tags.size(); ++tag) tags[tag] = make_tag(tag), keys[tag] = 100 + tag;
    header_only_metric_t metric;
    header_only_codec_t const codec;
    narrow_index_t::traversal_context_t context {7};
    std::array<u32_t, 8> assigned_ids {};
    if (failed(index.insert({tags}, {keys}, metric, {assigned_ids}, context).status)) return 1;
    std::array<std::array<char, 8>, 1> one_more {{make_tag(60)}};
    std::array<u64_t, 1> const one_more_key {200};
    if (failed(index
                   .template insert<synchronization_t::exclusive_k>({one_more}, {one_more_key}, metric, {}, context,
                                                                    basic_keep_closest_t {2})
                   .status))
        return 1;
    if (index.size() != 9) return 1;

    // All three search arms answer, ranked & settled-equal.
    auto const query = make_tag(0);
    auto const concurrent_found = index.search(query, metric, context);
    auto const immutable_found = index.template search<synchronization_t::immutable_k>(query, metric, context);
    auto const exclusive_found = index.template search<synchronization_t::exclusive_k>(query, metric, context);
    if (failed(concurrent_found.status) || !concurrent_found.count) return 1;
    if (immutable_found.count != concurrent_found.count || exclusive_found.count != concurrent_found.count) return 1;

    // Manual wiring, the checked readers, and the keyed lookups.
    auto const linked = index.link(0, 1);
    if (failed(linked.status)) return 1;
    if (index.unlink(0, 1) != status_t::success_k) return 1;
    if (index.unlink(0, 1) != status_t::malformed_k) return 1;
    if (!index.key_of(0) || index.key_of(0).value != 100) return 1;
    if (!index.value_of(0) || index.value_of(0).value() != index.published_value_of(0)) return 1;
    if (!index.degree_of(0)) return 1;
    if (index.key_of(static_cast<u32_t>(index.size())).status != status_t::malformed_k) return 1;
    if (!index.find(101) || index.find(101).value != 1) return 1;
    if (!index.contains(200) || index.count(999) != 0) return 1;
    std::size_t found_holders = 0;
    index.find_each(100, [&](u32_t) noexcept { ++found_holders; });
    if (found_holders != 1) return 1;

    auto const health = index.health(context);
    if (failed(health.status) || !health.connected_components) return 1;

    // The snapshot round-trip needs nothing outside the tier either: five columns in vectors.
    auto const extents = index.columns_extents(codec);
    if (failed(extents.status)) return 1;
    std::vector<u64_t> keys_column(extents.value.nodes);
    std::vector<u64_t> neighbor_offsets(extents.value.nodes + 1);
    std::vector<u32_t> neighbor_ids(extents.value.neighbors);
    std::vector<u64_t> value_offsets(extents.value.nodes + 1);
    std::vector<std::byte> values(extents.value.value_bytes);
    narrow_index_t::columns_t const columns {keys_column, neighbor_offsets, neighbor_ids, value_offsets, values};
    if (index.save(columns, codec) != status_t::success_k) return 1;
    narrow_index_t restored;
    if (failed(restored.reserve(16).status)) return 1;
    if (restored.load(columns.view(), codec) != status_t::success_k) return 1;
    if (restored.size() != index.size()) return 1;
    if (restored.key_of(0).value != index.key_of(0).value) return 1;
    if (restored.degree_of(3).value != index.degree_of(3).value) return 1;

    // The wide instantiation: uuid keys through `std::hash<uuid_t>`, 64-bit ids.
    wide_index_t wide_index;
    if (failed(wide_index.reserve(4).status)) return 1;
    std::array<uuid_t, 2> const wide_keys {std::bit_cast<uuid_t>(std::array<u64_t, 2> {1, 2}),
                                           std::bit_cast<uuid_t>(std::array<u64_t, 2> {3, 4})};
    std::array<std::array<char, 8>, 2> wide_tags {{make_tag(1), make_tag(2)}};
    wide_index_t::traversal_context_t wide_context {9};
    if (failed(wide_index.insert({wide_tags}, {wide_keys}, metric, {}, wide_context, basic_keep_diverse_t {2}).status))
        return 1;
    if (!wide_index.find(wide_keys[1]) || wide_index.find(wide_keys[1]).value != 1) return 1;
    if (failed(wide_index.search(make_tag(1), metric, wide_context).status)) return 1;
    return 0;
}
