/**
 *  @file include/usearch/sequence/vram_apple_index.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Similarity Search over byte sequences on Apple GPUs: the basic Apple graph under
 *      one sequence kit of @c vram_apple_metric.metal, Levenshtein over bytes or UTF-8 runes,
 *      window overlap or Needleman-Wunsch over packed codes.
 *
 *  Plain C++ for a unit building with Metal, as every Apple face: the kits are MSL the graph
 *  compiles at its first @c reserve, from StringZilla's Metal sources and the pieces each kit
 *  names here, so the host members reading a kit, which pick a launch's rung and size its spans,
 *  live beside its description. Memory is unified, so the kit parameters are one shared buffer the
 *  kernels read in place, and every load stores its sequences as saved and counts their prepared
 *  spans on the host.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - the folded costs into the parameters

#include <bit>         // `std::popcount` - a byte query's distinct bytes
#include <limits>      // `std::numeric_limits` - a length no `u32_t` states
#include <span>        // `std::span` - the overlap kind's window widths
#include <type_traits> // `std::is_empty_v` - a kit taking no parameters
#include <variant>     // `std::monostate` - Levenshtein takes no parameters

#include <stringzilla/overlap.h>           // `sz_overlap_query_span_bytes` - an overlap query's span
#include <stringzilla/levenshtein/metal.h> // `sz_metal_prelude_source_`, `sz_levenshtein_source_metal_`
#include <stringzilla/overlap/metal.h>     // `sz_overlap_source_metal_` - the kernels the kits call

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the widths the host members read
#include "usearch/sequence/ram_metric.hpp"   // `needleman_wunsch_costs_t`, `packed_codes_count`
#include "usearch/utilities/vram_apple_shared.hpp"
#include "usearch/sequence/vram_apple_metric.metal" // the kits' rungs and Needleman-Wunsch stripes alone, on a host
#include "usearch/basic/vram_apple_index.hpp"

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif

/** The sequence kits, spliced last of a face's pieces. */
inline constexpr char metal_sequence_metric_source_k[] = {
#embed "vram_apple_metric.metal" suffix(, 0)
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#pragma region Kits

/** Needleman-Wunsch over packed codes: a simdgroup to a sequence, an anti-diagonal wavefront down
 *  stripes of the query, the folded costs uploaded as the parameters. */
struct apple_needleman_wunsch_kit_t {

    /** The folded costs a face scores with, set once before the first insert. */
    using parameters_t = needleman_wunsch_costs_t;

    static_assert(sizeof(needleman_wunsch_costs_t) == metal_needleman_wunsch_costs_bytes_k &&
                      offsetof(needleman_wunsch_costs_t, substitutions) == 3,
                  "the MSL kit reads the costs as they are laid out");

    inline static constexpr std::size_t rungs_k = metal_needleman_wunsch_rungs_k;
    inline static constexpr char metal_kit_name_k[] = "sequence::metal_needleman_wunsch_kit_t";
    inline static constexpr char const* metal_sources_k[] = {sz_metal_prelude_source_, sz_levenshtein_source_metal_,
                                                             sz_overlap_source_metal_, metal_sequence_metric_source_k};

    /** Every rung scores a sequence with the whole simdgroup. */
    static constexpr u32_t lanes_per_row(std::size_t) noexcept { return metal_simdgroup_width_k; }

    /** Codes a pack of @p bytes holds at most past its one-byte header. */
    static constexpr std::size_t codes_bound(std::size_t bytes, parameters_t const& costs) noexcept {
        return bytes ? (bytes - 1) * 8 / costs.code_bits : 0;
    }

    /** The first rung holding whole the stripe of the most codes a @p longest -byte pack holds, the
     *  widest one any number of its stripes, or past every rung before any costs. */
    static std::size_t rung_serving(std::size_t longest, parameters_t const& costs) noexcept {
        if (!costs.code_bits) return rungs_k;
        std::size_t const codes = codes_bound(longest, costs), widest = rungs_k - 1;
        std::size_t rung = 0;
        while (rung != rungs_k &&
               codes > (rung == widest ? std::size_t {metal_needleman_wunsch_widest_codes_k}
                                       : std::size_t {metal_needleman_wunsch_rows(static_cast<u32_t>(rung))} *
                                             metal_simdgroup_width_k))
            ++rung;
        return rung;
    }

    /** Bytes the packed @p bytes long sequence at @p data takes once prepared, and the bound for
     *  any pack of up to @p longest bytes. */
    static std::size_t span_bytes(byte_t const* data, std::size_t bytes, parameters_t const& costs) noexcept {
        return metal_needleman_wunsch_span_bytes(static_cast<u32_t>(packed_codes_count(data, bytes, costs.code_bits)));
    }
    static std::size_t span_bytes(std::size_t longest, parameters_t const& costs) noexcept {
        std::size_t const codes = min_of<std::size_t>(codes_bound(longest, costs),
                                                      metal_needleman_wunsch_widest_codes_k);
        return metal_needleman_wunsch_span_bytes(static_cast<u32_t>(codes));
    }

    /** Lays the costs out as the kit reads them. */
    static std::size_t parameters_bytes(parameters_t const&) noexcept { return sizeof(parameters_t); }
    static void lay_out(parameters_t const& costs, byte_t* laid) noexcept { std::memcpy(laid, &costs, sizeof(costs)); }
};

/** Window overlap: a simdgroup lays the query's window hashes at every width out as a tree in its
 *  span, and a group of a rung's lanes counts each candidate's windows the tree holds. The widths
 *  are the parameters, their count first. */
struct apple_overlap_kit_t {

    /** The window widths a face hashes with, set once before the first insert. */
    using parameters_t = std::span<u32_t const>;

    inline static constexpr std::size_t rungs_k = metal_overlap_rungs_k;
    inline static constexpr char metal_kit_name_k[] = "sequence::metal_overlap_kit_t";
    inline static constexpr char const* metal_sources_k[] = {sz_metal_prelude_source_, sz_levenshtein_source_metal_,
                                                             sz_overlap_source_metal_, metal_sequence_metric_source_k};

    /** A rung is its lanes per sequence. */
    static constexpr u32_t lanes_per_row(std::size_t rung) noexcept {
        return metal_overlap_lanes(static_cast<u32_t>(rung));
    }

    /** The widest rung whose lanes still roll a widest window of a @p longest -byte sequence, or
     *  past every rung for a length no byte count states. */
    static std::size_t rung_serving(std::size_t longest, parameters_t widths) noexcept {
        if (longest > (std::numeric_limits<u32_t>::max)()) return rungs_k;
        u32_t widest = 0;
        for (u32_t const width : widths) widest = max_of(widest, width);
        std::size_t rung = 0;
        while (rung + 1 < rungs_k && std::size_t {lanes_per_row(rung + 1)} * widest <= longest) ++rung;
        return rung;
    }

    /** Bytes a sequence of @p bytes takes once prepared, from its length alone. */
    static std::size_t span_bytes(byte_t const*, std::size_t bytes, parameters_t widths) noexcept {
        return span_bytes(bytes, widths);
    }
    static std::size_t span_bytes(std::size_t bytes, parameters_t widths) noexcept {
        return round_up_to_multiple<std::size_t>(sz_overlap_query_span_bytes(bytes, widths.data(), widths.size()), 16);
    }

    /** Lays the widths out after their count, as the kit reads them. */
    static std::size_t parameters_bytes(parameters_t widths) noexcept { return (widths.size() + 1) * sizeof(u32_t); }
    static void lay_out(parameters_t widths, byte_t* laid) noexcept {
        u32_t const count = static_cast<u32_t>(widths.size());
        std::memcpy(laid, &count, sizeof(count));
        if (count) std::memcpy(laid + sizeof(count), widths.data(), widths.size_bytes());
    }
};

/**
 *  @brief Levenshtein over bytes or, under @c sz_levenshtein_runes_k, over UTF-8 runes: a lane or,
 *      past StringZilla Metal's threaded ceiling, a simdgroup to a sequence, the query's classes
 *      and masks prepared in its span by @c metal_levenshtein_kit.
 *
 *  A rung is the Myers words of the longest query its kernels score, so a sequence of @c b bytes
 *  takes the first rung of at least @c b bits, as on NVIDIA. The host counts each sequence's span
 *  itself, as its prepare lays it out.
 */
template <sz_levenshtein_symbol_t symbol_>
struct apple_levenshtein_kit {

    /** A Levenshtein distance takes no parameters. */
    using parameters_t = std::monostate;

    inline static constexpr std::size_t rungs_k = metal_levenshtein_rungs_k;
    inline static constexpr char const* metal_kit_name_k = symbol_ == sz_levenshtein_runes_k
                                                               ? "sequence::metal_levenshtein_utf8_kit_t"
                                                               : "sequence::metal_levenshtein_bytes_kit_t";
    inline static constexpr char const* metal_sources_k[] = {sz_metal_prelude_source_, sz_levenshtein_source_metal_,
                                                             sz_overlap_source_metal_, metal_sequence_metric_source_k};

    /** A lane scores a sequence alone up to the threaded ceiling, a simdgroup past it. */
    static constexpr u32_t lanes_per_row(std::size_t rung) noexcept {
        return metal_levenshtein_words(static_cast<u32_t>(rung)) > metal_levenshtein_thread_words_max_k
                   ? metal_simdgroup_width_k
                   : 1;
    }

    /** The first rung whose words cover a @p longest -byte sequence, a bit of a 64-bit word per
     *  byte, or past the widest; a sequence of @c b bytes holds at most @c b runes, so its bytes
     *  bound rune sequences too. */
    static std::size_t rung_serving(std::size_t longest, parameters_t) noexcept {
        std::size_t rung = 0;
        while (rung != rungs_k && longest > std::size_t {metal_levenshtein_words(static_cast<u32_t>(rung))} * 64)
            ++rung;
        return rung;
    }

    /** Bytes the @p bytes long sequence at @p data takes once prepared, as the Metal prepare lays
     *  it out: a class map over every byte value, or a page table with a class row per page its
     *  runes touch, then a mask row per class at the walk stride of the sequence's own words, one
     *  class per distinct symbol, one more, and a row past them a two-word lane may read. None
     *  from a length alone, as a load counts its sequences' spans. */
    static std::size_t span_bytes(byte_t const* data, std::size_t bytes, parameters_t) noexcept {
        if constexpr (symbol_ == sz_levenshtein_runes_k) {
            // Distinct pages in a bitset over every page, distinct runes in an open table twice the
            // widest rung's runes wide, cleared only as wide as this sequence needs.
            constexpr std::size_t table_slots_k = 2 * 64 * metal_levenshtein_words(metal_levenshtein_rungs_k - 1);
            u64_t pages[sz_levenshtein_utf8_pages_k / 64] = {};
            u32_t table[table_slots_k];
            std::size_t slots = 1;
            while (slots < 2 * bytes + 2 && slots < table_slots_k) slots *= 2;
            for (std::size_t slot = 0; slot != slots; ++slot) table[slot] = ~0u;
            sz_size_t runes = 0, rows = 1, classes = 1;
            for (sz_size_t position = 0; position < bytes; ++runes) {
                sz_rune_t const rune = sz_utf8_next_rune_(reinterpret_cast<sz_cptr_t>(data), bytes, &position);
                // A sequence past the widest rung is refused before it prepares, so needs no count.
                if (bytes * 2 > table_slots_k) {
                    ++classes;
                    continue;
                }
                rows += !(pages[rune >> 14] >> ((rune >> 8) & 63u) & 1u);
                pages[rune >> 14] |= u64_t {1} << ((rune >> 8) & 63u);
                std::size_t slot = (rune * 2654435761u) & (slots - 1);
                while (table[slot] != ~0u && table[slot] != rune) slot = (slot + 1) & (slots - 1);
                classes += table[slot] == ~0u;
                table[slot] = rune;
            }
            return round_up_to_multiple<std::size_t>(
                sz_levenshtein_utf8_span_header_k + sz_levenshtein_utf8_pages_k * sizeof(u16_t) +
                    rows * 256 * sizeof(u32_t) +
                    (classes + 1) * sz_levenshtein_walk_stride(sz_levenshtein_query_words(runes)) * sizeof(u64_t),
                16);
        }
        else {
            u64_t present[4] = {};
            for (std::size_t position = 0; position != bytes; ++position) {
                u8_t const value = static_cast<u8_t>(data[position]);
                present[value >> 6] |= u64_t {1} << (value & 63u);
            }
            std::size_t distinct = 0;
            for (u64_t const word : present) distinct += static_cast<std::size_t>(std::popcount(word));
            std::size_t const classes = min_of<std::size_t>(distinct + 1, sz_levenshtein_byte_classes_k);
            return round_up_to_multiple<std::size_t>(
                sz_levenshtein_byte_classes_k +
                    (classes + 1) * sz_levenshtein_walk_stride(sz_levenshtein_query_words(bytes)) * sizeof(u64_t),
                16);
        }
    }
    static std::size_t span_bytes(std::size_t, parameters_t) noexcept { return 0; }

    /** Nothing to lay out. */
    static std::size_t parameters_bytes(parameters_t) noexcept { return 0; }
    static void lay_out(parameters_t, byte_t*) noexcept {}
};

using apple_levenshtein_bytes_kit_t = apple_levenshtein_kit<sz_levenshtein_bytes_k>;
using apple_levenshtein_utf8_kit_t = apple_levenshtein_kit<sz_levenshtein_runes_k>;

/** The Apple kits on the ladders the faces and tests pick a kind's kit from, at Metal's
 *  capability. */
template <sz_capability_t caps_>
    requires((caps_ & sz_cap_metal_k) != 0)
struct levenshtein_ladder<sz_levenshtein_bytes_k, caps_> {
    using kit_t = apple_levenshtein_bytes_kit_t;
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_metal_k) != 0)
struct levenshtein_ladder<sz_levenshtein_runes_k, caps_> {
    using kit_t = apple_levenshtein_utf8_kit_t;
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_metal_k) != 0)
struct overlap_ladder<caps_> {
    using kit_t = apple_overlap_kit_t;
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_metal_k) != 0)
struct needleman_wunsch_ladder<caps_> {
    using kit_t = apple_needleman_wunsch_kit_t;
};

#pragma endregion Kits

#pragma region Face

/** The kit hooks a sequence graph on Apple GPUs reads: the kit @p kit_type_ describes, on its
 *  rungs, over node IDs of @p node_id_type_, with the parameters it scores with. */
template <typename kit_type_, typename node_id_type_>
struct vram_apple_kits {
    using parameters_t = typename kit_type_::parameters_t;

    inline static constexpr std::size_t rungs_k = kit_type_::rungs_k;
    inline static constexpr auto const& metal_sources_k = kit_type_::metal_sources_k;
    inline static constexpr auto const& metal_kit_name_k = kit_type_::metal_kit_name_k;

    /**
     *  @brief Keeps the parameters the kit scores with, uploaded at the first @c reserve.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The host copy found no memory.
     */
    status_t parameters(parameters_t parameters) noexcept {
        std::size_t const bytes = kit_type_::parameters_bytes(parameters);
        if (status_t const sized = host_parameters_.resize(divide_round_up(bytes, sizeof(u32_t))); failed(sized))
            return sized;
        kit_type_::lay_out(parameters, bytes_of(host_parameters_.data(), host_parameters_.size()).data());
        if constexpr (std::is_same_v<parameters_t, std::span<u32_t const>>)
            kept_parameters_ = to_span(host_parameters_).subspan(1);
        else kept_parameters_ = parameters;
        return status_code_t::success_k;
    }

  private:
    template <typename, typename, typename, typename>
    friend struct basic::vram_apple_graph;

    /** The kit's rung serving sequences of @p longest bytes, or @c rungs_k past every one. */
    std::size_t rung_serving_(std::size_t longest) const noexcept {
        return kit_type_::rung_serving(longest, kept_parameters_);
    }

    /** Lanes the kit scores a sequence with on @p rung. */
    static u32_t lanes_per_entry_(std::size_t rung) noexcept { return kit_type_::lanes_per_row(rung); }

    /** Bytes the @p bytes long sequence at @p data takes once prepared, and the bound any loaded
     *  sequence of up to @p longest bytes takes. */
    std::size_t span_bytes_(byte_t const* data, std::size_t bytes) const noexcept {
        return round_up_to_multiple<std::size_t>(kit_type_::span_bytes(data, bytes, kept_parameters_), 16);
    }
    std::size_t span_bytes_(std::size_t longest) const noexcept {
        return round_up_to_multiple<std::size_t>(kit_type_::span_bytes(longest, kept_parameters_), 16);
    }

    /** A sequence saves as its bytes, as a RAM string. */
    static expected<std::size_t> saved_bytes_(bytes_view_t stored) noexcept {
        return {stored.size(), status_code_t::success_k};
    }

    /**
     *  @brief Saves the @p stored sequence into @p saved as it is.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The two differ in size.
     */
    static status_t save_value_(bytes_view_t stored, bytes_span_t saved) noexcept { return copy_bytes(saved, stored); }

    /** Bytes a sequence of @p saved bytes stores into: its own. */
    static expected<std::size_t> stored_bytes_(std::size_t saved) noexcept { return {saved, status_code_t::success_k}; }

    /** A load stores sequences as saved, on the host over unified memory, then raises the widest
     *  span to the widest its sequences prepare into. */
    status_t queue_load_values_(basic::metal_load_step<node_id_type_> const& step) const noexcept {
        if (status_t const copied = step.copy_values(); failed(copied)) return copied;
        std::size_t widest = 0;
        for (std::size_t node = 0; node != step.arguments.count; ++node) {
            std::size_t const first = step.stored_offsets[node], bytes = step.stored_offsets[node + 1] - first;
            widest = max_of(widest, span_bytes_(step.stored + (first - step.stored_offsets[0]), bytes));
        }
        *step.span_bytes = max_of(*step.span_bytes, widest);
        return status_code_t::success_k;
    }

    /**
     *  @brief Copies the laid-out parameters into a shared buffer on @p context 's device.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The kit takes parameters, and none were set.
     *  - @c out_of_memory_k The device found no memory.
     */
    status_t upload_parameters_(metal_context_t& context) noexcept {
        if (host_parameters_.empty())
            return std::is_empty_v<parameters_t> ? status_code_t::success_k : status_code_t::uninitialized_k;
        std::size_t const bytes = host_parameters_.size() * sizeof(u32_t);
        device_parameters_ = metal_allocate(context, bytes);
        if (!device_parameters_) return status_code_t::out_of_memory_k;
        std::memcpy(device_parameters_.host, host_parameters_.data(), bytes);
        return status_code_t::success_k;
    }

    /** The kit's parameters on the device, empty for a kit taking none. */
    metal_buffer_t const& parameters_() const noexcept { return device_parameters_; }

    /** The kit's parameters as its host members read them, a span into @c host_parameters_ where
     *  they are one. */
    parameters_t kept_parameters_ {};

    /** The kit's parameters laid out as its kernels read them, on the host and, past the first
     *  @c reserve, in a shared buffer. */
    growing_buffer<u32_t, std::allocator<byte_t>> host_parameters_ {std::allocator<byte_t> {}};
    metal_buffer_t device_parameters_;
};

/** Similarity Search over byte sequences on Apple GPUs scored by the sequence kit @p kit_type_
 *  describes, on its rungs. Node IDs are @p node_id_type_. It owns a @c basic::vram_apple_graph
 *  over its kits and forwards its verbs, types and constants, which that graph documents. */
template <typename kit_type_, typename node_id_type_>
struct vram_apple_index {
    /* Components */
    using kits_t = vram_apple_kits<kit_type_, node_id_type_>;
    using parameters_t = typename kits_t::parameters_t;
    using graph_t = basic::vram_apple_graph<node_id_type_, kits_t>;
    using traversal_context_t = typename graph_t::traversal_context_t;

    /* IDs and Keys */
    using node_id_t = typename graph_t::node_id_t;
    using key_t = typename graph_t::key_t;
    using keyed_t = typename graph_t::keyed_t;
    using distance_t = typename graph_t::distance_t;
    using candidates_count_t = typename graph_t::candidates_count_t;

    /* Configs and Results */
    using reserve_capacity_t = typename graph_t::reserve_capacity_t;
    using insert_config_t = typename graph_t::insert_config_t;
    using insert_result_t = typename graph_t::insert_result_t;
    using insert_plan_t = typename graph_t::insert_plan_t;
    using search_config_t = typename graph_t::search_config_t;
    using search_result_t = typename graph_t::search_result_t;
    using refine_config_t = typename graph_t::refine_config_t;
    using reshard_config_t = typename graph_t::reshard_config_t;
    using refine_report_t = typename graph_t::refine_report_t;
    using reachability_t = typename graph_t::reachability_t;

    /* Plans */
    using refine_plan_t = typename graph_t::refine_plan_t;
    using reshard_plan_t = typename graph_t::reshard_plan_t;
    using save_plan_t = typename graph_t::save_plan_t;
    using load_plan_t = typename graph_t::load_plan_t;

    inline static constexpr std::size_t max_node_count_k = graph_t::max_node_count_k;
    inline static constexpr synchronization_t synchronization_ceiling_k = graph_t::synchronization_ceiling_k;

    /** Pins the index to the allocator's device; no sequence kind has a NumKong kind. */
    explicit vram_apple_index(metal_shared_allocator_t const& allocator = {}) noexcept
        : graph_(nk_kernel_unknown_k, allocator) {}

    /**
     *  @brief Keeps the parameters the kit scores with, frozen once the first @c reserve
     *      has uploaded them.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index is reserved already.
     *  - @c out_of_memory_k The host copy found no memory.
     */
    status_t parameters(parameters_t parameters) noexcept {
        if (graph_.connectivity()) return status_code_t::config_mismatch_k;
        return graph_.kits().parameters(parameters);
    }

    std::size_t size() const noexcept { return graph_.size(); }
    status_t status() const noexcept { return graph_.status(); }
    std::size_t connectivity() const noexcept { return graph_.connectivity(); }
    std::size_t capacity() const noexcept { return graph_.capacity(); }
    std::size_t bytes_capacity() const noexcept { return graph_.bytes_capacity(); }
    std::size_t memory_usage() const noexcept { return graph_.memory_usage(); }
    metal_device_specs_t const& device_specs() const noexcept { return graph_.device_specs(); }
    metal_context_t* context() const noexcept { return graph_.context(); }
    nk_kernel_kind_t kernel() const noexcept { return graph_.kernel(); }
    basic::metal_ragged_storage<node_id_t> storage() const noexcept { return graph_.storage(); }
    expected<std::size_t> reserve(reserve_capacity_t config) noexcept { return graph_.reserve(config); }

    /** The scratch a call of these arguments needs ahead of time: none, as a Metal context grows
     *  its own buffers; spelled for every verb the NVIDIA faces size. */
    template <typename... usage_types_>
    std::size_t traversal_memory_usage(usage_types_ const&... usage) const noexcept {
        return graph_.traversal_memory_usage(usage...);
    }

    expected<insert_plan_t> plan_insert(std::size_t entries) noexcept { return graph_.plan_insert(entries); }
    expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        return graph_.insert(keys, tape, context, config);
    }
    expected<search_result_t> search(tape_view<byte_t const> queries, nk::matrix_span<node_id_t> result_ids,
                                     nk::matrix_span<distance_t> result_distances, search_config_t config,
                                     traversal_context_t& context) noexcept {
        return graph_.search(queries, result_ids, result_distances, config, context);
    }

    expected<refine_plan_t> plan_refine(refine_config_t config, traversal_context_t& context) noexcept {
        return graph_.plan_refine(config, context);
    }
    expected<refine_report_t> refine(refine_config_t config, traversal_context_t& context) noexcept {
        return graph_.refine(config, context);
    }
    expected<reshard_plan_t> plan_reshard(std::span<vram_apple_index* const> sources,
                                          std::span<vram_apple_index* const> targets, reshard_config_t config,
                                          traversal_context_t& context) noexcept {
        return graph_.plan_reshard(sources, targets, &vram_apple_index::graph_, config, context);
    }
    expected<refine_report_t> reshard(std::span<vram_apple_index* const> sources,
                                      std::span<vram_apple_index* const> targets, reshard_config_t config,
                                      traversal_context_t& context) noexcept {
        return graph_.reshard(sources, targets, &vram_apple_index::graph_, config, context);
    }
    expected<reachability_t> health(traversal_context_t& context) noexcept { return graph_.health(context); }

    expected<basic::columns_extents_t> columns_extents(traversal_context_t& context) noexcept {
        return graph_.columns_extents(context);
    }
    expected<save_plan_t> plan_save(basic::columns<key_t, node_id_t, byte_t, u64_t> columns,
                                    traversal_context_t& context) noexcept {
        return graph_.plan_save(columns, context);
    }
    status_t save(basic::columns<key_t, node_id_t, byte_t, u64_t> columns, traversal_context_t& context) noexcept {
        return graph_.save(columns, context);
    }
    expected<load_plan_t> plan_load(safegraphs_saved_columns<node_id_t> const& saved,
                                    traversal_context_t& context) noexcept {
        return graph_.plan_load(saved, context);
    }
    status_t load(safegraphs_saved_columns<node_id_t> const& saved, traversal_context_t& context) noexcept {
        return graph_.load(saved, context);
    }

    keyed_t const& keys() const noexcept { return graph_.keys(); }
    expected<key_t> key_of(node_id_t node) const noexcept { return graph_.key_of(node); }
    template <typename callback_type_>
    void find_each(key_t const& key, callback_type_&& callback) const noexcept {
        graph_.find_each(key, callback);
    }
    status_t neighbors(node_id_t first, std::span<node_id_t> rows) const noexcept {
        return graph_.neighbors(first, rows);
    }
    std::size_t count(key_t const& key) const noexcept { return graph_.count(key); }
    bool contains(key_t const& key) const noexcept { return graph_.contains(key); }

    /** Copies the values of the @p listed nodes back to back into @p outputs, their offsets into
     *  @p output_offsets from the base its first holds, on the host over unified memory. */
    status_t gather_tape_(std::span<node_id_t const> listed, byte_t* outputs,
                          std::span<u64_t> output_offsets) const noexcept {
        return graph_.gather_tape_(listed, outputs, output_offsets);
    }

  private:

    /** A sharded index's shard table measures, closes and reopens the graph's claims. */
    template <typename, template <typename> class atomic_ref_template_, typename, typename>
    friend struct utilities::shard_set;

    /** A sharded index walks the graph below its snapshot and publishes its batches inside one
     *  commit. */
    template <typename index_type_, typename, typename, typename, typename>
        requires std::same_as<typename index_type_::node_id_t, u32_t>
    friend struct basic::vram_apple_indexes;

    std::size_t claimed_() const noexcept { return graph_.claimed_(); }
    status_t close_() noexcept { return graph_.close_(); }
    void reopen_() noexcept { graph_.reopen_(); }
    void retire_() noexcept { graph_.retire_(); }

    graph_t graph_;
};

#pragma endregion Face

} // namespace unum::usearch::sequence
