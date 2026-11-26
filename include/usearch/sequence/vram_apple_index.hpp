/**
 *  @file include/usearch/sequence/vram_apple_index.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Similarity Search over byte sequences on Apple GPUs: the basic Apple graph under
 *      one sequence kit of @c vram_apple_metric.metal, alignment over packed codes, banded
 *      overlap or window containment over folded UTF-8.
 *
 *  Every node's value is one run of the tape: the kit's sketch of the bytes its face set, none for
 *  a kit building canonically, the sequence, and padding to a word that closes with its own count.
 *  Inserts and refinements score the sketches of a sketching kit and the sequences of any other,
 *  searches score the sequences canonically from the seeds a walk over the sketches found, and a
 *  checkpoint holds the sequences alone, so a load derives every sketch on the device.
 *
 *  Plain C++ for a unit building with Metal, as every Apple face: the kits are MSL the graph
 *  compiles at its first @c reserve, from StringZilla's Metal sources and the pieces each kit
 *  names here, so the host members reading a kit, which pick a launch's rung and size its spans,
 *  live beside its description. Memory is unified, so the kit parameters are one shared buffer the
 *  kernels read in place, and the tables its sketches derive from are another.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - the parameters and the sketch tables into their buffers

#include <limits>      // `std::numeric_limits` - a length no `u32_t` states
#include <span>        // `std::span` - a reshard's sources and targets
#include <type_traits> // `std::is_empty_v`, `std::conditional_t` - a kit taking no parameters
#include <variant>     // `std::monostate` - a kit reading no sketch tables

#include <stringzilla/ngrams.h>       // `sz_ngrams_query_span_bytes`, `sz_ngrams_count` - a query's span
#include <stringzilla/metal.h>        // `sz_metal_prelude_source` - the prelude every StringZilla Metal source requires
#include <stringzilla/ngrams/metal.h> // `sz_ngrams_source_metal` - the n-gram helpers the kits call

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer` - the widths the host members read
#include "usearch/sequence/ram_metric.hpp"   // `alignment_costs_t`, `window_parameters_t`, `spectrum_view_t`
#include "usearch/sequence/ram_index.hpp"    // `vram_stored_bytes`, `vram_stored_sequence` - a tape value's layout
#include "usearch/utilities/vram_apple_shared.hpp"
#include "usearch/sequence/vram_apple_metric.metal" // the kits' rungs and alignment stripes alone, on a host
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

/** Alignment over packed codes: a simdgroup to a sequence, an anti-diagonal wavefront down
 *  stripes of the query, the folded costs uploaded as the parameters. */
struct apple_packed_alignment_kit_t {

    /** The folded costs a face scores with, set once before the first insert. */
    using canonical_parameters_t = alignment_costs_t;

    static_assert(sizeof(alignment_costs_t) == metal_alignment_costs_bytes_k &&
                      offsetof(alignment_costs_t, classes) == 0 && offsetof(alignment_costs_t, gap) == 1 &&
                      offsetof(alignment_costs_t, complements) == 2 &&
                      offsetof(alignment_costs_t, substitutions) == 3 &&
                      alignment_classes_k == metal_alignment_classes_k &&
                      packed_code_bits_k == metal_packed_code_bits_k,
                  "the MSL kit reads the costs as they are laid out");

    /** Builders score smooth sketches, which seed every search too. */
    inline static constexpr std::size_t canonical_rungs_count_k = metal_alignment_canonical_rungs_count_k;
    inline static constexpr bool sketching_k = true;
    inline static constexpr char metal_kit_name_k[] = "sequence::metal_alignment_kit_t";
    inline static constexpr char const* metal_sources_k[] = {sz_metal_prelude_source, sz_ngrams_source_metal,
                                                             metal_sequence_metric_source_k};

    /** Every rung scores a sequence with the whole simdgroup. */
    static constexpr u32_t canonical_lanes_per_entry(std::size_t) noexcept { return metal_simdgroup_width_k; }

    /** The first rung holding whole the stripe of the most codes a @p longest -byte pack holds, the
     *  widest one any number of its stripes, or past every rung before any costs. */
    static std::size_t canonical_rung_serving(std::size_t longest, canonical_parameters_t const& costs) noexcept {
        if (!costs.classes) return canonical_rungs_count_k;
        // A pack of these bytes holds at most this many codes past its one-byte header.
        std::size_t const codes = longest ? (longest - 1) * 8 / packed_code_bits_k : 0;
        std::size_t const widest = canonical_rungs_count_k - 1;
        std::size_t rung = 0;
        while (rung != canonical_rungs_count_k &&
               codes > (rung == widest
                            ? std::size_t {metal_alignment_widest_codes_k}
                            : std::size_t {metal_alignment_rows(static_cast<u32_t>(rung))} * metal_simdgroup_width_k))
            ++rung;
        return rung;
    }

    /** Bytes the packed @p bytes long sequence at @p data takes once prepared: its profile stripes,
     *  and one set of boundary rows when it takes more than one stripe. */
    static std::size_t canonical_span_bytes(byte_t const* data, std::size_t bytes,
                                            canonical_parameters_t const&) noexcept {
        return metal_alignment_span_bytes(static_cast<u32_t>(packed_codes_count(data, bytes)));
    }

    /** Lays the costs out as the kit reads them. */
    static std::size_t canonical_parameters_bytes(canonical_parameters_t const&) noexcept {
        return sizeof(canonical_parameters_t);
    }
    static void canonical_parameters_lay_out(canonical_parameters_t const& costs, byte_t* laid) noexcept {
        std::memcpy(laid, &costs, sizeof(costs));
    }

    /** What the producer reads, uploaded once beside the kit's parameters as the buffer the view
     *  borrows. */
    using sketch_tables_t = spectrum_view_t;
    static_assert(spectrum_width_max_k == metal_spectrum_width_max_k &&
                      spectrum_cosines_k == metal_spectrum_cosines_k &&
                      offsetof(spectrum_head_t, classes) == metal_spectrum_cosines_k &&
                      offsetof(spectrum_head_t, columns) == metal_spectrum_cosines_k + 4 * sizeof(u32_t) &&
                      spectrum_steps_offset_k == metal_spectrum_steps_offset_k &&
                      spectrum_phases_offset(4) == metal_spectrum_phases_offset(4) &&
                      spectrum_phases_offset(100) == metal_spectrum_phases_offset(100),
                  "the MSL kit reads the tables as they are laid out");

    /** A smooth sketch sums in registers, so it carves no scratch. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }
};

/** Bytes the window table of a @p bytes long sequence takes at the widths of @p windows, a
 *  multiple of 16, as the Metal prepare lays it out. */
inline std::size_t metal_window_table_bytes(std::size_t bytes, window_parameters_t const& windows) noexcept {
    return round_up_to_multiple<std::size_t>(sz_ngrams_query_span_bytes(bytes, windows.widths, windows.widths_count),
                                             16);
}

/** Window containment over folded UTF-8: a simdgroup files the query's distinct windows at every
 *  width into a table in its span, and a group of a rung's lanes counts the query windows each
 *  candidate holds. The widths are the parameters, behind the folded costs it leaves unread. */
struct apple_utf8_containment_kit_t {

    /** The window widths a face hashes with, set once before the first insert. */
    using canonical_parameters_t = window_parameters_t;

    static_assert(std::size(containment_window_widths_k) == 2 &&
                      containment_window_widths_k[0] == metal_containment_narrow_window_k &&
                      containment_window_widths_k[1] == metal_containment_wide_window_k,
                  "the MSL kit samples the windows the score counts");

    /** Builders score hash sketches, which seed every search too. */
    inline static constexpr std::size_t canonical_rungs_count_k = metal_containment_canonical_rungs_count_k;
    inline static constexpr bool sketching_k = true;
    inline static constexpr char metal_kit_name_k[] = "sequence::metal_containment_kit_t";
    inline static constexpr char const* metal_sources_k[] = {sz_metal_prelude_source, sz_ngrams_source_metal,
                                                             metal_sequence_metric_source_k};

    /** The lane groups of the narrowest rung, each with its own hit words. */
    inline static constexpr u32_t canonical_groups_k = metal_simdgroup_width_k / metal_containment_lanes(0);

    /** A rung is its lanes per sequence. */
    static constexpr u32_t canonical_lanes_per_entry(std::size_t rung) noexcept {
        return metal_containment_lanes(static_cast<u32_t>(rung));
    }

    /** The widest rung whose lanes still roll a widest window of a @p longest -byte sequence, or
     *  past every rung for one longer than a window table places its windows in. */
    static std::size_t canonical_rung_serving(std::size_t longest, canonical_parameters_t const& windows) noexcept {
        if (longest > sz_ngrams_query_bytes_max_k) return canonical_rungs_count_k;
        sz_size_t widest = 0;
        for (sz_size_t width = 0; width != windows.widths_count; ++width)
            widest = max_of(widest, windows.widths[width]);
        std::size_t rung = 0;
        while (rung + 1 < canonical_rungs_count_k && canonical_lanes_per_entry(rung + 1) * widest <= longest) ++rung;
        return rung;
    }

    /** Bytes a sequence of @p bytes takes once prepared, from its length alone: its table and
     *  every lane group's hit words. */
    static std::size_t canonical_span_bytes(byte_t const*, std::size_t bytes,
                                            canonical_parameters_t const& windows) noexcept {
        std::size_t const counted = sz_ngrams_count(bytes, windows.widths, windows.widths_count);
        return metal_window_table_bytes(bytes, windows) +
               round_up_to_multiple<std::size_t>(
                   canonical_groups_k * divide_round_up<std::size_t>(counted, 32) * sizeof(u32_t), 16);
    }

    /** Lays the costs and the widths out as the kit reads them. */
    static std::size_t canonical_parameters_bytes(canonical_parameters_t const&) noexcept {
        return sizeof(canonical_parameters_t);
    }
    static void canonical_parameters_lay_out(canonical_parameters_t const& windows, byte_t* laid) noexcept {
        std::memcpy(laid, &windows, sizeof(windows));
    }

    /** A hash sketch reads no tables, and takes a byte per slot in front of every stored
     *  sequence. */
    using sketch_tables_t = std::monostate;

    /** Each simdgroup's slot ranks, four bytes a slot, in threadgroup memory while it sketches. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t sketch_bytes) noexcept {
        return sketch_bytes * sizeof(u32_t);
    }
};

/** Banded overlap: a simdgroup files the query's distinct windows into a table in its span, as a
 *  containment does, and a lane alone votes a shift with it against a candidate, then aligns the
 *  symbols the shift lays over each other inside a band under the costs. The widths and the costs
 *  with the complement are the parameters. */
struct apple_u8_banded_overlap_kit_t {

    /** The window widths a face hashes with and the costs and complement it scores under, set once
     *  before the first insert. */
    using canonical_parameters_t = window_parameters_t;

    /** One rung: a lane to a sequence, which builders score canonically on too. */
    inline static constexpr std::size_t canonical_rungs_count_k = 1;
    inline static constexpr bool sketching_k = false;
    inline static constexpr char metal_kit_name_k[] = "sequence::metal_banded_overlap_kit_t";
    inline static constexpr char const* metal_sources_k[] = {sz_metal_prelude_source, sz_ngrams_source_metal,
                                                             metal_sequence_metric_source_k};

    static constexpr u32_t canonical_lanes_per_entry(std::size_t) noexcept { return 1; }

    /** The one rung, or past it for a sequence longer than a window table places its windows in. */
    static std::size_t canonical_rung_serving(std::size_t longest, canonical_parameters_t const&) noexcept {
        return longest > sz_ngrams_query_bytes_max_k ? canonical_rungs_count_k : 0;
    }

    /** Bytes a sequence of @p bytes takes once prepared, from its length alone: its table, where
     *  the costs complement the complement image's bytes and table, then every lane's vote
     *  counters, one per shift, and the cells of its band. */
    static std::size_t canonical_span_bytes(byte_t const*, std::size_t bytes,
                                            canonical_parameters_t const& windows) noexcept {
        std::size_t const table_bytes = metal_window_table_bytes(bytes, windows);
        std::size_t const complement_bytes = windows.costs.complements
                                                 ? round_up_to_multiple<std::size_t>(bytes, 16) + table_bytes
                                                 : 0;
        std::size_t const cells = 2 * std::size_t {banded_overlap_band(bytes)} + 1;
        return table_bytes + complement_bytes +
               round_up_to_multiple<std::size_t>(metal_simdgroup_width_k * bytes * sizeof(u32_t), 16) +
               round_up_to_multiple<std::size_t>(metal_simdgroup_width_k * cells * sizeof(u32_t), 16);
    }

    /** Lays the costs and the widths out as the kit reads them. */
    static std::size_t canonical_parameters_bytes(canonical_parameters_t const&) noexcept {
        return sizeof(canonical_parameters_t);
    }
    static void canonical_parameters_lay_out(canonical_parameters_t const& windows, byte_t* laid) noexcept {
        std::memcpy(laid, &windows, sizeof(windows));
    }

    /** No sketch: builders score canonically, and a stored value is its class bytes closed to a
     *  word. */
    using sketch_tables_t = std::monostate;

    /** No sketch, so no scratch beside one. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }
};

/** The Apple kits on the ladders the faces and tests pick a kind's kit from, at Metal's
 *  capability. */
template <sz_capability_t caps_>
    requires((caps_ & sz_cap_metal_k) != 0)
struct containment_ladder<caps_> {
    using kit_t = apple_utf8_containment_kit_t;
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_metal_k) != 0)
struct banded_overlap_ladder<caps_> {
    using kit_t = apple_u8_banded_overlap_kit_t;
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_metal_k) != 0)
struct alignment_ladder<caps_> {
    using kit_t = apple_packed_alignment_kit_t;
};

#pragma endregion Kits

#pragma region Face

/** The kit hooks a sequence graph on Apple GPUs reads: the kit @p kit_type_ describes, on its
 *  rungs, over node IDs of @p node_id_type_, with the parameters it scores with. */
template <typename kit_type_, typename node_id_type_>
struct vram_apple_kits {
    using canonical_parameters_t = typename kit_type_::canonical_parameters_t;
    using sketch_tables_t = typename kit_type_::sketch_tables_t;

    /** Rungs of the kit's canonical kernels, and the first of them its builders compile on: one
     *  under a sketching kit, as no length sizes a sketch, and every one otherwise. */
    inline static constexpr std::size_t canonical_rungs_count_k = kit_type_::canonical_rungs_count_k;
    inline static constexpr std::size_t build_rungs_count_k = kit_type_::sketching_k ? 1 : canonical_rungs_count_k;
    inline static constexpr auto const& metal_sources_k = kit_type_::metal_sources_k;
    inline static constexpr auto const& metal_kit_name_k = kit_type_::metal_kit_name_k;

    /**
     *  @brief Keeps the parameters the kit scores with, uploaded at the first @c reserve.
     *
     *  @par Statuses:
     *  - @c out_of_memory_k The host copy found no memory.
     */
    status_t canonical_parameters(canonical_parameters_t parameters) noexcept {
        std::size_t const bytes = kit_type_::canonical_parameters_bytes(parameters);
        if (status_t const sized = host_canonical_parameters_.resize(divide_round_up(bytes, sizeof(u32_t)));
            failed(sized))
            return sized;
        kit_type_::canonical_parameters_lay_out(
            parameters, bytes_of(host_canonical_parameters_.data(), host_canonical_parameters_.size()).data());
        kept_canonical_parameters_ = parameters;
        return status_code_t::success_k;
    }

    /**
     *  @brief Keeps the kit's sketch tables, which the first @c reserve uploads to the device,
     *      nothing for a kit reading none, and the @p sketch_bytes of every stored sketch.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The bytes are not a positive multiple of four for a sketching kit, or
     *    the tables hold fewer columns; or they are not zero for a kit writing no sketch.
     *  - @c out_of_memory_k The host copy found no memory.
     */
    status_t sketch_tables(sketch_tables_t const& tables, std::size_t sketch_bytes) noexcept {
        // A sketch lands on whole words, and a kit writing none stores none.
        if (kit_type_::sketching_k ? !sketch_bytes || sketch_bytes % sizeof(u32_t) : sketch_bytes != 0)
            return {status_code_t::config_mismatch_k, "Sketch bytes not words, or given to a sketchless kit"};
        sketch_bytes_ = static_cast<u32_t>(sketch_bytes);
        // A kit reading no tables keeps none, and none is uploaded.
        if constexpr (std::is_empty_v<sketch_tables_t>) return status_code_t::success_k;
        else {
            if (tables.head->columns < sketch_bytes)
                return {status_code_t::config_mismatch_k, "Sketch tables narrower than the sketch"};
            bytes_view_t const laid = spectrum_bytes_of(tables);
            if (status_t const sized = host_sketch_tables_.resize_uninitialized(laid.size()); failed(sized))
                return sized;
            std::memcpy(host_sketch_tables_.data(), laid.data(), laid.size());
            return status_code_t::success_k;
        }
    }

    /** Bytes of the sketch in front of every stored sequence and of a query's; zero for a kit
     *  writing none. */
    std::size_t sketch_bytes() const noexcept { return sketch_bytes_; }

  private:
    template <typename, typename, typename, typename>
    friend struct basic::vram_apple_graph;

    /** The kit's rung serving @p role over values of @p longest bytes under the kept parameters, or
     *  @c canonical_rungs_count_k past every one: by the value scored, but a sketch builder on the
     *  first, whatever the stored values' lengths. */
    std::size_t rung_serving_(score_role_t role, std::size_t longest) const noexcept {
        if (role == score_role_t::build_k && kit_type_::sketching_k) return 0;
        return kit_type_::canonical_rung_serving(longest, kept_canonical_parameters_);
    }

    /** Lanes the kit scores a value with for @p role on @p rung : one for a builder's sketch. */
    static u32_t lanes_per_entry_(score_role_t role, std::size_t rung) noexcept {
        return role == score_role_t::build_k && kit_type_::sketching_k ? 1 : kit_type_::canonical_lanes_per_entry(rung);
    }

    /** Bytes the @p bytes long query at @p data takes once prepared for @p role, a multiple of 16:
     *  none for a sketch builder, which reads the stored sketches in place. */
    std::size_t span_bytes_(score_role_t role, byte_t const* data, std::size_t bytes) const noexcept {
        if (role == score_role_t::build_k && kit_type_::sketching_k) return 0;
        return round_up_to_multiple<std::size_t>(
            kit_type_::canonical_span_bytes(data, bytes, kept_canonical_parameters_), 16);
    }

    /** Bytes the @p stored value saves into: its sequence's, as a RAM index saves it. */
    expected<std::size_t> saved_bytes_(bytes_view_t stored) const noexcept {
        return {vram_stored_sequence(stored, sketch_bytes_).size(), status_code_t::success_k};
    }

    /**
     *  @brief Saves the sequence of the @p stored value into @p saved as it is.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The two differ in size.
     */
    status_t save_value_(bytes_view_t stored, bytes_span_t saved) const noexcept {
        return copy_bytes(saved, vram_stored_sequence(stored, sketch_bytes_));
    }

    /** Bytes a sequence of @p saved bytes stores into: behind its sketch, closed to a word. */
    expected<std::size_t> stored_bytes_(std::size_t saved) const noexcept {
        return {vram_stored_bytes(saved, sketch_bytes_), status_code_t::success_k};
    }

    /** Shared bytes each warp of a load or sketch launch carves for the kept sketch bytes, which
     *  the graph's first @c reserve holds against the device's widest block grant. */
    std::size_t warp_scratch_bytes_() const noexcept { return kit_type_::sketch_scratch_bytes(sketch_bytes_); }

    /** A load scatters each saved sequence behind its sketch, which the same kernel derives from
     *  the tables uploaded with the parameters, where the kit has either; the graph's launch
     *  scalars carry the bytes. */
    status_t queue_load_values_(basic::metal_load_step<node_id_type_> step) const noexcept {
        if (device_sketch_tables_) step.tables = &device_sketch_tables_;
        step.warp_scratch_bytes = static_cast<u32_t>(warp_scratch_bytes_());
        return step.launch_kits(vram_kernel_t::load_values_k);
    }

    /** Queues the sketches of an insert's values or a search's queries, from the tables uploaded
     *  with the parameters, where the kit has any; the graph's launch scalars carry the bytes. */
    status_t queue_sketches_(basic::metal_sketch_step<node_id_type_> step) const noexcept
        requires kit_type_::sketching_k
    {
        if (device_sketch_tables_) step.tables = &device_sketch_tables_;
        step.warp_scratch_bytes = static_cast<u32_t>(warp_scratch_bytes_());
        return step.launch_kits(vram_kernel_t::sketch_values_k);
    }

    /**
     *  @brief Copies the laid-out parameters and the sketch tables into shared buffers on
     *      @p context 's device.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The kit reads sketch tables or parameters, or writes sketches, and one
     *    was not set.
     *  - @c out_of_memory_k The device found no memory.
     */
    status_t upload_parameters_(metal_context_t& context) noexcept {
        if ((host_sketch_tables_.empty() && !std::is_empty_v<sketch_tables_t>) ||
            (kit_type_::sketching_k && !sketch_bytes_) ||
            (host_canonical_parameters_.empty() && !std::is_empty_v<canonical_parameters_t>))
            return {status_code_t::uninitialized_k, "Sketch tables, sketch bytes or parameters not set"};
        if (!host_sketch_tables_.empty()) {
            device_sketch_tables_ = metal_allocate(context, host_sketch_tables_.size());
            if (!device_sketch_tables_) return status_code_t::out_of_memory_k;
            std::memcpy(device_sketch_tables_.host, host_sketch_tables_.data(), host_sketch_tables_.size());
        }
        if (host_canonical_parameters_.empty()) return status_code_t::success_k;
        std::size_t const bytes = host_canonical_parameters_.size() * sizeof(u32_t);
        device_canonical_parameters_ = metal_allocate(context, bytes);
        if (!device_canonical_parameters_) return status_code_t::out_of_memory_k;
        std::memcpy(device_canonical_parameters_.host, host_canonical_parameters_.data(), bytes);
        return status_code_t::success_k;
    }

    /** The kit's parameters on the device, empty for a kit taking none. */
    metal_buffer_t const& parameters_() const noexcept { return device_canonical_parameters_; }

    /** The kit's parameters as its host members read them. */
    canonical_parameters_t kept_canonical_parameters_ {};

    /** The kit's parameters laid out as its kernels read them, on the host and, past the first
     *  @c reserve, in a shared buffer. */
    growing_buffer<u32_t, std::allocator<byte_t>> host_canonical_parameters_ {std::allocator<byte_t> {}};
    metal_buffer_t device_canonical_parameters_;

    /** The kit's sketch tables as @ref sketch_tables laid them out, and in a shared buffer past the
     *  first @c reserve, where the sketch phases of an insert, a search and a load read them; empty
     *  for a kit reading none. */
    growing_buffer<byte_t, std::allocator<byte_t>> host_sketch_tables_ {std::allocator<byte_t> {}};
    metal_buffer_t device_sketch_tables_;

    /** Bytes of every stored sketch, as @ref sketch_tables set them. */
    u32_t sketch_bytes_ = 0;
};

/** Similarity Search over byte sequences on Apple GPUs scored by the sequence kit @p kit_type_
 *  describes, on its rungs. Node IDs are @p node_id_type_. It owns a @c basic::vram_apple_graph
 *  over its kits and forwards its verbs, types and constants, which that graph documents. Every
 *  value of its tape is the kit's sketch, the sequence and the closing padding, so @ref insert
 *  takes a tape laid out in that stored form, its sketches left for the device. */
template <typename kit_type_, typename node_id_type_>
struct vram_apple_index {
    /* Components */
    using kits_t = vram_apple_kits<kit_type_, node_id_type_>;
    using canonical_parameters_t = typename kits_t::canonical_parameters_t;
    using sketch_tables_t = typename kits_t::sketch_tables_t;
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
    status_t canonical_parameters(canonical_parameters_t parameters) noexcept {
        if (graph_.connectivity()) return status_code_t::config_mismatch_k;
        return graph_.kits().canonical_parameters(parameters);
    }

    /**
     *  @brief Keeps the tables the kit's sketches read and the @p sketch_bytes of every stored
     *      sketch, frozen once the first @c reserve has uploaded them to the device.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The index is reserved already, or the kits refuse the bytes.
     *  - @c out_of_memory_k The host copy found no memory.
     */
    status_t sketch_tables(sketch_tables_t const& tables, std::size_t sketch_bytes) noexcept {
        if (graph_.connectivity()) return status_code_t::config_mismatch_k;
        return graph_.kits().sketch_tables(tables, sketch_bytes);
    }

    /** Bytes of the sketch in front of every stored sequence; zero for a kit writing none. */
    std::size_t sketch_bytes() const noexcept { return graph_.kits().sketch_bytes(); }

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

    /** Inserts a tape of stored values, refusing with @c config_mismatch_k one holding a value no
     *  @ref vram_stored_bytes sizes: bare sequences, or values closed otherwise. */
    expected<insert_result_t> insert(std::span<key_t const> keys, tape_view<byte_t const> tape,
                                     traversal_context_t& context, insert_config_t config = {}) noexcept {
        if (!vram_tape_is_stored(tape, sketch_bytes()))
            return {{}, {status_code_t::config_mismatch_k, "Tape not laid out behind sketches"}};
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
    status_t adopt_links(typename graph_t::links_view_t links, traversal_context_t& context) noexcept {
        return graph_.adopt_links(links, context);
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
    friend struct util::shard_set;

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
