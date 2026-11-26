/**
 *  @file include/usearch/sequence/ram_metric.hpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Batched similarity kernels over variable-length byte sequences, bound to StringZilla's
 *      per-backend engines.
 *
 *  The sequence index scores one round at a time: @c reset opens it for a query, @c push stages
 *  candidate views, and @c drain scores every staged candidate within the radius the caller still
 *  tells apart. A metric here owns everything a round needs - the query's engine, the staging,
 *  the arena the engine lives in - so one instance serves one thread and never allocates on the
 *  scoring path once sized.
 *
 *  @section sequence_ram_metric_kits Kits
 *
 *  A kit binds one ISA's kernels of one score, and @c alignment_ladder, @c banded_overlap_ladder
 *  and @c containment_ladder pick the widest kit this build compiled in under a capability mask, at
 *  compile time; @c src/ climbs them once per CPU class and hands each metric the kit it picked, so
 *  nothing tests a capability at run time. Each engine draws its blocks from the metric's own
 *  arena, which @c reset rewinds, so after the widest query and the largest round seen nothing
 *  reaches the metric's allocator again.
 *
 *  @section sequence_ram_metric_sketches Sketches
 *
 *  A sketching kit also writes the fixed-size sketch an index stores in front of each sequence and
 *  scores two of them, so builders link a graph by the sketches and searches walk it by the
 *  canonical scores below. An alignment kit writes a smooth sketch, the sum of a spectrum of the
 *  sequence's windows under the tables @c spectrum_tables::derive draws from the folded costs,
 *  compared by squared Euclidean distance. A containment kit writes a hash sketch, the least hash
 *  of the text's windows in each of its slots, compared by the slots that differ. A banded overlap
 *  kit writes none, and its index builds by the canonical score.
 *
 *  @section sequence_ram_metric_alignment Alignment
 *
 *  Global alignment with one linear gap cost over packed class codes, where bytes sharing their
 *  rows and columns of the similarities share a class, at most 32 of them at five bits a code,
 *  scored through a 32×32 table of @c u4 costs. The query unpacks once per round; the round is
 *  scored in ascending length order, and a column whose every cell is already past the radius
 *  stops its candidate.
 *
 *  @section sequence_ram_metric_banded_overlap Banded Overlap
 *
 *  The cost of aligning two sequences where one slides over the other: every window of a candidate
 *  found once in the query votes for the shift between the two places, through the query's table of
 *  distinct windows, and the symbols the most voted shift lays over each other are aligned inside a
 *  band around that diagonal under the folded costs. Sequences are stored a class code a byte, and
 *  the distance is laid out in fixed point as a containment distance is. The costs name each
 *  class's complement, so every query is filed beside its reverse complement and the lower of the
 *  two distances counts.
 *
 *  @section sequence_ram_metric_containment Containment
 *
 *  Window containment through StringZilla's overlap engines over case-folded UTF-8: the query's
 *  distinct windows at both widths are hashed into one table once per round, and every candidate's
 *  window hashes are probed against it, four candidates' prefix-hash chains advancing together. A
 *  distance is the share of the query's distinct windows the candidate lacks, ties broken by the
 *  share of the candidate's windows the query lacks, in fixed point, exact within the radius.
 */
#pragma once

#include <cmath>   // `std::nextafter` - a radius's share, `std::sqrt`, `std::cos`, `std::log` - the sketch tables
#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - the bytes a prepared engine holds, `std::memset` - a group's padding codes

#include <concepts> // `std::convertible_to` - the queries a batched round takes, `std::same_as`
#include <limits>   // `std::numeric_limits` - the distance a failed round reports
#include <memory>   // `std::allocator`, `std::align` - the Skylake kit's cells on a ZMM boundary
#include <numbers>  // `std::numbers::pi` - phases in turns
#include <numeric>  // `std::inner_product` - a class's coordinates against a column's weights
#include <span>     // `std::span`
#include <variant>  // `std::monostate` - the tables of a kit whose sketch reads none

#include <forkunion/types.hpp> // `fu::standard_cache_hints_t`, `fu::prefetch_line_k` - the candidates' prefetches

#include <numkong/set.h>     // `nk_hamming_u8_*` - the distance between two hash sketches
#include <numkong/spatial.h> // `nk_sqeuclidean_i8_*` - the distance between two smooth sketches

#include <stringzilla/stringzilla.hpp>     // `sz::string_view_t`
#include <stringzilla/utf8_uncased_fold.h> // `sz_utf8_uncased_fold_*` - the fold a containment stores under
#include <stringzilla/ngrams.h>            // `sz_ngrams_*` - n-gram overlaps, MinHash sketches and residues

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer`, `growing_arena` - the engines' blocks
#include "usearch/utilities/ram_sorts.hpp"   // `partial_sort` - an alignment round in ascending length order
#include "usearch/utilities/sample.hpp"      // `splitmix64_at`, `uniform_unit_f64` - the tables' draws
#include "usearch/basic/ram_index.hpp"       // `basic::metric_staging_t`
#include "usearch/sequence/alphabet.hpp"     // `alignment_byte_alphabet_k`, `dna_complement`

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

/** StringZilla's allocator over @p arena: a block bumps inside it on a cache line and a free does
 *  nothing, the arena's rewind being the release, so an engine rebuilt every round reuses the bytes
 *  of the widest one before it. */
template <typename arena_type_>
sz_allocator_t engine_allocator(arena_type_& arena) noexcept {
    sz_allocator_t allocator {};
    allocator.allocate = [](sz_size_t bytes, void* handle, sz_stream_t) noexcept -> void* {
        return static_cast<arena_type_*>(handle)->allocate(bytes, 64);
    };
    allocator.free = [](void*, sz_size_t, void*, sz_stream_t) noexcept {};
    allocator.handle = &arena;
    return allocator;
}

/** StringZilla's C view over a C++ one. */
inline sz_string_view_t as_string_view(sz::string_view_t view) noexcept {
    return sz_string_view_t {.start = view.data(), .length = view.size()};
}

/** The strings view of @p queries, read in place through an accessor, so a batch is never
 *  copied. */
template <typename query_type_>
sz_strings_view_t as_strings_view(std::span<query_type_ const> queries) noexcept {
    sz_strings_view_t strings;
    strings.layout = sz_strings_accessor_k;
    strings.count = queries.size();
    strings.data.accessor.handle = queries.data();
    strings.data.accessor.get = [](void const* handle, sz_size_t const* indices, sz_size_t count,
                                   sz_string_view_t* views) noexcept {
        query_type_ const* const listed = static_cast<query_type_ const*>(handle);
        for (sz_size_t position = 0; position != count; ++position)
            views[position] = as_string_view(sz::string_view_t {listed[indices[position]]});
    };
    return strings;
}

#pragma region Sketches

struct alignment_costs_t;
struct alignment_alphabet_t;

/** Images a query scores as under @p costs: itself, and its reverse complement where the costs
 *  name complements. */
constexpr std::size_t complement_images(alignment_costs_t const& costs) noexcept;

/** Writes into @p image, as long as @p sequence, the reverse complement of a sequence stored a
 *  class code a byte under @p costs: its symbols reversed, each as its complement. */
inline void reverse_complement(alignment_costs_t const& costs, bytes_view_t sequence, bytes_span_t image) noexcept;

/** Bytes of the smooth sketch an alignment keeps in front of each stored sequence where its index
 *  names none, one a column. */
inline constexpr std::size_t smooth_sketch_bytes_default_k = 512;

/** Bytes of the hash sketch a containment keeps in front of each stored sequence where its index
 *  names none, one a slot. */
inline constexpr std::size_t hash_sketch_bytes_default_k = 512;

/** The tables of a kit whose sketch reads none, or that writes no sketch. */
inline constexpr std::monostate no_sketch_tables_k {};

/** A kit whose index builds by sketches: the tables its sketch reads, the scalar of a sketch's
 *  columns, the scratch its producer takes, the producer over a stored sequence, and the symmetric
 *  distance between two sketches. A sketch is as wide as the span it is written into. Any other
 *  sequence kit declares @c std::monostate tables, and its index builds by its canonical score. */
template <typename kit_>
concept a_sketching_kit = requires(typename kit_::sketch_tables_t const& tables, bytes_view_t sequence,
                                   bytes_span_t sketch, bytes_span_t scratch, std::size_t sketch_bytes) {
    typename kit_::sketch_scalar_t;
    { kit_::sketch_scratch_bytes(sketch_bytes) } noexcept -> std::same_as<std::size_t>;
    { kit_::sketch_write(tables, sequence, sketch, scratch) } noexcept;
    kit_::sketch_distance;
};

#if STRINGZILLA_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

/** Squared Euclidean distance between two @c int8 sketches of @p count columns on Arm NEON without
 *  the dot-product extension, sixteen columns a step, in the signature of NumKong's verbs. */
inline nk_status_t sqeuclidean_i8_neon(nk_i8_t const* first, nk_i8_t const* second, nk_size_t count, nk_u32_t* result,
                                       nk_stream_t) noexcept {
    uint32x4_t sums_u32x4 = vdupq_n_u32(0);
    nk_size_t column = 0;
    for (; column + 16 <= count; column += 16) {
        // An absolute difference of two `int8` fits a `u8`, and its square a `u16`.
        uint8x16_t const differences_u8x16 = vreinterpretq_u8_s8(
            vabdq_s8(vld1q_s8(first + column), vld1q_s8(second + column)));
        uint16x8_t const low_u16x8 = vmull_u8(vget_low_u8(differences_u8x16), vget_low_u8(differences_u8x16));
        uint16x8_t const high_u16x8 = vmull_high_u8(differences_u8x16, differences_u8x16);
        sums_u32x4 = vpadalq_u16(vpadalq_u16(sums_u32x4, low_u16x8), high_u16x8);
    }
    nk_u32_t sum = vaddvq_u32(sums_u32x4);
    for (; column != count; ++column) {
        nk_i32_t const difference = nk_i32_t {first[column]} - second[column];
        sum += static_cast<nk_u32_t>(difference * difference);
    }
    *result = sum;
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_NEON

#pragma endregion Sketches

#pragma region Containment Kernels

/** The widths of the windows a containment counts and its hash sketch samples, in bytes of folded
 *  UTF-8: the two the text experiments ranked by. */
inline constexpr sz_size_t containment_window_widths_k[] = {5, 8};

/** Scalar: one position per hashing step, one key per sort and probe step. */
struct utf8_serial_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_serial;
    static constexpr auto canonical_overlaps = sz_ngrams_overlaps_serial;

    /** Folds the case of a text before it is stored or scored. */
    static constexpr auto canonical_fold = sz_utf8_uncased_fold_serial;

    /** A hash sketch reads no tables. */
    using sketch_tables_t = std::monostate;
    using sketch_scalar_t = u8_t;

    /** StringZilla's per-text producer of this backend. */
    static constexpr auto sketch_text = sz_ngrams_sketch_serial;

    /** Scratch a sketch of @p sketch_bytes slots takes: one rank a slot. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t sketch_bytes) noexcept {
        return sketch_bytes * sizeof(sz_u32_t);
    }

    /** Writes the hash sketch of the stored @p sequence, folded UTF-8, into @p sketch, ranking its
     *  windows in @p scratch. */
    static void sketch_write(sketch_tables_t const&, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t scratch) noexcept {
        sketch_text(chars_of(sequence).data(), sequence.size(), containment_window_widths_k,
                    std::size(containment_window_widths_k), elements_of<sz_u32_t>(scratch).data(), sketch.size(),
                    elements_of<u8_t>(sketch).data());
    }

    /** Slots two hash sketches differ in. */
    static constexpr auto sketch_distance = nk_hamming_u8_serial;
};

#if STRINGZILLA_TARGET_HASWELL

/** Intel Haswell AVX2: four positions per hashing step, eight keys per sort and probe step. */
struct utf8_haswell_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_haswell;
    static constexpr auto canonical_overlaps = sz_ngrams_overlaps_haswell;

    /** Folds the case of a text before it is stored or scored. */
    static constexpr auto canonical_fold = sz_utf8_uncased_fold_haswell;

    /** A hash sketch reads no tables. */
    using sketch_tables_t = std::monostate;
    using sketch_scalar_t = u8_t;

    /** StringZilla's per-text producer of this backend. */
    static constexpr auto sketch_text = sz_ngrams_sketch_haswell;

    /** Scratch a sketch of @p sketch_bytes slots takes: one rank a slot. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t sketch_bytes) noexcept {
        return sketch_bytes * sizeof(sz_u32_t);
    }

    /** Writes the hash sketch of the stored @p sequence, folded UTF-8, into @p sketch, ranking its
     *  windows in @p scratch. */
    static void sketch_write(sketch_tables_t const&, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t scratch) noexcept {
        sketch_text(chars_of(sequence).data(), sequence.size(), containment_window_widths_k,
                    std::size(containment_window_widths_k), elements_of<sz_u32_t>(scratch).data(), sketch.size(),
                    elements_of<u8_t>(sketch).data());
    }

    /** Slots two hash sketches differ in. */
    static constexpr auto sketch_distance = nk_hamming_u8_haswell;
};
#else

/** The kit this build leaves out, kept so @c containment_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct utf8_haswell_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_TARGET_SKYLAKE

/** Intel Skylake AVX-512: eight positions per hashing step, sixteen keys per sort and probe
 *  step; Haswell's fold and sketch distance, neither having a Skylake form. */
struct utf8_skylake_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_skylake;
    static constexpr auto canonical_overlaps = sz_ngrams_overlaps_skylake;

    /** Folds the case of a text before it is stored or scored. */
    static constexpr auto canonical_fold = sz_utf8_uncased_fold_haswell;

    /** A hash sketch reads no tables. */
    using sketch_tables_t = std::monostate;
    using sketch_scalar_t = u8_t;

    /** StringZilla's per-text producer of this backend. */
    static constexpr auto sketch_text = sz_ngrams_sketch_skylake;

    /** Scratch a sketch of @p sketch_bytes slots takes: one rank a slot. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t sketch_bytes) noexcept {
        return sketch_bytes * sizeof(sz_u32_t);
    }

    /** Writes the hash sketch of the stored @p sequence, folded UTF-8, into @p sketch, ranking its
     *  windows in @p scratch. */
    static void sketch_write(sketch_tables_t const&, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t scratch) noexcept {
        sketch_text(chars_of(sequence).data(), sequence.size(), containment_window_widths_k,
                    std::size(containment_window_widths_k), elements_of<sz_u32_t>(scratch).data(), sketch.size(),
                    elements_of<u8_t>(sketch).data());
    }

    /** Slots two hash sketches differ in. */
    static constexpr auto sketch_distance = nk_hamming_u8_haswell;
};
#else

/** The kit this build leaves out, kept so @c containment_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct utf8_skylake_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_ICELAKE

/** Intel Ice Lake: Skylake's engine and sketch producer under its own fold and sketch distance. */
struct utf8_icelake_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_skylake;
    static constexpr auto canonical_overlaps = sz_ngrams_overlaps_skylake;

    /** Folds the case of a text before it is stored or scored. */
    static constexpr auto canonical_fold = sz_utf8_uncased_fold_icelake;

    /** A hash sketch reads no tables. */
    using sketch_tables_t = std::monostate;
    using sketch_scalar_t = u8_t;

    /** StringZilla's per-text producer of this backend. */
    static constexpr auto sketch_text = sz_ngrams_sketch_skylake;

    /** Scratch a sketch of @p sketch_bytes slots takes: one rank a slot. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t sketch_bytes) noexcept {
        return sketch_bytes * sizeof(sz_u32_t);
    }

    /** Writes the hash sketch of the stored @p sequence, folded UTF-8, into @p sketch, ranking its
     *  windows in @p scratch. */
    static void sketch_write(sketch_tables_t const&, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t scratch) noexcept {
        sketch_text(chars_of(sequence).data(), sequence.size(), containment_window_widths_k,
                    std::size(containment_window_widths_k), elements_of<sz_u32_t>(scratch).data(), sketch.size(),
                    elements_of<u8_t>(sketch).data());
    }

    /** Slots two hash sketches differ in. */
    static constexpr auto sketch_distance = nk_hamming_u8_icelake;
};
#else

/** The kit this build leaves out, kept so @c containment_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct utf8_icelake_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_ICELAKE

#if STRINGZILLA_TARGET_NEON

/** Arm NEON: two positions per hashing step, four keys per sort and probe step. */
struct utf8_neon_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_neon;
    static constexpr auto canonical_overlaps = sz_ngrams_overlaps_neon;

    /** Folds the case of a text before it is stored or scored. */
    static constexpr auto canonical_fold = sz_utf8_uncased_fold_neon;

    /** A hash sketch reads no tables. */
    using sketch_tables_t = std::monostate;
    using sketch_scalar_t = u8_t;

    /** StringZilla's per-text producer of this backend. */
    static constexpr auto sketch_text = sz_ngrams_sketch_neon;

    /** Scratch a sketch of @p sketch_bytes slots takes: one rank a slot. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t sketch_bytes) noexcept {
        return sketch_bytes * sizeof(sz_u32_t);
    }

    /** Writes the hash sketch of the stored @p sequence, folded UTF-8, into @p sketch, ranking its
     *  windows in @p scratch. */
    static void sketch_write(sketch_tables_t const&, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t scratch) noexcept {
        sketch_text(chars_of(sequence).data(), sequence.size(), containment_window_widths_k,
                    std::size(containment_window_widths_k), elements_of<sz_u32_t>(scratch).data(), sketch.size(),
                    elements_of<u8_t>(sketch).data());
    }

    /** Slots two hash sketches differ in. */
    static constexpr auto sketch_distance = nk_hamming_u8_neon;
};
#else

/** The kit this build leaves out, kept so @c containment_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct utf8_neon_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_NEON

#if STRINGZILLA_TARGET_POWERVSX

/** IBM Power VSX: two positions per hashing step, four keys per sort and probe step. */
struct utf8_powervsx_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_powervsx_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_powervsx;
    static constexpr auto canonical_overlaps = sz_ngrams_overlaps_powervsx;

    /** Folds the case of a text before it is stored or scored. */
    static constexpr auto canonical_fold = sz_utf8_uncased_fold_powervsx;

    /** A hash sketch reads no tables. */
    using sketch_tables_t = std::monostate;
    using sketch_scalar_t = u8_t;

    /** StringZilla's per-text producer of this backend. */
    static constexpr auto sketch_text = sz_ngrams_sketch_powervsx;

    /** Scratch a sketch of @p sketch_bytes slots takes: one rank a slot. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t sketch_bytes) noexcept {
        return sketch_bytes * sizeof(sz_u32_t);
    }

    /** Writes the hash sketch of the stored @p sequence, folded UTF-8, into @p sketch, ranking its
     *  windows in @p scratch. */
    static void sketch_write(sketch_tables_t const&, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t scratch) noexcept {
        sketch_text(chars_of(sequence).data(), sequence.size(), containment_window_widths_k,
                    std::size(containment_window_widths_k), elements_of<sz_u32_t>(scratch).data(), sketch.size(),
                    elements_of<u8_t>(sketch).data());
    }

    /** Slots two hash sketches differ in. */
    static constexpr auto sketch_distance = nk_hamming_u8_powervsx;
};
#else

/** The kit this build leaves out, kept so @c containment_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct utf8_powervsx_containment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_powervsx_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_POWERVSX

/** The widest containment kit this build compiled in among the backends @p caps_ admits, newest
 *  first; each kit states its @c capability_k, and a stub its @c compiled_k as false, so an
 *  unselected kit's body never has to exist. */
template <sz_capability_t caps_>
struct containment_ladder;

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cpus_k) != 0)
struct containment_ladder<caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        utf8_serial_containment_kit_t,  //
        utf8_icelake_containment_kit_t, //
        utf8_skylake_containment_kit_t, //
        utf8_haswell_containment_kit_t, //
        utf8_neon_containment_kit_t,    //
        utf8_powervsx_containment_kit_t //
        >::type;
};

#pragma endregion Containment Kernels

#pragma region Containment Metric

/** Low bits of a share distance, a containment's or a banded overlap's, which order candidates tied
 *  on the primary share by the secondary one. */
inline constexpr unsigned share_secondary_bits_k = 8;

/** The top level of a share distance's primary share: 15 bits, so every containment query of fewer
 *  distinct windows keeps each count of misses apart. */
inline constexpr u64_t share_primary_levels_k = (u64_t(1) << 15) - 1;

/** One past the largest share distance: 2^23, so a @c f32 short of a mantissa bit still
 *  holds every distance exactly. */
inline constexpr sz_size_t share_resolution_k = sz_size_t(1) << 23;

/**
 *  @brief The containment distance of a candidate holding @p held of the query's @p distinct
 *      windows and spanning @p candidate_windows windows of its own over every width.
 *
 *  Lexicographic in fixed point: the query's distinct windows the candidate lacks, over all of
 *  them, rounded down to @ref share_primary_levels_k, above @ref share_secondary_bits_k bits of the
 *  candidate's windows the query lacks, over all of them, rounded up. A candidate holding every
 *  query window sits at zero only when it adds no window of its own, so the exact row ranks above
 *  every longer row containing it. Never increasing in @p held, so @p held at its largest bounds
 *  the distance from below.
 *
 *  @verbatim
 *  primary = distinct ? (distinct - held) * (2^15 - 1) / distinct : 2^15 - 1
 *  secondary = windows ? ceil((windows - min(windows, held)) * 255 / windows) : 255
 *  distance = primary * 256 + secondary
 *  @endverbatim
 */
constexpr u64_t containment_distance(u64_t distinct, u64_t held, u64_t candidate_windows) noexcept {
    u64_t const secondary_top = (u64_t(1) << share_secondary_bits_k) - 1;
    u64_t const primary = distinct ? (distinct - held) * share_primary_levels_k / distinct : share_primary_levels_k;
    u64_t const uncovered = candidate_windows - (held < candidate_windows ? held : candidate_windows);
    u64_t const secondary = candidate_windows ? divide_round_up<u64_t>(uncovered * secondary_top, candidate_windows)
                                              : secondary_top;
    return (primary << share_secondary_bits_k) | secondary;
}

/**
 *  @brief Window containment over one round of candidates, through one table of the query's
 *      distinct windows at every width.
 *
 *  @c reset hashes the query's windows into the table, @c push stages candidate views, and
 *  @c drain streams every staged candidate through it, one prefix-hash chain per candidate and one
 *  probe per window.
 *
 *  Windows are taken at @ref containment_window_widths_k. One instance belongs to one thread and
 *  owns its engine, the arena the engine lives in, the staging and the match counts. The staging
 *  and the counts are sized once by @c init, and the engine's tree and chains bump in the arena,
 *  so a round allocates nothing once a query and a round as long have been scored. A search's
 *  distance is the query's distinct windows the candidate lacks, over the query's distinct
 *  windows, ties broken by the candidate's windows the query lacks, as @ref containment_distance
 *  lays both out below @c share_resolution_k: zero only where the candidate holds every query
 *  window and nothing more.
 *
 *  @tparam kit_type_ The engine of one ISA backend, as @c containment_ladder picks it.
 *  @tparam allocator_type_ places every buffer, one set per metric.
 *  @tparam cache_hints_type_ prefetches a staged candidate's bytes; ForkUnion's no-op policy by
 *      default.
 */
template <typename kit_type_ = typename containment_ladder<sz_cap_any_k>::kit_t,
          typename allocator_type_ = std::allocator<byte_t>, typename cache_hints_type_ = fu::standard_cache_hints_t>
class containment_metric {
  public:
    using kit_t = kit_type_;
    using distance_t = sz_size_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;

    /** The flat form a batched round hands @ref reset for each stored sequence: its bytes. */
    using query_view_t = sz::string_view_t;
    template <typename string_type_>
    static query_view_t query_view(string_type_ const& value) noexcept {
        return query_view_t {value};
    }

  private:

    /** The query's engine, one query's table of its distinct windows at every width, a query
     *  without windows as a table of none, and nothing before the first @ref reset or after a move.
     *  Its blocks live in @c arena_, and its chains grow there to the longest seen. */
    sz_ngrams_engine_t engine_ {};

    /** Where the engine's blocks bump, rewound whenever @ref reset prepares a new query. */
    growing_arena<allocator_t> arena_;

    /** The round's candidates in push order. */
    growing_buffer<sz_string_view_t, allocator_t> staged_;

    /** The bytes @c engine_ was prepared from, so a round reopening the same query keeps the
     *  engine; grows to the longest query seen. */
    growing_buffer<char, allocator_t> prepared_;

    /** The round's @b [queries,candidates,widths] match counts, a staged candidate's by its
     *  push slot. */
    growing_buffer<sz_u32_t, allocator_t> matches_;

    /** How the last @ref init, @ref reset or @ref drain fared. */
    status_t last_status_ = status_code_t::uninitialized_k;

  public:
    containment_metric(allocator_t const& allocator = {}) noexcept
        : arena_(allocator), staged_(allocator), prepared_(allocator), matches_(allocator) {}
    containment_metric(containment_metric const&) = delete;
    containment_metric& operator=(containment_metric const&) = delete;

    /** Takes the buffers of @p other but not its engine, whose allocator names the arena it left,
     *  so the next @ref reset prepares the query again. */
    containment_metric(containment_metric&& other) noexcept
        : arena_(std::move(other.arena_)), staged_(std::move(other.staged_)), prepared_(std::move(other.prepared_)),
          matches_(std::move(other.matches_)), last_status_(other.last_status_) {}
    containment_metric& operator=(containment_metric&&) = delete;

    /**
     *  @brief Sizes the staging and the match counts.
     *
     *  @param[in] staging_limit Candidates one round may hold before @c push refuses: at least the
     *      index's round capacity, <tt>max(fan_out * connectivity, connectivity + 1)</tt>, or rounds
     *      saturate rather than fail.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p staging_limit is zero.
     */
    status_t init(std::size_t staging_limit) noexcept {
        if (!staging_limit) return status_code_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        last_status_ = matches_.resize_uninitialized(staged_.capacity() * std::size(containment_window_widths_k));
        return last_status_;
    }

    /** How the last @ref init, @ref reset or @ref drain fared - the only place a failure surfaces,
     *  since the scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a round for @p query, building its table unless it holds these bytes already, and
     *  discarding whatever the previous round staged. */
    void reset(sz::string_view_t query) noexcept {
        staged_.clear();
        // A batch's engine never matches, though its cleared bytes equal an empty query's.
        if (engine_.count == 1 && succeeded(last_status_) && to_string_view(prepared_) == to_string_view(query)) return;
        engine_ = sz_ngrams_engine_t {};
        arena_.clear();
        // Grow-only, so a thread's arena keeps one block per doubling rather than one per query.
        prepared_.clear();
        last_status_ = prepared_.reserve(query.size());
        if (failed(last_status_)) return;
        prepared_.extend_uninitialized(query.size());
        if (!query.empty()) std::memcpy(prepared_.data(), query.data(), query.size());
        sz_string_view_t const view = as_string_view(query);
        sz_strings_view_t queries;
        sz_strings_from_string_views(&view, 1, &queries);
        prepare_(queries);
    }

    /** Opens rounds scoring every query @p queries holds at once, each in its own table, so each
     *  @ref drain writes @b [queries,candidates]; the views are read in place while the tables are
     *  built, and a batch of one is a single query. */
    template <typename query_type_>
        requires std::convertible_to<query_type_ const&, sz::string_view_t>
    void reset(std::span<query_type_ const> queries) noexcept {
        if (queries.size() == 1) return reset(sz::string_view_t {queries.front()});
        staged_.clear();
        engine_ = sz_ngrams_engine_t {};
        arena_.clear();
        prepared_.clear();
        last_status_ = matches_.resize_uninitialized(queries.size() * staged_.capacity() *
                                                     std::size(containment_window_widths_k));
        if (failed(last_status_)) return;
        prepare_(as_strings_view(queries));
    }

    /** Stages one candidate, or refuses once the round is full. */
    basic::metric_staging_t push(sz::string_view_t value) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        prefetch_(value);
        staged_.push_back_reserved(as_string_view(value));
        return basic::metric_staging_t::staged_k;
    }

    /**
     *  @brief Scores everything staged in one sweep and closes the round.
     *
     *  Distances up to and including @p radius are exact; above it the value is some value above
     *  the radius. Each pair may miss only as many of its query's distinct windows as keep its
     *  distance within @p radius, and a pair past its budget holds none, a full-scale distance. A
     *  failed round reports the furthest distance, so a full pool rejects it and an unfilled pool
     *  admits it last; @ref status is the only signal it leaves.
     *
     *  @param[out] distances One slot per staged candidate, in push order, for every prepared
     *      query, laid out query-major.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<sz_size_t> distances, sz_size_t radius) noexcept {
        std::size_t const count = staged_.size(), widths = std::size(containment_window_widths_k);
        sz_f32_t const share = share_of_(radius);
        sz_strings_view_t candidates;
        sz_strings_from_string_views(staged_.data(), count, &candidates);
        bool const scored = succeeded(last_status_) && distances.size() == engine_.count * count &&
                            kit_t::canonical_overlaps(&engine_, &candidates, share, matches_.data(), count * widths,
                                                      widths, nullptr) == sz_success_k;
        if (!scored && succeeded(last_status_)) last_status_ = status_code_t::out_of_memory_k;
        for (std::size_t slot = 0; slot != distances.size(); ++slot)
            distances[slot] = scored ? distance_(slot / count, matches_.data() + slot * widths,
                                                 staged_[slot % count].length)
                                     : (std::numeric_limits<sz_size_t>::max)();
        staged_.clear();
    }

  private:

    /** Hashes @p queries into the engine, leaving it empty and the status failed on a refusal. */
    void prepare_(sz_strings_view_t const& queries) noexcept {
        sz_allocator_t allocator = engine_allocator(arena_);
        sz_status_t const prepared = kit_t::canonical_engine_init(&engine_, &queries, containment_window_widths_k,
                                                                  std::size(containment_window_widths_k), 0, &allocator,
                                                                  nullptr);
        if (prepared == sz_success_k) return;
        engine_ = sz_ngrams_engine_t {};
        prepared_.clear();
        last_status_ = prepared == sz_bad_alloc_k ? status_code_t::out_of_memory_k : status_code_t::shape_mismatch_k;
    }

    /** Starts the misses on every line of @p value, which the drain reads. */
    static void prefetch_(sz::string_view_t value) noexcept {
        for (std::size_t offset = 0; offset < value.size(); offset += fu::default_alignment_k)
            cache_hints_type_ {}(value.data() + offset, fu::prefetch_line_k);
    }

    /** The share of its query's distinct windows a pair within @p radius may miss: every pair in
     *  the radius must count exactly, so the share rounds up past the radius's level plus one, over
     *  the primary levels. */
    static sz_f32_t share_of_(sz_size_t radius) noexcept {
        u64_t const level = radius >> share_secondary_bits_k;
        if (level >= share_primary_levels_k) return 1.0f;
        f64_t const share = static_cast<f64_t>(level + 1) / static_cast<f64_t>(share_primary_levels_k);
        return std::nextafter(static_cast<sz_f32_t>(share), 1.0f);
    }

    /** The distance of the pair of the engine's table @p filed off its @p held counts per width
     *  and the @p length bytes of its candidate. */
    sz_size_t distance_(std::size_t filed, sz_u32_t const* held, std::size_t length) const noexcept {
        sz_ngrams_query_t const query = sz_ngrams_engine_query(&engine_, filed);
        u64_t matched = 0;
        for (std::size_t index = 0; index != query.widths_count; ++index) matched += held[index];
        u64_t const windows = sz_ngrams_count(length, query.widths, query.widths_count);
        return static_cast<sz_size_t>(containment_distance(query.distinct, matched, windows));
    }
};

using containment_metric_t = containment_metric<>;

#pragma endregion Containment Metric

#pragma region Alignment Kernels

/** Classes an alignment alphabet folds into at most, five bits a code. */
inline constexpr std::size_t alignment_classes_k = 32;

/** The largest folded cost, the top of a @c u4. */
inline constexpr u32_t alignment_cost_ceiling_k = (1u << 4) - 1;

/** Bits one packed class code takes. */
inline constexpr u32_t packed_code_bits_k = 5;

/**
 *  @brief A substitution table and a linear gap cost folded to @c u4 over at most 32 classes.
 *
 *  Costs come from similarities: @c c(a,b) = (s(a,a) + s(b,b)) / 2 − s(a,b), zero on the diagonal
 *  and refused where negative. When the largest cost or the gap passes 15, every cost scales by
 *  the one factor bringing it to 15; each rounds to the nearest integer, halves up.
 *
 *  A nucleotide alphabet's complements ride with them as @c alignment_alphabet_t::fold_complement
 *  records them, so an overlap reads the costs and the complements from one table, on a host and
 *  on a device alike.
 */
struct alignment_costs_t {

    /** Classes the alphabet spans: the codes below this one, zero unfolded. */
    u8_t classes;

    /** Cost of one code facing a gap. */
    u8_t gap;

    /** One where the classes have complements, so a query also scores as its reverse
     *  complement. */
    u8_t complements;

    /** Cost of the row's code aligned to the column's, at most 15. */
    u8_t substitutions[alignment_classes_k][alignment_classes_k];

    /** The class each class pairs with on the other strand. */
    u8_t complemented[alignment_classes_k];

    /**
     *  @brief Folds @p similarities, a row-major square over byte codes whose classes end at the
     *      last code scoring anything, and the @p gap cost.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p similarities is not the byte square, or @p gap is negative.
     *  - @c config_mismatch_k No code scores anything, or the codes span past 32 classes.
     *  - @c config_mismatch_k A pair scores above its mean self-score.
     */
    static expected<alignment_costs_t> fold(std::span<i8_t const> similarities, i8_t gap) noexcept {
        constexpr std::size_t side_k = alignment_byte_alphabet_k;
        std::size_t classes = 0;
        for (std::size_t cell = 0; cell != similarities.size(); ++cell)
            if (similarities[cell]) classes = max_of(classes, max_of(cell / side_k, cell % side_k) + 1);
        return fold(similarities, gap, classes);
    }

    /** Folds the first @p classes codes of @p similarities, as the overload above finds them. */
    static expected<alignment_costs_t> fold(std::span<i8_t const> similarities, i8_t gap,
                                            std::size_t classes) noexcept {
        expected<alignment_costs_t> folded {{}, status_code_t::config_mismatch_k};
        constexpr std::size_t side_k = alignment_byte_alphabet_k;
        if (similarities.size() != side_k * side_k || gap < 0) return folded;
        if (!classes || classes > alignment_classes_k) return folded;
        // Every cost in halves, so a half-integer mean self-score stays exact.
        auto const doubled = [&](std::size_t row, std::size_t column) noexcept {
            return i32_t {similarities[row * side_k + row]} + similarities[column * side_k + column] -
                   2 * i32_t {similarities[row * side_k + column]};
        };
        i32_t largest = 2 * static_cast<i32_t>(alignment_cost_ceiling_k);
        largest = max_of(largest, 2 * i32_t {gap});
        for (std::size_t row = 0; row != classes; ++row)
            for (std::size_t column = 0; column != classes; ++column) {
                if (doubled(row, column) < 0) return {{}, status_code_t::config_mismatch_k};
                largest = max_of(largest, doubled(row, column));
            }
        // `halves / 2` scaled by `30 / largest`, rounded half up, unscaled while `largest` is 30.
        auto const rounded = [&](i32_t halves) noexcept {
            i32_t const ceiling = static_cast<i32_t>(alignment_cost_ceiling_k);
            return static_cast<u8_t>((2 * ceiling * halves + largest) / (2 * largest));
        };
        alignment_costs_t& costs = folded.value;
        costs.classes = static_cast<u8_t>(classes);
        // A positive gap stays positive, or two free gaps would replace any substitution.
        costs.gap = max_of<u8_t>(rounded(2 * i32_t {gap}), gap ? 1 : 0);
        for (std::size_t row = 0; row != classes; ++row)
            for (std::size_t column = 0; column != classes; ++column)
                costs.substitutions[row][column] = rounded(doubled(row, column));
        return {costs, status_code_t::success_k};
    }
};

/** Codes the packed @p bytes long @p sequence holds: as many as its bytes past the first fit, less
 *  the unused tail slots that first byte counts. */
constexpr std::size_t packed_codes_count(byte_t const* sequence, std::size_t bytes) noexcept {
    if (!bytes) return 0;
    std::size_t const slots = (bytes - 1) * 8 / packed_code_bits_k, unused = std::to_integer<std::size_t>(sequence[0]);
    return slots > unused ? slots - unused : 0;
}

/** The code at @p index of the packed @p sequence, little-endian from the byte past its first; a
 *  code reads its second byte only when it crosses into one. */
constexpr u32_t packed_code_at(byte_t const* sequence, std::size_t index) noexcept {
    std::size_t const bit = index * packed_code_bits_k, byte = 1 + (bit >> 3);
    u32_t const shift = bit & 7u;
    u32_t word = std::to_integer<u32_t>(sequence[byte]);
    if (shift + packed_code_bits_k > 8) word |= std::to_integer<u32_t>(sequence[byte + 1]) << 8;
    return (word >> shift) & ((1u << packed_code_bits_k) - 1u);
}

/** Bytes @p codes take packed, the leading count of unused slots included. */
constexpr std::size_t packed_codes_bytes(std::size_t codes) noexcept {
    return 1 + divide_round_up<std::size_t>(codes * packed_code_bits_k, 8);
}

/** Packs the @p count codes @p code_at hands out by index into @p sequence,
 *  @ref packed_codes_bytes long, behind the count of slots its last bytes leave unused. */
template <typename code_at_type_>
void pack_codes_of(std::size_t count, code_at_type_&& code_at, bytes_span_t sequence) noexcept {
    std::memset(sequence.data(), 0, sequence.size());
    sequence[0] = static_cast<byte_t>((sequence.size() - 1) * 8 / packed_code_bits_k - count);
    for (std::size_t index = 0; index != count; ++index) {
        std::size_t const bit = index * packed_code_bits_k, byte = 1 + (bit >> 3);
        u32_t const shifted = u32_t {code_at(index)} << (bit & 7u);
        sequence[byte] |= static_cast<byte_t>(shifted & 0xFFu);
        if (shifted > 0xFFu) sequence[byte + 1] |= static_cast<byte_t>(shifted >> 8);
    }
}

/** Packs @p codes into @p sequence, @ref packed_codes_bytes long. */
inline void pack_codes(std::span<u8_t const> codes, bytes_span_t sequence) noexcept {
    pack_codes_of(codes.size(), [&](std::size_t index) noexcept { return codes[index]; }, sequence);
}

/**
 *  @brief The classes an alignment folds bytes into: bytes sharing their rows and columns of the
 *      similarities share a class, numbered in the order their first bytes come.
 *
 *  A protein alignment stores and scores its strings as these codes, packed, and a
 *  DNA overlap as one code a byte; the fold is lossy, so the codes are what a stored
 *  string reads back as.
 */
struct alignment_alphabet_t {

    /** The class code of every byte. */
    u8_t codes[alignment_byte_alphabet_k];

    /**
     *  @brief Folds the row-major byte square of @p similarities into @p alphabet, and its classes'
     *      similarities with the @p gap cost into @p costs, as @c alignment_costs_t::fold does.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The bytes fold into more than 32 classes, or
     *    @c alignment_costs_t::fold refuses the folded square.
     */
    static status_t fold(std::span<i8_t const> similarities, i8_t gap, alignment_alphabet_t& alphabet,
                         alignment_costs_t& costs) noexcept {
        constexpr std::size_t side_k = alignment_byte_alphabet_k;
        if (similarities.size() != side_k * side_k) return status_code_t::config_mismatch_k;
        auto const alike = [&](std::size_t first, std::size_t second) noexcept {
            for (std::size_t other = 0; other != side_k; ++other)
                if (similarities[first * side_k + other] != similarities[second * side_k + other] ||
                    similarities[other * side_k + first] != similarities[other * side_k + second])
                    return false;
            return true;
        };
        std::size_t firsts[alignment_classes_k];
        std::size_t classes = 0;
        for (std::size_t byte = 0; byte != side_k; ++byte) {
            std::size_t code = 0;
            while (code != classes && !alike(firsts[code], byte)) ++code;
            if (code == classes) {
                if (classes == alignment_classes_k) return status_code_t::config_mismatch_k;
                firsts[classes++] = byte;
            }
            alphabet.codes[byte] = static_cast<u8_t>(code);
        }
        i8_t folded[side_k * side_k] {};
        for (std::size_t row = 0; row != classes; ++row)
            for (std::size_t column = 0; column != classes; ++column)
                folded[row * side_k + column] = similarities[firsts[row] * side_k + firsts[column]];
        auto const [class_costs, status] = alignment_costs_t::fold({folded, side_k * side_k}, gap, classes);
        if (succeeded(status)) costs = class_costs;
        return status;
    }

    /**
     *  @brief Records in @p costs the complement of every class of a nucleotide alphabet: the
     *      classes of @c A and @c T trade places, and those of @c C and @c G.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k The four letters do not take the four classes, one apiece, or a
     *    letter of either case, @c U read as @c T, sits in a class its complement's class does
     *    not pair with.
     */
    status_t fold_complement(alignment_costs_t& costs) const noexcept {
        constexpr char letters_k[] = {'A', 'C', 'G', 'T'};
        costs.complements = 0;
        std::memset(costs.complemented, 0, alignment_classes_k);
        if (costs.classes != std::size(letters_k)) return status_code_t::config_mismatch_k;
        u32_t taken = 0;
        for (char const letter : letters_k) {
            u8_t const code = codes[static_cast<u8_t>(letter)];
            taken |= 1u << code;
            costs.complemented[code] = codes[static_cast<u8_t>(dna_complement(letter))];
        }
        if (taken != (1u << std::size(letters_k)) - 1u) return status_code_t::config_mismatch_k;
        // Bytes of one class read alike, so their complements must read alike too.
        for (std::size_t byte = 0; byte != alignment_byte_alphabet_k; ++byte)
            if (char const complement = dna_complement(static_cast<char>(byte));
                complement && costs.complemented[codes[byte]] != codes[static_cast<u8_t>(complement)])
                return status_code_t::config_mismatch_k;
        costs.complements = 1;
        return status_code_t::success_k;
    }

    /** Writes @p bytes as their class codes, one a byte, into @p sequence, as long. */
    void encode(bytes_view_t bytes, bytes_span_t sequence) const noexcept {
        for (std::size_t symbol = 0; symbol != bytes.size(); ++symbol)
            sequence[symbol] = static_cast<byte_t>(codes[std::to_integer<std::size_t>(bytes[symbol])]);
    }

    /** Packs @p bytes as their class codes into @p sequence, @ref packed_codes_bytes long. */
    void pack(bytes_view_t bytes, bytes_span_t sequence) const noexcept {
        auto const code_at = [&](std::size_t index) noexcept {
            return codes[std::to_integer<std::size_t>(bytes[index])];
        };
        pack_codes_of(bytes.size(), code_at, sequence);
    }
};

constexpr std::size_t complement_images(alignment_costs_t const& costs) noexcept { return costs.complements ? 2 : 1; }

inline void reverse_complement(alignment_costs_t const& costs, bytes_view_t sequence, bytes_span_t image) noexcept {
    std::size_t const symbols = sequence.size();
    for (std::size_t symbol = 0; symbol != symbols; ++symbol)
        image[symbol] = static_cast<byte_t>(
            costs.complemented[std::to_integer<std::size_t>(sequence[symbols - 1 - symbol])]);
}

/** What the window kits score with on a device, laid out as their kernels read it: the folded
 *  costs with the complements, then the window widths behind their count. */
struct window_parameters_t {
    alignment_costs_t costs;
    sz_size_t widths_count;
    sz_size_t widths[sz_ngrams_widths_max_k];

    /**
     *  @brief Lays @p window_widths out beside @p costs.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k No width, more of them than one table holds, or a width of zero.
     */
    static expected<window_parameters_t> of(std::span<sz_size_t const> window_widths,
                                            alignment_costs_t const& costs) noexcept {
        expected<window_parameters_t> laid {{}, status_code_t::config_mismatch_k};
        if (window_widths.empty() || window_widths.size() > sz_ngrams_widths_max_k) return laid;
        for (std::size_t index = 0; index != window_widths.size(); ++index) {
            if (!window_widths[index]) return laid;
            laid.value.widths[index] = window_widths[index];
        }
        laid.value.costs = costs;
        laid.value.widths_count = window_widths.size();
        laid.status = status_code_t::success_k;
        return laid;
    }
};

/** Bits of class codes a window of a folded alphabet covers at least, which sets its width. */
inline constexpr u32_t spectrum_window_bits_k = 12;

/** Symbols a window of @p classes spans: the fewest whose classes cover
 *  @ref spectrum_window_bits_k. */
constexpr u32_t spectrum_window_width(std::size_t classes) noexcept {
    u32_t width = 1;
    for (std::size_t covered = classes; covered < (std::size_t {1} << spectrum_window_bits_k); covered *= classes)
        ++width;
    return width;
}
static_assert(spectrum_window_width(24) == 3);

/** Windows of position a column's step takes to turn by its own draw: a step's deviation is one
 *  radian over this many windows, at the window count the tables name as their reference. */
inline constexpr u32_t spectrum_step_windows_k = 12;

/** The share of a sequence's own windows the position kernel's bandwidth spans, so two places
 *  that far apart keep exp(-1/2) of a window's match: the value measured on held-out proteins. */
inline constexpr f64_t spectrum_bandwidth_share_k = 0.086;

/** The window count at which a proportional step of a folded alphabet equals the tables' own. A
 *  window's phase advances by its place times the step times this count over the sequence's
 *  windows, so the bandwidth is those windows times @ref spectrum_step_windows_k over the count. */
inline constexpr u32_t spectrum_reference_windows_k = static_cast<u32_t>(
    spectrum_step_windows_k / spectrum_bandwidth_share_k + 0.5);
static_assert(spectrum_reference_windows_k == 140);

/** Bits a column's sum of a folded alphabet drops before it saturates to an @c int8. */
inline constexpr u32_t spectrum_shift_k = 7;

/** Symbols a window spans at most, which the narrowest alphabet of two classes takes. */
inline constexpr std::size_t spectrum_width_max_k = spectrum_window_width(2);

/** Entries of the cosine table, read by a phase's top ten bits. */
inline constexpr std::size_t spectrum_cosines_k = 1024;

/** One turn of a cosine scaled to 127 and sampled at the middle of each of its steps, baked so
 *  every platform sums the same integers. */
inline constexpr i8_t spectrum_cosine_turn_k[spectrum_cosines_k] = {
    127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  126,  126,  126,  126,  126,
    126,  126,  126,  126,  126,  126,  125,  125,  125,  125,  125,  125,  125,  124,  124,  124,  124,  124,  124,
    123,  123,  123,  123,  123,  123,  122,  122,  122,  122,  121,  121,  121,  121,  120,  120,  120,  120,  119,
    119,  119,  119,  118,  118,  118,  117,  117,  117,  117,  116,  116,  116,  115,  115,  115,  114,  114,  114,
    113,  113,  113,  112,  112,  111,  111,  111,  110,  110,  110,  109,  109,  108,  108,  108,  107,  107,  106,
    106,  105,  105,  105,  104,  104,  103,  103,  102,  102,  101,  101,  100,  100,  99,   99,   98,   98,   97,
    97,   96,   96,   95,   95,   94,   94,   93,   93,   92,   92,   91,   91,   90,   90,   89,   88,   88,   87,
    87,   86,   86,   85,   84,   84,   83,   83,   82,   81,   81,   80,   80,   79,   78,   78,   77,   77,   76,
    75,   75,   74,   73,   73,   72,   72,   71,   70,   70,   69,   68,   68,   67,   66,   66,   65,   64,   64,
    63,   62,   62,   61,   60,   60,   59,   58,   57,   57,   56,   55,   55,   54,   53,   53,   52,   51,   50,
    50,   49,   48,   48,   47,   46,   45,   45,   44,   43,   42,   42,   41,   40,   39,   39,   38,   37,   36,
    36,   35,   34,   33,   33,   32,   31,   30,   30,   29,   28,   27,   27,   26,   25,   24,   24,   23,   22,
    21,   21,   20,   19,   18,   17,   17,   16,   15,   14,   14,   13,   12,   11,   11,   10,   9,    8,    7,
    7,    6,    5,    4,    4,    3,    2,    1,    0,    0,    -1,   -2,   -3,   -4,   -4,   -5,   -6,   -7,   -7,
    -8,   -9,   -10,  -11,  -11,  -12,  -13,  -14,  -14,  -15,  -16,  -17,  -17,  -18,  -19,  -20,  -21,  -21,  -22,
    -23,  -24,  -24,  -25,  -26,  -27,  -27,  -28,  -29,  -30,  -30,  -31,  -32,  -33,  -33,  -34,  -35,  -36,  -36,
    -37,  -38,  -39,  -39,  -40,  -41,  -42,  -42,  -43,  -44,  -45,  -45,  -46,  -47,  -48,  -48,  -49,  -50,  -50,
    -51,  -52,  -53,  -53,  -54,  -55,  -55,  -56,  -57,  -57,  -58,  -59,  -60,  -60,  -61,  -62,  -62,  -63,  -64,
    -64,  -65,  -66,  -66,  -67,  -68,  -68,  -69,  -70,  -70,  -71,  -72,  -72,  -73,  -73,  -74,  -75,  -75,  -76,
    -77,  -77,  -78,  -78,  -79,  -80,  -80,  -81,  -81,  -82,  -83,  -83,  -84,  -84,  -85,  -86,  -86,  -87,  -87,
    -88,  -88,  -89,  -90,  -90,  -91,  -91,  -92,  -92,  -93,  -93,  -94,  -94,  -95,  -95,  -96,  -96,  -97,  -97,
    -98,  -98,  -99,  -99,  -100, -100, -101, -101, -102, -102, -103, -103, -104, -104, -105, -105, -105, -106, -106,
    -107, -107, -108, -108, -108, -109, -109, -110, -110, -110, -111, -111, -111, -112, -112, -113, -113, -113, -114,
    -114, -114, -115, -115, -115, -116, -116, -116, -117, -117, -117, -117, -118, -118, -118, -119, -119, -119, -119,
    -120, -120, -120, -120, -121, -121, -121, -121, -122, -122, -122, -122, -123, -123, -123, -123, -123, -123, -124,
    -124, -124, -124, -124, -124, -125, -125, -125, -125, -125, -125, -125, -126, -126, -126, -126, -126, -126, -126,
    -126, -126, -126, -126, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
    -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126,
    -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -123,
    -123, -123, -123, -123, -123, -122, -122, -122, -122, -121, -121, -121, -121, -120, -120, -120, -120, -119, -119,
    -119, -119, -118, -118, -118, -117, -117, -117, -117, -116, -116, -116, -115, -115, -115, -114, -114, -114, -113,
    -113, -113, -112, -112, -111, -111, -111, -110, -110, -110, -109, -109, -108, -108, -108, -107, -107, -106, -106,
    -105, -105, -105, -104, -104, -103, -103, -102, -102, -101, -101, -100, -100, -99,  -99,  -98,  -98,  -97,  -97,
    -96,  -96,  -95,  -95,  -94,  -94,  -93,  -93,  -92,  -92,  -91,  -91,  -90,  -90,  -89,  -88,  -88,  -87,  -87,
    -86,  -86,  -85,  -84,  -84,  -83,  -83,  -82,  -81,  -81,  -80,  -80,  -79,  -78,  -78,  -77,  -77,  -76,  -75,
    -75,  -74,  -73,  -73,  -72,  -72,  -71,  -70,  -70,  -69,  -68,  -68,  -67,  -66,  -66,  -65,  -64,  -64,  -63,
    -62,  -62,  -61,  -60,  -60,  -59,  -58,  -57,  -57,  -56,  -55,  -55,  -54,  -53,  -53,  -52,  -51,  -50,  -50,
    -49,  -48,  -48,  -47,  -46,  -45,  -45,  -44,  -43,  -42,  -42,  -41,  -40,  -39,  -39,  -38,  -37,  -36,  -36,
    -35,  -34,  -33,  -33,  -32,  -31,  -30,  -30,  -29,  -28,  -27,  -27,  -26,  -25,  -24,  -24,  -23,  -22,  -21,
    -21,  -20,  -19,  -18,  -17,  -17,  -16,  -15,  -14,  -14,  -13,  -12,  -11,  -11,  -10,  -9,   -8,   -7,   -7,
    -6,   -5,   -4,   -4,   -3,   -2,   -1,   0,    0,    1,    2,    3,    4,    4,    5,    6,    7,    7,    8,
    9,    10,   11,   11,   12,   13,   14,   14,   15,   16,   17,   17,   18,   19,   20,   21,   21,   22,   23,
    24,   24,   25,   26,   27,   27,   28,   29,   30,   30,   31,   32,   33,   33,   34,   35,   36,   36,   37,
    38,   39,   39,   40,   41,   42,   42,   43,   44,   45,   45,   46,   47,   48,   48,   49,   50,   50,   51,
    52,   53,   53,   54,   55,   55,   56,   57,   57,   58,   59,   60,   60,   61,   62,   62,   63,   64,   64,
    65,   66,   66,   67,   68,   68,   69,   70,   70,   71,   72,   72,   73,   73,   74,   75,   75,   76,   77,
    77,   78,   78,   79,   80,   80,   81,   81,   82,   83,   83,   84,   84,   85,   86,   86,   87,   87,   88,
    88,   89,   90,   90,   91,   91,   92,   92,   93,   93,   94,   94,   95,   95,   96,   96,   97,   97,   98,
    98,   99,   99,   100,  100,  101,  101,  102,  102,  103,  103,  104,  104,  105,  105,  105,  106,  106,  107,
    107,  108,  108,  108,  109,  109,  110,  110,  110,  111,  111,  111,  112,  112,  113,  113,  113,  114,  114,
    114,  115,  115,  115,  116,  116,  116,  117,  117,  117,  117,  118,  118,  118,  119,  119,  119,  119,  120,
    120,  120,  120,  121,  121,  121,  121,  122,  122,  122,  122,  123,  123,  123,  123,  123,  123,  124,  124,
    124,  124,  124,  124,  125,  125,  125,  125,  125,  125,  125,  126,  126,  126,  126,  126,  126,  126,  126,
    126,  126,  126,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,
};

/** The head of the tables a smooth sketch reads, in front of their steps and phases in one buffer,
 *  so a device index uploads them byte for byte. */
struct spectrum_head_t {

    /** @ref spectrum_cosine_turn_k, where a device reads it. */
    i8_t cosines[spectrum_cosines_k];

    /** Class codes in use. */
    u32_t classes;

    /** Codes a window spans, so the phases hold <tt>width * classes</tt> rows. */
    u32_t width;

    /** The window count at which a sequence's step equals @c steps. */
    u32_t reference;

    /** Bits the sums drop before saturating to @c int8. */
    u32_t shift;

    /** Columns of every row, one per byte of the sketch. */
    u32_t columns;
};

/**
 *  @brief The tables a smooth sketch reads, borrowed from their buffer, as host kits and device
 *      kernels read them alike.
 *
 *  Column @c c of a sketch sums, over every window of @c width class codes, the cosine of the
 *  phases its codes hold at their places of the window plus the window's place times the column's
 *  step. Every phase is in 1/65536 turns, so sums of them wrap for free.
 */
struct spectrum_view_t {
    spectrum_head_t const* head;

    /** Phase one window of position adds at every column at the @c reference window count. */
    u16_t const* steps; // device-boundary

    /** Phase of row <tt>place * classes + code</tt> at every column, the class @c code at that
     *  place of a window, rows of @c columns end to end. */
    u16_t const* phases; // device-boundary
};

/** Bytes from the start of the tables to their steps: the head, padded to a cache line. */
inline constexpr std::size_t spectrum_steps_offset_k = round_up_to_multiple<std::size_t>(sizeof(spectrum_head_t), 64);

/** Bytes from the start of the tables to their phases, behind @p columns steps. */
constexpr std::size_t spectrum_phases_offset(std::size_t columns) noexcept {
    return spectrum_steps_offset_k + round_up_to_multiple<std::size_t>(columns * sizeof(u16_t), 64);
}

/** Zero bytes behind the last row, so a register loading the last columns of a row stays inside. */
inline constexpr std::size_t spectrum_slack_k = 64;

/** Bytes of the tables of @p rows rows of @p columns columns. */
constexpr std::size_t spectrum_tables_bytes(std::size_t rows, std::size_t columns) noexcept {
    return spectrum_phases_offset(columns) + rows * columns * sizeof(u16_t) + spectrum_slack_k;
}

/** The tables laid out at @p bytes, as @ref spectrum_tables writes them. */
constexpr spectrum_view_t spectrum_view_of(byte_t const* bytes) noexcept {
    spectrum_head_t const* const head = static_cast<spectrum_head_t const*>(static_cast<void const*>(bytes));
    return {head, static_cast<u16_t const*>(static_cast<void const*>(bytes + spectrum_steps_offset_k)),
            static_cast<u16_t const*>(static_cast<void const*>(bytes + spectrum_phases_offset(head->columns)))};
}

/** The buffer @p tables borrow from, whole. */
inline bytes_view_t spectrum_bytes_of(spectrum_view_t tables) noexcept {
    spectrum_head_t const& head = *tables.head;
    return {reinterpret_cast<byte_t const*>(tables.head),
            spectrum_tables_bytes(std::size_t {head.width} * head.classes, head.columns)};
}

/** Columns one pass over a sequence's windows sums, a multiple of every backend's register: the
 *  sums, angles and steps of a pass stay on the stack, and a sketch of any width takes passes. */
inline constexpr std::size_t spectrum_tile_columns_k = 128;

/** One pass's sums while a sequence's windows are added, on its caller's stack, so any number of
 *  threads sketch under one set of tables. */
struct spectrum_tile_t {

    /** The cosines summed so far at every column of the pass. */
    i32_t sums[spectrum_tile_columns_k];

    /** The phase the next window's place adds at every column, in 1/2³² turns. */
    u32_t angles[spectrum_tile_columns_k];

    /** What one window adds to every angle. */
    u32_t deltas[spectrum_tile_columns_k];

    /** The first column of the sketch the pass sums. */
    std::size_t first;

    /** Columns of the sketch the pass sums, @ref spectrum_tile_columns_k at most. */
    std::size_t columns;
};

/** A symmetric @b [classes,classes] matrix of f64, row-major, as the Jacobi rotations turn it. */
using spectrum_square_t = f64_t[alignment_classes_k][alignment_classes_k];

/** Rotates rows and columns @p first and @p second of @p matrix by the Jacobi angle zeroing their
 *  shared entry, and the same columns of @p vectors with them. */
inline void spectrum_rotate(spectrum_square_t& matrix, spectrum_square_t& vectors, std::size_t classes,
                            std::size_t first, std::size_t second) noexcept {
    f64_t const theta = (matrix[second][second] - matrix[first][first]) / (2 * matrix[first][second]);
    f64_t const tangent = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
    f64_t const cosine = 1 / std::sqrt(tangent * tangent + 1), sine = tangent * cosine;
    for (std::size_t other = 0; other != classes; ++other) {
        f64_t const left = matrix[other][first], right = matrix[other][second];
        matrix[other][first] = cosine * left - sine * right, matrix[other][second] = sine * left + cosine * right;
    }
    for (std::size_t other = 0; other != classes; ++other) {
        f64_t const top = matrix[first][other], bottom = matrix[second][other];
        matrix[first][other] = cosine * top - sine * bottom, matrix[second][other] = sine * top + cosine * bottom;
    }
    for (std::size_t other = 0; other != classes; ++other) {
        f64_t const left = vectors[other][first], right = vectors[other][second];
        vectors[other][first] = cosine * left - sine * right, vectors[other][second] = sine * left + cosine * right;
    }
}

/** The eigenvalues of the symmetric @p matrix onto its diagonal and its eigenvectors into the
 *  columns of @p vectors, by cyclic Jacobi sweeps until no off-diagonal mass is left to rotate. */
inline void spectrum_eigens(spectrum_square_t& matrix, spectrum_square_t& vectors, std::size_t classes) noexcept {
    f64_t whole = 0;
    for (std::size_t row = 0; row != classes; ++row)
        for (std::size_t column = 0; column != classes; ++column) {
            vectors[row][column] = row == column;
            whole += matrix[row][column] * matrix[row][column];
        }
    for (;;) {
        f64_t off_diagonal = 0;
        for (std::size_t row = 0; row != classes; ++row)
            for (std::size_t column = row + 1; column != classes; ++column)
                off_diagonal += matrix[row][column] * matrix[row][column];
        if (off_diagonal <= whole * 1e-24) return;
        for (std::size_t pair = 0; pair != classes * classes; ++pair)
            if (pair / classes < pair % classes && matrix[pair / classes][pair % classes] != 0)
                spectrum_rotate(matrix, vectors, classes, pair / classes, pair % classes);
    }
}

/** The standard normal at @p counter of the stream @p seed, the Box-Muller cosine of two
 *  SplitMix64 draws. */
inline f64_t spectrum_normal(u64_t seed, u64_t counter) noexcept {
    f64_t const radius = std::sqrt(-2.0 * std::log(uniform_unit_f64(splitmix64_at(seed, 2 * counter))));
    return radius * std::cos(2.0 * std::numbers::pi * uniform_unit_f64(splitmix64_at(seed, 2 * counter + 1)));
}

/** @p radians in 1/65536 turns, wrapped. */
inline u16_t spectrum_turns(f64_t radians) noexcept {
    return static_cast<u16_t>(static_cast<u64_t>(std::llround(radians / (2.0 * std::numbers::pi) * 65536.0)) & 0xFFFFu);
}

/**
 *  @brief The tables a smooth sketch reads, owned in one buffer through @p allocator_type_: derived
 *      once from the costs, read through @ref view on the host, and uploaded as @ref bytes.
 *
 *  Every draw of a column depends on the column alone, so the tables of fewer columns are a prefix
 *  of those of more, and so is every sketch they write.
 */
template <typename allocator_type_ = std::allocator<byte_t>>
class spectrum_tables {
  public:
    using allocator_t = allocator_type_;

    explicit spectrum_tables(allocator_t const& allocator = {}) noexcept : bytes_(allocator) {}

    /**
     *  @brief Derives the tables of @p columns columns from the folded @p costs of an alignment
     *      over packed class codes.
     *
     *  Classical scaling embeds the classes, negative eigenvalues clipped, and each class at each
     *  place of a window takes Gaussian features of its embedding. The kernel's bandwidth is the
     *  median cost between distinct classes, so a typical substitution keeps 1/e of a window's
     *  match, and a window spans @ref spectrum_window_width symbols, its phase shifted by its
     *  place in the sequence.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k Costs that are not folded or whose every distinct pair costs nothing,
     *    or @p columns that are zero, not a multiple of four, or past @c u32_t.
     *  - @c out_of_memory_k The buffer could not grow.
     */
    status_t derive(alignment_costs_t const& costs, std::size_t columns) noexcept;

    /** The tables, once derived. */
    spectrum_view_t view() const noexcept { return spectrum_view_of(bytes_.data()); }

    /** The buffer the tables live in, as a device uploads it. */
    bytes_view_t bytes() const noexcept { return bytes_; }

  private:
    growing_buffer<byte_t, allocator_t> bytes_;
};

template <typename allocator_type_>
status_t spectrum_tables<allocator_type_>::derive(alignment_costs_t const& costs, std::size_t columns) noexcept {
    constexpr std::size_t side_k = alignment_classes_k;
    std::size_t const classes = costs.classes;
    if (!classes || !columns || columns % 4 || columns > (std::numeric_limits<u32_t>::max)())
        return status_code_t::config_mismatch_k;

    // The median cost between distinct classes is the kernel's bandwidth, or the least positive
    // one where most pairs cost nothing.
    std::size_t histogram[alignment_cost_ceiling_k + 1] = {};
    std::size_t pairs = 0;
    for (std::size_t row = 0; row != classes; ++row)
        for (std::size_t column = row + 1; column != classes; ++column, ++pairs)
            ++histogram[(costs.substitutions[row][column] + costs.substitutions[column][row]) / 2];
    std::size_t median = 0;
    for (std::size_t below = 0; median <= alignment_cost_ceiling_k; ++median)
        if ((below += histogram[median]) * 2 > pairs) break;
    while (median <= alignment_cost_ceiling_k && (!median || !histogram[median])) ++median;
    if (median > alignment_cost_ceiling_k) return status_code_t::config_mismatch_k;

    // Classical scaling: G = −J C J, with cost(a, b) as half the squared distance of the embedding.
    spectrum_square_t gram, vectors;
    f64_t means[side_k] = {}, mean = 0;
    for (std::size_t row = 0; row != classes; ++row)
        for (std::size_t column = 0; column != classes; ++column) {
            f64_t const cost = (costs.substitutions[row][column] + costs.substitutions[column][row]) / 2.0;
            gram[row][column] = cost, means[row] += cost / classes, mean += cost / (classes * classes);
        }
    for (std::size_t row = 0; row != classes; ++row)
        for (std::size_t column = 0; column != classes; ++column)
            gram[row][column] = -(gram[row][column] - means[row] - means[column] + mean);
    spectrum_eigens(gram, vectors, classes);
    spectrum_square_t embedding;
    for (std::size_t row = 0; row != classes; ++row)
        for (std::size_t axis = 0; axis != classes; ++axis)
            embedding[row][axis] = vectors[row][axis] * std::sqrt(gram[axis][axis] > 0 ? gram[axis][axis] : 0.0);

    u32_t const width = spectrum_window_width(classes);
    if (status_t const grown = bytes_.resize(spectrum_tables_bytes(width * classes, columns)); failed(grown))
        return grown;
    byte_t* const bytes = bytes_.data();
    spectrum_head_t& head = *reinterpret_cast<spectrum_head_t*>(bytes);
    u16_t* const steps = reinterpret_cast<u16_t*>(bytes + spectrum_steps_offset_k);
    u16_t* const phases = reinterpret_cast<u16_t*>(bytes + spectrum_phases_offset(columns));

    // Gaussian features of variance 1/s per axis, and a uniform offset folded into the first place,
    // every draw counted from its column first.
    f64_t const deviation = 1 / std::sqrt(static_cast<f64_t>(median));
    for (std::size_t column = 0; column != columns; ++column) {
        f64_t const offset = 2.0 * std::numbers::pi * uniform_unit_f64(splitmix64_at(1, column));
        for (std::size_t place = 0; place != width; ++place) {
            f64_t weights[side_k];
            for (std::size_t axis = 0; axis != classes; ++axis)
                weights[axis] = deviation * spectrum_normal(0, (column * spectrum_width_max_k + place) * side_k + axis);
            for (std::size_t code = 0; code != classes; ++code)
                phases[(place * classes + code) * columns + column] = spectrum_turns(
                    std::inner_product(embedding[code], embedding[code] + classes, weights, place ? 0.0 : offset));
        }
        // A magnitude, as a cosine sum over symmetric draws reads a step and its negation alike.
        steps[column] = spectrum_turns(std::fabs(spectrum_normal(2, column)) / spectrum_step_windows_k);
    }
    std::memcpy(head.cosines, spectrum_cosine_turn_k, sizeof(head.cosines));
    head.classes = static_cast<u32_t>(classes), head.width = width, head.columns = static_cast<u32_t>(columns);
    head.reference = spectrum_reference_windows_k, head.shift = spectrum_shift_k;
    return status_code_t::success_k;
}

/** Opens @p tile for the @p columns columns of a sketch from @p first, of a sequence of @p windows
 *  windows: zeroed sums and angles, and every column's step scaled by the reference window count
 *  over @p windows. */
inline void spectrum_open(spectrum_view_t const& tables, std::size_t windows, std::size_t first, std::size_t columns,
                          spectrum_tile_t& tile) noexcept {
    // A step under half a turn times any reference stays under 2⁶³, so the product never wraps.
    u64_t const span = u64_t {tables.head->reference} << 16;
    tile.first = first, tile.columns = columns;
    for (std::size_t column = 0; column != spectrum_tile_columns_k; ++column) {
        tile.sums[column] = 0, tile.angles[column] = 0;
        tile.deltas[column] = windows && column < columns
                                  ? static_cast<u32_t>(tables.steps[first + column] * span / windows)
                                  : 0;
    }
}

/** The phase rows the window of @c width @p classes reads, from the columns of @p tile on. */
inline void spectrum_rows(spectrum_view_t const& tables, u8_t const* classes, spectrum_tile_t const& tile,
                          u16_t const** rows) noexcept {
    std::size_t const width = tables.head->width, columns = tables.head->columns;
    for (std::size_t place = 0; place != width; ++place)
        rows[place] = tables.phases + (place * tables.head->classes + classes[place]) * columns + tile.first;
}

/** Adds the window of @c width @p classes to every column of @p tile and advances its angles, one
 *  column at a time. */
inline void spectrum_step_serial(spectrum_view_t const& tables, u8_t const* classes, spectrum_tile_t& tile) noexcept {
    u16_t const* rows[spectrum_width_max_k];
    spectrum_rows(tables, classes, tile, rows);
    std::size_t const width = tables.head->width;
    for (std::size_t column = 0; column != tile.columns; ++column) {
        u32_t phase = tile.angles[column] >> 16;
        for (std::size_t place = 0; place != width; ++place) phase += rows[place][column];
        // The sums wrap as the SIMD adds do, which takes a sequence past sixteen million windows.
        tile.sums[column] = static_cast<i32_t>(static_cast<u32_t>(tile.sums[column]) +
                                               static_cast<u32_t>(tables.head->cosines[(phase & 0xFFFFu) >> 6]));
        tile.angles[column] += tile.deltas[column];
    }
}

/** Writes the columns of @p tile into @p sketch from its sums, rounded, shifted and saturated. */
inline void spectrum_close_serial(spectrum_view_t const& tables, spectrum_tile_t const& tile, i8_t* sketch) noexcept {
    u32_t const shift = tables.head->shift, rounding = (1u << shift) >> 1;
    for (std::size_t column = 0; column != tile.columns; ++column) {
        i32_t const scaled = static_cast<i32_t>(static_cast<u32_t>(tile.sums[column]) + rounding) >> shift;
        sketch[column] = static_cast<i8_t>(scaled < -128 ? -128 : scaled > 127 ? 127 : scaled);
    }
}

/** Writes the smooth sketch of the packed class codes of @p sequence into @p sketch, one byte a
 *  column, through the window steps @p kit_ binds: a pass over every window, decoded in place, per
 *  @ref spectrum_tile_columns_k columns, each closing into the stack before its columns land. */
template <typename kit_>
void sketch_packed_codes(spectrum_view_t const& tables, bytes_view_t sequence, bytes_span_t sketch) noexcept {
    std::size_t const width = tables.head->width;
    std::size_t const codes = packed_codes_count(sequence.data(), sequence.size());
    std::size_t const windows = codes >= width ? codes - width + 1 : 0;
    for (std::size_t first = 0; first < sketch.size(); first += spectrum_tile_columns_k) {
        spectrum_tile_t tile;
        spectrum_open(tables, windows, first, min_of(spectrum_tile_columns_k, sketch.size() - first), tile);
        for (std::size_t window = 0; window != windows; ++window) {
            u8_t classes[spectrum_width_max_k];
            for (std::size_t place = 0; place != width; ++place)
                classes[place] = static_cast<u8_t>(packed_code_at(sequence.data(), window + place));
            kit_::sketch_step(tables, classes, tile);
        }
        i8_t closed[spectrum_tile_columns_k];
        kit_::sketch_close(tables, tile, closed);
        std::memcpy(sketch.data() + first, closed, tile.columns);
    }
}

/** One candidate's cost from @p query down its columns, @p column holding a cell per query code and
 *  one more: exact up to @p radius, and past it the first column minimum above it. */
inline sz_size_t alignment_serial(alignment_costs_t const& costs, std::span<u8_t const> query, bytes_view_t candidate,
                                  sz_size_t radius, u32_t* column) noexcept {
    std::size_t const rows = query.size();
    std::size_t const columns = packed_codes_count(candidate.data(), candidate.size());
    u32_t const gap = costs.gap;
    for (std::size_t row = 0; row <= rows; ++row) column[row] = static_cast<u32_t>(row * gap);
    for (std::size_t index = 0; index != columns; ++index) {
        u32_t const code = packed_code_at(candidate.data(), index);
        u32_t diagonal = column[0], above = column[0] + gap, minimum = above;
        column[0] = above;
        for (std::size_t row = 1; row <= rows; ++row) {
            u32_t const left = column[row];
            u32_t const cell = min_of(diagonal + costs.substitutions[query[row - 1]][code], min_of(above, left) + gap);
            diagonal = left, above = cell, column[row] = cell;
            minimum = min_of(minimum, cell);
        }
        // Every alignment crosses each column, so none ends below the column's minimum.
        if (minimum > radius) return minimum;
    }
    return column[rows];
}

/** Scalar: one candidate at a time, a column of @c u32 cells per query code. */
struct packed_serial_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;

    /** What a smooth sketch reads, derived once from the costs. */
    using sketch_tables_t = spectrum_view_t;
    using sketch_scalar_t = i8_t;

    /** The window steps of this backend, driven by @c sketch_write over decoded codes. */
    static constexpr auto sketch_step = spectrum_step_serial;
    static constexpr auto sketch_close = spectrum_close_serial;

    /** A smooth sketch keeps its tiles on the stack, and takes no scratch. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }

    /** Writes the smooth sketch of the stored @p sequence, packed class codes, into @p sketch. */
    static void sketch_write(sketch_tables_t const& tables, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t = {}) noexcept {
        sketch_packed_codes<packed_serial_alignment_kit_t>(tables, sequence, sketch);
    }

    /** Squared Euclidean distance between two sketches. */
    static constexpr auto sketch_distance = nk_sqeuclidean_i8_serial;

    /** Scratch bytes a round of a @p query_codes long query and candidates of up to
     *  @p longest_codes takes. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_codes, std::size_t) noexcept {
        return (query_codes + 1) * sizeof(u32_t);
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, in @p order. */
    static void canonical_distances(alignment_costs_t const& costs, std::span<u8_t const> query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        u32_t* const column = elements_of<u32_t>(scratch).data();
        for (u32_t const slot : order)
            distances[slot] = alignment_serial(costs, query, candidates[slot], radius, column);
    }
};

#if STRINGZILLA_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw")
#endif

/** Adds the window of @c width @p classes to every column of @p tile and advances its
 *  angles, thirty-two @c u16 phases a ZMM, their cosines gathered. */
inline void spectrum_step_skylake(spectrum_view_t const& tables, u8_t const* classes, spectrum_tile_t& tile) noexcept {
    u16_t const* rows[spectrum_width_max_k];
    spectrum_rows(tables, classes, tile, rows);
    std::size_t const width = tables.head->width;
    for (std::size_t column = 0; column < tile.columns; column += sizeof(__m512i) / sizeof(u16_t)) {
        __m512i const angles_low_u32x16 = _mm512_loadu_si512(tile.angles + column);
        __m512i const angles_high_u32x16 = _mm512_loadu_si512(tile.angles + column + 16);
        __m512i phases_u16x32 = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm512_cvtepi32_epi16(_mm512_srli_epi32(angles_low_u32x16, 16))),
            _mm512_cvtepi32_epi16(_mm512_srli_epi32(angles_high_u32x16, 16)), 1);
        for (std::size_t place = 0; place != width; ++place)
            phases_u16x32 = _mm512_add_epi16(phases_u16x32, _mm512_loadu_si512(rows[place] + column));
        __m512i const entries_u16x32 = _mm512_srli_epi16(phases_u16x32, 6);
        __m512i const entries_low_i32x16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(entries_u16x32));
        __m512i const entries_high_i32x16 = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(entries_u16x32, 1));
        // A gathered word holds its cosine in the low byte, under the three bytes after it.
        __m512i const cosines_low_i32x16 = _mm512_srai_epi32(
            _mm512_slli_epi32(_mm512_i32gather_epi32(entries_low_i32x16, tables.head->cosines, 1), 24), 24);
        __m512i const cosines_high_i32x16 = _mm512_srai_epi32(
            _mm512_slli_epi32(_mm512_i32gather_epi32(entries_high_i32x16, tables.head->cosines, 1), 24), 24);
        _mm512_storeu_si512(tile.sums + column,
                            _mm512_add_epi32(_mm512_loadu_si512(tile.sums + column), cosines_low_i32x16));
        _mm512_storeu_si512(tile.sums + column + 16,
                            _mm512_add_epi32(_mm512_loadu_si512(tile.sums + column + 16), cosines_high_i32x16));
        _mm512_storeu_si512(tile.angles + column,
                            _mm512_add_epi32(angles_low_u32x16, _mm512_loadu_si512(tile.deltas + column)));
        _mm512_storeu_si512(tile.angles + column + 16,
                            _mm512_add_epi32(angles_high_u32x16, _mm512_loadu_si512(tile.deltas + column + 16)));
    }
}

/** Writes the columns of @p tile into @p sketch from its sums, sixteen a ZMM,
 *  rounded, shifted and saturated. */
inline void spectrum_close_skylake(spectrum_view_t const& tables, spectrum_tile_t const& tile, i8_t* sketch) noexcept {
    __m512i const rounding_i32x16 = _mm512_set1_epi32(static_cast<int>((1u << tables.head->shift) >> 1));
    __m128i const shift_u64x2 = _mm_cvtsi32_si128(static_cast<int>(tables.head->shift));
    for (std::size_t column = 0; column < tile.columns; column += sizeof(__m512i) / sizeof(i32_t)) {
        __m512i const sums_i32x16 = _mm512_loadu_si512(tile.sums + column);
        __m512i const scaled_i32x16 = _mm512_sra_epi32(_mm512_add_epi32(sums_i32x16, rounding_i32x16), shift_u64x2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(sketch + column), _mm512_cvtsepi32_epi8(scaled_i32x16));
    }
}

/**
 *  @brief Intel Skylake-X AVX-512: 32 candidates per ZMM, one @c u16 cell each, a column at a time.
 *
 *  A group's codes are transposed once, so every column holds one code per lane, and each column
 *  looks its codes up once per class the query holds with one @c VPERMW, leaving one load per cell.
 *  Cells saturate at 65535, which bounds every true cost above them, so a saturated cost is
 *  rescored by the scalar kernel whenever the radius reaches it. Candidates past 65535 codes take
 *  the scalar kernel outright.
 */
struct packed_skylake_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;

    /** What a smooth sketch reads, derived once from the costs. */
    using sketch_tables_t = spectrum_view_t;
    using sketch_scalar_t = i8_t;

    /** The window steps of this backend, driven by @c sketch_write over decoded codes. */
    static constexpr auto sketch_step = spectrum_step_skylake;
    static constexpr auto sketch_close = spectrum_close_skylake;

    /** A smooth sketch keeps its tiles on the stack, and takes no scratch. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }

    /** Writes the smooth sketch of the stored @p sequence, packed class codes, into @p sketch. */
    static void sketch_write(sketch_tables_t const& tables, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t = {}) noexcept {
        sketch_packed_codes<packed_skylake_alignment_kit_t>(tables, sequence, sketch);
    }

    /** Squared Euclidean distance between two sketches. */
    static constexpr auto sketch_distance = nk_sqeuclidean_i8_haswell;

    /** Candidates one group sweeps, the @c u16 lanes of a ZMM. */
    static constexpr std::size_t canonical_lanes_k = sizeof(__m512i) / sizeof(u16_t);

    /** Scratch bytes a round of a @p query_codes long query and candidates of up to
     *  @p longest_codes takes: a ZMM per query code and one more, a ZMM of codes per candidate
     *  code, and the slack aligning them. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_codes, std::size_t longest_codes) noexcept {
        return (query_codes + 1 + longest_codes + 1) * sizeof(__m512i);
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, @p order taken in
     *  groups of @ref canonical_lanes_k. */
    static void canonical_distances(alignment_costs_t const& costs, std::span<u8_t const> query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        constexpr u32_t saturated_k = (std::numeric_limits<u16_t>::max)();
        constexpr __mmask32 every_lane_k = ~__mmask32 {0};
        std::size_t const rows = query.size();
        u32_t const gap = costs.gap;
        void* aligned = scratch.data();
        std::size_t space = scratch.size();
        __m512i* const column = static_cast<__m512i*>(std::align(sizeof(__m512i), sizeof(__m512i), aligned, space));
        u16_t* const codes = reinterpret_cast<u16_t*>(column + rows + 1);

        // A row of the table per class the query holds, widened to the lanes.
        u32_t query_classes = 0;
        for (u8_t const code : query) query_classes = max_of<u32_t>(query_classes, code + 1u);
        __m512i views_u16x32[alignment_classes_k], profile_u16x32[alignment_classes_k];
        for (u32_t code = 0; code != query_classes; ++code)
            views_u16x32[code] = _mm512_cvtepu8_epi16(
                _mm256_loadu_si256(reinterpret_cast<__m256i const*>(costs.substitutions[code])));
        __m512i const gap_u16x32 = _mm512_set1_epi16(static_cast<short>(gap));
        __m512i const radius_u16x32 = _mm512_set1_epi16(static_cast<short>(min_of<sz_size_t>(radius, saturated_k)));

        for (std::size_t first = 0; first < order.size(); first += canonical_lanes_k) {
            std::size_t const lanes = min_of(canonical_lanes_k, order.size() - first);
            u16_t lengths[canonical_lanes_k] = {};
            __mmask32 scalar = 0;
            std::size_t longest = 0;
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                bytes_view_t const candidate = candidates[order[first + lane]];
                std::size_t const length = packed_codes_count(candidate.data(), candidate.size());
                if (length > saturated_k) scalar |= __mmask32 {1} << lane;
                else lengths[lane] = static_cast<u16_t>(length), longest = max_of(longest, length);
            }
            std::memset(codes, 0, longest * sizeof(__m512i));
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                bytes_view_t const candidate = candidates[order[first + lane]];
                for (std::size_t index = 0; index != lengths[lane]; ++index)
                    codes[index * canonical_lanes_k + lane] = static_cast<u16_t>(
                        packed_code_at(candidate.data(), index));
            }

            for (std::size_t row = 0; row <= rows; ++row)
                column[row] = _mm512_set1_epi16(static_cast<short>(min_of<std::size_t>(row * gap, saturated_k)));
            __m512i const lengths_u16x32 = _mm512_loadu_si512(lengths);
            __mmask32 const empty_mask = _mm512_cmpeq_epi16_mask(lengths_u16x32, _mm512_setzero_si512());
            __m512i results_u16x32 = column[rows];
            // Padding lanes and the scalar ones count as finished from the start.
            __mmask32 finished_mask = empty_mask | scalar | (lanes == canonical_lanes_k ? 0 : every_lane_k << lanes);
            for (std::size_t index = 0; index != longest && finished_mask != every_lane_k; ++index) {
                __m512i const codes_u16x32 = _mm512_loadu_si512(codes + index * canonical_lanes_k);
                for (u32_t code = 0; code != query_classes; ++code)
                    profile_u16x32[code] = _mm512_permutexvar_epi16(codes_u16x32, views_u16x32[code]);
                __m512i diagonal_u16x32 = column[0];
                __m512i above_u16x32 = _mm512_adds_epu16(diagonal_u16x32, gap_u16x32);
                __m512i minimum_u16x32 = above_u16x32;
                column[0] = above_u16x32;
                for (std::size_t row = 1; row <= rows; ++row) {
                    __m512i const left_u16x32 = column[row];
                    __m512i const aligned_u16x32 = _mm512_adds_epu16(diagonal_u16x32, profile_u16x32[query[row - 1]]);
                    __m512i const opened_u16x32 = _mm512_min_epu16(aligned_u16x32,
                                                                   _mm512_adds_epu16(left_u16x32, gap_u16x32));
                    above_u16x32 = _mm512_min_epu16(opened_u16x32, _mm512_adds_epu16(above_u16x32, gap_u16x32));
                    diagonal_u16x32 = left_u16x32;
                    column[row] = above_u16x32;
                    minimum_u16x32 = _mm512_min_epu16(minimum_u16x32, above_u16x32);
                }
                __mmask32 const ended_mask = _mm512_cmpeq_epi16_mask(lengths_u16x32,
                                                                     _mm512_set1_epi16(static_cast<short>(index + 1))) &
                                             ~finished_mask;
                results_u16x32 = _mm512_mask_mov_epi16(results_u16x32, ended_mask, above_u16x32);
                finished_mask |= ended_mask;
                // Every alignment crosses each column, so none ends below the column's minimum.
                __mmask32 const past_mask = _mm512_cmpgt_epu16_mask(minimum_u16x32, radius_u16x32) & ~finished_mask;
                results_u16x32 = _mm512_mask_mov_epi16(results_u16x32, past_mask, minimum_u16x32);
                finished_mask |= past_mask;
            }

            u16_t results[canonical_lanes_k];
            _mm512_storeu_si512(results, results_u16x32);
            u32_t* const scalar_column = static_cast<u32_t*>(aligned);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                u32_t const slot = order[first + lane];
                bool const rescored = (scalar >> lane & 1u) || (results[lane] == saturated_k && radius >= saturated_k);
                distances[slot] = rescored ? alignment_serial(costs, query, candidates[slot], radius, scalar_column)
                                           : results[lane];
            }
        }
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build leaves out, kept so @c alignment_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct packed_skylake_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_ICELAKE

/** Intel Ice Lake: Skylake's sweep and window steps under its own sketch distance. */
struct packed_icelake_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;

    /** What a smooth sketch reads, derived once from the costs. */
    using sketch_tables_t = spectrum_view_t;
    using sketch_scalar_t = i8_t;

    /** A smooth sketch keeps its tiles on the stack, and takes no scratch. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }

    /** Writes the smooth sketch of the stored @p sequence, packed class codes, into @p sketch. */
    static void sketch_write(sketch_tables_t const& tables, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t = {}) noexcept {
        packed_skylake_alignment_kit_t::sketch_write(tables, sequence, sketch);
    }

    /** Squared Euclidean distance between two sketches. */
    static constexpr auto sketch_distance = nk_sqeuclidean_i8_icelake;

    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_codes, std::size_t longest_codes) noexcept {
        return packed_skylake_alignment_kit_t::canonical_scratch_bytes(query_codes, longest_codes);
    }
    static constexpr auto canonical_distances = packed_skylake_alignment_kit_t::canonical_distances;
};
#else

/** The kit this build leaves out, kept so @c alignment_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct packed_icelake_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_ICELAKE

#if STRINGZILLA_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

/** Adds the window of @c width @p classes to every column of @p tile and advances its
 *  angles, sixteen @c u16 phases a YMM, their cosines gathered. */
inline void spectrum_step_haswell(spectrum_view_t const& tables, u8_t const* classes, spectrum_tile_t& tile) noexcept {
    u16_t const* rows[spectrum_width_max_k];
    spectrum_rows(tables, classes, tile, rows);
    std::size_t const width = tables.head->width;
    int const* const cosines = reinterpret_cast<int const*>(tables.head->cosines);
    for (std::size_t column = 0; column < tile.columns; column += sizeof(__m256i) / sizeof(u16_t)) {
        __m256i* const sums = reinterpret_cast<__m256i*>(tile.sums + column);
        __m256i* const angles = reinterpret_cast<__m256i*>(tile.angles + column);
        __m256i const* const deltas = reinterpret_cast<__m256i const*>(tile.deltas + column);
        __m256i const angles_low_u32x8 = _mm256_loadu_si256(angles);
        __m256i const angles_high_u32x8 = _mm256_loadu_si256(angles + 1);
        // The pack interleaves the halves of its two inputs, which the permute puts back in order.
        __m256i phases_u16x16 = _mm256_permute4x64_epi64(
            _mm256_packus_epi32(_mm256_srli_epi32(angles_low_u32x8, 16), _mm256_srli_epi32(angles_high_u32x8, 16)),
            0xD8);
        for (std::size_t place = 0; place != width; ++place)
            phases_u16x16 = _mm256_add_epi16(
                phases_u16x16, _mm256_loadu_si256(reinterpret_cast<__m256i const*>(rows[place] + column)));
        __m256i const entries_u16x16 = _mm256_srli_epi16(phases_u16x16, 6);
        __m256i const entries_low_i32x8 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(entries_u16x16));
        __m256i const entries_high_i32x8 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(entries_u16x16, 1));
        // A gathered word holds its cosine in the low byte, under the three bytes after it.
        __m256i const cosines_low_i32x8 = _mm256_srai_epi32(
            _mm256_slli_epi32(_mm256_i32gather_epi32(cosines, entries_low_i32x8, 1), 24), 24);
        __m256i const cosines_high_i32x8 = _mm256_srai_epi32(
            _mm256_slli_epi32(_mm256_i32gather_epi32(cosines, entries_high_i32x8, 1), 24), 24);
        _mm256_storeu_si256(sums, _mm256_add_epi32(_mm256_loadu_si256(sums), cosines_low_i32x8));
        _mm256_storeu_si256(sums + 1, _mm256_add_epi32(_mm256_loadu_si256(sums + 1), cosines_high_i32x8));
        _mm256_storeu_si256(angles, _mm256_add_epi32(angles_low_u32x8, _mm256_loadu_si256(deltas)));
        _mm256_storeu_si256(angles + 1, _mm256_add_epi32(angles_high_u32x8, _mm256_loadu_si256(deltas + 1)));
    }
}

/** Writes the columns of @p tile into @p sketch from its sums, sixteen a step,
 *  rounded, shifted and saturated. */
inline void spectrum_close_haswell(spectrum_view_t const& tables, spectrum_tile_t const& tile, i8_t* sketch) noexcept {
    __m256i const rounding_i32x8 = _mm256_set1_epi32(static_cast<int>((1u << tables.head->shift) >> 1));
    __m128i const shift_u64x2 = _mm_cvtsi32_si128(static_cast<int>(tables.head->shift));
    for (std::size_t column = 0; column < tile.columns; column += sizeof(__m256i) / sizeof(u16_t)) {
        __m256i const* const sums = reinterpret_cast<__m256i const*>(tile.sums + column);
        __m256i const scaled_low_i32x8 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_loadu_si256(sums), rounding_i32x8),
                                                          shift_u64x2);
        __m256i const scaled_high_i32x8 = _mm256_sra_epi32(
            _mm256_add_epi32(_mm256_loadu_si256(sums + 1), rounding_i32x8), shift_u64x2);
        // Two saturating packs narrow to bytes; the permute undoes the first one's interleave.
        __m256i const scaled_i16x16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(scaled_low_i32x8, scaled_high_i32x8),
                                                               0xD8);
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(sketch + column),
            _mm_packs_epi16(_mm256_castsi256_si128(scaled_i16x16), _mm256_extracti128_si256(scaled_i16x16, 1)));
    }
}

/**
 *  @brief Intel Haswell AVX2: 16 candidates per YMM, one @c u16 cell each, a column at a time.
 *
 *  The Skylake sweep with byte codes: each column looks its codes up once per class the query holds
 *  with two @c VPSHUFB over the low and high halves of that class's row, blended on the code's
 *  fifth bit and widened to the lanes. AVX2 has no unsigned 16-bit compare, so a column is past the
 *  radius when clamping its minimum to the radius changes it.
 */
struct packed_haswell_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;

    /** What a smooth sketch reads, derived once from the costs. */
    using sketch_tables_t = spectrum_view_t;
    using sketch_scalar_t = i8_t;

    /** The window steps of this backend, driven by @c sketch_write over decoded codes. */
    static constexpr auto sketch_step = spectrum_step_haswell;
    static constexpr auto sketch_close = spectrum_close_haswell;

    /** A smooth sketch keeps its tiles on the stack, and takes no scratch. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }

    /** Writes the smooth sketch of the stored @p sequence, packed class codes, into @p sketch. */
    static void sketch_write(sketch_tables_t const& tables, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t = {}) noexcept {
        sketch_packed_codes<packed_haswell_alignment_kit_t>(tables, sequence, sketch);
    }

    /** Squared Euclidean distance between two sketches. */
    static constexpr auto sketch_distance = nk_sqeuclidean_i8_haswell;

    /** Candidates one group sweeps, the @c u16 lanes of a YMM. */
    static constexpr std::size_t canonical_lanes_k = sizeof(__m256i) / sizeof(u16_t);

    /** Scratch bytes a round of a @p query_codes long query and candidates of up to
     *  @p longest_codes takes: a YMM per query code and one more, a byte per lane per candidate
     *  code, and the slack aligning them. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_codes, std::size_t longest_codes) noexcept {
        return (query_codes + 2) * sizeof(__m256i) + longest_codes * canonical_lanes_k;
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, @p order taken in
     *  groups of @ref canonical_lanes_k. */
    static void canonical_distances(alignment_costs_t const& costs, std::span<u8_t const> query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        constexpr u32_t saturated_k = (std::numeric_limits<u16_t>::max)();
        constexpr u32_t every_lane_k = ~u32_t {0};
        std::size_t const rows = query.size();
        u32_t const gap = costs.gap;
        void* aligned = scratch.data();
        std::size_t space = scratch.size();
        __m256i* const column = static_cast<__m256i*>(std::align(sizeof(__m256i), sizeof(__m256i), aligned, space));
        u8_t* const codes = reinterpret_cast<u8_t*>(column + rows + 1);

        // A row of the table per class the query holds, as its low and high sixteen costs.
        u32_t query_classes = 0;
        for (u8_t const code : query) query_classes = max_of<u32_t>(query_classes, code + 1u);
        __m128i low_views_u8x16[alignment_classes_k], high_views_u8x16[alignment_classes_k];
        __m256i profile_u16x16[alignment_classes_k];
        for (u32_t code = 0; code != query_classes; ++code) {
            low_views_u8x16[code] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(costs.substitutions[code]));
            high_views_u8x16[code] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(costs.substitutions[code] + 16));
        }
        __m128i const last_low_code_u8x16 = _mm_set1_epi8(15);
        __m256i const gap_u16x16 = _mm256_set1_epi16(static_cast<short>(gap));
        __m256i const radius_u16x16 = _mm256_set1_epi16(static_cast<short>(min_of<sz_size_t>(radius, saturated_k)));

        for (std::size_t first = 0; first < order.size(); first += canonical_lanes_k) {
            std::size_t const lanes = min_of(canonical_lanes_k, order.size() - first);
            u16_t lengths[canonical_lanes_k] = {};
            u16_t finished[canonical_lanes_k] = {};
            u32_t scalar = 0;
            std::size_t longest = 0;
            for (std::size_t lane = 0; lane != canonical_lanes_k; ++lane) {
                if (lane >= lanes) {
                    finished[lane] = static_cast<u16_t>(saturated_k);
                    continue;
                }
                bytes_view_t const candidate = candidates[order[first + lane]];
                std::size_t const length = packed_codes_count(candidate.data(), candidate.size());
                if (length > saturated_k) scalar |= u32_t {1} << lane;
                else lengths[lane] = static_cast<u16_t>(length), longest = max_of(longest, length);
                // Padding lanes, empty ones and scalar ones count as finished from the start.
                if (!lengths[lane]) finished[lane] = static_cast<u16_t>(saturated_k);
            }
            std::memset(codes, 0, longest * canonical_lanes_k);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                bytes_view_t const candidate = candidates[order[first + lane]];
                for (std::size_t index = 0; index != lengths[lane]; ++index)
                    codes[index * canonical_lanes_k + lane] = static_cast<u8_t>(
                        packed_code_at(candidate.data(), index));
            }

            for (std::size_t row = 0; row <= rows; ++row)
                column[row] = _mm256_set1_epi16(static_cast<short>(min_of<std::size_t>(row * gap, saturated_k)));
            __m256i const lengths_u16x16 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(lengths));
            __m256i finished_u16x16 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(finished));
            __m256i results_u16x16 = column[rows];
            for (std::size_t index = 0;
                 index != longest && static_cast<u32_t>(_mm256_movemask_epi8(finished_u16x16)) != every_lane_k;
                 ++index) {
                __m128i const codes_u8x16 = _mm_loadu_si128(
                    reinterpret_cast<__m128i const*>(codes + index * canonical_lanes_k));
                __m128i const high_codes_u8x16 = _mm_cmpgt_epi8(codes_u8x16, last_low_code_u8x16);
                for (u32_t code = 0; code != query_classes; ++code)
                    profile_u16x16[code] = _mm256_cvtepu8_epi16(
                        _mm_blendv_epi8(_mm_shuffle_epi8(low_views_u8x16[code], codes_u8x16),
                                        _mm_shuffle_epi8(high_views_u8x16[code], codes_u8x16), high_codes_u8x16));
                __m256i diagonal_u16x16 = column[0];
                __m256i above_u16x16 = _mm256_adds_epu16(diagonal_u16x16, gap_u16x16);
                __m256i minimum_u16x16 = above_u16x16;
                column[0] = above_u16x16;
                for (std::size_t row = 1; row <= rows; ++row) {
                    __m256i const left_u16x16 = column[row];
                    __m256i const aligned_u16x16 = _mm256_adds_epu16(diagonal_u16x16, profile_u16x16[query[row - 1]]);
                    __m256i const opened_u16x16 = _mm256_min_epu16(aligned_u16x16,
                                                                   _mm256_adds_epu16(left_u16x16, gap_u16x16));
                    above_u16x16 = _mm256_min_epu16(opened_u16x16, _mm256_adds_epu16(above_u16x16, gap_u16x16));
                    diagonal_u16x16 = left_u16x16;
                    column[row] = above_u16x16;
                    minimum_u16x16 = _mm256_min_epu16(minimum_u16x16, above_u16x16);
                }
                __m256i const ended_u16x16 = _mm256_andnot_si256(
                    finished_u16x16,
                    _mm256_cmpeq_epi16(lengths_u16x16, _mm256_set1_epi16(static_cast<short>(index + 1))));
                results_u16x16 = _mm256_blendv_epi8(results_u16x16, above_u16x16, ended_u16x16);
                finished_u16x16 = _mm256_or_si256(finished_u16x16, ended_u16x16);
                // Every alignment crosses each column, so none ends below the column's minimum.
                __m256i const within_u16x16 = _mm256_cmpeq_epi16(_mm256_min_epu16(minimum_u16x16, radius_u16x16),
                                                                 minimum_u16x16);
                __m256i const past_u16x16 = _mm256_andnot_si256(_mm256_or_si256(finished_u16x16, within_u16x16),
                                                                _mm256_set1_epi16(-1));
                results_u16x16 = _mm256_blendv_epi8(results_u16x16, minimum_u16x16, past_u16x16);
                finished_u16x16 = _mm256_or_si256(finished_u16x16, past_u16x16);
            }

            u16_t results[canonical_lanes_k];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), results_u16x16);
            u32_t* const scalar_column = static_cast<u32_t*>(aligned);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                u32_t const slot = order[first + lane];
                bool const rescored = (scalar >> lane & 1u) || (results[lane] == saturated_k && radius >= saturated_k);
                distances[slot] = rescored ? alignment_serial(costs, query, candidates[slot], radius, scalar_column)
                                           : results[lane];
            }
        }
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build leaves out, kept so @c alignment_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct packed_haswell_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

/** Adds the window of @c width @p classes to every column of @p tile and advances its
 *  angles, eight @c u16 phases a register, their cosines loaded one by one. */
inline void spectrum_step_neon(spectrum_view_t const& tables, u8_t const* classes, spectrum_tile_t& tile) noexcept {
    u16_t const* rows[spectrum_width_max_k];
    spectrum_rows(tables, classes, tile, rows);
    std::size_t const width = tables.head->width;
    for (std::size_t column = 0; column < tile.columns; column += sizeof(uint16x8_t) / sizeof(u16_t)) {
        uint32x4_t const angles_low_u32x4 = vld1q_u32(tile.angles + column);
        uint32x4_t const angles_high_u32x4 = vld1q_u32(tile.angles + column + 4);
        uint16x8_t phases_u16x8 = vcombine_u16(vshrn_n_u32(angles_low_u32x4, 16), vshrn_n_u32(angles_high_u32x4, 16));
        for (std::size_t place = 0; place != width; ++place)
            phases_u16x8 = vaddq_u16(phases_u16x8, vld1q_u16(rows[place] + column));
        uint16x8_t const entries_u16x8 = vshrq_n_u16(phases_u16x8, 6);
        // NEON has no gather, so the eight cosines are loaded one by one and meet in one register.
        i8_t cosines[8];
        cosines[0] = tables.head->cosines[vgetq_lane_u16(entries_u16x8, 0)];
        cosines[1] = tables.head->cosines[vgetq_lane_u16(entries_u16x8, 1)];
        cosines[2] = tables.head->cosines[vgetq_lane_u16(entries_u16x8, 2)];
        cosines[3] = tables.head->cosines[vgetq_lane_u16(entries_u16x8, 3)];
        cosines[4] = tables.head->cosines[vgetq_lane_u16(entries_u16x8, 4)];
        cosines[5] = tables.head->cosines[vgetq_lane_u16(entries_u16x8, 5)];
        cosines[6] = tables.head->cosines[vgetq_lane_u16(entries_u16x8, 6)];
        cosines[7] = tables.head->cosines[vgetq_lane_u16(entries_u16x8, 7)];
        int16x8_t const cosines_i16x8 = vmovl_s8(vld1_s8(cosines));
        vst1q_s32(tile.sums + column, vaddw_s16(vld1q_s32(tile.sums + column), vget_low_s16(cosines_i16x8)));
        vst1q_s32(tile.sums + column + 4, vaddw_s16(vld1q_s32(tile.sums + column + 4), vget_high_s16(cosines_i16x8)));
        vst1q_u32(tile.angles + column, vaddq_u32(angles_low_u32x4, vld1q_u32(tile.deltas + column)));
        vst1q_u32(tile.angles + column + 4, vaddq_u32(angles_high_u32x4, vld1q_u32(tile.deltas + column + 4)));
    }
}

/** Writes the columns of @p tile into @p sketch from its sums, eight a step,
 *  rounded, shifted and saturated. */
inline void spectrum_close_neon(spectrum_view_t const& tables, spectrum_tile_t const& tile, i8_t* sketch) noexcept {
    int32x4_t const rounding_i32x4 = vdupq_n_s32(static_cast<i32_t>((1u << tables.head->shift) >> 1));
    // A negative count shifts right, keeping the sign.
    int32x4_t const shift_i32x4 = vdupq_n_s32(-static_cast<i32_t>(tables.head->shift));
    for (std::size_t column = 0; column < tile.columns; column += sizeof(uint16x8_t) / sizeof(u16_t)) {
        int32x4_t const scaled_low_i32x4 = vshlq_s32(vaddq_s32(vld1q_s32(tile.sums + column), rounding_i32x4),
                                                     shift_i32x4);
        int32x4_t const scaled_high_i32x4 = vshlq_s32(vaddq_s32(vld1q_s32(tile.sums + column + 4), rounding_i32x4),
                                                      shift_i32x4);
        vst1_s8(sketch + column, vqmovn_s16(vcombine_s16(vqmovn_s32(scaled_low_i32x4), vqmovn_s32(scaled_high_i32x4))));
    }
}

/**
 *  @brief Arm NEON: 8 candidates per register, one @c u16 cell each, a column at a time.
 *
 *  The Skylake sweep with byte codes: each column looks its codes up once per class the query holds
 *  with one @c TBL over that class's 32-byte row, widened to the lanes.
 */
struct packed_neon_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;

    /** What a smooth sketch reads, derived once from the costs. */
    using sketch_tables_t = spectrum_view_t;
    using sketch_scalar_t = i8_t;

    /** The window steps of this backend, driven by @c sketch_write over decoded codes. */
    static constexpr auto sketch_step = spectrum_step_neon;
    static constexpr auto sketch_close = spectrum_close_neon;

    /** A smooth sketch keeps its tiles on the stack, and takes no scratch. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }

    /** Writes the smooth sketch of the stored @p sequence, packed class codes, into @p sketch. */
    static void sketch_write(sketch_tables_t const& tables, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t = {}) noexcept {
        sketch_packed_codes<packed_neon_alignment_kit_t>(tables, sequence, sketch);
    }

    /** Squared Euclidean distance between two sketches. */
    static constexpr auto sketch_distance = sqeuclidean_i8_neon;

    /** Candidates one group sweeps, the @c u16 lanes of a Q register. */
    static constexpr std::size_t canonical_lanes_k = sizeof(uint16x8_t) / sizeof(u16_t);

    /** Scratch bytes a round takes, @p query_codes the query's and @p longest_codes the candidates'
     *  at most: a register per query code and one more, and a byte per lane per candidate code. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_codes, std::size_t longest_codes) noexcept {
        return (query_codes + 2) * sizeof(uint16x8_t) + longest_codes * canonical_lanes_k;
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, @p order taken in
     *  groups of @ref canonical_lanes_k. */
    static void canonical_distances(alignment_costs_t const& costs, std::span<u8_t const> query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        constexpr u32_t saturated_k = (std::numeric_limits<u16_t>::max)();
        std::size_t const rows = query.size();
        u32_t const gap = costs.gap;
        void* aligned = scratch.data();
        std::size_t space = scratch.size();
        uint16x8_t* const column = static_cast<uint16x8_t*>(
            std::align(sizeof(uint16x8_t), sizeof(uint16x8_t), aligned, space));
        u8_t* const codes = reinterpret_cast<u8_t*>(column + rows + 1);

        // A row of the table per class the query holds, one lookup table of 32 costs each.
        u32_t query_classes = 0;
        for (u8_t const code : query) query_classes = max_of<u32_t>(query_classes, code + 1u);
        uint8x16x2_t views_u8x16x2[alignment_classes_k];
        uint16x8_t profile_u16x8[alignment_classes_k];
        for (u32_t code = 0; code != query_classes; ++code)
            views_u8x16x2[code] = vld1q_u8_x2(costs.substitutions[code]);
        uint16x8_t const gap_u16x8 = vdupq_n_u16(static_cast<u16_t>(gap));
        uint16x8_t const radius_u16x8 = vdupq_n_u16(static_cast<u16_t>(min_of<sz_size_t>(radius, saturated_k)));

        for (std::size_t first = 0; first < order.size(); first += canonical_lanes_k) {
            std::size_t const lanes = min_of(canonical_lanes_k, order.size() - first);
            u16_t lengths[canonical_lanes_k] = {};
            u16_t finished[canonical_lanes_k] = {};
            u32_t scalar = 0;
            std::size_t longest = 0;
            for (std::size_t lane = 0; lane != canonical_lanes_k; ++lane) {
                if (lane >= lanes) {
                    finished[lane] = static_cast<u16_t>(saturated_k);
                    continue;
                }
                bytes_view_t const candidate = candidates[order[first + lane]];
                std::size_t const length = packed_codes_count(candidate.data(), candidate.size());
                if (length > saturated_k) scalar |= u32_t {1} << lane;
                else lengths[lane] = static_cast<u16_t>(length), longest = max_of(longest, length);
                // Padding lanes, empty ones and scalar ones count as finished from the start.
                if (!lengths[lane]) finished[lane] = static_cast<u16_t>(saturated_k);
            }
            std::memset(codes, 0, longest * canonical_lanes_k);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                bytes_view_t const candidate = candidates[order[first + lane]];
                for (std::size_t index = 0; index != lengths[lane]; ++index)
                    codes[index * canonical_lanes_k + lane] = static_cast<u8_t>(
                        packed_code_at(candidate.data(), index));
            }

            for (std::size_t row = 0; row <= rows; ++row)
                column[row] = vdupq_n_u16(static_cast<u16_t>(min_of<std::size_t>(row * gap, saturated_k)));
            uint16x8_t const lengths_u16x8 = vld1q_u16(lengths);
            uint16x8_t finished_u16x8 = vld1q_u16(finished);
            uint16x8_t results_u16x8 = column[rows];
            for (std::size_t index = 0; index != longest && vminvq_u16(finished_u16x8) != saturated_k; ++index) {
                uint8x8_t const codes_u8x8 = vld1_u8(codes + index * canonical_lanes_k);
                for (u32_t code = 0; code != query_classes; ++code)
                    profile_u16x8[code] = vmovl_u8(vqtbl2_u8(views_u8x16x2[code], codes_u8x8));
                uint16x8_t diagonal_u16x8 = column[0];
                uint16x8_t above_u16x8 = vqaddq_u16(diagonal_u16x8, gap_u16x8);
                uint16x8_t minimum_u16x8 = above_u16x8;
                column[0] = above_u16x8;
                for (std::size_t row = 1; row <= rows; ++row) {
                    uint16x8_t const left_u16x8 = column[row];
                    uint16x8_t const aligned_u16x8 = vqaddq_u16(diagonal_u16x8, profile_u16x8[query[row - 1]]);
                    uint16x8_t const opened_u16x8 = vminq_u16(aligned_u16x8, vqaddq_u16(left_u16x8, gap_u16x8));
                    above_u16x8 = vminq_u16(opened_u16x8, vqaddq_u16(above_u16x8, gap_u16x8));
                    diagonal_u16x8 = left_u16x8;
                    column[row] = above_u16x8;
                    minimum_u16x8 = vminq_u16(minimum_u16x8, above_u16x8);
                }
                uint16x8_t const ended_u16x8 = vbicq_u16(
                    vceqq_u16(lengths_u16x8, vdupq_n_u16(static_cast<u16_t>(index + 1))), finished_u16x8);
                results_u16x8 = vbslq_u16(ended_u16x8, above_u16x8, results_u16x8);
                finished_u16x8 = vorrq_u16(finished_u16x8, ended_u16x8);
                // Every alignment crosses each column, so none ends below the column's minimum.
                uint16x8_t const past_u16x8 = vbicq_u16(vcgtq_u16(minimum_u16x8, radius_u16x8), finished_u16x8);
                results_u16x8 = vbslq_u16(past_u16x8, minimum_u16x8, results_u16x8);
                finished_u16x8 = vorrq_u16(finished_u16x8, past_u16x8);
            }

            u16_t results[canonical_lanes_k];
            vst1q_u16(results, results_u16x8);
            u32_t* const scalar_column = static_cast<u32_t*>(aligned);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                u32_t const slot = order[first + lane];
                bool const rescored = (scalar >> lane & 1u) || (results[lane] == saturated_k && radius >= saturated_k);
                distances[slot] = rescored ? alignment_serial(costs, query, candidates[slot], radius, scalar_column)
                                           : results[lane];
            }
        }
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build leaves out, kept so @c alignment_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct packed_neon_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_NEON

#if STRINGZILLA_TARGET_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/** Adds the window of @c width @p classes to every column of @p tile and advances its
 *  angles, eight @c u16 phases a register, their cosines loaded one by one. */
inline void spectrum_step_powervsx(spectrum_view_t const& tables, u8_t const* classes, spectrum_tile_t& tile) noexcept {
    u16_t const* rows[spectrum_width_max_k];
    spectrum_rows(tables, classes, tile, rows);
    std::size_t const width = tables.head->width;
    __vector unsigned int const sixteen_u32x4 = vec_splats(16u);
    for (std::size_t column = 0; column < tile.columns; column += sizeof(__vector unsigned short) / sizeof(u16_t)) {
        __vector unsigned int const angles_low_u32x4 = vec_xl(0, tile.angles + column);
        __vector unsigned int const angles_high_u32x4 = vec_xl(0, tile.angles + column + 4);
        __vector unsigned short phases_u16x8 = vec_pack(vec_sr(angles_low_u32x4, sixteen_u32x4),
                                                        vec_sr(angles_high_u32x4, sixteen_u32x4));
        for (std::size_t place = 0; place != width; ++place)
            phases_u16x8 = vec_add(phases_u16x8, vec_xl(0, rows[place] + column));
        __vector unsigned short const entries_u16x8 = vec_sr(phases_u16x8, vec_splats(static_cast<unsigned short>(6)));
        // VSX has no gather, so the eight cosines are loaded one by one and meet in two registers.
        __vector signed int const cosines_low_i32x4 = {
            tables.head->cosines[vec_extract(entries_u16x8, 0)], tables.head->cosines[vec_extract(entries_u16x8, 1)],
            tables.head->cosines[vec_extract(entries_u16x8, 2)], tables.head->cosines[vec_extract(entries_u16x8, 3)]};
        __vector signed int const cosines_high_i32x4 = {
            tables.head->cosines[vec_extract(entries_u16x8, 4)], tables.head->cosines[vec_extract(entries_u16x8, 5)],
            tables.head->cosines[vec_extract(entries_u16x8, 6)], tables.head->cosines[vec_extract(entries_u16x8, 7)]};
        vec_xst(vec_add(vec_xl(0, tile.sums + column), cosines_low_i32x4), 0, tile.sums + column);
        vec_xst(vec_add(vec_xl(0, tile.sums + column + 4), cosines_high_i32x4), 0, tile.sums + column + 4);
        vec_xst(vec_add(angles_low_u32x4, vec_xl(0, tile.deltas + column)), 0, tile.angles + column);
        vec_xst(vec_add(angles_high_u32x4, vec_xl(0, tile.deltas + column + 4)), 0, tile.angles + column + 4);
    }
}

/** Writes the columns of @p tile into @p sketch from its sums, eight a step,
 *  rounded, shifted and saturated. */
inline void spectrum_close_powervsx(spectrum_view_t const& tables, spectrum_tile_t const& tile, i8_t* sketch) noexcept {
    __vector signed int const rounding_i32x4 = vec_splats(static_cast<int>((1u << tables.head->shift) >> 1));
    __vector unsigned int const shift_u32x4 = vec_splats(static_cast<unsigned int>(tables.head->shift));
    for (std::size_t column = 0; column < tile.columns; column += sizeof(__vector unsigned short) / sizeof(u16_t)) {
        __vector signed int const scaled_low_i32x4 = vec_sra(vec_add(vec_xl(0, tile.sums + column), rounding_i32x4),
                                                             shift_u32x4);
        __vector signed int const scaled_high_i32x4 = vec_sra(
            vec_add(vec_xl(0, tile.sums + column + 4), rounding_i32x4), shift_u32x4);
        // Two saturating packs narrow to bytes, of which the first eight are the columns.
        __vector signed short const scaled_i16x8 = vec_packs(scaled_low_i32x4, scaled_high_i32x4);
        vec_xst_len(vec_packs(scaled_i16x8, scaled_i16x8), reinterpret_cast<signed char*>(sketch + column), 8);
    }
}

/**
 *  @brief IBM Power VSX: 8 candidates per register, one @c u16 cell each, a column at a time.
 *
 *  The Skylake sweep with byte codes: each column looks its codes up once per class the query holds
 *  with one @c vperm over that class's 32-byte row, widened to the lanes.
 */
struct packed_powervsx_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_powervsx_k;

    /** What a smooth sketch reads, derived once from the costs. */
    using sketch_tables_t = spectrum_view_t;
    using sketch_scalar_t = i8_t;

    /** The window steps of this backend, driven by @c sketch_write over decoded codes. */
    static constexpr auto sketch_step = spectrum_step_powervsx;
    static constexpr auto sketch_close = spectrum_close_powervsx;

    /** A smooth sketch keeps its tiles on the stack, and takes no scratch. */
    static constexpr std::size_t sketch_scratch_bytes(std::size_t) noexcept { return 0; }

    /** Writes the smooth sketch of the stored @p sequence, packed class codes, into @p sketch. */
    static void sketch_write(sketch_tables_t const& tables, bytes_view_t sequence, bytes_span_t sketch,
                             bytes_span_t = {}) noexcept {
        sketch_packed_codes<packed_powervsx_alignment_kit_t>(tables, sequence, sketch);
    }

    /** Squared Euclidean distance between two sketches. */
    static constexpr auto sketch_distance = nk_sqeuclidean_i8_powervsx;

    /** Candidates one group sweeps, the @c u16 lanes of a VSX register. */
    static constexpr std::size_t canonical_lanes_k = sizeof(__vector unsigned short) / sizeof(u16_t);

    /** Scratch bytes a round takes, @p query_codes the query's and @p longest_codes the candidates'
     *  at most: a register per query code and one more, and a byte per lane per candidate code. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_codes, std::size_t longest_codes) noexcept {
        return (query_codes + 2) * sizeof(__vector unsigned short) + longest_codes * canonical_lanes_k;
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, @p order taken in
     *  groups of @ref canonical_lanes_k. */
    static void canonical_distances(alignment_costs_t const& costs, std::span<u8_t const> query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        constexpr u32_t saturated_k = (std::numeric_limits<u16_t>::max)();
        std::size_t const rows = query.size();
        u32_t const gap = costs.gap;
        void* aligned = scratch.data();
        std::size_t space = scratch.size();
        __vector unsigned short* const column = static_cast<__vector unsigned short*>(
            std::align(sizeof(__vector unsigned short), sizeof(__vector unsigned short), aligned, space));
        u8_t* const codes = reinterpret_cast<u8_t*>(column + rows + 1);

        // A row of the table per class the query holds, one lookup table of 32 costs each.
        u32_t query_classes = 0;
        for (u8_t const code : query) query_classes = max_of<u32_t>(query_classes, code + 1u);
        __vector unsigned char views_low_u8x16[alignment_classes_k];
        __vector unsigned char views_high_u8x16[alignment_classes_k];
        __vector unsigned short profile_u16x8[alignment_classes_k];
        for (u32_t code = 0; code != query_classes; ++code) {
            views_low_u8x16[code] = vec_xl(0, costs.substitutions[code]);
            views_high_u8x16[code] = vec_xl(16, costs.substitutions[code]);
        }
        __vector unsigned short const byte_u16x8 = vec_splats(static_cast<unsigned short>(0xFF));
        __vector unsigned short const saturated_u16x8 = vec_splats(static_cast<unsigned short>(saturated_k));
        __vector unsigned short const gap_u16x8 = vec_splats(static_cast<unsigned short>(gap));
        __vector unsigned short const radius_u16x8 = vec_splats(
            static_cast<unsigned short>(min_of<sz_size_t>(radius, saturated_k)));

        for (std::size_t first = 0; first < order.size(); first += canonical_lanes_k) {
            std::size_t const lanes = min_of(canonical_lanes_k, order.size() - first);
            u16_t lengths[canonical_lanes_k] = {};
            u16_t finished[canonical_lanes_k] = {};
            u32_t scalar = 0;
            std::size_t longest = 0;
            for (std::size_t lane = 0; lane != canonical_lanes_k; ++lane) {
                if (lane >= lanes) {
                    finished[lane] = static_cast<u16_t>(saturated_k);
                    continue;
                }
                bytes_view_t const candidate = candidates[order[first + lane]];
                std::size_t const length = packed_codes_count(candidate.data(), candidate.size());
                if (length > saturated_k) scalar |= u32_t {1} << lane;
                else lengths[lane] = static_cast<u16_t>(length), longest = max_of(longest, length);
                // Padding lanes, empty ones and scalar ones count as finished from the start.
                if (!lengths[lane]) finished[lane] = static_cast<u16_t>(saturated_k);
            }
            std::memset(codes, 0, longest * canonical_lanes_k);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                bytes_view_t const candidate = candidates[order[first + lane]];
                for (std::size_t index = 0; index != lengths[lane]; ++index)
                    codes[index * canonical_lanes_k + lane] = static_cast<u8_t>(
                        packed_code_at(candidate.data(), index));
            }

            for (std::size_t row = 0; row <= rows; ++row)
                column[row] = vec_splats(static_cast<unsigned short>(min_of<std::size_t>(row * gap, saturated_k)));
            __vector unsigned short const lengths_u16x8 = vec_xl(0, lengths);
            __vector unsigned short finished_u16x8 = vec_xl(0, finished);
            __vector unsigned short results_u16x8 = column[rows];
            for (std::size_t index = 0; index != longest && !vec_all_eq(finished_u16x8, saturated_u16x8); ++index) {
                __vector unsigned char const codes_u8x16 = vec_xl_len(codes + index * canonical_lanes_k,
                                                                      canonical_lanes_k);
                for (u32_t code = 0; code != query_classes; ++code)
                    profile_u16x8[code] = vec_and(
                        reinterpret_cast<__vector unsigned short>(vec_unpackh(reinterpret_cast<__vector signed char>(
                            vec_perm(views_low_u8x16[code], views_high_u8x16[code], codes_u8x16)))),
                        byte_u16x8);
                __vector unsigned short diagonal_u16x8 = column[0];
                __vector unsigned short above_u16x8 = vec_adds(diagonal_u16x8, gap_u16x8);
                __vector unsigned short minimum_u16x8 = above_u16x8;
                column[0] = above_u16x8;
                for (std::size_t row = 1; row <= rows; ++row) {
                    __vector unsigned short const left_u16x8 = column[row];
                    __vector unsigned short const aligned_u16x8 = vec_adds(diagonal_u16x8,
                                                                           profile_u16x8[query[row - 1]]);
                    __vector unsigned short const opened_u16x8 = vec_min(aligned_u16x8,
                                                                         vec_adds(left_u16x8, gap_u16x8));
                    above_u16x8 = vec_min(opened_u16x8, vec_adds(above_u16x8, gap_u16x8));
                    diagonal_u16x8 = left_u16x8;
                    column[row] = above_u16x8;
                    minimum_u16x8 = vec_min(minimum_u16x8, above_u16x8);
                }
                __vector unsigned short const ended_u16x8 = vec_andc(
                    reinterpret_cast<__vector unsigned short>(
                        vec_cmpeq(lengths_u16x8, vec_splats(static_cast<unsigned short>(index + 1)))),
                    finished_u16x8);
                results_u16x8 = vec_sel(results_u16x8, above_u16x8, ended_u16x8);
                finished_u16x8 = vec_or(finished_u16x8, ended_u16x8);
                // Every alignment crosses each column, so none ends below the column's minimum.
                __vector unsigned short const past_u16x8 = vec_andc(
                    reinterpret_cast<__vector unsigned short>(vec_cmpgt(minimum_u16x8, radius_u16x8)), finished_u16x8);
                results_u16x8 = vec_sel(results_u16x8, minimum_u16x8, past_u16x8);
                finished_u16x8 = vec_or(finished_u16x8, past_u16x8);
            }

            u16_t results[canonical_lanes_k];
            vec_xst(results_u16x8, 0, results);
            u32_t* const scalar_column = static_cast<u32_t*>(aligned);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                u32_t const slot = order[first + lane];
                bool const rescored = (scalar >> lane & 1u) || (results[lane] == saturated_k && radius >= saturated_k);
                distances[slot] = rescored ? alignment_serial(costs, query, candidates[slot], radius, scalar_column)
                                           : results[lane];
            }
        }
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build leaves out, kept so @c alignment_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct packed_powervsx_alignment_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_powervsx_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_POWERVSX

/** The widest alignment kit this build compiled in among the backends @p caps_ admits,
 *  newest first. */
template <sz_capability_t caps_>
struct alignment_ladder;

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cpus_k) != 0)
struct alignment_ladder<caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        packed_serial_alignment_kit_t,  //
        packed_icelake_alignment_kit_t, //
        packed_skylake_alignment_kit_t, //
        packed_haswell_alignment_kit_t, //
        packed_neon_alignment_kit_t,    //
        packed_powervsx_alignment_kit_t //
        >::type;
};

#pragma endregion Alignment Kernels

#pragma region Alignment Metric

/**
 *  @brief Alignment cost from the query to each staged candidate over packed class codes: a
 *      32×32 table of @c u4 costs and one linear gap cost, exact up to the radius.
 *
 *  @c reset unpacks the query's codes, @c push stages candidate views, and @c drain scores the
 *  round in ascending length order. One instance belongs to one thread and owns the staging, the
 *  order and the kit's scratch, which grows to the longest query and candidate seen; the costs are
 *  viewed from @ref init's caller.
 *
 *  @tparam kit_type_ The kernel of one ISA backend, as @c alignment_ladder picks it.
 *  @tparam allocator_type_ places every buffer, one set per metric.
 *  @tparam cache_hints_type_ prefetches a staged candidate's bytes; ForkUnion's no-op policy by
 *      default.
 */
template <typename kit_type_ = typename alignment_ladder<sz_cap_any_k>::kit_t,
          typename allocator_type_ = std::allocator<byte_t>, typename cache_hints_type_ = fu::standard_cache_hints_t>
class alignment_metric {
  public:
    using kit_t = kit_type_;
    using distance_t = sz_size_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;

    /** The flat form a batched round hands @ref reset for each stored sequence: its packed
     *  bytes. */
    using query_view_t = sz::string_view_t;
    template <typename string_type_>
    static query_view_t query_view(string_type_ const& value) noexcept {
        return query_view_t {value};
    }

  private:

    /** The folded table and gap cost, read until the metric dies. */
    alignment_costs_t const* costs_ = nullptr;

    /** The codes of every prepared query, one byte apiece and back to back. */
    growing_buffer<u8_t, allocator_t> query_;

    /** Where each prepared query's codes end in @c query_, one entry per query. */
    growing_buffer<u32_t, allocator_t> query_ends_;

    /** The round's candidates in push order. */
    growing_buffer<bytes_view_t, allocator_t> staged_;

    /** The round's push slots in ascending length order. */
    growing_buffer<u32_t, allocator_t> order_;

    /** The kit's cells and transposed codes. */
    growing_buffer<byte_t, allocator_t> scratch_;

    /** How the last @ref init, @ref reset or @ref drain fared. */
    status_t last_status_ = status_code_t::uninitialized_k;

  public:
    alignment_metric(allocator_t const& allocator = {}) noexcept
        : query_(allocator), query_ends_(allocator), staged_(allocator), order_(allocator), scratch_(allocator) {}
    alignment_metric(alignment_metric const&) = delete;
    alignment_metric& operator=(alignment_metric const&) = delete;
    alignment_metric(alignment_metric&&) noexcept = default;
    alignment_metric& operator=(alignment_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging and the order, and records the costs a round scores with.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at
     *      least the index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or
     *      rounds saturate rather than fail. A round is bounded to @c u32_t slots by design.
     *  @param[in] costs A folded table, read until the metric dies.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p staging_limit is zero or past @c u32_t slots, or @p costs is not
     *    folded.
     */
    status_t init(std::size_t staging_limit, alignment_costs_t const& costs) noexcept {
        if (!staging_limit || staging_limit > (std::numeric_limits<u32_t>::max)() || !costs.classes)
            return status_code_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        if (status_t const ordered = order_.reserve_exactly(staging_limit); failed(ordered)) return ordered;
        costs_ = &costs;
        last_status_ = status_code_t::success_k;
        return last_status_;
    }

    /** How the last @ref init, @ref reset or @ref drain fared - the only place a failure surfaces,
     *  since the scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a round for the packed @p query, unpacking its codes, and discarding whatever the
     *  previous round staged. */
    void reset(sz::string_view_t query) noexcept {
        staged_.clear();
        if (!costs_) return;
        query_.clear(), query_ends_.clear();
        last_status_ = unpack_(query);
    }

    /** Opens rounds scoring every packed query @p queries holds at once, so each @ref drain writes
     *  @b [queries,candidates]. */
    template <typename query_type_>
        requires std::convertible_to<query_type_ const&, sz::string_view_t>
    void reset(std::span<query_type_ const> queries) noexcept {
        staged_.clear();
        if (!costs_) return;
        query_.clear(), query_ends_.clear();
        last_status_ = status_code_t::success_k;
        for (std::size_t query = 0; succeeded(last_status_) && query != queries.size(); ++query)
            last_status_ = unpack_(sz::string_view_t {queries[query]});
    }

    /** Stages one candidate, or refuses once the round is full. */
    basic::metric_staging_t push(sz::string_view_t value) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        prefetch_(value);
        staged_.push_back_reserved(bytes_of(value.data(), value.size()));
        return basic::metric_staging_t::staged_k;
    }

    /**
     *  @brief Scores everything staged and closes the round.
     *
     *  Distances up to and including @p radius are exact; above it the value is some value above
     *  the radius. A failed round reports the furthest distance, so a full pool rejects it and an
     *  unfilled pool admits it last; @ref status is the only signal.
     *
     *  @param[out] distances @b [queries,candidates] slots, a staged candidate's in push order.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<sz_size_t> distances, sz_size_t radius) noexcept {
        std::size_t const count = staged_.size(), queries = query_ends_.size();
        std::size_t longest = 0, longest_query = 0;
        order_.clear();
        for (std::size_t slot = 0; slot != count; ++slot) {
            order_.push_back_reserved(static_cast<u32_t>(slot));
            longest = max_of(longest, length_(slot));
        }
        partial_sort(to_writable_span(order_), order_.size(),
                     [&](u32_t first, u32_t second) noexcept { return length_(first) < length_(second); });
        for (std::size_t prepared = 0; prepared != query_ends_.size(); ++prepared)
            longest_query = max_of(longest_query, codes_(prepared).size());
        std::size_t const scratch_bytes = kit_t::canonical_scratch_bytes(longest_query, longest);
        if (succeeded(last_status_) && scratch_.size() < scratch_bytes) {
            scratch_.clear();
            last_status_ = scratch_.reserve(scratch_bytes);
            if (succeeded(last_status_)) scratch_.extend_uninitialized(scratch_.capacity());
        }
        bool const scored = succeeded(last_status_) && distances.size() == queries * count;
        if (!scored)
            for (sz_size_t& distance : distances) distance = (std::numeric_limits<sz_size_t>::max)();
        bytes_span_t const scratch = bytes_of(scratch_.data(), scratch_.size());
        for (std::size_t query = 0; scored && query != queries; ++query) {
            std::span<sz_size_t> const own = distances.subspan(query * count, count);
            kit_t::canonical_distances(*costs_, codes_(query), to_span(staged_), to_span(order_), radius, own, scratch);
        }
        staged_.clear();
    }

  private:

    /** The codes of the prepared query at @p prepared. */
    std::span<u8_t const> codes_(std::size_t prepared) const noexcept {
        std::size_t const first = prepared ? query_ends_[prepared - 1] : 0;
        return to_span(query_).subspan(first, query_ends_[prepared] - first);
    }

    /** Appends the codes of the packed @p query. */
    status_t unpack_(sz::string_view_t query) noexcept {
        bytes_view_t const packed = bytes_of(query.data(), query.size());
        std::size_t const codes = packed_codes_count(packed.data(), packed.size());
        if (status_t const grown = query_.reserve(query_.size() + codes); failed(grown)) return grown;
        if (status_t const grown = query_ends_.reserve(query_ends_.size() + 1); failed(grown)) return grown;
        for (std::size_t index = 0; index != codes; ++index)
            query_.push_back_reserved(static_cast<u8_t>(packed_code_at(packed.data(), index)));
        query_ends_.push_back_reserved(static_cast<u32_t>(query_.size()));
        return status_code_t::success_k;
    }

    /** Codes the candidate staged at @p slot holds. */
    std::size_t length_(std::size_t slot) const noexcept {
        return packed_codes_count(staged_[slot].data(), staged_[slot].size());
    }

    /** Starts the misses on every line of @p value, which the drain reads. */
    static void prefetch_(sz::string_view_t value) noexcept {
        for (std::size_t offset = 0; offset < value.size(); offset += fu::default_alignment_k)
            cache_hints_type_ {}(value.data() + offset, fu::prefetch_line_k);
    }
};

using alignment_metric_t = alignment_metric<>;

#pragma endregion Alignment Metric

#pragma region Banded Overlap Kernels

/** The furthest banded overlap distance, where fewer than two windows agree on any shift: laid out
 *  as a containment distance is, the top primary level above the top secondary one. */
inline constexpr u64_t banded_overlap_ceiling_k = (share_primary_levels_k << share_secondary_bits_k) |
                                                  ((u64_t(1) << share_secondary_bits_k) - 1);

/** The widths of the windows an overlap votes with, in symbols: the least whose windows span 24 and
 *  38 bits of four classes. */
inline constexpr sz_size_t banded_overlap_window_widths_k[] = {12, 19};

/** Windows that must agree on a shift before two sequences overlap at it. */
inline constexpr u32_t banded_overlap_votes_k = 2;

/** Cells either side of the voted diagonal the widest band keeps, 129 cells in all. */
inline constexpr u32_t banded_overlap_widest_band_k = 64;

/** Cells either side of the voted diagonal the band of an overlap of @p span symbols keeps: a
 *  sixteenth of the span, which absorbs 6.25 percent of drifting indels, at least four, which
 *  absorb a homopolymer slip in the shortest read, and at most 64, 129 cells in all. */
constexpr u32_t banded_overlap_band(u64_t span) noexcept {
    return static_cast<u32_t>(min_of<u64_t>(banded_overlap_widest_band_k, max_of<u64_t>(4, span / 16)));
}

/** Cost of one symbol facing a gap under @p costs. */
constexpr u32_t banded_overlap_gap(alignment_costs_t const& costs) noexcept { return costs.gap; }

/**
 *  @brief The banded overlap distance of two sequences, @p shorter and @p longer symbols long,
 *      whose voted shift lays @p span symbols over each other at the band's @p cost.
 *
 *  Lexicographic in fixed point, as @ref containment_distance is: the cost over twice the gap cost
 *  of the shorter side, which bounds it, since a substitution never costs a band more than two
 *  gaps, rounded down to @ref share_primary_levels_k, above the share of the longer side the
 *  overlap leaves out, rounded up.
 *
 *  @verbatim
 *  primary = min(2^15 - 1, cost * (2^15 - 1) / (2 * gap * shorter))
 *  secondary = ceil((longer - span) * 255 / longer)
 *  distance = primary * 256 + secondary
 *  @endverbatim
 */
constexpr u64_t banded_overlap_distance(u64_t cost, u64_t gap, u64_t span, u64_t shorter, u64_t longer) noexcept {
    u64_t const secondary_top = (u64_t(1) << share_secondary_bits_k) - 1;
    u64_t const bound = 2 * gap * shorter;
    u64_t const primary = bound  ? min_of<u64_t>(share_primary_levels_k, cost * share_primary_levels_k / bound)
                          : cost ? share_primary_levels_k
                                 : 0;
    u64_t const secondary = longer ? divide_round_up<u64_t>((longer - span) * secondary_top, longer) : secondary_top;
    return (primary << share_secondary_bits_k) | secondary;
}

/** The least banded overlap distance two sequences of @p shorter and @p longer symbols reach:
 *  no cost, and the whole shorter side laid over the longer. */
constexpr u64_t banded_overlap_lower_bound(u64_t shorter, u64_t longer) noexcept {
    return longer ? banded_overlap_distance(0, 0, shorter, shorter, longer) : 0;
}

/**
 *  @brief The cost of aligning the @p span symbols at @p first to the @p span at @p second end to
 *      end under @p costs, as @ref alignment_serial charges them, inside a band of @p band cells
 *      either side of the diagonal.
 *
 *  Symbols are stored bytes, class codes scored through the folded table. The band's cells of one
 *  query symbol are kept and stepped in place, each the column that many places off the diagonal.
 *
 *  @param[out] cells Working cells, <tt>2 * band + 1</tt> of them.
 */
constexpr u32_t alignment_band(alignment_costs_t const& costs, byte_t const* first, byte_t const* second, u32_t span,
                               u32_t band, u32_t* cells) noexcept {
    // Far past any cost, and still far from wrapping once a gap or a substitution is added.
    constexpr u32_t outside_k = (std::numeric_limits<u32_t>::max)() / 2;
    u32_t const gap = banded_overlap_gap(costs), count = 2 * band + 1;
    for (u32_t cell = 0; cell != count; ++cell)
        cells[cell] = cell >= band && cell - band <= span ? (cell - band) * gap : outside_k;
    for (u32_t index = 1; index <= span; ++index) {
        u32_t const symbol = std::to_integer<u32_t>(first[index - 1]);
        for (u32_t cell = 0; cell != count; ++cell) {
            // The column this cell holds for the symbol `index`, offset by the band.
            u32_t const column = index + cell;
            if (column < band || column - band > span) {
                cells[cell] = outside_k;
                continue;
            }
            if (column == band) {
                cells[cell] = index * gap;
                continue;
            }
            u32_t const other = std::to_integer<u32_t>(second[column - band - 1]);
            u32_t const substitution = costs.substitutions[symbol][other];
            u32_t const above = cell + 1 != count ? cells[cell + 1] : outside_k;
            u32_t const left = cell ? cells[cell - 1] : outside_k;
            cells[cell] = min_of(cells[cell] + substitution, min_of(above, left) + gap);
        }
    }
    return cells[band];
}

/** A prepared banded overlap query: the query and its reverse complement, each with
 *  the table of its distinct windows and the symbols the band aligns. */
struct banded_overlap_query_t {

    /** The window table of each image. */
    sz_ngrams_query_t tables[2];

    /** The symbols of each image, a byte apiece: the query, then its reverse complement. */
    bytes_view_t images[2];

    /** One image, or two under complements. */
    std::size_t count;
};

/** Scratch bytes a round of a @p query_symbols long query and candidates of up to
 *  @p longest_symbols takes: a candidate's prefix residues and the slack aligning them, its window
 *  residues at every width, a vote per shift the pair can take, and one band's cells. */
constexpr std::size_t banded_overlap_scratch_bytes(std::size_t query_symbols, std::size_t longest_symbols) noexcept {
    std::size_t const residues = sz_ngrams_count(longest_symbols, banded_overlap_window_widths_k,
                                                 std::size(banded_overlap_window_widths_k));
    return (longest_symbols + 2) * sizeof(sz_f64_t) +
           (residues + query_symbols + longest_symbols + 2 * banded_overlap_widest_band_k + 1) * sizeof(u32_t);
}

/**
 *  @brief The banded overlap distance from one image of a query, its @p table and @p symbols, to
 *      @p candidate, whose window residues at the table's widths are at @p residues.
 *
 *  Every window of the candidate found once in the image votes for the shift between the two
 *  places; the most voted shift wins, the least on a tie, and the symbols it lays over each other
 *  are aligned inside the band. @ref banded_overlap_ceiling_k where fewer than
 *  @ref banded_overlap_votes_k windows agree.
 *
 *  @param[out] votes Working space of one counter per shift, the two lengths' sum less one.
 *  @param[out] cells Working cells of the widest band.
 */
inline sz_size_t banded_overlap_image_distance(alignment_costs_t const& costs, sz_ngrams_query_t const& table,
                                               bytes_view_t symbols, bytes_view_t candidate, sz_u32_t const* residues,
                                               u32_t* votes, u32_t* cells) noexcept {
    std::size_t const own = symbols.size(), other = candidate.size();
    if (!own || !other) return static_cast<sz_size_t>(banded_overlap_ceiling_k);
    // A vote's shift is the candidate's place less the image's, counted from the lowest one.
    std::size_t const shifts = own + other - 1;
    std::memset(votes, 0, shifts * sizeof(u32_t));
    for (std::size_t width_index = 0; width_index != table.widths_count; ++width_index) {
        if (table.widths[width_index] > own) continue;
        // Residues run width after width, each width's windows in position order.
        sz_u32_t const* const width_residues = residues + sz_ngrams_count(other, table.widths, width_index);
        std::size_t const windows = sz_ngrams_count(other, table.widths + width_index, 1);
        for (std::size_t place = 0; place != windows; ++place) {
            sz_u64_t const slot = sz_ngrams_query_probe(&table, width_residues[place], width_index);
            // A window the image repeats names no single shift.
            if (!slot || sz_ngrams_slot_repeated(slot)) continue;
            ++votes[place + own - 1 - sz_ngrams_slot_first(slot)];
        }
    }
    std::size_t voted = 0;
    for (std::size_t shift = 1; shift != shifts; ++shift)
        if (votes[shift] > votes[voted]) voted = shift;
    if (votes[voted] < banded_overlap_votes_k) return static_cast<sz_size_t>(banded_overlap_ceiling_k);

    // The image's symbols from `low` to `high` face the candidate's the shift further on.
    std::size_t const low = voted < own - 1 ? own - 1 - voted : 0;
    std::size_t const facing = voted + low - (own - 1);
    std::size_t const span = min_of(own - low, other - facing);
    std::size_t const shorter = min_of(own, other), longer = max_of(own, other);
    u32_t const gap = banded_overlap_gap(costs);
    u64_t const cost = u64_t {alignment_band(costs, symbols.data() + low, candidate.data() + facing,
                                             static_cast<u32_t>(span), banded_overlap_band(span), cells)} +
                       u64_t {gap} * (shorter - span);
    return static_cast<sz_size_t>(banded_overlap_distance(cost, gap, span, shorter, longer));
}

/**
 *  @brief Scores every candidate of @p candidates against @p query into its slot of @p distances,
 *      in @p order, its windows' residues by the producer @p kit_ binds.
 *
 *  A distance is the lower one of the query's images, exact at any @p radius; a pair whose lengths
 *  alone put it past the radius takes the bound those lengths give.
 */
template <typename kit_>
void banded_overlap_distances(alignment_costs_t const& costs, banded_overlap_query_t const& query,
                              std::span<bytes_view_t const> candidates, std::span<u32_t const> order, sz_size_t radius,
                              std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
    std::size_t const own = query.images[0].size();
    std::size_t const widths_count = std::size(banded_overlap_window_widths_k);
    void* aligned = scratch.data();
    std::size_t space = scratch.size();
    sz_f64_t* const prefixes = static_cast<sz_f64_t*>(std::align(alignof(sz_f64_t), sizeof(sz_f64_t), aligned, space));
    for (u32_t const slot : order) {
        bytes_view_t const candidate = candidates[slot];
        std::size_t const other = candidate.size();
        sz_size_t const bound = static_cast<sz_size_t>(
            banded_overlap_lower_bound(min_of(own, other), max_of(own, other)));
        if (bound > radius) {
            distances[slot] = bound;
            continue;
        }
        sz_u32_t* const residues = reinterpret_cast<sz_u32_t*>(prefixes + other + 1);
        kit_::canonical_residues(chars_of(candidate).data(), other, banded_overlap_window_widths_k, widths_count,
                                 prefixes, residues);
        u32_t* const votes = residues + sz_ngrams_count(other, banded_overlap_window_widths_k, widths_count);
        u32_t* const cells = votes + own + other;
        sz_size_t distance = static_cast<sz_size_t>(banded_overlap_ceiling_k);
        for (std::size_t image = 0; image != query.count; ++image)
            distance = min_of(distance, banded_overlap_image_distance(costs, query.tables[image], query.images[image],
                                                                      candidate, residues, votes, cells));
        distances[slot] = distance;
    }
}

/** Scalar: one position per hashing step for the vote, one @c u32 cell per band column. */
struct u8_serial_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_serial;

    /** StringZilla's residues of a candidate's windows at every width, which vote. */
    static constexpr auto canonical_residues = sz_ngrams_residues_serial;

    /** An overlap index builds by this score, so it keeps no sketch. */
    using sketch_tables_t = std::monostate;

    /** Scratch bytes a round of a @p query_symbols long query and candidates of up to
     *  @p longest_symbols takes: prefix residues, window residues, votes and one band. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_symbols,
                                                         std::size_t longest_symbols) noexcept {
        return banded_overlap_scratch_bytes(query_symbols, longest_symbols);
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, in @p order. */
    static void canonical_distances(alignment_costs_t const& costs, banded_overlap_query_t const& query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        banded_overlap_distances<u8_serial_banded_overlap_kit_t>(costs, query, candidates, order, radius, distances,
                                                                 scratch);
    }
};

#if STRINGZILLA_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "bmi", "bmi2", "popcnt")
#endif

/** Intel Haswell AVX2: four positions per hashing step for the vote, the scalar band. */
struct u8_haswell_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_haswell;

    /** StringZilla's residues of a candidate's windows at every width, which vote. */
    static constexpr auto canonical_residues = sz_ngrams_residues_haswell;

    /** An overlap index builds by this score, so it keeps no sketch. */
    using sketch_tables_t = std::monostate;

    /** Scratch bytes a round of a @p query_symbols long query and candidates of up to
     *  @p longest_symbols takes: prefix residues, window residues, votes and one band. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_symbols,
                                                         std::size_t longest_symbols) noexcept {
        return banded_overlap_scratch_bytes(query_symbols, longest_symbols);
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, in @p order. */
    static void canonical_distances(alignment_costs_t const& costs, banded_overlap_query_t const& query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        banded_overlap_distances<u8_haswell_banded_overlap_kit_t>(costs, query, candidates, order, radius, distances,
                                                                  scratch);
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build leaves out, kept so @c banded_overlap_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct u8_haswell_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx2,avx512f,avx512vl,avx512dq,avx512bw,bmi,bmi2,popcnt"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx2", "avx512f", "avx512vl", "avx512dq", "avx512bw", "bmi", "bmi2", "popcnt")
#endif

/** Intel Skylake AVX-512: eight positions per hashing step for the vote, the scalar band. */
struct u8_skylake_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_skylake;

    /** StringZilla's residues of a candidate's windows at every width, which vote. */
    static constexpr auto canonical_residues = sz_ngrams_residues_skylake;

    /** An overlap index builds by this score, so it keeps no sketch. */
    using sketch_tables_t = std::monostate;

    /** Scratch bytes a round of a @p query_symbols long query and candidates of up to
     *  @p longest_symbols takes: prefix residues, window residues, votes and one band. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_symbols,
                                                         std::size_t longest_symbols) noexcept {
        return banded_overlap_scratch_bytes(query_symbols, longest_symbols);
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, in @p order. */
    static void canonical_distances(alignment_costs_t const& costs, banded_overlap_query_t const& query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        banded_overlap_distances<u8_skylake_banded_overlap_kit_t>(costs, query, candidates, order, radius, distances,
                                                                  scratch);
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build leaves out, kept so @c banded_overlap_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct u8_skylake_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_ICELAKE

/** Intel Ice Lake: Skylake's residues, vote and band. */
struct u8_icelake_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_skylake;

    /** StringZilla's Skylake residues of a candidate's windows at every width, which vote. */
    static constexpr auto canonical_residues = sz_ngrams_residues_skylake;

    /** An overlap index builds by this score, so it keeps no sketch. */
    using sketch_tables_t = std::monostate;

    static constexpr auto canonical_scratch_bytes = u8_skylake_banded_overlap_kit_t::canonical_scratch_bytes;
    static constexpr auto canonical_distances = u8_skylake_banded_overlap_kit_t::canonical_distances;
};
#else

/** The kit this build leaves out, kept so @c banded_overlap_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct u8_icelake_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_ICELAKE

#if STRINGZILLA_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

/** Arm NEON: two positions per hashing step for the vote, the scalar band. */
struct u8_neon_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_neon;

    /** StringZilla's residues of a candidate's windows at every width, which vote. */
    static constexpr auto canonical_residues = sz_ngrams_residues_neon;

    /** An overlap index builds by this score, so it keeps no sketch. */
    using sketch_tables_t = std::monostate;

    /** Scratch bytes a round of a @p query_symbols long query and candidates of up to
     *  @p longest_symbols takes: prefix residues, window residues, votes and one band. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_symbols,
                                                         std::size_t longest_symbols) noexcept {
        return banded_overlap_scratch_bytes(query_symbols, longest_symbols);
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, in @p order. */
    static void canonical_distances(alignment_costs_t const& costs, banded_overlap_query_t const& query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        banded_overlap_distances<u8_neon_banded_overlap_kit_t>(costs, query, candidates, order, radius, distances,
                                                               scratch);
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build leaves out, kept so @c banded_overlap_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct u8_neon_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_NEON

#if STRINGZILLA_TARGET_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/** IBM Power VSX: two positions per hashing step for the vote, the scalar band. */
struct u8_powervsx_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_powervsx_k;
    static constexpr auto canonical_engine_init = sz_ngrams_engine_init_powervsx;

    /** StringZilla's residues of a candidate's windows at every width, which vote. */
    static constexpr auto canonical_residues = sz_ngrams_residues_powervsx;

    /** An overlap index builds by this score, so it keeps no sketch. */
    using sketch_tables_t = std::monostate;

    /** Scratch bytes a round of a @p query_symbols long query and candidates of up to
     *  @p longest_symbols takes: prefix residues, window residues, votes and one band. */
    static constexpr std::size_t canonical_scratch_bytes(std::size_t query_symbols,
                                                         std::size_t longest_symbols) noexcept {
        return banded_overlap_scratch_bytes(query_symbols, longest_symbols);
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, in @p order. */
    static void canonical_distances(alignment_costs_t const& costs, banded_overlap_query_t const& query,
                                    std::span<bytes_view_t const> candidates, std::span<u32_t const> order,
                                    sz_size_t radius, std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        banded_overlap_distances<u8_powervsx_banded_overlap_kit_t>(costs, query, candidates, order, radius, distances,
                                                                   scratch);
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build leaves out, kept so @c banded_overlap_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct u8_powervsx_banded_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_powervsx_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_POWERVSX

/** The widest banded overlap kit this build compiled in among the backends @p caps_ admits, newest
 *  first. */
template <sz_capability_t caps_>
struct banded_overlap_ladder;

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cpus_k) != 0)
struct banded_overlap_ladder<caps_> {
    using kit_t = typename highest_kit<  //
        caps_,                           //
        u8_serial_banded_overlap_kit_t,  //
        u8_icelake_banded_overlap_kit_t, //
        u8_skylake_banded_overlap_kit_t, //
        u8_haswell_banded_overlap_kit_t, //
        u8_neon_banded_overlap_kit_t,    //
        u8_powervsx_banded_overlap_kit_t //
        >::type;
};

#pragma endregion Banded Overlap Kernels

#pragma region Banded Overlap Metric

/**
 *  @brief Shifted overlap over one round of candidates: the cost of aligning the two sequences
 *      where one slides over the other, at the shift most windows vote for.
 *
 *  @c reset files the query's distinct windows at every width into one table, @c push stages
 *  candidate views, and @c drain hashes every staged candidate once, lets its windows vote for a
 *  shift, and aligns the symbols the winning shift lays over each other inside a band under the
 *  alphabet's costs, as @ref banded_overlap_distance lays the result out. Sequences are stored a
 *  class code a byte. Under costs naming complements every query is filed beside its reverse
 *  complement, each in its own table, and a candidate's distance is the lower of the two. One
 *  instance belongs to one thread and owns its engine, the arena the engine lives in, the
 *  staging and the kit's scratch, so a round allocates nothing once a query and a round as long
 *  have been scored.
 *
 *  @tparam kit_type_ The hashing steps of one ISA backend, as @c banded_overlap_ladder picks it.
 *  @tparam allocator_type_ places every buffer, one set per metric.
 *  @tparam cache_hints_type_ prefetches a staged candidate's bytes; a no-op by default.
 */
template <typename kit_type_ = typename banded_overlap_ladder<sz_cap_any_k>::kit_t,
          typename allocator_type_ = std::allocator<byte_t>, typename cache_hints_type_ = fu::standard_cache_hints_t>
class banded_overlap_metric {
  public:
    using kit_t = kit_type_;
    using distance_t = sz_size_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;

    /** The flat form a batched round hands @ref reset for each stored sequence: its bytes. */
    using query_view_t = sz::string_view_t;
    template <typename string_type_>
    static query_view_t query_view(string_type_ const& value) noexcept {
        return query_view_t {value};
    }

  private:

    /** Every prepared image's table of its distinct windows at every width, and nothing before
     *  the first @ref reset. Its blocks live in @c arena_. */
    sz_ngrams_engine_t engine_ {};

    /** Where the engine's blocks bump, rewound whenever @ref reset prepares new queries. */
    growing_arena<allocator_t> arena_;

    /** The round's candidates in push order. */
    growing_buffer<bytes_view_t, allocator_t> staged_;

    /** The round's push slots in ascending length order. */
    growing_buffer<u32_t, allocator_t> order_;

    /** Every query's symbols, back to back in query order, which the band reads at every drain. */
    growing_buffer<byte_t, allocator_t> prepared_;

    /** Every query's reverse complement, back to back in query order. */
    growing_buffer<byte_t, allocator_t> images_;

    /** Every query, then every reverse complement, as the engine files them. */
    growing_buffer<sz_string_view_t, allocator_t> views_;

    /** The kit's prefix hashes, window hashes, votes and band for the round being drained. */
    growing_buffer<byte_t, allocator_t> round_cells_;

    /** The costs the band charges and the complements, read until the metric dies. */
    alignment_costs_t const* costs_ = nullptr;

    /** How the last @ref init, @ref reset or @ref drain fared. */
    status_t last_status_ = status_code_t::uninitialized_k;

  public:
    banded_overlap_metric(allocator_t const& allocator = {}) noexcept
        : arena_(allocator), staged_(allocator), order_(allocator), prepared_(allocator), images_(allocator),
          views_(allocator), round_cells_(allocator) {}
    banded_overlap_metric(banded_overlap_metric const&) = delete;
    banded_overlap_metric& operator=(banded_overlap_metric const&) = delete;
    banded_overlap_metric(banded_overlap_metric&&) noexcept = default;
    banded_overlap_metric& operator=(banded_overlap_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging and the order, and records the costs a round scores with.
     *
     *  @param[in] staging_limit Candidates one round may hold before @c push refuses: at least the
     *      index's round capacity, <tt>max(fan_out * connectivity, connectivity + 1)</tt>, or rounds
     *      saturate rather than fail.
     *
     *  @param[in] costs The class costs and the gap the band charges, with the complements;
     *      read until the metric dies.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p staging_limit is zero or past @c u32_t slots, or @p costs
     *    is not folded.
     */
    status_t init(std::size_t staging_limit, alignment_costs_t const& costs) noexcept {
        if (!staging_limit || staging_limit > (std::numeric_limits<u32_t>::max)() || !costs.classes)
            return status_code_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        if (status_t const ordered = order_.reserve_exactly(staging_limit); failed(ordered)) return ordered;
        costs_ = &costs;
        last_status_ = status_code_t::success_k;
        return last_status_;
    }

    /** How the last @ref init, @ref reset or @ref drain fared - the only place a failure surfaces,
     *  since the scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a round for @p query, filing its windows unless it holds these symbols already, and
     *  discarding whatever the previous round staged. */
    void reset(sz::string_view_t query) noexcept {
        staged_.clear();
        if (!costs_) return;
        // A batch's engine never matches, though its cleared symbols equal an empty query's.
        if (engine_.count == images_per_query_() && succeeded(last_status_) &&
            chars_of(to_span(prepared_)) == to_string_view(query))
            return;
        prepare_(std::span<sz::string_view_t const> {&query, 1});
    }

    /** Opens rounds scoring every query @p queries holds at once, each image in its own table, so
     *  each @ref drain writes @b [queries,candidates]. */
    template <typename query_type_>
        requires std::convertible_to<query_type_ const&, sz::string_view_t>
    void reset(std::span<query_type_ const> queries) noexcept {
        staged_.clear();
        if (!costs_) return;
        prepare_(queries);
    }

    /** Stages one candidate, or refuses once the round is full. */
    basic::metric_staging_t push(sz::string_view_t value) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        prefetch_(value);
        staged_.push_back_reserved(bytes_of(value.data(), value.size()));
        return basic::metric_staging_t::staged_k;
    }

    /**
     *  @brief Scores everything staged and closes the round.
     *
     *  Every distance is exact, but for a pair whose lengths alone put it past @p radius, which
     *  reports the bound those lengths give. A failed round reports the furthest distance, so a
     *  full pool rejects it and an unfilled pool admits it last; @ref status is the only signal.
     *
     *  @param[out] distances @b [queries,candidates] slots, a staged candidate's in push order.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<sz_size_t> distances, sz_size_t radius) noexcept {
        std::size_t const count = staged_.size(), images = images_per_query_();
        std::size_t const queries = engine_.count / images;
        std::size_t longest = 0, longest_query = 0;
        order_.clear();
        for (std::size_t slot = 0; slot != count; ++slot) {
            order_.push_back_reserved(static_cast<u32_t>(slot));
            longest = max_of(longest, staged_[slot].size());
        }
        partial_sort(to_writable_span(order_), order_.size(), [&](u32_t first, u32_t second) noexcept {
            return staged_[first].size() < staged_[second].size();
        });
        for (std::size_t query = 0; query != queries; ++query)
            longest_query = max_of<std::size_t>(longest_query, views_[query].length);
        std::size_t const scratch_bytes = kit_t::canonical_scratch_bytes(longest_query, longest);
        if (succeeded(last_status_) && round_cells_.size() < scratch_bytes) {
            round_cells_.clear();
            last_status_ = round_cells_.reserve(scratch_bytes);
            if (succeeded(last_status_)) round_cells_.extend_uninitialized(round_cells_.capacity());
        }
        bool const scored = succeeded(last_status_) && distances.size() == queries * count;
        if (!scored)
            for (sz_size_t& distance : distances) distance = (std::numeric_limits<sz_size_t>::max)();
        for (std::size_t query = 0; scored && query != queries; ++query) {
            banded_overlap_query_t prepared;
            prepared.count = images;
            for (std::size_t image = 0; image != images; ++image) {
                prepared.tables[image] = sz_ngrams_engine_query(&engine_, image * queries + query);
                prepared.images[image] = bytes_of(views_[image * queries + query].start,
                                                  views_[image * queries + query].length);
            }
            kit_t::canonical_distances(*costs_, prepared, to_span(staged_), to_span(order_), radius,
                                       distances.subspan(query * count, count),
                                       bytes_of(round_cells_.data(), round_cells_.size()));
        }
        staged_.clear();
    }

  private:

    /** Images the engine files per query: the query, and its reverse complement. */
    std::size_t images_per_query_() const noexcept { return costs_ ? complement_images(*costs_) : 1; }

    /** Keeps the symbols of @p queries, writes their reverse complements, and files every image
     *  into the engine, leaving it empty and the status failed on a refusal. */
    template <typename query_type_>
    void prepare_(std::span<query_type_ const> queries) noexcept {
        std::size_t const images = images_per_query_();
        engine_ = sz_ngrams_engine_t {};
        arena_.clear();
        prepared_.clear(), images_.clear(), views_.clear();
        std::size_t bytes = 0;
        for (query_type_ const& query : queries) bytes += sz::string_view_t {query}.size();
        last_status_ = prepared_.reserve(bytes);
        if (succeeded(last_status_) && images != 1) last_status_ = images_.reserve(bytes);
        if (succeeded(last_status_)) last_status_ = views_.reserve(queries.size() * images);
        if (failed(last_status_)) return;
        // Grow-only buffers reserved whole, so the views into them stay put.
        for (query_type_ const& query : queries) {
            sz::string_view_t const view {query};
            byte_t* const kept = prepared_.data() + prepared_.size();
            prepared_.extend_uninitialized(prepared_.size() + view.size());
            if (!view.empty()) std::memcpy(kept, view.data(), view.size());
            views_.push_back_reserved(as_string_view(chars_of(bytes_view_t {kept, view.size()})));
        }
        for (std::size_t query = 0; images != 1 && query != queries.size(); ++query) {
            sz_string_view_t const view = views_[query];
            bytes_span_t const image {images_.data() + images_.size(), view.length};
            images_.extend_uninitialized(images_.size() + view.length);
            reverse_complement(*costs_, bytes_of(view.start, view.length), image);
            views_.push_back_reserved(as_string_view(chars_of(image)));
        }
        sz_strings_view_t views;
        sz_strings_from_string_views(views_.data(), views_.size(), &views);
        sz_allocator_t allocator = engine_allocator(arena_);
        sz_status_t const filed = kit_t::canonical_engine_init(&engine_, &views, banded_overlap_window_widths_k,
                                                               std::size(banded_overlap_window_widths_k), 0, &allocator,
                                                               nullptr);
        if (filed == sz_success_k) return;
        engine_ = sz_ngrams_engine_t {};
        prepared_.clear();
        last_status_ = filed == sz_bad_alloc_k ? status_code_t::out_of_memory_k : status_code_t::shape_mismatch_k;
    }

    /** Starts the misses on every line of @p value, which the drain reads. */
    static void prefetch_(sz::string_view_t value) noexcept {
        for (std::size_t offset = 0; offset < value.size(); offset += fu::default_alignment_k)
            cache_hints_type_ {}(value.data() + offset, fu::prefetch_line_k);
    }
};

using banded_overlap_metric_t = banded_overlap_metric<>;

#pragma endregion Banded Overlap Metric

} // namespace unum::usearch::sequence
