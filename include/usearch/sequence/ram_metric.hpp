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
 *  A kit binds one ISA's engine: the @c init preparing a single query, and the one sweep scoring a
 *  whole round against it - @c distances for Levenshtein over one symbol, @c scores for overlap.
 *  @c levenshtein_ladder and @c overlap_ladder pick the widest kit this build compiled in under a
 *  capability mask, at compile time; @c src/ climbs them once per CPU class and hands each metric
 *  the kit it picked, so nothing tests a capability at run time. Each engine draws its blocks from
 *  the metric's own arena, which @c reset rewinds, so after the widest query and the largest round
 *  seen nothing reaches the metric's allocator again.
 *
 *  @section sequence_ram_metric_levenshtein Levenshtein
 *
 *  Unit-cost Myers: the query's match masks are prepared once, and StringZilla's sweep streams
 *  several candidates through them per step. The round is scored in ascending length order, since
 *  the candidates advanced together run as long as the longest of them, and the sweep stops a
 *  group whose every candidate is already past the radius. Distances count edits over bytes, or
 *  over UTF-8 runes under @c sz_levenshtein_runes_k.
 *
 *  @section sequence_ram_metric_overlap Overlap
 *
 *  Window overlap: the query's windows at every width are hashed into one B-tree once per round, and
 *  every candidate's window hashes are probed against it, four candidates' prefix-hash chains
 *  advancing together. A distance is the share of the longer side's windows the other side misses,
 *  in fixed point, exact within the radius.
 *
 *  @section sequence_ram_metric_needleman_wunsch Needleman-Wunsch
 *
 *  Global alignment with one linear gap cost over packed class codes: an alphabet of at most 32
 *  classes, two bits a code up to four of them and five past, scored through a 32×32 table of
 *  @c u4 costs. The query unpacks once per round; the round is scored in ascending length order,
 *  and a column whose every cell is already past the radius stops its candidate.
 */
#pragma once

#include <cmath>   // `std::nextafter` - the share a radius rounds up to
#include <cstddef> // `std::size_t`
#include <cstring> // `std::memcpy` - the bytes a prepared engine holds, `std::memset` - a group's padding codes

#include <concepts> // `std::convertible_to` - the queries a batched round takes
#include <limits>   // `std::numeric_limits` - the distance a failed round reports
#include <memory>   // `std::allocator`, `std::align` - the Skylake kit's cells on a ZMM boundary
#include <span>     // `std::span`

#include <forkunion/types.hpp> // `fu::standard_cache_hints_t`, `fu::prefetch_line_k` - the candidates' prefetches

#include <stringzilla/levenshtein.h>   // `sz_levenshtein_engine_*` & the per-backend distances
#include <stringzilla/overlap.h>       // `sz_overlap_engine_*` & the per-backend match counts
#include <stringzilla/stringzilla.hpp> // `sz::string_view_t` - every operand a metric takes

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer`, `growing_arena` - the engines' blocks
#include "usearch/utilities/ram_sorts.hpp"   // `partial_sort` - a Needleman-Wunsch round in ascending length order
#include "usearch/basic/ram_index.hpp"       // `basic::metric_staging_t`

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

/** StringZilla's view of @p queries, read in place through an accessor, so a batch is never
 *  copied. */
template <typename query_type_>
sz_strings_view_t as_strings_view(std::span<query_type_ const> queries) noexcept {
    sz_strings_view_t strings;
    strings.layout = sz_strings_accessor_k;
    strings.count = queries.size();
    strings.data.accessor.handle = queries.data();
    strings.data.accessor.get = [](void const* handle, sz_size_t const* indices, sz_size_t count,
                                   sz_string_view_t* views) noexcept {
        query_type_ const* const borrowed = static_cast<query_type_ const*>(handle);
        for (sz_size_t position = 0; position != count; ++position)
            views[position] = as_string_view(sz::string_view_t {borrowed[indices[position]]});
    };
    return strings;
}

#pragma region Levenshtein Kernels

/** Scalar Myers: eight candidates per sweep, each one's recurrence a chain of its own. */
struct u8_serial_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_bytes_k;
    static constexpr auto init = sz_levenshtein_engine_init_serial;
    static constexpr auto distances = sz_levenshtein_distances_serial;
};

/** Scalar Myers over UTF-8 runes, each rune's class from the query's page table. */
struct utf8_serial_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_runes_k;
    static constexpr auto init = sz_levenshtein_engine_init_serial;
    static constexpr auto distances = sz_levenshtein_distances_serial;
};

#if STRINGZILLA_TARGET_HASWELL

/** Intel Haswell AVX2: four candidates per YMM, a SIMD byte transpose. */
struct u8_haswell_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_bytes_k;
    static constexpr auto init = sz_levenshtein_engine_init_haswell;
    static constexpr auto distances = sz_levenshtein_distances_haswell;
};

/** Intel Haswell AVX2 over UTF-8 runes: four candidates per YMM, fed 32-bit rune classes. */
struct utf8_haswell_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_runes_k;
    static constexpr auto init = sz_levenshtein_engine_init_haswell;
    static constexpr auto distances = sz_levenshtein_distances_haswell;
};
#else

/** The kits this build cannot carry, kept so @c levenshtein_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct u8_haswell_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
struct utf8_haswell_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_TARGET_SKYLAKE

/** Intel Skylake-X AVX-512: eight candidates per ZMM, a SIMD byte transpose. */
struct u8_skylake_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_bytes_k;
    static constexpr auto init = sz_levenshtein_engine_init_skylake;
    static constexpr auto distances = sz_levenshtein_distances_skylake;
};

/** Intel Skylake-X AVX-512 over UTF-8 runes: eight candidates per ZMM, fed 32-bit rune classes. */
struct utf8_skylake_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_runes_k;
    static constexpr auto init = sz_levenshtein_engine_init_skylake;
    static constexpr auto distances = sz_levenshtein_distances_skylake;
};
#else

/** The kits this build cannot carry, kept so @c levenshtein_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct u8_skylake_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
struct utf8_skylake_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_ICELAKE

/** Intel Ice Lake AVX-512: the narrowest lanes a query fits, sixty-four candidates per ZMM up to
 *  eight bytes, thirty-two up to sixteen, the Skylake sweep above. */
struct u8_icelake_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_bytes_k;
    static constexpr auto init = sz_levenshtein_engine_init_icelake;
    static constexpr auto distances = sz_levenshtein_distances_icelake;
};

/** Intel Ice Lake over UTF-8 runes: the narrow lanes read bytes alone, so the engine sweeps runes
 *  as Skylake does. */
struct utf8_icelake_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_runes_k;
    static constexpr auto init = sz_levenshtein_engine_init_icelake;
    static constexpr auto distances = sz_levenshtein_distances_icelake;
};
#else

/** The kits this build cannot carry, kept so @c levenshtein_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct u8_icelake_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr bool compiled_k = false;
};
struct utf8_icelake_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_ICELAKE

#if STRINGZILLA_TARGET_NEON

/** Arm NEON: two candidates per register, four registers per position, a SIMD byte transpose. */
struct u8_neon_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_bytes_k;
    static constexpr auto init = sz_levenshtein_engine_init_neon;
    static constexpr auto distances = sz_levenshtein_distances_neon;
};

/** Arm NEON over UTF-8 runes: two candidates per register, four registers per position. */
struct utf8_neon_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = true;
    static constexpr sz_levenshtein_symbol_t symbol_k = sz_levenshtein_runes_k;
    static constexpr auto init = sz_levenshtein_engine_init_neon;
    static constexpr auto distances = sz_levenshtein_distances_neon;
};
#else

/** The kits this build cannot carry, kept so @c levenshtein_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct u8_neon_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = false;
};
struct utf8_neon_levenshtein_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_NEON

/** The widest Levenshtein kit over the symbols @p symbol_ names this build compiled in among the
 *  backends @p caps_ admits, newest first; each kit states its @c capability_k and its
 *  @c compiled_k, so an unselected kit's body never has to exist. */
template <sz_levenshtein_symbol_t symbol_, sz_capability_t caps_>
struct levenshtein_ladder;

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cpus_k) != 0)
struct levenshtein_ladder<sz_levenshtein_bytes_k, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        u8_serial_levenshtein_kit_t,    //
        u8_icelake_levenshtein_kit_t,   //
        u8_skylake_levenshtein_kit_t,   //
        u8_haswell_levenshtein_kit_t,   //
        u8_neon_levenshtein_kit_t       //
        >::type;
    static constexpr sz_capability_t picked_k = kit_t::capability_k;
};

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cpus_k) != 0)
struct levenshtein_ladder<sz_levenshtein_runes_k, caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        utf8_serial_levenshtein_kit_t,  //
        utf8_icelake_levenshtein_kit_t, //
        utf8_skylake_levenshtein_kit_t, //
        utf8_haswell_levenshtein_kit_t, //
        utf8_neon_levenshtein_kit_t     //
        >::type;
    static constexpr sz_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Levenshtein Kernels

#pragma region Overlap Kernels

/** Scalar: one position per hashing step, one key per sort and probe step. */
struct u8_serial_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    static constexpr bool compiled_k = true;
    static constexpr auto init = sz_overlap_engine_init_serial;
    static constexpr auto scores = sz_overlap_scores_serial;
};

#if STRINGZILLA_TARGET_HASWELL

/** Intel Haswell AVX2: four positions per hashing step, eight keys per sort and probe step. */
struct u8_haswell_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = true;
    static constexpr auto init = sz_overlap_engine_init_haswell;
    static constexpr auto scores = sz_overlap_scores_haswell;
};
#else

/** The kit this build cannot carry, kept so @c overlap_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct u8_haswell_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_TARGET_SKYLAKE

/** Intel Skylake AVX-512: eight positions per hashing step, sixteen keys per sort and probe
 *  step. */
struct u8_skylake_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = true;
    static constexpr auto init = sz_overlap_engine_init_skylake;
    static constexpr auto scores = sz_overlap_scores_skylake;
};
#else

/** The kit this build cannot carry, kept so @c overlap_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct u8_skylake_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_NEON

/** Arm NEON: two positions per hashing step, four keys per sort and probe step. */
struct u8_neon_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = true;
    static constexpr auto init = sz_overlap_engine_init_neon;
    static constexpr auto scores = sz_overlap_scores_neon;
};
#else

/** The kit this build cannot carry, kept so @c overlap_ladder stays one list rather than a run of
 *  preprocessor branches. */
struct u8_neon_overlap_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_NEON

/** The widest overlap kit this build compiled in among the backends @p caps_ admits, newest first;
 *  each kit states its @c capability_k and its @c compiled_k, so an unselected kit's body never
 *  has to exist. */
template <sz_capability_t caps_>
struct overlap_ladder;

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cpus_k) != 0)
struct overlap_ladder<caps_> {
    using kit_t = typename highest_kit< //
        caps_,                          //
        u8_serial_overlap_kit_t,        //
        u8_skylake_overlap_kit_t,       //
        u8_haswell_overlap_kit_t,       //
        u8_neon_overlap_kit_t           //
        >::type;
    static constexpr sz_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Overlap Kernels

#pragma region Levenshtein Metric

/**
 *  @brief Unit-cost Levenshtein over one round of candidates: @c reset prepares the query's
 *      engine, @c push stages candidate views, and @c drain streams every staged candidate through
 *      it, which sweeps them in ascending length order, exact up to the radius.
 *
 *  One instance belongs to one thread and owns its engine, the arena the engine lives in, and the
 *  staging; the sweep writes straight into the caller's distances, and nothing reaches the
 *  allocator on the scoring path once @c init sized the round and the arena has held the widest
 *  query and round seen.
 *
 *  @tparam kit_type_ The engine of one ISA backend, as @c levenshtein_ladder picks it; its
 *      @c symbol_k says what one edit replaces: a byte, or a UTF-8 rune.
 *  @tparam allocator_type_ places every buffer, one set per metric.
 *  @tparam cache_hints_type_ prefetches a staged candidate's bytes; ForkUnion's no-op policy by
 *      default.
 */
template <typename kit_type_ = typename levenshtein_ladder<sz_levenshtein_bytes_k, sz_cap_any_k>::kit_t,
          typename allocator_type_ = std::allocator<byte_t>, typename cache_hints_type_ = fu::standard_cache_hints_t>
class levenshtein_metric {
  public:
    using kit_t = kit_type_;
    using distance_t = sz_size_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;

  private:

    /** The query's engine: one query prepared, an empty one as a row of no symbols, and nothing
     *  before the first @ref reset. Its blocks live in @c arena_, and its sweep scratch grows there
     *  to the largest round seen. */
    sz_levenshtein_engine_t engine_ {};

    /** Where the engine's blocks bump, rewound by every @ref reset. */
    growing_arena<allocator_t> arena_;

    /** The round's candidates in push order. */
    growing_buffer<sz_string_view_t, allocator_t> staged_;

    /** The bytes @c engine_ was prepared from, so a round reopening the same query keeps the
     *  engine; grows to the longest query seen. */
    growing_buffer<char, allocator_t> prepared_;
    status_t last_status_ = status_t::uninitialized_k;

  public:
    levenshtein_metric(allocator_t const& allocator = {}) noexcept
        : arena_(allocator), staged_(allocator), prepared_(allocator) {}
    levenshtein_metric(levenshtein_metric const&) = delete;
    levenshtein_metric& operator=(levenshtein_metric const&) = delete;
    levenshtein_metric(levenshtein_metric&&) noexcept = default;
    levenshtein_metric& operator=(levenshtein_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at least the
     *      index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or rounds saturate rather
     *      than fail. A round is bounded to @c u32_t slots by design.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p staging_limit is zero or past @c u32_t slots.
     */
    status_t init(std::size_t staging_limit) noexcept {
        if (!staging_limit || staging_limit > (std::numeric_limits<u32_t>::max)()) return status_t::config_mismatch_k;
        last_status_ = staged_.reserve_exactly(staging_limit);
        return last_status_;
    }

    /** How the last @ref init, @ref reset or @ref drain fared - the only place a failure surfaces,
     *  since the scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a round for @p query, preparing its engine unless it holds these bytes already, and
     *  discarding whatever the previous round staged. */
    void reset(sz::string_view_t query) noexcept {
        staged_.clear();
        // A batch's engine never matches, though its cleared bytes equal an empty query's.
        if (engine_.count == 1 && succeeded(last_status_) && to_string_view(prepared_) == to_string_view(query)) return;
        engine_ = sz_levenshtein_engine_t {};
        arena_.clear();
        // Grow-only, so a thread's arena keeps one block per doubling rather than one per query.
        prepared_.clear();
        last_status_ = prepared_.reserve(query.size());
        if (failed(last_status_)) return;
        prepared_.extend_uninitialized(query.size());
        if (!query.empty()) std::memcpy(prepared_.data(), query.data(), query.size());
        // An empty query prepares as one row of no symbols, scoring every candidate's own length.
        sz_string_view_t const view = as_string_view(query);
        sz_strings_view_t queries;
        sz_strings_from_string_views(&view, 1, &queries);
        prepare_(queries);
    }

    /** Opens rounds scoring every query of @p queries at once, all prepared into one engine, so
     *  each @ref drain writes @b [queries,candidates]; a batch of one is a single query. */
    template <typename query_type_>
        requires std::convertible_to<query_type_ const&, sz::string_view_t>
    void reset(std::span<query_type_ const> queries) noexcept {
        if (queries.size() == 1) return reset(sz::string_view_t {queries.front()});
        staged_.clear();
        engine_ = sz_levenshtein_engine_t {};
        arena_.clear();
        prepared_.clear();
        last_status_ = status_t::success_k;
        prepare_(as_strings_view(queries));
    }

    /** @brief Stages one candidate, or refuses once the round is full. */
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
     *  the radius. The engine sweeps the round in ascending length order and skips each pair whose
     *  lengths alone put it past the radius. A failed round reports the furthest distance, so a
     *  full pool rejects it and an unfilled pool admits it last; @ref status is the only signal.
     *
     *  @param[out] distances One slot per staged candidate, in push order, for every prepared
     *      query, query-major, which the sweep writes in place.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<sz_size_t> distances, sz_size_t radius) noexcept {
        std::size_t const count = staged_.size();
        sz_strings_view_t candidates;
        sz_strings_from_string_views(staged_.data(), count, &candidates);
        // A moved metric keeps its engine, whose growth must bump the arena it moved with.
        engine_.allocator.handle = &arena_;
        bool const scored = succeeded(last_status_) && distances.size() == engine_.count * count &&
                            kit_t::distances(&engine_, &candidates, nullptr, nullptr, radius, distances.data(), count,
                                             nullptr) == sz_success_k;
        if (!scored && succeeded(last_status_)) last_status_ = status_t::bad_alloc_k;
        if (!scored)
            for (sz_size_t& distance : distances) distance = (std::numeric_limits<sz_size_t>::max)();
        staged_.clear();
    }

  private:

    /** Prepares the engine from @p queries, leaving it empty and the status failed on a
     *  refusal. */
    void prepare_(sz_strings_view_t const& queries) noexcept {
        sz_allocator_t allocator = engine_allocator(arena_);
        sz_status_t const prepared = kit_t::init(&engine_, &queries, kit_t::symbol_k, &allocator, nullptr);
        if (prepared == sz_success_k) return;
        engine_ = sz_levenshtein_engine_t {};
        prepared_.clear();
        last_status_ = prepared == sz_bad_alloc_k ? status_t::bad_alloc_k : status_t::shape_mismatch_k;
    }

    /** Starts the misses on every line of @p value, which the drain reads. */
    static void prefetch_(sz::string_view_t value) noexcept {
        for (std::size_t offset = 0; offset < value.size(); offset += fu::default_alignment_k)
            cache_hints_type_ {}(value.data() + offset, fu::prefetch_line_k);
    }
};

using levenshtein_metric_t = levenshtein_metric<>;

#pragma endregion Levenshtein Metric

#pragma region Overlap Metric

/** The full scale of an overlap distance: the widest that still fits @c i64 times a @c u32 window
 *  count. */
inline constexpr sz_size_t overlap_resolution_k = sz_size_t(1) << 31;

/** The share of the @p longer side's windows a pair misses with @p matches found, at full scale
 *  when neither side has a window. */
constexpr u64_t overlap_distance(u64_t longer, u64_t matches) noexcept {
    return longer ? (longer - matches) * static_cast<u64_t>(overlap_resolution_k) / longer
                  : static_cast<u64_t>(overlap_resolution_k);
}

/**
 *  @brief Window overlap over one round of candidates: @c reset hashes the query's windows at every width
 *      into one B-tree, @c push stages candidate views, and @c drain streams every staged candidate through
 *      it, one prefix-hash chain per candidate and one probe per window.
 *
 *  Nothing is stored per node, so the window widths may be chosen per corpus without invalidating
 *  anything. One instance belongs to one thread and owns its engine, the arena the engine lives in,
 *  the staging and the match counts. The staging and the counts are sized once by @c init, and the
 *  engine's tree and chains bump in the arena, so a round allocates nothing once a query and a
 *  round as long have been scored. A distance is the windows the shorter side misses of the longer,
 *  in fixed point over @c overlap_resolution_k: zero where the windows coincide at every width, and
 *  @ref overlap_resolution_k where they share none.
 *
 *  @tparam kit_type_ The engine of one ISA backend, as @c overlap_ladder picks it.
 *  @tparam allocator_type_ places every buffer, one set per metric.
 *  @tparam cache_hints_type_ prefetches a staged candidate's bytes; ForkUnion's no-op policy by
 *      default.
 */
template <typename kit_type_ = typename overlap_ladder<sz_cap_any_k>::kit_t,
          typename allocator_type_ = std::allocator<byte_t>, typename cache_hints_type_ = fu::standard_cache_hints_t>
class overlap_metric {
  public:
    using kit_t = kit_type_;
    using distance_t = sz_size_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;

  private:

    /** The query's engine, one query's tree over its windows at every width, a query without
     *  windows as a tree of none, and nothing before the first @ref reset. Its blocks live in
     *  @c arena_, its chains grow there to the longest seen. */
    sz_overlap_engine_t engine_ {};

    /** Where the engine's blocks bump, rewound whenever @ref reset prepares a new query. */
    growing_arena<allocator_t> arena_;

    /** The round's candidates in push order. */
    growing_buffer<sz_string_view_t, allocator_t> staged_;

    /** The bytes @c engine_ was prepared from, so a round reopening the same query keeps the
     *  engine; grows to the longest query seen. */
    growing_buffer<char, allocator_t> prepared_;

    /** The widths windows are taken at, all sharing one B-tree, borrowed from @ref init's
     *  caller. */
    std::span<std::size_t const> window_widths_;

    /** The round's @b [queries,candidates,widths] match counts, a staged candidate's row by its
     *  push slot. */
    growing_buffer<sz_u32_t, allocator_t> matches_;

    /** How the last @ref init, @ref reset or @ref drain fared. */
    status_t last_status_ = status_t::uninitialized_k;

  public:
    overlap_metric(allocator_t const& allocator = {}) noexcept
        : arena_(allocator), staged_(allocator), prepared_(allocator), matches_(allocator) {}
    overlap_metric(overlap_metric const&) = delete;
    overlap_metric& operator=(overlap_metric const&) = delete;
    overlap_metric(overlap_metric&&) noexcept = default;
    overlap_metric& operator=(overlap_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging and the match counts, and records the window widths a round scores
     *      over.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at least the
     *      index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or rounds saturate rather
     *      than fail.
     *  @param[in] window_widths Window widths in bytes, all sharing one B-tree, borrowed until the
     *      metric dies; a width past a text contributes nothing.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p staging_limit is zero, @p window_widths is empty, or a width is
     *    zero or past @c u32_t bytes.
     */
    status_t init(std::size_t staging_limit, std::span<std::size_t const> window_widths) noexcept {
        if (!staging_limit || window_widths.empty()) return status_t::config_mismatch_k;
        for (std::size_t const width : window_widths)
            if (!width || width > (std::numeric_limits<u32_t>::max)()) return status_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        if (status_t const sized = matches_.resize_uninitialized(staged_.capacity() * window_widths.size());
            failed(sized))
            return sized;
        window_widths_ = window_widths;
        last_status_ = status_t::success_k;
        return last_status_;
    }

    /** How the last @ref init, @ref reset or @ref drain fared - the only place a failure surfaces,
     *  since the scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a round for @p query, building its B-tree unless it holds these bytes already, and
     *  discarding whatever the previous round staged. */
    void reset(sz::string_view_t query) noexcept {
        staged_.clear();
        // A batch's engine never matches, though its cleared bytes equal an empty query's.
        if (engine_.count == 1 && succeeded(last_status_) && to_string_view(prepared_) == to_string_view(query)) return;
        engine_ = sz_overlap_engine_t {};
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

    /** Opens rounds scoring every query of @p queries at once, all hashed into one B-tree, so each
     *  @ref drain writes @b [queries,candidates]; the views are read in place while the tree is
     *  built, and a batch of one is a single query. */
    template <typename query_type_>
        requires std::convertible_to<query_type_ const&, sz::string_view_t>
    void reset(std::span<query_type_ const> queries) noexcept {
        if (queries.size() == 1) return reset(sz::string_view_t {queries.front()});
        staged_.clear();
        engine_ = sz_overlap_engine_t {};
        arena_.clear();
        prepared_.clear();
        last_status_ = matches_.resize_uninitialized(queries.size() * staged_.capacity() * window_widths_.size());
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
     *  the radius. Each pair may miss only as many windows as keep its own distance within
     *  @p radius, so a long candidate never loosens a short one's budget, and a pair past its
     *  budget counts no matches, a full-scale distance. A failed round reports the furthest
     *  distance, so a full pool rejects it and an unfilled pool admits it last; @ref status is the
     *  only signal.
     *
     *  @param[out] distances One slot per staged candidate, in push order, for every prepared
     *      query, query-major.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<sz_size_t> distances, sz_size_t radius) noexcept {
        std::size_t const count = staged_.size(), widths = window_widths_.size();
        sz_f32_t const share = share_of_(radius);
        sz_strings_view_t candidates;
        sz_strings_from_string_views(staged_.data(), count, &candidates);
        // A moved metric keeps its engine, whose growth must bump the arena it moved with.
        engine_.allocator.handle = &arena_;
        bool const scored = succeeded(last_status_) && distances.size() == engine_.count * count &&
                            kit_t::scores(&engine_, &candidates, share, matches_.data(), count * widths, widths,
                                          nullptr) == sz_success_k;
        if (!scored && succeeded(last_status_)) last_status_ = status_t::bad_alloc_k;
        for (std::size_t slot = 0; slot != distances.size(); ++slot)
            distances[slot] = scored ? distance_(engine_.lengths[slot / count], staged_[slot % count].length,
                                                 matches_.data() + slot * widths)
                                     : (std::numeric_limits<sz_size_t>::max)();
        staged_.clear();
    }

  private:

    /** Hashes @p queries into the engine, leaving it empty and the status failed on a refusal. */
    void prepare_(sz_strings_view_t const& queries) noexcept {
        sz_allocator_t allocator = engine_allocator(arena_);
        sz_status_t const prepared = kit_t::init(&engine_, &queries, window_widths_.data(), window_widths_.size(), 0,
                                                 &allocator, nullptr);
        if (prepared == sz_success_k) return;
        engine_ = sz_overlap_engine_t {};
        prepared_.clear();
        last_status_ = prepared == sz_bad_alloc_k ? status_t::bad_alloc_k : status_t::shape_mismatch_k;
    }

    /** Starts the misses on every line of @p value, which the drain reads. */
    static void prefetch_(sz::string_view_t value) noexcept {
        for (std::size_t offset = 0; offset < value.size(); offset += fu::default_alignment_k)
            cache_hints_type_ {}(value.data() + offset, fu::prefetch_line_k);
    }

    /** The share of its longer side's windows a pair within @p radius may miss: every pair within
     *  the radius must count exactly, so the share rounds up past @c (radius + 1) / scale. */
    static sz_f32_t share_of_(sz_size_t radius) noexcept {
        if (radius >= overlap_resolution_k - 1) return 1.0f;
        f64_t const share = static_cast<f64_t>(radius + 1) / static_cast<f64_t>(overlap_resolution_k);
        return std::nextafter(static_cast<sz_f32_t>(share), 1.0f);
    }

    /** One pair's distance off its @p matches: the misses over whichever side has more windows at
     *  every width, summed, scaled to @ref overlap_resolution_k. */
    sz_size_t distance_(std::size_t query_length, std::size_t length, sz_u32_t const* matches) const noexcept {
        u64_t longer = 0, matched = 0;
        for (std::size_t index = 0; index != window_widths_.size(); ++index) {
            std::size_t const width = window_widths_[index];
            u64_t const windows = width <= length ? length - width + 1 : 0;
            u64_t const query_windows = width <= query_length ? query_length - width + 1 : 0;
            longer += max_of(windows, query_windows);
            matched += matches[index];
        }
        // The longer text has more windows at every width, so the widths' longer sides sum to the
        // longer side in all, and a union denominator would let repeats score below zero.
        return static_cast<sz_size_t>(overlap_distance(longer, matched));
    }
};

using overlap_metric_t = overlap_metric<>;

#pragma endregion Overlap Metric

#pragma region Needleman Wunsch Kernels

/** Classes a Needleman-Wunsch alphabet folds into at most, five bits a code. */
inline constexpr std::size_t needleman_wunsch_classes_k = 32;

/** The largest folded cost, the top of a @c u4. */
inline constexpr u32_t needleman_wunsch_cost_ceiling_k = (1u << 4) - 1;

/** Bytes a code of a byte alphabet spans, the side of the square matrix a fold reads. */
inline constexpr std::size_t needleman_wunsch_byte_alphabet_k = std::size_t {1} << 8;

/** Bits one packed code of an alphabet of @p classes takes: two up to four classes, five past. */
constexpr u8_t packed_code_bits(std::size_t classes) noexcept { return classes <= 4 ? 2 : 5; }

/**
 *  @brief A substitution table and a linear gap cost folded to @c u4 over at most 32 classes, and
 *      the bits one packed code takes.
 *
 *  Costs come from similarities: @c c(a,b) = (s(a,a) + s(b,b)) / 2 − s(a,b), zero on the diagonal
 *  and refused where negative. When the largest cost or the gap passes 15, every cost scales by
 *  the one factor bringing it to 15; each rounds to the nearest integer, halves up.
 */
struct needleman_wunsch_costs_t {

    /** Bits one packed code takes: two for up to four classes, five for up to 32, zero unfolded. */
    u8_t code_bits;

    /** Classes the alphabet spans: the codes below this one. */
    u8_t classes;

    /** Cost of one code facing a gap. */
    u8_t gap;

    /** Cost of the row's code aligned to the column's, at most 15. */
    u8_t substitutions[needleman_wunsch_classes_k][needleman_wunsch_classes_k];

    /**
     *  Folds @p similarities, a row-major square over byte codes whose classes end at the last code
     *  scoring anything, and the @p gap cost.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p similarities is not the byte square, or @p gap is negative.
     *  - @c config_mismatch_k No code scores anything, or the codes span past 32 classes.
     *  - @c config_mismatch_k A pair scores above its mean self-score.
     */
    static expected<needleman_wunsch_costs_t> fold(std::span<i8_t const> similarities, i8_t gap) noexcept {
        expected<needleman_wunsch_costs_t> folded {{}, status_t::config_mismatch_k};
        constexpr std::size_t side_k = needleman_wunsch_byte_alphabet_k;
        if (similarities.size() != side_k * side_k || gap < 0) return folded;
        std::size_t classes = 0;
        for (std::size_t cell = 0; cell != similarities.size(); ++cell)
            if (similarities[cell]) classes = max_of(classes, max_of(cell / side_k, cell % side_k) + 1);
        if (!classes || classes > needleman_wunsch_classes_k) return folded;
        // Every cost in halves, so a half-integer mean self-score stays exact.
        auto const doubled = [&](std::size_t row, std::size_t column) noexcept {
            return i32_t {similarities[row * side_k + row]} + similarities[column * side_k + column] -
                   2 * i32_t {similarities[row * side_k + column]};
        };
        i32_t largest = 2 * static_cast<i32_t>(needleman_wunsch_cost_ceiling_k);
        largest = max_of(largest, 2 * i32_t {gap});
        for (std::size_t row = 0; row != classes; ++row)
            for (std::size_t column = 0; column != classes; ++column) {
                if (doubled(row, column) < 0) return folded;
                largest = max_of(largest, doubled(row, column));
            }
        // `halves / 2` scaled by `30 / largest`, rounded half up, unscaled while `largest` is 30.
        auto const rounded = [&](i32_t halves) noexcept {
            i32_t const ceiling = static_cast<i32_t>(needleman_wunsch_cost_ceiling_k);
            return static_cast<u8_t>((2 * ceiling * halves + largest) / (2 * largest));
        };
        needleman_wunsch_costs_t& costs = folded.value;
        costs.code_bits = packed_code_bits(classes);
        costs.classes = static_cast<u8_t>(classes);
        costs.gap = rounded(2 * i32_t {gap});
        for (std::size_t row = 0; row != classes; ++row)
            for (std::size_t column = 0; column != classes; ++column)
                costs.substitutions[row][column] = rounded(doubled(row, column));
        folded.status = status_t::success_k;
        return folded;
    }
};

/** Codes the packed @p bytes long @p sequence holds: as many as its bytes past the first fit at
 *  @p code_bits apiece, less the unused tail slots that first byte counts. */
constexpr std::size_t packed_codes_count(byte_t const* sequence, std::size_t bytes, u32_t code_bits) noexcept {
    if (!bytes) return 0;
    std::size_t const slots = (bytes - 1) * 8 / code_bits, unused = std::to_integer<std::size_t>(sequence[0]);
    return slots > unused ? slots - unused : 0;
}

/** The code at @p index of the packed @p sequence, little-endian from the byte past its first; a
 *  code reads its second byte only when it crosses into one. */
constexpr u32_t packed_code_at(byte_t const* sequence, std::size_t index, u32_t code_bits) noexcept {
    std::size_t const bit = index * code_bits, byte = 1 + (bit >> 3);
    u32_t const shift = bit & 7u;
    u32_t word = std::to_integer<u32_t>(sequence[byte]);
    if (shift + code_bits > 8) word |= std::to_integer<u32_t>(sequence[byte + 1]) << 8;
    return (word >> shift) & ((1u << code_bits) - 1u);
}

/** Bytes @p codes take packed at @p code_bits apiece, the leading count of unused slots
 *  included. */
constexpr std::size_t packed_codes_bytes(std::size_t codes, u32_t code_bits) noexcept {
    return 1 + divide_round_up<std::size_t>(codes * code_bits, 8);
}

/** Packs @p codes at @p code_bits apiece into @p sequence, @ref packed_codes_bytes long, behind the
 *  count of slots its last bytes leave unused. */
inline void pack_codes(std::span<u8_t const> codes, u32_t code_bits, bytes_span_t sequence) noexcept {
    std::memset(sequence.data(), 0, sequence.size());
    sequence[0] = static_cast<byte_t>((sequence.size() - 1) * 8 / code_bits - codes.size());
    for (std::size_t index = 0; index != codes.size(); ++index) {
        std::size_t const bit = index * code_bits, byte = 1 + (bit >> 3);
        u32_t const shifted = u32_t {codes[index]} << (bit & 7u);
        sequence[byte] |= static_cast<byte_t>(shifted & 0xFFu);
        if (shifted > 0xFFu) sequence[byte + 1] |= static_cast<byte_t>(shifted >> 8);
    }
}

/** One candidate's cost from @p query down its columns, @p column holding a cell per query code and
 *  one more: exact up to @p radius, and past it the first column minimum above it. */
inline sz_size_t needleman_wunsch_serial(needleman_wunsch_costs_t const& costs, std::span<u8_t const> query,
                                         bytes_view_t candidate, sz_size_t radius, u32_t* column) noexcept {
    std::size_t const rows = query.size();
    std::size_t const columns = packed_codes_count(candidate.data(), candidate.size(), costs.code_bits);
    u32_t const gap = costs.gap;
    for (std::size_t row = 0; row <= rows; ++row) column[row] = static_cast<u32_t>(row * gap);
    for (std::size_t index = 0; index != columns; ++index) {
        u32_t const code = packed_code_at(candidate.data(), index, costs.code_bits);
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
struct packed_serial_needleman_wunsch_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
    static constexpr bool compiled_k = true;

    /** Scratch bytes a round of a @p query_codes long query and candidates of up to
     *  @p longest_codes takes. */
    static constexpr std::size_t scratch_bytes(std::size_t query_codes, std::size_t) noexcept {
        return (query_codes + 1) * sizeof(u32_t);
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, in @p order. */
    static void distances(needleman_wunsch_costs_t const& costs, std::span<u8_t const> query,
                          std::span<bytes_view_t const> candidates, std::span<u32_t const> order, sz_size_t radius,
                          std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
        u32_t* const column = elements_of<u32_t>(scratch).data();
        for (u32_t const slot : order)
            distances[slot] = needleman_wunsch_serial(costs, query, candidates[slot], radius, column);
    }
};

#if STRINGZILLA_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw")
#endif

/**
 *  @brief Intel Skylake-X AVX-512: 32 candidates per ZMM, one @c u16 cell each, a column at a time.
 *
 *  A group's codes are transposed once, so every column holds one code per lane, and each column
 *  looks its codes up once per class the query holds with one @c VPERMW, leaving one load per cell.
 *  Cells saturate at 65535, which bounds every true cost above them, so a saturated cost is
 *  rescored by the scalar kernel whenever the radius reaches it. Candidates past 65535 codes take
 *  the scalar kernel outright.
 */
struct packed_skylake_needleman_wunsch_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = true;

    /** Candidates one group sweeps, the @c u16 lanes of a ZMM. */
    static constexpr std::size_t lanes_k = sizeof(__m512i) / sizeof(u16_t);

    /** Scratch bytes a round of a @p query_codes long query and candidates of up to
     *  @p longest_codes takes: a ZMM per query code and one more, a ZMM of codes per candidate
     *  code, and the slack aligning them. */
    static constexpr std::size_t scratch_bytes(std::size_t query_codes, std::size_t longest_codes) noexcept {
        return (query_codes + 1 + longest_codes + 1) * sizeof(__m512i);
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, @p order taken in
     *  groups of @ref lanes_k. */
    static void distances(needleman_wunsch_costs_t const& costs, std::span<u8_t const> query,
                          std::span<bytes_view_t const> candidates, std::span<u32_t const> order, sz_size_t radius,
                          std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
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
        __m512i rows_u16x32[needleman_wunsch_classes_k], profile_u16x32[needleman_wunsch_classes_k];
        for (u32_t code = 0; code != query_classes; ++code)
            rows_u16x32[code] = _mm512_cvtepu8_epi16(
                _mm256_loadu_si256(reinterpret_cast<__m256i const*>(costs.substitutions[code])));
        __m512i const gap_u16x32 = _mm512_set1_epi16(static_cast<short>(gap));
        __m512i const radius_u16x32 = _mm512_set1_epi16(static_cast<short>(min_of<sz_size_t>(radius, saturated_k)));

        for (std::size_t first = 0; first < order.size(); first += lanes_k) {
            std::size_t const lanes = min_of(lanes_k, order.size() - first);
            u16_t lengths[lanes_k] = {};
            __mmask32 scalar = 0;
            std::size_t longest = 0;
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                bytes_view_t const candidate = candidates[order[first + lane]];
                std::size_t const length = packed_codes_count(candidate.data(), candidate.size(), costs.code_bits);
                if (length > saturated_k) scalar |= __mmask32 {1} << lane;
                else lengths[lane] = static_cast<u16_t>(length), longest = max_of(longest, length);
            }
            std::memset(codes, 0, longest * sizeof(__m512i));
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                bytes_view_t const candidate = candidates[order[first + lane]];
                for (std::size_t index = 0; index != lengths[lane]; ++index)
                    codes[index * lanes_k + lane] = static_cast<u16_t>(
                        packed_code_at(candidate.data(), index, costs.code_bits));
            }

            for (std::size_t row = 0; row <= rows; ++row)
                column[row] = _mm512_set1_epi16(static_cast<short>(min_of<std::size_t>(row * gap, saturated_k)));
            __m512i const lengths_u16x32 = _mm512_loadu_si512(lengths);
            __mmask32 const empty_mask = _mm512_cmpeq_epi16_mask(lengths_u16x32, _mm512_setzero_si512());
            __m512i results_u16x32 = column[rows];
            // Padding lanes and the scalar ones count as finished from the start.
            __mmask32 finished_mask = empty_mask | scalar | (lanes == lanes_k ? 0 : every_lane_k << lanes);
            for (std::size_t index = 0; index != longest && finished_mask != every_lane_k; ++index) {
                __m512i const codes_u16x32 = _mm512_loadu_si512(codes + index * lanes_k);
                for (u32_t code = 0; code != query_classes; ++code)
                    profile_u16x32[code] = _mm512_permutexvar_epi16(codes_u16x32, rows_u16x32[code]);
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

            u16_t results[lanes_k];
            _mm512_storeu_si512(results, results_u16x32);
            u32_t* const scalar_column = static_cast<u32_t*>(aligned);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                u32_t const slot = order[first + lane];
                bool const rescored = (scalar >> lane & 1u) || (results[lane] == saturated_k && radius >= saturated_k);
                distances[slot] = rescored
                                      ? needleman_wunsch_serial(costs, query, candidates[slot], radius, scalar_column)
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

/** The kit this build cannot carry, kept so @c needleman_wunsch_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct packed_skylake_needleman_wunsch_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

/**
 *  @brief Intel Haswell AVX2: 16 candidates per YMM, one @c u16 cell each, a column at a time.
 *
 *  The Skylake sweep with byte codes: each column looks its codes up once per class the query holds
 *  with two @c VPSHUFB over the low and high halves of that class's row, blended on the code's
 *  fifth bit and widened to the lanes. AVX2 has no unsigned 16-bit compare, so a column is past the
 *  radius when clamping its minimum to the radius changes it.
 */
struct packed_haswell_needleman_wunsch_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr bool compiled_k = true;

    /** Candidates one group sweeps, the @c u16 lanes of a YMM. */
    static constexpr std::size_t lanes_k = sizeof(__m256i) / sizeof(u16_t);

    /** Scratch bytes a round of a @p query_codes long query and candidates of up to
     *  @p longest_codes takes: a YMM per query code and one more, a byte per lane per candidate
     *  code, and the slack aligning them. */
    static constexpr std::size_t scratch_bytes(std::size_t query_codes, std::size_t longest_codes) noexcept {
        return (query_codes + 2) * sizeof(__m256i) + longest_codes * lanes_k;
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, @p order taken in
     *  groups of @ref lanes_k. */
    static void distances(needleman_wunsch_costs_t const& costs, std::span<u8_t const> query,
                          std::span<bytes_view_t const> candidates, std::span<u32_t const> order, sz_size_t radius,
                          std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
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
        __m128i low_rows_u8x16[needleman_wunsch_classes_k], high_rows_u8x16[needleman_wunsch_classes_k];
        __m256i profile_u16x16[needleman_wunsch_classes_k];
        for (u32_t code = 0; code != query_classes; ++code) {
            low_rows_u8x16[code] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(costs.substitutions[code]));
            high_rows_u8x16[code] = _mm_loadu_si128(reinterpret_cast<__m128i const*>(costs.substitutions[code] + 16));
        }
        __m128i const last_low_code_u8x16 = _mm_set1_epi8(15);
        __m256i const gap_u16x16 = _mm256_set1_epi16(static_cast<short>(gap));
        __m256i const radius_u16x16 = _mm256_set1_epi16(static_cast<short>(min_of<sz_size_t>(radius, saturated_k)));

        for (std::size_t first = 0; first < order.size(); first += lanes_k) {
            std::size_t const lanes = min_of(lanes_k, order.size() - first);
            u16_t lengths[lanes_k] = {};
            u16_t finished[lanes_k] = {};
            u32_t scalar = 0;
            std::size_t longest = 0;
            for (std::size_t lane = 0; lane != lanes_k; ++lane) {
                if (lane >= lanes) {
                    finished[lane] = static_cast<u16_t>(saturated_k);
                    continue;
                }
                bytes_view_t const candidate = candidates[order[first + lane]];
                std::size_t const length = packed_codes_count(candidate.data(), candidate.size(), costs.code_bits);
                if (length > saturated_k) scalar |= u32_t {1} << lane;
                else lengths[lane] = static_cast<u16_t>(length), longest = max_of(longest, length);
                // Padding lanes, empty ones and scalar ones count as finished from the start.
                if (!lengths[lane]) finished[lane] = static_cast<u16_t>(saturated_k);
            }
            std::memset(codes, 0, longest * lanes_k);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                bytes_view_t const candidate = candidates[order[first + lane]];
                for (std::size_t index = 0; index != lengths[lane]; ++index)
                    codes[index * lanes_k + lane] = static_cast<u8_t>(
                        packed_code_at(candidate.data(), index, costs.code_bits));
            }

            for (std::size_t row = 0; row <= rows; ++row)
                column[row] = _mm256_set1_epi16(static_cast<short>(min_of<std::size_t>(row * gap, saturated_k)));
            __m256i const lengths_u16x16 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(lengths));
            __m256i finished_u16x16 = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(finished));
            __m256i results_u16x16 = column[rows];
            for (std::size_t index = 0;
                 index != longest && static_cast<u32_t>(_mm256_movemask_epi8(finished_u16x16)) != every_lane_k;
                 ++index) {
                __m128i const codes_u8x16 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(codes + index * lanes_k));
                __m128i const high_codes_u8x16 = _mm_cmpgt_epi8(codes_u8x16, last_low_code_u8x16);
                for (u32_t code = 0; code != query_classes; ++code)
                    profile_u16x16[code] = _mm256_cvtepu8_epi16(
                        _mm_blendv_epi8(_mm_shuffle_epi8(low_rows_u8x16[code], codes_u8x16),
                                        _mm_shuffle_epi8(high_rows_u8x16[code], codes_u8x16), high_codes_u8x16));
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

            u16_t results[lanes_k];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(results), results_u16x16);
            u32_t* const scalar_column = static_cast<u32_t*>(aligned);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                u32_t const slot = order[first + lane];
                bool const rescored = (scalar >> lane & 1u) || (results[lane] == saturated_k && radius >= saturated_k);
                distances[slot] = rescored
                                      ? needleman_wunsch_serial(costs, query, candidates[slot], radius, scalar_column)
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

/** The kit this build cannot carry, kept so @c needleman_wunsch_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct packed_haswell_needleman_wunsch_kit_t {
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

/**
 *  @brief Arm NEON: 8 candidates per register, one @c u16 cell each, a column at a time.
 *
 *  The Skylake sweep with byte codes: each column looks its codes up once per class the query holds
 *  with one @c TBL over that class's 32-byte row, widened to the lanes.
 */
struct packed_neon_needleman_wunsch_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = true;

    /** Candidates one group sweeps, the @c u16 lanes of a Q register. */
    static constexpr std::size_t lanes_k = sizeof(uint16x8_t) / sizeof(u16_t);

    /** Scratch bytes a round takes, @p query_codes the query's and @p longest_codes the candidates'
     *  at most: a register per query code and one more, and a byte per lane per candidate code. */
    static constexpr std::size_t scratch_bytes(std::size_t query_codes, std::size_t longest_codes) noexcept {
        return (query_codes + 2) * sizeof(uint16x8_t) + longest_codes * lanes_k;
    }

    /** Scores every candidate of @p candidates into its slot of @p distances, @p order taken in
     *  groups of @ref lanes_k. */
    static void distances(needleman_wunsch_costs_t const& costs, std::span<u8_t const> query,
                          std::span<bytes_view_t const> candidates, std::span<u32_t const> order, sz_size_t radius,
                          std::span<sz_size_t> distances, bytes_span_t scratch) noexcept {
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
        uint8x16x2_t rows_u8x16x2[needleman_wunsch_classes_k];
        uint16x8_t profile_u16x8[needleman_wunsch_classes_k];
        for (u32_t code = 0; code != query_classes; ++code) rows_u8x16x2[code] = vld1q_u8_x2(costs.substitutions[code]);
        uint16x8_t const gap_u16x8 = vdupq_n_u16(static_cast<u16_t>(gap));
        uint16x8_t const radius_u16x8 = vdupq_n_u16(static_cast<u16_t>(min_of<sz_size_t>(radius, saturated_k)));

        for (std::size_t first = 0; first < order.size(); first += lanes_k) {
            std::size_t const lanes = min_of(lanes_k, order.size() - first);
            u16_t lengths[lanes_k] = {};
            u16_t finished[lanes_k] = {};
            u32_t scalar = 0;
            std::size_t longest = 0;
            for (std::size_t lane = 0; lane != lanes_k; ++lane) {
                if (lane >= lanes) {
                    finished[lane] = static_cast<u16_t>(saturated_k);
                    continue;
                }
                bytes_view_t const candidate = candidates[order[first + lane]];
                std::size_t const length = packed_codes_count(candidate.data(), candidate.size(), costs.code_bits);
                if (length > saturated_k) scalar |= u32_t {1} << lane;
                else lengths[lane] = static_cast<u16_t>(length), longest = max_of(longest, length);
                // Padding lanes, empty ones and scalar ones count as finished from the start.
                if (!lengths[lane]) finished[lane] = static_cast<u16_t>(saturated_k);
            }
            std::memset(codes, 0, longest * lanes_k);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                bytes_view_t const candidate = candidates[order[first + lane]];
                for (std::size_t index = 0; index != lengths[lane]; ++index)
                    codes[index * lanes_k + lane] = static_cast<u8_t>(
                        packed_code_at(candidate.data(), index, costs.code_bits));
            }

            for (std::size_t row = 0; row <= rows; ++row)
                column[row] = vdupq_n_u16(static_cast<u16_t>(min_of<std::size_t>(row * gap, saturated_k)));
            uint16x8_t const lengths_u16x8 = vld1q_u16(lengths);
            uint16x8_t finished_u16x8 = vld1q_u16(finished);
            uint16x8_t results_u16x8 = column[rows];
            for (std::size_t index = 0; index != longest && vminvq_u16(finished_u16x8) != saturated_k; ++index) {
                uint8x8_t const codes_u8x8 = vld1_u8(codes + index * lanes_k);
                for (u32_t code = 0; code != query_classes; ++code)
                    profile_u16x8[code] = vmovl_u8(vqtbl2_u8(rows_u8x16x2[code], codes_u8x8));
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

            u16_t results[lanes_k];
            vst1q_u16(results, results_u16x8);
            u32_t* const scalar_column = static_cast<u32_t*>(aligned);
            for (std::size_t lane = 0; lane != lanes; ++lane) {
                u32_t const slot = order[first + lane];
                bool const rescored = (scalar >> lane & 1u) || (results[lane] == saturated_k && radius >= saturated_k);
                distances[slot] = rescored
                                      ? needleman_wunsch_serial(costs, query, candidates[slot], radius, scalar_column)
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

/** The kit this build cannot carry, kept so @c needleman_wunsch_ladder stays one list rather than a
 *  run of preprocessor branches. */
struct packed_neon_needleman_wunsch_kit_t {
    static constexpr sz_capability_t capability_k = sz_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // STRINGZILLA_TARGET_NEON

/** The widest Needleman-Wunsch kit this build compiled in among the backends @p caps_ admits,
 *  newest first. */
template <sz_capability_t caps_>
struct needleman_wunsch_ladder;

template <sz_capability_t caps_>
    requires((caps_ & sz_cap_cpus_k) != 0)
struct needleman_wunsch_ladder<caps_> {
    using kit_t = typename highest_kit<        //
        caps_,                                 //
        packed_serial_needleman_wunsch_kit_t,  //
        packed_skylake_needleman_wunsch_kit_t, //
        packed_haswell_needleman_wunsch_kit_t, //
        packed_neon_needleman_wunsch_kit_t     //
        >::type;
    static constexpr sz_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Needleman Wunsch Kernels

#pragma region Needleman Wunsch Metric

/**
 *  @brief Needleman-Wunsch cost from the query to each staged candidate over packed class codes: a
 *      32×32 table of @c u4 costs and one linear gap cost, exact up to the radius.
 *
 *  @c reset unpacks the query's codes, @c push stages candidate views, and @c drain scores the
 *  round in ascending length order. One instance belongs to one thread and owns the staging, the
 *  order and the kit's scratch, which grows to the longest query and candidate seen; the costs are
 *  borrowed from @ref init's caller.
 *
 *  @tparam kit_type_ The kernel of one ISA backend, as @c needleman_wunsch_ladder picks it.
 *  @tparam allocator_type_ places every buffer, one set per metric.
 *  @tparam cache_hints_type_ prefetches a staged candidate's bytes; ForkUnion's no-op policy by
 *      default.
 */
template <typename kit_type_ = typename needleman_wunsch_ladder<sz_cap_any_k>::kit_t,
          typename allocator_type_ = std::allocator<byte_t>, typename cache_hints_type_ = fu::standard_cache_hints_t>
class needleman_wunsch_metric {
  public:
    using kit_t = kit_type_;
    using distance_t = sz_size_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;

  private:

    /** The folded table and gap cost, borrowed until the metric dies. */
    needleman_wunsch_costs_t const* costs_ = nullptr;

    /** The query's codes, one byte apiece. */
    growing_buffer<u8_t, allocator_t> query_;

    /** The round's candidates in push order. */
    growing_buffer<bytes_view_t, allocator_t> staged_;

    /** The round's push slots in ascending length order. */
    growing_buffer<u32_t, allocator_t> order_;

    /** The kit's cells and transposed codes. */
    growing_buffer<byte_t, allocator_t> scratch_;

    /** How the last @ref init, @ref reset or @ref drain fared. */
    status_t last_status_ = status_t::uninitialized_k;

  public:
    needleman_wunsch_metric(allocator_t const& allocator = {}) noexcept
        : query_(allocator), staged_(allocator), order_(allocator), scratch_(allocator) {}
    needleman_wunsch_metric(needleman_wunsch_metric const&) = delete;
    needleman_wunsch_metric& operator=(needleman_wunsch_metric const&) = delete;
    needleman_wunsch_metric(needleman_wunsch_metric&&) noexcept = default;
    needleman_wunsch_metric& operator=(needleman_wunsch_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging and the order, and records the costs a round scores with.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at
     *      least the index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or
     *      rounds saturate rather than fail. A round is bounded to @c u32_t slots by design.
     *  @param[in] costs A folded table, borrowed until the metric dies.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p staging_limit is zero or past @c u32_t slots, or @p costs is not
     *    folded.
     */
    status_t init(std::size_t staging_limit, needleman_wunsch_costs_t const& costs) noexcept {
        if (!staging_limit || staging_limit > (std::numeric_limits<u32_t>::max)() || !costs.code_bits)
            return status_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        if (status_t const ordered = order_.reserve_exactly(staging_limit); failed(ordered)) return ordered;
        costs_ = &costs;
        last_status_ = status_t::success_k;
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
        bytes_view_t const packed = bytes_of(query.data(), query.size());
        std::size_t const codes = packed_codes_count(packed.data(), packed.size(), costs_->code_bits);
        query_.clear();
        last_status_ = query_.reserve(codes);
        if (failed(last_status_)) return;
        for (std::size_t index = 0; index != codes; ++index)
            query_.push_back_reserved(static_cast<u8_t>(packed_code_at(packed.data(), index, costs_->code_bits)));
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
     *  @param[out] distances One slot per staged candidate, in push order.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<sz_size_t> distances, sz_size_t radius) noexcept {
        std::size_t const count = staged_.size();
        std::size_t longest = 0;
        order_.clear();
        for (std::size_t slot = 0; slot != count; ++slot) {
            order_.push_back_reserved(static_cast<u32_t>(slot));
            longest = max_of(longest, length_(slot));
        }
        partial_sort(to_writable_span(order_), order_.size(),
                     [&](u32_t first, u32_t second) noexcept { return length_(first) < length_(second); });
        std::size_t const scratch_bytes = kit_t::scratch_bytes(query_.size(), longest);
        if (succeeded(last_status_) && scratch_.size() < scratch_bytes) {
            scratch_.clear();
            last_status_ = scratch_.reserve(scratch_bytes);
            if (succeeded(last_status_)) scratch_.extend_uninitialized(scratch_.capacity());
        }
        bool const scored = succeeded(last_status_) && distances.size() == count;
        if (scored)
            kit_t::distances(*costs_, to_span(query_), to_span(staged_), to_span(order_), radius, distances,
                             bytes_of(scratch_.data(), scratch_.size()));
        else
            for (sz_size_t& distance : distances) distance = (std::numeric_limits<sz_size_t>::max)();
        staged_.clear();
    }

  private:

    /** Codes the candidate staged at @p slot holds. */
    std::size_t length_(std::size_t slot) const noexcept {
        return packed_codes_count(staged_[slot].data(), staged_[slot].size(), costs_->code_bits);
    }

    /** Starts the misses on every line of @p value, which the drain reads. */
    static void prefetch_(sz::string_view_t value) noexcept {
        for (std::size_t offset = 0; offset < value.size(); offset += fu::default_alignment_k)
            cache_hints_type_ {}(value.data() + offset, fu::prefetch_line_k);
    }
};

using needleman_wunsch_metric_t = needleman_wunsch_metric<>;

#pragma endregion Needleman Wunsch Metric

} // namespace unum::usearch::sequence
