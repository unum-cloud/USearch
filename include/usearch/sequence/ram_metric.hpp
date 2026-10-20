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
 */
#pragma once

#include <cmath>   // `std::nextafter` - the share a radius rounds up to
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int64_t`
#include <cstring> // `std::memcpy` - the bytes a prepared engine holds

#include <concepts> // `std::convertible_to` - the queries a batched round takes
#include <limits>   // `std::numeric_limits` - the distance a failed round reports
#include <memory>   // `std::allocator`
#include <span>     // `std::span`

#include <forkunion/types.hpp> // `fu::standard_cache_hints_t`, `fu::prefetch_line_k` - the candidates' prefetches

#include <stringzilla/levenshtein.h>   // `sz_levenshtein_engine_*` & the per-backend distances
#include <stringzilla/overlap.h>       // `sz_overlap_engine_*` & the per-backend match counts
#include <stringzilla/stringzilla.hpp> // `sz::string_view_t` - every operand a metric takes

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp" // `growing_buffer`, `growing_arena` - the engines' blocks
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
 *  One instance belongs to one thread and owns its engine, the arena the engine lives in, the
 *  staging, and the sweep's distances; nothing reaches the allocator on the scoring path once
 *  @c init sized the round and the arena has held the widest query and round seen.
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

    /** The views a batched @ref reset prepares the engine from. */
    growing_buffer<sz_string_view_t, allocator_t> queries_;

    /** The sweep's @b [queries,candidates] distances, a query's row as wide as the staging. */
    growing_buffer<sz_size_t, allocator_t> swept_;
    status_t last_status_ = status_t::uninitialized_k;

  public:
    levenshtein_metric(allocator_t const& allocator = {}) noexcept
        : arena_(allocator), staged_(allocator), prepared_(allocator), queries_(allocator), swept_(allocator) {}
    levenshtein_metric(levenshtein_metric const&) = delete;
    levenshtein_metric& operator=(levenshtein_metric const&) = delete;
    levenshtein_metric(levenshtein_metric&&) noexcept = default;
    levenshtein_metric& operator=(levenshtein_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging and the sweep's distances.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at least the
     *      index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or rounds saturate rather
     *      than fail. A round is bounded to @c u32_t slots by design.
     */
    status_t init(std::size_t staging_limit) noexcept {
        if (!staging_limit || staging_limit > (std::numeric_limits<u32_t>::max)()) return status_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        last_status_ = swept_.resize_uninitialized(staged_.capacity());
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
        prepare_(std::span<sz_string_view_t const>(&view, 1));
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
        queries_.clear();
        last_status_ = queries_.reserve(queries.size());
        if (succeeded(last_status_)) last_status_ = swept_.resize_uninitialized(queries.size() * staged_.capacity());
        if (failed(last_status_)) return;
        for (query_type_ const& query : queries) queries_.push_back_reserved(as_string_view(query));
        prepare_(to_span(queries_));
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
     *      query, query-major.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<i64_t> distances, i64_t radius) noexcept {
        std::size_t const count = staged_.size();
        sz_strings_view_t candidates;
        sz_strings_from_string_views(staged_.data(), count, &candidates);
        sz_size_t const bound = radius < (std::numeric_limits<i64_t>::max)()
                                    ? static_cast<sz_size_t>(max_of<i64_t>(radius, 0))
                                    : STRINGZILLA_SIZE_MAX;
        // A moved metric keeps its engine, whose growth must bump the arena it moved with.
        engine_.allocator.handle = &arena_;
        bool const scored = succeeded(last_status_) && kit_t::distances(&engine_, &candidates, nullptr, nullptr, bound,
                                                                        swept_.data(), count, nullptr) == sz_success_k;
        if (!scored && succeeded(last_status_)) last_status_ = status_t::bad_alloc_k;
        for (std::size_t slot = 0; slot != distances.size(); ++slot)
            distances[slot] = scored ? static_cast<i64_t>(swept_[slot]) : (std::numeric_limits<i64_t>::max)();
        staged_.clear();
    }

  private:

    /** Prepares the engine from @p views, leaving it empty and the status failed on a refusal. */
    void prepare_(std::span<sz_string_view_t const> views) noexcept {
        sz_strings_view_t queries;
        sz_strings_from_string_views(views.data(), views.size(), &queries);
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
inline constexpr i64_t overlap_resolution_k = i64_t(1) << 31;

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
 *  in fixed point over @ref overlap_resolution_k: zero where the windows coincide at every width,
 *  and @ref overlap_resolution_k where they share none.
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

    /** The queries @c engine_ holds, whose lengths every distance reads: one view of
     *  @c prepared_, or a batched @ref reset's views. */
    growing_buffer<sz_string_view_t, allocator_t> queries_;

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
        : arena_(allocator), staged_(allocator), prepared_(allocator), queries_(allocator), matches_(allocator) {}
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
        queries_.clear();
        last_status_ = prepared_.reserve(query.size());
        if (succeeded(last_status_)) last_status_ = queries_.reserve(1);
        if (failed(last_status_)) return;
        prepared_.extend_uninitialized(query.size());
        if (!query.empty()) std::memcpy(prepared_.data(), query.data(), query.size());
        queries_.push_back_reserved(as_string_view(to_string_view(prepared_)));
        prepare_();
    }

    /** Opens rounds scoring every query of @p queries at once, all hashed into one B-tree, so each
     *  @ref drain writes @b [queries,candidates]; the views stay borrowed until the next reset, and
     *  a batch of one is a single query. */
    template <typename query_type_>
        requires std::convertible_to<query_type_ const&, sz::string_view_t>
    void reset(std::span<query_type_ const> queries) noexcept {
        if (queries.size() == 1) return reset(sz::string_view_t {queries.front()});
        staged_.clear();
        engine_ = sz_overlap_engine_t {};
        arena_.clear();
        prepared_.clear();
        queries_.clear();
        last_status_ = queries_.reserve(queries.size());
        if (succeeded(last_status_))
            last_status_ = matches_.resize_uninitialized(queries.size() * staged_.capacity() * window_widths_.size());
        if (failed(last_status_)) return;
        for (query_type_ const& query : queries) queries_.push_back_reserved(as_string_view(query));
        prepare_();
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
    void drain(std::span<i64_t> distances, i64_t radius) noexcept {
        std::size_t const count = staged_.size(), widths = window_widths_.size();
        sz_f32_t const share = share_of_(radius);
        sz_strings_view_t candidates;
        sz_strings_from_string_views(staged_.data(), count, &candidates);
        // A moved metric keeps its engine, whose growth must bump the arena it moved with.
        engine_.allocator.handle = &arena_;
        bool const scored = succeeded(last_status_) && kit_t::scores(&engine_, &candidates, share, matches_.data(),
                                                                     count * widths, widths, nullptr) == sz_success_k;
        if (!scored && succeeded(last_status_)) last_status_ = status_t::bad_alloc_k;
        for (std::size_t slot = 0; slot != distances.size(); ++slot)
            distances[slot] = scored ? distance_(queries_[slot / count].length, staged_[slot % count].length,
                                                 matches_.data() + slot * widths)
                                     : (std::numeric_limits<i64_t>::max)();
        staged_.clear();
    }

  private:

    /** Hashes @c queries_ into the engine, leaving it empty and the status failed on a refusal. */
    void prepare_() noexcept {
        sz_strings_view_t queries;
        sz_strings_from_string_views(queries_.data(), queries_.size(), &queries);
        sz_allocator_t allocator = engine_allocator(arena_);
        sz_status_t const prepared = kit_t::init(&engine_, &queries, window_widths_.data(), window_widths_.size(), 0,
                                                 &allocator, nullptr);
        if (prepared == sz_success_k) return;
        engine_ = sz_overlap_engine_t {};
        prepared_.clear();
        queries_.clear();
        last_status_ = prepared == sz_bad_alloc_k ? status_t::bad_alloc_k : status_t::shape_mismatch_k;
    }

    /** Starts the misses on every line of @p value, which the drain reads. */
    static void prefetch_(sz::string_view_t value) noexcept {
        for (std::size_t offset = 0; offset < value.size(); offset += fu::default_alignment_k)
            cache_hints_type_ {}(value.data() + offset, fu::prefetch_line_k);
    }

    /** The share of its longer side's windows a pair within @p radius may miss: every pair within
     *  the radius must count exactly, so the share rounds up past @c (radius + 1) / scale. */
    static sz_f32_t share_of_(i64_t radius) noexcept {
        if (radius >= overlap_resolution_k - 1) return 1.0f;
        f64_t const share = static_cast<f64_t>(max_of<i64_t>(radius, 0) + 1) / static_cast<f64_t>(overlap_resolution_k);
        return std::nextafter(static_cast<sz_f32_t>(share), 1.0f);
    }

    /** One pair's distance off its @p matches: the misses over whichever side has more windows at
     *  every width, summed, scaled to @ref overlap_resolution_k. */
    i64_t distance_(std::size_t query_length, std::size_t length, sz_u32_t const* matches) const noexcept {
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
        return static_cast<i64_t>(overlap_distance(longer, matched));
    }
};

using overlap_metric_t = overlap_metric<>;

#pragma endregion Overlap Metric

} // namespace unum::usearch::sequence
