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
 *  @c levenshtein_ladder and @c overlap_ladder pick the widest kit this build compiled in under the
 *  requested capability mask, at compile time, so nothing tests a capability at run time. Each
 *  engine draws its blocks from the metric's own arena, which @c reset rewinds, so after the widest
 *  query and the largest round seen nothing reaches the metric's allocator again.
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

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int64_t`
#include <cmath>   // `std::llround` - a width's matches off its share

#include <limits>  // `std::numeric_limits` - the distance a failed round reports
#include <memory>  // `std::allocator`
#include <span>    // `std::span`
#include <utility> // `std::swap`

#include <stringzilla/levenshtein.h>   // `sz_levenshtein_engine_*` & the per-backend distances
#include <stringzilla/overlap.h>       // `sz_overlap_engine_*` & the per-backend scores
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
 *  @brief Unit-cost Levenshtein over one round of candidates: @c reset prepares the query's engine,
 *      @c push stages candidate views, and @c drain streams every staged candidate through it in
 *      ascending length order, exact up to the radius.
 *
 *  One instance belongs to one thread and owns its engine, the arena the engine lives in, the
 *  staging, and the ordering; nothing reaches the allocator on the scoring path once @c init sized
 *  the round and the arena has held the widest query and round seen.
 *
 *  @tparam capability_ The admitted ISA backends as a capability mask; the widest kit this build compiled in
 *      among them is picked.
 *  @tparam symbol_ What one edit replaces: a byte, or a UTF-8 rune.
 *  @tparam allocator_type_ places every buffer, one set per metric.
 */
template <sz_capability_t capability_ = sz_cap_any_k, sz_levenshtein_symbol_t symbol_ = sz_levenshtein_bytes_k,
          typename allocator_type_ = std::allocator<byte_t>>
class levenshtein_metric {
  public:
    using kit_t = typename levenshtein_ladder<symbol_, capability_>::kit_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;

  private:

    /** The query's engine, one query prepared; empty for an empty query. Its blocks live in
     *  @c arena_, and its sweep scratch grows there to the largest round seen. */
    sz_levenshtein_engine_t engine_ {};

    /** Where the engine's blocks bump, rewound by every @ref reset. */
    growing_arena<allocator_t> arena_;

    /** The round's candidates in push order. */
    growing_buffer<sz_string_view_t, allocator_t> staged_;

    /** The candidates within reach of the radius, ascending by length: the sweep's own view. */
    growing_buffer<sz_string_view_t, allocator_t> ranked_;

    /** Two halves of the staging capacity: every staged candidate's length in the kit's symbols by
     *  push slot, then those of @c ranked_ by rank; measured per round. */
    growing_buffer<sz_size_t, allocator_t> symbol_counts_;

    /** Two halves of the staging capacity: @c order_[rank] = slot, ascending by length, the radix
     *  passes bouncing between them; the half the permutation does not end in takes the sweep's
     *  distances by rank. */
    growing_buffer<sz_size_t, allocator_t> order_;

    /** 0, 1, 2, … - the order the sweep visits @c ranked_ in, which is the one it lies in. */
    growing_buffer<sz_size_t, allocator_t> ranks_;
    status_t last_status_ = status_t::uninitialized_k;

  public:
    levenshtein_metric(allocator_t const& allocator = {}) noexcept
        : arena_(allocator), staged_(allocator), ranked_(allocator), symbol_counts_(allocator), order_(allocator),
          ranks_(allocator) {}
    levenshtein_metric(levenshtein_metric const&) = delete;
    levenshtein_metric& operator=(levenshtein_metric const&) = delete;
    levenshtein_metric(levenshtein_metric&&) noexcept = default;
    levenshtein_metric& operator=(levenshtein_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging, the ranked views, the symbol counts and the ordering.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at least the
     *      index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or rounds saturate rather
     *      than fail. A round is bounded to @c u32_t slots by design.
     */
    status_t init(std::size_t staging_limit) noexcept {
        if (!staging_limit || staging_limit > (std::numeric_limits<u32_t>::max)()) return status_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        std::size_t const capacity = staged_.capacity();
        if (status_t const ranked = ranked_.resize_uninitialized(capacity); failed(ranked)) return ranked;
        if (status_t const counted = symbol_counts_.resize_uninitialized(2 * capacity); failed(counted)) return counted;
        if (status_t const ordered = order_.resize_uninitialized(2 * capacity); failed(ordered)) return ordered;
        last_status_ = ranks_.resize_uninitialized(capacity);
        if (failed(last_status_)) return last_status_;
        for (std::size_t rank = 0; rank != capacity; ++rank) ranks_[rank] = rank;
        return last_status_;
    }

    /** How the last @ref init, @ref reset or @ref drain fared - the only place a failure surfaces,
     *  since the scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a round for @p query, preparing its engine and discarding whatever the previous round
     *  staged. */
    void reset(sz::string_view_t query) noexcept {
        staged_.clear();
        engine_ = sz_levenshtein_engine_t {};
        arena_.clear();
        last_status_ = status_t::success_k;
        if (query.empty()) return;
        sz_string_view_t const view = as_string_view(query);
        sz_strings_view_t queries;
        sz_strings_from_string_views(&view, 1, &queries);
        sz_allocator_t allocator = engine_allocator(arena_);
        sz_status_t const prepared = kit_t::init(&engine_, &queries, kit_t::symbol_k, &allocator, nullptr);
        if (prepared == sz_success_k) return;
        engine_ = sz_levenshtein_engine_t {};
        last_status_ = prepared == sz_bad_alloc_k ? status_t::bad_alloc_k : status_t::shape_mismatch_k;
    }

    /** @brief Stages one candidate, or refuses once the round is full. */
    basic::metric_staging_t push(sz::string_view_t value) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        staged_.push_back_reserved(as_string_view(value));
        return basic::metric_staging_t::staged_k;
    }

    /**
     *  @brief Scores everything staged in one sweep and closes the round.
     *
     *  Distances up to and including @p radius are exact; above it the value is some value above
     *  the radius. A failed round reports the furthest distance, so a full pool rejects it and an
     *  unfilled pool admits it last; @ref status is the only signal.
     *
     *  In length order the candidates whose length gap alone exceeds the radius form a prefix and a
     *  suffix that never enter the sweep and report that gap - the suffix cut is what keeps a long
     *  candidate out of a sweep of short ones - and the rest stream through it. Lengths count the
     *  kit's symbols, each candidate's measured once per round.
     *
     *  @param[out] distances One slot per staged candidate, in push order.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<i64_t> distances, i64_t radius) noexcept {
        std::size_t const count = staged_.size();
        if (failed(last_status_)) {
            for (std::size_t slot = 0; slot != count; ++slot) distances[slot] = (std::numeric_limits<i64_t>::max)();
            staged_.clear();
            return;
        }
        for (std::size_t slot = 0; slot != count; ++slot)
            symbol_counts_[slot] = symbols_(staged_[slot].start, staged_[slot].length);
        i64_t const query_length = engine_.count ? static_cast<i64_t>(engine_.lengths[0]) : 0;
        if (query_length == 0) {
            for (std::size_t slot = 0; slot != count; ++slot)
                distances[slot] = static_cast<i64_t>(symbol_counts_[slot]);
            staged_.clear();
            return;
        }
        ordering_t const ordering = order_by_length_(count);
        sz_size_t const* const order = ordering.order;
        sz_size_t* const swept = ordering.swept;
        std::size_t first_within = 0, end_within = count;
        while (first_within != count && query_length - static_cast<i64_t>(symbol_counts_[order[first_within]]) > radius)
            ++first_within;
        while (end_within != first_within &&
               static_cast<i64_t>(symbol_counts_[order[end_within - 1]]) - query_length > radius)
            --end_within;
        for (std::size_t rank = 0; rank != first_within; ++rank)
            distances[order[rank]] = query_length - static_cast<i64_t>(symbol_counts_[order[rank]]);
        for (std::size_t rank = end_within; rank != count; ++rank)
            distances[order[rank]] = static_cast<i64_t>(symbol_counts_[order[rank]]) - query_length;

        // The sweep reads a view of its own, the candidates within reach in rank order, so its
        // order, its counts and its distances all index that view and nothing past it.
        std::size_t const within = end_within - first_within;
        sz_size_t* const ranked_counts = symbol_counts_.data() + staged_.capacity();
        for (std::size_t rank = 0; rank != within; ++rank) {
            sz_size_t const slot = order[first_within + rank];
            ranked_[rank] = staged_[slot], ranked_counts[rank] = symbol_counts_[slot];
        }
        sz_strings_view_t candidates;
        sz_strings_from_string_views(ranked_.data(), within, &candidates);
        sz_size_t const bound = radius < (std::numeric_limits<i64_t>::max)()
                                    ? static_cast<sz_size_t>(max_of<i64_t>(radius, 0))
                                    : STRINGZILLA_SIZE_MAX;
        // A moved metric keeps its engine, whose growth must bump the arena it moved with.
        engine_.allocator.handle = &arena_;
        sz_status_t const scored = kit_t::distances(&engine_, &candidates, ranked_counts, ranks_.data(), bound, swept,
                                                    within, nullptr);
        if (scored != sz_success_k) last_status_ = status_t::bad_alloc_k;
        for (std::size_t rank = 0; rank != within; ++rank)
            distances[order[first_within + rank]] = scored == sz_success_k ? static_cast<i64_t>(swept[rank])
                                                                           : (std::numeric_limits<i64_t>::max)();
        staged_.clear();
    }

  private:

    /** Symbols in @p length bytes at @p text: the bytes, or the runes StringZilla decodes. */
    static std::size_t symbols_(sz_cptr_t text, std::size_t length) noexcept {
        if constexpr (symbol_ == sz_levenshtein_runes_k) return sz_levenshtein_utf8_runes(text, length);
        else return length;
    }

    /** The ascending-length permutation and the half of @c order_ left for the sweep's
     *  distances. */
    struct ordering_t {
        sz_size_t const* order;
        sz_size_t* swept;
    };

    /**
     *  @brief The ascending-length permutation of the first @p count staged views.
     *
     *  A stable LSD radix over the bytes of their symbol counts, low byte first, visiting only the
     *  bytes that differ across the round - so lengths under 256 cost one counting pass and one
     *  scatter, and nothing allocates or throws.
     *
     *  @return Whichever half of @c order_ holds the finished permutation, and the other one.
     */
    ordering_t order_by_length_(std::size_t count) noexcept {
        sz_size_t* source = order_.data();
        sz_size_t* target = order_.data() + order_.size() / 2;
        std::size_t all_ones = ~std::size_t(0), any_ones = 0;
        for (std::size_t slot = 0; slot != count; ++slot) {
            source[slot] = slot;
            all_ones &= symbol_counts_[slot], any_ones |= symbol_counts_[slot];
        }
        std::size_t const varying = all_ones ^ any_ones;
        for (std::size_t shift = 0; shift != std::numeric_limits<std::size_t>::digits; shift += 8) {
            if (((varying >> shift) & 0xFF) == 0) continue;
            std::size_t offsets[256] = {};
            for (std::size_t rank = 0; rank != count; ++rank) ++offsets[(symbol_counts_[source[rank]] >> shift) & 0xFF];
            std::size_t running = 0;
            for (std::size_t digit = 0; digit != 256; ++digit) {
                std::size_t const bucket = offsets[digit];
                offsets[digit] = running, running += bucket;
            }
            for (std::size_t rank = 0; rank != count; ++rank)
                target[offsets[(symbol_counts_[source[rank]] >> shift) & 0xFF]++] = source[rank];
            std::swap(source, target);
        }
        return {source, target};
    }
};

using levenshtein_metric_t = levenshtein_metric<>;

#pragma endregion Levenshtein Metric

#pragma region Overlap Metric

/**
 *  @brief Window overlap over one round of candidates: @c reset hashes the query's windows at every width
 *      into one B-tree, @c push stages candidate views, and @c drain streams every staged candidate through
 *      it, one prefix-hash chain per candidate and one probe per window.
 *
 *  Nothing is stored per node, so the window widths may be chosen per corpus without invalidating
 *  anything. One instance belongs to one thread and owns its engine, the arena the engine lives in,
 *  the staging and the shares. The staging and the shares are sized once by @c init, and the
 *  engine's tree and chains bump in the arena, so a round allocates nothing once a query and a
 *  round as long have been scored. A distance is the windows the shorter side misses of the longer,
 *  in fixed point over @ref resolution_k: zero where the windows coincide at every width, and
 *  @ref resolution_k where they share none.
 *
 *  @tparam capability_ The admitted ISA backends as a capability mask; the widest kit this build compiled in
 *      among them is picked.
 *  @tparam allocator_type_ places every buffer, one set per metric.
 */
template <sz_capability_t capability_ = sz_cap_any_k, typename allocator_type_ = std::allocator<byte_t>>
class overlap_metric {
  public:
    using kit_t = typename overlap_ladder<capability_>::kit_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;
    /** The full scale of a distance: the widest that still fits @c i64 times a @c u32 window count. */
    static constexpr i64_t resolution_k = i64_t(1) << 31;

  private:

    /** The query's engine, one query's tree over its windows at every width; empty for a query
     *  without windows. Its blocks live in @c arena_, its chains grow there to the longest seen. */
    sz_overlap_engine_t engine_ {};

    /** Where the engine's blocks bump, rewound by every @ref reset. */
    growing_arena<allocator_t> arena_;

    /** The round's candidates in push order. */
    growing_buffer<sz_string_view_t, allocator_t> staged_;

    /** The widths windows are taken at, all sharing one B-tree. */
    growing_buffer<sz_size_t, allocator_t> window_widths_;

    /** The round's @b [candidates, widths] shares, a staged candidate's row by its push slot. */
    growing_buffer<sz_f32_t, allocator_t> scores_;

    /** The query's windows summed over every width: the denominator's query side. */
    std::size_t query_windows_ = 0;

    /** How the last @ref init, @ref reset or @ref drain fared. */
    status_t last_status_ = status_t::uninitialized_k;

  public:
    overlap_metric(allocator_t const& allocator = {}) noexcept
        : arena_(allocator), staged_(allocator), window_widths_(allocator), scores_(allocator) {}
    overlap_metric(overlap_metric const&) = delete;
    overlap_metric& operator=(overlap_metric const&) = delete;
    overlap_metric(overlap_metric&&) noexcept = default;
    overlap_metric& operator=(overlap_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging and the shares, and records the window widths a round scores over.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at least the
     *      index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or rounds saturate rather
     *      than fail.
     *  @param[in] window_widths Window widths in bytes, all sharing one B-tree; a width past a text
     *      contributes nothing.
     */
    status_t init(std::size_t staging_limit, std::span<std::size_t const> window_widths) noexcept {
        if (!staging_limit || window_widths.empty()) return status_t::config_mismatch_k;
        for (std::size_t const width : window_widths)
            if (!width || width > (std::numeric_limits<u32_t>::max)()) return status_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        if (status_t const sized = window_widths_.resize_uninitialized(window_widths.size()); failed(sized))
            return sized;
        if (status_t const sized = scores_.resize_uninitialized(staged_.capacity() * window_widths.size());
            failed(sized))
            return sized;
        for (std::size_t index = 0; index != window_widths.size(); ++index)
            window_widths_[index] = window_widths[index];
        last_status_ = status_t::success_k;
        return last_status_;
    }

    /** How the last @ref init, @ref reset or @ref drain fared - the only place a failure surfaces,
     *  since the scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a round for @p query, building its B-tree once and discarding whatever the previous
     *  round staged. */
    void reset(sz::string_view_t query) noexcept {
        last_status_ = status_t::success_k;
        staged_.clear(), query_windows_ = 0;
        engine_ = sz_overlap_engine_t {};
        arena_.clear();
        for (std::size_t const width : window_widths_)
            if (width <= query.size()) query_windows_ += query.size() - width + 1;
        if (query_windows_ == 0) return;
        sz_string_view_t const view = as_string_view(query);
        sz_strings_view_t queries;
        sz_strings_from_string_views(&view, 1, &queries);
        sz_allocator_t allocator = engine_allocator(arena_);
        sz_status_t const prepared = kit_t::init(&engine_, &queries, window_widths_.data(), window_widths_.size(), 0,
                                                 &allocator, nullptr);
        if (prepared == sz_success_k) return;
        engine_ = sz_overlap_engine_t {};
        last_status_ = prepared == sz_bad_alloc_k ? status_t::bad_alloc_k : status_t::shape_mismatch_k;
    }

    /** Stages one candidate, or refuses once the round is full. */
    basic::metric_staging_t push(sz::string_view_t value) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        staged_.push_back_reserved(as_string_view(value));
        return basic::metric_staging_t::staged_k;
    }

    /**
     *  @brief Scores everything staged in one sweep and closes the round.
     *
     *  Distances up to and including @p radius are exact; above it the value is some value above
     *  the radius. A failed round reports the furthest distance, so a full pool rejects it and an
     *  unfilled pool admits it last; @ref status is the only signal.
     *
     *  A pair within @p radius misses at most @c floor(((radius+1) × longer - 1) / resolution_k) of
     *  the longer side's windows, so the sweep is bounded by that of the round's longest pair, past
     *  which a pair's shares read zero - a full-scale distance, past the radius as it must be.
     *
     *  @param[out] distances One slot per staged candidate, in push order.
     *  @param[in] radius The furthest distance the caller still tells apart.
     */
    void drain(std::span<i64_t> distances, i64_t radius) noexcept {
        std::size_t const count = staged_.size();
        if (failed(last_status_) || query_windows_ == 0) {
            for (std::size_t slot = 0; slot != count; ++slot)
                distances[slot] = failed(last_status_) ? (std::numeric_limits<i64_t>::max)() : resolution_k;
            staged_.clear();
            return;
        }
        sz_size_t bound = STRINGZILLA_SIZE_MAX;
        if (radius < resolution_k) {
            u64_t longest = query_windows_;
            for (std::size_t slot = 0; slot != count; ++slot)
                longest = max_of<u64_t>(longest, windows_(staged_[slot].length));
            bound = static_cast<sz_size_t>(((static_cast<u64_t>(max_of<i64_t>(radius, 0)) + 1) * longest - 1) /
                                           static_cast<u64_t>(resolution_k));
        }
        std::size_t const widths = window_widths_.size();
        sz_strings_view_t candidates;
        sz_strings_from_string_views(staged_.data(), count, &candidates);
        // A moved metric keeps its engine, whose growth must bump the arena it moved with.
        engine_.allocator.handle = &arena_;
        sz_status_t const scored = kit_t::scores(&engine_, &candidates, bound, scores_.data(), count * widths, widths,
                                                 nullptr);
        if (scored != sz_success_k) last_status_ = status_t::bad_alloc_k;
        for (std::size_t slot = 0; slot != count; ++slot)
            distances[slot] = scored == sz_success_k ? distance_(staged_[slot].length, scores_.data() + slot * widths)
                                                     : (std::numeric_limits<i64_t>::max)();
        staged_.clear();
    }

  private:

    /** Windows a text of @p length bytes holds, summed over every width. */
    u64_t windows_(std::size_t length) const noexcept {
        u64_t windows = 0;
        for (std::size_t const width : window_widths_)
            if (width <= length) windows += length - width + 1;
        return windows;
    }

    /** One candidate's distance off its @p shares: every width's matches recovered over that
     *  width's longer side, the misses over whichever side has more windows in all, scaled to
     *  @ref resolution_k. */
    i64_t distance_(std::size_t length, sz_f32_t const* shares) const noexcept {
        std::size_t const query_length = engine_.lengths[0];
        u64_t longer = 0, matches = 0;
        for (std::size_t index = 0; index != window_widths_.size(); ++index) {
            std::size_t const width = window_widths_[index];
            u64_t const windows = width <= length ? length - width + 1 : 0;
            u64_t const query_windows = width <= query_length ? query_length - width + 1 : 0;
            u64_t const width_longer = max_of(windows, query_windows);
            matches += static_cast<u64_t>(std::llround(static_cast<f64_t>(shares[index]) * width_longer));
            longer += width_longer;
        }
        // The longer text has more windows at every width, so the widths' longer sides sum to the
        // longer side in all, and a union denominator would let repeats score below zero.
        return static_cast<i64_t>((longer - matches) * static_cast<u64_t>(resolution_k) / longer);
    }
};

using overlap_metric_t = overlap_metric<>;

#pragma endregion Overlap Metric

} // namespace unum::usearch::sequence
