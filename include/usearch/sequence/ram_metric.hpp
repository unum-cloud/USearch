/**
 *  @file include/usearch/sequence/ram_metric.hpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Batched similarity kernels over variable-length byte sequences, assembled from StringZilla's
 *      per-backend step primitives.
 *
 *  The sequence index scores one round at a time: @c reset opens it for a query, @c push stages
 *  candidate views, and @c drain scores every staged candidate within the radius the caller still
 *  tells apart. A metric here owns everything a round needs - the query's preparation, the staging,
 *  the scratch - so one instance serves one thread and never allocates on the scoring path once sized.
 *
 *  @section sequence_ram_metric_kits Kits
 *
 *  A kit binds one ISA's state and vertical types, its @c init / @c step / @c any_active / @c score
 *  verbs, its symbol count, query preparation, transpose and class loaders, how many candidates one
 *  step advances, and how many registers it advances per position. @c levenshtein_kit picks the
 *  widest kit this build compiled in under the requested capability mask and symbol, at compile
 *  time, so the loops below unroll around the kit's constants and never test a capability at run
 *  time.
 *
 *  @section sequence_ram_metric_levenshtein Levenshtein
 *
 *  Unit-cost Myers: the query's match masks are prepared once, and every candidate streams through
 *  them one symbol per step, several candidates per step. The round is scored in ascending length
 *  order, since the candidates advanced together run as long as the longest of them, each candidate's
 *  score is read at the position where its text ends, and a freeze test every few positions stops a
 *  flight whose every candidate is already past the radius. Distances count edits over bytes, or
 *  over UTF-8 runes under @c sz_levenshtein_runes_k.
 *
 *  @section sequence_ram_metric_overlap Overlap
 *
 *  Window overlap: the query's windows at every width are hashed into one B-tree once per round, and
 *  every candidate's window hashes are probed against it. Four candidates' prefix-hash chains advance
 *  together, since one chain's recurrence is latency-bound and independent chains hide it. A distance
 *  is the share of the longer side's windows the other side misses, in fixed point, exact at every radius.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int64_t`

#include <limits>      // `std::numeric_limits` - the distance a failed round reports
#include <memory>      // `std::allocator`
#include <span>        // `std::span`
#include <type_traits> // `std::conditional_t`, `std::integral_constant` - the kit choice and the resident word count
#include <utility>     // `std::swap`

#include <stringzilla/levenshtein.h>   // `sz_levenshtein_query_*`, the transposes, and the per-backend step primitives
#include <stringzilla/overlap.h>       // `sz_overlap_*` & the per-backend step primitives
#include <stringzilla/stringzilla.hpp> // `sz::string_view_t` - every operand a metric takes

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/basic/ram_index.hpp" // `basic::metric_staging_t`

namespace unum::usearch::sequence {
using namespace unum::usearch::utilities;

#pragma region Levenshtein Kernels

struct levenshtein_serial_kit_t;
struct levenshtein_haswell_kit_t;
struct levenshtein_skylake_kit_t;
struct levenshtein_neon_kit_t;
struct levenshtein_utf8_serial_kit_t;
struct levenshtein_utf8_haswell_kit_t;
struct levenshtein_utf8_skylake_kit_t;
struct levenshtein_utf8_neon_kit_t;

/** Symbols in a byte string: its length, as every byte is one. */
constexpr std::size_t levenshtein_byte_symbols_(sz_cptr_t, sz_size_t length) noexcept { return length; }

/** Bytes a byte query's class table takes: one class per byte value, whatever the query. */
constexpr std::size_t levenshtein_byte_class_table_bytes_(sz_size_t) noexcept { return sz_levenshtein_byte_classes_k; }

/** Scalar Myers: one candidate per step, eight independent candidates per position to fill the pipeline. */
struct levenshtein_serial_kit_t {
    using state_t = sz_levenshtein_u64x1_state_serial_t;
    using vertical_t = sz_levenshtein_u64x1_vertical_serial_t;
    using class_t = sz_u32_t;
    using classes_t = sz_u32_t;
    using counts_t = sz_u64_t;
    static constexpr std::size_t candidates_per_step_k = sz_levenshtein_u64x1_candidates_per_step_serial_k;
    static constexpr std::size_t registers_per_position_k = sz_levenshtein_u64x1_registers_per_position_serial_k;
    static constexpr auto init = sz_levenshtein_u64x1_init_serial;
    static constexpr auto step = sz_levenshtein_u64x1_step_serial;
    static constexpr auto any_active = sz_levenshtein_u64x1_any_active_serial;
    static constexpr auto score = sz_levenshtein_u64x1_score_serial;
    static constexpr auto symbols = levenshtein_byte_symbols_;
    static constexpr auto mask_entries = sz_levenshtein_query_mask_entries;
    static constexpr auto class_table_bytes = levenshtein_byte_class_table_bytes_;
    static constexpr auto prepare = sz_levenshtein_query_prepare;
    static constexpr auto transpose = sz_levenshtein_transpose;
    static classes_t classes(class_t const* transpose_classes) noexcept { return *transpose_classes; }
    static counts_t counts(sz_u64_t const* lengths) noexcept { return *lengths; }
};

/** Scalar Myers over UTF-8 runes: the serial steps, fed the rune classes a page table hands out. */
struct levenshtein_utf8_serial_kit_t : levenshtein_serial_kit_t {
    static constexpr auto symbols = sz_levenshtein_utf8_runes;
    static constexpr auto mask_entries = sz_levenshtein_query_mask_entries_utf8;
    static constexpr auto class_table_bytes = sz_levenshtein_utf8_pages_bytes;
    static constexpr auto prepare = sz_levenshtein_query_prepare_utf8;
    static constexpr auto transpose = sz_levenshtein_transpose_utf8;
};

#if STRINGZILLA_TARGET_HASWELL

/** Intel Haswell AVX2: four candidates per step, one register per position, a SIMD byte transpose. */
struct levenshtein_haswell_kit_t {
    using state_t = sz_levenshtein_u64x4_state_haswell_t;
    using vertical_t = sz_levenshtein_u64x4_vertical_haswell_t;
    using class_t = sz_u8_t;
    using classes_t = sz_u256_vec_t;
    using counts_t = sz_u256_vec_t;
    static constexpr std::size_t candidates_per_step_k = sz_levenshtein_u64x4_candidates_per_step_haswell_k;
    static constexpr std::size_t registers_per_position_k = sz_levenshtein_u64x4_registers_per_position_haswell_k;
    static constexpr auto init = sz_levenshtein_u64x4_init_haswell;
    static constexpr auto step = sz_levenshtein_u64x4_step_haswell;
    static constexpr auto any_active = sz_levenshtein_u64x4_any_active_haswell;
    static constexpr auto score = sz_levenshtein_u64x4_score_haswell;
    static constexpr auto symbols = levenshtein_byte_symbols_;
    static constexpr auto mask_entries = sz_levenshtein_query_mask_entries;
    static constexpr auto class_table_bytes = levenshtein_byte_class_table_bytes_;
    static constexpr auto prepare = sz_levenshtein_query_prepare;
    static constexpr auto transpose = sz_levenshtein_u8x4_transpose_haswell;
    static classes_t classes(class_t const* transpose_classes) noexcept {
        return sz_levenshtein_u64x4_classes_u8_haswell(transpose_classes);
    }
    static counts_t counts(sz_u64_t const* lengths) noexcept {
        counts_t counts_vec;
        counts_vec.ymm = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(lengths));
        return counts_vec;
    }
};

/** Intel Haswell AVX2 over UTF-8 runes: four candidates per step, fed 32-bit rune classes. */
struct levenshtein_utf8_haswell_kit_t : levenshtein_haswell_kit_t {
    using class_t = sz_u32_t;
    static constexpr auto symbols = sz_levenshtein_utf8_runes;
    static constexpr auto mask_entries = sz_levenshtein_query_mask_entries_utf8;
    static constexpr auto class_table_bytes = sz_levenshtein_utf8_pages_bytes;
    static constexpr auto prepare = sz_levenshtein_query_prepare_utf8;
    static constexpr auto transpose = sz_levenshtein_transpose_utf8;
    static classes_t classes(class_t const* transpose_classes) noexcept {
        return sz_levenshtein_u64x4_classes_u32_haswell(transpose_classes);
    }
};
#endif // STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_TARGET_SKYLAKE

/** Intel Skylake-X AVX-512: eight candidates per step, one register per position, a SIMD byte transpose. */
struct levenshtein_skylake_kit_t {
    using state_t = sz_levenshtein_u64x8_state_skylake_t;
    using vertical_t = sz_levenshtein_u64x8_vertical_skylake_t;
    using class_t = sz_u8_t;
    using classes_t = sz_u512_vec_t;
    using counts_t = sz_u512_vec_t;
    static constexpr std::size_t candidates_per_step_k = sz_levenshtein_u64x8_candidates_per_step_skylake_k;
    static constexpr std::size_t registers_per_position_k = sz_levenshtein_u64x8_registers_per_position_skylake_k;
    static constexpr auto init = sz_levenshtein_u64x8_init_skylake;
    static constexpr auto step = sz_levenshtein_u64x8_step_skylake;
    static constexpr auto any_active = sz_levenshtein_u64x8_any_active_skylake;
    static constexpr auto score = sz_levenshtein_u64x8_score_skylake;
    static constexpr auto symbols = levenshtein_byte_symbols_;
    static constexpr auto mask_entries = sz_levenshtein_query_mask_entries;
    static constexpr auto class_table_bytes = levenshtein_byte_class_table_bytes_;
    static constexpr auto prepare = sz_levenshtein_query_prepare;
    static constexpr auto transpose = sz_levenshtein_u8x8_transpose_skylake;
    static classes_t classes(class_t const* transpose_classes) noexcept {
        return sz_levenshtein_u64x8_classes_u8_skylake(transpose_classes);
    }
    static counts_t counts(sz_u64_t const* lengths) noexcept {
        counts_t counts_vec;
        counts_vec.zmm = _mm512_loadu_si512(lengths);
        return counts_vec;
    }
};

/** Intel Skylake-X AVX-512 over UTF-8 runes: eight candidates per step, fed 32-bit rune classes. */
struct levenshtein_utf8_skylake_kit_t : levenshtein_skylake_kit_t {
    using class_t = sz_u32_t;
    static constexpr auto symbols = sz_levenshtein_utf8_runes;
    static constexpr auto mask_entries = sz_levenshtein_query_mask_entries_utf8;
    static constexpr auto class_table_bytes = sz_levenshtein_utf8_pages_bytes;
    static constexpr auto prepare = sz_levenshtein_query_prepare_utf8;
    static constexpr auto transpose = sz_levenshtein_transpose_utf8;
    static classes_t classes(class_t const* transpose_classes) noexcept {
        return sz_levenshtein_u64x8_classes_u32_skylake(transpose_classes);
    }
};
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_NEON

/** Arm NEON: two candidates per step, four registers per position, a SIMD byte transpose. */
struct levenshtein_neon_kit_t {
    using state_t = sz_levenshtein_u64x2_state_neon_t;
    using vertical_t = sz_levenshtein_u64x2_vertical_neon_t;
    using class_t = sz_u8_t;
    using classes_t = sz_u128_vec_t;
    using counts_t = sz_u128_vec_t;
    static constexpr std::size_t candidates_per_step_k = sz_levenshtein_u64x2_candidates_per_step_neon_k;
    static constexpr std::size_t registers_per_position_k = sz_levenshtein_u64x2_registers_per_position_neon_k;
    static constexpr auto init = sz_levenshtein_u64x2_init_neon;
    static constexpr auto step = sz_levenshtein_u64x2_step_neon;
    static constexpr auto any_active = sz_levenshtein_u64x2_any_active_neon;
    static constexpr auto score = sz_levenshtein_u64x2_score_neon;
    static constexpr auto symbols = levenshtein_byte_symbols_;
    static constexpr auto mask_entries = sz_levenshtein_query_mask_entries;
    static constexpr auto class_table_bytes = levenshtein_byte_class_table_bytes_;
    static constexpr auto prepare = sz_levenshtein_query_prepare;
    static constexpr auto transpose = sz_levenshtein_u8x8_transpose_neon;
    static classes_t classes(class_t const* transpose_classes) noexcept {
        return sz_levenshtein_u64x2_classes_u8_neon(transpose_classes);
    }
    static counts_t counts(sz_u64_t const* lengths) noexcept {
        counts_t counts_vec;
        counts_vec.u64x2 = vld1q_u64(lengths);
        return counts_vec;
    }
};

/** Arm NEON over UTF-8 runes: two candidates per step, four registers per position. */
struct levenshtein_utf8_neon_kit_t : levenshtein_neon_kit_t {
    using class_t = sz_u32_t;
    static constexpr auto symbols = sz_levenshtein_utf8_runes;
    static constexpr auto mask_entries = sz_levenshtein_query_mask_entries_utf8;
    static constexpr auto class_table_bytes = sz_levenshtein_utf8_pages_bytes;
    static constexpr auto prepare = sz_levenshtein_query_prepare_utf8;
    static constexpr auto transpose = sz_levenshtein_transpose_utf8;
    static classes_t classes(class_t const* transpose_classes) noexcept {
        return sz_levenshtein_u64x2_classes_u32_neon(transpose_classes);
    }
};
#endif // STRINGZILLA_TARGET_NEON

/** The widest Levenshtein kit this build compiled in among the backends @p capability_ admits, over
 *  the symbols @p symbol_ names; each row's flag is the backend's @c STRINGZILLA_TARGET_* and the
 *  mask's bit, so an unselected kit's body never has to exist. */
template <sz_capability_t capability_, sz_levenshtein_symbol_t symbol_ = sz_levenshtein_bytes_k>
using levenshtein_kit = std::conditional_t<
    symbol_ == sz_levenshtein_bytes_k,
    std::conditional_t<
        STRINGZILLA_TARGET_SKYLAKE && (capability_ & sz_cap_skylake_k) != 0, levenshtein_skylake_kit_t,
        std::conditional_t<STRINGZILLA_TARGET_HASWELL && (capability_ & sz_cap_haswell_k) != 0,
                           levenshtein_haswell_kit_t,
                           std::conditional_t<STRINGZILLA_TARGET_NEON && (capability_ & sz_cap_neon_k) != 0,
                                              levenshtein_neon_kit_t, levenshtein_serial_kit_t>>>,
    std::conditional_t<
        STRINGZILLA_TARGET_SKYLAKE && (capability_ & sz_cap_skylake_k) != 0, levenshtein_utf8_skylake_kit_t,
        std::conditional_t<STRINGZILLA_TARGET_HASWELL && (capability_ & sz_cap_haswell_k) != 0,
                           levenshtein_utf8_haswell_kit_t,
                           std::conditional_t<STRINGZILLA_TARGET_NEON && (capability_ & sz_cap_neon_k) != 0,
                                              levenshtein_utf8_neon_kit_t, levenshtein_utf8_serial_kit_t>>>>;

/** Query words whose verticals stay register-resident: one and two measured faster than the scratch walk,
 *  wider ones spill on sixteen-register machines. */
constexpr std::size_t levenshtein_resident_words_k = 2;

/** Positions between freeze tests: a test costs a sixth of a one-word step, so this keeps it near one
 *  percent while a flight past the radius still stops within a few steps. */
constexpr std::size_t levenshtein_freeze_cadence_k = 16;

/**
 *  @brief The scoring loop over one sorted round: @c candidates_per_position_k candidates per flight, advanced as
 *      @c registers_per_position_k registers per position, the candidates taken in @p order and each distance
 *      scattered back to its push slot.
 *
 *  Candidate ends arrive in ascending order within a register, so one compare per position finds the next candidate
 *  whose score to read; a freeze test every @c levenshtein_freeze_cadence_k positions ends a flight whose every
 *  candidate is past the radius. The verticals live in a local array on the resident path - the vector unions may
 *  alias, and only a local lets the step's stores be proven disjoint - and in @p verticals_scratch above it.
 *
 *  @param[in] symbol_counts Every staged candidate's length in the kit's symbols, by push slot.
 *  @param[in] words The unroll bound: an @c std::integral_constant on the resident path, the query's count otherwise.
 */
template <typename kit_, typename words_type_>
void levenshtein_distances_(sz_levenshtein_query_t const& shared_query, sz::string_view_t const* candidates,
                            std::size_t const* symbol_counts, u32_t const* order, std::size_t count, i64_t radius,
                            i64_t* distances, words_type_ words,
                            typename kit_::vertical_t* verticals_scratch) noexcept {
    // A local copy: nothing stored through the verticals can alias it, so the step keeps its fields in registers.
    sz_levenshtein_query_t const query = shared_query;
    constexpr std::size_t candidates_per_step_k = kit_::candidates_per_step_k;
    constexpr std::size_t registers_per_position_k = kit_::registers_per_position_k;
    constexpr std::size_t candidates_per_position_k = candidates_per_step_k * registers_per_position_k;
    constexpr std::size_t positions_per_transpose_k = sz_levenshtein_positions_per_transpose_k;
    constexpr bool resident_k = !std::is_same_v<words_type_, std::size_t>;
    static_assert(sizeof(typename kit_::class_t) * positions_per_transpose_k * candidates_per_position_k <= 16384,
                  "The transpose lives on the stack; shrink it before widening the flight");

    typename kit_::class_t transpose_classes[positions_per_transpose_k][candidates_per_position_k];
    typename kit_::vertical_t resident_verticals[registers_per_position_k * levenshtein_resident_words_k];
    typename kit_::vertical_t* const verticals = resident_k ? resident_verticals : verticals_scratch;
    sz_ssize_t const kernel_radius = static_cast<sz_ssize_t>(radius);

    for (std::size_t flight_start = 0; flight_start < count; flight_start += candidates_per_position_k) {
        std::size_t const flight_count = min_of(candidates_per_position_k, count - flight_start);
        sz_cptr_t starts[candidates_per_position_k] = {};
        sz_u64_t byte_counts[candidates_per_position_k] = {};
        sz_u64_t lengths[candidates_per_position_k] = {};
        std::size_t cursors[candidates_per_position_k] = {};
        std::size_t next_read[registers_per_position_k] = {};
        for (std::size_t candidate = 0; candidate != flight_count; ++candidate) {
            u32_t const slot = order[flight_start + candidate];
            starts[candidate] = candidates[slot].data(), byte_counts[candidate] = candidates[slot].size();
            lengths[candidate] = symbol_counts[slot];
        }
        // Padding candidates of a partial flight have no slot to read into.
        for (std::size_t candidate = flight_count; candidate != candidates_per_position_k; ++candidate)
            lengths[candidate] = 0;
        typename kit_::state_t states[registers_per_position_k];
        typename kit_::counts_t counts[registers_per_position_k];
        for (std::size_t register_index = 0; register_index != registers_per_position_k; ++register_index) {
            kit_::init(&states[register_index], verticals + register_index * words, words, &query);
            counts[register_index] = kit_::counts(lengths + register_index * candidates_per_step_k);
            // Candidates past the flight's count end at position zero and are skipped, never read.
            while (register_index * candidates_per_step_k + next_read[register_index] >= flight_count &&
                   next_read[register_index] != candidates_per_step_k)
                ++next_read[register_index];
        }

        bool alive = true;
        for (std::size_t transpose_start = 0, filled = positions_per_transpose_k;
             alive && filled == positions_per_transpose_k; transpose_start += filled) {
            // A rune transpose rewrites a dry candidate's symbol count to the value it holds.
            filled = kit_::transpose(&query, starts, byte_counts, candidates_per_position_k, cursors, lengths,
                                     transpose_start, positions_per_transpose_k, &transpose_classes[0][0]);
            for (std::size_t position = 0; position != filled; ++position) {
                std::size_t const global = transpose_start + position;
                if (global % levenshtein_freeze_cadence_k == 0 && global != 0) {
                    alive = false;
                    for (std::size_t register_index = 0; register_index != registers_per_position_k; ++register_index)
                        alive |= kit_::any_active(&states[register_index], counts[register_index], global,
                                                  kernel_radius) == sz_true_k;
                    if (!alive) break;
                }
                for (std::size_t register_index = 0; register_index != registers_per_position_k; ++register_index) {
                    std::size_t const first_candidate = register_index * candidates_per_step_k;
                    std::size_t& next = next_read[register_index];
                    while (next != candidates_per_step_k && first_candidate + next < flight_count &&
                           lengths[first_candidate + next] == global) {
                        distances[order[flight_start + first_candidate + next]] = static_cast<i64_t>(
                            kit_::score(&states[register_index], next));
                        ++next;
                    }
                    kit_::step(&states[register_index], verticals + register_index * words, words, &query,
                               kit_::classes(transpose_classes[position] + first_candidate));
                }
            }
        }
        // Candidates ending at the flight's end, or frozen past the radius, report their current score.
        for (std::size_t register_index = 0; register_index != registers_per_position_k; ++register_index) {
            std::size_t const first_candidate = register_index * candidates_per_step_k;
            for (std::size_t candidate = first_candidate + next_read[register_index];
                 candidate < flight_count && candidate < first_candidate + candidates_per_step_k; ++candidate)
                distances[order[flight_start + candidate]] = static_cast<i64_t>(
                    kit_::score(&states[register_index], candidate - first_candidate));
        }
    }
}

/** Scratch verticals a @p words -word query needs above the resident count; zero below it. */
template <typename kit_>
constexpr std::size_t levenshtein_scratch_verticals(std::size_t words) noexcept {
    return words > levenshtein_resident_words_k ? kit_::registers_per_position_k * words : 0;
}

/** The scoring entry: register-resident verticals for one- and two-word queries, scratch above. */
template <typename kit_>
void levenshtein_distances(sz_levenshtein_query_t const& query, sz::string_view_t const* candidates,
                           std::size_t const* symbol_counts, u32_t const* order, std::size_t count, i64_t radius,
                           i64_t* distances, typename kit_::vertical_t* verticals_scratch) noexcept {
    std::size_t const words = sz_levenshtein_query_words(query.length);
    if (words == 1)
        return levenshtein_distances_<kit_>(query, candidates, symbol_counts, order, count, radius, distances,
                                            std::integral_constant<std::size_t, 1> {}, verticals_scratch);
    if (words == 2)
        return levenshtein_distances_<kit_>(query, candidates, symbol_counts, order, count, radius, distances,
                                            std::integral_constant<std::size_t, 2> {}, verticals_scratch);
    levenshtein_distances_<kit_>(query, candidates, symbol_counts, order, count, radius, distances, words,
                                 verticals_scratch);
}

#pragma endregion Levenshtein Kernels

#pragma region Overlap Kernels

struct overlap_serial_kit_t;
struct overlap_haswell_kit_t;
struct overlap_skylake_kit_t;
struct overlap_neon_kit_t;

/** Scalar binding: one position per step, one key per sort and probe step. */
struct overlap_serial_kit_t {
    static constexpr std::size_t positions_per_step_k = sz_overlap_f64x1_positions_per_step_serial_k;
    static constexpr auto prefix_hash_step = sz_overlap_f64x1_prefix_hash_step_serial;
    static constexpr auto prefix_hash_step_tail = sz_overlap_f64x1_prefix_hash_step_tail_serial;
    static constexpr auto window_hash_step = sz_overlap_f64x1_window_hash_step_serial;
    static constexpr auto window_hash_step_tail = sz_overlap_f64x1_window_hash_step_tail_serial;
    static constexpr auto btree_sort = sz_overlap_u32x1_btree_sort_serial;
    static constexpr auto btree_probe = sz_overlap_u32x1_btree_probe_serial;
};

#if STRINGZILLA_TARGET_HASWELL

/** Intel Haswell AVX2: four positions per step, eight keys per sort and probe step. */
struct overlap_haswell_kit_t {
    static constexpr std::size_t positions_per_step_k = sz_overlap_f64x4_positions_per_step_haswell_k;
    static constexpr auto prefix_hash_step = sz_overlap_f64x4_prefix_hash_step_haswell;
    static constexpr auto prefix_hash_step_tail = sz_overlap_f64x4_prefix_hash_step_tail_haswell;
    static constexpr auto window_hash_step = sz_overlap_f64x4_window_hash_step_haswell;
    static constexpr auto window_hash_step_tail = sz_overlap_f64x4_window_hash_step_tail_haswell;
    static constexpr auto btree_sort = sz_overlap_u32x8_btree_sort_haswell;
    static constexpr auto btree_probe = sz_overlap_u32x8_btree_probe_haswell;
};
#endif // STRINGZILLA_TARGET_HASWELL

#if STRINGZILLA_TARGET_SKYLAKE

/** Intel Skylake AVX-512: eight positions per step, sixteen keys per sort and probe step. */
struct overlap_skylake_kit_t {
    static constexpr std::size_t positions_per_step_k = sz_overlap_f64x8_positions_per_step_skylake_k;
    static constexpr auto prefix_hash_step = sz_overlap_f64x8_prefix_hash_step_skylake;
    static constexpr auto prefix_hash_step_tail = sz_overlap_f64x8_prefix_hash_step_tail_skylake;
    static constexpr auto window_hash_step = sz_overlap_f64x8_window_hash_step_skylake;
    static constexpr auto window_hash_step_tail = sz_overlap_f64x8_window_hash_step_tail_skylake;
    static constexpr auto btree_sort = sz_overlap_u32x16_btree_sort_skylake;
    static constexpr auto btree_probe = sz_overlap_u32x16_btree_probe_skylake;
};
#endif // STRINGZILLA_TARGET_SKYLAKE

#if STRINGZILLA_TARGET_NEON

/** Arm NEON: two positions per step, four keys per sort and probe step. */
struct overlap_neon_kit_t {
    static constexpr std::size_t positions_per_step_k = sz_overlap_f64x2_positions_per_step_neon_k;
    static constexpr auto prefix_hash_step = sz_overlap_f64x2_prefix_hash_step_neon;
    static constexpr auto prefix_hash_step_tail = sz_overlap_f64x2_prefix_hash_step_tail_neon;
    static constexpr auto window_hash_step = sz_overlap_f64x2_window_hash_step_neon;
    static constexpr auto window_hash_step_tail = sz_overlap_f64x2_window_hash_step_tail_neon;
    static constexpr auto btree_sort = sz_overlap_u32x4_btree_sort_neon;
    static constexpr auto btree_probe = sz_overlap_u32x4_btree_probe_neon;
};
#endif // STRINGZILLA_TARGET_NEON

/** The widest overlap kit this build compiled in among the backends @p capability_ admits; each row's flag is
 *  the backend's @c STRINGZILLA_TARGET_* and the mask's bit, so an unselected kit's body never has to exist. */
template <sz_capability_t capability_>
using overlap_kit = std::conditional_t<
    STRINGZILLA_TARGET_SKYLAKE && (capability_ & sz_cap_skylake_k) != 0, overlap_skylake_kit_t,
    std::conditional_t<STRINGZILLA_TARGET_HASWELL && (capability_ & sz_cap_haswell_k) != 0, overlap_haswell_kit_t,
                       std::conditional_t<STRINGZILLA_TARGET_NEON && (capability_ & sz_cap_neon_k) != 0,
                                          overlap_neon_kit_t, overlap_serial_kit_t>>>;

/** Candidates whose prefix-hash chains advance together: one chain's recurrence is latency-bound at about fifty
 *  cycles per step, and four independent chains hide it - measured 39 → 26 µs per round of 64 candidates. */
constexpr std::size_t overlap_candidates_per_flight_k = 4;

/** One text's chain: @c length + 1 prefix hashes into @p prefix_hashes, starting at the empty one. */
template <typename kit_>
void overlap_prefix_hashes(char const* text, std::size_t length, f64_t* prefix_hashes) noexcept {
    constexpr std::size_t step_k = kit_::positions_per_step_k;
    prefix_hashes[0] = 0.0;
    f64_t prior = 0.0;
    std::size_t position = 0;
    for (; position + step_k <= length; position += step_k)
        prior = kit_::prefix_hash_step(prior, text + position, prefix_hashes + position + 1);
    if (position != length)
        kit_::prefix_hash_step_tail(prior, text + position, length - position, prefix_hashes + position + 1);
}

/** One flight's chains advanced in one loop, each candidate carrying its own prior into its own @p stride -wide
 *  slice of @p prefix_hashes; a chain past its text idles until the longest ends, then every tail finishes alone. */
template <typename kit_>
void overlap_prefix_hashes_flight(sz::string_view_t const* candidates, std::size_t stride,
                                  f64_t* prefix_hashes) noexcept {
    constexpr std::size_t step_k = kit_::positions_per_step_k;
    constexpr std::size_t flight_k = overlap_candidates_per_flight_k;
    f64_t priors[flight_k] = {};
    std::size_t longest = 0;
    for (std::size_t chain = 0; chain != flight_k; ++chain) {
        prefix_hashes[chain * stride] = 0.0;
        longest = max_of(longest, candidates[chain].size());
    }
    for (std::size_t position = 0; position + step_k <= longest; position += step_k)
        for (std::size_t chain = 0; chain != flight_k; ++chain)
            if (position + step_k <= candidates[chain].size())
                priors[chain] = kit_::prefix_hash_step(priors[chain], candidates[chain].data() + position,
                                                       prefix_hashes + chain * stride + position + 1);
    for (std::size_t chain = 0; chain != flight_k; ++chain) {
        std::size_t const length = candidates[chain].size();
        std::size_t const tail = length % step_k;
        if (tail == 0) continue;
        kit_::prefix_hash_step_tail(priors[chain], candidates[chain].data() + length - tail, tail,
                                    prefix_hashes + chain * stride + length - tail + 1);
    }
}

/** One width's window hashes off a chain already built; @p windows is @c length - width + 1. */
template <typename kit_>
void overlap_window_hashes(f64_t const* prefix_hashes, std::size_t windows, std::size_t width, f64_t window_power,
                           u32_t* window_hashes) noexcept {
    constexpr std::size_t step_k = kit_::positions_per_step_k;
    std::size_t window = 0;
    for (; window + step_k <= windows; window += step_k)
        kit_::window_hash_step(prefix_hashes + window, prefix_hashes + window + width, window_power,
                               window_hashes + window);
    if (window != windows)
        kit_::window_hash_step_tail(prefix_hashes + window, prefix_hashes + window + width, window_power,
                                    windows - window, window_hashes + window);
}

#pragma endregion Overlap Kernels

#pragma region Levenshtein Metric

/**
 *  @brief Unit-cost Levenshtein over one round of candidates: @c reset prepares the query's Myers masks,
 *      @c push stages candidate views, and @c drain streams every staged candidate through the masks in
 *      ascending length order, exact up to the radius.
 *
 *  One instance belongs to one thread and owns its prepared query, mask table, vertical scratch, staging, and
 *  ordering; nothing allocates on the scoring path once @c init sized the round and @c reset the query.
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
    using kit_t = levenshtein_kit<capability_, symbol_>;
    using allocator_t = allocator_type_;
    using is_transparent = void;

  private:

    /** The prepared query; its masks point into @c masks_ and its class table into @c class_table_,
     *  and @c length is zero for an empty query. */
    sz_levenshtein_query_t query_ {};

    /** The query's match masks, grown to the widest query seen. */
    growing_buffer<u64_t, allocator_t> masks_;

    /** The query's byte-to-class map or rune page table, grown to the widest query seen. */
    growing_buffer<sz_u8_t, allocator_t> class_table_;

    /** Verticals for queries wider than the resident count, grown to the widest query seen. */
    growing_buffer<typename kit_t::vertical_t, allocator_t> verticals_;

    /** The round's candidates in push order. */
    growing_buffer<sz::string_view_t, allocator_t> staged_;

    /** Every staged candidate's length in the kit's symbols, by push slot, measured per round. */
    growing_buffer<std::size_t, allocator_t> symbol_counts_;

    /** Two halves of the staging capacity: @c order_[rank] = slot, ascending by length, the radix passes
     *  bouncing between them. */
    growing_buffer<u32_t, allocator_t> order_;
    status_t last_status_ = status_t::uninitialized_k;

  public:
    levenshtein_metric(allocator_t const& allocator = {}) noexcept
        : masks_(allocator), class_table_(allocator), verticals_(allocator), staged_(allocator),
          symbol_counts_(allocator), order_(allocator) {}
    levenshtein_metric(levenshtein_metric const&) = delete;
    levenshtein_metric& operator=(levenshtein_metric const&) = delete;
    levenshtein_metric(levenshtein_metric&&) noexcept = default;
    levenshtein_metric& operator=(levenshtein_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging, the symbol counts and the ordering.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at least the
     *      index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or rounds saturate rather
     *      than fail. A round is bounded to @c u32_t slots by design.
     */
    status_t init(std::size_t staging_limit) noexcept {
        if (!staging_limit || staging_limit > (std::numeric_limits<u32_t>::max)()) return status_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        if (status_t const counted = symbol_counts_.resize_uninitialized(staged_.capacity()); failed(counted))
            return counted;
        last_status_ = order_.resize_uninitialized(2 * staged_.capacity());
        return last_status_;
    }

    /** @brief How the last @ref init or @ref reset fared - the only place a failure surfaces, since the scoring
     *      verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** @brief Opens a round for @p query, preparing its masks and discarding whatever the previous round staged. */
    void reset(sz::string_view_t query) noexcept {
        staged_.clear();
        query_ = sz_levenshtein_query_t {};
        last_status_ = status_t::success_k;
        if (query.empty()) return;
        // Grow only past the widest query seen: `reserve` keeps the allocation, the size then follows within it.
        std::size_t const symbols = kit_t::symbols(query.data(), query.size());
        std::size_t const scratch_verticals = levenshtein_scratch_verticals<kit_t>(sz_levenshtein_query_words(symbols));
        std::size_t const mask_entries = kit_t::mask_entries(symbols);
        std::size_t const class_table_bytes = kit_t::class_table_bytes(symbols);
        last_status_ = masks_.reserve(mask_entries);
        if (failed(last_status_)) return;
        last_status_ = class_table_.reserve(class_table_bytes);
        if (failed(last_status_)) return;
        last_status_ = verticals_.reserve(scratch_verticals);
        if (failed(last_status_)) return;
        masks_.resize_reserved(mask_entries);
        class_table_.resize_reserved(class_table_bytes);
        verticals_.resize_reserved(scratch_verticals);
        if (kit_t::prepare(query.data(), query.size(), masks_.data(), class_table_.data(), &query_) != sz_success_k)
            last_status_ = status_t::shape_mismatch_k;
    }

    /** @brief Stages one candidate, or refuses once the round is full. */
    basic::metric_staging_t push(sz::string_view_t value) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        staged_.push_back_reserved(value);
        return basic::metric_staging_t::staged_k;
    }

    /**
     *  @brief Scores everything staged in one sweep and closes the round. Distances up to and including
     *      @p radius are exact; above it the value is some value above the radius. A failed round reports
     *      the furthest distance, so a full pool rejects it and an unfilled pool admits it last; @ref status
     *      is the only signal.
     *
     *  In length order the candidates whose length gap alone exceeds the radius form a prefix and a suffix
     *  that never enter the kernel and report that gap - the suffix cut is what keeps a long candidate out
     *  of a register - and the rest stream through it. Lengths count the kit's symbols, each
     *  candidate's measured once per round.
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
            symbol_counts_[slot] = kit_t::symbols(staged_[slot].data(), staged_[slot].size());
        i64_t const query_length = static_cast<i64_t>(query_.length);
        if (query_length == 0) {
            for (std::size_t slot = 0; slot != count; ++slot)
                distances[slot] = static_cast<i64_t>(symbol_counts_[slot]);
            staged_.clear();
            return;
        }
        u32_t const* const order = order_by_length_(count);
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
        levenshtein_distances<kit_t>(query_, staged_.data(), symbol_counts_.data(), order + first_within,
                                     end_within - first_within, radius, distances.data(), verticals_.data());
        staged_.clear();
    }

  private:
    /**
     *  @brief The ascending-length permutation of the first @p count staged views: a stable LSD radix over the
     *      bytes of their symbol counts, low byte first, visiting only the bytes that differ across
     *      the round - so lengths under 256 cost one counting pass and one scatter, and nothing
     *      allocates or throws.
     *  @return Whichever half of @c order_ holds the finished permutation.
     */
    u32_t const* order_by_length_(std::size_t count) noexcept {
        u32_t* source = order_.data();
        u32_t* target = order_.data() + order_.size() / 2;
        std::size_t all_ones = ~std::size_t(0), any_ones = 0;
        for (std::size_t slot = 0; slot != count; ++slot) {
            source[slot] = static_cast<u32_t>(slot);
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
        return source;
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
 *  Nothing is stored per node, so the window widths may be chosen per corpus without invalidating anything. One
 *  instance belongs to one thread and owns its tree, its chain scratch and its staging. The staging is sized
 *  once by @c init and the tree by @c reset; the chain scratch grows only past the longest candidate seen, so
 *  a round allocates nothing once one as long has been scored. A distance is the windows the shorter side
 *  misses of the longer, in fixed point over @ref resolution_k: zero where the windows coincide at every
 *  width, and @ref resolution_k where they share none.
 *
 *  @tparam capability_ The admitted ISA backends as a capability mask; the widest kit this build compiled in
 *      among them is picked.
 *  @tparam allocator_type_ places every buffer, one set per metric.
 */
template <sz_capability_t capability_ = sz_cap_any_k, typename allocator_type_ = std::allocator<byte_t>>
class overlap_metric {
  public:
    using kit_t = overlap_kit<capability_>;
    using allocator_t = allocator_type_;
    using is_transparent = void;
    /** The full scale of a distance: the widest that still fits @c i64 times a @c u32 window count. */
    static constexpr i64_t resolution_k = i64_t(1) << 31;

  private:
    /** The round's candidates in push order. */
    growing_buffer<sz::string_view_t, allocator_t> staged_;
    /** The widths windows are taken at, all sharing one B-tree. */
    growing_buffer<std::size_t, allocator_t> window_widths_;
    /** Each width's @c sz_overlap_window_power, alongside @ref window_widths_. */
    growing_buffer<f64_t, allocator_t> window_powers_;
    /** The query's windows summed over every width: the denominator's query side. */
    std::size_t query_windows_ = 0;
    /** The prepared tree over the query's distinct window hashes, pointing into @ref nodes_. */
    sz_overlap_btree_t btree_ {};
    /** The tree's arena: the query's window hashes across every width are written at its start, sorted and
     *  deduplicated there, then laid out as nodes; @c sz_overlap_btree_entries of @ref query_windows_ wide. */
    growing_buffer<u32_t, allocator_t> nodes_;
    /** Prefix-hash chains: the query's during @ref reset, four candidates' of @c longest + 1 each in @ref drain. */
    growing_buffer<f64_t, allocator_t> prefix_hashes_;
    /** One candidate's window hashes at one width, before they are probed. */
    growing_buffer<u32_t, allocator_t> window_hashes_;
    /** How the last @ref init, @ref reset or @ref drain fared. */
    status_t last_status_ = status_t::uninitialized_k;

  public:
    overlap_metric(allocator_t const& allocator = {}) noexcept
        : staged_(allocator), window_widths_(allocator), window_powers_(allocator), nodes_(allocator),
          prefix_hashes_(allocator), window_hashes_(allocator) {}
    overlap_metric(overlap_metric const&) = delete;
    overlap_metric& operator=(overlap_metric const&) = delete;
    overlap_metric(overlap_metric&&) noexcept = default;
    overlap_metric& operator=(overlap_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging and records the window widths a round scores over.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at least the
     *      index's round capacity, @c max(fan_out × connectivity, connectivity + 1), or rounds saturate rather
     *      than fail.
     *  @param[in] window_widths Window widths in bytes, all sharing one B-tree; a width past a text
     *      contributes nothing.
     */
    status_t init(std::size_t staging_limit, std::span<std::size_t const> window_widths) noexcept {
        if (!staging_limit || window_widths.empty()) return status_t::config_mismatch_k;
        for (std::size_t const width : window_widths)
            if (!width) return status_t::config_mismatch_k;
        if (status_t const staged = staged_.reserve_exactly(staging_limit); failed(staged)) return staged;
        if (status_t const sized = window_widths_.resize_uninitialized(window_widths.size()); failed(sized))
            return sized;
        if (status_t const sized = window_powers_.resize_uninitialized(window_widths.size()); failed(sized))
            return sized;
        for (std::size_t index = 0; index != window_widths.size(); ++index) {
            window_widths_[index] = window_widths[index];
            window_powers_[index] = sz_overlap_window_power(window_widths[index]);
        }
        last_status_ = status_t::success_k;
        return last_status_;
    }

    /** How the last @ref init, @ref reset or @ref drain fared - the only place a failure surfaces, since the
     *  scoring verbs answer nothing by contract. */
    status_t status() const noexcept { return last_status_; }

    /** Opens a round for @p query, building its B-tree once and discarding whatever the previous round staged. */
    void reset(sz::string_view_t query) noexcept {
        last_status_ = status_t::success_k;
        staged_.clear(), query_windows_ = 0, btree_ = sz_overlap_btree_t {};
        if (query.empty()) return;
        for (std::size_t const width : window_widths_)
            if (width <= query.size()) query_windows_ += query.size() - width + 1;
        if (query_windows_ == 0) return;

        // Grow only past the longest query seen: `reserve` keeps the allocation, the size then follows within it.
        std::size_t const entries = sz_overlap_btree_entries(query_windows_);
        last_status_ = prefix_hashes_.reserve(query.size() + 1);
        if (failed(last_status_)) return;
        last_status_ = nodes_.reserve(entries);
        if (failed(last_status_)) return;
        prefix_hashes_.resize_reserved(query.size() + 1);
        nodes_.resize_reserved(entries);
        overlap_prefix_hashes<kit_t>(query.data(), query.size(), prefix_hashes_.data());

        // Every width's window hashes land in one array; a hash carries its width by construction, so a hit is
        // attributed to that width and a coincidence with another width's hash runs at 2^-32.
        std::size_t written = 0;
        for (std::size_t index = 0; index != window_widths_.size(); ++index) {
            std::size_t const width = window_widths_[index];
            if (width > query.size()) continue;
            std::size_t const windows = query.size() - width + 1;
            overlap_window_hashes<kit_t>(prefix_hashes_.data(), windows, width, window_powers_[index],
                                         nodes_.data() + written);
            written += windows;
        }
        std::size_t const distinct = kit_t::btree_sort(nodes_.data(), written);
        if (sz_overlap_btree_prepare(nodes_.data(), distinct, &btree_) != sz_success_k)
            last_status_ = status_t::shape_mismatch_k;
    }

    /** Stages one candidate, or refuses once the round is full. */
    basic::metric_staging_t push(sz::string_view_t value) noexcept {
        if (staged_.size() == staged_.capacity()) return basic::metric_staging_t::refused_k;
        staged_.push_back_reserved(value);
        return basic::metric_staging_t::staged_k;
    }

    /**
     *  @brief Scores everything staged in one sweep and closes the round. Every share is exact, so the radius
     *      goes unused. A failed round reports the furthest distance, so a full pool rejects it and an unfilled
     *      pool admits it last; @ref status is the only signal.
     *
     *  The chain scratch grows once to the longest staged candidate, then candidates score four at a time, their
     *  chains advancing together; a tail below four scores one at a time.
     *
     *  @param[out] distances One slot per staged candidate, in push order.
     */
    void drain(std::span<i64_t> distances, i64_t) noexcept {
        std::size_t const count = staged_.size();
        std::size_t longest = 0;
        for (std::size_t slot = 0; slot != count; ++slot) longest = max_of(longest, staged_[slot].size());
        std::size_t const stride = longest + 1;
        if (succeeded(last_status_) && query_windows_ != 0) {
            last_status_ = prefix_hashes_.reserve(stride * overlap_candidates_per_flight_k);
            if (succeeded(last_status_)) last_status_ = window_hashes_.reserve(longest);
        }
        if (failed(last_status_) || query_windows_ == 0) {
            for (std::size_t slot = 0; slot != count; ++slot)
                distances[slot] = failed(last_status_) ? (std::numeric_limits<i64_t>::max)() : resolution_k;
            staged_.clear();
            return;
        }
        prefix_hashes_.resize_reserved(stride * overlap_candidates_per_flight_k);
        window_hashes_.resize_reserved(longest);

        std::size_t slot = 0;
        for (; slot + overlap_candidates_per_flight_k <= count; slot += overlap_candidates_per_flight_k) {
            overlap_prefix_hashes_flight<kit_t>(staged_.data() + slot, stride, prefix_hashes_.data());
            for (std::size_t chain = 0; chain != overlap_candidates_per_flight_k; ++chain)
                distances[slot + chain] = distance_(staged_[slot + chain].size(),
                                                    prefix_hashes_.data() + chain * stride);
        }
        for (; slot != count; ++slot) {
            overlap_prefix_hashes<kit_t>(staged_[slot].data(), staged_[slot].size(), prefix_hashes_.data());
            distances[slot] = distance_(staged_[slot].size(), prefix_hashes_.data());
        }
        staged_.clear();
    }

  private:
    /** One candidate's distance off its finished chain: every width's window hashes probed against the tree, the
     *  misses over whichever side has more windows, scaled to @ref resolution_k. */
    i64_t distance_(std::size_t length, f64_t const* prefix_hashes) noexcept {
        std::size_t candidate_windows = 0, hits = 0;
        for (std::size_t index = 0; index != window_widths_.size(); ++index) {
            std::size_t const width = window_widths_[index];
            if (width > length) continue;
            std::size_t const windows = length - width + 1;
            overlap_window_hashes<kit_t>(prefix_hashes, windows, width, window_powers_[index], window_hashes_.data());
            hits += kit_t::btree_probe(&btree_, window_hashes_.data(), windows);
            candidate_windows += windows;
        }
        // A probe answers against the query's distinct windows, so hits count one side's occurrences and only
        // the longer side bounds them; a union denominator would let repeats score below zero.
        u64_t const longer = max_of<u64_t>(candidate_windows, query_windows_);
        return static_cast<i64_t>((longer - hits) * static_cast<u64_t>(resolution_k) / longer);
    }
};

using overlap_metric_t = overlap_metric<>;

#pragma endregion Overlap Metric

} // namespace unum::usearch::sequence
