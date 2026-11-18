/**
 *  @file include/usearch/sequence/vram_apple_metric.metal
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Metal kits for the sequence Apple indexes, the MSL twin of @c vram_nvidia_metric.cuh:
 *      Levenshtein over bytes and UTF-8 runes and window overlap through StringZilla's Metal
 *      prepared queries, and USearch's own Needleman-Wunsch over packed codes.
 *
 *  Each kit is the basic Apple walk's contract: rungs, lanes per entry, a query prepared in the
 *  walker's span, a lower bound from lengths and a score exact within the walk's radius. A
 *  Levenshtein rung is the Myers words of the longest query its kernels serve: a lone thread
 *  scores a candidate at the query's exact word count up to StringZilla's threaded ceiling, and a
 *  whole simdgroup past it. An overlap rung is the lanes rolling one candidate's windows. The
 *  Needleman-Wunsch kit scores a candidate with a simdgroup, an anti-diagonal wavefront down
 *  stripes of query codes, each lane holding a rung's worth of rows in registers, as the CUDA kit
 *  does step for step.
 *
 *  The rungs, the bounds of a sequence a kit serves and the layout of a Needleman-Wunsch query
 *  lead, in the C++ that MSL shares with every host compiler, so a rung the host picks is the rung
 *  the kernels score on: the faces include this file for them alone, as Metal defines
 *  @c __METAL_VERSION__ only on the device side, and splice it whole after StringZilla's Metal
 *  prelude, its Levenshtein and its overlap sources.
 */
#pragma once

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#endif

namespace unum::usearch::sequence {

using namespace unum::usearch::utilities;

/** Rungs of each sequence kit, narrowest first, as many as the NVIDIA kits climb. */
enum metal_sequence_rungs_t : unsigned int {

    /** Levenshtein: twice the Myers words a rung, from two up to sixty-four, 4 KiB of symbols. */
    metal_levenshtein_rungs_k = 6,

    /** Overlap: one lane alone, then four, eight, sixteen and a whole simdgroup per sequence. */
    metal_overlap_rungs_k = 5,

    /** Needleman-Wunsch: one to eight query codes a lane holds. */
    metal_needleman_wunsch_rungs_k = 8,
};

/** The bounds of a sequence a kit serves and the stripes a Needleman-Wunsch query takes. */
enum metal_sequence_extent_t : unsigned int {

    /** Words a lone thread holds, StringZilla Metal's threaded ceiling; wider queries take a
     *  simdgroup. */
    metal_levenshtein_thread_words_max_k = 15,

    /** Codes of the longest sequence the widest Needleman-Wunsch rung scores, as many as the widest
     *  Levenshtein rung's bytes; longer ones are refused. */
    metal_needleman_wunsch_widest_codes_k = 4096,

    /** Classes a Needleman-Wunsch alphabet folds into at most, five bits a code. */
    metal_needleman_wunsch_classes_k = 32,

    /** Bytes of one Needleman-Wunsch profile stripe: a 64-bit word of costs per class and lane. */
    metal_needleman_wunsch_stripe_bytes_k = 32 * 32 * 8,

    /** Bytes of the folded costs: the code bits, the classes, the gap, then the square of
     *  substitutions, as @c needleman_wunsch_costs_t lays them out. */
    metal_needleman_wunsch_costs_bytes_k = 3 + 32 * 32,
};

/** Myers words of the longest query Levenshtein rung @p rung scores. */
constexpr unsigned int metal_levenshtein_words(unsigned int rung) { return 2u << rung; }

/** Lanes rolling one sequence's windows on overlap rung @p rung. */
constexpr unsigned int metal_overlap_lanes(unsigned int rung) { return rung ? 2u << rung : 1u; }

/** Query codes each lane holds on Needleman-Wunsch rung @p rung. */
constexpr unsigned int metal_needleman_wunsch_rows(unsigned int rung) { return rung + 1; }

/** Query codes each lane holds for a query of @p codes: the fewest that fit the query in one stripe
 *  of a simdgroup's lanes, the widest rung's past it. */
constexpr unsigned int metal_needleman_wunsch_rows_per_lane(unsigned int codes) {
    unsigned int const fitting = divide_round_up(codes, 32u);
    unsigned int const widest = metal_needleman_wunsch_rows(metal_needleman_wunsch_rungs_k - 1);
    return fitting < 1 ? 1 : fitting > widest ? widest : fitting;
}

/** Stripes a query of @p codes takes, one at least. */
constexpr unsigned int metal_needleman_wunsch_stripes(unsigned int codes) {
    unsigned int const stripe_rows = 32 * metal_needleman_wunsch_rows_per_lane(codes);
    unsigned int const stripes = divide_round_up(codes, stripe_rows);
    return stripes ? stripes : 1;
}

/** Bytes a query of @p codes takes once prepared: its profile stripes, then, past one stripe, a
 *  boundary row of the widest codes per simdgroup of a threadgroup. */
constexpr unsigned int metal_needleman_wunsch_span_bytes(unsigned int codes) {
    unsigned int const stripes = metal_needleman_wunsch_stripes(codes);
    unsigned int const boundaries = stripes > 1 ? metal_simdgroups_k * (metal_needleman_wunsch_widest_codes_k + 1) * 4
                                                : 0;
    return stripes * metal_needleman_wunsch_stripe_bytes_k + boundaries;
}

} // namespace unum::usearch::sequence

#if defined(__METAL_VERSION__)

using namespace metal;

namespace unum::usearch::sequence {

#pragma region Levenshtein and Overlap

/** Words from one class row of a prepared query to the next, as @c sz_levenshtein_walk_stride
 *  pads the words of a @p bytes long query. */
inline ulong metal_levenshtein_walk_stride(uint bytes) { return ulong(divide_round_up(bytes, 64u)) | 1ul; }

/** Scores a lone thread's candidate at the query's exact word count, through the scorer of the
 *  first count from @p words_ down that matches it, as the host launches no rung narrower than the
 *  longest query it serves. */
template <uint words_, bool runes_>
struct metal_levenshtein_thread {
    static long score(thread sz_levenshtein_query_metal_t const& query, device uchar const* data, uint bytes,
                      long limit, uint words) {
        if (words == words_) return sz_levenshtein_group_score_metal_<words_, 1, runes_>(query, data, bytes, limit, 0);
        return metal_levenshtein_thread<words_ - 1, runes_>::score(query, data, bytes, limit, words);
    }
};

template <bool runes_>
struct metal_levenshtein_thread<0, runes_> {
    static long score(thread sz_levenshtein_query_metal_t const&, device uchar const*, uint, long, uint) { return 0; }
};

/**
 *  @brief Levenshtein over bytes or, with @p runes_, over UTF-8 runes, through StringZilla's Metal
 *      Myers scorers, the query's classes and masks in the span its walk prepares.
 *
 *  Up to the threaded ceiling a lane scores a candidate alone at the query's exact word count, so
 *  the verticals stay in registers; past it the lanes of a simdgroup split a candidate's words,
 *  and a threadgroup's four simdgroups score a hop's candidates together. An empty side scores
 *  the symbols of the other one.
 */
template <bool runes_>
struct metal_levenshtein_kit {
    using query_t = sz_levenshtein_query_metal_t;

    /** Myers words of the longest query each rung's kernels score, narrowest first. */
    static constexpr constant uint rungs_k[metal_levenshtein_rungs_k] = {
        metal_levenshtein_words(0), metal_levenshtein_words(1), metal_levenshtein_words(2),
        metal_levenshtein_words(3), metal_levenshtein_words(4), metal_levenshtein_words(5)};

    static constexpr uint lanes_per_row(uint words) {
        return words > metal_levenshtein_thread_words_max_k ? metal_simdgroup_width_k : 1;
    }

    static query_t prepare(device uchar const*, device uchar const* data, uint bytes, device uchar* span, uint lane) {
        query_t query;
        if (runes_) sz_levenshtein_query_prepare_utf8_metal_(data, bytes, span, query, lane);
        else sz_levenshtein_query_prepare_metal_(data, bytes, metal_levenshtein_walk_stride(bytes), span, query, lane);
        return query;
    }

    /** No edit sequence is shorter than the two lengths' difference; a sequence of @c b bytes
     *  holds between a quarter of @c b runes and @c b of them. */
    static float lower_bound(uint, thread query_t const& query, device uchar const*, uint bytes) {
        ulong const fewest = runes_ ? divide_round_up(bytes, 4u) : bytes, most = bytes;
        return float(fewest > query.length ? fewest - query.length : query.length > most ? query.length - most : 0);
    }

    template <uint words_>
    static float score(uint, thread query_t const& query, device uchar const* data, uint bytes, float radius, uint lane,
                       uint) {
        if (!query.length || !bytes) {
            ulong symbols = bytes;
            if (runes_)
                for (ulong position = symbols = 0; position < bytes; ++symbols)
                    sz_levenshtein_next_rune_metal_(data, bytes, position);
            return float(query.length + symbols);
        }
        long const limit = radius < 0x1p62f ? long(radius) : numeric_limits<long>::max();
        if constexpr (lanes_per_row(words_) == 1) {
            uint const words = uint(divide_round_up(uint(query.length), 64u));
            return float(metal_levenshtein_thread<words_, runes_>::score(query, data, bytes, limit, words));
        }
        else {
            constexpr uint words_per_lane_k = (words_ + metal_simdgroup_width_k - 1u) / metal_simdgroup_width_k;
            return float(sz_levenshtein_group_score_metal_<words_per_lane_k, metal_simdgroup_width_k, runes_>(
                query, data, bytes, limit, lane));
        }
    }
};

using metal_levenshtein_bytes_kit_t = metal_levenshtein_kit<false>;
using metal_levenshtein_utf8_kit_t = metal_levenshtein_kit<true>;

/** The fixed point the overlap distance takes, as @c overlap_resolution_k: the share of the
 *  longer side's windows a candidate misses, times 2^31. */
constant constexpr ulong metal_overlap_resolution_k = 1ul << 31;

/** The overlap distance of @p matches among @p longer windows, as @c overlap_distance scores
 *  it. */
inline ulong metal_overlap_distance(ulong longer, ulong matches) {
    return longer ? (longer - matches) * metal_overlap_resolution_k / longer : metal_overlap_resolution_k;
}

/**
 *  @brief Window overlap: a simdgroup lays the query's window hashes at every width out as a tree
 *      in its span, and a group of a rung's lanes counts each candidate's windows the tree holds,
 *      through StringZilla's rolling hashes, whose residues are the CPU kits'.
 *
 *  The widths are the parameters the face uploads, their count first.
 */
struct metal_overlap_kit_t {
    using query_t = sz_overlap_query_metal_t;

    /** Lanes rolling one sequence's windows, narrowest first. */
    static constexpr constant uint rungs_k[metal_overlap_rungs_k] = {metal_overlap_lanes(0), metal_overlap_lanes(1),
                                                                     metal_overlap_lanes(2), metal_overlap_lanes(3),
                                                                     metal_overlap_lanes(4)};

    /** A rung is its lanes per sequence. */
    static constexpr uint lanes_per_row(uint lanes) { return lanes; }

    static query_t prepare(device uchar const* parameters, device uchar const* data, uint bytes, device uchar* span,
                           uint lane) {
        device uint const* const widths = (device uint const*)parameters;
        query_t query;
        sz_overlap_query_prepare_metal_(data, bytes, widths + 1, widths[0], span, query, lane);
        return query;
    }

    /** No candidate matches more windows than the shorter side holds. */
    static float lower_bound(uint, thread query_t const& query, device uchar const*, uint bytes) {
        ulong const candidate_windows = sz_overlap_windows_metal_(bytes, query.widths, query.widths_count);
        ulong const longer = max(candidate_windows, ulong(query.windows));
        ulong const shorter = min(candidate_windows, ulong(query.windows));
        return float(metal_overlap_distance(longer, shorter));
    }

    /** A float radius rounds, so the misses StringZilla may stop past are those of the integer
     *  radius a float spacing above it, @c 2^7 at this scale. */
    template <uint lanes_>
    static float score(uint, thread query_t const& query, device uchar const* data, uint bytes, float radius, uint lane,
                       uint) {
        ulong const candidate_windows = sz_overlap_windows_metal_(bytes, query.widths, query.widths_count);
        ulong const longer = max(candidate_windows, ulong(query.windows));
        if (!longer) return float(metal_overlap_distance(longer, 0));
        ulong misses_limit = numeric_limits<ulong>::max();
        if (radius < float(metal_overlap_resolution_k)) {
            ulong const admitted = ulong(radius > 0 ? radius : 0) + (1ul << 7);
            misses_limit = ((admitted + 1) * longer - 1) / metal_overlap_resolution_k;
        }
        ulong const matches = sz_overlap_candidate_matches_metal_(query, data, bytes, misses_limit, lane % lanes_,
                                                                  lanes_);
        return float(metal_overlap_distance(longer, matches));
    }
};

#pragma endregion Levenshtein and Overlap

#pragma region Needleman Wunsch

/** The folded costs the face uploads, laid out as @c needleman_wunsch_costs_t. */
struct metal_needleman_wunsch_costs_t {
    uchar code_bits, classes, gap;
    uchar substitutions[metal_needleman_wunsch_classes_k][metal_needleman_wunsch_classes_k];
};

/** Codes the packed @p bytes long @p sequence holds, as @c packed_codes_count counts them: as
 *  many as its bytes past the first fit at @p code_bits apiece, less the unused tail slots that
 *  first byte counts. */
inline uint metal_packed_codes_count(device uchar const* sequence, uint bytes, uint code_bits) {
    if (!bytes) return 0;
    uint const slots = (bytes - 1) * 8 / code_bits, unused = sequence[0];
    return slots > unused ? slots - unused : 0;
}

/** The code at @p index of the packed @p sequence, as @c packed_code_at reads it: little-endian
 *  from the byte past its first, reading the second byte only when the code crosses into it. */
inline uint metal_packed_code_at(device uchar const* sequence, uint index, uint code_bits) {
    uint const bit = index * code_bits, byte = 1 + (bit >> 3), shift = bit & 7u;
    uint word = sequence[byte];
    if (shift + code_bits > 8) word |= uint(sequence[byte + 1]) << 8;
    return (word >> shift) & ((1u << code_bits) - 1u);
}

/** A query a Needleman-Wunsch walk scores rows against, laid out in its span. */
struct metal_needleman_wunsch_query_t {

    /** The span: a profile stripe per stripe of the query, the cost of every class against each
     *  lane's codes, then, past one stripe, a boundary row per simdgroup of a threadgroup. */
    device uchar* span;

    /** Codes of the query. */
    uint codes;

    /** Query codes each lane holds, at most its rung's. */
    uint rows_per_lane;

    /** Bits each packed code of the query and its candidates takes. */
    uint bits_per_code;

    /** Cost of one code facing a gap. */
    uint gap;
};

/** Lays the packed @p data out as a query in @p span under the costs at @p parameters, every lane
 *  of the simdgroup writing its own profile words. */
inline metal_needleman_wunsch_query_t metal_needleman_wunsch_prepare(device uchar const* parameters,
                                                                     device uchar const* data, uint bytes,
                                                                     device uchar* span, uint lane) {
    constexpr uint lanes_k = metal_simdgroup_width_k, classes_k = metal_needleman_wunsch_classes_k;
    device metal_needleman_wunsch_costs_t const& costs = *(device metal_needleman_wunsch_costs_t const*)parameters;
    metal_needleman_wunsch_query_t query;
    query.span = span, query.bits_per_code = costs.code_bits, query.gap = costs.gap;
    query.codes = metal_packed_codes_count(data, bytes, costs.code_bits);
    query.rows_per_lane = metal_needleman_wunsch_rows_per_lane(query.codes);
    uint const stripes = metal_needleman_wunsch_stripes(query.codes);
    device ulong* const profile = (device ulong*)span;
    // The span may still serve the previous query's readers until every lane gets here.
    simdgroup_barrier(mem_flags::mem_device);
    for (uint stripe = 0; stripe != stripes; ++stripe) {
        uint const lane_first = (stripe * lanes_k + lane) * query.rows_per_lane;
        // The lane's codes in this stripe, none past the query's last.
        uint const held = lane_first < query.codes ? min(query.rows_per_lane, query.codes - lane_first) : 0u;
        for (uint code = 0; code != classes_k; ++code) {
            ulong costs_word = 0;
            for (uint row_in_lane = 0; row_in_lane != held; ++row_in_lane)
                costs_word |= ulong(costs.substitutions[metal_packed_code_at(data, lane_first + row_in_lane,
                                                                             costs.code_bits)][code])
                              << (8 * row_in_lane);
            profile[(stripe * classes_k + code) * lanes_k + lane] = costs_word;
        }
    }
    simdgroup_barrier(mem_flags::mem_device);
    return query;
}

/** The word lane @p lane loads for the first lane per 32 columns from @p first_column : the top
 *  cell of its column, the boundary row's past the first stripe, beside the next column's code. */
inline uint metal_needleman_wunsch_loaded(device uchar const* data, device uint const* boundary, uint first_column,
                                          uint lane, uint columns, uint stripe, uint gap, uint code_bits) {
    uint const column = first_column + lane;
    uint const top = column > columns ? 0u : stripe ? boundary[column] : column * gap;
    uint const next_code = column < columns ? metal_packed_code_at(data, column, code_bits) : 0u;
    return top << 5 | next_code;
}

/**
 *  @brief The Needleman-Wunsch cost from @p query to the packed @p data, a simdgroup's
 *      anti-diagonal wavefront over stripes of query codes, @p rows_ codes at most to a lane:
 *      exact within @p radius, and past it a stripe's bottom row minimum above it.
 *
 *  Lane @c l holds a stripe's rows from its @c l -th share down and enters column @c j at step
 *  @c j+l, taking the cell above and the next column's code from lane @c l-1 in one shuffled
 *  word, the first lane from a word the simdgroup loads per 32 columns. Each stripe hands its
 *  bottom row to the next through the simdgroup's boundary row, which the last lane writes 31
 *  columns behind the simdgroup's reads, as the CUDA kit does; the simdgroup's lanes step
 *  together, so no column is written before it is read.
 */
template <uint rows_>
inline float metal_needleman_wunsch_score(thread metal_needleman_wunsch_query_t const& query, device uchar const* data,
                                          uint bytes, float radius, uint lane, uint simdgroup) {
    constexpr uint lanes_k = metal_simdgroup_width_k, classes_k = metal_needleman_wunsch_classes_k;
    uint const gap = query.gap, rows = query.codes, rows_per_lane = query.rows_per_lane;
    uint const code_bits = query.bits_per_code;
    uint const columns = metal_packed_codes_count(data, bytes, code_bits);
    if (!rows || !columns) return float((rows + columns) * gap);
    uint const stripe_rows = lanes_k * rows_per_lane, stripes = metal_needleman_wunsch_stripes(rows);
    device ulong const* const profile = (device ulong const*)query.span;
    device uint* const boundary = (device uint*)(query.span + stripes * metal_needleman_wunsch_stripe_bytes_k) +
                                  simdgroup * (metal_needleman_wunsch_widest_codes_k + 1);
    uint result = 0;
    for (uint stripe = 0; stripe != stripes; ++stripe) {
        uint const first_row = stripe * stripe_rows, lane_first = first_row + lane * rows_per_lane;
        uint const active = min(lanes_k, divide_round_up(rows - first_row, rows_per_lane));
        uint const last_lane = (rows - 1 - first_row) / rows_per_lane,
                   last_cell = (rows - 1 - first_row) % rows_per_lane;
        bool const closing = stripe + 1 == stripes;
        uint cells[rows_];
        for (uint cell = 0; cell != rows_; ++cell) cells[cell] = (lane_first + cell + 1) * gap;
        device ulong const* const stripe_profile = profile + stripe * classes_k * lanes_k + lane;
        // A cell rides one word with the code its receiver scores a step later, the cell in the
        // high bits, as far as 2^27 reaches; so each step loads the next step's costs first.
        uint const first_code = metal_packed_code_at(data, 0, code_bits);
        uint handed = lane == 0 ? first_code : 0u;
        uint loaded = metal_needleman_wunsch_loaded(data, boundary, 1, lane, columns, stripe, gap, code_bits);
        ulong costs_word = stripe_profile[first_code * lanes_k];
        uint diagonal_above = lane_first * gap, bottom_minimum = (stripe + 1) * stripe_rows * gap;
        uint const steps = columns + active - 1;
        for (uint step = 0; step != steps; ++step) {
            if (step % lanes_k == 0 && step)
                loaded = metal_needleman_wunsch_loaded(data, boundary, step + 1, lane, columns, stripe, gap, code_bits);
            uint const from_loaded = simd_shuffle(loaded, ushort(step % lanes_k));
            uint const from_above = simd_shuffle_up(handed, ushort(1));
            uint const incoming = lane == 0 ? from_loaded : from_above;
            uint const next_code = incoming & 31u, top = incoming >> 5;
            ulong const next_costs_word = stripe_profile[next_code * lanes_k];
            uint const column = step - lane + 1;
            // A lane outside the wavefront hands the cell above on unchanged.
            if (lane >= active || step < lane || column > columns) {
                handed = top << 5 | next_code, costs_word = next_costs_word;
                continue;
            }
            uint above = top, diagonal = diagonal_above;
            // Bounded by the rung too, so the cells stay in registers.
            for (uint cell = 0; cell != rows_ && cell != rows_per_lane; ++cell) {
                uint const substitution = uint(costs_word >> (8 * cell)) & 0xFFu;
                uint const opened = min(diagonal + substitution, cells[cell] + gap);
                uint const scored = min(opened, above + gap);
                diagonal = cells[cell], cells[cell] = scored, above = scored;
            }
            diagonal_above = top;
            if (!closing && lane == lanes_k - 1) boundary[column] = above, bottom_minimum = min(bottom_minimum, above);
            if (closing && lane == last_lane && column == columns) result = cells[last_cell];
            handed = above << 5 | next_code, costs_word = next_costs_word;
        }
        simdgroup_barrier(mem_flags::mem_device);
        if (closing) return float(simd_shuffle(result, ushort(last_lane)));
        // Every alignment crosses a stripe's bottom row, so none ends below its minimum.
        uint const crossing = simd_shuffle(bottom_minimum, ushort(lanes_k - 1));
        if (float(crossing) > radius) return float(crossing);
    }
    return float(result);
}

/** Needleman-Wunsch over packed codes: a simdgroup to a sequence, an anti-diagonal wavefront down
 *  stripes of the query, its costs a profile the walk lays out in its span from the folded table
 *  the face uploads as the parameters. */
struct metal_needleman_wunsch_kit_t {
    using query_t = metal_needleman_wunsch_query_t;

    /** Query codes each lane holds, narrowest first. */
    static constexpr constant uint rungs_k[metal_needleman_wunsch_rungs_k] = {
        metal_needleman_wunsch_rows(0), metal_needleman_wunsch_rows(1), metal_needleman_wunsch_rows(2),
        metal_needleman_wunsch_rows(3), metal_needleman_wunsch_rows(4), metal_needleman_wunsch_rows(5),
        metal_needleman_wunsch_rows(6), metal_needleman_wunsch_rows(7)};

    /** Every rung scores a sequence with the whole simdgroup. */
    static constexpr uint lanes_per_row(uint) { return metal_simdgroup_width_k; }

    static query_t prepare(device uchar const* parameters, device uchar const* data, uint bytes, device uchar* span,
                           uint lane) {
        return metal_needleman_wunsch_prepare(parameters, data, bytes, span, lane);
    }

    /** Every code one side holds past the other's length faces a gap. */
    static float lower_bound(uint, thread query_t const& query, device uchar const* data, uint bytes) {
        uint const columns = metal_packed_codes_count(data, bytes, query.bits_per_code);
        uint const difference = columns > query.codes ? columns - query.codes : query.codes - columns;
        return float(difference * query.gap);
    }

    template <uint rows_>
    static float score(uint, thread query_t const& query, device uchar const* data, uint bytes, float radius, uint lane,
                       uint simdgroup) {
        return metal_needleman_wunsch_score<rows_>(query, data, bytes, radius, lane, simdgroup);
    }
};

#pragma endregion Needleman Wunsch

} // namespace unum::usearch::sequence

#endif // defined(__METAL_VERSION__)
