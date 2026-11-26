/**
 *  @file include/usearch/sequence/vram_apple_metric.metal
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Metal kits for the sequence Apple indexes, the MSL twin of @c vram_nvidia_metric.cuh:
 *      USearch's own alignment over packed codes, and banded overlap and window containment
 *      through StringZilla's Metal prepared queries.
 *
 *  Each kit is the basic Apple walk's contract: rungs, lanes per entry, a query prepared in the
 *  walker's span, a lower bound from lengths and a score exact within the walk's radius. A
 *  containment rung is the lanes rolling one candidate's windows. The banded overlap kit scores a
 *  candidate with one lane, which votes a shift through the query's window table and aligns the
 *  band around it, the query beside its complement image, and the lower of the two distances
 *  stands. The alignment kit scores a candidate with a simdgroup, an anti-diagonal wavefront down
 *  stripes of query codes, each lane holding a rung's worth of rows in registers, as the CUDA kit
 *  does step for step.
 *
 *  A sketching kit also carries the sketch its builders score, as many bytes as the launch names:
 *  a simdgroup writes one sequence's sketch, each lane its share of the columns or the slots in
 *  passes, and one lane scores a pair of sketches. Alignment sketches smoothly and containment by
 *  hashes; banded overlap has no sketch and builds by its canonical score.
 *
 *  The rungs, the bounds of a sequence a kit serves and the layout of an alignment query lead, in
 *  the C++ that MSL shares with every host compiler, so a rung the host picks is the rung the
 *  kernels score on: the faces include this file for them alone, as Metal defines
 *  @c __METAL_VERSION__ only on the device side, and splice it whole after StringZilla's Metal
 *  prelude and its n-grams source.
 */
#pragma once

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#endif

namespace unum::usearch::sequence {

using namespace unum::usearch::utilities;

/** Rungs of each sequence kit, narrowest first, as many as the NVIDIA kits climb. */
enum metal_sequence_rungs_t : unsigned int {

    /** Containment: one lane alone, then four, eight, sixteen and a whole simdgroup per
     *  sequence. */
    metal_containment_canonical_rungs_count_k = 5,

    /** Banded overlap: a lane to a sequence. */
    metal_banded_overlap_canonical_rungs_count_k = 1,

    /** Alignment: one to eight query codes a lane holds. */
    metal_alignment_canonical_rungs_count_k = 8,
};

/** The bounds of a sequence a kit serves and the stripes an alignment query takes. */
enum metal_sequence_extent_t : unsigned int {

    /** Codes of the longest sequence the widest alignment rung scores, the most a power of two
     *  whose costliest alignment a 16-bit cell holds; longer ones are refused. */
    metal_alignment_widest_codes_k = 4096,

    /** Classes an alignment alphabet folds into at most, as @c alignment_classes_k. */
    metal_alignment_classes_k = 32,

    /** Bits of a packed alignment code, as @c packed_code_bits_k. */
    metal_packed_code_bits_k = 5,

    /** Bytes of one alignment profile stripe: a 64-bit word of costs per class and lane. */
    metal_alignment_stripe_bytes_k = 32 * 32 * 8,

    /** Bytes of the folded costs: the classes, the gap and the complement's mark, the square of
     *  substitutions, then the complement of every class, as @c alignment_costs_t lays them out. */
    metal_alignment_costs_bytes_k = 3 + 32 * 32 + 32,

    /** Symbols a smooth sketch's window spans at most, as @c spectrum_width_max_k. */
    metal_spectrum_width_max_k = 12,

    /** Entries of a smooth sketch's cosine table, as @c spectrum_cosines_k. */
    metal_spectrum_cosines_k = 1024,

    /** Bytes from the start of a smooth sketch's tables to their steps, as
     *  @c spectrum_steps_offset_k : the head, padded to a cache line. */
    metal_spectrum_steps_offset_k = 1088,

    /** The narrow window width a containment counts and its hash sketch samples, as
     *  @c containment_window_widths_k. */
    metal_containment_narrow_window_k = 5,

    /** The wide window width a containment counts and its hash sketch samples, as
     *  @c containment_window_widths_k. */
    metal_containment_wide_window_k = 8,
};

/** Bytes from the start of a smooth sketch's tables to their phases, behind @p columns steps, as
 *  @c spectrum_phases_offset. */
constexpr unsigned int metal_spectrum_phases_offset(unsigned int columns) {
    return metal_spectrum_steps_offset_k + divide_round_up(columns * 2, 64u) * 64;
}

/** Lanes rolling one sequence's windows on containment rung @p rung. */
constexpr unsigned int metal_containment_lanes(unsigned int rung) { return rung ? 2u << rung : 1u; }

/** Query codes each lane holds on alignment rung @p rung. */
constexpr unsigned int metal_alignment_rows(unsigned int rung) { return rung + 1; }

/** Query codes each lane holds for a query of @p codes: the fewest that fit the query in one stripe
 *  of a simdgroup's lanes, the widest rung's past it. */
constexpr unsigned int metal_alignment_rows_per_lane(unsigned int codes) {
    unsigned int const fitting = divide_round_up(codes, 32u);
    unsigned int const widest = metal_alignment_rows(metal_alignment_canonical_rungs_count_k - 1);
    return fitting < 1 ? 1 : fitting > widest ? widest : fitting;
}

/** Stripes a query of @p codes takes, one at least. */
constexpr unsigned int metal_alignment_stripes(unsigned int codes) {
    unsigned int const stripe_rows = 32 * metal_alignment_rows_per_lane(codes);
    unsigned int const stripes = divide_round_up(codes, stripe_rows);
    return stripes ? stripes : 1;
}

/** Bytes a query of @p codes takes once prepared: its profile stripes, then, past one stripe, a
 *  boundary row of the widest codes per simdgroup of a threadgroup. */
constexpr unsigned int metal_alignment_span_bytes(unsigned int codes) {
    unsigned int const stripes = metal_alignment_stripes(codes);
    unsigned int const boundaries = stripes > 1 ? metal_simdgroups_k * (metal_alignment_widest_codes_k + 1) * 4 : 0;
    return stripes * metal_alignment_stripe_bytes_k + boundaries;
}

} // namespace unum::usearch::sequence

#if defined(__METAL_VERSION__)

using namespace metal;

namespace unum::usearch::sequence {

#pragma region Sketches and Complements

/** The folded costs the face uploads, laid out as @c alignment_costs_t: the complement's mark and,
 *  past the substitutions, the class each class pairs with on the other strand. */
struct metal_alignment_costs_t {
    uchar classes, gap, complements;
    uchar substitutions[metal_alignment_classes_k][metal_alignment_classes_k];
    uchar complemented[metal_alignment_classes_k];
};

/** What the window kits score with, laid out as @c window_parameters_t: the folded costs with
 *  the complement, then the window widths behind their count. */
struct metal_window_parameters_t {
    metal_alignment_costs_t costs;
    ulong widths_count;
    ulong widths[sz_ngrams_widths_max_metal_k];
};

/** The tables of a kit reading none, as @c std::monostate. */
struct metal_no_tables_t {};

/** The head of the tables a smooth sketch reads, laid out as @c spectrum_head_t. */
struct metal_spectrum_head_t {
    char cosines[metal_spectrum_cosines_k];
    uint classes, width, reference, shift, columns;
};

/** The tables a smooth sketch reads, borrowed from their buffer, as @c spectrum_view_t: the head,
 *  a step per column, and a row of @c columns phases per class at every place of a window, rows
 *  end to end. */
struct metal_spectrum_view_t {
    device metal_spectrum_head_t const* head;
    device ushort const* steps;
    device ushort const* phases;
};

/** The tables laid out at @p tables, as @c spectrum_view_of reads them. */
inline metal_spectrum_view_t metal_spectrum_view_of(device uchar const* tables) {
    device metal_spectrum_head_t const* const head = (device metal_spectrum_head_t const*)tables;
    metal_spectrum_view_t view;
    view.head = head;
    view.steps = (device ushort const*)(tables + metal_spectrum_steps_offset_k);
    view.phases = (device ushort const*)(tables + metal_spectrum_phases_offset(head->columns));
    return view;
}

/** Columns of a smooth sketch one lane sums in a pass, its sums, angles and steps in registers, as
 *  @c cuda_spectrum_lane_columns_k. */
constant constexpr uint metal_spectrum_lane_columns_k = 16;

/** Codes the packed @p bytes long @p sequence holds, as @c packed_codes_count counts them: as
 *  many as its bytes past the first fit, less the unused tail slots that first byte counts. */
inline uint metal_packed_codes_count(device uchar const* sequence, uint bytes) {
    if (!bytes) return 0;
    uint const slots = (bytes - 1) * 8 / metal_packed_code_bits_k, unused = sequence[0];
    return slots > unused ? slots - unused : 0;
}

/** The code at @p index of the packed @p sequence, as @c packed_code_at reads it: little-endian
 *  from the byte past its first, reading the second byte only when the code crosses into it. */
inline uint metal_packed_code_at(device uchar const* sequence, uint index) {
    uint const bit = index * metal_packed_code_bits_k, byte = 1 + (bit >> 3), shift = bit & 7u;
    uint word = sequence[byte];
    if (shift + metal_packed_code_bits_k > 8) word |= uint(sequence[byte + 1]) << 8;
    return (word >> shift) & ((1u << metal_packed_code_bits_k) - 1u);
}

/** Adds the window whose class codes @p classes hold to a lane's @p sums at its @p owned columns
 *  from @p first, and steps their @p angles by @p deltas. */
inline void metal_sketch_window(metal_spectrum_view_t tables, thread uchar const (&classes)[metal_spectrum_width_max_k],
                                uint first, uint owned, thread uint (&sums)[metal_spectrum_lane_columns_k],
                                thread uint (&angles)[metal_spectrum_lane_columns_k],
                                thread uint const (&deltas)[metal_spectrum_lane_columns_k]) {
    constexpr uint share_k = metal_spectrum_lane_columns_k;
    device metal_spectrum_head_t const& head = *tables.head;
    for (uint column = 0; column != share_k; ++column) {
        if (column >= owned) break;
        uint phase = angles[column] >> 16;
        for (uint place = 0; place != head.width; ++place) {
            ulong const row = ulong(place * head.classes + classes[place]) * head.columns;
            phase += tables.phases[row + first + column];
        }
        sums[column] += uint(int(head.cosines[(phase & 0xFFFFu) >> 6]));
        angles[column] += deltas[column];
    }
}

/** Writes at @p sketch lane @p lane 's shares of the @p columns -byte smooth sketch of the packed
 *  class codes at @p sequence, every window of the codes in the order @c sketch_packed_codes sums
 *  them, as @c spectrum_step_serial sums every column; every lane of a simdgroup calls it, each
 *  summing its own consecutive columns of a pass, so no lane waits on another, and a sketch wider
 *  than a simdgroup's pass takes several. */
inline void metal_sketch_packed(metal_spectrum_view_t tables, device uchar const* sequence, uint bytes, uint lane,
                                device uchar* sketch, uint columns) {
    constexpr uint share_k = metal_spectrum_lane_columns_k;
    device metal_spectrum_head_t const& head = *tables.head;
    uint const width = head.width;
    uint const codes = metal_packed_codes_count(sequence, bytes);
    uint const windows = codes >= width ? codes - width + 1 : 0;
    // A step under half a turn times any reference stays under 2⁶³, so the product never wraps.
    ulong const span = ulong(head.reference) << 16;
    uint const rounding = (1u << head.shift) >> 1;
    for (uint pass = 0; pass < columns; pass += metal_simdgroup_width_k * share_k) {
        // A lane whose share starts past the last column owns none.
        uint const first = pass + lane * share_k;
        uint const owned = first < columns ? min(columns - first, share_k) : 0u;
        uint sums[share_k] = {0}, angles[share_k] = {0}, deltas[share_k] = {0};
        for (uint column = 0; column != share_k; ++column)
            if (column < owned && windows) deltas[column] = uint(ulong(tables.steps[first + column]) * span / windows);
        uchar classes[metal_spectrum_width_max_k] = {0};
        for (uint index = 0; index != codes; ++index) {
            for (uint place = 0; place + 1 < width; ++place) classes[place] = classes[place + 1];
            classes[width - 1] = uchar(metal_packed_code_at(sequence, index));
            if (index + 1 < width || !owned) continue;
            metal_sketch_window(tables, classes, first, owned, sums, angles, deltas);
        }
        for (uint column = 0; column != share_k; ++column) {
            if (column >= owned) break;
            sketch[first + column] = uchar(char(clamp(int(sums[column] + rounding) >> head.shift, -128, 127)));
        }
    }
}

/** The squared Euclidean distance between the @p bytes -byte smooth sketches at @p first and
 *  @p second, both four-byte aligned: the calling lane alone reads each a word at a time and sums
 *  in 32 bits. */
inline float metal_smooth_distance(device uchar const* first, device uchar const* second, uint bytes) {
    device uint const* const first_words = (device uint const*)first;
    device uint const* const second_words = (device uint const*)second;
    uint sum = 0;
    for (uint word = 0; word != bytes / 4; ++word) {
        uint const first_word = first_words[word], second_word = second_words[word];
        for (uint byte = 0; byte != 4; ++byte) {
            int const difference = int(char(first_word >> (8 * byte))) - int(char(second_word >> (8 * byte)));
            sum += uint(difference * difference);
        }
    }
    return float(sum);
}

/** The slots differing between the @p bytes -byte hash sketches at @p first and @p second, both
 *  four-byte aligned: the calling lane alone reads each a word at a time. */
inline float metal_hash_distance(device uchar const* first, device uchar const* second, uint bytes) {
    device uint const* const first_words = (device uint const*)first;
    device uint const* const second_words = (device uint const*)second;
    uint unequal = 0;
    for (uint word = 0; word != bytes / 4; ++word)
        unequal += sz_ngrams_sketch_word_distance_metal(first_words[word], second_words[word]);
    return float(unequal);
}

/** Writes at @p image the complement image of the @p bytes at @p data, stored a class byte a
 *  symbol, under @p costs, its symbols reversed and each complemented, with the simdgroup,
 *  and synchronizes it. */
inline void metal_complement_image(device metal_alignment_costs_t const& costs, device uchar const* data, uint bytes,
                                   device uchar* image, uint lane) {
    // The span may still serve the previous query's readers until every lane gets here.
    simdgroup_barrier(mem_flags::mem_device);
    for (uint symbol = lane; symbol < bytes; symbol += metal_simdgroup_width_k)
        image[symbol] = costs.complemented[data[bytes - 1 - symbol]];
    simdgroup_barrier(mem_flags::mem_device);
}

#pragma endregion Sketches and Complements

#pragma region Windows

/** Low bits of a share distance, a containment's or a banded overlap's, and the top level of its
 *  primary share, as @c share_secondary_bits_k and @c share_primary_levels_k. */
constant constexpr ulong metal_share_secondary_bits_k = 8, metal_share_primary_levels_k = (1ul << 15) - 1;

/** The containment distance of a candidate holding @p held of the query's @p distinct windows over
 *  @p candidate_windows of its own, as @c containment_distance lays it out. */
inline ulong metal_containment_distance(ulong distinct, ulong held, ulong candidate_windows) {
    ulong const secondary_top = (1ul << metal_share_secondary_bits_k) - 1;
    ulong const primary = distinct ? (distinct - held) * metal_share_primary_levels_k / distinct
                                   : metal_share_primary_levels_k;
    ulong const uncovered = candidate_windows - min(held, candidate_windows);
    // The prelude's `divide_round_up` takes 32-bit operands, so the 64-bit share rounds up here.
    ulong const scaled = uncovered * secondary_top;
    ulong const secondary = candidate_windows ? scaled / candidate_windows + (scaled % candidate_windows != 0)
                                              : secondary_top;
    return (primary << metal_share_secondary_bits_k) | secondary;
}

/** A prepared containment query: the window table of the sequence and the hit words its lane
 *  groups count in, @c (windows+31)/32 words to a group. */
struct metal_containment_query_t {
    sz_ngrams_query_metal_t table;
    device atomic_uint* hit_words;
};

/** Bytes the window table of a sequence takes once filed, a multiple of 16: the slots, a filter
 *  byte per slot, then the powers, as the host sizes the span. */
inline ulong metal_window_table_bytes(thread sz_ngrams_query_metal_t const& table) {
    return (table.slots_count * 9 + table.widths_count * 4 + 15) / 16 * 16;
}

/**
 *  @brief Window containment over folded UTF-8: a simdgroup files the query's distinct windows at
 *      every width into a table in its span, and a group of a rung's lanes counts the query
 *      windows each candidate holds, through rolling hashes whose residues match the CPU kits'.
 *
 *  The parameters are the window widths, which the hash sketch samples too, so builders link
 *  texts sharing the windows the score counts.
 */
struct metal_containment_kit_t {
    using canonical_query_t = metal_containment_query_t;

    /** Lanes rolling one sequence's windows, narrowest first. */
    static constexpr constant uint canonical_rungs_k[metal_containment_canonical_rungs_count_k] = {
        metal_containment_lanes(0), metal_containment_lanes(1), metal_containment_lanes(2), metal_containment_lanes(3),
        metal_containment_lanes(4)};

    /** The lane groups of the narrowest rung, each with its own hit words. */
    static constexpr constant uint canonical_groups_k = metal_simdgroup_width_k / metal_containment_lanes(0);

    /** A rung is its lanes per sequence. */
    static constexpr uint canonical_lanes_per_entry(uint lanes) { return lanes; }

    /** Prepares the query with the simdgroup and zeroes every group's hit words, which each walk
     *  leaves zero. */
    static canonical_query_t canonical_prepare(device uchar const* parameters, device uchar const* data, uint bytes,
                                               device uchar* span, uint lane) {
        device metal_window_parameters_t const& windows = *(device metal_window_parameters_t const*)parameters;
        canonical_query_t query;
        query.table.distinct = simd_sum(
            sz_ngrams_query_prepare_metal(data, bytes, windows.widths, windows.widths_count, span, query.table, lane));
        query.hit_words = (device atomic_uint*)(span + metal_window_table_bytes(query.table));
        ulong const words_count = canonical_groups_k * ((query.table.ngrams + 31) / 32);
        for (ulong word = lane; word < words_count; word += metal_simdgroup_width_k)
            atomic_store_explicit(query.hit_words + word, 0u, memory_order_relaxed);
        simdgroup_barrier(mem_flags::mem_device);
        return query;
    }

    /** No candidate holds more of the query's distinct windows than it has windows. */
    static float canonical_lower_bound(uint, thread canonical_query_t const& query, device uchar const*, uint bytes) {
        ulong const windows = sz_ngrams_count_metal(bytes, query.table.widths, query.table.widths_count);
        return float(metal_containment_distance(query.table.distinct, min(query.table.distinct, windows), windows));
    }

    /** Exact at every radius. A threadgroup shares simdgroup zero's span, so groups count across
     *  its simdgroups. */
    template <uint lanes_>
    static float canonical_distance(uint, thread canonical_query_t const& query, device uchar const* data, uint bytes,
                                    float, uint lane, uint simdgroup) {
        ulong const windows = sz_ngrams_count_metal(bytes, query.table.widths, query.table.widths_count);
        uint const group = (simdgroup * (metal_simdgroup_width_k / lanes_) + lane / lanes_) % metal_simdgroup_width_k;
        device atomic_uint* const hit_words = query.hit_words + group * ((query.table.ngrams + 31) / 32);
        uint held = sz_ngrams_candidate_matches_metal(query.table, data, bytes, hit_words, lane % lanes_, lanes_);
        if (lanes_ == metal_simdgroup_width_k) held = simd_sum(held);
        else
            for (uint offset = lanes_ / 2; offset != 0; offset >>= 1) held += simd_shuffle_xor(held, ushort(offset));
        sz_ngrams_candidate_clear_metal(query.table, hit_words, lane % lanes_, lanes_);
        // No group starts its next candidate before every lane's words are zero.
        simdgroup_barrier(mem_flags::mem_device);
        return float(metal_containment_distance(query.table.distinct, held, windows));
    }

    /** A hash sketch reads no tables, and takes a byte per slot in front of every stored
     *  sequence. */
    using sketch_tables_t = metal_no_tables_t;

    /** Each simdgroup's slot ranks, four bytes a slot, in threadgroup memory while it sketches. */
    static constexpr uint sketch_scratch_bytes(uint sketch_bytes) { return sketch_bytes * 4; }

    /** Writes the @p sketch_bytes -slot hash sketch of the stored sequence, folded UTF-8, at
     *  @p sketch, four-byte aligned, over the windows the score counts, ranking slots in the
     *  simdgroup's @p scratch ; every lane of one simdgroup calls it for one value. */
    static void sketch_write(device uchar const*, device uchar const* sequence, uint bytes, uint lane,
                             device uchar* sketch, uint sketch_bytes, threadgroup uchar* scratch) {
        ulong const widths[] = {metal_containment_narrow_window_k, metal_containment_wide_window_k};
        sz_ngrams_sketch_metal(sequence, bytes, widths, sizeof(widths) / sizeof(widths[0]),
                               (threadgroup atomic_uint*)scratch, sketch_bytes, sketch, lane);
    }

    /** Slots differing between the @p bytes -byte sketches at @p first and @p second, both
     *  four-byte aligned, counted by the calling lane alone; symmetric. */
    static float sketch_distance(device uchar const* first, device uchar const* second, uint bytes, uint) {
        return metal_hash_distance(first, second, bytes);
    }
};

/** The furthest banded overlap distance, the windows that must agree on a shift, and the widest
 *  band, as @c banded_overlap_ceiling_k, @c banded_overlap_votes_k and @c banded_overlap_band. */
constant constexpr ulong metal_banded_overlap_ceiling_k = (metal_share_primary_levels_k << 8) | 255ul;
constant constexpr uint metal_banded_overlap_votes_k = 2;

/** Cells either side of the voted diagonal the band of a @p span symbols long overlap keeps, as
 *  @c banded_overlap_band: a sixteenth of the span, at least four and at most 64. */
inline uint metal_banded_overlap_band(uint span) { return min(64u, max(4u, span / 16)); }

/** Cost of one symbol facing a gap under @p costs, as @c banded_overlap_gap. */
inline uint metal_banded_overlap_gap(device metal_alignment_costs_t const& costs) { return costs.gap; }

/** The banded overlap distance of two sequences, @p shorter and @p longer symbols long, whose
 *  voted shift lays @p span symbols over each other at the band's @p cost, as
 *  @c banded_overlap_distance lays it out. */
inline ulong metal_banded_overlap_distance(ulong cost, ulong gap, ulong span, ulong shorter, ulong longer) {
    ulong const secondary_top = (1ul << metal_share_secondary_bits_k) - 1;
    ulong const bound = 2 * gap * shorter;
    ulong const primary = bound  ? min(metal_share_primary_levels_k, cost * metal_share_primary_levels_k / bound)
                          : cost ? metal_share_primary_levels_k
                                 : 0ul;
    ulong const scaled = (longer - span) * secondary_top;
    ulong const secondary = longer ? scaled / longer + (scaled % longer != 0) : secondary_top;
    return (primary << metal_share_secondary_bits_k) | secondary;
}

/** The least banded overlap distance two sequences of @p shorter and @p longer symbols reach, as
 *  @c banded_overlap_lower_bound: no cost, and the whole shorter side laid over the longer. */
inline ulong metal_banded_overlap_lower_bound(ulong shorter, ulong longer) {
    return longer ? metal_banded_overlap_distance(0, 0, shorter, shorter, longer) : 0ul;
}

/**
 *  @brief The cost of aligning the @p span symbols at @p first to the @p span at @p second end to
 *      end under @p costs, inside a band of @p band cells either side of the diagonal, as
 *      @c alignment_band steps it.
 *
 *  Symbols are stored bytes, class codes scored through the folded table. The band's cells are
 *  kept once and stepped in place, a cell of them the column that many places off the diagonal;
 *  @p cells holds <tt>2 * band + 1</tt>.
 */
inline uint metal_alignment_band(device metal_alignment_costs_t const& costs, device uchar const* first,
                                 device uchar const* second, uint span, uint band, device uint* cells) {
    // Far past any cost, and still far from wrapping once a gap or a substitution is added.
    constexpr uint outside_k = 0xFFFFFFFFu / 2;
    uint const gap = metal_banded_overlap_gap(costs), count = 2 * band + 1;
    for (uint cell = 0; cell != count; ++cell)
        cells[cell] = cell >= band && cell - band <= span ? (cell - band) * gap : outside_k;
    for (uint index = 1; index <= span; ++index) {
        uint const symbol = first[index - 1];
        for (uint cell = 0; cell != count; ++cell) {
            // The column this cell holds at the symbol of `index`, one past it kept from wrapping.
            uint const column = index + cell;
            if (column < band || column - band > span) {
                cells[cell] = outside_k;
                continue;
            }
            if (column == band) {
                cells[cell] = index * gap;
                continue;
            }
            uint const other = second[column - band - 1];
            uint const substitution = costs.substitutions[symbol][other];
            uint const above = cell + 1 != count ? cells[cell + 1] : outside_k;
            uint const left = cell ? cells[cell - 1] : outside_k;
            cells[cell] = min(cells[cell] + substitution, min(above, left) + gap);
        }
    }
    return cells[band];
}

/** A prepared banded overlap query: the window table and the symbols of the sequence, its
 *  complement image's beside them, the costs a band aligns under, and every lane's own
 *  vote counters and band cells. */
struct metal_banded_overlap_query_t {
    sz_ngrams_query_metal_t table;
    sz_ngrams_query_metal_t complement;
    device uchar const* symbols;
    device uchar const* complement_symbols;
    device metal_alignment_costs_t const* costs;

    /** A tile of counters per lane, as many as the query has symbols. */
    device uint* votes;

    /** One band's cells per lane, two of the widest band and one more. */
    device uint* cells;

    /** One, or two where the costs complement. */
    uint images;
};

/**
 *  @brief Counts into @p votes the shifts of one tile, @p tile of them from @p tile_first, that
 *      the windows of width @p index of the @p bytes at @p data vote for against the image filed
 *      in @p table.
 *
 *  Every window of the sequence the image holds once votes for the shift between the two places,
 *  counted from the furthest the sequence starts before the image. A place votes a shift no
 *  further below it than the image's last symbol, so a tile scans the places from that far before
 *  its first shift up to its last, the width rolling its own hash over them, as StringZilla's
 *  candidate walk rolls it.
 */
inline void metal_banded_overlap_tile(thread sz_ngrams_query_metal_t const& table, device uchar const* data, uint bytes,
                                      uint index, uint tile_first, uint tile, device uint* votes) {
    uint const query_bytes = uint(table.length), width = uint(table.widths[index]);
    if (!width || width > query_bytes || width > bytes) return;
    uint const first_place = tile_first >= query_bytes ? tile_first - (query_bytes - 1) : 0;
    uint const last_ending = min(bytes, min(bytes, tile_first + tile) + width - 1);
    uint const power = table.powers[index];
    uint hash = 0;
    for (uint ending = first_place + 1; ending <= last_ending; ++ending) {
        uint const rolled = ending - first_place;
        uint const outgoing = rolled > width ? uint(data[ending - width - 1]) : 0u;
        hash = sz_ngrams_roll_metal(hash, data[ending - 1], outgoing, power);
        if (rolled < width) continue;
        ulong const slot = sz_ngrams_query_probe_metal(table, hash, index);
        // A window the image repeats names no single shift.
        if (!slot || sz_ngrams_slot_repeated_metal(slot)) continue;
        uint const first = sz_ngrams_slot_first_metal(slot);
        uint const shift = ending - width + (query_bytes - 1) - first;
        if (shift >= tile_first && shift - tile_first < tile) ++votes[shift - tile_first];
    }
}

/**
 *  @brief The banded overlap distance from one image of a query, its @p symbols filed in @p table,
 *      to the @p bytes at @p data under @p costs, by one lane over its own @p votes and @p cells.
 *
 *  A span is sized from its query alone, so the shifts are counted a tile at a time, a tile as
 *  wide as the image: tiles ascend, and a later shift wins only with strictly more votes, so the
 *  least of the most voted shifts wins, as one pass over every shift would pick it. The band is
 *  @ref metal_alignment_band 's.
 */
inline ulong metal_banded_overlap_image(device metal_alignment_costs_t const& costs,
                                        thread sz_ngrams_query_metal_t const& table, device uchar const* symbols,
                                        device uchar const* data, uint bytes, device uint* votes, device uint* cells) {
    uint const query_bytes = uint(table.length);
    if (!query_bytes || !bytes) return metal_banded_overlap_ceiling_k;
    uint const tile = query_bytes, shifts = query_bytes + bytes - 1;
    uint most_votes = 0, voted_shift = 0;
    for (uint tile_first = 0; tile_first < shifts; tile_first += tile) {
        for (uint counter = 0; counter != tile; ++counter) votes[counter] = 0;
        for (uint index = 0; index != uint(table.widths_count); ++index)
            metal_banded_overlap_tile(table, data, bytes, index, tile_first, tile, votes);
        for (uint counter = 0; counter != tile && tile_first + counter < shifts; ++counter)
            if (votes[counter] > most_votes) most_votes = votes[counter], voted_shift = tile_first + counter;
    }
    if (most_votes < metal_banded_overlap_votes_k) return metal_banded_overlap_ceiling_k;
    // The voted shift lays the image's symbols from `low` over the sequence's from `start`.
    uint const low = voted_shift < query_bytes - 1 ? query_bytes - 1 - voted_shift : 0;
    uint const start = voted_shift > query_bytes - 1 ? voted_shift - (query_bytes - 1) : 0;
    uint const span = min(query_bytes - low, bytes - start);
    uint const shorter = min(query_bytes, bytes), longer = max(query_bytes, bytes);
    uint const gap = metal_banded_overlap_gap(costs);
    uint const cost = metal_alignment_band(costs, symbols + low, data + start, span, metal_banded_overlap_band(span),
                                           cells) +
                      gap * (shorter - span);
    return metal_banded_overlap_distance(cost, gap, span, shorter, longer);
}

/**
 *  @brief Banded overlap: a simdgroup files the query's distinct windows into a table in its span,
 *      and each lane scores a candidate of its own, voting the shift most of the candidate's
 *      windows agree on and aligning the band around it under the costs.
 *
 *  A lane to a candidate, because a band's steps follow one another and take most of a score, so
 *  a simdgroup aligns as many candidates at once as it has lanes. Windows hash through
 *  StringZilla, whose residues are the CPU kits'. The parameters are the window widths and the
 *  folded costs with the complement.
 */
struct metal_banded_overlap_kit_t {
    using canonical_query_t = metal_banded_overlap_query_t;

    /** Lanes scoring one sequence: one, on the kit's one rung. */
    static constexpr constant uint canonical_rungs_k[metal_banded_overlap_canonical_rungs_count_k] = {1};

    /** A rung is its lanes per sequence. */
    static constexpr uint canonical_lanes_per_entry(uint lanes) { return lanes; }

    /** Prepares the query with the simdgroup, and where the costs complement its complement image
     *  behind the table; the counters and cells past them are each lane's to overwrite. */
    static canonical_query_t canonical_prepare(device uchar const* parameters, device uchar const* data, uint bytes,
                                               device uchar* span, uint lane) {
        device metal_window_parameters_t const& windows = *(device metal_window_parameters_t const*)parameters;
        canonical_query_t query;
        query.costs = &windows.costs, query.images = windows.costs.complements ? 2u : 1u;
        query.table.distinct = simd_sum(
            sz_ngrams_query_prepare_metal(data, bytes, windows.widths, windows.widths_count, span, query.table, lane));
        query.complement = query.table;
        query.symbols = query.complement_symbols = data;
        ulong const table_bytes = metal_window_table_bytes(query.table);
        device uchar* counters = span + table_bytes;
        if (windows.costs.complements) {
            device uchar* const image = counters;
            metal_complement_image(windows.costs, data, bytes, image, lane);
            device uchar* const complement_span = image + (bytes + 15) / 16 * 16;
            query.complement.distinct = simd_sum(sz_ngrams_query_prepare_metal(
                image, bytes, windows.widths, windows.widths_count, complement_span, query.complement, lane));
            query.complement_symbols = image, counters = complement_span + table_bytes;
        }
        query.votes = (device uint*)counters;
        query.cells = (device uint*)(counters + (ulong(metal_simdgroup_width_k) * bytes * 4 + 15) / 16 * 16);
        return query;
    }

    /** No overlap costs less than nothing, nor spans more than the shorter side. */
    static float canonical_lower_bound(uint, thread canonical_query_t const& query, device uchar const*, uint bytes) {
        ulong const query_bytes = query.table.length;
        return float(metal_banded_overlap_lower_bound(min(query_bytes, ulong(bytes)), max(query_bytes, ulong(bytes))));
    }

    /** The lower of the images' distances, scored by the calling lane alone. */
    template <uint lanes_>
    static float canonical_distance(uint, thread canonical_query_t const& query, device uchar const* data, uint bytes,
                                    float, uint lane, uint) {
        uint const query_bytes = uint(query.table.length);
        device uint* const votes = query.votes + ulong(lane) * query_bytes;
        device uint* const cells = query.cells + lane * (2 * metal_banded_overlap_band(query_bytes) + 1);
        ulong distance = metal_banded_overlap_image(*query.costs, query.table, query.symbols, data, bytes, votes,
                                                    cells);
        if (query.images != 1)
            distance = min(distance, metal_banded_overlap_image(*query.costs, query.complement,
                                                                query.complement_symbols, data, bytes, votes, cells));
        return float(distance);
    }

    /** No sketch: builders score canonically, and a stored value is its class bytes closed to a
     *  word. */
    using sketch_tables_t = metal_no_tables_t;
};

#pragma endregion Windows

#pragma region Alignment

/** A query an alignment walk scores rows against, laid out in its span. */
struct metal_alignment_query_t {

    /** The profile: a stripe per stripe of the query, the cost of every class against each lane's
     *  codes. */
    device uchar* span;

    /** Past the profile, for a query of more than one stripe, a boundary row per simdgroup of a
     *  threadgroup. */
    device uchar* boundaries;

    /** Codes of the query. */
    uint codes;

    /** Query codes each lane holds, at most its rung's. */
    uint rows_per_lane;

    /** Cost of one code facing a gap. */
    uint gap;
};

/** Writes lane @p lane 's profile words of @p query, the packed @p data under @p costs: the cost
 *  of every class against each of the lane's codes, a stripe at a time. */
inline void metal_alignment_profile(device metal_alignment_costs_t const& costs, device uchar const* data,
                                    thread metal_alignment_query_t const& query, uint lane) {
    constexpr uint lanes_k = metal_simdgroup_width_k, classes_k = metal_alignment_classes_k;
    uint const stripes = metal_alignment_stripes(query.codes);
    device ulong* const profile = (device ulong*)query.span;
    // A word per stripe and class.
    for (uint word = 0; word != stripes * classes_k; ++word) {
        uint const stripe = word / classes_k, code = word % classes_k;
        uint const lane_first = (stripe * lanes_k + lane) * query.rows_per_lane;
        ulong costs_word = 0;
        for (uint row_in_lane = 0; row_in_lane != query.rows_per_lane; ++row_in_lane) {
            uint const index = lane_first + row_in_lane;
            if (index >= query.codes) continue;
            uint const query_code = metal_packed_code_at(data, index);
            costs_word |= ulong(costs.substitutions[query_code][code]) << (8 * row_in_lane);
        }
        profile[word * lanes_k + lane] = costs_word;
    }
}

/** Lays the packed @p data out as a query in @p span under the costs at @p parameters, every lane
 *  of the simdgroup writing its own profile words. */
inline metal_alignment_query_t metal_alignment_prepare(device uchar const* parameters, device uchar const* data,
                                                       uint bytes, device uchar* span, uint lane) {
    device metal_alignment_costs_t const& costs = *(device metal_alignment_costs_t const*)parameters;
    metal_alignment_query_t query;
    query.span = span, query.gap = costs.gap;
    query.codes = metal_packed_codes_count(data, bytes);
    query.rows_per_lane = metal_alignment_rows_per_lane(query.codes);
    query.boundaries = span + ulong(metal_alignment_stripes(query.codes)) * metal_alignment_stripe_bytes_k;
    // The span may still serve the previous query's readers until every lane gets here.
    simdgroup_barrier(mem_flags::mem_device);
    metal_alignment_profile(costs, data, query, lane);
    simdgroup_barrier(mem_flags::mem_device);
    return query;
}

/** The word lane @p lane loads for the first lane per 32 columns from @p first_column : the top
 *  cell of its column, the boundary row's past the first stripe, beside the next column's code. */
inline uint metal_alignment_loaded(device uchar const* data, device uint const* boundary, uint first_column, uint lane,
                                   uint columns, uint stripe, uint gap) {
    uint const column = first_column + lane;
    uint const top = column > columns ? 0u : stripe ? boundary[column] : column * gap;
    uint const next_code = column < columns ? metal_packed_code_at(data, column) : 0u;
    return top << 5 | next_code;
}

/**
 *  @brief The alignment cost from @p query to the packed @p data, a simdgroup's anti-diagonal
 *      wavefront over stripes of query codes, @p rows_ codes at most to a lane: exact within
 *      @p radius, and past it a stripe's bottom row minimum above it.
 *
 *  Lane @c l holds a stripe's rows from its @c l -th share down and enters column @c j at step
 *  @c j+l, taking the cell above and the next column's code from lane @c l-1 in one shuffled
 *  word, the first lane from a word the simdgroup loads per 32 columns. Each stripe hands its
 *  bottom row to the next through the simdgroup's boundary row, which the last lane writes 31
 *  columns behind the simdgroup's reads, as the CUDA kit does; the simdgroup's lanes step
 *  together, so no column is written before it is read.
 */
template <uint rows_>
inline float metal_alignment_score(thread metal_alignment_query_t const& query, device uchar const* data, uint bytes,
                                   float radius, uint lane, uint simdgroup) {
    constexpr uint lanes_k = metal_simdgroup_width_k, classes_k = metal_alignment_classes_k;
    uint const gap = query.gap, rows = query.codes, rows_per_lane = query.rows_per_lane;
    uint const columns = metal_packed_codes_count(data, bytes);
    if (!rows || !columns) return float((rows + columns) * gap);
    uint const stripe_rows = lanes_k * rows_per_lane, stripes = metal_alignment_stripes(rows);
    device ulong const* const profile = (device ulong const*)query.span;
    device uint* const boundary = (device uint*)query.boundaries + simdgroup * (metal_alignment_widest_codes_k + 1);
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
        uint const first_code = metal_packed_code_at(data, 0);
        uint handed = lane == 0 ? first_code : 0u;
        uint loaded = metal_alignment_loaded(data, boundary, 1, lane, columns, stripe, gap);
        ulong costs_word = stripe_profile[first_code * lanes_k];
        uint diagonal_above = lane_first * gap, bottom_minimum = (stripe + 1) * stripe_rows * gap;
        uint const steps = columns + active - 1;
        for (uint step = 0; step != steps; ++step) {
            if (step % lanes_k == 0 && step)
                loaded = metal_alignment_loaded(data, boundary, step + 1, lane, columns, stripe, gap);
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

/**
 *  @brief The alignment costs from @p query to the packed @p first and @p second at once, each
 *      the value @ref metal_alignment_score returns for it under its own radius.
 *
 *  The wavefront of the single scorer with two cells in every @c ushort2 register, @p first in
 *  @c x and @p second in @c y, which 16-bit adds and minimums step together. A lane takes both
 *  tops from lane @c l-1 in one shuffled pair, and both of the next column's codes in a word from
 *  the lane holding them in the simdgroup's last two loads of 32 columns, five bits each. Steps
 *  run to the longer candidate; a lane past one candidate's last column keeps that half's cells,
 *  as the single scorer stops, and a candidate a stripe's crossing settles, or the last stripe
 *  finishes, keeps its distance while the other runs on, its half frozen from then on.
 */
template <uint rows_>
inline float2 metal_alignment_score_pair(thread metal_alignment_query_t const& query, device uchar const* first,
                                         uint first_bytes, device uchar const* second, uint second_bytes,
                                         float first_radius, float second_radius, uint lane, uint simdgroup) {
    constexpr uint lanes_k = metal_simdgroup_width_k, classes_k = metal_alignment_classes_k;
    constexpr uint code_bits_k = metal_packed_code_bits_k, codes_bits_k = 2 * code_bits_k;
    constexpr uint code_k = (1u << code_bits_k) - 1;
    uint const gap = query.gap, rows = query.codes, rows_per_lane = query.rows_per_lane;
    ushort2 const gaps = ushort2(ushort(gap));
    uint const first_columns = metal_packed_codes_count(first, first_bytes);
    uint const second_columns = metal_packed_codes_count(second, second_bytes);
    float2 distances = float2(float((rows + first_columns) * gap), float((rows + second_columns) * gap));
    // A candidate's last column while it scores, zero once it holds its distance.
    uint first_limit = rows ? first_columns : 0u, second_limit = rows ? second_columns : 0u;
    if (!first_limit && !second_limit) return distances;
    uint const stripe_rows = lanes_k * rows_per_lane, stripes = metal_alignment_stripes(rows);
    device ulong const* const profile = (device ulong const*)query.span;
    device ushort2* const boundary = (device ushort2*)query.boundaries +
                                     simdgroup * (metal_alignment_widest_codes_k + 1);
    for (uint stripe = 0; stripe != stripes; ++stripe) {
        uint const first_row = stripe * stripe_rows, lane_first = first_row + lane * rows_per_lane;
        uint const active = min(lanes_k, divide_round_up(rows - first_row, rows_per_lane));
        uint const last_lane = (rows - 1 - first_row) / rows_per_lane,
                   last_cell = (rows - 1 - first_row) % rows_per_lane;
        bool const closing = stripe + 1 == stripes;
        ushort2 cells[rows_];
        for (uint cell = 0; cell != rows_; ++cell) cells[cell] = ushort2(ushort((lane_first + cell + 1) * gap));
        device ulong const* const stripe_profile = profile + stripe * classes_k * lanes_k + lane;
        // The halves a lane scores at a column: the candidates still running that reach it.
        auto const halves_at = [&](uint column) { return bool2(column <= first_limit, column <= second_limit); };
        // The first lane's top cells of a column, a column to a lane.
        auto const load_tops = [&](uint first_column) {
            uint const column = first_column + lane;
            bool2 const halves = halves_at(column);
            if (!any(halves)) return ushort2(0);
            return select(ushort2(0), stripe ? boundary[column] : ushort2(ushort(column * gap)), halves);
        };
        // Both candidates' codes of a column, a column to a lane.
        auto const load_codes = [&](uint first_column) {
            uint const column = first_column + lane;
            uint const first_code = column < first_limit ? metal_packed_code_at(first, column) : 0u;
            uint const second_code = column < second_limit ? metal_packed_code_at(second, column) : 0u;
            return first_code | second_code << code_bits_k;
        };
        uint const opening_codes = (first_limit ? metal_packed_code_at(first, 0) : 0u) |
                                   (second_limit ? metal_packed_code_at(second, 0) : 0u) << code_bits_k;
        // The previous load's codes in the low ten bits and the current one's above them; the first
        // column's codes stand as the last lane's of a load before the first.
        uint codes = (lane == lanes_k - 1 ? opening_codes : 0u) | load_codes(1) << codes_bits_k;
        ushort2 tops = load_tops(1), handed = ushort2(0);
        ulong first_costs = stripe_profile[(opening_codes & code_k) * lanes_k];
        ulong second_costs = stripe_profile[(opening_codes >> code_bits_k) * lanes_k];
        ushort2 diagonal_above = ushort2(ushort(lane_first * gap));
        ushort2 bottom_minimum = ushort2(ushort((stripe + 1) * stripe_rows * gap));
        uint const steps = max(first_limit, second_limit) + active - 1;
        for (uint step = 0; step != steps; ++step) {
            if (step % lanes_k == 0 && step)
                tops = load_tops(step + 1), codes = codes >> codes_bits_k | load_codes(step + 1) << codes_bits_k;
            ushort2 const from_loaded = simd_shuffle(tops, ushort(step % lanes_k));
            ushort2 const from_above = simd_shuffle_up(handed, ushort(1));
            ushort2 const top = lane == 0 ? from_loaded : from_above;
            // Lane `l` scores code `step - l + 1` next, in the current load once `step % 32 >= l`.
            uint const next_codes = simd_shuffle(codes, ushort((step - lane) % lanes_k)) >>
                                    (step % lanes_k >= lane ? codes_bits_k : 0u);
            ulong const next_first_costs = stripe_profile[(next_codes & code_k) * lanes_k];
            ulong const next_second_costs = stripe_profile[(next_codes >> code_bits_k & code_k) * lanes_k];
            uint const column = step - lane + 1;
            bool2 const halves = lane < active && step >= lane ? halves_at(column) : bool2(false);
            // A lane scoring no half hands the cells above on, which reach no lane that scores.
            if (!any(halves)) {
                handed = top, first_costs = next_first_costs, second_costs = next_second_costs;
                continue;
            }
            ushort2 above = top, diagonal = diagonal_above;
            // Bounded by the rung too, so the cells stay in registers.
            for (uint cell = 0; cell != rows_ && cell != rows_per_lane; ++cell) {
                ushort2 const substitutions = ushort2(ushort((first_costs >> (8 * cell)) & 0xFFu),
                                                      ushort((second_costs >> (8 * cell)) & 0xFFu));
                ushort2 const scored = min(diagonal + substitutions, min(cells[cell], above) + gaps);
                diagonal = cells[cell], cells[cell] = select(cells[cell], scored, halves), above = cells[cell];
            }
            diagonal_above = select(diagonal_above, top, halves);
            // A kept half re-offers the bottom cell its last column already offered.
            if (!closing && lane == lanes_k - 1) boundary[column] = above, bottom_minimum = min(bottom_minimum, above);
            handed = above, first_costs = next_first_costs, second_costs = next_second_costs;
        }
        simdgroup_barrier(mem_flags::mem_device);
        // A running candidate's half froze at its last column, so its last row's cell is its cost.
        ushort2 last_row = ushort2(0);
        for (uint cell = 0; cell != rows_; ++cell) last_row = cell == last_cell ? cells[cell] : last_row;
        if (closing) {
            ushort2 const results = simd_shuffle(last_row, ushort(last_lane));
            if (first_limit) distances.x = float(results.x);
            if (second_limit) distances.y = float(results.y);
            return distances;
        }
        // Every alignment crosses a stripe's bottom row, so none ends below its minimum.
        ushort2 const crossing = simd_shuffle(bottom_minimum, ushort(lanes_k - 1));
        if (first_limit && float(crossing.x) > first_radius) distances.x = float(crossing.x), first_limit = 0;
        if (second_limit && float(crossing.y) > second_radius) distances.y = float(crossing.y), second_limit = 0;
        if (!first_limit && !second_limit) return distances;
    }
    return distances;
}

/** Alignment over packed codes: a simdgroup to a sequence, an anti-diagonal wavefront down
 *  stripes of the query, its costs a profile the walk lays out in its span from the folded table
 *  the face uploads as the parameters. */
struct metal_alignment_kit_t {
    using canonical_query_t = metal_alignment_query_t;

    /** Query codes each lane holds, narrowest first. */
    static constexpr constant uint canonical_rungs_k[metal_alignment_canonical_rungs_count_k] = {
        metal_alignment_rows(0), metal_alignment_rows(1), metal_alignment_rows(2), metal_alignment_rows(3),
        metal_alignment_rows(4), metal_alignment_rows(5), metal_alignment_rows(6), metal_alignment_rows(7)};

    /** Every rung scores a sequence with the whole simdgroup. */
    static constexpr uint canonical_lanes_per_entry(uint) { return metal_simdgroup_width_k; }

    static canonical_query_t canonical_prepare(device uchar const* parameters, device uchar const* data, uint bytes,
                                               device uchar* span, uint lane) {
        return metal_alignment_prepare(parameters, data, bytes, span, lane);
    }

    /** Every code one side holds past the other's length faces a gap. */
    static float canonical_lower_bound(uint, thread canonical_query_t const& query, device uchar const* data,
                                       uint bytes) {
        uint const columns = metal_packed_codes_count(data, bytes);
        uint const difference = columns > query.codes ? columns - query.codes : query.codes - columns;
        return float(difference * query.gap);
    }
    template <uint rows_>
    static float canonical_distance(uint, thread canonical_query_t const& query, device uchar const* data, uint bytes,
                                    float radius, uint lane, uint simdgroup) {
        return metal_alignment_score<rows_>(query, data, bytes, radius, lane, simdgroup);
    }
    template <uint rows_>
    static float2 canonical_distance_pair(uint, thread canonical_query_t const& query, device uchar const* first,
                                          uint first_bytes, device uchar const* second, uint second_bytes,
                                          float first_radius, float second_radius, uint lane, uint simdgroup) {
        return metal_alignment_score_pair<rows_>(query, first, first_bytes, second, second_bytes, first_radius,
                                                 second_radius, lane, simdgroup);
    }

    /** What the producer reads, uploaded once beside the kit's parameters as the buffer the view
     *  borrows. */
    using sketch_tables_t = metal_spectrum_view_t;

    /** A smooth sketch sums in registers, so it carves no scratch. */
    static constexpr uint sketch_scratch_bytes(uint) { return 0; }

    /** Writes the @p sketch_bytes -column smooth sketch of the stored sequence, packed class codes,
     *  at @p sketch, four-byte aligned, from the tables laid out at @p tables; every lane of one
     *  simdgroup calls it for one value. */
    static void sketch_write(device uchar const* tables, device uchar const* sequence, uint bytes, uint lane,
                             device uchar* sketch, uint sketch_bytes, threadgroup uchar*) {
        metal_sketch_packed(metal_spectrum_view_of(tables), sequence, bytes, lane, sketch, sketch_bytes);
    }

    /** Squared Euclidean distance between the @p bytes -byte sketches at @p first and @p second,
     *  both four-byte aligned, accumulated in 32 bits by the calling lane alone; symmetric. */
    static float sketch_distance(device uchar const* first, device uchar const* second, uint bytes, uint) {
        return metal_smooth_distance(first, second, bytes);
    }
};

#pragma endregion Alignment

} // namespace unum::usearch::sequence

#endif // defined(__METAL_VERSION__)
