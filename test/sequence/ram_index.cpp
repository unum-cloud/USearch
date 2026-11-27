/**
 *  @file test/sequence/ram_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The sequence suite: the basic scenarios over owned strings under containment and under
 *      alignment over packed classes, then every compiled kit against exact oracles.
 *
 *  An index storing sketches is held to its layout, a sketch then the sequence in one value, to
 *  the two roles reading their own part of it, and to a round trip through a checkpoint that
 *  holds the sequences alone; a kit that sketches nothing stores the sequence alone.
 *
 *  Strings are eight planted templates with an eighth of their symbols edited, so distances
 *  cluster by template while the bytes differ. The oracles are the two-row alignment recurrence
 *  over packed codes, a brute-force count of the windows two strings share, and the header's
 *  statement of a banded overlap.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cmath>   // `std::sqrt`, `std::exp`, `std::fabs` - the agreement of two places of a sketch
#include <cstring> // `std::memcmp` - the folded nucleotide costs

#include <algorithm>   // `std::find` - a found key among the batch's
#include <array>       // `std::array` - the window widths, the handmade triples, a round's distances
#include <limits>      // `std::numeric_limits` - the oracle's open best
#include <map>         // `std::map` - the overlap oracle's votes per shift
#include <memory>      // `std::unique_ptr`
#include <new>         // `std::nothrow`
#include <span>        // `std::span` - a batch of queries
#include <string>      // `std::string` - a generated string before it is stored
#include <string_view> // `std::string_view` - the symbols and the oracles' operands
#include <thread>      // `std::this_thread::yield` - a clear retried past a racing call
#include <vector>      // `std::vector` - generated values, templates, the oracle's rows

#include "usearch/sequence/ram_index.hpp"
#include "usearch/basic/ram_indexes.hpp"

#include "basic.hpp"

namespace unum::usearch::test::sequence {
using namespace unum::usearch::sequence;

/** Candidates one round may stage: past every round the scenarios open, so no walk saturates. */
constexpr std::size_t sequence_staging_limit_k = 256;

/** Runes one to four bytes wide, @c é and @c è sharing a lead byte, escaped for editors. */
constexpr std::string_view text_alphabet_k = "abcdefg "                             //
                                             "\xC3\xA9\xC3\xA8\xD0\xB6"             //
                                             "\xE8\xAA\x9E\xE6\xA4\x9C\xE2\x82\xAC" //
                                             "\xF0\x9F\x98\x80\xF0\x9F\x9A\x80";

/** Symbols in a text template, which pass the inline capacity of a stored string. */
constexpr std::size_t text_symbols_k = 24;

/** The nucleotides, whose short windows recur across templates. */
constexpr std::string_view overlap_alphabet_k = "ACGT";

/** Symbols in an alignment's nucleotide template, and in a containment's, whose longer windows need
 *  more: few, so no window hash collision meets the exact oracle. */
constexpr std::size_t alignment_symbols_k = 12, overlap_symbols_k = 24;

using sequence_index_t =
    sequence::ram_index<util::uuid128_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref, counting_allocator_t>;
using sequence_value_t = sequence_index_t::value_t;

/** The runes of @p text as views of their bytes, measured by their lead byte alone. */
static std::vector<std::string_view> split_symbols(std::string_view text) {
    std::vector<std::string_view> symbols;
    for (std::size_t position = 0; position < text.size();) {
        u8_t const lead = static_cast<u8_t>(text[position]);
        std::size_t const length = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
        symbols.push_back(text.substr(position, length));
        position += length;
    }
    return symbols;
}

/** A stored string holding @p text in memory of @p allocator, as the codec's load builds one. */
static sequence_value_t make_value(std::string_view text, counting_allocator_t const& allocator) noexcept {
    sequence_value_t value(allocator);
    [[maybe_unused]] sz::status_t const assigned = value.assign(text);
    return value;
}

/** Draws @p count stored strings as the file's docblock says, from templates of @p symbols_count
 *  runes of @p alphabet_text, each built through @p allocator. */
static std::vector<sequence_value_t> generate_values(std::size_t count, u64_t seed, std::string_view alphabet_text,
                                                     std::size_t symbols_count, counting_allocator_t const& allocator) {
    constexpr std::size_t templates_k = 8;
    std::vector<std::string_view> const alphabet = split_symbols(alphabet_text);
    split_mix_generator_t random_generator {seed};
    uniform_below<u32_t> const symbol_distribution {static_cast<u32_t>(alphabet.size())};
    uniform_below<u32_t> const template_distribution {templates_k};
    uniform_below<u32_t> const edit_kind_distribution {3};
    std::array<std::vector<std::string_view>, templates_k> templates;
    for (std::vector<std::string_view>& symbols : templates)
        for (std::size_t symbol = 0; symbol != symbols_count; ++symbol)
            symbols.push_back(alphabet[symbol_distribution(random_generator)]);

    std::vector<sequence_value_t> values;
    values.reserve(count);
    std::string text;
    for (std::size_t ordinal = 0; ordinal != count; ++ordinal) {
        std::vector<std::string_view> symbols = templates[template_distribution(random_generator)];
        for (std::size_t edit = 0; edit != symbols_count / 8; ++edit) {
            std::size_t const position = uniform_below<u32_t> {static_cast<u32_t>(symbols.size())}(random_generator);
            std::string_view const symbol = alphabet[symbol_distribution(random_generator)];
            switch (edit_kind_distribution(random_generator)) {
            case 0: symbols[position] = symbol; break;
            case 1: symbols.insert(symbols.begin() + position, symbol); break;
            default: symbols.erase(symbols.begin() + position); break;
            }
        }
        text.clear();
        for (std::string_view const symbol : symbols) text += symbol;
        values.push_back(make_value(text, allocator));
    }
    return values;
}

/** The containment distance by brute force: each distinct window of @p query at one width is a hit
 *  when @p stored holds its bytes, and the hits and @p stored 's own windows lay out as
 *  @c containment_distance lays them. */
static sz_size_t containment_reference(std::string_view query, std::string_view stored) noexcept {
    u64_t distinct = 0, held = 0, windows = 0;
    for (std::size_t const width : containment_window_widths_k) {
        windows += stored.size() >= width ? stored.size() - width + 1 : 0;
        for (std::size_t start = 0; start + width <= query.size(); ++start) {
            std::string_view const window = query.substr(start, width);
            if (query.find(window) != start) continue;
            ++distinct;
            held += stored.find(window) != std::string_view::npos;
        }
    }
    return static_cast<sz_size_t>(containment_distance(distinct, held, windows));
}

/** The alignment fixture's costs over the nucleotides: transitions cost one, transversions and gaps
 *  two. */
constexpr alignment_costs_t nucleotide_costs() noexcept {
    alignment_costs_t costs {};
    costs.classes = 4, costs.gap = 2;
    for (std::size_t first = 0; first != 4; ++first)
        for (std::size_t second = 0; second != 4; ++second)
            costs.substitutions[first][second] = first == second ? 0 : (first & 1) == (second & 1) ? 1 : 2;
    return costs;
}

constexpr alignment_costs_t nucleotide_costs_k = nucleotide_costs();

/** The codes of @p packed: its bits read low first from the byte past the first, as many as
 *  fit less the unused slots that first byte counts. */
static std::vector<u32_t> unpacked_codes(std::string_view packed) {
    std::vector<u32_t> codes;
    if (packed.empty()) return codes;
    std::size_t const count = (packed.size() - 1) * 8 / packed_code_bits_k - static_cast<u8_t>(packed[0]);
    for (std::size_t index = 0; index != count; ++index) {
        u32_t code = 0;
        for (u32_t bit = 0; bit != packed_code_bits_k; ++bit) {
            std::size_t const position = index * packed_code_bits_k + bit;
            code |= (static_cast<u8_t>(packed[1 + position / 8]) >> (position % 8) & 1u) << bit;
        }
        codes.push_back(code);
    }
    return codes;
}

/** @p codes packed behind the count of unused slots, as the metric reads them. */
static std::string packed_sequence(std::span<u32_t const> codes) {
    std::size_t const data_bytes = (codes.size() * packed_code_bits_k + 7) / 8;
    std::string packed(1 + data_bytes, '\0');
    packed[0] = static_cast<char>(data_bytes * 8 / packed_code_bits_k - codes.size());
    for (std::size_t index = 0; index != codes.size(); ++index)
        for (u32_t bit = 0; bit != packed_code_bits_k; ++bit) {
            std::size_t const position = index * packed_code_bits_k + bit;
            packed[1 + position / 8] = static_cast<char>(static_cast<u8_t>(packed[1 + position / 8]) |
                                                         ((codes[index] >> bit & 1u) << (position % 8)));
        }
    return packed;
}

/** The alignment cost between two packed sequences under @p costs through the two-row
 *  recurrence: the oracle, sharing no code with the metric's kernels or unpacking. */
static sz_size_t alignment_reference(std::string_view first, std::string_view second, alignment_costs_t const& costs) {
    std::vector<u32_t> const rows = unpacked_codes(first), columns = unpacked_codes(second);
    std::vector<sz_size_t> previous_row(columns.size() + 1), current_row(columns.size() + 1);
    for (std::size_t column = 0; column <= columns.size(); ++column) previous_row[column] = column * costs.gap;
    for (std::size_t row = 1; row <= rows.size(); ++row) {
        current_row[0] = row * costs.gap;
        for (std::size_t column = 1; column <= columns.size(); ++column)
            current_row[column] = min_of(
                previous_row[column - 1] + costs.substitutions[rows[row - 1]][columns[column - 1]],
                min_of(previous_row[column] + costs.gap, current_row[column - 1] + costs.gap));
        std::swap(previous_row, current_row);
    }
    return previous_row[columns.size()];
}

/** The nucleotides of @p text packed, A, C, G and T in that order of their codes. */
static std::string packed_nucleotides(std::string_view text) {
    std::vector<u32_t> codes;
    for (char const nucleotide : text) codes.push_back(static_cast<u32_t>(overlap_alphabet_k.find(nucleotide)));
    return packed_sequence(codes);
}

/** Three stored strings and their pairwise distances, stated by hand. */
struct sequence_handmade_t {
    std::array<sequence_value_t, 3> values;
    std::array<std::array<sz_size_t, 3>, 3> distances;
};

struct containment_fixture_t {
    using index_t = sequence_index_t;
    using value_t = sequence_value_t;
    using allocator_t = index_t::allocator_t;
    using metric_t = containment_metric_t;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static metric_t make_metric() noexcept {
        metric_t metric;
        [[maybe_unused]] status_t const sized = metric.init(sequence_staging_limit_k);
        return metric;
    }
    static codec_t make_codec() noexcept { return {}; }
    static util::uuid128_t make_key(std::size_t ordinal) noexcept { return uuid_of(0x0E7E1A90ull + ordinal); }
    static sz_size_t oracle(std::string_view query, std::string_view stored) noexcept {
        return containment_reference(query, stored);
    }
    static bool value_equals(std::string_view first, std::string_view second) noexcept { return first == second; }

    /** "abcdefghi" holds all five windows of "abcdefgh", and two of its seven are not the query's,
     *  so it sits past "abcdefgh" itself only by the tie break; "abcdefgh" holds five of the seven
     *  of "abcdefghi", and "zyxwvuts" none of either's. */
    static sequence_handmade_t make_handmade(allocator_t const& allocator) {
        constexpr sz_size_t superset_k = containment_distance(5, 5, 7);
        constexpr sz_size_t partial_k = containment_distance(7, 5, 5);
        constexpr sz_size_t disjoint_k = containment_distance(5, 0, 5),
                            disjoint_longer_k = containment_distance(5, 0, 7);
        return {
            {make_value("abcdefgh", allocator), make_value("abcdefghi", allocator), make_value("zyxwvuts", allocator)},
            {{{0, superset_k, disjoint_k},
              {partial_k, 0, containment_distance(7, 0, 5)},
              {disjoint_k, disjoint_longer_k, 0}}}};
    }
    static std::vector<value_t> make_values(std::size_t count, u64_t seed, allocator_t const& allocator) {
        return generate_values(count, seed, overlap_alphabet_k, overlap_symbols_k, allocator);
    }
};

struct alignment_fixture_t {
    using index_t = sequence_index_t;
    using value_t = sequence_value_t;
    using allocator_t = index_t::allocator_t;
    using metric_t = alignment_metric_t;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static metric_t make_metric() noexcept {
        metric_t metric;
        [[maybe_unused]] status_t const sized = metric.init(sequence_staging_limit_k, nucleotide_costs_k);
        return metric;
    }
    static codec_t make_codec() noexcept { return {}; }
    static util::uuid128_t make_key(std::size_t ordinal) noexcept { return uuid_of(0x0A11C0DEull + ordinal); }
    static sz_size_t oracle(std::string_view query, std::string_view stored) {
        return alignment_reference(query, stored, nucleotide_costs_k);
    }
    static bool value_equals(std::string_view first, std::string_view second) noexcept { return first == second; }

    /** "ACGT", "ACGA" and "TTTT": one transversion apart, then two transversions and a transition,
     *  then three transversions and a transition. */
    static sequence_handmade_t make_handmade(allocator_t const& allocator) {
        return {{make_value(packed_nucleotides("ACGT"), allocator), make_value(packed_nucleotides("ACGA"), allocator),
                 make_value(packed_nucleotides("TTTT"), allocator)},
                {{{0, 2, 5}, {2, 0, 7}, {5, 7, 0}}}};
    }
    static std::vector<value_t> make_values(std::size_t count, u64_t seed, allocator_t const& allocator) {
        std::vector<value_t> values = generate_values(count, seed, overlap_alphabet_k, alignment_symbols_k, allocator);
        for (value_t& value : values) value = make_value(packed_nucleotides(std::string_view {value}), allocator);
        return values;
    }
};

/** The distinct bytes of @ref text_alphabet_k, one class apiece, so unit costs over them count byte
 *  edits, finer than rune edits wherever a rune is wide. */
constexpr std::string_view unit_letters_k = "abcdefg "                                     //
                                            "\xC3\xA9\xA8\xD0\xB6\xE8\xAA\x9E\xE6\xA4\x9C" //
                                            "\xE2\x82\xAC\xF0\x9F\x98\x80\x9A";

/** Unit costs over @ref unit_letters_k: one per differing pair and per gap. */
constexpr alignment_costs_t unit_costs() noexcept {
    alignment_costs_t costs {};
    costs.classes = unit_letters_k.size(), costs.gap = 1;
    for (std::size_t first = 0; first != unit_letters_k.size(); ++first)
        for (std::size_t second = 0; second != unit_letters_k.size(); ++second)
            costs.substitutions[first][second] = first != second;
    return costs;
}

constexpr alignment_costs_t unit_costs_k = unit_costs();

/** Alignment at unit costs over the bytes of runed text, each packed as its class. */
struct unit_alignment_fixture_t {
    using index_t = sequence_index_t;
    using value_t = sequence_value_t;
    using allocator_t = index_t::allocator_t;
    using metric_t = alignment_metric_t;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static metric_t make_metric() noexcept {
        metric_t metric;
        [[maybe_unused]] status_t const sized = metric.init(sequence_staging_limit_k, unit_costs_k);
        return metric;
    }
    static codec_t make_codec() noexcept { return {}; }
    static util::uuid128_t make_key(std::size_t ordinal) noexcept { return uuid_of(0x0C0DE5ull + ordinal); }
    static sz_size_t oracle(std::string_view query, std::string_view stored) {
        return alignment_reference(query, stored, unit_costs_k);
    }
    static bool value_equals(std::string_view first, std::string_view second) noexcept { return first == second; }

    /** "abcd", "abce" and "abc": one substitution apart, then one gap each. */
    static sequence_handmade_t make_handmade(allocator_t const& allocator) {
        return {{make_value(packed_letters("abcd"), allocator), make_value(packed_letters("abce"), allocator),
                 make_value(packed_letters("abc"), allocator)},
                {{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}}};
    }
    static std::vector<value_t> make_values(std::size_t count, u64_t seed, allocator_t const& allocator) {
        std::vector<value_t> values = generate_values(count, seed, text_alphabet_k, text_symbols_k, allocator);
        for (value_t& value : values) value = make_value(packed_letters(std::string_view {value}), allocator);
        return values;
    }
    static std::string packed_letters(std::string_view text) {
        std::vector<u32_t> codes;
        for (char const letter : text) codes.push_back(static_cast<u32_t>(unit_letters_k.find(letter)));
        return packed_sequence(codes);
    }
};

static_assert(a_value<sequence_value_t>);
static_assert(a_metric<containment_fixture_t::metric_t, sequence_value_t, sz_size_t>);
static_assert(a_metric<alignment_fixture_t::metric_t, sequence_value_t, sz_size_t>);
static_assert(a_query<std::string_view, containment_fixture_t::metric_t, sequence_value_t>);
static_assert(a_value_codec<codec_t, sequence_value_t>);

/** @p count strings as @c unit_alignment_fixture_t stores them, in host memory of no ledger. */
static std::vector<std::string> generate_packed_texts(std::size_t count, u64_t seed) {
    allocation_ledger_t ledger;
    std::vector<std::string> texts;
    for (sequence_value_t const& value :
         unit_alignment_fixture_t::make_values(count, seed, counting_allocator_t {ledger}))
        texts.emplace_back(std::string_view {value});
    return texts;
}

/** One containment kit against the brute-force window count: exact within the radius and above it
 *  otherwise, at an unbounded radius and at half the full scale, past which each pair drops by its
 *  own budget. Generated strings, a string shorter than every width, and the empty one, opening a
 *  round that shares no window. */
template <sz_capability_t capability_>
void test_containment_kit(u64_t seed) {
    constexpr std::size_t generated_k = 24, candidates_k = generated_k + 2;
    allocation_ledger_t ledger;
    {
        counting_allocator_t const allocator {ledger};
        std::vector<sequence_value_t> candidates = generate_values(generated_k, stream_key(seed, "overlap"),
                                                                   overlap_alphabet_k, overlap_symbols_k, allocator);
        candidates.push_back(make_value("ACGT", allocator));
        candidates.push_back(make_value("", allocator));

        containment_metric<typename containment_ladder<capability_>::kit_t> metric;
        expect_equal(metric.init(candidates_k), status_code_t::success_k);
        for (sz_size_t const radius : {unbounded_radius<sz_size_t>(), share_resolution_k / 2}) {
            // A batch scores every query as that query alone would, and the empty query
            // reopened right after it is prepared anew rather than taken for the batch.
            std::array<std::string_view, 2> const batch {candidates[0], candidates[candidates_k - 1]};
            metric.reset(std::span<std::string_view const>(batch));
            for (sequence_value_t const& candidate : candidates)
                expect_true(metric.push(candidate) == metric_staging_t::staged_k);
            std::array<sz_size_t, candidates_k * 2> batched;
            metric.drain(batched, radius);
            expect_equal(metric.status(), status_code_t::success_k);
            for (std::size_t slot = 0; slot != batched.size(); ++slot) {
                sz_size_t const truth = containment_reference(batch[slot / candidates_k],
                                                              candidates[slot % candidates_k]);
                truth <= radius ? expect_equal(batched[slot], truth) : expect_true(batched[slot] > radius);
            }
            for (std::size_t const query : {candidates_k - 1, std::size_t {0}}) {
                metric.reset(candidates[query]);
                for (sequence_value_t const& candidate : candidates)
                    expect_true(metric.push(candidate) == metric_staging_t::staged_k);
                std::array<sz_size_t, candidates_k> distances;
                metric.drain(distances, radius);
                expect_equal(metric.status(), status_code_t::success_k);
                for (std::size_t candidate = 0; candidate != candidates_k; ++candidate) {
                    sz_size_t const truth = containment_reference(candidates[query], candidates[candidate]);
                    truth <= radius ? expect_equal(distances[candidate], truth)
                                    : expect_true(distances[candidate] > radius);
                }
            }
        }
    }
    expect_equal(ledger.live_bytes, 0);
}

/** One alignment kit against the two-row oracle over five-bit codes. The fold first: the
 *  nucleotides' scores into the fixture's costs, a table whose gap passes 15 scaled into it, and a
 *  pair scoring above its mean self-score refused. Then every length from empty to one whose cost
 *  from a short query passes a 16-bit cell, exact within the radius and above it otherwise, at an
 *  unbounded radius and at a tight one that stops columns early. */
template <sz_capability_t capability_>
void test_alignment_kit(u64_t seed) {
    constexpr std::size_t side_k = alignment_byte_alphabet_k, classes_k = 24, generated_k = 40;
    constexpr std::size_t longest_k = 4500;
    constexpr sz_size_t tight_radius_k = 40;

    std::vector<i8_t> nucleotides(side_k * side_k);
    for (std::size_t first = 0; first != 4; ++first)
        for (std::size_t second = 0; second != 4; ++second)
            nucleotides[first * side_k + second] = first == second ? 2 : (first & 1) == (second & 1) ? 1 : 0;
    auto const [nucleotide_folded, nucleotide_status] = alignment_costs_t::fold(nucleotides, 2);
    expect_equal(nucleotide_status, status_code_t::success_k);
    expect_true(std::memcmp(&nucleotide_folded, &nucleotide_costs_k, sizeof(nucleotide_costs_k)) == 0);

    // Self-scores of 8 to 11 and cross-scores of -3 to 3 keep every cost at 14 or below, so the gap
    // of 20 alone scales the table by 3/4.
    std::vector<i8_t> similarities(side_k * side_k);
    for (std::size_t first = 0; first != classes_k; ++first)
        for (std::size_t second = 0; second != classes_k; ++second)
            similarities[first * side_k + second] = static_cast<i8_t>(
                first == second ? 8 + first % 4 : static_cast<int>(first * second % 7) - 3);
    auto const [costs, fold_status] = alignment_costs_t::fold(similarities, 20);
    expect_equal(fold_status, status_code_t::success_k);
    expect_equal(costs.classes, classes_k);
    expect_equal(costs.gap, 15u);
    // Codes 0 and 1 cost (8 + 9) / 2 + 3 = 11.5, scaled to 8.625 and rounded to 9.
    expect_equal(costs.substitutions[0][1], 9u);
    expect_equal(costs.substitutions[5][5], 0u);
    similarities[1] = 100;
    expect_equal(alignment_costs_t::fold(similarities, 20).status, status_code_t::config_mismatch_k);
    // A gap of one under a largest cost past twice the ceiling scales below a half, and stays one.
    std::vector<i8_t> steep(side_k * side_k);
    steep[0] = 20, steep[side_k + 1] = 21, steep[1] = steep[side_k] = -10;
    auto const [scaled, scaled_status] = alignment_costs_t::fold(steep, 1);
    expect_equal(scaled_status, status_code_t::success_k);
    expect_equal(scaled.gap, 1u);

    split_mix_generator_t random_generator {stream_key(seed, "needleman_wunsch")};
    uniform_below<u32_t> const code_distribution {classes_k}, length_distribution {60};
    std::vector<std::string> candidates;
    for (std::size_t candidate = 0; candidate != generated_k; ++candidate) {
        std::size_t const length = candidate == 0                 ? 0
                                   : candidate == 1               ? 1
                                   : candidate + 1 == generated_k ? longest_k
                                                                  : length_distribution(random_generator) + 1;
        std::vector<u32_t> codes(length);
        for (u32_t& code : codes) code = code_distribution(random_generator);
        candidates.push_back(packed_sequence(codes));
    }

    alignment_metric<typename alignment_ladder<capability_>::kit_t> metric;
    expect_equal(metric.init(generated_k, costs), status_code_t::success_k);
    for (sz_size_t const radius : {unbounded_radius<sz_size_t>(), tight_radius_k}) {
        for (std::size_t const query : {std::size_t {2}, std::size_t {0}, generated_k - 1}) {
            metric.reset(std::string_view {candidates[query]});
            for (std::string const& candidate : candidates)
                expect_true(metric.push(std::string_view {candidate}) == metric_staging_t::staged_k);
            std::array<sz_size_t, generated_k> distances;
            metric.drain(distances, radius);
            expect_equal(metric.status(), status_code_t::success_k);
            for (std::size_t candidate = 0; candidate != generated_k; ++candidate) {
                sz_size_t const truth = alignment_reference(candidates[query], candidates[candidate], costs);
                truth <= radius ? expect_equal(distances[candidate], truth)
                                : expect_true(distances[candidate] > radius);
            }
        }
    }
}

/** @p text as the packed class codes @p alphabet folds its bytes into. */
static std::string packed_classes(std::string_view text, alignment_alphabet_t const& alphabet) {
    std::string packed(packed_codes_bytes(text.size()), '\0');
    alphabet.pack(bytes_of(text.data(), text.size()), bytes_of(packed.data(), packed.size()));
    return packed;
}

/** @p text as the class codes @p alphabet folds its bytes into, one a byte. */
static std::string encoded_classes(std::string_view text, alignment_alphabet_t const& alphabet) {
    std::string encoded(text.size(), '\0');
    alphabet.encode(bytes_of(text.data(), text.size()), bytes_of(encoded.data(), encoded.size()));
    return encoded;
}

/** The reverse complement of @p text over A, C, G and T. */
static std::string complement_text(std::string_view text) {
    std::string image;
    for (std::size_t symbol = text.size(); symbol != 0; --symbol) image.push_back(dna_complement(text[symbol - 1]));
    return image;
}

/** The byte similarities an index scoring by @p metric folds unless it names its own. */
static std::vector<i8_t> own_similarities(usearch_sequence_metric_t metric) {
    std::vector<i8_t> similarities(alignment_byte_alphabet_k * alignment_byte_alphabet_k);
    byte_similarities_span_t const square {similarities};
    if (metric == usearch_sequence_metric_dna_overlap_k) dna_similarities(square);
    else protein_similarities(square);
    return similarities;
}

/** @p count strings of up to @p longest symbols drawn from @p alphabet, the empty one first. */
static std::vector<std::string> generate_texts(std::size_t count, u64_t seed, std::string_view alphabet,
                                               std::size_t longest) {
    split_mix_generator_t random_generator {seed};
    uniform_below<u32_t> const symbol_distribution {static_cast<u32_t>(alphabet.size())};
    uniform_below<u32_t> const length_distribution {static_cast<u32_t>(longest + 1)};
    std::vector<std::string> texts(count);
    for (std::size_t text = 1; text != count; ++text)
        for (std::size_t symbol = length_distribution(random_generator); symbol != 0; --symbol)
            texts[text].push_back(alphabet[symbol_distribution(random_generator)]);
    return texts;
}

/** The smooth sketch of the alignment kit @p capability_ picks against the serial kit's: the tables
 *  first, a window's agreement with itself decaying with the places between the two, then every
 *  packed sequence sketching alike, graded by what a substitution costs, the distance between two
 *  sketches summing their squared differences, and a narrower sketch the prefix of a wider one. */
template <sz_capability_t capability_>
void test_smooth_sketch_kit(u64_t seed) {
    using packed_kit_t = typename alignment_ladder<capability_>::kit_t;
    constexpr std::size_t texts_k = 10000, longest_k = 48, columns_k = 512;
    static_assert(a_sketching_kit<packed_kit_t>);

    alignment_alphabet_t alphabet {};
    alignment_costs_t costs {};
    std::vector<i8_t> const similarities = own_similarities(usearch_sequence_metric_dna_overlap_k);
    expect_equal(alignment_alphabet_t::fold(similarities, dna_gap_k, alphabet, costs), status_code_t::success_k);
    spectrum_tables<> spectrum;
    expect_equal(spectrum.derive(costs, columns_k), status_code_t::success_k);
    spectrum_view_t const tables = spectrum.view();
    // Four classes take six symbols a window.
    expect_equal(tables.head->classes, 4u);
    expect_equal(tables.head->width, 6u);
    alignment_costs_t free_costs = costs;
    std::memset(free_costs.substitutions, 0, sizeof(free_costs.substitutions));
    spectrum_tables<> refused;
    expect_equal(refused.derive(free_costs, columns_k), status_code_t::config_mismatch_k);
    expect_equal(refused.derive(costs, 6), status_code_t::config_mismatch_k);

    // One window read at two places agrees less the further apart they stand, and keeps
    // exp(-1/2) of its match a bandwidth on, 8.6 percent of the windows.
    constexpr std::size_t windows = 278, bandwidth = 24;
    std::vector<std::array<i32_t, columns_k>> added(windows);
    u8_t const classes[spectrum_width_max_k] = {};
    for (std::size_t first = 0; first != columns_k; first += spectrum_tile_columns_k) {
        spectrum_tile_t summed;
        spectrum_open(tables, windows, first, spectrum_tile_columns_k, summed);
        for (std::array<i32_t, columns_k>& window : added) {
            std::memcpy(window.data() + first, summed.sums, sizeof(summed.sums));
            packed_serial_alignment_kit_t::sketch_step(tables, classes, summed);
            for (std::size_t column = 0; column != spectrum_tile_columns_k; ++column)
                window[first + column] = summed.sums[column] - window[first + column];
        }
    }
    f64_t nearer = 2;
    for (std::size_t const shift :
         {std::size_t {0}, bandwidth / 4, bandwidth / 2, bandwidth * 3 / 4, bandwidth, bandwidth * 3 / 2}) {
        f64_t agreed = 0, first = 0, second = 0;
        for (std::size_t window = 0; window + shift != windows; ++window)
            for (std::size_t column = 0; column != columns_k; ++column) {
                f64_t const here = added[window][column], there = added[window + shift][column];
                agreed += here * there, first += here * here, second += there * there;
            }
        f64_t const agreement = agreed / std::sqrt(first * second);
        expect_true(agreement < nearer);
        // Within what 512 columns' own draws scatter a mean of cosines by.
        if (shift == bandwidth) expect_true(std::fabs(agreement - std::exp(-0.5)) < 0.05);
        nearer = agreement;
    }

    std::array<i8_t, columns_k> sketch, expected, previous {};
    auto const sketch_into = [&]<typename kit_type_>(std::type_identity<kit_type_>, std::string const& packed,
                                                     std::array<i8_t, columns_k>& written) {
        kit_type_::sketch_write(tables, bytes_of(packed.data(), packed.size()),
                                bytes_of(written.data(), written.size()));
    };
    // A read sits nearer its copy with a few transitions, which cost one, than its copy with as
    // many transversions at the same places, which cost two.
    constexpr std::size_t graded_reads_k = 16, graded_symbols_k = 160;
    split_mix_generator_t graded_generator {stream_key(seed, "graded_reads")};
    uniform_below<u32_t> const nucleotide_distribution {4};
    u64_t to_transitioned = 0, to_transversed = 0;
    for (std::size_t graded = 0; graded != graded_reads_k; ++graded) {
        std::string read;
        for (std::size_t symbol = 0; symbol != graded_symbols_k; ++symbol)
            read.push_back(overlap_alphabet_k[nucleotide_distribution(graded_generator)]);
        std::string transitioned = read, transversed = read;
        for (std::size_t place = 10; place < graded_symbols_k; place += graded_symbols_k / 4) {
            std::size_t const nucleotide = overlap_alphabet_k.find(read[place]);
            transitioned[place] = overlap_alphabet_k[nucleotide ^ 2],
            transversed[place] = overlap_alphabet_k[nucleotide ^ 1];
        }
        auto const distance_to = [&](std::string const& other) {
            sketch_into(std::type_identity<packed_kit_t> {}, packed_classes(read, alphabet), expected);
            sketch_into(std::type_identity<packed_kit_t> {}, packed_classes(other, alphabet), sketch);
            u32_t distance = 0;
            expect_equal(packed_kit_t::sketch_distance(expected.data(), sketch.data(), columns_k, &distance, nullptr),
                         nk_success_k);
            return distance;
        };
        expect_equal(distance_to(read), 0u);
        to_transitioned += distance_to(transitioned), to_transversed += distance_to(transversed);
    }
    expect_true(0 < to_transitioned && to_transitioned < to_transversed);

    for (std::string const& read :
         generate_texts(texts_k, stream_key(seed, "sketched_reads"), overlap_alphabet_k, longest_k)) {
        std::string const packed = packed_classes(read, alphabet);
        sketch_into(std::type_identity<packed_kit_t> {}, packed, sketch);
        sketch_into(std::type_identity<packed_serial_alignment_kit_t> {}, packed, expected);
        expect_true(std::memcmp(sketch.data(), expected.data(), columns_k) == 0);
        u32_t distance = 0, squared = 0;
        for (std::size_t column = 0; column != columns_k; ++column)
            squared += static_cast<u32_t>((sketch[column] - previous[column]) * (sketch[column] - previous[column]));
        expect_equal(packed_kit_t::sketch_distance(previous.data(), sketch.data(), columns_k, &distance, nullptr),
                     nk_success_k);
        expect_equal(distance, squared);
        previous = sketch;
    }

    // Tables of fewer columns, a pass and a tail among them, sketch the prefix of the widest.
    std::string const packed = packed_classes(std::string(longest_k, 'G') + "ACGTTGCA", alphabet);
    sketch_into(std::type_identity<packed_serial_alignment_kit_t> {}, packed, expected);
    for (std::size_t const narrower : {std::size_t {4}, spectrum_tile_columns_k + 4, columns_k - 4}) {
        spectrum_tables<> narrow;
        expect_equal(narrow.derive(costs, narrower), status_code_t::success_k);
        packed_kit_t::sketch_write(narrow.view(), bytes_of(packed.data(), packed.size()),
                                   bytes_of(sketch.data(), narrower));
        expect_true(std::memcmp(sketch.data(), expected.data(), narrower) == 0);
    }
}

/** The hash sketch of the containment kit @p capability_ picks against the serial kit's,
 *  its distance counting unequal slots, and a text nearer to one holding its windows
 *  than to another. */
template <sz_capability_t capability_>
void test_hash_sketch_kit(u64_t seed) {
    using kit_t = typename containment_ladder<capability_>::kit_t;
    constexpr std::size_t texts_k = 2000, longest_k = 600;
    constexpr std::size_t slots_k = hash_sketch_bytes_default_k;
    static_assert(a_sketching_kit<kit_t>);
    static_assert(!a_sketching_kit<typename banded_overlap_ladder<capability_>::kit_t>);
    std::array<u8_t, slots_k> sketch, expected, previous {};
    std::vector<byte_t> ranks(kit_t::sketch_scratch_bytes(slots_k));
    bytes_span_t const scratch = bytes_of(ranks.data(), ranks.size());
    std::size_t texts = 0;
    for (std::string const& text :
         generate_texts(texts_k, stream_key(seed, "hashed_texts"), text_alphabet_k, longest_k)) {
        bytes_view_t const stored = bytes_of(text.data(), text.size());
        kit_t::sketch_write({}, stored, bytes_of(sketch.data(), sketch.size()), scratch);
        utf8_serial_containment_kit_t::sketch_write({}, stored, bytes_of(expected.data(), expected.size()), scratch);
        expect_true(std::memcmp(sketch.data(), expected.data(), slots_k) == 0);
        u32_t distance = 0, unequal = 0;
        for (std::size_t slot = 0; slot != slots_k; ++slot) unequal += sketch[slot] != previous[slot];
        expect_equal(kit_t::sketch_distance(previous.data(), sketch.data(), slots_k, &distance, nullptr), nk_success_k);
        if (texts++) expect_equal(distance, unequal);
        previous = sketch;
    }
    std::string const text = generate_texts(2, stream_key(seed, "held"), "abcdefghijklmnopqrstuvwxyz ", 400)[1];
    std::string const other = generate_texts(2, stream_key(seed, "other"), "abcdefghijklmnopqrstuvwxyz ", 400)[1];
    auto const distance_between = [&](std::string_view first, std::string_view second) {
        kit_t::sketch_write({}, bytes_of(first.data(), first.size()), bytes_of(sketch.data(), sketch.size()), scratch);
        kit_t::sketch_write({}, bytes_of(second.data(), second.size()), bytes_of(expected.data(), expected.size()),
                            scratch);
        u32_t distance = 0;
        expect_equal(kit_t::sketch_distance(sketch.data(), expected.data(), slots_k, &distance, nullptr), nk_success_k);
        return distance;
    };
    if (text.size() > 60 && other.size() > 60) {
        std::string_view const held {text.data(), text.size() * 2 / 3};
        expect_true(distance_between(text, held) < distance_between(text, other));
        expect_equal(distance_between(text, text), 0u);
    }
}

/**
 *  @brief The banded overlap of @p query over @p stored by the reference the C header states,
 *      sharing no code with the kits.
 *
 *  Every window of @p stored found once in an image of the query votes for the shift between the
 *  two places, and the symbols the most voted shift lays over each other are aligned inside the
 *  band, every cell of it kept. Symbols are class codes charged through @p costs, and the query
 *  also scores as its reverse complement.
 */
static sz_size_t banded_overlap_reference(std::string_view query, std::string_view stored,
                                          alignment_costs_t const& costs) {
    constexpr u64_t ceiling_k = 32767 * 256 + 255, outside_k = u64_t {1} << 40;
    u64_t const gap = costs.gap;
    u64_t best = ceiling_k;
    std::vector<std::string> images {std::string {query}, std::string {query}};
    for (std::size_t symbol = 0; symbol != query.size(); ++symbol)
        images[1][symbol] = static_cast<char>(costs.complemented[static_cast<u8_t>(query[query.size() - 1 - symbol])]);
    for (std::string const& image : images) {
        if (image.empty() || stored.empty()) continue;
        std::map<std::ptrdiff_t, std::size_t> votes;
        for (std::size_t const width : banded_overlap_window_widths_k)
            for (std::size_t place = 0; place + width <= stored.size() && width <= image.size(); ++place) {
                std::string_view const window = stored.substr(place, width);
                std::size_t const first = image.find(window);
                if (first == std::string::npos || image.find(window, first + 1) != std::string::npos) continue;
                ++votes[static_cast<std::ptrdiff_t>(place) - static_cast<std::ptrdiff_t>(first)];
            }
        std::ptrdiff_t shift = 0;
        std::size_t held = 0;
        for (auto const& [voted, count] : votes)
            if (count > held) shift = voted, held = count;
        if (held < 2) continue;
        std::ptrdiff_t const low = max_of<std::ptrdiff_t>(0, -shift);
        std::ptrdiff_t const high = min_of<std::ptrdiff_t>(image.size(),
                                                           static_cast<std::ptrdiff_t>(stored.size()) - shift);
        std::size_t const span = static_cast<std::size_t>(high - low);
        std::size_t const band = min_of<std::size_t>(64, max_of<std::size_t>(4, span / 16));
        std::string_view const first = std::string_view {image}.substr(low, span),
                               second = stored.substr(low + shift, span);
        std::vector<std::vector<u64_t>> cells(span + 1, std::vector<u64_t>(span + 1, outside_k));
        for (std::size_t row = 0; row <= span; ++row)
            for (std::size_t column = row > band ? row - band : 0; column <= min_of(span, row + band); ++column) {
                if (!row || !column) {
                    cells[row][column] = (row + column) * gap;
                    continue;
                }
                u8_t const symbol = static_cast<u8_t>(first[row - 1]), other = static_cast<u8_t>(second[column - 1]);
                u64_t const substitution = costs.substitutions[symbol][other];
                cells[row][column] = min_of(cells[row - 1][column - 1] + substitution,
                                            min_of(cells[row - 1][column], cells[row][column - 1]) + gap);
            }
        std::size_t const shorter = min_of(image.size(), stored.size()), longer = max_of(image.size(), stored.size());
        u64_t const cost = cells[span][span] + gap * (shorter - span);
        u64_t const primary = min_of<u64_t>(32767, cost * 32767 / (2 * gap * shorter));
        u64_t const secondary = ((longer - span) * 255 + longer - 1) / longer;
        best = min_of(best, primary * 256 + secondary);
    }
    return static_cast<sz_size_t>(best);
}

/** One banded overlap kit against the reference: the header's three reads by the distances its
 *  Python states, then reads cut from one sequence with edits, as class codes under the
 *  nucleotides' costs, one query at a time and in a batch, at an unbounded radius and at one the
 *  lengths alone can pass. */
template <sz_capability_t capability_>
void test_banded_overlap_kit(u64_t seed) {
    using metric_t = banded_overlap_metric<typename banded_overlap_ladder<capability_>::kit_t>;
    constexpr std::size_t reads_k = 48, source_k = 600;

    alignment_alphabet_t alphabet {};
    alignment_costs_t costs {};
    expect_equal(
        alignment_alphabet_t::fold(own_similarities(usearch_sequence_metric_dna_overlap_k), dna_gap_k, alphabet, costs),
        status_code_t::success_k);
    expect_equal(alphabet.fold_complement(costs), status_code_t::success_k);

    // The reverse complement of the second read scores zero, the first read half a read off it, and
    // the third shares no window with any.
    constexpr std::array<std::string_view, 3> handmade_k {
        "ACGTTGCAAGGCTTAACCGGTTACGATCGA", "GGCTTAACCGGTTACGATCGATTTGCAGCA", "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"};
    constexpr std::array<sz_size_t, 3> handmade_distances_k {1258317, 0, 8388607};
    {
        metric_t metric;
        expect_equal(metric.init(handmade_k.size(), costs), status_code_t::success_k);
        std::string const asked = encoded_classes("TGCTGCAAATCGATCGTAACCGGTTAAGCC", alphabet);
        metric.reset(std::string_view {asked});
        std::array<std::string, 3> coded;
        for (std::size_t read = 0; read != handmade_k.size(); ++read) {
            coded[read] = encoded_classes(handmade_k[read], alphabet);
            expect_true(metric.push(std::string_view {coded[read]}) == metric_staging_t::staged_k);
        }
        std::array<sz_size_t, 3> distances;
        metric.drain(distances, unbounded_radius<sz_size_t>());
        expect_equal(metric.status(), status_code_t::success_k);
        for (std::size_t read = 0; read != handmade_k.size(); ++read)
            expect_equal(distances[read], handmade_distances_k[read]);
    }

    // Reads of 40 to 120 symbols cut from one sequence, a fifth of them as reverse
    // complements, each with a few substitutions, insertions and deletions; then the empty
    // read and a lone symbol.
    split_mix_generator_t random_generator {stream_key(seed, "overlapping_reads")};
    uniform_below<u32_t> const symbol_distribution {4}, start_distribution {source_k - 120}, length_distribution {81},
        edits_distribution {4}, kind_distribution {3}, complement_distribution {5};
    std::string source;
    for (std::size_t symbol = 0; symbol != source_k; ++symbol)
        source.push_back(overlap_alphabet_k[symbol_distribution(random_generator)]);
    std::vector<std::string> reads;
    for (std::size_t read = 0; read + 2 != reads_k; ++read) {
        std::string text = source.substr(start_distribution(random_generator),
                                         40 + length_distribution(random_generator));
        for (std::size_t edit = edits_distribution(random_generator); edit != 0; --edit) {
            std::size_t const position = uniform_below<u32_t> {static_cast<u32_t>(text.size())}(random_generator);
            char const symbol = overlap_alphabet_k[symbol_distribution(random_generator)];
            switch (kind_distribution(random_generator)) {
            case 0: text[position] = symbol; break;
            case 1: text.insert(text.begin() + position, symbol); break;
            default: text.erase(text.begin() + position); break;
            }
        }
        reads.push_back(complement_distribution(random_generator) ? text : complement_text(text));
    }
    reads.emplace_back();
    reads.emplace_back("A");
    std::vector<std::string> stored;
    for (std::string const& read : reads) stored.push_back(encoded_classes(read, alphabet));

    metric_t metric;
    expect_equal(metric.init(reads_k, costs), status_code_t::success_k);
    for (sz_size_t const radius : {unbounded_radius<sz_size_t>(), sz_size_t {100}}) {
        std::array<std::string_view, 3> const batch {stored[0], stored[reads_k - 1], stored[7]};
        metric.reset(std::span<std::string_view const>(batch));
        for (std::string const& read : stored)
            expect_true(metric.push(std::string_view {read}) == metric_staging_t::staged_k);
        std::array<sz_size_t, reads_k * 3> batched;
        metric.drain(batched, radius);
        expect_equal(metric.status(), status_code_t::success_k);
        for (std::size_t slot = 0; slot != batched.size(); ++slot) {
            sz_size_t const truth = banded_overlap_reference(batch[slot / reads_k], stored[slot % reads_k], costs);
            truth <= radius ? expect_equal(batched[slot], truth) : expect_true(batched[slot] > radius);
        }
        for (std::size_t const query : {std::size_t {3}, reads_k - 2, std::size_t {11}}) {
            metric.reset(std::string_view {stored[query]});
            for (std::string const& read : stored)
                expect_true(metric.push(std::string_view {read}) == metric_staging_t::staged_k);
            std::array<sz_size_t, reads_k> distances;
            metric.drain(distances, radius);
            expect_equal(metric.status(), status_code_t::success_k);
            for (std::size_t read = 0; read != reads_k; ++read) {
                sz_size_t const truth = banded_overlap_reference(stored[query], stored[read], costs);
                truth <= radius ? expect_equal(distances[read], truth) : expect_true(distances[read] > radius);
            }
        }
    }
}

/** The alignment fixture over 64-bit node IDs, which a saved neighbors column records. */
struct alignment_wide_fixture_t : unit_alignment_fixture_t {
    using index_t =
        sequence::ram_index<util::uuid128_t, u64_t, flaky_waiter_t, fu::standard_atomic_ref, counting_allocator_t>;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
};

/** Alignment over storage: generated strings, the empty one, and one crossing a 4 KiB block,
 *  ragged entries the view locates through its directory, viewed at 32-bit and 64-bit node IDs. */
struct alignment_nvme_fixture_t {
    using ram_fixture_t = unit_alignment_fixture_t;
    using wide_fixture_t = alignment_wide_fixture_t;
    using shard_index_t =
        sequence::ram_index<util::uuid128_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref, forkunion_allocator_t>;
    using scratch_metric_t = alignment_metric<alignment_ladder<sz_cap_any_k>::kit_t, basic_nvme_scratch_t>;
    using query_t = std::string_view;
    using coordinates_t = no_fixed_coordinates_t;
    static constexpr std::string_view metric_token_k = "protein-alignment";
    template <typename engine_type_, typename node_id_type_ = u32_t>
    using nvme_t =
        us::basic::nvme_indexes<codec_t, util::uuid128_t, node_id_type_, sz_size_t, engine_type_, fu::pool_t>;

    std::vector<std::string> texts;

    explicit alignment_nvme_fixture_t(u64_t seed) {
        texts = generate_packed_texts(30, stream_key(seed, "nvme"));
        texts.emplace_back();
        std::vector<u32_t> codes(8000, 0);
        codes[7000] = 1;
        texts.push_back(packed_sequence(codes));
    }
    std::size_t size() const noexcept { return texts.size(); }
    template <typename index_type_>
    typename index_type_::value_t make_value(std::size_t entry,
                                             typename index_type_::allocator_t const& allocator) const {
        typename index_type_::value_t value(allocator);
        [[maybe_unused]] sz::status_t const assigned = value.assign(texts[entry]);
        return value;
    }
    static expected<scratch_metric_t> make_scratch_metric(basic_nvme_scratch_t const& allocator,
                                                          std::size_t staging) noexcept {
        scratch_metric_t metric {allocator};
        status_t const ready = metric.init(staging, unit_costs_k);
        return {std::move(metric), ready};
    }
    expected<query_t> query_of(std::size_t entry, basic_nvme_scratch_t const&) const noexcept {
        return {texts[entry], status_code_t::success_k};
    }
    static expected<query_t> stored_query(std::string_view stored, basic_nvme_scratch_t const&) noexcept {
        return {stored, status_code_t::success_k};
    }
    bool matches(std::size_t entry, std::string_view view) const noexcept { return view == texts[entry]; }
};

/** 5000 strings over sharded indexes, as @c test_basic_sharded checks, against the alignment oracle
 *  over every string. */
void test_sequence_sharded(u64_t seed) {
    using index_t =
        sequence::ram_index<util::uuid128_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref, forkunion_allocator_t>;
    using value_t = index_t::value_t;
    constexpr std::size_t entries_k = 5000;
    std::vector<std::string> const texts = generate_packed_texts(entries_k + recall_samples_k,
                                                                 stream_key(seed, "plural"));
    forkunion_allocator_t const allocator {no_memory_domain_k};
    auto make_values = [&] {
        std::vector<value_t> values;
        values.reserve(entries_k);
        for (std::size_t entry = 0; entry != entries_k; ++entry) [[maybe_unused]]
            sz::status_t const assigned = values.emplace_back(allocator).assign(texts[entry]);
        return values;
    };
    std::vector<util::uuid128_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(0x91A7A100ull + entry);
    auto query_of = [&](std::size_t sample) noexcept { return std::string_view {texts[entries_k + sample]}; };
    auto best_of = [&](std::size_t sample) {
        sz_size_t best = (std::numeric_limits<sz_size_t>::max)();
        for (std::size_t entry = 0; entry != entries_k; ++entry)
            best = min_of(best, unit_alignment_fixture_t::oracle(query_of(sample), texts[entry]));
        return best;
    };
    unit_alignment_fixture_t::metric_t metric = unit_alignment_fixture_t::make_metric();
    return test_basic_sharded<us::basic::ram_indexes<index_t>>(seed, metric, unit_alignment_fixture_t::make_codec(),
                                                               std::span<util::uuid128_t const>(keys), make_values,
                                                               query_of, best_of);
}

/** Generated strings sampled from two shards, as @c test_basic_sharded_sample checks. */
void test_sequence_sample(u64_t seed) {
    using index_t =
        sequence::ram_index<util::uuid128_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref, forkunion_allocator_t>;
    using value_t = index_t::value_t;
    constexpr std::size_t entries_k = basic_growth_nodes_k;
    std::vector<std::string> const texts = generate_packed_texts(entries_k, stream_key(seed, "plural_sample"));
    forkunion_allocator_t const allocator {no_memory_domain_k};
    auto make_values = [&] {
        std::vector<value_t> values;
        for (std::string const& text : texts) [[maybe_unused]]
            sz::status_t const assigned = values.emplace_back(allocator).assign(text);
        return values;
    };
    std::vector<util::uuid128_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(0x5A3B1E00ull + entry);
    unit_alignment_fixture_t::metric_t metric = unit_alignment_fixture_t::make_metric();
    return test_basic_sharded_sample<us::basic::ram_indexes<index_t>>(
        seed, metric, std::span<util::uuid128_t const>(keys), make_values, no_fixed_coordinates_t {});
}

/** 600 strings in a sharded index of two shards cut into two sharded indexes of two, as
 *  @c test_basic_sharded_cut checks. */
void test_sequence_sharded_cut(u64_t seed) {
    using index_t =
        sequence::ram_index<util::uuid128_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref, forkunion_allocator_t>;
    using value_t = index_t::value_t;
    constexpr std::size_t entries_k = 600;
    std::vector<std::string> const texts = generate_packed_texts(entries_k, stream_key(seed, "plural_cut"));
    forkunion_allocator_t const allocator {no_memory_domain_k};
    std::vector<util::uuid128_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(0x5917A100ull + entry);
    auto make_values = [&] {
        std::vector<value_t> values;
        values.reserve(entries_k);
        for (std::size_t entry = 0; entry != entries_k; ++entry) [[maybe_unused]]
            sz::status_t const assigned = values.emplace_back(allocator).assign(texts[entry]);
        return values;
    };
    unit_alignment_fixture_t::metric_t metric = unit_alignment_fixture_t::make_metric();
    return test_basic_sharded_cut<us::basic::ram_indexes<index_t>>(seed, metric, unit_alignment_fixture_t::make_codec(),
                                                                   std::span<util::uuid128_t const>(keys), make_values);
}

/** Searches racing a clear of one shard read only retired entries nothing frees: each answers
 *  entries of the batch or @c contended_k, never a foreign key, while the clear retires the shard
 *  and a fresh one takes the next batch, which the only shard then names. A cursor opened before a
 *  clear answers @c contended_k after it. Under a sanitizer, a value freed under a search would
 *  trap. */
void test_sequence_sharded_clear_race(u64_t seed) {
    using index_t =
        sequence::ram_index<util::uuid128_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref, forkunion_allocator_t>;
    using indexes_t = us::basic::ram_indexes<index_t>;
    using value_t = index_t::value_t;
    using distance_t = index_t::distance_t;
    using metric_t = unit_alignment_fixture_t::metric_t;
    constexpr std::size_t entries_k = 400, searchers_k = 4, rounds_k = 4, wanted_k = 8;
    std::vector<std::string> const texts = generate_packed_texts(entries_k, stream_key(seed, "plural_clear_race"));
    forkunion_allocator_t const allocator {no_memory_domain_k};
    std::vector<util::uuid128_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(0xC1EA0000ull + entry);
    keep_diverse_t const selector {basic_connectivity_k};
    std::vector<index_t::traversal_context_t> contexts;
    std::vector<metric_t> metrics;
    contexts.reserve(searchers_k + 1);
    metrics.reserve(searchers_k + 1);
    for (std::size_t thread = 0; thread != searchers_k + 1; ++thread) {
        contexts.emplace_back(stream_key(seed, "plural_clear_race") + thread, allocator);
        expect_equal(index_t::reserve_traversal(contexts.back(), basic_connectivity_k, {}), status_code_t::success_k);
        metrics.push_back(unit_alignment_fixture_t::make_metric());
    }

    indexes_t indexes {allocator};
    expect_equal(indexes.emplace(forkunion_allocator_t {no_memory_domain_k}), status_code_t::success_k);
    auto insert_batch = [&] {
        std::vector<value_t> values;
        values.reserve(entries_k);
        for (std::size_t entry = 0; entry != entries_k; ++entry) [[maybe_unused]]
            sz::status_t const assigned = values.emplace_back(allocator).assign(texts[entry]);
        auto [plan, plan_status] = indexes.plan_insert(entries_k);
        expect_equal(plan_status, status_code_t::success_k);
        if (failed(plan_status)) return;
        plan.insert(0, std::span<value_t>(values), std::span<util::uuid128_t const>(keys), metrics[0], contexts[0],
                    selector);
        expect_equal(plan.finish(), status_code_t::success_k);
    };

    fu::topology_t topology;
    fu::pool_t pool;
    if (fu::failed(topology.harvest()) || fu::failed(pool.spawn(topology, searchers_k + 1, fu::caller_inclusive_k)))
        return void(expect_true(false));
    std::size_t foreign_count = 0;
    fu::standard_atomic_ref<std::size_t> const foreign {foreign_count};
    for (std::size_t round = 0; round != rounds_k; ++round) {
        insert_batch();
        expect_equal(indexes.size(), entries_k);
        indexes_t::cursor_t cursor {forkunion_allocator_t {no_memory_domain_k}};
        expect_true(succeeded(cursor.open(indexes, {})));
        bool cleared = false;
        std::size_t searched = 0;
        expect_true(!fu::failed(pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                // Every searcher walks the entries once first, so the clear runs under live walks.
                while (fu::standard_atomic_ref<std::size_t>(searched).load(std::memory_order_acquire) < searchers_k)
                    std::this_thread::yield();
                while (indexes.clear() == status_code_t::contended_k) std::this_thread::yield();
                fu::standard_atomic_ref<bool>(cleared).store(true, std::memory_order_release);
                return;
            }
            std::vector<u32_t> watermarks;
            std::array<util::uuid128_t, wanted_k> found_keys;
            std::array<distance_t, wanted_k> found_distances;
            dummy_predicate_t unfiltered;
            for (std::size_t entry = thread; !fu::standard_atomic_ref<bool>(cleared).load(std::memory_order_acquire);
                 entry = (entry + searchers_k) % entries_k) {
                watermarks.resize(indexes.shards_count());
                if (indexes.snapshot(watermarks).status == status_code_t::shape_mismatch_k) continue;
                auto const [found, status] = indexes.search_if(std::string_view {texts[entry]}, watermarks, found_keys,
                                                               found_distances, {}, unfiltered, metrics[thread],
                                                               contexts[thread]);
                if (entry == thread)
                    fu::standard_atomic_ref<std::size_t>(searched).store_add(1, std::memory_order_release);
                if (status == status_code_t::contended_k) continue;
                if (failed(status)) foreign.store_add(1);
                else
                    for (std::size_t match = 0; match != found; ++match)
                        foreign.store_add(std::find(keys.begin(), keys.end(), found_keys[match]) == keys.end());
            }
        })));
        expect_equal(indexes.size(), std::size_t {0});
        expect_equal(indexes.count(keys[0]), std::size_t {0});
        expect_equal(indexes.shards_count(), round + 2);
        auto const live = indexes.live_shards();
        expect_equal(live.bound - live.first, std::size_t {1});
        std::array<util::uuid128_t, 4> read_keys;
        expect_equal(cursor.pull(indexes, std::span<util::uuid128_t> {read_keys}, nullptr, nullptr).status,
                     status_code_t::contended_k);
    }
    expect_equal(foreign_count, std::size_t {0});

    // The fresh shard takes a batch whole, and a clear of an empty index retires nothing.
    insert_batch();
    expect_equal(indexes.size(), entries_k);
    expect_equal(indexes.count(keys[entries_k - 1]), std::size_t {1});
    expect_equal(indexes.clear(), status_code_t::success_k);
    expect_equal(indexes.clear(), status_code_t::success_k);
    expect_equal(indexes.shards_count(), rounds_k + 2);
}

/** A stored value is its sketch, then its sequence: the codec of an index storing sketches loads
 *  every sequence behind the sketch of the index's bytes the widest kit writes in place, which
 *  equals the serial kit's of the same sequence, saves the sequence alone, and weighs a value whole
 *  only for a round's cache, for smooth and hash sketches alike; a kit that sketches nothing stores
 *  the bare sequence. */
void test_sequence_stored_sketches(u64_t seed) {
    constexpr std::size_t entries_k = 64;
    using text_kit_t = containment_ladder<sz_cap_any_k>::kit_t;
    using text_serial_kit_t = containment_ladder<sz_cap_serial_k>::kit_t;
    using packed_kit_t = alignment_ladder<sz_cap_any_k>::kit_t;
    using packed_serial_kit_t = alignment_ladder<sz_cap_serial_k>::kit_t;
    using overlap_kit_t = banded_overlap_ladder<sz_cap_any_k>::kit_t;
    allocation_ledger_t ledger;
    counting_allocator_t const allocator {ledger};
    auto const expect_stored = [&]<typename kit_type_, typename serial_kit_type_>(
                                   std::type_identity<kit_type_>, std::type_identity<serial_kit_type_>,
                                   typename kit_type_::sketch_tables_t const& tables, std::size_t sketched,
                                   std::string_view sequence) {
        stored_codec<kit_type_> const codec {tables, sketched};
        sequence_value_t value(allocator);
        std::vector<byte_t> scratch(codec.scratch_bytes());
        expect_equal(codec.load(bytes_of(sequence.data(), sequence.size()), value, scratch), status_code_t::success_k);
        if (!expect_equal(value.size(), sketched + sequence.size())) return;
        expect_true(std::string_view {value}.substr(sketched) == sequence);
        if constexpr (a_sketching_kit<kit_type_>) {
            std::vector<byte_t> serial(sketched);
            serial_kit_type_::sketch_write(tables, bytes_of(sequence.data(), sequence.size()), serial, scratch);
            expect_true(std::memcmp(value.data(), serial.data(), sketched) == 0);
        }
        expect_equal(codec.storage_requirements(value), sequence.size());
        expect_equal(codec_t {}.storage_requirements(value), value.size());
        std::string saved(sequence.size(), '\0');
        expect_equal(codec.save(value, bytes_of(saved.data(), saved.size())), status_code_t::success_k);
        expect_true(saved == sequence);
    };

    // Text, the empty one and one shorter than a window among it.
    std::vector<sequence_value_t> const texts = generate_values(entries_k, stream_key(seed, "texts"), text_alphabet_k,
                                                                text_symbols_k, allocator);
    for (sequence_value_t const& text : texts)
        expect_stored(std::type_identity<text_kit_t> {}, std::type_identity<text_serial_kit_t> {}, no_sketch_tables_k,
                      hash_sketch_bytes_default_k, std::string_view {text});
    for (std::string_view const text : {std::string_view {}, std::string_view {"ab"}}) {
        expect_stored(std::type_identity<text_kit_t> {}, std::type_identity<text_serial_kit_t> {}, no_sketch_tables_k,
                      hash_sketch_bytes_default_k, text);
        expect_stored(std::type_identity<overlap_kit_t> {}, std::type_identity<overlap_kit_t> {}, no_sketch_tables_k, 0,
                      text);
    }

    // Folded costs over the nucleotides, packed, behind sketches of a pass and a tail.
    constexpr std::size_t packed_sketch_bytes_k = spectrum_tile_columns_k + 4;
    spectrum_tables<> spectrum;
    expect_equal(spectrum.derive(nucleotide_costs_k, packed_sketch_bytes_k), status_code_t::success_k);
    split_mix_generator_t random_generator {stream_key(seed, "packed")};
    for (std::size_t entry = 0; entry != entries_k; ++entry) {
        std::vector<u32_t> codes(entry + random_generator.below(64));
        for (u32_t& code : codes) code = static_cast<u32_t>(random_generator.below(4));
        expect_stored(std::type_identity<packed_kit_t> {}, std::type_identity<packed_serial_kit_t> {}, spectrum.view(),
                      packed_sketch_bytes_k, packed_sequence(codes));
    }
}

/** The two roles read the two parts of one value. Entries in two groups by their sketches hold
 *  sequences whose twins sit in the other group, and start linked to those twins alone, as the
 *  canonical score would link them: a refinement under the sketch distance writes no link out of an
 *  entry's group and drops most twins, while a canonical search still answers a sequence with its
 *  two holders at no distance. */
void test_sequence_roles(u64_t seed) {
    using kit_t = alignment_ladder<sz_cap_any_k>::kit_t;
    using canonical_t = alignment_metric<kit_t>;
    using node_id_t = sequence_index_t::node_id_t;
    constexpr std::size_t pairs_k = 96, entries_k = 2 * pairs_k, length_k = 40;
    constexpr us::basic::keep_diverse_t selector {connectivity_32_k};
    allocation_ledger_t ledger;
    counting_allocator_t const allocator {ledger};
    split_mix_generator_t random_generator {seed};

    // An entry and its twin, `pairs_k` entries on, share a sequence; their sketches sit far apart.
    std::vector<std::string> sequences;
    for (std::size_t pair = 0; pair != pairs_k; ++pair) {
        std::vector<u32_t> codes(length_k);
        for (u32_t& code : codes) code = static_cast<u32_t>(random_generator.below(4));
        sequences.push_back(packed_sequence(codes));
    }
    std::vector<sequence_value_t> values;
    std::vector<util::uuid128_t> keys;
    for (std::size_t entry = 0; entry != entries_k; ++entry) {
        std::string_view const sequence = sequences[entry % pairs_k];
        i8_t const center = entry < pairs_k ? i8_t {-96} : i8_t {96};
        sequence_value_t& value = values.emplace_back(allocator);
        std::size_t const bytes = smooth_sketch_bytes_default_k + sequence.size();
        [[maybe_unused]] sz::status_t const written = value.resize_and_overwrite(
            bytes, [&](char* stored, std::size_t) noexcept {
                for (i8_t& column : std::span {reinterpret_cast<i8_t*>(stored), smooth_sketch_bytes_default_k})
                    column = static_cast<i8_t>(center + static_cast<int>(random_generator.below(17)) - 8);
                std::memcpy(stored + smooth_sketch_bytes_default_k, sequence.data(), sequence.size());
                return bytes;
            });
        keys.push_back(uuid_of(0x51E7C4ull + entry));
    }

    spectrum_tables<> spectrum;
    expect_equal(spectrum.derive(nucleotide_costs_k, smooth_sketch_bytes_default_k), status_code_t::success_k);
    stored_sketch_metric<kit_t, sequence_value_t> sketched;
    stored_canonical_metric<canonical_t, sequence_value_t> canonical;
    expect_equal(
        sketched.init(sequence_staging_limit_k * connectivity_32_k, smooth_sketch_bytes_default_k, spectrum.view()),
        status_code_t::success_k);
    expect_equal(
        canonical.init(sequence_staging_limit_k * connectivity_32_k, smooth_sketch_bytes_default_k, nucleotide_costs_k),
        status_code_t::success_k);
    std::unique_ptr<sequence_index_t> const index_ptr(new (std::nothrow) sequence_index_t(allocator));
    sequence_index_t& index = *index_ptr;
    sequence_index_t::traversal_context_t context {seed, allocator};
    expect_true(bool(index.reserve(entries_k)));
    expect_equal(index.insert(values, keys, sketched, {}, context, selector).status, status_code_t::success_k);
    std::vector<node_id_t> twins(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry)
        twins[entry] = static_cast<node_id_t>((entry + pairs_k) % entries_k);
    expect_equal(index.adopt_links({twins, {}, 1}), status_code_t::success_k);

    auto const [refined, refine_status] = index.refine({}, selector, codec_t {}, sketched, context);
    expect_equal(refine_status, status_code_t::success_k);
    expect_true(refined.added_links > 0 && refined.changed_nodes > 0);
    auto const [columns, save_status] = basic_saved_columns(
        index, stored_codec<kit_t> {spectrum.view(), smooth_sketch_bytes_default_k});
    expect_true(succeeded(save_status));
    // The canonical score keeps every twin, at no distance, and half its links leave the group.
    std::size_t strays = 0, twinned = 0;
    for (std::size_t entry = 0; succeeded(save_status) && entry != entries_k; ++entry)
        for (u64_t link = columns.neighbor_offsets[entry]; link != columns.neighbor_offsets[entry + 1]; ++link) {
            bool const twin = columns.neighbor_ids[link] == twins[entry];
            twinned += twin;
            strays += !twin && (columns.neighbor_ids[link] < pairs_k) != (entry < pairs_k);
        }
    expect_equal(strays, std::size_t {0});
    expect_true(4 * twinned < entries_k, "{} of {} entries still link their twin", twinned, entries_k);

    // A search reads the sequences: both holders of one at no distance, whatever their sketches.
    for (std::size_t pair = 0; pair != pairs_k; ++pair) {
        expect_equal(index.search(std::string_view {sequences[pair]}, {entries_k}, canonical, context).status,
                     status_code_t::success_k);
        if (!expect_true(context.results.size() >= 2)) break;
        expect_equal(context.results[0].distance, sz_size_t {0});
        expect_equal(context.results[1].distance, sz_size_t {0});
        expect_equal(static_cast<std::size_t>(context.results[0].node_id) % pairs_k, pair);
        expect_equal(static_cast<std::size_t>(context.results[1].node_id) % pairs_k, pair);
    }
}

/** Protein clans, their families and the families' members, each a mutation of the one above, built
 *  on their sketches in two batches, saved, reloaded and grown by a third, answer every stored
 *  entry with itself at rank one at no distance once a walk may hold them all, so the links every
 *  batch scored by sketches reach every entry under the canonical score, and hand every sequence
 *  back from @c get as its packed class codes with no sketch in front; the checkpoint, loaded in
 *  host memory and viewed on storage where this machine opens an NVMe executor, answers alike: in
 *  host memory, or on the device the listing names at @p ordinal. */
void test_sequence_sketched_round_trip(u64_t seed, usearch_residency_t residency, std::int32_t ordinal) {
    constexpr std::size_t clans_k = 32, families_k = 8, members_k = 8, entries_k = clans_k * families_k * members_k;
    constexpr std::size_t shortest_k = 80, longest_k = 140, standard_codes_k = 20, wanted_k = 10, expansion_k = 256;
    checks_held_t const held;

    // A family substitutes a fifth of its clan's codes, and a member a twentieth of its family's.
    split_mix_generator_t random_generator {seed};
    auto const mutated = [&](std::vector<u8_t> codes, std::size_t share) {
        for (std::size_t edit = 0; edit != codes.size() / share; ++edit)
            codes[random_generator.below(codes.size())] = static_cast<u8_t>(random_generator.below(standard_codes_k));
        return codes;
    };
    std::string tape;
    std::vector<std::size_t> offsets {0};
    for (std::size_t clan = 0; clan != clans_k; ++clan) {
        std::vector<u8_t> root(shortest_k + random_generator.below(longest_k - shortest_k + 1));
        for (u8_t& code : root) code = static_cast<u8_t>(random_generator.below(standard_codes_k));
        for (std::size_t family = 0; family != families_k; ++family) {
            std::vector<u8_t> const family_root = mutated(root, 5);
            for (std::size_t member = 0; member != members_k; ++member) {
                for (u8_t const code : mutated(family_root, 20)) tape += protein_letters_k[code];
                offsets.push_back(tape.size());
            }
        }
    }
    std::vector<usearch_key_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = us::tools::key_of_offset(seed, entry);

    usearch_outcome_t outcome {};
    usearch_arena_t arena = nullptr;
    usearch_arena_init_t arena_init {};
    arena_init.arena = &arena;
    arena_init.outcome = &outcome;
    usearch_arena_init(&arena_init);
    expect_true(!outcome.message);
    std::int32_t const devices[] = {ordinal};
    usearch_vram_options_t const vram {devices, 1};
    usearch_sequence_index_t index = nullptr;
    usearch_executor_t executor = nullptr;
    if (residency == usearch_residency_ram_k) {
        usearch_sequence_init_ram_t init {};
        init.index = &index, init.outcome = &outcome;
        init.metric = usearch_sequence_metric_protein_alignment_k;
        if (held() && !outcome.message) usearch_sequence_init_ram(&init);
        usearch_executor_for_ram_t placement {};
        placement.executor = &executor, placement.outcome = &outcome;
        if (held() && !outcome.message) usearch_executor_for_ram(&placement);
    }
    else {
        usearch_sequence_init_vram_t init {};
        init.index = &index, init.outcome = &outcome;
        init.metric = usearch_sequence_metric_protein_alignment_k;
        init.vram = vram;
        if (held() && !outcome.message) usearch_sequence_init_vram(&init);
        usearch_executor_for_vram_t placement {};
        placement.executor = &executor, placement.outcome = &outcome;
        placement.vram = vram;
        if (held() && !outcome.message) usearch_executor_for_vram(&placement);
    }
    expect_true(!outcome.message);

    auto const insert_run = [&](usearch_sequence_index_t into, std::size_t first, std::size_t bound) {
        usearch_sequence_insert_t insert {};
        insert.index = into;
        insert.executor = executor;
        insert.outcome = &outcome;
        insert.count = bound - first;
        insert.keys = {keys.data() + first, sizeof(usearch_key_t)};
        insert.sequences = tools::tape_of(tape.data(), offsets.data() + first, usearch_scalar_u8_k);
        insert.expansion_factor = expansion_k;
        if (held() && !outcome.message) usearch_sequence_insert(&insert);
        expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
    };
    usearch_tape_view_t const sequences = tools::tape_of(tape.data(), offsets.data(), usearch_scalar_u8_k);
    auto const expect_entries_first = [&](usearch_sequence_index_t searched, usearch_executor_t searching,
                                          std::size_t checked) {
        usearch_matches_view_t matches {};
        usearch_sequence_search_t search {};
        search.index = searched;
        search.executor = searching;
        search.outcome = &outcome;
        search.count = checked;
        search.queries = sequences;
        search.matches = &matches;
        search.arena = arena;
        search.wanted = wanted_k;
        search.expansion_factor = entries_k;
        if (held() && !outcome.message) usearch_sequence_search(&search);
        expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
        std::size_t found = 0;
        for (std::size_t entry = 0; held() && !outcome.message && entry != checked; ++entry)
            found += matches.counts[entry] && matches.scores[entry * matches.columns] == 0 &&
                     std::memcmp(&matches.keys[entry * matches.columns], &keys[entry], sizeof(usearch_key_t)) == 0;
        expect_equal(found, checked);
    };
    insert_run(index, 0, entries_k / 2);
    insert_run(index, entries_k / 2, entries_k * 3 / 4);

    // The checkpoint keeps the sequences and the links alone; a load derives every sketch again.
    scratch_directory_t const scratch;
    std::string const path = scratch.path("sketched.safegraphs"), grown_path = scratch.path("grown.safegraphs");
    auto const save_to = [&](std::string const& saved_path) {
        usearch_sequence_save_t save {};
        save.index = index, save.executor = executor, save.outcome = &outcome;
        save.path = tools::bytes_view(std::string_view {saved_path});
        if (held() && !outcome.message) usearch_sequence_save(&save);
        expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
    };
    save_to(path);
    usearch_sequence_free(index);
    index = nullptr;
    if (residency == usearch_residency_ram_k) {
        usearch_sequence_load_ram_t load {};
        load.index = &index, load.executor = executor, load.outcome = &outcome;
        load.path = tools::bytes_view(std::string_view {path});
        if (held() && !outcome.message) usearch_sequence_load_ram(&load);
    }
    else {
        usearch_sequence_load_vram_t load {};
        load.index = &index, load.executor = executor, load.outcome = &outcome;
        load.path = tools::bytes_view(std::string_view {path});
        load.vram = vram;
        if (held() && !outcome.message) usearch_sequence_load_vram(&load);
    }
    expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
    insert_run(index, entries_k * 3 / 4, entries_k);
    expect_entries_first(index, executor, entries_k);

    // The metric's own costs fold the letters into classes, so a sequence reads back as
    // its packed codes.
    alignment_alphabet_t folded {};
    alignment_costs_t folded_costs {};
    expect_equal(alignment_alphabet_t::fold(own_similarities(usearch_sequence_metric_protein_alignment_k),
                                            protein_gap_k, folded, folded_costs),
                 status_code_t::success_k);
    for (std::size_t entry = 0; entry < entries_k && held() && !outcome.message; entry += members_k * families_k) {
        std::vector<u32_t> codes;
        for (std::size_t symbol = offsets[entry]; symbol != offsets[entry + 1]; ++symbol)
            codes.push_back(folded.codes[static_cast<u8_t>(tape[symbol])]);
        std::string const expected = packed_sequence(codes);
        usearch_tape_view_t got {};
        std::size_t written = 0;
        usearch_sequence_get_t get {};
        get.index = index, get.executor = executor, get.outcome = &outcome;
        get.key = keys[entry];
        get.sequences = &got, get.written = &written;
        get.arena = arena;
        usearch_sequence_get(&get);
        expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
        if (outcome.message) break;
        expect_true(written == 1 && tools::tape_offset(got, 1) - tools::tape_offset(got, 0) == expected.size() &&
                    std::memcmp(static_cast<char const*>(got.data) + tools::tape_offset(got, 0), expected.data(),
                                expected.size()) == 0);
    }

    // The grown index's checkpoint answers a clan's entries alike once host memory holds it.
    save_to(grown_path);
    usearch_executor_t host_executor = nullptr;
    usearch_executor_for_ram_t host_placement {};
    host_placement.executor = &host_executor, host_placement.outcome = &outcome;
    if (held() && !outcome.message) usearch_executor_for_ram(&host_placement);
    usearch_sequence_index_t host = nullptr;
    usearch_sequence_load_ram_t host_load {};
    host_load.index = &host, host_load.executor = host_executor, host_load.outcome = &outcome;
    host_load.path = tools::bytes_view(std::string_view {grown_path});
    if (held() && !outcome.message) usearch_sequence_load_ram(&host_load);
    expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
    expect_entries_first(host, host_executor, families_k * members_k);

    // A view holds no sketch and walks the same links by the canonical score alone.
    usearch_executor_t storage = nullptr;
    usearch_outcome_t storage_outcome {};
    usearch_executor_for_nvme_t storage_placement {};
    storage_placement.executor = &storage, storage_placement.outcome = &storage_outcome;
    if (held() && !outcome.message) usearch_executor_for_nvme(&storage_placement);
    usearch_sequence_index_t viewed = nullptr;
    if (storage) {
        usearch_sequence_view_nvme_t view {};
        view.index = &viewed, view.executor = storage, view.outcome = &outcome;
        view.path = tools::bytes_view(std::string_view {grown_path});
        usearch_sequence_view_nvme(&view);
        expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
        expect_entries_first(viewed, storage, families_k * members_k);
    }

    usearch_sequence_free(viewed);
    usearch_sequence_free(host);
    usearch_sequence_free(index);
    usearch_executor_free(storage);
    usearch_executor_free(executor);
    usearch_executor_free(host_executor);
    usearch_arena_free(arena);
}

/** One score through the C interface on the host: an index under @p metric and @p costs over
 *  @p values, each searched exhaustively, whose every match scores exactly what @p oracle says. */
template <typename oracle_type_>
void expect_scores_through_abi(u64_t seed, usearch_sequence_metric_t metric, usearch_sequence_costs_t const* costs,
                               std::vector<sequence_value_t> const& values, oracle_type_&& oracle) {
    std::string tape;
    std::vector<std::size_t> offsets {0};
    for (sequence_value_t const& value : values) tape += std::string_view {value}, offsets.push_back(tape.size());
    std::vector<usearch_key_t> keys(values.size());
    for (std::size_t entry = 0; entry != values.size(); ++entry) keys[entry] = us::tools::key_of_offset(seed, entry);

    usearch_outcome_t outcome {};
    usearch_arena_t arena = nullptr;
    usearch_arena_init_t arena_init {};
    arena_init.arena = &arena, arena_init.outcome = &outcome;
    usearch_arena_init(&arena_init);
    usearch_sequence_index_t index = nullptr;
    usearch_sequence_init_ram_t init {};
    init.index = &index, init.outcome = &outcome, init.metric = metric, init.costs = costs;
    if (!outcome.message) usearch_sequence_init_ram(&init);
    usearch_executor_t executor = nullptr;
    usearch_executor_for_ram_t placement {};
    placement.executor = &executor, placement.outcome = &outcome;
    if (!outcome.message) usearch_executor_for_ram(&placement);
    usearch_tape_view_t const sequences = tools::tape_of(tape.data(), offsets.data(), usearch_scalar_u8_k);
    usearch_sequence_insert_t insert {};
    insert.index = index, insert.executor = executor, insert.outcome = &outcome, insert.count = values.size();
    insert.keys = {keys.data(), sizeof(usearch_key_t)}, insert.sequences = sequences;
    if (!outcome.message) usearch_sequence_insert(&insert);
    usearch_matches_view_t matches {};
    auto const search_all = [&](usearch_sequence_index_t searched) {
        usearch_sequence_search_t search {};
        search.index = searched, search.executor = executor, search.outcome = &outcome, search.count = values.size();
        search.queries = sequences, search.matches = &matches, search.arena = arena;
        search.wanted = values.size(), search.expansion_factor = values.size();
        if (!outcome.message) usearch_sequence_search(&search);
        expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
    };
    search_all(index);

    // A description counts every stored value whole, its sketch included.
    usearch_sequence_index_description_t description {};
    usearch_sequence_describe_t describe {};
    describe.index = index, describe.executor = executor, describe.outcome = &outcome;
    describe.description = &description;
    if (!outcome.message) usearch_sequence_describe(&describe);
    expect_true(description.memory_usage >= values.size() * description.metadata.sketch_bytes);
    expect_equal(description.metadata.sketch_bytes, metric == usearch_sequence_metric_dna_overlap_k ? 0u : 512u);

    std::size_t scored = 0, misscored = 0;
    for (std::size_t query = 0; !outcome.message && query != values.size(); ++query)
        for (std::size_t column = 0; column != matches.counts[query]; ++column) {
            usearch_key_t const& key = matches.keys[query * matches.columns + column];
            std::size_t const entry = static_cast<std::size_t>(
                std::find_if(keys.begin(), keys.end(),
                             [&](usearch_key_t const& stored) { return !std::memcmp(&stored, &key, sizeof key); }) -
                keys.begin());
            ++scored;
            misscored += entry == keys.size() || matches.scores[query * matches.columns + column] !=
                                                     static_cast<double>(oracle(std::string_view {values[query]},
                                                                                std::string_view {values[entry]}));
        }
    expect_true(scored >= values.size());
    expect_equal(misscored, std::size_t {0});

    // The checkpoint records whatever the costs folded, so a load naming none answers the same raw
    // queries with the same matches.
    std::vector<usearch_key_t> const saved_keys(matches.keys, matches.keys + values.size() * matches.columns);
    std::vector<usearch_distance_t> const saved_scores(matches.scores,
                                                       matches.scores + values.size() * matches.columns);
    std::vector<std::size_t> const saved_counts(matches.counts, matches.counts + values.size());
    usearch_arena_t saving = nullptr;
    arena_init.arena = &saving;
    if (!outcome.message) usearch_arena_init(&arena_init);
    usearch_bytes_view_t monolith {};
    usearch_sequence_save_buffer_t save {};
    save.index = index, save.executor = executor, save.outcome = &outcome, save.monolith = &monolith;
    save.arena = saving;
    if (!outcome.message) usearch_sequence_save_buffer(&save);
    usearch_sequence_index_t loaded = nullptr;
    usearch_sequence_load_ram_buffer_t load {};
    load.index = &loaded, load.executor = executor, load.outcome = &outcome, load.buffer = monolith;
    if (!outcome.message) usearch_sequence_load_ram_buffer(&load);
    search_all(loaded);
    std::size_t differing = 0;
    for (std::size_t query = 0; !outcome.message && query != values.size(); ++query) {
        differing += matches.counts[query] != saved_counts[query];
        for (std::size_t column = 0; column != min_of(matches.counts[query], saved_counts[query]); ++column) {
            std::size_t const slot = query * matches.columns + column;
            differing += std::memcmp(&matches.keys[slot], &saved_keys[slot], sizeof(usearch_key_t)) != 0 ||
                         matches.scores[slot] != saved_scores[slot];
        }
    }
    expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
    expect_equal(differing, std::size_t {0});

    // Past the last read of the matches, as a search rewinds the arena: a missing hint is counted,
    // and a sketching metric's walk by sketches spends distances counted apart.
    usearch_key_t const absent = us::tools::key_of_offset(~seed, 0);
    std::size_t const hinted = 1;
    usearch_matches_view_t const hints {&absent, nullptr, &hinted, 1};
    usearch_traversal_stats_t stats {};
    usearch_matches_view_t hinted_matches {};
    usearch_sequence_search_t hinted_search {};
    hinted_search.index = index, hinted_search.executor = executor, hinted_search.outcome = &outcome;
    hinted_search.count = 1, hinted_search.queries = sequences, hinted_search.matches = &hinted_matches;
    hinted_search.arena = arena, hinted_search.wanted = 1, hinted_search.hints = &hints, hinted_search.stats = &stats;
    if (!outcome.message) usearch_sequence_search(&hinted_search);
    expect_true(!outcome.message, "{}", outcome.message ? outcome.message : "");
    expect_equal(stats.missing_hints, std::size_t {1});
    expect_equal(stats.seeding_distances != 0, description.metadata.sketch_bytes != 0);
    expect_true(stats.seeding_distances < stats.computed_distances);

    usearch_sequence_free(loaded);
    usearch_sequence_free(index);
    usearch_executor_free(executor);
    usearch_arena_free(saving);
    usearch_arena_free(arena);
}

/** The nucleotide similarities over raw bytes score the costed oracle exactly on packed codes. */
void test_alignment_costs(u64_t seed) {
    constexpr std::size_t entries_k = 48, side_k = alignment_byte_alphabet_k;
    allocation_ledger_t ledger;
    counting_allocator_t const allocator {ledger};
    // Self-scores of two, one between purines or between pyrimidines: the nucleotide costs.
    std::vector<i8_t> similarities(side_k * side_k, 0);
    for (std::size_t first = 0; first != 4; ++first)
        for (std::size_t second = 0; second != 4; ++second)
            similarities[static_cast<u8_t>(overlap_alphabet_k[first]) * side_k +
                         static_cast<u8_t>(overlap_alphabet_k[second])] = first == second               ? 2
                                                                          : (first & 1) == (second & 1) ? 1
                                                                                                        : 0;
    usearch_sequence_costs_t const costs {similarities.data(), 2};
    auto const costed_reference = [](std::string_view query, std::string_view stored) {
        return alignment_reference(packed_nucleotides(query), packed_nucleotides(stored), nucleotide_costs_k);
    };
    expect_scores_through_abi(seed, usearch_sequence_metric_protein_alignment_k, &costs,
                              containment_fixture_t::make_values(entries_k, seed, allocator), costed_reference);
}

/** The metrics' own similarities fold into their classes: 24 for proteins and four for DNA, whose
 *  complements pair A with T and C with G, U read as T. */
void test_alignment_own_costs() {
    alignment_alphabet_t proteins {}, nucleotides {};
    alignment_costs_t protein_costs {}, dna_costs {};
    expect_equal(alignment_alphabet_t::fold(own_similarities(usearch_sequence_metric_protein_alignment_k),
                                            protein_gap_k, proteins, protein_costs),
                 status_code_t::success_k);
    expect_equal(protein_costs.classes, protein_letters_k.size());
    expect_equal(proteins.codes['W'], proteins.codes['w']);
    expect_equal(proteins.codes['?'], proteins.codes['X']);
    expect_equal(alignment_alphabet_t::fold(own_similarities(usearch_sequence_metric_dna_overlap_k), dna_gap_k,
                                            nucleotides, dna_costs),
                 status_code_t::success_k);
    expect_equal(nucleotides.fold_complement(dna_costs), status_code_t::success_k);
    expect_equal(dna_costs.classes, 4u);
    expect_equal(dna_costs.complemented[nucleotides.codes['A']], nucleotides.codes['T']);
    expect_equal(dna_costs.complemented[nucleotides.codes['g']], nucleotides.codes['C']);
    expect_equal(nucleotides.codes['U'], nucleotides.codes['T']);
    // Costs sending one letter's class to another letter's split the complement.
    alignment_alphabet_t mixed = nucleotides;
    mixed.codes['C'] = mixed.codes['A'];
    alignment_costs_t refused = dna_costs;
    expect_equal(mixed.fold_complement(refused), status_code_t::config_mismatch_k);
}

void run_sequence_cases() {
    u64_t const seed = test_run.settings.seed;
    run_test_case("test_alignment_costs", test_alignment_costs, stream_key(seed, "alignment_costs"));
    run_test_case("test_alignment_own_costs", test_alignment_own_costs);
    run_test_case("test_sequence_stored_sketches", test_sequence_stored_sketches, stream_key(seed, "stored_sketches"));
    run_test_case("test_sequence_roles", test_sequence_roles, stream_key(seed, "roles"));
    run_test_case("test_sequence_sketched_round_trip(usearch_residency_ram_k)", [&] {
        test_sequence_sketched_round_trip(stream_key(seed, "sketched_round_trip"), usearch_residency_ram_k, 0);
    });
    run_test_case("test_basic_scenarios<containment_fixture_t>", test_basic_scenarios<containment_fixture_t>, seed);
    run_test_case("test_basic_scenarios<alignment_fixture_t>", test_basic_scenarios<alignment_fixture_t>, seed);
    run_test_case("test_basic_safegraphs_round_trip<containment_fixture_t>",
                  test_basic_safegraphs_round_trip<containment_fixture_t>, seed, "uuid", "utf8-containment");
    run_test_case("test_basic_safegraphs_round_trip<alignment_fixture_t>",
                  test_basic_safegraphs_round_trip<alignment_fixture_t>, seed, "uuid", "protein-alignment");
    run_test_case("test_basic_safegraphs_hostile<containment_fixture_t>",
                  test_basic_safegraphs_hostile<containment_fixture_t>, seed, "utf8-containment");
    run_test_case("test_containment_kit<sz_cap_serial_k>", test_containment_kit<sz_cap_serial_k>, seed);
    run_test_case("test_containment_kit<sz_cap_any_k>", test_containment_kit<sz_cap_any_k>, seed);
    run_test_case("test_alignment_kit<sz_cap_serial_k>", test_alignment_kit<sz_cap_serial_k>, seed);
    run_test_case("test_alignment_kit<sz_cap_haswell_k>", test_alignment_kit<sz_cap_haswell_k>, seed);
    run_test_case("test_alignment_kit<sz_cap_neon_k>", test_alignment_kit<sz_cap_neon_k>, seed);
    run_test_case("test_alignment_kit<sz_cap_any_k>", test_alignment_kit<sz_cap_any_k>, seed);
    run_test_case("test_smooth_sketch_kit<sz_cap_haswell_k>", test_smooth_sketch_kit<sz_cap_haswell_k>, seed);
    run_test_case("test_smooth_sketch_kit<sz_cap_neon_k>", test_smooth_sketch_kit<sz_cap_neon_k>, seed);
    run_test_case("test_smooth_sketch_kit<sz_cap_any_k>", test_smooth_sketch_kit<sz_cap_any_k>, seed);
    run_test_case("test_hash_sketch_kit<sz_cap_haswell_k>", test_hash_sketch_kit<sz_cap_haswell_k>, seed);
    run_test_case("test_hash_sketch_kit<sz_cap_any_k>", test_hash_sketch_kit<sz_cap_any_k>, seed);
    run_test_case("test_banded_overlap_kit<sz_cap_serial_k>", test_banded_overlap_kit<sz_cap_serial_k>, seed);
    run_test_case("test_banded_overlap_kit<sz_cap_any_k>", test_banded_overlap_kit<sz_cap_any_k>, seed);
    run_basic_nvme_cases("alignment_nvme_fixture_t", alignment_nvme_fixture_t {seed}, seed);
    run_test_case("test_sequence_sharded", test_sequence_sharded, seed);
    run_test_case("test_sequence_sample", test_sequence_sample, seed);
    run_test_case("test_sequence_sharded_cut", test_sequence_sharded_cut, seed);
    run_test_case("test_sequence_sharded_clear_race", test_sequence_sharded_clear_race, seed);
}

} // namespace unum::usearch::test::sequence
