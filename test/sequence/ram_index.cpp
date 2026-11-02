/**
 *  @file test/sequence/ram_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The sequence suite: the basic scenarios over owned strings under byte Levenshtein, rune
 *      Levenshtein, window overlap and Needleman-Wunsch over packed nucleotides, then every
 *      compiled kit against exact oracles.
 *
 *  Strings are eight planted templates with an eighth of their symbols edited, so distances
 *  cluster by template while the bytes differ. The Levenshtein alphabet mixes runes of one to
 *  four bytes, a few sharing a lead byte, so byte and rune distances part ways on every edit that
 *  touches a wide rune. The overlap alphabet is the four nucleotides over short strings, whose
 *  windows recur across templates and keep the distances between clusters graded. The oracles are
 *  the two-row Wagner-Fischer recurrence over bytes or runes, and a brute-force count of the
 *  windows two strings share.
 */
#include <cstring> // `std::memcmp` - the folded nucleotide costs

#include <algorithm>   // `std::min`, `std::max`, `std::find` - a found key among the batch's
#include <array>       // `std::array` - the window widths, the handmade triples, a round's distances
#include <limits>      // `std::numeric_limits` - the oracle's open best
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

/** The overlap fixture's window widths, in bytes. */
constexpr std::array<std::size_t, 2> sequence_window_widths_k {2, 3};

/** Runes one to four bytes wide, @c é and @c è sharing a lead byte, escaped for editors. */
constexpr std::string_view levenshtein_alphabet_k = "abcdefg "                             //
                                                    "\xC3\xA9\xC3\xA8\xD0\xB6"             //
                                                    "\xE8\xAA\x9E\xE6\xA4\x9C\xE2\x82\xAC" //
                                                    "\xF0\x9F\x98\x80\xF0\x9F\x9A\x80";

/** Symbols in a Levenshtein template, which pass the inline capacity of a stored string. */
constexpr std::size_t levenshtein_symbols_k = 24;

/** The nucleotides, whose short windows recur across templates. */
constexpr std::string_view overlap_alphabet_k = "ACGT";

/** Symbols in an overlap template: few, so no window hash collision meets the exact oracle. */
constexpr std::size_t overlap_symbols_k = 12;

using sequence_index_t =
    sequence::ram_index<us::utilities::uuid_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref, counting_allocator_t>;
using sequence_value_t = sequence_index_t::value_t;

/** The symbols of @p text as views of their bytes: every byte under @c sz_levenshtein_bytes_k, and
 *  every rune of valid UTF-8 under @c sz_levenshtein_runes_k, measured by its lead byte alone. */
static std::vector<std::string_view> split_symbols(std::string_view text, sz_levenshtein_symbol_t symbol) {
    std::vector<std::string_view> symbols;
    for (std::size_t position = 0; position < text.size();) {
        u8_t const lead = static_cast<u8_t>(text[position]);
        std::size_t const length = symbol == sz_levenshtein_bytes_k || lead < 0x80 ? 1
                                   : lead < 0xE0                                   ? 2
                                   : lead < 0xF0                                   ? 3
                                                                                   : 4;
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
    std::vector<std::string_view> const alphabet = split_symbols(alphabet_text, sz_levenshtein_runes_k);
    splitmix64_t random_generator {seed};
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

/** Unit-cost edits between @p first and @p second over @p symbol through the two-row Wagner-Fischer
 *  recurrence: the Levenshtein oracle, sharing no code with the metric's kernels or decoder. */
static sz_size_t levenshtein_oracle(std::string_view first, std::string_view second, sz_levenshtein_symbol_t symbol) {
    std::vector<std::string_view> const rows = split_symbols(first, symbol), columns = split_symbols(second, symbol);
    std::vector<sz_size_t> previous_row(columns.size() + 1), current_row(columns.size() + 1);
    for (std::size_t column = 0; column <= columns.size(); ++column)
        previous_row[column] = static_cast<sz_size_t>(column);
    for (std::size_t row = 1; row <= rows.size(); ++row) {
        current_row[0] = static_cast<sz_size_t>(row);
        for (std::size_t column = 1; column <= columns.size(); ++column)
            current_row[column] = std::min({previous_row[column - 1] + (rows[row - 1] != columns[column - 1]),
                                            previous_row[column] + 1, current_row[column - 1] + 1});
        std::swap(previous_row, current_row);
    }
    return previous_row[columns.size()];
}

/** The overlap distance by brute force: a window of @p stored at one width is a hit when @p query
 *  holds its bytes, and the misses over the side with more windows scale to the fixed point. */
static sz_size_t overlap_oracle(std::string_view query, std::string_view stored) noexcept {
    std::size_t query_windows = 0, stored_windows = 0, hits = 0;
    for (std::size_t const width : sequence_window_widths_k) {
        if (width <= query.size()) query_windows += query.size() - width + 1;
        for (std::size_t start = 0; start + width <= stored.size(); ++start, ++stored_windows)
            hits += query.find(stored.substr(start, width)) != std::string_view::npos;
    }
    if (query_windows == 0) return overlap_resolution_k;
    u64_t const longer = std::max<u64_t>(query_windows, stored_windows);
    return static_cast<sz_size_t>((longer - hits) * static_cast<u64_t>(overlap_resolution_k) / longer);
}

/** The Needleman-Wunsch fixture's costs over the nucleotides, packed two bits a code: transitions
 *  cost one, transversions and gaps two. */
constexpr needleman_wunsch_costs_t nucleotide_costs() noexcept {
    needleman_wunsch_costs_t costs {};
    costs.code_bits = 2, costs.classes = 4, costs.gap = 2;
    for (std::size_t first = 0; first != 4; ++first)
        for (std::size_t second = 0; second != 4; ++second)
            costs.substitutions[first][second] = first == second ? 0 : (first & 1) == (second & 1) ? 1 : 2;
    return costs;
}

constexpr needleman_wunsch_costs_t nucleotide_costs_k = nucleotide_costs();

/** The codes of @p packed: its bits read low first from the byte past the first, as many as
 *  fit less the unused slots that first byte counts. */
static std::vector<u32_t> unpacked_codes(std::string_view packed, u32_t code_bits) {
    std::vector<u32_t> codes;
    if (packed.empty()) return codes;
    std::size_t const count = (packed.size() - 1) * 8 / code_bits - static_cast<u8_t>(packed[0]);
    for (std::size_t index = 0; index != count; ++index) {
        u32_t code = 0;
        for (u32_t bit = 0; bit != code_bits; ++bit) {
            std::size_t const position = index * code_bits + bit;
            code |= (static_cast<u8_t>(packed[1 + position / 8]) >> (position % 8) & 1u) << bit;
        }
        codes.push_back(code);
    }
    return codes;
}

/** @p codes packed at @p code_bits apiece behind the count of unused slots, as the metric reads
 *  them. */
static std::string packed_sequence(std::span<u32_t const> codes, u32_t code_bits) {
    std::size_t const data_bytes = (codes.size() * code_bits + 7) / 8;
    std::string packed(1 + data_bytes, '\0');
    packed[0] = static_cast<char>(data_bytes * 8 / code_bits - codes.size());
    for (std::size_t index = 0; index != codes.size(); ++index)
        for (u32_t bit = 0; bit != code_bits; ++bit) {
            std::size_t const position = index * code_bits + bit;
            packed[1 + position / 8] = static_cast<char>(static_cast<u8_t>(packed[1 + position / 8]) |
                                                         ((codes[index] >> bit & 1u) << (position % 8)));
        }
    return packed;
}

/** The Needleman-Wunsch cost between two packed sequences under @p costs through the two-row
 *  recurrence: the oracle, sharing no code with the metric's kernels or unpacking. */
static sz_size_t needleman_wunsch_oracle(std::string_view first, std::string_view second,
                                         needleman_wunsch_costs_t const& costs) {
    std::vector<u32_t> const rows = unpacked_codes(first, costs.code_bits);
    std::vector<u32_t> const columns = unpacked_codes(second, costs.code_bits);
    std::vector<sz_size_t> previous_row(columns.size() + 1), current_row(columns.size() + 1);
    for (std::size_t column = 0; column <= columns.size(); ++column) previous_row[column] = column * costs.gap;
    for (std::size_t row = 1; row <= rows.size(); ++row) {
        current_row[0] = row * costs.gap;
        for (std::size_t column = 1; column <= columns.size(); ++column)
            current_row[column] = std::min(
                {previous_row[column - 1] + costs.substitutions[rows[row - 1]][columns[column - 1]],
                 previous_row[column] + costs.gap, current_row[column - 1] + costs.gap});
        std::swap(previous_row, current_row);
    }
    return previous_row[columns.size()];
}

/** The nucleotides of @p text packed two bits a code, A, C, G and T in that order. */
static std::string packed_nucleotides(std::string_view text) {
    std::vector<u32_t> codes;
    for (char const nucleotide : text) codes.push_back(static_cast<u32_t>(overlap_alphabet_k.find(nucleotide)));
    return packed_sequence(codes, 2);
}

/** Three stored strings and their pairwise distances, stated by hand. */
struct sequence_handmade_t {
    std::array<sequence_value_t, 3> values;
    std::array<std::array<sz_size_t, 3>, 3> distances;
};

template <sz_levenshtein_symbol_t symbol_>
struct levenshtein_fixture {
    using index_t = sequence_index_t;
    using value_t = sequence_value_t;
    using allocator_t = typename index_t::allocator_t;
    using metric_t = levenshtein_metric<typename levenshtein_ladder<symbol_, sz_cap_any_k>::kit_t>;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static metric_t make_metric() noexcept {
        metric_t metric;
        [[maybe_unused]] status_t const sized = metric.init(sequence_staging_limit_k);
        return metric;
    }
    static codec_t make_codec() noexcept { return {}; }
    static us::utilities::uuid_t make_key(std::size_t ordinal) noexcept { return uuid_of(0x5E0E0CE0ull + ordinal); }
    static sz_size_t oracle(std::string_view query, std::string_view stored) noexcept {
        return levenshtein_oracle(query, stored, symbol_);
    }
    static bool value_equals(std::string_view first, std::string_view second) noexcept { return first == second; }

    /** "naïve", "naive" and "nave": the diaeresis is one rune of two bytes. */
    static sequence_handmade_t make_handmade(allocator_t const& allocator) {
        constexpr std::array<std::array<sz_size_t, 3>, 3> bytes_k {{{0, 2, 2}, {2, 0, 1}, {2, 1, 0}}};
        constexpr std::array<std::array<sz_size_t, 3>, 3> runes_k {{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}};
        return {{make_value("na\xC3\xAFve", allocator), make_value("naive", allocator), make_value("nave", allocator)},
                symbol_ == sz_levenshtein_bytes_k ? bytes_k : runes_k};
    }
    static std::vector<value_t> make_values(std::size_t count, u64_t seed, allocator_t const& allocator) {
        return generate_values(count, seed, levenshtein_alphabet_k, levenshtein_symbols_k, allocator);
    }
};

using levenshtein_bytes_fixture_t = levenshtein_fixture<sz_levenshtein_bytes_k>;
using levenshtein_runes_fixture_t = levenshtein_fixture<sz_levenshtein_runes_k>;

struct overlap_fixture_t {
    using index_t = sequence_index_t;
    using value_t = sequence_value_t;
    using allocator_t = index_t::allocator_t;
    using metric_t = overlap_metric_t;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static metric_t make_metric() noexcept {
        metric_t metric;
        [[maybe_unused]] status_t const sized = metric.init(sequence_staging_limit_k, sequence_window_widths_k);
        return metric;
    }
    static codec_t make_codec() noexcept { return {}; }
    static us::utilities::uuid_t make_key(std::size_t ordinal) noexcept { return uuid_of(0x0E7E1A90ull + ordinal); }
    static sz_size_t oracle(std::string_view query, std::string_view stored) noexcept {
        return overlap_oracle(query, stored);
    }
    static bool value_equals(std::string_view first, std::string_view second) noexcept { return first == second; }

    /** "abcd" and "abce" share three of their five windows, "xyzw" none of either's. */
    static sequence_handmade_t make_handmade(allocator_t const& allocator) {
        constexpr sz_size_t shared_k = overlap_resolution_k * 2 / 5;
        constexpr sz_size_t disjoint_k = overlap_resolution_k;
        return {{make_value("abcd", allocator), make_value("abce", allocator), make_value("xyzw", allocator)},
                {{{0, shared_k, disjoint_k}, {shared_k, 0, disjoint_k}, {disjoint_k, disjoint_k, 0}}}};
    }
    static std::vector<value_t> make_values(std::size_t count, u64_t seed, allocator_t const& allocator) {
        return generate_values(count, seed, overlap_alphabet_k, overlap_symbols_k, allocator);
    }
};

struct needleman_wunsch_fixture_t {
    using index_t = sequence_index_t;
    using value_t = sequence_value_t;
    using allocator_t = index_t::allocator_t;
    using metric_t = needleman_wunsch_metric_t;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static metric_t make_metric() noexcept {
        metric_t metric;
        [[maybe_unused]] status_t const sized = metric.init(sequence_staging_limit_k, nucleotide_costs_k);
        return metric;
    }
    static codec_t make_codec() noexcept { return {}; }
    static us::utilities::uuid_t make_key(std::size_t ordinal) noexcept { return uuid_of(0x0A11C0DEull + ordinal); }
    static sz_size_t oracle(std::string_view query, std::string_view stored) {
        return needleman_wunsch_oracle(query, stored, nucleotide_costs_k);
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
        std::vector<value_t> values = generate_values(count, seed, overlap_alphabet_k, overlap_symbols_k, allocator);
        for (value_t& value : values) value = make_value(packed_nucleotides(std::string_view {value}), allocator);
        return values;
    }
};

static_assert(value<sequence_value_t>);
static_assert(metric<levenshtein_bytes_fixture_t::metric_t, sequence_value_t, sz_size_t>);
static_assert(metric<levenshtein_runes_fixture_t::metric_t, sequence_value_t, sz_size_t>);
static_assert(metric<overlap_fixture_t::metric_t, sequence_value_t, sz_size_t>);
static_assert(metric<needleman_wunsch_fixture_t::metric_t, sequence_value_t, sz_size_t>);
static_assert(query<std::string_view, levenshtein_runes_fixture_t::metric_t, sequence_value_t>);
static_assert(query<std::string_view, overlap_fixture_t::metric_t, sequence_value_t>);
static_assert(value_codec<codec_t, sequence_value_t>);

/** One Levenshtein kit against the Wagner-Fischer oracle: exact within the radius and above it
 *  otherwise, at an unbounded radius and at a tight one that cuts by length and freezes flights.
 *  The joined strings pass two query words, so the scratch verticals score beside the resident. */
template <sz_capability_t capability_, sz_levenshtein_symbol_t symbol_>
std::size_t check_levenshtein_kit(u64_t seed) {
    constexpr std::size_t generated_k = 36, parts_per_join_k = 6, joined_k = generated_k / parts_per_join_k;
    constexpr std::size_t candidates_k = generated_k + joined_k;
    constexpr sz_size_t tight_radius_k = 3;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        counting_allocator_t const allocator {ledger};
        std::vector<sequence_value_t> candidates = generate_values(
            generated_k, stream_key(seed, "kits"), levenshtein_alphabet_k, levenshtein_symbols_k, allocator);
        std::string joined;
        for (std::size_t join = 0; join != joined_k; ++join) {
            joined.clear();
            for (std::size_t part = 0; part != parts_per_join_k; ++part)
                joined += candidates[join * parts_per_join_k + part];
            candidates.push_back(make_value(joined, allocator));
        }

        levenshtein_metric<typename levenshtein_ladder<symbol_, capability_>::kit_t> metric;
        failures += expect_eq(metric.init(candidates_k), status_t::success_k);
        for (sz_size_t const radius : {unbounded_radius<sz_size_t>(), tight_radius_k}) {
            for (std::size_t const query : {std::size_t {0}, generated_k}) {
                metric.reset(candidates[query]);
                for (sequence_value_t const& candidate : candidates)
                    failures += expect(metric.push(candidate) == metric_staging_t::staged_k);
                std::array<sz_size_t, candidates_k> distances;
                metric.drain(distances, radius);
                failures += expect_eq(metric.status(), status_t::success_k);
                for (std::size_t candidate = 0; candidate != candidates_k; ++candidate) {
                    sz_size_t const truth = levenshtein_oracle(candidates[query], candidates[candidate], symbol_);
                    failures += truth <= radius ? expect_eq(distances[candidate], truth)
                                                : expect(distances[candidate] > radius);
                }
            }
        }
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** One overlap kit against the brute-force window count: exact within the radius and above it
 *  otherwise, at an unbounded radius and at half the full scale, past which each pair drops by its
 *  own budget. Generated strings, a string shorter than every width, and the empty one, opening a
 *  round that shares no window. */
template <sz_capability_t capability_>
std::size_t check_overlap_kit(u64_t seed) {
    constexpr std::size_t generated_k = 24, candidates_k = generated_k + 2;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        counting_allocator_t const allocator {ledger};
        std::vector<sequence_value_t> candidates = generate_values(generated_k, stream_key(seed, "overlap"),
                                                                   overlap_alphabet_k, overlap_symbols_k, allocator);
        candidates.push_back(make_value("A", allocator));
        candidates.push_back(make_value("", allocator));

        overlap_metric<typename overlap_ladder<capability_>::kit_t> metric;
        failures += expect_eq(metric.init(candidates_k, sequence_window_widths_k), status_t::success_k);
        for (sz_size_t const radius : {unbounded_radius<sz_size_t>(), overlap_resolution_k / 2}) {
            // A batch scores every query as that query alone would, and the empty query
            // reopened right after it is prepared anew rather than taken for the batch.
            std::array<std::string_view, 2> const batch {candidates[0], candidates[candidates_k - 1]};
            metric.reset(std::span<std::string_view const>(batch));
            for (sequence_value_t const& candidate : candidates)
                failures += expect(metric.push(candidate) == metric_staging_t::staged_k);
            std::array<sz_size_t, candidates_k * 2> batched;
            metric.drain(batched, radius);
            failures += expect_eq(metric.status(), status_t::success_k);
            for (std::size_t slot = 0; slot != batched.size(); ++slot) {
                sz_size_t const truth = overlap_oracle(batch[slot / candidates_k], candidates[slot % candidates_k]);
                failures += truth <= radius ? expect_eq(batched[slot], truth) : expect(batched[slot] > radius);
            }
            for (std::size_t const query : {candidates_k - 1, std::size_t {0}}) {
                metric.reset(candidates[query]);
                for (sequence_value_t const& candidate : candidates)
                    failures += expect(metric.push(candidate) == metric_staging_t::staged_k);
                std::array<sz_size_t, candidates_k> distances;
                metric.drain(distances, radius);
                failures += expect_eq(metric.status(), status_t::success_k);
                for (std::size_t candidate = 0; candidate != candidates_k; ++candidate) {
                    sz_size_t const truth = overlap_oracle(candidates[query], candidates[candidate]);
                    failures += truth <= radius ? expect_eq(distances[candidate], truth)
                                                : expect(distances[candidate] > radius);
                }
            }
        }
    }
    failures += expect_eq(ledger.live_bytes, 0);
    return failures;
}

/** One Needleman-Wunsch kit against the two-row oracle over five-bit codes. The fold first: the
 *  nucleotides' scores into the fixture's costs, a table whose gap passes 15 scaled into it, and a
 *  pair scoring above its mean self-score refused. Then every length from empty to one whose cost
 *  from a short query passes a 16-bit cell, exact within the radius and above it otherwise, at an
 *  unbounded radius and at a tight one that stops columns early. */
template <sz_capability_t capability_>
std::size_t check_needleman_wunsch_kit(u64_t seed) {
    constexpr std::size_t side_k = needleman_wunsch_byte_alphabet_k, classes_k = 24, generated_k = 40;
    constexpr std::size_t longest_k = 4500;
    constexpr sz_size_t tight_radius_k = 40;
    std::size_t failures = 0;

    std::vector<i8_t> nucleotides(side_k * side_k);
    for (std::size_t first = 0; first != 4; ++first)
        for (std::size_t second = 0; second != 4; ++second)
            nucleotides[first * side_k + second] = first == second ? 2 : (first & 1) == (second & 1) ? 1 : 0;
    auto const [nucleotide_folded, nucleotide_status] = needleman_wunsch_costs_t::fold(nucleotides, 2);
    failures += expect_eq(nucleotide_status, status_t::success_k);
    failures += expect(std::memcmp(&nucleotide_folded, &nucleotide_costs_k, sizeof(nucleotide_costs_k)) == 0);

    // Self-scores of 8 to 11 and cross-scores of -3 to 3 keep every cost at 14 or below, so the gap
    // of 20 alone scales the table by 3/4.
    std::vector<i8_t> similarities(side_k * side_k);
    for (std::size_t first = 0; first != classes_k; ++first)
        for (std::size_t second = 0; second != classes_k; ++second)
            similarities[first * side_k + second] = static_cast<i8_t>(
                first == second ? 8 + first % 4 : static_cast<int>(first * second % 7) - 3);
    auto const [costs, fold_status] = needleman_wunsch_costs_t::fold(similarities, 20);
    failures += expect_eq(fold_status, status_t::success_k);
    failures += expect_eq(costs.code_bits, 5u);
    failures += expect_eq(costs.classes, classes_k);
    failures += expect_eq(costs.gap, 15u);
    // Codes 0 and 1 cost (8 + 9) / 2 + 3 = 11.5, scaled to 8.625 and rounded to 9.
    failures += expect_eq(costs.substitutions[0][1], 9u);
    failures += expect_eq(costs.substitutions[5][5], 0u);
    similarities[1] = 100;
    failures += expect_eq(needleman_wunsch_costs_t::fold(similarities, 20).status, status_t::config_mismatch_k);

    splitmix64_t random_generator {stream_key(seed, "needleman_wunsch")};
    uniform_below<u32_t> const code_distribution {classes_k}, length_distribution {60};
    std::vector<std::string> candidates;
    for (std::size_t candidate = 0; candidate != generated_k; ++candidate) {
        std::size_t const length = candidate == 0                 ? 0
                                   : candidate == 1               ? 1
                                   : candidate + 1 == generated_k ? longest_k
                                                                  : length_distribution(random_generator) + 1;
        std::vector<u32_t> codes(length);
        for (u32_t& code : codes) code = code_distribution(random_generator);
        candidates.push_back(packed_sequence(codes, costs.code_bits));
    }

    needleman_wunsch_metric<typename needleman_wunsch_ladder<capability_>::kit_t> metric;
    failures += expect_eq(metric.init(generated_k, costs), status_t::success_k);
    for (sz_size_t const radius : {unbounded_radius<sz_size_t>(), tight_radius_k}) {
        for (std::size_t const query : {std::size_t {2}, std::size_t {0}, generated_k - 1}) {
            metric.reset(std::string_view {candidates[query]});
            for (std::string const& candidate : candidates)
                failures += expect(metric.push(std::string_view {candidate}) == metric_staging_t::staged_k);
            std::array<sz_size_t, generated_k> distances;
            metric.drain(distances, radius);
            failures += expect_eq(metric.status(), status_t::success_k);
            for (std::size_t candidate = 0; candidate != generated_k; ++candidate) {
                sz_size_t const truth = needleman_wunsch_oracle(candidates[query], candidates[candidate], costs);
                failures += truth <= radius ? expect_eq(distances[candidate], truth)
                                            : expect(distances[candidate] > radius);
            }
        }
    }
    return failures;
}

/** The byte Levenshtein fixture over 64-bit node IDs, which a saved neighbors column records. */
struct levenshtein_wide_fixture_t : levenshtein_bytes_fixture_t {
    using index_t = sequence::ram_index<us::utilities::uuid_t, u64_t, flaky_waiter_t, fu::standard_atomic_ref,
                                        counting_allocator_t>;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
};

/** Byte Levenshtein over storage: generated strings, the empty one, and one crossing a 4 KiB block,
 *  ragged entries the view locates through its directory, viewed at 32-bit and 64-bit node IDs. */
struct levenshtein_nvme_fixture_t {
    using ram_fixture_t = levenshtein_bytes_fixture_t;
    using wide_fixture_t = levenshtein_wide_fixture_t;
    using shard_index_t = sequence::ram_index<us::utilities::uuid_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref,
                                              forkunion_allocator_t>;
    template <typename engine_type_, typename node_id_type_ = u32_t>
    using nvme_t = us::basic::nvme_indexes<codec_t, us::utilities::uuid_t, node_id_type_, sz_size_t, engine_type_>;
    using scratch_metric_t =
        levenshtein_metric<levenshtein_ladder<sz_levenshtein_bytes_k, sz_cap_any_k>::kit_t, basic_nvme_scratch_t>;
    using query_t = std::string_view;
    using coordinates_t = no_fixed_coordinates_t;
    static constexpr std::string_view metric_token_k = "levenshtein";

    std::vector<std::string> texts;

    explicit levenshtein_nvme_fixture_t(u64_t seed) {
        allocation_ledger_t ledger;
        for (sequence_value_t const& value : generate_values(30, stream_key(seed, "nvme"), levenshtein_alphabet_k,
                                                             levenshtein_symbols_k, counting_allocator_t {ledger}))
            texts.emplace_back(std::string_view {value});
        texts.emplace_back();
        std::string& crossing = texts.emplace_back(5000, 'a');
        crossing[4096] = 'b';
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
        status_t const ready = metric.init(staging);
        return {std::move(metric), ready};
    }
    expected<query_t> query_of(std::size_t entry, basic_nvme_scratch_t const&) const noexcept {
        return {texts[entry], status_t::success_k};
    }
    static expected<query_t> stored_query(std::string_view stored, basic_nvme_scratch_t const&) noexcept {
        return {stored, status_t::success_k};
    }
    bool matches(std::size_t entry, std::string_view view) const noexcept { return view == texts[entry]; }
};

/** 5000 strings over sharded indexes, as @c test_basic_sharded checks, against the Wagner-Fischer
 *  oracle over every string. */
std::size_t check_sequence_sharded(u64_t seed) {
    using index_t = sequence::ram_index<us::utilities::uuid_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref,
                                        forkunion_allocator_t>;
    using value_t = index_t::value_t;
    constexpr std::size_t entries_k = 5000;
    std::vector<std::string> texts;
    {
        allocation_ledger_t ledger;
        for (sequence_value_t const& value :
             generate_values(entries_k + basic_recall_samples_k, stream_key(seed, "plural"), levenshtein_alphabet_k,
                             levenshtein_symbols_k, counting_allocator_t {ledger}))
            texts.emplace_back(std::string_view {value});
    }
    forkunion_allocator_t const allocator {no_memory_domain_k};
    auto make_values = [&] {
        std::vector<value_t> values;
        values.reserve(entries_k);
        for (std::size_t entry = 0; entry != entries_k; ++entry) [[maybe_unused]]
            sz::status_t const assigned = values.emplace_back(allocator).assign(texts[entry]);
        return values;
    };
    std::vector<us::utilities::uuid_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(0x91A7A100ull + entry);
    auto query_of = [&](std::size_t sample) noexcept { return std::string_view {texts[entries_k + sample]}; };
    auto best_of = [&](std::size_t sample) {
        sz_size_t best = std::numeric_limits<sz_size_t>::max();
        for (std::size_t entry = 0; entry != entries_k; ++entry)
            best = std::min(best, levenshtein_oracle(query_of(sample), texts[entry], sz_levenshtein_bytes_k));
        return best;
    };
    levenshtein_bytes_fixture_t::metric_t metric = levenshtein_bytes_fixture_t::make_metric();
    return test_basic_sharded<us::basic::ram_indexes<index_t>>(
        seed, metric, std::span<us::utilities::uuid_t const>(keys), make_values, query_of, best_of);
}

/** Generated strings sampled from two shards, as @c test_basic_sharded_sample checks. */
std::size_t check_sequence_sample(u64_t seed) {
    using index_t = sequence::ram_index<us::utilities::uuid_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref,
                                        forkunion_allocator_t>;
    using value_t = index_t::value_t;
    constexpr std::size_t entries_k = basic_recall_samples_k * basic_connectivity_k;
    std::vector<std::string> texts;
    {
        allocation_ledger_t ledger;
        for (sequence_value_t const& value :
             generate_values(entries_k, stream_key(seed, "plural_sample"), levenshtein_alphabet_k,
                             levenshtein_symbols_k, counting_allocator_t {ledger}))
            texts.emplace_back(std::string_view {value});
    }
    forkunion_allocator_t const allocator {no_memory_domain_k};
    auto make_values = [&] {
        std::vector<value_t> values;
        for (std::string const& text : texts) [[maybe_unused]]
            sz::status_t const assigned = values.emplace_back(allocator).assign(text);
        return values;
    };
    std::vector<us::utilities::uuid_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(0x5A3B1E00ull + entry);
    levenshtein_bytes_fixture_t::metric_t metric = levenshtein_bytes_fixture_t::make_metric();
    return test_basic_sharded_sample<us::basic::ram_indexes<index_t>>(
        seed, metric, std::span<us::utilities::uuid_t const>(keys), make_values, no_fixed_coordinates_t {});
}

/** 600 strings in a sharded index of two shards split into two sharded indexes of two, as
 *  @c test_basic_sharded_split checks. */
std::size_t check_sequence_sharded_split(u64_t seed) {
    using index_t = sequence::ram_index<us::utilities::uuid_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref,
                                        forkunion_allocator_t>;
    using value_t = index_t::value_t;
    constexpr std::size_t entries_k = 600;
    std::vector<std::string> texts;
    {
        allocation_ledger_t ledger;
        for (sequence_value_t const& value :
             generate_values(entries_k, stream_key(seed, "plural_split"), levenshtein_alphabet_k, levenshtein_symbols_k,
                             counting_allocator_t {ledger}))
            texts.emplace_back(std::string_view {value});
    }
    forkunion_allocator_t const allocator {no_memory_domain_k};
    std::vector<us::utilities::uuid_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(0x5917A100ull + entry);
    auto make_values = [&] {
        std::vector<value_t> values;
        values.reserve(entries_k);
        for (std::size_t entry = 0; entry != entries_k; ++entry) [[maybe_unused]]
            sz::status_t const assigned = values.emplace_back(allocator).assign(texts[entry]);
        return values;
    };
    levenshtein_bytes_fixture_t::metric_t metric = levenshtein_bytes_fixture_t::make_metric();
    return test_basic_sharded_split<us::basic::ram_indexes<index_t>>(
        seed, metric, std::span<us::utilities::uuid_t const>(keys), make_values);
}

/** Searches racing a clear of one shard read only retired entries nothing frees: each answers
 *  entries of the batch or @c contended_k, never a foreign key, while the clear retires the shard
 *  and a fresh one takes the next batch, which the only shard then names. A cursor opened before a
 *  clear answers @c contended_k after it. Under a sanitizer, a value freed under a search would
 *  trap. */
std::size_t check_sequence_sharded_clear_race(u64_t seed) {
    using index_t = sequence::ram_index<us::utilities::uuid_t, u32_t, flaky_waiter_t, fu::standard_atomic_ref,
                                        forkunion_allocator_t>;
    using indexes_t = us::basic::ram_indexes<index_t>;
    using value_t = index_t::value_t;
    using distance_t = index_t::distance_t;
    using metric_t = levenshtein_bytes_fixture_t::metric_t;
    constexpr std::size_t entries_k = 400, searchers_k = 4, rounds_k = 4, wanted_k = 8;
    std::size_t failures = 0;
    std::vector<std::string> texts;
    {
        allocation_ledger_t ledger;
        for (sequence_value_t const& value :
             generate_values(entries_k, stream_key(seed, "plural_clear_race"), levenshtein_alphabet_k,
                             levenshtein_symbols_k, counting_allocator_t {ledger}))
            texts.emplace_back(std::string_view {value});
    }
    forkunion_allocator_t const allocator {no_memory_domain_k};
    std::vector<us::utilities::uuid_t> keys(entries_k);
    for (std::size_t entry = 0; entry != entries_k; ++entry) keys[entry] = uuid_of(0xC1EA0000ull + entry);
    keep_diverse_t const selector {basic_connectivity_k};
    std::vector<index_t::traversal_context_t> contexts;
    std::vector<metric_t> metrics;
    contexts.reserve(searchers_k + 1);
    metrics.reserve(searchers_k + 1);
    for (std::size_t thread = 0; thread != searchers_k + 1; ++thread) {
        contexts.emplace_back(stream_key(seed, "plural_clear_race") + thread, allocator);
        failures += expect_eq(index_t::reserve_traversal(contexts.back(), basic_connectivity_k, {}),
                              status_t::success_k);
        metrics.push_back(levenshtein_bytes_fixture_t::make_metric());
    }

    indexes_t indexes {allocator};
    failures += expect_eq(indexes.emplace(no_memory_domain_k), status_t::success_k);
    auto insert_batch = [&] {
        std::vector<value_t> values;
        values.reserve(entries_k);
        for (std::size_t entry = 0; entry != entries_k; ++entry) [[maybe_unused]]
            sz::status_t const assigned = values.emplace_back(allocator).assign(texts[entry]);
        auto [plan, plan_status] = indexes.plan_insert(entries_k);
        failures += expect_eq(plan_status, status_t::success_k);
        if (failed(plan_status)) return;
        plan.insert(0, std::span<value_t>(values), std::span<us::utilities::uuid_t const>(keys), metrics[0],
                    contexts[0], selector);
        failures += expect_eq(plan.finish(), status_t::success_k);
    };

    fu::flat_pool_t pool;
    if (fu::failed(pool.spawn(searchers_k + 1))) return failures + expect(false);
    std::size_t foreign_count = 0;
    fu::standard_atomic_ref<std::size_t> const foreign {foreign_count};
    for (std::size_t round = 0; round != rounds_k; ++round) {
        insert_batch();
        failures += expect_eq(indexes.size(), entries_k);
        indexes_t::cursor_t cursor {forkunion_allocator_t {no_memory_domain_k}};
        failures += expect(succeeded(cursor.open(indexes, {})));
        bool cleared = false;
        std::size_t searched = 0;
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                // Every searcher walks the entries once first, so the clear runs under live walks.
                while (fu::standard_atomic_ref<std::size_t>(searched).load(std::memory_order_acquire) < searchers_k)
                    std::this_thread::yield();
                while (indexes.clear() == status_t::contended_k) std::this_thread::yield();
                fu::standard_atomic_ref<bool>(cleared).store(true, std::memory_order_release);
                return;
            }
            std::vector<u32_t> watermarks;
            std::array<us::utilities::uuid_t, wanted_k> found_keys;
            std::array<distance_t, wanted_k> found_distances;
            dummy_predicate_t unfiltered;
            for (std::size_t entry = thread; !fu::standard_atomic_ref<bool>(cleared).load(std::memory_order_acquire);
                 entry = (entry + searchers_k) % entries_k) {
                watermarks.resize(indexes.shards_count());
                if (indexes.snapshot(watermarks).status == status_t::shape_mismatch_k) continue;
                auto const [found, status] = indexes.search_if(std::string_view {texts[entry]}, watermarks, found_keys,
                                                               found_distances, {}, unfiltered, metrics[thread],
                                                               contexts[thread]);
                if (entry == thread) fu::standard_atomic_ref<std::size_t>(searched).add(1, std::memory_order_release);
                if (status == status_t::contended_k) continue;
                if (failed(status)) foreign.add(1);
                else
                    for (std::size_t match = 0; match != found; ++match)
                        foreign.add(std::find(keys.begin(), keys.end(), found_keys[match]) == keys.end());
            }
        });
        failures += expect_eq(indexes.size(), std::size_t {0});
        failures += expect_eq(indexes.count(keys[0]), std::size_t {0});
        failures += expect_eq(indexes.shards_count(), round + 2);
        auto const live = indexes.live_shards();
        failures += expect_eq(live.bound - live.first, std::size_t {1});
        std::array<us::utilities::uuid_t, 4> read_keys;
        failures += expect_eq(
            cursor.pull(indexes, std::span<us::utilities::uuid_t> {read_keys}, nullptr, nullptr).status,
            status_t::contended_k);
    }
    failures += expect_eq(foreign_count, std::size_t {0});

    // The fresh shard takes a batch whole, and a clear of an empty index retires nothing.
    insert_batch();
    failures += expect_eq(indexes.size(), entries_k);
    failures += expect_eq(indexes.count(keys[entries_k - 1]), std::size_t {1});
    failures += expect_eq(indexes.clear(), status_t::success_k);
    failures += expect_eq(indexes.clear(), status_t::success_k);
    failures += expect_eq(indexes.shards_count(), rounds_k + 2);
    return failures;
}

std::size_t check_sequence(settings_t const& settings) {
    std::size_t failures = 0;
    failures += test_basic_scenarios<levenshtein_bytes_fixture_t>(settings.seed);
    failures += test_basic_scenarios<levenshtein_runes_fixture_t>(settings.seed);
    failures += test_basic_scenarios<overlap_fixture_t>(settings.seed);
    failures += test_basic_scenarios<needleman_wunsch_fixture_t>(settings.seed);
    failures += test_basic_safegraphs_roundtrip<levenshtein_bytes_fixture_t>(settings.seed, "uuid", "levenshtein");
    failures += test_basic_safegraphs_roundtrip<levenshtein_runes_fixture_t>(settings.seed, "uuid", "levenshtein-utf8");
    failures += test_basic_safegraphs_roundtrip<overlap_fixture_t>(settings.seed, "uuid", "overlap");
    failures += test_basic_safegraphs_hostile<levenshtein_bytes_fixture_t>(settings.seed, "levenshtein");
    failures += test_basic_safegraphs_hostile<levenshtein_runes_fixture_t>(settings.seed, "levenshtein-utf8");
    failures += test_basic_safegraphs_hostile<overlap_fixture_t>(settings.seed, "overlap");
    failures += check_levenshtein_kit<sz_cap_serial_k, sz_levenshtein_bytes_k>(settings.seed);
    failures += check_levenshtein_kit<sz_cap_any_k, sz_levenshtein_bytes_k>(settings.seed);
    failures += check_levenshtein_kit<sz_cap_serial_k, sz_levenshtein_runes_k>(settings.seed);
    failures += check_levenshtein_kit<sz_cap_any_k, sz_levenshtein_runes_k>(settings.seed);
    failures += check_overlap_kit<sz_cap_serial_k>(settings.seed);
    failures += check_overlap_kit<sz_cap_any_k>(settings.seed);
    failures += check_needleman_wunsch_kit<sz_cap_serial_k>(settings.seed);
    failures += check_needleman_wunsch_kit<sz_cap_haswell_k>(settings.seed);
    failures += check_needleman_wunsch_kit<sz_cap_neon_k>(settings.seed);
    failures += check_needleman_wunsch_kit<sz_cap_any_k>(settings.seed);
    failures += test_basic_nvme_engines(levenshtein_nvme_fixture_t {settings.seed}, "sequence", settings.seed);
    failures += check_sequence_sharded(settings.seed);
    failures += check_sequence_sample(settings.seed);
    failures += check_sequence_sharded_split(settings.seed);
    failures += check_sequence_sharded_clear_race(settings.seed);
    return report("sequence", failures);
}

} // namespace unum::usearch::test::sequence
