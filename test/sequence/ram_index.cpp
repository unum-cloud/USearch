/**
 *  @file test/sequence/ram_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The sequence suite: the basic scenarios over owned strings under byte Levenshtein, rune
 *      Levenshtein and window overlap, then every compiled kit against exact oracles.
 *
 *  Strings are eight planted templates with an eighth of their symbols edited, so distances
 *  cluster by template while the bytes differ. The Levenshtein alphabet mixes runes of one to
 *  four bytes, a few sharing a lead byte, so byte and rune distances part ways on every edit that
 *  touches a wide rune. The overlap alphabet is the four nucleotides over short strings, whose
 *  windows recur across templates and keep the distances between clusters graded. The oracles are
 *  the two-row Wagner-Fischer recurrence over bytes or runes, and a brute-force count of the
 *  windows two strings share.
 */
#include <algorithm>   // `std::min`, `std::max`
#include <array>       // `std::array` - the window widths, the handmade triples, a round's distances
#include <memory>      // `std::unique_ptr`
#include <new>         // `std::nothrow`
#include <string>      // `std::string` - a generated string before it is stored
#include <string_view> // `std::string_view` - the symbols and the oracles' operands
#include <vector>      // `std::vector` - generated values, templates, the oracle's rows

#include "usearch/sequence/ram_index.hpp"

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
    sequence::ram_index<us::utilities::uuid_t, u32_t, flaky_waiter_t, standard_atomic_ref, counting_allocator_t>;
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
static i64_t levenshtein_oracle(std::string_view first, std::string_view second, sz_levenshtein_symbol_t symbol) {
    std::vector<std::string_view> const rows = split_symbols(first, symbol), columns = split_symbols(second, symbol);
    std::vector<i64_t> previous_row(columns.size() + 1), current_row(columns.size() + 1);
    for (std::size_t column = 0; column <= columns.size(); ++column) previous_row[column] = static_cast<i64_t>(column);
    for (std::size_t row = 1; row <= rows.size(); ++row) {
        current_row[0] = static_cast<i64_t>(row);
        for (std::size_t column = 1; column <= columns.size(); ++column)
            current_row[column] = std::min({previous_row[column - 1] + (rows[row - 1] != columns[column - 1]),
                                            previous_row[column] + 1, current_row[column - 1] + 1});
        std::swap(previous_row, current_row);
    }
    return previous_row[columns.size()];
}

/** The overlap distance by brute force: a window of @p stored at one width is a hit when @p query
 *  holds its bytes, and the misses over the side with more windows scale to the fixed point. */
static i64_t overlap_oracle(std::string_view query, std::string_view stored) noexcept {
    std::size_t query_windows = 0, stored_windows = 0, hits = 0;
    for (std::size_t const width : sequence_window_widths_k) {
        if (width <= query.size()) query_windows += query.size() - width + 1;
        for (std::size_t start = 0; start + width <= stored.size(); ++start, ++stored_windows)
            hits += query.find(stored.substr(start, width)) != std::string_view::npos;
    }
    if (query_windows == 0) return overlap_metric_t::resolution_k;
    u64_t const longer = std::max<u64_t>(query_windows, stored_windows);
    return static_cast<i64_t>((longer - hits) * static_cast<u64_t>(overlap_metric_t::resolution_k) / longer);
}

/** Three stored strings and their pairwise distances, stated by hand. */
struct sequence_handmade_t {
    std::array<sequence_value_t, 3> values;
    std::array<std::array<i64_t, 3>, 3> distances;
};

template <sz_levenshtein_symbol_t symbol_>
struct levenshtein_fixture {
    using index_t = sequence_index_t;
    using value_t = sequence_value_t;
    using allocator_t = typename index_t::allocator_t;
    using metric_t = levenshtein_metric<sz_cap_any_k, symbol_>;

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
    static i64_t oracle(std::string_view query, std::string_view stored) noexcept {
        return levenshtein_oracle(query, stored, symbol_);
    }
    static bool value_equals(std::string_view first, std::string_view second) noexcept { return first == second; }

    /** "naïve", "naive" and "nave": the diaeresis is one rune of two bytes. */
    static sequence_handmade_t make_handmade(allocator_t const& allocator) {
        constexpr std::array<std::array<i64_t, 3>, 3> bytes_k {{{0, 2, 2}, {2, 0, 1}, {2, 1, 0}}};
        constexpr std::array<std::array<i64_t, 3>, 3> runes_k {{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}};
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
    static i64_t oracle(std::string_view query, std::string_view stored) noexcept {
        return overlap_oracle(query, stored);
    }
    static bool value_equals(std::string_view first, std::string_view second) noexcept { return first == second; }

    /** "abcd" and "abce" share three of their five windows, "xyzw" none of either's. */
    static sequence_handmade_t make_handmade(allocator_t const& allocator) {
        constexpr i64_t shared_k = overlap_metric_t::resolution_k * 2 / 5;
        constexpr i64_t disjoint_k = overlap_metric_t::resolution_k;
        return {{make_value("abcd", allocator), make_value("abce", allocator), make_value("xyzw", allocator)},
                {{{0, shared_k, disjoint_k}, {shared_k, 0, disjoint_k}, {disjoint_k, disjoint_k, 0}}}};
    }
    static std::vector<value_t> make_values(std::size_t count, u64_t seed, allocator_t const& allocator) {
        return generate_values(count, seed, overlap_alphabet_k, overlap_symbols_k, allocator);
    }
};

static_assert(value<sequence_value_t>);
static_assert(metric<levenshtein_bytes_fixture_t::metric_t, sequence_value_t, i64_t>);
static_assert(metric<levenshtein_runes_fixture_t::metric_t, sequence_value_t, i64_t>);
static_assert(metric<overlap_fixture_t::metric_t, sequence_value_t, i64_t>);
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
    constexpr i64_t tight_radius_k = 3;
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

        levenshtein_metric<capability_, symbol_> metric;
        failures += expect_eq(metric.init(candidates_k), status_t::success_k);
        for (i64_t const radius : {unbounded_radius<i64_t>(), tight_radius_k}) {
            for (std::size_t const query : {std::size_t {0}, generated_k}) {
                metric.reset(candidates[query]);
                for (sequence_value_t const& candidate : candidates)
                    failures += expect(metric.push(candidate) == metric_staging_t::staged_k);
                std::array<i64_t, candidates_k> distances;
                metric.drain(distances, radius);
                failures += expect_eq(metric.status(), status_t::success_k);
                for (std::size_t candidate = 0; candidate != candidates_k; ++candidate) {
                    i64_t const truth = levenshtein_oracle(candidates[query], candidates[candidate], symbol_);
                    failures += truth <= radius ? expect_eq(distances[candidate], truth)
                                                : expect(distances[candidate] > radius);
                }
            }
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** One overlap kit against the brute-force window count, exact at any radius: generated strings, a
 *  string shorter than every width, and the empty one, opening a round that shares no window. */
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

        overlap_metric<capability_> metric;
        failures += expect_eq(metric.init(candidates_k, sequence_window_widths_k), status_t::success_k);
        for (std::size_t const query : {std::size_t {0}, candidates_k - 1}) {
            metric.reset(candidates[query]);
            for (sequence_value_t const& candidate : candidates)
                failures += expect(metric.push(candidate) == metric_staging_t::staged_k);
            std::array<i64_t, candidates_k> distances;
            metric.drain(distances, i64_t {0});
            failures += expect_eq(metric.status(), status_t::success_k);
            for (std::size_t candidate = 0; candidate != candidates_k; ++candidate)
                failures += expect_eq(distances[candidate], overlap_oracle(candidates[query], candidates[candidate]));
        }
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

std::size_t check_sequence(settings_t const& settings) {
    std::size_t failures = 0;
    failures += test_basic_scenarios<levenshtein_bytes_fixture_t>(settings.seed);
    failures += test_basic_scenarios<levenshtein_runes_fixture_t>(settings.seed);
    failures += test_basic_scenarios<overlap_fixture_t>(settings.seed);
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
    return report("sequence", failures);
}

} // namespace unum::usearch::test::sequence
