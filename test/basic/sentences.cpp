/**
 *  @brief The sentences unit of the basic-index suite: string payloads under unit-cost
 *      Levenshtein distance - the transparent-metric, exact-integer-distance, narrow-id,
 *      fuzzed-waiter corner of the instantiation space.
 *
 *  @author Ash Vardanian
 *  @file test/basic/sentences.cpp
 *  @date August 26, 2026
 *
 *  Sentences are drawn from planted templates over a small vocabulary with a few character
 *  edits each, so edit distances bimodalize - within a template only the edits differ,
 *  across templates most characters do - giving the graph structure to navigate.
 */
#include <array>       // `std::array` - the vocabulary & handmade triple
#include <span>        // `std::span` - the metric's drain surface
#include <string>      // `std::string` - the stored payload
#include <string_view> // `std::string_view` - the transparent query form
#include <vector>      // `std::vector`

#include "basic.hpp"

using namespace unum::usearch;

/** @brief Unit-cost Levenshtein through the classic two-row Wagner-Fischer recurrence -
 *      both the metric's kernel and the scenarios' oracle, so the two always agree. */
static u64_t levenshtein_distance(std::string_view first, std::string_view second) noexcept {
    std::vector<u64_t> previous_row(second.size() + 1), current_row(second.size() + 1);
    for (std::size_t column = 0; column <= second.size(); ++column) previous_row[column] = column;
    for (std::size_t row = 1; row <= first.size(); ++row) {
        current_row[0] = row;
        for (std::size_t column = 1; column <= second.size(); ++column) {
            u64_t const substituted = previous_row[column - 1] + (first[row - 1] != second[column - 1]);
            u64_t const erased = previous_row[column] + 1;
            u64_t const inserted = current_row[column - 1] + 1;
            current_row[column] = std::min(substituted, std::min(erased, inserted));
        }
        std::swap(previous_row, current_row);
    }
    return previous_row[second.size()];
}

/** @brief The unit's batched metric: stages views over the stored strings, scores a whole
 *      round in one drain, and refuses pushes past a fixed staging cap - so the index's
 *      saturation paths are reachable with a real metric, not only the auditing double. */
struct levenshtein_metric_t {
    using is_transparent = void;
    static constexpr std::size_t staging_limit_k = 256;

    std::string_view query_;
    std::vector<std::string_view> staged_;

    void reset(std::string_view query) noexcept { query_ = query, staged_.clear(); }
    void reset(std::string const& query) noexcept { reset(std::string_view {query}); }
    basic_metric_staging_t push(std::string const& value) noexcept {
        if (staged_.size() == staging_limit_k) return basic_metric_staging_t::refused_k;
        return staged_.push_back(value), basic_metric_staging_t::staged_k;
    }
    void drain(std::span<u64_t> distances) noexcept {
        for (std::size_t lane = 0; lane != distances.size(); ++lane)
            distances[lane] = levenshtein_distance(query_, staged_[lane]);
        staged_.clear();
    }
};

/** @brief Identity-bytes codec: the payload is the string's own characters, and a @c 0xFF
 *      byte - impossible in generated ASCII - is the unit's poison marker on load. */
struct sentence_codec_t {
    std::size_t saved_bytes(std::string const& value) const noexcept { return value.size(); }
    status_t save(std::string const& value, bytes_span_t output) const noexcept {
        if (output.size() < value.size()) return status_t::overflow_k;
        for (std::size_t position = 0; position != value.size(); ++position)
            output[position] = static_cast<std::byte>(value[position]);
        return status_t::success_k;
    }
    status_t load(bytes_view_t input, std::string& value) const noexcept {
        for (std::byte const byte : input)
            if (byte == std::byte {0xFF}) return status_t::malformed_k;
        value.assign(reinterpret_cast<char const*>(input.data()), input.size());
        return status_t::success_k;
    }
};

/** @brief The handmade triple with its paper-verified pairwise Levenshtein matrix. */
struct sentences_handmade_t {
    std::array<std::string, 3> values;
    std::array<std::array<u64_t, 3>, 3> distances;
};

struct sentences_fixture_t {
    using index_t = basic_ram_index<std::string, u64_t, u32_t, u64_t, flaky_waiter_t, counting_allocator<std::byte>>;
    static constexpr bool exact_distances_k = true;

    static index_t make_index(allocation_ledger_t& ledger) { return index_t {counting_allocator<std::byte> {ledger}}; }
    static levenshtein_metric_t make_metric() { return {}; }
    static sentence_codec_t make_codec() { return {}; }
    static u64_t make_key(std::size_t ordinal) { return 0xC0FFEE00000000ull + ordinal; }
    static u64_t oracle(std::string const& query, std::string const& stored) noexcept {
        return levenshtein_distance(query, stored);
    }
    static bool value_equals(std::string const& first, std::string const& second) noexcept { return first == second; }

    static sentences_handmade_t make_handmade() {
        return {{"kitten", "sitting", "sitten"}, {{{0, 3, 1}, {3, 0, 2}, {1, 2, 0}}}};
    }

    static std::vector<std::string> make_values(std::size_t count, u64_t seed) {
        constexpr std::array<std::string_view, 12> vocabulary_k {"graph", "index",  "vector",  "search",
                                                                 "batch", "metric", "kernel",  "thread",
                                                                 "cache", "tensor", "payload", "cluster"};
        constexpr std::size_t templates_k = 8;
        constexpr std::size_t words_per_sentence_k = 6;
        splitmix64_t random_generator {seed};
        uniform_below<u32_t> word_distribution {static_cast<u32_t>(vocabulary_k.size())};
        uniform_below<u32_t> template_distribution {templates_k};
        uniform_below<u32_t> letter_distribution {26};
        uniform_below<u32_t> edit_kind_distribution {3};

        std::array<std::string, templates_k> templates;
        for (auto& sentence : templates)
            for (std::size_t word = 0; word != words_per_sentence_k; ++word) {
                if (word) sentence += ' ';
                sentence += vocabulary_k[word_distribution(random_generator)];
            }

        std::vector<std::string> values;
        for (std::size_t ordinal = 0; ordinal != count; ++ordinal) {
            std::string sentence = templates[template_distribution(random_generator)];
            std::size_t const edits = sentence.size() / 12;
            for (std::size_t edit = 0; edit != edits; ++edit) {
                std::size_t const position = uniform_below<u32_t> {static_cast<u32_t>(sentence.size())}(
                    random_generator);
                char const letter = static_cast<char>('a' + letter_distribution(random_generator));
                switch (edit_kind_distribution(random_generator)) {
                case 0: sentence[position] = letter; break;
                case 1: sentence.insert(sentence.begin() + position, letter); break;
                default: sentence.erase(sentence.begin() + position); break;
                }
            }
            values.push_back(std::move(sentence));
        }
        return values;
    }
};

static_assert(basic_value<std::string>);
static_assert(basic_metric<levenshtein_metric_t, std::string, u64_t>);
static_assert(transparent_metric<levenshtein_metric_t>);
static_assert(basic_query<std::string_view, levenshtein_metric_t, std::string>);
static_assert(basic_value_codec<sentence_codec_t, std::string>);

/** @brief The transparency arm: a @c string_view query - never a stored value - opens the
 *      metric's rounds, and its self-match still lands at distance zero. */
static std::size_t check_sentences_transparent_search() {
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        sentences_fixture_t const fixture;
        auto index = sentences_fixture_t::make_index(ledger);
        auto metric = fixture.make_metric();
        sentences_fixture_t::index_t::traversal_context_t context {basic_seed_guards_k,
                                                                   counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(16, basic_seed_guards_k ^ 0x11ull);
        std::string const copied_value = values[0];
        failures += expect(bool(index.reserve(values.size())));
        std::vector<u32_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context, basic_keep_diverse_t {4});

        auto const found = index.search(std::string_view {copied_value}, metric, context);
        failures += expect_eq(found.status, status_t::success_k);
        failures += expect(found.count > 0);
        failures += expect_eq(context.results[0].distance, u64_t {0});
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief The codec-rejection row of the hostile battery: a poison byte in the values
 *      section must surface the codec's own refusal through @c load. */
static std::size_t check_sentences_codec_poison() {
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        sentences_fixture_t const fixture;
        auto index = sentences_fixture_t::make_index(ledger);
        auto metric = fixture.make_metric();
        auto const codec = fixture.make_codec();
        sentences_fixture_t::index_t::traversal_context_t context {basic_seed_hostile_k,
                                                                   counting_allocator<std::byte> {ledger}};
        auto values = fixture.make_values(8, basic_seed_hostile_k ^ 0x22ull);
        failures += expect(bool(index.reserve(values.size())));
        std::vector<u32_t> assigned_ids;
        failures += basic_build(fixture, index, metric, values, assigned_ids, context, basic_keep_diverse_t {4});

        auto const bytes = index.snapshot_bytes(codec);
        failures += expect(bool(bytes));
        std::vector<std::byte> payload(bytes.value);
        failures += expect_eq(index.save({payload}, codec), status_t::success_k);
        payload.back() = std::byte {0xFF}; // The values section is last, so this poisons one value.

        auto poisoned = sentences_fixture_t::make_index(ledger);
        failures += expect(bool(poisoned.reserve(values.size())));
        failures += expect_eq(poisoned.load({payload}, codec), status_t::malformed_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** @brief Myers' bit-parallel Levenshtein - a genuinely different algorithm from the
 *      unit's Wagner-Fischer rows, so a bug in the shared metric-and-oracle DP cannot
 *      cancel out of the checks. The single-word form bounds patterns to 64 characters. */
static u64_t myers_levenshtein_distance(std::string_view first, std::string_view second) noexcept {
    if (first.size() > second.size()) return myers_levenshtein_distance(second, first);
    if (first.empty()) return second.size();
    std::array<u64_t, 256> pattern_masks {};
    for (std::size_t position = 0; position != first.size(); ++position)
        pattern_masks[static_cast<u8_t>(first[position])] |= u64_t {1} << position;
    u64_t positive_vertical = ~u64_t {0};
    u64_t negative_vertical = 0;
    u64_t distance = first.size();
    u64_t const last_bit = u64_t {1} << (first.size() - 1);
    for (char const character : second) {
        u64_t const pattern_mask = pattern_masks[static_cast<u8_t>(character)];
        u64_t const diagonal = (((pattern_mask & positive_vertical) + positive_vertical) ^ positive_vertical) |
                               pattern_mask | negative_vertical;
        u64_t positive_horizontal = negative_vertical | ~(diagonal | positive_vertical);
        u64_t negative_horizontal = diagonal & positive_vertical;
        if (positive_horizontal & last_bit) ++distance;
        if (negative_horizontal & last_bit) --distance;
        positive_horizontal = (positive_horizontal << 1) | 1;
        negative_horizontal = negative_horizontal << 1;
        positive_vertical = negative_horizontal | ~(diagonal | positive_horizontal);
        negative_vertical = positive_horizontal & diagonal;
    }
    return distance;
}

/** @brief The independent-oracle audit over sampled pairs, truncated to the bit-parallel
 *      form's 64-character bound. */
static std::size_t check_sentences_independent_oracle() {
    std::size_t failures = 0;
    auto values = sentences_fixture_t::make_values(24, basic_seed_contract_k ^ 0x0Bull);
    for (std::size_t pair = 0; pair + 1 < values.size(); pair += 2) {
        std::string_view const first = std::string_view {values[pair]}.substr(0, 64);
        std::string_view const second = std::string_view {values[pair + 1]}.substr(0, 64);
        failures += expect_eq(levenshtein_distance(first, second), myers_levenshtein_distance(first, second));
    }
    failures += expect_eq(levenshtein_distance("kitten", "sitting"), myers_levenshtein_distance("kitten", "sitting"));
    return failures;
}

std::size_t test_sentences() {
    sentences_fixture_t const fixture;
    std::size_t failures = 0;
    failures += test_basic_scenarios(fixture);
    failures += check_sentences_transparent_search();
    failures += check_sentences_codec_poison();
    failures += check_sentences_independent_oracle();
    failures += test_basic_safegraphs_adapter(fixture, "u64");
    return report("sentences", failures);
}
