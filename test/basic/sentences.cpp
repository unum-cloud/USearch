/**
 *  @file test/basic/sentences.cpp
 *  @author Ash Vardanian
 *  @date August 26, 2026
 *  @brief The sentences unit of the basic-index suite: string payloads under unit-cost Levenshtein
 *      distance - the transparent-metric, exact-integer-distance, narrow-ID, fuzzed-waiter corner
 *      of the instantiation space.
 *
 *  Sentences are drawn from planted templates over a small vocabulary with a few character
 *  edits each, so edit distances bimodalize - within a template only the edits differ,
 *  across templates most characters do - giving the graph structure to navigate.
 */
#include <array>       // `std::array` - the vocabulary & handmade triple
#include <atomic>      // `std::atomic` - the parked batch's rendezvous
#include <span>        // `std::span` - the metric's drain surface
#include <string>      // `std::string` - the stored payload
#include <string_view> // `std::string_view` - the transparent query form
#include <vector>      // `std::vector`

#include "basic.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

/** Unit-cost Levenshtein through the classic two-row Wagner-Fischer recurrence - both
 *  the metric's kernel and the scenarios' oracle, so the two always agree; the caller owns
 *  the rows, so a whole batch scores without allocating. */
static u64_t levenshtein_distance(std::string_view first, std::string_view second, std::vector<u64_t>& previous_row,
                                  std::vector<u64_t>& current_row) noexcept {
    previous_row.assign(second.size() + 1, 0), current_row.assign(second.size() + 1, 0);
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

/** The unit's batched metric: stages views over the stored strings, scores a whole
 *  round in one drain, and refuses pushes past a fixed staging cap - so the index's
 *  saturation paths are reachable with a real metric, not only the auditing double. */
struct levenshtein_metric_t {
    using is_transparent = void;
    static constexpr std::size_t staging_limit_k = 256;

    std::string_view query_;
    std::vector<std::string_view> staged_;
    std::vector<u64_t> previous_row_;
    std::vector<u64_t> current_row_;

    void reset(std::string_view query) noexcept { query_ = query, staged_.clear(); }
    void reset(std::string const& query) noexcept { reset(std::string_view {query}); }
    metric_staging_t push(std::string const& value) noexcept {
        if (staged_.size() == staging_limit_k) return metric_staging_t::refused_k;
        return staged_.push_back(value), metric_staging_t::staged_k;
    }
    void drain(std::span<u64_t> distances, u64_t) noexcept {
        for (std::size_t lane = 0; lane != distances.size(); ++lane)
            distances[lane] = levenshtein_distance(query_, staged_[lane], previous_row_, current_row_);
        staged_.clear();
    }
};

/** Identity-bytes codec: the payload is the string's own characters, a @c utf8
 *  column in the interchange, and a @c 0xFF byte - impossible in generated ASCII - is
 *  the unit's poison marker on load. */
struct sentence_codec_t {
    inline static constexpr dtype_t value_dtype_k = dtype_t::utf8_k;

    std::size_t storage_requirements(std::string const& value) const noexcept { return value.size(); }
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

/** The handmade triple with its paper-verified pairwise Levenshtein matrix. */
struct sentences_handmade_t {
    std::array<std::string, 3> values;
    std::array<std::array<u64_t, 3>, 3> distances;
};

template <typename wait_type_ = dummy_waiter_t, template <typename> class atomic_ref_template_ = standard_atomic_ref>
struct sentences_fixture {
    using index_t = ram_index<std::string, u64_t, u32_t, u64_t, wait_type_, atomic_ref_template_, counting_allocator_t>;

    static std::unique_ptr<index_t> make_index(allocation_ledger_t& ledger) {
        return std::unique_ptr<index_t>(new (std::nothrow) index_t(counting_allocator_t {ledger}));
    }
    static levenshtein_metric_t make_metric() { return {}; }
    static sentence_codec_t make_codec() { return {}; }
    static u64_t make_key(std::size_t ordinal) { return 0xC0FFEE00000000ull + ordinal; }
    static u64_t oracle(std::string const& query, std::string const& stored) noexcept {
        std::vector<u64_t> previous_row, current_row;
        return levenshtein_distance(query, stored, previous_row, current_row);
    }
    static bool value_equals(std::string const& first, std::string const& second) noexcept { return first == second; }

    static sentences_handmade_t make_handmade(typename index_t::allocator_t const&) {
        return {{"kitten", "sitting", "sitten"}, {{{0, 3, 1}, {3, 0, 2}, {1, 2, 0}}}};
    }

    static std::vector<std::string> make_values(std::size_t count, u64_t seed, typename index_t::allocator_t const&) {
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

using sentences_fixture_t = sentences_fixture<flaky_waiter_t>;

static_assert(value<std::string>);
static_assert(metric<levenshtein_metric_t, std::string, u64_t>);
static_assert(transparent_metric<levenshtein_metric_t>);
static_assert(query<std::string_view, levenshtein_metric_t, std::string>);
static_assert(value_codec<sentence_codec_t, std::string>);

/** The transparency case: a @c string_view query - never a stored value - opens the
 *  metric's rounds, and its self-match still lands at distance zero. */
static std::size_t check_sentences_transparent_search(u64_t seed) {
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        auto const index_ptr = sentences_fixture_t::make_index(ledger);
        auto& index = *index_ptr;
        auto metric = sentences_fixture_t::make_metric();
        sentences_fixture_t::index_t::traversal_context_t context {stream_key(seed, "transparent_search"),
                                                                   counting_allocator_t {ledger}};
        auto values = sentences_fixture_t::make_values(16, stream_key(seed, "transparent_search", 1),
                                                       counting_allocator_t {ledger});
        std::string const copied_value = values[0];
        failures += expect(bool(index.reserve(values.size())));
        std::vector<u32_t> assigned_ids;
        failures += basic_build<sentences_fixture_t>(index, metric, values, assigned_ids, context, keep_diverse_t {4});

        auto const [result, status] = index.search(std::string_view {copied_value}, metric, context);
        failures += expect_eq(status, status_t::success_k);
        failures += expect(result.count > 0);
        failures += expect_eq(context.results[0].distance, u64_t {0});
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** The codec-rejection row of the hostile battery: a poison byte in the values
 *  section must surface the codec's own refusal through @c load, and an insert into the
 *  failed index meets neither the dead attempt's keys nor its rows. */
static std::size_t check_sentences_codec_poison(u64_t seed) {
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        auto const index_ptr = sentences_fixture_t::make_index(ledger);
        auto& index = *index_ptr;
        auto metric = sentences_fixture_t::make_metric();
        auto const codec = sentences_fixture_t::make_codec();
        sentences_fixture_t::index_t::traversal_context_t context {stream_key(seed, "codec_poison"),
                                                                   counting_allocator_t {ledger}};
        auto values = sentences_fixture_t::make_values(8, stream_key(seed, "codec_poison", 1),
                                                       counting_allocator_t {ledger});
        failures += expect(bool(index.reserve(values.size())));
        std::vector<u32_t> assigned_ids;
        failures += basic_build<sentences_fixture_t>(index, metric, values, assigned_ids, context, keep_diverse_t {4});

        auto [columns, status] = basic_saved_columns(index, codec);
        failures += expect(succeeded(status));
        columns.values.back() = std::byte {0xFF}; // Poisons the last value's final byte.

        auto const poisoned_ptr = sentences_fixture_t::make_index(ledger);
        auto& poisoned = *poisoned_ptr;
        failures += expect(bool(poisoned.reserve(values.size())));
        failures += expect_eq(poisoned.load(columns.view(), codec), status_t::malformed_k);

        // The dead plan decoded every row but the last, so the IDs an insert now dispenses held its keys and rows.
        auto fresh_values = sentences_fixture_t::make_values(2, stream_key(seed, "codec_poison", 2),
                                                             counting_allocator_t {ledger});
        std::array<u64_t, 2> const fresh_keys {sentences_fixture_t::make_key(values.size()),
                                               sentences_fixture_t::make_key(values.size() + 1)};
        auto const [inserted, insert_status] = poisoned.insert(std::span(fresh_values), std::span(fresh_keys), metric,
                                                               {}, context, keep_diverse_t {4});
        failures += expect_eq(insert_status, status_t::success_k);
        failures += expect(!poisoned.contains(sentences_fixture_t::make_key(0)));
        failures += expect_eq(poisoned.degree_of(0).value, 1u);
        failures += expect_eq(poisoned.health(context).status, status_t::success_k);
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** A batch publishes whole: a two-value batch on an empty index parks in its second row's walk,
 *  which already admitted the first row through its `wired_k` bit, and no reader sees either row
 *  until the batch's turn. */
static std::size_t check_sentences_whole_batch(u64_t seed) {
    using index_t = sentences_fixture_t::index_t;
    using context_t = index_t::traversal_context_t;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_t> const index_ptr = sentences_fixture_t::make_index(ledger);
        index_t& index = *index_ptr;
        auto values = sentences_fixture_t::make_values(2, stream_key(seed, "whole_batch"),
                                                       counting_allocator_t {ledger});
        std::array<u64_t, 2> const keys {sentences_fixture_t::make_key(0), sentences_fixture_t::make_key(1)};
        failures += expect(bool(index.reserve(values.size())));
        std::atomic<bool> entered {false}, released {false};
        std::atomic<std::size_t> thread_failures {0};
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(2))) return failures + expect(false);
        pool.for_threads([&](std::size_t thread) noexcept {
            if (thread == 0) {
                basic_gated_metric<levenshtein_metric_t, u64_t> gated {sentences_fixture_t::make_metric(), &entered,
                                                                       &released};
                context_t context {stream_key(seed, "whole_batch", 1), counting_allocator_t {ledger}};
                auto const [result, status] = index.insert(std::span(values), std::span(keys), gated, {}, context,
                                                           keep_diverse_t {4});
                if (status != status_t::success_k) thread_failures.fetch_add(1);
                return;
            }
            if (!spin_until([&]() noexcept { return entered.load(std::memory_order_acquire); })) {
                thread_failures.fetch_add(1);
                released.store(true, std::memory_order_release); // The gate opens even on failure.
                return;
            }
            if (index.size() != 0 || succeeded(index.key_of(0).status) || index.contains(keys[0]))
                thread_failures.fetch_add(1);
            released.store(true, std::memory_order_release);
        });
        failures += expect_eq(thread_failures.load(), 0u);
        failures += expect_eq(index.size(), 2u);
        failures += expect(index.contains(keys[1]));
        failures += expect_eq(index.degree_of(1).value, 1u); // Linked to the first row before either published.
    }
    failures += expect_eq(ledger.live_bytes.load(), 0);
    return failures;
}

/** Myers' bit-parallel Levenshtein - a genuinely different algorithm from the
 *  unit's Wagner-Fischer rows, so a bug in the shared metric-and-oracle DP cannot
 *  cancel out of the checks. The single-word form bounds patterns to 64 characters. */
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

/** The independent-oracle audit over sampled pairs, truncated to the bit-parallel
 *  form's 64-character bound. */
static std::size_t check_sentences_independent_oracle(u64_t seed) {
    std::size_t failures = 0;
    std::vector<u64_t> previous_row, current_row;
    allocation_ledger_t ledger;
    auto values = sentences_fixture_t::make_values(24, stream_key(seed, "independent_oracle"),
                                                   counting_allocator_t {ledger});
    for (std::size_t pair = 0; pair + 1 < values.size(); pair += 2) {
        std::string_view const first = std::string_view {values[pair]}.substr(0, 64);
        std::string_view const second = std::string_view {values[pair + 1]}.substr(0, 64);
        failures += expect_eq(levenshtein_distance(first, second, previous_row, current_row),
                              myers_levenshtein_distance(first, second));
    }
    failures += expect_eq(levenshtein_distance("kitten", "sitting", previous_row, current_row),
                          myers_levenshtein_distance("kitten", "sitting"));
    return failures;
}

std::size_t test::basic::check_sentences(settings_t const& settings) {
    std::size_t failures = 0;
    failures += test_basic_scenarios<sentences_fixture_t>(settings.seed);
    failures += test_basic_scenarios<sentences_fixture<dummy_waiter_t>>(settings.seed);
    failures += test_basic_scenarios<sentences_fixture<flaky_waiter_t, fu::preferred_atomic_ref>>(settings.seed);
    failures += check_sentences_transparent_search(settings.seed);
    failures += check_sentences_codec_poison(settings.seed);
    failures += check_sentences_whole_batch(settings.seed);
    failures += check_sentences_independent_oracle(settings.seed);
    failures += test_basic_safegraphs_roundtrip<sentences_fixture_t>(settings.seed, "u64", "levenshtein");
    failures += test_basic_safegraphs_hostile<sentences_fixture_t>(settings.seed, "levenshtein");
    failures += test_basic_safegraphs_distributed<sentences_fixture_t>(settings.seed, "levenshtein");
    return report("sentences", failures);
}
