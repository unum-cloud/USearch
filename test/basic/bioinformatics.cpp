/**
 *  @file test/basic/bioinformatics.cpp
 *  @author Ash Vardanian
 *  @date August 26, 2026
 *  @brief The bioinformatics unit of the basic-index suite.
 *
 *  ACGT reads under an alignment distance with transition/transversion-weighted substitutions - the
 *  non-transparent, integer-exact, wide-ID, @c unum::usearch::uuid_t-keyed corner, and the only
 *  unit whose serialized bytes differ from the in-RAM form.
 *
 *  Reads are drawn from planted reference sequences with a few point mutations & indels
 *  each, so alignment distances bimodalize the way real read-vs-reference data does. The
 *  codec packs nucleotides two bits each, four per byte, proving @c load decodes rather
 *  than copies.
 */
#include <array>       // `std::array` - the handmade triple
#include <span>        // `std::span` - the metric's drain surface
#include <string>      // `std::string` - the read payload, one `ACGT` letter per base
#include <string_view> // `std::string_view` - the staged views
#include <vector>      // `std::vector`

#include "basic.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;

/** One read: a plain string over the @c ACGT alphabet. */
using dna_t = std::string;

/** Substituting within a chemical class - purine ↔ purine, pyrimidine ↔
 *  pyrimidine - is a transition & costs half of a transversion. */
static u64_t substitution_cost(char first, char second) noexcept {
    if (first == second) return 0;
    auto is_purine = [](char nucleotide) noexcept { return nucleotide == 'A' || nucleotide == 'G'; };
    return is_purine(first) == is_purine(second) ? 1u : 2u;
}

/** Global alignment distance under the weighted substitutions & unit-gap cost of two -
 *  both the metric's kernel and the scenarios' oracle; the caller owns the rows, so a whole
 *  batch scores without allocating. */
static u64_t alignment_distance(std::string_view first, std::string_view second, std::vector<u64_t>& previous_row,
                                std::vector<u64_t>& current_row) noexcept {
    constexpr u64_t gap_cost_k = 2;
    previous_row.assign(second.size() + 1, 0), current_row.assign(second.size() + 1, 0);
    for (std::size_t column = 0; column <= second.size(); ++column) previous_row[column] = column * gap_cost_k;
    for (std::size_t row = 1; row <= first.size(); ++row) {
        current_row[0] = row * gap_cost_k;
        for (std::size_t column = 1; column <= second.size(); ++column) {
            u64_t const substituted = previous_row[column - 1] + substitution_cost(first[row - 1], second[column - 1]);
            u64_t const erased = previous_row[column] + gap_cost_k;
            u64_t const inserted = current_row[column - 1] + gap_cost_k;
            current_row[column] = std::min(substituted, std::min(erased, inserted));
        }
        std::swap(previous_row, current_row);
    }
    return previous_row[second.size()];
}

/** The unit's batched metric - deliberately non-transparent: queries must be
 *  stored-form reads, proving the same-type fallback of @c basic_query. */
struct alignment_metric_t {
    std::string_view query_;
    std::vector<std::string_view> staged_;
    std::vector<u64_t> previous_row_;
    std::vector<u64_t> current_row_;

    void reset(dna_t const& query) noexcept { query_ = query, staged_.clear(); }
    basic_metric_staging_t push(dna_t const& value) noexcept {
        return staged_.push_back(value), basic_metric_staging_t::staged_k;
    }
    void drain(std::span<u64_t> distances) noexcept {
        for (std::size_t lane = 0; lane != distances.size(); ++lane)
            distances[lane] = alignment_distance(query_, staged_[lane], previous_row_, current_row_);
        staged_.clear();
    }
};

/** Packs nucleotides two bits each behind a @c u32_t base count, so the payload is
 *  about a quarter of the in-RAM form and @c load provably decodes; the count stays,
 *  since padding hides it from the byte length, so the bytes ride as opaque @c u8. */
struct packed_dna_codec_t {
    inline static constexpr dtype_t value_dtype_k = dtype_t::u8_k;

    static std::size_t packed_bytes(std::size_t bases) noexcept { return divide_round_up<std::size_t>(bases, 4); }
    static u32_t code_of(char nucleotide) noexcept {
        switch (nucleotide) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        default: return 3;
        }
    }

    std::size_t storage_requirements(dna_t const& value) const noexcept {
        return sizeof(u32_t) + packed_bytes(value.size());
    }
    status_t save(dna_t const& value, bytes_span_t output) const noexcept {
        if (output.size() < storage_requirements(value)) return status_t::overflow_k;
        auto const count_bytes = std::bit_cast<std::array<std::byte, sizeof(u32_t)>>(static_cast<u32_t>(value.size()));
        for (std::size_t position = 0; position != count_bytes.size(); ++position)
            output[position] = count_bytes[position];
        for (std::size_t byte = 0; byte != packed_bytes(value.size()); ++byte) {
            u32_t packed = 0;
            for (std::size_t lane = 0; lane != 4 && byte * 4 + lane != value.size(); ++lane)
                packed |= code_of(value[byte * 4 + lane]) << (lane * 2);
            output[sizeof(u32_t) + byte] = static_cast<std::byte>(packed);
        }
        return status_t::success_k;
    }
    status_t load(bytes_view_t input, dna_t& value) const noexcept {
        if (input.size() < sizeof(u32_t)) return status_t::malformed_k;
        std::array<std::byte, sizeof(u32_t)> count_bytes;
        for (std::size_t position = 0; position != count_bytes.size(); ++position)
            count_bytes[position] = input[position];
        std::size_t const bases = std::bit_cast<u32_t>(count_bytes);
        if (input.size() != sizeof(u32_t) + packed_bytes(bases)) return status_t::malformed_k;
        value.clear();
        for (std::size_t base = 0; base != bases; ++base) {
            u32_t const packed = static_cast<u32_t>(input[sizeof(u32_t) + base / 4]);
            value += "ACGT"[(packed >> (base % 4 * 2)) & 3u];
        }
        return status_t::success_k;
    }
};

/** The handmade triple: one transition, one transversion, both cheaper than gaps. */
struct bioinformatics_handmade_t {
    std::array<dna_t, 3> values;
    std::array<std::array<u64_t, 3>, 3> distances;
};

template <typename wait_type_ = dummy_waiter_t, template <typename> class atomic_ref_template_ = std::atomic_ref>
struct bioinformatics_fixture {
    using index_t =
        basic_ram_index<dna_t, us::uuid_t, u64_t, u64_t, wait_type_, atomic_ref_template_, counting_allocator_t>;

    static index_t make_index(allocation_ledger_t& ledger) { return index_t {counting_allocator_t {ledger}}; }
    static alignment_metric_t make_metric() { return {}; }
    static packed_dna_codec_t make_codec() { return {}; }
    static us::uuid_t make_key(std::size_t ordinal) {
        return std::bit_cast<us::uuid_t>(std::array<u64_t, 2> {mix_bits(ordinal), 0xB10B10B1ull ^ ordinal});
    }
    static u64_t oracle(dna_t const& query, dna_t const& stored) noexcept {
        std::vector<u64_t> previous_row, current_row;
        return alignment_distance(query, stored, previous_row, current_row);
    }
    static bool value_equals(dna_t const& first, dna_t const& second) noexcept { return first == second; }

    static bioinformatics_handmade_t make_handmade() {
        return {{"AAAA", "AAAG", "AAAC"}, {{{0, 1, 2}, {1, 0, 2}, {2, 2, 0}}}};
    }

    static std::vector<dna_t> make_values(std::size_t count, u64_t seed) {
        constexpr std::size_t references_k = 6;
        constexpr std::size_t reference_bases_k = 48;
        splitmix64_t random_generator {seed};
        uniform_below<u32_t> nucleotide_distribution {4};
        uniform_below<u32_t> reference_distribution {references_k};
        uniform_below<u32_t> mutation_kind_distribution {10};

        std::array<dna_t, references_k> references;
        for (auto& reference : references)
            for (std::size_t base = 0; base != reference_bases_k; ++base)
                reference += "ACGT"[nucleotide_distribution(random_generator)];

        std::vector<dna_t> values;
        for (std::size_t ordinal = 0; ordinal != count; ++ordinal) {
            dna_t read = references[reference_distribution(random_generator)];
            std::size_t const mutations = read.size() / 12;
            for (std::size_t mutation = 0; mutation != mutations; ++mutation) {
                std::size_t const position = uniform_below<u32_t> {static_cast<u32_t>(read.size())}(random_generator);
                char const nucleotide = "ACGT"[nucleotide_distribution(random_generator)];
                switch (mutation_kind_distribution(random_generator)) {
                case 8: read.insert(read.begin() + position, nucleotide); break;
                case 9: read.erase(read.begin() + position); break;
                default: read[position] = nucleotide; break; // Point mutations dominate.
                }
            }
            values.push_back(std::move(read));
        }
        return values;
    }
};

using bioinformatics_fixture_t = bioinformatics_fixture<>;

static_assert(basic_value<dna_t>);
static_assert(basic_metric<alignment_metric_t, dna_t, u64_t>);
static_assert(!transparent_metric<alignment_metric_t>);
static_assert(basic_query<dna_t, alignment_metric_t, dna_t>);
static_assert(basic_value_codec<packed_dna_codec_t, dna_t>);

/** The codec on its own: a bit-exact pack/unpack roundtrip at every length modulo
 *  four, and a length prefix disagreeing with its slice refused as malformed. */
static std::size_t check_bioinformatics_codec() {
    std::size_t failures = 0;
    packed_dna_codec_t const codec;
    for (std::size_t bases = 0; bases != 9; ++bases) {
        dna_t original;
        for (std::size_t base = 0; base != bases; ++base) original += "ACGT"[(base * 7 + bases) % 4];
        std::vector<std::byte> encoded(codec.storage_requirements(original));
        failures += expect_eq(codec.save(original, {encoded}), status_t::success_k);
        dna_t decoded;
        failures += expect_eq(codec.load({encoded}, decoded), status_t::success_k);
        failures += expect(decoded == original);
        if (encoded.size() > sizeof(u32_t)) {
            dna_t rejected;
            failures += expect_eq(codec.load({encoded.data(), encoded.size() - 1}, rejected), status_t::malformed_k);
        }
    }
    return failures;
}

/** Exhaustive alignment-tree search - enumerates every gap/substitution alignment
 *  instead of running the row DP, so a bug in the unit's shared metric-and-oracle
 *  kernel cannot cancel out. Exponential, so only short prefixes feed it. */
static u64_t exhaustive_alignment_distance(std::string_view first, std::string_view second) noexcept {
    constexpr u64_t gap_cost_k = 2;
    if (first.empty()) return second.size() * gap_cost_k;
    if (second.empty()) return first.size() * gap_cost_k;
    u64_t const substituted = exhaustive_alignment_distance(first.substr(1), second.substr(1)) +
                              substitution_cost(first[0], second[0]);
    u64_t const erased = exhaustive_alignment_distance(first.substr(1), second) + gap_cost_k;
    u64_t const inserted = exhaustive_alignment_distance(first, second.substr(1)) + gap_cost_k;
    return std::min(substituted, std::min(erased, inserted));
}

/** The independent-oracle audit: sampled read prefixes short enough for the
 *  exhaustive search, plus the handmade triple at full length. */
static std::size_t check_bioinformatics_independent_oracle() {
    std::size_t failures = 0;
    std::vector<u64_t> previous_row, current_row;
    auto values = bioinformatics_fixture_t::make_values(16, basic_seed_contract_k ^ 0x0Cull);
    for (std::size_t pair = 0; pair + 1 < values.size(); pair += 2) {
        std::string_view const first = std::string_view {values[pair]}.substr(0, 6);
        std::string_view const second = std::string_view {values[pair + 1]}.substr(0, 6);
        failures += expect_eq(alignment_distance(first, second, previous_row, current_row),
                              exhaustive_alignment_distance(first, second));
    }
    auto const handmade = bioinformatics_fixture_t::make_handmade();
    for (std::size_t row = 0; row != 3; ++row)
        for (std::size_t column = 0; column != 3; ++column)
            failures += expect_eq(exhaustive_alignment_distance(handmade.values[row], handmade.values[column]),
                                  handmade.distances[row][column]);
    return failures;
}

std::size_t test::basic::check_bioinformatics() {
    bioinformatics_fixture_t const fixture;
    std::size_t failures = 0;
    failures += test_basic_scenarios(fixture);
    failures += test_basic_scenarios(bioinformatics_fixture<flaky_waiter_t> {});
    failures += test_basic_scenarios(bioinformatics_fixture<dummy_waiter_t, fu::preferred_atomic_ref> {});
    failures += check_bioinformatics_codec();
    failures += check_bioinformatics_independent_oracle();
    failures += test_basic_safegraphs_roundtrip(fixture, "uuid", "needleman-wunsch"); // The uuid key column.
    failures += test_basic_safegraphs_hostile(fixture, "needleman-wunsch");
    failures += test_basic_safegraphs_distributed(fixture, "needleman-wunsch");
    return report("bioinformatics", failures);
}
