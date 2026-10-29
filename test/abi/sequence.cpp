/**
 *  @file test/abi/sequence.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The sequence suite over the C interface, swept on the diagonal: every metric at the best
 *      class and byte Levenshtein at every admitted class, over each compiled residency at one
 *      shard and at three.
 *
 *  Sequences are planted templates over the nucleotides with an eighth of their symbols edited, so
 *  neighbors cluster by template while the bytes differ; Needleman-Wunsch aligns them packed two
 *  bits apiece under transition and transversion costs. Each cell checks the verbs a caller
 *  reaches: the sequences land under their keys across the shards and each counts once and reads
 *  back, a walk keeps most of the exact scan's top ten, a RAM index refines and absorbs while a GPU
 *  one refuses both, the index saves and reloads into every residency and as a view byte for byte,
 *  an NVMe view answers and refuses every write, and a cleared index holds nothing.
 */
#include <array>  // `std::array` - the templates, the window widths
#include <string> // `std::string` - the strings, the cells' names
#include <vector> // `std::vector` - the tape's offsets, the weights

#include "usearch/sequence_index.h"
#include "usearch/sequence/ram_metric.hpp" // `pack_codes`, `packed_code_bits`, `packed_codes_bytes`

#include "harness.hpp"

namespace unum::usearch::test::abi {
using us::sequence::pack_codes;
using us::sequence::packed_code_bits;
using us::sequence::packed_codes_bytes;

/** Sequences a cell builds, the queries held out past them, the donor's sequences, and the
 *  nucleotides per template. */
struct sequence_sizes_t {
    std::size_t sequences = 0, queries = 0, donor_sequences = 0, length = 0;
};

/** The sizes of the bit-parallel metrics' cells, and the smaller ones of Needleman-Wunsch, whose
 *  every distance fills a whole alignment matrix. */
static constexpr sequence_sizes_t sequence_sizes_k {1024, 128, 256, 100}, aligned_sizes_k {256, 64, 64, 32};

/** The breadth every walk takes. */
static constexpr std::size_t sequence_expansion_k = 128;

/** The nucleotides, whose position is their code under Needleman-Wunsch. */
static constexpr std::string_view nucleotides_k = "ACGT";

/** The window widths an overlap index compares at. */
static constexpr std::array<std::size_t, 2> sequence_window_widths_k {3, 5};

/** @p count sequences of about @p length nucleotides from eight templates, an eighth of every
 *  sequence's symbols substituted, inserted or erased. */
static std::vector<std::string> generate_nucleotides(std::size_t count, std::size_t length, u64_t seed) {
    constexpr std::size_t templates_k = 8;
    splitmix64_t random_generator {seed};
    uniform_below<u32_t> const symbol_distribution {static_cast<u32_t>(nucleotides_k.size())};
    uniform_below<u32_t> const template_distribution {templates_k};
    uniform_below<u32_t> const edit_kind_distribution {3};
    std::array<std::string, templates_k> templates;
    for (std::string& symbols : templates)
        for (std::size_t symbol = 0; symbol != length; ++symbol)
            symbols += nucleotides_k[symbol_distribution(random_generator)];
    std::vector<std::string> strings(count);
    for (std::string& symbols : strings) {
        symbols = templates[template_distribution(random_generator)];
        for (std::size_t edit = 0; edit != length / 8; ++edit) {
            std::size_t const position = uniform_below<u32_t> {static_cast<u32_t>(symbols.size())}(random_generator);
            char const symbol = nucleotides_k[symbol_distribution(random_generator)];
            switch (edit_kind_distribution(random_generator)) {
            case 0: symbols[position] = symbol; break;
            case 1: symbols.insert(symbols.begin() + position, symbol); break;
            default: symbols.erase(symbols.begin() + position); break;
            }
        }
    }
    return strings;
}

/** Sequences end to end and the offsets around them, as the C interface takes a tape: the raw
 *  bytes, or the nucleotides' codes packed two bits apiece. */
struct sequence_tape_t {
    std::string elements;
    std::vector<std::size_t> offsets {0};

    usearch_tape_view_t view() const noexcept { return {elements.data(), offsets.data(), usearch_scalar_u8_k}; }
};

static sequence_tape_t tape_of(std::vector<std::string> const& strings, bool packed) {
    sequence_tape_t tape;
    u32_t const code_bits = packed_code_bits(nucleotides_k.size());
    std::vector<u8_t> codes;
    for (std::string const& symbols : strings) {
        if (!packed) {
            tape.elements += symbols, tape.offsets.push_back(tape.elements.size());
            continue;
        }
        codes.clear();
        for (char const symbol : symbols) codes.push_back(static_cast<u8_t>(nucleotides_k.find(symbol)));
        std::size_t const first = tape.elements.size(), bytes = packed_codes_bytes(codes.size(), code_bits);
        tape.elements.resize(first + bytes);
        pack_codes(codes, code_bits, bytes_of(tape.elements.data() + first, bytes));
        tape.offsets.push_back(tape.elements.size());
    }
    return tape;
}

/** Transitions keep half the self-score and transversions none, over a gap of two. */
static std::vector<std::int8_t> nucleotide_similarities() {
    std::vector<std::int8_t> similarities(256 * 256);
    // Purines take even codes and pyrimidines odd, so a transition keeps half the self-score.
    for (std::size_t first = 0; first != nucleotides_k.size(); ++first)
        for (std::size_t second = 0; second != nucleotides_k.size(); ++second)
            similarities[first * 256 + second] = first == second ? 2 : (first & 1) == (second & 1) ? 1 : 0;
    return similarities;
}

/** Every key of the first @p count entries reads one sequence back, on host residencies. */
static std::size_t expect_gets(handle<sequence_verbs_t>& index, std::size_t count) {
    if (index.residency == usearch_residency_vram_k) return 0;
    std::size_t misread = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        usearch_tape_view_t sequences {};
        std::size_t written = 0;
        usearch_sequence_get_t get {};
        get.index = index.get(), get.executor = index.executor(), get.outcome = &outcome;
        get.key = key_of_offset(index.session->seed, offset);
        get.sequences = &sequences, get.written = &written, get.arena = index.session->arena;
        usearch_sequence_get(&get);
        misread += outcome.status != usearch_status_success_k || written != 1;
    }
    return check(misread == 0);
}

/** One metric of one cell, named @p name, as the file's docblock describes. */
static std::size_t check_sequence_metric(sweep_cell_t const& cell, session_t& session, sequence_geometry_t geometry,
                                         std::string const& name, u64_t seed) {
    bool const aligned = geometry.metadata.metric == usearch_sequence_metric_needleman_wunsch_k;
    std::vector<std::int8_t> const similarities = nucleotide_similarities();
    usearch_sequence_weights_t const weights {similarities.data(), 2};
    if (aligned) geometry.weights = &weights;
    if (geometry.metadata.metric == usearch_sequence_metric_overlap_k)
        geometry.window_widths = sequence_window_widths_k;
    handle<sequence_verbs_t> index = open_index<sequence_verbs_t>(session, geometry, cell.shards);
    if (!index) return 1;
    sequence_sizes_t const sizes = aligned ? aligned_sizes_k : sequence_sizes_k;
    sequence_tape_t const sequences = tape_of(generate_nucleotides(sizes.sequences + sizes.queries, sizes.length, seed),
                                              aligned);
    sequence_tape_t const donated = tape_of(generate_nucleotides(sizes.donor_sequences, sizes.length, ~seed), aligned);
    usearch_tape_view_t const queries = sequence_verbs_t::slice(sequences.view(), sizes.sequences);

    std::size_t failures = expect_built(cell, index, sequences.view(), sizes.sequences);
    if (failures) return failures;
    failures += expect_gets(index, sizes.sequences);
    failures += expect_walk_recall(index, queries, sizes.queries, sequence_expansion_k, 0.8);
    failures += expect_refined(index, usearch_status_unsupported_k);
    std::string const stem = fmt::format("/tmp/usearch-abi-sequence-{:016x}", stream_key(seed, name));
    failures += expect_persistence(cell, index, sequences.view(), queries, sizes.queries, sequence_expansion_k, 0.8,
                                   stem, saved_bytes_t::identical_k, 0.0);
    if (cell.residency == usearch_residency_nvme_k) return failures;
    failures += expect_absorbed(index, donated.view(), sizes.donor_sequences, sizes.sequences,
                                usearch_status_unsupported_k);
    return failures + expect_cleared(index);
}

std::size_t check_sequence_abi(settings_t const& settings) {
    constexpr std::array<usearch_sequence_metric_t, 4> metrics_k {
        usearch_sequence_metric_levenshtein_bytes_k, usearch_sequence_metric_levenshtein_utf8_k,
        usearch_sequence_metric_overlap_k, usearch_sequence_metric_needleman_wunsch_k};
    constexpr std::array<std::string_view, 4> metric_names_k {"levenshtein", "levenshtein-utf8", "overlap",
                                                              "needleman-wunsch"};
    std::size_t failures = 0;
    for (sweep_cell_t const& cell : diagonal_sweep(usearch_scalar_unknown_k)) {
        // The best class sweeps every metric, and a pinned class the first.
        std::size_t const metrics_count = cell.cpu_class == usearch_cpu_class_unknown_k ? metrics_k.size() : 1;
        for (std::size_t metric = 0; metric != metrics_count; ++metric) {
            std::string const name = fmt::format("sequence {} {}", name_of(cell), metric_names_k[metric]);
            if (!settings.filter.selects(name)) continue;
            session_t session = open_session(settings_of(settings, cell), cell.cpu_class);
            sequence_geometry_t const geometry {{metrics_k[metric], 0}, nullptr, {}};
            failures += report(
                name.c_str(),
                session ? check_sequence_metric(cell, session, geometry, name, stream_key(settings.seed, name)) : 1);
        }
    }
    return failures;
}

} // namespace unum::usearch::test::abi
