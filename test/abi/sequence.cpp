/**
 *  @file test/abi/sequence.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief The sequence suite over the C interface, swept on the diagonal: every metric at the best
 *      class and a protein alignment at every admitted class, over each compiled residency at one
 *      shard and at three.
 *
 *  Sequences are planted templates over the nucleotides with an eighth of their symbols edited, so
 *  neighbors cluster by template while the bytes differ; every metric takes the same bytes, which
 *  are proteins, DNA or text alike. Each cell checks the verbs a caller reaches: the sequences land
 *  under their keys across the shards and each counts once and reads back, a walk keeps most of the
 *  exact scan's top ten, a RAM index refines, through a plan too, and gathers a second source
 *  through a reshard plan while a GPU one refuses all three, an audit reaches every entry, cursors
 *  walk every entry and a sample draws distinct ones, the index saves into memory and back and into
 *  every residency and as a view byte for byte, an NVMe view answers, audits, walks and samples and
 *  refuses every write, a RAM index reshards into two targets through a plan, every key in one, a
 *  cleared index holds nothing and ends a cursor opened before it. A text is found and read back
 *  folded whatever its case, costs no metric takes are refused, and every metric's name reads back
 *  while the names it replaced are refused.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstring> // `std::memcpy` - the header's length

#include <array>   // `std::array` - the templates, the costs
#include <string>  // `std::string` - the strings, the cells' names
#include <utility> // `std::pair` - a metric beside its spelling
#include <vector>  // `std::vector` - the tape's offsets, the similarities

#include "usearch/sequence_index.h"

#include "harness.hpp"

namespace unum::usearch::test::abi {

/** Sequences a cell builds, the sequences a second source holds, and the nucleotides per template;
 *  the queries held out past them number @c recall_samples_k. */
struct sequence_sizes_t {
    std::size_t sequences = 0, packed_sequences = 0, length = 0;
};

/** The sizes of the bit-parallel metrics' cells, the smaller ones of alignment with costs, whose
 *  every distance fills a whole alignment matrix, and the smallest of an NVMe alignment on the
 *  serial class, whose seeding walks sketch every sequence they fetch with the serial kit. */
static constexpr sequence_sizes_t sequence_sizes_k {1024, 256, 100}, aligned_sizes_k {256, 64, 32},
    serial_nvme_aligned_sizes_k {64, 16, 32};

/** The breadth every walk takes. */
static constexpr std::size_t sequence_expansion_k = 128;

/** The nucleotides, whose position is their code under alignment with costs. */
static constexpr std::string_view nucleotides_k = "ACGT";

/** @p count sequences of about @p length nucleotides from eight templates, an eighth of every
 *  sequence's symbols substituted, inserted or erased. */
static std::vector<std::string> generate_nucleotides(std::size_t count, std::size_t length, util::u64_t seed) {
    constexpr std::size_t templates_k = 8;
    split_mix_generator_t random_generator {seed};
    util::uniform_below<util::u32_t> const symbol_distribution {static_cast<util::u32_t>(nucleotides_k.size())};
    util::uniform_below<util::u32_t> const template_distribution {templates_k};
    util::uniform_below<util::u32_t> const edit_kind_distribution {3};
    std::array<std::string, templates_k> templates;
    for (std::string& symbols : templates)
        for (std::size_t symbol = 0; symbol != length; ++symbol)
            symbols += nucleotides_k[symbol_distribution(random_generator)];
    std::vector<std::string> strings(count);
    for (std::string& symbols : strings) {
        symbols = templates[template_distribution(random_generator)];
        for (std::size_t edit = 0; edit != length / 8; ++edit) {
            std::size_t const position = util::uniform_below<util::u32_t> {static_cast<util::u32_t>(symbols.size())}(
                random_generator);
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

/** Sequences end to end and the offsets around them, as the C interface takes a tape. */
struct sequence_tape_t {
    std::string elements;
    std::vector<std::size_t> offsets {0};

    usearch_tape_view_t view() const noexcept { return tape_of(elements.data(), offsets.data(), usearch_scalar_u8_k); }
};

static sequence_tape_t generate_sequences(std::vector<std::string> const& strings) {
    sequence_tape_t tape;
    for (std::string const& symbols : strings) tape.elements += symbols, tape.offsets.push_back(tape.elements.size());
    return tape;
}

/** Every key of the first @p count entries reads one sequence back, on host residencies. */
static void expect_gets(handle<sequence_verbs_t>& index, std::size_t count) {
    if (index.residency == usearch_residency_vram_k) return;
    std::size_t misread = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_outcome_t outcome {};
        usearch_tape_view_t sequences {};
        std::size_t written = 0;
        usearch_sequence_get_t get {};
        get.index = index.get(), get.executor = index.executor(), get.outcome = &outcome;
        get.key = key_of_offset(index.session->seed, offset);
        get.sequences = &sequences, get.written = &written, get.arena = index.arena();
        usearch_sequence_get(&get);
        misread += outcome.status != usearch_status_success_k || written != 1;
    }
    expect_true(misread == 0);
}

/** The three metrics spell "protein-alignment", "dna-overlap" and "utf8-containment" both ways, and
 *  the names they replaced spell none. */
static void test_sequence_metric_spellings() {
    constexpr std::array<std::pair<usearch_sequence_metric_t, std::string_view>, 3> spellings_k {{
        {usearch_sequence_metric_protein_alignment_k, "protein-alignment"},
        {usearch_sequence_metric_dna_overlap_k, "dna-overlap"},
        {usearch_sequence_metric_utf8_containment_k, "utf8-containment"},
    }};
    for (auto const& [metric, name] : spellings_k) {
        usearch_sequence_metric_t parsed = usearch_sequence_metric_unknown_k;
        expect_true(std::string_view {usearch_sequence_metric_name(metric)} == name);
        expect_true(usearch_sequence_metric_from_name(bytes_view(name), &parsed) == usearch_status_success_k);
        expect_true(parsed == metric);
    }
    for (std::string_view const retired : {"alignment", "containment", "overlap", "needleman-wunsch"}) {
        usearch_sequence_metric_t parsed = usearch_sequence_metric_unknown_k;
        expect_true(usearch_sequence_metric_from_name(bytes_view(retired), &parsed) ==
                    usearch_status_invalid_argument_k);
    }
}

/** One metric of one cell, as the file's docblock describes. */
static void test_sequence_metric(sweep_cell_t const& cell, sequence_geometry_t geometry, util::u64_t seed) {
    session_t session = open_session(settings_of(session_settings(), cell), cell.cpu_class);
    if (!expect_true(bool(session))) return;
    bool const aligned = geometry.metadata.metric == usearch_sequence_metric_protein_alignment_k;
    handle<sequence_verbs_t> index = open_index<sequence_verbs_t>(session, geometry, cell.shards);
    if (!expect_true(bool(index))) return;
    bool const serial_nvme = cell.residency == usearch_residency_nvme_k && cell.cpu_class == usearch_cpu_class_serial_k;
    sequence_sizes_t const sizes = !aligned      ? sequence_sizes_k
                                   : serial_nvme ? serial_nvme_aligned_sizes_k
                                                 : aligned_sizes_k;
    sequence_tape_t const sequences = generate_sequences(
        generate_nucleotides(sizes.sequences + recall_samples_k, sizes.length, seed));
    sequence_tape_t const second_source = generate_sequences(
        generate_nucleotides(sizes.packed_sequences, sizes.length, ~seed));
    usearch_tape_view_t const queries = sequence_verbs_t::slice(sequences.view(), sizes.sequences);

    if (!check_built(cell, index, sequences.view(), sizes.sequences)) return;
    expect_gets(index, sizes.sequences);
    expect_walk_recall(index, queries, recall_samples_k, sequence_expansion_k, 0.8);
    expect_refined(index, queries, recall_samples_k, sequence_expansion_k, usearch_status_success_k);
    expect_health(index, sizes.sequences);
    expect_cursor(index, sizes.sequences);
    expect_sample(index, sizes.sequences);
    expect_buffer_round_trip(index, queries, recall_samples_k, sequence_expansion_k, 0.8, usearch_status_unsupported_k);
    scratch_directory_t const scratch;
    std::string const stem = scratch.path("checkpoint");
    expect_widened(index, stem + "-widened");
    expect_persistence(cell, index, sequences.view(), queries, recall_samples_k, sequence_expansion_k, 0.8, stem,
                       saved_bytes_t::identical_k, 0.0);
    if (cell.residency == usearch_residency_nvme_k) return;
    expect_refine_plan(index, sequences.view(), sizes.sequences, usearch_status_success_k);
    expect_reshard_gather(index, second_source.view(), sizes.packed_sequences, sizes.sequences,
                          usearch_status_success_k);
    expect_reshard_cut(index, usearch_status_success_k);
    expect_cleared(index, sequences.view());
}

/** A containment finds a text stored in another case, whose folding lengthens it, and reads it
 *  back folded, an ill-formed byte kept as it was. */
static void test_sequence_folded_texts(sweep_cell_t const& cell) {
    session_t session = open_session(settings_of(session_settings(), cell), cell.cpu_class);
    if (!expect_true(bool(session))) return;
    sequence_geometry_t const geometry {{usearch_sequence_metric_utf8_containment_k, 0}, nullptr};
    handle<sequence_verbs_t> index = open_index<sequence_verbs_t>(session, geometry, cell.shards);
    if (!expect_true(bool(index))) return;
    // Sharp s doubles in length, dotted capital I grows, and the last byte is no UTF-8.
    std::vector<std::string> const texts {"STRASSENWEG 12 \xC3\x9F", "\xC4\xB0STANBUL BO\xC4\x9E\x41ZI",
                                          "\xCE\x91\xCE\x98\xCE\x97\xCE\x9D\xCE\x91 is far from here\xFF"};
    std::vector<std::string> const queries {"strassenweg 12 ss", "i\xCC\x87stanbul bo\xC4\x9F\x61zi",
                                            "\xCE\xB1\xCE\xB8\xCE\xB7\xCE\xBD\xCE\xB1 IS FAR FROM HERE"};
    sequence_tape_t const stored = generate_sequences(texts), asked = generate_sequences(queries);
    usearch_traversal_stats_t stats {};
    if (!expect_success(insert_entries(index, stored.view(), texts.size(), 0, cell_batch_k, 0, stats))) return;
    expect_gets(index, texts.size());
    usearch_outcome_t outcome {};
    usearch_tape_view_t read {};
    std::size_t written = 0;
    usearch_sequence_get_t get {};
    get.index = index.get(), get.executor = index.executor(), get.outcome = &outcome;
    get.key = key_of_offset(index.session->seed, 0);
    get.sequences = &read, get.written = &written, get.arena = index.arena();
    if (index.residency != usearch_residency_vram_k) {
        usearch_sequence_get(&get);
        if (!expect_success(outcome) || !expect_true(written == 1)) return;
        std::string_view const folded {static_cast<char const*>(read.data), tape_offset(read, 1)};
        expect_true(folded == "strassenweg 12 ss");
    }
    copied_matches_t found;
    usearch_traversal_stats_t walked {};
    if (!expect_success(search_queries(index, asked.view(), queries.size(), cell_batch_k, cell_wanted_k,
                                       sequence_expansion_k, usearch_traversal_walk_k, walked, found)))
        return;
    st::vector<std::size_t> const offsets = match_offsets(found);
    for (std::size_t query = 0; query != queries.size(); ++query)
        expect_true(offset_of_key(index.session->seed, std::bit_cast<usearch_key_t>(found.keys[offsets[query]])) ==
                    query);
}

/** Costs no metric folds are refused: any for a containment, none without similarities, and for a
 *  DNA overlap costs outside the four letters' classes or splitting one across its complement; so
 *  are sketch bytes off a multiple of four, and any for a DNA overlap, which keeps no sketch. */
static void test_sequence_costs_refused() {
    session_t session = open_session(session_settings(), usearch_cpu_class_unknown_k);
    if (!expect_true(bool(session))) return;
    std::vector<std::int8_t> five_letters(256 * 256), twinned(256 * 256);
    for (std::size_t byte = 0; byte != 256; ++byte) five_letters[byte * 257] = 1, twinned[byte * 257] = 1;
    // The letters A, C, G and T keep classes of their own beside the ones of every other byte.
    for (std::size_t byte = 0; byte != 256; ++byte) five_letters[byte * 257] = 2;
    // One class for A and C splits their complements T and G.
    for (std::size_t other = 0; other != 256; ++other) {
        twinned['C' * 256 + other] = twinned['A' * 256 + other];
        twinned[other * 256 + 'C'] = twinned[other * 256 + 'A'];
    }
    usearch_sequence_costs_t const any {five_letters.data(), 1}, split {twinned.data(), 1}, bare {nullptr, 1};
    struct refusal_t {
        usearch_sequence_metric_t metric;
        usearch_sequence_costs_t const* costs;
        std::size_t sketch_bytes;
        usearch_status_t status;
    };
    std::array<refusal_t, 6> const refusals {{
        {usearch_sequence_metric_utf8_containment_k, &any, 0, usearch_status_config_mismatch_k},
        {usearch_sequence_metric_protein_alignment_k, &bare, 0, usearch_status_invalid_argument_k},
        {usearch_sequence_metric_dna_overlap_k, &any, 0, usearch_status_config_mismatch_k},
        {usearch_sequence_metric_dna_overlap_k, &split, 0, usearch_status_config_mismatch_k},
        {usearch_sequence_metric_dna_overlap_k, nullptr, 64, usearch_status_config_mismatch_k},
        {usearch_sequence_metric_protein_alignment_k, nullptr, 66, usearch_status_config_mismatch_k},
    }};
    for (refusal_t const& refusal : refusals) {
        usearch_sequence_index_t index = nullptr;
        usearch_outcome_t outcome {};
        usearch_sequence_init_ram_t arguments {};
        arguments.index = &index, arguments.outcome = &outcome, arguments.metric = refusal.metric;
        arguments.costs = refusal.costs, arguments.sketch_bytes = refusal.sketch_bytes;
        usearch_sequence_init_ram(&arguments);
        expect_status(outcome, refusal.status);
        expect_true(index == nullptr);
    }
}

/** A checkpoint recording its classes twice is refused as malformed by a metadata read and by a
 *  load, where the same bytes with the second copy under a name no reader knows read back. */
static void test_sequence_metric_repeated_field(sweep_cell_t const& cell, sequence_geometry_t geometry,
                                                util::u64_t seed) {
    constexpr std::size_t count_k = 16, length_k = 64;
    session_t session = open_session(settings_of(session_settings(), cell), cell.cpu_class);
    if (!expect_true(bool(session))) return;
    handle<sequence_verbs_t> index = open_index<sequence_verbs_t>(session, geometry, cell.shards);
    if (!expect_true(bool(index))) return;
    sequence_tape_t const sequences = generate_sequences(generate_nucleotides(count_k, length_k, seed));
    usearch_traversal_stats_t stats {};
    if (!expect_success(insert_entries(index, sequences.view(), count_k, 0, cell_batch_k, 0, stats))) return;
    usearch_outcome_t outcome {};
    usearch_bytes_view_t monolith {};
    sequence_verbs_t::save_buffer(index.get(), index.executor(), index.arena(), &monolith, &outcome);
    if (!expect_success(outcome)) return;
    std::string saved(monolith.length, '\0');
    expect_true(download(saved.data(), monolith.data, saved.size()));

    // The classes' entry of the header, from its key to the quote closing its value.
    constexpr std::string_view key_k = "\"byte_classes\"";
    constexpr std::size_t length_at_k = 8, padded_to_k = 4096;
    std::size_t const first = saved.find(key_k);
    if (!expect_true(first != std::string::npos)) return;
    std::size_t const value = saved.find('"', first + key_k.size());
    std::string entry = saved.substr(first, saved.find('"', value + 1) + 1 - first) + ',';
    // Spaces keep the bytes behind the header on the alignment the header declares.
    entry.resize(util::round_up_to_multiple(entry.size(), padded_to_k), ' ');
    util::u64_t header_length = 0;
    std::memcpy(&header_length, saved.data() + length_at_k, sizeof(header_length));
    header_length += entry.size();
    std::string repeated = saved.substr(0, first) + entry + saved.substr(first);
    std::memcpy(repeated.data() + length_at_k, &header_length, sizeof(header_length));
    std::string renamed = repeated;
    renamed[first + key_k.size() - 2] = 'x';

    usearch_sequence_metadata_t metadata {};
    sequence_verbs_t::metadata_read_buffer(session.executor, bytes_view(renamed), &metadata, &outcome);
    expect_success(outcome);
    expect_true(metadata.metric == geometry.metadata.metric);
    sequence_verbs_t::metadata_read_buffer(session.executor, bytes_view(repeated), &metadata, &outcome);
    expect_status(outcome, usearch_status_malformed_k);
    st::vector<std::int32_t> placement;
    usearch_sequence_index_t copy = nullptr;
    sequence_verbs_t::load_ram_buffer(&copy, session.executor_for(usearch_residency_ram_k), bytes_view(repeated),
                                      session.ram_options(0, placement), geometry, &outcome);
    expect_status(outcome, usearch_status_malformed_k);
    expect_true(copy == nullptr);
}

/** A VRAM index's exact scan under every metric scores as the same index reloaded on RAM scans
 *  it. */
static void test_sequence_device_scan(sweep_cell_t const& cell, util::u64_t seed) {
    constexpr std::size_t sequences_k = 256, queries_k = 16, length_k = 64;
    session_t session = open_session(settings_of(session_settings(), cell), cell.cpu_class);
    if (!expect_true(bool(session))) return;
    sequence_tape_t const sequences = generate_sequences(generate_nucleotides(sequences_k + queries_k, length_k, seed));
    usearch_tape_view_t const queries = sequence_verbs_t::slice(sequences.view(), sequences_k);
    scratch_directory_t const scratch;
    for (usearch_sequence_metric_t const metric :
         {usearch_sequence_metric_protein_alignment_k, usearch_sequence_metric_dna_overlap_k,
          usearch_sequence_metric_utf8_containment_k}) {
        handle<sequence_verbs_t> index = open_index<sequence_verbs_t>(
            session, sequence_geometry_t {{metric, 0}, nullptr}, cell.shards);
        if (!expect_true(bool(index))) return;
        usearch_traversal_stats_t stats {};
        if (!expect_success(insert_entries(index, sequences.view(), sequences_k, 0, cell_batch_k, 0, stats))) return;
        expect_device_scan(index, queries, queries_k, scratch.path(usearch_sequence_metric_name(metric)), 0.0);
    }
}

void run_sequence_abi_cases() {
    constexpr std::array<usearch_sequence_metric_t, 3> metrics_k {usearch_sequence_metric_protein_alignment_k,
                                                                  usearch_sequence_metric_dna_overlap_k,
                                                                  usearch_sequence_metric_utf8_containment_k};
    constexpr std::array<std::string_view, 3> metric_constants_k {"usearch_sequence_metric_protein_alignment_k",
                                                                  "usearch_sequence_metric_dna_overlap_k",
                                                                  "usearch_sequence_metric_utf8_containment_k"};
    constexpr std::array<std::string_view, 3> metric_names_k {"protein-alignment", "dna-overlap", "utf8-containment"};
    run_test_case("test_metric_names<sequence_verbs_t>", test_metric_names<sequence_verbs_t>);
    run_test_case("test_sequence_metric_spellings", test_sequence_metric_spellings);
    run_test_case("test_sequence_costs_refused", test_sequence_costs_refused);
    for (sweep_cell_t const& cell : diagonal_sweep(usearch_scalar_unknown_k)) {
        // The best class sweeps every metric, and a class set as the ceiling the first.
        std::size_t const metrics_count = cell.cpu_class == usearch_cpu_class_unknown_k ? metrics_k.size() : 1;
        for (std::size_t metric = 0; metric != metrics_count; ++metric) {
            std::string const name = fmt::format("sequence {} {}", name_of(cell), metric_names_k[metric]);
            sequence_geometry_t const geometry {{metrics_k[metric], 0}, nullptr};
            std::string const label = fmt::format("test_sequence_metric({}, {}, {}, {})", constant_of(cell.residency),
                                                  constant_of(cell.cpu_class), cell.shards, metric_constants_k[metric]);
            run_test_case(label.c_str(), test_sequence_metric, cell, geometry,
                          stream_key(test_run.settings.seed, name));
        }
        if (cell.cpu_class != usearch_cpu_class_unknown_k) continue;
        std::string const folded_label = fmt::format("test_sequence_folded_texts({}, {}, {})",
                                                     constant_of(cell.residency), constant_of(cell.cpu_class),
                                                     cell.shards);
        run_test_case(folded_label.c_str(), test_sequence_folded_texts, cell);
        if (cell.residency == usearch_residency_vram_k) {
            std::string const scan_label = fmt::format("test_sequence_device_scan({}, {})", constant_of(cell.cpu_class),
                                                       cell.shards);
            run_test_case(scan_label.c_str(), test_sequence_device_scan, cell,
                          stream_key(test_run.settings.seed, fmt::format("sequence {} device-scan", name_of(cell))));
        }
        // One host cell tampers with the header a checkpoint under costs records.
        if (cell.residency != usearch_residency_ram_k || cell.shards != 1) continue;
        std::string const label = fmt::format("test_sequence_metric_repeated_field({}, {}, {})",
                                              constant_of(cell.residency), constant_of(cell.cpu_class), cell.shards);
        run_test_case(label.c_str(), test_sequence_metric_repeated_field, cell,
                      sequence_geometry_t {{usearch_sequence_metric_dna_overlap_k, 0}, nullptr},
                      stream_key(test_run.settings.seed, fmt::format("sequence {} repeated-field", name_of(cell))));
    }
}

} // namespace unum::usearch::test::abi
