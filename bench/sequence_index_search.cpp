/**
 *  @file bench/sequence_index_search.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The sequence search benchmark: a text file, one entry per line, built into one RAM or
 *      VRAM index through the C interface and judged against an exact Levenshtein, overlap or
 *      Needleman-Wunsch scan.
 *
 *  A VRAM index lands its matches in a host arena, as a C caller reading them would. An NVMe run
 *  builds in RAM, saves a checkpoint to the temporary directory and searches a view of it.
 *  Needleman-Wunsch folds every byte into an alphabet's classes and packs the codes at ingest:
 *  DNA as A, C, G and T, anything else as T, under transition 1, transversion 2 and gap 2;
 *  proteins as BLOSUM62's 20 amino acids, B, Z, X and the stop, anything else as X, with gap 7,
 *  the linear gap score of 4 plus half the mean self-score; text already folded to a-z, the space,
 *  a digit as 0 and anything else as a full stop, at unit costs. Searching under uniform costs
 *  reloads a RAM or NVMe index under unit costs over the same classes, judged against them.
 *
 *  @verbatim
 *  --base-texts     required  Entries, one per line
 *  --window-widths  none      Window widths in bytes, which the overlap kind alone takes
 *  --residency      ram       Where the index lives: ram, vram on GPU zero, or nvme
 *  --metric         levenshtein, levenshtein-utf8, overlap or needleman-wunsch; levenshtein by default
 *  --alphabet       none      dna, protein or text, which Needleman-Wunsch alone takes
 *  --search-costs   alphabet  alphabet, or uniform to search a Needleman-Wunsch index under unit costs
 *  --weights        none      65,536 int8 similarities and an int8 gap cost replacing the alphabet's
 *  --query-targets  none      The base row each held-out query intends, one decimal per line
 *  @endverbatim
 */
#include <cstdint> // `std::uint8_t` - the checkpoint path's bytes, `std::int8_t` - the alphabets' scores
#include <cstdio>  // `std::remove` - the checkpoint an NVMe run saved
#include <cstring> // `std::memset` - the unit scores' blank matrix

#include <array>       // `std::array` - one round of the exact scan
#include <charconv>    // `std::from_chars` - the queries' intended rows
#include <filesystem>  // `std::filesystem::temp_directory_path` - where an NVMe run saves
#include <limits>      // `std::numeric_limits` - the exact scan's open radius
#include <string>      // `std::string` - the lines end to end
#include <string_view> // `std::string_view`
#include <type_traits> // `std::type_identity` - the exact-scan metric a kind picks
#include <vector>      // `std::vector` - one exact-scan metric per thread

#include "usearch/sequence_index.h"
#include "usearch/sequence/ram_metric.hpp"
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t`

#include "basic.cuh"

namespace unum::usearch::bench {

/** Entries per round of the exact scan. */
constexpr std::size_t oracle_round_k = 4096;

/** The proteins' classes in BLOSUM62's order: 20 amino acids, then B, Z, X and the stop. */
constexpr std::string_view protein_letters_k = "ARNDCQEGHILKMFPSTWYVBZX*";

/** BLOSUM62 over @ref protein_letters_k, as NCBI publishes it. */
constexpr std::int8_t blosum62_k[24][24] = {
    {4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4},
    {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4},
    {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4},
    {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
    {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4},
    {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
    {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4},
    {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4},
    {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4},
    {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
    {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4},
    {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4},
    {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4},
    {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4},
    {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
    {0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4},
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1},
};

/** The classes text folds into: a-z, the space, any digit, and anything else. */
constexpr std::size_t text_classes_k = 29;

/** An alphabet's byte-to-class fold, and the weights its Needleman-Wunsch index takes. */
struct alphabet_t {
    std::array<std::uint8_t, 256> codes {};
    std::vector<std::int8_t> similarities = std::vector<std::int8_t>(256 * 256);
    std::size_t classes = 0;
    std::int8_t gap = 0;

    /** The weights the C interface reads, borrowing the similarities. */
    usearch_sequence_weights_t weights() const noexcept { return {similarities.data(), gap}; }
};

/** The alphabet @p name spells, under its own costs or, when @p uniform, unit costs over its
 *  classes; no classes for a name it does not know. */
inline alphabet_t alphabet_of(std::string_view name, bool uniform) {
    alphabet_t alphabet;
    if (name == "dna") {
        alphabet.classes = 4, alphabet.gap = 2;
        alphabet.codes.fill(3);
        for (char const letter : {'A', 'C', 'G', 'T'}) {
            std::uint8_t const code = static_cast<std::uint8_t>(std::string_view {"ACGT"}.find(letter));
            alphabet.codes[static_cast<std::uint8_t>(letter)] = code;
            alphabet.codes[static_cast<std::uint8_t>(letter | 0x20)] = code;
        }
        // Purines take even codes and pyrimidines odd, so a transition keeps half the self-score.
        for (std::size_t first = 0; first != 4; ++first)
            for (std::size_t second = 0; second != 4; ++second)
                alphabet.similarities[first * 256 + second] = first == second ? 2 : (first & 1) == (second & 1) ? 1 : 0;
    }
    else if (name == "protein") {
        alphabet.classes = protein_letters_k.size(), alphabet.gap = 7;
        alphabet.codes.fill(static_cast<std::uint8_t>(protein_letters_k.find('X')));
        for (std::size_t code = 0; code != protein_letters_k.size(); ++code) {
            char const letter = protein_letters_k[code];
            alphabet.codes[static_cast<std::uint8_t>(letter)] = static_cast<std::uint8_t>(code);
            if (letter >= 'A' && letter <= 'Z')
                alphabet.codes[static_cast<std::uint8_t>(letter | 0x20)] = static_cast<std::uint8_t>(code);
        }
        for (std::size_t first = 0; first != protein_letters_k.size(); ++first)
            for (std::size_t second = 0; second != protein_letters_k.size(); ++second)
                alphabet.similarities[first * 256 + second] = blosum62_k[first][second];
    }
    else if (name == "text") {
        alphabet.classes = text_classes_k, alphabet.gap = 1;
        alphabet.codes.fill(text_classes_k - 1);
        for (char letter = 'a'; letter <= 'z'; ++letter)
            alphabet.codes[static_cast<std::uint8_t>(letter)] = static_cast<std::uint8_t>(letter - 'a');
        alphabet.codes[' '] = 26;
        for (char digit = '0'; digit <= '9'; ++digit) alphabet.codes[static_cast<std::uint8_t>(digit)] = 27;
        for (std::size_t code = 0; code != text_classes_k; ++code) alphabet.similarities[code * 256 + code] = 1;
    }
    if (uniform) {
        alphabet.gap = 1;
        std::memset(alphabet.similarities.data(), 0, alphabet.similarities.size());
        for (std::size_t code = 0; code != alphabet.classes; ++code) alphabet.similarities[code * 256 + code] = 1;
    }
    return alphabet;
}

/** Packs every entry of @p tape over @p characters as its codes under @p alphabet into
 *  @p packed, one packed row per entry. */
inline tape_t packed_codes_of(tape_t const& tape, std::string_view characters, alphabet_t const& alphabet,
                              std::string& packed) {
    u32_t const code_bits = sequence::packed_code_bits(alphabet.classes);
    tape_t packed_tape;
    packed_tape.scalar = usearch_scalar_u8_k;
    std::size_t total = 0;
    for (std::size_t row = 0; row != tape.size(); ++row)
        total += sequence::packed_codes_bytes(tape.offsets[row + 1] - tape.offsets[row], code_bits);
    packed.resize(total);
    std::vector<std::uint8_t> codes;
    for (std::size_t row = 0; row != tape.size(); ++row) {
        std::string_view const entry = characters.substr(tape.offsets[row], tape.offsets[row + 1] - tape.offsets[row]);
        codes.clear();
        for (char const character : entry) codes.push_back(alphabet.codes[static_cast<std::uint8_t>(character)]);
        std::size_t const first = packed_tape.offsets.back(),
                          bytes = sequence::packed_codes_bytes(codes.size(), code_bits);
        sequence::pack_codes(codes, code_bits, bytes_of(packed.data() + first, bytes));
        packed_tape.offsets.push_back(first + bytes);
    }
    packed_tape.elements = packed.data();
    return packed_tape;
}

/** The lines of @p text as entries end to end in @p characters, each newline dropped with a
 *  carriage return before it. */
inline tape_t lines_of(bytes_view_t text, std::string& characters) {
    tape_t tape;
    tape.scalar = usearch_scalar_u8_k;
    characters.reserve(text.size());
    for (byte_t const byte : text) {
        char const character = static_cast<char>(byte);
        if (character != '\n') {
            characters.push_back(character);
            continue;
        }
        if (characters.size() != tape.offsets.back() && characters.back() == '\r') characters.pop_back();
        tape.offsets.push_back(characters.size());
    }
    if (characters.size() != tape.offsets.back()) tape.offsets.push_back(characters.size());
    tape.elements = characters.data();
    return tape;
}

/** Fills one distance per base entry for the entry at @p query, scored by @p metric in rounds and
 *  scaled by @p unit onto the C interface's scale. */
template <typename metric_type_>
void scan_exactly(metric_type_& metric, std::string_view characters, std::span<std::size_t const> offsets,
                  std::size_t query, std::span<double> distances, double unit) {
    using distance_t = typename metric_type_::distance_t;
    auto const text_of = [&](std::size_t row) noexcept {
        return characters.substr(offsets[row], offsets[row + 1] - offsets[row]);
    };
    std::array<distance_t, oracle_round_k> round;
    metric.reset(text_of(query));
    for (std::size_t first = 0; first < distances.size(); first += oracle_round_k) {
        std::size_t const count = std::min(oracle_round_k, distances.size() - first);
        for (std::size_t row = first; row != first + count; ++row) static_cast<void>(metric.push(text_of(row)));
        metric.drain(std::span {round}.first(count), (std::numeric_limits<distance_t>::max)());
        for (std::size_t slot = 0; slot != count; ++slot)
            distances[first + slot] = static_cast<double>(round[slot]) * unit;
    }
}

/** Runs the sequence search benchmark over the flags in @p argv. */
int bench_sequence_index_search(int argc, char const* const* argv) {
    basic_settings_t settings;
    std::string_view base_texts_path, residency = "ram", alphabet_name, search_costs = "alphabet";
    std::string_view weights_path, query_targets_path;
    counts_t window_widths {.minimum_size = 0};
    std::array<flag_t, 7> const flags {{
        {"--base-texts", "path", "entries, one per line", &base_texts_path},
        {"--window-widths", "bytes", "window widths for the overlap kind, like 3,5", &window_widths},
        {"--residency", "name", "where the index lives: ram, vram on GPU zero, or nvme; ram by default", &residency},
        {"--alphabet", "name", "dna, protein or text, which the needleman-wunsch kind alone takes", &alphabet_name},
        {"--search-costs", "name", "alphabet, or uniform to search a needleman-wunsch index under unit costs",
         &search_costs},
        {"--weights", "path", "256x256 int8 similarities and an int8 gap cost replacing the alphabet's", &weights_path},
        {"--query-targets", "path", "the base row each query intends, one per line, judged at rank 1 and top-k",
         &query_targets_path},
    }};
    read_basic_settings(argc, argv, "usearch_bench_sequence_index_search --base-texts <path> [flags]", settings, flags);

    std::string_view const metric_name = settings.metric_name.empty() ? "levenshtein" : settings.metric_name;
    usearch_sequence_metric_t const metric = metric_name == "levenshtein" ? usearch_sequence_metric_levenshtein_bytes_k
                                             : metric_name == "levenshtein-utf8"
                                                 ? usearch_sequence_metric_levenshtein_utf8_k
                                             : metric_name == "overlap" ? usearch_sequence_metric_overlap_k
                                             : metric_name == "needleman-wunsch"
                                                 ? usearch_sequence_metric_needleman_wunsch_k
                                                 : usearch_sequence_metric_unknown_k;
    if (metric == usearch_sequence_metric_unknown_k)
        return fmt::println(
                   stderr,
                   "--metric=\"{}\" does not parse, expected levenshtein, levenshtein-utf8, overlap, needleman-wunsch",
                   metric_name),
               1;
    if (residency != "ram" && residency != "vram" && residency != "nvme")
        return fmt::println(stderr, "--residency=\"{}\" does not parse, expected ram, vram or nvme", residency), 1;
    bool const aligned = metric == usearch_sequence_metric_needleman_wunsch_k;
    alphabet_t alphabet = alphabet_of(alphabet_name, false);
    alphabet_t searched_alphabet = alphabet_of(alphabet_name, search_costs == "uniform");
    if (!weights_path.empty()) {
        auto [weights_file, weights_status] = mapped_file_t::read(weights_path);
        if (failed(weights_status) || weights_file.view().size() != alphabet.similarities.size() + 1 || !aligned)
            return fmt::println(stderr, "--weights=\"{}\" must hold 65,537 bytes for needleman-wunsch", weights_path),
                   1;
        std::memcpy(alphabet.similarities.data(), weights_file.view().data(), alphabet.similarities.size());
        std::memcpy(&alphabet.gap, weights_file.view().data() + alphabet.similarities.size(), 1);
        if (search_costs != "uniform") searched_alphabet = alphabet;
    }
    if (aligned != (alphabet.classes != 0))
        return fmt::println(stderr, "--alphabet=\"{}\" must be dna, protein or text for needleman-wunsch alone",
                            alphabet_name),
               1;
    if (search_costs != "alphabet" && (search_costs != "uniform" || !aligned || residency == "vram"))
        return fmt::println(
                   stderr, "--search-costs=\"{}\" must be alphabet, or uniform on a RAM or NVMe needleman-wunsch index",
                   search_costs),
               1;
    bool const reloaded = search_costs == "uniform";
    usearch_sequence_weights_t const weights = alphabet.weights(), searched_weights = searched_alphabet.weights();
    usearch_vram_options_t const vram {};
    auto [file, read_status] = mapped_file_t::read(base_texts_path);
    if (failed(read_status)) return fmt::println(stderr, "--base-texts=\"{}\" could not be read", base_texts_path), 1;
    std::string characters, packed;
    tape_t tape = lines_of(file.view(), characters);
    // The index and the exact scan read the packed codes alone.
    if (aligned) tape = packed_codes_of(tape, characters, alphabet, packed), std::string {}.swap(characters);
    std::string_view const entries = aligned ? std::string_view {packed} : std::string_view {characters};
    // The base row each held-out query intends, judged by key at rank 1 and among the top-k.
    std::vector<std::size_t> query_targets;
    if (!query_targets_path.empty()) {
        auto [targets_file, targets_status] = mapped_file_t::read(query_targets_path);
        if (failed(targets_status))
            return fmt::println(stderr, "--query-targets=\"{}\" could not be read", query_targets_path), 1;
        std::string targets_text;
        tape_t const targets = lines_of(targets_file.view(), targets_text);
        for (std::size_t row = 0; row != targets.size(); ++row) {
            std::size_t target = 0;
            std::from_chars(targets_text.data() + targets.offsets[row], targets_text.data() + targets.offsets[row + 1],
                            target);
            query_targets.push_back(target);
        }
        if (query_targets.size() != settings.queries.value)
            return fmt::println(stderr, "--query-targets holds {} rows for {} queries", query_targets.size(),
                                settings.queries.value),
                   1;
    }
    std::size_t task_queries = 0, task_first = 0, task_kept = 0;

    usearch_outcome_t outcome {};
    usearch_sequence_index_t index = nullptr;
    if (residency == "vram") {
        usearch_sequence_init_vram_t init {.index = &index,
                                           .outcome = &outcome,
                                           .metric = metric,
                                           .weights = aligned ? &weights : nullptr,
                                           .window_widths = window_widths.values.data(),
                                           .window_widths_count = window_widths.size,
                                           .connectivity = settings.connectivity.value,
                                           .vram = vram};
        usearch_sequence_init_vram(&init);
    }
    else {
        usearch_sequence_init_ram_t init {.index = &index,
                                          .outcome = &outcome,
                                          .metric = metric,
                                          .weights = aligned ? &weights : nullptr,
                                          .window_widths = window_widths.values.data(),
                                          .window_widths_count = window_widths.size,
                                          .connectivity = settings.connectivity.value};
        usearch_sequence_init_ram(&init);
    }
    if (outcome.message) return fmt::println(stderr, "Init: {}", outcome.message), 1;
    usearch_sequence_index_description_t description {};
    usearch_sequence_describe_t describe {.index = index, .outcome = &outcome, .description = &description};
    usearch_sequence_describe(&describe);
    fmt::println("- Metric: {} · residency: {} · class: {}", metric_name, residency, description.capabilities);
    if (aligned)
        fmt::println("- Alphabet: {} · {} classes · {} bytes packed · searched under {} costs", alphabet_name,
                     alphabet.classes, humanized(entries.size()), search_costs);

    auto const insert = [&](usearch_executor_t executor, usearch_keys_view_t keys, usearch_tape_view_t entries,
                            std::size_t count, usearch_traversal_stats_t* stats, usearch_outcome_t* reached) {
        usearch_sequence_insert_t arguments {.index = index,
                                             .executor = executor,
                                             .outcome = reached,
                                             .count = count,
                                             .keys = keys,
                                             .sequences = entries,
                                             .expansion_factor = settings.expansion_add.value,
                                             .stats = stats};
        usearch_sequence_insert(&arguments);
    };
    auto const search = [&](usearch_executor_t executor, usearch_arena_t arena, usearch_tape_view_t queries,
                            std::size_t count, std::size_t expansion, usearch_matches_view_t* matches,
                            usearch_traversal_stats_t* stats, usearch_outcome_t* reached) {
        usearch_sequence_search_t arguments {.index = index,
                                             .executor = executor,
                                             .outcome = reached,
                                             .count = count,
                                             .queries = queries,
                                             .matches = matches,
                                             .arena = arena,
                                             .wanted = settings.top_k.value,
                                             .expansion_factor = expansion,
                                             .stats = stats};
        usearch_sequence_search(&arguments);
        if (query_targets.empty() || reached->message) return;
        std::size_t const first = static_cast<std::size_t>(queries.offsets - tape.offsets.data());
        std::size_t const query_first = tape.size() - query_targets.size();
        for (std::size_t query = 0; query != count; ++query) {
            usearch_key_t const target = key_of(query_targets[first - query_first + query]);
            usearch_key_t const* const found = matches->keys + query * matches->columns;
            for (std::size_t rank = 0; rank != matches->counts[query]; ++rank)
                if (std::memcmp(&found[rank], &target, sizeof(usearch_key_t)) == 0)
                    task_first += rank == 0, task_kept += 1;
        }
        task_queries += count;
        if (task_queries != query_targets.size()) return;
        fmt::println("- Task at expansion {}: intended entry at rank 1 {:.4f} · in the top {} {:.4f} over {} queries",
                     expansion, static_cast<double>(task_first) / task_queries, settings.top_k.value,
                     static_cast<double>(task_kept) / task_queries, task_queries);
        task_queries = task_first = task_kept = 0;
    };
    // An NVMe run saves what it built and searches a view of the checkpoint, on storage threads; a
    // search under other costs reloads the checkpoint under them.
    std::string const checkpoint =
        (std::filesystem::temp_directory_path() / "usearch-bench-sequence.safegraphs").string();
    usearch_bytes_view_t const checkpoint_view {reinterpret_cast<std::uint8_t const*>(checkpoint.data()),
                                                checkpoint.size()};
    usearch_executor_t storage = nullptr;
    auto const serve = [&](usearch_executor_t executor, usearch_outcome_t* reached) {
        if (residency != "nvme" && !reloaded) return executor;
        usearch_sequence_save_t save {.index = index,
                                      .executor = executor,
                                      .outcome = reached,
                                      .path = checkpoint_view};
        usearch_sequence_save(&save);
        if (reached->message) return executor;
        usearch_sequence_free(index);
        index = nullptr;
        if (residency == "ram") {
            usearch_sequence_load_ram_t load {.index = &index,
                                              .executor = executor,
                                              .outcome = reached,
                                              .path = checkpoint_view,
                                              .weights = &searched_weights};
            usearch_sequence_load_ram(&load);
            return executor;
        }
        usearch_executor_for_nvme_t storage_arguments {.executor = &storage,
                                                       .outcome = reached,
                                                       .threads = settings.threads.value};
        usearch_executor_for_nvme(&storage_arguments);
        if (reached->message) return executor;
        usearch_sequence_view_nvme_t view {.index = &index,
                                           .executor = storage,
                                           .outcome = reached,
                                           .path = checkpoint_view,
                                           .weights = aligned ? &searched_weights : nullptr};
        usearch_sequence_view_nvme(&view);
        return storage;
    };
    // One exact-scan metric per thread, so the judged queries spread over every thread.
    auto const run_with = [&]<typename oracle_type_>(std::type_identity<oracle_type_>, auto&& prepare, double unit) {
        std::vector<oracle_type_> oracles(settings.threads.value);
        for (oracle_type_& oracle : oracles)
            if (failed(prepare(oracle))) return fmt::println(stderr, "The exact scan could not size its rounds"), 1;
        return run_basic_bench<usearch_traversal_stats_t>(
            settings, tape, residency == "vram" ? &vram : nullptr, insert, search,
            [&](std::size_t thread, std::size_t query, std::span<double> distances) {
                scan_exactly(oracles[thread], entries, tape.offsets, query, distances, unit);
            },
            serve);
    };

    int exit_code = 0;
    if (aligned) {
        auto const [costs, fold_status] = sequence::needleman_wunsch_costs_t::fold(searched_alphabet.similarities,
                                                                                   searched_alphabet.gap);
        if (failed(fold_status)) return fmt::println(stderr, "The {} alphabet folds into no costs", alphabet_name), 1;
        exit_code = run_with(
            std::type_identity<sequence::needleman_wunsch_metric<>> {},
            [&](sequence::needleman_wunsch_metric<>& oracle) { return oracle.init(oracle_round_k, costs); }, 1);
    }
    else if (metric == usearch_sequence_metric_overlap_k)
        exit_code = run_with(
            std::type_identity<sequence::overlap_metric<>> {},
            [&](sequence::overlap_metric<>& oracle) { return oracle.init(oracle_round_k, window_widths.view()); },
            1.0 / sequence::overlap_resolution_k);
    else if (metric == usearch_sequence_metric_levenshtein_utf8_k)
        exit_code = run_with(
            std::type_identity<sequence::levenshtein_metric<
                sequence::levenshtein_ladder<sz_levenshtein_runes_k, sz_cap_any_k>::kit_t>> {},
            [](auto& oracle) { return oracle.init(oracle_round_k); }, 1);
    else
        exit_code = run_with(
            std::type_identity<sequence::levenshtein_metric<>> {},
            [](auto& oracle) { return oracle.init(oracle_round_k); }, 1);
    usearch_sequence_free(index);
    usearch_executor_free(storage);
    if (residency == "nvme" || reloaded) std::remove(checkpoint.c_str());
    return exit_code;
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_sequence_index_search(argc, argv); }
