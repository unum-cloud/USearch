/**
 *  @file bench/sequence_index_search.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The sequence search benchmark: a text file, one entry per line, built into one index of
 *      any residency through the C interface and judged against an exact Levenshtein, overlap or
 *      Needleman-Wunsch scan.
 *
 *  Queries are the file's last entries, kept out of the index, and a hit counts when its distance
 *  is within the exact k-th, so integer edit distances, where many entries tie, are not
 *  undercounted. An NVMe run builds in RAM, saves a checkpoint and searches a view of it.
 *  Needleman-Wunsch folds every byte into an alphabet's classes and packs the codes at ingest: DNA
 *  as A, C, G and T, anything else as T, under transition 1, transversion 2 and gap 2; proteins as
 *  BLOSUM62's 20 amino acids, B, Z, X and the stop, anything else as X, with gap 7, the linear gap
 *  score of 4 plus half the mean self-score; text already folded to a-z, the space, a digit as 0
 *  and anything else as a full stop, at unit costs. Searching under uniform costs reloads a RAM or
 *  NVMe index under unit costs over the same classes, judged against them.
 *
 *  @verbatim
 *  Flag             Default   Meaning
 *  --base-texts     required  Entries, one per line
 *  --window-widths  none      Window widths in bytes, which the overlap kind alone takes
 *  --metric         levenshtein  levenshtein, levenshtein-utf8, overlap or needleman-wunsch
 *  --alphabet       none      dna, protein or text, which Needleman-Wunsch alone takes
 *  --search-costs   alphabet  alphabet, or uniform to search a Needleman-Wunsch index under unit costs
 *  --weights        none      65,536 int8 similarities and an int8 gap cost replacing the alphabet's
 *  --classes        none      256 bytes, each byte's class code, replacing the alphabet's fold
 *  --query-targets  none      The base row each held-out query intends, one decimal per line
 *  --matches-output none      A file per search breadth, <path>.<breadth>, each query's matched base rows on a line
 *  @endverbatim
 */
#include <cstdint> // `std::uint8_t` - the codes, `std::int8_t` - the alphabets' scores
#include <cstdio>  // `std::remove` - the reloaded checkpoint, `std::fopen` - the matches output
#include <cstring> // `std::memset`, `std::memcpy` - the unit scores' blank matrix, the weights

#include <array>       // `std::array` - the flags, the byte-to-class fold
#include <charconv>    // `std::from_chars` - the queries' intended rows
#include <filesystem>  // `std::filesystem::temp_directory_path` - where a reloaded run saves
#include <string>      // `std::string` - the matches output, the reloaded checkpoint
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector` - the offsets, the similarities

#include "usearch/sequence_index.h"
#include "usearch/sequence/ram_metric.hpp"      // `pack_codes`, `packed_code_bits`, `packed_codes_bytes`
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t`

#include "abi.hpp"

namespace unum::usearch::bench {
using namespace unum::usearch::tools;

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

/** Packs every entry of @p tape as its codes under @p alphabet, one packed row per entry. */
inline byte_tape_t packed_codes_of(byte_tape_t const& tape, alphabet_t const& alphabet) {
    u32_t const code_bits = sequence::packed_code_bits(alphabet.classes);
    byte_tape_t packed;
    std::size_t total = 0;
    for (std::size_t row = 0; row != tape.size(); ++row)
        total += sequence::packed_codes_bytes(tape.offsets[row + 1] - tape.offsets[row], code_bits);
    packed.bytes.resize(total);
    std::vector<std::uint8_t> codes;
    for (std::size_t row = 0; row != tape.size(); ++row) {
        std::string_view const entry = std::string_view {tape.bytes}.substr(tape.offsets[row],
                                                                            tape.offsets[row + 1] - tape.offsets[row]);
        codes.clear();
        for (char const character : entry) codes.push_back(alphabet.codes[static_cast<std::uint8_t>(character)]);
        std::size_t const first = packed.offsets.back(), bytes = sequence::packed_codes_bytes(codes.size(), code_bits);
        sequence::pack_codes(codes, code_bits, bytes_of(packed.bytes.data() + first, bytes));
        packed.offsets.push_back(first + bytes);
    }
    return packed;
}

/** Runs the sequence search benchmark over the flags in @p argv. */
int bench_sequence_index_search(int argc, char const* const* argv) {
    search_bench_settings_t search;
    std::string_view base_texts_path, alphabet_name, search_costs = "alphabet", weights_path, classes_path,
                                                     query_targets_path, matches_output_path;
    counts_t window_widths {.minimum_size = 0};
    std::array<flag_t, 9> const shared = search_bench_flags(search);
    std::vector<flag_t> flags(shared.begin(), shared.end());
    std::array<flag_t, 8> const own {{
        {"--base-texts", "path", "entries, one per line", &base_texts_path},
        {"--window-widths", "bytes", "window widths for the overlap kind, like 3,5", &window_widths},
        {"--alphabet", "name", "dna, protein or text, which the needleman-wunsch kind alone takes", &alphabet_name},
        {"--search-costs", "name", "alphabet, or uniform to search a needleman-wunsch index under unit costs",
         &search_costs},
        {"--weights", "path", "256x256 int8 similarities and an int8 gap cost replacing the alphabet's", &weights_path},
        {"--classes", "path", "256 bytes, each byte's class code, replacing the alphabet's fold", &classes_path},
        {"--query-targets", "path", "the base row each query intends, one per line, judged at rank 1 and top-k",
         &query_targets_path},
        {"--matches-output", "path", "a file per search breadth, <path>.<breadth>, each query's matched base rows",
         &matches_output_path},
    }};
    flags.insert(flags.end(), own.begin(), own.end());
    settings_t const settings = read_settings(argc, argv,
                                              "usearch_bench_sequence_index_search --base-texts <path> [flags]", flags);
    print(settings);

    std::string_view const metric_name = search.metric_name.empty() ? "levenshtein" : search.metric_name;
    usearch_sequence_metric_t const metric = usearch_sequence_metric_from_name(bytes_view(metric_name));
    if (metric == usearch_sequence_metric_unknown_k)
        return fmt::println(
                   stderr,
                   "--metric=\"{}\" does not parse, expected levenshtein, levenshtein-utf8, overlap, needleman-wunsch",
                   metric_name),
               1;
    bool const aligned = metric == usearch_sequence_metric_needleman_wunsch_k;
    alphabet_t alphabet = alphabet_of(alphabet_name, false);
    alphabet_t searched_alphabet = alphabet_of(alphabet_name, search_costs == "uniform");
    if (!classes_path.empty()) {
        auto [classes_file, classes_status] = mapped_file_t::read(classes_path);
        if (failed(classes_status) || classes_file.view().size() != alphabet.codes.size() || !aligned)
            return fmt::println(stderr, "--classes=\"{}\" must hold 256 bytes for needleman-wunsch", classes_path), 1;
        std::memcpy(alphabet.codes.data(), classes_file.view().data(), alphabet.codes.size());
        alphabet.classes = 0;
        for (std::uint8_t const code : alphabet.codes)
            if (code + 1u > alphabet.classes) alphabet.classes = code + 1u;
        searched_alphabet.codes = alphabet.codes, searched_alphabet.classes = alphabet.classes;
        if (search_costs == "uniform")
            for (std::size_t code = 0; code != searched_alphabet.classes; ++code)
                searched_alphabet.similarities[code * 256 + code] = 1;
    }
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
    bool const reloaded = search_costs == "uniform";
    if (search_costs != "alphabet" && (!reloaded || !aligned || settings.residency == usearch_residency_vram_k))
        return fmt::println(
                   stderr, "--search-costs=\"{}\" must be alphabet, or uniform on a RAM or NVMe needleman-wunsch index",
                   search_costs),
               1;
    usearch_sequence_weights_t const weights = alphabet.weights(), searched_weights = searched_alphabet.weights();
    auto [file, read_status] = mapped_file_t::read(base_texts_path);
    if (failed(read_status)) return fmt::println(stderr, "--base-texts=\"{}\" could not be read", base_texts_path), 1;
    byte_tape_t tape = lines_of(file.view());
    // The index reads the packed codes alone.
    if (aligned) tape = packed_codes_of(tape, alphabet);
    if (tape.size() <= search.queries.value)
        return fmt::println(stderr, "The file holds {} entries, too few to hold out {} queries", tape.size(),
                            search.queries.value),
               1;
    std::size_t const query_first = tape.size() - search.queries.value;
    std::size_t const base_count = search.max_base_entries.value ? min_of(search.max_base_entries.value, query_first)
                                                                 : query_first;
    if (aligned)
        fmt::println("- Alphabet: {} · {} classes · {} bytes packed · searched under {} costs", alphabet_name,
                     alphabet.classes, tape.bytes.size(), search_costs);

    // The base row each held-out query intends, judged by key at rank 1 and among the top-k.
    std::vector<u64_t> query_targets;
    if (!query_targets_path.empty()) {
        auto [targets_file, targets_status] = mapped_file_t::read(query_targets_path);
        if (failed(targets_status))
            return fmt::println(stderr, "--query-targets=\"{}\" could not be read", query_targets_path), 1;
        byte_tape_t const targets = lines_of(targets_file.view());
        for (std::size_t row = 0; row != targets.size(); ++row) {
            u64_t target = 0;
            std::from_chars(targets.bytes.data() + targets.offsets[row],
                            targets.bytes.data() + targets.offsets[row + 1], target);
            query_targets.push_back(target);
        }
        if (query_targets.size() != search.queries.value)
            return fmt::println(stderr, "--query-targets holds {} rows for {} queries", query_targets.size(),
                                search.queries.value),
                   1;
    }
    auto const judge_task = [&](std::size_t expansion, copied_matches_t const& found) {
        std::vector<std::size_t> const offsets = match_offsets(found);
        if (!matches_output_path.empty()) {
            std::string rows;
            for (std::size_t query = 0; query + 1 < offsets.size(); ++query) {
                for (std::size_t match = offsets[query]; match != offsets[query + 1]; ++match) {
                    if (match != offsets[query]) rows += ' ';
                    rows += fmt::format("{}",
                                        offset_of_key(settings.seed, std::bit_cast<usearch_key_t>(found.keys[match]))
                                            .value_or(~u64_t {0}));
                }
                rows += '\n';
            }
            std::string const path = fmt::format("{}.{}", matches_output_path, expansion);
            std::FILE* const file = std::fopen(path.c_str(), "wb");
            if (!file) fmt::println(stderr, "--matches-output=\"{}\" could not be written", path);
            else std::fwrite(rows.data(), 1, rows.size(), file), std::fclose(file);
        }
        if (query_targets.empty()) return;
        std::size_t task_first = 0, task_kept = 0;
        for (std::size_t query = 0; query != query_targets.size(); ++query)
            for (std::size_t match = offsets[query]; match != offsets[query + 1]; ++match)
                if (offset_of_key(settings.seed, std::bit_cast<usearch_key_t>(found.keys[match])) ==
                    query_targets[query])
                    task_first += match == offsets[query], task_kept += 1;
        fmt::println("- Task at expansion {}: intended entry at rank 1 {:.4f} · in the top {} {:.4f} over {} queries",
                     expansion, static_cast<double>(task_first) / query_targets.size(), search.top_k.value,
                     static_cast<double>(task_kept) / query_targets.size(), query_targets.size());
    };

    sequence_geometry_t const geometry {{metric, search.connectivity.value},
                                        aligned ? &weights : nullptr,
                                        window_widths.view()};
    sequence_geometry_t const served {{metric, search.connectivity.value},
                                      aligned ? &searched_weights : nullptr,
                                      window_widths.view()};
    // A search under other costs reloads the checkpoint under them.
    std::string const checkpoint =
        !settings.checkpoint.empty() ? std::string(settings.checkpoint)
        : reloaded ? (std::filesystem::temp_directory_path() / "usearch-bench-sequence-reloaded.safegraphs").string()
                   : std::string();
    usearch_tape_view_t const entries = tape.view();
    int const exit_code = run_search_bench<sequence_verbs_t>(settings, search, geometry, served, entries, base_count,
                                                             sequence_verbs_t::slice(entries, query_first),
                                                             search.queries.value, checkpoint, judge_task);
    if (reloaded && settings.checkpoint.empty())
        for (std::string const& stale : files_starting_with(checkpoint.substr(0, checkpoint.rfind(".safegraphs"))))
            std::remove(stale.c_str());
    return exit_code;
}

} // namespace unum::usearch::bench

namespace us = unum::usearch;

int main(int argc, char** argv) { return us::bench::bench_sequence_index_search(argc, argv); }
