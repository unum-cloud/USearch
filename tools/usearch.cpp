/**
 *  @file tools/usearch.cpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The command line of @c usearch and @c usearch-mpi: a modality and its verb, or a verb
 *      over checkpoints of any modality, dispatched to the unit implementing it.
 *
 *  Every verb calls the library through its C interface, and prints one line per step. Both
 *  binaries share every verb and differ only in the world they run on.
 */
#include <cstdio>  // `std::setvbuf`, `stdout`
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`, `std::getenv`

#include <string>      // `std::string`, for the variable names
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "world.hpp"

namespace unum::usearch::tools {

std::optional<int> parse_verb(world_t const& world, int argc, char const* const* argv, std::string_view synopsis,
                              std::span<flag_t const> flags, std::span<std::string_view> positionals,
                              std::size_t& given) {
    // Every rank parses the same arguments, so they agree on failing, and only rank 0 says why.
    bool const speaks = world.rank == 0;
    for (flag_t const& flag : flags) {
        std::string const variable = retrieval_variable(flag);
        char const* const value = std::getenv(variable.c_str());
        if (!value || !*value || assign_variable_(flag, value) == argument_status_t::parsed_k) continue;
        if (speaks) fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", variable, value, grammar(flag));
        return EXIT_FAILURE;
    }
    arguments_t const parsed = parse_arguments(argc, argv, flags, positionals);
    given = parsed.positionals_count;
    if (parsed.status == argument_status_t::parsed_k) return std::nullopt;
    if (parsed.status == argument_status_t::help_k) {
        if (speaks) print_usage(synopsis, flags);
        return EXIT_SUCCESS;
    }
    if (!speaks) return EXIT_FAILURE;
    if (parsed.status == argument_status_t::too_many_items_k)
        fmt::println(stderr, "{} is one argument too many, see --help", parsed.argument);
    else if (!parsed.flag) fmt::println(stderr, "{} is not a flag, see --help", parsed.argument);
    else
        fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", parsed.flag->name, parsed.value,
                     grammar(*parsed.flag));
    return EXIT_FAILURE;
}

} // namespace unum::usearch::tools

namespace us = unum::usearch;

int main(int argc, char** argv) {
    // Steps take minutes, so each line reaches a redirected log as it is printed.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    us::tools::world_t world {argc, argv};
    std::string_view const first = argc > 1 ? argv[1] : "";
    if (std::optional<int> const exit = us::tools::run_modality_verb<us::tools::dense_verbs_t>(world, argc, argv))
        return *exit;
    if (std::optional<int> const exit = us::tools::run_modality_verb<us::tools::sequence_verbs_t>(world, argc, argv))
        return *exit;
    if (std::optional<int> const exit = us::tools::run_modality_verb<us::tools::shapes_verbs_t>(world, argc, argv))
        return *exit;
    if (std::optional<int> const exit = us::tools::run_modality_verb<us::tools::geospatial_verbs_t>(world, argc, argv))
        return *exit;
    if (first == "metadata") return us::tools::checkpoints_metadata(world, argc - 1, argv + 1);
    if (first == "copy") return us::tools::checkpoints_copy(world, argc - 1, argv + 1);
    if (first == "export") return us::tools::checkpoints_export(world, argc - 1, argv + 1);
    if (first == "import") return us::tools::checkpoints_import(world, argc - 1, argv + 1);
    if (first == "recall") return us::tools::checkpoints_recall(world, argc - 1, argv + 1);
    bool const asked = first == "--help";
    std::FILE* const stream = asked ? stdout : stderr;
    if (world.rank == 0)
        fmt::println(stream, "Usage: usearch <dense|sequence|shapes|geospatial> " //
                             "<insert|absorb|refine|split|health|search|metadata> --help");
    if (world.rank == 0) fmt::println(stream, "       usearch dense get --help");
    if (world.rank == 0) fmt::println(stream, "       usearch <metadata|copy|export|import|recall> --help");
    return asked ? EXIT_SUCCESS : EXIT_FAILURE;
}
