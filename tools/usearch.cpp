/**
 *  @file tools/usearch.cpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The command line of @c usearch and @c usearch-mpi: a modality and its verb, or a verb
 *      over checkpoints of any modality, dispatched to the unit implementing it.
 *
 *  Every verb calls the library through its C interface, and prints one line per step. Both
 *  binaries share every verb and differ only in the world they run on. The first SIGINT or SIGTERM
 *  stops the verb at its next barrier, which saves what stands and exits with 130; a second ends
 *  the process at once.
 */
#include <csignal> // `std::signal`, `SIGINT`, `SIGTERM`
#include <cstdio>  // `std::setvbuf`, `stdout`
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`, `std::getenv`

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
        sz::string_t const variable = retrieval_variable(flag);
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

/** Asks the verb in hand to stop at its next barrier; only a flag is safe to touch here. */
static void request_stop(int) noexcept {
    stop_requested = 1;
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
}

/** Runs the verb @p argv names and answers its exit status. */
static int run_verb(world_t& world, int argc, char** argv) {
    std::string_view const first = argc > 1 ? argv[1] : "";
    if (std::optional<int> const exit = run_modality_verb<dense_verbs_t>(world, argc, argv)) return *exit;
    if (std::optional<int> const exit = run_modality_verb<sequence_verbs_t>(world, argc, argv)) return *exit;
    if (std::optional<int> const exit = run_modality_verb<shapes_verbs_t>(world, argc, argv)) return *exit;
    if (std::optional<int> const exit = run_modality_verb<geospatial_verbs_t>(world, argc, argv)) return *exit;
    if (first == "metadata") return checkpoints_metadata(world, argc - 1, argv + 1);
    if (first == "copy") return checkpoints_copy(world, argc - 1, argv + 1);
    if (first == "export") return checkpoints_export(world, argc - 1, argv + 1);
    if (first == "import") return checkpoints_import(world, argc - 1, argv + 1);
    if (first == "recall") return checkpoints_recall(world, argc - 1, argv + 1);
    bool const asked = first == "--help";
    std::FILE* const stream = asked ? stdout : stderr;
    if (world.rank == 0)
        fmt::println(stream, "Usage: usearch <dense|sequence|shapes|geospatial> " //
                             "<insert|refine|reshard|health|search|metadata> --help");
    if (world.rank == 0) fmt::println(stream, "       usearch dense get --help");
    if (world.rank == 0) fmt::println(stream, "       usearch <metadata|copy|export|import|recall> --help");
    return asked ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace unum::usearch::tools

namespace us = unum::usearch;

int main(int argc, char** argv) {
    // Steps take minutes, so each line reaches a redirected log as it is printed.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    us::tools::world_t world {std::span<char*> {argv, static_cast<std::size_t>(argc)}};
    std::signal(SIGINT, us::tools::request_stop);
    std::signal(SIGTERM, us::tools::request_stop);
    int const exit = us::tools::run_verb(world, static_cast<int>(world.arguments.size()), world.arguments.data());
    return exit == EXIT_SUCCESS && us::tools::stop_requested ? us::tools::exit_stopped_k : exit;
}
