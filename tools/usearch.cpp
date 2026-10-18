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
                              std::span<bench::flag_t const> flags, std::span<std::string_view> positionals,
                              std::size_t& given) {
    // Every rank parses the same arguments, so they agree on failing, and only rank 0 says why.
    bool const speaks = world.rank == 0;
    for (bench::flag_t const& flag : flags) {
        std::string const variable = bench::retrieval_variable(flag);
        char const* const value = std::getenv(variable.c_str());
        if (!value || !*value || bench::assign_variable_(flag, value) == bench::argument_status_t::parsed_k) continue;
        if (speaks)
            fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", variable, value, bench::grammar(flag));
        return EXIT_FAILURE;
    }
    bench::arguments_t const parsed = bench::parse_arguments(argc, argv, flags, positionals);
    given = parsed.positionals_count;
    if (parsed.status == bench::argument_status_t::parsed_k) return std::nullopt;
    if (parsed.status == bench::argument_status_t::help_k) {
        if (speaks) bench::print_usage(synopsis, flags);
        return EXIT_SUCCESS;
    }
    if (!speaks) return EXIT_FAILURE;
    if (parsed.status == bench::argument_status_t::too_many_items_k)
        fmt::println(stderr, "{} is one argument too many, see --help", parsed.argument);
    else if (!parsed.flag) fmt::println(stderr, "{} is not a flag, see --help", parsed.argument);
    else
        fmt::println(stderr, "{}=\"{}\" does not parse, expected {}", parsed.flag->name, parsed.value,
                     bench::grammar(*parsed.flag));
    return EXIT_FAILURE;
}

} // namespace unum::usearch::tools

namespace us = unum::usearch;

int main(int argc, char** argv) {
    // Steps take minutes, so each line reaches a redirected log as it is printed.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    us::tools::world_t world {argc, argv};
    std::string_view const first = argc > 1 ? argv[1] : "";
    std::string_view const second = argc > 2 ? argv[2] : "";
    // Each verb parses the arguments after its name, reading the name where a program name stands.
    if (first == "dense" && second == "insert") return us::tools::dense_insert(world, argc - 2, argv + 2);
    if (first == "dense" && second == "absorb") return us::tools::dense_absorb(world, argc - 2, argv + 2);
    if (first == "dense" && second == "refine") return us::tools::dense_refine(world, argc - 2, argv + 2);
    if (first == "dense" && second == "split") return us::tools::dense_split(world, argc - 2, argv + 2);
    if (first == "dense" && second == "health") return us::tools::dense_health(world, argc - 2, argv + 2);
    if (first == "metadata") return us::tools::checkpoints_metadata(world, argc - 1, argv + 1);
    if (first == "slots") return us::tools::checkpoints_slots(world, argc - 1, argv + 1);
    if (first == "copy") return us::tools::checkpoints_copy(world, argc - 1, argv + 1);
    bool const asked = first == "--help";
    std::FILE* const stream = asked ? stdout : stderr;
    if (world.rank == 0) fmt::println(stream, "Usage: usearch dense <insert|absorb|refine|split|health> --help");
    if (world.rank == 0) fmt::println(stream, "       usearch <metadata|slots|copy> --help");
    return asked ? EXIT_SUCCESS : EXIT_FAILURE;
}
