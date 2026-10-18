/**
 *  @file tools/world.hpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief The process group a verb runs on, and the verbs both tools dispatch to.
 *
 *  Each binary links one definition of @ref world_t: @c world_local.cpp for a world of one process,
 *  @c world_mpi.cpp for an MPI world, the only unit that includes MPI, so the library and every
 *  verb stay free of it. Ranks talk only through @ref world_t::barrier and @ref world_t::gather.
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <optional>    // `std::optional`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`

#include <fmt/format.h>

#include "arguments.hpp" // `bench::flag_t`

namespace unum::usearch::tools {

/** The ranks a verb runs on, started from the program's arguments and finished when it dies. */
struct world_t {

    /** This process among @ref size, counted from zero. */
    std::size_t rank = 0;

    /** Processes the verb runs on. */
    std::size_t size = 1;

    /** This process among those sharing its node, which picks its GPU or memory domain. */
    std::size_t local_rank = 0;

    world_t(int& argc, char**& argv) noexcept;
    world_t(world_t const&) = delete;
    world_t& operator=(world_t const&) = delete;
    ~world_t() noexcept;

    /** Waits for every rank and answers whether any of them passed a true @p failed. */
    bool barrier(bool failed) noexcept;

    /** Gathers @p sent from every rank into @p received on rank 0, rank after rank; the other ranks
     *  pass an empty @p received. */
    void gather(std::span<std::uint64_t const> sent, std::span<std::uint64_t> received) noexcept;
};

/** What leads a rank's lines while several run, like "rank 3: ", and nothing in a world of one. */
inline std::string rank_label(world_t const& world) {
    return world.size > 1 ? fmt::format("rank {}: ", world.rank) : std::string {};
}

/** Whether a verb that runs on one process may run here, rank 0 saying why not under several. */
inline bool runs_alone(world_t const& world) {
    if (world.size == 1) return true;
    if (world.rank == 0) fmt::println(stderr, "only dense insert runs across ranks; run this verb under usearch");
    return false;
}

/** Parses @p argv against @p flags, each defaulting to its @c RETRIEVAL_ variable, and collects
 *  up to @p positionals.size() positionals, counting them into @p given. Answers the status to
 *  exit with when the verb ends here: after the usage on @c --help, or after rank 0 names what
 *  failed. */
std::optional<int> parse_verb(world_t const& world, int argc, char const* const* argv, std::string_view synopsis,
                              std::span<bench::flag_t const> flags, std::span<std::string_view> positionals,
                              std::size_t& given);

/** Each verb takes the world, then the arguments after its name with the name where a program name
 *  stands, and answers the process's exit status. */
int dense_insert(world_t& world, int argc, char const* const* argv);
int dense_absorb(world_t& world, int argc, char const* const* argv);
int dense_refine(world_t& world, int argc, char const* const* argv);
int dense_split(world_t& world, int argc, char const* const* argv);
int dense_health(world_t& world, int argc, char const* const* argv);
int checkpoints_metadata(world_t& world, int argc, char const* const* argv);
int checkpoints_slots(world_t& world, int argc, char const* const* argv);
int checkpoints_copy(world_t& world, int argc, char const* const* argv);

/** Whether a save may write to @p address, printing why not: a drive takes a byte base on a slot
 *  boundary, at a slot no live file covers, since nothing tells a save's size before it writes. */
bool checkpoints_admit_save(std::string_view address);

} // namespace unum::usearch::tools
