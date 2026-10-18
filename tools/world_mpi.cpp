/**
 *  @file tools/world_mpi.cpp
 *  @author Ash Vardanian
 *  @date October 12, 2026
 *  @brief The MPI world @c usearch-mpi links, the only unit that includes MPI.
 *
 *  Calls MPI through its C interface alone, so the binary runs under any implementation it was
 *  configured against: Open MPI, MPICH, Intel MPI or MS-MPI.
 */
#include <mpi.h>

#include "world.hpp"

namespace unum::usearch::tools {

world_t::world_t(int& argc, char**& argv) noexcept {
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1, node_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &node_rank);
    MPI_Comm_free(&node);
    rank = static_cast<std::size_t>(world_rank);
    size = static_cast<std::size_t>(world_size);
    local_rank = static_cast<std::size_t>(node_rank);
}

world_t::~world_t() noexcept { MPI_Finalize(); }

bool world_t::barrier(bool failed) noexcept {
    int any_failed = failed;
    MPI_Allreduce(MPI_IN_PLACE, &any_failed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return any_failed;
}

void world_t::gather(std::span<std::uint64_t const> sent, std::span<std::uint64_t> received) noexcept {
    int const count = static_cast<int>(sent.size());
    MPI_Gather(sent.data(), count, MPI_UINT64_T, received.data(), count, MPI_UINT64_T, 0, MPI_COMM_WORLD);
}

} // namespace unum::usearch::tools
