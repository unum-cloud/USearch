/**
 *  @file tools/world_local.cpp
 *  @author Ash Vardanian
 *  @date October 18, 2026
 *  @brief A world of one process, which @c usearch links.
 */
#include <cstring> // `std::memcpy`

#include "world.hpp"

namespace unum::usearch::tools {

world_t::world_t(int&, char**&) noexcept {}
world_t::~world_t() noexcept {}

bool world_t::barrier(bool failed) noexcept { return failed; }

void world_t::gather(std::span<std::uint64_t const> sent, std::span<std::uint64_t> received) noexcept {
    std::memcpy(received.data(), sent.data(), sent.size_bytes());
}

} // namespace unum::usearch::tools
