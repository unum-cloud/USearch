/**
 *  @brief The u8 shard of the dense router library: carries this dtype's index
 *      instantiations for the dtype-free trampoline to dispatch onto.
 *  @author Ash Vardanian
 *  @file cpp/dense_indexes_u8.cpp
 *  @date July 23, 2026
 *
 *  The alias below freezes the shard's coordinate; the explicit instantiations and the
 *  C-surface wrappers land here as the router facade's verbs firm up.
 */
#include <memory> // `std::allocator`

#include <usearch/dense/ram_index.hpp>

namespace us = unum::usearch;

using u8_angular_ram_index_t =
    us::dense_ram_index<nk_u8_k, nk_kernel_angular_k, nk_cap_serial_k, us::dummy_waiter_t, std::allocator<std::byte>,
                        std::allocator<std::byte>, std::allocator<std::byte>>;
