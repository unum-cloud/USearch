/**
 *  @brief The e2m3 shard of the dense index library: carries this dtype's index
 *      instantiations for the dtype-free trampoline to dispatch onto.
 *  @author Ash Vardanian
 *  @file src/dense/ram_index_e2m3.cpp
 *  @date August 15, 2026
 *
 *  The alias below freezes the shard's coordinate; the explicit instantiations and the
 *  C-surface wrappers land here as the C entry points firm up.
 */
#include <memory> // `std::allocator`

#include <usearch/dense/ram_index.hpp>

namespace us = unum::usearch;

using e2m3_angular_ram_index_t =
    us::dense_ram_index<nk_e2m3_k, nk_kernel_angular_k, nk_cap_serial_k, us::dummy_waiter_t, std::allocator<std::byte>,
                        std::allocator<std::byte>, std::allocator<std::byte>>;
