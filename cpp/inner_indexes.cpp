/**
 *  @brief Vector Search index instantiations for a single uniform latency memory space.
 *  @author Ash Vardanian
 *  @file cpp/inner_indexes.cpp
 *  @date Friday 17, 2026
 *
 *  This file is the god-level meeting point of various templates - fusing together the graph structures,
 *  the memory allocators, and the thread management primitives. It hides under a single dispatch layer:
 *
 *  - @b Volatile-in-RAM indexes on different NUMA nodes
 *  - @b CUDA-in-VRAM indexes on different physical GPU devices
 *  - @b Persisted-on-Disk indexes on different NVME SSDs
 *
 *
 *  For in-memory inner-product indicies
 *  we support two modes of scheduling:
 *
 *  -   a single @b colocation of threads and memory, as defined by the ForkUnion pool handling one
 *      nearest neighbors graph, processed concurrently by threads with an equal access latency;
 *  -   an entire @b distributed pool of colocations working together on the same structure, despite
 *      different communication latency and potentially NUMA-interleaved underlying memory.
 *
 *
 */
#include <forkunion.h>
#include <forkunion/allocators.hpp>
#include <numkong/numkong.hpp>

#include <usearch/dense/ram_index.hpp>

namespace us = unum::usearch;
namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;

using bf6_angular_domain_t = dense_ram_index<nk_bf16_k, nk_kernel_angular_k, nk_cap_icelake_k, fu::preferred_yield_t,
                                             fu::domain_allocator<nk::bf16_t>,
                                             fu::domain_allocator<us::dense_node_id_t>, std::allocator<std::byte>>;
