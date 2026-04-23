/**
 *  @brief Vector Search index for a single uniform latency memory space.
 *  @author Ash Vardanian
 *  @file inner_index.hpp
 *  @date July 26, 2023
 */
#pragma once

#include "usearch/inner_metric.hpp"

namespace unum {
namespace usearch {

/**
 *  @brief Vector Search index for a single uniform latency memory space.
 *
 *  @section Supported Metrics: Only "Inner Product class"
 *
 *  This template is designed exclusively for dense representations in the most common metric spaces:
 *  Inner Product, Angular (Cosine), Euclidean (L2), Hamming (Manhattan), and Jaccard (Tanimoto).
 *  All of them can be trivially inferred in a single pass of vector-vector products if L1 and L2
 *  norms of the vectors are known ahead of time and stored locally.
 *
 *  @section Growth Limits
 *
 *  This index is designed exclusively for up to 4 billion entries. Float32 distance representations are used
 *  to compare entries which may result in recall regressions for small K queries. The typical mode of use is
 *  to retrieve significantly more entries from the index of quantized vectors and rerank them with either
 *  pre-quantized vectors or original data.
 *
 *  @section Concurrency & Transactions
 *
 *  This class is designed to allow simutanous construction and traversal in heavily multi-threaded environments.
 *  The number of locks (userspace mutexes) is proportion to the number of configured threads, not the index size.
 *  All threads are expected to have the same access latency to the entries in this index. As for @b Visibility
 *  contracts in this class, insertions follow a two-stage commit semantic, where changes to the graph are not
 *  made visible to other threads until the change is committed.
 *
 *  @section Persistence
 *
 *  This index is not expected to be accessed from disk. For that, explore io_uring and SPDK reimplementations.
 *
 */
template <                                               //
    nk_dtype_t vector_dtype_ = nk_bf16_k,                //
    nk_kernel_kind_t kernel_kind_ = nk_kernel_angular_k, //
    nk_capability_t capability_ = nk_cap_icelake_k       //
    >
class inner_index;

template <nk_capability_t capability_ = nk_cap_icelake_k> //
class traveral_logic_;

template <nk_dtype_t vector_dtype_, nk_kernel_kind_t kernel_kind_, nk_capability_t capability_> //
class inner_index {

    using metric_logic_t = metric_logic_<vector_dtype_, kernel_kind_, capability_>;
    using traveral_logic_t = traveral_logic_<capability_>;
}

} // namespace usearch
} // namespace unum
