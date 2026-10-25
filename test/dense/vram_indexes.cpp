/**
 *  @file test/dense/vram_indexes.cpp
 *  @author Ash Vardanian
 *  @date September 1, 2026
 *  @brief The two VRAM rung cells over bf16: the sharded suite, whose walks keep to the register
 *      ladder and shared memory, and the ledgered index's builds that spill into the warp's arena.
 */
#include "vram_indexes.cuh"

using namespace unum::usearch;
using namespace unum::usearch::utilities;

template <typename executor_type_>
std::size_t test::dense::check_vram_indexes(settings_t const& settings, executor_type_& executor) {
    if (!settings.filter.selects("vram_indexes")) return 0;
    std::size_t failures_count = 0;
    failures_count += test_vram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k>("sharded bf16+angular",
                                                                                      settings.seed, executor);
    failures_count += test_vram_arena_configuration<nk_bf16_k, nk_kernel_angular_k>("bf16+angular+arena", settings.seed,
                                                                                    executor);
    return failures_count;
}

#if USEARCH_ARCH_CUDA_
template std::size_t test::dense::check_vram_indexes<cuda_executor_t>(settings_t const&, cuda_executor_t&);
#endif
#if USEARCH_ARCH_ROCM_
template std::size_t test::dense::check_vram_indexes<hip_executor_t>(settings_t const&, hip_executor_t&);
#endif
#if USEARCH_ARCH_METAL_
template std::size_t test::dense::check_vram_indexes<metal_executor_t>(settings_t const&, metal_executor_t&);
#endif
