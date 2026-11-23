/**
 *  @file test/dense/vram_indexes.cpp
 *  @author Ash Vardanian
 *  @date September 1, 2026
 *  @brief The two VRAM rung cells over bf16: the sharded suite, whose walks keep to the register
 *      ladder and shared memory, and the ledgered index's builds that spill into the warp's arena.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <string> // `std::string` - the cases' vendor labels

#include "vram_indexes.cuh"

using namespace unum::usearch;
namespace util = unum::usearch::utilities;

template <typename executor_type_>
void test::dense::run_vram_indexes_cases(executor_type_& executor) {
    util::u64_t const seed = test_run.settings.seed;
    char const* const vendor = constant_of(executor_type_::vendor_k);
    run_test_case(fmt::format("test_vram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k>({})", vendor).c_str(),
                  test_vram_indexes_configuration<nk_bf16_k, nk_kernel_angular_k, executor_type_>, seed, executor);
    run_test_case(fmt::format("test_vram_arena_configuration<nk_bf16_k, nk_kernel_angular_k>({})", vendor).c_str(),
                  test_vram_arena_configuration<nk_bf16_k, nk_kernel_angular_k, executor_type_>, seed, executor);
}

#if USEARCH_ARCH_CUDA_
template void test::dense::run_vram_indexes_cases<util::cuda_executor_t>(util::cuda_executor_t&);
#endif
#if USEARCH_ARCH_ROCM_
template void test::dense::run_vram_indexes_cases<util::hip_executor_t>(util::hip_executor_t&);
#endif
#if USEARCH_ARCH_METAL_
template void test::dense::run_vram_indexes_cases<util::metal_executor_t>(util::metal_executor_t&);
#endif
