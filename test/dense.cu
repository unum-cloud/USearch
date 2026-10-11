/**
 *  @file test/dense.cu
 *  @author Ash Vardanian
 *  @date July 22, 2026
 *  @brief The CUDA suite's one entry point: probes the device, composes every per-dtype shard
 *      explicitly, and rolls their failure counts into the exit code.
 *
 *  The scenarios live in @c dense_index.cuh, the ledgered device adapter & the cell
 *  composer in @c dense_index_cuda.cuh; the cells themselves are named by the per-dtype
 *  shard translation units, so their instantiation grids build in parallel.
 */
#include <cuda_runtime.h>

#include <type_traits> // `std::is_move_constructible_v` - the plans' witnesses

#include <fmt/format.h>

#include "dense/vram_index.cuh"
#include "dense/vram_indexes.cuh"

using namespace unum::usearch;
using namespace unum::usearch::test;
using namespace unum::usearch::test::dense;

/** The device plans are move-only and never assigned over. */
using witnessed_index_t = dense_vram_index<>;
static_assert(std::is_move_constructible_v<witnessed_index_t::save_plan_t> &&
              !std::is_copy_constructible_v<witnessed_index_t::save_plan_t> &&
              !std::is_copy_assignable_v<witnessed_index_t::save_plan_t> &&
              !std::is_move_assignable_v<witnessed_index_t::save_plan_t>);
static_assert(std::is_move_constructible_v<witnessed_index_t::load_plan_t> &&
              !std::is_copy_constructible_v<witnessed_index_t::load_plan_t> &&
              !std::is_copy_assignable_v<witnessed_index_t::load_plan_t> &&
              !std::is_move_assignable_v<witnessed_index_t::load_plan_t>);

/** Ends the environment banner with the first visible device; false when there is none to test on. */
static bool log_cuda_device() {
    int devices_count = 0;
    cudaDeviceProp properties {};
    bool const reachable = cudaGetDeviceCount(&devices_count) == cudaSuccess && devices_count &&
                           cudaGetDeviceProperties(&properties, 0) == cudaSuccess;
    if (reachable) fmt::println("- CUDA: {} sm_{}{}", properties.name, properties.major, properties.minor);
    else fmt::println("- CUDA: no device");
    return reachable;
}

int main() {
    settings_t const settings = read_settings();
    print(probe_machine());
    print(settings);
    if (!log_cuda_device()) return 0;
    std::size_t failures_count = 0;
    if (settings.selects("vram_index_bf16")) failures_count += dense::check_vram_index_bf16(settings);
#if USEARCH_DENSE_DTYPES >= 2
    if (settings.selects("vram_index_u1")) failures_count += dense::check_vram_index_u1(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 3
    if (settings.selects("vram_index_i8")) failures_count += dense::check_vram_index_i8(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 4
    if (settings.selects("vram_index_e4m3")) failures_count += dense::check_vram_index_e4m3(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 5
    if (settings.selects("vram_index_f16")) failures_count += dense::check_vram_index_f16(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 6
    if (settings.selects("vram_index_u8")) failures_count += dense::check_vram_index_u8(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 7
    if (settings.selects("vram_index_e5m2")) failures_count += dense::check_vram_index_e5m2(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 8
    if (settings.selects("vram_index_i4")) failures_count += dense::check_vram_index_i4(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 9
    if (settings.selects("vram_index_u4")) failures_count += dense::check_vram_index_u4(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 10
    if (settings.selects("vram_index_e3m2")) failures_count += dense::check_vram_index_e3m2(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 11
    if (settings.selects("vram_index_e2m3")) failures_count += dense::check_vram_index_e2m3(settings);
#endif
    if (settings.selects("vram_indexes")) failures_count += dense::check_vram_indexes(settings);
    return report("total", failures_count) ? 1 : 0;
}
