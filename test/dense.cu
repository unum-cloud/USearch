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

int main() {
    int devices_count = 0;
    cudaError_t const probed = cudaGetDeviceCount(&devices_count);
    if (probed != cudaSuccess || !devices_count)
        return static_cast<int>(us::test::report("cuda", "no CUDA device reachable"));
    fmt::println("CUDA devices: {}", devices_count);
    std::size_t failures_count = 0;
    failures_count += dense::check_vram_index_bf16();
    failures_count += dense::check_vram_index_f16();
    failures_count += dense::check_vram_index_e5m2();
    failures_count += dense::check_vram_index_e4m3();
    failures_count += dense::check_vram_index_e3m2();
    failures_count += dense::check_vram_index_e2m3();
    failures_count += dense::check_vram_index_i8();
    failures_count += dense::check_vram_index_i4();
    failures_count += dense::check_vram_index_u8();
    failures_count += dense::check_vram_index_u4();
    failures_count += dense::check_vram_index_u1();
    failures_count += dense::check_vram_indexes();
    return failures_count ? 1 : 0;
}
