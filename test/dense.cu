/**
 *  @brief The CUDA suite's one entry point: probes the device, composes every per-dtype
 *      shard explicitly, and rolls their failure counts into the exit code.
 *  @author Ash Vardanian
 *  @file test/dense.cu
 *  @date July 22, 2026
 *
 *  The scenarios live in @c dense_index.cuh, the ledgered device adapter & the cell
 *  composer in @c dense_index_cuda.cuh; the cells themselves are named by the per-dtype
 *  shard translation units, so their instantiation grids build in parallel.
 */
#include <cuda_runtime.h>

#include <fmt/format.h>

#include "dense/vram_index.cuh"
#include "dense/vram_indexes.cuh"

int main() {
    int devices_count = 0;
    cudaError_t const probed = cudaGetDeviceCount(&devices_count);
    if (probed != cudaSuccess || !devices_count)
        return static_cast<int>(unum::usearch::report("cuda", "no CUDA device reachable"));
    fmt::println("CUDA devices: {}", devices_count);
    std::size_t failures_count = 0;
    failures_count += test_dense_index_bf16_cuda();
    failures_count += test_dense_index_f16_cuda();
    failures_count += test_dense_index_i8_cuda();
    failures_count += test_dense_index_i4_cuda();
    failures_count += test_dense_index_u8_cuda();
    failures_count += test_dense_index_u1_cuda();
    failures_count += test_dense_indexes_cuda();
    return failures_count ? 1 : 0;
}
