/**
 *  @file bench/dense/vram_kernels.cpp
 *  @author Ash Vardanian
 *  @date October 12, 2026
 *  @brief Each compiled dtype's fatbin the CUDA benchmarks load, linked in as raw bytes beside them
 *      from the device units of @c vram_kernels_<dtype>.cu.
 */
#include "dense/vram_kernels.cuh"

namespace unum::usearch::bench::dense {

extern "C" byte_t const bench_vram_nvidia_bf16_bundle_start[], bench_vram_nvidia_bf16_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_bf16_k>() noexcept {
    return {bench_vram_nvidia_bf16_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_bf16_bundle_end - bench_vram_nvidia_bf16_bundle_start)};
}
#if USEARCH_DENSE_DTYPES >= 2
extern "C" byte_t const bench_vram_nvidia_u1_bundle_start[], bench_vram_nvidia_u1_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_u1_k>() noexcept {
    return {bench_vram_nvidia_u1_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_u1_bundle_end - bench_vram_nvidia_u1_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern "C" byte_t const bench_vram_nvidia_i8_bundle_start[], bench_vram_nvidia_i8_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_i8_k>() noexcept {
    return {bench_vram_nvidia_i8_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_i8_bundle_end - bench_vram_nvidia_i8_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern "C" byte_t const bench_vram_nvidia_e4m3_bundle_start[], bench_vram_nvidia_e4m3_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_e4m3_k>() noexcept {
    return {bench_vram_nvidia_e4m3_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_e4m3_bundle_end - bench_vram_nvidia_e4m3_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern "C" byte_t const bench_vram_nvidia_f16_bundle_start[], bench_vram_nvidia_f16_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_f16_k>() noexcept {
    return {bench_vram_nvidia_f16_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_f16_bundle_end - bench_vram_nvidia_f16_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern "C" byte_t const bench_vram_nvidia_u8_bundle_start[], bench_vram_nvidia_u8_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_u8_k>() noexcept {
    return {bench_vram_nvidia_u8_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_u8_bundle_end - bench_vram_nvidia_u8_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern "C" byte_t const bench_vram_nvidia_e5m2_bundle_start[], bench_vram_nvidia_e5m2_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_e5m2_k>() noexcept {
    return {bench_vram_nvidia_e5m2_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_e5m2_bundle_end - bench_vram_nvidia_e5m2_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern "C" byte_t const bench_vram_nvidia_i4_bundle_start[], bench_vram_nvidia_i4_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_i4_k>() noexcept {
    return {bench_vram_nvidia_i4_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_i4_bundle_end - bench_vram_nvidia_i4_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern "C" byte_t const bench_vram_nvidia_u4_bundle_start[], bench_vram_nvidia_u4_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_u4_k>() noexcept {
    return {bench_vram_nvidia_u4_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_u4_bundle_end - bench_vram_nvidia_u4_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern "C" byte_t const bench_vram_nvidia_e3m2_bundle_start[], bench_vram_nvidia_e3m2_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_e3m2_k>() noexcept {
    return {bench_vram_nvidia_e3m2_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_e3m2_bundle_end - bench_vram_nvidia_e3m2_bundle_start)};
}
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern "C" byte_t const bench_vram_nvidia_e2m3_bundle_start[], bench_vram_nvidia_e2m3_bundle_end[];
template <>
bytes_view_t cuda_bench_bundle<nk_e2m3_k>() noexcept {
    return {bench_vram_nvidia_e2m3_bundle_start,
            static_cast<std::size_t>(bench_vram_nvidia_e2m3_bundle_end - bench_vram_nvidia_e2m3_bundle_start)};
}
#endif

} // namespace unum::usearch::bench::dense
