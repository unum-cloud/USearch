/**
 *  @file src/dense/vram_variants_nvidia_blackwell.cpp
 *  @author Ash Vardanian
 *  @date September 26, 2026
 *  @brief The nvidia_blackwell variants: they inline the mixed-precision multiply-add &
 *      fp6-conversion kits, so their device units compile for "100f" alone.
 *
 *  Clang compiles it host-only into the variants, which find each dtype's fatbin linked in beside
 *  them; NVCC compiles every @c vram_kernels_nvidia_<dtype>.cu device-only into this class's.
 */
#include "dense_specializations.cuh"
#include "usearch/dense/vram_nvidia_index.cuh"

/** Each dtype's fatbin, linked in as raw bytes beside the variants. */
extern "C" byte_t const dense_nvidia_blackwell_bf16_bundle_start[], dense_nvidia_blackwell_bf16_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_bf16_k>::bytes() noexcept {
    return {
        dense_nvidia_blackwell_bf16_bundle_start,
        static_cast<std::size_t>(dense_nvidia_blackwell_bf16_bundle_end - dense_nvidia_blackwell_bf16_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_blackwell_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern "C" byte_t const dense_nvidia_blackwell_u1_bundle_start[], dense_nvidia_blackwell_u1_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_u1_k>::bytes() noexcept {
    return {dense_nvidia_blackwell_u1_bundle_start,
            static_cast<std::size_t>(dense_nvidia_blackwell_u1_bundle_end - dense_nvidia_blackwell_u1_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern "C" byte_t const dense_nvidia_blackwell_i8_bundle_start[], dense_nvidia_blackwell_i8_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_i8_k>::bytes() noexcept {
    return {dense_nvidia_blackwell_i8_bundle_start,
            static_cast<std::size_t>(dense_nvidia_blackwell_i8_bundle_end - dense_nvidia_blackwell_i8_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern "C" byte_t const dense_nvidia_blackwell_e4m3_bundle_start[], dense_nvidia_blackwell_e4m3_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_e4m3_k>::bytes() noexcept {
    return {
        dense_nvidia_blackwell_e4m3_bundle_start,
        static_cast<std::size_t>(dense_nvidia_blackwell_e4m3_bundle_end - dense_nvidia_blackwell_e4m3_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern "C" byte_t const dense_nvidia_blackwell_f16_bundle_start[], dense_nvidia_blackwell_f16_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_f16_k>::bytes() noexcept {
    return {dense_nvidia_blackwell_f16_bundle_start,
            static_cast<std::size_t>(dense_nvidia_blackwell_f16_bundle_end - dense_nvidia_blackwell_f16_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern "C" byte_t const dense_nvidia_blackwell_u8_bundle_start[], dense_nvidia_blackwell_u8_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_u8_k>::bytes() noexcept {
    return {dense_nvidia_blackwell_u8_bundle_start,
            static_cast<std::size_t>(dense_nvidia_blackwell_u8_bundle_end - dense_nvidia_blackwell_u8_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern "C" byte_t const dense_nvidia_blackwell_e5m2_bundle_start[], dense_nvidia_blackwell_e5m2_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_e5m2_k>::bytes() noexcept {
    return {
        dense_nvidia_blackwell_e5m2_bundle_start,
        static_cast<std::size_t>(dense_nvidia_blackwell_e5m2_bundle_end - dense_nvidia_blackwell_e5m2_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern "C" byte_t const dense_nvidia_blackwell_i4_bundle_start[], dense_nvidia_blackwell_i4_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_i4_k>::bytes() noexcept {
    return {dense_nvidia_blackwell_i4_bundle_start,
            static_cast<std::size_t>(dense_nvidia_blackwell_i4_bundle_end - dense_nvidia_blackwell_i4_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern "C" byte_t const dense_nvidia_blackwell_u4_bundle_start[], dense_nvidia_blackwell_u4_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_u4_k>::bytes() noexcept {
    return {dense_nvidia_blackwell_u4_bundle_start,
            static_cast<std::size_t>(dense_nvidia_blackwell_u4_bundle_end - dense_nvidia_blackwell_u4_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern "C" byte_t const dense_nvidia_blackwell_e3m2_bundle_start[], dense_nvidia_blackwell_e3m2_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_e3m2_k>::bytes() noexcept {
    return {
        dense_nvidia_blackwell_e3m2_bundle_start,
        static_cast<std::size_t>(dense_nvidia_blackwell_e3m2_bundle_end - dense_nvidia_blackwell_e3m2_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern "C" byte_t const dense_nvidia_blackwell_e2m3_bundle_start[], dense_nvidia_blackwell_e2m3_bundle_end[];
template <>
bytes_view_t cuda_fatbin<gpu_class_t::nvidia_blackwell_k, nk_e2m3_k>::bytes() noexcept {
    return {
        dense_nvidia_blackwell_e2m3_bundle_start,
        static_cast<std::size_t>(dense_nvidia_blackwell_e2m3_bundle_end - dense_nvidia_blackwell_e2m3_bundle_start)};
}
template struct specialized_index<vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_blackwell_k>>;
#endif
