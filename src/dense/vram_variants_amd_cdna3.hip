/**
 *  @file src/dense/vram_variants_amd_cdna3.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The amd_cdna3 unit: MI300X & MI325X, gfx942; its variants inline the CDNA3 dot kits and widen
 *      fp8 in software.
 *
 *  HIP-Clang compiles it host-only into the variants, which find each dtype's code object linked in
 *  beside them; HIP-Clang compiles every @c vram_kernels_amd_<dtype>.hip device-only into this
 *  class's.
 */
#include "dense_specializations.cuh"
#include "usearch/dense/vram_amd_index.hip"

/** Each dtype's code object, linked in as raw bytes beside the variants. */
extern "C" byte_t const dense_amd_cdna3_bf16_bundle_start[], dense_amd_cdna3_bf16_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_bf16_k>::bytes() noexcept {
    return {dense_amd_cdna3_bf16_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_bf16_bundle_end - dense_amd_cdna3_bf16_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_bf16_k, gpu_class_t::amd_cdna3_k>>;
#if USEARCH_DENSE_DTYPES >= 2
extern "C" byte_t const dense_amd_cdna3_u1_bundle_start[], dense_amd_cdna3_u1_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_u1_k>::bytes() noexcept {
    return {dense_amd_cdna3_u1_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_u1_bundle_end - dense_amd_cdna3_u1_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_u1_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
extern "C" byte_t const dense_amd_cdna3_i8_bundle_start[], dense_amd_cdna3_i8_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_i8_k>::bytes() noexcept {
    return {dense_amd_cdna3_i8_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_i8_bundle_end - dense_amd_cdna3_i8_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_i8_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
extern "C" byte_t const dense_amd_cdna3_e4m3_bundle_start[], dense_amd_cdna3_e4m3_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_e4m3_k>::bytes() noexcept {
    return {dense_amd_cdna3_e4m3_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_e4m3_bundle_end - dense_amd_cdna3_e4m3_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_e4m3_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
extern "C" byte_t const dense_amd_cdna3_f16_bundle_start[], dense_amd_cdna3_f16_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_f16_k>::bytes() noexcept {
    return {dense_amd_cdna3_f16_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_f16_bundle_end - dense_amd_cdna3_f16_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_f16_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
extern "C" byte_t const dense_amd_cdna3_u8_bundle_start[], dense_amd_cdna3_u8_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_u8_k>::bytes() noexcept {
    return {dense_amd_cdna3_u8_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_u8_bundle_end - dense_amd_cdna3_u8_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_u8_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
extern "C" byte_t const dense_amd_cdna3_e5m2_bundle_start[], dense_amd_cdna3_e5m2_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_e5m2_k>::bytes() noexcept {
    return {dense_amd_cdna3_e5m2_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_e5m2_bundle_end - dense_amd_cdna3_e5m2_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_e5m2_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
extern "C" byte_t const dense_amd_cdna3_i4_bundle_start[], dense_amd_cdna3_i4_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_i4_k>::bytes() noexcept {
    return {dense_amd_cdna3_i4_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_i4_bundle_end - dense_amd_cdna3_i4_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_i4_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
extern "C" byte_t const dense_amd_cdna3_u4_bundle_start[], dense_amd_cdna3_u4_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_u4_k>::bytes() noexcept {
    return {dense_amd_cdna3_u4_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_u4_bundle_end - dense_amd_cdna3_u4_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_u4_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
extern "C" byte_t const dense_amd_cdna3_e3m2_bundle_start[], dense_amd_cdna3_e3m2_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_e3m2_k>::bytes() noexcept {
    return {dense_amd_cdna3_e3m2_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_e3m2_bundle_end - dense_amd_cdna3_e3m2_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_e3m2_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
extern "C" byte_t const dense_amd_cdna3_e2m3_bundle_start[], dense_amd_cdna3_e2m3_bundle_end[];
template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k, nk_e2m3_k>::bytes() noexcept {
    return {dense_amd_cdna3_e2m3_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_e2m3_bundle_end - dense_amd_cdna3_e2m3_bundle_start)};
}
template struct specialized_index<vram_amd_indexes<nk_e2m3_k, gpu_class_t::amd_cdna3_k>>;
#endif
