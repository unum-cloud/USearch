/**
 *  @file src/dense/vram_variants_amd_cdna3.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The amd_cdna3 unit: MI300X & MI325X, gfx942; its variants inline the CDNA3 dot kits and widen
 *      fp8 in software.
 *
 *  HIP-Clang compiles it host-only into the variants, which find the class's code object linked in
 *  beside them; @c vram_kernels_amd_cdna3.hip compiles that code object device-only.
 */
#include "dense_kernels.cuh"
#include "usearch/dense/vram_amd_index.hip"

/** The class's code object, linked in as raw bytes beside the variants. */
extern "C" byte_t const dense_amd_cdna3_bundle_start[], dense_amd_cdna3_bundle_end[];

template <>
bytes_view_t hip_code_object<gpu_class_t::amd_cdna3_k>::bytes() noexcept {
    return {dense_amd_cdna3_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna3_bundle_end - dense_amd_cdna3_bundle_start)};
}

template struct specialized_index<vram_amd_indexes<nk_bf16_k, gpu_class_t::amd_cdna3_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<vram_amd_indexes<nk_u1_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<vram_amd_indexes<nk_i8_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<vram_amd_indexes<nk_e4m3_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<vram_amd_indexes<nk_f16_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<vram_amd_indexes<nk_u8_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<vram_amd_indexes<nk_e5m2_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<vram_amd_indexes<nk_i4_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<vram_amd_indexes<nk_u4_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<vram_amd_indexes<nk_e3m2_k, gpu_class_t::amd_cdna3_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<vram_amd_indexes<nk_e2m3_k, gpu_class_t::amd_cdna3_k>>;
#endif
