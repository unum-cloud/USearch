/**
 *  @file src/dense/vram_variants_amd_cdna4.hip
 *  @author Ash Vardanian
 *  @date October 11, 2026
 *  @brief The amd_cdna4 unit: MI350X & MI355X, gfx950; its arms inline the bf16 dots and the OCP
 *      fp8 conversions.
 *
 *  HIP-Clang compiles it twice: device-only, once over every compiled dtype, into the class's code
 *  object, and host-only into the arms, which find that code object linked in beside them.
 */
#include "dense_specializations.cuh"
#include "usearch/dense/vram_amd_index.hip"

/** The class's code object, linked in as raw bytes beside the arms. */
extern "C" us::byte_t const dense_amd_cdna4_bundle_start[], dense_amd_cdna4_bundle_end[];

template <>
us::bytes_view_t us::dense_amd_code_object<us::dense_amd_capabilities_of(gpu_class_t::amd_cdna4_k)>() noexcept {
    return {dense_amd_cdna4_bundle_start,
            static_cast<std::size_t>(dense_amd_cdna4_bundle_end - dense_amd_cdna4_bundle_start)};
}

template struct specialized_index<vram_amd_index<nk_bf16_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_bf16_k>();
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<vram_amd_index<nk_u1_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_u1_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<vram_amd_index<nk_i8_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_i8_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<vram_amd_index<nk_e4m3_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_e4m3_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<vram_amd_index<nk_f16_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_f16_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<vram_amd_index<nk_u8_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_u8_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<vram_amd_index<nk_e5m2_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_e5m2_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<vram_amd_index<nk_i4_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_i4_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<vram_amd_index<nk_u4_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_u4_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<vram_amd_index<nk_e3m2_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_e3m2_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<vram_amd_index<nk_e2m3_k, gpu_class_t::amd_cdna4_k>>;
template __device__ void us::dense_amd_instantiate<gpu_class_t::amd_cdna4_k, nk_e2m3_k>();
#endif
