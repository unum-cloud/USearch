/**
 *  @file src/dense/vram_variants_nvidia_ada.cu
 *  @author Ash Vardanian
 *  @date September 26, 2026
 *  @brief The nvidia_ada unit: its arms inline the fp8-conversion kits, so the unit compiles for "89" alone.
 *
 *  Compiled twice: by NVCC device-only, over every compiled dtype, into the class's fatbin, and by
 *  the host compiler into the arms, which find that fatbin linked in beside them.
 */
#include "dense_specializations.cuh"

/** The class's fatbin, linked in as raw bytes beside the arms. */
extern "C" us::utilities::byte_t const dense_nvidia_ada_bundle_start[], dense_nvidia_ada_bundle_end[];

#if defined(__CUDACC__)
#include "usearch/dense/vram_nvidia_index.cuh"

inline constexpr nk_capability_t ada_capabilities_k = numkong_capabilities_of(gpu_class_t::nvidia_ada_k);
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_bf16_k, nvidia_attend_lookup_t::device_lookup_k>();
#if USEARCH_DENSE_DTYPES >= 2
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_u1_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 3
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_i8_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 4
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_e4m3_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 5
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_f16_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 6
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_u8_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 7
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_e5m2_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 8
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_i4_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 9
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_u4_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 10
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_e3m2_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 11
template void us::dense::cuda_instantiate<ada_capabilities_k, nk_e2m3_k, nvidia_attend_lookup_t::device_lookup_k>();
#endif
#else

template <>
us::utilities::bytes_view_t us::dense::cuda_fatbin<numkong_capabilities_of(gpu_class_t::nvidia_ada_k)>() noexcept {
    return {dense_nvidia_ada_bundle_start,
            static_cast<std::size_t>(dense_nvidia_ada_bundle_end - dense_nvidia_ada_bundle_start)};
}

template struct specialized_index<vram_nvidia_indexes<nk_bf16_k, gpu_class_t::nvidia_ada_k>>;
#if USEARCH_DENSE_DTYPES >= 2
template struct specialized_index<vram_nvidia_indexes<nk_u1_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 3
template struct specialized_index<vram_nvidia_indexes<nk_i8_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 4
template struct specialized_index<vram_nvidia_indexes<nk_e4m3_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 5
template struct specialized_index<vram_nvidia_indexes<nk_f16_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 6
template struct specialized_index<vram_nvidia_indexes<nk_u8_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 7
template struct specialized_index<vram_nvidia_indexes<nk_e5m2_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 8
template struct specialized_index<vram_nvidia_indexes<nk_i4_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 9
template struct specialized_index<vram_nvidia_indexes<nk_u4_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 10
template struct specialized_index<vram_nvidia_indexes<nk_e3m2_k, gpu_class_t::nvidia_ada_k>>;
#endif
#if USEARCH_DENSE_DTYPES >= 11
template struct specialized_index<vram_nvidia_indexes<nk_e2m3_k, gpu_class_t::nvidia_ada_k>>;
#endif
#endif // defined(__CUDACC__)
