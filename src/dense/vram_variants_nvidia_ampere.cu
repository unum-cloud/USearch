/**
 *  @file src/dense/vram_variants_nvidia_ampere.cu
 *  @author Ash Vardanian
 *  @date September 26, 2026
 *  @brief The nvidia_ampere unit: the floor kits, compiled for "80" alone with the PTX every GPU without a
 *      class of its own runs.
 */
#include "dense_specializations.cuh"

template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_bf16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_f16_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_f16_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_f16_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e5m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e4m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e3m2_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_e2m3_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_i8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_i8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_i8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_i4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_i4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_i4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_u8_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_u8_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_u8_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_u4_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_u4_k, nk_kernel_angular_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_u4_k, nk_kernel_euclidean_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_u1_k, nk_kernel_dot_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_u1_k, nk_kernel_hamming_k, gpu_class_t::nvidia_ampere_k>>;
template struct specialized_index<vram_index<nk_u1_k, nk_kernel_jaccard_k, gpu_class_t::nvidia_ampere_k>>;
