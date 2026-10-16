/**
 *  @file test/dense/vram_kernels_amd.hip
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The AMD suite's device entry: every cell's kernels and attends, compiled by HIP-Clang
 *      device-only into the suite's own code object.
 *
 *  HIP-Clang refuses a kernel's address in device code, so @c amd_instantiate takes it from a host
 *  function, which only this device-only pass compiles; @c vram_amd.hip, compiled host-only, never
 *  names them.
 */
#include "vram_amd.hip"

namespace unum::usearch::dense {

// Every cell's kernels and attends, and the keyed attends of the sharded suite's collection.
template void amd_instantiate<dense_amd_capabilities_k, nk_bf16_k, hip_test_lookup_t::device_lookup_k,
                              hip_test_keyed_lookup_t::device_lookup_k>();
#if USEARCH_DENSE_DTYPES >= 2
template void amd_instantiate<dense_amd_capabilities_k, nk_u1_k, hip_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 3
template void amd_instantiate<dense_amd_capabilities_k, nk_i8_k, hip_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 4
template void amd_instantiate<dense_amd_capabilities_k, nk_e4m3_k, hip_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 5
template void amd_instantiate<dense_amd_capabilities_k, nk_f16_k, hip_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 6
template void amd_instantiate<dense_amd_capabilities_k, nk_u8_k, hip_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 7
template void amd_instantiate<dense_amd_capabilities_k, nk_e5m2_k, hip_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 8
template void amd_instantiate<dense_amd_capabilities_k, nk_i4_k, hip_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 9
template void amd_instantiate<dense_amd_capabilities_k, nk_u4_k, hip_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 10
template void amd_instantiate<dense_amd_capabilities_k, nk_e3m2_k, hip_test_lookup_t::device_lookup_k>();
#endif
#if USEARCH_DENSE_DTYPES >= 11
template void amd_instantiate<dense_amd_capabilities_k, nk_e2m3_k, hip_test_lookup_t::device_lookup_k>();
#endif

} // namespace unum::usearch::dense
