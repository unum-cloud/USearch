/**
 *  @file test/dense/vram_kernels_amd.hip
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief The AMD suite's device entry: the bf16 cells' kernels, compiled by HIP-Clang device-only
 *      into the suite's own code object.
 *
 *  HIP-Clang refuses a kernel's address in device code, so @c amd_instantiate takes it from a host
 *  function, which only this device-only pass compiles; @c vram_amd.hip, compiled host-only, never
 *  names them.
 */
#include "vram_amd.hip"

namespace unum::usearch::dense {

template void amd_instantiate<dense_amd_capabilities_k, nk_bf16_k>();

} // namespace unum::usearch::dense
