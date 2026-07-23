/**
 *  @brief Smoke & concurrency tests for the CPU-resident @c inner_volatile_index over a
 *      curated set of (dtype, kernel) instantiations - not the whole combinatorial space.
 *
 *  @author Ash Vardanian
 *  @file cpp/inner_index_test.cpp
 *  @date July 17, 2026
 *
 *  The scenarios themselves live in @c inner_index_test.cuh, templated over the index type;
 *  this translation unit names the volatile instantiations, their compile-time & runtime
 *  gates, and composes the suite explicitly. The covered matrix is:
 *  f32+dot, f32+angular, f32+euclidean, u8+euclidean, u8+angular, i4+angular - each gated
 *  by a compile-time concept, so combinations whose serial kernels are not yet adapted in
 *  @c dense/ram_metric.hpp report as skipped and light up automatically once the kits land.
 *  Two already-adapted stand-ins (bf16+angular, i8+euclidean) keep the index machinery
 *  genuinely exercised in the meantime, and repeat with a deliberately flaky waiter that
 *  oversleeps at random, widening the interleaving windows around locks and watermark turns
 *  far beyond hardware timing. A u8+euclidean tiles cell exercises the sapphireamx engine
 *  end to end where the machine offers it. Removal, update, and filtering tests are
 *  deliberately absent: those features are banned at this layer by design.
 */
#include <fmt/format.h>
#include <numkong/numkong.hpp>

#include "inner_index_test.cuh"
#include "usearch/dense/ram_index.hpp"

using namespace unum::usearch;

/** @brief Gate: a (dtype, kernel) cell is testable once its serial kernel kit is adapted. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
concept index_combo_supported = requires {
    typename inner_quads_chain<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>::kit_t;
};

/** @brief No-op sibling picked when the tuple's serial kernels are not adapted yet. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_, typename waiter_type_ = dummy_waiter_t,
          nk_capability_t caps_ = nk_cap_serial_k>
    requires(!index_combo_supported<dtype_, kernel_>)
static std::size_t test_configuration(char const* name) {
    fmt::println("skipped  {:<20} : serial kernels not adapted yet", name);
    return 0;
}

template <                                       //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_, //
    typename waiter_type_ = dummy_waiter_t, nk_capability_t caps_ = nk_cap_serial_k>
    requires(index_combo_supported<dtype_, kernel_>)
static std::size_t test_configuration(char const* name) {

    // SIMD-tier cells execute their intrinsics unconditionally, so they run only where
    // the machine offers every requested bit; the tile permission handshake is per-process.
    if constexpr (caps_ != nk_cap_serial_k) {
        if ((nk_capabilities_available() & caps_) != caps_) {
            fmt::println("skipped  {:<20} : capabilities missing at runtime", name);
            return 0;
        }
        nk_configure_thread(nk_capabilities_available());
    }

    using index_t = inner_volatile_index<dtype_, kernel_, caps_, waiter_type_, counting_allocator<std::byte>,
                                         counting_allocator<std::byte>, std::allocator<std::byte>>;
    auto make_index = [](allocation_ledger_t& ledger) {
        return index_t {counting_allocator<std::byte>(ledger), counting_allocator<std::byte>(ledger)};
    };

    // The suite is composed here, not reflected: every call below is a capability claim
    // this fully-concurrent, serializable category makes about itself.
    std::size_t failures_count = 0;
    failures_count += test_guards<index_t>(make_index);
    failures_count += test_three_vectors<index_t, kernel_>(make_index);
    failures_count += test_growth_and_recall<index_t, kernel_>(make_index);
    failures_count += test_predicated_search<index_t, kernel_>(make_index);
    failures_count += test_save_load_roundtrip<index_t, kernel_>(make_index);
    failures_count += test_concurrent_build<index_t, kernel_>(make_index);
    failures_count += test_concurrent_search<index_t, kernel_>(make_index);
    return report_scenarios(name, failures_count);
}

int main() {
    std::size_t failures_count = 0;
    failures_count += test_configuration<nk_f32_k, nk_kernel_dot_k>("f32+dot");
    failures_count += test_configuration<nk_f32_k, nk_kernel_angular_k>("f32+angular");
    failures_count += test_configuration<nk_f32_k, nk_kernel_euclidean_k>("f32+euclidean");
    failures_count += test_configuration<nk_u8_k, nk_kernel_euclidean_k>("u8+euclidean");
    failures_count += test_configuration<nk_u8_k, nk_kernel_angular_k>("u8+angular");
    failures_count += test_configuration<nk_i4_k, nk_kernel_angular_k>("i4+angular");
    failures_count += test_configuration<nk_bf16_k, nk_kernel_angular_k>("bf16+angular");
    failures_count += test_configuration<nk_i8_k, nk_kernel_euclidean_k>("i8+euclidean");
    failures_count += test_configuration<nk_bf16_k, nk_kernel_angular_k, flaky_waiter_t>("bf16+angular+flaky");
    failures_count += test_configuration<nk_i8_k, nk_kernel_euclidean_k, flaky_waiter_t>("i8+euclidean+flaky");

    // The tiled u8 tier: the serial floor, icelake quads, and sapphireamx tiles.
    constexpr nk_capability_t tiles_caps_k = static_cast<nk_capability_t>( //
        nk_cap_serial_k | nk_cap_icelake_k | nk_cap_sapphireamx_k);
    failures_count += test_configuration<nk_u8_k, nk_kernel_euclidean_k, dummy_waiter_t, tiles_caps_k>( //
        "u8+euclidean+tiles");
    fmt::println("{}: {} total failures", failures_count ? "FAILED" : "OK", failures_count);
    return failures_count ? 1 : 0;
}
