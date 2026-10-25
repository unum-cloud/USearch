/**
 *  @file test/dense/ram_i8.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The i8 cells of the CPU-resident dense residency: the metric and the sharded suite under
 *      Icelake's kits.
 *
 *  Icelake's i8 finalize subtracts the row sums, so even a dot index saves its norms column.
 *
 *  CMake compiles the unit only when @c USEARCH_DENSE_DTYPES keeps i8, and the unit registers
 *  itself with the runner, which names no dtype.
 */
#include "ram_indexes.hpp"
#include "ram_metric.hpp"

namespace unum::usearch::test::dense {

static std::size_t check_ram_i8(settings_t const& settings, gpu_device_t const&) {
    if (!settings.filter.selects("ram_i8")) return 0;
    std::size_t failures_count = check_named_scalar<nk::i8_t>(settings, "ram_i8 metric");
    failures_count +=
        test_ram_indexes_configuration<nk_i8_k, nk_kernel_dot_k, fu::standard_yield_t, cpu_class_t::intel_icelake_k>(
            "ram_i8 sharded dot+icelake", settings.seed);
    return failures_count;
}

static bool const ram_i8_registered = (registered_suites({gpu_vendor_t::unknown_k, &check_ram_i8}), true);

} // namespace unum::usearch::test::dense
