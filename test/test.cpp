/**
 *  @brief The suite's one entry point: composes every per-family, per-dtype shard
 *      explicitly and rolls their failure counts into the exit code.
 *  @author Ash Vardanian
 *  @file test/test.cpp
 *  @date July 23, 2026
 */
#include <fmt/format.h>

#include "test.hpp"

int main() {
    std::size_t failures_count = 0;

    failures_count += test_ram_queue();

    failures_count += test_dense_metric_f64();
    failures_count += test_dense_metric_f32();
    failures_count += test_dense_metric_bf16();
    failures_count += test_dense_metric_f16();
    failures_count += test_dense_metric_e5m2();
    failures_count += test_dense_metric_e4m3();
    failures_count += test_dense_metric_e3m2();
    failures_count += test_dense_metric_e2m3();
    failures_count += test_dense_metric_i8();
    failures_count += test_dense_metric_i4();
    failures_count += test_dense_metric_u8();
    failures_count += test_dense_metric_u4();
    failures_count += test_dense_metric_u1();

    failures_count += test_dense_index_f32();
    failures_count += test_dense_index_u8();
    failures_count += test_dense_index_i8();
    failures_count += test_dense_index_bf16();
    failures_count += test_dense_index_i4();

    fmt::println("{}: {} total failures", failures_count ? "FAILED" : "OK", failures_count);
    return failures_count ? 1 : 0;
}
