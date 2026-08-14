/**
 *  @brief The dense modality's CPU composer: rolls every per-dtype metric & index shard
 *      into one failure count for the suite's entry point.
 *  @author Ash Vardanian
 *  @file test/dense.cpp
 *  @date August 14, 2026
 *
 *  The shards are named here, never reflected: each call is a coverage claim the modality
 *  makes about itself, and a dtype that stops building stops being counted loudly.
 */
#include "dense/ram_metric.hpp"
#include "dense/ram_index.hpp"

std::size_t test_dense_all() {
    std::size_t failures_count = 0;

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

    return failures_count;
}
