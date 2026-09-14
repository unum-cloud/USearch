/**
 *  @file test/dense.cpp
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The dense modality's CPU composer: rolls every per-dtype metric & index shard into one
 *      failure count for the suite's entry point.
 *
 *  The shards are named here, never reflected: each call is a coverage claim the modality
 *  makes about itself, and a dtype that stops building stops being counted loudly.
 */
#include <type_traits> // `std::is_move_constructible_v` - the plans' and claims' witnesses

#include "dense/ram_metric.hpp"
#include "dense/ram_index.hpp"
#include "dense/ram_indexes.hpp"

using namespace unum::usearch;

/** The plans and claims are move-only, and the plans are never assigned over. */
using witnessed_index_t = dense_ram_index<>;
static_assert(std::is_move_constructible_v<witnessed_index_t::save_plan_t> &&
              !std::is_copy_constructible_v<witnessed_index_t::save_plan_t> &&
              !std::is_copy_assignable_v<witnessed_index_t::save_plan_t> &&
              !std::is_move_assignable_v<witnessed_index_t::save_plan_t>);
static_assert(std::is_move_constructible_v<witnessed_index_t::load_plan_t> &&
              !std::is_copy_constructible_v<witnessed_index_t::load_plan_t> &&
              !std::is_copy_assignable_v<witnessed_index_t::load_plan_t> &&
              !std::is_move_assignable_v<witnessed_index_t::load_plan_t>);
static_assert(std::is_move_constructible_v<witnessed_index_t::capacity_claim_t> &&
              !std::is_copy_constructible_v<witnessed_index_t::capacity_claim_t>);

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
    failures_count += test_dense_index_bf16();
    failures_count += test_dense_index_f16();
    failures_count += test_dense_index_e5m2();
    failures_count += test_dense_index_i8();
    failures_count += test_dense_index_i4();
    failures_count += test_dense_index_u8();
    failures_count += test_dense_index_u1();

    failures_count += test_dense_indexes();

    return failures_count;
}
