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
using namespace unum::usearch::test;
using namespace unum::usearch::test::dense;

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

std::size_t test::check_dense(settings_t const& settings) {
    std::size_t failures_count = 0;

    if (settings.selects("ram_metric_f64")) failures_count += dense::check_ram_metric_f64();
    if (settings.selects("ram_metric_f32")) failures_count += dense::check_ram_metric_f32();
    if (settings.selects("ram_metric_bf16")) failures_count += dense::check_ram_metric_bf16();
#if USEARCH_DENSE_DTYPES >= 2
    if (settings.selects("ram_metric_u1")) failures_count += dense::check_ram_metric_u1();
#endif
#if USEARCH_DENSE_DTYPES >= 3
    if (settings.selects("ram_metric_i8")) failures_count += dense::check_ram_metric_i8();
#endif
#if USEARCH_DENSE_DTYPES >= 4
    if (settings.selects("ram_metric_e4m3")) failures_count += dense::check_ram_metric_e4m3();
#endif
#if USEARCH_DENSE_DTYPES >= 5
    if (settings.selects("ram_metric_f16")) failures_count += dense::check_ram_metric_f16();
#endif
#if USEARCH_DENSE_DTYPES >= 6
    if (settings.selects("ram_metric_u8")) failures_count += dense::check_ram_metric_u8();
#endif
#if USEARCH_DENSE_DTYPES >= 7
    if (settings.selects("ram_metric_e5m2")) failures_count += dense::check_ram_metric_e5m2();
#endif
#if USEARCH_DENSE_DTYPES >= 8
    if (settings.selects("ram_metric_i4")) failures_count += dense::check_ram_metric_i4();
#endif
#if USEARCH_DENSE_DTYPES >= 9
    if (settings.selects("ram_metric_u4")) failures_count += dense::check_ram_metric_u4();
#endif
#if USEARCH_DENSE_DTYPES >= 10
    if (settings.selects("ram_metric_e3m2")) failures_count += dense::check_ram_metric_e3m2();
#endif
#if USEARCH_DENSE_DTYPES >= 11
    if (settings.selects("ram_metric_e2m3")) failures_count += dense::check_ram_metric_e2m3();
#endif

    if (settings.selects("ram_index_f32")) failures_count += dense::check_ram_index_f32(settings);
    if (settings.selects("ram_index_bf16")) failures_count += dense::check_ram_index_bf16(settings);
#if USEARCH_DENSE_DTYPES >= 2
    if (settings.selects("ram_index_u1")) failures_count += dense::check_ram_index_u1(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 3
    if (settings.selects("ram_index_i8")) failures_count += dense::check_ram_index_i8(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 5
    if (settings.selects("ram_index_f16")) failures_count += dense::check_ram_index_f16(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 6
    if (settings.selects("ram_index_u8")) failures_count += dense::check_ram_index_u8(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 7
    if (settings.selects("ram_index_e5m2")) failures_count += dense::check_ram_index_e5m2(settings);
#endif
#if USEARCH_DENSE_DTYPES >= 8
    if (settings.selects("ram_index_i4")) failures_count += dense::check_ram_index_i4(settings);
#endif

    if (settings.selects("ram_indexes")) failures_count += dense::check_ram_indexes(settings);

    return failures_count;
}
