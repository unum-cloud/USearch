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
#include "dense/nvme_indexes.hpp"

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
    if (settings.selects("ram_metric_f16")) failures_count += dense::check_ram_metric_f16();
    if (settings.selects("ram_metric_e5m2")) failures_count += dense::check_ram_metric_e5m2();
    if (settings.selects("ram_metric_e4m3")) failures_count += dense::check_ram_metric_e4m3();
    if (settings.selects("ram_metric_e3m2")) failures_count += dense::check_ram_metric_e3m2();
    if (settings.selects("ram_metric_e2m3")) failures_count += dense::check_ram_metric_e2m3();
    if (settings.selects("ram_metric_i8")) failures_count += dense::check_ram_metric_i8();
    if (settings.selects("ram_metric_i4")) failures_count += dense::check_ram_metric_i4();
    if (settings.selects("ram_metric_u8")) failures_count += dense::check_ram_metric_u8();
    if (settings.selects("ram_metric_u4")) failures_count += dense::check_ram_metric_u4();
    if (settings.selects("ram_metric_u1")) failures_count += dense::check_ram_metric_u1();

    if (settings.selects("ram_index_f32")) failures_count += dense::check_ram_index_f32(settings);
    if (settings.selects("ram_index_bf16")) failures_count += dense::check_ram_index_bf16(settings);
    if (settings.selects("ram_index_f16")) failures_count += dense::check_ram_index_f16(settings);
    if (settings.selects("ram_index_e5m2")) failures_count += dense::check_ram_index_e5m2(settings);
    if (settings.selects("ram_index_i8")) failures_count += dense::check_ram_index_i8(settings);
    if (settings.selects("ram_index_i4")) failures_count += dense::check_ram_index_i4(settings);
    if (settings.selects("ram_index_u8")) failures_count += dense::check_ram_index_u8(settings);
    if (settings.selects("ram_index_u1")) failures_count += dense::check_ram_index_u1(settings);

    if (settings.selects("ram_indexes")) failures_count += dense::check_ram_indexes(settings);

    if (settings.selects("nvme_indexes")) failures_count += dense::check_nvme_indexes(settings);
    if (settings.selects("nvme_rewrites")) failures_count += dense::check_nvme_rewrites(settings);

    return failures_count;
}
