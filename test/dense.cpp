/**
 *  @file test/dense.cpp
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief The dense modality's CPU composer: rolls the metric suite and the behavior cells of every
 *      CPU residency into one failure count for the suite's entry point.
 */
#include <type_traits> // `std::is_move_constructible_v` - the plans' and claims' witnesses

#include "usearch/dense/ram_index.hpp"

#include "dense/ram_indexes.hpp"
#include "dense/nvme_indexes.hpp"

using namespace unum::usearch;
using namespace unum::usearch::test;
using namespace unum::usearch::test::dense;

/** The plans and claims are move-only, and the plans are never assigned over. */
using witnessed_index_t = ram_index<>;
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

    failures_count += test::dense::check_ram_metric(settings);
    failures_count += test::dense::check_nvme_indexes(settings);
    return failures_count;
}
