/**
 *  @file bench/dense_ram_index_search.cpp
 *  @author Ash Vardanian
 *  @date July 23, 2026
 *  @brief The sparse single-shard probe of the CPU-resident dense index: what only a benchmark
 *      looks inside, at f32 and bf16 storage and at the serial class or the host's newest one.
 *
 *  The benchmarks over the C interface measure what users run; this one keeps the probes they
 *  cannot reach: plan stepping, saturated walks and islands, per-query latency percentiles,
 *  immutable mode, indexer heads and managed memory. An f32 file stores as f32 or bf16, and a bf16
 *  file as bf16; any other file or stored dtype belongs to `usearch_bench_dense_index_search`.
 */
#include "dense/ram_index_search.hpp"

namespace unum::usearch::bench {

/** Runs the single-index probe over the flags in @p argv. */
int bench_dense_ram_index_search(int argc, char const* const* argv) {
    return run_cpu_bench(
        argc, argv, "usearch_bench_dense_ram_index_search",
        [&]<typename dataset_scalar_type_>(bench_launch<dataset_scalar_type_> const& launch) {
            nk_dtype_t const stored = launch.settings.stored_dtype;
            if constexpr (std::is_same_v<dataset_scalar_type_, nk::f32_t>) {
                if (stored == nk_f32_k) return dense::run_with_stored_dtype<nk::f32_t, nk_f32_k>(launch);
                if (stored == nk_bf16_k) return dense::run_with_stored_dtype<nk::f32_t, nk_bf16_k>(launch);
            }
            else if constexpr (std::is_same_v<dataset_scalar_type_, nk::bf16_t>) {
                if (stored == nk_bf16_k) return dense::run_with_stored_dtype<nk::bf16_t, nk_bf16_k>(launch);
            }
            fmt::println(stderr, "--stored-dtype=\"{}\" over this file belongs to usearch_bench_dense_index_search",
                         dtype_name(stored));
            return 1;
        });
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_dense_ram_index_search(argc, argv); }
