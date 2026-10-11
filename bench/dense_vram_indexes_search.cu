/**
 *  @file bench/dense_vram_indexes_search.cu
 *  @author Ash Vardanian
 *  @date October 7, 2026
 *  @brief The entry of the multi-device CUDA search benchmark - one shard per device, built, refined and searched:
 *      the merge flags, and the stored-dtype hop into the per-dtype shards.
 *
 *  The sharded pipeline lives in @c dense/vram_indexes_search.cuh, and the process setup it shares with every CUDA
 *  binary in @c dense/vram_index_search.cuh; this translation unit stays dtype-free by extern-declaring the
 *  @c run_indexes_with_stored_dtype instantiations the per-dtype shards carry, so the heavy device instantiation
 *  grids build in parallel and this unit recompiles in seconds.
 */
#include <fmt/format.h>

#include "dense/vram_indexes_search.cuh"

namespace unum::usearch::bench {

extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_bf16_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_f16_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_e5m2_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_e4m3_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_e3m2_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_e2m3_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_i8_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_i4_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_u8_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_u4_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f32_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f32_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::bf16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::bf16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::f16_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::f16_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::i8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::i8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);
extern template int dense::run_indexes_with_stored_dtype<nk::u1x8_t, nk_u1_k>( //
    utilities::persisted_dataset<nk::u1x8_t> const&, settings_t const&, std::span<int const>, nk_capability_t);

/**
 *  @brief The flags only the multi-device search binary takes: the shard merge.
 *
 *  @verbatim
 *  --merge            off                  Stitch the shards into one graph after ingest
 *  --expansion-merge  --expansion-add      Candidate pool while merging
 *  @endverbatim
 */
std::array<flag_t, 2> merge_flags(settings_t& settings) noexcept {
    return {{
        {"--merge", "", "stitch the shards into one graph after ingest; single-walk queries", &settings.merge},
        {"--expansion-merge", "breadth", "merge candidate pool; inherits --expansion-add by default",
         &settings.expansion_merge},
    }};
}

/** Runs the multi-device CUDA search benchmark over the flags in @p argv. */
int bench_dense_vram_indexes_search(int argc, char const* const* argv) {
    auto const on_dataset = []<typename dataset_scalar_type_>(
                                utilities::persisted_dataset<dataset_scalar_type_> const& dataset,
                                settings_t const& settings, std::span<int const> devices, nk_capability_t max_compute,
                                cuda_address_space_t plane_space) {
        if (settings.merge == switch_t::present_k || settings.expansion_merge.value)
            return fmt::println(stderr, "--merge has no verb in the sharded index; queries scatter to every shard"), 1;
        if (plane_space == cuda_address_space_t::managed_k)
            return fmt::println(stderr, "--managed-memory pages one index's planes; every shard here stays in VRAM"), 1;
        return dispatch_stored_value<dataset_scalar_type_, nk_bf16_k, nk_f16_k, nk_e5m2_k, nk_e4m3_k, nk_e3m2_k,
                                     nk_e2m3_k, nk_i8_k, nk_i4_k, nk_u8_k, nk_u4_k, nk_u1_k>(
            settings.stored_dtype,
            [&](auto dtype) {
                return dense::run_indexes_with_stored_dtype<dataset_scalar_type_, dtype.value>(dataset, settings,
                                                                                               devices, max_compute);
            },
            [&](auto dtype) {
                fmt::println(stderr, "--stored-dtype=\"{}\" has no device kits for this file", dtype_name(dtype));
                return 1;
            });
    };
    return dense::run_cuda_bench(argc, argv, "usearch_bench_dense_vram_indexes_search", on_dataset, &merge_flags);
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_dense_vram_indexes_search(argc, argv); }
