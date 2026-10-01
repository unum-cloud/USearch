/**
 *  @file bench/dense_ram_indexes.cpp
 *  @author Ash Vardanian
 *  @date August 30, 2026
 *  @brief The entry of the sharded-index CPU benchmark: the flags, the topology setup, and the
 *      dataset-scalar hop into the per-dtype shards.
 *
 *  The compile-time cascade lives in @c dense/ram_indexes.hpp; this translation unit stays
 *  dtype-free by extern-declaring the @c run_with_stored_dtype instantiations the per-dtype
 *  shards - one per stored dtype: @c f32, @c bf16, @c f16, @c e5m2, @c e4m3, @c e3m2, @c e2m3,
 *  @c i8, @c i4, @c u8, @c u4, and @c u1 - carry, so the heavy kernel grids build in parallel and
 *  this unit recompiles in seconds.
 *  No pool is created here: the executor spawns inside the run and owns its own.
 */
#include "dense/ram_indexes.hpp"

namespace unum::usearch::bench {

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_f32_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_f32_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_f32_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_f32_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_f32_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_bf16_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_bf16_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_bf16_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_bf16_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_bf16_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_f16_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_f16_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_f16_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_f16_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_f16_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e5m2_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e5m2_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e5m2_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e5m2_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e5m2_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e4m3_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e4m3_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e4m3_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e4m3_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e4m3_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e3m2_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e3m2_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e3m2_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e3m2_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e3m2_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_e2m3_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_e2m3_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_e2m3_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_e2m3_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_e2m3_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_i8_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_i8_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_i8_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_i8_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_i8_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_i4_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_i4_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_i4_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_i4_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_i4_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_u8_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_u8_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_u8_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_u8_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_u8_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_u4_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_u4_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_u4_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_u4_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_u4_k>(dense::bench_launch<nk::u8_t> const&);

extern template int dense::run_with_stored_dtype<nk::f32_t, nk_u1_k>(dense::bench_launch<nk::f32_t> const&);
extern template int dense::run_with_stored_dtype<nk::bf16_t, nk_u1_k>(dense::bench_launch<nk::bf16_t> const&);
extern template int dense::run_with_stored_dtype<nk::f16_t, nk_u1_k>(dense::bench_launch<nk::f16_t> const&);
extern template int dense::run_with_stored_dtype<nk::i8_t, nk_u1_k>(dense::bench_launch<nk::i8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u8_t, nk_u1_k>(dense::bench_launch<nk::u8_t> const&);
extern template int dense::run_with_stored_dtype<nk::u1x8_t, nk_u1_k>(dense::bench_launch<nk::u1x8_t> const&);

/** Opens the dataset with the file-native scalar and enters the typed cascade. */
template <typename dataset_scalar_type_>
int open_and_run(settings_t const& settings, fu_topology_t topology) {
    auto [dataset, status] = utilities::persisted_dataset<dataset_scalar_type_>::open(
        settings.vectors_path, settings.queries_path, settings.neighbors_path);
    if (status != status_t::success_k) {
        fmt::println("Failed to open the dataset: {}", settings.vectors_path);
        return 1;
    }
    return dispatch_stored_dtype<dataset_scalar_type_>(
        settings.stored_dtype,
        [&](auto dtype) {
            return dense::run_with_stored_dtype<dataset_scalar_type_, dtype.value>(
                dense::bench_launch<dataset_scalar_type_> {dataset, settings, topology});
        },
        [&](auto dtype) {
            fmt::println(stderr, "--stored-dtype=\"{}\" has no adapted kernels for this file", dtype_name(dtype));
            return 1;
        });
}

/** Runs the sharded-index retrieval benchmark over the flags in @p argv. */
int bench_dense_ram_indexes(int argc, char const* const* argv) {
    settings_t settings = read_settings(argc, argv, "usearch_bench_dense_ram_indexes");
    machine_t machine = probe_machine();
    if (!machine.topology || !machine.fabric) {
        fmt::println("Failed to inspect the topology");
        return 1;
    }

    // Resolve `--memory-domain` into the engaged-domain list: empty means one heap shard,
    // a single ID one pinned shard, and `all` one shard per memory domain - the list IS the
    // placement, no modes.
    if (settings.memory_domain_name == "all")
        for (std::size_t domain = 0; domain != machine.memory_domains; ++domain)
            settings.memory_domains.push_back(static_cast<fu_memory_domain_id_t>(domain));
    else if (!settings.memory_domain_name.empty()) {
        std::size_t domain = 0;
        char const* const end = settings.memory_domain_name.data() + settings.memory_domain_name.size();
        auto const [parsed_end, parse_status] = std::from_chars(settings.memory_domain_name.data(), end, domain);
        if (parse_status != std::errc {} || parsed_end != end || domain >= machine.memory_domains) {
            fmt::println(stderr, "--memory-domain=\"{}\" does not parse, expected all or a domain below {}",
                         settings.memory_domain_name, machine.memory_domains);
            return 1;
        }
        settings.memory_domains.push_back(static_cast<fu_memory_domain_id_t>(domain));
    }

    // The executor spawns inside the run; resolving its default here lets the report name it.
    if (!settings.threads.value) {
        auto const [threads, threads_status] = forkunion_executor_t::default_threads(machine.topology);
        if (failed(threads_status)) {
            fmt::println("Failed to count the logical cores");
            return 1;
        }
        settings.threads.value = threads;
    }
    print(machine);
    print(settings);

    // Huge pages are provisioned outside the process, and `--huge-pages` declares the
    // expectation, enforced against what the domains report.
    if (settings.huge_pages.value) {
        std::size_t volume = 0;
        for (std::size_t domain = 0; domain != machine.memory_domains; ++domain) {
            std::size_t domain_volume = 0;
            if (fu_volume_huge_pages_in(machine.topology, domain, &domain_volume) != fu_success_k) continue;
            volume += domain_volume;
        }
        if (!volume) {
            fmt::println("--huge-pages expects {}, but the domains report none", settings.huge_pages.value);
            return 1;
        }
    }

    // File-scalar axis: the newest hop of the compile-time cascade, lowering the file's dtype
    // to the C++ wrapper the whole run is instantiated over through NumKong's own `type_for`.
    int const exit_code = dispatch_file_dtype(
        utilities::dtype_of_extension(settings.vectors_path),
        [&](auto dtype) { return open_and_run<typename nk::type_for<dtype.value>::type>(settings, machine.topology); },
        [&](auto dtype) {
            fmt::println("Unsupported dataset dtype {}", dtype_name(dtype));
            return 1;
        });

    fu_fabric_delete(machine.fabric);
    fu_topology_delete(machine.topology);
    return exit_code;
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_dense_ram_indexes(argc, argv); }
