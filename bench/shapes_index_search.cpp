/**
 *  @file bench/shapes_index_search.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The shapes search benchmark: point clouds end to end and their point counts, built into
 *      one RAM or VRAM index through the C interface and judged against an exact scan.
 *
 *  Points are stored in the scalar the file holds, so the oracle packs exactly what the index
 *  stores. A VRAM index places one shard on every device @c USEARCH_DEVICES names, device zero
 *  when unset, and lands its matches in a host arena, as a C caller reading them would. An NVMe
 *  run builds in RAM, saves a checkpoint to the checkpoints directory and searches a view of it.
 *  A named checkpoint is stamped with the stored scalar, the metric and the connectivity, and
 *  several shards save a `.safegraphs.json` manifest beside a monolith apiece.
 *
 *  @verbatim
 *  --base-points      required                 Every shape's points end to end, `.fbin` or `.f16bin` with 3 columns
 *  --base-counts      required                 Points per shape, `.ibin` with 1 column
 *  --residency        ram                      Where the index lives: ram, vram or nvme
 *  --metric           chamfer                  chamfer, hausdorff, gromov_wasserstein or wasserstein_procrustes
 *  --save             none                     Checkpoint name to serialize the index into after ingestion
 *  --load             none                     Checkpoint name to restore, then insert the entries past it
 *  --checkpoints-dir  the temporary directory  Directory for checkpoint blobs
 *  @endverbatim
 */
#include <cstdint> // `std::uint8_t` - the checkpoint path's bytes, `std::int32_t` - the device ordinals
#include <cstdio>  // `std::remove` - the checkpoint an NVMe run saved

#include <array>       // `std::array` - the kinds, and one round of the exact scan
#include <filesystem>  // `std::filesystem::temp_directory_path` - the default checkpoints directory
#include <limits>      // `std::numeric_limits` - the exact scan's open radius
#include <string>      // `std::string` - the checkpoint path
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector` - the devices a VRAM index spans

#include "usearch/shapes_index.h"
#include "usearch/shapes/ram_metric.hpp"
#include "usearch/utilities/blocking_files.hpp" // `mapped_file_t`

#include "basic.cuh"
#include "utilities/persisted_dataset.hpp" // `parse_matrix_header`

namespace unum::usearch::bench {

/** Entries per round of the exact scan. */
constexpr std::size_t shapes_round_k = 1024;

/** One shapes kind: its checkpoint token, its C enumerator, and the NumKong kernel scoring it. */
struct shapes_kind_t {
    std::string_view token;
    usearch_shapes_metric_t metric;
    nk_kernel_kind_t kernel;
};

inline constexpr std::array<shapes_kind_t, 4> shapes_kinds_k {{
    {"chamfer", usearch_shapes_metric_chamfer_k, nk_kernel_chamfers_packed_k},
    {"hausdorff", usearch_shapes_metric_hausdorff_k, nk_kernel_hausdorffs_packed_k},
    {"gromov_wasserstein", usearch_shapes_metric_gromov_wasserstein_k, nk_kernel_gromovwassersteins_packed_k},
    {"wasserstein_procrustes", usearch_shapes_metric_wasserstein_procrustes_k,
     nk_kernel_wassersteinprocrustes_packed_k},
}};

/** The checkpoint flags: the names a run saves into and restores from, and where both resolve. */
struct shapes_checkpoints_t {

    /** Checkpoint name to serialize the index into after ingestion. */
    std::string_view save_name;

    /** Checkpoint name to restore in place of an empty index, then insert the entries past it. */
    std::string_view load_name;

    /** Directory the checkpoint blobs land in; empty takes the temporary directory. */
    std::string_view checkpoints_dir;
};

/** The C interface's view of a checkpoint @p path. */
inline usearch_bytes_view_t path_view_of(std::string const& path) noexcept {
    return {reinterpret_cast<std::uint8_t const*>(path.data()), path.size()};
}

/** Packs the shape at @p row of @p coordinates, bounded by @p offsets, into @p packed. */
template <typename scalar_type_>
status_t pack_shape(std::span<scalar_type_ const> coordinates, std::span<std::size_t const> offsets, std::size_t row,
                    nk::shapes_packed<scalar_type_>& packed) {
    std::array const clouds {coordinates.subspan(offsets[row] * 3, (offsets[row + 1] - offsets[row]) * 3)};
    auto [made, status] = nk::shapes_packed<scalar_type_>::make(clouds, {}, 0);
    if (nk::failed(status)) return shapes::status_of(status);
    packed = std::move(made);
    return status_t::success_k;
}

/** Builds, searches and judges an index over the shapes in @p points_file, which hold
 *  @p scalar_type_ points, cut by the counts in @p counts_file; a VRAM one spans the devices of
 *  @p vram. */
template <typename scalar_type_>
int run_shapes(basic_settings_t const& settings, shapes_kind_t const& kind, std::string_view residency,
               usearch_vram_options_t const& vram, shapes_checkpoints_t const& checkpoints, bytes_view_t points_file,
               bytes_view_t counts_file) {
    auto const [points, points_parse_status] = utilities::parse_matrix_header<scalar_type_, u32_t>(points_file);
    if (failed(points_parse_status) || points.columns != 3)
        return fmt::println(stderr, "--base-points must hold 3 columns per point"), 1;
    auto const [counts, counts_parse_status] = utilities::parse_matrix_header<u32_t, u32_t>(counts_file);
    if (failed(counts_parse_status) || counts.columns != 1)
        return fmt::println(stderr, "--base-counts must hold 1 column per shape"), 1;
    std::span<scalar_type_ const> const coordinates = elements_of<scalar_type_>(
        points_file.subspan(2 * sizeof(u32_t), points.rows * 3 * sizeof(scalar_type_)));
    std::span<u32_t const> const points_per_shape = elements_of<u32_t>(
        counts_file.subspan(2 * sizeof(u32_t), counts.rows * sizeof(u32_t)));

    bool constexpr half_k = std::is_same_v<scalar_type_, nk::f16_t>;
    tape_t tape;
    tape.elements = coordinates.data();
    tape.scalar = half_k ? usearch_scalar_f16_k : usearch_scalar_f32_k;
    tape.offsets.reserve(points_per_shape.size() + 1);
    for (u32_t const count : points_per_shape) tape.offsets.push_back(tape.offsets.back() + count);
    if (tape.offsets.back() != points.rows)
        return fmt::println(stderr, "--base-counts sum to {} points, --base-points holds {}", tape.offsets.back(),
                            points.rows),
               1;

    // A named checkpoint carries what its payload depends on, and several shards save a manifest.
    std::error_code folder_error;
    std::filesystem::path const folder = checkpoints.checkpoints_dir.empty()
                                             ? std::filesystem::temp_directory_path(folder_error)
                                             : std::filesystem::path(checkpoints.checkpoints_dir);
    bool const sharded = residency == "vram" && vram.devices_count > 1;
    auto const checkpoint_of = [&](std::string_view name) {
        return (folder / fmt::format("{}-{}-{}-c{}.safegraphs{}", name, half_k ? "f16" : "f32", kind.token,
                                     settings.connectivity.value, sharded ? ".json" : ""))
            .string();
    };

    usearch_outcome_t outcome {};
    usearch_shapes_index_t index = nullptr;
    std::string const loaded_path = checkpoints.load_name.empty() ? std::string {}
                                                                  : checkpoint_of(checkpoints.load_name);
    if (!loaded_path.empty()) {
        usearch_executor_t loading = nullptr;
        if (residency == "vram") {
            usearch_executor_for_vram_t loading_arguments {.executor = &loading, .outcome = &outcome, .vram = vram};
            usearch_executor_for_vram(&loading_arguments);
            usearch_shapes_load_vram_t load {.index = &index,
                                             .executor = loading,
                                             .outcome = &outcome,
                                             .path = path_view_of(loaded_path),
                                             .vram = vram};
            if (!outcome.message) usearch_shapes_load_vram(&load);
        }
        else {
            usearch_executor_for_ram_t loading_arguments {.executor = &loading,
                                                          .outcome = &outcome,
                                                          .threads = settings.threads.value};
            usearch_executor_for_ram(&loading_arguments);
            usearch_shapes_load_ram_t load {.index = &index,
                                            .executor = loading,
                                            .outcome = &outcome,
                                            .path = path_view_of(loaded_path)};
            if (!outcome.message) usearch_shapes_load_ram(&load);
        }
        usearch_executor_free(loading);
    }
    else if (residency == "vram") {
        usearch_shapes_init_vram_t init {.index = &index,
                                         .outcome = &outcome,
                                         .metric = kind.metric,
                                         .quantization = tape.scalar,
                                         .connectivity = settings.connectivity.value,
                                         .vram = vram};
        usearch_shapes_init_vram(&init);
    }
    else {
        usearch_shapes_init_ram_t init {.index = &index,
                                        .outcome = &outcome,
                                        .metric = kind.metric,
                                        .quantization = tape.scalar,
                                        .connectivity = settings.connectivity.value};
        usearch_shapes_init_ram(&init);
    }
    if (outcome.message)
        return fmt::println(stderr, "{}: {}", loaded_path.empty() ? "Init" : "Load", outcome.message), 1;
    usearch_shapes_index_description_t description {};
    usearch_shapes_describe_t describe {.index = index, .outcome = &outcome, .description = &description};
    usearch_shapes_describe(&describe);
    fmt::println("- Metric: {} · scalar: {} · residency: {} · class: {}", kind.token, half_k ? "f16" : "f32", residency,
                 description.capabilities);

    // A restored checkpoint holds the first base entries, so the build inserts only those past it.
    std::size_t const loaded = description.size;
    std::size_t const before_queries = tape.size() - std::min(tape.size(), settings.queries.value);
    std::size_t const base_entries = settings.max_base_entries.value
                                         ? std::min(settings.max_base_entries.value, before_queries)
                                         : before_queries;
    if (loaded > base_entries)
        return fmt::println(stderr, "{} holds {} entries, past the {} base ones", loaded_path, loaded, base_entries), 1;
    if (loaded) fmt::println("- Loaded: {} entries from {}", humanized(loaded), loaded_path);

    auto const insert = [&](usearch_executor_t executor, usearch_keys_view_t keys, usearch_tape_view_t entries,
                            std::size_t count, usearch_traversal_stats_t* stats, usearch_outcome_t* reached) {
        std::size_t const first = static_cast<std::size_t>(entries.offsets - tape.offsets.data());
        std::size_t const skipped = std::min(count, loaded - std::min(loaded, first));
        if (skipped == count) return;
        keys.data += skipped, entries.offsets += skipped;
        usearch_shapes_insert_t arguments {.index = index,
                                           .executor = executor,
                                           .outcome = reached,
                                           .count = count - skipped,
                                           .keys = keys,
                                           .shapes = entries,
                                           .expansion_factor = settings.expansion_add.value,
                                           .stats = stats};
        usearch_shapes_insert(&arguments);
    };
    auto const search = [&](usearch_executor_t executor, usearch_arena_t arena, usearch_tape_view_t queries,
                            std::size_t count, std::size_t expansion, usearch_matches_view_t* matches,
                            usearch_traversal_stats_t* stats, usearch_outcome_t* reached) {
        usearch_shapes_search_t arguments {.index = index,
                                           .executor = executor,
                                           .outcome = reached,
                                           .count = count,
                                           .queries = queries,
                                           .matches = matches,
                                           .arena = arena,
                                           .wanted = settings.top_k.value,
                                           .expansion_factor = expansion,
                                           .stats = stats};
        usearch_shapes_search(&arguments);
    };

    using kit_t = typename shapes::shapes_ladder<scalar_type_, nk_cap_any_k>::kit_t;
    using distance_t = typename kit_t::distance_t;

    /** One thread's exact scan: its metric, the round's packs, the query's pack and the round's
     *  distances. */
    using oracle_metric_t = shapes::shapes_metric<kit_t, std::allocator<byte_t>>;
    struct oracle_t {
        oracle_metric_t metric;
        std::vector<nk::shapes_packed<scalar_type_>> packs;
        nk::shapes_packed<scalar_type_> query_pack;
        std::array<distance_t, shapes_round_k> round;
    };
    std::vector<oracle_t> oracles;
    oracles.reserve(settings.threads.value);
    for (std::size_t thread = 0; thread != settings.threads.value; ++thread) {
        oracles.push_back(
            {oracle_metric_t {kind.kernel, {}}, std::vector<nk::shapes_packed<scalar_type_>>(shapes_round_k), {}, {}});
        if (failed(oracles.back().metric.init(shapes_round_k)))
            return fmt::println(stderr, "The exact scan could not size its rounds"), 1;
    }
    auto const exact = [&](std::size_t thread, std::size_t query, std::span<double> distances) {
        auto& [oracle, packs, query_pack, round] = oracles[thread];
        bool packed = succeeded(pack_shape(coordinates, std::span {tape.offsets}, query, query_pack));
        if (packed) oracle.reset(query_pack);
        for (std::size_t first = 0; first < distances.size(); first += shapes_round_k) {
            std::size_t const count = std::min(shapes_round_k, distances.size() - first);
            for (std::size_t slot = 0; packed && slot != count; ++slot) {
                packed = succeeded(pack_shape(coordinates, std::span {tape.offsets}, first + slot, packs[slot]));
                if (packed) static_cast<void>(oracle.push(packs[slot]));
            }
            if (packed) oracle.drain(std::span {round}.first(count), (std::numeric_limits<distance_t>::max)());
            for (std::size_t slot = 0; slot != count; ++slot)
                distances[first + slot] = packed ? static_cast<double>(round[slot])
                                                 : (std::numeric_limits<double>::max)();
        }
    };

    // A named save keeps what the run built; an NVMe run saves it too and searches a view of the
    // checkpoint, on storage threads.
    std::string const checkpoint = checkpoints.save_name.empty() ? (folder / "usearch-bench-shapes.safegraphs").string()
                                                                 : checkpoint_of(checkpoints.save_name);
    usearch_bytes_view_t const checkpoint_view = path_view_of(checkpoint);
    usearch_executor_t storage = nullptr;
    auto const serve = [&](usearch_executor_t executor, usearch_outcome_t* reached) {
        if (residency != "nvme" && checkpoints.save_name.empty()) return executor;
        usearch_shapes_save_t save {.index = index, .executor = executor, .outcome = reached, .path = checkpoint_view};
        usearch_shapes_save(&save);
        if (reached->message) return executor;
        fmt::println("- Saved: {}", checkpoint);
        if (residency != "nvme") return executor;
        usearch_shapes_free(index);
        index = nullptr;
        usearch_executor_for_nvme_t storage_arguments {.executor = &storage,
                                                       .outcome = reached,
                                                       .threads = settings.threads.value};
        usearch_executor_for_nvme(&storage_arguments);
        if (reached->message) return executor;
        usearch_shapes_view_nvme_t view {.index = &index,
                                         .executor = storage,
                                         .outcome = reached,
                                         .path = checkpoint_view};
        usearch_shapes_view_nvme(&view);
        return storage;
    };
    int const exit_code = run_basic_bench<usearch_traversal_stats_t>(
        settings, tape, residency == "vram" ? &vram : nullptr, insert, search, exact, serve);
    usearch_shapes_free(index);
    usearch_executor_free(storage);
    if (residency == "nvme" && checkpoints.save_name.empty()) std::remove(checkpoint.c_str());
    return exit_code;
}

/** Runs the shapes search benchmark over the flags in @p argv. */
int bench_shapes_index_search(int argc, char const* const* argv) {
    basic_settings_t settings;
    std::string_view base_points_path, base_counts_path, residency = "ram";
    shapes_checkpoints_t checkpoints;
    std::array<flag_t, 6> const flags {{
        {"--base-points", "path", "every shape's points end to end, `.fbin` or `.f16bin` with 3 columns",
         &base_points_path},
        {"--base-counts", "path", "points per shape, `.ibin` with 1 column", &base_counts_path},
        {"--residency", "name", "where the index lives: ram, vram on the USEARCH_DEVICES GPUs, or nvme; ram by default",
         &residency},
        {"--save", "name", "checkpoint name to serialize the index into after ingestion", &checkpoints.save_name},
        {"--load", "name", "checkpoint name to restore, then insert the entries past it", &checkpoints.load_name},
        {"--checkpoints-dir", "dir", "directory for checkpoint blobs; the temporary directory by default",
         &checkpoints.checkpoints_dir},
    }};
    read_basic_settings(argc, argv,
                        "usearch_bench_shapes_index_search --base-points <path> --base-counts <path> [flags]", settings,
                        flags);

    std::string_view const metric_name = settings.metric_name.empty() ? "chamfer" : settings.metric_name;
    shapes_kind_t const* kind = nullptr;
    for (shapes_kind_t const& candidate : shapes_kinds_k)
        if (candidate.token == metric_name) kind = &candidate;
    if (!kind) return fmt::println(stderr, "--metric=\"{}\" names no shapes kind", metric_name), 1;
    if (residency != "ram" && residency != "vram" && residency != "nvme")
        return fmt::println(stderr, "--residency=\"{}\" does not parse, expected ram, vram or nvme", residency), 1;
    counts_t const named = env_devices();
    std::vector<std::int32_t> const devices(named.view().begin(), named.view().end());
    usearch_vram_options_t const vram {.devices = devices.data(), .devices_count = devices.size()};
    auto [points_file, points_read_status] = mapped_file_t::read(base_points_path);
    if (failed(points_read_status))
        return fmt::println(stderr, "--base-points=\"{}\" could not be read", base_points_path), 1;
    auto [counts_file, counts_read_status] = mapped_file_t::read(base_counts_path);
    if (failed(counts_read_status))
        return fmt::println(stderr, "--base-counts=\"{}\" could not be read", base_counts_path), 1;
    if (base_points_path.ends_with(".fbin"))
        return run_shapes<nk::f32_t>(settings, *kind, residency, vram, checkpoints, points_file.view(),
                                     counts_file.view());
    if (base_points_path.ends_with(".f16bin"))
        return run_shapes<nk::f16_t>(settings, *kind, residency, vram, checkpoints, points_file.view(),
                                     counts_file.view());
    return fmt::println(stderr, "--base-points=\"{}\" does not parse, expected a .fbin or .f16bin path",
                        base_points_path),
           1;
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_shapes_index_search(argc, argv); }
