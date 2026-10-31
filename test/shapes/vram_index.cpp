/**
 *  @file test/shapes/vram_index.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The shapes VRAM cell: Chamfer and Hausdorff on an NVIDIA GPU beside the RAM index on the
 *      same clouds, judged by recall against exact neighbors, every reported distance against
 *      NumKong's serial kernels.
 *
 *  Collections hold 10 to 60 points a cloud: random points; coincident points, which pack to one
 *  point and tie every pair at zero; collinear points, whose packs degenerate on two axes; and
 *  molecule-like clouds, eight jittered prototypes, one in four mirrored, so neighbors cluster by
 *  prototype. Shapes insert as packs in three batches whose tapes start an alignment line past
 *  their own first byte, as an Arrow slice's do, so the later batches walk graphs whose reverse
 *  links earlier batches merged.
 */
#include <algorithm> // `std::nth_element`
#include <array>     // `std::array` - the prototypes, one cloud a pack takes
#include <limits>    // `std::numeric_limits` - the node-ID type's sentinel
#include <span>      // `std::span`
#include <vector>    // `std::vector` - clouds, the tape, offsets, matches

#include "usearch/shapes_index.h"
#include "usearch/shapes/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/shapes/vram_nvidia_index.hpp"
#endif

#include "harness.hpp"

#if USEARCH_ARCH_CUDA_

/** The library's fatbin of one class's shapes kits over one dtype, defined in that class's
 *  @c src/shapes/vram_variants_nvidia*.cpp unit. */
template <us::utilities::gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct shapes_nvidia_fatbin {
    static us::utilities::bytes_view_t bytes() noexcept;
};

#endif

namespace unum::usearch::test::shapes {
using namespace unum::usearch::shapes;

#if USEARCH_ARCH_CUDA_

/** How a collection draws its clouds, as the file's docblock describes. */
enum class cloud_kind_t { random_k, coincident_k, collinear_k, molecular_k };

/** Draws @p count clouds of @p kind, @b [points,3] interleaved coordinates each. */
static std::vector<std::vector<f64_t>> generate_clouds(cloud_kind_t kind, std::size_t count, u64_t seed) {
    constexpr std::size_t min_points_k = 10, max_points_k = 60, prototypes_k = 8, jittered_points_k = 4;
    splitmix64_t random_generator {seed};
    uniform_between<f64_t> const centered {-1, 1};
    uniform_below<u32_t> const points_distribution {max_points_k - min_points_k + 1};
    uniform_below<u32_t> const prototype_distribution {prototypes_k};
    uniform_below<u32_t> const mirror_distribution {4};
    std::array<std::vector<f64_t>, prototypes_k> prototypes;
    for (std::vector<f64_t>& prototype : prototypes) {
        prototype.resize(3 * (min_points_k + points_distribution(random_generator)));
        for (f64_t& coordinate : prototype) coordinate = 2 * centered(random_generator);
    }
    std::vector<std::vector<f64_t>> clouds(count);
    for (std::vector<f64_t>& cloud : clouds) {
        std::size_t const points = min_points_k + points_distribution(random_generator);
        std::array<f64_t, 3> const origin {centered(random_generator), centered(random_generator),
                                           centered(random_generator)};
        std::array<f64_t, 3> const direction {centered(random_generator), centered(random_generator),
                                              centered(random_generator)};
        switch (kind) {
        case cloud_kind_t::random_k:
            cloud.resize(points * 3);
            for (f64_t& coordinate : cloud) coordinate = 4 * centered(random_generator);
            break;
        case cloud_kind_t::coincident_k:
            for (std::size_t point = 0; point != points; ++point)
                cloud.insert(cloud.end(), origin.begin(), origin.end());
            break;
        case cloud_kind_t::collinear_k:
            for (std::size_t point = 0; point != points; ++point) {
                f64_t const step = 4 * centered(random_generator);
                for (std::size_t axis = 0; axis != 3; ++axis) cloud.push_back(origin[axis] + step * direction[axis]);
            }
            break;
        case cloud_kind_t::molecular_k: {
            cloud = prototypes[prototype_distribution(random_generator)];
            uniform_below<u32_t> const point_distribution {static_cast<u32_t>(cloud.size() / 3)};
            for (std::size_t jitter = 0; jitter != jittered_points_k; ++jitter)
                for (std::size_t axis = 0, point = point_distribution(random_generator); axis != 3; ++axis)
                    cloud[point * 3 + axis] += 0.15 * centered(random_generator);
            f64_t const mirror = mirror_distribution(random_generator) == 0 ? -1 : 1;
            for (std::size_t point = 0; point != cloud.size() / 3; ++point) {
                cloud[point * 3] *= mirror;
                for (std::size_t axis = 0; axis != 3; ++axis) cloud[point * 3 + axis] += 4 * origin[axis];
            }
            break;
        }
        }
    }
    return clouds;
}

/** Matches, of @p wanted at most, whose distances reach no further than the @p wanted -th exact
 *  one, within the float tolerance: ties make the IDs ambiguous, the distances never. */
template <typename distance_type_>
static std::size_t matched_count(std::span<f64_t const> found, std::vector<distance_type_> exact, std::size_t wanted) {
    std::nth_element(exact.begin(), exact.begin() + (wanted - 1), exact.end());
    f64_t const reach = static_cast<f64_t>(exact[wanted - 1]);
    std::size_t matched = 0;
    for (f64_t const distance : found) matched += distance <= reach + 1e-3 * std::fmax(reach, 1.0);
    return min_of(matched, wanted);
}

/**
 *  @brief One collection of @p count clouds of @p kind and @p queries_count held-out queries of
 *      @p scalar_type_, built on the GPU under the kits of @p gpu_class_, whose capabilities are
 *      @p capabilities_, and in RAM under the kits of @p kernel alike.
 *
 *  Every match the GPU reports must carry the pair's serial distance, ascending, and the GPU's
 *  recall at ten must reach the RAM index's at the same breadth, give or take a twentieth.
 */
template <typename scalar_type_, gpu_class_t gpu_class_, nk_capability_t capabilities_>
static std::size_t check_shapes_vram_collection(u64_t seed, cuda_executor_t& executor, nk_kernel_kind_t kernel,
                                                cloud_kind_t kind, std::size_t count, std::size_t queries_count) {
    using vram_index_t =
        vram_nvidia_index<scalar_type_, u32_t, capabilities_, shapes_nvidia_fatbin<gpu_class_, scalar_type_::dtype()>>;
    using kit_t = typename shapes_ladder<scalar_type_, nk_cap_any_k>::kit_t;
    using serial_kit_t = typename shapes_ladder<scalar_type_, nk_cap_serial_k>::kit_t;
    using ram_index_t = ram_index<kit_t>;
    using value_t = typename ram_index_t::value_t;
    using distance_t = typename kit_t::distance_t;
    constexpr std::size_t wanted_k = 10, expansion_k = 64, batches_k = 3;
    std::size_t failures = 0;

    std::vector<std::vector<f64_t>> const clouds = generate_clouds(kind, count + queries_count, seed);
    std::vector<std::vector<scalar_type_>> stored(clouds.size());
    std::vector<std::size_t> offsets {nk_default_alignment_k};
    for (std::size_t entry = 0; entry != clouds.size(); ++entry) {
        for (f64_t const coordinate : clouds[entry]) stored[entry].push_back(scalar_type_(coordinate));
        offsets.push_back(offsets.back() + nk::shapes_pack_size<scalar_type_>(clouds[entry].size() / 3, 0).value);
    }
    std::vector<byte_t, nk::aligned_allocator<byte_t>> tape(offsets.back());
    for (std::size_t entry = 0; entry != clouds.size(); ++entry)
        failures += expect(!nk::failed(
            nk::shapes_pack<scalar_type_>(reinterpret_cast<typename scalar_type_::raw_t const*>(stored[entry].data()),
                                          stored[entry].size() / 3, tape.data() + offsets[entry], 0)));

    // The serial distances of every query to every stored cloud, over the packs the GPU stores.
    std::vector<std::vector<distance_t>> exact(queries_count, std::vector<distance_t>(count));
    for (std::size_t query = 0; query != queries_count; ++query)
        for (std::size_t entry = 0; entry != count; ++entry) {
            void const* const query_pack = tape.data() + offsets[count + query];
            void const* const entry_pack = tape.data() + offsets[entry];
            auto const pairwise = kernel == nk_kernel_hausdorffs_packed_k ? serial_kit_t::hausdorffs
                                                                          : serial_kit_t::chamfers;
            failures += expect(pairwise(&query_pack, 1, &entry_pack, 1, &exact[query][entry], sizeof(distance_t),
                                        nullptr, nullptr) == nk_success_k);
        }

    ram_index_t ram_index;
    shapes_metric<kit_t> metric {kernel, forkunion_allocator_t {}};
    failures += expect_eq(metric.init(expansion_k * connectivity_32_k), status_t::success_k);
    typename ram_index_t::traversal_context_t ram_context {seed};
    typename value_t::allocator_type const value_allocator {ram_index.get_allocator()};
    failures += expect(bool(ram_index.reserve(count)));
    for (std::size_t entry = 0; entry != count; ++entry) {
        std::array const cloud {std::span<scalar_type_ const> {stored[entry]}};
        value_t value = value_t::make(cloud, value_allocator, 0).value;
        us::utilities::uuid_t const key = uuid_of(entry);
        failures += expect_eq(ram_index
                                  .insert({&value, 1}, {&key, 1}, metric, {}, ram_context,
                                          us::basic::keep_diverse_t {connectivity_32_k}, {expansion_k})
                                  .status,
                              status_t::success_k);
    }
    std::size_t ram_matched = 0;
    for (std::size_t query = 0; query != queries_count; ++query) {
        std::array const cloud {std::span<scalar_type_ const> {stored[count + query]}};
        value_t const value = value_t::make(cloud, value_allocator, 0).value;
        failures += expect_eq(ram_index.search(value, {expansion_k}, metric, ram_context).status, status_t::success_k);
        std::vector<f64_t> found;
        for (std::size_t rank = 0; rank != min_of(wanted_k, ram_context.results.size()); ++rank)
            found.push_back(static_cast<f64_t>(ram_context.results[rank].distance));
        ram_matched += matched_count(std::span<f64_t const> {found}, exact[query], wanted_k);
    }

    int const device = executor.device(0);
    vram_index_t vram_index {kernel, cuda_virtual_memory_allocator<byte_t> {executor.api(), device}};
    failures += expect_eq(vram_index.reserve({count, connectivity_32_k, offsets[count] - offsets[0]}).status,
                          status_t::success_k);
    typename vram_index_t::insert_config_t const insert_config {expansion_k};
    std::vector<us::utilities::uuid_t> keys(count);
    for (std::size_t entry = 0; entry != count; ++entry) keys[entry] = uuid_of(entry);
    for (std::size_t batch = 0; batch != batches_k && !failures; ++batch) {
        std::size_t const first = count * batch / batches_k, bound = count * (batch + 1) / batches_k;
        std::span<std::size_t const> const batch_offsets {offsets.data() + first, bound - first + 1};
        failures += expect_eq(
            executor.reserve(vram_index.traversal_memory_usage({tape.data(), batch_offsets}, insert_config)),
            status_t::success_k);
        typename vram_index_t::traversal_context_t context {
            cuda_arena_allocator_t {executor.stream(0).arena, executor.stream(0)}};
        auto const [inserted, status] = vram_index.insert({keys.data() + first, bound - first},
                                                          {tape.data(), batch_offsets}, context, insert_config);
        failures += expect_eq(status, status_t::success_k);
        failures += expect_eq(inserted.first_id, first);
        failures += expect_eq(inserted.new_size, bound);
    }
    failures += expect_eq(vram_index.size(), count);

    std::vector<u32_t> ids(queries_count * wanted_k);
    std::vector<f32_t> distances(queries_count * wanted_k);
    std::span<std::size_t const> const query_offsets {offsets.data() + count, queries_count + 1};
    typename vram_index_t::search_config_t const search_config {expansion_k, wanted_k};
    failures += expect_eq(
        executor.reserve(vram_index.traversal_memory_usage({tape.data(), query_offsets}, search_config)),
        status_t::success_k);
    typename vram_index_t::traversal_context_t context {
        cuda_arena_allocator_t {executor.stream(0).arena, executor.stream(0)}};
    if (!failures)
        failures += expect_eq(vram_index
                                  .search({tape.data(), query_offsets}, {ids.data(), {queries_count, wanted_k}},
                                          {distances.data(), {queries_count, wanted_k}}, search_config, context)
                                  .status,
                              status_t::success_k);
    std::size_t vram_matched = 0;
    error_stats_t distance_errors;
    for (std::size_t query = 0; query != queries_count && !failures; ++query) {
        std::vector<f64_t> found;
        for (std::size_t rank = 0; rank != wanted_k; ++rank) {
            u32_t const id = ids[query * wanted_k + rank];
            if (id == std::numeric_limits<u32_t>::max()) break;
            f64_t const distance = distances[query * wanted_k + rank];
            failures += expect(id < count);
            if (id < count) distance_errors.add(distance, exact[query][id]);
            failures += expect(found.empty() || found.back() <= distance);
            found.push_back(distance);
        }
        failures += expect_eq(found.size(), wanted_k);
        vram_matched += matched_count(std::span<f64_t const> {found}, exact[query], wanted_k);
    }
    f64_t const ram_recall = static_cast<f64_t>(ram_matched) / (queries_count * wanted_k);
    f64_t const vram_recall = static_cast<f64_t>(vram_matched) / (queries_count * wanted_k);
    if (vram_recall + 0.05 < ram_recall)
        fmt::print(stderr, "  recall at ten on {} clouds: {:.3f} on the GPU, {:.3f} in RAM\n", count, vram_recall,
                   ram_recall);
    failures += expect(vram_recall + 0.05 >= ram_recall);
    return failures + expect_errors("vram shapes distances", comparison_family_t::approximate_k,
                                    comparison_verdict_t::assert_k, distance_errors, 0);
}

/** Every collection under the kits of @p gpu_class_, whose capabilities are @p capabilities_: the
 *  F16 ones, and the F32 one on the vendor-wide floor, which alone builds F32 packs. */
template <gpu_class_t gpu_class_, nk_capability_t capabilities_>
static std::size_t check_shapes_vram_class(settings_t const& settings, cuda_executor_t& executor, char const* tier) {
    constexpr nk_kernel_kind_t chamfer_k = nk_kernel_chamfers_packed_k, hausdorff_k = nk_kernel_hausdorffs_packed_k;
    struct cell_t {
        char const* name;
        nk_kernel_kind_t kernel;
        cloud_kind_t kind;
        std::size_t count, queries_count;
    };
    cell_t const cells[] = {{"random", chamfer_k, cloud_kind_t::random_k, 1024, 64},
                            {"coincident", hausdorff_k, cloud_kind_t::coincident_k, 256, 16},
                            {"collinear", chamfer_k, cloud_kind_t::collinear_k, 1024, 64},
                            {"molecular", hausdorff_k, cloud_kind_t::molecular_k, 1024, 64}};
    std::size_t failures = 0;
    for (cell_t const& cell : cells)
        failures += report(fmt::format("shapes_vram {} {}", tier, cell.name).c_str(),
                           check_shapes_vram_collection<nk::f16_t, gpu_class_, capabilities_>(
                               stream_key(settings.seed, cell.name), executor, cell.kernel, cell.kind, cell.count,
                               cell.queries_count));
    if constexpr (gpu_class_ == gpu_class_t::nvidia_ampere_k)
        failures += report(
            fmt::format("shapes_vram {} f32", tier).c_str(),
            check_shapes_vram_collection<nk::f32_t, gpu_class_, capabilities_>(
                stream_key(settings.seed, "f32"), executor, chamfer_k, cloud_kind_t::molecular_k, 1024, 64));
    return failures;
}

/** Every class the device runs, as the dispatch ladder would admit it: the floor on any NVIDIA
 *  GPU, and datacenter Blackwell's packed kits on one. */
std::size_t check_shapes_vram(settings_t const& settings, gpu_device_t const& device) {
    if (device.vendor != gpu_vendor_t::nvidia_k || !settings.filter.selects("shapes_vram")) return 0;
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    auto [executor, spawn_status] = cuda_executor_t::spawn(test::gpu_drivers().cuda,
                                                           std::span<int const> {&ordinal, 1});
    if (failed(spawn_status)) return fmt::println(stderr, "- CUDA: device #{} opens no stream", ordinal), 1;
    constexpr nk_capability_t ampere_k = nk_cap_cuda_k | nk_cap_ampere_k;
    constexpr nk_capability_t blackwell_k = ampere_k | nk_cap_ada_k | nk_cap_blackwell_k;
    std::size_t failures = check_shapes_vram_class<gpu_class_t::nvidia_ampere_k, ampere_k>(settings, executor,
                                                                                           "ampere");
    // A build whose architectures name no Blackwell compiles no Blackwell shapes kits to link
    // against.
    if constexpr (NUMKONG_TARGET_BLACKWELL)
        if (nvidia_class_of(cuda_arch_of(device.architecture_major, device.architecture_minor)) ==
            gpu_class_t::nvidia_blackwell_k)
            failures += check_shapes_vram_class<gpu_class_t::nvidia_blackwell_k, blackwell_k>(settings, executor,
                                                                                              "blackwell");
    return failures;
}

#else

std::size_t check_shapes_vram(settings_t const&, gpu_device_t const&) { return 0; }

#endif

} // namespace unum::usearch::test::shapes
