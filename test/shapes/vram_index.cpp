/**
 *  @file test/shapes/vram_index.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The shapes VRAM cell: Chamfer and Hausdorff on an NVIDIA, AMD or Apple GPU beside the RAM
 *      index on the same clouds, judged by recall against exact neighbors, every reported distance
 *      against NumKong's serial kernels.
 *
 *  Collections hold 10 to 60 points a cloud: random points; coincident points, which pack to one
 *  point and tie every pair at zero; collinear points, whose packs degenerate on two axes; and
 *  molecule-like clouds, eight jittered prototypes, one in four mirrored, so neighbors cluster by
 *  prototype. Shapes insert as packs in three batches whose tapes start an alignment line past
 *  their own first byte, as an Arrow slice's do, so the later batches walk graphs whose reverse
 *  links earlier batches merged.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstring> // `std::memcpy` - coincident points, prototypes and each query's exact row

#include <algorithm> // `std::nth_element`
#include <array>     // `std::array` - the prototypes' point counts, one cloud a pack takes
#include <limits>    // `std::numeric_limits` - the node-ID type's sentinel
#include <span>      // `std::span`
#include <string>    // `std::string` - the cases' labels
#include <vector>    // `std::vector` - the tape and its offsets

#include "usearch/shapes_index.h"
#include "usearch/shapes/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/shapes/vram_nvidia_metric.cuh"
#endif
#if USEARCH_ARCH_ROCM_
#include "usearch/shapes/vram_amd_metric.hip"
#endif
#if USEARCH_ARCH_METAL_
#include "usearch/shapes/vram_apple_index.hpp"
#endif

#include "harness.hpp"
#include "lib.hpp" // `gpu_class_t`, `pick_gpu_class` - the class the library picks a unit by

#if USEARCH_ARCH_CUDA_

namespace util = unum::usearch::utilities;

/** The library's fatbin of one class's shapes kits over one dtype, defined in that class's
 *  @c src/shapes/vram_variants_nvidia*.cpp unit. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct shapes_nvidia_fatbin {
    static bytes_view_t bytes() noexcept;
};

#endif
#if USEARCH_ARCH_ROCM_

/** The library's code object of one AMD class's shapes kits over one dtype, defined in that class's
 *  @c src/shapes/vram_variants_amd.hip unit. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct shapes_amd_code_object {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static bytes_view_t bytes() noexcept;
};

#endif

namespace unum::usearch::test::shapes {
using namespace unum::usearch::shapes;

#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_ || USEARCH_ARCH_METAL_

/** How a collection draws its clouds, as the file's docblock describes. */
enum class cloud_kind_t { random_k, coincident_k, collinear_k, molecular_k };

/** The fewest and the most points a drawn cloud holds. */
constexpr std::size_t shapes_min_points_k = 10, shapes_max_points_k = 60;

/** Draws a cloud of @p kind into every row of @p clouds, @b [clouds,points·3] interleaved
 *  coordinates from the row's start in rows @c shapes_max_points_k points wide, and its point count
 *  into @p points. */
static void generate_clouds(cloud_kind_t kind, u64_t seed, nk::matrix_span<f64_t> clouds,
                            std::span<std::size_t> points) {
    constexpr std::size_t prototypes_k = 8, jittered_points_k = 4;
    split_mix_generator_t random_generator {seed};
    uniform_between<f64_t> const centered {-1, 1};
    uniform_below<u32_t> const points_distribution {shapes_max_points_k - shapes_min_points_k + 1};
    uniform_below<u32_t> const prototype_distribution {prototypes_k};
    uniform_below<u32_t> const mirror_distribution {4};
    nk::matrix<f64_t> prototypes = make_matrix<f64_t>(prototypes_k, shapes_max_points_k * 3);
    std::array<std::size_t, prototypes_k> prototype_points;
    for (std::size_t prototype = 0; prototype != prototypes_k; ++prototype) {
        prototype_points[prototype] = shapes_min_points_k + points_distribution(random_generator);
        for (std::size_t coordinate = 0; coordinate != prototype_points[prototype] * 3; ++coordinate)
            prototypes(prototype, coordinate) = 2 * centered(random_generator);
    }
    for (std::size_t entry = 0; entry != clouds.extent(0); ++entry) {
        f64_t* const cloud = clouds.row(entry).data();
        std::size_t const drawn_points = shapes_min_points_k + points_distribution(random_generator);
        std::array<f64_t, 3> const origin {centered(random_generator), centered(random_generator),
                                           centered(random_generator)};
        std::array<f64_t, 3> const direction {centered(random_generator), centered(random_generator),
                                              centered(random_generator)};
        points[entry] = drawn_points;
        switch (kind) {
        case cloud_kind_t::random_k:
            for (std::size_t coordinate = 0; coordinate != drawn_points * 3; ++coordinate)
                cloud[coordinate] = 4 * centered(random_generator);
            break;
        case cloud_kind_t::coincident_k:
            for (std::size_t point = 0; point != drawn_points; ++point)
                std::memcpy(cloud + point * 3, origin.data(), sizeof(origin));
            break;
        case cloud_kind_t::collinear_k:
            for (std::size_t point = 0; point != drawn_points; ++point) {
                f64_t const step = 4 * centered(random_generator);
                for (std::size_t axis = 0; axis != 3; ++axis)
                    cloud[point * 3 + axis] = origin[axis] + step * direction[axis];
            }
            break;
        case cloud_kind_t::molecular_k: {
            std::size_t const prototype = prototype_distribution(random_generator);
            points[entry] = prototype_points[prototype];
            std::memcpy(cloud, prototypes.row(prototype).data(), points[entry] * 3 * sizeof(f64_t));
            uniform_below<u32_t> const point_distribution {static_cast<u32_t>(points[entry])};
            for (std::size_t jitter = 0; jitter != jittered_points_k; ++jitter)
                for (std::size_t axis = 0, point = point_distribution(random_generator); axis != 3; ++axis)
                    cloud[point * 3 + axis] += 0.15 * centered(random_generator);
            f64_t const mirror = mirror_distribution(random_generator) == 0 ? -1 : 1;
            for (std::size_t point = 0; point != points[entry]; ++point) {
                cloud[point * 3] *= mirror;
                for (std::size_t axis = 0; axis != 3; ++axis) cloud[point * 3 + axis] += 4 * origin[axis];
            }
            break;
        }
        }
    }
}

/** Matches, of @p wanted at most, whose distances reach no further than @p reach, the @p wanted -th
 *  exact one, within the float tolerance: ties make the IDs ambiguous, the distances never. */
static std::size_t matched_count(std::span<f64_t const> found, f64_t reach, std::size_t wanted) {
    std::size_t matched = 0;
    for (f64_t const distance : found) matched += distance <= reach + 1e-3 * std::fmax(reach, 1.0);
    return min_of(matched, wanted);
}

#if USEARCH_ARCH_CUDA_

/** What the NVIDIA collections build on: the executor, the face's storage and the contexts carved
 *  from the executor's run. */
struct nvidia_cells_t {
    using executor_t = cuda_executor_t;

    static cuda_virtual_memory_allocator<byte_t> storage(executor_t& executor, int device) noexcept {
        return cuda_virtual_memory_allocator<byte_t> {executor.api(), device};
    }
    static cuda_arena_allocator_t arena(executor_t& executor) noexcept {
        return cuda_arena_allocator_t {executor.stream(0).arena, executor.stream(0)};
    }
    static cuda_api_t const& api() noexcept { return test::gpu_drivers().cuda; }
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::nvidia_k;
    inline static constexpr char runtime_k[] = "CUDA";
};
using cell_vendor_t = nvidia_cells_t;

#elif USEARCH_ARCH_ROCM_

/** What the AMD collections build on: the executor, the face's storage and the contexts carved from
 *  the executor's run. */
struct amd_cells_t {
    using executor_t = hip_executor_t;

    static hip_virtual_memory_allocator<byte_t> storage(executor_t& executor, int device) noexcept {
        return hip_virtual_memory_allocator<byte_t> {executor.api(), device};
    }
    static hip_arena_allocator_t arena(executor_t& executor) noexcept {
        return hip_arena_allocator_t {executor.stream(0).arena, executor.stream(0)};
    }
    static hip_api_t const& api() noexcept { return test::gpu_drivers().hip; }
    inline static constexpr gpu_vendor_t vendor_k = gpu_vendor_t::amd_k;
    inline static constexpr char runtime_k[] = "ROCm";
};
using cell_vendor_t = amd_cells_t;

#else

/** What the Apple collections build on: the executor, the face's storage and the contexts over its
 *  stream. */
struct apple_cells_t {
    using executor_t = metal_executor_t;

    static metal_shared_allocator_t storage(executor_t& executor, int) noexcept {
        return metal_shared_allocator_t {executor.stream(0).handle};
    }
    static metal_stream_t arena(executor_t& executor) noexcept { return executor.stream(0).handle.share(); }
};
using cell_vendor_t = apple_cells_t;

#endif

/**
 *  @brief One collection of @p count clouds of @p kind and @p queries_count held-out queries of
 *      @p scalar_type_, built on the GPU in the face @p vram_index_type_, and in RAM under the kits
 *      of @p kernel alike.
 *
 *  Every match the GPU reports must carry the pair's serial distance, ascending, and the GPU's
 *  recall at ten must reach the RAM index's at the same breadth, give or take a twentieth.
 */
template <typename scalar_type_, typename vram_index_type_>
static void test_shapes_vram_collection(u64_t seed, cell_vendor_t::executor_t& executor, nk_kernel_kind_t kernel,
                                        cloud_kind_t kind, std::size_t count, std::size_t queries_count) {
    using vram_index_t = vram_index_type_;
    checks_held_t const held;
    using kit_t = typename shapes_ladder<scalar_type_, nk_cap_cpus_k>::kit_t;
    using serial_kit_t = typename shapes_ladder<scalar_type_, nk_cap_serial_k>::kit_t;
    using ram_index_t = ram_index<kit_t>;
    using value_t = typename ram_index_t::value_t;
    using distance_t = typename kit_t::distance_t;
    constexpr std::size_t wanted_k = 10, expansion_k = 64, batches_k = 3;

    std::size_t const clouds_count = count + queries_count;
    nk::matrix<f64_t> clouds = make_matrix<f64_t>(clouds_count, shapes_max_points_k * 3);
    nk::matrix<scalar_type_> stored = make_matrix<scalar_type_>(clouds_count, shapes_max_points_k * 3);
    nk::vector<std::size_t> points = make_vector<std::size_t>(clouds_count);
    std::vector<std::size_t> offsets {nk_default_alignment_k};
    generate_clouds(kind, seed, clouds.span(), to_writable_span(points));
    for (std::size_t entry = 0; entry != clouds_count; ++entry) {
        for (std::size_t coordinate = 0; coordinate != points[entry] * 3; ++coordinate)
            stored(entry, coordinate) = scalar_type_(clouds(entry, coordinate));
        auto const [pack_bytes, pack_size_status] = nk::shapes_pack_size<scalar_type_>(points[entry], 0);
        if (!expect_true(succeeded(pack_size_status))) return;
        offsets.push_back(offsets.back() + pack_bytes);
    }
    auto stored_cloud = [&](std::size_t entry) noexcept {
        return to_span(stored.view().row(entry).as_vector()).first(points[entry] * 3);
    };
    std::vector<byte_t, nk::aligned_allocator<byte_t>> tape(offsets.back());
    for (std::size_t entry = 0; entry != clouds_count; ++entry)
        expect_true(!nk::failed(nk::shapes_pack<scalar_type_>(
            reinterpret_cast<typename scalar_type_::raw_t const*>(stored_cloud(entry).data()), points[entry],
            tape.data() + offsets[entry], 0)));

    // The serial distances of every query to every stored cloud, over the packs the GPU stores, and
    // the `wanted_k`-th of each query's, which bounds a match.
    nk::matrix<distance_t> exact = make_matrix<distance_t>(queries_count, count);
    nk::vector<distance_t> ranked = make_vector<distance_t>(count);
    nk::vector<f64_t> reaches = make_vector<f64_t>(queries_count);
    for (std::size_t query = 0; query != queries_count; ++query) {
        for (std::size_t entry = 0; entry != count; ++entry) {
            void const* const query_pack = tape.data() + offsets[count + query];
            void const* const entry_pack = tape.data() + offsets[entry];
            distance_t const infinity = std::numeric_limits<distance_t>::infinity();
            expect_true((kernel == nk_kernel_hausdorffs_packed_k
                             ? serial_kit_t::hausdorffs(&query_pack, 1, &entry_pack, 1, &exact(query, entry),
                                                        sizeof(distance_t), nullptr, &infinity, nullptr)
                             : serial_kit_t::chamfers(&query_pack, 1, &entry_pack, 1, &exact(query, entry),
                                                      sizeof(distance_t), nullptr, &infinity, nullptr, 0, nullptr)) ==
                        nk_success_k);
        }
        std::span<distance_t> const nearest = to_writable_span(ranked);
        std::memcpy(nearest.data(), exact.view().row(query).data(), count * sizeof(distance_t));
        std::nth_element(nearest.begin(), nearest.begin() + (wanted_k - 1), nearest.end());
        reaches[query] = static_cast<f64_t>(nearest[wanted_k - 1]);
    }

    ram_index_t ram_index;
    shapes_metric<kit_t> metric {kernel, forkunion_allocator_t {}};
    typename ram_index_t::traversal_context_t ram_context {seed};
    typename value_t::allocator_type const value_allocator {ram_index.get_allocator()};
    expect_true(bool(ram_index.reserve(count)));
    for (std::size_t entry = 0; entry != count; ++entry) {
        std::array const cloud {stored_cloud(entry)};
        auto [value, pack_status] = value_t::make(cloud, value_allocator, 0);
        if (!expect_true(succeeded(pack_status))) return;
        util::uuid128_t const key = uuid_of(entry);
        expect_equal(ram_index
                         .insert({&value, 1}, {&key, 1}, metric, {}, ram_context,
                                 us::basic::keep_diverse_t {connectivity_32_k}, {expansion_k})
                         .status,
                     status_code_t::success_k);
    }
    std::size_t ram_matched = 0;
    nk::vector<f64_t> found = make_vector<f64_t>(wanted_k);
    for (std::size_t query = 0; query != queries_count; ++query) {
        std::array const cloud {stored_cloud(count + query)};
        auto const [value, pack_status] = value_t::make(cloud, value_allocator, 0);
        if (!expect_true(succeeded(pack_status))) return;
        expect_equal(ram_index.search(value, {expansion_k}, metric, ram_context).status, status_code_t::success_k);
        std::size_t const found_count = min_of(wanted_k, ram_context.results.size());
        for (std::size_t rank = 0; rank != found_count; ++rank)
            found[rank] = static_cast<f64_t>(ram_context.results[rank].distance);
        ram_matched += matched_count(to_span(found).first(found_count), reaches[query], wanted_k);
    }

    int const device = executor.device(0);
    vram_index_t vram_index {kernel, cell_vendor_t::storage(executor, device)};
    expect_equal(vram_index.reserve({count, connectivity_32_k, offsets[count] - offsets[0]}).status,
                 status_code_t::success_k);
    typename vram_index_t::insert_config_t const insert_config {expansion_k};
    nk::vector<util::uuid128_t> keys = make_vector<util::uuid128_t>(count);
    for (std::size_t entry = 0; entry != count; ++entry) keys[entry] = uuid_of(entry);
    for (std::size_t batch = 0; batch != batches_k && held(); ++batch) {
        std::size_t const first = count * batch / batches_k, bound = count * (batch + 1) / batches_k;
        std::span<std::size_t const> const batch_offsets {offsets.data() + first, bound - first + 1};
        expect_equal(executor.reserve(vram_index.traversal_memory_usage(
                         tape_view<byte_t const> {tape.data(), batch_offsets}, insert_config)),
                     status_code_t::success_k);
        typename vram_index_t::traversal_context_t context {cell_vendor_t::arena(executor)};
        auto const [inserted, status] = vram_index.insert({keys.data() + first, bound - first},
                                                          {tape.data(), batch_offsets}, context, insert_config);
        expect_equal(status, status_code_t::success_k);
        expect_equal(inserted.first_id, first);
        expect_equal(inserted.new_size, bound);
    }
    expect_equal(vram_index.size(), count);

    nk::matrix<u32_t> ids = make_matrix<u32_t>(queries_count, wanted_k);
    nk::matrix<f32_t> distances = make_matrix<f32_t>(queries_count, wanted_k);
    std::span<std::size_t const> const query_offsets {offsets.data() + count, queries_count + 1};
    typename vram_index_t::search_config_t const search_config {expansion_k, wanted_k};
    expect_equal(executor.reserve(vram_index.traversal_memory_usage(
                     tape_view<byte_t const> {tape.data(), query_offsets}, search_config)),
                 status_code_t::success_k);
    typename vram_index_t::traversal_context_t context {cell_vendor_t::arena(executor)};
    if (held())
        expect_equal(
            vram_index.search({tape.data(), query_offsets}, ids.span(), distances.span(), search_config, context)
                .status,
            status_code_t::success_k);
    std::size_t vram_matched = 0;
    error_stats_t distance_errors;
    for (std::size_t query = 0; query != queries_count && held(); ++query) {
        std::size_t found_count = 0;
        for (std::size_t rank = 0; rank != wanted_k; ++rank) {
            u32_t const id = ids(query, rank);
            if (id == (std::numeric_limits<u32_t>::max)()) break;
            f64_t const distance = distances(query, rank);
            expect_true(id < count);
            if (id < count) distance_errors.add(distance, exact(query, id));
            expect_true(found_count == 0 || found[found_count - 1] <= distance);
            found[found_count++] = distance;
        }
        expect_equal(found_count, wanted_k);
        vram_matched += matched_count(to_span(found).first(found_count), reaches[query], wanted_k);
    }
    f64_t const ram_recall = static_cast<f64_t>(ram_matched) / (queries_count * wanted_k);
    f64_t const vram_recall = static_cast<f64_t>(vram_matched) / (queries_count * wanted_k);
    if (vram_recall + 0.05 < ram_recall)
        fmt::print(stderr, "  recall at ten on {} clouds: {:.3f} on the GPU, {:.3f} in RAM\n", count, vram_recall,
                   ram_recall);
    expect_true(vram_recall + 0.05 >= ram_recall);
    expect_errors("vram shapes distances", comparison_family_t::approximate_k, comparison_verdict_t::assert_k,
                  distance_errors, 0);
}

#endif // USEARCH_ARCH_CUDA_ || USEARCH_ARCH_METAL_
#if USEARCH_ARCH_CUDA_

/** The NVIDIA face over packs of @p scalar_type_ under the kits of @p gpu_class_, whose
 *  capabilities are @p capabilities_. */
template <typename scalar_type_, gpu_class_t gpu_class_, nk_capability_t capabilities_>
using shapes_face_t = vram_nvidia_index<typename shapes_ladder<scalar_type_, capabilities_>::kit_t, u32_t,
                                        shapes_nvidia_fatbin<gpu_class_, scalar_type_::dtype()>>;

#elif USEARCH_ARCH_ROCM_

/** The AMD face over packs of @p scalar_type_ under the kits of @p gpu_class_, whose capabilities
 *  are @p capabilities_. */
template <typename scalar_type_, gpu_class_t gpu_class_, nk_capability_t capabilities_>
using shapes_face_t = vram_amd_index<typename shapes_ladder<scalar_type_, capabilities_>::kit_t, u32_t,
                                     shapes_amd_code_object<gpu_class_, scalar_type_::dtype()>>;

#endif
#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_

/** Every collection under the kits of @p gpu_class_, whose capabilities are @p capabilities_: the
 *  F16 ones, and the F32 one on the vendor-wide floor, which alone builds F32 packs. */
template <gpu_class_t gpu_class_, nk_capability_t capabilities_>
static void run_shapes_vram_class_cases(cell_vendor_t::executor_t& executor, char const* gpu_class) {
    constexpr nk_kernel_kind_t chamfer_k = nk_kernel_chamfers_packed_k, hausdorff_k = nk_kernel_hausdorffs_packed_k;
    struct cell_t {
        char const* name;
        char const* kind_constant;
        nk_kernel_kind_t kernel;
        cloud_kind_t kind;
        std::size_t count, queries_count;
    };
    cell_t const cells[] = {
        {"random", "cloud_kind_t::random_k", chamfer_k, cloud_kind_t::random_k, 1024, 64},
        {"coincident", "cloud_kind_t::coincident_k", hausdorff_k, cloud_kind_t::coincident_k, 256, 16},
        {"collinear", "cloud_kind_t::collinear_k", chamfer_k, cloud_kind_t::collinear_k, 1024, 64},
        {"molecular", "cloud_kind_t::molecular_k", hausdorff_k, cloud_kind_t::molecular_k, 1024, 64}};
    u64_t const seed = test_run.settings.seed;
    for (cell_t const& cell : cells)
        run_test_case(
            fmt::format("test_shapes_vram_collection<nk::f16_t>({}, {})", gpu_class, cell.kind_constant).c_str(),
            test_shapes_vram_collection<nk::f16_t, shapes_face_t<nk::f16_t, gpu_class_, capabilities_>>,
            stream_key(seed, cell.name), executor, cell.kernel, cell.kind, cell.count, cell.queries_count);
    if constexpr (gpu_class_ == gpu_class_t::nvidia_ampere_k || gpu_class_ == gpu_class_t::amd_cdna3_k)
        run_test_case(
            fmt::format("test_shapes_vram_collection<nk::f32_t>({}, cloud_kind_t::molecular_k)", gpu_class).c_str(),
            test_shapes_vram_collection<nk::f32_t, shapes_face_t<nk::f32_t, gpu_class_, capabilities_>>,
            stream_key(seed, "f32"), executor, chamfer_k, cloud_kind_t::molecular_k, std::size_t {1024},
            std::size_t {64});
}

/** Every class the device runs, as the dispatch ladder would admit it: the floor on any NVIDIA
 *  GPU, and datacenter Blackwell's packed kits on one; the CDNA3 floor on a 64-lane AMD GPU. */
template <>
void run_shapes_vram_cases<cell_vendor_t::executor_t>(gpu_device_t const& device) {
    if (device.vendor != cell_vendor_t::vendor_k) return;
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    auto [executor, spawn_status] = cell_vendor_t::executor_t::spawn(cell_vendor_t::api(),
                                                                     std::span<int const> {&ordinal, 1});
    if (!expect_true(succeeded(spawn_status), "{} device #{} opens no stream", cell_vendor_t::runtime_k, ordinal))
        return;
#if USEARCH_ARCH_CUDA_
    constexpr nk_capability_t ampere_k = nk_cap_cuda_k | nk_cap_ampere_k;
    constexpr nk_capability_t blackwell_k = ampere_k | nk_cap_ada_k | nk_cap_blackwell_k;
    run_shapes_vram_class_cases<gpu_class_t::nvidia_ampere_k, ampere_k>(executor, "gpu_class_t::nvidia_ampere_k");
    // A build that keeps no Blackwell class compiles no Blackwell shapes kits to link against.
    if constexpr (USEARCH_TARGET_NVIDIA_BLACKWELL)
        if (pick_gpu_class(device, usearch_gpu_class_unknown_k) == gpu_class_t::nvidia_blackwell_k)
            run_shapes_vram_class_cases<gpu_class_t::nvidia_blackwell_k, blackwell_k>(
                executor, "gpu_class_t::nvidia_blackwell_k");
#else
    // The AMD cells run the CDNA3 class's kits over 64-lane wavefronts, which CDNA3 and CDNA4 run.
    if (device.warp_lanes != 64)
        return run_test_case("run_shapes_vram_cases(gpu_vendor_t::amd_k)",
                             [] { return case_result_t::skipped(needs_64_lane_waves_k); });
    constexpr nk_capability_t cdna3_k = numkong_capabilities_of(gpu_class_t::amd_cdna3_k);
    // A build that keeps no CDNA3 class compiles no CDNA3 shapes kits to link against.
    if constexpr (USEARCH_TARGET_AMD_CDNA3)
        run_shapes_vram_class_cases<gpu_class_t::amd_cdna3_k, cdna3_k>(executor, "gpu_class_t::amd_cdna3_k");
    else
        run_test_case("run_shapes_vram_cases(gpu_vendor_t::amd_k)",
                      [] { return case_result_t::skipped(built_without_cdna3_k); });
#endif
}

#elif USEARCH_ARCH_METAL_

/** The Apple cells: the F16 collections the NVIDIA floor checks, on the same clouds, judged the
 *  same, and the refusal of F32 clouds, which no Metal kernel scores to the F32 contract. */
template <>
void run_shapes_vram_cases<metal_executor_t>(gpu_device_t const& device) {
    if (device.vendor != gpu_vendor_t::apple_k) return;
    int const ordinal = static_cast<int>(device.vendor_ordinal);
    auto [executor, spawn_status] = metal_executor_t::spawn(test::gpu_drivers().metal_contexts,
                                                            std::span<int const> {&ordinal, 1});
    if (!expect_true(succeeded(spawn_status), "Metal device #{} opens no stream", ordinal)) return;
    constexpr nk_kernel_kind_t chamfer_k = nk_kernel_chamfers_packed_k, hausdorff_k = nk_kernel_hausdorffs_packed_k;
    struct cell_t {
        char const* name;
        char const* kind_constant;
        nk_kernel_kind_t kernel;
        cloud_kind_t kind;
        std::size_t count, queries_count;
    };
    cell_t const cells[] = {
        {"random", "cloud_kind_t::random_k", chamfer_k, cloud_kind_t::random_k, 1024, 64},
        {"coincident", "cloud_kind_t::coincident_k", hausdorff_k, cloud_kind_t::coincident_k, 256, 16},
        {"collinear", "cloud_kind_t::collinear_k", chamfer_k, cloud_kind_t::collinear_k, 1024, 64},
        {"molecular", "cloud_kind_t::molecular_k", hausdorff_k, cloud_kind_t::molecular_k, 1024, 64}};
    u64_t const seed = test_run.settings.seed;
    for (cell_t const& cell : cells)
        run_test_case(
            fmt::format("test_shapes_vram_collection<nk::f16_t>(gpu_vendor_t::apple_k, {})", cell.kind_constant)
                .c_str(),
            test_shapes_vram_collection<nk::f16_t, vram_apple_index<nk::f16_t, u32_t>>, stream_key(seed, cell.name),
            executor, cell.kernel, cell.kind, cell.count, cell.queries_count);
    // The C interface names devices by their place in the listing, which `device` belongs to.
    std::int32_t const devices[] = {static_cast<std::int32_t>(&device - test::gpu_devices().data())};
    run_test_case("usearch_shapes_init_vram", [&] {
        usearch_outcome_t outcome {};
        usearch_shapes_index_t index = nullptr;
        usearch_shapes_init_vram_t init {};
        init.index = &index;
        init.outcome = &outcome;
        init.metric = usearch_shapes_metric_chamfer_k;
        init.quantization = usearch_scalar_f32_k;
        init.vram = {devices, 1};
        usearch_shapes_init_vram(&init);
        expect_status(outcome, usearch_status_unsupported_k);
        expect_true(!index);
        usearch_shapes_free(index);
    });
}

#endif // USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_

} // namespace unum::usearch::test::shapes
