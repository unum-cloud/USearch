/**
 *  @file src/shapes_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief What one shapes variant is spelled with and compiled through: the kinds' tokens and
 *      kernels, the geometry every shapes variant freezes, the metric factory, the packing of
 *      clouds, and the variants over the basic indexes.
 *
 *  Every shapes class unit in @c src/shapes/ includes this to instantiate its variants, and
 *  @c shapes_index.cpp adds the variant, the traits and the exported C entry points on top of it.
 */
#pragma once

#include <cstring> // `std::memcpy` - a view's points exported as saved

#include <array>       // `std::array` - the one cloud a query packs, the checkpoint's recorded fields
#include <memory>      // `std::allocator_traits` - a query's packs on a thread's arena, a GPU shard's face
#include <new>         // `std::nothrow` - a GPU shard built on its device
#include <span>        // `std::span` - a cloud's coordinates
#include <string_view> // `std::string_view` - tokens and paths
#include <variant>     // `std::monostate` - nothing a freeze reads past the geometry

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the scan totals

#include "usearch/shapes/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/shapes/vram_nvidia_index.hpp"
#endif

#include "basic_specializations.cuh"

#pragma region Shapes Variants

/** The NumKong kernel one shapes kind scores with. */
constexpr nk_kernel_kind_t kernel_of(usearch_shapes_metric_t metric) noexcept {
    switch (metric) {
    case usearch_shapes_metric_hausdorff_k: return nk_kernel_hausdorffs_packed_k;
    case usearch_shapes_metric_gromov_wasserstein_k: return nk_kernel_gromovwassersteins_packed_k;
    case usearch_shapes_metric_wasserstein_procrustes_k: return nk_kernel_wassersteinprocrustes_packed_k;
    default: return nk_kernel_chamfers_packed_k;
    }
}

/** A factory of @p metric_type_ under @p metric over a thread's arena, sized for a round of
 *  candidates. */
template <typename metric_type_>
auto shapes_metric_factory(usearch_shapes_metric_t metric) noexcept {
    return [kernel = kernel_of(metric)](thread_allocator_t const& allocator, std::size_t staging) noexcept {
        metric_type_ made {kernel, allocator};
        status_t const ready = made.init(staging);
        return expected<metric_type_> {std::move(made), ready};
    };
}

/** One shapes alternative: the basic shards over packs of @p scalar_type_, the kind they score
 *  with, and the kits @p cpu_class_ picks; verbs defined below the class, like the sequence ones.
 *  Checkpoints and refinements serve an index of one shard. */
template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
struct shapes_ram_variant {
    using executor_t = forkunion_executor_t;
    template <typename word_type_>
    using atomic_ref = typename atomic_ref_for_cpu_class<cpu_class_>::template type<word_type_>;
    using scalar_t = scalar_type_;
    using kit_t = typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(cpu_class_)>::kit_t;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    using shard_t = us::shapes::ram_index<kit_t, us::utilities::uuid_t, waiter_of<cpu_class_>,
                                          atomic_ref_for_cpu_class<cpu_class_>::template type, forkunion_allocator_t,
                                          cache_hints_t>;
    using index_t = basic::ram_indexes<shard_t>;
    using value_t = typename shard_t::value_t;
    using codec_t = us::shapes::codec<kit_t>;
    using metric_t = us::shapes::shapes_metric<kit_t, thread_allocator_t>;
    using refine_plan_t = typename index_t::refine_plan_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr nk_dtype_t dtype_k = scalar_type_::dtype();

    index_t index;
    shapes_geometry_t geometry;

    /** Points across every stored entry, added to by insertions and loads through @c atomic_ref. */
    std::size_t total_points = 0;

    explicit shapes_ram_variant(forkunion_allocator_t const& allocator) noexcept
        : index(allocator, machine_cache_line_bytes()) {}

    /** Reserves the entries @p arguments name. */
    status_t reserve(usearch_shapes_reserve_t const& arguments) noexcept {
        return index.reserve(arguments.entries).status;
    }

    /** Empties the index and its point count. */
    status_t clear() noexcept {
        status_t const cleared = index.clear();
        if (succeeded(cleared)) atomic_ref<std::size_t>(total_points).store(0, std::memory_order_relaxed);
        return cleared;
    }

    expected<traversal_telemetry_t> insert(usearch_shapes_insert_t const& arguments, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;
    expected<traversal_telemetry_t> search(usearch_shapes_search_t const& arguments, matches_landing_t& landing,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** Opens a refinement that first moves every shard of each of @p donors into this index, then
     *  re-links across, its walks as wide as @p expansion or twice the connectivity. */
    expected<refine_plan_t> plan_absorb(std::span<index_t*> donors, std::size_t expansion) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_absorb(donors, planning, basic::keep_diverse_t {geometry.connectivity}, {expansion});
    }

    /** Opens a refinement of every live shard in place, its walks as wide as @p expansion. */
    expected<refine_plan_t> plan_refine(std::size_t expansion) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_refine(planning, basic::keep_diverse_t {geometry.connectivity}, {expansion});
    }
    status_t refine(refine_plan_t& plan, progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** Adds the points of a landed @p donor, whose rows this index now holds. */
    void absorbed(shapes_ram_variant const& donor) noexcept {
        atomic_ref<std::size_t>(total_points).add(donor.total_points, std::memory_order_relaxed);
    }

    /** The points of one stored value. */
    static std::size_t points_of(value_t const& value) noexcept { return value.max_points(); }

    /** Writes the @p entries points of one stored value into @p output in @p dtype, through
     *  @p scratch when the dtype is not the stored one. */
    status_t export_points(value_t const& value, nk_dtype_t dtype, void* output,
                           thread_allocator_t const& scratch) const noexcept;
    status_t save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Serving a checkpoint in place and sampling answer @c unsupported_k until the index serves
     *  them. */
    status_t view(std::string_view) noexcept { return status_t::unsupported_k; }
    status_t sample(usearch_shapes_sample_t const&) noexcept { return status_t::unsupported_k; }
};

/** Packs entry @p row of @p tape into @p packed under @p allocator, casting the caller's scalar
 *  into @p scalar_type_ with @p kit_type_ through @p scratch where they differ; refuses a count
 *  other than a frozen nonzero @p points. */
template <nk::shapes_dtype scalar_type_, typename kit_type_, typename packed_type_, typename allocator_type_>
status_t pack_entry(usearch_tape_view_t tape, std::size_t row, std::size_t points, packed_type_& packed,
                    allocator_type_ const& allocator, thread_allocator_t const& scratch) noexcept {
    constexpr nk_dtype_t dtype_k = scalar_type_::dtype();
    std::size_t const count = tape.offsets[row + 1] - tape.offsets[row];
    if (points && count != points) return status_t::shape_mismatch_k;
    nk_dtype_t const from = nk_dtype_of(tape.scalar);
    std::size_t const from_bytes = nk_dtype_bits(from) / 8;
    std::size_t const elements = count * 3;
    bytes_view_t const coordinates = bytes_of(
        static_cast<byte_t const*>(tape.data) + tape.offsets[row] * 3 * from_bytes, elements * from_bytes);
    growing_buffer<scalar_type_, thread_allocator_t> converted {scratch};
    std::span<scalar_type_ const> cloud = elements_of<scalar_type_>(coordinates);
    if (from != dtype_k) {
        if (failed(converted.resize_uninitialized(elements))) return status_t::bad_alloc_k;
        if (status_t const cast = us::shapes::status_of(
                kit_type_::cast(coordinates.data(), from, converted.data(), dtype_k, elements, nullptr));
            failed(cast))
            return cast;
        cloud = to_span(converted);
    }
    std::array const clouds {cloud};
    auto [made, status] = packed_type_::make(clouds, allocator, kit_type_::pack_size, kit_type_::pack);
    if (nk::failed(status)) return us::shapes::status_of(status);
    packed = std::move(made);
    return status_t::success_k;
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
expected<traversal_telemetry_t> shapes_ram_variant<scalar_type_, cpu_class_>::insert(
    usearch_shapes_insert_t const& arguments, progress_ticks_t& ticks, executor_t& executor) noexcept {
    usearch_tape_view_t const& shapes = arguments.shapes;
    std::size_t const count = arguments.count, connectivity = geometry.connectivity;
    auto build = [&](std::size_t row, value_t& value, std::size_t thread) noexcept {
        return pack_entry<scalar_type_, kit_t>(shapes, row, geometry.points, value, value.get_allocator(),
                                               thread_allocator_t {executor.arena(thread)});
    };
    std::size_t const expansion = arguments.expansion_factor ? arguments.expansion_factor : connectivity * 2;
    expected<traversal_telemetry_t> const inserted = insert_entries(
        index, arguments.keys, count, connectivity, expansion, shapes_metric_factory<metric_t>(geometry.metric), build,
        ticks, executor);
    if (inserted)
        atomic_ref<std::size_t>(total_points).add(shapes.offsets[count] - shapes.offsets[0], std::memory_order_relaxed);
    return inserted;
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
expected<traversal_telemetry_t> shapes_ram_variant<scalar_type_, cpu_class_>::search(
    usearch_shapes_search_t const& arguments, matches_landing_t& landing, progress_ticks_t& ticks,
    executor_t& executor) noexcept {
    using query_t = nk::shapes_packed<scalar_type_,
                                      typename std::allocator_traits<thread_allocator_t>::template rebind_alloc<char>>;
    auto query_of = [&](std::size_t row, std::size_t thread, auto&& answer) noexcept {
        thread_allocator_t const scratch {executor.arena(thread)};
        query_t query {scratch};
        if (status_t const packed = pack_entry<scalar_type_, kit_t>(arguments.queries, row, geometry.points, query,
                                                                    scratch, scratch);
            failed(packed))
            return packed;
        return answer(query);
    };
    return search_entries(index, arguments.count, arguments.wanted, geometry.connectivity, arguments.expansion_factor,
                          arguments.traversal, arguments.predicate, shapes_metric_factory<metric_t>(geometry.metric),
                          query_of, landing, ticks, executor);
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
status_t shapes_ram_variant<scalar_type_, cpu_class_>::refine(refine_plan_t& plan, progress_ticks_t& ticks,
                                                              executor_t& executor) noexcept {
    auto [scratch, prepare_status] = prepare_refinement<shard_t>(
        plan, geometry.connectivity, shapes_metric_factory<metric_t>(geometry.metric), executor);
    if (failed(prepare_status)) return prepare_status;
    return drive_refinement(plan, scratch, basic::keep_diverse_t {geometry.connectivity}, ticks, executor);
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
status_t shapes_ram_variant<scalar_type_, cpu_class_>::export_points(value_t const& value, nk_dtype_t dtype,
                                                                     void* output,
                                                                     thread_allocator_t const& scratch) const noexcept {
    if (value.empty()) return status_t::success_k;
    std::size_t const elements = value.max_points() * 3;
    if (dtype == dtype_k)
        return us::shapes::status_of(
            kit_t::unpack(value[0], static_cast<typename scalar_type_::raw_t*>(output), nullptr));
    growing_buffer<typename scalar_type_::raw_t, thread_allocator_t> unpacked {scratch};
    if (failed(unpacked.resize_uninitialized(elements))) return status_t::bad_alloc_k;
    if (status_t const opened = us::shapes::status_of(kit_t::unpack(value[0], unpacked.data(), nullptr));
        failed(opened))
        return opened;
    return us::shapes::status_of(kit_t::cast(unpacked.data(), dtype_k, output, dtype, elements, nullptr));
}

/** Hands @p write, as <tt>write(config) → status_t</tt>, the save configuration recording what
 *  @p geometry froze, the connectivity and the point count, as every residency's checkpoint
 *  records them. */
template <typename write_type_>
status_t with_shapes_provenance(shapes_geometry_t const& geometry, write_type_&& write) noexcept {
    std::array<char, 24> connectivity_text, points_text;
    std::array<safegraphs_custom_field_t, 2> const fields {{
        {"connectivity", decimal_of(geometry.connectivity, connectivity_text)},
        {"points", decimal_of(geometry.points, points_text)},
    }};
    save_config_t config;
    config.provenance = fields;
    return write(config);
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
status_t shapes_ram_variant<scalar_type_, cpu_class_>::save(std::string_view path, executor_t& executor,
                                                            progress_ticks_t& ticks) noexcept {
    return with_shapes_provenance(geometry, [&](save_config_t const& config) noexcept {
        return index.save(path, codec_t {}, executor, config, token_of(geometry.metric), ticks);
    });
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
status_t shapes_ram_variant<scalar_type_, cpu_class_>::load(std::string_view path, executor_t& executor,
                                                            progress_ticks_t& ticks) noexcept {
    if (status_t const loaded = index.load(path, codec_t {}, executor, load_config_t {}, ticks); failed(loaded))
        return loaded;
    for (shard_t const& one : index.live_shards())
        for (std::size_t node = 0; node != one.size(); ++node)
            total_points += one.published_value_of(static_cast<typename shard_t::node_id_t>(node)).max_points();
    return status_t::success_k;
}

/** One shapes alternative on block storage: the basic NVMe index over the clouds of @p scalar_type_
 *  a RAM variant saved, read-only, each row its points as saved, scored with the RAM variant's
 *  metric and the kits @p cpu_class_ picks. Its verbs are instantiated in the class's unit, as the
 *  RAM variant's are. */
template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
struct shapes_nvme_variant {
    using executor_t = nvme_executor_t;
    template <typename word_type_>
    using atomic_ref = typename atomic_ref_for_cpu_class<cpu_class_>::template type<word_type_>;
    using scalar_t = scalar_type_;
    using kit_t = typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(cpu_class_)>::kit_t;
    using codec_t = us::shapes::codec<kit_t>;
    using metric_t = us::shapes::shapes_metric<kit_t, thread_allocator_t>;
    using index_t = basic::nvme_indexes<codec_t, us::utilities::uuid_t, u32_t, typename kit_t::distance_t,
                                        default_io_engine_t, row_kit_of<cpu_class_>,
                                        atomic_ref_for_cpu_class<cpu_class_>::template type, waiter_of<cpu_class_>>;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr nk_dtype_t dtype_k = scalar_type_::dtype();

    index_t index;
    shapes_geometry_t geometry;

    /** Points across every stored entry, counted once by the view. */
    std::size_t total_points = 0;

    explicit shapes_nvme_variant(forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    /** Serves the checkpoint at @p path in place, its parts from @p paths when any are named. */
    status_t view(std::string_view path, std::span<std::string_view const> paths, executor_t& executor) noexcept {
        status_t const viewed = index.view(path, paths, token_of(geometry.metric), executor);
        total_points = index.value_elements() / 3;
        return viewed;
    }

    expected<traversal_telemetry_t> search(usearch_shapes_search_t const& arguments, matches_landing_t& landing,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** The points of one stored @p cloud, its bytes as saved, three coordinates apiece. */
    static std::size_t points_of(bytes_view_t cloud) noexcept {
        return cloud.size() / (3 * sizeof(typename scalar_type_::raw_t));
    }

    /** Writes the points of one stored @p cloud, its bytes as saved, into @p output in @p dtype; a
     *  cast reads them in place, so it needs no scratch. */
    status_t export_points(bytes_view_t cloud, nk_dtype_t dtype, void* output,
                           thread_allocator_t const&) const noexcept {
        if (dtype != dtype_k)
            return us::shapes::status_of(
                kit_t::cast(cloud.data(), dtype_k, output, dtype, cloud.size() / sizeof(scalar_type_), nullptr));
        if (!cloud.empty()) std::memcpy(output, cloud.data(), cloud.size());
        return status_t::success_k;
    }

    /** A view already is its checkpoint, so saving answers @c unsupported_k, as does sampling until
     *  the view serves it. */
    status_t save(std::string_view, executor_t&, progress_ticks_t&) noexcept { return status_t::unsupported_k; }
    status_t sample(usearch_shapes_sample_t const&) noexcept { return status_t::unsupported_k; }
};

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
expected<traversal_telemetry_t> shapes_nvme_variant<scalar_type_, cpu_class_>::search(
    usearch_shapes_search_t const& arguments, matches_landing_t& landing, progress_ticks_t& ticks,
    executor_t& executor) noexcept {
    using query_t = nk::shapes_packed<scalar_type_,
                                      typename std::allocator_traits<thread_allocator_t>::template rebind_alloc<char>>;
    auto query_of = [&](std::size_t row, thread_allocator_t const& scratch) noexcept {
        query_t query {scratch};
        status_t const packed = pack_entry<scalar_type_, kit_t>(arguments.queries, row, geometry.points, query, scratch,
                                                                scratch);
        return expected<query_t> {std::move(query), packed};
    };
    return search_entries(index, arguments.count, arguments.wanted, arguments.expansion_factor, arguments.traversal,
                          arguments.predicate, shapes_metric_factory<metric_t>(geometry.metric), query_of, landing,
                          ticks, executor);
}

#if USEARCH_ARCH_CUDA_

/** One class's shapes kits over one dtype as a fatbin, defined in its class unit,
 *  @c src/shapes/vram_variants_nvidia.cpp for the vendor-wide Ampere floor, beside the bytes it
 *  embeds. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct shapes_nvidia_fatbin {
    static bytes_view_t bytes() noexcept;
};

/** The faces' members reading the kits, defined in each class unit. */
extern template struct us::shapes::vram_nvidia_index<nk::f16_t, u32_t,
                                                     numkong_capabilities_of(gpu_class_t::nvidia_ampere_k),
                                                     shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f16_k>>;
extern template struct us::shapes::vram_nvidia_index<nk::f32_t, u32_t,
                                                     numkong_capabilities_of(gpu_class_t::nvidia_ampere_k),
                                                     shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f32_k>>;
extern template struct us::shapes::vram_nvidia_index<nk::f16_t, u32_t,
                                                     numkong_capabilities_of(gpu_class_t::nvidia_blackwell_k),
                                                     shapes_nvidia_fatbin<gpu_class_t::nvidia_blackwell_k, nk_f16_k>>;

/** Builds one shard of the shapes face over packs of @p scalar_type_ for @p gpu_class_ on a CUDA
 *  device, scoring under the kernel the variant's frozen geometry names. */
template <nk::shapes_dtype scalar_type_, gpu_class_t gpu_class_>
struct shapes_vram_nvidia_build {
    using face_t = us::shapes::vram_nvidia_index<scalar_type_, u32_t, numkong_capabilities_of(gpu_class_),
                                                 shapes_nvidia_fatbin<gpu_class_, scalar_type_::dtype()>>;

    /** The kernel every shard scores under. */
    nk_kernel_kind_t kernel = nk_kernel_unknown_k;

    std::unique_ptr<face_t> operator()(int device) const noexcept {
        return std::unique_ptr<face_t> {
            new (std::nothrow) face_t(kernel, cuda_virtual_memory_allocator<byte_t> {gpu_drivers().cuda, device})};
    }
};

/** The shapes alternative on NVIDIA GPUs of @p gpu_class_ or newer: the basic VRAM shards over
 *  packs of @p scalar_type_, scoring Chamfer or Hausdorff as the geometry names, with 32-bit node
 *  IDs. A batch packs its clouds on the host end to end through the serial kit, as the RAM variants
 *  pack them. */
template <nk::shapes_dtype scalar_type_, gpu_class_t gpu_class_>
struct shapes_vram_nvidia_variant {
    using executor_t = cuda_executor_t;
    using build_t = shapes_vram_nvidia_build<scalar_type_, gpu_class_>;
    using face_t = typename build_t::face_t;
    using index_t = basic::vram_nvidia_indexes<face_t, us::utilities::uuid_t, build_t>;
    using refine_plan_t = vram_nvidia_refine_plan_t;
    using scalar_t = scalar_type_;
    using pack_kit_t = typename us::shapes::shapes_ladder<scalar_type_, nk_cap_serial_k>::kit_t;
    static constexpr nk_dtype_t dtype_k = scalar_type_::dtype();
    static constexpr gpu_class_t gpu_class_k = gpu_class_;

    index_t index;

    /** Holds no shard until the handle places them, each scoring under the kernel @p frozen
     *  names. */
    shapes_vram_nvidia_variant(int, shapes_geometry_t const& frozen) noexcept
        : index(forkunion_allocator_t {no_memory_domain_k},
                build_t {frozen.metric == usearch_shapes_metric_hausdorff_k ? nk_kernel_hausdorffs_packed_k
                                                                            : nk_kernel_chamfers_packed_k}) {}

    shapes_geometry_t geometry;

    /** Points across every stored entry, added to by insertions through @c atomic_ref. */
    std::size_t total_points = 0;

    /** Reserves the entries and points @p arguments name, a pack taking a header and a partial
     *  alignment line per axis past its points. */
    status_t reserve(usearch_shapes_reserve_t const& arguments) noexcept {
        nk_size_t header = 0;
        if (status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(0, &header)); failed(sized))
            return sized;
        std::size_t const pack_overhead = header + 3 * nk_default_alignment_k;
        std::size_t const bytes = arguments.entries * pack_overhead +
                                  arguments.points * 3 * sizeof(typename scalar_type_::raw_t);
        return vram_nvidia_reserve(index, arguments.entries, geometry.connectivity, bytes);
    }

    expected<traversal_telemetry_t> insert(usearch_shapes_insert_t const& arguments, progress_ticks_t& ticks,
                                           cuda_executor_t& executor) noexcept;

    /** Packs the queries into @p pages, the call arena's host pages, where the matches land too. A
     *  device walk takes no host predicate. */
    expected<traversal_telemetry_t> search(usearch_shapes_search_t const& arguments,
                                           growing_arena<forkunion_allocator_t>& pages, matches_landing_t& landing,
                                           progress_ticks_t& ticks, cuda_executor_t& executor) noexcept;

    /** Writes the clouds as the RAM variants write them: a lone live shard's monolith, or one per
     *  live shard beside the manifest the path names. */
    status_t save(std::string_view path, cuda_executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Reads a checkpoint any residency wrote into the placed shards, one part per shard. */
    status_t load(std::string_view path, cuda_executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Empties the index and its point count, retiring every shard behind a fresh shard generation
     *  on the same devices. */
    status_t clear() noexcept {
        status_t const cleared = index.clear();
        if (succeeded(cleared)) fu::standard_atomic_ref<std::size_t>(total_points).store(0, std::memory_order_relaxed);
        return cleared;
    }

    /** The refinements, which no GPU variant opens yet, each answering @c unsupported_k. */
    expected<refine_plan_t> plan_absorb(std::span<index_t*>, std::size_t) noexcept {
        return {{}, status_t::unsupported_k};
    }
    expected<refine_plan_t> plan_refine(std::size_t) noexcept { return {{}, status_t::unsupported_k}; }
    status_t refine(refine_plan_t&, progress_ticks_t&, executor_t&) noexcept { return status_t::unsupported_k; }
    status_t sample(usearch_shapes_sample_t const&) noexcept { return status_t::unsupported_k; }

  private:

    /** The wire format every residency's checkpoints share: each cloud's points, unpacked. */
    using codec_t = us::shapes::codec<pack_kit_t>;

    /** The @p count clouds of @p tape packed end to end on @p pages, cast to the stored scalar
     *  where the caller's differs; refuses a count other than a frozen one. */
    expected<tape_view<byte_t const>> pack_(usearch_tape_view_t tape, std::size_t count,
                                            growing_arena<forkunion_allocator_t>& pages) const noexcept;
};

template <nk::shapes_dtype scalar_type_, gpu_class_t gpu_class_>
expected<tape_view<byte_t const>> shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::pack_(
    usearch_tape_view_t tape, std::size_t count, growing_arena<forkunion_allocator_t>& pages) const noexcept {
    using raw_t = typename scalar_type_::raw_t;
    nk_dtype_t const from = nk_dtype_of(tape.scalar);
    std::size_t const from_bytes = nk_dtype_bits(from) / 8;
    std::size_t* const offsets = carve<std::size_t>(pages, count + 1);
    if (!offsets) return {{}, status_t::bad_alloc_k};
    offsets[0] = 0;
    std::size_t largest = 0;
    for (std::size_t row = 0; row != count; ++row) {
        std::size_t const cloud = tape.offsets[row + 1] - tape.offsets[row];
        if (geometry.points && cloud != geometry.points) return {{}, status_t::shape_mismatch_k};
        largest = max_of(largest, cloud);
        nk_size_t bytes = 0;
        if (status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(cloud, &bytes)); failed(sized))
            return {{}, sized};
        offsets[row + 1] = offsets[row] + bytes;
    }
    byte_t* const packed = carve<byte_t>(pages, offsets[count]);
    raw_t* const converted = from == dtype_k ? nullptr : carve<raw_t>(pages, largest * 3);
    if (!packed || (from != dtype_k && !converted)) return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count; ++row) {
        std::size_t const cloud = tape.offsets[row + 1] - tape.offsets[row];
        void const* const coordinates = static_cast<byte_t const*>(tape.data) + tape.offsets[row] * 3 * from_bytes;
        raw_t const* cloud_points = static_cast<raw_t const*>(coordinates);
        if (from != dtype_k) {
            if (status_t const cast = us::shapes::status_of(
                    nk_cast_serial(coordinates, from, converted, dtype_k, cloud * 3, nullptr));
                failed(cast))
                return {{}, cast};
            cloud_points = converted;
        }
        if (status_t const made = us::shapes::status_of(
                pack_kit_t::pack(cloud_points, cloud, packed + offsets[row], nullptr));
            failed(made))
            return {{}, made};
    }
    return {{packed, {offsets, count + 1}}, status_t::success_k};
}

template <nk::shapes_dtype scalar_type_, gpu_class_t gpu_class_>
expected<traversal_telemetry_t> shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::insert(
    usearch_shapes_insert_t const& arguments, progress_ticks_t& ticks, cuda_executor_t& executor) noexcept {
    usearch_tape_view_t const& shapes = arguments.shapes;
    std::size_t const count = arguments.count;
    // The batch's own pages, since batches routed to disjoint shards may land at once.
    growing_arena<forkunion_allocator_t> staging {forkunion_allocator_t {no_memory_domain_k}};
    auto const [packs, pack_status] = pack_(shapes, count, staging);
    if (failed(pack_status)) return {{}, pack_status};
    expected<traversal_telemetry_t> const landed = vram_nvidia_insert(
        index, arguments.keys, packs, geometry.connectivity, arguments.expansion_factor, ticks, executor);
    if (failed(landed.status)) return landed;
    fu::standard_atomic_ref<std::size_t>(total_points)
        .add(shapes.offsets[count] - shapes.offsets[0], std::memory_order_relaxed);
    return landed;
}

template <nk::shapes_dtype scalar_type_, gpu_class_t gpu_class_>
status_t shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::save(std::string_view path, cuda_executor_t& executor,
                                                                    progress_ticks_t& ticks) noexcept {
    return with_shapes_provenance(geometry, [&](save_config_t const& config) noexcept {
        return vram_nvidia_save(index, path, codec_t {}, config, token_of(geometry.metric), executor, ticks);
    });
}

template <nk::shapes_dtype scalar_type_, gpu_class_t gpu_class_>
status_t shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::load(std::string_view path, cuda_executor_t& executor,
                                                                    progress_ticks_t& ticks) noexcept {
    if (status_t const loaded = index.load(path, codec_t {}, executor, load_config_t {},
                                           static_cast<connectivity_t>(geometry.connectivity), ticks);
        failed(loaded))
        return loaded;
    // The value columns hold three scalars per point, summed over every part.
    auto const [metadata, status] = safegraphs_read_index_metadata(path);
    if (failed(status)) return status;
    total_points += metadata.value_bytes / (3 * sizeof(typename scalar_type_::raw_t));
    return status_t::success_k;
}

template <nk::shapes_dtype scalar_type_, gpu_class_t gpu_class_>
expected<traversal_telemetry_t> shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::search(
    usearch_shapes_search_t const& arguments, growing_arena<forkunion_allocator_t>& pages, matches_landing_t& landing,
    progress_ticks_t& ticks, cuda_executor_t& executor) noexcept {
    auto const [queries, pack_status] = pack_(arguments.queries, arguments.count, pages);
    if (failed(pack_status)) return {{}, pack_status};
    return vram_nvidia_search(index, queries, arguments.wanted, arguments.expansion_factor, landing, ticks, executor);
}

#endif

#pragma endregion Shapes Variants

#pragma region Shapes Bodies

/** The shapes variants each class unit compiles: one per CPU class and stored scalar on the host
 *  and on storage, and one per stored scalar a GPU class's kits serve. */
template <auto class_>
struct shapes_variants_of<dense_home_t::ram_k, class_> {
    using type = type_list<shapes_ram_variant<nk::f32_t, class_>, shapes_ram_variant<nk::f16_t, class_>>;
};
template <auto class_>
struct shapes_variants_of<dense_home_t::nvme_k, class_> {
    using type = type_list<shapes_nvme_variant<nk::f32_t, class_>, shapes_nvme_variant<nk::f16_t, class_>>;
};
#if USEARCH_ARCH_CUDA_
template <>
struct shapes_variants_of<dense_home_t::vram_nvidia_k, gpu_class_t::nvidia_blackwell_k> {
    using type = type_list<shapes_vram_nvidia_variant<nk::f16_t, gpu_class_t::nvidia_blackwell_k>>;
};
template <>
struct shapes_variants_of<dense_home_t::vram_nvidia_k, gpu_class_t::nvidia_ampere_k> {
    using type = type_list<shapes_vram_nvidia_variant<nk::f32_t, gpu_class_t::nvidia_ampere_k>,
                           shapes_vram_nvidia_variant<nk::f16_t, gpu_class_t::nvidia_ampere_k>>;
};
#endif

/** What the shapes verbs do to one variant: describe it, and export and tally its clouds. */
template <>
struct variant_bodies<shapes_traits_t> {
    using geometry_t = shapes_geometry_t;

    /** What a view keeps of a cloud it read: its points as saved, in its thread's arena. */
    using kept_t = bytes_view_t;

    /** Whether @p donor lands in @p variant: the same kind, scalar, point count and
     *  connectivity. */
    template <typename variant_type_, typename donor_type_>
    static bool matches(variant_type_ const& variant, donor_type_ const& donor) noexcept {
        geometry_t const &index = variant.geometry, &other = donor.geometry;
        return index.metric == other.metric && index.quantization == other.quantization &&
               index.points == other.points && index.connectivity == other.connectivity;
    }

    /** Copies @p geometry into @p variant, a zero connectivity taking the default. */
    template <typename variant_type_>
    static bool freeze(variant_type_& variant, geometry_t& geometry, std::monostate, usearch_outcome_t*) noexcept {
        if (!geometry.connectivity) geometry.connectivity = connectivity_32_k;
        variant.geometry = geometry;
        return true;
    }

    template <typename variant_type_>
    static void describe(variant_type_ const& variant, usearch_shapes_index_description_t& description) noexcept {
        description.metadata = shapes_traits_t::metadata_of(variant.geometry);
        description.total_points =
            fu::standard_atomic_ref<std::size_t const>(variant.total_points).load(std::memory_order_relaxed);
    }

    template <typename variant_type_>
    static bool accepts(variant_type_ const&, usearch_shapes_search_t const& arguments,
                        usearch_outcome_t* outcome) noexcept {
        return accepts_shapes(arguments.queries, arguments.count, outcome);
    }

    template <typename cloud_type_>
    static status_t keep(nvme_executor_t& executor, std::size_t thread, cloud_type_ const& cloud,
                         kept_t& kept) noexcept {
        return keep_row(executor, thread, bytes_of(cloud.data(), cloud.extent(0) * 3), kept);
    }

    /** Lands @p count clouds, the @c i th as @p value_at hands it, as points in the scalar the call
     *  asks for in @p memory, their offsets counting points; a variant's own values are packs,
     *  which unpack through @p memory too where the scalar is not the stored one. */
    template <typename variant_type_, typename value_at_type_, typename arguments_type_>
    static bool land(variant_type_ const& variant, std::size_t count, value_at_type_&& value_at,
                     growing_arena<forkunion_allocator_t>& memory, arguments_type_ const& arguments,
                     usearch_outcome_t* outcome) noexcept {
        nk_dtype_t const exported = nk_dtype_of(arguments.outputs_scalar);
        std::size_t const element_bytes = nk_dtype_bits(exported) / 8;
        std::size_t* const offsets = carve<std::size_t>(memory, count + 1);
        if (!offsets) return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != count; ++entry)
            offsets[entry + 1] = offsets[entry] + variant.points_of(value_at(entry));
        byte_t* const points = carve<byte_t>(memory, offsets[count] * 3 * element_bytes);
        if (offsets[count] && !points)
            return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena");
        thread_allocator_t const scratch {memory};
        for (std::size_t entry = 0; entry != count; ++entry) {
            byte_t* const output = points + offsets[entry] * 3 * element_bytes;
            if (status_t const exported_status = variant.export_points(value_at(entry), exported, output, scratch);
                failed(exported_status))
                return propagate_error(exported_status, outcome);
        }
        *arguments.shapes = {points, offsets, arguments.outputs_scalar};
        return true;
    }

    /** Adds one entry's points to a key's statistics, from any thread: a variant's packs count
     *  their points, and a view's clouds hold a row per point. */
    template <typename cloud_type_>
    static void tally(usearch_shapes_key_stats_t& stats, cloud_type_ const& cloud) noexcept {
        std::size_t points = 0;
        if constexpr (requires { cloud.max_points(); }) points = cloud.max_points();
        else points = cloud.extent(0);
        fu::standard_atomic_ref<std::size_t>(stats.total_points).add(points, std::memory_order_relaxed);
        raise_to<fu::standard_atomic_ref>(stats.largest_points, points);
    }
};

#pragma endregion Shapes Bodies
