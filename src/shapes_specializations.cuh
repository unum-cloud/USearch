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
#include <memory>      // `std::allocator_traits` - a query's packs on a thread's arena
#include <span>        // `std::span` - a cloud's coordinates
#include <string_view> // `std::string_view` - tokens and paths

#include "usearch/shapes/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/shapes/vram_nvidia_index.hpp"
#endif

#include "basic_specializations.cuh"

#pragma region Shapes Variants

/** The checkpoint token of one shapes kind, spelled as NumKong spells its kernels. */
inline std::string_view token_of(usearch_shapes_metric_t metric) noexcept {
    switch (metric) {
    case usearch_shapes_metric_chamfer_k: return "chamfer";
    case usearch_shapes_metric_hausdorff_k: return "hausdorff";
    case usearch_shapes_metric_gromov_wasserstein_k: return "gromov_wasserstein";
    case usearch_shapes_metric_wasserstein_procrustes_k: return "wasserstein_procrustes";
    default: return {};
    }
}

/** The shapes kind a checkpoint token names, or @c unknown_k. */
inline usearch_shapes_metric_t shapes_metric_of(std::string_view token) noexcept {
    for (usearch_shapes_metric_t metric :
         {usearch_shapes_metric_chamfer_k, usearch_shapes_metric_hausdorff_k,
          usearch_shapes_metric_gromov_wasserstein_k, usearch_shapes_metric_wasserstein_procrustes_k})
        if (token == token_of(metric)) return metric;
    return usearch_shapes_metric_unknown_k;
}

/** The NumKong kernel one shapes kind scores with. */
constexpr nk_kernel_kind_t kernel_of(usearch_shapes_metric_t metric) noexcept {
    switch (metric) {
    case usearch_shapes_metric_hausdorff_k: return nk_kernel_hausdorffs_packed_k;
    case usearch_shapes_metric_gromov_wasserstein_k: return nk_kernel_gromovwassersteins_packed_k;
    case usearch_shapes_metric_wasserstein_procrustes_k: return nk_kernel_wassersteinprocrustes_packed_k;
    default: return nk_kernel_chamfers_packed_k;
    }
}

/** The kind, scalar, point count and connectivity a shapes handle freezes, as an init or a
 *  checkpoint states them; every variant holds one. */
struct shapes_geometry_t {
    usearch_shapes_metric_t metric = usearch_shapes_metric_unknown_k;

    /** The scalar every point is stored in. */
    usearch_scalar_t quantization = usearch_scalar_unknown_k;

    /** The frozen point count, or zero for any count. */
    std::size_t points = 0;

    /** Neighbors per node the insertions select; zero until frozen, which takes the default. */
    std::size_t connectivity = 0;
};

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

/** One shapes alternative: the basic index over packs of @p scalar_type_, the kind it scores with,
 *  and the kits @p cpu_class_ picks; verbs defined below the class, like the sequence ones. */
template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
struct shapes_ram_variant {
    using executor_t = forkunion_executor_t;
    using scalar_t = scalar_type_;
    using kit_t = typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(cpu_class_)>::kit_t;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    using index_t = us::shapes::ram_index<kit_t, us::utilities::uuid_t, waiter_of<cpu_class_>,
                                          atomic_ref_for_cpu_class<cpu_class_>::template type, forkunion_allocator_t,
                                          cache_hints_t>;
    using value_t = typename index_t::value_t;
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
        if (succeeded(cleared)) standard_atomic_ref<std::size_t>(total_points).store(0, std::memory_order_relaxed);
        return cleared;
    }

    expected<traversal_telemetry_t> insert(usearch_shapes_insert_t const& arguments, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;
    expected<traversal_telemetry_t> search(usearch_shapes_search_t const& arguments, matches_landing_t& landing,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** Opens a refinement once @p donors land, its walks as wide as @p expansion or twice the
     *  connectivity, as an insertion's. */
    expected<refine_plan_t> plan_refine(std::span<index_t*> donors, std::size_t expansion) noexcept {
        typename index_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_absorb(donors, planning, basic::keep_diverse_t {geometry.connectivity}, {expansion});
    }
    status_t refine(refine_plan_t& plan, progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** Adds the points of a landed @p donor, whose rows this index now holds. */
    void absorbed(shapes_ram_variant const& donor) noexcept {
        standard_atomic_ref<std::size_t>(total_points).fetch_add(donor.total_points, std::memory_order_relaxed);
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
        standard_atomic_ref<std::size_t>(total_points)
            .fetch_add(shapes.offsets[count] - shapes.offsets[0], std::memory_order_relaxed);
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
    auto [scratch, prepare_status] = prepare_refinement<index_t>(
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

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
status_t shapes_ram_variant<scalar_type_, cpu_class_>::save(std::string_view path, executor_t& executor,
                                                            progress_ticks_t& ticks) noexcept {
    std::array<char, 24> connectivity_text, points_text;
    std::array<safegraphs_custom_field_t, 2> const fields {{
        {"connectivity", decimal_of(geometry.connectivity, connectivity_text)},
        {"points", decimal_of(geometry.points, points_text)},
    }};
    save_config_t config;
    config.provenance = fields;
    return safegraphs_write_index(path, index, codec_t {}, executor, config, token_of(geometry.metric), ticks);
}

template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
status_t shapes_ram_variant<scalar_type_, cpu_class_>::load(std::string_view path, executor_t& executor,
                                                            progress_ticks_t& ticks) noexcept {
    if (status_t const loaded = safegraphs_read_index(path, index, codec_t {}, executor, load_config_t {}, ticks);
        failed(loaded))
        return loaded;
    std::size_t const nodes = index.size();
    for (std::size_t node = 0; node != nodes; ++node)
        total_points += index.published_value_of(static_cast<typename index_t::node_id_t>(node)).max_points();
    return status_t::success_k;
}

/** One shapes alternative on block storage: the basic NVMe index over the clouds of @p scalar_type_
 *  a RAM variant saved, read-only, each row its points as saved, scored with the RAM variant's
 *  metric and the kits @p cpu_class_ picks. Its verbs are instantiated in the class's unit, as the
 *  RAM variant's are. */
template <nk::shapes_dtype scalar_type_, cpu_class_t cpu_class_>
struct shapes_nvme_variant {
    using executor_t = nvme_executor_t;
    using scalar_t = scalar_type_;
    using kit_t = typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(cpu_class_)>::kit_t;
    using codec_t = us::shapes::codec<kit_t>;
    using metric_t = us::shapes::shapes_metric<kit_t, thread_allocator_t>;
    using index_t = basic::nvme_index<codec_t, us::utilities::uuid_t, u32_t, typename kit_t::distance_t,
                                      default_io_engine_t, row_kit_of<cpu_class_>>;
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

/** The fatbin of every shapes kit, defined in @c src/shapes/vram_variants_nvidia.cu. */
struct shapes_nvidia_fatbin_t {
    static bytes_view_t bytes() noexcept;
};

/** The capabilities the shapes kits compile under on NVIDIA: the Ampere floor, which one cubin per
 *  class carries. */
inline constexpr nk_capability_t shapes_nvidia_capabilities_k = nk_cap_cuda_k | nk_cap_ampere_k;

/** The faces' members reading the kits, defined in @c src/shapes/vram_variants_nvidia.cu. */
extern template struct us::shapes::vram_nvidia_index<nk::f16_t, u32_t, shapes_nvidia_capabilities_k,
                                                     shapes_nvidia_fatbin_t>;
extern template struct us::shapes::vram_nvidia_index<nk::f32_t, u32_t, shapes_nvidia_capabilities_k,
                                                     shapes_nvidia_fatbin_t>;

/** The shapes alternative on one NVIDIA GPU: the basic VRAM index over packs of @p scalar_type_
 *  under the kernel kind @p kernel_, Chamfer or Hausdorff, with 32-bit node IDs. A batch packs its
 *  clouds on the host end to end through the serial kit, as the RAM variants pack them. */
template <nk::shapes_dtype scalar_type_, nk_kernel_kind_t kernel_>
struct shapes_vram_nvidia_variant
    : vram_nvidia_variant<
          us::shapes::vram_nvidia_index<scalar_type_, u32_t, shapes_nvidia_capabilities_k, shapes_nvidia_fatbin_t>> {
    using scalar_t = scalar_type_;
    using pack_kit_t = typename us::shapes::shapes_ladder<scalar_type_, nk_cap_serial_k>::kit_t;
    static constexpr nk_dtype_t dtype_k = scalar_type_::dtype();

    /** Places the index on CUDA device @p device, scoring under @p kernel_. */
    explicit shapes_vram_nvidia_variant(int device) noexcept
        : vram_nvidia_variant<
              us::shapes::vram_nvidia_index<scalar_type_, u32_t, shapes_nvidia_capabilities_k, shapes_nvidia_fatbin_t>>(
              device, kernel_) {}

    shapes_geometry_t geometry;

    /** Points across every stored entry, added to by insertions through @c atomic_ref. */
    std::size_t total_points = 0;

    /** Claimed by the one batch packing its clouds, landing its keys and inserting its rows,
     *  which another answers with @c contended_k meanwhile. */
    claim_word_t claims;

    /** The host pages a batch packs its clouds into, rewound per batch. */
    growing_arena<forkunion_allocator_t> staging {forkunion_allocator_t {no_memory_domain_k}};

    /** Reserves the entries and points @p arguments name, a pack taking a header and a partial
     *  alignment line per axis past its points. */
    status_t reserve(usearch_shapes_reserve_t const& arguments) noexcept {
        nk_size_t header = 0;
        if (status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(0, &header)); failed(sized))
            return sized;
        std::size_t const pack_overhead = header + 3 * nk_default_alignment_k;
        std::size_t const bytes = arguments.entries * pack_overhead +
                                  arguments.points * 3 * sizeof(typename scalar_type_::raw_t);
        return this->reserve_(arguments.entries, geometry.connectivity, bytes);
    }

    expected<traversal_telemetry_t> insert(usearch_shapes_insert_t const& arguments, progress_ticks_t& ticks,
                                           cuda_executor_t& executor) noexcept;

    /** Packs the queries into @p pages, the call arena's host pages, where the matches land too. A
     *  device walk takes no host predicate. */
    expected<traversal_telemetry_t> search(usearch_shapes_search_t const& arguments,
                                           growing_arena<forkunion_allocator_t>& pages, matches_landing_t& landing,
                                           progress_ticks_t& ticks, cuda_executor_t& executor) noexcept;

  private:

    /** Packs the @p count clouds of @p tape end to end into @p pages at @p packed, their byte
     *  starts at @p offsets, casting the caller's scalar there too where it is not the stored one;
     *  refuses a count other than a frozen one. */
    status_t pack_(usearch_tape_view_t tape, std::size_t count, growing_arena<forkunion_allocator_t>& pages,
                   byte_t*& packed, std::size_t*& offsets) const noexcept;
};

template <nk::shapes_dtype scalar_type_, nk_kernel_kind_t kernel_>
status_t shapes_vram_nvidia_variant<scalar_type_, kernel_>::pack_(usearch_tape_view_t tape, std::size_t count,
                                                                  growing_arena<forkunion_allocator_t>& pages,
                                                                  byte_t*& packed,
                                                                  std::size_t*& offsets) const noexcept {
    using raw_t = typename scalar_type_::raw_t;
    nk_dtype_t const from = nk_dtype_of(tape.scalar);
    std::size_t const from_bytes = nk_dtype_bits(from) / 8;
    offsets = carve<std::size_t>(pages, count + 1);
    if (!offsets) return status_t::bad_alloc_k;
    offsets[0] = 0;
    std::size_t largest = 0;
    for (std::size_t row = 0; row != count; ++row) {
        std::size_t const cloud = tape.offsets[row + 1] - tape.offsets[row];
        if (geometry.points && cloud != geometry.points) return status_t::shape_mismatch_k;
        largest = max_of(largest, cloud);
        nk_size_t bytes = 0;
        if (status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(cloud, &bytes)); failed(sized))
            return sized;
        offsets[row + 1] = offsets[row] + bytes;
    }
    packed = carve<byte_t>(pages, offsets[count]);
    raw_t* const converted = from == dtype_k ? nullptr : carve<raw_t>(pages, largest * 3);
    if (!packed || (from != dtype_k && !converted)) return status_t::bad_alloc_k;
    for (std::size_t row = 0; row != count; ++row) {
        std::size_t const cloud = tape.offsets[row + 1] - tape.offsets[row];
        void const* const coordinates = static_cast<byte_t const*>(tape.data) + tape.offsets[row] * 3 * from_bytes;
        raw_t const* cloud_points = static_cast<raw_t const*>(coordinates);
        if (from != dtype_k) {
            if (status_t const cast = us::shapes::status_of(
                    nk_cast_serial(coordinates, from, converted, dtype_k, cloud * 3, nullptr));
                failed(cast))
                return cast;
            cloud_points = converted;
        }
        if (status_t const made = us::shapes::status_of(
                pack_kit_t::pack(cloud_points, cloud, packed + offsets[row], nullptr));
            failed(made))
            return made;
    }
    return status_t::success_k;
}

template <nk::shapes_dtype scalar_type_, nk_kernel_kind_t kernel_>
expected<traversal_telemetry_t> shapes_vram_nvidia_variant<scalar_type_, kernel_>::insert(
    usearch_shapes_insert_t const& arguments, progress_ticks_t& ticks, cuda_executor_t& executor) noexcept {
    usearch_tape_view_t const& shapes = arguments.shapes;
    std::size_t const count = arguments.count;
    auto const claimed = claims.claim(this->index.size(), count, this->index.max_node_count_k);
    if (failed(claimed.status)) return {{}, claimed.status};
    expected<traversal_telemetry_t> const landed = [&]() noexcept -> expected<traversal_telemetry_t> {
        staging.clear();
        byte_t* packed = nullptr;
        std::size_t* offsets = nullptr;
        if (status_t const packed_status = pack_(shapes, count, staging, packed, offsets); failed(packed_status))
            return {{}, packed_status};
        return this->insert_(arguments.keys, packed, {offsets, count + 1}, claimed.value, geometry.connectivity,
                             arguments.expansion_factor, ticks, executor);
    }();
    if (failed(landed.status)) return claims.unclaim(claimed.value, count), landed;
    standard_atomic_ref<std::size_t>(total_points)
        .fetch_add(shapes.offsets[count] - shapes.offsets[0], std::memory_order_relaxed);
    return landed;
}

template <nk::shapes_dtype scalar_type_, nk_kernel_kind_t kernel_>
expected<traversal_telemetry_t> shapes_vram_nvidia_variant<scalar_type_, kernel_>::search(
    usearch_shapes_search_t const& arguments, growing_arena<forkunion_allocator_t>& pages, matches_landing_t& landing,
    progress_ticks_t& ticks, cuda_executor_t& executor) noexcept {
    byte_t* packed = nullptr;
    std::size_t* offsets = nullptr;
    // Packs only when a match is wanted, as a walk for none answers before reading any query.
    if (arguments.wanted)
        if (status_t const packed_status = pack_(arguments.queries, arguments.count, pages, packed, offsets);
            failed(packed_status))
            return {{}, packed_status};
    return this->search_(packed, offsets, arguments.count, arguments.wanted, arguments.expansion_factor, landing, ticks,
                         executor);
}

#endif

#pragma endregion Shapes Variants
