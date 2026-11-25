/**
 *  @file src/shapes_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief What one shapes variant is spelled with and compiled through: the kinds' tokens and
 *      kernels, the geometry every shapes variant adopts, the metric factory, the packing of
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
#include <variant>     // `std::monostate` - nothing a variant adopts past its geometry

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the scan totals

#include "usearch/shapes/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/shapes/vram_nvidia_metric.cuh"
#endif
#if USEARCH_ARCH_METAL_
#include "usearch/shapes/vram_apple_index.hpp"
#endif
#if USEARCH_ARCH_ROCM_
#include "usearch/shapes/vram_amd_metric.hip"
#endif

#include "basic_specializations.cuh"

#pragma region Shapes Variants

namespace util = unum::usearch::utilities;

/** The NumKong kernel one shapes kind scores with. */
constexpr nk_kernel_kind_t kernel_of(usearch_shapes_metric_t metric) noexcept {
    switch (metric) {
    case usearch_shapes_metric_hausdorff_k: return nk_kernel_hausdorffs_packed_k;
    case usearch_shapes_metric_gromov_wasserstein_k: return nk_kernel_gromovwassersteins_packed_k;
    case usearch_shapes_metric_wasserstein_procrustes_k: return nk_kernel_wassersteinprocrustes_packed_k;
    default: return nk_kernel_chamfers_packed_k;
    }
}

/** A factory of @p metric_type_ under @p metric over a thread's arena. */
template <typename metric_type_>
auto shapes_metric_factory(usearch_shapes_metric_t metric) noexcept {
    return [kernel = kernel_of(metric)](thread_allocator_t const& allocator, std::size_t) noexcept {
        return util::expected<metric_type_> {metric_type_ {kernel, allocator}, util::status_code_t::success_k};
    };
}

/** One shapes alternative: the basic shards over packs of @p scalar_type_, the kind they score
 *  with, and the kits @p cpu_class_ picks; verbs defined below the class, like the sequence ones.
 *  Checkpoints and refinements serve an index of one shard. */
template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
struct shapes_ram_variant {
    using executor_t = util::ram_executor_t;
    template <typename word_type_>
    using atomic_ref = typename atomic_ref_for_cpu_class<cpu_class_>::template type<word_type_>;
    using scalar_t = scalar_type_;
    using kit_t = typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(cpu_class_)>::kit_t;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    using shard_t = us::shapes::ram_index<kit_t, util::uuid128_t, waiter_of<cpu_class_>,
                                          atomic_ref_for_cpu_class<cpu_class_>::template type,
                                          util::forkunion_allocator_t, cache_hints_t>;
    using index_t = basic::ram_indexes<shard_t>;
    using value_t = typename shard_t::value_t;
    using codec_t = us::shapes::codec<kit_t>;
    using metric_t = us::shapes::shapes_metric<kit_t, thread_allocator_t>;
    using refine_plan_t = typename index_t::refine_plan_t;
    using reshard_plan_t = typename index_t::reshard_plan_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr nk_dtype_t dtype_k = scalar_type_::dtype();

    index_t index;
    shapes_geometry_t geometry;

    /** Points across every stored entry, added to by insertions and loads through @c atomic_ref. */
    std::size_t total_points = 0;

    explicit shapes_ram_variant(util::forkunion_allocator_t const& allocator) noexcept
        : index(allocator, machine_cache_line_bytes()) {}

    /** Reserves the entries @p capacity names. */
    util::status_t reserve(reserve_capacity_t capacity) noexcept { return index.reserve(capacity.entries).status; }

    /** Empties the index and its point count. */
    util::status_t clear() noexcept {
        util::status_t const cleared = index.clear();
        if (succeeded(cleared)) atomic_ref<std::size_t>(total_points).store(0, std::memory_order_relaxed);
        return cleared;
    }

    util::expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t shapes, insert_config_t config,
                                             executor_t& executor, progress_ticks_t& ticks) noexcept;
    util::expected<traversal_stats_t> search(counted_tape_t queries, usearch_matches_view_t const* hints,
                                             matches_outputs_t& outputs, search_config_t config,
                                             predicate_t const& predicate, executor_t& executor,
                                             progress_ticks_t& ticks) noexcept;

    /** Opens one round of every live shard in place under the config's cache. */
    util::expected<refine_plan_t> plan_refine(refine_config_t config) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_refine({config.cache_bytes}, basic::keep_diverse_t {geometry.connectivity}, codec_t {},
                                 planning);
    }

    /** Opens a reshard of @p sources into @p targets driven by this index, the one target of a
     *  pack or the one source of a split, a split's part roots drawn from @p seed, then one round
     *  per shard that took entries. */
    util::expected<reshard_plan_t> plan_reshard(std::span<index_t* const> sources, std::span<index_t* const> targets,
                                                refine_config_t config, util::u64_t seed) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_reshard(sources, targets, {{config.cache_bytes}, 0, seed},
                                  basic::keep_diverse_t {geometry.connectivity}, codec_t {}, planning);
    }

    /** Drives @p plan 's pending phases on the executor's threads until its round ends or @p ticks
     *  cancels. */
    template <typename plan_type_>
    util::status_t drive(plan_type_& plan, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Takes over the points of an emptied @p source, whose nodes this index now holds. */
    void count_packed_source(shapes_ram_variant& source) noexcept {
        std::size_t const points = atomic_ref<std::size_t>(source.total_points).exchange(0, std::memory_order_relaxed);
        atomic_ref<std::size_t>(total_points).store_add(points, std::memory_order_relaxed);
    }

    /** Counts the points of the part @p target now holds, once a split emptied this index. */
    void count_split_target(shapes_ram_variant& target) noexcept {
        std::size_t points = 0;
        for (shard_t const& one : target.index.live_shards())
            for (std::size_t node = 0; node != one.size(); ++node)
                points += points_of(one.published_value_of(static_cast<typename shard_t::node_id_t>(node)));
        atomic_ref<std::size_t>(target.total_points).store(points, std::memory_order_relaxed);
        atomic_ref<std::size_t>(total_points).store(0, std::memory_order_relaxed);
    }

    /** The points of one stored value. */
    static std::size_t points_of(value_t const& value) noexcept { return value.max_points(); }

    /** Writes the @p entries points of one stored value into @p output in @p dtype, through
     *  @p scratch when the dtype is not the stored one. */
    util::status_t export_points(value_t const& value, nk_dtype_t dtype, void* output,
                                 thread_allocator_t const& scratch) const noexcept;

    /** Writes the checkpoint under a path, or into a buffer of @ref storage_requirements bytes. */
    template <typename checkpoint_type_>
    util::status_t save(checkpoint_type_ checkpoint, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Reads the checkpoint at a path or in a buffer, then counts its points. */
    template <typename checkpoint_type_>
    util::status_t load(checkpoint_type_ checkpoint, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** The bytes a save into memory takes, under the provenance the save records. */
    util::expected<std::size_t> storage_requirements() const noexcept;

    /** Serving a checkpoint in place answers @c unsupported_k until the index serves it. */
    util::status_t view(std::string_view) noexcept { return util::status_code_t::unsupported_k; }
    util::expected<util::sample_result_t> sample(basic_sample_inputs_t inputs, basic_sample_outputs_t outputs,
                                                 util::sample_config_t config, executor_t& executor) noexcept;
};

/** Packs entry @p entry of @p tape into @p packed under @p allocator, casting the caller's scalar
 *  into @p scalar_type_ with @p kit_type_ through @p scratch where they differ; refuses a count
 *  other than a frozen nonzero @p points. */
template <nk::a_shapes_dtype scalar_type_, typename kit_type_, typename packed_type_, typename allocator_type_>
util::status_t pack_entry(counted_tape_t const& tape, std::size_t entry, std::size_t points, packed_type_& packed,
                          allocator_type_ const& allocator, thread_allocator_t const& scratch) noexcept {
    constexpr nk_dtype_t dtype_k = scalar_type_::dtype();
    std::size_t const count = tape.offsets[entry + 1] - tape.offsets[entry];
    if (points && count != points) return util::status_code_t::shape_mismatch_k;
    nk_dtype_t const from = nk_dtype_of(tape.scalar);
    std::size_t const from_bytes = nk_dtype_bits(from) / 8;
    std::size_t const elements = count * 3;
    util::bytes_view_t const coordinates = util::bytes_of(
        static_cast<util::byte_t const*>(tape.elements) + tape.offsets[entry] * 3 * from_bytes, elements * from_bytes);
    util::growing_buffer<scalar_type_, thread_allocator_t> converted {scratch};
    std::span<scalar_type_ const> cloud = util::elements_of<scalar_type_>(coordinates);
    if (from != dtype_k) {
        if (failed(converted.resize_uninitialized(elements))) return util::status_code_t::out_of_memory_k;
        if (util::status_t const cast = us::shapes::status_of(
                kit_type_::cast(coordinates.data(), from, converted.data(), dtype_k, elements, nullptr));
            failed(cast))
            return cast;
        cloud = to_span(converted);
    }
    std::array const clouds {cloud};
    auto [made, status] = packed_type_::make(clouds, allocator, kit_type_::pack_size, kit_type_::pack);
    if (nk::failed(status)) return us::shapes::status_of(status);
    packed = std::move(made);
    return util::status_code_t::success_k;
}

template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
util::expected<traversal_stats_t> shapes_ram_variant<scalar_type_, cpu_class_>::insert(
    usearch_keys_view_t keys, caller_tape_t shapes, insert_config_t config, executor_t& executor,
    progress_ticks_t& ticks) noexcept {
    std::size_t const connectivity = geometry.connectivity;
    auto build = [&](counted_tape_t const& entries, std::size_t offset, value_t& value, std::size_t thread) noexcept {
        return pack_entry<scalar_type_, kit_t>(entries, offset, geometry.points, value, value.get_allocator(),
                                               thread_allocator_t {executor.arena(thread)});
    };
    std::size_t const expansion = config.expansion_factor ? config.expansion_factor : connectivity * 2;
    auto const [inserted, insert_status] = insert_entries(index, keys, shapes, connectivity, expansion,
                                                          shapes_metric_factory<metric_t>(geometry.metric), build,
                                                          ticks, executor);
    if (succeeded(insert_status))
        atomic_ref<std::size_t>(total_points).store_add(elements_in(shapes), std::memory_order_relaxed);
    return {inserted, insert_status};
}

template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
util::expected<traversal_stats_t> shapes_ram_variant<scalar_type_, cpu_class_>::search(
    counted_tape_t queries, usearch_matches_view_t const* hints, matches_outputs_t& outputs, search_config_t config,
    predicate_t const& predicate, executor_t& executor, progress_ticks_t& ticks) noexcept {
    using query_t = nk::shapes_packed<scalar_type_,
                                      typename std::allocator_traits<thread_allocator_t>::template rebind_alloc<char>>;
    auto query_of = [&](std::size_t query_index, std::size_t thread, auto&& answer) noexcept {
        thread_allocator_t const scratch {executor.arena(thread)};
        query_t query {scratch};
        if (util::status_t const packed = pack_entry<scalar_type_, kit_t>(queries, query_index, geometry.points, query,
                                                                          scratch, scratch);
            failed(packed))
            return packed;
        return answer(query);
    };
    return search_entries(index, queries.count(), geometry.connectivity,
                          shapes_metric_factory<metric_t>(geometry.metric), query_of, outputs, config, predicate,
                          executor, ticks, hints);
}

template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
util::expected<util::sample_result_t> shapes_ram_variant<scalar_type_, cpu_class_>::sample(
    basic_sample_inputs_t inputs, basic_sample_outputs_t outputs, util::sample_config_t config,
    executor_t& executor) noexcept {
    return sample_entries(index, geometry.connectivity, shapes_metric_factory<metric_t>(geometry.metric), inputs,
                          outputs, config, executor);
}

template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
template <typename plan_type_>
util::status_t shapes_ram_variant<scalar_type_, cpu_class_>::drive(plan_type_& plan, executor_t& executor,
                                                                   progress_ticks_t& ticks) noexcept {
    auto [scratch, prepare_status] = prepare_refinement<shard_t>(
        plan, geometry.connectivity, shapes_metric_factory<metric_t>(geometry.metric), executor);
    if (failed(prepare_status)) return prepare_status;
    return drive_refinement(plan, scratch, basic::keep_diverse_t {geometry.connectivity}, ticks, executor);
}

template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
util::status_t shapes_ram_variant<scalar_type_, cpu_class_>::export_points(
    value_t const& value, nk_dtype_t dtype, void* output, thread_allocator_t const& scratch) const noexcept {
    if (value.empty()) return util::status_code_t::success_k;
    std::size_t const elements = value.max_points() * 3;
    if (dtype == dtype_k)
        return us::shapes::status_of(
            kit_t::unpack(value[0], static_cast<typename scalar_type_::raw_t*>(output), nullptr));
    util::growing_buffer<typename scalar_type_::raw_t, thread_allocator_t> unpacked {scratch};
    if (failed(unpacked.resize_uninitialized(elements))) return util::status_code_t::out_of_memory_k;
    if (util::status_t const opened = us::shapes::status_of(kit_t::unpack(value[0], unpacked.data(), nullptr));
        failed(opened))
        return opened;
    return us::shapes::status_of(kit_t::cast(unpacked.data(), dtype_k, output, dtype, elements, nullptr));
}

/** Hands @p write, as <tt>write(config) → status_t</tt>, the save configuration recording what
 *  @p geometry froze, the connectivity and the point count, as every residency's checkpoint
 *  records them. */
template <typename write_type_>
util::status_t with_shapes_provenance(shapes_geometry_t const& geometry, write_type_&& write) noexcept {
    std::array<char, 24> connectivity_text, points_text;
    std::array<util::safegraphs_custom_field_t, 2> const fields {{
        {"connectivity", decimal_of(geometry.connectivity, connectivity_text)},
        {"points", decimal_of(geometry.points, points_text)},
    }};
    util::save_config_t config;
    config.provenance = fields;
    return write(config);
}

template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
template <typename checkpoint_type_>
util::status_t shapes_ram_variant<scalar_type_, cpu_class_>::save(checkpoint_type_ checkpoint, executor_t& executor,
                                                                  progress_ticks_t& ticks) noexcept {
    return with_shapes_provenance(geometry, [&](util::save_config_t const& config) noexcept {
        return index.save(checkpoint, codec_t {}, executor, config, token_of(geometry.metric), ticks);
    });
}

template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
util::expected<std::size_t> shapes_ram_variant<scalar_type_, cpu_class_>::storage_requirements() const noexcept {
    std::size_t required = 0;
    util::status_t const sized = with_shapes_provenance(geometry, [&](util::save_config_t const& config) noexcept {
        auto const [bytes, bytes_status] = index.storage_requirements(codec_t {}, config, token_of(geometry.metric));
        required = bytes;
        return bytes_status;
    });
    return {required, sized};
}

template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
template <typename checkpoint_type_>
util::status_t shapes_ram_variant<scalar_type_, cpu_class_>::load(checkpoint_type_ checkpoint, executor_t& executor,
                                                                  progress_ticks_t& ticks) noexcept {
    if (util::status_t const loaded = index.load(checkpoint, codec_t {}, executor, util::load_config_t {}, ticks);
        failed(loaded))
        return loaded;
    for (shard_t const& one : index.live_shards())
        for (std::size_t node = 0; node != one.size(); ++node)
            total_points += one.published_value_of(static_cast<typename shard_t::node_id_t>(node)).max_points();
    return util::status_code_t::success_k;
}

/** One shapes alternative on block storage: the basic NVMe index over the clouds of @p scalar_type_
 *  a RAM variant saved, read-only, each shape its points as saved, scored with the RAM variant's
 *  metric and the kits @p cpu_class_ picks. Its verbs are instantiated in the class's unit, as the
 *  RAM variant's are. */
template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
struct shapes_nvme_variant {
    using executor_t = nvme_executor_t;
    template <typename word_type_>
    using atomic_ref = typename atomic_ref_for_cpu_class<cpu_class_>::template type<word_type_>;
    using scalar_t = scalar_type_;
    using kit_t = typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(cpu_class_)>::kit_t;
    using codec_t = us::shapes::codec<kit_t>;
    using metric_t = us::shapes::shapes_metric<kit_t, thread_allocator_t>;
    using index_t = basic::nvme_indexes<codec_t, util::uuid128_t, util::u32_t, typename kit_t::distance_t,
                                        default_io_queue_t, row_kit_of<cpu_class_>,
                                        atomic_ref_for_cpu_class<cpu_class_>::template type, waiter_of<cpu_class_>>;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr nk_dtype_t dtype_k = scalar_type_::dtype();

    index_t index;
    shapes_geometry_t geometry;

    /** Points across every stored entry, counted once by the view. */
    std::size_t total_points = 0;

    explicit shapes_nvme_variant(util::forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    /** Serves the checkpoint at @p path in place, its parts from @p paths when any are named. */
    util::status_t view(std::string_view path, std::span<std::string_view const> paths, executor_t& executor) noexcept {
        util::status_t const viewed = index.view(path, paths, token_of(geometry.metric), executor);
        total_points = index.value_elements() / 3;
        return viewed;
    }

    util::expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                             predicate_t const& predicate, executor_t& executor,
                                             progress_ticks_t& ticks) noexcept;

    /** The points of one stored @p cloud, its bytes as saved, three coordinates apiece. */
    static std::size_t points_of(util::bytes_view_t cloud) noexcept {
        return cloud.size() / (3 * sizeof(typename scalar_type_::raw_t));
    }

    /** Writes the points of one stored @p cloud, its bytes as saved, into @p output in @p dtype; a
     *  cast reads them in place, so it needs no scratch. */
    util::status_t export_points(util::bytes_view_t cloud, nk_dtype_t dtype, void* output,
                                 thread_allocator_t const&) const noexcept {
        if (dtype != dtype_k)
            return us::shapes::status_of(
                kit_t::cast(cloud.data(), dtype_k, output, dtype, cloud.size() / sizeof(scalar_type_), nullptr));
        if (!cloud.empty()) std::memcpy(output, cloud.data(), cloud.size());
        return util::status_code_t::success_k;
    }

    /** A view already is its checkpoint, so saving answers @c unsupported_k. */
    util::status_t save(std::string_view, executor_t&, progress_ticks_t&) noexcept {
        return util::status_code_t::unsupported_k;
    }
    util::expected<util::sample_result_t> sample(basic_sample_inputs_t inputs, basic_sample_outputs_t outputs,
                                                 util::sample_config_t config, executor_t& executor) noexcept;
};

template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
util::expected<traversal_stats_t> shapes_nvme_variant<scalar_type_, cpu_class_>::search(
    counted_tape_t queries, matches_outputs_t& outputs, search_config_t config, predicate_t const& predicate,
    executor_t& executor, progress_ticks_t& ticks) noexcept {
    using query_t = nk::shapes_packed<scalar_type_,
                                      typename std::allocator_traits<thread_allocator_t>::template rebind_alloc<char>>;
    auto query_of = [&](std::size_t query_index, thread_allocator_t const& scratch) noexcept {
        query_t query {scratch};
        util::status_t const packed = pack_entry<scalar_type_, kit_t>(queries, query_index, geometry.points, query,
                                                                      scratch, scratch);
        return util::expected<query_t> {std::move(query), packed};
    };
    return search_entries(index, queries.count(), shapes_metric_factory<metric_t>(geometry.metric), query_of, outputs,
                          config, predicate, executor, ticks);
}

/** Packs each stored cloud into a query as a search packs a tape entry, its points already in the
 *  stored scalar. */
template <nk::a_shapes_dtype scalar_type_, cpu_class_t cpu_class_>
util::expected<util::sample_result_t> shapes_nvme_variant<scalar_type_, cpu_class_>::sample(
    basic_sample_inputs_t inputs, basic_sample_outputs_t outputs, util::sample_config_t config,
    executor_t& executor) noexcept {
    using query_t = nk::shapes_packed<scalar_type_,
                                      typename std::allocator_traits<thread_allocator_t>::template rebind_alloc<char>>;
    auto query_of = [&](typename codec_t::view_t stored, thread_allocator_t const& scratch) noexcept {
        std::array<std::size_t, 2> const offsets {0, stored.extent(0)};
        counted_tape_t const cloud {stored.data(), offsets, usearch_scalar_of(dtype_k)};
        query_t query {scratch};
        util::status_t const packed = pack_entry<scalar_type_, kit_t>(cloud, 0, geometry.points, query, scratch,
                                                                      scratch);
        return util::expected<query_t> {std::move(query), packed};
    };
    return sample_entries(*this, shapes_metric_factory<metric_t>(geometry.metric), query_of, inputs, outputs, config,
                          executor);
}

#if USEARCH_ARCH_CUDA_

/** One class's shapes kits over one dtype as a fatbin, defined in its class unit,
 *  @c src/shapes/vram_variants_nvidia.cpp for the floor or a monoversioned build's one class,
 *  beside the bytes it embeds. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct shapes_nvidia_fatbin {
    static bytes_view_t bytes() noexcept;
};

/* The faces' members reading the kits, defined in each class unit. */
extern template struct us::shapes::vram_nvidia_index<
    typename us::shapes::shapes_ladder<nk::f16_t, numkong_capabilities_of(gpu_class_t::nvidia_ampere_k)>::kit_t, u32_t,
    shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f16_k>>;
extern template struct us::shapes::vram_nvidia_index<
    typename us::shapes::shapes_ladder<nk::f32_t, numkong_capabilities_of(gpu_class_t::nvidia_ampere_k)>::kit_t, u32_t,
    shapes_nvidia_fatbin<gpu_class_t::nvidia_ampere_k, nk_f32_k>>;
extern template struct us::shapes::vram_nvidia_index<
    typename us::shapes::shapes_ladder<nk::f16_t, numkong_capabilities_of(gpu_class_t::nvidia_blackwell_k)>::kit_t,
    u32_t, shapes_nvidia_fatbin<gpu_class_t::nvidia_blackwell_k, nk_f16_k>>;

/** Builds one shard of the shapes face over packs of @p scalar_type_ for @p gpu_class_ on a CUDA
 *  device, scoring under the kernel the variant's frozen geometry names. */
template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
struct shapes_vram_nvidia_build {
    using face_t = us::shapes::vram_nvidia_index<
        typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(gpu_class_)>::kit_t, u32_t,
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
 *  IDs. A batch packs its clouds on the host end to end through the CPU floor's kit, as the RAM
 *  variants pack them. */
template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
struct shapes_vram_nvidia_variant {
    using executor_t = cuda_executor_t;
    using build_t = shapes_vram_nvidia_build<scalar_type_, gpu_class_>;
    using face_t = typename build_t::face_t;
    using index_t = basic::vram_nvidia_indexes<face_t, util::uuid128_t, build_t>;
    using refine_plan_t = typename index_t::refine_plan_t;
    using scalar_t = scalar_type_;
    using pack_kit_t =
        typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(cpu_ladder_floor_k)>::kit_t;
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

    /** Reserves the entries and points @p capacity names, a pack taking a header and a partial
     *  alignment line per axis past its points. */
    status_t reserve(reserve_capacity_t capacity) noexcept {
        nk_size_t header = 0;
        if (status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(0, &header)); failed(sized))
            return sized;
        std::size_t const pack_overhead = header + 3 * nk_default_alignment_k;
        std::size_t const bytes = capacity.entries * pack_overhead +
                                  capacity.points * 3 * sizeof(typename scalar_type_::raw_t);
        return vram_nvidia_reserve(index, capacity.entries, geometry.connectivity, bytes);
    }

    expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t shapes, insert_config_t config,
                                       cuda_executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Packs the queries into @p pages, the call arena's host pages, where the matches land too. A
     *  device walk takes no host predicate. */
    expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                       growing_arena<forkunion_allocator_t>& pages, cuda_executor_t& executor,
                                       progress_ticks_t& ticks) noexcept;

    /** Writes the clouds as the RAM variants write them: a lone live shard's monolith, or one per
     *  live shard beside the manifest the path names. */
    status_t save(std::string_view path, cuda_executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Reads a checkpoint any residency wrote into the placed shards, one part per shard. */
    template <typename checkpoint_type_>
    status_t load(checkpoint_type_ checkpoint, cuda_executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Empties the index and its point count, retiring every shard behind a fresh shard generation
     *  on the same devices. */
    status_t clear() noexcept {
        status_t const cleared = index.clear();
        if (succeeded(cleared)) fu::standard_atomic_ref<std::size_t>(total_points).store(0, std::memory_order_relaxed);
        return cleared;
    }

    /** Takes over the points of an emptied @p source, whose nodes this index now holds. */
    void count_packed_source(shapes_vram_nvidia_variant& source) noexcept {
        std::size_t const points =
            fu::standard_atomic_ref<std::size_t>(source.total_points).exchange(0, std::memory_order_relaxed);
        fu::standard_atomic_ref<std::size_t>(total_points).store_add(points, std::memory_order_relaxed);
    }

    /** Sampling answers @c unsupported_k until the shards sample on their devices. */
    expected<sample_result_t> sample(basic_sample_inputs_t, basic_sample_outputs_t, sample_config_t,
                                     executor_t&) noexcept {
        return {{}, status_code_t::unsupported_k};
    }

  private:

    /** The wire format every residency's checkpoints share: each cloud's points, unpacked. */
    using codec_t = us::shapes::codec<pack_kit_t>;

    /** The clouds of @p tape packed end to end on @p pages, cast to the stored scalar where the
     *  caller's differs; refuses a count other than a frozen one. */
    expected<tape_view<byte_t const>> pack_(counted_tape_t const& tape,
                                            growing_arena<forkunion_allocator_t>& pages) const noexcept;
};

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
expected<tape_view<byte_t const>> shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::pack_(
    counted_tape_t const& tape, growing_arena<forkunion_allocator_t>& pages) const noexcept {
    using raw_t = typename scalar_type_::raw_t;
    std::size_t const count = tape.count();
    nk_dtype_t const from = nk_dtype_of(tape.scalar);
    std::size_t const from_bytes = nk_dtype_bits(from) / 8;
    std::size_t* const offsets = carve<std::size_t>(pages, count + 1);
    if (!offsets) return {{}, status_code_t::out_of_memory_k};
    offsets[0] = 0;
    std::size_t largest = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        std::size_t const cloud = tape.offsets[offset + 1] - tape.offsets[offset];
        if (geometry.points && cloud != geometry.points) return {{}, status_code_t::shape_mismatch_k};
        largest = max_of(largest, cloud);
        nk_size_t bytes = 0;
        if (status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(cloud, &bytes)); failed(sized))
            return {{}, sized};
        offsets[offset + 1] = offsets[offset] + bytes;
    }
    byte_t* const packed = carve<byte_t>(pages, offsets[count]);
    raw_t* const converted = from == dtype_k ? nullptr : carve<raw_t>(pages, largest * 3);
    if (!packed || (from != dtype_k && !converted)) return {{}, status_code_t::out_of_memory_k};
    for (std::size_t offset = 0; offset != count; ++offset) {
        std::size_t const cloud = tape.offsets[offset + 1] - tape.offsets[offset];
        void const* const coordinates = static_cast<byte_t const*>(tape.elements) +
                                        tape.offsets[offset] * 3 * from_bytes;
        raw_t const* cloud_points = static_cast<raw_t const*>(coordinates);
        if (from != dtype_k) {
            if (status_t const cast = us::shapes::status_of(
                    nk_cast_serial(coordinates, from, converted, dtype_k, cloud * 3, nullptr));
                failed(cast))
                return {{}, cast};
            cloud_points = converted;
        }
        if (status_t const made = us::shapes::status_of(
                pack_kit_t::pack(cloud_points, cloud, packed + offsets[offset], nullptr));
            failed(made))
            return {{}, made};
    }
    return {{packed, {offsets, count + 1}}, status_code_t::success_k};
}

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
expected<traversal_stats_t> shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::insert(
    usearch_keys_view_t keys, caller_tape_t shapes, insert_config_t config, cuda_executor_t& executor,
    progress_ticks_t& ticks) noexcept {
    // The batch's own pages, since batches routed to disjoint shards may land at once.
    growing_arena<forkunion_allocator_t> staging {forkunion_allocator_t {no_memory_domain_k}};
    auto const [entries, lower_status] = counted_tape_of(shapes.tape, shapes.count, staging);
    if (failed(lower_status)) return {{}, lower_status};
    auto const [packs, pack_status] = pack_(entries, staging);
    if (failed(pack_status)) return {{}, pack_status};
    expected<traversal_stats_t> const inserted = vram_nvidia_insert(index, keys, packs, geometry.connectivity,
                                                                    config.expansion_factor, ticks, executor);
    if (failed(inserted.status)) return inserted;
    fu::standard_atomic_ref<std::size_t>(total_points).store_add(elements_in(shapes), std::memory_order_relaxed);
    return inserted;
}

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
status_t shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::save(std::string_view path, cuda_executor_t& executor,
                                                                    progress_ticks_t& ticks) noexcept {
    return with_shapes_provenance(geometry, [&](save_config_t const& config) noexcept {
        return vram_nvidia_save(index, path, codec_t {}, config, token_of(geometry.metric), executor, ticks);
    });
}

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
template <typename checkpoint_type_>
status_t shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::load(checkpoint_type_ checkpoint,
                                                                    cuda_executor_t& executor,
                                                                    progress_ticks_t& ticks) noexcept {
    if (status_t const loaded = index.load(checkpoint, codec_t {}, executor, load_config_t {},
                                           static_cast<connectivity_t>(geometry.connectivity), ticks);
        failed(loaded))
        return loaded;
    // The value columns hold three scalars per point, summed over every part.
    auto const [metadata, status] = safegraphs_read_index_metadata(checkpoint);
    if (failed(status)) return status;
    total_points += metadata.value_bytes / (3 * sizeof(typename scalar_type_::raw_t));
    return status_code_t::success_k;
}

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
expected<traversal_stats_t> shapes_vram_nvidia_variant<scalar_type_, gpu_class_>::search(
    counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
    growing_arena<forkunion_allocator_t>& pages, cuda_executor_t& executor, progress_ticks_t& ticks) noexcept {
    auto const [packs, pack_status] = pack_(queries, pages);
    if (failed(pack_status)) return {{}, pack_status};
    return vram_nvidia_search(index, packs, outputs.wanted, config.expansion_factor, outputs, ticks, executor);
}

#endif
#if USEARCH_ARCH_METAL_

/** Builds one shard of the shapes face over packs of @p scalar_type_ on an Apple device, scoring
 *  under the kernel the variant's frozen geometry names. */
template <nk::a_shapes_dtype scalar_type_>
struct shapes_vram_apple_build {
    using face_t = us::shapes::vram_apple_index<scalar_type_, util::u32_t>;

    /** The kernel every shard scores under. */
    nk_kernel_kind_t kernel = nk_kernel_unknown_k;

    std::unique_ptr<face_t> operator()(int device) const noexcept {
        util::metal_context_t* const context = gpu_drivers().metal_contexts.context(device);
        if (!context) return {};
        return std::unique_ptr<face_t> {new (std::nothrow) face_t(
            kernel, util::metal_shared_allocator_t {util::metal_stream_t {*context, device}})};
    }
};

/** The shapes alternative on Apple GPUs of @p gpu_class_ or newer: the basic Apple shards over
 *  packs of @p scalar_type_, scoring Chamfer or Hausdorff as the geometry names, with 32-bit node
 *  IDs, as @c shapes_vram_nvidia_variant. Only f16 packs score here, so only they have one. */
template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
struct shapes_vram_apple_variant {
    using executor_t = util::metal_executor_t;
    using build_t = shapes_vram_apple_build<scalar_type_>;
    using face_t = typename build_t::face_t;
    using index_t = basic::vram_apple_indexes<face_t, util::uuid128_t, build_t>;
    using refine_plan_t = typename index_t::refine_plan_t;
    using scalar_t = scalar_type_;
    using pack_kit_t =
        typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(cpu_ladder_floor_k)>::kit_t;
    static constexpr nk_dtype_t dtype_k = scalar_type_::dtype();
    static constexpr gpu_class_t gpu_class_k = gpu_class_;

    index_t index;

    /** Holds no shard until the handle places them, each scoring under the kernel @p frozen
     *  names. */
    shapes_vram_apple_variant(int, shapes_geometry_t const& frozen) noexcept
        : index(util::forkunion_allocator_t {util::no_memory_domain_k},
                build_t {frozen.metric == usearch_shapes_metric_hausdorff_k ? nk_kernel_hausdorffs_packed_k
                                                                            : nk_kernel_chamfers_packed_k}) {}

    shapes_geometry_t geometry;

    /** Points across every stored entry, added to by insertions through @c atomic_ref. */
    std::size_t total_points = 0;

    /** Reserves the entries and points @p capacity names, a pack taking a header and a partial
     *  alignment line per axis past its points. */
    util::status_t reserve(reserve_capacity_t capacity) noexcept {
        nk_size_t header = 0;
        if (util::status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(0, &header)); failed(sized))
            return sized;
        std::size_t const pack_overhead = header + 3 * nk_default_alignment_k;
        std::size_t const bytes = capacity.entries * pack_overhead +
                                  capacity.points * 3 * sizeof(typename scalar_type_::raw_t);
        return vram_apple_reserve(index, capacity.entries, geometry.connectivity, bytes);
    }

    util::expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t shapes, insert_config_t config,
                                             util::metal_executor_t& executor, progress_ticks_t& ticks) noexcept {
        // The batch's own pages, since batches routed to disjoint shards may land at once.
        util::growing_arena<util::forkunion_allocator_t> staging {
            util::forkunion_allocator_t {util::no_memory_domain_k}};
        auto const [entries, lower_status] = counted_tape_of(shapes.tape, shapes.count, staging);
        if (failed(lower_status)) return {{}, lower_status};
        auto const [packs, pack_status] = pack_(entries, staging);
        if (failed(pack_status)) return {{}, pack_status};
        auto const [inserted, insert_status] = vram_apple_insert(index, keys, packs, geometry.connectivity,
                                                                 config.expansion_factor, ticks, executor);
        if (failed(insert_status)) return {inserted, insert_status};
        fu::standard_atomic_ref<std::size_t>(total_points).store_add(elements_in(shapes), std::memory_order_relaxed);
        return {inserted, insert_status};
    }

    /** Packs the queries into @p pages, the call arena's host pages, where the matches land too. A
     *  device walk takes no host predicate. */
    util::expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                             util::growing_arena<util::forkunion_allocator_t>& pages,
                                             util::metal_executor_t& executor, progress_ticks_t& ticks) noexcept {
        auto const [packs, pack_status] = pack_(queries, pages);
        if (failed(pack_status)) return {{}, pack_status};
        return vram_apple_search(index, packs, outputs.wanted, config.expansion_factor, outputs, ticks, executor);
    }

    /** Writes the clouds as the RAM variants write them. */
    util::status_t save(std::string_view path, util::metal_executor_t& executor, progress_ticks_t& ticks) noexcept {
        return with_shapes_provenance(geometry, [&](util::save_config_t const& config) noexcept {
            return vram_apple_save(index, path, codec_t {}, config, token_of(geometry.metric), executor, ticks);
        });
    }

    /** Reads a checkpoint any residency wrote into the placed shards, one part per shard. */
    template <typename checkpoint_type_>
    util::status_t load(checkpoint_type_ checkpoint, util::metal_executor_t& executor,
                        progress_ticks_t& ticks) noexcept {
        if (util::status_t const loaded = index.load(checkpoint, codec_t {}, executor, util::load_config_t {},
                                                     static_cast<util::connectivity_t>(geometry.connectivity), ticks);
            failed(loaded))
            return loaded;
        // The value columns hold three scalars per point, summed over every part.
        auto const [metadata, status] = util::safegraphs_read_index_metadata(checkpoint);
        if (failed(status)) return status;
        total_points += metadata.value_bytes / (3 * sizeof(typename scalar_type_::raw_t));
        return util::status_code_t::success_k;
    }

    /** Empties the index and its point count, retiring every shard behind a fresh shard generation
     *  on the same devices. */
    util::status_t clear() noexcept {
        util::status_t const cleared = index.clear();
        if (succeeded(cleared)) fu::standard_atomic_ref<std::size_t>(total_points).store(0, std::memory_order_relaxed);
        return cleared;
    }

    /** Takes over the points of an emptied @p source, whose nodes this index now holds. */
    void count_packed_source(shapes_vram_apple_variant& source) noexcept {
        std::size_t const points =
            fu::standard_atomic_ref<std::size_t>(source.total_points).exchange(0, std::memory_order_relaxed);
        fu::standard_atomic_ref<std::size_t>(total_points).store_add(points, std::memory_order_relaxed);
    }

    /** Sampling answers @c unsupported_k until the shards sample on their devices. */
    util::expected<util::sample_result_t> sample(basic_sample_inputs_t, basic_sample_outputs_t, util::sample_config_t,
                                                 executor_t&) noexcept {
        return {{}, util::status_code_t::unsupported_k};
    }

  private:

    /** The wire format every residency's checkpoints share: each cloud's points, unpacked. */
    using codec_t = us::shapes::codec<pack_kit_t>;

    /** The clouds of @p tape packed end to end on @p pages, cast to the stored scalar where the
     *  caller's differs; refuses a count other than a frozen one. */
    util::expected<util::tape_view<util::byte_t const>> pack_(
        counted_tape_t const& tape, util::growing_arena<util::forkunion_allocator_t>& pages) const noexcept {
        using raw_t = typename scalar_type_::raw_t;
        std::size_t const count = tape.count();
        nk_dtype_t const from = nk_dtype_of(tape.scalar);
        std::size_t const from_bytes = nk_dtype_bits(from) / 8;
        std::size_t* const offsets = carve<std::size_t>(pages, count + 1);
        if (!offsets) return {{}, util::status_code_t::out_of_memory_k};
        offsets[0] = 0;
        std::size_t largest = 0;
        for (std::size_t offset = 0; offset != count; ++offset) {
            std::size_t const cloud = tape.offsets[offset + 1] - tape.offsets[offset];
            if (geometry.points && cloud != geometry.points) return {{}, util::status_code_t::shape_mismatch_k};
            largest = util::max_of(largest, cloud);
            nk_size_t bytes = 0;
            if (util::status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(cloud, &bytes)); failed(sized))
                return {{}, sized};
            offsets[offset + 1] = offsets[offset] + bytes;
        }
        util::byte_t* const packed = carve<util::byte_t>(pages, offsets[count]);
        raw_t* const converted = from == dtype_k ? nullptr : carve<raw_t>(pages, largest * 3);
        if (!packed || (from != dtype_k && !converted)) return {{}, util::status_code_t::out_of_memory_k};
        for (std::size_t offset = 0; offset != count; ++offset) {
            std::size_t const cloud = tape.offsets[offset + 1] - tape.offsets[offset];
            void const* const coordinates = static_cast<util::byte_t const*>(tape.elements) +
                                            tape.offsets[offset] * 3 * from_bytes;
            raw_t const* cloud_points = static_cast<raw_t const*>(coordinates);
            if (from != dtype_k) {
                if (util::status_t const cast = us::shapes::status_of(
                        nk_cast_serial(coordinates, from, converted, dtype_k, cloud * 3, nullptr));
                    failed(cast))
                    return {{}, cast};
                cloud_points = converted;
            }
            if (util::status_t const made = us::shapes::status_of(
                    pack_kit_t::pack(cloud_points, cloud, packed + offsets[offset], nullptr));
                failed(made))
                return {{}, made};
        }
        return {{packed, {offsets, count + 1}}, util::status_code_t::success_k};
    }
};

#endif

#if USEARCH_ARCH_ROCM_

/** One class's shapes kits over one dtype as a code object, defined in its class unit,
 *  @c src/shapes/vram_variants_amd.hip, beside the bytes it embeds. */
template <gpu_class_t gpu_class_, nk_dtype_t dtype_>
struct shapes_amd_code_object {
    inline static constexpr nk_capability_t capability_k = numkong_capabilities_of(gpu_class_);
    static bytes_view_t bytes() noexcept;
};

/* The faces' members reading the kits, defined in the class unit this one compiles as. */
extern template struct us::shapes::vram_amd_index<
    typename us::shapes::shapes_ladder<nk::f16_t, numkong_capabilities_of(hip_class_k)>::kit_t, u32_t,
    shapes_amd_code_object<hip_class_k, nk_f16_k>>;
extern template struct us::shapes::vram_amd_index<
    typename us::shapes::shapes_ladder<nk::f32_t, numkong_capabilities_of(hip_class_k)>::kit_t, u32_t,
    shapes_amd_code_object<hip_class_k, nk_f32_k>>;

/** Builds one shard of the shapes face over packs of @p scalar_type_ for @p gpu_class_ on a HIP
 *  device, scoring under the kernel the variant's frozen geometry names. */
template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
struct shapes_vram_amd_build {
    using face_t = us::shapes::vram_amd_index<
        typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(gpu_class_)>::kit_t, u32_t,
        shapes_amd_code_object<gpu_class_, scalar_type_::dtype()>>;

    /** The kernel every shard scores under. */
    nk_kernel_kind_t kernel = nk_kernel_unknown_k;

    std::unique_ptr<face_t> operator()(int device) const noexcept {
        return std::unique_ptr<face_t> {
            new (std::nothrow) face_t(kernel, hip_virtual_memory_allocator<byte_t> {gpu_drivers().hip, device})};
    }
};

/** The shapes alternative on AMD GPUs of @p gpu_class_ : the basic VRAM shards over packs of
 *  @p scalar_type_, scoring Chamfer or Hausdorff as the geometry names, with 32-bit node IDs. A
 *  batch packs its clouds on the host end to end through the CPU floor's kit, the same way every
 *  RAM variant packs them. */
template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
struct shapes_vram_amd_variant {
    using executor_t = hip_executor_t;
    using build_t = shapes_vram_amd_build<scalar_type_, gpu_class_>;
    using face_t = typename build_t::face_t;
    using index_t = basic::vram_amd_indexes<face_t, util::uuid128_t, build_t>;
    using refine_plan_t = typename index_t::refine_plan_t;
    using scalar_t = scalar_type_;
    using pack_kit_t =
        typename us::shapes::shapes_ladder<scalar_type_, numkong_capabilities_of(cpu_ladder_floor_k)>::kit_t;
    static constexpr nk_dtype_t dtype_k = scalar_type_::dtype();
    static constexpr gpu_class_t gpu_class_k = gpu_class_;

    index_t index;

    /** Holds no shard until the handle places them, each scoring under the kernel @p frozen
     *  names. */
    shapes_vram_amd_variant(int, shapes_geometry_t const& frozen) noexcept
        : index(forkunion_allocator_t {no_memory_domain_k},
                build_t {frozen.metric == usearch_shapes_metric_hausdorff_k ? nk_kernel_hausdorffs_packed_k
                                                                            : nk_kernel_chamfers_packed_k}) {}

    shapes_geometry_t geometry;

    /** Points across every stored entry, added to by insertions through @c atomic_ref. */
    std::size_t total_points = 0;

    /** Reserves the entries and points @p capacity names, a pack taking a header and a partial
     *  alignment line per axis past its points. */
    status_t reserve(reserve_capacity_t capacity) noexcept {
        nk_size_t header = 0;
        if (status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(0, &header)); failed(sized))
            return sized;
        std::size_t const pack_overhead = header + 3 * nk_default_alignment_k;
        std::size_t const bytes = capacity.entries * pack_overhead +
                                  capacity.points * 3 * sizeof(typename scalar_type_::raw_t);
        return vram_amd_reserve(index, capacity.entries, geometry.connectivity, bytes);
    }

    expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t shapes, insert_config_t config,
                                       hip_executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Packs the queries into @p pages, the call arena's host pages, where the matches land too. A
     *  device walk takes no host predicate. */
    expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                       growing_arena<forkunion_allocator_t>& pages, hip_executor_t& executor,
                                       progress_ticks_t& ticks) noexcept;

    /** Writes the clouds as the RAM variants write them: a lone live shard's monolith, or one per
     *  live shard beside the manifest the path names. */
    status_t save(std::string_view path, hip_executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Reads a checkpoint any residency wrote into the placed shards, one part per shard. */
    template <typename checkpoint_type_>
    status_t load(checkpoint_type_ checkpoint, hip_executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Empties the index and its point count, retiring every shard behind a fresh shard generation
     *  on the same devices. */
    status_t clear() noexcept {
        status_t const cleared = index.clear();
        if (succeeded(cleared)) fu::standard_atomic_ref<std::size_t>(total_points).store(0, std::memory_order_relaxed);
        return cleared;
    }

    /** Takes over the points of an emptied @p source, whose nodes this index now holds. */
    void count_packed_source(shapes_vram_amd_variant& source) noexcept {
        std::size_t const points =
            fu::standard_atomic_ref<std::size_t>(source.total_points).exchange(0, std::memory_order_relaxed);
        fu::standard_atomic_ref<std::size_t>(total_points).store_add(points, std::memory_order_relaxed);
    }

    /** Sampling answers @c unsupported_k until the shards sample on their devices. */
    expected<sample_result_t> sample(basic_sample_inputs_t, basic_sample_outputs_t, sample_config_t,
                                     executor_t&) noexcept {
        return {{}, status_code_t::unsupported_k};
    }

  private:

    /** The wire format every residency's checkpoints share: each cloud's points, unpacked. */
    using codec_t = us::shapes::codec<pack_kit_t>;

    /** The clouds of @p tape packed end to end on @p pages, cast to the stored scalar where the
     *  caller's differs; refuses a count other than a frozen one. */
    expected<tape_view<byte_t const>> pack_(counted_tape_t const& tape,
                                            growing_arena<forkunion_allocator_t>& pages) const noexcept;
};

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
expected<tape_view<byte_t const>> shapes_vram_amd_variant<scalar_type_, gpu_class_>::pack_(
    counted_tape_t const& tape, growing_arena<forkunion_allocator_t>& pages) const noexcept {
    using raw_t = typename scalar_type_::raw_t;
    std::size_t const count = tape.count();
    nk_dtype_t const from = nk_dtype_of(tape.scalar);
    std::size_t const from_bytes = nk_dtype_bits(from) / 8;
    std::size_t* const offsets = carve<std::size_t>(pages, count + 1);
    if (!offsets) return {{}, status_code_t::out_of_memory_k};
    offsets[0] = 0;
    std::size_t largest = 0;
    for (std::size_t offset = 0; offset != count; ++offset) {
        std::size_t const cloud = tape.offsets[offset + 1] - tape.offsets[offset];
        if (geometry.points && cloud != geometry.points) return {{}, status_code_t::shape_mismatch_k};
        largest = max_of(largest, cloud);
        nk_size_t bytes = 0;
        if (status_t const sized = us::shapes::status_of(pack_kit_t::pack_size(cloud, &bytes)); failed(sized))
            return {{}, sized};
        offsets[offset + 1] = offsets[offset] + bytes;
    }
    byte_t* const packed = carve<byte_t>(pages, offsets[count]);
    raw_t* const converted = from == dtype_k ? nullptr : carve<raw_t>(pages, largest * 3);
    if (!packed || (from != dtype_k && !converted)) return {{}, status_code_t::out_of_memory_k};
    for (std::size_t offset = 0; offset != count; ++offset) {
        std::size_t const cloud = tape.offsets[offset + 1] - tape.offsets[offset];
        void const* const coordinates = static_cast<byte_t const*>(tape.elements) +
                                        tape.offsets[offset] * 3 * from_bytes;
        raw_t const* cloud_points = static_cast<raw_t const*>(coordinates);
        if (from != dtype_k) {
            if (status_t const cast = us::shapes::status_of(
                    nk_cast_serial(coordinates, from, converted, dtype_k, cloud * 3, nullptr));
                failed(cast))
                return {{}, cast};
            cloud_points = converted;
        }
        if (status_t const made = us::shapes::status_of(
                pack_kit_t::pack(cloud_points, cloud, packed + offsets[offset], nullptr));
            failed(made))
            return {{}, made};
    }
    return {{packed, {offsets, count + 1}}, status_code_t::success_k};
}

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
expected<traversal_stats_t> shapes_vram_amd_variant<scalar_type_, gpu_class_>::insert(
    usearch_keys_view_t keys, caller_tape_t shapes, insert_config_t config, hip_executor_t& executor,
    progress_ticks_t& ticks) noexcept {
    // The batch's own pages, since batches routed to disjoint shards may land at once.
    growing_arena<forkunion_allocator_t> staging {forkunion_allocator_t {no_memory_domain_k}};
    auto const [entries, lower_status] = counted_tape_of(shapes.tape, shapes.count, staging);
    if (failed(lower_status)) return {{}, lower_status};
    auto const [packs, pack_status] = pack_(entries, staging);
    if (failed(pack_status)) return {{}, pack_status};
    expected<traversal_stats_t> const inserted = vram_amd_insert(index, keys, packs, geometry.connectivity,
                                                                 config.expansion_factor, ticks, executor);
    if (failed(inserted.status)) return inserted;
    fu::standard_atomic_ref<std::size_t>(total_points).store_add(elements_in(shapes), std::memory_order_relaxed);
    return inserted;
}

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
status_t shapes_vram_amd_variant<scalar_type_, gpu_class_>::save(std::string_view path, hip_executor_t& executor,
                                                                 progress_ticks_t& ticks) noexcept {
    return with_shapes_provenance(geometry, [&](save_config_t const& config) noexcept {
        return vram_amd_save(index, path, codec_t {}, config, token_of(geometry.metric), executor, ticks);
    });
}

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
template <typename checkpoint_type_>
status_t shapes_vram_amd_variant<scalar_type_, gpu_class_>::load(checkpoint_type_ checkpoint, hip_executor_t& executor,
                                                                 progress_ticks_t& ticks) noexcept {
    if (status_t const loaded = index.load(checkpoint, codec_t {}, executor, load_config_t {},
                                           static_cast<connectivity_t>(geometry.connectivity), ticks);
        failed(loaded))
        return loaded;
    // The value columns hold three scalars per point, summed over every part.
    auto const [metadata, status] = safegraphs_read_index_metadata(checkpoint);
    if (failed(status)) return status;
    total_points += metadata.value_bytes / (3 * sizeof(typename scalar_type_::raw_t));
    return status_code_t::success_k;
}

template <nk::a_shapes_dtype scalar_type_, gpu_class_t gpu_class_>
expected<traversal_stats_t> shapes_vram_amd_variant<scalar_type_, gpu_class_>::search(
    counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
    growing_arena<forkunion_allocator_t>& pages, hip_executor_t& executor, progress_ticks_t& ticks) noexcept {
    auto const [packs, pack_status] = pack_(queries, pages);
    if (failed(pack_status)) return {{}, pack_status};
    return vram_amd_search(index, packs, outputs.wanted, config.expansion_factor, outputs, ticks, executor);
}

#endif

#pragma endregion Shapes Variants

#pragma region Shapes Bodies

/** The shapes variants each class unit compiles: one per CPU class and stored scalar on the host
 *  and on storage, and one per stored scalar a GPU class's kits serve. */
template <auto class_>
struct shapes_variants_of<vendored_residency_t::ram_k, class_> {
    using type = type_list<shapes_ram_variant<nk::f32_t, class_>, shapes_ram_variant<nk::f16_t, class_>>;
};
template <auto class_>
struct shapes_variants_of<vendored_residency_t::nvme_k, class_> {
    using type = type_list<shapes_nvme_variant<nk::f32_t, class_>, shapes_nvme_variant<nk::f16_t, class_>>;
};
#if USEARCH_ARCH_CUDA_ && USEARCH_TARGET_MONOVERSIONED
template <auto class_>
struct shapes_variants_of<vendored_residency_t::vram_nvidia_k, class_> {
    using type =
        type_list<shapes_vram_nvidia_variant<nk::f32_t, class_>, shapes_vram_nvidia_variant<nk::f16_t, class_>>;
};
#elif USEARCH_ARCH_CUDA_
template <>
struct shapes_variants_of<vendored_residency_t::vram_nvidia_k, gpu_class_t::nvidia_blackwell_k> {
    using type = type_list<shapes_vram_nvidia_variant<nk::f16_t, gpu_class_t::nvidia_blackwell_k>>;
};
template <>
struct shapes_variants_of<vendored_residency_t::vram_nvidia_k, gpu_class_t::nvidia_ampere_k> {
    using type = type_list<shapes_vram_nvidia_variant<nk::f32_t, gpu_class_t::nvidia_ampere_k>,
                           shapes_vram_nvidia_variant<nk::f16_t, gpu_class_t::nvidia_ampere_k>>;
};
#endif
#if USEARCH_ARCH_METAL_
template <>
struct shapes_variants_of<vendored_residency_t::vram_apple_k, gpu_class_t::apple_metal7_k> {
    using type = type_list<shapes_vram_apple_variant<nk::f16_t, gpu_class_t::apple_metal7_k>>;
};
#endif
#if USEARCH_ARCH_ROCM_
template <auto class_>
struct shapes_variants_of<vendored_residency_t::vram_amd_k, class_> {
    using type = type_list<shapes_vram_amd_variant<nk::f32_t, class_>, shapes_vram_amd_variant<nk::f16_t, class_>>;
};
#endif

/** What the shapes verbs do to one variant: describe it, and export and tally its clouds. */
template <>
struct variant_bodies<shapes_traits_t> {
    using geometry_t = shapes_geometry_t;

    /** What a view keeps of a cloud it read: its points as saved, in its thread's arena. */
    using kept_t = util::bytes_view_t;

    /** Whether @p source enters @p variant: the same kind, scalar, point count and connectivity. */
    template <typename variant_type_, typename source_type_>
    static bool matches(variant_type_ const& variant, source_type_ const& source) noexcept {
        geometry_t const &index = variant.geometry, &other = source.geometry;
        return index.metric == other.metric && index.quantization == other.quantization &&
               index.points == other.points && index.connectivity == other.connectivity;
    }

    /** Copies @p geometry into @p variant, a zero connectivity taking the default. */
    template <typename variant_type_>
    static util::status_t adopt(variant_type_& variant, geometry_t& geometry, std::monostate) noexcept {
        if (!geometry.connectivity) geometry.connectivity = util::connectivity_32_k;
        variant.geometry = geometry;
        return util::status_code_t::success_k;
    }

    template <typename variant_type_>
    static void describe(variant_type_ const& variant, usearch_shapes_index_description_t& description) noexcept {
        description.metadata = shapes_traits_t::metadata_of(variant.geometry);
        description.total_points =
            fu::standard_atomic_ref<std::size_t const>(variant.total_points).load(std::memory_order_relaxed);
    }

    template <typename cloud_type_>
    static util::status_t keep(nvme_executor_t& executor, std::size_t thread, cloud_type_ const& cloud,
                               kept_t& kept) noexcept {
        return keep_entry(executor, thread, util::bytes_of(cloud.data(), cloud.extent(0) * 3), kept);
    }

    /** Writes @p count clouds, the @c i th as @p value_at hands it, as points in the scalar the
     *  call asks for in @p memory, their offsets counting points; a variant's own values are packs,
     *  which unpack through @p memory too where the scalar is not the stored one. */
    template <typename variant_type_, typename value_at_type_>
    static util::status_t write_values(variant_type_ const& variant, std::size_t count, value_at_type_&& value_at,
                                       util::growing_arena<util::forkunion_allocator_t>& memory,
                                       shapes_get_outputs_t outputs) noexcept {
        nk_dtype_t const exported = nk_dtype_of(outputs.scalar);
        std::size_t const element_bytes = nk_dtype_bits(exported) / 8;
        std::size_t* const offsets = carve<std::size_t>(memory, count + 1);
        if (!offsets) return util::status_code_t::out_of_memory_k;
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != count; ++entry)
            offsets[entry + 1] = offsets[entry] + variant.points_of(value_at(entry));
        util::byte_t* const points = carve<util::byte_t>(memory, offsets[count] * 3 * element_bytes);
        if (offsets[count] && !points) return util::status_code_t::out_of_memory_k;
        thread_allocator_t const scratch {memory};
        for (std::size_t entry = 0; entry != count; ++entry) {
            util::byte_t* const output = points + offsets[entry] * 3 * element_bytes;
            if (util::status_t const exported_status = variant.export_points(value_at(entry), exported, output,
                                                                             scratch);
                failed(exported_status))
                return exported_status;
        }
        write_output_tape(points, offsets, outputs.scalar, *outputs.shapes);
        return util::status_code_t::success_k;
    }

    /** Adds one entry's points to a key's statistics, from any thread: a variant's packs count
     *  their points, and a view's shapes hold a row per point. */
    template <typename cloud_type_>
    static void tally(usearch_shapes_key_stats_t& stats, cloud_type_ const& cloud) noexcept {
        std::size_t points = 0;
        if constexpr (requires { cloud.max_points(); }) points = cloud.max_points();
        else points = cloud.extent(0);
        fu::standard_atomic_ref<std::size_t>(stats.total_points).store_add(points, std::memory_order_relaxed);
        raise_to<fu::standard_atomic_ref>(stats.largest_points, points);
    }
};

#pragma endregion Shapes Bodies
