/**
 *  @file src/sequence_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief What one sequence variant is spelled with and compiled through: the CPU classes'
 *      StringZilla capabilities, the geometry every sequence variant freezes, and the variants
 *      over the basic indexes.
 *
 *  Every sequence class unit in @c src/sequence/ includes this to instantiate its variants, and
 *  @c sequence_index.cpp adds the variant, the traits and the exported C entry points on top of it.
 */
#pragma once

#include <array>       // `std::array` - the checkpoint's recorded fields
#include <span>        // `std::span` - the overlap kind's window widths
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity`

#include "usearch/sequence/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/sequence/vram_nvidia_index.hpp"
#endif

#include "basic_specializations.cuh"

#pragma region Sequence Variants

/** An overlap distance is fixed point over the shared @c overlap_resolution_k. */
template <typename kit_type_, typename allocator_type_, typename cache_hints_type_>
inline constexpr usearch_distance_t
    distance_unit<us::sequence::overlap_metric<kit_type_, allocator_type_, cache_hints_type_>> =
        1.0 / us::sequence::overlap_resolution_k;

/** Lowers one CPU class onto StringZilla's capability bits, which name backends rather than
 *  accumulate, so each class spells every backend its kits may pick, and a class adding none
 *  spells its parent's. */
constexpr sz_capability_t stringzilla_capabilities_of(cpu_class_t cpu_class) noexcept {
    switch (cpu_class) {
    case cpu_class_t::serial_k: return sz_cap_serial_k;
    case cpu_class_t::intel_haswell_k: return static_cast<sz_capability_t>(sz_cap_serial_k | sz_cap_haswell_k);
    case cpu_class_t::intel_skylake_k:
        return static_cast<sz_capability_t>(sz_cap_serial_k | sz_cap_haswell_k | sz_cap_skylake_k);
    case cpu_class_t::aws_g2_k: return static_cast<sz_capability_t>(sz_cap_serial_k | sz_cap_neon_k);
    default: return stringzilla_capabilities_of(parent_class_of(cpu_class));
    }
}

/** The checkpoint token of one sequence kind, spelled as the SafeGraphs registry spells it; empty
 *  for the kinds the engine refuses. */
inline std::string_view token_of(usearch_sequence_metric_t metric) noexcept {
    switch (metric) {
    case usearch_sequence_metric_levenshtein_bytes_k: return "levenshtein";
    case usearch_sequence_metric_levenshtein_utf8_k: return "levenshtein-utf8";
    case usearch_sequence_metric_overlap_k: return "overlap";
    default: return {};
    }
}

/** The sequence kind a checkpoint token names, or @c unknown_k. */
inline usearch_sequence_metric_t sequence_metric_of(std::string_view token) noexcept {
    for (usearch_sequence_metric_t metric :
         {usearch_sequence_metric_levenshtein_bytes_k, usearch_sequence_metric_levenshtein_utf8_k,
          usearch_sequence_metric_overlap_k})
        if (token == token_of(metric)) return metric;
    return usearch_sequence_metric_unknown_k;
}

/** The kind, window widths and connectivity a sequence handle freezes, as an init or a checkpoint
 *  states them; every variant holds one. */
struct sequence_geometry_t {
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;

    /** The window widths the overlap kind hashes, in bytes; empty for every other kind. */
    growing_buffer<std::size_t, forkunion_allocator_t> window_widths {forkunion_allocator_t {no_memory_domain_k}};

    /** Neighbors per node the insertions select; zero until frozen, which takes the default. */
    std::size_t connectivity = 0;
};

/** Calls @p callback with a factory of the metric @p metric names over a thread's arena, sized for
 *  a round of candidates, scoring with the kits @p capability_ picks and prefetching through
 *  @p cache_hints_type_; @p window_widths are the overlap kind's alone. */
template <sz_capability_t capability_, typename cache_hints_type_, typename callback_type_>
decltype(auto) with_sequence_metric(usearch_sequence_metric_t metric, std::span<std::size_t const> window_widths,
                                    callback_type_&& callback) noexcept {
    using bytes_t = us::sequence::levenshtein_metric<
        typename us::sequence::levenshtein_ladder<sz_levenshtein_bytes_k, capability_>::kit_t, thread_allocator_t,
        cache_hints_type_>;
    using runes_t = us::sequence::levenshtein_metric<
        typename us::sequence::levenshtein_ladder<sz_levenshtein_runes_k, capability_>::kit_t, thread_allocator_t,
        cache_hints_type_>;
    using overlap_t = us::sequence::overlap_metric<typename us::sequence::overlap_ladder<capability_>::kit_t,
                                                   thread_allocator_t, cache_hints_type_>;
    auto levenshtein = [&]<typename metric_type_>(std::type_identity<metric_type_>) noexcept {
        return callback([](thread_allocator_t const& allocator, std::size_t staging) noexcept {
            metric_type_ made {allocator};
            status_t const ready = made.init(staging);
            return expected<metric_type_> {std::move(made), ready};
        });
    };
    switch (metric) {
    case usearch_sequence_metric_levenshtein_utf8_k: return levenshtein(std::type_identity<runes_t> {});
    case usearch_sequence_metric_overlap_k:
        return callback([window_widths](thread_allocator_t const& allocator, std::size_t staging) noexcept {
            overlap_t made {allocator};
            status_t const ready = made.init(staging, window_widths);
            return expected<overlap_t> {std::move(made), ready};
        });
    default: return levenshtein(std::type_identity<bytes_t> {});
    }
}

/** One sequence alternative: the basic index over owned strings, the kind it scores with, and the
 *  kits @p cpu_class_ picks. Its verbs are defined below the class, so an explicit instantiation in
 *  the class's unit compiles them with that class's flags and an @c extern one keeps them out of
 *  the dispatch unit. */
template <cpu_class_t cpu_class_>
struct sequence_ram_variant {
    using executor_t = forkunion_executor_t;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    using index_t = us::sequence::ram_index<us::utilities::uuid_t, u32_t, waiter_of<cpu_class_>,
                                            atomic_ref_for_cpu_class<cpu_class_>::template type, forkunion_allocator_t,
                                            cache_hints_t>;
    using value_t = typename index_t::value_t;
    using refine_plan_t = typename index_t::refine_plan_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr sz_capability_t capability_k = stringzilla_capabilities_of(cpu_class_);

    index_t index;
    sequence_geometry_t geometry;

    explicit sequence_ram_variant(forkunion_allocator_t const& allocator) noexcept
        : index(allocator, machine_cache_line_bytes()) {}

    status_t reserve(usearch_sequence_reserve_t const& arguments) noexcept {
        return index.reserve(arguments.entries).status;
    }
    status_t clear() noexcept { return index.clear(); }
    expected<traversal_telemetry_t> insert(usearch_sequence_insert_t const& arguments, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;
    expected<traversal_telemetry_t> search(usearch_sequence_search_t const& arguments, matches_landing_t& landing,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** Opens a refinement once @p donors land, its walks as wide as @p expansion or twice the
     *  connectivity, as an insertion's. */
    expected<refine_plan_t> plan_refine(std::span<index_t*> donors, std::size_t expansion) noexcept {
        typename index_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_absorb(donors, planning, basic::keep_diverse_t {geometry.connectivity}, {expansion});
    }
    status_t refine(refine_plan_t& plan, progress_ticks_t& ticks, executor_t& executor) noexcept;
    status_t save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Serving a checkpoint in place and sampling answer @c unsupported_k until the index serves
     *  them. */
    status_t view(std::string_view) noexcept { return status_t::unsupported_k; }
    status_t sample(usearch_sequence_sample_t const&) noexcept { return status_t::unsupported_k; }
};

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> sequence_ram_variant<cpu_class_>::insert(usearch_sequence_insert_t const& arguments,
                                                                         progress_ticks_t& ticks,
                                                                         executor_t& executor) noexcept {
    std::size_t const connectivity = geometry.connectivity, expansion = arguments.expansion_factor;
    auto with_metric = [&](auto make_metric) noexcept {
        auto build = [&](std::size_t row, value_t& value, std::size_t) noexcept {
            return us::sequence::status_of(value.assign(chars_at(arguments.sequences, row)));
        };
        return insert_entries(index, arguments.keys, arguments.count, connectivity,
                              expansion ? expansion : connectivity * 2, make_metric, build, ticks, executor);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry.metric, to_span(geometry.window_widths),
                                                             with_metric);
}

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> sequence_ram_variant<cpu_class_>::search(usearch_sequence_search_t const& arguments,
                                                                         matches_landing_t& landing,
                                                                         progress_ticks_t& ticks,
                                                                         executor_t& executor) noexcept {
    auto with_metric = [&](auto make_metric) noexcept {
        auto query_of = [&](std::size_t row, std::size_t, auto&& answer) noexcept {
            return answer(chars_at(arguments.queries, row));
        };
        return search_entries(index, arguments.count, arguments.wanted, geometry.connectivity,
                              arguments.expansion_factor, arguments.traversal, arguments.predicate, make_metric,
                              query_of, landing, ticks, executor);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry.metric, to_span(geometry.window_widths),
                                                             with_metric);
}

template <cpu_class_t cpu_class_>
status_t sequence_ram_variant<cpu_class_>::refine(refine_plan_t& plan, progress_ticks_t& ticks,
                                                  executor_t& executor) noexcept {
    auto with_metric = [&](auto make_metric) noexcept {
        auto [scratch, prepare_status] = prepare_refinement<index_t>(plan, geometry.connectivity, make_metric,
                                                                     executor);
        if (failed(prepare_status)) return prepare_status;
        return drive_refinement(plan, scratch, basic::keep_diverse_t {geometry.connectivity}, ticks, executor);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry.metric, to_span(geometry.window_widths),
                                                             with_metric);
}

template <cpu_class_t cpu_class_>
status_t sequence_ram_variant<cpu_class_>::save(std::string_view path, executor_t& executor,
                                                progress_ticks_t& ticks) noexcept {
    auto const& widths = geometry.window_widths;
    std::array<char, 24> connectivity_text;
    growing_buffer<char, forkunion_allocator_t> widths_text {forkunion_allocator_t {}};
    if (failed(widths_text.reserve_exactly(widths.size() * 21))) return status_t::bad_alloc_k;
    for (std::size_t const width : widths) {
        std::array<char, 21> width_text;
        if (!widths_text.empty()) widths_text.push_back_reserved(',');
        for (char const digit : decimal_of(width, width_text)) widths_text.push_back_reserved(digit);
    }
    std::array<safegraphs_custom_field_t, 2> const fields {{
        {"connectivity", decimal_of(geometry.connectivity, connectivity_text)},
        {"window_widths", to_string_view(widths_text)},
    }};
    save_config_t config;
    config.provenance = std::span {fields}.first(widths.empty() ? 1 : 2);
    return safegraphs_write_index(path, index, us::sequence::codec_t {}, executor, config, token_of(geometry.metric),
                                  ticks);
}

template <cpu_class_t cpu_class_>
status_t sequence_ram_variant<cpu_class_>::load(std::string_view path, executor_t& executor,
                                                progress_ticks_t& ticks) noexcept {
    return safegraphs_read_index(path, index, us::sequence::codec_t {}, executor, load_config_t {}, ticks);
}

/** One sequence alternative on block storage: the basic NVMe index over the strings a RAM variant
 *  saved, read-only, scored with the RAM variant's metrics and the kits @p cpu_class_ picks. Its
 *  verbs are instantiated in the class's unit, as the RAM variant's are. */
template <cpu_class_t cpu_class_>
struct sequence_nvme_variant {
    using executor_t = nvme_executor_t;
    using index_t = basic::nvme_index<us::sequence::codec_t, us::utilities::uuid_t, u32_t, sz_size_t,
                                      default_io_engine_t, row_kit_of<cpu_class_>>;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr sz_capability_t capability_k = stringzilla_capabilities_of(cpu_class_);

    index_t index;
    sequence_geometry_t geometry;

    explicit sequence_nvme_variant(forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    /** Serves the checkpoint at @p path in place, its parts from @p paths when any are named. */
    status_t view(std::string_view path, std::span<std::string_view const> paths, executor_t& executor) noexcept {
        return index.view(path, paths, token_of(geometry.metric), executor);
    }

    expected<traversal_telemetry_t> search(usearch_sequence_search_t const& arguments, matches_landing_t& landing,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** A view already is its checkpoint, so saving answers @c unsupported_k, as does sampling until
     *  the view serves it. */
    status_t save(std::string_view, executor_t&, progress_ticks_t&) noexcept { return status_t::unsupported_k; }
    status_t sample(usearch_sequence_sample_t const&) noexcept { return status_t::unsupported_k; }
};

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> sequence_nvme_variant<cpu_class_>::search(usearch_sequence_search_t const& arguments,
                                                                          matches_landing_t& landing,
                                                                          progress_ticks_t& ticks,
                                                                          executor_t& executor) noexcept {
    auto with_metric = [&](auto make_metric) noexcept {
        auto query_of = [&](std::size_t row, thread_allocator_t const&) noexcept {
            return expected<std::string_view> {chars_at(arguments.queries, row), status_t::success_k};
        };
        return search_entries(index, arguments.count, arguments.wanted, arguments.expansion_factor, arguments.traversal,
                              arguments.predicate, make_metric, query_of, landing, ticks, executor);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry.metric, to_span(geometry.window_widths),
                                                             with_metric);
}

#if USEARCH_ARCH_CUDA_

/** One kind's sequence kits as a fatbin, defined in @c src/sequence/vram_variants_nvidia.cpp beside
 *  the bytes it embeds. */
template <usearch_sequence_metric_t metric_>
struct sequence_nvidia_fatbin {
    static bytes_view_t bytes() noexcept;
};

/** The capabilities the sequence kits compile under on NVIDIA: StringZilla's CUDA floor, which one
 *  cubin per class carries. */
inline constexpr sz_capability_t sequence_nvidia_capabilities_k = sz_cap_cuda_k;

/** The faces' members reading the kits, defined in @c src/sequence/vram_variants_nvidia.cpp. */
extern template struct us::sequence::vram_nvidia_index<
    usearch_sequence_metric_levenshtein_bytes_k, u32_t, sequence_nvidia_capabilities_k,
    sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_bytes_k>>;
extern template struct us::sequence::vram_nvidia_index<
    usearch_sequence_metric_levenshtein_utf8_k, u32_t, sequence_nvidia_capabilities_k,
    sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_utf8_k>>;
extern template struct us::sequence::vram_nvidia_index<usearch_sequence_metric_overlap_k, u32_t,
                                                       sequence_nvidia_capabilities_k,
                                                       sequence_nvidia_fatbin<usearch_sequence_metric_overlap_k>>;

/** A GPU overlap index scores in the same fixed point as the RAM metric. */
template <typename node_id_type_, sz_capability_t capability_, typename kernels_type_>
inline constexpr usearch_distance_t distance_unit<
    us::sequence::vram_nvidia_index<usearch_sequence_metric_overlap_k, node_id_type_, capability_, kernels_type_>> =
    1.0 / us::sequence::overlap_resolution_k;

/** The sequence alternative on any NVIDIA GPU: the basic VRAM index under the sequence kind
 *  @p metric_ - Levenshtein over bytes or UTF-8 runes, or overlap - with 32-bit node IDs. */
template <usearch_sequence_metric_t metric_>
struct sequence_vram_nvidia_variant
    : vram_nvidia_variant<us::sequence::vram_nvidia_index<metric_, u32_t, sequence_nvidia_capabilities_k,
                                                          sequence_nvidia_fatbin<metric_>>> {
    static constexpr usearch_sequence_metric_t metric_k = metric_;
    static constexpr gpu_class_t gpu_class_k = gpu_class_t::nvidia_ampere_k;

    /** Places the index on CUDA device @p device; its kind is the template's. */
    sequence_vram_nvidia_variant(int device, sequence_geometry_t const&) noexcept
        : sequence_vram_nvidia_variant::vram_nvidia_variant(device) {}

    sequence_geometry_t geometry;

    status_t reserve(usearch_sequence_reserve_t const& arguments) noexcept {
        return this->reserve_(arguments.entries, geometry.connectivity, arguments.payload_bytes);
    }

    expected<traversal_telemetry_t> insert(usearch_sequence_insert_t const& arguments, progress_ticks_t& ticks,
                                           cuda_executor_t& executor) noexcept {
        std::size_t const count = arguments.count;
        auto [plan, plan_status] = this->index.plan_insert(count);
        if (failed(plan_status)) return {{}, plan_status};
        return this->insert_(arguments.keys, static_cast<byte_t const*>(arguments.sequences.data),
                             {arguments.sequences.offsets, count + 1}, plan, geometry.connectivity,
                             arguments.expansion_factor, ticks, executor);
    }

    /** A device walk takes no host predicate. */
    expected<traversal_telemetry_t> search(usearch_sequence_search_t const& arguments, matches_landing_t& landing,
                                           progress_ticks_t& ticks, cuda_executor_t& executor) noexcept {
        return this->search_(static_cast<byte_t const*>(arguments.queries.data), arguments.queries.offsets,
                             arguments.count, arguments.wanted, arguments.expansion_factor, landing, ticks, executor);
    }
};

#endif

#pragma endregion Sequence Variants
