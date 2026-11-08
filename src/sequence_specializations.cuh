/**
 *  @file src/sequence_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief What one sequence variant is spelled with and compiled through: the CPU classes'
 *      StringZilla capabilities, the geometry every sequence variant adopts, and the variants
 *      over the basic indexes.
 *
 *  Every sequence class unit in @c src/sequence/ includes this to instantiate its variants, and
 *  @c sequence_index.cpp adds the variant, the traits and the exported C entry points on top of it.
 */
#pragma once

#include <cstring> // `std::memcmp` - a donor's costs

#include <array>       // `std::array` - the checkpoint's recorded fields
#include <memory>      // `std::unique_ptr` - a GPU shard built on its device
#include <new>         // `std::nothrow` - a GPU shard built on its device
#include <span>        // `std::span` - the overlap kind's window widths
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity`
#include <variant>     // `std::monostate` - no radius walk

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the scan totals

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

/** Calls @p callback with a factory of the metric @p geometry names over a thread's arena, sized
 *  for a round of candidates, scoring with the kits @p capability_ picks, the alignment kind with
 *  the folded @p costs, and prefetching through @p cache_hints_type_. */
template <sz_capability_t capability_, typename cache_hints_type_, typename callback_type_>
decltype(auto) with_sequence_metric(sequence_geometry_t const& geometry,
                                    us::sequence::needleman_wunsch_costs_t const& costs,
                                    callback_type_&& callback) noexcept {
    using bytes_t = us::sequence::levenshtein_metric<
        typename us::sequence::levenshtein_ladder<sz_levenshtein_bytes_k, capability_>::kit_t, thread_allocator_t,
        cache_hints_type_>;
    using runes_t = us::sequence::levenshtein_metric<
        typename us::sequence::levenshtein_ladder<sz_levenshtein_runes_k, capability_>::kit_t, thread_allocator_t,
        cache_hints_type_>;
    using overlap_t = us::sequence::overlap_metric<typename us::sequence::overlap_ladder<capability_>::kit_t,
                                                   thread_allocator_t, cache_hints_type_>;
    using needleman_wunsch_t =
        us::sequence::needleman_wunsch_metric<typename us::sequence::needleman_wunsch_ladder<capability_>::kit_t,
                                              thread_allocator_t, cache_hints_type_>;
    auto levenshtein = [&]<typename metric_type_>(std::type_identity<metric_type_>) noexcept {
        return callback([](thread_allocator_t const& allocator, std::size_t staging) noexcept {
            metric_type_ made {allocator};
            status_t const ready = made.init(staging);
            return expected<metric_type_> {std::move(made), ready};
        });
    };
    switch (geometry.metric) {
    case usearch_sequence_metric_levenshtein_utf8_k: return levenshtein(std::type_identity<runes_t> {});
    case usearch_sequence_metric_overlap_k:
        return callback([window_widths = to_span(geometry.window_widths)](thread_allocator_t const& allocator,
                                                                          std::size_t staging) noexcept {
            overlap_t made {allocator};
            status_t const ready = made.init(staging, window_widths);
            return expected<overlap_t> {std::move(made), ready};
        });
    case usearch_sequence_metric_needleman_wunsch_k:
        return callback([&costs](thread_allocator_t const& allocator, std::size_t staging) noexcept {
            needleman_wunsch_t made {allocator};
            status_t const ready = made.init(staging, costs);
            return expected<needleman_wunsch_t> {std::move(made), ready};
        });
    default: return levenshtein(std::type_identity<bytes_t> {});
    }
}

/** One sequence alternative: the basic shards over owned strings, the kind they score with, and the
 *  kits @p cpu_class_ picks. Its verbs are defined below the class, so an explicit instantiation in
 *  the class's unit compiles them with that class's flags and an @c extern one keeps them out of
 *  the dispatch unit. Checkpoints and refinements serve an index of one shard. */
template <cpu_class_t cpu_class_>
struct sequence_ram_variant {
    using executor_t = forkunion_executor_t;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    using shard_t = us::sequence::ram_index<us::utilities::uuid_t, u32_t, waiter_of<cpu_class_>,
                                            atomic_ref_for_cpu_class<cpu_class_>::template type, forkunion_allocator_t,
                                            cache_hints_t>;
    using index_t = basic::ram_indexes<shard_t>;
    using value_t = typename shard_t::value_t;
    using refine_plan_t = typename index_t::refine_plan_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr sz_capability_t capability_k = stringzilla_capabilities_of(cpu_class_);

    index_t index;
    sequence_geometry_t geometry;

    /** The table and gap cost the alignment kind scores with, folded at freeze. */
    us::sequence::needleman_wunsch_costs_t costs {};

    explicit sequence_ram_variant(forkunion_allocator_t const& allocator) noexcept
        : index(allocator, machine_cache_line_bytes()) {}

    status_t reserve(reserve_capacity_t capacity) noexcept { return index.reserve(capacity.entries).status; }
    status_t clear() noexcept { return index.clear(); }
    expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t sequences, insert_config_t config,
                                       executor_t& executor, progress_ticks_t& ticks) noexcept;
    expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                       predicate_t const& predicate, executor_t& executor,
                                       progress_ticks_t& ticks) noexcept;

    /** Opens a refinement that first moves every shard of each of @p donors into this index, then
     *  re-links across, its walks as wide as @p expansion or twice the connectivity. */
    expected<refine_plan_t> plan_absorb(std::span<index_t*> donors, refine_config_t config) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_absorb(donors, {config.expansion_factor, config.memory_limit},
                                 basic::keep_diverse_t {geometry.connectivity}, planning);
    }

    /** Opens a refinement of every live shard in place, its walks as wide as @p expansion. */
    expected<refine_plan_t> plan_refine(refine_config_t config) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_refine({config.expansion_factor, config.memory_limit},
                                 basic::keep_diverse_t {geometry.connectivity}, planning);
    }

    /** Opens a split of every live shard into the same shard of each of @p receivers, its part
     *  roots drawn from the config's seed, then re-links each receiver's picked nodes with walks as
     *  wide as @p expansion. */
    expected<refine_plan_t> plan_split(std::span<index_t*> receivers, split_config_t config) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_split(receivers, {{config.refine.expansion_factor, config.refine.memory_limit}, config.seed},
                                basic::keep_diverse_t {geometry.connectivity}, planning);
    }
    status_t refine(refine_plan_t& plan, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Writes the checkpoint under a path, or into a buffer of @ref storage_requirements bytes. */
    template <typename checkpoint_type_>
    status_t save(checkpoint_type_ checkpoint, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Reads the checkpoint at a path or in a buffer. */
    template <typename checkpoint_type_>
    status_t load(checkpoint_type_ checkpoint, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** The bytes a save into memory takes, under the provenance the save records. */
    expected<std::size_t> storage_requirements() const noexcept;

    /** Serving a checkpoint in place answers @c unsupported_k until the index serves it. */
    status_t view(std::string_view) noexcept { return status_t::unsupported_k; }
    expected<sample_result_t> sample(basic_sample_inputs_t inputs, basic_sample_outputs_t outputs,
                                     sample_config_t config, executor_t& executor) noexcept;
};

template <cpu_class_t cpu_class_>
expected<traversal_stats_t> sequence_ram_variant<cpu_class_>::insert(usearch_keys_view_t keys, caller_tape_t sequences,
                                                                     insert_config_t config, executor_t& executor,
                                                                     progress_ticks_t& ticks) noexcept {
    std::size_t const connectivity = geometry.connectivity, expansion = config.expansion_factor;
    auto with_metric = [&](auto make_metric) noexcept {
        auto build = [](counted_tape_t const& entries, std::size_t offset, value_t& value, std::size_t) noexcept {
            return us::sequence::status_of(value.assign(chars_at(entries, offset)));
        };
        return insert_entries(index, keys, sequences, connectivity, expansion ? expansion : connectivity * 2,
                              make_metric, build, ticks, executor);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry, costs, with_metric);
}

template <cpu_class_t cpu_class_>
expected<traversal_stats_t> sequence_ram_variant<cpu_class_>::search(counted_tape_t queries, matches_outputs_t& outputs,
                                                                     search_config_t config,
                                                                     predicate_t const& predicate, executor_t& executor,
                                                                     progress_ticks_t& ticks) noexcept {
    auto with_metric = [&](auto make_metric) noexcept {
        auto query_of = [&](std::size_t query_index, std::size_t, auto&& answer) noexcept {
            return answer(chars_at(queries, query_index));
        };
        return search_entries(index, queries.count(), geometry.connectivity, make_metric, query_of, outputs, config,
                              predicate, executor, ticks);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry, costs, with_metric);
}

template <cpu_class_t cpu_class_>
expected<sample_result_t> sequence_ram_variant<cpu_class_>::sample(basic_sample_inputs_t inputs,
                                                                   basic_sample_outputs_t outputs,
                                                                   sample_config_t config,
                                                                   executor_t& executor) noexcept {
    auto with_metric = [&](auto make_metric) noexcept {
        return sample_entries(index, geometry.connectivity, make_metric, inputs, outputs, config, executor);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry, costs, with_metric);
}

template <cpu_class_t cpu_class_>
status_t sequence_ram_variant<cpu_class_>::refine(refine_plan_t& plan, executor_t& executor,
                                                  progress_ticks_t& ticks) noexcept {
    auto with_metric = [&](auto make_metric) noexcept {
        auto [scratch, prepare_status] = prepare_refinement<shard_t>(plan, geometry.connectivity, make_metric,
                                                                     executor);
        if (failed(prepare_status)) return prepare_status;
        return drive_refinement(plan, scratch, basic::keep_diverse_t {geometry.connectivity}, ticks, executor);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry, costs, with_metric);
}

/** Hands @p write, as <tt>write(config) → status_t</tt>, the save configuration recording what
 *  @p geometry and @p costs froze: the connectivity, the window widths as decimals between
 *  commas, and the costs as one hexadecimal digit per entry, as every residency's checkpoint
 *  records them. */
template <typename write_type_>
status_t with_sequence_provenance(sequence_geometry_t const& geometry,
                                  us::sequence::needleman_wunsch_costs_t const& costs, write_type_&& write) noexcept {
    auto const& widths = geometry.window_widths;
    std::array<char, 24> connectivity_text;
    growing_buffer<char, forkunion_allocator_t> widths_text {forkunion_allocator_t {}};
    if (failed(widths_text.reserve_exactly(widths.size() * 21))) return status_t::bad_alloc_k;
    for (std::size_t const width : widths) {
        std::array<char, 21> width_text;
        if (!widths_text.empty()) widths_text.push_back_reserved(',');
        for (char const digit : decimal_of(width, width_text)) widths_text.push_back_reserved(digit);
    }
    // The folded costs as one hexadecimal digit per entry, row by row over the classes.
    std::array<char, 24> gap_text;
    std::array<char, us::sequence::needleman_wunsch_classes_k * us::sequence::needleman_wunsch_classes_k> costs_text;
    std::size_t const entries = std::size_t {costs.classes} * costs.classes;
    for (std::size_t entry = 0; entry != entries; ++entry)
        costs_text[entry] = "0123456789abcdef"[costs.substitutions[entry / costs.classes][entry % costs.classes]];
    std::array<safegraphs_custom_field_t, 3> fields {
        {{"connectivity", decimal_of(geometry.connectivity, connectivity_text)}}};
    std::size_t recorded = 1;
    if (!widths.empty()) fields[recorded++] = {"window_widths", to_string_view(widths_text)};
    if (costs.code_bits) {
        fields[recorded++] = {"substitution_costs", std::string_view {costs_text.data(), entries}};
        fields[recorded++] = {"gap_cost", decimal_of(costs.gap, gap_text)};
    }
    save_config_t config;
    config.provenance = std::span {fields}.first(recorded);
    return write(config);
}

template <cpu_class_t cpu_class_>
template <typename checkpoint_type_>
status_t sequence_ram_variant<cpu_class_>::save(checkpoint_type_ checkpoint, executor_t& executor,
                                                progress_ticks_t& ticks) noexcept {
    return with_sequence_provenance(geometry, costs, [&](save_config_t const& config) noexcept {
        return index.save(checkpoint, us::sequence::codec_t {}, executor, config, token_of(geometry.metric), ticks);
    });
}

template <cpu_class_t cpu_class_>
template <typename checkpoint_type_>
status_t sequence_ram_variant<cpu_class_>::load(checkpoint_type_ checkpoint, executor_t& executor,
                                                progress_ticks_t& ticks) noexcept {
    return index.load(checkpoint, us::sequence::codec_t {}, executor, load_config_t {}, ticks);
}

template <cpu_class_t cpu_class_>
expected<std::size_t> sequence_ram_variant<cpu_class_>::storage_requirements() const noexcept {
    expected<std::size_t> required {0, status_t::success_k};
    status_t const sized = with_sequence_provenance(geometry, costs, [&](save_config_t const& config) noexcept {
        required = index.storage_requirements(us::sequence::codec_t {}, config, token_of(geometry.metric));
        return required.status;
    });
    return {required.value, sized};
}

/** One sequence alternative on block storage: the basic NVMe index over the strings a RAM variant
 *  saved, read-only, scored with the RAM variant's metrics and the kits @p cpu_class_ picks. Its
 *  verbs are instantiated in the class's unit, as the RAM variant's are. */
template <cpu_class_t cpu_class_>
struct sequence_nvme_variant {
    using executor_t = nvme_executor_t;
    using index_t = basic::nvme_indexes<us::sequence::codec_t, us::utilities::uuid_t, u32_t, sz_size_t,
                                        default_io_engine_t, row_kit_of<cpu_class_>,
                                        atomic_ref_for_cpu_class<cpu_class_>::template type, waiter_of<cpu_class_>>;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr sz_capability_t capability_k = stringzilla_capabilities_of(cpu_class_);

    index_t index;
    sequence_geometry_t geometry;

    /** The table and gap cost the alignment kind scores with, folded at freeze. */
    us::sequence::needleman_wunsch_costs_t costs {};

    explicit sequence_nvme_variant(forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    /** Serves the checkpoint at @p path in place, its parts from @p paths when any are named. */
    status_t view(std::string_view path, std::span<std::string_view const> paths, executor_t& executor) noexcept {
        return index.view(path, paths, token_of(geometry.metric), executor);
    }

    expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                       predicate_t const& predicate, executor_t& executor,
                                       progress_ticks_t& ticks) noexcept;

    /** A view already is its checkpoint, so saving answers @c unsupported_k. */
    status_t save(std::string_view, executor_t&, progress_ticks_t&) noexcept { return status_t::unsupported_k; }
    expected<sample_result_t> sample(basic_sample_inputs_t inputs, basic_sample_outputs_t outputs,
                                     sample_config_t config, executor_t& executor) noexcept;
};

template <cpu_class_t cpu_class_>
expected<traversal_stats_t> sequence_nvme_variant<cpu_class_>::search(
    counted_tape_t queries, matches_outputs_t& outputs, search_config_t config, predicate_t const& predicate,
    executor_t& executor, progress_ticks_t& ticks) noexcept {
    auto with_metric = [&](auto make_metric) noexcept {
        auto query_of = [&](std::size_t query_index, thread_allocator_t const&) noexcept {
            return expected<std::string_view> {chars_at(queries, query_index), status_t::success_k};
        };
        return search_entries(index, queries.count(), make_metric, query_of, outputs, config, predicate, executor,
                              ticks);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry, costs, with_metric);
}

/** Every stored string is a query as it is read. */
template <cpu_class_t cpu_class_>
expected<sample_result_t> sequence_nvme_variant<cpu_class_>::sample(basic_sample_inputs_t inputs,
                                                                    basic_sample_outputs_t outputs,
                                                                    sample_config_t config,
                                                                    executor_t& executor) noexcept {
    auto with_metric = [&](auto make_metric) noexcept {
        auto query_of = [](us::sequence::codec_t::view_t stored, thread_allocator_t const&) noexcept {
            return expected<std::string_view> {stored, status_t::success_k};
        };
        return sample_entries(*this, make_metric, query_of, inputs, outputs, config, executor);
    };
    return with_sequence_metric<capability_k, cache_hints_t>(geometry, costs, with_metric);
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

/* The engine's metric keeps the C enum's values, so a variant maps one to the other with a cast. */
static_assert(static_cast<us::sequence::vram_sequence_metric_t>(usearch_sequence_metric_levenshtein_bytes_k) ==
              us::sequence::vram_sequence_metric_t::levenshtein_bytes_k);
static_assert(static_cast<us::sequence::vram_sequence_metric_t>(usearch_sequence_metric_levenshtein_utf8_k) ==
              us::sequence::vram_sequence_metric_t::levenshtein_runes_k);
static_assert(static_cast<us::sequence::vram_sequence_metric_t>(usearch_sequence_metric_needleman_wunsch_k) ==
              us::sequence::vram_sequence_metric_t::needleman_wunsch_k);
static_assert(static_cast<us::sequence::vram_sequence_metric_t>(usearch_sequence_metric_overlap_k) ==
              us::sequence::vram_sequence_metric_t::overlap_k);

/** The faces' members reading the kits, defined in @c src/sequence/vram_variants_nvidia.cpp. */
extern template struct us::sequence::vram_nvidia_index<
    us::sequence::vram_sequence_metric_t::levenshtein_bytes_k, u32_t, sequence_nvidia_capabilities_k,
    sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_bytes_k>>;
extern template struct us::sequence::vram_nvidia_index<
    us::sequence::vram_sequence_metric_t::levenshtein_runes_k, u32_t, sequence_nvidia_capabilities_k,
    sequence_nvidia_fatbin<usearch_sequence_metric_levenshtein_utf8_k>>;
extern template struct us::sequence::vram_nvidia_index<us::sequence::vram_sequence_metric_t::overlap_k, u32_t,
                                                       sequence_nvidia_capabilities_k,
                                                       sequence_nvidia_fatbin<usearch_sequence_metric_overlap_k>>;
extern template struct us::sequence::vram_nvidia_index<
    us::sequence::vram_sequence_metric_t::needleman_wunsch_k, u32_t, sequence_nvidia_capabilities_k,
    sequence_nvidia_fatbin<usearch_sequence_metric_needleman_wunsch_k>>;

/** A GPU overlap index scores in the same fixed point as the RAM metric. */
template <typename node_id_type_, sz_capability_t capability_, typename kernels_type_>
inline constexpr usearch_distance_t distance_unit<us::sequence::vram_nvidia_index<
    us::sequence::vram_sequence_metric_t::overlap_k, node_id_type_, capability_, kernels_type_>> =
    1.0 / us::sequence::overlap_resolution_k;

/** Builds one shard of the sequence kind @p metric_ on a CUDA device, holding the window widths and
 *  costs the variant's frozen @c geometry names, which its first reservation uploads. */
template <usearch_sequence_metric_t metric_>
struct sequence_vram_nvidia_build {
    using face_t = us::sequence::vram_nvidia_index<static_cast<us::sequence::vram_sequence_metric_t>(metric_), u32_t,
                                                   sequence_nvidia_capabilities_k, sequence_nvidia_fatbin<metric_>>;

    /** The geometry and costs of the variant holding the shards, frozen before the first is
     *  placed. */
    sequence_geometry_t const* geometry = nullptr;
    us::sequence::needleman_wunsch_costs_t const* costs = nullptr;

    std::unique_ptr<face_t> operator()(int device) const noexcept {
        std::unique_ptr<face_t> face {new (std::nothrow)
                                          face_t(cuda_virtual_memory_allocator<byte_t> {gpu_drivers().cuda, device})};
        if (face && (failed(face->window_widths(to_span(geometry->window_widths))) || failed(face->costs(*costs))))
            face.reset();
        return face;
    }
};

/** The sequence alternative on NVIDIA GPUs: the basic VRAM shards under the sequence kind
 *  @p metric_ - Levenshtein over bytes or UTF-8 runes, overlap, or Needleman-Wunsch - with 32-bit
 *  node IDs. */
template <usearch_sequence_metric_t metric_>
struct sequence_vram_nvidia_variant {
    using executor_t = cuda_executor_t;
    using build_t = sequence_vram_nvidia_build<metric_>;
    using face_t = typename build_t::face_t;
    using index_t = basic::vram_nvidia_indexes<face_t, us::utilities::uuid_t, build_t>;
    using refine_plan_t = typename index_t::refine_plan_t;
    static constexpr usearch_sequence_metric_t metric_k = metric_;
    static constexpr gpu_class_t gpu_class_k = gpu_class_t::nvidia_ampere_k;

    sequence_geometry_t geometry;
    us::sequence::needleman_wunsch_costs_t costs {};
    index_t index;

    /** Holds no shard until the handle places them; their kind is the template's, and their window
     *  widths and costs the geometry's. */
    sequence_vram_nvidia_variant(int, sequence_geometry_t const&) noexcept
        : index(forkunion_allocator_t {no_memory_domain_k}, build_t {&geometry, &costs}) {}

    status_t reserve(reserve_capacity_t capacity) noexcept {
        return vram_nvidia_reserve(index, capacity.entries, geometry.connectivity, capacity.bytes);
    }

    expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t sequences, insert_config_t config,
                                       cuda_executor_t& executor, progress_ticks_t& ticks) noexcept {
        // The batch's own pages, since batches routed to disjoint shards may land at once.
        growing_arena<forkunion_allocator_t> staging {forkunion_allocator_t {no_memory_domain_k}};
        auto const [entries, lower_status] = counted_tape_of(sequences.tape, sequences.count, staging);
        if (failed(lower_status)) return {{}, lower_status};
        return vram_nvidia_insert(index, keys, bytes_tape_of(entries), geometry.connectivity, config.expansion_factor,
                                  ticks, executor);
    }

    /** A device walk takes no host predicate. */
    expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                       cuda_executor_t& executor, progress_ticks_t& ticks) noexcept {
        return vram_nvidia_search(index, bytes_tape_of(queries), outputs.wanted, config.expansion_factor, outputs,
                                  ticks, executor);
    }

    /** Writes the strings as the RAM variants write them: a lone live shard's monolith, or one per
     *  live shard beside the manifest the path names. */
    status_t save(std::string_view path, cuda_executor_t& executor, progress_ticks_t& ticks) noexcept {
        return with_sequence_provenance(geometry, costs, [&](save_config_t const& config) noexcept {
            return vram_nvidia_save(index, path, us::sequence::codec_t {}, config, token_of(geometry.metric), executor,
                                    ticks);
        });
    }

    /** Reads a checkpoint any residency wrote, at a path or in host memory, into the placed shards,
     *  one part per shard. */
    template <typename checkpoint_type_>
    status_t load(checkpoint_type_ checkpoint, cuda_executor_t& executor, progress_ticks_t& ticks) noexcept {
        return index.load(checkpoint, us::sequence::codec_t {}, executor, load_config_t {},
                          static_cast<connectivity_t>(geometry.connectivity), ticks);
    }

    /** Empties the index, retiring every shard behind a fresh shard generation on the same
     *  devices. */
    status_t clear() noexcept { return index.clear(); }

    /** Sampling answers @c unsupported_k until the shards sample on their devices. */
    expected<sample_result_t> sample(basic_sample_inputs_t, basic_sample_outputs_t, sample_config_t,
                                     executor_t&) noexcept {
        return {{}, status_t::unsupported_k};
    }
};

#endif

#pragma endregion Sequence Variants

#pragma region Sequence Bodies

/** The sequence variants each class unit compiles: one per CPU class on the host and on storage,
 *  and one per kind on an NVIDIA GPU. */
template <auto class_>
struct sequence_variants_of<vendored_residency_t::ram_k, class_> {
    using type = type_list<sequence_ram_variant<class_>>;
};
template <auto class_>
struct sequence_variants_of<vendored_residency_t::nvme_k, class_> {
    using type = type_list<sequence_nvme_variant<class_>>;
};
#if USEARCH_ARCH_CUDA_
template <>
struct sequence_variants_of<vendored_residency_t::vram_nvidia_k, gpu_class_t::nvidia_ampere_k> {
    using type = type_list<sequence_vram_nvidia_variant<usearch_sequence_metric_levenshtein_bytes_k>,
                           sequence_vram_nvidia_variant<usearch_sequence_metric_levenshtein_utf8_k>,
                           sequence_vram_nvidia_variant<usearch_sequence_metric_overlap_k>,
                           sequence_vram_nvidia_variant<usearch_sequence_metric_needleman_wunsch_k>>;
};
#endif

/** What the sequence verbs do to one variant: fold its costs, describe it, and land and tally its
 *  strings. */
template <>
struct variant_bodies<sequence_traits_t> {
    using geometry_t = sequence_geometry_t;
    using geometry_extras_t = sequence_traits_t::geometry_extras_t;
    using costs_t = us::sequence::needleman_wunsch_costs_t;

    /** What a view keeps of a string it read, copied into its thread's arena. */
    using kept_t = bytes_view_t;

    /** Whether @p donor enters @p variant: the same kind, window widths, costs and
     *  connectivity. */
    template <typename variant_type_, typename donor_type_>
    static bool matches(variant_type_ const& variant, donor_type_ const& donor) noexcept {
        geometry_t const &index = variant.geometry, &other = donor.geometry;
        if (index.metric != other.metric || index.connectivity != other.connectivity ||
            index.window_widths.size() != other.window_widths.size() ||
            std::memcmp(&variant.costs, &donor.costs, sizeof(costs_t)) != 0)
            return false;
        for (std::size_t window = 0; window != index.window_widths.size(); ++window)
            if (index.window_widths[window] != other.window_widths[window]) return false;
        return true;
    }

    /** Folds @p named weights into @p costs, the 256×256 scores read as similarities between class
     *  codes, or else unfolds the costs @p geometry recalled, one hexadecimal digit per entry;
     *  @c config_mismatch_k for weights past 32 classes, a negative gap, or a pair above its mean
     *  self-score, and @c malformed_k for recorded costs no square holds. */
    static status_t fold(geometry_t const& geometry, geometry_extras_t named, costs_t& costs) noexcept {
        if (named) {
            constexpr std::size_t side_k = us::sequence::needleman_wunsch_byte_alphabet_k;
            usearch_sequence_weights_t const& weights = *named;
            auto const [folded, fold_status] = costs_t::fold(
                std::span<i8_t const> {weights.substitution_matrix, side_k * side_k}, weights.gap);
            if (failed(fold_status)) return status_t::config_mismatch_k;
            costs = folded;
            return status_t::success_k;
        }
        std::string_view const recalled = to_string_view(geometry.substitution_costs);
        if (recalled.empty()) return status_t::success_k;
        std::size_t classes = 0;
        while (classes * classes < recalled.size()) ++classes;
        if (classes * classes != recalled.size() || classes > us::sequence::needleman_wunsch_classes_k)
            return status_t::malformed_k;
        costs.classes = static_cast<u8_t>(classes);
        costs.code_bits = us::sequence::packed_code_bits(classes);
        costs.gap = static_cast<u8_t>(geometry.gap_cost);
        for (std::size_t entry = 0; entry != recalled.size(); ++entry) {
            char const digit = recalled[entry];
            costs.substitutions[entry / classes][entry % classes] = static_cast<u8_t>(digit <= '9' ? digit - '0'
                                                                                                   : digit - 'a' + 10);
        }
        return status_t::success_k;
    }

    /** Moves @p geometry into @p variant, its weights folded, a zero connectivity taking the
     *  default; a GPU variant's shards take the window widths and the costs from it as they are
     *  built. */
    template <typename variant_type_>
    static status_t adopt(variant_type_& variant, geometry_t& geometry, geometry_extras_t weights) noexcept {
        if (!geometry.connectivity) geometry.connectivity = connectivity_32_k;
        if (status_t const folded = fold(geometry, weights, variant.costs); failed(folded)) return folded;
        variant.geometry = std::move(geometry);
        return status_t::success_k;
    }

    template <typename variant_type_>
    static void describe(variant_type_ const& variant, usearch_sequence_index_description_t& description) noexcept {
        description.metadata = sequence_traits_t::metadata_of(variant.geometry);
    }

    static status_t keep(nvme_executor_t& executor, std::size_t thread, std::string_view value, kept_t& kept) noexcept {
        return keep_entry(executor, thread, bytes_of(value.data(), value.size()), kept);
    }

    /** A device's stored string is its own bytes, kept where the call's arena holds them. */
    static status_t keep_stored(bytes_view_t stored, kept_t& kept) noexcept {
        return kept = stored, status_t::success_k;
    }

    /** Writes @p count strings, the @c i th as @p value_at hands it, as one u8 tape in
     *  @p memory. */
    template <typename variant_type_, typename value_at_type_>
    static status_t write_values(variant_type_ const&, std::size_t count, value_at_type_&& value_at,
                                 growing_arena<forkunion_allocator_t>& memory,
                                 usearch_tape_view_t* sequences) noexcept {
        std::size_t* const offsets = carve<std::size_t>(memory, count + 1);
        if (!offsets) return status_t::bad_alloc_k;
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != count; ++entry)
            offsets[entry + 1] = offsets[entry] + value_at(entry).size();
        byte_t* const bytes = carve<byte_t>(memory, offsets[count]);
        if (offsets[count] && !bytes) return status_t::bad_alloc_k;
        for (std::size_t entry = 0; entry != count; ++entry)
            if (auto const& value = value_at(entry); value.size())
                std::memcpy(bytes + offsets[entry], value.data(), value.size());
        write_output_tape(bytes, offsets, usearch_scalar_u8_k, *sequences);
        return status_t::success_k;
    }

    /** Adds one entry's bytes to a key's statistics, from any thread. */
    template <typename value_type_>
    static void tally(usearch_sequence_key_stats_t& stats, value_type_ const& value) noexcept {
        fu::standard_atomic_ref<std::size_t>(stats.total_bytes).add(value.size(), std::memory_order_relaxed);
        raise_to<fu::standard_atomic_ref>(stats.largest_bytes, value.size());
    }
};

#pragma endregion Sequence Bodies
