/**
 *  @file src/sequence_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The sequence modality's C surface: the variant over every compiled sequence arm, the
 *      handle owning the chosen one, and every exported entry point as one visit apiece.
 *
 *  Arms are ordered newest class first, the serial floor last. Their bodies compile in the basic
 *  class units, never here: the @c extern declarations below keep them out.
 */
#include <cstdio>  // `std::snprintf` - the capability string
#include <cstring> // `std::memcpy` - entries landing in an arena

#include <atomic>      // `std::atomic_flag` - a cursor's one pull at a time
#include <new>         // `std::nothrow`
#include <string_view> // `std::string_view` - checkpoint paths and tokens
#include <variant>     // `std::variant`, `std::visit`

#include "basic_specializations.cuh"

#if STRINGZILLA_TARGET_SKYLAKE
extern template struct sequence_arm<cpu_class_t::intel_skylake_k>;
#endif
#if STRINGZILLA_TARGET_HASWELL
extern template struct sequence_arm<cpu_class_t::intel_haswell_k>;
#endif
#if STRINGZILLA_TARGET_NEON
extern template struct sequence_arm<cpu_class_t::aws_g2_k>;
#endif
extern template struct sequence_arm<cpu_class_t::serial_k>;

#pragma region Handles

/** The one dynamically-typed sequence index every binding consumes: a variant over the arms this
 *  build compiled, each the basic index over owned strings scored with one class's kits. */
struct usearch_sequence_index_handle_t {

    using variants_t = std::variant<
#if STRINGZILLA_TARGET_SKYLAKE
        sequence_arm<cpu_class_t::intel_skylake_k>,
#endif
#if STRINGZILLA_TARGET_HASWELL
        sequence_arm<cpu_class_t::intel_haswell_k>,
#endif
#if STRINGZILLA_TARGET_NEON
        sequence_arm<cpu_class_t::aws_g2_k>,
#endif
        sequence_arm<cpu_class_t::serial_k>>;

    variants_t variants;

    /** The class whose kits the arm scores with, named once at construction. */
    char capabilities_string[32] {};

    template <typename arm_type_>
    usearch_sequence_index_handle_t(std::in_place_type_t<arm_type_> tag,
                                    forkunion_allocator_t const& allocator) noexcept
        : variants(tag, allocator) {}
};

/** The basic cursor over the sequence index, whose type no class changes - only the kits a metric
 *  scores with. */
using sequence_cursor_t = basic::ram_cursor<sequence_arm<cpu_class_t::serial_k>::index_t,
                                            synchronization_t::concurrent_k, forkunion_allocator_t>;

struct usearch_sequence_cursor_handle_t {
    std::variant<std::monostate, sequence_cursor_t> cursor;

    /** The handle the cursor was opened on, which every pull names again. */
    usearch_sequence_index_t index = nullptr;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag busy;
};

/** The kind, window widths and connectivity one handle freezes, as an init or a checkpoint states
 *  them. */
struct sequence_geometry_t {
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;
    std::span<std::size_t const> window_widths;
    std::size_t connectivity = 0;
};

/** Allocates the handle holding @p arm_type_, freezing @p geometry into it; null once @p outcome
 *  records why not. */
template <typename arm_type_>
static usearch_sequence_index_t make_sequence_handle(sequence_geometry_t const& geometry,
                                                     usearch_outcome_t* outcome) noexcept {
    usearch_sequence_index_t const handle = new (std::nothrow)
        usearch_sequence_index_handle_t(std::in_place_type<arm_type_>, forkunion_allocator_t {no_memory_domain_k});
    if (!handle)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    arm_type_& arm = std::get<arm_type_>(handle->variants);
    arm.metric = geometry.metric;
    arm.connectivity = geometry.connectivity ? geometry.connectivity : std::size_t(connectivity_32_k);
    if (failed(arm.window_widths.reserve_exactly(geometry.window_widths.size())))
        return delete handle, propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to keep the widths"),
               nullptr;
    for (std::size_t const width : geometry.window_widths) arm.window_widths.push_back_reserved(width);
    std::snprintf(handle->capabilities_string, sizeof(handle->capabilities_string), "%s",
                  usearch_cpu_class_name(static_cast<usearch_cpu_class_t>(arm_type_::cpu_class_k)));
    return handle;
}

/** One rung: taken when the machine reports the class, else falls through, past the last rung onto
 *  the serial floor; a class that picks no kit of its own is skipped without instantiating its
 *  arm. */
template <cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
static usearch_sequence_index_t descend_sequence_ladder(sequence_geometry_t const& geometry, nk_capability_t available,
                                                        usearch_outcome_t* outcome) noexcept {
    using arm_t = sequence_arm<cpu_class_>;
    if constexpr (sequence_rung_advances<cpu_class_>()) {
        static_assert(variant_carries<arm_t, usearch_sequence_index_handle_t::variants_t>,
                      "This rung advances past the one below - list its arm in the variant");
        if (numkong_admits(available, cpu_class_)) return make_sequence_handle<arm_t>(geometry, outcome);
    }
    else {
        static_assert(!variant_carries<arm_t, usearch_sequence_index_handle_t::variants_t>,
                      "This rung no longer advances past the one below - drop its arm from the variant list");
    }
    if constexpr (sizeof...(lower_classes_) != 0)
        return descend_sequence_ladder<lower_classes_...>(geometry, available, outcome);
    else return make_sequence_handle<sequence_arm<cpu_class_t::serial_k>>(geometry, outcome);
}

/** Lowers an init's or a checkpoint's frozen kind onto one arm, refusing the kinds no kit serves
 *  and the parameters a kind does not take; null once @p outcome records why not. */
static usearch_sequence_index_t make_sequence_index(sequence_geometry_t const& geometry,
                                                    usearch_sequence_weights_t const* weights,
                                                    usearch_ram_options_t const& ram,
                                                    usearch_outcome_t* outcome) noexcept {
    switch (geometry.metric) {
    case usearch_sequence_metric_levenshtein_bytes_k:
    case usearch_sequence_metric_levenshtein_utf8_k:
    case usearch_sequence_metric_overlap_k: break;
    case usearch_sequence_metric_needleman_wunsch_k:
    case usearch_sequence_metric_needleman_wunsch_gotoh_k:
    case usearch_sequence_metric_smith_waterman_k:
    case usearch_sequence_metric_smith_waterman_gotoh_k:
        return propagate_error(usearch_status_unsupported_k, outcome, "No alignment kernels serve this kind yet"),
               nullptr;
    default: return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown sequence metric"), nullptr;
    }
    if (weights)
        return propagate_error(usearch_status_config_mismatch_k, outcome, "Only an alignment kind takes weights"),
               nullptr;
    bool const overlap = geometry.metric == usearch_sequence_metric_overlap_k;
    if (overlap == geometry.window_widths.empty())
        return propagate_error(usearch_status_config_mismatch_k, outcome,
                               "The overlap kind needs window widths, and only it takes any"),
               nullptr;
    for (std::size_t const width : geometry.window_widths)
        if (!width) return propagate_error(usearch_status_invalid_argument_k, outcome, "A window is empty"), nullptr;
    // One basic index per handle, so one memory domain at most; values live on the heap and land
    // where first touched.
    if (ram.memory_domains && ram.memory_domains_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A sequence index places one shard"), nullptr;
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
#if NUMKONG_ARCH_X8664_
    return descend_sequence_ladder<cpu_class_t::intel_skylake_k, cpu_class_t::intel_haswell_k>(geometry, available,
                                                                                               outcome);
#elif NUMKONG_ARCH_ARM64_
    return descend_sequence_ladder<cpu_class_t::aws_g2_k>(geometry, available, outcome);
#else
    return make_sequence_handle<sequence_arm<cpu_class_t::serial_k>>(geometry, outcome);
#endif
}

/** Whether @p sequences holds @p count byte strings; false once @p outcome records why not. */
static bool accepts_sequences(usearch_tape_view_t const& sequences, std::size_t count,
                              usearch_outcome_t* outcome) noexcept {
    if (!sequences.data || !monotone_offsets(sequences, count))
        return !propagate_error(usearch_status_invalid_argument_k, outcome,
                                "The sequences need data and non-decreasing offsets");
    if (sequences.scalar != usearch_scalar_u8_k)
        return !propagate_error(usearch_status_unsupported_k, outcome, "Sequences arrive as u8 bytes");
    return true;
}

#pragma endregion Handles

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT void usearch_sequence_init_ram(usearch_sequence_init_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    if (arguments->window_widths_count && !arguments->window_widths)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The window widths must not be null"),
               void();
    sequence_geometry_t const geometry {
        arguments->metric,
        {arguments->window_widths, arguments->window_widths_count},
        arguments->connectivity,
    };
    usearch_sequence_index_t const index = make_sequence_index(geometry, arguments->weights, arguments->ram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments) {
    if (submitted<usearch_sequence_load_ram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    std::string_view const path = as_text(arguments->path);
    std::size_t connectivity = 0;
    // A checkpoint spells its widths as decimals between commas, and the field's text dies with the
    // callback.
    growing_buffer<std::size_t, forkunion_allocator_t> widths {forkunion_allocator_t {no_memory_domain_k}};
    status_t widths_status = status_t::success_k;
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "connectivity") connectivity = decimal_in(field.value);
        else if (field.key == "window_widths") {
            std::size_t windows = 1;
            for (char const character : field.value) windows += character == ',';
            widths_status = widths.reserve_exactly(windows);
            for (std::string_view rest = field.value; succeeded(widths_status) && !rest.empty();) {
                std::size_t const comma = rest.find(',');
                widths.push_back_reserved(decimal_in(rest.substr(0, comma)));
                rest = comma == std::string_view::npos ? std::string_view {} : rest.substr(comma + 1);
            }
        }
    };
    auto const [metadata, read_status] = safegraphs_read_index_metadata(path, std::allocator<byte_t> {}, per_field);
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    if (failed(widths_status)) return propagate_error(widths_status, outcome), void();
    sequence_geometry_t const geometry {sequence_metric_of(metadata.metric.view()), to_span(widths), connectivity};
    usearch_sequence_index_t const index = make_sequence_index(geometry, arguments->weights, arguments->ram, outcome);
    if (!index) return;
    bool loaded = false;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, executor->size()};
            loaded = propagate_error(arm.load(path, *executor, ticks), outcome);
        },
        index->variants);
    if (!loaded) return delete index, void();
    *arguments->index = index;
}

USEARCH_EXPORT void usearch_sequence_view_ram(usearch_sequence_view_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "Sequences own their bytes, so a checkpoint loads instead");
}

USEARCH_EXPORT void usearch_sequence_view_nvme(usearch_sequence_view_nvme_t* arguments) {
    if (submitted<usearch_sequence_view_nvme>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency serves NVMe views yet");
}

USEARCH_EXPORT void usearch_sequence_free(usearch_sequence_index_t index) {
    finish_calls_using(index);
    delete index;
}

USEARCH_EXPORT void usearch_sequence_reserve(usearch_sequence_reserve_t* arguments) {
    if (submitted<usearch_sequence_reserve>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::visit([&](auto& arm) noexcept { propagate_error(arm.index.reserve(arguments->entries).status, outcome); },
               arguments->index->variants);
}

USEARCH_EXPORT void usearch_sequence_clear(usearch_sequence_clear_t* arguments) {
    if (submitted<usearch_sequence_clear>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::visit([&](auto& arm) noexcept { propagate_error(arm.index.clear(), outcome); }, arguments->index->variants);
}

#pragma endregion Lifetime

#pragma region Introspection

USEARCH_EXPORT void usearch_sequence_describe(usearch_sequence_describe_t* arguments) {
    if (submitted<usearch_sequence_describe>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_sequence_index_t const index = arguments->index;
    std::visit(
        [&](auto& arm) noexcept {
            usearch_sequence_index_description_t& description = *arguments->description;
            description.metadata = {arm.metric, arm.connectivity};
            description.size = arm.index.size();
            description.capacity = arm.index.capacity();
            description.memory_usage = sizeof(*index) + arm.index.memory_usage();
            description.capabilities = index->capabilities_string;
        },
        index->variants);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_sequence_count(usearch_sequence_count_t* arguments) {
    if (submitted<usearch_sequence_count>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_sequence_key_stats_t& stats = *arguments->key_stats;
            arm.index.find_each(uuid_of(arguments->key), [&](typename arm_type_::index_t::node_id_t node) noexcept {
                std::size_t const bytes = arm.index.published_value_of(node).size();
                ++stats.entries;
                stats.total_bytes += bytes;
                stats.largest_bytes = max_of(stats.largest_bytes, bytes);
            });
        },
        arguments->index->variants);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Introspection

#pragma region Insertion and Search

USEARCH_EXPORT void usearch_sequence_insert(usearch_sequence_insert_t* arguments) {
    if (submitted<usearch_sequence_insert>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    std::size_t const count = arguments->count;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!arguments->keys.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The keys must not be null"), void();
    if (arguments->keys.stride && arguments->keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (!accepts_sequences(arguments->sequences, count, outcome)) return;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            if (crossed(arm.insert(arguments->keys, arguments->sequences, count, arguments->expansion_factor, ticks,
                                   *executor),
                        arguments->stats, outcome))
                propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_sequence_refine(usearch_sequence_refine_t* arguments) {
    if (submitted<usearch_sequence_refine>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency refines yet");
}

USEARCH_EXPORT void usearch_sequence_search(usearch_sequence_search_t* arguments) {
    if (submitted<usearch_sequence_search>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count, wanted = arguments->wanted;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {arguments->queries.data}, outcome);
    if (!arena) return;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!accepts_sequences(arguments->queries, count, outcome)) return;
    matches_landing_t landing;
    if (!carve_matches(arena->memory, count, wanted, landing, outcome)) return;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            if (!crossed(arm.search(arguments->queries, count, wanted, arguments->expansion_factor,
                                    arguments->predicate, landing, ticks, *executor),
                         arguments->stats, outcome))
                return;
            *arguments->matches = landing.view();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_sequence_get(usearch_sequence_get_t* arguments) {
    if (submitted<usearch_sequence_get>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->sequences = {};
    if (arguments->written) *arguments->written = 0;
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (!arena) return;
    std::visit(
        [&](auto& arm) noexcept {
            usearch_executor_handle_t call_executor;
            auto* const executor = executor_of(arm, arguments->executor, call_executor, outcome);
            if (!executor) return;
            if (status_t const rewound = executor->reserve(0); failed(rewound))
                return propagate_error(rewound, outcome), void();
            auto const ids = ids_under(arm.index, arguments->key, thread_allocator_t {executor->arena(0)});
            if (ids.empty()) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            std::size_t* const offsets = carve<std::size_t>(arena->memory, ids.size() + 1);
            if (!offsets)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            offsets[0] = 0;
            for (std::size_t entry = 0; entry != ids.size(); ++entry)
                offsets[entry + 1] = offsets[entry] + arm.index.published_value_of(ids[entry]).size();
            byte_t* const bytes = carve<byte_t>(arena->memory, offsets[ids.size()]);
            if (offsets[ids.size()] && !bytes)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            for (std::size_t entry = 0; entry != ids.size(); ++entry) {
                auto const& value = arm.index.published_value_of(ids[entry]);
                std::memcpy(bytes + offsets[entry], value.data(), value.size());
            }
            *arguments->sequences = {bytes, offsets, usearch_scalar_u8_k};
            if (arguments->written) *arguments->written = ids.size();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

#pragma endregion Insertion and Search

#pragma region Cursors Sampling and Absorbing

USEARCH_EXPORT void usearch_sequence_cursor_init(usearch_sequence_cursor_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (arguments->start)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency steps through graph links yet"),
               void();
    usearch_sequence_cursor_t const cursor = new (std::nothrow) usearch_sequence_cursor_handle_t {};
    if (!cursor)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the cursor"), void();
    cursor->index = arguments->index;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            using cursor_t =
                basic::ram_cursor<typename arm_type_::index_t, synchronization_t::concurrent_k, forkunion_allocator_t>;
            cursor_t& opened = cursor->cursor.template emplace<cursor_t>(arm.index,
                                                                         forkunion_allocator_t {no_memory_domain_k});
            // Stands on the first entry, so every pull copies from where it stands; an empty
            // snapshot stays past the end.
            if (opened.visible_count()) [[maybe_unused]]
                status_t const entered = opened.seek(0);
        },
        arguments->index->variants);
    *arguments->cursor = cursor;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_sequence_cursor_next(usearch_sequence_cursor_next_t* arguments) {
    if (submitted<usearch_sequence_cursor_next>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    if (arguments->sequences) *arguments->sequences = {};
    *arguments->written = 0;
    usearch_sequence_cursor_t const cursor = arguments->cursor;
    if (cursor->index != arguments->index)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (cursor->busy.test_and_set(std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull holds the cursor"), void();
    if (usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                          host_device_k, {}, outcome))
        std::visit(
            [&]<typename arm_type_>(arm_type_& arm) noexcept {
                using node_id_t = typename arm_type_::index_t::node_id_t;
                using cursor_t = basic::ram_cursor<typename arm_type_::index_t, synchronization_t::concurrent_k,
                                                   forkunion_allocator_t>;
                cursor_t* const typed = std::get_if<cursor_t>(&cursor->cursor);
                if (!typed)
                    return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its arm"), void();
                std::size_t const remaining = *typed ? typed->visible_count() - typed->id() : 0;
                std::size_t const taking = min_of(remaining, arguments->wanted);
                usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, taking);
                std::size_t* const offsets = arguments->sequences ? carve<std::size_t>(arena->memory, taking + 1)
                                                                  : nullptr;
                if ((taking && !keys) || (arguments->sequences && !offsets))
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                node_id_t const first = typed->id();
                if (offsets) offsets[0] = 0;
                for (std::size_t entry = 0; entry != taking; ++entry) {
                    node_id_t const node = static_cast<node_id_t>(first + entry);
                    keys[entry] = std::bit_cast<usearch_key_t>(arm.index.key_of(node).value);
                    if (offsets) offsets[entry + 1] = offsets[entry] + arm.index.published_value_of(node).size();
                }
                if (offsets) {
                    byte_t* const bytes = carve<byte_t>(arena->memory, offsets[taking]);
                    if (offsets[taking] && !bytes)
                        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"),
                               void();
                    for (std::size_t entry = 0; entry != taking; ++entry) {
                        auto const& value = arm.index.published_value_of(static_cast<node_id_t>(first + entry));
                        std::memcpy(bytes + offsets[entry], value.data(), value.size());
                    }
                    *arguments->sequences = {bytes, offsets, usearch_scalar_u8_k};
                }
                for (std::size_t step = 0; step != taking; ++step) ++*typed;
                *arguments->keys = {keys, sizeof(usearch_key_t)};
                *arguments->written = taking;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            },
            arguments->index->variants);
    cursor->busy.clear(std::memory_order_release);
}

USEARCH_EXPORT void usearch_sequence_cursor_free(usearch_sequence_cursor_t cursor) {
    finish_calls_using(cursor);
    delete cursor;
}

USEARCH_EXPORT void usearch_sequence_sample(usearch_sequence_sample_t* arguments) {
    if (submitted<usearch_sequence_sample>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No residency samples yet");
}

USEARCH_EXPORT void usearch_sequence_absorb(usearch_sequence_absorb_t* arguments) {
    if (submitted<usearch_sequence_absorb>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency absorbs donor handles yet");
}

#pragma endregion Cursors Sampling and Absorbing

#pragma region Serialization

USEARCH_EXPORT void usearch_sequence_save(usearch_sequence_save_t* arguments) {
    if (submitted<usearch_sequence_save>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::string_view const path = as_text(arguments->path);
    if (path.empty()) return propagate_error(usearch_status_invalid_argument_k, outcome, "The path is empty"), void();
    if (path.ends_with(".json"))
        return propagate_error(usearch_status_unsupported_k, outcome, "A sequence index saves one monolith"), void();
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, executor->size()};
            propagate_error(arm.save(path, *executor, ticks), outcome);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_sequence_metadata_read(usearch_sequence_metadata_read_t* arguments) {
    if (submitted<usearch_sequence_metadata_read>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_sequence_metadata_t& metadata = *arguments->metadata;
    metadata = {};
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "connectivity") metadata.connectivity = decimal_in(field.value);
    };
    auto const [read, status] = safegraphs_read_index_metadata(as_text(arguments->path), std::allocator<byte_t> {},
                                                               per_field);
    if (failed(status)) return propagate_error(status, outcome), void();
    metadata.metric = sequence_metric_of(read.metric.view());
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Serialization

} // extern "C"
