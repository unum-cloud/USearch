/**
 *  @file src/shapes_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The shapes modality's C surface: the variant over every compiled shapes arm, the handle
 *      owning the chosen one, and every exported entry point as one visit apiece.
 *
 *  Arms are ordered newest class first inside each scalar, the serial floor last. Their bodies
 *  compile in the basic class units, never here: the @c extern declarations below keep them out, as
 *  the dense dispatch unit keeps its own.
 */
#include <cstdio>  // `std::snprintf` - the capability string
#include <cstring> // `std::memcpy` - keys landing in a cursor's run

#include <atomic>      // `std::atomic_flag` - a cursor's one pull at a time
#include <new>         // `std::nothrow`
#include <string_view> // `std::string_view` - checkpoint paths and tokens
#include <variant>     // `std::variant`, `std::visit`

#include "basic_specializations.cuh"

#if NUMKONG_TARGET_SKYLAKE
extern template struct shapes_arm<nk::f32_t, cpu_class_t::intel_skylake_k>;
extern template struct shapes_arm<nk::f16_t, cpu_class_t::intel_skylake_k>;
#endif
#if NUMKONG_TARGET_HASWELL
extern template struct shapes_arm<nk::f32_t, cpu_class_t::intel_haswell_k>;
extern template struct shapes_arm<nk::f16_t, cpu_class_t::intel_haswell_k>;
#endif
#if NUMKONG_TARGET_NEONFHM
extern template struct shapes_arm<nk::f16_t, cpu_class_t::apple_m2_k>;
#endif
#if NUMKONG_TARGET_NEON
extern template struct shapes_arm<nk::f32_t, cpu_class_t::aws_g2_k>;
extern template struct shapes_arm<nk::f16_t, cpu_class_t::aws_g2_k>;
#endif
#if NUMKONG_TARGET_RVV
extern template struct shapes_arm<nk::f32_t, cpu_class_t::alibaba_c930_k>;
extern template struct shapes_arm<nk::f16_t, cpu_class_t::alibaba_c930_k>;
#endif
#if NUMKONG_TARGET_V128RELAXED
extern template struct shapes_arm<nk::f32_t, cpu_class_t::wasm64_extended_k>;
extern template struct shapes_arm<nk::f16_t, cpu_class_t::wasm64_extended_k>;
#endif
extern template struct shapes_arm<nk::f32_t, cpu_class_t::serial_k>;
extern template struct shapes_arm<nk::f16_t, cpu_class_t::serial_k>;

#pragma region Handles

/** The one dynamically-typed shapes index every binding consumes: a variant over the arms this
 *  build compiled, each the basic index over one scalar's packs scored with one class's kits. */
struct usearch_shapes_index_handle_t {

    using variants_t = std::variant<
#if NUMKONG_TARGET_SKYLAKE
        shapes_arm<nk::f32_t, cpu_class_t::intel_skylake_k>, shapes_arm<nk::f16_t, cpu_class_t::intel_skylake_k>,
#endif
#if NUMKONG_TARGET_HASWELL
        shapes_arm<nk::f32_t, cpu_class_t::intel_haswell_k>, shapes_arm<nk::f16_t, cpu_class_t::intel_haswell_k>,
#endif
#if NUMKONG_TARGET_NEONFHM
        shapes_arm<nk::f16_t, cpu_class_t::apple_m2_k>,
#endif
#if NUMKONG_TARGET_NEON
        shapes_arm<nk::f32_t, cpu_class_t::aws_g2_k>, shapes_arm<nk::f16_t, cpu_class_t::aws_g2_k>,
#endif
#if NUMKONG_TARGET_RVV
        shapes_arm<nk::f32_t, cpu_class_t::alibaba_c930_k>, shapes_arm<nk::f16_t, cpu_class_t::alibaba_c930_k>,
#endif
#if NUMKONG_TARGET_V128RELAXED
        shapes_arm<nk::f32_t, cpu_class_t::wasm64_extended_k>, shapes_arm<nk::f16_t, cpu_class_t::wasm64_extended_k>,
#endif
        shapes_arm<nk::f32_t, cpu_class_t::serial_k>, shapes_arm<nk::f16_t, cpu_class_t::serial_k>>;

    variants_t variants;

    /** The class whose kits the arm scores with, named once at construction. */
    char capabilities_string[32] {};

    template <typename arm_type_>
    usearch_shapes_index_handle_t(std::in_place_type_t<arm_type_> tag, forkunion_allocator_t const& allocator) noexcept
        : variants(tag, allocator) {}
};

/** The basic cursor over one scalar's index, whose type no class changes - only the kits a metric
 *  scores with. */
template <typename scalar_type_>
using shapes_cursor_t = basic::ram_cursor<typename shapes_arm<scalar_type_, cpu_class_t::serial_k>::index_t,
                                          synchronization_t::concurrent_k, forkunion_allocator_t>;

struct usearch_shapes_cursor_handle_t {
    std::variant<std::monostate, shapes_cursor_t<nk::f32_t>, shapes_cursor_t<nk::f16_t>> cursor;

    /** The handle the cursor was opened on, which every pull names again. */
    usearch_shapes_index_t index = nullptr;

    /** Set while a pull runs, so an overlapping one is refused rather than racing it. */
    std::atomic_flag busy;
};

/** The geometry and metric one handle freezes, as an init or a checkpoint states them. */
struct shapes_geometry_t {
    usearch_shapes_metric_t metric = usearch_shapes_metric_unknown_k;
    std::size_t points = 0;
    std::size_t connectivity = 0;
};

/** Allocates the handle holding @p arm_type_, freezing @p geometry into it; null once @p outcome
 *  records why not. */
template <typename arm_type_>
static usearch_shapes_index_t make_shapes_handle(shapes_geometry_t const& geometry,
                                                 usearch_outcome_t* outcome) noexcept {
    usearch_shapes_index_t const handle = new (std::nothrow)
        usearch_shapes_index_handle_t(std::in_place_type<arm_type_>, forkunion_allocator_t {no_memory_domain_k});
    if (!handle)
        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to allocate the index"), nullptr;
    arm_type_& arm = std::get<arm_type_>(handle->variants);
    arm.metric = geometry.metric;
    arm.points = geometry.points;
    arm.connectivity = geometry.connectivity ? geometry.connectivity : std::size_t(connectivity_32_k);
    std::snprintf(handle->capabilities_string, sizeof(handle->capabilities_string), "%s",
                  usearch_cpu_class_name(static_cast<usearch_cpu_class_t>(arm_type_::cpu_class_k)));
    return handle;
}

/** The unconditional serial floor - every ladder's last rung. */
template <typename scalar_type_>
static usearch_shapes_index_t descend_shapes_ladder(shapes_geometry_t const& geometry, nk_capability_t,
                                                    usearch_outcome_t* outcome) noexcept {
    return make_shapes_handle<shapes_arm<scalar_type_, cpu_class_t::serial_k>>(geometry, outcome);
}

/** One rung: taken when the machine reports the class, else falls through; a class that picks no
 *  kit of its own is skipped without instantiating its arm. */
template <typename scalar_type_, cpu_class_t cpu_class_, cpu_class_t... lower_classes_>
static usearch_shapes_index_t descend_shapes_ladder(shapes_geometry_t const& geometry, nk_capability_t available,
                                                    usearch_outcome_t* outcome) noexcept {
    using arm_t = shapes_arm<scalar_type_, cpu_class_>;
    if constexpr (shapes_rung_advances<scalar_type_, cpu_class_>()) {
        static_assert(variant_carries<arm_t, usearch_shapes_index_handle_t::variants_t>,
                      "This rung advances past the one below - list its arm in the variant");
        if (numkong_admits(available, cpu_class_)) return make_shapes_handle<arm_t>(geometry, outcome);
    }
    else {
        static_assert(!variant_carries<arm_t, usearch_shapes_index_handle_t::variants_t>,
                      "This rung no longer advances past the one below - drop its arm from the variant list");
    }
    return descend_shapes_ladder<scalar_type_, lower_classes_...>(geometry, available, outcome);
}

/** Picks the arm for one scalar: CPU classes newest first, each compiled only where its kits differ
 *  from the class it degrades to. */
template <typename scalar_type_>
static usearch_shapes_index_t make_shapes_index(shapes_geometry_t const& geometry,
                                                usearch_outcome_t* outcome) noexcept {
    nk_capability_t available = 0;
    if (nk_cpu_capabilities_enabled(&available) != nk_success_k) available = nk_cap_serial_k;
#if NUMKONG_ARCH_X8664_
    return descend_shapes_ladder<scalar_type_, cpu_class_t::intel_skylake_k, cpu_class_t::intel_haswell_k>(
        geometry, available, outcome);
#elif NUMKONG_ARCH_ARM64_
    return descend_shapes_ladder<scalar_type_, cpu_class_t::apple_m2_k, cpu_class_t::aws_g2_k>(geometry, available,
                                                                                               outcome);
#elif NUMKONG_ARCH_RISCV64_
    return descend_shapes_ladder<scalar_type_, cpu_class_t::alibaba_c930_k>(geometry, available, outcome);
#elif NUMKONG_ARCH_WASM_
    return descend_shapes_ladder<scalar_type_, cpu_class_t::wasm64_extended_k>(geometry, available, outcome);
#else
    return descend_shapes_ladder<scalar_type_>(geometry, available, outcome);
#endif
}

/** Lowers an init's or a checkpoint's frozen geometry onto one arm, refusing the scalars and kinds
 *  no arm serves; null once @p outcome records why not. */
static usearch_shapes_index_t make_shapes_index(shapes_geometry_t const& geometry, usearch_scalar_t quantization,
                                                usearch_ram_options_t const& ram, usearch_outcome_t* outcome) noexcept {
    if (token_of(geometry.metric).empty())
        return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown shapes metric"), nullptr;
    // One basic index per handle, so one memory domain at most; values live on the heap and land
    // where first touched.
    if (ram.memory_domains && ram.memory_domains_count > 1)
        return propagate_error(usearch_status_unsupported_k, outcome, "A shapes index places one shard"), nullptr;
    switch (quantization) {
    case usearch_scalar_f16_k: return make_shapes_index<nk::f16_t>(geometry, outcome);
    case usearch_scalar_f32_k: return make_shapes_index<nk::f32_t>(geometry, outcome);
    default: return propagate_error(usearch_status_unsupported_k, outcome, "Shapes store f16 or f32 points"), nullptr;
    }
}

/** Whether @p shapes holds @p count entries of points in a scalar the arms cast from, its offsets
 *  counting points; false once @p outcome records why not. */
static bool accepts_shapes(usearch_tape_view_t const& shapes, std::size_t count, usearch_outcome_t* outcome) noexcept {
    if (!shapes.data || !monotone_offsets(shapes, count))
        return !propagate_error(usearch_status_invalid_argument_k, outcome,
                                "The shapes need data and non-decreasing offsets");
    nk_dtype_t const dtype = nk_dtype_of(shapes.scalar);
    if (dtype == nk_dtype_unknown_k || nk_dtype_bits(dtype) % 8)
        return !propagate_error(usearch_status_unsupported_k, outcome, "Points arrive in a whole-byte scalar");
    return true;
}

#pragma endregion Handles

extern "C" {

#pragma region Lifetime

USEARCH_EXPORT void usearch_shapes_init_ram(usearch_shapes_init_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    shapes_geometry_t const geometry {arguments->metric, arguments->points, arguments->connectivity};
    usearch_shapes_index_t const index = make_shapes_index(geometry, arguments->quantization, arguments->ram, outcome);
    if (!index) return;
    *arguments->index = index;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_shapes_init_vram(usearch_shapes_init_vram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "No GPU kernels serve shapes yet");
}

USEARCH_EXPORT void usearch_shapes_load_ram(usearch_shapes_load_ram_t* arguments) {
    if (submitted<usearch_shapes_load_ram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    std::string_view const path = as_text(arguments->path);
    shapes_geometry_t geometry;
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "connectivity") geometry.connectivity = decimal_in(field.value);
        else if (field.key == "points") geometry.points = decimal_in(field.value);
    };
    auto const [metadata, read_status] = safegraphs_read_index_metadata(path, std::allocator<byte_t> {}, per_field);
    if (failed(read_status)) return propagate_error(read_status, outcome), void();
    geometry.metric = shapes_metric_of(metadata.metric.view());
    usearch_scalar_t const quantization = metadata.value_dtype == dtype_t::f16_k   ? usearch_scalar_f16_k
                                          : metadata.value_dtype == dtype_t::f32_k ? usearch_scalar_f32_k
                                                                                   : usearch_scalar_unknown_k;
    usearch_shapes_index_t const index = make_shapes_index(geometry, quantization, arguments->ram, outcome);
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

USEARCH_EXPORT void usearch_shapes_load_vram(usearch_shapes_load_vram_t* arguments) {
    if (submitted<usearch_shapes_load_vram>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_device_k, outcome, "No GPU kernels serve shapes yet");
}

USEARCH_EXPORT void usearch_shapes_view_ram(usearch_shapes_view_ram_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "Shapes own their packs, so a checkpoint loads instead");
}

USEARCH_EXPORT void usearch_shapes_view_nvme(usearch_shapes_view_nvme_t* arguments) {
    if (submitted<usearch_shapes_view_nvme>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->index = nullptr;
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency serves NVMe views yet");
}

USEARCH_EXPORT void usearch_shapes_free(usearch_shapes_index_t index) {
    finish_calls_using(index);
    delete index;
}

USEARCH_EXPORT void usearch_shapes_reserve(usearch_shapes_reserve_t* arguments) {
    if (submitted<usearch_shapes_reserve>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::visit([&](auto& arm) noexcept { propagate_error(arm.index.reserve(arguments->entries).status, outcome); },
               arguments->index->variants);
}

USEARCH_EXPORT void usearch_shapes_clear(usearch_shapes_clear_t* arguments) {
    if (submitted<usearch_shapes_clear>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::visit(
        [&](auto& arm) noexcept {
            status_t const cleared = arm.index.clear();
            if (succeeded(cleared))
                standard_atomic_ref<std::size_t>(arm.total_points).store(0, std::memory_order_relaxed);
            propagate_error(cleared, outcome);
        },
        arguments->index->variants);
}

#pragma endregion Lifetime

#pragma region Introspection

USEARCH_EXPORT void usearch_shapes_describe(usearch_shapes_describe_t* arguments) {
    if (submitted<usearch_shapes_describe>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_shapes_index_t const index = arguments->index;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_shapes_index_description_t& description = *arguments->description;
            description.metadata = {
                arm.metric,
                std::is_same_v<typename arm_type_::scalar_t, nk::f16_t> ? usearch_scalar_f16_k : usearch_scalar_f32_k,
                arm.points, arm.connectivity};
            description.size = arm.index.size();
            description.capacity = arm.index.capacity();
            description.total_points =
                standard_atomic_ref<std::size_t const>(arm.total_points).load(std::memory_order_relaxed);
            description.memory_usage = sizeof(*index) + arm.index.memory_usage();
            description.capabilities = index->capabilities_string;
        },
        index->variants);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

USEARCH_EXPORT void usearch_shapes_count(usearch_shapes_count_t* arguments) {
    if (submitted<usearch_shapes_count>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    *arguments->key_stats = {};
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_shapes_key_stats_t& stats = *arguments->key_stats;
            arm.index.find_each(uuid_of(arguments->key), [&](typename arm_type_::index_t::node_id_t node) noexcept {
                std::size_t const points = arm.index.published_value_of(node).max_points();
                ++stats.entries;
                stats.total_points += points;
                stats.largest_points = max_of(stats.largest_points, points);
            });
        },
        arguments->index->variants);
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Introspection

#pragma region Insertion and Search

USEARCH_EXPORT void usearch_shapes_insert(usearch_shapes_insert_t* arguments) {
    if (submitted<usearch_shapes_insert>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    std::size_t const count = arguments->count;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!arguments->keys.data)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The keys must not be null"), void();
    if (arguments->keys.stride && arguments->keys.stride < sizeof(usearch_key_t))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "The key stride is shorter than a key"),
               void();
    if (!accepts_shapes(arguments->shapes, count, outcome)) return;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            progress_ticks_t ticks {arguments->progress, count};
            if (crossed(arm.insert(arguments->keys, arguments->shapes, count, arguments->expansion_factor, ticks,
                                   *executor),
                        arguments->stats, outcome))
                propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

USEARCH_EXPORT void usearch_shapes_refine(usearch_shapes_refine_t* arguments) {
    if (submitted<usearch_shapes_refine>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency refines yet");
}

USEARCH_EXPORT void usearch_shapes_search(usearch_shapes_search_t* arguments) {
    if (submitted<usearch_shapes_search>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    std::size_t const count = arguments->count, wanted = arguments->wanted;
    if (arguments->stats) *arguments->stats = {};
    *arguments->matches = {};
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {arguments->queries.data}, outcome);
    if (!arena) return;
    if (!count) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
    if (!accepts_shapes(arguments->queries, count, outcome)) return;
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

USEARCH_EXPORT void usearch_shapes_get(usearch_shapes_get_t* arguments) {
    if (submitted<usearch_shapes_get>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->shapes = {};
    if (arguments->written) *arguments->written = 0;
    nk_dtype_t const exported = nk_dtype_of(arguments->outputs_scalar);
    if (exported == nk_dtype_unknown_k || nk_dtype_bits(exported) % 8)
        return propagate_error(usearch_status_unsupported_k, outcome, "Points export to a whole-byte scalar"), void();
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (!arena) return;
    std::visit(
        [&]<typename arm_type_>(arm_type_& arm) noexcept {
            usearch_executor_handle_t call_executor;
            typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                         outcome);
            if (!executor) return;
            if (status_t const rewound = executor->reserve(0); failed(rewound))
                return propagate_error(rewound, outcome), void();
            thread_allocator_t const scratch {executor->arena(0)};
            auto const ids = ids_under(arm.index, arguments->key, scratch);
            if (ids.empty()) return propagate_error(usearch_status_success_k, outcome, nullptr), void();
            std::size_t const element_bytes = nk_dtype_bits(exported) / 8;
            std::size_t* const offsets = carve<std::size_t>(arena->memory, ids.size() + 1);
            if (!offsets)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            offsets[0] = 0;
            for (std::size_t entry = 0; entry != ids.size(); ++entry)
                offsets[entry + 1] = offsets[entry] + arm.index.published_value_of(ids[entry]).max_points();
            byte_t* const points = carve<byte_t>(arena->memory, offsets[ids.size()] * 3 * element_bytes);
            if (offsets[ids.size()] && !points)
                return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
            for (std::size_t entry = 0; entry != ids.size(); ++entry)
                if (status_t const exported_status = arm.export_points(
                        arm.index.published_value_of(ids[entry]), exported, points + offsets[entry] * 3 * element_bytes,
                        scratch);
                    failed(exported_status))
                    return propagate_error(exported_status, outcome), void();
            *arguments->shapes = {points, offsets, arguments->outputs_scalar};
            if (arguments->written) *arguments->written = ids.size();
            propagate_error(usearch_status_success_k, outcome, nullptr);
        },
        arguments->index->variants);
}

#pragma endregion Insertion and Search

#pragma region Cursors and Absorbing

USEARCH_EXPORT void usearch_shapes_cursor_init(usearch_shapes_cursor_init_t* arguments) {
    call_outcome_t outcome(arguments->outcome);
    *arguments->cursor = nullptr;
    if (arguments->start)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency steps through graph links yet"),
               void();
    usearch_shapes_cursor_t const cursor = new (std::nothrow) usearch_shapes_cursor_handle_t {};
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

USEARCH_EXPORT void usearch_shapes_cursor_next(usearch_shapes_cursor_next_t* arguments) {
    if (submitted<usearch_shapes_cursor_next>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    *arguments->keys = {};
    if (arguments->shapes) *arguments->shapes = {};
    *arguments->written = 0;
    usearch_shapes_cursor_t const cursor = arguments->cursor;
    if (cursor->index != arguments->index)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The cursor was opened on another index"),
               void();
    if (cursor->busy.test_and_set(std::memory_order_acquire))
        return propagate_error(usearch_status_contended_k, outcome, "Another pull holds the cursor"), void();
    nk_dtype_t const exported = nk_dtype_of(arguments->outputs_scalar);
    usearch_arena_handle_t* const arena = claim_arena(arguments->arena, arguments->outcome, arguments->executor,
                                                      host_device_k, {}, outcome);
    if (arena && arguments->shapes && (exported == nk_dtype_unknown_k || nk_dtype_bits(exported) % 8))
        propagate_error(usearch_status_unsupported_k, outcome, "Points export to a whole-byte scalar");
    else if (arena)
        std::visit(
            [&]<typename arm_type_>(arm_type_& arm) noexcept {
                using cursor_t = basic::ram_cursor<typename arm_type_::index_t, synchronization_t::concurrent_k,
                                                   forkunion_allocator_t>;
                cursor_t* const typed = std::get_if<cursor_t>(&cursor->cursor);
                if (!typed)
                    return propagate_error(usearch_status_internal_k, outcome, "The cursor lost its arm"), void();
                usearch_executor_handle_t call_executor;
                typename arm_type_::executor_t* const executor = executor_of(arm, arguments->executor, call_executor,
                                                                             outcome);
                if (!executor) return;
                if (status_t const rewound = executor->reserve(0); failed(rewound))
                    return propagate_error(rewound, outcome), void();
                thread_allocator_t const scratch {executor->arena(0)};
                std::size_t const remaining = *typed ? typed->visible_count() - typed->id() : 0;
                std::size_t const taking = min_of(remaining, arguments->wanted);
                usearch_key_t* const keys = carve<usearch_key_t>(arena->memory, taking);
                std::size_t* const offsets = arguments->shapes ? carve<std::size_t>(arena->memory, taking + 1)
                                                               : nullptr;
                if ((taking && !keys) || (arguments->shapes && !offsets))
                    return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"), void();
                typename arm_type_::index_t::node_id_t const first = typed->id();
                if (offsets) offsets[0] = 0;
                for (std::size_t entry = 0; entry != taking; ++entry) {
                    auto const node = static_cast<typename arm_type_::index_t::node_id_t>(first + entry);
                    keys[entry] = std::bit_cast<usearch_key_t>(arm.index.key_of(node).value);
                    if (offsets) offsets[entry + 1] = offsets[entry] + arm.index.published_value_of(node).max_points();
                }
                if (offsets) {
                    std::size_t const element_bytes = nk_dtype_bits(exported) / 8;
                    byte_t* const points = carve<byte_t>(arena->memory, offsets[taking] * 3 * element_bytes);
                    if (offsets[taking] && !points)
                        return propagate_error(usearch_status_out_of_memory_k, outcome, "Failed to grow the arena"),
                               void();
                    for (std::size_t entry = 0; entry != taking; ++entry) {
                        auto const node = static_cast<typename arm_type_::index_t::node_id_t>(first + entry);
                        if (status_t const exported_status = arm.export_points(
                                arm.index.published_value_of(node), exported,
                                points + offsets[entry] * 3 * element_bytes, scratch);
                            failed(exported_status))
                            return propagate_error(exported_status, outcome), void();
                    }
                    *arguments->shapes = {points, offsets, arguments->outputs_scalar};
                }
                for (std::size_t step = 0; step != taking; ++step) ++*typed;
                *arguments->keys = {keys, sizeof(usearch_key_t)};
                *arguments->written = taking;
                propagate_error(usearch_status_success_k, outcome, nullptr);
            },
            arguments->index->variants);
    cursor->busy.clear(std::memory_order_release);
}

USEARCH_EXPORT void usearch_shapes_cursor_free(usearch_shapes_cursor_t cursor) {
    finish_calls_using(cursor);
    delete cursor;
}

USEARCH_EXPORT void usearch_shapes_sample(usearch_shapes_sample_t* arguments) {
    if (submitted<usearch_shapes_sample>(arguments)) return;
    call_outcome_t outcome(arguments->outcome, arguments->arena);
    if (arguments->stats) *arguments->stats = {};
    propagate_error(usearch_status_unsupported_k, outcome, "No residency samples yet");
}

USEARCH_EXPORT void usearch_shapes_absorb(usearch_shapes_absorb_t* arguments) {
    if (submitted<usearch_shapes_absorb>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    propagate_error(usearch_status_unsupported_k, outcome, "No basic residency absorbs donor handles yet");
}

#pragma endregion Cursors and Absorbing

#pragma region Serialization

USEARCH_EXPORT void usearch_shapes_save(usearch_shapes_save_t* arguments) {
    if (submitted<usearch_shapes_save>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    std::string_view const path = as_text(arguments->path);
    if (path.empty()) return propagate_error(usearch_status_invalid_argument_k, outcome, "The path is empty"), void();
    if (path.ends_with(".json"))
        return propagate_error(usearch_status_unsupported_k, outcome, "A shapes index saves one monolith"), void();
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

USEARCH_EXPORT void usearch_shapes_metadata_read(usearch_shapes_metadata_read_t* arguments) {
    if (submitted<usearch_shapes_metadata_read>(arguments)) return;
    call_outcome_t outcome(arguments->outcome);
    usearch_shapes_metadata_t& metadata = *arguments->metadata;
    metadata = {};
    auto per_field = [&](safegraphs_custom_field_t field) noexcept {
        if (field.key == "connectivity") metadata.connectivity = decimal_in(field.value);
        else if (field.key == "points") metadata.points = decimal_in(field.value);
    };
    auto const [read, status] = safegraphs_read_index_metadata(as_text(arguments->path), std::allocator<byte_t> {},
                                                               per_field);
    if (failed(status)) return propagate_error(status, outcome), void();
    metadata.metric = shapes_metric_of(read.metric.view());
    metadata.quantization = read.value_dtype == dtype_t::f16_k   ? usearch_scalar_f16_k
                            : read.value_dtype == dtype_t::f32_k ? usearch_scalar_f32_k
                                                                 : usearch_scalar_unknown_k;
    propagate_error(usearch_status_success_k, outcome, nullptr);
}

#pragma endregion Serialization

} // extern "C"
