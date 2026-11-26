/**
 *  @file src/basic_punned.cuh
 *  @author Ash Vardanian
 *  @date October 25, 2026
 *  @brief What the basic dispatch units and the class units agree on: the verb table one variant
 *      compiles, the class tables each class unit instantiates, and the traits each modality's
 *      verbs differ in, with the composites they lower each C argument structure onto once.
 *
 *  It includes no tower: a modality's dispatch unit reaches its variant only through its
 *  residency's verb table, and @c basic_specializations.cuh with each
 *  @c <modality>_specializations.cuh fills those tables from the towers. What a verb does to one
 *  variant, as adopting a geometry or writing its values, is a @c variant_bodies specialization
 *  beside the variants.
 */
#pragma once

#include <cmath>   // `std::isnan` - an observation without a position
#include <cstring> // `std::memcpy` - a key read through its stride

#include <array>       // `std::array` - the devices a GPU executor spans
#include <charconv>    // `std::to_chars`, `std::from_chars` - the metadata a checkpoint records as text
#include <limits>      // `std::numeric_limits` - the widths of a tape's offsets
#include <span>        // `std::span`
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity_t`
#include <variant>     // `std::monostate` - a verb a modality takes no input for

#include "usearch/utilities/safegraphs_format.hpp" // `safegraphs_read_index_metadata` - a checkpoint's fields

#include "lib.hpp"

#pragma region Crossings

namespace util = unum::usearch::utilities;

/** Key @p offset of the caller's keys, through their stride; zero repeats the first. */
inline usearch_key_t key_at(usearch_keys_view_t keys, std::size_t offset) noexcept {
    std::size_t const stride = keys.stride;
    usearch_key_t key;
    std::memcpy(&key, reinterpret_cast<util::byte_t const*>(keys.data) + offset * stride, sizeof(key));
    return key;
}

/** Whether @p count + 1 offsets never fall; they may start anywhere, as an Arrow slice's do. */
template <typename offset_type_>
bool monotone_offsets(offset_type_ const* offsets, std::size_t count) noexcept {
    if (!offsets) return false;
    for (std::size_t entry = 0; entry != count; ++entry)
        if (offsets[entry + 1] < offsets[entry]) return false;
    return true;
}

/** Whether a tape's @p count + 1 offsets never fall, in the width they arrive in. */
inline bool monotone_offsets(usearch_tape_view_t const& tape, std::size_t count) noexcept {
    return tape.layout == usearch_tape_u64_k ? monotone_offsets(tape.offsets.u64, count)
                                             : monotone_offsets(tape.offsets.u32, count);
}

/** Whether @p layout names a width of offsets. */
inline bool known_layout(usearch_tape_layout_t layout) noexcept {
    return layout == usearch_tape_u32_k || layout == usearch_tape_u64_k;
}

/** Copies the offsets at @p source into @p widened, refusing one past what @c std::size_t holds. */
template <typename offset_type_>
util::status_t widen_offsets(offset_type_ const* source, std::span<std::size_t> widened) noexcept {
    for (std::size_t& offset : widened) {
        if (*source > (std::numeric_limits<std::size_t>::max)()) return util::status_code_t::overflow_k;
        offset = static_cast<std::size_t>(*source++);
    }
    return util::status_code_t::success_k;
}

/** The @p count + 1 offsets of the caller's @p tape as the engines read them: a u64 tape's in place
 *  where @c std::size_t matches it, and any other widened onto @p pages. */
inline util::expected<std::span<std::size_t const>> offsets_of(
    usearch_tape_view_t const& tape, std::size_t count,
    util::growing_arena<util::forkunion_allocator_t>& pages) noexcept {
    if constexpr (sizeof(std::size_t) == sizeof(util::u64_t))
        if (tape.layout == usearch_tape_u64_k)
            return {{reinterpret_cast<std::size_t const*>(tape.offsets.u64), count + 1},
                    util::status_code_t::success_k};
    std::size_t* const widened = carve<std::size_t>(pages, count + 1);
    if (!widened) return {{}, util::status_code_t::out_of_memory_k};
    std::span<std::size_t> const offsets {widened, count + 1};
    util::status_t const status = tape.layout == usearch_tape_u64_k ? widen_offsets(tape.offsets.u64, offsets)
                                                                    : widen_offsets(tape.offsets.u32, offsets);
    return {offsets, status};
}

/** Lays the @p offsets a get wrote out as @p tape 's over @p data of @p scalar, in place, as wide
 *  as @c std::size_t. */
inline void write_output_tape(void const* data, std::size_t const* offsets, usearch_scalar_t scalar,
                              usearch_tape_view_t& tape) noexcept {
    tape = {data, usearch_tape_u64_k, {}, scalar};
    if constexpr (sizeof(std::size_t) == sizeof(util::u64_t))
        tape.offsets.u64 = reinterpret_cast<util::u64_t const*>(offsets);
    else tape.layout = usearch_tape_u32_k, tape.offsets.u32 = reinterpret_cast<util::u32_t const*>(offsets);
}

/** Raises @p word to @p value, if it is larger, from any thread, through the CPU class's reference;
 *  a relaxed peek first, so a word already that large is never written. */
template <template <typename> class atomic_ref_template_, typename word_type_>
void raise_to(word_type_& word, std::type_identity_t<word_type_> value) noexcept {
    atomic_ref_template_<word_type_> const raised {word};
    if (raised.load(std::memory_order_relaxed) < value) raised.store_max(value, std::memory_order_relaxed);
}

/** The decimal spelling of @p value in @p text, which a checkpoint's metadata records. */
inline std::string_view decimal_of(std::size_t value, std::span<char> text) noexcept {
    std::to_chars_result const printed = std::to_chars(text.data(), text.data() + text.size(), value);
    return {text.data(), static_cast<std::size_t>(printed.ptr - text.data())};
}

/** The number a checkpoint's metadata spells in @p text, or zero when it spells none. */
inline std::size_t decimal_in(std::string_view text) noexcept {
    std::size_t value = 0;
    std::from_chars(text.data(), text.data() + text.size(), value);
    return value;
}

/** The value of the hexadecimal @p digit a checkpoint's metadata spells, or minus one for any
 *  other character. */
constexpr int hexadecimal_in(char digit) noexcept {
    if (digit >= '0' && digit <= '9') return digit - '0';
    if (digit >= 'a' && digit <= 'f') return digit - 'a' + 10;
    return -1;
}

#pragma endregion Crossings

#pragma region Variant Tables

/** A caller's tape as every variant reads it, bytes or points in its scalar around the offsets
 *  @ref offsets_of lowered once. */
struct counted_tape_t {
    void const* elements = nullptr;
    std::span<std::size_t const> offsets;
    usearch_scalar_t scalar = usearch_scalar_unknown_k;

    /** Entries the tape holds, one fewer than its offsets. */
    std::size_t count() const noexcept { return offsets.size() - 1; }
};

/** A caller's tape of @c count entries as its C view holds them, which an insertion lowers into its
 *  executor's scratch after the one rewind. */
struct caller_tape_t {
    usearch_tape_view_t tape {};
    std::size_t count = 0;
};

/** The elements a caller's @p tape spans, from its first offset to its last. */
inline std::size_t elements_in(caller_tape_t const& tape) noexcept {
    return tape.tape.layout == usearch_tape_u64_k
               ? static_cast<std::size_t>(tape.tape.offsets.u64[tape.count] - tape.tape.offsets.u64[0])
               : tape.tape.offsets.u32[tape.count] - tape.tape.offsets.u32[0];
}

/** Lowers the @p count entries of the caller's @p tape, widening its offsets onto @p pages. */
inline util::expected<counted_tape_t> counted_tape_of(
    usearch_tape_view_t const& tape, std::size_t count,
    util::growing_arena<util::forkunion_allocator_t>& pages) noexcept {
    auto const [offsets, status] = offsets_of(tape, count, pages);
    return {{tape.data, offsets, tape.scalar}, status};
}

/** Folds the case of every text of @p texts back to back on @p memory, on @p threads where there
 *  are any and on the caller otherwise. */
using fold_texts_t = util::expected<counted_tape_t> (*)(counted_tape_t texts,
                                                        util::growing_arena<util::forkunion_allocator_t>& memory,
                                                        util::ram_executor_t* threads) noexcept;

/** Entry @p entry of a byte tape, as the view every string type converts from. */
inline std::string_view chars_at(counted_tape_t const& tape, std::size_t entry) noexcept {
    return {static_cast<char const*>(tape.elements) + tape.offsets[entry],
            tape.offsets[entry + 1] - tape.offsets[entry]};
}

/** The entries of the caller's byte tape @p tape, as the engines read tapes. */
inline util::tape_view<util::byte_t const> bytes_tape_of(counted_tape_t const& tape) noexcept {
    return {static_cast<util::byte_t const*>(tape.elements), tape.offsets};
}

/** Where a radius walk writes: a tape of matches per query in the arena, and how many it wrote. */
struct within_outputs_t {
    usearch_matches_tape_view_t matches {};
    std::size_t written = 0;
};

/** What a reservation asks room for: entries, and the bytes or points their values take where a
 *  residency reserves those too. */
struct reserve_capacity_t {
    std::size_t entries = 0;
    std::size_t bytes = 0;
    std::size_t points = 0;
};

/** What a basic SAMPLE reads and writes, in the keys every basic index stores. */
using basic_sample_inputs_t = util::sample_inputs<util::uuid128_t>;
using basic_sample_outputs_t = util::sample_outputs<util::uuid128_t>;

/** What a modality's verbs do to one variant beyond its index: adopting a geometry, describing it,
 *  and writing, keeping and tallying its values; each modality's specializations define it
 *  alongside their own variants. */
template <typename traits_type_>
struct variant_bodies;

/**
 *  @brief The verbs one basic variant compiles in its class unit, as function pointers over the
 *      variant passed as @c void*, which only these entries cast back to its type.
 *
 *  Every entry takes <tt>(variant, inputs, outputs, config, executor, ticks)</tt>, as much of it as
 *  the verb reads, with a walk's @c predicate_t beside its config, in the C++ composites
 *  @p traits_type_ names. Every residency fills every entry, answering @c read_only_k or
 *  @c unsupported_k where it serves none, so a modality's dispatch unit visits three residencies
 *  and never a variant. Errors only return; the dispatch unit alone writes outcomes.
 */
template <typename traits_type_, vendored_residency_t residency_>
struct basic_verbs {
    using traits_t = traits_type_;
    using executor_t = typename vendored_residency_executor<residency_>::type;
    using geometry_t = typename traits_type_::geometry_t;
    using values_t = typename traits_type_::values_t;
    using queries_t = typename traits_type_::queries_t;
    using within_queries_t = typename traits_type_::within_queries_t;
    using get_outputs_t = typename traits_type_::get_outputs_t;

    /** The scalar the variant stores values in, or unknown where it stores any. */
    nk_dtype_t dtype;

    /** The metric the variant fixes, as the modality's enumerator, or zero where it scores by
     *  any. */
    util::u32_t metric;

    // Lifetime

    /** Builds an unplaced variant of @p geometry, which adopts @p adopted: a host one keeping its
     *  tables through @p tables, a device one on @p residency 's device. */
    util::expected<void*> (*create)(geometry_t geometry, typename traits_type_::geometry_extras_t const& adopted,
                                    arena_residency_t residency, util::forkunion_allocator_t tables) noexcept;
    void (*destroy)(void* variant) noexcept;
    void (*bound_shards)(void* variant, std::size_t nodes) noexcept;

    /** Appends one shard on a device, or on RAM on @p memory_domain. */
    util::status_t (*emplace)(void* variant, arena_residency_t residency, fu_memory_domain_id_t memory_domain) noexcept;

    /** Serves a checkpoint in place, an NVMe view through @p executor, which a host variant does
     *  without. */
    util::status_t (*view)(void* variant, std::string_view path, std::span<std::string_view const> paths,
                           executor_t* executor) noexcept;

    // Shape

    /** Where the variant computes, and where a call's arena must live. */
    arena_residency_t (*residency)(void const* variant) noexcept;

    /** The distinct devices the shards sit on into @p listed, as many as fit; answers how many. */
    std::size_t (*devices)(void const* variant, std::span<arena_residency_t> listed) noexcept;
    void (*describe)(void const* variant, typename traits_type_::description_t& description) noexcept;

    /** Whether the variant at @p source shares the geometry of the one at @p variant. */
    bool (*matches)(void const* variant, void const* source) noexcept;

    /** The case folding of the variant's class, null for a modality storing no text. */
    fold_texts_t fold_texts;

    /** Whether the variant takes writes, which a view refuses with @c read_only_k. */
    util::status_t (*serves_writes)(void const* variant) noexcept;
    util::status_t (*reserve)(void* variant, reserve_capacity_t capacity) noexcept;
    util::status_t (*clear)(void* variant) noexcept;
    util::expected<util::traversal_stats_t> (*insert)(void* variant, usearch_keys_view_t keys, values_t values,
                                                      insert_config_t config, executor_t& executor,
                                                      progress_ticks_t& ticks) noexcept;

    // Refining

    /** Audits the reachability of every shard. */
    util::expected<util::reachability_t> (*health)(void* variant, executor_t& executor) noexcept;

    // Plans

    /** Opens a refine plan of rounds over the variant's shards. */
    util::expected<void*> (*refine_plan_open)(void* variant, refine_config_t config, executor_t& executor) noexcept;

    /** Opens a reshard plan moving the entries of @p sources into @p targets, then running the
     *  rounds of every target that took entries; the plan keeps its own lists. */
    util::expected<void*> (*reshard_plan_open)(variants_view_t sources, variants_view_t targets,
                                               reshard_options_t options, executor_t& executor) noexcept;
    plan_verbs<executor_t> refine_plan;
    plan_verbs<executor_t> reshard_plan;

    // Lookups
    util::status_t (*count)(void* variant, usearch_key_t key, typename traits_type_::key_stats_t& stats,
                            executor_t& executor) noexcept;

    /** Writes the entries of @p key to @p outputs, carved from @p memory; answers how many. */
    util::expected<std::size_t> (*get)(void* variant, usearch_key_t key, get_outputs_t outputs,
                                       util::growing_arena<util::forkunion_allocator_t>& memory,
                                       executor_t& executor) noexcept;

    // Walks

    /** Whether the variant serves @p queries, past what the traits check without it. */
    util::status_t (*serves_search)(void const* variant, queries_t const& queries) noexcept;

    /** A device walk stages its queries on the arena's host @p pages, where it writes its
     *  matches; a variant whose walks take no @p hints refuses them with @c unsupported_k. */
    util::expected<util::traversal_stats_t> (*search)(void* variant, queries_t queries,
                                                      usearch_matches_view_t const* hints, matches_outputs_t& outputs,
                                                      search_config_t config, predicate_t predicate,
                                                      util::growing_arena<util::forkunion_allocator_t>& pages,
                                                      executor_t& executor, progress_ticks_t& ticks) noexcept;
    util::status_t (*serves_within)(void const* variant, within_queries_t const& queries,
                                    predicate_t const& predicate) noexcept;
    util::expected<util::traversal_stats_t> (*within)(void* variant, within_queries_t queries,
                                                      within_outputs_t& outputs, search_config_t config,
                                                      predicate_t predicate,
                                                      util::growing_arena<util::forkunion_allocator_t>& memory,
                                                      executor_t& executor, progress_ticks_t& ticks) noexcept;

    // Exploration

    /** Opens a walk over every entry as stored, or breadth-first from @p starts. */
    util::expected<void*> (*cursor_init)(void* variant, std::span<usearch_key_t const> starts,
                                         util::forkunion_allocator_t allocator) noexcept;

    /** Pulls up to @p wanted next keys into @p memory, gathering their links into @p links where
     *  named. */
    util::expected<std::span<usearch_key_t const>> (*cursor_next)(
        void* variant, void* cursor, links_outputs_t* links, std::size_t wanted,
        util::growing_arena<util::forkunion_allocator_t>& memory, executor_t& executor) noexcept;
    void (*cursor_free)(void* cursor) noexcept;

    /** Draws up to the outputs' length around the focus keys, beside and placed among the shown
     *  ones, never scanning; a GPU variant does not sample yet. */
    util::expected<util::sample_result_t> (*sample)(void* variant, basic_sample_inputs_t inputs,
                                                    basic_sample_outputs_t outputs, util::sample_config_t config,
                                                    executor_t& executor) noexcept;

    // Persistence
    util::status_t (*load)(void* variant, std::string_view path, executor_t& executor,
                           progress_ticks_t& ticks) noexcept;
    util::status_t (*load_buffer)(void* variant, util::bytes_view_t buffer, executor_t& executor,
                                  progress_ticks_t& ticks) noexcept;
    util::status_t (*save)(void* variant, std::string_view path, executor_t& executor,
                           progress_ticks_t& ticks) noexcept;

    /** Writes the checkpoint into @p buffer, exactly @c storage_requirements bytes long. */
    util::status_t (*save_buffer)(void* variant, util::bytes_span_t buffer, executor_t& executor,
                                  progress_ticks_t& ticks) noexcept;
    util::expected<std::size_t> (*storage_requirements)(void const* variant) noexcept;
};

/** The variants one modality's class unit compiles for one residency: @c find answers the verb
 *  table of the first variant whose scalar and kind the geometry names, or null where none of the
 *  class's variants does. */
template <typename traits_type_, vendored_residency_t residency_, auto class_>
struct basic_class_tables {
    static basic_verbs<traits_type_, residency_> const* find(
        typename traits_type_::geometry_t const& geometry) noexcept;
};

/** One picked variant as a modality's dispatch unit holds it for one call: its residency's verbs,
 *  and the variant they serve. */
template <typename traits_type_, vendored_residency_t residency_>
struct punned_basic_index {
    using executor_t = typename basic_verbs<traits_type_, residency_>::executor_t;

    basic_verbs<traits_type_, residency_> const& verbs;
    void* variant;
};

/** The executor a null handle runs a device variant's work on: the null legacy stream of every
 *  device its shards sit on, which the call opens nothing for. */
template <typename traits_type_, vendored_residency_t residency_>
    requires is_gpu_executor<typename punned_basic_index<traits_type_, residency_>::executor_t>
bool make_call_executor(punned_basic_index<traits_type_, residency_> const& punned,
                        typename punned_basic_index<traits_type_, residency_>::executor_t& call_executor,
                        usearch_outcome_t* outcome) noexcept {
    std::array<arena_residency_t, util::gpu_devices_capacity_k> residencies;
    std::size_t devices_count = punned.verbs.devices(punned.variant, residencies);
    if (!devices_count) residencies[devices_count++] = punned.verbs.residency(punned.variant);
    std::array<int, util::gpu_devices_capacity_k> devices;
    for (std::size_t device = 0; device != devices_count; ++device)
        devices[device] = static_cast<int>(residencies[device].device);
    using executor_t = typename punned_basic_index<traits_type_, residency_>::executor_t;
    std::size_t no_limit = 0;
    auto [built, status] = on_default_streams(driver_of(gpu_drivers(), std::type_identity<executor_t> {}),
                                              std::span<int const> {devices.data(), devices_count}, gpu_runtime(),
                                              no_limit);
    if (failed(status)) return propagate_error(status, outcome);
    call_executor = std::move(built);
    return true;
}

#pragma endregion Variant Tables

#pragma region Sequence Traits

/** The checkpoint token of one sequence metric, spelled as the SafeGraphs registry spells it;
 *  empty for the metrics the engine refuses. */
inline std::string_view token_of(usearch_sequence_metric_t metric) noexcept {
    switch (metric) {
    case usearch_sequence_metric_protein_alignment_k: return "protein-alignment";
    case usearch_sequence_metric_dna_overlap_k: return "dna-overlap";
    case usearch_sequence_metric_utf8_containment_k: return "utf8-containment";
    default: return {};
    }
}

/** The sequence metric a checkpoint token names, or @c unknown_k. */
inline usearch_sequence_metric_t sequence_metric_of(std::string_view token) noexcept {
    for (usearch_sequence_metric_t metric :
         {usearch_sequence_metric_protein_alignment_k, usearch_sequence_metric_dna_overlap_k,
          usearch_sequence_metric_utf8_containment_k})
        if (token == token_of(metric)) return metric;
    return usearch_sequence_metric_unknown_k;
}

/** The metric, recorded costs, connectivity and sketch bytes a sequence handle freezes, as an init
 *  or a checkpoint states them; every variant holds one. */
struct sequence_geometry_t {
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;

    /** The folded substitution costs a checkpoint recorded, one hexadecimal digit per entry row by
     *  row, which a freeze unfolds; empty for a fresh index, whose costs a freeze folds. */
    util::growing_buffer<char, util::forkunion_allocator_t> substitution_costs {
        util::forkunion_allocator_t {util::no_memory_domain_k}};

    /** The gap cost a checkpoint recorded beside its substitution costs. */
    std::size_t gap_cost = 0;

    /** The class code of every byte a checkpoint recorded beside its substitution costs, one digit
     *  of base 32 per byte, which a freeze unfolds. */
    util::growing_buffer<char, util::forkunion_allocator_t> byte_classes {
        util::forkunion_allocator_t {util::no_memory_domain_k}};

    /** Neighbors per node the insertions select; zero until frozen, which takes the default. */
    std::size_t connectivity = 0;

    /** Bytes of the sketch in front of each stored sequence; zero until frozen, which takes the
     *  metric's default. */
    std::size_t sketch_bytes = 0;
};

/** Whether @p sequences holds @p count byte strings; false once @p outcome records why not. */
inline bool accepts_sequences(usearch_tape_view_t const& sequences, std::size_t count,
                              usearch_outcome_t* outcome) noexcept {
    if (!known_layout(sequences.layout))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "Unknown width of a tape's offsets");
    if (!sequences.data || !monotone_offsets(sequences, count))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The sequences need data and non-decreasing offsets");
    if (sequences.scalar != usearch_scalar_u8_k)
        return propagate_error(usearch_status_unsupported_k, outcome, "Sequences arrive as u8 bytes");
    return true;
}

/** The sequence variants one class unit compiles in each residency, which
 *  @c sequence_specializations.cuh lists. */
template <vendored_residency_t residency_, auto class_>
struct sequence_variants_of {
    using type = type_list<>;
};

/** The fold of the newest CPU class this machine runs, picked once on the CPU ladder for a
 *  variant whose own class is a GPU's; defined in the dispatch unit. */
fold_texts_t host_fold_texts() noexcept;

/** What the sequence verbs differ in: protein alignment, DNA overlap and UTF-8 containment over
 *  u8 strings. */
struct sequence_traits_t {
    using geometry_t = sequence_geometry_t;
    using description_t = usearch_sequence_index_description_t;
    using key_stats_t = usearch_sequence_key_stats_t;

    /** The strings an insertion writes and a search ranks. */
    using values_t = caller_tape_t;
    using queries_t = counted_tape_t;

    /** No radius walk serves strings. */
    using within_queries_t = std::monostate;

    /** Where a get writes its strings. */
    using get_outputs_t = usearch_tape_view_t*;

    /** What a variant adopts past its geometry: the costs whose similarities it folds, or null
     *  for the metric's own. */
    using geometry_extras_t = usearch_sequence_costs_t const*;

    /** The variants each class unit compiles per residency. */
    template <vendored_residency_t residency_, auto class_>
    using variants = typename sequence_variants_of<residency_, class_>::type;

    /** The GPU classes with a unit of sequence kits on each vendor, newest first: the NVIDIA and
     *  Apple floors. */
    using nvidia_gpu_ladder_t = basic_gpu_ladder<gpu_ladder<gpu_class_t::nvidia_ampere_k>>;

    /** Every AMD class, each with a unit of its own over its wavefront. */
    using amd_gpu_ladder_t = amd_ladder_t;
    using apple_gpu_ladder_t = gpu_ladder<gpu_class_t::apple_metal7_k>;

    /** Takes the metric, the connectivity and the sketch bytes an init states. */
    template <typename arguments_type_>
    static bool state(arguments_type_ const& arguments, geometry_t& geometry, usearch_outcome_t*) noexcept {
        geometry.metric = arguments.metric;
        geometry.connectivity = arguments.connectivity;
        geometry.sketch_bytes = arguments.sketch_bytes;
        return true;
    }

    /**
     *  @brief Reads the metric, costs, byte classes, connectivity and sketch bytes @p checkpoint
     *      recorded, at a path or in a buffer, the costs and classes kept for a freeze.
     *
     *  @par Statuses:
     *  - @c malformed_k A field recorded twice, costs beside a containment, or an alignment or an
     *    overlap recording none.
     */
    template <typename checkpoint_type_>
    static util::status_t recall(checkpoint_type_ checkpoint, geometry_t& geometry) noexcept {
        util::status_t fields_status = util::status_code_t::success_k;
        auto per_field = [&](util::safegraphs_custom_field_t field) noexcept {
            // The first field that fails decides, and one met twice would overrun its buffer.
            if (failed(fields_status)) return;
            if (field.key == "connectivity") geometry.connectivity = decimal_in(field.value);
            else if (field.key == "sketch_bytes") geometry.sketch_bytes = decimal_in(field.value);
            else if (field.key == "gap_cost") geometry.gap_cost = decimal_in(field.value);
            else if (field.key == "substitution_costs") {
                fields_status = geometry.substitution_costs.empty()
                                    ? geometry.substitution_costs.reserve_exactly(field.value.size())
                                    : util::status_t {util::status_code_t::malformed_k, "Costs recorded twice"};
                for (char const digit : field.value)
                    if (succeeded(fields_status)) geometry.substitution_costs.push_back_reserved(digit);
            }
            else if (field.key == "byte_classes") {
                fields_status = geometry.byte_classes.empty()
                                    ? geometry.byte_classes.reserve_exactly(field.value.size())
                                    : util::status_t {util::status_code_t::malformed_k, "Byte classes recorded twice"};
                for (char const digit : field.value)
                    if (succeeded(fields_status)) geometry.byte_classes.push_back_reserved(digit);
            }
        };
        auto const [metadata, status] = util::safegraphs_read_index_metadata(
            checkpoint, std::allocator<util::byte_t> {}, per_field);
        geometry.metric = sequence_metric_of(metadata.metric.view());
        if (failed(status)) return status;
        if (failed(fields_status)) return fields_status;
        bool const costed = !geometry.substitution_costs.empty();
        if (geometry.metric != usearch_sequence_metric_unknown_k &&
            costed == (geometry.metric == usearch_sequence_metric_utf8_containment_k))
            return {util::status_code_t::malformed_k, "Costs recorded for a containment or missing for an alignment"};
        return util::status_code_t::success_k;
    }

    /** Refuses the metrics no kit serves, sketch bytes off a multiple of four or for a DNA overlap,
     *  which keeps no sketch, costs naming no similarities, and costs for a containment, which
     *  compares folded bytes. */
    template <typename arguments_type_>
    static bool admits(geometry_t const& geometry, arguments_type_ const& arguments,
                       usearch_outcome_t* outcome) noexcept {
        if (token_of(geometry.metric).empty())
            return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown sequence metric");
        if (geometry.sketch_bytes % 4)
            return propagate_error(usearch_status_config_mismatch_k, outcome, "Sketch bytes come in fours");
        if (geometry.sketch_bytes && geometry.metric == usearch_sequence_metric_dna_overlap_k)
            return propagate_error(usearch_status_config_mismatch_k, outcome, "A DNA overlap keeps no sketch");
        geometry_extras_t const costs = geometry_extras_of(arguments);
        if (!costs) return true;
        if (!costs->similarities)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "Costs need similarities");
        if (geometry.metric == usearch_sequence_metric_utf8_containment_k)
            return propagate_error(usearch_status_config_mismatch_k, outcome, "A containment takes no costs");
        return true;
    }

    /** The costs an init names, whose similarities a variant folds; loads and views unfold the
     *  checkpoint's. */
    static geometry_extras_t geometry_extras_of(usearch_sequence_init_ram_t const& arguments) noexcept {
        return arguments.costs;
    }
    static geometry_extras_t geometry_extras_of(usearch_sequence_init_vram_t const& arguments) noexcept {
        return arguments.costs;
    }
    template <typename arguments_type_>
    static geometry_extras_t geometry_extras_of(arguments_type_ const&) noexcept {
        return nullptr;
    }

    /** Classes the costs @p geometry recalled span, the side of their square; zero for none. */
    static std::size_t classes_recorded(geometry_t const& geometry) noexcept {
        std::size_t classes = 0;
        while (classes * classes < geometry.substitution_costs.size()) ++classes;
        return classes;
    }

    static usearch_sequence_metadata_t metadata_of(geometry_t const& geometry) noexcept {
        return {geometry.metric, geometry.connectivity, classes_recorded(geometry), geometry.sketch_bytes};
    }

    static bool accepts(usearch_sequence_insert_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_sequences(arguments.sequences, arguments.count, outcome);
    }

    static bool accepts(usearch_sequence_search_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_sequences(arguments.queries, arguments.count, outcome);
    }

    static values_t values_of(usearch_sequence_insert_t const& arguments) noexcept {
        return {arguments.sequences, arguments.count};
    }
    static util::expected<queries_t> queries_of(usearch_sequence_search_t const& arguments,
                                                util::growing_arena<util::forkunion_allocator_t>& pages) noexcept {
        return counted_tape_of(arguments.queries, arguments.count, pages);
    }
    static reserve_capacity_t capacity_of(usearch_sequence_reserve_t const& arguments) noexcept {
        return {arguments.entries, arguments.payload_bytes, 0};
    }
    static get_outputs_t get_outputs_of(usearch_sequence_get_t const& arguments) noexcept {
        return arguments.sequences;
    }

    /** Where a get hands its strings out. */
    template <typename arguments_type_>
    static usearch_tape_view_t* output_of(arguments_type_ const& arguments) noexcept {
        return arguments.sequences;
    }
};

#pragma endregion Sequence Traits

#pragma region Shapes Traits

/** The checkpoint token of one shapes kind, lowercase and dash-separated like every metric name. */
inline std::string_view token_of(usearch_shapes_metric_t metric) noexcept {
    switch (metric) {
    case usearch_shapes_metric_chamfer_k: return "chamfer";
    case usearch_shapes_metric_hausdorff_k: return "hausdorff";
    case usearch_shapes_metric_gromov_wasserstein_k: return "gromov-wasserstein";
    case usearch_shapes_metric_wasserstein_procrustes_k: return "wasserstein-procrustes";
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

/** Whether @p shapes holds @p count entries of points in a scalar the variants cast from, its
 *  offsets counting points; false once @p outcome records why not. */
inline bool accepts_shapes(usearch_tape_view_t const& shapes, std::size_t count, usearch_outcome_t* outcome) noexcept {
    if (!known_layout(shapes.layout))
        return propagate_error(usearch_status_unsupported_layout_k, outcome, "Unknown width of a tape's offsets");
    if (!shapes.data || !monotone_offsets(shapes, count))
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "The shapes need data and non-decreasing offsets");
    nk_dtype_t const dtype = nk_dtype_of(shapes.scalar);
    if (dtype == nk_dtype_unknown_k || nk_dtype_bits(dtype) % 8)
        return propagate_error(usearch_status_unsupported_k, outcome, "Points arrive in a whole-byte scalar");
    return true;
}

/** Where a get writes its clouds, and the scalar it exports their points in. */
struct shapes_get_outputs_t {
    usearch_tape_view_t* shapes = nullptr;
    usearch_scalar_t scalar = usearch_scalar_unknown_k;
};

/** The shapes variants one class unit compiles in each residency, which
 *  @c shapes_specializations.cuh lists. */
template <vendored_residency_t residency_, auto class_>
struct shapes_variants_of {
    using type = type_list<>;
};

/** What the shapes verbs differ in: clouds of 3D points stored in one scalar, crossing in any. */
struct shapes_traits_t {
    using geometry_t = shapes_geometry_t;
    using description_t = usearch_shapes_index_description_t;
    using key_stats_t = usearch_shapes_key_stats_t;

    /** The clouds an insertion writes and a search ranks. */
    using values_t = caller_tape_t;
    using queries_t = counted_tape_t;

    /** No radius walk serves clouds. */
    using within_queries_t = std::monostate;
    using get_outputs_t = shapes_get_outputs_t;

    /** A variant adopts nothing past its geometry. */
    using geometry_extras_t = std::monostate;

    /** The variants each class unit compiles per residency. */
    template <vendored_residency_t residency_, auto class_>
    using variants = typename shapes_variants_of<residency_, class_>::type;

    /** The GPU classes with a unit of shapes kits on each vendor, newest first: on NVIDIA,
     *  datacenter Blackwell's packed f16 kit, then the Ampere floor; on Apple, the floor alone. */
    using nvidia_gpu_ladder_t =
        basic_gpu_ladder<gpu_ladder<gpu_class_t::nvidia_blackwell_k, gpu_class_t::nvidia_ampere_k>>;

    /** Every AMD class, each with a unit of its own over its wavefront. */
    using amd_gpu_ladder_t = amd_ladder_t;
    using apple_gpu_ladder_t = gpu_ladder<gpu_class_t::apple_metal7_k>;

    template <typename arguments_type_>
    static bool state(arguments_type_ const& arguments, geometry_t& geometry, usearch_outcome_t*) noexcept {
        geometry = {arguments.metric, arguments.quantization, arguments.points, arguments.connectivity};
        return true;
    }

    /** Reads the kind, scalar, point count and connectivity @p checkpoint recorded, at a path or in
     *  a buffer. */
    template <typename checkpoint_type_>
    static util::status_t recall(checkpoint_type_ checkpoint, geometry_t& geometry) noexcept {
        auto per_field = [&](util::safegraphs_custom_field_t field) noexcept {
            if (field.key == "connectivity") geometry.connectivity = decimal_in(field.value);
            else if (field.key == "points") geometry.points = decimal_in(field.value);
        };
        auto const [metadata, status] = util::safegraphs_read_index_metadata(
            checkpoint, std::allocator<util::byte_t> {}, per_field);
        geometry.metric = shapes_metric_of(metadata.metric.view());
        geometry.quantization = metadata.value_dtype == util::dtype_t::f16_k   ? usearch_scalar_f16_k
                                : metadata.value_dtype == util::dtype_t::f32_k ? usearch_scalar_f32_k
                                                                               : usearch_scalar_unknown_k;
        return status;
    }

    /** Refuses the kinds and scalars no CPU variant serves. */
    template <typename arguments_type_>
    static bool admits(geometry_t const& geometry, arguments_type_ const&, usearch_outcome_t* outcome) noexcept {
        if (token_of(geometry.metric).empty())
            return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown shapes metric");
        if (geometry.quantization != usearch_scalar_f16_k && geometry.quantization != usearch_scalar_f32_k)
            return propagate_error(usearch_status_unsupported_k, outcome, "Shapes store f16 or f32 points");
        return true;
    }

    /** Refuses the kinds and scalars no device kit serves, for a call placing on a device. */
    template <typename arguments_type_>
        requires requires(arguments_type_ const& arguments) { arguments.vram; }
    static bool admits(geometry_t const& geometry, arguments_type_ const&, usearch_outcome_t* outcome) noexcept {
        switch (geometry.metric) {
        case usearch_shapes_metric_chamfer_k:
        case usearch_shapes_metric_hausdorff_k: break;
        case usearch_shapes_metric_gromov_wasserstein_k:
        case usearch_shapes_metric_wasserstein_procrustes_k:
            return propagate_error(usearch_status_unsupported_k, outcome, "Only Chamfer and Hausdorff serve a GPU yet");
        default: return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown shapes metric");
        }
        if (geometry.quantization != usearch_scalar_f16_k && geometry.quantization != usearch_scalar_f32_k)
            return propagate_error(usearch_status_unsupported_k, outcome, "Shapes store f16 or f32 points");
        return true;
    }

    /** Nothing an init, a load or a view names past the geometry reaches a variant. */
    template <typename arguments_type_>
    static geometry_extras_t geometry_extras_of(arguments_type_ const&) noexcept {
        return {};
    }

    static usearch_shapes_metadata_t metadata_of(geometry_t const& geometry) noexcept {
        return {geometry.metric, geometry.quantization, geometry.points, geometry.connectivity};
    }

    static bool accepts(usearch_shapes_insert_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_shapes(arguments.shapes, arguments.count, outcome);
    }

    static bool accepts(usearch_shapes_search_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_shapes(arguments.queries, arguments.count, outcome);
    }

    static values_t values_of(usearch_shapes_insert_t const& arguments) noexcept {
        return {arguments.shapes, arguments.count};
    }
    static util::expected<queries_t> queries_of(usearch_shapes_search_t const& arguments,
                                                util::growing_arena<util::forkunion_allocator_t>& pages) noexcept {
        return counted_tape_of(arguments.queries, arguments.count, pages);
    }
    static reserve_capacity_t capacity_of(usearch_shapes_reserve_t const& arguments) noexcept {
        return {arguments.entries, 0, arguments.points};
    }
    static get_outputs_t get_outputs_of(usearch_shapes_get_t const& arguments) noexcept {
        return {arguments.shapes, arguments.outputs_scalar};
    }

    /** Whether the points a get asks for export to a whole-byte scalar; false once
     *  @p outcome records why not. */
    template <typename arguments_type_>
    static bool exports(arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
        nk_dtype_t const exported = nk_dtype_of(arguments.outputs_scalar);
        if (!arguments.shapes || (exported != nk_dtype_unknown_k && nk_dtype_bits(exported) % 8 == 0)) return true;
        return propagate_error(usearch_status_unsupported_k, outcome, "Points export to a whole-byte scalar");
    }

    /** Where a get hands its clouds out. */
    template <typename arguments_type_>
    static usearch_tape_view_t* output_of(arguments_type_ const& arguments) noexcept {
        return arguments.shapes;
    }
};

#pragma endregion Shapes Traits

#pragma region Geospatial Traits

/** The connectivity a geospatial handle freezes, as an init or a checkpoint states it; every
 *  variant holds one. */
struct geospatial_geometry_t {

    /** Neighbors per node: a winner per cone and long links for the rest, rounded up to a power of
     *  two on a GPU; zero until frozen, which takes the default. */
    std::size_t connectivity = 0;
};

/** The checkpoint token of the one geospatial kind, the squared chord. */
inline std::string_view token_of(geospatial_geometry_t const&) noexcept { return "chord"; }

/** Whether @p terrain names exactly one source and none a handle cannot serve yet; false once
 *  @p outcome records why not. */
inline bool accepts_terrain(usearch_geospatial_terrain_t const& terrain, usearch_outcome_t* outcome) noexcept {
    if (terrain.path.data)
        return propagate_error(usearch_status_unsupported_k, outcome, "No residency reads a terrain dataset yet");
    if (terrain.heights && terrain.callback)
        return propagate_error(usearch_status_invalid_argument_k, outcome,
                               "A terrain is either read in place or filled, never both");
    return true;
}

/** Whether @p observations names storage for @p count entries, each with a position and an interval
 *  in order; false once @p outcome records why not. */
inline bool accepts_observations(usearch_geospatial_observation_t const* observations, std::size_t count,
                                 usearch_outcome_t* outcome) noexcept {
    if (!observations)
        return propagate_error(usearch_status_invalid_argument_k, outcome, "The observations must not be null");
    for (std::size_t offset = 0; offset != count; ++offset) {
        usearch_geospatial_observation_t const& observation = observations[offset];
        if (std::isnan(observation.latitude_degrees) || std::isnan(observation.longitude_degrees))
            return propagate_error(usearch_status_invalid_argument_k, outcome,
                                   "An observation needs a latitude and a longitude");
        if (observation.end_seconds < observation.begin_seconds)
            return propagate_error(usearch_status_invalid_argument_k, outcome, "An observation ends before it begins");
    }
    return true;
}

/** The caller's observations, as an insertion writes them or a query batch asks with them. */
using observations_view_t = std::span<usearch_geospatial_observation_t const>;

/** What a geospatial search or containment walk asks of each query observation: the relation and
 *  the tolerance it admits entries under. */
struct geospatial_queries_t {
    observations_view_t observations;
    usearch_geospatial_relation_t relation = usearch_geospatial_relation_near_k;
    usearch_geospatial_tolerance_t tolerance {};
};

/** Whether @p relation is one the geospatial kind knows; false once @p outcome records why not. */
inline bool accepts_relation(usearch_geospatial_relation_t relation, usearch_outcome_t* outcome) noexcept {
    if (relation == usearch_geospatial_relation_near_k || relation == usearch_geospatial_relation_visible_k ||
        relation == usearch_geospatial_relation_sees_k || relation == usearch_geospatial_relation_seen_k)
        return true;
    return propagate_error(usearch_status_unknown_kind_k, outcome, "Unknown geospatial relation");
}

/** The geospatial variants one class unit compiles in each residency, which
 *  @c geospatial_specializations.cuh lists. */
template <vendored_residency_t residency_, auto class_>
struct geospatial_variants_of {
    using type = type_list<>;
};

/** What the geospatial verbs differ in: observations in time over a terrain, under one kind. */
struct geospatial_traits_t {
    using geometry_t = geospatial_geometry_t;
    using description_t = usearch_geospatial_index_description_t;
    using key_stats_t = usearch_geospatial_key_stats_t;
    using values_t = observations_view_t;
    using queries_t = geospatial_queries_t;
    using within_queries_t = geospatial_queries_t;

    /** Where a get writes its observations. */
    using get_outputs_t = usearch_geospatial_observation_t const**;

    /** What a variant adopts past its geometry: the terrain. */
    using geometry_extras_t = usearch_geospatial_terrain_t;

    /** The variants each class unit compiles per residency. */
    template <vendored_residency_t residency_, auto class_>
    using variants = typename geospatial_variants_of<residency_, class_>::type;

    /** The GPU classes with a unit of the geospatial kit on each vendor, newest first: none on
     *  Apple, whose GPUs hold no f64 to the contract the kit keeps. */
    using nvidia_gpu_ladder_t = basic_gpu_ladder<gpu_ladder<gpu_class_t::nvidia_ampere_k>>;

    /** Every AMD class, each with a unit of its own over its wavefront. */
    using amd_gpu_ladder_t = amd_ladder_t;
    using apple_gpu_ladder_t = gpu_ladder<>;

    template <typename arguments_type_>
    static bool state(arguments_type_ const& arguments, geometry_t& geometry, usearch_outcome_t*) noexcept {
        geometry.connectivity = arguments.connectivity;
        return true;
    }

    /** Reads the connectivity @p checkpoint recorded, at a path or in a buffer, the one field
     *  beside its observations. */
    template <typename checkpoint_type_>
    static util::status_t recall(checkpoint_type_ checkpoint, geometry_t& geometry) noexcept {
        auto per_field = [&](util::safegraphs_custom_field_t field) noexcept {
            if (field.key == "connectivity") geometry.connectivity = decimal_in(field.value);
        };
        auto const [metadata, status] = util::safegraphs_read_index_metadata(
            checkpoint, std::allocator<util::byte_t> {}, per_field);
        return status;
    }

    template <typename arguments_type_>
    static bool admits(geometry_t const&, arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_terrain(arguments.terrain, outcome);
    }

    /** Refuses a terrain a device cannot read: one filled by a host callback. */
    template <typename arguments_type_>
        requires requires(arguments_type_ const& arguments) { arguments.vram; }
    static bool admits(geometry_t const&, arguments_type_ const& arguments, usearch_outcome_t* outcome) noexcept {
        if (!accepts_terrain(arguments.terrain, outcome)) return false;
        if (!arguments.terrain.callback) return true;
        return propagate_error(usearch_status_unsupported_k, outcome,
                               "A GPU terrain needs its heights, not a callback");
    }

    /** The terrain an init, a load or a view names, which a variant adopts. */
    template <typename arguments_type_>
    static geometry_extras_t geometry_extras_of(arguments_type_ const& arguments) noexcept {
        return arguments.terrain;
    }

    static usearch_geospatial_metadata_t metadata_of(geometry_t const& geometry) noexcept {
        return {geometry.connectivity};
    }

    static bool accepts(usearch_geospatial_insert_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_observations(arguments.observations, arguments.count, outcome);
    }

    /** Whether a containment walk names a positive radius and, when it has queries, observations
     *  each with a position and an interval in order, under a relation the kind knows. */
    static bool accepts(usearch_geospatial_within_t const& arguments, usearch_outcome_t* outcome) noexcept {
        if (!(arguments.tolerance.metres > 0))
            return propagate_error(usearch_status_invalid_argument_k, outcome,
                                   "A containment walk needs a positive radius in metres");
        return accepts_relation(arguments.relation, outcome) &&
               (!arguments.count || accepts_observations(arguments.queries, arguments.count, outcome));
    }

    static bool accepts(usearch_geospatial_search_t const& arguments, usearch_outcome_t* outcome) noexcept {
        return accepts_observations(arguments.queries, arguments.count, outcome) &&
               accepts_relation(arguments.relation, outcome);
    }

    static values_t values_of(usearch_geospatial_insert_t const& arguments) noexcept {
        return {arguments.observations, arguments.count};
    }
    template <typename arguments_type_>
    static util::expected<queries_t> queries_of(arguments_type_ const& arguments,
                                                util::growing_arena<util::forkunion_allocator_t>&) noexcept {
        return {{{arguments.queries, arguments.count}, arguments.relation, arguments.tolerance},
                util::status_code_t::success_k};
    }
    static reserve_capacity_t capacity_of(usearch_geospatial_reserve_t const& arguments) noexcept {
        return {arguments.entries, 0, 0};
    }
    static get_outputs_t get_outputs_of(usearch_geospatial_get_t const& arguments) noexcept {
        return arguments.observations;
    }

    /** Where a get hands its observations out. */
    template <typename arguments_type_>
    static usearch_geospatial_observation_t const** output_of(arguments_type_ const& arguments) noexcept {
        return arguments.observations;
    }
};

#pragma endregion Geospatial Traits
