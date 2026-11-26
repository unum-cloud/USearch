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

#include <cstring> // `std::memcmp` - a source's costs

#include <array>       // `std::array` - the checkpoint's recorded fields
#include <memory>      // `std::unique_ptr` - a GPU shard built on its device
#include <new>         // `std::nothrow` - a GPU shard built on its device
#include <span>        // `std::span` - containment's window widths
#include <string_view> // `std::string_view` - entries, tokens and paths
#include <type_traits> // `std::type_identity`, `std::conditional_t` - the kit and the metric a family scores with
#include <variant>     // `std::monostate` - no radius walk

#include <forkunion/atomics.hpp> // `fu::standard_atomic_ref` - the scan totals

#include "usearch/sequence/ram_index.hpp"
#if USEARCH_ARCH_CUDA_
#include "usearch/sequence/vram_nvidia_metric.cuh"
#endif
#if USEARCH_ARCH_METAL_
#include "usearch/sequence/vram_apple_index.hpp"
#endif
#if USEARCH_ARCH_ROCM_
#include "usearch/sequence/vram_amd_metric.hip"
#endif

#include "sequence_kits.hpp"         // `sequence_vram_kit`, `sequence_nvidia_fatbin`, `sequence_amd_code_object`
#include "basic_specializations.cuh" // the basic variants, `lib.hpp` under them

#pragma region Sequence Variants

/** A containment and a banded overlap distance are fixed point over @c share_resolution_k, and a
 *  metric reading the sequence past a stored sketch scores in the unit of the one it wraps. */
template <typename kit_type_, typename allocator_type_, typename cache_hints_type_>
inline constexpr usearch_distance_t
    distance_unit<us::sequence::containment_metric<kit_type_, allocator_type_, cache_hints_type_>> =
        1.0 / us::sequence::share_resolution_k;
template <typename kit_type_, typename allocator_type_, typename cache_hints_type_>
inline constexpr usearch_distance_t
    distance_unit<us::sequence::banded_overlap_metric<kit_type_, allocator_type_, cache_hints_type_>> =
        1.0 / us::sequence::share_resolution_k;
template <typename metric_type_, typename value_type_>
inline constexpr usearch_distance_t distance_unit<us::sequence::stored_canonical_metric<metric_type_, value_type_>> =
    distance_unit<metric_type_>;

/** The tables of a smooth sketch every sequence variant owns, derived for an alignment alone. */
using sequence_spectrum_t = us::sequence::spectrum_tables<util::forkunion_allocator_t>;

/** The tables the sketch of a kit reading @p tables_type_ takes: the variant's @p spectrum for a
 *  smooth sketch, and none otherwise. */
inline us::sequence::spectrum_view_t sketch_tables_for(std::type_identity<us::sequence::spectrum_view_t>,
                                                       sequence_spectrum_t const& spectrum) noexcept {
    return spectrum.view();
}
inline std::monostate sketch_tables_for(std::type_identity<std::monostate>, sequence_spectrum_t const&) noexcept {
    return {};
}

/** Folds the case of @p text into @p folded, three bytes a source byte long, by the fold
 *  @p kit_type_ binds; answers the bytes it wrote. */
template <typename kit_type_>
std::size_t fold_text_into(std::string_view text, char* folded) noexcept {
    sz_size_t length = 0;
    [[maybe_unused]] sz_status_t const status = kit_type_::canonical_fold(text.data(), text.size(), folded, &length,
                                                                          nullptr);
    assert(status == sz_success_k);
    return length;
}

/**
 *  @brief Folds the case of every text of @p texts through the fold @p kit_type_ binds, back to
 *      back in @p memory, on @p threads where there are any and on the caller otherwise.
 *
 *  Folding triples a text at most, so every text folds at three times its source offset, where no
 *  two overlap and only the pages a fold touches cost memory; the gaps then close in order, each
 *  text moving towards the front. Malformed bytes pass through.
 */
template <typename kit_type_>
util::expected<counted_tape_t> fold_texts_with(counted_tape_t texts,
                                               util::growing_arena<util::forkunion_allocator_t>& memory,
                                               util::ram_executor_t* threads) noexcept {
    std::size_t const count = texts.count(), first = texts.offsets[0];
    std::size_t const source_bytes = texts.offsets[count] - first;
    std::size_t* const offsets = carve<std::size_t>(memory, count + 1);
    char* const folded = carve<char>(memory, source_bytes * 3);
    if (!offsets || (source_bytes && !folded)) return {{}, util::status_code_t::out_of_memory_k};
    auto fold_slice = [&](std::size_t slice, std::size_t slices) noexcept {
        for (std::size_t entry = count * slice / slices; entry != count * (slice + 1) / slices; ++entry) {
            offsets[entry + 1] = fold_text_into<kit_type_>(chars_at(texts, entry),
                                                           folded + 3 * (texts.offsets[entry] - first));
        }
    };
    if (!threads) fold_slice(0, 1);
    else if (util::status_t const forked = threads->for_threads(
                 [&](std::size_t thread, std::size_t) noexcept { fold_slice(thread, threads->size()); });
             failed(forked))
        return {{}, forked};
    offsets[0] = 0;
    for (std::size_t entry = 0; entry != count; ++entry) {
        std::size_t const length = offsets[entry + 1];
        std::memmove(folded + offsets[entry], folded + 3 * (texts.offsets[entry] - first), length);
        offsets[entry + 1] = offsets[entry] + length;
    }
    return {{folded, {offsets, count + 1}, usearch_scalar_u8_k}, util::status_code_t::success_k};
}

/** The tape a variant of @p metric stores or scores for the caller's @p texts : folded by @p fold
 *  for a containment, and the tape itself otherwise. */
inline util::expected<counted_tape_t> folded_tape_in(usearch_sequence_metric_t metric, fold_texts_t fold,
                                                     counted_tape_t texts,
                                                     util::growing_arena<util::forkunion_allocator_t>& memory,
                                                     util::ram_executor_t* threads) noexcept {
    if (metric != usearch_sequence_metric_utf8_containment_k) return {texts, util::status_code_t::success_k};
    if (!fold) return {{}, util::status_code_t::unsupported_k};
    return fold(texts, memory, threads);
}

/** The tape an insertion of @p metric lowers for the caller's @p texts : lowered and folded on
 *  @p memory for a containment, and the tape itself otherwise. */
inline util::expected<caller_tape_t> folded_caller_tape_in(usearch_sequence_metric_t metric, fold_texts_t fold,
                                                           caller_tape_t texts,
                                                           util::growing_arena<util::forkunion_allocator_t>& memory,
                                                           util::ram_executor_t* threads) noexcept {
    if (metric != usearch_sequence_metric_utf8_containment_k) return {texts, util::status_code_t::success_k};
    auto const [lowered, lower_status] = counted_tape_of(texts.tape, texts.count, memory);
    if (failed(lower_status)) return {{}, lower_status};
    auto const [folded, folding_status] = folded_tape_in(metric, fold, lowered, memory, threads);
    if (failed(folding_status)) return {{}, folding_status};
    caller_tape_t folded_texts {{}, texts.count};
    write_output_tape(folded.elements, folded.offsets.data(), folded.scalar, folded_texts.tape);
    return {folded_texts, util::status_code_t::success_k};
}

/** A query @p text in the form @p metric stores a sequence in, as every canonical score reads
 *  both sides: its class codes under @p alphabet, packed or a byte apiece, carved from @p memory,
 *  or the folded text itself for a containment. */
inline util::expected<std::string_view> stored_form_in(util::growing_arena<util::forkunion_allocator_t>& memory,
                                                       std::string_view text, usearch_sequence_metric_t metric,
                                                       us::sequence::alignment_alphabet_t const& alphabet) noexcept {
    std::size_t const bytes = stored_sequence_bytes(metric, text.size());
    if (metric == usearch_sequence_metric_utf8_containment_k || !bytes) return {text, util::status_code_t::success_k};
    char* const packed = carve<char>(memory, bytes);
    if (!packed) return {{}, util::status_code_t::out_of_memory_k};
    store_sequence(metric, util::bytes_of(text.data(), text.size()), alphabet, util::bytes_of(packed, bytes));
    return {{packed, bytes}, util::status_code_t::success_k};
}

/** Whether @p kit_type_ binds the case fold of a containment. */
template <typename kit_type_>
concept a_kit_folding_texts = requires { kit_type_::canonical_fold; };

/** A query @p text in the stored form of @p metric, folded by the fold @p kit_type_ binds for a
 *  containment, carved from @p memory. */
template <typename kit_type_>
util::expected<std::string_view> stored_query_in(util::growing_arena<util::forkunion_allocator_t>& memory,
                                                 std::string_view text, usearch_sequence_metric_t metric,
                                                 us::sequence::alignment_alphabet_t const& alphabet) noexcept {
    if constexpr (a_kit_folding_texts<kit_type_>) {
        if (text.empty()) return {text, util::status_code_t::success_k};
        char* const folded = carve<char>(memory, text.size() * 3);
        if (!folded) return {{}, util::status_code_t::out_of_memory_k};
        return {{folded, fold_text_into<kit_type_>(text, folded)}, util::status_code_t::success_k};
    }
    else return stored_form_in(memory, text, metric, alphabet);
}

/** Every query of @p tape in the form @p stored_form_in gives one, offsets and bytes carved from
 *  @p memory; the tape itself for a containment. */
inline util::expected<counted_tape_t> stored_form_tape_in(util::growing_arena<util::forkunion_allocator_t>& memory,
                                                          counted_tape_t const& tape, usearch_sequence_metric_t metric,
                                                          us::sequence::alignment_alphabet_t const& alphabet) noexcept {
    if (metric == usearch_sequence_metric_utf8_containment_k) return {tape, util::status_code_t::success_k};
    std::size_t const count = tape.count();
    std::size_t* const offsets = carve<std::size_t>(memory, count + 1);
    if (!offsets) return {{}, util::status_code_t::out_of_memory_k};
    offsets[0] = 0;
    for (std::size_t entry = 0; entry != count; ++entry)
        offsets[entry + 1] = offsets[entry] + stored_sequence_bytes(metric, chars_at(tape, entry).size());
    util::byte_t* const bytes = carve<util::byte_t>(memory, offsets[count]);
    if (offsets[count] && !bytes) return {{}, util::status_code_t::out_of_memory_k};
    for (std::size_t entry = 0; entry != count; ++entry) {
        std::string_view const text = chars_at(tape, entry);
        store_sequence(metric, util::bytes_of(text.data(), text.size()), alphabet,
                       {bytes + offsets[entry], offsets[entry + 1] - offsets[entry]});
    }
    return {{bytes, {offsets, count + 1}, usearch_scalar_u8_k}, util::status_code_t::success_k};
}

/** Every sequence of @p tape laid out as a device tape stores it: a gap of @p sketch_bytes the
 *  device fills, the sequence in its stored form under @p metric and @p alphabet, and its closing
 *  padding; offsets and bytes carved from @p memory. */
inline util::expected<counted_tape_t> stored_tape_in(util::growing_arena<util::forkunion_allocator_t>& memory,
                                                     counted_tape_t const& tape, usearch_sequence_metric_t metric,
                                                     std::size_t sketch_bytes,
                                                     us::sequence::alignment_alphabet_t const& alphabet) noexcept {
    std::size_t const count = tape.count();
    std::size_t* const offsets = carve<std::size_t>(memory, count + 1);
    if (!offsets) return {{}, util::status_code_t::out_of_memory_k};
    offsets[0] = 0;
    for (std::size_t entry = 0; entry != count; ++entry)
        offsets[entry + 1] = offsets[entry] +
                             us::sequence::vram_stored_bytes(
                                 stored_sequence_bytes(metric, chars_at(tape, entry).size()), sketch_bytes);
    util::byte_t* const bytes = carve<util::byte_t>(memory, offsets[count]);
    if (offsets[count] && !bytes) return {{}, util::status_code_t::out_of_memory_k};
    for (std::size_t entry = 0; entry != count; ++entry) {
        std::string_view const text = chars_at(tape, entry);
        std::size_t const sequence_bytes = stored_sequence_bytes(metric, text.size());
        util::bytes_span_t const stored {bytes + offsets[entry], offsets[entry + 1] - offsets[entry]};
        store_sequence(metric, util::bytes_of(text.data(), text.size()), alphabet,
                       stored.subspan(sketch_bytes, sequence_bytes));
        us::sequence::vram_close_stored(stored, sequence_bytes, sketch_bytes);
    }
    return {{bytes, {offsets, count + 1}, usearch_scalar_u8_k}, util::status_code_t::success_k};
}

/** The factory of the sketch metric of @p kit_type_ over values of @p value_type_, staging over a
 *  thread's arena and prefetching through @p cache_hints_type_, every metric scoring sketches of
 *  @p sketch_bytes and writing them under @p spectrum: the sketches values keep in front, or for a
 *  view's sequences, the sketches it writes as they land. */
template <typename kit_type_, typename value_type_, typename cache_hints_type_>
auto sketch_metric_factory(std::size_t sketch_bytes, sequence_spectrum_t const& spectrum) noexcept {
    using sketch_metric_t = std::conditional_t<
        std::is_same_v<value_type_, std::string_view>,
        us::sequence::fetched_sketch_metric<kit_type_, thread_allocator_t>,
        us::sequence::stored_sketch_metric<kit_type_, value_type_, thread_allocator_t, cache_hints_type_>>;
    using sketch_tables_t = typename kit_type_::sketch_tables_t;
    sketch_tables_t const tables = sketch_tables_for(std::type_identity<sketch_tables_t> {}, spectrum);
    return [sketch_bytes, tables](thread_allocator_t const& allocator, std::size_t staging) noexcept {
        sketch_metric_t made {allocator};
        util::status_t const ready = made.init(staging, sketch_bytes, tables);
        return util::expected<sketch_metric_t> {std::move(made), ready};
    };
}

/** Calls @p callback with the kit the metric of @p geometry picks under @p capability_ and a
 *  factory of the metric @p role_ scores with over values of @p value_type_, an owned one keeping
 *  a sketch of the geometry's bytes in front of its sequence and a view none: a sketching kit's
 *  sketch distance under @p spectrum for a builder, and the canonical metric over the sequence for
 *  a search and for the builder of a kit that writes no sketch, under the folded @p costs, over a
 *  thread's arena and prefetching through @p cache_hints_type_. */
template <util::score_role_t role_, sz_capability_t capability_, typename cache_hints_type_, typename value_type_,
          typename callback_type_>
auto with_sequence_metric(sequence_geometry_t const& geometry, us::sequence::alignment_costs_t const& costs,
                          sequence_spectrum_t const& spectrum, callback_type_&& callback) noexcept {
    using containment_t =
        us::sequence::containment_metric<typename us::sequence::containment_ladder<capability_>::kit_t,
                                         thread_allocator_t, cache_hints_type_>;
    using banded_overlap_t =
        us::sequence::banded_overlap_metric<typename us::sequence::banded_overlap_ladder<capability_>::kit_t,
                                            thread_allocator_t, cache_hints_type_>;
    using alignment_t = us::sequence::alignment_metric<typename us::sequence::alignment_ladder<capability_>::kit_t,
                                                       thread_allocator_t, cache_hints_type_>;
    auto with_kit = [&]<typename metric_type_, typename... arguments_types_>(
                        std::type_identity<metric_type_>, arguments_types_ const&... arguments) noexcept {
        using kit_t = typename metric_type_::kit_t;
        std::size_t const sketch_bytes = geometry.sketch_bytes;
        if constexpr (role_ == util::score_role_t::build_k && us::sequence::a_sketching_kit<kit_t>)
            return callback(std::type_identity<kit_t> {},
                            sketch_metric_factory<kit_t, value_type_, cache_hints_type_>(sketch_bytes, spectrum));
        else if constexpr (std::is_same_v<value_type_, std::string_view>) {
            return callback(std::type_identity<kit_t> {},
                            [&arguments...](thread_allocator_t const& allocator, std::size_t staging) noexcept {
                                metric_type_ made {allocator};
                                util::status_t const ready = made.init(staging, arguments...);
                                return util::expected<metric_type_> {std::move(made), ready};
                            });
        }
        else {
            using canonical_metric_t = us::sequence::stored_canonical_metric<metric_type_, value_type_>;
            return callback(
                std::type_identity<kit_t> {},
                [sketch_bytes, &arguments...](thread_allocator_t const& allocator, std::size_t staging) noexcept {
                    canonical_metric_t made {allocator};
                    util::status_t const ready = made.init(staging, sketch_bytes, arguments...);
                    return util::expected<canonical_metric_t> {std::move(made), ready};
                });
        }
    };
    switch (geometry.metric) {
    case usearch_sequence_metric_utf8_containment_k: return with_kit(std::type_identity<containment_t> {});
    case usearch_sequence_metric_dna_overlap_k: return with_kit(std::type_identity<banded_overlap_t> {}, costs);
    default: return with_kit(std::type_identity<alignment_t> {}, costs);
    }
}

/** Calls @p callback with the kit, the factory of the canonical metric a search of @p geometry
 *  ranks by, as @ref with_sequence_metric hands them, and the factory of the metric its seeding
 *  walk scores with over stored values of @p value_type_: a sketching kit's sketch metric, and
 *  no seeding for a kit that writes no sketch. */
template <sz_capability_t capability_, typename cache_hints_type_, typename value_type_, typename callback_type_>
auto with_seeded_sequence_metric(sequence_geometry_t const& geometry, us::sequence::alignment_costs_t const& costs,
                                 sequence_spectrum_t const& spectrum, callback_type_&& callback) noexcept {
    return with_sequence_metric<util::score_role_t::search_k, capability_, cache_hints_type_, value_type_>(
        geometry, costs, spectrum,
        [&]<typename kit_type_, typename make_metric_type_>(std::type_identity<kit_type_> kit,
                                                            make_metric_type_ make_metric) noexcept {
            if constexpr (us::sequence::a_sketching_kit<kit_type_>)
                return callback(
                    kit, make_metric,
                    sketch_metric_factory<kit_type_, value_type_, cache_hints_type_>(geometry.sketch_bytes, spectrum));
            else return callback(kit, make_metric, util::unseeded_factory_t {});
        });
}

/** One sequence alternative: the basic shards over owned strings, the score they rank by, and the
 *  kits @p cpu_class_ picks. Its verbs are defined below the class, so an explicit instantiation in
 *  the class's unit compiles them with that class's flags and an @c extern one keeps them out of
 *  the dispatch unit. Checkpoints and refinements serve an index of one shard. */
template <cpu_class_t cpu_class_>
struct sequence_ram_variant {
    using executor_t = util::ram_executor_t;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    using shard_t = us::sequence::ram_index<util::uuid128_t, util::u32_t, waiter_of<cpu_class_>,
                                            atomic_ref_for_cpu_class<cpu_class_>::template type,
                                            util::forkunion_allocator_t, cache_hints_t>;
    using index_t = basic::ram_indexes<shard_t>;
    using value_t = typename shard_t::value_t;
    using refine_plan_t = typename index_t::refine_plan_t;
    using reshard_plan_t = typename index_t::reshard_plan_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr sz_capability_t capability_k = stringzilla_capabilities_of(cpu_class_);

    index_t index;
    sequence_geometry_t geometry;

    /** The costs an alignment or an overlap scores with, folded at freeze. */
    us::sequence::alignment_costs_t costs {};

    /** The class code of every byte an alignment or an overlap stores, folded with @c costs. */
    us::sequence::alignment_alphabet_t alphabet {};

    /** The tables a smooth sketch reads, derived from @c costs at the geometry's sketch bytes when
     *  the index is created or loaded, and never recorded. */
    sequence_spectrum_t spectrum {util::forkunion_allocator_t {util::no_memory_domain_k}};

    /** The case folding of the class, for the variants without a class of their own. */
    static constexpr fold_texts_t fold_texts =
        &fold_texts_with<sequence_kit<usearch_sequence_metric_utf8_containment_k, capability_k>>;

    explicit sequence_ram_variant(util::forkunion_allocator_t const& allocator) noexcept
        : index(allocator, machine_cache_line_bytes()) {}

    util::status_t reserve(reserve_capacity_t capacity) noexcept { return index.reserve(capacity.entries).status; }
    util::status_t clear() noexcept { return index.clear(); }
    util::expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t sequences, insert_config_t config,
                                             executor_t& executor, progress_ticks_t& ticks) noexcept;
    util::expected<traversal_stats_t> search(counted_tape_t queries, usearch_matches_view_t const* hints,
                                             matches_outputs_t& outputs, search_config_t config,
                                             predicate_t const& predicate, executor_t& executor,
                                             progress_ticks_t& ticks) noexcept;

    /** Calls @p callback with the codec of the kit the geometry's metric picks, over the variant's
     *  sketch tables. */
    template <typename callback_type_>
    auto with_codec(callback_type_&& callback) const noexcept;

    /** Opens one round of every live shard in place under the config's cache, which weighs every
     *  value whole, its sketch included. */
    util::expected<refine_plan_t> plan_refine(refine_config_t config) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_refine({config.cache_bytes, config.expansion_factor},
                                 basic::keep_diverse_t {geometry.connectivity}, us::sequence::codec_t {}, planning);
    }

    /** Opens a reshard of @p sources into @p targets driven by this index, the one target of a
     *  pack or the one source of a split, a split's part roots drawn from @p seed, then one round
     *  per shard that took entries. */
    util::expected<reshard_plan_t> plan_reshard(std::span<index_t* const> sources, std::span<index_t* const> targets,
                                                refine_config_t config, util::u64_t seed) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        return index.plan_reshard(sources, targets, {{config.cache_bytes, config.expansion_factor}, 0, seed},
                                  basic::keep_diverse_t {geometry.connectivity}, us::sequence::codec_t {}, planning);
    }

    /** Drives @p plan 's pending phases on the executor's threads until its round ends or @p ticks
     *  cancels. */
    template <typename plan_type_>
    util::status_t drive(plan_type_& plan, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Writes the checkpoint under a path, or into a buffer of @ref storage_requirements bytes. */
    template <typename checkpoint_type_>
    util::status_t save(checkpoint_type_ checkpoint, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Reads the checkpoint at a path or in a buffer. */
    template <typename checkpoint_type_>
    util::status_t load(checkpoint_type_ checkpoint, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** The bytes a save into memory takes, under the provenance the save records. */
    util::expected<std::size_t> storage_requirements() const noexcept;

    /** Serving a checkpoint in place answers @c unsupported_k until the index serves it. */
    util::status_t view(std::string_view) noexcept { return util::status_code_t::unsupported_k; }
    util::expected<util::sample_result_t> sample(basic_sample_inputs_t inputs, basic_sample_outputs_t outputs,
                                                 util::sample_config_t config, executor_t& executor) noexcept;
};

template <cpu_class_t cpu_class_>
template <typename callback_type_>
auto sequence_ram_variant<cpu_class_>::with_codec(callback_type_&& callback) const noexcept {
    auto with_kit = [&]<typename kit_type_>(std::type_identity<kit_type_>) noexcept {
        return callback(us::sequence::stored_codec<kit_type_> {
            sketch_tables_for(std::type_identity<typename kit_type_::sketch_tables_t> {}, spectrum),
            geometry.sketch_bytes});
    };
    switch (geometry.metric) {
    case usearch_sequence_metric_utf8_containment_k:
        return with_kit(std::type_identity<sequence_kit<usearch_sequence_metric_utf8_containment_k, capability_k>> {});
    case usearch_sequence_metric_dna_overlap_k:
        return with_kit(std::type_identity<sequence_kit<usearch_sequence_metric_dna_overlap_k, capability_k>> {});
    default:
        return with_kit(std::type_identity<sequence_kit<usearch_sequence_metric_protein_alignment_k, capability_k>> {});
    }
}

template <cpu_class_t cpu_class_>
util::expected<traversal_stats_t> sequence_ram_variant<cpu_class_>::insert(usearch_keys_view_t keys,
                                                                           caller_tape_t sequences,
                                                                           insert_config_t config, executor_t& executor,
                                                                           progress_ticks_t& ticks) noexcept {
    std::size_t const connectivity = geometry.connectivity, expansion = config.expansion_factor;
    // The batch's own pages, since the insert rewinds every arena of the executor.
    util::growing_arena<util::forkunion_allocator_t> staging {util::forkunion_allocator_t {util::no_memory_domain_k}};
    auto const [folded, folding_status] = folded_caller_tape_in(geometry.metric, fold_texts, sequences, staging,
                                                                &executor);
    if (failed(folding_status)) return {{}, folding_status};
    auto with_metric = [&]<typename kit_type_>(std::type_identity<kit_type_>, auto make_metric) noexcept {
        // Each value is written once: the sequence in its stored form, then its sketch in front.
        using sketch_tables_t = typename kit_type_::sketch_tables_t;
        sketch_tables_t const tables = sketch_tables_for(std::type_identity<sketch_tables_t> {}, spectrum);
        std::size_t const sketch_bytes = geometry.sketch_bytes;
        // One sketch scratch a thread, in the batch's pages, which the insert never rewinds.
        std::size_t scratch_bytes = 0;
        if constexpr (us::sequence::a_sketching_kit<kit_type_>)
            scratch_bytes = kit_type_::sketch_scratch_bytes(sketch_bytes);
        util::growing_buffer<util::byte_t, thread_allocator_t> scratch {thread_allocator_t {staging}};
        if (failed(scratch.resize_uninitialized(executor.size() * scratch_bytes)))
            return util::expected<traversal_stats_t> {{}, util::status_code_t::out_of_memory_k};
        auto build = [&](counted_tape_t const& entries, std::size_t offset, value_t& value,
                         std::size_t thread) noexcept {
            std::string_view const text = chars_at(entries, offset);
            std::size_t const sequence_bytes = stored_sequence_bytes(geometry.metric, text.size());
            std::size_t const bytes = sketch_bytes + sequence_bytes;
            return us::sequence::status_of(value.resize_and_overwrite(bytes, [&](char* stored, std::size_t) noexcept {
                util::bytes_span_t const sequence = util::bytes_of(stored + sketch_bytes, sequence_bytes);
                store_sequence(geometry.metric, util::bytes_of(text.data(), text.size()), alphabet, sequence);
                if constexpr (us::sequence::a_sketching_kit<kit_type_>)
                    kit_type_::sketch_write(
                        tables, sequence, util::bytes_of(stored, sketch_bytes),
                        util::to_writable_span(scratch).subspan(thread * scratch_bytes, scratch_bytes));
                return bytes;
            }));
        };
        return insert_entries(index, keys, folded, connectivity, expansion ? expansion : connectivity * 2, make_metric,
                              build, ticks, executor);
    };
    return with_sequence_metric<util::score_role_t::build_k, capability_k, cache_hints_t, value_t>(
        geometry, costs, spectrum, with_metric);
}

template <cpu_class_t cpu_class_>
util::expected<traversal_stats_t> sequence_ram_variant<cpu_class_>::search(
    counted_tape_t queries, usearch_matches_view_t const* hints, matches_outputs_t& outputs, search_config_t config,
    predicate_t const& predicate, executor_t& executor, progress_ticks_t& ticks) noexcept {
    auto with_metric = [&]<typename kit_type_>(std::type_identity<kit_type_>, auto make_metric,
                                               auto make_seeding) noexcept {
        // Each query turns into the metric's stored form in its thread's arena.
        auto query_of = [&](std::size_t query_index, std::size_t thread, auto&& answer) noexcept {
            auto const [query, status] = stored_query_in<kit_type_>(
                executor.arena(thread), chars_at(queries, query_index), geometry.metric, alphabet);
            return failed(status) ? status : answer(query);
        };
        return search_entries(index, queries.count(), geometry.connectivity, make_metric, query_of, outputs, config,
                              predicate, executor, ticks, hints, make_seeding);
    };
    return with_seeded_sequence_metric<capability_k, cache_hints_t, value_t>(geometry, costs, spectrum, with_metric);
}

template <cpu_class_t cpu_class_>
util::expected<util::sample_result_t> sequence_ram_variant<cpu_class_>::sample(basic_sample_inputs_t inputs,
                                                                               basic_sample_outputs_t outputs,
                                                                               util::sample_config_t config,
                                                                               executor_t& executor) noexcept {
    auto with_metric = [&]<typename kit_type_>(std::type_identity<kit_type_>, auto make_metric,
                                               auto make_seeding) noexcept {
        return sample_entries(index, geometry.connectivity, make_metric, inputs, outputs, config, executor,
                              util::no_fixed_coordinates_t {}, make_seeding);
    };
    return with_seeded_sequence_metric<capability_k, cache_hints_t, value_t>(geometry, costs, spectrum, with_metric);
}

template <cpu_class_t cpu_class_>
template <typename plan_type_>
util::status_t sequence_ram_variant<cpu_class_>::drive(plan_type_& plan, executor_t& executor,
                                                       progress_ticks_t& ticks) noexcept {
    auto with_metric = [&]<typename kit_type_>(std::type_identity<kit_type_>, auto make_metric) noexcept {
        auto [scratch, prepare_status] = prepare_refinement<shard_t>(plan, geometry.connectivity, make_metric,
                                                                     executor);
        if (failed(prepare_status)) return prepare_status;
        return drive_refinement(plan, scratch, basic::keep_diverse_t {geometry.connectivity}, ticks, executor);
    };
    return with_sequence_metric<util::score_role_t::build_k, capability_k, cache_hints_t, value_t>(
        geometry, costs, spectrum, with_metric);
}

/** Hands @p write, as <tt>write(config) → status_t</tt>, the save configuration recording what
 *  @p geometry, @p costs and @p alphabet froze: the connectivity, the sketch bytes of a metric
 *  keeping a sketch, and for the metrics with classes the costs as one hexadecimal digit per entry,
 *  the gap and the class of every byte as one digit of base 32, as every residency's checkpoint
 *  records them. */
template <typename write_type_>
util::status_t with_sequence_provenance(sequence_geometry_t const& geometry,
                                        us::sequence::alignment_costs_t const& costs,
                                        us::sequence::alignment_alphabet_t const& alphabet,
                                        write_type_&& write) noexcept {
    std::array<char, 24> connectivity_text, sketch_bytes_text;
    // The folded costs as one hexadecimal digit per entry, row by row over the classes.
    std::array<char, 24> gap_text;
    std::array<char, us::sequence::alignment_classes_k * us::sequence::alignment_classes_k> costs_text;
    std::size_t const entries = std::size_t {costs.classes} * costs.classes;
    for (std::size_t entry = 0; entry != entries; ++entry)
        costs_text[entry] = "0123456789abcdef"[costs.substitutions[entry / costs.classes][entry % costs.classes]];
    std::array<char, us::sequence::alignment_byte_alphabet_k> classes_text;
    for (std::size_t byte = 0; byte != classes_text.size(); ++byte)
        classes_text[byte] = "0123456789abcdefghijklmnopqrstuv"[alphabet.codes[byte]];
    // Connectivity, sketch bytes and the three fields of costs: five at most.
    std::array<util::safegraphs_custom_field_t, 5> fields {
        {{"connectivity", decimal_of(geometry.connectivity, connectivity_text)}}};
    std::size_t recorded = 1;
    if (geometry.sketch_bytes)
        fields[recorded++] = {"sketch_bytes", decimal_of(geometry.sketch_bytes, sketch_bytes_text)};
    if (costs.classes) {
        fields[recorded++] = {"substitution_costs", std::string_view {costs_text.data(), entries}};
        fields[recorded++] = {"gap_cost", decimal_of(costs.gap, gap_text)};
        fields[recorded++] = {"byte_classes", std::string_view {classes_text.begin(), classes_text.end()}};
    }
    util::save_config_t config;
    config.provenance = std::span {fields}.first(recorded);
    return write(config);
}

template <cpu_class_t cpu_class_>
template <typename checkpoint_type_>
util::status_t sequence_ram_variant<cpu_class_>::save(checkpoint_type_ checkpoint, executor_t& executor,
                                                      progress_ticks_t& ticks) noexcept {
    return with_sequence_provenance(geometry, costs, alphabet, [&](util::save_config_t const& config) noexcept {
        return with_codec([&](auto const& codec) noexcept {
            return basic::save(index, checkpoint, codec, executor, config, token_of(geometry.metric), ticks);
        });
    });
}

template <cpu_class_t cpu_class_>
template <typename checkpoint_type_>
util::status_t sequence_ram_variant<cpu_class_>::load(checkpoint_type_ checkpoint, executor_t& executor,
                                                      progress_ticks_t& ticks) noexcept {
    return with_codec([&](auto const& codec) noexcept {
        return basic::load(index, checkpoint, codec, executor, util::load_config_t {}, ticks);
    });
}

template <cpu_class_t cpu_class_>
util::expected<std::size_t> sequence_ram_variant<cpu_class_>::storage_requirements() const noexcept {
    std::size_t required = 0;
    util::status_t const sized = with_sequence_provenance(
        geometry, costs, alphabet, [&](util::save_config_t const& config) noexcept {
            auto const [bytes, bytes_status] = with_codec([&](auto const& codec) noexcept {
                return basic::storage_requirements(index, codec, config, token_of(geometry.metric));
            });
            required = bytes;
            return bytes_status;
        });
    return {required, sized};
}

/** One sequence alternative on block storage: the basic NVMe index over the strings a RAM variant
 *  saved, read-only, scored with the RAM variant's metrics and the kits @p cpu_class_ picks. Its
 *  verbs are instantiated in the class's unit, as the RAM variant's are. */
template <cpu_class_t cpu_class_>
struct sequence_nvme_variant {
    using executor_t = nvme_executor_t;
    using index_t = basic::nvme_indexes<us::sequence::codec_t, util::uuid128_t, util::u32_t, sz_size_t,
                                        default_io_queue_t, row_kit_of<cpu_class_>,
                                        atomic_ref_for_cpu_class<cpu_class_>::template type, waiter_of<cpu_class_>>;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;
    static constexpr sz_capability_t capability_k = stringzilla_capabilities_of(cpu_class_);

    index_t index;
    sequence_geometry_t geometry;

    /** The costs an alignment or an overlap scores with, folded at freeze. */
    us::sequence::alignment_costs_t costs {};

    /** The class code of every byte an alignment or an overlap stores, folded with @c costs. */
    us::sequence::alignment_alphabet_t alphabet {};

    /** The tables a smooth sketch reads, derived from @c costs at the geometry's sketch bytes when
     *  the index is viewed, so each fetched sequence is sketched as it lands. */
    sequence_spectrum_t spectrum {util::forkunion_allocator_t {util::no_memory_domain_k}};

    explicit sequence_nvme_variant(util::forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    /** Serves the checkpoint at @p path in place, its parts from @p paths when any are named. */
    util::status_t view(std::string_view path, std::span<std::string_view const> paths, executor_t& executor) noexcept {
        return index.view(path, paths, token_of(geometry.metric), executor);
    }

    util::expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                             predicate_t const& predicate, executor_t& executor,
                                             progress_ticks_t& ticks) noexcept;

    /** A view already is its checkpoint, so saving answers @c unsupported_k. */
    util::status_t save(std::string_view, executor_t&, progress_ticks_t&) noexcept {
        return util::status_code_t::unsupported_k;
    }
    util::expected<util::sample_result_t> sample(basic_sample_inputs_t inputs, basic_sample_outputs_t outputs,
                                                 util::sample_config_t config, executor_t& executor) noexcept;
};

template <cpu_class_t cpu_class_>
util::expected<traversal_stats_t> sequence_nvme_variant<cpu_class_>::search(
    counted_tape_t queries, matches_outputs_t& outputs, search_config_t config, predicate_t const& predicate,
    executor_t& executor, progress_ticks_t& ticks) noexcept {
    auto with_metric = [&]<typename kit_type_>(std::type_identity<kit_type_>, auto make_metric,
                                               auto make_seeding) noexcept {
        // Each query turns into the metric's stored form in its thread's arena.
        auto query_of = [&](std::size_t query_index, thread_allocator_t const& allocator) noexcept {
            return stored_query_in<kit_type_>(*allocator.arena, chars_at(queries, query_index), geometry.metric,
                                              alphabet);
        };
        return search_entries(index, queries.count(), make_metric, query_of, outputs, config, predicate, executor,
                              ticks, make_seeding);
    };
    return with_seeded_sequence_metric<capability_k, cache_hints_t, std::string_view>(geometry, costs, spectrum,
                                                                                      with_metric);
}

/** Every stored string is a query as it is read. */
template <cpu_class_t cpu_class_>
util::expected<util::sample_result_t> sequence_nvme_variant<cpu_class_>::sample(basic_sample_inputs_t inputs,
                                                                                basic_sample_outputs_t outputs,
                                                                                util::sample_config_t config,
                                                                                executor_t& executor) noexcept {
    auto with_metric = [&]<typename kit_type_>(std::type_identity<kit_type_>, auto make_metric,
                                               auto make_seeding) noexcept {
        auto query_of = [](us::sequence::codec_t::view_t stored, thread_allocator_t const&) noexcept {
            return util::expected<std::string_view> {stored, util::status_code_t::success_k};
        };
        return sample_entries(*this, make_metric, query_of, inputs, outputs, config, executor,
                              util::no_fixed_coordinates_t {}, make_seeding);
    };
    return with_seeded_sequence_metric<capability_k, cache_hints_t, std::string_view>(geometry, costs, spectrum,
                                                                                      with_metric);
}

/* The faces' members reading the kits, defined in @c src/sequence/vram_variants_nvidia.cpp. */
#if USEARCH_ARCH_CUDA_

extern template struct us::sequence::vram_nvidia_index<
    sequence_vram_kit<usearch_sequence_metric_protein_alignment_k, cuda_class_k>, util::u32_t,
    sequence_nvidia_fatbin<usearch_sequence_metric_protein_alignment_k>>;
extern template struct us::sequence::vram_nvidia_index<
    sequence_vram_kit<usearch_sequence_metric_dna_overlap_k, cuda_class_k>, util::u32_t,
    sequence_nvidia_fatbin<usearch_sequence_metric_dna_overlap_k>>;
extern template struct us::sequence::vram_nvidia_index<
    sequence_vram_kit<usearch_sequence_metric_utf8_containment_k, cuda_class_k>, util::u32_t,
    sequence_nvidia_fatbin<usearch_sequence_metric_utf8_containment_k>>;

/** A GPU containment or banded overlap index scores in the same fixed point as the RAM metric. */
template <typename node_id_type_, typename kernels_type_>
inline constexpr usearch_distance_t distance_unit<us::sequence::vram_nvidia_index<
    sequence_vram_kit<usearch_sequence_metric_utf8_containment_k, cuda_class_k>, node_id_type_, kernels_type_>> =
    1.0 / us::sequence::share_resolution_k;
template <typename node_id_type_, typename kernels_type_>
inline constexpr usearch_distance_t distance_unit<us::sequence::vram_nvidia_index<
    sequence_vram_kit<usearch_sequence_metric_dna_overlap_k, cuda_class_k>, node_id_type_, kernels_type_>> =
    1.0 / us::sequence::share_resolution_k;

#endif
#if USEARCH_ARCH_CUDA_ || USEARCH_ARCH_ROCM_ || USEARCH_ARCH_METAL_

/** The parameters a sequence kit scores with on a GPU, from the frozen @p geometry and @p costs:
 *  the metric's window widths beside the costs for a containment and an overlap, and the folded
 *  costs for an alignment. */
inline util::expected<us::sequence::window_parameters_t> parameters_of(
    std::type_identity<us::sequence::window_parameters_t>, sequence_geometry_t const& geometry,
    us::sequence::alignment_costs_t const& costs) noexcept {
    return us::sequence::window_parameters_t::of(geometry.metric == usearch_sequence_metric_utf8_containment_k
                                                     ? std::span {us::sequence::containment_window_widths_k}
                                                     : std::span {us::sequence::banded_overlap_window_widths_k},
                                                 costs);
}
inline util::expected<us::sequence::alignment_costs_t> parameters_of(
    std::type_identity<us::sequence::alignment_costs_t>, sequence_geometry_t const&,
    us::sequence::alignment_costs_t const& costs) noexcept {
    return {costs, util::status_code_t::success_k};
}

#endif
#if USEARCH_ARCH_CUDA_

/** Builds one shard of the sequence kernels @p metric_ on a CUDA device, holding its kit's
 *  parameters and sketch tables from the frozen @c geometry and costs, which its first
 *  reservation uploads to the device. */
template <usearch_sequence_metric_t metric_>
struct sequence_vram_nvidia_build {
    using face_t = us::sequence::vram_nvidia_index<sequence_vram_kit<metric_, cuda_class_k>, util::u32_t,
                                                   sequence_nvidia_fatbin<metric_>>;

    /** The geometry, costs and sketch tables of the variant holding the shards, frozen before the
     *  first is placed. */
    sequence_geometry_t const* geometry = nullptr;
    us::sequence::alignment_costs_t const* costs = nullptr;
    sequence_spectrum_t const* spectrum = nullptr;

    std::unique_ptr<face_t> operator()(int device) const noexcept {
        std::unique_ptr<face_t> face {new (std::nothrow) face_t(util::cuda_virtual_memory_allocator<util::byte_t> {
            gpu_drivers().cuda, gpu_drivers().extensions.cuda, device})};
        auto const [parameters, parameters_status] = parameters_of(
            std::type_identity<typename face_t::canonical_parameters_t> {}, *geometry, *costs);
        if (face && (failed(parameters_status) || failed(face->canonical_parameters(parameters)) ||
                     failed(face->sketch_tables(
                         sketch_tables_for(std::type_identity<typename face_t::sketch_tables_t> {}, *spectrum),
                         geometry->sketch_bytes))))
            face.reset();
        return face;
    }
};

/** The sequence alternative on NVIDIA GPUs of @p gpu_class_ : the basic VRAM shards under the
 *  sequence metric @p metric_, with 32-bit node IDs. */
template <usearch_sequence_metric_t metric_, gpu_class_t gpu_class_>
struct sequence_vram_nvidia_variant {
    using executor_t = util::cuda_executor_t;
    using build_t = sequence_vram_nvidia_build<metric_>;
    using face_t = typename build_t::face_t;
    using index_t = basic::vram_nvidia_indexes<face_t, util::uuid128_t, build_t>;
    using refine_plan_t = typename index_t::refine_plan_t;
    static constexpr usearch_sequence_metric_t metric_k = metric_;
    static constexpr gpu_class_t gpu_class_k = gpu_class_;

    sequence_geometry_t geometry;
    us::sequence::alignment_costs_t costs {};

    /** The class code of every byte an alignment or an overlap stores, folded with @c costs. */
    us::sequence::alignment_alphabet_t alphabet {};

    /** The tables a smooth sketch reads, derived from @c costs at the geometry's sketch bytes when
     *  the index is created or loaded, and never recorded. */
    sequence_spectrum_t spectrum {util::forkunion_allocator_t {util::no_memory_domain_k}};
    index_t index;

    /** Holds no shard until the handle places them; their metric is the template's, and their
     *  costs and sketch tables the variant's. */
    sequence_vram_nvidia_variant(int, sequence_geometry_t const&) noexcept
        : index(util::forkunion_allocator_t {util::no_memory_domain_k}, build_t {&geometry, &costs, &spectrum}) {}

    /** Reserves @p capacity as the tape stores it: the metric's sketch and a closing word per
     *  sequence. */
    util::status_t reserve(reserve_capacity_t capacity) noexcept {
        std::size_t const stored_bytes = capacity.bytes +
                                         capacity.entries * us::sequence::vram_stored_bytes(0, geometry.sketch_bytes);
        return vram_nvidia_reserve(index, capacity.entries, geometry.connectivity, stored_bytes);
    }

    util::expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t sequences, insert_config_t config,
                                             util::cuda_executor_t& executor, progress_ticks_t& ticks) noexcept {
        // The batch's own pages, since batches routed to disjoint shards may land at once.
        util::growing_arena<util::forkunion_allocator_t> staging {
            util::forkunion_allocator_t {util::no_memory_domain_k}};
        auto const [entries, lower_status] = counted_tape_of(sequences.tape, sequences.count, staging);
        if (failed(lower_status)) return {{}, lower_status};
        auto const [folded, folding_status] = folded_tape_in(metric_, host_fold_texts(), entries, staging, nullptr);
        if (failed(folding_status)) return {{}, folding_status};
        auto const [stored, storing_status] = stored_tape_in(staging, folded, metric_, geometry.sketch_bytes, alphabet);
        if (failed(storing_status)) return {{}, storing_status};
        return vram_nvidia_insert(index, keys, bytes_tape_of(stored), geometry.connectivity, config.expansion_factor,
                                  ticks, executor);
    }

    /** A device walk takes no host predicate; the queries turn into the metric's stored form on
     *  the call's @p pages. */
    util::expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                             util::growing_arena<util::forkunion_allocator_t>& pages,
                                             util::cuda_executor_t& executor, progress_ticks_t& ticks) noexcept {
        auto const [folded, folding_status] = folded_tape_in(metric_, host_fold_texts(), queries, pages, nullptr);
        if (failed(folding_status)) return {{}, folding_status};
        auto const [packed, packing_status] = stored_form_tape_in(pages, folded, metric_, alphabet);
        if (failed(packing_status)) return {{}, packing_status};
        return vram_nvidia_search(index, bytes_tape_of(packed), outputs.wanted, config, outputs, ticks, executor);
    }

    /** Writes the strings as the RAM variants write them: a lone live shard's monolith, or one per
     *  live shard beside the manifest the path names. */
    util::status_t save(std::string_view path, util::cuda_executor_t& executor, progress_ticks_t& ticks) noexcept {
        return with_sequence_provenance(geometry, costs, alphabet, [&](util::save_config_t const& config) noexcept {
            return vram_nvidia_save(index, path, us::sequence::codec_t {}, config, token_of(geometry.metric), executor,
                                    ticks);
        });
    }

    /** Reads a checkpoint any residency wrote, at a path or in host memory, into the placed shards,
     *  one part per shard. */
    template <typename checkpoint_type_>
    util::status_t load(checkpoint_type_ checkpoint, util::cuda_executor_t& executor,
                        progress_ticks_t& ticks) noexcept {
        return index.load(checkpoint, us::sequence::codec_t {}, executor, util::load_config_t {},
                          static_cast<util::connectivity_t>(geometry.connectivity), ticks);
    }

    /** Empties the index, retiring every shard behind a fresh shard generation on the same
     *  devices. */
    util::status_t clear() noexcept { return index.clear(); }

    /** Sampling answers @c unsupported_k until the shards sample on their devices. */
    util::expected<util::sample_result_t> sample(basic_sample_inputs_t, basic_sample_outputs_t, util::sample_config_t,
                                                 executor_t&) noexcept {
        return {{}, util::status_code_t::unsupported_k};
    }
};

#endif
#if USEARCH_ARCH_METAL_

/** A GPU containment or banded overlap index on Apple scores in the same fixed point as the RAM
 *  metric. */
template <typename node_id_type_>
inline constexpr usearch_distance_t
    distance_unit<us::sequence::vram_apple_index<us::sequence::apple_containment_kit_t, node_id_type_>> =
        1.0 / us::sequence::share_resolution_k;
template <typename node_id_type_>
inline constexpr usearch_distance_t
    distance_unit<us::sequence::vram_apple_index<us::sequence::apple_u8_banded_overlap_kit_t, node_id_type_>> =
        1.0 / us::sequence::share_resolution_k;

/** Builds one shard of the sequence kernels @p metric_ on an Apple device, holding its kit's
 *  parameters and sketch tables from the variant's frozen @c geometry and costs, which its first
 *  reservation uploads to the device. */
template <usearch_sequence_metric_t metric_>
struct sequence_vram_apple_build {
    using face_t = us::sequence::vram_apple_index<sequence_vram_kit<metric_, gpu_class_t::apple_metal7_k>, util::u32_t>;

    /** The geometry, costs and sketch tables of the variant holding the shards, frozen before the
     *  first is placed. */
    sequence_geometry_t const* geometry = nullptr;
    us::sequence::alignment_costs_t const* costs = nullptr;
    sequence_spectrum_t const* spectrum = nullptr;

    std::unique_ptr<face_t> operator()(int device) const noexcept {
        util::metal_context_t* const context = gpu_drivers().metal_contexts.context(device);
        if (!context) return {};
        std::unique_ptr<face_t> face {
            new (std::nothrow) face_t(util::metal_shared_allocator_t {*context, device, gpu_runtime().metal_kernels})};
        auto const [parameters, parameters_status] = parameters_of(
            std::type_identity<typename face_t::canonical_parameters_t> {}, *geometry, *costs);
        if (face && (failed(parameters_status) || failed(face->canonical_parameters(parameters)) ||
                     failed(face->sketch_tables(
                         sketch_tables_for(std::type_identity<typename face_t::sketch_tables_t> {}, *spectrum),
                         geometry->sketch_bytes))))
            face.reset();
        return face;
    }
};

/** The sequence alternative on Apple GPUs of @p gpu_class_ or newer: the basic Apple shards under
 *  the sequence metric @p metric_, with 32-bit node IDs, as @c sequence_vram_nvidia_variant. */
template <usearch_sequence_metric_t metric_, gpu_class_t gpu_class_>
struct sequence_vram_apple_variant {
    using executor_t = util::metal_executor_t;
    using build_t = sequence_vram_apple_build<metric_>;
    using face_t = typename build_t::face_t;
    using index_t = basic::vram_apple_indexes<face_t, util::uuid128_t, build_t>;
    using refine_plan_t = typename index_t::refine_plan_t;
    static constexpr usearch_sequence_metric_t metric_k = metric_;
    static constexpr gpu_class_t gpu_class_k = gpu_class_;

    sequence_geometry_t geometry;
    us::sequence::alignment_costs_t costs {};

    /** The class code of every byte an alignment or an overlap stores, folded with @c costs. */
    us::sequence::alignment_alphabet_t alphabet {};

    /** The tables a smooth sketch reads, derived from @c costs at the geometry's sketch bytes when
     *  the index is created or loaded, and never recorded. */
    sequence_spectrum_t spectrum {util::forkunion_allocator_t {util::no_memory_domain_k}};
    index_t index;

    /** Holds no shard until the handle places them; their metric is the template's, and their
     *  costs and sketch tables the variant's. */
    sequence_vram_apple_variant(int, sequence_geometry_t const&) noexcept
        : index(util::forkunion_allocator_t {util::no_memory_domain_k}, build_t {&geometry, &costs, &spectrum}) {}

    /** Reserves @p capacity as the tape stores it: the metric's sketch and a closing word per
     *  sequence. */
    util::status_t reserve(reserve_capacity_t capacity) noexcept {
        std::size_t const stored_bytes = capacity.bytes +
                                         capacity.entries * us::sequence::vram_stored_bytes(0, geometry.sketch_bytes);
        return vram_apple_reserve(index, capacity.entries, geometry.connectivity, stored_bytes);
    }

    util::expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t sequences, insert_config_t config,
                                             util::metal_executor_t& executor, progress_ticks_t& ticks) noexcept {
        // The batch's own pages, since batches routed to disjoint shards may land at once.
        util::growing_arena<util::forkunion_allocator_t> staging {
            util::forkunion_allocator_t {util::no_memory_domain_k}};
        auto const [entries, lower_status] = counted_tape_of(sequences.tape, sequences.count, staging);
        if (failed(lower_status)) return {{}, lower_status};
        auto const [folded, folding_status] = folded_tape_in(metric_, host_fold_texts(), entries, staging, nullptr);
        if (failed(folding_status)) return {{}, folding_status};
        auto const [stored, storing_status] = stored_tape_in(staging, folded, metric_, geometry.sketch_bytes, alphabet);
        if (failed(storing_status)) return {{}, storing_status};
        return vram_apple_insert(index, keys, bytes_tape_of(stored), geometry.connectivity, config.expansion_factor,
                                 ticks, executor);
    }

    /** A device walk takes no host predicate; the queries turn into the metric's stored form on
     *  the call's @p pages. */
    util::expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                             util::growing_arena<util::forkunion_allocator_t>& pages,
                                             util::metal_executor_t& executor, progress_ticks_t& ticks) noexcept {
        auto const [folded, folding_status] = folded_tape_in(metric_, host_fold_texts(), queries, pages, nullptr);
        if (failed(folding_status)) return {{}, folding_status};
        auto const [packed, packing_status] = stored_form_tape_in(pages, folded, metric_, alphabet);
        if (failed(packing_status)) return {{}, packing_status};
        return vram_apple_search(index, bytes_tape_of(packed), outputs.wanted, config, outputs, ticks, executor);
    }

    /** Writes the strings as the RAM variants write them. */
    util::status_t save(std::string_view path, util::metal_executor_t& executor, progress_ticks_t& ticks) noexcept {
        return with_sequence_provenance(geometry, costs, alphabet, [&](util::save_config_t const& config) noexcept {
            return vram_apple_save(index, path, us::sequence::codec_t {}, config, token_of(geometry.metric), executor,
                                   ticks);
        });
    }

    /** Reads a checkpoint any residency wrote, at a path or in host memory, into the placed shards,
     *  one part per shard. */
    template <typename checkpoint_type_>
    util::status_t load(checkpoint_type_ checkpoint, util::metal_executor_t& executor,
                        progress_ticks_t& ticks) noexcept {
        return index.load(checkpoint, us::sequence::codec_t {}, executor, util::load_config_t {},
                          static_cast<util::connectivity_t>(geometry.connectivity), ticks);
    }

    /** Empties the index, retiring every shard behind a fresh shard generation on the same
     *  devices. */
    util::status_t clear() noexcept { return index.clear(); }

    /** Sampling answers @c unsupported_k until the shards sample on their devices. */
    util::expected<util::sample_result_t> sample(basic_sample_inputs_t, basic_sample_outputs_t, util::sample_config_t,
                                                 executor_t&) noexcept {
        return {{}, util::status_code_t::unsupported_k};
    }
};

#endif

#if USEARCH_ARCH_ROCM_

/*  The faces' members reading the kits, defined in @c src/sequence/vram_variants_amd.hip for its
 *  class. */
extern template struct us::sequence::vram_amd_index<
    sequence_vram_kit<usearch_sequence_metric_protein_alignment_k, hip_class_k>, util::u32_t,
    sequence_amd_code_object<usearch_sequence_metric_protein_alignment_k, hip_class_k>>;
extern template struct us::sequence::vram_amd_index<
    sequence_vram_kit<usearch_sequence_metric_dna_overlap_k, hip_class_k>, util::u32_t,
    sequence_amd_code_object<usearch_sequence_metric_dna_overlap_k, hip_class_k>>;
extern template struct us::sequence::vram_amd_index<
    sequence_vram_kit<usearch_sequence_metric_utf8_containment_k, hip_class_k>, util::u32_t,
    sequence_amd_code_object<usearch_sequence_metric_utf8_containment_k, hip_class_k>>;

/** A GPU containment or banded overlap index scores in the same fixed point as the RAM metric. */
template <typename node_id_type_, typename kernels_type_>
inline constexpr usearch_distance_t distance_unit<us::sequence::vram_amd_index<
    sequence_vram_kit<usearch_sequence_metric_utf8_containment_k, hip_class_k>, node_id_type_, kernels_type_>> =
    1.0 / us::sequence::share_resolution_k;
template <typename node_id_type_, typename kernels_type_>
inline constexpr usearch_distance_t distance_unit<us::sequence::vram_amd_index<
    sequence_vram_kit<usearch_sequence_metric_dna_overlap_k, hip_class_k>, node_id_type_, kernels_type_>> =
    1.0 / us::sequence::share_resolution_k;

/** Builds one shard of the sequence kernels @p metric_ on an AMD device of @p gpu_class_, holding
 *  the kit's parameters and sketch tables from the frozen @c geometry and costs of the variant,
 *  which the first reservation uploads to the device. */
template <usearch_sequence_metric_t metric_, gpu_class_t gpu_class_>
struct sequence_vram_amd_build {
    using face_t = us::sequence::vram_amd_index<sequence_vram_kit<metric_, hip_class_k>, util::u32_t,
                                                sequence_amd_code_object<metric_, gpu_class_>>;

    /** The geometry, costs and sketch tables of the variant holding the shards, frozen before the
     *  first is placed. */
    sequence_geometry_t const* geometry = nullptr;
    us::sequence::alignment_costs_t const* costs = nullptr;
    sequence_spectrum_t const* spectrum = nullptr;

    std::unique_ptr<face_t> operator()(int device) const noexcept {
        std::unique_ptr<face_t> face {new (std::nothrow) face_t(util::hip_virtual_memory_allocator<util::byte_t> {
            gpu_drivers().hip, gpu_drivers().extensions.hip, device})};
        auto const [parameters, parameters_status] = parameters_of(
            std::type_identity<typename face_t::canonical_parameters_t> {}, *geometry, *costs);
        if (face && (failed(parameters_status) || failed(face->canonical_parameters(parameters)) ||
                     failed(face->sketch_tables(
                         sketch_tables_for(std::type_identity<typename face_t::sketch_tables_t> {}, *spectrum),
                         geometry->sketch_bytes))))
            face.reset();
        return face;
    }
};

/** The sequence alternative on AMD GPUs of @p gpu_class_ : the basic VRAM shards under the sequence
 *  metric @p metric_, with 32-bit node IDs. */
template <usearch_sequence_metric_t metric_, gpu_class_t gpu_class_>
struct sequence_vram_amd_variant {
    using executor_t = util::hip_executor_t;
    using build_t = sequence_vram_amd_build<metric_, gpu_class_>;
    using face_t = typename build_t::face_t;
    using index_t = basic::vram_amd_indexes<face_t, util::uuid128_t, build_t>;
    using refine_plan_t = typename index_t::refine_plan_t;
    static constexpr usearch_sequence_metric_t metric_k = metric_;
    static constexpr gpu_class_t gpu_class_k = gpu_class_;

    sequence_geometry_t geometry;
    us::sequence::alignment_costs_t costs {};

    /** The class code of every byte an alignment or an overlap stores, folded with @c costs. */
    us::sequence::alignment_alphabet_t alphabet {};

    /** The tables a smooth sketch reads, derived from @c costs at the geometry's sketch bytes when
     *  the index is created or loaded, and never recorded. */
    sequence_spectrum_t spectrum {util::forkunion_allocator_t {util::no_memory_domain_k}};
    index_t index;

    /** Holds no shard until the handle places them; their metric is the template's, and their
     *  costs and sketch tables the variant's. */
    sequence_vram_amd_variant(int, sequence_geometry_t const&) noexcept
        : index(util::forkunion_allocator_t {util::no_memory_domain_k}, build_t {&geometry, &costs, &spectrum}) {}

    /** Reserves @p capacity as the tape stores it: the metric's sketch and a closing word per
     *  sequence. */
    util::status_t reserve(reserve_capacity_t capacity) noexcept {
        std::size_t const stored_bytes = capacity.bytes +
                                         capacity.entries * us::sequence::vram_stored_bytes(0, geometry.sketch_bytes);
        return vram_amd_reserve(index, capacity.entries, geometry.connectivity, stored_bytes);
    }

    util::expected<traversal_stats_t> insert(usearch_keys_view_t keys, caller_tape_t sequences, insert_config_t config,
                                             util::hip_executor_t& executor, progress_ticks_t& ticks) noexcept {
        // The batch's own pages, since batches routed to disjoint shards may land at once.
        util::growing_arena<util::forkunion_allocator_t> staging {
            util::forkunion_allocator_t {util::no_memory_domain_k}};
        auto const [entries, lower_status] = counted_tape_of(sequences.tape, sequences.count, staging);
        if (failed(lower_status)) return {{}, lower_status};
        auto const [folded, folding_status] = folded_tape_in(metric_, host_fold_texts(), entries, staging, nullptr);
        if (failed(folding_status)) return {{}, folding_status};
        auto const [stored, storing_status] = stored_tape_in(staging, folded, metric_, geometry.sketch_bytes, alphabet);
        if (failed(storing_status)) return {{}, storing_status};
        return vram_amd_insert(index, keys, bytes_tape_of(stored), geometry.connectivity, config.expansion_factor,
                               ticks, executor);
    }

    /** A device walk takes no host predicate; the queries turn into the metric's stored form on
     *  the call's @p pages. */
    util::expected<traversal_stats_t> search(counted_tape_t queries, matches_outputs_t& outputs, search_config_t config,
                                             util::growing_arena<util::forkunion_allocator_t>& pages,
                                             util::hip_executor_t& executor, progress_ticks_t& ticks) noexcept {
        auto const [folded, folding_status] = folded_tape_in(metric_, host_fold_texts(), queries, pages, nullptr);
        if (failed(folding_status)) return {{}, folding_status};
        auto const [packed, packing_status] = stored_form_tape_in(pages, folded, metric_, alphabet);
        if (failed(packing_status)) return {{}, packing_status};
        return vram_amd_search(index, bytes_tape_of(packed), outputs.wanted, config, outputs, ticks, executor);
    }

    /** Writes the strings as the RAM variants write them: a lone live shard's monolith, or one per
     *  live shard beside the manifest the path names. */
    util::status_t save(std::string_view path, util::hip_executor_t& executor, progress_ticks_t& ticks) noexcept {
        return with_sequence_provenance(geometry, costs, alphabet, [&](util::save_config_t const& config) noexcept {
            return vram_amd_save(index, path, us::sequence::codec_t {}, config, token_of(geometry.metric), executor,
                                 ticks);
        });
    }

    /** Reads a checkpoint any residency wrote, at a path or in host memory, into the placed shards,
     *  one part per shard. */
    template <typename checkpoint_type_>
    util::status_t load(checkpoint_type_ checkpoint, util::hip_executor_t& executor, progress_ticks_t& ticks) noexcept {
        return index.load(checkpoint, us::sequence::codec_t {}, executor, util::load_config_t {},
                          static_cast<util::connectivity_t>(geometry.connectivity), ticks);
    }

    /** Empties the index, retiring every shard behind a fresh shard generation on the same
     *  devices. */
    util::status_t clear() noexcept { return index.clear(); }

    /** Sampling answers @c unsupported_k until the shards sample on their devices. */
    util::expected<util::sample_result_t> sample(basic_sample_inputs_t, basic_sample_outputs_t, util::sample_config_t,
                                                 executor_t&) noexcept {
        return {{}, util::status_code_t::unsupported_k};
    }
};

#endif

#pragma endregion Sequence Variants

#pragma region Sequence Bodies

/** The sequence variants each class unit compiles: one per CPU class on the host and on storage,
 *  and one per metric on an NVIDIA, AMD or Apple GPU. */
template <auto class_>
struct sequence_variants_of<vendored_residency_t::ram_k, class_> {
    using type = type_list<sequence_ram_variant<class_>>;
};
template <auto class_>
struct sequence_variants_of<vendored_residency_t::nvme_k, class_> {
    using type = type_list<sequence_nvme_variant<class_>>;
};
#if USEARCH_ARCH_CUDA_ && USEARCH_TARGET_MONOVERSIONED
template <auto class_>
struct sequence_variants_of<vendored_residency_t::vram_nvidia_k, class_> {
    using type = type_list<sequence_vram_nvidia_variant<usearch_sequence_metric_protein_alignment_k, class_>,
                           sequence_vram_nvidia_variant<usearch_sequence_metric_dna_overlap_k, class_>,
                           sequence_vram_nvidia_variant<usearch_sequence_metric_utf8_containment_k, class_>>;
};
#elif USEARCH_ARCH_CUDA_
template <>
struct sequence_variants_of<vendored_residency_t::vram_nvidia_k, gpu_class_t::nvidia_ampere_k> {
    using type = type_list<
        sequence_vram_nvidia_variant<usearch_sequence_metric_protein_alignment_k, gpu_class_t::nvidia_ampere_k>,
        sequence_vram_nvidia_variant<usearch_sequence_metric_dna_overlap_k, gpu_class_t::nvidia_ampere_k>,
        sequence_vram_nvidia_variant<usearch_sequence_metric_utf8_containment_k, gpu_class_t::nvidia_ampere_k>>;
};
#endif
#if USEARCH_ARCH_METAL_
template <>
struct sequence_variants_of<vendored_residency_t::vram_apple_k, gpu_class_t::apple_metal7_k> {
    using type =
        type_list<sequence_vram_apple_variant<usearch_sequence_metric_protein_alignment_k, gpu_class_t::apple_metal7_k>,
                  sequence_vram_apple_variant<usearch_sequence_metric_dna_overlap_k, gpu_class_t::apple_metal7_k>,
                  sequence_vram_apple_variant<usearch_sequence_metric_utf8_containment_k, gpu_class_t::apple_metal7_k>>;
};
#endif
#if USEARCH_ARCH_ROCM_ && USEARCH_TARGET_MONOVERSIONED
template <auto class_>
struct sequence_variants_of<vendored_residency_t::vram_amd_k, class_> {
    using type = type_list<sequence_vram_amd_variant<usearch_sequence_metric_protein_alignment_k, class_>,
                           sequence_vram_amd_variant<usearch_sequence_metric_dna_overlap_k, class_>,
                           sequence_vram_amd_variant<usearch_sequence_metric_utf8_containment_k, class_>>;
};
#elif USEARCH_ARCH_ROCM_
template <>
struct sequence_variants_of<vendored_residency_t::vram_amd_k, gpu_class_t::amd_cdna3_k> {
    using type =
        type_list<sequence_vram_amd_variant<usearch_sequence_metric_protein_alignment_k, gpu_class_t::amd_cdna3_k>,
                  sequence_vram_amd_variant<usearch_sequence_metric_dna_overlap_k, gpu_class_t::amd_cdna3_k>,
                  sequence_vram_amd_variant<usearch_sequence_metric_utf8_containment_k, gpu_class_t::amd_cdna3_k>>;
};
template <>
struct sequence_variants_of<vendored_residency_t::vram_amd_k, gpu_class_t::amd_cdna4_k> {
    using type =
        type_list<sequence_vram_amd_variant<usearch_sequence_metric_protein_alignment_k, gpu_class_t::amd_cdna4_k>,
                  sequence_vram_amd_variant<usearch_sequence_metric_dna_overlap_k, gpu_class_t::amd_cdna4_k>,
                  sequence_vram_amd_variant<usearch_sequence_metric_utf8_containment_k, gpu_class_t::amd_cdna4_k>>;
};
template <>
struct sequence_variants_of<vendored_residency_t::vram_amd_k, gpu_class_t::amd_cdna5_k> {
    using type =
        type_list<sequence_vram_amd_variant<usearch_sequence_metric_protein_alignment_k, gpu_class_t::amd_cdna5_k>,
                  sequence_vram_amd_variant<usearch_sequence_metric_dna_overlap_k, gpu_class_t::amd_cdna5_k>,
                  sequence_vram_amd_variant<usearch_sequence_metric_utf8_containment_k, gpu_class_t::amd_cdna5_k>>;
};
#endif

/** Whether a variant of @p variant_type_ keeps the tables of a smooth sketch. */
template <typename variant_type_>
concept a_variant_deriving_sketch_tables = requires(variant_type_& variant) {
    { variant.spectrum } -> std::same_as<sequence_spectrum_t&>;
};

/** What the sequence verbs do to one variant: fold its alphabet, describe it, and land and tally
 *  its strings. */
template <>
struct variant_bodies<sequence_traits_t> {
    using geometry_t = sequence_geometry_t;
    using geometry_extras_t = sequence_traits_t::geometry_extras_t;
    using costs_t = us::sequence::alignment_costs_t;
    using alphabet_t = us::sequence::alignment_alphabet_t;

    /** What a view keeps of a string it read, copied into its thread's arena. */
    using kept_t = util::bytes_view_t;

    /** Whether @p source enters @p variant: the same metric, costs, byte classes and
     *  connectivity. */
    template <typename variant_type_, typename source_type_>
    static bool matches(variant_type_ const& variant, source_type_ const& source) noexcept {
        geometry_t const &index = variant.geometry, &other = source.geometry;
        return index.metric == other.metric && index.connectivity == other.connectivity &&
               index.sketch_bytes == other.sketch_bytes &&
               std::memcmp(&variant.costs, &source.costs, sizeof(costs_t)) == 0 &&
               std::memcmp(&variant.alphabet, &source.alphabet, sizeof(alphabet_t)) == 0;
    }

    /**
     *  @brief Folds the similarities of the @p named costs, or else the metric's own, 256×256
     *      between bytes, into @p alphabet and the @p costs between its classes, or else unfolds
     *      the costs and byte classes @p geometry recalled; an overlap's record their complements.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k Bytes folding into one class or past 32, a negative gap, a pair above
     *    its mean self-score, or an overlap's costs outside the four letters' classes.
     *  - @c malformed_k Recorded costs no square of 2 to 32 classes holds, or classes not one per
     *    byte below the square's side.
     */
    static util::status_t fold(geometry_t const& geometry, geometry_extras_t named, costs_t& costs,
                               alphabet_t& alphabet) noexcept {
        constexpr std::size_t side_k = us::sequence::alignment_byte_alphabet_k;
        if (geometry.metric == usearch_sequence_metric_utf8_containment_k) return util::status_code_t::success_k;
        std::string_view const recalled = to_string_view(geometry.substitution_costs);
        if (!recalled.empty()) {
            std::size_t const classes = sequence_traits_t::classes_recorded(geometry);
            std::string_view const byte_classes = to_string_view(geometry.byte_classes);
            if (classes * classes != recalled.size() || classes < 2 || classes > us::sequence::alignment_classes_k ||
                byte_classes.size() != side_k)
                return {util::status_code_t::malformed_k, "Recorded costs hold no square of 2 to 32 classes"};
            for (std::size_t byte = 0; byte != side_k; ++byte) {
                char const digit = byte_classes[byte];
                std::size_t const code = digit >= '0' && digit <= '9'   ? std::size_t(digit - '0')
                                         : digit >= 'a' && digit <= 'v' ? std::size_t(digit - 'a' + 10)
                                                                        : classes;
                if (code >= classes) return {util::status_code_t::malformed_k, "Recorded byte class past the costs"};
                alphabet.codes[byte] = static_cast<util::u8_t>(code);
            }
            costs.classes = static_cast<util::u8_t>(classes);
            costs.gap = static_cast<util::u8_t>(geometry.gap_cost);
            for (std::size_t entry = 0; entry != recalled.size(); ++entry) {
                int const cost = hexadecimal_in(recalled[entry]);
                if (cost < 0) return {util::status_code_t::malformed_k, "Recorded cost is no hexadecimal digit"};
                costs.substitutions[entry / classes][entry % classes] = static_cast<util::u8_t>(cost);
            }
        }
        else {
            bool const dna = geometry.metric == usearch_sequence_metric_dna_overlap_k;
            util::i8_t own[side_k * side_k];
            if (!named) {
                if (dna) us::sequence::dna_similarities(own);
                else us::sequence::protein_similarities(own);
            }
            util::status_t const folded = alphabet_t::fold(
                std::span<util::i8_t const> {named ? named->similarities : own, side_k * side_k},
                named ? named->gap
                : dna ? us::sequence::dna_gap_k
                      : us::sequence::protein_gap_k,
                alphabet, costs);
            if (failed(folded) || costs.classes < 2)
                return {util::status_code_t::config_mismatch_k,
                        "Costs fold into one class, past 32, or above self-scores"};
        }
        return geometry.metric == usearch_sequence_metric_dna_overlap_k ? alphabet.fold_complement(costs)
                                                                        : util::status_code_t::success_k;
    }

    /**
     *  @brief Moves @p geometry into @p variant, its costs folded and the tables of a smooth sketch
     *      derived, a zero connectivity and zero sketch bytes taking the defaults; a GPU variant's
     *      shards take the costs from it as they are built.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k Sketch bytes for a metric that writes no sketch, or bytes not a
     *    multiple of four.
     */
    template <typename variant_type_>
    static util::status_t adopt(variant_type_& variant, geometry_t& geometry, geometry_extras_t named) noexcept {
        if (!geometry.connectivity) geometry.connectivity = util::connectivity_32_k;
        std::size_t const default_bytes = default_sketch_bytes(geometry.metric);
        if (!geometry.sketch_bytes) geometry.sketch_bytes = default_bytes;
        if ((geometry.sketch_bytes && !default_bytes) || geometry.sketch_bytes % 4)
            return {util::status_code_t::config_mismatch_k, "Sketch bytes for a sketchless metric or not words"};
        if (util::status_t const folded = fold(geometry, named, variant.costs, variant.alphabet); failed(folded))
            return folded;
        if constexpr (a_variant_deriving_sketch_tables<variant_type_>)
            if (geometry.metric == usearch_sequence_metric_protein_alignment_k)
                if (util::status_t const derived = variant.spectrum.derive(variant.costs, geometry.sketch_bytes);
                    failed(derived))
                    return derived;
        variant.geometry = std::move(geometry);
        return util::status_code_t::success_k;
    }

    /** The geometry @p variant froze, with the classes its alphabet folded into; a RAM index adds
     *  the bytes of every stored value, sketch included, which its shards leave uncounted. */
    template <typename variant_type_>
    static void describe(variant_type_ const& variant, usearch_sequence_index_description_t& description) noexcept {
        description.metadata = sequence_traits_t::metadata_of(variant.geometry);
        description.metadata.classes = variant.costs.classes;
        if constexpr (is_ram_index<variant_type_>)
            for (typename variant_type_::shard_t const& shard : variant.index.live_shards())
                for (std::size_t node = 0, published = shard.size(); node != published; ++node)
                    description.memory_usage +=
                        shard.published_value_of(static_cast<typename variant_type_::shard_t::node_id_t>(node)).size();
    }

    static util::status_t keep(nvme_executor_t& executor, std::size_t thread, std::string_view value,
                               kept_t& kept) noexcept {
        return keep_entry(executor, thread, util::bytes_of(value.data(), value.size()), kept);
    }

    /** A device's stored value keeps its sequence between the metric's sketch and the closing
     *  padding; the sequence alone is kept, where the call's arena holds it. */
    template <typename variant_type_>
    static util::status_t keep_stored(variant_type_ const& variant, util::bytes_view_t stored, kept_t& kept) noexcept {
        kept = stored.empty() ? stored : us::sequence::vram_stored_sequence(stored, variant.geometry.sketch_bytes);
        return util::status_code_t::success_k;
    }

    /** Writes @p count sequences, the @c i th from what @p value_at hands, as one u8 tape in
     *  @p memory: a RAM value past its sketch, a kept sequence whole. */
    template <typename variant_type_, typename value_at_type_>
    static util::status_t write_values(variant_type_ const& variant, std::size_t count, value_at_type_&& value_at,
                                       util::growing_arena<util::forkunion_allocator_t>& memory,
                                       usearch_tape_view_t* sequences) noexcept {
        std::size_t const sketch_bytes = is_ram_index<variant_type_> ? variant.geometry.sketch_bytes : 0;
        std::size_t* const offsets = carve<std::size_t>(memory, count + 1);
        if (!offsets) return util::status_code_t::out_of_memory_k;
        offsets[0] = 0;
        for (std::size_t entry = 0; entry != count; ++entry)
            offsets[entry + 1] = offsets[entry] + value_at(entry).size() - sketch_bytes;
        util::byte_t* const bytes = carve<util::byte_t>(memory, offsets[count]);
        if (offsets[count] && !bytes) return util::status_code_t::out_of_memory_k;
        for (std::size_t entry = 0; entry != count; ++entry)
            if (auto const& value = value_at(entry); value.size() != sketch_bytes)
                std::memcpy(bytes + offsets[entry], value.data() + sketch_bytes, value.size() - sketch_bytes);
        write_output_tape(bytes, offsets, usearch_scalar_u8_k, *sequences);
        return util::status_code_t::success_k;
    }

    /** Adds one entry's sequence bytes to a key's statistics, from any thread: a view's own, a RAM
     *  value's past its sketch. */
    template <typename variant_type_, typename value_type_>
    static void tally(variant_type_ const& variant, usearch_sequence_key_stats_t& stats,
                      value_type_ const& value) noexcept {
        std::size_t const sketch_bytes = is_ram_index<variant_type_> ? variant.geometry.sketch_bytes : 0;
        std::size_t const bytes = value.size() - sketch_bytes;
        fu::standard_atomic_ref<std::size_t>(stats.total_bytes).store_add(bytes, std::memory_order_relaxed);
        raise_to<fu::standard_atomic_ref>(stats.largest_bytes, bytes);
    }
};

#pragma endregion Sequence Bodies
