/**
 *  @file cpp/usearch/sequence_index.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief RAII handles and typed verbs over the sequence modality, which add no ABI of their own
 *      and throw nothing at all.
 *
 *  Sequences cross as byte tapes, the bytes of every entry end to end around one more offset than
 *  entries, and come back from @c get the same way. The receivers, plans and handlers are the dense
 *  ones: the index and @c on block and answer a @c status_t or an @c expected, while @c deferred
 *  submits and answers a @c deferred call. Each verb answers the statuses its C function lists in
 *  @c usearch/sequence_index.h.
 *
 *  @code{.cpp}
 *  namespace us = unum::usearch;
 *  auto [index, index_status] = us::sequence_index_t::init_ram({
 *      .metric = usearch_sequence_metric_levenshtein_utf8_k});
 *  std::uint8_t const bytes[] = {'k', 'i', 't', 't', 'e', 'n'};
 *  std::uint64_t const offsets[] = {0, 6};
 *  us::status_t const inserted = index.on(executor).insert(keys, {bytes, offsets});
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`

#include <span> // `std::span`

#include "usearch/sequence_index.h" // `usearch_sequence_*` verbs and their argument structs
#include "usearch/types.hpp"        // `deferrable`, `answered_`, `opened_`, `owner`, `plan`, `executor_t`, `arena_t`

namespace unum::usearch {

/** The canonical spelling of a metric kind, as @c usearch_sequence_metric_name answers it. */
inline char const* sequence_metric_name(usearch_sequence_metric_t metric) noexcept {
    return usearch_sequence_metric_name(metric);
}

/** Reads a metric kind spelled @p name back, or fails with @c status_code_t::invalid_argument_k. */
inline expected<usearch_sequence_metric_t> sequence_metric_from_name(usearch_bytes_view_t name) noexcept {
    return metric_read_(usearch_sequence_metric_from_name, name);
}

/** Owns one cursor, opened by @c sequence_index_t::cursor_init. */
using sequence_cursor_t = owner<usearch_sequence_cursor_t, usearch_sequence_cursor_free>;

/** A refine in flight, which @c refine opens and continues until @c refine_plan_finish ends. */
using sequence_refine_plan_t = plan<usearch_sequence_refine_plan_t>;

/** A reshard in flight, which @c reshard opens and continues until @c reshard_plan_finish ends. */
using sequence_reshard_plan_t = plan<usearch_sequence_reshard_plan_t>;

/** The knobs of @c refine and the plan it opens or continues. */
using sequence_refine_config_t = refine_config<sequence_refine_plan_t>;

/** The knobs of @c reshard and the plan it opens or continues. */
using sequence_reshard_config_t = reshard_config<sequence_reshard_plan_t>;

/** The byte tapes @c get wrote into its arena, one entry per sequence stored under the key. */
using sequence_entries_t = tape_entries<std::uint8_t>;

/** The knobs of @c search. */
struct sequence_search_config_t {

    /** The breadth of the candidates the walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** A walk or a scan; unknown derives. */
    usearch_traversal_t traversal = usearch_traversal_unknown_k;

    /** Which candidate keys may be reported; empty admits every key. */
    predicate_t predicate {};

    /** Hears the queries answered, and may cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

/**
 *  @brief A sequence index and its verbs, completed as @p receiver_type_ says: the owning handle
 *      blocks on the calling thread, @c on blocks on an executor, and both answer a @c status_t or
 *      an @c expected, while @c deferred submits to an executor and answers a @c deferred call.
 *
 *  Each verb is written once, filling its C struct with the call's @c outputs. Every
 *  input, handler, plan and arena a deferred call names stays borrowed until it is final.
 */
template <typename receiver_type_>
class sequence_index {
    static constexpr usearch_completion_t completion_k = receiver_type_::completion;
    template <typename value_type_ = void>
    using deferrable = usearch::deferrable<value_type_, completion_k>;
    receiver_type_ receiver_;

    template <typename>
    friend class sequence_index;
    explicit sequence_index(receiver_type_ receiver) noexcept : receiver_(std::move(receiver)) {}

  public:
    sequence_index() noexcept
        requires owns_handle<receiver_type_>
    = default;

    usearch_sequence_index_t handle() const noexcept { return receiver_.handle(); }

    /** The output field a lifecycle verb writes the new handle into. */
    usearch_sequence_index_t* handle_field() noexcept
        requires owns_handle<receiver_type_>
    {
        return receiver_.handle_field();
    }

    /** The same handle, whose verbs block on @p executor. */
    sequence_index<borrowing<usearch_sequence_index_t, usearch_completion_blocking_k>> on(
        executor_t const& executor) const noexcept {
        return sequence_index<borrowing<usearch_sequence_index_t, usearch_completion_blocking_k>>(
            {receiver_.handle(), executor.handle()});
    }

    /** The same handle, whose verbs submit to @p executor, each answering a @c deferred call. */
    sequence_index<borrowing<usearch_sequence_index_t, usearch_completion_deferred_k>> deferred(
        executor_t const& executor) const noexcept {
        return sequence_index<borrowing<usearch_sequence_index_t, usearch_completion_deferred_k>>(
            {receiver_.handle(), executor.handle()});
    }

#pragma region Mutation

    /** Grows the shards ahead of @p entries entries and @p payload_bytes sequence bytes in all. */
    deferrable<> reserve(std::size_t entries, std::size_t payload_bytes = 0) const noexcept {
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_sequence_reserve_t arguments {
                receiver_.handle(), receiver_.executor(), outcome, completion_k, entries, payload_bytes};
            usearch_sequence_reserve(&arguments);
        });
    }

    /** Erases all entries, keeping the reserved capacity and frozen metric. */
    deferrable<> clear() const noexcept {
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_sequence_clear_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k};
            usearch_sequence_clear(&arguments);
        });
    }

    /** Adds @p sequences under @p keys, one key per entry, all or nothing per call. */
    deferrable<> insert(std::span<usearch_key_t const> keys, tape_view<std::uint8_t> sequences,
                        insert_config_t const& config = {}) const noexcept {
        // The call has one count, so only this side sees the keys and the tape disagree.
        if (keys.size() != sequences.count)
            return answered_<deferrable<>>(
                {status_code_t::shape_mismatch_k, "The key count differs from the sequence count"});
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_sequence_insert_t arguments {
                .index = receiver_.handle(),
                .executor = receiver_.executor(),
                .outcome = outcome,
                .completion = completion_k,
                .count = keys.size(),
                .keys = {keys.data(), sizeof(usearch_key_t)},
                .sequences = sequences.view(),
                .expansion_factor = config.expansion_factor,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            usearch_sequence_insert(&arguments);
        });
    }

    /** Runs one refine round, or continues @c config.plan, answering the round's report. */
    deferrable<usearch_refine_report_t> refine(sequence_refine_config_t const& config = {}) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_sequence_refine_t arguments {
                    .index = receiver_.handle(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .report = &written->value,
                    .refine = config.refine,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                    .plan = handle_field_of(config.plan),
                    .phase = phase_field_of(config.plan),
                };
                usearch_sequence_refine(&arguments);
            });
    }

    /** Closes and frees @p plan, opened on this index and executor, answering all it did. */
    deferrable<usearch_refine_report_t> refine_plan_finish(sequence_refine_plan_t plan) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_sequence_refine_plan_finish_t arguments {
                    receiver_.handle(), receiver_.executor(), outcome, completion_k, plan.handle(), &written->value};
                usearch_sequence_refine_plan_finish(&arguments);
            });
    }

    /** Moves the entries of @p sources into @p targets with this receiver's executor, answering the
     *  report of the closing rounds; every handle named outlives the call and the plan's finish. */
    deferrable<usearch_refine_report_t> reshard(std::span<usearch_sequence_index_t const> sources,
                                                std::span<usearch_sequence_index_t const> targets,
                                                sequence_reshard_config_t const& config = {}) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_sequence_reshard_t arguments {
                    .sources_count = sources.size(),
                    .sources = sources.data(),
                    .targets_count = targets.size(),
                    .targets = targets.data(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .report = &written->value,
                    .refine = config.refine,
                    .seed = config.seed,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                    .plan = handle_field_of(config.plan),
                    .phase = phase_field_of(config.plan),
                };
                usearch_sequence_reshard(&arguments);
            });
    }

    /** Closes and frees @p plan, opened with this executor, answering what it did in all. */
    deferrable<usearch_refine_report_t> reshard_plan_finish(sequence_reshard_plan_t plan) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_sequence_reshard_plan_finish_t arguments {receiver_.executor(), outcome, completion_k,
                                                                  plan.handle(), &written->value};
                usearch_sequence_reshard_plan_finish(&arguments);
            });
    }

    /** Writes the index to the checkpoint at @p path. */
    deferrable<> save(usearch_bytes_view_t path, progress_t progress = {}) const noexcept {
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_sequence_save_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k, path,
                                               progress.raw()};
            usearch_sequence_save(&arguments);
        });
    }

    /** Writes the index into one monolith in @p arena. */
    deferrable<usearch_bytes_view_t> save_buffer(arena_t& arena, progress_t progress = {}) const noexcept {
        return answered_<deferrable<usearch_bytes_view_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_bytes_view_t>* written) noexcept {
                usearch_sequence_save_buffer_t arguments {receiver_.handle(), receiver_.executor(), outcome,
                                                          completion_k,       &written->value,      arena.handle(),
                                                          progress.raw()};
                usearch_sequence_save_buffer(&arguments);
            });
    }

#pragma endregion Mutation

#pragma region Reads

    /** Reports the frozen metadata and the current counters. */
    deferrable<usearch_sequence_index_description_t> describe() const noexcept {
        return answered_<deferrable<usearch_sequence_index_description_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_sequence_index_description_t>* written) noexcept {
                usearch_sequence_describe_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k,
                                                       &written->value};
                usearch_sequence_describe(&arguments);
            });
    }

    /** Counts the entries stored under @p key, with their total and largest byte lengths. */
    deferrable<usearch_sequence_key_stats_t> count(usearch_key_t key) const noexcept {
        return answered_<deferrable<usearch_sequence_key_stats_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_sequence_key_stats_t>* written) noexcept {
                usearch_sequence_count_t arguments {
                    receiver_.handle(), receiver_.executor(), outcome, completion_k, key, &written->value};
                usearch_sequence_count(&arguments);
            });
    }

    /** Reads every sequence stored under @p key into @p arena as one byte tape. */
    deferrable<sequence_entries_t> get(usearch_key_t key, arena_t& arena) const noexcept {
        return answered_<deferrable<sequence_entries_t>>(
            [&](usearch_outcome_t* outcome, outputs<sequence_entries_t>* written) noexcept {
                usearch_sequence_get_t arguments {
                    receiver_.handle(),   receiver_.executor(),    outcome,       completion_k, key,
                    &written->value.view, &written->value.written, arena.handle()};
                usearch_sequence_get(&arguments);
            });
    }

    /** Audits how much of the graph a search can reach. */
    deferrable<usearch_reachability_t> health() const noexcept {
        return answered_<deferrable<usearch_reachability_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_reachability_t>* written) noexcept {
                usearch_sequence_health_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k,
                                                     &written->value};
                usearch_sequence_health(&arguments);
            });
    }

    /** Draws up to @p wanted keys around @c config.focus, or across the whole index, into @p arena,
     *  each hanging from a shown or earlier draw and placed in three dimensions. */
    deferrable<sample_t> sample(std::size_t wanted, arena_t& arena, sample_config_t const& config = {}) const noexcept {
        return answered_<deferrable<sample_t>>([&](usearch_outcome_t* outcome, outputs<sample_t>* written) noexcept {
            usearch_sequence_sample_t arguments = sample_arguments_<usearch_sequence_sample_t>(
                receiver_.handle(), receiver_.executor(), outcome, completion_k, wanted, arena, config,
                &written->value);
            usearch_sequence_sample(&arguments);
        });
    }

    /** Finds the @p wanted nearest entries of each sequence of @p queries, written to @p arena. */
    deferrable<usearch_matches_view_t> search(tape_view<std::uint8_t> queries, std::size_t wanted, arena_t& arena,
                                              sequence_search_config_t const& config = {}) const noexcept {
        return answered_<deferrable<usearch_matches_view_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_matches_view_t>* written) noexcept {
                usearch_sequence_search_t arguments {
                    .index = receiver_.handle(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .count = queries.count,
                    .queries = queries.view(),
                    .matches = &written->value,
                    .arena = arena.handle(),
                    .wanted = wanted,
                    .expansion_factor = config.expansion_factor,
                    .traversal = config.traversal,
                    .predicate = config.predicate.raw(),
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                usearch_sequence_search(&arguments);
            });
    }

    /** Pulls the next @p wanted entries @p cursor walks to into @p arena, with their links when
     *  @p links asks. */
    deferrable<cursor_batch_t> cursor_next(sequence_cursor_t const& cursor, std::size_t wanted, arena_t& arena,
                                           bool links = false) const noexcept {
        return answered_<deferrable<cursor_batch_t>>(
            [&](usearch_outcome_t* outcome, outputs<cursor_batch_t>* written) noexcept {
                usearch_sequence_cursor_next_t arguments {receiver_.handle(),
                                                          receiver_.executor(),
                                                          outcome,
                                                          completion_k,
                                                          cursor.handle(),
                                                          &written->value.keys,
                                                          links ? &written->value.links : nullptr,
                                                          &written->value.written,
                                                          arena.handle(),
                                                          wanted};
                usearch_sequence_cursor_next(&arguments);
            });
    }

#pragma endregion Reads

#pragma region Lifecycle

    /** Creates an empty RAM index scoring with the metric @p arguments freezes. */
    static expected<sequence_index> init_ram(usearch_sequence_init_ram_t arguments) noexcept
        requires owns_handle<receiver_type_>
    {
        return opened_<sequence_index>(usearch_sequence_init_ram, arguments, &usearch_sequence_init_ram_t::index);
    }

    /** Creates an empty VRAM index of the metric @p arguments freezes, on the device it names. */
    static expected<sequence_index> init_vram(usearch_sequence_init_vram_t arguments) noexcept
        requires owns_handle<receiver_type_>
    {
        return opened_<sequence_index>(usearch_sequence_init_vram, arguments, &usearch_sequence_init_vram_t::index);
    }

    /** Copies a checkpoint into a new, mutable index in host memory. */
    static expected<sequence_index> load_ram(usearch_sequence_load_ram_t arguments, executor_t const& executor = {},
                                             progress_t progress = {}) noexcept
        requires owns_handle<receiver_type_>
    {
        return opened_<sequence_index>(usearch_sequence_load_ram, arguments, &usearch_sequence_load_ram_t::index,
                                       executor, progress);
    }

    /** Copies a checkpoint into a new, mutable index in device memory. */
    static expected<sequence_index> load_vram(usearch_sequence_load_vram_t arguments, executor_t const& executor = {},
                                              progress_t progress = {}) noexcept
        requires owns_handle<receiver_type_>
    {
        return opened_<sequence_index>(usearch_sequence_load_vram, arguments, &usearch_sequence_load_vram_t::index,
                                       executor, progress);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in host memory. */
    static expected<sequence_index> load_ram_buffer(usearch_sequence_load_ram_buffer_t arguments,
                                                    executor_t const& executor = {}, progress_t progress = {}) noexcept
        requires owns_handle<receiver_type_>
    {
        return opened_<sequence_index>(usearch_sequence_load_ram_buffer, arguments,
                                       &usearch_sequence_load_ram_buffer_t::index, executor, progress);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in device memory. */
    static expected<sequence_index> load_vram_buffer(usearch_sequence_load_vram_buffer_t arguments,
                                                     executor_t const& executor = {}, progress_t progress = {}) noexcept
        requires owns_handle<receiver_type_>
    {
        return opened_<sequence_index>(usearch_sequence_load_vram_buffer, arguments,
                                       &usearch_sequence_load_vram_buffer_t::index, executor, progress);
    }

    /** Serves a checkpoint in place through a memory mapping, as a read-only index. */
    static expected<sequence_index> view_ram(usearch_sequence_view_ram_t arguments) noexcept
        requires owns_handle<receiver_type_>
    {
        return opened_<sequence_index>(usearch_sequence_view_ram, arguments, &usearch_sequence_view_ram_t::index);
    }

    /** Serves a checkpoint in place from NVMe drives as a read-only index, every later call reading
     *  it through @p executor 's engines. */
    static expected<sequence_index> view_nvme(usearch_sequence_view_nvme_t arguments,
                                              executor_t const& executor = {}) noexcept
        requires owns_handle<receiver_type_>
    {
        return opened_<sequence_index>(usearch_sequence_view_nvme, arguments, &usearch_sequence_view_nvme_t::index,
                                       executor);
    }

    /** Reads a checkpoint's metric & connectivity without loading it. */
    static expected<usearch_sequence_metadata_t> metadata_read(usearch_sequence_metadata_read_t arguments,
                                                               executor_t const& executor = {}) noexcept
        requires owns_handle<receiver_type_>
    {
        return metadata_read_<usearch_sequence_metadata_t>(usearch_sequence_metadata_read, arguments,
                                                           executor.handle());
    }

    /** Reads metric & connectivity from a serialized checkpoint buffer. */
    static expected<usearch_sequence_metadata_t> metadata_read_buffer(usearch_sequence_metadata_read_buffer_t arguments,
                                                                      executor_t const& executor = {}) noexcept
        requires owns_handle<receiver_type_>
    {
        return metadata_read_<usearch_sequence_metadata_t>(usearch_sequence_metadata_read_buffer, arguments,
                                                           executor.handle());
    }

    /** Opens a cursor over every entry, or over the entries @p starts names first. */
    expected<sequence_cursor_t> cursor_init(std::span<usearch_key_t const> starts = {}) const noexcept
        requires owns_handle<receiver_type_>
    {
        usearch_sequence_cursor_init_t arguments {handle(), nullptr, starts.size(), starts.data(), nullptr};
        return opened_<sequence_cursor_t>(usearch_sequence_cursor_init, arguments,
                                          &usearch_sequence_cursor_init_t::cursor);
    }

#pragma endregion Lifecycle
};

/** Owns one sequence index; destruction waits for its deferred calls. */
using sequence_index_t = sequence_index<owning<usearch_sequence_index_t, usearch_sequence_free>>;

/** A borrowed sequence index and its executor, completing its verbs as @p completion_ says. */
template <usearch_completion_t completion_>
using sequence_index_on = sequence_index<borrowing<usearch_sequence_index_t, completion_>>;

} // namespace unum::usearch
