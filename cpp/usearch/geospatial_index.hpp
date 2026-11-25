/**
 *  @file cpp/usearch/geospatial_index.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief RAII handles and typed verbs over the geospatial modality, which add no ABI of their own
 *      and throw nothing at all.
 *
 *  Observations cross as spans of @c usearch_geospatial_observation_t, eleven doubles in natural
 *  units where a field no device recorded is NaN. @c search ranks the nearest entries a relation
 *  admits, and @c within reports every one inside the tolerance. The receivers, plans and handlers
 *  are the dense ones: the index and @c on block and answer a @c status_t or an @c expected, while
 *  @c deferred submits and answers a @c deferred call. Each verb answers the statuses its C
 *  function lists in @c usearch/geospatial_index.h.
 *
 *  @code{.cpp}
 *  namespace us = unum::usearch;
 *  auto [index, index_status] = us::geospatial_index_t::init_ram({});
 *  auto [nearby, status] = index.on(executor).within({&station, 1}, usearch_geospatial_relation_near_k,
 *                                                    {.metres = 500}, arena);
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span>        // `std::span`
#include <type_traits> // `std::is_nothrow_invocable_r_v`

#include "usearch/geospatial_index.h" // `usearch_geospatial_*` verbs and their argument structs
#include "usearch/types.hpp"          // `deferrable`, `answered_`, `opened_`, `owner`, `plan`, `executor_t`, `arena_t`

namespace unum::usearch {

/** Owns one cursor, opened by @c geospatial_index_t::cursor_init. */
using geospatial_cursor_t = owner<usearch_geospatial_cursor_t, usearch_geospatial_cursor_free>;

/** A refine in flight, which @c refine opens and continues until @c refine_plan_finish ends. */
using geospatial_refine_plan_t = plan<usearch_geospatial_refine_plan_t>;

/** A reshard in flight, which @c reshard opens and continues until @c reshard_plan_finish ends. */
using geospatial_reshard_plan_t = plan<usearch_geospatial_reshard_plan_t>;

/** The knobs of @c refine and the plan it opens or continues. */
using geospatial_refine_config_t = refine_config<geospatial_refine_plan_t>;

/** The knobs of @c reshard and the plan it opens or continues. */
using geospatial_reshard_config_t = reshard_config<geospatial_reshard_plan_t>;

/** The observations @c get wrote into its arena, readable until the arena's next call. */
struct geospatial_observations_t {

    /** The first observation. */
    usearch_geospatial_observation_t const* first = nullptr;

    /** Observations written. */
    std::size_t written = 0;

    std::span<usearch_geospatial_observation_t const> span() const noexcept { return {first, written}; }
};

/** Every match @c within wrote into its arena, one ragged run per query. */
struct geospatial_within_t {

    /** Keys, chords in metres and offsets, end to end. */
    usearch_matches_tape_view_t view {};

    /** Matches written across every query. */
    std::size_t written = 0;
};

/** A terrain handler: fills a block of heights in metres, leaving NaN where it has none, through a
 *  const call operator, since every executor thread may ask it at once. */
template <typename handler_type_>
concept a_terrain_handler = std::is_nothrow_invocable_r_v<void, handler_type_ const&, std::size_t, std::size_t,
                                                          std::size_t, std::size_t, float*, std::size_t>;

/** The @p grid filled on demand by @p handler, which every index naming it keeps for its life
 *  and calls from the executor's threads. */
template <a_terrain_handler handler_type_>
usearch_geospatial_terrain_t geospatial_terrain(handler_type_& handler, usearch_geospatial_terrain_t grid) noexcept {
    grid.callback = [](std::size_t first_row, std::size_t first_column, std::size_t rows, std::size_t columns,
                       float* heights, std::size_t heights_stride, void* state) noexcept {
        (*static_cast<handler_type_ const*>(state))(first_row, first_column, rows, columns, heights, heights_stride);
    };
    grid.state = &handler;
    return grid;
}

/** The knobs of @c search. */
struct geospatial_search_config_t {

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

/** The knobs of @c within. */
struct geospatial_within_config_t {

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
 *  @brief A geospatial index and its verbs, completed as @p receiver_type_ says: the owning handle
 *      blocks on the calling thread, @c on blocks on an executor, and both answer a @c status_t or
 *      an @c expected, while @c deferred submits to an executor and answers a @c deferred call.
 *
 *  Each verb is written once, filling its C struct with the call's @c outputs.
 *  Every input, handler, plan and arena a deferred call names must stay allocated and unchanged
 *  until the call is final.
 */
template <typename receiver_type_>
class geospatial_index {
    static constexpr usearch_completion_t completion_k = receiver_type_::completion;
    template <typename value_type_ = void>
    using deferrable = usearch::deferrable<value_type_, completion_k>;
    receiver_type_ receiver_;

    template <typename>
    friend class geospatial_index;
    explicit geospatial_index(receiver_type_ receiver) noexcept : receiver_(std::move(receiver)) {}

  public:
    geospatial_index() noexcept
        requires a_handle_owner<receiver_type_>
    = default;

    usearch_geospatial_index_t handle() const noexcept { return receiver_.handle(); }

    /** The output field a lifecycle verb writes the new handle into. */
    usearch_geospatial_index_t* handle_field() noexcept
        requires a_handle_owner<receiver_type_>
    {
        return receiver_.handle_field();
    }

    /** The same handle, whose verbs block on @p executor. */
    geospatial_index<caller_owned<usearch_geospatial_index_t, usearch_completion_blocking_k>> on(
        executor_t const& executor) const noexcept {
        return geospatial_index<caller_owned<usearch_geospatial_index_t, usearch_completion_blocking_k>>(
            {receiver_.handle(), executor.handle()});
    }

    /** The same handle, whose verbs submit to @p executor, each answering a @c deferred call. */
    geospatial_index<caller_owned<usearch_geospatial_index_t, usearch_completion_deferred_k>> deferred(
        executor_t const& executor) const noexcept {
        return geospatial_index<caller_owned<usearch_geospatial_index_t, usearch_completion_deferred_k>>(
            {receiver_.handle(), executor.handle()});
    }

#pragma region Mutation

    /** Grows the shards ahead of @p entries entries in all. */
    deferrable<> reserve(std::size_t entries) const noexcept {
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_geospatial_reserve_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k,
                                                    entries};
            usearch_geospatial_reserve(&arguments);
        });
    }

    /** Erases all entries, keeping the reserved capacity and the terrain. */
    deferrable<> clear() const noexcept {
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_geospatial_clear_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k};
            usearch_geospatial_clear(&arguments);
        });
    }

    /** Adds @p observations under @p keys, one key per observation, all or nothing per call. */
    deferrable<> insert(std::span<usearch_key_t const> keys,
                        std::span<usearch_geospatial_observation_t const> observations,
                        insert_config_t const& config = {}) const noexcept {
        // The call has one count, so only this side sees the keys and the observations disagree.
        if (keys.size() != observations.size())
            return answered_<deferrable<>>(
                {status_code_t::shape_mismatch_k, "The key count differs from the observation count"});
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_geospatial_insert_t arguments {
                .index = receiver_.handle(),
                .executor = receiver_.executor(),
                .outcome = outcome,
                .completion = completion_k,
                .count = keys.size(),
                .keys = {keys.data(), sizeof(usearch_key_t)},
                .observations = observations.data(),
                .expansion_factor = config.expansion_factor,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            usearch_geospatial_insert(&arguments);
        });
    }

    /** Runs one refine round, or continues @c config.plan, answering the round's report. */
    deferrable<usearch_refine_report_t> refine(geospatial_refine_config_t const& config = {}) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_geospatial_refine_t arguments {
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
                usearch_geospatial_refine(&arguments);
            });
    }

    /** Closes and frees @p plan, opened on this index and executor, answering all it did. */
    deferrable<usearch_refine_report_t> refine_plan_finish(geospatial_refine_plan_t plan) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_geospatial_refine_plan_finish_t arguments {
                    receiver_.handle(), receiver_.executor(), outcome, completion_k, plan.handle(), &written->value};
                usearch_geospatial_refine_plan_finish(&arguments);
            });
    }

    /** Moves the entries of @p sources into @p targets with this receiver's executor, answering the
     *  report of the closing rounds; every handle named outlives the call and the plan's finish. */
    deferrable<usearch_refine_report_t> reshard(std::span<usearch_geospatial_index_t const> sources,
                                                std::span<usearch_geospatial_index_t const> targets,
                                                geospatial_reshard_config_t const& config = {}) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_geospatial_reshard_t arguments {
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
                usearch_geospatial_reshard(&arguments);
            });
    }

    /** Closes and frees @p plan, opened with this executor, answering what it did in all. */
    deferrable<usearch_refine_report_t> reshard_plan_finish(geospatial_reshard_plan_t plan) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_geospatial_reshard_plan_finish_t arguments {receiver_.executor(), outcome, completion_k,
                                                                    plan.handle(), &written->value};
                usearch_geospatial_reshard_plan_finish(&arguments);
            });
    }

    /** Writes the index to the checkpoint at @p path, which holds no terrain. */
    deferrable<> save(usearch_bytes_view_t path, progress_t progress = {}) const noexcept {
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_geospatial_save_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k, path,
                                                 progress.raw()};
            usearch_geospatial_save(&arguments);
        });
    }

    /** Writes the index into one monolith in @p arena. */
    deferrable<usearch_bytes_view_t> save_buffer(arena_t& arena, progress_t progress = {}) const noexcept {
        return answered_<deferrable<usearch_bytes_view_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_bytes_view_t>* written) noexcept {
                usearch_geospatial_save_buffer_t arguments {receiver_.handle(), receiver_.executor(), outcome,
                                                            completion_k,       &written->value,      arena.handle(),
                                                            progress.raw()};
                usearch_geospatial_save_buffer(&arguments);
            });
    }

#pragma endregion Mutation

#pragma region Reads

    /** Reports the frozen metadata and the current counters. */
    deferrable<usearch_geospatial_index_description_t> describe() const noexcept {
        return answered_<deferrable<usearch_geospatial_index_description_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_geospatial_index_description_t>* written) noexcept {
                usearch_geospatial_describe_t arguments {receiver_.handle(), receiver_.executor(), outcome,
                                                         completion_k, &written->value};
                usearch_geospatial_describe(&arguments);
            });
    }

    /** Counts the observations stored under @p key. */
    deferrable<usearch_geospatial_key_stats_t> count(usearch_key_t key) const noexcept {
        return answered_<deferrable<usearch_geospatial_key_stats_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_geospatial_key_stats_t>* written) noexcept {
                usearch_geospatial_count_t arguments {
                    receiver_.handle(), receiver_.executor(), outcome, completion_k, key, &written->value};
                usearch_geospatial_count(&arguments);
            });
    }

    /** Reads every observation stored under @p key into @p arena, within the stored rounding. */
    deferrable<geospatial_observations_t> get(usearch_key_t key, arena_t& arena) const noexcept {
        return answered_<deferrable<geospatial_observations_t>>(
            [&](usearch_outcome_t* outcome, outputs<geospatial_observations_t>* written) noexcept {
                usearch_geospatial_get_t arguments {
                    receiver_.handle(),    receiver_.executor(),    outcome,       completion_k, key,
                    &written->value.first, &written->value.written, arena.handle()};
                usearch_geospatial_get(&arguments);
            });
    }

    /** Audits how much of the graph a search can reach. */
    deferrable<usearch_reachability_t> health() const noexcept {
        return answered_<deferrable<usearch_reachability_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_reachability_t>* written) noexcept {
                usearch_geospatial_health_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k,
                                                       &written->value};
                usearch_geospatial_health(&arguments);
            });
    }

    /** Draws up to @p wanted keys around @c config.focus, or across the whole index, into @p arena,
     *  each hanging from a shown or earlier draw and placed in three dimensions. */
    deferrable<sample_t> sample(std::size_t wanted, arena_t& arena, sample_config_t const& config = {}) const noexcept {
        return answered_<deferrable<sample_t>>([&](usearch_outcome_t* outcome, outputs<sample_t>* written) noexcept {
            usearch_geospatial_sample_t arguments = sample_arguments_<usearch_geospatial_sample_t>(
                receiver_.handle(), receiver_.executor(), outcome, completion_k, wanted, arena, config,
                &written->value);
            usearch_geospatial_sample(&arguments);
        });
    }

    /** Finds the @p wanted nearest entries @p relation admits within @p tolerance of every query,
     *  written into @p arena with chords in metres. */
    deferrable<usearch_matches_view_t> search(std::span<usearch_geospatial_observation_t const> queries,
                                              usearch_geospatial_relation_t relation,
                                              usearch_geospatial_tolerance_t tolerance, std::size_t wanted,
                                              arena_t& arena,
                                              geospatial_search_config_t const& config = {}) const noexcept {
        return answered_<deferrable<usearch_matches_view_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_matches_view_t>* written) noexcept {
                usearch_geospatial_search_t arguments {
                    .index = receiver_.handle(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .count = queries.size(),
                    .queries = queries.data(),
                    .relation = relation,
                    .tolerance = tolerance,
                    .matches = &written->value,
                    .arena = arena.handle(),
                    .wanted = wanted,
                    .expansion_factor = config.expansion_factor,
                    .traversal = config.traversal,
                    .predicate = config.predicate.raw(),
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                usearch_geospatial_search(&arguments);
            });
    }

    /** Reports every entry @p relation admits within @p tolerance of every query, however many,
     *  written into @p arena as one ragged run per query. */
    deferrable<geospatial_within_t> within(std::span<usearch_geospatial_observation_t const> queries,
                                           usearch_geospatial_relation_t relation,
                                           usearch_geospatial_tolerance_t tolerance, arena_t& arena,
                                           geospatial_within_config_t const& config = {}) const noexcept {
        return answered_<deferrable<geospatial_within_t>>(
            [&](usearch_outcome_t* outcome, outputs<geospatial_within_t>* written) noexcept {
                usearch_geospatial_within_t arguments {
                    .index = receiver_.handle(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .count = queries.size(),
                    .queries = queries.data(),
                    .relation = relation,
                    .tolerance = tolerance,
                    .matches = &written->value.view,
                    .written = &written->value.written,
                    .arena = arena.handle(),
                    .traversal = config.traversal,
                    .predicate = config.predicate.raw(),
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                usearch_geospatial_within(&arguments);
            });
    }

    /** Pulls the next @p wanted entries @p cursor walks to into @p arena, with their links when
     *  @p links asks. */
    deferrable<cursor_batch_t> cursor_next(geospatial_cursor_t const& cursor, std::size_t wanted, arena_t& arena,
                                           bool links = false) const noexcept {
        return answered_<deferrable<cursor_batch_t>>(
            [&](usearch_outcome_t* outcome, outputs<cursor_batch_t>* written) noexcept {
                usearch_geospatial_cursor_next_t arguments {receiver_.handle(),
                                                            receiver_.executor(),
                                                            outcome,
                                                            completion_k,
                                                            cursor.handle(),
                                                            &written->value.keys,
                                                            links ? &written->value.links : nullptr,
                                                            &written->value.written,
                                                            arena.handle(),
                                                            wanted};
                usearch_geospatial_cursor_next(&arguments);
            });
    }

#pragma endregion Reads

#pragma region Lifecycle

    /** Creates an empty RAM index over the terrain @p arguments names, if any. */
    static expected<geospatial_index> init_ram(usearch_geospatial_init_ram_t arguments) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<geospatial_index>(usearch_geospatial_init_ram, arguments, &usearch_geospatial_init_ram_t::index);
    }

    /** Creates an empty VRAM index over the heights @p arguments points at, if any, on the
     *  devices it names. */
    static expected<geospatial_index> init_vram(usearch_geospatial_init_vram_t arguments) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<geospatial_index>(usearch_geospatial_init_vram, arguments,
                                         &usearch_geospatial_init_vram_t::index);
    }

    /** Copies a checkpoint into a new, mutable index in host memory. */
    static expected<geospatial_index> load_ram(usearch_geospatial_load_ram_t arguments, executor_t const& executor = {},
                                               progress_t progress = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<geospatial_index>(usearch_geospatial_load_ram, arguments, &usearch_geospatial_load_ram_t::index,
                                         executor, progress);
    }

    /** Copies a checkpoint into a new, mutable index in device memory. */
    static expected<geospatial_index> load_vram(usearch_geospatial_load_vram_t arguments,
                                                executor_t const& executor = {}, progress_t progress = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<geospatial_index>(usearch_geospatial_load_vram, arguments,
                                         &usearch_geospatial_load_vram_t::index, executor, progress);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in host memory. */
    static expected<geospatial_index> load_ram_buffer(usearch_geospatial_load_ram_buffer_t arguments,
                                                      executor_t const& executor = {},
                                                      progress_t progress = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<geospatial_index>(usearch_geospatial_load_ram_buffer, arguments,
                                         &usearch_geospatial_load_ram_buffer_t::index, executor, progress);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in device memory. */
    static expected<geospatial_index> load_vram_buffer(usearch_geospatial_load_vram_buffer_t arguments,
                                                       executor_t const& executor = {},
                                                       progress_t progress = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<geospatial_index>(usearch_geospatial_load_vram_buffer, arguments,
                                         &usearch_geospatial_load_vram_buffer_t::index, executor, progress);
    }

    /** Serves a checkpoint in place through a memory mapping, as a read-only index. */
    static expected<geospatial_index> view_ram(usearch_geospatial_view_ram_t arguments) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<geospatial_index>(usearch_geospatial_view_ram, arguments, &usearch_geospatial_view_ram_t::index);
    }

    /** Serves a checkpoint in place from NVMe drives as a read-only index, every later call reading
     *  it through @p executor 's engines. */
    static expected<geospatial_index> view_nvme(usearch_geospatial_view_nvme_t arguments,
                                                executor_t const& executor = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<geospatial_index>(usearch_geospatial_view_nvme, arguments,
                                         &usearch_geospatial_view_nvme_t::index, executor);
    }

    /** Reads a checkpoint's connectivity without loading it. */
    static expected<usearch_geospatial_metadata_t> metadata_read(usearch_geospatial_metadata_read_t arguments,
                                                                 executor_t const& executor = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return metadata_read_<usearch_geospatial_metadata_t>(usearch_geospatial_metadata_read, arguments,
                                                             executor.handle());
    }

    /** Reads connectivity from a serialized checkpoint buffer. */
    static expected<usearch_geospatial_metadata_t> metadata_read_buffer(
        usearch_geospatial_metadata_read_buffer_t arguments, executor_t const& executor = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return metadata_read_<usearch_geospatial_metadata_t>(usearch_geospatial_metadata_read_buffer, arguments,
                                                             executor.handle());
    }

    /** Opens a cursor over every entry, or over the entries @p starts names first. */
    expected<geospatial_cursor_t> cursor_init(std::span<usearch_key_t const> starts = {}) const noexcept
        requires a_handle_owner<receiver_type_>
    {
        usearch_geospatial_cursor_init_t arguments {handle(), nullptr, nullptr, starts.size(), starts.data(), nullptr};
        return opened_<geospatial_cursor_t>(usearch_geospatial_cursor_init, arguments,
                                            &usearch_geospatial_cursor_init_t::cursor);
    }

#pragma endregion Lifecycle
};

/** Owns one geospatial index; destruction waits for its deferred calls. */
using geospatial_index_t = geospatial_index<owning<usearch_geospatial_index_t, usearch_geospatial_free>>;

/** A geospatial index its caller owns and the executor its verbs run on, completing them as
 *  @p completion_ says. */
template <usearch_completion_t completion_>
using geospatial_index_on = geospatial_index<caller_owned<usearch_geospatial_index_t, completion_>>;

} // namespace unum::usearch
