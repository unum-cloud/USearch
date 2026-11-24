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
 *  are the dense ones: @c geospatial_index_t::on blocks and answers an @c outcome, while
 *  @c geospatial_index_t::deferred submits and answers a @c deferred call. Each verb answers the
 *  statuses its C function lists in @c usearch/geospatial_index.h.
 *
 *  @code{.cpp}
 *  namespace us = unum::usearch;
 *  auto [index, status, message] = us::geospatial_index_t::init_ram({});
 *  auto nearby = index.on(executor).within({&station, 1}, usearch_geospatial_relation_near_k,
 *                                          {.metres = 500}, arena);
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span>        // `std::span`
#include <type_traits> // `std::is_nothrow_invocable_r_v`

#include "usearch/geospatial_index.h"
#include "usearch/types.hpp"

namespace unum::usearch {

/** Owns one cursor, opened by @c geospatial_index_t::cursor_init. */
using geospatial_cursor_t = cursor<usearch_geospatial_cursor_t, usearch_geospatial_cursor_free>;

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

/** A terrain handler: fills a block of heights in metres, leaving NaN where it has none. */
template <typename handler_type_>
concept terrain_handler = std::is_nothrow_invocable_r_v<void, handler_type_&, std::size_t, std::size_t, std::size_t,
                                                        std::size_t, float*, std::size_t>;

/** The @p grid filled on demand by @p handler, which every index naming it keeps for its life
 *  and calls from the executor's threads. */
template <terrain_handler handler_type_>
usearch_geospatial_terrain_t geospatial_terrain(handler_type_& handler, usearch_geospatial_terrain_t grid) noexcept {
    grid.callback = [](std::size_t first_row, std::size_t first_column, std::size_t rows, std::size_t columns,
                       float* heights, std::size_t heights_stride, void* state) noexcept {
        (*static_cast<handler_type_*>(state))(first_row, first_column, rows, columns, heights, heights_stride);
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
 *  @brief The verbs of a geospatial index, completed as @p completion_ says: blocking answers an
 *      @c outcome, deferred answers a @c deferred call. Made by @c geospatial_index_t::on and
 *      @c geospatial_index_t::deferred.
 *
 *  Each verb is written once, filling its C struct with the result as its output slot. Every input,
 *  handler, plan and arena a deferred call names must stay allocated and unchanged until the call
 *  is final, whichever verb it is.
 */
template <usearch_completion_t completion_>
class geospatial_index_on {
  public:
#pragma region Mutation

    /** Grows the shards ahead of @p entries entries in all. */
    answer<completion_, void> reserve(std::size_t entries) const noexcept {
        return answered_<completion_, void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_geospatial_reserve_t arguments {index_, executor_, outcome, completion_, entries};
            usearch_geospatial_reserve(&arguments);
        });
    }

    /** Erases all entries, keeping the reserved capacity and the terrain. */
    answer<completion_, void> clear() const noexcept {
        return answered_<completion_, void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_geospatial_clear_t arguments {index_, executor_, outcome, completion_};
            usearch_geospatial_clear(&arguments);
        });
    }

    /** Adds @p observations under @p keys, one key per observation, all or nothing per call. */
    answer<completion_, void> insert(std::span<usearch_key_t const> keys,
                                     std::span<usearch_geospatial_observation_t const> observations,
                                     insert_config_t const& config = {}) const noexcept {
        // The call has one count, so only this side sees the keys and the observations disagree.
        if (keys.size() != observations.size())
            return refused_<completion_, void>(usearch_status_shape_mismatch_k,
                                               "The key count differs from the observation count");
        return answered_<completion_, void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_geospatial_insert_t arguments {
                .index = index_,
                .executor = executor_,
                .outcome = outcome,
                .completion = completion_,
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
    answer<completion_, usearch_refine_report_t> refine(geospatial_refine_config_t const& config = {}) const noexcept {
        return answered_<completion_, usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_geospatial_refine_t arguments {
                    .index = index_,
                    .executor = executor_,
                    .outcome = outcome,
                    .completion = completion_,
                    .report = report,
                    .refine = config.refine,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                    .plan = config.plan ? config.plan->handle_field() : nullptr,
                    .phase = config.plan ? config.plan->phase_field() : nullptr,
                };
                usearch_geospatial_refine(&arguments);
            });
    }

    /** Closes and frees @p plan, opened on this index and executor, answering all it did. */
    answer<completion_, usearch_refine_report_t> refine_plan_finish(geospatial_refine_plan_t plan) const noexcept {
        return answered_<completion_, usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_geospatial_refine_plan_finish_t arguments {index_,      executor_,     outcome,
                                                                   completion_, plan.handle(), report};
                usearch_geospatial_refine_plan_finish(&arguments);
            });
    }

    /** Moves the entries of @p sources into @p targets with this receiver's executor, answering the
     *  report of the closing rounds; every handle named outlives the call and the plan's finish. */
    answer<completion_, usearch_refine_report_t> reshard(
        std::span<usearch_geospatial_index_t const> sources, std::span<usearch_geospatial_index_t const> targets,
        geospatial_reshard_config_t const& config = {}) const noexcept {
        return answered_<completion_, usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_geospatial_reshard_t arguments {
                    .sources_count = sources.size(),
                    .sources = sources.data(),
                    .targets_count = targets.size(),
                    .targets = targets.data(),
                    .executor = executor_,
                    .outcome = outcome,
                    .completion = completion_,
                    .report = report,
                    .refine = config.refine,
                    .seed = config.seed,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                    .plan = config.plan ? config.plan->handle_field() : nullptr,
                    .phase = config.plan ? config.plan->phase_field() : nullptr,
                };
                usearch_geospatial_reshard(&arguments);
            });
    }

    /** Closes and frees @p plan, opened with this executor, answering what it did in all. */
    answer<completion_, usearch_refine_report_t> reshard_plan_finish(geospatial_reshard_plan_t plan) const noexcept {
        return answered_<completion_, usearch_refine_report_t>([&](usearch_outcome_t* outcome,
                                                                   usearch_refine_report_t* report) noexcept {
            usearch_geospatial_reshard_plan_finish_t arguments {executor_, outcome, completion_, plan.handle(), report};
            usearch_geospatial_reshard_plan_finish(&arguments);
        });
    }

    /** Writes the index to the checkpoint at @p path, which holds no terrain. */
    answer<completion_, void> save(usearch_bytes_view_t path, progress_t progress = {}) const noexcept {
        return answered_<completion_, void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_geospatial_save_t arguments {index_, executor_, outcome, completion_, path, progress.raw()};
            usearch_geospatial_save(&arguments);
        });
    }

    /** Writes the index into one monolith in @p arena. */
    answer<completion_, usearch_bytes_view_t> save_buffer(arena_t& arena, progress_t progress = {}) const noexcept {
        return answered_<completion_, usearch_bytes_view_t>(
            [&](usearch_outcome_t* outcome, usearch_bytes_view_t* monolith) noexcept {
                usearch_geospatial_save_buffer_t arguments {index_,   executor_,      outcome,       completion_,
                                                            monolith, arena.handle(), progress.raw()};
                usearch_geospatial_save_buffer(&arguments);
            });
    }

#pragma endregion Mutation

#pragma region Reads

    /** Reports the frozen metadata and the current counters. */
    answer<completion_, usearch_geospatial_index_description_t> describe() const noexcept {
        return answered_<completion_, usearch_geospatial_index_description_t>(
            [&](usearch_outcome_t* outcome, usearch_geospatial_index_description_t* description) noexcept {
                usearch_geospatial_describe_t arguments {index_, executor_, outcome, completion_, description};
                usearch_geospatial_describe(&arguments);
            });
    }

    /** Counts the observations stored under @p key. */
    answer<completion_, usearch_geospatial_key_stats_t> count(usearch_key_t key) const noexcept {
        return answered_<completion_, usearch_geospatial_key_stats_t>(
            [&](usearch_outcome_t* outcome, usearch_geospatial_key_stats_t* key_stats) noexcept {
                usearch_geospatial_count_t arguments {index_, executor_, outcome, completion_, key, key_stats};
                usearch_geospatial_count(&arguments);
            });
    }

    /** Reads every observation stored under @p key into @p arena, within the stored rounding. */
    answer<completion_, geospatial_observations_t> get(usearch_key_t key, arena_t& arena) const noexcept {
        return answered_<completion_, geospatial_observations_t>(
            [&](usearch_outcome_t* outcome, geospatial_observations_t* observations) noexcept {
                usearch_geospatial_get_t arguments {
                    index_,        executor_, outcome, completion_, key, &observations->first, &observations->written,
                    arena.handle()};
                usearch_geospatial_get(&arguments);
            });
    }

    /** Audits how much of the graph a search can reach. */
    answer<completion_, usearch_reachability_t> health() const noexcept {
        return answered_<completion_, usearch_reachability_t>(
            [&](usearch_outcome_t* outcome, usearch_reachability_t* reachability) noexcept {
                usearch_geospatial_health_t arguments {index_, executor_, outcome, completion_, reachability};
                usearch_geospatial_health(&arguments);
            });
    }

    /** Draws up to @p wanted keys around @c config.focus, or across the whole index, into @p arena,
     *  each hanging from a shown or earlier draw and placed in three dimensions. */
    answer<completion_, sample_t> sample(std::size_t wanted, arena_t& arena,
                                         sample_config_t const& config = {}) const noexcept {
        return answered_<completion_, sample_t>([&](usearch_outcome_t* outcome, sample_t* sample) noexcept {
            auto arguments = sample_arguments_<usearch_geospatial_sample_t>(index_, executor_, outcome, completion_,
                                                                            wanted, arena, config, sample);
            usearch_geospatial_sample(&arguments);
        });
    }

    /** Finds the @p wanted nearest entries @p relation admits within @p tolerance of every query,
     *  written into @p arena with chords in metres. */
    answer<completion_, usearch_matches_view_t> search(std::span<usearch_geospatial_observation_t const> queries,
                                                       usearch_geospatial_relation_t relation,
                                                       usearch_geospatial_tolerance_t tolerance, std::size_t wanted,
                                                       arena_t& arena,
                                                       geospatial_search_config_t const& config = {}) const noexcept {
        return answered_<completion_, usearch_matches_view_t>(
            [&](usearch_outcome_t* outcome, usearch_matches_view_t* matches) noexcept {
                usearch_geospatial_search_t arguments {
                    .index = index_,
                    .executor = executor_,
                    .outcome = outcome,
                    .completion = completion_,
                    .count = queries.size(),
                    .queries = queries.data(),
                    .relation = relation,
                    .tolerance = tolerance,
                    .matches = matches,
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
    answer<completion_, geospatial_within_t> within(std::span<usearch_geospatial_observation_t const> queries,
                                                    usearch_geospatial_relation_t relation,
                                                    usearch_geospatial_tolerance_t tolerance, arena_t& arena,
                                                    geospatial_within_config_t const& config = {}) const noexcept {
        return answered_<completion_, geospatial_within_t>(
            [&](usearch_outcome_t* outcome, geospatial_within_t* matches) noexcept {
                usearch_geospatial_within_t arguments {
                    .index = index_,
                    .executor = executor_,
                    .outcome = outcome,
                    .completion = completion_,
                    .count = queries.size(),
                    .queries = queries.data(),
                    .relation = relation,
                    .tolerance = tolerance,
                    .matches = &matches->view,
                    .written = &matches->written,
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
    answer<completion_, cursor_batch_t> cursor_next(geospatial_cursor_t const& cursor, std::size_t wanted,
                                                    arena_t& arena, bool links = false) const noexcept {
        return answered_<completion_, cursor_batch_t>([&](usearch_outcome_t* outcome, cursor_batch_t* batch) noexcept {
            usearch_geospatial_cursor_next_t arguments {index_,
                                                        executor_,
                                                        outcome,
                                                        completion_,
                                                        cursor.handle(),
                                                        &batch->keys,
                                                        links ? &batch->links : nullptr,
                                                        &batch->written,
                                                        arena.handle(),
                                                        wanted};
            usearch_geospatial_cursor_next(&arguments);
        });
    }

#pragma endregion Reads

  private:
    friend class geospatial_index_t;

    geospatial_index_on(usearch_geospatial_index_t index, usearch_executor_t executor) noexcept
        : index_(index), executor_(executor) {}

    usearch_geospatial_index_t index_ = nullptr;
    usearch_executor_t executor_ = nullptr;
};

/** Owns one geospatial index; destruction waits for its deferred calls. */
class geospatial_index_t {
  public:
    geospatial_index_t() noexcept = default;
    geospatial_index_t(geospatial_index_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    geospatial_index_t& operator=(geospatial_index_t&& other) noexcept {
        return std::swap(handle_, other.handle_), *this;
    }
    geospatial_index_t(geospatial_index_t const&) = delete;
    geospatial_index_t& operator=(geospatial_index_t const&) = delete;
    ~geospatial_index_t() noexcept { usearch_geospatial_free(handle_); }

    usearch_geospatial_index_t handle() const noexcept { return handle_; }

    /** The output field a lifecycle verb writes the new handle into. */
    usearch_geospatial_index_t* handle_field() noexcept { return &handle_; }

    /** The receiver whose verbs block on @p executor, or on the caller's own thread without one. */
    geospatial_index_on<usearch_completion_blocking_k> on(executor_t const& executor = {}) const noexcept {
        return {handle_, executor.handle()};
    }

    /** The receiver whose verbs submit to @p executor, each answering a @c deferred call. */
    geospatial_index_on<usearch_completion_deferred_k> deferred(executor_t const& executor) const noexcept {
        return {handle_, executor.handle()};
    }

#pragma region Lifecycle

    /** Creates an empty RAM index over the terrain @p options names, if any. */
    static outcome<geospatial_index_t> init_ram(usearch_geospatial_init_ram_t options) noexcept {
        return opened_<geospatial_index_t>(usearch_geospatial_init_ram, options);
    }

    /** Creates an empty VRAM index over the heights @p options points at, if any, on the
     *  devices it names. */
    static outcome<geospatial_index_t> init_vram(usearch_geospatial_init_vram_t options) noexcept {
        return opened_<geospatial_index_t>(usearch_geospatial_init_vram, options);
    }

    /** Copies a checkpoint into a new, mutable index in host memory. */
    static outcome<geospatial_index_t> load_ram(usearch_geospatial_load_ram_t options, executor_t const& executor = {},
                                                progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_<geospatial_index_t>(usearch_geospatial_load_ram, options);
    }

    /** Copies a checkpoint into a new, mutable index in device memory. */
    static outcome<geospatial_index_t> load_vram(usearch_geospatial_load_vram_t options,
                                                 executor_t const& executor = {}, progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_<geospatial_index_t>(usearch_geospatial_load_vram, options);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in host memory. */
    static outcome<geospatial_index_t> load_ram_buffer(usearch_geospatial_load_ram_buffer_t options,
                                                       executor_t const& executor = {},
                                                       progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_<geospatial_index_t>(usearch_geospatial_load_ram_buffer, options);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in device memory. */
    static outcome<geospatial_index_t> load_vram_buffer(usearch_geospatial_load_vram_buffer_t options,
                                                        executor_t const& executor = {},
                                                        progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_<geospatial_index_t>(usearch_geospatial_load_vram_buffer, options);
    }

    /** Serves a checkpoint in place through a memory mapping, as a read-only index. */
    static outcome<geospatial_index_t> view_ram(usearch_geospatial_view_ram_t options) noexcept {
        return opened_<geospatial_index_t>(usearch_geospatial_view_ram, options);
    }

    /** Serves a checkpoint in place from NVMe drives as a read-only index, every later call reading
     *  it through @p executor 's engines. */
    static outcome<geospatial_index_t> view_nvme(usearch_geospatial_view_nvme_t options,
                                                 executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return loaded_<geospatial_index_t>(usearch_geospatial_view_nvme, options);
    }

    /** Reads a checkpoint's connectivity without loading it. */
    static outcome<usearch_geospatial_metadata_t> metadata_read(usearch_geospatial_metadata_read_t arguments,
                                                                executor_t const& executor = {}) noexcept {
        return metadata_read_<usearch_geospatial_metadata_t>(usearch_geospatial_metadata_read, arguments,
                                                             executor.handle());
    }

    /** Reads connectivity from a serialized checkpoint buffer. */
    static outcome<usearch_geospatial_metadata_t> metadata_read_buffer(
        usearch_geospatial_metadata_read_buffer_t arguments, executor_t const& executor = {}) noexcept {
        return metadata_read_<usearch_geospatial_metadata_t>(usearch_geospatial_metadata_read_buffer, arguments,
                                                             executor.handle());
    }

    /** Opens a cursor over every entry, or over the entries @p starts names first. */
    outcome<geospatial_cursor_t> cursor_init(std::span<usearch_key_t const> starts = {}) const noexcept {
        usearch_geospatial_cursor_init_t arguments {handle_, nullptr, nullptr, starts.size(), starts.data(), nullptr};
        return cursor_opened_<geospatial_cursor_t>(usearch_geospatial_cursor_init, arguments);
    }

#pragma endregion Lifecycle

  private:
    usearch_geospatial_index_t handle_ = nullptr;
};

} // namespace unum::usearch
