/**
 *  @file cpp/usearch/shapes_index.hpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief RAII handles and typed verbs over the shapes modality, which add no ABI of their own
 *      and throw nothing at all.
 *
 *  Shapes cross as point tapes: every entry's points end to end, @c x, @c y and @c z interleaved,
 *  around one more offset than entries, counted in points. Each verb that passes or asks for points
 *  names their scalar, as in @c insert<float> and @c get<f16_t>. The receivers, plans and handlers
 *  are the dense ones: @c shapes_index_t::on blocks and answers an @c outcome, while
 *  @c shapes_index_t::deferred submits and answers a @c deferred call. Each verb answers the
 *  statuses its C function lists in @c usearch/shapes_index.h.
 *
 *  @code{.cpp}
 *  namespace us = unum::usearch;
 *  auto [index, status, message] = us::shapes_index_t::init_ram({
 *      .metric = usearch_shapes_metric_hausdorff_k, .quantization = usearch_scalar_f32_k});
 *  float const atoms[2][3] = {{0, 0, 0}, {1.1f, 0, 0}};
 *  std::uint64_t const offsets[] = {0, 2};
 *  auto inserted = index.on(executor).insert<float>(keys, {&atoms[0][0], offsets});
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <span> // `std::span`

#include "usearch/shapes_index.h"
#include "usearch/types.hpp"

namespace unum::usearch {

/** The canonical spelling of a metric kind, as @c usearch_shapes_metric_name answers it. */
inline char const* shapes_metric_name(usearch_shapes_metric_t metric) noexcept {
    return usearch_shapes_metric_name(metric);
}

/** Reads a metric kind spelled @p name back, or fails with @c usearch_status_invalid_argument_k. */
inline outcome<usearch_shapes_metric_t> shapes_metric_from_name(usearch_bytes_view_t name) noexcept {
    return metric_read_(usearch_shapes_metric_from_name, name);
}

/** Owns one cursor, opened by @c shapes_index_t::cursor_init. */
using shapes_cursor_t = cursor<usearch_shapes_cursor_t, usearch_shapes_cursor_free>;

/** A refine in flight, which @c refine opens and continues until @c refine_plan_finish ends. */
using shapes_refine_plan_t = plan<usearch_shapes_refine_plan_t>;

/** A reshard in flight, which @c reshard opens and continues until @c reshard_plan_finish ends. */
using shapes_reshard_plan_t = plan<usearch_shapes_reshard_plan_t>;

/** The knobs of @c refine and the plan it opens or continues. */
using shapes_refine_config_t = refine_config<shapes_refine_plan_t>;

/** The knobs of @c reshard and the plan it opens or continues. */
using shapes_reshard_config_t = reshard_config<shapes_reshard_plan_t>;

/** Coordinates per point. */
inline constexpr std::size_t shapes_point_scalars = 3;

/** The point tapes @c get wrote into its arena as @p element_type_, one entry per shape stored
 *  under the key, each in the pose it was inserted in. */
template <element element_type_>
using shapes_entries = tape_entries<element_type_, shapes_point_scalars>;

/** The knobs of @c search. */
struct shapes_search_config_t {

    /** The breadth of the candidates the walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** A walk or a scan; unknown derives. */
    usearch_traversal_t traversal = usearch_traversal_unknown_k;

    /** Which candidate keys may be reported; empty admits every key. */
    predicate_t predicate {};

    /** Matches of an earlier search to start from; @c nullptr starts where searches start. */
    usearch_matches_view_t const* hints = nullptr;

    /** Hears the queries answered, and may cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

/**
 *  @brief The verbs of a shapes index, completed as @p completion_ says: blocking answers an
 *      @c outcome, deferred answers a @c deferred call. Made by @c shapes_index_t::on and
 *      @c shapes_index_t::deferred.
 *
 *  Each verb is written once, filling its C struct with the result as its output slot. Every
 *  input, handler, plan and arena a deferred call names stays borrowed until it is final.
 */
template <usearch_completion_t completion_>
class shapes_index_on {
  public:
#pragma region Mutation

    /** Grows the shards ahead of @p entries entries and @p points points in all. */
    answer<completion_, void> reserve(std::size_t entries, std::size_t points = 0) const noexcept {
        return answered_<completion_, void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_shapes_reserve_t arguments {index_, executor_, outcome, completion_, entries, points};
            usearch_shapes_reserve(&arguments);
        });
    }

    /** Erases all entries, keeping the reserved capacity and frozen metric. */
    answer<completion_, void> clear() const noexcept {
        return answered_<completion_, void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_shapes_clear_t arguments {index_, executor_, outcome, completion_};
            usearch_shapes_clear(&arguments);
        });
    }

    /** Adds @p shapes under @p keys, one key per entry, all or nothing per call. */
    template <element element_type_>
    answer<completion_, void> insert(std::span<usearch_key_t const> keys, tape_view<element_type_> shapes,
                                     insert_config_t const& config = {}) const noexcept {
        // The call has one count, so only this side sees the keys and the tape disagree.
        if (keys.size() != shapes.count)
            return refused_<completion_, void>(usearch_status_shape_mismatch_k,
                                               "The key count differs from the shape count");
        return answered_<completion_, void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_shapes_insert_t arguments {
                .index = index_,
                .executor = executor_,
                .outcome = outcome,
                .completion = completion_,
                .count = keys.size(),
                .keys = {keys.data(), sizeof(usearch_key_t)},
                .shapes = shapes.view(),
                .expansion_factor = config.expansion_factor,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            usearch_shapes_insert(&arguments);
        });
    }

    /** Runs one refine round, or continues @c config.plan, answering the round's report. */
    answer<completion_, usearch_refine_report_t> refine(shapes_refine_config_t const& config = {}) const noexcept {
        return answered_<completion_, usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_shapes_refine_t arguments {
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
                usearch_shapes_refine(&arguments);
            });
    }

    /** Closes and frees @p plan, opened on this index and executor, answering all it did. */
    answer<completion_, usearch_refine_report_t> refine_plan_finish(shapes_refine_plan_t plan) const noexcept {
        return answered_<completion_, usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_shapes_refine_plan_finish_t arguments {index_,      executor_,     outcome,
                                                               completion_, plan.handle(), report};
                usearch_shapes_refine_plan_finish(&arguments);
            });
    }

    /** Moves the entries of @p sources into @p targets with this receiver's executor, answering the
     *  report of the closing rounds; every handle named outlives the call and the plan's finish. */
    answer<completion_, usearch_refine_report_t> reshard(std::span<usearch_shapes_index_t const> sources,
                                                         std::span<usearch_shapes_index_t const> targets,
                                                         shapes_reshard_config_t const& config = {}) const noexcept {
        return answered_<completion_, usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_shapes_reshard_t arguments {
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
                usearch_shapes_reshard(&arguments);
            });
    }

    /** Closes and frees @p plan, opened with this executor, answering what it did in all. */
    answer<completion_, usearch_refine_report_t> reshard_plan_finish(shapes_reshard_plan_t plan) const noexcept {
        return answered_<completion_, usearch_refine_report_t>(
            [&](usearch_outcome_t* outcome, usearch_refine_report_t* report) noexcept {
                usearch_shapes_reshard_plan_finish_t arguments {executor_, outcome, completion_, plan.handle(), report};
                usearch_shapes_reshard_plan_finish(&arguments);
            });
    }

    /** Writes the index to the checkpoint at @p path. */
    answer<completion_, void> save(usearch_bytes_view_t path, progress_t progress = {}) const noexcept {
        return answered_<completion_, void>([&](usearch_outcome_t* outcome, auto*) noexcept {
            usearch_shapes_save_t arguments {index_, executor_, outcome, completion_, path, progress.raw()};
            usearch_shapes_save(&arguments);
        });
    }

    /** Writes the index into one monolith in @p arena. */
    answer<completion_, usearch_bytes_view_t> save_buffer(arena_t& arena, progress_t progress = {}) const noexcept {
        return answered_<completion_, usearch_bytes_view_t>(
            [&](usearch_outcome_t* outcome, usearch_bytes_view_t* monolith) noexcept {
                usearch_shapes_save_buffer_t arguments {index_,   executor_,      outcome,       completion_,
                                                        monolith, arena.handle(), progress.raw()};
                usearch_shapes_save_buffer(&arguments);
            });
    }

#pragma endregion Mutation

#pragma region Reads

    /** Reports the frozen metadata and the current counters. */
    answer<completion_, usearch_shapes_index_description_t> describe() const noexcept {
        return answered_<completion_, usearch_shapes_index_description_t>(
            [&](usearch_outcome_t* outcome, usearch_shapes_index_description_t* description) noexcept {
                usearch_shapes_describe_t arguments {index_, executor_, outcome, completion_, description};
                usearch_shapes_describe(&arguments);
            });
    }

    /** Counts the entries stored under @p key, with their total and largest point counts. */
    answer<completion_, usearch_shapes_key_stats_t> count(usearch_key_t key) const noexcept {
        return answered_<completion_, usearch_shapes_key_stats_t>(
            [&](usearch_outcome_t* outcome, usearch_shapes_key_stats_t* key_stats) noexcept {
                usearch_shapes_count_t arguments {index_, executor_, outcome, completion_, key, key_stats};
                usearch_shapes_count(&arguments);
            });
    }

    /** Reads every shape stored under @p key into @p arena, one tape of @p element_type_ points. */
    template <element element_type_>
    answer<completion_, shapes_entries<element_type_>> get(usearch_key_t key, arena_t& arena) const noexcept {
        return answered_<completion_, shapes_entries<element_type_>>(
            [&](usearch_outcome_t* outcome, shapes_entries<element_type_>* entries) noexcept {
                usearch_shapes_get_t arguments {index_,
                                                executor_,
                                                outcome,
                                                completion_,
                                                key,
                                                &entries->view,
                                                &entries->written,
                                                arena.handle(),
                                                scalar_of<element_type_>};
                usearch_shapes_get(&arguments);
            });
    }

    /** Audits how much of the graph a search can reach. */
    answer<completion_, usearch_reachability_t> health() const noexcept {
        return answered_<completion_, usearch_reachability_t>(
            [&](usearch_outcome_t* outcome, usearch_reachability_t* reachability) noexcept {
                usearch_shapes_health_t arguments {index_, executor_, outcome, completion_, reachability};
                usearch_shapes_health(&arguments);
            });
    }

    /** Draws up to @p wanted keys around @c config.focus, or across the whole index, into @p arena,
     *  each hanging from a shown or earlier draw and placed in three dimensions. */
    answer<completion_, sample_t> sample(std::size_t wanted, arena_t& arena,
                                         sample_config_t const& config = {}) const noexcept {
        return answered_<completion_, sample_t>([&](usearch_outcome_t* outcome, sample_t* sample) noexcept {
            auto arguments = sample_arguments_<usearch_shapes_sample_t>(index_, executor_, outcome, completion_, wanted,
                                                                        arena, config, sample);
            usearch_shapes_sample(&arguments);
        });
    }

    /** Finds the @p wanted nearest entries of every shape of @p queries, written into @p arena. */
    template <element element_type_>
    answer<completion_, usearch_matches_view_t> search(tape_view<element_type_> queries, std::size_t wanted,
                                                       arena_t& arena,
                                                       shapes_search_config_t const& config = {}) const noexcept {
        return answered_<completion_, usearch_matches_view_t>(
            [&](usearch_outcome_t* outcome, usearch_matches_view_t* matches) noexcept {
                usearch_shapes_search_t arguments {
                    .index = index_,
                    .executor = executor_,
                    .outcome = outcome,
                    .completion = completion_,
                    .count = queries.count,
                    .queries = queries.view(),
                    .matches = matches,
                    .arena = arena.handle(),
                    .wanted = wanted,
                    .expansion_factor = config.expansion_factor,
                    .traversal = config.traversal,
                    .predicate = config.predicate.raw(),
                    .hints = config.hints,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                usearch_shapes_search(&arguments);
            });
    }

    /** Pulls the next @p wanted entries @p cursor walks to into @p arena, with their links when
     *  @p links asks. */
    answer<completion_, cursor_batch_t> cursor_next(shapes_cursor_t const& cursor, std::size_t wanted, arena_t& arena,
                                                    bool links = false) const noexcept {
        return answered_<completion_, cursor_batch_t>([&](usearch_outcome_t* outcome, cursor_batch_t* batch) noexcept {
            usearch_shapes_cursor_next_t arguments {index_,
                                                    executor_,
                                                    outcome,
                                                    completion_,
                                                    cursor.handle(),
                                                    &batch->keys,
                                                    links ? &batch->links : nullptr,
                                                    &batch->written,
                                                    arena.handle(),
                                                    wanted};
            usearch_shapes_cursor_next(&arguments);
        });
    }

#pragma endregion Reads

  private:
    friend class shapes_index_t;

    shapes_index_on(usearch_shapes_index_t index, usearch_executor_t executor) noexcept
        : index_(index), executor_(executor) {}

    usearch_shapes_index_t index_ = nullptr;
    usearch_executor_t executor_ = nullptr;
};

/** Owns one shapes index; destruction waits for its deferred calls. */
class shapes_index_t {
  public:
    shapes_index_t() noexcept = default;
    shapes_index_t(shapes_index_t&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    shapes_index_t& operator=(shapes_index_t&& other) noexcept { return std::swap(handle_, other.handle_), *this; }
    shapes_index_t(shapes_index_t const&) = delete;
    shapes_index_t& operator=(shapes_index_t const&) = delete;
    ~shapes_index_t() noexcept { usearch_shapes_free(handle_); }

    usearch_shapes_index_t handle() const noexcept { return handle_; }

    /** The output field a lifecycle verb writes the new handle into. */
    usearch_shapes_index_t* handle_field() noexcept { return &handle_; }

    /** The receiver whose verbs block on @p executor, or on the caller's own thread without one. */
    shapes_index_on<usearch_completion_blocking_k> on(executor_t const& executor = {}) const noexcept {
        return {handle_, executor.handle()};
    }

    /** The receiver whose verbs submit to @p executor, each answering a @c deferred call. */
    shapes_index_on<usearch_completion_deferred_k> deferred(executor_t const& executor) const noexcept {
        return {handle_, executor.handle()};
    }

#pragma region Lifecycle

    /** Creates an empty RAM index of the metric and quantization @p options freezes. */
    static outcome<shapes_index_t> init_ram(usearch_shapes_init_ram_t options) noexcept {
        return opened_<shapes_index_t>(usearch_shapes_init_ram, options);
    }

    /** Creates an empty VRAM index of the metric and quantization @p options freezes, on the
     *  devices those options name. */
    static outcome<shapes_index_t> init_vram(usearch_shapes_init_vram_t options) noexcept {
        return opened_<shapes_index_t>(usearch_shapes_init_vram, options);
    }

    /** Copies a checkpoint into a new, mutable index in host memory. */
    static outcome<shapes_index_t> load_ram(usearch_shapes_load_ram_t options, executor_t const& executor = {},
                                            progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_<shapes_index_t>(usearch_shapes_load_ram, options);
    }

    /** Copies a checkpoint into a new, mutable index in device memory. */
    static outcome<shapes_index_t> load_vram(usearch_shapes_load_vram_t options, executor_t const& executor = {},
                                             progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_<shapes_index_t>(usearch_shapes_load_vram, options);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in host memory. */
    static outcome<shapes_index_t> load_ram_buffer(usearch_shapes_load_ram_buffer_t options,
                                                   executor_t const& executor = {}, progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_<shapes_index_t>(usearch_shapes_load_ram_buffer, options);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in device memory. */
    static outcome<shapes_index_t> load_vram_buffer(usearch_shapes_load_vram_buffer_t options,
                                                    executor_t const& executor = {},
                                                    progress_t progress = {}) noexcept {
        options.executor = executor.handle(), options.progress = progress.raw();
        return loaded_<shapes_index_t>(usearch_shapes_load_vram_buffer, options);
    }

    /** Serves a checkpoint in place through a memory mapping, as a read-only index. */
    static outcome<shapes_index_t> view_ram(usearch_shapes_view_ram_t options) noexcept {
        return opened_<shapes_index_t>(usearch_shapes_view_ram, options);
    }

    /** Serves a checkpoint in place from NVMe drives as a read-only index, every later call reading
     *  it through @p executor 's engines. */
    static outcome<shapes_index_t> view_nvme(usearch_shapes_view_nvme_t options,
                                             executor_t const& executor = {}) noexcept {
        options.executor = executor.handle();
        return loaded_<shapes_index_t>(usearch_shapes_view_nvme, options);
    }

    /** Reads a checkpoint's metric, quantization, points & connectivity without loading it. */
    static outcome<usearch_shapes_metadata_t> metadata_read(usearch_shapes_metadata_read_t arguments,
                                                            executor_t const& executor = {}) noexcept {
        return metadata_read_<usearch_shapes_metadata_t>(usearch_shapes_metadata_read, arguments, executor.handle());
    }

    /** Reads metric, quantization, points & connectivity from a serialized checkpoint buffer. */
    static outcome<usearch_shapes_metadata_t> metadata_read_buffer(usearch_shapes_metadata_read_buffer_t arguments,
                                                                   executor_t const& executor = {}) noexcept {
        return metadata_read_<usearch_shapes_metadata_t>(usearch_shapes_metadata_read_buffer, arguments,
                                                         executor.handle());
    }

    /** Opens a cursor over every entry, or over the entries @p starts names first. */
    outcome<shapes_cursor_t> cursor_init(std::span<usearch_key_t const> starts = {}) const noexcept {
        usearch_shapes_cursor_init_t arguments {handle_, nullptr, nullptr, starts.size(), starts.data(), nullptr};
        return cursor_opened_<shapes_cursor_t>(usearch_shapes_cursor_init, arguments);
    }

#pragma endregion Lifecycle

  private:
    usearch_shapes_index_t handle_ = nullptr;
};

} // namespace unum::usearch
