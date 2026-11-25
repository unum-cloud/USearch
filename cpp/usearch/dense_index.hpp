/**
 *  @file cpp/usearch/dense_index.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief RAII handles and typed verbs over the dynamically dispatched dense modality, adding no
 *      ABI of its own and throwing nothing.
 *
 *  An index answers its verbs through a receiver, whose type picks how they complete: the index
 *  itself blocks on the calling thread and @c on blocks on an executor, both answering a
 *  @c status_t or an @c expected, while @c deferred submits to an executor and answers a
 *  @c deferred call its one owner polls, waits on or awaits. Each verb names the scalar of the
 *  matrix it passes, as in @c search<float>, and takes its handlers as @c noexcept callables it
 *  calls only during the call. Each verb answers the statuses its C function lists in
 *  @c usearch/dense_index.h.
 *
 *  @code{.cpp}
 *  namespace us = unum::usearch;
 *  auto [executor, executor_status] = us::executor_t::for_ram();
 *  auto [index, index_status] = us::dense_index_t::init_ram({
 *      .metric = usearch_dense_metric_angular_k, .quantization = usearch_scalar_bf16_k,
 *      .dimensions = 256});
 *  auto [arena, arena_status] = us::arena_t::make({.executor = executor.handle()});
 *  auto [matches, status] = index.on(executor).search<float>({query, 1, 256}, 10, arena);
 *  @endcode
 */
#pragma once

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`

#include <span>        // `std::span`
#include <type_traits> // `std::is_nothrow_invocable_r_v`

#include "usearch/dense_index.h" // `usearch_dense_*` verbs and their argument structs
#include "usearch/types.hpp"     // `deferrable`, `answered_`, `opened_`, `owner`, `plan`, `executor_t`, `arena_t`

namespace unum::usearch {

/** The canonical spelling of a metric kind, as @c usearch_dense_metric_name answers it. */
inline char const* dense_metric_name(usearch_dense_metric_t metric) noexcept {
    return usearch_dense_metric_name(metric);
}

/** Reads a metric kind spelled @p name back, or fails with @c status_code_t::invalid_argument_k. */
inline expected<usearch_dense_metric_t> dense_metric_from_name(usearch_bytes_view_t name) noexcept {
    return metric_read_(usearch_dense_metric_from_name, name);
}

#pragma region Outputs

/** Rows a call wrote packed into its arena as @p element_type_, readable until the arena's next
 *  call: the entries @c get found under a key, or the outputs of @c attend. */
template <an_element element_type_>
struct dense_vectors {

    /** The rows as the call wrote them. */
    usearch_rows_view_t view {};

    /** Rows written. */
    std::size_t written = 0;

    /** The row at @p position, below @c written. */
    element_type_ const* row(std::size_t position) const noexcept {
        return reinterpret_cast<element_type_ const*>(static_cast<std::uint8_t const*>(view.data) +
                                                      position * view.stride);
    }
};

/** The ragged products of @c multiply or the matches of @c maxsim, laid out in the arena. */
struct dense_products_t {

    /** Keys, scores and offsets, end to end. */
    usearch_matches_tape_view_t view {};

    /** Pairs written across every row or query. */
    std::size_t written = 0;

    /** Pairs scored before @c wanted cut any. */
    std::size_t required = 0;
};

#pragma endregion Outputs

#pragma region Handlers

/** A value lookup handler: points each key at one row of the columns and scalar asked for. */
template <typename handler_type_>
concept a_value_lookup_handler = std::is_nothrow_invocable_r_v<void, handler_type_&, usearch_key_t const*, std::size_t,
                                                               usearch_scalar_t, std::size_t, void const**>;

/** The value rows @c attend gathers: a host handler read only during the call, or a device
 *  function. */
class value_lookup_t {
  public:
    value_lookup_t() noexcept = default;

    /** Resolves keys on the host through @p handler, for a RAM handle. */
    template <a_value_lookup_handler handler_type_>
    value_lookup_t(handler_type_& handler) noexcept
        : function_(&trampoline_<handler_type_>), handler_state_(&handler) {}

    /** A device function and its device state for a VRAM handle, untyped as the host sees them. */
    static value_lookup_t device(usearch_value_lookup_t function, void* state) noexcept {
        value_lookup_t lookup;
        lookup.function_ = function;
        lookup.handler_state_ = state;
        return lookup;
    }

    usearch_value_lookup_t function() const noexcept { return function_; }
    void* state() const noexcept { return handler_state_; }

  private:
    template <typename handler_type_>
    static void trampoline_(void* state, usearch_key_t const* keys, std::size_t count, usearch_scalar_t scalar,
                            std::size_t columns, void const** rows) noexcept {
        (*static_cast<handler_type_*>(state))(keys, count, scalar, columns, rows);
    }

    usearch_value_lookup_t function_ = nullptr;
    void* handler_state_ = nullptr;
};

#pragma endregion Handlers

/** Owns one cursor, opened by @c dense_index_t::cursor_init. */
using dense_cursor_t = owner<usearch_dense_cursor_t, usearch_dense_cursor_free>;

/** Owns one search stream, opened by @c dense_index_t::search_stream_init. */
using dense_search_stream_t = owner<usearch_dense_search_stream_t, usearch_dense_search_stream_free>;

/** A refine in flight, which @c refine opens and continues until @c refine_plan_finish ends. */
using dense_refine_plan_t = plan<usearch_dense_refine_plan_t>;

/** A reshard in flight, which @c reshard opens and continues until @c reshard_plan_finish ends. */
using dense_reshard_plan_t = plan<usearch_dense_reshard_plan_t>;

#pragma region Configs

/** The knobs of @c refine and the plan it opens or continues. */
using dense_refine_config_t = refine_config<dense_refine_plan_t>;

/** The placement @c reshard recuts toward, the knobs of its closing rounds and its plan. */
struct dense_reshard_config_t {

    /** The placement of a RAM handle recut in place; @c nullptr keeps it. */
    usearch_ram_options_t const* ram = nullptr;

    /** The placement of a VRAM handle recut in place; @c nullptr keeps it. */
    usearch_vram_options_t const* vram = nullptr;

    /** The placement of an NVMe handle recut in place; @c nullptr keeps it. */
    usearch_nvme_options_t const* nvme = nullptr;

    /** The walk breadth and the bytes a loaded NVMe handle holds per part; zero derives each. */
    usearch_refine_options_t refine {};

    /** Picks the part roots of a split: equal seeds over an unchanged source split equally. */
    std::uint64_t seed = 0;

    /** Hears every barrier of the closing rounds, and may cancel them there, never a move. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;

    /** The plan a cancel leaves live; @c nullptr moves and runs to the end. */
    dense_reshard_plan_t* plan = nullptr;
};

/** The knobs of @c search. */
struct dense_search_config_t {

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

/** The knobs of @c multiply. */
struct dense_multiply_config_t {

    /** Pairs kept per left row, the first ones its walk scored; zero keeps every pair. */
    std::size_t wanted = 0;

    /** The breadth of the candidates the walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** A walk or a scan; unknown derives. */
    usearch_traversal_t traversal = usearch_traversal_unknown_k;

    /** Which candidate keys may be reported; empty admits every key. */
    predicate_t predicate {};

    /** Hears the rows answered, and may cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

/** The knobs of @c attend. */
struct dense_attend_config_t {

    /** Query heads sharing the index's key-value head, dividing the row count; zero means one. */
    std::size_t head_count = 0;

    /** One online-softmax state per query row, passed by the caller; @c nullptr skips it. */
    usearch_dense_softmax_state_t* carry = nullptr;

    /** The breadth of the candidates the walk explores; zero derives. */
    std::size_t expansion_factor = 0;

    /** The logit scale; zero derives it from the frozen geometry. */
    float score_scale = 0;

    /** A walk or a scan; unknown derives. */
    usearch_traversal_t traversal = usearch_traversal_unknown_k;

    /** Which candidate keys may be attended; empty admits every key. */
    predicate_t predicate {};

    /** Hears the rows answered, and may cancel. */
    progress_t progress {};

    /** The traversal telemetry; @c nullptr skips reporting. */
    usearch_traversal_stats_t* stats = nullptr;
};

/** The knobs of @c maxsim. */
struct dense_maxsim_config_t {

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

#pragma endregion Configs

/**
 *  @brief A dense index and its verbs, completed as @p receiver_type_ says: the owning handle
 *      blocks on the calling thread, @c on blocks on an executor, and both answer a @c status_t or
 *      an @c expected, while @c deferred submits to an executor and answers a @c deferred call.
 *
 *  Each verb is written once, filling its C struct with the call's @c outputs.
 *  Every input, handler, plan and arena a deferred call names must stay allocated and unchanged
 *  until the call is final.
 */
template <typename receiver_type_>
class dense_index {
    static constexpr usearch_completion_t completion_k = receiver_type_::completion;
    template <typename value_type_ = void>
    using deferrable = usearch::deferrable<value_type_, completion_k>;
    receiver_type_ receiver_;

    template <typename>
    friend class dense_index;
    explicit dense_index(receiver_type_ receiver) noexcept : receiver_(std::move(receiver)) {}

  public:
    dense_index() noexcept
        requires a_handle_owner<receiver_type_>
    = default;

    usearch_dense_index_t handle() const noexcept { return receiver_.handle(); }

    /** The output field a lifecycle verb writes the new handle into. */
    usearch_dense_index_t* handle_field() noexcept
        requires a_handle_owner<receiver_type_>
    {
        return receiver_.handle_field();
    }

    /** The same handle, whose verbs block on @p executor. */
    dense_index<caller_owned<usearch_dense_index_t, usearch_completion_blocking_k>> on(
        executor_t const& executor) const noexcept {
        return dense_index<caller_owned<usearch_dense_index_t, usearch_completion_blocking_k>>(
            {receiver_.handle(), executor.handle()});
    }

    /** The same handle, whose verbs submit to @p executor, each answering a @c deferred call. */
    dense_index<caller_owned<usearch_dense_index_t, usearch_completion_deferred_k>> deferred(
        executor_t const& executor) const noexcept {
        return dense_index<caller_owned<usearch_dense_index_t, usearch_completion_deferred_k>>(
            {receiver_.handle(), executor.handle()});
    }

#pragma region Mutation

    /** Grows the shards ahead of @p capacity entries in all. */
    deferrable<> reserve(std::size_t capacity) const noexcept {
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_dense_reserve_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k,
                                               capacity};
            usearch_dense_reserve(&arguments);
        });
    }

    /** Erases all entries, keeping the reserved capacity and frozen geometry. */
    deferrable<> clear() const noexcept {
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_dense_clear_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k};
            usearch_dense_clear(&arguments);
        });
    }

    /** Adds @p vectors under @p keys, one key per row, all or nothing per call. */
    template <an_element element_type_>
    deferrable<> insert(std::span<usearch_key_t const> keys, rows_view<element_type_> vectors,
                        insert_config_t const& config = {}) const noexcept {
        // The call has one count, so only this side sees the keys and the rows disagree.
        if (keys.size() != vectors.count)
            return answered_<deferrable<>>(
                {status_code_t::shape_mismatch_k, "The key count differs from the row count"});
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_dense_insert_t arguments {
                .index = receiver_.handle(),
                .executor = receiver_.executor(),
                .outcome = outcome,
                .completion = completion_k,
                .count = keys.size(),
                .keys = {keys.data(), sizeof(usearch_key_t)},
                .vectors = vectors.view(),
                .expansion_factor = config.expansion_factor,
                .progress = config.progress.raw(),
                .stats = config.stats,
            };
            usearch_dense_insert(&arguments);
        });
    }

    /** Runs one refine round, or continues @c config.plan, answering the report: entries
     *  re-selected and changed, islands found and reconnected, neighborhoods, links added and
     *  dropped, the verdict @c usearch_refine_verdict_name spells, and both health checks. */
    deferrable<usearch_refine_report_t> refine(dense_refine_config_t const& config = {}) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_dense_refine_t arguments {
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
                usearch_dense_refine(&arguments);
            });
    }

    /** Closes and frees @p plan, opened on this index and executor, answering all it did. */
    deferrable<usearch_refine_report_t> refine_plan_finish(dense_refine_plan_t plan) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_dense_refine_plan_finish_t arguments {
                    receiver_.handle(), receiver_.executor(), outcome, completion_k, plan.handle(), &written->value};
                usearch_dense_refine_plan_finish(&arguments);
            });
    }

    /** Moves the entries of @p sources into @p targets with this receiver's executor, answering the
     *  report of the closing rounds; every handle named outlives the call and the plan's finish. */
    deferrable<usearch_refine_report_t> reshard(std::span<usearch_dense_index_t const> sources,
                                                std::span<usearch_dense_index_t const> targets,
                                                dense_reshard_config_t const& config = {}) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_dense_reshard_t arguments {
                    .sources_count = sources.size(),
                    .sources = sources.data(),
                    .targets_count = targets.size(),
                    .targets = targets.data(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .ram = config.ram,
                    .vram = config.vram,
                    .nvme = config.nvme,
                    .report = &written->value,
                    .refine = config.refine,
                    .seed = config.seed,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                    .plan = handle_field_of(config.plan),
                    .phase = phase_field_of(config.plan),
                };
                usearch_dense_reshard(&arguments);
            });
    }

    /** Closes and frees @p plan, opened with this executor, answering what it did in all. */
    deferrable<usearch_refine_report_t> reshard_plan_finish(dense_reshard_plan_t plan) const noexcept {
        return answered_<deferrable<usearch_refine_report_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_refine_report_t>* written) noexcept {
                usearch_dense_reshard_plan_finish_t arguments {receiver_.executor(), outcome, completion_k,
                                                               plan.handle(), &written->value};
                usearch_dense_reshard_plan_finish(&arguments);
            });
    }

    /** Writes the index to the checkpoint at @p path. */
    deferrable<> save(usearch_bytes_view_t path, progress_t progress = {}) const noexcept {
        return answered_<deferrable<>>([&](usearch_outcome_t* outcome, outputs<void>*) noexcept {
            usearch_dense_save_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k, path,
                                            progress.raw()};
            usearch_dense_save(&arguments);
        });
    }

    /** Writes the index into one monolith in @p arena. */
    deferrable<usearch_bytes_view_t> save_buffer(arena_t& arena, progress_t progress = {}) const noexcept {
        return answered_<deferrable<usearch_bytes_view_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_bytes_view_t>* written) noexcept {
                usearch_dense_save_buffer_t arguments {receiver_.handle(), receiver_.executor(), outcome,
                                                       completion_k,       &written->value,      arena.handle(),
                                                       progress.raw()};
                usearch_dense_save_buffer(&arguments);
            });
    }

#pragma endregion Mutation

#pragma region Reads

    /** Reports the frozen metadata and the current counters. */
    deferrable<usearch_dense_index_description_t> describe() const noexcept {
        return answered_<deferrable<usearch_dense_index_description_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_dense_index_description_t>* written) noexcept {
                usearch_dense_describe_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k,
                                                    &written->value};
                usearch_dense_describe(&arguments);
            });
    }

    /** Counts the entries stored under @p key. */
    deferrable<std::size_t> count(usearch_key_t key) const noexcept {
        return answered_<deferrable<std::size_t>>(
            [&](usearch_outcome_t* outcome, outputs<std::size_t>* written) noexcept {
                usearch_dense_count_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k, key,
                                                 &written->value};
                usearch_dense_count(&arguments);
            });
    }

    /** Reads every vector stored under @p key into @p arena as @p element_type_ rows. */
    template <an_element element_type_>
    deferrable<dense_vectors<element_type_>> get(usearch_key_t key, arena_t& arena) const noexcept {
        return answered_<deferrable<dense_vectors<element_type_>>>(
            [&](usearch_outcome_t* outcome, outputs<dense_vectors<element_type_>>* written) noexcept {
                usearch_dense_get_t arguments {receiver_.handle(),
                                               receiver_.executor(),
                                               outcome,
                                               completion_k,
                                               key,
                                               &written->value.view,
                                               &written->value.written,
                                               arena.handle(),
                                               scalar_of<element_type_>};
                usearch_dense_get(&arguments);
            });
    }

    /** Audits how much of the graph a search can reach. */
    deferrable<usearch_reachability_t> health() const noexcept {
        return answered_<deferrable<usearch_reachability_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_reachability_t>* written) noexcept {
                usearch_dense_health_t arguments {receiver_.handle(), receiver_.executor(), outcome, completion_k,
                                                  &written->value};
                usearch_dense_health(&arguments);
            });
    }

    /** Draws up to @p wanted keys around @c config.focus, or across the whole index, into @p arena,
     *  each hanging from a shown or earlier draw and placed in three dimensions. */
    deferrable<sample_t> sample(std::size_t wanted, arena_t& arena, sample_config_t const& config = {}) const noexcept {
        return answered_<deferrable<sample_t>>([&](usearch_outcome_t* outcome, outputs<sample_t>* written) noexcept {
            usearch_dense_sample_t arguments = sample_arguments_<usearch_dense_sample_t>(
                receiver_.handle(), receiver_.executor(), outcome, completion_k, wanted, arena, config,
                &written->value);
            usearch_dense_sample(&arguments);
        });
    }

    /** Finds the @p wanted nearest entries of every row of @p queries, written into @p arena. */
    template <an_element element_type_>
    deferrable<usearch_matches_view_t> search(rows_view<element_type_> queries, std::size_t wanted, arena_t& arena,
                                              dense_search_config_t const& config = {}) const noexcept {
        return answered_<deferrable<usearch_matches_view_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_matches_view_t>* written) noexcept {
                usearch_dense_search_t arguments {
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
                    .hints = config.hints,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                usearch_dense_search(&arguments);
            });
    }

    /** Scores every row of @p left against the index, written into @p arena. */
    template <an_element element_type_>
    deferrable<dense_products_t> multiply(rows_view<element_type_> left, arena_t& arena,
                                          dense_multiply_config_t const& config = {}) const noexcept {
        return answered_<deferrable<dense_products_t>>(
            [&](usearch_outcome_t* outcome, outputs<dense_products_t>* written) noexcept {
                usearch_dense_multiply_t arguments {
                    .index = receiver_.handle(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .count = left.count,
                    .left = left.view(),
                    .products = &written->value.view,
                    .written = &written->value.written,
                    .required = &written->value.required,
                    .arena = arena.handle(),
                    .wanted = config.wanted,
                    .expansion_factor = config.expansion_factor,
                    .traversal = config.traversal,
                    .predicate = config.predicate.raw(),
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                usearch_dense_multiply(&arguments);
            });
    }

    /** Attends every row of @p queries over the values @p value_lookup resolves, writing one
     *  @p output_type_ row per query into @p arena. */
    template <an_element output_type_, an_element element_type_>
    deferrable<dense_vectors<output_type_>> attend(rows_view<element_type_> queries, value_lookup_t value_lookup,
                                                   arena_t& arena,
                                                   dense_attend_config_t const& config = {}) const noexcept {
        return answered_<deferrable<dense_vectors<output_type_>>>(
            [&](usearch_outcome_t* outcome, outputs<dense_vectors<output_type_>>* written) noexcept {
                usearch_dense_attend_t arguments {
                    .index = receiver_.handle(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .count = queries.count,
                    .queries = queries.view(),
                    .head_count = config.head_count,
                    .value_lookup = value_lookup.function(),
                    .value_lookup_state = value_lookup.state(),
                    .carry = config.carry,
                    .outputs = &written->value.view,
                    .arena = arena.handle(),
                    .outputs_scalar = scalar_of<output_type_>,
                    .expansion_factor = config.expansion_factor,
                    .score_scale = config.score_scale,
                    .traversal = config.traversal,
                    .predicate = config.predicate.raw(),
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                written->value.written = queries.count;
                usearch_dense_attend(&arguments);
            });
    }

    /** Scores groups of @p rows_per_query rows of @p queries by late interaction, keeping the
     *  @p wanted best keys per query in @p arena. */
    template <an_element element_type_>
    deferrable<dense_products_t> maxsim(rows_view<element_type_> queries, std::size_t rows_per_query,
                                        std::size_t wanted, arena_t& arena,
                                        dense_maxsim_config_t const& config = {}) const noexcept {
        // The call counts queries, so only this side sees the rows fail to divide into them.
        if (rows_per_query == 0 || queries.count % rows_per_query != 0)
            return answered_<deferrable<dense_products_t>>(
                {status_code_t::shape_mismatch_k, "The rows do not divide into whole queries"});
        return answered_<deferrable<dense_products_t>>(
            [&](usearch_outcome_t* outcome, outputs<dense_products_t>* written) noexcept {
                usearch_dense_maxsim_t arguments {
                    .index = receiver_.handle(),
                    .executor = receiver_.executor(),
                    .outcome = outcome,
                    .completion = completion_k,
                    .count = queries.count / rows_per_query,
                    .queries = queries.view(),
                    .rows_per_query = rows_per_query,
                    .matches = &written->value.view,
                    .written = &written->value.written,
                    .required = &written->value.required,
                    .arena = arena.handle(),
                    .wanted = wanted,
                    .expansion_factor = config.expansion_factor,
                    .traversal = config.traversal,
                    .predicate = config.predicate.raw(),
                    .hints = config.hints,
                    .progress = config.progress.raw(),
                    .stats = config.stats,
                };
                usearch_dense_maxsim(&arguments);
            });
    }

    /** Pulls the next @p wanted entries @p cursor walks to into @p arena, with their links when
     *  @p links asks. */
    deferrable<cursor_batch_t> cursor_next(dense_cursor_t const& cursor, std::size_t wanted, arena_t& arena,
                                           bool links = false) const noexcept {
        return answered_<deferrable<cursor_batch_t>>(
            [&](usearch_outcome_t* outcome, outputs<cursor_batch_t>* written) noexcept {
                usearch_dense_cursor_next_t arguments {receiver_.handle(),
                                                       receiver_.executor(),
                                                       outcome,
                                                       completion_k,
                                                       cursor.handle(),
                                                       &written->value.keys,
                                                       links ? &written->value.links : nullptr,
                                                       &written->value.written,
                                                       arena.handle(),
                                                       wanted};
                usearch_dense_cursor_next(&arguments);
            });
    }

    /** Answers the next @p wanted matches of @p stream, written into @p arena. */
    deferrable<usearch_matches_view_t> search_stream_next(dense_search_stream_t const& stream, std::size_t wanted,
                                                          arena_t& arena) const noexcept {
        return answered_<deferrable<usearch_matches_view_t>>(
            [&](usearch_outcome_t* outcome, outputs<usearch_matches_view_t>* written) noexcept {
                usearch_dense_search_stream_next_t arguments {
                    receiver_.handle(), receiver_.executor(), outcome,        completion_k,
                    stream.handle(),    &written->value,      arena.handle(), wanted};
                usearch_dense_search_stream_next(&arguments);
            });
    }

#pragma endregion Reads

#pragma region Lifecycle

    /** Creates an empty RAM index of the geometry @p arguments freezes. */
    static expected<dense_index> init_ram(usearch_dense_init_ram_t arguments) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<dense_index>(usearch_dense_init_ram, arguments, &usearch_dense_init_ram_t::index);
    }

    /** Creates an empty VRAM index of the geometry @p arguments freezes, on the devices named. */
    static expected<dense_index> init_vram(usearch_dense_init_vram_t arguments) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<dense_index>(usearch_dense_init_vram, arguments, &usearch_dense_init_vram_t::index);
    }

    /** Copies a checkpoint into a new, mutable index in host memory. */
    static expected<dense_index> load_ram(usearch_dense_load_ram_t arguments, executor_t const& executor = {},
                                          progress_t progress = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<dense_index>(usearch_dense_load_ram, arguments, &usearch_dense_load_ram_t::index, executor,
                                    progress);
    }

    /** Copies a checkpoint into a new, mutable index in device memory. */
    static expected<dense_index> load_vram(usearch_dense_load_vram_t arguments, executor_t const& executor = {},
                                           progress_t progress = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<dense_index>(usearch_dense_load_vram, arguments, &usearch_dense_load_vram_t::index, executor,
                                    progress);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in host memory. */
    static expected<dense_index> load_ram_buffer(usearch_dense_load_ram_buffer_t arguments,
                                                 executor_t const& executor = {}, progress_t progress = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<dense_index>(usearch_dense_load_ram_buffer, arguments, &usearch_dense_load_ram_buffer_t::index,
                                    executor, progress);
    }

    /** Copies a checkpoint from a memory buffer into a new, mutable index in device memory. */
    static expected<dense_index> load_vram_buffer(usearch_dense_load_vram_buffer_t arguments,
                                                  executor_t const& executor = {}, progress_t progress = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<dense_index>(usearch_dense_load_vram_buffer, arguments, &usearch_dense_load_vram_buffer_t::index,
                                    executor, progress);
    }

    /** Serves a checkpoint in place through a memory mapping, as a read-only index. */
    static expected<dense_index> view_ram(usearch_dense_view_ram_t arguments) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<dense_index>(usearch_dense_view_ram, arguments, &usearch_dense_view_ram_t::index);
    }

    /** Serves a checkpoint in place from NVMe drives as a read-only index, every later call reading
     *  it through @p executor 's engines. */
    static expected<dense_index> view_nvme(usearch_dense_view_nvme_t arguments,
                                           executor_t const& executor = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<dense_index>(usearch_dense_view_nvme, arguments, &usearch_dense_view_nvme_t::index, executor);
    }

    /** Opens a checkpoint on NVMe drives as a handle that refines and reshards by replacing whole
     *  parts, reading it through @p executor 's engines. */
    static expected<dense_index> load_nvme(usearch_dense_load_nvme_t arguments,
                                           executor_t const& executor = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return opened_<dense_index>(usearch_dense_load_nvme, arguments, &usearch_dense_load_nvme_t::index, executor);
    }

    /** Reads a checkpoint's geometry & metric without loading it. */
    static expected<usearch_dense_metadata_t> metadata_read(usearch_dense_metadata_read_t arguments,
                                                            executor_t const& executor = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return metadata_read_<usearch_dense_metadata_t>(usearch_dense_metadata_read, arguments, executor.handle());
    }

    /** Reads geometry & metric from a serialized checkpoint buffer. */
    static expected<usearch_dense_metadata_t> metadata_read_buffer(usearch_dense_metadata_read_buffer_t arguments,
                                                                   executor_t const& executor = {}) noexcept
        requires a_handle_owner<receiver_type_>
    {
        return metadata_read_<usearch_dense_metadata_t>(usearch_dense_metadata_read_buffer, arguments,
                                                        executor.handle());
    }

    /** Opens a cursor over every entry, or over the entries @p starts names first. */
    expected<dense_cursor_t> cursor_init(std::span<usearch_key_t const> starts = {}) const noexcept
        requires a_handle_owner<receiver_type_>
    {
        usearch_dense_cursor_init_t arguments {handle(), nullptr, nullptr, starts.size(), starts.data(), nullptr};
        return opened_<dense_cursor_t>(usearch_dense_cursor_init, arguments, &usearch_dense_cursor_init_t::cursor);
    }

    /** Opens a stream of matches around the one row of @p query that later calls extend. */
    template <an_element element_type_>
    expected<dense_search_stream_t> search_stream_init(rows_view<element_type_> query,
                                                       std::size_t expansion_factor = 0) const noexcept
        requires a_handle_owner<receiver_type_>
    {
        usearch_dense_search_stream_init_t arguments {nullptr, handle(), nullptr, query.view(), expansion_factor};
        return opened_<dense_search_stream_t>(usearch_dense_search_stream_init, arguments,
                                              &usearch_dense_search_stream_init_t::stream);
    }

#pragma endregion Lifecycle
};

/** Owns one dense index; destruction waits for its deferred calls. */
using dense_index_t = dense_index<owning<usearch_dense_index_t, usearch_dense_free>>;

/** A dense index its caller owns and the executor its verbs run on, completing them as
 *  @p completion_ says. */
template <usearch_completion_t completion_>
using dense_index_on = dense_index<caller_owned<usearch_dense_index_t, completion_>>;

} // namespace unum::usearch
