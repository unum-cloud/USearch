/**
 *  @file include/usearch/dense/nvme_reshard.hpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The refine and reshard plans that write new SafeGraphs parts from the parts of
 *      @c nvme_indexes views, freestanding or for a loaded index.
 *
 *  Every rewrite is an @c nvme_reshard_plan: phases of tasks that view an NVMe executor one at a
 *  time and write SafeGraphs parts beside a manifest. A rewrite never edits its sources. It lays
 *  its target out as spec-standard parts, keys, values, neighbors and the producer's norms, whose
 *  headers arrive last and the manifest after them, so a stopped rewrite reads as absent. Its RAM
 *  is the workspace the caller passes or the plan holds, the executor's arenas and its staging
 *  buffers: the workspace keeps a changed bit per target row at its front, a streamed round its
 *  columns after them, and every task carves its blocks and batches off an arena over the rest,
 *  sized from it. Reads go through the executor's queues, the target's included, and writes are
 *  blocking positional runs.
 *
 *  A reshard packs the source parts largest first into target parts of at most a shard capacity of
 *  rows, so one verb packs small parts together and, under a capacity below a part's rows, cuts it.
 *
 *  Loaded indexes reshard through the same rewrite, which owns their lists: one target among the
 *  sources takes every source's parts, packed beside its manifest, which then names them, and one
 *  source split into many empty targets cuts every part into one range per target, each target
 *  loading a manifest of its own. A loaded index's @c nvme_refine_plan rewrites one part at a time,
 *  each replacing its own.
 *
 *  @section dense_nvme_reshard_phases Phases
 *
 *  - @b Copy moves keys and values in target order, derives the norms, and rewires the neighbors a
 *    target keeps.
 *  - @b Build builds contiguous blocks of rows in a RAM index inside the workspace and writes their
 *    rows, each block healed by a RAM round stopped at its hubs.
 *  - Then one refinement round runs on every target part that took in rows, lost edges or was
 *    asked to refine, one part at a time: in RAM when the workspace holds the part as a RAM index,
 *    streamed over its files otherwise.
 *
 *  A streamed round keeps about 14.5 bytes per entry at the workspace front: the health check's
 *  four bytes and three bits, the neighborhood a gather gave the entry, its place in the order by
 *  neighborhood and its start, and a changed bit. @b Health audits the part from its root and the
 *  opening check heals the islands it names without a distance, row by row. @b Hubs reads and packs
 *  about the square root of the neighborhoods it needs as hubs, and @b gather streams the value
 *  column once past them, filing every entry under its nearest, then gathers each neighborhood over
 *  capacity again among hubs drawn from its own entries. @b Restitch reads one neighborhood's rows
 *  and links by points, scores it against itself in cache, keeps each entry's diverse links among
 *  its own and its nearest neighborhood-mates under the reach rule, and appends the changed rows to
 *  the restitch tape, a scratch file beside the part holding one run per tile of rows. @b Write
 *  applies each run to its tile of the neighbors column in one read and one write, and the closing
 *  @b health check measures the round. Each entry's row is written only by its own neighborhood,
 *  which weighs both ends of every pair, so no sweep collects reverse links.
 */
#pragma once

#include <cstdio>  // `std::remove`, `std::snprintf`
#include <cstring> // `std::memcpy`, `std::memset`

#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr`, `std::destroy_at` - the RAM index a block builds in
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <utility>     // `std::move`, `std::exchange`, `std::as_const`

#include "usearch/utilities/ram_sorts.hpp" // `partial_sort`
#include "usearch/dense/types.hpp"         // `reshard_config_t`, `reshard_report_t`
#include "usearch/dense/ram_index.hpp"     // `ram_index`, the RAM index a block build fills
#include "usearch/dense/nvme_indexes.hpp"
#include "usearch/dense/shard_checkpoints.hpp" // `monolith_layout_t`, `monolith_header` - the target parts' headers

namespace unum::usearch::dense {

template <typename indexes_type_>
struct nvme_reshard_plan;

template <typename indexes_type_>
struct nvme_refine_plan;

/**
 *  @brief Reshards the parts of @p sources into new parts at @p target, each holding at most
 *      @p config.shard_capacity rows.
 *
 *  Parts pack largest first, each bin's largest part keeps its IDs, every other part arrives after
 *  it with its edges rebased, and every target part that took in a part runs one round, which
 *  links them; one manifest commits them all. A part past the capacity is cut into ranges of it,
 *  the edges the cut crosses dropped and counted, and their target parts run a round too. @p save
 *  names the target parts' alignment and provenance, and the target is a manifest beside its parts
 *  whatever its layout. The sources must outlive the plan, which never edits them.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k No source, a @c NULL source, an empty @p target, or a target part path
 *    that does not parse.
 *  - @c uninitialized_k A first source viewing nothing, or sources holding no rows.
 *  - @c config_mismatch_k @p save names an alignment below the floor or not a power of two or a
 *    reserved provenance field, or a source walks by another kernel or differs in geometry or
 *    key width from the first.
 *  - @c bad_alloc_k @p workspace cannot hold one row of some phase.
 *  - @c overflow_k A target part of zero rows or past the node ID space, or a device ending before
 *    its base plus a part.
 *  - @c malformed_k A framed target header has no width.
 */
template <typename indexes_type_>
expected<nvme_reshard_plan<indexes_type_>> plan_reshard(std::span<indexes_type_* const> sources,
                                                        std::string_view target, bytes_span_t workspace,
                                                        reshard_config_t const& config = {},
                                                        save_config_t const& save = {}) noexcept;

/**
 *  @brief Copies @p source and runs one round on every part holding some of @p rows, in placement
 *      order across its parts, in RAM for a part the workspace holds and streamed otherwise.
 *
 *  A round takes the whole part. Every part keeps its rows, so the capacity plays no part. It
 *  writes fresh files, as a reshard does, so it runs as the same @c nvme_reshard_plan that
 *  @ref plan_reshard and @ref plan_build open. The source must outlive the plan, which reads it and
 *  never edits it.
 *
 *  @par Statuses:
 *  - @c uninitialized_k @p source views nothing, or holds no rows.
 *  - @c config_mismatch_k @p save names an alignment below the floor or not a power of two or a
 *    reserved provenance field, or a part of @p source differs in geometry or key width.
 *  - @c invalid_argument_k An empty @p target, or a target part path that does not parse.
 *  - @c bad_alloc_k @p workspace cannot hold one row of some phase.
 *  - @c overflow_k A target part of zero rows or past the node ID space, or a device ending before
 *    its base plus a part.
 *  - @c malformed_k A framed target header has no width.
 */
template <typename indexes_type_>
expected<nvme_reshard_plan<indexes_type_>> plan_refine(indexes_type_ const& source, iota_view<std::size_t> rows,
                                                       std::string_view target, bytes_span_t workspace,
                                                       reshard_config_t const& config = {},
                                                       save_config_t const& save = {}) noexcept;

/**
 *  @brief Builds one graph per target part over every row of the parts at @p sources, their
 *      neighbors ignored, the parts packed as a reshard packs them.
 *
 *  Contiguous blocks the workspace holds as a RAM index build there, each healed before it is
 *  written, then one streamed round links each part's blocks as a reshard links its parts.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No source, or sources holding no rows.
 *  - @c bad_alloc_k @p workspace cannot hold one row of some phase.
 *  - @c invalid_argument_k A source path, the empty @p target or a target part path that does not
 *    parse or names a base off its medium's block.
 *  - @c malformed_k A source manifest names no part, a manifest or header does not parse, or a
 *    part has no width or too many rows.
 *  - @c config_mismatch_k A source part disagrees with the others or with this instantiation, or
 *    @p save names an alignment below the floor or not a power of two or a reserved provenance
 *    field.
 *  - @c unsupported_k A transport ID in a build without SPDK.
 *  - @c unknown_kind_k A kernel the stored scalar's kits cannot finalize.
 *  - @c overflow_k A target part of zero rows or past the node ID space, or a device ending before
 *    its base plus a part.
 */
template <typename indexes_type_>
expected<nvme_reshard_plan<indexes_type_>> plan_build(std::span<std::string_view const> sources,
                                                      std::string_view target, bytes_span_t workspace,
                                                      reshard_config_t const& config = {},
                                                      save_config_t const& save = {}) noexcept;

/**
 *  @brief Opens a refine of the loaded @p index: every part in turn is copied into a fresh part
 *      beside its own and re-selected there in one round, then replaces it.
 *
 *  The plan holds the index's writer while it lives and its own workspace, @p config.memory_limit
 *  bytes, zero deriving one that rebuilds each part in RAM. Each part is sealed, the manifest
 *  replaced durably to name it, and only then is it swapped in for the index's readers, at the
 *  barrier closing its round; a plan dropped before that removes the fresh part, and the manifest
 *  still names the old one. Peak extra disk is one part beyond the parts superseded so far, which
 *  stay until the index is released.
 *
 *  @par Statuses:
 *  - @c read_only_k @p index is a view.
 *  - @c contended_k Another refine or reshard holds the writer, or another writer took the
 *    manifest's lock.
 *  - @c bad_alloc_k The workspace cannot be allocated or hold one row of some phase.
 *  - Any status opening the first part's rewrite answers, as @ref plan_refine lists them.
 */
template <typename indexes_type_>
expected<nvme_refine_plan<indexes_type_>> plan_refine(indexes_type_& index, refine_config_t config,
                                                      typename indexes_type_::executor_t& executor) noexcept;

/**
 *  @brief Opens a reshard of the loaded @p sources into the loaded @p targets: several sources into
 *      one target, or any sources into several empty targets.
 *
 *  Into one target, its own parts and every source's pack as @ref plan_reshard packs them into
 *  parts beside the target's manifest, which then names them, every target part that took a part in
 *  running rounds that link it until one stops improving; the other sources are read, never edited,
 *  and a caller drops them once @c finish answers success. A target alone, of one part and under a
 *  capacity it fits, keeps it. Into several targets, every part of every source, in source order,
 *  is cut into one contiguous range per target, the edges the cuts cross dropped, and range @c r of
 *  part @c s arrives as part @c s of target @c r, running rounds until one stops improving; ranges
 *  by ID, rather than parts grown over the links as the RAM split grows them, keep the split a
 *  streamed copy, and the rounds re-stitch what the cuts dropped. Target @c r's manifest is the
 *  first source manifest's stem, a dash, the first free split number, and a dash and @c r from one
 *  on; @c finish empties every source and loads every target.
 *
 *  The plan owns both lists and its workspace, @p config.refine.memory_limit bytes, zero deriving
 *  one that runs each round in RAM, and holds the writer of the target, or of the first source it
 *  splits, while it lives; a plan dropped before @c finish leaves every index serving its own
 *  parts. No other call on any of the indexes may overlap it.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k No source, no target, or a null or repeated index in either list.
 *  - @c read_only_k The target, or a source split, is a view.
 *  - @c config_mismatch_k A split target that holds entries or is a source, or a source of
 *    another kernel, geometry or key width.
 *  - @c overflow_k A part holds too few entries to give every split target one.
 *  - @c contended_k A refine or reshard holds the writer.
 *  - Any status opening the rewrite answers, as @ref plan_reshard lists them.
 */
template <typename indexes_type_, typename queue_type_>
    requires std::is_same_v<nvme_executor<queue_type_>, typename indexes_type_::executor_t>
expected<nvme_reshard_plan<indexes_type_>> plan_reshard(std::span<indexes_type_* const> sources,
                                                        std::span<indexes_type_* const> targets,
                                                        nvme_executor<queue_type_>& executor,
                                                        reshard_config_t config = {}) noexcept;

/** Gives a loaded index's writer back as the plan that held it ends. */
struct end_rewrite_t {
    template <typename indexes_type_>
    void operator()(indexes_type_* index) const noexcept {
        index->end_rewrite();
    }
};

/**
 *  @brief One rewrite in flight: phases of tasks, each task run across every thread of an executor.
 *
 *  Tasks of a phase may run in any order, but one at a time, since each claims the executor while
 *  it runs, so searches on the same executor run between tasks; a failed task may run again. The
 *  round's phases repeat once per target part that runs one, its columns living across their
 *  tasks. @ref advance is the single-caller barrier between phases. A dying plan removes every
 *  target part no commit named and the restitch tape, so a stopped rewrite reads as absent, and
 *  gives back the writer of a loaded index it holds.
 */
template <typename indexes_type_>
struct [[nodiscard]] nvme_reshard_plan {
    using indexes_t = indexes_type_;
    using executor_t = typename indexes_t::executor_t;
    using key_t = typename indexes_t::key_t;
    using result_t = reshard_report_t;

    /** One unit of the current phase. */
    struct task_t {

        /** The task's position in its phase. */
        std::size_t position = 0;

        /** The source part a copy task reads, or the target part every other task writes. */
        std::size_t part = 0;

        /** The node IDs it covers: source ones for a copy task, positions in the order by
         *  neighborhood for a task over one neighborhood, target ones otherwise. */
        iota_view<node_id_t> nodes;
    };

    nvme_reshard_plan() noexcept = default;
    nvme_reshard_plan(nvme_reshard_plan&&) noexcept = default;
    nvme_reshard_plan& operator=(nvme_reshard_plan&&) = delete;
    nvme_reshard_plan(nvme_reshard_plan const&) = delete;
    nvme_reshard_plan& operator=(nvme_reshard_plan const&) = delete;

    /** Removes the target parts unless @ref finish committed them, and the restitch tape; the
     *  writer it held goes back with it. */
    ~nvme_reshard_plan() noexcept {
        close_tape_();
        for (target_t& target : targets_) {
            [[maybe_unused]] status_t const closed = target.file.close();
            target.view = indexes_t {allocator_};
            std::remove(target.path.data());
        }
    }

    /** Task @p position of the current phase; an empty one past the last. */
    task_t operator[](std::size_t position) const noexcept {
        if (rounding_()) return round_task_(position);
        for (std::size_t unit = 0, first = 0; unit != units_(); ++unit) {
            cut_t const cut = cut_(unit);
            std::size_t const count = cut.tasks();
            if (position - first < count) {
                iota_view<std::size_t> const rows = cut.rows_of(position - first);
                return {position, unit, {static_cast<node_id_t>(rows.first), static_cast<node_id_t>(rows.bound)}};
            }
            first += count;
        }
        return {};
    }

    /** The current phase's tasks, positionally; none once the plan is done. */
    indexed_range<nvme_reshard_plan> tasks() const noexcept {
        if (rounding_()) return {this, round_tasks_()};
        std::size_t count = 0;
        for (std::size_t unit = 0; unit != units_(); ++unit) count += cut_(unit).tasks();
        return {this, count};
    }

    /** The phase whose tasks run now. */
    phase_t phase() const noexcept { return phase_; }

    /** The position of the round's phase in hand among its six, from the opening health check to
     *  the closing one, the write counting as the restitch it applies; zero outside a round. */
    std::size_t ordinal() const noexcept {
        return round_ordinal(phase_ == phase_t::write_k ? phase_t::restitch_k : phase_,
                             health_after_ == phase_t::write_k ? phase_t::write_k : phase_t::none_k);
    }

    /** Tasks of the phase in hand run so far, and in all. */
    std::size_t completed() const noexcept { return completed_tasks_; }
    std::size_t total() const noexcept { return tasks().size(); }

    /** The rounds folded so far, the report @ref finish completes. */
    refine_report_t running() const noexcept { return result_.graph; }

    /** Opens another round over every target part that ran one, once the plan ended: the
     *  report's rounds start over, and each part runs until a round stops improving, as the
     *  first did; a no-op before. */
    void next_round() noexcept {
        if (phase_ != phase_t::done_k || targets_.empty()) return;
        result_.graph = {};
        folded_parts_ = 0;
        open_round_(0);
    }

    /**
     *  @brief Runs @p task of the current phase across every thread of @p executor, which it claims
     *      while it runs.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p task lies past the current phase's tasks, a part rebuilt in RAM
     *    disagrees with its rows, or a medium the queues cannot read.
     *  - @c uninitialized_k The plan is done, or @p executor holds no pool or workers.
     *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
     *  - @c bad_alloc_k The workspace cannot hold the task's slices.
     *  - @c overflow_k A part rebuilt in RAM passes a bound, a staging region's byte count
     *    overflows, or a write runs past a device's end.
     *  - @c unknown_kind_k A part rebuilt in RAM cannot finalize its kernel.
     */
    status_t run(task_t const& task, executor_t& executor) noexcept {
        if (task.position >= tasks().size()) return status_t::config_mismatch_k;
        // The first task fixes a zero cache to the executor's, before any round sizes from it.
        if (!cache_bytes_) cache_bytes_ = executor.cache_bytes_per_core();
        status_t const ran = [&]() noexcept {
            switch (phase_) {
            case phase_t::copy_k: return copy_(task, executor);
            case phase_t::build_k: return build_(task, executor);
            case phase_t::health_k: return health_(task, executor);
            case phase_t::hubs_k: return draw_hubs_(task, executor);
            case phase_t::gather_k: return gather_(task, executor);
            case phase_t::restitch_k:
                return in_ram_(task.part) ? refine_in_ram_(task.part, executor) : restitch_(task, executor);
            case phase_t::write_k: return write_(task, executor);
            default: return status_t::uninitialized_k;
            }
        }();
        if (succeeded(ran)) ++completed_tasks_;
        return ran;
    }

    /** The barrier between phases, one caller after every task ran: enters the next phase with
     *  tasks, ordering a gather's entries by neighborhood, and answers whether one follows. */
    bool advance() noexcept {
        completed_tasks_ = 0;
        while (phase_ != phase_t::done_k) {
            switch (phase_) {
            case phase_t::copy_k: phase_ = phase_t::build_k; break;
            case phase_t::build_k: open_round_(0); break;
            case phase_t::health_k:
                if (health_after_ != phase_t::write_k) phase_ = phase_t::hubs_k;
                else if (another_round_()) health_after_ = phase_t::build_k;
                else close_part_();
                break;
            case phase_t::hubs_k:
                neighborhoods_ = 0;
                phase_ = phase_t::gather_k;
                break;
            // Neighborhoods over capacity gather again among their own entries.
            case phase_t::gather_k:
                if (order_neighborhoods_() <= neighborhood_capacity_) phase_ = phase_t::restitch_k;
                break;
            case phase_t::restitch_k:
                if (in_ram_(round_part_)) close_part_();
                else phase_ = phase_t::write_k;
                break;
            case phase_t::write_k:
                health_after_ = phase_t::write_k;
                phase_ = phase_t::health_k;
                break;
            default: phase_ = phase_t::done_k;
            }
            if (!tasks().empty()) return true;
        }
        return false;
    }

    /**
     *  @brief Writes every part header, then the record naming the parts: the commit.
     *
     *  A freestanding rewrite writes its manifest. A reshard into one loaded target replaces the
     *  target's parts with its own, and one into many targets writes each target's manifest,
     *  empties the source and loads every target; a loaded reshard that moved nothing commits
     *  nothing. The writer it held goes back either way.
     *
     *  @par Statuses:
     *  - @c uninitialized_k A freestanding rewrite holds no target part.
     *  - @c cancelled_k The last phase has not ended.
     *  - @c config_mismatch_k A part header does not fit before its data.
     *  - @c overflow_k A part is shorter than its header, or a device ends before its base plus the
     *    manifest.
     *  - @c invalid_argument_k A manifest path does not parse.
     *  - Any status @ref nvme_indexes::replace_parts, or a target's load, answers.
     */
    expected<result_t> finish(executor_t& executor) noexcept {
        if (target_indexes_.empty()) {
            if (status_t const sealed = seal_(); failed(sealed)) return {result_, sealed};
            auto part_at = [&](std::size_t part) noexcept { return part_entry(part); };
            return committed_(safegraphs_write_manifest(path_of_(manifest_), targets_.size(), part_at));
        }
        if (targets_.empty() && phase_ == phase_t::done_k)
            return writing_.reset(), expected<result_t> {result_, status_t::success_k};
        if (target_indexes_.size() == 1) {
            expected<result_t> const replaced = finish([&](nvme_reshard_plan const& sealed) noexcept {
                return target_indexes_[0]->replace_parts(
                    sealed.parts(),
                    [&](std::size_t part) noexcept {
                        safegraphs_part_t const entry = sealed.part_entry(part);
                        return safegraphs_part_t {sealed.part_path(part), entry.bytes, entry.nodes};
                    },
                    executor);
            });
            if (succeeded(replaced.status)) writing_.reset();
            return replaced;
        }
        return finish_split_();
    }

    /** The entry a record names target part @p part by: its file's name, bytes and rows. */
    safegraphs_part_t part_entry(std::size_t part) const noexcept {
        target_t const& target = targets_[part];
        return safegraphs_part_t {safegraphs_basename(path_of_(target.path)), target.header.monolith_bytes(),
                                  target.view.size()};
    }

    /** The target parts the plan writes, and where target part @p part sits. */
    std::size_t parts() const noexcept { return targets_.size(); }
    std::string_view part_path(std::size_t part) const noexcept { return path_of_(targets_[part].path); }

    /** Bytes of workspace that run a round over a part of @p rows rows of the geometry of @p shape
     *  in RAM: its changed bits, and one RAM index of its rows with the round's columns beside a
     *  staged row, under the cache @p config names. */
    static std::size_t round_in_ram_bytes(indexes_t const& shape, std::size_t rows,
                                          reshard_config_t const& config = {}) noexcept {
        nvme_reshard_plan plan;
        plan.config_ = config;
        plan.cache_bytes_ = config.refine.cache_bytes;
        if (failed(plan.add_sources_(shape, shape)) || plan.sources_.empty()) return 0;
        carver_t counter;
        [[maybe_unused]] refine_slices_t const sized = plan.refine_slices_(rows, 1, counter);
        std::size_t const changed_bytes = workspace_allocator_t::reserved_bytes(divide_round_up<std::size_t>(rows, 64) *
                                                                                sizeof(u64_t));
        return workspace_allocator_t::alignment_k + changed_bytes + counter.bytes;
    }

  private:
    /** Writes and flushes every part header, the parts' half of a commit. */
    status_t seal_() noexcept {
        if (targets_.empty()) return status_t::uninitialized_k;
        if (phase_ != phase_t::done_k) return status_t::cancelled_k;
        std::span<safegraphs_custom_field_t const> const fields = to_span(fields_);
        for (target_t& target : targets_) {
            std::size_t const bytes = target.header.monolith_bytes();
            if (status_t const framed = safegraphs_write_header(file_span_t {&target.file, 0, bytes}, target.header,
                                                                fields);
                failed(framed))
                return framed;
            if (status_t const flushed = target.file.flush(); failed(flushed)) return flushed;
        }
        return status_t::success_k;
    }

    /** Ends a commit whose record answered @p committed: a committed plan's parts outlive it, so
     *  its destructor finds none to remove. */
    expected<result_t> committed_(status_t committed) noexcept {
        if (failed(committed)) return {result_, committed};
        result_.source_shards = sources_.size();
        result_.target_shards = targets_.size();
        targets_.clear();
        return {result_, status_t::success_k};
    }

    /**
     *  @brief Writes every part header, then calls @p commit with the plan, which writes the record
     *      naming the parts in place of the plan's own manifest and answers whether it did.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The plan holds no target part.
     *  - @c cancelled_k The last phase has not ended.
     *  - @c config_mismatch_k A part header does not fit before its data.
     *  - Whatever @p commit answers; the parts are removed unless it succeeded.
     */
    template <typename commit_type_>
    expected<result_t> finish(commit_type_&& commit) noexcept {
        if (status_t const sealed = seal_(); failed(sealed)) return {result_, sealed};
        return committed_(commit(std::as_const(*this)));
    }

    template <typename other_type_>
    friend struct nvme_refine_plan;
    template <typename other_type_>
    friend expected<nvme_reshard_plan<other_type_>> plan_reshard(std::span<other_type_* const>, std::string_view,
                                                                 bytes_span_t, reshard_config_t const&,
                                                                 save_config_t const&) noexcept;
    template <typename other_type_>
    friend expected<nvme_reshard_plan<other_type_>> plan_refine(other_type_ const&, iota_view<std::size_t>,
                                                                std::string_view, bytes_span_t, reshard_config_t const&,
                                                                save_config_t const&) noexcept;
    template <typename other_type_>
    friend expected<nvme_reshard_plan<other_type_>> plan_build(std::span<std::string_view const>, std::string_view,
                                                               bytes_span_t, reshard_config_t const&,
                                                               save_config_t const&) noexcept;
    template <typename other_type_, typename queue_type_>
        requires std::is_same_v<nvme_executor<queue_type_>, typename other_type_::executor_t>
    friend expected<nvme_reshard_plan<other_type_>> plan_reshard(std::span<other_type_* const>,
                                                                 std::span<other_type_* const>,
                                                                 nvme_executor<queue_type_>&,
                                                                 reshard_config_t) noexcept;

    /** Opens a freestanding reshard of @p sources into the manifest @p target, as @ref plan_reshard
     *  describes. */
    status_t open_packed_(std::span<indexes_t* const> sources, std::string_view target, bytes_span_t workspace,
                          reshard_config_t const& config, save_config_t const& save) noexcept {
        if (sources.empty() || !sources[0]) return status_t::invalid_argument_k;
        if (status_t const opened = open_(target, workspace, config, save, *sources[0]); failed(opened)) return opened;
        for (indexes_t* source : sources)
            if (status_t const added = source ? add_sources_(*source, *sources[0]) : status_t::invalid_argument_k;
                failed(added))
                return added;
        if (sources_.empty()) return status_t::uninitialized_k;
        for (source_t& part : sources_) part.kept = {0, part.rows};
        if (status_t const sized = size_tasks_(); failed(sized)) return sized;
        growing_buffer<std::size_t, forkunion_allocator_t> target_rows {allocator_};
        if (status_t const packed = pack_sources_(config.shard_capacity, target_rows); failed(packed)) return packed;
        if (status_t const created = open_targets_(to_span(target_rows)); failed(created)) return created;

        // The part arriving first in a target part keeps its IDs; every later part arrives after it
        // with its own edges rebased, and the round links them. A cut range lost the edges past it,
        // so its target part runs a round too.
        for (source_t const& part : sources_) {
            if (part.base) continue;
            bool const cut = part.stride < part.rows;
            for (std::size_t first = 0, target_index = part.target; first < part.rows;
                 first += part.stride, ++target_index) {
                target_t& bin = targets_[target_index];
                bin.selected = {cut ? 0 : min_of(part.stride, part.rows - first), bin.view.size()};
            }
        }
        phase_ = phase_t::copy_k;
        return status_t::success_k;
    }

    /** Opens the rewrite of part @p shard of the loaded @p index, read in place, into the part file
     *  @p target, one round over it and no manifest, which a commit through @c finish hands to the
     *  caller's own record. */
    status_t open_part_refine_(indexes_t const& index, std::size_t shard, std::string_view target,
                               bytes_span_t workspace, reshard_config_t const& config) noexcept {
        if (shard >= index.shards_count()) return status_t::uninitialized_k;
        if (target.empty()) return status_t::invalid_argument_k;
        if (status_t const opened = open_workspace_(workspace, config, {}, index); failed(opened)) return opened;
        if (status_t const added = add_part_(index, index, static_cast<u32_t>(shard)); failed(added)) return added;
        if (sources_.empty()) return status_t::uninitialized_k;
        source_t& part = sources_[0];
        part.kept = {0, part.rows};
        std::array<std::size_t, 1> const rows {part.rows};
        if (status_t const sized = size_tasks_(); failed(sized)) return sized;
        if (status_t const created = open_targets_(std::span<std::size_t const> {rows}, replacing_part_path_t {target});
            failed(created))
            return created;
        phase_ = phase_t::copy_k;
        return status_t::success_k;
    }

    /** Takes the loaded @p sources and @p targets as its own lists, the writer of @p writer, and a
     *  workspace of @p config.refine.memory_limit bytes, or of what a round over @p rows rows of
     *  @p writer 's geometry takes in RAM. */
    status_t hold_(std::span<indexes_t* const> sources, std::span<indexes_t* const> targets, indexes_t& writer,
                   std::size_t rows, reshard_config_t const& config) noexcept {
        if (failed(source_indexes_.reserve_exactly(sources.size())) ||
            failed(target_indexes_.reserve_exactly(targets.size())))
            return status_t::bad_alloc_k;
        for (indexes_t* source : sources) source_indexes_.push_back_reserved(source);
        for (indexes_t* target : targets) target_indexes_.push_back_reserved(target);
        if (status_t const claimed = writer.begin_rewrite(); failed(claimed)) return claimed;
        writing_.reset(&writer);
        std::size_t const bytes = config.refine.memory_limit ? config.refine.memory_limit
                                                             : round_in_ram_bytes(writer, rows, config);
        return failed(held_workspace_.resize_uninitialized(bytes)) ? status_t::bad_alloc_k : status_t::success_k;
    }

    /** Opens a reshard of the one loaded target of @p targets and every loaded source into it,
     *  packed beside its manifest; a lone target of one part its capacity holds moves nothing. */
    status_t open_gathered_(std::span<indexes_t* const> sources, std::span<indexes_t* const> targets,
                            reshard_config_t const& config) noexcept {
        indexes_t& target = *targets[0];
        if (status_t const refused = target.refuses_writes(); refused == status_t::read_only_k) return refused;
        growing_buffer<indexes_t*, forkunion_allocator_t> packed {allocator_};
        if (failed(packed.reserve_exactly(sources.size() + 1))) return status_t::bad_alloc_k;
        packed.push_back_reserved(&target);
        std::size_t rows = target.size(), parts = target.shards_count();
        for (indexes_t* source : sources)
            if (source != &target && source->shards_count()) {
                rows += source->size(), parts += source->shards_count();
                packed.push_back_reserved(source);
            }
        std::size_t const capacity = config.shard_capacity && config.shard_capacity < node_id_unknown_k
                                         ? config.shard_capacity
                                         : node_id_unknown_k - 1;
        if (packed.size() == 1 && target.shards_count() < 2 && target.size() <= capacity)
            return hold_(sources, targets, target, 0, config);
        if (status_t const held = hold_(sources, targets, target, min_of(rows, capacity), config); failed(held))
            return held;
        paths_t stem_buffer {allocator_}, manifest {allocator_};
        auto const [stem, stem_status] = nvme_unused_stem_(target, ".", 0, parts, stem_buffer);
        if (failed(stem_status)) return stem_status;
        std::string_view const suffix = ".safegraphs";
        if (failed(manifest.resize_uninitialized(stem.size() + suffix.size()))) return status_t::bad_alloc_k;
        std::memcpy(manifest.data(), stem.data(), stem.size());
        std::memcpy(manifest.data() + stem.size(), suffix.data(), suffix.size());
        return open_packed_(to_span(packed), to_string_view(manifest), to_writable_span(held_workspace_), config, {});
    }

    /** Opens a split of the loaded @p sources into the empty loaded @p receivers: every part of
     *  every source, in source order, cut into one contiguous range per receiver, range @c r of
     *  part @c s arriving as target part <tt>s * receivers + r</tt> under receiver @c r 's stem and
     *  running its rounds of linking. */
    status_t open_scattered_(std::span<indexes_t* const> sources, std::span<indexes_t* const> receivers,
                             reshard_config_t const& config) noexcept {
        indexes_t& source = *sources[0];
        std::size_t const count = receivers.size();
        std::size_t widest = 0, parts = 0;
        for (indexes_t* from : sources) {
            if (status_t const refused = from->refuses_writes(); refused == status_t::read_only_k) return refused;
            for (std::size_t part = 0; part != from->shards_count(); ++part)
                widest = max_of<std::size_t>(widest, from->part_size(part));
            parts += from->shards_count();
        }
        if (lists_share(sources, receivers)) return status_t::config_mismatch_k;
        for (indexes_t* receiver : receivers)
            if (receiver->shards_count()) return status_t::config_mismatch_k;
        if (status_t const held = hold_(sources, receivers, source, divide_round_up(widest, count), config);
            failed(held))
            return held;
        paths_t stem_buffer {allocator_};
        auto const [stem, stem_status] = nvme_unused_stem_(source, "-split", count, parts, stem_buffer);
        if (failed(stem_status)) return stem_status;
        if (failed(keep_path_(stem_, stem))) return status_t::bad_alloc_k;
        if (status_t const opened = open_workspace_(to_writable_span(held_workspace_), config, {}, source);
            failed(opened))
            return opened;
        for (indexes_t* from : sources)
            if (status_t const added = add_sources_(*from, source); failed(added)) return added;
        if (sources_.empty()) return status_t::uninitialized_k;
        growing_buffer<std::size_t, forkunion_allocator_t> rows {allocator_};
        if (failed(rows.reserve_exactly(sources_.size() * count))) return status_t::bad_alloc_k;
        for (std::size_t part = 0; part != sources_.size(); ++part) {
            source_t& one = sources_[part];
            std::size_t const stride = divide_round_up(one.rows, count);
            if ((count - 1) * stride >= one.rows) return status_t::overflow_k;
            one.target = part * count;
            one.stride = stride;
            one.kept = {0, one.rows};
            for (std::size_t range = 0; range != count; ++range)
                rows.push_back_reserved(min_of(stride, one.rows - range * stride));
            result_.moved_entries += one.rows - stride;
        }
        if (status_t const sized = size_tasks_(); failed(sized)) return sized;
        paths_t receiver_stem {allocator_};
        auto part_path = [&](paths_t& buffer, std::size_t target, std::size_t) noexcept -> expected<std::string_view> {
            std::string_view const receiver = receiver_path_(receiver_stem, target % count, {});
            if (receiver.empty()) return {{}, status_t::bad_alloc_k};
            return safegraphs_part_path(buffer, receiver, target / count, sources_.size());
        };
        if (status_t const created = open_targets_(to_span(rows), part_path); failed(created)) return created;
        phase_ = phase_t::copy_k;
        return status_t::success_k;
    }

    template <typename word_type_>
    using atomic_ref = typename indexes_t::template atomic_ref<word_type_>;
    using wait_t = typename indexes_t::wait_t;
    using scalar_t = typename indexes_t::scalar_t;
    using dot_result_t = typename indexes_t::dot_result_t;
    using vector_view_t = typename indexes_t::vector_view_t;
    using vectors_view_t = typename indexes_t::vectors_view_t;
    using vector_norms_t = typename indexes_t::vector_norms_t;
    using medium_t = typename indexes_t::medium_t;
    using arena_t = typename indexes_t::arena_t;
    using scratch_allocator_t = typename indexes_t::scratch_allocator_t;
    using scan_scorer_t = typename indexes_t::scan_scorer_t;
    using raw_dot_t = typename dot_result_t::raw_t;
    using islands_t = audit_islands<node_id_t>;
    using paths_t = growing_buffer<char, forkunion_allocator_t>;

    /** The arena every task carves the workspace from, the workspace its only chunk. */
    using workspace_upstream_t = caller_span_allocator<byte_t>;
    using workspace_arena_t = growing_arena<workspace_upstream_t>;
    using workspace_allocator_t = arena_allocator<byte_t, workspace_upstream_t>;

    /** The RAM index a block build or a whole-part refine places inside the workspace, each node
     *  keyed by its row in the block. */
    using ram_index_t = ram_index<typename indexes_t::quads_kit_t, typename indexes_t::tiles_kit_t,
                                  typename indexes_t::elementwise_kit_t, typename indexes_t::rmscast_kit_t,
                                  typename indexes_t::visits_kit_t, wait_t, indexes_t::template atomic_ref,
                                  dummy_cache_hints_t, workspace_allocator_t, workspace_allocator_t,
                                  arena_allocator<byte_t, forkunion_allocator_t>, node_id_t>;
    using traversal_context_t = typename ram_index_t::traversal_context_t;
    using traversals_t = growing_buffer<traversal_context_t, scratch_allocator_t>;

    /** Ends the life of a RAM index built inside the workspace, which keeps its bytes. */
    struct destroy_in_place_t {
        void operator()(ram_index_t* index) const noexcept { std::destroy_at(index); }
    };
    using ram_index_ptr_t = std::unique_ptr<ram_index_t, destroy_in_place_t>;
    using ram_cursor_t =
        ram_cursor<ram_index_t, synchronization_t::immutable_k, arena_allocator<byte_t, forkunion_allocator_t>>;

    inline static constexpr node_id_t sentinel_k = node_id_unknown_k;
    static_assert(sizeof(vector_norms_t) == 2 * sizeof(dot_result_t), "A norms row is a sum and a square");

    /** One part of a source, flattened across the sources: where its rows arrive and which of its
     *  edges survive. */
    struct source_t {

        /** The view that reads it. */
        indexes_t const* view = nullptr;

        /** The part inside that view. */
        u32_t part = 0;

        /** Rows the part holds. */
        std::size_t rows = 0;

        /** The target part its first range arrives in. */
        std::size_t target = 0;

        /** The target row its row zero arrives at. */
        std::size_t base = 0;

        /** Rows per target part, each next range arriving in the next target part. */
        std::size_t stride = 0;

        /** The source nodes whose edges may survive; empty writes no edges at all. */
        iota_view<std::size_t> kept;
    };

    /** One target part: its file, its header, a one-part view the scans and rounds read it
     *  through, and the rows that call for its round. */
    struct target_t {
        paths_t path;
        seekable_file_t file;
        safegraphs_header_t header;
        indexes_t view;

        /** The norms column, which the header's producer columns view. */
        safegraphs_named_column_t norms;

        /** The rows that call for a round, all of them unless the plan narrows them; a part runs
         *  one whole round when any do. */
        iota_view<std::size_t> selected;

        /** Where the part's rows start among the changed bits. */
        std::size_t changed_first = 0;
    };

    /** One column a scan fetches: where its row zero sits in the part, a row's width, and where row
     *  zero arrives. */
    struct staged_column_t {
        u64_t begin = 0;
        std::size_t row_bytes = 0;
        byte_t* staging = nullptr;
    };

    /** The slices of one task so far: the arena they come off, null to only sum their bytes, and
     *  the bytes they reserve, the padding up to every next boundary included. */
    struct carver_t {
        workspace_arena_t* arena = nullptr;
        std::size_t bytes = 0;
    };

    /** The columns a copy task stages, one row each per row it copies. */
    struct copy_slices_t {
        byte_t* keys = nullptr;
        byte_t* values = nullptr;
        byte_t* norms = nullptr;
        node_id_t* neighbors = nullptr;
    };

    /** A block build's value rows, the node each row goes in as, its neighbors on the way out, and
     *  its RAM index, which keys each node by its row. */
    struct build_slices_t {
        byte_t* values = nullptr;
        node_id_t* nodes = nullptr;
        node_id_t* staging = nullptr;
        ram_index_t* index = nullptr;
    };

    /** A whole-part refine's RAM index and its neighbors on the way out. */
    struct refine_slices_t {
        ram_index_t* index = nullptr;
        node_id_t* staging = nullptr;
    };

    /** A streamed round's columns at the workspace front, past the changed bits: the health check's
     *  bytes and bitsets, per entry the neighborhood its last gather gave it and its place in the
     *  order by neighborhood, the starts every split may write, and per tile of rows its run's
     *  length on the restitch tape. */
    struct round_slices_t {
        u8_t* depth_columns = nullptr;
        u64_t* reach_words = nullptr;
        u16_t* entry_neighborhoods = nullptr;
        node_id_t* neighborhood_entries = nullptr;
        node_id_t* neighborhood_offsets = nullptr;
        std::size_t* tape_runs = nullptr;
    };

    /** The hubs a gather scores against, past the round's columns: their nodes, value rows, norms
     *  and pack. */
    struct hub_slices_t {
        node_id_t* nodes = nullptr;
        byte_t* rows = nullptr;
        vector_norms_t* norms = nullptr;
        byte_t* packs = nullptr;
    };

    /** One neighborhood a restitch stages past the round's columns: the value rows of its entries
     *  and then of the links leading out of it, its entries' norms and packs, its links, the slots
     *  of those leading out, every entry's nearest neighborhood-mates, and its tape rows. */
    struct restitch_slices_t {
        byte_t* rows = nullptr;
        vector_norms_t* norms = nullptr;
        byte_t* packs = nullptr;
        node_id_t* links = nullptr;
        u32_t* outside_slots = nullptr;
        scored_node_t* mates = nullptr;
        node_id_t* tape_rows = nullptr;
    };

    /** One tile of neighbor rows a write stages past the round's columns, and its run off the
     *  tape. */
    struct write_slices_t {
        node_id_t* neighbors = nullptr;
        node_id_t* tape_rows = nullptr;
    };

    /** One thread's share of a gather, from its own arena: each row's nearest hub so far and its
     *  norms over one read, the hubs' thresholds, and one scorer task's dots and distances. */
    struct gather_share_t {
        scored_node_t* nearest = nullptr;
        vector_norms_t* row_norms = nullptr;
        std::span<distance_t> thresholds;
        nk::matrix_span<raw_dot_t> dots;
        nk::matrix_span<f32_t> step_distances;
    };

    /** One thread's share of a restitch, from its own arena: one block's thresholds, one scorer
     *  task's dots and distances, and for one entry its live links, its candidates by staged row,
     *  and its picks with the norms the diversity rule compares. */
    struct restitch_share_t {
        std::span<distance_t> thresholds;
        nk::matrix_span<raw_dot_t> dots;
        nk::matrix_span<f32_t> step_distances;
        scored_node_t* live = nullptr;
        scored_node_t* candidates = nullptr;
        scored_node_t* selected = nullptr;
        u32_t* kept_rows = nullptr;
        vector_norms_t* kept_norms = nullptr;
    };

    /** Receiver @p receiver 's stem beside a split's source, or with @p suffix its manifest, into
     *  @p buffer: the split's stem, a dash and the receiver's number from one on; empty when
     *  @p buffer cannot grow. */
    std::string_view receiver_path_(paths_t& buffer, std::size_t receiver, std::string_view suffix) const noexcept {
        std::string_view const stem = path_of_(stem_);
        // An empty view may hold a null pointer, which `%.*s` must never see.
        char const* const stem_chars = stem.empty() ? "" : stem.data();
        char const* const suffix_chars = suffix.empty() ? "" : suffix.data();
        auto print = [&](char* sink, std::size_t capacity) noexcept {
            return std::snprintf(sink, capacity, "%.*s-%zu%.*s", static_cast<int>(stem.size()), stem_chars,
                                 receiver + 1, static_cast<int>(suffix.size()), suffix_chars);
        };
        int const length = print(nullptr, 0);
        if (length < 0 || failed(buffer.resize_uninitialized(static_cast<std::size_t>(length) + 1))) return {};
        print(buffer.data(), buffer.size());
        return {buffer.data(), static_cast<std::size_t>(length)};
    }

    /** Commits a split: every receiver's manifest naming its parts, removed again if a later one
     *  fails; then the writer goes back, every source empties, and each receiver loads its own. */
    expected<result_t> finish_split_() noexcept {
        std::size_t const count = target_indexes_.size(), parts = targets_.size() / max_of<std::size_t>(count, 1);
        paths_t manifest {allocator_};
        expected<result_t> const split = finish([&](nvme_reshard_plan const& sealed) noexcept {
            for (std::size_t receiver = 0; receiver != count; ++receiver) {
                std::string_view const path = receiver_path_(manifest, receiver, ".safegraphs");
                status_t const written = path.empty()
                                             ? status_t::bad_alloc_k
                                             : safegraphs_write_manifest(path, parts, [&](std::size_t part) noexcept {
                                                   return sealed.part_entry(part * count + receiver);
                                               });
                if (failed(written)) {
                    for (std::size_t back = 0; back <= receiver; ++back)
                        if (!receiver_path_(manifest, back, ".safegraphs").empty()) std::remove(manifest.data());
                    return written;
                }
            }
            return status_t::success_k;
        });
        if (failed(split.status)) return split;
        // The receivers' manifests are durable, so the source empties and each receiver loads.
        writing_.reset();
        for (indexes_t* source : source_indexes_) source->release();
        for (std::size_t receiver = 0; receiver != count; ++receiver)
            if (status_t const loaded = target_indexes_[receiver]->load(
                    receiver_path_(manifest, receiver, ".safegraphs"));
                failed(loaded))
                return {split.value, loaded};
        return split;
    }

#pragma region Opening

    /** The text of @p path without its terminator. */
    static std::string_view path_of_(paths_t const& path) noexcept { return {path.data(), path.size() - 1}; }

    /** Copies @p text into @p path, terminated for the OS. */
    static status_t keep_path_(paths_t& path, std::string_view text) noexcept {
        if (failed(path.resize_uninitialized(text.size() + 1))) return status_t::bad_alloc_k;
        std::memcpy(path.data(), text.data(), text.size());
        path[text.size()] = '\0';
        return status_t::success_k;
    }

    /** Opens the plan's common state, then the target manifest's paths, removing a stale manifest
     *  so no half-written part is ever named. */
    status_t open_(std::string_view target, bytes_span_t workspace, reshard_config_t const& config,
                   save_config_t const& save, indexes_t const& view) noexcept {
        if (target.empty()) return status_t::invalid_argument_k;
        if (status_t const opened = open_workspace_(workspace, config, save, view); failed(opened)) return opened;
        if (failed(keep_path_(manifest_, target)) || failed(keep_path_(stem_, safegraphs_stem(target))))
            return status_t::bad_alloc_k;
        std::remove(manifest_.data());
        return status_t::success_k;
    }

    /** Opens the plan's common state: the workspace, the configurations and the header fields
     *  naming the kernel of @p view. */
    status_t open_workspace_(bytes_span_t workspace, reshard_config_t const& config, save_config_t const& save,
                             indexes_t const& view) noexcept {
        if (status_t const valid = save.validate(); failed(valid)) return valid;
        if (!view.shards_count()) return status_t::uninitialized_k;
        // Trimmed to whole boundaries, so a task's slices take exactly the bytes they reserve.
        std::size_t const alignment = workspace_allocator_t::alignment_k;
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(workspace.data());
        std::size_t const padding = min_of<std::size_t>(
            round_up_to_multiple<std::uintptr_t>(address, alignment) - address, workspace.size());
        workspace_ = workspace.subspan(padding, (workspace.size() - padding) / alignment * alignment);
        config_ = config;
        cache_bytes_ = config.refine.cache_bytes;
        save_ = save;
        if (failed(fields_.reserve_exactly(1 + save.provenance.size()))) return status_t::bad_alloc_k;
        fields_.push_back_reserved({"metric", nk_kernel_name(view.kernel())});
        for (safegraphs_custom_field_t const& field : save.provenance) fields_.push_back_reserved(field);
        return status_t::success_k;
    }

    /** Appends every non-empty part of @p view to the sources, refusing a geometry or a kernel
     *  other than those of the first part of @p first, the view the plan opened with. */
    status_t add_sources_(indexes_t const& view, indexes_t const& first) noexcept {
        for (u32_t part = 0; part != view.parts_.size(); ++part)
            if (status_t const added = add_part_(view, first, part); failed(added)) return added;
        return status_t::success_k;
    }

    /** Appends part @p part of @p view to the sources unless it is empty, as @ref add_sources_
     *  refuses it. */
    status_t add_part_(indexes_t const& view, indexes_t const& first, u32_t part) noexcept {
        nvme_part_t const& shape = first.parts_[0];
        if (view.kernel() != first.kernel()) return status_t::config_mismatch_k;
        nvme_part_t const& opened = view.parts_[part];
        if (opened.dimensions != shape.dimensions || opened.connectivity != shape.connectivity ||
            opened.value_row_bytes != shape.value_row_bytes || opened.key_dtype != safegraphs_key_dtype<key_t>())
            return status_t::config_mismatch_k;
        if (!opened.nodes) return status_t::success_k;
        source_t source;
        source.view = &view;
        source.part = part;
        source.rows = opened.nodes;
        source.stride = opened.nodes;
        return failed(sources_.push_back(source)) ? status_t::bad_alloc_k : status_t::success_k;
    }

    /** Names target part @c part of @c parts after the manifest's stem, as SafeGraphs numbers
     *  parts. */
    struct stem_part_paths_t {
        std::string_view stem;
        expected<std::string_view> operator()(paths_t& buffer, std::size_t part, std::size_t parts) const noexcept {
            return safegraphs_part_path(buffer, stem, part, parts);
        }
    };

    /** Names the one target part a part replacement writes at @c path. */
    struct replacing_part_path_t {
        std::string_view path;
        expected<std::string_view> operator()(paths_t&, std::size_t, std::size_t) const noexcept {
            return {path, status_t::success_k};
        }
    };

    /** The @ref open_targets_ of a rewrite committing a manifest, naming parts after the stem. */
    status_t open_targets_(std::span<std::size_t const> rows) noexcept {
        return open_targets_(rows, stem_part_paths_t {path_of_(stem_)});
    }

    /** Creates one target part per entry of @p rows at the path @p part_path names, with its header
     *  framed, its file sized and its one-part view open, and keeps a changed bit per row of them
     *  at the workspace's front. */
    template <typename part_path_type_>
    status_t open_targets_(std::span<std::size_t const> rows, part_path_type_ const& part_path) noexcept {
        if (failed(targets_.reserve_exactly(rows.size()))) return status_t::bad_alloc_k;
        std::size_t total = 0;
        for (std::size_t part = 0; part != rows.size(); ++part) {
            if (!rows[part] || rows[part] >= node_id_unknown_k) return status_t::overflow_k;
            monolith_layout_t const layout {rows[part],
                                            safegraphs_key_dtype<key_t>(),
                                            dtype_of(scalar_t::dtype()),
                                            dimensions_(),
                                            dtype_t::u32_k,
                                            connectivity_(),
                                            norms_bytes_() ? dtype_of(dot_result_t::dtype()) : dtype_t::unknown_k};
            targets_.push_back_reserved(
                target_t {paths_t {allocator_}, seekable_file_t {}, safegraphs_header_t {}, indexes_t {allocator_}});
            target_t& target = targets_.back();
            target.changed_first = total;
            total += rows[part];
            auto const [header, frame_status] = monolith_header(layout, save_.alignment, to_span(fields_),
                                                                target.norms);
            if (failed(frame_status)) return frame_status;
            target.header = header;
            target.selected = {0, rows[part]};
            target.view.kernel_ = kernel_();
            paths_t name {allocator_};
            auto const [path, path_status] = part_path(name, part, rows.size());
            if (failed(path_status)) return path_status;
            if (status_t const kept = keep_path_(target.path, path); failed(kept)) return kept;
            // The restitch tape takes the part's name and a suffix, so a round names it in place.
            if (failed(tape_path_.reserve_exactly(path.size() + 8))) return status_t::bad_alloc_k;
            auto [file, open_status] = seekable_file_t::open(path, file_access_t::truncate_read_write_k, allocator_);
            if (failed(open_status)) return open_status;
            target.file = std::move(file);
            if (status_t const sized = target.file.truncate(header.monolith_bytes()); failed(sized)) return sized;

            auto [file_medium, medium_status] = file_medium_t::open(target.path.data());
            if (failed(medium_status)) return medium_status;
            nvme_part_t viewed;
            if (status_t const derived = indexes_t::index_t::part_of_columns_(safegraphs_header_columns(header),
                                                                              dtype_t::unknown_k, viewed);
                failed(derived))
                return derived;
            if (failed(target.view.media_.push_back(medium_t {std::move(file_medium)})) ||
                failed(target.view.parts_.push_back(viewed)))
                return status_t::bad_alloc_k;
            auto const [identity, identity_status] = media_identity_from_entropy();
            if (failed(identity_status)) return identity_status;
            target.view.media_identity_ = identity;
        }
        std::size_t const changed_words = divide_round_up<std::size_t>(total, 64);
        std::size_t const changed_bytes = workspace_allocator_t::reserved_bytes(changed_words * sizeof(u64_t));
        if (changed_bytes > workspace_.size()) return status_t::bad_alloc_k;
        changed_ = {reinterpret_cast<u64_t*>(workspace_.data()), changed_words};
        std::memset(changed_.data(), 0, changed_.size_bytes());
        workspace_ = workspace_.subspan(changed_bytes);
        return size_tasks_();
    }

    /** Packs the source parts largest first into target parts of at most @p capacity rows, each
     *  into the first that fits, appending @p rows per target part. A part past the capacity is cut
     *  into ranges of it, each opening a target part of its own. Counts as moved every row but
     *  those of a part's first range arriving at row zero, which keep their IDs. */
    status_t pack_sources_(std::size_t capacity, growing_buffer<std::size_t, forkunion_allocator_t>& rows) noexcept {
        // Zero, or a capacity past the 32-bit ID space, means as many rows as a part can address.
        std::size_t const limit = capacity && capacity < node_id_unknown_k ? capacity : node_id_unknown_k - 1;
        partial_sort(to_writable_span(sources_), sources_.size(),
                     [](source_t const& first, source_t const& second) noexcept { return first.rows > second.rows; });
        for (source_t& source : sources_) {
            if (source.rows > limit) {
                source.target = rows.size();
                source.stride = limit;
                for (std::size_t first = 0; first < source.rows; first += limit)
                    if (failed(rows.push_back(min_of(limit, source.rows - first)))) return status_t::bad_alloc_k;
                result_.moved_entries += source.rows - limit;
                continue;
            }
            std::size_t bin = 0;
            while (bin != rows.size() && rows[bin] + source.rows > limit) ++bin;
            if (bin == rows.size() && failed(rows.push_back(0))) return status_t::bad_alloc_k;
            source.target = bin;
            source.base = rows[bin];
            rows[bin] += source.rows;
            if (source.base) result_.moved_entries += source.rows;
        }
        return status_t::success_k;
    }

    /** Refuses a workspace too small for a single row of a copy; a round refuses its own sizes when
     *  it opens, since the cache it sizes from is known only then. */
    status_t size_tasks_() const noexcept { return copy_rows_() ? status_t::success_k : status_t::bad_alloc_k; }

    /** The units the copy and build phases cut into tasks: source parts while copying, target parts
     *  while building. */
    std::size_t units_() const noexcept {
        switch (phase_) {
        case phase_t::copy_k: return sources_.size();
        case phase_t::build_k: return targets_.size();
        default: return 0;
        }
    }

    /** Rows of one unit of the current phase: @b [first,first+rows) cut at every @c stride rows
     *  past @c first, and into tasks of at most @c chunk rows. */
    struct cut_t {
        std::size_t first = 0;
        std::size_t rows = 0;
        std::size_t stride = 1;
        std::size_t chunk = 1;

        std::size_t tasks() const noexcept {
            return rows / stride * divide_round_up(stride, chunk) + divide_round_up(rows % stride, chunk);
        }

        /** The rows of task @p task of the unit. */
        iota_view<std::size_t> rows_of(std::size_t task) const noexcept {
            std::size_t const per_stride = divide_round_up(stride, chunk), range = task / per_stride;
            std::size_t const offset = range * stride + task % per_stride * chunk;
            return {first + offset, first + min_of(offset + chunk, min_of((range + 1) * stride, rows))};
        }
    };

    /** How the current phase cuts unit @p unit: a copy task stays inside one target range of its
     *  source part, and a build task is a block. */
    cut_t cut_(std::size_t unit) const noexcept {
        if (phase_ == phase_t::copy_k) {
            source_t const& source = sources_[unit];
            return {0, source.rows, source.stride, copy_rows_()};
        }
        std::size_t const nodes = targets_[unit].view.size(), block_rows = build_rows_();
        return {0, block_rows ? nodes : 0, nodes, max_of<std::size_t>(block_rows, 1)};
    }

    /** Whether the current phase belongs to a round, whose tasks lie in one target part. */
    bool rounding_() const noexcept {
        return phase_ != phase_t::copy_k && phase_ != phase_t::build_k && phase_ != phase_t::done_k;
    }

    /** Opens the round of the first target part from @p part on that calls for one, in RAM where
     *  the workspace holds it and from the opening health check otherwise; past the last part, it
     *  ends the plan. */
    void open_round_(std::size_t part) noexcept {
        while (part != targets_.size() && targets_[part].selected.empty()) ++part;
        round_part_ = part;
        part_rounds_ = 0;
        if (part == targets_.size()) {
            phase_ = phase_t::done_k;
            return;
        }
        health_after_ = phase_t::build_k;
        phase_ = in_ram_(part) ? phase_t::restitch_k : phase_t::health_k;
    }

    /** The round's tasks in the current phase: one health check, one draw, one streaming gather
     *  or one per neighborhood after it, one per neighborhood re-stitched or a whole part held in
     *  RAM, and one per tile of rows written. */
    std::size_t round_tasks_() const noexcept {
        switch (phase_) {
        case phase_t::gather_k: return neighborhoods_ ? neighborhoods_ : 1;
        case phase_t::restitch_k: return in_ram_(round_part_) ? 1 : neighborhoods_;
        case phase_t::write_k: return divide_round_up(targets_[round_part_].view.size(), tile_rows_);
        default: return 1;
        }
    }

    /** The round's task @p position: a neighborhood's positions in the order by neighborhood, a
     *  tile's rows, or the whole part; an empty one past the last. */
    task_t round_task_(std::size_t position) const noexcept {
        if (position >= round_tasks_()) return {};
        std::size_t const nodes = targets_[round_part_].view.size();
        bool const per_neighborhood = (phase_ == phase_t::gather_k && neighborhoods_) ||
                                      (phase_ == phase_t::restitch_k && !in_ram_(round_part_));
        if (per_neighborhood)
            return {position, round_part_, {neighborhood_offsets_[position], neighborhood_offsets_[position + 1]}};
        if (phase_ == phase_t::write_k) {
            std::size_t const first = position * tile_rows_;
            return {position,
                    round_part_,
                    {static_cast<node_id_t>(first), static_cast<node_id_t>(min_of(first + tile_rows_, nodes))}};
        }
        return {position, round_part_, {0, static_cast<node_id_t>(nodes)}};
    }

#pragma endregion Opening

#pragma region Sizes

    /** The first source part, whose geometry every source part and every target part shares. */
    nvme_part_t const& shape_() const noexcept { return sources_[0].view->parts_[sources_[0].part]; }
    std::size_t dimensions_() const noexcept { return shape_().dimensions; }
    std::size_t connectivity_() const noexcept { return shape_().connectivity; }
    std::size_t value_row_bytes_() const noexcept { return shape_().value_row_bytes; }

    /** The distance kind of every source, which the targets name and their selections rank by. */
    nk_kernel_kind_t kernel_() const noexcept { return sources_[0].view->kernel(); }

    /** Bytes of one norms row, zero when the kernel reads no norms and the targets hold no
     *  column. */
    std::size_t norms_bytes_() const noexcept { return sources_[0].view->stores_norms() ? sizeof(vector_norms_t) : 0; }

    std::size_t expansion_factor_() const noexcept {
        return config_.refine.expansion_factor ? config_.refine.expansion_factor
                                               : default_expansion_factor(connectivity_());
    }

    /** Rows one copy task covers, the most the workspace stages. */
    std::size_t copy_rows_() const noexcept {
        return fitting_rows_(&nvme_reshard_plan::copy_slices_, workspace_.size() / value_row_bytes_(),
                             workspace_.size());
    }

    /** Rows one build task covers, the largest block the workspace fits; none unless the plan
     *  builds over the views it opened. */
    std::size_t build_rows_() const noexcept {
        if (opened_.empty()) return 0;
        std::size_t rows = 0;
        for (source_t const& source : sources_) rows += source.rows;
        return fitting_rows_(&nvme_reshard_plan::build_slices_, rows, workspace_.size());
    }

    std::size_t neighbors_row_bytes_() const noexcept { return connectivity_() * sizeof(node_id_t); }

    /** Bytes one entry's row takes on the restitch tape: its node, then its links. */
    std::size_t tape_row_bytes_() const noexcept { return (1 + connectivity_()) * sizeof(node_id_t); }

    /** Whether target part @p part runs its round in RAM: its index, its round and one staged row
     *  fit the workspace. */
    bool in_ram_(std::size_t part) const noexcept {
        carver_t counter;
        refine_slices_(targets_[part].view.size(), 1, counter);
        return counter.bytes <= workspace_.size();
    }

    /** Bytes the columns a RAM round over @p rows rows keeps across its barriers take off the
     *  index's arena under the plan's cache, its hubs counted at their most: the changed bits, one
     *  pack of hubs with their nodes and norms, per entry its neighborhood and its place in the
     *  order by neighborhood, and the starts every split may write. */
    std::size_t round_bytes_(std::size_t rows, std::size_t row_bytes) const noexcept {
        using ram_scorer_t = typename ram_index_t::scan_scorer_t;
        std::size_t const hubs = ram_scorer_t::block_columns(rows, cache_bytes_, row_bytes);
        return workspace_allocator_t::reserved_bytes(divide_round_up<std::size_t>(rows, 64) * sizeof(u64_t)) +
               workspace_allocator_t::reserved_bytes(hubs * sizeof(node_id_t)) +
               workspace_allocator_t::reserved_bytes(
                   round_up_to_multiple<std::size_t>(hubs * row_bytes, default_alignment_k) +
                   ram_scorer_t::packs_bytes(hubs, dimensions_())) +
               workspace_allocator_t::reserved_bytes(hubs * sizeof(vector_norms_t)) +
               workspace_allocator_t::reserved_bytes(rows * sizeof(u16_t)) +
               workspace_allocator_t::reserved_bytes(rows * sizeof(node_id_t)) +
               workspace_allocator_t::reserved_bytes(split_room(rows, hubs) * sizeof(node_id_t));
    }

    /** Bytes the columns of a RAM index of @p rows rows take off the arena it is built on: a slice
     *  per column of every chunk the rows reach, the key map's control bytes and slots over them,
     *  and the audit and the round its refine and heal run on the same allocator. */
    std::size_t ram_bytes_(std::size_t rows) const noexcept {
        using schedule_t = typename ram_index_t::chunk_schedule_t;
        using slots_schedule_t = typename ram_index_t::keyed_t::nodes_map_t::slots_schedule_t;
        typename ram_index_t::geometry_t const geometry {dimensions_(), static_cast<connectivity_t>(connectivity_())};
        std::size_t const values_bytes = ram_index_t::padded_row_values(geometry) * sizeof(scalar_t);
        std::size_t bytes = workspace_allocator_t::reserved_bytes(ram_index_t::audit_memory_usage(rows)) +
                            round_bytes_(rows, values_bytes);
        std::size_t const chunks = schedule_t::chunks_for(rows);
        for (std::size_t chunk = 0; chunk != chunks; ++chunk) {
            std::size_t const chunk_rows = schedule_t::nodes_in(chunk);
            bytes += workspace_allocator_t::reserved_bytes(chunk_rows * values_bytes) +
                     workspace_allocator_t::reserved_bytes(chunk_rows * norms_bytes_()) +
                     workspace_allocator_t::reserved_bytes(chunk_rows * neighbors_row_bytes_()) +
                     workspace_allocator_t::reserved_bytes(chunk_rows * sizeof(node_id_t));
        }
        // The map holds every covered row and probes twice as many slots, each a key beside its
        // node.
        for (std::size_t chunk = 0, map_chunks = slots_schedule_t::chunks_for(2 * schedule_t::nodes_before(chunks));
             chunk != map_chunks; ++chunk) {
            std::size_t const slots = slots_schedule_t::nodes_in(chunk);
            bytes += workspace_allocator_t::reserved_bytes(slots) +
                     workspace_allocator_t::reserved_bytes(slots * 2 * sizeof(node_id_t));
        }
        return bytes;
    }

    /** The most rows, at most @p rows, whose slices @p slices_of lays out within @p bytes. */
    template <typename slices_of_type_>
    std::size_t fitting_rows_(slices_of_type_ slices_of, std::size_t rows, std::size_t bytes) const noexcept {
        std::size_t low = 0, high = rows;
        while (low < high) {
            std::size_t const middle = high - (high - low) / 2;
            carver_t counter;
            (this->*slices_of)(middle, counter);
            if (counter.bytes <= bytes) low = middle;
            else high = middle - 1;
        }
        return low;
    }

    /** Value rows one pack block of @p columns holds under the plan's cache, at least one. */
    std::size_t block_columns_(std::size_t columns) const noexcept {
        return max_of<std::size_t>(scan_scorer_t::block_columns(columns, cache_bytes_, value_row_bytes_()), 1);
    }

    /** Bytes the packs of @p columns staged rows take, one pack per block. */
    std::size_t packs_bytes_(std::size_t columns) const noexcept {
        std::size_t const block_columns = block_columns_(columns);
        return divide_round_up(columns, block_columns) * scan_scorer_t::packs_bytes(block_columns, dimensions_());
    }

#pragma endregion Sizes

#pragma region Slices

    /** An arena whose only chunk is the whole of @p space, which starts on a boundary. */
    static workspace_arena_t arena_over_(bytes_span_t space) noexcept {
        workspace_arena_t arena {workspace_upstream_t {space}};
        // Never refused: the upstream hands out exactly the span it passes.
        [[maybe_unused]] status_t const seeded = arena.reserve_exactly(space.size());
        return arena;
    }

    /** Takes @p count elements on the next boundary off the arena of @p carver, counting the bytes
     *  they reserve; null when the arena is short or absent. */
    template <typename value_type_>
    static value_type_* carve_(carver_t& carver, std::size_t count) noexcept {
        carver.bytes += workspace_allocator_t::reserved_bytes(count * sizeof(value_type_));
        if (!carver.arena) return nullptr;
        return arena_allocator<value_type_, workspace_upstream_t>(*carver.arena).allocate(count);
    }

    copy_slices_t copy_slices_(std::size_t rows, carver_t& carver) const noexcept {
        return {carve_<byte_t>(carver, rows * sizeof(key_t)), carve_<byte_t>(carver, rows * value_row_bytes_()),
                carve_<byte_t>(carver, rows * norms_bytes_()), carve_<node_id_t>(carver, rows * connectivity_())};
    }

    /** The RAM index of a block of @p rows rows takes its columns off the same arena after the
     *  slices. */
    build_slices_t build_slices_(std::size_t rows, carver_t& carver) const noexcept {
        build_slices_t const slices {carve_<byte_t>(carver, rows * value_row_bytes_()), carve_<node_id_t>(carver, rows),
                                     carve_<node_id_t>(carver, rows * connectivity_()), carve_<ram_index_t>(carver, 1)};
        carver.bytes += ram_bytes_(rows);
        return slices;
    }

    /** The RAM index of a part of @p rows rows, @p staged rows of neighbors on their way out, and
     *  the columns the index then takes off the same arena. */
    refine_slices_t refine_slices_(std::size_t rows, std::size_t staged, carver_t& carver) const noexcept {
        refine_slices_t const slices {carve_<ram_index_t>(carver, 1),
                                      carve_<node_id_t>(carver, staged * connectivity_())};
        carver.bytes += ram_bytes_(rows);
        return slices;
    }

    /** The columns of a streamed round over @p rows rows drawing at most @p hubs_per_draw hubs a
     *  gather, writing @p tiles tiles of rows. */
    round_slices_t round_slices_(std::size_t rows, std::size_t hubs_per_draw, std::size_t tiles,
                                 carver_t& carver) const noexcept {
        return {carve_<u8_t>(carver, 4 * rows),
                carve_<u64_t>(carver, audit_columns_t::bitsets_k * audit_columns_t::words_for(rows)),
                carve_<u16_t>(carver, rows),
                carve_<node_id_t>(carver, rows),
                carve_<node_id_t>(carver, split_room(rows, hubs_per_draw)),
                carve_<std::size_t>(carver, tiles)};
    }

    /** One pack of @p hubs hubs. */
    hub_slices_t hub_slices_(std::size_t hubs, carver_t& carver) const noexcept {
        return {carve_<node_id_t>(carver, hubs), carve_<byte_t>(carver, hubs * value_row_bytes_()),
                carve_<vector_norms_t>(carver, hubs),
                carve_<byte_t>(carver, scan_scorer_t::packs_bytes(hubs, dimensions_()))};
    }

    /** One neighborhood of @p width entries, with its links and every one leading out staged. */
    restitch_slices_t restitch_slices_(std::size_t width, carver_t& carver) const noexcept {
        std::size_t const connectivity = connectivity_();
        return {carve_<byte_t>(carver, width * (1 + connectivity) * value_row_bytes_()),
                carve_<vector_norms_t>(carver, width),
                carve_<byte_t>(carver, packs_bytes_(width)),
                carve_<node_id_t>(carver, width * connectivity),
                carve_<u32_t>(carver, width * connectivity),
                carve_<scored_node_t>(carver, width * connectivity),
                carve_<node_id_t>(carver, width * (1 + connectivity))};
    }

    /** One tile of @p rows neighbor rows and as many rows off the tape. */
    write_slices_t write_slices_(std::size_t rows, carver_t& carver) const noexcept {
        return {carve_<node_id_t>(carver, rows * connectivity_()),
                carve_<node_id_t>(carver, rows * (1 + connectivity_()))};
    }

#pragma endregion Slices

#pragma region Scans and Writes

    /** Scans rows [first, first + count) of every column of @p columns in part @p part of @p view
     *  into their staging, as runs of whole rows, in a call of its own. */
    status_t scan_rows_(indexes_t const& view, u32_t part, std::size_t first, std::size_t count,
                        std::span<staged_column_t const> columns, executor_t& executor) noexcept {
        if (!count || columns.empty()) return status_t::success_k;
        auto [call, call_status] = view.begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        return read_rows_(view, part, first, count, columns, executor);
    }

    /** The @c scan_rows_ of a task whose call is already open, a scan's worth of slots
     *  begun. */
    status_t read_rows_(indexes_t const& view, u32_t part, std::size_t first, std::size_t count,
                        std::span<staged_column_t const> columns, executor_t& executor) noexcept {
        auto run_rows_of = [&](staged_column_t const& column) noexcept {
            return min_of(view.scan_run_rows_(part, column.row_bytes), count);
        };
        std::size_t tasks = 0;
        for (staged_column_t const& column : columns) tasks += divide_round_up(count, run_rows_of(column));
        // Task t is run t of the first column, then run t of the next once the first runs out.
        auto locate = [&](std::size_t task, std::size_t& run) noexcept -> staged_column_t const& {
            for (staged_column_t const& column : columns) {
                std::size_t const runs = divide_round_up(count, run_rows_of(column));
                if (task < runs) return run = task, column;
                task -= runs;
            }
            return run = 0, columns.back();
        };
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
            if (progress++) return {};
            std::size_t run = 0;
            staged_column_t const& column = locate(task, run);
            std::size_t const run_rows = run_rows_of(column);
            std::size_t const row = run * run_rows, rows = min_of(run_rows, count - row);
            return {nvme_extent_t {column.begin + u64_t(first + row) * column.row_bytes,
                                   static_cast<u32_t>(rows * column.row_bytes), part}};
        };
        auto consume = [&](std::size_t, std::size_t task, u64_t, auto const& extents, auto const& fetched) noexcept {
            std::size_t run = 0;
            staged_column_t const& column = locate(task, run);
            std::memcpy(column.staging + run * run_rows_of(column) * column.row_bytes, fetched[0], extents[0].length);
        };
        return view.scan_reads_(executor, tasks, describe, consume).status;
    }

    /** Reads one row of the column at @p begin, @p row_bytes wide, of part zero of @p view per
     *  task, inside a call already open: row <tt>row_of(task)</tt> reaches
     *  <tt>consume_row(thread, task, bytes)</tt> on the thread that fetched it. */
    template <typename row_of_type_, typename consume_row_type_>
    status_t read_points_(indexes_t const& view, u64_t begin, std::size_t row_bytes, std::size_t tasks,
                          row_of_type_&& row_of, consume_row_type_&& consume_row, executor_t& executor) noexcept {
        if (!tasks) return status_t::success_k;
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
            if (progress++) return {};
            return {nvme_extent_t {begin + u64_t(row_of(task)) * row_bytes, static_cast<u32_t>(row_bytes), 0}};
        };
        auto consume = [&](std::size_t thread, std::size_t task, u64_t, auto const&, auto const& fetched) noexcept {
            consume_row(thread, task, fetched[0]);
        };
        return view.scan_reads_(executor, tasks, describe, consume).status;
    }

    /** The @c read_points_ that copies row @c row_of(task) to @c destination_of(task). */
    template <typename row_of_type_, typename destination_of_type_>
    status_t copy_points_(indexes_t const& view, u64_t begin, std::size_t row_bytes, std::size_t tasks,
                          row_of_type_&& row_of, destination_of_type_&& destination_of, executor_t& executor) noexcept {
        return read_points_(
            view, begin, row_bytes, tasks, row_of,
            [&](std::size_t, std::size_t task, byte_t const* row) noexcept {
                std::memcpy(destination_of(task), row, row_bytes);
            },
            executor);
    }

    /** Runs @p per_task over @p tasks tasks on the threads of @p executor, claimed for the
     *  fan-out. */
    template <typename per_task_type_>
    static status_t fan_out_(executor_t& executor, std::size_t tasks, per_task_type_&& per_task) noexcept {
        auto [claim, claim_status] = executor.claim();
        if (failed(claim_status)) return claim_status;
        return executor.for_tasks(tasks, per_task);
    }

    /** Writes @p rows rows of @p row_bytes bytes from @p bytes into the column of target part
     *  @p target at @p column, from row @p first on: one positional write. */
    status_t write_rows_(target_t& target, u64_t column, std::size_t row_bytes, std::size_t first, std::size_t rows,
                         byte_t const* bytes) noexcept {
        if (!rows || !row_bytes) return status_t::success_k;
        return target.file.write_at(column + first * row_bytes, bytes_view_t {bytes, rows * row_bytes});
    }

    /** Where the norms column of target part @p target starts. */
    static u64_t norms_begin_(target_t const& target) noexcept {
        return target.header.data_offset + target.norms.column.data.first;
    }

    /** The one part of target part @p target. */
    static nvme_part_t const& part_of_(target_t const& target) noexcept { return target.view.parts_[0]; }

#pragma endregion Scans and Writes

#pragma region Selection

    /** The norms the kernel reads of the row whose bytes start at @p bytes, or none. */
    vector_norms_t norms_of_(byte_t const* bytes) const noexcept {
        return norms_bytes_() ? dense::norms<typename indexes_t::quads_kit_t>(view_of_(bytes)) : vector_norms_t {};
    }
    vector_view_t view_of_(byte_t const* bytes) const noexcept {
        return vector_view_t {reinterpret_cast<scalar_t const*>(bytes), dimensions_()};
    }

    /** @p count value rows staged from @p bytes as the scorer reads them, a value row apart. */
    vectors_view_t rows_view_(byte_t const* bytes, std::size_t count) const noexcept {
        vectors_view_t const packed {reinterpret_cast<scalar_t const*>(bytes), count, dimensions_()};
        nk::shape_storage_<2> const strided {{count, dimensions_()},
                                             {static_cast<std::ptrdiff_t>(value_row_bytes_()), packed.stride_bytes(1)},
                                             2};
        return {packed.byte_data(), strided};
    }

    /** The exact distance between two rows. */
    f32_t distance_(byte_t const* first, vector_norms_t first_norms, byte_t const* second,
                    vector_norms_t second_norms) const noexcept {
        return dense::distance<typename indexes_t::quads_kit_t>(kernel_(), view_of_(first), first_norms,
                                                                view_of_(second), second_norms);
    }

    /** Marks row @p row of @p target counted as changed, answering whether it was already. */
    bool count_changed_(target_t const& target, std::size_t row) noexcept {
        std::size_t const bit = target.changed_first + row;
        u64_t const mask = u64_t {1} << (bit % 64);
        return atomic_ref<u64_t>(changed_[bit / 64]).fetch_or(mask, std::memory_order_relaxed) & mask;
    }

    /** Adds @p value to the count at @p into, which several threads may add to at once. */
    static void add_(std::size_t& into, std::size_t value) noexcept {
        if (value) atomic_ref<std::size_t>(into).store_add(value, std::memory_order_relaxed);
    }

    /** Whether @p left lies farther than @p right, the larger ID farther among equals. */
    static constexpr auto further_ = [](scored_node_t const& left, scored_node_t const& right) noexcept {
        return left.distance > right.distance || (left.distance == right.distance && left.node_id > right.node_id);
    };

    /** An empty place in a kept heap, farther than any entry. */
    static constexpr scored_node_t farthest_ {node_id_unknown_k, std::numeric_limits<distance_t>::infinity()};

#pragma endregion Selection

#pragma region Reach

    /** The depth no health check reached, 255; the depths saturate at 254. */
    inline static constexpr u8_t unknown_depth_k = (std::numeric_limits<u8_t>::max)();

    /** The round's health check's bitsets, in the order @c nvme_audit_reachability lays them. */
    enum bitset_t : std::size_t { reached_k, reaches_root_k, two_ways_to_root_k };

    std::size_t audit_word_(bitset_t bitset, std::size_t node) const noexcept {
        return bitset * audit_columns_t::words_for(targets_[round_part_].view.size()) + node / audit_columns_t::lanes_k;
    }
    bool audit_contains_(bitset_t bitset, std::size_t node) const noexcept {
        u64_t const word =
            atomic_ref<u64_t>(audit_.bitset_words[audit_word_(bitset, node)]).load(std::memory_order_relaxed);
        return (word >> (node % audit_columns_t::lanes_k)) & 1u;
    }
    bool release_two_ways_to_root_(std::size_t node) noexcept {
        return atomic_ref<u64_t>(audit_.bitset_words[audit_word_(two_ways_to_root_k, node)])
            .fetch_test_and_reset_bit(static_cast<unsigned>(node % audit_columns_t::lanes_k),
                                      std::memory_order_relaxed);
    }
    bool in_component_(std::size_t node) const noexcept {
        return audit_contains_(reached_k, node) && audit_contains_(reaches_root_k, node);
    }
    u8_t depth_from_root_(std::size_t node) const noexcept {
        return atomic_ref<u8_t>(audit_.depths_from_root[node]).load(std::memory_order_relaxed);
    }
    u8_t depth_to_root_(std::size_t node) const noexcept {
        return atomic_ref<u8_t>(audit_.depths_to_root[node]).load(std::memory_order_relaxed);
    }
    u8_t in_links_(std::size_t node) const noexcept {
        return atomic_ref<u8_t>(audit_.in_links[node]).load(std::memory_order_relaxed);
    }
    void count_in_link_(std::size_t node) noexcept {
        fu::atomic_fetch_add_if_at_most(atomic_ref<u8_t>(audit_.in_links[node]), u8_t {1}, unknown_depth_k,
                                        std::memory_order_relaxed);
    }

    /** Takes one of @p node 's in-links, answering whether another remains. */
    bool take_in_link_if_spare_(std::size_t node) noexcept {
        return fu::atomic_fetch_sub_if_at_least(atomic_ref<u8_t>(audit_.in_links[node]), u8_t {1}, u8_t {1},
                                                std::memory_order_relaxed) > 1;
    }

    /** Takes one of @p node 's links from the depth before it, answering whether one remains. */
    bool take_shallower_in_link_if_spare_(std::size_t node) noexcept {
        atomic_ref<u8_t> const shallower_in_links {audit_.shallower_in_links[node]};
        u8_t observed = shallower_in_links.load(std::memory_order_relaxed);
        while (observed > 1 &&
               !shallower_in_links.compare_exchange_weak(observed, static_cast<u8_t>(observed - 1),
                                                         std::memory_order_relaxed, std::memory_order_relaxed)) {}
        return observed > 1;
    }

    /** The child depth of an entry at @p depth, none past the saturated one. */
    static u8_t child_depth_(u8_t depth) noexcept {
        return depth < unknown_depth_k - 1 ? static_cast<u8_t>(depth + 1) : unknown_depth_k;
    }

    /** Takes the link from @p owner to @p resident for a swap: one not from the depth before the
     *  resident goes freely, a shallower in-link only while another remains. */
    bool take_link_for_swap_if_spare_(std::size_t owner, std::size_t resident) noexcept {
        u8_t const child_depth = child_depth_(depth_from_root_(owner));
        return child_depth == unknown_depth_k || depth_from_root_(resident) != child_depth ||
               take_shallower_in_link_if_spare_(resident);
    }

    /** Takes the link from @p owner to @p resident for a heal's swap, answering whether the reach
     *  rule releases it: never the resident's last shallower in-link, and a link toward the root
     *  only while the owner keeps another at that depth to root. */
    bool take_link_under_lock_if_spare_(std::size_t owner, std::size_t resident) noexcept {
        bool const way_to_root = audit_contains_(reaches_root_k, owner) && depth_to_root_(owner) != 0 &&
                                 depth_to_root_(resident) < depth_to_root_(owner);
        if (way_to_root && !audit_contains_(two_ways_to_root_k, owner)) return false;
        if (!take_link_for_swap_if_spare_(owner, resident)) return false;
        if (way_to_root) release_two_ways_to_root_(owner);
        return true;
    }

    /** Files a heal's link from @p owner to island entry @p member in the depths a restitch keeps
     *  reach by, so no restitch drops it. */
    void record_heal_link_(std::size_t owner, std::size_t member) noexcept {
        u8_t const owner_depth = depth_from_root_(owner);
        if (depth_from_root_(member) == unknown_depth_k && owner_depth < unknown_depth_k - 1) {
            atomic_ref<u8_t>(audit_.depths_from_root[member])
                .store(static_cast<u8_t>(owner_depth + 1), std::memory_order_relaxed);
            atomic_ref<u8_t>(audit_.shallower_in_links[member]).store(1, std::memory_order_relaxed);
        }
        u8_t const member_depth_to_root = depth_to_root_(member);
        if (depth_to_root_(owner) == unknown_depth_k && member_depth_to_root < unknown_depth_k - 1)
            atomic_ref<u8_t>(audit_.depths_to_root[owner])
                .store(static_cast<u8_t>(member_depth_to_root + 1), std::memory_order_relaxed);
    }

    /** Puts back what the @p count picks at @p selected dropped of the @p live links that
     *  reachability needs, as the RAM round does: a link whose last shallower in-link @p node is,
     *  and, when no pick stands one depth to root closer, the row's nearest live link that does. A
     *  short row grows by each; a full one gives up its farthest pick kept by neither. */
    void keep_reach_(std::size_t node, std::span<scored_node_t const> live, scored_node_t* selected,
                     std::size_t& count) noexcept {
        u8_t const depth_to_root = depth_to_root_(node), child_depth = child_depth_(depth_from_root_(node));
        auto position_in = [](scored_node_t const* among, std::size_t size, node_id_t member) noexcept {
            std::size_t position = 0;
            while (position != size && among[position].node_id != member) ++position;
            return position;
        };
        auto releasable = [&](node_id_t member) noexcept {
            return child_depth == unknown_depth_k || depth_from_root_(member) != child_depth ||
                   position_in(live.data(), live.size(), member) == live.size() ||
                   take_shallower_in_link_if_spare_(member);
        };
        // The kept links gather at the tail, ahead of which the farthest releasable pick gives way.
        std::size_t kept = 0;
        auto keep = [&](scored_node_t member) noexcept {
            if (position_in(selected, count, member.node_id) != count) return;
            if (count < connectivity_()) {
                selected[count++] = member;
                ++kept;
                return;
            }
            std::size_t const tail = count - kept;
            std::size_t position = tail;
            while (position && !releasable(selected[position - 1].node_id)) --position;
            if (!position--) return;
            selected[position] = member;
            std::swap(selected[position], selected[tail - 1]);
            ++kept;
        };
        if (child_depth != unknown_depth_k)
            for (scored_node_t const member : live) {
                if (depth_from_root_(member.node_id) != child_depth ||
                    position_in(selected, count, member.node_id) != count)
                    continue;
                if (!take_shallower_in_link_if_spare_(member.node_id)) keep(member);
            }
        if (depth_to_root != unknown_depth_k && depth_to_root != 0) {
            std::size_t standing = 0;
            while (standing != count && depth_to_root_(selected[standing].node_id) >= depth_to_root) ++standing;
            scored_node_t way_to_root = standing != count ? selected[standing] : farthest_;
            for (std::size_t position = 0; position != live.size() && standing == count; ++position)
                if (depth_to_root_(live[position].node_id) < depth_to_root &&
                    live[position].distance < way_to_root.distance)
                    way_to_root = live[position];
            if (way_to_root.node_id != node_id_unknown_k) keep(way_to_root);
        }
    }

#pragma endregion Reach

#pragma region Copy

    /** Rewires the neighbors of source row @p row of @p source in place onto the target IDs of its
     *  range, padding with sentinels, and answers how many edges were dropped for arriving in a
     *  part other than its own. */
    std::size_t rewire_(source_t const& source, std::size_t row, node_id_t* neighbors) const noexcept {
        std::size_t const range = row / source.stride;
        std::size_t const low = max_of(source.kept.first, range * source.stride);
        std::size_t const high = min_of(source.kept.bound, (range + 1) * source.stride);
        std::size_t kept = 0, cut = 0;
        for (std::size_t slot = 0; slot != connectivity_(); ++slot) {
            std::size_t const neighbor = neighbors[slot];
            if (neighbor == row) continue;
            if (neighbor >= low && neighbor < high)
                neighbors[kept++] = static_cast<node_id_t>(source.base + neighbor - range * source.stride);
            else cut += !source.kept.empty() && neighbor < source.rows;
        }
        for (std::size_t slot = kept; slot != connectivity_(); ++slot) neighbors[slot] = sentinel_k;
        return cut;
    }

    /** Copies one block of a source part into the target: keys and values are scanned in, norms are
     *  derived, neighbors rewired, and the target part the rows arrive in takes one positional
     *  write per column. */
    status_t copy_(task_t const& task, executor_t& executor) noexcept {
        source_t const& source = sources_[task.part];
        nvme_part_t const& part = source.view->parts_[source.part];
        std::size_t const count = task.nodes.size(), first = task.nodes.first;
        std::size_t const value_bytes = value_row_bytes_(), neighbor_bytes = neighbors_row_bytes_();
        workspace_arena_t arena = arena_over_(workspace_);
        carver_t carver {&arena};
        auto const [keys, values, norms, neighbors] = copy_slices_(count, carver);
        if (carver.bytes > workspace_.size()) return status_t::bad_alloc_k;

        std::array<staged_column_t, 3> columns {
            {{part.keys_begin, sizeof(key_t), keys},
             {part.values_begin, value_bytes, values},
             {part.neighbors_begin, neighbor_bytes, reinterpret_cast<byte_t*>(neighbors)}}};
        // A part whose edges never survive reads no neighbors at all.
        std::size_t const scanned = source.kept.empty() ? 2 : 3;
        if (status_t const read = scan_rows_(*source.view, source.part, first, count,
                                             std::span<staged_column_t const>(columns).first(scanned), executor);
            failed(read))
            return read;

        std::size_t cut = 0;
        status_t const prepared = fan_out_(executor, count, [&](std::size_t row, std::size_t, std::size_t) noexcept {
            if (norms_bytes_()) {
                vector_norms_t const derived = norms_of_(values + row * value_bytes);
                std::memcpy(norms + row * norms_bytes_(), &derived, norms_bytes_());
            }
            node_id_t* const row_neighbors = neighbors + row * connectivity_();
            std::size_t const row_cut = source.kept.empty() ? 0 : rewire_(source, first + row, row_neighbors);
            if (row_cut) atomic_ref<std::size_t>(cut).store_add(row_cut, std::memory_order_relaxed);
        });
        if (failed(prepared)) return prepared;
        result_.cut_edges += cut;

        std::size_t const range = first / source.stride;
        target_t& target = targets_[source.target + range];
        nvme_part_t const& written = part_of_(target);
        std::size_t const target_first = source.base + first - range * source.stride;
        status_t wrote = write_rows_(target, written.keys_begin, sizeof(key_t), target_first, count, keys);
        if (succeeded(wrote))
            wrote = write_rows_(target, written.values_begin, value_bytes, target_first, count, values);
        if (succeeded(wrote))
            wrote = write_rows_(target, norms_begin_(target), norms_bytes_(), target_first, count, norms);
        // A block build writes every row's neighbors itself.
        if (succeeded(wrote) && !source.kept.empty())
            wrote = write_rows_(target, written.neighbors_begin, neighbor_bytes, target_first, count,
                                reinterpret_cast<byte_t const*>(neighbors));
        return wrote;
    }

#pragma endregion Copy

#pragma region RAM Blocks

    /** One traversal per executor thread over its own arena, sized for @p index, listed in the
     *  first arena; runs inside the call its task began, which rewound the arenas. */
    status_t traversals_(ram_index_t& index, executor_t& executor, traversals_t& traversals) noexcept {
        typename ram_index_t::insert_config_t const config {static_cast<u32_t>(expansion_factor_())};
        if (failed(traversals.reserve_exactly(executor.size()))) return status_t::bad_alloc_k;
        for (std::size_t thread = 0; thread != executor.size(); ++thread) {
            traversals.push_back_reserved(
                traversal_context_t {typename traversal_context_t::allocator_t {executor.arena(thread)}});
            traversals[thread].random_generator.state = mix_bits(thread + 1);
            if (status_t const sized = index.reserve_traversal(traversals[thread], config); failed(sized)) return sized;
        }
        return status_t::success_k;
    }

    /** Folds the telemetry a RAM index's traversals ran up into the result. */
    void fold_traversals_(traversals_t const& traversals) noexcept {
        for (traversal_context_t const& traversal : traversals) result_.graph.traversal_stats += traversal.telemetry();
    }

    /** Writes the neighbors of RAM rows [0, rows) of @p index as target rows from @p first on,
     *  every ID through @p target_of and every row through @p node_of, staged in @p staged and
     *  written in runs of @p run_rows rows. */
    template <typename node_of_type_, typename target_of_type_>
    status_t write_ram_rows_(ram_index_t const& index, target_t& target, std::size_t first, std::size_t rows,
                             node_id_t* staged, std::size_t run_rows, executor_t& executor, node_of_type_&& node_of,
                             target_of_type_&& target_of) noexcept {
        if (!run_rows) return status_t::bad_alloc_k;
        ram_cursor_t cursor {index};
        if (status_t const reserved = cursor.reserve(arena_allocator<byte_t, forkunion_allocator_t>(executor.arena(0)));
            failed(reserved))
            return reserved;
        for (std::size_t run = 0; run < rows; run += run_rows) {
            std::size_t const run_count = min_of(run_rows, rows - run);
            for (std::size_t row = 0; row != run_count; ++row) {
                node_id_t* const out = staged + row * connectivity_();
                std::size_t written = 0;
                if (succeeded(cursor.seek(node_of(run + row))))
                    for (node_id_t const neighbor : cursor.neighbors()) out[written++] = target_of(neighbor);
                for (; written != connectivity_(); ++written) out[written] = sentinel_k;
            }
            if (status_t const wrote = write_rows_(target, part_of_(target).neighbors_begin, neighbors_row_bytes_(),
                                                   first + run, run_count, reinterpret_cast<byte_t const*>(staged));
                failed(wrote))
                return wrote;
        }
        return status_t::success_k;
    }

    /** Builds target rows @p task.nodes in a RAM index inside the workspace from their value rows,
     *  heals it there, and writes their neighbors, every edge inside the block. */
    status_t build_(task_t const& task, executor_t& executor) noexcept {
        target_t& target = targets_[task.part];
        std::size_t const count = task.nodes.size(), first = task.nodes.first;
        workspace_arena_t arena = arena_over_(workspace_);
        carver_t carver {&arena};
        auto const [values, nodes, staging, index_bytes] = build_slices_(count, carver);
        if (carver.bytes > workspace_.size()) return status_t::bad_alloc_k;
        typename ram_index_t::geometry_t const geometry {dimensions_(), static_cast<connectivity_t>(connectivity_())};
        ram_index_ptr_t const index_ptr {new (index_bytes) ram_index_t(
            kernel_(), geometry, workspace_allocator_t(arena), workspace_allocator_t(arena))};
        ram_index_t& index = *index_ptr;
        if (status_t const reserved = index.reserve({count}).status; failed(reserved)) return reserved;

        // One call holds the executor for the block's scan, its inserts, its heal and its writes.
        typename ram_index_t::insert_config_t const config {static_cast<u32_t>(expansion_factor_())};
        auto [call, call_status] = target.view.begin_scans_(executor, 1, index.traversal_memory_usage(config));
        if (failed(call_status)) return call_status;
        std::array<staged_column_t, 1> const columns {{{part_of_(target).values_begin, value_row_bytes_(), values}}};
        if (status_t const read = read_rows_(target.view, 0, first, count, columns, executor); failed(read))
            return read;
        traversals_t traversals {scratch_allocator_t {executor.arena(0)}};
        if (status_t const ready = traversals_(index, executor, traversals); failed(ready)) return ready;
        status_t refusal = status_t::success_k;
        auto refuse = [&](status_t status) noexcept { keep_first_failure<atomic_ref>(refusal, status); };
        // One row per task, as a RAM group inserts, so the threads interleave from the first row.
        auto insert = [&](std::size_t row, std::size_t thread, std::size_t) noexcept {
            vectors_view_t const inserted_row {reinterpret_cast<scalar_t const*>(values + row * value_row_bytes_()),
                                               std::size_t {1}, dimensions_()};
            node_id_t const block_row = static_cast<node_id_t>(row);
            refuse(index
                       .insert(typename ram_index_t::keys_view_t {&block_row, std::size_t {1}}, inserted_row,
                               typename ram_index_t::node_ids_span_t {nodes + row, std::size_t {1}}, traversals[thread],
                               config)
                       .status);
        };
        if (status_t const fanned = executor.for_tasks(count, insert); failed(fanned)) return fanned;
        if (failed(refusal)) return refusal;
        // A built block heals and stops at its hubs, so the heal's links stand and the closing
        // health check measures them: a build stays a plain build.
        typename ram_index_t::refine_config_t healing_config {static_cast<u32_t>(expansion_factor_())};
        healing_config.cache_bytes = cache_bytes_;
        auto [healing, heal_status] = index.plan_refine(healing_config);
        if (failed(heal_status)) return heal_status;
        do {
            if (healing.phase() == phase_t::hubs_k) healing.stop();
            auto const tasks = healing.tasks();
            auto heal = [&](std::size_t position, std::size_t thread, std::size_t) noexcept {
                refuse(healing.run(tasks[position], traversals[thread]));
            };
            if (status_t const fanned = executor.for_tasks(tasks.size(), heal); failed(fanned)) return fanned;
            if (failed(refusal)) return refusal;
        } while (healing.advance());
        // A stopped round finishes cancelled, its heal standing, and never reaches the scratch a
        // context reserves for gathers and restitches; only a failure ends the block.
        if (failed(healing.status())) return healing.status();
        refine_report_t const healed = healing.finish().value;
        fold_traversals_(traversals);
        result_.graph.reselected_nodes += count;
        result_.graph.changed_nodes += healed.changed_nodes;
        result_.graph.islands_found += healed.islands_found;
        result_.graph.islands_reconnected += healed.islands_reconnected;
        typename ram_index_t::keyed_t const& keys = index.keys();
        return write_ram_rows_(
            index, target, first, count, staging, count, executor, [&](std::size_t row) noexcept { return nodes[row]; },
            [&](node_id_t node) noexcept { return static_cast<node_id_t>(first + keys[node]); });
    }

    /** Runs the round of target part @p part in a RAM index inside the workspace: its value and
     *  neighbors columns load through a read-only mapping, one RAM round heals and re-stitches
     *  every entry, and the rows write back. */
    status_t refine_in_ram_(std::size_t part, executor_t& executor) noexcept {
        target_t& target = targets_[part];
        std::size_t const nodes = target.view.size();
        auto [mapping, map_status] = mapped_file_t::read(path_of_(target.path));
        if (failed(map_status)) return map_status;
        bytes_view_t const bytes = mapping.view();
        safegraphs_header_t const& header = target.header;
        typename ram_index_t::columns_view_t const columns {header.data_of(bytes, header.value),
                                                            header.data_of(bytes, header.neighbors),
                                                            {}};
        // The neighbors on their way out stage in whatever the index leaves.
        carver_t fixed;
        refine_slices_(nodes, 0, fixed);
        std::size_t const staged = min_of(
            nodes, (workspace_.size() - min_of(fixed.bytes, workspace_.size())) / neighbors_row_bytes_());
        workspace_arena_t arena = arena_over_(workspace_);
        carver_t carver {&arena};
        refine_slices_t const slices = refine_slices_(nodes, staged, carver);
        if (carver.bytes > workspace_.size()) return status_t::bad_alloc_k;
        typename ram_index_t::geometry_t const geometry {dimensions_(), static_cast<connectivity_t>(connectivity_())};
        ram_index_ptr_t const index_ptr {new (slices.index) ram_index_t(
            kernel_(), geometry, workspace_allocator_t(arena), workspace_allocator_t(arena))};
        ram_index_t& index = *index_ptr;
        if (status_t const reserved = index.reserve({nodes}).status; failed(reserved)) return reserved;
        // One call holds the executor for the load, the refine and the writes; it reads no medium.
        typename ram_index_t::insert_config_t const sizing {static_cast<u32_t>(expansion_factor_())};
        auto [call, call_status] = nvme_begin_call(executor, std::span<medium_t const> {}, media_list_key_t {},
                                                   executor.queue_depth(), index.traversal_memory_usage(sizing), 0, 0);
        if (failed(call_status)) return call_status;
        traversals_t traversals {scratch_allocator_t {executor.arena(0)}};
        if (status_t const ready = traversals_(index, executor, traversals); failed(ready)) return ready;
        status_t refusal = status_t::success_k;
        auto refuse = [&](status_t status) noexcept { keep_first_failure<atomic_ref>(refusal, status); };
        {
            auto [load, load_status] = index.plan_load(nodes);
            if (failed(load_status)) return load_status;
            auto const tasks = load.tasks();
            iota_view<node_id_t> const source_range {0, static_cast<node_id_t>(nodes)};
            if (status_t const fanned = executor.for_tasks(
                    tasks.size(),
                    [&](std::size_t position, std::size_t, std::size_t) noexcept {
                        refuse(load.load(tasks[position], columns, source_range,
                                         [](node_id_t row) noexcept { return row; }));
                    });
                failed(fanned))
                return fanned;
            if (failed(refusal)) return refusal;
            if (status_t const loaded = load.finish(); failed(loaded)) return loaded;
        }
        typename ram_index_t::refine_config_t config {static_cast<u32_t>(expansion_factor_())};
        config.cache_bytes = cache_bytes_;
        auto [refining, refine_status] = index.plan_refine(config);
        if (failed(refine_status)) return refine_status;
        part_rounds_ = 0;
        do {
            if (part_rounds_) refining.next_round();
            for (traversal_context_t& traversal : traversals)
                if (status_t const sized = refining.reserve(traversal); failed(sized)) return sized;
            do {
                auto const tasks = refining.tasks();
                if (status_t const fanned = executor.for_tasks(
                        tasks.size(),
                        [&](std::size_t position, std::size_t thread, std::size_t) noexcept {
                            refuse(refining.run(tasks[position], traversals[thread]));
                        });
                    failed(fanned))
                    return fanned;
                if (failed(refusal)) return refusal;
            } while (refining.advance());
            auto const [refined, finish_status] = refining.finish();
            if (failed(finish_status)) return finish_status;
            close_round_(refined);
        } while (another_round_());
        return write_ram_rows_(
            index, target, 0, nodes, slices.staging, staged, executor,
            [](std::size_t row) noexcept { return static_cast<node_id_t>(row); },
            [](node_id_t node) noexcept { return node; });
    }

#pragma endregion RAM Blocks

#pragma region Rounds

    /** The arena a share is carved from, null to only sum its bytes; a refused carve nulls it. */
    struct share_carver_t {
        arena_t* arena = nullptr;
        std::size_t bytes = 0;
    };

    /** Takes @p count elements on the default alignment off the arena of @p carver, counting the
     *  bytes they reserve; null when the arena is absent or refuses. */
    template <typename value_type_>
    static value_type_* carve_share_(share_carver_t& carver, std::size_t count) noexcept {
        std::size_t const bytes = max_of<std::size_t>(count * sizeof(value_type_), 1);
        carver.bytes += arena_allocator<byte_t, forkunion_allocator_t>::reserved_bytes(bytes);
        if (!carver.arena) return nullptr;
        value_type_* const carved = reinterpret_cast<value_type_*>(carver.arena->allocate(bytes, default_alignment_k));
        if (!carved) carver.arena = nullptr;
        return carved;
    }

    /** A gather's share of a thread scoring reads of up to @p run_rows rows. */
    gather_share_t gather_share_(share_carver_t& carver, std::size_t run_rows) const noexcept {
        std::size_t const step_rows = scan_scorer_t::step_rows_k, dots_columns = scan_scorer_t::dots_columns_k;
        std::size_t const hubs = hubs_per_draw_;
        return {carve_share_<scored_node_t>(carver, run_rows),
                carve_share_<vector_norms_t>(carver, run_rows),
                {carve_share_<distance_t>(carver, hubs), hubs},
                {carve_share_<raw_dot_t>(carver, step_rows * dots_columns), step_rows, dots_columns},
                {carve_share_<f32_t>(carver, step_rows * hubs), step_rows, hubs}};
    }

    /** A restitch's share of a thread scoring neighborhoods of up to @p width entries. */
    restitch_share_t restitch_share_(share_carver_t& carver, std::size_t width) const noexcept {
        std::size_t const step_rows = scan_scorer_t::step_rows_k, dots_columns = scan_scorer_t::dots_columns_k;
        std::size_t const columns = block_columns_(width), connectivity = connectivity_();
        return {{carve_share_<distance_t>(carver, columns), columns},
                {carve_share_<raw_dot_t>(carver, step_rows * dots_columns), step_rows, dots_columns},
                {carve_share_<f32_t>(carver, step_rows * columns), step_rows, columns},
                carve_share_<scored_node_t>(carver, connectivity),
                carve_share_<scored_node_t>(carver, 2 * connectivity),
                carve_share_<scored_node_t>(carver, connectivity),
                carve_share_<u32_t>(carver, connectivity),
                carve_share_<vector_norms_t>(carver, connectivity)};
    }

    /** One share per thread of @p executor off its own arena, the shares listed in the first,
     *  inside a call begun for @ref shares_bytes_; null when an arena refuses. */
    template <typename share_type_, typename share_of_type_>
    static share_type_* carve_shares_(executor_t& executor, share_of_type_&& share_of) noexcept {
        std::size_t const threads = executor.size();
        auto* const shares = reinterpret_cast<share_type_*>(
            executor.arena(0).allocate(threads * sizeof(share_type_), default_alignment_k));
        if (!shares) return nullptr;
        for (std::size_t thread = 0; thread != threads; ++thread) {
            share_carver_t carver {&executor.arena(thread)};
            new (shares + thread) share_type_ {share_of(carver)};
            if (!carver.arena) return nullptr;
        }
        return shares;
    }

    /** Bytes each arena of @p executor sets aside for one share per thread from @p share_of. */
    template <typename share_type_, typename share_of_type_>
    static std::size_t shares_bytes_(executor_t& executor, share_of_type_&& share_of) noexcept {
        share_carver_t sizing;
        static_cast<void>(share_of(sizing));
        return sizing.bytes + executor.size() * sizeof(share_type_) + default_alignment_k;
    }

    /** Reads the neighbors row of @p node of @p target into @p row. */
    status_t read_row_(target_t& target, std::size_t node, node_id_t* row) noexcept {
        std::size_t const row_bytes = neighbors_row_bytes_();
        return target.file.read_at(part_of_(target).neighbors_begin + node * row_bytes,
                                   bytes_span_t {reinterpret_cast<byte_t*>(row), row_bytes});
    }

    /** Sizes and places the round of target part @p target at the workspace front: as many hubs a
     *  draw as one cache-sized pack holds and the workspace past the columns stages, a neighborhood
     *  capacity both the core's cache and that workspace hold, tiles of rows the workspace rewrites
     *  at once, and the restitch tape over them. */
    status_t open_round_columns_(target_t const& target) noexcept {
        std::size_t const rows = target.view.size(), row_bytes = value_row_bytes_();
        std::size_t hubs_per_draw = scan_scorer_t::block_columns(rows, cache_bytes_, row_bytes);
        carver_t fixed;
        round_slices_(rows, hubs_per_draw, 0, fixed);
        if (fixed.bytes >= workspace_.size()) return status_t::bad_alloc_k;
        hubs_per_draw = fitting_rows_(&nvme_reshard_plan::hub_slices_, hubs_per_draw, workspace_.size() - fixed.bytes);
        fixed = {};
        round_slices_(rows, hubs_per_draw, 0, fixed);
        std::size_t const widest_tile = fitting_rows_(&nvme_reshard_plan::write_slices_, rows,
                                                      workspace_.size() - min_of(fixed.bytes, workspace_.size()));
        if (!hubs_per_draw || !widest_tile) return status_t::bad_alloc_k;
        std::size_t const tiles = divide_round_up(rows, widest_tile);
        workspace_arena_t arena = arena_over_(workspace_);
        carver_t carver {&arena};
        round_slices_t const slices = round_slices_(rows, hubs_per_draw, tiles, carver);
        if (carver.bytes > workspace_.size()) return status_t::bad_alloc_k;
        round_spare_ = workspace_.subspan(carver.bytes);
        hubs_per_draw_ = hubs_per_draw;
        tile_rows_ = divide_round_up(rows, tiles);
        carver_t tile;
        write_slices_(tile_rows_, tile);
        // A neighborhood shares the core's cache with one pack of hubs, as the RAM round sizes it.
        std::size_t const pack_bytes = scan_scorer_t::block_columns(hubs_per_draw, cache_bytes_, row_bytes) * row_bytes;
        std::size_t const cached = cache_bytes_ ? max_of((cache_bytes_ - min_of(cache_bytes_, pack_bytes)) / row_bytes,
                                                         scan_scorer_t::step_rows_k)
                                                : rows;
        neighborhood_capacity_ = fitting_rows_(&nvme_reshard_plan::restitch_slices_, min_of(cached, rows),
                                               round_spare_.size());
        if (tile.bytes > round_spare_.size() || neighborhood_capacity_ < min_of<std::size_t>(rows, 2))
            return status_t::bad_alloc_k;
        audit_ = {slices.depth_columns, slices.depth_columns + rows, slices.depth_columns + 2 * rows,
                  slices.depth_columns + 3 * rows, slices.reach_words};
        entry_neighborhoods_ = {slices.entry_neighborhoods, rows};
        neighborhood_entries_ = {slices.neighborhood_entries, rows};
        neighborhood_offsets_ = {slices.neighborhood_offsets, split_room(rows, hubs_per_draw)};
        tape_runs_ = {slices.tape_runs, tiles};
        for (std::size_t& run : tape_runs_) run = 0;
        return open_tape_(target, tiles * tile_rows_ * tape_row_bytes_());
    }

    /** Opens the restitch tape of @p bytes beside target part @p target, truncating any left. */
    status_t open_tape_(target_t const& target, std::size_t bytes) noexcept {
        close_tape_();
        std::string_view const part = path_of_(target.path), suffix = ".tape";
        if (failed(tape_path_.resize_uninitialized(part.size() + suffix.size() + 1))) return status_t::bad_alloc_k;
        std::memcpy(tape_path_.data(), part.data(), part.size());
        std::memcpy(tape_path_.data() + part.size(), suffix.data(), suffix.size());
        tape_path_[part.size() + suffix.size()] = '\0';
        auto [file, open_status] = seekable_file_t::open_terminated(tape_path_.data(),
                                                                    file_access_t::truncate_read_write_k);
        if (failed(open_status)) return open_status;
        tape_ = std::move(file);
        return tape_.truncate(bytes);
    }

    /** Closes and removes the restitch tape, if one is open. */
    void close_tape_() noexcept {
        if (tape_path_.empty()) return;
        [[maybe_unused]] status_t const closed = tape_.close();
        std::remove(tape_path_.data());
        tape_path_.clear();
    }

    /** Audits target part @p task.part from its root. The opening check places the round's columns
     *  and its tape, seeds the round's draws from what it measured, and heals the islands it names
     *  in the same call, whose first arena holds them; the closing check concludes the round and
     *  folds it into the result. */
    status_t health_(task_t const& task, executor_t& executor) noexcept {
        target_t& target = targets_[task.part];
        std::size_t const nodes = target.view.size(), connectivity = connectivity_(),
                          row_bytes = neighbors_row_bytes_();
        bool const opening = health_after_ != phase_t::write_k;
        if (opening) {
            if (status_t const opened = open_round_columns_(target); failed(opened)) return opened;
        }
        else close_tape_();
        auto [call, call_status] = target.view.begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        islands_t islands {typename islands_t::scratch_allocator_t {executor.arena(0)}};
        u64_t const neighbors_begin = part_of_(target).neighbors_begin;
        auto neighbors_extent = [&](std::size_t node) noexcept {
            return nvme_extent_t {neighbors_begin + node * row_bytes, static_cast<u32_t>(row_bytes), 0};
        };
        auto const [audited, audit_status] = nvme_audit_reachability<node_id_t, atomic_ref>(
            executor, to_span(target.view.media_), target.view.staging_bytes_(), audit_, islands, nodes, connectivity,
            target.view.scan_run_rows_(0, row_bytes), neighbors_extent);
        if (failed(audit_status)) return audit_status;
        if (!opening) {
            round_.reachability = audited;
            // The restitch gave the islands the heal could not link their links, which a heal from
            // this check follows without a distance, while each heal leaves fewer waiting.
            if (heals_again(audited.audited_nodes - audited.reachable_nodes, islands_waiting_, waiting_before_)) {
                for (std::size_t& run : tape_runs_) run = 0;
                if (status_t const reopened = open_tape_(target, tape_runs_.size() * tile_rows_ * tape_row_bytes_());
                    failed(reopened))
                    return reopened;
                health_after_ = phase_t::health_k;
                heal_from_(audited);
                return heal_(target, islands);
            }
            round_.verdict = concluded_verdict(round_);
            close_round_(round_);
            return status_t::success_k;
        }
        round_ = {};
        round_.opening = audited;
        round_.islands_found = audited.audited_nodes - audited.reachable_nodes;
        islands_waiting_ = (std::numeric_limits<std::size_t>::max)();
        heal_from_(audited);
        return heal_(target, islands);
    }

    /** Seeds the draws from the health check @p measured and starts counting anew the islands its
     *  heal leaves waiting. */
    void heal_from_(reachability_t const& measured) noexcept {
        round_seed_ = round_seed(measured);
        waiting_before_ = std::exchange(islands_waiting_, 0);
    }

    /** Rounds a part runs at most while each improves when the plan stitches moved entries in, so
     *  a graph that never settles still ends; a refine, which moves none, runs one. */
    inline static constexpr std::size_t stitching_rounds_limit_k = 8;

    /** Whether the part in hand runs another round: the plan built blocks or moved entries, its
     *  last round improved, and the limit is not reached. */
    bool another_round_() const noexcept {
        bool const stitching = !opened_.empty() || result_.moved_entries;
        return stitching && part_.verdict == refine_verdict_t::improved_k && part_rounds_ < stitching_rounds_limit_k;
    }

    /** Takes in one round of the part in hand: its counts add to the part's, which keeps its first
     *  opening check and its islands, and takes this round's closing check and verdict. */
    void close_round_(refine_report_t const& round) noexcept {
        if (!part_rounds_++) {
            part_ = round;
            return;
        }
        refine_report_t part = part_;
        part += round;
        part.opening = part_.opening;
        part.islands_found = part_.islands_found;
        part.reachability = round.reachability;
        part.verdict = round.verdict;
        part_ = part;
    }

    /** Folds the part in hand's rounds into the result, the first part setting the verdict the
     *  others rank against, and opens the next part's round. */
    void close_part_() noexcept {
        if (!folded_parts_++) result_.graph.verdict = part_.verdict;
        result_.graph += part_;
        open_round_(round_part_ + 1);
    }

    /** Links every unreached or stranded island the health check named through the root's
     *  component with no distance, as the RAM round heals: an unreached one's entry from the first
     *  of its own links inside the component that takes it, a stranded one's entry to the entry of
     *  the component linking to it, each row read, changed and written alone. A component with
     *  neither counts as waiting for the links a restitch gives it. */
    status_t heal_(target_t& target, islands_t const& islands) noexcept {
        std::size_t const nodes = target.view.size(), connectivity = connectivity_();
        workspace_arena_t arena = arena_over_(round_spare_);
        carver_t carver {&arena};
        node_id_t* const links = carve_<node_id_t>(carver, connectivity);
        node_id_t* const row = carve_<node_id_t>(carver, connectivity);
        if (carver.bytes > round_spare_.size()) return status_t::bad_alloc_k;
        for (std::size_t rank = 0; rank != islands.islands(); ++rank) {
            if (!islands.island_first(rank)) continue;
            u8_t const roles = islands.island_roles(rank);
            node_id_t const entry = islands.island_entry(rank);
            if (roles & islands_t::unreached_island_k) {
                if (status_t const read = read_row_(target, entry, links); failed(read)) return read;
                // A full owner row may refuse, so the next link inside the component offers its
                // own; a row led by its own entry is a forgotten one, as the health check reads it.
                bool linked = false, owned = false;
                for (std::size_t slot = 0; slot != connectivity && links[0] != entry && !linked; ++slot) {
                    node_id_t const owner = links[slot];
                    if (owner >= nodes || owner == entry || !in_component_(owner)) continue;
                    owned = true;
                    auto const [written, status] = link_island_(target, owner, entry, row);
                    if (failed(status)) return status;
                    linked = written;
                }
                islands_waiting_ += !owned;
            }
            node_id_t const parent = islands.island_entry_parent(rank);
            bool const stranded = (roles & islands_t::stranded_island_k) != 0;
            if (stranded && parent != node_id_unknown_k)
                if (status_t const status = link_island_(target, entry, parent, row).status; failed(status))
                    return status;
            islands_waiting_ += stranded && parent == node_id_unknown_k;
        }
        return status_t::success_k;
    }

    /** Links @p member from @p owner 's row, as the RAM heal does: a free slot first; else, inside
     *  the root's component, the slot whose resident is linked to by the most entries, when it
     *  stays linked to by more than @p member and the reach rule releases it. The owner's row
     *  stages in @p row, and a written link counts the member's in-link and files its depths. */
    expected<bool> link_island_(target_t& target, node_id_t owner, node_id_t member, node_id_t* row) noexcept {
        std::size_t const nodes = target.view.size(), connectivity = connectivity_();
        if (owner == member || owner >= nodes || member >= nodes) return {false, status_t::success_k};
        if (status_t const read = read_row_(target, owner, row); failed(read)) return {false, read};
        if (row[0] == owner)
            for (std::size_t slot = 0; slot != connectivity; ++slot) row[slot] = sentinel_k;
        std::size_t free_slot = connectivity;
        for (std::size_t slot = 0; slot != connectivity; ++slot) {
            if (row[slot] == member) return {false, status_t::success_k};
            if (free_slot == connectivity && row[slot] >= nodes) free_slot = slot;
        }
        if (free_slot == connectivity && in_component_(owner)) {
            std::size_t richest = connectivity;
            unsigned richest_links = in_links_(member);
            for (std::size_t slot = 0; slot != connectivity; ++slot) {
                if (row[slot] >= nodes || in_links_(row[slot]) <= richest_links + 1u) continue;
                richest = slot, richest_links = in_links_(row[slot]);
            }
            if (richest == connectivity || !take_link_under_lock_if_spare_(owner, row[richest]))
                return {false, status_t::success_k};
            static_cast<void>(take_in_link_if_spare_(row[richest]));
            free_slot = richest;
        }
        if (free_slot == connectivity) return {false, status_t::success_k};
        row[free_slot] = member;
        if (status_t const wrote = write_rows_(target, part_of_(target).neighbors_begin, neighbors_row_bytes_(), owner,
                                               1, reinterpret_cast<byte_t const*>(row));
            failed(wrote))
            return {false, wrote};
        count_in_link_(member);
        record_heal_link_(owner, member);
        round_.changed_nodes += !count_changed_(target, owner);
        ++round_.islands_reconnected;
        return {true, status_t::success_k};
    }

    /** The round's pack of hubs at the front of the workspace past its columns, placed alike by
     *  every task that reads it. */
    hub_slices_t hubs_() const noexcept {
        workspace_arena_t arena = arena_over_(round_spare_);
        carver_t carver {&arena};
        return hub_slices_(hubs_per_draw_, carver);
    }

    /** Derives the norms of the first @p count staged hubs and packs them. */
    void pack_hubs_(hub_slices_t const& hubs, std::size_t count) const noexcept {
        for (std::size_t rank = 0; rank != count; ++rank)
            hubs.norms[rank] = norms_of_(hubs.rows + rank * value_row_bytes_());
        scan_scorer_t::pack(rows_view_(hubs.rows, count), hubs.packs);
    }

    /** Draws the round's hubs by seeded bijection over target part @p task.part, ascending, reads
     *  their value rows and packs them. */
    status_t draw_hubs_(task_t const& task, executor_t& executor) noexcept {
        target_t& target = targets_[task.part];
        std::size_t const rows = target.view.size(), row_bytes = value_row_bytes_();
        std::size_t const hubs = round_first_hubs(rows, hubs_per_draw_, neighborhood_capacity_);
        hub_slices_t const slices = hubs_();
        for (std::size_t rank = 0; rank != hubs; ++rank)
            slices.nodes[rank] = static_cast<node_id_t>(seeded_bijection(rank, rows, round_seed_));
        partial_sort(std::span<node_id_t>(slices.nodes, hubs), hubs);
        auto [call, call_status] = target.view.begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        if (status_t const read = copy_points_(
                target.view, part_of_(target).values_begin, row_bytes, hubs,
                [&](std::size_t rank) noexcept { return slices.nodes[rank]; },
                [&](std::size_t rank) noexcept { return slices.rows + rank * row_bytes; }, executor);
            failed(read))
            return read;
        pack_hubs_(slices, hubs);
        return status_t::success_k;
    }

    /** Files each of the @p count rows staged at @p bytes under the nearest of the first
     *  @p hubs_count packed @p hubs, the first among equals, as the neighborhood its node takes,
     *  which @p node_of answers. */
    template <typename node_of_type_>
    void file_under_nearest_(gather_share_t const& share, hub_slices_t const& hubs, std::size_t hubs_count,
                             byte_t const* bytes, std::size_t count, node_of_type_ const& node_of) noexcept {
        std::span<vector_norms_t> const row_norms {share.row_norms, count};
        for (std::size_t row = 0; row != count; ++row) {
            row_norms[row] = norms_of_(bytes + row * value_row_bytes_());
            share.nearest[row] = farthest_;
        }
        std::span<distance_t> const thresholds = share.thresholds.first(hubs_count);
        for (distance_t& threshold : thresholds) threshold = farthest_.distance;
        // Every hub's distance is heard, so the threshold never moves.
        auto keep = [&](std::size_t hub, scored_node_t entry) noexcept {
            scored_node_t& nearest = share.nearest[entry.node_id];
            if (entry.distance < nearest.distance) nearest = {static_cast<node_id_t>(hub), entry.distance};
            return farthest_.distance;
        };
        scan_scorer_t const scorer {kernel_(), rows_view_(hubs.rows, hubs_count),
                                    std::span<vector_norms_t const>(hubs.norms, hubs_count), hubs.packs, share.dots};
        dummy_predicate_t admits;
        scan_block(scorer, rows_view_(bytes, count), 0, std::span<vector_norms_t const>(row_norms), thresholds,
                   share.step_distances, admits, keep);
        for (std::size_t row = 0; row != count; ++row) {
            node_id_t const hub = share.nearest[row].node_id;
            entry_neighborhoods_[node_of(row)] = static_cast<u16_t>(hub == node_id_unknown_k ? 0 : hub);
        }
    }

    /** Files every entry of target part @p task.part under its nearest hub: the first gather
     *  streams the value column once past the round's pack, each read scored where it landed; a
     *  later one takes one neighborhood over capacity, draws hubs among its own entries as the RAM
     *  round draws them, and scores each entry's row read by point. */
    status_t gather_(task_t const& task, executor_t& executor) noexcept {
        target_t& target = targets_[task.part];
        nvme_part_t const& part = part_of_(target);
        std::size_t const rows = target.view.size(), row_bytes = value_row_bytes_();
        std::span<node_id_t const> const members {neighborhood_entries_.data() + task.nodes.first, task.nodes.size()};
        if (neighborhoods_ && members.size() <= neighborhood_capacity_) return status_t::success_k;
        std::size_t const entries = neighborhoods_ ? members.size() : rows;
        std::size_t const hubs = neighborhoods_ ? round_hubs(entries, hubs_per_draw_, neighborhood_capacity_)
                                                : round_first_hubs(entries, hubs_per_draw_, neighborhood_capacity_);
        std::size_t const run_rows = neighborhoods_ ? 1 : target.view.scan_run_rows_(0, row_bytes);
        auto share_of = [&](share_carver_t& carver) noexcept { return gather_share_(carver, run_rows); };
        auto [call, call_status] = target.view.begin_scans_(executor, 1,
                                                            shares_bytes_<gather_share_t>(executor, share_of));
        if (failed(call_status)) return call_status;
        gather_share_t* const shares = carve_shares_<gather_share_t>(executor, share_of);
        if (!shares) return status_t::bad_alloc_k;
        hub_slices_t const slices = hubs_();
        round_.traversal_stats.computed_distances += entries * hubs;
        if (!neighborhoods_) {
            auto describe = [&](std::size_t run, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
                if (progress++) return {};
                std::size_t const first = run * run_rows;
                return {nvme_extent_t {part.values_begin + u64_t(first) * row_bytes,
                                       static_cast<u32_t>(min_of(run_rows, rows - first) * row_bytes), 0}};
            };
            auto consume = [&](std::size_t thread, std::size_t run, u64_t, auto const& extents,
                               auto const& fetched) noexcept {
                file_under_nearest_(shares[thread], slices, hubs, fetched[0], extents[0].length / row_bytes,
                                    [&](std::size_t row) noexcept { return run * run_rows + row; });
            };
            return target.view.scan_reads_(executor, divide_round_up(rows, run_rows), describe, consume).status;
        }
        u64_t const seed = mix_bits(round_seed_ ^ task.nodes.first);
        if (status_t const read = copy_points_(
                target.view, part.values_begin, row_bytes, hubs,
                [&](std::size_t rank) noexcept { return members[seeded_bijection(rank, members.size(), seed)]; },
                [&](std::size_t rank) noexcept { return slices.rows + rank * row_bytes; }, executor);
            failed(read))
            return read;
        pack_hubs_(slices, hubs);
        return read_points_(
            target.view, part.values_begin, row_bytes, members.size(),
            [&](std::size_t position) noexcept { return members[position]; },
            [&](std::size_t thread, std::size_t position, byte_t const* row) noexcept {
                file_under_nearest_(shares[thread], slices, hubs, row, 1,
                                    [&](std::size_t) noexcept { return members[position]; });
            },
            executor);
    }

    /** Sorts the round's entries by neighborhood after a gather, as @ref order_by_neighborhood does
     *  after the draw and @ref split_wide_neighborhoods after a gather over the neighborhoods over
     *  capacity, and answers the widest one's entries. */
    std::size_t order_neighborhoods_() noexcept {
        std::size_t const rows = targets_[round_part_].view.size();
        std::span<u16_t const> const tags {entry_neighborhoods_.data(), rows};
        neighborhood_order_t const order =
            neighborhoods_ ? split_wide_neighborhoods(tags, neighborhood_entries_, neighborhood_offsets_,
                                                      neighborhoods_, neighborhood_capacity_, hubs_per_draw_)
                           : order_by_neighborhood(tags, round_first_hubs(rows, hubs_per_draw_, neighborhood_capacity_),
                                                   neighborhood_entries_, neighborhood_offsets_);
        neighborhoods_ = order.neighborhoods;
        round_.neighborhoods = order.neighborhoods;
        round_.widest_neighborhood = order.widest;
        return order.widest;
    }

    /** Re-stitches the neighborhood at positions @p task.nodes of the order by neighborhood: reads
     *  its entries' value rows and links by points and the value rows its links lead out to, scores
     *  it against itself in cache, and re-selects every entry's links among its own and its nearest
     *  neighborhood-mates, keeping only the diverse ones and those @ref keep_reach_ needs. The
     *  changed rows go onto the restitch tape, one write per tile they fall in, each entry's row
     *  written by this task alone. */
    status_t restitch_(task_t const& task, executor_t& executor) noexcept {
        target_t& target = targets_[task.part];
        nvme_part_t const& part = part_of_(target);
        std::span<node_id_t const> const members {neighborhood_entries_.data() + task.nodes.first, task.nodes.size()};
        std::size_t const width = members.size(), nodes = target.view.size(), connectivity = connectivity_();
        std::size_t const row_bytes = value_row_bytes_(), tape_values = 1 + connectivity;
        workspace_arena_t arena = arena_over_(round_spare_);
        carver_t carver {&arena};
        restitch_slices_t const slices = restitch_slices_(width, carver);
        if (carver.bytes > round_spare_.size()) return status_t::bad_alloc_k;
        auto share_of = [&](share_carver_t& carver) noexcept { return restitch_share_(carver, width); };
        auto [call, call_status] = target.view.begin_scans_(executor, 1,
                                                            shares_bytes_<restitch_share_t>(executor, share_of));
        if (failed(call_status)) return call_status;
        restitch_share_t* const shares = carve_shares_<restitch_share_t>(executor, share_of);
        if (!shares) return status_t::bad_alloc_k;

        byte_t* const rows = slices.rows;
        node_id_t* const links = slices.links;
        if (status_t const read = copy_points_(
                target.view, part.values_begin, row_bytes, width,
                [&](std::size_t position) noexcept { return members[position]; },
                [&](std::size_t position) noexcept { return rows + position * row_bytes; }, executor);
            failed(read))
            return read;
        if (status_t const read = copy_points_(
                target.view, part.neighbors_begin, neighbors_row_bytes_(), width,
                [&](std::size_t position) noexcept { return members[position]; },
                [&](std::size_t position) noexcept {
                    return reinterpret_cast<byte_t*>(links + position * connectivity);
                },
                executor);
            failed(read))
            return read;

        // A link's row is staged at its entry's position inside the neighborhood, or past the
        // entries in the order of the slots leading out.
        auto position_of = [&](node_id_t node) noexcept {
            return static_cast<std::size_t>(lower_bound(members, node) - members.data());
        };
        std::size_t outside = 0;
        for (std::size_t slot = 0; slot != width * connectivity; ++slot) {
            node_id_t const link = links[slot];
            if (link >= nodes || link == members[slot / connectivity]) continue;
            std::size_t const position = position_of(link);
            if (position == width || members[position] != link)
                slices.outside_slots[outside++] = static_cast<u32_t>(slot);
        }
        std::span<u32_t const> const outside_slots {slices.outside_slots, outside};
        auto staged_of = [&](std::size_t slot) noexcept {
            std::size_t const position = position_of(links[slot]);
            if (position != width && members[position] == links[slot]) return position;
            return width + static_cast<std::size_t>(lower_bound(outside_slots, static_cast<u32_t>(slot)) -
                                                    outside_slots.data());
        };
        auto node_of = [&](std::size_t staged) noexcept {
            return staged < width ? members[staged] : links[outside_slots[staged - width]];
        };
        if (status_t const read = copy_points_(
                target.view, part.values_begin, row_bytes, outside,
                [&](std::size_t rank) noexcept { return links[outside_slots[rank]]; },
                [&](std::size_t rank) noexcept { return rows + (width + rank) * row_bytes; }, executor);
            failed(read))
            return read;

        std::span<vector_norms_t> const norms {slices.norms, width};
        for (std::size_t position = 0; position != width; ++position)
            norms[position] = norms_of_(rows + position * row_bytes);
        vectors_view_t const neighborhood = rows_view_(rows, width);
        std::size_t const block_columns = block_columns_(width), blocks = divide_round_up(width, block_columns);
        std::size_t const pack_stride = scan_scorer_t::packs_bytes(block_columns, dimensions_());
        for (std::size_t block = 0; block != blocks; ++block) {
            std::size_t const first = block * block_columns;
            scan_scorer_t::pack(neighborhood(nk::range(first, min_of(first + block_columns, width)), nk::slice),
                                slices.packs + block * pack_stride);
        }

        // Every entry's own heap, so the threads scoring other steps never touch it.
        std::span<scored_node_t> const mates {slices.mates, width * connectivity};
        for (scored_node_t& mate : mates) mate = farthest_;
        auto mates_of = [&](std::size_t position) noexcept {
            return mates.subspan(position * connectivity, connectivity);
        };
        std::size_t const step_rows = scan_scorer_t::step_rows_k;
        auto score_step = [&](std::size_t step, std::size_t thread, std::size_t) noexcept {
            restitch_share_t const& share = shares[thread];
            std::size_t const first = step * step_rows, count = min_of(step_rows, width - first);
            dummy_predicate_t admits;
            for (std::size_t block = 0; block != blocks; ++block) {
                std::size_t const first_column = block * block_columns;
                std::size_t const columns = min_of(block_columns, width - first_column);
                std::span<distance_t> const thresholds = share.thresholds.first(columns);
                for (distance_t& threshold : thresholds) threshold = farthest_.distance;
                auto keep = [&](std::size_t column, scored_node_t row) noexcept {
                    std::size_t const mate = first_column + column;
                    if (mate == row.node_id) return farthest_.distance;
                    std::span<scored_node_t> const heap = mates_of(row.node_id);
                    scored_node_t const offered {static_cast<node_id_t>(mate), row.distance};
                    if (further_(heap[0], offered)) replace_farthest(heap, offered, further_);
                    return farthest_.distance;
                };
                scan_scorer_t const scorer {kernel_(),
                                            neighborhood(nk::range(first_column, first_column + columns), nk::slice),
                                            std::span<vector_norms_t const>(norms).subspan(first_column, columns),
                                            slices.packs + block * pack_stride, share.dots};
                scan_block(scorer, neighborhood(nk::range(first, first + count), nk::slice),
                           static_cast<node_id_t>(first), std::span<vector_norms_t const>(norms).subspan(first, count),
                           thresholds, share.step_distances, admits, keep);
            }
        };
        if (status_t const scored = executor.for_tasks(divide_round_up(width, step_rows), score_step); failed(scored))
            return scored;

        std::size_t added = 0, dropped = 0, changed = 0, computed = 0;
        auto select = [&](std::size_t position, std::size_t thread, std::size_t) noexcept {
            restitch_share_t const& share = shares[thread];
            node_id_t const node = members[position];
            byte_t const* const row = rows + position * row_bytes;
            std::size_t live = 0, candidates = 0, local_computed = 0;
            for (std::size_t slot = position * connectivity; slot != (position + 1) * connectivity; ++slot) {
                if (links[slot] >= nodes || links[slot] == node) continue;
                std::size_t const staged = staged_of(slot);
                byte_t const* const link_row = rows + staged * row_bytes;
                distance_t const distance = distance_(row, norms[position], link_row,
                                                      staged < width ? norms[staged] : norms_of_(link_row));
                ++local_computed;
                share.live[live++] = {links[slot], distance};
                share.candidates[candidates++] = {static_cast<node_id_t>(staged), distance};
            }
            for (scored_node_t const mate : mates_of(position)) {
                bool listed = mate.node_id == node_id_unknown_k;
                for (std::size_t rank = 0; rank != live && !listed; ++rank)
                    listed = share.candidates[rank].node_id == mate.node_id;
                if (!listed) share.candidates[candidates++] = mate;
            }
            // Equal distances order on the hashed ring around the entry, so a duplicate class links
            // in a ring, not a star.
            partial_sort(std::span<scored_node_t>(share.candidates, candidates), candidates,
                         [&](scored_node_t const& first, scored_node_t const& second) noexcept {
                             if (first.distance != second.distance) return first.distance < second.distance;
                             return ring_offset(node_of(first.node_id), node) <
                                    ring_offset(node_of(second.node_id), node);
                         });
            // The diverse picks alone: a candidate stays only when no kept one is nearer to it than
            // the entry is; a second duplicate of the entry would only repeat the first's links.
            std::size_t picks = 0;
            bool kept_zero = false;
            for (std::size_t rank = 0; rank != candidates && picks != connectivity; ++rank) {
                scored_node_t const candidate = share.candidates[rank];
                if (kept_zero && candidate.distance == 0) continue;
                byte_t const* const candidate_row = rows + candidate.node_id * row_bytes;
                vector_norms_t const candidate_norms = candidate.node_id < width ? norms[candidate.node_id]
                                                                                 : norms_of_(candidate_row);
                std::size_t kept = 0;
                for (; kept != picks; ++kept) {
                    ++local_computed;
                    if (distance_(candidate_row, candidate_norms, rows + share.kept_rows[kept] * row_bytes,
                                  share.kept_norms[kept]) < candidate.distance)
                        break;
                }
                if (kept != picks) continue;
                kept_zero |= candidate.distance == 0;
                share.selected[picks] = {node_of(candidate.node_id), candidate.distance};
                share.kept_rows[picks] = candidate.node_id;
                share.kept_norms[picks++] = candidate_norms;
            }
            keep_reach_(node, {share.live, live}, share.selected, picks);
            std::size_t kept = 0;
            for (std::size_t pick = 0; pick != picks; ++pick)
                for (std::size_t rank = 0; rank != live; ++rank)
                    kept += share.selected[pick].node_id == share.live[rank].node_id;
            add_(added, picks - kept);
            add_(dropped, live - kept);
            add_(computed, local_computed);
            node_id_t* const taped = slices.tape_rows + position * tape_values;
            taped[0] = sentinel_k;
            if (kept == picks && kept == live) return;
            taped[0] = node;
            for (std::size_t slot = 0; slot != connectivity; ++slot)
                taped[1 + slot] = slot < picks ? share.selected[slot].node_id : sentinel_k;
            add_(changed, !count_changed_(target, node));
        };
        if (status_t const selected = executor.for_tasks(width, select); failed(selected)) return selected;
        round_.reselected_nodes += width;
        round_.changed_nodes += changed;
        round_.added_links += added;
        round_.dropped_links += dropped;
        round_.traversal_stats.computed_distances += width * width + computed;
        round_.traversal_stats.computed_distances_in_refines += computed;

        // The changed rows ascend with their entries, so each tile's come in one run.
        std::size_t const tape_bytes = tape_row_bytes_();
        std::size_t taped_rows = 0;
        for (std::size_t position = 0; position != width; ++position) {
            node_id_t const* const taped = slices.tape_rows + position * tape_values;
            if (taped[0] == sentinel_k) continue;
            if (taped_rows != position) std::memmove(slices.tape_rows + taped_rows * tape_values, taped, tape_bytes);
            ++taped_rows;
        }
        for (std::size_t first = 0; first != taped_rows;) {
            std::size_t const tile = slices.tape_rows[first * tape_values] / tile_rows_;
            std::size_t bound = first + 1;
            while (bound != taped_rows && slices.tape_rows[bound * tape_values] / tile_rows_ == tile) ++bound;
            std::size_t& run = tape_runs_[tile];
            if (run + (bound - first) > tile_rows_) return status_t::overflow_k;
            bytes_view_t const taped {reinterpret_cast<byte_t const*>(slices.tape_rows + first * tape_values),
                                      (bound - first) * tape_bytes};
            if (status_t const wrote = tape_.write_at((tile * tile_rows_ + run) * tape_bytes, taped); failed(wrote))
                return wrote;
            run += bound - first;
            first = bound;
        }
        return status_t::success_k;
    }

    /** Rewrites tile @p task.position of target part @p task.part from its run on the restitch
     *  tape: one read of its neighbors, its run applied, one write. */
    status_t write_(task_t const& task, executor_t& executor) noexcept {
        std::size_t const run = tape_runs_[task.position];
        if (!run) return status_t::success_k;
        target_t& target = targets_[task.part];
        u64_t const neighbors_begin = part_of_(target).neighbors_begin;
        std::size_t const first = task.nodes.first, count = task.nodes.size(), connectivity = connectivity_();
        std::size_t const tape_bytes = tape_row_bytes_();
        workspace_arena_t arena = arena_over_(round_spare_);
        carver_t carver {&arena};
        write_slices_t const slices = write_slices_(count, carver);
        if (carver.bytes > round_spare_.size()) return status_t::bad_alloc_k;
        std::array<staged_column_t, 1> const columns {
            {{neighbors_begin, neighbors_row_bytes_(), reinterpret_cast<byte_t*>(slices.neighbors)}}};
        if (status_t const read = scan_rows_(target.view, 0, first, count, columns, executor); failed(read))
            return read;
        if (status_t const fetched = tape_.read_at(
                task.position * tile_rows_ * tape_bytes,
                bytes_span_t {reinterpret_cast<byte_t*>(slices.tape_rows), run * tape_bytes});
            failed(fetched))
            return fetched;
        for (std::size_t taped = 0; taped != run; ++taped) {
            node_id_t const* const row = slices.tape_rows + taped * (1 + connectivity);
            std::memcpy(slices.neighbors + (row[0] - first) * connectivity, row + 1, neighbors_row_bytes_());
        }
        return write_rows_(target, neighbors_begin, neighbors_row_bytes_(), first, count,
                           reinterpret_cast<byte_t const*>(slices.neighbors));
    }

#pragma endregion Rounds

    /** The host allocator behind the plan's catalogs, which never grow with the rows. */
    forkunion_allocator_t allocator_ {};

    /** The memory every task carves its blocks, batches and tables from. */
    bytes_span_t workspace_;

    /** The caller's configurations, the provenance read until the plan finishes. */
    reshard_config_t config_;
    save_config_t save_;

    /** The custom fields every target header holds. */
    growing_buffer<safegraphs_custom_field_t, forkunion_allocator_t> fields_ {allocator_};

    /** The manifest's path, and the stem its parts are named after. */
    paths_t manifest_ {allocator_};
    paths_t stem_ {allocator_};

    /** The views a build opened itself; the sources point into them. */
    growing_buffer<indexes_t, forkunion_allocator_t> opened_ {allocator_};

    /** Every source part, flattened across the sources. */
    growing_buffer<source_t, forkunion_allocator_t> sources_ {allocator_};

    /** Every target part, in manifest order. */
    growing_buffer<target_t, forkunion_allocator_t> targets_ {allocator_};

    /** One bit per target row, each target part's after every earlier part's, set once the row
     *  counts as changed, at the front of the workspace the caller passed. */
    std::span<u64_t> changed_;

    /** Bytes of cache each core keeps to itself, which size a round's hubs and its neighborhood
     *  capacity; zero until the first task reads the executor's. */
    std::size_t cache_bytes_ = 0;

    /** The target part whose round runs now. */
    std::size_t round_part_ = 0;

    /** The phase the health check in hand follows: write for the closing check, any other for the
     *  opening one, and health for a heal a closing check opened again. */
    phase_t health_after_ = phase_t::build_k;

    /** The round's health check columns, at the workspace front past the changed bits. */
    audit_columns_t audit_;

    /** Per entry, the neighborhood its nearest hub gave it, among the hubs of its last gather. */
    std::span<u16_t> entry_neighborhoods_;

    /** The entries by neighborhood, each one's ascending, and the starts every split may write,
     *  the first @c neighborhoods_ and one past them closing the order. */
    std::span<node_id_t> neighborhood_entries_;
    std::span<node_id_t> neighborhood_offsets_;
    std::size_t neighborhoods_ = 0;

    /** Per tile of rows, the rows its run on the restitch tape holds so far. */
    std::span<std::size_t> tape_runs_;

    /** The workspace past the round's columns, which its tasks stage in. */
    bytes_span_t round_spare_;

    /** Island components the last heal could not link, having no link to or from the root's, and
     *  those the heal before it could not, which the repeats must shrink. */
    std::size_t islands_waiting_ = 0;
    std::size_t waiting_before_ = 0;

    /** Hubs one gather draws at most, the most entries one neighborhood holds, and rows a tile. */
    std::size_t hubs_per_draw_ = 0;
    std::size_t neighborhood_capacity_ = 0;
    std::size_t tile_rows_ = 0;

    /** The seed of the round's draws. */
    u64_t round_seed_ = 0;

    /** The round in hand so far; the part's rounds so far, taken in by @ref close_round_, and how
     *  many; and the parts folded into the result. */
    refine_report_t round_;
    refine_report_t part_;
    std::size_t part_rounds_ = 0;
    std::size_t folded_parts_ = 0;

    /** The restitch tape's path and file, open from a streamed round's opening check to its
     *  closing one. */
    paths_t tape_path_ {allocator_};
    seekable_file_t tape_;

    phase_t phase_ = phase_t::done_k;

    /** Tasks of the phase in hand that ran to success. */
    std::size_t completed_tasks_ = 0;

    result_t result_;

    /** The loaded indexes a reshard moves entries from and commits to; both empty for a rewrite
     *  committing its own manifest. */
    growing_buffer<indexes_t*, forkunion_allocator_t> source_indexes_ {allocator_};
    growing_buffer<indexes_t*, forkunion_allocator_t> target_indexes_ {allocator_};

    /** The workspace a reshard of loaded indexes holds; empty while a caller passes one. */
    growing_buffer<byte_t, forkunion_allocator_t> held_workspace_ {allocator_};

    /** The loaded index whose writer the plan holds: the one target, or the split's source. */
    std::unique_ptr<indexes_t, end_rewrite_t> writing_;
};

template <typename indexes_type_>
expected<nvme_reshard_plan<indexes_type_>> plan_reshard(std::span<indexes_type_* const> sources,
                                                        std::string_view target, bytes_span_t workspace,
                                                        reshard_config_t const& config,
                                                        save_config_t const& save) noexcept {
    nvme_reshard_plan<indexes_type_> plan;
    status_t const opened = plan.open_packed_(sources, target, workspace, config, save);
    return {std::move(plan), opened};
}

template <typename indexes_type_>
expected<nvme_reshard_plan<indexes_type_>> plan_refine(indexes_type_ const& source, iota_view<std::size_t> rows,
                                                       std::string_view target, bytes_span_t workspace,
                                                       reshard_config_t const& config,
                                                       save_config_t const& save) noexcept {
    using plan_t = nvme_reshard_plan<indexes_type_>;
    plan_t plan;
    if (!source.shards_count()) return {std::move(plan), status_t::uninitialized_k};
    if (status_t const opened = plan.open_(target, workspace, config, save, source); failed(opened))
        return {std::move(plan), opened};
    if (status_t const added = plan.add_sources_(source, source); failed(added)) return {std::move(plan), added};
    if (plan.sources_.empty()) return {std::move(plan), status_t::uninitialized_k};
    growing_buffer<std::size_t, forkunion_allocator_t> part_rows {forkunion_allocator_t {}};
    if (failed(part_rows.reserve_exactly(plan.sources_.size()))) return {std::move(plan), status_t::bad_alloc_k};
    for (std::size_t part = 0; part != plan.sources_.size(); ++part) {
        typename plan_t::source_t& one = plan.sources_[part];
        one.target = part;
        one.kept = {0, one.rows};
        part_rows.push_back_reserved(one.rows);
    }
    if (status_t const sized = plan.size_tasks_(); failed(sized)) return {std::move(plan), sized};
    if (status_t const created = plan.open_targets_(to_span(part_rows)); failed(created))
        return {std::move(plan), created};
    // The range runs in placement order across the parts, so each part re-selects its own slice of
    // it.
    for (std::size_t part = 0, placed = 0; part != plan.targets_.size(); placed += plan.targets_[part++].view.size()) {
        typename plan_t::target_t& one = plan.targets_[part];
        std::size_t const low = max_of(rows.first, placed), high = min_of(rows.bound, placed + one.view.size());
        one.selected = low < high ? iota_view<std::size_t> {low - placed, high - placed} : iota_view<std::size_t> {};
    }
    plan.phase_ = phase_t::copy_k;
    return {std::move(plan), status_t::success_k};
}

template <typename indexes_type_>
expected<nvme_reshard_plan<indexes_type_>> plan_build(std::span<std::string_view const> sources,
                                                      std::string_view target, bytes_span_t workspace,
                                                      reshard_config_t const& config,
                                                      save_config_t const& save) noexcept {
    using plan_t = nvme_reshard_plan<indexes_type_>;
    plan_t plan;
    if (sources.empty()) return {std::move(plan), status_t::uninitialized_k};
    if (failed(plan.opened_.reserve_exactly(sources.size()))) return {std::move(plan), status_t::bad_alloc_k};
    for (std::string_view const path : sources) {
        plan.opened_.push_back_reserved(indexes_type_ {plan.allocator_});
        if (status_t const viewed = plan.opened_.back().view(path); failed(viewed)) return {std::move(plan), viewed};
    }
    if (status_t const opened = plan.open_(target, workspace, config, save, plan.opened_[0]); failed(opened))
        return {std::move(plan), opened};
    for (indexes_type_ const& view : plan.opened_)
        if (status_t const added = plan.add_sources_(view, plan.opened_[0]); failed(added))
            return {std::move(plan), added};
    if (plan.sources_.empty()) return {std::move(plan), status_t::uninitialized_k};
    std::size_t const block_rows = plan.build_rows_();
    if (!block_rows) return {std::move(plan), status_t::bad_alloc_k};
    if (status_t const sized = plan.size_tasks_(); failed(sized)) return {std::move(plan), sized};
    growing_buffer<std::size_t, forkunion_allocator_t> target_rows {forkunion_allocator_t {}};
    if (status_t const packed = plan.pack_sources_(config.shard_capacity, target_rows); failed(packed))
        return {std::move(plan), packed};
    if (status_t const created = plan.open_targets_(to_span(target_rows)); failed(created))
        return {std::move(plan), created};
    // Every block links only inside itself, so a part of more than one block runs a round.
    for (typename plan_t::target_t& one : plan.targets_)
        one.selected = {min_of(block_rows, one.view.size()), one.view.size()};
    plan.phase_ = phase_t::copy_k;
    return {std::move(plan), status_t::success_k};
}

/** Runs every task of @p plan over @p executor, @p progress hearing the phase's tasks run and in
 *  all after each, so a cancel stops at the next task. */
template <typename plan_type_, typename progress_type_>
status_t nvme_drive_(plan_type_& plan, typename plan_type_::executor_t& executor, progress_type_& progress) noexcept {
    do
        for (typename plan_type_::task_t const task : plan.tasks()) {
            if (status_t const ran = plan.run(task, executor); failed(ran)) return ran;
            if (progress(plan.completed(), plan.total()) == continuation_t::cancel_k) return status_t::cancelled_k;
        }
    while (plan.advance());
    return status_t::success_k;
}

/** Drains @p planned over @p executor: every task of every phase in order, then the commit. */
template <typename plan_type_>
expected<reshard_report_t> nvme_drain_(expected<plan_type_>&& planned,
                                       typename plan_type_::executor_t& executor) noexcept {
    if (!planned) return {{}, planned.status};
    plan_type_& plan = planned.value;
    dummy_progress_t progress;
    if (status_t const driven = nvme_drive_(plan, executor, progress); failed(driven)) return {{}, driven};
    return plan.finish(executor);
}

/**
 *  @brief The one-shot freestanding @c plan_reshard, drained over @p executor.
 *
 *  @par Statuses:
 *  - @c invalid_argument_k No source, a @c NULL source, an empty @p target, or a target part or
 *    manifest path that does not parse.
 *  - @c uninitialized_k A first source viewing nothing, sources holding no rows, or @p executor
 *    holds no pool or workers.
 *  - @c config_mismatch_k @p save names an alignment below the floor or not a power of two or a
 *    reserved provenance field, a source differs in kernel, geometry or key width, a part rebuilt
 *    in RAM disagrees with its rows, a medium the queues cannot read, or a header does not fit.
 *  - @c bad_alloc_k @p workspace cannot hold one row of some phase.
 *  - @c overflow_k A target part of zero rows or past the node ID space, a part rebuilt in RAM
 *    passing a bound, a staging region's byte count, or a write past a device's end.
 *  - @c malformed_k A framed target header has no width.
 *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
 *  - @c unknown_kind_k A part rebuilt in RAM cannot finalize its kernel.
 */
template <typename indexes_type_>
expected<reshard_report_t> reshard(std::span<indexes_type_* const> sources, std::string_view target,
                                   typename indexes_type_::executor_t& executor, bytes_span_t workspace,
                                   reshard_config_t const& config = {}, save_config_t const& save = {}) noexcept {
    return nvme_drain_(plan_reshard(sources, target, workspace, config, save), executor);
}

/**
 *  @brief The one-shot freestanding @c plan_refine, drained over @p executor.
 *
 *  @par Statuses:
 *  - @c uninitialized_k @p source views nothing or holds no rows, or @p executor holds no pool or
 *    workers.
 *  - @c config_mismatch_k @p save names an alignment below the floor or not a power of two or a
 *    reserved provenance field, a part of @p source differs in geometry or key width, a part
 *    rebuilt in RAM disagrees with its rows, a medium the queues cannot read, or a bad header.
 *  - @c invalid_argument_k An empty @p target, or a target part or manifest path that does not
 *    parse.
 *  - @c bad_alloc_k @p workspace cannot hold one row of some phase.
 *  - @c overflow_k A target part of zero rows or past the node ID space, a part rebuilt in RAM
 *    passing a bound, a staging region's byte count, or a write past a device's end.
 *  - @c malformed_k A framed target header has no width.
 *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
 *  - @c unknown_kind_k A part rebuilt in RAM cannot finalize its kernel.
 */
template <typename indexes_type_>
expected<reshard_report_t> refine(indexes_type_ const& source, iota_view<std::size_t> rows, std::string_view target,
                                  typename indexes_type_::executor_t& executor, bytes_span_t workspace,
                                  reshard_config_t const& config = {}, save_config_t const& save = {}) noexcept {
    return nvme_drain_(plan_refine(source, rows, target, workspace, config, save), executor);
}

/**
 *  @brief The one-shot @c plan_build, drained over @p executor.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No source, sources holding no rows, or @p executor holds no pool or
 *    workers.
 *  - @c bad_alloc_k @p workspace cannot hold one row of some phase.
 *  - @c invalid_argument_k A source, target part or manifest path that does not parse or names a
 *    base off its medium's block, or an empty @p target.
 *  - @c malformed_k A source manifest names no part, a manifest or header does not parse, or a
 *    part or framed target header has no width or too many rows.
 *  - @c config_mismatch_k A source part disagrees with the others or with this instantiation,
 *    @p save names an alignment below the floor or not a power of two or a reserved provenance
 *    field, a part rebuilt in RAM disagrees with its rows, a medium the queues cannot read, or a
 *    part header does not fit before its data.
 *  - @c unsupported_k A transport ID in a build without SPDK.
 *  - @c unknown_kind_k A kernel the stored scalar's kits cannot finalize, or a part rebuilt in RAM
 *    cannot finalize its kernel.
 *  - @c overflow_k A target part of zero rows or past the node ID space, a part rebuilt in RAM
 *    passing a bound, a staging region's byte count, or a write past a device's end.
 *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
 */
template <typename indexes_type_>
expected<reshard_report_t> build(std::span<std::string_view const> sources, std::string_view target,
                                 typename indexes_type_::executor_t& executor, bytes_span_t workspace,
                                 reshard_config_t const& config = {}, save_config_t const& save = {}) noexcept {
    return nvme_drain_(plan_build<indexes_type_>(sources, target, workspace, config, save), executor);
}

/** Names a fresh part beside part @p shard of the loaded @p index into @p buffer: the manifest's
 *  stem with the first generation no file holds yet, then the part's position, as
 *  <tt>index.1-00003-of-00006.safegraphs</tt>, so a replacement never truncates a part that a reader
 *  may still be holding open. @p current_buffer and @p stem_buffer hold the part's own path and
 *  each generation's stem, so a caller that sized all three allocates nothing here. */
template <typename indexes_type_>
expected<std::string_view> nvme_replacing_part_path_(indexes_type_ const& index, std::size_t shard,
                                                     typename indexes_type_::paths_t& buffer,
                                                     typename indexes_type_::paths_t& current_buffer,
                                                     typename indexes_type_::paths_t& stem_buffer) noexcept {
    auto const [current, current_status] = index.part_path(shard, current_buffer);
    if (failed(current_status)) return {{}, current_status};
    std::string_view const directory = current.substr(0, current.size() - safegraphs_basename(current).size());
    std::string_view const stem = safegraphs_basename(safegraphs_stem(index.manifest_path()));
    for (std::size_t generation = 1;; ++generation) {
        auto print = [&](char* sink, std::size_t capacity) noexcept {
            return std::snprintf(sink, capacity, "%.*s%.*s.%zu", static_cast<int>(directory.size()), directory.data(),
                                 static_cast<int>(stem.size()), stem.data(), generation);
        };
        int const length = print(nullptr, 0);
        if (length < 0) return {{}, status_t::io_k};
        if (failed(stem_buffer.resize_uninitialized(static_cast<std::size_t>(length) + 1)))
            return {{}, status_t::bad_alloc_k};
        print(stem_buffer.data(), stem_buffer.size());
        std::string_view const generation_stem {stem_buffer.data(), static_cast<std::size_t>(length)};
        auto const [candidate, candidate_status] = safegraphs_part_path(buffer, generation_stem, shard,
                                                                        index.shards_count());
        if (failed(candidate_status)) return {{}, candidate_status};
        if (candidate == current) continue;
        // The candidate is terminated in place, so the probe opens it without a copy.
        native_file_t const probe = open_native_file(candidate.data(), file_access_t::read_only_k,
                                                     file_buffering_t::buffered_k);
        if (probe == invalid_native_file_k) return {candidate, status_t::success_k};
        [[maybe_unused]] status_t const closed = close_native_file(probe);
    }
}

/** A stem beside the manifest of the loaded @p index that no file uses: the manifest's stem, then
 *  @p label and the first number from one on, into @p buffer. With no @p receivers its manifest and
 *  its parts of every count up to @p parts are free; with some, those of every receiver's stem, the
 *  stem then a dash and the receiver's number from one on. */
template <typename indexes_type_>
expected<std::string_view> nvme_unused_stem_(indexes_type_ const& index, std::string_view label, std::size_t receivers,
                                             std::size_t parts, typename indexes_type_::paths_t& buffer) noexcept {
    using paths_t = typename indexes_type_::paths_t;
    std::string_view const stem = safegraphs_stem(index.manifest_path());
    paths_t named {forkunion_allocator_t {}}, probe {forkunion_allocator_t {}};
    auto used = [&](std::string_view candidate) noexcept {
        if (failed(named.resize_uninitialized(candidate.size() + 12))) return true;
        std::snprintf(named.data(), named.size(), "%.*s.safegraphs", static_cast<int>(candidate.size()),
                      candidate.data());
        if (succeeded(seekable_file_t::open(named.data(), file_access_t::read_only_k).status)) return true;
        for (std::size_t count = 1; count <= parts; ++count)
            for (std::size_t part = 0; part != count; ++part) {
                auto const [path, path_status] = safegraphs_part_path(probe, candidate, part, count);
                if (failed(path_status) || succeeded(seekable_file_t::open(path, file_access_t::read_only_k).status))
                    return true;
            }
        return false;
    };
    paths_t receiver {forkunion_allocator_t {}};
    for (std::size_t number = 1;; ++number) {
        auto print = [&](char* sink, std::size_t capacity) noexcept {
            return std::snprintf(sink, capacity, "%.*s%.*s%zu", static_cast<int>(stem.size()), stem.data(),
                                 static_cast<int>(label.size()), label.data(), number);
        };
        int const length = print(nullptr, 0);
        if (length < 0) return {{}, status_t::io_k};
        if (failed(buffer.resize_uninitialized(static_cast<std::size_t>(length) + 1)))
            return {{}, status_t::bad_alloc_k};
        print(buffer.data(), buffer.size());
        std::string_view const candidate {buffer.data(), static_cast<std::size_t>(length)};
        bool taken = !receivers && used(candidate);
        for (std::size_t position = 0; position != receivers && !taken; ++position) {
            if (failed(receiver.resize_uninitialized(candidate.size() + 24))) return {{}, status_t::bad_alloc_k};
            int const written = std::snprintf(receiver.data(), receiver.size(), "%.*s-%zu",
                                              static_cast<int>(candidate.size()), candidate.data(), position + 1);
            taken = used({receiver.data(), static_cast<std::size_t>(written)});
        }
        if (!taken) return {candidate, status_t::success_k};
    }
}

/**
 *  @brief One refine of a loaded index in flight: every part in turn rewritten by an
 *      @c nvme_reshard_plan into a fresh part beside its own, one round over it, and committed
 *      in its place.
 *
 *  Its tasks and phases are those of the part in hand. The barrier closing a part's round commits
 *  it and opens the next part's rewrite, so one round of the plan is a pass over every part, and
 *  @ref next_round starts another once one ended. It holds the index's writer and its workspace,
 *  sized once for the widest part, while it lives; the part in hand is read in place.
 */
template <typename indexes_type_>
struct [[nodiscard]] nvme_refine_plan {
    using indexes_t = indexes_type_;
    using executor_t = typename indexes_t::executor_t;
    using rewrite_t = nvme_reshard_plan<indexes_t>;
    using task_t = typename rewrite_t::task_t;

    nvme_refine_plan() noexcept = default;
    nvme_refine_plan(nvme_refine_plan&&) noexcept = default;
    nvme_refine_plan& operator=(nvme_refine_plan&&) = delete;
    nvme_refine_plan(nvme_refine_plan const&) = delete;
    nvme_refine_plan& operator=(nvme_refine_plan const&) = delete;

    /** The tasks of the part in hand's phase; none once the pass ended. */
    indexed_range<rewrite_t> tasks() const noexcept { return part_.tasks(); }

    /** Runs @p task of the part in hand, as @ref nvme_reshard_plan::run does. */
    status_t run(task_t const& task, executor_t& executor) noexcept { return part_.run(task, executor); }

    /** The barrier between phases: past a part's last one, commits the part in place of its old one
     *  and opens the next part's rewrite; answers whether a phase with tasks follows. A failed
     *  commit or opening ends the pass, kept for @ref finish. */
    bool advance() noexcept {
        if (part_.advance()) return true;
        if (!index_ || part_.parts() == 0 || failed(status_)) return false;
        if (status_t const committed = commit_part_(); failed(committed)) return status_ = committed, false;
        if (status_t const opened = open_part_(shard_ + 1); failed(opened)) return status_ = opened, false;
        return !part_.tasks().empty();
    }

    /** The phase of the part in hand, done once the pass ended; @c none_k in an empty plan. */
    phase_t phase() const noexcept { return index_ ? part_.phase() : phase_t::none_k; }

    /** Where the part in hand's phase sits among a round's six, and its tasks run and in all. */
    std::size_t ordinal() const noexcept { return part_.ordinal(); }
    std::size_t completed() const noexcept { return part_.completed(); }
    std::size_t total() const noexcept { return part_.total(); }

    /** The parts committed in this pass, folded as @ref finish reports them. */
    refine_report_t running() const noexcept { return report_; }

    /** Starts another pass over every part once one ended, its report starting over; a no-op
     *  mid-pass or after a failure. */
    void next_round() noexcept {
        if (!index_ || part_.phase() != phase_t::done_k || failed(status_)) return;
        report_ = {};
        folded_parts_ = 0;
        status_ = open_part_(0);
    }

    /**
     *  @brief Closes the pass: one health check of the committed parts measures reachability as a
     *      fresh health call would, and the writer goes back; a pass not ended drops its part in
     *      hand, whose old part stays named, and answers @c cancelled_k beside the parts committed.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The plan is empty or finished.
     *  - @c cancelled_k The pass has not ended.
     *  - Any status a commit, the next part's opening or the health check answered.
     */
    expected<refine_report_t> finish(executor_t& executor) noexcept {
        if (!index_) return {{}, status_t::uninitialized_k};
        if (failed(status_) || part_.phase() != phase_t::done_k || part_.parts()) {
            replace_part_(rewrite_t {});
            status_t const ended = failed(status_) ? status_ : status_t::cancelled_k;
            index_.reset();
            return {report_, ended};
        }
        auto const [audit, audit_status] = index_->health(executor);
        report_.reachability = audit;
        index_.reset();
        return {report_, audit_status};
    }

  private:
    using paths_t = typename indexes_t::paths_t;

    template <typename other_type_>
    friend expected<nvme_refine_plan<other_type_>> plan_refine(other_type_&, refine_config_t,
                                                               typename other_type_::executor_t&) noexcept;

    /** Ends the part in hand's rewrite, removing its fresh part unless committed, and takes @p next
     *  in its place. */
    void replace_part_(rewrite_t&& next) noexcept {
        std::destroy_at(&part_);
        std::construct_at(&part_, std::move(next));
    }

    /** Sizes what every part's rewrite reuses, once: the workspace, for the widest part unless
     *  @c memory_limit names its bytes, and the path buffers, past the longest part path. */
    status_t size_parts_() noexcept {
        std::size_t widest = 0, longest = 0;
        for (std::size_t shard = 0; shard != index_->shards_count(); ++shard) {
            widest = max_of<std::size_t>(widest, index_->part_size(shard));
            auto const [path, path_status] = index_->part_path(shard, current_path_);
            if (failed(path_status)) return path_status;
            longest = max_of<std::size_t>(longest, path.size());
        }
        reshard_config_t const config {config_, 0, 0};
        std::size_t const workspace_bytes = config_.memory_limit
                                                ? config_.memory_limit
                                                : rewrite_t::round_in_ram_bytes(*index_, widest, config);
        // A replacement's generation number and the separators it adds stay within this slack.
        std::size_t const path_bytes = longest + 64;
        if (failed(workspace_.resize_uninitialized(workspace_bytes)) ||
            failed(target_path_.reserve_exactly(path_bytes)) || failed(current_path_.reserve_exactly(path_bytes)) ||
            failed(stem_path_.reserve_exactly(path_bytes)))
            return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** Opens the rewrite of the first part from @p shard on that holds rows, read in place from the
     *  index, beside its own; past the last one the pass ends, with no rewrite in hand. */
    status_t open_part_(std::size_t shard) noexcept {
        replace_part_(rewrite_t {});
        while (shard != index_->shards_count() && !index_->part_size(shard)) ++shard;
        shard_ = shard;
        if (shard == index_->shards_count()) return status_t::success_k;
        reshard_config_t const config {config_, 0, 0};
        auto const [target, target_status] = nvme_replacing_part_path_(*index_, shard, target_path_, current_path_,
                                                                       stem_path_);
        if (failed(target_status)) return target_status;
        rewrite_t next;
        if (status_t const opened = next.open_part_refine_(*index_, shard, target, to_writable_span(workspace_),
                                                           config);
            failed(opened))
            return opened;
        replace_part_(std::move(next));
        return status_t::success_k;
    }

    /** Commits the part in hand: seals it, replaces the manifest durably to name it, swaps it in,
     *  and folds its round, the first part's setting the verdict the later ones rank against. */
    status_t commit_part_() noexcept {
        std::size_t const shard = shard_;
        auto const [refined, status] = part_.finish([&](rewrite_t const& sealed) noexcept {
            return index_->replace_part(shard, sealed.part_path(0), sealed.part_entry(0).bytes);
        });
        if (failed(status)) return status;
        if (folded_parts_++) report_ += refined.graph;
        else report_ = refined.graph;
        return index_->relock_manifest();
    }

    /** The index refined, whose writer the plan holds; null in an empty plan and once finished. */
    std::unique_ptr<indexes_t, end_rewrite_t> index_;

    /** The re-selections' knobs, the cache fixed to the executor's at opening. */
    refine_config_t config_ {};

    /** The shard whose part is in hand; the shard count once the pass ended. */
    std::size_t shard_ = 0;

    /** Parts committed this pass. */
    std::size_t folded_parts_ = 0;

    /** The fresh part's path, the part in hand's own and each generation's stem the fresh one is
     *  named after, and the workspace the rewrite carves, all sized once when the plan opens. */
    paths_t target_path_ {forkunion_allocator_t {}};
    paths_t current_path_ {forkunion_allocator_t {}};
    paths_t stem_path_ {forkunion_allocator_t {}};
    growing_buffer<byte_t, forkunion_allocator_t> workspace_ {forkunion_allocator_t {}};

    /** The rewrite of the part in hand. */
    rewrite_t part_;

    /** The parts committed this pass, folded. */
    refine_report_t report_;

    /** The first failure of a commit or an opening, which ended the pass. */
    status_t status_ = status_t::success_k;
};

template <typename indexes_type_>
expected<nvme_refine_plan<indexes_type_>> plan_refine(indexes_type_& index, refine_config_t config,
                                                      typename indexes_type_::executor_t& executor) noexcept {
    nvme_refine_plan<indexes_type_> plan;
    if (!config.cache_bytes) config.cache_bytes = executor.cache_bytes_per_core();
    if (status_t const claimed = index.begin_rewrite(); failed(claimed)) return {std::move(plan), claimed};
    plan.index_.reset(&index);
    plan.config_ = config;
    if (status_t const sized = plan.size_parts_(); failed(sized)) return {std::move(plan), sized};
    status_t const opened = plan.open_part_(0);
    return {std::move(plan), opened};
}

/**
 *  @brief Refines every part of the loaded @p index: the one-shot @ref plan_refine, run to its end
 *      and finished. @p progress hears the phase's tasks run and in all after each, and a cancel
 *      stops at the next task, leaving the parts before it refined.
 *
 *  @par Statuses:
 *  - @c cancelled_k @p progress stopped the pass; the report counts the parts committed.
 *  - Any status @ref plan_refine, its tasks or its finish answers.
 */
template <typename indexes_type_, typename progress_type_ = dummy_progress_t>
expected<refine_report_t> refine(indexes_type_& index, refine_config_t config,
                                 typename indexes_type_::executor_t& executor,
                                 progress_type_&& progress = {}) noexcept {
    auto [plan, status] = plan_refine(index, config, executor);
    if (failed(status)) return {{}, status};
    if (status_t const driven = nvme_drive_(plan, executor, progress); failed(driven)) return {plan.running(), driven};
    return plan.finish(executor);
}

template <typename indexes_type_, typename queue_type_>
    requires std::is_same_v<nvme_executor<queue_type_>, typename indexes_type_::executor_t>
expected<nvme_reshard_plan<indexes_type_>> plan_reshard(std::span<indexes_type_* const> sources,
                                                        std::span<indexes_type_* const> targets,
                                                        nvme_executor<queue_type_>& executor,
                                                        reshard_config_t config) noexcept {
    nvme_reshard_plan<indexes_type_> plan;
    if (sources.empty() || targets.empty()) return {std::move(plan), status_t::invalid_argument_k};
    for (std::span<indexes_type_* const> const listed : {sources, targets})
        for (std::size_t position = 0; position != listed.size(); ++position) {
            if (!listed[position]) return {std::move(plan), status_t::invalid_argument_k};
            for (std::size_t earlier = 0; earlier != position; ++earlier)
                if (listed[earlier] == listed[position]) return {std::move(plan), status_t::invalid_argument_k};
        }
    if (!config.refine.cache_bytes) config.refine.cache_bytes = executor.cache_bytes_per_core();
    status_t const opened = targets.size() == 1 ? plan.open_gathered_(sources, targets, config)
                                                : plan.open_scattered_(sources, targets, config);
    return {std::move(plan), opened};
}

/**
 *  @brief Reshards the loaded @p sources into the loaded @p targets: the one-shot
 *      @ref plan_reshard, run to its end and finished. @p progress hears the phase's tasks run and
 *      in all after each; a cancel leaves every index serving its own parts.
 *
 *  @par Statuses:
 *  - @c cancelled_k @p progress stopped the rewrite.
 *  - Any status @ref plan_reshard, its tasks or its finish answers.
 */
template <typename indexes_type_, typename queue_type_, typename progress_type_ = dummy_progress_t>
    requires std::is_same_v<nvme_executor<queue_type_>, typename indexes_type_::executor_t>
expected<reshard_report_t> reshard(std::span<indexes_type_* const> sources, std::span<indexes_type_* const> targets,
                                   nvme_executor<queue_type_>& executor, reshard_config_t config = {},
                                   progress_type_&& progress = {}) noexcept {
    auto [plan, status] = plan_reshard(sources, targets, executor, config);
    if (failed(status)) return {{}, status};
    if (status_t const driven = nvme_drive_(plan, executor, progress); failed(driven)) return {{}, driven};
    return plan.finish(executor);
}

} // namespace unum::usearch::dense
