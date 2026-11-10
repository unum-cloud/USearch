/**
 *  @file include/usearch/dense/nvme_compaction.hpp
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The freestanding compaction, refine and build that write new SafeGraphs parts from the
 *      parts of @c nvme_indexes views.
 *
 *  Compaction, refine and build are free functions over the views, each planning its rewrite as
 *  phases of steps that borrow an NVMe executor one at a time and write SafeGraphs parts beside a
 *  manifest. A rewrite never edits its sources. It lays its target out as spec-standard parts,
 *  keys, values, neighbors and the producer's norms, whose headers arrive last and the manifest
 *  after them, so a stopped rewrite reads as absent. Its RAM is the workspace the caller lends, the
 *  executor's arenas and its staging buffers: the workspace keeps a changed bit per target row at
 *  its front, every step carves its blocks and batches off an arena over the rest, sized from it,
 *  and a sweep tile's links stay at the front of that across the tile's steps. Reads go through the
 *  executor's engines, the target's included, and writes are blocking positional runs.
 *
 *  A compaction packs the source parts largest first into target parts of at most a shard capacity
 *  of rows, so one verb packs small parts together and, under a capacity below a part's rows, cuts
 *  it.
 *
 *  @section dense_nvme_compaction_phases Phases
 *
 *  - @b Copy moves keys and values in target order, derives the norms, and rewires the neighbors a
 *    target keeps.
 *  - @b Build builds contiguous blocks of rows in a RAM index inside the workspace and writes their
 *    rows.
 *  - @b Re-select re-selects a tile of target rows by walks of the target itself, each row's picks
 *    competing with its neighbors, and writes the tile, or runs in RAM when a whole part fits.
 *
 *  @b Sweep collects the walked rows' links into tiles of target rows sized from the workspace, so
 *  no reverse link is lost between walk tiles. One sequential scan of the walked rows' neighbors
 *  counts the links into a tile, cut at the last row whose links fit its tape, and the rest opens
 *  the next tile. A second scan scatters them onto the tape, and each row of the tile scores its
 *  run, merges it with its members, and is written a second time.
 *
 *  @b Heal sweeps reachability from the landmarks each audit seeds from, and links islands back in.
 */
#pragma once

#include <cstdio>  // `std::remove`, `std::snprintf`
#include <cstring> // `std::memcpy`, `std::memset`

#include <array>       // `std::array`
#include <memory>      // `std::unique_ptr`, `std::destroy_at` - the RAM index a block builds in
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <utility>     // `std::move`, `std::exchange`

#include "usearch/utilities/ram_sorts.hpp" // `partial_sort`
#include "usearch/dense/types.hpp"         // `compact_config_t`, `compact_result_t`
#include "usearch/dense/ram_index.hpp"     // `ram_index`, the RAM index a block build fills
#include "usearch/dense/nvme_indexes.hpp"
#include "usearch/dense/shard_checkpoints.hpp" // `monolith_layout_t`, `monolith_header` - the target parts' headers

namespace unum::usearch::dense {

/** What the current phase's steps do, declared in the order every rewrite runs them, which
 *  @c nvme_compact_plan::advance steps through one at a time, skipping phases with no steps, the
 *  three back-link phases once per sweep tile; @c finish_k once it answered false. */
enum class nvme_compact_phase_t : u8_t {

    /** Moves keys and values into the target in target order, with their norms and the neighbors it
     *  keeps. */
    copy_k,

    /** Builds one block of target rows in a RAM index and writes their neighbors. */
    build_k,

    /** Re-selects target rows: the appended part, repaired rows, or the refined range. */
    reselect_k,

    /** Counts the links of the walked rows into the open sweep tile of target rows. */
    count_back_links_k,

    /** Scatters those links onto the tile's tape, into each target's run. */
    scatter_back_links_k,

    /** Scores each target's run, merges it into the target's row, and writes the row. */
    merge_back_links_k,

    /** Counts every row's in-links, then links the rows nothing links to back in. */
    heal_k,

    /** Nothing left but @ref nvme_compact_plan::finish. */
    finish_k,
};

/** Outcome of an NVMe compaction, refine or build: what every compaction reports, and the
 *  reads and writes behind it. */
struct [[nodiscard]] nvme_compact_result_t {

    /** The shards, the moved rows, the cut edges, and the re-selections, block builds and heal,
     *  counted as a RAM refine counts them, but for the heal of each target part: its
     *  @c islands_found are the rows nothing links to, and no sweep runs on it, so
     *  @c reachable_nodes and @c sweeps stay zero; @c health on the committed view measures them.
     *  A refine moves no rows. */
    compact_result_t compaction;

    /** Neighbors rows the walks and the heal read. */
    nvme_reads_t neighbor_reads;

    /** Value rows the walks, the selections and the back-links read. */
    nvme_reads_t value_reads;

    /** Runs of whole rows the copies, the batches, the sweeps and the heal scanned. */
    nvme_reads_t scan_reads;

    /** Bytes the target parts received. */
    std::size_t written_bytes = 0;
};

template <typename indexes_type_>
struct nvme_compact_plan;

/**
 *  @brief Compacts the parts of @p sources into new parts at @p target, each holding at most
 *      @p config.shard_capacity rows.
 *
 *  Parts pack largest first, each bin's largest part keeps its IDs, every other part arrives after
 *  it with its edges rebased, its rows walk the target and link in, a sweep merges their back-links
 *  into the rows they name, then each target part heals; one manifest commits them all. A part past
 *  the capacity is cut into ranges of it, the edges the cut crosses dropped and counted, and every
 *  row of their target parts re-selects. @p save names the target parts' alignment and provenance,
 *  and the target is a manifest beside its parts whatever its layout. The sources are borrowed
 *  until the plan dies and never edited.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No source, a @c NULL first source, a first source viewing nothing, or
 *    sources holding no rows.
 *  - @c config_mismatch_k @p save names an alignment below the floor or not a power of two or a
 *    reserved provenance field, or a source walks by another kernel or differs in geometry or
 *    key width from the first.
 *  - @c invalid_argument_k An empty @p target, a @c NULL later source, or a target part path that
 *    does not parse.
 *  - @c bad_alloc_k @p workspace cannot hold one row of some phase.
 *  - @c overflow_k A target part of zero rows or past the node ID space, or a device ending before
 *    its base plus a part.
 *  - @c malformed_k A framed target header has no width.
 */
template <typename indexes_type_>
expected<nvme_compact_plan<indexes_type_>> plan_compact(std::span<indexes_type_ const* const> sources,
                                                        std::string_view target, bytes_span_t workspace,
                                                        compact_config_t const& config = {},
                                                        save_config_t const& save = {}) noexcept;

/**
 *  Copies @p source and re-selects @p rows of it, in placement order across its parts, by walks of
 *  the copy, or in RAM for a whole part the workspace holds, then merges the back-links and heals;
 *  empty rows heal alone. Every part keeps its rows, so the capacity plays no part. The source is
 *  borrowed until the plan dies.
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
expected<nvme_compact_plan<indexes_type_>> plan_refine(indexes_type_ const& source, iota_view<std::size_t> rows,
                                                       std::string_view target, bytes_span_t workspace,
                                                       compact_config_t const& config = {},
                                                       save_config_t const& save = {}) noexcept;

/**
 *  Builds one graph per target part over every row of the parts at @p sources, their neighbors
 *  ignored, the parts packed as a compaction packs them, in contiguous blocks the workspace holds
 *  as a RAM index, each healed there before it is written, then links each part's later blocks
 *  across as a compaction links its appended parts.
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
expected<nvme_compact_plan<indexes_type_>> plan_build(std::span<std::string_view const> sources,
                                                      std::string_view target, bytes_span_t workspace,
                                                      compact_config_t const& config = {},
                                                      save_config_t const& save = {}) noexcept;

/**
 *  @brief One rewrite in flight: phases of steps, each step run across every thread of an executor.
 *
 *  Steps of a phase may run in any order, but one at a time, since each claims the executor while
 *  it runs, so searches on the same executor run between steps; a failed step may run again. The
 *  back-link phases repeat once per sweep tile, each tile's links living across its steps.
 *  @ref advance is the single-caller barrier between phases. A dying plan removes every target part
 *  no manifest committed, so a stopped rewrite reads as absent.
 */
template <typename indexes_type_>
struct [[nodiscard]] nvme_compact_plan {
    using indexes_t = indexes_type_;
    using executor_t = typename indexes_t::executor_t;
    using key_t = typename indexes_t::key_t;
    using result_t = nvme_compact_result_t;

    /** One unit of the current phase. */
    struct step_t {

        /** The step's position in its phase. */
        std::size_t position = 0;

        /** The source part a copy step reads, or the target part every other step writes. */
        std::size_t part = 0;

        /** The node IDs it covers: source ones for a copy step, target ones otherwise. */
        iota_view<node_id_t> nodes;
    };

    nvme_compact_plan() noexcept = default;
    nvme_compact_plan(nvme_compact_plan&&) noexcept = default;
    nvme_compact_plan& operator=(nvme_compact_plan&&) = delete;
    nvme_compact_plan(nvme_compact_plan const&) = delete;
    nvme_compact_plan& operator=(nvme_compact_plan const&) = delete;

    /** Removes the target parts unless @ref finish committed them. */
    ~nvme_compact_plan() noexcept {
        for (target_t& target : targets_) {
            [[maybe_unused]] status_t const closed = target.file.close();
            target.view = indexes_t {allocator_};
            std::remove(target.path.data());
        }
    }

    /** Step @p position of the current phase; an empty one past the last. */
    step_t operator[](std::size_t position) const noexcept {
        for (std::size_t unit = 0, first = 0; unit != units_(); ++unit) {
            cut_t const cut = cut_(unit);
            std::size_t const count = cut.steps();
            if (position - first < count) {
                iota_view<std::size_t> const rows = cut.rows_of(position - first);
                return {position, unit, {static_cast<node_id_t>(rows.first), static_cast<node_id_t>(rows.bound)}};
            }
            first += count;
        }
        return {};
    }

    /** The current phase's steps, positionally; none once the plan is done. */
    indexed_range<nvme_compact_plan> steps() const noexcept {
        std::size_t count = 0;
        for (std::size_t unit = 0; unit != units_(); ++unit) count += cut_(unit).steps();
        return {this, count};
    }

    /** The phase whose steps run now. */
    nvme_compact_phase_t phase() const noexcept { return phase_; }

    /**
     *  Runs @p step of the current phase across every thread of @p executor, which it claims while
     *  it runs.
     *
     *  @par Statuses:
     *  - @c config_mismatch_k @p step lies past the current phase's steps, a part rebuilt in RAM
     *    disagrees with its rows, or a medium the engines cannot read.
     *  - @c uninitialized_k The plan is done, or @p executor holds no pool or workers.
     *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
     *  - @c bad_alloc_k The workspace cannot hold the step's slices.
     *  - @c overflow_k A part rebuilt in RAM passes a bound, a staging region's byte count
     *    overflows, or a write runs past a device's end.
     *  - @c unknown_kind_k A part rebuilt in RAM cannot finalize its kernel.
     */
    status_t run(step_t const& step, executor_t& executor) noexcept {
        if (step.position >= steps().size()) return status_t::config_mismatch_k;
        switch (phase_) {
        case nvme_compact_phase_t::copy_k: return copy_(step, executor);
        case nvme_compact_phase_t::build_k: return build_(step, executor);
        case nvme_compact_phase_t::reselect_k: return reselect_(step, executor);
        case nvme_compact_phase_t::count_back_links_k: return count_back_links_(step, executor);
        case nvme_compact_phase_t::scatter_back_links_k: return scatter_back_links_(step, executor);
        case nvme_compact_phase_t::merge_back_links_k: return merge_back_links_(step, executor);
        case nvme_compact_phase_t::heal_k: return heal_(step, executor);
        default: return status_t::uninitialized_k;
        }
    }

    /** The barrier between phases, one caller after every step ran: enters the next phase with
     *  steps and answers whether one follows. */
    bool advance() noexcept {
        while (phase_ != nvme_compact_phase_t::finish_k) {
            switch (phase_) {
            case nvme_compact_phase_t::reselect_k: open_sweep_(0, 0); break;
            case nvme_compact_phase_t::count_back_links_k:
                offset_back_links_();
                phase_ = nvme_compact_phase_t::scatter_back_links_k;
                break;
            case nvme_compact_phase_t::merge_back_links_k:
                if (sweep_.bound < targets_[sweep_part_].view.size()) open_sweep_(sweep_part_, sweep_.bound);
                else open_sweep_(sweep_part_ + 1, 0);
                break;
            default: phase_ = static_cast<nvme_compact_phase_t>(static_cast<u8_t>(phase_) + 1);
            }
            if (!steps().empty()) return true;
        }
        return false;
    }

    /**
     *  Writes every part header, then the manifest: the commit.
     *
     *  @par Statuses:
     *  - @c uninitialized_k The plan holds no target part.
     *  - @c cancelled_k The last phase has not ended.
     *  - @c config_mismatch_k A part header does not fit before its data.
     *  - @c overflow_k A part is shorter than its header, or a device ends before its base plus the
     *    manifest.
     *  - @c invalid_argument_k The manifest path does not parse.
     */
    expected<result_t> finish() noexcept {
        if (targets_.empty()) return {result_, status_t::uninitialized_k};
        if (phase_ != nvme_compact_phase_t::finish_k) return {result_, status_t::cancelled_k};
        std::span<safegraphs_custom_field_t const> const fields = to_span(fields_);
        for (target_t& target : targets_) {
            std::size_t const bytes = target.header.monolith_bytes();
            if (status_t const framed = safegraphs_write_header(file_span_t {&target.file, 0, bytes}, target.header,
                                                                fields);
                failed(framed))
                return {result_, framed};
            if (status_t const flushed = target.file.flush(); failed(flushed)) return {result_, flushed};
        }
        auto part_at = [&](std::size_t part) noexcept {
            target_t const& target = targets_[part];
            return safegraphs_part_t {safegraphs_basename(path_of_(target.path)), target.header.monolith_bytes(),
                                      target.view.size()};
        };
        if (status_t const committed = safegraphs_write_manifest(path_of_(manifest_), targets_.size(), part_at);
            failed(committed))
            return {result_, committed};
        result_.compaction.source_shards = sources_.size();
        result_.compaction.target_shards = targets_.size();
        // Committed parts outlive the plan, so its destructor finds none to remove.
        targets_.clear();
        return {result_, status_t::success_k};
    }

  private:
    template <typename other_type_>
    friend expected<nvme_compact_plan<other_type_>> plan_compact(std::span<other_type_ const* const>, std::string_view,
                                                                 bytes_span_t, compact_config_t const&,
                                                                 save_config_t const&) noexcept;
    template <typename other_type_>
    friend expected<nvme_compact_plan<other_type_>> plan_refine(other_type_ const&, iota_view<std::size_t>,
                                                                std::string_view, bytes_span_t, compact_config_t const&,
                                                                save_config_t const&) noexcept;
    template <typename other_type_>
    friend expected<nvme_compact_plan<other_type_>> plan_build(std::span<std::string_view const>, std::string_view,
                                                               bytes_span_t, compact_config_t const&,
                                                               save_config_t const&) noexcept;

    template <typename word_type_>
    using atomic_ref = typename indexes_t::template atomic_ref<word_type_>;
    using wait_t = typename indexes_t::wait_t;
    using scalar_t = typename indexes_t::scalar_t;
    using dot_result_t = typename indexes_t::dot_result_t;
    using vector_view_t = typename indexes_t::vector_view_t;
    using vectors_view_t = typename indexes_t::vectors_view_t;
    using vector_norms_t = typename indexes_t::vector_norms_t;
    using search_config_t = typename indexes_t::search_config_t;
    using medium_t = typename indexes_t::medium_t;
    using walk_t = typename indexes_t::walk_t;
    using results_array_t = typename indexes_t::results_array_t;
    using scratch_allocator_t = typename indexes_t::scratch_allocator_t;
    using paths_t = growing_buffer<char, forkunion_allocator_t>;

    /** The arena every step carves the workspace from, the workspace its only chunk. */
    using workspace_upstream_t = lent_allocator<byte_t>;
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

    /** One target part: its file, its header, a one-part view the walks and scans read it
     *  through, and the rows it re-selects. */
    struct target_t {
        paths_t path;
        seekable_file_t file;
        safegraphs_header_t header;
        indexes_t view;

        /** The norms column, which the header's producer columns view. */
        safegraphs_named_column_t norms;

        /** The rows the re-select phase re-selects, all of them unless the plan narrows them. */
        iota_view<std::size_t> selected;

        /** The first row of the appended nodes of a compaction or a build, every row below it
         *  linking to no appended row yet; zero for none. */
        std::size_t resident = 0;

        /** Where the part's rows start among the changed bits. */
        std::size_t changed_first = 0;
    };

    /** Every row's candidates of one batch: the best rows by exact distance, with their bytes
     *  when the batch keeps them, so a selection reads nothing again. */
    struct candidates_t {
        scored_node_t* entries = nullptr;
        byte_t* vectors = nullptr;
        u32_t* counts = nullptr;
        std::size_t slots = 0;
        std::size_t row_bytes = 0;

        scored_node_t* entries_of(std::size_t row) const noexcept { return entries + row * slots; }
        byte_t* vector_of(std::size_t row, std::size_t slot) const noexcept {
            return vectors + (row * slots + slot) * row_bytes;
        }

        /** Keeps @p candidate among the candidates of @p row, at most @p limit long, unless it is
         *  there already or every kept one is closer; answers the slot it took, or @p limit. */
        std::size_t offer(std::size_t row, scored_node_t candidate, std::size_t limit) const noexcept {
            scored_node_t* const kept = entries_of(row);
            std::size_t const count = counts[row];
            std::size_t worst = 0;
            for (std::size_t slot = 0; slot != count; ++slot) {
                if (kept[slot].node_id == candidate.node_id) return limit;
                if (kept[slot].distance > kept[worst].distance) worst = slot;
            }
            std::size_t slot = count;
            if (count >= limit) {
                if (!count || !(candidate.distance < kept[worst].distance)) return limit;
                slot = worst;
            }
            else ++counts[row];
            kept[slot] = candidate;
            return slot;
        }
    };

    /** The rows of a re-select batch: every scored row of every walk is offered to the row's
     *  candidates with its bytes, the row itself excepted. */
    struct reselect_query_t {
        using key_t = typename indexes_t::key_t;
        inline static constexpr nvme_key_reads_t key_reads_k = nvme_key_reads_t::outputs_k;
        inline static constexpr bool admitting_k = false;

        /** Offers every scored lane of a quad to the row's candidates with the bytes its rescoring
         *  saw. */
        struct fold_t {
            reselect_query_t* row = nullptr;

            walk_demand_t absorb(std::span<scored_node_t const> scored, std::span<dot_result_t const>) const noexcept {
                inputs_t const& inputs = row->inputs;
                candidates_t const& candidates = inputs.candidates;
                for (std::size_t lane = 0; lane != scored.size(); ++lane) {
                    if (scored[lane].node_id == inputs.first + row->row) continue;
                    std::size_t const slot = candidates.offer(row->row, scored[lane], inputs.limit);
                    if (slot != inputs.limit)
                        std::memcpy(candidates.vector_of(row->row, slot), row->fetched[lane].byte_data(),
                                    candidates.row_bytes);
                }
                return walk_demand_t::continue_k;
            }
        };

        /** The batch every row of the call reads and fills. */
        struct inputs_t {
            vectors_view_t queries;
            candidates_t candidates;
            node_id_t first = 0;
            std::size_t limit = 0;
        };

        inputs_t const& inputs;
        std::size_t row = 0;
        vector_view_t query;
        vector_norms_t query_norms {};

        /** The quad being scored, between its rescoring and its fold. */
        quad<vector_view_t> fetched {};

        reselect_query_t(inputs_t const& inputs, scratch_allocator_t) noexcept : inputs(inputs) {}

        status_t reserve(std::size_t, std::size_t) noexcept { return status_t::success_k; }

        /** Primes @p new_query_index; its candidates keep what earlier walks of the batch kept. */
        void prime_query(std::size_t new_query_index) noexcept {
            row = new_query_index;
            query = inputs.queries.row(row).as_vector();
        }
        void rescore_quad(quad<node_id_t> const&, quad<vector_view_t> const& vectors, quad<vector_norms_t> const&,
                          scores_result<dot_result_t>&, std::size_t, std::span<key_t const>) noexcept {
            fetched = vectors;
        }
        fold_t fold_of_walk(walk_t const&) noexcept { return {this}; }
        results_array_t const& stopping_candidates(walk_t const& walk) const noexcept { return walk.results; }
        void finish_walk(walk_t const&) noexcept {}
        std::size_t output_count() const noexcept { return 0; }
        shard_node_t output_locator(std::size_t) const noexcept { return {}; }
        void resolve_output_key(std::size_t, key_t const&) noexcept {}
        status_t finish_query() noexcept { return status_t::success_k; }
    };

    /** One column a scan fetches: where its row zero sits in the part, a row's width, and where row
     *  zero arrives. */
    struct staged_column_t {
        u64_t begin = 0;
        std::size_t row_bytes = 0;
        byte_t* staging = nullptr;
    };

    /** The per-row scratch of one diverse selection. */
    struct selection_t {
        u32_t* order = nullptr;
        u32_t* kept = nullptr;
        vector_norms_t* kept_norms = nullptr;
    };

    /** The slices of one step so far: the arena they come off, null to only sum their bytes, and
     *  the bytes they reserve, the padding up to every next boundary included. */
    struct carver_t {
        workspace_arena_t* arena = nullptr;
        std::size_t bytes = 0;
    };

    /** The columns a copy step stages, one row each per row it copies. */
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

    /** A re-select or merge batch: the rows' own values, their neighbors in and out and as they
     *  were, the candidates with their rows, the reads and then the selections, and for a merge
     *  what each row's last round carried over and where it stands in its run. */
    struct reselect_slices_t {
        byte_t* queries = nullptr;
        node_id_t* neighbors = nullptr;
        node_id_t* previous_neighbors = nullptr;
        u32_t* counts = nullptr;
        scored_node_t* entries = nullptr;
        byte_t* vectors = nullptr;
        u32_t* orders = nullptr;
        u32_t* kept_slots = nullptr;
        vector_norms_t* kept_norms = nullptr;
        u32_t* carried = nullptr;
        u32_t* cursors = nullptr;
    };

    /** A sweep tile's links: each row's count, then the offset of its run in the tape, then that
     *  run's end, and the tape of rows linking to each. */
    struct sweep_slices_t {
        std::size_t* incoming_offsets = nullptr;
        node_id_t* back_links = nullptr;
    };

    /** A heal's in-link counts, the target row it rewrites and its island links. */
    struct heal_slices_t {
        u64_t* in_links = nullptr;
        node_id_t* target_row = nullptr;
        u64_t* island_links = nullptr;
    };

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

    /** Opens the plan's common state: the workspace, the configurations and the header fields
     *  naming the kernel of @p view, and the target's paths, and removes a stale manifest so no
     *  half-written part is ever named. */
    status_t open_(std::string_view target, bytes_span_t workspace, compact_config_t const& config,
                   save_config_t const& save, indexes_t const& view) noexcept {
        if (status_t const valid = save.validate(); failed(valid)) return valid;
        if (target.empty()) return status_t::invalid_argument_k;
        if (!view.shards_count()) return status_t::uninitialized_k;
        // Trimmed to whole boundaries, so a step's slices take exactly the bytes they reserve.
        std::size_t const alignment = workspace_allocator_t::alignment_k;
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(workspace.data());
        std::size_t const padding = min_of<std::size_t>(
            round_up_to_multiple<std::uintptr_t>(address, alignment) - address, workspace.size());
        workspace_ = workspace.subspan(padding, (workspace.size() - padding) / alignment * alignment);
        config_ = config;
        save_ = save;
        if (failed(fields_.reserve_exactly(1 + save.provenance.size()))) return status_t::bad_alloc_k;
        fields_.push_back_reserved({"metric", nk_kernel_name(view.kernel())});
        for (safegraphs_custom_field_t const& field : save.provenance) fields_.push_back_reserved(field);
        if (failed(keep_path_(manifest_, target)) || failed(keep_path_(stem_, safegraphs_stem(target))))
            return status_t::bad_alloc_k;
        std::remove(manifest_.data());
        return status_t::success_k;
    }

    /** Appends every non-empty part of @p view to the sources, refusing a geometry or a kernel
     *  other than those of the first part of @p first, the view the plan opened with. */
    status_t add_sources_(indexes_t const& view, indexes_t const& first) noexcept {
        nvme_part_t const& shape = first.parts_[0];
        if (view.kernel() != first.kernel()) return status_t::config_mismatch_k;
        for (u32_t part = 0; part != view.parts_.size(); ++part) {
            nvme_part_t const& opened = view.parts_[part];
            if (opened.dimensions != shape.dimensions || opened.connectivity != shape.connectivity ||
                opened.value_row_bytes != shape.value_row_bytes || opened.key_dtype != safegraphs_key_dtype<key_t>())
                return status_t::config_mismatch_k;
            if (!opened.nodes) continue;
            source_t source;
            source.view = &view;
            source.part = part;
            source.rows = opened.nodes;
            source.stride = opened.nodes;
            if (failed(sources_.push_back(source))) return status_t::bad_alloc_k;
        }
        return status_t::success_k;
    }

    /** Creates one target part per entry of @p rows, with its header framed, its file sized and its
     *  one-part view open, and keeps a changed bit per row of them at the workspace's front. */
    status_t open_targets_(std::span<std::size_t const> rows) noexcept {
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
            auto const [path, path_status] = safegraphs_part_path(name, path_of_(stem_), part, rows.size());
            if (failed(path_status)) return path_status;
            if (status_t const kept = keep_path_(target.path, path); failed(kept)) return kept;
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
        }
        std::size_t const changed_words = divide_round_up<std::size_t>(total, 64);
        std::size_t const changed_bytes = workspace_allocator_t::reserved_bytes(changed_words * sizeof(u64_t));
        if (changed_bytes > workspace_.size()) return status_t::bad_alloc_k;
        changed_ = {reinterpret_cast<u64_t*>(workspace_.data()), changed_words};
        std::memset(changed_.data(), 0, changed_.size_bytes());
        workspace_ = workspace_.subspan(changed_bytes);
        std::size_t widest = 0;
        for (std::size_t const part_rows : rows) widest = max_of(widest, part_rows);
        if (heal_bytes_(widest) > workspace_.size()) return status_t::bad_alloc_k;
        return size_steps_();
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
                result_.compaction.moved_entries += source.rows - limit;
                continue;
            }
            std::size_t bin = 0;
            while (bin != rows.size() && rows[bin] + source.rows > limit) ++bin;
            if (bin == rows.size() && failed(rows.push_back(0))) return status_t::bad_alloc_k;
            source.target = bin;
            source.base = rows[bin];
            rows[bin] += source.rows;
            if (source.base) result_.compaction.moved_entries += source.rows;
        }
        return status_t::success_k;
    }

    /** Refuses a workspace too small for a single row of any phase the plan runs. */
    status_t size_steps_() const noexcept {
        if (!copy_rows_() || !reselect_rows_() || !sweep_rows_() || !sweep_scan_rows_() || !merge_rows_())
            return status_t::bad_alloc_k;
        return status_t::success_k;
    }

    /** The units the current phase cuts into steps: source parts while copying, target parts
     *  after. */
    std::size_t units_() const noexcept {
        switch (phase_) {
        case nvme_compact_phase_t::copy_k: return sources_.size();
        case nvme_compact_phase_t::finish_k: return 0;
        default: return targets_.size();
        }
    }

    /** Rows of one unit of the current phase: @b [first,first+rows) cut at every @c stride rows
     *  past @c first, and into steps of at most @c chunk rows. */
    struct cut_t {
        std::size_t first = 0;
        std::size_t rows = 0;
        std::size_t stride = 1;
        std::size_t chunk = 1;

        std::size_t steps() const noexcept {
            return rows / stride * divide_round_up(stride, chunk) + divide_round_up(rows % stride, chunk);
        }

        /** The rows of step @p step of the unit. */
        iota_view<std::size_t> rows_of(std::size_t step) const noexcept {
            std::size_t const per_stride = divide_round_up(stride, chunk), range = step / per_stride;
            std::size_t const offset = range * stride + step % per_stride * chunk;
            return {first + offset, first + min_of(offset + chunk, min_of((range + 1) * stride, rows))};
        }
    };

    /** Whether the re-select phase walks rows of target part @p part, whose links a sweep collects;
     *  a part refined whole in RAM sweeps there. */
    bool walked_(std::size_t part) const noexcept {
        iota_view<std::size_t> const rows = targets_[part].selected;
        std::size_t const nodes = targets_[part].view.size();
        return !rows.empty() && !(rows.size() == nodes && refines_in_ram_(nodes));
    }

    /** How the current phase cuts unit @p unit: a copy step stays inside one target range of its
     *  source part, a build step is a block, a re-select step a batch or a whole part refined in
     *  RAM, a count or scatter step a run of the walked rows, a merge step a batch of the sweep's
     *  tile, and a heal step a whole part. */
    cut_t cut_(std::size_t unit) const noexcept {
        switch (phase_) {
        case nvme_compact_phase_t::copy_k: {
            source_t const& source = sources_[unit];
            return {0, source.rows, source.stride, copy_rows_()};
        }
        case nvme_compact_phase_t::build_k: {
            std::size_t const nodes = targets_[unit].view.size(), block_rows = build_rows_();
            return {0, block_rows ? nodes : 0, nodes, max_of<std::size_t>(block_rows, 1)};
        }
        case nvme_compact_phase_t::reselect_k: {
            iota_view<std::size_t> const rows = targets_[unit].selected;
            std::size_t const nodes = targets_[unit].view.size();
            // A whole part the workspace holds as one RAM index is refined in RAM, in one step.
            bool const in_ram = rows.size() == nodes && refines_in_ram_(nodes);
            return {rows.first, rows.size(), max_of<std::size_t>(rows.size(), 1), in_ram ? nodes : reselect_rows_()};
        }
        case nvme_compact_phase_t::count_back_links_k:
        case nvme_compact_phase_t::scatter_back_links_k: {
            iota_view<std::size_t> const rows = targets_[unit].selected;
            return unit == sweep_part_ ? cut_t {rows.first, rows.size(), rows.size(), sweep_scan_rows_()} : cut_t {};
        }
        case nvme_compact_phase_t::merge_back_links_k:
            return unit == sweep_part_ ? cut_t {sweep_.first, sweep_.size(), sweep_.size(), merge_rows_()} : cut_t {};
        case nvme_compact_phase_t::heal_k: {
            std::size_t const nodes = targets_[unit].view.size();
            return {0, nodes, nodes, nodes};
        }
        default: return {};
        }
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

    /** Bytes of one norms row, zero when the kernel reads no norms and the targets carry no
     *  column. */
    std::size_t norms_bytes_() const noexcept { return sources_[0].view->stores_norms() ? sizeof(vector_norms_t) : 0; }

    std::size_t expansion_factor_() const noexcept {
        return config_.expansion_factor ? config_.expansion_factor : default_expansion_factor(connectivity_());
    }

    /** Rows one copy step covers, the most the workspace stages. */
    std::size_t copy_rows_() const noexcept {
        return fitting_rows_(&nvme_compact_plan::copy_slices_, workspace_.size() / value_row_bytes_(),
                             workspace_.size());
    }

    /** Rows one re-select step covers, the largest batch the workspace holds. */
    std::size_t reselect_rows_() const noexcept {
        return fitting_rows_(&nvme_compact_plan::reselect_slices_, workspace_.size() / value_row_bytes_(),
                             workspace_.size());
    }

    /** Rows one sweep tile opens with: its counts and tape take at most half the workspace, and the
     *  rest stages its scans and merges, whose batches a wider tile would narrow to single rows. */
    std::size_t sweep_rows_() const noexcept {
        return fitting_rows_(&nvme_compact_plan::sweep_slices_, workspace_.size() / neighbors_row_bytes_(),
                             workspace_.size() / 2);
    }

    /** The workspace past a sweep tile's links, staging the tile's scans and merges. */
    bytes_span_t sweep_spare_() const noexcept {
        carver_t counter;
        sweep_slices_(sweep_rows_(), counter);
        return workspace_.subspan(counter.bytes);
    }

    /** Rows of the walked ones that one count or scatter step scans. */
    std::size_t sweep_scan_rows_() const noexcept { return sweep_spare_().size() / neighbors_row_bytes_(); }

    /** Rows of a sweep tile that one merge step covers. */
    std::size_t merge_rows_() const noexcept {
        std::size_t const spare = sweep_spare_().size();
        return fitting_rows_(&nvme_compact_plan::reselect_slices_, spare / value_row_bytes_(), spare);
    }

    /** Rows one build step covers, the largest block the workspace fits; none unless the plan
     *  builds over the views it opened. */
    std::size_t build_rows_() const noexcept {
        if (opened_.empty()) return 0;
        std::size_t rows = 0;
        for (source_t const& source : sources_) rows += source.rows;
        return fitting_rows_(&nvme_compact_plan::build_slices_, rows, workspace_.size());
    }

    std::size_t neighbors_row_bytes_() const noexcept { return connectivity_() * sizeof(node_id_t); }
    std::size_t candidate_slots_() const noexcept { return expansion_factor_() + connectivity_(); }

    /** Bytes a heal of a part of @p rows rows takes at least, with room for one island link; the
     *  rest of the workspace holds more island links. */
    std::size_t heal_bytes_(std::size_t rows) const noexcept {
        carver_t counter;
        heal_slices_(rows, 1, counter);
        return counter.bytes;
    }

    /** Whether a whole part of @p rows rows refines in RAM: its index and one staged row fit. */
    bool refines_in_ram_(std::size_t rows) const noexcept {
        carver_t counter;
        refine_slices_(rows, 1, counter);
        return counter.bytes <= workspace_.size();
    }

    /** Bytes the columns of a RAM index of @p rows rows take off the arena it is built on: a slice
     *  per column of every chunk the rows reach, the key map's control bytes and slots over them,
     *  and the audit and the pass its refine and heal run on the same allocator. */
    std::size_t ram_bytes_(std::size_t rows) const noexcept {
        using schedule_t = typename ram_index_t::chunk_schedule_t;
        using slots_schedule_t = typename ram_index_t::keyed_t::nodes_map_t::slots_schedule_t;
        typename ram_index_t::geometry_t const geometry {dimensions_(), static_cast<connectivity_t>(connectivity_())};
        std::size_t const values_bytes = ram_index_t::padded_row_values(geometry) * sizeof(scalar_t);
        std::size_t bytes = workspace_allocator_t::reserved_bytes(ram_index_t::audit_memory_usage(rows)) +
                            workspace_allocator_t::reserved_bytes(
                                ram_index_t::refine_memory_usage(rows, rows, connectivity_(), 0));
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

#pragma endregion Sizes

#pragma region Slices

    /** An arena whose only chunk is the whole of @p space, which starts on a boundary. */
    static workspace_arena_t arena_over_(bytes_span_t space) noexcept {
        workspace_arena_t arena {workspace_upstream_t {space}};
        // Never refused: the upstream hands out exactly the span it lends.
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

    reselect_slices_t reselect_slices_(std::size_t rows, carver_t& carver) const noexcept {
        std::size_t const slots = candidate_slots_(), connectivity = connectivity_();
        return {carve_<byte_t>(carver, rows * value_row_bytes_()),
                carve_<node_id_t>(carver, rows * connectivity),
                carve_<node_id_t>(carver, rows * connectivity),
                carve_<u32_t>(carver, rows),
                carve_<scored_node_t>(carver, rows * slots),
                carve_<byte_t>(carver, rows * slots * value_row_bytes_()),
                carve_<u32_t>(carver, rows * slots),
                carve_<u32_t>(carver, rows * connectivity),
                carve_<vector_norms_t>(carver, rows * connectivity),
                carve_<u32_t>(carver, rows),
                carve_<u32_t>(carver, rows)};
    }

    sweep_slices_t sweep_slices_(std::size_t rows, carver_t& carver) const noexcept {
        return {carve_<std::size_t>(carver, rows), carve_<node_id_t>(carver, rows * connectivity_())};
    }

    /** Two bits per row, the first in-link setting the low one and the second the high one, and
     *  @p island_links island links. */
    heal_slices_t heal_slices_(std::size_t rows, std::size_t island_links, carver_t& carver) const noexcept {
        return {carve_<u64_t>(carver, divide_round_up<std::size_t>(rows, 32)),
                carve_<node_id_t>(carver, connectivity_()), carve_<u64_t>(carver, island_links)};
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

    /** The @c scan_rows_ of a step whose call is already open, a scan's worth of slots
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
        auto const [reads, status] = view.scan_reads_(executor, tasks, describe, consume);
        result_.scan_reads += reads;
        return status;
    }

    /** Reads one row of the column at @p begin, @p row_bytes wide, of part zero of @p view per
     *  task: row @c row_of(task) arrives at @c destination_of(task), and the reads count into
     *  @p reads. */
    template <typename row_of_type_, typename destination_of_type_>
    status_t read_points_(indexes_t const& view, u64_t begin, std::size_t row_bytes, std::size_t tasks,
                          row_of_type_&& row_of, destination_of_type_&& destination_of, executor_t& executor,
                          nvme_reads_t& reads) noexcept {
        if (!tasks) return status_t::success_k;
        auto describe = [&](std::size_t task, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
            if (progress++) return {};
            return {nvme_extent_t {begin + u64_t(row_of(task)) * row_bytes, static_cast<u32_t>(row_bytes), 0}};
        };
        auto consume = [&](std::size_t, std::size_t task, u64_t, auto const&, auto const& fetched) noexcept {
            std::memcpy(destination_of(task), fetched[0], row_bytes);
        };
        auto [call, call_status] = view.begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;
        auto const [point_reads, status] = view.scan_reads_(executor, tasks, describe, consume);
        reads += point_reads;
        return status;
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
        status_t const wrote = target.file.write_at(column + first * row_bytes, bytes_view_t {bytes, rows * row_bytes});
        if (succeeded(wrote)) result_.written_bytes += rows * row_bytes;
        return wrote;
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

    /** The exact distance between two rows. */
    f32_t distance_(byte_t const* first, vector_norms_t first_norms, byte_t const* second,
                    vector_norms_t second_norms) const noexcept {
        return dense::distance<typename indexes_t::quads_kit_t>(kernel_(), view_of_(first), first_norms,
                                                                view_of_(second), second_norms);
    }

    /**
     *  @brief Picks the neighbors of candidate row @p row into @p out, as the RAM insertion picks
     *      them: walking its candidates nearest-first, a candidate is kept only when no kept one is
     *      closer to it than the row is, and the slots left take the nearest rejected candidates.
     *
     *  Sentinels pad @p out.
     *  @return The picks; @p computed counts the distances it took.
     */
    std::size_t select_diverse_(candidates_t const& candidates, std::size_t row, selection_t scratch,
                                std::span<node_id_t> out, std::size_t& computed) const noexcept {
        scored_node_t const* const entries = candidates.entries_of(row);
        std::size_t const count = candidates.counts[row], limit = out.size();
        for (u32_t slot = 0; slot != count; ++slot) scratch.order[slot] = slot;
        partial_sort(std::span<u32_t>(scratch.order, count), count, [&](u32_t first, u32_t second) noexcept {
            return entries[first].distance != entries[second].distance
                       ? entries[first].distance < entries[second].distance
                       : entries[first].node_id < entries[second].node_id;
        });
        std::size_t picks = 0;
        for (std::size_t position = 0; position != count && picks != limit; ++position) {
            u32_t const slot = scratch.order[position];
            byte_t const* const candidate = candidates.vector_of(row, slot);
            vector_norms_t const candidate_norms = norms_of_(candidate);
            std::size_t kept = 0;
            for (; kept != picks; ++kept) {
                ++computed;
                if (distance_(candidate, candidate_norms, candidates.vector_of(row, scratch.kept[kept]),
                              scratch.kept_norms[kept]) < entries[slot].distance)
                    break;
            }
            if (kept != picks) continue;
            scratch.kept[picks] = slot;
            scratch.kept_norms[picks++] = candidate_norms;
        }
        std::size_t const diverse = picks;
        for (std::size_t position = 0; position != count && picks != limit; ++position) {
            u32_t const slot = scratch.order[position];
            std::size_t kept = 0;
            while (kept != diverse && scratch.kept[kept] != slot) ++kept;
            if (kept == diverse) scratch.kept[picks++] = slot;
        }
        for (std::size_t pick = 0; pick != limit; ++pick)
            out[pick] = pick < picks ? entries[scratch.kept[pick]].node_id : sentinel_k;
        return picks;
    }

    /** The scratch of one diverse selection of batch row @p row, out of per-row arrays carved for
     *  the batch. */
    selection_t selection_of_(u32_t* orders, u32_t* kept_slots, vector_norms_t* norms, std::size_t row,
                              std::size_t slots) const noexcept {
        return {orders + row * slots, kept_slots + row * connectivity_(), norms + row * connectivity_()};
    }

    /** Whether the sentinel-padded row @p row holds @p node. */
    static bool holds_(std::span<node_id_t const> row, node_id_t node) noexcept {
        for (node_id_t const neighbor : row)
            if (neighbor == node) return true;
        return false;
    }

    /** Whether the sentinel-padded row @p row holds a member @p before lacks; a set gains a member
     *  only through a pick it lacked, and dropping one alone changes nothing. */
    static bool gained_(std::span<node_id_t const> row, std::span<node_id_t const> before) noexcept {
        for (node_id_t const neighbor : row)
            if (neighbor != sentinel_k && !holds_(before, neighbor)) return true;
        return false;
    }

    /** Marks row @p row of @p target counted as changed, answering whether it was already. */
    bool count_changed_(target_t const& target, std::size_t row) noexcept {
        std::size_t const bit = target.changed_first + row;
        u64_t const mask = u64_t {1} << (bit % 64);
        return atomic_ref<u64_t>(changed_[bit / 64]).fetch_or(mask, std::memory_order_relaxed) & mask;
    }

    /** Folds the telemetry of one batch of walks into the result. */
    void absorb_walks_(typename indexes_t::search_result_t const& walked) noexcept {
        result_.compaction.graph.traversal_stats.visited_members += walked.visited_members;
        result_.compaction.graph.traversal_stats.computed_distances += walked.computed_distances;
        result_.compaction.graph.traversal_stats.restarts += walked.restarts;
        result_.compaction.graph.traversal_stats.saturated_walks += walked.saturated_walks;
        result_.neighbor_reads += walked.neighbor_reads;
        result_.value_reads += walked.value_reads;
    }

#pragma endregion Selection

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
    status_t copy_(step_t const& step, executor_t& executor) noexcept {
        source_t const& source = sources_[step.part];
        nvme_part_t const& part = source.view->parts_[source.part];
        std::size_t const count = step.nodes.size(), first = step.nodes.first;
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
            if (row_cut) atomic_ref<std::size_t>(cut).add(row_cut, std::memory_order_relaxed);
        });
        if (failed(prepared)) return prepared;
        result_.compaction.cut_edges += cut;

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
     *  first arena; runs inside the call its step began, which rewound the arenas. */
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
    void absorb_traversals_(traversals_t const& traversals) noexcept {
        for (traversal_context_t const& traversal : traversals)
            result_.compaction.graph.traversal_stats += traversal.telemetry();
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

    /** Builds target rows @p step.nodes in a RAM index inside the workspace from their value rows,
     *  heals it there, and writes their neighbors, every edge inside the block. */
    status_t build_(step_t const& step, executor_t& executor) noexcept {
        target_t& target = targets_[step.part];
        std::size_t const count = step.nodes.size(), first = step.nodes.first;
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
        // In RAM a block's orphan clusters cost DRAM sweeps to find; once written, only `health`
        // finds them.
        auto [healing, heal_status] = index.plan_refine({}, {}, {}, {static_cast<u32_t>(expansion_factor_())});
        if (failed(heal_status)) return heal_status;
        do {
            auto const steps = healing.steps();
            auto heal = [&](std::size_t position, std::size_t thread, std::size_t) noexcept {
                refuse(healing.run(steps[position], traversals[thread]));
            };
            if (status_t const fanned = executor.for_tasks(steps.size(), heal); failed(fanned)) return fanned;
            if (failed(refusal)) return refusal;
        } while (healing.advance());
        auto const [healed, healed_status] = healing.finish();
        if (failed(healed_status)) return healed_status;
        // The heal walked the same traversals, so their counters already hold its telemetry.
        absorb_traversals_(traversals);
        result_.compaction.graph.reselected_nodes += count + healed.reselected_nodes;
        result_.compaction.graph.changed_nodes += healed.changed_nodes;
        result_.compaction.graph.islands_found += healed.islands_found;
        result_.compaction.graph.islands_reconnected += healed.islands_reconnected;
        typename ram_index_t::keyed_t const& keys = index.keys();
        return write_ram_rows_(
            index, target, first, count, staging, count, executor, [&](std::size_t row) noexcept { return nodes[row]; },
            [&](node_id_t node) noexcept { return static_cast<node_id_t>(first + keys[node]); });
    }

    /** Refines target part @p part whole in a RAM index inside the workspace: its value and
     *  neighbors columns load through a read-only mapping, the RAM refine re-selects and heals
     *  every row, and the rows write back. */
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
        auto [call, call_status] = nvme_begin_call(executor, std::span<medium_t const> {}, executor.queue_depth(),
                                                   index.traversal_memory_usage(sizing), 0, 0);
        if (failed(call_status)) return call_status;
        traversals_t traversals {scratch_allocator_t {executor.arena(0)}};
        if (status_t const ready = traversals_(index, executor, traversals); failed(ready)) return ready;
        status_t refusal = status_t::success_k;
        auto refuse = [&](status_t status) noexcept { keep_first_failure<atomic_ref>(refusal, status); };
        {
            auto [load, load_status] = index.plan_load(nodes);
            if (failed(load_status)) return load_status;
            auto const steps = load.steps();
            iota_view<node_id_t> const source_range {0, static_cast<node_id_t>(nodes)};
            if (status_t const fanned = executor.for_tasks(
                    steps.size(),
                    [&](std::size_t position, std::size_t, std::size_t) noexcept {
                        refuse(load.load(steps[position], columns, source_range,
                                         [](node_id_t row) noexcept { return row; }));
                    });
                failed(fanned))
                return fanned;
            if (failed(refusal)) return refusal;
            if (status_t const loaded = load.finish(); failed(loaded)) return loaded;
        }
        typename ram_index_t::refine_config_t const config {static_cast<u32_t>(expansion_factor_())};
        auto [refining, refine_status] = index.plan_refine({0, static_cast<node_id_t>(nodes)}, {}, {}, config);
        if (failed(refine_status)) return refine_status;
        do {
            auto const steps = refining.steps();
            if (status_t const fanned = executor.for_tasks(
                    steps.size(),
                    [&](std::size_t position, std::size_t thread, std::size_t) noexcept {
                        refuse(refining.run(steps[position], traversals[thread]));
                    });
                failed(fanned))
                return fanned;
            if (failed(refusal)) return refusal;
        } while (refining.advance());
        auto const [refined, finish_status] = refining.finish();
        if (failed(finish_status)) return finish_status;
        result_.compaction.graph.reselected_nodes += refined.reselected_nodes;
        result_.compaction.graph.changed_nodes += refined.changed_nodes;
        result_.compaction.graph.islands_found += refined.islands_found;
        result_.compaction.graph.islands_reconnected += refined.islands_reconnected;
        result_.compaction.graph.traversal_stats += refined.traversal_stats;
        return write_ram_rows_(
            index, target, 0, nodes, slices.staging, staged, executor,
            [](std::size_t row) noexcept { return static_cast<node_id_t>(row); },
            [](node_id_t node) noexcept { return node; });
    }

#pragma endregion RAM Blocks

#pragma region Reselection and Back Links

    /** Reads the value rows of the candidates at @p points, each a position into @p candidates, and
     *  scores each against its own row of @p queries. */
    status_t score_points_(target_t const& target, candidates_t const& candidates, byte_t const* queries,
                           std::span<u32_t const> points, executor_t& executor) noexcept {
        std::size_t const row_bytes = candidates.row_bytes;
        if (status_t const read = read_points_(
                target.view, part_of_(target).values_begin, row_bytes, points.size(),
                [&](std::size_t task) noexcept { return candidates.entries[points[task]].node_id; },
                [&](std::size_t task) noexcept { return candidates.vectors + std::size_t(points[task]) * row_bytes; },
                executor, result_.value_reads);
            failed(read))
            return read;
        return fan_out_(executor, points.size(), [&](std::size_t point, std::size_t, std::size_t) noexcept {
            std::size_t const position = points[point];
            byte_t const* const query = queries + position / candidates.slots * row_bytes;
            byte_t const* const candidate = candidates.vectors + position * row_bytes;
            candidates.entries[position].distance = distance_(query, norms_of_(query), candidate, norms_of_(candidate));
        });
    }

    /** Re-selects the tile of target rows @p step.nodes: each walks the target part from random
     *  landmarks, the appended nodes of a compaction or a build walking across into the resident
     *  rows alone first, its candidates keeping the best rows it scored and its current neighbors,
     *  then a diverse pick, and the tile is written. A whole part the workspace holds is refined
     *  in RAM instead. */
    status_t reselect_(step_t const& step, executor_t& executor) noexcept {
        target_t& target = targets_[step.part];
        std::size_t const nodes = target.view.size(), count = step.nodes.size(), first = step.nodes.first;
        if (count == nodes && refines_in_ram_(nodes)) return refine_in_ram_(step.part, executor);
        std::size_t const slots = candidate_slots_(), connectivity = connectivity_();
        nvme_part_t const& part = part_of_(target);
        workspace_arena_t arena = arena_over_(workspace_);
        carver_t carver {&arena};
        auto const [queries, neighbors, previous_neighbors, counts, entries, vectors, orders, kept_slots, kept_norms,
                    carried, cursors] = reselect_slices_(count, carver);
        if (carver.bytes > workspace_.size()) return status_t::bad_alloc_k;
        std::array<staged_column_t, 2> const columns {
            {{part.values_begin, value_row_bytes_(), queries},
             {part.neighbors_begin, neighbors_row_bytes_(), reinterpret_cast<byte_t*>(neighbors)}}};
        if (status_t const read = scan_rows_(target.view, 0, first, count, columns, executor); failed(read))
            return read;

        candidates_t const candidates {entries, vectors, counts, slots, value_row_bytes_()};
        std::memset(counts, 0, count * sizeof(u32_t));
        using query_t = reselect_query_t;
        typename query_t::inputs_t const inputs {
            vectors_view_t {reinterpret_cast<scalar_t const*>(queries), count, dimensions_()}, candidates,
            static_cast<node_id_t>(first), expansion_factor_()};
        search_config_t config {};
        config.expansion_factor = static_cast<typename indexes_t::index_t::candidates_count_t>(expansion_factor_());
        // Appended rows first walk across into the resident rows alone, since no resident row links
        // to them yet: the part read as ending at the resident rows draws and expands nothing past
        // them, and both walks fill the same candidates.
        if (target.resident && first >= target.resident) {
            u32_t& part_nodes = target.view.parts_[0].nodes;
            u32_t const all_nodes = std::exchange(part_nodes, static_cast<u32_t>(target.resident));
            auto const [crossed, cross_status] = target.view.template walk_queries_<query_t>(count, config, inputs, 0,
                                                                                             {}, {}, executor);
            part_nodes = all_nodes;
            if (failed(cross_status)) return cross_status;
            absorb_walks_(crossed);
        }
        auto const [walked, walk_status] = target.view.template walk_queries_<query_t>(count, config, inputs, 0, {}, {},
                                                                                       executor);
        if (failed(walk_status)) return walk_status;
        absorb_walks_(walked);

        // The current neighbors compete as the RAM refine's seeds do, their rows read into the
        // candidates' spare slots.
        std::size_t current = 0;
        for (std::size_t row = 0; row != count; ++row)
            for (std::size_t slot = 0; slot != connectivity; ++slot) {
                node_id_t const neighbor = neighbors[row * connectivity + slot];
                if (neighbor >= nodes || neighbor == first + row) continue;
                std::size_t const taken = candidates.offer(row, {neighbor, 0}, slots);
                if (taken != slots) orders[current++] = static_cast<u32_t>(row * slots + taken);
            }
        if (status_t const scored = score_points_(target, candidates, queries, {orders, current}, executor);
            failed(scored))
            return scored;

        std::size_t computed = current, changed = 0;
        status_t const selected = fan_out_(executor, count, [&](std::size_t row, std::size_t, std::size_t) noexcept {
            std::span<node_id_t> const out {neighbors + row * connectivity, connectivity};
            std::span<node_id_t> const before {previous_neighbors + row * connectivity, connectivity};
            std::memcpy(before.data(), out.data(), neighbors_row_bytes_());
            std::size_t local_computed = 0;
            select_diverse_(candidates, row, selection_of_(orders, kept_slots, kept_norms, row, slots), out,
                            local_computed);
            atomic_ref<std::size_t>(computed).fetch_add(local_computed, std::memory_order_relaxed);
            if (gained_(out, before) && !count_changed_(target, first + row))
                atomic_ref<std::size_t>(changed).fetch_add(1, std::memory_order_relaxed);
        });
        if (failed(selected)) return selected;
        result_.compaction.graph.traversal_stats.computed_distances += computed;
        result_.compaction.graph.traversal_stats.computed_distances_in_refines += computed;
        result_.compaction.graph.reselected_nodes += count;
        result_.compaction.graph.changed_nodes += changed;
        return write_rows_(target, part.neighbors_begin, neighbors_row_bytes_(), first, count,
                           reinterpret_cast<byte_t const*>(neighbors));
    }

    /** Opens the sweep's tile of target rows from @p first in target part @p part, up to
     *  @c sweep_rows_ of them, none linked yet, or from row zero of the next part walked; past the
     *  last one, enters the heal. */
    void open_sweep_(std::size_t part, std::size_t first) noexcept {
        for (; part != targets_.size() && !walked_(part); ++part) first = 0;
        if (part == targets_.size()) {
            phase_ = nvme_compact_phase_t::heal_k;
            return;
        }
        std::size_t const tile_rows = sweep_rows_();
        workspace_arena_t arena = arena_over_(workspace_);
        carver_t carver {&arena};
        sweep_slices_t const slices = sweep_slices_(tile_rows, carver);
        sweep_part_ = part;
        sweep_ = {first, min_of(first + tile_rows, targets_[part].view.size())};
        incoming_offsets_ = {slices.incoming_offsets, sweep_.size()};
        back_links_ = {slices.back_links, tile_rows * connectivity_()};
        std::memset(slices.incoming_offsets, 0, incoming_offsets_.size_bytes());
        ++result_.compaction.graph.swept_tiles;
        phase_ = nvme_compact_phase_t::count_back_links_k;
    }

    /** Turns the sweep tile's counts into each row's first link's offset, which the scatter
     *  advances to its end: the tile ends before the first row whose links pass the tape, and a
     *  lone row past it keeps the tape's worth of them. */
    void offset_back_links_() noexcept {
        std::size_t offset = 0, rows = 0;
        for (; rows != sweep_.size(); ++rows) {
            std::size_t count = incoming_offsets_[rows];
            if (offset + count > back_links_.size()) {
                if (rows) break;
                count = back_links_.size();
            }
            incoming_offsets_[rows] = offset;
            offset += count;
        }
        sweep_.bound = sweep_.first + max_of<std::size_t>(rows, 1);
        result_.compaction.graph.swept_links += offset;
    }

    /** Scans the walked rows of @p step and calls @p per_link with every link into the sweep's
     *  tile: the target's row in the tile and the linking row. A failed scan calls it for none, so
     *  the step may run again, and no row is written between the count and the scatter, so both
     *  read the same links. */
    template <typename per_link_type_>
    status_t scan_back_links_(step_t const& step, executor_t& executor, per_link_type_&& per_link) noexcept {
        target_t& target = targets_[step.part];
        std::size_t const count = step.nodes.size(), first = step.nodes.first, connectivity = connectivity_();
        bytes_span_t const spare = sweep_spare_();
        workspace_arena_t arena = arena_over_(spare);
        carver_t carver {&arena};
        node_id_t* const links = carve_<node_id_t>(carver, count * connectivity);
        if (carver.bytes > spare.size()) return status_t::bad_alloc_k;
        std::array<staged_column_t, 1> const columns {
            {{part_of_(target).neighbors_begin, neighbors_row_bytes_(), reinterpret_cast<byte_t*>(links)}}};
        if (status_t const read = scan_rows_(target.view, 0, first, count, columns, executor); failed(read))
            return read;
        return fan_out_(executor, count, [&](std::size_t row, std::size_t, std::size_t) noexcept {
            node_id_t const source = static_cast<node_id_t>(first + row);
            for (std::size_t slot = 0; slot != connectivity; ++slot) {
                std::size_t const linked = links[row * connectivity + slot];
                if (linked >= sweep_.first && linked < sweep_.bound && linked != source)
                    per_link(linked - sweep_.first, source);
            }
        });
    }

    /** Counts every link of the walked rows of @p step into the sweep's tile. */
    status_t count_back_links_(step_t const& step, executor_t& executor) noexcept {
        return scan_back_links_(step, executor, [&](std::size_t row, node_id_t) noexcept {
            atomic_ref<std::size_t>(incoming_offsets_[row]).fetch_add(1, std::memory_order_relaxed);
        });
    }

    /** Scatters every link of the walked rows of @p step into the sweep's tile onto its target's
     *  run; a link the tape has no room for is left out. */
    status_t scatter_back_links_(step_t const& step, executor_t& executor) noexcept {
        return scan_back_links_(step, executor, [&](std::size_t row, node_id_t source) noexcept {
            std::size_t const position =
                atomic_ref<std::size_t>(incoming_offsets_[row]).fetch_add(1, std::memory_order_relaxed);
            if (position < back_links_.size()) back_links_[position] = source;
        });
    }

    /** Merges the rows @p step.nodes of the sweep's tile with the rows linking to them. A row
     *  holding all of those stays as it is; any other reads its members' value rows and its run's,
     *  and they compete over rounds within its candidates, each round's picks carrying into the
     *  next, so no link on the tape is dropped however many a hub draws and the row is never
     *  thinner. The batch goes back in one write. */
    status_t merge_back_links_(step_t const& step, executor_t& executor) noexcept {
        target_t& target = targets_[step.part];
        nvme_part_t const& part = part_of_(target);
        std::size_t const nodes = target.view.size(), count = step.nodes.size(), first = step.nodes.first;
        std::size_t const slots = candidate_slots_(), connectivity = connectivity_(), row_bytes = value_row_bytes_();
        bytes_span_t const spare = sweep_spare_();
        workspace_arena_t arena = arena_over_(spare);
        carver_t carver {&arena};
        auto const [queries, neighbors, previous_neighbors, counts, entries, vectors, orders, kept_slots, kept_norms,
                    carried, cursors] = reselect_slices_(count, carver);
        if (carver.bytes > spare.size()) return status_t::bad_alloc_k;
        std::array<staged_column_t, 2> const columns {
            {{part.values_begin, row_bytes, queries},
             {part.neighbors_begin, neighbors_row_bytes_(), reinterpret_cast<byte_t*>(neighbors)}}};
        if (status_t const read = scan_rows_(target.view, 0, first, count, columns, executor); failed(read))
            return read;
        std::memcpy(previous_neighbors, neighbors, count * neighbors_row_bytes_());
        candidates_t const candidates {entries, vectors, counts, slots, row_bytes};

        auto run_of = [&](std::size_t row) noexcept -> std::span<node_id_t> {
            std::size_t const tile_row = first - sweep_.first + row, tape = back_links_.size();
            std::size_t const begin = tile_row ? min_of(incoming_offsets_[tile_row - 1], tape) : 0;
            return back_links_.subspan(begin, min_of(incoming_offsets_[tile_row], tape) - begin);
        };
        auto members_of = [&](std::size_t row) noexcept -> std::span<node_id_t const> {
            return {previous_neighbors + row * connectivity, connectivity};
        };
        // Row `row`'s next links take the slots its last selection left; its members are no news.
        auto take_links = [&](std::size_t row) noexcept {
            std::span<node_id_t const> const run = run_of(row);
            std::size_t cursor = cursors[row];
            for (; cursor != run.size() && counts[row] != slots; ++cursor)
                if (!holds_(members_of(row), run[cursor])) entries[row * slots + counts[row]++] = {run[cursor], 0};
            cursors[row] = static_cast<u32_t>(cursor);
        };
        status_t const opened = fan_out_(executor, count, [&](std::size_t row, std::size_t, std::size_t) noexcept {
            std::span<node_id_t> const run = run_of(row);
            // Filled in arrival order by the scatter; sorted, a hub's rounds are reproducible.
            partial_sort(run, run.size());
            std::size_t cursor = 0;
            while (cursor != run.size() && holds_(members_of(row), run[cursor])) ++cursor;
            carried[row] = counts[row] = 0;
            cursors[row] = static_cast<u32_t>(cursor);
            if (cursor == run.size()) return;
            for (node_id_t const member : members_of(row))
                if (member < nodes && member != first + row) entries[row * slots + counts[row]++] = {member, 0};
            take_links(row);
        });
        if (failed(opened)) return opened;

        std::size_t reads = 0;
        auto queue_reads = [&]() noexcept {
            reads = 0;
            for (std::size_t row = 0; row != count; ++row)
                for (std::size_t slot = carried[row]; slot != counts[row]; ++slot)
                    orders[reads++] = static_cast<u32_t>(row * slots + slot);
            return reads != 0;
        };
        if (!queue_reads()) return status_t::success_k;
        std::size_t computed = 0, changed = 0;
        do {
            if (status_t const scored = score_points_(target, candidates, queries, {orders, reads}, executor);
                failed(scored))
                return scored;
            computed += reads;
            status_t const merged = fan_out_(executor, count, [&](std::size_t row, std::size_t, std::size_t) noexcept {
                if (carried[row] == counts[row]) return;
                selection_t const scratch = selection_of_(orders, kept_slots, kept_norms, row, slots);
                std::size_t local_computed = 0;
                std::size_t const picks = select_diverse_(
                    candidates, row, scratch, {neighbors + row * connectivity, connectivity}, local_computed);
                atomic_ref<std::size_t>(computed).fetch_add(local_computed, std::memory_order_relaxed);
                // In ascending order, so no pick lands on one not yet moved.
                partial_sort(std::span<u32_t>(scratch.kept, picks), picks);
                for (std::size_t pick = 0; pick != picks; ++pick) {
                    std::size_t const slot = scratch.kept[pick];
                    if (slot == pick) continue;
                    entries[row * slots + pick] = entries[row * slots + slot];
                    std::memcpy(candidates.vector_of(row, pick), candidates.vector_of(row, slot), row_bytes);
                }
                carried[row] = counts[row] = static_cast<u32_t>(picks);
                take_links(row);
            });
            if (failed(merged)) return merged;
        } while (queue_reads());
        // A row the pass never walked stays out of the report, whatever it gained.
        iota_view<std::size_t> const walked = target.selected;
        status_t const compared = fan_out_(executor, count, [&](std::size_t row, std::size_t, std::size_t) noexcept {
            std::size_t const node = first + row;
            if (node < walked.first || node >= walked.bound) return;
            std::span<node_id_t const> const out {neighbors + row * connectivity, connectivity};
            if (gained_(out, members_of(row)) && !count_changed_(target, node))
                atomic_ref<std::size_t>(changed).fetch_add(1, std::memory_order_relaxed);
        });
        if (failed(compared)) return compared;
        result_.compaction.graph.traversal_stats.computed_distances += computed;
        result_.compaction.graph.traversal_stats.computed_distances_in_reverse_refines += computed;
        result_.compaction.graph.changed_nodes += changed;
        return write_rows_(target, part.neighbors_begin, neighbors_row_bytes_(), first, count,
                           reinterpret_cast<byte_t const*>(neighbors));
    }

#pragma endregion Reselection and Back Links

#pragma region Heal

    /** Heals target part @p step.part: one sequential pass of its neighbors column counts every
     *  row's in-links, then each row nothing links to is read and arrives, in target order, in the
     *  row of its first neighbor something links to, or of the landmark covering it. A cluster
     *  linking only to itself stays, for @c health to report. */
    status_t heal_(step_t const& step, executor_t& executor) noexcept {
        target_t& target = targets_[step.part];
        nvme_part_t const& part = part_of_(target);
        std::size_t const nodes = target.view.size(), connectivity = connectivity_(),
                          row_bytes = neighbors_row_bytes_();
        std::size_t const run_rows = target.view.scan_run_rows_(0, row_bytes);
        std::size_t const seeds = min_of<std::size_t>(connectivity, nodes), spread = heal_spread(connectivity);
        std::size_t const words = divide_round_up<std::size_t>(nodes, 32);
        // The island links take whatever the counts and the target row leave.
        carver_t fixed;
        heal_slices_(nodes, 0, fixed);
        std::size_t const capacity = (workspace_.size() - min_of(fixed.bytes, workspace_.size())) / sizeof(u64_t);
        workspace_arena_t arena = arena_over_(workspace_);
        carver_t carver {&arena};
        auto const [in_links, target_row, island_links] = heal_slices_(nodes, capacity, carver);
        if (!capacity || carver.bytes > workspace_.size()) return status_t::bad_alloc_k;
        std::memset(in_links, 0, words * sizeof(u64_t));
        auto links_of = [&](std::size_t node) noexcept {
            return static_cast<unsigned>(in_links[node / 32] >> (node % 32 * 2)) & 3u;
        };
        auto [call, call_status] = target.view.begin_scans_(executor, 1);
        if (failed(call_status)) return call_status;

        auto whole_run = [&](std::size_t task, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
            if (progress++) return {};
            std::size_t const first = task * run_rows;
            return {nvme_extent_t {part.neighbors_begin + first * row_bytes,
                                   static_cast<u32_t>(min_of(run_rows, nodes - first) * row_bytes), 0}};
        };
        auto count_links = [&](std::size_t, std::size_t task, u64_t, auto const& extents,
                               auto const& fetched) noexcept {
            auto const* const neighbors = reinterpret_cast<node_id_t const*>(fetched[0]);
            for (std::size_t slot = 0; slot != extents[0].length / sizeof(node_id_t); ++slot) {
                std::size_t const neighbor = neighbors[slot];
                if (neighbor >= nodes || neighbor == task * run_rows + slot / connectivity) continue;
                atomic_ref<u64_t> const word {in_links[neighbor / 32]};
                u64_t const once = u64_t {1} << (neighbor % 32 * 2);
                if (word.fetch_or(once, std::memory_order_relaxed) & once)
                    word.set_bits(once << 1, std::memory_order_relaxed);
            }
        };
        auto const [scanned, scan_status] = target.view.scan_reads_(executor, divide_round_up(nodes, run_rows),
                                                                    whole_run, count_links);
        result_.scan_reads += scanned;
        if (failed(scan_status)) return scan_status;

        for (std::size_t from = 0; from != nodes;) {
            // The rows from `from` on whose islands the island links hold.
            std::size_t bound = from, islands = 0;
            for (; bound != nodes && islands != capacity; ++bound) islands += !links_of(bound);
            result_.compaction.graph.islands_found += islands;
            alignas(default_alignment_k) std::size_t taken = 0;
            auto island_rows = [&](std::size_t task, u64_t& progress) noexcept -> std::array<nvme_extent_t, 1> {
                std::size_t const task_first = from + task * run_rows;
                std::size_t row = task_first + progress, end = min_of(task_first + run_rows, bound);
                while (row != end && links_of(row)) ++row;
                if (row == end) return {};
                progress = row + 1 - task_first;
                return {nvme_extent_t {part.neighbors_begin + row * row_bytes, static_cast<u32_t>(row_bytes), 0}};
            };
            auto pick_targets = [&](std::size_t, std::size_t, u64_t, auto const& extents,
                                    auto const& fetched) noexcept {
                std::size_t const island = static_cast<std::size_t>((extents[0].begin - part.neighbors_begin) /
                                                                    row_bytes);
                auto const* const neighbors = reinterpret_cast<node_id_t const*>(fetched[0]);
                std::size_t linked = island * seeds / nodes * nodes / seeds;
                for (std::size_t slot = 0; slot != connectivity; ++slot)
                    if (neighbors[slot] < nodes && neighbors[slot] != island && links_of(neighbors[slot])) {
                        linked = neighbors[slot];
                        break;
                    }
                if (linked == island) return;
                island_links[atomic_ref<std::size_t>(taken).fetch_add(1, std::memory_order_relaxed)] =
                    static_cast<u64_t>(linked) << 32 | island;
            };
            auto const [read, pick_status] = target.view.scan_reads_(executor, divide_round_up(bound - from, run_rows),
                                                                     island_rows, pick_targets);
            result_.neighbor_reads += read;
            if (failed(pick_status)) return pick_status;

            // In target order, so every target row is read and written once, near-sequentially.
            partial_sort(std::span<u64_t> {island_links, taken}, taken);
            for (std::size_t position = 0; position != taken;) {
                std::size_t const linked = static_cast<std::size_t>(island_links[position] >> 32);
                result_.neighbor_reads += nvme_reads_t {1, row_bytes};
                if (status_t const fetched = target.file.read_at(part.neighbors_begin + linked * row_bytes,
                                                                 {reinterpret_cast<byte_t*>(target_row), row_bytes});
                    failed(fetched))
                    return fetched;
                for (; position != taken && island_links[position] >> 32 == linked; ++position) {
                    std::size_t const island = static_cast<u32_t>(island_links[position]);
                    std::size_t target_slot = 0;
                    while (target_slot != connectivity && target_row[target_slot] < nodes) ++target_slot;
                    // A full row gives up a tail slot spread by the island's ID whose resident
                    // keeps another in-link; an island linked before counts none, so its slot
                    // stays.
                    for (std::size_t attempt = 0; target_slot == connectivity && attempt != spread; ++attempt) {
                        std::size_t const slot = connectivity - 1 - (island + attempt) % spread;
                        std::size_t const resident = target_row[slot];
                        if (links_of(resident) != 3u) continue;
                        in_links[resident / 32] &= ~(u64_t {2} << (resident % 32 * 2));
                        target_slot = slot;
                    }
                    if (target_slot == connectivity) continue;
                    target_row[target_slot] = static_cast<node_id_t>(island);
                    ++result_.compaction.graph.islands_reconnected;
                }
                if (status_t const wrote = write_rows_(target, part.neighbors_begin, row_bytes, linked, 1,
                                                       reinterpret_cast<byte_t const*>(target_row));
                    failed(wrote))
                    return wrote;
            }
            from = bound;
        }
        return status_t::success_k;
    }

#pragma endregion Heal

    /** The host allocator behind the plan's catalogs, which never grow with the rows. */
    forkunion_allocator_t allocator_ {};

    /** The memory every step carves its blocks, batches and tables from. */
    bytes_span_t workspace_;

    /** The caller's configurations, the provenance borrowed until the plan finishes. */
    compact_config_t config_;
    save_config_t save_;

    /** The custom fields every target header carries. */
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
     *  counts as changed, at the front of the workspace the caller lent. */
    std::span<u64_t> changed_;

    /** The target part the sweep is in, and its open tile of target rows. */
    std::size_t sweep_part_ = 0;
    iota_view<std::size_t> sweep_;

    /** The open tile's counts, then offsets, one per row of it, and its tape, both at the
     *  workspace's front. */
    std::span<std::size_t> incoming_offsets_;
    std::span<node_id_t> back_links_;

    nvme_compact_phase_t phase_ = nvme_compact_phase_t::finish_k;

    result_t result_;
};

template <typename indexes_type_>
expected<nvme_compact_plan<indexes_type_>> plan_compact(std::span<indexes_type_ const* const> sources,
                                                        std::string_view target, bytes_span_t workspace,
                                                        compact_config_t const& config,
                                                        save_config_t const& save) noexcept {
    using plan_t = nvme_compact_plan<indexes_type_>;
    plan_t plan;
    if (sources.empty() || !sources[0]) return {std::move(plan), status_t::uninitialized_k};
    if (status_t const opened = plan.open_(target, workspace, config, save, *sources[0]); failed(opened))
        return {std::move(plan), opened};
    for (indexes_type_ const* source : sources)
        if (status_t const added = source ? plan.add_sources_(*source, *sources[0]) : status_t::invalid_argument_k;
            failed(added))
            return {std::move(plan), added};
    if (plan.sources_.empty()) return {std::move(plan), status_t::uninitialized_k};
    for (typename plan_t::source_t& part : plan.sources_) part.kept = {0, part.rows};
    if (status_t const sized = plan.size_steps_(); failed(sized)) return {std::move(plan), sized};
    growing_buffer<std::size_t, forkunion_allocator_t> target_rows {forkunion_allocator_t {}};
    if (status_t const packed = plan.pack_sources_(config.shard_capacity, target_rows); failed(packed))
        return {std::move(plan), packed};
    if (status_t const created = plan.open_targets_(to_span(target_rows)); failed(created))
        return {std::move(plan), created};

    // The part arriving first in a target part stays resident and keeps its IDs; every later part
    // arrives after it with its own edges rebased, so an appended row's walk navigates its own part
    // and its old neighbors compete with what the walk finds. A cut range lost the edges past it,
    // so every row of its target part re-selects.
    for (typename plan_t::source_t const& part : plan.sources_) {
        if (part.base) continue;
        bool const cut = part.stride < part.rows;
        for (std::size_t first = 0, target_index = part.target; first < part.rows;
             first += part.stride, ++target_index) {
            typename plan_t::target_t& bin = plan.targets_[target_index];
            bin.resident = min_of(part.stride, part.rows - first);
            bin.selected = {cut ? 0 : bin.resident, bin.view.size()};
        }
    }
    plan.phase_ = nvme_compact_phase_t::copy_k;
    return {std::move(plan), status_t::success_k};
}

template <typename indexes_type_>
expected<nvme_compact_plan<indexes_type_>> plan_refine(indexes_type_ const& source, iota_view<std::size_t> rows,
                                                       std::string_view target, bytes_span_t workspace,
                                                       compact_config_t const& config,
                                                       save_config_t const& save) noexcept {
    using plan_t = nvme_compact_plan<indexes_type_>;
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
    if (status_t const sized = plan.size_steps_(); failed(sized)) return {std::move(plan), sized};
    if (status_t const created = plan.open_targets_(to_span(part_rows)); failed(created))
        return {std::move(plan), created};
    // The range runs in placement order across the parts, so each part re-selects its own slice of
    // it.
    for (std::size_t part = 0, placed = 0; part != plan.targets_.size(); placed += plan.targets_[part++].view.size()) {
        typename plan_t::target_t& one = plan.targets_[part];
        std::size_t const low = max_of(rows.first, placed), high = min_of(rows.bound, placed + one.view.size());
        one.selected = low < high ? iota_view<std::size_t> {low - placed, high - placed} : iota_view<std::size_t> {};
    }
    plan.phase_ = nvme_compact_phase_t::copy_k;
    return {std::move(plan), status_t::success_k};
}

template <typename indexes_type_>
expected<nvme_compact_plan<indexes_type_>> plan_build(std::span<std::string_view const> sources,
                                                      std::string_view target, bytes_span_t workspace,
                                                      compact_config_t const& config,
                                                      save_config_t const& save) noexcept {
    using plan_t = nvme_compact_plan<indexes_type_>;
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
    if (status_t const sized = plan.size_steps_(); failed(sized)) return {std::move(plan), sized};
    growing_buffer<std::size_t, forkunion_allocator_t> target_rows {forkunion_allocator_t {}};
    if (status_t const packed = plan.pack_sources_(config.shard_capacity, target_rows); failed(packed))
        return {std::move(plan), packed};
    if (status_t const created = plan.open_targets_(to_span(target_rows)); failed(created))
        return {std::move(plan), created};
    // The first block of every target part stays resident, and every later block links in by walks
    // across as a compaction's appended part does.
    for (typename plan_t::target_t& one : plan.targets_) {
        one.resident = min_of(block_rows, one.view.size());
        one.selected = {one.resident, one.view.size()};
    }
    plan.phase_ = nvme_compact_phase_t::copy_k;
    return {std::move(plan), status_t::success_k};
}

/** Drains @p planned over @p executor: every step of every phase in order, then the commit. */
template <typename plan_type_>
expected<nvme_compact_result_t> nvme_compact_drain_(expected<plan_type_>&& planned,
                                                    typename plan_type_::executor_t& executor) noexcept {
    if (!planned) return {{}, planned.status};
    plan_type_& plan = planned.value;
    do
        for (typename plan_type_::step_t const step : plan.steps())
            if (status_t const ran = plan.run(step, executor); failed(ran)) return {{}, ran};
    while (plan.advance());
    return plan.finish();
}

/**
 *  The one-shot @c plan_compact, drained over @p executor.
 *
 *  @par Statuses:
 *  - @c uninitialized_k No source, a @c NULL first source, a first source viewing nothing, sources
 *    holding no rows, or @p executor holds no pool or workers.
 *  - @c config_mismatch_k @p save names an alignment below the floor or not a power of two or a
 *    reserved provenance field, a source differs in kernel, geometry or key width from the first,
 *    a part rebuilt in RAM disagrees with its rows, a medium the engines cannot read, or a part
 *    header does not fit before its data.
 *  - @c invalid_argument_k An empty @p target, a @c NULL later source, or a target part or
 *    manifest path that does not parse.
 *  - @c bad_alloc_k @p workspace cannot hold one row of some phase.
 *  - @c overflow_k A target part of zero rows or past the node ID space, a part rebuilt in RAM
 *    passing a bound, a staging region's byte count, or a write past a device's end.
 *  - @c malformed_k A framed target header has no width.
 *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
 *  - @c unknown_kind_k A part rebuilt in RAM cannot finalize its kernel.
 */
template <typename indexes_type_>
expected<nvme_compact_result_t> compact(std::span<indexes_type_ const* const> sources, std::string_view target,
                                        typename indexes_type_::executor_t& executor, bytes_span_t workspace,
                                        compact_config_t const& config = {}, save_config_t const& save = {}) noexcept {
    return nvme_compact_drain_(plan_compact(sources, target, workspace, config, save), executor);
}

/**
 *  The one-shot @c plan_refine, drained over @p executor.
 *
 *  @par Statuses:
 *  - @c uninitialized_k @p source views nothing or holds no rows, or @p executor holds no pool or
 *    workers.
 *  - @c config_mismatch_k @p save names an alignment below the floor or not a power of two or a
 *    reserved provenance field, a part of @p source differs in geometry or key width, a part
 *    rebuilt in RAM disagrees with its rows, a medium the engines cannot read, or a part header
 *    does not fit before its data.
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
expected<nvme_compact_result_t> refine(indexes_type_ const& source, iota_view<std::size_t> rows,
                                       std::string_view target, typename indexes_type_::executor_t& executor,
                                       bytes_span_t workspace, compact_config_t const& config = {},
                                       save_config_t const& save = {}) noexcept {
    return nvme_compact_drain_(plan_refine(source, rows, target, workspace, config, save), executor);
}

/**
 *  The one-shot @c plan_build, drained over @p executor.
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
 *    field, a part rebuilt in RAM disagrees with its rows, a medium the engines cannot read, or a
 *    part header does not fit before its data.
 *  - @c unsupported_k A transport ID in a build without SPDK.
 *  - @c unknown_kind_k A kernel the stored scalar's kits cannot finalize, or a part rebuilt in RAM
 *    cannot finalize its kernel.
 *  - @c overflow_k A target part of zero rows or past the node ID space, a part rebuilt in RAM
 *    passing a bound, a staging region's byte count, or a write past a device's end.
 *  - @c contended_k Another call holds @p executor, or its pool runs another dispatch.
 */
template <typename indexes_type_>
expected<nvme_compact_result_t> build(std::span<std::string_view const> sources, std::string_view target,
                                      typename indexes_type_::executor_t& executor, bytes_span_t workspace,
                                      compact_config_t const& config = {}, save_config_t const& save = {}) noexcept {
    return nvme_compact_drain_(plan_build<indexes_type_>(sources, target, workspace, config, save), executor);
}

} // namespace unum::usearch::dense
